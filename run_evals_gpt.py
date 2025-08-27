import os
import json
import time
import random
from datetime import datetime
from functools import partial
from pprint import pprint

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm

from dotenv import load_dotenv
from transformer_lens import HookedTransformer
from lm_eval import simple_evaluate
from lm_eval.api.model import LM  # if you're on lm-eval 0.3.*; for 0.4.* use: from lm_eval.base import BaseLM

# ------------- Utilities -------------

def norm(x, eps=1e-12):
    return x.norm(dim=-1, keepdim=True).clamp_min(eps)

def estimate_resid_scale(hooked_model, tokenizer, prompts, hook_point, device):
    """Rough scale for the residual norm at the hook site on the last token."""
    with torch.no_grad():
        norms = []
        for p in prompts[:8]:
            messages = [{"role": "user", "content": p}]
            toks = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(device)
            _, cache = hooked_model.run_with_cache(toks)
            acts = cache[hook_point]  # [B, T, d_model]
            last = acts[:, -1, :]
            norms.append(last.norm(dim=-1))  # [B]
        return torch.cat(norms).mean()  # scalar tensor

def get_activations_at_layer(hooked_model, tokenizer, prompts, layer_idx, device, hook_name="hook_resid_post"):
    """Return last-token activations for each prompt at a specific layer."""
    hook_point = f"blocks.{layer_idx}.{hook_name}"
    outs = []
    with torch.no_grad():
        for prompt in prompts:
            msgs = [{"role": "user", "content": prompt}]
            toks = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(device)
            _, cache = hooked_model.run_with_cache(toks)
            act = cache[hook_point]          # [B, T, d_model]
            outs.append(act[:, -1, :])       # [B, d_model]
    return torch.cat(outs, dim=0)            # [N, d_model]

def compute_mean_pairwise_difference(expert_acts, non_expert_acts):
    assert expert_acts.shape[0] == non_expert_acts.shape[0], "Expert and non-expert counts must match."
    diffs = expert_acts - non_expert_acts   # [N, d_model]
    return diffs.mean(dim=0), diffs         # [d_model], [N, d_model]

# This is for simple_evaluate
def make_expertise_steer_hook(vector, coefficient):
    """Add expertise vector to ALL positions at the hook site (broadcast-safe)."""
    vec = (coefficient * vector).detach()  # [d_model]
    def _hook(activation, hook):
        v = vec.to(device=activation.device, dtype=activation.dtype)
        return activation + v.view(1, 1, -1)
    return _hook

# ------------- LM-eval wrapper -------------

class HookedModelWrapper(LM):  # if lm-eval==0.4.*, subclass BaseLM and same methods
    def __init__(self, wrapped_model, tokenizer, hook_point, steer_hook_fn, max_length=2048, batch_size=1):
        super().__init__()
        self.wrapped_model = wrapped_model    # HookedTransformer
        self.tokenizer = tokenizer
        self.hook_point = hook_point
        self.steer_hook_fn = steer_hook_fn
        self._max_length = max_length         # FIX: no trailing comma
        self._batch_size = batch_size

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self._batch_size

    def _score(self, input_ids):
        with self.wrapped_model.hooks(fwd_hooks=[(self.hook_point, self.steer_hook_fn)]):
            logits = self.wrapped_model(input_ids)  # [B, T, V]
        return logits

    def loglikelihood(self, requests):
        device = self.wrapped_model.cfg.device
        out = []
        for req in requests:
            context, continuation = req.args[0], req.args[1]
            ctx_ids  = self.tokenizer(context, return_tensors="pt").input_ids.to(device)
            full_ids = self.tokenizer(context + continuation, return_tensors="pt").input_ids.to(device)
            logits = self._score(full_ids)[0]  # [T, V]
            ctx_len = ctx_ids.shape[1]
            cont_len = full_ids.shape[1] - ctx_len
            if cont_len > 0:
                rel_logits = logits[ctx_len-1:ctx_len+cont_len-1]           # [cont_len, V]
                cont_ids   = full_ids[0, ctx_len:ctx_len+cont_len]          # [cont_len]
                log_probs  = rel_logits.log_softmax(-1).gather(1, cont_ids.unsqueeze(1)).squeeze(1)
                out.append((float(log_probs.sum().item()), True))
            else:
                out.append((0.0, True))
        return out

    def loglikelihood_rolling(self, requests):
        device = self.wrapped_model.cfg.device
        vals = []
        for req in requests:
            s = req.args[0]
            ids = self.tokenizer(s, return_tensors="pt").input_ids.to(device)
            logits = self._score(ids)[0]  # [T, V]
            lp = 0.0
            for t in range(1, ids.shape[1]):
                lp += torch.log_softmax(logits[t-1], dim=-1)[ids[0, t]].item()
            vals.append(lp)
        return vals

    def generate_until(self, requests):
        device = self.wrapped_model.cfg.device
        outs = []
        for req in requests:
            context = req.args[0]
            until   = req.args[1] if len(req.args) > 1 else []
            input_ids = self.tokenizer(context, return_tensors="pt").input_ids.to(device)
            with self.wrapped_model.hooks(fwd_hooks=[(self.hook_point, self.steer_hook_fn)]):
                gen = self.wrapped_model.generate(input_ids, max_new_tokens=256, do_sample=False)
            new_tokens = gen[0, input_ids.shape[1]:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            for stop in until:
                idx = text.find(stop)
                if idx != -1:
                    text = text[:idx]
                    break
            outs.append(text)
        return outs

# ------------- Main -------------

def main():
    t1 = time.time()
    load_dotenv()
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    hooked_model = HookedTransformer.from_pretrained_no_processing(model_name, device="cuda")
    tokenizer = hooked_model.tokenizer
    device = hooked_model.cfg.device

    # Load your paired prompts
    data = pd.read_csv("expert_nonexpert.csv")
    expert_train      = data["expert_prompt"].tolist()[:30]
    non_expert_train  = data["non_expert_prompt"].tolist()[:30]

    # Sanity print
    pprint(expert_train[:2]); pprint(non_expert_train[:2])

    # Choose layers & coeffs
    layers = [28, 15]
    #coeffs = [5, 2, 1, 0.5, 0, -0.5, -1, -2, -5]
    #coeffs = [-0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4]
    coeffs = [-0.5, 0.5]
    tasks  = ["truthfulqa"]
    #tasks  = ["mmlu_college_mathematics", "mmlu_philosophy"]

    # Compute expertise directions for one reference layer (example: 28)
    # (You already saved them; leaving this here in case you need to recompute.)
    extraction_layer = 28
    print(f"Extracting activations at layer {extraction_layer} …")
    non_expert_acts = get_activations_at_layer(hooked_model, tokenizer, non_expert_train, extraction_layer, device)
    expert_acts     = get_activations_at_layer(hooked_model, tokenizer, expert_train, extraction_layer, device)
    print(f"Non-expert activations shape: {non_expert_acts.shape}")
    print(f"Expert activations shape:     {expert_acts.shape}")

    expertise_direction, pairwise = compute_mean_pairwise_difference(expert_acts, non_expert_acts)
    diff_means = expert_acts.mean(0) - non_expert_acts.mean(0)
    print(f"Cosine(expertise_dir, diff_means) = {torch.cosine_similarity(expertise_direction, diff_means, dim=0).item():.4f}")

    torch.save({
        "expertise_direction": expertise_direction.cpu(),
        "difference_of_means": diff_means.cpu(),
        "layer": extraction_layer,
    }, f"expertise_activations_layer_{extraction_layer}.pt")

    # Eval sweep
    all_results = {}
    calib_prompts = [
        "Explain sorting algorithms",
        "What is machine learning?",
        "How does the internet work?",
        "Describe neural networks",
    ]
    t_layer = time.time()
    for layer in layers:
        hook_point = f"blocks.{layer}.hook_resid_post"

        # Estimate residual norm scale per layer (AFTER hook_point is defined)
        avg_resid_norm = estimate_resid_scale(hooked_model, tokenizer, calib_prompts, hook_point, device)

        # Load expertise direction for this layer
        saved = torch.load(f"expertise_activations_layer_{layer}.pt", map_location="cpu")
        exp_dir = saved["expertise_direction"].to(device)
        exp_dir = exp_dir / exp_dir.norm().clamp_min(1e-12)

        for coeff in coeffs:
            print(f"[Eval] layer={layer} coeff={coeff}")
            steer_hook_fn = make_expertise_steer_hook(exp_dir, float(coeff) * float(avg_resid_norm))
            batch_size = 1
            hh = HookedModelWrapper(hooked_model, tokenizer, hook_point, steer_hook_fn, batch_size=batch_size)
            t_before = time.time()
            results = simple_evaluate(
                model=hh,
                tasks=tasks,
                batch_size=batch_size,     # match wrapper
                limit=100         # sanity: remove for full runs
            )
            t_after = time.time()
            print(f"Time taken for this eval {t_after - t_before}")
            all_results[(layer, coeff)] = results

            # Incremental, atomic save
            tmp = "some_truth_results.pkl.tmp"
            with open(tmp, "wb") as f:
                import pickle
                pickle.dump(all_results, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush(); os.fsync(f.fileno())
            os.replace(tmp, "some_truth_results.pkl")
    t2 = time.time()
    print("Finished with evals and saved to disk")
    print(f"Time taken in evals: {t2 - t_layer} seconds")
    print(f"Total time taken: {t2 - t1} seconds")
# 11 mins for 1 eval on truthfulqa
if __name__ == "__main__":
    main()

