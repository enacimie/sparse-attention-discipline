"""Variance analysis: run stochastic methods across many seeds and report mean/std.

Critical for stochastic methods (random_topk) to quantify stability. We measure
logit cosine vs dense and needle recall across N seeds.

Usage:
  python scripts/run_variance.py --model lfm2.5-350m-base --context 2048 --seeds 5
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from attention_interceptor import apply_method, restore, reset_pair_counter, PAIR_COUNTER  # noqa: E402
from attention_methods import AttentionMethod  # noqa: E402
from model_loader import MODELS, inspect_model, load_model  # noqa: E402
from utils import append_jsonl, fidelity_metrics, set_seed  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "variance.jsonl")


@torch.no_grad()
def get_logits(model, ids):
    return model(ids).logits[0].float().cpu()


def run(model_key, context, dtype, n_seeds):
    model_id = MODELS[model_key]
    info = inspect_model(model_id)
    model, tok = load_model(model_id, dtype=dtype)
    device = next(model.parameters()).device
    full_layers = info.full_attention_layers
    print(f"[load] {model_id} full_attn={full_layers}")
    ids = tok("The history of the region spans many centuries. " * (context // 12),
              return_tensors="pt").input_ids[:, :context].to(device)

    set_seed(0)
    ref = get_logits(model, ids)

    methods = {
        "random_topk_32_32": AttentionMethod(name="random_topk", topk=32, random_k=32),
        "local_global_topk_opt": AttentionMethod(name="local_global_topk", window=64, n_global=4, topk=32),
        "lsh": AttentionMethod(name="lsh"),
    }
    for name, base_method in methods.items():
        cosines = []
        for seed in range(n_seeds):
            m = AttentionMethod(**{**vars(base_method), "seed": seed})
            originals = apply_method(model, m, full_layers)
            set_seed(seed)
            approx = get_logits(model, ids)
            fid = fidelity_metrics(approx, ref)
            restore(model, originals, full_layers)
            cosines.append(fid["logit_cosine"])
        arr = np.array(cosines)
        rec = {"experiment": "variance", "method": name, "model": model_id,
               "context": context, "n_seeds": n_seeds,
               "cosine_mean": round(float(arr.mean()), 4),
               "cosine_std": round(float(arr.std()), 4),
               "cosine_min": round(float(arr.min()), 4),
               "cosine_max": round(float(arr.max()), 4)}
        append_jsonl(RESULTS, rec)
        print(f"  {name}: cosine {arr.mean():.4f} ± {arr.std():.4f} "
              f"[{arr.min():.4f}, {arr.max():.4f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS.keys()))
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    run(args.model, args.context, args.dtype, args.n_seeds)


if __name__ == "__main__":
    main()
