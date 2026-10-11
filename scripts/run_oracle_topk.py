"""Reviewer round-7 (W5/Q4): dense-posterior ORACLE top-k as an upper bound.

The manuscript selects free keys by mean q.k score. Reviewers ask (a) for an ORACLE
top-k upper bound based on the dense attention posterior, and (b) for value-aware
selection. This measures both at the working budget (window 64 + sinks 4 + top-k 32,
pair_ratio ~0.09 at ctx 2048):

  lg_oracle   = same structure, free keys = top-k by dense per-head softmax weights
                averaged over query heads (the dense attention's own posterior).
  lg_voracle  = same, but weights times value norm (value-aware selection).
  lg_topk     = the paper's q.k recipe (reference).
  lg_randtopk = structurally-matched random control (lower reference).

The oracle is a DIAGNOSTIC upper bound: its selection requires the full dense score
matrix, so it is not a practical method. The gap lg_oracle - lg_topk bounds how much
a better (but same-budget) selector could gain in fidelity.

Output: results/oracle_topk.jsonl
  {"experiment": "oracle_topk", "model", "context", "method", "seed",
   "cos_list": [...], "pair_ratio", "n_samples"}

Usage:
  python scripts/run_oracle_topk.py --model qwen3.5-0.8b --ctx 2048 --seeds 42 43 44 45 46
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from attention_interceptor import (apply_method, reset_pair_counter, restore,
                                   PAIR_COUNTER)  # noqa: E402
from attention_methods import AttentionMethod  # noqa: E402
from model_loader import MODELS, inspect_model, load_model  # noqa: E402
from utils import append_jsonl, fidelity_metrics, set_seed  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "oracle_topk.jsonl")

METHODS = {
    "lg_topk": AttentionMethod(name="local_global_topk", window=64, n_global=4, topk=32),
    "lg_oracle": AttentionMethod(name="local_global_oracle", window=64, n_global=4, topk=32),
    "lg_voracle": AttentionMethod(name="local_global_voracle", window=64, n_global=4, topk=32),
    "lg_randtopk": AttentionMethod(name="local_global_randtopk", window=64, n_global=4,
                                   topk=32, seed=42),
}


@torch.no_grad()
def get_logits(model, ids):
    return model(ids).logits[0].cpu().float()


def build_chunks(tok, ctx, n_samples, seed):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    text = "\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    stride = max(1, (ids.numel() - ctx) // max(1, n_samples + 1))
    chunks, s = [], seed * stride % max(1, ids.numel() - ctx)
    for _ in range(n_samples):
        if s + ctx > ids.numel():
            break
        chunks.append(ids[s:s + ctx])
        s += stride
    return chunks or [ids[:ctx]]


def measure_pair_ratio(model, ids, device, method, full_layers):
    orig = apply_method(model, method, full_layers)
    try:
        reset_pair_counter()
        with torch.no_grad():
            model(ids.unsqueeze(0).to(device))
        return PAIR_COUNTER["computed"] / max(1, PAIR_COUNTER["full"])
    finally:
        restore(model, orig, full_layers)


def run(model_key, ctx, seeds, dtype, n_samples, load_in_4bit, methods):
    model_id = MODELS[model_key]
    info = inspect_model(model_id)
    model, tok = load_model(model_id, dtype=dtype, load_in_4bit=load_in_4bit)
    device = next(model.parameters()).device
    full_layers = info.full_attention_layers
    print(f"[load] {model_id} full_attn={len(full_layers)} ctx={ctx} methods={methods}")

    chunks0 = build_chunks(tok, ctx, n_samples, seeds[0])
    pair_ratios = {}
    for name in methods:
        m = METHODS[name]
        if name == "lg_randtopk":
            m = AttentionMethod(name="local_global_randtopk", window=64, n_global=4,
                                topk=32, seed=seeds[0])
        pair_ratios[name] = measure_pair_ratio(model, chunks0[0], device, m, full_layers)
        print(f"  {name:12s} pair_ratio={pair_ratios[name]:.4f}")

    for name in methods:
        for seed in seeds:
            set_seed(seed)
            if name == "lg_randtopk":
                m = AttentionMethod(name="local_global_randtopk", window=64, n_global=4,
                                    topk=32, seed=seed)
            else:
                m = METHODS[name]
            chunks = build_chunks(tok, ctx, n_samples, seed)
            cos_list = []
            for ids in chunks:
                orig = apply_method(model, m, full_layers)
                try:
                    with torch.no_grad():
                        sparse = get_logits(model, ids.unsqueeze(0).to(device))
                finally:
                    restore(model, orig, full_layers)
                with torch.no_grad():
                    dense = get_logits(model, ids.unsqueeze(0).to(device))
                cos_list.append(float(fidelity_metrics(sparse, dense)["logit_cosine"]))
            print(f"  ctx={ctx} {name:12s} seed={seed} cos={[round(c, 4) for c in cos_list]}")
            append_jsonl(RESULTS, {"experiment": "oracle_topk", "model": model_id,
                                   "context": ctx, "method": name, "seed": seed,
                                   "cos_list": cos_list,
                                   "pair_ratio": round(pair_ratios[name], 4),
                                   "n_samples": len(cos_list)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-0.8b", choices=list(MODELS.keys()))
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--no-quantize", action="store_true")
    ap.add_argument("--methods", nargs="+",
                    default=["lg_topk", "lg_oracle", "lg_voracle", "lg_randtopk"],
                    choices=list(METHODS.keys()))
    args = ap.parse_args()
    run(args.model, args.ctx, args.seeds, args.dtype, args.n_samples,
        load_in_4bit=not args.no_quantize, methods=args.methods)


if __name__ == "__main__":
    main()
