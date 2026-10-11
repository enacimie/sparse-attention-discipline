"""Unified approximate-attention benchmark.

Runs a list of AttentionMethod configs against the dense baseline on the full
_attention layers, measuring fidelity (KL/MSE/cosine), needle recall, computed
pair ratio, and VRAM. This single script drives experiments 3,4,5,6,8,9.

Usage:
  python scripts/run_approx.py --model qwen3.5-0.8b --contexts 2048 \
      --methods local_window:window=256 local_global:window=256,n_global=4 \
                topk:topk=64 cluster:n_clusters=32,top_clusters=4 ann:ann_k=64
"""

import argparse
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from attention_interceptor import (apply_method, reset_pair_counter, restore,
                                   PAIR_COUNTER)  # noqa: E402
from attention_methods import AttentionMethod  # noqa: E402
from model_loader import MODELS, inspect_model, load_model  # noqa: E402
from needle import build_needle_prompt, check_needle, make_needle  # noqa: E402
from utils import GpuMeter, append_jsonl, fidelity_metrics, set_seed  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "approx.jsonl")


def parse_method(spec: str) -> AttentionMethod:
    name, _, kv = spec.partition(":")
    m = AttentionMethod(name=name)
    if kv:
        for pair in kv.split(","):
            k, v = pair.split("=")
            setattr(m, k, int(v) if v.lstrip("-").isdigit() else v)
    return m


@torch.no_grad()
def get_logits(model, ids):
    return model(ids).logits[0].float().cpu()


@torch.no_grad()
def needle_recall(model, tok, ctx, depths, seeds, gen_tokens=24):
    hits, total = 0, 0
    for depth in depths:
        for seed in seeds:
            set_seed(seed)
            rng = random.Random(seed)
            needle = make_needle(rng)
            prompt, needle = build_needle_prompt(tok, ctx, needle, depth)
            ids = tok(prompt, return_tensors="pt").input_ids.to(next(model.parameters()).device)
            out = model.generate(ids, max_new_tokens=gen_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            hits += int(check_needle(gen, needle)); total += 1
    return hits / total


def run(model_key, contexts, methods, dtype, seed, load_in_4bit):
    model_id = MODELS[model_key]
    info = inspect_model(model_id)
    model, tok = load_model(model_id, dtype=dtype, load_in_4bit=load_in_4bit)
    device = next(model.parameters()).device
    full_layers = info.full_attention_layers
    print(f"[load] {model_id} full_attn_layers={full_layers}")
    depths, seeds = [0.1, 0.5, 0.9], [42, 43, 44]

    for ctx in contexts:
        ids = tok("The history of the region spans many centuries. " * (ctx // 12),
                  return_tensors="pt").input_ids[:, :ctx].to(device)
        L = ids.shape[1]
        causal_pairs = L * (L + 1) // 2
        dense_pairs = causal_pairs * info.n_full_attention

        set_seed(seed)
        with GpuMeter() as dm:
            ref_logits = get_logits(model, ids)
        dense_needle = needle_recall(model, tok, ctx, depths, seeds)
        append_jsonl(RESULTS, {"experiment": "approx", "method": "dense", "model": model_id,
                               "context": ctx, "params": "{}", "computed_pairs": dense_pairs,
                               "pair_ratio": 1.0, "kl_div": 0.0, "logit_mse": 0.0,
                               "logit_cosine": 1.0, "needle_recall": dense_needle, **dm.to_dict()})
        print(f"  ctx={ctx} DENSE pairs={dense_pairs} needle={dense_needle:.2f}")

        for method in methods:
            params = {k: v for k, v in vars(method).items() if k != "name"}
            originals = apply_method(model, method, full_layers)
            set_seed(seed)
            with GpuMeter() as m:
                approx_logits = get_logits(model, ids)
                fid = fidelity_metrics(approx_logits, ref_logits)
            reset_pair_counter()
            with torch.no_grad():
                get_logits(model, ids)
            computed = PAIR_COUNTER["computed"]
            nr = needle_recall(model, tok, ctx, depths, seeds)
            restore(model, originals, full_layers)
            ratio = computed / dense_pairs if dense_pairs else 0.0
            rec = {"experiment": "approx", "method": method.name, "model": model_id,
                   "context": ctx, "params": str(params), "computed_pairs": computed,
                   "pair_ratio": round(ratio, 4), "needle_recall": nr, **fid, **m.to_dict()}
            append_jsonl(RESULTS, rec)
            print(f"  ctx={ctx} {method.name} {params}: pair_ratio={ratio:.3f} "
                  f"kl={fid['kl_div']} cos={fid['logit_cosine']} needle={nr:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS.keys()))
    ap.add_argument("--contexts", type=int, nargs="+", default=[2048])
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()
    methods = [parse_method(s) for s in args.methods]
    run(args.model, args.contexts, methods, args.dtype, args.seed, args.load_in_4bit)


if __name__ == "__main__":
    main()
