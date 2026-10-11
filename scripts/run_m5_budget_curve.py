"""Reviewer M5 / P5.1: retrieval budget curve (recall vs pair_ratio).

The retrieval boundary was only measured at one low budget (~4-16% pairs), where every sparse
single-forward pattern scores recall 0.00. The reviewer asks whether exact retrieval RECovers at
higher pair budgets (25/50/75/90%). This sweeps the content top-k budget (via `topk`) from low to
near-dense and measures needle recall and the actual pair_ratio at each point, plus dense as the
upper bound. If recall stays ~0 until very high budget, the boundary is sharp; if it recovers at
~50%, the boundary is budget-dependent.

Usage:
  python scripts/run_m5_budget_curve.py --model qwen3.5-0.8b --ctx 2048 --seeds 42 43 44
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
from needle import FILLER, check_needle, make_needle  # noqa: E402
from utils import append_jsonl, set_seed  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "m5_budget_curve.jsonl")
QUESTION = "What is the special passkey mentioned in the text?"
# topk values spanning low -> near-dense pair budgets (window=64, n_global=4 fixed)
TOPK_GRID = [32, 100, 300, 600, 1000, 1500, 1800]


def build_haystack(tok, ctx, needle, depth):
    needle_sentence = f"The special passkey is {needle}."
    needle_ids = tok(" " + needle_sentence + " ", add_special_tokens=False).input_ids
    n_rep = ctx // 30 + 5
    filler_ids = tok(FILLER * n_rep, add_special_tokens=False).input_ids
    filler_ids = filler_ids[: max(ctx - len(needle_ids), 0)]
    n_filler = len(filler_ids)
    pos = int(n_filler * depth)
    ids = filler_ids[:pos] + needle_ids + filler_ids[pos:]
    return tok.decode(ids)


@torch.no_grad()
def gen_answer(model, tok, prompt, device, gen_tokens=24):
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    out = model.generate(ids, max_new_tokens=gen_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def measure_pair_ratio(model, tok, prompt, device, method, full_layers):
    """Pair_ratio over the prefill forward of the full prompt."""
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    orig = apply_method(model, method, full_layers)
    try:
        reset_pair_counter()
        with torch.no_grad():
            model(ids)
        pr = PAIR_COUNTER["computed"] / max(1, PAIR_COUNTER["full"])
    finally:
        restore(model, orig, full_layers)
    return pr


def run(model_key, ctx, depths, seeds, dtype, load_in_4bit, methods):
    model_id = MODELS[model_key]
    info = inspect_model(model_id)
    model, tok = load_model(model_id, dtype=dtype, load_in_4bit=load_in_4bit)
    device = next(model.parameters()).device
    full_layers = info.full_attention_layers
    print(f"[load] {model_id} full_attn={len(full_layers)} ctx={ctx} methods={methods}")

    cases = []
    if "dense" in methods:
        dense_hits = 0
        dense_n = 0
        for depth in depths:
            for seed in seeds:
                set_seed(seed)
                rng = random.Random(seed)
                needle = make_needle(rng)
                haystack = build_haystack(tok, ctx, needle, depth)
                prompt = haystack + "\nQuestion: " + QUESTION + "\nAnswer:"
                cases.append((depth, seed, needle, prompt))
                g = gen_answer(model, tok, prompt, device)
                dense_hits += int(check_needle(g, needle))
                dense_n += 1
        print(f"  dense recall = {dense_hits}/{dense_n}")
        append_jsonl(RESULTS, {"experiment": "m5_budget_curve", "model": model_id, "context": ctx,
                               "method": "dense", "topk": -1, "pair_ratio": 1.0,
                               "recall": round(dense_hits / dense_n, 4), "n": dense_n})
    else:
        for depth in depths:
            for seed in seeds:
                set_seed(seed)
                rng = random.Random(seed)
                needle = make_needle(rng)
                haystack = build_haystack(tok, ctx, needle, depth)
                prompt = haystack + "\nQuestion: " + QUESTION + "\nAnswer:"
                cases.append((depth, seed, needle, prompt))

    if "lg_topk" in methods:
        for topk in TOPK_GRID:
            method = AttentionMethod(name="local_global_topk", window=64, n_global=4, topk=topk)
            pr = measure_pair_ratio(model, tok, cases[0][3], device, method, full_layers)
            hits = 0
            for (depth, seed, needle, prompt) in cases:
                set_seed(seed)
                orig = apply_method(model, method, full_layers)
                try:
                    g = gen_answer(model, tok, prompt, device)
                finally:
                    restore(model, orig, full_layers)
                hits += int(check_needle(g, needle))
            rec = hits / len(cases)
            print(f"  lg_topk topk={topk:5d} pair_ratio={pr:.4f} recall={hits}/{len(cases)} ({rec:.3f})")
            append_jsonl(RESULTS, {"experiment": "m5_budget_curve", "model": model_id, "context": ctx,
                                   "method": "local_global_topk", "topk": topk,
                                   "pair_ratio": round(pr, 4), "recall": round(rec, 4),
                                   "n": len(cases)})

    if "lg_randtopk" in methods:
        for topk in TOPK_GRID:
            probe = AttentionMethod(name="local_global_randtopk", window=64, n_global=4,
                                    topk=topk, seed=seeds[0])
            pr = measure_pair_ratio(model, tok, cases[0][3], device, probe, full_layers)
            hits = 0
            for (depth, seed, needle, prompt) in cases:
                set_seed(seed)
                method = AttentionMethod(name="local_global_randtopk", window=64, n_global=4,
                                         topk=topk, seed=seed)
                orig = apply_method(model, method, full_layers)
                try:
                    g = gen_answer(model, tok, prompt, device)
                finally:
                    restore(model, orig, full_layers)
                hits += int(check_needle(g, needle))
            rec = hits / len(cases)
            print(f"  lg_randtopk topk={topk:5d} pair_ratio={pr:.4f} recall={hits}/{len(cases)} ({rec:.3f})")
            append_jsonl(RESULTS, {"experiment": "m5_budget_curve", "model": model_id, "context": ctx,
                                   "method": "local_global_randtopk", "topk": topk,
                                   "pair_ratio": round(pr, 4), "recall": round(rec, 4),
                                   "n": len(cases)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-0.8b", choices=list(MODELS.keys()))
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--no-quantize", action="store_true")
    ap.add_argument("--methods", nargs="+", default=["dense", "lg_topk"],
                    choices=["dense", "lg_topk", "lg_randtopk"])
    args = ap.parse_args()
    run(args.model, args.ctx, args.depths, args.seeds, args.dtype,
        load_in_4bit=not args.no_quantize, methods=args.methods)


if __name__ == "__main__":
    main()
