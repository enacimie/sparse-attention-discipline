"""Reviewer round-7 (W3/E2): statistical power for needle recall at the working budget.

The manuscript's needle recall at pair_ratio ~0.09 was n=9 (Wilson CI 0.66-1.0), which
cannot separate a perfect method from one that fails 1/3 of the time. This re-runs the
working point (window 64 + sinks 4 + top-k 32) with ~100 probes (depths x seeds) per
method and stores per-case outcomes so paired tests (McNemar vs dense) and Wilson CIs
can be computed without reruns.

Methods: dense, lg_topk, lg_randtopk (structurally-matched random), streamingllm
(budget-matched local window), uniform_random (budget-matched), and optionally
lg_oracle (dense-posterior upper bound, slow).

Output: results/needle_power.jsonl
  {"experiment": "needle_power", "model", "context", "method", "pair_ratio",
   "hits", "n", "recall", "wilson_lo", "wilson_hi",
   "cases": [{"depth", "seed", "hit"}, ...]}

Usage:
  python scripts/run_needle_power.py --model qwen3.5-0.8b --ctx 2048
  python scripts/run_needle_power.py --model qwen3.5-4b-base --ctx 2048 --seeds $(seq 42 50)
"""
import argparse
import math
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

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "needle_power.jsonl")
QUESTION = "What is the special passkey mentioned in the text?"
WINDOW, SINKS, TOPK = 64, 4, 32


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


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
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    orig = apply_method(model, method, full_layers)
    try:
        reset_pair_counter()
        with torch.no_grad():
            model(ids)
        return PAIR_COUNTER["computed"] / max(1, PAIR_COUNTER["full"])
    finally:
        restore(model, orig, full_layers)


def build_method(name, seed, streaming_window, random_topk):
    if name == "dense":
        return AttentionMethod(name="dense")
    if name == "lg_topk":
        return AttentionMethod(name="local_global_topk", window=WINDOW, n_global=SINKS,
                               topk=TOPK)
    if name == "lg_randtopk":
        return AttentionMethod(name="local_global_randtopk", window=WINDOW, n_global=SINKS,
                               topk=TOPK, seed=seed)
    if name == "lg_oracle":
        return AttentionMethod(name="local_global_oracle", window=WINDOW, n_global=SINKS,
                               topk=TOPK)
    if name == "streamingllm":
        return AttentionMethod(name="local_global", window=streaming_window,
                               n_global=SINKS, topk=0)
    if name == "uniform_random":
        return AttentionMethod(name="uniform_random", topk=random_topk, random_k=0, seed=seed)
    raise ValueError(name)


def run(model_key, ctx, depths, seeds, dtype, load_in_4bit, methods):
    model_id = MODELS[model_key]
    info = inspect_model(model_id)
    model, tok = load_model(model_id, dtype=dtype, load_in_4bit=load_in_4bit)
    device = next(model.parameters()).device
    full_layers = info.full_attention_layers
    print(f"[load] {model_id} full_attn={len(full_layers)} ctx={ctx} "
          f"probes={len(depths) * len(seeds)} methods={methods}")

    cases = []
    for depth in depths:
        for seed in seeds:
            set_seed(seed)
            rng = random.Random(seed)
            needle = make_needle(rng)
            haystack = build_haystack(tok, ctx, needle, depth)
            prompt = haystack + "\nQuestion: " + QUESTION + "\nAnswer:"
            cases.append({"depth": depth, "seed": seed, "needle": needle, "prompt": prompt})

    # budget matching: measure lg_topk pair_ratio, then calibrate the two
    # budget-matched controls (streamingllm window, uniform_random topk)
    pr_lg = measure_pair_ratio(model, tok, cases[0]["prompt"], device,
                               build_method("lg_topk", seeds[0], 0, 0), full_layers)
    w_try = max(4, round(pr_lg * ctx / 2) - SINKS)
    streaming_window = w_try
    for _ in range(6):
        m = build_method("streamingllm", seeds[0], w_try, 0)
        pr_s = measure_pair_ratio(model, tok, cases[0]["prompt"], device, m, full_layers)
        if abs(pr_s - pr_lg) < 0.01:
            break
        w_try = max(4, w_try + round((pr_lg - pr_s) * ctx / 2))
        streaming_window = w_try
    k_try = max(8, round(pr_lg * ctx / 2))
    random_topk = k_try
    for _ in range(6):
        m = build_method("uniform_random", seeds[0], 0, k_try)
        pr_r = measure_pair_ratio(model, tok, cases[0]["prompt"], device, m, full_layers)
        if abs(pr_r - pr_lg) < 0.01:
            break
        k_try = max(4, k_try + round((pr_lg - pr_r) * ctx / 2))
        random_topk = k_try
    print(f"  lg_topk pr={pr_lg:.4f} | streamingllm window={streaming_window} "
          f"pr={pr_s:.4f} | uniform_random topk={random_topk} pr={pr_r:.4f}")

    for name in methods:
        pr = {"dense": 1.0, "lg_topk": pr_lg, "lg_randtopk": None, "lg_oracle": None,
              "streamingllm": pr_s, "uniform_random": pr_r}[name]
        if pr is None:
            pr = measure_pair_ratio(model, tok, cases[0]["prompt"], device,
                                    build_method(name, seeds[0], streaming_window, random_topk),
                                    full_layers)
        outcomes = []
        hits = 0
        for c in cases:
            set_seed(c["seed"])
            method = build_method(name, c["seed"], streaming_window, random_topk)
            orig = apply_method(model, method, full_layers)
            try:
                g = gen_answer(model, tok, c["prompt"], device)
            finally:
                restore(model, orig, full_layers)
            hit = int(check_needle(g, c["needle"]))
            hits += hit
            outcomes.append({"depth": c["depth"], "seed": c["seed"], "hit": hit})
        n = len(cases)
        lo, hi = wilson(hits, n)
        print(f"  {name:14s} pr={pr:.4f} recall={hits}/{n} ({hits / n:.3f}) "
              f"wilson=[{lo:.3f}-{hi:.3f}]")
        append_jsonl(RESULTS, {"experiment": "needle_power", "model": model_id,
                               "context": ctx, "method": name,
                               "pair_ratio": round(pr, 4), "hits": hits, "n": n,
                               "recall": round(hits / n, 4),
                               "wilson_lo": round(lo, 4), "wilson_hi": round(hi, 4),
                               "window": WINDOW, "sinks": SINKS, "topk": TOPK,
                               "streaming_window": streaming_window,
                               "random_topk": random_topk,
                               "cases": outcomes})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-0.8b", choices=list(MODELS.keys()))
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--depths", type=float, nargs="+",
                    default=[0.05, 0.15, 0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 60)))
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--no-quantize", action="store_true")
    ap.add_argument("--methods", nargs="+",
                    default=["dense", "lg_topk", "lg_randtopk", "streamingllm",
                             "uniform_random"],
                    choices=["dense", "lg_topk", "lg_randtopk", "lg_oracle",
                             "streamingllm", "uniform_random"])
    args = ap.parse_args()
    run(args.model, args.ctx, args.depths, args.seeds, args.dtype,
        load_in_4bit=not args.no_quantize, methods=args.methods)


if __name__ == "__main__":
    main()
