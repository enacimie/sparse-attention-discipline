"""Reviewer M3: multi-fact retrieval re-run with per-seed records (CI data).

The PAPER.md 5 multi-fact row (dense 0.81/1.00, lg_topk 0.63/1.00, controls 0.00) was
reported without a stored per-seed artifact. This re-runs it on the current fixed
harness and stores per-case hits so means/CI can be computed.

Methods: dense, local_global_topk, local_global_randtopk (structural control),
streamingllm_bm (budget-matched), uniform_random (budget-matched).

Output: results/multifact_recall.jsonl (per case).

Usage:
  python scripts/run_multifact_recall.py --models qwen3.5-0.8b qwen3.5-4b-base
"""
import argparse
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from attention_interceptor import apply_method, restore  # noqa: E402
from attention_methods import AttentionMethod  # noqa: E402
from model_loader import MODELS, inspect_model, load_model  # noqa: E402
from needle import build_multifact_haystack  # noqa: E402
from utils import append_jsonl, set_seed  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "multifact_recall.jsonl")


@torch.no_grad()
def gen_answer(model, tok, prompt, device, gen_tokens=16):
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    out = model.generate(ids, max_new_tokens=gen_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def methods_for(pr_lg, seed):
    m = {
        "lg_topk": AttentionMethod(name="local_global_topk", window=64, n_global=4, topk=32),
        "lg_randtopk": AttentionMethod(name="local_global_randtopk", window=64, n_global=4,
                                       topk=32, seed=seed),
        "streamingllm": AttentionMethod(name="local_global", window=max(16, round(pr_lg * 2048)),
                                        n_global=4),
        "uniform_random": AttentionMethod(name="uniform_random",
                                          topk=max(16, round(pr_lg * 2048)), random_k=0,
                                          seed=seed),
    }
    return m


def run(model_key, contexts, n_facts_grid, depths, seeds, dtype, load_in_4bit):
    model_id = MODELS[model_key]
    info = inspect_model(model_id)
    model, tok = load_model(model_id, dtype=dtype, load_in_4bit=load_in_4bit)
    device = next(model.parameters()).device
    full_layers = info.full_attention_layers
    print(f"[load] {model_id} full_attn={len(full_layers)}")

    for ctx in contexts:
        for n_facts in n_facts_grid:
            for depth in depths:
                for seed in seeds:
                    set_seed(seed)
                    rng = random.Random(seed)
                    body, facts, key, value = build_multifact_haystack(
                        tok, ctx, n_facts, rng, target_depth=depth)
                    prompt = body + "\n\nQuestion: What is the access code for " + key + "?\nAnswer:"
                    for name, method in methods_for(0.09, seed).items():
                        orig = apply_method(model, method, full_layers)
                        try:
                            g = gen_answer(model, tok, prompt, device)
                        finally:
                            restore(model, orig, full_layers)
                        hit = int(str(value) in g)
                        print(f"  ctx={ctx} n_facts={n_facts} depth={depth} seed={seed} "
                              f"{name:14s} hit={hit} gen='{g.strip()[:40]}'")
                        append_jsonl(RESULTS, {"experiment": "multifact_recall", "model": model_id,
                                               "context": ctx, "n_facts": n_facts, "depth": depth,
                                               "seed": seed, "method": name, "hit": bool(hit),
                                               "target_value": str(value), "generated": g})
                    # dense
                    g = gen_answer(model, tok, prompt, device)
                    hit = int(str(value) in g)
                    print(f"  ctx={ctx} n_facts={n_facts} depth={depth} seed={seed} "
                          f"{'dense':14s} hit={hit}")
                    append_jsonl(RESULTS, {"experiment": "multifact_recall", "model": model_id,
                                           "context": ctx, "n_facts": n_facts, "depth": depth,
                                           "seed": seed, "method": "dense", "hit": bool(hit),
                                           "target_value": str(value), "generated": g})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen3.5-0.8b", "qwen3.5-4b-base"],
                    choices=list(MODELS.keys()))
    ap.add_argument("--contexts", type=int, nargs="+", default=[2048, 4096])
    ap.add_argument("--n-facts", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--depths", type=float, nargs="+", default=[0.5])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--no-quantize", action="store_true")
    args = ap.parse_args()
    for m in args.models:
        run(m, args.contexts, args.n_facts, args.depths, args.seeds, args.dtype,
            load_in_4bit=not args.no_quantize)


if __name__ == "__main__":
    main()
