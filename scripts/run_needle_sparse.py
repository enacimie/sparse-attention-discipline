"""Needle-in-a-haystack under sparse attention: does the method preserve retrieval?

Applies each attention method to the full_attention layers (SDPA-mask interceptor, the
fidelity-exact path) and measures needle recall across depths and seeds. Compares
local_global_topk against dense and the uniform_random control at matched budget.

Usage:
  python scripts/run_needle_sparse.py --model qwen3.5-4b-base --contexts 2048 \
      --depths 0.1 0.5 0.9 --seeds 42 43 44
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
from needle import build_needle_prompt, check_needle, make_needle  # noqa: E402
from utils import append_jsonl, set_seed  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "needle_sparse.jsonl")


def build_methods(topk_budget, lg_topk=32):
    return [
        ("dense", None),
        (f"lg_topk_k{lg_topk}", AttentionMethod(name="local_global_topk", window=64,
                                                 n_global=4, topk=lg_topk)),
        ("uniform_random_m", AttentionMethod(name="uniform_random", topk=topk_budget,
                                             random_k=0)),
    ]


@torch.no_grad()
def run_one(model, tok, context_len, depth, seed, gen_tokens=24):
    set_seed(seed)
    rng = random.Random(seed)
    needle = make_needle(rng)
    prompt, needle = build_needle_prompt(tok, context_len, needle, depth)
    ids = tok(prompt, return_tensors="pt").input_ids.to(next(model.parameters()).device)
    out = model.generate(ids, max_new_tokens=gen_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    return check_needle(gen, needle), gen, needle


def run(model_key, contexts, depths, seeds, dtype, gen_tokens, topk_budget, load_in_4bit,
        lg_topk=32):
    model_id = MODELS[model_key]
    info = inspect_model(model_id)
    model, tok = load_model(model_id, dtype=dtype, load_in_4bit=load_in_4bit)
    full_layers = info.full_attention_layers
    print(f"[load] {model_id} full_attn={len(full_layers)} lg_topk={lg_topk}")
    for ctx in contexts:
        for name, method in build_methods(topk_budget, lg_topk):
            originals = apply_method(model, method, full_layers) if method else {}
            hits = 0
            total = 0
            try:
                for depth in depths:
                    h_depth = 0
                    for seed in seeds:
                        hit, gen, needle = run_one(model, tok, ctx, depth, seed, gen_tokens)
                        hits += int(hit); h_depth += int(hit); total += 1
                        append_jsonl(RESULTS, {"experiment": "needle_sparse", "method": name,
                                               "model": model_id, "context": ctx, "depth": depth,
                                               "seed": seed, "hit": bool(hit), "needle": needle,
                                               "generated": gen[:80]})
                    print(f"  ctx={ctx} {name:18s} depth={depth} recall={h_depth/len(seeds):.2f}")
            finally:
                if method:
                    restore(model, originals, full_layers)
            print(f"  ctx={ctx} {name:18s} OVERALL recall={hits/max(total,1):.3f} ({hits}/{total})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-4b-base", choices=list(MODELS.keys()))
    ap.add_argument("--contexts", type=int, nargs="+", default=[2048])
    ap.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--topk-budget", type=int, default=96)
    ap.add_argument("--lg-topk", type=int, default=32)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--gen-tokens", type=int, default=24)
    ap.add_argument("--no-quantize", action="store_true")
    args = ap.parse_args()
    run(args.model, args.contexts, args.depths, args.seeds, args.dtype, args.gen_tokens,
        args.topk_budget, load_in_4bit=not args.no_quantize, lg_topk=args.lg_topk)


if __name__ == "__main__":
    main()
