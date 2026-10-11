"""Run needle-in-a-haystack retrieval evaluation across depths and seeds.

Measures whether the model retrieves a passkey hidden at a given depth in a long
context. This is the canonical full-attention-sensitive task.

Usage:
  python scripts/run_needle.py --model lfm2.5-350m-base --contexts 1024 2048 --depths 0.1 0.5 0.9
"""

import argparse
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model_loader import MODELS, inspect_model, load_model  # noqa: E402
from needle import build_needle_prompt, check_needle, make_needle  # noqa: E402
from utils import GpuMeter, append_jsonl, set_seed  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "needle.jsonl")


@torch.no_grad()
def run_one(model, tok, context_len, depth, seed, gen_tokens=24):
    set_seed(seed)
    rng = random.Random(seed)
    needle = make_needle(rng)
    prompt, needle = build_needle_prompt(tok, context_len, needle, depth)
    ids = tok(prompt, return_tensors="pt").input_ids.to(next(model.parameters()).device)
    with GpuMeter() as m:
        out = model.generate(ids, max_new_tokens=gen_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    hit = check_needle(gen, needle)
    return hit, gen, needle, m.to_dict()


def run(model_key, contexts, depths, seeds, dtype, gen_tokens):
    model_id = MODELS[model_key]
    info = inspect_model(model_id)
    model, tok = load_model(model_id, dtype=dtype)
    print(f"[load] {model_id} full_attn={info.full_attention_layers}")
    for ctx in contexts:
        for depth in depths:
            hits = 0
            for seed in seeds:
                hit, gen, needle, gpu = run_one(model, tok, ctx, depth, seed, gen_tokens)
                hits += int(hit)
                rec = {
                    "experiment": "needle", "model": model_id, "model_key": model_key,
                    "context": ctx, "depth": depth, "seed": seed,
                    "n_full_attention": info.n_full_attention,
                    "hit": bool(hit), "needle": needle, "generated": gen[:80],
                    **gpu,
                }
                append_jsonl(RESULTS, rec)
            acc = hits / len(seeds)
            print(f"  ctx={ctx} depth={depth} recall={acc:.2f} ({hits}/{len(seeds)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS.keys()))
    ap.add_argument("--contexts", type=int, nargs="+", default=[1024])
    ap.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--gen-tokens", type=int, default=24)
    args = ap.parse_args()
    run(args.model, args.contexts, args.depths, args.seeds, args.dtype, args.gen_tokens)


if __name__ == "__main__":
    main()
