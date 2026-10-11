#!/usr/bin/env python3
"""Does BLOCK-LEVEL content top-k preserve retrieval? (the fusable variant)

The FlexAttention kernel speedup requires block-aligned masks, but the validated recipe is
per-key top-k (not block-aligned). This measures whether block-level content top-k
(lg_topk_block) still preserves needle retrieval, vs the block-random control (lgr_block)
and the validated per-key recipe (local_global_topk) as reference.

If block top-k preserves retrieval, then a cached/fused block mask can deliver BOTH the
kernel speedup (microbenchmark) and retrieval -> the Palanca-1 result.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from attention_methods import AttentionMethod  # noqa: E402
from model_loader import MODELS, inspect_model, load_model  # noqa: E402
from run_leaderboard import run_fidelity, run_needle  # noqa: E402
from utils import append_jsonl  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "block_retrieval.jsonl")


def build_block_methods(blocks, btops):
    methods = []
    for blk in blocks:
        for bt in btops:
            methods.append((f"lg_topk_block_b{blk}_B{bt}",
                            AttentionMethod(name="lg_topk_block", window=blk,
                                            n_global=4, topk=bt)))
    methods.append(("lg_topk_perkey",
                    AttentionMethod(name="local_global_topk", window=64, n_global=4, topk=32)))
    return methods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-0.8b")
    ap.add_argument("--contexts", type=int, nargs="+", default=[2048, 4096])
    ap.add_argument("--blocks", type=int, nargs="+", default=[64])
    ap.add_argument("--btops", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--load-in-4bit", action="store_true", default=True)
    args = ap.parse_args()

    model_id = MODELS[args.model]
    info = inspect_model(model_id)
    model, tok = load_model(model_id, dtype="float16", load_in_4bit=args.load_in_4bit,
                            device_map="cuda")
    full_layers = info.full_attention_layers
    print(f"[load] {model_id} full_attn={len(full_layers)}", flush=True)

    depths = [0.1, 0.5, 0.9]
    seeds = [42, 43, 44]
    for ctx in args.contexts:
        for name, method in build_block_methods(args.blocks, args.btops):
            fid = run_fidelity(model, tok, method, full_layers, ctx, n_chunks=3)
            rec, hits, total = run_needle(model, tok, method, full_layers, ctx, depths, seeds)
            row = {"experiment": "block_retrieval", "model": args.model, "method": name,
                   "context": ctx, "cos_mean": round(fid["cos_mean"], 4),
                   "pr_mean": round(fid["pr_mean"], 4), "needle_recall": round(rec, 4),
                   "needle_hits": hits, "needle_total": total}
            append_jsonl(RESULTS, row)
            print(f"ctx={ctx} {name:20s} cos={fid['cos_mean']:.4f} pr={fid['pr_mean']:.4f} "
                  f"needle={rec:.3f} ({hits}/{total})", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
