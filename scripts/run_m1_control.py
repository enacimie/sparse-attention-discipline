"""Reviewer M1 control: structurally-matched random baseline.

The central claim is that content top-k beats a budget-matched random mask. But the prior
uniform_random control lacks the local-window + sinks structure of local_global_topk, so part of
the advantage could come from structure, not content-awareness. The clean control is
local_global_randtopk: SAME local window + SAME sinks, but the top-k free keys chosen uniformly
at random instead of by content. If lg_topk does not beat THIS at equal pair budget, the
content-awareness claim is confounded with local/sinks structure.

Methods compared (all held-out test split, same chunks/seed):
  local_global_topk      local(64) + sinks(4) + content-topk(32)   <- the recipe
  local_global_randtopk  local(64) + sinks(4) + random(32)          <- THE M1 control
  uniform_random_m       random at budget 100, no structure         <- prior control
  streamingllm_bm        local(96) + sinks(4), no top-k             <- structural baseline

Usage:
  python scripts/run_m1_control.py --model qwen3.5-0.8b --ctx 2048 --n-samples 5 --n-seeds 3
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

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "m1_control.jsonl")


@torch.no_grad()
def get_logits(model, ids):
    return model(ids).logits[0].cpu().float()


def build_chunks(tok, ctx, n_samples, split="test"):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
    text = "\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    chunks, step = [], max(1, (ids.numel() - ctx) // max(1, n_samples))
    for s in range(0, ids.numel() - ctx + 1, step):
        chunks.append(ids[s:s + ctx])
        if len(chunks) >= n_samples:
            break
    return chunks if chunks else [ids[:ctx]]


def mean_std(xs):
    m = sum(xs) / len(xs)
    if len(xs) > 1:
        var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
        sd = var ** 0.5
    else:
        sd = 0.0
    return m, sd


def eval_method(model, chunks, refs, method, full_layers, seed, dense_pairs):
    originals = apply_method(model, method, full_layers) if method else {}
    try:
        cos_l, ratios = [], []
        for ids, ref in zip(chunks, refs):
            set_seed(seed)
            fid = fidelity_metrics(get_logits(model, ids), ref)
            cos_l.append(fid["logit_cosine"])
            if method is not None:
                reset_pair_counter()
                with torch.no_grad():
                    get_logits(model, ids)
                ratios.append(PAIR_COUNTER["computed"] / dense_pairs)
    finally:
        if method:
            restore(model, originals, full_layers)
    return cos_l, (sum(ratios) / len(ratios) if ratios else 1.0)


def run(model_key, ctx, split, n_samples, n_seeds, dtype, device_map):
    model_id = MODELS[model_key]
    info = inspect_model(model_id)
    model, tok = load_model(model_id, dtype=dtype, load_in_4bit=True, device_map=device_map)
    device = next(model.parameters()).device
    full_layers = info.full_attention_layers
    chunks = [c.unsqueeze(0).to(device) for c in build_chunks(tok, ctx, n_samples, split)]
    L = chunks[0].shape[1]
    dense_pairs = (L * (L + 1) // 2) * info.n_full_attention
    print(f"[load] {model_id} split={split} ctx={ctx} n={len(chunks)} n_seeds={n_seeds}")

    refs = []
    for ids in chunks:
        set_seed(42)
        refs.append(get_logits(model, ids))

    budget = 64 + 4 + 32
    # deterministic recipe vs the two random controls (run over n_seeds) + structural baseline
    methods_det = [
        ("local_global_topk", AttentionMethod(name="local_global_topk", window=64,
                                              n_global=4, topk=32)),
        ("streamingllm_bm", AttentionMethod(name="local_global", window=budget - 4,
                                            n_global=4)),
    ]
    methods_rand = [
        ("local_global_randtopk", lambda s: AttentionMethod(name="local_global_randtopk",
                                                            window=64, n_global=4, topk=32,
                                                            seed=s)),
        ("uniform_random_m", lambda s: AttentionMethod(name="uniform_random", topk=budget,
                                                       random_k=0, seed=s)),
    ]

    for tag, method in methods_det:
        cos_l, pr = eval_method(model, chunks, refs, method, full_layers, 42, dense_pairs)
        m, sd = mean_std(cos_l)
        rec = {"experiment": "m1_control", "method": tag, "model": model_id, "context": ctx,
               "split": split, "n_samples": len(chunks), "n_seeds": 1, "cos_mean": round(m, 4),
               "cos_std": round(sd, 4), "pair_ratio": round(pr, 4),
               "params": str({k: v for k, v in vars(method).items() if k != "name"})}
        append_jsonl(RESULTS, rec)
        print(f"  {tag:22s} cos={m:.4f}+/-{sd:.4f} pr={pr:.4f}")

    for tag, mk in methods_rand:
        all_cos, all_pr = [], []
        for s in range(42, 42 + n_seeds):
            method = mk(s)
            cos_l, pr = eval_method(model, chunks, refs, method, full_layers, s, dense_pairs)
            all_cos.extend(cos_l)
            all_pr.append(pr)
        m, sd = mean_std(all_cos)
        pr_m = sum(all_pr) / len(all_pr)
        rec = {"experiment": "m1_control", "method": tag, "model": model_id, "context": ctx,
               "split": split, "n_samples": len(chunks), "n_seeds": n_seeds,
               "cos_mean": round(m, 4), "cos_std": round(sd, 4), "pair_ratio": round(pr_m, 4),
               "params": str({k: v for k, v in vars(mk(42)).items() if k != "name"})}
        append_jsonl(RESULTS, rec)
        print(f"  {tag:22s} cos={m:.4f}+/-{sd:.4f} pr={pr_m:.4f}  (n_seeds={n_seeds})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-0.8b", choices=list(MODELS.keys()))
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--device-map", default="cuda")
    args = ap.parse_args()
    run(args.model, args.ctx, args.split, args.n_samples, args.n_seeds, args.dtype,
        args.device_map)


if __name__ == "__main__":
    main()
