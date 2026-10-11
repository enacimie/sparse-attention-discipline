"""ANTI-OVERFITTING robustness validation for local_global_topk.

Everything so far was evaluated on the wikitext VALIDATION split with 1-2 chunks; the config
(w=64,g=4,k=32) was chosen earlier. To rule out overfitting we check three things, all on the
held-out TEST split (never used for tuning/eval):
  A. Held-out generalization: chosen config vs budget-matched uniform_random, n_samples chunks,
     reporting logit-cosine mean +/- std (a CI proxy) and the advantage over random.
  B. Hyperparameter sensitivity: sweep a neighbourhood of the config (window x topk, g fixed).
     If the advantage is knife-edge (only at the one chosen point) it smells of overfitting; if it
     holds across the neighbourhood it is a robust effect.

Usage:
  python scripts/run_robustness.py --model qwen3.5-4b-base --ctx 2048 --split test \
      --n-samples 5 --sweep
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

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "robustness.jsonl")


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
    """Return per-chunk cos list, mean pair_ratio. method=None -> dense (refs)."""
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


def run(model_key, ctx, split, n_samples, dtype, seed, sweep, load_in_4bit,
        device_map="cuda"):
    model_id = MODELS[model_key]
    info = inspect_model(model_id)
    model, tok = load_model(model_id, dtype=dtype, load_in_4bit=load_in_4bit,
                            device_map=device_map)
    device = next(model.parameters()).device
    full_layers = info.full_attention_layers
    chunks = [c.unsqueeze(0).to(device) for c in build_chunks(tok, ctx, n_samples, split)]
    L = chunks[0].shape[1]
    dense_pairs = (L * (L + 1) // 2) * info.n_full_attention
    print(f"[load] {model_id} split={split} ctx={ctx} n={len(chunks)}")

    # dense references
    refs = []
    for ids in chunks:
        set_seed(seed)
        refs.append(get_logits(model, ids))

    # ---- Part A: chosen config vs matched random, held-out, with variance ----
    chosen = AttentionMethod(name="local_global_topk", window=64, n_global=4, topk=32)
    budget = 64 + 4 + 32  # approximate pair budget of the chosen config
    rand = AttentionMethod(name="uniform_random", topk=budget, random_k=0, seed=seed)
    # SOTA static-pattern baseline: StreamingLLM = sinks + sliding window, NO content top-k.
    # (a) its natural config (w=64,g=4); (b) budget-matched: widen the window to ~budget so the
    # comparison isolates whether content top-k beats pure locality at EQUAL budget.
    stream_nat = AttentionMethod(name="local_global", window=64, n_global=4)
    stream_bm = AttentionMethod(name="local_global", window=budget - 4, n_global=4)
    lg_block = AttentionMethod(name="lg_topk_block", window=64, n_global=4, topk=1)
    lgr_block = AttentionMethod(name="lgr_block", window=64, n_global=4, topk=1)
    for tag, method in [("local_global_topk", chosen), ("uniform_random_m", rand),
                        ("streamingllm_nat", stream_nat), ("streamingllm_bm", stream_bm),
                        ("lg_topk_block", lg_block), ("lgr_block", lgr_block)]:
        cos_l, pr = eval_method(model, chunks, refs, method, full_layers, seed, dense_pairs)
        m, sd = mean_std(cos_l)
        rec = {"experiment": "robustness", "part": "A_heldout", "method": tag, "model": model_id,
               "context": ctx, "split": split, "n_samples": len(chunks), "cos_mean": round(m, 4),
               "cos_std": round(sd, 4), "pair_ratio": round(pr, 4),
               "params": str({k: v for k, v in vars(method).items() if k != "name"})}
        append_jsonl(RESULTS, rec)
        print(f"  [A] {tag:18s} cos={m:.4f}+/-{sd:.4f} pr={pr:.4f}")

    # ---- Part B: hyperparameter sensitivity sweep on the held-out split ----
    if sweep:
        for window in [32, 64, 128]:
            for topk in [16, 32, 64]:
                cfg = AttentionMethod(name="local_global_topk", window=window, n_global=4,
                                      topk=topk)
                cos_l, pr = eval_method(model, chunks, refs, cfg, full_layers, seed, dense_pairs)
                m, sd = mean_std(cos_l)
                rec = {"experiment": "robustness", "part": "B_sweep", "method": "local_global_topk",
                       "model": model_id, "context": ctx, "split": split, "window": window,
                       "topk": topk, "n_global": 4, "n_samples": len(chunks),
                       "cos_mean": round(m, 4), "cos_std": round(sd, 4), "pair_ratio": round(pr, 4)}
                append_jsonl(RESULTS, rec)
                print(f"  [B] w={window:3d} k={topk:2d} cos={m:.4f}+/-{sd:.4f} pr={pr:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-4b-base", choices=list(MODELS.keys()))
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--no-quantize", action="store_true")
    ap.add_argument("--device-map", default="cuda",
                    help="'cuda' (single GPU) or 'auto' (split across GPUs/CPU)")
    args = ap.parse_args()
    run(args.model, args.ctx, args.split, args.n_samples, args.dtype, args.seed, args.sweep,
        load_in_4bit=not args.no_quantize, device_map=args.device_map)


if __name__ == "__main__":
    main()
