"""PER-LAYER sensitivity: which full_attention layers drive the fidelity drop under sparsification.

For each full_attention layer we sparsify ONLY that layer (all others stay dense) and measure the
logit-cosine drop vs the dense reference. A layer whose solo-sparsification already costs a lot of
fidelity is "sensitive" (the method must get it right); a layer that barely moves fidelity is
"cheap" to sparsify. We also report the all-layers condition as the reference point.

This addresses the reviewer request for a per-layer/head mechanical analysis.

Usage:
  python scripts/run_layer_sensitivity.py --model qwen3.5-4b-base --ctx 2048 --n-samples 5
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from attention_interceptor import apply_method, restore  # noqa: E402
from attention_methods import AttentionMethod  # noqa: E402
from model_loader import MODELS, inspect_model, load_model  # noqa: E402
from utils import append_jsonl, fidelity_metrics, set_seed  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "layer_sensitivity.jsonl")


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


def cos_for(model, chunks, refs, method, layers, seed):
    originals = apply_method(model, method, layers) if layers else {}
    try:
        vals = []
        for ids, ref in zip(chunks, refs):
            set_seed(seed)
            vals.append(fidelity_metrics(get_logits(model, ids), ref)["logit_cosine"])
    finally:
        if layers:
            restore(model, originals, layers)
    return sum(vals) / len(vals)


def run(model_key, ctx, split, n_samples, dtype, seed, load_in_4bit):
    model_id = MODELS[model_key]
    info = inspect_model(model_id)
    model, tok = load_model(model_id, dtype=dtype, load_in_4bit=load_in_4bit)
    device = next(model.parameters()).device
    full_layers = info.full_attention_layers
    chunks = [c.unsqueeze(0).to(device) for c in build_chunks(tok, ctx, n_samples, split)]
    print(f"[load] {model_id} ctx={ctx} n={len(chunks)} full_layers={full_layers}")

    refs = []
    for ids in chunks:
        set_seed(seed)
        refs.append(get_logits(model, ids))

    method = AttentionMethod(name="local_global_topk", window=64, n_global=4, topk=32)

    # all-layers reference
    c_all = cos_for(model, chunks, refs, method, full_layers, seed)
    rec = {"experiment": "layer_sensitivity", "scope": "all", "layer": -1, "model": model_id,
           "context": ctx, "n_samples": len(chunks), "cos_mean": round(c_all, 4)}
    append_jsonl(RESULTS, rec)
    print(f"  [all-layers]            cos={c_all:.4f}")

    # one layer at a time
    for idx in full_layers:
        c = cos_for(model, chunks, refs, method, [idx], seed)
        rec = {"experiment": "layer_sensitivity", "scope": "single", "layer": idx,
               "model": model_id, "context": ctx, "n_samples": len(chunks),
               "cos_mean": round(c, 4)}
        append_jsonl(RESULTS, rec)
        print(f"  [only layer {idx:2d}]        cos={c:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-4b-base", choices=list(MODELS.keys()))
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-quantize", action="store_true")
    args = ap.parse_args()
    run(args.model, args.ctx, args.split, args.n_samples, args.dtype, args.seed,
        load_in_4bit=not args.no_quantize)


if __name__ == "__main__":
    main()
