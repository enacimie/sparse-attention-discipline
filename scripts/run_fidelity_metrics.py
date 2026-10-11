"""Full fidelity-metric panel (reviewer §5.2): report KL, logit-MSE, logit-cosine AND
hidden-state cosine, not just logit-cosine. Hidden-state cosine compares the pre-lm_head
representation, showing the approximation error before the output projection.

Usage:
  python scripts/run_fidelity_metrics.py --model qwen3.5-4b-base --ctx 2048 --n-samples 5
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from attention_interceptor import apply_method, restore  # noqa: E402
from attention_methods import AttentionMethod  # noqa: E402
from model_loader import MODELS, inspect_model, load_model  # noqa: E402
from utils import append_jsonl, fidelity_metrics, set_seed, cosine_sim  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "fidelity_metrics.jsonl")


@torch.no_grad()
def get_logits_hidden(model, ids):
    out = model(ids, output_hidden_states=True)
    return out.logits[0].cpu().float(), out.hidden_states[-1][0].cpu().float()


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


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def eval_method(model, chunks, refs, method, full_layers, seed):
    originals = apply_method(model, method, full_layers) if method else {}
    try:
        kls, mses, lcos, hcos = [], [], [], []
        for ids, (ref_l, ref_h) in zip(chunks, refs):
            set_seed(seed)
            l, h = get_logits_hidden(model, ids)
            fid = fidelity_metrics(l, ref_l)
            kls.append(fid["kl_div"]); mses.append(fid["logit_mse"]); lcos.append(fid["logit_cosine"])
            hcos.append(cosine_sim(h, ref_h))
    finally:
        if method:
            restore(model, originals, full_layers)
    return mean(kls), mean(mses), mean(lcos), mean(hcos)


def run(model_key, ctx, split, n_samples, dtype, seed, load_in_4bit):
    model_id = MODELS[model_key]
    info = inspect_model(model_id)
    model, tok = load_model(model_id, dtype=dtype, load_in_4bit=load_in_4bit)
    device = next(model.parameters()).device
    full_layers = info.full_attention_layers
    chunks = [c.unsqueeze(0).to(device) for c in build_chunks(tok, ctx, n_samples, split)]
    print(f"[load] {model_id} ctx={ctx} n={len(chunks)}")

    refs = []
    for ids in chunks:
        set_seed(seed)
        refs.append(get_logits_hidden(model, ids))

    budget = 64 + 4 + 32
    methods = [("dense", None),
               ("local_global_topk", AttentionMethod(name="local_global_topk", window=64,
                                                     n_global=4, topk=32)),
               ("streamingllm_bm", AttentionMethod(name="local_global", window=budget - 4,
                                                   n_global=4)),
               ("uniform_random_m", AttentionMethod(name="uniform_random", topk=budget,
                                                    random_k=0))]
    for tag, method in methods:
        kl, mse_v, lc, hc = eval_method(model, chunks, refs, method, full_layers, seed)
        rec = {"experiment": "fidelity_metrics", "method": tag, "model": model_id, "context": ctx,
               "n_samples": len(chunks), "kl_div": round(kl, 5), "logit_mse": round(mse_v, 5),
               "logit_cosine": round(lc, 5), "hidden_cosine": round(hc, 5)}
        append_jsonl(RESULTS, rec)
        print(f"  {tag:18s} kl={kl:.4f} mse={mse_v:.4f} logit_cos={lc:.4f} hidden_cos={hc:.4f}")


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
