"""Fidelity beyond WikiText (reviewer §3.4): does the top-k advantage hold on other domains?

Evaluates local_global_topk vs budget-matched uniform_random (and StreamingLLM) on a CODE corpus
(streamed, so we only pull as many tokens as needed). Each domain is wrapped in try/except so a
dataset/network failure logs an error and the remaining domains still run.

Usage:
  python scripts/run_domains.py --model qwen3.5-4b-base --ctx 2048 --n-samples 5
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

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "domains.jsonl")


@torch.no_grad()
def get_logits(model, ids):
    return model(ids).logits[0].cpu().float()


def ids_for_domain(tok, domain, min_tokens):
    """Return a 1-D token id tensor for the domain with at least min_tokens tokens."""
    from datasets import load_dataset
    if domain == "code":
        ds = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
        parts = []
        n = 0
        for ex in ds:
            parts.append(ex.get("content", ""))
            n += len(ex.get("content", "")) // 4
            if n >= min_tokens:
                break
        text = "\n".join(parts)
    elif domain == "math":
        # genuine math text (gsm8k question+answer). NO wikitext fallback: a silent fallback
        # duplicated the main domain (reviewer caught it). If the math set fails to load, raise so
        # the domain is logged as FAILED, not disguised as wikitext.
        ds = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
        parts, n = [], 0
        for ex in ds:
            t = (ex.get("question", "") or "") + " " + (ex.get("answer", "") or "")
            parts.append(t)
            n += len(t) // 4
            if n >= min_tokens:
                break
        text = "\n".join(parts)
    else:
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
        text = "\n".join(t for t in ds["text"] if t.strip())
    return tok(text, return_tensors="pt").input_ids[0]


def build_chunks(ids, ctx, n_samples):
    chunks, step = [], max(1, (ids.numel() - ctx) // max(1, n_samples))
    for s in range(0, ids.numel() - ctx + 1, step):
        chunks.append(ids[s:s + ctx])
        if len(chunks) >= n_samples:
            break
    return chunks if chunks else [ids[:ctx]]


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def cos_for(model, chunks, refs, method, full_layers, seed):
    originals = apply_method(model, method, full_layers) if method else {}
    try:
        vals = []
        for ids, ref in zip(chunks, refs):
            set_seed(seed)
            vals.append(fidelity_metrics(get_logits(model, ids), ref)["logit_cosine"])
    finally:
        if method:
            restore(model, originals, full_layers)
    return mean(vals)


def run(model_key, ctx, n_samples, dtype, seed, load_in_4bit, domains):
    model_id = MODELS[model_key]
    info = inspect_model(model_id)
    model, tok = load_model(model_id, dtype=dtype, load_in_4bit=load_in_4bit)
    device = next(model.parameters()).device
    full_layers = info.full_attention_layers
    budget = 64 + 4 + 32
    methods = [("local_global_topk", AttentionMethod(name="local_global_topk", window=64,
                                                     n_global=4, topk=32)),
               ("streamingllm_bm", AttentionMethod(name="local_global", window=budget - 4,
                                                   n_global=4)),
               ("uniform_random_m", AttentionMethod(name="uniform_random", topk=budget,
                                                    random_k=0))]
    for domain in domains:
        try:
            ids = ids_for_domain(tok, domain, ctx * (n_samples + 1))
            chunks = [c.unsqueeze(0).to(device) for c in build_chunks(ids, ctx, n_samples)]
            refs = []
            for c in chunks:
                set_seed(seed)
                refs.append(get_logits(model, c))
            for tag, method in methods:
                cs = cos_for(model, chunks, refs, method, full_layers, seed)
                rec = {"experiment": "domains", "domain": domain, "method": tag, "model": model_id,
                       "context": ctx, "n_samples": len(chunks), "cos_mean": round(cs, 4)}
                append_jsonl(RESULTS, rec)
                print(f"  [{domain}] {tag:18s} cos={cs:.4f}")
        except Exception as e:
            print(f"  [{domain}] FAILED: {e}")
            append_jsonl(RESULTS, {"experiment": "domains", "domain": domain, "error": str(e)[:200]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-4b-base", choices=list(MODELS.keys()))
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--domains", nargs="+", default=["code", "math"])
    ap.add_argument("--no-quantize", action="store_true")
    args = ap.parse_args()
    run(args.model, args.ctx, args.n_samples, args.dtype, args.seed,
        load_in_4bit=not args.no_quantize, domains=args.domains)


if __name__ == "__main__":
    main()
