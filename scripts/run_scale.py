"""SCALE validation of the winning method `local_global_topk` on a current-family model (Kaggle/cloud).

Goal: confirm at scale the two local findings that define the contribution:
  1. `local_global_topk` keeps high fidelity at a low pair_ratio on a larger model.
  2. The random-control result holds: content selection must BEAT a uniform-random mask at
     the same pair budget, otherwise it adds nothing.

Model: Qwen3.5-9B-Base (primary; hybrid, 8 full-attn layers, sits at Q2model's >=9B
sensitivity-inversion threshold) or Qwen3.5-4B-Base (safer on memory). Both are current-family
HYBRIDS, so this tests the hybrid-specific claim at scale (only the 8 full_attention layers are
intercepted). Designed for a Kaggle T4/P100 (16GB): loads the model in 4-bit and measures
fidelity (logit cosine / KL), pair_ratio and peak VRAM. Metrics kept separate, never fused.

Run on Kaggle after uploading the repo (or git clone) so `src/` is importable:
  python scripts/run_scale.py --model qwen3.5-9b-base --contexts 1024 2048

NOTE: verify the interceptor resolves Qwen3.5 attention (q_norm/k_norm, gated q_proj, o_proj)
before the full sweep — it is the same family as the locally-validated 0.8B/2B.
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
from utils import GpuMeter, append_jsonl, fidelity_metrics, set_seed  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "scale.jsonl")


@torch.no_grad()
def get_logits(model, ids):
    # .cpu() BEFORE .float(): avoid allocating the fp32 logits tensor on GPU
    # ([ctx, vocab] is several GB at long context) — move fp16 to CPU first.
    return model(ids).logits[0].cpu().float()


def build_chunks(tok, ctx, n_samples, split="validation"):
    """Real wikitext chunks of EXACTLY ctx tokens (concatenate then chunk), so the scale
    fidelity is measured on natural text, not a repetitive low-entropy prompt."""
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


def build_methods(topk_budget):
    """Method configs. uniform_random budgets bracket local_global_topk's pair count so the
    fidelity-vs-pair_ratio comparison is fair."""
    return [
        ("local_global_topk", AttentionMethod(name="local_global_topk", window=64,
                                              n_global=4, topk=32)),
        ("local_window", AttentionMethod(name="local_window", window=64)),
        ("uniform_random_m", AttentionMethod(name="uniform_random", topk=topk_budget,
                                             random_k=0)),
        ("uniform_random_s", AttentionMethod(name="uniform_random", topk=topk_budget // 2,
                                             random_k=0)),
    ]


def run(model_key, contexts, dtype, seed, topk_budget, n_samples, load_in_4bit=True,
        device_map="cuda"):
    model_id = MODELS[model_key]
    info = inspect_model(model_id)
    model, tok = load_model(model_id, dtype=dtype, load_in_4bit=load_in_4bit,
                            device_map=device_map)
    device = next(model.parameters()).device
    full_layers = info.full_attention_layers
    print(f"[load] {model_id} n_layers={info.num_hidden_layers} "
          f"full_attn={len(full_layers)} (dense={len(full_layers)==info.num_hidden_layers})")

    for ctx in contexts:
        try:
            run_context(model, tok, info, full_layers, device, ctx, seed, topk_budget,
                        n_samples, model_id)
        except Exception as e:
            if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower():
                print(f"  ctx={ctx} OOM skipped: {str(e)[:80]}")
                append_jsonl(RESULTS, {"experiment": "scale", "method": "OOM",
                                       "model": model_id, "context": ctx,
                                       "error": str(e)[:200]})
                torch.cuda.empty_cache()
            else:
                raise


def run_context(model, tok, info, full_layers, device, ctx, seed, topk_budget, n_samples,
                model_id):
    chunks = [c.unsqueeze(0).to(device) for c in build_chunks(tok, ctx, n_samples)]
    L = chunks[0].shape[1]
    dense_pairs = (L * (L + 1) // 2) * info.n_full_attention

    refs, dense_vram = [], 0.0
    for ids in chunks:
        set_seed(seed)
        with GpuMeter() as dm:
            refs.append(get_logits(model, ids))
        dense_vram = max(dense_vram, dm.peak_vram_mb)
    append_jsonl(RESULTS, {"experiment": "scale", "method": "dense", "model": model_id,
                           "context": ctx, "pair_ratio": 1.0, "logit_cosine": 1.0,
                           "kl_div": 0.0, "peak_vram_mb": dense_vram,
                           "n_samples": len(chunks)})
    print(f"  ctx={ctx} DENSE vram={dense_vram:.0f}MB (n={len(chunks)})")

    for name, method in build_methods(topk_budget):
        originals = apply_method(model, method, full_layers)
        try:
            cos_l, kl_l, ratios, vram = [], [], [], 0.0
            for ids, ref in zip(chunks, refs):
                set_seed(seed)
                with GpuMeter() as m:
                    fid = fidelity_metrics(get_logits(model, ids), ref)
                reset_pair_counter()
                with torch.no_grad():
                    get_logits(model, ids)
                ratios.append(PAIR_COUNTER["computed"] / dense_pairs)
                cos_l.append(fid["logit_cosine"]); kl_l.append(fid["kl_div"])
                vram = max(vram, m.peak_vram_mb)
        finally:
            restore(model, originals, full_layers)
        cos = sum(cos_l) / len(cos_l); kl = sum(kl_l) / len(kl_l)
        ratio = sum(ratios) / len(ratios)
        rec = {"experiment": "scale", "method": name, "model": model_id, "context": ctx,
               "pair_ratio": round(ratio, 4), "logit_cosine": round(cos, 4),
               "kl_div": round(kl, 4), "peak_vram_mb": vram, "n_samples": len(chunks),
               "params": str({k: v for k, v in vars(method).items() if k != "name"})}
        append_jsonl(RESULTS, rec)
        print(f"  ctx={ctx} {name:18s} pair_ratio={ratio:.4f} cos={cos:.4f} "
              f"kl={kl:.4f} vram={vram:.0f}MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-9b-base", choices=list(MODELS.keys()))
    ap.add_argument("--contexts", type=int, nargs="+", default=[1024, 2048])
    ap.add_argument("--topk-budget", type=int, default=96,
                    help="uniform_random budget to bracket local_global_topk's pair count")
    ap.add_argument("--n-samples", type=int, default=2, help="real wikitext chunks per context")
    ap.add_argument("--dtype", default="float16", help="compute dtype for 4-bit")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-quantize", action="store_true",
                    help="load full precision (local smoke-test on small models)")
    ap.add_argument("--device-map", default="cuda",
                    help="'cuda' (single GPU) or 'auto' (spread across all GPUs)")
    args = ap.parse_args()
    run(args.model, args.contexts, args.dtype, args.seed, args.topk_budget,
        args.n_samples, load_in_4bit=not args.no_quantize, device_map=args.device_map)


if __name__ == "__main__":
    main()
