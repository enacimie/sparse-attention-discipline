"""FlexAttention model-level validation: REAL speedup + fidelity on real wikitext.

Same fidelity metric as run_scale.py (logit cosine vs dense), but the sparse methods are applied
through flex_interceptor (flex_attention), which skips fully-masked blocks -> real latency savings.
We additionally measure wall-clock forward latency to report measured speedup (dense vs sparse).

Methods compared at matched budget:
  - local_global_topk (window=64, n_global=4, topk=32)
  - uniform_random_m  (budget = topk_budget)  [the control]

Usage (remote host / Kaggle):
  python scripts/run_flex_model.py --model qwen3.5-4b-base --contexts 1024 2048 --block-size 128
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from attention_interceptor import reset_pair_counter, restore, PAIR_COUNTER  # noqa: E402
from attention_methods import AttentionMethod  # noqa: E402
from flex_interceptor import apply_method_flex  # noqa: E402
from model_loader import MODELS, inspect_model, load_model  # noqa: E402
from utils import GpuMeter, append_jsonl, fidelity_metrics, set_seed  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "flex_model.jsonl")


@torch.no_grad()
def get_logits(model, ids):
    return model(ids).logits[0].cpu().float()


def timed_logits(model, ids):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = get_logits(model, ids)
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def build_chunks(tok, ctx, n_samples, split="validation"):
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


def build_methods(topk_budget, mode="perkey", method_block=64, B_top=1):
    if mode == "block":
        return [
            ("lg_topk_block", AttentionMethod(name="lg_topk_block", window=method_block,
                                              n_global=4, topk=B_top)),
            ("lgr_block", AttentionMethod(name="lgr_block", window=method_block,
                                          n_global=4, topk=B_top)),
        ]
    return [
        ("local_global_topk", AttentionMethod(name="local_global_topk", window=64,
                                              n_global=4, topk=32)),
        ("uniform_random_m", AttentionMethod(name="uniform_random", topk=topk_budget,
                                             random_k=0)),
    ]


def run(model_key, contexts, dtype, seed, topk_budget, n_samples, block_size,
        load_in_4bit=True, device_map="cuda", mode="perkey", method_block=64, B_top=1):
    model_id = MODELS[model_key]
    info = inspect_model(model_id)
    model, tok = load_model(model_id, dtype=dtype, load_in_4bit=load_in_4bit,
                            device_map=device_map)
    device = next(model.parameters()).device
    full_layers = info.full_attention_layers
    flex_block = method_block if mode == "block" else block_size
    print(f"[load] {model_id} n_layers={info.num_hidden_layers} "
          f"full_attn={len(full_layers)} mode={mode} flex_block={flex_block}")

    for ctx in contexts:
        try:
            run_context(model, tok, info, full_layers, device, ctx, seed, topk_budget,
                        n_samples, model_id, flex_block, mode, method_block, B_top)
        except Exception as e:
            if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower():
                print(f"  ctx={ctx} OOM skipped: {str(e)[:80]}")
                append_jsonl(RESULTS, {"experiment": "flex_model", "method": "OOM",
                                       "model": model_id, "context": ctx,
                                       "error": str(e)[:200]})
                torch.cuda.empty_cache()
            else:
                raise


def run_context(model, tok, info, full_layers, device, ctx, seed, topk_budget, n_samples,
                model_id, block_size, mode, method_block, B_top):
    chunks = [c.unsqueeze(0).to(device) for c in build_chunks(tok, ctx, n_samples)]
    L = chunks[0].shape[1]
    dense_pairs = (L * (L + 1) // 2) * info.n_full_attention

    # warmup dense path (first-call init) so latency is comparable to the warmed sparse path
    set_seed(seed)
    with torch.no_grad():
        get_logits(model, chunks[0])
    # dense reference + latency
    refs, dense_times, dense_vram = [], [], 0.0
    for ids in chunks:
        set_seed(seed)
        with GpuMeter() as dm:
            ref, dt = timed_logits(model, ids)
        refs.append(ref)
        dense_times.append(dt)
        dense_vram = max(dense_vram, dm.peak_vram_mb)
    dense_ms = 1000 * sum(dense_times) / len(dense_times)
    append_jsonl(RESULTS, {"experiment": "flex_model", "method": "dense", "model": model_id,
                           "context": ctx, "pair_ratio": 1.0, "logit_cosine": 1.0, "kl_div": 0.0,
                           "latency_ms": round(dense_ms, 1), "speedup": 1.0,
                           "peak_vram_mb": dense_vram, "block_size": block_size,
                           "n_samples": len(chunks)})
    print(f"  ctx={ctx} DENSE {dense_ms:.1f}ms vram={dense_vram:.0f}MB (n={len(chunks)})")

    for name, method in build_methods(topk_budget, mode, method_block, B_top):
        originals = apply_method_flex(model, method, full_layers, block_size=block_size)
        try:
            # warmup (triggers flex compile) - not timed
            set_seed(seed)
            with torch.no_grad():
                get_logits(model, chunks[0])
            cos_l, kl_l, ratios, times, vram = [], [], [], [], 0.0
            for ids, ref in zip(chunks, refs):
                set_seed(seed)
                with GpuMeter() as m:
                    logits, st = timed_logits(model, ids)
                    fid = fidelity_metrics(logits, ref)
                reset_pair_counter()
                with torch.no_grad():
                    get_logits(model, ids)
                ratios.append(PAIR_COUNTER["computed"] / dense_pairs)
                cos_l.append(fid["logit_cosine"]); kl_l.append(fid["kl_div"])
                times.append(st)
                vram = max(vram, m.peak_vram_mb)
        finally:
            restore(model, originals, full_layers)
        cos = sum(cos_l) / len(cos_l); kl = sum(kl_l) / len(kl_l)
        ratio = sum(ratios) / len(ratios)
        sparse_ms = 1000 * sum(times) / len(times)
        speedup = dense_ms / sparse_ms if sparse_ms > 0 else float("nan")
        rec = {"experiment": "flex_model", "method": name, "model": model_id, "context": ctx,
               "pair_ratio": round(ratio, 4), "logit_cosine": round(cos, 4),
               "kl_div": round(kl, 4), "latency_ms": round(sparse_ms, 1),
               "speedup": round(speedup, 3), "peak_vram_mb": vram, "block_size": block_size,
               "n_samples": len(chunks),
               "params": str({k: v for k, v in vars(method).items() if k != "name"})}
        append_jsonl(RESULTS, rec)
        print(f"  ctx={ctx} {name:18s} pr={ratio:.4f} cos={cos:.4f} kl={kl:.4f} "
              f"{sparse_ms:.1f}ms speedup={speedup:.2f}x vram={vram:.0f}MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-4b-base", choices=list(MODELS.keys()))
    ap.add_argument("--contexts", type=int, nargs="+", default=[1024, 2048])
    ap.add_argument("--topk-budget", type=int, default=96)
    ap.add_argument("--n-samples", type=int, default=2)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--mode", default="perkey", choices=["perkey", "block"])
    ap.add_argument("--method-block", type=int, default=64, help="block size for block mode")
    ap.add_argument("--B-top", type=int, default=1, help="top key-blocks per query-block")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-quantize", action="store_true")
    ap.add_argument("--device-map", default="cuda")
    args = ap.parse_args()
    run(args.model, args.contexts, args.dtype, args.seed, args.topk_budget, args.n_samples,
        args.block_size, load_in_4bit=not args.no_quantize, device_map=args.device_map,
        mode=args.mode, method_block=args.method_block, B_top=args.B_top)


if __name__ == "__main__":
    main()
