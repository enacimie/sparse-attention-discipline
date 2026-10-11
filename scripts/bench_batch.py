"""Eficiencia Qwen3.8-27B Q4: KV cache vs contexto + KV/VRAM vs batch (dense vs evicted keep=128)."""
import sys, os, torch
sys.path.insert(0, "/root/mlattention/src")
from model_loader import inspect_model, load_model
from kv_eviction import KNormEviction, evict_cache, cache_kv_bytes
from utils import append_jsonl

MODEL_PATH = "/root/qwen38"
OUT = "/root/mlattention/results/qwen38_batch.jsonl"

def main():
    model, tok = load_model(MODEL_PATH, load_in_4bit=True, attn_impl="eager")
    info = inspect_model(MODEL_PATH)
    dev = next(model.parameters()).device
    print(f"[load] eager 4-bit, full_attn={len(info.full_attention_layers)}", flush=True)

    filler = tok("The grass is green and the sky is blue today. ", return_tensors="pt").input_ids[0]

    def seq(ctx, batch=1):
        n = (ctx + filler.numel() - 1) // filler.numel()
        ids = filler.repeat(n)[:ctx].unsqueeze(0)
        return ids.repeat(batch, 1).to(dev)

    # 1. KV cache bytes vs contexto (batch 1)
    print("[kv_vs_ctx]", flush=True)
    for ctx in [512, 1024, 2048, 4096, 8192]:
        ids = seq(ctx)
        with torch.no_grad():
            out = model(ids, use_cache=True)
            dense_kv = cache_kv_bytes(out.past_key_values) / 2**20
            evict_cache(out.past_key_values, KNormEviction(keep=128, n_sink=4))
            ev_kv = cache_kv_bytes(out.past_key_values) / 2**20
        append_jsonl(OUT, {"metric": "kv_vs_ctx", "ctx": ctx, "dense_mb": round(dense_kv, 1), "h2o_mb": round(ev_kv, 1)})
        print(f"  ctx={ctx}: dense={dense_kv:.1f}MB h2o={ev_kv:.1f}MB ({(dense_kv/ev_kv):.0f}x)", flush=True)
        del out; torch.cuda.empty_cache()

    # 2. KV + VRAM vs batch (ctx 2048)
    print("[kv_vram_vs_batch] ctx=2048", flush=True)
    for b in [1, 2, 4, 8, 16, 32]:
        ids = seq(2048, b)
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            out = model(ids, use_cache=True)
            dense_peak = torch.cuda.max_memory_allocated() / 2**30
            dense_kv = cache_kv_bytes(out.past_key_values) / 2**20
            evict_cache(out.past_key_values, KNormEviction(keep=128, n_sink=4))
            ev_kv = cache_kv_bytes(out.past_key_values) / 2**20
        append_jsonl(OUT, {"metric": "vram_vs_batch", "batch": b, "dense_peak_gb": round(dense_peak, 2),
                           "dense_kv_mb": round(dense_kv, 1), "h2o_kv_mb": round(ev_kv, 1)})
        print(f"  b={b}: peak={dense_peak:.2f}GB dense_kv={dense_kv:.1f}MB h2o_kv={ev_kv:.1f}MB", flush=True)
        del out; torch.cuda.empty_cache()

    print("[done]", flush=True)

if __name__ == "__main__":
    main()
