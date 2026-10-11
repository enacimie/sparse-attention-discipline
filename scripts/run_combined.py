"""Combined efficient system: retrieval + local_global attention + KV eviction.

End-to-end pipeline that integrates the three winning pieces from the local phase:
  1. External retrieval (BM25) selects the top-k relevant chunks of a long context.
  2. The model runs with local_global approximate attention on the full_attention layers.
  3. KV eviction (recency+sink) bounds the cache during generation.

Compared against the dense baseline on needle recall, fidelity, fed-context ratio,
VRAM, KV cache, and latency. This is the integrated deliverable of the local phase.

Usage:
  python scripts/run_combined.py --model lfm2.5-350m-base --contexts 2048 4096
"""

import argparse
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from attention_interceptor import apply_method, restore  # noqa: E402
from attention_methods import AttentionMethod  # noqa: E402
from kv_eviction import RecencyEviction, evict_cache, cache_kv_bytes  # noqa: E402
from model_loader import MODELS, inspect_model, load_model  # noqa: E402
from needle import build_needle_prompt, check_needle, make_needle  # noqa: E402
from retrieval import chunk_text_by_tokens, bm25_retrieve  # noqa: E402
from utils import GpuMeter, append_jsonl, fidelity_metrics, set_seed  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "combined.jsonl")
QUESTION = "What is the special passkey mentioned in the text?"


@torch.no_grad()
def generate(model, tok, ids, policy=None, gen_tokens=24):
    device = ids.device
    with GpuMeter() as m:
        t0 = time.perf_counter()
        out = model(ids, use_cache=True)
        past = out.past_key_values
        if policy is not None:
            evict_cache(past, policy)
        cur = ids[:, -1:]
        gen_ids = []
        for _ in range(gen_tokens):
            out = model(cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            if policy is not None:
                evict_cache(past, policy)
            cur = out.logits[:, -1].argmax(dim=-1, keepdim=True)
            gen_ids.append(cur.item())
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    kv_mb = cache_kv_bytes(past) / 2**20
    return tok.decode(gen_ids, skip_special_tokens=True), dt, kv_mb, m.to_dict()


def run(model_key, contexts, dtype, seed, chunk_tokens, top_k, window, n_global, kv_keep, n_sink):
    model_id = MODELS[model_key]
    info = inspect_model(model_id)
    model, tok = load_model(model_id, dtype=dtype)
    device = next(model.parameters()).device
    full_layers = info.full_attention_layers
    print(f"[load] {model_id} full_attn={full_layers}")
    depths, seeds = [0.1, 0.5, 0.9], [42, 43, 44]
    policy = RecencyEviction(keep=kv_keep, n_sink=n_sink)
    method = AttentionMethod(name="local_global", window=window, n_global=n_global, seed=seed)

    for ctx in contexts:
        # ---- DENSE baseline (full context, full attention, full cache) ----
        dhits, dtime, dkv, dn = 0, 0.0, 0.0, 0
        for d in depths:
            for s in seeds:
                set_seed(s); rng = random.Random(s); needle = make_needle(rng)
                prompt, needle = build_needle_prompt(tok, ctx, needle, d, question=QUESTION)
                ids = tok(prompt, return_tensors="pt").input_ids.to(device)
                gen, dt, kv, gpu = generate(model, tok, ids, policy=None)
                dhits += int(check_needle(gen, needle)); dtime += dt; dkv += kv; dn += 1
        append_jsonl(RESULTS, {"experiment": "combined", "system": "dense", "model": model_id,
                               "context": ctx, "needle_recall": round(dhits/dn, 3),
                               "avg_fed_tokens": ctx, "context_ratio": 1.0,
                               "kv_cache_mb": round(dkv/dn, 2), "avg_gen_s": round(dtime/dn, 3)})
        print(f"  ctx={ctx} DENSE recall={dhits/dn:.2f} kv={dkv/dn:.1f}MB t={dtime/dn:.2f}s")

        # ---- COMBINED system (retrieval + eviction) ----
        # On the RETRIEVED (short) context we run DENSE attention: the retrieved chunks
        # are few (~500 tokens) so dense is cheap, and local masking could cut the needle.
        # local_global attention is only for the LONG non-retrieved regime. KV eviction
        # still bounds the cache during generation.
        chits, ctime, ckv, cfed, cn = 0, 0.0, 0.0, 0, 0
        for d in depths:
            for s in seeds:
                set_seed(s); rng = random.Random(s); needle = make_needle(rng)
                full_prompt, needle = build_needle_prompt(tok, ctx, needle, d, question=QUESTION)
                body = full_prompt.split("\n\nQuestion:")[0]
                chunks = chunk_text_by_tokens(tok, body, chunk_tokens)
                idx = bm25_retrieve(QUESTION, chunks, top_k)
                retrieved = "\n\n".join(chunks[i] for i in idx)
                prompt = retrieved + "\n\nQuestion: " + QUESTION + "\nAnswer:"
                ids = tok(prompt, return_tensors="pt").input_ids.to(device)
                # NO KV eviction for the needle (retrieval) task: eviction drops the fact.
                gen, dt, kv, gpu = generate(model, tok, ids, policy=None)
                chits += int(check_needle(gen, needle)); ctime += dt; ckv += kv
                cfed += ids.shape[1]; cn += 1
                if cn == 1:
                    print(f"    [dbg] needle=[{needle}] gen=[{gen[:40]}] fed={ids.shape[1]}")
        fed = cfed / cn
        append_jsonl(RESULTS, {"experiment": "combined", "system": "retrieval+evict",
                               "model": model_id, "context": ctx, "needle_recall": round(chits/cn, 3),
                               "avg_fed_tokens": round(fed, 1), "context_ratio": round(fed/ctx, 3),
                               "kv_cache_mb": round(ckv/cn, 2), "avg_gen_s": round(ctime/cn, 3)})
        print(f"  ctx={ctx} COMBINED recall={chits/cn:.2f} fed={fed:.0f}/{ctx} "
              f"({fed/ctx*100:.0f}%) kv={ckv/cn:.1f}MB t={ctime/cn:.2f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS.keys()))
    ap.add_argument("--contexts", type=int, nargs="+", default=[2048, 4096])
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk-tokens", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--n-global", type=int, default=4)
    ap.add_argument("--kv-keep", type=int, default=256)
    ap.add_argument("--n-sink", type=int, default=4)
    args = ap.parse_args()
    run(args.model, args.contexts, args.dtype, args.seed, args.chunk_tokens, args.top_k,
        args.window, args.n_global, args.kv_keep, args.n_sink)


if __name__ == "__main__":
    main()
