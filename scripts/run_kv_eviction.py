"""Experiment 15: KV cache eviction during generation.

Compares dense full-cache generation vs bounded-cache eviction policies, measuring
needle recall on long generation, peak VRAM, KV cache memory, and tokens/s.

Usage:
  python scripts/run_kv_eviction.py --model lfm2.5-350m-base --contexts 2048 \
      --keep 128 256 --n-sink 4
"""

import argparse
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model_loader import MODELS, inspect_model, load_model  # noqa: E402
from kv_eviction import RecencyEviction, KNormEviction, evict_cache, cache_kv_bytes  # noqa: E402
from needle import build_needle_prompt, check_needle, make_needle  # noqa: E402
from utils import GpuMeter, append_jsonl, set_seed  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "kv_eviction.jsonl")


@torch.no_grad()
def generate_with_policy(model, tok, ctx, depth, seed, policy, gen_tokens=24):
    set_seed(seed)
    rng = random.Random(seed)
    needle = make_needle(rng)
    prompt, needle = build_needle_prompt(tok, ctx, needle, depth)
    ids = tok(prompt, return_tensors="pt").input_ids.to(next(model.parameters()).device)
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
    gen = tok.decode(gen_ids, skip_special_tokens=True)
    hit = check_needle(gen, needle)
    return hit, gen, dt, kv_mb, m.to_dict()


def run(model_key, contexts, keeps, n_sink, dtype, gen_tokens, policies):
    model_id = MODELS[model_key]
    info = inspect_model(model_id)
    model, tok = load_model(model_id, dtype=dtype)
    print(f"[load] {model_id} full_attn={info.full_attention_layers}")
    depths, seeds = [0.1, 0.5, 0.9], [42, 43, 44]

    for ctx in contexts:
        # dense (no eviction)
        hits, tot_time, tot_kv, tot = 0, 0.0, 0.0, 0
        for d in depths:
            for s in seeds:
                hit, gen, dt, kv, gpu = generate_with_policy(model, tok, ctx, d, s, None, gen_tokens)
                hits += int(hit); tot_time += dt; tot_kv += kv; tot += 1
        rec = {"experiment": "kv_eviction", "policy": "dense", "model": model_id,
               "context": ctx, "keep": -1, "n_sink": 0, "needle_recall": round(hits/tot, 3),
               "kv_cache_mb": round(tot_kv/tot, 2), "avg_gen_s": round(tot_time/tot, 3)}
        append_jsonl(RESULTS, rec)
        print(f"  ctx={ctx} DENSE recall={rec['needle_recall']} kv={rec['kv_cache_mb']}MB")

        for keep in keeps:
            for pol in policies:
                policy = (RecencyEviction(keep=keep, n_sink=n_sink) if pol == "recency"
                          else KNormEviction(keep=keep, n_sink=n_sink))
                hits, tot_time, tot_kv, tot = 0, 0.0, 0.0, 0
                for d in depths:
                    for s in seeds:
                        hit, gen, dt, kv, gpu = generate_with_policy(model, tok, ctx, d, s, policy, gen_tokens)
                        hits += int(hit); tot_time += dt; tot_kv += kv; tot += 1
                rec = {"experiment": "kv_eviction", "policy": f"{pol}+sink", "model": model_id,
                       "context": ctx, "keep": keep, "n_sink": n_sink,
                       "needle_recall": round(hits/tot, 3), "kv_cache_mb": round(tot_kv/tot, 2),
                       "avg_gen_s": round(tot_time/tot, 3)}
                append_jsonl(RESULTS, rec)
                print(f"  ctx={ctx} {pol} keep={keep}+{n_sink}: recall={rec['needle_recall']} "
                      f"kv={rec['kv_cache_mb']}MB gen={rec['avg_gen_s']}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS.keys()))
    ap.add_argument("--contexts", type=int, nargs="+", default=[2048])
    ap.add_argument("--keep", type=int, nargs="+", default=[128, 256])
    ap.add_argument("--n-sink", type=int, default=4)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--gen-tokens", type=int, default=24)
    ap.add_argument("--policies", nargs="+", default=["recency", "knorm"],
                    choices=["recency", "knorm"])
    args = ap.parse_args()
    run(args.model, args.contexts, args.keep, args.n_sink, args.dtype, args.gen_tokens,
        args.policies)


if __name__ == "__main__":
    main()
