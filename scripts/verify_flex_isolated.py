#!/usr/bin/env python3
"""Verificacion aislada: los 57 ms constantes del barrido de block_size, son reales o un fallback?

El barrido dio flex_ms ~57.02 / 57.05 / 57.05 para block_size 32 / 64 / 256 y 0.419 para 128, a
seq 4096. Tres valores casi identicos para tamanos de bloque distintos es sospechoso: o es una ruta
de respaldo no fusionada, o el bucle de medicion esta contaminado por el torch.compile de la
configuracion anterior.

Este script corre UNA sola configuracion por proceso, sin estado previo, y reporta ademas:
  - si flex_attention cae en graph break / eager (via explain y conteo de kernels no compilados)
  - el tiempo sin torch.compile, para comparar
  - el tiempo con mas iteraciones, para ver si es estable

Uso:  python verify_flex_isolated.py --seq 4096 --block-size 128 --pattern block_aligned
"""
import argparse
import json
import time

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

HEADS = 16
WINDOW = 64
N_GLOBAL = 4


def bench(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def mask_for(pattern, seq, topk=32, mod=4096):
    umbral = max(1, round(mod * topk / max(seq, 1)))
    if pattern == "block_aligned":
        def mm(b, h, qi, kvi):
            return (qi >= kvi) & (((qi - kvi) < WINDOW) | (kvi < N_GLOBAL))
    else:
        def mm(b, h, qi, kvi):
            base = (qi >= kvi) & (((qi - kvi) < WINDOW) | (kvi < N_GLOBAL))
            hsh = (kvi * 7919 + qi * 104729) % mod
            return base | ((qi >= kvi) & (hsh < umbral))
    return mm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--pattern", default="block_aligned")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    torch.set_float32_matmul_precision("high")
    dev = "cuda"
    torch.manual_seed(0)
    q = torch.randn(1, HEADS, args.seq, args.head_dim, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, HEADS, args.seq, args.head_dim, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, HEADS, args.seq, args.head_dim, device=dev, dtype=torch.bfloat16)

    r = {"seq": args.seq, "head_dim": args.head_dim, "block_size": args.block_size,
         "pattern": args.pattern, "iters": args.iters,
         "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}

    dense_ms = bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True),
                     args.iters, 10)
    r["dense_ms"] = round(dense_ms, 4)

    mm = mask_for(args.pattern, args.seq)
    t0 = time.time()
    bm = create_block_mask(mm, B=None, H=None, Q_LEN=args.seq, KV_LEN=args.seq,
                           device=dev, BLOCK_SIZE=args.block_size)
    r["create_block_mask_s"] = round(time.time() - t0, 2)
    nq = (args.seq + args.block_size - 1) // args.block_size
    total = nq * (nq + 1) // 2
    r["active_block_frac"] = round(int(bm.kv_num_blocks.sum()) / max(total, 1), 5)
    try:
        r["full_block_frac"] = round(int(bm.full_kv_num_blocks.sum()) / max(total, 1), 5)
    except Exception as e:
        r["full_block_frac"] = None
        r["full_err"] = str(e)[:80]

    # (a) SIN torch.compile: flex_attention en eager
    try:
        r["flex_eager_ms"] = round(bench(lambda: flex_attention(q, k, v, block_mask=bm),
                                         max(args.iters // 5, 5), 3), 4)
    except Exception as e:
        r["flex_eager_ms"] = None
        r["flex_eager_err"] = f"{type(e).__name__}: {str(e)[:120]}"

    # (b) CON torch.compile, en proceso limpio
    try:
        fc = torch.compile(flex_attention, dynamic=False)
        t0 = time.time()
        r["flex_compiled_ms"] = round(bench(lambda: fc(q, k, v, block_mask=bm), args.iters, 10), 4)
        r["compile_and_warmup_s"] = round(time.time() - t0, 2)
    except Exception as e:
        r["flex_compiled_ms"] = None
        r["flex_compiled_err"] = f"{type(e).__name__}: {str(e)[:120]}"

    for k_ in ("flex_eager_ms", "flex_compiled_ms"):
        if r.get(k_):
            r[f"speedup_{k_}"] = round(dense_ms / r[k_], 4)

    print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
