"""Cost of the content top-k SCORING vs a fixed window (reviewer Q5).

local_global_topk must score q.k over all keys before selecting, while a pure sliding window
(local_window / StreamingLLM) needs no scoring. We time the mask construction (the scoring step)
against the attention it enables, to check whether the low pair_ratio is bought at the price of an
offsetting scoring pass. We use head-mean q/k for scoring (as build_content_mask does), so the
scoring is a single (q_len x k_len x d) matmul, i.e. 1/n_heads of the full QK^T cost.

Usage:
  python scripts/run_overhead.py --ctx 2048 --heads 16 --kv 8 --head-dim 128
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from attention_methods import AttentionMethod, build_attention_mask, build_content_mask  # noqa: E402


def timeit(fn, iters=20, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000  # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--kv", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--n-global", type=int, default=4)
    args = ap.parse_args()

    dev = "cuda"
    L, H, KV, D = args.ctx, args.heads, args.kv, args.head_dim
    torch.manual_seed(0)
    q = torch.randn(1, H, L, D, device=dev, dtype=torch.float16)
    k = torch.randn(1, KV, L, D, device=dev, dtype=torch.float16)
    v = torch.randn(1, KV, L, D, device=dev, dtype=torch.float16)

    method_topk = AttentionMethod(name="local_global_topk", window=args.window,
                                  n_global=args.n_global, topk=args.topk)
    method_win = AttentionMethod(name="local_window", window=args.window)

    # 1) dense attention (no mask): the baseline cost
    k_rep = k.repeat_interleave(H // KV, dim=1)
    v_rep = v.repeat_interleave(H // KV, dim=1)
    t_dense = timeit(lambda: F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True))

    # 2) mask construction: content top-k (has scoring) vs pure window (no scoring)
    t_mask_content = timeit(lambda: build_content_mask(method_topk, q, k, causal=True))
    t_mask_window = timeit(lambda: build_attention_mask(method_win, L, dev, causal=True))
    scoring_overhead = t_mask_content - t_mask_window

    # 3) sparse attention over the selected pairs (SDPA with additive mask)
    keep = build_content_mask(method_topk, q, k, causal=True)
    additive = torch.zeros(L, L, device=dev, dtype=torch.float16)
    additive.masked_fill_(~keep, torch.finfo(torch.float16).min)
    am = additive[None, None]
    t_sparse = timeit(lambda: F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=am))

    pair_ratio = keep.float().mean().item()
    print(f"ctx={L} heads={H} kv={KV} d={D}")
    print(f"  dense_attn          {t_dense:8.3f} ms")
    print(f"  mask content-topk   {t_mask_content:8.3f} ms  (includes scoring)")
    print(f"  mask fixed-window   {t_mask_window:8.3f} ms  (no scoring)")
    print(f"  scoring overhead    {scoring_overhead:8.3f} ms  "
          f"({scoring_overhead / t_dense * 100:.1f}% of dense attn)")
    print(f"  sparse_attn(masked) {t_sparse:8.3f} ms")
    print(f"  pair_ratio          {pair_ratio:8.4f}")
    total_sparse = t_mask_content + t_sparse
    print(f"  content path total  {total_sparse:8.3f} ms vs dense {t_dense:.3f} ms "
          f"-> {t_dense / total_sparse:.2f}x")


if __name__ == "__main__":
    main()
