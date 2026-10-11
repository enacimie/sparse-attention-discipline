"""Microbenchmark: FlexAttention vs SDPA for the local+sinks(+topk) block-sparse pattern.

Goal: check whether a true block-sparse kernel realizes actual latency speedup over dense SDPA,
across context lengths and head_dim. Self-contained (random QKV), no model loading.
"""
import json
import time

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask


def bench(fn, iters=30, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def run_config(seq, heads, head_dim, window, n_global, topk_blocks, block_size, dtype):
    device = "cuda"
    torch.manual_seed(0)
    q = torch.randn(1, heads, seq, head_dim, device=device, dtype=dtype)
    k = torch.randn(1, heads, seq, head_dim, device=device, dtype=dtype)
    v = torch.randn(1, heads, seq, head_dim, device=device, dtype=dtype)

    def dense():
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    # backbone: causal + local window + sinks
    def mask_mod(b, h, qi, kvi):
        return (qi >= kvi) & (((qi - kvi) < window) | (kvi < n_global))

    block_mask = create_block_mask(
        mask_mod, B=None, H=None, Q_LEN=seq, KV_LEN=seq, device=device,
        _compile=True,
    )
    flex_c = torch.compile(flex_attention, dynamic=False)

    def sparse():
        return flex_c(q, k, v, block_mask=block_mask)

    out = {"seq": seq, "head_dim": head_dim, "window": window, "n_global": n_global,
           "block_size": block_size}
    try:
        out["dense_ms"] = round(bench(dense), 3)
    except Exception as e:
        out["dense_ms"] = None
        out["dense_err"] = str(e)[:120]
    try:
        out["flex_ms"] = round(bench(sparse), 3)
    except Exception as e:
        out["flex_ms"] = None
        out["flex_err"] = str(e)[:120]
    if out.get("dense_ms") and out.get("flex_ms"):
        out["speedup"] = round(out["dense_ms"] / out["flex_ms"], 3)
    # sparsity of the block mask (fraction of active blocks)
    try:
        kv_num_blocks = block_mask.kv_num_blocks
        total_blocks = kv_num_blocks.numel()
        active = int((kv_num_blocks >= 0).sum().item())
        out["active_block_frac"] = round(active / max(total_blocks, 1), 4)
    except Exception:
        pass
    return out


def main():
    torch.set_float32_matmul_precision("high")
    results = []
    heads = 16
    window, n_global = 64, 4
    for head_dim in [128, 256]:
        for seq in [1024, 2048, 4096, 8192]:
            try:
                r = run_config(seq, heads, head_dim, window, n_global,
                               topk_blocks=0, block_size=128, dtype=torch.bfloat16)
            except Exception as e:
                r = {"seq": seq, "head_dim": head_dim, "error": str(e)[:200]}
            results.append(r)
            print(json.dumps(r), flush=True)
    with open("flex_bench.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
