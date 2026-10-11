"""Block-level local_global_topk with FlexAttention: fidelity + real speedup, vs block-random control.

Re-validates the core finding at block granularity (where real speedup lives):
  method  = causal + diagonal block (local) + block0 (sinks) + top-B key-blocks by block-level q.k
  control = same structure but B RANDOM key-blocks (matched budget)
Metrics: attention-output cosine vs dense (fidelity) and latency speedup vs dense SDPA.
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


def cos_sim(a, b):
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    return float(torch.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def build_block_keep(Q, K, seq, heads, BLOCK, B_sel, mode, n_global):
    dev = Q.device
    num_b = seq // BLOCK
    Qb = Q[0].reshape(heads, num_b, BLOCK, -1).mean(dim=2)
    Kb = K[0].reshape(heads, num_b, BLOCK, -1).mean(dim=2)
    scores = torch.bmm(Qb, Kb.transpose(1, 2))
    block_keep = torch.zeros(heads, num_b, num_b, dtype=torch.bool, device=dev)
    diag = torch.eye(num_b, dtype=torch.bool, device=dev)
    block_keep |= diag.unsqueeze(0)
    block_keep[:, :, 0] = True  # sinks
    strict = torch.tril(torch.ones(num_b, num_b, dtype=torch.bool, device=dev), diagonal=-1)
    B_sel = min(B_sel, num_b - 1)
    if B_sel > 0:
        if mode == "topk":
            cand = scores.masked_fill(~strict.unsqueeze(0), float("-inf"))
        else:  # random
            cand = torch.rand(heads, num_b, num_b, device=dev).masked_fill(
                ~strict.unsqueeze(0), float("-inf"))
        _, topi = cand.topk(B_sel, dim=-1)
        block_keep.scatter_(2, topi, True)
    causal_blocks = torch.tril(torch.ones(num_b, num_b, dtype=torch.bool, device=dev))
    block_keep &= causal_blocks.unsqueeze(0)
    return block_keep, num_b


def pair_ratio(block_keep, num_b, BLOCK, seq):
    # causal pairs kept, averaged over heads
    below = torch.tril(torch.ones(num_b, num_b, dtype=torch.bool, device=block_keep.device), diagonal=-1)
    diag = torch.eye(num_b, dtype=torch.bool, device=block_keep.device)
    full_pairs = BLOCK * BLOCK
    diag_pairs = BLOCK * (BLOCK + 1) / 2.0
    n_below = (block_keep & below.unsqueeze(0)).sum(dim=(1, 2)).float()
    n_diag = (block_keep & diag.unsqueeze(0)).sum(dim=(1, 2)).float()
    active = n_below * full_pairs + n_diag * diag_pairs
    total = seq * (seq + 1) / 2.0
    return float((active / total).mean().item())


def run(seq, heads, head_dim, BLOCK, B_sel, n_global, dtype):
    dev = "cuda"
    torch.manual_seed(0)
    q = torch.randn(1, heads, seq, head_dim, device=dev, dtype=dtype)
    k = torch.randn(1, heads, seq, head_dim, device=dev, dtype=dtype)
    v = torch.randn(1, heads, seq, head_dim, device=dev, dtype=dtype)
    dense_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    flex_c = torch.compile(flex_attention, dynamic=False)
    out = {"seq": seq, "head_dim": head_dim, "BLOCK": BLOCK, "B_sel": B_sel}

    def dense():
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out["dense_ms"] = round(bench(dense), 3)

    for mode in ["topk", "random"]:
        block_keep, num_b = build_block_keep(q, k, seq, heads, BLOCK, B_sel, mode, n_global)

        def mask_mod(b, h, qi, kvi, _bk=block_keep):
            return (qi >= kvi) & _bk[h, qi // BLOCK, kvi // BLOCK]

        bm = create_block_mask(mask_mod, B=None, H=heads, Q_LEN=seq, KV_LEN=seq,
                               BLOCK_SIZE=BLOCK, device=dev)

        def sparse():
            return flex_c(q, k, v, block_mask=bm)
        s_out = sparse()
        out[f"cos_{mode}"] = round(cos_sim(s_out, dense_out), 5)
        out[f"ms_{mode}"] = round(bench(sparse), 3)
        out[f"pr_{mode}"] = round(pair_ratio(block_keep, num_b, BLOCK, seq), 4)
    out["speedup_topk"] = round(out["dense_ms"] / out["ms_topk"], 3)
    return out


def main():
    torch.set_float32_matmul_precision("high")
    heads, head_dim, n_global = 16, 256, 4
    results = []
    for BLOCK in [16, 32]:
        for B_sel in [4]:
            for seq in [2048, 4096]:
                try:
                    r = run(seq, heads, head_dim, BLOCK, B_sel, n_global, torch.bfloat16)
                except Exception as e:
                    r = {"seq": seq, "BLOCK": BLOCK, "B_sel": B_sel, "error": str(e)[:200]}
                results.append(r)
                print(json.dumps(r), flush=True)
    with open("flex_full.jsonl", "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
