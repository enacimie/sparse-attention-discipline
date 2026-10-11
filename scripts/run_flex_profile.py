"""Profile the FlexAttention pipeline phases to find the overhead bottleneck."""
import time
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask


def t(fn, n=20, w=5):
    for _ in range(w):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def build_keep(seq, BLOCK, dev):
    n_kb = seq // BLOCK
    qb = torch.arange(seq, device=dev) // BLOCK
    kb = torch.arange(seq, device=dev) // BLOCK
    qbi = torch.arange(n_kb, device=dev).view(-1, 1)
    kbi = torch.arange(n_kb, device=dev).view(1, -1)
    keep_b = torch.eye(n_kb, dtype=torch.bool, device=dev)
    keep_b[:, 0] = True
    strict = kbi < qbi
    cand = torch.rand(n_kb, n_kb, device=dev).masked_fill(~strict, float("-inf"))
    _, topb = cand.topk(1, dim=-1)
    keep_b.scatter_(1, topb, True)
    keep_b &= (kbi <= qbi)
    return keep_b[qb][:, kb]


def main():
    dev = "cuda"
    heads, hd, BLOCK = 16, 256, 64
    flex_c = torch.compile(flex_attention, dynamic=False)
    for seq in [2048, 4096]:
        torch.manual_seed(0)
        q = torch.randn(1, heads, seq, hd, device=dev, dtype=torch.bfloat16)
        k = torch.randn(1, heads, seq, hd, device=dev, dtype=torch.bfloat16)
        v = torch.randn(1, heads, seq, hd, device=dev, dtype=torch.bfloat16)
        keep = build_keep(seq, BLOCK, dev)

        def mk():
            def mask_mod(b, h, qi, kvi):
                return keep[qi, kvi]
            return create_block_mask(mask_mod, B=1, H=heads, Q_LEN=seq, KV_LEN=seq,
                                     BLOCK_SIZE=BLOCK, device=dev)
        bm = mk()
        cbm = t(mk, n=10, w=2)
        fl = t(lambda: flex_c(q, k, v, block_mask=bm))
        dn = t(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
        print(f"seq={seq} create_block_mask={cbm:.2f}ms flex={fl:.2f}ms dense={dn:.2f}ms",
              flush=True)


if __name__ == "__main__":
    main()
