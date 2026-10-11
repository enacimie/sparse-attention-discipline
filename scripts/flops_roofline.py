"""WS4a: FLOPs / memory-traffic / roofline breakdown for sparse attention efficiency.
Computes analytical bounds without requiring GPU — uses model config + measured pair_ratios.

Output -> stdout (paper-ready table)
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from model_loader import inspect_model, MODELS


def flops_qk(n_q, n_kv, L, d_h, groups):
    """GQA Q·K^T matmul: q (n_q, L, d) x k (n_kv, L, d) -> group-expanded."""
    return 2 * n_q * L * L * d_h  # n_q heads x L x L x d_h


def flops_pv(n_q, L, d_h, pr=1.0):
    """P·V matmul with pair ratio."""
    return 2 * n_q * pr * L * L * d_h


def flops_softmax(n_q, L):
    return 5 * n_q * L * L


def flops_scoring_avg(L, d_h):
    """Head-mean q·k scoring: one (L,d) x (d,L) matmul."""
    return 2 * L * L * d_h


def flops_projection(L, hidden, n_q, n_kv, d_h, gated=True):
    """Q/K/V/O projection FLOPs (one linear layer each)."""
    q_in, q_out = hidden, n_q * d_h * (2 if gated else 1)
    k_in, k_out = hidden, n_kv * d_h
    v_in, v_out = hidden, n_kv * d_h
    o_in, o_out = n_q * d_h, hidden
    return (
        2 * L * q_in * q_out +
        2 * L * k_in * k_out +
        2 * L * v_in * v_out +
        2 * L * o_in * o_out
    )


def bytes_traffic_component(name, L, dim1, dim2, dtype_bytes=2):
    """Estimated DRAM traffic for a matmul or tensor."""
    if name == 'qkv_read':
        return 3 * L * dim1 * dtype_bytes
    elif name == 'score_mat':
        return L * L * dim1 * dtype_bytes
    elif name == 'attn_mask':
        return L * L * 1  # bool
    return 0


print("=" * 70)
print("WS4a — FLOPs / MEMORY / ROOFLINE BREAKDOWN")
print("=" * 70)

for model_key in ['qwen3.5-0.8b', 'qwen3.5-4b-base']:
    try:
        info = inspect_model(MODELS[model_key])
    except Exception as e:
        print(f"  {model_key}: skip ({e})")
        continue

    n_q = info.num_attention_heads
    n_kv = info.num_key_value_heads
    d_h = info.head_dim
    hidden = info.hidden_size
    n_full = info.n_full_attention
    groups = n_q // n_kv

    print(f"\n{'='*70}")
    print(f"Model: {model_key}  |  q_heads={n_q}  kv_heads={n_kv}  head_dim={d_h}  "
          f"hidden={hidden}  n_full_attn={n_full}")
    print(f"{'='*70}")

    for ctx, pair_ratio, scoring_frac_label in [
        (2048, 0.085, "~8.5%"),
        (4096, 0.047, "~4.7%"),
    ]:
        L = ctx
        pr = pair_ratio

        # --- Projection FLOPs (identical for dense and sparse) ---
        proj_flops = flops_projection(L, hidden, n_q, n_kv, d_h, gated=True)

        # --- Dense attention FLOPs ---
        dense_qk = flops_qk(n_q, n_kv, L, d_h, groups)
        dense_sm = flops_softmax(n_q, L)
        dense_pv = flops_pv(n_q, L, d_h, pr=1.0)
        dense_attn = dense_qk + dense_sm + dense_pv
        dense_total = proj_flops + dense_attn

        # --- Sparse (lg_topk) FLOPs ---
        scoring_flops = flops_scoring_avg(L, d_h)
        sparse_qk = dense_qk  # unfused: still full Q·K^T
        sparse_sm = dense_sm  # unfused: still full softmax
        sparse_pv = flops_pv(n_q, L, d_h, pr=pr)  # P·V could skip zeros
        sparse_attn = sparse_qk + sparse_sm + sparse_pv
        sparse_total = proj_flops + scoring_flops + sparse_attn

        # --- Fused sparse (hypothetical) ---
        fused_qk = flops_qk(n_q, n_kv, L, d_h, groups) * pr
        fused_sm = flops_softmax(n_q, L) * pr
        fused_pv = flops_pv(n_q, L, d_h, pr=pr)
        fused_attn = fused_qk + fused_sm + fused_pv
        fused_total = proj_flops + scoring_flops + fused_attn

        # --- Amdahl: max speedup ---
        attn_frac = dense_attn / dense_total
        max_speedup = 1 / (1 - attn_frac + attn_frac * pr)
        fused_speedup = dense_total / fused_total

        # --- Memory (key difference) ---
        # Dense (fused SDPA): block-tiled, doesn't materialize full score matrix
        # Masked (unfused SDPA): materializes (n_q, L, L) float32 attention weights
        score_bytes = n_q * L * L * 4  # float32 attention weights
        # For the unfused path, the internal score matrix + softmax output = huge traffic

        print(f"\n  ctx={ctx} | pair_ratio={scoring_frac_label} | "
              f"selected_pairs_pr={pr:.3f}")
        print(f"  {'Component':<30s} {'Dense (GFLOPs)':>16s} {'Sparse(unfused)':>16s} {'Fused(hypoth)':>16s}")
        print(f"  {'-'*78}")
        for name, d, s, f in [
            ("Projections (Q/K/V/O)", proj_flops, proj_flops, proj_flops),
            ("Scoring (q.k head-avg)", 0, scoring_flops, scoring_flops),
            ("Q·K^T matmul", dense_qk, sparse_qk, fused_qk),
            ("Softmax", dense_sm, sparse_sm, fused_sm),
            ("P·V matmul", dense_pv, sparse_pv, fused_pv),
        ]:
            print(f"  {name:<30s} {d/1e9:>10.2f} G   {s/1e9:>10.2f} G   {f/1e9:>10.2f} G")
        print(f"  {'-'*78}")
        print(f"  {'TOTAL (per layer)':<30s} {dense_total/1e9:>10.2f} G   "
              f"{sparse_total/1e9:>10.2f} G   {fused_total/1e9:>10.2f} G")

        # Fraction of dense FLOPs
        score_pct = scoring_flops / dense_attn * 100
        unfused_vs_dense = sparse_total / dense_total * 100
        fused_vs_dense = fused_total / dense_total * 100

        print(f"\n  --- Efficiency metrics ---")
        print(f"  Scoring FLOPs / dense attention FLOPs = {score_pct:.1f}%")
        print(f"  Unfused FLOP ratio = {unfused_vs_dense:.1f}% (ECR logic = {pr*100:.1f}%, "
              f"scoring+unfused masks the gap)")
        print(f"  Fused FLOP ratio = {fused_vs_dense:.1f}%")
        print(f"  Amdahl speedup (fused, attn only) = {1/pr:.1f}x")
        print(f"  Amdahl speedup (fused, full layer) = {fused_speedup:.1f}x")
        print(f"  Max theoretical speedup (Amdahl, pr={pr}) = {max_speedup:.1f}x "
              f"(attention is {attn_frac*100:.0f}% of layer)")

        # --- Memory-bottleneck explanation ---
        print(f"\n  --- Why unfused is 3-5x slower despite ~same FLOPs ---")
        print(f"  Materialized score matrix: {score_bytes/1e6:.0f} MB per layer "
              f"(fp32, {n_q} heads x {L} x {L})")
        print(f"  RTX 3070 mem BW ~448 GB/s: score mat alone = {score_bytes/1e6:.0f} MB / "
              f"{448e3:.0f} MB/s = {score_bytes/448e6*1000:.1f} us")
        print(f"  Dense (fused SDPA): block-tiled, no materialized scores, "
              f"compute-bound (~{dense_total/1e9:.1f} GFLOPs)")
        print(f"  Masked (unfused): materializes score matrix, "
              f"memory-bound (bottleneck: {score_bytes/1e6:.0f} MB DRAM writes)")

        print(f"\n  --- Path to speedup ---")
        print(f"  1. Fuse selection + attention into single kernel (Triton/Flash-based)")
        print(f"  2. Expected wall-clock speedup (attn layers): {fused_speedup:.1f}x")
        print(f"  3. Full-model speedup: ~{max_speedup:.1f}x (Amdahl, "
              f"{n_full} attn layers out of {info.num_hidden_layers} total)")
        print(f"  4. Score reuse across multiple forward passes (caching): "
              f"amortize scoring FLOPs ({score_pct:.1f}%) over N forwards")

print("\nDone.")
