"""Experiment 14: effective-rank analysis of Q, K and attention matrices.

Determines whether attention is low-rank (and thus whether low-rank / landmark /
Nyström approximations are viable). For each full_attention layer we capture the
post-RoPE Q and K, form the score matrix, and measure the singular-value spectrum:
  - effective rank (participation ratio of singular values)
  - energy captured by top-r (r = 8..128)
  - attention output error if we truncate to rank r

Usage:
  python scripts/run_lowrank.py --model lfm2.5-350m-base --context 1024
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model_loader import MODELS, inspect_model, load_model  # noqa: E402
from utils import append_jsonl, set_seed  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "lowrank.jsonl")


def effective_rank(svals: torch.Tensor) -> float:
    """Participation ratio: (sum s)^2 / sum(s^2). Higher = flatter spectrum."""
    s = svals.float()
    return float((s.sum() ** 2) / (s ** 2).sum().clamp_min(1e-12))


def energy_at_rank(svals: torch.Tensor, r: int) -> float:
    """Fraction of Frobenius energy captured by the top-r singular values."""
    s2 = (svals.float()) ** 2
    r = min(r, s2.numel())
    return float(s2[:r].sum() / s2.sum().clamp_min(1e-12))


@torch.no_grad()
def capture_qk(model, tok, ids, layer_idx):
    """Capture post-norm post-RoPE Q and K for one full-attention layer via a hook."""
    layer_list = None
    for n, mod in model.named_modules():
        if n.endswith("layers") and isinstance(mod, torch.nn.ModuleList):
            layer_list = mod
            break
    attn = getattr(layer_list[layer_idx], "self_attn", None) or getattr(layer_list[layer_idx], "attention")
    captured = {}

    orig_forward = attn.forward
    def spy(hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kw):
        # recompute q/k exactly as the module does, then capture
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)
        q_proj = attn.q_proj(hidden_states)
        if hasattr(attn, "q_norm"):  # Qwen gated
            q, _g = torch.chunk(q_proj.view(*input_shape, -1, attn.head_dim * 2), 2, dim=-1)
            q = attn.q_norm(q.view(hidden_shape)).transpose(1, 2)
            k = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        else:
            q_norm = getattr(attn, "q_layernorm", None)
            k_norm = getattr(attn, "k_layernorm", None)
            q = attn.q_proj(hidden_states).view(hidden_shape)
            k = attn.k_proj(hidden_states).view(hidden_shape)
            if q_norm is not None:
                q = q_norm(q); k = k_norm(k)
            q = q.transpose(1, 2); k = k.transpose(1, 2)
        cos, sin = position_embeddings
        import sys as _s
        mod = _s.modules.get(type(attn).__module__)
        rope = getattr(mod, "apply_rotary_pos_emb", None)
        if rope is not None:
            q, k = rope(q, k, cos, sin)
        captured["q"] = q.detach()
        captured["k"] = k.detach()
        return orig_forward(hidden_states, position_embeddings, attention_mask,
                            past_key_values, **kw)
    attn.forward = spy
    try:
        model(ids)
    finally:
        attn.forward = orig_forward
    return captured["q"], captured["k"]


def run(model_key, context, dtype, seed):
    model_id = MODELS[model_key]
    info = inspect_model(model_id)
    model, tok = load_model(model_id, dtype=dtype)
    device = next(model.parameters()).device
    full_layers = info.full_attention_layers
    print(f"[load] {model_id} full_attn={full_layers}")
    set_seed(seed)
    ids = tok("The history of the region spans many centuries. " * (context // 12),
              return_tensors="pt").input_ids[:, :context].to(device)

    ranks = [8, 16, 32, 64, 128]
    for layer_idx in full_layers:
        q, k = capture_qk(model, tok, ids, layer_idx)  # (1, n_kv, len, head_dim) or (1,h,L,d)
        # mean over heads -> (L, d); use head_dim as feature dim
        qm = q[0].mean(dim=0).float()   # (L, d)
        km = k[0].mean(dim=0).float()   # (L, d)
        S = qm @ km.T                    # (L, L) score matrix
        svals = torch.linalg.svdvals(S)
        er = effective_rank(svals)
        energies = {r: round(energy_at_rank(svals, r), 4) for r in ranks}
        rec = {"experiment": "lowrank", "model": model_id, "layer": layer_idx,
               "context": context, "effective_rank": round(er, 1),
               "head_dim": qm.shape[1], **{f"energy_r{r}": energies[r] for r in ranks}}
        append_jsonl(RESULTS, rec)
        print(f"  layer {layer_idx}: eff_rank={er:.0f}/{qm.shape[1]} energy_r32={energies[32]:.2f} "
              f"energy_r64={energies[64]:.2f} energy_r128={energies[128]:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS.keys()))
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(args.model, args.context, args.dtype, args.seed)


if __name__ == "__main__":
    main()
