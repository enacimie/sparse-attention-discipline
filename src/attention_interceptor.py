"""Interception of full_attention layers to apply approximate attention.

Strategy: wrap the attention module's forward to compute attention ONLY over the
selected query-key pairs defined by an AttentionMethod. We build a boolean keep
mask, convert it to an additive float mask (0 keep / -inf drop), and run SDPA.
Because masked-out positions get -inf, their softmax weight is exactly 0, so the
result equals computing attention over the selected pairs only -- this is a real
complexity reduction in the number of attended pairs (counted and reported).

The wrapper replicates the module's original projection + RoPE + norm logic and
only changes the final attention computation, so it stays faithful to the model.
"""

import torch
import torch.nn.functional as F

from attention_methods import (AttentionMethod, build_attention_mask,
                               build_content_mask, count_selected_pairs)

# Global registry: number of computed pairs per forward (efficiency metric).
PAIR_COUNTER = {"computed": 0, "full": 0}

# Optional diagnostic recorder: called per full-attention layer inside the patched
# forward with (layer_idx, query, key, keep) — query/key are post-RoPE, pre-GQA-repeat.
# Default None: no behavior change.
RECORDER = None


def set_recorder(fn):
    global RECORDER
    RECORDER = fn


def clear_recorder():
    global RECORDER
    RECORDER = None


def reset_pair_counter():
    PAIR_COUNTER["computed"] = 0
    PAIR_COUNTER["full"] = 0


def causal_dense_pairs(q_len: int, kv_len: int) -> int:
    """Number of causally-valid dense (query,key) pairs when the q_len queries are the
    LAST q_len positions of a kv_len cache. Prefill q_len=kv_len=L -> L(L+1)/2;
    decode q_len=1 -> kv_len."""
    return q_len * (2 * kv_len - q_len + 1) // 2


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    x = x[:, :, None, :, :].expand(b, h, n_rep, s, d)
    return x.reshape(b, h * n_rep, s, d)


def make_method_forward(orig_forward, method: AttentionMethod):
    """Return a forward function that applies the given attention method.

    Generic across Qwen3.5 (q_norm/k_norm, gated q_proj, o_proj) and
    LFM2 (q_layernorm/k_layernorm, out_proj). RoPE is applied via the model's
    own apply_rotary_pos_emb if available.
    """
    # resolve norm / projection / rope attributes once (model-family specific)
    def resolve(self):
        q_norm = getattr(self, "q_norm", None) or getattr(self, "q_layernorm", None)
        k_norm = getattr(self, "k_norm", None) or getattr(self, "k_layernorm", None)
        out_proj = getattr(self, "o_proj", None) or getattr(self, "out_proj", None)
        # GATED attention (Qwen3.5: q_proj emits query+gate) is NOT the same as having q_norm:
        # Qwen3 DENSE also has QK-norm (q_norm/k_norm) but no gate. Detect by q_proj width:
        # gated <=> q_proj.out_features == 2 * (kv_heads * groups * head_dim).
        expected_q = self.k_proj.out_features * self.num_key_value_groups
        gated = (self.q_proj.out_features == 2 * expected_q)
        # RoPE function from the module's own modeling file
        import sys
        mod = sys.modules.get(type(self).__module__)
        if mod is None:
            import importlib
            mod = importlib.import_module(type(self).__module__)
        rope = getattr(mod, "apply_rotary_pos_emb", None)
        # Falcon-H1 scales keys by a per-model constant (key_multiplier).
        key_multiplier = getattr(self, "key_multiplier", None)
        # SmolLM3 applies RoPE only on RoPE layers (no_rope_layers toggle); default
        # True for families where every intercepted layer uses RoPE.
        use_rope = getattr(self, "use_rope", True)
        return q_norm, k_norm, out_proj, gated, rope, key_multiplier, use_rope

    def forward(hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, **kwargs):
        self = forward._self  # bound attention module
        q_norm, k_norm, out_proj, gated, rope, key_multiplier, use_rope = forward._resolved
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # --- projections + norms + RoPE (faithful to the original module) ---
        if gated:
            query_states, gate = torch.chunk(
                self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
            gate = gate.reshape(*input_shape, -1)
        else:
            query_states = self.q_proj(hidden_states).view(hidden_shape)
            gate = None
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        if key_multiplier is not None:
            key_states = key_states * key_multiplier
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        if q_norm is not None:
            query_states = q_norm(query_states.view(hidden_shape))
            key_states = k_norm(key_states.view(hidden_shape))
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if rope is not None and position_embeddings is not None and use_rope:
            cos, sin = position_embeddings
            query_states, key_states = rope(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx)

        # --- selection mask (which pairs we actually compute) ---
        b, _, q_len, _ = query_states.shape
        kv_len = key_states.shape[2]
        if method.name in ("topk", "random_topk", "uniform_random", "selector", "cluster",
                           "ann", "local_global_topk", "local_global_randtopk", "lsh", "bandit",
                           "nystrom", "lg_topk_block", "lgr_block",
                           "local_global_oracle", "local_global_voracle",
                           "local_global_adaptive", "local_global_entropy",
                           "local_global_perhead"):
            # content-based selection on pre-GQA key_states (n_kv, kv_len, d)
            keep = build_content_mask(method, query_states, key_states, causal=True,
                                      value=value_states)
        else:
            keep = build_attention_mask(method, kv_len, hidden_states.device, causal=True)
        if q_len != kv_len:  # decode: query is the last token(s)
            keep = keep[-q_len:, :]
        PAIR_COUNTER["computed"] += count_selected_pairs(keep)
        PAIR_COUNTER["full"] += causal_dense_pairs(q_len, kv_len)
        if RECORDER is not None:
            RECORDER(self.layer_idx, query_states, key_states, keep)

        additive = torch.zeros(q_len, kv_len, device=hidden_states.device,
                               dtype=query_states.dtype)
        additive.masked_fill_(~keep, torch.finfo(query_states.dtype).min)
        attn_mask = additive[None, None, :, :]

        # --- attention over selected pairs only ---
        key_states = _repeat_kv(key_states, self.num_key_value_groups)
        value_states = _repeat_kv(value_states, self.num_key_value_groups)
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=attn_mask,
            dropout_p=0.0, is_causal=False)

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        if gate is not None:
            attn_output = attn_output * torch.sigmoid(gate)
        attn_output = out_proj(attn_output)
        return attn_output, None

    # attach resolver; both the wrapper and inner forward share attribute holders
    def forward_with_resolve(*a, **k):
        if not hasattr(forward_with_resolve, "_resolved"):
            forward_with_resolve._resolved = resolve(forward_with_resolve._self)
        forward._self = forward_with_resolve._self
        forward._resolved = forward_with_resolve._resolved
        return forward(*a, **k)
    return forward_with_resolve


def apply_method(model, method: AttentionMethod, full_attention_layers):
    """Patch the full_attention layers of the model to use the given method."""
    # locate the decoder layer list (works for Qwen3.5 and LFM2)
    layer_list = None
    for n, mod in model.named_modules():
        if n.endswith("layers") and isinstance(mod, torch.nn.ModuleList):
            layer_list = mod
            break
    if layer_list is None:
        raise RuntimeError("could not locate decoder layers")

    originals = {}
    for idx in full_attention_layers:
        layer = layer_list[idx]
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
        if attn is None:
            continue
        originals[idx] = attn.forward
        new_fwd = make_method_forward(attn.forward, method)
        new_fwd._self = attn
        attn.forward = new_fwd
    return originals


def restore(model, originals, full_attention_layers):
    """Restore original forwards."""
    layer_list = None
    for n, mod in model.named_modules():
        if n.endswith("layers") and isinstance(mod, torch.nn.ModuleList):
            layer_list = mod
            break
    for idx, fwd in originals.items():
        layer = layer_list[idx]
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
        attn.forward = fwd
