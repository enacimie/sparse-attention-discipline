"""FlexAttention backend for the attention interceptor.

Applies the SAME per-key keep-mask as attention_interceptor (so fidelity is identical to the
validated SDPA-mask path) but runs it through flex_attention, which skips fully-masked blocks.
This turns the logical pair_ratio into REAL latency savings. The mask is built exactly as before
(build_content_mask / build_attention_mask); only the final attention kernel changes.
"""
import sys

import torch

from attention_methods import (AttentionMethod, build_attention_mask,
                               build_content_mask, count_selected_pairs)
from attention_interceptor import PAIR_COUNTER, _repeat_kv, causal_dense_pairs

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
except Exception:  # pragma: no cover
    flex_attention = None
    create_block_mask = None

_FLEX = {"fn": None}


def _flex_fn():
    if _FLEX["fn"] is None:
        _FLEX["fn"] = torch.compile(flex_attention, dynamic=False)
    return _FLEX["fn"]


def make_method_forward_flex(method: AttentionMethod, block_size: int = 128):
    def resolve(self):
        q_norm = getattr(self, "q_norm", None) or getattr(self, "q_layernorm", None)
        k_norm = getattr(self, "k_norm", None) or getattr(self, "k_layernorm", None)
        out_proj = getattr(self, "o_proj", None) or getattr(self, "out_proj", None)
        # GATED attention (Qwen3.5) is NOT the same as having q_norm: Qwen3 DENSE also has
        # QK-norm but no gate. Detect by q_proj width: gated <=> q_proj == 2 * kv_heads*groups*head_dim.
        expected_q = self.k_proj.out_features * self.num_key_value_groups
        gated = (self.q_proj.out_features == 2 * expected_q)
        mod = sys.modules.get(type(self).__module__)
        if mod is None:
            import importlib
            mod = importlib.import_module(type(self).__module__)
        rope = getattr(mod, "apply_rotary_pos_emb", None)
        return q_norm, k_norm, out_proj, gated, rope

    def forward(hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, **kwargs):
        self = forward._self
        q_norm, k_norm, out_proj, gated, rope = forward._resolved
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        if gated:
            query_states, gate = torch.chunk(
                self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
            gate = gate.reshape(*input_shape, -1)
        else:
            query_states = self.q_proj(hidden_states).view(hidden_shape)
            gate = None
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        if q_norm is not None:
            query_states = q_norm(query_states.view(hidden_shape))
            key_states = k_norm(key_states.view(hidden_shape))
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if rope is not None and position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = rope(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx)

        b, _, q_len, _ = query_states.shape
        kv_len = key_states.shape[2]
        if method.name in ("topk", "random_topk", "uniform_random", "selector", "cluster",
                           "ann", "local_global_topk", "local_global_randtopk", "lsh", "bandit",
                           "nystrom", "lg_topk_block", "lgr_block"):
            keep = build_content_mask(method, query_states, key_states, causal=True)
        else:
            keep = build_attention_mask(method, kv_len, hidden_states.device, causal=True)
        if q_len != kv_len:
            keep = keep[-q_len:, :]
        PAIR_COUNTER["computed"] += count_selected_pairs(keep)
        PAIR_COUNTER["full"] += causal_dense_pairs(q_len, kv_len)

        keep_c = keep.contiguous()

        def mask_mod(bidx, h, qi, kvi):
            return keep_c[qi, kvi]

        n_heads = query_states.shape[1]
        block_mask = create_block_mask(mask_mod, B=b, H=n_heads, Q_LEN=q_len, KV_LEN=kv_len,
                                       BLOCK_SIZE=block_size, device=hidden_states.device)

        key_states = _repeat_kv(key_states, self.num_key_value_groups)
        value_states = _repeat_kv(value_states, self.num_key_value_groups)
        attn_output = _flex_fn()(query_states, key_states, value_states, block_mask=block_mask)

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        if gate is not None:
            attn_output = attn_output * torch.sigmoid(gate)
        attn_output = out_proj(attn_output)
        return attn_output, None

    def forward_with_resolve(*a, **k):
        if not hasattr(forward_with_resolve, "_resolved"):
            forward_with_resolve._resolved = resolve(forward_with_resolve._self)
        forward._self = forward_with_resolve._self
        forward._resolved = forward_with_resolve._resolved
        return forward(*a, **k)
    return forward_with_resolve


def apply_method_flex(model, method: AttentionMethod, full_attention_layers, block_size=128):
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
        new_fwd = make_method_forward_flex(method, block_size)
        new_fwd._self = attn
        attn.forward = new_fwd
    return originals
