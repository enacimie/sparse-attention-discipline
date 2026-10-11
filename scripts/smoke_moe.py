"""Smoke test for the LFM2.5-8B-A1B MoE hybrid: load, inspect the full-attn module attributes the
interceptor relies on, and run a 1-chunk fidelity eval with local_global_topk vs dense."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from attention_interceptor import apply_method, restore  # noqa: E402
from attention_methods import AttentionMethod  # noqa: E402
from model_loader import MODELS, inspect_model, load_model  # noqa: E402
from utils import fidelity_metrics, set_seed  # noqa: E402


@torch.no_grad()
def get_logits(model, ids):
    return model(ids).logits[0].cpu().float()


def main():
    model_id = MODELS["lfm2.5-8b-a1b"]
    info = inspect_model(model_id)
    print(f"[info] type={info.model_type} layers={info.num_hidden_layers} "
          f"full_attn={info.full_attention_layers} heads={info.num_attention_heads} "
          f"kv={info.num_key_value_heads} head_dim={info.head_dim}")
    model, tok = load_model(model_id, dtype="float16", load_in_4bit=True)
    device = next(model.parameters()).device

    # inspect one full-attn module
    layer_list = None
    for n, mod in model.named_modules():
        if n.endswith("layers") and isinstance(mod, torch.nn.ModuleList):
            layer_list = mod
            break
    attn = getattr(layer_list[info.full_attention_layers[0]], "self_attn", None) or \
        getattr(layer_list[info.full_attention_layers[0]], "attention", None)
    print("[attn] class:", type(attn).__name__)
    for a in ["q_proj", "k_proj", "v_proj", "o_proj", "out_proj", "q_norm", "k_norm",
              "q_layernorm", "k_layernorm"]:
        print(f"  has {a}:", hasattr(attn, a))

    ids = tok("The quick brown fox jumps over the lazy dog. " * 80,
              return_tensors="pt").input_ids[:, :1024].to(device)
    set_seed(42)
    ref = get_logits(model, ids)
    method = AttentionMethod(name="local_global_topk", window=64, n_global=4, topk=32)
    originals = apply_method(model, method, info.full_attention_layers)
    try:
        set_seed(42)
        approx = get_logits(model, ids)
    finally:
        restore(model, originals, info.full_attention_layers)
    fid = fidelity_metrics(approx, ref)
    print(f"[fidelity] local_global_topk ctx=1024: {fid}")


if __name__ == "__main__":
    main()
