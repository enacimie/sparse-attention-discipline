"""Model loading utilities for the efficient-attention study.

Handles the hybrid families:
- Qwen3.5 (VLM; we load ONLY the text decoder via Qwen3_5ForCausalLM, skipping the
  vision tower to save VRAM). Layers: 18 linear_attention + 6 full_attention.
- LFM2.5 (text-only; Lfm2ForCausalLM). Layers: conv + full_attention.
- Falcon-H1 (parallel hybrid: Mamba branch + full-attention branch in EVERY layer).

Only the `full_attention` layers (or, for Falcon-H1, the attention branch of every
layer) are targets for attention-approximation methods.
"""

from dataclasses import dataclass

import torch
from transformers import AutoConfig, AutoTokenizer

# Model registry: friendly key -> HF id. Prefer Base variants for perplexity
# baselines (instruct-tuned models give erratic absolute PPL on raw wiki text).
MODELS = {
    "lfm2.5-350m": "LiquidAI/LFM2.5-350M",
    "lfm2.5-350m-base": "LiquidAI/LFM2.5-350M-Base",
    "lfm2.5-2.6b": "LiquidAI/LFM2.5-2.6B",
    "lfm2.5-2.6b-base": "LiquidAI/LFM2.5-2.6B-Base",
    # MoE hybrid (8B total / ~1B active): addresses the reviewer "no MoE / scale" gap.
    "lfm2.5-8b-a1b": "LiquidAI/LFM2.5-8B-A1B",
    "qwen3.5-0.8b": "Qwen/Qwen3.5-0.8B",
    "qwen3.5-2b": "Qwen/Qwen3.5-2B",
    "qwen3.5-2b-base": "Qwen/Qwen3.5-2B-Base",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",   # dense baseline reference
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",   # dense baseline reference
    # SCALE targets (cloud / Kaggle, run in 4-bit). Current-family HYBRIDS (32 layers,
    # 8 full_attention at L3,7,...,31), so they test the hybrid claim at scale. 9B sits at
    # Q2model's sensitivity-inversion threshold (>=9B) -> the scientifically key point.
    "qwen3.5-4b-base": "Qwen/Qwen3.5-4B-Base",
    "qwen3.5-9b-base": "Qwen/Qwen3.5-9B-Base",
    # SCALE >9B (RunPod 24GB, 4-bit): addresses the reviewer "scale limited to 9B" gap.
    "qwen3.5-27b": "Qwen/Qwen3.5-27B",
    # Qwen3.8-27B (released 2026-08-14): same qwen3_5 hybrid arch as 3.5 (Gated DeltaNet
    # + Gated Attention), 64 layers / 16 full-attn. VLM; load text decoder only.
    "qwen3.8-27b": "Qwen/Qwen3.8-27B",
    # Cross-family dense replication: non-Qwen dense transformer (Llama architecture).
    "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B",
    # SmolLM3-3B-Base is DENSE (GQA + NoPE on every 4th layer), NOT a sub-quadratic
    # hybrid. Used only as a dense reference (intercept nothing / dense baseline).
    "smollm3-3b": "HuggingFaceTB/SmolLM3-3B-Base",
    # Falcon-H1 (TII): PARALLEL hybrid -- every layer runs a Mamba (SSM, sub-quadratic)
    # branch AND a full-attention branch, summed (layers_block_type == ["hybrid"] * N).
    # A different hybrid topology from Qwen3.5/LFM2.5 (serial). We sparsify the
    # attention branch in EVERY layer (full_attention_layers == all layers).
    "falcon-h1-0.5b": "tiiuae/Falcon-H1-0.5B-Base",
    "falcon-h1-1.5b": "tiiuae/Falcon-H1-1.5B-Base",
    "falcon-h1-1.5b-deep": "tiiuae/Falcon-H1-1.5B-Deep-Base",
    "falcon-h1-3b": "tiiuae/Falcon-H1-3B-Base",
}


@dataclass
class ModelInfo:
    model_id: str
    model_type: str
    num_hidden_layers: int
    layer_types: list
    full_attention_layers: list  # indices of layers that are full attention
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    hidden_size: int

    @property
    def n_full_attention(self) -> int:
        return len(self.full_attention_layers)


def get_config(model_id: str):
    """Return the text-relevant config (unwraps Qwen3.5 text_config)."""
    cfg = AutoConfig.from_pretrained(model_id)
    return getattr(cfg, "text_config", cfg), cfg


def inspect_model(model_id: str) -> ModelInfo:
    """Read config and identify which layers are full attention."""
    text_cfg, root_cfg = get_config(model_id)
    model_type = root_cfg.model_type
    n_layers = text_cfg.num_hidden_layers
    layer_types = getattr(text_cfg, "layer_types", None)
    if model_type == "falcon_h1":
        # Parallel hybrid: every layer has BOTH a Mamba branch and a full-attention
        # branch. There is no "full_attention-only" layer, so we intercept the
        # attention branch in EVERY layer.
        layer_types = getattr(text_cfg, "layers_block_type", None) or ["hybrid"] * n_layers
        full_layers = list(range(n_layers))
    elif layer_types is None:
        # Dense transformer: every layer is full attention.
        layer_types = ["full_attention"] * n_layers
        full_layers = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    else:
        full_layers = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    n_heads = getattr(text_cfg, "num_attention_heads", 0)
    head_dim = getattr(text_cfg, "head_dim", None) or (
        text_cfg.hidden_size // n_heads if n_heads else 0
    )
    return ModelInfo(
        model_id=model_id,
        model_type=model_type,
        num_hidden_layers=n_layers,
        layer_types=layer_types,
        full_attention_layers=full_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=getattr(text_cfg, "num_key_value_heads", n_heads),
        head_dim=head_dim,
        hidden_size=text_cfg.hidden_size,
    )


def load_model(model_id: str, dtype: str = "bfloat16", attn_impl: str = "sdpa",
               load_in_4bit: bool = False, device_map: str = "cuda"):
    """Load a text-only causal LM.

    For Qwen3.5 (VLM) we use Qwen3_5ForCausalLM to avoid loading the vision tower.
    For LFM2 / Qwen3 we use AutoModelForCausalLM.
    """
    torch_dtype = getattr(torch, dtype)
    kwargs = dict(dtype=torch_dtype, device_map=device_map)
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
        )
        kwargs.pop("dtype", None)

    model_type = AutoConfig.from_pretrained(model_id).model_type
    if model_type == "qwen3_5":
        from transformers import Qwen3_5ForCausalLM
        kwargs["attn_implementation"] = attn_impl
        model = Qwen3_5ForCausalLM.from_pretrained(model_id, **kwargs)
    else:
        from transformers import AutoModelForCausalLM
        kwargs["attn_implementation"] = attn_impl
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.eval()
    return model, tokenizer
