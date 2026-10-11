#!/usr/bin/env python3
"""WHERE DOES RETRIEVAL LIVE IN A HYBRID? (causal full-attention layer knockout)

Palanca 2(b): characterize whether long-context retrieval is carried by the few
full_attention layers of a hybrid transformer, while the linear/conv layers contribute
nothing to it.

Method: attention KNOCKOUT. For a chosen set of full_attention layers we wrap their
forward so the (fully computed, KV-cache-consistent) attention output is zeroed before
the residual add. The residual stream then passes through that layer with NO attention
contribution (MLP still runs). Because the linear/conv layers stay active throughout,
"retrieval dies under knockout" simultaneously proves (a) the full-attn layers are
NECESSARY and (b) the linear layers alone are INSUFFICIENT.

Conditions (each = needle recall over depths x seeds):
  dense        : no knockout (baseline, expect ~1.0 on base models)
  ablate_all   : knock out ALL full-attn layers
  loo_<L>      : leave-one-out, knock out only layer L      -> necessity per layer
  loi_<L>      : leave-one-in,  keep only layer L dense     -> sufficiency per layer
  ctrl_short   : ablate_all at a SHORT context               -> control: is the deficit
                   long-range-specific, or does knockout just break the model globally?

Usage:
  python scripts/run_layer_ablation.py --model qwen3.5-0.8b --ctx 4096 --ctrl-ctx 256
"""
import argparse
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model_loader import MODELS, inspect_model, load_model  # noqa: E402
from needle import build_needle_prompt, check_needle, make_needle  # noqa: E402
from utils import append_jsonl, set_seed  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "layer_ablation.jsonl")


def robust_hit(generated: str, needle: str) -> bool:
    """Exact-match is brittle when a knockout changes output FORMATTING (e.g. the model
    emits the passkey digits concatenated '10433' instead of spaced '1 0 4 3 3'). The
    filler/question contain no digits, so the only digit source is the needle: compare
    the needle's digit string against the digits present in the generation."""
    needle_digits = "".join(c for c in needle if c.isdigit())
    gen_digits = "".join(c for c in generated if c.isdigit())
    return needle_digits in gen_digits


def _locate_layers(model):
    for n, mod in model.named_modules():
        if n.endswith("layers") and isinstance(mod, torch.nn.ModuleList):
            return mod
    raise RuntimeError("could not locate decoder layers")


def _attn_of(layer):
    return getattr(layer, "self_attn", None) or getattr(layer, "attention", None)


def make_zero_forward(orig_forward):
    """Wrap the REAL attention forward: compute everything (projections, RoPE, KV-cache
    update, SDPA) then zero the output so the residual add contributes nothing. Keeping
    the original compute makes the KV cache consistent across generate() steps."""
    def zero_forward(*args, **kwargs):
        out = orig_forward(*args, **kwargs)
        return (torch.zeros_like(out[0]),) + tuple(out[1:])
    return zero_forward


def knockout(model, layer_idxs):
    layer_list = _locate_layers(model)
    originals = {}
    for idx in layer_idxs:
        attn = _attn_of(layer_list[idx])
        originals[idx] = attn.forward
        attn.forward = make_zero_forward(attn.forward)
    return originals


def restore(model, originals):
    layer_list = _locate_layers(model)
    for idx, fwd in originals.items():
        _attn_of(layer_list[idx]).forward = fwd


@torch.no_grad()
def needle_recall(model, tok, ablate_layers, ctx, depths, seeds, gen_tokens=24):
    hits, rhits, total = 0, 0, 0
    device = next(model.parameters()).device
    originals = knockout(model, ablate_layers) if ablate_layers else {}
    try:
        for depth in depths:
            for seed in seeds:
                set_seed(seed)
                rng = random.Random(seed)
                needle = make_needle(rng)
                prompt, ndl = build_needle_prompt(tok, ctx, needle, depth)
                ids = tok(prompt, return_tensors="pt").input_ids.to(device)
                out = model.generate(ids, max_new_tokens=gen_tokens, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
                gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
                hits += int(check_needle(gen, ndl))
                rhits += int(robust_hit(gen, ndl))
                total += 1
    finally:
        if ablate_layers:
            restore(model, originals)
    n = total if total else 1
    return (hits / n, rhits / n), hits, rhits, total


def emit(model_key, ctx, cond, layers, recs, hits, rhits, total):
    rec_exact, rec_robust = recs
    row = {"experiment": "layer_ablation", "model": model_key, "context": ctx,
           "condition": cond, "ablated_layers": layers, "n_ablated": len(layers),
           "needle_recall": round(rec_robust, 4), "needle_recall_exact": round(rec_exact, 4),
           "needle_hits": rhits, "needle_hits_exact": hits, "needle_total": total}
    append_jsonl(RESULTS, row)
    print(f"ctx={ctx:5d} {cond:12s} nabla={len(layers)} robust={rec_robust:.3f} "
          f"exact={rec_exact:.3f} ({rhits}/{total})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-0.8b")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--ctrl-ctx", type=int, default=256,
                    help="short-context control for ablate_all")
    ap.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    model_id = MODELS[args.model]
    info = inspect_model(model_id)
    model, tok = load_model(model_id, dtype=args.dtype, load_in_4bit=args.load_in_4bit,
                            device_map="cuda")
    full = info.full_attention_layers
    print(f"[load] {model_id} full_attn={full}", flush=True)

    ctx = args.ctx
    # 1) dense baseline
    recs, h, rh, t = needle_recall(model, tok, [], ctx, args.depths, args.seeds)
    emit(args.model, ctx, "dense", [], recs, h, rh, t)
    # 2) ablate all
    recs, h, rh, t = needle_recall(model, tok, full, ctx, args.depths, args.seeds)
    emit(args.model, ctx, "ablate_all", full, recs, h, rh, t)
    # 3) leave-one-out (necessity)
    for L in full:
        recs, h, rh, t = needle_recall(model, tok, [L], ctx, args.depths, args.seeds)
        emit(args.model, ctx, f"loo_{L}", [L], recs, h, rh, t)
    # 4) leave-one-in (sufficiency)
    for L in full:
        keep = [x for x in full if x != L]
        recs, h, rh, t = needle_recall(model, tok, keep, ctx, args.depths, args.seeds)
        emit(args.model, ctx, f"loi_{L}", keep, recs, h, rh, t)
    # 5) short-context control (ablate_all at ctrl_ctx)
    recs, h, rh, t = needle_recall(model, tok, full, args.ctrl_ctx, args.depths, args.seeds)
    emit(args.model, args.ctrl_ctx, "ctrl_short", full, recs, h, rh, t)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
