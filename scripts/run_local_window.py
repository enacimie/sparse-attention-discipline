"""Experiment 3: local (sliding-window) attention vs dense baseline.

For each window size we:
  - capture dense baseline logits (reference),
  - apply local_window attention to the full_attention layers,
  - measure fidelity (KL/MSE/cosine vs dense) + computed-pair ratio + needle recall.

Usage:
  python scripts/run_local_window.py --model qwen3.5-0.8b --contexts 1024 2048 \
      --windows 64 128 256 512
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from attention_interceptor import (PAIR_COUNTER, apply_method, reset_pair_counter,
                                   restore)  # noqa: E402
from attention_methods import AttentionMethod  # noqa: E402
from model_loader import MODELS, inspect_model, load_model  # noqa: E402
from needle import build_needle_prompt, check_needle, make_needle  # noqa: E402
import random  # noqa: E402
from utils import GpuMeter, append_jsonl, fidelity_metrics, set_seed  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "local_window.jsonl")


@torch.no_grad()
def get_logits(model, ids):
    return model(ids).logits[0].float().cpu()


@torch.no_grad()
def needle_recall(model, tok, ctx, depths, seeds, gen_tokens=24):
    hits, total = 0, 0
    for depth in depths:
        for seed in seeds:
            set_seed(seed)
            rng = random.Random(seed)
            needle = make_needle(rng)
            prompt, needle = build_needle_prompt(tok, ctx, needle, depth)
            ids = tok(prompt, return_tensors="pt").input_ids.to(next(model.parameters()).device)
            out = model.generate(ids, max_new_tokens=gen_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            hits += int(check_needle(gen, needle)); total += 1
    return hits / total


def run(model_key, contexts, windows, dtype, seed):
    model_id = MODELS[model_key]
    info = inspect_model(model_id)
    model, tok = load_model(model_id, dtype=dtype)
    device = next(model.parameters()).device
    full_layers = info.full_attention_layers
    print(f"[load] {model_id} full_attn_layers={full_layers}")

    depths, seeds = [0.1, 0.5, 0.9], [42, 43, 44]
    for ctx in contexts:
        ids = tok("The history of the region spans many centuries. " * (ctx // 12),
                  return_tensors="pt").input_ids[:, :ctx].to(device)

        # dense reference
        set_seed(seed)
        reset_pair_counter()
        with GpuMeter() as dm:
            ref_logits = get_logits(model, ids)
            # dense runs the ORIGINAL forward (no wrapper), so PAIR_COUNTER stays 0.
            # For causal dense attention, computed pairs = full causal pairs per layer.
            L = ids.shape[1]
            causal_pairs = L * (L + 1) // 2
            dense_pairs = causal_pairs * info.n_full_attention
            dense_full = dense_pairs
        dense_needle = needle_recall(model, tok, ctx, depths, seeds)
        print(f"  ctx={ctx} DENSE pairs={dense_pairs} needle={dense_needle:.2f}")
        append_jsonl(RESULTS, {"experiment": "local_window", "method": "dense",
                               "model": model_id, "context": ctx, "window": -1,
                               "computed_pairs": dense_pairs, "full_pairs": dense_full,
                               "pair_ratio": 1.0, "kl_div": 0.0, "logit_mse": 0.0,
                               "logit_cosine": 1.0, "needle_recall": dense_needle,
                               **dm.to_dict()})

        for w in windows:
            method = AttentionMethod(name="local_window", window=w, seed=seed)
            originals = apply_method(model, method, full_layers)
            set_seed(seed)
            # measure logits without polluting the pair counter
            with GpuMeter() as m:
                approx_logits = get_logits(model, ids)
                fid = fidelity_metrics(approx_logits, ref_logits)
            # count computed pairs on a dedicated forward (full-attn layers only)
            reset_pair_counter()
            with torch.no_grad():
                get_logits(model, ids)
            computed = PAIR_COUNTER["computed"]
            nr = needle_recall(model, tok, ctx, depths, seeds)
            restore(model, originals, full_layers)
            ratio = computed / dense_pairs if dense_pairs else 0.0
            rec = {"experiment": "local_window", "method": "local_window",
                   "model": model_id, "context": ctx, "window": w,
                   "computed_pairs": computed, "full_pairs": dense_full,
                   "pair_ratio": round(ratio, 4), "needle_recall": nr,
                   **fid, **m.to_dict()}
            append_jsonl(RESULTS, rec)
            print(f"  ctx={ctx} w={w}: pair_ratio={ratio:.3f} kl={fid['kl_div']} "
                  f"cos={fid['logit_cosine']} needle={nr:.2f} vram={rec['peak_vram_mb']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS.keys()))
    ap.add_argument("--contexts", type=int, nargs="+", default=[1024])
    ap.add_argument("--windows", type=int, nargs="+", default=[64, 128, 256, 512])
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(args.model, args.contexts, args.windows, args.dtype, args.seed)


if __name__ == "__main__":
    main()
