#!/usr/bin/env python3
"""CONTROL PROBE for run_layer_ablation: is the knockout retrieval-specific or does it
just break the model globally?

For each condition we report (a) short-context wikitext PPL (general LM ability) and
(b) one sample needle generation (is the output coherent or garbage?). A clean
dissociation = PPL barely moves while needle recall collapses.
"""
import argparse
import math
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from model_loader import MODELS, inspect_model, load_model  # noqa: E402
from needle import build_needle_prompt, make_needle  # noqa: E402
from run_layer_ablation import knockout, restore, robust_hit  # noqa: E402
from utils import append_jsonl, set_seed  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "layer_ablation.jsonl")


@torch.no_grad()
def short_ppl(model, tok, ctx, n_chunks=3):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    text = "\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    device = next(model.parameters()).device
    losses = []
    for s in range(0, n_chunks * ctx, ctx):
        if s + ctx > ids.numel():
            break
        chunk = ids[s:s + ctx].unsqueeze(0).to(device)
        losses.append(model(chunk, labels=chunk).loss.item())
    return math.exp(sum(losses) / len(losses)) if losses else float("nan")


@torch.no_grad()
def one_generation(model, tok, ctx, depth=0.5, seed=42, gen_tokens=24):
    set_seed(seed)
    rng = random.Random(seed)
    needle = make_needle(rng)
    prompt, ndl = build_needle_prompt(tok, ctx, needle, depth)
    ids = tok(prompt, return_tensors="pt").input_ids.to(next(model.parameters()).device)
    out = model.generate(ids, max_new_tokens=gen_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    return needle, gen, (needle in gen), robust_hit(gen, ndl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-0.8b")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--ppl-ctx", type=int, default=1024)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--loo-layers", type=int, nargs="+", default=None,
                    help="explicit full-attn layers to leave-one-out (default: auto mid/late)")
    ap.add_argument("--loi-layer", type=int, default=None,
                    help="explicit single full-attn layer to leave-IN (default: none)")
    args = ap.parse_args()

    model_id = MODELS[args.model]
    info = inspect_model(model_id)
    model, tok = load_model(model_id, dtype=args.dtype, load_in_4bit=args.load_in_4bit,
                            device_map="cuda")
    full = info.full_attention_layers
    print(f"[load] {model_id} full_attn={full}", flush=True)

    conditions = [("dense", []), ("ablate_all", full)]
    loo = args.loo_layers
    if loo is None:  # auto: two most middle-to-late full-attn layers
        loo = [full[len(full) // 2 - 1], full[len(full) // 2]] if len(full) >= 2 else [full[0]]
    for L in loo:
        conditions.append((f"loo_{L}", [L]))
    if args.loi_layer is not None:
        L = args.loi_layer
        conditions.append((f"loi_{L}", [x for x in full if x != L]))
    for name, layers in conditions:
        originals = knockout(model, layers) if layers else {}
        try:
            ppl = short_ppl(model, tok, args.ppl_ctx)
            needle, gen, hit, rhit = one_generation(model, tok, args.ctx)
        finally:
            if layers:
                restore(model, originals)
        row = {"experiment": "layer_ablation_probe", "model": args.model,
               "condition": name, "ablated_layers": layers,
               "ppl_ctx": args.ppl_ctx, "short_ppl": round(ppl, 2),
               "gen_hit": int(hit), "gen_hit_robust": int(rhit),
               "gen_sample": gen.strip()[:160], "needle": needle}
        append_jsonl(RESULTS, row)
        print(f"\n[{name}] ppl@{args.ppl_ctx}={ppl:.2f} exact={int(hit)} robust={int(rhit)} "
              f"needle={needle}", flush=True)
        print(f"    gen: {gen.strip()[:160]!r}", flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
