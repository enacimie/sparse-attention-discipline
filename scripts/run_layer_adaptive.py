#!/usr/bin/env python3
"""LAYER-ADAPTIVE pair budget (PoC): does concentrating the content top-k budget on the
retrieval-critical full-attn layers (L15/L19 at 0.8B) and shrinking it on the redundant
ones preserve retrieval/fidelity at a LOWER total pair budget than the uniform recipe?

Motivated by the layer-knockout finding (run_layer_ablation.py): at 0.8B, L15/L19 are
necessary for retrieval while L3/7/11/23 are redundant. If adaptive allocation matches
uniform quality at fewer total pairs, it is a real efficiency win; if not, the redundancy
means uniform is already near-optimal (and the idea is dead at scale, where LOO=1.0).

Per-layer methods are applied by patching disjoint layer subsets (apply_method is called
once per budget group). Fidelity = logit-cosine vs dense; retrieval = robust digit recall;
efficiency = total pair_ratio across the patched layers.
"""
import argparse
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from attention_interceptor import (PAIR_COUNTER, apply_method, reset_pair_counter,  # noqa: E402
                                   restore)
from attention_methods import AttentionMethod  # noqa: E402
from model_loader import MODELS, inspect_model, load_model  # noqa: E402
from needle import build_needle_prompt, make_needle  # noqa: E402
from run_layer_ablation import robust_hit  # noqa: E402
from utils import append_jsonl, fidelity_metrics, set_seed  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "layer_adaptive.jsonl")


def method_topk(k, window=64, n_global=4):
    return AttentionMethod(name="local_global_topk", window=window, n_global=n_global, topk=k)


def apply_perlayer(model, layer_to_method):
    originals = {}
    groups = {}
    for L, m in layer_to_method.items():
        groups.setdefault(id(m), (m, []))[1].append(L)
    for m, layers in groups.values():
        originals.update(apply_method(model, m, layers))
    return originals


def restore_all(model, originals):
    restore(model, originals, list(originals.keys()))


@torch.no_grad()
def get_logits(model, ids):
    return model(ids).logits[0].cpu().float()


def build_chunks(tok, ctx, n_chunks, device):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    text = "\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    step = max(1, (ids.numel() - ctx) // max(1, n_chunks))
    out = [ids[s:s + ctx] for s in range(0, ids.numel() - ctx + 1, step)][:n_chunks]
    return [c.unsqueeze(0).to(device) for c in out]


def fidelity_adaptive(model, tok, layer_to_method, ctx, n_chunks=3):
    chunks = build_chunks(tok, ctx, n_chunks, next(model.parameters()).device)
    reset_pair_counter()
    cos_list = []
    for chunk in chunks:  # one chunk at a time (avoids holding all [ctx,vocab] logits)
        ref = get_logits(model, chunk)
        originals = apply_perlayer(model, layer_to_method)
        try:
            patched = get_logits(model, chunk)
        finally:
            restore_all(model, originals)
        cos_list.append(fidelity_metrics(patched, ref)["logit_cosine"])
    pr = PAIR_COUNTER["computed"] / PAIR_COUNTER["full"] if PAIR_COUNTER["full"] else 0.0
    return sum(cos_list) / len(cos_list), pr


def needle_adaptive(model, tok, layer_to_method, ctx, depths, seeds, gen_tokens=24):
    hits, rhits, total = 0, 0, 0
    device = next(model.parameters()).device
    originals = apply_perlayer(model, layer_to_method)
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
                hits += int(ndl in gen)
                rhits += int(robust_hit(gen, ndl))
                total += 1
    finally:
        restore_all(model, originals)
    n = total if total else 1
    return hits / n, rhits / n, rhits, total


def build_conditions(full, protect):
    """protect = retrieval-critical layers (keep high budget); rest get reduced budget."""
    rest = [L for L in full if L not in protect]
    def lm(k_prot, k_rest):
        return {**{L: method_topk(k_prot) for L in protect},
                **{L: method_topk(k_rest) for L in rest}}
    return [
        ("uniform_32", {L: method_topk(32) for L in full}),
        ("adapt_mild", lm(32, 8)),
        ("adapt_aggr", lm(32, 4)),
        ("concentrate", lm(48, 4)),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-0.8b")
    ap.add_argument("--protect", type=int, nargs="+", default=[15, 19])
    ap.add_argument("--contexts", type=int, nargs="+", default=[2048, 4096])
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
    print(f"[load] {model_id} full_attn={full} protect={args.protect}", flush=True)

    conditions = build_conditions(full, set(args.protect))
    for ctx in args.contexts:
        for name, lm in conditions:
            cos, pr = fidelity_adaptive(model, tok, lm, ctx, n_chunks=3)
            e, r, rh, t = needle_adaptive(model, tok, lm, ctx, args.depths, args.seeds)
            row = {"experiment": "layer_adaptive", "model": args.model, "condition": name,
                   "protect": args.protect, "context": ctx, "cos_mean": round(cos, 4),
                   "pr_mean": round(pr, 4), "needle_robust": round(r, 4),
                   "needle_exact": round(e, 4), "needle_hits": rh, "needle_total": t}
            append_jsonl(RESULTS, row)
            print(f"ctx={ctx} {name:12s} cos={cos:.4f} pr={pr:.4f} "
                  f"needle={r:.3f} ({rh}/{t})", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
