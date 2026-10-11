"""RULER-style downstream subset: multi-fact retrieval with lexical distractors.

Uses needle.build_multifact_haystack: n_facts 'access code' sentences are hidden in neutral
filler; distractor keys share an adjective or noun with the target key, so the model must
discriminate the exact key, not just keyword-match. We ask for the target key's code and score
whether the exact value is generated. Compares dense vs local_global_topk vs budget-matched
uniform_random. This is the downstream/retrieval evidence beyond the single needle probe.

Usage:
  python scripts/run_ruler_subset.py --model qwen3.5-4b-base --contexts 2048 \
      --n-facts 4 8 --seeds 42 43 44
"""
import argparse
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from attention_interceptor import apply_method, restore  # noqa: E402
from attention_methods import AttentionMethod  # noqa: E402
from model_loader import MODELS, inspect_model, load_model  # noqa: E402
from needle import build_multifact_haystack  # noqa: E402
from utils import append_jsonl, set_seed  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "ruler_subset.jsonl")


def build_methods(topk_budget, lg_topk=32):
    return [
        ("dense", None),
        (f"lg_topk_k{lg_topk}", AttentionMethod(name="local_global_topk", window=64,
                                                 n_global=4, topk=lg_topk)),
        (f"lg_randtopk_k{lg_topk}", AttentionMethod(name="local_global_randtopk", window=64,
                                                     n_global=4, topk=lg_topk)),
        ("streamingllm_bm", AttentionMethod(name="local_global", window=max(topk_budget - 4, 4),
                                             n_global=4)),
        ("uniform_random_m", AttentionMethod(name="uniform_random", topk=topk_budget,
                                              random_k=0)),
    ]


@torch.no_grad()
def run_one(model, tok, ctx, n_facts, seed, gen_tokens=16):
    set_seed(seed)
    rng = random.Random(seed)
    body, facts, target_key, target_value = build_multifact_haystack(tok, ctx, n_facts, rng)
    prompt = (body + f"\n\nQuestion: What is the access code for {target_key}?\nAnswer:")
    ids = tok(prompt, return_tensors="pt", truncation=True,
              max_length=ctx).input_ids.to(next(model.parameters()).device)
    out = model.generate(ids, max_new_tokens=gen_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    hit = str(target_value) in gen
    return hit, gen, target_key, target_value


def run(model_key, contexts, n_facts_list, seeds, dtype, gen_tokens, topk_budget, load_in_4bit,
        lg_topk=32, only_methods=None):
    model_id = MODELS[model_key]
    info = inspect_model(model_id)
    model, tok = load_model(model_id, dtype=dtype, load_in_4bit=load_in_4bit)
    full_layers = info.full_attention_layers
    print(f"[load] {model_id} full_attn={len(full_layers)} lg_topk={lg_topk}")
    for ctx in contexts:
        for n_facts in n_facts_list:
            for name, method in build_methods(topk_budget, lg_topk):
                if only_methods and name not in only_methods:
                    continue
                originals = apply_method(model, method, full_layers) if method else {}
                hits = 0
                try:
                    for seed in seeds:
                        if method is not None and method.name in ("uniform_random",
                                                                   "local_global_randtopk"):
                            method.seed = seed
                        hit, gen, tk, tv = run_one(model, tok, ctx, n_facts, seed, gen_tokens)
                        hits += int(hit)
                        append_jsonl(RESULTS, {"experiment": "ruler_subset", "method": name,
                                                "model": model_id, "context": ctx,
                                                "n_facts": n_facts, "seed": seed, "hit": bool(hit),
                                                "target_key": tk, "target_value": tv,
                                                "generated": gen[:60]})
                finally:
                    if method:
                        restore(model, originals, full_layers)
                print(f"  ctx={ctx} facts={n_facts} {name:18s} recall={hits/len(seeds):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-4b-base", choices=list(MODELS.keys()))
    ap.add_argument("--contexts", type=int, nargs="+", default=[2048])
    ap.add_argument("--n-facts", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--topk-budget", type=int, default=96)
    ap.add_argument("--lg-topk", type=int, default=32)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--gen-tokens", type=int, default=16)
    ap.add_argument("--no-quantize", action="store_true")
    ap.add_argument("--methods", nargs="+", default=None)
    args = ap.parse_args()
    run(model_key=args.model, contexts=args.contexts, n_facts_list=args.n_facts, seeds=args.seeds,
        dtype=args.dtype, gen_tokens=args.gen_tokens, topk_budget=args.topk_budget,
        load_in_4bit=not args.no_quantize, lg_topk=args.lg_topk, only_methods=args.methods)


if __name__ == "__main__":
    main()
