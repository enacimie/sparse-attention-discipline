"""RULER-style multi-task battery: NIAH variants + variable tracking + frequent-word.

Runs the task families in src/ruler_tasks.py (niah_single, niah_multikey, niah_multivalue,
niah_multiquery, vt, fwe) across dense / local_global_topk / budget-matched StreamingLLM /
budget-matched uniform_random, scoring exact-answer recall. This is the broader downstream
evidence beyond the single multi-fact subset, covering retrieval, reasoning and extraction.

Usage:
  python scripts/run_ruler_battery.py --model qwen3.5-0.8b --contexts 2048 \
      --tasks niah_single vt fwe --seeds 42 43 44
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
from ruler_tasks import TASKS  # noqa: E402
from utils import append_jsonl, set_seed  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "ruler_battery.jsonl")


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


import re


def _norm(s):
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


@torch.no_grad()
def run_one(task, model, tok, ctx, seed, gen_tokens=24):
    set_seed(seed)
    rng = random.Random(seed)
    body, question, targets = TASKS[task](tok, ctx, rng)
    prompt = body + "\n\nQuestion: " + question + "\nAnswer:"
    ids = tok(prompt, return_tensors="pt", truncation=True,
              max_length=ctx).input_ids.to(next(model.parameters()).device)
    gt = 64 if task.startswith("qa") else gen_tokens
    out = model.generate(ids, max_new_tokens=gt, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    gn = _norm(gen)
    hit = all(_norm(t) in gn for t in targets)
    return hit, gen, targets


def run(model_key, contexts, tasks, seeds, dtype, gen_tokens, topk_budget, load_in_4bit,
        lg_topk=32, only_methods=None, out=None, device_map="cuda"):
    results_path = out or RESULTS
    model_id = MODELS[model_key]
    info = inspect_model(model_id)
    model, tok = load_model(model_id, dtype=dtype, load_in_4bit=load_in_4bit,
                            device_map=device_map)
    full_layers = info.full_attention_layers
    print(f"[load] {model_id} full_attn={len(full_layers)} lg_topk={lg_topk}")
    for ctx in contexts:
        for task in tasks:
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
                        hit, gen, targets = run_one(task, model, tok, ctx, seed, gen_tokens)
                        hits += int(hit)
                        append_jsonl(results_path, {"experiment": "ruler_battery", "task": task,
                                                "method": name, "model": model_id,
                                                "context": ctx, "seed": seed, "hit": bool(hit),
                                                "targets": [str(t) for t in targets],
                                                "generated": gen[:80]})
                finally:
                    if method:
                        restore(model, originals, full_layers)
                print(f"  ctx={ctx} {task:16s} {name:16s} recall={hits/len(seeds):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-0.8b", choices=list(MODELS.keys()))
    ap.add_argument("--contexts", type=int, nargs="+", default=[2048])
    ap.add_argument("--tasks", nargs="+", default=list(TASKS.keys()))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--topk-budget", type=int, default=96)
    ap.add_argument("--lg-topk", type=int, default=32)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--gen-tokens", type=int, default=24)
    ap.add_argument("--no-quantize", action="store_true")
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--out", default=None,
                    help="ruta JSONL de salida; por defecto results/ruler_battery.jsonl")
    ap.add_argument("--device-map", default="cuda",
                    help="'cuda' (1 GPU) o 'auto' (reparte entre varias)")
    args = ap.parse_args()
    run(model_key=args.model, contexts=args.contexts, tasks=args.tasks, seeds=args.seeds,
        dtype=args.dtype, gen_tokens=args.gen_tokens, topk_budget=args.topk_budget,
        load_in_4bit=not args.no_quantize, lg_topk=args.lg_topk, only_methods=args.methods,
        out=args.out, device_map=args.device_map)


if __name__ == "__main__":
    main()
