"""LongBench representative subset (truncated) — reviewer-optional benchmark.

The full LongBench needs 4k-16k contexts that do not fit an 8 GB GPU and ~14 datasets.
This runs a representative subset (single-doc QA, multi-doc QA, summarization,
passage count, passage retrieval, code) truncated to `max_len` tokens, on dense vs
local_global_topk vs the structurally-matched random control (local_global_randtopk).

Metrics follow LongBench families but simplified:
  qasper / hotpotqa            -> token F1
  gov_report                   -> ROUGE-L F1
  passage_count / retrieval    -> exact-match accuracy
  lcc                          -> EditSim (SequenceMatcher ratio)

Usage:
  python scripts/run_longbench_subset.py --model qwen3.5-0.8b --max-len 4096 \
      --n-samples 5 --tasks qasper hotpotqa gov_report passage_count passage_retrieval_en lcc
"""
import argparse
import difflib
import json
import os
import random
import re
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from attention_interceptor import apply_method, restore  # noqa: E402
from attention_methods import AttentionMethod  # noqa: E402
from model_loader import MODELS, inspect_model, load_model  # noqa: E402
from utils import append_jsonl, set_seed  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "longbench_subset.jsonl")

TASKS = ["qasper", "hotpotqa", "2wikimqa", "narrativeqa", "passage_retrieval_en", "qmsum"]
GEN_TOKENS = {"qasper": 96, "hotpotqa": 96, "2wikimqa": 96, "narrativeqa": 128,
              "passage_retrieval_en": 32, "qmsum": 256}
QA_TASKS = {"qasper", "hotpotqa", "2wikimqa", "narrativeqa"}
SUM_TASKS = {"qmsum"}


def _norm(s):
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def token_f1(pred, ref):
    p, r = _norm(pred).split(), _norm(ref).split()
    if not p or not r:
        return 0.0
    rc, pc = {}, {}
    for t in r:
        rc[t] = rc.get(t, 0) + 1
    for t in p:
        pc[t] = pc.get(t, 0) + 1
    tp = sum(min(rc.get(t, 0), pc.get(t, 0)) for t in rc)
    pn = sum(pc.values())
    rn = sum(rc.values())
    if pn == 0 or rn == 0:
        return 0.0
    prec, rec = tp / pn, tp / rn
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def rouge_l(pred, ref):
    pw, rw = _norm(pred).split(), _norm(ref).split()
    n, m = len(pw), len(rw)
    if n == 0 or m == 0:
        return 0.0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if pw[i - 1] == rw[j - 1] \
                else max(dp[i - 1][j], dp[i][j - 1])
    L = dp[n][m]
    prec, rec = L / n, L / m
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def score(task, pred, ref):
    if task in QA_TASKS:
        return {"f1": round(token_f1(pred, ref), 4)}
    if task in SUM_TASKS:
        return {"rouge_l": round(rouge_l(pred, ref), 4)}
    return {"acc": float(_norm(pred) == _norm(ref))}


def truncate_middle(tok, text, cap):
    ids = tok(text, add_special_tokens=False).input_ids
    if len(ids) <= cap:
        return text
    head = ids[: cap // 2]
    tail = ids[-(cap - cap // 2):]
    return tok.decode(head + tail)


def build_prompt(task, context, instruction):
    body = context + "\n\n" + instruction
    if task in SUM_TASKS:
        return body + "\n\nSummary:"
    return body + "\n\nAnswer:"


def get_ref(task, sample):
    out = sample.get("output", "")
    answers = sample.get("answers", out)
    if isinstance(out, list):
        out = out[0] if out else ""
    if isinstance(answers, dict):
        spans = answers.get("answer_spans") or answers.get("extractive_spans") or []
        if spans and isinstance(spans[0], list) and len(spans[0]) >= 2:
            out = str(sample.get("input", ""))[spans[0][0]:spans[0][1]]
        else:
            out = str(answers)
    elif isinstance(answers, list) and answers:
        out = str(answers[0])
    return str(out)


def load_samples(data_dir, task):
    path = os.path.join(data_dir, task + ".jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@torch.no_grad()
def gen_answer(model, tok, prompt, device, gen_tokens):
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    out = model.generate(ids, max_new_tokens=gen_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def run(model_key, tasks, n_samples, max_len, methods, dtype, load_in_4bit, data_dir):
    model_id = MODELS[model_key]
    info = inspect_model(model_id)
    model, tok = load_model(model_id, dtype=dtype, load_in_4bit=load_in_4bit)
    device = next(model.parameters()).device
    full_layers = info.full_attention_layers
    print(f"[load] {model_id} full_attn={len(full_layers)} max_len={max_len}")

    for task in tasks:
        ds = load_samples(data_dir, task)
        rng = random.Random(42)
        idx = rng.sample(range(len(ds)), min(n_samples, len(ds)))
        for name, mname in methods:
            hits = 0
            scores = []
            for si in idx:
                sample = ds[si]
                context = str(sample.get("context", "") or "")
                instruction = str(sample.get("input", "") or "")
                if not context.strip() or not instruction.strip():
                    continue
                ref = get_ref(task, sample)
                prompt = build_prompt(task, truncate_middle(tok, context,
                                                            max_len - GEN_TOKENS[task]),
                                      instruction)
                set_seed(si)
                method = None
                if mname == "lg_topk":
                    method = AttentionMethod(name="local_global_topk", window=64,
                                             n_global=4, topk=32)
                elif mname == "lg_randtopk":
                    method = AttentionMethod(name="local_global_randtopk", window=64,
                                             n_global=4, topk=32, seed=si)
                orig = apply_method(model, method, full_layers) if method else {}
                try:
                    pred = gen_answer(model, tok, prompt, device, GEN_TOKENS[task])
                finally:
                    if method:
                        restore(model, orig, full_layers)
                sc = score(task, pred, ref)
                scores.append(sc)
                if "acc" in sc:
                    hits += int(sc["acc"])
                rec = {"experiment": "longbench_subset", "task": task, "method": name,
                       "model": model_id, "max_len": max_len, "sample_idx": si, **sc,
                       "generated": pred[:80], "ref": ref[:80]}
                append_jsonl(RESULTS, rec)
            n = len(scores)
            if n == 0:
                print(f"  {task:22s} {name:16s} no valid samples")
                continue
            mean = {k: round(sum(s[k] for s in scores) / n, 4) for k in scores[0]}
            print(f"  {task:22s} {name:16s} n={n} {mean}"
                  + (f" acc={hits}/{n}" if "acc" in scores[0] else ""))
            append_jsonl(RESULTS, {"experiment": "longbench_subset_summary", "task": task,
                                   "method": name, "model": model_id, "max_len": max_len,
                                   "n": n, **mean})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-0.8b", choices=list(MODELS.keys()))
    ap.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--no-quantize", action="store_true")
    ap.add_argument("--data-dir", default="/tmp/lb/data")
    args = ap.parse_args()
    methods = [("dense", None), ("lg_topk", "lg_topk"), ("lg_randtopk", "lg_randtopk")]
    run(args.model, args.tasks, args.n_samples, args.max_len, methods, args.dtype,
        load_in_4bit=not args.no_quantize, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
