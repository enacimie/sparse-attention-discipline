"""Format-free probe: does ATTENTION retrieve the fact, independent of generation format?

Generation-based needle recall on base models conflates attention with format compliance.
Here we never ask the model to generate an answer. Instead we SCORE each candidate value as
a continuation of 'The access code for <key> is' and take the argmax of mean token log-prob.
This reads the model's internal preference via likelihood, with no instruction following.

Use: quantify how much of the base model's "dense collapse" was attention vs QA format.
Compare probe_recall (attention-only) vs generation needle_recall at the same contexts.
"""

import argparse
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model_loader import MODELS, inspect_model, load_model  # noqa: E402
from needle import build_multifact_haystack  # noqa: E402
from utils import append_jsonl, set_seed, timer  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "formatfree_probe.jsonl")


@torch.no_grad()
def score_candidates(model, tok, prefix_text, candidates, device):
    """Mean token log-prob of each candidate continuing prefix_text (length-fair)."""
    prefix = tok(prefix_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    P = prefix.shape[1]
    scores = []
    for cand in candidates:
        cids = tok(" " + cand, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        full = torch.cat([prefix, cids], dim=1)
        logits = model(full, use_cache=False).logits[0]          # (L, V)
        lp = 0.0
        for i, tid in enumerate(cids[0]):
            logp = torch.log_softmax(logits[P - 1 + i].float(), dim=-1)
            lp += logp[tid].item()
        scores.append(lp / cids.shape[1])
    return scores


@torch.no_grad()
def run_case(model, tok, ctx, n_facts, depth, seed, device):
    set_seed(seed)
    rng = random.Random(seed)
    body, facts, key, value = build_multifact_haystack(tok, ctx, n_facts, rng, depth)
    prefix = body + f"\nThe access code for {key} is"
    keys = list(facts.keys())
    candidates = [str(facts[k]) for k in keys]
    correct_idx = keys.index(key)
    with timer() as t:
        scores = score_candidates(model, tok, prefix, candidates, device)
    pred = max(range(len(scores)), key=lambda i: scores[i])
    rank = sorted(scores, reverse=True).index(scores[correct_idx]) + 1
    return pred == correct_idx, rank, t["seconds"]


def run(model_key, contexts, n_facts, dtype, depths, seeds, tag):
    model_id = MODELS[model_key]
    info = inspect_model(model_id)
    model, tok = load_model(model_id, dtype=dtype)
    device = next(model.parameters()).device
    print(f"[load] {model_id} full_attn={info.full_attention_layers} tag={tag}")
    for ctx in contexts:
        hits, totrank, totlat, n = 0, 0, 0.0, 0
        for d in depths:
            for s in seeds:
                try:
                    hit, rank, lat = run_case(model, tok, ctx, n_facts, d, s, device)
                except Exception as e:
                    print(f"  ctx={ctx} d={d} s={s} SKIP ({type(e).__name__})")
                    continue
                hits += int(hit); totrank += rank; totlat += lat; n += 1
                print(f"  ctx={ctx} d={d} s={s}: {'OK' if hit else 'no'} rank={rank}")
        if n:
            rec = {"experiment": "formatfree_probe", "model": model_id, "tag": tag,
                   "context": ctx, "n_facts": n_facts, "probe_recall": round(hits/n, 3),
                   "avg_correct_rank": round(totrank/n, 2),
                   "avg_latency_s": round(totlat/n, 3), "n_cases": n}
            append_jsonl(RESULTS, rec)
            print(f"  >> ctx={ctx} PROBE recall={rec['probe_recall']} "
                  f"rank={rec['avg_correct_rank']} lat={rec['avg_latency_s']}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS.keys()))
    ap.add_argument("--contexts", type=int, nargs="+", default=[2048, 4096, 8192])
    ap.add_argument("--n-facts", type=int, default=8)
    ap.add_argument("--depths", type=float, nargs="+", default=[0.2, 0.5, 0.8])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--tag", default="base")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    run(args.model, args.contexts, args.n_facts, args.dtype, args.depths, args.seeds, args.tag)


if __name__ == "__main__":
    main()
