"""Reviewer M4: needle-selection diagnosis (why content top-k preserves retrieval).

Fidelity shows content top-k adds only ~+0.02, yet retrieval jumps 0.0 -> dense. This
measures the mechanism directly on the needle task (0.8B, ctx 2048): for every full-attn
layer and head, is the needle key inside (a) the local window, (b) the sinks, (c) the
content top-k, (d) the random top-k; what softmax attention weight does the needle receive
(lg_topk vs lg_randtopk vs dense); and is generation recall CONDITIONED on the needle being
selected by the last query's top-k.

Uses the production interceptor (patched forwards + the optional RECORDER hook), so the
selection stats are measured on the exact tensors the masked attention uses.

Output: results/needle_diag.jsonl (per sample) + printed aggregate.

Usage:
  python scripts/run_needle_diag.py --model qwen3.5-0.8b --ctx 2048 --topk 32
"""
import argparse
import math
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from attention_interceptor import (apply_method, clear_recorder, restore,
                                   set_recorder)  # noqa: E402
from attention_methods import AttentionMethod  # noqa: E402
from model_loader import MODELS, inspect_model, load_model  # noqa: E402
from needle import FILLER as FILLER_TEXT  # noqa: E402
from needle import check_needle, make_needle  # noqa: E402
from utils import append_jsonl, set_seed  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "needle_diag.jsonl")
QUESTION = "What is the special passkey mentioned in the text?"


def build_haystack(tok, ctx, needle, depth):
    needle_sentence = f"The special passkey is {needle}."
    needle_ids = tok(" " + needle_sentence + " ", add_special_tokens=False).input_ids
    n_rep = ctx // 30 + 5
    filler_ids = tok(FILLER_TEXT * n_rep, add_special_tokens=False).input_ids
    filler_ids = filler_ids[: max(ctx - len(needle_ids), 0)]
    pos = int(len(filler_ids) * depth)
    ids = filler_ids[:pos] + needle_ids + filler_ids[pos:]
    code_start = pos + len(tok(" The special passkey is ", add_special_tokens=False).input_ids)
    code_len = len(tok(" " + needle + " ", add_special_tokens=False).input_ids)
    return ids, code_start, code_start + code_len


def make_recorder(needle_span, full_layers, records):
    np_start, np_end = needle_span
    for i in full_layers:
        records[i] = {"local_last": [], "sink_last": [], "topk_last": [],
                      "frac_queries_topk": [], "attn_last": [], "attn_last_max": [],
                      "attn_all": []}

    def rec(layer_idx, q, k, keep):
        if layer_idx not in records:
            return
        rows = records[layer_idx]
        q = q[0].float()
        k = k[0].float()
        q_len, k_len = q.shape[1], k.shape[1]
        if q_len != k_len:
            return  # decode-ish call; stats belong to the prefill call
        last = q_len - 1
        local_last = bool(keep[last, np_start:np_end].any())
        rows["local_last"].append(local_last and np_start > last - 64)
        rows["sink_last"].append(local_last and np_start < 4)
        rows["topk_last"].append(local_last)
        rows["frac_queries_topk"].append(float(keep[:, np_start:np_end].any(dim=1).float().mean()))
        n_rep = q.shape[0] // max(1, k.shape[0])
        kr = k.repeat_interleave(n_rep, dim=0)
        scores = torch.einsum("hqd,hkd->hqk", q, kr) / math.sqrt(q.shape[-1])
        soft = torch.softmax(scores.masked_fill(~keep.unsqueeze(0), float("-inf")), dim=-1)
        attn_np = soft[:, :, np_start:np_end].sum(dim=-1)  # (h, q_len)
        rows["attn_last"].append(float(attn_np[:, last].mean()))
        rows["attn_last_max"].append(float(attn_np[:, last].max()))
        rows["attn_all"].append(float(attn_np.mean()))

    return rec


@torch.no_grad()
def gen_answer(model, tok, prompt, device, gen_tokens=24):
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    out = model.generate(ids, max_new_tokens=gen_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def run(model_key, ctx, depths, seeds, topk, dtype, load_in_4bit):
    model_id = MODELS[model_key]
    info = inspect_model(model_id)
    model, tok = load_model(model_id, dtype=dtype, load_in_4bit=load_in_4bit)
    device = next(model.parameters()).device
    full_layers = info.full_attention_layers
    print(f"[load] {model_id} full_attn={len(full_layers)}")

    methods = {
        "dense": AttentionMethod(name="dense"),
        "lg_topk": AttentionMethod(name="local_global_topk", window=64, n_global=4, topk=topk),
        "lg_randtopk": AttentionMethod(name="local_global_randtopk", window=64, n_global=4,
                                       topk=topk, seed=0),
    }
    recs = []
    for depth in depths:
        for seed in seeds:
            set_seed(seed)
            rng = random.Random(seed)
            needle = make_needle(rng)
            ids, np_start, np_end = build_haystack(tok, ctx, needle, depth)
            prompt = tok.decode(ids) + "\nQuestion: " + QUESTION + "\nAnswer:"
            pids = tok(prompt, return_tensors="pt").input_ids.to(device)
            records = {}
            for name, method in methods.items():
                rec = make_recorder((np_start, np_end), full_layers, records)
                set_recorder(rec)
                orig = apply_method(model, method, full_layers)
                try:
                    model(pids, use_cache=False)
                finally:
                    restore(model, orig, full_layers)
                clear_recorder()
                agg = {k: float(sum(v) / len(v)) for k, v in records[full_layers[0]].items()}
                r = {"experiment": "needle_diag", "model": model_id, "context": ctx,
                     "depth": depth, "seed": seed, "method": name, "needle": needle, **agg}
                recs.append(r)
                print(f"  depth={depth} seed={seed} {name:12s} "
                      f"topk_last={agg['topk_last']:.2f} attn_last={agg['attn_last']:.4f}")

            rec_dense, rec_lg, rec_rand = recs[-3], recs[-2], recs[-1]
            for name, target in (("lg_topk", rec_lg), ("lg_randtopk", rec_rand)):
                orig = apply_method(model, methods[name], full_layers)
                try:
                    g = gen_answer(model, tok, prompt, device)
                finally:
                    restore(model, orig, full_layers)
                target["recall"] = int(check_needle(g, needle))
            g = gen_answer(model, tok, prompt, device)
            rec_dense["recall"] = int(check_needle(g, needle))
            for r in (rec_dense, rec_lg, rec_rand):
                append_jsonl(RESULTS, r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-0.8b", choices=list(MODELS.keys()))
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46, 47])
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--no-quantize", action="store_true")
    args = ap.parse_args()
    run(args.model, args.ctx, args.depths, args.seeds, args.topk, args.dtype,
        load_in_4bit=not args.no_quantize)


if __name__ == "__main__":
    main()
