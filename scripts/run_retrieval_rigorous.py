"""Rigorous external-retrieval validation (post random-control correction).

Answers: does retrieval EXTERNally beat dense attention robustly, and does it survive
(a) lexical distractors, (b) a random-chunk control, (c) longer contexts?

Task: multi-fact 'access code' lookup. One target fact plus distractors whose keys share
a token with the target, so retrieval must discriminate the exact key (not keyword-match).

Methods compared per (context, target_depth, seed):
  - dense      : feed the FULL haystack + question (upper bound of pure attention).
  - bm25       : retrieve top-k chunks by BM25, feed only those + question.
  - dense_ret  : retrieve top-k chunks by dense embeddings, feed only those + question.
  - random     : CONTROL, feed top-k RANDOM chunks + question.

Diagnostics reported per method:
  - retrieval_hit : did the fed context actually contain the target fact? (retrieval quality)
  - needle_recall : did the generation contain the target value? (end-to-end)
  - fed_tokens, latency_s.

Usage:
  python scripts/run_retrieval_rigorous.py --model lfm2.5-350m-base \
      --contexts 2048 4096 --n-facts 8 --top-k 4 --chunk-tokens 128
"""

import argparse
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model_loader import MODELS, inspect_model, load_model  # noqa: E402
from needle import build_multifact_haystack  # noqa: E402
from retrieval import (chunk_text_by_tokens, bm25_retrieve, dense_retrieve,  # noqa: E402
                       random_retrieve, mean_pool_embed_fn, contextual_embed_fn,
                       sentence_transformer_embed_fn)
from utils import GpuMeter, append_jsonl, set_seed, timer  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "retrieval_rigorous.jsonl")


def question_for(key):
    return f"What is the access code for {key}?"


def format_prompt(tok, context_text, question, chat):
    """Raw QA prompt for base models; chat-template prompt for instruct models.

    For thinking models (e.g. Qwen3.5) we disable thinking, otherwise the reasoning
    preamble consumes the generation budget and needle recall reads 0 spuriously.
    """
    if not chat:
        return context_text + "\n\nQuestion: " + question + "\nAnswer:"
    messages = [{"role": "user", "content": context_text + "\n\n" + question}]
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except Exception:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def gen_from(model, tok, prompt, gen_tokens):
    ids = tok(prompt, return_tensors="pt").input_ids.to(next(model.parameters()).device)
    fed = ids.shape[1]
    with GpuMeter() as m:
        out = model.generate(ids, max_new_tokens=gen_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0][fed:], skip_special_tokens=True)
    return gen, fed, m


def fact_present(text, key, value):
    return (key in text) and (str(value) in text)


@torch.no_grad()
def run_case(model, tok, embed_fns, ctx, n_facts, target_depth, seed,
             chunk_tokens, top_k, gen_tokens, chat, retrievers):
    set_seed(seed)
    rng = random.Random(seed)
    body, facts, key, value = build_multifact_haystack(tok, ctx, n_facts, rng, target_depth)
    question = question_for(key)
    chunks = chunk_text_by_tokens(tok, body, chunk_tokens)
    out = {}

    # dense baseline: full haystack + question
    with timer() as t:
        gen, fed, m = gen_from(model, tok, format_prompt(tok, body, question, chat), gen_tokens)
    out["dense"] = {"hit": True, "recall": str(value) in gen, "fed": fed,
                    "lat": t["seconds"], "vram": m.peak_vram_mb}

    builders = {
        "bm25": lambda: bm25_retrieve(question, chunks, top_k),
        "dense_ret": lambda: dense_retrieve(question, chunks, top_k, embed_fns["input"]),
        "contextual": lambda: dense_retrieve(question, chunks, top_k, embed_fns["contextual"]),
        "st": lambda: dense_retrieve(question, chunks, top_k, embed_fns["st"]),
        "random": lambda: random_retrieve(chunks, top_k, rng),
    }
    for name in retrievers:
        with timer() as t:
            idx = builders[name]()
            retrieved = "\n\n".join(chunks[i] for i in idx)
            hit = fact_present(retrieved, key, value)
            gen, fed, m = gen_from(model, tok, format_prompt(tok, retrieved, question, chat),
                                   gen_tokens)
        out[name] = {"hit": hit, "recall": str(value) in gen, "fed": fed,
                     "lat": t["seconds"], "vram": m.peak_vram_mb}
    return out, len(chunks)


def run(model_key, contexts, n_facts, top_k, chunk_tokens, dtype, gen_tokens,
        depths, seeds, chat, retrievers, tag):
    model_id = MODELS[model_key]
    info = inspect_model(model_id)
    model, tok = load_model(model_id, dtype=dtype)
    device = next(model.parameters()).device
    embed_fns = {}
    if "dense_ret" in retrievers:
        embed_fns["input"] = mean_pool_embed_fn(model, tok, device)
    if "contextual" in retrievers:
        embed_fns["contextual"] = contextual_embed_fn(model, tok, device)
    if "st" in retrievers:
        embed_fns["st"] = sentence_transformer_embed_fn(device=str(device))
    print(f"[load] {model_id} full_attn={info.full_attention_layers} chat={chat} tag={tag}")

    methods = ["dense"] + list(retrievers)
    for ctx in contexts:
        agg = {mth: {"hit": 0, "recall": 0, "fed": 0, "lat": 0.0, "n": 0} for mth in methods}
        for d in depths:
            for s in seeds:
                try:
                    res, n_chunks = run_case(model, tok, embed_fns, ctx, n_facts, d, s,
                                             chunk_tokens, top_k, gen_tokens, chat, retrievers)
                except Exception as e:  # position-embed / OOM guard at long ctx
                    print(f"  ctx={ctx} depth={d} seed={s} SKIP ({type(e).__name__})")
                    continue
                for mth in methods:
                    r = res[mth]
                    agg[mth]["hit"] += int(r["hit"]); agg[mth]["recall"] += int(r["recall"])
                    agg[mth]["fed"] += r["fed"]; agg[mth]["lat"] += r["lat"]
                    agg[mth]["n"] += 1
                print(f"  ctx={ctx} d={d} s={s}: " +
                      " ".join(f"{m}={'HIT' if res[m]['hit'] else 'miss'}/"
                               f"{'OK' if res[m]['recall'] else 'no'}" for m in methods))
        for mth in methods:
            a = agg[mth]
            if a["n"] == 0:
                continue
            rec = {"experiment": "retrieval_rigorous", "method": mth, "model": model_id,
                   "tag": tag, "chat": chat, "context": ctx, "n_facts": n_facts,
                   "top_k": top_k, "chunk_tokens": chunk_tokens,
                   "retrieval_hit": round(a["hit"]/a["n"], 3),
                   "needle_recall": round(a["recall"]/a["n"], 3),
                   "avg_fed_tokens": round(a["fed"]/a["n"], 1),
                   "avg_latency_s": round(a["lat"]/a["n"], 3), "n_cases": a["n"]}
            append_jsonl(RESULTS, rec)
            print(f"  >> ctx={ctx} {mth}: hit={rec['retrieval_hit']} "
                  f"recall={rec['needle_recall']} fed={rec['avg_fed_tokens']} "
                  f"lat={rec['avg_latency_s']}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS.keys()))
    ap.add_argument("--contexts", type=int, nargs="+", default=[2048, 4096])
    ap.add_argument("--n-facts", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--chunk-tokens", type=int, default=128)
    ap.add_argument("--depths", type=float, nargs="+", default=[0.2, 0.5, 0.8])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--retrievers", nargs="+",
                    default=["bm25", "dense_ret", "contextual", "random"],
                    choices=["bm25", "dense_ret", "contextual", "st", "random"])
    ap.add_argument("--chat", action="store_true", help="use chat template (instruct models)")
    ap.add_argument("--tag", default="base", help="label for JSONL records")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--gen-tokens", type=int, default=16)
    args = ap.parse_args()
    run(args.model, args.contexts, args.n_facts, args.top_k, args.chunk_tokens,
        args.dtype, args.gen_tokens, args.depths, args.seeds, args.chat,
        args.retrievers, args.tag)


if __name__ == "__main__":
    main()
