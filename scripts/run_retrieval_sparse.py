"""BM25 mitigation of the sparse-attention retrieval failure.

Finding so far: content-topk sparse attention breaks exact needle retrieval (recall 0.00) even at
high budget, while dense gets 1.00. Mitigation hypothesis: combine sparse attention with cheap
external retrieval (BM25) -- retrieve the relevant chunks, then run sparse attention on the short
retrieved context. Conditions per (ctx, depth, seed):
  dense_full     dense attention on the full haystack            (upper bound)
  sparse_full    local_global_topk on the full haystack          (the failure)
  bm25_dense     BM25 top-k chunks + dense attention             (retrieval alone)
  bm25_sparse    BM25 top-k chunks + local_global_topk           (the mitigation)
Reports recall and the fed-token ratio (retrieved tokens / full context).

Usage:
  python scripts/run_retrieval_sparse.py --model qwen3.5-4b-base --contexts 2048 \
      --depths 0.1 0.5 0.9 --seeds 42 43 44 --top-k 4 --chunk-tokens 128
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
from needle import FILLER, check_needle, make_needle  # noqa: E402
from retrieval import bm25_retrieve, chunk_text_by_tokens  # noqa: E402
from utils import append_jsonl, set_seed  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "retrieval_sparse.jsonl")
QUESTION = "What is the special passkey mentioned in the text?"


def build_haystack(tok, ctx, needle, depth):
    needle_sentence = f"The special passkey is {needle}."
    needle_ids = tok(" " + needle_sentence + " ", add_special_tokens=False).input_ids
    n_rep = ctx // 30 + 5
    filler_ids = tok(FILLER * n_rep, add_special_tokens=False).input_ids
    filler_ids = filler_ids[: max(ctx - len(needle_ids), 0)]
    n_filler = len(filler_ids)
    pos = int(n_filler * depth)
    ids = filler_ids[:pos] + needle_ids + filler_ids[pos:]
    return tok.decode(ids)


@torch.no_grad()
def gen_answer(model, tok, prompt, device, gen_tokens=24):
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    out = model.generate(ids, max_new_tokens=gen_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True), ids.shape[1]


def run(model_key, contexts, depths, seeds, dtype, top_k, chunk_tokens, load_in_4bit):
    model_id = MODELS[model_key]
    info = inspect_model(model_id)
    model, tok = load_model(model_id, dtype=dtype, load_in_4bit=load_in_4bit)
    device = next(model.parameters()).device
    full_layers = info.full_attention_layers
    method = AttentionMethod(name="local_global_topk", window=64, n_global=4, topk=32)
    print(f"[load] {model_id} full_attn={len(full_layers)}")

    for ctx in contexts:
        for depth in depths:
            tally = {c: 0 for c in ["dense_full", "sparse_full", "bm25_dense", "bm25_sparse"]}
            fed_ratios = []
            retr_hits = 0
            for seed in seeds:
                set_seed(seed)
                rng = random.Random(seed)
                needle = make_needle(rng)
                haystack = build_haystack(tok, ctx, needle, depth)
                full_prompt = haystack + "\nQuestion: " + QUESTION + "\nAnswer:"
                chunks = chunk_text_by_tokens(tok, haystack, chunk_tokens)
                from rank_bm25 import BM25Okapi
                import numpy as np
                _bm = BM25Okapi([c.lower().split() for c in chunks])
                _sc = _bm.get_scores(QUESTION.lower().split())
                _top = np.argsort(_sc)[::-1][:top_k]
                # relevance order with the MOST relevant chunk LAST (adjacent to the question),
                # so the needle lands in the sparse method's local window
                _ord = sorted(_top.tolist(), key=lambda i: _sc[i])
                retrieved = " ".join(chunks[i] for i in _ord)
                retr_prompt = retrieved + "\nQuestion: " + QUESTION + "\nAnswer:"
                fed = len(tok(retr_prompt).input_ids)
                fed_ratios.append(fed / ctx)
                needle_in_retr = needle in retrieved
                retr_hits += int(needle_in_retr)

                # dense_full
                g, _ = gen_answer(model, tok, full_prompt, device)
                tally["dense_full"] += int(check_needle(g, needle))
                # sparse_full
                orig = apply_method(model, method, full_layers)
                try:
                    g, _ = gen_answer(model, tok, full_prompt, device)
                    tally["sparse_full"] += int(check_needle(g, needle))
                finally:
                    restore(model, orig, full_layers)
                # bm25_dense
                g, _ = gen_answer(model, tok, retr_prompt, device)
                tally["bm25_dense"] += int(check_needle(g, needle))
                # bm25_sparse
                orig = apply_method(model, method, full_layers)
                try:
                    g, _ = gen_answer(model, tok, retr_prompt, device)
                    tally["bm25_sparse"] += int(check_needle(g, needle))
                finally:
                    restore(model, orig, full_layers)

                append_jsonl(RESULTS, {"experiment": "retrieval_sparse", "model": model_id,
                                       "context": ctx, "depth": depth, "seed": seed,
                                       "needle": needle, "top_k": top_k,
                                       "fed_ratio": round(fed / ctx, 4)})
            n = len(seeds)
            fr = sum(fed_ratios) / len(fed_ratios)
            line = (f"  ctx={ctx} depth={depth} fed={fr:.3f} bm25got={retr_hits}/{n} | "
                    + " ".join(f"{c}={tally[c]}/{n}" for c in tally))
            print(line)
            append_jsonl(RESULTS, {"experiment": "retrieval_sparse_summary", "model": model_id,
                                   "context": ctx, "depth": depth, "n": n,
                                   "fed_ratio": round(fr, 4), "bm25_retrieved": retr_hits / n,
                                   **{c: tally[c] / n for c in tally}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-4b-base", choices=list(MODELS.keys()))
    ap.add_argument("--contexts", type=int, nargs="+", default=[2048])
    ap.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--chunk-tokens", type=int, default=128)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--no-quantize", action="store_true")
    args = ap.parse_args()
    run(args.model, args.contexts, args.depths, args.seeds, args.dtype, args.top_k,
        args.chunk_tokens, load_in_4bit=not args.no_quantize)


if __name__ == "__main__":
    main()
