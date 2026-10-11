"""Run external-retrieval needle evaluation: retrieve chunks, answer from top-k only.

Compares BM25 vs dense chunk embeddings. Measures needle recall and the context
length actually fed to the model (much shorter than full -> cheaper than dense).

Usage:
  python scripts/run_retrieval.py --model lfm2.5-350m-base --contexts 2048 \
      --chunk-tokens 128 --top-k 4
"""

import argparse
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model_loader import MODELS, inspect_model, load_model  # noqa: E402
from needle import build_needle_prompt, check_needle, make_needle  # noqa: E402
from retrieval import chunk_text_by_tokens, bm25_retrieve, dense_retrieve  # noqa: E402
from utils import GpuMeter, append_jsonl, set_seed  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "retrieval.jsonl")

QUESTION = "What is the special passkey mentioned in the text?"


def make_embed_fn(model, tok, device):
    """Mean-pool the model's input embeddings over chunk tokens (cheap, no full forward)."""
    emb = model.get_input_embeddings().weight  # (vocab, d)

    @torch.no_grad()
    def embed(texts):
        out = []
        for t in texts:
            ids = tok(t, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            e = emb[ids].mean(dim=1)  # (1, d)
            out.append(e.float().cpu())
        return torch.cat(out, dim=0).numpy()
    return embed


@torch.no_grad()
def run_one(model, tok, ctx, depth, seed, method, chunk_tokens, top_k, embed_fn, gen_tokens=24):
    set_seed(seed)
    rng = random.Random(seed)
    needle = make_needle(rng)
    full_prompt, needle = build_needle_prompt(tok, ctx, needle, depth, question=QUESTION)
    # strip the question from the body; we re-add after retrieval
    body = full_prompt.split("\n\nQuestion:")[0]
    chunks = chunk_text_by_tokens(tok, body, chunk_tokens)
    if method == "bm25":
        idx = bm25_retrieve(QUESTION, chunks, top_k)
    else:
        idx = dense_retrieve(QUESTION, chunks, top_k, embed_fn)
    retrieved = "\n\n".join(chunks[i] for i in idx)
    prompt = retrieved + "\n\nQuestion: " + QUESTION + "\nAnswer:"
    ids = tok(prompt, return_tensors="pt").input_ids.to(next(model.parameters()).device)
    fed_tokens = ids.shape[1]
    with GpuMeter() as m:
        out = model.generate(ids, max_new_tokens=gen_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0][fed_tokens:], skip_special_tokens=True)
    return check_needle(gen, needle), fed_tokens, len(chunks), m.to_dict()


def run(model_key, contexts, chunk_tokens, top_k, dtype, gen_tokens):
    model_id = MODELS[model_key]
    info = inspect_model(model_id)
    model, tok = load_model(model_id, dtype=dtype)
    device = next(model.parameters()).device
    embed_fn = make_embed_fn(model, tok, device)
    print(f"[load] {model_id} full_attn={info.full_attention_layers}")
    depths, seeds = [0.1, 0.5, 0.9], [42, 43, 44]

    for ctx in contexts:
        for method in ["bm25", "dense"]:
            hits, fed, nch, tot = 0, 0, 0, 0
            for d in depths:
                for s in seeds:
                    hit, fed_tokens, n_chunks, gpu = run_one(
                        model, tok, ctx, d, s, method, chunk_tokens, top_k, embed_fn, gen_tokens)
                    hits += int(hit); fed += fed_tokens; nch += n_chunks; tot += 1
            rec = {"experiment": "retrieval", "retrieval": method, "model": model_id,
                   "context": ctx, "chunk_tokens": chunk_tokens, "top_k": top_k,
                   "needle_recall": round(hits/tot, 3), "avg_fed_tokens": round(fed/tot, 1),
                   "avg_chunks": round(nch/tot, 1),
                   "context_reduction": round((fed/tot)/ctx, 3)}
            append_jsonl(RESULTS, rec)
            print(f"  ctx={ctx} {method}: recall={rec['needle_recall']} "
                  f"fed={rec['avg_fed_tokens']}/{ctx} ({rec['context_reduction']*100:.0f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS.keys()))
    ap.add_argument("--contexts", type=int, nargs="+", default=[2048])
    ap.add_argument("--chunk-tokens", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--gen-tokens", type=int, default=24)
    args = ap.parse_args()
    run(args.model, args.contexts, args.chunk_tokens, args.top_k, args.dtype, args.gen_tokens)


if __name__ == "__main__":
    main()
