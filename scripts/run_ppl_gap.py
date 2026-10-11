"""PPL gap introduced by the sparse attention method (training-free), step 1 of the LoRA study.

Measures teacher-forced perplexity on wikitext-2 for:
  - the dense model (reference)
  - the same model with `local_global_topk` applied to its full_attention layers

The gap tells us how much quality the sparsity costs before any fine-tuning, i.e. how much a
LoRA/QLoRA adaptation would need to recover. Absolute PPL is only a reference (use -Base models);
the DELTA dense-vs-sparse is the quantity of interest.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from attention_interceptor import apply_method, restore  # noqa: E402
from attention_methods import AttentionMethod  # noqa: E402
from model_loader import MODELS, inspect_model, load_model  # noqa: E402
from utils import set_seed  # noqa: E402


def get_eval_chunks(tok, seq_len, n_docs):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    chunks = [ids[i:i + seq_len]
              for i in range(0, min(n_docs * seq_len, ids.numel() - seq_len - 1), seq_len)]
    return [c for c in chunks if c.numel() == seq_len]


@torch.no_grad()
def perplexity(model, chunks, device):
    nll, ntok = 0.0, 0
    for c in chunks:
        ids = c.unsqueeze(0).to(device)
        out = model(ids, labels=ids)
        n = ids.shape[1] - 1
        nll += out.loss.item() * n
        ntok += n
    return float(np.exp(nll / ntok))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-4b-base", choices=list(MODELS.keys()))
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--n-docs", type=int, default=8)
    ap.add_argument("--device-map", default="cuda")
    args = ap.parse_args()
    set_seed(0)

    model_id = MODELS[args.model]
    info = inspect_model(model_id)
    model, tok = load_model(model_id, dtype="float16", load_in_4bit=True,
                            device_map=args.device_map)
    device = next(model.parameters()).device
    chunks = get_eval_chunks(tok, args.seq_len, args.n_docs)
    print(f"[data] {len(chunks)} chunks of {args.seq_len} on {model_id}", flush=True)

    dense_ppl = perplexity(model, chunks, device)
    print(f"dense               PPL = {dense_ppl:.3f}", flush=True)

    method = AttentionMethod(name="local_global_topk", window=64, n_global=4, topk=32)
    originals = apply_method(model, method, info.full_attention_layers)
    sparse_ppl = perplexity(model, chunks, device)
    restore(model, originals, info.full_attention_layers)
    print(f"local_global_topk   PPL = {sparse_ppl:.3f}   "
          f"(gap {sparse_ppl - dense_ppl:+.3f}, {100 * (sparse_ppl / dense_ppl - 1):+.2f}%)",
          flush=True)


if __name__ == "__main__":
    main()
