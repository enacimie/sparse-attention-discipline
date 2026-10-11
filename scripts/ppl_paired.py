"""W3/W4: paired bootstrap CI on the PPL delta (sparse vs dense) for a given base.

Dense and sparse are computed on the SAME 2048-token chunks in the same pass, so
the delta CI is a paired comparison (much tighter than the unpaired chunk CI).
"""
import sys, os, torch, argparse
import numpy as np
sys.path.insert(0, "/root/mlattention/src")
from attention_interceptor import apply_method, restore
from attention_methods import AttentionMethod
from model_loader import load_model, inspect_model
from utils import append_jsonl

MODEL_PATH = "/root/qwen38"
OUT = "/root/mlattention/results/qwen38_ppl_paired.jsonl"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", choices=["q4", "16bit"], required=True)
    ap.add_argument("--max_chunks", type=int, default=100)
    args = ap.parse_args()

    if args.base == "q4":
        model, tok = load_model(MODEL_PATH, load_in_4bit=True)
    else:
        model, tok = load_model(MODEL_PATH, dtype="bfloat16")
    info = inspect_model(MODEL_PATH)
    fl = info.full_attention_layers
    dev = next(model.parameters()).device

    from datasets import load_dataset
    wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(t for t in wt["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]

    chunk = 2048
    nchunks = min((ids.numel() // chunk), args.max_chunks)
    dense_nll = []
    sparse_nll = []
    sparse = AttentionMethod(name="local_global_topk", window=64, n_global=4, topk=32)
    for i in range(nchunks):
        c = ids[i * chunk:(i + 1) * chunk].unsqueeze(0).to(dev)
        with torch.no_grad():
            out = model(c, labels=c)
            dense_nll.append(float(out.loss.item()))
            orig = apply_method(model, sparse, fl)
            out2 = model(c, labels=c)
            restore(model, orig, fl)
            sparse_nll.append(float(out2.loss.item()))
        del c; torch.cuda.empty_cache()
        if (i + 1) % 25 == 0:
            print(f"  chunk {i+1}/{nchunks}", flush=True)

    dense_nll = np.array(dense_nll)
    sparse_nll = np.array(sparse_nll)
    ppl_dense = float(np.exp(dense_nll.mean()))
    ppl_sparse = float(np.exp(sparse_nll.mean()))
    rng = np.random.default_rng(42)
    deltas = []
    for _ in range(2000):
        idx = rng.integers(0, nchunks, nchunks)
        deltas.append(float(np.exp(sparse_nll[idx].mean()) - np.exp(dense_nll[idx].mean())))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    rec = {"base": args.base, "n_chunks": nchunks,
           "ppl_dense": round(ppl_dense, 3), "ppl_sparse": round(ppl_sparse, 3),
           "delta_ppl": round(ppl_sparse - ppl_dense, 3),
           "delta_ci95_lo": round(float(lo), 3), "delta_ci95_hi": round(float(hi), 3)}
    append_jsonl(OUT, rec)
    print(f"[ppl_paired] {args.base}: dense={ppl_dense:.3f} sparse={ppl_sparse:.3f} "
          f"delta={ppl_sparse - ppl_dense:+.3f} CI95=[{lo:+.3f},{hi:+.3f}]", flush=True)
    print("[done]", flush=True)

if __name__ == "__main__":
    main()
