"""W3: PPL over wikitext-2 test in 2048-token chunks -> per-chunk NLL for bootstrap CI.

Runs one config (16bit / q4 / q4_sparse / 16bit_sparse). Saves per-chunk losses;
the CI is computed offline (mean NLL -> exp = PPL, bootstrap over chunks).
"""
import sys, os, torch, argparse
import numpy as np
sys.path.insert(0, "/root/mlattention/src")
from attention_interceptor import apply_method, restore
from attention_methods import AttentionMethod
from model_loader import load_model, inspect_model
from utils import append_jsonl

MODEL_PATH = "/root/qwen38"
OUT = "/root/mlattention/results/qwen38_ppl_bootstrap.jsonl"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["16bit", "q4", "q4_sparse", "16bit_sparse"], required=True)
    ap.add_argument("--max_chunks", type=int, default=100)
    args = ap.parse_args()

    sparse = None
    if args.config == "16bit":
        model, tok = load_model(MODEL_PATH, dtype="bfloat16")
    elif args.config == "q4":
        model, tok = load_model(MODEL_PATH, load_in_4bit=True)
    elif args.config == "q4_sparse":
        model, tok = load_model(MODEL_PATH, load_in_4bit=True)
        sparse = AttentionMethod(name="local_global_topk", window=64, n_global=4, topk=32)
    else:
        model, tok = load_model(MODEL_PATH, dtype="bfloat16")
        sparse = AttentionMethod(name="local_global_topk", window=64, n_global=4, topk=32)
    info = inspect_model(MODEL_PATH)
    fl = info.full_attention_layers
    dev = next(model.parameters()).device

    from datasets import load_dataset
    wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(t for t in wt["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]

    chunk = 2048
    nchunks = min((ids.numel() // chunk), args.max_chunks)
    losses = []
    for i in range(nchunks):
        c = ids[i * chunk:(i + 1) * chunk].unsqueeze(0).to(dev)
        with torch.no_grad():
            if sparse is not None:
                orig = apply_method(model, sparse, fl)
            out = model(c, labels=c)
            if sparse is not None:
                restore(model, orig, fl)
        losses.append(float(out.loss.item()))
        del c; torch.cuda.empty_cache()
        if (i + 1) % 20 == 0:
            print(f"  chunk {i+1}/{nchunks}", flush=True)

    ppl = float(np.exp(np.mean(losses)))
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, len(losses), len(losses))
        boots.append(float(np.exp(np.mean([losses[j] for j in idx]))))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    rec = {"config": args.config, "bench": "wikitext2_ppl_bootstrap",
           "n_chunks": len(losses), "ppl": round(ppl, 3),
           "ppl_ci95_lo": round(float(lo), 3), "ppl_ci95_hi": round(float(hi), 3)}
    append_jsonl(OUT, rec)
    print(f"[ppl_boot] {args.config} n={len(losses)} PPL={ppl:.3f} CI95=[{lo:.3f},{hi:.3f}]", flush=True)
    print("[done]", flush=True)

if __name__ == "__main__":
    main()
