"""W2: lg_topk topk sensitivity on the FULL wikitext-2 test (same scale as bootstrap PPL)."""
import sys, os, torch
sys.path.insert(0, "/root/mlattention/src")
from attention_interceptor import apply_method, restore
from attention_methods import AttentionMethod
from model_loader import load_model, inspect_model
from utils import append_jsonl

MODEL_PATH = "/root/qwen38"
OUT = "/root/mlattention/results/qwen38_sens_topk_full.jsonl"

def main():
    topks = [int(x) for x in os.environ.get("TOPKS", "16,32,64").split(",")]
    max_chunks = int(os.environ.get("MAX_CHUNKS", "100"))
    model, tok = load_model(MODEL_PATH, load_in_4bit=True)
    info = inspect_model(MODEL_PATH)
    fl = info.full_attention_layers
    dev = next(model.parameters()).device
    from datasets import load_dataset
    wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(t for t in wt["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    chunk = 2048
    nchunks = min((ids.numel() // chunk), max_chunks)

    for tk in topks:
        sparse = AttentionMethod(name="local_global_topk", window=64, n_global=4, topk=tk)
        losses = []
        for i in range(nchunks):
            c = ids[i * chunk:(i + 1) * chunk].unsqueeze(0).to(dev)
            with torch.no_grad():
                orig = apply_method(model, sparse, fl)
                out = model(c, labels=c)
                restore(model, orig, fl)
            losses.append(float(out.loss.item()))
            del c; torch.cuda.empty_cache()
        import numpy as np
        ppl = float(np.exp(np.mean(losses)))
        append_jsonl(OUT, {"metric": "sens_topk_full", "topk": tk, "ppl": round(ppl, 3), "n_chunks": nchunks})
        print(f"[sens_topk_full] topk={tk} PPL={ppl:.3f}", flush=True)
    print("[done]", flush=True)

if __name__ == "__main__":
    main()
