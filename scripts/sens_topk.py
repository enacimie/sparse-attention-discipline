"""W2: lg_topk topk sensitivity — PPL (log-prob) for topk in {16,32,64} on Q4."""
import sys, os, torch
sys.path.insert(0, "/root/mlattention/src")
from attention_interceptor import apply_method, restore
from attention_methods import AttentionMethod
from model_loader import load_model, inspect_model
from utils import append_jsonl

MODEL_PATH = "/root/qwen38"
OUT = "/root/mlattention/results/qwen38_sens_topk.jsonl"

def main():
    topks = [int(x) for x in os.environ.get("TOPKS", "16,32,64").split(",")]
    model, tok = load_model(MODEL_PATH, load_in_4bit=True)
    info = inspect_model(MODEL_PATH)
    fl = info.full_attention_layers
    dev = next(model.parameters()).device
    from datasets import load_dataset
    wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(t for t in wt["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0][:2048].unsqueeze(0).to(dev)
    labels = ids.clone()

    for tk in topks:
        sparse = AttentionMethod(name="local_global_topk", window=64, n_global=4, topk=tk)
        with torch.no_grad():
            orig = apply_method(model, sparse, fl)
            out = model(ids, labels=labels)
            restore(model, orig, fl)
        ppl = torch.exp(out.loss).item()
        append_jsonl(OUT, {"metric": "sens_topk", "topk": tk, "ppl": round(ppl, 3)})
        print(f"[sens_topk] topk={tk} PPL={ppl:.3f}", flush=True)
    print("[done]", flush=True)

if __name__ == "__main__":
    main()
