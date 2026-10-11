"""W6: fidelity vs 16-bit dense — KL divergence + top-k overlap, for Q4 / lg_topk / lg_perhead."""
import sys, os, torch
sys.path.insert(0, "/root/mlattention/src")
from attention_interceptor import apply_method, restore
from attention_methods import AttentionMethod
from model_loader import load_model, inspect_model
from utils import append_jsonl, fidelity_metrics

MODEL_PATH = "/root/qwen38"
OUT = "/root/mlattention/results/qwen38_fidelity_kl.jsonl"

def topk_overlap(approx, ref, k=10):
    a = approx.topk(k, dim=-1).indices
    r = ref.topk(k, dim=-1).indices
    return float((a == r).float().mean().item())

def main():
    model16, tok = load_model(MODEL_PATH, dtype="bfloat16")
    info = inspect_model(MODEL_PATH)
    fl = info.full_attention_layers
    dev = next(model16.parameters()).device
    from datasets import load_dataset
    wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(t for t in wt["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0][:512].unsqueeze(0).to(dev)

    with torch.no_grad():
        ref = model16(ids).logits  # (1, L, vocab)
    del model16; torch.cuda.empty_cache()

    # Q4
    model, _ = load_model(MODEL_PATH, load_in_4bit=True)
    with torch.no_grad():
        out = model(ids).logits
    m = fidelity_metrics(out[0], ref[0])
    m["topk_overlap"] = round(topk_overlap(out, ref), 4)
    m["config"] = "q4"
    append_jsonl(OUT, m)
    print(f"[fidelity] q4: KL={m['kl_div']} cos={m['logit_cosine']} topk={m['topk_overlap']}", flush=True)
    del model; torch.cuda.empty_cache()

    # sparse variants on Q4
    for name in ["local_global_topk", "local_global_perhead"]:
        model, _ = load_model(MODEL_PATH, load_in_4bit=True)
        sparse = AttentionMethod(name=name, window=64, n_global=4, topk=32)
        with torch.no_grad():
            orig = apply_method(model, sparse, fl)
            out = model(ids).logits
            restore(model, orig, fl)
        m = fidelity_metrics(out[0], ref[0])
        m["topk_overlap"] = round(topk_overlap(out, ref), 4)
        m["config"] = "q4_" + name
        append_jsonl(OUT, m)
        print(f"[fidelity] {name}: KL={m['kl_div']} cos={m['logit_cosine']} topk={m['topk_overlap']}", flush=True)
        del model; torch.cuda.empty_cache()

    print("[done]", flush=True)

if __name__ == "__main__":
    main()
