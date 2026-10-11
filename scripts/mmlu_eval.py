"""Qwen3.8-27B: MMLU (0-shot logprob) para 16bit / q4 / q4_sparse."""
import sys, os, torch, argparse
sys.path.insert(0, "/root/mlattention/src")
from attention_interceptor import apply_method, restore
from attention_methods import AttentionMethod
from model_loader import inspect_model, load_model
from utils import append_jsonl

MODEL_PATH = "/root/qwen38"
OUT = "/root/mlattention/results/qwen38_evals.jsonl"

def choice_logprob(model, tok, dev, context, choice, sparse=None, fl=None):
    ids = tok(context + " " + choice, return_tensors="pt").input_ids[0]
    c_len = tok(context, return_tensors="pt").input_ids[0].numel()
    labels = torch.full_like(ids, -100)
    labels[c_len:] = ids[c_len:]
    with torch.no_grad():
        if sparse is not None:
            orig = apply_method(model, sparse, fl)
        out = model(ids.unsqueeze(0).to(dev), labels=labels.unsqueeze(0).to(dev))
        if sparse is not None:
            restore(model, orig, fl)
    return -out.loss.item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["16bit", "q4", "q4_sparse"], default="q4")
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()

    sparse = None
    if args.config == "16bit":
        model, tok = load_model(MODEL_PATH, dtype="bfloat16")
    elif args.config == "q4":
        model, tok = load_model(MODEL_PATH, load_in_4bit=True)
    else:
        model, tok = load_model(MODEL_PATH, load_in_4bit=True)
        sparse = AttentionMethod(name="local_global_topk", window=64, n_global=4, topk=32)
    info = inspect_model(MODEL_PATH)
    fl = info.full_attention_layers
    dev = next(model.parameters()).device
    print(f"[config] {args.config} full_attn={len(fl)}", flush=True)

    from datasets import load_dataset

    print("[mmlu] ...", flush=True)
    try:
        mmlu = load_dataset("cais/mmlu", "all", split="test")
        correct = total = 0
        for i, ex in enumerate(mmlu):
            if i >= args.n: break
            q = ex["question"]
            choices = list(ex["choices"])
            scores = [choice_logprob(model, tok, dev, q, c, sparse, fl) for c in choices]
            correct += int(scores.index(max(scores)) == int(ex["answer"]))
            total += 1
            if total % 50 == 0: print(f"    {total} acc={correct/total:.3f}", flush=True)
        print(f"  MMLU acc={correct}/{total} ({correct/total:.3f})", flush=True)
        append_jsonl(OUT, {"config": args.config, "bench": "mmlu", "acc": round(correct/total, 4), "n": total})
    except Exception as e:
        print(f"  MMLU error: {repr(e)}", flush=True)

    # --- GPQA-Diamond ---
    print("[gpqa] diamond ...", flush=True)
    try:
        gp = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        correct = total = 0
        for i, ex in enumerate(gp):
            if i >= args.n: break
            q = ex["Question"].strip()
            choices = [ex["Correct Answer"].strip(), ex["Incorrect Answer 1"].strip(), ex["Incorrect Answer 2"].strip(), ex["Incorrect Answer 3"].strip()]
            scores = [choice_logprob(model, tok, dev, q, c, sparse, fl) for c in choices]
            correct += int(scores.index(max(scores)) == 0)
            total += 1
            if total % 50 == 0: print(f"    {total} acc={correct/total:.3f}", flush=True)
        print(f"  GPQA acc={correct}/{total} ({correct/total:.3f})", flush=True)
        append_jsonl(OUT, {"config": args.config, "bench": "gpqa_diamond", "acc": round(correct/total, 4), "n": total})
    except Exception as e:
        print(f"  GPQA error: {repr(e)}", flush=True)

    print(f"[done] {args.config}", flush=True)

if __name__ == "__main__":
    main()
