"""Qwen3.8-27B: benchmarks (PPL wikitext + GPQA + HellaSwag + ARC) para 16bit / q4 / q4_sparse."""
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

def score_choices(model, tok, dev, context, choices, sparse, fl):
    return [choice_logprob(model, tok, dev, context, c, sparse, fl) for c in choices]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["16bit", "q4", "q4_sparse"], default="q4")
    ap.add_argument("--n", type=int, default=400)
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

    # --- PPL wikitext-2 ---
    print("[ppl] wikitext-2 ...", flush=True)
    try:
        wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n".join(t for t in wt["text"] if t.strip())
        ids = tok(text, return_tensors="pt").input_ids[0][:2048]
        with torch.no_grad():
            if sparse is not None:
                orig = apply_method(model, sparse, fl)
            out = model(ids.unsqueeze(0).to(dev), labels=ids.unsqueeze(0).to(dev))
            if sparse is not None:
                restore(model, orig, fl)
        ppl = torch.exp(out.loss).item()
        print(f"  PPL={ppl:.2f}", flush=True)
        append_jsonl(OUT, {"config": args.config, "bench": "wikitext2_ppl", "ppl": round(ppl, 2)})
    except Exception as e:
        print(f"  PPL error: {repr(e)}", flush=True)

    # --- GPQA-Diamond ---
    print("[gpqa] diamond ...", flush=True)
    try:
        gp = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="test")
        correct = total = 0
        for i, ex in enumerate(gp):
            if i >= args.n: break
            q = ex["Question"]
            choices = [ex["Correct Answer"]] + list(ex["Incorrect Answer"])
            scores = score_choices(model, tok, dev, q, choices, sparse, fl)
            correct += int(scores.index(max(scores)) == 0)
            total += 1
            if total % 50 == 0: print(f"    {total} acc={correct/total:.3f}", flush=True)
        print(f"  GPQA acc={correct}/{total} ({correct/total:.3f})", flush=True)
        append_jsonl(OUT, {"config": args.config, "bench": "gpqa_diamond", "acc": round(correct/total, 4), "n": total})
    except Exception as e:
        print(f"  GPQA error: {repr(e)}", flush=True)

    # --- HellaSwag ---
    print("[hellaswag] ...", flush=True)
    try:
        hs = load_dataset("Rowan/hellaswag", split="validation")
        correct = total = 0
        for i, ex in enumerate(hs):
            if i >= args.n: break
            scores = score_choices(model, tok, dev, ex["ctx"], list(ex["endings"]), sparse, fl)
            correct += int(scores.index(max(scores)) == int(ex["label"]))
            total += 1
            if total % 50 == 0: print(f"    {total} acc={correct/total:.3f}", flush=True)
        print(f"  HellaSwag acc={correct}/{total} ({correct/total:.3f})", flush=True)
        append_jsonl(OUT, {"config": args.config, "bench": "hellaswag", "acc": round(correct/total, 4), "n": total})
    except Exception as e:
        print(f"  HellaSwag error: {repr(e)}", flush=True)

    # --- ARC-Challenge ---
    print("[arc] challenge ...", flush=True)
    try:
        arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        correct = total = 0
        for i, ex in enumerate(arc):
            if i >= args.n: break
            q = ex["question"]
            ch = ex["choices"]
            # ARC choices puede ser dict {text:[], label:[]} o list de {text,label}
            if isinstance(ch, dict) and "text" in ch:
                texts = ch["text"]; labels = ch["label"]
            elif isinstance(ch, list):
                texts = [c["text"] for c in ch]; labels = [c["label"] for c in ch]
            else:
                texts = [ch[k] for k in ch]; labels = list(ch.keys())
            scores = score_choices(model, tok, dev, q, texts, sparse, fl)
            pred = labels[scores.index(max(scores))]
            correct += int(pred == ex["answerKey"])
            total += 1
            if total % 50 == 0: print(f"    {total} acc={correct/total:.3f}", flush=True)
        print(f"  ARC-C acc={correct}/{total} ({correct/total:.3f})", flush=True)
        append_jsonl(OUT, {"config": args.config, "bench": "arc_challenge", "acc": round(correct/total, 4), "n": total})
    except Exception as e:
        print(f"  ARC error: {repr(e)}", flush=True)

    print(f"[done] {args.config}", flush=True)

if __name__ == "__main__":
    main()
