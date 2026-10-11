"""GGUF eval (llama-cpp-python): PPL wikitext (2048 tokens) + HellaSwag + ARC + GPQA, logprob scoring.
Metodología IDÉNTICA a bench_eval.py (mismo texto, misma puntuación logprob por choice)."""
import sys, os, argparse, math
import numpy as np
from datasets import load_dataset

MODEL_PATH = "/root/Qwen3.8-27B-Q4_K_M.gguf"
OUT = "/root/mlattention/results/qwen38_evals.jsonl"

def log_softmax(logits):
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max()
    return x - np.log(np.exp(x).sum())

def get_logits(llm):
    return np.ctypeslib.as_array(llm._ctx.get_logits(), shape=(llm.n_vocab(),))

def choice_logprob(llm, context, choice):
    full = llm.tokenize((context + " " + choice).encode("utf-8"))
    ctx_len = len(llm.tokenize(context.encode("utf-8")))
    if ctx_len >= len(full):
        return float("-inf")
    completion = full[ctx_len:]
    llm.reset()
    llm.eval(full[:ctx_len])
    logits = get_logits(llm)
    total = 0.0
    for i, t in enumerate(completion):
        total += float(log_softmax(logits)[t])
        if i < len(completion) - 1:
            llm.eval([t])
            logits = get_logits(llm)
    return total / len(completion)

def append_jsonl(path, obj):
    import json
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--n-ctx", type=int, default=4096)
    args = ap.parse_args()

    from llama_cpp import Llama
    llm = Llama(model_path=MODEL_PATH, n_ctx=args.n_ctx, n_gpu_layers=-1, verbose=False)
    print(f"[gguf] modelo cargado, n_vocab={llm.n_vocab()}", flush=True)

    # --- PPL wikitext-2 (mismo texto que bench_eval.py) ---
    print("[ppl] wikitext-2 (2048 tokens) ...", flush=True)
    try:
        wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n".join(t for t in wt["text"] if t.strip())
        tokens = llm.tokenize(text.encode("utf-8"))[:2048]
        n = len(tokens)
        llm.reset()
        llm.eval([tokens[0]])
        logits = get_logits(llm)
        total = 0.0
        for i in range(1, n):
            total += float(log_softmax(logits)[tokens[i]])
            if i < n - 1:
                llm.eval([tokens[i]])
                logits = get_logits(llm)
        avg = total / (n - 1)
        ppl = math.exp(-avg)
        print(f"  PPL={ppl:.2f} (n_tokens={n}, avg_logp={avg:.3f})", flush=True)
        append_jsonl(OUT, {"config": "gguf_q4km", "bench": "wikitext2_ppl", "ppl": round(ppl, 2)})
    except Exception as e:
        print(f"  PPL error: {repr(e)}", flush=True)

    # --- HellaSwag ---
    print("[hellaswag] ...", flush=True)
    try:
        hs = load_dataset("Rowan/hellaswag", split="validation")
        correct = total = 0
        for i, ex in enumerate(hs):
            if i >= args.n: break
            scores = [choice_logprob(llm, ex["ctx"], e) for e in ex["endings"]]
            correct += int(scores.index(max(scores)) == int(ex["label"]))
            total += 1
            if total % 50 == 0: print(f"    {total} acc={correct/total:.3f}", flush=True)
        print(f"  HellaSwag acc={correct}/{total} ({correct/total:.3f})", flush=True)
        append_jsonl(OUT, {"config": "gguf_q4km", "bench": "hellaswag", "acc": round(correct/total, 4), "n": total})
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
            if isinstance(ch, dict) and "text" in ch:
                texts = ch["text"]; labels = ch["label"]
            elif isinstance(ch, list):
                texts = [c["text"] for c in ch]; labels = [c["label"] for c in ch]
            else:
                texts = [ch[k] for k in ch]; labels = list(ch.keys())
            scores = [choice_logprob(llm, q, t) for t in texts]
            pred = labels[scores.index(max(scores))]
            correct += int(pred == ex["answerKey"])
            total += 1
            if total % 50 == 0: print(f"    {total} acc={correct/total:.3f}", flush=True)
        print(f"  ARC-C acc={correct}/{total} ({correct/total:.3f})", flush=True)
        append_jsonl(OUT, {"config": "gguf_q4km", "bench": "arc_challenge", "acc": round(correct/total, 4), "n": total})
    except Exception as e:
        print(f"  ARC error: {repr(e)}", flush=True)

    # --- GPQA-Diamond ---
    print("[gpqa] diamond ...", flush=True)
    try:
        gp = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        correct = total = 0
        for i, ex in enumerate(gp):
            if i >= args.n: break
            q = ex["Question"].strip()
            choices = [ex["Correct Answer"].strip(), ex["Incorrect Answer 1"].strip(), ex["Incorrect Answer 2"].strip(), ex["Incorrect Answer 3"].strip()]
            scores = [choice_logprob(llm, q, c) for c in choices]
            correct += int(scores.index(max(scores)) == 0)
            total += 1
            if total % 50 == 0: print(f"    {total} acc={correct/total:.3f}", flush=True)
        print(f"  GPQA acc={correct}/{total} ({correct/total:.3f})", flush=True)
        append_jsonl(OUT, {"config": "gguf_q4km", "bench": "gpqa_diamond", "acc": round(correct/total, 4), "n": total})
    except Exception as e:
        print(f"  GPQA error: {repr(e)}", flush=True)

    print("[done] gguf_q4km", flush=True)

if __name__ == "__main__":
    main()
