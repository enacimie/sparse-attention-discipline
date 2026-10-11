"""Qwen3.8-27B 4-bit: needle retrieval dense vs sparse (sin thinking, raw prompt)."""
import sys, os, time, torch, random
sys.path.insert(0, "/root/mlattention/src")
from attention_interceptor import apply_method, reset_pair_counter, restore, PAIR_COUNTER
from attention_methods import AttentionMethod
from model_loader import inspect_model, load_model
from utils import set_seed, append_jsonl

MODEL_PATH = "/root/qwen38"
OUT = "/root/mlattention/results/qwen38_needle.jsonl"

def main():
    set_seed(42)
    model, tok = load_model(MODEL_PATH, load_in_4bit=True)
    info = inspect_model(MODEL_PATH)
    fl = info.full_attention_layers
    dev = next(model.parameters()).device
    print(f"[load] ok, full_attn={len(fl)} layers", flush=True)

    methods = [
        ("dense", None),
        ("lg_topk", AttentionMethod(name="local_global_topk", window=64, n_global=4, topk=32)),
        ("lg_randtopk", AttentionMethod(name="local_global_randtopk", window=64, n_global=4, topk=32, seed=42)),
        ("uniform_random", AttentionMethod(name="uniform_random", topk=32, seed=42)),
    ]
    depths = [0.1, 0.5, 0.9]
    seeds = [42, 43]

    for name, m in methods:
        hits = total = 0
        for depth in depths:
            for seed in seeds:
                rng = random.Random(seed * 100 + int(depth * 10))
                needle = str(rng.randint(1000000, 9999999))
                filler = "The grass is green and the sky is blue today. " * 180
                fids = tok(filler, return_tensors="pt").input_ids[0]
                nids = tok(needle, return_tensors="pt").input_ids[0]
                pos = int(len(fids) * depth)
                seq = torch.cat([fids[:pos], nids, fids[pos:]])
                prompt_ids = torch.cat([seq, tok(" The passcode is", return_tensors="pt").input_ids[0]])
                inp = prompt_ids.unsqueeze(0).to(dev)
                if m is not None:
                    orig = apply_method(model, m, fl)
                out = model.generate(inp, max_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id)
                if m is not None:
                    restore(model, orig, fl)
                gen = tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True)
                hit = needle in gen
                hits += int(hit); total += 1
        recall = hits / total if total else 0
        print(f"[needle] {name:14s} recall={hits}/{total} ({recall:.3f})", flush=True)
        append_jsonl(OUT, {"metric":"needle_sparse","method":name,"hits":hits,"total":total,
                           "recall":round(recall,4)})

    print("[done]", flush=True)

if __name__ == "__main__":
    main()
