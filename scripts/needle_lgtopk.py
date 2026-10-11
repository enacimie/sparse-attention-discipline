"""W5: lg_topk-only needle (sparse mask, NO eviction) at 16K/32K.

Isolates the sparsity effect from H2O eviction at long context. Note lg_topk's
content selection forms an O(L^2) score matrix, so we cap at 32K (64K+ is where
its prefill becomes prohibitive -- exactly why H2O is the generation mechanism).
"""
import sys, os, time, torch, random
sys.path.insert(0, "/root/mlattention/src")
from attention_interceptor import apply_method, restore
from attention_methods import AttentionMethod
from model_loader import load_model, inspect_model
from kv_eviction import cache_kv_bytes
from utils import append_jsonl

MODEL_PATH = "/root/qwen38"
OUT = "/root/mlattention/results/qwen38_needle_lgtopk.jsonl"

def needle_prompt(tok, ctx, needle, depth):
    filler = "The grass is green and the sky is blue today. "
    fids = tok(filler, return_tensors="pt").input_ids[0]
    nids = tok(needle, return_tensors="pt").input_ids[0]
    q = tok(" The passcode is", return_tensors="pt").input_ids[0]
    target = ctx - nids.numel() - q.numel()
    n_rep = (target + fids.numel() - 1) // fids.numel()
    seq = fids.repeat(n_rep)[:target]
    pos = int(seq.numel() * depth)
    seq = torch.cat([seq[:pos], nids, seq[pos:]])
    return torch.cat([seq, q])

def main():
    ctxs = [int(x) for x in os.environ.get("CTXS", "16384,32768").split(",")]
    depths = [float(x) for x in os.environ.get("DEPTHS", "0.1,0.5,0.9").split(",")]
    gen_tokens = 32

    model, tok = load_model(MODEL_PATH, load_in_4bit=True)
    info = inspect_model(MODEL_PATH)
    fl = info.full_attention_layers
    dev = next(model.parameters()).device
    sparse = AttentionMethod(name="local_global_topk", window=64, n_global=4, topk=32)
    apply_method(model, sparse, fl)  # apply once for all ctxs

    for ctx in ctxs:
        hits = tot = 0; tot_kv = 0.0; t0 = time.time(); ntok = 0
        for d in depths:
            rng = random.Random(42 + int(d * 100))
            needle = str(rng.randint(1000000, 9999999))
            ids = needle_prompt(tok, ctx, needle, d).unsqueeze(0).to(dev)
            try:
                with torch.no_grad():
                    out = model(ids, use_cache=True)
                    past = out.past_key_values
                    cur = out.logits[:, -1].argmax(dim=-1, keepdim=True)
                    gen_ids = []
                    for _ in range(gen_tokens):
                        out = model(cur, past_key_values=past, use_cache=True)
                        past = out.past_key_values
                        cur = out.logits[:, -1].argmax(dim=-1, keepdim=True)
                        gen_ids.append(cur.item())
                gen_txt = tok.decode(gen_ids, skip_special_tokens=True)
                hits += int(needle in gen_txt); tot += 1
                tot_kv += cache_kv_bytes(past) / 2**20
                ntok += len(gen_ids)
            except Exception as e:
                print(f"  ctx={ctx} depth={d} ERROR {repr(e)[:100]}", flush=True)
            del ids; torch.cuda.empty_cache()
        dt = time.time() - t0
        rec = {"metric": "needle_lgtopk", "ctx": ctx,
               "recall": round(hits/tot, 3) if tot else None,
               "kv_mb": round(tot_kv/tot, 1) if tot else None,
               "tok_s": round(ntok/dt, 2) if dt else None}
        append_jsonl(OUT, rec)
        print(f"[lgtopk] ctx={ctx} recall={hits}/{tot} kv={(tot_kv/tot if tot else 0):.0f}MB", flush=True)
    print("[done]", flush=True)

if __name__ == "__main__":
    main()
