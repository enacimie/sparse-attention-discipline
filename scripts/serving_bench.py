"""W1: serving — decode throughput vs context length (dense vs H2O keep=128), batch=1.

Shows eviction makes decode latency independent of context (the serving argument),
vs dense which slows linearly. Also records the KV bytes held at the end of each run.
"""
import sys, os, time, torch
sys.path.insert(0, "/root/mlattention/src")
from model_loader import load_model, inspect_model
from kv_eviction import cache_kv_bytes
from utils import append_jsonl

MODEL_PATH = "/root/qwen38"
OUT = "/root/mlattention/results/qwen38_serving.jsonl"

def full_layers(past):
    return [i for i, ly in enumerate(past.layers) if torch.is_tensor(getattr(ly, "keys", None))]

def attn_map(attentions, past):
    out = {}
    if attentions is None:
        return out
    ai = 0
    for i, ly in enumerate(past.layers):
        if not torch.is_tensor(getattr(ly, "keys", None)):
            continue
        if ai < len(attentions) and attentions[ai] is not None:
            out[i] = attentions[ai]
        ai += 1
    return out

def evict_by_scores(past, scores, keep, n_sink):
    for li in full_layers(past):
        k = past.layers[li].keys
        clen = k.shape[2]
        if clen <= keep + n_sink:
            continue
        sc = scores.get(li)
        if sc is None or sc.numel() != clen:
            sc = k[0].float().norm(dim=-1).mean(dim=0)
        sink = torch.arange(n_sink, device=k.device)
        sc2 = sc.clone(); sc2[sink] = float("-inf")
        top = sc2.topk(keep).indices
        idx = torch.cat([sink, top]).unique().sort().values
        past.layers[li].keys = k.index_select(2, idx)
        past.layers[li].values = past.layers[li].values.index_select(2, idx)
        scores[li] = sc[idx]

def gen(model, ids, method, keep, n_sink, gen_tokens, warmup=4):
    out = model(ids, use_cache=True)
    past = out.past_key_values
    scores = {}
    if method == "h2o":
        scores = {li: torch.zeros(past.layers[li].keys.shape[2], device=past.layers[li].keys.device)
                  for li in full_layers(past)}
        model.config._attn_implementation = "eager"
    cur = out.logits[:, -1].argmax(dim=-1, keepdim=True)
    ntok = 0
    t0 = time.perf_counter()
    for step in range(gen_tokens):
        if method == "h2o":
            out = model(cur, past_key_values=past, use_cache=True, output_attentions=True)
            for li, a in attn_map(out.attentions, past).items():
                add = a[0].mean(dim=0).sum(dim=0).float()
                scores[li] = scores[li] + add if (scores[li].numel() == add.numel()) else add
        else:
            out = model(cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        if method == "h2o" and step >= warmup:
            evict_by_scores(past, scores, keep, n_sink)
        cur = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        ntok += 1
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    if method == "h2o":
        model.config._attn_implementation = "sdpa"
    return ntok, dt, past

def main():
    ctxs = [int(x) for x in os.environ.get("CTXS", "512,1024,2048,4096,8192").split(",")]
    gen_tokens = int(os.environ.get("GEN", "64"))
    model, tok = load_model(MODEL_PATH, load_in_4bit=True)
    dev = next(model.parameters()).device
    filler = tok("The grass is green and the sky is blue today. ", return_tensors="pt").input_ids[0]

    for ctx in ctxs:
        n = (ctx + filler.numel() - 1) // filler.numel()
        ids = filler.repeat(n)[:ctx].unsqueeze(0).to(dev)
        for method in ["dense", "h2o"]:
            with torch.no_grad():
                ntok, dt, past = gen(model, ids, method, 128, 4, gen_tokens)
            kv = cache_kv_bytes(past) / 2**20
            append_jsonl(OUT, {"metric": "serving_decode", "ctx": ctx, "method": method,
                               "tok_s": round(ntok / dt, 2), "kv_mb": round(kv, 1)})
            print(f"  ctx={ctx} {method:6s}: {ntok/dt:.2f} tok/s  kv={kv:.1f}MB", flush=True)
            del past; torch.cuda.empty_cache()
    print("[done]", flush=True)

if __name__ == "__main__":
    main()
