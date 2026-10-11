"""Needle-in-haystack a contexto largo (16K-256K): dense vs eviction (KNorm/H2O keep=128).

Control clave: si dense también falla a un contexto, el culpable es el modelo (GDNA),
no nuestra eviction. SDPA para prefill (O(ctx) memoria), eager solo en decode (O(ctx))."""
import sys, os, time, torch, random
sys.path.insert(0, "/root/mlattention/src")
from model_loader import inspect_model, load_model
from kv_eviction import RecencyEviction, evict_cache, cache_kv_bytes
from utils import append_jsonl

MODEL_PATH = "/root/qwen38"
OUT = "/root/mlattention/results/qwen38_needle_long.jsonl"

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
    out = model(ids, use_cache=True)  # prefill SDPA (O(ctx) memoria)
    past = out.past_key_values
    scores = {}
    for li in full_layers(past):
        if method == "h2o":
            scores[li] = torch.zeros(past.layers[li].keys.shape[2], device=past.layers[li].keys.device)
        else:
            scores[li] = past.layers[li].keys[0].float().norm(dim=-1).mean(dim=0)
    if method == "h2o":
        model.config._attn_implementation = "eager"  # toggle: eager solo en decode (O(ctx))
    cur = ids[:, -1:]
    gen_ids = []
    for step in range(gen_tokens):
        if method == "h2o":
            out = model(cur, past_key_values=past, use_cache=True, output_attentions=True)
            for li, a in attn_map(out.attentions, past).items():
                add = a[0].mean(dim=0).sum(dim=0).float()
                scores[li] = scores[li] + add if (scores[li].numel() == add.numel()) else add
        else:
            out = model(cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        if method == "recency":
            evict_cache(past, RecencyEviction(keep=keep, n_sink=n_sink))
        elif method == "knorm":
            evict_by_scores(past, scores, keep, n_sink)
        elif method == "h2o" and step >= warmup:
            evict_by_scores(past, scores, keep, n_sink)
        cur = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        gen_ids.append(cur.item())
    if method == "h2o":
        model.config._attn_implementation = "sdpa"  # restaurar
    return gen_ids, past

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
    return torch.cat([seq, q]), needle

def main():
    keep = int(os.environ.get("KEEP", "128"))
    n_sink = 4
    gen_tokens = int(os.environ.get("GEN", "32"))
    ctxs = [int(x) for x in os.environ.get("CTXS", "16384,32768,65536,131072,262144").split(",")]
    methods = os.environ.get("METHODS", "dense,h2o").split(",")
    depths = [float(x) for x in os.environ.get("DEPTHS", "0.1,0.5,0.9").split(",")]

    model, tok = load_model(MODEL_PATH, load_in_4bit=True)  # SDPA default
    info = inspect_model(MODEL_PATH)
    dev = next(model.parameters()).device
    print(f"[load] sdpa 4-bit full_attn={len(info.full_attention_layers)}", flush=True)

    for ctx in ctxs:
        for method in methods:
            hits = tot = 0; tot_kv = 0.0; t0 = time.time(); ntok = 0
            for d in depths:
                rng = random.Random(42 + int(d * 100))
                needle = str(rng.randint(1000000, 9999999))
                ids, ndl = needle_prompt(tok, ctx, needle, d)
                ids = ids.unsqueeze(0).to(dev)
                try:
                    with torch.no_grad():
                        gen_ids, past = gen(model, ids, method, keep, n_sink, gen_tokens)
                    gen_txt = tok.decode(gen_ids, skip_special_tokens=True)
                    hits += int(ndl in gen_txt); tot += 1
                    tot_kv += cache_kv_bytes(past) / 2**20
                    ntok += len(gen_ids)
                except Exception as e:
                    print(f"  ctx={ctx} {method} depth={d} ERROR: {repr(e)[:120]}", flush=True)
                del ids; torch.cuda.empty_cache()
            dt = time.time() - t0
            rec = {"metric": "needle_long", "ctx": ctx, "method": method, "keep": keep,
                   "recall": round(hits/tot, 3) if tot else None, "kv_mb": round(tot_kv/tot, 2) if tot else None,
                   "tok_s": round(ntok/dt, 2) if dt else None}
            append_jsonl(OUT, rec)
            print(f"[needle] ctx={ctx} {method:8s} recall={hits}/{tot} kv={(tot_kv/tot if tot else 0):.1f}MB tok/s={ntok/dt:.1f}", flush=True)
    print("[done]", flush=True)

if __name__ == "__main__":
    main()
