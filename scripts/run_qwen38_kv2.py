"""Qwen3.8-27B Q4: KV eviction CORRECTO (H2O real, prompt crudo) — recall + KV MB."""
import sys, os, time, torch, random
sys.path.insert(0, "/root/mlattention/src")
from attention_interceptor import apply_method, restore
from attention_methods import AttentionMethod
from model_loader import inspect_model, load_model
from kv_eviction import RecencyEviction, evict_cache, cache_kv_bytes
from utils import append_jsonl

MODEL_PATH = "/root/qwen38"
OUT = "/root/mlattention/results/qwen38_kv2.jsonl"

def _full_with_kv(past):
    return [i for i, ly in enumerate(past.layers) if torch.is_tensor(getattr(ly, "keys", None))]

def _attn_map(attentions, past):
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

def gen_h2o(model, ids, keep, n_sink, gen_tokens, obs_recent=16):
    out = model(ids, use_cache=True, output_attentions=True)
    past = out.past_key_values
    full = _full_with_kv(past)
    scores = {}
    for li, a in _attn_map(out.attentions, past).items():
        scores[li] = a[0].mean(dim=0).sum(dim=0).float()
    cur = ids[:, -1:]
    gen_ids = []
    for _ in range(gen_tokens):
        out = model(cur, past_key_values=past, use_cache=True, output_attentions=True)
        past = out.past_key_values
        for li, a in _attn_map(out.attentions, past).items():
            add = a[0].mean(dim=0).sum(dim=0).float()
            scores[li] = scores[li] + add if (li in scores and scores[li].numel() == add.numel()) else add
        for li in full:
            ly = past.layers[li]
            k = ly.keys
            clen = k.shape[2]
            if clen <= keep + n_sink:
                continue
            sc = scores.get(li)
            if sc is None or sc.numel() != clen:
                sc = k[0].float().norm(dim=-1).mean(dim=0)
            sink = torch.arange(n_sink, device=k.device)
            recent = torch.arange(clen - obs_recent, clen, device=k.device)
            prot = torch.cat([sink, recent]).unique()
            sc2 = sc.clone(); sc2[prot] = float("-inf")
            top = sc2.topk(min(max(0, keep - prot.numel()), clen)).indices
            idx = torch.cat([prot, top]).unique().sort().values
            ly.keys = k.index_select(2, idx)
            ly.values = ly.values.index_select(2, idx)
            scores[li] = scores[li][idx]
        cur = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        gen_ids.append(cur.item())
    return gen_ids, past

def raw_needle_prompt(tok, ctx, needle, depth):
    filler = "The grass is green and the sky is blue today. " * 180
    fids = tok(filler, return_tensors="pt").input_ids[0]
    nids = tok(needle, return_tensors="pt").input_ids[0]
    pos = int(len(fids) * depth)
    seq = torch.cat([fids[:pos], nids, fids[pos:]])
    q = tok(" The passcode is", return_tensors="pt").input_ids[0]
    return torch.cat([seq, q]), needle

def main():
    ctx = int(os.environ.get("CTX", "2048"))
    keep = int(os.environ.get("KEEP", "128"))
    n_sink = 4
    gen_tokens = 40
    model, tok = load_model(MODEL_PATH, load_in_4bit=True, attn_impl="eager")
    info = inspect_model(MODEL_PATH)
    fl = info.full_attention_layers
    dev = next(model.parameters()).device
    print(f"[load] eager 4-bit, full_attn={len(fl)}", flush=True)

    methods = ["dense", "lg_topk", "h2o", "recency"]
    for name in methods:
        hits = tot = 0; tot_kv = 0.0; t0 = time.time(); ntok = 0
        for d in [0.1, 0.5, 0.9]:
            for s in [42, 43]:
                rng = random.Random(s)
                needle = str(rng.randint(1000000, 9999999))
                ids, ndl = raw_needle_prompt(tok, ctx, needle, d)
                ids = ids.unsqueeze(0).to(dev)
                with torch.no_grad():
                    if name == "h2o":
                        gen_ids, past = gen_h2o(model, ids, keep, n_sink, gen_tokens)
                    elif name == "recency":
                        out = model(ids, use_cache=True); past = out.past_key_values
                        evict_cache(past, RecencyEviction(keep=keep, n_sink=n_sink))
                        cur = ids[:, -1:]; gen_ids = []
                        for _ in range(gen_tokens):
                            out = model(cur, past_key_values=past, use_cache=True)
                            past = out.past_key_values
                            evict_cache(past, RecencyEviction(keep=keep, n_sink=n_sink))
                            cur = out.logits[:, -1].argmax(dim=-1, keepdim=True)
                            gen_ids.append(cur.item())
                    elif name == "lg_topk":
                        m = AttentionMethod(name="local_global_topk", window=64, n_global=4, topk=32)
                        orig = apply_method(model, m, fl)
                        out = model(ids, use_cache=True); past = out.past_key_values
                        restore(model, orig, fl)
                        cur = ids[:, -1:]; gen_ids = []
                        for _ in range(gen_tokens):
                            orig = apply_method(model, m, fl)
                            out = model(cur, past_key_values=past, use_cache=True)
                            past = out.past_key_values
                            restore(model, orig, fl)
                            cur = out.logits[:, -1].argmax(dim=-1, keepdim=True)
                            gen_ids.append(cur.item())
                    else:  # dense
                        out = model(ids, use_cache=True); past = out.past_key_values
                        cur = ids[:, -1:]; gen_ids = []
                        for _ in range(gen_tokens):
                            out = model(cur, past_key_values=past, use_cache=True)
                            past = out.past_key_values
                            cur = out.logits[:, -1].argmax(dim=-1, keepdim=True)
                            gen_ids.append(cur.item())
                gen = tok.decode(gen_ids, skip_special_tokens=True)
                hits += int(ndl in gen); tot += 1
                tot_kv += cache_kv_bytes(past) / 2**20
                ntok += len(gen_ids)
        dt = time.time() - t0
        rec = {"metric":"kv2","method":name,"ctx":ctx,"keep":keep,
               "recall":round(hits/tot,3),"kv_mb":round(tot_kv/tot,2),"tok_s":round(ntok/dt,2)}
        append_jsonl(OUT, rec)
        print(f"[kv2] {name:10s} recall={hits}/{tot} kv={tot_kv/tot:.1f}MB tok/s={ntok/dt:.1f}", flush=True)

    print("[done]", flush=True)

if __name__ == "__main__":
    main()
