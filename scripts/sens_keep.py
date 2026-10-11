"""W2: H2O keep sensitivity — needle retrieval at long context for keep in {64,128,256}."""
import sys, os, time, torch, random
sys.path.insert(0, "/root/mlattention/src")
from model_loader import load_model, inspect_model
from utils import append_jsonl

MODEL_PATH = "/root/qwen38"
OUT = "/root/mlattention/results/qwen38_sens_keep.jsonl"

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

def gen_h2o(model, ids, keep, n_sink, gen_tokens, warmup=4):
    out = model(ids, use_cache=True)
    past = out.past_key_values
    scores = {li: torch.zeros(past.layers[li].keys.shape[2], device=past.layers[li].keys.device)
              for li in full_layers(past)}
    model.config._attn_implementation = "eager"
    cur = out.logits[:, -1].argmax(dim=-1, keepdim=True)
    gen_ids = []
    for step in range(gen_tokens):
        out = model(cur, past_key_values=past, use_cache=True, output_attentions=True)
        for li, a in attn_map(out.attentions, past).items():
            add = a[0].mean(dim=0).sum(dim=0).float()
            scores[li] = scores[li] + add if (scores[li].numel() == add.numel()) else add
        past = out.past_key_values
        if step >= warmup:
            evict_by_scores(past, scores, keep, n_sink)
        cur = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        gen_ids.append(cur.item())
    model.config._attn_implementation = "sdpa"
    return gen_ids

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
    ctx = int(os.environ.get("CTX", "32768"))
    keeps = [int(x) for x in os.environ.get("KEEPS", "64,128,256").split(",")]
    depths = [float(x) for x in os.environ.get("DEPTHS", "0.1,0.5,0.9").split(",")]
    gen_tokens = 32
    n_sink = 4

    model, tok = load_model(MODEL_PATH, load_in_4bit=True)
    dev = next(model.parameters()).device
    print(f"[load] Q4 sdpa", flush=True)

    for keep in keeps:
        hits = tot = 0
        t0 = time.time()
        for d in depths:
            rng = random.Random(42 + int(d * 100))
            needle = str(rng.randint(1000000, 9999999))
            ids = needle_prompt(tok, ctx, needle, d).unsqueeze(0).to(dev)
            try:
                with torch.no_grad():
                    gen_ids = gen_h2o(model, ids, keep, n_sink, gen_tokens)
                gen_txt = tok.decode(gen_ids, skip_special_tokens=True)
                hits += int(needle in gen_txt); tot += 1
            except Exception as e:
                print(f"  keep={keep} depth={d} ERROR {repr(e)[:100]}", flush=True)
            del ids; torch.cuda.empty_cache()
        dt = time.time() - t0
        rec = {"metric": "sens_keep", "ctx": ctx, "keep": keep,
               "recall": round(hits/tot, 3) if tot else None,
               "tok_s": round(gen_tokens * tot / dt, 2) if dt else None}
        append_jsonl(OUT, rec)
        print(f"[sens_keep] ctx={ctx} keep={keep} recall={hits}/{tot}", flush=True)
    print("[done]", flush=True)

if __name__ == "__main__":
    main()
