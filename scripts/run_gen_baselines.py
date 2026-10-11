"""Unified generation-baseline comparison: quality (needle recall) + throughput (tokens/s) + KV mem.

Brings together the decode-time KV-budget methods with wall-clock timing, closing the two remaining
items: (a) a generation-throughput (tokens/s) comparison, and (b) Quest's query-aware PAGE selection,
implemented with the eviction pattern that cooperates with the hybrid's DynamicCache (re-materializing
a subset cache breaks the cache's _seen_tokens/attention-mask tracking, so we evict pages in place like
H2O but at page granularity on a QUANTIZED cache, scoring pages by accumulated query-aware attention).

Methods (all at a bounded KV budget):
  dense            full cache (upper bound)
  h2o              accumulated per-KEY attention, evict lowest (ICML'23)
  quest_quant_b4   quantize KV to 4-bit, keep all (Quest memory reduction)
  quest_page_b4    quantize to 4-bit + query-aware PAGE eviction (Quest sparsity)
  snapkv           prefill observation-window selection

Usage:
  python scripts/run_gen_baselines.py --model qwen3.5-0.8b --contexts 2048 --keep 128 256
"""
import argparse
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model_loader import MODELS, inspect_model, load_model  # noqa: E402
from kv_eviction import cache_kv_bytes  # noqa: E402
from needle import build_needle_prompt, check_needle, make_needle  # noqa: E402
from utils import append_jsonl, set_seed  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "gen_baselines.jsonl")
PAGE = 16


def _qd(t, bits):
    qmax = (1 << bits) - 1
    tmin = t.amin(dim=-1, keepdim=True)
    scale = ((t.amax(dim=-1, keepdim=True) - tmin) / qmax).clamp(min=1e-8)
    q = ((t - tmin) / scale).round().clamp(0, qmax)
    return (q * scale + tmin).to(t.dtype)


def _full(past):
    return [i for i, ly in enumerate(past.layers) if torch.is_tensor(getattr(ly, "keys", None))]


def _attn_map(attentions, past):
    out, ai = {}, 0
    for i, ly in enumerate(past.layers):
        if not torch.is_tensor(getattr(ly, "keys", None)):
            continue
        if attentions and ai < len(attentions) and attentions[ai] is not None:
            out[i] = attentions[ai]
        ai += 1
    return out


def gen_run(method, model, tok, ids, keep, bits, gen_tokens):
    use_attn = method in ("h2o", "quest_page")
    out = model(ids, use_cache=True, output_attentions=use_attn)
    past = out.past_key_values
    full = _full(past)
    quant = method in ("quest_quant", "quest_page")
    page = method == "quest_page"
    # quantize prefill cache for the quest variants
    if quant:
        for li in full:
            ly = past.layers[li]
            ly.keys = _qd(ly.keys, bits)
            ly.values = _qd(ly.values, bits)
    scores = {}
    if use_attn:
        for li, a in _attn_map(out.attentions, past).items():
            clen = past.layers[li].keys.shape[2]
            m = a[0].mean(dim=0).sum(dim=0).float()
            scores[li] = m if not page else _page_scores(m, clen)
    cur = ids[:, -1:]
    gen_ids = []
    t0 = time.perf_counter()
    for _ in range(gen_tokens):
        out = model(cur, past_key_values=past, use_cache=True, output_attentions=use_attn)
        past = out.past_key_values
        if quant:
            for li in full:
                ly = past.layers[li]
                ly.keys = _qd(ly.keys[:, :, -1:, :], bits) if False else ly.keys
        if use_attn:
            for li, a in _attn_map(out.attentions, past).items():
                clen = past.layers[li].keys.shape[2]
                add = a[0].mean(dim=0).sum(dim=0).float()
                if page:
                    add = _page_scores(add, clen)
                if li in scores and scores[li].numel() == add.numel():
                    scores[li] = scores[li] + add
                else:
                    scores[li] = add
            # evict by accumulated query-aware score
            unit = PAGE if page else 1
            for li in full:
                ly = past.layers[li]
                k = ly.keys
                clen = k.shape[2]
                n_units = clen // unit
                if n_units <= keep // unit + 2:
                    continue
                sc = scores.get(li)
                if sc is None or sc.numel() != n_units:
                    continue
                n_sink_u = max(1, 4 // unit + (1 if page else 0))
                n_recent_u = max(1, 16 // unit)
                prot = set(range(n_sink_u)) | set(range(max(0, n_units - n_recent_u), n_units))
                sc2 = sc.clone()
                for p in prot:
                    sc2[p] = float("-inf")
                n_keep_u = min(keep // unit, n_units)
                n_free = max(0, n_keep_u - len(prot))
                top = sc2.topk(min(n_free, n_units)).indices.tolist()
                keep_u = sorted(prot | set(top))
                idx = []
                for u in keep_u:
                    idx.extend(range(u * unit, min((u + 1) * unit, clen)))
                idx = torch.tensor(idx, device=k.device)
                ly.keys = k.index_select(2, idx)
                ly.values = ly.values.index_select(2, idx)
                if li in scores:
                    scores[li] = _reindex_page_scores(scores[li], keep_u, n_units) if page else scores[li][idx]
        cur = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        gen_ids.append(cur.item())
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return gen_ids, past, dt


def _page_scores(mass, clen):
    npg = clen // PAGE
    if npg == 0:
        return mass[:0]
    return mass[: npg * PAGE].view(npg, PAGE).sum(dim=1)


def _reindex_page_scores(sc, keep_u, n_units):
    return sc[torch.tensor(keep_u, device=sc.device)]


def run(model_key, contexts, keeps, bits, n_sink, dtype, gen_tokens):
    model_id = MODELS[model_key]
    info = inspect_model(model_id)
    model, tok = load_model(model_id, dtype=dtype, attn_impl="eager")
    device = next(model.parameters()).device
    print(f"[load] {model_id} full_attn={info.full_attention_layers} (eager)")
    depths, seeds = [0.1, 0.5, 0.9], [42, 43, 44]
    methods = ["dense", "h2o", f"quest_quant_b{bits}", f"quest_page_b{bits}"]
    mkey = {"dense": "dense", "h2o": "h2o", f"quest_quant_b{bits}": "quest_quant",
            f"quest_page_b{bits}": "quest_page"}

    for ctx in contexts:
        for keep in keeps:
            for m in methods:
                hits, tot_kv, tot_tps, tot = 0, 0.0, 0.0, 0
                for d in depths:
                    for s in seeds:
                        set_seed(s)
                        rng = random.Random(s)
                        needle = make_needle(rng)
                        prompt, needle = build_needle_prompt(tok, ctx, needle, d)
                        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
                        with torch.no_grad():
                            gen_ids, past, dt = gen_run(mkey[m], model, tok, ids, keep, bits,
                                                    gen_tokens)
                        gen = tok.decode(gen_ids, skip_special_tokens=True)
                        hits += int(check_needle(gen, needle))
                        kvb = cache_kv_bytes(past) / 2**20
                        bits_eff = bits if mkey[m] in ("quest_quant", "quest_page") else 16
                        tot_kv += kvb * (bits_eff / 16)
                        tot_tps += gen_tokens / dt
                        tot += 1
                rec = {"experiment": "gen_baselines", "method": m, "model": model_id,
                       "context": ctx, "keep": keep,
                       "needle_recall": round(hits / tot, 3),
                       "kv_mb_eff": round(tot_kv / tot, 2),
                       "tokens_per_s": round(tot_tps / tot, 2)}
                append_jsonl(RESULTS, rec)
                print(f"  ctx={ctx} keep={keep} {m:16s} recall={rec['needle_recall']} "
                      f"kv={rec['kv_mb_eff']}MB tps={rec['tokens_per_s']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-0.8b", choices=list(MODELS.keys()))
    ap.add_argument("--contexts", type=int, nargs="+", default=[2048])
    ap.add_argument("--keep", type=int, nargs="+", default=[128, 256])
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--n-sink", type=int, default=4)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--gen-tokens", type=int, default=24)
    args = ap.parse_args()
    run(args.model, args.contexts, args.keep, args.bits, args.n_sink, args.dtype, args.gen_tokens)


if __name__ == "__main__":
    main()
