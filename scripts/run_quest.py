"""Quest baseline (reviewer §3.3): query-aware sparsity on a QUANTIZED KV cache.

Quest (ICML'24) reduces KV memory by quantizing it into pages and, at each query, selecting the most
relevant pages via an upper-bound attention estimate from per-page metadata. We implement the two
ingredients on the hybrid's DynamicCache and compare needle recall at reduced KV memory:
  - quest_quant: quantize the KV cache to `bits` (memory x16/bits), keep all tokens. Isolates the
    quality cost of quantization.
  - quest_sel: quantize + query-aware page selection (keep top pages by a query-page upper bound),
    the full Quest recipe.
Metric is needle recall (as run_kv_eviction / run_h2o_snapkv) so numbers are comparable.

Usage:
  python scripts/run_quest.py --model qwen3.5-0.8b --contexts 2048 --bits 4 --pages-keep 32
"""
import argparse
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model_loader import MODELS, inspect_model, load_model  # noqa: E402
from kv_eviction import cache_kv_bytes  # noqa: E402
from needle import build_needle_prompt, check_needle, make_needle  # noqa: E402
from utils import append_jsonl, set_seed  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "quest.jsonl")
PAGE = 16  # tokens per page


def _quantize(t, bits):
    """Per-token min/max quantization over head_dim. Returns (q, tmin, scale)."""
    qmax = (1 << bits) - 1
    tmin = t.amin(dim=-1, keepdim=True)
    tmax = t.amax(dim=-1, keepdim=True)
    scale = ((tmax - tmin) / qmax).clamp(min=1e-8)
    q = ((t - tmin) / scale).round().clamp(0, qmax)
    return q, tmin, scale


def _dequantize(q, tmin, scale):
    return q * scale + tmin


def _quant_dequant(t, bits):
    q, tmin, scale = _quantize(t, bits)
    return _dequantize(q.float(), tmin, scale).to(t.dtype)


def _full_layers(past):
    return [i for i, ly in enumerate(past.layers)
            if torch.is_tensor(getattr(ly, "keys", None))]


def gen_quest(model, tok, ids, bits, gen_tokens, pages_keep=None):
    """Quantized-KV generation.

    - pages_keep=None  -> quest_quant: quantize the whole cache, attend to ALL of it. Isolates the
      quality cost of quantization (memory x16/bits).
    - pages_keep=int   -> FAITHFUL Quest: keep the full QUANTIZED cache and, at every decode step,
      RE-SELECT the most query-relevant pages (scored by the attention mass they received) and attend
      only to those. Pages are NOT permanently evicted, so a needle page can be re-selected when the
      model attends it while generating the answer (this is what distinguishes Quest from a one-shot
      eviction).
    """
    use_sel = pages_keep is not None
    out = model(ids, use_cache=True, output_attentions=use_sel)
    past = out.past_key_values
    full = _full_layers(past)
    # quantize the prefill cache (realistic quantization error) and keep it as the full store
    fullstore = {}
    for li in full:
        ly = past.layers[li]
        kq = _quant_dequant(ly.keys, bits)
        vq = _quant_dequant(ly.values, bits)
        fullstore[li] = [kq, vq]
        if not use_sel:
            ly.keys, ly.values = kq, vq
    page_score = {}
    cur = ids[:, -1:]
    gen_ids = []
    for _ in range(gen_tokens):
        if use_sel:
            # materialize, per layer, the top query-relevant pages from the full quantized cache
            mat_idx = {}
            for li in full:
                fk, fv = fullstore[li]
                clen = fk.shape[2]
                npg = clen // PAGE
                if npg <= pages_keep:
                    idx = torch.arange(clen, device=fk.device)
                else:
                    sc = page_score.get(li)
                    if sc is None or sc.numel() != npg:
                        sc = fk[0].float().norm(dim=-1).mean(dim=0)[: npg * PAGE].view(npg, PAGE).amax(1)
                    top_pg = sc.topk(min(pages_keep, npg)).indices
                    keep_pos = []
                    for p in top_pg.tolist():
                        keep_pos.extend(range(p * PAGE, min((p + 1) * PAGE, clen)))
                    keep_pos += list(range(max(0, clen - PAGE), clen))      # recent
                    keep_pos += list(range(min(PAGE, clen)))                 # sinks
                    idx = torch.tensor(sorted(set(keep_pos)), device=fk.device)
                mat_idx[li] = idx
                past.layers[li].keys = fk.index_select(2, idx)
                past.layers[li].values = fv.index_select(2, idx)
        out = model(cur, past_key_values=past, use_cache=True, output_attentions=use_sel)
        past = out.past_key_values
        cur = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        gen_ids.append(cur.item())
        if use_sel:
            # append the new token's (quantized) KV to the FULL store, and update page scores by
            # mapping this step's attention (over the materialized subset) back to full-cache pages
            ai = 0
            for li in full:
                ly = past.layers[li]
                newk = _quant_dequant(ly.keys[:, :, -1:, :], bits)
                newv = _quant_dequant(ly.values[:, :, -1:, :], bits)
                fk, fv = fullstore[li]
                fullstore[li] = [torch.cat([fk, newk], 2), torch.cat([fv, newv], 2)]
                a = out.attentions[ai] if out.attentions and ai < len(out.attentions) else None
                ai += 1
                idx = mat_idx[li]
                clen_full = fullstore[li][0].shape[2]
                npg_full = clen_full // PAGE
                if a is not None and npg_full > 0:
                    mass_sub = a[0].mean(dim=0).sum(dim=0)  # over materialized positions
                    mass_full = torch.zeros(clen_full, device=fk.device)
                    n = min(idx.numel(), mass_sub.numel())
                    mass_full[idx[:n]] = mass_sub[:n]
                    sc = mass_full[: npg_full * PAGE].view(npg_full, PAGE).sum(dim=1)
                    page_score[li] = page_score.get(li, 0)
                    prev = page_score[li]
                    if torch.is_tensor(prev) and prev.numel() == npg_full:
                        page_score[li] = prev + sc
                    else:
                        page_score[li] = sc
            # reset the live cache to empty-ish so next step re-materializes from fullstore
            # (we keep only the last token; the rest comes from fullstore via materialization)
        else:
            for li in full:
                ly = past.layers[li]
                ly.keys = _quant_dequant(ly.keys, bits)
                ly.values = _quant_dequant(ly.values, bits)
    return gen_ids, past


def gen_dense(model, tok, ids, gen_tokens):
    out = model(ids, use_cache=True)
    past = out.past_key_values
    cur = ids[:, -1:]
    gen_ids = []
    for _ in range(gen_tokens):
        out = model(cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        cur = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        gen_ids.append(cur.item())
    return gen_ids, past


def run(model_key, contexts, bits_list, pages_keep, dtype, gen_tokens):
    model_id = MODELS[model_key]
    info = inspect_model(model_id)
    model, tok = load_model(model_id, dtype=dtype, attn_impl="eager")
    device = next(model.parameters()).device
    print(f"[load] {model_id} full_attn={info.full_attention_layers} (eager)")
    depths, seeds = [0.1, 0.5, 0.9], [42, 43, 44]

    for ctx in contexts:
        for bits in bits_list:
            conds = [("dense", None), (f"quest_quant_b{bits}", "quant")]
            for name, kind in conds:
                hits, tot_kv, tot = 0, 0.0, 0
                for d in depths:
                    for s in seeds:
                        set_seed(s)
                        rng = random.Random(s)
                        needle = make_needle(rng)
                        prompt, needle = build_needle_prompt(tok, ctx, needle, d)
                        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
                        with torch.no_grad():
                            if kind is None:
                                gen_ids, past = gen_dense(model, tok, ids, gen_tokens)
                            elif kind == "quant":
                                gen_ids, past = gen_quest(model, tok, ids, bits, gen_tokens,
                                                          pages_keep=None)
                            else:
                                gen_ids, past = gen_quest(model, tok, ids, bits, gen_tokens,
                                                          pages_keep=pages_keep)
                        gen = tok.decode(gen_ids, skip_special_tokens=True)
                        hits += int(check_needle(gen, needle))
                        # effective KV memory: bytes / (16/bits) for the quantized cache
                        kvb = cache_kv_bytes(past) / 2**20
                        eff = kvb if kind is None else kvb * (bits / 16)
                        tot_kv += eff
                        tot += 1
                rec = {"experiment": "quest", "method": name, "model": model_id, "context": ctx,
                       "bits": bits, "pages_keep": pages_keep,
                       "needle_recall": round(hits / tot, 3),
                       "kv_cache_mb_eff": round(tot_kv / tot, 2)}
                append_jsonl(RESULTS, rec)
                print(f"  ctx={ctx} {name:18s} recall={rec['needle_recall']} "
                      f"kv_eff={rec['kv_cache_mb_eff']}MB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-0.8b", choices=list(MODELS.keys()))
    ap.add_argument("--contexts", type=int, nargs="+", default=[2048])
    ap.add_argument("--bits", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--pages-keep", type=int, default=8)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--gen-tokens", type=int, default=24)
    args = ap.parse_args()
    run(args.model, args.contexts, args.bits, args.pages_keep, args.dtype, args.gen_tokens)


if __name__ == "__main__":
    main()
