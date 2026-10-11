"""Generation-time KV-eviction baselines the reviewer named: H2O and SnapKV.

Experiment 15 already compared dense vs recency vs k_norm eviction (needle recall). Here we add the
two SOTA decode-time KV-budget methods and compare at a MATCHED KV budget:
  - H2O (Heavy-Hitter Oracle): accumulate per-key attention mass over decode steps; keep heavy
    hitters + recent + sinks.
  - SnapKV: at the end of prefill, score keys by the attention they receive from a recent
    observation window; keep the top keys, then decode with that reduced cache.
Both need attention weights, so the model is loaded with attn_implementation="eager" and
output_attentions=True. Metric is needle recall (as in run_kv_eviction) so numbers are comparable.

Usage:
  python scripts/run_h2o_snapkv.py --model qwen3.5-0.8b --contexts 2048 --keep 128 256 --n-sink 4
"""
import argparse
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model_loader import MODELS, inspect_model, load_model  # noqa: E402
from kv_eviction import RecencyEviction, KNormEviction, evict_cache, cache_kv_bytes  # noqa: E402
from needle import build_needle_prompt, check_needle, make_needle  # noqa: E402
from utils import GpuMeter, append_jsonl, set_seed  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "h2o_snapkv.jsonl")


def _full_layers_with_kv(past):
    """Indices of cache layers that are full-attention (have keys/values tensors)."""
    idx = []
    for i, ly in enumerate(past.layers):
        if torch.is_tensor(getattr(ly, "keys", None)):
            idx.append(i)
    return idx


def _attn_for_full_layers(attentions, past):
    """Map the output_attentions tuple to {cache_layer_idx: attn_tensor} for full-attn layers."""
    out = {}
    if attentions is None:
        return out
    # attentions is indexed by model layer; full-attn layers produce a tensor, others None
    ai = 0
    for i, ly in enumerate(past.layers):
        if not torch.is_tensor(getattr(ly, "keys", None)):
            continue
        if ai < len(attentions) and attentions[ai] is not None:
            out[i] = attentions[ai]
        ai += 1
    return out


def gen_h2o(model, tok, ids, keep, n_sink, gen_tokens, obs_recent):
    """H2O: accumulate attention mass per key, evict lowest (keep heavy hitters + recent + sinks)."""
    out = model(ids, use_cache=True, output_attentions=True)
    past = out.past_key_values
    full = _full_layers_with_kv(past)
    scores = {}
    # accumulate from prefill attention
    for li, a in _attn_for_full_layers(out.attentions, past).items():
        scores[li] = a[0].mean(dim=0).sum(dim=0).float()  # sum over queries, mean over heads
    cur = ids[:, -1:]
    gen_ids = []
    for _ in range(gen_tokens):
        out = model(cur, past_key_values=past, use_cache=True, output_attentions=True)
        past = out.past_key_values
        for li, a in _attn_for_full_layers(out.attentions, past).items():
            add = a[0].mean(dim=0).sum(dim=0).float()  # (kv_len,)
            if li in scores and scores[li].numel() == add.numel():
                scores[li] = scores[li] + add
            else:
                scores[li] = add
        # evict per full-attn layer by accumulated score
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
            protected = torch.cat([sink, recent]).unique()
            sc2 = sc.clone()
            sc2[protected] = float("-inf")
            n_keep = max(0, keep - protected.numel())
            top = sc2.topk(min(n_keep, clen)).indices
            idx = torch.cat([protected, top]).unique().sort().values
            ly.keys = k.index_select(2, idx)
            ly.values = ly.values.index_select(2, idx)
            if li in scores:
                scores[li] = scores[li][idx]
        cur = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        gen_ids.append(cur.item())
    return gen_ids, past


def gen_snapkv(model, tok, ids, keep, n_sink, gen_tokens, obs_window):
    """SnapKV: score keys by attention from the last obs_window queries at prefill, keep top, decode."""
    out = model(ids, use_cache=True, output_attentions=True)
    past = out.past_key_values
    full = _full_layers_with_kv(past)
    # one-time prefill selection using the observation window
    for li, a in _attn_for_full_layers(out.attentions, past).items():
        ly = past.layers[li]
        k = ly.keys
        clen = k.shape[2]
        if clen <= keep + n_sink:
            continue
        obs = a[0][:, -obs_window:, :]          # (heads, W, kv_len)
        sc = obs.mean(dim=0).sum(dim=0).float()  # (kv_len,)
        sink = torch.arange(n_sink, device=k.device)
        recent = torch.arange(max(0, clen - obs_window), clen, device=k.device)
        protected = torch.cat([sink, recent]).unique()
        sc2 = sc.clone()
        sc2[protected] = float("-inf")
        n_keep = max(0, keep - protected.numel())
        top = sc2.topk(min(n_keep, clen)).indices
        idx = torch.cat([protected, top]).unique().sort().values
        ly.keys = k.index_select(2, idx)
        ly.values = ly.values.index_select(2, idx)
    cur = ids[:, -1:]
    gen_ids = []
    for _ in range(gen_tokens):
        out = model(cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        cur = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        gen_ids.append(cur.item())
    return gen_ids, past


def gen_policy(model, tok, ids, policy, gen_tokens):
    out = model(ids, use_cache=True)
    past = out.past_key_values
    if policy is not None:
        evict_cache(past, policy)
    cur = ids[:, -1:]
    gen_ids = []
    for _ in range(gen_tokens):
        out = model(cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        if policy is not None:
            evict_cache(past, policy)
        cur = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        gen_ids.append(cur.item())
    return gen_ids, past


def run(model_key, contexts, keeps, n_sink, dtype, gen_tokens, obs_window):
    model_id = MODELS[model_key]
    info = inspect_model(model_id)
    model, tok = load_model(model_id, dtype=dtype, attn_impl="eager")
    device = next(model.parameters()).device
    print(f"[load] {model_id} full_attn={info.full_attention_layers} (eager)")
    depths, seeds = [0.1, 0.5, 0.9], [42, 43, 44]

    for ctx in contexts:
        for keep in keeps:
            conds = []
            conds.append(("dense", None, "none"))
            conds.append((f"recency+sink", RecencyEviction(keep=keep, n_sink=n_sink), "policy"))
            conds.append((f"knorm+sink", KNormEviction(keep=keep, n_sink=n_sink), "policy"))
            conds.append((f"h2o", keep, "h2o"))
            conds.append((f"snapkv", keep, "snapkv"))
            for name, arg, kind in conds:
                hits, tot_kv, tot = 0, 0.0, 0
                for d in depths:
                    for s in seeds:
                        set_seed(s)
                        rng = random.Random(s)
                        needle = make_needle(rng)
                        prompt, needle = build_needle_prompt(tok, ctx, needle, d)
                        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
                        with torch.no_grad():
                            if kind == "h2o":
                                gen_ids, past = gen_h2o(model, tok, ids, keep, n_sink,
                                                        gen_tokens, obs_recent=16)
                            elif kind == "snapkv":
                                gen_ids, past = gen_snapkv(model, tok, ids, keep, n_sink,
                                                           gen_tokens, obs_window)
                            else:
                                gen_ids, past = gen_policy(model, tok, ids, arg, gen_tokens)
                        gen = tok.decode(gen_ids, skip_special_tokens=True)
                        hits += int(check_needle(gen, needle))
                        tot_kv += cache_kv_bytes(past) / 2**20
                        tot += 1
                rec = {"experiment": "h2o_snapkv", "method": name, "model": model_id,
                       "context": ctx, "keep": keep, "n_sink": n_sink,
                       "needle_recall": round(hits / tot, 3),
                       "kv_cache_mb": round(tot_kv / tot, 2)}
                append_jsonl(RESULTS, rec)
                print(f"  ctx={ctx} keep={keep} {name:14s} recall={rec['needle_recall']} "
                      f"kv={rec['kv_cache_mb']}MB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-0.8b", choices=list(MODELS.keys()))
    ap.add_argument("--contexts", type=int, nargs="+", default=[2048])
    ap.add_argument("--keep", type=int, nargs="+", default=[128, 256])
    ap.add_argument("--n-sink", type=int, default=4)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--gen-tokens", type=int, default=24)
    ap.add_argument("--obs-window", type=int, default=32)
    args = ap.parse_args()
    run(args.model, args.contexts, args.keep, args.n_sink, args.dtype, args.gen_tokens,
        args.obs_window)


if __name__ == "__main__":
    main()
