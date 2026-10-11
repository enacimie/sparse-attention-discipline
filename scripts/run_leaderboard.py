"""FASE 1 — Multi-method leaderboard on Qwen3.5 hybrids (Kaggle T4 16GB).
Runs fidelity (logit-cosine + pair_ratio) and needle retrieval for ~8 methods
across 3 models × 2 contexts. Checkpoints incrementally: interrupted sessions
resume from last completed (model, ctx, method).
"""

import argparse, json, math, os, random, sys, time
from collections import defaultdict

import torch
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from attention_interceptor import (apply_method, reset_pair_counter, restore,
                                   PAIR_COUNTER)
from attention_methods import AttentionMethod
from model_loader import MODELS, inspect_model, load_model
from needle import build_needle_prompt, check_needle, make_needle
from utils import append_jsonl, fidelity_metrics, set_seed

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "leaderboard.jsonl")
CHECKPOINT = os.path.join(os.path.dirname(__file__), "..", "results", "leaderboard_checkpoint.jsonl")


def load_done():
    """Return set of (model_name, context, method_name) already completed."""
    if not os.path.exists(CHECKPOINT):
        return set()
    done = set()
    with open(CHECKPOINT) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                done.add((rec["model"], rec["context"], rec["method"]))
    return done


def save_checkpoint(model_name, context, method_name):
    append_jsonl(CHECKPOINT, {"model": model_name, "context": context,
                              "method": method_name})


def build_methods():
    """Return list of (name, AttentionMethod_or_None_for_dense)."""
    methods = [
        ("dense", None),
        ("lg_topk_k32", AttentionMethod(name="local_global_topk", window=64,
                                        n_global=4, topk=32)),
        ("lg_randtopk_k32", AttentionMethod(name="local_global_randtopk", window=64,
                                            n_global=4, topk=32, seed=42)),
        ("lg_adaptive_m09", AttentionMethod(name="local_global_adaptive", window=64,
                                            n_global=4, topk=32, mass_tau=0.9)),
        ("lg_entropy", AttentionMethod(name="local_global_entropy", window=64,
                                       n_global=4, topk=32)),
        ("lg_perhead", AttentionMethod(name="local_global_perhead", window=64,
                                       n_global=4, topk=32)),
        ("streamingllm_bm", AttentionMethod(name="local_global", window=85, n_global=4)),
        ("uniform_random_m", AttentionMethod(name="uniform_random", topk=32, seed=42)),
    ]
    return methods


def run_fidelity(model, tok, method, full_layers, ctx, n_chunks=5):
    """Run fidelity on n_chunks of wikitext test split, return {cos, kl, mse, pr}."""
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    text = "\n".join(t for t in ds["text"] if t.strip())
    ids_all = tok(text, return_tensors="pt").input_ids[0]
    step = max(1, (len(ids_all) - ctx) // max(1, n_chunks))
    chunks = [ids_all[s:s + ctx] for s in range(0, len(ids_all) - ctx + 1, step)][:n_chunks]
    chunks = [c.unsqueeze(0).to(next(model.parameters()).device) for c in chunks]

    @torch.no_grad()
    def get_logits(m, inp):
        return m(inp).logits[0].cpu().float()

    cos_list, kl_list, mse_list, pr_list = [], [], [], []
    for chunk in chunks:
        ref_logits = get_logits(model, chunk)
        if method is not None:
            reset_pair_counter()
            originals = apply_method(model, method, full_layers)
            logits = get_logits(model, chunk)
            restore(model, originals, full_layers)
            pr = PAIR_COUNTER["computed"] / PAIR_COUNTER["full"] if PAIR_COUNTER["full"] else 0.0
        else:
            logits = ref_logits
            pr = 1.0
        fm = fidelity_metrics(logits, ref_logits)
        cos_list.append(fm["logit_cosine"])
        kl_list.append(fm["kl_div"])
        mse_list.append(fm["logit_mse"])
        pr_list.append(pr)

    return {
        "cos_mean": sum(cos_list) / len(cos_list),
        "cos_std": (sum((c - sum(cos_list)/len(cos_list))**2 for c in cos_list)
                     / (len(cos_list)-1))**0.5 if len(cos_list) > 1 else 0.0,
        "kl_mean": sum(kl_list) / len(kl_list),
        "mse_mean": sum(mse_list) / len(mse_list),
        "pr_mean": sum(pr_list) / len(pr_list),
    }


def run_needle(model, tok, method, full_layers, ctx, depths, seeds, gen_tokens=24, records=None):
    """Run needle-in-a-haystack and return recall.

    records: lista opcional a la que se anade un dict por sonda (depth, seed, hit, needle). Permite
    tests apareados entre metodos (McNemar) sobre las mismas sondas. Si es None el comportamiento es
    exactamente el original: solo se devuelven agregados.
    """
    hits, total = 0, 0
    device = next(model.parameters()).device
    for depth in depths:
        for seed in seeds:
            set_seed(seed)
            rng = random.Random(seed)
            needle = make_needle(rng)
            prompt, ndl = build_needle_prompt(tok, ctx, needle, depth)
            ids = tok(prompt, return_tensors="pt").input_ids.to(device)
            if method is not None:
                originals = apply_method(model, method, full_layers)
            out = model.generate(ids, max_new_tokens=gen_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            if method is not None:
                restore(model, originals, full_layers)
            gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            hit = int(check_needle(gen, ndl))
            hits += hit
            total += 1
            if records is not None:
                records.append({"depth": depth, "seed": seed, "hit": hit, "needle": ndl})
    return hits / total if total else 0.0, hits, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b-base"])
    parser.add_argument("--contexts", type=int, nargs="+", default=[2048, 4096])
    parser.add_argument("--fidelity-chunks", type=int, default=5)
    parser.add_argument("--needle-depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    parser.add_argument("--needle-seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--skip-needle", action="store_true")
    parser.add_argument("--skip-fidelity", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load models in 4-bit (for GPUs with <16GB VRAM)")
    args = parser.parse_args()

    done = load_done() if args.skip_existing else set()
    methods = build_methods()
    total_combos = len(args.models) * len(args.contexts) * len(methods)
    completed = 0

    print(f"=== LEADERBOARD: {len(args.models)} models × {len(args.contexts)} contexts × "
          f"{len(methods)} methods = {total_combos} combos ===")
    if done:
        print(f"Resuming — {len(done)} already completed, skipping those")

    for model_key in args.models:
        model_id = MODELS[model_key]
        try:
            info = inspect_model(model_id)
        except Exception as e:
            print(f"SKIP {model_key}: config error {e}")
            continue
        model, tok = load_model(model_id, load_in_4bit=args.load_in_4bit)
        fl = info.full_attention_layers
        print(f"\n[model] {model_key} ({model_id})  full_attn_layers={fl}  "
              f"q_heads={info.num_attention_heads}  kv_heads={info.num_key_value_heads}  "
              f"head_dim={info.head_dim}")

        for ctx in args.contexts:
            for name, method in methods:
                key = (model_key, ctx, name)
                if key in done:
                    completed += 1
                    continue

                t0 = time.time()
                record = {
                    "experiment": "leaderboard",
                    "model": model_key,
                    "model_id": model_id,
                    "context": ctx,
                    "method": name,
                    "n_full_attn": info.n_full_attention,
                    "total_layers": info.num_hidden_layers,
                }

                # Fidelity
                if not args.skip_fidelity:
                    fid = run_fidelity(model, tok, method, fl, ctx, args.fidelity_chunks)
                    record["cos_mean"] = round(fid["cos_mean"], 6)
                    record["cos_std"] = round(fid["cos_std"], 6)
                    record["kl_mean"] = round(fid["kl_mean"], 5)
                    record["mse_mean"] = round(fid["mse_mean"], 5)
                    record["pr_mean"] = round(fid["pr_mean"], 5)

                # Needle
                if not args.skip_needle:
                    recall, hits, total = run_needle(
                        model, tok, method, fl, ctx,
                        args.needle_depths, args.needle_seeds)
                    record["needle_recall"] = round(recall, 4)
                    record["needle_hits"] = hits
                    record["needle_total"] = total

                dt = time.time() - t0
                record["wall_time_s"] = round(dt, 1)
                append_jsonl(RESULTS, record)
                save_checkpoint(model_key, ctx, name)
                completed += 1

                fid_str = f"cos={record.get('cos_mean','?'):.4f}" if not args.skip_fidelity else ""
                ndl_str = f"needle={record.get('needle_recall','?'):.3f}" if not args.skip_needle else ""
                pr_str = f"pr={record.get('pr_mean','?'):.4f}" if not args.skip_fidelity else ""
                print(f"  [{completed}/{total_combos}] {model_key:18s} ctx={ctx:5d} "
                      f"{name:20s}  {fid_str}  {ndl_str}  {pr_str}  ({dt:.0f}s)")

        # Free model memory before loading next
        del model, tok
        torch.cuda.empty_cache()

    print(f"\n=== DONE: {completed}/{total_combos} combos completed ===")
    print(f"Results -> {RESULTS}")


if __name__ == "__main__":
    main()
