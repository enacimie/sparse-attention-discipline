"""FASE0.5: Sinks g=0 ablation — does needle retrieval survive without attention sinks?
Runs lg_topk with n_global=0 alongside lg_topk n_global=4 (standard) on 0.8B ctx 2048.
"""
import argparse, os, random, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from attention_interceptor import apply_method, restore
from attention_methods import AttentionMethod
from model_loader import MODELS, inspect_model, load_model
from needle import build_needle_prompt, check_needle, make_needle
from utils import append_jsonl, set_seed

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "sinks_g0_needle.jsonl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3.5-0.8b")
    parser.add_argument("--ctx", type=int, default=2048)
    parser.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()

    model_id = MODELS[args.model]
    info = inspect_model(model_id)
    fl = info.full_attention_layers
    print(f"Loading {model_id}  |  full_attn_layers={fl}")
    model, tok = load_model(model_id)
    device = next(model.parameters()).device

    methods = [
        ("lg_topk_g4", AttentionMethod(name="local_global_topk", window=64,
                                       n_global=4, topk=32)),
        ("lg_topk_g0", AttentionMethod(name="local_global_topk", window=64,
                                       n_global=0, topk=32)),
    ]

    for ctx in [args.ctx]:
        for name, method in methods:
            originals = apply_method(model, method, fl)
            hits = total = 0
            try:
                for depth in args.depths:
                    h = 0
                    for seed in args.seeds:
                        set_seed(seed)
                        rng = random.Random(seed)
                        needle = make_needle(rng)
                        prompt, ndl = build_needle_prompt(tok, ctx, needle, depth)
                        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
                        out = model.generate(ids, max_new_tokens=24, do_sample=False,
                                             pad_token_id=tok.eos_token_id)
                        gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
                        hit = check_needle(gen, ndl)
                        hits += int(hit); h += int(hit); total += 1
                        append_jsonl(RESULTS, {
                            "experiment": "sinks_g0_ablation",
                            "method": name, "model": model_id, "context": ctx,
                            "depth": depth, "seed": seed, "hit": bool(hit),
                            "needle": ndl, "generated": gen[:80],
                        })
                    print(f"  {name:15s}  ctx={ctx}  depth={depth:.1f}  "
                          f"recall={h/len(args.seeds):.2f}")
            finally:
                restore(model, originals, fl)
            print(f"  {name:15s}  OVERALL  recall={hits}/{total}={hits/total:.3f}\n")

    print(f"Done. -> {RESULTS}")


if __name__ == "__main__":
    main()
