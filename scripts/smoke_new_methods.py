"""Smoke test for the three new content-selection methods: local_global_adaptive,
local_global_entropy, local_global_perhead.

Runs one wikitext chunk (ctx 512) on 0.8B, comparing logit cosine vs dense + pair_ratio.
Quick sanity check before adding them to the full harness.
"""

import argparse, math, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from attention_interceptor import (apply_method, reset_pair_counter, restore,
                                   PAIR_COUNTER)
from attention_methods import AttentionMethod
from model_loader import MODELS, inspect_model, load_model
from utils import append_jsonl, fidelity_metrics, set_seed

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "smoke_new_methods.jsonl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3.5-0.8b")
    parser.add_argument("--ctx", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    model_id = MODELS[args.model]
    print(f"Loading {model_id}")
    model, tok = load_model(model_id)
    device = next(model.parameters()).device
    info = inspect_model(model_id)
    fl = info.full_attention_layers

    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    text = "\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0][:args.ctx]
    input_ids = ids.unsqueeze(0).to(device)

    @torch.no_grad()
    def get_logits(m, inp):
        return m(inp).logits[0].cpu().float()

    ref_logits = get_logits(model, input_ids)

    methods = [
        AttentionMethod(name="local_global_topk", window=64, n_global=4, topk=32),
        AttentionMethod(name="local_global_adaptive", window=64, n_global=4,
                        topk=32, mass_tau=0.9),
        AttentionMethod(name="local_global_entropy", window=64, n_global=4, topk=32),
        AttentionMethod(name="local_global_perhead", window=64, n_global=4, topk=32),
    ]

    for m in methods:
        reset_pair_counter()
        originals = apply_method(model, m, fl)
        logits = get_logits(model, input_ids)
        restore(model, originals, fl)

        fm = fidelity_metrics(logits, ref_logits)
        cos = fm["logit_cosine"]
        kl = fm["kl_div"]
        computed = PAIR_COUNTER["computed"]
        full = PAIR_COUNTER["full"]
        pr = computed / full if full else 0.0
        record = {
            "experiment": "smoke_new_methods",
            "model": args.model,
            "ctx": args.ctx,
            "seed": args.seed,
            "method": m.name,
            "cos": round(cos, 6),
            "kl": round(kl, 6),
            "computed_pairs": computed,
            "full_pairs": full,
            "pair_ratio": round(pr, 6),
        }
        append_jsonl(RESULTS, record)
        n_content = PAIR_COUNTER["computed"] - (args.ctx * (64 + 8 + (64 - 1) / 2))  # rough
        print(f"{m.name:30s}  cos={cos:.4f}  kl={kl:.6f}  pairs={computed}/{full}  "
              f"pr={pr:.4f}")

    print(f"Done. Results -> {RESULTS}")


if __name__ == "__main__":
    main()
