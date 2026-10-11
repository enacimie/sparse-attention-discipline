"""LFM2.5-8B-A1B MoE with CPU OFFLOAD on a T4.

bitsandbytes nf4 does not quantize lfm2_moe experts, so the model loads in fp16 (~14.4GB) and does
not fit a 14.5GB T4 outright. We use accelerate device_map="auto" + max_memory to keep ~11GB on the
GPU and spill the rest to CPU. A smoke test runs first: if the SDPA-mask interceptor does not cooperate
with accelerate's offload hooks (device alignment), we fail fast with a clear message.

Usage:
  python scripts/run_moe_offload.py --ctx 1024 --n-samples 3
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from attention_interceptor import apply_method, restore  # noqa: E402
from attention_methods import AttentionMethod  # noqa: E402
from model_loader import MODELS, inspect_model  # noqa: E402
from utils import append_jsonl, fidelity_metrics, set_seed  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "moe_offload.jsonl")


@torch.no_grad()
def get_logits(model, ids):
    return model(ids).logits[0].cpu().float()


def build_chunks(tok, ctx, n_samples, split="test"):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
    text = "\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    chunks, step = [], max(1, (ids.numel() - ctx) // max(1, n_samples))
    for s in range(0, ids.numel() - ctx + 1, step):
        chunks.append(ids[s:s + ctx])
        if len(chunks) >= n_samples:
            break
    return chunks if chunks else [ids[:ctx]]


def mean_std(xs):
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.0
    return m, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--gpu-gb", type=int, default=11)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = MODELS["lfm2.5-8b-a1b"]
    info = inspect_model(model_id)
    print(f"[info] {model_id} full_attn={info.full_attention_layers} "
          f"heads={info.num_attention_heads} kv={info.num_key_value_heads}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto",
        max_memory={0: f"{args.gpu_gb}GiB", "cpu": "12GiB"},
        attn_implementation="sdpa", low_cpu_mem_usage=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    model.eval()
    in_device = model.get_input_embeddings().weight.device
    print(f"[load] done; input device={in_device}", flush=True)

    chunks = [c.unsqueeze(0).to(in_device) for c in build_chunks(tok, args.ctx, args.n_samples)]
    full_layers = info.full_attention_layers

    # ---- smoke test: dense forward then local_global_topk forward on chunk 0 ----
    set_seed(args.seed)
    try:
        ref = get_logits(model, chunks[0])
    except Exception as e:
        print(f"[smoke] DENSE forward FAILED: {e}", flush=True)
        raise
    method = AttentionMethod(name="local_global_topk", window=64, n_global=4, topk=32)
    originals = apply_method(model, method, full_layers)
    try:
        set_seed(args.seed)
        approx = get_logits(model, chunks[0])
    except Exception as e:
        print(f"[smoke] SPARSE forward FAILED under offload: {e}", flush=True)
        restore(model, originals, full_layers)
        raise
    finally:
        restore(model, originals, full_layers)
    sm = fidelity_metrics(approx, ref)
    print(f"[smoke] local_global_topk chunk0 cos={sm['logit_cosine']:.4f}", flush=True)
    append_jsonl(RESULTS, {"experiment": "moe_offload", "part": "smoke", "model": model_id,
                           "context": args.ctx, "cos": sm["logit_cosine"]})

    # ---- full Part A ----
    refs = []
    for c in chunks:
        set_seed(args.seed)
        refs.append(get_logits(model, c))
    budget = 64 + 4 + 32
    methods = [("local_global_topk", AttentionMethod(name="local_global_topk", window=64,
                                                     n_global=4, topk=32)),
               ("streamingllm_bm", AttentionMethod(name="local_global", window=budget - 4,
                                                   n_global=4)),
               ("uniform_random_m", AttentionMethod(name="uniform_random", topk=budget,
                                                    random_k=0, seed=args.seed))]
    for tag, m in methods:
        originals = apply_method(model, m, full_layers)
        try:
            vals = []
            for c, r in zip(chunks, refs):
                set_seed(args.seed)
                vals.append(fidelity_metrics(get_logits(model, c), r)["logit_cosine"])
        finally:
            restore(model, originals, full_layers)
        mean, sd = mean_std(vals)
        append_jsonl(RESULTS, {"experiment": "moe_offload", "part": "A_heldout", "method": tag,
                               "model": model_id, "context": args.ctx,
                               "n_samples": len(chunks), "cos_mean": round(mean, 4),
                               "cos_std": round(sd, 4)})
        print(f"  [A] {tag:18s} cos={mean:.4f}+/-{sd:.4f}", flush=True)


if __name__ == "__main__":
    main()
