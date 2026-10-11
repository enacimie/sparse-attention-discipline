"""Reviewer M3: seed-resolved CI repeats for the two central fidelity tables.

The main 9B table (PAPER.md 4.1) and the dense 0.6B/1.7B table (4.3) were single-run
(n=2-5 samples, one random mask seed). This re-runs them with `--seeds` independent
random-mask seeds and seed-shifted wikitext chunks, storing the PER-SAMPLE cosine list
so mean/std/95% CI (over seed x sample) can be computed without reruns.

Output: results/ci_repeat.jsonl
  {"experiment": "ci_repeat", "model", "context", "method", "seed", "cos_list": [...],
   "pair_ratio", "n_samples"}

Usage:
  python scripts/run_ci_repeat.py --model qwen3.5-9b-base --contexts 1024 2048 --seeds 42 43 44 45
  python scripts/run_ci_repeat.py --model qwen3-0.6b --contexts 1024 2048 4096 --seeds 42 43 44 45
  python scripts/run_ci_repeat.py --model qwen3-1.7b --contexts 1024 2048 4096 --seeds 42 43 44 45
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from attention_interceptor import (apply_method, reset_pair_counter, restore,
                                   PAIR_COUNTER)  # noqa: E402
from attention_methods import AttentionMethod  # noqa: E402
from model_loader import MODELS, inspect_model, load_model  # noqa: E402
from utils import append_jsonl, fidelity_metrics, set_seed  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "ci_repeat.jsonl")


@torch.no_grad()
def get_logits(model, ids):
    return model(ids).logits[0].cpu().float()


def build_chunks(tok, ctx, n_samples, seed):
    """Seed-shifted wikitext chunks (shift by seed so each seed sees different text)."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    text = "\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    stride = max(1, (ids.numel() - ctx) // max(1, n_samples + 1))
    chunks, s = [], seed * stride % max(1, ids.numel() - ctx)
    for _ in range(n_samples):
        if s + ctx > ids.numel():
            break
        chunks.append(ids[s:s + ctx])
        s += stride
    return chunks or [ids[:ctx]]


def measure_pair_ratio(model, ids, device, method, full_layers):
    orig = apply_method(model, method, full_layers)
    try:
        reset_pair_counter()
        with torch.no_grad():
            model(ids.unsqueeze(0).to(device))
        return PAIR_COUNTER["computed"] / max(1, PAIR_COUNTER["full"])
    finally:
        restore(model, orig, full_layers)


def run(model_key, contexts, seeds, dtype, n_samples, load_in_4bit, device_map, cpu_offload,
        methods=('lg_topk', 'uniform_random'), out=None):
    results_path = out or RESULTS
    model_id = MODELS[model_key]
    info = inspect_model(model_id)
    if cpu_offload:
        # 9B on an 8 GB card: fp16 + auto device map (spills MLPs to CPU/disk). Slow but
        # correct; bnb 4-bit offload paths are broken in this transformers/bnb combo.
        import transformers
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True,
            max_memory={0: "5.2GiB", "cpu": "7GiB"})
        tok = transformers.AutoTokenizer.from_pretrained(model_id)
    else:
        model, tok = load_model(model_id, dtype=dtype, load_in_4bit=load_in_4bit,
                                device_map=device_map)
    full_layers = inspect_model(model_id).full_attention_layers
    layer_list = None
    for n, mod in model.named_modules():
        if n.endswith("layers") and isinstance(mod, torch.nn.ModuleList):
            layer_list = mod
            break
    attn0 = getattr(layer_list[full_layers[0]], "self_attn", None) or \
        getattr(layer_list[full_layers[0]], "attention", None)
    device = next(attn0.parameters()).device
    print(f"[load] {model_id} device={device} full_attn={len(full_layers)}")

    for ctx in contexts:
        lg = AttentionMethod(name="local_global_topk", window=64, n_global=4, topk=32)
        chunks0 = build_chunks(tok, ctx, n_samples, seeds[0])
        full_layers = inspect_model(model_id).full_attention_layers
        pr_lg = measure_pair_ratio(model, chunks0[0], device, lg, full_layers)
        # budget-matched uniform-random: avg causal row length is (ctx+1)/2, so a row-wise
        # top-(total_k) random mask yields pr ~= total_k/avg_rowlen ~= 2*total_k/ctx
        pr_rand = 0.0
        topk_try = max(8, round(pr_lg * ctx / 2))
        for _ in range(6):
            ru = AttentionMethod(name="uniform_random", topk=topk_try, random_k=0, seed=seeds[0])
            pr_rand = measure_pair_ratio(model, chunks0[0], device, ru, full_layers)
            if abs(pr_rand - pr_lg) < 0.01:
                break
            topk_try = max(4, topk_try + round((pr_lg - pr_rand) * ctx / 2))
        print(f"  ctx={ctx} lg_topk pr={pr_lg:.4f} random topk={topk_try} pr={pr_rand:.4f}")

        for method in methods:
            for seed in seeds:
                set_seed(seed)
                if method == "lg_topk":
                    m = AttentionMethod(name="local_global_topk", window=64, n_global=4,
                                        topk=32)
                elif method == "lg_randtopk":
                    # control ESTRUCTURALMENTE igualado: misma ventana y mismos sinks, free
                    # keys aleatorizadas. El seed va en el metodo para que cada semilla de
                    # una realizacion distinta del control.
                    m = AttentionMethod(name="local_global_randtopk", window=64, n_global=4,
                                        topk=32, seed=seed)
                elif method == "uniform_random":
                    m = AttentionMethod(name="uniform_random", topk=topk_try, random_k=0,
                                        seed=seed)
                else:
                    raise ValueError(f"metodo desconocido: {method}")
                chunks = build_chunks(tok, ctx, n_samples, seed)
                cos_list = []
                for ids in chunks:
                    orig = apply_method(model, m, full_layers)
                    try:
                        with torch.no_grad():
                            sparse = get_logits(model, ids.unsqueeze(0).to(device))
                    finally:
                        restore(model, orig, full_layers)
                    with torch.no_grad():
                        dense = get_logits(model, ids.unsqueeze(0).to(device))
                    cos = float(fidelity_metrics(sparse, dense)["logit_cosine"])
                    cos_list.append(cos)
                print(f"  ctx={ctx:5d} {method:14s} seed={seed} cos={[round(c, 4) for c in cos_list]}")
                append_jsonl(results_path, {"experiment": "ci_repeat", "model": model_id,
                                       "context": ctx, "method": method, "seed": seed,
                                       "cos_list": cos_list, "pair_ratio": round(
                                           pr_rand if method == "uniform_random" else pr_lg, 4),
                                       "n_samples": len(cos_list)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS.keys()))
    ap.add_argument("--contexts", type=int, nargs="+", default=[1024, 2048])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--no-quantize", action="store_true")
    ap.add_argument("--device-map", default="cuda")
    ap.add_argument("--cpu-offload", action="store_true",
                    help="9B: fp16 + auto device map (spills to CPU/disk)")
    ap.add_argument("--methods", nargs="+", default=["lg_topk", "uniform_random"],
                    choices=["lg_topk", "lg_randtopk", "uniform_random"],
                    help="lg_randtopk es el control estructuralmente igualado")
    ap.add_argument("--out", default=None,
                    help="ruta JSONL de salida; por defecto results/ci_repeat.jsonl")
    args = ap.parse_args()
    run(args.model, args.contexts, args.seeds, args.dtype, args.n_samples,
        load_in_4bit=not args.no_quantize, device_map=args.device_map,
        cpu_offload=args.cpu_offload, methods=args.methods, out=args.out)


if __name__ == "__main__":
    main()
