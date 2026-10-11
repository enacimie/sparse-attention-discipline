"""Experiment 1: dense baseline + profiling.

Measures for a given model and a list of context lengths:
  perplexity (wikitext-2), prefill latency, decode latency, tokens/s,
  peak VRAM, KV-cache memory, GPU temperature.

Results are appended to results/baseline.jsonl. Greedy decoding, fixed seed.

Usage (on the remote ULL machine):
  source ~/activate_mlattention.sh
  python scripts/run_baseline.py --model lfm2.5-350m --contexts 512 1024
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model_loader import MODELS, inspect_model, load_model  # noqa: E402
from utils import GpuMeter, append_jsonl, set_seed  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "baseline.jsonl")


def get_eval_text(tokenizer, seq_len: int, n_docs: int = 4):
    """Load wikitext-2 and return tokenized chunks of seq_len for perplexity."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    chunks = [ids[i:i + seq_len] for i in range(0, min(n_docs * seq_len, len(ids) - seq_len - 1), seq_len)]
    return [c for c in chunks if c.numel() == seq_len]


@torch.no_grad()
def measure_perplexity(model, chunks, device):
    """Teacher-forced perplexity: standard token-weighted mean of NLL.

    NOTE: absolute PPL is only a reference. Instruct-tuned hybrid models give
    erratic absolute PPL on raw wiki text. The PRIMARY quality metric for this
    study is fidelity to the dense baseline (KL/MSE on logits), not absolute PPL.
    """
    total_nll, total_tok = 0.0, 0
    for c in chunks:
        ids = c.unsqueeze(0).to(device)
        out = model(ids, labels=ids)
        n_tok = ids.shape[1] - 1  # predicted tokens (labels shifted internally)
        total_nll += out.loss.item() * n_tok
        total_tok += n_tok
    return float(np.exp(total_nll / total_tok))


@torch.no_grad()
def capture_reference_logits(model, chunks, device):
    """Capture dense-baseline logits on eval chunks (reference for fidelity)."""
    refs = []
    for c in chunks:
        ids = c.unsqueeze(0).to(device)
        out = model(ids)
        refs.append(out.logits[0].float().cpu())  # (seq_len, vocab)
    return refs


@torch.no_grad()
def measure_latency(model, seq_len: int, gen_tokens: int, device, vocab: int):
    """Measure prefill time and per-token decode time with a random prompt."""
    ids = torch.randint(1000, vocab - 1, (1, seq_len), device=device)
    # Prefill
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model(ids, use_cache=True)
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0
    past = out.past_key_values
    # KV cache memory: transformers 5.x DynamicCache exposes .layers. Full-attention
    # layers store keys/values tensors; linear/conv layers store conv/recurrent states.
    kv_bytes = 0
    if past is not None and hasattr(past, "layers"):
        for ly in past.layers:
            for attr in ("keys", "values", "key_cache", "value_cache"):
                t = getattr(ly, attr, None)
                if torch.is_tensor(t):
                    kv_bytes += t.numel() * t.element_size()
    kv_mb = kv_bytes / 2**20
    # Decode: greedy, one token at a time
    cur = ids[:, -1:]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(gen_tokens):
        out = model(cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        cur = nxt
    torch.cuda.synchronize()
    decode_s = time.perf_counter() - t0
    return {
        "prefill_s": round(prefill_s, 4),
        "prefill_tokens_per_s": round(seq_len / prefill_s, 1),
        "decode_ms_per_token": round(1000 * decode_s / gen_tokens, 2),
        "decode_tokens_per_s": round(gen_tokens / decode_s, 1),
        "kv_cache_mb": round(kv_mb, 1),
    }


def run(model_key: str, contexts, dtype, gen_tokens: int, seed: int, load_in_4bit: bool,
        skip_ppl: bool):
    set_seed(seed)
    model_id = MODELS[model_key]
    info = inspect_model(model_id)
    print(f"[load] {model_id} | type={info.model_type} | layers={info.num_hidden_layers} "
          f"| full_attn={info.full_attention_layers} ({info.n_full_attention} layers)")

    with GpuMeter() as load_meter:
        model, tok = load_model(model_id, dtype=dtype, load_in_4bit=load_in_4bit)
    device = next(model.parameters()).device
    vocab = model.get_input_embeddings().weight.shape[0]
    print(f"[loaded] peak_vram_load={load_meter.peak_vram_mb:.0f}MB | device={device}")

    for seq_len in contexts:
        print(f"[ctx={seq_len}] evaluating...")
        chunks = get_eval_text(tok, seq_len)
        with GpuMeter() as m:
            if skip_ppl:
                ppl = float("nan")
            else:
                ppl = measure_perplexity(model, chunks, device)
            lat = measure_latency(model, seq_len, gen_tokens, device, vocab)
        rec = {
            "experiment": "baseline",
            "model": model_id,
            "model_key": model_key,
            "model_type": info.model_type,
            "num_layers": info.num_hidden_layers,
            "full_attention_layers": info.full_attention_layers,
            "n_full_attention": info.n_full_attention,
            "dtype": dtype,
            "load_in_4bit": load_in_4bit,
            "seed": seed,
            "context": seq_len,
            "perplexity": round(ppl, 3),
            "model_load_peak_vram_mb": round(load_meter.peak_vram_mb, 1),
            **lat,
            **m.to_dict(),
        }
        append_jsonl(RESULTS, rec)
        print(f"  -> ppl={rec['perplexity']} prefill={lat['prefill_s']}s "
              f"decode={lat['decode_ms_per_token']}ms/tok peak_vram={rec['peak_vram_mb']}MB "
              f"kv={rec['kv_cache_mb']}MB temp={rec['temp_c']}C")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS.keys()))
    ap.add_argument("--contexts", type=int, nargs="+", default=[512])
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--skip-ppl", action="store_true",
                    help="skip perplexity; measure efficiency only")
    args = ap.parse_args()
    run(args.model, args.contexts, args.dtype, args.gen_tokens, args.seed,
        args.load_in_4bit, args.skip_ppl)


if __name__ == "__main__":
    main()
