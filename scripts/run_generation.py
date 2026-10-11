"""AUTOREGRESSIVE generation under sparse attention (reviewer Q3: do logit errors accumulate?).

All fidelity results so far are single-forward. In generation, each step conditions on previously
generated tokens, so small logit errors can compound. We generate greedily with DENSE vs SPARSE
attention from the same prompts and measure token-level agreement: the fraction of generated
positions where sparse picks the same token as dense. High agreement => generation is robust to the
sparsification; rapid divergence => errors accumulate.

Usage:
  python scripts/run_generation.py --model qwen3.5-0.8b --n-prompts 5 --prompt-len 512 --gen-len 128
"""
import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from attention_interceptor import apply_method, restore  # noqa: E402
from attention_methods import AttentionMethod  # noqa: E402
from model_loader import MODELS, inspect_model, load_model  # noqa: E402
from utils import append_jsonl, set_seed  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "generation.jsonl")


def build_prompts(tok, n_prompts, prompt_len, split="test"):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
    text = "\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    step = max(1, (ids.numel() - prompt_len) // max(1, n_prompts))
    prompts = []
    for s in range(0, ids.numel() - prompt_len + 1, step):
        prompts.append(ids[s:s + prompt_len])
        if len(prompts) >= n_prompts:
            break
    return prompts


@torch.no_grad()
def gen_tokens(model, prompt, gen_len):
    out = model.generate(prompt, max_new_tokens=gen_len, do_sample=False,
                         pad_token_id=model.config.eos_token_id)
    return out[0, prompt.shape[1]:]


@torch.no_grad()
def gen_tokens_sample(model, prompt, gen_len, temperature=1.0):
    out = model.generate(prompt, max_new_tokens=gen_len, do_sample=True,
                         temperature=temperature, top_k=0,
                         pad_token_id=model.config.eos_token_id)
    return out[0, prompt.shape[1]:]


def _agree(a_gens, b_gens):
    ags = []
    for ag_, bg_ in zip(a_gens, b_gens):
        n = min(ag_.numel(), bg_.numel())
        ags.append((ag_[:n] == bg_[:n]).float().mean().item())
    return sum(ags) / len(ags)


@torch.no_grad()
def ppl_under_dense(model, prompt, gen):
    """Perplexity of prompt+gen under the (dense) model, loss on the generated tokens only."""
    full = torch.cat([prompt, gen.unsqueeze(0)], dim=1)
    logits = model(full).logits[0]
    plen = prompt.shape[1]
    preds = logits[plen - 1:-1].float()          # predict each generated token
    logp = torch.log_softmax(preds, dim=-1)
    nll = -logp[torch.arange(gen.numel()), gen].mean().item()
    return math.exp(nll)


def run(model_key, n_prompts, prompt_len, gen_len, dtype, seed, load_in_4bit):
    model_id = MODELS[model_key]
    info = inspect_model(model_id)
    model, tok = load_model(model_id, dtype=dtype, load_in_4bit=load_in_4bit)
    device = next(model.parameters()).device
    full_layers = info.full_attention_layers
    prompts = [p.unsqueeze(0).to(device) for p in build_prompts(tok, n_prompts, prompt_len)]
    print(f"[load] {model_id} n_prompts={len(prompts)} prompt_len={prompt_len} gen_len={gen_len}")

    method = AttentionMethod(name="local_global_topk", window=64, n_global=4, topk=32)

    # dense generations
    set_seed(seed)
    dense_gens = [gen_tokens(model, p, gen_len) for p in prompts]

    # sparse generations
    originals = apply_method(model, method, full_layers)
    try:
        set_seed(seed)
        sparse_gens = [gen_tokens(model, p, gen_len) for p in prompts]
    finally:
        restore(model, originals, full_layers)

    mean_ag = _agree(dense_gens, sparse_gens)

    # CONTROL (reviewer Q4): contextualize the dense-vs-sparse agreement with the dense
    # model's agreement WITH ITSELF.
    #  (a) greedy is deterministic -> dense-vs-dense greedy self-agreement must be ~1.0
    #      (sanity check that the metric is not trivially low).
    #  (b) sampling with two different seeds -> the model's own stochastic divergence;
    #      if this is also low, single-digit sparse-vs-dense agreement is not alarming.
    set_seed(seed + 7777)
    dense_greedy2 = [gen_tokens(model, p, gen_len) for p in prompts]
    greedy_self_ag = _agree(dense_gens, dense_greedy2)
    set_seed(seed)
    dense_samp_a = [gen_tokens_sample(model, p, gen_len) for p in prompts]
    set_seed(seed + 1000)
    dense_samp_b = [gen_tokens_sample(model, p, gen_len) for p in prompts]
    sample_self_ag = _agree(dense_samp_a, dense_samp_b)

    # quality: perplexity of each generated continuation under the DENSE model (model is
    # restored to dense here). Similar PPL => sparse text is coherent (just different);
    # much higher PPL => sparse generation is degraded/incoherent.
    ppl_dense_gen = [ppl_under_dense(model, p, dg) for p, dg in zip(prompts, dense_gens)]
    ppl_sparse_gen = [ppl_under_dense(model, p, sg) for p, sg in zip(prompts, sparse_gens)]
    mp_d = sum(ppl_dense_gen) / len(ppl_dense_gen)
    mp_s = sum(ppl_sparse_gen) / len(ppl_sparse_gen)

    rec = {"experiment": "generation", "model": model_id, "n_prompts": len(prompts),
           "prompt_len": prompt_len, "gen_len": gen_len, "method": "local_global_topk",
           "token_agreement": round(mean_ag, 4),
           "dense_greedy_self_ag": round(greedy_self_ag, 4),
           "dense_sample_self_ag": round(sample_self_ag, 4),
           "ppl_dense_gen": round(mp_d, 2), "ppl_sparse_gen": round(mp_s, 2)}
    append_jsonl(RESULTS, rec)
    print(f"  token_agreement(dense-vs-sparse, greedy)={mean_ag:.4f}")
    print(f"  CONTROL dense_greedy_self_ag={greedy_self_ag:.4f} "
          f"dense_sample_self_ag={sample_self_ag:.4f}")
    print(f"  ppl_under_dense: dense_gen={mp_d:.2f} sparse_gen={mp_s:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-0.8b", choices=list(MODELS.keys()))
    ap.add_argument("--n-prompts", type=int, default=5)
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--gen-len", type=int, default=128)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-quantize", action="store_true")
    args = ap.parse_args()
    run(args.model, args.n_prompts, args.prompt_len, args.gen_len, args.dtype, args.seed,
        load_in_4bit=not args.no_quantize)


if __name__ == "__main__":
    main()
