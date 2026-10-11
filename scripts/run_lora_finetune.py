"""QLoRA adaptation of a hybrid model to the sparse attention method (step 2 of the LoRA study).

Question: does a cheap LoRA fine-tune UNDER the sparse attention recover the PPL gap that the
training-free sparsity introduces?

Measured on wikitext-2 (teacher-forced PPL):
  dense_base          dense PPL of the base model (reference)
  sparse_base         local_global_topk PPL before any training (the gap)
  sparse_lora         local_global_topk PPL after QLoRA adaptation (should approach dense_base)
  dense_lora          dense PPL of the LoRA model (checks LoRA did not hurt the dense model)

Training applies local_global_topk to the full_attention layers at every step, so the adapters
learn to make the model work with the sparse pattern. Only q/k/v/o_proj of the full-attn layers
get LoRA; the linear/conv layers stay frozen.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from attention_interceptor import apply_method, restore  # noqa: E402
from attention_methods import AttentionMethod  # noqa: E402
from model_loader import MODELS, inspect_model, load_model  # noqa: E402
from utils import append_jsonl, set_seed  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "lora_finetune.jsonl")


def get_chunks(tok, seq_len, n, split):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    return [ids[i:i + seq_len] for i in range(0, n * seq_len, seq_len)
            if i + seq_len <= ids.numel()]


@torch.no_grad()
def perplexity(model, chunks, device):
    nll, ntok = 0.0, 0
    for c in chunks:
        ids = c.unsqueeze(0).to(device)
        out = model(ids, labels=ids)
        n = ids.shape[1] - 1
        nll += out.loss.item() * n
        ntok += n
    return float(np.exp(nll / ntok))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-4b-base", choices=list(MODELS.keys()))
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--eval-docs", type=int, default=8)
    ap.add_argument("--train-steps", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--train-mode", default="sparse", choices=["sparse", "dense"],
                    help="attention used DURING training. 'dense' is the control: it shows "
                         "whether any LoRA (domain adaptation) shrinks the sparse penalty, or "
                         "only a LoRA trained under the sparse pattern does.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)

    model_id = MODELS[args.model]
    info = inspect_model(model_id)
    full_layers = info.full_attention_layers
    model, tok = load_model(model_id, dtype="float16", load_in_4bit=True, device_map="cuda")
    device = next(model.parameters()).device
    method = AttentionMethod(name="local_global_topk", window=64, n_global=4, topk=32)

    eval_chunks = get_chunks(tok, args.seq_len, args.eval_docs, "test")
    print(f"[data] {len(eval_chunks)} eval chunks of {args.seq_len}", flush=True)

    dense_base = perplexity(model, eval_chunks, device)
    print(f"dense_base   PPL = {dense_base:.3f}", flush=True)

    orig = apply_method(model, method, full_layers)
    sparse_base = perplexity(model, eval_chunks, device)
    restore(model, orig, full_layers)
    print(f"sparse_base  PPL = {sparse_base:.3f}  (gap {sparse_base-dense_base:+.3f})", flush=True)

    # ---- QLoRA setup ----
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=2 * args.lora_r,
                     lora_dropout=0.05, bias="none",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()

    # ---- train UNDER the sparse attention ----
    train_chunks = get_chunks(tok, args.seq_len, args.train_steps, "train")
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model.train()
    orig = apply_method(model, method, full_layers) if args.train_mode == "sparse" else None
    print(f"[train] attention during training = {args.train_mode}", flush=True)
    losses = []
    for step, c in enumerate(train_chunks):
        ids = c.unsqueeze(0).to(device)
        out = model(ids, labels=ids)
        out.loss.backward()
        opt.step(); opt.zero_grad()
        losses.append(out.loss.item())
        if step % 20 == 0:
            print(f"  step {step:4d} loss={out.loss.item():.4f}", flush=True)
    if orig is not None:
        restore(model, orig, full_layers)
    print(f"[train] mean loss last 20 = {np.mean(losses[-20:]):.4f}", flush=True)

    # ---- evaluate the adapted model ----
    model.eval()
    orig = apply_method(model, method, full_layers)
    sparse_lora = perplexity(model, eval_chunks, device)
    restore(model, orig, full_layers)
    dense_lora = perplexity(model, eval_chunks, device)

    print(f"sparse_lora  PPL = {sparse_lora:.3f}  "
          f"(recovered {100*(sparse_base-sparse_lora)/max(1e-9,(sparse_base-dense_base)):.0f}% of gap)",
          flush=True)
    print(f"dense_lora   PPL = {dense_lora:.3f}", flush=True)
    print(f"SUMMARY dense_base={dense_base:.3f} sparse_base={sparse_base:.3f} "
          f"sparse_lora={sparse_lora:.3f} dense_lora={dense_lora:.3f}", flush=True)
    append_jsonl(RESULTS, {"experiment": "lora_finetune", "model": model_id,
                           "seq_len": args.seq_len, "train_mode": args.train_mode,
                           "seed": args.seed, "train_steps": args.train_steps,
                           "lora_r": args.lora_r,
                           "dense_base": round(dense_base, 3),
                           "sparse_base": round(sparse_base, 3),
                           "sparse_lora": round(sparse_lora, 3),
                           "dense_lora": round(dense_lora, 3),
                           "sparse_penalty_before": round(sparse_base - dense_base, 3),
                           "sparse_penalty_after": round(sparse_lora - dense_lora, 3)})


if __name__ == "__main__":
    main()
