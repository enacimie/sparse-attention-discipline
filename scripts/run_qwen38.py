"""Qwen3.8-27B en 4-bit: fidelidad + VRAM + retrieval. Carga desde ruta local."""
import argparse, json, os, sys, time, torch

sys.path.insert(0, "/root/mlattention/src")
from attention_interceptor import apply_method, reset_pair_counter, restore, PAIR_COUNTER
from attention_methods import AttentionMethod
from model_loader import inspect_model, load_model
from utils import fidelity_metrics, set_seed, append_jsonl

MODEL_PATH = "/root/qwen38"
OUT = "/root/mlattention/results/qwen38_run.jsonl"

def main():
    ctx = int(os.environ.get("CTX", "2048"))
    torch.cuda.reset_peak_memory_stats()
    set_seed(42)

    print(f"[load] {MODEL_PATH} (4-bit) ...", flush=True)
    model, tok = load_model(MODEL_PATH, load_in_4bit=True)
    info = inspect_model(MODEL_PATH)
    fl = info.full_attention_layers
    dev = next(model.parameters()).device
    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"[arch] layers={info.num_hidden_layers} full_attn={fl} q={info.num_attention_heads} "
          f"kv={info.num_key_value_heads} head_dim={info.head_dim} hidden={info.hidden_size}", flush=True)
    print(f"[vram] post-load Q4: {vram:.2f} GB", flush=True)
    append_jsonl(OUT, {"metric": "vram_load_q4_gb", "value": round(vram, 3),
                       "n_layers": info.num_hidden_layers, "n_full_attn": info.n_full_attention})

    # input de texto real (wikitext si está, si no sintético)
    text = ("The history of machine learning traces back to early statistical methods. "
            "A language model predicts the next token given previous tokens. " * 40).strip()
    ids = tok(text, return_tensors="pt").input_ids[0][:ctx]
    input_ids = ids.unsqueeze(0).to(dev)
    print(f"[ctx] {input_ids.shape[1]} tokens", flush=True)

    @torch.no_grad()
    def get_logits(m, inp):
        return m(inp).logits[0].cpu().float()

    methods = [
        ("dense", None),
        ("lg_topk", AttentionMethod(name="local_global_topk", window=64, n_global=4, topk=32)),
        ("lg_perhead", AttentionMethod(name="local_global_perhead", window=64, n_global=4, topk=32)),
        ("lg_randtopk", AttentionMethod(name="local_global_randtopk", window=64, n_global=4, topk=32, seed=42)),
        ("uniform_random", AttentionMethod(name="uniform_random", topk=32, seed=42)),
        ("streamingllm", AttentionMethod(name="local_global", window=89, n_global=4)),
    ]

    ref = get_logits(model, input_ids)
    for name, m in methods:
        reset_pair_counter()
        if m is None:
            lg, pr = ref, 1.0
        else:
            orig = apply_method(model, m, fl)
            lg = get_logits(model, input_ids)
            restore(model, orig, fl)
            pr = PAIR_COUNTER["computed"] / PAIR_COUNTER["full"]
        fm = fidelity_metrics(lg, ref)
        print(f"[fid] {name:14s} cos={fm['logit_cosine']:.4f} kl={fm['kl_div']:.5f} "
              f"mse={fm['logit_mse']:.6f} pair_ratio={pr:.4f}", flush=True)
        append_jsonl(OUT, {"metric": "fidelity", "method": name, "ctx": ctx,
                           "cos": round(fm["logit_cosine"], 5), "kl": round(fm["kl_div"], 6),
                           "mse": round(fm["logit_mse"], 6), "pair_ratio": round(pr, 5)})

    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[vram] peak: {peak:.2f} GB", flush=True)
    append_jsonl(OUT, {"metric": "vram_peak_gb", "value": round(peak, 3)})
    print(f"[done] -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
