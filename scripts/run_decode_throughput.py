#!/usr/bin/env python3
"""Throughput de DECODIFICACION a contexto largo: dense vs lg_topk vs controles.

Segunda campana de medicion, segunda plataforma:
  "the decode throughput comparison (46 vs 33 tok/s) at a 64-token context is misleading.
   Suggestion: ... measure decoding at long context or remove it."

La medicion anterior se tomo a ctx ~64 tokens. Con una ventana local de 64, a ese contexto la
mascara cubre todo el prefijo causal: es densa por construccion y no puede informar sobre nada.
Aqui se mide a 2048 y 4096, donde el pair ratio del receta es ~0.085 y ~0.044, y se registra el
pair ratio medido en cada celda para que quede constancia de que la mascara es de verdad dispersa.

Como se separa la decodificacion del prefill sin reimplementar el bucle:
    T(1)  = prefill + 1 paso de decodificacion
    T(65) = prefill + 65 pasos de decodificacion
    (T(65) - T(1)) / 64 = milisegundos por token en decodificacion
Es una diferencia sobre la MISMA ruta de codigo, asi que no hay que duplicar la logica de generate
ni arriesgar que el bucle manual no coincida con lo que hace el modelo en condiciones normales.

Se registra la plataforma en cada fila: el artefacto original de efficiency no lo hacia, y ese es
justo uno de los puntos del revisor.

Uso:
  python scripts/run_decode_throughput.py --model qwen3.5-0.8b --contexts 2048 4096 \
      --reps 3 --gen-tokens 64 --out results/REV2_decode_throughput.jsonl
"""
import argparse
import json
import os
import platform
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from attention_interceptor import (PAIR_COUNTER, apply_method,  # noqa: E402
                                   reset_pair_counter, restore)
from attention_methods import AttentionMethod  # noqa: E402
from model_loader import MODELS, inspect_model, load_model  # noqa: E402
from utils import set_seed  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "decode_throughput.jsonl")


def build_methods(topk_budget=96, lg_topk=32):
    """Los mismos metodos que la bateria RULER, para que la comparacion sea homogenea."""
    return [
        ("dense", None),
        (f"lg_topk_k{lg_topk}", AttentionMethod(name="local_global_topk", window=64,
                                                n_global=4, topk=lg_topk)),
        (f"lg_randtopk_k{lg_topk}", AttentionMethod(name="local_global_randtopk", window=64,
                                                    n_global=4, topk=lg_topk)),
        ("streamingllm_bm", AttentionMethod(name="local_global", window=max(topk_budget - 4, 4),
                                            n_global=4)),
    ]


def build_prompt(tok, ctx, device, split="test"):
    """Un prompt held-out de exactamente `ctx` tokens, de wikitext-103 raw."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
    text = "\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    if ids.numel() < ctx:
        raise RuntimeError(f"wikitext demasiado corto: {ids.numel()} < {ctx}")
    return ids[:ctx].unsqueeze(0).to(device)


@torch.no_grad()
def timed_generate(model, prompt, max_new):
    t0 = time.perf_counter()
    model.generate(prompt, max_new_tokens=max_new, do_sample=False,
                   pad_token_id=model.config.eos_token_id)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def platform_info():
    p = torch.cuda.get_device_properties(0)
    return {"gpu": p.name, "capability": f"{p.major}.{p.minor}",
            "vram_gb": round(p.total_memory / 1e9, 2), "torch": torch.__version__,
            "cuda": torch.version.cuda, "python": platform.python_version()}


def measure(model, prompt, gen_tokens, warmup=1):
    """Devuelve (ms_por_token_en_decodificacion, ms_totales_T1, ms_totales_T65, pair_ratio)."""
    for _ in range(warmup):
        model.generate(prompt, max_new_tokens=4, do_sample=False,
                       pad_token_id=model.config.eos_token_id)
    torch.cuda.synchronize()

    reset_pair_counter()
    t1 = timed_generate(model, prompt, 1)
    reset_pair_counter()
    t_full = timed_generate(model, prompt, gen_tokens + 1)
    computed, full = PAIR_COUNTER["computed"], PAIR_COUNTER["full"]
    pr = (computed / full) if full else None

    ms_per_tok = (t_full - t1) / gen_tokens
    return ms_per_tok, t1, t_full, pr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-0.8b", choices=list(MODELS.keys()))
    ap.add_argument("--contexts", type=int, nargs="+", default=[2048, 4096])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--topk-budget", type=int, default=96)
    ap.add_argument("--lg-topk", type=int, default=32)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--no-quantize", action="store_true")
    ap.add_argument("--device-map", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = args.out or RESULTS
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    info = platform_info()
    print("plataforma:", json.dumps(info, ensure_ascii=False), flush=True)

    model_id = MODELS[args.model]
    minfo = inspect_model(model_id)
    model, tok = load_model(model_id, dtype=args.dtype,
                            load_in_4bit=not args.no_quantize, device_map=args.device_map)
    model.eval()
    device = next(model.parameters()).device
    full_layers = minfo.full_attention_layers
    print(f"[load] {model_id} full_attn={len(full_layers)} de {minfo.num_hidden_layers} capas, "
          f"heads={minfo.num_attention_heads}/{minfo.num_key_value_heads} "
          f"head_dim={minfo.head_dim}", flush=True)

    filas = []
    for ctx in args.contexts:
        try:
            prompt = build_prompt(tok, ctx, device)
        except Exception as e:
            print(f"  ctx={ctx}: no se pudo construir el prompt ({e})", flush=True)
            continue
        for name, method in build_methods(args.topk_budget, args.lg_topk):
            set_seed(42)
            originals = apply_method(model, method, full_layers) if method else {}
            if method is not None and method.name in ("uniform_random", "local_global_randtopk"):
                method.seed = 42
            rec = dict(info)
            rec.update({"experiment": "decode_throughput", "model": model_id, "context": ctx,
                        "method": name, "gen_tokens": args.gen_tokens, "reps": args.reps,
                        "window": 64 if method is None or "local_global" in str(
                            getattr(method, "name", "")) else None})
            try:
                ms, t1s, tfs, prs = [], [], [], []
                for _ in range(args.reps):
                    a, b, c, d = measure(model, prompt, args.gen_tokens)
                    ms.append(a); t1s.append(b); tfs.append(c)
                    if d is not None:
                        prs.append(d)
                import statistics as st
                rec["decode_ms_per_tok"] = round(st.mean(ms), 3)
                rec["decode_ms_std"] = round(st.stdev(ms), 3) if len(ms) > 1 else 0.0
                rec["decode_tok_s"] = round(1000.0 / st.mean(ms), 2)
                rec["t_gen1_ms"] = round(st.mean(t1s), 1)
                rec["t_gen_full_ms"] = round(st.mean(tfs), 1)
                rec["pair_ratio"] = round(st.mean(prs), 5) if prs else None
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {str(e)[:160]}"
            finally:
                if method:
                    restore(model, originals, full_layers)
            filas.append(rec)
            with open(out_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  ctx={ctx:5d} {name:18s} "
                  f"decode={rec.get('decode_tok_s','-'):>7} tok/s  "
                  f"p_ratio={rec.get('pair_ratio','-')}  {rec.get('error','')}", flush=True)
            torch.cuda.empty_cache()

    print(f"\nDONE -> {out_path} ({len(filas)} celdas)", flush=True)


if __name__ == "__main__":
    main()
