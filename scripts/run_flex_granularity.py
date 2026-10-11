#!/usr/bin/env python3
"""Benchmark de granularidad FlexAttention vs SDPA, con plataforma registrada.

Segunda campana de medicion, segunda plataforma:
  "The speed-retrieval incompatibility depends on one kernel and one hardware platform. The result
   applies to FlexAttention with blocks >=128 on an A100."

Este script responde a las dos mitades del comentario.

(1) SEGUNDA PLATAFORMA. Reproduce en la GPU local la misma rejilla que se midio en el A100
    (head_dim 128 y 256, seq 1024/2048/4096/8192, block 128, patron local+sinks), registrando
    device, capability, torch, CUDA y triton en cada fila. Sin eso la comparacion no es defendible.

(2) EL GAP DE GRANULARIDAD, MEDIDO. El argumento del paper es que la seleccion que preserva
    retrieval es por clave individual, y que un kernel block-sparse solo acelera si los bloques
    estan alineados. Aqui se contrastan dos patrones con el MISMO presupuesto logico de pares:

      A `block_aligned`  : causal + ventana local de 64 + 4 sinks. Contiguo, asi que deja bloques
                           enteros fuera y el kernel los salta.
      B `perkey_scatter` : causal + ventana local de 64 + 4 sinks + 32 claves por query repartidas
                           por todo el contexto. Es el modelo de lo que hace un top-k por contenido:
                           las claves que puntuan alto estan en posiciones arbitrarias. Logicamente
                           cuesta casi lo mismo que A, pero toca practicamente todos los bloques, asi
                           que el kernel no puede saltarse nada.

    La diferencia de speedup entre A y B a igual presupuesto ES el gap, y `active_block_frac` es el
    mecanismo que lo explica.

Correcciones frente a scripts/run_flex_bench.py, que se deja intacto porque es el artefacto original:
  - alli `block_size` se registra pero nunca se pasa a create_block_mask, luego no hay barrido real;
  - alli `active_block_frac` usa `(kv_num_blocks >= 0).sum() / numel()`, que es siempre 1.0 y no
    significa nada: kv_num_blocks cuenta cuantos bloques KV atiende cada bloque de queries, y por
    causalidad todos atienden al menos a su diagonal. Contar `> 0` da el mismo 1.0. Lo correcto es
    SUMAR los bloques activos y dividir por el total de bloques causales, que es lo que se hace aqui,
    anadiendo ademas la fraccion de bloques FULL (los que el kernel recorre sin evaluar mask_mod);
  - aqui se registra la plataforma.

El patron disperso se genera con una funcion determinista de (qi, kvi) porque mask_mod tiene que ser
pura para que FlexAttention pueda trazarla. Se usa un hashing modular, que reparte las 32 claves de
cada query de forma estable y sin depender de datos. Es un proxy de la seleccion por contenido, no la
seleccion real: lo que importa aqui es la dispersion espacial, no que las claves sean las correctas.

Uso:
    python scripts/run_flex_granularity.py [--out results/flex_granularity.jsonl] [--seqs 2048 4096]
"""
import argparse
import json
import os
import platform
import sys
import time

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

HEADS = 16
WINDOW = 64
N_GLOBAL = 4
TOPK_KEYS = 32


def bench(fn, iters=30, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def make_mask_mod(pattern, seq, window=WINDOW, n_global=N_GLOBAL, topk=TOPK_KEYS):
    """Devuelve un mask_mod puro y la proporcion de pares causales que deja pasar."""
    if pattern == "block_aligned":
        def mask_mod(b, h, qi, kvi):
            return (qi >= kvi) & (((qi - kvi) < window) | (kvi < n_global))
    elif pattern == "perkey_scatter":
        # hashing modular determinista: para cada query, las kvi cuyo hash cae por debajo del umbral
        # estan seleccionadas. Reparte las claves por todo el contexto. Es un proxy de la seleccion
        # por contenido: lo que importa aqui es la dispersion espacial, no que las claves sean las
        # correctas.
        #
        # El umbral se calibra para que el numero ESPERADO de claves dispersas por query sea `topk`,
        # y por tanto para que el presupuesto logico de pares iguale al del patron alineado. Con un
        # umbral fijo el numero de claves creceria con seq y la comparacion dejaria de estar igualada.
        mod = 4096
        umbral = max(1, round(mod * topk / max(seq, 1)))

        def mask_mod(b, h, qi, kvi):
            base = (qi >= kvi) & (((qi - kvi) < window) | (kvi < n_global))
            hsh = (kvi * 7919 + qi * 104729) % mod
            return base | ((qi >= kvi) & (hsh < umbral))

        # se exponen para dejarlos registrados en la salida: sin ellos el patron no es reproducible
        mask_mod.scatter_umbral = umbral
        mask_mod.scatter_mod = mod
    else:
        raise ValueError(f"patron desconocido: {pattern}")
    return mask_mod


def pair_ratio_from_mask(mask_mod, seq, device):
    """Proporcion de pares causales que el patron deja pasar, contada de forma exacta."""
    qi = torch.arange(seq, device=device).unsqueeze(1)
    kvi = torch.arange(seq, device=device).unsqueeze(0)
    m = mask_mod(None, None, qi, kvi)
    causal = qi >= kvi
    return float((m & causal).sum()) / float(causal.sum())


def platform_info():
    p = torch.cuda.get_device_properties(0)
    try:
        import triton
        triton_v = triton.__version__
    except Exception:
        triton_v = "n/d"
    return {
        "gpu": p.name,
        "capability": f"{p.major}.{p.minor}",
        "vram_gb": round(p.total_memory / 1e9, 2),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton_v,
        "python": platform.python_version(),
    }


def run_one(pattern, seq, head_dim, block_size, dtype, info):
    device = "cuda"
    torch.manual_seed(0)
    q = torch.randn(1, HEADS, seq, head_dim, device=device, dtype=dtype)
    k = torch.randn(1, HEADS, seq, head_dim, device=device, dtype=dtype)
    v = torch.randn(1, HEADS, seq, head_dim, device=device, dtype=dtype)

    mask_mod = make_mask_mod(pattern, seq)

    def dense():
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    out = dict(info)
    out.update({"pattern": pattern, "seq": seq, "head_dim": head_dim, "heads": HEADS,
                "window": WINDOW, "n_global": N_GLOBAL, "topk_keys": TOPK_KEYS,
                "block_size": block_size})
    try:
        out["pair_ratio"] = round(pair_ratio_from_mask(mask_mod, seq, device), 5)
    except Exception as e:
        out["pair_ratio"] = None
        out["pair_ratio_err"] = str(e)[:120]
    if hasattr(mask_mod, "scatter_umbral"):
        out["scatter_umbral"] = mask_mod.scatter_umbral
        out["scatter_mod"] = mask_mod.scatter_mod

    try:
        block_mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=seq, KV_LEN=seq,
                                       device=device, BLOCK_SIZE=block_size, _compile=True)
        flex_c = torch.compile(flex_attention, dynamic=False)

        def sparse():
            return flex_c(q, k, v, block_mask=block_mask)

        # Ocupacion real de bloques. OJO: contar `(kv_num_blocks > 0).sum()` da siempre 1.0 y no
        # significa nada, porque kv_num_blocks cuenta cuantos bloques KV atiende CADA bloque de
        # queries y por causalidad todos atienden al menos a su diagonal. Lo que importa es la SUMA
        # de bloques activos sobre el total de bloques causales posibles.
        #
        # Se registran ademas los bloques FULL, que son los que el kernel recorre por la ruta rapida
        # sin evaluar mask_mod. Un patron disperso por clave produce bloques que no estan ni llenos
        # ni vacios, y ahi es donde se pierde la aceleracion: esa es la diferencia entre full y active.
        kv = block_mask.kv_num_blocks
        nq = (seq + block_size - 1) // block_size
        causal_total = nq * (nq + 1) // 2
        activos = int(kv.sum())
        out["kv_blocks_active"] = activos
        out["kv_blocks_causal_total"] = causal_total
        out["active_block_frac"] = round(activos / max(causal_total, 1), 5)
        try:
            llenos = int(block_mask.full_kv_num_blocks.sum())
            out["kv_blocks_full"] = llenos
            out["full_block_frac"] = round(llenos / max(causal_total, 1), 5)
        except Exception as e:
            out["full_block_frac"] = None
            out["full_block_err"] = str(e)[:100]
        try:
            # sparsity() de torch usa la rejilla cuadrada completa, no el triangulo causal:
            # 100 * (1 - activos / nq^2). Se registra tal cual para no mezclar denominadores.
            out["sparsity_torch_pct"] = round(float(block_mask.sparsity()), 3)
        except Exception:
            out["sparsity_torch_pct"] = None

        out["dense_ms"] = round(bench(dense), 3)
        out["flex_ms"] = round(bench(sparse), 3)
        out["speedup"] = round(out["dense_ms"] / out["flex_ms"], 3)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:180]}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/flex_granularity.jsonl")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--solo-rejilla", action="store_true",
                    help="solo la rejilla comparable con el A100, sin barridos extra")
    args = ap.parse_args()

    torch.set_float32_matmul_precision("high")
    info = platform_info()
    print("plataforma:", json.dumps(info, ensure_ascii=False), flush=True)
    dtype = getattr(torch, args.dtype)

    configs = []
    # (1) rejilla comparable con la corrida original en A100
    for head_dim in (128, 256):
        for seq in (1024, 2048, 4096, 8192):
            configs.append(("block_aligned", seq, head_dim, 128))
    if not args.solo_rejilla:
        # (2) barrido de granularidad sobre el patron alineado
        for bs in (32, 64, 256):
            configs.append(("block_aligned", 4096, 128, bs))
        # (3) el gap: mismo presupuesto logico, patron disperso por clave
        for seq in (2048, 4096):
            configs.append(("perkey_scatter", seq, 128, 128))
            configs.append(("perkey_scatter", seq, 128, 64))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    resultados = []
    for pattern, seq, hd, bs in configs:
        t0 = time.time()
        r = run_one(pattern, seq, hd, bs, dtype, info)
        r["wall_s"] = round(time.time() - t0, 1)
        resultados.append(r)
        print(json.dumps({k: r.get(k) for k in ("pattern", "seq", "head_dim", "block_size",
                                                "pair_ratio", "active_block_frac",
                                                "full_block_frac", "dense_ms",
                                                "flex_ms", "speedup", "error", "wall_s")},
                         ensure_ascii=False), flush=True)
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        for r in resultados:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nDONE -> {args.out} ({len(resultados)} configs)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
