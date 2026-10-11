#!/usr/bin/env python3
"""Analisis de la bateria RULER ejecutada a n=40 por tarea bajo un diseno balanceado unico.

Aborda el requisito de potencia estadistica:
  "Statistical power of the retrieval results is insufficient. Samples are small (n=9, n=3 per
   RULER task, 10 cases in multi-fact retrieval), and even the McNemar test cannot separate the
   method from dense attention. Suggestion: Increase sample sizes in key cells and report effect
   sizes with confidence intervals."

Diseno: 13 tareas x 40 seeds (1000-1039) x 5 metodos = 2.600 generaciones por modelo, ctx 2048,
mismos seeds para todos los metodos => comparacion APAREADA por (tarea, seed).

Lee los JSONL producidos por scripts/rev2_battery.sh y emite:
  1. recall por tarea y metodo con IC Wilson
  2. recall global por metodo (n=520) con IC Wilson
  3. McNemar exacto de cada metodo contra dense (b, c, p)
  4. diferencia de proporciones apareada con IC (efecto + intervalo, lo que pide el revisor)
  5. comparativa contra los valores publicados en el manuscrito (n=53)

No escribe nada fuera de stdout. Uso:
  python3 analisis_bateria_rev2.py [fichero.jsonl ...]
"""
import json
import math
import sys
from collections import defaultdict
from itertools import combinations

Z = 1.959963984540054  # 95%


def wilson(k, n, z=Z):
    """Intervalo de Wilson para una proporcion. Mismo metodo que tab:ruler del manuscrito."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def mcnemar_exact(b, c):
    """Test exacto de McNemar (binomial dos colas sobre los pares discordantes)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)


def load(paths):
    rows = []
    for p in paths:
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def main(paths):
    rows = load(paths)
    models = sorted({r["model"] for r in rows})
    methods = ["dense", "lg_topk_k32", "lg_randtopk_k32", "streamingllm_bm", "uniform_random_m"]
    tasks = sorted({r["task"] for r in rows})

    # indice (modelo, metodo, tarea, seed) -> hit
    hit = {}
    for r in rows:
        hit[(r["model"], r["method"], r["task"], r["seed"])] = bool(r["hit"])

    for model in models:
        print("=" * 100)
        print(f"MODELO {model}")
        seeds = sorted({r["seed"] for r in rows if r["model"] == model})
        nseed = len(seeds)
        print(f"  seeds {seeds[0]}..{seeds[-1]} (n={nseed}) | tareas {len(tasks)} | "
              f"sondas por metodo {nseed * len(tasks)}")
        dups = len(rows) - len(hit)
        print(f"  filas {len([r for r in rows if r['model']==model])}, claves unicas {len(hit)}, "
              f"duplicados {dups}")
        print()

        # ---- 1. recall por tarea ----
        print("--- 1. RECALL POR TAREA (n por celda = %d) ---" % nseed)
        hdr = f"{'tarea':24s}" + "".join(f"{m.replace('_k32',''):>21s}" for m in methods)
        print(hdr)
        for t in tasks:
            line = f"{t:24s}"
            for m in methods:
                k = sum(1 for s in seeds if hit.get((model, m, t, s)))
                lo, hi = wilson(k, nseed)
                line += f"{k/nseed:.3f}[{lo:.2f},{hi:.2f}]".rjust(21)
            print(line)
        print()

        # ---- 2. recall global ----
        print(f"--- 2. RECALL GLOBAL (n = {nseed}x{len(tasks)} = {nseed*len(tasks)}) ---")
        glob = {}
        for m in methods:
            k = sum(1 for t in tasks for s in seeds if hit.get((model, m, t, s)))
            n = nseed * len(tasks)
            lo, hi = wilson(k, n)
            glob[m] = (k, n, lo, hi)
            print(f"  {m:20s} {k:4d}/{n} = {k/n:.4f}   IC Wilson 95% [{lo:.4f}, {hi:.4f}]")
        print()

        # ---- 3. McNemar contra dense, apareado por (tarea, seed) ----
        print("--- 3. McNEMAR EXACTO CONTRA dense (apareado por (tarea,seed)) ---")
        pairs = [(t, s) for t in tasks for s in seeds]
        for m in methods:
            if m == "dense":
                continue
            b = sum(1 for t, s in pairs if hit.get((model, "dense", t, s)) and not hit.get((model, m, t, s)))
            c = sum(1 for t, s in pairs if not hit.get((model, "dense", t, s)) and hit.get((model, m, t, s)))
            p = mcnemar_exact(b, c)
            concl = "NO separable de dense" if p >= 0.05 else "separable de dense"
            print(f"  {m:20s} b={b:3d} (dense solo)  c={c:3d} (metodo solo)  "
                  f"p={p:.3g}  -> {concl}")
        print()

        # ---- 4. efecto: diferencia de proporciones con IC ----
        print("--- 4. EFECTO (diferencia de proporciones vs dense) CON IC 95% ---")
        kd, nd, _, _ = glob["dense"]
        pd_ = kd / nd
        for m in methods:
            if m == "dense":
                continue
            k, n, _, _ = glob[m]
            pm = k / n
            d = pm - pd_
            se = math.sqrt(pm * (1 - pm) / n + pd_ * (1 - pd_) / n)
            print(f"  {m:20s} delta = {d:+.4f}   IC 95% [{d-Z*se:+.4f}, {d+Z*se:+.4f}]   "
                  f"{'excluye el 0' if (d-Z*se)*(d+Z*se) > 0 else 'INCLUYE el 0'}")
        print()

        # ---- 5. comparativa con lo publicado ----
        print("--- 5. COMPARATIVA CON LO PUBLICADO EN tab:ruler (n=53, 0.8B) ---")
        publicado = {"dense": (41, 53), "lg_topk_k32": (36, 53), "lg_randtopk_k32": (12, 53),
                     "streamingllm_bm": (13, 53), "uniform_random_m": (2, 53)}
        if "0.8B" in model:
            print(f"  {'metodo':20s} {'publicado n=53':>18s} {'nuevo n=520':>18s} {'delta':>10s}")
            for m in methods:
                kp, np_ = publicado[m]
                k, n, _, _ = glob[m]
                print(f"  {m:20s} {kp/np_:>18.4f} {k/n:>18.4f} {k/n-kp/np_:>+10.4f}")
            print("  NOTA: el publicado uniform_random_m=2/53 era erroneo; la medicion deduplicada")
            print("        de n=10 daba 1/53. Ver NOTAS_REVISION.md 3.1.")
        print()


if __name__ == "__main__":
    args = sys.argv[1:] or ["datos_rev2/REV2_battery_qwen3.5-0.8b.jsonl"]
    main(args)
