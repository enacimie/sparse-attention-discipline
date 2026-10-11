"""Analisis de REV2_external_patterns.jsonl: patrones publicados externos bajo presupuesto igualado.

Genera los numeros de la tabla de patrones externos (Longformer, BigBird, StreamingLLM) y sus
controles, con IC Wilson exactos sobre el brazo de aguja (n=45 por metodo y contexto).

Clave de lectura: el JSONL trae DOS presupuestos distintos y no son intercambiables.
  pr_mean            -> pair_ratio medido en el brazo de FIDELIDAD (5 chunks de wikitext)
  pair_ratio_needle  -> pair_ratio medido en el brazo de AGUJA (45 sondas)
La comparacion de retrieval (45/45 frente a 0/45) es del brazo de aguja, asi que la afirmacion de
'presupuesto igualado' debe sostenerse con pair_ratio_needle, no con pr_mean. Se imprimen los dos.

Uso:  python3 analisis_patrones_externos.py [ruta.jsonl] [--latex]
"""
import json
import sys
from collections import defaultdict

RUTA = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") \
    else "datos_rev2/REV2_external_patterns.jsonl"
LATEX = "--latex" in sys.argv

# Nombre canonico para el paper (tab:names) y si el metodo selecciona por contenido.
NOMBRE = {
    "dense": ("Dense (reference)", "ref"),
    "lg_topk_w64_g4_k32": ("lg\\_topk (this work)", "contenido"),
    "lg_randtopk_w64_g4_k32": ("lg\\_randtopk (struct.\\ control)", "no"),
    "bigbird_w64_g4_r1_b32": ("BigBird", "no"),
    "streamingllm_w92_g4": ("StreamingLLM", "no"),
    "longformer_w92_g4_d1": ("Longformer $d$=1", "no"),
    "longformer_w92_g4_d2": ("Longformer $d$=2", "no"),
    "uniform_random_k96": ("uniform\\_random (budget control)", "no"),
}
ORDEN = ["dense", "lg_topk_w64_g4_k32", "lg_randtopk_w64_g4_k32", "bigbird_w64_g4_r1_b32",
         "streamingllm_w92_g4", "longformer_w92_g4_d1", "longformer_w92_g4_d2",
         "uniform_random_k96"]


def wilson(k, n, z=1.959963985):
    """IC Wilson para una proporcion. Exacto en los extremos (0/n y n/n)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - m) / d, (c + m) / d)


rows = [json.loads(l) for l in open(RUTA)]
print(f"filas: {len(rows)}")

# Duplicados: en un JSONL construido por append, comprobar antes de agregar.
claves = [(r["context"], r["method"]) for r in rows]
dup = {c for c in claves if claves.count(c) > 1}
print("duplicados (ctx, method):", sorted(dup) if dup else "NINGUNO")

plat = {(r["gpu"], r["capability"], r["torch"], r["cuda"], r["python"]) for r in rows}
print("plataformas distintas:", len(plat))
for g, cap, t, cu, py in sorted(plat):
    print(f"  {g} | capability {cap} | torch {t} | cuda {cu} | python {py}")

por_ctx = defaultdict(dict)
for r in rows:
    por_ctx[r["context"]][r["method"]] = r

for ctx in sorted(por_ctx):
    print("\n" + "=" * 108)
    print(f"ctx {ctx}   (aguja: n={por_ctx[ctx][ORDEN[0]]['needle_total']} sondas por metodo; "
          f"fidelidad: {por_ctx[ctx][ORDEN[0]]['fidelity_chunks']} chunks)")
    print("=" * 108)
    hdr = (f"{'metodo':34s} {'cos':>7s} {'needle':>9s} {'Wilson 95%':>17s} "
           f"{'pr_needle':>10s} {'pr_fidel':>9s} {'VRAM':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for m in ORDEN:
        r = por_ctx[ctx].get(m)
        if r is None:
            print(f"{m:34s}  -- FALTA --")
            continue
        k, n = r["needle_hits"], r["needle_total"]
        lo, hi = wilson(k, n)
        def f4(v):
            return f"{v:10.4f}" if isinstance(v, (int, float)) else f"{'--':>10s}"

        def f4b(v):
            return f"{v:9.4f}" if isinstance(v, (int, float)) else f"{'--':>9s}"

        def f2(v):
            return f"{v:6.2f}" if isinstance(v, (int, float)) else f"{'--':>6s}"

        nom = NOMBRE.get(m, (m, "?"))[0].replace(chr(92) + "_", "_")
        print(f"{nom:34s} {r['cos_mean']:7.4f} {str(k)+'/'+str(n):>9s} "
              f"{'['+format(lo,'.3f')+','+format(hi,'.3f')+']':>17s} "
              f"{f4(r.get('pair_ratio_needle'))} {f4b(r.get('pr_mean'))} {f2(r.get('vram_gb'))}")

    # La afirmacion central: dispersion del presupuesto en el brazo de aguja, excluyendo dense.
    sp = [por_ctx[ctx][m]["pair_ratio_needle"] for m in ORDEN[1:]
          if m in por_ctx[ctx] and isinstance(por_ctx[ctx][m].get("pair_ratio_needle"), float)]
    if sp:
        print(f"\n  presupuesto (aguja) de los {len(sp)} metodos no-densos: "
              f"min {min(sp):.4f}  max {max(sp):.4f}  amplitud {max(sp)-min(sp):.4f}")
    else:
        print("\n  AVISO: ningun pair_ratio_needle numerico en este ctx")
    sin_contenido = [m for m in ORDEN[2:] if m in por_ctx[ctx] and NOMBRE[m][1] == "no"]
    print(f"  metodos SIN seleccion por contenido: {len(sin_contenido)} -> "
          f"aciertos {sum(por_ctx[ctx][m]['needle_hits'] for m in sin_contenido)}"
          f"/{sum(por_ctx[ctx][m]['needle_total'] for m in sin_contenido)}")
    r = por_ctx[ctx]["lg_topk_w64_g4_k32"]
    print(f"  lg_topk: {r['needle_hits']}/{r['needle_total']} con "
          f"pair_ratio {r['pair_ratio_needle']:.4f}")

if LATEX:
    print("\n\n% ---------- filas LaTeX ----------")
    for ctx in sorted(por_ctx):
        print(f"% ctx {ctx}")
        for m in ORDEN:
            r = por_ctx[ctx].get(m)
            if r is None:
                continue
            k, n = r["needle_hits"], r["needle_total"]
            lo, hi = wilson(k, n)
            nm = NOMBRE.get(m, (m, "?"))[0]
            if m == "dense":
                pr = "--"
            else:
                pr = f"{r['pair_ratio_needle']:.4f}"
            rec = (f"{nm} & {r['cos_mean']:.4f} & {k}/{n} "
                   f"& [{lo*100:.1f}--{hi*100:.1f}] & {pr} \\\\")
            print(rec)
        print("%")
