#!/usr/bin/env python3
"""Tabla unificada de la ganancia del content top-k sobre el control
ESTRUCTURALMENTE igualado (local_global_randtopk: misma ventana local + mismos sinks, free keys
aleatorizadas), agregada de todas las fuentes disponibles.

Motivo: se senalo que "the claim that the advantage is stable across
scale rests on heterogeneous cells" y que abstract/introduccion usan wording mas fuerte que el
texto. En vez de suavizar la frase, aqui se rederiva el rango verdadero celda a celda y se declara
la cobertura real.

Fuentes (results/ de MLattention):
  - leaderboard.jsonl   : cos_mean de lg_topk_k32 y lg_randtopk_k32, 6 modelos x ctx {2048,4096}
  - m1_control.jsonl    : cos_mean de local_global_topk y local_global_randtopk, 4 modelos, ctx 2048
  - smollm_fidelity.jsonl : replicacion cross-family (SmolLM2-1.7B), ctx 2048

Salida por stdout en formato listo para pegar en LaTeX, mas el rango/mediana y la lista de modelos
DONDE NO HAY control estructuralmente igualado (importante para acotar la claim).

No escribe nada: solo lee.
"""
import json
import os
import statistics as st
from collections import defaultdict

RESULTS = os.path.expanduser("~/Escritorio/Code/MLattention/results")


def load(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


cells = []  # (modelo, ctx, cos_lgtopk, cos_randtopk, ganancia, fuente)

# --- leaderboard.jsonl -------------------------------------------------------
d = defaultdict(dict)
for r in load("leaderboard.jsonl"):
    if r.get("method") in ("lg_topk_k32", "lg_randtopk_k32"):
        d[(r.get("model"), r.get("context"))][r["method"]] = r.get("cos_mean")
for (model, ctx), v in sorted(d.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
    a, b = v.get("lg_topk_k32"), v.get("lg_randtopk_k32")
    if a is None or b is None:
        continue
    cells.append((model, ctx, a, b, a - b, "leaderboard"))

# --- m1_control.jsonl --------------------------------------------------------
d = defaultdict(dict)
for r in load("m1_control.jsonl"):
    if r.get("method") in ("local_global_topk", "local_global_randtopk"):
        d[(r.get("model"), r.get("context"))][r["method"]] = r.get("cos_mean")
for (model, ctx), v in sorted(d.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
    a, b = v.get("local_global_topk"), v.get("local_global_randtopk")
    if a is None or b is None:
        continue
    cells.append((model, ctx, a, b, a - b, "m1_control"))

# --- SmolLM2-1.7B: control estructuralmente igualado con datos por muestra ---------
# smollm_fidelity.jsonl solo tiene agregados (cos_mean, cos_std, n) y no lo genera ningun
# script del repo: se hizo ad hoc. Se sustituye por REV2_smollm_ci_n20.jsonl, producido por
# scripts/run_ci_repeat.py con --methods lg_topk lg_randtopk uniform_random y 20 seeds, que
# si guarda cos_list por muestra y permite el estadistico apareado.
sm = defaultdict(list)
SM_PATH = os.path.join(RESULTS, "REV2_smollm_ci_n20.jsonl")
SM_FALLBACK = os.path.join(RESULTS, "smollm_fidelity.jsonl")
ruta_sm = SM_PATH if os.path.exists(SM_PATH) else SM_FALLBACK
if os.path.exists(ruta_sm):
    for r in load(os.path.basename(ruta_sm)):
        cl = r.get("cos_list")
        m = str(r.get("method", ""))
        if cl:
            sm[m].extend(cl)
        elif "topk" in m:
            v = r.get("cos_mean")
            if v is not None:
                sm[m].append(v)
    a = sm.get("lg_topk") or []
    b = sm.get("lg_randtopk") or []
    if a and b:
        cells.append(("HuggingFaceTB/SmolLM2-1.7B", 2048, st.mean(a), st.mean(b),
                      st.mean(a) - st.mean(b), os.path.basename(ruta_sm)))
        print(f"# SmolLM2-1.7B desde {os.path.basename(ruta_sm)}: "
              f"lg_topk n={len(a)} media={st.mean(a):.4f} sd={st.stdev(a):.4f} | "
              f"lg_randtopk n={len(b)} media={st.mean(b):.4f} sd={st.stdev(b):.4f}")

# --- modelos con control solo budget-matched (sin estructuralmente igualado) --
matched = {c[0] for c in cells}
budget_only = defaultdict(lambda: defaultdict(dict))
for r in load("robustness.jsonl"):
    budget_only[r.get("model")][r.get("context")][r.get("method")] = r.get("cos_mean")

print("=== GANANCIA DEL CONTENT TOP-k SOBRE CONTROL ESTRUCTURALMENTE IGUALADO ===")
print(f"{'modelo':30s} {'ctx':>5s} {'lg_topk':>8s} {'randtopk':>9s} {'ganancia':>9s}  fuente")
gains = []
for model, ctx, a, b, g, src in cells:
    gains.append(g)
    print(f"{model:30s} {ctx:5d} {a:8.4f} {b:9.4f} {g:+9.4f}  {src}")

print()
if gains:
    print(f"celdas con control estructuralmente igualado : {len(gains)}")
    print(f"rango verdadero                             : {min(gains):+.4f} a {max(gains):+.4f}")
    print(f"mediana                                     : {st.median(gains):+.4f}")
    print(f"media                                       : {st.mean(gains):+.4f}")
    modelos = sorted({c[0] for c in cells})
    print(f"modelos cubiertos ({len(modelos)})                  : {', '.join(modelos)}")
    ctxs = sorted({c[1] for c in cells})
    print(f"contextos cubiertos                         : {ctxs}")

print()
print("=== MODELOS USADOS EN EL MANUSCRITO SIN CONTROL ESTRUCTURALMENTE IGUALADO ===")
for model, per_ctx in sorted(budget_only.items(), key=lambda kv: str(kv[0])):
    if model in matched:
        continue
    for ctx, meths in sorted(per_ctx.items()):
        tiene_sm = any("randtopk" in str(m) or m == "lgr_block" for m in meths)
        lg = meths.get("local_global_topk")
        ur = meths.get("uniform_random_m")
        nota = "solo budget-matched" if not tiene_sm else "block-level"
        if lg is not None and ur is not None:
            print(f"  {model} ctx={ctx}: lg_topk={lg:.4f} uniform_random={ur:.4f} "
                  f"gap budget-matched={lg-ur:+.4f}   [{nota}]")

print()
print("=== LATEX (fila por celda, para una tabla de la revision) ===")
print("% modelo & ctx & lg\\_topk & randtopk estructural & ganancia \\\\")
for model, ctx, a, b, g, src in cells:
    tex = model.replace("_", "\\_")
    print(f"{tex} & {ctx} & {a:.4f} & {b:.4f} & {g:+.4f} \\\\")
