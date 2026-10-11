"""Test apareado (McNemar) sobre el brazo de aguja: dense frente a la receta, mismas sondas.

Por que existe este script: el artefacto de patrones externos (`REV2_external_patterns.jsonl`) guarda
solo agregados (`needle_hits`, `needle_total`), asi que la comparacion dense-vs-receta ahi es NO
apareada y pierde potencia: daba Fisher p=0.058 con 38/45 frente a 44/45. Este analisis usa
`REV2_needle_paired.jsonl`, un re-run de las dos celdas con `--per-probe`, que guarda el resultado de
cada sonda. Con las mismas 45 sondas el test aparece y es significativo.

Verificaciones que hace antes de calcular nada, porque sin ellas el test no vale:
  1. que ambos metodos tengan el mismo numero de sondas
  2. que las ternas (depth, seed, needle) coincidan una a una y en el mismo orden
  3. que los aciertos recountados desde las sondas coincidan con los agregados declarados

Uso:  python3 analisis_needle_apareado.py [ruta.jsonl]
"""
import json
import sys
from math import comb

RUTA = sys.argv[1] if len(sys.argv) > 1 else "datos_rev2/REV2_needle_paired.jsonl"
A = "dense"                 # referencia
B = "lg_topk_w64_g4_k32"    # receta


def mcnemar_exacto(b, c):
    """Binomial exacta dos colas sobre los discordantes, p=0.5."""
    n = b + c
    if n == 0:
        return 1.0, n
    k = min(b, c)
    cola = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * cola), n


rows = [json.loads(l) for l in open(RUTA)]
by = {r["method"]: r for r in rows}
for m in (A, B):
    if m not in by:
        raise SystemExit(f"FALTA el metodo {m}; presentes: {sorted(by)}")
ra, rb = by[A], by[B]
pa, pb = ra["needle_per_probe"], rb["needle_per_probe"]

print("=" * 78)
print("VERIFICACIONES PREVIAS (si alguna falla, el test no es apareado)")
print("=" * 78)
ka = [(p["depth"], p["seed"], p["needle"]) for p in pa]
kb = [(p["depth"], p["seed"], p["needle"]) for p in pb]
checks = [
    ("mismo numero de sondas", len(pa) == len(pb), f"{len(pa)} vs {len(pb)}"),
    ("mismas (depth, seed, needle) en el mismo orden", ka == kb, "identicas" if ka == kb else "DIFIEREN"),
    ("agregados de dense cuadran con sus sondas",
     sum(p["hit"] for p in pa) == ra["needle_hits"] and len(pa) == ra["needle_total"],
     f"{sum(p['hit'] for p in pa)}/{len(pa)} vs declarado {ra['needle_hits']}/{ra['needle_total']}"),
    ("agregados de la receta cuadran con sus sondas",
     sum(p["hit"] for p in pb) == rb["needle_hits"] and len(pb) == rb["needle_total"],
     f"{sum(p['hit'] for p in pb)}/{len(pb)} vs declarado {rb['needle_hits']}/{rb['needle_total']}"),
    ("mismo contexto en ambas filas", ra["context"] == rb["context"], f"ctx {ra['context']} vs {rb['context']}"),
    ("misma plataforma", ra.get("gpu") == rb.get("gpu"), f"{ra.get('gpu')} vs {rb.get('gpu')}"),
]
ok = True
for nom, pasa, det in checks:
    print(f"  [{'OK ' if pasa else 'FAIL'}] {nom}: {det}")
    ok = ok and pasa
if not ok:
    raise SystemExit("Hay verificaciones fallidas: no se calcula el test.")

ha = {(p["depth"], p["seed"]): p["hit"] for p in pa}
hb = {(p["depth"], p["seed"]): p["hit"] for p in pb}
both = sum(1 for k in ha if ha[k] and hb[k])
solo_a = sum(1 for k in ha if ha[k] and not hb[k])       # b: solo dense
solo_b = sum(1 for k in ha if not ha[k] and hb[k])       # c: solo receta
ning = sum(1 for k in ha if not ha[k] and not hb[k])

print()
print("=" * 78)
print(f"TABLA 2x2 APAREADA  (ctx {ra['context']}, n={len(ha)} sondas)")
print("=" * 78)
print(f"  ambos aciertan              : {both}")
print(f"  solo dense        (b)       : {solo_a}")
print(f"  solo la receta    (c)       : {solo_b}")
print(f"  ninguno                     : {ning}")
print(f"  total                       : {both + solo_a + solo_b + ning}")

p, n = mcnemar_exacto(solo_a, solo_b)
print(f"\n  McNemar exacto: b={solo_a} c={solo_b} (discordantes {n}) -> p = {p:.4f}")
print(f"  significativo al 5%: {'SI' if p < 0.05 else 'NO'}")
print(f"  direccion de los {n} discordantes: "
      + ("todos a favor de la receta" if solo_a == 0 and solo_b > 0 else
         "todos a favor de dense" if solo_b == 0 and solo_a > 0 else "mixta"))

print("\n  sondas discordantes:")
for k in sorted(ha):
    if ha[k] != hb[k]:
        nd = next(x["needle"] for x in pa if (x["depth"], x["seed"]) == k)
        print(f"    depth={k[0]} seed={k[1]} dense={ha[k]} receta={hb[k]} needle={nd!r}")

print("\n  distribucion por profundidad de los discordantes:")
dd = {}
for k in ha:
    if ha[k] != hb[k]:
        dd[k[0]] = dd.get(k[0], 0) + 1
for k in sorted(dd):
    print(f"    depth={k}: {dd[k]}")
print("\n  artefacto:", RUTA)
print("  plataforma:", ra.get("gpu"), "| capability", ra.get("capability"),
      "| torch", ra.get("torch"), "| python", ra.get("python"))
