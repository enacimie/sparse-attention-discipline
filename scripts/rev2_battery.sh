#!/bin/bash
# Bateria RULER unificada a n=40 por tarea, con las mismas semillas para todos los metodos.
#
# Proposito: medir la recuperacion exacta con potencia suficiente ("Statistical power of the retrieval
# results is insufficient... Increase sample sizes in key cells and report effect sizes with
# confidence intervals").
#
# Diseno UNIFICADO, que es lo que el revisor echa en falta en las celdas actuales:
#   - mismos 40 seeds (1000..1039) para TODAS las tareas, TODOS los metodos y TODOS los modelos
#   - las 13 tareas de src/ruler_tasks.py, incluido FWE/CWE que antes tenian n=10
#   - ctx 2048, y los defaults del run original (float16, load_in_4bit, topk-budget 96, lg-topk 32,
#     gen-tokens 24) para que sea comparable con ruler_battery.jsonl
#
# Los seeds 1000..1039 son deliberadamente distintos de los 42..51 historicos: asi el fichero nuevo
# no puede mezclarse por accidente con el antiguo, que tiene seeds 42/43/44 duplicados en FWE/CWE
# (dos runs volcados al mismo JSONL) y que causo el agregado erroneo de uniform_random_m (2/53 en vez
# de 1/53).
#
# NO escribe en results/ruler_battery.jsonl: la evidencia original queda intacta en ese fichero.
set -uo pipefail
export PYTHONUNBUFFERED=1
cd ~/mlattention || exit 1
PY=~/venvs/mlattention/bin/python
SEEDS=$(seq 1000 1039)

echo "=== [REV2] inicio $(date -Is) ==="
echo "seeds 1000..1039 (n=40), 13 tareas, 5 metodos, ctx 2048"

for M in qwen3.5-0.8b qwen3.5-4b-base; do
  OUT=results/REV2_battery_${M}.jsonl
  LOG=results/REV2_battery_${M}.log
  echo "--- [REV2] modelo $M -> $OUT   $(date -Is) ---"
  rm -f "$OUT"
  $PY scripts/run_ruler_battery.py --model "$M" --contexts 2048 --seeds $SEEDS --out "$OUT" > "$LOG" 2>&1
  RC=$?
  echo "--- [REV2] modelo $M exit=$RC filas=$(wc -l < "$OUT" 2>/dev/null || echo 0)   $(date -Is) ---"
done

echo "=== [REV2] fin $(date -Is) ==="
