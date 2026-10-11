#!/bin/bash
set -euo pipefail
source ~/activate_mlattention.sh
cd ~/mlattention
export PYTHONUNBUFFERED=1

rm -f results/needle_sparse.jsonl results/ruler_subset.jsonl results/ruler_battery.jsonl

python scripts/run_needle_sparse.py --model qwen3.5-4b-base --contexts 2048 --depths 0.1 0.5 0.9 --seeds 42 43 44 --topk-budget 96 --lg-topk 32

python scripts/run_ruler_subset.py --model qwen3.5-0.8b --contexts 2048 4096 --n-facts 4 8 --seeds 42 43 44 45
python scripts/run_ruler_subset.py --model qwen3.5-4b-base --contexts 1024 2048 --n-facts 4 8 --seeds 42 43 44 45 46
python scripts/run_ruler_subset.py --model qwen3.5-4b-base --contexts 4096 --n-facts 4 8 --seeds 42 43 44 45

python scripts/run_ruler_battery.py --model qwen3.5-0.8b --contexts 2048 --seeds 42 43 44
python scripts/run_ruler_battery.py --model qwen3.5-4b-base --contexts 2048 --seeds 42 43 44
