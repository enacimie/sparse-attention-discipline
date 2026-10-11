#!/bin/bash
set -euo pipefail
source ~/activate_mlattention.sh
cd ~/mlattention
export PYTHONUNBUFFERED=1

python scripts/run_m5_budget_curve.py --model qwen3.5-0.8b --ctx 2048 --methods lg_randtopk

python scripts/run_ruler_subset.py --model qwen3.5-0.8b --contexts 2048 4096 --n-facts 4 8 --seeds 42 43 44 45 --methods lg_randtopk_k32
python scripts/run_ruler_subset.py --model qwen3.5-4b-base --contexts 1024 2048 --n-facts 4 8 --seeds 42 43 44 45 46 --methods lg_randtopk_k32
python scripts/run_ruler_subset.py --model qwen3.5-4b-base --contexts 4096 --n-facts 4 8 --seeds 42 43 44 45 --methods lg_randtopk_k32

python scripts/run_ruler_battery.py --model qwen3.5-0.8b --contexts 2048 --seeds 42 43 44 --methods lg_randtopk_k32
python scripts/run_ruler_battery.py --model qwen3.5-4b-base --contexts 2048 --seeds 42 43 44 --methods lg_randtopk_k32
