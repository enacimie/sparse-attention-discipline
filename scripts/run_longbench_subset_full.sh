#!/bin/bash
set -euo pipefail
source ~/activate_mlattention.sh
cd ~/mlattention
export PYTHONUNBUFFERED=1

rm -f results/longbench_subset.jsonl

python scripts/run_longbench_subset.py --model qwen3.5-0.8b --n-samples 5 --max-len 4096
python scripts/run_longbench_subset.py --model qwen3.5-4b-base --n-samples 5 --max-len 4096
