#!/bin/bash
set -euo pipefail
source ~/activate_mlattention.sh
cd ~/mlattention
export PYTHONUNBUFFERED=1

mkdir -p results/old_bug
mv -f results/generation.jsonl results/retrieval_sparse.jsonl results/old_bug/ 2>/dev/null || true
rm -f results/generation.jsonl results/retrieval_sparse.jsonl

python scripts/run_generation.py --model qwen3.5-0.8b --n-prompts 5 --prompt-len 512 --gen-len 128
python scripts/run_retrieval_sparse.py --model qwen3.5-4b-base --contexts 2048 --depths 0.1 0.5 0.9 --seeds 42 43 44 --top-k 4 --chunk-tokens 128
