#!/bin/bash
# Reviewer round-2 runs on ULL: CI repeats, needle diagnosis, multi-fact recall.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/activate_mlattention.sh
cd ~/mlattention
rm -f results/ci_repeat.jsonl results/needle_diag.jsonl results/multifact_recall.jsonl
python scripts/run_ci_repeat.py --model qwen3.5-9b-base --contexts 1024 2048 --seeds 42 43 44 45 --device-map auto || echo STEP_FAIL_9B
python scripts/run_ci_repeat.py --model qwen3-0.6b --contexts 1024 2048 4096 --seeds 42 43 44 45 || echo STEP_FAIL_06B
python scripts/run_ci_repeat.py --model qwen3-1.7b --contexts 1024 2048 4096 --seeds 42 43 44 45 || echo STEP_FAIL_17B
python scripts/run_needle_diag.py --model qwen3.5-0.8b --ctx 2048 --topk 32 || echo STEP_FAIL_DIAG
python scripts/run_multifact_recall.py --models qwen3.5-0.8b qwen3.5-4b-base --contexts 2048 4096 --n-facts 1 3 || echo STEP_FAIL_MULTIFACT
echo ALL_DONE
