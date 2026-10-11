#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/activate_mlattention.sh
cd ~/mlattention
python scripts/run_ci_repeat.py --model qwen3.5-9b-base --contexts 1024 2048 --seeds 42 43 44 45 --cpu-offload || echo STEP_FAIL_9B
python scripts/run_ci_repeat.py --model qwen3-0.6b --contexts 1024 2048 4096 --seeds 42 43 44 45 --no-quantize || echo STEP_FAIL_06B
python scripts/run_ci_repeat.py --model qwen3-1.7b --contexts 1024 2048 4096 --seeds 42 43 44 45 --no-quantize || echo STEP_FAIL_17B
python scripts/run_needle_diag.py --model qwen3.5-0.8b --ctx 2048 --topk 32 || echo STEP_FAIL_DIAG
echo ALL_DONE2
