#!/bin/bash
# Reviewer round-7 runs (LLM reviews #4-#6) on ULL: needle statistical power
# (n=108 probes at the working budget) and the dense-posterior oracle top-k
# upper bound (fidelity + small-n needle).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/activate_mlattention.sh
cd ~/mlattention
rm -f results/needle_power.jsonl results/oracle_topk.jsonl
python scripts/run_needle_power.py --model qwen3.5-0.8b --ctx 2048 || echo STEP_FAIL_POWER_08B
python scripts/run_needle_power.py --model qwen3.5-4b-base --ctx 2048 --seeds 42 43 44 45 46 47 48 49 50 || echo STEP_FAIL_POWER_4B
python scripts/run_oracle_topk.py --model qwen3.5-0.8b --ctx 2048 --seeds 42 43 44 45 46 || echo STEP_FAIL_ORACLE_FID
python scripts/run_needle_power.py --model qwen3.5-0.8b --ctx 2048 --depths 0.1 0.5 0.9 --seeds 42 43 44 45 46 47 48 49 50 --methods lg_oracle lg_topk || echo STEP_FAIL_ORACLE_NEEDLE
echo ALL_DONE
