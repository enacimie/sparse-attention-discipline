#!/bin/bash
# Improved runs: needle at longer context (4096) + RULER FWE/CWE with n=10 seeds.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/activate_mlattention.sh
cd ~/mlattention
rm -f results/needle_power_4096.jsonl results/ruler_fwe_cwe_n10.jsonl
cp results/needle_power.jsonl results/needle_power_2048_backup.jsonl
python scripts/run_needle_power.py --model qwen3.5-0.8b --ctx 4096 \
    --seeds 42 43 44 45 46 47 48 49 50 --methods dense lg_topk lg_randtopk \
    || echo STEP_FAIL_NEEDLE4096
mv results/needle_power.jsonl results/needle_power_4096.jsonl
mv results/needle_power_2048_backup.jsonl results/needle_power.jsonl
python scripts/run_ruler_battery.py --model qwen3.5-0.8b --contexts 2048 \
    --tasks fwe cwe --seeds 42 43 44 45 46 47 48 49 50 51 \
    || echo STEP_FAIL_RULER_N10
echo ALL_DONE
