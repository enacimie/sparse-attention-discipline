#!/bin/bash
# Overnight experiment chain for the peer-review hardening pass.
# Each step runs as an isolated python process with its own log and a hard timeout, so a single
# failure (crash/hang/OOM) never stops the rest of the chain. Results append to results/*.jsonl.
source ~/activate_mlattention.sh 2>/dev/null
cd ~/mlattention || exit 1
LOGDIR=/tmp/overnight
mkdir -p "$LOGDIR"

echo "[$(date)] === START overnight chain ===" > "$LOGDIR/master.log"

run_step() {
  local name="$1"; local tmo="$2"; shift 2
  echo "[$(date)] START $name" >> "$LOGDIR/master.log"
  timeout "$tmo" "$@" > "$LOGDIR/$name.log" 2>&1
  local code=$?
  echo "[$(date)] END   $name (exit=$code)" >> "$LOGDIR/master.log"
  sleep 5
}

# 1. Downstream RULER-style multi-fact retrieval @2048 (highest new value)
run_step ruler_2048 3000 python -u scripts/run_ruler_subset.py --model qwen3.5-4b-base \
    --contexts 2048 --n-facts 4 8 --seeds 42 43 44 45 46

# 2. Fidelity beyond WikiText: code + math domains
run_step domains_2048 3000 python -u scripts/run_domains.py --model qwen3.5-4b-base \
    --ctx 2048 --n-samples 5 --domains code math

# 3. Full fidelity panel: KL / MSE / logit-cos / hidden-cos
run_step fid_metrics_2048 2400 python -u scripts/run_fidelity_metrics.py --model qwen3.5-4b-base \
    --ctx 2048 --n-samples 5

# 4. RULER subset @1024 (shorter-context downstream)
run_step ruler_1024 3000 python -u scripts/run_ruler_subset.py --model qwen3.5-4b-base \
    --contexts 1024 --n-facts 4 8 --seeds 42 43 44 45 46

# 5-6. Seed variance of the central claim @2048 (n=6 each, different seeds)
run_step robust_seed43 2400 python -u scripts/run_robustness.py --model qwen3.5-4b-base \
    --ctx 2048 --n-samples 6 --seed 43
run_step robust_seed44 2400 python -u scripts/run_robustness.py --model qwen3.5-4b-base \
    --ctx 2048 --n-samples 6 --seed 44

# 7. Scale CI @1024
run_step robust_1024 2400 python -u scripts/run_robustness.py --model qwen3.5-4b-base \
    --ctx 1024 --n-samples 6 --seed 42

# 8. Fidelity panel @1024
run_step fid_metrics_1024 2400 python -u scripts/run_fidelity_metrics.py --model qwen3.5-4b-base \
    --ctx 1024 --n-samples 5

echo "[$(date)] === DONE overnight chain ===" >> "$LOGDIR/master.log"
