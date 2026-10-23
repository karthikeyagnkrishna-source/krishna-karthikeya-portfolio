#!/usr/bin/env bash
set -u
RUN_ROOT=/path/to/ugp/merged_feelwm_v7_runs/tartanground_k1_k5_leg_compare
while true; do
  clear
  printf 'PHASE: '
  cat "$RUN_ROOT/phase.txt" 2>/dev/null || true
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader
  for file in \
    "$RUN_ROOT/baseline/k1/train.log" \
    "$RUN_ROOT/per_leg/k1/train.log" \
    "$RUN_ROOT/baseline/k5/train.log" \
    "$RUN_ROOT/per_leg/k5/train.log" \
    "$RUN_ROOT/logs/lambda_calibration.log"; do
    if [[ -f "$file" ]]; then
      printf '\n--- %s ---\n' "$file"
      tail -n 3 "$file"
    fi
  done
  sleep 2
done
