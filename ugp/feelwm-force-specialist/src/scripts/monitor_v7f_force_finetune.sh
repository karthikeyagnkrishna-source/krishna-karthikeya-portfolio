#!/usr/bin/env bash
set -u
OUT=/path/to/ugp/merged_feelwm_v7_runs/tartanground_k1_k5_leg_compare_10ep_test10/v7f_force_finetune
printf '%s\n' '=== phase ==='
cat "$OUT/phase.txt" 2>/dev/null || true
printf '%s\n' '=== processes ==='
pgrep -af 'run_v7f_force_finetune_after_compare|tartanground_leg_k5_v7f_finetune|select_force_checkpoint|evaluate_strict.py' || true
printf '%s\n' '=== latest training ==='
tail -n 3 "$OUT/train.log" 2>/dev/null || true
printf '%s\n' '=== artifacts ==='
find "$OUT" -maxdepth 2 -type f \( -name 'best_force.pt' -o -name 'latest.pt' -o -name '*RESULTS.zip' -o -name '*.sha256' \) -printf '%TY-%Tm-%Td %TH:%TM:%TS %12s %p\n' 2>/dev/null | sort
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader
