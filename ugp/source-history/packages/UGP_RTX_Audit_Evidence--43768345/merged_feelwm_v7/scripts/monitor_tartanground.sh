#!/usr/bin/env bash
RUN=/path/to/ugp/merged_feelwm_v7_runs/tartanground_pilot
clear
date -Is
echo "--- process ---"
if [[ -f "$RUN/k1_pipeline.pid" ]]; then
  pid=$(cat "$RUN/k1_pipeline.pid")
  if kill -0 "$pid" 2>/dev/null; then echo "RUNNING pid=$pid"; else echo "STOPPED pid=$pid"; fi
else
  echo "PID file not created"
fi
echo "--- GPUs ---"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
  --format=csv,noheader
echo "--- latest pipeline log ---"
tail -n 25 "$RUN/logs/k1_pipeline.log" 2>/dev/null || true
echo "--- latest metrics ---"
tail -n 10 "$RUN/k1_train/metrics.jsonl" 2>/dev/null || true
echo "--- disk ---"
du -sh "$RUN" 2>/dev/null || true
