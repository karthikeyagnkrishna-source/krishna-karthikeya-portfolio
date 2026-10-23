#!/usr/bin/env bash
set -euo pipefail

cd /path/to/ugp/merged_feelwm_v7

PY=/path/to/ugp/python
RUN=/path/to/ugp/merged_feelwm_v7_runs/tartanground_pilot

export TMPDIR=/tmp/s1v7
export XDG_CACHE_HOME="$RUN/cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"

CUDA_VISIBLE_DEVICES=0 "$PY" -u scripts/export_shared_p2000_k1.py --max-frames 1200
"$PY" scripts/validate_manifest.py "$RUN/prepared_k1/train_manifest.jsonl" --n 10
"$PY" scripts/validate_manifest.py "$RUN/prepared_k1/val_manifest.jsonl" --n 10

CUDA_VISIBLE_DEVICES=0,1 "$PY" -u -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  train.py --config configs/tartanground_pilot_k1.yaml
