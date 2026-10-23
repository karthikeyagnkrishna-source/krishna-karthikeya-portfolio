#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

torchrun --standalone --nproc_per_node=2 train.py --config configs/train_2x4080.yaml
