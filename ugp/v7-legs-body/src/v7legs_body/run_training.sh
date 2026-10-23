#!/usr/bin/env bash
set -euo pipefail
RUN=/path/to/ugp/V7Legs_V7B_20261007
FOUNDATION=/path/to/ugp/V7F_4080_Run
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$RUN/deps${PYTHONPATH:+:$PYTHONPATH}"
cd "$RUN/code"
exec /path/to/ugp/python -u train.py --root "$FOUNDATION" --output "$RUN/results"
