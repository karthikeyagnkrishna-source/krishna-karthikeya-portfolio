#!/usr/bin/env bash
set -euo pipefail
RUN=${RUN:-/path/to/ugp/V7Feedback_20261007}
PARENT=${PARENT:-/path/to/ugp/V7Legs_V7B_20261007}
FOUNDATION=${FOUNDATION:-/path/to/ugp/V7F_4080_Run}
export CUDA_VISIBLE_DEVICES=${GPU:-1} OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PARENT/deps${PYTHONPATH:+:$PYTHONPATH}"
cd "$RUN/code/v7feedback"
exec /path/to/ugp/python -u train_feedback.py --root "$FOUNDATION" --parent "$PARENT/results" --output "$RUN/results" "$@"
