#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/path/to/ugp/krishna
CODE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$CODE_DIR:$CODE_DIR/../v7legs_body:$CODE_DIR/../v7feedback${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
exec /path/to/ugp/python "$CODE_DIR/train_joint_specialists.py" \
  --root "$TASK_ROOT/V7F_4080_Run" \
  --parent "$TASK_ROOT/V7Legs_V7B_20261007/results" \
  --feedback "$TASK_ROOT/V7Feedback_20261007/results" \
  --output "$TASK_ROOT/V7JointSpecialists_20261008/results" "$@"
