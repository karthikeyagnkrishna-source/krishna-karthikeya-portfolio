#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/path/to/ugp/krishna
RUN_ROOT="$TASK_ROOT/V7JointSpecialists_20261008"
CODE="$RUN_ROOT/code/v7_joint_specialists"
PYTHON=/path/to/ugp/python
export PYTHONUNBUFFERED=1
export HTTPS_PROXY=http://127.0.0.1:18881
export PYTHONPATH="$CODE:$CODE/../v7legs_body:$CODE/../v7feedback"
"$PYTHON" "$CODE/ops/prepare_evaluation_commands.py" --root "$TASK_ROOT/V7F_4080_Run" --output "$RUN_ROOT/results"
# Recheck process owners immediately before launch. Do not interrupt any existing job.
GPU=""
while [[ -z "$GPU" ]]; do
GPU="$($PYTHON - <<'PY'
import csv,io,subprocess
import json,os,time
from pathlib import Path
text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
busy={row[0].strip() for row in csv.reader(io.StringIO(processes)) if len(row)==2}
for row in csv.reader(io.StringIO(text)):
 index,uuid,free=[x.strip() for x in row]
 if uuid not in busy and int(free)>=12000:
  print(index);break
else:
 root=Path('/path/to/ugp/results')
 temp=root/'status.json.tmp'
 temp.write_text(json.dumps(dict(phase='waiting_for_gpu',pid=os.getppid(),updated_unix=time.time(),training_trajectories=28)))
 temp.replace(root/'status.json')
PY
)"
if [[ -z "$GPU" ]]; then sleep 60; fi
done
export CUDA_VISIBLE_DEVICES="$GPU"
exec bash "$CODE/run_training.sh"
