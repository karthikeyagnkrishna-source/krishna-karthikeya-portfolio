#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0,1
export V7_PERSIST_ROOT="$PWD/storage"
export V7_WORK_ROOT="$PWD/storage/.work"
export TMPDIR="$PWD/storage/.tmp"
export HF_HOME="$PWD/storage/.cache/huggingface"
export TORCH_HOME="$PWD/storage/.cache/torch"
export XDG_CACHE_HOME="$PWD/storage/.cache"
export MPLBACKEND=Agg
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=0
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HTTPS_PROXY=http://127.0.0.1:18765
export HTTP_PROXY=http://127.0.0.1:18765
mkdir -p "$TMPDIR" storage/logs
exec 9>storage/train.lock
flock -n 9 || { echo 'A training process already holds the project lock.'; exit 1; }
echo "$$" > storage/train.pid
printf 'V7F run started at %s\n' "$(date -Is)"
exec /path/to/ugp/python -u train.py
