#!/usr/bin/env bash
set -euo pipefail

ROOT=/path/to/ugp/merged_feelwm_v7
RUN_ROOT=/path/to/ugp/merged_feelwm_v7_runs/tartanground_k1_k5_leg_compare
PY=/path/to/ugp/python
cd "$ROOT"
mkdir -p "$RUN_ROOT/logs"

phase() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$1" | tee "$RUN_ROOT/phase.txt"
}

if [[ ! -f "$RUN_ROOT/prepared/preparation_metadata.json" ]]; then
  phase prepare_shared_cache
  "$PY" scripts/prepare_tartanground_indexed.py --output "$RUN_ROOT/prepared" --device cuda:0 --batch-size 16 \
    > >(tee -a "$RUN_ROOT/logs/prepare.log") 2>&1
fi

phase unit_smoke_tests
"$PY" -m unittest discover -s tests -v > "$RUN_ROOT/logs/tests.log" 2>&1

run_k1() {
  local label="$1" gpu="$2" config="$3"
  local output="$RUN_ROOT/$label/k1"
  mkdir -p "$output"
  if [[ -f "$output/checkpoints/latest.pt" ]] && grep -q 'Step cap reached at step=200' "$output/train.log" 2>/dev/null; then
    echo "K1_REUSE $label latest checkpoint is at step 200" | tee -a "$RUN_ROOT/logs/orchestrator.log"
    return 0
  fi
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    "$PY" -u train.py --config "$ROOT/$config" >> "$output/train.log" 2>&1
}

phase k1_baseline_and_per_leg
run_k1 baseline 0 configs/tartanground_baseline_k1.yaml & p0=$!
run_k1 per_leg 1 configs/tartanground_leg_k1.yaml & p1=$!
wait "$p0"
wait "$p1"

phase dev_only_lambda_calibration
"$PY" scripts/calibrate_lambdas.py \
  --config configs/tartanground_baseline_k5.yaml \
  --output "$RUN_ROOT/lambda_calibration" --set-configs \
  > "$RUN_ROOT/logs/lambda_calibration.log" 2>&1

run_k5() {
  local label="$1" gpu="$2" config="$3"
  local output="$RUN_ROOT/$label/k5"
  mkdir -p "$output"
  if [[ -f "$output/checkpoints/latest.pt" ]] && grep -q 'Step cap reached at step=100' "$output/train.log" 2>/dev/null; then
    echo "K5_REUSE $label latest checkpoint is at step 100" | tee -a "$RUN_ROOT/logs/orchestrator.log"
    return 0
  fi
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    "$PY" -u train.py --config "$ROOT/$config" >> "$output/train.log" 2>&1
}

phase k5_baseline_and_per_leg
run_k5 baseline 0 configs/tartanground_baseline_k5.yaml & p0=$!
run_k5 per_leg 1 configs/tartanground_leg_k5.yaml & p1=$!
wait "$p0"
wait "$p1"

phase strict_unseen_trajectory_evaluation
for label in baseline per_leg; do
  config_label="$label"
  [[ "$label" == "per_leg" ]] && config_label="leg"
  for horizon in 1 5; do
    config="configs/tartanground_${config_label}_k${horizon}.yaml"
    checkpoint="$RUN_ROOT/$label/k${horizon}/checkpoints/best.pt"
    "$PY" scripts/evaluate_strict.py \
      --config "$config" --checkpoint "$checkpoint" \
      --train-manifest "$RUN_ROOT/prepared/train_k${horizon}.jsonl" \
      --dev-manifest "$RUN_ROOT/prepared/dev_k${horizon}.jsonl" \
      --test-manifest "$RUN_ROOT/prepared/test_k${horizon}.jsonl" \
      --output "$RUN_ROOT/$label/k${horizon}/strict_test" --device cuda:0 \
      > "$RUN_ROOT/logs/eval_${label}_k${horizon}.log" 2>&1
  done
done

phase results_package
"$PY" scripts/package_tartanground_compare.py \
  --root "$ROOT" --run-root "$RUN_ROOT" \
  --output "$RUN_ROOT/tartanground_k1_k5_leg_compare.zip" \
  > "$RUN_ROOT/logs/package.log" 2>&1
phase complete
printf 'DONE %s\n' "$RUN_ROOT/tartanground_k1_k5_leg_compare.zip"
