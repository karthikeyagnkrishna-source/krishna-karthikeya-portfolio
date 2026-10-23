#!/usr/bin/env bash
set -euo pipefail

ROOT=/path/to/ugp/merged_feelwm_v7_force_finetune
BASE_RUN=/path/to/ugp/merged_feelwm_v7_runs/tartanground_k1_k5_leg_compare_10ep_test10
PREP=/path/to/ugp/merged_feelwm_v7_runs/tartanground_k1_k5_leg_compare/prepared
OUT="$BASE_RUN/v7f_force_finetune"
PY=/path/to/ugp/python
WAIT_PID="${1:?comparison launcher PID is required}"
mkdir -p "$OUT"

phase() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$1" | tee "$OUT/phase.txt"
}

phase "waiting_for_comparison_pid_${WAIT_PID}"
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 60
done

phase verify_comparison_complete
grep -Fq 'Training complete.' "$BASE_RUN/baseline/k5/train.log"
grep -Fq 'Training complete.' "$BASE_RUN/per_leg/k5/train.log"
grep -Eq ' complete$' "$BASE_RUN/phase.txt"

shopt -s nullglob
candidates=(
  "$BASE_RUN/per_leg/k5/checkpoints/best.pt"
  "$BASE_RUN/per_leg/k5/checkpoints/latest.pt"
  "$BASE_RUN/per_leg/k5/checkpoints"/candidate_snapshot_*.pt
)
if (( ${#candidates[@]} < 2 )); then
  echo "Need at least best.pt and latest.pt candidates" >&2
  exit 2
fi

phase select_checkpoint_by_full_development_force_score
CUDA_VISIBLE_DEVICES=1 "$PY" "$ROOT/scripts/select_force_checkpoint.py" \
  --config "$ROOT/configs/tartanground_leg_k5_v7f_finetune.yaml" \
  --manifest "$PREP/dev_k5.jsonl" \
  --candidates "${candidates[@]}" \
  --output "$OUT/checkpoint_selection.json" \
  --selected-link "$OUT/init_selected.pt" \
  --device cuda:0 > "$OUT/checkpoint_selection.log" 2>&1

phase train_v7f_force_specialist_seven_epochs
cd "$ROOT"
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$PY" -u train.py --config configs/tartanground_leg_k5_v7f_finetune.yaml \
  >> "$OUT/train.log" 2>&1
grep -Fq 'Training complete.' "$OUT/train.log"

phase strict_unseen_full_manifest_evaluation
CUDA_VISIBLE_DEVICES=1 "$PY" scripts/evaluate_strict.py \
  --config configs/tartanground_leg_k5_v7f_finetune.yaml \
  --checkpoint "$OUT/checkpoints/best_force.pt" \
  --train-manifest "$PREP/train_k5.jsonl" \
  --dev-manifest "$PREP/dev_k5.jsonl" \
  --test-manifest "$PREP/test_k5.jsonl" \
  --output "$OUT/strict_test" --device cuda:0 \
  > "$OUT/strict_test.log" 2>&1

phase package_lightweight_results
"$PY" scripts/package_v7f_force_finetune.py \
  --project-root "$ROOT" --run-root "$OUT" \
  --output "$OUT/V7F_FORCE_FINETUNE_FULL_STRICT_RESULTS.zip" \
  > "$OUT/package.log" 2>&1
sha256sum "$OUT/V7F_FORCE_FINETUNE_FULL_STRICT_RESULTS.zip" > "$OUT/V7F_FORCE_FINETUNE_FULL_STRICT_RESULTS.zip.sha256"
phase complete
