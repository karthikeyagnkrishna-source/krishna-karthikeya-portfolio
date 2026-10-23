# V7F Force-Specialist Continuation Plan

**Goal:** Preserve the active ten-epoch five-stream run, then run a resumable five-to-seven epoch V7F-only continuation selected by development force accuracy and evaluated on every eligible strict unseen-trajectory anchor.

## Constraints

- Never interrupt or duplicate the active `run_tartanground_compare.sh` job.
- Keep the four strict holdout trajectories out of training, validation, and checkpoint selection.
- Reuse the existing complete-frame prepared cache and the 28,103-anchor training manifest.
- Initialize from a completed five-stream K=5 checkpoint chosen only with development data.
- Keep checkpoints on the remote server and package only compact results.

## Implementation

1. Add a normalized V7F force score that measures mean absolute force error divided by training-set force scale.
2. Add configurable checkpoint selection, validation batch coverage, trainable module prefixes, and teacher-forcing probability endpoints.
3. Freeze all modules except `v7f` for the continuation stage and build the optimizer only from trainable parameters.
4. Train with uniform five-horizon weights for five required epochs, continuing through epoch seven when the development force score is still improving.
5. Save `best_force.pt` using the force score and retain durable `latest.pt` checkpoints for restart safety.
6. Add a detached post-run launcher that waits for the current comparison job to finish successfully before selecting the initialization checkpoint and starting the continuation.
7. Evaluate the selected checkpoint on the complete 5,037-anchor strict K=5 holdout manifest and export aggregate, per-terrain, per-trajectory, per-leg, and per-horizon metrics.

## Verification

- Unit tests first for force score, prefix freezing, checkpoint metric selection, and probability scheduling.
- Run the complete local unit suite.
- Run remote syntax/import checks and the remote unit suite before launch.
- Confirm the detached launcher PID, parent PID, waiting/running phase, logs, and output paths.
- Do not claim completion until `Training complete.` is present, no trainer remains, strict evaluation exists, and the result archive checksum is verified.

