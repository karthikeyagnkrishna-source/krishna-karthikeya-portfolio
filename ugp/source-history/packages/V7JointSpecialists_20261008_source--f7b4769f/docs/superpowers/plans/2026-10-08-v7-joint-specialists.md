# V7 Joint Specialists Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Native execution is recommended for this experiment to reduce repeated context and review costs.

**Goal:** Implement and train dedicated joint specialists on the existing 28 trajectories, then report whether every individual leg prediction meets the approved 2% or exact-zero tolerance.

**Architecture:** Freeze the V7 foundation and parent body predictor. Warm-start nine independent joint specialists per short/long branch from V7Legs, add causal raw-history and valid planned-command encoders, and train torque, angles/speeds, torque retuning, then force/contact. Reuse the existing data, robot geometry, checkpoints and 96-case evaluator.

**Tech Stack:** Existing Linux `v7f4080` Python environment, PyTorch, NumPy, rosbags, SciPy and the existing CSV/Excel exporter. No new vision models or dependencies.

**Spec:** `docs/superpowers/specs/2026-10-08-v7-joint-specialists-design.md`, approved 2026-10-08.

## Global Constraints

- Exactly the parent's 28 unique training trajectories, eligible time ranges, temporal train/validation masks and four excluded trajectories.
- +0.1 s uses the short branch with 0.5 s history; +0.2 through +1.0 s uses the long branch with 4 s history.
- Every individual nonzero prediction requires error **strictly below** 2%. Effective zero is `abs(actual) <= 1e-8` in the reported unit; small nonzero values remain subject to 2%.
- Exact-zero limits: angle 0.05 degrees, speed 0.01 rad/s, torque 0.05 N m, force component/vector 1 N. Contact probability error must be below 0.02. Foot relative error uses the fixed URDF base origin; if its actual position norm is zero, report relative error as undefined and require zero predicted displacement.
- No actual future robot state enters an input. Future recorded commands are assumed known controls in this offline benchmark. Active position/velocity/gain inputs are explicitly an expansion over the old torque-only controls.
- Keep body outputs at parent seed values. No recurrent correction loop. No unseen fitting, calibration, scaling or checkpoint selection.
- Separate stages: up to 30 epochs, patience five, three-epoch absolute-loss warmup; retuning up to 15 epochs. AdamW: 3e-4 separate stages, 5e-5 retuning, weight decay 1e-4, gradient norm limit 1. Short batch 16; long batch 8.
- New files live under `v7_joint_specialists/`; `sources/`, `v7legs_body/`, `v7feedback/` and completed outputs are references. Do not modify them.
- Preserve source/weight/data/scaler signatures in resumable checkpoints. Do not commit to an ambient drive-root Git repository; use source snapshots and manifests unless a workspace-owned repository is verified.
- Read GPU owners before launch; do not stop unrelated processes. Training completion and achieving the accuracy target are separate status fields.

## Review Focus

1. Tiny nonzero torque and exact 2% boundary: no expanded zero mask, rounded-score pass or division-by-zero shortcut (Task 1).
2. Inactive or unexpected actuator modes, duplicate timestamps and mismatched joint ordering: invalid fields cannot become future-state hints; retain records and expose diagnostic flags (Task 2).
3. Empty command prefixes/padding and future sensor changes: finite predictions with no future-state dependence or added post-target command access (Tasks 2-3).
4. Interrupted training and changed dependencies: resume the same sampling/optimizer state and reject mismatched signatures (Task 4).
5. Partial or stale evaluation: missing cases cannot yield a complete report or `goal_achieved=true` (Task 6).

## Files and shared interfaces

Create `contracts.py`, `strict_metrics.py`, `command_data.py`, `joint_data.py`, `joint_models.py`, `joint_objectives.py`, `force_models.py`, `train_joint_specialists.py`, `evaluate_joint_specialists.py`, `run_training.sh`, `README.md`, `IMPLEMENTATION.md` and focused tests under `v7_joint_specialists/`.

`contracts.py` owns `ToleranceConfig`, `InputBatch`, `Targets` and `Predictions`:

- `InputBatch.base`: original prediction inputs only: x, base_leg, base_body, force, contact, horizon. Never target fields.
- `InputBatch.history`: `[B,4,T,83]`; `history_valid`: `[B,4,T]` bool.
- `InputBatch.commands`: `[B,4,S,55]`; `command_valid`: `[B,H,4,S]` bool, permitting raw command timestamps from the anchor up to, but strictly before, each target timestamp.
- `Targets`: leg `[B,H,4,9]` in rad/rad-s/N m, force `[B,H,4,3]`, body `[B,H,12]`, contact `[B,H,4]`, target timestamps and row identifiers.
- `Predictions`: leg, force, body, contact in the corresponding shapes; r `[B,H,4,3]` in metres; specialist hidden states `[B,H,4,9,256]` where required.
- `ScaleBundle` is a dictionary containing unchanged parent reconstruction scales, new input scales, `loss_leg_std [4,9]`, `loss_force_std [4,3]`, and schema/signature metadata.

The command schema per joint is six values `(joint_torque, position, velocity, pid_gains_p, pid_gains_i, pid_gains_d)`, six validity bits and mode one-hot `(10,12,13,other)`: 16 entries. Three joints plus U(6) and relative time(1) give 55 entries per leg. History contains P(12), E(3), contact(1), body(12), U(6), command fields/masks/modes(48) and relative time(1): 83 entries. Current and other raw fields remain audit metadata, not assumed valid controls.

## Task 1: Exact scoring and tolerance loss

**Files:** `contracts.py`, `strict_metrics.py`, `joint_objectives.py`, `tests/test_strict_metrics.py`.

**Interfaces:** `score_scalar(actual, predicted, kind, cfg) -> dict[str,np.ndarray]`; `score_vector(actual, predicted, kind, cfg) -> dict`; `strict_summary(rows) -> dict`; `scalar_tolerance(target:Tensor, kind:str, cfg) -> Tensor`; `specialist_loss(predicted, target, kind, scale, cfg, warmup:bool) -> Tensor`.

- [ ] Write failing tests with these exact assertions: torque actual `[100,100,0,0.1]`, predicted `[101.99,102,0.04999,0.10201]` gives pass `[true,false,true,false]`; zero torque prediction 0.05 fails. A tiny nonzero actual 1e-7 uses 2%, not the zero limit.
- [ ] Add tests for rad-to-degree scoring, zero force vector, a force-vector pass with an individual component failure, contact error exactly 0.02 failing, and nonfinite values being rejected. `strict_summary` must keep exact-zero failures while excluding undefined percentages from percentage averages.
- [ ] Run `python -m unittest discover -s tests -p test_strict_metrics.py -v`; observe meaningful failure before implementing.
- [ ] Implement float64 scoring and the spec's loss: normalized SmoothL1 plus 0.2 mean violation and 0.5 worst-10% violation after warmup. Use `b=log1p(relu(error/tolerance-1))`; zero-case tolerances stay positive. Make an empty loss selection return finite zero, never NaN. No scoring denominator clamp beyond the approved zero rule.
- [ ] Re-run the focused tests; record pass output and source hashes.

## Task 2: Command extraction and lazy aligned windows

**Files:** `command_data.py`, `joint_data.py`, `tests/test_command_data.py`.

**Consumes:** contracts from Task 1, parent caches/masks, `foundation.sha256`, `data.atomic_save/atomic_json`, `provenance.canonical_record`, original ROS parsing conventions.

**Produces:** `extract_commands(bag:Path, physics:dict, mapping:np.ndarray, output:Path) -> Path`; `JointDataset(parent:Path, feedback:Path, command_root:Path, branch:str, split:str)` returning `(InputBatch,Targets)`; `fit_input_scales(dataset) -> ScaleBundle`; `collate_windows(items) -> tuple`.

- [ ] Write tests that reorder a deliberately permuted actuator array; mode 10 exposes torque only, mode 12 position/velocity/torque, and mode 13 also its supplied gains. Inactive garbage/NaNs become masked zeros. An unexpected mode remains a flagged record with new command fields masked and the legacy inputs retained.
- [ ] Test duplicate/out-of-order stamps, a cache target shifted by one sample being rejected, all-padding command masks, and sensor perturbations after the anchor leaving inputs unchanged. Commands exactly at the target timestamp or later cannot enter its new prefix; a valid just-before-target raw command remains available.
- [ ] Observe the focused test failures, then implement the parser using the same median command timestamp and causal-hold grid as the legacy extraction. Verify reconstructed torque commands against `L_LEG`. Retain the original raw command stamps for the new sequence encoder, including valid sub-grid commands before the target. Label this finer input resolution as an input expansion; the output sampling grid is unchanged. Verify a consistent hardware mapping from training evidence; never estimate one from future held-out state.
- [ ] Store per-trajectory arrays and indices, not duplicated histories. Preserve original train/val masks and match anchors/targets with absolute tolerance 1e-7, rtol zero. Fit new scales on optimization windows only; masks, mode bits and time features retain declared representations. Apply validity masks after normalization.
- [ ] Bind cache signatures to source bags, physics, parent caches, code/schema and mapping. Audit all 28 trajectories: modes, active fields, timestamp discrepancies, command/effort tracking and label extremes. Do not expand optimization ranges using the earlier exploratory audit.
- [ ] Pass tests and compare real cached force/leg targets with the original parent timestamps before any GPU training.

## Task 3: Nine warm-started joint specialists

**Files:** `joint_models.py`, `tests/test_joint_models.py`.

**Consumes:** InputBatch, ScaleBundle, parent `Hybrid`, `transformer`, `TorchFK` and checkpoints.

**Produces:** `JointSpecialist(parent_leg_head, kind:str, joint:int, output_only:bool=False)`; `JointSpecialists(seed:Hybrid)` with `configure(stage:str)`, `forward(inputs, scales) -> Predictions` and `active_parameters()`.

- [ ] Write tests for all nine scalar outputs, copied weights not sharing trainable storage, no future-target arguments, finite all-padding inputs, and initialization matching the parent leg prediction within the existing real-data check tolerance (`atol=1e-5,rtol=1e-5`). Body and legacy force must remain identical.
- [ ] Observe failure, then copy the parent projection, 256-wide two-layer temporal Transformer and the appropriate scalar output row. Add separate 64-wide two-layer history and causal command encoders, each with four attention heads, feedforward width 256 and dropout 0.1. Insert their zero-initialized adapters **after** the copied temporal representation so newly added command-prefix features do not flow through other future horizons. In prefix tests, first set adapters nonzero so invariance cannot pass trivially at zero initialization.
- [ ] Read command tokens with each horizon's prefix mask; ensure future tokens cannot contaminate earlier token encodings. For an entirely padded prefix, use a masked zero representation without an all-negative-infinity softmax.
- [ ] Torque conditioning uses detached predicted q/dq, the fixed seed torque prior and legacy predicted force. Angles/speeds and seed body are evaluated without gradients during torque training. Seed and all inactive specialists remain in `.eval()` when a selected head trains, so dropout cannot change their fixed predictions.
- [ ] Implement `output_only=True` as the cheap control: new encoders off, copied projection/Transformer frozen, scalar MLP trainable. Retain internal radians and explicit physical reconstruction; no float16 final output/subtraction. Tests must show gradients only in the selected specialist.
- [ ] Pass tests and verify one real batch for both branches; reuse existing FK correctness checks rather than rerunning a broad smoke suite.

## Task 4: Staged trainer, hard examples and resumability

**Files:** `train_joint_specialists.py`, `joint_objectives.py`, `tests/test_training_state.py`.

**Consumes:** Tasks 1-3; reuse `train.Reporter`, `balanced_order`, `capture_rng`, `restore_rng`, existing atomic/checkpoint signature helpers.

**Produces:** `validate_head(model,dataset,scales,stage,cfg) -> tuple[tuple,dict]`; `train_head(model,datasets,scales,stage,output,signature,report) -> Path`; `main()` supporting `--root --parent --feedback --output --audit-only --verify-only --train-only`.

- [ ] Test one optimizer step changing only its selected specialist, resumed next-step equality with uninterrupted execution, changed cache/code/scaler signatures being rejected, and changed unseen labels having no effect on training selection.
- [ ] Observe failure, then implement seed 2781, approved hyperparameters, trajectory/terrain balance and training-only transition/hard-example sampling. Score a head using its active task only: `(failure_count, worst_relative_error, physical_MAE)` lexicographically. Record absolute-zero failures separately, including them in failure count.
- [ ] Execute stages for each branch: torque-output-only control; T-HAA/HFE/KFE; Q-HAA/HFE/KFE; D-HAA/HFE/KFE; T-HAA/HFE/KFE retuning; force/contact. Torque retuning receives final detached q/dq. Angle training adds 0.2 normalized geometry supervision and retains its own joint target loss.
- [ ] Save per-head best/latest state, independent optimizer, epoch/batch cursor, sample order, RNG, patience and dependency hashes every epoch and 250 steps. Save manifests referencing frozen predecessors instead of repeatedly duplicating all frozen weights.
- [ ] Report phase, branch/head, actual trajectory/window counts, optimizer steps, finite loss/gradients, validation failures and throughput. Estimate duration only after measured real steps. Reject empty splits and nonfinite active labels; preserve diagnostics rather than silently dropping records.
- [ ] Pass tests; provide audit-only and real-batch verification paths using the existing server environment without installation.

## Task 5: Force/contact extension with fixed joints and body

**Files:** `force_models.py`, `tests/test_force_extension.py`.

**Consumes:** frozen final joint specialists, parent seed, existing `FeedbackModel` one-pass checkpoint, `ForceRefiner` processing layers, strict metrics/loss.

**Produces:** `ForceContactSpecialist(joints:JointSpecialists, force_prior:FeedbackModel)` with `forward(inputs,scales) -> Predictions` and `configure_force()`.

- [ ] Test a force update leaving all joint/body predictions unchanged and gradients absent from their weights; a linear correction can exceed two training standard deviations; future force/contact labels cannot affect forward output.
- [ ] Observe failure, then use the frozen existing one-pass force prediction as the prior. Warm-start the force-processing layers from its checkpoint, initialize the new final linear correction to zero, and add the unbounded physical correction to that prior. The total new correction has no tanh cap.
- [ ] Project the nine leg representations to the existing 256-value force-token interface; initialize the projection as their average. Use parent seed body/hidden values unchanged. Add zero-initialized contact-logit correction to clipped parent logits and document its initialization tolerance.
- [ ] Train component and vector force supervision, strict tolerance violations and contact BCE. Contact remains a predicted feature; do not hard-zero all forces using a true future contact label or an assumed zero-force rule for contact-negative samples.
- [ ] Pass tests, then retain the force/contact best checkpoint using validation failures, worst error and physical MAE; do not select it using unseen tests.

## Task 6: Matched 96-case evaluation and exact exports

**Files:** `evaluate_joint_specialists.py`, `tests/test_evaluation.py`.

**Consumes:** trained bundles, strict scoring, reused `load_runtime/load_models/load_record`, `prediction_record`, causal local adaptation, `export_case` and existing exporter formatting.

**Produces:** `evaluate_all(rt,output,parent,feedback,fk,manifest,report) -> dict`; `export_variant(meta,variant,inputs,targets,predictions) -> dict`; strict summaries, failure tables and one exact workbook per variant.

- [ ] Test 96 unique case identities, preservation of leg/joint/horizon/target times, resumable case signatures, zero/nonzero denominators, and an incomplete case set being unable to finalize. A synthetic single failing point must set `goal_achieved=false` even if every aggregate mean is below 2%.
- [ ] Observe failure, then copy the existing evaluator flow into the new package, reusing its helpers. Keep 16 familiar, 64 gap and 16 unseen cases. Share original V7 four-epoch and V7-F five-epoch local adaptation across variants; all new corrections use zero test-time local epochs.
- [ ] Export `base_v7`, `base_v7f`, `seed`, `torque_control`, `torque_only`, `joint_specialists`, `full`. Retain the first torque-stage snapshot for `torque_only`. Compute standalone V7 force correctly from its original force residual plus the original force baseline/scales, instead of copying V7-F force into that row.
- [ ] Use exactly predicted states in forward calls. Compute FK(actual angles) outside the model for loss/export only. Include angle/speed/torque and Fx/Fy/Fz errors, force vector/cosine, contact and foot-position errors, zero flags, applicable tolerances and strict pass/fail in exact records.
- [ ] Export per-leg/joint/horizon means, 95th/99th percentiles, maxima, valid-point counts and every failing exact case. Keep all original rows. Retain `Joint exact`, `Body exact`, `Force exact` and `How to read`, with additional failure/strict-summary outputs.
- [ ] Hash inputs/checkpoints/code/scoring configuration in each case signature. `complete` means every case/export exists; `goal_achieved` requires zero required failures for the full model across all splits. Describe existing excluded trajectories as development holdouts, not fresh blind-test evidence.
- [ ] Pass tests and reconcile exported values with saved tensors; no claim of <2% before the complete observed report.

## Task 7: Delivery, review and verified GPU launch

**Files:** `run_training.sh`, `README.md`, `IMPLEMENTATION.md` and the isolated source bundle.

- [ ] Record the approved spec/scoring, source hashes, reused dependencies, material test commands/results and all stages. Complete one independent whole-implementation review after native execution and fix concrete findings before launch; do not dispatch a new reviewer for every task.
- [ ] Package only the new source and required read-only parent modules; omit local dependency caches, other projects and trained parent artifacts. Remote root: `/path/to/ugp/V7JointSpecialists_20261008`.
- [ ] Reuse `/path/to/ugp/python`; set explicit root/parent/feedback/output paths and PYTHONPATH. Acquire a run lock. An existing run with different signatures must remain recoverable; use a new identity rather than overwriting it.
- [ ] Read current GPU utilization, memory and compute PIDs/owners. Use an available GPU; low utilization with retained unrelated jobs is not proof of availability. Do not interrupt other jobs. Start on one available GPU to avoid shared-manifest races.
- [ ] Run the focused material suite once and real-data verification for both short/long, then launch detached training. Confirm a matching live PID, an optimizer-step log and a new resumable checkpoint before reporting training started.
- [ ] Leave readable live status and automatic evaluation. Report a failed accuracy target honestly, preserving checkpoints and exact failure tables for the next evidence-based change.

## Verification commands

From the new source folder on the server, use the existing environment:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=../v7legs_body:../v7feedback /path/to/ugp/python -m unittest discover -s tests -v
```

The targeted tests in Tasks 1-6 are the required checks. No additional blanket smoke-test round is required after this suite and real-data gradient/alignment checks pass.

## Self-review and execution gate

- [x] Spec coverage: nine specialists, full valid commands, torque-first schedule, corrected force/contact, fixed body, strict metrics, raw inputs and all 96 cases map to Tasks 1-7.
- [x] Interface/type consistency: contracts own shapes and units; all tasks use the same scoring, dataset, forward, training and export interfaces.
- [x] Review Focus: all five conditions have explicit material tests in their owning tasks.
- [x] Efficiency: frozen vision, immutable caches, lazy windows, per-head snapshots, existing environment and one review after implementation.
- [x] User review of this implementation plan and execution method: approved direct execution.
- [ ] Implementation, verification, deployment, training and evaluation.

Recommended execution: **native** (implement directly in this chat, one independent review at the end). The interfaces are tightly connected and this minimizes repeated context costs. Subagent-driven execution remains an alternative if the user prefers per-task independent reviews.
