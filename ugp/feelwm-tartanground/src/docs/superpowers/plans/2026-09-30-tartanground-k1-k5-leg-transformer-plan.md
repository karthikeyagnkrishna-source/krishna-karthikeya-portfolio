# TartanGround K=1/K=5 Baseline vs Per-Leg Transformer Implementation Plan

> **For inline execution:** Use `superpowers:executing-plans` and complete every task continuously with TDD.

**Goal:** Launch matched baseline and per-leg-transformer K=1 then K=5 experiments on the 28-trajectory split, evaluate only four unseen trajectories, and package results locally and remotely.

**Architecture:** Reuse frozen multimodal caches and one deduplicated post-Perceiver latent cache indexed by manifests. Add optional independent FL/FR/HL/HR pre-transformers between FiLM groups and the unchanged shared transformer. Roll K=5 recurrently with predicted state/refined force, deterministic 50/50 teacher forcing, weighted per-horizon losses, dev-only lambda calibration, and strict holdout evaluation.

**Tech Stack:** Python 3.10, PyTorch, NumPy, YAML, shell orchestration, `unittest`.

**Spec:** `docs/superpowers/specs/2026-09-30-tartanground-k1-k5-leg-transformer-design.md`

## Global Constraints

- Two 700-frame windows are centered at trajectory quarters; overlap is deduplicated and memberships retained.
- Training/development use exactly the authoritative 28 trajectories; the four named holdouts are test-only.
- GPU 0 runs baseline and GPU 1 runs per-leg, with identical seeds/sample order/effective batch size.
- K=5 development and testing are fully autoregressive; no future measured state enters an autoregressive continuation.
- Existing shared world-transformer depth and prediction heads do not change.
- All phases resume safely; expensive frozen features are computed once.

## Task 1: Deterministic indexed dataset preparation

**Files:**
- Create: `tartanground/windows.py`
- Create: `tartanground/indexed.py`
- Create: `scripts/prepare_tartanground_indexed.py`
- Test: `tests/test_tartanground_indexed.py`

**Interfaces:**
- `quarter_windows(frame_count, length=700, min_center_gap=50) -> tuple[Window, Window]`
- `split_anchors(anchor_indices, purge_frames=50, train_fraction=0.8) -> dict[str, list[int]]`
- `encode_cache_native_action(commands) -> ndarray[...,4]`
- `IndexedTartanGroundDataset(manifest_path, split, horizon)` returns the existing model batch contract.

**Steps:**
1. Add tests for quarter centers, clamping, overlap deduplication with dual membership, five-second center-gap rejection, chronological 80%/purge/dev isolation, holdout exclusion, and `[HAA,HFE,KFE,||delta||]` action encoding. Run them and confirm failures are missing APIs.
2. Implement the pure window/split/action functions and rerun the focused tests to green.
3. Add a synthetic indexed-cache fixture and tests proving K=1/K=5 sample shapes, boundary rejection, deterministic balanced trajectory order, train-only normalization, and no target-derived action channels. Confirm failure, implement the dataset/preparer, then rerun focused and full suites.
4. Commit preparation code and tests.

**Completion command:** `python -m unittest tests.test_tartanground_indexed -v`

## Task 2: Optional four-leg pre-transformers

**Files:**
- Modify: `config.py`
- Modify: `model.py`
- Test: `tests/test_leg_pretransformers.py`

**Interfaces:**
- Config fields: `use_leg_pretransformers`, width `384`, layers `4`, heads `8`, FFN ratio `4`, residual scale `0.1`.
- `MergedFeelWMV7.apply_leg_pretransformers(groups)` transforms only G4/FL, G5/FR, G1/HL, G2/HR before `world_forward`.

**Steps:**
1. Add tests that disabled mode preserves the exact baseline parameter set/output path, enabled mode creates four non-shared blocks, maps correct groups, leaves raw/body groups unchanged, preserves `[B,9,64,800]`, and propagates gradients through every block. Run and confirm expected failures.
2. Implement minimal config/model changes, rerun focused tests, then full suite.
3. Commit architecture code and tests.

**Completion command:** `python -m unittest tests.test_leg_pretransformers tests.test_model_forward -v`

## Task 3: Correct K=5 recurrence, teacher forcing, and horizon losses

**Files:**
- Modify: `model.py`
- Modify: `losses.py`
- Create: `tartanground/rollout.py`
- Test: `tests/test_k5_training.py`

**Interfaces:**
- `teacher_forcing_mask(sample_ids, seed) -> bool tensor` is deterministic and exactly balanced over every even schedule cycle.
- `rollout(..., teacher_force_mask=None)` uses recorded prior targets only where mask is true.
- Autoregressive state contains predicted visual, q/dq/body, tau, per-prefix V7-F refined force, contact, and load; planned actions remain exogenous.
- `compute_losses(..., horizon_lambdas)` returns `horizon_1..horizon_5` and weighted total.

**Steps:**
1. Add controlled tests for the exact 50/50 schedule, predicted-state feedback, refined-force feedback, absence of future-measurement leakage, causal prefix V7-F, and all five lambda contributions. Run and confirm failures.
2. Implement recurrent-state construction and teacher forcing with differentiable predicted kinematics; rerun focused tests.
3. Implement per-horizon loss accounting/weighting and audit metrics; rerun focused and full suites.
4. Commit recurrent training code and tests.

**Completion command:** `python -m unittest tests.test_k5_training tests.test_model_forward -v`

## Task 4: Calibration, strict evaluation, and resumable orchestration

**Files:**
- Create: `tartanground/evaluate.py`
- Create: `scripts/calibrate_lambdas.py`
- Create: `scripts/evaluate_strict.py`
- Create: `scripts/run_tartanground_compare.sh`
- Create: `scripts/monitor_tartanground_compare.sh`
- Create: `configs/tartanground_baseline_k1.yaml`
- Create: `configs/tartanground_leg_k1.yaml`
- Create: `configs/tartanground_baseline_k5.yaml`
- Create: `configs/tartanground_leg_k5.yaml`
- Test: `tests/test_tartanground_experiment.py`

**Interfaces:**
- Lambda calibration accepts train/dev manifests and rejects holdout rows.
- Strict evaluator rejects any train/dev trajectory and exports per-horizon/per-leg physical metrics plus predictions.
- Orchestrator phases: prepare, K1 pair, calibration, K5 pair, strict evaluation, package; each has an atomic completion marker and resumable command.

**Steps:**
1. Add synthetic tests for dev-only lambda selection, strict holdout enforcement, exact physical metrics, restart markers, and GPU/config parity. Run and confirm failures.
2. Implement calibration/evaluation/orchestration/configs, rerun focused and full suites.
3. Run one-step CPU/GPU synthetic smoke tests for baseline/per-leg K=1/K=5; verify finite losses and active new gradients.
4. Commit experiment code and tests.

**Completion command:** `python -m unittest discover -s tests -v`

## Task 5: Remote launch, monitoring, evaluation, and package transfer

**Files:**
- Create: `scripts/package_tartanground_compare.py`
- Modify: `README.md`
- Test: `tests/test_package_tartanground_compare.py`

**Steps:**
1. Add packaging tests for required checkpoints, configs, manifests, metrics, predictions, logs, source, README, and checksums; confirm failure, implement, and rerun.
2. Sync committed source to `/path/to/ugp/merged_feelwm_v7`, run the remote full suite, and start the resumable orchestrator under `nohup` in the comparison run root.
3. Confirm both GPUs have the intended live phase/process and report the single monitoring command immediately.
4. When training completes, run strict evaluation and packaging, download the ZIP without transforming it, and verify matching SHA-256 hashes.
5. Commit packaging/docs.

**Completion command:** `python -m unittest discover -s tests -v`

## Review Focus

- Window semantics under overlap and edge clamping.
- Any path by which measured future q/dq/tau/force/contact/load/visual data could enter autoregressive inputs.
- Per-prefix V7-F force, rather than preliminary base force, is recurrent.
- Exact split isolation and absence of holdout data in statistics/calibration/checkpoint choice.
- True baseline equivalence when leg blocks are disabled and independence/correct mapping when enabled.
- Resume behavior cannot silently mix stale caches/checkpoints/configs.
