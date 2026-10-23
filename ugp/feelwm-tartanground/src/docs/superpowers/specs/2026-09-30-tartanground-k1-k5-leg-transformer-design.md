# TartanGround K=1/K=5 Baseline vs Per-Leg Transformer Design

Date: 2026-09-30  
Status: Approved design, pending written-spec review

## Intent

Train and compare two otherwise matched merged Feel-WM V7 architectures on the established 28-trajectory TartanGround training split:

1. the current baseline, where all FiLM-conditioned leg groups enter the shared world transformer directly; and
2. a per-leg variant, where each FiLM-conditioned leg group first passes through its own substantial transformer before the existing shared world transformer.

Both variants must first learn a strong one-step (`K=1`) predictor and then initialize a five-step (`K=5`) predictor. The `K=5` model must feed its own predicted future state back into later steps, train with an exactly balanced mixture of teacher-forced and autoregressive examples, and apply an explicitly weighted loss at every horizon. Evaluation must use only four trajectories absent from training and development. Results and reproducibility artifacts must be packaged on the SSH server and downloaded locally.

Compute and storage must be used conservatively because the available execution allowance is limited.

## Fixed Dataset Split

The authoritative 28-trajectory training list is `V7_EXPANDED_TRAIN_TRAJECTORIES.csv` in the shared V7 results. The following four complete cached trajectories are excluded from all training, development, normalization, lambda selection, and checkpoint selection:

- `ForestEnv/P2001`
- `GreatMarsh/P2027`
- `OldTownFall/P2001`
- `SeasonalForestWinter/P2006`

Those four trajectories form the strict evaluation set. Test metrics may be read only after all hyperparameters and checkpoints are fixed from training/development data.

### Frame budget and quarter-centered windows

Preparation uses **two 700-camera-frame windows per trajectory**, not 700 frames total. For a trajectory with `F` camera frames, their desired centers are `0.25 * (F - 1)` and `0.75 * (F - 1)`. Each start is rounded to the nearest camera index and clamped to `[0, F - 700]`; its inclusive end is `start + 699`. The two centers must be at least five seconds apart (50 camera frames at 10 Hz).

The windows may overlap on short trajectories. Overlapping camera frames are stored only once in the post-Perceiver cache, while the preparation manifest preserves both window memberships and both intended centers. This avoids redundant compute without changing which window an anchor belongs to. Anchors near a window boundary are retained only when their four visual inputs and all required future visual targets lie inside at least one complete 700-frame window. Exact frame indices, timestamps, desired centers, clamped centers, and memberships are written to the manifest.

For each of the 28 training trajectories, valid anchors from the sorted union of the two windows are split chronologically:

- the earliest 80% are training candidates;
- the next 50 camera frames form a five-second purge region and are never sampled; and
- the remaining anchors are development candidates.

This preserves both quarter-centered windows while preventing adjacent train/development anchors from sharing a five-second neighborhood. Anchors that lack the required four visual input frames, 50 proprioceptive samples, or five future target frames are removed. Consequently, final anchor counts are recorded rather than assumed.

For each strict evaluation trajectory, all valid anchors from the union of both 700-frame windows are test candidates. Test data never contributes normalization statistics.

## Shared Cache Preparation

The existing DINO, ConvNeXt, geometry/Utonia, flow/WAFT, physics, contact, and frame-manifest caches are reused read-only. The frozen V7 resampler checkpoint is applied once to the unique union of the two selected 700-frame windows for all experiments. Post-Perceiver outputs are stored per trajectory and reused by baseline, per-leg, `K=1`, `K=5`, development, and testing.

Preparation is idempotent and atomic. A trajectory cache is reused only when its source fingerprint, selected frame indices, shapes, finite-value checks, and resampler SHA-256 match.

To avoid duplicating large tensors, training examples are index records into per-trajectory arrays rather than complete copied samples. Balanced sampling cycles across all 28 trajectories so every trajectory contributes to both architecture variants.

## Cache-Native Action Representation

Each leg retains the existing four-channel action interface, but every channel is populated by an available causal/planned control signal:

1. cached HAA low-level command;
2. cached HFE low-level command;
3. cached KFE low-level command; and
4. Euclidean magnitude of the command change from the current control step to the next planned control step.

For a per-leg command vector `L_t`, channel four is `||L_(t+1) - L_t||_2`. Commands are normalized using training-only statistics. The same representation is used for all architectures, horizons, splits, and evaluation trajectories.

Recorded future commands are treated as conditional exogenous controls. At deployment they must be replaced by controller/planner commands. Actual future measured `q`, `dq`, `tau`, force, contact, or visual state must never be encoded as an action.

## Architecture Variants

### Baseline

The baseline is the current `MergedFeelWMV7` architecture. Its four action/proprioception-conditioned FiLM leg groups and the raw/body groups are flattened directly into the existing eight-layer, 800-wide, ten-head world transformer.

### Per-leg pre-transformer variant

The modified model changes only the path between leg FiLM and the existing shared world transformer. The FiLM groups are:

- `G4` for FL,
- `G5` for FR,
- `G1` for HL, and
- `G2` for HR.

Each group contains 64 visual-modality tokens at width 800. Each leg owns an independent pre-transformer with:

- input projection `800 -> 384`;
- four self-attention blocks;
- width 384;
- eight attention heads;
- feed-forward ratio 4;
- dropout matching the base model;
- output projection `384 -> 800`; and
- a residual connection with a small initial residual scale of 0.1.

There is no parameter sharing across FL, FR, HL, and HR. Raw visual and body groups bypass these blocks. After leg-local processing, all nine groups enter the unchanged shared world transformer and all existing prediction heads remain unchanged.

The variant is controlled by configuration so the same codepath can instantiate a true baseline with the leg blocks absent. Shared baseline weights are initialized identically between variants; the added leg blocks use the same experiment seed.

## K=1 Training

Baseline and per-leg models train concurrently on separate RTX 4080 GPUs:

- GPU 0: baseline;
- GPU 1: per-leg variant.

Both use identical sample order, seed 781, optimizer family, learning-rate schedule, precision, normalization, loss definitions, and effective batch size. Single-GPU batch size is one with gradient accumulation eight, giving effective batch size eight for each model.

Each `K=1` sample uses causal history through time `t`, planned commands for `t -> t+0.1`, and targets at `t+0.1`. Each model is capped at 200 optimizer steps with validation and durable checkpoints every 50 steps. The best development checkpoint and latest resumable checkpoint are preserved.

The two best `K=1` model states initialize their corresponding `K=5` models. K=5 receives model weights but starts a fresh optimizer and scheduler.

## K=5 Autoregressive State Flow

K=5 predicts horizons `+0.1`, `+0.2`, `+0.3`, `+0.4`, and `+0.5` seconds.

In autoregressive mode, step `k+1` consumes step `k` predictions for:

- visual latent state;
- joint position and velocity;
- body state;
- joint torque;
- final refined V7-F foot force;
- contact probability; and
- load-regime probability.

Predicted joint state is also used to update leg query features and differentiable ANYmal foot kinematics. Predicted state constructs subsequent physics features and baselines. The last observed tracking-error vector is carried forward because a future measured tracking error is unavailable. Only planned commands and horizon values come from the future control sequence.

No actual future measured state may enter an autoregressive continuation.

The V7-F specialist is evaluated causally on each available prefix so its refined force, rather than the preliminary base-force prediction, is fed into the next step.

## Teacher Forcing

K=5 training uses an exactly balanced deterministic mode schedule shared by both architecture variants:

- 50% of training examples are teacher-forced after step one; and
- 50% are fully autoregressive after step one.

Teacher-forced examples substitute the recorded previous target state for the previous prediction when constructing the next recurrent input. Target values still never enter action encodings. Development and strict testing are always 100% autoregressive.

The training metrics record separate teacher-forced and autoregressive loss summaries so the mixture can be audited.

## Per-Horizon Loss and Lambda Selection

For every horizon `k`, `L_k` contains the existing visual, contact, load, joint/body, torque, base-force, and V7-F supervision evaluated against the actual target at that horizon. The K=5 objective is:

`L_K5 = lambda_1 L_1 + lambda_2 L_2 + lambda_3 L_3 + lambda_4 L_4 + lambda_5 L_5 + temporal/impulse regularizers`.

All lambda profiles sum to one. Three profiles are calibrated from the 28-trajectory development anchors only:

- uniform: `[0.20, 0.20, 0.20, 0.20, 0.20]`;
- mild decay: `[0.30, 0.25, 0.20, 0.15, 0.10]`; and
- strong decay: `[0.516, 0.258, 0.129, 0.065, 0.032]`.

Starting from the baseline K=1 checkpoint, each candidate receives the same short 20-step K=5 calibration sequence and is evaluated autoregressively on development anchors. The lowest development physical score wins. The score is the equal-weight mean of training-normalized q error, torque error, force-vector error, and contact error across all five horizons.

The selected lambda profile is then used unchanged for both baseline and per-leg K=5 training. This prevents test leakage and keeps the architecture comparison fair.

Both full K=5 runs execute concurrently on separate GPUs and are capped at 100 optimizer steps, with validation and checkpoints every 25 steps.

## Evaluation

After training and checkpoint selection are complete:

- each K=1 model is tested at `+0.1 s` on all four strict trajectories;
- each K=5 model is tested fully autoregressively at all five horizons on all four strict trajectories; and
- no teacher forcing is used.

Metrics are exported globally and by architecture, checkpoint type, environment, trajectory, horizon, and leg:

- q MAE and RMSE in radians and degrees;
- torque MAE and RMSE in N m;
- force-vector MAE/RMSE;
- force-magnitude MAE;
- force-direction cosine on contacting feet;
- contact accuracy, precision, recall, and F1;
- load-regime accuracy;
- visual loss components; and
- per-sample predicted and actual q, torque, force, contact, and load values.

The report must distinguish development selection from strict-test results and must not describe validation losses as physical-unit test metrics.

## Compute Controls and Live Monitoring

Expensive foundation features are never recomputed. Post-Perceiver preparation is shared and batched. Training uses BF16/TF32, no compilation warm-up, deterministic balanced sampling, and minimal data-loader workers compatible with the server.

Every phase is resumable and writes a state file. The orchestrator records phase, elapsed time, GPU utilization, step, horizon losses, teacher-forcing mode counts, validation summaries, and checkpoint paths. A single monitoring script displays both GPUs and both architecture runs.

If the compute allowance approaches exhaustion, optional additional calibration is skipped; already completed preparation and checkpoints remain usable. Strict-test evaluation and packaging run only after model/lambda selection is fixed.

## Packaging

One results ZIP is created on the SSH server and downloaded byte-for-byte to the local workspace. It contains:

- baseline and per-leg K=1 best/latest checkpoints;
- baseline and per-leg K=5 best/latest checkpoints;
- selected lambda profile and all calibration results;
- resolved configurations and split/window manifests;
- raw and summarized training/development/test metrics;
- per-sample strict-test predictions;
- logs and live-monitor script;
- source code and tests;
- a results README stating limitations;
- SHA-256 checksums.

Raw datasets, duplicated prepared shards, and redundant intermediate checkpoints are excluded. Server and local ZIP hashes must match.

## Testing and Acceptance

Implementation follows test-driven development. Required coverage includes:

- deterministic two-window selection and strict split isolation;
- cache-native action encoding and no future-state leakage;
- four distinct leg pre-transformers attached to the correct FiLM groups;
- baseline behavior unchanged when the feature is disabled;
- K=5 autoregressive feedback uses predicted refined force and predicted state;
- exact 50/50 teacher-forcing schedule;
- every `L_1..L_5` contributes with the selected lambda;
- lambda selection reads development data only;
- test evaluator rejects any training/development trajectory;
- per-horizon physical metrics on controlled synthetic examples;
- restart-safe phase orchestration and package integrity.

Acceptance requires passing unit tests, finite smoke-training losses for both variants, active gradients in all added leg-transformer parameters, successful K=1 and K=5 strict-test evaluation, valid checkpoints, and matching server/local ZIP SHA-256 hashes.

## Explicit Non-Goals

- No changes to the existing shared world-transformer depth or prediction heads.
- No additional foundation-model training.
- No claim of generalization beyond the four strict unseen trajectories.
- No use of strict-test data for normalization, lambda selection, early stopping, or checkpoint selection.
