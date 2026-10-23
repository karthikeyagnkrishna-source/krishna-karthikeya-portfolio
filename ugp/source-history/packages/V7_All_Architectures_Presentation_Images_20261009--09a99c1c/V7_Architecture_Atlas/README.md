# Architecture images — verified code, 9 October 2026

The package contains29 images. All images are 4096 × 2560 PNGs. Editable SVG copies preserve exact labels and can be resized without losing clarity.

## How to read

- Numbers on each image identify detailed computation panels. Arrows within panels show successive transforms; unconnected rows are parallel paths or separate facts.
- Inputs are blue; learned layers purple; geometry green; predictions orange; frozen/inherited paths grey; losses red.
- q is joint angle, dq joint speed, τ measured joint effort, F foot-force xyz, r body-relative foot position. Angles are internal radians and exported as degrees.
- Body12 is acceleration xyz, gyro xyz, linear velocity xyz, angular velocity xyz. There is no learned body world-position/orientation output in these V7 correction models.
- The known future controller commands are an offline benchmark input. Predictions target t+h. Feedback rounds revise the same t+h.
- Numbers after a channel name give vector width, e.g. z768 means a learned vector of768 values. A Transformer FFN is its feed-forward sublayer.

## Latest connected diagram

[Joint Specialists PNG](png/29_V7_Joint_Specialists_connected.png) · [Editable SVG](svg/29_V7_Joint_Specialists_connected.svg)

## Images

### 01_V7_4trajectories: V7 foundation — 4-trajectory version

[PNG image](png/01_V7_4trajectories.png) · [Editable SVG](svg/01_V7_4trajectories.svg)

Four visual branches + causal robot histories + horizon-conditioned residual prediction

Source: V7FullDynamics; FullFourBranchTerrainEncoder; v7_loss — source_snapshots/v7_rtx_notebook.py
Status: Completed experiment / result artifacts; source trajectories: 4.

### 02_V7_28trajectories: V7 foundation — 28-trajectory version

[PNG image](png/02_V7_28trajectories.png) · [Editable SVG](svg/02_V7_28trajectories.svg)

Four visual branches + causal robot histories + horizon-conditioned residual prediction

Source: V7FullDynamics; FullFourBranchTerrainEncoder; v7_loss — source_snapshots/v7_rtx_notebook.py
Status: Completed experiment / result artifacts; source trajectories: 28.

### 03_V7F_4trajectories: V7-F force specialist — 4-trajectory version

[PNG image](png/03_V7F_4trajectories.png) · [Editable SVG](svg/03_V7F_4trajectories.svg)

Original V7 representation → four force experts → vector/magnitude fusion; one shared specialist across legs

Source: V7ForceSpecialistHead; v7f_loss — source_snapshots/v7_rtx_notebook.py
Status: Completed experiment / result artifacts; source trajectories: 4.

### 04_V7F_28trajectories: V7-F force specialist — 28-trajectory version

[PNG image](png/04_V7F_28trajectories.png) · [Editable SVG](svg/04_V7F_28trajectories.svg)

Original V7 representation → four force experts → vector/magnitude fusion; one shared specialist across legs

Source: V7ForceSpecialistHead; v7f_loss — source_snapshots/v7_rtx_notebook.py
Status: Completed experiment / result artifacts; source trajectories: 28.

### 05_FeelWM_baseline_K1: Five-stream FeelWM + V7 + V7-F — baseline, K=1

[PNG image](png/05_FeelWM_baseline_K1.png) · [Editable SVG](svg/05_FeelWM_baseline_K1.svg)

Detailed merged world-model implementation; front/rear terrain timing, token construction and physical prediction

Source: MergedFeelWMV7; LegPreTransformer; compute_losses — source_snapshots/feelwm_notebook.py
Status: Completed experiment / result artifacts; source trajectories: 28.

### 06_FeelWM_baseline_K5: Five-stream FeelWM + V7 + V7-F — baseline, K=5

[PNG image](png/06_FeelWM_baseline_K5.png) · [Editable SVG](svg/06_FeelWM_baseline_K5.svg)

Detailed merged world-model implementation; front/rear terrain timing, token construction and physical prediction

Source: MergedFeelWMV7; LegPreTransformer; compute_losses — source_snapshots/feelwm_notebook.py
Status: Completed experiment / result artifacts; source trajectories: 28.

### 07_FeelWM_per-leg_pre-Transformers_K1: Five-stream FeelWM + V7 + V7-F — per-leg pre-Transformers, K=1

[PNG image](png/07_FeelWM_per-leg_pre-Transformers_K1.png) · [Editable SVG](svg/07_FeelWM_per-leg_pre-Transformers_K1.svg)

Detailed merged world-model implementation; front/rear terrain timing, token construction and physical prediction

Source: MergedFeelWMV7; LegPreTransformer; compute_losses — source_snapshots/feelwm_notebook.py
Status: Completed experiment / result artifacts; source trajectories: 28.

### 08_FeelWM_per-leg_pre-Transformers_K5: Five-stream FeelWM + V7 + V7-F — per-leg pre-Transformers, K=5

[PNG image](png/08_FeelWM_per-leg_pre-Transformers_K5.png) · [Editable SVG](svg/08_FeelWM_per-leg_pre-Transformers_K5.svg)

Detailed merged world-model implementation; front/rear terrain timing, token construction and physical prediction

Source: MergedFeelWMV7; LegPreTransformer; compute_losses — source_snapshots/feelwm_notebook.py
Status: Completed experiment / result artifacts; source trajectories: 28.

### 09_FeelWM_K5_uniform: FeelWM K=5 — uniform loss experiment

[PNG image](png/09_FeelWM_K5_uniform.png) · [Editable SVG](svg/09_FeelWM_K5_uniform.svg)

Same baseline architecture; this experiment changes the weighting of the five supervised prediction times

Source: compute_losses + resolved_config.json horizon_lambdas; source_snapshots/feelwm_notebook.py
Status: Completed experiment / result artifacts; source trajectories: 28.

### 10_FeelWM_K5_mild_decay: FeelWM K=5 — mild decay loss experiment

[PNG image](png/10_FeelWM_K5_mild_decay.png) · [Editable SVG](svg/10_FeelWM_K5_mild_decay.svg)

Same baseline architecture; this experiment changes the weighting of the five supervised prediction times

Source: compute_losses + resolved_config.json horizon_lambdas; source_snapshots/feelwm_notebook.py
Status: Completed experiment / result artifacts; source trajectories: 28.

### 11_FeelWM_K5_strong_decay: FeelWM K=5 — strong decay loss experiment

[PNG image](png/11_FeelWM_K5_strong_decay.png) · [Editable SVG](svg/11_FeelWM_K5_strong_decay.svg)

Same baseline architecture; this experiment changes the weighting of the five supervised prediction times

Source: compute_losses + resolved_config.json horizon_lambdas; source_snapshots/feelwm_notebook.py
Status: Completed experiment / result artifacts; source trajectories: 28.

### 12_V7Legs_V7B_base: V7Legs + geometry + V7B — base

[PNG image](png/12_V7Legs_V7B_base.png) · [Editable SVG](svg/12_V7Legs_V7B_base.svg)

Architecture of the exact base / leg-only / body-only / hybrid workbook comparison

Source: Hybrid.forward; LegHead; BodyHead; objective; evaluate.py — v7legs_body
Status: Completed experiment / result artifacts; source trajectories: 28.

### 13_V7Legs_V7B_leg_only: V7Legs + geometry + V7B — leg only

[PNG image](png/13_V7Legs_V7B_leg_only.png) · [Editable SVG](svg/13_V7Legs_V7B_leg_only.svg)

Architecture of the exact base / leg-only / body-only / hybrid workbook comparison

Source: Hybrid.forward; LegHead; BodyHead; objective; evaluate.py — v7legs_body
Status: Completed experiment / result artifacts; source trajectories: 28.

### 14_V7Legs_V7B_body_only: V7Legs + geometry + V7B — body only

[PNG image](png/14_V7Legs_V7B_body_only.png) · [Editable SVG](svg/14_V7Legs_V7B_body_only.svg)

Architecture of the exact base / leg-only / body-only / hybrid workbook comparison

Source: Hybrid.forward; LegHead; BodyHead; objective; evaluate.py — v7legs_body
Status: Completed experiment / result artifacts; source trajectories: 28.

### 15_V7Legs_V7B_hybrid: V7Legs + geometry + V7B — hybrid

[PNG image](png/15_V7Legs_V7B_hybrid.png) · [Editable SVG](svg/15_V7Legs_V7B_hybrid.svg)

Architecture of the exact base / leg-only / body-only / hybrid workbook comparison

Source: Hybrid.forward; LegHead; BodyHead; objective; evaluate.py — v7legs_body
Status: Completed experiment / result artifacts; source trajectories: 28.

### 16_V7Feedback_seed: Leg → body → force refinement — seed

[PNG image](png/16_V7Feedback_seed.png) · [Editable SVG](svg/16_V7Feedback_seed.svg)

Same future target time is revised; seed, one-pass and recurrent checkpoints are distinct comparison routes

Source: FeedbackModel.forward; ForceRefiner; feedback_objective — v7feedback/feedback_models.py
Status: Completed experiment / result artifacts; source trajectories: 28.

### 17_V7Feedback_one_pass: Leg → body → force refinement — one pass

[PNG image](png/17_V7Feedback_one_pass.png) · [Editable SVG](svg/17_V7Feedback_one_pass.svg)

Same future target time is revised; seed, one-pass and recurrent checkpoints are distinct comparison routes

Source: FeedbackModel.forward; ForceRefiner; feedback_objective — v7feedback/feedback_models.py
Status: Completed experiment / result artifacts; source trajectories: 28.

### 18_V7Feedback_round1: Leg → body → force refinement — round1

[PNG image](png/18_V7Feedback_round1.png) · [Editable SVG](svg/18_V7Feedback_round1.svg)

Same future target time is revised; seed, one-pass and recurrent checkpoints are distinct comparison routes

Source: FeedbackModel.forward; ForceRefiner; feedback_objective — v7feedback/feedback_models.py
Status: Completed experiment / result artifacts; source trajectories: 28.

### 19_V7Feedback_round2: Leg → body → force refinement — round2

[PNG image](png/19_V7Feedback_round2.png) · [Editable SVG](svg/19_V7Feedback_round2.svg)

Same future target time is revised; seed, one-pass and recurrent checkpoints are distinct comparison routes

Source: FeedbackModel.forward; ForceRefiner; feedback_objective — v7feedback/feedback_models.py
Status: Completed experiment / result artifacts; source trajectories: 28.

### 20_V7Feedback_round3: Leg → body → force refinement — round3

[PNG image](png/20_V7Feedback_round3.png) · [Editable SVG](svg/20_V7Feedback_round3.svg)

Same future target time is revised; seed, one-pass and recurrent checkpoints are distinct comparison routes

Source: FeedbackModel.forward; ForceRefiner; feedback_objective — v7feedback/feedback_models.py
Status: Completed experiment / result artifacts; source trajectories: 28.

### 21_V8A_causal_terrain_memory: V8-A — causal terrain memory

[PNG image](png/21_V8A_causal_terrain_memory.png) · [Editable SVG](svg/21_V8A_causal_terrain_memory.svg)

Built spatial memory module from four diagnostic recordings; no optimizer-trained prediction head

Source: V8 classes / corresponding functions — source_snapshots/v7_rtx_notebook.py; verified RTX storage/V8_contactmap artifacts
Status: Completed experiment / result artifacts; source trajectories: 4.

### 22_V8B_touchdown: V8-B — contact and future foot-location Transformer

[PNG image](png/22_V8B_touchdown.png) · [Editable SVG](svg/22_V8B_touchdown.svg)

Shared per-leg history encoder predicts contact probability and a Gaussian future foot position

Source: V8 classes / corresponding functions — source_snapshots/v7_rtx_notebook.py; verified RTX storage/V8_contactmap artifacts
Status: Completed experiment / result artifacts; source trajectories: 4.

### 23_V8C_contact_field: V8-C — uncertain contact load field with four experts

[PNG image](png/23_V8C_contact_field.png) · [Editable SVG](svg/23_V8C_contact_field.svg)

Candidate terrain-cell descriptor + current robot state → joint torque and foot-force distribution

Source: V8 classes / corresponding functions — source_snapshots/v7_rtx_notebook.py; verified RTX storage/V8_contactmap artifacts
Status: Completed experiment / result artifacts; source trajectories: 4.

### 24_V8D_terrain_delta: V8-D — paired-terrain response-change model

[PNG image](png/24_V8D_terrain_delta.png) · [Editable SVG](svg/24_V8D_terrain_delta.svg)

Matched current-state samples across terrains → predicted change in torque and force

Source: V8 classes / corresponding functions — source_snapshots/v7_rtx_notebook.py; verified RTX storage/V8_contactmap artifacts
Status: Completed experiment / result artifacts; source trajectories: 4.

### 25_V8E_microplanner: V8-E — candidate foothold microplanner

[PNG image](png/25_V8E_microplanner.png) · [Editable SVG](svg/25_V8E_microplanner.svg)

Shadow planner combining causal terrain memory, future foot prediction and uncertain response estimates

Source: V8 classes / corresponding functions — source_snapshots/v7_rtx_notebook.py; verified RTX storage/V8_contactmap artifacts
Status: Completed experiment / result artifacts; source trajectories: 4.

### 26_V8F_sparse_adaptation: V8-F — causal sparse residual adaptation

[PNG image](png/26_V8F_sparse_adaptation.png) · [Editable SVG](svg/26_V8F_sparse_adaptation.svg)

Online memory of completed prediction errors corrects later local torque/force estimates

Source: V8 classes / corresponding functions — source_snapshots/v7_rtx_notebook.py; verified RTX storage/V8_contactmap artifacts
Status: Completed experiment / result artifacts; source trajectories: 4.

### 27_V7_Joint_Specialists_full: V7 Joint Specialists — implemented end-to-end architecture

[PNG image](png/27_V7_Joint_Specialists_full.png) · [Editable SVG](svg/27_V7_Joint_Specialists_full.svg)

Latest 28-trajectory experiment from the 8/28 and 16/28 preparation updates; source verified against server code

Source: JointSpecialists; JointSpecialist; SequenceEncoder; ForceContactSpecialist; stage_loss — v7_joint_specialists
Status: Implemented; optimizer training not verified / preparation blocked; source trajectories: 28.

### 28_V7_Joint_Specialists_force_detail: V7 Joint Specialists — force/contact stage enlarged

[PNG image](png/28_V7_Joint_Specialists_force_detail.png) · [Editable SVG](svg/28_V7_Joint_Specialists_force_detail.svg)

Layer-by-layer detail of the final force extension; frozen joint/body boundary and exact residual reconstruction

Source: force_models.py:ForceExtension/ForceContactSpecialist; joint_objectives.py:stage_loss
Status: Implemented; optimizer training not verified / preparation blocked; source trajectories: 28.

## Provenance and limits

- V7, Perceiver, visual fusion, causal history and V7-F specialist class definitions are AST-identical between the original four-trajectory notebook and the inspected expanded RTX notebook.
- Server SHA256s match the six local leg/body, feedback and Joint Specialists model/objective/command sources inspected for this atlas.
- V7Legs/V7B and V7Feedback results/protocols contain the displayed four and five comparison variants.
- V8-A through V8-F artifacts were found on the RTX server; B/C/D have trained model files. A is constructed memory; E is a shadow planner; F is causal sparse residual adaptation.
- Original pilot/10-epoch FeelWM repeats and horizon-loss calibrations share the pictured topology. Configured epoch counts are not presented as completed training counts.
- The current Joint Specialists experiment is the one formerly at8/28 and16/28 preparation. Fresh inspection found no matching process, no checkpoint, and a saved evaluation-input-preparation0/4 status.
- The old image used width768 for10 heads; actual FeelWM world width is800. The current scalar specialists each copy a trained256-wide leg Transformer and add two64-wide causal encoders.
- Independent one-pass and feedback round1 use different checkpoints and computational paths. Validation selected feedback SHORT round1, LONG round2; this alone is not a claim of unseen-test superiority.
- Scientific diagrams are deterministic renders of code-derived specifications; this avoids generated text introducing errors. The unverified image-generation drafts are excluded; use the exact PNG/SVG files for presentation.