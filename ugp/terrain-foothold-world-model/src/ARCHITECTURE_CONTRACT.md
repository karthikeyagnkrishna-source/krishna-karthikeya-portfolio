# 3D Terrain–Foothold World Model: non-negotiable architecture contract

**This is a new, independent project. Do not call it V7 or reuse the old models, weights, checkpoints, datasets by assumption, or prior-chat architecture decisions.** The starting `terrain_foothold_world_model.py` is an **unvalidated research implementation**, not a certified complete system. Its previous synthetic smoke tests do not establish the following requirements are correctly implemented. Independently audit every claim.

## Causal computation graph (mandatory)

`five recent calibrated RGB-D frames + observed body/leg states + actual historical foothold-to-patch interactions + candidate action sequence`  
`→ ego-pose/3D observation registration → world-registered 3D surface reconstruction with temporal identities and confidences`  
`→ temporal attention along corresponding terrain elements + spatial attention among elements`  
`→ action-conditioned future scene evolution: relative-camera motion separated from actual contact-induced surface change`  
`→ future 3D geometry and appearance, patch identities and uncertainty at each requested horizon`  
`→ leg-conditioned cross-attention on that **predicted** future terrain + actual historical foot–terrain memory`  
`→ per-leg reachable landing patch, continuous 3D foothold, contact time, safety/contact probability, 3D force, slip/uncertainty`  
`→ dynamics and future proprioception conditioned on those predicted interactions and evolved terrain.`

No future body prediction may bypass evolved geometry. Do not incorrectly assume a nonzero gradient guarantees actual reliance; use counterfactual tests replacing geometry, patch appearance, and contact memory, plus held-out comparisons. For non-contact prediction, model relevant base dynamics without falsely forcing a foot impact.

## Observations and 3D geometry

- Input includes **five** temporally ordered views, depth, camera intrinsics, calibrated extrinsics, base pose/ego-estimate, validity and uncertainty. Register points into a world frame. Use an observed-surface representation (surfels/point cloud/voxels), not a single 8×8 height map.
- Preserve each terrain point's **world coordinates, relative camera coordinates, RGB/visual features, geometric features, original observation timestamp, observation frame, stable correspondence ID when reliable, and confidence**. Repeated observations of a place can be associated without pretending exact material identity is guaranteed. Keep unknown/occluded regions explicitly unknown.
- Independently implement patch-local temporal attention (tracked observations across time) and neighborhood spatial attention, plus leg-conditioned attention constrained by feasible leg reach. Avoid averaging all historical contacts into one vector.
- High-fidelity pretrained visual features (e.g. DINOv2 semantic, a separate texture branch, 3D geometry encoder) may be integrated if weights/installation are available; never label random projections or mock features as pretrained. Verify correspondence between RGB pixels, depth, and 3D patches. Visual appearance at future horizons must be predicted rather than observed at inference.

## Past foot–terrain interaction memory

For every **observed historical touchdown** of each of four legs, keep time, 3D contact position/velocity, correspondence to the **same actual surface patch seen at that time**, original patch geometry and appearance, foot contact state, measured force components if available, body/proprio response and subsequent measured geometry/RGB changes when observed. Distinguish measured, estimated and missing values with masks. A later hind foot must be able to retrieve a patch previously touched by a forefoot. Retain order, limb identity, event identity and confidence.

## Evolve the world before feet and dynamics

- Predict base egomotion under each **candidate action sequence known at current time**, transform static surfaces accordingly, and separately predict **physical terrain deformation** and appearance changes under anticipated foot–terrain interaction. Make the two causes distinguishable.
- Maintain persistent terrain identities when supported by tracking; allow birth/death/occlusion/confidence changes and avoid double counting. Roll out an explicit spatial 3D surface state for horizons around 0.1, 0.2, 0.5 and 1.0 seconds (configurable).
- Predict candidate patch suitability/safety, landing likelihood, touchdown time and continuous foothold location, then interaction force/slip and body/leg proprioceptive state from evolved 3D terrain. Condition predicted deformation on contacts and contact forces consistently; if a contact is predicted later, do not apply it before touchdown.
- Predicted footholds are **never ground truth**. At inference, only observations at or before present time and candidate future actions are allowed. Report uncertainty and periodic correction from actual new observations.

## Training and loss contracts

- Separate direct multi-horizon predictions from recursive rollouts and explicitly couple their future 3D states. Train with **true** future observed geometry/appearance, patch correspondences where measurable, observed foot contacts/positions, forces, and proprioception; use masks for missing labels.
- Ground future appearance with geometry reprojection/differentiable rendering only when calibrations/visibility support it; occlusion and unseen surfaces must be masked, not hallucinated as ground truth.
- Require verified action provenance (command vs actual), timestamp synchronization, coordinate conventions and transformations, camera calibration, measured contact/forces, frame-to-frame alignment and unit scaling. Missing sensor streams must result in an explicit limitation/error or a properly masked objective, not fabricated physical labels.
- Data loading must avoid target leakage. Training-time observed future actions can be used as a *known candidate future control sequence*, but during actual planning candidate actions must be supplied by the planner at present time.
- Train from scratch (except independently selected, disclosed pretrained visual encoders). Maintain untouched validation and terrain/trajectory-heldout testing, with per-leg/per-terrain force MAE, p95, vector direction; foothold patch top-k, 3D distance and timing; future 3D geometric error and reprojection; body/leg state errors; uncertainty calibration. No real metric should be claimed before testing on real measurements.

## Hardware / operations

- Target is the user's **remote RTX 4080 connected through VS Code Remote-SSH**. This is a machine location, not a reason to load older weights.
- The orchestration should run ON THE REMOTE HOST if command execution and GPU tests need that host; verify host and `nvidia-smi` from the same terminal.
- **Never** start long GPU training, spend substantial cloud/API budget, modify SSH/auth settings, delete old folders, or overwrite original data without specific user approval. Always keep a Git baseline and diff/review trail.
- Build and reviewer must be separately started agent sessions, exchange findings through local report files, and fail closed on unresolved blocker. Two-model agreement is not proof: human approval and real heldout measurements remain mandatory.
