# V7 joint specialists: torque, angle, speed and force

Date: 2026-10-08. Status: architecture and proposed scoring limits approved by the user; implementation plan in preparation. Not implemented or trained.

## 1. User requirement

Improve all four legs, with dedicated specialists where useful. The user explicitly selected **every individual prediction below 2%**, with zero torque handled separately. A pooled average below 2% does not meet this requirement. Each leg, joint, physical quantity and requested future time must be checked.

The starting experiment remains the completed 28-training-trajectory V7 RTX foundation, its V7-F force specialist, and the existing V7Legs/V7B corrections. Preserve the exact training identities, eligible time ranges and four excluded trajectories. Reuse the code and frozen features; do not rebuild vision models.

This is an experimental target, not a promised outcome. Passing a finite set of tests would establish performance on those points, not a guarantee on arbitrary future robot motion.

## 2. Evidence that determines the design

- Existing `v7legs_body/models.py:68-89` has one shared leg Transformer and three output MLPs: angles, speeds and torques. Each MLP emits three joints. Simply adding a torque output layer would repeat part of the current architecture.
- Current round 2 on the matched unseen tests: 648/1920 angle predictions (33.75%), 127/1920 speed predictions (6.615%), 255/1917 nonzero-torque predictions (13.302%), and 14/339 nonzero-force-vector predictions (4.130%) have error below 2%.
- Its maximum absolute errors are 35.3215 degrees, 7.6027 rad/s, 42.5063 N m and 1140.9019 N respectively. These tails must remain visible in evaluation.
- A read-only audit of four training trajectories compared commanded torque with joint-state effort. A delay was chosen using the first 60% of each raw recording and examined on the next 20%. Mean absolute differences on those later sections were 2.5005, 1.0622, 1.4995 and 0.9669 N m. Only approximately 14.48-35.66% of nonzero samples met 2%. These are exploratory diagnostics, not official held-out results or model training. Do not use these raw-recording sections to expand the approved optimization ranges.
- Measured torque is `/state_estimator/joint_states.effort`; commanded torque is a distinct actuator-command field.
- Raw actuator commands contain mode, current, position, velocity, joint_torque and PID gain fields. One inspected message used torque mode 10. The population of modes and active gain fields has not yet been audited across all 28 trajectories.
- Existing future control features retain the last value and mean of U/L over the allowed interval. They do not expose the full command sequence. Loss of transition detail is a hypothesis to test, not an established cause of all errors.

## 3. Alternatives and recommendation

1. **Small independent output layers on the current shared leg Transformer.** Cheapest control experiment, but preserves the shared representation and compressed command inputs.
2. **Dedicated joint specialists with raw history and complete valid command sequences. Recommended.** Reuses the frozen foundation and pretrained corrections while separating the trainable processing of angle, speed and torque for each joint type.
3. **A large independent model for every leg/joint/quantity.** Greater separation, but substantially more compute and less sharing across the 28 trajectories. Do not start with this design.

## 4. Specialist architecture

Use separate short and long instances, preserving +0.1 s short and +0.2 through +1.0 s long.

| Family | Specialists | Outputs per future time |
|---|---|---|
| Angle | V7-Q-HAA, V7-Q-HFE, V7-Q-KFE | One angle per leg per specialist: 12 angles total |
| Speed | V7-D-HAA, V7-D-HFE, V7-D-KFE | One angular speed per leg per specialist: 12 speeds total |
| Torque | V7-T-HAA, V7-T-HFE, V7-T-KFE | One torque per leg per specialist: 12 torques total |
| Force/contact | Reused force refiner adapted to the corrected legs, with explicit Fx/Fy/Fz outputs and a contact classifier | Four 3D force vectors and four contact probabilities |
| Foot geometry | Existing verified TorchFK | Four body-relative foot positions and Jacobians |

Each joint specialist handles FL, FR, RL and RR with leg identity and the verified joint mapping. This produces individual joint predictions without training 36 unrelated large networks.

Each of the nine joint specialists gets its own trainable copy of the existing projection and two-layer 256-wide leg Transformer. Initialize its scalar output from the corresponding row of the pretrained angle/speed/torque output MLP. Preserve the old baseline/residual reconstruction. All expensive V7 features remain frozen.

Add two small encoders to each specialist:

- A causal history encoder for observed joint angles, speeds, effort, foot forces, command tracking errors, contact and valid actuator commands. Use the existing 0.5 s / 4 s history lengths.
- A masked command-sequence encoder for the known future control plan within the same interval available to the original benchmark. Use 64-wide, two-layer attention blocks and explicit time and mode encodings. A horizon query reads the applicable prefix of that sequence. Do not substitute raw endpoint/mean pooling for this encoder.

Zero-initialize the adapters that add these encoded features to the pretrained specialist. Verify that initialization reproduces the parent prediction before optimization. Joint specialists have independent optimizer states and validation records.

## 5. Control information and timing

Store compact per-trajectory raw history/command arrays and per-record anchor/target indices beside the existing latent records. Slice windows at batching time instead of duplicating overlapping histories in every record. Match exact source timestamps. Reuse canonical leg/joint names and verify actuator order on optimization data and message definitions. Do not infer a new mapping from future held-out observations.

Mode masks are mandatory: torque mode 10 activates torque command; position/velocity/torque modes 12 and 13 activate their applicable fields; supplied gains are used only where their command mode makes them active. An inactive position field must not become a future-angle hint. Preserve unexpected modes in the audit and handle them explicitly; do not discard difficult samples to improve the score.

Audit source timestamp offsets, duplicate messages, causal holds, torque signs, command/effort differences, mode transitions and extreme labels. Estimate any command-response delay, gain or offset only inside optimization ranges. Do not relabel or remove spikes merely because the model predicts them badly.

The existing benchmark assumes recorded future U/L are known planned controls. This proposal keeps that conditional-dynamics assumption and does not claim prediction with unknown future controls. Adding active position/velocity/gain fields is an input expansion and must be reported as such. Run the small specialist control with original control inputs to separate input-information gains from architecture gains.

No actual future angle, speed, effort, force, contact or body state may enter a prediction. They are loss targets only. Future actuator **state** fields are forbidden inputs. Default control prefixes end before the target timestamp, matching the original sampling interval. A target-time command extension would be a separately labeled protocol change.

## 6. V7-T details

V7-T learns measured actuator response from command history, valid planned commands, observed tracking errors, V7 terrain/body context and predicted leg state. Include a command-derived torque candidate as a feature and compare it with the existing torque prior. The residual model learns tracking effects and transitions.

Do not equate joint_torque with measured effort. Do not use `J^T F` as a complete actuator-torque formula: inertia, gravity, controller response and coordinate conventions matter. Mode/gain data can support a controller-response feature after its semantics are verified; they do not establish a complete inverse-dynamics model.

During torque training, freeze the angle, speed, force and body predictors. Future leg-state features must be predicted values, not ground truth. This isolates torque changes and prevents a torque objective from worsening existing angle predictions.

## 7. Geometry, force and body

Calculate foot positions using predicted HAA/HFE/KFE through the existing URDF-based TorchFK. Supervise against FK of actual angles at the identical target time. Retain direct per-joint angle losses so a good foot location cannot conceal a wrong joint angle.

After angle/speed/torque specialists are trained, adapt the existing force refiner using their predicted states, geometry, histories, valid commands and frozen body/context features. Project the nine specialist representations to the force model's existing leg-token interface. Train force and contact losses while joint specialists stay frozen.

Audit whether the current two-standard-deviation force-correction bounds prevent fitting valid large forces. The proposed force output uses a linear residual rather than an unvalidated hard correction cap; training scales and gradient clipping provide numerical safeguards. Do not clip predictions using actual future force.

Keep body outputs at the fixed parent seed predictions during this first leg-focused experiment. No recurrent body/force/leg loop is introduced. A later body experiment can consume the improved legs after their accuracy has been measured.

## 8. Strict scoring and zero-value limits

For each nonzero scalar target y, require `abs(prediction - y) < 0.02 * abs(y)`. Apply the existing effective-zero convention `abs(y) <= 1e-8` in the reported unit (degrees for angles, rad/s for speeds, N m for torque, N for force). Small nonzero targets retain the full 2% requirement; do not widen the zero mask to make them pass.

Proposed exact-zero limits for review:

| Quantity | Absolute error limit at an effective-zero target |
|---|---|
| Angle | 0.05 degrees |
| Angular speed | 0.01 rad/s |
| Joint torque | 0.05 N m |
| Force component / zero force vector | 1 N |

These are proposed scoring tolerances, not verified sensor-accuracy claims. Record approved limits before training and leave them fixed.

Score force components individually and also `norm(predicted F - actual F) / norm(actual F)` for nonzero force vectors. Report foot-position error in metres and its ratio to the actual body-relative foot-position norm, with the URDF base origin fixed. Contact is a binary target: report every classification error and propose `abs(predicted probability - actual label) < 0.02` as its strict probability target. These additional definitions require review alongside the zero limits.

Report the number and fraction of failures, maximum error, 95th/99th percentile, mean error and worst exact cases for every leg, joint/channel and horizon. Acceptance requires zero failures in every required group. One failing prediction means the strict target is not achieved, even if the mean is excellent.

## 9. Training objective

Start with the existing training-scale-normalized SmoothL1 supervision, fitting loss scales separately for each leg/joint from optimization records. Retain the parent's original scales for reconstruction of copied pretrained corrections. Add tolerance-violation and tail emphasis. For error e and the applicable acceptance tolerance T, define `b = log(1 + max(e/T - 1, 0))`. Train with `L_abs + 0.2*mean(b) + 0.5*mean(top 10% b)` as initial, predeclared weights. Use weak geometry supervision for angle specialists and supervised contact/force losses for the force stage. Revisit weights using validation only.

Warm up absolute-error learning before the tolerance term. Sample trajectories/terrains evenly and include training-only examples around mode, contact and command transitions. Hard-example sampling uses optimization records only and keeps ordinary samples in each batch.

Use explicit physical-unit reconstruction, float32 low-level inputs, and float64 diagnostic ratios. Avoid float16 final low-level outputs, percentage arithmetic and residual subtraction near zero. Clip optimizer gradient norms and reject nonfinite losses. Preserve the uncapped original percentage metric in evaluation regardless of optimization safeguards.

## 10. Stages and controlled comparison

1. Audit all 28 approved training trajectories and augment caches with verified command/history information. Compare command-only and existing predictions within eligible optimization/validation ranges.
2. Train the three V7-T specialists first with current predicted angles/speeds fixed. Compare pretrained seed, small output-layer control, and command-aware specialist on validation.
3. Train the three angle and three speed specialists independently; add geometry supervision to angle training.
4. Retune torque specialists using the improved predicted angles/speeds while those predictors are frozen.
5. Adapt force/contact using the corrected legs, keeping joints and body fixed.
6. Lock weights and evaluate the original 96 familiar/gap/unseen cases with the same correction-adaptation budget for all comparison variants. Keep the unseen trajectories outside optimization, calibration, normalization and checkpoint selection.

Reuse the original 28 identities, four excluded trajectories, cache timestamps, optimizers/checkpoint conventions, data readers, FK, evaluators and Excel exporters. Use resumable isolated output directories. Select each checkpoint on validation by tolerance-failure count, then worst relative error, then physical-unit mean error. Initial stage caps are 30 epochs, with patience five and a three-epoch absolute-loss warmup; small retuning stages have a 15-epoch cap. Reuse AdamW and the existing 3e-4 separate-head / 5e-5 retuning rates as defaults. Do not choose heads or rounds using unseen results.

The four existing excluded trajectories have already been inspected during architecture development. They remain excluded from fitting, but results on them are development-holdout evidence. A fresh, uninspected trajectory would be needed for a stronger final generalization claim.

## 11. Implementation checks and launch

Material checks: canonical joint/mode masks; exact timestamps and no future-state inputs; preservation of pretrained outputs at initialization; physical reconstruction; finite near-zero loss; real gradients into each specialist; isolation of frozen predictions; reproducible resume; strict failure counts and exact-value exports. Avoid repeated trivial smoke tests.

Training requires a reviewed specification and implementation plan. Recheck GPU owners and available memory immediately before a detached launch; never stop unrelated jobs. No duration estimate is supported until real cache construction and optimizer throughput are measured.

If 2% at every point is not achieved, report the failures directly. Consider more informative planned controls, raw timestamp resolution, extra trajectories or a shorter horizon based on evidence; do not change the requirement or hide failures.

## 12. Review decisions

The user approved the command-aware nine-specialist architecture and proposed scoring definitions on 2026-10-08. The next gate is review of the implementation plan and selection of execution method. The strict 2% result remains unproven.
