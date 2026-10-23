# V7 terrain-grounded 3D world model — runnable implementation

This package implements the agreed **V7 terrain-first future world-model dependency**. It is **not** a trained checkpoint, nor a claim that the user’s remote VS Code files were changed. It was built from the V7/V7-F design discussion and an archived V7F notebook. The archived notebook was not overwritten.

## One forward path, not a blind proprioception model

1. Five calibrated RGB-D observations are fused into a **world-registered 8×8 (or general P-cell) grid**. Each patch retains a persistent 64-bit WORLD cell ID; temporal Transformer attention tracks its five observation histories. Spatial Transformer attention relates patches, including geometric coordinates. `grid_from_rgbd()` implements reprojection from calibrated RGB-D; use a **fixed world-anchored grid origin and physical cell size** for both input and targets.
2. Per-leg touchdown memories retain **each** past world-space contact XYZ, persistent patch ID, time elapsed, RGB, geometry, measured pre/post visual and proprioception changes and physical response. Events remain separate, and leg queries cross-attend to all observed events. Current terrain features are linked by exact patch ID when present; prior RGB/geometry stay available after a patch leaves view.
3. Candidate **future actions known at the observation time** drive predicted base egomotion, while a separate **contact-dependent residual field** evolves WORLD-frame 3D terrain and RGB. Each horizon combines a direct-from-t geometrical anchor with a predicted-contact-conditioned recursive update. Grid IDs remain attached to the same ground cells. Ego motion transforms that world model into the robot’s future viewpoint; camera motion is not mistaken for physical deformation.
4. Four per-leg action-conditioned FiLM units modulate the foot queries at each requested horizon; leg queries then attend to the **predicted future** terrain, with reachable-patch masking and past contact memory. Heads produce patch classification, 3D touchdown, contact state, 3D force and uncertainty. Predicted contacts condition the **next** terrain update. Only those predicted interactions and terrain features reach the body-change head. The measured current body state is a residual origin, not a blind future predictor.
5. Losses explicitly require measured FUTURE 3D geometry, RGB, foot positions, landing patch IDs, forces, contacts, base pose, and body states, plus direct-vs-recursive consistency. Optional calibrated sparse pinhole reprojection adds RGB-D pixel-coordinate/depth consistency where measured targets exist. No target is an input to `forward()`. Outputs incorporate projected **future depth** from evolving geometry, not the last camera depth.

The 10 attention heads, 8 spatial Transformer layers, and 7:1 feed-forward expansion default to the V7 settings agreed in conversation. Temporal-patch attention uses 2 additional layers. You may lower dimensions for a CPU sanity check; do not present such a small model as the full trained V7.

## Get it running in remote VS Code

Requires Python 3.10+; PyTorch 2.x and NumPy.

```bash
pip install -r requirements.txt
python -m tests.test_model
python train.py --train /absolute/path/train_real.npz --val /absolute/path/heldout_real.npz --out checkpoints --epochs 20 --batch 2
python predict.py --ckpt checkpoints/best.pt --observations /absolute/path/current_only.npz --output /absolute/path/predictions.npz
```

The bundled tests use **synthetic tensors only to test computation and gradients**; they cannot establish robotic prediction quality. Training explicitly refuses datasets missing measured future labels and cannot begin until your source logs have been synchronized and converted to the NPZ contract below. Your existing V7F notebooks may supply data utilities, but some of their earlier runs did not have verified low-level action, force frame and sensor calibration. Check these before training.

## Required `.npz` schema (each array starts with sample dimension N)

| Input key | Shape | Meaning |
| --- | --- | --- |
| `rgbd` | `[N,5,P,4]` | RGB `[0,1]` and depth (m), per world patch for 5 causal frames |
| `xyz_world` | `[N,5,P,3]` | calibrated WORLD XYZ in meters |
| `valid`, `patch_ids` | `[N,5,P]` | observed bool, persistent int64 IDs (same grid across all 5 frames) |
| `proprio` | `[N,5,D]` | calibrated body/joint state history, consistent units |
| `actions` | `[N,H,A]` | candidate future control plan, available at t; **NOT recorded future actions unless used solely in teacher-forced evaluation** |
| `base_pose` | `[N,6]` | current WORLD pose: xyz m + roll/pitch/yaw rad |
| `foot_xyz_world`, `foot_velocity_world` | `[N,4,3]` | current foot position and velocity in WORLD coordinates |
| `foot_contact`, `gait_phase` | `[N,4]` | measured contact `[0,1]`, normalized phase `[0,1]` |
| `history_xyz_world` | `[N,4,K,3]` | each actual historical touchdown, WORLD XYZ |
| `history_patch_ids` | `[N,4,K]` | world terrain patch touched, from touchdown frame registration |
| `history_rgb` | `[N,4,K,3]` | ACTUAL RGB of same historical terrain patch |
| `history_geometry` | `[N,4,K,4]` | real registered patch features, e.g. normal xyz + roughness |
| `history_response` | `[N,4,K,6]` | three-axis contact force (N), then three other measured response channels with documented units |
| `history_delta_rgbd` | `[N,4,K,4]` | measured RGB-D patch change **after each historical touchdown** (observed before t) |
| `history_delta_xyz` | `[N,4,K,3]` | measured 3D patch deformation after historical touchdown |
| `history_delta_proprio` | `[N,4,K,D]` | measured local proprioceptive response change caused by that touchdown |
| `history_age`, `history_valid` | `[N,4,K]` | time elapsed (seconds), actual event mask |

Optional jointly required foundation-model feature pair: `patch_features [N,5,P,C]` consists of **world-aligned** DINO/ConvNeXt/PTv3/flow patch encodings (or other calibrated visual tokens); `history_token_delta [N,4,K,C]` contains corresponding before/after visual encoding changes for every past foothold. The loader refuses the pair if one is missing; if supplied, the model learns their input projections and DOES NOT silently omit them. The pretrained backbone encoders themselves are not included.

Required **measured** future targets, all horizon-local timestamps and physical frames aligned:

| Target key | Shape |
| --- | --- |
| `future_xyz_world` | `[N,H,P,3]` |
| `future_rgb` | `[N,H,P,3]` |
| `future_patch_valid` | `[N,H,P]` |
| `future_foot_xyz_world` | `[N,H,4,3]` |
| `future_patch_ids` | `[N,H,4]` |
| `future_force` | `[N,H,4,3]` |
| `future_contact` | `[N,H,4]` |
| `future_body` | `[N,H,D]` |
| `future_base_pose` | `[N,H,6]` |

Optional: `intrinsics [N,3,3]`, `camera_from_base [N,4,4]`, and future measured `future_uvd [N,H,P,3]` for calibrated sparse reprojection. These are calibration constants/actual target labels, **not** future model inputs. The supplied calibrated grid builder supports an **XY height-field** with one dominant surface per grid cell, not general overhangs/occluded multi-layer 3D structures; it does not perform neural mesh reconstruction or full differentiable rasterization.

### Scientific and deployment limits

- The model is an **implementation and tested computation graph**, not trained robot policy. No force/error/foothold accuracy is claimed until evaluation on strictly held-out *trajectories*.
- Without depth-camera calibration and WORLD pose, persistent world-cell registration is not trustworthy. Without measured foot contact force or truly observed future geometry, the corresponding loss cannot be trained honestly. Do not substitute predictions for measurements.
- Patch IDs identify world XY cells, not deformable material particles; large sliding objects need explicit tracked surfels/scene flow. Missing or occluded surface is masked, not reconstructed as verified ground truth.
- The categorical landing head predicts an *observed/likely* landing patch; it is **not** an optimal safety policy or a validated counterfactual safety score for every patch. Model predicts future footholds and contact response, but it is not a certified safe foothold controller. This is a prediction module, not robot-actuator firmware.
- A future action plan is needed to predict action-conditioned dynamics. If only `a_t` is known, use a controller-supplied candidate action plan or reduce the horizon. Do not leak the *actual* future commands during inference.
- Long-horizon rollout is approximate: direct-vs-recursive consistency reduces but cannot eliminate compounding error. Contact response and terrain deformation remain data-limited.
- Full model inference requires precomputed aligned patch RGB-D and measured contact memory, which must be supplied by the user's preprocessing pipeline or recorded bag extraction.

## Integration with original V7F

The archived V7/V7-F model had a direct proprioception/force head. **Do not silently append this model while continuing to train/use that head as the primary prediction path**. Route the robot's synchronized world-patch features, actual foothold history and proposed actions to `V7GroundedWorldModel`, and train/infer using its `body`, `force`, `foot_xyz_world`, `predicted_patch_ids` outputs. The old direct head may be retained as a *separately labeled blind baseline* but must not be averaged into predictions and must not drive the robot's terrain-conditioned dynamics. This repo does **not** change the remote code or claim to replace dataset-specific encoders (DINOv3, ConvNeXt, PTv3, WAFT); their calibrated output must be adapted to this world-registered patch contract. Spatial-temporal RGB-D reconstruction is explicit; if you want to preserve high-dimensional pretrained DINO/Point Transformer features, add them to a patch-feature adapter with the same IDs, rather than erasing their spatial identities.

The inference script reads **only the input keys** (plus optional calibrated projection matrices and registered foundation features), and rejects missing causal inputs. Do not put observed future data into any of its input fields. A real trained checkpoint is needed for meaningful inference.
