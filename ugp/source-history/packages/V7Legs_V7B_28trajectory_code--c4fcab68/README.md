# V7Legs + geometry + V7B, 28 trajectories

This adds residual leg and body specialists to the completed RTX V7/V7-F foundation. No visual backbone is retrained. Training reuses the original frozen latent caches, original joint order, per-run baseline/residual reconstruction, and force heads. Differentiable URDF kinematics supplies foot position, velocity, and the soft planted-foot consistency loss.

Local source is this directory. Remote deployment is `/path/to/ugp/code`; artifacts are in its sibling `results` directory. The previous run is read as reference material.

Run on the RTX host: `bash run_training.sh`. The launcher selects physical GPU 1. It resumes incomplete stages automatically after validating provenance and acquiring the run lock. It does not stop or modify any other job.

Progress: `results/status.json`, `results/metrics.jsonl`, and the top-level `training.log`. Checkpoints: `results/checkpoints/{short,long}/{legs,body,joint,body_only}_{latest,best}.pt`.

Sequence per short/long expert: leg specialist, body specialist with legs fixed, joint refinement, and body-only comparison. Separate stages use up to 30 epochs; joint refinement up to 15. Validation early stopping after five unimproved epochs selects the saved models. After training, 96 evaluation cases cover 16 familiar anchors, 64 gap cases, and 16 strictly unseen anchors. Exact joint, force, and body CSVs and one Excel workbook per comparison variant are produced under `results/evaluation`.

The 28 trajectory identities and four strict holdouts exactly match the source notebook. New-head validation excludes a one-second gap and uses no overlapping target timestamps. Its frozen foundation was trained previously on the original eligible ranges, so this validation is a head-selection diagnostic. True generalization is evaluated on the disjoint trajectory holdouts. The existing offline protocol assumes the recorded future controller commands are known. Never describe it as forecasting with unknown future controls.

New correction scales use optimization windows only. Global V7/V7-F normalization stays as in the saved source caches. Body-frame evidence and checkpoint provenance are recorded in the manifest. Percentage errors are informational and undefined for zero actual values; judge angles in degrees and body signals in their physical units.

The focused material verification is `verify.py`: differentiable FK, Jacobians, zero corrections, real target reconstruction, and body-to-leg gradient flow. It is not a repeated collection of trivial module smoke tests.
