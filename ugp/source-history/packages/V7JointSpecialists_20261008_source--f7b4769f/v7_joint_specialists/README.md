# V7 joint specialists

This experiment uses the original **28 training trajectories**. It preserves their temporal training/validation masks and four excluded development trajectories.

## Model

For each short/long branch there are nine independent warm-started specialists: angle, speed and measured torque for HAA, HFE and KFE. Each specialist handles all four legs using leg identity. It copies the trained leg projection, two-layer 256-wide Transformer and the appropriate scalar MLP output. Two-layer 64-wide causal encoders add observed history and the known planned command prefix through initially zero adapters after the copied temporal block.

Torque uses detached predicted angles/speeds, seed torque, legacy predicted force and a commanded-torque candidate. Measured effort is the torque target; commanded torque is a separate input. Future measured actuator state is forbidden. Only mode-valid commanded fields are active. Future controls are assumed known in this offline benchmark. Active position/velocity/gains and raw sub-grid command timing are an explicit input expansion.

The force/contact extension starts from the existing one-pass force prediction and trained force-processing layers. It adds an unbounded linear correction and a contact-logit correction. Nine leg representations are initially averaged into its existing 256-value interface. Body predictions remain at the fixed parent seed output, and all joints are frozen during force training. No feedback loop is used.

## Schedule

Per branch: small torque-output-only control; three torque specialists; three angle specialists; three speed specialists; three torque retuning stages; force/contact. Separate stages have caps of 30 epochs, patience five and three absolute-loss warmup epochs. Retuning has a 15-epoch cap. AdamW learning rates are 3e-4 / 5e-5, weight decay 1e-4, gradient norm limit 1. Seed 2781; batches 16 short and 8 long. Training balances trajectories and adds optimization-only transition/hard examples.

Normalized SmoothL1 is supplemented after warmup by 0.2 times mean tolerance violation and 0.5 times the worst 10% violation, where violation is log(1+max(error/tolerance-1,0)). Angles retain direct joint supervision plus 0.2 geometry supervision. Force has component/vector losses; contact has BCE and tolerance supervision. Each head selects its checkpoint on temporal validation by failure count, worst tolerance ratio, then physical MAE.

## Scoring

Every nonzero prediction must have error **strictly less than 2%** of the absolute actual value. Exactly 2% fails. Effective zero means abs(actual) <= 1e-8 in the reported unit. Approved zero limits are .05 degrees, .01 rad/s, .05 Nm and 1 N. Contact probability error must be below .02. Force components and vector error are both checked. Foot positions use the fixed URDF base origin and actual position norm; zero norm requires zero displacement.

This is an experimental accuracy target. Training completion does not establish it. Every failure remains in the exact exports.

## Operation

Set CUDA_VISIBLE_DEVICES to an available GPU after checking process owners, then run `bash run_training.sh`. The launcher acquires an exclusive run lock. Source/parent/cache/scaler signatures must match on resume. Use `--audit-only` for ROS extraction, `--verify-only` for aligned real batches in both branches, or `--train-only` to omit final evaluation. No new packages or vision training are required.

The pipeline writes readable `results/status.json` and epoch `metrics.jsonl`, per-head resumable checkpoints, and then automatically evaluates the original 96 cases (16 familiar, 64 time-gap, 16 excluded-trajectory). Original V7 uses four local epochs and V7-F five; new specialists have zero local epochs. Seven exact workbooks are produced: base_v7, base_v7f, seed, torque_control, torque_only, joint_specialists and full. Each includes Joint exact, Body exact, Force exact, Strict summary, Failures and How to read. All required 57,600 full-model points must exist with zero failures before `goal_achieved` can be true.

The source snapshot and hashes replace drive-root Git commits. Parent files and completed results are not edited. Local `.verification_deps` and optimizer archives are excluded from delivery.

Missing raw ROS bags are retrieved individually from the public TartanGround repository, using its existing archive naming convention. Only the requested bag is extracted into a private temporary folder, converted into a hashed compact command cache, then the temporary archive and bag are removed. Existing parent recordings are reused and preserved. Resume verifies compact-cache hashes instead of redownloading recordings.
