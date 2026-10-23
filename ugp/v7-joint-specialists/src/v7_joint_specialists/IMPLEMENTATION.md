# V7 joint specialists execution ledger

Plan: `docs/superpowers/plans/2026-10-08-v7-joint-specialists.md`.

The user approved the architecture, strict per-point limits and direct execution.
All new work is isolated here. Parent code, checkpoints and `sources/` are references.

## Rulings

- Git discovery resolves to the Windows drive root C:/ with a different owner. No drive-root Git configuration, worktree or commit will be made. Use isolated source files, snapshots and hashes as allowed by the approved plan.
- Reuse the installed local verification torch and remote v7f4080 environment. No vision retraining or dependency installation.
- Only material scoring, timing, gradient isolation, resume and export checks are required. One independent implementation review before launch.

## Progress

- Task 1: RED missing strict scoring; GREEN 4/4. Strict boundary, tiny nonzero, degree conversion, vector/component distinction, zero failures and finite loss tested.
- Task 2: RED missing command handling; GREEN 3/3. Mode/mapping masks, inactive NaNs, duplicates, sub-grid prefix and sensor/target timing checked. Real 28-trajectory audit pending deployment.
- Task 3: RED missing specialists; GREEN 3/3. Parent initialization, independent weights, nontrivial prefix invariance, all-padding and selected-head gradients pass.
- Task 4: RED missing trainer; GREEN 2/2. Exact next optimizer step and changed-signature rejection pass. Windows cannot import Linux fcntl launcher; reuse its exact Reporter/RNG AST definitions locally. Linux imports the parent normally.
- Task 5: RED missing force extension; GREEN 1/1. Uncapped correction preserves joints/body with no frozen gradients.
- Task 6: RED missing evaluator; GREEN 2/2. Incomplete/duplicate cases and missing points cannot finalize; exact times/values and zero-case failures exported. Standalone V7 reconstructs its own force residual.
- Task 7: whole material suite 18/18 GREEN locally. Independent review found three Important issues; all fixed in one pass with failing regression tests first. No deferred minors.
- Final: fixed stale predecessor gradients — joint-to-force transition RED to GREEN.
- Final: fixed cross-horizon raw-command leakage in force/contact — nonzero-adapter/output prefix perturbation RED to GREEN; extension temporal attention is causal.
- Final: fixed duplicate completion protocol cases field — final protocol write RED to GREEN.
- Final suite: `python -m unittest discover -s v7_joint_specialists/tests -v`, 18/18 pass, 6.328 seconds. Real audit/verification and launch pending.
- Server suite: 18/18 pass, 2.489 seconds. Both GPUs had no compute PIDs. First real audit reconstructed ForestEnv/P2000 commands exactly, then found P2003 raw ROS bag absent.
- Added missing-recording recovery using public dataset archive conventions; RED missing module, isolated archive/reuse tests. Keep one temporary recording at a time because server storage is limited. The failed preflight output/source identity will be preserved before deploying corrected source.
- Recovery suite GREEN 20/20 locally (6.579 s), 20/20 on server (1.682 s). Original failed source/output preserved under code_preflight_v1 and results_preflight_v1.
- Server direct Internet remained blocked. Secure private CONNECT relay uses existing SSH with end-to-end certificate verification; real public archive range download returned HTTP 206 and 1024 bytes. Relay PID 25704 and SSH tunnel PID 28224 on this computer; private server port 18881.
- Operational route/launch gate tests RED to GREEN 2/2. Verification PID 17361 audits all 28 before checking both branches. Controller PID 17831 waits for matching successful verification, prefetches excluded input commands without fitting, rechecks compute PIDs/free memory and then starts the locked trainer. No unrelated job is stopped. Local computer must remain connected during recording preparation. Training has not yet been confirmed by an optimizer step/checkpoint.
- Live inspection found preparation stopped at 7/28 on GreatMarsh/P2000; no optimizer steps/checkpoints existed. Diagnostic source has 51,741 commands and 2,373 duplicate stamps. Stable original-axis causal hold matches every saved command exactly; rebasing timestamps changes three frames, up to 4.0891 Nm. Quicksort would introduce 133 errors and is not the fix.
- Timestamp regression RED (future command entered observed history) to GREEN: all holds and strict prefix comparisons now use original uncropped timestamps/grid. Relative time is only a numeric feature. Schema v2 binds original grid/crop in the physical digest. Preserve the previous failed run and verify every reused cache against the original grid before migration.
- Original-axis fix suite: 23/23 locally (6.818 s), 23/23 server (1.708 s). Real GreatMarsh/P2000 command reconstruction now passes exactly. Seven prior caches retained with old bytes/source preserved, restored original coordinates and exact verification against every saved original-grid command; migration lineage is stored in their metadata. Previous failed outputs/source preserved under results_preflight_v2 and code_preflight_v2.
- Restarted preflight PID 19283 and gated controller PID 19375. Fresh status: preparing GreatMarsh/P2005, 8/28 completed. GPUs: 0 at 0%, 92 MiB; 1 at 0%, 15 MiB; no compute PIDs. No optimizer step/checkpoint exists yet. Automatic training remains conditional on complete real-data verification.
- 2026-10-09 recovery: SSH restored. Server uptime was shorter than the last progress age, confirming a restart after preparation reached 19/28; prior process PIDs and local relay were gone, and no training checkpoints existed. Existing caches/signatures preserved. Restored private verified TLS route (relay PID 13580, tunnel PID 16792); public archive returned HTTP 206/1024 bytes. No matching model job existed before restart.
- Resumed preflight PID 25905 from all 19 verified caches. Initial controller correctly rejected the previous stale status PID during startup; after fresh status PID 25905 was confirmed, a new controller was attached using pipeline_recovered_20261009.log. Both GPUs idle (0: 240 MiB, 1: 15 MiB; no compute PIDs). Full training still awaits completion of data preparation and verification.
- Training/evaluation: not started. Accuracy goal: unproven.
