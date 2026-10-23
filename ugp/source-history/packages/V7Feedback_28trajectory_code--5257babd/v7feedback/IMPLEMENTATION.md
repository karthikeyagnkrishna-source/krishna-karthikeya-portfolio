# Implementation ledger

Plan: docs/superpowers/plans/2026-10-07-v7-feedback.md

Ruling: The user explicitly approved the in-chat design and asked to code and launch it. That authorization covers the written recording and native implementation; do not ask for another plan approval.

Ruling: The ambient Git root is C:/ with unrelated ownership. Preserve it; isolate all new code in v7feedback/ and the new remote run instead of altering global Git configuration or the previous model source.

Ruling: Reuse existing code as requested; tests target new mathematical/gradient/export risks, not trivial wrappers or repeated smoke tests.

Initial remote check: SSH to UGP_USER (SSH_HOST:22) timed out. No current GPU or launch claim is supported. User was asked asynchronously to restore IITK routing while implementation continues.

Local verification runtime: isolated CPU-only torch 2.5.1 in .verification_deps; excluded from delivery. Parent sources remain unchanged.

Task 1: model checks observed RED (five missing-feature failures), then GREEN (5/5). Actual gradients reach reused leg/body transformers, future labels cannot affect output, force changes affect later rounds, zero initialization preserves the seed, and all-swing/zero-force losses are finite.

Ruling: Evaluation uses zero local correction-head epochs for all variants, while preserving the original 4-epoch V7 and 5-epoch V7-F causal adaptation. This keeps seed/one-pass leg/body predictions identical and isolates learned force transfer/feedback from unequal local head training. Strict unseen remains unchanged. All exports explicitly describe this matched rerun; do not compare its familiar/gap values as identical to the prior workbooks' five local correction-head epochs.

Final review: fresh reviewer found two Important provenance gaps (unhashed reused implementation and unbound heldout visual/physics/frame inputs), no Critical architecture/force-export issues. Two new tests observed RED. Fix binds all executed reused Python files and URDF, plus every train physics source and queried diagnostic/heldout visual/frame/physics source, to run/case signatures.

Task 2: data/export/alignment and provenance checks passed (9/9). Both GPU branches verified real 28-trajectory caches, seed equality and nonzero force-loss gradients; long branch has 9,602 optimization/2,476 validation windows.

Launch diagnosis: the first launch exited before any optimizer step because Python checkpoint signatures contained tuples, while stored JSON contained lists. Read-only diagnostic confirmed raw equality false but JSON-normalized equality true. Regression observed RED; canonical JSON normalization is the fix. Preserve the pretraining verification manifest and rejected-launch log before generating the corrected run identity, after asserting no training checkpoint exists.

Task 3: complete. Full material suite GREEN, 10/10; both real-data branch checks passed. Corrected detached launch on physical GPU 1 has PID 772556. Live verification: short one-pass epoch 6, 5,500 optimizer steps, loss .1636313, finite gradient norm .4117028, GPU1 utilization35%; best/latest checkpoints exist (60,247,784/89,816,026 bytes). All 28 trajectories enforced; short optimization/validation windows16,825/4,281 and long9,602/2,476. After Stage A and Stage B for both experts, automatic96-case evaluation is configured. This proves a launch and progress, not training completion or improved results.

Source delivery excludes the isolated local verification dependencies and trained weights. Parent source and unrelated shared jobs are unchanged. No further Python edits may be made in this active run without a new run identity.
