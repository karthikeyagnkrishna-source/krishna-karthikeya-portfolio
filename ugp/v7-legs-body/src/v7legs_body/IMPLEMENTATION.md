# Implementation ledger

Goal: implement the approved V7Legs + geometry + V7B architecture and launch a resumable 28-trajectory training pipeline with automatic familiar, time-gap, and unseen evaluation.

Work is isolated in this new directory; the project mirror is not a standalone Git checkout. Existing notebook sources and completed runs remain references.

Tasks: reuse notebook definitions and persisted caches; implement residual heads and differentiable FK; construct causal train/validation records; implement staged training and checkpoint recovery; wire comparison evaluation/export; verify material invariants; deploy and verify live training.

User directions: reuse existing code, preserve architecture, avoid small redundant smoke checks, spend credits efficiently.

Ruling: use one implementation context and targeted checks rather than per-task smoke runs. User specifically requested efficient reuse. Keep a persistent ledger and inspect the complete implementation before launch.

Live starting GPU state: physical GPU1 has no active compute job, approximately 4.3 GiB held by existing jobs; GPU0 has Isaac Sim active. No signals or process changes authorized or needed.

Completed: notebook definitions reused through an AST adapter; source installation/download/training/V8 driver statements do not execute. Frozen feature caches avoid visual inference during new-head training.

Completed: differentiable URDF FK and Jacobians, shared residual leg head, cross-leg/horizon body head, normalized supervised/geometry/stance losses, purged head validation splits, staged training and cursor/RNG recovery, automatic exact-value CSV/Excel exports and matched comparison evaluation.

Verified: FK matches the original NumPy implementation; Jacobians agree with finite differences; zero corrections preserve base outputs; a real body update produces gradients in the leg head; foundation weights receive no gradients. Four retained raw recordings all label twist as `base`, matching the URDF root, so stance loss is enabled.

Ruling: reuse original float16 latent caches and their original foundation normalization. Fit the new correction scales on optimization windows only. The frozen foundation already saw head-validation time ranges during its original training; these head validation losses are selection diagnostics, not new unseen evidence. The four trajectory holdouts remain disjoint.

Ruling: preserve the notebook's assumption that recorded future controller commands are known exogenous inputs. Future measured joint/body/contact labels are targets only. This is the same offline prediction setting, not a claim of deployment with unknown future controls.

Evaluation defaults: common legacy causal base adaptation (four epochs) and force adaptation (five epochs) for familiar/gap cases, then five epochs of causal new-head adaptation selected on purged pre-cutoff validation. No adaptation or heldout fitting for unseen. Matched base results are regenerated rather than importing incomparable historical aggregates.

Final review: one fresh reviewer inspected the new modules, scoped to reconstruction, leakage, freeze/gradients, resume, and evaluation. Found an Important stance-loss reduction issue. check_contact_loss.py reproduced it, then reduction='none' was added before coordinate averaging/contact masking. The regression covers mixed contact and all-swing batches. No further Important/Critical findings were reported.

Excel writer: openpyxl 3.1.5 and et_xmlfile reused from the bundled local runtime and copied offline to the isolated remote dependency directory. No global environment changes or certificate bypass.

Launched: detached PID 738350 on physical GPU1. Observed PPID=1, 28 short-cache trajectories completed, epoch 1 validation finite (0.0848000 normalized supervised loss), and epoch 2 optimizer step 1500. NVIDIA process memory approximately 382 MiB. Automatic stages and evaluations are queued by train.py. These observations establish a progressing training job, not completed results.
