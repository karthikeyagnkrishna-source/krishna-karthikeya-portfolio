# SDD ledger — plan: docs/superpowers/plans/2026-09-30-tartanground-k1-k5-leg-transformer-plan.md

Setup: Ruling: continue in the existing dedicated `codex/tartanground-pilot` checkout — the user explicitly required immediate execution without another question and this checkout contains only this experiment's prior work — cost if wrong: changes remain isolated by branch but not by a second linked checkout.
Setup: Ruling: created the standard ledger workspace directly — the bundled helper misinterprets the Windows `C:/` path under Git Bash and loops on failed directory creation — cost if wrong: none to experiment behavior; only helper-generated bookkeeping is bypassed.
Pre-flight: Task 1 indexed sample contract feeds Task 2 model inputs — existing batch keys will be preserved.
Pre-flight: Task 2 optional leg blocks feed Task 3 rollout — disabled behavior must stay exact and enabled groups keep `[B,9,64,800]`.
Pre-flight: Task 3 loss/rollout audit outputs feed Task 4 evaluator/orchestrator — named per-horizon losses and teacher-mode counts are required.
Pre-flight: Task 4 artifacts feed Task 5 packaging — phase state names and paths must be stable and included in package validation.
