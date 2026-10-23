# SDD ledger — plan: C:/path/to/ugp/2026-09-29-tartanground-pilot-implementation.md

Setup: standalone project repository initialized at the local deployed-source copy on branch `codex/tartanground-pilot`; baseline commit `5423716`.
Setup ruling: bundled `sdd-workspace` cannot run because its Git Bash process cannot resolve `basename`; use the identical plan-scoped `.superpowers/sdd/<plan>/progress.md` location directly — cost if wrong: only execution bookkeeping, never project behavior.
Task 1 ruling: local Windows host has no `python` command; execute all Python RED/GREEN and suite gates with `/path/to/ugp/python` on the SSH server — cost if wrong: local-only platform issues could be missed, but the requested target is the server.
Task 1 ruling: the remote environment has no `pytest`; use the standard-library `unittest` runner and the repository's existing unittest style rather than installing a test dependency — cost if wrong: pytest-only fixtures cannot be used, but none are required by this plan.
Pre-flight Task 1 -> Task 5: `TrainConfig.max_steps` is consumed verbatim by K=1/K=5 YAML; consistent.
Pre-flight Task 2 -> Task 3: `RobotSeries` and mapping helpers are the exporter inputs; consistent.
Pre-flight Task 4 -> Task 3: `[N,64,256]` frozen latents are the exporter visual input; consistent.
Pre-flight Tasks 1-4 -> Task 5: orchestration imports public functions and supplies exact spec limits; consistent.
Pre-flight Task 5 -> Task 6: remote stages consume the checked-in CLIs/configurations; consistent.
Task 4 ruling: execute before Tasks 2-3 because validated shared P2000 raw caches and a completed V7 resampler checkpoint were discovered; this eliminates all foundation-model preprocessing without changing the approved tensors — cost if wrong: cache/checkpoint incompatibility, guarded by exact key/shape/fingerprint checks.
Task 4 ruling: use `atol=1e-6` for hand-derived float32 leg-action literals; the default tolerance rejected a correct value by `4.8e-8` — cost if wrong: deviations below one part per million are not detected.
Task 6 ruling: set the K=1 pilot DataLoader to `num_workers=0` and use short `TMPDIR=/tmp/s1v7`; worker sockets failed with `AF_UNIX path too long`, and the small cached dataset does not need multiprocessing — cost if wrong: lower input throughput, not changed training semantics.
