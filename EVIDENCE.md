# Evidence and provenance

**Snapshot date:** 23 October 2026. Claims below come from named local artifacts checked for this portfolio. The RTX SSH host was unreachable on port 22 during this pass, so there is no fresh server-process or checkpoint claim.

| Claim | Primary saved source | Boundary |
|---|---|---|
| V7/V7-F familiar force scores | `V7F_VS_V7_FORCE_BY_HORIZON.csv` | Local diagnostic fitting; no strict-holdout comparison |
| V7-F strict zero-shot force scores | `UNSEEN_TRAJECTORY_OVERALL.csv` | Four excluded trajectories; 64 leg-vectors per horizon |
| FeelWM/TartanGround pilot | Four `strict_test_metrics.json` exports | Capped optimizer steps; not convergence |
| Foothold memory | `STRICT_HOLDOUT_SUMMARY.csv` plus the saved event-coverage audit | Scored-window conditional results; only 449/2,928 unique touchdowns scored |
| Independent terrain model | Epoch-3 case metrics and project status report | Validation/model-selection cases; not untouched test |
| CS781 Phase 1 | `CS781_ARCHIVE_INVENTORY_20261006.md` and named ZIP manifests | Historical screenshot scores; private leaderboard not freshly retrieved |
| CS781 Assignment 3 | Saved official grader evidence and run package | 258/258 checks; no convergence/grade inference |

`PROVENANCE.json` records the original artifact basename and SHA-256 for each copied file. Notebook entries also record the transform applied. Original source files are not silently edited. The repository's generated plot derives only from the two published aggregate CSVs.

## Verification still needed

1. Restore authorized RTX access and inspect current run state and final strict exports.
2. Produce matched V7 versus V7-F strict-holdout evaluation with a sealed protocol.
3. Complete all four matched FeelWM ten-epoch evaluations.
4. Resolve the foothold model's low event coverage and missing rear-leg scored touchdowns.
5. Obtain independent test and reconstruction benchmarks for the separate 3D model.
6. Confirm publication rights for shared source, data, and course solutions before expanding this public copy.
