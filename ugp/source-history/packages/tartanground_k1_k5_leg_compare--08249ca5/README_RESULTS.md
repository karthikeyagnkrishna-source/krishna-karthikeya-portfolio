# TartanGround K=1/K=5 comparison results

Selected development-only lambda profile: `uniform` with weights `[0.2, 0.2, 0.2, 0.2, 0.2]`.

Training trajectories: 28; strict unseen test trajectories: ForestEnv/P2001, GreatMarsh/P2027, OldTownFall/P2001, SeasonalForestWinter/P2006.

Both architectures were trained first at K=1, then K=5 initialized from the corresponding K=1 checkpoint. K=5 training used a deterministic 50/50 teacher-forced/autoregressive schedule; development and strict testing were fully autoregressive. Each trajectory used two 700-frame windows centered as close as possible to 25% and 75%, with overlapping frames cached once.

The archive contains best/latest checkpoints, training logs and metrics, development lambda calibration, strict-test metrics and per-sample predictions, split/window manifests, source code, tests, and SHA-256 checksums. Raw sensor and foundation-feature caches are excluded.

Results describe only the four held-out trajectories and do not establish broader generalization. Per-horizon physical metrics are in each architecture/checkpoint `strict_test` directory.
