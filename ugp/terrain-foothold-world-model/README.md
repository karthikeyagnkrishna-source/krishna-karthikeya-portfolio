# 3D Terrain–Foothold World Model

This is an independent project, separate from V7/FeelWM. The supplied architecture combines a terrain representation with robot dynamics and foothold prediction. A synthetic CUDA smoke test established that the staged model could execute; later saved reports show a real-data pilot and an epoch-3 checkpoint from the 28-trajectory run.

The epoch-3 export measured 5,117 overlapping validation windows on four whole trajectories. Reported foot error was **1.10 m** at +0.1 s and **3.78 m** at +1.0 s; contact accuracy was about **47.1%**. Those trajectories were also used for validation/model selection, so this is **not an independent test**. A completed 20-epoch run, benchmark-quality 3D reconstruction score, and physical robot validation were not verified.

The full source is now public: [implementation](src/), [supplied original](snapshots/supplied-original/), and [causal-pilot revision](snapshots/causal-pilot/). The supplied handoff is attributed as a shared input; this repository does not assert sole authorship. The source manifest preserves original hashes and publication changes. Private data and learned weights remain excluded. Configure the portable data paths and obtain the required visual foundations before running. An independent test export is still needed to establish generalization.
