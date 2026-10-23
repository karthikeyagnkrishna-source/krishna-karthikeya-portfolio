# 3D Terrain–Foothold World Model

This is an independent project, separate from V7/FeelWM. The supplied architecture combines a terrain representation with robot dynamics and foothold prediction. A synthetic CUDA smoke test established that the staged model could execute; later saved reports show a real-data pilot and an epoch-3 checkpoint from the 28-trajectory run.

The epoch-3 export measured 5,117 overlapping validation windows on four whole trajectories. Reported foot error was **1.10 m** at +0.1 s and **3.78 m** at +1.0 s; contact accuracy was about **47.1%**. Those trajectories were also used for validation/model selection, so this is **not an independent test**. A completed 20-epoch run, benchmark-quality 3D reconstruction score, and physical robot validation were not verified.

The source came from a shared handoff whose public redistribution rights are unresolved, so its code, data, and weights are withheld from this public copy. Release requires permission from the source owner(s), a clean provenance record, and an independent test export.
