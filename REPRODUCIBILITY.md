# Reproducing the source

Choose a model family from the [UGP map](ugp/README.md), then use the corresponding source tree and its requirements. Historical notebook variants are retained for inspection and lineage; filenames containing FINAL or REAL do not select a canonical trained checkpoint.

1. Create an isolated Python environment compatible with that snapshot. The saved RTX work commonly used PyTorch 2.5.1 with CUDA 12.1; some Utonia installation paths require Python 3.10, while Colab notebooks target other versions. There is no single verified environment for every historical experiment.
2. Obtain external foundations under their original licenses. The source references [Utonia](https://github.com/Pointcept/Utonia), [WAFT](https://github.com/princeton-vl/WAFT), [TartanAir tooling](https://github.com/castacks/tartanairpy), ANYmal robot descriptions and pretrained visual models. Installed dependency checkouts and environments are not copied into this repository.
3. Supply authorized robot recordings, calibration, feature caches and pretrained checkpoints. Public source does not grant access or redistribution rights to these artifacts.
4. Replace `/path/to/ugp/...` placeholders in the selected configuration or environment variables. Set the relevant package `src` folders on the Python import path when combining `v7legs_body`, feedback, visual history and joint-specialist modules.
5. Run the package's synthetic/CPU checks before configuring a GPU training run. Verify the trajectory masks and whether an experiment uses recorded future controls, local adaptation, oracle inputs or validation-selected checkpoints.

Notebook `.py` exports are source-inspection aids: shell and IPython magic lines are comments, and cell order remains relevant. Execute the canonical notebook in a compatible environment when those installation/bootstrap cells are needed. Full training and published research scores were not reproduced during this release.
