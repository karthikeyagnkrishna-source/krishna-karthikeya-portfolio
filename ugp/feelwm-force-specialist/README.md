# FeelWM force-specialist continuation

The isolated force-specialist branch includes the full model, data pipeline, loss code, checkpointing, training and evaluation scripts, launch configurations and tests. It is a proposed/experimental continuation in the saved source record; no fresh successful training result is claimed.

Browse the [full source tree](src/) and its [historical implementation README](src/README.md). File hashes and publication transformations are listed in the [complete source manifest](../source-history/SOURCE_MANIFEST.json).

Use each package's original requirements/configuration with compatible Python and PyTorch versions. Private recordings, feature caches and pretrained checkpoints must be supplied separately. Package imports such as `v7legs_body` require the relevant `src` directories on the Python import path. Machine-specific defaults were replaced with `/path/to/ugp/...`; configure those locations before running. This publication pass inspected and packaged source; it did not start training or reproduce research scores.
