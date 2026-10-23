# V7 timestamped visual history

Timestamp-aware image and robot history extends the original V7 interface. The saved project README records CPU behavior and initialization checks, with no training or accuracy improvement established at that snapshot. The release includes the full adapter, history builder, runtime, entry point and tests.

Browse the [full source tree](src/v7_visual_history/) and its [historical implementation README](src/v7_visual_history/README.md). File hashes and publication transformations are listed in the [complete source manifest](../source-history/SOURCE_MANIFEST.json).

Use each package's original requirements/configuration with compatible Python and PyTorch versions. Private recordings, feature caches and pretrained checkpoints must be supplied separately. Package imports such as `v7legs_body` require the relevant `src` directories on the Python import path. Machine-specific defaults were replaced with `/path/to/ugp/...`; configure those locations before running. This publication pass inspected and packaged source; it did not start training or reproduce research scores.
