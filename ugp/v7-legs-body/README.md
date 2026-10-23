# V7 leg and body specialists

Leg/body specialist models, frozen foundation loading, data construction, training, evaluation and verification. This package supplies parent code used by the feedback, joint-specialist and visual-history experiments.

Browse the [full source tree](src/v7legs_body/) and its [historical implementation README](src/v7legs_body/README.md). File hashes and publication transformations are listed in the [complete source manifest](../source-history/SOURCE_MANIFEST.json).

Use each package's original requirements/configuration with compatible Python and PyTorch versions. Private recordings, feature caches and pretrained checkpoints must be supplied separately. Package imports such as `v7legs_body` require the relevant `src` directories on the Python import path. Machine-specific defaults were replaced with `/path/to/ugp/...`; configure those locations before running. This publication pass inspected and packaged source; it did not start training or reproduce research scores.
