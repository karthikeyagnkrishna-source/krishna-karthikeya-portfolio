# V7 leg/body-to-force feedback

An isolated refinement experiment using the pretrained foundation and parent leg/body model. It includes the data pipeline, recurrent refinement models, training, evaluation, provenance checks and tests. The presence of the code does not establish that its configured training or all evaluation exports completed.

Browse the [full source tree](src/v7feedback/) and its [historical implementation README](src/v7feedback/README.md). File hashes and publication transformations are listed in the [complete source manifest](../source-history/SOURCE_MANIFEST.json).

Use each package's original requirements/configuration with compatible Python and PyTorch versions. Private recordings, feature caches and pretrained checkpoints must be supplied separately. Package imports such as `v7legs_body` require the relevant `src` directories on the Python import path. Machine-specific defaults were replaced with `/path/to/ugp/...`; configure those locations before running. This publication pass inspected and packaged source; it did not start training or reproduce research scores.
