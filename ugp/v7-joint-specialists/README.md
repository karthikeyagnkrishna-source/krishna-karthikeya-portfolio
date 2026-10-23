# V7 joint specialists

Independent angle, speed and torque specialists with a force/contact extension. The source includes data extraction, losses, strict metrics, training, evaluation, configuration, operational helpers and tests. The strict two-percent target in the historical README is an experimental goal, not a performance claim for this release.

Browse the [full source tree](src/v7_joint_specialists/) and its [historical implementation README](src/v7_joint_specialists/README.md). File hashes and publication transformations are listed in the [complete source manifest](../source-history/SOURCE_MANIFEST.json).

Use each package's original requirements/configuration with compatible Python and PyTorch versions. Private recordings, feature caches and pretrained checkpoints must be supplied separately. Package imports such as `v7legs_body` require the relevant `src` directories on the Python import path. Machine-specific defaults were replaced with `/path/to/ugp/...`; configure those locations before running. This publication pass inspected and packaged source; it did not start training or reproduce research scores.
