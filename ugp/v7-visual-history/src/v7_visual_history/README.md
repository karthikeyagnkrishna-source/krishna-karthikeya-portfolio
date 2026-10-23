# V7 visual history aligned with robot history

This package adds the current and five previous cached DINO, ConvNeXt and geometry encodings to V7. Every image and robot token carries its actual timestamp relative to the prediction anchor. Attention learns time relationships; there is no forced one-to-one pairing or fixed contact delay.

Verified on 2026-10-10: all 15 new CPU behavior tests and 23 existing joint-specialist regression tests passed. Both actual pretrained branches passed the real-data initialization and gradient checks using ForestEnv/P2000. The receipt is in `verification/real_data_cpu.json`. Training has not started, and prediction improvement has not been evaluated.

- Robot inputs remain 50 samples for SHORT (+0.1 s) and 400 for LONG (+0.2...+1.0 s).
- Image history is six consecutive available frames, normally about 0.5 s for either branch. Missing images can make the actual span longer; early absent frames are masked. This does not provide four seconds of visual history.
- Reuse all original image resamplers, causal robot Transformers, four query-fusion blocks and prediction heads. Only a 149,888-parameter timestamp/projection/residual adapter per branch is trainable by default. Frozen V7 remains in evaluation mode. Reading historical tokens adds computation even though the Transformer weights are reused.
- Five queries read 554 SHORT or 2304 LONG memory tokens of width256. Query self-attention operates on five tokens, not the entire memory bank.
- Zero-initialized correction reproduces the pretrained V7 predictions exactly under the same frozen evaluation execution mode. This is a compatibility check, not evidence of improved accuracy.
- Preserve original supervised losses, known future command assumption and eligible ranges for the 28 training trajectories; four original excluded trajectories stay excluded. The original trainer does not select checkpoints on a new validation set, and that limitation remains explicit.
- Reuse raw per-image feature caches. Rebuild frozen V7/V7-F downstream context caches after training, using the new checkpoint fingerprint and separate namespace. Do not resume old specialist/feedback checkpoints as if they contain visual history.
- Past geometric points remain in their original per-camera coordinate frames. Common-frame terrain tracking and observed pose inputs are not added here. Attention may learn associations but does not guarantee identifying the same terrain patch.

## CPU verification (default; no training)

From the new source directory on UGP_USER:

```bash
/path/to/ugp/python -m v7_visual_history.run \
  --foundation-root /path/to/ugp/V7F_4080_Run \
  --output /path/to/ugp/final \
  --device cpu --check
```

Checks both real 50/400-sample windows and saved pretrained weights, exact initialization equivalence, the original V7 loss/backward and V7-F feature extraction. No optimizer is created and no parameters are updated.

## Training entry point (explicit action)

Use a fresh output directory, such as `/path/to/ugp/training`, and `--train --device cuda:0 --branch short` or `--train --device cuda:1 --branch long`. Defaults preserve the original SHORT three epochs and LONG four epochs; `--epochs` overrides the chosen branch limit. The original per-window optimizer/RNG resume machinery is reused in the new recovery directory; saved RNG tensors are normalized to CPU before restoration. Completion is training completion, not evaluation completion. Training is not launched by installation or verification.

The output directory must not overlap the foundation. Its immutable manifest includes source/checkpoint hashes, the image-selection policy and trajectory identities; incompatible manifests require a separate directory.

## Code interface

```python
from v7_visual_history.model import V7VisualHistory
from v7_visual_history.data import build_history

sample['visual_history'] = build_history(data, visual, anchor_index, 50, 'cpu')
model = V7VisualHistory(pretrained_short_v7)
prediction = model.forward_sample(sample)  # Original V7 residual/contact/regime keys.
force_features = model.features(sample)    # [batch, future_times, 4, 1285]
features, diagnostic = model.features(sample, return_attention=True)
```

Actual force/angle/torque/body values use the original baseline + standardized residual reconstruction. Historical image weights can be inspected through the five-query attention diagnostic, but these weights are not a proof of physical cause.

Run the behavior suite with `python -m unittest discover -s v7_visual_history/tests -v` in the existing PyTorch environment.
