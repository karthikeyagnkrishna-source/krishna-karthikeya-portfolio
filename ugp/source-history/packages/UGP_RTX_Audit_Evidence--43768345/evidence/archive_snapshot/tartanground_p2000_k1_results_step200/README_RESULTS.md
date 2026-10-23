# TartanGround P2000 K=1 pilot results

This package contains the verified result of the credit-capped merged Feel-WM V7 pilot.

## Run summary

- Status: completed at global optimizer step 200
- Effective training: approximately 1.91 epochs
- Data: one ForestEnv/P2000 trajectory, 1,200 frames spanning 119.9 seconds
- Split: 834 training samples and 238 validation-manifest samples, with a 5-second purge gap
- Compute: two RTX 4080 GPUs, DDP, BF16, TF32, effective batch approximately 8
- Trainable parameters: 110.53 million
- End-to-end successful pipeline time: approximately 592 seconds
- Final/best validation total: 16473.87546875
- Final checkpoint SHA-256: c463db53de0c1fedef9803bd150dc6d56b2d2c298856e37314b964afc71f671e

## Evaluation scope

Validation ran at steps 50, 100, 150, and 200. Each event covered 100 distributed
validation samples because the evaluator was capped at 50 batches per GPU. The raw
records are in `run/metrics.jsonl`; compact copies are in `run/validation_summary.csv`
and `run/validation_summary.json`.

This is not an unseen-trajectory test. Validation came from a later chronological
segment of P2000 after a 5-second purge. P2001 was unavailable and was not evaluated.
No physical-unit RMSE, accuracy, precision, recall, or F1 values were produced.

## Checkpoint

`checkpoint/latest.pt` is a resumable PyTorch checkpoint containing model, optimizer,
scheduler, RNG, configuration, epoch index, global step, and best-validation state.
Use only with the included trusted source. Loading the full resumable checkpoint uses
PyTorch pickle semantics.

The raw dataset, prepared sample shards, shared frozen V7 resampler checkpoint, and
redundant intermediate checkpoints are intentionally excluded to keep this package
small. Their provenance and the frozen-resampler SHA-256 are recorded in metadata.

## Verification

The source unit suite was rerun while packaging and passed. See
`verification/tests.txt`. `SHA256SUMS.txt` records hashes for packaged files.
