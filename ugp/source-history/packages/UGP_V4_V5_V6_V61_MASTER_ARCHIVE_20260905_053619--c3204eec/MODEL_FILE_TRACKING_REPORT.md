# UGP MODEL / FILE TRACKING REPORT

## Canonical model lineage

V4
- `DirectResidualDynamics`
- SHORT: 0.5 s history -> +0.1 s residual
- LONG: 4 s history -> direct +0.1..+1.0 s residual
- Checkpoints: `RESULTS/V4/PRIOR_SHORT.pt`, `RESULTS/V4/PRIOR_LONG.pt`

V5
- `PhysicsContactResidualDynamics`
- New V5 global models were trained; V5 reuses V4 processed data/visual features, not V4 checkpoint weights.
- Adds causal identified physics features/losses and 4 contact/load regimes.
- Checkpoints: `RESULTS/V5/V5_PRIOR_SHORT.pt`, `RESULTS/V5/V5_PRIOR_LONG.pt`

V6
- `ContinuousContactPhysicsDynamicsV6`
- Binary contact + continuous force magnitude + force direction.
- Force is reconstructed from contact probability, magnitude and direction.
- Torque is physics estimate + bounded neural residual.
- V6 global models warm-start compatible representation layers from the completed V5 global priors and are then trained under V6 losses.
- Checkpoints: `RESULTS/V6_AND_V6_1/V6_PRIOR_SHORT.pt`, `RESULTS/V6_AND_V6_1/V6_PRIOR_LONG.pt`

V6.1
- **NO NEW TRAINING AND NO NEW CHECKPOINT.**
- Deterministic post-processing on V6 predictions.
- +0.1 s = V6 SHORT ONLY.
- +0.2..+1.0 s = V6 LONG ONLY.
- Canonical V6.1 evaluation files have prefix `V6_1_`.

## Output folders

### `RESULTS/V4/`
Contains:
- V4 short/long checkpoints.
- `ALL_16_TESTS_ALL_SENSOR_VALUES.csv`
- `ALL_16_TESTS_GROUPED_METRICS.csv`
- `MEAN_METRICS_ACROSS_16_TESTS.csv`
- terrain/test-time map.
- 16-test RGB montage and ROI montage.
- per-test CSV/RGB/ROI files.

### `RESULTS/V5/`
Contains:
- V5 short/long checkpoints.
- V5 all-sensor predictions.
- grouped and mean metrics.
- 4-regime contact accuracy.
- V4-vs-V5 comparison.
- per-test V5 CSVs.

### `RESULTS/V6_AND_V6_1/`
Contains BOTH:
- original V6 trained-model results (`V6_*`);
- corrected postprocessed V6.1 results (`V6_1_*`);
- V6 short/long checkpoints (`V6_PRIOR_SHORT.pt`, `V6_PRIOR_LONG.pt`);
- contact / force-magnitude metrics;
- V5-vs-V6 and V6.1-vs-old-V6 comparisons.

## Which result is the current final one?

**Weights:** V6 short + V6 long checkpoints.

**Final expert-selection evaluation:** V6.1 CSVs.

Therefore:
`V6.1 = trained V6 weights + hard expert-selection post-processing`.
Do not describe V6.1 as a separately trained Transformer.

## Preprocessing cache

`PREPROCESSING_CACHE/v3_multiterrain_cache/`

Contains the V4 preprocessing cache, including physics `.npz` and visual-feature `.npz` files. Preserve this to avoid repeating expensive preprocessing.

## Code cells

`CODE_CELLS/` contains individually named copies of the executed cells, plus:
- `ALL_EXECUTED_INPUT_HISTORY.txt`
- `SOURCE_ONLY_NOTEBOOK_SNAPSHOT.ipynb`

The current Colab notebook itself may live in Google Drive rather than `/content`; Colab does not always expose the UI notebook as a local file. The source-only snapshot protects the executable code even if no physical `.ipynb` is found in `/content`.

## Raw dataset

Raw `/content/tartanground/` is excluded by default because it is several GB and redownloadable. It can be included by changing:
`INCLUDE_RAW_TARTANGROUND = True`.
