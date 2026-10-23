# V7 SHORT Foothold Memory experiment

The included notebook extends the V7 RTX baseline with a causal five-frame RGB-D visual window, four optical-flow pairs, a world-aligned 8x8 ground grid, last-five touchdown memory per leg, per-cell prior contact experience, location-aware leg attention, and next-footfall cell, 3D position, and time heads. The original V7 baseline notebook cells remain in the edited notebook. The run on `UGP_USER` reused the completed V7 visual caches and trained the extension for three epochs on 28 trajectories; it evaluated on four strict unseen trajectories. The notebook runner reported `status: complete` on 2026-10-10 at 08:45:22 UTC.

## Dataset calibration

`V7_SHORT_CAMERA_CALIBRATION_DATASET.json` applies to TartanGround synthetic `lcam_front` RGB-D only. Official TartanAir modality documentation gives a 640x640 pinhole camera, 90-degree FoV, `fx=fy=cx=cy=320`, zero distortion, and synchronized RGB-D. A published TartanGround processing script provides the front optical-camera-to-base axis conversion and zero translation. The V7 cache stores the camera in forward/right/down axes, so its corresponding base transform is `diag(1,-1,-1,1)`.

Direct comparison of four downloaded ANYmal bag `odom->base` trajectories against the rendered front-camera pose, after optimizing clock offset, gave 1.4-3.3 mm median position discrepancy and 0.016-0.023 degree median orientation discrepancy; details are in `CALIBRATION_VALIDATION.csv`. This supports the synthetic alignment, not a measured calibration for a physical robot. The robot bag also has TF entries for the physical depth cameras; those are different from the AirSim rendered `lcam_front` used here.

The robot and image clocks differ. A causal 3-20 second prefix of each training run estimated 0.115-0.175 second lag. The notebook conservatively delays image timestamps by 0.20 second and uses prediction anchors from 21 seconds onward. Holdout clock estimates are in `HOLDOUT_CLOCK_AUDIT.csv`; the fixed delay was selected from training trajectories, before holdout evaluation.

Sources: https://tartanair.org/modalities.html ; https://tartanair.org/tartanground.html ; https://arxiv.org/html/2505.10696 ; https://github.com/rai-opensource/vernata/blob/main/source/scripts/datasets/tartanground/prepare_dataset.py

## Results and limits

Training processed 5,993 valid anchors per epoch; 2,689 anchors per epoch had no usable labels and were skipped. On holdouts, 1,007 valid leg-anchor rows were scored, including 483 touchdown events. A further 2,445 leg-anchor rows were excluded because the observed touchdown was outside the valid visible grid. Thus these results are conditional on a visible, valid cell and must not be interpreted as accuracy over all footfalls.

Across the four holdouts, cell accuracy on scored rows was 93.25%; on scored touchdown events it was 86.13%. The 3D position MAE on scored touchdowns was 0.179 m and touchdown-time MAE was 0.0196 s. `STRICT_HOLDOUT_SUMMARY.csv` has per-trajectory results, while `STRICT_HOLDOUT_RAW.csv` and `STRICT_HOLDOUT_SKIPPED.csv` allow the exclusions and metrics to be independently checked. No comparison to the original V7 baseline on these new targets was run. This extension trained on GPU 0; GPU 1 was idle.

The checkpoint is `V7_SHORT_FM.pt`. It records the 28 training keys and the calibration file SHA-256. `V7F_2xRTX4080_EXPANDED_28TRAIN_4STRICT_HOLDOUT_SHORT_FOOTHOLD_MEMORY.ipynb` is the full edited notebook with the original baseline cells. Its extension cell expects the calibration JSON at `PERSIST_ROOT/calibration/v7_short_camera.json`. `V7_SHORT_FOOTHOLD_MEMORY_SERVER_RUN.ipynb` is the shorter server variant that reused the completed V7 baseline caches for this run.
