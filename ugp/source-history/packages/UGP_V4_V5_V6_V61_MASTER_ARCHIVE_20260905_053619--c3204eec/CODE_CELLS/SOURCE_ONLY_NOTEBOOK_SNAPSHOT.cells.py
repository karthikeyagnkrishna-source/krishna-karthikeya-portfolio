# Source inspection export of the adjacent notebook.
# Notebook execution order and dependencies still apply.
# IPython magic, shell lines and prose-only conclusion cells are comments.

# %% notebook cell 0
_VIS_MODELS = None

# %% notebook cell 1
# ======================================================================
# FINAL UGP ARCHIVER / DOWNLOADER
#
# Run this AFTER V4, V5, V6 and V6.1 have completed.
#
# Archives:
#   - V4 results + checkpoints
#   - V5 results + checkpoints
#   - V6 original results + checkpoints
#   - V6.1 postprocessed results
#   - preprocessing cache (/content/v3_multiterrain_cache)
#   - exact executed source cells recovered from current IPython history
#   - model registry + file manifest + SHA256 checksums
#
# Raw /content/tartanground is EXCLUDED by default because it is multi-GB,
# public/redownloadable source data. Set INCLUDE_RAW_TARTANGROUND=True if
# you explicitly want to archive it too.
# ======================================================================

from pathlib import Path
import os, shutil, hashlib, json, csv, datetime
import pandas as pd
from google.colab import files

INCLUDE_PREPROCESSING_CACHE = True
INCLUDE_RAW_TARTANGROUND = False
INCLUDE_EXISTING_LEGACY_ZIPS = True

ROOT = Path("/content")
STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
ARCHIVE_ROOT = ROOT / f"UGP_V4_V5_V6_V61_ARCHIVE_{STAMP}"

if ARCHIVE_ROOT.exists():
    shutil.rmtree(ARCHIVE_ROOT)
ARCHIVE_ROOT.mkdir(parents=True)

# ----------------------------------------------------------------------
# 1) COPY RESULT DIRECTORIES
# ----------------------------------------------------------------------
sources = {
    "RESULTS/V4": ROOT / "v4_residual_prior_direct_results",
    "RESULTS/V5": ROOT / "v5_physics_contact_results",
    "RESULTS/V6_AND_V6_1": ROOT / "v6_continuous_contact_physics_results",
}

if INCLUDE_PREPROCESSING_CACHE:
    sources["PREPROCESSING_CACHE/v3_multiterrain_cache"] = ROOT / "v3_multiterrain_cache"

if INCLUDE_RAW_TARTANGROUND:
    sources["RAW_DATA/tartanground"] = ROOT / "tartanground"

copy_log = []
for rel, src in sources.items():
    dst = ARCHIVE_ROOT / rel
    if src.exists():
        shutil.copytree(src, dst)
        copy_log.append((rel, str(src), "COPIED"))
    else:
        copy_log.append((rel, str(src), "MISSING"))

# Preserve old standalone ZIP if it exists. This is redundant with V4 folder
# but included because it was an explicit output of notebook Cell 2.
if INCLUDE_EXISTING_LEGACY_ZIPS:
    oldzip = ROOT / "V4_RESIDUAL_PRIOR_DIRECT_ALL_RESULTS.zip"
    if oldzip.exists():
        d = ARCHIVE_ROOT / "LEGACY_PACKAGES"
        d.mkdir(exist_ok=True)
        shutil.copy2(oldzip, d / oldzip.name)

# ----------------------------------------------------------------------
# 2) RECOVER EXECUTED NOTEBOOK CELL SOURCES FROM IPYTHON HISTORY
# ----------------------------------------------------------------------
code_dir = ARCHIVE_ROOT / "CODE_CELLS"
code_dir.mkdir(exist_ok=True)

history = list(get_ipython().history_manager.input_hist_raw)

cell_specs = [
    ("00_INIT_VISUAL_MODEL_CACHE.txt",
     lambda s: s.strip() == "_VIS_MODELS = None"),
    ("01_V4_STANDALONE_RESIDUAL_PRIOR_DIRECT.txt",
     lambda s: "STANDALONE V4 EXPERIMENT" in s),
    ("02_V4_DOWNLOAD_RESULTS.txt",
     lambda s: "DOWNLOAD ALL V4 EXPERIMENT RESULTS" in s),
    ("03_V5_PHYSICS_4REGIME_CONTACT.txt",
     lambda s: "V5 PATCH" in s and "4-REGIME CONTACT" in s),
    ("04_V6_CONTINUOUS_CONTACT_PHYSICS.txt",
     lambda s: "V6 PATCH" in s and "CONTINUOUS CONTACT LOAD" in s),
    ("05_V6_1_POSTPROCESS_HARD_SWITCH.txt",
     lambda s: "V6.1 POST-PROCESS PATCH" in s),
]

recovered = []
recovered_sources = []
for filename, predicate in cell_specs:
    matches = [s for s in history if isinstance(s, str) and predicate(s)]
    if matches:
        src = matches[-1]
        (code_dir / filename).write_text(src)
        recovered.append((filename, "RECOVERED", len(src.splitlines())))
        recovered_sources.append(src)
    else:
        (code_dir / filename).write_text(
            "# SOURCE NOT FOUND IN CURRENT IPYTHON HISTORY.\n"
            "# Keep the original .ipynb as the authoritative source.\n"
        )
        recovered.append((filename, "NOT_FOUND", 0))
        recovered_sources.append(
            "# Source not found in this runtime history.\n"
        )

# Complete raw executed input history as a secondary safety backup.
with open(code_dir / "ALL_EXECUTED_INPUT_HISTORY.txt", "w") as f:
    for i, s in enumerate(history):
        if not isinstance(s, str) or not s.strip():
            continue
        f.write("\n\n" + "="*90 + f"\nHISTORY ENTRY {i}\n" + "="*90 + "\n")
        f.write(s)

# Create a source-only notebook snapshot from the six identified cells.
snapshot_nb = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in src.splitlines()],
        }
        for src in recovered_sources
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
(code_dir / "SOURCE_ONLY_NOTEBOOK_SNAPSHOT.ipynb").write_text(
    json.dumps(snapshot_nb)
)

# Also copy any .ipynb physically present in /content.
notebook_backup_dir = ARCHIVE_ROOT / "NOTEBOOK_FILES_FOUND_IN_CONTENT"
notebook_backup_dir.mkdir(exist_ok=True)
for p in ROOT.glob("*.ipynb"):
    try:
        shutil.copy2(p, notebook_backup_dir / p.name)
    except Exception:
        pass

# ----------------------------------------------------------------------
# 3) WRITE MODEL REGISTRY
# ----------------------------------------------------------------------
registry = pd.DataFrame([
    dict(version="V4", expert="SHORT", checkpoint="RESULTS/V4/PRIOR_SHORT.pt",
         architecture="DirectResidualDynamics", initialization="fresh global prior",
         training="0.5 s history -> direct residual +0.1 s; chronological/no shuffle",
         final_use="V4 native short expert"),
    dict(version="V4", expert="LONG", checkpoint="RESULTS/V4/PRIOR_LONG.pt",
         architecture="DirectResidualDynamics", initialization="fresh global prior",
         training="4 s history -> direct residual +0.1..+1.0 s in one pass",
         final_use="V4 direct long expert"),
    dict(version="V5", expert="SHORT", checkpoint="RESULTS/V5/V5_PRIOR_SHORT.pt",
         architecture="PhysicsContactResidualDynamics", initialization="fresh V5 model",
         training="physics features/loss + 4 contact/load regimes; 0.5 s -> +0.1 s",
         final_use="V5 short expert"),
    dict(version="V5", expert="LONG", checkpoint="RESULTS/V5/V5_PRIOR_LONG.pt",
         architecture="PhysicsContactResidualDynamics", initialization="fresh V5 model",
         training="physics features/loss + 4 contact/load regimes; 4 s -> +0.1..+1.0 s",
         final_use="V5 long expert"),
    dict(version="V6", expert="SHORT", checkpoint="RESULTS/V6_AND_V6_1/V6_PRIOR_SHORT.pt",
         architecture="ContinuousContactPhysicsDynamicsV6",
         initialization="warm-start compatible representation tensors from V5 short, then V6-trained",
         training="binary contact + continuous force magnitude/direction + physics torque residual",
         final_use="V6.1 uses this expert ONLY at +0.1 s"),
    dict(version="V6", expert="LONG", checkpoint="RESULTS/V6_AND_V6_1/V6_PRIOR_LONG.pt",
         architecture="ContinuousContactPhysicsDynamicsV6",
         initialization="warm-start compatible representation tensors from V5 long, then V6-trained",
         training="4 s direct multi-horizon; binary contact + continuous force + physics torque residual",
         final_use="V6.1 uses this expert at +0.2..+1.0 s"),
    dict(version="V6.1", expert="POSTPROCESS", checkpoint="NONE",
         architecture="No new neural network",
         initialization="uses already-trained V6 short/long prediction CSVs",
         training="NO RETRAINING",
         final_use="+0.1 s short only; +0.2..1.0 s long only"),
])
registry.to_csv(ARCHIVE_ROOT / "MODEL_REGISTRY.csv", index=False)

# ----------------------------------------------------------------------
# 4) WRITE HUMAN-READABLE TRACKING REPORT
# ----------------------------------------------------------------------
report = r"""# UGP MODEL / FILE TRACKING REPORT

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
"""
(ARCHIVE_ROOT / "MODEL_FILE_TRACKING_REPORT.md").write_text(report)

pd.DataFrame(copy_log, columns=["archive_location","source","status"]).to_csv(
    ARCHIVE_ROOT / "COPY_STATUS.csv", index=False
)
pd.DataFrame(recovered, columns=["cell_file","status","lines"]).to_csv(
    ARCHIVE_ROOT / "CODE_CELL_RECOVERY_STATUS.csv", index=False
)

# ----------------------------------------------------------------------
# 5) BUILD FILE MANIFEST + SHA256 CHECKSUMS
# ----------------------------------------------------------------------
manifest_rows = []
for p in sorted(ARCHIVE_ROOT.rglob("*")):
    if not p.is_file():
        continue
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    manifest_rows.append({
        "relative_path": str(p.relative_to(ARCHIVE_ROOT)),
        "size_bytes": p.stat().st_size,
        "size_MB": p.stat().st_size / (1024**2),
        "sha256": h.hexdigest(),
    })

manifest = pd.DataFrame(manifest_rows)
manifest.to_csv(ARCHIVE_ROOT / "ARCHIVE_MANIFEST_SHA256.csv", index=False)

# ----------------------------------------------------------------------
# 6) VERIFY IMPORTANT CHECKPOINTS / FINAL RESULT FILES
# ----------------------------------------------------------------------
important = [
    "RESULTS/V4/PRIOR_SHORT.pt",
    "RESULTS/V4/PRIOR_LONG.pt",
    "RESULTS/V5/V5_PRIOR_SHORT.pt",
    "RESULTS/V5/V5_PRIOR_LONG.pt",
    "RESULTS/V6_AND_V6_1/V6_PRIOR_SHORT.pt",
    "RESULTS/V6_AND_V6_1/V6_PRIOR_LONG.pt",
    "RESULTS/V6_AND_V6_1/V6_1_ALL_16_TESTS_ALL_SENSOR_VALUES.csv",
    "RESULTS/V6_AND_V6_1/V6_1_MEAN_METRICS_ACROSS_16_TESTS.csv",
]
print("\nIMPORTANT FILE CHECK")
print("="*100)
missing_important = []
for rel in important:
    p = ARCHIVE_ROOT / rel
    ok = p.exists()
    print(("OK      " if ok else "MISSING ") + rel)
    if not ok:
        missing_important.append(rel)

# ----------------------------------------------------------------------
# 7) ZIP EVERYTHING AND DOWNLOAD
# ----------------------------------------------------------------------
zip_base = ROOT / f"UGP_V4_V5_V6_V61_MASTER_ARCHIVE_{STAMP}"
zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=ARCHIVE_ROOT)

size_mb = Path(zip_path).stat().st_size / (1024**2)
print("\n" + "="*100)
print("MASTER ARCHIVE READY")
print("="*100)
print("Folder :", ARCHIVE_ROOT)
print("ZIP    :", zip_path)
print(f"Size   : {size_mb:.2f} MB")
print("Files  :", len(manifest_rows))

if missing_important:
    print("\nWARNING: Some important expected files were missing:")
    for x in missing_important:
        print(" -", x)
    print("The ZIP was still created so existing work is preserved.")
else:
    print("\nAll important model checkpoints and V6.1 final result files were found.")

print("\nDownloading master archive...")
files.download(zip_path)

# %% notebook cell 2
# ======================================================================
# FINAL UGP ARCHIVER / DOWNLOADER
#
# Run this AFTER V4, V5, V6 and V6.1 have completed.
#
# Archives:
#   - V4 results + checkpoints
#   - V5 results + checkpoints
#   - V6 original results + checkpoints
#   - V6.1 postprocessed results
#   - preprocessing cache (/content/v3_multiterrain_cache)
#   - exact executed source cells recovered from current IPython history
#   - model registry + file manifest + SHA256 checksums
#
# Raw /content/tartanground is EXCLUDED by default because it is multi-GB,
# public/redownloadable source data. Set INCLUDE_RAW_TARTANGROUND=True if
# you explicitly want to archive it too.
# ======================================================================

from pathlib import Path
import os, shutil, hashlib, json, csv, datetime
import pandas as pd
from google.colab import files

INCLUDE_PREPROCESSING_CACHE = True
INCLUDE_RAW_TARTANGROUND = False
INCLUDE_EXISTING_LEGACY_ZIPS = True

ROOT = Path("/content")
STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
ARCHIVE_ROOT = ROOT / f"UGP_V4_V5_V6_V61_ARCHIVE_{STAMP}"

if ARCHIVE_ROOT.exists():
    shutil.rmtree(ARCHIVE_ROOT)
ARCHIVE_ROOT.mkdir(parents=True)

# ----------------------------------------------------------------------
# 1) COPY RESULT DIRECTORIES
# ----------------------------------------------------------------------
sources = {
    "RESULTS/V4": ROOT / "v4_residual_prior_direct_results",
    "RESULTS/V5": ROOT / "v5_physics_contact_results",
    "RESULTS/V6_AND_V6_1": ROOT / "v6_continuous_contact_physics_results",
}

if INCLUDE_PREPROCESSING_CACHE:
    sources["PREPROCESSING_CACHE/v3_multiterrain_cache"] = ROOT / "v3_multiterrain_cache"

if INCLUDE_RAW_TARTANGROUND:
    sources["RAW_DATA/tartanground"] = ROOT / "tartanground"

copy_log = []
for rel, src in sources.items():
    dst = ARCHIVE_ROOT / rel
    if src.exists():
        shutil.copytree(src, dst)
        copy_log.append((rel, str(src), "COPIED"))
    else:
        copy_log.append((rel, str(src), "MISSING"))

# Preserve old standalone ZIP if it exists. This is redundant with V4 folder
# but included because it was an explicit output of notebook Cell 2.
if INCLUDE_EXISTING_LEGACY_ZIPS:
    oldzip = ROOT / "V4_RESIDUAL_PRIOR_DIRECT_ALL_RESULTS.zip"
    if oldzip.exists():
        d = ARCHIVE_ROOT / "LEGACY_PACKAGES"
        d.mkdir(exist_ok=True)
        shutil.copy2(oldzip, d / oldzip.name)

# ----------------------------------------------------------------------
# 2) RECOVER EXECUTED NOTEBOOK CELL SOURCES FROM IPYTHON HISTORY
# ----------------------------------------------------------------------
code_dir = ARCHIVE_ROOT / "CODE_CELLS"
code_dir.mkdir(exist_ok=True)

history = list(get_ipython().history_manager.input_hist_raw)

cell_specs = [
    ("00_INIT_VISUAL_MODEL_CACHE.txt",
     lambda s: s.strip() == "_VIS_MODELS = None"),
    ("01_V4_STANDALONE_RESIDUAL_PRIOR_DIRECT.txt",
     lambda s: "STANDALONE V4 EXPERIMENT" in s),
    ("02_V4_DOWNLOAD_RESULTS.txt",
     lambda s: "DOWNLOAD ALL V4 EXPERIMENT RESULTS" in s),
    ("03_V5_PHYSICS_4REGIME_CONTACT.txt",
     lambda s: "V5 PATCH" in s and "4-REGIME CONTACT" in s),
    ("04_V6_CONTINUOUS_CONTACT_PHYSICS.txt",
     lambda s: "V6 PATCH" in s and "CONTINUOUS CONTACT LOAD" in s),
    ("05_V6_1_POSTPROCESS_HARD_SWITCH.txt",
     lambda s: "V6.1 POST-PROCESS PATCH" in s),
]

recovered = []
recovered_sources = []
for filename, predicate in cell_specs:
    matches = [s for s in history if isinstance(s, str) and predicate(s)]
    if matches:
        src = matches[-1]
        (code_dir / filename).write_text(src)
        recovered.append((filename, "RECOVERED", len(src.splitlines())))
        recovered_sources.append(src)
    else:
        (code_dir / filename).write_text(
            "# SOURCE NOT FOUND IN CURRENT IPYTHON HISTORY.\n"
            "# Keep the original .ipynb as the authoritative source.\n"
        )
        recovered.append((filename, "NOT_FOUND", 0))
        recovered_sources.append(
            "# Source not found in this runtime history.\n"
        )

# Complete raw executed input history as a secondary safety backup.
with open(code_dir / "ALL_EXECUTED_INPUT_HISTORY.txt", "w") as f:
    for i, s in enumerate(history):
        if not isinstance(s, str) or not s.strip():
            continue
        f.write("\n\n" + "="*90 + f"\nHISTORY ENTRY {i}\n" + "="*90 + "\n")
        f.write(s)

# Create a source-only notebook snapshot from the six identified cells.
snapshot_nb = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in src.splitlines()],
        }
        for src in recovered_sources
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
(code_dir / "SOURCE_ONLY_NOTEBOOK_SNAPSHOT.ipynb").write_text(
    json.dumps(snapshot_nb)
)

# Also copy any .ipynb physically present in /content.
notebook_backup_dir = ARCHIVE_ROOT / "NOTEBOOK_FILES_FOUND_IN_CONTENT"
notebook_backup_dir.mkdir(exist_ok=True)
for p in ROOT.glob("*.ipynb"):
    try:
        shutil.copy2(p, notebook_backup_dir / p.name)
    except Exception:
        pass

# ----------------------------------------------------------------------
# 3) WRITE MODEL REGISTRY
# ----------------------------------------------------------------------
registry = pd.DataFrame([
    dict(version="V4", expert="SHORT", checkpoint="RESULTS/V4/PRIOR_SHORT.pt",
         architecture="DirectResidualDynamics", initialization="fresh global prior",
         training="0.5 s history -> direct residual +0.1 s; chronological/no shuffle",
         final_use="V4 native short expert"),
    dict(version="V4", expert="LONG", checkpoint="RESULTS/V4/PRIOR_LONG.pt",
         architecture="DirectResidualDynamics", initialization="fresh global prior",
         training="4 s history -> direct residual +0.1..+1.0 s in one pass",
         final_use="V4 direct long expert"),
    dict(version="V5", expert="SHORT", checkpoint="RESULTS/V5/V5_PRIOR_SHORT.pt",
         architecture="PhysicsContactResidualDynamics", initialization="fresh V5 model",
         training="physics features/loss + 4 contact/load regimes; 0.5 s -> +0.1 s",
         final_use="V5 short expert"),
    dict(version="V5", expert="LONG", checkpoint="RESULTS/V5/V5_PRIOR_LONG.pt",
         architecture="PhysicsContactResidualDynamics", initialization="fresh V5 model",
         training="physics features/loss + 4 contact/load regimes; 4 s -> +0.1..+1.0 s",
         final_use="V5 long expert"),
    dict(version="V6", expert="SHORT", checkpoint="RESULTS/V6_AND_V6_1/V6_PRIOR_SHORT.pt",
         architecture="ContinuousContactPhysicsDynamicsV6",
         initialization="warm-start compatible representation tensors from V5 short, then V6-trained",
         training="binary contact + continuous force magnitude/direction + physics torque residual",
         final_use="V6.1 uses this expert ONLY at +0.1 s"),
    dict(version="V6", expert="LONG", checkpoint="RESULTS/V6_AND_V6_1/V6_PRIOR_LONG.pt",
         architecture="ContinuousContactPhysicsDynamicsV6",
         initialization="warm-start compatible representation tensors from V5 long, then V6-trained",
         training="4 s direct multi-horizon; binary contact + continuous force + physics torque residual",
         final_use="V6.1 uses this expert at +0.2..+1.0 s"),
    dict(version="V6.1", expert="POSTPROCESS", checkpoint="NONE",
         architecture="No new neural network",
         initialization="uses already-trained V6 short/long prediction CSVs",
         training="NO RETRAINING",
         final_use="+0.1 s short only; +0.2..1.0 s long only"),
])
registry.to_csv(ARCHIVE_ROOT / "MODEL_REGISTRY.csv", index=False)

# ----------------------------------------------------------------------
# 4) WRITE HUMAN-READABLE TRACKING REPORT
# ----------------------------------------------------------------------
report = r"""# UGP MODEL / FILE TRACKING REPORT

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
"""
(ARCHIVE_ROOT / "MODEL_FILE_TRACKING_REPORT.md").write_text(report)

pd.DataFrame(copy_log, columns=["archive_location","source","status"]).to_csv(
    ARCHIVE_ROOT / "COPY_STATUS.csv", index=False
)
pd.DataFrame(recovered, columns=["cell_file","status","lines"]).to_csv(
    ARCHIVE_ROOT / "CODE_CELL_RECOVERY_STATUS.csv", index=False
)

# ----------------------------------------------------------------------
# 5) BUILD FILE MANIFEST + SHA256 CHECKSUMS
# ----------------------------------------------------------------------
manifest_rows = []
for p in sorted(ARCHIVE_ROOT.rglob("*")):
    if not p.is_file():
        continue
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    manifest_rows.append({
        "relative_path": str(p.relative_to(ARCHIVE_ROOT)),
        "size_bytes": p.stat().st_size,
        "size_MB": p.stat().st_size / (1024**2),
        "sha256": h.hexdigest(),
    })

manifest = pd.DataFrame(manifest_rows)
manifest.to_csv(ARCHIVE_ROOT / "ARCHIVE_MANIFEST_SHA256.csv", index=False)

# ----------------------------------------------------------------------
# 6) VERIFY IMPORTANT CHECKPOINTS / FINAL RESULT FILES
# ----------------------------------------------------------------------
important = [
    "RESULTS/V4/PRIOR_SHORT.pt",
    "RESULTS/V4/PRIOR_LONG.pt",
    "RESULTS/V5/V5_PRIOR_SHORT.pt",
    "RESULTS/V5/V5_PRIOR_LONG.pt",
    "RESULTS/V6_AND_V6_1/V6_PRIOR_SHORT.pt",
    "RESULTS/V6_AND_V6_1/V6_PRIOR_LONG.pt",
    "RESULTS/V6_AND_V6_1/V6_1_ALL_16_TESTS_ALL_SENSOR_VALUES.csv",
    "RESULTS/V6_AND_V6_1/V6_1_MEAN_METRICS_ACROSS_16_TESTS.csv",
]
print("\nIMPORTANT FILE CHECK")
print("="*100)
missing_important = []
for rel in important:
    p = ARCHIVE_ROOT / rel
    ok = p.exists()
    print(("OK      " if ok else "MISSING ") + rel)
    if not ok:
        missing_important.append(rel)

# ----------------------------------------------------------------------
# 7) ZIP EVERYTHING AND DOWNLOAD
# ----------------------------------------------------------------------
zip_base = ROOT / f"UGP_V4_V5_V6_V61_MASTER_ARCHIVE_{STAMP}"
zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=ARCHIVE_ROOT)

size_mb = Path(zip_path).stat().st_size / (1024**2)
print("\n" + "="*100)
print("MASTER ARCHIVE READY")
print("="*100)
print("Folder :", ARCHIVE_ROOT)
print("ZIP    :", zip_path)
print(f"Size   : {size_mb:.2f} MB")
print("Files  :", len(manifest_rows))

if missing_important:
    print("\nWARNING: Some important expected files were missing:")
    for x in missing_important:
        print(" -", x)
    print("The ZIP was still created so existing work is preserved.")
else:
    print("\nAll important model checkpoints and V6.1 final result files were found.")

print("\nDownloading master archive...")
files.download(zip_path)

# %% notebook cell 3
# ======================================================================
# FINAL UGP ARCHIVER / DOWNLOADER
#
# Run this AFTER V4, V5, V6 and V6.1 have completed.
#
# Archives:
#   - V4 results + checkpoints
#   - V5 results + checkpoints
#   - V6 original results + checkpoints
#   - V6.1 postprocessed results
#   - preprocessing cache (/content/v3_multiterrain_cache)
#   - exact executed source cells recovered from current IPython history
#   - model registry + file manifest + SHA256 checksums
#
# Raw /content/tartanground is EXCLUDED by default because it is multi-GB,
# public/redownloadable source data. Set INCLUDE_RAW_TARTANGROUND=True if
# you explicitly want to archive it too.
# ======================================================================

from pathlib import Path
import os, shutil, hashlib, json, csv, datetime
import pandas as pd
from google.colab import files

INCLUDE_PREPROCESSING_CACHE = True
INCLUDE_RAW_TARTANGROUND = False
INCLUDE_EXISTING_LEGACY_ZIPS = True

ROOT = Path("/content")
STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
ARCHIVE_ROOT = ROOT / f"UGP_V4_V5_V6_V61_ARCHIVE_{STAMP}"

if ARCHIVE_ROOT.exists():
    shutil.rmtree(ARCHIVE_ROOT)
ARCHIVE_ROOT.mkdir(parents=True)

# ----------------------------------------------------------------------
# 1) COPY RESULT DIRECTORIES
# ----------------------------------------------------------------------
sources = {
    "RESULTS/V4": ROOT / "v4_residual_prior_direct_results",
    "RESULTS/V5": ROOT / "v5_physics_contact_results",
    "RESULTS/V6_AND_V6_1": ROOT / "v6_continuous_contact_physics_results",
}

if INCLUDE_PREPROCESSING_CACHE:
    sources["PREPROCESSING_CACHE/v3_multiterrain_cache"] = ROOT / "v3_multiterrain_cache"

if INCLUDE_RAW_TARTANGROUND:
    sources["RAW_DATA/tartanground"] = ROOT / "tartanground"

copy_log = []
for rel, src in sources.items():
    dst = ARCHIVE_ROOT / rel
    if src.exists():
        shutil.copytree(src, dst)
        copy_log.append((rel, str(src), "COPIED"))
    else:
        copy_log.append((rel, str(src), "MISSING"))

# Preserve old standalone ZIP if it exists. This is redundant with V4 folder
# but included because it was an explicit output of notebook Cell 2.
if INCLUDE_EXISTING_LEGACY_ZIPS:
    oldzip = ROOT / "V4_RESIDUAL_PRIOR_DIRECT_ALL_RESULTS.zip"
    if oldzip.exists():
        d = ARCHIVE_ROOT / "LEGACY_PACKAGES"
        d.mkdir(exist_ok=True)
        shutil.copy2(oldzip, d / oldzip.name)

# ----------------------------------------------------------------------
# 2) RECOVER EXECUTED NOTEBOOK CELL SOURCES FROM IPYTHON HISTORY
# ----------------------------------------------------------------------
code_dir = ARCHIVE_ROOT / "CODE_CELLS"
code_dir.mkdir(exist_ok=True)

history = list(get_ipython().history_manager.input_hist_raw)

cell_specs = [
    ("00_INIT_VISUAL_MODEL_CACHE.txt",
     lambda s: s.strip() == "_VIS_MODELS = None"),
    ("01_V4_STANDALONE_RESIDUAL_PRIOR_DIRECT.txt",
     lambda s: "STANDALONE V4 EXPERIMENT" in s),
    ("02_V4_DOWNLOAD_RESULTS.txt",
     lambda s: "DOWNLOAD ALL V4 EXPERIMENT RESULTS" in s),
    ("03_V5_PHYSICS_4REGIME_CONTACT.txt",
     lambda s: "V5 PATCH" in s and "4-REGIME CONTACT" in s),
    ("04_V6_CONTINUOUS_CONTACT_PHYSICS.txt",
     lambda s: "V6 PATCH" in s and "CONTINUOUS CONTACT LOAD" in s),
    ("05_V6_1_POSTPROCESS_HARD_SWITCH.txt",
     lambda s: "V6.1 POST-PROCESS PATCH" in s),
]

recovered = []
recovered_sources = []
for filename, predicate in cell_specs:
    matches = [s for s in history if isinstance(s, str) and predicate(s)]
    if matches:
        src = matches[-1]
        (code_dir / filename).write_text(src)
        recovered.append((filename, "RECOVERED", len(src.splitlines())))
        recovered_sources.append(src)
    else:
        (code_dir / filename).write_text(
            "# SOURCE NOT FOUND IN CURRENT IPYTHON HISTORY.\n"
            "# Keep the original .ipynb as the authoritative source.\n"
        )
        recovered.append((filename, "NOT_FOUND", 0))
        recovered_sources.append(
            "# Source not found in this runtime history.\n"
        )

# Complete raw executed input history as a secondary safety backup.
with open(code_dir / "ALL_EXECUTED_INPUT_HISTORY.txt", "w") as f:
    for i, s in enumerate(history):
        if not isinstance(s, str) or not s.strip():
            continue
        f.write("\n\n" + "="*90 + f"\nHISTORY ENTRY {i}\n" + "="*90 + "\n")
        f.write(s)

# Create a source-only notebook snapshot from the six identified cells.
snapshot_nb = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in src.splitlines()],
        }
        for src in recovered_sources
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
(code_dir / "SOURCE_ONLY_NOTEBOOK_SNAPSHOT.ipynb").write_text(
    json.dumps(snapshot_nb)
)

# Also copy any .ipynb physically present in /content.
notebook_backup_dir = ARCHIVE_ROOT / "NOTEBOOK_FILES_FOUND_IN_CONTENT"
notebook_backup_dir.mkdir(exist_ok=True)
for p in ROOT.glob("*.ipynb"):
    try:
        shutil.copy2(p, notebook_backup_dir / p.name)
    except Exception:
        pass

# ----------------------------------------------------------------------
# 3) WRITE MODEL REGISTRY
# ----------------------------------------------------------------------
registry = pd.DataFrame([
    dict(version="V4", expert="SHORT", checkpoint="RESULTS/V4/PRIOR_SHORT.pt",
         architecture="DirectResidualDynamics", initialization="fresh global prior",
         training="0.5 s history -> direct residual +0.1 s; chronological/no shuffle",
         final_use="V4 native short expert"),
    dict(version="V4", expert="LONG", checkpoint="RESULTS/V4/PRIOR_LONG.pt",
         architecture="DirectResidualDynamics", initialization="fresh global prior",
         training="4 s history -> direct residual +0.1..+1.0 s in one pass",
         final_use="V4 direct long expert"),
    dict(version="V5", expert="SHORT", checkpoint="RESULTS/V5/V5_PRIOR_SHORT.pt",
         architecture="PhysicsContactResidualDynamics", initialization="fresh V5 model",
         training="physics features/loss + 4 contact/load regimes; 0.5 s -> +0.1 s",
         final_use="V5 short expert"),
    dict(version="V5", expert="LONG", checkpoint="RESULTS/V5/V5_PRIOR_LONG.pt",
         architecture="PhysicsContactResidualDynamics", initialization="fresh V5 model",
         training="physics features/loss + 4 contact/load regimes; 4 s -> +0.1..+1.0 s",
         final_use="V5 long expert"),
    dict(version="V6", expert="SHORT", checkpoint="RESULTS/V6_AND_V6_1/V6_PRIOR_SHORT.pt",
         architecture="ContinuousContactPhysicsDynamicsV6",
         initialization="warm-start compatible representation tensors from V5 short, then V6-trained",
         training="binary contact + continuous force magnitude/direction + physics torque residual",
         final_use="V6.1 uses this expert ONLY at +0.1 s"),
    dict(version="V6", expert="LONG", checkpoint="RESULTS/V6_AND_V6_1/V6_PRIOR_LONG.pt",
         architecture="ContinuousContactPhysicsDynamicsV6",
         initialization="warm-start compatible representation tensors from V5 long, then V6-trained",
         training="4 s direct multi-horizon; binary contact + continuous force + physics torque residual",
         final_use="V6.1 uses this expert at +0.2..+1.0 s"),
    dict(version="V6.1", expert="POSTPROCESS", checkpoint="NONE",
         architecture="No new neural network",
         initialization="uses already-trained V6 short/long prediction CSVs",
         training="NO RETRAINING",
         final_use="+0.1 s short only; +0.2..1.0 s long only"),
])
registry.to_csv(ARCHIVE_ROOT / "MODEL_REGISTRY.csv", index=False)

# ----------------------------------------------------------------------
# 4) WRITE HUMAN-READABLE TRACKING REPORT
# ----------------------------------------------------------------------
report = r"""# UGP MODEL / FILE TRACKING REPORT

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
"""
(ARCHIVE_ROOT / "MODEL_FILE_TRACKING_REPORT.md").write_text(report)

pd.DataFrame(copy_log, columns=["archive_location","source","status"]).to_csv(
    ARCHIVE_ROOT / "COPY_STATUS.csv", index=False
)
pd.DataFrame(recovered, columns=["cell_file","status","lines"]).to_csv(
    ARCHIVE_ROOT / "CODE_CELL_RECOVERY_STATUS.csv", index=False
)

# ----------------------------------------------------------------------
# 5) BUILD FILE MANIFEST + SHA256 CHECKSUMS
# ----------------------------------------------------------------------
manifest_rows = []
for p in sorted(ARCHIVE_ROOT.rglob("*")):
    if not p.is_file():
        continue
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    manifest_rows.append({
        "relative_path": str(p.relative_to(ARCHIVE_ROOT)),
        "size_bytes": p.stat().st_size,
        "size_MB": p.stat().st_size / (1024**2),
        "sha256": h.hexdigest(),
    })

manifest = pd.DataFrame(manifest_rows)
manifest.to_csv(ARCHIVE_ROOT / "ARCHIVE_MANIFEST_SHA256.csv", index=False)

# ----------------------------------------------------------------------
# 6) VERIFY IMPORTANT CHECKPOINTS / FINAL RESULT FILES
# ----------------------------------------------------------------------
important = [
    "RESULTS/V4/PRIOR_SHORT.pt",
    "RESULTS/V4/PRIOR_LONG.pt",
    "RESULTS/V5/V5_PRIOR_SHORT.pt",
    "RESULTS/V5/V5_PRIOR_LONG.pt",
    "RESULTS/V6_AND_V6_1/V6_PRIOR_SHORT.pt",
    "RESULTS/V6_AND_V6_1/V6_PRIOR_LONG.pt",
    "RESULTS/V6_AND_V6_1/V6_1_ALL_16_TESTS_ALL_SENSOR_VALUES.csv",
    "RESULTS/V6_AND_V6_1/V6_1_MEAN_METRICS_ACROSS_16_TESTS.csv",
]
print("\nIMPORTANT FILE CHECK")
print("="*100)
missing_important = []
for rel in important:
    p = ARCHIVE_ROOT / rel
    ok = p.exists()
    print(("OK      " if ok else "MISSING ") + rel)
    if not ok:
        missing_important.append(rel)

# ----------------------------------------------------------------------
# 7) ZIP EVERYTHING AND DOWNLOAD
# ----------------------------------------------------------------------
zip_base = ROOT / f"UGP_V4_V5_V6_V61_MASTER_ARCHIVE_{STAMP}"
zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=ARCHIVE_ROOT)

size_mb = Path(zip_path).stat().st_size / (1024**2)
print("\n" + "="*100)
print("MASTER ARCHIVE READY")
print("="*100)
print("Folder :", ARCHIVE_ROOT)
print("ZIP    :", zip_path)
print(f"Size   : {size_mb:.2f} MB")
print("Files  :", len(manifest_rows))

if missing_important:
    print("\nWARNING: Some important expected files were missing:")
    for x in missing_important:
        print(" -", x)
    print("The ZIP was still created so existing work is preserved.")
else:
    print("\nAll important model checkpoints and V6.1 final result files were found.")

print("\nDownloading master archive...")
files.download(zip_path)

# %% notebook cell 4
# ======================================================================
# FINAL UGP ARCHIVER / DOWNLOADER
#
# Run this AFTER V4, V5, V6 and V6.1 have completed.
#
# Archives:
#   - V4 results + checkpoints
#   - V5 results + checkpoints
#   - V6 original results + checkpoints
#   - V6.1 postprocessed results
#   - preprocessing cache (/content/v3_multiterrain_cache)
#   - exact executed source cells recovered from current IPython history
#   - model registry + file manifest + SHA256 checksums
#
# Raw /content/tartanground is EXCLUDED by default because it is multi-GB,
# public/redownloadable source data. Set INCLUDE_RAW_TARTANGROUND=True if
# you explicitly want to archive it too.
# ======================================================================

from pathlib import Path
import os, shutil, hashlib, json, csv, datetime
import pandas as pd
from google.colab import files

INCLUDE_PREPROCESSING_CACHE = True
INCLUDE_RAW_TARTANGROUND = False
INCLUDE_EXISTING_LEGACY_ZIPS = True

ROOT = Path("/content")
STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
ARCHIVE_ROOT = ROOT / f"UGP_V4_V5_V6_V61_ARCHIVE_{STAMP}"

if ARCHIVE_ROOT.exists():
    shutil.rmtree(ARCHIVE_ROOT)
ARCHIVE_ROOT.mkdir(parents=True)

# ----------------------------------------------------------------------
# 1) COPY RESULT DIRECTORIES
# ----------------------------------------------------------------------
sources = {
    "RESULTS/V4": ROOT / "v4_residual_prior_direct_results",
    "RESULTS/V5": ROOT / "v5_physics_contact_results",
    "RESULTS/V6_AND_V6_1": ROOT / "v6_continuous_contact_physics_results",
}

if INCLUDE_PREPROCESSING_CACHE:
    sources["PREPROCESSING_CACHE/v3_multiterrain_cache"] = ROOT / "v3_multiterrain_cache"

if INCLUDE_RAW_TARTANGROUND:
    sources["RAW_DATA/tartanground"] = ROOT / "tartanground"

copy_log = []
for rel, src in sources.items():
    dst = ARCHIVE_ROOT / rel
    if src.exists():
        shutil.copytree(src, dst)
        copy_log.append((rel, str(src), "COPIED"))
    else:
        copy_log.append((rel, str(src), "MISSING"))

# Preserve old standalone ZIP if it exists. This is redundant with V4 folder
# but included because it was an explicit output of notebook Cell 2.
if INCLUDE_EXISTING_LEGACY_ZIPS:
    oldzip = ROOT / "V4_RESIDUAL_PRIOR_DIRECT_ALL_RESULTS.zip"
    if oldzip.exists():
        d = ARCHIVE_ROOT / "LEGACY_PACKAGES"
        d.mkdir(exist_ok=True)
        shutil.copy2(oldzip, d / oldzip.name)

# ----------------------------------------------------------------------
# 2) RECOVER EXECUTED NOTEBOOK CELL SOURCES FROM IPYTHON HISTORY
# ----------------------------------------------------------------------
code_dir = ARCHIVE_ROOT / "CODE_CELLS"
code_dir.mkdir(exist_ok=True)

history = list(get_ipython().history_manager.input_hist_raw)

cell_specs = [
    ("00_INIT_VISUAL_MODEL_CACHE.txt",
     lambda s: s.strip() == "_VIS_MODELS = None"),
    ("01_V4_STANDALONE_RESIDUAL_PRIOR_DIRECT.txt",
     lambda s: "STANDALONE V4 EXPERIMENT" in s),
    ("02_V4_DOWNLOAD_RESULTS.txt",
     lambda s: "DOWNLOAD ALL V4 EXPERIMENT RESULTS" in s),
    ("03_V5_PHYSICS_4REGIME_CONTACT.txt",
     lambda s: "V5 PATCH" in s and "4-REGIME CONTACT" in s),
    ("04_V6_CONTINUOUS_CONTACT_PHYSICS.txt",
     lambda s: "V6 PATCH" in s and "CONTINUOUS CONTACT LOAD" in s),
    ("05_V6_1_POSTPROCESS_HARD_SWITCH.txt",
     lambda s: "V6.1 POST-PROCESS PATCH" in s),
]

recovered = []
recovered_sources = []
for filename, predicate in cell_specs:
    matches = [s for s in history if isinstance(s, str) and predicate(s)]
    if matches:
        src = matches[-1]
        (code_dir / filename).write_text(src)
        recovered.append((filename, "RECOVERED", len(src.splitlines())))
        recovered_sources.append(src)
    else:
        (code_dir / filename).write_text(
            "# SOURCE NOT FOUND IN CURRENT IPYTHON HISTORY.\n"
            "# Keep the original .ipynb as the authoritative source.\n"
        )
        recovered.append((filename, "NOT_FOUND", 0))
        recovered_sources.append(
            "# Source not found in this runtime history.\n"
        )

# Complete raw executed input history as a secondary safety backup.
with open(code_dir / "ALL_EXECUTED_INPUT_HISTORY.txt", "w") as f:
    for i, s in enumerate(history):
        if not isinstance(s, str) or not s.strip():
            continue
        f.write("\n\n" + "="*90 + f"\nHISTORY ENTRY {i}\n" + "="*90 + "\n")
        f.write(s)

# Create a source-only notebook snapshot from the six identified cells.
snapshot_nb = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in src.splitlines()],
        }
        for src in recovered_sources
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
(code_dir / "SOURCE_ONLY_NOTEBOOK_SNAPSHOT.ipynb").write_text(
    json.dumps(snapshot_nb)
)

# Also copy any .ipynb physically present in /content.
notebook_backup_dir = ARCHIVE_ROOT / "NOTEBOOK_FILES_FOUND_IN_CONTENT"
notebook_backup_dir.mkdir(exist_ok=True)
for p in ROOT.glob("*.ipynb"):
    try:
        shutil.copy2(p, notebook_backup_dir / p.name)
    except Exception:
        pass

# ----------------------------------------------------------------------
# 3) WRITE MODEL REGISTRY
# ----------------------------------------------------------------------
registry = pd.DataFrame([
    dict(version="V4", expert="SHORT", checkpoint="RESULTS/V4/PRIOR_SHORT.pt",
         architecture="DirectResidualDynamics", initialization="fresh global prior",
         training="0.5 s history -> direct residual +0.1 s; chronological/no shuffle",
         final_use="V4 native short expert"),
    dict(version="V4", expert="LONG", checkpoint="RESULTS/V4/PRIOR_LONG.pt",
         architecture="DirectResidualDynamics", initialization="fresh global prior",
         training="4 s history -> direct residual +0.1..+1.0 s in one pass",
         final_use="V4 direct long expert"),
    dict(version="V5", expert="SHORT", checkpoint="RESULTS/V5/V5_PRIOR_SHORT.pt",
         architecture="PhysicsContactResidualDynamics", initialization="fresh V5 model",
         training="physics features/loss + 4 contact/load regimes; 0.5 s -> +0.1 s",
         final_use="V5 short expert"),
    dict(version="V5", expert="LONG", checkpoint="RESULTS/V5/V5_PRIOR_LONG.pt",
         architecture="PhysicsContactResidualDynamics", initialization="fresh V5 model",
         training="physics features/loss + 4 contact/load regimes; 4 s -> +0.1..+1.0 s",
         final_use="V5 long expert"),
    dict(version="V6", expert="SHORT", checkpoint="RESULTS/V6_AND_V6_1/V6_PRIOR_SHORT.pt",
         architecture="ContinuousContactPhysicsDynamicsV6",
         initialization="warm-start compatible representation tensors from V5 short, then V6-trained",
         training="binary contact + continuous force magnitude/direction + physics torque residual",
         final_use="V6.1 uses this expert ONLY at +0.1 s"),
    dict(version="V6", expert="LONG", checkpoint="RESULTS/V6_AND_V6_1/V6_PRIOR_LONG.pt",
         architecture="ContinuousContactPhysicsDynamicsV6",
         initialization="warm-start compatible representation tensors from V5 long, then V6-trained",
         training="4 s direct multi-horizon; binary contact + continuous force + physics torque residual",
         final_use="V6.1 uses this expert at +0.2..+1.0 s"),
    dict(version="V6.1", expert="POSTPROCESS", checkpoint="NONE",
         architecture="No new neural network",
         initialization="uses already-trained V6 short/long prediction CSVs",
         training="NO RETRAINING",
         final_use="+0.1 s short only; +0.2..1.0 s long only"),
])
registry.to_csv(ARCHIVE_ROOT / "MODEL_REGISTRY.csv", index=False)

# ----------------------------------------------------------------------
# 4) WRITE HUMAN-READABLE TRACKING REPORT
# ----------------------------------------------------------------------
report = r"""# UGP MODEL / FILE TRACKING REPORT

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
"""
(ARCHIVE_ROOT / "MODEL_FILE_TRACKING_REPORT.md").write_text(report)

pd.DataFrame(copy_log, columns=["archive_location","source","status"]).to_csv(
    ARCHIVE_ROOT / "COPY_STATUS.csv", index=False
)
pd.DataFrame(recovered, columns=["cell_file","status","lines"]).to_csv(
    ARCHIVE_ROOT / "CODE_CELL_RECOVERY_STATUS.csv", index=False
)

# ----------------------------------------------------------------------
# 5) BUILD FILE MANIFEST + SHA256 CHECKSUMS
# ----------------------------------------------------------------------
manifest_rows = []
for p in sorted(ARCHIVE_ROOT.rglob("*")):
    if not p.is_file():
        continue
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    manifest_rows.append({
        "relative_path": str(p.relative_to(ARCHIVE_ROOT)),
        "size_bytes": p.stat().st_size,
        "size_MB": p.stat().st_size / (1024**2),
        "sha256": h.hexdigest(),
    })

manifest = pd.DataFrame(manifest_rows)
manifest.to_csv(ARCHIVE_ROOT / "ARCHIVE_MANIFEST_SHA256.csv", index=False)

# ----------------------------------------------------------------------
# 6) VERIFY IMPORTANT CHECKPOINTS / FINAL RESULT FILES
# ----------------------------------------------------------------------
important = [
    "RESULTS/V4/PRIOR_SHORT.pt",
    "RESULTS/V4/PRIOR_LONG.pt",
    "RESULTS/V5/V5_PRIOR_SHORT.pt",
    "RESULTS/V5/V5_PRIOR_LONG.pt",
    "RESULTS/V6_AND_V6_1/V6_PRIOR_SHORT.pt",
    "RESULTS/V6_AND_V6_1/V6_PRIOR_LONG.pt",
    "RESULTS/V6_AND_V6_1/V6_1_ALL_16_TESTS_ALL_SENSOR_VALUES.csv",
    "RESULTS/V6_AND_V6_1/V6_1_MEAN_METRICS_ACROSS_16_TESTS.csv",
]
print("\nIMPORTANT FILE CHECK")
print("="*100)
missing_important = []
for rel in important:
    p = ARCHIVE_ROOT / rel
    ok = p.exists()
    print(("OK      " if ok else "MISSING ") + rel)
    if not ok:
        missing_important.append(rel)

# ----------------------------------------------------------------------
# 7) ZIP EVERYTHING AND DOWNLOAD
# ----------------------------------------------------------------------
zip_base = ROOT / f"UGP_V4_V5_V6_V61_MASTER_ARCHIVE_{STAMP}"
zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=ARCHIVE_ROOT)

size_mb = Path(zip_path).stat().st_size / (1024**2)
print("\n" + "="*100)
print("MASTER ARCHIVE READY")
print("="*100)
print("Folder :", ARCHIVE_ROOT)
print("ZIP    :", zip_path)
print(f"Size   : {size_mb:.2f} MB")
print("Files  :", len(manifest_rows))

if missing_important:
    print("\nWARNING: Some important expected files were missing:")
    for x in missing_important:
        print(" -", x)
    print("The ZIP was still created so existing work is preserved.")
else:
    print("\nAll important model checkpoints and V6.1 final result files were found.")

print("\nDownloading master archive...")
files.download(zip_path)

# %% notebook cell 5
# ======================================================================
# FINAL UGP ARCHIVER / DOWNLOADER
#
# Run this AFTER V4, V5, V6 and V6.1 have completed.
#
# Archives:
#   - V4 results + checkpoints
#   - V5 results + checkpoints
#   - V6 original results + checkpoints
#   - V6.1 postprocessed results
#   - preprocessing cache (/content/v3_multiterrain_cache)
#   - exact executed source cells recovered from current IPython history
#   - model registry + file manifest + SHA256 checksums
#
# Raw /content/tartanground is EXCLUDED by default because it is multi-GB,
# public/redownloadable source data. Set INCLUDE_RAW_TARTANGROUND=True if
# you explicitly want to archive it too.
# ======================================================================

from pathlib import Path
import os, shutil, hashlib, json, csv, datetime
import pandas as pd
from google.colab import files

INCLUDE_PREPROCESSING_CACHE = True
INCLUDE_RAW_TARTANGROUND = False
INCLUDE_EXISTING_LEGACY_ZIPS = True

ROOT = Path("/content")
STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
ARCHIVE_ROOT = ROOT / f"UGP_V4_V5_V6_V61_ARCHIVE_{STAMP}"

if ARCHIVE_ROOT.exists():
    shutil.rmtree(ARCHIVE_ROOT)
ARCHIVE_ROOT.mkdir(parents=True)

# ----------------------------------------------------------------------
# 1) COPY RESULT DIRECTORIES
# ----------------------------------------------------------------------
sources = {
    "RESULTS/V4": ROOT / "v4_residual_prior_direct_results",
    "RESULTS/V5": ROOT / "v5_physics_contact_results",
    "RESULTS/V6_AND_V6_1": ROOT / "v6_continuous_contact_physics_results",
}

if INCLUDE_PREPROCESSING_CACHE:
    sources["PREPROCESSING_CACHE/v3_multiterrain_cache"] = ROOT / "v3_multiterrain_cache"

if INCLUDE_RAW_TARTANGROUND:
    sources["RAW_DATA/tartanground"] = ROOT / "tartanground"

copy_log = []
for rel, src in sources.items():
    dst = ARCHIVE_ROOT / rel
    if src.exists():
        shutil.copytree(src, dst)
        copy_log.append((rel, str(src), "COPIED"))
    else:
        copy_log.append((rel, str(src), "MISSING"))

# Preserve old standalone ZIP if it exists. This is redundant with V4 folder
# but included because it was an explicit output of notebook Cell 2.
if INCLUDE_EXISTING_LEGACY_ZIPS:
    oldzip = ROOT / "V4_RESIDUAL_PRIOR_DIRECT_ALL_RESULTS.zip"
    if oldzip.exists():
        d = ARCHIVE_ROOT / "LEGACY_PACKAGES"
        d.mkdir(exist_ok=True)
        shutil.copy2(oldzip, d / oldzip.name)

# ----------------------------------------------------------------------
# 2) RECOVER EXECUTED NOTEBOOK CELL SOURCES FROM IPYTHON HISTORY
# ----------------------------------------------------------------------
code_dir = ARCHIVE_ROOT / "CODE_CELLS"
code_dir.mkdir(exist_ok=True)

history = list(get_ipython().history_manager.input_hist_raw)

cell_specs = [
    ("00_INIT_VISUAL_MODEL_CACHE.txt",
     lambda s: s.strip() == "_VIS_MODELS = None"),
    ("01_V4_STANDALONE_RESIDUAL_PRIOR_DIRECT.txt",
     lambda s: "STANDALONE V4 EXPERIMENT" in s),
    ("02_V4_DOWNLOAD_RESULTS.txt",
     lambda s: "DOWNLOAD ALL V4 EXPERIMENT RESULTS" in s),
    ("03_V5_PHYSICS_4REGIME_CONTACT.txt",
     lambda s: "V5 PATCH" in s and "4-REGIME CONTACT" in s),
    ("04_V6_CONTINUOUS_CONTACT_PHYSICS.txt",
     lambda s: "V6 PATCH" in s and "CONTINUOUS CONTACT LOAD" in s),
    ("05_V6_1_POSTPROCESS_HARD_SWITCH.txt",
     lambda s: "V6.1 POST-PROCESS PATCH" in s),
]

recovered = []
recovered_sources = []
for filename, predicate in cell_specs:
    matches = [s for s in history if isinstance(s, str) and predicate(s)]
    if matches:
        src = matches[-1]
        (code_dir / filename).write_text(src)
        recovered.append((filename, "RECOVERED", len(src.splitlines())))
        recovered_sources.append(src)
    else:
        (code_dir / filename).write_text(
            "# SOURCE NOT FOUND IN CURRENT IPYTHON HISTORY.\n"
            "# Keep the original .ipynb as the authoritative source.\n"
        )
        recovered.append((filename, "NOT_FOUND", 0))
        recovered_sources.append(
            "# Source not found in this runtime history.\n"
        )

# Complete raw executed input history as a secondary safety backup.
with open(code_dir / "ALL_EXECUTED_INPUT_HISTORY.txt", "w") as f:
    for i, s in enumerate(history):
        if not isinstance(s, str) or not s.strip():
            continue
        f.write("\n\n" + "="*90 + f"\nHISTORY ENTRY {i}\n" + "="*90 + "\n")
        f.write(s)

# Create a source-only notebook snapshot from the six identified cells.
snapshot_nb = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in src.splitlines()],
        }
        for src in recovered_sources
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
(code_dir / "SOURCE_ONLY_NOTEBOOK_SNAPSHOT.ipynb").write_text(
    json.dumps(snapshot_nb)
)

# Also copy any .ipynb physically present in /content.
notebook_backup_dir = ARCHIVE_ROOT / "NOTEBOOK_FILES_FOUND_IN_CONTENT"
notebook_backup_dir.mkdir(exist_ok=True)
for p in ROOT.glob("*.ipynb"):
    try:
        shutil.copy2(p, notebook_backup_dir / p.name)
    except Exception:
        pass

# ----------------------------------------------------------------------
# 3) WRITE MODEL REGISTRY
# ----------------------------------------------------------------------
registry = pd.DataFrame([
    dict(version="V4", expert="SHORT", checkpoint="RESULTS/V4/PRIOR_SHORT.pt",
         architecture="DirectResidualDynamics", initialization="fresh global prior",
         training="0.5 s history -> direct residual +0.1 s; chronological/no shuffle",
         final_use="V4 native short expert"),
    dict(version="V4", expert="LONG", checkpoint="RESULTS/V4/PRIOR_LONG.pt",
         architecture="DirectResidualDynamics", initialization="fresh global prior",
         training="4 s history -> direct residual +0.1..+1.0 s in one pass",
         final_use="V4 direct long expert"),
    dict(version="V5", expert="SHORT", checkpoint="RESULTS/V5/V5_PRIOR_SHORT.pt",
         architecture="PhysicsContactResidualDynamics", initialization="fresh V5 model",
         training="physics features/loss + 4 contact/load regimes; 0.5 s -> +0.1 s",
         final_use="V5 short expert"),
    dict(version="V5", expert="LONG", checkpoint="RESULTS/V5/V5_PRIOR_LONG.pt",
         architecture="PhysicsContactResidualDynamics", initialization="fresh V5 model",
         training="physics features/loss + 4 contact/load regimes; 4 s -> +0.1..+1.0 s",
         final_use="V5 long expert"),
    dict(version="V6", expert="SHORT", checkpoint="RESULTS/V6_AND_V6_1/V6_PRIOR_SHORT.pt",
         architecture="ContinuousContactPhysicsDynamicsV6",
         initialization="warm-start compatible representation tensors from V5 short, then V6-trained",
         training="binary contact + continuous force magnitude/direction + physics torque residual",
         final_use="V6.1 uses this expert ONLY at +0.1 s"),
    dict(version="V6", expert="LONG", checkpoint="RESULTS/V6_AND_V6_1/V6_PRIOR_LONG.pt",
         architecture="ContinuousContactPhysicsDynamicsV6",
         initialization="warm-start compatible representation tensors from V5 long, then V6-trained",
         training="4 s direct multi-horizon; binary contact + continuous force + physics torque residual",
         final_use="V6.1 uses this expert at +0.2..+1.0 s"),
    dict(version="V6.1", expert="POSTPROCESS", checkpoint="NONE",
         architecture="No new neural network",
         initialization="uses already-trained V6 short/long prediction CSVs",
         training="NO RETRAINING",
         final_use="+0.1 s short only; +0.2..1.0 s long only"),
])
registry.to_csv(ARCHIVE_ROOT / "MODEL_REGISTRY.csv", index=False)

# ----------------------------------------------------------------------
# 4) WRITE HUMAN-READABLE TRACKING REPORT
# ----------------------------------------------------------------------
report = r"""# UGP MODEL / FILE TRACKING REPORT

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
"""
(ARCHIVE_ROOT / "MODEL_FILE_TRACKING_REPORT.md").write_text(report)

pd.DataFrame(copy_log, columns=["archive_location","source","status"]).to_csv(
    ARCHIVE_ROOT / "COPY_STATUS.csv", index=False
)
pd.DataFrame(recovered, columns=["cell_file","status","lines"]).to_csv(
    ARCHIVE_ROOT / "CODE_CELL_RECOVERY_STATUS.csv", index=False
)

# ----------------------------------------------------------------------
# 5) BUILD FILE MANIFEST + SHA256 CHECKSUMS
# ----------------------------------------------------------------------
manifest_rows = []
for p in sorted(ARCHIVE_ROOT.rglob("*")):
    if not p.is_file():
        continue
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    manifest_rows.append({
        "relative_path": str(p.relative_to(ARCHIVE_ROOT)),
        "size_bytes": p.stat().st_size,
        "size_MB": p.stat().st_size / (1024**2),
        "sha256": h.hexdigest(),
    })

manifest = pd.DataFrame(manifest_rows)
manifest.to_csv(ARCHIVE_ROOT / "ARCHIVE_MANIFEST_SHA256.csv", index=False)

# ----------------------------------------------------------------------
# 6) VERIFY IMPORTANT CHECKPOINTS / FINAL RESULT FILES
# ----------------------------------------------------------------------
important = [
    "RESULTS/V4/PRIOR_SHORT.pt",
    "RESULTS/V4/PRIOR_LONG.pt",
    "RESULTS/V5/V5_PRIOR_SHORT.pt",
    "RESULTS/V5/V5_PRIOR_LONG.pt",
    "RESULTS/V6_AND_V6_1/V6_PRIOR_SHORT.pt",
    "RESULTS/V6_AND_V6_1/V6_PRIOR_LONG.pt",
    "RESULTS/V6_AND_V6_1/V6_1_ALL_16_TESTS_ALL_SENSOR_VALUES.csv",
    "RESULTS/V6_AND_V6_1/V6_1_MEAN_METRICS_ACROSS_16_TESTS.csv",
]
print("\nIMPORTANT FILE CHECK")
print("="*100)
missing_important = []
for rel in important:
    p = ARCHIVE_ROOT / rel
    ok = p.exists()
    print(("OK      " if ok else "MISSING ") + rel)
    if not ok:
        missing_important.append(rel)

# ----------------------------------------------------------------------
# 7) ZIP EVERYTHING AND DOWNLOAD
# ----------------------------------------------------------------------
zip_base = ROOT / f"UGP_V4_V5_V6_V61_MASTER_ARCHIVE_{STAMP}"
zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=ARCHIVE_ROOT)

size_mb = Path(zip_path).stat().st_size / (1024**2)
print("\n" + "="*100)
print("MASTER ARCHIVE READY")
print("="*100)
print("Folder :", ARCHIVE_ROOT)
print("ZIP    :", zip_path)
print(f"Size   : {size_mb:.2f} MB")
print("Files  :", len(manifest_rows))

if missing_important:
    print("\nWARNING: Some important expected files were missing:")
    for x in missing_important:
        print(" -", x)
    print("The ZIP was still created so existing work is preserved.")
else:
    print("\nAll important model checkpoints and V6.1 final result files were found.")

print("\nDownloading master archive...")
files.download(zip_path)
