# Source inspection export of the adjacent notebook.
# Notebook execution order and dependencies still apply.
# IPython magic, shell lines and prose-only conclusion cells are comments.

# %% notebook cell 1
# ======================================================================
# 0. FRESH-COLAB SETUP — LIGHTWEIGHT, NO FOUNDATION MODEL DOWNLOADS
# ======================================================================
import sys, subprocess, os
from pathlib import Path

def pip_install(*pkgs):
    subprocess.check_call([sys.executable,'-m','pip','install','-q',*pkgs])

# Colab usually already has these; this is intentionally far lighter than V7 setup.
pip_install('numpy','pandas','scipy','scikit-learn','matplotlib','pillow')

# Official ANYmal-D description is needed only for verified leg FK.
EXT=Path('/content/v8_external'); EXT.mkdir(parents=True,exist_ok=True)
ANYMAL_REPO=EXT/'anymal_d_simple_description'
if not ANYMAL_REPO.exists():
    subprocess.check_call(['git','clone','--depth','1','https://github.com/ANYbotics/anymal_d_simple_description.git',str(ANYMAL_REPO)])
ANYMAL_D_URDF=ANYMAL_REPO/'urdf'/'anymal.urdf'
assert ANYMAL_D_URDF.exists(), ANYMAL_D_URDF

print('Setup complete. No DINO/ConvNeXt/Utonia/WAFT model was downloaded.')
print('ANYmal-D URDF:',ANYMAL_D_URDF)

# %% notebook cell 2
# ======================================================================
# 1. CONFIGURATION + DRIVE + ARTIFACT DISCOVERY
# ======================================================================
import os, re, gc, math, random, copy, time, hashlib, warnings, contextlib, json, zipfile
from pathlib import Path
from collections import defaultdict
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from scipy.spatial import cKDTree, ConvexHull
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from sklearn.metrics import precision_recall_fscore_support

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from google.colab import drive
drive.mount('/content/drive')

DRIVE_PROJECT_DIR='MyDrive/UGP_V7_FULL4B'
PERSIST_ROOT=Path('/content/drive')/DRIVE_PROJECT_DIR
CACHE_ROOT=PERSIST_ROOT/'cache'
OUT_ROOT=PERSIST_ROOT/'results'
V8_ROOT=PERSIST_ROOT/'V8_contactmap_standalone'
V8_ROOT.mkdir(parents=True,exist_ok=True)

DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type!='cuda':
    raise RuntimeError('Use a GPU Colab runtime. V8-A re-encodes cached four-stream tokens through the trained V7 adapters.')

SEED=781
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# Exact V7 architectural constants required only to reconstruct the trained terrain encoder.
SHORT_CONTEXT_SEC=0.5
RAW_GRID=8; RAW_TOKENS=64; LATENTS_PER_BRANCH=16; VIS_DIM=256
FLOW_HISTORY=4; GEOM_REP_POINTS=64; PI_PHYS_DIM=47; N_REGIMES=4
FX=320.0; FY=320.0; CX=320.0; CY=320.0

EXPERIMENTS=[
    dict(terrain_type='forest',environment='ForestEnv',run='P2000',ttests=[28.0,45.0,75.0,90.0]),
    dict(terrain_type='swamp_marsh',environment='GreatMarsh',run='P2023',ttests=[47.0,61.0,73.0,87.0]),
    dict(terrain_type='rough_uneven',environment='SeasonalForestWinter',run='P2000',ttests=[29.0,46.0,64.0,92.0]),
    dict(terrain_type='wet_uneven_proxy',environment='OldTownFall',run='P2000',ttests=[32.0,84.0,100.0,115.0]),
]

# User-level science targets. These are GOALS, not claims about current performance.
FORCE_VECTOR_TARGET_N=5.0
TORQUE_COMPONENT_TARGET_NM=2.0
ENFORCE_V7F_MASTER_GATE=False  # True = stop before V8-C/E/F if V7-F strict force gate is missed.

print('DEVICE:',DEVICE,torch.cuda.get_device_name(0))
print('PERSIST_ROOT:',PERSIST_ROOT)
print('V8_ROOT:',V8_ROOT)

# %% notebook cell 3
# ======================================================================
# 2. CACHE / CHECKPOINT PREFLIGHT + OPTIONAL ZIP RECOVERY
# ======================================================================
def _extract_member_from_zip(zip_path, member_suffix, dest_dir):
    zip_path=Path(zip_path); dest_dir=Path(dest_dir); dest_dir.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        matches=[n for n in z.namelist() if n.endswith(member_suffix)]
        if not matches: return None
        n=matches[0]
        out=dest_dir/Path(n).name
        with z.open(n) as src, open(out,'wb') as dst: dst.write(src.read())
        return out

def _find_zip(pattern):
    cands=list(PERSIST_ROOT.glob(pattern))+list(PERSIST_ROOT.parent.glob(pattern))+list(Path('/content').glob(pattern))
    return cands[0] if cands else None

# Check mandatory cache contracts.
missing=[]
for e in EXPERIMENTS:
    env,run=e['environment'],e['run']
    req=[
        CACHE_ROOT/f'{env}_{run}_physics.npz',
        CACHE_ROOT/f'{env}_{run}_frame_manifest.npz',
        CACHE_ROOT/f'{env}_{run}_v7_dinov3.npy',
        CACHE_ROOT/f'{env}_{run}_v7_convnext.npy',
        CACHE_ROOT/f'{env}_{run}_v7_utonia.npy',
        CACHE_ROOT/f'{env}_{run}_v7_waft.npy',
    ]
    missing += [p for p in req if not p.exists()]

if missing:
    print('MISSING mandatory V7 persisted caches:')
    for p in missing: print(' ',p)
    raise RuntimeError('Fresh V8 cannot truthfully recreate the four-stream terrain memory without these persisted caches. Do not substitute a fake branch.')

V7_SHORT_CKPT=OUT_ROOT/'V7_PRIOR_SHORT.pt'
if not V7_SHORT_CKPT.exists():
    z=_find_zip('V7_RESULTS_ONLY*.zip')
    if z:
        print('Recovering V7 short prior from',z)
        p=_extract_member_from_zip(z,'V7_PRIOR_SHORT.pt','/content/v8_restore')
        if p: V7_SHORT_CKPT=Path(p)
if not V7_SHORT_CKPT.exists():
    raise RuntimeError('V7_PRIOR_SHORT.pt missing. Put V7_RESULTS_ONLY*.zip in MyDrive/UGP_V7_FULL4B or restore the results folder.')

print('All 4x4 visual caches + physics/manifests found.')
print('V7 short checkpoint:',V7_SHORT_CKPT)

# %% notebook cell 4
EXPERIMENTS = [
    dict(
        terrain_type='forest',
        environment='ForestEnv',
        run='P2000',
        ttests=[28.0, 45.0, 75.0, 90.0]
    ),
    dict(
        terrain_type='swamp_marsh',
        environment='GreatMarsh',
        run='P2023',
        ttests=[47.0, 61.0, 73.0, 87.0]
    ),
    dict(
        terrain_type='rough_uneven',
        environment='SeasonalForestWinter',
        run='P2000',
        ttests=[29.0, 46.0, 64.0, 92.0]
    ),
    dict(
        terrain_type='wet_uneven_proxy',
        environment='OldTownFall',
        run='P2000',
        ttests=[32.0, 84.0, 100.0, 115.0]
    ),
]

print("EXPERIMENTS restored correctly.")
print(EXPERIMENTS[0])

# %% notebook cell 5
# ============================================================
# V7 -> V8 DRIVE REPAIR / AUTO-ORGANIZER
# Run this BEFORE the V8 mandatory-cache-check cell.
# ============================================================

from google.colab import drive
drive.mount("/content/drive", force_remount=False)

from pathlib import Path
import shutil
import zipfile
import os

MYDRIVE = Path("/content/drive/MyDrive")

ROOT = MYDRIVE / "UGP_V7_FULL4B"
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
V8 = ROOT / "V8_contactmap_standalone"

B_DIR = V8 / "B_touchdown"
C_DIR = V8 / "C_contact_field"
D_DIR = V8 / "D_counterfactual"

for d in [ROOT, CACHE, RESULTS, B_DIR, C_DIR, D_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("Expected project root:", ROOT)


# ============================================================
# 1. FILES V8 REQUIRES
# ============================================================

EXPERIMENTS = [
    ("ForestEnv", "P2000"),
    ("GreatMarsh", "P2023"),
    ("SeasonalForestWinter", "P2000"),
    ("OldTownFall", "P2000"),
]

SUFFIXES = [
    "physics.npz",
    "frame_manifest.npz",
    "v7_dinov3.npy",
    "v7_convnext.npy",
    "v7_utonia.npy",
    "v7_waft.npy",
]

REQUIRED_CACHE_NAMES = {
    f"{env}_{run}_{suffix}"
    for env, run in EXPERIMENTS
    for suffix in SUFFIXES
}


# ============================================================
# 2. SEARCH NEW DRIVE FOR THE BIG V7 CACHE ZIP
# ============================================================

print("\nSearching My Drive for V7 cache ZIP...")

zip_candidates = list(
    MYDRIVE.rglob("V7_FULL4B_RESULTS_AND_4BRANCH_CACHES*.zip")
)

if zip_candidates:
    for z in zip_candidates:
        print("FOUND ZIP:", z)
else:
    print("No matching big V7 ZIP visible through the Colab Drive mount.")


# ============================================================
# 3. EXTRACT REQUIRED CACHE FILES FROM ZIP
# ============================================================

for zpath in zip_candidates:

    print("\nInspecting archive:")
    print(zpath)

    try:
        with zipfile.ZipFile(zpath, "r") as z:

            extracted = 0

            for member in z.infolist():

                if member.is_dir():
                    continue

                basename = Path(member.filename).name

                # mandatory V7 caches
                if basename in REQUIRED_CACHE_NAMES:

                    destination = CACHE / basename

                    if destination.exists() and destination.stat().st_size > 0:
                        print("already present:", basename)
                        continue

                    print("extracting:", basename)

                    with z.open(member) as src, open(destination, "wb") as dst:
                        shutil.copyfileobj(src, dst)

                    extracted += 1

                # trained V7 prior, if archive contains it
                elif basename == "V7_PRIOR_SHORT.pt":

                    destination = RESULTS / basename

                    if not destination.exists():
                        print("extracting:", basename)

                        with z.open(member) as src, open(destination, "wb") as dst:
                            shutil.copyfileobj(src, dst)

            print("Extracted", extracted, "mandatory cache files from", zpath.name)

    except zipfile.BadZipFile:
        print("BAD ZIP:", zpath)


# ============================================================
# 4. IF CACHE FILES EXIST ELSEWHERE IN DRIVE, COPY THEM
# ============================================================

missing_now = [
    name for name in REQUIRED_CACHE_NAMES
    if not (CACHE / name).exists()
]

if missing_now:

    print("\nSome caches still missing.")
    print("Searching entire My Drive for already-extracted copies...")

    wanted = set(missing_now)

    for path in MYDRIVE.rglob("*"):

        if not path.is_file():
            continue

        if path.name in wanted:

            destination = CACHE / path.name

            # don't copy file onto itself
            if path.resolve() != destination.resolve():

                print("COPY:", path)
                print("  -> ", destination)

                shutil.copy2(path, destination)

            wanted.remove(path.name)

            if not wanted:
                break


# ============================================================
# 5. FIND/COPY IMPORTANT LOOSE CHECKPOINTS
# ============================================================

CHECKPOINT_ROUTES = {
    "V7_PRIOR_SHORT.pt": RESULTS / "V7_PRIOR_SHORT.pt",
    "V8B_TOUCHDOWN_MODEL.pt": B_DIR / "V8B_TOUCHDOWN_MODEL.pt",
    "V8C_CONTACT_FIELD.pt": C_DIR / "V8C_CONTACT_FIELD.pt",
    "V8D_SIAMESE_DELTA.pt": D_DIR / "V8D_SIAMESE_DELTA.pt",
}

print("\nSearching for model checkpoints...")

for filename, destination in CHECKPOINT_ROUTES.items():

    if destination.exists():
        print("OK:", destination)
        continue

    matches = [
        p for p in MYDRIVE.rglob(filename)
        if p.is_file()
    ]

    # avoid destination itself
    matches = [
        p for p in matches
        if p.resolve() != destination.resolve()
    ]

    if matches:

        source = matches[0]

        destination.parent.mkdir(parents=True, exist_ok=True)

        print("COPY:", source)
        print("  -> ", destination)

        shutil.copy2(source, destination)

    else:
        print("NOT FOUND:", filename)


# ============================================================
# 6. FINAL VERIFICATION
# ============================================================

print("\n")
print("=" * 78)
print("FINAL V7 -> V8 MIGRATION CHECK")
print("=" * 78)

missing = []

for env, run in EXPERIMENTS:

    print(f"\n{env}/{run}")

    for suffix in SUFFIXES:

        p = CACHE / f"{env}_{run}_{suffix}"

        if p.exists() and p.stat().st_size > 0:
            print("  OK     ", p.name)
        else:
            print("  MISSING", p.name)
            missing.append(p)


print("\nCHECKPOINTS")

for filename, destination in CHECKPOINT_ROUTES.items():

    if destination.exists():
        size_mb = destination.stat().st_size / 1024**2
        print(f"  OK      {filename}   ({size_mb:.1f} MB)")
    else:
        print(f"  MISSING {filename}")


print("\n" + "=" * 78)

if not missing:
    print("SUCCESS: ALL 24 MANDATORY V7 CACHE FILES ARE IN THE CORRECT LOCATION.")
    print()
    print("You can now rerun the V8 cache-check cell that previously crashed.")
else:
    print(f"STILL MISSING {len(missing)} MANDATORY CACHE FILES.")
    print()
    print("If all 24 are still missing, the shared folder is only a Google Drive")
    print("shortcut and Colab cannot see the real old-account files through MyDrive.")

print("=" * 78)

# %% notebook cell 7
# ======================================================================
# 3. V7-F MASTERY AUDIT FROM SAVED RESULTS
# ======================================================================
def locate_v7f_csv():
    direct=PERSIST_ROOT/'V7F_force_specialist'/'V7F_ALL_16_TESTS_ALL_SENSOR_VALUES.csv'
    if direct.exists(): return direct
    for root in [PERSIST_ROOT,PERSIST_ROOT.parent,Path('/content')]:
        for z in root.glob('V7F_FORCE_SPECIALIST_RESULTS*.zip'):
            out=_extract_member_from_zip(z,'V7F_ALL_16_TESTS_ALL_SENSOR_VALUES.csv','/content/v8_restore/v7f')
            if out: return Path(out)
    return None

V7F_CSV=locate_v7f_csv()
V7F_MASTER_FORCE_GATE=False
V7F_AUDIT=None
if V7F_CSV is None:
    print('V7-F result CSV not found. V8 can still run, but Stage-0 audit is skipped.')
else:
    d=pd.read_csv(V7F_CSV)
    force=d[d.signal.str.match(r'^(FL|FR|RL|RR)\\.F[xyz]$')].copy()
    force['leg']=force.signal.str.split('.').str[0]
    force['axis']=force.signal.str[-1]

    # Component metrics.
    comp=force.groupby('horizon_s').agg(
        component_MAE_N=('abs_error','mean'),
        component_p95_N=('abs_error',lambda s:float(np.quantile(s,.95))),
        component_max_N=('abs_error','max'),
        component_fraction_within5=('abs_error',lambda s:float((s<=FORCE_VECTOR_TARGET_N).mean()))
    ).reset_index()

    # Assemble one 3-D vector per test/horizon/leg.
    keys=['terrain_type','test_id','ttest_s','horizon_s','leg']
    pp=force.pivot_table(index=keys,columns='axis',values='prediction').reset_index()
    gg=force.pivot_table(index=keys,columns='axis',values='ground_truth').reset_index()
    v=pp[keys].copy()
    pred=pp[['x','y','z']].to_numpy(float); gt=gg[['x','y','z']].to_numpy(float)
    v['vector_error_N']=np.linalg.norm(pred-gt,axis=1)
    v['magnitude_error_N']=np.abs(np.linalg.norm(pred,axis=1)-np.linalg.norm(gt,axis=1))
    pm=np.linalg.norm(pred,axis=1); gm=np.linalg.norm(gt,axis=1)
    loaded=(pm>=10)&(gm>=10)
    cos=np.full(len(v),np.nan); cos[loaded]=np.sum(pred[loaded]*gt[loaded],axis=1)/(pm[loaded]*gm[loaded]+1e-9)
    cos=np.clip(cos,-1,1); v['direction_cosine']=cos; v['direction_angle_deg']=np.degrees(np.arccos(cos))
    v['vector_within5']=v.vector_error_N<=FORCE_VECTOR_TARGET_N
    vec=v.groupby('horizon_s').agg(
        vector_mean_N=('vector_error_N','mean'),vector_p95_N=('vector_error_N',lambda s:float(np.quantile(s,.95))),vector_max_N=('vector_error_N','max'),
        vector_fraction_within5=('vector_within5','mean'),magnitude_MAE_N=('magnitude_error_N','mean'),
        direction_angle_median_deg=('direction_angle_deg','median'),direction_angle_p95_deg=('direction_angle_deg',lambda s:float(np.nanquantile(s,.95)))
    ).reset_index()

    tau=d[d.signal.str.match(r'^(FL|FR|RL|RR)\\.tau_')].copy()
    tau_summary=tau.groupby('horizon_s').agg(tau_MAE_Nm=('abs_error','mean'),tau_p95_Nm=('abs_error',lambda s:float(np.quantile(s,.95))),tau_max_Nm=('abs_error','max')).reset_index()
    audit=comp.merge(vec,on='horizon_s').merge(tau_summary,on='horizon_s')
    V7F_AUDIT=audit
    display(audit)

    focus=audit[audit.horizon_s.isin([0.1,1.0])]
    # Literal interpretation of "maximum ±5N per leg": every evaluated 3-D leg-force vector <=5 N error.
    V7F_MASTER_FORCE_GATE=bool((focus.vector_max_N<=FORCE_VECTOR_TARGET_N).all())
    print('\nStrict ±5 N vector-error mastery gate:',V7F_MASTER_FORCE_GATE)
    if not V7F_MASTER_FORCE_GATE:
        print('Important: V7-F beats V5 in mean MAE, but it has NOT yet mastered the much stricter ±5 N-per-leg maximum-error goal.')
        print('V8-A/B can still proceed. Treat V8-E as shadow research until the contact-response gate improves.')

# %% notebook cell 8
# ============================================================
# FIX FORCE PIVOT: robustly parse FL.Fx / FL.Fy / FL.Fz
# DO NOT RERUN TRAINING
# ============================================================

import pandas as pd
import numpy as np

# Find the live dataframe containing V7/V7-F sensor predictions
required = {
    "terrain_type", "test_id", "ttest_s", "horizon_s",
    "signal", "prediction", "ground_truth"
}

candidates = []

for name, obj in list(globals().items()):
    if isinstance(obj, pd.DataFrame) and required.issubset(obj.columns):
        candidates.append((name, obj))

if not candidates:
    raise RuntimeError(
        "Could not find the V7/V7-F prediction dataframe in RAM."
    )

# Prefer the largest matching dataframe
src_name, src = max(candidates, key=lambda x: len(x[1]))

print("Using source dataframe:", src_name, src.shape)

# ------------------------------------------------------------
# Keep ONLY true per-leg 3D force signals:
# FL.Fx, FL.Fy, FL.Fz, ..., RR.Fz
# ------------------------------------------------------------

force_long = src[
    src["signal"].astype(str).str.match(
        r"^(FL|FR|RL|RR)\.F[xXyYzZ]$",
        na=False
    )
].copy()

# Extract leg and axis directly from signal string
parsed = force_long["signal"].str.extract(
    r"^(FL|FR|RL|RR)\.F([xXyYzZ])$"
)

force_long["leg"] = parsed[0]
force_long["axis"] = parsed[1].str.lower()

print("Force rows:", len(force_long))
print("Legs      :", sorted(force_long["leg"].dropna().unique()))
print("Axes      :", sorted(force_long["axis"].dropna().unique()))
print("Examples  :", force_long["signal"].head(12).tolist())

IDX = [
    "terrain_type",
    "test_id",
    "ttest_s",
    "horizon_s",
    "leg"
]

# ------------------------------------------------------------
# Prediction force vectors
# ------------------------------------------------------------

pp = (
    force_long
    .pivot_table(
        index=IDX,
        columns="axis",
        values="prediction",
        aggfunc="first"
    )
    .reset_index()
)

pp.columns.name = None

# ------------------------------------------------------------
# Ground-truth force vectors
# ------------------------------------------------------------

gg = (
    force_long
    .pivot_table(
        index=IDX,
        columns="axis",
        values="ground_truth",
        aggfunc="first"
    )
    .reset_index()
)

gg.columns.name = None

print("\nPrediction pivot columns:", pp.columns.tolist())
print("Ground-truth pivot columns:", gg.columns.tolist())

assert all(a in pp.columns for a in ["x", "y", "z"]), pp.columns.tolist()
assert all(a in gg.columns for a in ["x", "y", "z"]), gg.columns.tolist()

# Useful vector arrays if later V8 code expects them
pp["force_vector"] = list(
    pp[["x", "y", "z"]].to_numpy(dtype=np.float32)
)

gg["force_vector"] = list(
    gg[["x", "y", "z"]].to_numpy(dtype=np.float32)
)

pp["force_magnitude"] = np.linalg.norm(
    pp[["x", "y", "z"]].to_numpy(float), axis=1
)

gg["force_magnitude"] = np.linalg.norm(
    gg[["x", "y", "z"]].to_numpy(float), axis=1
)

print("\n✅ FORCE PIVOT FIXED")
print("Prediction vectors:", pp.shape)
print("GT vectors        :", gg.shape)

display(pp.head())
display(gg.head())

# %% notebook cell 9
# ============================================================
# RESTORE V7-F PREDICTION DATAFRAME INTO RAM
# No training is rerun.
# ============================================================

from pathlib import Path
import pandas as pd
import zipfile
import shutil

ROOT = Path("/content/drive/MyDrive/UGP_V7_FULL4B")

CSV_NAME = "V7F_ALL_16_TESTS_ALL_SENSOR_VALUES.csv"

# ------------------------------------------------------------
# 1. Look for the CSV anywhere in the project / MyDrive
# ------------------------------------------------------------

search_roots = [
    ROOT,
    Path("/content/drive/MyDrive"),
    Path("/content"),
]

csv_candidates = []

for root in search_roots:
    if root.exists():
        try:
            csv_candidates.extend(root.rglob(CSV_NAME))
        except Exception:
            pass

csv_candidates = [
    p for p in csv_candidates
    if p.is_file()
]

# ------------------------------------------------------------
# 2. If CSV isn't extracted, search ZIPs and extract it
# ------------------------------------------------------------

if not csv_candidates:

    print("Direct CSV not found. Searching ZIP files...")

    zip_candidates = []

    for root in search_roots:
        if root.exists():
            try:
                zip_candidates.extend(
                    root.rglob("V7F_FORCE_SPECIALIST_RESULTS*.zip")
                )
            except Exception:
                pass

    # Also inspect the big V7 archive in case the V7F CSV is inside it.
    for root in search_roots:
        if root.exists():
            try:
                zip_candidates.extend(
                    root.rglob("V7_FULL4B_RESULTS_AND_4BRANCH_CACHES*.zip")
                )
            except Exception:
                pass

    # Last fallback: inspect other ZIPs under the project
    try:
        zip_candidates.extend(ROOT.rglob("*.zip"))
    except Exception:
        pass

    # remove duplicates
    zip_candidates = list(dict.fromkeys(zip_candidates))

    restore_dir = Path("/content/v7f_restore")
    restore_dir.mkdir(parents=True, exist_ok=True)

    for zp in zip_candidates:

        try:
            with zipfile.ZipFile(zp, "r") as z:

                member = next(
                    (
                        m for m in z.namelist()
                        if Path(m).name == CSV_NAME
                    ),
                    None
                )

                if member is not None:

                    out = restore_dir / CSV_NAME

                    print("Found V7-F CSV inside:")
                    print(" ", zp)
                    print("Extracting:", member)

                    with z.open(member) as src, open(out, "wb") as dst:
                        shutil.copyfileobj(src, dst)

                    csv_candidates = [out]
                    break

        except zipfile.BadZipFile:
            pass


# ------------------------------------------------------------
# 3. Load dataframe as `d` exactly like original notebook
# ------------------------------------------------------------

if not csv_candidates:
    raise RuntimeError(
        "\nV7F_ALL_16_TESTS_ALL_SENSOR_VALUES.csv was not found.\n"
        "You need to copy either:\n"
        "  V7F_ALL_16_TESTS_ALL_SENSOR_VALUES.csv\n"
        "or\n"
        "  V7F_FORCE_SPECIALIST_RESULTS*.zip\n"
        "from the OLD account."
    )

V7F_CSV = csv_candidates[0]

print("\nUsing:")
print(V7F_CSV)

d = pd.read_csv(V7F_CSV)

# ------------------------------------------------------------
# 4. Verify exactly what the next notebook cell needs
# ------------------------------------------------------------

required = {
    "terrain_type",
    "test_id",
    "ttest_s",
    "horizon_s",
    "signal",
    "prediction",
    "ground_truth",
}

missing_cols = required - set(d.columns)

print("\nLoaded dataframe shape:", d.shape)
print("Columns:", d.columns.tolist())

if missing_cols:
    raise RuntimeError(
        f"Wrong CSV / incomplete V7-F dataframe. Missing columns: {missing_cols}"
    )

print("\n✅ V7-F prediction dataframe restored to RAM as variable `d`.")
print("Now rerun the FORCE PIVOT FIX cell.")

# %% notebook cell 11
# ======================================================================
# 4. MINIMAL V7 ARCHITECTURE + VERIFIED ANYMAL-D LEG FK
# ======================================================================
from scipy.spatial.transform import Rotation

def quat_xyzw_to_R(q):
    q=np.asarray(q,np.float64)
    return Rotation.from_quat(q).as_matrix().astype(np.float32)


def dense_camera_points(depth):
    """OpenCV pixels -> Tartan camera-NED vectors: x=fwd, y=right, z=down."""
    H,W=depth.shape
    u=np.arange(W,dtype=np.float32)[None,:].repeat(H,0)
    v=np.arange(H,dtype=np.float32)[:,None].repeat(W,1)
    Z=depth.astype(np.float32)
    x=Z
    y=(u-CX)*Z/FX
    z=(v-CY)*Z/FY
    return np.stack([x,y,z],axis=-1)


def dense_normals_camera(points,valid):
    # Central differences in organized point cloud.
    dx=np.zeros_like(points); dy=np.zeros_like(points)
    dx[:,1:-1]=points[:,2:]-points[:,:-2]
    dy[1:-1,:]=points[2:]-points[:-2]
    n=np.cross(dx,dy)
    norm=np.linalg.norm(n,axis=-1,keepdims=True)
    n=n/(norm+1e-6)
    n[~valid]=0
    return n.astype(np.float32)


def heading_frame_from_pose(pose7):
    """
    pose format: tx ty tz qx qy qz qw.
    Camera axes are forward/right/down; world is NED. Return R_local_world whose
    rows are local forward/right/up axes expressed in world NED.
    """
    Rwc=quat_xyzw_to_R(pose7[3:7])
    f=Rwc[:,0].astype(np.float64)
    fh=np.array([f[0],f[1],0.0],np.float64)
    if np.linalg.norm(fh)<1e-6: fh=np.array([1.,0.,0.])
    fh/=np.linalg.norm(fh)
    rh=np.array([-fh[1],fh[0],0.0],np.float64)
    up=np.array([0.,0.,-1.],np.float64)  # world NED -> up
    return np.stack([fh,rh,up],axis=0).astype(np.float32),Rwc

def _rpy_R(rpy):
    r,p,y=[float(v) for v in rpy]
    cr,sr=np.cos(r),np.sin(r); cp,sp=np.cos(p),np.sin(p); cy,sy=np.cos(y),np.sin(y)
    Rx=np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]],np.float64)
    Ry=np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]],np.float64)
    Rz=np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]],np.float64)
    return Rz@Ry@Rx


def _axis_R(axis,angle):
    a=np.asarray(axis,np.float64); a=a/(np.linalg.norm(a)+1e-12)
    x,y,z=a; c=np.cos(angle); ss=np.sin(angle); C=1-c
    return np.array([
        [c+x*x*C, x*y*C-z*ss, x*z*C+y*ss],
        [y*x*C+z*ss, c+y*y*C, y*z*C-x*ss],
        [z*x*C-y*ss, z*y*C+x*ss, c+z*z*C],
    ],np.float64)


class StrictURDFFK:
    def __init__(self,urdf_path,leg_prefixes=('LF','RF','LH','RH')):
        self.urdf_path=Path(urdf_path)
        root=ET.parse(self.urdf_path).getroot()
        self.links={x.attrib['name'] for x in root.findall('link')}
        self.joints={}; self.child_to_joint={}
        all_children=set()
        for je in root.findall('joint'):
            name=je.attrib['name']; typ=je.attrib.get('type','fixed')
            parent=je.find('parent').attrib['link']; child=je.find('child').attrib['link']
            org=je.find('origin')
            xyz=np.zeros(3); rpy=np.zeros(3)
            if org is not None:
                if 'xyz' in org.attrib: xyz=np.fromstring(org.attrib['xyz'],sep=' ',dtype=np.float64)
                if 'rpy' in org.attrib: rpy=np.fromstring(org.attrib['rpy'],sep=' ',dtype=np.float64)
            axis_e=je.find('axis'); axis=np.array([1.,0.,0.])
            if axis_e is not None and 'xyz' in axis_e.attrib:
                axis=np.fromstring(axis_e.attrib['xyz'],sep=' ',dtype=np.float64)
            d=dict(name=name,type=typ,parent=parent,child=child,xyz=xyz,rpy=rpy,axis=axis)
            self.joints[name]=d; self.child_to_joint[child]=d; all_children.add(child)
        roots=list(self.links-all_children)
        if len(roots)!=1: raise RuntimeError(f'ANYmal-D URDF must have one root link; got {roots}')
        self.root_link=roots[0]
        self.leg_prefixes=tuple(leg_prefixes)
        self.foot_links=[]; self.chains=[]
        for pref in self.leg_prefixes:
            # Prefer an explicit FOOT link. If several exist, select the deepest chain
            # that contains all HAA/HFE/KFE joints for this leg.
            cands=[l for l in self.links if l.upper().startswith(pref+'_') and 'FOOT' in l.upper()]
            valid=[]
            for link in cands:
                try: chain=self._chain_to_root(link)
                except Exception: continue
                names=[j['name'] for j in chain]
                required=[f'{pref}_HAA',f'{pref}_HFE',f'{pref}_KFE']
                if all(q in names for q in required): valid.append((len(chain),link,chain))
            if not valid:
                raise RuntimeError(f'Could not identify a verified {pref} foot link/chain in {self.urdf_path}. Candidates={cands}')
            _,link,chain=max(valid,key=lambda x:x[0])
            self.foot_links.append(link); self.chains.append(chain)
        expected=[f'{p}_{j}' for p in self.leg_prefixes for j in ('HAA','HFE','KFE')]
        missing=[j for j in expected if j not in self.joints]
        if missing: raise RuntimeError('Official ANYmal-D URDF is missing expected dataset joints: '+repr(missing))
        print('ANYmal-D FK root:',self.root_link)
        print('ANYmal-D FK feet:',dict(zip(self.leg_prefixes,self.foot_links)))

    def _chain_to_root(self,link):
        chain=[]; cur=link; guard=0
        while cur!=self.root_link:
            if cur not in self.child_to_joint:
                raise RuntimeError(f'No joint connects {cur} to URDF root {self.root_link}')
            j=self.child_to_joint[cur]; chain.append(j); cur=j['parent']; guard+=1
            if guard>100: raise RuntimeError('URDF cycle detected')
        return list(reversed(chain))

    @staticmethod
    def _T(R=None,t=None):
        T=np.eye(4,dtype=np.float64)
        if R is not None: T[:3,:3]=R
        if t is not None: T[:3,3]=t
        return T

    def foot_positions(self,q_4x3):
        q=np.asarray(q_4x3,np.float64)
        if q.shape!=(4,3): raise ValueError(f'Expected q shape (4,3), got {q.shape}')
        qmap={f'{p}_{j}':float(q[i,k]) for i,p in enumerate(self.leg_prefixes) for k,j in enumerate(('HAA','HFE','KFE'))}
        out=[]
        for chain in self.chains:
            T=np.eye(4,dtype=np.float64)
            for j in chain:
                T=T@self._T(_rpy_R(j['rpy']),j['xyz'])
                if j['type'] in ('revolute','continuous'):
                    T=T@self._T(_axis_R(j['axis'],qmap.get(j['name'],0.0)),None)
                elif j['type']=='prismatic':
                    T=T@self._T(None,j['axis']*qmap.get(j['name'],0.0))
                elif j['type']!='fixed':
                    raise RuntimeError('Unsupported URDF joint type in FK: '+repr(j['type']))
            out.append(T[:3,3])
        out=np.asarray(out,np.float32)
        if not np.isfinite(out).all() or np.max(np.linalg.norm(out,axis=1))>2.0:
            raise RuntimeError('ANYmal-D FK produced physically implausible foot positions: '+repr(out))
        return out


ANYMAL_D_FK=StrictURDFFK(ANYMAL_D_URDF)

class CausalHistoryEncoder(nn.Module):
    def __init__(self, input_dim, history_steps, hidden=128, layers=2, heads=4):
        super().__init__()
        self.history_steps = int(history_steps)
        self.input_proj = nn.Linear(input_dim, hidden)
        self.pos = nn.Parameter(torch.randn(1, history_steps, hidden) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden*4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
            activation='gelu',
        )
        self.tx = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        T = x.shape[1]
        x = self.input_proj(x) + self.pos[:, :T]
        mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool),
            diagonal=1
        )
        return self.norm(self.tx(x, mask=mask))[:, -1]

# ======================================================================
# 6. V7 MODEL — 4 BRANCH RESAMPLERS, 5 TERRAIN QUERIES, CAUSAL DYNAMICS
# ======================================================================

class PerceiverResampler(nn.Module):
    def __init__(self,in_dim,d=VIS_DIM,n_latents=LATENTS_PER_BRANCH,heads=8,layers=2):
        super().__init__()
        self.adapter=nn.Sequential(nn.LayerNorm(in_dim),nn.Linear(in_dim,d),nn.GELU(),nn.LayerNorm(d))
        self.latents=nn.Parameter(torch.randn(1,n_latents,d)*0.02)
        self.cross=nn.ModuleList([nn.MultiheadAttention(d,heads,batch_first=True) for _ in range(layers)])
        self.lnq=nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.lnkv=nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.ffn=nn.ModuleList([nn.Sequential(nn.LayerNorm(d),nn.Linear(d,4*d),nn.GELU(),nn.Dropout(.1),nn.Linear(4*d,d)) for _ in range(layers)])
    def forward(self,x):
        x=self.adapter(x)
        q=self.latents.expand(x.shape[0],-1,-1)
        for attn,lq,lk,ff in zip(self.cross,self.lnq,self.lnkv,self.ffn):
            a,_=attn(lq(q),lk(x),lk(x),need_weights=False); q=q+a; q=q+ff(q)
        return q

class FlowTemporalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(6,64,3,padding=1),nn.GELU(),nn.Conv1d(64,128,3,padding=1),nn.GELU())
        self.out=nn.Linear(128,256)
    def forward(self,x):
        # B,K,64,6 -> one feature per spatial token after causal history has already been constructed.
        B,K,S,C=x.shape
        y=x.permute(0,2,3,1).reshape(B*S,C,K)
        y=self.net(y)[:,:,-1]
        return self.out(y).reshape(B,S,256)

class QueryFusionBlock(nn.Module):
    def __init__(self,d=256,heads=8):
        super().__init__()
        self.qn=nn.LayerNorm(d); self.vn=nn.LayerNorm(d)
        self.cross=nn.MultiheadAttention(d,heads,batch_first=True)
        self.sn=nn.LayerNorm(d); self.selfa=nn.MultiheadAttention(d,heads,batch_first=True)
        self.ff=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,1024),nn.GELU(),nn.Dropout(.1),nn.Linear(1024,d))
    def forward(self,q,v):
        a,_=self.cross(self.qn(q),self.vn(v),self.vn(v),need_weights=False); q=q+a
        a,_=self.selfa(self.sn(q),self.sn(q),self.sn(q),need_weights=False); q=q+a
        return q+self.ff(q)

class FullFourBranchTerrainEncoder(nn.Module):
    def __init__(self,dino_dim,conv_dim,geom_dim):
        super().__init__()
        self.dino=PerceiverResampler(dino_dim)
        self.conv=PerceiverResampler(conv_dim)
        self.geom=PerceiverResampler(geom_dim)
        self.flow_temporal=FlowTemporalEncoder()
        self.flow=PerceiverResampler(256)
        self.modality=nn.Parameter(torch.randn(1,4,1,VIS_DIM)*0.02)
        self.global_query=nn.Parameter(torch.randn(1,1,VIS_DIM)*0.02)
        self.leg_query=nn.Embedding(4,VIS_DIM)
        self.state_to_query=nn.Sequential(nn.LayerNorm(24),nn.Linear(24,VIS_DIM),nn.GELU(),nn.Linear(VIS_DIM,VIS_DIM))
        self.blocks=nn.ModuleList([QueryFusionBlock() for _ in range(4)])
        self.norm=nn.LayerNorm(VIS_DIM)
    def forward(self,dino,conv,geom,flow,current_p,current_u,foot_query_kin):
        branches=[self.dino(dino),self.conv(conv),self.geom(geom),self.flow(self.flow_temporal(flow))]
        branches=[b+self.modality[:,i] for i,b in enumerate(branches)]
        visual=torch.cat(branches,dim=1)  # B,64,256
        B=current_p.shape[0]
        g=self.global_query.expand(B,-1,-1)
        u=current_u[:,None,:].expand(-1,4,-1)
        cond=torch.cat([current_p,u,foot_query_kin],dim=-1)  # B,4,24 = proprio12 + U6 + FK/planned6
        ids=torch.arange(4,device=current_p.device)[None].expand(B,-1)
        fq=self.leg_query(ids)+self.state_to_query(cond)
        q=torch.cat([g,fq],dim=1)
        for blk in self.blocks: q=blk(q,visual)
        q=self.norm(q)
        return q[:,0],q[:,1:]  # global B,256; foot B,4,256

class V7FullDynamics(nn.Module):
    def __init__(self,history_steps,n_horizons,dino_dim,conv_dim,geom_dim):
        super().__init__(); self.history_steps=int(history_steps); self.n_horizons=int(n_horizons)
        self.leg_encoder=CausalHistoryEncoder(24,history_steps,128,2,4)
        self.leg_identity=nn.Embedding(4,128)
        self.body_encoder=CausalHistoryEncoder(18,history_steps,128,2,4)
        self.terrain_encoder=FullFourBranchTerrainEncoder(dino_dim,conv_dim,geom_dim)
        self.leg_visual_fuse=nn.Sequential(nn.LayerNorm(384),nn.Linear(384,256),nn.GELU(),nn.Linear(256,256),nn.GELU(),nn.LayerNorm(256))
        self.context_net=nn.Sequential(nn.LayerNorm(1408),nn.Linear(1408,768),nn.GELU(),nn.Dropout(.1),nn.Linear(768,384),nn.GELU(),nn.LayerNorm(384))
        self.control_proj=nn.Sequential(nn.Linear(36,128),nn.GELU(),nn.LayerNorm(128))
        self.horizon_proj=nn.Sequential(nn.Linear(1,64),nn.GELU(),nn.Linear(64,128),nn.GELU(),nn.LayerNorm(128))
        self.physics_proj=nn.Sequential(nn.Linear(PI_PHYS_DIM,128),nn.GELU(),nn.Linear(128,128),nn.GELU(),nn.LayerNorm(128))
        # context 384 + control 128 + horizon 128 + physics 128 = 768
        self.binary_contact_head=nn.Sequential(nn.Linear(768,256),nn.GELU(),nn.Linear(256,4))
        self.regime_head=nn.Sequential(nn.Linear(768,256),nn.GELU(),nn.Linear(256,4*N_REGIMES))
        # q,dq 24 + body12
        self.state_body_head=nn.Sequential(nn.Linear(768,384),nn.GELU(),nn.Dropout(.1),nn.Linear(384,36))
        # Per leg: horizon ctx768 + leg fused256 + foot vision256 + pcontact1 + regime4 = 1285
        self.tau_force_head=nn.Sequential(nn.LayerNorm(1285),nn.Linear(1285,512),nn.GELU(),nn.Dropout(.1),nn.Linear(512,256),nn.GELU(),nn.Linear(256,6))
        self.force_aux_head=nn.Sequential(nn.LayerNorm(1285),nn.Linear(1285,256),nn.GELU(),nn.Linear(256,4)) # logmag + direction3

    def encode(self,p_hist,b_hist,u_hist,l_hist,e_hist,dino,conv,geom,flow,foot_query_kin):
        B=p_hist.shape[0]; T=p_hist.shape[2]
        ulegs=u_hist[:,None].expand(-1,4,-1,-1)
        li=torch.cat([p_hist,ulegs,l_hist,e_hist],dim=-1).reshape(B*4,T,24)
        lp=self.leg_encoder(li)
        ids=torch.arange(4,device=p_hist.device).repeat(B)
        lp=(lp+self.leg_identity(ids)).reshape(B,4,128)
        body=self.body_encoder(torch.cat([b_hist,u_hist],dim=-1))
        current_p=p_hist[:,:,-1,:]
        current_u=u_hist[:,-1,:]
        vg,vf=self.terrain_encoder(dino,conv,geom,flow,current_p,current_u,foot_query_kin)
        legf=self.leg_visual_fuse(torch.cat([lp,vf],dim=-1))
        ctx=self.context_net(torch.cat([legf.reshape(B,-1),body,vg],dim=-1))
        return ctx,legf,vf

    def forward(self,p_hist,b_hist,u_hist,l_hist,e_hist,dino,conv,geom,flow,foot_query_kin,future_control_features,horizon_values,physics_features):
        ctx,legf,vf=self.encode(p_hist,b_hist,u_hist,l_hist,e_hist,dino,conv,geom,flow,foot_query_kin)
        H=future_control_features.shape[1]
        if H!=self.n_horizons: raise RuntimeError(f'Expected {self.n_horizons} horizons, got {H}')
        z=torch.cat([ctx[:,None].expand(-1,H,-1),self.control_proj(future_control_features),self.horizon_proj(horizon_values),self.physics_proj(physics_features)],dim=-1)
        binary_logits=self.binary_contact_head(z)
        regime_logits=self.regime_head(z).reshape(-1,H,4,N_REGIMES)
        regime_probs=regime_logits.softmax(-1); pcontact=torch.sigmoid(binary_logits)
        sb=self.state_body_head(z)
        residual_p=torch.zeros((p_hist.shape[0],H,4,12),device=z.device,dtype=z.dtype)
        residual_p[:,:,:,0:6]=sb[...,:24].reshape(-1,H,4,6)
        residual_b=sb[...,24:]
        lf=legf[:,None].expand(-1,H,-1,-1); vf2=vf[:,None].expand(-1,H,-1,-1); zh=z[:,:,None].expand(-1,-1,4,-1)
        head_in=torch.cat([zh,lf,vf2,pcontact.unsqueeze(-1),regime_probs],dim=-1)
        tf=self.tau_force_head(head_in)
        residual_p[:,:,:,6:12]=tf
        aux=self.force_aux_head(head_in)
        aux_dir=F.normalize(aux[...,1:4],dim=-1,eps=1e-6)
        return dict(residual_p=residual_p,residual_b=residual_b,binary_contact_logits=binary_logits,
                    regime_logits=regime_logits,regime_probs=regime_probs,force_aux_logmag=aux[...,0],force_aux_dir=aux_dir)


def partial_warmstart_from_v5(model,path):
    path=Path(path)
    if not path.exists():
        print('No optional V5 checkpoint:',path); return 0
    sd=torch.load(path,map_location='cpu')
    own=model.state_dict(); copied=0
    allowed=('leg_encoder.','leg_identity.','body_encoder.','control_proj.','horizon_proj.','physics_proj.')
    load={k:v for k,v in sd.items() if k.startswith(allowed) and k in own and own[k].shape==v.shape}
    own.update(load); model.load_state_dict(own); copied=len(load)
    print('Warm-started',copied,'compatible V5 tensors from',path)
    return copied

print('V7 model classes ready.')

# %% notebook cell 12
EXPERIMENTS = [
    {
        'terrain_type': 'forest',
        'environment': 'ForestEnv',
        'run': 'P2000',
        'ttests': [28.0, 45.0, 75.0, 90.0]
    },
    {
        'terrain_type': 'swamp_marsh',
        'environment': 'GreatMarsh',
        'run': 'P2023',
        'ttests': [47.0, 61.0, 73.0, 87.0]
    },
    {
        'terrain_type': 'rough_uneven',
        'environment': 'SeasonalForestWinter',
        'run': 'P2000',
        'ttests': [29.0, 46.0, 64.0, 92.0]
    },
    {
        'terrain_type': 'wet_uneven_proxy',
        'environment': 'OldTownFall',
        'run': 'P2000',
        'ttests': [32.0, 84.0, 100.0, 115.0]
    },
]

print("EXPERIMENTS repaired")
for e in EXPERIMENTS:
    print(e['environment'], e['run'])

# %% notebook cell 13
# ======================================================================
# 5. CACHE-ONLY V7 RESTORE — NO ROSBAG, NO RAW RGB-D, NO TRAINING
# ======================================================================
def _npz_dict(path):
    z=np.load(path,allow_pickle=False)
    return {k:z[k] for k in z.files}

def _visual_bundle(env,run,manifest):
    paths={
        'DINO':CACHE_ROOT/f'{env}_{run}_v7_dinov3.npy',
        'CONV':CACHE_ROOT/f'{env}_{run}_v7_convnext.npy',
        'GEOM':CACHE_ROOT/f'{env}_{run}_v7_utonia.npy',
        'FLOW':CACHE_ROOT/f'{env}_{run}_v7_waft.npy',
    }
    out={k:np.load(p,mmap_mode='r') for k,p in paths.items()}
    out.update(manifest)
    N=len(manifest['frame_ids'])
    for k in ['DINO','CONV','GEOM','FLOW']:
        assert len(out[k])==N, (k,len(out[k]),N)
        assert np.isfinite(np.asarray(out[k][0],np.float32)).all(), k
    return out

run_records=[]
for e in EXPERIMENTS:
    env,run=e['environment'],e['run']
    data=_npz_dict(CACHE_ROOT/f'{env}_{run}_physics.npz')
    manifest=_npz_dict(CACHE_ROOT/f'{env}_{run}_frame_manifest.npz')
    visual=_visual_bundle(env,run,manifest)
    rec=dict(**e,data=data,manifest=manifest,visual=visual)
    run_records.append(rec)
    print(e['terrain_type'], 'P',data['P'].shape,'B',data['B'].shape,'U',data['U'].shape,
          '| DINO',visual['DINO'].shape,'CONV',visual['CONV'].shape,'GEOM',visual['GEOM'].shape,'FLOW',visual['FLOW'].shape)

rec0=run_records[0]
dt=float(np.median(np.diff(rec0['data']['time'])))
Hs=int(round(SHORT_CONTEXT_SEC/dt))
prior_short=V7FullDynamics(Hs,1,rec0['visual']['DINO'].shape[-1],rec0['visual']['CONV'].shape[-1],rec0['visual']['GEOM'].shape[-1]).to(DEVICE)
try:
    sd=torch.load(V7_SHORT_CKPT,map_location=DEVICE,weights_only=True)
except TypeError:
    sd=torch.load(V7_SHORT_CKPT,map_location=DEVICE)
prior_short.load_state_dict(sd,strict=True)
prior_short.eval()
for p in prior_short.parameters(): p.requires_grad_(False)

print('\nV7 short prior restored, frozen, and ready for V8-A.')
print('history steps:',Hs,'parameters:',sum(p.numel() for p in prior_short.parameters()))

# %% notebook cell 15
# ======================================================================
# V8-A — PERSISTENT FOUR-STREAM CONTACT MEMORY
# STANDALONE CELL. Uses the V7 state restored earlier in this notebook.
#
# What it does:
#   1) Reuses the trained V7 four-stream adapters (DINOv3/ConvNeXt/Utonia/WAFT).
#   2) Converts the 64 Utonia metric anchors in every frame to world coordinates.
#   3) Associates each 3-D anchor with the matching 8x8 DINO/Conv/WAFT token.
#   4) Builds a CAUSAL event memory: (world cell, time, 256-D token, geometry, confidence).
#   5) Benchmarks whether future/contact foot locations were already observed earlier.
#
# Important scientific note:
#   TartanGround provides ground-truth camera poses. For the offline foot-location
#   proxy below, the front-camera optical center is used as the robot reference
#   origin and ANYmal-D FK is expressed in the camera-heading frame. This is good
#   for the dataset memory experiment, but exact hardware deployment MUST replace
#   this with the verified camera<->base extrinsic.
# ======================================================================

import os, json, math, gc, hashlib, itertools
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

# --------------------------- preflight ---------------------------
_req = [
    'run_records','prior_short','ANYMAL_D_FK','heading_frame_from_pose',
    'CACHE_ROOT','DEVICE','RAW_GRID','FX','FY','CX','CY'
]
_missing=[x for x in _req if x not in globals()]
if _missing:
    raise RuntimeError('V8-A standalone restore is incomplete. Missing: '+repr(_missing))

# V8_ROOT is defined by the standalone configuration cell.
V8A_ROOT = V8_ROOT/'A_memory'
V8A_ROOT.mkdir(parents=True,exist_ok=True)

V8_CELL_RES_M = 0.05
V8_MEMORY_QUERY_RADIUS_M = 0.18
V8_MEMORY_KNN = 48
V8A_BATCH = 24
V8A_TOKEN_DIM = 256
V8A_MIN_CONF = 0.02

# Camera-origin -> ANYmal base-origin translation in camera-heading coordinates.
# Keep ZERO for TartanGround offline research prototype unless a verified extrinsic
# is supplied. Exact robot deployment must replace this value.
V8_BASE_IN_CAM_HEADING = np.array([0.0,0.0,0.0],np.float32)
V8_ALIGNMENT_MODE = 'camera_origin_proxy__replace_with_verified_T_C_B_for_hardware'


def _safe_norm(x,axis=-1,eps=1e-8):
    n=np.linalg.norm(x,axis=axis,keepdims=True)
    return x/np.maximum(n,eps)


def _data_camera_pose(rec,idx):
    vi=int(rec['manifest']['visual_index'][idx])
    if vi<0: return None,None
    return rec['manifest']['poses'][vi].astype(np.float64),vi


def _world_from_heading_local(local_xyz,pose7):
    # heading_frame_from_pose returns R_lw with rows = local basis in world.
    Rlw,_=heading_frame_from_pose(pose7)
    local=np.asarray(local_xyz,np.float64)
    return np.asarray(pose7[:3],np.float64) + local @ Rlw.astype(np.float64)


def _heading_local_from_world(world_xyz,pose7):
    Rlw,_=heading_frame_from_pose(pose7)
    dv=np.asarray(world_xyz,np.float64)-np.asarray(pose7[:3],np.float64)
    return dv @ Rlw.astype(np.float64).T


def v8_foot_world_proxy(rec,idx):
    """Offline dataset proxy for four foot world positions, shape (4,3)."""
    pose,vi=_data_camera_pose(rec,idx)
    if pose is None: return None
    pB=ANYMAL_D_FK.foot_positions(rec['data']['P'][idx,:,0:3]).astype(np.float32)
    # URDF: x forward, y left, z up. Camera-heading local: x forward, y right, z up.
    pl=np.stack([pB[:,0],-pB[:,1],pB[:,2]],axis=-1)+V8_BASE_IN_CAM_HEADING[None]
    return _world_from_heading_local(pl,pose).astype(np.float32)


def _project_heading_local_to_grid(local_xyz,pose7,crop_box):
    """Map Utonia heading-frame anchors to the 8x8 image-token grid."""
    local=np.asarray(local_xyz,np.float64)
    Rlw,Rwc=heading_frame_from_pose(pose7)
    world_vec=local @ Rlw.astype(np.float64)
    cam=world_vec @ Rwc.astype(np.float64)   # because world_vec = cam @ Rwc.T
    x=cam[:,0]; y=cam[:,1]; z=cam[:,2]
    u=CX + FX*y/np.maximum(x,1e-5)
    v=CY + FY*z/np.maximum(x,1e-5)
    x0,y0,x1,y1=[float(a) for a in crop_box]
    gx=np.floor((u-x0)/max(x1-x0,1.0)*RAW_GRID).astype(np.int64)
    gy=np.floor((v-y0)/max(y1-y0,1.0)*RAW_GRID).astype(np.int64)
    valid=(x>0)&(gx>=0)&(gx<RAW_GRID)&(gy>=0)&(gy<RAW_GRID)
    gx=np.clip(gx,0,RAW_GRID-1); gy=np.clip(gy,0,RAW_GRID-1)
    return gy*RAW_GRID+gx,valid


@torch.inference_mode()
def _encode_local_fourstream_batch(visual,manifest,start,end):
    """Return Bx64x256 cell-local tokens using the TRAINED V7 branch adapters."""
    enc=prior_short.terrain_encoder
    d=torch.as_tensor(np.asarray(visual['DINO'][start:end],np.float32),device=DEVICE)
    c=torch.as_tensor(np.asarray(visual['CONV'][start:end],np.float32),device=DEVICE)
    g=torch.as_tensor(np.asarray(visual['GEOM'][start:end],np.float32),device=DEVICE)
    fl=torch.as_tensor(np.asarray(visual['FLOW'][start:end],np.float32),device=DEVICE)
    de=enc.dino.adapter(d)+enc.modality[:,0]
    ce=enc.conv.adapter(c)+enc.modality[:,1]
    ge=enc.geom.adapter(g)+enc.modality[:,2]
    fe=enc.flow.adapter(enc.flow_temporal(fl))+enc.modality[:,3]
    return de,ce,ge,fe


def _build_memory_events_for_run(rec):
    env,run=rec['environment'],rec['run']
    out=V8A_ROOT/f'{env}_{run}_V8A_MEMORY_EVENTS.npz'
    meta=V8A_ROOT/f'{env}_{run}_V8A_MEMORY_META.json'
    if out.exists() and meta.exists():
        z=np.load(out,mmap_mode='r',allow_pickle=False)
        print('Reusing V8-A memory',env,run,'events=',len(z['time']))
        return dict(path=out,meta=json.loads(meta.read_text()))

    vis=rec['visual']; man=rec['manifest']; N=len(man['frame_ids'])
    all_xy=[]; all_z=[]; all_t=[]; all_tok=[]; all_n=[]; all_r=[]; all_conf=[]; all_frame=[]; all_grid=[]
    print(f'Building V8-A causal world memory {env}/{run}: {N} frames')

    for s in range(0,N,V8A_BATCH):
        e=min(N,s+V8A_BATCH)
        de,ce,ge,fe=_encode_local_fourstream_batch(vis,man,s,e)
        de=de.detach().cpu().numpy(); ce=ce.detach().cpu().numpy(); ge=ge.detach().cpu().numpy(); fe=fe.detach().cpu().numpy()
        rawg=np.asarray(vis['GEOM'][s:e],np.float32)
        rawd=np.asarray(vis['DINO'][s:e],np.float32)
        rawc=np.asarray(vis['CONV'][s:e],np.float32)
        rawf=np.asarray(vis['FLOW'][s:e],np.float32)

        for bi,j in enumerate(range(s,e)):
            # Utonia cache contract: [... learned feat ..., coord3, normal3, color3, rough1]
            local=rawg[bi,:,-10:-7]
            normal_local=rawg[bi,:,-7:-4]
            rough=rawg[bi,:,-1]
            pose=man['poses'][j]
            grid,valid=_project_heading_local_to_grid(local,pose,man['crop_boxes'][j])
            if not np.any(valid): continue
            k=np.where(valid)[0]
            gi=grid[k]
            # Spatially matched four-stream token. Geometry is point-specific; image/motion use grid token.
            tok=(de[bi,gi]+ce[bi,gi]+ge[bi,k]+fe[bi,gi])/4.0
            tok=_safe_norm(tok,axis=-1).astype(np.float16)
            world=_world_from_heading_local(local[k],pose).astype(np.float32)
            # Rotate local normals into world using heading basis.
            Rlw,_=heading_frame_from_pose(pose)
            nw=(normal_local[k] @ Rlw).astype(np.float32)
            nw=_safe_norm(nw,axis=-1).astype(np.float32)
            # confidence = average explicit validity channels + flow finite confidence proxy
            # DINO/Conv append mask-validity as their final raw channel in V7 cache.
            dc=np.clip(rawd[bi,gi,-1],0,1)
            cc=np.clip(rawc[bi,gi,-1],0,1)
            fc=np.isfinite(rawf[bi,-1,gi]).all(-1).astype(np.float32)
            conf=((dc+cc+fc)/3.0).astype(np.float32)
            keep=conf>=V8A_MIN_CONF
            if not np.any(keep): continue
            kk=np.where(keep)[0]
            all_xy.append(world[kk,:2]); all_z.append(world[kk,2]); all_t.append(np.full(len(kk),float(man['camera_times'][j]),np.float32))
            all_tok.append(tok[kk]); all_n.append(nw[kk]); all_r.append(rough[k][kk].astype(np.float32)); all_conf.append(conf[kk])
            all_frame.append(np.full(len(kk),j,np.int32)); all_grid.append(gi[kk].astype(np.int16))

        if e%120==0 or e==N:
            print(f'  {e:04d}/{N:04d}')
        del de,ce,ge,fe
        torch.cuda.empty_cache()

    xy=np.concatenate(all_xy).astype(np.float32); z=np.concatenate(all_z).astype(np.float32)
    tt=np.concatenate(all_t).astype(np.float32); tok=np.concatenate(all_tok).astype(np.float16)
    nn=np.concatenate(all_n).astype(np.float32); rr=np.concatenate(all_r).astype(np.float32)
    cf=np.concatenate(all_conf).astype(np.float32); fr=np.concatenate(all_frame).astype(np.int32); gr=np.concatenate(all_grid).astype(np.int16)
    cell=np.floor(xy/V8_CELL_RES_M).astype(np.int32)
    # Sort by time so every query can enforce strict causal lookup.
    order=np.argsort(tt,kind='stable')
    np.savez_compressed(out,xy=xy[order],z=z[order],time=tt[order],token=tok[order],normal=nn[order],rough=rr[order],
                        confidence=cf[order],frame=fr[order],grid=gr[order],cell=cell[order])
    info=dict(environment=env,run=run,n_events=int(len(order)),cell_resolution_m=V8_CELL_RES_M,
              token_dim=V8A_TOKEN_DIM,alignment_mode=V8_ALIGNMENT_MODE)
    meta.write_text(json.dumps(info,indent=2))
    print(' Saved',out,'events=',len(order))
    return dict(path=out,meta=info)


class V8MemoryIndex:
    def __init__(self,path):
        z=np.load(path,mmap_mode='r',allow_pickle=False)
        self.path=Path(path); self.xy=np.asarray(z['xy']); self.z=np.asarray(z['z']); self.time=np.asarray(z['time'])
        self.token=np.asarray(z['token']); self.normal=np.asarray(z['normal']); self.rough=np.asarray(z['rough']); self.conf=np.asarray(z['confidence'])
        self.cell=np.asarray(z['cell']); self.tree=cKDTree(self.xy)
    def query(self,xy,t,radius=V8_MEMORY_QUERY_RADIUS_M,k=V8_MEMORY_KNN):
        xy=np.asarray(xy,np.float32).reshape(2)
        ids=np.asarray(self.tree.query_ball_point(xy,r=float(radius)),dtype=np.int64)
        if ids.size==0: return None
        causal=ids[self.time[ids] <= float(t)+1e-8]
        if causal.size==0: return None
        pts=self.xy[causal]
        dd=np.linalg.norm(pts-xy[None,:],axis=1)
        age=np.maximum(0,float(t)-self.time[causal])
        # Freshness and distance both matter; confidence weakly breaks ties.
        score=age + 1.5*dd - 0.05*np.clip(self.conf[causal],0,1)
        j=int(np.argmin(score)); q=int(causal[j])
        return dict(index=q,xy=self.xy[q].astype(np.float32),z=float(self.z[q]),time=float(self.time[q]),age_s=float(age[j]),
                    distance_m=float(dd[j]),token=self.token[q].astype(np.float32),normal=self.normal[q].astype(np.float32),
                    rough=float(self.rough[q]),confidence=float(self.conf[q]),cell=self.cell[q].copy())
    def query_many(self,xy,t,radius=V8_MEMORY_QUERY_RADIUS_M):
        return [self.query(p,t,radius=radius) for p in np.asarray(xy)]


# --------------------------- build all four memories ---------------------------
v8a_memory_files={}
v8_memory={}
for rec in run_records:
    key=(rec['terrain_type'],rec['environment'],rec['run'])
    info=_build_memory_events_for_run(rec)
    v8a_memory_files[key]=info['path']
    v8_memory[key]=V8MemoryIndex(info['path'])

# --------------------------- causal memory coverage benchmark ---------------------------
rows=[]
for rec in run_records:
    key=(rec['terrain_type'],rec['environment'],rec['run']); mem=v8_memory[key]; data=rec['data']
    # 10 Hz diagnostic sampling is enough; only actual-contact legs are evaluated.
    dt=float(np.median(np.diff(data['time']))); step=max(1,int(round(0.10/dt)))
    for idx in range(0,len(data['time']),step):
        feet=v8_foot_world_proxy(rec,idx)
        if feet is None: continue
        t=float(data['time'][idx])
        for leg in range(4):
            if data['contact'][idx,leg] < 0.5: continue
            q=mem.query(feet[leg,:2],t)
            rows.append(dict(terrain_type=rec['terrain_type'],environment=rec['environment'],run=rec['run'],time=t,leg=leg,
                             found=q is not None,distance_m=np.nan if q is None else q['distance_m'],age_s=np.nan if q is None else q['age_s'],
                             seen_0p2s_earlier=False if q is None else q['age_s']>=0.2,
                             seen_0p5s_earlier=False if q is None else q['age_s']>=0.5,
                             seen_1p0s_earlier=False if q is None else q['age_s']>=1.0,
                             seen_2p0s_earlier=False if q is None else q['age_s']>=2.0))

v8a_coverage=pd.DataFrame(rows)
v8a_summary=v8a_coverage.groupby(['terrain_type']).agg(
    contact_queries=('found','size'),memory_hit_rate=('found','mean'),median_distance_m=('distance_m','median'),median_age_s=('age_s','median'),
    seen_0p5s_earlier=('seen_0p5s_earlier','mean'),seen_1p0s_earlier=('seen_1p0s_earlier','mean'),seen_2p0s_earlier=('seen_2p0s_earlier','mean')
).reset_index()

v8a_coverage.to_csv(V8A_ROOT/'V8A_CAUSAL_MEMORY_CONTACT_COVERAGE_RAW.csv',index=False)
v8a_summary.to_csv(V8A_ROOT/'V8A_CAUSAL_MEMORY_CONTACT_COVERAGE_SUMMARY.csv',index=False)

print('\nV8-A MEMORY COVERAGE')
display(v8a_summary)
print('Alignment mode:',V8_ALIGNMENT_MODE)
print('V8-A COMPLETE:',V8A_ROOT)

# %% notebook cell 16
# ======================================================================
# V8-A+ — EPISODIC KEYFRAME MEMORY + FRONT/HIND COVERAGE GATE
# ======================================================================
class V8KeyframeMemory:
    def __init__(self,rec):
        self.rec=rec
        self.time=np.asarray(rec['manifest']['camera_times'],np.float64)
        self.pose=np.asarray(rec['manifest']['poses'],np.float64)
        self.crop=np.asarray(rec['manifest']['crop_boxes'],np.int32)
    def prior_frames(self,t,max_age_s=3.0,max_frames=8):
        ids=np.where((self.time<=float(t)+1e-9)&(self.time>=float(t)-max_age_s))[0]
        if len(ids)==0: return []
        ids=ids[-max_frames:][::-1]
        return [dict(frame_index=int(i),time=float(self.time[i]),age_s=float(t-self.time[i]),pose=self.pose[i].copy(),crop_box=self.crop[i].copy()) for i in ids]

v8_keyframes={(r['terrain_type'],r['environment'],r['run']):V8KeyframeMemory(r) for r in run_records}

leg_name={0:'FL',1:'FR',2:'RL',3:'RR'}
v8a_coverage['leg_name']=v8a_coverage.leg.map(leg_name)
v8a_coverage['leg_group']=v8a_coverage.leg.map(lambda x:'front' if x<2 else 'hind')
leg_summary=v8a_coverage.groupby(['leg_group','leg_name']).agg(
    queries=('found','size'),hit_rate=('found','mean'),seen_0p5=('seen_0p5s_earlier','mean'),seen_1p0=('seen_1p0s_earlier','mean'),seen_2p0=('seen_2p0s_earlier','mean'),median_age_s=('age_s','median')
).reset_index()
display(leg_summary)

V8A_HIT=float(v8a_coverage.found.mean())
V8A_HIND_05=float(v8a_coverage[v8a_coverage.leg>=2].seen_0p5s_earlier.mean())
V8A_GATE=bool(V8A_HIT>=0.60 and V8A_HIND_05>=0.20)
print('V8-A gate:',V8A_GATE,'overall hit=',V8A_HIT,'hind seen >=0.5s earlier=',V8A_HIND_05)
print('This threshold is an engineering go/no-go diagnostic, not a published benchmark.')

# %% notebook cell 18
# ======================================================================
# V8-B — FUTURE PER-LEG TOUCHDOWN DISTRIBUTION
# Run AFTER V8-A in the SAME live Colab session.
#
# Learns, from proprioception/body/command history only:
#   p(contact_i(t+h)), mean future foot XYZ, and XYZ uncertainty
# for FL/FR/RL/RR over 0.2...2.0 s.
#
# Output coordinates are in the CURRENT camera-heading frame
# (x forward, y right, z up), so they can be transformed directly into
# the V8-A world memory. No future image or future sensor is used as input.
# ======================================================================

import json, math, copy, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

_req=['run_records','v8_foot_world_proxy','_data_camera_pose','_heading_local_from_world','V8_ROOT','DEVICE']
_missing=[x for x in _req if x not in globals()]
if _missing: raise RuntimeError('Run V8-A first. Missing: '+repr(_missing))

V8B_ROOT=V8_ROOT/'B_touchdown'
V8B_ROOT.mkdir(parents=True,exist_ok=True)
V8B_HORIZONS=np.asarray([0.2,0.4,0.6,0.8,1.0,1.2,1.5,2.0],np.float32)
V8B_CONTEXT_SEC=0.5
V8B_HOP_SEC=0.10
V8B_BATCH=32
V8B_EPOCHS=18
V8B_LR=3e-4
V8B_SEED=int(globals().get('SEED',1337))+8200

def _seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
_seed(V8B_SEED)


def _raw_leg_history(data,idx,H):
    h0=idx-H+1
    if h0<0: return None
    # Per leg: P12 + U6 + body B12 = 30 channels.
    P=data['P'][h0:idx+1].transpose(1,0,2)                    # 4,T,12
    U=np.broadcast_to(data['U'][h0:idx+1][None],(4,H,6))     # 4,T,6
    B=np.broadcast_to(data['B'][h0:idx+1][None],(4,H,12))    # 4,T,12
    return np.concatenate([P,U,B],axis=-1).astype(np.float32)


def _touch_targets(rec,idx,horizons):
    data=rec['data']; t0=float(data['time'][idx]); pose0,_=_data_camera_pose(rec,idx)
    if pose0 is None: return None
    dt=float(np.median(np.diff(data['time'])))
    C=[]; POS=[]
    for h in horizons:
        j=idx+int(round(float(h)/dt))
        if j>=len(data['time']): return None
        fw=v8_foot_world_proxy(rec,j)
        if fw is None: return None
        loc=_heading_local_from_world(fw,pose0).astype(np.float32)
        C.append(data['contact'][j].astype(np.float32)); POS.append(loc)
    return np.asarray(C,np.float32),np.asarray(POS,np.float32)  # H,4 ; H,4,3


def _build_touch_dataset():
    X=[]; C=[]; Y=[]; meta=[]; all_feat=[]
    for ri,rec in enumerate(run_records):
        data=rec['data']; dt=float(np.median(np.diff(data['time']))); H=int(round(V8B_CONTEXT_SEC/dt))
        stop=float(min(rec['ttests']))-float(V8B_HORIZONS.max())-0.05
        start=float(data['time'][0])+V8B_CONTEXT_SEC
        tt=start
        seen=set()
        while tt<=stop+1e-8:
            idx=int(np.argmin(np.abs(data['time']-tt)))
            if idx in seen: tt+=V8B_HOP_SEC; continue
            seen.add(idx)
            x=_raw_leg_history(data,idx,H); tar=_touch_targets(rec,idx,V8B_HORIZONS)
            if x is not None and tar is not None:
                c,y=tar; X.append(x); C.append(c); Y.append(y)
                meta.append((ri,idx,float(data['time'][idx])))
                all_feat.append(x.reshape(-1,30))
            tt+=V8B_HOP_SEC
    if not X: raise RuntimeError('No V8-B training windows.')
    feat=np.concatenate(all_feat,axis=0)
    mean=feat.mean(0).astype(np.float32); std=feat.std(0).astype(np.float32); std=np.maximum(std,1e-4)
    return np.asarray(X,np.float32),np.asarray(C,np.float32),np.asarray(Y,np.float32),meta,mean,std


Xb,Cb,Yb,v8b_meta,V8B_MEAN,V8B_STD=_build_touch_dataset()
print('V8-B training windows:',len(Xb),'history shape=',Xb.shape[1:],'targets=',Yb.shape)


class _TouchDS(Dataset):
    def __init__(self,X,C,Y,mean,std): self.X=X; self.C=C; self.Y=Y; self.mean=mean; self.std=std
    def __len__(self): return len(self.X)
    def __getitem__(self,i):
        x=(self.X[i]-self.mean[None,None,:])/self.std[None,None,:]
        return torch.from_numpy(x),torch.from_numpy(self.C[i]),torch.from_numpy(self.Y[i])


class V8TouchdownTransformer(nn.Module):
    def __init__(self,horizons):
        super().__init__(); self.register_buffer('horizons',torch.as_tensor(horizons,dtype=torch.float32))
        d=128
        self.inp=nn.Linear(30,d); self.leg=nn.Embedding(4,d); self.pos=nn.Parameter(torch.randn(1,128,d)*0.01)
        layer=nn.TransformerEncoderLayer(d_model=d,nhead=4,dim_feedforward=512,dropout=.1,batch_first=True,norm_first=True,activation='gelu')
        self.tx=nn.TransformerEncoder(layer,num_layers=3); self.norm=nn.LayerNorm(d)
        self.hproj=nn.Sequential(nn.Linear(1,64),nn.GELU(),nn.Linear(64,d),nn.LayerNorm(d))
        self.fuse=nn.Sequential(nn.LayerNorm(2*d),nn.Linear(2*d,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,128),nn.GELU())
        self.contact=nn.Linear(128,1); self.mu=nn.Linear(128,3); self.logstd=nn.Linear(128,3)
    def forward(self,x):
        # x B,4,T,30
        B,L,T,C=x.shape; xx=x.reshape(B*L,T,C); z=self.inp(xx)+self.pos[:,:T]
        mask=torch.triu(torch.ones(T,T,device=x.device,dtype=torch.bool),diagonal=1)
        z=self.norm(self.tx(z,mask=mask)[:,-1]).reshape(B,L,-1)
        ids=torch.arange(4,device=x.device)[None].expand(B,-1); z=z+self.leg(ids)
        H=len(self.horizons); hz=self.hproj(self.horizons.view(1,H,1).expand(B,-1,-1))
        zz=torch.cat([z[:,None].expand(-1,H,-1,-1),hz[:,:,None].expand(-1,-1,4,-1)],dim=-1)
        h=self.fuse(zz)
        return dict(contact_logits=self.contact(h).squeeze(-1),mu=self.mu(h),logstd=self.logstd(h).clamp(-4.0,1.5))


def _touch_loss(out,c,y):
    logits=out['contact_logits']; mu=out['mu']; ls=out['logstd']; var=torch.exp(2*ls)
    # Give contact positives extra weight without making swing invisible.
    posw=torch.tensor(1.6,device=logits.device)
    lc=F.binary_cross_entropy_with_logits(logits,c,pos_weight=posw)
    mask=c[...,None]
    nll=(0.5*((y-mu)**2/var + 2*ls)).sum(-1)
    lpos=(nll*c).sum()/(c.sum()+1e-6)
    # Weak all-phase trajectory supervision stabilizes the future foot cloud.
    ltrack=F.smooth_l1_loss(mu,y,beta=.08)
    return lc+0.75*lpos+0.20*ltrack,dict(contact=float(lc.detach()),pos=float(lpos.detach()),track=float(ltrack.detach()))

# Grouped temporal validation: last 15% of PRE-TEST windows from each run.
idx=np.arange(len(Xb)); val_mask=np.zeros(len(Xb),dtype=bool)
for ri in sorted(set(m[0] for m in v8b_meta)):
    ids=np.array([i for i,m in enumerate(v8b_meta) if m[0]==ri],dtype=int)
    times=np.array([v8b_meta[i][2] for i in ids],np.float64)
    cut=np.quantile(times,0.85)
    val_mask[ids[times>=cut]]=True
tr=np.where(~val_mask)[0]; va=np.where(val_mask)[0]
train_ds=_TouchDS(Xb[tr],Cb[tr],Yb[tr],V8B_MEAN,V8B_STD); val_ds=_TouchDS(Xb[va],Cb[va],Yb[va],V8B_MEAN,V8B_STD)
train_dl=DataLoader(train_ds,batch_size=V8B_BATCH,shuffle=True,num_workers=0); val_dl=DataLoader(val_ds,batch_size=64,shuffle=False,num_workers=0)

v8_touchdown_model=V8TouchdownTransformer(V8B_HORIZONS).to(DEVICE)
opt=torch.optim.AdamW(v8_touchdown_model.parameters(),lr=V8B_LR,weight_decay=1e-4)
best=None; bestv=float('inf')
for ep in range(V8B_EPOCHS):
    v8_touchdown_model.train(); tl=[]
    for x,c,y in train_dl:
        x=x.to(DEVICE); c=c.to(DEVICE); y=y.to(DEVICE); opt.zero_grad(set_to_none=True)
        o=v8_touchdown_model(x); loss,_=_touch_loss(o,c,y); loss.backward(); torch.nn.utils.clip_grad_norm_(v8_touchdown_model.parameters(),1.0); opt.step(); tl.append(float(loss.detach().cpu()))
    v8_touchdown_model.eval(); vl=[]
    with torch.inference_mode():
        for x,c,y in val_dl:
            x=x.to(DEVICE); c=c.to(DEVICE); y=y.to(DEVICE); o=v8_touchdown_model(x); loss,_=_touch_loss(o,c,y); vl.append(float(loss.cpu()))
    vm=float(np.mean(vl)); print(f'V8-B epoch {ep+1:02d}/{V8B_EPOCHS} train={np.mean(tl):.4f} val={vm:.4f}')
    if vm<bestv: bestv=vm; best=copy.deepcopy(v8_touchdown_model.state_dict())
v8_touchdown_model.load_state_dict(best); v8_touchdown_model.eval()


def v8b_predict(rec,idx):
    data=rec['data']; dt=float(np.median(np.diff(data['time']))); H=int(round(V8B_CONTEXT_SEC/dt))
    x=_raw_leg_history(data,idx,H)
    if x is None: return None
    xn=(x-V8B_MEAN[None,None,:])/V8B_STD[None,None,:]
    with torch.inference_mode():
        o=v8_touchdown_model(torch.as_tensor(xn[None],dtype=torch.float32,device=DEVICE))
    return dict(contact_prob=torch.sigmoid(o['contact_logits'])[0].cpu().numpy(),mu=o['mu'][0].cpu().numpy(),sigma=torch.exp(o['logstd'])[0].cpu().numpy())

# --------------------------- same 16 held-out diagnostic times ---------------------------
rows=[]; timing=[]
for rec in run_records:
    data=rec['data']; dt=float(np.median(np.diff(data['time'])))
    for ti,tt in enumerate(rec['ttests'],1):
        idx0=int(np.argmin(np.abs(data['time']-float(tt))))
        pred=v8b_predict(rec,idx0); tar=_touch_targets(rec,idx0,V8B_HORIZONS)
        if pred is None or tar is None: continue
        c,y=tar; p=pred['contact_prob']; mu=pred['mu']; sig=pred['sigma']
        for hi,h in enumerate(V8B_HORIZONS):
            for leg in range(4):
                err=float(np.linalg.norm(mu[hi,leg]-y[hi,leg]))
                rows.append(dict(terrain_type=rec['terrain_type'],test_index=ti,ttest=float(tt),horizon_s=float(h),leg=leg,
                                 contact_true=float(c[hi,leg]),contact_prob=float(p[hi,leg]),position_error_m=err,
                                 sigma_mean_m=float(sig[hi,leg].mean()),within_5cm=err<=.05,within_10cm=err<=.10,within_20cm=err<=.20))
        for leg in range(4):
            gt=np.where(c[:,leg]>0.5)[0]; pr=np.where(p[:,leg]>0.5)[0]
            gt_t=np.nan if len(gt)==0 else float(V8B_HORIZONS[gt[0]]); pr_t=np.nan if len(pr)==0 else float(V8B_HORIZONS[pr[0]])
            timing.append(dict(terrain_type=rec['terrain_type'],test_index=ti,leg=leg,gt_first_contact_s=gt_t,pred_first_contact_s=pr_t,
                               timing_abs_error_s=np.nan if not(np.isfinite(gt_t) and np.isfinite(pr_t)) else abs(gt_t-pr_t)))

v8b_eval=pd.DataFrame(rows); v8b_timing=pd.DataFrame(timing)
summary=[]
for h,g in v8b_eval.groupby('horizon_s'):
    yt=g.contact_true.values.astype(int); yp=(g.contact_prob.values>=.5).astype(int)
    pr,re,f1,_=precision_recall_fscore_support(yt,yp,average='binary',zero_division=0)
    contact_rows=g[g.contact_true>0.5]
    summary.append(dict(horizon_s=h,contact_precision=pr,contact_recall=re,contact_F1=f1,
                        contact_position_MAE_m=contact_rows.position_error_m.mean(),
                        Recall_5cm=contact_rows.within_5cm.mean(),Recall_10cm=contact_rows.within_10cm.mean(),Recall_20cm=contact_rows.within_20cm.mean(),
                        mean_sigma_m=contact_rows.sigma_mean_m.mean()))
v8b_summary=pd.DataFrame(summary)

v8b_eval.to_csv(V8B_ROOT/'V8B_16TEST_TOUCHDOWN_RAW.csv',index=False)
v8b_summary.to_csv(V8B_ROOT/'V8B_TOUCHDOWN_METRICS_BY_HORIZON.csv',index=False)
v8b_timing.to_csv(V8B_ROOT/'V8B_CONTACT_TIMING.csv',index=False)
torch.save(dict(state_dict=v8_touchdown_model.state_dict(),mean=V8B_MEAN,std=V8B_STD,horizons=V8B_HORIZONS),V8B_ROOT/'V8B_TOUCHDOWN_MODEL.pt')

print('\nV8-B HELD-OUT TOUCHDOWN METRICS')
display(v8b_summary)
print('Median first-contact timing error [s]:',float(v8b_timing.timing_abs_error_s.median()))
print('V8-B COMPLETE:',V8B_ROOT)

# %% notebook cell 19
# ======================================================================
# V8-B SCIENCE GATE
# ======================================================================
contact_b=v8b_summary.copy()
V8B_RECALL10=float(contact_b.Recall_10cm.mean()) if len(contact_b) else 0.0
V8B_F1=float(contact_b.contact_F1.mean()) if len(contact_b) else 0.0
V8B_GATE=bool(V8B_RECALL10>=0.50 and V8B_F1>=0.80)
print('V8-B gate:',V8B_GATE,'mean Recall@10cm=',V8B_RECALL10,'mean contact F1=',V8B_F1)
print('If this fails, do not trust precise foothold optimization yet; improve touchdown prediction first.')

# %% notebook cell 21
# ======================================================================
# V8-C — QUERYABLE LOCAL CONTACT-RESPONSE FIELD
# Run AFTER V8-A and V8-B.
#
# Learns a CELL-LOCAL probabilistic law:
#   p([tau,F]_{t+h} | terrain_cell_token, geometry, current foot/body state,
#                         command, kinematics, horizon)
# using only terrain evidence that was already observed by the current time.
#
# This is the first V8 module that can be queried at an ALTERNATIVE terrain cell.
# ======================================================================

import json, math, copy, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

_req=['run_records','v8_memory','v8_foot_world_proxy','V8_ROOT','ANYMAL_D_FK','DEVICE']
_missing=[x for x in _req if x not in globals()]
if _missing: raise RuntimeError('Run V8-A/B first. Missing: '+repr(_missing))

V8C_ROOT=V8_ROOT/'C_contact_field'; V8C_ROOT.mkdir(parents=True,exist_ok=True)
V8C_HORIZONS=np.asarray([0.1,0.3,0.5,0.7,1.0],np.float32)
V8C_HOP_SEC=0.15
V8C_BATCH=128
V8C_EPOCHS=22
V8C_LR=3e-4
V8C_SEED=int(globals().get('SEED',1337))+8300
V8C_N_EXPERTS=4

def _seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
_seed(V8C_SEED)


def v8_numeric_leg_jacobian(q3,leg,eps=1e-4):
    q3=np.asarray(q3,np.float64); J=np.zeros((3,3),np.float64)
    q4=np.zeros((4,3),np.float64); q4[leg]=q3
    p0=ANYMAL_D_FK.foot_positions(q4)[leg].astype(np.float64)
    for j in range(3):
        qq=q4.copy(); qq[leg,j]+=eps
        p=ANYMAL_D_FK.foot_positions(qq)[leg].astype(np.float64)
        J[:,j]=(p-p0)/eps
    return J.astype(np.float32)


def _current_foot_kin(rec,idx,leg):
    p=rec['data']['P'][idx,leg]
    q=p[0:3]; dq=p[3:6]
    q4=np.zeros((4,3),np.float32); q4[leg]=q
    pos=ANYMAL_D_FK.foot_positions(q4)[leg].astype(np.float32)
    J=v8_numeric_leg_jacobian(q,leg); vel=(J@dq.astype(np.float32)).astype(np.float32)
    return pos,vel


def v8c_feature_from_memory_query(rec,idx,leg,horizon,mq):
    """297-D candidate-cell feature. mq is result of V8MemoryIndex.query()."""
    data=rec['data']; p=data['P'][idx,leg]; pos,vel=_current_foot_kin(rec,idx,leg)
    geom=np.concatenate([
        np.array([mq['z']],np.float32),mq['normal'].astype(np.float32),
        np.array([mq['rough'],mq['confidence'],min(mq['age_s'],5.0)/5.0],np.float32)
    ]) # 7
    state=np.concatenate([
        p[0:9].astype(np.float32),            # q,dq,tau (NO measured force target leakage)
        data['U'][idx].astype(np.float32),    # 6
        data['B'][idx].astype(np.float32),    # 12
        pos,vel,                              # 6
        np.array([float(horizon)],np.float32)
    ])
    x=np.concatenate([mq['token'].astype(np.float32),geom,state]).astype(np.float32)
    return x


def _build_v8c_dataset():
    X=[]; Y=[]; M=[]
    for ri,rec in enumerate(run_records):
        data=rec['data']; mem=v8_memory[(rec['terrain_type'],rec['environment'],rec['run'])]
        dt=float(np.median(np.diff(data['time']))); stop=float(min(rec['ttests']))-float(V8C_HORIZONS.max())-.05
        tt=float(data['time'][0])+.6
        while tt<=stop+1e-9:
            idx=int(np.argmin(np.abs(data['time']-tt)))
            for h in V8C_HORIZONS:
                j=idx+int(round(float(h)/dt))
                if j>=len(data['time']): continue
                fw=v8_foot_world_proxy(rec,j)
                if fw is None: continue
                for leg in range(4):
                    mq=mem.query(fw[leg,:2],float(data['time'][idx]))
                    if mq is None: continue
                    x=v8c_feature_from_memory_query(rec,idx,leg,float(h),mq)
                    y=np.concatenate([data['P'][j,leg,6:9],data['P'][j,leg,9:12]]).astype(np.float32)
                    X.append(x); Y.append(y); M.append((ri,idx,j,leg,float(h),mq['age_s'],mq['distance_m']))
            tt+=V8C_HOP_SEC
    if not X: raise RuntimeError('No V8-C causal-memory training samples. V8-A map alignment may be too sparse.')
    X=np.asarray(X,np.float32); Y=np.asarray(Y,np.float32)
    xm=X.mean(0).astype(np.float32); xs=np.maximum(X.std(0),1e-4).astype(np.float32)
    ym=Y.mean(0).astype(np.float32); ys=np.maximum(Y.std(0),1e-3).astype(np.float32)
    return X,Y,M,xm,xs,ym,ys


V8C_X,V8C_Y,v8c_meta,V8C_XMEAN,V8C_XSTD,V8C_YMEAN,V8C_YSTD=_build_v8c_dataset()
print('V8-C samples:',len(V8C_X),'feature_dim=',V8C_X.shape[1])


class _V8CDS(Dataset):
    def __init__(self,idx): self.idx=np.asarray(idx)
    def __len__(self): return len(self.idx)
    def __getitem__(self,k):
        i=int(self.idx[k]); x=(V8C_X[i]-V8C_XMEAN)/V8C_XSTD; y=(V8C_Y[i]-V8C_YMEAN)/V8C_YSTD
        return torch.from_numpy(x),torch.from_numpy(y)


class V8ContactFieldMoE(nn.Module):
    def __init__(self,din,nexp=4):
        super().__init__(); self.nexp=nexp
        self.trunk=nn.Sequential(nn.LayerNorm(din),nn.Linear(din,512),nn.GELU(),nn.Dropout(.1),nn.Linear(512,256),nn.GELU(),nn.LayerNorm(256))
        self.gate=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Linear(128,nexp))
        self.experts=nn.ModuleList([nn.Sequential(nn.Linear(256,256),nn.GELU(),nn.Linear(256,12)) for _ in range(nexp)])
    def forward(self,x):
        z=self.trunk(x); logits=self.gate(z); pi=logits.softmax(-1)
        eo=torch.stack([e(z) for e in self.experts],dim=1)   # B,E,12
        mu=eo[...,:6]; ls=eo[...,6:].clamp(-4,2)
        mix=(pi[...,None]*mu).sum(1)
        # Law of total variance in normalized target units.
        var=(pi[...,None]*(torch.exp(2*ls)+mu**2)).sum(1)-mix**2
        return dict(gate_logits=logits,pi=pi,expert_mu=mu,expert_logstd=ls,mean=mix,var=var.clamp_min(1e-6))


def _v8c_loss(o,y):
    # True diagonal Gaussian mixture NLL.
    mu=o['expert_mu']; ls=o['expert_logstd']; yy=y[:,None,:]
    logp=-0.5*(((yy-mu)/torch.exp(ls))**2 + 2*ls + math.log(2*math.pi)).sum(-1)
    logmix=torch.log_softmax(o['gate_logits'],dim=-1)+logp
    nll=-torch.logsumexp(logmix,dim=-1).mean()
    # Force is the critical quantity; add point losses on mixture mean.
    hub_tau=F.smooth_l1_loss(o['mean'][:,:3],y[:,:3],beta=.4)
    hub_force=F.smooth_l1_loss(o['mean'][:,3:],y[:,3:],beta=.4)
    # Encourage non-collapsed experts without forcing uniform gating per sample.
    pbar=o['pi'].mean(0); bal=((pbar-1.0/o['pi'].shape[1])**2).mean()
    return nll+0.25*hub_tau+0.65*hub_force+0.10*bal

# Grouped temporal validation: last 15% of PRE-TEST source times within each run.
mi=np.arange(len(V8C_X)); val=np.zeros(len(V8C_X),dtype=bool)
for ri in sorted(set(m[0] for m in v8c_meta)):
    ids=np.array([i for i,m in enumerate(v8c_meta) if m[0]==ri],dtype=int)
    times=np.array([run_records[ri]['data']['time'][v8c_meta[i][1]] for i in ids],np.float64)
    cut=np.quantile(times,0.85)
    val[ids[times>=cut]]=True
tr=mi[~val]; va=mi[val]
trdl=DataLoader(_V8CDS(tr),batch_size=V8C_BATCH,shuffle=True,num_workers=0); vadl=DataLoader(_V8CDS(va),batch_size=256,shuffle=False,num_workers=0)
v8_contact_field=V8ContactFieldMoE(V8C_X.shape[1],V8C_N_EXPERTS).to(DEVICE)
opt=torch.optim.AdamW(v8_contact_field.parameters(),lr=V8C_LR,weight_decay=1e-4)
bestv=float('inf'); best=None
for ep in range(V8C_EPOCHS):
    v8_contact_field.train(); tl=[]
    for x,y in trdl:
        x=x.to(DEVICE); y=y.to(DEVICE); opt.zero_grad(set_to_none=True); o=v8_contact_field(x); l=_v8c_loss(o,y); l.backward(); torch.nn.utils.clip_grad_norm_(v8_contact_field.parameters(),1.0); opt.step(); tl.append(float(l.detach().cpu()))
    v8_contact_field.eval(); vl=[]
    with torch.inference_mode():
        for x,y in vadl:
            x=x.to(DEVICE); y=y.to(DEVICE); vl.append(float(_v8c_loss(v8_contact_field(x),y).cpu()))
    vm=float(np.mean(vl)); print(f'V8-C epoch {ep+1:02d}/{V8C_EPOCHS} train={np.mean(tl):.4f} val={vm:.4f}')
    if vm<bestv: bestv=vm; best=copy.deepcopy(v8_contact_field.state_dict())
v8_contact_field.load_state_dict(best); v8_contact_field.eval()


def v8c_predict_feature(x):
    xn=(np.asarray(x,np.float32)-V8C_XMEAN)/V8C_XSTD
    with torch.inference_mode():
        o=v8_contact_field(torch.as_tensor(xn[None],dtype=torch.float32,device=DEVICE))
    mu=(o['mean'][0].cpu().numpy()*V8C_YSTD+V8C_YMEAN).astype(np.float32)
    sd=(np.sqrt(o['var'][0].cpu().numpy())*V8C_YSTD).astype(np.float32)
    pi=o['pi'][0].cpu().numpy().astype(np.float32)
    return dict(tau=mu[:3],force=mu[3:],tau_sigma=sd[:3],force_sigma=sd[3:],expert_prob=pi)


def v8c_predict_cell(rec,idx,leg,horizon,mq):
    return v8c_predict_feature(v8c_feature_from_memory_query(rec,idx,leg,horizon,mq))

# --------------------------- held-out 16-test evaluation ---------------------------
rows=[]
for rec in run_records:
    data=rec['data']; mem=v8_memory[(rec['terrain_type'],rec['environment'],rec['run'])]; dt=float(np.median(np.diff(data['time'])))
    for ti,tt in enumerate(rec['ttests'],1):
        idx=int(np.argmin(np.abs(data['time']-float(tt))))
        for h in V8C_HORIZONS:
            j=idx+int(round(float(h)/dt))
            if j>=len(data['time']): continue
            fw=v8_foot_world_proxy(rec,j)
            if fw is None: continue
            for leg in range(4):
                mq=mem.query(fw[leg,:2],float(data['time'][idx]))
                if mq is None: continue
                pr=v8c_predict_cell(rec,idx,leg,float(h),mq); gt_tau=data['P'][j,leg,6:9]; gt_f=data['P'][j,leg,9:12]
                pred_f=np.asarray(pr['force'],np.float32); gt_f=np.asarray(gt_f,np.float32)
                pred_tau=np.asarray(pr['tau'],np.float32); gt_tau=np.asarray(gt_tau,np.float32)
                pf=float(np.linalg.norm(pred_f)); gf=float(np.linalg.norm(gt_f))
                cos=np.nan if min(pf,gf)<10.0 else float(np.clip(np.dot(pred_f,gt_f)/(pf*gf+1e-9),-1.0,1.0))
                ang=np.nan if not np.isfinite(cos) else float(np.degrees(np.arccos(cos)))
                fve=float(np.linalg.norm(pred_f-gt_f)); tve=float(np.linalg.norm(pred_tau-gt_tau))
                rows.append(dict(terrain_type=rec['terrain_type'],test_index=ti,ttest=float(tt),horizon_s=float(h),leg=leg,
                                 memory_age_s=mq['age_s'],memory_distance_m=mq['distance_m'],
                                 tau_MAE=float(np.mean(np.abs(pred_tau-gt_tau))),tau_vector_error=tve,
                                 force_MAE=float(np.mean(np.abs(pred_f-gt_f))),force_vector_error=fve,
                                 force_magnitude_error=float(abs(pf-gf)),force_direction_cosine=cos,force_direction_angle_deg=ang,
                                 force_vector_within_5N=bool(fve<=5.0),force_component_mean_within_5N=bool(np.mean(np.abs(pred_f-gt_f))<=5.0),
                                 force_sigma_mean=float(pr['force_sigma'].mean()),
                                 pred_Fx=float(pred_f[0]),pred_Fy=float(pred_f[1]),pred_Fz=float(pred_f[2]),
                                 gt_Fx=float(gt_f[0]),gt_Fy=float(gt_f[1]),gt_Fz=float(gt_f[2])))

v8c_eval=pd.DataFrame(rows)
v8c_summary=v8c_eval.groupby('horizon_s').agg(samples=('force_MAE','size'),tau_MAE=('tau_MAE','mean'),force_MAE=('force_MAE','mean'),
                                               force_vector_error=('force_vector_error','mean'),force_sigma_mean=('force_sigma_mean','mean')).reset_index()
v8c_eval.to_csv(V8C_ROOT/'V8C_16TEST_CONTACT_FIELD_RAW.csv',index=False)
v8c_summary.to_csv(V8C_ROOT/'V8C_CONTACT_FIELD_METRICS_BY_HORIZON.csv',index=False)
torch.save(dict(state_dict=v8_contact_field.state_dict(),xmean=V8C_XMEAN,xstd=V8C_XSTD,ymean=V8C_YMEAN,ystd=V8C_YSTD,horizons=V8C_HORIZONS),V8C_ROOT/'V8C_CONTACT_FIELD.pt')

print('\nV8-C HELD-OUT QUERYABLE FIELD METRICS')
display(v8c_summary)
print('V8-C COMPLETE:',V8C_ROOT)

# %% notebook cell 22
# ======================================================================
# V8-C STRICT PHYSICAL GATE: FORCE MAGNITUDE, DIRECTION, VECTOR + TORQUE
# ======================================================================
if len(v8c_eval):
    cgate=v8c_eval.groupby('horizon_s').agg(
        samples=('force_vector_error','size'),
        force_component_MAE_N=('force_MAE','mean'),
        force_vector_mean_N=('force_vector_error','mean'),
        force_vector_p95_N=('force_vector_error',lambda s:float(np.quantile(s,.95))),
        force_vector_max_N=('force_vector_error','max'),
        force_fraction_within5=('force_vector_within_5N','mean'),
        force_magnitude_MAE_N=('force_magnitude_error','mean'),
        force_direction_angle_median_deg=('force_direction_angle_deg','median'),
        force_direction_angle_p95_deg=('force_direction_angle_deg',lambda s:float(np.nanquantile(s,.95))),
        tau_component_MAE_Nm=('tau_MAE','mean'),tau_vector_mean_Nm=('tau_vector_error','mean')
    ).reset_index()
    display(cgate)
    focus=cgate[cgate.horizon_s.isin([0.1,1.0])]
    V8C_STRICT_GATE=bool(len(focus)==2 and (focus.force_vector_p95_N<=FORCE_VECTOR_TARGET_N).all())
    print('V8-C strict 95%-within-5N-vector gate:',V8C_STRICT_GATE)
else:
    cgate=pd.DataFrame(); V8C_STRICT_GATE=False

if ENFORCE_V7F_MASTER_GATE and not V7F_MASTER_FORCE_GATE:
    raise RuntimeError('ENFORCE_V7F_MASTER_GATE=True and the Stage-0 V7-F ±5 N mastery gate failed.')

# %% notebook cell 24
# ======================================================================
# V8-D — MATCHED-STATE COUNTERFACTUAL TERRAIN CHANGE HEAD
# Run AFTER V8-C.
#
# Learns the proposal's Siamese relation:
#   same/similar robot state + same leg + same horizon, different terrain token
#        -> change in [tau,F]
#
# This does NOT manufacture arbitrary counterfactual ground truth. Pairs are formed
# only from observed samples whose state/action contexts are nearest matches across
# different terrain runs.
# ======================================================================

import copy, random, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from scipy.spatial import cKDTree

_req=['V8C_X','V8C_Y','v8c_meta','V8C_XMEAN','V8C_XSTD','V8C_YSTD','run_records','V8_ROOT','DEVICE']
_missing=[x for x in _req if x not in globals()]
if _missing: raise RuntimeError('Run V8-C first. Missing: '+repr(_missing))

V8D_ROOT=V8_ROOT/'D_counterfactual'; V8D_ROOT.mkdir(parents=True,exist_ok=True)
V8D_SEED=int(globals().get('SEED',1337))+8400
V8D_MAX_PAIRS=12000
V8D_BATCH=128
V8D_EPOCHS=20
V8D_LR=3e-4
random.seed(V8D_SEED); np.random.seed(V8D_SEED); torch.manual_seed(V8D_SEED)

# First 256 dimensions are the persistent terrain token by V8-C contract.
Z=V8C_X[:,:256].astype(np.float32)
S=((V8C_X[:,256:]-V8C_XMEAN[256:])/V8C_XSTD[256:]).astype(np.float32)
Y=V8C_Y.astype(np.float32)
terrain=np.asarray([run_records[m[0]]['terrain_type'] for m in v8c_meta])
leg=np.asarray([m[3] for m in v8c_meta],np.int64)
hor=np.asarray([m[4] for m in v8c_meta],np.float32)


def _build_pairs():
    rng=np.random.default_rng(V8D_SEED); pairs=[]
    for lg in range(4):
        for h in sorted(np.unique(hor)):
            ids=np.where((leg==lg)&(np.isclose(hor,h)))[0]
            if len(ids)<20: continue
            terrs=np.unique(terrain[ids])
            trees={}
            for t in terrs:
                jj=ids[terrain[ids]==t]
                if len(jj)>=3: trees[t]=(jj,cKDTree(S[jj]))
            for ta in terrs:
                ia=ids[terrain[ids]==ta]
                if len(ia)==0: continue
                # Limit adjacent/correlated anchors per group.
                if len(ia)>350: ia=rng.choice(ia,350,replace=False)
                for a in ia:
                    best=None
                    for tb,(jj,tree) in trees.items():
                        if tb==ta: continue
                        d,k=tree.query(S[a],k=1)
                        if np.isfinite(d) and (best is None or d<best[0]): best=(float(d),int(jj[int(k)]))
                    if best is not None:
                        pairs.append((int(a),best[1],best[0]))
    if not pairs: raise RuntimeError('No cross-terrain matched-state pairs.')
    # Keep best matches first, then subsample to control size.
    pairs=sorted(pairs,key=lambda x:x[2])
    # Reject the worst half of state matches; this makes the causal claim stricter.
    cutoff=np.quantile([p[2] for p in pairs],0.55)
    pairs=[p for p in pairs if p[2]<=cutoff]
    if len(pairs)>V8D_MAX_PAIRS:
        sel=rng.choice(len(pairs),V8D_MAX_PAIRS,replace=False); pairs=[pairs[i] for i in sel]
    return pairs

V8D_PAIRS=_build_pairs()
print('V8-D matched cross-terrain pairs:',len(V8D_PAIRS),'median state distance=',float(np.median([p[2] for p in V8D_PAIRS])))


def _descriptor(a,b):
    za,zb=Z[a],Z[b]
    siam=np.concatenate([za,zb,zb-za,np.abs(zb-za),za*zb],axis=0)
    shared=.5*(S[a]+S[b])
    return np.concatenate([siam,shared],axis=0).astype(np.float32)

DX=np.asarray([_descriptor(a,b) for a,b,_ in V8D_PAIRS],np.float32)
DY=np.asarray([Y[b]-Y[a] for a,b,_ in V8D_PAIRS],np.float32)
# Normalize response delta only; token/state descriptor is already well behaved.
DYM=DY.mean(0).astype(np.float32); DYS=np.maximum(DY.std(0),1e-3).astype(np.float32)
DYN=(DY-DYM)/DYS


class _PairDS(Dataset):
    def __init__(self,ids): self.ids=np.asarray(ids)
    def __len__(self): return len(self.ids)
    def __getitem__(self,k):
        i=int(self.ids[k]); return torch.from_numpy(DX[i]),torch.from_numpy(DYN[i])


class V8SiameseDelta(nn.Module):
    def __init__(self,din):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(din),nn.Linear(din,768),nn.GELU(),nn.Dropout(.1),nn.Linear(768,384),nn.GELU(),nn.Dropout(.1),nn.Linear(384,192),nn.GELU())
        self.out=nn.Linear(192,12)
    def forward(self,x):
        y=self.out(self.net(x)); return dict(mean=y[:,:6],logstd=y[:,6:].clamp(-4,2))


def _loss(o,y):
    ls=o['logstd']; mu=o['mean']; nll=(.5*((y-mu)/torch.exp(ls))**2+ls).mean()
    # emphasize force change over torque change
    return nll+.2*F.smooth_l1_loss(mu[:,:3],y[:,:3],beta=.4)+.6*F.smooth_l1_loss(mu[:,3:],y[:,3:],beta=.4)

ids=np.arange(len(DX)); rng=np.random.default_rng(V8D_SEED); rng.shuffle(ids); cut=int(.8*len(ids)); tr,va=ids[:cut],ids[cut:]
trdl=DataLoader(_PairDS(tr),batch_size=V8D_BATCH,shuffle=True); vadl=DataLoader(_PairDS(va),batch_size=256,shuffle=False)
v8_delta_model=V8SiameseDelta(DX.shape[1]).to(DEVICE); opt=torch.optim.AdamW(v8_delta_model.parameters(),lr=V8D_LR,weight_decay=1e-4)
bestv=float('inf'); best=None
for ep in range(V8D_EPOCHS):
    v8_delta_model.train(); tl=[]
    for x,y in trdl:
        x=x.to(DEVICE); y=y.to(DEVICE); opt.zero_grad(set_to_none=True); o=v8_delta_model(x); l=_loss(o,y); l.backward(); torch.nn.utils.clip_grad_norm_(v8_delta_model.parameters(),1.0); opt.step(); tl.append(float(l.detach().cpu()))
    v8_delta_model.eval(); vl=[]
    with torch.inference_mode():
        for x,y in vadl: vl.append(float(_loss(v8_delta_model(x.to(DEVICE)),y.to(DEVICE)).cpu()))
    vm=float(np.mean(vl)); print(f'V8-D epoch {ep+1:02d}/{V8D_EPOCHS} train={np.mean(tl):.4f} val={vm:.4f}')
    if vm<bestv: bestv=vm; best=copy.deepcopy(v8_delta_model.state_dict())
v8_delta_model.load_state_dict(best); v8_delta_model.eval()


def v8d_predict_delta(token_a,token_b,state_raw_41):
    za=np.asarray(token_a,np.float32); zb=np.asarray(token_b,np.float32)
    sr=(np.asarray(state_raw_41,np.float32)-V8C_XMEAN[256:])/V8C_XSTD[256:]
    x=np.concatenate([za,zb,zb-za,np.abs(zb-za),za*zb,sr]).astype(np.float32)
    with torch.inference_mode(): o=v8_delta_model(torch.as_tensor(x[None],dtype=torch.float32,device=DEVICE))
    mu=o['mean'][0].cpu().numpy()*DYS+DYM; sd=np.exp(o['logstd'][0].cpu().numpy())*DYS
    return dict(delta_tau=mu[:3],delta_force=mu[3:],tau_sigma=sd[:3],force_sigma=sd[3:])

# --------------------------- held-out pair diagnostics ---------------------------
rows=[]
v8_delta_model.eval()
with torch.inference_mode():
    for ii in va:
        a,b,d=V8D_PAIRS[int(ii)]; x=torch.as_tensor(DX[int(ii)][None],dtype=torch.float32,device=DEVICE); o=v8_delta_model(x)
        pm=o['mean'][0].cpu().numpy()*DYS+DYM; gt=DY[int(ii)]
        fa=Y[a,3:]; fb=Y[b,3:]; pred_fb=fa+pm[3:]
        rows.append(dict(state_distance=d,delta_tau_MAE=float(np.mean(np.abs(pm[:3]-gt[:3]))),delta_force_MAE=float(np.mean(np.abs(pm[3:]-gt[3:]))),
                         delta_force_vector_error=float(np.linalg.norm(pm[3:]-gt[3:])),
                         force_magnitude_order_correct=bool(np.sign(np.linalg.norm(pred_fb)-np.linalg.norm(fa))==np.sign(np.linalg.norm(fb)-np.linalg.norm(fa)))))
v8d_eval=pd.DataFrame(rows)
v8d_summary=pd.DataFrame([dict(pairs=len(v8d_eval),median_state_distance=v8d_eval.state_distance.median(),delta_tau_MAE=v8d_eval.delta_tau_MAE.mean(),
                                  delta_force_MAE=v8d_eval.delta_force_MAE.mean(),delta_force_vector_error=v8d_eval.delta_force_vector_error.mean(),
                                  force_magnitude_ranking_accuracy=v8d_eval.force_magnitude_order_correct.mean())])
v8d_eval.to_csv(V8D_ROOT/'V8D_MATCHED_PAIR_RAW.csv',index=False); v8d_summary.to_csv(V8D_ROOT/'V8D_MATCHED_PAIR_SUMMARY.csv',index=False)
torch.save(dict(state_dict=v8_delta_model.state_dict(),dy_mean=DYM,dy_std=DYS,input_dim=DX.shape[1]),V8D_ROOT/'V8D_SIAMESE_DELTA.pt')

print('\nV8-D MATCHED-STATE COUNTERFACTUAL METRICS')
display(v8d_summary)
print('V8-D COMPLETE:',V8D_ROOT)

# %% notebook cell 26
# ======================================================================
# V8-E — CAUSAL MICRO-FOOTHOLD PLANNER (OFFLINE SHADOW MODE)
# Run AFTER V8-D.
#
# Pipeline per future leg touchdown:
#   V8-B touchdown distribution -> local candidate grid -> causal V8-A memory
#   -> geometry + numerical-Jacobian reachability filter
#   -> V8-C physical response -> V8-D terrain-change refinement
#   -> uncertainty/risk score -> top foothold(s)
#
# This cell DOES NOT command the robot. TartanGround logs contain only the executed
# trajectory, so alternative footholds are counterfactual. Results are therefore
# reported as SHADOW-PLANNER predicted risk reduction, not real navigation success.
# ======================================================================

import itertools, math, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

_req=['run_records','v8_memory','v8b_predict','v8c_predict_cell','v8c_feature_from_memory_query','v8d_predict_delta',
      '_data_camera_pose','_world_from_heading_local','_heading_local_from_world','v8_numeric_leg_jacobian','V8_ROOT']
_missing=[x for x in _req if x not in globals()]
if _missing: raise RuntimeError('Run V8-A through V8-D first. Missing: '+repr(_missing))

V8E_ROOT=V8_ROOT/'E_microplanner'; V8E_ROOT.mkdir(parents=True,exist_ok=True)
V8E_GRID_RADIUS_M=0.22
V8E_GRID_STEP_M=0.05
V8E_MAX_SLOPE_DEG=32.0
V8E_MAX_ROUGH=0.60
V8E_MAX_Q_DELTA_RAD=0.48
V8E_TOPK_PER_LEG=4

# Dimensionless planner weights. They are engineering starting values, not claimed optimal.
W_FORCE=1.00; W_TAU=0.45; W_UNC=0.40; W_SLOPE=0.30; W_ROUGH=0.25; W_OFFSET=0.20; W_MAP_AGE=0.10


def _slope_deg(normal_world):
    n=np.asarray(normal_world,np.float64); nz=np.clip(abs(n[2])/(np.linalg.norm(n)+1e-9),0,1)
    return float(np.degrees(np.arccos(nz)))


def _candidate_reachable(rec,idx,leg,nom_local,cand_world,pose0):
    cl=_heading_local_from_world(np.asarray(cand_world,np.float32),pose0)
    dl=(cl-np.asarray(nom_local,np.float32)).astype(np.float32)
    # heading-local y is RIGHT; URDF base y is LEFT.
    db=np.array([dl[0],-dl[1],dl[2]],np.float32)
    q=rec['data']['P'][idx,leg,0:3]
    J=v8_numeric_leg_jacobian(q,leg)
    dq=np.linalg.pinv(J,rcond=1e-3)@db
    return bool(np.max(np.abs(dq))<=V8E_MAX_Q_DELTA_RAD and np.linalg.norm(dq)<=0.75),dq.astype(np.float32)


def _risk(pr,mq,offset_m):
    fn=float(np.linalg.norm(pr['force'])); tn=float(np.linalg.norm(pr['tau']))
    unc=float(np.linalg.norm(pr['force_sigma']))
    slope=_slope_deg(mq['normal']); rough=float(mq['rough']); age=min(float(mq['age_s']),5.0)
    # Normalize to approximately order-one terms.
    score=(W_FORCE*fn/300.0 + W_TAU*tn/30.0 + W_UNC*unc/80.0 + W_SLOPE*slope/35.0 +
           W_ROUGH*rough + W_OFFSET*offset_m/max(V8E_GRID_RADIUS_M,1e-6) + W_MAP_AGE*age/5.0)
    return float(score),dict(force_norm=fn,tau_norm=tn,unc_norm=unc,slope_deg=slope,rough=rough,age_s=age)


def _choose_touch_horizon(pred,leg):
    hs=np.asarray(V8B_HORIZONS); p=pred['contact_prob'][:,leg]
    eligible=np.where((hs>=0.4)&(hs<=1.5)&(p>=0.50))[0]
    if len(eligible): return int(eligible[0])
    ids=np.where((hs>=0.4)&(hs<=1.5))[0]
    return int(ids[np.argmax(p[ids])])


def _plan_leg(rec,idx,leg,pred):
    pose0,_=_data_camera_pose(rec,idx); t=float(rec['data']['time'][idx]); mem=v8_memory[(rec['terrain_type'],rec['environment'],rec['run'])]
    hi=_choose_touch_horizon(pred,leg); h=float(V8B_HORIZONS[hi]); cp=float(pred['contact_prob'][hi,leg]); nom_local=pred['mu'][hi,leg]
    nom_world=_world_from_heading_local(nom_local,pose0).astype(np.float32)
    nom_mq=mem.query(nom_world[:2],t)
    if nom_mq is None: return [],dict(horizon_s=h,contact_prob=cp,reason='nominal_not_in_causal_memory')
    nom_pr=v8c_predict_cell(rec,idx,leg,h,nom_mq)
    nom_score,nom_diag=_risk(nom_pr,nom_mq,0.0)
    state_raw=v8c_feature_from_memory_query(rec,idx,leg,h,nom_mq)[256:]

    cand=[]; seen=set()
    offs=np.arange(-V8E_GRID_RADIUS_M,V8E_GRID_RADIUS_M+1e-6,V8E_GRID_STEP_M)
    for dx in offs:
        for dy in offs:
            if dx*dx+dy*dy>V8E_GRID_RADIUS_M**2+1e-9: continue
            loc=nom_local.copy(); loc[0]+=dx; loc[1]+=dy
            w=_world_from_heading_local(loc,pose0).astype(np.float32)
            mq=mem.query(w[:2],t)
            if mq is None: continue
            cell=tuple(int(x) for x in mq['cell'])
            if cell in seen: continue
            seen.add(cell)
            slope=_slope_deg(mq['normal'])
            if slope>V8E_MAX_SLOPE_DEG or mq['rough']>V8E_MAX_ROUGH: continue
            cand_world=np.array([mq['xy'][0],mq['xy'][1],mq['z']],np.float32)
            reach,dq=_candidate_reachable(rec,idx,leg,nom_local,cand_world,pose0)
            if not reach: continue
            pr=v8c_predict_cell(rec,idx,leg,h,mq)
            # Siamese terrain-change refinement around nominal terrain.
            try:
                dd=v8d_predict_delta(nom_mq['token'],mq['token'],state_raw)
                ref_tau=nom_pr['tau']+dd['delta_tau']; ref_force=nom_pr['force']+dd['delta_force']
                pr_tau=.75*pr['tau']+.25*ref_tau; pr_force=.75*pr['force']+.25*ref_force
                pr=dict(**pr); pr['tau']=pr_tau.astype(np.float32); pr['force']=pr_force.astype(np.float32)
                pr['tau_sigma']=np.sqrt(pr['tau_sigma']**2+.25**2*dd['tau_sigma']**2)
                pr['force_sigma']=np.sqrt(pr['force_sigma']**2+.25**2*dd['force_sigma']**2)
            except Exception:
                pass
            offset=float(np.linalg.norm(cand_world[:2]-nom_world[:2])); score,diag=_risk(pr,mq,offset)
            cand.append(dict(leg=leg,horizon_s=h,contact_prob=cp,world_x=float(cand_world[0]),world_y=float(cand_world[1]),world_z=float(cand_world[2]),
                             offset_m=offset,score=score,nominal_score=nom_score,predicted_score_improvement=nom_score-score,
                             force_norm=diag['force_norm'],tau_norm=diag['tau_norm'],unc_norm=diag['unc_norm'],slope_deg=diag['slope_deg'],rough=diag['rough'],
                             map_age_s=diag['age_s'],dq_norm=float(np.linalg.norm(dq)),cell_x=cell[0],cell_y=cell[1]))
    cand=sorted(cand,key=lambda r:r['score'])
    return cand[:max(V8E_TOPK_PER_LEG,1)],dict(horizon_s=h,contact_prob=cp,nominal_score=nom_score,nominal_force_norm=nom_diag['force_norm'],
                                              nominal_tau_norm=nom_diag['tau_norm'],nominal_world=nom_world)


def _support_area(points_xy):
    p=np.asarray(points_xy,np.float64)
    if len(p)<3: return 0.0
    try: return float(ConvexHull(p).volume)   # area in 2-D
    except Exception: return 0.0

# --------------------------- 16 test shadow plans ---------------------------
cand_rows=[]; plan_rows=[]
for rec in run_records:
    for ti,tt in enumerate(rec['ttests'],1):
        idx=int(np.argmin(np.abs(rec['data']['time']-float(tt))))
        pred=v8b_predict(rec,idx)
        if pred is None: continue
        tops={}; infos={}
        for leg in range(4):
            cc,inf=_plan_leg(rec,idx,leg,pred); tops[leg]=cc; infos[leg]=inf
            for rank,r in enumerate(cc,1): cand_rows.append(dict(terrain_type=rec['terrain_type'],test_index=ti,ttest=float(tt),rank=rank,**r))
        active=[lg for lg in range(4) if len(tops[lg])>0]
        if not active: continue
        # Joint beam over top candidates. Penalize collapsed support polygons and foot crowding.
        candidate_lists=[tops[lg][:3] for lg in active]
        best=None
        for combo in itertools.product(*candidate_lists):
            pts=np.array([[r['world_x'],r['world_y']] for r in combo],np.float64)
            score=sum(r['score'] for r in combo)
            # pairwise crowding penalty
            crowd=0.0
            for a,b in itertools.combinations(range(len(pts)),2):
                d=np.linalg.norm(pts[a]-pts[b]); crowd+=max(0,0.14-d)/0.14
            area=_support_area(pts); area_pen=0.0 if len(pts)<3 else max(0,0.035-area)/0.035
            joint=score+0.35*crowd+0.35*area_pen
            if best is None or joint<best[0]: best=(joint,combo,area,crowd,area_pen)
        joint,combo,area,crowd,area_pen=best
        nom=sum(infos[lg].get('nominal_score',0.0) for lg in active)
        row=dict(terrain_type=rec['terrain_type'],test_index=ti,ttest=float(tt),active_legs=len(active),joint_score=float(joint),
                 nominal_independent_score=float(nom),predicted_risk_reduction=float(nom-joint),support_area_m2=float(area),crowding_penalty=float(crowd))
        for lg,r in zip(active,combo):
            row[f'leg{lg}_x']=r['world_x']; row[f'leg{lg}_y']=r['world_y']; row[f'leg{lg}_horizon']=r['horizon_s']; row[f'leg{lg}_score']=r['score']
        plan_rows.append(row)

v8e_candidates=pd.DataFrame(cand_rows); v8e_plan=pd.DataFrame(plan_rows)
v8e_candidates.to_csv(V8E_ROOT/'V8E_ALL_CANDIDATES_TOPK.csv',index=False)
v8e_plan.to_csv(V8E_ROOT/'V8E_16TEST_SHADOW_PLANS.csv',index=False)

print('\nV8-E SHADOW-PLANNER SUMMARY')
if len(v8e_plan):
    display(pd.DataFrame([dict(tests=len(v8e_plan),mean_predicted_risk_reduction=v8e_plan.predicted_risk_reduction.mean(),
                               median_predicted_risk_reduction=v8e_plan.predicted_risk_reduction.median(),mean_support_area_m2=v8e_plan.support_area_m2.mean())]))
else:
    print('No complete shadow plans: memory/touchdown coverage gate failed. Inspect V8-A/B metrics before changing thresholds.')
print('NOTE: predicted counterfactual improvement only — not executed navigation evidence.')
print('V8-E COMPLETE:',V8E_ROOT)

# %% notebook cell 28
# ======================================================================
# V8-F — BOUNDED CONTACT POSTERIOR / RESIDUAL MEMORY + SAFE CURIOSITY
# Run AFTER V8-E.
#
# Part 1: strict causal replay of real contacts.
#   predict -> observe real [tau,F] -> update ONLY local sparse memory -> next prediction
# No backbone/contact-field weights are changed online.
#
# Part 2: value-of-information overlay on V8-E shadow foothold candidates.
# It identifies low-risk candidates that are informative enough for a hypothetical
# low-load exploratory touch. It does NOT claim those touches were executed.
# ======================================================================

import math, json, zipfile, hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

_req=['run_records','v8_memory','v8_foot_world_proxy','v8c_predict_cell','V8_ROOT','v8e_candidates']
_missing=[x for x in _req if x not in globals()]
if _missing: raise RuntimeError('Run V8-A through V8-E first. Missing: '+repr(_missing))

V8F_ROOT=V8_ROOT/'F_active_adaptation'; V8F_ROOT.mkdir(parents=True,exist_ok=True)
V8F_HORIZON=0.1
V8F_REPLAY_HOP=0.10
V8F_KERNEL_RADIUS_M=0.30
V8F_KERNEL_L_M=0.12
V8F_TOKEN_COS_MIN=0.45
V8F_MEMORY_MAX_CELLS=800
V8F_TAU_CORR_CAP=8.0
V8F_FORCE_CORR_CAP=40.0
V8F_INFO_LAMBDA=0.10


def _cos(a,b):
    a=np.asarray(a,np.float64); b=np.asarray(b,np.float64)
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))


class V8SparseResidualMemory:
    def __init__(self,max_cells=800):
        self.max_cells=max_cells; self.cells={}; self.clock=0
    def _key(self,mq): return tuple(int(x) for x in mq['cell'])
    def query(self,mq,t):
        if not self.cells: return np.zeros(6,np.float32),0.0,0
        xy=np.asarray(mq['xy']); tok=mq['token']; num=np.zeros(6,np.float64); den=0.0; nused=0; totaln=0
        for k,s in self.cells.items():
            d=float(np.linalg.norm(xy-s['xy']))
            if d>V8F_KERNEL_RADIUS_M: continue
            c=_cos(tok,s['token'])
            if c<V8F_TOKEN_COS_MIN: continue
            age=max(0,float(t)-s['last_t'])
            w=math.exp(-.5*(d/V8F_KERNEL_L_M)**2)*(max(c,0.0)**2)*math.exp(-age/12.0)
            if w<=1e-5: continue
            num+=w*s['mean']; den+=w; nused+=1; totaln+=s['n']
        if den<=1e-9: return np.zeros(6,np.float32),0.0,0
        corr=(num/den).astype(np.float32)
        # Conservative physical caps prevent a neighbor from rewriting prediction arbitrarily.
        tn=float(np.linalg.norm(corr[:3])); fn=float(np.linalg.norm(corr[3:]))
        if tn>V8F_TAU_CORR_CAP: corr[:3]*=V8F_TAU_CORR_CAP/(tn+1e-9)
        if fn>V8F_FORCE_CORR_CAP: corr[3:]*=V8F_FORCE_CORR_CAP/(fn+1e-9)
        return corr,float(den),int(totaln)
    def update(self,mq,residual,t):
        self.clock+=1; k=self._key(mq); r=np.asarray(residual,np.float64)
        if k not in self.cells:
            self.cells[k]=dict(mean=r.copy(),m2=np.zeros(6),n=1,xy=np.asarray(mq['xy'],np.float64),token=np.asarray(mq['token'],np.float32),last_t=float(t),clock=self.clock)
        else:
            s=self.cells[k]; s['n']+=1; d=r-s['mean']; s['mean']+=d/s['n']; s['m2']+=d*(r-s['mean']);
            # slow token update keeps the cell identity stable
            s['token']=(.9*s['token']+.1*np.asarray(mq['token'],np.float32)); s['last_t']=float(t); s['clock']=self.clock
        if len(self.cells)>self.max_cells:
            # evict least-recently updated cell
            old=min(self.cells.items(),key=lambda kv:kv[1]['clock'])[0]; del self.cells[old]


# --------------------------- strict causal replay ---------------------------
rows=[]
for rec in run_records:
    data=rec['data']; mem=v8_memory[(rec['terrain_type'],rec['environment'],rec['run'])]; rm=V8SparseResidualMemory(V8F_MEMORY_MAX_CELLS)
    dt=float(np.median(np.diff(data['time']))); off=int(round(V8F_HORIZON/dt)); step=max(1,int(round(V8F_REPLAY_HOP/dt)))
    # Replay starts before first diagnostic test so later tests can benefit from prior real contacts.
    for idx in range(0,len(data['time'])-off,step):
        j=idx+off; t=float(data['time'][idx]); fw=v8_foot_world_proxy(rec,j)
        if fw is None: continue
        for leg in range(4):
            # update/score only physically loaded targets; swing zero-force is already easy and would dominate statistics
            if data['contact'][j,leg] < .5: continue
            mq=mem.query(fw[leg,:2],t)
            if mq is None: continue
            base=v8c_predict_cell(rec,idx,leg,V8F_HORIZON,mq)
            y=np.concatenate([data['P'][j,leg,6:9],data['P'][j,leg,9:12]]).astype(np.float32)
            b=np.concatenate([base['tau'],base['force']]).astype(np.float32)
            corr,w,nprev=rm.query(mq,t); adapted=b+corr
            eb=np.abs(b-y); ea=np.abs(adapted-y)
            rows.append(dict(terrain_type=rec['terrain_type'],environment=rec['environment'],run=rec['run'],time=t,leg=leg,
                             prior_neighbor_contacts=nprev,kernel_weight=w,memory_age_s=mq['age_s'],
                             base_tau_MAE=float(eb[:3].mean()),adapt_tau_MAE=float(ea[:3].mean()),
                             base_force_MAE=float(eb[3:].mean()),adapt_force_MAE=float(ea[3:].mean()),
                             improved_force=bool(ea[3:].mean()<eb[3:].mean()),improved_tau=bool(ea[:3].mean()<eb[:3].mean())))
            # Strict causality: update only AFTER scoring the prediction whose target has now been observed.
            residual=y-b
            rm.update(mq,residual,float(data['time'][j]))

v8f_replay=pd.DataFrame(rows)
if len(v8f_replay)==0: raise RuntimeError('V8-F replay produced no contact samples.')

def _bucket(n):
    if n<=0:return '0'
    if n==1:return '1'
    if n<=4:return '2-4'
    return '5+'
v8f_replay['prior_contact_bucket']=v8f_replay.prior_neighbor_contacts.map(_bucket)
v8f_curve=v8f_replay.groupby('prior_contact_bucket').agg(samples=('base_force_MAE','size'),base_force_MAE=('base_force_MAE','mean'),adapt_force_MAE=('adapt_force_MAE','mean'),
                                                            base_tau_MAE=('base_tau_MAE','mean'),adapt_tau_MAE=('adapt_tau_MAE','mean'),
                                                            force_improvement_rate=('improved_force','mean'),tau_improvement_rate=('improved_tau','mean')).reset_index()
v8f_curve['force_improvement_percent']=100*(v8f_curve.base_force_MAE-v8f_curve.adapt_force_MAE)/(v8f_curve.base_force_MAE+1e-9)
v8f_curve['tau_improvement_percent']=100*(v8f_curve.base_tau_MAE-v8f_curve.adapt_tau_MAE)/(v8f_curve.base_tau_MAE+1e-9)

#------# --------------------------- safe-curiosity overlay ---------------------------
voi_rows = []

if len(v8e_candidates):

    group_cols = [
        'terrain_type',
        'test_index',
        'ttest',
        'leg',
        'horizon_s'
    ]

    for keys, g in v8e_candidates.groupby(group_cols):

        # Work on a clean sorted copy
        g = g.copy().sort_values('score')

        # IMPORTANT:
        # calculate info_gain BEFORE extracting `safest`
        g['info_gain'] = (
            0.5
            * np.log1p(
                (g['unc_norm'].clip(lower=0) / 15.0) ** 2
            )
            * (
                1.0
                + g['map_age_s'].clip(lower=0, upper=5) / 5.0
            )
        )

        # NOW this Series contains info_gain
        safest = g.iloc[0]

        # Safe exploration gate
        threshold = float(safest['score']) * 1.12 + 0.02

        safe = g[
            g['score'] <= threshold
        ].copy()

        # Numerical/logic safety fallback:
        # safest candidate itself must always be available.
        if len(safe) == 0:
            safe = g.iloc[[0]].copy()

        safe['active_score'] = (
            safe['score']
            - V8F_INFO_LAMBDA * safe['info_gain']
        )

        active = safe.sort_values('active_score').iloc[0]

        voi_rows.append(
            dict(
                terrain_type=keys[0],
                test_index=keys[1],
                ttest=keys[2],
                leg=keys[3],
                horizon_s=keys[4],

                safest_score=float(safest['score']),
                active_score=float(active['score']),

                safest_info_gain=float(safest['info_gain']),
                active_info_gain=float(active['info_gain']),

                exploration_offset_m=float(active['offset_m']),
                same_as_safest=bool(active.name == safest.name),
            )
        )

v8f_voi = pd.DataFrame(voi_rows)

print("✅ V8-F safe-curiosity overlay built.")
print("Groups:", len(v8f_voi))

if len(v8f_voi):
    display(v8f_voi.head())
readme=V8_ROOT/'README_V8_A_TO_F.txt'
readme.write_text('''V8 A-F standalone research prototype (fresh Colab restore from persisted V7 caches)\n\nA: persistent causal four-stream terrain memory\nB: probabilistic future per-leg touchdown predictor\nC: queryable local contact-response field\nD: matched-state Siamese terrain-change head\nE: causal micro-foothold planner in OFFLINE SHADOW MODE\nF: bounded causal contact adaptation + safe-curiosity shadow policy\n\nScientific limitation: TartanGround logs do not contain executed alternative footholds. E/F counterfactual planner outputs are predictions and offline replay diagnostics, not real closed-loop navigation evidence. Exact robot deployment also requires a verified camera-to-base extrinsic.\n''')

manifest=[]
for p in sorted(V8_ROOT.rglob('*')):
    if p.is_file():
        h=hashlib.sha256(p.read_bytes()).hexdigest(); manifest.append(dict(file=str(p.relative_to(V8_ROOT)),sha256=h,size_bytes=p.stat().st_size))
pd.DataFrame(manifest).to_csv(V8_ROOT/'V8_SHA256_MANIFEST.csv',index=False)
zip_path=Path('/content/V8_A_TO_F_RESULTS.zip')
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED,allowZip64=True) as z:
    for p in sorted(V8_ROOT.rglob('*')):
        if p.is_file(): z.write(p,arcname=str(p.relative_to(V8_ROOT.parent)))

print('\n'+'='*100)
print('V8 A-F EXPERIMENT STACK COMPLETE')
print('Results:',V8_ROOT)
print('ZIP    :',zip_path)
print('Interpret E as shadow counterfactual planning; do not claim executed improvement.')
print('='*100)
try:
    from google.colab import files
    files.download(str(zip_path))
except Exception as e:
    print('Automatic download skipped:',e)

# %% notebook cell 29
# ======================================================================
# FINAL RESEARCH DASHBOARD — WHAT IS ACTUALLY VALIDATED?
# ======================================================================
rows=[]
rows.append(dict(stage='V7-F strict force mastery',pass_gate=bool(V7F_MASTER_FORCE_GATE),metric='max 3D leg-force vector error <= 5 N at 0.1 & 1.0 s'))
rows.append(dict(stage='V8-A memory',pass_gate=bool(globals().get('V8A_GATE',False)),metric=f'hit={globals().get("V8A_HIT",np.nan):.3f}, hind>=0.5s={globals().get("V8A_HIND_05",np.nan):.3f}'))
rows.append(dict(stage='V8-B touchdown',pass_gate=bool(globals().get('V8B_GATE',False)),metric=f'Recall@10cm={globals().get("V8B_RECALL10",np.nan):.3f}, F1={globals().get("V8B_F1",np.nan):.3f}'))
rows.append(dict(stage='V8-C contact field strict',pass_gate=bool(globals().get('V8C_STRICT_GATE',False)),metric='p95 3D force-vector error <= 5 N at 0.1 & 1.0 s'))
if 'v8d_summary' in globals() and len(v8d_summary):
    rows.append(dict(stage='V8-D counterfactual ranking',pass_gate=bool(v8d_summary.force_magnitude_ranking_accuracy.iloc[0]>=0.65),metric=f'ranking acc={v8d_summary.force_magnitude_ranking_accuracy.iloc[0]:.3f}'))
if 'v8e_plan' in globals() and len(v8e_plan):
    rows.append(dict(stage='V8-E shadow planner',pass_gate=bool((v8e_plan.predicted_risk_reduction>0).mean()>=0.5),metric=f'positive predicted reduction fraction={(v8e_plan.predicted_risk_reduction>0).mean():.3f}'))
if 'v8f_curve' in globals() and len(v8f_curve):
    rows.append(dict(stage='V8-F adaptation',pass_gate=bool((v8f_curve.force_improvement_percent>0).any()),metric=f'best force improvement={v8f_curve.force_improvement_percent.max():.2f}%'))

dash=pd.DataFrame(rows)
display(dash)
dash.to_csv(V8_ROOT/'V8_FINAL_RESEARCH_GATE_DASHBOARD.csv',index=False)

print('\nINTERPRETATION RULE:')
if not V7F_MASTER_FORCE_GATE:
    print('- V7-F is the strongest tested V5/V7 family in mean MAE, but your ±5 N maximum-error goal is NOT yet mastered.')
if not globals().get('V8C_STRICT_GATE',False):
    print('- V8-E/F are useful research/shadow diagnostics, not a validated precision foothold controller yet.')
else:
    print('- V8-C passed the strict force gate; next evidence needed is executed simulator/hardware counterfactual footholds.')
print('- Exact robot claims still require verified camera↔base extrinsics and executed alternative foothold tests.')

# %% notebook cell 30
# ======================================================================
# V9-0 — CONTACT BELIEF MEMORY
# LIVE-RUNTIME LOCK + NON-DESTRUCTIVE PREFLIGHT
#
# IMPORTANT:
#   - Run at the VERY BOTTOM of the current V8 notebook.
#   - Does NOT retrain or modify V7/V8.
#   - Does NOT overwrite EXPERIMENTS or run_records.
# ======================================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import time

print("=" * 100)
print("V9-0 LIVE RUNTIME PREFLIGHT")
print("=" * 100)

# ----------------------------------------------------------------------
# 1. Require the valuable objects currently alive in RAM
# ----------------------------------------------------------------------

V9_REQUIRED = [
    "PERSIST_ROOT",
    "V8_ROOT",
    "DEVICE",
    "run_records",
    "prior_short",
    "v8_memory",
    "v8_foot_world_proxy",
]

missing = [x for x in V9_REQUIRED if x not in globals()]

if missing:
    raise RuntimeError(
        "DO NOT CONTINUE. Current V8 runtime is incomplete. Missing: "
        + repr(missing)
    )

print("✅ Core V7/V8 runtime objects are alive.")


# ----------------------------------------------------------------------
# 2. Create a completely separate V9 directory
# ----------------------------------------------------------------------

V9_ROOT = Path(PERSIST_ROOT) / "V9_contact_belief_memory"

V9_A_ROOT = V9_ROOT / "A_belief_memory"
V9_B_ROOT = V9_ROOT / "B_contact_events"
V9_C_ROOT = V9_ROOT / "C_residual_dynamics"
V9_CAL_ROOT = V9_ROOT / "camera_extrinsic"
V9_EVAL_ROOT = V9_ROOT / "evaluation"
V9_SNAPSHOT_ROOT = V9_ROOT / "runtime_snapshot"

for p in [
    V9_ROOT,
    V9_A_ROOT,
    V9_B_ROOT,
    V9_C_ROOT,
    V9_CAL_ROOT,
    V9_EVAL_ROOT,
    V9_SNAPSHOT_ROOT,
]:
    p.mkdir(parents=True, exist_ok=True)

print("✅ V9 workspace:")
print("   ", V9_ROOT)


# ----------------------------------------------------------------------
# 3. Freeze all inherited neural networks.
#
# V9 must initially learn ONLY small new modules.
# We do not destroy the V7/V8 representation that already exists.
# ----------------------------------------------------------------------

models_to_freeze = [
    "prior_short",
    "v8_touchdown_model",
    "v8_contact_field",
    "v8_delta_model",
]

frozen_info = {}

for name in models_to_freeze:

    if name not in globals():
        continue

    model = globals()[name]

    try:
        model.eval()

        n_params = 0

        for p in model.parameters():
            p.requires_grad_(False)
            n_params += p.numel()

        frozen_info[name] = int(n_params)

        print(
            f"✅ frozen {name:24s} "
            f"{n_params/1e6:8.3f} M parameters"
        )

    except Exception as e:
        print("⚠️ could not freeze", name, ":", repr(e))


# ----------------------------------------------------------------------
# 4. Record exact run/data state
# ----------------------------------------------------------------------

run_inventory = []

for ri, rec in enumerate(run_records):

    data = rec["data"]

    entry = dict(
        run_index=int(ri),

        terrain_type=str(rec["terrain_type"]),
        environment=str(rec["environment"]),
        run=str(rec["run"]),

        n_physics=int(len(data["time"])),

        duration_s=float(
            data["time"][-1] - data["time"][0]
        ),

        dt_s=float(
            np.median(np.diff(data["time"]))
        ),

        P_shape=list(data["P"].shape),
        B_shape=list(data["B"].shape),
        U_shape=list(data["U"].shape),

        test_times=[
            float(x) for x in rec["ttests"]
        ],
    )

    if "contact" in data:
        entry["contact_shape"] = list(data["contact"].shape)
        entry["contact_fraction"] = float(
            np.asarray(data["contact"]).mean()
        )

    run_inventory.append(entry)


# ----------------------------------------------------------------------
# 5. Inventory V8-A memory
# ----------------------------------------------------------------------

memory_inventory = []

for key, mem in v8_memory.items():

    memory_inventory.append(
        dict(
            key=[str(x) for x in key],

            events=int(len(mem.time)),

            token_dim=int(mem.token.shape[-1]),

            time_min=float(np.min(mem.time)),
            time_max=float(np.max(mem.time)),

            mean_confidence=float(
                np.mean(mem.conf)
            ),

            mean_roughness=float(
                np.mean(mem.rough)
            ),
        )
    )


# ----------------------------------------------------------------------
# 6. Save important V8 DataFrames currently in RAM
#
# This is insurance before we start experimenting.
# ----------------------------------------------------------------------

tables_to_save = [
    "v8a_coverage",
    "v8a_summary",
    "leg_summary",

    "v8b_eval",
    "v8b_summary",
    "v8b_timing",

    "v8c_eval",
    "v8c_summary",
    "cgate",

    "v8d_eval",
    "v8d_summary",

    "v8e_candidates",
    "v8e_plan",

    "v8f_replay",
    "v8f_curve",
    "v8f_voi",
]

saved_tables = {}

for name in tables_to_save:

    obj = globals().get(name, None)

    if isinstance(obj, pd.DataFrame):

        out = V9_SNAPSHOT_ROOT / f"{name}.csv"

        obj.to_csv(out, index=False)

        saved_tables[name] = dict(
            rows=int(len(obj)),
            columns=list(obj.columns),
            path=str(out),
        )

        print(
            f"✅ snapshot {name:20s}"
            f" rows={len(obj):6d}"
        )


# ----------------------------------------------------------------------
# 7. Capture GPU state
# ----------------------------------------------------------------------

gpu_info = {}

if torch.cuda.is_available():

    gpu_info = dict(
        device_name=torch.cuda.get_device_name(0),

        allocated_GB=float(
            torch.cuda.memory_allocated() / 1024**3
        ),

        reserved_GB=float(
            torch.cuda.memory_reserved() / 1024**3
        ),
    )


# ----------------------------------------------------------------------
# 8. Write permanent runtime manifest
# ----------------------------------------------------------------------

manifest = dict(
    created_unix=float(time.time()),

    v9_root=str(V9_ROOT),

    inherited_v8_root=str(V8_ROOT),

    device=str(DEVICE),

    frozen_models=frozen_info,

    runs=run_inventory,

    memories=memory_inventory,

    saved_tables=saved_tables,

    gpu=gpu_info,

    scientific_plan=dict(
        V9A="multi-observation causal world belief memory",
        V9B="joint four-leg event-time touchdown model",
        V9C="V7 baseline + ContactMap residual dynamics",
        calibration="stance-constrained camera/base extrinsic refinement",
        primary_question=(
            "Does world-anchored visual+haptic terrain memory "
            "improve future hind-leg contact prediction?"
        ),
    ),
)

manifest_path = (
    V9_SNAPSHOT_ROOT /
    "V9_RUNTIME_PREFLIGHT.json"
)

manifest_path.write_text(
    json.dumps(manifest, indent=2)
)

print()
print("=" * 100)
print("V9-0 PREFLIGHT COMPLETE")
print("=" * 100)

print("Runs:", len(run_records))

print(
    "Memory events:",
    sum(len(m.time) for m in v8_memory.values())
)

print(
    "Snapshot tables:",
    len(saved_tables)
)

print(
    "Manifest:",
    manifest_path
)

if torch.cuda.is_available():
    print(
        f"GPU: {gpu_info['device_name']} | "
        f"allocated={gpu_info['allocated_GB']:.2f} GB | "
        f"reserved={gpu_info['reserved_GB']:.2f} GB"
    )

print()
print("✅ KEEP THIS RUNTIME ALIVE.")
print("✅ V8 is now treated as a frozen experimental baseline.")
print("✅ Next: replace V8-A nearest-event lookup with V9-A belief retrieval.")

# %% notebook cell 31
# ======================================================================
# V9-A — CAUSAL MULTI-OBSERVATION TERRAIN BELIEF MEMORY
#
# Research change relative to V8-A:
#
# V8:
#       query XY -> choose ONE past event
#
# V9:
#       query XYZ
#           -> causal neighborhood
#           -> independent-frame selection
#           -> geometry/confidence weighting
#           -> semantic-consensus weighting
#           -> fused terrain belief + uncertainty
#
# NO TRAINING.
# DOES NOT MODIFY V8 MEMORY.
# ======================================================================

from pathlib import Path
from collections import defaultdict
import json
import math
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

print("=" * 100)
print("V9-A — CAUSAL MULTI-OBSERVATION TERRAIN BELIEF MEMORY")
print("=" * 100)

# ----------------------------------------------------------------------
# 0. Preconditions
# ----------------------------------------------------------------------

_req = [
    "V9_A_ROOT",
    "run_records",
    "v8_memory",
    "v8_foot_world_proxy",
]

_missing = [x for x in _req if x not in globals()]

if _missing:
    raise RuntimeError(
        "V9-A preflight failed. Missing: " + repr(_missing)
    )


# ======================================================================
# 1. RESEARCH HYPERPARAMETERS
# ======================================================================

# Spatial search.
V9A_RADIUS_M = 0.35

# Number of independent observations to fuse.
V9A_MAX_OBS = 24

# We first keep a larger candidate pool before temporal/frame diversity.
V9A_PRESELECT = 96

# Geometry scales.
V9A_SIGMA_XY_M = 0.10
V9A_SIGMA_Z_M = 0.08

# Terrain memory should not forget too rapidly.
# Hind-leg evidence is often ~0.9 s old in the present dataset.
V9A_AGE_TAU_S = 2.5

# Prevent extremely old observations dominating.
V9A_MAX_AGE_S = 6.0

# Semantic agreement strength.
V9A_TOKEN_CONSENSUS_BETA = 2.0

# Confidence floor avoids log(0).
V9A_CONF_FLOOR = 0.05

# Diagnostic rate.
V9A_DIAG_PERIOD_S = 0.10


# ======================================================================
# 2. NUMERICAL HELPERS
# ======================================================================

def _v9_safe_normalize(x, axis=-1, eps=1e-8):
    x = np.asarray(x, np.float32)
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


def _v9_weighted_mean(x, w):
    x = np.asarray(x, np.float32)
    w = np.asarray(w, np.float64)
    w = w / (w.sum() + 1e-12)
    return np.sum(x * w.reshape((-1,) + (1,) * (x.ndim - 1)), axis=0)


def _v9_effective_n(w):
    w = np.asarray(w, np.float64)
    w = w / (w.sum() + 1e-12)
    return float(1.0 / (np.sum(w ** 2) + 1e-12))


# ======================================================================
# 3. V9 BELIEF INDEX
# ======================================================================

class V9BeliefIndex:

    def __init__(self, v8_mem):

        self.source_v8 = v8_mem
        self.path = Path(v8_mem.path)

        # Reuse already-loaded V8 arrays.
        self.xy = np.asarray(v8_mem.xy)
        self.z = np.asarray(v8_mem.z)
        self.time = np.asarray(v8_mem.time)

        self.token = np.asarray(v8_mem.token)
        self.normal = np.asarray(v8_mem.normal)
        self.rough = np.asarray(v8_mem.rough)
        self.conf = np.asarray(v8_mem.conf)
        self.cell = np.asarray(v8_mem.cell)

        # Reuse V8 tree.
        self.tree = v8_mem.tree

        # V8 did not retain frame/grid IDs in RAM,
        # but they are stored in the persisted V8-A NPZ.
        zf = np.load(
            self.path,
            mmap_mode="r",
            allow_pickle=False
        )

        self.frame = (
            np.asarray(zf["frame"])
            if "frame" in zf.files
            else np.arange(len(self.time), dtype=np.int32)
        )

        self.grid = (
            np.asarray(zf["grid"])
            if "grid" in zf.files
            else np.zeros(len(self.time), dtype=np.int16)
        )

        assert len(self.xy) == len(self.frame)

    # ------------------------------------------------------------------
    # Retrieve a probabilistic terrain belief.
    # ------------------------------------------------------------------

    def query(
        self,
        xyz,
        t,
        radius=V9A_RADIUS_M,
        max_obs=V9A_MAX_OBS,
    ):

        xyz = np.asarray(xyz, np.float32).reshape(-1)

        if len(xyz) < 2:
            raise ValueError("V9 query requires at least XY.")

        qxy = xyz[:2]

        qz = (
            float(xyz[2])
            if len(xyz) >= 3 and np.isfinite(xyz[2])
            else None
        )

        t = float(t)

        # --------------------------------------------------------------
        # A. Spatial neighborhood.
        # --------------------------------------------------------------

        ids = np.asarray(
            self.tree.query_ball_point(
                qxy,
                r=float(radius)
            ),
            dtype=np.int64
        )

        if len(ids) == 0:
            return None

        # --------------------------------------------------------------
        # B. Strict causality.
        # --------------------------------------------------------------

        age = t - self.time[ids]

        keep = (
            (age >= -1e-8)
            &
            (age <= V9A_MAX_AGE_S)
        )

        ids = ids[keep]

        if len(ids) == 0:
            return None

        age = np.maximum(
            0.0,
            t - self.time[ids]
        )

        # --------------------------------------------------------------
        # C. Geometry.
        # --------------------------------------------------------------

        dxy = np.linalg.norm(
            self.xy[ids] - qxy[None, :],
            axis=1
        )

        if qz is None:
            dz = np.zeros(len(ids), np.float32)
        else:
            dz = np.abs(
                self.z[ids] - qz
            )

        conf = np.clip(
            self.conf[ids],
            V9A_CONF_FLOOR,
            1.0
        )

        # --------------------------------------------------------------
        # D. Initial log-likelihood.
        #
        # Geometry is dominant.
        # Age is a soft prior, NOT a hard nearest-time selector.
        # --------------------------------------------------------------

        logw = (
            -0.5 * (dxy / V9A_SIGMA_XY_M) ** 2
            -0.5 * (dz / V9A_SIGMA_Z_M) ** 2
            -age / V9A_AGE_TAU_S
            +0.75 * np.log(conf)
        )

        # Keep strongest raw candidates.
        if len(ids) > V9A_PRESELECT:

            ii = np.argpartition(
                -logw,
                V9A_PRESELECT - 1
            )[:V9A_PRESELECT]

            ids = ids[ii]
            age = age[ii]
            dxy = dxy[ii]
            dz = dz[ii]
            conf = conf[ii]
            logw = logw[ii]

        # --------------------------------------------------------------
        # E. Temporal diversity.
        #
        # CRITICAL:
        # 64 grid tokens from one image are NOT 64 independent
        # observations of terrain.
        #
        # Select at most one strongest token from each camera frame.
        # --------------------------------------------------------------

        best_per_frame = {}

        for local_i, global_id in enumerate(ids):

            fr = int(self.frame[global_id])

            if (
                fr not in best_per_frame
                or
                logw[local_i] >
                logw[best_per_frame[fr]]
            ):
                best_per_frame[fr] = local_i

        chosen_local = np.asarray(
            list(best_per_frame.values()),
            dtype=np.int64
        )

        # Sort strongest first.
        chosen_local = chosen_local[
            np.argsort(-logw[chosen_local])
        ]

        chosen_local = chosen_local[:max_obs]

        ids = ids[chosen_local]
        age = age[chosen_local]
        dxy = dxy[chosen_local]
        dz = dz[chosen_local]
        conf = conf[chosen_local]
        logw = logw[chosen_local]

        if len(ids) == 0:
            return None

        # --------------------------------------------------------------
        # F. First-pass normalized weights.
        # --------------------------------------------------------------

        logw = logw - np.max(logw)

        w0 = np.exp(logw)

        w0 = w0 / (
            np.sum(w0) + 1e-12
        )

        # --------------------------------------------------------------
        # G. Semantic consistency.
        #
        # Do not blindly average conflicting visual observations.
        #
        # First build a provisional semantic center,
        # then down-weight observations inconsistent with that center.
        # --------------------------------------------------------------

        tok = np.asarray(
            self.token[ids],
            np.float32
        )

        tok_unit = _v9_safe_normalize(
            tok,
            axis=1
        )

        provisional = np.sum(
            tok_unit * w0[:, None],
            axis=0
        )

        provisional = _v9_safe_normalize(
            provisional
        )

        cosine = np.clip(
            tok_unit @ provisional,
            -1.0,
            1.0
        )

        semantic_factor = np.exp(
            V9A_TOKEN_CONSENSUS_BETA *
            (cosine - 1.0)
        )

        w = (
            w0 *
            semantic_factor
        )

        w = w / (
            np.sum(w) + 1e-12
        )

        # --------------------------------------------------------------
        # H. Fused terrain belief.
        # --------------------------------------------------------------

        belief_token = np.sum(
            tok * w[:, None],
            axis=0
        ).astype(np.float32)

        belief_xy = np.sum(
            self.xy[ids] * w[:, None],
            axis=0
        ).astype(np.float32)

        belief_z = float(
            np.sum(
                self.z[ids] * w
            )
        )

        belief_rough = float(
            np.sum(
                self.rough[ids] * w
            )
        )

        belief_confidence = float(
            np.sum(
                self.conf[ids] * w
            )
        )

        # --------------------------------------------------------------
        # I. Surface normal fusion.
        #
        # Orient normals upward before averaging.
        # --------------------------------------------------------------

        normals = np.asarray(
            self.normal[ids],
            np.float32
        ).copy()

        flip = normals[:, 2] < 0
        normals[flip] *= -1.0

        nmean = np.sum(
            normals * w[:, None],
            axis=0
        )

        nmean = _v9_safe_normalize(
            nmean
        ).astype(np.float32)

        normal_cos = np.clip(
            _v9_safe_normalize(normals, axis=1)
            @ nmean,
            -1.0,
            1.0
        )

        # --------------------------------------------------------------
        # J. Epistemic / geometric uncertainty.
        # --------------------------------------------------------------

        fused_unit = _v9_safe_normalize(
            belief_token
        )

        token_cos = np.clip(
            tok_unit @ fused_unit,
            -1.0,
            1.0
        )

        token_dispersion = float(
            np.sum(
                w * (1.0 - token_cos)
            )
        )

        z_variance = float(
            np.sum(
                w *
                (self.z[ids] - belief_z) ** 2
            )
        )

        xy_variance = float(
            np.sum(
                w *
                np.sum(
                    (
                        self.xy[ids]
                        - belief_xy[None, :]
                    ) ** 2,
                    axis=1
                )
            )
        )

        normal_dispersion = float(
            np.sum(
                w *
                (1.0 - normal_cos)
            )
        )

        effective_n = _v9_effective_n(w)

        # --------------------------------------------------------------
        # K. Return complete belief object.
        # --------------------------------------------------------------

        best = int(
            ids[int(np.argmax(w))]
        )

        return dict(

            # Fused physical location.
            xy=belief_xy,
            z=belief_z,

            # Fused representation.
            token=belief_token,
            normal=nmean,
            rough=belief_rough,
            confidence=belief_confidence,

            # Observation statistics.
            n_obs=int(len(ids)),
            effective_n=float(effective_n),

            weighted_age_s=float(
                np.sum(w * age)
            ),

            min_age_s=float(
                np.min(age)
            ),

            max_age_s=float(
                np.max(age)
            ),

            weighted_distance_m=float(
                np.sum(w * dxy)
            ),

            nearest_distance_m=float(
                np.min(dxy)
            ),

            weighted_z_error_m=float(
                np.sum(w * dz)
            ),

            # Uncertainty.
            token_dispersion=token_dispersion,

            z_std_m=float(
                np.sqrt(
                    max(z_variance, 0.0)
                )
            ),

            xy_std_m=float(
                np.sqrt(
                    max(xy_variance, 0.0)
                )
            ),

            normal_dispersion=normal_dispersion,

            # Debug/provenance.
            source_ids=ids.astype(np.int64),
            source_frames=self.frame[ids].astype(np.int32),
            weights=w.astype(np.float32),

            strongest_id=best,

            strongest_frame=int(
                self.frame[best]
            ),
        )


# ======================================================================
# 4. BUILD V9 MEMORY WRAPPERS
# ======================================================================

v9_belief_memory = {}

for key, mem in v8_memory.items():

    v9_belief_memory[key] = V9BeliefIndex(mem)

    print(
        "✅ V9 belief index:",
        key,
        "events=",
        len(mem.time)
    )


# ======================================================================
# 5. CAUSAL CONTACT-POINT DIAGNOSTIC
#
# This does NOT claim predictive improvement.
# It checks whether V9 produces:
#   - high causal coverage
#   - multiple independent observations
#   - sensible uncertainty
#   - useful hind-leg prior memory
# ======================================================================

diag_rows = []

for rec in run_records:

    key = (
        rec["terrain_type"],
        rec["environment"],
        rec["run"]
    )

    v8mem = v8_memory[key]
    v9mem = v9_belief_memory[key]

    data = rec["data"]

    dt = float(
        np.median(
            np.diff(data["time"])
        )
    )

    step = max(
        1,
        int(
            round(
                V9A_DIAG_PERIOD_S / dt
            )
        )
    )

    print(
        "\nDiagnostic:",
        rec["terrain_type"]
    )

    for idx in range(
        0,
        len(data["time"]),
        step
    ):

        feet = v8_foot_world_proxy(
            rec,
            idx
        )

        if feet is None:
            continue

        t = float(
            data["time"][idx]
        )

        for leg in range(4):

            # Evaluate real contacts only.
            if (
                data["contact"][idx, leg]
                < 0.5
            ):
                continue

            xyz = np.asarray(
                feet[leg],
                np.float32
            )

            # Existing V8 baseline.
            q8 = v8mem.query(
                xyz[:2],
                t
            )

            # New V9 belief.
            q9 = v9mem.query(
                xyz,
                t
            )

            diag_rows.append(
                dict(
                    terrain_type=rec["terrain_type"],
                    environment=rec["environment"],
                    run=rec["run"],

                    time=t,
                    leg=int(leg),

                    leg_group=(
                        "front"
                        if leg < 2
                        else "hind"
                    ),

                    v8_found=bool(
                        q8 is not None
                    ),

                    v8_distance_m=(
                        np.nan
                        if q8 is None
                        else float(
                            q8["distance_m"]
                        )
                    ),

                    v8_age_s=(
                        np.nan
                        if q8 is None
                        else float(
                            q8["age_s"]
                        )
                    ),

                    v9_found=bool(
                        q9 is not None
                    ),

                    v9_n_obs=(
                        0
                        if q9 is None
                        else int(
                            q9["n_obs"]
                        )
                    ),

                    v9_effective_n=(
                        0.0
                        if q9 is None
                        else float(
                            q9["effective_n"]
                        )
                    ),

                    v9_weighted_distance_m=(
                        np.nan
                        if q9 is None
                        else float(
                            q9["weighted_distance_m"]
                        )
                    ),

                    v9_nearest_distance_m=(
                        np.nan
                        if q9 is None
                        else float(
                            q9["nearest_distance_m"]
                        )
                    ),

                    v9_weighted_age_s=(
                        np.nan
                        if q9 is None
                        else float(
                            q9["weighted_age_s"]
                        )
                    ),

                    v9_min_age_s=(
                        np.nan
                        if q9 is None
                        else float(
                            q9["min_age_s"]
                        )
                    ),

                    v9_z_std_m=(
                        np.nan
                        if q9 is None
                        else float(
                            q9["z_std_m"]
                        )
                    ),

                    v9_xy_std_m=(
                        np.nan
                        if q9 is None
                        else float(
                            q9["xy_std_m"]
                        )
                    ),

                    v9_token_dispersion=(
                        np.nan
                        if q9 is None
                        else float(
                            q9["token_dispersion"]
                        )
                    ),

                    v9_normal_dispersion=(
                        np.nan
                        if q9 is None
                        else float(
                            q9["normal_dispersion"]
                        )
                    ),

                    # Was terrain observed sufficiently earlier
                    # to represent genuine predictive memory?
                    v9_seen_0p5s_earlier=(
                        False
                        if q9 is None
                        else bool(
                            q9["weighted_age_s"]
                            >= 0.5
                        )
                    ),
                )
            )


v9a_diag = pd.DataFrame(
    diag_rows
)

if len(v9a_diag) == 0:
    raise RuntimeError(
        "V9-A diagnostic produced zero contact queries."
    )


# ======================================================================
# 6. SUMMARIES
# ======================================================================

v9a_summary = (
    v9a_diag
    .groupby(
        [
            "terrain_type",
            "leg_group"
        ]
    )
    .agg(
        queries=(
            "v9_found",
            "size"
        ),

        hit_rate=(
            "v9_found",
            "mean"
        ),

        median_observations=(
            "v9_n_obs",
            "median"
        ),

        median_effective_n=(
            "v9_effective_n",
            "median"
        ),

        median_nearest_distance_m=(
            "v9_nearest_distance_m",
            "median"
        ),

        median_weighted_distance_m=(
            "v9_weighted_distance_m",
            "median"
        ),

        median_age_s=(
            "v9_weighted_age_s",
            "median"
        ),

        seen_0p5s_earlier=(
            "v9_seen_0p5s_earlier",
            "mean"
        ),

        median_z_std_m=(
            "v9_z_std_m",
            "median"
        ),

        median_xy_std_m=(
            "v9_xy_std_m",
            "median"
        ),

        median_token_dispersion=(
            "v9_token_dispersion",
            "median"
        ),

        median_normal_dispersion=(
            "v9_normal_dispersion",
            "median"
        ),
    )
    .reset_index()
)


# ----------------------------------------------------------------------
# Overall comparison to V8-A
# ----------------------------------------------------------------------

v9a_overall = pd.DataFrame(
    [
        dict(
            metric="hit_rate",

            V8=float(
                v9a_diag.v8_found.mean()
            ),

            V9=float(
                v9a_diag.v9_found.mean()
            ),
        ),

        dict(
            metric="median_spatial_distance_m",

            V8=float(
                v9a_diag.v8_distance_m.median()
            ),

            V9=float(
                v9a_diag.v9_nearest_distance_m.median()
            ),
        ),

        dict(
            metric="median_independent_observations",

            V8=1.0,

            V9=float(
                v9a_diag.v9_n_obs.median()
            ),
        ),

        dict(
            metric="median_effective_observations",

            V8=1.0,

            V9=float(
                v9a_diag.v9_effective_n.median()
            ),
        ),
    ]
)


# ======================================================================
# 7. SAVE EVERYTHING
# ======================================================================

raw_path = (
    V9_A_ROOT /
    "V9A_BELIEF_QUERY_DIAGNOSTIC_RAW.csv"
)

summary_path = (
    V9_A_ROOT /
    "V9A_BELIEF_QUERY_SUMMARY.csv"
)

compare_path = (
    V9_A_ROOT /
    "V9A_V8_VS_V9_MEMORY_COMPARISON.csv"
)

v9a_diag.to_csv(
    raw_path,
    index=False
)

v9a_summary.to_csv(
    summary_path,
    index=False
)

v9a_overall.to_csv(
    compare_path,
    index=False
)


# ======================================================================
# 8. SCIENTIFIC DIAGNOSTICS
# ======================================================================

hind = v9a_diag[
    v9a_diag.leg_group == "hind"
]

V9A_HIT = float(
    v9a_diag.v9_found.mean()
)

V9A_MED_EFFECTIVE_N = float(
    v9a_diag.v9_effective_n.median()
)

V9A_HIND_PRIOR_05 = float(
    hind.v9_seen_0p5s_earlier.mean()
)

# Diagnostic, NOT a published threshold.
V9A_GATE = bool(
    V9A_HIT >= 0.95
    and
    V9A_MED_EFFECTIVE_N >= 2.0
    and
    V9A_HIND_PRIOR_05 >= 0.80
)


print()
print("=" * 100)
print("V9-A MEMORY BELIEF SUMMARY")
print("=" * 100)

display(
    v9a_summary
)

print()
print("V8-A vs V9-A")
display(
    v9a_overall
)

print()
print(
    "Overall V9 hit rate:",
    V9A_HIT
)

print(
    "Median effective independent observations:",
    V9A_MED_EFFECTIVE_N
)

print(
    "Hind terrain observed >=0.5 s earlier:",
    V9A_HIND_PRIOR_05
)

print(
    "V9-A engineering diagnostic gate:",
    V9A_GATE
)

print()
print(
    "Saved:",
    raw_path
)

print(
    "Saved:",
    summary_path
)

print(
    "Saved:",
    compare_path
)

print()
print("✅ V9-A COMPLETE")
print("✅ V8-A remains untouched.")
print("✅ Next experiment: front-contact -> hind-contact belief transfer.")

# %% notebook cell 32
# ======================================================================
# V9-A2 — FRONT-CONTACT -> HIND-CONTACT CAUSAL TRANSFER
#
# PURPOSE
# -------
# Test the central V9 hypothesis:
#
#   Can physical information learned when a FRONT foot contacts a
#   world location improve a later HIND-foot contact prediction?
#
# IMPORTANT:
#   * V8-C remains frozen.
#   * V9-A remains frozen.
#   * FL/FR write haptic memory.
#   * RL/RR are NEVER allowed to write haptic memory.
#   * RL/RR are the evaluation targets.
#
# This first experiment deliberately uses the known future contact
# location as an ORACLE LOCATION.
#
# Why?
# It isolates the scientific question:
#
#     "Does contact memory itself contain transferable information?"
#
# from the separate question:
#
#     "Can we predict the touchdown location accurately?"
#
# V9-B will remove this oracle later.
# ======================================================================

from pathlib import Path
from collections import defaultdict
import math
import json
import numpy as np
import pandas as pd

print("=" * 105)
print("V9-A2 — FRONT -> HIND CAUSAL HAPTIC TRANSFER")
print("=" * 105)


# ======================================================================
# 0. PREFLIGHT
# ======================================================================

_req = [
    "V9_A_ROOT",
    "run_records",
    "v8_memory",
    "v9_belief_memory",
    "v8_foot_world_proxy",
    "v8c_predict_cell",
]

_missing = [x for x in _req if x not in globals()]

if _missing:
    raise RuntimeError(
        "V9-A2 preflight incomplete. Missing: " + repr(_missing)
    )


V9_A2_ROOT = V9_A_ROOT / "front_to_hind_transfer"
V9_A2_ROOT.mkdir(parents=True, exist_ok=True)


# ======================================================================
# 1. EXPERIMENT CONSTANTS
# ======================================================================

# Keep exactly the V8-F short predictive horizon initially.
V9A2_HORIZON_S = 0.10

# Causal replay frequency.
V9A2_REPLAY_HOP_S = 0.10

# World-memory spatial scale.
V9A2_CELL_M = 0.10
V9A2_RADIUS_M = 0.35
V9A2_KERNEL_L_M = 0.14

# Semantic matching.
V9A2_TOKEN_COS_MIN = 0.40

# Surface compatibility.
V9A2_NORMAL_COS_MIN = 0.55

# Old haptic evidence should slowly fade.
V9A2_HAPTIC_TAU_S = 10.0

# Shrink corrections toward zero until enough real contacts exist.
V9A2_SHRINK_K = 4.0

# Physical safety bounds.
V9A2_TAU_CORR_CAP = 8.0
V9A2_FORCE_CORR_CAP = 40.0
V9A2_MAG_CORR_CAP = 50.0

FRONT_LEGS = (0, 1)
HIND_LEGS = (2, 3)

LEG_NAMES = {
    0: "FL",
    1: "FR",
    2: "RL",
    3: "RR",
}


# ======================================================================
# 2. HELPERS
# ======================================================================

def _v9a2_norm(x, eps=1e-9):
    return float(np.linalg.norm(np.asarray(x, np.float64)) + eps)


def _v9a2_unit(x, eps=1e-9):
    x = np.asarray(x, np.float64)
    n = np.linalg.norm(x)

    if n < eps:
        return np.zeros_like(x)

    return x / n


def _v9a2_cos(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)

    den = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
        + 1e-9
    )

    return float(
        np.dot(a, b) / den
    )


def _v9a2_clip_residual6(r):

    r = np.asarray(
        r,
        np.float64
    ).copy()

    # Robust torque observation bound.
    tn = np.linalg.norm(r[:3])

    if tn > 20.0:
        r[:3] *= 20.0 / (tn + 1e-9)

    # Robust force observation bound.
    fn = np.linalg.norm(r[3:])

    if fn > 100.0:
        r[3:] *= 100.0 / (fn + 1e-9)

    return r


def _v9a2_clip_output_corr(r):

    r = np.asarray(
        r,
        np.float64
    ).copy()

    tn = np.linalg.norm(r[:3])

    if tn > V9A2_TAU_CORR_CAP:
        r[:3] *= (
            V9A2_TAU_CORR_CAP /
            (tn + 1e-9)
        )

    fn = np.linalg.norm(r[3:])

    if fn > V9A2_FORCE_CORR_CAP:
        r[3:] *= (
            V9A2_FORCE_CORR_CAP /
            (fn + 1e-9)
        )

    return r.astype(np.float32)


# ======================================================================
# 3. WORLD-ANCHORED FRONT-FOOT HAPTIC MEMORY
# ======================================================================

class V9FrontHapticMemory:

    def __init__(self):

        # Spatial hash:
        #
        # world cell -> posterior summary learned ONLY from FL/FR.
        self.cells = {}

        # Global causal calibration is retained as a negative/control
        # comparison. It does not know WHERE a contact occurred.
        self.global_n = 0
        self.global_mean6 = np.zeros(
            6,
            np.float64
        )
        self.global_mag_delta = 0.0

    # ------------------------------------------------------------------
    # World cell.
    # ------------------------------------------------------------------

    def _key(self, xy):

        xy = np.asarray(
            xy,
            np.float64
        )

        return tuple(
            np.floor(
                xy / V9A2_CELL_M
            ).astype(int)
        )

    # ------------------------------------------------------------------
    # Observe a FRONT contact.
    # ------------------------------------------------------------------

    def update(
        self,
        belief,
        residual6,
        mag_delta,
        obs_time,
    ):

        residual6 = _v9a2_clip_residual6(
            residual6
        )

        mag_delta = float(
            np.clip(
                mag_delta,
                -80.0,
                80.0
            )
        )

        xy = np.asarray(
            belief["xy"],
            np.float64
        )

        token = _v9a2_unit(
            belief["token"]
        ).astype(np.float32)

        normal = np.asarray(
            belief["normal"],
            np.float64
        )

        if normal[2] < 0:
            normal = -normal

        normal = _v9a2_unit(
            normal
        ).astype(np.float32)

        key = self._key(xy)

        # --------------------------------------------------------------
        # Update global causal control.
        # --------------------------------------------------------------

        self.global_n += 1

        dg = residual6 - self.global_mean6

        self.global_mean6 += (
            dg / self.global_n
        )

        self.global_mag_delta += (
            mag_delta
            - self.global_mag_delta
        ) / self.global_n

        # --------------------------------------------------------------
        # Update spatial posterior.
        # --------------------------------------------------------------

        if key not in self.cells:

            self.cells[key] = dict(
                n=1,

                mean6=residual6.copy(),

                m2_6=np.zeros(
                    6,
                    np.float64
                ),

                mag_delta=float(
                    mag_delta
                ),

                xy=xy.copy(),

                token=token.copy(),

                normal=normal.copy(),

                last_t=float(
                    obs_time
                ),
            )

            return

        s = self.cells[key]

        old_n = int(s["n"])
        new_n = old_n + 1

        # Residual posterior.
        d = residual6 - s["mean6"]

        s["mean6"] += (
            d / new_n
        )

        s["m2_6"] += (
            d *
            (
                residual6
                - s["mean6"]
            )
        )

        # Force-magnitude response posterior.
        s["mag_delta"] += (
            mag_delta
            - s["mag_delta"]
        ) / new_n

        # Slowly update physical location.
        s["xy"] += (
            xy - s["xy"]
        ) / new_n

        # Slowly update semantic identity.
        tok_new = (
            old_n * s["token"]
            + token
        ) / new_n

        s["token"] = _v9a2_unit(
            tok_new
        ).astype(np.float32)

        # Surface normal posterior.
        n_new = (
            old_n * s["normal"]
            + normal
        ) / new_n

        s["normal"] = _v9a2_unit(
            n_new
        ).astype(np.float32)

        s["n"] = new_n

        s["last_t"] = float(
            obs_time
        )

    # ------------------------------------------------------------------
    # GLOBAL CONTROL.
    #
    # Answers:
    # "Would merely knowing the average front-foot model error help?"
    #
    # If LOCAL beats GLOBAL, world location matters.
    # ------------------------------------------------------------------

    def query_global(self):

        if self.global_n <= 0:

            return dict(
                corr6=np.zeros(
                    6,
                    np.float32
                ),

                mag_delta=0.0,

                contacts=0,

                shrink=0.0,
            )

        shrink = (
            self.global_n /
            (
                self.global_n
                + V9A2_SHRINK_K
            )
        )

        corr = (
            shrink *
            self.global_mean6
        )

        corr = _v9a2_clip_output_corr(
            corr
        )

        md = float(
            np.clip(
                shrink *
                self.global_mag_delta,

                -V9A2_MAG_CORR_CAP,
                V9A2_MAG_CORR_CAP
            )
        )

        return dict(
            corr6=corr,

            mag_delta=md,

            contacts=int(
                self.global_n
            ),

            shrink=float(
                shrink
            ),
        )

    # ------------------------------------------------------------------
    # LOCAL + SEMANTIC QUERY.
    #
    # Answers:
    # "What did front feet physically learn about THIS world location?"
    # ------------------------------------------------------------------

    def query_local(
        self,
        belief,
        query_time,
    ):

        if not self.cells:

            return dict(
                corr6=np.zeros(
                    6,
                    np.float32
                ),

                mag_delta=0.0,

                contacts=0,

                cells=0,

                effective_cells=0.0,

                weight=0.0,

                mean_age_s=np.nan,
            )

        qxy = np.asarray(
            belief["xy"],
            np.float64
        )

        qtok = _v9a2_unit(
            belief["token"]
        )

        qnormal = np.asarray(
            belief["normal"],
            np.float64
        )

        if qnormal[2] < 0:
            qnormal = -qnormal

        qnormal = _v9a2_unit(
            qnormal
        )

        base_key = self._key(
            qxy
        )

        nr = int(
            math.ceil(
                V9A2_RADIUS_M /
                V9A2_CELL_M
            )
        )

        vals = []

        # Search neighboring spatial hash cells only.
        for dx in range(
            -nr,
            nr + 1
        ):

            for dy in range(
                -nr,
                nr + 1
            ):

                key = (
                    base_key[0] + dx,
                    base_key[1] + dy
                )

                if key not in self.cells:
                    continue

                s = self.cells[key]

                # Strict causality.
                if (
                    float(s["last_t"])
                    >
                    float(query_time)
                    + 1e-8
                ):
                    continue

                dist = float(
                    np.linalg.norm(
                        qxy
                        - s["xy"]
                    )
                )

                if dist > V9A2_RADIUS_M:
                    continue

                tcos = _v9a2_cos(
                    qtok,
                    s["token"]
                )

                if (
                    tcos
                    <
                    V9A2_TOKEN_COS_MIN
                ):
                    continue

                ncos = _v9a2_cos(
                    qnormal,
                    s["normal"]
                )

                if (
                    ncos
                    <
                    V9A2_NORMAL_COS_MIN
                ):
                    continue

                age = max(
                    0.0,
                    float(query_time)
                    - float(s["last_t"])
                )

                # ------------------------------------------------------
                # World-distance likelihood.
                # Semantic consistency.
                # Surface orientation consistency.
                # Time decay.
                # Mild evidence-count confidence.
                # ------------------------------------------------------

                w = (
                    math.exp(
                        -0.5
                        *
                        (
                            dist /
                            V9A2_KERNEL_L_M
                        ) ** 2
                    )
                    *
                    max(
                        tcos,
                        0.0
                    ) ** 2
                    *
                    max(
                        ncos,
                        0.0
                    ) ** 2
                    *
                    math.exp(
                        -age /
                        V9A2_HAPTIC_TAU_S
                    )
                    *
                    min(
                        3.0,
                        math.log1p(
                            s["n"]
                        )
                    )
                )

                if w <= 1e-6:
                    continue

                vals.append(
                    (
                        w,
                        s,
                        age,
                        dist,
                        tcos,
                        ncos,
                    )
                )

        if not vals:

            return dict(
                corr6=np.zeros(
                    6,
                    np.float32
                ),

                mag_delta=0.0,

                contacts=0,

                cells=0,

                effective_cells=0.0,

                weight=0.0,

                mean_age_s=np.nan,
            )

        weights = np.asarray(
            [x[0] for x in vals],
            np.float64
        )

        wsum = float(
            weights.sum()
        )

        wn = (
            weights /
            (wsum + 1e-12)
        )

        corr = np.zeros(
            6,
            np.float64
        )

        mag_delta = 0.0
        total_contacts = 0

        for ww, (
            raw_w,
            s,
            age,
            dist,
            tc,
            nc
        ) in zip(
            wn,
            vals
        ):

            corr += (
                ww *
                s["mean6"]
            )

            mag_delta += (
                ww *
                s["mag_delta"]
            )

            total_contacts += int(
                s["n"]
            )

        # Posterior shrinkage.
        shrink = (
            total_contacts /
            (
                total_contacts
                + V9A2_SHRINK_K
            )
        )

        corr *= shrink

        mag_delta *= shrink

        corr = _v9a2_clip_output_corr(
            corr
        )

        mag_delta = float(
            np.clip(
                mag_delta,
                -V9A2_MAG_CORR_CAP,
                V9A2_MAG_CORR_CAP
            )
        )

        effective_cells = float(
            1.0 /
            (
                np.sum(
                    wn ** 2
                )
                + 1e-12
            )
        )

        ages = np.asarray(
            [x[2] for x in vals],
            np.float64
        )

        return dict(
            corr6=corr,

            mag_delta=mag_delta,

            contacts=int(
                total_contacts
            ),

            cells=int(
                len(vals)
            ),

            effective_cells=effective_cells,

            weight=wsum,

            shrink=float(
                shrink
            ),

            mean_age_s=float(
                np.sum(
                    wn * ages
                )
            ),
        )


# ======================================================================
# 4. MAGNITUDE-ONLY PHYSICS CONSERVATIVE CORRECTION
#
# Instead of transferring a full Fx/Fy/Fz residual from FL/FR -> RL/RR,
# transfer only:
#
#       "this terrain produces more/less load than the visual model expected"
#
# Direction remains V8-C's predicted direction.
#
# This is a cleaner terrain-property interpretation.
# ======================================================================

def _v9a2_apply_mag_delta(
    base_force,
    delta_mag,
):

    f = np.asarray(
        base_force,
        np.float64
    )

    mag = np.linalg.norm(f)

    if mag < 5.0:

        # Direction is unreliable when predicted load is near zero.
        return f.astype(np.float32)

    new_mag = max(
        0.0,
        mag + float(delta_mag)
    )

    return (
        f *
        (
            new_mag /
            (mag + 1e-9)
        )
    ).astype(np.float32)


# ======================================================================
# 5. STRICT CAUSAL REPLAY
#
# At query time t:
#
#   1. Predict future HIND contact at t + 0.1
#   2. Query ONLY front haptic observations with timestamp <= t
#   3. Score hind prediction
#   4. Observe actual FRONT contact at t + 0.1
#   5. Update front haptic posterior
#
# Therefore a target cannot update memory before it is predicted.
# ======================================================================

rows = []

front_update_rows = []

for rec in run_records:

    terrain = rec["terrain_type"]

    key = (
        rec["terrain_type"],
        rec["environment"],
        rec["run"]
    )

    data = rec["data"]

    mem8 = v8_memory[key]
    mem9 = v9_belief_memory[key]

    haptic = V9FrontHapticMemory()

    dt = float(
        np.median(
            np.diff(
                data["time"]
            )
        )
    )

    off = int(
        round(
            V9A2_HORIZON_S /
            dt
        )
    )

    step = max(
        1,
        int(
            round(
                V9A2_REPLAY_HOP_S /
                dt
            )
        )
    )

    print()
    print(
        "Replay:",
        terrain,
        "| dt=",
        dt,
        "| offset=",
        off,
        "| step=",
        step
    )

    # --------------------------------------------------------------
    # Regular 10-Hz replay.
    #
    # Using step ~= prediction horizon ensures observations created
    # at target t+0.1 first become available at/after the next query.
    # --------------------------------------------------------------

    for idx in range(
        0,
        len(data["time"]) - off,
        step
    ):

        j = idx + off

        query_t = float(
            data["time"][idx]
        )

        target_t = float(
            data["time"][j]
        )

        # ORACLE target location.
        #
        # This is deliberate for this ablation.
        fw_target = v8_foot_world_proxy(
            rec,
            j
        )

        if fw_target is None:
            continue

        # ==========================================================
        # PART A:
        # SCORE HIND LEGS BEFORE OBSERVING NEW FRONT CONTACTS.
        # ==========================================================

        for leg in HIND_LEGS:

            # Evaluate only actual loaded contacts.
            if (
                data["contact"][j, leg]
                < 0.5
            ):
                continue

            xyz = np.asarray(
                fw_target[leg],
                np.float32
            )

            # Frozen V8 query used for frozen V8-C prediction.
            mq8 = mem8.query(
                xyz[:2],
                query_t
            )

            if mq8 is None:
                continue

            # V9 multi-observation belief used to identify
            # physically equivalent terrain.
            b9 = mem9.query(
                xyz,
                query_t
            )

            if b9 is None:
                continue

            # ------------------------------------------------------
            # Frozen baseline.
            # ------------------------------------------------------

            base = v8c_predict_cell(
                rec,
                idx,
                leg,
                V9A2_HORIZON_S,
                mq8
            )

            base6 = np.concatenate(
                [
                    np.asarray(
                        base["tau"],
                        np.float32
                    ),

                    np.asarray(
                        base["force"],
                        np.float32
                    )
                ]
            )

            gt6 = np.concatenate(
                [
                    np.asarray(
                        data["P"][
                            j,
                            leg,
                            6:9
                        ],
                        np.float32
                    ),

                    np.asarray(
                        data["P"][
                            j,
                            leg,
                            9:12
                        ],
                        np.float32
                    )
                ]
            )

            # ------------------------------------------------------
            # Control: global front-contact calibration.
            # ------------------------------------------------------

            gc = haptic.query_global()

            global6 = (
                base6
                + gc["corr6"]
            )

            # ------------------------------------------------------
            # Hypothesis: local world-anchored haptic correction.
            # ------------------------------------------------------

            lc = haptic.query_local(
                b9,
                query_t
            )

            local_vec6 = (
                base6
                + lc["corr6"]
            )

            # ------------------------------------------------------
            # Conservative terrain-severity transfer.
            #
            # Keep torque and force direction untouched.
            # Modify only predicted force magnitude.
            # ------------------------------------------------------

            local_mag6 = (
                base6.copy()
            )

            local_mag6[3:] = (
                _v9a2_apply_mag_delta(
                    base6[3:],
                    lc["mag_delta"]
                )
            )

            # ------------------------------------------------------
            # Errors.
            # ------------------------------------------------------

            def force_mae(x):
                return float(
                    np.mean(
                        np.abs(
                            x[3:] - gt6[3:]
                        )
                    )
                )

            def force_vec(x):
                return float(
                    np.linalg.norm(
                        x[3:] - gt6[3:]
                    )
                )

            def tau_mae(x):
                return float(
                    np.mean(
                        np.abs(
                            x[:3] - gt6[:3]
                        )
                    )
                )

            b_mae = force_mae(
                base6
            )

            g_mae = force_mae(
                global6
            )

            lv_mae = force_mae(
                local_vec6
            )

            lm_mae = force_mae(
                local_mag6
            )

            # Was this near one of original V8 diagnostic query times?
            test_delta = min(
                abs(
                    query_t
                    - float(tt)
                )
                for tt in rec["ttests"]
            )

            rows.append(
                dict(
                    terrain_type=terrain,

                    environment=rec[
                        "environment"
                    ],

                    run=rec["run"],

                    query_time=query_t,

                    target_time=target_t,

                    horizon_s=(
                        V9A2_HORIZON_S
                    ),

                    leg=int(
                        leg
                    ),

                    leg_name=LEG_NAMES[
                        leg
                    ],

                    # Controlled experimental limitation.
                    oracle_contact_location=True,

                    # Original V9 visual belief.
                    visual_effective_n=float(
                        b9["effective_n"]
                    ),

                    visual_token_dispersion=float(
                        b9[
                            "token_dispersion"
                        ]
                    ),

                    visual_distance_m=float(
                        b9[
                            "weighted_distance_m"
                        ]
                    ),

                    # Haptic posterior evidence.
                    prior_front_contacts=int(
                        lc["contacts"]
                    ),

                    local_haptic_cells=int(
                        lc["cells"]
                    ),

                    local_effective_cells=float(
                        lc[
                            "effective_cells"
                        ]
                    ),

                    local_haptic_weight=float(
                        lc["weight"]
                    ),

                    local_haptic_age_s=float(
                        lc["mean_age_s"]
                    )
                    if np.isfinite(
                        lc["mean_age_s"]
                    )
                    else np.nan,

                    global_front_contacts=int(
                        gc["contacts"]
                    ),

                    # Force metrics.
                    base_force_MAE=b_mae,

                    global_force_MAE=g_mae,

                    local_vector_force_MAE=lv_mae,

                    local_magnitude_force_MAE=lm_mae,

                    base_force_vector_error=force_vec(
                        base6
                    ),

                    global_force_vector_error=force_vec(
                        global6
                    ),

                    local_vector_force_vector_error=force_vec(
                        local_vec6
                    ),

                    local_magnitude_force_vector_error=force_vec(
                        local_mag6
                    ),

                    # Torque.
                    base_tau_MAE=tau_mae(
                        base6
                    ),

                    global_tau_MAE=tau_mae(
                        global6
                    ),

                    local_vector_tau_MAE=tau_mae(
                        local_vec6
                    ),

                    # Paired wins.
                    global_improved=bool(
                        g_mae < b_mae
                    ),

                    local_vector_improved=bool(
                        lv_mae < b_mae
                    ),

                    local_magnitude_improved=bool(
                        lm_mae < b_mae
                    ),

                    local_beats_global=bool(
                        lv_mae < g_mae
                    ),

                    has_local_front_evidence=bool(
                        lc["contacts"] > 0
                    ),

                    original_test_time_near=bool(
                        test_delta <= 0.051
                    ),

                    original_test_time_delta_s=float(
                        test_delta
                    ),
                )
            )

        # ==========================================================
        # PART B:
        # CREATE FRONT UPDATES ONLY AFTER HIND SCORING.
        # ==========================================================

        pending = []

        for leg in FRONT_LEGS:

            if (
                data["contact"][j, leg]
                < 0.5
            ):
                continue

            xyz = np.asarray(
                fw_target[leg],
                np.float32
            )

            mq8 = mem8.query(
                xyz[:2],
                query_t
            )

            if mq8 is None:
                continue

            b9 = mem9.query(
                xyz,
                query_t
            )

            if b9 is None:
                continue

            # What did frozen V8-C EXPECT before this front contact?
            pred = v8c_predict_cell(
                rec,
                idx,
                leg,
                V9A2_HORIZON_S,
                mq8
            )

            pred6 = np.concatenate(
                [
                    np.asarray(
                        pred["tau"],
                        np.float32
                    ),

                    np.asarray(
                        pred["force"],
                        np.float32
                    )
                ]
            )

            actual6 = np.concatenate(
                [
                    np.asarray(
                        data["P"][
                            j,
                            leg,
                            6:9
                        ],
                        np.float32
                    ),

                    np.asarray(
                        data["P"][
                            j,
                            leg,
                            9:12
                        ],
                        np.float32
                    )
                ]
            )

            residual6 = (
                actual6
                - pred6
            )

            mag_delta = float(
                np.linalg.norm(
                    actual6[3:]
                )
                -
                np.linalg.norm(
                    pred6[3:]
                )
            )

            pending.append(
                (
                    b9,
                    residual6,
                    mag_delta,
                    target_t,
                    leg,
                )
            )

        # ----------------------------------------------------------
        # CRITICAL CAUSAL LINE:
        #
        # Only NOW is the real target contact inserted.
        # ----------------------------------------------------------

        for (
            b9,
            residual6,
            mag_delta,
            obs_t,
            leg,
        ) in pending:

            haptic.update(
                b9,
                residual6,
                mag_delta,
                obs_t
            )

            front_update_rows.append(
                dict(
                    terrain_type=terrain,

                    environment=rec[
                        "environment"
                    ],

                    run=rec["run"],

                    observation_time=float(
                        obs_t
                    ),

                    front_leg=int(
                        leg
                    ),

                    front_leg_name=LEG_NAMES[
                        leg
                    ],

                    residual_tau_norm=float(
                        np.linalg.norm(
                            residual6[:3]
                        )
                    ),

                    residual_force_norm=float(
                        np.linalg.norm(
                            residual6[3:]
                        )
                    ),

                    force_magnitude_delta=float(
                        mag_delta
                    ),

                    visual_effective_n=float(
                        b9["effective_n"]
                    ),

                    visual_token_dispersion=float(
                        b9[
                            "token_dispersion"
                        ]
                    ),
                )
            )


# ======================================================================
# 6. TABLES
# ======================================================================

v9a2_eval = pd.DataFrame(
    rows
)

v9a2_front_updates = pd.DataFrame(
    front_update_rows
)

if len(v9a2_eval) == 0:

    raise RuntimeError(
        "V9-A2 produced zero hind-contact evaluation samples."
    )


# ======================================================================
# 7. OVERALL SUMMARY
# ======================================================================

def _pct_improvement(
    base,
    new
):

    return float(
        100.0
        *
        (
            base - new
        )
        /
        (
            base + 1e-9
        )
    )


def _summarize_transfer(df):

    if len(df) == 0:
        return {}

    b = float(
        df.base_force_MAE.mean()
    )

    g = float(
        df.global_force_MAE.mean()
    )

    lv = float(
        df.local_vector_force_MAE.mean()
    )

    lm = float(
        df.local_magnitude_force_MAE.mean()
    )

    return dict(
        samples=int(
            len(df)
        ),

        local_evidence_fraction=float(
            df.has_local_front_evidence.mean()
        ),

        base_force_MAE_N=b,

        global_force_MAE_N=g,

        local_vector_force_MAE_N=lv,

        local_magnitude_force_MAE_N=lm,

        global_improvement_percent=(
            _pct_improvement(
                b,
                g
            )
        ),

        local_vector_improvement_percent=(
            _pct_improvement(
                b,
                lv
            )
        ),

        local_magnitude_improvement_percent=(
            _pct_improvement(
                b,
                lm
            )
        ),

        global_win_rate=float(
            df.global_improved.mean()
        ),

        local_vector_win_rate=float(
            df.local_vector_improved.mean()
        ),

        local_magnitude_win_rate=float(
            df.local_magnitude_improved.mean()
        ),

        local_vector_beats_global_rate=float(
            df.local_beats_global.mean()
        ),

        base_force_vector_error_N=float(
            df.base_force_vector_error.mean()
        ),

        local_vector_force_vector_error_N=float(
            df[
                "local_vector_force_vector_error"
            ].mean()
        ),

        local_magnitude_force_vector_error_N=float(
            df[
                "local_magnitude_force_vector_error"
            ].mean()
        ),

        base_tau_MAE_Nm=float(
            df.base_tau_MAE.mean()
        ),

        local_vector_tau_MAE_Nm=float(
            df.local_vector_tau_MAE.mean()
        ),
    )


overall = _summarize_transfer(
    v9a2_eval
)

evidence = _summarize_transfer(
    v9a2_eval[
        v9a2_eval.has_local_front_evidence
    ]
)

v9a2_summary = pd.DataFrame(
    [
        dict(
            subset="all_hind_contacts",
            **overall
        ),

        dict(
            subset="hind_contacts_with_local_front_evidence",
            **evidence
        ),
    ]
)


# ======================================================================
# 8. TERRAIN + HIND LEG BREAKDOWN
# ======================================================================

terrain_rows = []

for (
    terrain,
    leg_name
), g in v9a2_eval.groupby(
    [
        "terrain_type",
        "leg_name"
    ]
):

    terrain_rows.append(
        dict(
            terrain_type=terrain,
            leg_name=leg_name,
            **_summarize_transfer(
                g
            )
        )
    )

v9a2_by_terrain = pd.DataFrame(
    terrain_rows
)


# ======================================================================
# 9. ORIGINAL 16-TIME DIAGNOSTIC SUBSET
#
# This is approximate because replay is sampled at 10 Hz.
# ======================================================================

v9a2_test_subset = (
    v9a2_eval[
        v9a2_eval.original_test_time_near
    ]
    .copy()
)


# ======================================================================
# 10. EFFECT AS A FUNCTION OF PRIOR FRONT EXPERIENCE
# ======================================================================

def _front_bucket(n):

    if n <= 0:
        return "0"

    if n == 1:
        return "1"

    if n <= 4:
        return "2-4"

    if n <= 10:
        return "5-10"

    return "11+"


v9a2_eval[
    "front_contact_bucket"
] = (
    v9a2_eval[
        "prior_front_contacts"
    ]
    .map(
        _front_bucket
    )
)


curve_rows = []

bucket_order = [
    "0",
    "1",
    "2-4",
    "5-10",
    "11+"
]

for bucket in bucket_order:

    g = v9a2_eval[
        v9a2_eval.front_contact_bucket
        == bucket
    ]

    if len(g) == 0:
        continue

    curve_rows.append(
        dict(
            prior_front_contact_bucket=bucket,
            **_summarize_transfer(
                g
            )
        )
    )

v9a2_learning_curve = pd.DataFrame(
    curve_rows
)


# ======================================================================
# 11. SAVE
# ======================================================================

raw_path = (
    V9_A2_ROOT /
    "V9A2_FRONT_TO_HIND_RAW.csv"
)

front_path = (
    V9_A2_ROOT /
    "V9A2_FRONT_CONTACT_UPDATES.csv"
)

summary_path = (
    V9_A2_ROOT /
    "V9A2_FRONT_TO_HIND_SUMMARY.csv"
)

terrain_path = (
    V9_A2_ROOT /
    "V9A2_FRONT_TO_HIND_BY_TERRAIN.csv"
)

curve_path = (
    V9_A2_ROOT /
    "V9A2_FRONT_EXPERIENCE_CURVE.csv"
)

test_path = (
    V9_A2_ROOT /
    "V9A2_ORIGINAL_TEST_TIME_SUBSET.csv"
)


v9a2_eval.to_csv(
    raw_path,
    index=False
)

v9a2_front_updates.to_csv(
    front_path,
    index=False
)

v9a2_summary.to_csv(
    summary_path,
    index=False
)

v9a2_by_terrain.to_csv(
    terrain_path,
    index=False
)

v9a2_learning_curve.to_csv(
    curve_path,
    index=False
)

v9a2_test_subset.to_csv(
    test_path,
    index=False
)


# ======================================================================
# 12. RESEARCH INTERPRETATION
# ======================================================================

print()
print("=" * 105)
print("V9-A2 FRONT -> HIND TRANSFER SUMMARY")
print("=" * 105)

display(
    v9a2_summary
)

print()
print(
    "BY TERRAIN / HIND LEG"
)

display(
    v9a2_by_terrain
)

print()
print(
    "EFFECT VS PRIOR FRONT-CONTACT EXPERIENCE"
)

display(
    v9a2_learning_curve
)

print()

print(
    "Front-contact observations inserted:",
    len(v9a2_front_updates)
)

print(
    "Hind-contact predictions evaluated:",
    len(v9a2_eval)
)

print(
    "Fraction with local front evidence:",
    float(
        v9a2_eval[
            "has_local_front_evidence"
        ].mean()
    )
)

print()

print(
    "Base force MAE:",
    overall[
        "base_force_MAE_N"
    ]
)

print(
    "Global front correction MAE:",
    overall[
        "global_force_MAE_N"
    ]
)

print(
    "LOCAL vector correction MAE:",
    overall[
        "local_vector_force_MAE_N"
    ]
)

print(
    "LOCAL magnitude correction MAE:",
    overall[
        "local_magnitude_force_MAE_N"
    ]
)

print()

print(
    "Local-vector improvement:",
    overall[
        "local_vector_improvement_percent"
    ],
    "%"
)

print(
    "Local-magnitude improvement:",
    overall[
        "local_magnitude_improvement_percent"
    ],
    "%"
)

print(
    "Local vector beats global:",
    overall[
        "local_vector_beats_global_rate"
    ]
)

print()
print("Scientific status:")

if (
    overall[
        "local_vector_improvement_percent"
    ] > 0
    and
    overall[
        "local_vector_force_MAE_N"
    ]
    <
    overall[
        "global_force_MAE_N"
    ]
):

    print(
        "✅ POSITIVE SIGNAL: local world-anchored front contact "
        "contains information useful for later hind prediction."
    )

elif (
    overall[
        "local_magnitude_improvement_percent"
    ] > 0
):

    print(
        "🟡 PARTIAL SIGNAL: transferable information appears to "
        "exist mainly in contact severity / force magnitude."
    )

else:

    print(
        "❌ NO TRANSFER YET: do NOT hide this result. "
        "We will inspect whether the failure comes from coordinate-frame, "
        "load-sharing, spatial matching, or the V8-C baseline."
    )

print()
print(
    "IMPORTANT: this is an ORACLE-CONTACT-LOCATION causal ablation."
)

print(
    "It tests haptic-memory VALUE, not deployment-level touchdown prediction."
)

print()
print("Saved:", V9_A2_ROOT)
print("✅ V9-A2 COMPLETE")

# %% notebook cell 33
# ======================================================================
# V9-A3 — INDEPENDENT TOUCHDOWN EVENT TRANSFER
#
# Front feet WRITE only at true touchdown transitions.
# Hind feet are evaluated only at true touchdown transitions.
#
# Controls:
#   BASE      = frozen V8-C
#   GLOBAL    = average prior front-touch magnitude bias
#   SEMANTIC  = prior physically-similar front contacts, location ignored
#   LOCAL     = world-location + semantic + geometry matched front contacts
#
# Purpose:
#   Determine whether front-foot physical experience transfers to later
#   hind-foot touchdown prediction at the SAME terrain location.
#
# Still an ORACLE-CONTACT-LOCATION ablation.
# No neural-network training.
# ======================================================================

from pathlib import Path
import math
import numpy as np
import pandas as pd

print("=" * 110)
print("V9-A3 — INDEPENDENT TOUCHDOWN EVENT TRANSFER")
print("=" * 110)

# ----------------------------------------------------------------------
# 0. PREFLIGHT
# ----------------------------------------------------------------------

_req = [
    "V9_A_ROOT",
    "run_records",
    "v8_memory",
    "v9_belief_memory",
    "v8_foot_world_proxy",
    "v8c_predict_cell",
]

_missing = [x for x in _req if x not in globals()]

if _missing:
    raise RuntimeError(
        "V9-A3 missing live objects: " + repr(_missing)
    )

V9_A3_ROOT = (
    V9_A_ROOT /
    "independent_touchdown_transfer"
)

V9_A3_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

# ----------------------------------------------------------------------
# 1. CONFIG
# ----------------------------------------------------------------------

V9A3_HORIZON_S = 0.10

V9A3_CELL_M = 0.10
V9A3_RADIUS_M = 0.35
V9A3_KERNEL_L_M = 0.14

V9A3_TOKEN_COS_MIN = 0.40
V9A3_NORMAL_COS_MIN = 0.55

V9A3_AGE_TAU_S = 12.0

# Fewer touchdown events exist than 0.1-s stance samples,
# so shrink less aggressively than A2.
V9A3_SHRINK_K = 2.0

V9A3_MAX_MAG_CORR_N = 60.0

FRONT_LEGS = (0, 1)
HIND_LEGS = (2, 3)

LEG_NAMES = {
    0: "FL",
    1: "FR",
    2: "RL",
    3: "RR",
}

BOOTSTRAP_REPS = 3000
BOOTSTRAP_SEED = 781


# ======================================================================
# 2. HELPERS
# ======================================================================

def v9a3_unit(x, eps=1e-9):

    x = np.asarray(
        x,
        np.float64
    )

    n = np.linalg.norm(x)

    if n < eps:
        return np.zeros_like(x)

    return x / n


def v9a3_cos(a, b):

    a = np.asarray(
        a,
        np.float64
    )

    b = np.asarray(
        b,
        np.float64
    )

    return float(
        np.dot(a, b)
        /
        (
            np.linalg.norm(a)
            * np.linalg.norm(b)
            + 1e-9
        )
    )


def v9a3_touchdowns(contact_column):
    """
    Return only genuine 0 -> 1 contact transitions.
    One continuous stance = ONE physical contact event.
    """

    c = (
        np.asarray(contact_column)
        >= 0.5
    )

    if len(c) < 2:
        return np.empty(
            0,
            dtype=np.int64
        )

    return (
        np.where(
            c[1:]
            &
            (~c[:-1])
        )[0]
        + 1
    ).astype(np.int64)


def v9a3_apply_mag_delta(
    base_force,
    delta_mag
):
    """
    Coordinate-frame-safe intervention.

    Preserve V8-C predicted force direction.
    Change only magnitude.
    """

    f = np.asarray(
        base_force,
        np.float64
    )

    mag = float(
        np.linalg.norm(f)
    )

    if mag < 5.0:
        return f.astype(
            np.float32
        )

    dm = float(
        np.clip(
            delta_mag,
            -V9A3_MAX_MAG_CORR_N,
            V9A3_MAX_MAG_CORR_N
        )
    )

    new_mag = max(
        0.0,
        mag + dm
    )

    return (
        f *
        (
            new_mag /
            (mag + 1e-9)
        )
    ).astype(np.float32)


# ======================================================================
# 3. EVENT-LEVEL HAPTIC POSTERIOR
# ======================================================================

class V9TouchdownMagnitudeMemory:

    def __init__(self):

        self.cells = {}

        self.global_n = 0

        self.global_mean = 0.0

        self.total_updates = 0

    # ------------------------------------------------------------------

    def _key(self, xy):

        xy = np.asarray(
            xy,
            np.float64
        )

        return tuple(
            np.floor(
                xy / V9A3_CELL_M
            ).astype(int)
        )

    # ------------------------------------------------------------------
    # Every call corresponds to ONE independent front touchdown.
    # ------------------------------------------------------------------

    def update(
        self,
        belief,
        delta_mag,
        obs_time,
    ):

        delta_mag = float(
            np.clip(
                delta_mag,
                -100.0,
                100.0
            )
        )

        xy = np.asarray(
            belief["xy"],
            np.float64
        )

        token = v9a3_unit(
            belief["token"]
        ).astype(np.float32)

        normal = np.asarray(
            belief["normal"],
            np.float64
        )

        if normal[2] < 0:
            normal = -normal

        normal = v9a3_unit(
            normal
        ).astype(np.float32)

        # --------------------------------------------------------------
        # Global calibration control.
        # --------------------------------------------------------------

        self.global_n += 1

        self.global_mean += (
            delta_mag
            - self.global_mean
        ) / self.global_n

        self.total_updates += 1

        # --------------------------------------------------------------
        # World-local posterior.
        # --------------------------------------------------------------

        k = self._key(xy)

        if k not in self.cells:

            self.cells[k] = dict(
                n=1,

                mean=float(
                    delta_mag
                ),

                m2=0.0,

                xy=xy.copy(),

                token=token.copy(),

                normal=normal.copy(),

                last_t=float(
                    obs_time
                ),
            )

            return

        s = self.cells[k]

        old_n = int(
            s["n"]
        )

        new_n = (
            old_n + 1
        )

        d = (
            delta_mag
            - s["mean"]
        )

        s["mean"] += (
            d / new_n
        )

        s["m2"] += (
            d *
            (
                delta_mag
                - s["mean"]
            )
        )

        s["xy"] += (
            xy
            - s["xy"]
        ) / new_n

        tok = (
            old_n * s["token"]
            + token
        ) / new_n

        s["token"] = (
            v9a3_unit(tok)
            .astype(np.float32)
        )

        nn = (
            old_n * s["normal"]
            + normal
        ) / new_n

        s["normal"] = (
            v9a3_unit(nn)
            .astype(np.float32)
        )

        s["n"] = (
            new_n
        )

        s["last_t"] = float(
            obs_time
        )

    # ------------------------------------------------------------------

    def query_global(self):

        if self.global_n == 0:

            return dict(
                delta_mag=0.0,
                events=0,
            )

        shrink = (
            self.global_n
            /
            (
                self.global_n
                + V9A3_SHRINK_K
            )
        )

        dm = (
            shrink
            * self.global_mean
        )

        return dict(
            delta_mag=float(
                np.clip(
                    dm,
                    -V9A3_MAX_MAG_CORR_N,
                    V9A3_MAX_MAG_CORR_N
                )
            ),

            events=int(
                self.global_n
            ),
        )

    # ------------------------------------------------------------------
    # Generic posterior query.
    #
    # spatial=True:
    #       world-local haptic memory.
    #
    # spatial=False:
    #       semantic-only control.
    #       Same prior front-contact experience, but WHERE it occurred
    #       is deliberately ignored.
    # ------------------------------------------------------------------

    def _query(
        self,
        belief,
        query_time,
        spatial,
    ):

        if not self.cells:

            return dict(
                delta_mag=0.0,
                events=0,
                cells=0,
                effective_cells=0.0,
                mean_age_s=np.nan,
                posterior_std_N=np.nan,
            )

        qxy = np.asarray(
            belief["xy"],
            np.float64
        )

        qt = v9a3_unit(
            belief["token"]
        )

        qn = np.asarray(
            belief["normal"],
            np.float64
        )

        if qn[2] < 0:
            qn = -qn

        qn = v9a3_unit(qn)

        vals = []

        for s in self.cells.values():

            # Strict causal protection.
            if (
                float(s["last_t"])
                >
                float(query_time)
                + 1e-8
            ):
                continue

            dist = float(
                np.linalg.norm(
                    qxy
                    - s["xy"]
                )
            )

            if (
                spatial
                and
                dist > V9A3_RADIUS_M
            ):
                continue

            tc = v9a3_cos(
                qt,
                s["token"]
            )

            if (
                tc
                <
                V9A3_TOKEN_COS_MIN
            ):
                continue

            nc = v9a3_cos(
                qn,
                s["normal"]
            )

            if (
                nc
                <
                V9A3_NORMAL_COS_MIN
            ):
                continue

            age = max(
                0.0,
                float(query_time)
                - float(s["last_t"])
            )

            # ----------------------------------------------------------
            # Evidence weight.
            # ----------------------------------------------------------

            w = (
                max(tc, 0.0) ** 2
                *
                max(nc, 0.0) ** 2
                *
                math.exp(
                    -age /
                    V9A3_AGE_TAU_S
                )
                *
                min(
                    3.0,
                    math.log1p(
                        s["n"]
                    )
                )
            )

            if spatial:

                w *= math.exp(
                    -0.5
                    *
                    (
                        dist /
                        V9A3_KERNEL_L_M
                    ) ** 2
                )

            if w <= 1e-7:
                continue

            var = (
                s["m2"]
                /
                max(
                    s["n"] - 1,
                    1
                )
                if s["n"] > 1
                else 0.0
            )

            vals.append(
                dict(
                    w=float(w),

                    mean=float(
                        s["mean"]
                    ),

                    variance=float(
                        max(
                            var,
                            0.0
                        )
                    ),

                    n=int(
                        s["n"]
                    ),

                    age=float(
                        age
                    ),

                    dist=float(
                        dist
                    ),
                )
            )

        if not vals:

            return dict(
                delta_mag=0.0,
                events=0,
                cells=0,
                effective_cells=0.0,
                mean_age_s=np.nan,
                posterior_std_N=np.nan,
            )

        ww = np.asarray(
            [x["w"] for x in vals],
            np.float64
        )

        ww /= (
            ww.sum()
            + 1e-12
        )

        means = np.asarray(
            [x["mean"] for x in vals],
            np.float64
        )

        mean = float(
            np.sum(
                ww * means
            )
        )

        total_events = int(
            sum(
                x["n"]
                for x in vals
            )
        )

        shrink = (
            total_events
            /
            (
                total_events
                + V9A3_SHRINK_K
            )
        )

        mean *= shrink

        # --------------------------------------------------------------
        # Posterior spread:
        #
        # within-cell variance
        # +
        # disagreement between cells.
        # --------------------------------------------------------------

        within = np.asarray(
            [
                x["variance"]
                for x in vals
            ],
            np.float64
        )

        total_var = float(
            np.sum(
                ww
                *
                (
                    within
                    +
                    (
                        means
                        - np.sum(
                            ww * means
                        )
                    ) ** 2
                )
            )
        )

        ages = np.asarray(
            [
                x["age"]
                for x in vals
            ],
            np.float64
        )

        return dict(
            delta_mag=float(
                np.clip(
                    mean,
                    -V9A3_MAX_MAG_CORR_N,
                    V9A3_MAX_MAG_CORR_N
                )
            ),

            events=total_events,

            cells=int(
                len(vals)
            ),

            effective_cells=float(
                1.0 /
                (
                    np.sum(
                        ww ** 2
                    )
                    + 1e-12
                )
            ),

            mean_age_s=float(
                np.sum(
                    ww * ages
                )
            ),

            posterior_std_N=float(
                np.sqrt(
                    max(
                        total_var,
                        0.0
                    )
                )
            ),
        )

    def query_local(
        self,
        belief,
        query_time,
    ):

        return self._query(
            belief,
            query_time,
            spatial=True,
        )

    def query_semantic(
        self,
        belief,
        query_time,
    ):

        return self._query(
            belief,
            query_time,
            spatial=False,
        )


# ======================================================================
# 4. PRECOMPUTE TRUE TOUCHDOWN EVENTS
# ======================================================================

all_rows = []

event_inventory = []

for rec in run_records:

    terrain = rec[
        "terrain_type"
    ]

    key = (
        rec["terrain_type"],
        rec["environment"],
        rec["run"]
    )

    data = rec["data"]

    mem8 = v8_memory[key]

    mem9 = v9_belief_memory[key]

    dt = float(
        np.median(
            np.diff(
                data["time"]
            )
        )
    )

    off = int(
        round(
            V9A3_HORIZON_S /
            dt
        )
    )

    # --------------------------------------------------------------
    # Detect genuine touchdown events.
    # --------------------------------------------------------------

    td = {
        leg: v9a3_touchdowns(
            data["contact"][:, leg]
        )
        for leg in range(4)
    }

    print()
    print(
        terrain,
        "touchdowns:",
        {
            LEG_NAMES[k]: len(v)
            for k, v in td.items()
        }
    )

    # ==============================================================
    # A. FRONT events.
    #
    # Precompute the physical observation that will later be inserted
    # into memory only when obs_time <= hind query_time.
    # ==============================================================

    front_events = []

    for leg in FRONT_LEGS:

        for j in td[leg]:

            idx = int(
                j - off
            )

            if idx < 0:
                continue

            query_t = float(
                data["time"][idx]
            )

            obs_t = float(
                data["time"][j]
            )

            fw = v8_foot_world_proxy(
                rec,
                j
            )

            if fw is None:
                continue

            xyz = np.asarray(
                fw[leg],
                np.float32
            )

            mq8 = mem8.query(
                xyz[:2],
                query_t
            )

            if mq8 is None:
                continue

            b9 = mem9.query(
                xyz,
                query_t
            )

            if b9 is None:
                continue

            pred = v8c_predict_cell(
                rec,
                idx,
                leg,
                V9A3_HORIZON_S,
                mq8
            )

            pred_force = np.asarray(
                pred["force"],
                np.float32
            )

            gt_force = np.asarray(
                data["P"][
                    j,
                    leg,
                    9:12
                ],
                np.float32
            )

            delta_mag = float(
                np.linalg.norm(
                    gt_force
                )
                -
                np.linalg.norm(
                    pred_force
                )
            )

            front_events.append(
                dict(
                    obs_t=obs_t,

                    query_t=query_t,

                    index=idx,

                    target_index=int(
                        j
                    ),

                    leg=int(
                        leg
                    ),

                    belief=b9,

                    delta_mag=delta_mag,
                )
            )

    front_events.sort(
        key=lambda x: x["obs_t"]
    )

    # ==============================================================
    # B. HIND touchdown prediction events.
    # ==============================================================

    hind_events = []

    for leg in HIND_LEGS:

        for j in td[leg]:

            idx = int(
                j - off
            )

            if idx < 0:
                continue

            query_t = float(
                data["time"][idx]
            )

            obs_t = float(
                data["time"][j]
            )

            fw = v8_foot_world_proxy(
                rec,
                j
            )

            if fw is None:
                continue

            xyz = np.asarray(
                fw[leg],
                np.float32
            )

            mq8 = mem8.query(
                xyz[:2],
                query_t
            )

            if mq8 is None:
                continue

            b9 = mem9.query(
                xyz,
                query_t
            )

            if b9 is None:
                continue

            pred = v8c_predict_cell(
                rec,
                idx,
                leg,
                V9A3_HORIZON_S,
                mq8
            )

            base_force = np.asarray(
                pred["force"],
                np.float32
            )

            gt_force = np.asarray(
                data["P"][
                    j,
                    leg,
                    9:12
                ],
                np.float32
            )

            hind_events.append(
                dict(
                    query_t=query_t,

                    obs_t=obs_t,

                    index=idx,

                    target_index=int(
                        j
                    ),

                    leg=int(
                        leg
                    ),

                    belief=b9,

                    base_force=base_force,

                    gt_force=gt_force,
                )
            )

    hind_events.sort(
        key=lambda x: x["query_t"]
    )

    # ==============================================================
    # C. STRICT ONLINE EVENT REPLAY
    # ==============================================================

    memory = (
        V9TouchdownMagnitudeMemory()
    )

    fp = 0

    for h in hind_events:

        # ----------------------------------------------------------
        # Insert ONLY front touchdown events already physically
        # observed by this hind prediction time.
        # ----------------------------------------------------------

        while (
            fp < len(front_events)
            and
            front_events[fp][
                "obs_t"
            ]
            <= h["query_t"] + 1e-9
        ):

            fe = front_events[fp]

            memory.update(
                fe["belief"],
                fe["delta_mag"],
                fe["obs_t"],
            )

            fp += 1

        base_force = h[
            "base_force"
        ]

        gt_force = h[
            "gt_force"
        ]

        # ----------------------------------------------------------
        # Three controls / interventions.
        # ----------------------------------------------------------

        g = memory.query_global()

        s = memory.query_semantic(
            h["belief"],
            h["query_t"],
        )

        l = memory.query_local(
            h["belief"],
            h["query_t"],
        )

        global_force = (
            v9a3_apply_mag_delta(
                base_force,
                g["delta_mag"]
            )
        )

        semantic_force = (
            v9a3_apply_mag_delta(
                base_force,
                s["delta_mag"]
            )
        )

        local_force = (
            v9a3_apply_mag_delta(
                base_force,
                l["delta_mag"]
            )
        )

        # ----------------------------------------------------------
        # Metrics.
        # ----------------------------------------------------------

        def component_mae(x):

            return float(
                np.mean(
                    np.abs(
                        np.asarray(x)
                        -
                        gt_force
                    )
                )
            )

        def vector_error(x):

            return float(
                np.linalg.norm(
                    np.asarray(x)
                    -
                    gt_force
                )
            )

        def magnitude_error(x):

            return float(
                abs(
                    np.linalg.norm(x)
                    -
                    np.linalg.norm(
                        gt_force
                    )
                )
            )

        bmae = component_mae(
            base_force
        )

        gmae = component_mae(
            global_force
        )

        smae = component_mae(
            semantic_force
        )

        lmae = component_mae(
            local_force
        )

        all_rows.append(
            dict(
                terrain_type=terrain,

                environment=rec[
                    "environment"
                ],

                run=rec["run"],

                query_time=float(
                    h["query_t"]
                ),

                touchdown_time=float(
                    h["obs_t"]
                ),

                leg=int(
                    h["leg"]
                ),

                leg_name=LEG_NAMES[
                    h["leg"]
                ],

                oracle_contact_location=True,

                # --------------------------------------------------
                # How many TRUE prior front touchdown events exist?
                # --------------------------------------------------

                total_prior_front_touchdowns=int(
                    memory.total_updates
                ),

                local_front_touchdowns=int(
                    l["events"]
                ),

                semantic_front_touchdowns=int(
                    s["events"]
                ),

                local_cells=int(
                    l["cells"]
                ),

                local_effective_cells=float(
                    l["effective_cells"]
                ),

                local_posterior_std_N=float(
                    l["posterior_std_N"]
                )
                if np.isfinite(
                    l[
                        "posterior_std_N"
                    ]
                )
                else np.nan,

                local_memory_age_s=float(
                    l["mean_age_s"]
                )
                if np.isfinite(
                    l["mean_age_s"]
                )
                else np.nan,

                # --------------------------------------------------
                # Component MAE.
                # --------------------------------------------------

                base_force_MAE_N=bmae,

                global_force_MAE_N=gmae,

                semantic_force_MAE_N=smae,

                local_force_MAE_N=lmae,

                # --------------------------------------------------
                # Full vector error.
                # --------------------------------------------------

                base_vector_error_N=vector_error(
                    base_force
                ),

                global_vector_error_N=vector_error(
                    global_force
                ),

                semantic_vector_error_N=vector_error(
                    semantic_force
                ),

                local_vector_error_N=vector_error(
                    local_force
                ),

                # --------------------------------------------------
                # Magnitude prediction error.
                # --------------------------------------------------

                base_magnitude_error_N=magnitude_error(
                    base_force
                ),

                global_magnitude_error_N=magnitude_error(
                    global_force
                ),

                semantic_magnitude_error_N=magnitude_error(
                    semantic_force
                ),

                local_magnitude_error_N=magnitude_error(
                    local_force
                ),

                has_local_evidence=bool(
                    l["events"] > 0
                ),

                local_beats_base=bool(
                    lmae < bmae
                ),

                local_beats_global=bool(
                    lmae < gmae
                ),

                local_beats_semantic=bool(
                    lmae < smae
                ),
            )
        )

    event_inventory.append(
        dict(
            terrain_type=terrain,

            front_touchdowns_detected=int(
                sum(
                    len(td[x])
                    for x in FRONT_LEGS
                )
            ),

            usable_front_touchdowns=int(
                len(front_events)
            ),

            hind_touchdowns_detected=int(
                sum(
                    len(td[x])
                    for x in HIND_LEGS
                )
            ),

            usable_hind_touchdowns=int(
                len(hind_events)
            ),
        )
    )


# ======================================================================
# 5. RESULTS
# ======================================================================

v9a3_eval = pd.DataFrame(
    all_rows
)

v9a3_event_inventory = (
    pd.DataFrame(
        event_inventory
    )
)

if len(v9a3_eval) == 0:

    raise RuntimeError(
        "V9-A3 produced zero usable hind touchdown events."
    )


# ======================================================================
# 6. SUMMARY FUNCTION
# ======================================================================

def v9a3_summary_for(df):

    if len(df) == 0:
        return {}

    return dict(
        samples=int(
            len(df)
        ),

        local_evidence_fraction=float(
            df.has_local_evidence.mean()
        ),

        base_force_MAE_N=float(
            df.base_force_MAE_N.mean()
        ),

        global_force_MAE_N=float(
            df.global_force_MAE_N.mean()
        ),

        semantic_force_MAE_N=float(
            df.semantic_force_MAE_N.mean()
        ),

        local_force_MAE_N=float(
            df.local_force_MAE_N.mean()
        ),

        base_vector_error_N=float(
            df.base_vector_error_N.mean()
        ),

        global_vector_error_N=float(
            df.global_vector_error_N.mean()
        ),

        semantic_vector_error_N=float(
            df.semantic_vector_error_N.mean()
        ),

        local_vector_error_N=float(
            df.local_vector_error_N.mean()
        ),

        base_magnitude_error_N=float(
            df.base_magnitude_error_N.mean()
        ),

        global_magnitude_error_N=float(
            df.global_magnitude_error_N.mean()
        ),

        semantic_magnitude_error_N=float(
            df.semantic_magnitude_error_N.mean()
        ),

        local_magnitude_error_N=float(
            df.local_magnitude_error_N.mean()
        ),

        local_beats_base_rate=float(
            df.local_beats_base.mean()
        ),

        local_beats_global_rate=float(
            df.local_beats_global.mean()
        ),

        local_beats_semantic_rate=float(
            df.local_beats_semantic.mean()
        ),

        mean_local_front_touchdowns=float(
            df.local_front_touchdowns.mean()
        ),
    )


v9a3_summary = pd.DataFrame(
    [
        dict(
            subset="all_hind_touchdowns",
            **v9a3_summary_for(
                v9a3_eval
            )
        ),

        dict(
            subset="with_local_front_touchdown_evidence",
            **v9a3_summary_for(
                v9a3_eval[
                    v9a3_eval.has_local_evidence
                ]
            )
        ),
    ]
)


# ======================================================================
# 7. INDEPENDENT TOUCHDOWN EXPERIENCE CURVE
# ======================================================================

def v9a3_event_bucket(n):

    n = int(n)

    if n <= 0:
        return "0"

    if n == 1:
        return "1"

    if n <= 3:
        return "2-3"

    if n <= 7:
        return "4-7"

    return "8+"


v9a3_eval[
    "local_touchdown_bucket"
] = (
    v9a3_eval[
        "local_front_touchdowns"
    ]
    .map(
        v9a3_event_bucket
    )
)

bucket_order = [
    "0",
    "1",
    "2-3",
    "4-7",
    "8+"
]

curve_rows = []

for b in bucket_order:

    g = v9a3_eval[
        v9a3_eval.local_touchdown_bucket
        == b
    ]

    if len(g) == 0:
        continue

    curve_rows.append(
        dict(
            local_touchdown_bucket=b,
            **v9a3_summary_for(g)
        )
    )

v9a3_curve = pd.DataFrame(
    curve_rows
)


# ======================================================================
# 8. TERRAIN / LEG BREAKDOWN
# ======================================================================

terrain_rows = []

for (
    terrain,
    leg_name
), g in v9a3_eval.groupby(
    [
        "terrain_type",
        "leg_name"
    ]
):

    terrain_rows.append(
        dict(
            terrain_type=terrain,
            leg_name=leg_name,
            **v9a3_summary_for(g)
        )
    )

v9a3_by_terrain = pd.DataFrame(
    terrain_rows
)


# ======================================================================
# 9. PAIRED BOOTSTRAP
#
# Positive difference means LOCAL is BETTER.
#
# global - local
# semantic - local
# base - local
# ======================================================================

def paired_bootstrap(
    df,
    a_col,
    b_col,
    reps=BOOTSTRAP_REPS,
    seed=BOOTSTRAP_SEED,
):

    d = (
        df[a_col].to_numpy(
            np.float64
        )
        -
        df[b_col].to_numpy(
            np.float64
        )
    )

    d = d[
        np.isfinite(d)
    ]

    if len(d) == 0:

        return dict(
            n=0,
            mean_difference_N=np.nan,
            ci95_low_N=np.nan,
            ci95_high_N=np.nan,
            probability_positive=np.nan,
        )

    rng = np.random.default_rng(
        seed
    )

    boots = np.empty(
        reps,
        np.float64
    )

    n = len(d)

    for r in range(reps):

        ii = rng.integers(
            0,
            n,
            size=n
        )

        boots[r] = float(
            d[ii].mean()
        )

    return dict(
        n=int(n),

        mean_difference_N=float(
            d.mean()
        ),

        ci95_low_N=float(
            np.quantile(
                boots,
                0.025
            )
        ),

        ci95_high_N=float(
            np.quantile(
                boots,
                0.975
            )
        ),

        probability_positive=float(
            np.mean(
                boots > 0
            )
        ),
    )


evidence_df = v9a3_eval[
    v9a3_eval.has_local_evidence
].copy()

boot_rows = []

for label, a in [
    (
        "BASE minus LOCAL",
        "base_force_MAE_N"
    ),
    (
        "GLOBAL minus LOCAL",
        "global_force_MAE_N"
    ),
    (
        "SEMANTIC minus LOCAL",
        "semantic_force_MAE_N"
    ),
]:

    boot_rows.append(
        dict(
            comparison=label,
            **paired_bootstrap(
                evidence_df,
                a,
                "local_force_MAE_N"
            )
        )
    )

v9a3_bootstrap = pd.DataFrame(
    boot_rows
)


# ======================================================================
# 10. SAVE
# ======================================================================

v9a3_eval.to_csv(
    V9_A3_ROOT /
    "V9A3_EVENT_LEVEL_RAW.csv",
    index=False
)

v9a3_summary.to_csv(
    V9_A3_ROOT /
    "V9A3_EVENT_LEVEL_SUMMARY.csv",
    index=False
)

v9a3_curve.to_csv(
    V9_A3_ROOT /
    "V9A3_INDEPENDENT_TOUCHDOWN_CURVE.csv",
    index=False
)

v9a3_by_terrain.to_csv(
    V9_A3_ROOT /
    "V9A3_BY_TERRAIN_LEG.csv",
    index=False
)

v9a3_bootstrap.to_csv(
    V9_A3_ROOT /
    "V9A3_PAIRED_BOOTSTRAP.csv",
    index=False
)

v9a3_event_inventory.to_csv(
    V9_A3_ROOT /
    "V9A3_EVENT_INVENTORY.csv",
    index=False
)


# ======================================================================
# 11. DISPLAY
# ======================================================================

print()
print("=" * 110)
print("TOUCHDOWN EVENT INVENTORY")
print("=" * 110)

display(
    v9a3_event_inventory
)

print()
print("=" * 110)
print("EVENT-LEVEL FRONT -> HIND SUMMARY")
print("=" * 110)

display(
    v9a3_summary
)

print()
print("=" * 110)
print("LOCAL FRONT-TOUCHDOWN EXPERIENCE CURVE")
print("=" * 110)

display(
    v9a3_curve
)

print()
print("=" * 110)
print("BY TERRAIN / HIND LEG")
print("=" * 110)

display(
    v9a3_by_terrain
)

print()
print("=" * 110)
print("PAIRED BOOTSTRAP — POSITIVE = LOCAL WINS")
print("=" * 110)

display(
    v9a3_bootstrap
)


# ======================================================================
# 12. AUTOMATIC RESEARCH INTERPRETATION
# ======================================================================

s = v9a3_summary.iloc[1]

bg = v9a3_bootstrap[
    v9a3_bootstrap.comparison
    ==
    "GLOBAL minus LOCAL"
].iloc[0]

bs = v9a3_bootstrap[
    v9a3_bootstrap.comparison
    ==
    "SEMANTIC minus LOCAL"
].iloc[0]

print()
print("=" * 110)
print("V9-A3 RESEARCH VERDICT")
print("=" * 110)

print(
    "Independent hind touchdowns evaluated:",
    len(v9a3_eval)
)

print(
    "Fraction with prior LOCAL front touchdown:",
    float(
        v9a3_eval.has_local_evidence.mean()
    )
)

print()

print(
    "BASE MAE     :",
    float(
        s.base_force_MAE_N
    )
)

print(
    "GLOBAL MAE   :",
    float(
        s.global_force_MAE_N
    )
)

print(
    "SEMANTIC MAE :",
    float(
        s.semantic_force_MAE_N
    )
)

print(
    "LOCAL MAE    :",
    float(
        s.local_force_MAE_N
    )
)

print()

print(
    "GLOBAL - LOCAL bootstrap CI:",
    (
        float(bg.ci95_low_N),
        float(bg.ci95_high_N)
    )
)

print(
    "SEMANTIC - LOCAL bootstrap CI:",
    (
        float(bs.ci95_low_N),
        float(bs.ci95_high_N)
    )
)

print()

if (
    bg.ci95_low_N > 0
    and
    bs.ci95_low_N > 0
):

    print(
        "✅ STRONG LOCATION-SPECIFIC TRANSFER:"
    )

    print(
        "Independent prior front touchdowns at the same "
        "world terrain location improve later hind prediction "
        "beyond both global calibration and semantic-only memory."
    )

elif (
    s.local_force_MAE_N
    <
    s.global_force_MAE_N
):

    print(
        "🟡 LOCATION SIGNAL PRESENT, BUT NOT YET STATISTICALLY CLEAN."
    )

    print(
        "We next improve the posterior / extrinsic alignment "
        "before making a strong scientific claim."
    )

elif (
    s.semantic_force_MAE_N
    <
    s.global_force_MAE_N
):

    print(
        "🟡 PHYSICAL TERRAIN-CLASS TRANSFER EXISTS, "
        "BUT WORLD LOCATION IS NOT YET ADDING VALUE."
    )

    print(
        "That points us toward a learned physical terrain latent "
        "rather than a strict spatial residual map."
    )

else:

    print(
        "❌ EVENT-LEVEL LOCALITY DID NOT SURVIVE THE STRICT TEST."
    )

    print(
        "The earlier A2 gain was largely repeated-stance calibration. "
        "We do not hide this; we move to V9-C baseline reconstruction."
    )

print()
print(
    "IMPORTANT: contact position remains ORACLE in V9-A3."
)

print(
    "No deployment-level foothold claim is made."
)

print()
print(
    "Saved:",
    V9_A3_ROOT
)

print(
    "✅ V9-A3 COMPLETE"
)

# %% notebook cell 34
# ======================================================================
# V9-C0 — EXACT FROZEN V7 SHORT-PRIOR BASELINE RECONSTRUCTION
#
# PURPOSE
# -------
# Reconstruct the ORIGINAL V7 +0.1 s physical predictor exactly:
#
#     baseline:
#         q(t+h) = q(t) + h*dq(t)
#         dq      = dq(t)
#         tau     = tau(t)
#         F       = F(t)
#         body    = body(t)
#
#     final prediction:
#
#         P_hat = baseline_P + normalized_residual * P_std
#
# This is compared on EXACTLY the independent hind-touchdown samples
# from V9-A3.
#
# No training.
# No V8 modification.
# No V9-A modification.
#
# NOTE:
# Original V7 uses recorded future U/L as known exogenous controls in
# this offline conditional-dynamics benchmark. That convention is kept
# here solely to reproduce the checkpoint faithfully.
# ======================================================================

from pathlib import Path
import math
import json
import numpy as np
import pandas as pd
import torch

print("=" * 110)
print("V9-C0 — EXACT FROZEN V7 SHORT-PRIOR BASELINE")
print("=" * 110)


# ======================================================================
# 0. PREFLIGHT
# ======================================================================

_req = [
    "prior_short",
    "run_records",
    "ANYMAL_D_FK",
    "DEVICE",
    "V9_C_ROOT",
    "v9a3_eval",
]

_missing = [
    x for x in _req
    if x not in globals()
]

if _missing:
    raise RuntimeError(
        "V9-C0 missing live objects: "
        + repr(_missing)
    )

prior_short.eval()

for p in prior_short.parameters():
    p.requires_grad_(False)

V9C0_ROOT = (
    V9_C_ROOT /
    "C0_exact_v7_baseline"
)

V9C0_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ======================================================================
# 1. EXACT ORIGINAL V7 CONSTANTS
# ======================================================================

V9C0_CONTACT_THRESHOLD = 10.0
V9C0_EPS = 1e-6

V9C0_N_REGIMES = 4

V9C0_PI_PHYS_DIM = 47

# Exact original V7 foot-query conditioning horizon.
V9C0_FOOT_QUERY_PLAN_SEC = 0.30

V9C0_HORIZON_S = 0.10

V9C0_RIDGE = 1e-3


# Recover history length from the live model, not a hardcoded guess.
if hasattr(
    prior_short,
    "history_steps"
):
    V9C0_HISTORY_STEPS = int(
        prior_short.history_steps
    )
else:
    V9C0_HISTORY_STEPS = int(
        prior_short
        .leg_encoder
        .history_steps
    )

print(
    "Frozen V7 history steps:",
    V9C0_HISTORY_STEPS
)


# ======================================================================
# 2. ORIGINAL V7 NORMALIZATION
# ======================================================================

def v9c0_make_stats(
    data,
    start_time,
    end_time,
):

    t = np.asarray(
        data["time"],
        np.float64
    )

    mask = (
        (t >= float(start_time))
        &
        (t <= float(end_time))
    )

    if int(mask.sum()) < 50:
        raise RuntimeError(
            "Too few samples for V7 stats."
        )

    P = np.asarray(
        data["P"][mask],
        np.float32
    )

    B = np.asarray(
        data["B"][mask],
        np.float32
    )

    U = np.asarray(
        data["U"][mask],
        np.float32
    )

    return dict(
        P_mean=P.mean(
            axis=(0, 1)
        ).astype(np.float32),

        P_std=(
            P.std(
                axis=(0, 1)
            )
            + V9C0_EPS
        ).astype(np.float32),

        B_mean=B.mean(
            axis=0
        ).astype(np.float32),

        B_std=(
            B.std(
                axis=0
            )
            + V9C0_EPS
        ).astype(np.float32),

        U_mean=U.mean(
            axis=0
        ).astype(np.float32),

        U_std=(
            U.std(
                axis=0
            )
            + V9C0_EPS
        ).astype(np.float32),
    )


def v9c0_normalize(
    data,
    stats,
):

    P = (
        np.asarray(
            data["P"],
            np.float32
        )
        -
        stats["P_mean"]
    ) / stats["P_std"]

    B = (
        np.asarray(
            data["B"],
            np.float32
        )
        -
        stats["B_mean"]
    ) / stats["B_std"]

    U = (
        np.asarray(
            data["U"],
            np.float32
        )
        -
        stats["U_mean"]
    ) / stats["U_std"]

    L = (
        np.asarray(
            data["L_LEG"],
            np.float32
        )
        -
        stats["P_mean"][6:9]
    ) / stats["P_std"][6:9]

    E = (
        np.asarray(
            data["E"],
            np.float32
        )
        /
        stats["P_std"][6:9]
    )

    return (
        P.astype(np.float32),
        B.astype(np.float32),
        U.astype(np.float32),
        L.astype(np.float32),
        E.astype(np.float32),
    )


# ======================================================================
# 3. ORIGINAL GLOBAL CONTACT REGIME THRESHOLDS
#
# Learned only from pre-first-test data across the 4 runs.
# ======================================================================

def v9c0_force_regime(
    force_xyz,
    thresholds,
):

    mag = np.linalg.norm(
        force_xyz,
        axis=-1
    )

    out = np.zeros_like(
        mag,
        dtype=np.int64
    )

    out[
        mag >= thresholds[0]
    ] = 1

    out[
        mag >= thresholds[1]
    ] = 2

    out[
        mag >= thresholds[2]
    ] = 3

    return out


positive_force = []

for rec in run_records:

    data = rec["data"]

    mask = (
        data["time"]
        <= float(
            min(
                rec["ttests"]
            )
        )
        + 1e-9
    )

    mag = np.linalg.norm(
        data["P"][
            mask,
            :,
            9:12
        ],
        axis=-1
    ).reshape(-1)

    positive_force.append(
        mag[
            mag
            >=
            V9C0_CONTACT_THRESHOLD
        ]
    )

positive_force = np.concatenate(
    positive_force
)

if len(positive_force) < 100:
    raise RuntimeError(
        "Insufficient positive force samples "
        "for exact V7 regime reconstruction."
    )

q1, q2 = np.quantile(
    positive_force,
    [
        1.0 / 3.0,
        2.0 / 3.0,
    ]
)

q1 = max(
    float(q1),
    V9C0_CONTACT_THRESHOLD
    + 1e-3
)

q2 = max(
    float(q2),
    q1 + 1e-3
)

V9C0_REGIME_THRESHOLDS = (
    np.asarray(
        [
            V9C0_CONTACT_THRESHOLD,
            q1,
            q2,
        ],
        np.float32
    )
)

print(
    "Reconstructed V7 regime thresholds:",
    V9C0_REGIME_THRESHOLDS
)


# ======================================================================
# 4. ORIGINAL V7 SOFT PHYSICS PROXY
# ======================================================================

def v9c0_ridge_fit(
    X,
    Y,
    ridge=V9C0_RIDGE,
):

    X = np.asarray(
        X,
        np.float64
    )

    Y = np.asarray(
        Y,
        np.float64
    )

    scale = np.sqrt(
        (
            X * X
        ).mean(axis=0)
        + 1e-8
    )

    Xs = (
        X /
        scale[None, :]
    )

    A = (
        Xs.T @ Xs
        +
        ridge
        *
        np.eye(
            Xs.shape[1]
        )
    )

    B = (
        Xs.T @ Y
    )

    W_scaled = (
        np.linalg.solve(
            A,
            B
        )
    )

    W = (
        W_scaled /
        scale[:, None]
    )

    return W.astype(
        np.float32
    )


def v9c0_fit_physics(
    data,
    start_time,
    end_time,
):

    t = np.asarray(
        data["time"],
        np.float64
    )

    P = np.asarray(
        data["P"],
        np.float32
    )

    B = np.asarray(
        data["B"],
        np.float32
    )

    # Faithfully reproduce original V7 derivative construction.
    dq = (
        P[
            :,
            :,
            3:6
        ]
        .astype(np.float64)
    )

    ddq = np.gradient(
        dq,
        t,
        axis=0
    )

    bad = ~np.isfinite(
        ddq
    )

    if bad.any():
        ddq[bad] = 0.0

    lim = (
        np.quantile(
            np.abs(ddq),
            0.995,
            axis=0,
            keepdims=True
        )
        + 1e-6
    )

    ddq = np.clip(
        ddq,
        -lim,
        lim
    ).astype(np.float32)

    mask = (
        (t >= float(start_time))
        &
        (t <= float(end_time))
    )

    ids = np.flatnonzero(
        mask
    )

    if len(ids) < 100:
        raise RuntimeError(
            "Physics interval too short."
        )

    invW = []

    for leg in range(4):

        q = P[
            ids,
            leg,
            0:3
        ]

        dq_leg = P[
            ids,
            leg,
            3:6
        ]

        tau = P[
            ids,
            leg,
            6:9
        ]

        F = P[
            ids,
            leg,
            9:12
        ]

        ddq_leg = ddq[
            ids,
            leg
        ]

        cls = v9c0_force_regime(
            F,
            V9C0_REGIME_THRESHOLDS
        )

        onehot = np.eye(
            V9C0_N_REGIMES,
            dtype=np.float32
        )[cls]

        X = np.concatenate(
            [
                ddq_leg,
                dq_leg,
                np.sin(q),
                np.cos(q),
                F,
                onehot,
                np.ones(
                    (
                        len(ids),
                        1
                    ),
                    np.float32
                ),
            ],
            axis=1
        )

        invW.append(
            v9c0_ridge_fit(
                X,
                tau
            )
        )

    invW = np.stack(
        invW,
        axis=0
    ).astype(np.float32)

    sumF = P[
        ids,
        :,
        9:12
    ].sum(
        axis=1
    )

    Xb = np.concatenate(
        [
            sumF,
            np.ones(
                (
                    len(ids),
                    1
                ),
                np.float32
            )
        ],
        axis=1
    )

    bodyW = v9c0_ridge_fit(
        Xb,
        B[
            ids,
            0:3
        ]
    )

    return dict(
        invW=invW,
        bodyW=bodyW,
    )


def v9c0_physics_tau(
    physics,
    q,
    dq,
    ddq,
    force,
    regime_onehot,
):

    X = np.concatenate(
        [
            ddq,
            dq,
            np.sin(q),
            np.cos(q),
            force,
            regime_onehot,
            np.ones(
                (
                    4,
                    1
                ),
                np.float32
            ),
        ],
        axis=1
    ).astype(np.float32)

    return np.einsum(
        "lf,lfj->lj",
        X,
        physics["invW"]
    ).astype(np.float32)


def v9c0_physics_acc(
    physics,
    force,
):

    X = np.concatenate(
        [
            force.sum(
                axis=0
            ),
            np.asarray(
                [1.0],
                np.float32
            )
        ]
    )

    return (
        X
        @
        physics["bodyW"]
    ).astype(np.float32)


# ======================================================================
# 5. PRECOMPUTE RUN-SPECIFIC V7 NORMALIZATION + PHYSICS
#
# Exactly as global prior training used:
# start of run -> first fixed diagnostic test.
# ======================================================================

v9c0_aux = {}

for rec in run_records:

    key = (
        rec["terrain_type"],
        rec["environment"],
        rec["run"]
    )

    data = rec["data"]

    start = float(
        data["time"][0]
    )

    end = float(
        min(
            rec["ttests"]
        )
    )

    stats = v9c0_make_stats(
        data,
        start,
        end
    )

    norm = v9c0_normalize(
        data,
        stats
    )

    physics = v9c0_fit_physics(
        data,
        start,
        end
    )

    v9c0_aux[key] = dict(
        stats=stats,
        norm=norm,
        physics=physics,
    )

    print(
        "✅ reconstructed preprocessing:",
        key
    )


# ======================================================================
# 6. EXACT ONE-SAMPLE V7 SHORT-PRIOR INFERENCE
# ======================================================================

def v9c0_tensor(x):

    return torch.as_tensor(
        x,
        dtype=torch.float32,
        device=DEVICE
    )


@torch.inference_mode()
def v9c0_predict(
    rec,
    idx,
):

    data = rec["data"]

    visual = rec["visual"]

    key = (
        rec["terrain_type"],
        rec["environment"],
        rec["run"]
    )

    aux = v9c0_aux[key]

    stats = aux["stats"]

    Pn, Bn, Un, Ln, En = (
        aux["norm"]
    )

    physics = aux[
        "physics"
    ]

    idx = int(idx)

    h0 = (
        idx
        -
        V9C0_HISTORY_STEPS
        + 1
    )

    if h0 < 0:
        return None

    dt = float(
        np.median(
            np.diff(
                data["time"]
            )
        )
    )

    off = int(
        round(
            V9C0_HORIZON_S
            /
            dt
        )
    )

    ti = (
        idx + off
    )

    if ti >= len(
        data["time"]
    ):
        return None

    vi = int(
        visual[
            "visual_index"
        ][idx]
    )

    if vi < 0:
        return None

    if (
        float(
            visual[
                "camera_times"
            ][vi]
        )
        >
        float(
            data["time"][idx]
        )
        + 1e-9
    ):
        raise RuntimeError(
            "Future visual leakage "
            "in V9-C0."
        )

    # --------------------------------------------------------------
    # Current raw state.
    # --------------------------------------------------------------

    curP = np.asarray(
        data["P"][idx],
        np.float32
    )

    curB = np.asarray(
        data["B"][idx],
        np.float32
    )

    # --------------------------------------------------------------
    # Exact V7 foot query.
    # --------------------------------------------------------------

    foot_pos = (
        ANYMAL_D_FK
        .foot_positions(
            curP[
                :,
                0:3
            ]
        )
        .astype(np.float32)
    )

    curUraw = np.asarray(
        data["U"][idx],
        np.float32
    )

    planned_body_delta = np.tile(
        (
            curUraw[:3]
            *
            V9C0_FOOT_QUERY_PLAN_SEC
        )[None, :],
        (
            4,
            1
        )
    ).astype(np.float32)

    foot_query_kin = np.concatenate(
        [
            foot_pos,
            planned_body_delta
        ],
        axis=-1
    ).astype(np.float32)

    # --------------------------------------------------------------
    # Exact V7 persistence baseline.
    #
    # q gets linear motion extrapolation.
    # dq / tau / force persist.
    # --------------------------------------------------------------

    bp = curP.copy()

    bp[
        :,
        0:3
    ] = (
        curP[
            :,
            0:3
        ]
        +
        V9C0_HORIZON_S
        *
        curP[
            :,
            3:6
        ]
    )

    # body baseline is persistence.
    bb = curB.copy()

    # --------------------------------------------------------------
    # Future recorded U/L.
    #
    # Original V7 offline benchmark treats these as known exogenous
    # controls; this is NOT deployment sensor leakage.
    # --------------------------------------------------------------

    us = Un[
        idx:ti
    ]

    ls = Ln[
        idx:ti
    ]

    if len(us) == 0:

        us = Un[
            idx:idx + 1
        ]

        ls = Ln[
            idx:idx + 1
        ]

    control = np.concatenate(
        [
            us[-1],

            ls[-1].reshape(
                -1
            ),

            us.mean(
                axis=0
            ),

            ls.mean(
                axis=0
            ).reshape(
                -1
            ),
        ]
    ).astype(np.float32)

    if control.shape[0] != 36:
        raise RuntimeError(
            "V7 control feature "
            "dimension != 36."
        )

    # --------------------------------------------------------------
    # Exact V7 physics features.
    # --------------------------------------------------------------

    curF = curP[
        :,
        9:12
    ]

    cur_cls = v9c0_force_regime(
        curF,
        V9C0_REGIME_THRESHOLDS
    )

    cur_oh = np.eye(
        V9C0_N_REGIMES,
        dtype=np.float32
    )[cur_cls]

    fscale = max(
        float(
            V9C0_REGIME_THRESHOLDS[
                -1
            ]
        ),
        1.0
    )

    qcv = bp[
        :,
        0:3
    ]

    dq0 = curP[
        :,
        3:6
    ]

    ddq0 = np.zeros(
        (
            4,
            3
        ),
        np.float32
    )

    tau_phys = v9c0_physics_tau(
        physics,
        qcv,
        dq0,
        ddq0,
        curF,
        cur_oh
    )

    acc_phys = v9c0_physics_acc(
        physics,
        curF
    )

    qn = (
        (
            qcv
            -
            stats["P_mean"][
                0:3
            ]
        )
        /
        stats["P_std"][
            0:3
        ]
    ).reshape(-1)

    taun = (
        (
            tau_phys
            -
            stats["P_mean"][
                6:9
            ]
        )
        /
        stats["P_std"][
            6:9
        ]
    ).reshape(-1)

    accn = (
        (
            acc_phys
            -
            stats["B_mean"][
                0:3
            ]
        )
        /
        stats["B_std"][
            0:3
        ]
    )

    fn = (
        np.linalg.norm(
            curF,
            axis=-1
        )
        /
        fscale
    )

    phys = np.concatenate(
        [
            qn,
            taun,
            accn,
            fn,
            cur_oh.reshape(
                -1
            ),
        ]
    ).astype(np.float32)

    if len(phys) != V9C0_PI_PHYS_DIM:

        raise RuntimeError(
            "V7 physics feature "
            f"dim {len(phys)} != 47"
        )

    # --------------------------------------------------------------
    # History tensors.
    # --------------------------------------------------------------

    p_hist = v9c0_tensor(
        Pn[
            h0:idx + 1
        ]
        .transpose(
            1,
            0,
            2
        )[None]
    )

    b_hist = v9c0_tensor(
        Bn[
            h0:idx + 1
        ][None]
    )

    u_hist = v9c0_tensor(
        Un[
            h0:idx + 1
        ][None]
    )

    l_hist = v9c0_tensor(
        Ln[
            h0:idx + 1
        ]
        .transpose(
            1,
            0,
            2
        )[None]
    )

    e_hist = v9c0_tensor(
        En[
            h0:idx + 1
        ]
        .transpose(
            1,
            0,
            2
        )[None]
    )

    # --------------------------------------------------------------
    # Cached four-stream visual frame.
    # --------------------------------------------------------------

    dino = v9c0_tensor(
        np.asarray(
            visual[
                "DINO"
            ][vi],
            np.float32
        )[None]
    )

    conv = v9c0_tensor(
        np.asarray(
            visual[
                "CONV"
            ][vi],
            np.float32
        )[None]
    )

    geom = v9c0_tensor(
        np.asarray(
            visual[
                "GEOM"
            ][vi],
            np.float32
        )[None]
    )

    flow = v9c0_tensor(
        np.asarray(
            visual[
                "FLOW"
            ][vi],
            np.float32
        )[None]
    )

    future_control = (
        v9c0_tensor(
            control[
                None,
                None,
                :
            ]
        )
    )

    horizon = v9c0_tensor(
        np.asarray(
            [
                [
                    [
                        V9C0_HORIZON_S
                    ]
                ]
            ],
            np.float32
        )
    )

    physics_features = (
        v9c0_tensor(
            phys[
                None,
                None,
                :
            ]
        )
    )

    foot_query = (
        v9c0_tensor(
            foot_query_kin[
                None
            ]
        )
    )

    # --------------------------------------------------------------
    # Frozen ORIGINAL V7 model.
    # --------------------------------------------------------------

    out = prior_short(
        p_hist,
        b_hist,
        u_hist,
        l_hist,
        e_hist,
        dino,
        conv,
        geom,
        flow,
        foot_query,
        future_control,
        horizon,
        physics_features,
    )

    residual_p = (
        out[
            "residual_p"
        ][
            0,
            0
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    # --------------------------------------------------------------
    # EXACT V7 physical reconstruction.
    #
    # IMPORTANT:
    # residual is normalized.
    # --------------------------------------------------------------

    predP = (
        bp
        +
        residual_p
        *
        stats[
            "P_std"
        ][None, :]
    ).astype(np.float32)

    contact_probability = (
        torch.sigmoid(
            out[
                "binary_contact_logits"
            ][
                0,
                0
            ]
        )
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    return dict(
        predP=predP,

        baselineP=bp,

        residual_p=residual_p,

        contact_probability=(
            contact_probability
        ),

        vi=int(vi),

        target_index=int(ti),
    )


# ======================================================================
# 7. RE-EVALUATE EXACT V9-A3 HIND TOUCHDOWNS
# ======================================================================

rec_lookup = {
    (
        r[
            "terrain_type"
        ],
        r[
            "environment"
        ],
        r[
            "run"
        ]
    ): r

    for r in run_records
}


rows = []

N = len(
    v9a3_eval
)

print()
print(
    "Evaluating",
    N,
    "independent hind touchdowns..."
)

for k, r in v9a3_eval.iterrows():

    key = (
        r[
            "terrain_type"
        ],
        r[
            "environment"
        ],
        r[
            "run"
        ]
    )

    rec = rec_lookup[
        key
    ]

    data = rec[
        "data"
    ]

    # query_time came directly from this run's own timeline.
    idx = int(
        np.argmin(
            np.abs(
                data[
                    "time"
                ]
                -
                float(
                    r[
                        "query_time"
                    ]
                )
            )
        )
    )

    leg = int(
        r[
            "leg"
        ]
    )

    pr = v9c0_predict(
        rec,
        idx
    )

    if pr is None:
        continue

    j = int(
        pr[
            "target_index"
        ]
    )

    gt_force = np.asarray(
        data[
            "P"
        ][
            j,
            leg,
            9:12
        ],
        np.float32
    )

    current_force = np.asarray(
        data[
            "P"
        ][
            idx,
            leg,
            9:12
        ],
        np.float32
    )

    v7_force = np.asarray(
        pr[
            "predP"
        ][
            leg,
            9:12
        ],
        np.float32
    )

    gt_tau = np.asarray(
        data[
            "P"
        ][
            j,
            leg,
            6:9
        ],
        np.float32
    )

    current_tau = np.asarray(
        data[
            "P"
        ][
            idx,
            leg,
            6:9
        ],
        np.float32
    )

    v7_tau = np.asarray(
        pr[
            "predP"
        ][
            leg,
            6:9
        ],
        np.float32
    )

    def mae3(a, b):

        return float(
            np.mean(
                np.abs(
                    np.asarray(a)
                    -
                    np.asarray(b)
                )
            )
        )

    def vecerr(a, b):

        return float(
            np.linalg.norm(
                np.asarray(a)
                -
                np.asarray(b)
            )
        )

    def magerr(a, b):

        return float(
            abs(
                np.linalg.norm(a)
                -
                np.linalg.norm(b)
            )
        )

    rows.append(
        dict(
            terrain_type=r[
                "terrain_type"
            ],

            environment=r[
                "environment"
            ],

            run=r[
                "run"
            ],

            leg=int(
                leg
            ),

            leg_name=r[
                "leg_name"
            ],

            query_time=float(
                r[
                    "query_time"
                ]
            ),

            touchdown_time=float(
                r[
                    "touchdown_time"
                ]
            ),

            current_contact=float(
                data[
                    "contact"
                ][
                    idx,
                    leg
                ]
            ),

            # ------------------------------------------------------
            # Persistence baseline.
            # ------------------------------------------------------

            persistence_force_MAE_N=mae3(
                current_force,
                gt_force
            ),

            persistence_vector_error_N=vecerr(
                current_force,
                gt_force
            ),

            persistence_magnitude_error_N=magerr(
                current_force,
                gt_force
            ),

            persistence_tau_MAE_Nm=mae3(
                current_tau,
                gt_tau
            ),

            # ------------------------------------------------------
            # Frozen V8-C error from exact same A3 sample.
            # ------------------------------------------------------

            v8c_force_MAE_N=float(
                r[
                    "base_force_MAE_N"
                ]
            ),

            v8c_vector_error_N=float(
                r[
                    "base_vector_error_N"
                ]
            ),

            v8c_magnitude_error_N=float(
                r[
                    "base_magnitude_error_N"
                ]
            ),

            # ------------------------------------------------------
            # Exact frozen V7 short prior.
            # ------------------------------------------------------

            v7_force_MAE_N=mae3(
                v7_force,
                gt_force
            ),

            v7_vector_error_N=vecerr(
                v7_force,
                gt_force
            ),

            v7_magnitude_error_N=magerr(
                v7_force,
                gt_force
            ),

            v7_tau_MAE_Nm=mae3(
                v7_tau,
                gt_tau
            ),

            v7_contact_probability=float(
                pr[
                    "contact_probability"
                ][leg]
            ),

            # paired outcomes
            v7_beats_v8c=bool(
                mae3(
                    v7_force,
                    gt_force
                )
                <
                float(
                    r[
                        "base_force_MAE_N"
                    ]
                )
            ),

            v7_beats_persistence=bool(
                mae3(
                    v7_force,
                    gt_force
                )
                <
                mae3(
                    current_force,
                    gt_force
                )
            ),
        )
    )

    if (
        len(rows) % 200
        == 0
    ):
        print(
            f"  {len(rows):4d}/{N}"
        )


v9c0_eval = pd.DataFrame(
    rows
)

if len(v9c0_eval) == 0:

    raise RuntimeError(
        "V9-C0 produced no evaluation rows."
    )


# ======================================================================
# 8. SUMMARY
# ======================================================================

def v9c0_summarize(df):

    return dict(
        samples=int(
            len(df)
        ),

        query_contact_rate=float(
            df[
                "current_contact"
            ].mean()
        ),

        persistence_force_MAE_N=float(
            df[
                "persistence_force_MAE_N"
            ].mean()
        ),

        v8c_force_MAE_N=float(
            df[
                "v8c_force_MAE_N"
            ].mean()
        ),

        v7_force_MAE_N=float(
            df[
                "v7_force_MAE_N"
            ].mean()
        ),

        persistence_vector_error_N=float(
            df[
                "persistence_vector_error_N"
            ].mean()
        ),

        v8c_vector_error_N=float(
            df[
                "v8c_vector_error_N"
            ].mean()
        ),

        v7_vector_error_N=float(
            df[
                "v7_vector_error_N"
            ].mean()
        ),

        persistence_magnitude_error_N=float(
            df[
                "persistence_magnitude_error_N"
            ].mean()
        ),

        v8c_magnitude_error_N=float(
            df[
                "v8c_magnitude_error_N"
            ].mean()
        ),

        v7_magnitude_error_N=float(
            df[
                "v7_magnitude_error_N"
            ].mean()
        ),

        persistence_tau_MAE_Nm=float(
            df[
                "persistence_tau_MAE_Nm"
            ].mean()
        ),

        v7_tau_MAE_Nm=float(
            df[
                "v7_tau_MAE_Nm"
            ].mean()
        ),

        v7_beats_v8c_rate=float(
            df[
                "v7_beats_v8c"
            ].mean()
        ),

        v7_beats_persistence_rate=float(
            df[
                "v7_beats_persistence"
            ].mean()
        ),

        v8c_force_p95_N=float(
            df[
                "v8c_vector_error_N"
            ].quantile(
                0.95
            )
        ),

        v7_force_p95_N=float(
            df[
                "v7_vector_error_N"
            ].quantile(
                0.95
            )
        ),
    )


v9c0_summary = pd.DataFrame(
    [
        dict(
            subset=(
                "all_independent_hind_touchdowns"
            ),
            **v9c0_summarize(
                v9c0_eval
            )
        )
    ]
)


# ======================================================================
# 9. TERRAIN + LEG
# ======================================================================

group_rows = []

for (
    terrain,
    leg
), g in v9c0_eval.groupby(
    [
        "terrain_type",
        "leg_name"
    ]
):

    group_rows.append(
        dict(
            terrain_type=terrain,
            leg_name=leg,
            **v9c0_summarize(g)
        )
    )

v9c0_by_terrain = pd.DataFrame(
    group_rows
)


# ======================================================================
# 10. PAIRED BOOTSTRAP
#
# Positive number means V7 is better.
# ======================================================================

def v9c0_bootstrap_difference(
    a,
    b,
    reps=3000,
    seed=781,
):

    # a - b
    # positive -> b has lower error.
    d = (
        np.asarray(
            a,
            np.float64
        )
        -
        np.asarray(
            b,
            np.float64
        )
    )

    d = d[
        np.isfinite(d)
    ]

    rng = np.random.default_rng(
        seed
    )

    boots = np.empty(
        reps,
        np.float64
    )

    n = len(d)

    for i in range(reps):

        ids = rng.integers(
            0,
            n,
            size=n
        )

        boots[i] = float(
            d[
                ids
            ].mean()
        )

    return dict(
        n=int(n),

        mean_difference=float(
            d.mean()
        ),

        ci95_low=float(
            np.quantile(
                boots,
                0.025
            )
        ),

        ci95_high=float(
            np.quantile(
                boots,
                0.975
            )
        ),

        probability_positive=float(
            np.mean(
                boots > 0
            )
        ),
    )


v9c0_bootstrap = pd.DataFrame(
    [
        dict(
            comparison=(
                "V8-C minus exact V7"
            ),
            **v9c0_bootstrap_difference(
                v9c0_eval[
                    "v8c_force_MAE_N"
                ],
                v9c0_eval[
                    "v7_force_MAE_N"
                ],
            )
        ),

        dict(
            comparison=(
                "persistence minus exact V7"
            ),
            **v9c0_bootstrap_difference(
                v9c0_eval[
                    "persistence_force_MAE_N"
                ],
                v9c0_eval[
                    "v7_force_MAE_N"
                ],
            )
        ),
    ]
)


# ======================================================================
# 11. SAVE
# ======================================================================

v9c0_eval.to_csv(
    V9C0_ROOT /
    "V9C0_EXACT_V7_TOUCHDOWN_RAW.csv",
    index=False
)

v9c0_summary.to_csv(
    V9C0_ROOT /
    "V9C0_EXACT_V7_TOUCHDOWN_SUMMARY.csv",
    index=False
)

v9c0_by_terrain.to_csv(
    V9C0_ROOT /
    "V9C0_EXACT_V7_BY_TERRAIN_LEG.csv",
    index=False
)

v9c0_bootstrap.to_csv(
    V9C0_ROOT /
    "V9C0_EXACT_V7_BOOTSTRAP.csv",
    index=False
)


# ======================================================================
# 12. DISPLAY
# ======================================================================

print()
print("=" * 110)
print("V9-C0 EXACT V7 BASELINE SUMMARY")
print("=" * 110)

display(
    v9c0_summary
)

print()
print(
    "BY TERRAIN / HIND LEG"
)

display(
    v9c0_by_terrain
)

print()
print(
    "PAIRED BOOTSTRAP — POSITIVE = EXACT V7 WINS"
)

display(
    v9c0_bootstrap
)

s = v9c0_summary.iloc[0]

print()
print("=" * 110)
print("V9-C0 DECISION")
print("=" * 110)

print(
    "Persistence force MAE:",
    float(
        s[
            "persistence_force_MAE_N"
        ]
    )
)

print(
    "V8-C force MAE:",
    float(
        s[
            "v8c_force_MAE_N"
        ]
    )
)

print(
    "Exact frozen V7 force MAE:",
    float(
        s[
            "v7_force_MAE_N"
        ]
    )
)

print()

print(
    "V8-C vector error:",
    float(
        s[
            "v8c_vector_error_N"
        ]
    )
)

print(
    "Exact V7 vector error:",
    float(
        s[
            "v7_vector_error_N"
        ]
    )
)

print()

print(
    "V8-C p95 vector error:",
    float(
        s[
            "v8c_force_p95_N"
        ]
    )
)

print(
    "Exact V7 p95 vector error:",
    float(
        s[
            "v7_force_p95_N"
        ]
    )
)

print()

v8_v7 = (
    v9c0_bootstrap[
        v9c0_bootstrap[
            "comparison"
        ]
        ==
        "V8-C minus exact V7"
    ]
    .iloc[0]
)

if (
    v8_v7[
        "ci95_low"
    ]
    > 0
):

    print(
        "✅ EXACT V7 IS A SIGNIFICANTLY BETTER "
        "TOUCHDOWN BASELINE THAN V8-C."
    )

    print(
        "Next: V9-C1 learns ONLY a bounded terrain/contact "
        "residual on top of V7."
    )

elif (
    float(
        s[
            "v7_force_MAE_N"
        ]
    )
    <
    float(
        s[
            "v8c_force_MAE_N"
        ]
    )
):

    print(
        "🟡 V7 IS BETTER, BUT THE PAIRED ADVANTAGE "
        "IS NOT YET STATISTICALLY CLEAN."
    )

    print(
        "We still use V7 as the structural baseline, "
        "but inspect the weak terrain/leg cases."
    )

else:

    print(
        "❌ FROZEN GLOBAL V7 DOES NOT BEAT V8-C "
        "ON TRUE TOUCHDOWN EVENTS."
    )

    print(
        "Next step is NOT to discard V7 architecture; "
        "we test the original local-adaptation path and "
        "current-force/load-sharing baselines separately."
    )

print()
print(
    "Saved:",
    V9C0_ROOT
)

print(
    "✅ V9-C0 COMPLETE — no training performed."
)

# %% notebook cell 35
# ======================================================================
# V9-C1 — EVENT-CENTRIC JOINT LOAD-SHARING RESIDUAL
#
# QUESTION
# --------
# On TRUE touchdown events, can we improve frozen V8-C using:
#
#   1. full four-leg state
#   2. body dynamics / current stance load
#   3. controller state
#   4. optional V9 multi-observation terrain belief
#
# TWO MATCHED MODELS
# ------------------
# C1-DYN:
#     V8-C prediction + joint/body dynamics context
#
# C1-BELIEF:
#     exact same model + V9-A terrain belief
#
# Both predict ONLY a correction to FORCE MAGNITUDE.
# V8-C retains responsibility for force direction.
#
# TRAIN:
#     genuine touchdown events strictly BEFORE first fixed test
#
# VALIDATION:
#     final pre-test window with safety gap
#
# TEST:
#     genuine HIND touchdowns AFTER first fixed test
#
# Oracle target contact position is still used.
# No claim of deployment-level touchdown localization yet.
# ======================================================================

from pathlib import Path
import copy
import math
import random
import json

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


print("=" * 112)
print("V9-C1 — EVENT-CENTRIC JOINT LOAD-SHARING RESIDUAL")
print("=" * 112)


# ======================================================================
# 0. PREFLIGHT
# ======================================================================

_req = [
    "run_records",
    "v8_memory",
    "v9_belief_memory",
    "v8_foot_world_proxy",
    "v8c_predict_cell",
    "v8_numeric_leg_jacobian",
    "ANYMAL_D_FK",
    "V9_C_ROOT",
    "DEVICE",
]

_missing = [
    x for x in _req
    if x not in globals()
]

if _missing:
    raise RuntimeError(
        "V9-C1 missing: " + repr(_missing)
    )


V9C1_ROOT = (
    V9_C_ROOT /
    "C1_event_joint_residual"
)

V9C1_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ======================================================================
# 1. CONFIG
# ======================================================================

V9C1_HORIZON_S = 0.10

# Reproduce V8-C's safe pre-test training philosophy.
V9C1_PRETEST_BUFFER_S = 1.10

# Chronological validation window immediately before buffer.
V9C1_VAL_WINDOW_S = 4.0

# Small train -> validation gap.
V9C1_TRAINVAL_GAP_S = 0.25

# Test begins after first diagnostic time.
V9C1_TEST_BUFFER_S = 0.25

V9C1_BATCH = 128
V9C1_MAX_EPOCHS = 180
V9C1_PATIENCE = 22

V9C1_LR = 1.0e-3
V9C1_WEIGHT_DECAY = 2.0e-4

V9C1_SEED = 9781

# Magnitude correction cannot become arbitrarily large.
V9C1_DELTA_CAP_N = 140.0

FRONT_LEGS = (0, 1)
HIND_LEGS = (2, 3)

LEG_NAMES = {
    0: "FL",
    1: "FR",
    2: "RL",
    3: "RR",
}


# ======================================================================
# 2. DETERMINISM
# ======================================================================

def v9c1_seed(s):

    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


v9c1_seed(
    V9C1_SEED
)


# ======================================================================
# 3. TRUE TOUCHDOWN DETECTOR
# ======================================================================

def v9c1_touchdowns(c):

    c = (
        np.asarray(c)
        >= 0.5
    )

    if len(c) < 2:
        return np.empty(
            0,
            dtype=np.int64
        )

    return (
        np.where(
            c[1:]
            &
            (~c[:-1])
        )[0]
        + 1
    ).astype(np.int64)


# ======================================================================
# 4. FORCE MAGNITUDE CORRECTION
# ======================================================================

def v9c1_apply_delta(
    base_force,
    delta_mag,
):

    f = np.asarray(
        base_force,
        np.float64
    )

    mag = float(
        np.linalg.norm(f)
    )

    if mag < 1e-6:

        # We refuse to invent an arbitrary direction.
        return f.astype(
            np.float32
        )

    dm = float(
        np.clip(
            delta_mag,
            -V9C1_DELTA_CAP_N,
            V9C1_DELTA_CAP_N
        )
    )

    new_mag = max(
        0.0,
        mag + dm
    )

    return (
        f *
        (
            new_mag /
            (
                mag
                + 1e-9
            )
        )
    ).astype(np.float32)


# ======================================================================
# 5. FULL FOUR-LEG KINEMATICS
# ======================================================================

def v9c1_foot_kinematics(
    rec,
    idx,
):

    P = np.asarray(
        rec["data"]["P"][idx],
        np.float32
    )

    q4 = P[
        :,
        0:3
    ]

    dq4 = P[
        :,
        3:6
    ]

    pos = (
        ANYMAL_D_FK
        .foot_positions(q4)
        .astype(np.float32)
    )

    vel = np.zeros(
        (
            4,
            3
        ),
        np.float32
    )

    for leg in range(4):

        J = v8_numeric_leg_jacobian(
            q4[leg],
            leg
        )

        vel[leg] = (
            J
            @
            dq4[leg]
        ).astype(np.float32)

    return pos, vel


# ======================================================================
# 6. DYNAMICS FEATURE
#
# Explicitly gives the model ALL FOUR LEGS.
#
# No future state.
# ======================================================================

def v9c1_dynamics_feature(
    rec,
    idx,
    target_leg,
    base_pred,
):

    data = rec[
        "data"
    ]

    P = np.asarray(
        data["P"][idx],
        np.float32
    )

    B = np.asarray(
        data["B"][idx],
        np.float32
    )

    U = np.asarray(
        data["U"][idx],
        np.float32
    )

    contact = np.asarray(
        data["contact"][idx],
        np.float32
    )

    L = np.asarray(
        data["L_LEG"][idx],
        np.float32
    )

    E = np.asarray(
        data["E"][idx],
        np.float32
    )

    pos, vel = (
        v9c1_foot_kinematics(
            rec,
            idx
        )
    )

    leg_oh = np.eye(
        4,
        dtype=np.float32
    )[int(target_leg)]

    base_tau = np.asarray(
        base_pred["tau"],
        np.float32
    )

    base_force = np.asarray(
        base_pred["force"],
        np.float32
    )

    tau_sigma = np.asarray(
        base_pred["tau_sigma"],
        np.float32
    )

    force_sigma = np.asarray(
        base_pred["force_sigma"],
        np.float32
    )

    expert = np.asarray(
        base_pred["expert_prob"],
        np.float32
    )

    current_force_mag = (
        np.linalg.norm(
            P[:, 9:12],
            axis=-1
        )
        .astype(np.float32)
    )

    stance_force_sum = (
        P[:, 9:12]
        .sum(axis=0)
        .astype(np.float32)
    )

    derived = np.concatenate(
        [
            current_force_mag,                  # 4
            stance_force_sum,                   # 3

            np.asarray(
                [
                    np.linalg.norm(
                        base_force
                    ),

                    np.linalg.norm(
                        force_sigma
                    ),

                    float(
                        V9C1_HORIZON_S
                    ),
                ],
                np.float32
            ),                                  # 3
        ]
    )

    x = np.concatenate(
        [
            P.reshape(-1),                      # 48
            B,                                  # 12
            U,                                  # 6
            contact,                            # 4
            L.reshape(-1),                      # 12
            E.reshape(-1),                      # 12
            pos.reshape(-1),                    # 12
            vel.reshape(-1),                    # 12
            leg_oh,                             # 4
            base_tau,                           # 3
            base_force,                         # 3
            tau_sigma,                          # 3
            force_sigma,                        # 3
            expert,                             # 4
            derived,                            # 10
        ]
    ).astype(np.float32)

    return x


# ======================================================================
# 7. V9-A BELIEF FEATURE
#
# This is deliberately separate from dynamics so we can test whether
# terrain belief contributes beyond load sharing.
# ======================================================================

def v9c1_belief_feature(
    belief,
):

    token = np.asarray(
        belief["token"],
        np.float32
    )

    normal = np.asarray(
        belief["normal"],
        np.float32
    )

    summary = np.concatenate(
        [
            np.asarray(
                [
                    belief["z"],
                ],
                np.float32
            ),

            normal,

            np.asarray(
                [
                    belief["rough"],
                    belief["confidence"],

                    belief["effective_n"]
                    / 24.0,

                    min(
                        belief[
                            "weighted_age_s"
                        ],
                        6.0
                    )
                    / 6.0,

                    belief[
                        "weighted_distance_m"
                    ]
                    / 0.35,

                    belief[
                        "z_std_m"
                    ]
                    / 0.20,

                    belief[
                        "xy_std_m"
                    ]
                    / 0.35,

                    belief[
                        "token_dispersion"
                    ],

                    belief[
                        "normal_dispersion"
                    ],
                ],
                np.float32
            ),
        ]
    )

    return np.concatenate(
        [
            token,
            summary
        ]
    ).astype(np.float32)


# ======================================================================
# 8. BUILD STRICT TEMPORAL EVENT DATASET
#
# TRAIN = all-leg touchdowns well before first fixed test
# VAL   = later pre-test all-leg touchdowns
# TEST  = hind touchdowns AFTER first fixed test
#
# Thus evaluation targets never train the model.
# ======================================================================

Xd = []
Xt = []
Y = []

META = []

print()
print("Building event-centric C1 dataset...")


for ri, rec in enumerate(
    run_records
):

    data = rec[
        "data"
    ]

    key = (
        rec["terrain_type"],
        rec["environment"],
        rec["run"]
    )

    mem8 = v8_memory[
        key
    ]

    mem9 = v9_belief_memory[
        key
    ]

    first_tt = float(
        min(
            rec["ttests"]
        )
    )

    dt = float(
        np.median(
            np.diff(
                data["time"]
            )
        )
    )

    off = int(
        round(
            V9C1_HORIZON_S
            /
            dt
        )
    )

    val_end = (
        first_tt
        -
        V9C1_PRETEST_BUFFER_S
    )

    val_start = (
        val_end
        -
        V9C1_VAL_WINDOW_S
    )

    train_end = (
        val_start
        -
        V9C1_TRAINVAL_GAP_S
    )

    test_start = (
        first_tt
        +
        V9C1_TEST_BUFFER_S
    )

    terrain_counts = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    for leg in range(4):

        tds = v9c1_touchdowns(
            data["contact"][
                :,
                leg
            ]
        )

        for j in tds:

            idx = int(
                j - off
            )

            if idx < 0:
                continue

            target_t = float(
                data["time"][j]
            )

            query_t = float(
                data["time"][idx]
            )

            # ------------------------------------------------------
            # Strict split.
            # ------------------------------------------------------

            if target_t <= train_end:

                split = "train"

            elif (
                target_t >= val_start
                and
                target_t <= val_end
            ):

                split = "val"

            elif (
                target_t >= test_start
                and
                leg in HIND_LEGS
            ):

                split = "test"

            else:
                continue

            # ------------------------------------------------------
            # ORACLE future touchdown coordinate.
            #
            # Still an ablation, explicitly NOT deployment.
            # ------------------------------------------------------

            fw = v8_foot_world_proxy(
                rec,
                j
            )

            if fw is None:
                continue

            xyz = np.asarray(
                fw[leg],
                np.float32
            )

            mq8 = mem8.query(
                xyz[:2],
                query_t
            )

            if mq8 is None:
                continue

            belief = mem9.query(
                xyz,
                query_t
            )

            if belief is None:
                continue

            # ------------------------------------------------------
            # Frozen V8-C baseline.
            # ------------------------------------------------------

            base = v8c_predict_cell(
                rec,
                idx,
                leg,
                V9C1_HORIZON_S,
                mq8
            )

            base_force = np.asarray(
                base["force"],
                np.float32
            )

            gt_force = np.asarray(
                data["P"][
                    j,
                    leg,
                    9:12
                ],
                np.float32
            )

            base_mag = float(
                np.linalg.norm(
                    base_force
                )
            )

            gt_mag = float(
                np.linalg.norm(
                    gt_force
                )
            )

            # Target is only the residual contact severity.
            target_delta = (
                gt_mag
                -
                base_mag
            )

            xd = (
                v9c1_dynamics_feature(
                    rec,
                    idx,
                    leg,
                    base
                )
            )

            xt = (
                v9c1_belief_feature(
                    belief
                )
            )

            Xd.append(
                xd
            )

            Xt.append(
                xt
            )

            Y.append(
                float(
                    target_delta
                )
            )

            META.append(
                dict(
                    ri=int(
                        ri
                    ),

                    terrain_type=rec[
                        "terrain_type"
                    ],

                    environment=rec[
                        "environment"
                    ],

                    run=rec[
                        "run"
                    ],

                    split=split,

                    leg=int(
                        leg
                    ),

                    leg_name=LEG_NAMES[
                        leg
                    ],

                    query_index=int(
                        idx
                    ),

                    target_index=int(
                        j
                    ),

                    query_time=query_t,

                    target_time=target_t,

                    first_test_time=first_tt,

                    oracle_contact_location=True,

                    base_Fx=float(
                        base_force[0]
                    ),

                    base_Fy=float(
                        base_force[1]
                    ),

                    base_Fz=float(
                        base_force[2]
                    ),

                    gt_Fx=float(
                        gt_force[0]
                    ),

                    gt_Fy=float(
                        gt_force[1]
                    ),

                    gt_Fz=float(
                        gt_force[2]
                    ),

                    base_force_mag=base_mag,

                    gt_force_mag=gt_mag,

                    target_delta_mag=float(
                        target_delta
                    ),

                    memory_effective_n=float(
                        belief[
                            "effective_n"
                        ]
                    ),

                    memory_distance_m=float(
                        belief[
                            "weighted_distance_m"
                        ]
                    ),

                    memory_age_s=float(
                        belief[
                            "weighted_age_s"
                        ]
                    ),

                    memory_token_dispersion=float(
                        belief[
                            "token_dispersion"
                        ]
                    ),
                )
            )

            terrain_counts[
                split
            ] += 1

    print(
        rec["terrain_type"],
        terrain_counts
    )


Xd = np.asarray(
    Xd,
    np.float32
)

Xt = np.asarray(
    Xt,
    np.float32
)

Y = np.asarray(
    Y,
    np.float32
)

meta = pd.DataFrame(
    META
)


if (
    len(Xd) == 0
    or
    len(Xt) == 0
):

    raise RuntimeError(
        "V9-C1 dataset is empty."
    )


print()
print(
    "Dynamics dim:",
    Xd.shape[1]
)

print(
    "Belief dim:",
    Xt.shape[1]
)

print()
print(
    meta.groupby(
        [
            "split",
            "terrain_type"
        ]
    )
    .size()
)


# ======================================================================
# 9. SPLIT INDICES
# ======================================================================

train_idx = np.flatnonzero(
    meta.split.to_numpy()
    ==
    "train"
)

val_idx = np.flatnonzero(
    meta.split.to_numpy()
    ==
    "val"
)

test_idx = np.flatnonzero(
    meta.split.to_numpy()
    ==
    "test"
)


if len(train_idx) < 100:
    raise RuntimeError(
        "Too few C1 training touchdowns: "
        + str(
            len(train_idx)
        )
    )

if len(val_idx) < 20:
    raise RuntimeError(
        "Too few C1 validation touchdowns: "
        + str(
            len(val_idx)
        )
    )

if len(test_idx) < 100:
    raise RuntimeError(
        "Too few C1 held-out hind touchdowns: "
        + str(
            len(test_idx)
        )
    )


print()
print(
    "TRAIN:",
    len(train_idx)
)

print(
    "VAL  :",
    len(val_idx)
)

print(
    "TEST :",
    len(test_idx)
)


# ======================================================================
# 10. TRAIN-ONLY NORMALIZATION
# ======================================================================

XD_MEAN = Xd[
    train_idx
].mean(
    axis=0
).astype(np.float32)

XD_STD = np.maximum(
    Xd[
        train_idx
    ].std(
        axis=0
    ),
    1e-4
).astype(np.float32)


XT_MEAN = Xt[
    train_idx
].mean(
    axis=0
).astype(np.float32)

XT_STD = np.maximum(
    Xt[
        train_idx
    ].std(
        axis=0
    ),
    1e-4
).astype(np.float32)


Y_MEAN = float(
    Y[
        train_idx
    ].mean()
)

Y_STD = float(
    max(
        Y[
            train_idx
        ].std(),
        1.0
    )
)


print()
print(
    "Train target delta:",
    "mean=",
    Y_MEAN,
    "std=",
    Y_STD
)


# ======================================================================
# 11. DATASET
# ======================================================================

class V9C1Dataset(
    Dataset
):

    def __init__(
        self,
        ids
    ):

        self.ids = np.asarray(
            ids,
            np.int64
        )

    def __len__(
        self
    ):

        return len(
            self.ids
        )

    def __getitem__(
        self,
        k
    ):

        i = int(
            self.ids[k]
        )

        xd = (
            (
                Xd[i]
                -
                XD_MEAN
            )
            /
            XD_STD
        ).astype(np.float32)

        xt = (
            (
                Xt[i]
                -
                XT_MEAN
            )
            /
            XT_STD
        ).astype(np.float32)

        y = np.float32(
            (
                Y[i]
                -
                Y_MEAN
            )
            /
            Y_STD
        )

        return (
            torch.from_numpy(
                xd
            ),

            torch.from_numpy(
                xt
            ),

            torch.tensor(
                y,
                dtype=torch.float32
            ),
        )


# ======================================================================
# 12. MATCHED MODELS
#
# DYN and BELIEF have the SAME dynamics branch.
#
# Only BELIEF receives V9-A information.
# ======================================================================

class V9C1ResidualNet(
    nn.Module
):

    def __init__(
        self,
        dyn_dim,
        terrain_dim,
        use_terrain,
    ):

        super().__init__()

        self.use_terrain = bool(
            use_terrain
        )

        self.dyn = nn.Sequential(
            nn.LayerNorm(
                dyn_dim
            ),

            nn.Linear(
                dyn_dim,
                192
            ),

            nn.GELU(),

            nn.Dropout(
                0.10
            ),

            nn.Linear(
                192,
                96
            ),

            nn.GELU(),

            nn.LayerNorm(
                96
            ),
        )

        if self.use_terrain:

            self.terrain = nn.Sequential(
                nn.LayerNorm(
                    terrain_dim
                ),

                nn.Linear(
                    terrain_dim,
                    96
                ),

                nn.GELU(),

                nn.Dropout(
                    0.10
                ),

                nn.Linear(
                    96,
                    32
                ),

                nn.GELU(),

                nn.LayerNorm(
                    32
                ),
            )

            fuse_dim = 128

        else:

            self.terrain = None

            fuse_dim = 96

        self.head = nn.Sequential(
            nn.Linear(
                fuse_dim,
                96
            ),

            nn.GELU(),

            nn.Dropout(
                0.10
            ),

            nn.Linear(
                96,
                48
            ),

            nn.GELU(),

            # mean + log sigma
            nn.Linear(
                48,
                2
            ),
        )

    def forward(
        self,
        xd,
        xt,
    ):

        d = self.dyn(
            xd
        )

        if self.use_terrain:

            t = self.terrain(
                xt
            )

            z = torch.cat(
                [
                    d,
                    t
                ],
                dim=-1
            )

        else:

            z = d

        o = self.head(
            z
        )

        mu = o[
            :,
            0
        ]

        log_sigma = (
            o[
                :,
                1
            ]
            .clamp(
                -3.0,
                2.0
            )
        )

        return (
            mu,
            log_sigma
        )


# ======================================================================
# 13. LOSS
# ======================================================================

def v9c1_loss(
    mu,
    log_sigma,
    y,
):

    sigma = torch.exp(
        log_sigma
    )

    z = (
        y
        -
        mu
    ) / sigma

    nll = (
        0.5
        *
        z.pow(2)
        +
        log_sigma
    ).mean()

    robust = (
        F.smooth_l1_loss(
            mu,
            y,
            beta=0.35
        )
    )

    return (
        nll
        +
        0.20
        *
        robust
    )


# ======================================================================
# 14. TRAINING
# ======================================================================

def v9c1_train_model(
    use_terrain,
    label,
):

    v9c1_seed(
        V9C1_SEED
        +
        (
            1
            if use_terrain
            else 0
        )
    )

    model = V9C1ResidualNet(
        Xd.shape[1],
        Xt.shape[1],
        use_terrain=use_terrain,
    ).to(
        DEVICE
    )

    train_loader = DataLoader(
        V9C1Dataset(
            train_idx
        ),

        batch_size=V9C1_BATCH,

        shuffle=True,

        num_workers=0,
    )

    val_loader = DataLoader(
        V9C1Dataset(
            val_idx
        ),

        batch_size=V9C1_BATCH,

        shuffle=False,

        num_workers=0,
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=V9C1_LR,
        weight_decay=V9C1_WEIGHT_DECAY,
    )

    best = np.inf
    best_state = None
    best_epoch = None

    patience = 0

    history = []

    print()
    print(
        "=" * 90
    )

    print(
        "TRAIN",
        label
    )

    print(
        "=" * 90
    )

    for ep in range(
        1,
        V9C1_MAX_EPOCHS
        + 1
    ):

        model.train()

        train_losses = []

        for xd, xt, y in train_loader:

            xd = xd.to(
                DEVICE
            )

            xt = xt.to(
                DEVICE
            )

            y = y.to(
                DEVICE
            )

            opt.zero_grad(
                set_to_none=True
            )

            mu, ls = model(
                xd,
                xt
            )

            loss = v9c1_loss(
                mu,
                ls,
                y
            )

            if not torch.isfinite(
                loss
            ):

                raise RuntimeError(
                    "Nonfinite C1 loss."
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                2.0
            )

            opt.step()

            train_losses.append(
                float(
                    loss.detach().cpu()
                )
            )

        # ----------------------------------------------------------
        # Validation.
        # ----------------------------------------------------------

        model.eval()

        val_losses = []

        with torch.inference_mode():

            for xd, xt, y in val_loader:

                xd = xd.to(
                    DEVICE
                )

                xt = xt.to(
                    DEVICE
                )

                y = y.to(
                    DEVICE
                )

                mu, ls = model(
                    xd,
                    xt
                )

                vl = v9c1_loss(
                    mu,
                    ls,
                    y
                )

                val_losses.append(
                    float(
                        vl.cpu()
                    )
                )

        tr = float(
            np.mean(
                train_losses
            )
        )

        va = float(
            np.mean(
                val_losses
            )
        )

        history.append(
            dict(
                epoch=ep,
                train_loss=tr,
                val_loss=va,
            )
        )

        improved = (
            va
            <
            best
            -
            1e-4
        )

        if improved:

            best = va
            best_epoch = ep

            best_state = {
                k:
                v.detach()
                .cpu()
                .clone()

                for k, v
                in model.state_dict().items()
            }

            patience = 0

        else:

            patience += 1

        if (
            ep == 1
            or
            ep % 10 == 0
            or
            improved
        ):

            print(
                f"{label:10s} "
                f"epoch={ep:03d} "
                f"train={tr:.5f} "
                f"val={va:.5f} "
                f"best={best:.5f}"
            )

        if (
            patience
            >=
            V9C1_PATIENCE
        ):

            print(
                "early stop at",
                ep
            )

            break

    if best_state is None:

        raise RuntimeError(
            "C1 produced no valid checkpoint."
        )

    model.load_state_dict(
        best_state
    )

    model.to(
        DEVICE
    )

    model.eval()

    return (
        model,
        pd.DataFrame(
            history
        ),
        int(
            best_epoch
        ),
        float(
            best
        ),
    )


v9c1_dyn_model, v9c1_dyn_history, dyn_best_epoch, dyn_best_val = (
    v9c1_train_model(
        False,
        "C1-DYN"
    )
)


v9c1_belief_model, v9c1_belief_history, belief_best_epoch, belief_best_val = (
    v9c1_train_model(
        True,
        "C1-BELIEF"
    )
)


# ======================================================================
# 15. SIMPLE CALIBRATION BASELINES
#
# Fit on TRAIN only.
# ======================================================================

GLOBAL_DELTA = float(
    Y[
        train_idx
    ].mean()
)


LEG_DELTA = {}

for leg in range(4):

    ids = train_idx[
        meta.iloc[
            train_idx
        ]
        .leg
        .to_numpy()
        ==
        leg
    ]

    if len(ids):

        LEG_DELTA[
            leg
        ] = float(
            Y[
                ids
            ].mean()
        )

    else:

        LEG_DELTA[
            leg
        ] = GLOBAL_DELTA


print()
print(
    "GLOBAL train correction:",
    GLOBAL_DELTA,
    "N"
)

print(
    "LEG corrections:",
    LEG_DELTA
)


# ======================================================================
# 16. MODEL INFERENCE
# ======================================================================

@torch.inference_mode()
def v9c1_predict_delta(
    model,
    i,
):

    xd = (
        (
            Xd[i]
            -
            XD_MEAN
        )
        /
        XD_STD
    ).astype(np.float32)

    xt = (
        (
            Xt[i]
            -
            XT_MEAN
        )
        /
        XT_STD
    ).astype(np.float32)

    xd_t = torch.as_tensor(
        xd[
            None
        ],
        dtype=torch.float32,
        device=DEVICE
    )

    xt_t = torch.as_tensor(
        xt[
            None
        ],
        dtype=torch.float32,
        device=DEVICE
    )

    mu, ls = model(
        xd_t,
        xt_t
    )

    mu_n = float(
        mu[0].cpu()
    )

    sigma_n = float(
        torch.exp(
            ls[0]
        ).cpu()
    )

    delta = (
        mu_n
        *
        Y_STD
        +
        Y_MEAN
    )

    sigma = (
        sigma_n
        *
        Y_STD
    )

    delta = float(
        np.clip(
            delta,
            -V9C1_DELTA_CAP_N,
            V9C1_DELTA_CAP_N
        )
    )

    return (
        delta,
        float(
            sigma
        )
    )


# ======================================================================
# 17. STRICT HELD-OUT EVALUATION
# ======================================================================

eval_rows = []


for count, i in enumerate(
    test_idx,
    1
):

    r = meta.iloc[
        int(i)
    ]

    base = np.asarray(
        [
            r.base_Fx,
            r.base_Fy,
            r.base_Fz,
        ],
        np.float32
    )

    gt = np.asarray(
        [
            r.gt_Fx,
            r.gt_Fy,
            r.gt_Fz,
        ],
        np.float32
    )

    leg = int(
        r.leg
    )

    # --------------------------------------------------------------
    # Calibration baselines.
    # --------------------------------------------------------------

    global_force = (
        v9c1_apply_delta(
            base,
            GLOBAL_DELTA
        )
    )

    leg_force = (
        v9c1_apply_delta(
            base,
            LEG_DELTA[
                leg
            ]
        )
    )

    # --------------------------------------------------------------
    # Learned models.
    # --------------------------------------------------------------

    dyn_delta, dyn_sigma = (
        v9c1_predict_delta(
            v9c1_dyn_model,
            int(i)
        )
    )

    belief_delta, belief_sigma = (
        v9c1_predict_delta(
            v9c1_belief_model,
            int(i)
        )
    )

    dyn_force = (
        v9c1_apply_delta(
            base,
            dyn_delta
        )
    )

    belief_force = (
        v9c1_apply_delta(
            base,
            belief_delta
        )
    )

    # --------------------------------------------------------------
    # Metrics.
    # --------------------------------------------------------------

    def mae(x):

        return float(
            np.mean(
                np.abs(
                    x - gt
                )
            )
        )

    def ve(x):

        return float(
            np.linalg.norm(
                x - gt
            )
        )

    def me(x):

        return float(
            abs(
                np.linalg.norm(x)
                -
                np.linalg.norm(gt)
            )
        )

    eval_rows.append(
        dict(
            terrain_type=r.terrain_type,
            environment=r.environment,
            run=r.run,

            leg=int(
                leg
            ),

            leg_name=r.leg_name,

            query_time=float(
                r.query_time
            ),

            target_time=float(
                r.target_time
            ),

            oracle_contact_location=True,

            base_force_MAE_N=mae(
                base
            ),

            global_force_MAE_N=mae(
                global_force
            ),

            legcal_force_MAE_N=mae(
                leg_force
            ),

            dyn_force_MAE_N=mae(
                dyn_force
            ),

            belief_force_MAE_N=mae(
                belief_force
            ),

            base_vector_error_N=ve(
                base
            ),

            global_vector_error_N=ve(
                global_force
            ),

            legcal_vector_error_N=ve(
                leg_force
            ),

            dyn_vector_error_N=ve(
                dyn_force
            ),

            belief_vector_error_N=ve(
                belief_force
            ),

            base_magnitude_error_N=me(
                base
            ),

            global_magnitude_error_N=me(
                global_force
            ),

            legcal_magnitude_error_N=me(
                leg_force
            ),

            dyn_magnitude_error_N=me(
                dyn_force
            ),

            belief_magnitude_error_N=me(
                belief_force
            ),

            dyn_delta_N=float(
                dyn_delta
            ),

            belief_delta_N=float(
                belief_delta
            ),

            dyn_sigma_N=float(
                dyn_sigma
            ),

            belief_sigma_N=float(
                belief_sigma
            ),

            true_delta_N=float(
                r.target_delta_mag
            ),

            memory_effective_n=float(
                r.memory_effective_n
            ),

            memory_distance_m=float(
                r.memory_distance_m
            ),

            memory_age_s=float(
                r.memory_age_s
            ),

            memory_token_dispersion=float(
                r.memory_token_dispersion
            ),

            dyn_beats_base=bool(
                mae(
                    dyn_force
                )
                <
                mae(
                    base
                )
            ),

            belief_beats_base=bool(
                mae(
                    belief_force
                )
                <
                mae(
                    base
                )
            ),

            belief_beats_dyn=bool(
                mae(
                    belief_force
                )
                <
                mae(
                    dyn_force
                )
            ),
        )
    )

    if (
        count % 200
        == 0
    ):

        print(
            "eval",
            count,
            "/",
            len(
                test_idx
            )
        )


v9c1_eval = pd.DataFrame(
    eval_rows
)


# ======================================================================
# 18. SUMMARY
# ======================================================================

def v9c1_summary_for(
    df
):

    return dict(
        samples=int(
            len(df)
        ),

        base_force_MAE_N=float(
            df[
                "base_force_MAE_N"
            ].mean()
        ),

        global_force_MAE_N=float(
            df[
                "global_force_MAE_N"
            ].mean()
        ),

        legcal_force_MAE_N=float(
            df[
                "legcal_force_MAE_N"
            ].mean()
        ),

        dyn_force_MAE_N=float(
            df[
                "dyn_force_MAE_N"
            ].mean()
        ),

        belief_force_MAE_N=float(
            df[
                "belief_force_MAE_N"
            ].mean()
        ),

        base_vector_error_N=float(
            df[
                "base_vector_error_N"
            ].mean()
        ),

        dyn_vector_error_N=float(
            df[
                "dyn_vector_error_N"
            ].mean()
        ),

        belief_vector_error_N=float(
            df[
                "belief_vector_error_N"
            ].mean()
        ),

        base_magnitude_error_N=float(
            df[
                "base_magnitude_error_N"
            ].mean()
        ),

        dyn_magnitude_error_N=float(
            df[
                "dyn_magnitude_error_N"
            ].mean()
        ),

        belief_magnitude_error_N=float(
            df[
                "belief_magnitude_error_N"
            ].mean()
        ),

        dyn_win_rate=float(
            df[
                "dyn_beats_base"
            ].mean()
        ),

        belief_win_rate=float(
            df[
                "belief_beats_base"
            ].mean()
        ),

        belief_beats_dyn_rate=float(
            df[
                "belief_beats_dyn"
            ].mean()
        ),

        base_vector_p95_N=float(
            df[
                "base_vector_error_N"
            ].quantile(
                0.95
            )
        ),

        dyn_vector_p95_N=float(
            df[
                "dyn_vector_error_N"
            ].quantile(
                0.95
            )
        ),

        belief_vector_p95_N=float(
            df[
                "belief_vector_error_N"
            ].quantile(
                0.95
            )
        ),
    )


v9c1_summary = pd.DataFrame(
    [
        dict(
            subset=(
                "strict_post_first_test_hind_touchdowns"
            ),

            **v9c1_summary_for(
                v9c1_eval
            )
        )
    ]
)


# ======================================================================
# 19. TERRAIN / LEG BREAKDOWN
# ======================================================================

group_rows = []

for (
    terrain,
    leg_name
), g in v9c1_eval.groupby(
    [
        "terrain_type",
        "leg_name"
    ]
):

    group_rows.append(
        dict(
            terrain_type=terrain,
            leg_name=leg_name,

            **v9c1_summary_for(
                g
            )
        )
    )


v9c1_by_terrain = pd.DataFrame(
    group_rows
)


# ======================================================================
# 20. PAIRED BOOTSTRAP
#
# A - B > 0 means B has lower MAE.
# ======================================================================

def v9c1_bootstrap(
    a,
    b,
    reps=4000,
    seed=V9C1_SEED,
):

    d = (
        np.asarray(
            a,
            np.float64
        )
        -
        np.asarray(
            b,
            np.float64
        )
    )

    d = d[
        np.isfinite(d)
    ]

    rng = (
        np.random.default_rng(
            seed
        )
    )

    n = len(d)

    boots = np.empty(
        reps,
        np.float64
    )

    for r in range(
        reps
    ):

        ids = rng.integers(
            0,
            n,
            size=n
        )

        boots[r] = float(
            d[
                ids
            ].mean()
        )

    return dict(
        n=int(
            n
        ),

        mean_difference_N=float(
            d.mean()
        ),

        ci95_low_N=float(
            np.quantile(
                boots,
                0.025
            )
        ),

        ci95_high_N=float(
            np.quantile(
                boots,
                0.975
            )
        ),

        probability_positive=float(
            np.mean(
                boots > 0
            )
        ),
    )


v9c1_bootstrap = pd.DataFrame(
    [
        dict(
            comparison="BASE minus DYN",

            **v9c1_bootstrap(
                v9c1_eval[
                    "base_force_MAE_N"
                ],

                v9c1_eval[
                    "dyn_force_MAE_N"
                ],
            )
        ),

        dict(
            comparison="BASE minus BELIEF",

            **v9c1_bootstrap(
                v9c1_eval[
                    "base_force_MAE_N"
                ],

                v9c1_eval[
                    "belief_force_MAE_N"
                ],
            )
        ),

        dict(
            comparison="GLOBAL minus BELIEF",

            **v9c1_bootstrap(
                v9c1_eval[
                    "global_force_MAE_N"
                ],

                v9c1_eval[
                    "belief_force_MAE_N"
                ],
            )
        ),

        dict(
            comparison="LEG-CAL minus BELIEF",

            **v9c1_bootstrap(
                v9c1_eval[
                    "legcal_force_MAE_N"
                ],

                v9c1_eval[
                    "belief_force_MAE_N"
                ],
            )
        ),

        dict(
            comparison="DYN minus BELIEF",

            **v9c1_bootstrap(
                v9c1_eval[
                    "dyn_force_MAE_N"
                ],

                v9c1_eval[
                    "belief_force_MAE_N"
                ],
            )
        ),
    ]
)


# ======================================================================
# 21. SAVE
# ======================================================================

meta.to_csv(
    V9C1_ROOT /
    "V9C1_EVENT_DATASET_MANIFEST.csv",
    index=False
)

v9c1_eval.to_csv(
    V9C1_ROOT /
    "V9C1_STRICT_TEST_RAW.csv",
    index=False
)

v9c1_summary.to_csv(
    V9C1_ROOT /
    "V9C1_SUMMARY.csv",
    index=False
)

v9c1_by_terrain.to_csv(
    V9C1_ROOT /
    "V9C1_BY_TERRAIN_LEG.csv",
    index=False
)

v9c1_bootstrap.to_csv(
    V9C1_ROOT /
    "V9C1_BOOTSTRAP.csv",
    index=False
)

v9c1_dyn_history.to_csv(
    V9C1_ROOT /
    "V9C1_DYN_TRAIN_HISTORY.csv",
    index=False
)

v9c1_belief_history.to_csv(
    V9C1_ROOT /
    "V9C1_BELIEF_TRAIN_HISTORY.csv",
    index=False
)


torch.save(
    dict(
        state_dict=(
            v9c1_dyn_model
            .state_dict()
        ),

        XD_MEAN=XD_MEAN,
        XD_STD=XD_STD,

        XT_MEAN=XT_MEAN,
        XT_STD=XT_STD,

        Y_MEAN=Y_MEAN,
        Y_STD=Y_STD,

        use_terrain=False,
    ),

    V9C1_ROOT /
    "V9C1_DYN.pt"
)


torch.save(
    dict(
        state_dict=(
            v9c1_belief_model
            .state_dict()
        ),

        XD_MEAN=XD_MEAN,
        XD_STD=XD_STD,

        XT_MEAN=XT_MEAN,
        XT_STD=XT_STD,

        Y_MEAN=Y_MEAN,
        Y_STD=Y_STD,

        use_terrain=True,
    ),

    V9C1_ROOT /
    "V9C1_BELIEF.pt"
)


# ======================================================================
# 22. OUTPUT
# ======================================================================

print()
print("=" * 112)
print("V9-C1 STRICT HELD-OUT SUMMARY")
print("=" * 112)

display(
    v9c1_summary
)

print()
print(
    "BY TERRAIN / HIND LEG"
)

display(
    v9c1_by_terrain
)

print()
print(
    "PAIRED BOOTSTRAP — POSITIVE MEANS SECOND METHOD WINS"
)

display(
    v9c1_bootstrap
)


s = (
    v9c1_summary
    .iloc[0]
)

print()
print("=" * 112)
print("V9-C1 DECISION")
print("=" * 112)

print(
    "V8-C BASE       :",
    float(
        s[
            "base_force_MAE_N"
        ]
    )
)

print(
    "GLOBAL CAL      :",
    float(
        s[
            "global_force_MAE_N"
        ]
    )
)

print(
    "LEG CAL         :",
    float(
        s[
            "legcal_force_MAE_N"
        ]
    )
)

print(
    "C1-DYN          :",
    float(
        s[
            "dyn_force_MAE_N"
        ]
    )
)

print(
    "C1-BELIEF       :",
    float(
        s[
            "belief_force_MAE_N"
        ]
    )
)

print()

print(
    "V8-C p95 vector:",
    float(
        s[
            "base_vector_p95_N"
        ]
    )
)

print(
    "DYN p95 vector :",
    float(
        s[
            "dyn_vector_p95_N"
        ]
    )
)

print(
    "BELIEF p95 vec :",
    float(
        s[
            "belief_vector_p95_N"
        ]
    )
)


dyn_vs_belief = (
    v9c1_bootstrap[
        v9c1_bootstrap[
            "comparison"
        ]
        ==
        "DYN minus BELIEF"
    ]
    .iloc[0]
)

base_vs_dyn = (
    v9c1_bootstrap[
        v9c1_bootstrap[
            "comparison"
        ]
        ==
        "BASE minus DYN"
    ]
    .iloc[0]
)

base_vs_belief = (
    v9c1_bootstrap[
        v9c1_bootstrap[
            "comparison"
        ]
        ==
        "BASE minus BELIEF"
    ]
    .iloc[0]
)


print()
print("Scientific verdict:")


if (
    base_vs_belief.ci95_low_N > 0
    and
    dyn_vs_belief.ci95_low_N > 0
):

    print(
        "✅ STRONG RESULT:"
    )

    print(
        "Joint dynamics improves V8-C AND "
        "V9 terrain belief adds significant "
        "additional held-out predictive value."
    )

    print(
        "Proceed to V9-C2 structured "
        "contact-response posterior."
    )


elif (
    base_vs_dyn.ci95_low_N > 0
    and
    dyn_vs_belief.ci95_low_N <= 0
):

    print(
        "🟡 DYNAMICS RESULT:"
    )

    print(
        "Four-leg/body load-sharing significantly "
        "improves touchdown prediction, but the "
        "V9 terrain belief is not yet adding "
        "independent value."
    )

    print(
        "Do NOT make ContactMap force-prediction "
        "claims yet."
    )


elif (
    base_vs_belief.ci95_low_N > 0
):

    print(
        "🟡 BELIEF MODEL IMPROVES V8-C, "
        "but its advantage over dynamics-only "
        "is not statistically separated."
    )


else:

    print(
        "❌ NEITHER learned residual is a clean "
        "improvement over frozen V8-C."
    )

    print(
        "Then global/leg calibration is the "
        "stronger causal baseline and we redesign "
        "the response target rather than adding capacity."
    )


print()
print(
    "IMPORTANT:"
)

print(
    "C1 still uses ORACLE touchdown location."
)

print(
    "Touchdown-location prediction remains a "
    "separate V9-B problem."
)

print()
print(
    "Saved:",
    V9C1_ROOT
)

print(
    "✅ V9-C1 COMPLETE"
)

# %% notebook cell 36
# ======================================================================
# V9-C1.1 — CAP-SENSITIVITY + BLOCK-BOOTSTRAP AUDIT
#
# WHY
# ----
# C1 used a fixed ±140 N residual cap.
#
# But:
#     train target mean ≈ +203 N
#     global correction ≈ +203 N
#     hind-leg corrections ≈ +193 / +227 N
#
# Therefore GLOBAL and LEG-CAL were both clipped to +140 N,
# making that comparison artificially identical.
#
# This audit:
#
#   1. obtains RAW, UNCLIPPED C1-DYN / C1-BELIEF outputs
#   2. selects correction caps on VALIDATION ONLY
#   3. freezes those caps
#   4. evaluates once on strict held-out TEST
#   5. runs 5-second TIME-BLOCK bootstrap
#   6. checks whether BELIEF's small gain is broad or tail-driven
#
# NO RETRAINING.
# NO TEST-TUNED HYPERPARAMETERS.
# ======================================================================

from pathlib import Path
import numpy as np
import pandas as pd
import torch

from scipy.stats import trim_mean


print("=" * 112)
print("V9-C1.1 — CAP-SENSITIVITY + BLOCK-BOOTSTRAP AUDIT")
print("=" * 112)


# ======================================================================
# 0. PREFLIGHT
# ======================================================================

_req = [
    "Xd",
    "Xt",
    "meta",
    "val_idx",
    "test_idx",

    "XD_MEAN",
    "XD_STD",
    "XT_MEAN",
    "XT_STD",
    "Y_MEAN",
    "Y_STD",

    "GLOBAL_DELTA",
    "LEG_DELTA",

    "v9c1_dyn_model",
    "v9c1_belief_model",

    "V9C1_ROOT",
    "DEVICE",
]

_missing = [
    x for x in _req
    if x not in globals()
]

if _missing:
    raise RuntimeError(
        "V9-C1.1 missing objects: "
        + repr(_missing)
    )


V9C11_ROOT = (
    V9C1_ROOT /
    "C1_1_cap_and_block_audit"
)

V9C11_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ======================================================================
# 1. CAP CANDIDATES
#
# Selected ONLY from validation.
# ======================================================================

V9C11_CAPS = [
    80.0,
    100.0,
    140.0,
    180.0,
    220.0,
    260.0,
    320.0,
    400.0,
    600.0,
    np.inf,
]

V9C11_BLOCK_SEC = 5.0

V9C11_BOOT_REPS = 5000

V9C11_SEED = 19781


# ======================================================================
# 2. RAW UNCLIPPED NETWORK OUTPUT
# ======================================================================

@torch.inference_mode()
def v9c11_raw_delta(
    model,
    i,
):

    i = int(i)

    xd = (
        (
            Xd[i]
            -
            XD_MEAN
        )
        /
        XD_STD
    ).astype(np.float32)

    xt = (
        (
            Xt[i]
            -
            XT_MEAN
        )
        /
        XT_STD
    ).astype(np.float32)

    xd_t = torch.as_tensor(
        xd[None],
        dtype=torch.float32,
        device=DEVICE
    )

    xt_t = torch.as_tensor(
        xt[None],
        dtype=torch.float32,
        device=DEVICE
    )

    mu, log_sigma = model(
        xd_t,
        xt_t
    )

    delta = (
        float(
            mu[0].cpu()
        )
        *
        float(
            Y_STD
        )
        +
        float(
            Y_MEAN
        )
    )

    sigma = (
        float(
            torch.exp(
                log_sigma[0]
            ).cpu()
        )
        *
        float(
            Y_STD
        )
    )

    return (
        float(delta),
        float(sigma),
    )


# ======================================================================
# 3. CACHE RAW PREDICTIONS FOR VAL + TEST
#
# Important:
# model forward is executed once.
# All cap sweeps afterward are deterministic arithmetic.
# ======================================================================

audit_rows = []


all_ids = np.concatenate(
    [
        np.asarray(
            val_idx,
            np.int64
        ),

        np.asarray(
            test_idx,
            np.int64
        )
    ]
)


for count, i in enumerate(
    all_ids,
    1
):

    i = int(i)

    r = meta.iloc[i]

    base = np.asarray(
        [
            r.base_Fx,
            r.base_Fy,
            r.base_Fz,
        ],
        np.float32
    )

    gt = np.asarray(
        [
            r.gt_Fx,
            r.gt_Fy,
            r.gt_Fz,
        ],
        np.float32
    )

    leg = int(
        r.leg
    )

    dyn_delta, dyn_sigma = (
        v9c11_raw_delta(
            v9c1_dyn_model,
            i
        )
    )

    belief_delta, belief_sigma = (
        v9c11_raw_delta(
            v9c1_belief_model,
            i
        )
    )

    audit_rows.append(
        dict(
            dataset_index=i,

            split=str(
                r.split
            ),

            terrain_type=str(
                r.terrain_type
            ),

            environment=str(
                r.environment
            ),

            run=str(
                r.run
            ),

            leg=leg,

            leg_name=str(
                r.leg_name
            ),

            query_time=float(
                r.query_time
            ),

            target_time=float(
                r.target_time
            ),

            base_Fx=float(
                base[0]
            ),

            base_Fy=float(
                base[1]
            ),

            base_Fz=float(
                base[2]
            ),

            gt_Fx=float(
                gt[0]
            ),

            gt_Fy=float(
                gt[1]
            ),

            gt_Fz=float(
                gt[2]
            ),

            global_raw_delta_N=float(
                GLOBAL_DELTA
            ),

            leg_raw_delta_N=float(
                LEG_DELTA[
                    leg
                ]
            ),

            dyn_raw_delta_N=float(
                dyn_delta
            ),

            belief_raw_delta_N=float(
                belief_delta
            ),

            dyn_sigma_N=float(
                dyn_sigma
            ),

            belief_sigma_N=float(
                belief_sigma
            ),
        )
    )

    if (
        count % 250
        == 0
    ):

        print(
            "cached",
            count,
            "/",
            len(
                all_ids
            )
        )


v9c11_cache = pd.DataFrame(
    audit_rows
)


# ======================================================================
# 4. DIRECTION-PRESERVING FORCE CORRECTION
# ======================================================================

def v9c11_apply_batch(
    base,
    raw_delta,
    cap,
):

    base = np.asarray(
        base,
        np.float64
    )

    raw_delta = np.asarray(
        raw_delta,
        np.float64
    )

    if np.isfinite(
        cap
    ):

        delta = np.clip(
            raw_delta,
            -float(cap),
            float(cap)
        )

    else:

        delta = (
            raw_delta.copy()
        )

    mag = np.linalg.norm(
        base,
        axis=1
    )

    new_mag = np.maximum(
        0.0,
        mag
        +
        delta
    )

    ratio = np.ones_like(
        mag
    )

    good = (
        mag > 1e-8
    )

    ratio[
        good
    ] = (
        new_mag[
            good
        ]
        /
        mag[
            good
        ]
    )

    pred = (
        base
        *
        ratio[
            :,
            None
        ]
    )

    return (
        pred.astype(
            np.float32
        ),
        delta.astype(
            np.float32
        )
    )


# ======================================================================
# 5. SCORING
# ======================================================================

def v9c11_arrays(
    df
):

    base = df[
        [
            "base_Fx",
            "base_Fy",
            "base_Fz",
        ]
    ].to_numpy(
        np.float64
    )

    gt = df[
        [
            "gt_Fx",
            "gt_Fy",
            "gt_Fz",
        ]
    ].to_numpy(
        np.float64
    )

    return (
        base,
        gt
    )


METHOD_COLUMNS = {
    "GLOBAL": "global_raw_delta_N",
    "LEG_CAL": "leg_raw_delta_N",
    "DYN": "dyn_raw_delta_N",
    "BELIEF": "belief_raw_delta_N",
}


def v9c11_score(
    df,
    method,
    cap,
):

    base, gt = (
        v9c11_arrays(
            df
        )
    )

    raw_delta = df[
        METHOD_COLUMNS[
            method
        ]
    ].to_numpy(
        np.float64
    )

    pred, applied = (
        v9c11_apply_batch(
            base,
            raw_delta,
            cap
        )
    )

    component_mae = (
        np.abs(
            pred
            -
            gt
        )
        .mean(
            axis=1
        )
    )

    vector_error = (
        np.linalg.norm(
            pred
            -
            gt,
            axis=1
        )
    )

    magnitude_error = (
        np.abs(
            np.linalg.norm(
                pred,
                axis=1
            )
            -
            np.linalg.norm(
                gt,
                axis=1
            )
        )
    )

    if np.isfinite(
        cap
    ):

        cap_hit = (
            np.abs(
                raw_delta
            )
            >=
            float(cap)
            -
            1e-8
        )

    else:

        cap_hit = np.zeros(
            len(raw_delta),
            dtype=bool
        )

    return dict(
        MAE=float(
            component_mae.mean()
        ),

        vector_error=float(
            vector_error.mean()
        ),

        vector_p95=float(
            np.quantile(
                vector_error,
                0.95
            )
        ),

        magnitude_error=float(
            magnitude_error.mean()
        ),

        cap_hit_rate=float(
            cap_hit.mean()
        ),

        component_mae_values=(
            component_mae
        ),

        vector_values=(
            vector_error
        ),

        magnitude_values=(
            magnitude_error
        ),

        applied_delta=(
            applied
        ),
    )


# ======================================================================
# 6. VALIDATION-ONLY CAP SELECTION
# ======================================================================

val_df = (
    v9c11_cache[
        v9c11_cache.split
        ==
        "val"
    ]
    .copy()
)


test_df = (
    v9c11_cache[
        v9c11_cache.split
        ==
        "test"
    ]
    .copy()
)


print()
print(
    "VAL samples :",
    len(
        val_df
    )
)

print(
    "TEST samples:",
    len(
        test_df
    )
)


sweep_rows = []


for method in METHOD_COLUMNS:

    for cap in V9C11_CAPS:

        s = v9c11_score(
            val_df,
            method,
            cap
        )

        sweep_rows.append(
            dict(
                method=method,

                cap_N=(
                    float(cap)
                    if np.isfinite(
                        cap
                    )
                    else np.inf
                ),

                validation_MAE_N=s[
                    "MAE"
                ],

                validation_vector_error_N=s[
                    "vector_error"
                ],

                validation_vector_p95_N=s[
                    "vector_p95"
                ],

                cap_hit_rate=s[
                    "cap_hit_rate"
                ],
            )
        )


v9c11_cap_sweep = pd.DataFrame(
    sweep_rows
)


selected_caps = {}


for method in METHOD_COLUMNS:

    q = (
        v9c11_cap_sweep[
            v9c11_cap_sweep.method
            ==
            method
        ]
        .sort_values(
            [
                "validation_MAE_N",
                "cap_N"
            ]
        )
    )

    selected_caps[
        method
    ] = float(
        q.iloc[0].cap_N
    )


print()
print(
    "VALIDATION-SELECTED CAPS:"
)

for k, v in selected_caps.items():

    print(
        f"  {k:8s}:",
        (
            "unbounded"
            if not np.isfinite(v)
            else f"{v:.1f} N"
        )
    )


# ======================================================================
# 7. STRICT TEST USING FROZEN VALIDATION CHOICES
# ======================================================================

base_test, gt_test = (
    v9c11_arrays(
        test_df
    )
)


base_mae_values = (
    np.abs(
        base_test
        -
        gt_test
    )
    .mean(
        axis=1
    )
)


base_vector_values = (
    np.linalg.norm(
        base_test
        -
        gt_test,
        axis=1
    )
)


result_arrays = {
    "BASE": dict(
        mae=base_mae_values,

        vector=(
            base_vector_values
        ),

        cap_hit_rate=0.0,
    )
}


test_summary_rows = [
    dict(
        method="BASE",

        selected_cap_N=np.nan,

        MAE_N=float(
            base_mae_values.mean()
        ),

        vector_error_N=float(
            base_vector_values.mean()
        ),

        vector_p95_N=float(
            np.quantile(
                base_vector_values,
                0.95
            )
        ),

        cap_hit_rate=0.0,
    )
]


for method in METHOD_COLUMNS:

    cap = selected_caps[
        method
    ]

    s = v9c11_score(
        test_df,
        method,
        cap
    )

    result_arrays[
        method
    ] = dict(
        mae=s[
            "component_mae_values"
        ],

        vector=s[
            "vector_values"
        ],

        cap_hit_rate=s[
            "cap_hit_rate"
        ],
    )

    test_summary_rows.append(
        dict(
            method=method,

            selected_cap_N=(
                cap
            ),

            MAE_N=s[
                "MAE"
            ],

            vector_error_N=s[
                "vector_error"
            ],

            vector_p95_N=s[
                "vector_p95"
            ],

            magnitude_error_N=s[
                "magnitude_error"
            ],

            cap_hit_rate=s[
                "cap_hit_rate"
            ],
        )
    )


v9c11_test_summary = pd.DataFrame(
    test_summary_rows
)


# ======================================================================
# 8. PER-EVENT TEST TABLE
# ======================================================================

v9c11_test = (
    test_df[
        [
            "terrain_type",
            "environment",
            "run",
            "leg",
            "leg_name",
            "query_time",
            "target_time",
        ]
    ]
    .reset_index(
        drop=True
    )
)


for method, arr in result_arrays.items():

    v9c11_test[
        f"{method}_MAE_N"
    ] = arr[
        "mae"
    ]

    v9c11_test[
        f"{method}_vector_error_N"
    ] = arr[
        "vector"
    ]


# ======================================================================
# 9. TERRAIN / LEG BREAKDOWN
# ======================================================================

terrain_rows = []


for (
    terrain,
    leg
), g in v9c11_test.groupby(
    [
        "terrain_type",
        "leg_name"
    ]
):

    terrain_rows.append(
        dict(
            terrain_type=terrain,

            leg_name=leg,

            n=int(
                len(g)
            ),

            BASE_MAE=float(
                g.BASE_MAE_N.mean()
            ),

            GLOBAL_MAE=float(
                g.GLOBAL_MAE_N.mean()
            ),

            LEG_CAL_MAE=float(
                g.LEG_CAL_MAE_N.mean()
            ),

            DYN_MAE=float(
                g.DYN_MAE_N.mean()
            ),

            BELIEF_MAE=float(
                g.BELIEF_MAE_N.mean()
            ),

            DYN_minus_BELIEF=float(
                (
                    g.DYN_MAE_N
                    -
                    g.BELIEF_MAE_N
                ).mean()
            ),
        )
    )


v9c11_by_terrain = pd.DataFrame(
    terrain_rows
)


# ======================================================================
# 10. ROBUST PAIRED EFFECT DIAGNOSTICS
# ======================================================================

paired = (
    v9c11_test[
        "DYN_MAE_N"
    ].to_numpy(
        np.float64
    )
    -
    v9c11_test[
        "BELIEF_MAE_N"
    ].to_numpy(
        np.float64
    )
)


v9c11_robust_effect = pd.DataFrame(
    [
        dict(
            comparison="DYN minus BELIEF",

            n=int(
                len(paired)
            ),

            mean_difference_N=float(
                paired.mean()
            ),

            median_difference_N=float(
                np.median(
                    paired
                )
            ),

            trimmed10_mean_difference_N=float(
                trim_mean(
                    paired,
                    0.10
                )
            ),

            belief_win_rate=float(
                np.mean(
                    paired > 0
                )
            ),

            belief_loss_rate=float(
                np.mean(
                    paired < 0
                )
            ),

            p05_difference_N=float(
                np.quantile(
                    paired,
                    0.05
                )
            ),

            p95_difference_N=float(
                np.quantile(
                    paired,
                    0.95
                )
            ),
        )
    ]
)


# ======================================================================
# 11. 5-SECOND TIME-BLOCK BOOTSTRAP
#
# Events close in time are correlated.
# We therefore resample temporal blocks rather than individual contacts.
#
# Positive difference:
#     second model wins.
# ======================================================================

def v9c11_block_bootstrap(
    df,
    first_col,
    second_col,
    block_sec=V9C11_BLOCK_SEC,
    reps=V9C11_BOOT_REPS,
    seed=V9C11_SEED,
):

    q = df.copy()

    q[
        "block_index"
    ] = np.floor(
        q[
            "target_time"
        ].to_numpy(
            np.float64
        )
        /
        float(
            block_sec
        )
    ).astype(
        np.int64
    )

    q[
        "block_id"
    ] = (
        q[
            "terrain_type"
        ].astype(str)
        +
        "|"
        +
        q[
            "run"
        ].astype(str)
        +
        "|"
        +
        q[
            "block_index"
        ].astype(str)
    )

    q[
        "_paired_d"
    ] = (
        q[
            first_col
        ]
        -
        q[
            second_col
        ]
    )

    blocks = (
        q.groupby(
            "block_id"
        )
        ._paired_d
        .agg(
            [
                "sum",
                "count"
            ]
        )
        .reset_index()
    )

    sums = blocks[
        "sum"
    ].to_numpy(
        np.float64
    )

    counts = blocks[
        "count"
    ].to_numpy(
        np.float64
    )

    rng = (
        np.random.default_rng(
            seed
        )
    )

    nb = len(
        blocks
    )

    boots = np.empty(
        reps,
        np.float64
    )

    for r in range(
        reps
    ):

        ids = rng.integers(
            0,
            nb,
            size=nb
        )

        boots[r] = (
            sums[
                ids
            ].sum()
            /
            counts[
                ids
            ].sum()
        )

    point = float(
        q[
            "_paired_d"
        ].mean()
    )

    return dict(
        events=int(
            len(q)
        ),

        blocks=int(
            nb
        ),

        block_sec=float(
            block_sec
        ),

        mean_difference_N=point,

        ci95_low_N=float(
            np.quantile(
                boots,
                0.025
            )
        ),

        ci95_high_N=float(
            np.quantile(
                boots,
                0.975
            )
        ),

        probability_positive=float(
            np.mean(
                boots > 0
            )
        ),
    )


v9c11_block_bootstrap = pd.DataFrame(
    [
        dict(
            comparison="BASE minus DYN",

            **v9c11_block_bootstrap(
                v9c11_test,
                "BASE_MAE_N",
                "DYN_MAE_N",
            )
        ),

        dict(
            comparison="BASE minus BELIEF",

            **v9c11_block_bootstrap(
                v9c11_test,
                "BASE_MAE_N",
                "BELIEF_MAE_N",
            )
        ),

        dict(
            comparison="GLOBAL minus DYN",

            **v9c11_block_bootstrap(
                v9c11_test,
                "GLOBAL_MAE_N",
                "DYN_MAE_N",
            )
        ),

        dict(
            comparison="GLOBAL minus BELIEF",

            **v9c11_block_bootstrap(
                v9c11_test,
                "GLOBAL_MAE_N",
                "BELIEF_MAE_N",
            )
        ),

        dict(
            comparison="DYN minus BELIEF",

            **v9c11_block_bootstrap(
                v9c11_test,
                "DYN_MAE_N",
                "BELIEF_MAE_N",
            )
        ),
    ]
)


# ======================================================================
# 12. GROUP CONSISTENCY
# ======================================================================

groups_positive = int(
    (
        v9c11_by_terrain[
            "DYN_minus_BELIEF"
        ]
        >
        0
    ).sum()
)

groups_total = int(
    len(
        v9c11_by_terrain
    )
)


# ======================================================================
# 13. SAVE
# ======================================================================

v9c11_cache.to_csv(
    V9C11_ROOT /
    "V9C11_RAW_UNCLIPPED_CACHE.csv",
    index=False
)

v9c11_cap_sweep.to_csv(
    V9C11_ROOT /
    "V9C11_VALIDATION_CAP_SWEEP.csv",
    index=False
)

v9c11_test.to_csv(
    V9C11_ROOT /
    "V9C11_STRICT_TEST_EVENT_ERRORS.csv",
    index=False
)

v9c11_test_summary.to_csv(
    V9C11_ROOT /
    "V9C11_STRICT_TEST_SUMMARY.csv",
    index=False
)

v9c11_by_terrain.to_csv(
    V9C11_ROOT /
    "V9C11_BY_TERRAIN_LEG.csv",
    index=False
)

v9c11_robust_effect.to_csv(
    V9C11_ROOT /
    "V9C11_ROBUST_EFFECT.csv",
    index=False
)

v9c11_block_bootstrap.to_csv(
    V9C11_ROOT /
    "V9C11_TIME_BLOCK_BOOTSTRAP.csv",
    index=False
)


# ======================================================================
# 14. DISPLAY
# ======================================================================

print()
print("=" * 112)
print("VALIDATION CAP SWEEP")
print("=" * 112)

display(
    v9c11_cap_sweep
)


print()
print("=" * 112)
print("STRICT TEST — VALIDATION-SELECTED CAPS")
print("=" * 112)

display(
    v9c11_test_summary
)


print()
print("=" * 112)
print("BY TERRAIN / HIND LEG")
print("=" * 112)

display(
    v9c11_by_terrain
)


print()
print("=" * 112)
print("ROBUST DYN vs BELIEF EFFECT")
print("=" * 112)

display(
    v9c11_robust_effect
)


print()
print("=" * 112)
print("5-SECOND TIME-BLOCK BOOTSTRAP")
print("=" * 112)

display(
    v9c11_block_bootstrap
)


# ======================================================================
# 15. DECISION
# ======================================================================

bb = (
    v9c11_block_bootstrap[
        v9c11_block_bootstrap.comparison
        ==
        "DYN minus BELIEF"
    ]
    .iloc[0]
)


rob = (
    v9c11_robust_effect
    .iloc[0]
)


dyn_test = float(
    v9c11_test_summary[
        v9c11_test_summary.method
        ==
        "DYN"
    ]
    .MAE_N
    .iloc[0]
)


belief_test = float(
    v9c11_test_summary[
        v9c11_test_summary.method
        ==
        "BELIEF"
    ]
    .MAE_N
    .iloc[0]
)


belief_gain = (
    dyn_test
    -
    belief_test
)


belief_gain_pct = (
    100.0
    *
    belief_gain
    /
    (
        dyn_test
        +
        1e-9
    )
)


print()
print("=" * 112)
print("V9-C1.1 RESEARCH DECISION")
print("=" * 112)

print(
    "Validation-selected caps:",
    selected_caps
)

print()

print(
    "DYN test MAE:",
    dyn_test
)

print(
    "BELIEF test MAE:",
    belief_test
)

print(
    "BELIEF incremental gain:",
    belief_gain,
    "N"
)

print(
    "BELIEF relative gain:",
    belief_gain_pct,
    "%"
)

print()

print(
    "Median paired gain:",
    float(
        rob[
            "median_difference_N"
        ]
    )
)

print(
    "10% trimmed paired gain:",
    float(
        rob[
            "trimmed10_mean_difference_N"
        ]
    )
)

print(
    "Belief individual-event win rate:",
    float(
        rob[
            "belief_win_rate"
        ]
    )
)

print(
    "Terrain/leg groups where BELIEF wins:",
    groups_positive,
    "/",
    groups_total
)

print()

print(
    "Time-block bootstrap CI:",
    (
        float(
            bb[
                "ci95_low_N"
            ]
        ),
        float(
            bb[
                "ci95_high_N"
            ]
        )
    )
)


# ----------------------------------------------------------------------
# Engineering/research gate.
#
# Not claimed as universal statistical law.
# ----------------------------------------------------------------------

if (
    bb[
        "ci95_low_N"
    ]
    > 0
    and
    belief_gain
    >= 0.5
    and
    groups_positive
    >= 5
):

    V9C11_BELIEF_STATUS = (
        "ROBUST_INDEPENDENT_VALUE"
    )

    print()
    print(
        "✅ BELIEF SURVIVES THE STRICT AUDIT."
    )

    print(
        "The terrain belief adds practically meaningful "
        "information beyond four-leg dynamics."
    )

    print(
        "Proceed to structured V9-C2."
    )


elif (
    bb[
        "ci95_low_N"
    ]
    > 0
):

    V9C11_BELIEF_STATUS = (
        "SMALL_DETECTABLE_VALUE"
    )

    print()
    print(
        "🟡 BELIEF EFFECT IS DETECTABLE BUT SMALL."
    )

    print(
        "Keep terrain belief as an auxiliary feature, "
        "but do NOT make it the main force-prediction claim."
    )

    print(
        "The main C1 result is four-leg/body load sharing."
    )


else:

    V9C11_BELIEF_STATUS = (
        "NOT_ROBUST"
    )

    print()
    print(
        "❌ BELIEF DOES NOT SURVIVE CORRELATED-EVENT AUDITING."
    )

    print(
        "Use C1-DYN as the force predictor."
    )

    print(
        "ContactMap can still be used for terrain memory, "
        "touchdown localization, uncertainty, and planning."
    )


print()
print(
    "Status:",
    V9C11_BELIEF_STATUS
)

print()

print(
    "Saved:",
    V9C11_ROOT
)

print(
    "✅ V9-C1.1 COMPLETE"
)

# %% notebook cell 37
# ======================================================================
# V9-C2 — STRUCTURED DYN-ONLY CONTACT-SEVERITY POSTERIOR
#
# BASE:
#     frozen V8-C force direction + magnitude
#
# CONTEXT:
#     exact C1 four-leg/body dynamics features
#
# TARGET:
#     log(1 + |F_true|) - log(1 + |F_V8C|)
#
# OUTPUT:
#     q10 / q50 / q90 posterior over log-magnitude correction
#
# VALIDATION:
#     early stopping + split-conformal interval calibration
#
# TEST:
#     same strict post-first-test hind touchdowns as C1/C1.1
#
# NO V9 TERRAIN BELIEF IN FORCE PREDICTION.
# NO TEST-TUNED CAP.
# ======================================================================

from pathlib import Path
import copy
import random
import math

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


print("=" * 112)
print("V9-C2 — STRUCTURED DYN-ONLY CONTACT-SEVERITY POSTERIOR")
print("=" * 112)


# ======================================================================
# 0. PREFLIGHT
# ======================================================================

_req = [
    "Xd",
    "meta",

    "train_idx",
    "val_idx",
    "test_idx",

    "XD_MEAN",
    "XD_STD",

    "V9_C_ROOT",
    "DEVICE",

    "v9c11_test",
]

_missing = [
    x for x in _req
    if x not in globals()
]

if _missing:
    raise RuntimeError(
        "V9-C2 missing objects: "
        + repr(_missing)
    )


V9C2_ROOT = (
    V9_C_ROOT /
    "C2_structured_contact_posterior"
)

V9C2_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ======================================================================
# 1. CONFIG
# ======================================================================

V9C2_BATCH = 128

V9C2_MAX_EPOCHS = 220
V9C2_PATIENCE = 28

V9C2_LR = 8e-4

V9C2_WEIGHT_DECAY = 2e-4

V9C2_SEED = 12781

# Quantile posterior.
V9C2_QLO = 0.10
V9C2_QMID = 0.50
V9C2_QHI = 0.90

# Split-conformal target coverage = 80%.
V9C2_ALPHA = 0.20

# Numerical sanity bound ONLY.
# This should almost never activate.
V9C2_MAX_FORCE_MAG_N = 5000.0

V9C2_BLOCK_SEC = 5.0

V9C2_BOOT_REPS = 5000


# ======================================================================
# 2. DETERMINISM
# ======================================================================

def v9c2_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


v9c2_seed(
    V9C2_SEED
)


# ======================================================================
# 3. STRUCTURED TARGET
#
# y =
#
#   log(1 + true force magnitude)
#   -
#   log(1 + frozen V8-C force magnitude)
#
# This is substantially better conditioned than a raw +200 N residual.
# ======================================================================

base_mag_all = (
    meta[
        "base_force_mag"
    ]
    .to_numpy(
        np.float32
    )
)

gt_mag_all = (
    meta[
        "gt_force_mag"
    ]
    .to_numpy(
        np.float32
    )
)


Y2 = (
    np.log1p(
        gt_mag_all
    )
    -
    np.log1p(
        base_mag_all
    )
).astype(
    np.float32
)


Y2_MEAN = float(
    Y2[
        train_idx
    ].mean()
)

Y2_STD = float(
    max(
        Y2[
            train_idx
        ].std(),
        1e-3
    )
)


print(
    "Structured train target:"
)

print(
    "  mean =",
    Y2_MEAN
)

print(
    "  std  =",
    Y2_STD
)

print(
    "  min  =",
    float(
        Y2[
            train_idx
        ].min()
    )
)

print(
    "  max  =",
    float(
        Y2[
            train_idx
        ].max()
    )
)


# ======================================================================
# 4. DATASET
# ======================================================================

class V9C2Dataset(
    Dataset
):

    def __init__(
        self,
        ids
    ):

        self.ids = np.asarray(
            ids,
            np.int64
        )

    def __len__(
        self
    ):

        return len(
            self.ids
        )

    def __getitem__(
        self,
        k
    ):

        i = int(
            self.ids[
                k
            ]
        )

        x = (
            (
                Xd[i]
                -
                XD_MEAN
            )
            /
            XD_STD
        ).astype(
            np.float32
        )

        y = np.float32(
            (
                Y2[i]
                -
                Y2_MEAN
            )
            /
            Y2_STD
        )

        return (
            torch.from_numpy(
                x
            ),

            torch.tensor(
                y,
                dtype=torch.float32
            ),
        )


# ======================================================================
# 5. QUANTILE MODEL
#
# q10 <= q50 <= q90 is guaranteed by construction.
# ======================================================================

class V9C2QuantileNet(
    nn.Module
):

    def __init__(
        self,
        din
    ):

        super().__init__()

        self.trunk = nn.Sequential(

            nn.LayerNorm(
                din
            ),

            nn.Linear(
                din,
                192
            ),

            nn.GELU(),

            nn.Dropout(
                0.10
            ),

            nn.Linear(
                192,
                128
            ),

            nn.GELU(),

            nn.Dropout(
                0.08
            ),

            nn.Linear(
                128,
                96
            ),

            nn.GELU(),

            nn.LayerNorm(
                96
            ),
        )

        self.head = nn.Sequential(

            nn.Linear(
                96,
                64
            ),

            nn.GELU(),

            nn.Linear(
                64,
                3
            ),
        )

    def forward(
        self,
        x
    ):

        z = self.trunk(
            x
        )

        raw = self.head(
            z
        )

        center = raw[
            :,
            0
        ]

        lower_width = (
            F.softplus(
                raw[
                    :,
                    1
                ]
            )
        )

        upper_width = (
            F.softplus(
                raw[
                    :,
                    2
                ]
            )
        )

        q50 = center

        q10 = (
            center
            -
            lower_width
        )

        q90 = (
            center
            +
            upper_width
        )

        return (
            q10,
            q50,
            q90,
        )


# ======================================================================
# 6. PINBALL LOSS
# ======================================================================

def v9c2_pinball(
    pred,
    target,
    tau,
):

    e = (
        target
        -
        pred
    )

    return torch.maximum(
        tau
        *
        e,

        (
            tau - 1.0
        )
        *
        e
    ).mean()


def v9c2_loss(
    q10,
    q50,
    q90,
    y,
):

    l10 = v9c2_pinball(
        q10,
        y,
        V9C2_QLO
    )

    l50 = v9c2_pinball(
        q50,
        y,
        V9C2_QMID
    )

    l90 = v9c2_pinball(
        q90,
        y,
        V9C2_QHI
    )

    robust_center = (
        F.smooth_l1_loss(
            q50,
            y,
            beta=0.35
        )
    )

    return (
        l10
        +
        1.5
        *
        l50
        +
        l90
        +
        0.15
        *
        robust_center
    )


# ======================================================================
# 7. LOADERS
# ======================================================================

train_loader = DataLoader(

    V9C2Dataset(
        train_idx
    ),

    batch_size=V9C2_BATCH,

    shuffle=True,

    num_workers=0,
)


val_loader = DataLoader(

    V9C2Dataset(
        val_idx
    ),

    batch_size=V9C2_BATCH,

    shuffle=False,

    num_workers=0,
)


# ======================================================================
# 8. TRAIN
# ======================================================================

v9c2_model = (
    V9C2QuantileNet(
        Xd.shape[
            1
        ]
    )
    .to(
        DEVICE
    )
)


opt = torch.optim.AdamW(

    v9c2_model.parameters(),

    lr=V9C2_LR,

    weight_decay=V9C2_WEIGHT_DECAY,
)


best_val = np.inf

best_epoch = None

best_state = None

patience = 0

history = []


print()
print(
    "Training C2 structured posterior..."
)


for ep in range(
    1,
    V9C2_MAX_EPOCHS
    + 1
):

    # --------------------------------------------------------------
    # Train
    # --------------------------------------------------------------

    v9c2_model.train()

    tl = []

    for x, y in train_loader:

        x = x.to(
            DEVICE
        )

        y = y.to(
            DEVICE
        )

        opt.zero_grad(
            set_to_none=True
        )

        q10, q50, q90 = (
            v9c2_model(
                x
            )
        )

        loss = v9c2_loss(
            q10,
            q50,
            q90,
            y
        )

        if not torch.isfinite(
            loss
        ):

            raise RuntimeError(
                "Nonfinite V9-C2 loss."
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            v9c2_model.parameters(),
            2.0
        )

        opt.step()

        tl.append(
            float(
                loss.detach().cpu()
            )
        )

    # --------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------

    v9c2_model.eval()

    vl = []

    with torch.inference_mode():

        for x, y in val_loader:

            x = x.to(
                DEVICE
            )

            y = y.to(
                DEVICE
            )

            q10, q50, q90 = (
                v9c2_model(
                    x
                )
            )

            loss = v9c2_loss(
                q10,
                q50,
                q90,
                y
            )

            vl.append(
                float(
                    loss.cpu()
                )
            )

    train_loss = float(
        np.mean(
            tl
        )
    )

    val_loss = float(
        np.mean(
            vl
        )
    )

    history.append(
        dict(
            epoch=ep,

            train_loss=train_loss,

            val_loss=val_loss,
        )
    )

    improved = (
        val_loss
        <
        best_val
        -
        1e-4
    )

    if improved:

        best_val = (
            val_loss
        )

        best_epoch = (
            ep
        )

        best_state = {

            k:
            v.detach()
            .cpu()
            .clone()

            for k, v
            in v9c2_model
            .state_dict()
            .items()
        }

        patience = 0

    else:

        patience += 1

    if (
        ep == 1
        or
        ep % 10 == 0
        or
        improved
    ):

        print(
            f"epoch={ep:03d} "
            f"train={train_loss:.5f} "
            f"val={val_loss:.5f} "
            f"best={best_val:.5f}"
        )

    if (
        patience
        >=
        V9C2_PATIENCE
    ):

        print(
            "early stop at",
            ep
        )

        break


if best_state is None:

    raise RuntimeError(
        "V9-C2 did not produce "
        "a valid checkpoint."
    )


v9c2_model.load_state_dict(
    best_state
)

v9c2_model.to(
    DEVICE
)

v9c2_model.eval()


v9c2_history = pd.DataFrame(
    history
)


print()
print(
    "Best epoch:",
    best_epoch
)

print(
    "Best validation loss:",
    best_val
)


# ======================================================================
# 9. INFERENCE
# ======================================================================

@torch.inference_mode()
def v9c2_predict_ids(
    ids
):

    ids = np.asarray(
        ids,
        np.int64
    )

    out10 = []
    out50 = []
    out90 = []

    bs = 256

    for start in range(
        0,
        len(ids),
        bs
    ):

        ii = ids[
            start:start + bs
        ]

        x = (
            (
                Xd[ii]
                -
                XD_MEAN
            )
            /
            XD_STD
        ).astype(
            np.float32
        )

        xt = torch.as_tensor(
            x,
            dtype=torch.float32,
            device=DEVICE
        )

        q10, q50, q90 = (
            v9c2_model(
                xt
            )
        )

        for arr, dest in [
            (
                q10,
                out10
            ),
            (
                q50,
                out50
            ),
            (
                q90,
                out90
            ),
        ]:

            q = (
                arr
                .cpu()
                .numpy()
                .astype(
                    np.float64
                )
            )

            # Undo target normalization.
            q = (
                q
                *
                Y2_STD
                +
                Y2_MEAN
            )

            dest.append(
                q
            )

    return (
        np.concatenate(
            out10
        ),

        np.concatenate(
            out50
        ),

        np.concatenate(
            out90
        ),
    )


# ======================================================================
# 10. STRUCTURED MAGNITUDE RECONSTRUCTION
# ======================================================================

def v9c2_reconstruct_mag(
    base_mag,
    residual,
):

    base_mag = np.asarray(
        base_mag,
        np.float64
    )

    residual = np.asarray(
        residual,
        np.float64
    )

    logmag = (
        np.log1p(
            base_mag
        )
        +
        residual
    )

    # Only a broad numerical-sanity ceiling.
    max_log = math.log1p(
        V9C2_MAX_FORCE_MAG_N
    )

    clipped = (
        logmag
        >
        max_log
    )

    logmag = np.minimum(
        logmag,
        max_log
    )

    pred = np.maximum(
        0.0,
        np.expm1(
            logmag
        )
    )

    return (
        pred.astype(
            np.float32
        ),

        clipped,
    )


# ======================================================================
# 11. VALIDATION PREDICTIONS
# ======================================================================

v10, v50, v90 = (
    v9c2_predict_ids(
        val_idx
    )
)


vbase = base_mag_all[
    val_idx
]


vgt = gt_mag_all[
    val_idx
]


vpred10, _ = (
    v9c2_reconstruct_mag(
        vbase,
        v10
    )
)


vpred50, _ = (
    v9c2_reconstruct_mag(
        vbase,
        v50
    )
)


vpred90, _ = (
    v9c2_reconstruct_mag(
        vbase,
        v90
    )
)


# ======================================================================
# 12. SPLIT-CONFORMAL CALIBRATION
#
# Calibration uses VALIDATION ONLY.
#
# Nonconformity:
#
#   score =
#       max(lower - truth,
#           truth - upper,
#           0)
#
# Then expand interval by q_hat.
# ======================================================================

val_score = np.maximum(

    np.maximum(
        vpred10
        -
        vgt,

        vgt
        -
        vpred90
    ),

    0.0
)


ncal = len(
    val_score
)


level = min(

    1.0,

    math.ceil(
        (
            ncal + 1
        )
        *
        (
            1.0
            -
            V9C2_ALPHA
        )
    )
    /
    ncal
)


try:

    V9C2_QHAT = float(
        np.quantile(
            val_score,
            level,
            method="higher"
        )
    )

except TypeError:

    V9C2_QHAT = float(
        np.quantile(
            val_score,
            level,
            interpolation="higher"
        )
    )


print()
print(
    "Conformal calibration:"
)

print(
    "  target coverage =",
    1.0
    -
    V9C2_ALPHA
)

print(
    "  validation n    =",
    ncal
)

print(
    "  quantile level  =",
    level
)

print(
    "  q_hat           =",
    V9C2_QHAT,
    "N"
)


# ======================================================================
# 13. STRICT TEST PREDICTIONS
# ======================================================================

t10, t50, t90 = (
    v9c2_predict_ids(
        test_idx
    )
)


tbase_mag = (
    base_mag_all[
        test_idx
    ]
)


tgt_mag = (
    gt_mag_all[
        test_idx
    ]
)


tp10, clip10 = (
    v9c2_reconstruct_mag(
        tbase_mag,
        t10
    )
)


tp50, clip50 = (
    v9c2_reconstruct_mag(
        tbase_mag,
        t50
    )
)


tp90, clip90 = (
    v9c2_reconstruct_mag(
        tbase_mag,
        t90
    )
)


# conformal interval
conf_lo = np.maximum(
    0.0,
    tp10
    -
    V9C2_QHAT
)


conf_hi = (
    tp90
    +
    V9C2_QHAT
)


coverage = (
    (
        tgt_mag
        >=
        conf_lo
    )
    &
    (
        tgt_mag
        <=
        conf_hi
    )
)


# ======================================================================
# 14. RECONSTRUCT FORCE VECTOR
#
# V8-C retains direction.
# C2 changes magnitude only.
# ======================================================================

test_meta = (
    meta.iloc[
        test_idx
    ]
    .reset_index(
        drop=True
    )
)


base_force = test_meta[
    [
        "base_Fx",
        "base_Fy",
        "base_Fz",
    ]
].to_numpy(
    np.float64
)


gt_force = test_meta[
    [
        "gt_Fx",
        "gt_Fy",
        "gt_Fz",
    ]
].to_numpy(
    np.float64
)


base_norm = np.linalg.norm(
    base_force,
    axis=1
)


direction = np.zeros_like(
    base_force
)


good = (
    base_norm
    >
    1e-8
)


direction[
    good
] = (
    base_force[
        good
    ]
    /
    base_norm[
        good,
        None
    ]
)


c2_force = (
    direction
    *
    tp50[
        :,
        None
    ]
)


# ======================================================================
# 15. C2 ERRORS
# ======================================================================

c2_mae = (
    np.abs(
        c2_force
        -
        gt_force
    )
    .mean(
        axis=1
    )
)


c2_vector = np.linalg.norm(
    c2_force
    -
    gt_force,
    axis=1
)


c2_mag_error = np.abs(
    tp50
    -
    tgt_mag
)


# ======================================================================
# 16. ALIGN WITH VALIDATION-SELECTED C1-DYN AUDIT
# ======================================================================

c2_result = pd.DataFrame(
    dict(
        terrain_type=(
            test_meta[
                "terrain_type"
            ].to_numpy()
        ),

        environment=(
            test_meta[
                "environment"
            ].to_numpy()
        ),

        run=(
            test_meta[
                "run"
            ].to_numpy()
        ),

        leg=(
            test_meta[
                "leg"
            ].to_numpy(
                int
            )
        ),

        leg_name=(
            test_meta[
                "leg_name"
            ].to_numpy()
        ),

        query_time=(
            test_meta[
                "query_time"
            ].to_numpy(
                float
            )
        ),

        target_time=(
            test_meta[
                "target_time"
            ].to_numpy(
                float
            )
        ),

        C2_MAE_N=c2_mae,

        C2_vector_error_N=(
            c2_vector
        ),

        C2_magnitude_error_N=(
            c2_mag_error
        ),

        C2_q10_mag_N=tp10,

        C2_q50_mag_N=tp50,

        C2_q90_mag_N=tp90,

        C2_conformal_lo_N=(
            conf_lo
        ),

        C2_conformal_hi_N=(
            conf_hi
        ),

        C2_interval_covered=(
            coverage
        ),

        true_force_mag_N=(
            tgt_mag
        ),
    )
)


# Stable float join.
c2_result[
    "_time_key"
] = np.round(
    c2_result[
        "target_time"
    ],
    6
)


c1_compare = (
    v9c11_test
    .copy()
)


c1_compare[
    "_time_key"
] = np.round(
    c1_compare[
        "target_time"
    ],
    6
)


keep = [
    "terrain_type",
    "run",
    "leg",
    "_time_key",

    "DYN_MAE_N",

    "DYN_vector_error_N",
]


c2_result = (
    c2_result
    .merge(
        c1_compare[
            keep
        ],

        on=[
            "terrain_type",
            "run",
            "leg",
            "_time_key",
        ],

        how="left",

        validate="one_to_one",
    )
)


if (
    c2_result[
        "DYN_MAE_N"
    ].isna().any()
):

    raise RuntimeError(
        "Failed to align C2 test "
        "with C1.1 DYN baseline."
    )


# ======================================================================
# 17. SUMMARY
# ======================================================================

V9C2_C1_DYN_MAE = float(
    c2_result[
        "DYN_MAE_N"
    ].mean()
)


V9C2_C2_MAE = float(
    c2_result[
        "C2_MAE_N"
    ].mean()
)


V9C2_C1_DYN_VEC = float(
    c2_result[
        "DYN_vector_error_N"
    ].mean()
)


V9C2_C2_VEC = float(
    c2_result[
        "C2_vector_error_N"
    ].mean()
)


V9C2_C1_P95 = float(
    c2_result[
        "DYN_vector_error_N"
    ].quantile(
        0.95
    )
)


V9C2_C2_P95 = float(
    c2_result[
        "C2_vector_error_N"
    ].quantile(
        0.95
    )
)


V9C2_COVERAGE = float(
    c2_result[
        "C2_interval_covered"
    ].mean()
)


V9C2_INTERVAL_WIDTH = float(
    (
        c2_result[
            "C2_conformal_hi_N"
        ]
        -
        c2_result[
            "C2_conformal_lo_N"
        ]
    ).mean()
)


v9c2_summary = pd.DataFrame(
    [
        dict(
            samples=int(
                len(
                    c2_result
                )
            ),

            C1_DYN_MAE_N=(
                V9C2_C1_DYN_MAE
            ),

            C2_MAE_N=(
                V9C2_C2_MAE
            ),

            improvement_N=(
                V9C2_C1_DYN_MAE
                -
                V9C2_C2_MAE
            ),

            improvement_percent=(
                100.0
                *
                (
                    V9C2_C1_DYN_MAE
                    -
                    V9C2_C2_MAE
                )
                /
                (
                    V9C2_C1_DYN_MAE
                    +
                    1e-9
                )
            ),

            C1_DYN_vector_error_N=(
                V9C2_C1_DYN_VEC
            ),

            C2_vector_error_N=(
                V9C2_C2_VEC
            ),

            C1_DYN_vector_p95_N=(
                V9C2_C1_P95
            ),

            C2_vector_p95_N=(
                V9C2_C2_P95
            ),

            conformal_target_coverage=(
                1.0
                -
                V9C2_ALPHA
            ),

            conformal_test_coverage=(
                V9C2_COVERAGE
            ),

            mean_interval_width_N=(
                V9C2_INTERVAL_WIDTH
            ),

            numerical_force_ceiling_hit_rate=float(
                (
                    clip10
                    |
                    clip50
                    |
                    clip90
                ).mean()
            ),
        )
    ]
)


# ======================================================================
# 18. TERRAIN / LEG
# ======================================================================

terrain_rows = []


for (
    terrain,
    leg
), g in c2_result.groupby(
    [
        "terrain_type",
        "leg_name"
    ]
):

    terrain_rows.append(
        dict(
            terrain_type=terrain,

            leg_name=leg,

            n=int(
                len(g)
            ),

            C1_DYN_MAE_N=float(
                g[
                    "DYN_MAE_N"
                ].mean()
            ),

            C2_MAE_N=float(
                g[
                    "C2_MAE_N"
                ].mean()
            ),

            C1_minus_C2_N=float(
                (
                    g[
                        "DYN_MAE_N"
                    ]
                    -
                    g[
                        "C2_MAE_N"
                    ]
                ).mean()
            ),

            C2_interval_coverage=float(
                g[
                    "C2_interval_covered"
                ].mean()
            ),
        )
    )


v9c2_by_terrain = pd.DataFrame(
    terrain_rows
)


# ======================================================================
# 19. TIME-BLOCK BOOTSTRAP
#
# Positive C1 - C2 difference = C2 is better.
# ======================================================================

def v9c2_block_bootstrap(
    df,
    reps=V9C2_BOOT_REPS,
    seed=V9C2_SEED,
):

    q = df.copy()

    q[
        "block_index"
    ] = np.floor(
        q[
            "target_time"
        ]
        /
        V9C2_BLOCK_SEC
    ).astype(
        np.int64
    )

    q[
        "block_id"
    ] = (
        q[
            "terrain_type"
        ].astype(str)
        +
        "|"
        +
        q[
            "run"
        ].astype(str)
        +
        "|"
        +
        q[
            "block_index"
        ].astype(str)
    )

    q[
        "_d"
    ] = (
        q[
            "DYN_MAE_N"
        ]
        -
        q[
            "C2_MAE_N"
        ]
    )

    b = (
        q.groupby(
            "block_id"
        )
        ._d
        .agg(
            [
                "sum",
                "count"
            ]
        )
        .reset_index()
    )

    sums = b[
        "sum"
    ].to_numpy(
        np.float64
    )

    counts = b[
        "count"
    ].to_numpy(
        np.float64
    )

    rng = np.random.default_rng(
        seed
    )

    nb = len(
        b
    )

    boots = np.empty(
        reps,
        np.float64
    )

    for r in range(
        reps
    ):

        ids = rng.integers(
            0,
            nb,
            size=nb
        )

        boots[r] = (
            sums[
                ids
            ].sum()
            /
            counts[
                ids
            ].sum()
        )

    return dict(
        events=int(
            len(q)
        ),

        blocks=int(
            nb
        ),

        mean_difference_N=float(
            q[
                "_d"
            ].mean()
        ),

        median_difference_N=float(
            q[
                "_d"
            ].median()
        ),

        ci95_low_N=float(
            np.quantile(
                boots,
                0.025
            )
        ),

        ci95_high_N=float(
            np.quantile(
                boots,
                0.975
            )
        ),

        probability_positive=float(
            np.mean(
                boots > 0
            )
        ),
    )


v9c2_bootstrap = pd.DataFrame(
    [
        dict(
            comparison=(
                "C1-DYN minus C2"
            ),

            **v9c2_block_bootstrap(
                c2_result
            )
        )
    ]
)


# ======================================================================
# 20. SAVE
# ======================================================================

v9c2_history.to_csv(
    V9C2_ROOT /
    "V9C2_TRAIN_HISTORY.csv",
    index=False
)


c2_result.to_csv(
    V9C2_ROOT /
    "V9C2_STRICT_TEST_RAW.csv",
    index=False
)


v9c2_summary.to_csv(
    V9C2_ROOT /
    "V9C2_SUMMARY.csv",
    index=False
)


v9c2_by_terrain.to_csv(
    V9C2_ROOT /
    "V9C2_BY_TERRAIN_LEG.csv",
    index=False
)


v9c2_bootstrap.to_csv(
    V9C2_ROOT /
    "V9C2_TIME_BLOCK_BOOTSTRAP.csv",
    index=False
)


torch.save(
    dict(
        state_dict=(
            v9c2_model
            .state_dict()
        ),

        XD_MEAN=XD_MEAN,

        XD_STD=XD_STD,

        Y2_MEAN=Y2_MEAN,

        Y2_STD=Y2_STD,

        qhat=V9C2_QHAT,

        alpha=V9C2_ALPHA,

        best_epoch=best_epoch,

        best_val=best_val,
    ),

    V9C2_ROOT /
    "V9C2_STRUCTURED_POSTERIOR.pt"
)


# ======================================================================
# 21. OUTPUT
# ======================================================================

print()
print("=" * 112)
print("V9-C2 STRICT SUMMARY")
print("=" * 112)

display(
    v9c2_summary
)


print()
print(
    "BY TERRAIN / HIND LEG"
)

display(
    v9c2_by_terrain
)


print()
print(
    "5-SECOND BLOCK BOOTSTRAP"
)

display(
    v9c2_bootstrap
)


b = (
    v9c2_bootstrap
    .iloc[0]
)


groups_better = int(
    (
        v9c2_by_terrain[
            "C1_minus_C2_N"
        ]
        >
        0
    ).sum()
)


print()
print("=" * 112)
print("V9-C2 DECISION")
print("=" * 112)


print(
    "C1-DYN MAE:",
    V9C2_C1_DYN_MAE
)


print(
    "C2 MAE:",
    V9C2_C2_MAE
)


print(
    "Improvement:",
    V9C2_C1_DYN_MAE
    -
    V9C2_C2_MAE,
    "N"
)


print(
    "C1-DYN p95:",
    V9C2_C1_P95
)


print(
    "C2 p95:",
    V9C2_C2_P95
)


print()
print(
    "Conformal target coverage:",
    1.0
    -
    V9C2_ALPHA
)


print(
    "Observed test coverage:",
    V9C2_COVERAGE
)


print(
    "Mean interval width:",
    V9C2_INTERVAL_WIDTH,
    "N"
)


print()
print(
    "Terrain/leg groups where C2 wins:",
    groups_better,
    "/",
    len(
        v9c2_by_terrain
    )
)


print(
    "Block-bootstrap CI:",
    (
        float(
            b[
                "ci95_low_N"
            ]
        ),

        float(
            b[
                "ci95_high_N"
            ]
        )
    )
)


print()


if (
    b[
        "ci95_low_N"
    ]
    >
    0
    and
    V9C2_C2_MAE
    <
    V9C2_C1_DYN_MAE
):

    V9C2_STATUS = (
        "STRUCTURED_POSTERIOR_WINS"
    )

    print(
        "✅ V9-C2 WINS."
    )

    print(
        "Use the structured log-magnitude "
        "posterior as the final contact-response model."
    )


elif (
    V9C2_C2_MAE
    <=
    V9C2_C1_DYN_MAE
):

    V9C2_STATUS = (
        "SMALL_UNCERTAIN_GAIN"
    )

    print(
        "🟡 C2 is numerically better but "
        "the correlated-event CI crosses zero."
    )

    print(
        "Retain C1-DYN as the accuracy baseline "
        "and C2 primarily for uncertainty."
    )


else:

    V9C2_STATUS = (
        "C1_DYN_REMAINS_BEST"
    )

    print(
        "🟡 C1-DYN remains the better point predictor."
    )

    print(
        "Keep C2 only if its calibrated uncertainty "
        "is useful for the planner."
    )


print()
print(
    "Status:",
    V9C2_STATUS
)


print()
print(
    "IMPORTANT:"
)

print(
    "C2 is still an ORACLE-touchdown-location experiment."
)

print(
    "The next major problem after this is V9-B: "
    "predict the touchdown location without using future foot position."
)


print()
print(
    "Saved:",
    V9C2_ROOT
)


print(
    "✅ V9-C2 COMPLETE"
)
