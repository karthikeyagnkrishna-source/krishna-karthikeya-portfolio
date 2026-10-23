# Source inspection export of the adjacent notebook.
# Notebook execution order and dependencies still apply.
# IPython magic, shell lines and prose-only conclusion cells are comments.

# %% notebook cell 1
# ======================================================================
# 0. INSTALL / CLONE EXTERNAL BACKBONES — PYTHON 3.13 COLAB SAFE
#
# IMPORTANT:
# Current Colab uses Python 3.13 / Torch 2.11 / CUDA 12.8.
# Utonia's official environment is Python 3.10 / Torch 2.5 / CUDA 12.4.
# spconv / torch-scatter do NOT have reliable Python-3.13 wheels for the
# exact environment Utonia expects.
#
# Therefore:
#   - main V7 notebook stays on Colab's Python 3.13 / Torch 2.11
#   - Utonia runs in an ISOLATED Python-3.10 worker environment
#   - geometry features are exchanged through .npy cache files
#
# This keeps the FULL Utonia/PTv3 branch. Nothing is removed.
# ======================================================================

import time
NOTEBOOK_WALL_T0=time.perf_counter()

import os, sys, subprocess, importlib.util, json, textwrap, shutil, platform
from pathlib import Path

# Runtime-portable working root. On the SSH server, if V7_PERSIST_ROOT is
# supplied, even external environments/checkpoints live on that persistent disk.
if os.environ.get('V7_WORK_ROOT'):
    WORK_ROOT=Path(os.environ['V7_WORK_ROOT']).expanduser()
elif os.environ.get('V7_PERSIST_ROOT'):
    WORK_ROOT=Path(os.environ['V7_PERSIST_ROOT']).expanduser()/'.work'
elif Path('/content').exists():
    WORK_ROOT=Path('/content')
elif Path('/workspace').exists():
    WORK_ROOT=Path('/workspace')
else:
    WORK_ROOT=Path.home()/'.v7_work'
WORK_ROOT.mkdir(parents=True,exist_ok=True)
print('WORK_ROOT:',WORK_ROOT)

def run(cmd, **kwargs):
    print(" ".join(map(str, cmd)))
    subprocess.check_call(list(map(str, cmd)), **kwargs)

def pip_install(*args):
    run([sys.executable, "-m", "pip", "install", "-q", *args])

# ----------------------------------------------------------------------
# Main-runtime packages
# ----------------------------------------------------------------------
pip_install(
    "timm>=1.0.20", "huggingface_hub>=0.33", "safetensors", "gdown",
    "rosbags", "scipy", "scikit-learn", "pandas", "matplotlib", "Pillow",
    "opencv-python-headless", "einops", "addict", "psutil", "natsort",
    "tqdm", "uv"
)
pip_install("git+https://github.com/castacks/tartanairpy.git")

import torch

print("Python:", sys.version)
print("Torch :", torch.__version__)
print("CUDA  :", torch.version.cuda)

if not torch.cuda.is_available():
    raise RuntimeError(
        "V7 requires a CUDA GPU runtime. "
        "Colab: Runtime -> Change runtime type -> GPU."
    )

# ----------------------------------------------------------------------
# WAFT / xformers in MAIN Colab environment
# ----------------------------------------------------------------------
try:
    import xformers  # noqa
    print("xformers:", getattr(xformers, "__version__", "available"))
except Exception:
    print("Installing xformers without replacing Colab torch...")
    run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "xformers"])
    try:
        import xformers  # noqa
        print("xformers:", getattr(xformers, "__version__", "available"))
    except Exception as e:
        raise RuntimeError(
            "WAFT requires xformers and the current Colab build could not import it."
        ) from e

# ----------------------------------------------------------------------
# External source/checkpoint directory
# ----------------------------------------------------------------------
EXT = WORK_ROOT / "v7_external"
EXT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# ISOLATED OFFICIAL-LIKE UTONIA ENVIRONMENT
#
# Utonia's official repo currently specifies roughly:
# Python 3.10, PyTorch 2.5.0, CUDA 12.4, spconv-cu124,
# torch-scatter from the PyG torch-2.5/cu124 wheel index.
#
# We deliberately do NOT attempt spconv-cu128 in Python 3.13.
# ----------------------------------------------------------------------
UV = shutil.which("uv")
if UV is None:
    raise RuntimeError("uv was installed but its executable was not found.")

UTONIA_ENV = EXT / "utonia_py310"
UTONIA_PY = UTONIA_ENV / "bin" / "python"

if not UTONIA_PY.exists():
    run([UV, "python", "install", "3.10"])
    run([UV, "venv", str(UTONIA_ENV), "--python", "3.10", "--seed"])

def uv_pip(*args):
    run([UV, "pip", "install", "--python", str(UTONIA_PY), *args])

# Install only if the environment is not already healthy.
uton_test = (
    "import torch, spconv.pytorch, torch_scatter, utonia; "
    "print(torch.__version__, torch.version.cuda)"
)
healthy = False
try:
    subprocess.check_call(
        [str(UTONIA_PY), "-c", uton_test],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    healthy = True
except Exception:
    healthy = False

if not healthy:
    print("Building isolated Utonia Python-3.10 environment...")

    # Exact stable stack used by the official Utonia environment.
    uv_pip(
        "torch==2.5.0",
        "torchvision==0.20.0",
        "torchaudio==2.5.0",
        "--index-url", "https://download.pytorch.org/whl/cu124",
    )

    uv_pip(
        "numpy==1.26.4", "ninja", "scipy", "addict", "einops",
        "psutil", "natsort", "opencv-python-headless",
        "huggingface_hub", "timm",
    )

    uv_pip("spconv-cu124")

    uv_pip(
        "torch-scatter",
        "--find-links", "https://data.pyg.org/whl/torch-2.5.0+cu124.html",
    )

    # FlashAttention is OPTIONAL for Utonia. The worker explicitly sets
    # enable_flash=False, so we avoid another compiled dependency.
    #
    # IMPORTANT FIX: installing Utonia directly with `uv pip install git+...`
    # fails in the exact Python-3.10 isolated environment on current runtimes.
    # Utonia itself is pure Python, so clone the repository and expose it
    # through a .pth file instead of building/installing it as a package.
    UTONIA_REPO = EXT / "Utonia"
    if not UTONIA_REPO.exists():
        run([
            "git", "clone", "--depth", "1",
            "https://github.com/Pointcept/Utonia.git",
            str(UTONIA_REPO),
        ])
    else:
        print("Reusing Utonia clone:", UTONIA_REPO)

    site_packages = subprocess.check_output(
        [
            str(UTONIA_PY), "-c",
            "import site; print(site.getsitepackages()[0])"
        ],
        text=True,
    ).strip()
    (Path(site_packages) / "utonia_repo.pth").write_text(
        str(UTONIA_REPO.resolve()) + "\n"
    )

    # Optional dependencies used by some Utonia import paths.
    test = subprocess.run(
        [
            str(UTONIA_PY), "-c",
            "import torch, spconv.pytorch, torch_scatter, utonia"
        ],
        text=True, capture_output=True,
    )
    if test.returncode != 0:
        uv_pip("camtools", "trimesh")

# Hard test. This is the branch-integrity gate.
run([
    str(UTONIA_PY), "-c",
    (
        "import sys, torch, spconv.pytorch, torch_scatter, utonia; "
        "print('Utonia worker Python:', sys.version.split()[0]); "
        "print('Utonia worker Torch :', torch.__version__); "
        "print('Utonia worker CUDA  :', torch.version.cuda); "
        "print('Utonia import OK')"
    )
])

# ----------------------------------------------------------------------
# Official ANYmal-D simplified URDF
# ----------------------------------------------------------------------
ANYMAL_D_DESC = EXT / "anymal_d_simple_description"
if not ANYMAL_D_DESC.exists():
    run([
        "git", "clone", "--depth", "1",
        "https://github.com/ANYbotics/anymal_d_simple_description.git",
        str(ANYMAL_D_DESC),
    ])

ANYMAL_D_URDF = ANYMAL_D_DESC / "urdf" / "anymal.urdf"
if not ANYMAL_D_URDF.exists():
    raise RuntimeError("Official ANYmal-D URDF clone is incomplete.")

# ----------------------------------------------------------------------
# WAFT
# ----------------------------------------------------------------------
WAFT = EXT / "WAFT"
if not WAFT.exists():
    run([
        "git", "clone", "--depth", "1", "--recurse-submodules",
        "--branch", "waftv2",
        "https://github.com/princeton-vl/WAFT.git",
        str(WAFT),
    ])
else:
    print("Reusing WAFT clone:", WAFT)

waft_ckpt = WAFT / "ckpts" / "tar-c-t.pth"
waft_ckpt.parent.mkdir(parents=True, exist_ok=True)
if not waft_ckpt.exists():
    run([
        "gdown", "--id", "1CxzBQx0iSg6AyIgt6MF0ROlF_cAeZLPC",
        "-O", str(waft_ckpt),
    ])

da_dir = WAFT / "depth-anything-ckpts"
da_dir.mkdir(exist_ok=True)
da_ckpt = da_dir / "depth_anything_v2_vits.pth"
if not da_ckpt.exists():
    run([
        "wget", "-q", "-O", str(da_ckpt),
        "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth",
    ])

cfg = WAFT / "config" / "a1" / "tar-c-t.json"
if not cfg.exists() or not waft_ckpt.exists() or not da_ckpt.exists():
    raise RuntimeError("WAFT installation/checkpoints incomplete.")

print("Main Python       :", sys.version.split()[0])
print("Main Torch        :", torch.__version__)
print("Main CUDA         :", torch.version.cuda)
print("Utonia worker     :", UTONIA_PY)
print("ANYmal-D URDF     :", ANYMAL_D_URDF)
print("WAFT cfg          :", cfg)
print("WAFT checkpoint   :", waft_ckpt)
print("INSTALL / PREFLIGHT COMPLETE.")

# %% notebook cell 2
# ============================================================
# V7 INSTALL VERIFICATION — fixed notebook
# ============================================================
# The previous notebook needed a recovery cell because direct
# `uv pip install git+...Utonia` failed. Cell 0 above now performs the
# clone + .pth fix directly, so this cell only verifies the environment.

import subprocess
from pathlib import Path

if "UTONIA_PY" not in globals():
    raise RuntimeError("Run the install cell immediately above first.")

UTONIA_PY_PATH = Path(UTONIA_PY)
if not UTONIA_PY_PATH.exists():
    raise RuntimeError(f"Utonia Python missing: {UTONIA_PY_PATH}")

subprocess.check_call([
    str(UTONIA_PY_PATH), "-c",
    (
        "import sys, torch, spconv.pytorch, torch_scatter, utonia; "
        "print('Utonia worker Python:',sys.version.split()[0]); "
        "print('Utonia worker Torch :',torch.__version__); "
        "print('Utonia worker CUDA  :',torch.version.cuda); "
        "print('Utonia package      :',utonia.__file__); "
        "print('Utonia import OK')"
    )
])
print("Install verification passed.")

# %% notebook cell 3
# ======================================================================
# 1. CONFIGURATION, REPRODUCIBILITY, PATHS
# ======================================================================
import os, re, gc, math, random, copy, time, hashlib, warnings, contextlib
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation
from rosbags.highlevel import AnyReader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    brier_score_loss, roc_auc_score, confusion_matrix
)
import tartanair as ta

# -------- Durable storage / recovery --------
# Colab => Google Drive. SSH/Linux server => the persistent 140-GB allocation.
# If the server storage is mounted somewhere specific, set before launching:
#   export V7_PERSIST_ROOT=/path/to/your/140GB/mount/UGP_V7_FULL4B
IN_COLAB = bool(os.environ.get('COLAB_RELEASE_TAG')) or ('google.colab' in sys.modules)
USE_GOOGLE_DRIVE_CACHE = IN_COLAB
DRIVE_PROJECT_DIR = 'MyDrive/UGP_V7_FULL4B'

if 'WORK_ROOT' not in globals():
    if os.environ.get('V7_WORK_ROOT'):
        WORK_ROOT=Path(os.environ['V7_WORK_ROOT']).expanduser()
    elif os.environ.get('V7_PERSIST_ROOT'):
        WORK_ROOT=Path(os.environ['V7_PERSIST_ROOT']).expanduser()/'.work'
    elif Path('/content').exists():
        WORK_ROOT=Path('/content')
    elif Path('/workspace').exists():
        WORK_ROOT=Path('/workspace')
    else:
        WORK_ROOT=Path.home()/'.v7_work'
    WORK_ROOT.mkdir(parents=True,exist_ok=True)

PERSIST_ROOT=None
if USE_GOOGLE_DRIVE_CACHE:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        PERSIST_ROOT = Path('/content/drive')/DRIVE_PROJECT_DIR
        print('Using Google Drive persistent storage.')
    except Exception as e:
        print('Google Drive mount unavailable; falling back to local persistent storage:',e)

if PERSIST_ROOT is None:
    default_server_root=Path.home()/'UGP_V7_FULL4B_SERVER'
    PERSIST_ROOT=Path(os.environ.get('V7_PERSIST_ROOT',str(default_server_root))).expanduser()
    print('Using server/local persistent storage:',PERSIST_ROOT)

# IMPORTANT: raw-in-progress data is also placed on persistent storage. Auxiliary
# raw trajectories are still deleted only after physics + manifest + all four
# visual caches validate, so this does not cause all 35 raw runs to accumulate.
ROOT = PERSIST_ROOT/'raw_tartanground'
CACHE_ROOT = PERSIST_ROOT/'cache'
OUT_ROOT = PERSIST_ROOT/'results'
RECOVERY_ROOT = PERSIST_ROOT/'recovery'
V7F_FEATURE_RECOVERY_ROOT = RECOVERY_ROOT/'v7f_frozen_features'
for _d in (PERSIST_ROOT,ROOT,CACHE_ROOT,OUT_ROOT,RECOVERY_ROOT,V7F_FEATURE_RECOVERY_ROOT):
    _d.mkdir(parents=True,exist_ok=True)

# Recovery cadence. These overwrite a single checkpoint, so storage does not grow
# linearly with the number of saves.
V7_CHECKPOINT_EVERY_WINDOWS = 100
V7F_CHECKPOINT_EVERY_BATCHES = 20
DISK_HARD_RESERVE_GB = 8.0
DISK_WARN_FREE_GB = 40.0


def _fsync_dir(path):
    try:
        fd=os.open(str(Path(path)),os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    except Exception:
        pass


def atomic_torch_save(obj,path):
    """Power-loss safer torch.save: temp file -> fsync -> atomic rename -> fsync dir."""
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with open(tmp,'wb') as f:
        torch.save(obj,f)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path); _fsync_dir(path.parent)


def atomic_write_text(path,text):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with open(tmp,'w',encoding='utf-8') as f:
        f.write(text); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path); _fsync_dir(path.parent)


def atomic_json_write(path,obj):
    def _default(x):
        if isinstance(x,np.generic): return x.item()
        if isinstance(x,Path): return str(x)
        return str(x)
    atomic_write_text(path,json.dumps(obj,indent=2,default=_default))


def _capture_rng_state():
    state={
        'python':random.getstate(),
        'numpy':np.random.get_state(),
        'torch_cpu':torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['torch_cuda']=torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    if not state: return
    try:
        random.setstate(state['python']); np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch_cpu'])
        if torch.cuda.is_available() and 'torch_cuda' in state:
            torch.cuda.set_rng_state_all(state['torch_cuda'])
    except Exception as e:
        raise RuntimeError('Recovery RNG state is incompatible: '+repr(e))


def model_state_fingerprint(model):
    """Strong fingerprint for invalidating stale downstream feature/test caches."""
    h=hashlib.sha256()
    for name,t in model.state_dict().items():
        h.update(name.encode('utf-8'))
        a=t.detach().cpu().contiguous().numpy()
        h.update(str(a.shape).encode()); h.update(str(a.dtype).encode()); h.update(a.tobytes())
    return h.hexdigest()


def disk_status(path=PERSIST_ROOT):
    u=shutil.disk_usage(str(path))
    return dict(total_gb=u.total/1024**3,used_gb=u.used/1024**3,free_gb=u.free/1024**3)


def assert_disk_headroom(label='stage'):
    st=disk_status(PERSIST_ROOT)
    if st['free_gb'] < DISK_HARD_RESERVE_GB:
        raise RuntimeError(
            f'Persistent disk critically low before {label}: {st["free_gb"]:.1f} GB free. '
            f'Need at least {DISK_HARD_RESERVE_GB:.1f} GB reserve.'
        )
    if st['free_gb'] < DISK_WARN_FREE_GB:
        warnings.warn(f'Persistent disk has only {st["free_gb"]:.1f} GB free before {label}.',RuntimeWarning)
    return st

# Verify persistent path is writable now, not after hours of work.
_probe=RECOVERY_ROOT/'_persistent_write_probe.txt'
atomic_write_text(_probe,'v7 persistent storage write test\n')
if _probe.read_text().strip()!='v7 persistent storage write test':
    raise RuntimeError('Persistent storage write/read verification failed.')
_probe.unlink(missing_ok=True)
print('Persistent disk status [GB]:',disk_status())
print('Persistent ROOT:',ROOT)
print('Persistent CACHE_ROOT:',CACHE_ROOT)
print('Persistent RECOVERY_ROOT:',RECOVERY_ROOT)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU required.")
if torch.cuda.device_count() < 2:
    raise RuntimeError(
        f"This notebook is the 2x-RTX-4080 build and requires 2 visible CUDA GPUs; "
        f"found {torch.cuda.device_count()}."
    )
DEVICE=torch.device('cuda:0')
AUX_DEVICE=torch.device('cuda:1')
print("GPU 0:",torch.cuda.get_device_name(0))
print("GPU 1:",torch.cuda.get_device_name(1))
SEED=781
EPS=1e-6
CONTACT_THRESHOLD=10.0
TRAIN_LOOKBACK_SEC=20.0

# -------- Temporal experts --------
SHORT_CONTEXT_SEC=0.5
SHORT_HOP_SEC=0.3
SHORT_HORIZONS=[0.1]
LONG_CONTEXT_SEC=4.0
LONG_HOP_SEC=1.0
LONG_HORIZONS=[round(x,1) for x in np.arange(0.2,1.01,0.1)]
ALL_HORIZONS=[round(x,1) for x in np.arange(0.1,1.01,0.1)]
SHORT_RECENCY_HALF_LIFE_SEC=5.0
LONG_RECENCY_HALF_LIFE_SEC=6.0

PRIOR_EPOCHS_SHORT=3
PRIOR_EPOCHS_LONG=4
LOCAL_EPOCHS_SHORT=4
LOCAL_EPOCHS_LONG=4
LR_PRIOR=2e-4
LR_LOCAL=8e-5
WEIGHT_DECAY=1e-4
GRAD_CLIP=1.0

# -------- Full visual architecture --------
DINO_MODEL='hf_hub:timm/vit_large_patch16_dinov3.lvd1689m'
CONV_MODEL='hf_hub:timm/convnext_large.dinov3_lvd1689m'
VISION_INPUT=448
RAW_GRID=8                   # 8x8=64 cached spatial tokens per branch
RAW_TOKENS=RAW_GRID*RAW_GRID
LATENTS_PER_BRANCH=16
VIS_DIM=256
FLOW_HISTORY=4
POINTS_PER_FRAME=4096
# Must be defined BEFORE the Utonia cache builder is ever called.
GEOM_INPUT_POINTS=int(POINTS_PER_FRAME)
GEOM_REP_POINTS=64
# Auxiliary training trajectories are fully cached before their raw RGB-D/rosbag
# folders are removed from ephemeral local disk. The four diagnostic runs are kept.
CLEAN_AUX_RAW_AFTER_CACHE=True
FOOT_QUERY_PLAN_SEC=0.30       # planned body displacement conditioning for foot terrain queries
MAX_DEPTH_M=8.0
MIN_DEPTH_M=0.25

# TartanGround pinhole camera: 640x640, 90 degree FOV => f=320, c=(320,320).
CAM_W=640; CAM_H=640
FX=320.0; FY=320.0; CX=320.0; CY=320.0

# Geometry-defined traversable-looking support window in a camera-heading, gravity-aligned local frame.
# This is a geometric filter, NOT a manually chosen pixel crop.
GROUND_X_RANGE=(0.15,4.0)      # metres forward
GROUND_Y_ABS=2.5               # metres lateral
GROUND_Z_RANGE=(-2.0,0.35)     # metres relative to camera, z-up
GROUND_NORMAL_Z_MIN=0.35
GROUND_CROP_MARGIN=0.08
MIN_GROUND_PIXELS=500

# -------- V5-style loss design --------
W_Q=1.0; W_DQ=1.0; W_TAU=1.5; W_FORCE=2.0; W_BODY=0.5
W_BINARY_CONTACT=0.25; W_REGIME=0.40
W_FORCE_LOGMAG=0.25; W_FORCE_DIR=0.10
W_KIN=0.20; W_INVDYN=0.20; W_NEWTON=0.10
W_ZERO_FORCE=0.08; W_TREND=0.10
RIDGE=1e-3
N_REGIMES=4
PI_PHYS_DIM=47

# -------- Exact same diagnostic runs/tests as V4/V5/V6 --------
EXPERIMENTS=[
    dict(terrain_type='forest',environment='ForestEnv',run='P2000',ttests=[28.0,45.0,75.0,90.0]),
    dict(terrain_type='swamp_marsh',environment='GreatMarsh',run='P2023',ttests=[47.0,61.0,73.0,87.0]),
    dict(terrain_type='rough_uneven',environment='SeasonalForestWinter',run='P2000',ttests=[29.0,46.0,64.0,92.0]),
    dict(terrain_type='wet_uneven_proxy',environment='OldTownFall',run='P2000',ttests=[32.0,84.0,100.0,115.0]),
]

# -------- Expanded balanced multi-trajectory training split --------
# Legacy four buckets remain EXACTLY as before (7 training trajectories each).
#
# IMPORTANT CORRECTION:
# Official TartanGround metadata reports ModUrbanCity has 0 ANYmal trajectories.
# It therefore cannot supply the ROS-bag/proprioceptive leg data required here.
# Stair trajectories are discovered at runtime from URBAN environments that DO
# have ANYmal runs. Within each environment we try the highest run ID first
# ("newest" in the dataset's contiguous P20xx numbering), then move downward.
EXPECTED_TRAIN_TRAJECTORIES = 35
EXPECTED_STRICT_HOLDOUTS = 5
TRAIN_TRAJECTORIES_PER_BUCKET = 7
STAIR_TRAJECTORIES_TO_DISCOVER = 8   # 7 train + 1 strict unseen holdout

STAIR_TERRAIN_TYPE = 'stairs_urban'

# Official TartanGround ANYmal counts for eligible urban environments.
# Search order keeps the scan inside one urban environment as long as possible;
# if it cannot provide eight verified stair traversals, discovery continues in
# the next urban environment. OldTownFall is excluded because all eight of its
# ANYmal runs are already consumed by the legacy wet_uneven split/holdout.
STAIR_URBAN_ENV_COUNTS = [
    ('ModernCityDowntown', 11),
    ('Downtown', 18),
    ('ModularNeighborhood', 30),
    ('NordicHarbor', 28),
    ('OldTownSummer', 8),
]

# Stage-1 cheap pose prefilter.
STAIR_AUDIT_MIN_ROBUST_VERTICAL_SPAN_M = 0.25

# Stage-2 depth-geometry stair test. The values are deliberately broad enough
# for ordinary stairs while rejecting smooth ramps. These are data-quality
# thresholds, not model-training hyperparameters.
STAIR_GEOM_X_RANGE_M = (0.35, 3.20)
STAIR_GEOM_Y_ABS_M = 1.00
STAIR_GEOM_BIN_M = 0.06
STAIR_GEOM_MIN_POINTS_PER_BIN = 10
STAIR_GEOM_MIN_HORIZONTAL_NORMAL_Z = 0.70
STAIR_GEOM_MIN_RISE_M = 0.055
STAIR_GEOM_MAX_RISE_M = 0.30
STAIR_GEOM_MIN_TREAD_SPACING_M = 0.10
STAIR_GEOM_MAX_TREAD_SPACING_M = 0.65
STAIR_GEOM_MIN_STEP_GROUPS = 3
STAIR_GEOM_MIN_FLAT_FRACTION = 0.40
STAIR_GEOM_MIN_JUMP_ENERGY_RATIO = 0.45
STAIR_GEOM_MIN_PROFILE_SPAN_M = 0.18
STAIR_GEOM_MAX_RISE_CV = 0.70
STAIR_GEOM_MIN_SIGN_CONSISTENCY = 0.75
STAIR_GEOM_MIN_PASS_FRAMES = 3
STAIR_GEOM_MIN_EVENT_VERTICAL_SPAN_M = 0.15
STAIR_GEOM_EVENT_HALF_WINDOW_FRAMES = 15   # 10 Hz camera => about +/-1.5 s
STAIR_GEOM_FRAME_STRIDE = 3
STAIR_GEOM_MAX_FRAMES_PER_RUN = 120

# Selection is persisted. Rerunning the notebook does not silently choose a
# different staircase split after caches/checkpoints already exist.
STAIR_SELECTION_FILE = PERSIST_ROOT/'V7_STAIR_SELECTION.json'
STAIR_DISCOVERY_STATE_FILE = RECOVERY_ROOT/'V7_STAIR_DISCOVERY_INPROGRESS.json'

STRICT_HOLDOUTS = {
    'forest':            ('ForestEnv','P2001'),
    'swamp_marsh':       ('GreatMarsh','P2027'),
    'rough_uneven':      ('SeasonalForestWinter','P2006'),
    'wet_uneven_proxy':  ('OldTownFall','P2001'),
    # stairs_urban is filled only AFTER a run passes the true-stair audit.
}

EXPANDED_TRAIN_RUNS = {
    # evenly spread through the available trajectory IDs; exact original
    # diagnostic run is included in every legacy terrain.
    'forest': [
        ('ForestEnv','P2000'), ('ForestEnv','P2003'), ('ForestEnv','P2007'),
        ('ForestEnv','P2011'), ('ForestEnv','P2015'), ('ForestEnv','P2019'),
        ('ForestEnv','P2022'),
    ],
    'swamp_marsh': [
        ('GreatMarsh','P2000'), ('GreatMarsh','P2005'), ('GreatMarsh','P2010'),
        ('GreatMarsh','P2015'), ('GreatMarsh','P2020'), ('GreatMarsh','P2023'),
        ('GreatMarsh','P2029'),
    ],
    'rough_uneven': [
        ('SeasonalForestWinter','P2000'), ('SeasonalForestWinter','P2003'),
        ('SeasonalForestWinter','P2009'), ('SeasonalForestWinter','P2012'),
        ('SeasonalForestWinter','P2015'), ('SeasonalForestWinter','P2018'),
        ('SeasonalForestWinter','P2021'),
    ],
    'wet_uneven_proxy': [
        ('OldTownFall','P2000'), ('OldTownFall','P2002'), ('OldTownFall','P2003'),
        ('OldTownFall','P2004'), ('OldTownFall','P2005'), ('OldTownFall','P2006'),
        ('OldTownFall','P2007'),
    ],
    STAIR_TERRAIN_TYPE: [],  # populated by automatic true-stair discovery
}

# The static 4-terrain part must be exactly intact before any download.
_static_bad = {
    k:len(v) for k,v in EXPANDED_TRAIN_RUNS.items()
    if k!=STAIR_TERRAIN_TYPE and len(v)!=TRAIN_TRAJECTORIES_PER_BUCKET
}
if _static_bad:
    raise RuntimeError('Legacy training split changed unexpectedly: '+repr(_static_bad))

_static_train_keys={
    x for k,runs in EXPANDED_TRAIN_RUNS.items() if k!=STAIR_TERRAIN_TYPE for x in runs
}
_static_holdout_keys=set(STRICT_HOLDOUTS.values())
if len(_static_train_keys)!=28:
    raise RuntimeError(f'Legacy split should contain 28 unique training runs, got {len(_static_train_keys)}')
if len(_static_holdout_keys)!=4:
    raise RuntimeError(f'Legacy split should contain 4 strict holdouts, got {len(_static_holdout_keys)}')
if _static_train_keys & _static_holdout_keys:
    raise RuntimeError('Legacy strict-holdout leakage: '+repr(sorted(_static_train_keys&_static_holdout_keys)))

print('Static legacy training trajectories:',len(_static_train_keys))
print('Stair trajectories             : automatic discovery in expensive cell')
print('Static strict holdouts         :',len(_static_holdout_keys))


# Strongest existing V5 references measured on these same 16 tests.
BEST_EXISTING={
    0.1: {'q':0.0725,'dq':0.5004,'tau':2.9492,'force':6.2079},
    1.0: {'q':0.6432,'dq':1.2505,'tau':6.6952,'force':13.6940},
}

# Optional partial warm-start of V7 history/control/physics encoders from V5 checkpoints.
# Put the .pt files here if desired. It is NOT required for a complete run.
V5_SHORT_CKPT = PERSIST_ROOT/'optional_v5'/'V5_PRIOR_SHORT.pt'
V5_LONG_CKPT  = PERSIST_ROOT/'optional_v5'/'V5_PRIOR_LONG.pt'


def seed_everything_v7(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

seed_everything_v7(SEED)
print('DEVICE:',DEVICE, torch.cuda.get_device_name(0))
print('PERSIST_ROOT:',PERSIST_ROOT)
print('CACHE_ROOT  :',CACHE_ROOT)
print('OUT_ROOT    :',OUT_ROOT)
print('RECOVERY_ROOT:',RECOVERY_ROOT)
print('ANYmal-D URDF:',ANYMAL_D_URDF)
print('Foot-query plan horizon:',FOOT_QUERY_PLAN_SEC,'s')
display(pd.DataFrame(EXPERIMENTS))

# %% notebook cell 4
# ======================================================================
# 2. TARTANGROUND PHYSICS EXTRACTION + NORMALIZATION
# Reuses the exact signal alignment logic that underpinned V4/V5.
# ======================================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def causal_hold(t_src, x_src, t_target):
    idx = np.searchsorted(t_src, t_target, side='right') - 1
    if np.any(idx < 0):
        raise RuntimeError('Target time occurs before first source measurement.')
    return x_src[idx]

def stamp_to_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9

def load_metric_depth(path):
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError(f'Could not read {path}')
    if raw.dtype == np.uint8 and raw.ndim == 3 and raw.shape[2] == 4:
        raw = np.ascontiguousarray(raw)
        depth = raw.view('<f4').reshape(raw.shape[0], raw.shape[1])
    elif raw.dtype == np.float32 and raw.ndim == 2:
        depth = raw
    else:
        raise RuntimeError(f'Unexpected depth format: {raw.dtype}, {raw.shape}')
    depth = depth.astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    return depth

def nearest_index(time, t):
    return int(np.argmin(np.abs(time - float(t))))

def extract_run_physics(base, run):
    cache = CACHE_ROOT / f'{base.parent.parent.name}_{run}_physics.npz'
    if cache.exists():
        try:
            z = np.load(cache, allow_pickle=False)
            out = {k: z[k] for k in z.files}
            required = {
                'time','grid_original','crop_start_original','P','B','U',
                'L_LEG','E','contact','actuator_for_joint'
            }
            missing = required - set(out)
            if missing:
                raise RuntimeError('missing physics-cache fields '+repr(sorted(missing)))
            N = len(out['time'])
            if N < 2:
                raise RuntimeError(f'physics cache has too few samples: {N}')
            expected_shapes = {
                'P': (N,4,12), 'B': (N,12), 'U': (N,6),
                'L_LEG': (N,4,3), 'E': (N,4,3), 'contact': (N,4),
                'grid_original': (N,), 'actuator_for_joint': (12,),
            }
            for k,shape in expected_shapes.items():
                if tuple(out[k].shape) != tuple(shape):
                    raise RuntimeError(f'{k} shape {out[k].shape} != {shape}')
            if np.any(np.diff(np.asarray(out['time'],np.float64)) <= 0):
                raise RuntimeError('physics cache time is not strictly increasing')
            for k in ('P','B','U','L_LEG','E'):
                a=out[k]
                if not (np.isfinite(a[0]).all() and np.isfinite(a[-1]).all()):
                    raise RuntimeError(f'non-finite edge sample in {k}')
            print('Reusing validated physics cache',cache)
            return out
        except Exception as e:
            print('Cached physics file is invalid; rebuilding:',e)
            try: cache.unlink()
            except Exception: pass

    imu_dir = base/'imu'
    bag_path = base/'rosbags'/f'{run}.bag'
    imu_time = np.load(imu_dir/'imu_time.npy').astype(np.float64)
    acc = np.load(imu_dir/'acc.npy').astype(np.float32)
    gyro = np.loadtxt(imu_dir/'gyro.txt').astype(np.float32)

    topics = {
        'joints':'/state_estimator/joint_states',
        'lf':'/state_estimator/contact_force_lf_foot',
        'rf':'/state_estimator/contact_force_rf_foot',
        'lh':'/state_estimator/contact_force_lh_foot',
        'rh':'/state_estimator/contact_force_rh_foot',
        'twist':'/state_estimator/twist',
        'command':'/path_planning_and_following/twist',
    }
    physical_topics = {topics[k] for k in ['joints','lf','rf','lh','rh','twist']}
    with AnyReader([bag_path]) as reader:
        selected = [c for c in reader.connections if c.topic in physical_topics]
        _, t0_ns, _ = next(reader.messages(connections=selected))
    t0_sec = t0_ns * 1e-9

    def msg_time(msg, bag_ns):
        try:
            absolute = stamp_to_seconds(msg.header.stamp)
        except Exception:
            absolute = bag_ns * 1e-9
        return absolute - t0_sec

    data = {k:{'t':[],'x':[]} for k in topics}
    expected_joint_names = [
        'LF_HAA','LF_HFE','LF_KFE','RF_HAA','RF_HFE','RF_KFE',
        'LH_HAA','LH_HFE','LH_KFE','RH_HAA','RH_HFE','RH_KFE'
    ]
    joint_reorder = None
    with AnyReader([bag_path]) as reader:
        selected = [c for c in reader.connections if c.topic in set(topics.values())]
        for connection, timestamp, rawdata in reader.messages(connections=selected):
            msg = reader.deserialize(rawdata, connection.msgtype)
            topic = connection.topic
            t_msg = msg_time(msg, timestamp)
            if topic == topics['joints']:
                if joint_reorder is None:
                    names = list(msg.name)
                    missing = [n for n in expected_joint_names if n not in names]
                    if missing:
                        raise RuntimeError(f'Missing expected joints: {missing}')
                    joint_reorder = np.array([names.index(n) for n in expected_joint_names], dtype=int)
                pos = np.asarray(msg.position, np.float32)[joint_reorder]
                vel = np.asarray(msg.velocity, np.float32)[joint_reorder]
                eff = np.asarray(msg.effort, np.float32)[joint_reorder]
                x = np.concatenate([pos,vel,eff])
                key='joints'
            elif topic in {topics['lf'],topics['rf'],topics['lh'],topics['rh']}:
                x = np.array([msg.wrench.force.x,msg.wrench.force.y,msg.wrench.force.z], np.float32)
                key = 'lf' if topic==topics['lf'] else 'rf' if topic==topics['rf'] else 'lh' if topic==topics['lh'] else 'rh'
            elif topic == topics['twist']:
                tw=msg.twist.twist
                x=np.array([tw.linear.x,tw.linear.y,tw.linear.z,tw.angular.x,tw.angular.y,tw.angular.z],np.float32)
                key='twist'
            elif topic == topics['command']:
                tw=msg.twist
                x=np.array([tw.linear.x,tw.linear.y,tw.linear.z,tw.angular.x,tw.angular.y,tw.angular.z],np.float32)
                key='command'
            else:
                continue
            data[key]['t'].append(t_msg); data[key]['x'].append(x)

    for k in data:
        data[k]['t']=np.asarray(data[k]['t'],np.float64)
        data[k]['x']=np.asarray(data[k]['x'],np.float32)
        o=np.argsort(data[k]['t']); data[k]['t']=data[k]['t'][o]; data[k]['x']=data[k]['x'][o]

    stream_keys=['joints','lf','rf','lh','rh','twist','command']
    common_start=max(imu_time[0], *[data[k]['t'][0] for k in stream_keys])
    common_end=min(imu_time[-1], *[data[k]['t'][-1] for k in stream_keys])
    grid_mask=(imu_time>=common_start)&(imu_time<=common_end)
    grid_original=imu_time[grid_mask]
    acc_sync=acc[grid_mask]; gyro_sync=gyro[grid_mask]
    joint_sync=causal_hold(data['joints']['t'],data['joints']['x'],grid_original)
    lf=causal_hold(data['lf']['t'],data['lf']['x'],grid_original)
    rf=causal_hold(data['rf']['t'],data['rf']['x'],grid_original)
    lh=causal_hold(data['lh']['t'],data['lh']['x'],grid_original)
    rh=causal_hold(data['rh']['t'],data['rh']['x'],grid_original)
    body_twist=causal_hold(data['twist']['t'],data['twist']['x'],grid_original)
    U=causal_hold(data['command']['t'],data['command']['x'],grid_original).astype(np.float32)

    q=joint_sync[:,0:12]; dq=joint_sync[:,12:24]; tau=joint_sync[:,24:36]
    legs=[[0,1,2],[3,4,5],[6,7,8],[9,10,11]]
    force_list=[lf,rf,lh,rh]
    P=np.stack([np.concatenate([q[:,idx],dq[:,idx],tau[:,idx],force_list[j]],axis=1) for j,idx in enumerate(legs)],axis=1).astype(np.float32)
    B=np.concatenate([acc_sync,gyro_sync,body_twist],axis=1).astype(np.float32)

    # Low-level actuator commands.
    ACT='/state_estimator/actuator_readings'
    cmd_t=[]; cmd_tau=[]; cmd_mode=[]; st_t=[]; st_pos=[]
    with AnyReader([bag_path]) as reader:
        conns=[c for c in reader.connections if c.topic==ACT]
        if not conns: raise RuntimeError(f'{ACT} missing in run {run}')
        for c,bag_ns,raw in reader.messages(connections=conns):
            msg=reader.deserialize(raw,c.msgtype); readings=list(msg.readings)
            if len(readings)!=12: continue
            cs=[]; ct=[]; cm=[]; ss=[]; sp=[]
            for r in readings:
                cs.append(stamp_to_seconds(r.commanded.header.stamp)); ct.append(float(r.commanded.joint_torque)); cm.append(int(r.commanded.mode))
                ss.append(stamp_to_seconds(r.state.header.stamp)); sp.append(float(r.state.joint_position))
            cmd_t.append(float(np.median(cs))-t0_sec); cmd_tau.append(ct); cmd_mode.append(cm)
            st_t.append(float(np.median(ss))-t0_sec); st_pos.append(sp)
    cmd_t=np.asarray(cmd_t,np.float64); cmd_tau=np.asarray(cmd_tau,np.float32); cmd_mode=np.asarray(cmd_mode,np.int32)
    st_t=np.asarray(st_t,np.float64); st_pos=np.asarray(st_pos,np.float32)
    o=np.argsort(cmd_t); cmd_t=cmd_t[o]; cmd_tau=cmd_tau[o]; cmd_mode=cmd_mode[o]
    o=np.argsort(st_t); st_t=st_t[o]; st_pos=st_pos[o]

    # Crop again if actuator stream begins later. This avoids backward extrapolation.
    valid_start=max(grid_original[0],cmd_t[0],st_t[0])
    keep=grid_original>=valid_start
    grid_original=grid_original[keep]; P=P[keep]; B=B[keep]; U=U[keep]

    act_pos_sync=causal_hold(st_t,st_pos,grid_original)
    qcan=P[:,:,0:3].reshape(len(P),12)
    m=min(5000,len(P)); cost=np.zeros((12,12),np.float64)
    for ai in range(12):
        for ji in range(12):
            d=act_pos_sync[:m,ai]-qcan[:m,ji]; scale=np.std(qcan[:m,ji])+1e-6
            cost[ai,ji]=np.sqrt(np.mean(d*d))/scale
    rows,cols=linear_sum_assignment(cost)
    actuator_for_joint=np.empty(12,dtype=int)
    for ai,ji in zip(rows,cols): actuator_for_joint[ji]=ai

    cmd_tau_sync=causal_hold(cmd_t,cmd_tau,grid_original)
    cmd_mode_sync=causal_hold(cmd_t,cmd_mode,grid_original)
    L=cmd_tau_sync[:,actuator_for_joint].astype(np.float32)
    L_MODE=cmd_mode_sync[:,actuator_for_joint]
    unsupported=~np.isin(L_MODE,[10,12,13])
    if float(unsupported.mean())>0.01:
        raise RuntimeError('>1% actuator samples are outside torque-containing modes.')
    L_LEG=L.reshape(len(L),4,3)
    measured_tau=P[:,:,6:9]
    E=np.zeros_like(measured_tau,np.float32)
    E[0]=measured_tau[0]-L_LEG[0]
    E[1:]=measured_tau[1:]-L_LEG[:-1]

    crop_start_original=float(grid_original[0])
    time=(grid_original-crop_start_original).astype(np.float64)
    contact=(np.linalg.norm(P[:,:,9:12],axis=-1)>CONTACT_THRESHOLD).astype(np.float32)

    out=dict(time=time,grid_original=grid_original.astype(np.float64),crop_start_original=np.array(crop_start_original),
             P=P,B=B,U=U,L_LEG=L_LEG,E=E,contact=contact,actuator_for_joint=actuator_for_joint)
    np.savez_compressed(cache,**out)
    return out

def make_stats_between(data, start_time, end_time):
    t = data['time']
    mask = (t >= float(start_time)) & (t <= float(end_time))
    if int(mask.sum()) < 50:
        raise RuntimeError(
            f'Insufficient samples for normalization: '
            f'{start_time:.2f} -> {end_time:.2f}, n={int(mask.sum())}'
        )
    P = data['P'][mask]
    B = data['B'][mask]
    U = data['U'][mask]
    return {
        'P_mean': P.mean(axis=(0,1)).astype(np.float32),
        'P_std': (P.std(axis=(0,1)) + EPS).astype(np.float32),
        'B_mean': B.mean(axis=0).astype(np.float32),
        'B_std': (B.std(axis=0) + EPS).astype(np.float32),
        'U_mean': U.mean(axis=0).astype(np.float32),
        'U_std': (U.std(axis=0) + EPS).astype(np.float32),
    }

def make_local_stats(data, ttest):
    return make_stats_between(
        data,
        max(float(data['time'][0]), float(ttest)-TRAIN_LOOKBACK_SEC),
        float(ttest)
    )

def normalize_arrays(data, stats):
    P = (data['P'] - stats['P_mean']) / stats['P_std']
    B = (data['B'] - stats['B_mean']) / stats['B_std']
    U = (data['U'] - stats['U_mean']) / stats['U_std']
    # L is a torque-like quantity; normalize with measured-torque stats.
    L = (data['L_LEG'] - stats['P_mean'][6:9]) / stats['P_std'][6:9]
    E = data['E'] / stats['P_std'][6:9]
    return (
        P.astype(np.float32),
        B.astype(np.float32),
        U.astype(np.float32),
        L.astype(np.float32),
        E.astype(np.float32),
    )

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

def tensor(x):
    return torch.as_tensor(x, dtype=torch.float32, device=DEVICE)

def horizon_offsets(data, horizons):
    dt = float(np.median(np.diff(data['time'])))
    offs = [int(round(float(h)/dt)) for h in horizons]
    if any(o < 1 for o in offs):
        raise RuntimeError('Horizon offset < 1 sample.')
    return offs


def download_run_v7(env,run):
    base=ROOT/env/'Data_anymal'/run
    required=[
        base/'imu', base/'rosbags'/f'{run}.bag', base/'image_lcam_front',
        base/'depth_lcam_front', base/'pose_lcam_front.txt'
    ]
    if all(p.exists() for p in required):
        print('Reusing raw data',env,run)
        return base
    ta.init(str(ROOT))
    print('Downloading',env,run)
    ta.download_ground(
        env=[env],version=['anymal'],traj=[run],
        modality=['imu','rosbag','image','depth','meta'],
        camera_name=['lcam_front'],unzip=True
    )
    missing=[str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError('TartanGround download incomplete: '+repr(missing))
    return base

# Cache key from V4 extract_run_physics expects CACHE_ROOT and still works.
print('Physics extraction helpers ready.')

# %% notebook cell 5
# ======================================================================
# 3. METRIC RGB-D GEOMETRY + CAUSAL CAMERA MANIFEST
# ======================================================================

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


def ground_geometry(depth,pose7,relax=False):
    valid=np.isfinite(depth)&(depth>=MIN_DEPTH_M)&(depth<=MAX_DEPTH_M)
    pc=dense_camera_points(depth)
    nc=dense_normals_camera(pc,valid)
    Rlw,Rwc=heading_frame_from_pose(pose7)
    # local vectors relative to camera
    world=np.einsum('ij,hwj->hwi',Rwc,pc)
    local=np.einsum('ij,hwj->hwi',Rlw,world)
    nw=np.einsum('ij,hwj->hwi',Rwc,nc)
    nl=np.einsum('ij,hwj->hwi',Rlw,nw)
    x,y,z=local[...,0],local[...,1],local[...,2]
    nz=np.abs(nl[...,2])
    mask=(valid & (x>=GROUND_X_RANGE[0])&(x<=GROUND_X_RANGE[1]) &
          (np.abs(y)<=GROUND_Y_ABS) & (z>=GROUND_Z_RANGE[0])&(z<=GROUND_Z_RANGE[1]) &
          (nz>=GROUND_NORMAL_Z_MIN))
    if mask.sum()<MIN_GROUND_PIXELS:
        # Relax only the surface-normal criterion; metric bounds remain unchanged.
        mask=(valid & (x>=GROUND_X_RANGE[0])&(x<=GROUND_X_RANGE[1]) &
              (np.abs(y)<=GROUND_Y_ABS) & (z>=GROUND_Z_RANGE[0])&(z<=GROUND_Z_RANGE[1]))
    if mask.sum()<128:
        raise RuntimeError(f'Only {int(mask.sum())} metric ground pixels after geometric selection.')
    return mask,local.astype(np.float32),nl.astype(np.float32)


def bbox_from_mask(mask):
    ys,xs=np.where(mask)
    x0,x1=int(xs.min()),int(xs.max()+1); y0,y1=int(ys.min()),int(ys.max()+1)
    H,W=mask.shape
    mx=int((x1-x0)*GROUND_CROP_MARGIN); my=int((y1-y0)*GROUND_CROP_MARGIN)
    x0=max(0,x0-mx); x1=min(W,x1+mx); y0=max(0,y0-my); y1=min(H,y1+my)
    # ensure non-degenerate crop
    if x1-x0<32 or y1-y0<32: raise RuntimeError('Geometric crop too small.')
    return np.array([x0,y0,x1,y1],np.int32)


def build_frame_manifest(base,env,run,data):
    path=CACHE_ROOT/f'{env}_{run}_frame_manifest.npz'
    if path.exists():
        try:
            z=np.load(path,allow_pickle=False); out={k:z[k] for k in z.files}
            required={'frame_ids','frame_nums','camera_times','poses','crop_boxes','ground_fraction','visual_index'}
            missing=required-set(out)
            if missing:
                raise RuntimeError('missing manifest fields '+repr(sorted(missing)))
            N=len(out['frame_ids'])
            if N<=0:
                raise RuntimeError('manifest has zero frames')
            if not all(len(out[k])==N for k in ('frame_nums','camera_times','poses','crop_boxes','ground_fraction')):
                raise RuntimeError('manifest per-frame lengths disagree')
            if len(out['visual_index'])!=len(data['time']):
                raise RuntimeError(
                    f'manifest robot-grid length {len(out["visual_index"])} != physics length {len(data["time"])}'
                )
            if np.any(np.diff(np.asarray(out['camera_times'],np.float64)) < -1e-10):
                raise RuntimeError('manifest camera times are not monotonic')
            vi=np.asarray(out['visual_index'],np.int64)
            if np.any(vi>=N):
                raise RuntimeError('manifest visual_index exceeds frame count')
            print('Reusing validated frame manifest',path)
            return out
        except Exception as e:
            print('Cached frame manifest is invalid; rebuilding:',e)
            try: path.unlink()
            except Exception: pass
    rgb_dir=base/'image_lcam_front'; depth_dir=base/'depth_lcam_front'
    rgb_ids={f.name.split('_')[0] for f in rgb_dir.glob('*_lcam_front.png')}
    dep_ids={f.name.split('_')[0] for f in depth_dir.glob('*_lcam_front_depth.png')}
    ids=sorted(rgb_ids&dep_ids,key=lambda x:int(x))
    nums=np.asarray([int(x) for x in ids],np.int64)
    cam_orig=np.loadtxt(base/'imu'/'cam_time.txt').reshape(-1).astype(np.float64)
    poses=np.loadtxt(base/'pose_lcam_front.txt').astype(np.float64)
    if nums.max()>=len(cam_orig) or nums.max()>=len(poses):
        raise RuntimeError('Frame id exceeds camera-time or pose rows.')
    times=cam_orig[nums]-float(data['crop_start_original'])
    keep=(times>=data['time'][0]-1e-8)&(times<=data['time'][-1]+1e-8)
    ids=np.asarray(ids)[keep]; nums=nums[keep]; times=times[keep]; pose_rows=poses[nums]
    boxes=[]; ground_fracs=[]
    print('Computing metric ground support/crop for',len(ids),'frames...')
    for j,fid in enumerate(ids):
        depth=load_metric_depth(depth_dir/f'{fid}_lcam_front_depth.png')
        mask,_,_=ground_geometry(depth,pose_rows[j])
        boxes.append(bbox_from_mask(mask)); ground_fracs.append(float(mask.mean()))
        if (j+1)%100==0: print(' ',j+1,'/',len(ids))
    vi=np.searchsorted(times,data['time'],side='right')-1
    if np.any((vi>=0)&(times[np.maximum(vi,0)]>data['time']+1e-9)):
        raise RuntimeError('Causal image-index test failed.')
    out=dict(frame_ids=ids.astype('<U16'),frame_nums=nums,camera_times=times.astype(np.float64),
             poses=pose_rows.astype(np.float64),crop_boxes=np.asarray(boxes,np.int32),
             ground_fraction=np.asarray(ground_fracs,np.float32),visual_index=vi.astype(np.int64))
    np.savez_compressed(path,**out)
    return out


def get_frame_inputs(base,manifest,j):
    fid=str(manifest['frame_ids'][j]); box=manifest['crop_boxes'][j]; pose=manifest['poses'][j]
    rgb=np.asarray(Image.open(base/'image_lcam_front'/f'{fid}_lcam_front.png').convert('RGB'))
    depth=load_metric_depth(base/'depth_lcam_front'/f'{fid}_lcam_front_depth.png')
    mask,local,normals=ground_geometry(depth,pose)
    x0,y0,x1,y1=[int(x) for x in box]
    crop_rgb=rgb[y0:y1,x0:x1]
    crop_mask=mask[y0:y1,x0:x1].astype(np.float32)
    return fid,rgb,depth,mask,local,normals,crop_rgb,crop_mask


def weighted_pool(feat,mask,out_hw=RAW_GRID):
    # feat: B,C,H,W; mask: B,1,H0,W0. Pool only geometrically selected pixels.
    m=F.interpolate(mask,size=feat.shape[-2:],mode='nearest')
    num=F.adaptive_avg_pool2d(feat*m,(out_hw,out_hw))
    den=F.adaptive_avg_pool2d(m,(out_hw,out_hw)).clamp_min(1e-3)
    pooled=num/den
    valid=F.adaptive_avg_pool2d(m,(out_hw,out_hw))
    return torch.cat([pooled,valid],dim=1)


# ----------------------------------------------------------------------
# VERIFIED ANYMAL-D LEG FORWARD KINEMATICS
# ----------------------------------------------------------------------
# We intentionally use the official ANYbotics ANYmal-D URDF for leg FK only.
# The TartanGround camera rig is synthetic; this code does NOT invent a URDF
# camera-to-base transform. Visual ground support stays camera-metric, while
# foot queries receive verified base-frame foot positions.
import xml.etree.ElementTree as ET


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


print('Metric geometric mask helpers ready.')

# %% notebook cell 6
# ======================================================================
# 4. FOUR MANDATORY VISUAL CACHE BUILDERS
# Each foundation model is loaded, used, and unloaded separately.
# ======================================================================

def _validated_visual_cache(path,meta,manifest,label,expected_model=None,expected_ndim=None,expected_axes=None):
    """Reuse only a complete cache that matches the current manifest/config."""
    if not (path.exists() and meta.exists()):
        return None
    try:
        info=json.loads(meta.read_text())
        arr=np.load(path,mmap_mode='r')
        N=len(manifest['frame_ids'])
        if len(arr)!=N:
            raise RuntimeError(f'frame count {len(arr)} != manifest {N}')
        if list(info.get('shape',[]))!=list(arr.shape):
            raise RuntimeError('metadata shape does not match .npy shape')
        if expected_model is not None and info.get('model')!=expected_model:
            raise RuntimeError(f'model mismatch: {info.get("model")} != {expected_model}')
        if expected_ndim is not None and arr.ndim!=expected_ndim:
            raise RuntimeError(f'ndim {arr.ndim} != {expected_ndim}')
        for ax,val in (expected_axes or {}).items():
            if arr.shape[int(ax)]!=int(val):
                raise RuntimeError(f'axis {ax} size {arr.shape[int(ax)]} != {val}')
        if N:
            for j in sorted({0,N-1}):
                if not np.isfinite(np.asarray(arr[j],np.float32)).all():
                    raise RuntimeError(f'non-finite values at frame {j}')
        print(f'Reusing validated {label}',arr.shape)
        return arr
    except Exception as e:
        print(f'Cached {label} is invalid/stale; rebuilding:',e)
        for q in (path,meta):
            try: q.unlink()
            except Exception: pass
        return None

def _preprocess_imagenet(rgb,size=VISION_INPUT):
    x=cv2.resize(rgb,(size,size),interpolation=cv2.INTER_AREA)
    x=torch.from_numpy(x).permute(2,0,1).float()/255.0
    mean=torch.tensor([0.485,0.456,0.406])[:,None,None]
    std=torch.tensor([0.229,0.224,0.225])[:,None,None]
    return (x-mean)/std


def _mask_tensor(mask,size=VISION_INPUT):
    m=cv2.resize(mask.astype(np.float32),(size,size),interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(m)[None]


def _nchw(x):
    if x.ndim!=4: raise RuntimeError('Expected 4D feature map, got '+repr(x.shape))
    # timm feature maps are normally NCHW. Detect NHWC defensively.
    if x.shape[1] < x.shape[-1] and x.shape[-1] in (192,384,768,1024,1536):
        x=x.permute(0,3,1,2).contiguous()
    return x


def build_dinov3_cache(base,env,run,manifest):
    path=CACHE_ROOT/f'{env}_{run}_v7_dinov3.npy'
    meta=CACHE_ROOT/f'{env}_{run}_v7_dinov3_meta.json'
    arr=_validated_visual_cache(
        path,meta,manifest,'DINOv3',expected_model=DINO_MODEL,
        expected_ndim=3,expected_axes={1:RAW_TOKENS}
    )
    if arr is not None: return arr
    print('Loading DINOv3 ViT-L/16...')
    model=timm.create_model(DINO_MODEL,pretrained=True,features_only=True,out_indices=(-1,)).to(DEVICE).eval()
    for p in model.parameters(): p.requires_grad=False
    N=len(manifest['frame_ids']); mm=None
    with torch.inference_mode():
        for j in range(N):
            *_,crop_rgb,crop_mask=get_frame_inputs(base,manifest,j)
            x=_preprocess_imagenet(crop_rgb)[None].to(DEVICE)
            m=_mask_tensor(crop_mask)[None].to(DEVICE)
            f=_nchw(model(x)[-1])
            p=weighted_pool(f,m).flatten(2).transpose(1,2)[0].float().cpu().numpy()
            if mm is None:
                mm=np.lib.format.open_memmap(path,mode='w+',dtype=np.float16,shape=(N,RAW_TOKENS,p.shape[-1]))
            mm[j]=p.astype(np.float16)
            if (j+1)%50==0: mm.flush(); print(' DINO',j+1,'/',N)
    mm.flush(); json.dump({'model':DINO_MODEL,'shape':list(mm.shape)},open(meta,'w'))
    del model,mm; gc.collect(); torch.cuda.empty_cache()
    return np.load(path,mmap_mode='r')


def build_convnext_cache(base,env,run,manifest):
    path=CACHE_ROOT/f'{env}_{run}_v7_convnext.npy'; meta=CACHE_ROOT/f'{env}_{run}_v7_convnext_meta.json'
    arr=_validated_visual_cache(
        path,meta,manifest,'ConvNeXt',expected_model=CONV_MODEL,
        expected_ndim=3,expected_axes={1:RAW_TOKENS}
    )
    if arr is not None: return arr
    print('Loading DINOv3 ConvNeXt-Large...')
    model=timm.create_model(CONV_MODEL,pretrained=True,features_only=True,out_indices=(0,1,2,3)).to(DEVICE).eval()
    for p in model.parameters(): p.requires_grad=False
    N=len(manifest['frame_ids']); mm=None
    with torch.inference_mode():
        for j in range(N):
            *_,crop_rgb,crop_mask=get_frame_inputs(base,manifest,j)
            x=_preprocess_imagenet(crop_rgb)[None].to(DEVICE); m=_mask_tensor(crop_mask)[None].to(DEVICE)
            feats=[_nchw(q) for q in model(x)]
            pooled=[]; valids=[]
            for f in feats:
                q=weighted_pool(f,m)
                pooled.append(q[:,:-1]); valids.append(q[:,-1:])
            q=torch.cat(pooled+[torch.stack(valids).mean(0)],dim=1)
            p=q.flatten(2).transpose(1,2)[0].float().cpu().numpy()
            if mm is None:
                mm=np.lib.format.open_memmap(path,mode='w+',dtype=np.float16,shape=(N,RAW_TOKENS,p.shape[-1]))
            mm[j]=p.astype(np.float16)
            if (j+1)%50==0: mm.flush(); print(' CONV',j+1,'/',N)
    mm.flush(); json.dump({'model':CONV_MODEL,'shape':list(mm.shape)},open(meta,'w'))
    del model,mm; gc.collect(); torch.cuda.empty_cache()
    return np.load(path,mmap_mode='r')


def farthest_point_indices(x,k):
    x=np.asarray(x,np.float32); N=len(x)
    if N==0: raise RuntimeError('FPS received zero points.')
    if N<=k:
        return np.resize(np.arange(N,dtype=np.int64),k)
    idx=np.empty(k,np.int64); idx[0]=int(np.argmin(np.sum(x*x,axis=1)))
    d=np.sum((x-x[idx[0]])**2,axis=1)
    for i in range(1,k):
        idx[i]=int(np.argmax(d)); d=np.minimum(d,np.sum((x-x[idx[i]])**2,axis=1))
    return idx


def select_geometry_points(rgb,mask,local,normals,n=POINTS_PER_FRAME):
    ys,xs=np.where(mask)
    if len(xs)<128: raise RuntimeError('Too few ground points.')
    # Uniform deterministic subsample before Utonia. Keep spatial coverage with FPS if necessary.
    coord=local[ys,xs]; normal=normals[ys,xs]; color=rgb[ys,xs].astype(np.float32)/255.0
    # local roughness: neighbor normal deviation approximated by |1-|n_z||
    rough=(1.0-np.clip(np.abs(normal[:,2]),0,1))[:,None]
    if len(coord)>n:
        # Fast deterministic spatially spread sampling for the Utonia INPUT cloud.
        # Do not run O(N*k) FPS for k=4096 on every frame; Utonia itself is the geometric encoder.
        # np.where is raster ordered, so evenly spaced indices cover the selected image/ground support.
        sel=np.linspace(0,len(coord)-1,n,dtype=np.int64)
        coord,normal,color,rough=coord[sel],normal[sel],color[sel],rough[sel]
    elif len(coord)<n:
        sel=np.resize(np.arange(len(coord)),n); coord,normal,color,rough=coord[sel],normal[sel],color[sel],rough[sel]
    return coord.astype(np.float32),color.astype(np.float32),normal.astype(np.float32),rough.astype(np.float32)


# ----------------------------------------------------------------------
# UTONIA / PTv3 GEOMETRY BRANCH
#
# Utonia is executed in the isolated Python-3.10 / Torch-2.5 / cu124
# worker created in Cell 0. This avoids the Python-3.13 spconv ABI problem
# without deleting or replacing the Utonia branch.
# ----------------------------------------------------------------------

def _write_utonia_worker(worker_path):
    worker_code = r"""
import sys, json, gc
from pathlib import Path
import numpy as np
import torch
import utonia

stage = Path(sys.argv[1])
out_path = Path(sys.argv[2])
meta_path = Path(sys.argv[3])
rep_points = int(sys.argv[4])

if not torch.cuda.is_available():
    raise RuntimeError("Utonia worker cannot see CUDA.")

device = torch.device("cuda")

cfg = dict(
    enc_patch_size=[1024 for _ in range(5)],
    enable_flash=False,
)

print("Loading Utonia pretrained PTv3 in isolated worker...")
model = utonia.load(
    "utonia",
    repo_id="Pointcept/Utonia",
    custom_config=cfg,
).to(device)
model.eval()
for p in model.parameters():
    p.requires_grad = False

transform = utonia.transform.default(
    scale=0.5,
    apply_z_positive=False,
    normalize_coord=False,
)

coord = np.load(stage / "coord.npy", mmap_mode="r")
color = np.load(stage / "color.npy", mmap_mode="r")
normal = np.load(stage / "normal.npy", mmap_mode="r")
rough = np.load(stage / "rough.npy", mmap_mode="r")
rep = np.load(stage / "rep.npy", mmap_mode="r")

N = len(coord)
mm = None
feat_dim = None

def extract_features(c, col, nor):
    point = {
        "coord": np.asarray(c, np.float32).copy(),
        "color": np.asarray(col, np.float32).copy(),
        "normal": np.asarray(nor, np.float32).copy(),
    }
    point = transform(point)

    for key in list(point.keys()):
        if isinstance(point[key], torch.Tensor):
            point[key] = point[key].to(device, non_blocking=True)

    with torch.inference_mode():
        point = model(point)

    # Same hierarchical recovery logic as the official Utonia usage pattern.
    for _ in range(2):
        if "pooling_parent" not in point.keys():
            break
        if "pooling_inverse" not in point.keys():
            raise RuntimeError("Utonia returned pooling_parent without pooling_inverse.")
        parent = point.pop("pooling_parent")
        inv = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inv]], dim=-1)
        point = parent

    while "pooling_parent" in point.keys():
        if "pooling_inverse" not in point.keys():
            raise RuntimeError("Utonia returned pooling_parent without pooling_inverse.")
        parent = point.pop("pooling_parent")
        inv = point.pop("pooling_inverse")
        parent.feat = point.feat[inv]
        point = parent

    feat = point.feat
    if "inverse" in point.keys():
        feat = feat[point.inverse]

    feat = feat.float().cpu().numpy()

    if len(feat) != len(c):
        raise RuntimeError(
            f"Utonia feature/input count mismatch: {len(feat)} vs {len(c)}"
        )
    return feat

for j in range(N):
    c = np.asarray(coord[j], np.float32)
    col = np.asarray(color[j], np.float32)
    nor = np.asarray(normal[j], np.float32)
    rou = np.asarray(rough[j], np.float32)
    r = np.asarray(rep[j], np.int64)

    feat = extract_features(c, col, nor)

    raw = np.concatenate(
        [
            feat[r],
            c[r],
            nor[r],
            col[r],
            rou[r],
        ],
        axis=1,
    )

    if mm is None:
        feat_dim = int(feat.shape[1])
        mm = np.lib.format.open_memmap(
            out_path,
            mode="w+",
            dtype=np.float16,
            shape=(N, rep_points, raw.shape[-1]),
        )

    mm[j] = raw.astype(np.float16)

    if (j + 1) % 20 == 0 or j == 0 or j == N - 1:
        mm.flush()
        print(f" UTONIA {j+1}/{N}")

mm.flush()

meta = {
    "model": "Pointcept/Utonia",
    "worker_python": sys.version,
    "worker_torch": torch.__version__,
    "worker_cuda": torch.version.cuda,
    "feature_dim": feat_dim,
    "shape": list(mm.shape),
    "flash_attention": False,
}
meta_path.write_text(json.dumps(meta, indent=2))

del model, mm
gc.collect()
torch.cuda.empty_cache()

print("Utonia worker complete:", out_path)
"""
    worker_path.write_text(worker_code)


def _prepare_utonia_staging(base, env, run, manifest):
    stage = WORK_ROOT / f"v7_utonia_stage_{env}_{run}"
    done = stage / "STAGING_COMPLETE.json"

    if done.exists():
        try:
            info = json.loads(done.read_text())
            if (
                int(info.get("N",-1)) == len(manifest["frame_ids"])
                and int(info.get("GEOM_INPUT_POINTS",-1)) == int(GEOM_INPUT_POINTS)
                and int(info.get("GEOM_REP_POINTS",-1)) == int(GEOM_REP_POINTS)
            ):
                print("Reusing validated Utonia geometry staging:", stage)
                return stage
        except Exception:
            pass

    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)

    N = len(manifest["frame_ids"])

    coord_mm = np.lib.format.open_memmap(
        stage / "coord.npy",
        mode="w+",
        dtype=np.float32,
        shape=(N, GEOM_INPUT_POINTS, 3),
    )
    color_mm = np.lib.format.open_memmap(
        stage / "color.npy",
        mode="w+",
        dtype=np.float32,
        shape=(N, GEOM_INPUT_POINTS, 3),
    )
    normal_mm = np.lib.format.open_memmap(
        stage / "normal.npy",
        mode="w+",
        dtype=np.float32,
        shape=(N, GEOM_INPUT_POINTS, 3),
    )
    rough_mm = np.lib.format.open_memmap(
        stage / "rough.npy",
        mode="w+",
        dtype=np.float32,
        shape=(N, GEOM_INPUT_POINTS, 1),
    )
    rep_mm = np.lib.format.open_memmap(
        stage / "rep.npy",
        mode="w+",
        dtype=np.int32,
        shape=(N, GEOM_REP_POINTS),
    )

    for j in range(N):
        fid, rgb, depth, mask, local, normals, *_ = get_frame_inputs(
            base, manifest, j
        )

        coord, color, normal, rough = select_geometry_points(
            rgb, mask, local, normals
        )

        if coord.shape != (GEOM_INPUT_POINTS, 3):
            raise RuntimeError(
                f"Geometry staging expected {(GEOM_INPUT_POINTS,3)}, "
                f"got {coord.shape} at frame {j}"
            )

        rep = farthest_point_indices(coord, GEOM_REP_POINTS)

        coord_mm[j] = coord.astype(np.float32)
        color_mm[j] = color.astype(np.float32)
        normal_mm[j] = normal.astype(np.float32)
        rough_mm[j] = rough.astype(np.float32)
        rep_mm[j] = rep.astype(np.int32)

        if (j + 1) % 20 == 0 or j == 0 or j == N - 1:
            for mm in [coord_mm, color_mm, normal_mm, rough_mm, rep_mm]:
                mm.flush()
            print(f" UTONIA STAGING {j+1}/{N}")

    for mm in [coord_mm, color_mm, normal_mm, rough_mm, rep_mm]:
        mm.flush()

    done.write_text(
        json.dumps(
            {
                "N": N,
                "GEOM_INPUT_POINTS": int(GEOM_INPUT_POINTS),
                "GEOM_REP_POINTS": int(GEOM_REP_POINTS),
            },
            indent=2,
        )
    )

    del coord_mm, color_mm, normal_mm, rough_mm, rep_mm
    gc.collect()

    return stage


def build_utonia_cache(base, env, run, manifest):
    path = CACHE_ROOT / f"{env}_{run}_v7_utonia.npy"
    meta = CACHE_ROOT / f"{env}_{run}_v7_utonia_meta.json"

    arr = _validated_visual_cache(
        path,meta,manifest,'Utonia',expected_model='Pointcept/Utonia',
        expected_ndim=3,expected_axes={1:GEOM_REP_POINTS}
    )
    if arr is not None:
        return arr

    if "UTONIA_PY" not in globals():
        raise RuntimeError(
            "UTONIA_PY is not defined. Run the fixed Python-3.13-safe "
            "installation cell first."
        )

    UTONIA_PY_PATH = Path(UTONIA_PY)
    if not UTONIA_PY_PATH.exists():
        raise RuntimeError(
            f"Isolated Utonia Python does not exist: {UTONIA_PY_PATH}"
        )

    stage = _prepare_utonia_staging(base, env, run, manifest)

    worker = EXT / "utonia_cache_worker.py"
    _write_utonia_worker(worker)

    # Release any main-process CUDA cache before the child Utonia process starts.
    gc.collect()
    torch.cuda.empty_cache()

    cmd = [
        str(UTONIA_PY_PATH),
        str(worker),
        str(stage),
        str(path),
        str(meta),
        str(GEOM_REP_POINTS),
    ]
    print("Launching isolated Utonia worker:")
    print(" ".join(cmd))
    child_env = os.environ.copy()
    # Optional device routing used by the 2-GPU held-out-run preparation.
    # Inside the isolated worker, the selected physical GPU becomes cuda:0.
    if "UTONIA_GPU_INDEX" in globals():
        child_env["CUDA_VISIBLE_DEVICES"] = str(int(UTONIA_GPU_INDEX))
    subprocess.check_call(cmd, env=child_env)

    if not path.exists() or not meta.exists():
        raise RuntimeError("Utonia worker returned without producing its cache.")

    arr = np.load(path, mmap_mode="r")
    if len(arr) != len(manifest["frame_ids"]):
        raise RuntimeError(
            f"Utonia cache length mismatch: {len(arr)} vs "
            f"{len(manifest['frame_ids'])}"
        )

    # Staging is no longer needed once the persistent feature cache exists.
    try:
        shutil.rmtree(stage)
    except Exception as e:
        print("Could not remove Utonia staging directory:", e)

    print("Utonia isolated-worker cache complete:", arr.shape)
    return arr


def load_waft():
    WAFT=EXT/'WAFT'
    if str(WAFT) not in sys.path: sys.path.insert(0,str(WAFT))
    old=os.getcwd(); os.chdir(WAFT)
    try:
        from config.parser import json_to_args
        from model import fetch_model
        from utils.utils import load_ckpt
        from inference_tools import InferenceWrapper
        args=json_to_args(str(WAFT/'config/a1/tar-c-t.json'))
        model=fetch_model(args); load_ckpt(model,str(WAFT/'ckpts/tar-c-t.pth'))
        model=model.to(DEVICE).eval()
        for p in model.parameters(): p.requires_grad=False
        wrapped=InferenceWrapper(model,scale=args.scale,train_size=args.image_size,pad_to_train_size=False,tiling=False)
    finally:
        os.chdir(old)
    return wrapped,args


def flow_confidence(info,args):
    raw_b=info[:,2:]
    log_b=torch.zeros_like(raw_b)
    w=info[:,:2].softmax(dim=1)
    log_b[:,0]=torch.clamp(raw_b[:,0],min=0,max=float(args.var_max))
    log_b[:,1]=torch.clamp(raw_b[:,1],min=float(args.var_min),max=0)
    uncertainty=(log_b*w).sum(dim=1,keepdim=True)
    return torch.exp(-uncertainty).clamp(0,1)


def flow_six_channels(flow,conf):
    # Normalize pixel flow by image scale. Spatial derivatives then become dimensionless-ish.
    u=flow[:,0:1]/max(CAM_H,CAM_W); v=flow[:,1:2]/max(CAM_H,CAM_W)
    mag=torch.sqrt(u*u+v*v+1e-12)
    du_dx=F.pad(u[:,:,:,2:]-u[:,:,:,:-2],(1,1,0,0))/2
    dv_dy=F.pad(v[:,:,2:,:]-v[:,:,:-2,:],(0,0,1,1))/2
    dv_dx=F.pad(v[:,:,:,2:]-v[:,:,:,:-2],(1,1,0,0))/2
    du_dy=F.pad(u[:,:,2:,:]-u[:,:,:-2,:],(0,0,1,1))/2
    div=du_dx+dv_dy; curl=dv_dx-du_dy
    return torch.cat([u,v,mag,div,curl,conf],dim=1)


def build_waft_cache(base,env,run,manifest):
    path=CACHE_ROOT/f'{env}_{run}_v7_waft.npy'; meta=CACHE_ROOT/f'{env}_{run}_v7_waft_meta.json'
    arr=_validated_visual_cache(
        path,meta,manifest,'WAFT',expected_model='WAFT a1 tar-c-t',
        expected_ndim=4,expected_axes={1:FLOW_HISTORY,2:RAW_TOKENS,3:6}
    )
    if arr is not None: return arr
    print('Loading WAFT a1...')
    model,args=load_waft(); N=len(manifest['frame_ids'])
    # pair_tokens[j] describes flow from frame j-1 -> j. frame 0 is zeros.
    pair=np.zeros((N,RAW_TOKENS,6),np.float16)
    for j in range(1,N):
        fid0=str(manifest['frame_ids'][j-1]); fid1=str(manifest['frame_ids'][j])
        im0=np.asarray(Image.open(base/'image_lcam_front'/f'{fid0}_lcam_front.png').convert('RGB'))
        im1=np.asarray(Image.open(base/'image_lcam_front'/f'{fid1}_lcam_front.png').convert('RGB'))
        x0=torch.from_numpy(im0).permute(2,0,1).float()[None].to(DEVICE)
        x1=torch.from_numpy(im1).permute(2,0,1).float()[None].to(DEVICE)
        with torch.inference_mode(): out=model.calc_flow(x0,x1)
        flow=out['flow'][-1]; info=out.get('info',[None])[-1]
        conf=flow_confidence(info,args) if info is not None else torch.ones_like(flow[:,0:1])
        six=flow_six_channels(flow,conf)
        # target frame's exact metric ground support
        _,_,_,mask,_,_,_,_=get_frame_inputs(base,manifest,j)
        bx=manifest['crop_boxes'][j]; xa,ya,xb,yb=[int(q) for q in bx]
        six=six[:,:,ya:yb,xa:xb]
        m=torch.from_numpy(mask[ya:yb,xa:xb].astype(np.float32))[None,None].to(DEVICE)
        q=weighted_pool(six,m).flatten(2).transpose(1,2)[0]
        # weighted_pool appends validity; six already has confidence, so remove the appended seventh channel
        q=q[:,:6].float().cpu().numpy().astype(np.float16)
        pair[j]=q
        if (j+1)%20==0: print(' WAFT',j+1,'/',N)
    # Causal temporal stacks: current visual frame j gets only flows ending <= j.
    mm=np.lib.format.open_memmap(path,mode='w+',dtype=np.float16,shape=(N,FLOW_HISTORY,RAW_TOKENS,6))
    for j in range(N):
        ids=np.arange(j-FLOW_HISTORY+1,j+1)
        for k,ii in enumerate(ids):
            if ii>=1: mm[j,k]=pair[ii]
            else: mm[j,k]=0
    mm.flush(); json.dump({'model':'WAFT a1 tar-c-t','shape':list(mm.shape)},open(meta,'w'))
    del model,mm,pair; gc.collect(); torch.cuda.empty_cache()
    return np.load(path,mmap_mode='r')


def load_visual_bundle(env,run,manifest):
    paths={
      'DINO':CACHE_ROOT/f'{env}_{run}_v7_dinov3.npy',
      'CONV':CACHE_ROOT/f'{env}_{run}_v7_convnext.npy',
      'GEOM':CACHE_ROOT/f'{env}_{run}_v7_utonia.npy',
      'FLOW':CACHE_ROOT/f'{env}_{run}_v7_waft.npy',
    }
    missing=[k for k,p in paths.items() if not p.exists()]
    if missing: raise RuntimeError('Mandatory V7 visual branches missing: '+repr(missing))
    out={k:np.load(p,mmap_mode='r') for k,p in paths.items()}
    out.update({k:manifest[k] for k in manifest.files} if hasattr(manifest,'files') else manifest)
    N=len(out['frame_ids'])
    expected={
        'DINO':(3,{1:RAW_TOKENS}),
        'CONV':(3,{1:RAW_TOKENS}),
        'GEOM':(3,{1:GEOM_REP_POINTS}),
        'FLOW':(4,{1:FLOW_HISTORY,2:RAW_TOKENS,3:6}),
    }
    for k in ['DINO','CONV','GEOM','FLOW']:
        arr=out[k]; ndim,axes=expected[k]
        bad=(len(arr)!=N or arr.ndim!=ndim)
        for ax,val in axes.items():
            bad = bad or (arr.shape[ax]!=val)
        if N:
            bad = bad or (not np.isfinite(np.asarray(arr[0],np.float32)).all())
            bad = bad or (not np.isfinite(np.asarray(arr[-1],np.float32)).all())
        if bad:
            raise RuntimeError(f'{k} cache integrity failure: shape={arr.shape}, N={N}')
    return out

print('All four visual cache builders defined. No branch has a fallback.')

# %% notebook cell 7
# ======================================================================
# 4B. OVERNIGHT SERVER PREFLIGHT — FAIL EARLY, BEFORE HOURS OF DATA WORK
# ======================================================================
# This does NOT change the model. It verifies the exact frozen backbones,
# isolated Utonia CUDA environment, WAFT checkpoint, both GPUs, and the
# persistent checkpoint filesystem before the expensive run cell starts.

assert torch.cuda.device_count() >= 2
assert_disk_headroom('deep preflight')

_preflight_fingerprint=dict(
    schema='V7_2X4080_PREFLIGHT_20260926_v2',
    python=sys.version.split()[0],torch=torch.__version__,cuda=str(torch.version.cuda),
    gpu0=torch.cuda.get_device_name(0),gpu1=torch.cuda.get_device_name(1),
    dino=DINO_MODEL,conv=CONV_MODEL,
    waft_ckpt_size=int(waft_ckpt.stat().st_size),
    da_ckpt_size=int(da_ckpt.stat().st_size),
)
_preflight_marker=RECOVERY_ROOT/'SERVER_DEEP_PREFLIGHT_OK.json'
_reuse=False
if _preflight_marker.exists():
    try:
        _reuse=(json.loads(_preflight_marker.read_text())==_preflight_fingerprint)
    except Exception:
        _reuse=False

if _reuse:
    print('Reusing matching successful deep preflight:',_preflight_marker)
else:
    print('Running one-time deep backbone preflight...')
    # DINOv3 ViT-L/16 exact features_only path at the real 448 input.
    _m=timm.create_model(DINO_MODEL,pretrained=True,features_only=True,out_indices=(-1,)).to(DEVICE).eval()
    with torch.inference_mode():
        _y=_m(torch.zeros(1,3,VISION_INPUT,VISION_INPUT,device=DEVICE))[-1]
    if _y.ndim!=4 or not torch.isfinite(_y).all():
        raise RuntimeError('DINOv3 deep preflight failed.')
    print(' DINOv3 OK:',tuple(_y.shape)); del _m,_y; gc.collect(); torch.cuda.empty_cache()

    # DINOv3 ConvNeXt-Large exact multi-scale features path.
    _m=timm.create_model(CONV_MODEL,pretrained=True,features_only=True,out_indices=(0,1,2,3)).to(DEVICE).eval()
    with torch.inference_mode():
        _ys=_m(torch.zeros(1,3,VISION_INPUT,VISION_INPUT,device=DEVICE))
    if len(_ys)!=4 or not all(torch.isfinite(_q).all() for _q in _ys):
        raise RuntimeError('ConvNeXt deep preflight failed.')
    print(' ConvNeXt OK:',[tuple(_q.shape) for _q in _ys]); del _m,_ys; gc.collect(); torch.cuda.empty_cache()

    # WAFT model construction/checkpoint load.
    _waft,_waft_args=load_waft()
    print(' WAFT load OK')
    del _waft,_waft_args; gc.collect(); torch.cuda.empty_cache()

    # Utonia: actually load pretrained PTv3 on physical GPU 1 in its isolated env.
    _uenvironment=os.environ.copy(); _uenvironment['CUDA_VISIBLE_DEVICES']='1'
    _ucode=(
        "import torch,utonia; "
        "cfg=dict(enc_patch_size=[1024 for _ in range(5)],enable_flash=False); "
        "m=utonia.load('utonia',repo_id='Pointcept/Utonia',custom_config=cfg).to('cuda').eval(); "
        "print('Utonia pretrained CUDA load OK',torch.cuda.get_device_name(0)); del m; torch.cuda.empty_cache()"
    )
    subprocess.check_call([str(UTONIA_PY),'-c',_ucode],env=_uenvironment)

    atomic_json_write(_preflight_marker,_preflight_fingerprint)
    print('DEEP PREFLIGHT PASSED AND PERSISTED:',_preflight_marker)

# Verify atomic torch checkpoint I/O too.
_ckprobe=RECOVERY_ROOT/'_atomic_torch_probe.pt'
atomic_torch_save({'x':torch.arange(8)},_ckprobe)
_q=torch.load(_ckprobe,map_location='cpu',weights_only=False)
if not torch.equal(_q['x'],torch.arange(8)):
    raise RuntimeError('Atomic torch checkpoint read-back failed.')
_ckprobe.unlink(missing_ok=True)
print('ATOMIC CHECKPOINT WRITE/READ TEST PASSED.')

# %% notebook cell 8
# ======================================================================
# 5. V5-STYLE CONTACT REGIMES + SOFT CAUSAL PHYSICS
# ======================================================================
def _force_norms(P):
    return np.linalg.norm(P[:,:,9:12], axis=-1)

def learn_global_regime_thresholds(run_records):
    positive = []
    all_classes_source = []
    for rec in run_records:
        t = rec['data']['time']
        first_tt = min(rec['ttests'])
        mask = t <= float(first_tt) + 1e-9
        mag = _force_norms(rec['data']['P'][mask]).reshape(-1)
        positive.append(mag[mag >= CONTACT_THRESHOLD])
        all_classes_source.append(mag)

    positive = np.concatenate(positive)
    if len(positive) < 100:
        raise RuntimeError('Not enough positive-contact force samples for regime bins.')

    q1, q2 = np.quantile(positive, [1/3, 2/3])
    # Ensure strictly increasing thresholds even in degenerate data.
    q1 = max(float(q1), CONTACT_THRESHOLD + 1e-3)
    q2 = max(float(q2), q1 + 1e-3)
    thresholds = np.array([CONTACT_THRESHOLD, q1, q2], dtype=np.float32)
    return thresholds

def force_regime_np(force_xyz, thresholds):
    """0=no, 1=light, 2=medium, 3=heavy."""
    mag = np.linalg.norm(force_xyz, axis=-1)
    out = np.zeros_like(mag, dtype=np.int64)
    out[mag >= thresholds[0]] = 1
    out[mag >= thresholds[1]] = 2
    out[mag >= thresholds[2]] = 3
    return out

def learn_regime_centers_and_weights(run_records, thresholds):
    mags = []
    classes = []
    for rec in run_records:
        data = rec['data']
        mask = data['time'] <= min(rec['ttests']) + 1e-9
        f = data['P'][mask,:,9:12]
        m = np.linalg.norm(f, axis=-1).reshape(-1)
        c = force_regime_np(f, thresholds).reshape(-1)
        mags.append(m); classes.append(c)
    mags = np.concatenate(mags); classes = np.concatenate(classes)

    centers = []
    counts = []
    for c in range(N_REGIMES):
        vals = mags[classes == c]
        counts.append(max(len(vals),1))
        centers.append(float(np.median(vals)) if len(vals) else 0.0)
    centers = np.asarray(centers, np.float32)

    # Inverse-sqrt frequency is gentler than full inverse frequency.
    counts = np.asarray(counts, np.float64)
    w = 1.0 / np.sqrt(counts / counts.sum() + 1e-12)
    w = w / w.mean()
    w = np.clip(w, 0.5, 3.0).astype(np.float32)
    return centers, w, counts.astype(np.int64)

def _ridge_fit(X, Y, ridge=RIDGE):
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    scale = np.sqrt((X*X).mean(axis=0) + 1e-8)
    # Do not scale the constant column if exactly constant.
    Xs = X / scale[None,:]
    A = Xs.T @ Xs + ridge * np.eye(Xs.shape[1])
    B = Xs.T @ Y
    W_scaled = np.linalg.solve(A, B)
    W = W_scaled / scale[:,None]
    return W.astype(np.float32)

def fit_causal_physics(data, start_time, end_time, thresholds=None):
    if thresholds is None:
        thresholds = REGIME_THRESHOLDS
    """
    Fits two causal, low-order physical proxies using ONLY [start_time,end_time].

    Inverse dynamics proxy, per leg:
      tau = Phi(q,dq,ddq,F,contact_regime) W
    where Phi contains ddq, dq, sin(q), cos(q), F, regime one-hot, 1.

    Newton proxy:
      body_acc = [sum(Fx,Fy,Fz),1] W_body
    The affine calibration absorbs unknown force frame / gravity convention.
    """
    t = data['time'].astype(np.float64)
    P = data['P'].astype(np.float32)
    B = data['B'].astype(np.float32)

    # Numerical joint acceleration. Clip extreme derivative spikes robustly.
    dq = P[:,:,3:6].astype(np.float64)
    ddq = np.gradient(dq, t, axis=0)
    finite = np.isfinite(ddq)
    if not finite.all():
        ddq[~finite] = 0.0
    lim = np.quantile(np.abs(ddq), 0.995, axis=0, keepdims=True) + 1e-6
    ddq = np.clip(ddq, -lim, lim).astype(np.float32)

    mask = (t >= float(start_time)) & (t <= float(end_time))
    idx = np.flatnonzero(mask)
    if len(idx) < 100:
        raise RuntimeError(f'Physics fit too short: {start_time:.2f}->{end_time:.2f}')

    invW = []
    for leg in range(4):
        ql = P[idx,leg,0:3]
        dql = P[idx,leg,3:6]
        taul = P[idx,leg,6:9]
        Fl = P[idx,leg,9:12]
        ddql = ddq[idx,leg]
        cls = force_regime_np(Fl, thresholds)
        onehot = np.eye(N_REGIMES, dtype=np.float32)[cls]

        # 3 + 3 + 3 + 3 + 3 + 4 + 1 = 20 features.
        X = np.concatenate([
            ddql, dql, np.sin(ql), np.cos(ql), Fl, onehot,
            np.ones((len(idx),1), np.float32)
        ], axis=1)
        invW.append(_ridge_fit(X, taul))  # (20,3)

    invW = np.stack(invW, axis=0).astype(np.float32)  # (4,20,3)

    sumF = P[idx,:,9:12].sum(axis=1)
    Xb = np.concatenate([sumF, np.ones((len(idx),1),np.float32)], axis=1)
    bodyW = _ridge_fit(Xb, B[idx,0:3])  # (4,3)

    return {
        'invW': invW,
        'bodyW': bodyW,
        'thresholds': np.asarray(thresholds,np.float32),
        'regime_centers': REGIME_CENTERS.copy(),
    }

def physics_tau_np(physics, q, dq, ddq, force, regime_soft_or_onehot):
    """q,dq,ddq,force: (4,3); regime: (4,4) -> tau (4,3)."""
    X = np.concatenate([
        ddq, dq, np.sin(q), np.cos(q), force,
        regime_soft_or_onehot,
        np.ones((4,1), np.float32)
    ], axis=1).astype(np.float32)
    return np.einsum('lf,lfj->lj', X, physics['invW']).astype(np.float32)

def physics_acc_np(physics, force):
    X = np.concatenate([force.sum(axis=0), np.array([1.0],np.float32)])
    return (X @ physics['bodyW']).astype(np.float32)


def initialize_regimes_and_physics(run_records):
    global REGIME_THRESHOLDS,REGIME_CENTERS,REGIME_CLASS_WEIGHTS,REGIME_COUNTS
    REGIME_THRESHOLDS=learn_global_regime_thresholds(run_records)
    REGIME_CENTERS,REGIME_CLASS_WEIGHTS,REGIME_COUNTS=learn_regime_centers_and_weights(run_records,REGIME_THRESHOLDS)
    for rec in run_records:
        rec['physics_prior_v7']=fit_causal_physics(rec['data'],float(rec['data']['time'][0]),float(min(rec['ttests'])))
    print('Regime thresholds:',REGIME_THRESHOLDS)
    print('Regime centers   :',REGIME_CENTERS)
    print('Regime counts    :',REGIME_COUNTS)

print('Physics definitions ready; thresholds will be learned after run preparation.')

# %% notebook cell 9
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


def synthetic_v7_architecture_smoke_test():
    # No foundation weights/data required: catches architecture/tensor wiring errors early.
    torch.manual_seed(7)
    B,T,Hn=2,5,3
    model=V7FullDynamics(T,Hn,dino_dim=33,conv_dim=47,geom_dim=29).cpu().eval()
    with torch.no_grad():
        out=model(
            torch.randn(B,4,T,12), torch.randn(B,T,12), torch.randn(B,T,6),
            torch.randn(B,4,T,3), torch.randn(B,4,T,3),
            torch.randn(B,RAW_TOKENS,33), torch.randn(B,RAW_TOKENS,47),
            torch.randn(B,GEOM_REP_POINTS,29), torch.randn(B,FLOW_HISTORY,RAW_TOKENS,6),
            torch.randn(B,4,6),
            torch.randn(B,Hn,36), torch.rand(B,Hn,1), torch.randn(B,Hn,PI_PHYS_DIM)
        )
    assert out['residual_p'].shape==(B,Hn,4,12)
    assert out['residual_b'].shape==(B,Hn,12)
    assert out['binary_contact_logits'].shape==(B,Hn,4)
    assert out['regime_logits'].shape==(B,Hn,4,N_REGIMES)
    assert torch.isfinite(out['residual_p']).all()
    print('SYNTHETIC V7 ARCHITECTURE SMOKE TEST PASSED')
    del model,out

synthetic_v7_architecture_smoke_test()


# %% notebook cell 10
# ======================================================================
# 7. V7 SAMPLE BUILDING, WINDOW GENERATORS, LOSSES, TRAINING
# ======================================================================

def visual_tensors(visual,vi):
    return dict(
        dino=tensor(np.asarray(visual['DINO'][vi],np.float32)[None]),
        conv=tensor(np.asarray(visual['CONV'][vi],np.float32)[None]),
        geom=tensor(np.asarray(visual['GEOM'][vi],np.float32)[None]),
        flow=tensor(np.asarray(visual['FLOW'][vi],np.float32)[None]),
    )


def make_v7_sample(data,visual,stats,physics,idx,history_steps,horizons,norm=None):
    if norm is None: Pn,Bn,Un,Ln,En=normalize_arrays(data,stats)
    else: Pn,Bn,Un,Ln,En=norm
    h0=idx-history_steps+1
    if h0<0: raise RuntimeError('History starts before run.')
    offs=horizon_offsets(data,horizons)
    if idx+max(offs)>=len(data['time']): raise RuntimeError('Future target outside run.')
    vi=int(visual['visual_index'][idx])
    if vi<0: raise RuntimeError('No causal visual frame.')
    if float(visual['camera_times'][vi])>float(data['time'][idx])+1e-9:
        raise RuntimeError('Future visual leakage detected.')

    curP=data['P'][idx].astype(np.float32); curB=data['B'][idx].astype(np.float32)
    # Foot-query kinematics from the official ANYmal-D URDF. Planned displacement
    # uses only the currently observed high-level twist command, never future sensors.
    foot_pos=ANYMAL_D_FK.foot_positions(curP[:,0:3])
    curUraw=data['U'][idx].astype(np.float32)
    planned_body_delta=np.tile((curUraw[:3]*FOOT_QUERY_PLAN_SEC)[None,:],(4,1)).astype(np.float32)
    foot_query_kin=np.concatenate([foot_pos,planned_body_delta],axis=-1).astype(np.float32)
    curF=curP[:,9:12]; cur_cls=force_regime_np(curF,REGIME_THRESHOLDS); cur_oh=np.eye(N_REGIMES,dtype=np.float32)[cur_cls]
    fscale=max(float(REGIME_THRESHOLDS[-1]),1.0)
    bp_list=[]; bb_list=[]; rp_list=[]; rb_list=[]; tc=[]; tr=[]; controls=[]; phys=[]
    tpraw=[]; tbraw=[]
    for h,off in zip(horizons,offs):
        ti=idx+off
        bp=curP.copy(); bp[:,0:3]=curP[:,0:3]+float(h)*curP[:,3:6]; bb=curB.copy()
        tp=data['P'][ti].astype(np.float32); tb=data['B'][ti].astype(np.float32)
        rp=(tp-bp)/stats['P_std'][None,:]; rb=(tb-bb)/stats['B_std']
        # recorded future controller signals are exogenous controls only
        us=Un[idx:ti]; ls=Ln[idx:ti]
        if len(us)==0: us=Un[idx:idx+1]; ls=Ln[idx:idx+1]
        controls.append(np.concatenate([us[-1],ls[-1].reshape(-1),us.mean(0),ls.mean(0).reshape(-1)]).astype(np.float32))
        qcv=bp[:,0:3]; dq0=curP[:,3:6]; ddq0=np.zeros((4,3),np.float32)
        tau_phys=physics_tau_np(physics,qcv,dq0,ddq0,curF,cur_oh); acc_phys=physics_acc_np(physics,curF)
        qn=((qcv-stats['P_mean'][0:3])/stats['P_std'][0:3]).reshape(-1)
        taun=((tau_phys-stats['P_mean'][6:9])/stats['P_std'][6:9]).reshape(-1)
        accn=(acc_phys-stats['B_mean'][0:3])/stats['B_std'][0:3]
        fn=np.linalg.norm(curF,axis=-1)/fscale
        ph=np.concatenate([qn,taun,accn,fn,cur_oh.reshape(-1)]).astype(np.float32)
        if len(ph)!=PI_PHYS_DIM: raise RuntimeError('physics feature dim mismatch')
        phys.append(ph)
        bp_list.append(bp); bb_list.append(bb); rp_list.append(rp); rb_list.append(rb)
        tc.append(data['contact'][ti].astype(np.float32)); tr.append(force_regime_np(tp[:,9:12],REGIME_THRESHOLDS))
        tpraw.append(tp); tbraw.append(tb)
    vis=visual_tensors(visual,vi)
    sample=dict(
        p_hist=tensor(Pn[h0:idx+1].transpose(1,0,2)[None]),b_hist=tensor(Bn[h0:idx+1][None]),u_hist=tensor(Un[h0:idx+1][None]),
        l_hist=tensor(Ln[h0:idx+1].transpose(1,0,2)[None]),e_hist=tensor(En[h0:idx+1].transpose(1,0,2)[None]),
        future_control_features=tensor(np.asarray(controls,np.float32)[None]),
        horizon_values=tensor((np.asarray(horizons,np.float32)[:,None]/1.0)[None]),
        physics_features=tensor(np.asarray(phys,np.float32)[None]),
        foot_query_kin=tensor(foot_query_kin[None]),
        target_residual_p=tensor(np.asarray(rp_list,np.float32)[None]),target_residual_b=tensor(np.asarray(rb_list,np.float32)[None]),
        target_contact=tensor(np.asarray(tc,np.float32)[None]),target_regime=torch.as_tensor(np.asarray(tr,np.int64)[None],dtype=torch.long,device=DEVICE),
        baseline_p_raw=np.asarray(bp_list,np.float32),baseline_b_raw=np.asarray(bb_list,np.float32),
        target_p_raw=np.asarray(tpraw,np.float32),target_b_raw=np.asarray(tbraw,np.float32),target_contact_raw=np.asarray(tc,np.float32),
        current_p_raw=curP,current_b_raw=curB,horizons_raw=np.asarray(horizons,np.float32),idx=int(idx),vi=vi
    ); sample.update(vis); return sample


def make_range_windows_v7(data,visual,start_target_time,end_target_time,context_sec,hop_sec,max_horizon):
    dt=float(np.median(np.diff(data['time']))); H=int(round(context_sec/dt))
    te=max(float(start_target_time),float(data['time'][0])+context_sec+max_horizon); rows=[]; seen=set()
    while te<=float(end_target_time)+1e-9:
        idx=nearest_index(data['time'],te-max_horizon); off=int(round(max_horizon/dt))
        if idx not in seen:
            seen.add(idx)
            if idx-H+1>=0 and idx+off<len(data['time']) and int(visual['visual_index'][idx])>=0:
                ate=float(data['time'][idx+off])
                if ate<=float(end_target_time)+.015: rows.append((idx,float(data['time'][idx]),ate))
        te+=hop_sec
    if not rows: raise RuntimeError('No prior windows.')
    return sorted(rows,key=lambda x:x[2]),H


def make_local_windows_v7(data,visual,ttest,context_sec,hop_sec,max_horizon,half_life):
    dt=float(np.median(np.diff(data['time']))); H=int(round(context_sec/dt)); tes=[]; te=float(ttest)
    while te-max_horizon-context_sec>=float(ttest)-TRAIN_LOOKBACK_SEC-1e-9: tes.append(te); te-=hop_sec
    rows=[]; seen=set(); off=int(round(max_horizon/dt))
    for desired in reversed(tes):
        idx=nearest_index(data['time'],desired-max_horizon)
        if idx in seen: continue
        seen.add(idx)
        if idx-H+1<0 or idx+off>=len(data['time']) or int(visual['visual_index'][idx])<0: continue
        ate=float(data['time'][idx+off])
        if ate>ttest+.015: continue
        age=max(0.,ttest-ate); w=.5**(age/half_life); rows.append((idx,w,float(data['time'][idx]),ate))
    if not rows: raise RuntimeError('No local windows.')
    rows.sort(key=lambda x:x[3]); wm=max(x[1] for x in rows); rows=[(i,w/wm,c,t) for i,w,c,t in rows]
    if not np.all(np.diff([r[3] for r in rows])>=-1e-10): raise RuntimeError('Chronology failure.')
    if not np.all(np.diff([r[1] for r in rows])>=-1e-10): raise RuntimeError('Recency-weight failure.')
    if rows[-1][3]>ttest+.015: raise RuntimeError('Post-test leakage in local adaptation.')
    return rows,H


def _torch_physics_tau_v7(physics,q,dq,ddq,force,regime_probs):
    ones=torch.ones((*q.shape[:-1],1),device=q.device,dtype=q.dtype)
    X=torch.cat([ddq,dq,torch.sin(q),torch.cos(q),force,regime_probs,ones],dim=-1)
    W=torch.as_tensor(physics['invW'],device=q.device,dtype=q.dtype)
    return torch.einsum('bhlf,lfj->bhlj',X,W)

def _torch_physics_acc_v7(physics,force):
    sf=force.sum(2); X=torch.cat([sf,torch.ones((*sf.shape[:-1],1),device=sf.device,dtype=sf.dtype)],-1)
    return X@torch.as_tensor(physics['bodyW'],device=sf.device,dtype=sf.dtype)


def v7_loss(out,s,stats,physics):
    pp,pb=out['residual_p'],out['residual_b']; tp,tb=s['target_residual_p'],s['target_residual_b']; H=pp.shape[1]
    hw=torch.linspace(1.0,1.5,H,device=pp.device) if H>1 else torch.ones(1,device=pp.device)
    def hub(a,b):
        e=F.smooth_l1_loss(a,b,reduction='none',beta=.5); dims=tuple(range(2,e.ndim)); return (e.mean(dims)*hw[None]).mean()
    lq=hub(pp[:,:,:,0:3],tp[:,:,:,0:3]); ldq=hub(pp[:,:,:,3:6],tp[:,:,:,3:6]); ltau=hub(pp[:,:,:,6:9],tp[:,:,:,6:9]); lf=hub(pp[:,:,:,9:12],tp[:,:,:,9:12]); lb=hub(pb,tb)
    lcontact=F.binary_cross_entropy_with_logits(out['binary_contact_logits'],s['target_contact'])
    weights=torch.as_tensor(REGIME_CLASS_WEIGHTS,device=pp.device,dtype=pp.dtype)
    lreg=F.cross_entropy(out['regime_logits'].reshape(-1,N_REGIMES),s['target_regime'].reshape(-1),weight=weights)
    Pstd=torch.as_tensor(stats['P_std'],device=pp.device,dtype=pp.dtype); Bstd=torch.as_tensor(stats['B_std'],device=pp.device,dtype=pp.dtype)
    bp=torch.as_tensor(s['baseline_p_raw'],device=pp.device,dtype=pp.dtype)[None]; bb=torch.as_tensor(s['baseline_b_raw'],device=pp.device,dtype=pp.dtype)[None]
    predP=bp+pp*Pstd.view(1,1,1,12); predB=bb+pb*Bstd.view(1,1,12)
    gtP=torch.as_tensor(s['target_p_raw'],device=pp.device,dtype=pp.dtype)[None]
    force=predP[:,:,:,9:12]; gtF=gtP[:,:,:,9:12]
    logmag=torch.log1p(torch.linalg.norm(gtF,dim=-1)); lmag=F.smooth_l1_loss(out['force_aux_logmag'],logmag,beta=.25)
    gt_dir=F.normalize(gtF,dim=-1,eps=1e-6); contact=s['target_contact']
    cos=(out['force_aux_dir']*gt_dir).sum(-1); ldir=((1-cos)*contact).sum()/(contact.sum()+1e-6)
    cur=torch.as_tensor(s['current_p_raw'],device=pp.device,dtype=pp.dtype); q0=cur[:,0:3][None,None]; dq0=cur[:,3:6][None,None]
    hs=torch.as_tensor(s['horizons_raw'],device=pp.device,dtype=pp.dtype).view(1,H,1,1)
    q1=predP[:,:,:,0:3]; dq1=predP[:,:,:,3:6]; tau1=predP[:,:,:,6:9]
    qphys=q0+.5*hs*(dq0+dq1)
    lkin=F.smooth_l1_loss((q1-qphys)/(Pstd[0:3].view(1,1,1,3)+1e-6),torch.zeros_like(q1),beta=.5)
    ddq=(dq1-dq0)/(hs+1e-6); taup=_torch_physics_tau_v7(physics,q1,dq1,ddq,force,out['regime_probs'])
    inv_res=(tau1-taup)/(Pstd[6:9].view(1,1,1,3)+1e-6)
    linv=F.smooth_l1_loss(inv_res,torch.zeros_like(inv_res),beta=.5)
    accp=_torch_physics_acc_v7(physics,force)
    new_res=(predB[:,:,:3]-accp)/(Bstd[:3].view(1,1,3)+1e-6)
    lnew=F.smooth_l1_loss(new_res,torch.zeros_like(new_res),beta=.5)
    p0=1-torch.sigmoid(out['binary_contact_logits']); lzero=(p0*torch.linalg.norm(force,dim=-1)/(float(REGIME_CENTERS[-1])+1e-6)).mean()
    # Long-horizon trend regularizer: first-difference residual dynamics, only when H>1.
    if H>1:
        d_pred=(predP[:,1:]-predP[:,:-1])/(Pstd.view(1,1,1,12)+1e-6)
        d_gt=(gtP[:,1:]-gtP[:,:-1])/(Pstd.view(1,1,1,12)+1e-6)
        ltrend=F.smooth_l1_loss(d_pred,d_gt,beta=.5)
    else: ltrend=pp.sum()*0
    data=W_Q*lq+W_DQ*ldq+W_TAU*ltau+W_FORCE*lf+W_BODY*lb
    total=(data+W_BINARY_CONTACT*lcontact+W_REGIME*lreg+W_FORCE_LOGMAG*lmag+W_FORCE_DIR*ldir+
           W_KIN*lkin+W_INVDYN*linv+W_NEWTON*lnew+W_ZERO_FORCE*lzero+W_TREND*ltrend)
    parts={k:float(v.detach().cpu()) for k,v in dict(data=data,q=lq,dq=ldq,tau=ltau,force=lf,contact=lcontact,regime=lreg,kin=lkin,dyn=linv,newton=lnew).items()}
    return total,parts


def model_forward_v7(model,s):
    return model(s['p_hist'],s['b_hist'],s['u_hist'],s['l_hist'],s['e_hist'],s['dino'],s['conv'],s['geom'],s['flow'],s['foot_query_kin'],s['future_control_features'],s['horizon_values'],s['physics_features'])


def v7_step(model,opt,s,stats,physics,weight=1.0):
    opt.zero_grad(set_to_none=True); out=model_forward_v7(model,s); raw,parts=v7_loss(out,s,stats,physics); loss=raw*float(weight)
    if not torch.isfinite(loss): raise RuntimeError('Non-finite training loss.')
    loss.backward()
    for n,p in model.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all(): raise RuntimeError('Non-finite gradient in '+n)
    torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP); opt.step(); return float(raw.detach().cpu()),parts


def make_model_for_record(rec,H,horizons):
    v=rec['visual']; return V7FullDynamics(H,len(horizons),v['DINO'].shape[-1],v['CONV'].shape[-1],v['GEOM'].shape[-1]).to(DEVICE)


def train_v7_prior(run_records,context_sec,hop_sec,horizons,epochs,seed,label,warm_ckpt=None):
    seed_everything_v7(seed)
    first=run_records[0]
    dt=float(np.median(np.diff(first['data']['time'])))
    H=int(round(context_sec/dt))
    model=make_model_for_record(first,H,horizons)
    if warm_ckpt is not None: partial_warmstart_from_v5(model,warm_ckpt)
    opt=torch.optim.AdamW(model.parameters(),lr=LR_PRIOR,weight_decay=WEIGHT_DECAY)

    # Fine-grained restart state: next epoch / trajectory / window plus all RNGs.
    ckpt_tag=f'{int(round(context_sec*1000))}ms_{len(horizons)}h'
    recovery_path=RECOVERY_ROOT/f'_RECOVERY_V7_PRIOR_{ckpt_tag}_v2.pt'
    signature=dict(
        schema='V7_SERVER_RESUME_20260926_v2',
        context_sec=float(context_sec),hop_sec=float(hop_sec),
        horizons=[float(x) for x in horizons],epochs=int(epochs),seed=int(seed),
        runs=[(r['environment'],r['run'],float(min(r['ttests']))) for r in run_records],
        lr=float(LR_PRIOR),weight_decay=float(WEIGHT_DECAY),
        checkpoint_every_windows=int(V7_CHECKPOINT_EVERY_WINDOWS),
        loss_weights=[W_Q,W_DQ,W_TAU,W_FORCE,W_BODY,W_BINARY_CONTACT,W_REGIME,
                      W_FORCE_LOGMAG,W_FORCE_DIR,W_KIN,W_INVDYN,W_NEWTON,W_ZERO_FORCE,W_TREND],
    )
    resume_ep=0; resume_ri=0; resume_wi=0; global_step=0
    if recovery_path.exists():
        try:
            ck=torch.load(recovery_path,map_location=DEVICE,weights_only=False)
            if ck.get('signature')==signature:
                model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
                resume_ep=int(ck.get('epoch',ck.get('completed_epochs',0)))
                resume_ri=int(ck.get('record_index',0)); resume_wi=int(ck.get('next_window_index',0))
                global_step=int(ck.get('global_step',0)); _restore_rng_state(ck.get('rng_state'))
                print(f'Resuming {label}: epoch={resume_ep}/{epochs}, record={resume_ri}/{len(run_records)}, next_window={resume_wi}, global_step={global_step}')
            else:
                print('Ignoring incompatible V7 recovery checkpoint:',recovery_path)
        except Exception as e:
            print('Ignoring unreadable V7 recovery checkpoint:',e)

    def _save_position(next_ep,next_ri,next_wi):
        assert_disk_headroom('V7 checkpoint')
        atomic_torch_save(dict(
            signature=signature,epoch=int(next_ep),record_index=int(next_ri),
            next_window_index=int(next_wi),global_step=int(global_step),
            model=model.state_dict(),optimizer=opt.state_dict(),rng_state=_capture_rng_state(),
        ),recovery_path)

    print('\n'+'='*100+'\nV7 GLOBAL PRIOR '+label+'\n'+'='*100)
    for ep in range(resume_ep,epochs):
        vals=[]
        for ri,rec in enumerate(run_records):
            if ep==resume_ep and ri<resume_ri:
                continue
            data,vis=rec['data'],rec['visual']; end=float(min(rec['ttests']))
            stats=make_stats_between(data,float(data['time'][0]),end)
            norm=normalize_arrays(data,stats); physics=rec['physics_prior_v7']
            wins,H2=make_range_windows_v7(data,vis,float(data['time'][0]),end,context_sec,hop_sec,max(horizons)); assert H2==H
            wi0=(resume_wi if (ep==resume_ep and ri==resume_ri) else 0)
            if wi0>len(wins): raise RuntimeError('Recovery next_window_index exceeds current window count.')
            print(f' ep {ep+1}/{epochs} {rec["terrain_type"]} windows={len(wins)} resume_at={wi0}')
            for widx in range(wi0,len(wins)):
                idx,_,te=wins[widx]
                samp=make_v7_sample(data,vis,stats,physics,idx,H,horizons,norm)
                v,parts=v7_step(model,opt,samp,stats,physics); vals.append(v); global_step+=1
                wi=widx+1
                if wi in (1,len(wins)) or wi%100==0:
                    print(f'  {wi:04d}/{len(wins):04d} target_end={te:.3f} loss={v:.4f} tau={parts["tau"]:.3f} force={parts["force"]:.3f}')
                del samp
                if (global_step % V7_CHECKPOINT_EVERY_WINDOWS)==0:
                    _save_position(ep,ri,wi)
            # Trajectory boundary is always durable. Advance position to next record/epoch.
            if ri+1 < len(run_records): _save_position(ep,ri+1,0)
            else: _save_position(ep+1,0,0)
        print(' mean raw loss over executed portion',float(np.mean(vals)) if vals else float('nan'))
        resume_ri=0; resume_wi=0
    print('V7 prior complete. Durable recovery checkpoint:',recovery_path)
    return model

def adapt_v7(prior,data,vis,ttest,context_sec,hop_sec,horizons,half_life,epochs,seed,label):
    seed_everything_v7(seed); model=copy.deepcopy(prior).to(DEVICE); start=max(float(data['time'][0]),ttest-TRAIN_LOOKBACK_SEC)
    stats=make_local_stats(data,ttest); physics=fit_causal_physics(data,start,ttest,REGIME_THRESHOLDS); norm=normalize_arrays(data,stats)
    wins,H=make_local_windows_v7(data,vis,ttest,context_sec,hop_sec,max(horizons),half_life)
    if H!=model.history_steps: raise RuntimeError('History mismatch.')
    opt=torch.optim.AdamW(model.parameters(),lr=LR_LOCAL,weight_decay=WEIGHT_DECAY)
    print(f' V7 LOCAL {label} t={ttest:.1f}: windows={len(wins)} oldest_w={wins[0][1]:.4f} newest_w={wins[-1][1]:.4f}')
    for ep in range(epochs):
        vals=[]
        for wi,(idx,w,_,te) in enumerate(wins,1):
            s=make_v7_sample(data,vis,stats,physics,idx,H,horizons,norm); v,p=v7_step(model,opt,s,stats,physics,w); vals.append(v); del s
        print(f'  epoch {ep+1}/{epochs} mean={np.mean(vals):.5f}')
    return model,stats,physics,H

print('V7 sample/training/loss pipeline ready.')

# %% notebook cell 11
# ======================================================================
# 8. PREDICTION + COMPLETE EVALUATION
# ======================================================================

def predict_v7_raw(model,s,stats):
    model.eval()
    with torch.inference_mode(): out=model_forward_v7(model,s)
    rp=out['residual_p'][0].cpu().numpy(); rb=out['residual_b'][0].cpu().numpy()
    predP=s['baseline_p_raw']+rp*stats['P_std'][None,None,:]
    predB=s['baseline_b_raw']+rb*stats['B_std'][None,:]
    cp=torch.sigmoid(out['binary_contact_logits'][0]).cpu().numpy()
    reg=out['regime_probs'][0].cpu().numpy()
    auxmag=out['force_aux_logmag'][0].cpu().numpy(); auxdir=out['force_aux_dir'][0].cpu().numpy()
    return predP,predB,cp,reg,auxmag,auxdir

LEG_NAMES=['FL','FR','RL','RR']
P_NAMES=['q_HAA','q_HFE','q_KFE','dq_HAA','dq_HFE','dq_KFE','tau_HAA','tau_HFE','tau_KFE','Fx','Fy','Fz']
B_NAMES=['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z','body_vx','body_vy','body_vz','body_wx','body_wy','body_wz']

def signal_group_v7(s):
    if 'contact_probability' in s:return 'contact'
    if '.q_' in s:return 'q'
    if '.dq_' in s:return 'dq'
    if '.tau_' in s:return 'tau'
    if s.endswith(('.Fx','.Fy','.Fz')):return 'force'
    if s.startswith('acc_'):return 'acc'
    if s.startswith('gyro_'):return 'gyro'
    if s.startswith('body_v'):return 'body_linear_velocity'
    if s.startswith('body_w'):return 'body_angular_velocity'
    return 'other'


def evaluate_v7_case(short_model,short_stats,short_phys,Hs,long_model,long_stats,long_phys,Hl,data,vis,ttest):
    idx=nearest_index(data['time'],ttest); sn=normalize_arrays(data,short_stats); ln=normalize_arrays(data,long_stats)
    ss=make_v7_sample(data,vis,short_stats,short_phys,idx,Hs,SHORT_HORIZONS,sn)
    ls=make_v7_sample(data,vis,long_stats,long_phys,idx,Hl,LONG_HORIZONS,ln)
    sp,sb,sc,sreg,smag,sdir=predict_v7_raw(short_model,ss,short_stats)
    lp,lb,lc,lreg,lmag,ldir=predict_v7_raw(long_model,ls,long_stats)
    rows=[]; regime_rows=[]; force_rows=[]
    for h in ALL_HORIZONS:
        if abs(h-.1)<1e-8:
            hi=0; predP,predB,predC,reg=sp[0],sb[0],sc[0],sreg[0]; samp=ss; src='SHORT'
            gtP,gtB,gtC=samp['target_p_raw'][0],samp['target_b_raw'][0],samp['target_contact_raw'][0]; bp,bb=samp['baseline_p_raw'][0],samp['baseline_b_raw'][0]
            longP=None; longB=None; longC=None
        else:
            hi=LONG_HORIZONS.index(round(h,1)); predP,predB,predC,reg=lp[hi],lb[hi],lc[hi],lreg[hi]; samp=ls; src='LONG'
            gtP,gtB,gtC=samp['target_p_raw'][hi],samp['target_b_raw'][hi],samp['target_contact_raw'][hi]; bp,bb=samp['baseline_p_raw'][hi],samp['baseline_b_raw'][hi]
            longP,longB,longC=predP,predB,predC
        bc=data['contact'][idx].astype(np.float32)
        for li,L in enumerate(LEG_NAMES):
            for fi,n in enumerate(P_NAMES):
                pred=float(predP[li,fi]); gt=float(gtP[li,fi]); base=float(bp[li,fi])
                rows.append(dict(horizon_s=h,expert_used=src,signal=f'{L}.{n}',baseline_pred=base,
                    short_native_pred=float(sp[0,li,fi]) if abs(h-.1)<1e-8 else np.nan,
                    long_direct_pred=float(longP[li,fi]) if longP is not None else np.nan,
                    prediction=pred,ground_truth=gt,baseline_abs_error=abs(base-gt),signed_error=pred-gt,abs_error=abs(pred-gt)))
            pred=float(predC[li]); gt=float(gtC[li]); base=float(bc[li])
            rows.append(dict(horizon_s=h,expert_used=src,signal=f'{L}.contact_probability',baseline_pred=base,
                    short_native_pred=float(sc[0,li]) if abs(h-.1)<1e-8 else np.nan,
                    long_direct_pred=float(longC[li]) if longC is not None else np.nan,prediction=pred,ground_truth=gt,
                    baseline_abs_error=abs(base-gt),signed_error=pred-gt,abs_error=abs(pred-gt)))
            true_reg=int(force_regime_np(gtP[li,9:12][None,:],REGIME_THRESHOLDS)[0]); pred_reg=int(np.argmax(reg[li]))
            regime_rows.append(dict(horizon_s=h,leg=L,pred_regime=pred_reg,true_regime=true_reg,correct=int(pred_reg==true_reg)))
            pf=predP[li,9:12]; gf=gtP[li,9:12]; pm=float(np.linalg.norm(pf)); gm=float(np.linalg.norm(gf));
            cos=float(np.dot(pf,gf)/(pm*gm+1e-9)) if gm>=CONTACT_THRESHOLD and pm>1e-6 else np.nan
            force_rows.append(dict(horizon_s=h,leg=L,pred_force_mag=pm,true_force_mag=gm,force_mag_abs_error=abs(pm-gm),force_vector_error=float(np.linalg.norm(pf-gf)),force_direction_cosine=cos))
        for bi,n in enumerate(B_NAMES):
            pred=float(predB[bi]); gt=float(gtB[bi]); base=float(bb[bi])
            rows.append(dict(horizon_s=h,expert_used=src,signal=n,baseline_pred=base,short_native_pred=float(sb[0,bi]) if abs(h-.1)<1e-8 else np.nan,
                long_direct_pred=float(longB[bi]) if longB is not None else np.nan,prediction=pred,ground_truth=gt,baseline_abs_error=abs(base-gt),signed_error=pred-gt,abs_error=abs(pred-gt)))
    return pd.DataFrame(rows),pd.DataFrame(regime_rows),pd.DataFrame(force_rows),idx,int(vis['visual_index'][idx])


def summarize_metrics(all_df):
    d=all_df.copy(); d['signal_group']=d.signal.map(signal_group_v7); d['sq_error']=d.signed_error**2
    grouped=d.groupby(['terrain_type','environment','run','test_id','ttest_s','horizon_s','signal_group'],as_index=False).agg(
        baseline_MAE=('baseline_abs_error','mean'),MAE=('abs_error','mean'),RMSE=('sq_error',lambda x:float(np.sqrt(np.mean(x)))))
    # NRMSE uses within-test truth std by group/horizon; keep unit groups separate.
    std=d.groupby(['terrain_type','test_id','ttest_s','horizon_s','signal_group']).ground_truth.std().reset_index(name='truth_std')
    grouped=grouped.merge(std,on=['terrain_type','test_id','ttest_s','horizon_s','signal_group'],how='left'); grouped['NRMSE']=grouped.RMSE/(grouped.truth_std.abs()+1e-6)
    mean=grouped.groupby(['horizon_s','signal_group'],as_index=False)[['baseline_MAE','MAE','RMSE','NRMSE']].mean()
    return grouped,mean


def contact_metrics(all_df):
    d=all_df[all_df.signal.str.contains('contact_probability')].copy(); out=[]
    for h,g in d.groupby('horizon_s'):
        y=g.ground_truth.to_numpy(int); p=np.clip(g.prediction.to_numpy(float),0,1); yh=(p>=.5).astype(int)
        try: auc=roc_auc_score(y,p) if len(np.unique(y))>1 else np.nan
        except: auc=np.nan
        out.append(dict(horizon_s=h,accuracy=accuracy_score(y,yh),precision=precision_score(y,yh,zero_division=0),recall=recall_score(y,yh,zero_division=0),F1=f1_score(y,yh,zero_division=0),Brier=brier_score_loss(y,p),AUROC=auc))
    return pd.DataFrame(out)


def trend_metrics(all_df):
    rows=[]
    keys=['terrain_type','test_id','signal']
    for key,g in all_df[all_df.horizon_s>=.2].groupby(keys):
        g=g.sort_values('horizon_s'); p=g.prediction.to_numpy(float); y=g.ground_truth.to_numpy(float)
        if len(p)<3: continue
        corr=np.corrcoef(p,y)[0,1] if np.std(p)>1e-9 and np.std(y)>1e-9 else np.nan
        sign=float(np.mean(np.sign(np.diff(p))==np.sign(np.diff(y))))
        rows.append(dict(terrain_type=key[0],test_id=key[1],signal=key[2],correlation=corr,first_difference_sign_accuracy=sign))
    return pd.DataFrame(rows)


def acceptance_table(mean):
    rows=[]
    for h in [0.1,1.0]:
        for group,ref in BEST_EXISTING[h].items():
            q=mean[(np.isclose(mean.horizon_s,h))&(mean.signal_group==group)]
            val=float(q.MAE.iloc[0]) if len(q) else np.nan
            rows.append(dict(horizon_s=h,group=group,V7_MAE=val,best_existing_V5_MAE=ref,percent_change=100*(val-ref)/(ref+1e-12),PASS=bool(val<ref)))
    return pd.DataFrame(rows)

print('Evaluation functions ready.')

# %% notebook cell 12
# ======================================================================
# 9. END-TO-END RUN: DATA -> 4-BRANCH CACHE -> PRIORS -> 16 TESTS
# This is the expensive cell.
# ======================================================================
seed_everything_v7(SEED)

# A) Prepare the expanded BALANCED training set.
#    The four original diagnostic trajectories retain their exact original ttests
#    so the original 16-test protocol remains unchanged.
#    Auxiliary training trajectories use almost the entire run for global training.

_diag_map={(e['environment'],e['run']):e for e in EXPERIMENTS}
TRAIN_RECORDS=[]

def _v7_bundle_paths(env,run):
    return [
        CACHE_ROOT/f'{env}_{run}_physics.npz',
        CACHE_ROOT/f'{env}_{run}_frame_manifest.npz',
        CACHE_ROOT/f'{env}_{run}_v7_dinov3.npy',
        CACHE_ROOT/f'{env}_{run}_v7_dinov3_meta.json',
        CACHE_ROOT/f'{env}_{run}_v7_convnext.npy',
        CACHE_ROOT/f'{env}_{run}_v7_convnext_meta.json',
        CACHE_ROOT/f'{env}_{run}_v7_utonia.npy',
        CACHE_ROOT/f'{env}_{run}_v7_utonia_meta.json',
        CACHE_ROOT/f'{env}_{run}_v7_waft.npy',
        CACHE_ROOT/f'{env}_{run}_v7_waft_meta.json',
    ]

def v7_persistent_bundle_complete(env,run):
    paths=_v7_bundle_paths(env,run)
    if not all(p.exists() for p in paths):
        return False
    # Reject caches made with different mandatory backbone identities.
    expected=[
        (CACHE_ROOT/f'{env}_{run}_v7_dinov3_meta.json',DINO_MODEL),
        (CACHE_ROOT/f'{env}_{run}_v7_convnext_meta.json',CONV_MODEL),
        (CACHE_ROOT/f'{env}_{run}_v7_utonia_meta.json','Pointcept/Utonia'),
        (CACHE_ROOT/f'{env}_{run}_v7_waft_meta.json','WAFT a1 tar-c-t'),
    ]
    try:
        for mp,model_name in expected:
            info=json.loads(mp.read_text())
            if info.get('model')!=model_name:
                return False
        return True
    except Exception:
        return False

def prepare_v7_record_data(env,run,require_raw=False):
    """Load a complete persistent bundle without redownloading, or safely build it."""
    base=ROOT/env/'Data_anymal'/run

    # Auxiliary runs never need their raw files after all persistent products exist.
    if (not require_raw) and v7_persistent_bundle_complete(env,run):
        try:
            data=extract_run_physics(base,run)
            manifest=build_frame_manifest(base,env,run,data)
            visual=load_visual_bundle(env,run,manifest)
            print('Persistent bundle complete; raw download not required:',env,run)
            return base,data,manifest,visual
        except Exception as e:
            print('Persistent bundle failed validation; raw data will be restored:',e)

    # Diagnostic runs require RGB/depth later for the audit montage and V8 cells.
    # Incomplete/stale auxiliary bundles also come through this path.
    base=download_run_v7(env,run)
    data=extract_run_physics(base,run)
    manifest=build_frame_manifest(base,env,run,data)

    print('\n'+'='*100+'\nVISUAL CACHE '+env+'/'+run+'\n'+'='*100)
    build_dinov3_cache(base,env,run,manifest)
    build_convnext_cache(base,env,run,manifest)
    build_utonia_cache(base,env,run,manifest)
    build_waft_cache(base,env,run,manifest)
    visual=load_visual_bundle(env,run,manifest)
    assert all(k in visual for k in ('DINO','CONV','GEOM','FLOW'))
    return base,data,manifest,visual

def _pose_audit_from_poses(poses):
    """Cheap prefilter: reject trajectories with essentially no vertical motion."""
    poses=np.asarray(poses,np.float64)
    if poses.ndim!=2 or poses.shape[1]<3 or len(poses)<3:
        raise RuntimeError('Stair pose audit requires >=3 valid camera poses.')
    elevation=-poses[:,2]  # world NED -> elevation
    elevation=elevation[np.isfinite(elevation)]
    if len(elevation)<3:
        raise RuntimeError('Too few finite poses for stair audit.')
    p05,p95=np.percentile(elevation,[5.0,95.0])
    robust_span=float(p95-p05)
    total_vertical=float(np.abs(np.diff(elevation)).sum())
    net_vertical=float(elevation[-1]-elevation[0])
    return dict(
        robust_vertical_span_m=robust_span,
        total_vertical_travel_m=total_vertical,
        net_vertical_change_m=net_vertical,
        pose_pass=bool(robust_span>=STAIR_AUDIT_MIN_ROBUST_VERTICAL_SPAN_M),
    )


def _download_stair_meta_only(env,run):
    """Download only tiny pose/meta data first; do not spend depth bandwidth on flat runs."""
    base=ROOT/env/'Data_anymal'/run
    pose_path=base/'pose_lcam_front.txt'
    if not pose_path.exists():
        ta.init(str(ROOT))
        print('STAIR DISCOVERY meta-only download:',env,run)
        ta.download_ground(
            env=[env],version=['anymal'],traj=[run],
            modality=['meta'],camera_name=['lcam_front'],unzip=True
        )
    if not pose_path.exists():
        raise RuntimeError(f'Missing ANYmal front pose after metadata download: {env}/{run}')
    return base


def _download_stair_depth(env,run):
    """Download only front depth after the pose prefilter passes."""
    base=ROOT/env/'Data_anymal'/run
    depth_dir=base/'depth_lcam_front'
    if not depth_dir.exists() or not any(depth_dir.glob('*_lcam_front_depth.png')):
        ta.init(str(ROOT))
        print('STAIR DISCOVERY depth download:',env,run)
        ta.download_ground(
            env=[env],version=['anymal'],traj=[run],
            modality=['depth'],camera_name=['lcam_front'],unzip=True
        )
    if not depth_dir.exists() or not any(depth_dir.glob('*_lcam_front_depth.png')):
        raise RuntimeError(f'Missing ANYmal front depth after download: {env}/{run}')
    return base


def _longest_consecutive_indices(mask):
    idx=np.flatnonzero(mask)
    if len(idx)==0:
        return np.empty(0,np.int64)
    cuts=np.where(np.diff(idx)>1)[0]+1
    groups=np.split(idx,cuts)
    return max(groups,key=len)


def _stair_frame_geometry_metrics(depth,pose7):
    """Detect repeated tread/riser structure in a gravity-aligned depth profile.

    A smooth ramp produces a distributed non-zero slope. A staircase produces long
    near-flat tread regions separated by a small number of discrete same-direction
    height jumps. The test therefore requires repeated plausible jumps + flat treads
    + regular spacing + jump-dominated vertical change.
    """
    valid=np.isfinite(depth)&(depth>=MIN_DEPTH_M)&(depth<=MAX_DEPTH_M)
    pc=dense_camera_points(depth)
    nc=dense_normals_camera(pc,valid)
    Rlw,Rwc=heading_frame_from_pose(pose7)
    world=np.einsum('ij,hwj->hwi',Rwc,pc)
    local=np.einsum('ij,hwj->hwi',Rlw,world)
    nw=np.einsum('ij,hwj->hwi',Rwc,nc)
    nl=np.einsum('ij,hwj->hwi',Rlw,nw)
    x,y,z=local[...,0],local[...,1],local[...,2]  # z is UP
    horiz=np.abs(nl[...,2])>=STAIR_GEOM_MIN_HORIZONTAL_NORMAL_Z
    sel=(valid & horiz &
         (x>=STAIR_GEOM_X_RANGE_M[0])&(x<=STAIR_GEOM_X_RANGE_M[1]) &
         (np.abs(y)<=STAIR_GEOM_Y_ABS_M) &
         (z>=GROUND_Z_RANGE[0])&(z<=max(1.0,GROUND_Z_RANGE[1])))
    if int(sel.sum())<300:
        return dict(frame_pass=False,reason='too_few_horizontal_points',step_groups=0)

    xv=x[sel].astype(np.float64); zv=z[sel].astype(np.float64)
    bw=float(STAIR_GEOM_BIN_M)
    edges=np.arange(STAIR_GEOM_X_RANGE_M[0],STAIR_GEOM_X_RANGE_M[1]+bw,bw)
    ids=np.digitize(xv,edges)-1
    nb=max(0,len(edges)-1)
    med=np.full(nb,np.nan,np.float64); cnt=np.zeros(nb,np.int32)
    for bi in range(nb):
        zz=zv[ids==bi]
        cnt[bi]=len(zz)
        if len(zz)>=STAIR_GEOM_MIN_POINTS_PER_BIN:
            med[bi]=float(np.median(zz))
    good=np.isfinite(med)
    seg=_longest_consecutive_indices(good)
    if len(seg)<10:
        return dict(frame_pass=False,reason='insufficient_contiguous_profile',step_groups=0)
    zz=med[seg]
    xx=(edges[:-1]+0.5*bw)[seg]

    # 3-bin median denoise without turning a step profile into a slope.
    if len(zz)>=3:
        zs=zz.copy()
        for i in range(1,len(zz)-1):
            zs[i]=np.median(zz[i-1:i+2])
    else:
        zs=zz
    dz=np.diff(zs)
    adz=np.abs(dz)
    profile_span=float(np.nanpercentile(zs,95)-np.nanpercentile(zs,5))
    flat_fraction=float(np.mean(adz<=0.025)) if len(adz) else 0.0

    # Merge adjacent bins belonging to the same physical riser.
    raw_jump=np.flatnonzero((adz>=STAIR_GEOM_MIN_RISE_M)&(adz<=STAIR_GEOM_MAX_RISE_M))
    groups=[]
    if len(raw_jump):
        for g in np.split(raw_jump,np.where(np.diff(raw_jump)>1)[0]+1):
            if len(g):
                rise=float(dz[g].sum())
                if STAIR_GEOM_MIN_RISE_M<=abs(rise)<=STAIR_GEOM_MAX_RISE_M:
                    groups.append((int(g[0]),int(g[-1]),rise))

    rises=np.asarray([g[2] for g in groups],np.float64)
    centers=np.asarray([0.5*(xx[g[0]]+xx[min(g[1]+1,len(xx)-1)]) for g in groups],np.float64)
    nsteps=int(len(rises))
    if nsteps:
        sign_consistency=float(max(np.mean(rises>0),np.mean(rises<0)))
        rise_abs=np.abs(rises)
        rise_cv=float(np.std(rise_abs)/(np.mean(rise_abs)+1e-9))
    else:
        sign_consistency=0.0; rise_cv=float('inf')
    if len(centers)>=2:
        spac=np.diff(centers)
        spacing_fraction=float(np.mean(
            (spac>=STAIR_GEOM_MIN_TREAD_SPACING_M)&(spac<=STAIR_GEOM_MAX_TREAD_SPACING_M)
        ))
        median_spacing=float(np.median(spac))
    else:
        spacing_fraction=0.0; median_spacing=float('nan')

    total_variation=float(adz.sum())
    jump_energy=float(np.abs(rises).sum())
    jump_energy_ratio=float(jump_energy/(total_variation+1e-9))

    # Smooth ramps fit a straight line extremely well; stair profiles retain a
    # characteristic saw/plateau residual. This is a supporting, not sole, gate.
    try:
        coef=np.polyfit(xx,zs,1)
        lin=np.polyval(coef,xx)
        linear_rmse=float(np.sqrt(np.mean((zs-lin)**2)))
    except Exception:
        linear_rmse=0.0

    frame_pass=bool(
        nsteps>=STAIR_GEOM_MIN_STEP_GROUPS and
        profile_span>=STAIR_GEOM_MIN_PROFILE_SPAN_M and
        flat_fraction>=STAIR_GEOM_MIN_FLAT_FRACTION and
        jump_energy_ratio>=STAIR_GEOM_MIN_JUMP_ENERGY_RATIO and
        sign_consistency>=STAIR_GEOM_MIN_SIGN_CONSISTENCY and
        rise_cv<=STAIR_GEOM_MAX_RISE_CV and
        spacing_fraction>=0.60 and
        (linear_rmse>=0.012 or jump_energy_ratio>=0.65)
    )
    return dict(
        frame_pass=frame_pass,
        reason='pass' if frame_pass else 'geometry_thresholds',
        step_groups=nsteps,
        profile_span_m=profile_span,
        flat_fraction=flat_fraction,
        jump_energy_ratio=jump_energy_ratio,
        sign_consistency=sign_consistency,
        rise_cv=rise_cv,
        median_rise_m=float(np.median(np.abs(rises))) if nsteps else np.nan,
        median_tread_spacing_m=median_spacing,
        spacing_fraction=spacing_fraction,
        linear_ramp_fit_rmse_m=linear_rmse,
    )


def _stair_candidate_audit(env,run):
    """Two-stage true-stair audit: pose motion first, then repeated-step depth geometry."""
    base=_download_stair_meta_only(env,run)
    poses=np.loadtxt(base/'pose_lcam_front.txt').astype(np.float64)
    pose_audit=_pose_audit_from_poses(poses)
    result=dict(environment=env,run=run,**pose_audit)
    if not pose_audit['pose_pass']:
        result.update(
            depth_geometry_pass=False,
            true_stair_pass=False,
            rejection_reason='insufficient_vertical_motion',
            geometry_pass_frames=0,
        )
        return result

    _download_stair_depth(env,run)
    depth_dir=base/'depth_lcam_front'
    depth_files={
        int(p.name.split('_')[0]):p
        for p in depth_dir.glob('*_lcam_front_depth.png')
        if p.name.split('_')[0].isdigit()
    }
    if not depth_files:
        raise RuntimeError(f'No indexed depth frames for {env}/{run}')
    max_frame=min(max(depth_files),len(poses)-1)
    elev=-poses[:max_frame+1,2]
    half=int(STAIR_GEOM_EVENT_HALF_WINDOW_FRAMES)
    local_span=np.zeros(len(elev),np.float64)
    for i in range(len(elev)):
        a=max(0,i-half); b=min(len(elev),i+half+1)
        ee=elev[a:b]
        local_span[i]=float(np.nanmax(ee)-np.nanmin(ee)) if len(ee) else 0.0

    candidate_idx=[
        i for i in sorted(depth_files)
        if i<=max_frame and local_span[i]>=STAIR_GEOM_MIN_EVENT_VERTICAL_SPAN_M
    ]
    candidate_idx=candidate_idx[::max(1,int(STAIR_GEOM_FRAME_STRIDE))]
    if len(candidate_idx)>STAIR_GEOM_MAX_FRAMES_PER_RUN:
        # Evenly retain a bounded set around the vertical-motion events.
        keep=np.linspace(0,len(candidate_idx)-1,STAIR_GEOM_MAX_FRAMES_PER_RUN).round().astype(int)
        candidate_idx=[candidate_idx[i] for i in np.unique(keep)]

    frame_metrics=[]
    pass_idx=[]
    for i in candidate_idx:
        try:
            depth=load_metric_depth(depth_files[i])
            m=_stair_frame_geometry_metrics(depth,poses[i])
        except Exception as e:
            m=dict(frame_pass=False,reason='frame_error:'+repr(e),step_groups=0)
        m=dict(frame_index=int(i),event_vertical_span_m=float(local_span[i]),**m)
        frame_metrics.append(m)
        if m.get('frame_pass',False):
            pass_idx.append(i)

    # Require repeated evidence, not one lucky noisy frame.
    max_consecutive=0; cur=0; prev=None
    max_gap=max(2,2*int(STAIR_GEOM_FRAME_STRIDE)+1)
    for i in pass_idx:
        if prev is not None and i-prev<=max_gap:
            cur+=1
        else:
            cur=1
        max_consecutive=max(max_consecutive,cur); prev=i

    geom_pass=bool(
        len(pass_idx)>=STAIR_GEOM_MIN_PASS_FRAMES and
        max_consecutive>=2
    )
    best=max(frame_metrics,key=lambda x:(int(x.get('frame_pass',False)),x.get('step_groups',0),x.get('jump_energy_ratio',0.0)),default={})
    result.update(
        depth_geometry_pass=geom_pass,
        true_stair_pass=bool(pose_audit['pose_pass'] and geom_pass),
        rejection_reason='pass' if geom_pass else 'no_repeated_step_geometry',
        audited_geometry_frames=len(frame_metrics),
        geometry_pass_frames=len(pass_idx),
        max_consecutive_geometry_pass_frames=int(max_consecutive),
        max_event_vertical_span_m=float(max(local_span) if len(local_span) else 0.0),
        best_step_groups=int(best.get('step_groups',0)),
        best_profile_span_m=float(best.get('profile_span_m',np.nan)),
        best_flat_fraction=float(best.get('flat_fraction',np.nan)),
        best_jump_energy_ratio=float(best.get('jump_energy_ratio',np.nan)),
        best_sign_consistency=float(best.get('sign_consistency',np.nan)),
        best_median_rise_m=float(best.get('median_rise_m',np.nan)),
        best_median_tread_spacing_m=float(best.get('median_tread_spacing_m',np.nan)),
        best_linear_ramp_fit_rmse_m=float(best.get('linear_ramp_fit_rmse_m',np.nan)),
    )
    return result


def _cleanup_rejected_stair_raw(env,run):
    base=ROOT/env/'Data_anymal'/run
    if base.exists():
        shutil.rmtree(base)
        print('Removed rejected stair-candidate raw data:',base)


def _stair_candidate_pairs_newest_first():
    for env,count in STAIR_URBAN_ENV_COUNTS:
        for k in range(int(count)-1,-1,-1):
            yield env,f'P{2000+k:04d}'


def _validate_and_apply_stair_selection(payload):
    train=[tuple(x) for x in payload.get('train',[])]
    hold=tuple(payload.get('holdout',()))
    if len(train)!=TRAIN_TRAJECTORIES_PER_BUCKET or len(hold)!=2:
        raise RuntimeError('Persisted stair selection has wrong shape.')
    if len(set(train+[hold]))!=STAIR_TRAJECTORIES_TO_DISCOVER:
        raise RuntimeError('Persisted stair selection contains duplicates.')
    EXPANDED_TRAIN_RUNS[STAIR_TERRAIN_TYPE]=train
    STRICT_HOLDOUTS[STAIR_TERRAIN_TYPE]=hold
    return train,hold


def discover_true_stair_trajectories():
    """Find 7 train + 1 holdout stair traversals; persist progress after every candidate."""
    global STAIR_DISCOVERY_LOG, STAIR_SELECTED_AUDITS

    if STAIR_SELECTION_FILE.exists():
        payload=json.loads(STAIR_SELECTION_FILE.read_text())
        if payload.get('policy_version') in ('true_stair_v2','true_stair_v3_resumable'):
            train,hold=_validate_and_apply_stair_selection(payload)
            STAIR_DISCOVERY_LOG=payload.get('audit_log',[])
            STAIR_SELECTED_AUDITS={(x['environment'],x['run']):x for x in payload.get('selected_audits',[])}
            print('Reusing persisted true-stair selection:',train,'holdout=',hold)
            return
        print('Ignoring stale stair-selection policy:',payload.get('policy_version'))

    legacy_used={x for k,runs in EXPANDED_TRAIN_RUNS.items() if k!=STAIR_TERRAIN_TYPE for x in runs}
    legacy_used |= set(STRICT_HOLDOUTS.values())
    passed=[]; log=[]; selected_audits={}; tried=set()

    # Resume discovery itself after a power/kernel loss.
    if STAIR_DISCOVERY_STATE_FILE.exists():
        try:
            st=json.loads(STAIR_DISCOVERY_STATE_FILE.read_text())
            if st.get('policy_version')=='true_stair_v3_resumable':
                passed=[tuple(x) for x in st.get('passed',[])]
                log=list(st.get('audit_log',[]))
                selected_audits={(x['environment'],x['run']):x for x in st.get('selected_audits',[])}
                tried={(x.get('environment'),x.get('run')) for x in log}
                print(f'Resuming stair discovery: {len(passed)} verified passes, {len(tried)} candidates already audited.')
        except Exception as e:
            print('Ignoring unreadable stair discovery state:',e)

    def _persist_discovery_state():
        atomic_json_write(STAIR_DISCOVERY_STATE_FILE,dict(
            policy_version='true_stair_v3_resumable',passed=[list(x) for x in passed],
            audit_log=log,selected_audits=list(selected_audits.values()),
        ))
        # Human-readable running log too.
        tmp=OUT_ROOT/'V7_STAIR_DISCOVERY_AUDIT.partial.csv'
        pd.DataFrame(log).to_csv(tmp,index=False)

    for env,run in _stair_candidate_pairs_newest_first():
        if (env,run) in legacy_used or (env,run) in tried:
            continue
        print('\n'+'='*100)
        print('TRUE-STAIR DISCOVERY | trying newest remaining candidate:',env,run)
        print('='*100)
        assert_disk_headroom('stair candidate download')
        try:
            audit=_stair_candidate_audit(env,run)
        except Exception as e:
            audit=dict(environment=env,run=run,pose_pass=False,depth_geometry_pass=False,
                       true_stair_pass=False,rejection_reason='audit_exception:'+repr(e))
        log.append(audit); tried.add((env,run))
        print(
            f'STAIR CANDIDATE {env}/{run} | pose_span={audit.get("robust_vertical_span_m",float("nan")):.3f} m | '
            f'geometry_frames={audit.get("geometry_pass_frames",0)} | best_steps={audit.get("best_step_groups",0)} | '
            f'PASS={audit.get("true_stair_pass",False)} | {audit.get("rejection_reason","")}'
        )
        if audit.get('true_stair_pass',False):
            passed.append((env,run)); selected_audits[(env,run)]=audit
        else:
            _cleanup_rejected_stair_raw(env,run)
        _persist_discovery_state()
        if len(passed)>=STAIR_TRAJECTORIES_TO_DISCOVER:
            break

    if len(passed)<STAIR_TRAJECTORIES_TO_DISCOVER:
        pd.DataFrame(log).to_csv(OUT_ROOT/'V7_STAIR_DISCOVERY_AUDIT.csv',index=False)
        raise RuntimeError(
            f'Only {len(passed)} genuine stair traversals were found. Need {STAIR_TRAJECTORIES_TO_DISCOVER}. '
            'Configured urban ANYmal candidates were exhausted; no ramp/flat substitute is allowed.'
        )

    train=passed[:TRAIN_TRAJECTORIES_PER_BUCKET]; hold=passed[TRAIN_TRAJECTORIES_PER_BUCKET]
    EXPANDED_TRAIN_RUNS[STAIR_TERRAIN_TYPE]=train; STRICT_HOLDOUTS[STAIR_TERRAIN_TYPE]=hold
    STAIR_DISCOVERY_LOG=log; STAIR_SELECTED_AUDITS=selected_audits

    train_keys={x for runs in EXPANDED_TRAIN_RUNS.values() for x in runs}; holdout_keys=set(STRICT_HOLDOUTS.values())
    if len(train_keys)!=EXPECTED_TRAIN_TRAJECTORIES: raise RuntimeError(f'Expected {EXPECTED_TRAIN_TRAJECTORIES} training runs after stair discovery, got {len(train_keys)}')
    if len(holdout_keys)!=EXPECTED_STRICT_HOLDOUTS: raise RuntimeError(f'Expected {EXPECTED_STRICT_HOLDOUTS} holdouts after stair discovery, got {len(holdout_keys)}')
    if train_keys & holdout_keys: raise RuntimeError('Stair discovery created train/holdout leakage: '+repr(sorted(train_keys&holdout_keys)))

    payload=dict(policy_version='true_stair_v3_resumable',train=[list(x) for x in train],holdout=list(hold),
                 audit_log=log,selected_audits=list(selected_audits.values()))
    atomic_json_write(STAIR_SELECTION_FILE,payload)
    pd.DataFrame(log).to_csv(OUT_ROOT/'V7_STAIR_DISCOVERY_AUDIT.csv',index=False)
    STAIR_DISCOVERY_STATE_FILE.unlink(missing_ok=True)
    print('\nVERIFIED STAIR TRAIN RUNS:',train)
    print('VERIFIED STAIR STRICT HOLDOUT:',hold)


discover_true_stair_trajectories()

# Hard proof immediately after discovery.
_train_keys={x for runs in EXPANDED_TRAIN_RUNS.values() for x in runs}
_holdout_keys=set(STRICT_HOLDOUTS.values())
if len(_train_keys)!=EXPECTED_TRAIN_TRAJECTORIES or len(_holdout_keys)!=EXPECTED_STRICT_HOLDOUTS:
    raise RuntimeError('Expanded split size is wrong after true-stair discovery.')
if _train_keys & _holdout_keys:
    raise RuntimeError('STRICT HOLDOUT LEAKAGE after true-stair discovery.')


# Prepare/cache one trajectory completely before moving to the next. This prevents
# all 35 raw RGB-D/rosbag runs from accumulating on ephemeral local disk.
for terrain_type,runs in EXPANDED_TRAIN_RUNS.items():
    for env,run in runs:
        print('\n'+'#'*100)
        print('TRAIN PREP',terrain_type,env,run)
        print('#'*100)

        key=(env,run)
        is_diagnostic=(key in _diag_map)
        base,data,manifest,visual=prepare_v7_record_data(
            env,run,require_raw=is_diagnostic
        )

        stair_audit=None
        if terrain_type==STAIR_TERRAIN_TYPE:
            stair_audit=copy.deepcopy(STAIR_SELECTED_AUDITS.get((env,run),{}))
            if not stair_audit.get('true_stair_pass',False):
                raise RuntimeError(
                    f'{env}/{run} entered the stair training pool without passing '
                    'the repeated-step geometry audit.'
                )
            print(
                'VERIFIED STAIR TRAIN RUN | '
                f'{env}/{run} | pose span={stair_audit.get("robust_vertical_span_m",np.nan):.3f} m | '
                f'geometry pass frames={stair_audit.get("geometry_pass_frames",0)} | '
                f'best repeated steps={stair_audit.get("best_step_groups",0)}'
            )

        fk0=ANYMAL_D_FK.foot_positions(data['P'][0,:,0:3])
        print('ANYmal-D FK sample foot positions [m]:\n',np.array2string(fk0,precision=4))

        if is_diagnostic:
            ttests=[float(x) for x in _diag_map[key]['ttests']]
            for tt in ttests:
                if tt+1.0>float(data['time'][-1]):
                    raise RuntimeError(f'diagnostic ttest too close to run end: {env}/{run} t={tt}')
        else:
            # The helper functions use min(ttests) as the causal GLOBAL training cutoff.
            # For auxiliary TRAINING-ONLY trajectories, place that cutoff just before
            # the run end so almost all recorded data contributes to V7 and V7-F.
            t0=float(data['time'][0]); t1=float(data['time'][-1])
            cutoff=t1-1.05
            if cutoff < t0 + LONG_CONTEXT_SEC + max(LONG_HORIZONS) + 1.0:
                raise RuntimeError(f'Auxiliary trajectory too short for long model: {env}/{run}')
            ttests=[float(cutoff)]

        TRAIN_RECORDS.append(dict(
            terrain_type=terrain_type,
            environment=env,
            run=run,
            ttests=ttests,
            is_diagnostic=is_diagnostic,
            base=base,
            data=data,
            manifest=manifest,
            visual=visual,
            stair_audit=stair_audit,
        ))

        # Delete ONLY redownloadable raw data after physics+manifest+all 4 visual
        # caches have passed integrity checks. Keep all four diagnostic runs.
        if CLEAN_AUX_RAW_AFTER_CACHE and (not is_diagnostic) and base.exists():
            shutil.rmtree(base)
            print('Removed cached auxiliary raw trajectory from local disk:',base)

        try:
            free_gb=shutil.disk_usage(str(ROOT.parent)).free/(1024**3)
            print(f'Local free disk after {env}/{run}: {free_gb:.1f} GB')
        except Exception:
            pass

# Absolute proof that all strict held-out runs are absent from ALL model training.
ALL_TRAIN_KEYS={(r['environment'],r['run']) for r in TRAIN_RECORDS}
STRICT_HOLDOUT_KEYS=set(STRICT_HOLDOUTS.values())
_overlap=ALL_TRAIN_KEYS & STRICT_HOLDOUT_KEYS
print('\nALL V7/V7-F TRAIN KEYS:',len(ALL_TRAIN_KEYS))
print('STRICT HOLDOUT KEYS   :',sorted(STRICT_HOLDOUT_KEYS))
print('TRAIN/HOLDOUT OVERLAP :',_overlap)
if _overlap:
    raise RuntimeError('STRICT HOLDOUT FAILURE before training: '+repr(sorted(_overlap)))
if len(TRAIN_RECORDS)!=EXPECTED_TRAIN_TRAJECTORIES:
    raise RuntimeError(
        f'Expected {EXPECTED_TRAIN_TRAJECTORIES} training trajectories, '
        f'got {len(TRAIN_RECORDS)}'
    )

# Keep the exact four original runs as the diagnostic/local-adaptation set.
# They are references to objects already prepared above, not duplicate downloads.
_train_lookup={(r['environment'],r['run']):r for r in TRAIN_RECORDS}
run_records=[_train_lookup[(e['environment'],e['run'])] for e in EXPERIMENTS]

# Restore the exact original diagnostic ttest arrays in case the training object
# construction above is ever modified independently.
for rec,e in zip(run_records,EXPERIMENTS):
    rec['ttests']=[float(x) for x in e['ttests']]

print('\nTRAINING SET BY TERRAIN:')
for terrain in EXPANDED_TRAIN_RUNS:
    rr=[r for r in TRAIN_RECORDS if r['terrain_type']==terrain]
    print(f'  {terrain}: {len(rr)} trajectories -> {[r["run"] for r in rr]}')
print('Diagnostic runs:',[(r['environment'],r['run']) for r in run_records])

# C) Learn the exact same global load regimes + per-run soft physics,
#    now from all 35 training trajectories, including 7 geometry-verified staircase traversals.
initialize_regimes_and_physics(TRAIN_RECORDS)

# D) Sanity forward pass before hours of training.
rec0=run_records[0]; data0=rec0['data']; vis0=rec0['visual']; dt=float(np.median(np.diff(data0['time']))); Hs=int(round(SHORT_CONTEXT_SEC/dt))
idx0=nearest_index(data0['time'],min(rec0['ttests'])-1.0); stats0=make_stats_between(data0,float(data0['time'][0]),min(rec0['ttests'])); norm0=normalize_arrays(data0,stats0)
s0=make_v7_sample(data0,vis0,stats0,rec0['physics_prior_v7'],idx0,Hs,SHORT_HORIZONS,norm0)
m0=make_model_for_record(rec0,Hs,SHORT_HORIZONS); o0=model_forward_v7(m0,s0)
assert o0['residual_p'].shape==(1,1,4,12) and o0['residual_b'].shape==(1,1,12)
assert o0['binary_contact_logits'].shape==(1,1,4) and o0['regime_logits'].shape==(1,1,4,4)
smoke_loss,smoke_parts=v7_loss(o0,s0,stats0,rec0['physics_prior_v7'])
if not torch.isfinite(smoke_loss): raise RuntimeError('Real-data V7 loss smoke test is non-finite.')
smoke_loss.backward()
for n,p in m0.named_parameters():
    if p.grad is not None and not torch.isfinite(p.grad).all(): raise RuntimeError('Real-data smoke gradient failure: '+n)
print('REAL-DATA MODEL + LOSS + BACKWARD SMOKE TEST PASSED | loss=',float(smoke_loss.detach().cpu()))
del m0,o0,s0,smoke_loss; gc.collect(); torch.cuda.empty_cache()

# E) Global priors.
prior_short=train_v7_prior(TRAIN_RECORDS,SHORT_CONTEXT_SEC,SHORT_HOP_SEC,SHORT_HORIZONS,PRIOR_EPOCHS_SHORT,SEED+10,'SHORT 0.5s -> +0.1s',V5_SHORT_CKPT)
atomic_torch_save(prior_short.state_dict(),OUT_ROOT/'V7_PRIOR_SHORT.pt')
prior_long=train_v7_prior(TRAIN_RECORDS,LONG_CONTEXT_SEC,LONG_HOP_SEC,LONG_HORIZONS,PRIOR_EPOCHS_LONG,SEED+20,'LONG 4s -> +0.2..+1.0s',V5_LONG_CKPT)
atomic_torch_save(prior_long.state_dict(),OUT_ROOT/'V7_PRIOR_LONG.pt')

# F) 4x4 exact tests. Independent tests are checkpointed after each completion.
_v7_test_recovery=RECOVERY_ROOT/'_RECOVERY_V7_16_DIAGNOSTIC_TESTS_v2.pt'
_v7_test_signature=dict(
    schema='V7_DIAGNOSTIC_RESUME_20260926_v2',
    short_prior=model_state_fingerprint(prior_short),long_prior=model_state_fingerprint(prior_long),
    tests=[(r['terrain_type'],[(i+1,float(t)) for i,t in enumerate(r['ttests'])]) for r in run_records],
    local_epochs_short=int(LOCAL_EPOCHS_SHORT),local_epochs_long=int(LOCAL_EPOCHS_LONG),lr_local=float(LR_LOCAL),
)
all_values=[]; all_reg=[]; all_force=[]; test_manifest=[]; _v7_done=set()
if _v7_test_recovery.exists():
    try:
        _ck=torch.load(_v7_test_recovery,map_location='cpu',weights_only=False)
        if _ck.get('signature')==_v7_test_signature:
            all_values=_ck.get('all_values',[]); all_reg=_ck.get('all_reg',[]); all_force=_ck.get('all_force',[])
            test_manifest=_ck.get('test_manifest',[]); _v7_done={tuple(x) for x in _ck.get('done',[])}
            print(f'Resuming V7 diagnostics: {len(_v7_done)}/16 tests already complete.')
    except Exception as e: print('Ignoring unreadable V7 diagnostic recovery:',e)
for ri,rec in enumerate(run_records):
    for ti,tt in enumerate(rec['ttests'],1):
        _key=(rec['terrain_type'],int(ti),float(tt))
        if _key in _v7_done:
            print('Skipping completed V7 diagnostic:',_key); continue
        print('\n'+'-'*100+f'\nTEST {rec["terrain_type"]} #{ti} t={tt:.1f}\n'+'-'*100)
        seed=SEED+ri*1000+ti*100
        sm,ss,sp,Hs=adapt_v7(prior_short,rec['data'],rec['visual'],tt,SHORT_CONTEXT_SEC,SHORT_HOP_SEC,SHORT_HORIZONS,SHORT_RECENCY_HALF_LIFE_SEC,LOCAL_EPOCHS_SHORT,seed+1,'SHORT')
        lm,ls,lp,Hl=adapt_v7(prior_long,rec['data'],rec['visual'],tt,LONG_CONTEXT_SEC,LONG_HOP_SEC,LONG_HORIZONS,LONG_RECENCY_HALF_LIFE_SEC,LOCAL_EPOCHS_LONG,seed+2,'LONG')
        vals,regs,forces,idx,vi=evaluate_v7_case(sm,ss,sp,Hs,lm,ls,lp,Hl,rec['data'],rec['visual'],tt)
        meta=dict(terrain_type=rec['terrain_type'],environment=rec['environment'],run=rec['run'],test_id=ti,ttest_s=float(tt))
        for k,v in meta.items(): vals[k]=v; regs[k]=v; forces[k]=v
        all_values.append(vals); all_reg.append(regs); all_force.append(forces)
        test_manifest.append(dict(**meta,robot_idx=idx,visual_idx=vi,visual_time=float(rec['visual']['camera_times'][vi]),frame_id=str(rec['visual']['frame_ids'][vi])))
        _v7_done.add(_key)
        atomic_torch_save(dict(signature=_v7_test_signature,all_values=all_values,all_reg=all_reg,all_force=all_force,
                               test_manifest=test_manifest,done=[list(x) for x in sorted(_v7_done)]),_v7_test_recovery)
        print(' Saved V7 diagnostic recovery:',_v7_test_recovery)
        del sm,lm; gc.collect(); torch.cuda.empty_cache()
if len(_v7_done)!=16: raise RuntimeError(f'V7 diagnostic recovery expected 16 complete tests, got {len(_v7_done)}')

all_df=pd.concat(all_values,ignore_index=True); reg_df=pd.concat(all_reg,ignore_index=True); force_df=pd.concat(all_force,ignore_index=True)
# integrity: 16 tests x 10 horizons x 64 signals
expected=16*10*64
if len(all_df)!=expected: raise RuntimeError(f'Expected {expected} result rows, got {len(all_df)}')
key=['terrain_type','test_id','horizon_s','signal']
if all_df.duplicated(key).any(): raise RuntimeError('Duplicate result rows detected.')
if set(all_df[all_df.horizon_s==.1].expert_used)!= {'SHORT'}: raise RuntimeError('+0.1 expert policy broken')
if set(all_df[all_df.horizon_s>=.2].expert_used)!= {'LONG'}: raise RuntimeError('long expert policy broken')

# G) Metrics and files.
grouped,mean=summarize_metrics(all_df); cm=contact_metrics(all_df); trend=trend_metrics(all_df); acceptance=acceptance_table(mean)
qdeg=mean[mean.signal_group=='q'][['horizon_s','MAE']].copy(); qdeg['MAE_degrees']=qdeg.MAE*180/np.pi
reg_cm=confusion_matrix(reg_df.true_regime,reg_df.pred_regime,labels=[0,1,2,3])
reg_cm_df=pd.DataFrame(reg_cm,index=['true_0','true_1','true_2','true_3'],columns=['pred_0','pred_1','pred_2','pred_3'])
force_summary=force_df.groupby('horizon_s',as_index=False).agg(force_vector_error=('force_vector_error','mean'),force_magnitude_MAE=('force_mag_abs_error','mean'),force_direction_cosine=('force_direction_cosine','mean'))

all_df.to_csv(OUT_ROOT/'V7_ALL_16_TESTS_ALL_SENSOR_VALUES.csv',index=False)
grouped.to_csv(OUT_ROOT/'V7_ALL_16_TESTS_GROUPED_METRICS.csv',index=False)
mean.to_csv(OUT_ROOT/'V7_MEAN_METRICS_ACROSS_16_TESTS.csv',index=False)
cm.to_csv(OUT_ROOT/'V7_CONTACT_METRICS_BY_HORIZON.csv',index=False)
force_summary.to_csv(OUT_ROOT/'V7_FORCE_VECTOR_METRICS_BY_HORIZON.csv',index=False)
reg_df.to_csv(OUT_ROOT/'V7_4REGIME_RAW.csv',index=False); reg_cm_df.to_csv(OUT_ROOT/'V7_4REGIME_CONFUSION_MATRIX.csv')
trend.to_csv(OUT_ROOT/'V7_LONG_HORIZON_TREND_METRICS_RAW.csv',index=False)
trend_mean=trend.copy(); trend_mean['signal_group']=trend_mean['signal'].map(signal_group_v7)
trend_mean=trend_mean.groupby('signal_group',as_index=False)[['correlation','first_difference_sign_accuracy']].mean()
trend_mean.to_csv(OUT_ROOT/'V7_LONG_HORIZON_TREND_METRICS_MEAN.csv',index=False)
acceptance.to_csv(OUT_ROOT/'V7_ACCEPTANCE_VS_BEST_EXISTING.csv',index=False); qdeg.to_csv(OUT_ROOT/'V7_Q_MAE_RADIANS_AND_DEGREES.csv',index=False)
pd.DataFrame(test_manifest).to_csv(OUT_ROOT/'V7_16_TEST_MANIFEST.csv',index=False)

# Save a 4x4 audit montage showing the exact causal RGB frame and the geometric ground mask/bbox
# used by all four visual branches at each test time.
fig,axes=plt.subplots(4,4,figsize=(16,16))
for ax,tm in zip(axes.ravel(),test_manifest):
    rec=next(r for r in run_records if r['terrain_type']==tm['terrain_type'] and r['environment']==tm['environment'] and r['run']==tm['run'])
    vi=int(tm['visual_idx']); base=rec['base']; manifest=rec['manifest']
    _,rgb,_,mask,_,_,_,_=get_frame_inputs(base,manifest,vi)
    ov=rgb.copy(); green=np.zeros_like(ov); green[...,1]=255
    ov=np.where(mask[...,None],(0.70*ov+0.30*green).astype(np.uint8),ov)
    x0,y0,x1,y1=[int(x) for x in manifest['crop_boxes'][vi]]
    cv2.rectangle(ov,(x0,y0),(x1-1,y1-1),(255,255,255),2)
    ax.imshow(ov); ax.axis('off'); ax.set_title(f"{tm['terrain_type']} T{tm['test_id']}  t={tm['ttest_s']:.1f}s\ncam={tm['visual_time']:.3f}s")
plt.tight_layout(); plt.savefig(OUT_ROOT/'V7_ALL_16_TEST_RGB_GEOMETRIC_MASKS.png',dpi=150); plt.show()

# H) plots: one separate figure per metric group.
for group in ['q','dq','tau','force','contact']:
    g=mean[mean.signal_group==group].sort_values('horizon_s')
    if len(g):
        plt.figure(figsize=(7,4)); plt.plot(g.horizon_s,g.MAE,marker='o'); plt.xlabel('prediction horizon (s)'); plt.ylabel(f'{group} MAE'); plt.title(f'V7 {group} MAE vs horizon'); plt.grid(True,alpha=.25); plt.tight_layout(); plt.savefig(OUT_ROOT/f'V7_{group}_MAE_vs_horizon.png',dpi=160); plt.show()

print('\nV7 MEAN METRICS'); display(mean)
print('\nCONTACT'); display(cm)
print('\nFORCE VECTOR'); display(force_summary)
print('\nACCEPTANCE VS BEST EXISTING'); display(acceptance)
print('\nEND-TO-END V7 EXPERIMENT COMPLETE.')

# %% notebook cell 13
# ============================================================
# V7 GEOMETRY-CONSTANT ORDERING CHECK
# GEOM_INPUT_POINTS is now defined in the configuration cell,
# before the Utonia builder can be called.
# ============================================================

assert GEOM_INPUT_POINTS == int(POINTS_PER_FRAME) == 4096
assert GEOM_REP_POINTS == 64

print("GEOM_INPUT_POINTS =", GEOM_INPUT_POINTS)
print("GEOM_REP_POINTS   =", GEOM_REP_POINTS)
print("Geometry constant ordering is correct.")

# %% notebook cell 14
# ======================================================================
# 10. FINAL AUDIT, README, CHECKSUM MANIFEST, MASTER ZIP + DOWNLOAD
# ======================================================================
import shutil, hashlib, datetime, json

# Save the expanded model-training split used by BOTH V7 and V7-F.
if 'TRAIN_RECORDS' in globals():
    pd.DataFrame([
        dict(
            terrain_type=r['terrain_type'],
            environment=r['environment'],
            run=r['run'],
            global_training_cutoff_s=float(min(r['ttests'])),
            diagnostic_run=bool(r.get('is_diagnostic',False)),
            strict_holdout=False,
            stair_candidate=bool(r['terrain_type']==STAIR_TERRAIN_TYPE),
            stair_robust_vertical_span_m=(
                float(r['stair_audit']['robust_vertical_span_m'])
                if r.get('stair_audit') is not None else np.nan
            ),
            stair_total_vertical_travel_m=(
                float(r['stair_audit']['total_vertical_travel_m'])
                if r.get('stair_audit') is not None else np.nan
            ),
            stair_true_stair_pass=(
                bool(r['stair_audit']['true_stair_pass'])
                if r.get('stair_audit') is not None else np.nan
            ),
            stair_geometry_pass_frames=(
                int(r['stair_audit'].get('geometry_pass_frames',0))
                if r.get('stair_audit') is not None else np.nan
            ),
            stair_best_step_groups=(
                int(r['stair_audit'].get('best_step_groups',0))
                if r.get('stair_audit') is not None else np.nan
            ),
        )
        for r in TRAIN_RECORDS
    ]).to_csv(OUT_ROOT/'V7_EXPANDED_TRAIN_TRAJECTORIES.csv',index=False)


# Save the full staircase discovery audit, including rejected ramp/flat candidates.
if 'STAIR_DISCOVERY_LOG' in globals():
    pd.DataFrame(STAIR_DISCOVERY_LOG).to_csv(
        OUT_ROOT/'V7_STAIR_DISCOVERY_AUDIT.csv',index=False
    )

# Save the seven actually-used training stairs with their strict geometry evidence.
if 'TRAIN_RECORDS' in globals():
    _stair_rows=[]
    for r in TRAIN_RECORDS:
        if r['terrain_type']!=STAIR_TERRAIN_TYPE:
            continue
        a=r.get('stair_audit') or {}
        _stair_rows.append(dict(
            terrain_type=r['terrain_type'],
            environment=r['environment'],
            run=r['run'],
            robust_vertical_span_m=a.get('robust_vertical_span_m',np.nan),
            total_vertical_travel_m=a.get('total_vertical_travel_m',np.nan),
            net_vertical_change_m=a.get('net_vertical_change_m',np.nan),
            geometry_pass_frames=a.get('geometry_pass_frames',0),
            max_consecutive_geometry_pass_frames=a.get('max_consecutive_geometry_pass_frames',0),
            best_step_groups=a.get('best_step_groups',0),
            best_median_rise_m=a.get('best_median_rise_m',np.nan),
            best_median_tread_spacing_m=a.get('best_median_tread_spacing_m',np.nan),
            best_jump_energy_ratio=a.get('best_jump_energy_ratio',np.nan),
            best_linear_ramp_fit_rmse_m=a.get('best_linear_ramp_fit_rmse_m',np.nan),
            true_stair_pass=a.get('true_stair_pass',False),
        ))
    pd.DataFrame(_stair_rows).to_csv(
        OUT_ROOT/'V7_STAIR_TRAJECTORY_AUDIT.csv',index=False
    )

# Persist a human-readable copy of the final dynamic split inside results too.
if STAIR_TERRAIN_TYPE in STRICT_HOLDOUTS:
    (OUT_ROOT/'V7_STAIR_SELECTION.json').write_text(json.dumps(dict(
        train=[list(x) for x in EXPANDED_TRAIN_RUNS[STAIR_TERRAIN_TYPE]],
        strict_holdout=list(STRICT_HOLDOUTS[STAIR_TERRAIN_TYPE]),
        policy='pose vertical motion + repeated tread/riser depth geometry',
    ),indent=2))

# Save exact configuration.
config={k:v for k,v in globals().items() if k in [
    'SEED','TRAIN_LOOKBACK_SEC','SHORT_CONTEXT_SEC','LONG_CONTEXT_SEC','SHORT_HOP_SEC','LONG_HOP_SEC',
    'SHORT_HORIZONS','LONG_HORIZONS','PRIOR_EPOCHS_SHORT','PRIOR_EPOCHS_LONG','LOCAL_EPOCHS_SHORT','LOCAL_EPOCHS_LONG',
    'LR_PRIOR','LR_LOCAL','DINO_MODEL','CONV_MODEL','VISION_INPUT','RAW_GRID','LATENTS_PER_BRANCH','VIS_DIM','FLOW_HISTORY',
    'POINTS_PER_FRAME','CONTACT_THRESHOLD','PI_PHYS_DIM']}
with open(OUT_ROOT/'V7_CONFIG.json','w') as f: json.dump(config,f,indent=2,default=str)

readme=f"""# V7 Full Four-Branch Experiment\n\nFinal policy: +0.1 s = SHORT only; +0.2..1.0 s = LONG only.\n\nMandatory vision branches:\n- {DINO_MODEL}\n- {CONV_MODEL}\n- Pointcept/Utonia pretrained PTv3\n- WAFT a1 tar-c-t optical flow + temporal CNN\n\nPhysics is a causal identified soft proxy, not exact ANYmal rigid-body inverse dynamics.\nFuture recorded U/L are conditional exogenous controls in this offline benchmark and must be replaced by planned controller commands at deployment.\nThe 16 fixed tests are a controlled diagnostic comparison to V4/V5/V6, not a claim of held-out terrain generalization.\nGlobal V7/V7-F training uses 35 trajectories: 7 each from forest, swamp/marsh, rough/uneven winter, wet/uneven proxy, and dynamically discovered stairs_urban.\nThe staircase bucket is selected only from urban environments with official ANYmal trajectories. Candidates must pass both a pose vertical-motion prefilter and repeated tread/riser depth geometry; smooth ramps are rejected. The eighth passing stair trajectory is the strict unseen staircase holdout.\n"""
(OUT_ROOT/'README_V7_EXPERIMENT.md').write_text(readme)

# Manifest/checksums for results only.
rows=[]
for p in sorted(OUT_ROOT.rglob('*')):
    if p.is_file():
        h=hashlib.sha256()
        with open(p,'rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
        rows.append(dict(relative_path=str(p.relative_to(OUT_ROOT)),size_bytes=p.stat().st_size,sha256=h.hexdigest()))
pd.DataFrame(rows).to_csv(OUT_ROOT/'V7_RESULT_MANIFEST_SHA256.csv',index=False)

# Results-only ZIP is small enough to download directly.
import zipfile
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
results_zip=WORK_ROOT/f'V7_RESULTS_ONLY_{stamp}.zip'
with zipfile.ZipFile(results_zip,'w',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as z:
    for p in sorted(OUT_ROOT.rglob('*')):
        if p.is_file(): z.write(p,arcname=str(Path('results')/p.relative_to(OUT_ROOT)))
print('RESULTS ZIP:',results_zip,results_zip.stat().st_size/(1024**2),'MB')

# Full master ZIP streams directly from the persistent result/cache roots. This avoids making a second
# multi-GB copy in /content. It excludes the raw redownloadable TartanGround directory.
BUILD_FULL_CACHE_ZIP=False  # do not duplicate many GB before V7-F/V8 overnight stages
full_zip=None
if BUILD_FULL_CACHE_ZIP:
    full_zip=PERSIST_ROOT/f'V7_FULL4B_RESULTS_AND_4BRANCH_CACHES_{stamp}.zip'
    with zipfile.ZipFile(full_zip,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for root,name in [(OUT_ROOT,'results'),(CACHE_ROOT,'cache')]:
            for p in sorted(root.rglob('*')):
                if p.is_file(): z.write(p,arcname=str(Path(name)/p.relative_to(root)))
    print('FULL CACHE ZIP:',full_zip,full_zip.stat().st_size/(1024**3),'GB')
else:
    print('Full multi-GB cache ZIP intentionally skipped during overnight run.')

try:
    from google.colab import files
    files.download(str(results_zip))
except Exception as e:
    print('Automatic results download skipped:',e)

if full_zip is not None:
    print('The full cache ZIP remains at:',full_zip)

# %% notebook cell 15
# ======================================================================
# V7-F / V7.1 FORCE SPECIALIST — APPEND-ONLY PATCH FOR A LIVE V7 SESSION
#
# Paste/run this ONE cell AFTER the completed V7 experiment cell.
# It REUSES run_records, prior_short, prior_long, all_df and all visual caches.
# It does NOT rerun DINOv3 / ConvNeXt / Utonia / WAFT.
#
# Scientific changes vs V7:
#   1) freeze the successful V7 dynamics/vision trunk;
#   2) train force as a dedicated specialist task;
#   3) true mixed-terrain mini-batches for the GLOBAL force head;
#   4) denser global windows, especially LONG (0.25 s vs 1.0 s hop);
#   5) contact-phase-conditioned mixture-of-experts;
#   6) joint multi-horizon force Transformer for +0.2 ... +1.0 s;
#   7) direct-vector + magnitude/direction branches with learned fusion;
#   8) force-magnitude/high-load/contact-transition weighting;
#   9) long-horizon temporal-difference + impulse consistency losses;
#  10) local adaptation changes ONLY the small force specialist and uses
#      a causal train/validation split of the previous 20 s.
#
# IMPORTANT:
#   - q/dq/tau/body/contact predictions in the final CSV are preserved from
#     the already-completed V7 all_df.
#   - only FORCE rows are replaced by V7-F predictions.
#   - no V5 test predictions are used for training (that would leak the test).
# ======================================================================

import os, gc, copy, json, math, random, hashlib, zipfile, shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# 0) HARD PRECONDITION CHECKS
# ----------------------------------------------------------------------
_required = [
    'run_records','TRAIN_RECORDS','prior_short','prior_long','all_df','OUT_ROOT','PERSIST_ROOT','RECOVERY_ROOT','WORK_ROOT','DEVICE',
    'SHORT_CONTEXT_SEC','LONG_CONTEXT_SEC','SHORT_HORIZONS','LONG_HORIZONS',
    'SHORT_RECENCY_HALF_LIFE_SEC','LONG_RECENCY_HALF_LIFE_SEC','TRAIN_LOOKBACK_SEC',
    'make_range_windows_v7','make_local_windows_v7','make_v7_sample',
    'normalize_arrays','make_stats_between','nearest_index','horizon_offsets',
    'tensor','REGIME_THRESHOLDS','CONTACT_THRESHOLD','signal_group_v7',
    'summarize_metrics','acceptance_table','LEG_NAMES','P_NAMES','B_NAMES',
]
_missing=[x for x in _required if x not in globals()]
if _missing:
    raise RuntimeError(
        'V7-F must be appended to the SAME live session after V7 completed. '
        'Missing globals: '+repr(_missing)
    )

if len(run_records)!=4 or len(all_df)==0:
    raise RuntimeError('Expected completed 4-run V7 experiment in memory.')

if len(TRAIN_RECORDS)!=EXPECTED_TRAIN_TRAJECTORIES:
    raise RuntimeError(
        f'Expected {EXPECTED_TRAIN_TRAJECTORIES} expanded training trajectories, '
        f'got {len(TRAIN_RECORDS)}'
    )

# Every four-stream cache used by global V7-F training must still be available.
for _r in TRAIN_RECORDS:
    if 'visual' not in _r:
        raise RuntimeError('run_records visual cache missing for '+_r['terrain_type'])
    for _k in ('DINO','CONV','GEOM','FLOW'):
        if _k not in _r['visual']:
            raise RuntimeError(f'Mandatory visual branch {_k} missing for {_r["terrain_type"]}')

print('V7-F preflight: live V7 session + all four visual branches found for all training trajectories.')

# ----------------------------------------------------------------------
# 1) CONFIG
# ----------------------------------------------------------------------
V7F_SEED = int(globals().get('SEED', 1337)) + 7100

# More global examples than V7. These are correlated windows, so they are
# optimization samples, not claimed as independent data points.
V7F_GLOBAL_HOP_SHORT = 0.15
V7F_GLOBAL_HOP_LONG  = 0.25

# True mixed-terrain mini-batches are used for the specialist head.
V7F_BATCH_SHORT = 16
V7F_BATCH_LONG  = 8
V7F_EPOCHS_SHORT = 5
V7F_EPOCHS_LONG  = 6
V7F_LR_GLOBAL = 3e-4
V7F_WEIGHT_DECAY = 1e-4
V7F_GRAD_CLIP = 1.0

# Local adaptation is deliberately small/frozen-trunk.
V7F_LOCAL_HOP_SHORT = float(globals().get('SHORT_HOP_SEC', 0.3))
V7F_LOCAL_HOP_LONG  = float(globals().get('LONG_HOP_SEC', 1.0))
V7F_LOCAL_EPOCHS = 5
V7F_LR_LOCAL = 6e-5
V7F_LOCAL_VAL_FRAC = 0.20

# Loss weights. Force magnitude is the known V7 weakness.
V7F_W_VEC      = 2.00
V7F_W_MAG      = 1.60
V7F_W_DIR      = 0.10  # direction is already strong; do not over-disturb it
V7F_W_PHASE    = 0.25
V7F_W_TEMP     = 0.45
V7F_W_IMPULSE  = 0.35
V7F_W_ANCHOR   = 0.015
V7F_W_GATE_ENT = 0.005

# The last 10 Utonia-cache channels are [xyz3, normal3, rgb3, rough1].
# We do NOT use these normals to rotate force because the notebook correctly
# avoids inventing an unverified camera->base extrinsic. The specialist stays
# in the measured force coordinate system.

V7F_ROOT = PERSIST_ROOT/'V7F_force_specialist'
V7F_ROOT.mkdir(parents=True, exist_ok=True)


def _seed_v7f(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

_seed_v7f(V7F_SEED)

# ----------------------------------------------------------------------
# 2) FORCE CONTACT-PHASE TARGETS
# phase 0=swing, 1=touchdown, 2=stance, 3=liftoff
# ----------------------------------------------------------------------
PHASE_NAMES = ['swing','touchdown','stance','liftoff']
N_PHASES = 4


def v7f_phase_targets(data, idx, horizons):
    offs = horizon_offsets(data, horizons)
    c = np.asarray(data['contact'], np.int64)
    out=[]
    for off in offs:
        end=min(idx+off, len(c)-1)
        seq=c[idx:end+1]  # T,4
        ph=np.zeros(4,np.int64)
        for leg in range(4):
            s=seq[:,leg]
            on=np.any((s[:-1]==0)&(s[1:]==1)) if len(s)>1 else False
            offtr=np.any((s[:-1]==1)&(s[1:]==0)) if len(s)>1 else False
            # End-state-aware phase. If several gait transitions occur inside a
            # long horizon, use the last transition type that agrees with end state.
            if s[-1]==1:
                ph[leg]=1 if on else 2
            else:
                ph[leg]=3 if offtr else 0
        out.append(ph)
    return np.asarray(out,np.int64)


# ----------------------------------------------------------------------
# 3) EXTRACT THE FROZEN V7 FORCE CONTEXT
# ----------------------------------------------------------------------
@torch.inference_mode()
def v7f_extract_base_context(base, s, stats):
    base.eval()
    ctx,legf,vf = base.encode(
        s['p_hist'],s['b_hist'],s['u_hist'],s['l_hist'],s['e_hist'],
        s['dino'],s['conv'],s['geom'],s['flow'],s['foot_query_kin']
    )
    H=s['future_control_features'].shape[1]
    z=torch.cat([
        ctx[:,None].expand(-1,H,-1),
        base.control_proj(s['future_control_features']),
        base.horizon_proj(s['horizon_values']),
        base.physics_proj(s['physics_features'])
    ],dim=-1)

    binary_logits=base.binary_contact_head(z)
    regime_logits=base.regime_head(z).reshape(-1,H,4,N_REGIMES)
    regime_probs=regime_logits.softmax(-1)
    pcontact=torch.sigmoid(binary_logits)

    lf=legf[:,None].expand(-1,H,-1,-1)
    vf2=vf[:,None].expand(-1,H,-1,-1)
    zh=z[:,:,None].expand(-1,-1,4,-1)
    head_in=torch.cat([zh,lf,vf2,pcontact.unsqueeze(-1),regime_probs],dim=-1)

    base_tf=base.tau_force_head(head_in)
    base_force_resid=base_tf[...,3:6]

    Pstd=torch.as_tensor(stats['P_std'],device=head_in.device,dtype=head_in.dtype)
    bp=torch.as_tensor(s['baseline_p_raw'],device=head_in.device,dtype=head_in.dtype)[None]
    baseF=bp[:,:,:,9:12] + base_force_resid*Pstd[9:12].view(1,1,1,3)

    return {
        'x': head_in.detach(),                 # B,H,4,1285
        'baseF': baseF.detach(),               # B,H,4,3 raw force units
        'pcontact': pcontact.detach(),         # B,H,4
        'regime': regime_probs.detach(),       # B,H,4,4
    }


# ----------------------------------------------------------------------
# 4) FORCE SPECIALIST MODEL
# ----------------------------------------------------------------------
class V7ForceSpecialistHead(nn.Module):
    def __init__(self, in_dim=1285, d=256, n_experts=4, horizon_layers=2):
        super().__init__()
        self.in_dim=in_dim; self.d=d; self.n_experts=n_experts

        self.in_proj=nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim,d), nn.GELU(), nn.Dropout(.10),
            nn.Linear(d,d), nn.GELU(), nn.LayerNorm(d)
        )

        layer=nn.TransformerEncoderLayer(
            d_model=d,nhead=8,dim_feedforward=4*d,dropout=.10,
            activation='gelu',batch_first=True,norm_first=True
        )
        self.horizon_tx=nn.TransformerEncoder(layer,num_layers=horizon_layers)

        self.phase_head=nn.Sequential(
            nn.LayerNorm(d),nn.Linear(d,128),nn.GELU(),nn.Linear(128,N_PHASES)
        )

        self.experts=nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d),nn.Linear(d,192),nn.GELU(),nn.Dropout(.08),
                nn.Linear(192,96),nn.GELU(),nn.Linear(96,3)
            ) for _ in range(n_experts)
        ])

        # gate sees dynamics latent + phase + load regime + contact probability
        gate_dim=d+N_PHASES+N_REGIMES+1
        self.expert_gate=nn.Sequential(
            nn.LayerNorm(gate_dim),nn.Linear(gate_dim,128),nn.GELU(),nn.Linear(128,n_experts)
        )

        # Second independent direct-vector correction branch.
        self.direct_corr=nn.Sequential(
            nn.LayerNorm(d),nn.Linear(d,192),nn.GELU(),nn.Linear(192,3)
        )

        # Magnitude/direction branch: delta log-magnitude + direction perturbation.
        self.magdir=nn.Sequential(
            nn.LayerNorm(d),nn.Linear(d,192),nn.GELU(),nn.Linear(192,4)
        )

        # Fuse corrected-direct vector with magnitude/direction reconstruction.
        self.fusion_gate=nn.Sequential(
            nn.LayerNorm(gate_dim),nn.Linear(gate_dim,96),nn.GELU(),nn.Linear(96,1)
        )

        # Start exactly at V7 force prediction: all specialist corrections = 0.
        for mod in list(self.experts)+[self.direct_corr,self.magdir]:
            last=[m for m in mod.modules() if isinstance(m,nn.Linear)][-1]
            nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
        last=[m for m in self.fusion_gate.modules() if isinstance(m,nn.Linear)][-1]
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)

    def forward(self,x,baseF,pstd_force,pcontact,regime):
        # x: B,H,4,1285.  Run a joint future-horizon Transformer separately per leg.
        B,H,L,D=x.shape
        h=self.in_proj(x)
        ht=h.permute(0,2,1,3).reshape(B*L,H,self.d)
        ht=self.horizon_tx(ht)
        h=ht.reshape(B,L,H,self.d).permute(0,2,1,3).contiguous()

        phase_logits=self.phase_head(h)
        phase_prob=phase_logits.softmax(-1)
        gate_in=torch.cat([h,phase_prob,regime,pcontact.unsqueeze(-1)],dim=-1)
        expert_w=self.expert_gate(gate_in).softmax(-1)

        exp=torch.stack([e(h) for e in self.experts],dim=-2)  # B,H,4,E,3
        moe_corr=(expert_w.unsqueeze(-1)*exp).sum(-2)
        direct_corr=self.direct_corr(h)
        corr_norm=moe_corr+direct_corr

        # Direct vector path: correction is in per-axis normalized force units.
        directF=baseF + corr_norm*pstd_force

        # Magnitude/direction path starts from the V7 vector and learns only a bounded correction.
        md=self.magdir(h)
        base_mag=torch.linalg.norm(baseF,dim=-1).clamp_min(1e-5)
        base_dir=F.normalize(baseF,dim=-1,eps=1e-6)
        logm=torch.log1p(base_mag) + 0.75*torch.tanh(md[...,0])
        mag=torch.expm1(logm).clamp_min(0.0)
        direction=F.normalize(base_dir + 0.20*torch.tanh(md[...,1:4]),dim=-1,eps=1e-6)
        magdirF=mag.unsqueeze(-1)*direction

        alpha=torch.sigmoid(self.fusion_gate(gate_in))
        finalF=alpha*directF+(1.0-alpha)*magdirF

        return {
            'force':finalF,
            'phase_logits':phase_logits,
            'phase_prob':phase_prob,
            'expert_w':expert_w,
            'corr_norm':corr_norm,
            'directF':directF,
            'magdirF':magdirF,
            'fusion_alpha':alpha.squeeze(-1),
        }


def _v7f_head_smoke_test():
    m=V7ForceSpecialistHead().to(DEVICE)
    for H in (1,9):
        x=torch.randn(2,H,4,1285,device=DEVICE)
        bf=torch.randn(2,H,4,3,device=DEVICE)*100
        ps=torch.ones(2,1,1,3,device=DEVICE)*20
        pc=torch.rand(2,H,4,device=DEVICE)
        rg=torch.softmax(torch.randn(2,H,4,4,device=DEVICE),-1)
        o=m(x,bf,ps,pc,rg)
        assert o['force'].shape==(2,H,4,3)
        assert o['phase_logits'].shape==(2,H,4,4)
        assert torch.isfinite(o['force']).all()
    del m; gc.collect(); torch.cuda.empty_cache()
    print('V7-F specialist synthetic shape smoke test PASSED.')

_v7f_head_smoke_test()

# ----------------------------------------------------------------------
# 5) GLOBAL WINDOW METADATA + FROZEN V7 FEATURE CACHE
# ----------------------------------------------------------------------
def _global_run_training_objects(rec, context_sec, hop_sec, horizons):
    data,vis=rec['data'],rec['visual']
    end=float(min(rec['ttests']))
    stats=make_stats_between(data,float(data['time'][0]),end)
    norm=normalize_arrays(data,stats)
    physics=rec['physics_prior_v7']
    wins,H=make_range_windows_v7(
        data,vis,float(data['time'][0]),end,
        context_sec,hop_sec,max(horizons)
    )
    return stats,norm,physics,wins,H


def _force_bin(gtF):
    # Bin by the largest future per-foot force magnitude in the window.
    m=float(np.max(np.linalg.norm(gtF,axis=-1)))
    if m < float(REGIME_THRESHOLDS[0]): return 0
    if m < float(REGIME_THRESHOLDS[1]): return 1
    if m < float(REGIME_THRESHOLDS[2]): return 2
    return 3


@torch.inference_mode()
def build_v7f_global_cache(base, context_sec, hop_sec, horizons, label):
    """Frozen-V7 features persisted per trajectory, so a restart does not recompute finished runs."""
    base.eval(); records=[]; base_fp=model_state_fingerprint(base)
    safe=''.join(ch if ch.isalnum() else '_' for ch in str(label)).strip('_')
    cache_dir=V7F_FEATURE_RECOVERY_ROOT/safe; cache_dir.mkdir(parents=True,exist_ok=True)
    print('\n'+'='*100); print('V7-F FROZEN-V7 FEATURE CACHE',label); print('='*100)

    for ri,rec in enumerate(TRAIN_RECORDS):
        run_file=cache_dir/f'{ri:02d}_{rec["environment"]}_{rec["run"]}.pt'
        sig=dict(schema='V7F_FROZEN_FEATURES_20260926_v2',base_fp=base_fp,
                 context_sec=float(context_sec),hop_sec=float(hop_sec),horizons=[float(x) for x in horizons],
                 environment=rec['environment'],run=rec['run'],terrain=rec['terrain_type'],
                 cutoff=float(min(rec['ttests'])))
        rec_records=None
        if run_file.exists():
            try:
                pack=torch.load(run_file,map_location='cpu',weights_only=False)
                if pack.get('signature')==sig:
                    rec_records=pack['records']; print(f' {rec["terrain_type"]}: reusing {len(rec_records)} persisted frozen-feature windows')
            except Exception as e:
                print('  frozen feature cache unreadable; rebuilding',run_file,e)
        if rec_records is None:
            assert_disk_headroom('V7-F frozen feature cache')
            stats,norm,physics,wins,H=_global_run_training_objects(rec,context_sec,hop_sec,horizons)
            print(f' {rec["terrain_type"]}: building {len(wins)} dense windows')
            rec_records=[]
            for wi,(idx,_,te) in enumerate(wins,1):
                samp=make_v7_sample(rec['data'],rec['visual'],stats,physics,idx,H,horizons,norm)
                ctx=v7f_extract_base_context(base,samp,stats); phase=v7f_phase_targets(rec['data'],idx,horizons)
                gtF=np.asarray(samp['target_p_raw'][...,9:12],np.float32)
                rec_records.append({
                    'x':ctx['x'][0].half().cpu(),'baseF':ctx['baseF'][0].float().cpu(),
                    'pcontact':ctx['pcontact'][0].float().cpu(),'regime':ctx['regime'][0].float().cpu(),
                    'pstd_force':torch.as_tensor(stats['P_std'][9:12],dtype=torch.float32),
                    'gtF':torch.as_tensor(gtF,dtype=torch.float32),'phase':torch.as_tensor(phase,dtype=torch.long),
                    'force_bin':_force_bin(gtF),'terrain':rec['terrain_type'],'target_end':float(te),
                })
                if wi in (1,len(wins)) or wi%100==0: print(f'   {wi:04d}/{len(wins):04d}')
                del samp,ctx
            atomic_torch_save(dict(signature=sig,records=rec_records),run_file)
            print('  persisted V7-F frozen features:',run_file)
            gc.collect(); torch.cuda.empty_cache()
        records.extend(rec_records)

    bins=defaultdict(list)
    for j,r in enumerate(records): bins[int(r['force_bin'])].append(j)
    print(' force-bin counts:',{k:len(v) for k,v in bins.items()})
    return records


def balanced_epoch_indices(records,rng):
    bins=defaultdict(list)
    for i,r in enumerate(records): bins[int(r['force_bin'])].append(i)
    nonempty=[v for v in bins.values() if len(v)]
    target=max(len(v) for v in nonempty)
    # Avoid extreme duplication of a very rare class.
    out=[]
    for b in range(4):
        v=bins.get(b,[])
        if not v: continue
        n=min(target,max(len(v),int(2.5*len(v))))
        if n<=len(v): chosen=list(v)
        else: chosen=list(v)+[rng.choice(v) for _ in range(n-len(v))]
        out.extend(chosen)
    rng.shuffle(out)
    return out


def collate_v7f(records,idxs):
    # Real true mini-batch across terrains. pstd is sample-specific.
    return {
        'x':torch.stack([records[i]['x'].float() for i in idxs]).to(DEVICE,non_blocking=True),
        'baseF':torch.stack([records[i]['baseF'] for i in idxs]).to(DEVICE,non_blocking=True),
        'pcontact':torch.stack([records[i]['pcontact'] for i in idxs]).to(DEVICE,non_blocking=True),
        'regime':torch.stack([records[i]['regime'] for i in idxs]).to(DEVICE,non_blocking=True),
        'pstd_force':torch.stack([records[i]['pstd_force'] for i in idxs])[:,None,None,:].to(DEVICE,non_blocking=True),
        'gtF':torch.stack([records[i]['gtF'] for i in idxs]).to(DEVICE,non_blocking=True),
        'phase':torch.stack([records[i]['phase'] for i in idxs]).to(DEVICE,non_blocking=True),
    }

# ----------------------------------------------------------------------
# 6) SPECIALIST LOSS
# ----------------------------------------------------------------------
def v7f_force_loss(out,batch):
    pred=out['force']; gt=batch['gtF']; phase=batch['phase']
    pc=batch['pcontact']; pstd=batch['pstd_force']

    gtmag=torch.linalg.norm(gt,dim=-1)
    predmag=torch.linalg.norm(pred,dim=-1)

    # More weight for large-load/impact examples without letting them dominate infinitely.
    scale=max(float(REGIME_THRESHOLDS[-1]),1.0)
    phase_boost=((phase==1)|(phase==3)).float()
    w=(1.0 + 1.25*(gtmag/scale).clamp(0,2.0) + 0.50*phase_boost).detach()

    e=(pred-gt)/(pstd+1e-6)
    lvec=(F.smooth_l1_loss(e,torch.zeros_like(e),reduction='none',beta=.35).mean(-1)*w).mean()

    # Log magnitude makes both small and large forces learnable.
    lmag=F.smooth_l1_loss(torch.log1p(predmag),torch.log1p(gtmag),beta=.20)

    gtdir=F.normalize(gt,dim=-1,eps=1e-6)
    pdir=F.normalize(pred,dim=-1,eps=1e-6)
    contact=(gtmag>=float(CONTACT_THRESHOLD)).float()
    cos=(gtdir*pdir).sum(-1)
    ldir=((1.0-cos)*contact).sum()/(contact.sum()+1e-6)

    lphase=F.cross_entropy(out['phase_logits'].reshape(-1,N_PHASES),phase.reshape(-1))

    H=pred.shape[1]
    if H>1:
        # Model the entire future force curve jointly.
        dp=(pred[:,1:]-pred[:,:-1])/(pstd+1e-6)
        dg=(gt[:,1:]-gt[:,:-1])/(pstd+1e-6)
        ltemp=F.smooth_l1_loss(dp,dg,beta=.35)

        # Approximate integral consistency over the uniformly-spaced horizon queries.
        # This is a discrete force-trajectory regularizer, not claimed to be a measured impulse label.
        dt=0.1
        ip=(pred*dt).sum(1)
        ig=(gt*dt).sum(1)
        imp_scale=(pstd[:,0]+1e-6)
        limp=F.smooth_l1_loss((ip-ig)/imp_scale,torch.zeros_like(ip),beta=.35)
    else:
        ltemp=pred.sum()*0.0; limp=pred.sum()*0.0

    # Stay close to the strong V7 solution unless the data supports a correction.
    lanchor=(out['corr_norm']**2).mean()

    # Mild load-balancing regularizer for phase experts.
    mean_gate=out['expert_w'].mean(dim=(0,1,2))
    uniform=torch.full_like(mean_gate,1.0/len(mean_gate))
    lgent=((mean_gate-uniform)**2).mean()

    total=(
        V7F_W_VEC*lvec + V7F_W_MAG*lmag + V7F_W_DIR*ldir +
        V7F_W_PHASE*lphase + V7F_W_TEMP*ltemp + V7F_W_IMPULSE*limp +
        V7F_W_ANCHOR*lanchor + V7F_W_GATE_ENT*lgent
    )
    parts={
        'vec':float(lvec.detach().cpu()),'mag':float(lmag.detach().cpu()),
        'dir':float(ldir.detach().cpu()),'phase':float(lphase.detach().cpu()),
        'temp':float(ltemp.detach().cpu()),'imp':float(limp.detach().cpu()),
        'anchor':float(lanchor.detach().cpu()),
    }
    return total,parts


# ----------------------------------------------------------------------
# 7) TRUE MIXED-TERRAIN MINI-BATCH GLOBAL TRAINING
# ----------------------------------------------------------------------
def train_v7f_head(records,H,epochs,batch_size,seed,label):
    _seed_v7f(seed); model=V7ForceSpecialistHead().to(DEVICE)
    use_dp=torch.cuda.device_count()>=2
    runner=nn.DataParallel(model,device_ids=[0,1],output_device=0) if use_dp else model
    opt=torch.optim.AdamW(model.parameters(),lr=V7F_LR_GLOBAL,weight_decay=V7F_WEIGHT_DECAY)
    rng=random.Random(seed)

    safe_label=''.join(ch if ch.isalnum() else '_' for ch in str(label)).strip('_')
    recovery_path=RECOVERY_ROOT/f'_RECOVERY_V7F_{safe_label}_v2.pt'
    signature=dict(schema='V7F_SERVER_RESUME_20260926_v2',label=str(label),H=int(H),epochs=int(epochs),
                   batch_size=int(batch_size),seed=int(seed),n_records=int(len(records)),lr=float(V7F_LR_GLOBAL),
                   weight_decay=float(V7F_WEIGHT_DECAY),checkpoint_every_batches=int(V7F_CHECKPOINT_EVERY_BATCHES),
                   loss_weights=[V7F_W_VEC,V7F_W_MAG,V7F_W_DIR,V7F_W_PHASE,V7F_W_TEMP,V7F_W_IMPULSE,V7F_W_ANCHOR,V7F_W_GATE_ENT])
    resume_ep=0; resume_bi=0; resume_idxs=None; global_batches=0
    if recovery_path.exists():
        try:
            ck=torch.load(recovery_path,map_location=DEVICE,weights_only=False)
            if ck.get('signature')==signature:
                model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
                resume_ep=int(ck.get('epoch',ck.get('completed_epochs',0))); resume_bi=int(ck.get('next_batch_start',0))
                resume_idxs=ck.get('epoch_indices'); global_batches=int(ck.get('global_batches',0))
                if 'rng_state' in ck: rng.setstate(ck['rng_state'])
                _restore_rng_state(ck.get('torch_rng_state'))
                print(f'Resuming V7-F {label}: epoch={resume_ep}/{epochs}, next_batch_start={resume_bi}, global_batches={global_batches}')
            else: print('Ignoring incompatible V7-F recovery checkpoint:',recovery_path)
        except Exception as e: print('Ignoring unreadable V7-F recovery checkpoint:',e)

    def _save_pos(next_ep,next_bi,idxs):
        assert_disk_headroom('V7-F checkpoint')
        atomic_torch_save(dict(signature=signature,epoch=int(next_ep),next_batch_start=int(next_bi),
                               epoch_indices=idxs,global_batches=int(global_batches),model=model.state_dict(),
                               optimizer=opt.state_dict(),rng_state=rng.getstate(),torch_rng_state=_capture_rng_state()),recovery_path)

    print('\n'+'='*100); print('V7-F GLOBAL FORCE SPECIALIST',label); print('TRUE mixed-terrain mini-batches | frozen V7 trunk')
    print('GPU mode:','2-GPU DataParallel' if use_dp else 'single GPU'); print('='*100)
    for ep in range(resume_ep,epochs):
        model.train(); runner.train()
        if ep==resume_ep and resume_idxs is not None:
            idxs=list(resume_idxs); bi0=resume_bi
        else:
            idxs=balanced_epoch_indices(records,rng); bi0=0
        vals=[]; pacc=defaultdict(list)
        for bi in range(bi0,len(idxs),batch_size):
            ids=idxs[bi:bi+batch_size]; batch=collate_v7f(records,ids); opt.zero_grad(set_to_none=True)
            out=runner(batch['x'],batch['baseF'],batch['pstd_force'],batch['pcontact'],batch['regime'])
            loss,parts=v7f_force_loss(out,batch)
            if not torch.isfinite(loss): raise RuntimeError('V7-F non-finite global loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),V7F_GRAD_CLIP); opt.step(); global_batches+=1
            vals.append(float(loss.detach().cpu())); [pacc[k].append(v) for k,v in parts.items()]
            del batch,out,loss
            next_bi=min(bi+batch_size,len(idxs))
            if (global_batches % V7F_CHECKPOINT_EVERY_BATCHES)==0:
                _save_pos(ep,next_bi,idxs)
        msg=' '.join(f'{k}={np.mean(v):.4f}' for k,v in pacc.items())
        print(f' epoch {ep+1}/{epochs} batches={math.ceil(len(idxs)/batch_size)} executed_loss={np.mean(vals) if vals else float("nan"):.4f} | {msg}')
        _save_pos(ep+1,0,None)
        resume_bi=0; resume_idxs=None
    print('V7-F complete. Durable recovery checkpoint:',recovery_path)
    return model


# ----------------------------------------------------------------------
# 8) LOCAL CAUSAL ADAPTATION — SPECIALIST ONLY, WITH PRE-TEST VALIDATION
# ----------------------------------------------------------------------
@torch.inference_mode()
def _make_local_force_item(base,rec,idx,H,horizons,stats,norm,physics):
    s=make_v7_sample(rec['data'],rec['visual'],stats,physics,idx,H,horizons,norm)
    b=v7f_extract_base_context(base,s,stats)
    return {
        'x':b['x'][0].half().cpu(),
        'baseF':b['baseF'][0].float().cpu(),
        'pcontact':b['pcontact'][0].float().cpu(),
        'regime':b['regime'][0].float().cpu(),
        'pstd_force':torch.as_tensor(stats['P_std'][9:12],dtype=torch.float32),
        'gtF':torch.as_tensor(np.asarray(s['target_p_raw'][...,9:12],np.float32)),
        'phase':torch.as_tensor(v7f_phase_targets(rec['data'],idx,horizons),dtype=torch.long),
    }


def adapt_v7f_head(global_head,base,rec,ttest,context_sec,hop_sec,horizons,half_life,seed,label):
    _seed_v7f(seed)
    head=copy.deepcopy(global_head).to(DEVICE)

    # IMPORTANT: use the same run-level causal normalization that the frozen global V7
    # base saw during prior training; do not silently change its input distribution.
    end0=float(min(rec['ttests']))
    stats=make_stats_between(rec['data'],float(rec['data']['time'][0]),end0)
    norm=normalize_arrays(rec['data'],stats)
    physics=rec['physics_prior_v7']

    wins,H=make_local_windows_v7(
        rec['data'],rec['visual'],ttest,context_sec,hop_sec,
        max(horizons),half_life
    )
    items=[]
    for idx,w,_,te in wins:
        it=_make_local_force_item(base,rec,idx,H,horizons,stats,norm,physics)
        it['w']=float(w); it['te']=float(te); items.append(it)

    # Causal split entirely BEFORE ttest. Newest 20% is validation, never trained on.
    n=len(items)
    nval=max(1,int(round(n*V7F_LOCAL_VAL_FRAC))) if n>=5 else 1
    train_items=items[:-nval] if n>1 else items
    val_items=items[-nval:] if n>1 else items
    if not train_items: train_items=items

    opt=torch.optim.AdamW(head.parameters(),lr=V7F_LR_LOCAL,weight_decay=V7F_WEIGHT_DECAY)
    best=copy.deepcopy(head.state_dict()); best_val=float('inf')

    def one_item_batch(it): return collate_v7f([it],[0])

    print(f' V7-F LOCAL {label} t={ttest:.1f}: total={n} train={len(train_items)} val={len(val_items)}')
    for ep in range(V7F_LOCAL_EPOCHS):
        head.train(); train_losses=[]
        # preserve chronological order and recency weighting
        for it in train_items:
            b=one_item_batch(it)
            opt.zero_grad(set_to_none=True)
            o=head(b['x'],b['baseF'],b['pstd_force'],b['pcontact'],b['regime'])
            raw,_=v7f_force_loss(o,b); loss=raw*float(it['w'])
            loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(),V7F_GRAD_CLIP); opt.step()
            train_losses.append(float(raw.detach().cpu())); del b,o,raw,loss

        head.eval(); vals=[]
        with torch.inference_mode():
            for it in val_items:
                b=one_item_batch(it)
                o=head(b['x'],b['baseF'],b['pstd_force'],b['pcontact'],b['regime'])
                raw,_=v7f_force_loss(o,b); vals.append(float(raw.detach().cpu())); del b,o,raw
        v=float(np.mean(vals))
        if v<best_val:
            best_val=v; best=copy.deepcopy(head.state_dict())
        print(f'   epoch {ep+1}/{V7F_LOCAL_EPOCHS} train={np.mean(train_losses):.4f} causal_val={v:.4f}')

    head.load_state_dict(best)
    return head,stats,physics,H


# ----------------------------------------------------------------------
# 9) PREDICT FORCE ONLY — PRESERVE ALL SUCCESSFUL V7 NON-FORCE RESULTS
# ----------------------------------------------------------------------
@torch.inference_mode()
def predict_v7f_force(base,head,rec,ttest,context_sec,horizons,stats,physics,H):
    idx=nearest_index(rec['data']['time'],ttest)
    norm=normalize_arrays(rec['data'],stats)
    s=make_v7_sample(rec['data'],rec['visual'],stats,physics,idx,H,horizons,norm)
    b=v7f_extract_base_context(base,s,stats)
    pstd=torch.as_tensor(stats['P_std'][9:12],dtype=torch.float32,device=DEVICE).view(1,1,1,3)
    o=head(b['x'],b['baseF'],pstd,b['pcontact'],b['regime'])
    force=o['force'][0].cpu().numpy()
    phase_prob=o['phase_prob'][0].cpu().numpy()
    alpha=o['fusion_alpha'][0].cpu().numpy()
    return force,phase_prob,alpha,s,idx


def replace_force_rows_from_v7(v7_df,force_predictions):
    out=v7_df.copy()
    for key,pred in force_predictions.items():
        terrain,test_id,h=key
        for li,L in enumerate(LEG_NAMES):
            for ci,axis in enumerate(('Fx','Fy','Fz')):
                sig=f'{L}.{axis}'
                m=(out.terrain_type==terrain)&(out.test_id==test_id)&(np.isclose(out.horizon_s,h))&(out.signal==sig)
                if int(m.sum())!=1:
                    raise RuntimeError(f'Expected one V7 result row for {key} {sig}; got {m.sum()}')
                gt=float(out.loc[m,'ground_truth'].iloc[0]); p=float(pred[li,ci])
                out.loc[m,'prediction']=p
                out.loc[m,'signed_error']=p-gt
                out.loc[m,'abs_error']=abs(p-gt)
                if abs(h-.1)<1e-8: out.loc[m,'short_native_pred']=p
                else: out.loc[m,'long_direct_pred']=p
    return out


def force_vector_metrics_from_df(df):
    rows=[]
    for keys,g in df[df.signal.map(signal_group_v7)=='force'].groupby(['terrain_type','test_id','horizon_s']):
        terrain,tid,h=keys
        for L in LEG_NAMES:
            gg=g[g.signal.str.startswith(L+'.')]
            # deterministic Fx,Fy,Fz order
            p=[]; y=[]
            for axis in ('Fx','Fy','Fz'):
                r=gg[gg.signal==f'{L}.{axis}']
                if len(r)!=1: continue
                p.append(float(r.prediction.iloc[0])); y.append(float(r.ground_truth.iloc[0]))
            if len(p)!=3: continue
            p=np.asarray(p); y=np.asarray(y); pm=np.linalg.norm(p); ym=np.linalg.norm(y)
            cos=float(np.dot(p,y)/(pm*ym+1e-9)) if ym>=CONTACT_THRESHOLD and pm>1e-6 else np.nan
            rows.append(dict(terrain_type=terrain,test_id=tid,horizon_s=h,leg=L,
                             force_vector_error=float(np.linalg.norm(p-y)),
                             force_magnitude_MAE=float(abs(pm-ym)),
                             force_direction_cosine=cos))
    raw=pd.DataFrame(rows)
    summary=raw.groupby('horizon_s',as_index=False).agg(
        force_vector_error=('force_vector_error','mean'),
        force_magnitude_MAE=('force_magnitude_MAE','mean'),
        force_direction_cosine=('force_direction_cosine','mean'))
    return raw,summary

# ----------------------------------------------------------------------
# 10) RUN THE V7-F EXPERIMENT
# ----------------------------------------------------------------------
print('\n'+'#'*100)
print('V7-F: BUILDING DENSE FROZEN-TRUNK GLOBAL FORCE DATASETS')
print('#'*100)

# Keep the successful V7 priors frozen. The new head learns only force corrections.
for p in prior_short.parameters(): p.requires_grad_(False)
for p in prior_long.parameters(): p.requires_grad_(False)
prior_short.eval(); prior_long.eval()

short_cache=build_v7f_global_cache(
    prior_short,SHORT_CONTEXT_SEC,V7F_GLOBAL_HOP_SHORT,SHORT_HORIZONS,'SHORT +0.1s'
)
long_cache=build_v7f_global_cache(
    prior_long,LONG_CONTEXT_SEC,V7F_GLOBAL_HOP_LONG,LONG_HORIZONS,'LONG +0.2..+1.0s'
)

v7f_head_short=train_v7f_head(
    short_cache,1,V7F_EPOCHS_SHORT,V7F_BATCH_SHORT,V7F_SEED+1,'SHORT +0.1s'
)
v7f_head_long=train_v7f_head(
    long_cache,len(LONG_HORIZONS),V7F_EPOCHS_LONG,V7F_BATCH_LONG,V7F_SEED+2,'LONG +0.2..+1.0s'
)

atomic_torch_save(v7f_head_short.state_dict(),V7F_ROOT/'V7F_FORCE_HEAD_SHORT.pt')
atomic_torch_save(v7f_head_long.state_dict(),V7F_ROOT/'V7F_FORCE_HEAD_LONG.pt')

# Free the large CPU feature caches before the 16 local tests.
del short_cache,long_cache
gc.collect(); torch.cuda.empty_cache()

print('\n'+'#'*100)
print('V7-F: 16 CAUSAL LOCAL FORCE-HEAD ADAPTATIONS + TESTS')
print('#'*100)

_v7f_test_recovery=RECOVERY_ROOT/'_RECOVERY_V7F_16_LOCAL_TESTS_v2.pt'
_v7f_test_signature=dict(
    schema='V7F_LOCAL_TEST_RESUME_20260926_v2',
    head_short=model_state_fingerprint(v7f_head_short),head_long=model_state_fingerprint(v7f_head_long),
    prior_short=model_state_fingerprint(prior_short),prior_long=model_state_fingerprint(prior_long),
    tests=[(r['terrain_type'],[(i+1,float(t)) for i,t in enumerate(r['ttests'])]) for r in run_records],
    local_epochs=int(V7F_LOCAL_EPOCHS),lr_local=float(V7F_LR_LOCAL),
)
force_predictions={}; phase_rows=[]; _v7f_done=set()
if _v7f_test_recovery.exists():
    try:
        _ck=torch.load(_v7f_test_recovery,map_location='cpu',weights_only=False)
        if _ck.get('signature')==_v7f_test_signature:
            force_predictions=_ck.get('force_predictions',{}); phase_rows=_ck.get('phase_rows',[])
            _v7f_done={tuple(x) for x in _ck.get('done',[])}
            print(f'Resuming V7-F local tests: {len(_v7f_done)}/16 complete.')
    except Exception as e: print('Ignoring unreadable V7-F test recovery:',e)
for ri,rec in enumerate(run_records):
    for ti,tt in enumerate(rec['ttests'],1):
        _key=(rec['terrain_type'],int(ti),float(tt))
        if _key in _v7f_done:
            print('Skipping completed V7-F local test:',_key); continue
        print('\n'+'-'*100); print(f'V7-F TEST {rec["terrain_type"]} #{ti} t={tt:.1f}'); print('-'*100)
        seed=V7F_SEED+ri*1000+ti*100
        sh,ss,sp,Hs=adapt_v7f_head(v7f_head_short,prior_short,rec,tt,SHORT_CONTEXT_SEC,V7F_LOCAL_HOP_SHORT,SHORT_HORIZONS,SHORT_RECENCY_HALF_LIFE_SEC,seed+1,'SHORT')
        lh,ls,lp,Hl=adapt_v7f_head(v7f_head_long,prior_long,rec,tt,LONG_CONTEXT_SEC,V7F_LOCAL_HOP_LONG,LONG_HORIZONS,LONG_RECENCY_HALF_LIFE_SEC,seed+2,'LONG')
        sf,sphase,salpha,ssamp,idx=predict_v7f_force(prior_short,sh,rec,tt,SHORT_CONTEXT_SEC,SHORT_HORIZONS,ss,sp,Hs)
        lf,lphase,lalpha,lsamp,_=predict_v7f_force(prior_long,lh,rec,tt,LONG_CONTEXT_SEC,LONG_HORIZONS,ls,lp,Hl)
        force_predictions[(rec['terrain_type'],ti,0.1)]=sf[0]
        for hi,h in enumerate(LONG_HORIZONS): force_predictions[(rec['terrain_type'],ti,float(h))]=lf[hi]
        for h,pp,aa in [(0.1,sphase[0],salpha[0])]+[(float(h),lphase[j],lalpha[j]) for j,h in enumerate(LONG_HORIZONS)]:
            for leg,L in enumerate(LEG_NAMES):
                phase_rows.append(dict(terrain_type=rec['terrain_type'],test_id=ti,ttest_s=float(tt),horizon_s=h,leg=L,
                                       phase_pred=PHASE_NAMES[int(np.argmax(pp[leg]))],phase_confidence=float(np.max(pp[leg])),fusion_alpha_direct=float(aa[leg])))
        _v7f_done.add(_key)
        atomic_torch_save(dict(signature=_v7f_test_signature,force_predictions=force_predictions,phase_rows=phase_rows,
                               done=[list(x) for x in sorted(_v7f_done)]),_v7f_test_recovery)
        print(' Saved V7-F local-test recovery:',_v7f_test_recovery)
        del sh,lh,ssamp,lsamp; gc.collect(); torch.cuda.empty_cache()
if len(_v7f_done)!=16: raise RuntimeError(f'V7-F local-test recovery expected 16 complete tests, got {len(_v7f_done)}')

# ----------------------------------------------------------------------
# 11) MERGE WITH V7 RESULTS AND EVALUATE
# ----------------------------------------------------------------------
v7f_all=replace_force_rows_from_v7(all_df,force_predictions)
v7f_grouped,v7f_mean=summarize_metrics(v7f_all)
v7f_acceptance=acceptance_table(v7f_mean)
v7f_force_raw,v7f_force_summary=force_vector_metrics_from_df(v7f_all)
v7f_phase_df=pd.DataFrame(phase_rows)

# Direct V7 vs V7-F force comparison at every horizon.
v7_mean_force=(all_df.assign(signal_group=all_df.signal.map(signal_group_v7))
               .query("signal_group == 'force'")
               .groupby('horizon_s',as_index=False).abs_error.mean()
               .rename(columns={'abs_error':'V7_force_MAE'}))
f_mean_force=(v7f_all.assign(signal_group=v7f_all.signal.map(signal_group_v7))
              .query("signal_group == 'force'")
              .groupby('horizon_s',as_index=False).abs_error.mean()
              .rename(columns={'abs_error':'V7F_force_MAE'}))
v7f_vs_v7=v7_mean_force.merge(f_mean_force,on='horizon_s')
v7f_vs_v7['improvement_percent']=100*(v7f_vs_v7.V7_force_MAE-v7f_vs_v7.V7F_force_MAE)/(v7f_vs_v7.V7_force_MAE+1e-12)

# Save everything.
v7f_all.to_csv(V7F_ROOT/'V7F_ALL_16_TESTS_ALL_SENSOR_VALUES.csv',index=False)
v7f_grouped.to_csv(V7F_ROOT/'V7F_ALL_16_TESTS_GROUPED_METRICS.csv',index=False)
v7f_mean.to_csv(V7F_ROOT/'V7F_MEAN_METRICS_ACROSS_16_TESTS.csv',index=False)
v7f_acceptance.to_csv(V7F_ROOT/'V7F_ACCEPTANCE_VS_BEST_EXISTING.csv',index=False)
v7f_force_raw.to_csv(V7F_ROOT/'V7F_FORCE_VECTOR_RAW.csv',index=False)
v7f_force_summary.to_csv(V7F_ROOT/'V7F_FORCE_VECTOR_METRICS_BY_HORIZON.csv',index=False)
v7f_phase_df.to_csv(V7F_ROOT/'V7F_CONTACT_PHASE_AND_FUSION.csv',index=False)
v7f_vs_v7.to_csv(V7F_ROOT/'V7F_VS_V7_FORCE_BY_HORIZON.csv',index=False)

cfg=dict(
    global_hop_short=V7F_GLOBAL_HOP_SHORT,global_hop_long=V7F_GLOBAL_HOP_LONG,
    batch_short=V7F_BATCH_SHORT,batch_long=V7F_BATCH_LONG,
    epochs_short=V7F_EPOCHS_SHORT,epochs_long=V7F_EPOCHS_LONG,
    lr_global=V7F_LR_GLOBAL,lr_local=V7F_LR_LOCAL,local_epochs=V7F_LOCAL_EPOCHS,
    local_val_frac=V7F_LOCAL_VAL_FRAC,
    architecture='frozen V7 trunk + joint-horizon phase-MoE force specialist + direct/magnitude-direction fusion',
    note='No V5 test prediction used for training; V5 is evaluation reference only.'
)
(V7F_ROOT/'V7F_CONFIG.json').write_text(json.dumps(cfg,indent=2))

# Plot separate force-MAE figure.
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7,4))
    plt.plot(v7f_vs_v7.horizon_s,v7f_vs_v7.V7_force_MAE,marker='o',label='V7')
    plt.plot(v7f_vs_v7.horizon_s,v7f_vs_v7.V7F_force_MAE,marker='o',label='V7-F')
    plt.xlabel('prediction horizon (s)'); plt.ylabel('force MAE (N)')
    plt.title('V7-F force specialist vs V7'); plt.grid(True,alpha=.25); plt.legend(); plt.tight_layout()
    plt.savefig(V7F_ROOT/'V7F_VS_V7_FORCE_MAE.png',dpi=170); plt.show()
except Exception as e:
    print('Plot warning:',e)

# Integrity checks.
if len(v7f_all)!=len(all_df): raise RuntimeError('V7-F result row count changed unexpectedly.')
if len(force_predictions)!=16*10: raise RuntimeError(f'Expected 160 test/horizon force predictions, got {len(force_predictions)}')

# Verify that non-force V7 predictions are bit-for-bit unchanged.
_nonforce=all_df.signal.map(signal_group_v7)!='force'
if not np.allclose(
    v7f_all.loc[_nonforce,'prediction'].to_numpy(float),
    all_df.loc[_nonforce,'prediction'].to_numpy(float),
    equal_nan=True
):
    raise RuntimeError('V7-F unexpectedly changed a non-force V7 prediction.')

print('\nV7-F FORCE IMPROVEMENT VS V7')
display(v7f_vs_v7)
print('\nV7-F ACCEPTANCE VS BEST EXISTING V5')
display(v7f_acceptance)
print('\nV7-F FORCE VECTOR METRICS')
display(v7f_force_summary)

# ----------------------------------------------------------------------
# 12) CHECKSUM + ZIP THE SMALL V7-F RESULTS (visual caches stay persistent)
# ----------------------------------------------------------------------
manifest=[]
for p in sorted(V7F_ROOT.rglob('*')):
    if p.is_file() and p.name!='V7F_SHA256_MANIFEST.csv':
        h=hashlib.sha256()
        with open(p,'rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):
                h.update(chunk)
        manifest.append(dict(file=str(p.relative_to(V7F_ROOT)),sha256=h.hexdigest(),size_bytes=p.stat().st_size))
pd.DataFrame(manifest).to_csv(V7F_ROOT/'V7F_SHA256_MANIFEST.csv',index=False)

zip_path=WORK_ROOT/'V7F_FORCE_SPECIALIST_RESULTS.zip'
if zip_path.exists(): zip_path.unlink()
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED,allowZip64=True) as z:
    for p in sorted(V7F_ROOT.rglob('*')):
        if p.is_file(): z.write(p,arcname=str(p.relative_to(V7F_ROOT.parent)))

print('\n'+'='*100)
print('V7-F EXPERIMENT COMPLETE')
if 'NOTEBOOK_WALL_T0' in globals():
    print(f'WALL TIME FROM FIRST INSTALL CELL THROUGH V7-F: {(time.perf_counter()-NOTEBOOK_WALL_T0)/60:.2f} min')
print('Results folder:',V7F_ROOT)
print('Download ZIP  :',zip_path)
print('Main decision : open V7F_ACCEPTANCE_VS_BEST_EXISTING.csv')
print('='*100)

try:
    from google.colab import files as colab_files
    colab_files.download(str(zip_path))
except Exception as e:
    print('Automatic download skipped:',e)

# %% notebook cell 16
# ======================================================================
# V8-A — PERSISTENT FOUR-STREAM CONTACT MEMORY
# APPEND-ONLY CELL. Run AFTER completed V7 + V7-F in the same Colab session.
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
    'CACHE_ROOT','PERSIST_ROOT','DEVICE','RAW_GRID','FX','FY','CX','CY'
]
_missing=[x for x in _req if x not in globals()]
if _missing:
    raise RuntimeError('V8-A needs the completed V7 live session. Missing: '+repr(_missing))

V8_ROOT = PERSIST_ROOT/'V8_contactmap'
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
        dd,ii=self.tree.query(xy,k=min(k,len(self.xy)),distance_upper_bound=radius)
        dd=np.atleast_1d(dd); ii=np.atleast_1d(ii)
        ok=np.isfinite(dd)&(ii<len(self.xy))
        if not np.any(ok): return None
        dd=dd[ok]; ii=ii[ok]
        causal=self.time[ii] <= float(t)+1e-8
        if not np.any(causal): return None
        dd=dd[causal]; ii=ii[causal]
        # Prefer the freshest observation, mildly penalized by spatial distance.
        age=np.maximum(0,float(t)-self.time[ii])
        score=age + 1.5*dd
        j=int(np.argmin(score)); q=int(ii[j])
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

# %% notebook cell 17
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
best=None; bestv=float('inf'); _v8b_start=0
_v8b_rec=RECOVERY_ROOT/'_RECOVERY_V8B_EPOCH_v2.pt'
_v8b_sig=dict(schema='V8B_SERVER_RESUME_20260926_v2',epochs=int(V8B_EPOCHS),lr=float(V8B_LR),batch=int(V8B_BATCH),
              horizons=[float(x) for x in V8B_HORIZONS],n=int(len(Xb)),seed=int(V8B_SEED))
if _v8b_rec.exists():
    try:
        ck=torch.load(_v8b_rec,map_location=DEVICE,weights_only=False)
        if ck.get('signature')==_v8b_sig:
            v8_touchdown_model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
            best=ck.get('best'); bestv=float(ck.get('bestv',float('inf'))); _v8b_start=int(ck.get('completed_epochs',0))
            _restore_rng_state(ck.get('rng_state')); print('Resuming V8-B from epoch',_v8b_start,'/',V8B_EPOCHS)
    except Exception as e: print('Ignoring unreadable V8-B recovery:',e)
for ep in range(_v8b_start,V8B_EPOCHS):
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
    atomic_torch_save(dict(signature=_v8b_sig,completed_epochs=ep+1,model=v8_touchdown_model.state_dict(),optimizer=opt.state_dict(),
                           best=best,bestv=bestv,rng_state=_capture_rng_state()),_v8b_rec)
if best is None: raise RuntimeError('V8-B best checkpoint is empty.')
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
atomic_torch_save(dict(state_dict=v8_touchdown_model.state_dict(),mean=V8B_MEAN,std=V8B_STD,horizons=V8B_HORIZONS),V8B_ROOT/'V8B_TOUCHDOWN_MODEL.pt')

print('\nV8-B HELD-OUT TOUCHDOWN METRICS')
display(v8b_summary)
print('Median first-contact timing error [s]:',float(v8b_timing.timing_abs_error_s.median()))
print('V8-B COMPLETE:',V8B_ROOT)

# %% notebook cell 18
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
bestv=float('inf'); best=None; _v8c_start=0
_v8c_rec=RECOVERY_ROOT/'_RECOVERY_V8C_EPOCH_v2.pt'
_v8c_sig=dict(schema='V8C_SERVER_RESUME_20260926_v2',epochs=int(V8C_EPOCHS),lr=float(V8C_LR),batch=int(V8C_BATCH),
              experts=int(V8C_N_EXPERTS),n=int(len(V8C_X)),seed=int(V8C_SEED))
if _v8c_rec.exists():
    try:
        ck=torch.load(_v8c_rec,map_location=DEVICE,weights_only=False)
        if ck.get('signature')==_v8c_sig:
            v8_contact_field.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
            best=ck.get('best'); bestv=float(ck.get('bestv',float('inf'))); _v8c_start=int(ck.get('completed_epochs',0))
            _restore_rng_state(ck.get('rng_state')); print('Resuming V8-C from epoch',_v8c_start,'/',V8C_EPOCHS)
    except Exception as e: print('Ignoring unreadable V8-C recovery:',e)
for ep in range(_v8c_start,V8C_EPOCHS):
    v8_contact_field.train(); tl=[]
    for x,y in trdl:
        x=x.to(DEVICE); y=y.to(DEVICE); opt.zero_grad(set_to_none=True); o=v8_contact_field(x); l=_v8c_loss(o,y); l.backward(); torch.nn.utils.clip_grad_norm_(v8_contact_field.parameters(),1.0); opt.step(); tl.append(float(l.detach().cpu()))
    v8_contact_field.eval(); vl=[]
    with torch.inference_mode():
        for x,y in vadl:
            x=x.to(DEVICE); y=y.to(DEVICE); vl.append(float(_v8c_loss(v8_contact_field(x),y).cpu()))
    vm=float(np.mean(vl)); print(f'V8-C epoch {ep+1:02d}/{V8C_EPOCHS} train={np.mean(tl):.4f} val={vm:.4f}')
    if vm<bestv: bestv=vm; best=copy.deepcopy(v8_contact_field.state_dict())
    atomic_torch_save(dict(signature=_v8c_sig,completed_epochs=ep+1,model=v8_contact_field.state_dict(),optimizer=opt.state_dict(),
                           best=best,bestv=bestv,rng_state=_capture_rng_state()),_v8c_rec)
if best is None: raise RuntimeError('V8-C best checkpoint is empty.')
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
                rows.append(dict(terrain_type=rec['terrain_type'],test_index=ti,ttest=float(tt),horizon_s=float(h),leg=leg,
                                 memory_age_s=mq['age_s'],memory_distance_m=mq['distance_m'],
                                 tau_MAE=float(np.mean(np.abs(pr['tau']-gt_tau))),force_MAE=float(np.mean(np.abs(pr['force']-gt_f))),
                                 force_vector_error=float(np.linalg.norm(pr['force']-gt_f)),force_sigma_mean=float(pr['force_sigma'].mean())))

v8c_eval=pd.DataFrame(rows)
v8c_summary=v8c_eval.groupby('horizon_s').agg(samples=('force_MAE','size'),tau_MAE=('tau_MAE','mean'),force_MAE=('force_MAE','mean'),
                                               force_vector_error=('force_vector_error','mean'),force_sigma_mean=('force_sigma_mean','mean')).reset_index()
v8c_eval.to_csv(V8C_ROOT/'V8C_16TEST_CONTACT_FIELD_RAW.csv',index=False)
v8c_summary.to_csv(V8C_ROOT/'V8C_CONTACT_FIELD_METRICS_BY_HORIZON.csv',index=False)
atomic_torch_save(dict(state_dict=v8_contact_field.state_dict(),xmean=V8C_XMEAN,xstd=V8C_XSTD,ymean=V8C_YMEAN,ystd=V8C_YSTD,horizons=V8C_HORIZONS),V8C_ROOT/'V8C_CONTACT_FIELD.pt')

print('\nV8-C HELD-OUT QUERYABLE FIELD METRICS')
display(v8c_summary)
print('V8-C COMPLETE:',V8C_ROOT)

# %% notebook cell 19
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
v8_delta_model=V8SiameseDelta(DX.shape[1]).to(DEVICE)
opt=torch.optim.AdamW(v8_delta_model.parameters(),lr=V8D_LR,weight_decay=1e-4)
bestv=float('inf'); best=None; _v8d_start=0
_v8d_rec=RECOVERY_ROOT/'_RECOVERY_V8D_EPOCH_v2.pt'
_v8d_sig=dict(schema='V8D_SERVER_RESUME_20260926_v2',epochs=int(V8D_EPOCHS),lr=float(V8D_LR),batch=int(V8D_BATCH),
              n=int(len(DX)),input_dim=int(DX.shape[1]),seed=int(V8D_SEED))
if _v8d_rec.exists():
    try:
        ck=torch.load(_v8d_rec,map_location=DEVICE,weights_only=False)
        if ck.get('signature')==_v8d_sig:
            v8_delta_model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
            best=ck.get('best'); bestv=float(ck.get('bestv',float('inf'))); _v8d_start=int(ck.get('completed_epochs',0))
            _restore_rng_state(ck.get('rng_state')); print('Resuming V8-D from epoch',_v8d_start,'/',V8D_EPOCHS)
    except Exception as e: print('Ignoring unreadable V8-D recovery:',e)
for ep in range(_v8d_start,V8D_EPOCHS):
    v8_delta_model.train(); tl=[]
    for x,y in trdl:
        x=x.to(DEVICE); y=y.to(DEVICE); opt.zero_grad(set_to_none=True); o=v8_delta_model(x); l=_loss(o,y); l.backward(); torch.nn.utils.clip_grad_norm_(v8_delta_model.parameters(),1.0); opt.step(); tl.append(float(l.detach().cpu()))
    v8_delta_model.eval(); vl=[]
    with torch.inference_mode():
        for x,y in vadl: vl.append(float(_loss(v8_delta_model(x.to(DEVICE)),y.to(DEVICE)).cpu()))
    vm=float(np.mean(vl)); print(f'V8-D epoch {ep+1:02d}/{V8D_EPOCHS} train={np.mean(tl):.4f} val={vm:.4f}')
    if vm<bestv: bestv=vm; best=copy.deepcopy(v8_delta_model.state_dict())
    atomic_torch_save(dict(signature=_v8d_sig,completed_epochs=ep+1,model=v8_delta_model.state_dict(),optimizer=opt.state_dict(),
                           best=best,bestv=bestv,rng_state=_capture_rng_state()),_v8d_rec)
if best is None: raise RuntimeError('V8-D best checkpoint is empty.')
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
atomic_torch_save(dict(state_dict=v8_delta_model.state_dict(),dy_mean=DYM,dy_std=DYS,input_dim=DX.shape[1]),V8D_ROOT/'V8D_SIAMESE_DELTA.pt')

print('\nV8-D MATCHED-STATE COUNTERFACTUAL METRICS')
display(v8d_summary)
print('V8-D COMPLETE:',V8D_ROOT)

# %% notebook cell 20
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

# %% notebook cell 21
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

_req=['WORK_ROOT','run_records','v8_memory','v8_foot_world_proxy','v8c_predict_cell','V8_ROOT','v8e_candidates']
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

# --------------------------- safe-curiosity overlay ---------------------------
voi_rows=[]
if len(v8e_candidates):
    for keys,g in v8e_candidates.groupby(['terrain_type','test_index','ttest','leg','horizon_s']):
        g=g.copy().sort_values('score'); safest=g.iloc[0]
        # Approximate information gain from predictive uncertainty, map age, and low observation confidence.
        g['info_gain']=0.5*np.log1p((g['unc_norm'].clip(lower=0)/15.0)**2)*(1.0+g['map_age_s'].clip(0,5)/5.0)
        # Safe exploration gate: never explore a candidate much riskier than the current safest one.
        safe=g[g['score']<=float(safest.score)*1.12+0.02].copy()
        safe['active_score']=safe['score']-V8F_INFO_LAMBDA*safe['info_gain']
        active=safe.sort_values('active_score').iloc[0]
        voi_rows.append(dict(terrain_type=keys[0],test_index=keys[1],ttest=keys[2],leg=keys[3],horizon_s=keys[4],
                             safest_score=float(safest.score),active_score=float(active.score),
                             safest_info_gain=float(safest.info_gain),active_info_gain=float(active.info_gain),
                             exploration_offset_m=float(active.offset_m),same_as_safest=bool(active.name==safest.name)))
v8f_voi=pd.DataFrame(voi_rows)

v8f_replay.to_csv(V8F_ROOT/'V8F_CAUSAL_CONTACT_REPLAY_RAW.csv',index=False)
v8f_curve.to_csv(V8F_ROOT/'V8F_ADAPTATION_CURVE.csv',index=False)
v8f_voi.to_csv(V8F_ROOT/'V8F_SAFE_CURIOSITY_SHADOW.csv',index=False)

print('\nV8-F CONTACT ADAPTATION CURVE')
display(v8f_curve)
print('Overall force negative-transfer rate:',float((~v8f_replay.improved_force).mean()))
if len(v8f_voi):
    print('\nV8-F SAFE-CURIOSITY SHADOW SUMMARY')
    display(pd.DataFrame([dict(groups=len(v8f_voi),fraction_different_from_safest=float((~v8f_voi.same_as_safest).mean()),
                               mean_extra_information=float((v8f_voi.active_info_gain-v8f_voi.safest_info_gain).mean()))]))

# --------------------------- package every V8 result ---------------------------
readme=V8_ROOT/'README_V8_A_TO_F.txt'
readme.write_text('''V8 A-F research prototype\n\nA: persistent causal four-stream terrain memory\nB: probabilistic future per-leg touchdown predictor\nC: queryable local contact-response field\nD: matched-state Siamese terrain-change head\nE: causal micro-foothold planner in OFFLINE SHADOW MODE\nF: bounded causal contact adaptation + safe-curiosity shadow policy\n\nScientific limitation: TartanGround logs do not contain executed alternative footholds. E/F counterfactual planner outputs are predictions and offline replay diagnostics, not real closed-loop navigation evidence. Exact robot deployment also requires a verified camera-to-base extrinsic.\n''')

manifest=[]
for p in sorted(V8_ROOT.rglob('*')):
    if p.is_file():
        h=hashlib.sha256(p.read_bytes()).hexdigest(); manifest.append(dict(file=str(p.relative_to(V8_ROOT)),sha256=h,size_bytes=p.stat().st_size))
pd.DataFrame(manifest).to_csv(V8_ROOT/'V8_SHA256_MANIFEST.csv',index=False)
zip_path=WORK_ROOT/'V8_A_TO_F_RESULTS.zip'
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

# %% notebook cell 22
# ======================================================================
# APPEND CELL 1 — 2x RTX 4080 SAME-RUN TEMPORAL-GAP SWEEP
#
# Purpose:
#   For every original V7/V7-F test point, force the latest gradient-training
#   INPUT anchor to be 4 s / 7 s / 9 s / 10 s before ttest.
#
# Two-GPU policy:
#   Independent (terrain, ttest, gap) jobs are split across cuda:0 and cuda:1.
#   This preserves the local optimizer semantics while actually using both GPUs.
#
# Leakage policy:
#   NO training and NO validation/model-selection information after the requested
#   cutoff. Therefore "gap=9 s" really means the specialist receives no gradient
#   or checkpoint-selection information from the final 9 seconds before ttest.
# ======================================================================
import os, gc, copy, json, math, random, hashlib, zipfile, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import torch

if torch.cuda.device_count() < 2:
    raise RuntimeError(
        f"2 visible CUDA GPUs are required; found {torch.cuda.device_count()}."
    )

GPU_DEVICES = [torch.device('cuda:0'), torch.device('cuda:1')]
print("Dual-GPU gap sweep")
for i,d in enumerate(GPU_DEVICES):
    print(i, torch.cuda.get_device_name(i))

_required = [
    'run_records','prior_short','prior_long','v7f_head_short','v7f_head_long','PERSIST_ROOT','WORK_ROOT',
    'SHORT_CONTEXT_SEC','LONG_CONTEXT_SEC','SHORT_HORIZONS','LONG_HORIZONS',
    'V7F_LOCAL_HOP_SHORT','V7F_LOCAL_HOP_LONG',
    'SHORT_RECENCY_HALF_LIFE_SEC','LONG_RECENCY_HALF_LIFE_SEC',
    'TRAIN_LOOKBACK_SEC','V7F_LOCAL_EPOCHS','V7F_LR_LOCAL',
    'V7F_WEIGHT_DECAY','V7F_GRAD_CLIP','CONTACT_THRESHOLD',
    'make_stats_between','normalize_arrays','nearest_index','horizon_offsets',
    'physics_tau_np','physics_acc_np','force_regime_np','ANYMAL_D_FK',
    'FOOT_QUERY_PLAN_SEC','N_REGIMES','PI_PHYS_DIM',
    'v7f_extract_base_context','v7f_force_loss','v7f_phase_targets',
    'LEG_NAMES','V7F_SEED'
]
_missing=[x for x in _required if x not in globals()]
if _missing:
    raise RuntimeError("Run the completed V7 + V7-F notebook first. Missing: "+repr(_missing))

GAP_SWEEP_SEC=[4.0,7.0,9.0,10.0]
GAP_ROOT=PERSIST_ROOT/'V7F_same_run_gap_sweep_2GPU'
GAP_ROOT.mkdir(parents=True,exist_ok=True)
print('Resume-safe result root:',GAP_ROOT)

# ----------------------------------------------------------------------
# Device-aware copies of the exact sample builder.
# They avoid the notebook-global DEVICE variable so two worker threads can
# safely execute on different GPUs at the same time.
# ----------------------------------------------------------------------
def _tensor_dev(x,device):
    return torch.as_tensor(x,dtype=torch.float32,device=device)

def _visual_tensors_dev(visual,vi,device):
    return dict(
        dino=_tensor_dev(np.asarray(visual['DINO'][vi],np.float32)[None],device),
        conv=_tensor_dev(np.asarray(visual['CONV'][vi],np.float32)[None],device),
        geom=_tensor_dev(np.asarray(visual['GEOM'][vi],np.float32)[None],device),
        flow=_tensor_dev(np.asarray(visual['FLOW'][vi],np.float32)[None],device),
    )

def _make_v7_sample_dev(data,visual,stats,physics,idx,history_steps,horizons,device,norm=None):
    if norm is None:
        Pn,Bn,Un,Ln,En=normalize_arrays(data,stats)
    else:
        Pn,Bn,Un,Ln,En=norm

    h0=idx-history_steps+1
    if h0<0:
        raise RuntimeError('History starts before run.')
    offs=horizon_offsets(data,horizons)
    if idx+max(offs)>=len(data['time']):
        raise RuntimeError('Future target outside run.')

    vi=int(visual['visual_index'][idx])
    if vi<0:
        raise RuntimeError('No causal visual frame.')
    if float(visual['camera_times'][vi])>float(data['time'][idx])+1e-9:
        raise RuntimeError('Future visual leakage detected.')

    curP=data['P'][idx].astype(np.float32)
    curB=data['B'][idx].astype(np.float32)

    foot_pos=ANYMAL_D_FK.foot_positions(curP[:,0:3])
    curUraw=data['U'][idx].astype(np.float32)
    planned_body_delta=np.tile(
        (curUraw[:3]*FOOT_QUERY_PLAN_SEC)[None,:],(4,1)
    ).astype(np.float32)
    foot_query_kin=np.concatenate(
        [foot_pos,planned_body_delta],axis=-1
    ).astype(np.float32)

    curF=curP[:,9:12]
    cur_cls=force_regime_np(curF,REGIME_THRESHOLDS)
    cur_oh=np.eye(N_REGIMES,dtype=np.float32)[cur_cls]
    fscale=max(float(REGIME_THRESHOLDS[-1]),1.0)

    bp_list=[]; bb_list=[]; rp_list=[]; rb_list=[]
    tc=[]; tr=[]; controls=[]; phys=[]
    tpraw=[]; tbraw=[]

    for h,off in zip(horizons,offs):
        ti=idx+off
        bp=curP.copy()
        bp[:,0:3]=curP[:,0:3]+float(h)*curP[:,3:6]
        bb=curB.copy()

        tp=data['P'][ti].astype(np.float32)
        tb=data['B'][ti].astype(np.float32)
        rp=(tp-bp)/stats['P_std'][None,:]
        rb=(tb-bb)/stats['B_std']

        us=Un[idx:ti]
        ls=Ln[idx:ti]
        if len(us)==0:
            us=Un[idx:idx+1]
            ls=Ln[idx:idx+1]
        controls.append(
            np.concatenate([
                us[-1],ls[-1].reshape(-1),
                us.mean(0),ls.mean(0).reshape(-1)
            ]).astype(np.float32)
        )

        qcv=bp[:,0:3]
        dq0=curP[:,3:6]
        ddq0=np.zeros((4,3),np.float32)
        tau_phys=physics_tau_np(physics,qcv,dq0,ddq0,curF,cur_oh)
        acc_phys=physics_acc_np(physics,curF)

        qn=((qcv-stats['P_mean'][0:3])/stats['P_std'][0:3]).reshape(-1)
        taun=((tau_phys-stats['P_mean'][6:9])/stats['P_std'][6:9]).reshape(-1)
        accn=(acc_phys-stats['B_mean'][0:3])/stats['B_std'][0:3]
        fn=np.linalg.norm(curF,axis=-1)/fscale
        ph=np.concatenate(
            [qn,taun,accn,fn,cur_oh.reshape(-1)]
        ).astype(np.float32)
        if len(ph)!=PI_PHYS_DIM:
            raise RuntimeError('physics feature dim mismatch')
        phys.append(ph)

        bp_list.append(bp); bb_list.append(bb)
        rp_list.append(rp); rb_list.append(rb)
        tc.append(data['contact'][ti].astype(np.float32))
        tr.append(force_regime_np(tp[:,9:12],REGIME_THRESHOLDS))
        tpraw.append(tp); tbraw.append(tb)

    vis=_visual_tensors_dev(visual,vi,device)
    T=lambda x:_tensor_dev(x,device)

    sample=dict(
        p_hist=T(Pn[h0:idx+1].transpose(1,0,2)[None]),
        b_hist=T(Bn[h0:idx+1][None]),
        u_hist=T(Un[h0:idx+1][None]),
        l_hist=T(Ln[h0:idx+1].transpose(1,0,2)[None]),
        e_hist=T(En[h0:idx+1].transpose(1,0,2)[None]),
        future_control_features=T(np.asarray(controls,np.float32)[None]),
        horizon_values=T((np.asarray(horizons,np.float32)[:,None]/1.0)[None]),
        physics_features=T(np.asarray(phys,np.float32)[None]),
        foot_query_kin=T(foot_query_kin[None]),
        target_residual_p=T(np.asarray(rp_list,np.float32)[None]),
        target_residual_b=T(np.asarray(rb_list,np.float32)[None]),
        target_contact=T(np.asarray(tc,np.float32)[None]),
        target_regime=torch.as_tensor(
            np.asarray(tr,np.int64)[None],
            dtype=torch.long,device=device
        ),
        baseline_p_raw=np.asarray(bp_list,np.float32),
        baseline_b_raw=np.asarray(bb_list,np.float32),
        target_p_raw=np.asarray(tpraw,np.float32),
        target_b_raw=np.asarray(tbraw,np.float32),
        target_contact_raw=np.asarray(tc,np.float32),
        current_p_raw=curP,current_b_raw=curB,
        horizons_raw=np.asarray(horizons,np.float32),
        idx=int(idx),vi=vi,
    )
    sample.update(vis)
    return sample

@torch.inference_mode()
def _make_local_force_item_dev(base,rec,idx,H,horizons,stats,norm,physics,device):
    s=_make_v7_sample_dev(
        rec['data'],rec['visual'],stats,physics,idx,H,horizons,device,norm
    )
    b=v7f_extract_base_context(base,s,stats)
    item={
        'x':b['x'][0].half().cpu(),
        'baseF':b['baseF'][0].float().cpu(),
        'pcontact':b['pcontact'][0].float().cpu(),
        'regime':b['regime'][0].float().cpu(),
        'pstd_force':torch.as_tensor(
            stats['P_std'][9:12],dtype=torch.float32
        ),
        'gtF':torch.as_tensor(
            np.asarray(s['target_p_raw'][...,9:12],np.float32)
        ),
        'phase':torch.as_tensor(
            v7f_phase_targets(rec['data'],idx,horizons),dtype=torch.long
        ),
    }
    del s,b
    return item

def _collate_v7f_dev(records,idxs,device):
    return {
        'x':torch.stack([records[i]['x'].float() for i in idxs]).to(device,non_blocking=True),
        'baseF':torch.stack([records[i]['baseF'] for i in idxs]).to(device,non_blocking=True),
        'pcontact':torch.stack([records[i]['pcontact'] for i in idxs]).to(device,non_blocking=True),
        'regime':torch.stack([records[i]['regime'] for i in idxs]).to(device,non_blocking=True),
        'pstd_force':torch.stack(
            [records[i]['pstd_force'] for i in idxs]
        )[:,None,None,:].to(device,non_blocking=True),
        'gtF':torch.stack([records[i]['gtF'] for i in idxs]).to(device,non_blocking=True),
        'phase':torch.stack([records[i]['phase'] for i in idxs]).to(device,non_blocking=True),
    }

def _gap_windows_exact(data,visual,ttest,anchor_gap_s,context_sec,hop_sec,max_horizon,half_life):
    dt=float(np.median(np.diff(data['time'])))
    H=int(round(float(context_sec)/dt))
    max_off=int(round(float(max_horizon)/dt))
    anchor_cut=float(ttest)-float(anchor_gap_s)

    if anchor_cut<=float(data['time'][0])+float(context_sec):
        raise RuntimeError(
            f'Not enough history for ttest={ttest:.3f}, gap={anchor_gap_s:.3f}'
        )

    desired=[]
    a=anchor_cut
    lower=max(
        float(data['time'][0])+float(context_sec),
        anchor_cut-float(TRAIN_LOOKBACK_SEC)
    )
    while a>=lower-1e-9:
        desired.append(a)
        a-=float(hop_sec)

    rows=[]; seen=set()
    for da in reversed(desired):
        idx=nearest_index(data['time'],da)
        if idx in seen:
            continue
        seen.add(idx)
        if idx-H+1<0 or idx+max_off>=len(data['time']):
            continue
        if int(visual['visual_index'][idx])<0:
            continue

        actual_anchor=float(data['time'][idx])
        target_end=float(data['time'][idx+max_off])
        if actual_anchor>anchor_cut+0.015:
            continue
        if target_end>=float(ttest)-1e-9:
            continue

        age=max(0.0,anchor_cut-actual_anchor)
        w=0.5**(age/float(half_life))
        rows.append((idx,w,actual_anchor,target_end))

    if not rows:
        raise RuntimeError(
            f'No gap-sweep windows for ttest={ttest}, gap={anchor_gap_s}'
        )

    rows.sort(key=lambda x:x[2])
    wmax=max(r[1] for r in rows)
    rows=[(i,w/wmax,a,te) for i,w,a,te in rows]
    return rows,H,anchor_cut

def _adapt_v7f_at_fixed_gap_dev(
    global_head,base,rec,ttest,anchor_gap_s,
    context_sec,hop_sec,horizons,half_life,
    seed,label,device
):
    # Seed only the assigned GPU generator; avoid cross-thread global seeding.
    with torch.cuda.device(device):
        torch.cuda.manual_seed(int(seed))

    head=copy.deepcopy(global_head).to(device)

    end0=float(min(rec['ttests']))
    stats=make_stats_between(
        rec['data'],float(rec['data']['time'][0]),end0
    )
    norm=normalize_arrays(rec['data'],stats)
    physics=rec['physics_prior_v7']

    wins,H,anchor_cut=_gap_windows_exact(
        rec['data'],rec['visual'],ttest,anchor_gap_s,
        context_sec,hop_sec,max(horizons),half_life
    )

    items=[]
    for idx,w,anchor_t,target_end in wins:
        it=_make_local_force_item_dev(
            base,rec,idx,H,horizons,stats,norm,physics,device
        )
        it['w']=float(w)
        it['anchor_t']=float(anchor_t)
        it['target_end']=float(target_end)
        items.append(it)

    opt=torch.optim.AdamW(
        head.parameters(),
        lr=V7F_LR_LOCAL,
        weight_decay=V7F_WEIGHT_DECAY
    )

    for ep in range(V7F_LOCAL_EPOCHS):
        head.train()
        losses=[]
        for it in items:
            b=_collate_v7f_dev([it],[0],device)
            opt.zero_grad(set_to_none=True)
            o=head(
                b['x'],b['baseF'],b['pstd_force'],
                b['pcontact'],b['regime']
            )
            raw,_=v7f_force_loss(o,b)
            loss=raw*float(it['w'])
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite gap-sweep loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                head.parameters(),V7F_GRAD_CLIP
            )
            opt.step()
            losses.append(float(raw.detach().cpu()))
            del b,o,raw,loss

        print(
            f'GPU{device.index} {rec["terrain_type"]} '
            f't={ttest:g} {label} gap={anchor_gap_s:g}s '
            f'epoch {ep+1}/{V7F_LOCAL_EPOCHS} '
            f'n={len(items)} loss={np.mean(losses):.4f}'
        )

    audit=dict(
        n_train_windows=int(len(items)),
        requested_anchor_gap_s=float(anchor_gap_s),
        cutoff_anchor_s=float(anchor_cut),
        latest_train_anchor_s=float(items[-1]['anchor_t']),
        latest_train_target_end_s=float(items[-1]['target_end']),
        actual_anchor_gap_s=float(ttest-items[-1]['anchor_t']),
        actual_target_gap_s=float(ttest-items[-1]['target_end']),
    )
    return head,stats,physics,H,audit

@torch.inference_mode()
def _predict_v7f_force_dev(
    base,head,rec,ttest,context_sec,horizons,
    stats,physics,H,device
):
    idx=nearest_index(rec['data']['time'],ttest)
    norm=normalize_arrays(rec['data'],stats)
    s=_make_v7_sample_dev(
        rec['data'],rec['visual'],stats,physics,
        idx,H,horizons,device,norm
    )
    b=v7f_extract_base_context(base,s,stats)
    pstd=torch.as_tensor(
        stats['P_std'][9:12],
        dtype=torch.float32,device=device
    ).view(1,1,1,3)
    o=head(
        b['x'],b['baseF'],pstd,b['pcontact'],b['regime']
    )
    force=o['force'][0].detach().cpu().numpy()
    phase_prob=o['phase_prob'][0].detach().cpu().numpy()
    alpha=o['fusion_alpha'][0].detach().cpu().numpy()
    return force,phase_prob,alpha,s,idx

def _force_rows(rec,test_id,ttest,gap_s,branch,horizons,predF,sample,audit):
    rows=[]
    for hi,h in enumerate(horizons):
        for li,L in enumerate(LEG_NAMES):
            pred=np.asarray(predF[hi,li],np.float64)
            gt=np.asarray(
                sample['target_p_raw'][hi,li,9:12],np.float64
            )
            err=pred-gt
            ve=float(np.linalg.norm(err))
            pm=float(np.linalg.norm(pred))
            gm=float(np.linalg.norm(gt))
            cos=(
                float(np.dot(pred,gt)/(pm*gm+1e-12))
                if gm>=CONTACT_THRESHOLD and pm>1e-9 else np.nan
            )
            rows.append(dict(
                terrain_type=rec['terrain_type'],
                environment=rec['environment'],
                run=rec['run'],
                test_id=int(test_id),
                ttest_s=float(ttest),
                requested_anchor_gap_s=float(gap_s),
                branch=branch,
                horizon_s=float(h),
                leg=L,
                pred_Fx=float(pred[0]),
                pred_Fy=float(pred[1]),
                pred_Fz=float(pred[2]),
                gt_Fx=float(gt[0]),
                gt_Fy=float(gt[1]),
                gt_Fz=float(gt[2]),
                component_MAE_N=float(np.abs(err).mean()),
                force_vector_error_N=ve,
                pred_force_mag_N=pm,
                true_force_mag_N=gm,
                force_magnitude_abs_error_N=float(abs(pm-gm)),
                force_direction_cosine=cos,
                vector_error_percent=(
                    100.0*ve/gm if gm>1e-9 else np.nan
                ),
                latest_train_anchor_s=audit['latest_train_anchor_s'],
                latest_train_target_end_s=audit['latest_train_target_end_s'],
                actual_anchor_gap_s=audit['actual_anchor_gap_s'],
                actual_target_gap_s=audit['actual_target_gap_s'],
                n_train_windows=audit['n_train_windows'],
            ))
    return rows

# ----------------------------------------------------------------------
# One persistent frozen-model replica set per GPU.
# GPU0 uses the original trained objects. GPU1 receives exact state copies.
# ----------------------------------------------------------------------
prior_short.to(GPU_DEVICES[0]).eval()
prior_long.to(GPU_DEVICES[0]).eval()
v7f_head_short.to(GPU_DEVICES[0]).eval()
v7f_head_long.to(GPU_DEVICES[0]).eval()

REPLICAS={
    0:dict(
        prior_short=prior_short,
        prior_long=prior_long,
        head_short=v7f_head_short,
        head_long=v7f_head_long,
    ),
    1:dict(
        prior_short=copy.deepcopy(prior_short).to(GPU_DEVICES[1]).eval(),
        prior_long=copy.deepcopy(prior_long).to(GPU_DEVICES[1]).eval(),
        head_short=copy.deepcopy(v7f_head_short).to(GPU_DEVICES[1]).eval(),
        head_long=copy.deepcopy(v7f_head_long).to(GPU_DEVICES[1]).eval(),
    )
}

jobs=[]
for ri,rec in enumerate(run_records):
    for ti,tt in enumerate(rec['ttests'],1):
        for gi,gap_s in enumerate(GAP_SWEEP_SEC):
            jobs.append((ri,ti,float(tt),gi,float(gap_s)))

# Deterministic split: each GPU owns half the independent jobs.
job_chunks=[jobs[0::2],jobs[1::2]]

def _gap_worker(gpu_idx,worker_jobs):
    device=GPU_DEVICES[gpu_idx]
    rep=REPLICAS[gpu_idx]
    rows=[]; audits=[]

    for ri,ti,tt,gi,gap_s in worker_jobs:
        rec=run_records[ri]
        seed=int(V7F_SEED+50000+ri*10000+ti*1000+gi*10)

        print('\n'+'='*80)
        print(
            f'GPU{gpu_idx} SAME-RUN GAP | {rec["terrain_type"]} '
            f'{rec["environment"]}/{rec["run"]} '
            f'| test#{ti} t={tt:g}s | gap={gap_s:g}s'
        )
        print('='*80)

        sh,ss,sp,Hs,sa=_adapt_v7f_at_fixed_gap_dev(
            rep['head_short'],rep['prior_short'],rec,tt,gap_s,
            SHORT_CONTEXT_SEC,V7F_LOCAL_HOP_SHORT,SHORT_HORIZONS,
            SHORT_RECENCY_HALF_LIFE_SEC,seed+1,'SHORT',device
        )
        lh,ls,lp,Hl,la=_adapt_v7f_at_fixed_gap_dev(
            rep['head_long'],rep['prior_long'],rec,tt,gap_s,
            LONG_CONTEXT_SEC,V7F_LOCAL_HOP_LONG,LONG_HORIZONS,
            LONG_RECENCY_HALF_LIFE_SEC,seed+2,'LONG',device
        )

        sf,_,_,ssamp,_=_predict_v7f_force_dev(
            rep['prior_short'],sh,rec,tt,
            SHORT_CONTEXT_SEC,SHORT_HORIZONS,
            ss,sp,Hs,device
        )
        lf,_,_,lsamp,_=_predict_v7f_force_dev(
            rep['prior_long'],lh,rec,tt,
            LONG_CONTEXT_SEC,LONG_HORIZONS,
            ls,lp,Hl,device
        )

        rows += _force_rows(
            rec,ti,tt,gap_s,'SHORT',
            SHORT_HORIZONS,sf,ssamp,sa
        )
        rows += _force_rows(
            rec,ti,tt,gap_s,'LONG',
            LONG_HORIZONS,lf,lsamp,la
        )

        for branch,a in [('SHORT',sa),('LONG',la)]:
            audits.append(dict(
                terrain_type=rec['terrain_type'],
                environment=rec['environment'],
                run=rec['run'],
                test_id=int(ti),
                ttest_s=float(tt),
                gpu=int(gpu_idx),
                branch=branch,
                **a
            ))

        del sh,lh,ssamp,lsamp
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

    return rows,audits

with ThreadPoolExecutor(max_workers=2) as ex:
    fut0=ex.submit(_gap_worker,0,job_chunks[0])
    fut1=ex.submit(_gap_worker,1,job_chunks[1])
    rows0,audit0=fut0.result()
    rows1,audit1=fut1.result()

raw=pd.DataFrame(rows0+rows1)
audit=pd.DataFrame(audit0+audit1)

summary=(
    raw.groupby(
        ['terrain_type','environment','run',
         'requested_anchor_gap_s','horizon_s','leg'],
        as_index=False
    )
    .agg(
        n_tests=('test_id','count'),
        mean_component_MAE_N=('component_MAE_N','mean'),
        mean_vector_error_N=('force_vector_error_N','mean'),
        mean_true_force_mag_N=('true_force_mag_N','mean'),
        mean_force_magnitude_abs_error_N=(
            'force_magnitude_abs_error_N','mean'
        ),
        mean_direction_cosine=('force_direction_cosine','mean'),
        mean_actual_anchor_gap_s=('actual_anchor_gap_s','mean'),
        mean_actual_target_gap_s=('actual_target_gap_s','mean'),
    )
)
summary['vector_error_percent_of_mean_force']=np.where(
    summary['mean_true_force_mag_N']>1e-9,
    100.0*summary['mean_vector_error_N']/
    summary['mean_true_force_mag_N'],
    np.nan
)

overall=(
    raw.groupby(
        ['requested_anchor_gap_s','horizon_s'],as_index=False
    )
    .agg(
        n_vectors=('force_vector_error_N','count'),
        mean_component_MAE_N=('component_MAE_N','mean'),
        mean_vector_error_N=('force_vector_error_N','mean'),
        mean_true_force_mag_N=('true_force_mag_N','mean'),
        mean_force_magnitude_abs_error_N=(
            'force_magnitude_abs_error_N','mean'
        ),
        mean_direction_cosine=('force_direction_cosine','mean'),
    )
)
overall['vector_error_percent_of_mean_force']=np.where(
    overall['mean_true_force_mag_N']>1e-9,
    100.0*overall['mean_vector_error_N']/
    overall['mean_true_force_mag_N'],
    np.nan
)

raw.to_csv(GAP_ROOT/'SAME_RUN_GAP_SWEEP_RAW.csv',index=False)
summary.to_csv(
    GAP_ROOT/'SAME_RUN_GAP_SWEEP_BY_TERRAIN_LEG.csv',index=False
)
overall.to_csv(
    GAP_ROOT/'SAME_RUN_GAP_SWEEP_OVERALL.csv',index=False
)
audit.to_csv(GAP_ROOT/'SAME_RUN_GAP_AUDIT.csv',index=False)

cfg=dict(
    experiment='2-GPU same-run temporal-gap sweep',
    gpus=[torch.cuda.get_device_name(0),torch.cuda.get_device_name(1)],
    original_runs=[
        dict(
            terrain_type=r['terrain_type'],
            environment=r['environment'],
            run=r['run']
        ) for r in run_records
    ],
    original_ttests={
        r['terrain_type']:[float(x) for x in r['ttests']]
        for r in run_records
    },
    requested_anchor_gaps_s=GAP_SWEEP_SEC,
    definition=(
        'gap = ttest - latest gradient-training INPUT anchor. '
        'No training or validation/model-selection data after that cutoff.'
    ),
    local_epochs=int(V7F_LOCAL_EPOCHS),
    lr_local=float(V7F_LR_LOCAL),
)
(GAP_ROOT/'CONFIG.json').write_text(json.dumps(cfg,indent=2))

manifest=[]
for p in sorted(GAP_ROOT.rglob('*')):
    if p.is_file():
        manifest.append(dict(
            file=str(p.relative_to(GAP_ROOT)),
            sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            size_bytes=p.stat().st_size
        ))
pd.DataFrame(manifest).to_csv(
    GAP_ROOT/'SHA256_MANIFEST.csv',index=False
)

zip_path=WORK_ROOT/'V7F_SAME_RUN_GAP_SWEEP_2GPU_RESULTS.zip'
if zip_path.exists():
    zip_path.unlink()
with zipfile.ZipFile(
    zip_path,'w',zipfile.ZIP_DEFLATED,allowZip64=True
) as z:
    for p in sorted(GAP_ROOT.rglob('*')):
        if p.is_file():
            z.write(
                p,
                arcname=str(
                    Path(GAP_ROOT.name)/p.relative_to(GAP_ROOT)
                )
            )

print('\n2-GPU SAME-RUN GAP SWEEP COMPLETE')
display(overall.sort_values(
    ['requested_anchor_gap_s','horizon_s']
))
print('ZIP:',zip_path)
try:
    from google.colab import files
    files.download(str(zip_path))
except Exception as e:
    print('Auto-download skipped:',e)

# %% notebook cell 23
# ======================================================================
# APPEND CELL 2 — 2x RTX 4080 STRICT UNSEEN-TRAJECTORY ZERO-SHOT TEST
#
# Held-out trajectories:
#   forest            ForestEnv/P2001
#   swamp_marsh       GreatMarsh/P2027
#   rough_uneven      SeasonalForestWinter/P2006
#   wet_uneven_proxy  OldTownFall/P2001
#   stairs_urban      dynamically discovered true-stair holdout
#
# STRICTNESS:
#   - none of these (environment,run) pairs may appear in run_records;
#   - V7 receives ZERO gradient updates from the held-out run;
#   - V7-F receives ZERO gradient updates from the held-out run;
#   - NO local adaptation on the held-out run;
#   - NO normalization fit on the held-out run;
#   - NO physics fit on the held-out run.
#
# The held-out run supplies only causal observations at inference time and
# future ground truth after prediction for scoring.
# ======================================================================
import os, gc, json, hashlib, zipfile, shutil, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import torch

if torch.cuda.device_count()<2:
    raise RuntimeError(
        f'2 visible CUDA GPUs are required; found {torch.cuda.device_count()}.'
    )

_required=[
    'run_records','TRAIN_RECORDS','prior_short','prior_long','PERSIST_ROOT','WORK_ROOT','v7_persistent_bundle_complete','prepare_v7_record_data',
    'v7f_head_short','v7f_head_long',
    'download_run_v7','extract_run_physics','build_frame_manifest',
    'build_dinov3_cache','build_convnext_cache','build_utonia_cache',
    'build_waft_cache','load_visual_bundle',
    'make_stats_between','horizon_offsets',
    'SHORT_CONTEXT_SEC','LONG_CONTEXT_SEC',
    'SHORT_HORIZONS','LONG_HORIZONS',
    'CONTACT_THRESHOLD','LEG_NAMES',
    'GPU_DEVICES','REPLICAS','_predict_v7f_force_dev','STRICT_HOLDOUTS','STAIR_TERRAIN_TYPE'
]
_missing=[x for x in _required if x not in globals()]
if _missing:
    raise RuntimeError(
        'Run the completed V7/V7-F notebook and APPEND CELL 1 first. '
        'Missing: '+repr(_missing)
    )

UNSEEN_SPECS=[
    dict(
        terrain_type='forest',
        environment='ForestEnv',
        run='P2001'
    ),
    dict(
        terrain_type='swamp_marsh',
        environment='GreatMarsh',
        run='P2027'
    ),
    dict(
        terrain_type='rough_uneven',
        environment='SeasonalForestWinter',
        run='P2006'
    ),
    dict(
        terrain_type='wet_uneven_proxy',
        environment='OldTownFall',
        run='P2001'
    ),
    dict(
        terrain_type=STAIR_TERRAIN_TYPE,
        environment=STRICT_HOLDOUTS[STAIR_TERRAIN_TYPE][0],
        run=STRICT_HOLDOUTS[STAIR_TERRAIN_TYPE][1]
    ),
]

# ----------------------------------------------------------------------
# Hard proof-of-split check against the exact data source used by BOTH
# V7 global training and V7-F global-cache construction: run_records.
# ----------------------------------------------------------------------
TRAIN_KEYS={(r['environment'],r['run']) for r in TRAIN_RECORDS}
HOLDOUT_KEYS={(r['environment'],r['run']) for r in UNSEEN_SPECS}

overlap=TRAIN_KEYS & HOLDOUT_KEYS
if overlap:
    raise RuntimeError(
        'STRICT HOLDOUT FAILURE: trajectory seen by V7/V7-F training: '
        +repr(sorted(overlap))
    )

if len(HOLDOUT_KEYS)!=EXPECTED_STRICT_HOLDOUTS:
    raise RuntimeError(
        f'Expected {EXPECTED_STRICT_HOLDOUTS} unique held-out trajectories, '
        f'got {len(HOLDOUT_KEYS)}.'
    )

print('ALL TRAIN TRAJECTORIES USED BY V7 AND V7-F:')
for x in sorted(TRAIN_KEYS):
    print('  ',x)
print('\nSTRICT HELD-OUT TRAJECTORIES:')
for x in sorted(HOLDOUT_KEYS):
    print('  ',x)
print('\nOverlap:',overlap)
assert not overlap

ZERO_ROOT=PERSIST_ROOT/'V7F_unseen_trajectory_zero_shot_2GPU'
ZERO_ROOT.mkdir(parents=True,exist_ok=True)
print('Resume-safe result root:',ZERO_ROOT)

stage_times=[]

# ----------------------------------------------------------------------
# A) Prepare held-out raw data and visual caches.
#
# We alternate the held-out runs between GPU0 and GPU1. The current
# cache-builder implementation is intentionally executed one run at a time
# because the external Utonia worker and cache writes are not thread-safe
# under a single notebook process. The later prediction stage runs on both
# GPUs concurrently.
# ----------------------------------------------------------------------
unseen_records=[]
_old_DEVICE=globals().get('DEVICE',torch.device('cuda:0'))
_old_utonia_gpu=globals().get('UTONIA_GPU_INDEX',None)

try:
    for si,spec in enumerate(UNSEEN_SPECS):
        gpu_idx=si%2
        DEVICE=GPU_DEVICES[gpu_idx]
        UTONIA_GPU_INDEX=gpu_idx

        env,run=spec['environment'],spec['run']
        print('\n'+'#'*100)
        print(
            f'UNSEEN PREP GPU{gpu_idx} | '
            f'{spec["terrain_type"]} {env}/{run}'
        )
        print('#'*100)

        t0=time.perf_counter()
        # Held-out raw images/rosbag are not needed after persistent feature caches
        # are complete, so this helper can resume from cache without redownloading.
        base,data,manifest,visual=prepare_v7_record_data(
            env,run,require_raw=False
        )

        elapsed=time.perf_counter()-t0
        stage_times.append(dict(
            stage='heldout_preprocess',
            terrain_type=spec['terrain_type'],
            environment=env,run=run,gpu=gpu_idx,
            elapsed_sec=float(elapsed)
        ))

        unseen_records.append(dict(
            **spec,
            prep_gpu=int(gpu_idx),
            base=base,data=data,
            manifest=manifest,visual=visual
        ))

        # All held-out inference uses cached tensors/arrays, never raw image files.
        # Remove raw held-out data only after all persistent branches validated.
        if base.exists():
            shutil.rmtree(base)
            print('Removed held-out raw trajectory after cache validation:',base)

        gc.collect()
        with torch.cuda.device(DEVICE):
            torch.cuda.empty_cache()
finally:
    DEVICE=_old_DEVICE
    if _old_utonia_gpu is None:
        if 'UTONIA_GPU_INDEX' in globals():
            del UTONIA_GPU_INDEX
    else:
        UTONIA_GPU_INDEX=_old_utonia_gpu

# Legacy terrain calibration comes from the original diagnostic source run.
# The staircase bucket has no legacy diagnostic run, so use the first staircase
# TRAINING record as its already-fitted source statistics/physics. The held-out
# staircase trajectory itself is never used to fit either quantity.
source_by_terrain={r['terrain_type']:r for r in run_records}
for _r in TRAIN_RECORDS:
    source_by_terrain.setdefault(_r['terrain_type'],_r)

_missing_sources={
    r['terrain_type'] for r in UNSEEN_SPECS
    if r['terrain_type'] not in source_by_terrain
}
if _missing_sources:
    raise RuntimeError(
        'No training source record for held-out terrain(s): '
        +repr(sorted(_missing_sources))
    )

# ----------------------------------------------------------------------
# B) Choose four test anchors from each unseen trajectory. Their successive
# separations are approximately 7 s, 9 s and 10 s.
#
# Selection uses ONLY time coverage and existence of a causal visual frame.
# Force/contact values are not used to pick the test points.
# ----------------------------------------------------------------------
def _snap_valid_unseen_ttests(rec):
    data,vis=rec['data'],rec['visual']
    t=np.asarray(data['time'],np.float64)
    dt=float(np.median(np.diff(t)))

    Hlong=int(
        getattr(
            REPLICAS[0]['prior_long'],
            'history_steps',
            round(LONG_CONTEXT_SEC/dt)
        )
    )
    offmax=max(horizon_offsets(data,LONG_HORIZONS))

    valid=np.zeros(len(t),dtype=bool)
    lo=Hlong-1
    hi=len(t)-offmax
    valid[lo:hi]=True
    valid &= (np.asarray(vis['visual_index'])>=0)

    valid_idx=np.flatnonzero(valid)
    if len(valid_idx)==0:
        raise RuntimeError(
            'No valid unseen test anchors for '
            +rec['environment']+'/'+rec['run']
        )

    valid_times=t[valid_idx]
    offsets=np.asarray([0.0,7.0,16.0,26.0],np.float64)

    earliest=float(valid_times[0])
    latest=float(valid_times[-1])

    base=max(earliest+1.0,20.0)
    if base+offsets[-1]>latest:
        base=latest-offsets[-1]

    if base<earliest:
        raise RuntimeError(
            f'Run too short for four tests spaced 7/9/10 s: '
            f'{rec["environment"]}/{rec["run"]}; '
            f'valid interval={earliest:.2f}..{latest:.2f}s'
        )

    picked=[]
    used=set()
    for desired in base+offsets:
        j=int(np.argmin(np.abs(valid_times-desired)))
        idx=int(valid_idx[j])
        if idx in used:
            raise RuntimeError(
                'Duplicate snapped ttest: '+rec['environment']+'/'+rec['run']
            )
        used.add(idx)
        picked.append(float(t[idx]))

    diffs=np.diff(picked)
    print(
        rec['terrain_type'],
        'ttests=',[round(x,3) for x in picked],
        'actual spacings=',[round(float(x),3) for x in diffs]
    )
    return picked

for rec in unseen_records:
    rec['zero_shot_ttests']=_snap_valid_unseen_ttests(rec)

# ----------------------------------------------------------------------
# C) STRICT zero-shot prediction on two GPUs.
#
# Every inference job uses:
#   trained V7 global prior + trained V7-F global head
# from the ORIGINAL training trajectories only.
#
# The source training run supplies its already-fitted normalization and
# physics prior. The held-out trajectory never fits either quantity.
# ----------------------------------------------------------------------
def _zero_shot_rows_for_job(gpu_idx,rec,src,test_id,tt):
    device=GPU_DEVICES[gpu_idx]
    rep=REPLICAS[gpu_idx]

    src_end=float(min(src['ttests']))
    src_stats=make_stats_between(
        src['data'],float(src['data']['time'][0]),src_end
    )
    src_physics=src['physics_prior_v7']

    dt=float(np.median(np.diff(rec['data']['time'])))
    Hs=int(
        getattr(
            rep['prior_short'],
            'history_steps',
            round(SHORT_CONTEXT_SEC/dt)
        )
    )
    Hl=int(
        getattr(
            rep['prior_long'],
            'history_steps',
            round(LONG_CONTEXT_SEC/dt)
        )
    )

    sf,_,_,ssamp,sidx=_predict_v7f_force_dev(
        rep['prior_short'],rep['head_short'],rec,float(tt),
        SHORT_CONTEXT_SEC,SHORT_HORIZONS,
        src_stats,src_physics,Hs,device
    )
    lf,_,_,lsamp,lidx=_predict_v7f_force_dev(
        rep['prior_long'],rep['head_long'],rec,float(tt),
        LONG_CONTEXT_SEC,LONG_HORIZONS,
        src_stats,src_physics,Hl,device
    )

    rows=[]
    for branch,horizons,predF,samp in [
        ('SHORT',SHORT_HORIZONS,sf,ssamp),
        ('LONG',LONG_HORIZONS,lf,lsamp)
    ]:
        for hi,h in enumerate(horizons):
            for li,L in enumerate(LEG_NAMES):
                pred=np.asarray(predF[hi,li],np.float64)
                gt=np.asarray(
                    samp['target_p_raw'][hi,li,9:12],np.float64
                )
                err=pred-gt
                ve=float(np.linalg.norm(err))
                pm=float(np.linalg.norm(pred))
                gm=float(np.linalg.norm(gt))
                cos=(
                    float(np.dot(pred,gt)/(pm*gm+1e-12))
                    if gm>=CONTACT_THRESHOLD and pm>1e-9
                    else np.nan
                )

                rows.append(dict(
                    terrain_type=rec['terrain_type'],
                    training_environment=src['environment'],
                    training_run=src['run'],
                    unseen_environment=rec['environment'],
                    unseen_run=rec['run'],
                    test_id=int(test_id),
                    ttest_s=float(tt),
                    inference_gpu=int(gpu_idx),
                    branch=branch,
                    horizon_s=float(h),
                    leg=L,
                    pred_Fx=float(pred[0]),
                    pred_Fy=float(pred[1]),
                    pred_Fz=float(pred[2]),
                    gt_Fx=float(gt[0]),
                    gt_Fy=float(gt[1]),
                    gt_Fz=float(gt[2]),
                    component_MAE_N=float(np.abs(err).mean()),
                    force_vector_error_N=ve,
                    pred_force_mag_N=pm,
                    true_force_mag_N=gm,
                    force_magnitude_abs_error_N=float(abs(pm-gm)),
                    force_direction_cosine=cos,
                    vector_error_percent=(
                        100.0*ve/gm if gm>1e-9 else np.nan
                    ),
                ))

    manifest=dict(
        terrain_type=rec['terrain_type'],
        training_environment=src['environment'],
        training_run=src['run'],
        unseen_environment=rec['environment'],
        unseen_run=rec['run'],
        test_id=int(test_id),
        ttest_s=float(tt),
        inference_gpu=int(gpu_idx),
        short_robot_idx=int(sidx),
        long_robot_idx=int(lidx),
        seen_by_V7_training=False,
        seen_by_V7F_training=False,
        local_adaptation=False,
        gradient_updates_on_heldout=False,
        heldout_physics_fit=False,
        heldout_normalization_fit=False,
    )

    del ssamp,lsamp
    with torch.cuda.device(device):
        torch.cuda.empty_cache()

    return rows,manifest

jobs=[]
for ri,rec in enumerate(unseen_records):
    src=source_by_terrain[rec['terrain_type']]
    if (rec['environment'],rec['run']) in TRAIN_KEYS:
        raise RuntimeError('Leakage detected before inference.')
    for ti,tt in enumerate(rec['zero_shot_ttests'],1):
        jobs.append((ri,ti,float(tt)))

chunks=[jobs[0::2],jobs[1::2]]

def _zero_worker(gpu_idx,worker_jobs):
    out_rows=[]; out_manifest=[]
    for ri,ti,tt in worker_jobs:
        rec=unseen_records[ri]
        src=source_by_terrain[rec['terrain_type']]
        print(
            f'GPU{gpu_idx} ZERO-SHOT | '
            f'{rec["terrain_type"]} '
            f'{rec["environment"]}/{rec["run"]} '
            f'test#{ti} t={tt:.3f}s'
        )
        rr,mm=_zero_shot_rows_for_job(
            gpu_idx,rec,src,ti,tt
        )
        out_rows+=rr
        out_manifest.append(mm)
    return out_rows,out_manifest

infer_t0=time.perf_counter()
with ThreadPoolExecutor(max_workers=2) as ex:
    f0=ex.submit(_zero_worker,0,chunks[0])
    f1=ex.submit(_zero_worker,1,chunks[1])
    rows0,man0=f0.result()
    rows1,man1=f1.result()
infer_elapsed=time.perf_counter()-infer_t0

stage_times.append(dict(
    stage='all_zero_shot_inference',
    terrain_type='ALL',
    environment='ALL',run='ALL',gpu='0+1',
    elapsed_sec=float(infer_elapsed)
))

raw=pd.DataFrame(rows0+rows1)
manifest_df=pd.DataFrame(man0+man1)

# Final leakage audit: there must be no held-out key in the exact training key set.
for _,r in manifest_df.iterrows():
    key=(r['unseen_environment'],r['unseen_run'])
    if key in TRAIN_KEYS:
        raise RuntimeError(
            'FINAL LEAKAGE AUDIT FAILED: '+repr(key)
        )
if not (
    (manifest_df['seen_by_V7_training']==False).all()
    and (manifest_df['seen_by_V7F_training']==False).all()
    and (manifest_df['gradient_updates_on_heldout']==False).all()
):
    raise RuntimeError('Strict zero-shot flags failed.')

by_terrain_leg=(
    raw.groupby(
        [
            'terrain_type',
            'training_environment','training_run',
            'unseen_environment','unseen_run',
            'horizon_s','leg'
        ],
        as_index=False
    )
    .agg(
        n_tests=('test_id','count'),
        mean_component_MAE_N=('component_MAE_N','mean'),
        mean_vector_error_N=('force_vector_error_N','mean'),
        mean_true_force_mag_N=('true_force_mag_N','mean'),
        mean_force_magnitude_abs_error_N=(
            'force_magnitude_abs_error_N','mean'
        ),
        mean_direction_cosine=('force_direction_cosine','mean'),
    )
)
by_terrain_leg['vector_error_percent_of_mean_force']=np.where(
    by_terrain_leg['mean_true_force_mag_N']>1e-9,
    100.0*by_terrain_leg['mean_vector_error_N']/
    by_terrain_leg['mean_true_force_mag_N'],
    np.nan
)

by_test=(
    raw.groupby(
        [
            'terrain_type','unseen_environment','unseen_run',
            'test_id','ttest_s','horizon_s'
        ],
        as_index=False
    )
    .agg(
        mean_component_MAE_N=('component_MAE_N','mean'),
        mean_vector_error_N=('force_vector_error_N','mean'),
        mean_true_force_mag_N=('true_force_mag_N','mean'),
        mean_force_magnitude_abs_error_N=(
            'force_magnitude_abs_error_N','mean'
        ),
        mean_direction_cosine=('force_direction_cosine','mean'),
    )
)
by_test['vector_error_percent_of_mean_force']=np.where(
    by_test['mean_true_force_mag_N']>1e-9,
    100.0*by_test['mean_vector_error_N']/
    by_test['mean_true_force_mag_N'],
    np.nan
)

overall=(
    raw.groupby(['horizon_s'],as_index=False)
    .agg(
        n_vectors=('force_vector_error_N','count'),
        mean_component_MAE_N=('component_MAE_N','mean'),
        mean_vector_error_N=('force_vector_error_N','mean'),
        mean_true_force_mag_N=('true_force_mag_N','mean'),
        mean_force_magnitude_abs_error_N=(
            'force_magnitude_abs_error_N','mean'
        ),
        mean_direction_cosine=('force_direction_cosine','mean'),
    )
)
overall['vector_error_percent_of_mean_force']=np.where(
    overall['mean_true_force_mag_N']>1e-9,
    100.0*overall['mean_vector_error_N']/
    overall['mean_true_force_mag_N'],
    np.nan
)

raw.to_csv(
    ZERO_ROOT/'UNSEEN_TRAJECTORY_RAW.csv',index=False
)
by_terrain_leg.to_csv(
    ZERO_ROOT/'UNSEEN_TRAJECTORY_BY_TERRAIN_LEG.csv',
    index=False
)
by_test.to_csv(
    ZERO_ROOT/'UNSEEN_TRAJECTORY_BY_TEST.csv',
    index=False
)
overall.to_csv(
    ZERO_ROOT/'UNSEEN_TRAJECTORY_OVERALL.csv',
    index=False
)
manifest_df.to_csv(
    ZERO_ROOT/'UNSEEN_TEST_MANIFEST.csv',
    index=False
)
pd.DataFrame(stage_times).to_csv(
    ZERO_ROOT/'TIMING.csv',index=False
)

cfg=dict(
    experiment='2-GPU strict unseen-trajectory zero-shot V7-F',
    gpus=[
        torch.cuda.get_device_name(0),
        torch.cuda.get_device_name(1)
    ],
    exact_training_keys=sorted(
        [list(x) for x in TRAIN_KEYS]
    ),
    held_out_runs=UNSEEN_SPECS,
    training_holdout_overlap=[],
    requested_test_spacing_s=[7.0,9.0,10.0],
    local_adaptation=False,
    V7_gradient_updates_on_heldout=False,
    V7F_gradient_updates_on_heldout=False,
    heldout_physics_fit=False,
    heldout_normalization_fit=False,
    calibration_policy=(
        'For each terrain, use normalization statistics and the '
        'already-fitted physics prior from a corresponding training '
        'trajectory (legacy diagnostic source for the original four; '
        'training-only source for stairs); never fit them on the held-out run.'
    )
)
(ZERO_ROOT/'CONFIG.json').write_text(
    json.dumps(cfg,indent=2)
)

sha=[]
for p in sorted(ZERO_ROOT.rglob('*')):
    if p.is_file():
        sha.append(dict(
            file=str(p.relative_to(ZERO_ROOT)),
            sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            size_bytes=p.stat().st_size
        ))
pd.DataFrame(sha).to_csv(
    ZERO_ROOT/'SHA256_MANIFEST.csv',index=False
)

zip_path=WORK_ROOT/'V7F_UNSEEN_TRAJECTORY_ZERO_SHOT_2GPU_RESULTS.zip'
if zip_path.exists():
    zip_path.unlink()

with zipfile.ZipFile(
    zip_path,'w',zipfile.ZIP_DEFLATED,allowZip64=True
) as z:
    for p in sorted(ZERO_ROOT.rglob('*')):
        if p.is_file():
            z.write(
                p,
                arcname=str(
                    Path(ZERO_ROOT.name)/
                    p.relative_to(ZERO_ROOT)
                )
            )

print('\nSTRICT 2-GPU UNSEEN-TRAJECTORY TEST COMPLETE')
print('Training/holdout overlap =',TRAIN_KEYS & HOLDOUT_KEYS)
display(manifest_df.sort_values(
    ['terrain_type','test_id']
))
display(overall)
print('\nTIMING')
display(pd.DataFrame(stage_times))
print('ZIP:',zip_path)

if 'NOTEBOOK_WALL_T0' in globals():
    print(
        'TOTAL NOTEBOOK WALL TIME SO FAR: '
        f'{(time.perf_counter()-NOTEBOOK_WALL_T0)/60:.2f} min'
    )

try:
    from google.colab import files
    files.download(str(zip_path))
except Exception as e:
    print('Auto-download skipped:',e)
