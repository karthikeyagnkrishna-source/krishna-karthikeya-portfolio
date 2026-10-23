# Source inspection export of the adjacent notebook.
# Notebook execution order and dependencies still apply.
# IPython magic, shell lines and prose-only conclusion cells are comments.

# %% notebook cell 1
# ======================================================================
# 0. INSTALL / CLONE EXTERNAL BACKBONES
# Run once in a fresh Colab GPU runtime.
# ======================================================================
import os, sys, subprocess, importlib.util, json, textwrap, shutil, platform
from pathlib import Path


def pip_install(*args):
    cmd=[sys.executable,'-m','pip','install','-q',*args]
    print(' '.join(cmd))
    subprocess.check_call(cmd)

# Core packages first. Do NOT replace torch unless absolutely necessary.
pip_install('timm>=1.0.20','huggingface_hub>=0.33','safetensors','gdown',
            'rosbags','scipy','scikit-learn','pandas','matplotlib','Pillow',
            'opencv-python-headless','einops','addict','psutil','natsort','tqdm')
pip_install('git+https://github.com/castacks/tartanairpy.git')

import torch
# WAFT's official repository explicitly requires xformers. Do not let pip replace
# Colab's torch build: install a matching wheel without dependencies if absent.
try:
    import xformers  # noqa
    print('xformers:',getattr(xformers,'__version__','available'))
except Exception:
    print('Installing xformers without changing torch...')
    subprocess.check_call([sys.executable,'-m','pip','install','-q','--no-deps','xformers'])
    try:
        import xformers  # noqa
    except Exception as e:
        raise RuntimeError('WAFT requires xformers, but the installed wheel is not compatible with this Colab torch build. '
                           'Restart with a current Colab GPU runtime and rerun from the top.') from e

print('Python:',sys.version)
print('Torch :',torch.__version__)
print('CUDA  :',torch.version.cuda)
if not torch.cuda.is_available():
    raise RuntimeError('V7 requires a CUDA GPU runtime. Colab: Runtime -> Change runtime type -> GPU.')

# ---------- Utonia / PTv3 compiled dependencies ----------
def ensure_utonia_dependencies():
    cuda = torch.version.cuda
    if cuda is None:
        raise RuntimeError('CUDA-enabled PyTorch is required for Utonia.')
    cu = 'cu' + cuda.replace('.','')
    torch_ver = torch.__version__.split('+')[0]

    # spconv package names follow spconv-cu124, spconv-cu126, spconv-cu128, ...
    try:
        import spconv.pytorch as spconv  # noqa
        print('spconv already available')
    except Exception:
        print('Installing',f'spconv-{cu}')
        pip_install(f'spconv-{cu}')

    try:
        import torch_scatter  # noqa
        print('torch-scatter already available')
    except Exception:
        # PyG publishes wheels keyed by exact torch+CUDA. Try exact version first,
        # then major.minor.0 because Colab sometimes ships a patch release.
        candidates=[torch_ver]
        mm='.'.join(torch_ver.split('.')[:2])+'.0'
        if mm not in candidates: candidates.append(mm)
        errors=[]
        ok=False
        for tv in candidates:
            url=f'https://data.pyg.org/whl/torch-{tv}+{cu}.html'
            try:
                subprocess.check_call([sys.executable,'-m','pip','install','-q','torch-scatter','-f',url])
                import torch_scatter  # noqa
                print('torch-scatter installed from',url)
                ok=True; break
            except Exception as e:
                errors.append((url,repr(e)))
        if not ok:
            raise RuntimeError('Could not install a torch-scatter wheel matching this Colab torch/CUDA. '
                               f'Tried: {errors}. Do not continue without Utonia.')

ensure_utonia_dependencies()
pip_install('git+https://github.com/Pointcept/Utonia.git')

# ---------- WAFT ----------
EXT=Path('/content/v7_external')
EXT.mkdir(parents=True,exist_ok=True)

# Official ANYmal-D simplified URDF. It is used ONLY for verified leg forward
# kinematics / foot-query conditioning; camera extrinsics are not guessed from it.
ANYMAL_D_DESC=EXT/'anymal_d_simple_description'
if not ANYMAL_D_DESC.exists():
    subprocess.check_call(['git','clone','--depth','1',
                           'https://github.com/ANYbotics/anymal_d_simple_description.git',str(ANYMAL_D_DESC)])
ANYMAL_D_URDF=ANYMAL_D_DESC/'urdf'/'anymal.urdf'
if not ANYMAL_D_URDF.exists():
    raise RuntimeError('Official ANYmal-D URDF clone is incomplete.')

WAFT=EXT/'WAFT'
if not WAFT.exists():
    subprocess.check_call(['git','clone','--depth','1','--recurse-submodules','--branch','waftv2',
                           'https://github.com/princeton-vl/WAFT.git',str(WAFT)])
else:
    print('Reusing WAFT clone:',WAFT)

# Official recommended downstream a1 checkpoint.
waft_ckpt=WAFT/'ckpts'/'tar-c-t.pth'
waft_ckpt.parent.mkdir(parents=True,exist_ok=True)
if not waft_ckpt.exists():
    subprocess.check_call(['gdown','--id','1CxzBQx0iSg6AyIgt6MF0ROlF_cAeZLPC','-O',str(waft_ckpt)])

# WAFT a1 uses Depth-Anything-V2-Small as its frozen feature backbone.
da_dir=WAFT/'depth-anything-ckpts'
da_dir.mkdir(exist_ok=True)
da_ckpt=da_dir/'depth_anything_v2_vits.pth'
if not da_ckpt.exists():
    subprocess.check_call(['wget','-q','-O',str(da_ckpt),
        'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth'])

cfg=WAFT/'config'/'a1'/'tar-c-t.json'
if not cfg.exists() or not waft_ckpt.exists() or not da_ckpt.exists():
    raise RuntimeError('WAFT installation/checkpoints incomplete.')

print('Utonia import:')
import utonia
print('  OK',utonia)
print('WAFT cfg :',cfg)
print('WAFT ckpt:',waft_ckpt)
print('Install/preflight complete.')

# %% notebook cell 2
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

# -------- Storage --------
USE_GOOGLE_DRIVE_CACHE = True   # strongly recommended for long visual preprocessing
DRIVE_PROJECT_DIR = 'MyDrive/UGP_V7_FULL4B'

if USE_GOOGLE_DRIVE_CACHE:
    from google.colab import drive
    drive.mount('/content/drive')
    PERSIST_ROOT = Path('/content/drive')/DRIVE_PROJECT_DIR
else:
    PERSIST_ROOT = Path('/content/V7_PERSIST')

ROOT = Path('/content/tartanground')      # raw public data: local Colab disk
CACHE_ROOT = PERSIST_ROOT/'cache'          # expensive four-branch visual caches
OUT_ROOT = PERSIST_ROOT/'results'
CACHE_ROOT.mkdir(parents=True,exist_ok=True)
OUT_ROOT.mkdir(parents=True,exist_ok=True)

DEVICE=torch.device('cuda')
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
GEOM_REP_POINTS=64
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
print('ANYmal-D URDF:',ANYMAL_D_URDF)
print('Foot-query plan horizon:',FOOT_QUERY_PLAN_SEC,'s')
display(pd.DataFrame(EXPERIMENTS))

# %% notebook cell 3
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
        z = np.load(cache, allow_pickle=False)
        return {k: z[k] for k in z.files}

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

# %% notebook cell 4
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
        z=np.load(path,allow_pickle=False); out={k:z[k] for k in z.files}
        print('Reusing frame manifest',path); return out
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

# %% notebook cell 5
# ======================================================================
# 4. FOUR MANDATORY VISUAL CACHE BUILDERS
# Each foundation model is loaded, used, and unloaded separately.
# ======================================================================

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
    if path.exists() and meta.exists():
        arr=np.load(path,mmap_mode='r'); print('Reusing DINOv3',arr.shape); return arr
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
    if path.exists() and meta.exists():
        arr=np.load(path,mmap_mode='r'); print('Reusing ConvNeXt',arr.shape); return arr
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


def load_utonia_backbone():
    import utonia
    try:
        import flash_attn  # noqa
        model=utonia.load('utonia',repo_id='Pointcept/Utonia').to(DEVICE)
    except Exception:
        cfg=dict(enc_patch_size=[1024 for _ in range(5)],enable_flash=False)
        model=utonia.load('utonia',repo_id='Pointcept/Utonia',custom_config=cfg).to(DEVICE)
    model.eval();
    for p in model.parameters(): p.requires_grad=False
    transform=utonia.transform.default(scale=0.5,apply_z_positive=False,normalize_coord=False)
    return model,transform


def utonia_features(model,transform,coord,color,normal):
    point={'coord':coord.copy(),'color':color.copy(),'normal':normal.copy()}
    point=transform(point)
    for key in list(point.keys()):
        if isinstance(point[key],torch.Tensor): point[key]=point[key].to(DEVICE,non_blocking=True)
    with torch.inference_mode(): point=model(point)
    # Official Utonia README pattern: concatenate two upcast levels, then
    # propagate the remaining hierarchy without repeatedly exploding channels.
    for _ in range(2):
        if 'pooling_parent' not in point.keys():
            break
        if 'pooling_inverse' not in point.keys():
            raise RuntimeError('Utonia returned pooling_parent without pooling_inverse.')
        parent=point.pop('pooling_parent'); inv=point.pop('pooling_inverse')
        parent.feat=torch.cat([parent.feat,point.feat[inv]],dim=-1)
        point=parent
    while 'pooling_parent' in point.keys():
        if 'pooling_inverse' not in point.keys():
            raise RuntimeError('Utonia returned pooling_parent without pooling_inverse.')
        parent=point.pop('pooling_parent'); inv=point.pop('pooling_inverse')
        parent.feat=point.feat[inv]
        point=parent
    # GridSample changed point count; inverse maps features back to our original
    # 4096 selected metric RGB-D points.
    feat=point.feat
    if 'inverse' in point.keys(): feat=feat[point.inverse]
    feat=feat.float().cpu().numpy()
    if len(feat)!=len(coord):
        raise RuntimeError(f'Utonia feature/input count mismatch: {len(feat)} vs {len(coord)}')
    return feat


def build_utonia_cache(base,env,run,manifest):
    path=CACHE_ROOT/f'{env}_{run}_v7_utonia.npy'; meta=CACHE_ROOT/f'{env}_{run}_v7_utonia_meta.json'
    if path.exists() and meta.exists():
        arr=np.load(path,mmap_mode='r'); print('Reusing Utonia',arr.shape); return arr
    print('Loading Utonia pretrained PTv3...')
    model,transform=load_utonia_backbone(); N=len(manifest['frame_ids']); mm=None
    for j in range(N):
        fid,rgb,depth,mask,local,normals,*_=get_frame_inputs(base,manifest,j)
        coord,color,normal,rough=select_geometry_points(rgb,mask,local,normals)
        feat=utonia_features(model,transform,coord,color,normal)
        rep=farthest_point_indices(coord,GEOM_REP_POINTS)
        raw=np.concatenate([feat[rep],coord[rep],normal[rep],color[rep],rough[rep]],axis=1)
        if mm is None:
            mm=np.lib.format.open_memmap(path,mode='w+',dtype=np.float16,shape=(N,GEOM_REP_POINTS,raw.shape[-1]))
        mm[j]=raw.astype(np.float16)
        if (j+1)%20==0: mm.flush(); print(' UTONIA',j+1,'/',N)
    mm.flush(); json.dump({'model':'Pointcept/Utonia','shape':list(mm.shape)},open(meta,'w'))
    del model,transform,mm; gc.collect(); torch.cuda.empty_cache()
    return np.load(path,mmap_mode='r')


def load_waft():
    WAFT=Path('/content/v7_external/WAFT')
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
    if path.exists() and meta.exists():
        arr=np.load(path,mmap_mode='r'); print('Reusing WAFT',arr.shape); return arr
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
    for k in ['DINO','CONV','GEOM','FLOW']:
        if len(out[k])!=N or not np.isfinite(np.asarray(out[k][0],np.float32)).all():
            raise RuntimeError(f'{k} cache integrity failure.')
    return out

print('All four visual cache builders defined. No branch has a fallback.')

# %% notebook cell 6
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

# %% notebook cell 7
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


# %% notebook cell 8
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
    seed_everything_v7(seed); first=run_records[0]; dt=float(np.median(np.diff(first['data']['time']))); H=int(round(context_sec/dt))
    model=make_model_for_record(first,H,horizons)
    if warm_ckpt is not None: partial_warmstart_from_v5(model,warm_ckpt)
    opt=torch.optim.AdamW(model.parameters(),lr=LR_PRIOR,weight_decay=WEIGHT_DECAY)
    print('\n'+'='*100+'\nV7 GLOBAL PRIOR',label,'\n'+'='*100)
    for ep in range(epochs):
        vals=[]
        for rec in run_records:
            data,vis=rec['data'],rec['visual']; end=float(min(rec['ttests'])); stats=make_stats_between(data,float(data['time'][0]),end); norm=normalize_arrays(data,stats); physics=rec['physics_prior_v7']
            wins,H2=make_range_windows_v7(data,vis,float(data['time'][0]),end,context_sec,hop_sec,max(horizons)); assert H2==H
            print(f' ep {ep+1}/{epochs} {rec["terrain_type"]} windows={len(wins)}')
            for wi,(idx,_,te) in enumerate(wins,1):
                s=make_v7_sample(data,vis,stats,physics,idx,H,horizons,norm); v,p=v7_step(model,opt,s,stats,physics); vals.append(v)
                if wi in (1,len(wins)): print(f'  {wi:03d}/{len(wins):03d} target_end={te:.3f} loss={v:.4f} tau={p["tau"]:.3f} force={p["force"]:.3f}')
                del s
        print(' mean raw loss',np.mean(vals))
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

# %% notebook cell 9
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

# %% notebook cell 10
# ======================================================================
# 9. END-TO-END RUN: DATA -> 4-BRANCH CACHE -> PRIORS -> 16 TESTS
# This is the expensive cell.
# ======================================================================
seed_everything_v7(SEED)

# A) Prepare raw runs + physics + metric frame manifests.
run_records=[]
for exp in EXPERIMENTS:
    print('\n'+'#'*100); print('PREP',exp['terrain_type'],exp['environment'],exp['run']); print('#'*100)
    base=download_run_v7(exp['environment'],exp['run'])
    data=extract_run_physics(base,exp['run'])
    # Hard-check the dataset's 12 joint signals against the official ANYmal-D FK.
    fk0=ANYMAL_D_FK.foot_positions(data['P'][0,:,0:3])
    print('ANYmal-D FK sample foot positions [m]:\n',np.array2string(fk0,precision=4))
    manifest=build_frame_manifest(base,exp['environment'],exp['run'],data)
    for tt in exp['ttests']:
        if tt+1.0>float(data['time'][-1]): raise RuntimeError('ttest too close to run end')
    run_records.append(dict(**exp,base=base,data=data,manifest=manifest))

# B) Build all four mandatory visual branches sequentially.
for rec in run_records:
    env,run,base,man=rec['environment'],rec['run'],rec['base'],rec['manifest']
    print('\n'+'='*100+'\nVISUAL CACHE '+env+'/'+run+'\n'+'='*100)
    build_dinov3_cache(base,env,run,man)
    build_convnext_cache(base,env,run,man)
    build_utonia_cache(base,env,run,man)
    build_waft_cache(base,env,run,man)
    rec['visual']=load_visual_bundle(env,run,man)
    # hard four-branch check
    assert all(k in rec['visual'] for k in ('DINO','CONV','GEOM','FLOW'))

# C) Learn V5-style load regimes + soft physics from causal pre-first-test data.
initialize_regimes_and_physics(run_records)

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
prior_short=train_v7_prior(run_records,SHORT_CONTEXT_SEC,SHORT_HOP_SEC,SHORT_HORIZONS,PRIOR_EPOCHS_SHORT,SEED+10,'SHORT 0.5s -> +0.1s',V5_SHORT_CKPT)
torch.save(prior_short.state_dict(),OUT_ROOT/'V7_PRIOR_SHORT.pt')
prior_long=train_v7_prior(run_records,LONG_CONTEXT_SEC,LONG_HOP_SEC,LONG_HORIZONS,PRIOR_EPOCHS_LONG,SEED+20,'LONG 4s -> +0.2..+1.0s',V5_LONG_CKPT)
torch.save(prior_long.state_dict(),OUT_ROOT/'V7_PRIOR_LONG.pt')

# F) 4x4 exact tests. Each test gets a fresh copy of the corresponding global prior.
all_values=[]; all_reg=[]; all_force=[]; test_manifest=[]
for ri,rec in enumerate(run_records):
    for ti,tt in enumerate(rec['ttests'],1):
        print('\n'+'-'*100+f'\nTEST {rec["terrain_type"]} #{ti} t={tt:.1f}\n'+'-'*100)
        seed=SEED+ri*1000+ti*100
        sm,ss,sp,Hs=adapt_v7(prior_short,rec['data'],rec['visual'],tt,SHORT_CONTEXT_SEC,SHORT_HOP_SEC,SHORT_HORIZONS,SHORT_RECENCY_HALF_LIFE_SEC,LOCAL_EPOCHS_SHORT,seed+1,'SHORT')
        lm,ls,lp,Hl=adapt_v7(prior_long,rec['data'],rec['visual'],tt,LONG_CONTEXT_SEC,LONG_HOP_SEC,LONG_HORIZONS,LONG_RECENCY_HALF_LIFE_SEC,LOCAL_EPOCHS_LONG,seed+2,'LONG')
        vals,regs,forces,idx,vi=evaluate_v7_case(sm,ss,sp,Hs,lm,ls,lp,Hl,rec['data'],rec['visual'],tt)
        meta=dict(terrain_type=rec['terrain_type'],environment=rec['environment'],run=rec['run'],test_id=ti,ttest_s=float(tt))
        for k,v in meta.items(): vals[k]=v; regs[k]=v; forces[k]=v
        all_values.append(vals); all_reg.append(regs); all_force.append(forces)
        test_manifest.append(dict(**meta,robot_idx=idx,visual_idx=vi,visual_time=float(rec['visual']['camera_times'][vi]),frame_id=str(rec['visual']['frame_ids'][vi])))
        del sm,lm; gc.collect(); torch.cuda.empty_cache()

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

# %% notebook cell 11
# ======================================================================
# 10. FINAL AUDIT, README, CHECKSUM MANIFEST, MASTER ZIP + DOWNLOAD
# ======================================================================
import shutil, hashlib, datetime, json

# Save exact configuration.
config={k:v for k,v in globals().items() if k in [
    'SEED','TRAIN_LOOKBACK_SEC','SHORT_CONTEXT_SEC','LONG_CONTEXT_SEC','SHORT_HOP_SEC','LONG_HOP_SEC',
    'SHORT_HORIZONS','LONG_HORIZONS','PRIOR_EPOCHS_SHORT','PRIOR_EPOCHS_LONG','LOCAL_EPOCHS_SHORT','LOCAL_EPOCHS_LONG',
    'LR_PRIOR','LR_LOCAL','DINO_MODEL','CONV_MODEL','VISION_INPUT','RAW_GRID','LATENTS_PER_BRANCH','VIS_DIM','FLOW_HISTORY',
    'POINTS_PER_FRAME','CONTACT_THRESHOLD','PI_PHYS_DIM']}
with open(OUT_ROOT/'V7_CONFIG.json','w') as f: json.dump(config,f,indent=2,default=str)

readme=f"""# V7 Full Four-Branch Experiment\n\nFinal policy: +0.1 s = SHORT only; +0.2..1.0 s = LONG only.\n\nMandatory vision branches:\n- {DINO_MODEL}\n- {CONV_MODEL}\n- Pointcept/Utonia pretrained PTv3\n- WAFT a1 tar-c-t optical flow + temporal CNN\n\nPhysics is a causal identified soft proxy, not exact ANYmal rigid-body inverse dynamics.\nFuture recorded U/L are conditional exogenous controls in this offline benchmark and must be replaced by planned controller commands at deployment.\nThe 16 fixed tests are a controlled diagnostic comparison to V4/V5/V6, not a claim of held-out terrain generalization.\n"""
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
results_zip=Path('/content')/f'V7_RESULTS_ONLY_{stamp}.zip'
with zipfile.ZipFile(results_zip,'w',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as z:
    for p in sorted(OUT_ROOT.rglob('*')):
        if p.is_file(): z.write(p,arcname=str(Path('results')/p.relative_to(OUT_ROOT)))
print('RESULTS ZIP:',results_zip,results_zip.stat().st_size/(1024**2),'MB')

# Full master ZIP streams directly from the persistent result/cache roots. This avoids making a second
# multi-GB copy in /content. It excludes the raw redownloadable TartanGround directory.
BUILD_FULL_CACHE_ZIP=True
full_zip=PERSIST_ROOT/f'V7_FULL4B_RESULTS_AND_4BRANCH_CACHES_{stamp}.zip'
if BUILD_FULL_CACHE_ZIP:
    with zipfile.ZipFile(full_zip,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for root,name in [(OUT_ROOT,'results'),(CACHE_ROOT,'cache')]:
            for p in sorted(root.rglob('*')):
                if p.is_file(): z.write(p,arcname=str(Path(name)/p.relative_to(root)))
    print('FULL CACHE ZIP:',full_zip,full_zip.stat().st_size/(1024**3),'GB')

from google.colab import files
files.download(str(results_zip))
print('The full cache ZIP remains at:',full_zip)
