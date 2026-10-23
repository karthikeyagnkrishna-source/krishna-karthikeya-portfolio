# Source inspection export of the adjacent notebook.
# Notebook execution order and dependencies still apply.
# IPython magic, shell lines and prose-only conclusion cells are comments.

# %% notebook cell 0
_VIS_MODELS = None

# %% notebook cell 1

# ======================================================================
# STANDALONE V4 EXPERIMENT
#
# CHANGES FROM THE PREVIOUS MULTITERRAIN NOTEBOOK:
#
# 1) RESIDUAL / DELTA prediction:
#      baseline q(t+h) = q(t) + h*dq(t)
#      baseline dq,tau,F,B = persistence
#      Transformer learns only correction to baseline.
#
# 2) PRETRAINED PRIOR:
#      one global short prior + one global long prior are trained first
#      on causal pre-test data from all four terrain runs.
#      Every t_test then creates a NEW model instance by loading the
#      prior weights, followed by local chronological adaptation.
#
# 3) NO 100-STEP AUTOREGRESSIVE LONG ROLLOUT:
#      4-s long expert predicts h = 0.1,...,1.0 s DIRECTLY in one pass.
#
# LOCAL ADAPTATION:
#      oldest window -> weighted loss -> backward -> optimizer.step()
#      ...
#      newest window -> largest weight -> optimizer.step()
#
# NO BATCHES. NO SHUFFLE.
# ======================================================================

import sys, subprocess, importlib.util

def _ensure(import_name, pip_spec):
    if importlib.util.find_spec(import_name) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_spec])

_ensure("tartanair", "git+https://github.com/castacks/tartanairpy.git")
_ensure("rosbags", "rosbags")
_ensure("transformers", "transformers")
_ensure("huggingface_hub", "huggingface_hub")

import os, re, gc, math, random, copy
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from rosbags.highlevel import AnyReader

import tartanair as ta
from huggingface_hub import HfApi
from transformers import AutoImageProcessor, AutoModel
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

# ----------------------------- CONFIG ---------------------------------
ROOT = Path('/content/tartanground')

# Reuse previous cache files if they are still present in this runtime.
CACHE_ROOT = Path('/content/v3_multiterrain_cache')
OUT_ROOT = Path('/content/v4_residual_prior_direct_results')
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
OUT_ROOT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 781
seed_everything = None  # replaced below by helper block

TRAIN_LOOKBACK_SEC = 20.0

SHORT_CONTEXT_SEC = 0.5
SHORT_HOP_SEC = 0.3
SHORT_HORIZONS = [0.1]

LONG_CONTEXT_SEC = 4.0
LONG_HOP_SEC = 1.0
LONG_HORIZONS = [round(x,1) for x in np.arange(0.1, 1.01, 0.1)]

SHORT_RECENCY_HALF_LIFE_SEC = 5.0
LONG_RECENCY_HALF_LIFE_SEC = 6.0

# Prior is trained first; local stage then does strict chronological updates.
PRIOR_EPOCHS_SHORT = 2
PRIOR_EPOCHS_LONG = 3
LOCAL_EPOCHS_SHORT = 4
LOCAL_EPOCHS_LONG = 4

LR_PRIOR = 2e-4
LR_LOCAL = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

# Same Gaussian reliability concept.
SHORT_NATIVE_HORIZON = 0.1
LONG_NATIVE_HORIZON = 1.0
SIGMA_SHORT = 0.28
SIGMA_LONG = 0.42

CONTACT_THRESHOLD = 10.0
EPS = 1e-6

ROI_X_MIN, ROI_X_MAX = 0.18, 0.82
ROI_Y_MIN, ROI_Y_MAX = 0.45, 0.97
DEPTH_MIN_METERS, DEPTH_MAX_METERS = 0.25, 2.0
DEPTH_SIZE = 64
VIS_BATCH = 16

# EXACT same four terrain/run choices and exact same four test times
# from the completed previous experiment.
EXPERIMENTS = [
    dict(
        terrain_type='forest',
        environment='ForestEnv',
        run='P2000',
        ttests=[28.0,45.0,75.0,90.0],
    ),
    dict(
        terrain_type='swamp_marsh',
        environment='GreatMarsh',
        run='P2023',
        ttests=[47.0,61.0,73.0,87.0],
    ),
    dict(
        terrain_type='rough_uneven',
        environment='SeasonalForestWinter',
        run='P2000',
        ttests=[29.0,46.0,64.0,92.0],
    ),
    dict(
        terrain_type='wet_uneven_proxy',
        environment='OldTownFall',
        run='P2000',
        ttests=[32.0,84.0,100.0,115.0],
    ),
]


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

def get_visual_models():
    global _VIS_MODELS
    if _VIS_MODELS is not None: return _VIS_MODELS
    dino_name='facebook/dinov2-small'
    proc=AutoImageProcessor.from_pretrained(dino_name)
    dino=AutoModel.from_pretrained(dino_name).to(DEVICE).eval()
    for p in dino.parameters(): p.requires_grad=False
    weights=ConvNeXt_Tiny_Weights.DEFAULT
    conv=convnext_tiny(weights=weights).to(DEVICE); conv.classifier=nn.Identity(); conv.eval()
    for p in conv.parameters(): p.requires_grad=False
    _VIS_MODELS=(proc,dino,conv,weights.transforms())
    return _VIS_MODELS


def prepare_visual_features(base, env, run, physics):
    cache=CACHE_ROOT/f'{env}_{run}_visual_v3.npz'
    if cache.exists():
        z=np.load(cache,allow_pickle=False)
        return {k:z[k] for k in z.files}

    rgb_dir=base/'image_lcam_front'; depth_dir=base/'depth_lcam_front'; cam_file=base/'imu'/'cam_time.txt'
    rgb_ids={f.name.split('_')[0] for f in rgb_dir.glob('*_lcam_front.png')}
    dep_ids={f.name.split('_')[0] for f in depth_dir.glob('*_lcam_front_depth.png')}
    frame_ids_all=sorted(rgb_ids & dep_ids,key=lambda x:int(x))
    nums=np.asarray([int(x) for x in frame_ids_all],int)
    cam_orig=np.loadtxt(cam_file).reshape(-1).astype(np.float64)
    cam_times_all=cam_orig[nums]-float(physics['crop_start_original'])
    valid=(cam_times_all>=physics['time'][0]-1e-8)&(cam_times_all<=physics['time'][-1]+1e-8)
    frame_ids=np.asarray(frame_ids_all)[valid].tolist(); cam_times=cam_times_all[valid]

    proc,dino,conv,conv_transform=get_visual_models()
    all_dino=[]; all_conv=[]; all_depth=[]; rough=[]
    for bs in range(0,len(frame_ids),VIS_BATCH):
        batch_ids=frame_ids[bs:bs+VIS_BATCH]; pil_batch=[]; conv_batch=[]; depth_batch=[]; rough_batch=[]
        for fid in batch_ids:
            rgb=np.asarray(Image.open(rgb_dir/f'{fid}_lcam_front.png').convert('RGB'))
            depth=load_metric_depth(depth_dir/f'{fid}_lcam_front_depth.png')
            H,W=rgb.shape[:2]; x0=int(ROI_X_MIN*W); x1=int(ROI_X_MAX*W); y0=int(ROI_Y_MIN*H); y1=int(ROI_Y_MAX*H)
            rr=rgb[y0:y1,x0:x1].copy(); dd=depth[y0:y1,x0:x1].copy()
            valid_d=np.isfinite(dd)&(dd>=DEPTH_MIN_METERS)&(dd<=DEPTH_MAX_METERS)
            rr[~valid_d]=0; dd[~valid_d]=0.0
            pil=Image.fromarray(rr); pil_batch.append(pil); conv_batch.append(conv_transform(pil))
            small=cv2.resize(dd,(DEPTH_SIZE,DEPTH_SIZE),interpolation=cv2.INTER_AREA)
            small=np.clip(small,0,DEPTH_MAX_METERS)/DEPTH_MAX_METERS
            depth_batch.append(small[None,:,:])
            v=small[small>0]
            if len(v)<20: rough_batch.append(0.0)
            else:
                gy,gx=np.gradient(small); rough_batch.append(float(np.std(v)+np.mean(np.sqrt(gx*gx+gy*gy))))
        din_in=proc(images=pil_batch,return_tensors='pt').to(DEVICE)
        with torch.no_grad(): dfeat=dino(**din_in).last_hidden_state[:,0,:]
        ct=torch.stack(conv_batch).to(DEVICE)
        with torch.no_grad(): cfeat=conv(ct).flatten(1)
        all_dino.append(dfeat.cpu().numpy().astype(np.float32)); all_conv.append(cfeat.cpu().numpy().astype(np.float32))
        all_depth.append(np.asarray(depth_batch,np.float32)); rough.extend(rough_batch)
    D=np.concatenate(all_dino); C=np.concatenate(all_conv); G=np.concatenate(all_depth); rough=np.asarray(rough,np.float32)
    vi=np.searchsorted(cam_times,physics['time'],side='right')-1
    out=dict(DINO=D,CONV=C,DEPTH=G,camera_times=cam_times.astype(np.float64),visual_index=vi.astype(np.int64),
             frame_ids=np.asarray(frame_ids),roughness=rough)
    np.savez_compressed(cache,**out)
    return out


# ------------------------- DOWNLOAD / REUSE ----------------------------
def download_run(env, run):
    base = ROOT / env / 'Data_anymal' / run
    required = [
        base/'imu',
        base/'rosbags'/f'{run}.bag',
        base/'image_lcam_front',
        base/'depth_lcam_front',
    ]
    if all(p.exists() for p in required):
        print(f'Reusing existing data: {env}/{run}')
        return base

    print(f'Downloading missing data: {env}/{run}')
    ta.init(str(ROOT))
    ta.download_ground(
        env=[env],
        version=['anymal'],
        traj=[run],
        modality=['imu','rosbag','image','depth'],
        camera_name=['lcam_front'],
        unzip=True,
    )
    for p in required:
        if not p.exists():
            raise RuntimeError(f'Missing downloaded path: {p}')
    return base


# ------------------------- NORMALIZATION -------------------------------
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


# --------------------------- MODEL -------------------------------------
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


class TerrainEncoder(nn.Module):
    def __init__(self, dino_dim, conv_dim):
        super().__init__()
        self.dino_proj = nn.Sequential(
            nn.Linear(dino_dim,256), nn.GELU(), nn.LayerNorm(256)
        )
        self.conv_proj = nn.Sequential(
            nn.Linear(conv_dim,256), nn.GELU(), nn.LayerNorm(256)
        )
        self.depth_net = nn.Sequential(
            nn.Conv2d(1,32,5,2,2), nn.GELU(),
            nn.Conv2d(32,64,3,2,1), nn.GELU(),
            nn.Conv2d(64,128,3,2,1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.depth_proj = nn.Sequential(
            nn.Linear(128,256), nn.GELU(), nn.LayerNorm(256)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
            activation='gelu',
        )
        self.fusion = nn.TransformerEncoder(layer, num_layers=2)
        self.type_embedding = nn.Parameter(torch.randn(1,3,256)*0.02)
        self.norm = nn.LayerNorm(256)

    def forward(self, dino, conv, depth):
        toks = torch.stack([
            self.dino_proj(dino),
            self.conv_proj(conv.flatten(1)),
            self.depth_proj(self.depth_net(depth).flatten(1)),
        ], dim=1)
        toks = self.fusion(toks + self.type_embedding)
        return self.norm(toks.mean(1))


class DirectResidualDynamics(nn.Module):
    """
    History + terrain + known future controls -> DIRECT residual predictions.

    No sensor state is autoregressively fed back.
    The output is a residual in normalized units relative to a simple
    physics/persistence baseline.
    """
    def __init__(self, history_steps, n_horizons, dino_dim=384, conv_dim=768):
        super().__init__()
        self.history_steps = int(history_steps)
        self.n_horizons = int(n_horizons)

        # Same V3 history content:
        # per leg: P12 + U6 + L3 + E3 = 24
        # body:    B12 + U6 = 18
        self.leg_encoder = CausalHistoryEncoder(24, history_steps, 128, 2, 4)
        self.leg_identity = nn.Embedding(4, 128)
        self.body_encoder = CausalHistoryEncoder(18, history_steps, 128, 2, 4)
        self.terrain_encoder = TerrainEncoder(dino_dim, conv_dim)

        self.context_net = nn.Sequential(
            nn.LayerNorm(896),
            nn.Linear(896,512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512,256),
            nn.GELU(),
            nn.LayerNorm(256),
        )

        # For each requested horizon we provide:
        # endpoint U6 + endpoint L12 + mean U6 + mean L12 = 36.
        self.control_proj = nn.Sequential(
            nn.Linear(36,128),
            nn.GELU(),
            nn.LayerNorm(128),
        )

        self.horizon_proj = nn.Sequential(
            nn.Linear(1,64),
            nn.GELU(),
            nn.Linear(64,128),
            nn.GELU(),
            nn.LayerNorm(128),
        )

        # context256 + control128 + horizon128 = 512
        self.response_head = nn.Sequential(
            nn.Linear(512,256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256,60),  # P=48 + B=12 residuals
        )
        self.contact_head = nn.Sequential(
            nn.Linear(512,128),
            nn.GELU(),
            nn.Linear(128,4),
        )

    def encode_context(self, p_hist, b_hist, u_hist, l_hist, e_hist, dino, conv, depth):
        Bn = p_hist.shape[0]
        T = p_hist.shape[2]

        ulegs = u_hist.unsqueeze(1).expand(-1,4,-1,-1)
        leg_in = torch.cat(
            [p_hist, ulegs, l_hist, e_hist],
            dim=-1
        ).reshape(Bn*4, T, 24)

        lf = self.leg_encoder(leg_in)
        ids = torch.arange(4, device=p_hist.device).repeat(Bn)
        lf = (lf + self.leg_identity(ids)).reshape(Bn,4,128).reshape(Bn,512)

        bf = self.body_encoder(torch.cat([b_hist,u_hist], dim=-1))
        terrain = self.terrain_encoder(dino,conv,depth)

        return self.context_net(torch.cat([lf,bf,terrain], dim=-1))

    def forward(
        self,
        p_hist, b_hist, u_hist, l_hist, e_hist,
        dino, conv, depth,
        future_control_features,
        horizon_values,
    ):
        ctx = self.encode_context(
            p_hist,b_hist,u_hist,l_hist,e_hist,dino,conv,depth
        )

        Hn = future_control_features.shape[1]
        if Hn != self.n_horizons:
            raise RuntimeError(
                f'Model expects {self.n_horizons} horizons, got {Hn}'
            )

        c = self.control_proj(future_control_features)
        h = self.horizon_proj(horizon_values)
        ctx = ctx.unsqueeze(1).expand(-1,Hn,-1)

        z = torch.cat([ctx,c,h], dim=-1)
        delta = self.response_head(z)

        residual_p = delta[...,:48].reshape(-1,Hn,4,12)
        residual_b = delta[...,48:]
        contact_logits = self.contact_head(z)

        return {
            'residual_p': residual_p,
            'residual_b': residual_b,
            'contact_logits': contact_logits,
        }


# ------------------------- SAMPLE BUILDING -----------------------------
def tensor(x):
    return torch.as_tensor(x, dtype=torch.float32, device=DEVICE)


def horizon_offsets(data, horizons):
    dt = float(np.median(np.diff(data['time'])))
    offs = [int(round(float(h)/dt)) for h in horizons]
    if any(o < 1 for o in offs):
        raise RuntimeError('Horizon offset < 1 sample.')
    return offs


def make_direct_sample(data, visual, stats, idx, history_steps, horizons, norm=None):
    """
    Input uses only sensor history ending at idx and the causal image at idx.
    Future U/L are allowed exogenous controls for this OFFLINE dynamics benchmark.
    Future P/B/contact are targets only.
    """
    if norm is None:
        Pn,Bn,Un,Ln,En = normalize_arrays(data,stats)
    else:
        Pn,Bn,Un,Ln,En = norm
    h0 = int(idx) - int(history_steps) + 1
    if h0 < 0:
        raise RuntimeError('History starts before data.')

    offs = horizon_offsets(data,horizons)
    max_off = max(offs)
    if idx + max_off >= len(data['time']):
        raise RuntimeError('Future target outside run.')

    vi = int(visual['visual_index'][idx])
    if vi < 0:
        raise RuntimeError('No causal image for sample.')

    current_p = data['P'][idx].astype(np.float32)
    current_b = data['B'][idx].astype(np.float32)

    baseline_p = []
    baseline_b = []
    target_resid_p = []
    target_resid_b = []
    target_contact = []
    control_features = []

    for h,off in zip(horizons,offs):
        target_idx = idx + off

        bp = current_p.copy()
        # Constant-velocity baseline only for q.
        bp[:,0:3] = (
            current_p[:,0:3]
            + float(h) * current_p[:,3:6]
        )
        # dq, measured tau and force remain persistence baselines.
        bb = current_b.copy()

        tp = data['P'][target_idx].astype(np.float32)
        tb = data['B'][target_idx].astype(np.float32)

        rp = (tp - bp) / stats['P_std'][None,:]
        rb = (tb - bb) / stats['B_std']

        # Commands from [t, t+h): no future sensor truth.
        u_segment = Un[idx:target_idx]
        l_segment = Ln[idx:target_idx]
        if len(u_segment) == 0:
            u_segment = Un[idx:idx+1]
            l_segment = Ln[idx:idx+1]

        end_u = u_segment[-1]
        end_l = l_segment[-1].reshape(-1)
        mean_u = u_segment.mean(axis=0)
        mean_l = l_segment.mean(axis=0).reshape(-1)

        ctrl = np.concatenate(
            [end_u,end_l,mean_u,mean_l],
            axis=0
        ).astype(np.float32)

        baseline_p.append(bp)
        baseline_b.append(bb)
        target_resid_p.append(rp.astype(np.float32))
        target_resid_b.append(rb.astype(np.float32))
        target_contact.append(
            data['contact'][target_idx].astype(np.float32)
        )
        control_features.append(ctrl)

    horizon_values = (
        np.asarray(horizons,dtype=np.float32)[:,None]
        / max(LONG_HORIZONS)
    )

    return {
        'p_hist': tensor(Pn[h0:idx+1].transpose(1,0,2)[None]),
        'b_hist': tensor(Bn[h0:idx+1][None]),
        'u_hist': tensor(Un[h0:idx+1][None]),
        'l_hist': tensor(Ln[h0:idx+1].transpose(1,0,2)[None]),
        'e_hist': tensor(En[h0:idx+1].transpose(1,0,2)[None]),

        'dino': tensor(visual['DINO'][vi][None]),
        'conv': tensor(visual['CONV'][vi][None]),
        'depth': tensor(visual['DEPTH'][vi][None]),

        'future_control_features': tensor(
            np.asarray(control_features,np.float32)[None]
        ),
        'horizon_values': tensor(horizon_values[None]),

        'target_residual_p': tensor(
            np.asarray(target_resid_p,np.float32)[None]
        ),
        'target_residual_b': tensor(
            np.asarray(target_resid_b,np.float32)[None]
        ),
        'target_contact': tensor(
            np.asarray(target_contact,np.float32)[None]
        ),

        'baseline_p_raw': np.asarray(baseline_p,np.float32),
        'baseline_b_raw': np.asarray(baseline_b,np.float32),
        'target_p_raw': np.asarray(
            [data['P'][idx+o] for o in offs],
            np.float32
        ),
        'target_b_raw': np.asarray(
            [data['B'][idx+o] for o in offs],
            np.float32
        ),
        'target_contact_raw': np.asarray(
            [data['contact'][idx+o] for o in offs],
            np.float32
        ),
        'vi': vi,
        'idx': int(idx),
    }


# ---------------------------- LOSS -------------------------------------
BCE = nn.BCEWithLogitsLoss(reduction='none')

def direct_residual_loss(out, sample):
    pp = out['residual_p']
    pb = out['residual_b']
    tp = sample['target_residual_p']
    tb = sample['target_residual_b']

    Hn = pp.shape[1]
    # Long expert still emphasizes farther horizons slightly.
    hw = torch.linspace(1.0,2.0,Hn,device=pp.device)

    def hmse(a,b):
        e = (a-b)**2
        dims = tuple(range(2,e.ndim))
        return (e.mean(dim=dims) * hw[None,:]).mean()

    lq = hmse(pp[:,:,:,0:3], tp[:,:,:,0:3])
    ldq = hmse(pp[:,:,:,3:6], tp[:,:,:,3:6])
    ltau = hmse(pp[:,:,:,6:9], tp[:,:,:,6:9])
    lforce = hmse(pp[:,:,:,9:12], tp[:,:,:,9:12])
    lbody = hmse(pb,tb)

    ce = BCE(
        out['contact_logits'],
        sample['target_contact']
    ).mean(-1)
    lcontact = (ce * hw[None,:]).mean()

    return (
        lq + ldq + ltau
        + 1.25*lforce
        + 0.5*lbody
        + 0.25*lcontact
    )


# ------------------------ WINDOW GENERATORS ----------------------------
def make_range_windows(
    data,visual,start_target_time,end_target_time,
    context_sec,hop_sec,max_horizon
):
    dt = float(np.median(np.diff(data['time'])))
    H = int(round(context_sec/dt))

    # "target end" refers to the furthest requested horizon.
    te = max(
        float(start_target_time),
        float(data['time'][0]) + context_sec + max_horizon
    )

    rows = []
    seen = set()

    while te <= float(end_target_time) + 1e-9:
        current_end = te - max_horizon
        idx = nearest_index(data['time'], current_end)

        if idx not in seen:
            seen.add(idx)
            if (
                idx-H+1 >= 0
                and idx + int(round(max_horizon/dt)) < len(data['time'])
                and int(visual['visual_index'][idx]) >= 0
            ):
                actual_target_end = float(
                    data['time'][
                        idx + int(round(max_horizon/dt))
                    ]
                )
                if actual_target_end <= float(end_target_time) + 0.015:
                    rows.append(
                        (idx,float(data['time'][idx]),actual_target_end)
                    )
        te += hop_sec

    if not rows:
        raise RuntimeError(
            f'No prior windows: {start_target_time} -> {end_target_time}'
        )

    rows.sort(key=lambda x:x[2])
    return rows,H


def make_local_windows(
    data,visual,ttest,
    context_sec,hop_sec,max_horizon,half_life
):
    dt = float(np.median(np.diff(data['time'])))
    H = int(round(context_sec/dt))

    target_ends = []
    te = float(ttest)

    # Build backwards so the newest target end is exactly ttest.
    while (
        te - max_horizon - context_sec
        >= float(ttest) - TRAIN_LOOKBACK_SEC - 1e-9
    ):
        target_ends.append(te)
        te -= hop_sec

    rows = []
    seen = set()

    for desired_target_end in reversed(target_ends):
        current_end = desired_target_end - max_horizon
        idx = nearest_index(data['time'], current_end)

        if idx in seen:
            continue
        seen.add(idx)

        if idx-H+1 < 0:
            continue

        max_off = int(round(max_horizon/dt))
        if idx + max_off >= len(data['time']):
            continue

        if int(visual['visual_index'][idx]) < 0:
            continue

        actual_target_end = float(data['time'][idx+max_off])
        if actual_target_end > float(ttest)+0.015:
            continue

        age = max(0.0,float(ttest)-actual_target_end)
        weight = 0.5**(age/half_life)

        rows.append(
            (
                idx,
                weight,
                float(data['time'][idx]),
                actual_target_end,
            )
        )

    if not rows:
        raise RuntimeError('No local adaptation windows.')

    rows.sort(key=lambda x:x[3])

    # Newest weight is 1.0, and weights increase with time.
    wmax = max(r[1] for r in rows)
    rows = [
        (idx,w/wmax,cur,te)
        for idx,w,cur,te in rows
    ]

    if not np.all(
        np.diff([r[3] for r in rows]) >= -1e-10
    ):
        raise RuntimeError('Local windows not chronological.')

    if not np.all(
        np.diff([r[1] for r in rows]) >= -1e-10
    ):
        raise RuntimeError('Local weights not increasing.')

    return rows,H


# ---------------------------- TRAINING ---------------------------------
def model_step(model,optimizer,sample,weight=1.0):
    optimizer.zero_grad(set_to_none=True)

    out = model(
        sample['p_hist'],
        sample['b_hist'],
        sample['u_hist'],
        sample['l_hist'],
        sample['e_hist'],
        sample['dino'],
        sample['conv'],
        sample['depth'],
        sample['future_control_features'],
        sample['horizon_values'],
    )

    raw_loss = direct_residual_loss(out,sample)
    loss = raw_loss * float(weight)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        GRAD_CLIP
    )
    optimizer.step()

    value = float(raw_loss.detach().cpu())
    del out,loss,raw_loss
    return value


def train_global_prior(
    run_records,
    context_sec,
    hop_sec,
    horizons,
    epochs,
    seed,
    label,
):
    """
    Global prior uses ONLY data up to the first t_test of each run.
    Therefore it never uses post-test sensor truth.
    """
    seed_everything(seed)

    first = run_records[0]
    dt = float(np.median(np.diff(first['data']['time'])))
    H = int(round(context_sec/dt))

    model = DirectResidualDynamics(
        history_steps=H,
        n_horizons=len(horizons),
        dino_dim=first['visual']['DINO'].shape[1],
        conv_dim=first['visual']['CONV'].reshape(
            len(first['visual']['CONV']),-1
        ).shape[1],
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR_PRIOR,
        weight_decay=WEIGHT_DECAY
    )

    max_h = max(horizons)

    print('\n'+'='*90)
    print(f'GLOBAL PRIOR :: {label}')
    print(f'context={context_sec}s | horizons={horizons}')
    print('NO batching / NO shuffle / chronological within each run')
    print('='*90)

    for epoch in range(epochs):
        losses = []

        for rec in run_records:
            data = rec['data']
            visual = rec['visual']
            first_ttest = min(rec['ttests'])

            stats = make_stats_between(
                data,
                float(data['time'][0]),
                float(first_ttest)
            )

            norm = normalize_arrays(data,stats)

            windows,Hcheck = make_range_windows(
                data,
                visual,
                start_target_time=float(data['time'][0]),
                end_target_time=float(first_ttest),
                context_sec=context_sec,
                hop_sec=hop_sec,
                max_horizon=max_h,
            )

            if Hcheck != H:
                raise RuntimeError('History length mismatch across runs.')

            print(
                f'  prior epoch {epoch+1}/{epochs} | '
                f'{rec["terrain_type"]} | windows={len(windows)}'
            )

            # Strict oldest -> newest within this run.
            for wi,(idx,_,target_end) in enumerate(windows,1):
                sample = make_direct_sample(
                    data,visual,stats,idx,H,horizons,norm=norm
                )
                value = model_step(
                    model,optimizer,sample,weight=1.0
                )
                losses.append(value)
                del sample

                if wi in (1,len(windows)):
                    print(
                        f'    window {wi:03d}/{len(windows):03d} '
                        f'target_end={target_end:.3f} '
                        f'loss={value:.5f}'
                    )

        print(
            f'  PRIOR {label} epoch {epoch+1} mean loss='
            f'{np.mean(losses):.6f}'
        )

    return model


def adapt_fresh_local_model(
    prior_model,
    data,
    visual,
    ttest,
    context_sec,
    hop_sec,
    horizons,
    half_life,
    epochs,
    seed,
    label,
):
    """
    Fresh model object for every test case.
    It starts from pretrained prior weights, NOT random weights.
    Every individual window performs optimizer.step().
    """
    seed_everything(seed)

    # NEW object; weights copied from prior.
    model = copy.deepcopy(prior_model).to(DEVICE)

    stats = make_local_stats(data,ttest)
    max_h = max(horizons)

    windows,H = make_local_windows(
        data,visual,ttest,
        context_sec,hop_sec,max_h,half_life
    )

    if H != model.history_steps:
        raise RuntimeError(
            f'{label}: prior/local history mismatch {model.history_steps} vs {H}'
        )

    norm = normalize_arrays(data,stats)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR_LOCAL,
        weight_decay=WEIGHT_DECAY
    )

    print(
        f'\n  LOCAL {label}: ttest={ttest:.3f} '
        f'context={context_sec}s hop={hop_sec}s '
        f'windows={len(windows)} '
        f'oldest_w={windows[0][1]:.5f} newest_w={windows[-1][1]:.5f}'
    )

    for epoch in range(epochs):
        losses = []

        # STRICT chronological order every pass.
        for wi,(idx,weight,_,target_end) in enumerate(windows,1):
            sample = make_direct_sample(
                data,visual,stats,idx,H,horizons,norm=norm
            )

            value = model_step(
                model,optimizer,sample,weight=weight
            )
            losses.append(value)

            if wi <= 2 or wi > len(windows)-2:
                print(
                    f'    {label} ep {epoch+1:02d} | '
                    f'window {wi:03d}/{len(windows):03d} | '
                    f'target_end={target_end:.3f}s | '
                    f'w={weight:.5f} | raw_loss={value:.6f}'
                )

            del sample

        print(
            f'    {label} epoch {epoch+1}/{epochs} '
            f'mean raw loss={np.mean(losses):.6f}'
        )

    return model,stats,H


# -------------------------- PREDICTION ---------------------------------
def predict_raw(model,sample,stats):
    model.eval()
    with torch.no_grad():
        out = model(
            sample['p_hist'],
            sample['b_hist'],
            sample['u_hist'],
            sample['l_hist'],
            sample['e_hist'],
            sample['dino'],
            sample['conv'],
            sample['depth'],
            sample['future_control_features'],
            sample['horizon_values'],
        )

    rp = out['residual_p'][0].detach().cpu().numpy()
    rb = out['residual_b'][0].detach().cpu().numpy()
    cp = torch.sigmoid(
        out['contact_logits'][0]
    ).detach().cpu().numpy()

    pred_p = (
        sample['baseline_p_raw']
        + rp * stats['P_std'][None,None,:]
    )
    pred_b = (
        sample['baseline_b_raw']
        + rb * stats['B_std'][None,:]
    )

    return pred_p,pred_b,cp


def gaussian_mix_weights(h):
    gs = math.exp(
        -0.5*((float(h)-SHORT_NATIVE_HORIZON)/SIGMA_SHORT)**2
    )
    gl = math.exp(
        -0.5*((float(h)-LONG_NATIVE_HORIZON)/SIGMA_LONG)**2
    )
    z = gs+gl+1e-12
    return gs/z,gl/z


def evaluate_case(
    short_model,short_stats,Hs,
    long_model,long_stats,Hl,
    data,visual,ttest
):
    idx = nearest_index(data['time'],ttest)

    short_norm = normalize_arrays(data,short_stats)
    long_norm = normalize_arrays(data,long_stats)

    short_sample = make_direct_sample(
        data,visual,short_stats,idx,Hs,SHORT_HORIZONS,norm=short_norm
    )
    long_sample = make_direct_sample(
        data,visual,long_stats,idx,Hl,LONG_HORIZONS,norm=long_norm
    )

    sp,sb,sc = predict_raw(
        short_model,short_sample,short_stats
    )
    lp,lb,lc = predict_raw(
        long_model,long_sample,long_stats
    )

    # Short expert has one native forecast at 0.1 s.
    # Convert it into an "urgent correction" to the baseline.
    short_delta_p = (
        sp[0] - short_sample['baseline_p_raw'][0]
    )
    short_delta_b = (
        sb[0] - short_sample['baseline_b_raw'][0]
    )

    leg_names = ['FL','FR','RL','RR']
    feat = [
        'q_HAA','q_HFE','q_KFE',
        'dq_HAA','dq_HFE','dq_KFE',
        'tau_HAA','tau_HFE','tau_KFE',
        'Fx','Fy','Fz'
    ]
    body = [
        'acc_x','acc_y','acc_z',
        'gyro_x','gyro_y','gyro_z',
        'body_vx','body_vy','body_vz',
        'body_wx','body_wy','body_wz'
    ]

    rows = []

    for hi,h in enumerate(LONG_HORIZONS):
        baseline_p = long_sample['baseline_p_raw'][hi]
        baseline_b = long_sample['baseline_b_raw'][hi]
        baseline_c = data['contact'][idx].astype(np.float32)

        # Short model is never autoregressed.
        # Its learned 0.1-s residual correction is carried as a local
        # urgency correction and Gaussian-decayed as h moves away from 0.1.
        short_proxy_p = baseline_p + short_delta_p
        short_proxy_b = baseline_b + short_delta_b
        short_proxy_c = sc[0]

        ws,wl = gaussian_mix_weights(h)

        final_p = ws*short_proxy_p + wl*lp[hi]
        final_b = ws*short_proxy_b + wl*lb[hi]
        final_c = ws*short_proxy_c + wl*lc[hi]

        gt_p = long_sample['target_p_raw'][hi]
        gt_b = long_sample['target_b_raw'][hi]
        gt_c = long_sample['target_contact_raw'][hi]

        for li,LN in enumerate(leg_names):
            for fi,FN in enumerate(feat):
                truth = float(gt_p[li,fi])
                bp = float(baseline_p[li,fi])
                spp = float(short_proxy_p[li,fi])
                lpp = float(lp[hi,li,fi])
                pred = float(final_p[li,fi])

                rows.append(dict(
                    horizon_s=float(h),
                    signal=f'{LN}.{FN}',
                    baseline_pred=bp,
                    short_native_pred=(
                        float(sp[0,li,fi])
                        if abs(float(h)-0.1)<1e-8
                        else np.nan
                    ),
                    short_proxy_pred=spp,
                    long_direct_pred=lpp,
                    lambda_short=ws,
                    lambda_long=wl,
                    prediction=pred,
                    ground_truth=truth,
                    baseline_abs_error=abs(bp-truth),
                    short_proxy_abs_error=abs(spp-truth),
                    long_abs_error=abs(lpp-truth),
                    signed_error=pred-truth,
                    abs_error=abs(pred-truth),
                ))

            truth = float(gt_c[li])
            bp = float(baseline_c[li])
            spp = float(short_proxy_c[li])
            lpp = float(lc[hi,li])
            pred = float(final_c[li])

            rows.append(dict(
                horizon_s=float(h),
                signal=f'{LN}.contact_probability',
                baseline_pred=bp,
                short_native_pred=(
                    float(sc[0,li])
                    if abs(float(h)-0.1)<1e-8
                    else np.nan
                ),
                short_proxy_pred=spp,
                long_direct_pred=lpp,
                lambda_short=ws,
                lambda_long=wl,
                prediction=pred,
                ground_truth=truth,
                baseline_abs_error=abs(bp-truth),
                short_proxy_abs_error=abs(spp-truth),
                long_abs_error=abs(lpp-truth),
                signed_error=pred-truth,
                abs_error=abs(pred-truth),
            ))

        for bi,BN in enumerate(body):
            truth = float(gt_b[bi])
            bp = float(baseline_b[bi])
            spp = float(short_proxy_b[bi])
            lpp = float(lb[hi,bi])
            pred = float(final_b[bi])

            rows.append(dict(
                horizon_s=float(h),
                signal=BN,
                baseline_pred=bp,
                short_native_pred=(
                    float(sb[0,bi])
                    if abs(float(h)-0.1)<1e-8
                    else np.nan
                ),
                short_proxy_pred=spp,
                long_direct_pred=lpp,
                lambda_short=ws,
                lambda_long=wl,
                prediction=pred,
                ground_truth=truth,
                baseline_abs_error=abs(bp-truth),
                short_proxy_abs_error=abs(spp-truth),
                long_abs_error=abs(lpp-truth),
                signed_error=pred-truth,
                abs_error=abs(pred-truth),
            ))

    vi = int(visual['visual_index'][idx])
    return pd.DataFrame(rows),idx,vi


# ------------------------- IMAGE OUTPUT --------------------------------
def get_test_images(base,visual,vi):
    fid = str(visual['frame_ids'][vi])
    rgb = np.asarray(
        Image.open(
            base/'image_lcam_front'/f'{fid}_lcam_front.png'
        ).convert('RGB')
    )
    depth = load_metric_depth(
        base/'depth_lcam_front'/f'{fid}_lcam_front_depth.png'
    )

    H,W = rgb.shape[:2]
    x0=int(ROI_X_MIN*W); x1=int(ROI_X_MAX*W)
    y0=int(ROI_Y_MIN*H); y1=int(ROI_Y_MAX*H)

    roi = rgb[y0:y1,x0:x1].copy()
    dd = depth[y0:y1,x0:x1]

    valid = (
        np.isfinite(dd)
        & (dd>=DEPTH_MIN_METERS)
        & (dd<=DEPTH_MAX_METERS)
    )
    roi[~valid] = 0
    return fid,rgb,roi


def signal_group(signal):
    if 'contact_probability' in signal:
        return 'contact'
    if signal.startswith(('FL.q_','FR.q_','RL.q_','RR.q_')):
        return 'q'
    if signal.startswith(('FL.dq_','FR.dq_','RL.dq_','RR.dq_')):
        return 'dq'
    if '.tau_' in signal:
        return 'tau'
    if signal.endswith(('.Fx','.Fy','.Fz')):
        return 'force'
    return 'body'


# --------------------------- MAIN --------------------------------------
seed_everything(SEED)
print('DEVICE:',DEVICE)
print('Output:',OUT_ROOT)

terrain_map = pd.DataFrame(EXPERIMENTS)
display(
    terrain_map[['terrain_type','environment','run','ttests']]
)

# 1) Download/reuse and preprocess every run ONCE.
run_records = []

for exp in EXPERIMENTS:
    print('\n'+'='*100)
    print(
        f'PREPARE {exp["terrain_type"].upper()} :: '
        f'{exp["environment"]}/{exp["run"]}'
    )
    print('='*100)

    base = download_run(exp['environment'],exp['run'])
    data = extract_run_physics(base,exp['run'])
    visual = prepare_visual_features(
        base,exp['environment'],exp['run'],data
    )

    # Verify exact fixed ttests fit in the run.
    for tt in exp['ttests']:
        if tt + max(LONG_HORIZONS) > float(data['time'][-1]):
            raise RuntimeError(
                f'ttest {tt} too close to end for '
                f'{exp["environment"]}/{exp["run"]}'
            )

    run_records.append(dict(
        **exp,
        base=base,
        data=data,
        visual=visual,
    ))

# 2) Train two priors ONCE.
# The prior sees only data at/before the FIRST ttest of each run.
prior_short = train_global_prior(
    run_records=run_records,
    context_sec=SHORT_CONTEXT_SEC,
    hop_sec=SHORT_HOP_SEC,
    horizons=SHORT_HORIZONS,
    epochs=PRIOR_EPOCHS_SHORT,
    seed=SEED+10,
    label='SHORT 0.5s -> direct residual 0.1s',
)

prior_long = train_global_prior(
    run_records=run_records,
    context_sec=LONG_CONTEXT_SEC,
    hop_sec=LONG_HOP_SEC,
    horizons=LONG_HORIZONS,
    epochs=PRIOR_EPOCHS_LONG,
    seed=SEED+20,
    label='LONG 4.0s -> direct residual 0.1..1.0s',
)

torch.save(
    prior_short.state_dict(),
    OUT_ROOT/'PRIOR_SHORT.pt'
)
torch.save(
    prior_long.state_dict(),
    OUT_ROOT/'PRIOR_LONG.pt'
)

# 3) Exact 4 terrains x 4 ttests.
all_results = []
all_summary = []
image_records = []

for terrain_i,rec in enumerate(run_records):
    data=rec['data']
    visual=rec['visual']
    base=rec['base']

    print('\n'+'#'*100)
    print(
        f'{rec["terrain_type"].upper()} :: '
        f'{rec["environment"]}/{rec["run"]}'
    )
    print('#'*100)

    for test_no,ttest in enumerate(rec['ttests'],1):
        print('\n'+'-'*100)
        print(
            f'TEST {test_no}/4 | '
            f'ttest={ttest:.3f}s'
        )
        print('-'*100)

        base_seed = (
            SEED + terrain_i*1000 + test_no*100
        )

        # Fresh local model object from prior weights.
        short_model,short_stats,Hs = adapt_fresh_local_model(
            prior_model=prior_short,
            data=data,
            visual=visual,
            ttest=ttest,
            context_sec=SHORT_CONTEXT_SEC,
            hop_sec=SHORT_HOP_SEC,
            horizons=SHORT_HORIZONS,
            half_life=SHORT_RECENCY_HALF_LIFE_SEC,
            epochs=LOCAL_EPOCHS_SHORT,
            seed=base_seed+1,
            label='SHORT',
        )

        long_model,long_stats,Hl = adapt_fresh_local_model(
            prior_model=prior_long,
            data=data,
            visual=visual,
            ttest=ttest,
            context_sec=LONG_CONTEXT_SEC,
            hop_sec=LONG_HOP_SEC,
            horizons=LONG_HORIZONS,
            half_life=LONG_RECENCY_HALF_LIFE_SEC,
            epochs=LOCAL_EPOCHS_LONG,
            seed=base_seed+2,
            label='LONG',
        )

        result,idx,vi = evaluate_case(
            short_model,short_stats,Hs,
            long_model,long_stats,Hl,
            data,visual,ttest
        )

        actual_ttest = float(data['time'][idx])

        result.insert(0,'test_id',test_no)
        result.insert(0,'run',rec['run'])
        result.insert(0,'environment',rec['environment'])
        result.insert(0,'terrain_type',rec['terrain_type'])
        result.insert(4,'ttest_s',actual_ttest)

        all_results.append(result)

        perfile = (
            OUT_ROOT
            / f'{rec["terrain_type"]}_{rec["environment"]}_'
              f'{rec["run"]}_test{test_no}_t{actual_ttest:.2f}.csv'
        )
        result.to_csv(perfile,index=False)

        # Unit-aware grouped summary, not one meaningless mixed-unit RMSE.
        temp = result.copy()
        temp['signal_group'] = temp['signal'].map(signal_group)

        grouped = (
            temp.groupby(['horizon_s','signal_group'],as_index=False)
            .agg(
                baseline_MAE=('baseline_abs_error','mean'),
                long_MAE=('long_abs_error','mean'),
                final_MAE=('abs_error','mean'),
            )
        )
        grouped.insert(0,'test_id',test_no)
        grouped.insert(0,'run',rec['run'])
        grouped.insert(0,'environment',rec['environment'])
        grouped.insert(0,'terrain_type',rec['terrain_type'])
        grouped.insert(4,'ttest_s',actual_ttest)
        all_summary.append(grouped)

        fid,rgb,roi = get_test_images(
            base,visual,vi
        )

        rgb_path = (
            OUT_ROOT
            / f'{rec["terrain_type"]}_{rec["environment"]}_'
              f'{rec["run"]}_test{test_no}_t{actual_ttest:.2f}_RGB.png'
        )
        roi_path = (
            OUT_ROOT
            / f'{rec["terrain_type"]}_{rec["environment"]}_'
              f'{rec["run"]}_test{test_no}_t{actual_ttest:.2f}_ROI.png'
        )

        Image.fromarray(rgb).save(rgb_path)
        Image.fromarray(roi).save(roi_path)

        image_records.append(dict(
            terrain_type=rec['terrain_type'],
            environment=rec['environment'],
            run=rec['run'],
            test_id=test_no,
            ttest=actual_ttest,
            frame_id=fid,
            rgb=rgb,
            roi=roi,
            rgb_path=str(rgb_path),
            roi_path=str(roi_path),
        ))

        print('\nRESULT TABLE')
        display(result)
        print('Saved:',perfile)

        del short_model,long_model,result,temp,grouped
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# 4) Save global outputs.
results_v4 = pd.concat(
    all_results,
    ignore_index=True
)
metrics_v4 = pd.concat(
    all_summary,
    ignore_index=True
)

results_v4.to_csv(
    OUT_ROOT/'ALL_16_TESTS_ALL_SENSOR_VALUES.csv',
    index=False
)
metrics_v4.to_csv(
    OUT_ROOT/'ALL_16_TESTS_GROUPED_METRICS.csv',
    index=False
)
terrain_map.to_csv(
    OUT_ROOT/'TERRAIN_RUN_MAP_AND_TTESTS.csv',
    index=False
)

# 5) Exact 16 test RGB images.
fig,axes = plt.subplots(4,4,figsize=(20,18))
axes = axes.reshape(-1)

for ax,rec in zip(axes,image_records):
    ax.imshow(rec['rgb'])
    ax.axis('off')
    ax.set_title(
        f"{rec['terrain_type']}\n"
        f"{rec['environment']}/{rec['run']} | "
        f"test {rec['test_id']} | t={rec['ttest']:.2f}s",
        fontsize=9
    )

plt.tight_layout()
fig.savefig(
    OUT_ROOT/'ALL_16_TTEST_RGB_IMAGES.png',
    dpi=180,
    bbox_inches='tight'
)
plt.show()

# 6) Exact model ROI images.
fig,axes = plt.subplots(4,4,figsize=(20,18))
axes = axes.reshape(-1)

for ax,rec in zip(axes,image_records):
    ax.imshow(rec['roi'])
    ax.axis('off')
    ax.set_title(
        f"MODEL ROI: {rec['terrain_type']}\n"
        f"test {rec['test_id']} | t={rec['ttest']:.2f}s",
        fontsize=9
    )

plt.tight_layout()
fig.savefig(
    OUT_ROOT/'ALL_16_TTEST_MODEL_ROIS.png',
    dpi=180,
    bbox_inches='tight'
)
plt.show()

# 7) Compact comparison table.
print('\n'+'='*100)
print('FINAL GROUPED METRICS')
print('='*100)

display(metrics_v4)

overall_by_group = (
    metrics_v4.groupby(
        ['signal_group','horizon_s'],
        as_index=False
    )
    .agg(
        baseline_MAE=('baseline_MAE','mean'),
        long_MAE=('long_MAE','mean'),
        final_MAE=('final_MAE','mean'),
    )
)

overall_by_group.to_csv(
    OUT_ROOT/'MEAN_METRICS_ACROSS_16_TESTS.csv',
    index=False
)

print('\nMEAN ACROSS ALL 16 TESTS')
display(overall_by_group)

print('\nFILES')
print(
    'All sensor values :',
    OUT_ROOT/'ALL_16_TESTS_ALL_SENSOR_VALUES.csv'
)
print(
    'Grouped metrics   :',
    OUT_ROOT/'ALL_16_TESTS_GROUPED_METRICS.csv'
)
print(
    'Mean metrics      :',
    OUT_ROOT/'MEAN_METRICS_ACROSS_16_TESTS.csv'
)
print(
    '16 RGB images     :',
    OUT_ROOT/'ALL_16_TTEST_RGB_IMAGES.png'
)
print(
    '16 model ROIs     :',
    OUT_ROOT/'ALL_16_TTEST_MODEL_ROIS.png'
)
print(
    'Short prior       :',
    OUT_ROOT/'PRIOR_SHORT.pt'
)
print(
    'Long prior        :',
    OUT_ROOT/'PRIOR_LONG.pt'
)

print('\nDONE.')

# %% notebook cell 2
# ============================================================
# DOWNLOAD ALL V4 EXPERIMENT RESULTS
# ============================================================

from pathlib import Path
import shutil
import os
from google.colab import files

RESULT_DIR = Path("/content/v4_residual_prior_direct_results")
ZIP_NAME = "/content/V4_RESIDUAL_PRIOR_DIRECT_ALL_RESULTS"

expected_files = [
    "ALL_16_TESTS_ALL_SENSOR_VALUES.csv",
    "ALL_16_TESTS_GROUPED_METRICS.csv",
    "MEAN_METRICS_ACROSS_16_TESTS.csv",
    "ALL_16_TTEST_RGB_IMAGES.png",
    "ALL_16_TTEST_MODEL_ROIS.png",
    "PRIOR_SHORT.pt",
    "PRIOR_LONG.pt",
]

print("=" * 90)
print("CHECKING MAIN OUTPUT FILES")
print("=" * 90)

missing = []

for name in expected_files:
    p = RESULT_DIR / name

    if p.exists():
        size_mb = p.stat().st_size / (1024**2)
        print(f"✓ {name:<45} {size_mb:8.2f} MB")
    else:
        print(f"✗ MISSING: {name}")
        missing.append(name)

print("\n" + "=" * 90)
print("ALL FILES ACTUALLY PRESENT IN RESULT DIRECTORY")
print("=" * 90)

all_files = sorted([p for p in RESULT_DIR.rglob("*") if p.is_file()])

for i, p in enumerate(all_files, 1):
    size_mb = p.stat().st_size / (1024**2)
    print(f"{i:03d}. {p.relative_to(RESULT_DIR)}   [{size_mb:.2f} MB]")

print(f"\nTotal files found: {len(all_files)}")

if missing:
    print("\nWARNING: Some expected main files are missing:")
    for x in missing:
        print(" -", x)

# ------------------------------------------------------------
# CREATE ONE ZIP CONTAINING EVERYTHING
# ------------------------------------------------------------

zip_path = shutil.make_archive(
    ZIP_NAME,
    "zip",
    root_dir=RESULT_DIR
)

zip_size_mb = Path(zip_path).stat().st_size / (1024**2)

print("\n" + "=" * 90)
print("ZIP READY")
print("=" * 90)
print("Path :", zip_path)
print(f"Size : {zip_size_mb:.2f} MB")

# ------------------------------------------------------------
# DOWNLOAD TO YOUR COMPUTER
# ------------------------------------------------------------

files.download(zip_path)

# %% notebook cell 3
# ============================================================================
# V5 PATCH — PHYSICS-INFORMED + 4-REGIME CONTACT CONDITIONING
# Paste this SINGLE CELL at the END of the completed V4 notebook and run it.
# It REUSES run_records / visual features already in RAM. No re-download.
#
# CHANGE 1: physics fusion
#   (a) parameter-free kinematic consistency:
#         q(t+h) ≈ q(t) + 0.5 h [dq(t)+dq(t+h)]
#   (b) causal identified inverse-dynamics proxy of
#         tau ≈ M(q)ddq + C(q,dq)dq + g(q) - J(q)^T F
#       fitted only from data at/before the prediction time.
#   (c) causal calibrated Newton proxy of
#         sum_i F_i ≈ m(a-g)
#       implemented as an affine force->accelerometer map so we do NOT fake
#       unknown robot mass / frame conventions.
#   Physics predictions are fed into the Transformer AND used as losses.
#
# CHANGE 2: 4 contact-force regimes instead of binary contact
#       class 0 = no contact (<10 N)
#       class 1/2/3 = light / medium / heavy contact, split by causal global
#                     force-norm quantiles learned from pre-test data.
#   Contact probabilities condition a dedicated tau/force head.
#
# IMPORTANT: this is scientifically safer than inserting a fake ANYmal mass,
# inertia matrix or Jacobian that the downloaded run does not provide here.
# ============================================================================

import copy, math, gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import display

# ---------- sanity: V4 must have finished in this same runtime ----------
_required = [
    'run_records','CausalHistoryEncoder','TerrainEncoder','make_direct_sample',
    'make_range_windows','make_local_windows','make_stats_between',
    'make_local_stats','normalize_arrays','nearest_index','horizon_offsets',
    'seed_everything','SHORT_CONTEXT_SEC','LONG_CONTEXT_SEC','SHORT_HOP_SEC',
    'LONG_HOP_SEC','SHORT_HORIZONS','LONG_HORIZONS','TRAIN_LOOKBACK_SEC',
    'SHORT_RECENCY_HALF_LIFE_SEC','LONG_RECENCY_HALF_LIFE_SEC','DEVICE','SEED',
    'WEIGHT_DECAY','GRAD_CLIP','LR_PRIOR','LR_LOCAL','EPS','CONTACT_THRESHOLD'
]
missing = [x for x in _required if x not in globals()]
if missing:
    raise RuntimeError(
        'V5 patch must be run after the completed V4 cell in the SAME runtime. '
        f'Missing: {missing}'
    )

V5_OUT = Path('/content/v5_physics_contact_results')
V5_OUT.mkdir(parents=True, exist_ok=True)

# Training epochs: keep comparable to V4.
V5_PRIOR_EPOCHS_SHORT = 2
V5_PRIOR_EPOCHS_LONG = 3
V5_LOCAL_EPOCHS_SHORT = 4
V5_LOCAL_EPOCHS_LONG = 4

# Loss weights. Data loss is still dominant.
W_TAU_DATA = 1.50
W_FORCE_DATA = 2.00
W_BODY_DATA = 0.50
W_REGIME = 0.50
W_KIN = 0.25
W_INVDYN = 0.25
W_NEWTON = 0.15
W_FORCE_MAG = 0.20
W_ZERO_FORCE = 0.10
RIDGE = 1e-3

# Explicit physics features per horizon:
# q_cv(12), tau_phys_baseline(12), body_acc_phys(3), force_norm(4), contact_1hot(16)
PI_PHYS_DIM = 47
N_REGIMES = 4

# ---------------------------------------------------------------------------
# 1) CONTACT REGIMES — thresholds learned ONLY from causal pre-first-test data
# ---------------------------------------------------------------------------
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


REGIME_THRESHOLDS = learn_global_regime_thresholds(run_records)
REGIME_CENTERS, REGIME_CLASS_WEIGHTS, REGIME_COUNTS = learn_regime_centers_and_weights(
    run_records, REGIME_THRESHOLDS
)

print('\nV5 CONTACT FORCE REGIMES')
print('  class 0 no-contact : |F| < %.3f N' % REGIME_THRESHOLDS[0])
print('  class 1 light      : %.3f <= |F| < %.3f N' % (REGIME_THRESHOLDS[0], REGIME_THRESHOLDS[1]))
print('  class 2 medium     : %.3f <= |F| < %.3f N' % (REGIME_THRESHOLDS[1], REGIME_THRESHOLDS[2]))
print('  class 3 heavy      : |F| >= %.3f N' % REGIME_THRESHOLDS[2])
print('  class median |F|   :', REGIME_CENTERS.tolist())
print('  class weights      :', REGIME_CLASS_WEIGHTS.tolist())


# ---------------------------------------------------------------------------
# 2) CAUSAL PHYSICS IDENTIFICATION
# ---------------------------------------------------------------------------
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


def fit_causal_physics(data, start_time, end_time, thresholds=REGIME_THRESHOLDS):
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


# Pre-fit one causal physics prior per run, ending at its first ttest.
for rec in run_records:
    rec['physics_prior_v5'] = fit_causal_physics(
        rec['data'], float(rec['data']['time'][0]), float(min(rec['ttests']))
    )


# ---------------------------------------------------------------------------
# 3) MODEL: explicit physics branch + contact-regime conditioned tau/force head
# ---------------------------------------------------------------------------
class PhysicsContactResidualDynamics(nn.Module):
    def __init__(self, history_steps, n_horizons, dino_dim=384, conv_dim=768):
        super().__init__()
        self.history_steps = int(history_steps)
        self.n_horizons = int(n_horizons)

        self.leg_encoder = CausalHistoryEncoder(24, history_steps, 128, 2, 4)
        self.leg_identity = nn.Embedding(4,128)
        self.body_encoder = CausalHistoryEncoder(18, history_steps, 128, 2, 4)
        self.terrain_encoder = TerrainEncoder(dino_dim, conv_dim)

        self.context_net = nn.Sequential(
            nn.LayerNorm(896), nn.Linear(896,512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512,256), nn.GELU(), nn.LayerNorm(256)
        )
        self.control_proj = nn.Sequential(
            nn.Linear(36,128), nn.GELU(), nn.LayerNorm(128)
        )
        self.horizon_proj = nn.Sequential(
            nn.Linear(1,64), nn.GELU(), nn.Linear(64,128), nn.GELU(), nn.LayerNorm(128)
        )
        self.physics_proj = nn.Sequential(
            nn.Linear(PI_PHYS_DIM,128), nn.GELU(), nn.Linear(128,128), nn.GELU(), nn.LayerNorm(128)
        )

        # z = context256 + control128 + horizon128 + physics128 = 640
        self.regime_head = nn.Sequential(
            nn.Linear(640,256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256,4*N_REGIMES)
        )
        self.regime_embed = nn.Sequential(
            nn.Linear(4*N_REGIMES,64), nn.GELU(), nn.LayerNorm(64)
        )

        # q/dq + body are separated from contact-heavy tau/force channels.
        self.state_body_head = nn.Sequential(
            nn.Linear(704,256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256,36)  # q/dq 24 + body 12
        )
        self.tau_force_head = nn.Sequential(
            nn.Linear(704,256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256,24)  # 4 legs * (tau3 + F3)
        )

    def encode_context(self,p_hist,b_hist,u_hist,l_hist,e_hist,dino,conv,depth):
        Bn,T = p_hist.shape[0], p_hist.shape[2]
        ulegs = u_hist.unsqueeze(1).expand(-1,4,-1,-1)
        leg_in = torch.cat([p_hist,ulegs,l_hist,e_hist],dim=-1).reshape(Bn*4,T,24)
        lf = self.leg_encoder(leg_in)
        ids = torch.arange(4,device=p_hist.device).repeat(Bn)
        lf = (lf + self.leg_identity(ids)).reshape(Bn,4,128).reshape(Bn,512)
        bf = self.body_encoder(torch.cat([b_hist,u_hist],dim=-1))
        terrain = self.terrain_encoder(dino,conv,depth)
        return self.context_net(torch.cat([lf,bf,terrain],dim=-1))

    def forward(self,p_hist,b_hist,u_hist,l_hist,e_hist,dino,conv,depth,
                future_control_features,horizon_values,physics_features):
        ctx = self.encode_context(p_hist,b_hist,u_hist,l_hist,e_hist,dino,conv,depth)
        Hn = future_control_features.shape[1]
        if Hn != self.n_horizons:
            raise RuntimeError(f'Expected {self.n_horizons} horizons, got {Hn}')

        c = self.control_proj(future_control_features)
        h = self.horizon_proj(horizon_values)
        ph = self.physics_proj(physics_features)
        ctx = ctx.unsqueeze(1).expand(-1,Hn,-1)
        z = torch.cat([ctx,c,h,ph],dim=-1)

        regime_logits = self.regime_head(z).reshape(-1,Hn,4,N_REGIMES)
        regime_probs = torch.softmax(regime_logits,dim=-1)
        re = self.regime_embed(regime_probs.reshape(-1,Hn,4*N_REGIMES))
        zr = torch.cat([z,re],dim=-1)

        sb = self.state_body_head(zr)
        tf = self.tau_force_head(zr)

        residual_p = torch.zeros(
            (p_hist.shape[0],Hn,4,12),dtype=sb.dtype,device=sb.device
        )
        residual_p[:,:,:,0:6] = sb[...,:24].reshape(-1,Hn,4,6)
        residual_p[:,:,:,6:12] = tf.reshape(-1,Hn,4,6)
        residual_b = sb[...,24:36]

        return {
            'residual_p': residual_p,
            'residual_b': residual_b,
            'regime_logits': regime_logits,
            'regime_probs': regime_probs,
        }


# ---------------------------------------------------------------------------
# 4) SAMPLE BUILDING WITH PHYSICS FEATURES
# ---------------------------------------------------------------------------
def make_pi_sample(data,visual,stats,physics,idx,history_steps,horizons,norm=None):
    base = make_direct_sample(data,visual,stats,idx,history_steps,horizons,norm=norm)

    current_p = data['P'][idx].astype(np.float32)
    current_b = data['B'][idx].astype(np.float32)
    horizons_arr = np.asarray(horizons,np.float32)
    offs = horizon_offsets(data,horizons)

    phys_features = []
    target_regime = []

    current_force = current_p[:,9:12]
    current_cls = force_regime_np(current_force, REGIME_THRESHOLDS)
    current_oh = np.eye(N_REGIMES,dtype=np.float32)[current_cls]
    force_scale = max(float(REGIME_THRESHOLDS[-1]), 1.0)

    for h,off,bp in zip(horizons_arr,offs,base['baseline_p_raw']):
        # Physics baseline uses constant dq => ddq=0 at input stage.
        q_cv = bp[:,0:3]
        dq0 = current_p[:,3:6]
        ddq0 = np.zeros((4,3),np.float32)
        tau_phys = physics_tau_np(
            physics,q_cv,dq0,ddq0,current_force,current_oh
        )
        acc_phys = physics_acc_np(physics,current_force)

        # Normalize physical features before giving them to the net.
        qn = ((q_cv - stats['P_mean'][0:3]) / stats['P_std'][0:3]).reshape(-1)
        taun = ((tau_phys - stats['P_mean'][6:9]) / stats['P_std'][6:9]).reshape(-1)
        accn = (acc_phys - stats['B_mean'][0:3]) / stats['B_std'][0:3]
        fn = np.linalg.norm(current_force,axis=-1) / force_scale
        coh = current_oh.reshape(-1)
        feat = np.concatenate([qn,taun,accn,fn,coh]).astype(np.float32)
        if feat.size != PI_PHYS_DIM:
            raise RuntimeError(f'physics feature dim {feat.size} != {PI_PHYS_DIM}')
        phys_features.append(feat)

        target_idx = idx + off
        target_regime.append(
            force_regime_np(data['P'][target_idx,:,9:12],REGIME_THRESHOLDS)
        )

    base['physics_features'] = torch.as_tensor(
        np.asarray(phys_features,np.float32)[None],dtype=torch.float32,device=DEVICE
    )
    base['target_regime'] = torch.as_tensor(
        np.asarray(target_regime,np.int64)[None],dtype=torch.long,device=DEVICE
    )
    base['current_p_raw'] = current_p
    base['current_b_raw'] = current_b
    base['horizons_raw'] = horizons_arr
    return base


# ---------------------------------------------------------------------------
# 5) PHYSICS-INFORMED LOSS
# ---------------------------------------------------------------------------
REGIME_CE_WEIGHTS_T = torch.as_tensor(REGIME_CLASS_WEIGHTS,dtype=torch.float32,device=DEVICE)


def _torch_physics_tau(physics,q,dq,ddq,force,regime_probs):
    # q,dq,ddq,force: (B,H,4,3), regime_probs: (B,H,4,4)
    ones = torch.ones((*q.shape[:-1],1),dtype=q.dtype,device=q.device)
    X = torch.cat([ddq,dq,torch.sin(q),torch.cos(q),force,regime_probs,ones],dim=-1)
    W = torch.as_tensor(physics['invW'],dtype=q.dtype,device=q.device)  # 4,20,3
    return torch.einsum('bhlf,lfj->bhlj',X,W)


def _torch_physics_acc(physics,force):
    # force B,H,4,3 -> B,H,3
    sf = force.sum(dim=2)
    ones = torch.ones((*sf.shape[:-1],1),dtype=sf.dtype,device=sf.device)
    X = torch.cat([sf,ones],dim=-1)
    W = torch.as_tensor(physics['bodyW'],dtype=sf.dtype,device=sf.device)
    return X @ W


def pi_loss(out,sample,stats,physics):
    pp = out['residual_p']
    pb = out['residual_b']
    tp = sample['target_residual_p']
    tb = sample['target_residual_b']
    Hn = pp.shape[1]
    hw = torch.linspace(1.0,2.0,Hn,device=pp.device)

    def hmse(a,b):
        e = (a-b)**2
        dims = tuple(range(2,e.ndim))
        return (e.mean(dim=dims)*hw[None,:]).mean()

    lq = hmse(pp[:,:,:,0:3],tp[:,:,:,0:3])
    ldq = hmse(pp[:,:,:,3:6],tp[:,:,:,3:6])
    ltau = hmse(pp[:,:,:,6:9],tp[:,:,:,6:9])
    # Force is evaluated after contact-regime gating below; do not use the raw residual
    # force loss here because a swing foot should be physically driven toward zero force.
    lbody = hmse(pb,tb)

    # Multi-class contact regime classification.
    logits = out['regime_logits'].reshape(-1,N_REGIMES)
    targets = sample['target_regime'].reshape(-1)
    lreg = F.cross_entropy(logits,targets,weight=REGIME_CE_WEIGHTS_T)

    # Convert model outputs to RAW physical units for physics penalties.
    Pstd = torch.as_tensor(stats['P_std'],dtype=pp.dtype,device=pp.device)
    Bstd = torch.as_tensor(stats['B_std'],dtype=pb.dtype,device=pb.device)
    bp = torch.as_tensor(sample['baseline_p_raw'],dtype=pp.dtype,device=pp.device).unsqueeze(0)
    bb = torch.as_tensor(sample['baseline_b_raw'],dtype=pb.dtype,device=pb.device).unsqueeze(0)
    predP = bp + pp * Pstd.view(1,1,1,12)
    predB = bb + pb * Bstd.view(1,1,12)

    q1 = predP[:,:,:,0:3]
    dq1 = predP[:,:,:,3:6]
    tau1 = predP[:,:,:,6:9]
    force1_raw = predP[:,:,:,9:12]
    p_contact = 1.0 - out['regime_probs'][...,0]
    force1 = force1_raw * p_contact.unsqueeze(-1)
    gt_force = torch.as_tensor(
        sample['target_p_raw'][:,:,:,9:12] if sample['target_p_raw'].ndim == 4 else sample['target_p_raw'][:,:,9:12],
        dtype=pp.dtype,device=pp.device
    )
    if gt_force.ndim == 3:
        gt_force = gt_force.unsqueeze(0)
    fscale = Pstd[9:12].view(1,1,1,3)
    ferr = ((force1-gt_force)/(fscale+1e-6))**2
    lforce = (ferr.mean(dim=(2,3))*hw[None,:]).mean()

    cur = torch.as_tensor(sample['current_p_raw'],dtype=pp.dtype,device=pp.device)
    q0 = cur[:,0:3].view(1,1,4,3)
    dq0 = cur[:,3:6].view(1,1,4,3)
    hs = torch.as_tensor(sample['horizons_raw'],dtype=pp.dtype,device=pp.device).view(1,Hn,1,1)

    # q(t+h) = q(t) + 1/2 h [dq(t)+dq(t+h)]  (trapezoid kinematics)
    q_phys = q0 + 0.5*hs*(dq0+dq1)
    qscale = Pstd[0:3].view(1,1,1,3)
    lkin = (((q1-q_phys)/(qscale+1e-6))**2).mean()

    # ddq definition + identified inverse dynamics proxy.
    ddq = (dq1-dq0)/(hs+1e-6)
    tau_phys = _torch_physics_tau(
        physics,q1,dq1,ddq,force1,out['regime_probs']
    )
    tauscale = Pstd[6:9].view(1,1,1,3)
    linvdyn = (((tau1-tau_phys)/(tauscale+1e-6))**2).mean()

    # Newton force -> body acceleration consistency.
    acc_phys = _torch_physics_acc(physics,force1)
    acc1 = predB[:,:,0:3]
    accscale = Bstd[0:3].view(1,1,3)
    lnewton = (((acc1-acc_phys)/(accscale+1e-6))**2).mean()

    # Contact regime should explain force magnitude.
    centers = torch.as_tensor(
        REGIME_CENTERS,dtype=pp.dtype,device=pp.device
    )
    expected_logmag = (
        out['regime_probs'] * torch.log1p(centers).view(1,1,1,N_REGIMES)
    ).sum(-1)
    pred_logmag = torch.log1p(torch.linalg.norm(force1,dim=-1).clamp_min(0.0))
    lmag = ((pred_logmag-expected_logmag)**2).mean()

    # If class-0 probability is high, large force should be discouraged.
    p0 = out['regime_probs'][...,0]
    force_scale = float(max(REGIME_CENTERS[-1],REGIME_THRESHOLDS[-1],1.0))
    lzero = (p0 * torch.linalg.norm(force1,dim=-1)/force_scale).mean()

    data_loss = (
        lq + ldq + W_TAU_DATA*ltau + W_FORCE_DATA*lforce + W_BODY_DATA*lbody
    )
    total = (
        data_loss + W_REGIME*lreg + W_KIN*lkin + W_INVDYN*linvdyn
        + W_NEWTON*lnewton + W_FORCE_MAG*lmag + W_ZERO_FORCE*lzero
    )

    parts = {
        'data':float(data_loss.detach().cpu()),
        'reg':float(lreg.detach().cpu()),
        'kin':float(lkin.detach().cpu()),
        'dyn':float(linvdyn.detach().cpu()),
        'newton':float(lnewton.detach().cpu()),
    }
    return total,parts


# ---------------------------------------------------------------------------
# 6) STRICT CHRONOLOGICAL TRAINING — one optimizer.step PER WINDOW
# ---------------------------------------------------------------------------
def pi_model_step(model,optimizer,sample,stats,physics,weight=1.0):
    optimizer.zero_grad(set_to_none=True)
    out = model(
        sample['p_hist'],sample['b_hist'],sample['u_hist'],sample['l_hist'],sample['e_hist'],
        sample['dino'],sample['conv'],sample['depth'],sample['future_control_features'],
        sample['horizon_values'],sample['physics_features']
    )
    raw,parts = pi_loss(out,sample,stats,physics)
    loss = raw*float(weight)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP)
    optimizer.step()
    value=float(raw.detach().cpu())
    del out,raw,loss
    return value,parts


def train_pi_prior(run_records,context_sec,hop_sec,horizons,epochs,seed,label):
    seed_everything(seed)
    first=run_records[0]
    dt=float(np.median(np.diff(first['data']['time'])))
    H=int(round(context_sec/dt))
    model=PhysicsContactResidualDynamics(
        H,len(horizons),
        first['visual']['DINO'].shape[1],
        first['visual']['CONV'].reshape(len(first['visual']['CONV']),-1).shape[1]
    ).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=LR_PRIOR,weight_decay=WEIGHT_DECAY)

    print('\n'+'='*100)
    print('V5 GLOBAL PRIOR ::',label)
    print('physics input + physics losses + 4 force/contact regimes')
    print('NO BATCH / NO SHUFFLE / one optimizer.step per window')
    print('='*100)

    for ep in range(epochs):
        vals=[]
        for rec in run_records:
            data,visual=rec['data'],rec['visual']
            first_tt=float(min(rec['ttests']))
            stats=make_stats_between(data,float(data['time'][0]),first_tt)
            norm=normalize_arrays(data,stats)
            physics=rec['physics_prior_v5']
            windows,H2=make_range_windows(
                data,visual,float(data['time'][0]),first_tt,
                context_sec,hop_sec,max(horizons)
            )
            if H2!=H: raise RuntimeError('V5 prior history mismatch')
            print(f'  prior ep {ep+1}/{epochs} | {rec["terrain_type"]} | windows={len(windows)}')
            for wi,(idx,_,te) in enumerate(windows,1):
                s=make_pi_sample(data,visual,stats,physics,idx,H,horizons,norm=norm)
                v,parts=pi_model_step(model,opt,s,stats,physics,1.0)
                vals.append(v)
                if wi in (1,len(windows)):
                    print(f'    {wi:03d}/{len(windows):03d} target_end={te:.3f} loss={v:.4f} '
                          f'data={parts["data"]:.3f} dyn={parts["dyn"]:.3f} reg={parts["reg"]:.3f}')
                del s
        print(f'  V5 PRIOR {label} epoch {ep+1} mean loss={np.mean(vals):.6f}')
    return model


def adapt_pi_local(prior_model,data,visual,ttest,context_sec,hop_sec,horizons,
                   half_life,epochs,seed,label):
    seed_everything(seed)
    model=copy.deepcopy(prior_model).to(DEVICE)
    start=max(float(data['time'][0]),float(ttest)-TRAIN_LOOKBACK_SEC)
    stats=make_local_stats(data,ttest)
    physics=fit_causal_physics(data,start,float(ttest),REGIME_THRESHOLDS)
    windows,H=make_local_windows(
        data,visual,ttest,context_sec,hop_sec,max(horizons),half_life
    )
    if H!=model.history_steps: raise RuntimeError('V5 local history mismatch')
    norm=normalize_arrays(data,stats)
    opt=torch.optim.AdamW(model.parameters(),lr=LR_LOCAL,weight_decay=WEIGHT_DECAY)

    print(f'\n  V5 LOCAL {label}: ttest={ttest:.3f}s windows={len(windows)} '
          f'oldest_w={windows[0][1]:.5f} newest_w={windows[-1][1]:.5f}')
    for ep in range(epochs):
        vals=[]
        for wi,(idx,w,_,te) in enumerate(windows,1):
            s=make_pi_sample(data,visual,stats,physics,idx,H,horizons,norm=norm)
            v,parts=pi_model_step(model,opt,s,stats,physics,w)
            vals.append(v)
            if wi<=1 or wi==len(windows):
                print(f'    {label} ep {ep+1:02d} {wi:03d}/{len(windows):03d} '
                      f'target_end={te:.3f} w={w:.4f} loss={v:.4f} '
                      f'dyn={parts["dyn"]:.3f} reg={parts["reg"]:.3f}')
            del s
        print(f'    {label} epoch {ep+1}/{epochs} mean loss={np.mean(vals):.6f}')
    return model,stats,physics,H


# ---------------------------------------------------------------------------
# 7) RAW PREDICTION + 4-REGIME CONTACT PROBABILITIES
# ---------------------------------------------------------------------------
def predict_pi_raw(model,sample,stats):
    model.eval()
    with torch.no_grad():
        out=model(
            sample['p_hist'],sample['b_hist'],sample['u_hist'],sample['l_hist'],sample['e_hist'],
            sample['dino'],sample['conv'],sample['depth'],sample['future_control_features'],
            sample['horizon_values'],sample['physics_features']
        )
    rp=out['residual_p'][0].cpu().numpy()
    rb=out['residual_b'][0].cpu().numpy()
    regime=out['regime_probs'][0].cpu().numpy()  # H,4,4
    predP=sample['baseline_p_raw'] + rp*stats['P_std'][None,None,:]
    predB=sample['baseline_b_raw'] + rb*stats['B_std'][None,:]
    # Hard physical coupling at inference: predicted force is gated by predicted
    # probability of being in ANY contact regime (classes 1/2/3).
    p_contact = 1.0-regime[...,0]
    predP[:,:,9:12] *= p_contact[...,None]
    return predP,predB,regime


def contact_prob_from_regime(regime_probs):
    return 1.0-regime_probs[...,0]


# Keep the same V4 short/long Gaussian mixture for direct comparability.
def evaluate_pi_case(short_model,short_stats,short_physics,Hs,
                     long_model,long_stats,long_physics,Hl,
                     data,visual,ttest):
    idx=nearest_index(data['time'],ttest)
    snorm=normalize_arrays(data,short_stats)
    lnorm=normalize_arrays(data,long_stats)
    ss=make_pi_sample(data,visual,short_stats,short_physics,idx,Hs,SHORT_HORIZONS,norm=snorm)
    ls=make_pi_sample(data,visual,long_stats,long_physics,idx,Hl,LONG_HORIZONS,norm=lnorm)
    sp,sb,sreg=predict_pi_raw(short_model,ss,short_stats)
    lp,lb,lreg=predict_pi_raw(long_model,ls,long_stats)
    sc=contact_prob_from_regime(sreg)
    lc=contact_prob_from_regime(lreg)

    short_delta_p=sp[0]-ss['baseline_p_raw'][0]
    short_delta_b=sb[0]-ss['baseline_b_raw'][0]
    rows=[]
    leg_names=['FL','FR','RL','RR']
    feat=['q_HAA','q_HFE','q_KFE','dq_HAA','dq_HFE','dq_KFE',
          'tau_HAA','tau_HFE','tau_KFE','Fx','Fy','Fz']
    body=['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z',
          'body_vx','body_vy','body_vz','body_wx','body_wy','body_wz']

    for hi,h in enumerate(LONG_HORIZONS):
        baseline_p=ls['baseline_p_raw'][hi]
        baseline_b=ls['baseline_b_raw'][hi]
        baseline_c=(force_regime_np(data['P'][idx,:,9:12],REGIME_THRESHOLDS)>0).astype(np.float32)
        short_proxy_p=baseline_p+short_delta_p
        short_proxy_b=baseline_b+short_delta_b
        short_proxy_c=sc[0]
        ws,wl=gaussian_mix_weights(h)
        final_p=ws*short_proxy_p+wl*lp[hi]
        final_b=ws*short_proxy_b+wl*lb[hi]
        final_c=ws*short_proxy_c+wl*lc[hi]
        final_reg=ws*sreg[0]+wl*lreg[hi]
        gt_p=ls['target_p_raw'][hi]
        gt_b=ls['target_b_raw'][hi]
        gt_reg=ls['target_regime'][0,hi].cpu().numpy()
        gt_c=(gt_reg>0).astype(np.float32)

        for li,LN in enumerate(leg_names):
            for fi,FN in enumerate(feat):
                truth=float(gt_p[li,fi]); bp=float(baseline_p[li,fi]); spp=float(short_proxy_p[li,fi]); lpp=float(lp[hi,li,fi]); pred=float(final_p[li,fi])
                rows.append(dict(
                    horizon_s=float(h),signal=f'{LN}.{FN}',baseline_pred=bp,
                    short_native_pred=float(sp[0,li,fi]) if abs(float(h)-0.1)<1e-8 else np.nan,
                    short_proxy_pred=spp,long_direct_pred=lpp,lambda_short=ws,lambda_long=wl,
                    prediction=pred,ground_truth=truth,baseline_abs_error=abs(bp-truth),
                    short_proxy_abs_error=abs(spp-truth),long_abs_error=abs(lpp-truth),
                    signed_error=pred-truth,abs_error=abs(pred-truth),
                    contact_regime_pred=np.nan,contact_regime_truth=np.nan
                ))

            truth=float(gt_c[li]); bp=float(baseline_c[li]); spp=float(short_proxy_c[li]); lpp=float(lc[hi,li]); pred=float(final_c[li])
            rows.append(dict(
                horizon_s=float(h),signal=f'{LN}.contact_probability',baseline_pred=bp,
                short_native_pred=float(sc[0,li]) if abs(float(h)-0.1)<1e-8 else np.nan,
                short_proxy_pred=spp,long_direct_pred=lpp,lambda_short=ws,lambda_long=wl,
                prediction=pred,ground_truth=truth,baseline_abs_error=abs(bp-truth),
                short_proxy_abs_error=abs(spp-truth),long_abs_error=abs(lpp-truth),
                signed_error=pred-truth,abs_error=abs(pred-truth),
                contact_regime_pred=int(np.argmax(final_reg[li])),contact_regime_truth=int(gt_reg[li])
            ))

        for bi,BN in enumerate(body):
            truth=float(gt_b[bi]); bp=float(baseline_b[bi]); spp=float(short_proxy_b[bi]); lpp=float(lb[hi,bi]); pred=float(final_b[bi])
            rows.append(dict(
                horizon_s=float(h),signal=BN,baseline_pred=bp,
                short_native_pred=float(sb[0,bi]) if abs(float(h)-0.1)<1e-8 else np.nan,
                short_proxy_pred=spp,long_direct_pred=lpp,lambda_short=ws,lambda_long=wl,
                prediction=pred,ground_truth=truth,baseline_abs_error=abs(bp-truth),
                short_proxy_abs_error=abs(spp-truth),long_abs_error=abs(lpp-truth),
                signed_error=pred-truth,abs_error=abs(pred-truth),
                contact_regime_pred=np.nan,contact_regime_truth=np.nan
            ))

    vi=int(visual['visual_index'][idx])
    return pd.DataFrame(rows),idx,vi


# ---------------------------------------------------------------------------
# 8) RUN ALL 16 TESTS — cached data/visuals are reused
# ---------------------------------------------------------------------------
seed_everything(SEED+5000)

pi_prior_short=train_pi_prior(
    run_records,SHORT_CONTEXT_SEC,SHORT_HOP_SEC,SHORT_HORIZONS,
    V5_PRIOR_EPOCHS_SHORT,SEED+5010,'SHORT physics/contact'
)
pi_prior_long=train_pi_prior(
    run_records,LONG_CONTEXT_SEC,LONG_HOP_SEC,LONG_HORIZONS,
    V5_PRIOR_EPOCHS_LONG,SEED+5020,'LONG physics/contact'
)

torch.save(pi_prior_short.state_dict(),V5_OUT/'V5_PRIOR_SHORT.pt')
torch.save(pi_prior_long.state_dict(),V5_OUT/'V5_PRIOR_LONG.pt')

v5_results=[]
v5_grouped=[]
v5_regime_rows=[]

for terrain_i,rec in enumerate(run_records):
    print('\n'+'#'*100)
    print('V5',rec['terrain_type'].upper(),rec['environment'],rec['run'])
    print('#'*100)
    data,visual=rec['data'],rec['visual']

    for test_no,ttest in enumerate(rec['ttests'],1):
        print('\n'+'-'*100)
        print(f'V5 TEST {test_no}/4 | ttest={ttest:.3f}s')
        print('-'*100)
        base_seed=SEED+6000+terrain_i*1000+test_no*100

        sm,ss,sp,Hs=adapt_pi_local(
            pi_prior_short,data,visual,ttest,SHORT_CONTEXT_SEC,SHORT_HOP_SEC,
            SHORT_HORIZONS,SHORT_RECENCY_HALF_LIFE_SEC,V5_LOCAL_EPOCHS_SHORT,
            base_seed+1,'SHORT'
        )
        lm,ls,lp_,Hl=adapt_pi_local(
            pi_prior_long,data,visual,ttest,LONG_CONTEXT_SEC,LONG_HOP_SEC,
            LONG_HORIZONS,LONG_RECENCY_HALF_LIFE_SEC,V5_LOCAL_EPOCHS_LONG,
            base_seed+2,'LONG'
        )

        result,idx,vi=evaluate_pi_case(sm,ss,sp,Hs,lm,ls,lp_,Hl,data,visual,ttest)
        actual=float(data['time'][idx])
        result.insert(0,'test_id',test_no)
        result.insert(0,'run',rec['run'])
        result.insert(0,'environment',rec['environment'])
        result.insert(0,'terrain_type',rec['terrain_type'])
        result.insert(4,'ttest_s',actual)
        v5_results.append(result)

        out_csv=V5_OUT/f'{rec["terrain_type"]}_{rec["environment"]}_{rec["run"]}_test{test_no}_t{actual:.2f}.csv'
        result.to_csv(out_csv,index=False)

        temp=result.copy(); temp['signal_group']=temp['signal'].map(signal_group)
        grp=(temp.groupby(['horizon_s','signal_group'],as_index=False)
             .agg(baseline_MAE=('baseline_abs_error','mean'),
                  long_MAE=('long_abs_error','mean'),final_MAE=('abs_error','mean')))
        grp.insert(0,'test_id',test_no); grp.insert(0,'run',rec['run']); grp.insert(0,'environment',rec['environment']); grp.insert(0,'terrain_type',rec['terrain_type']); grp.insert(4,'ttest_s',actual)
        v5_grouped.append(grp)

        rr=result[result['signal'].str.contains('contact_probability')].dropna(subset=['contact_regime_pred'])
        regime_acc=float((rr['contact_regime_pred'].astype(int)==rr['contact_regime_truth'].astype(int)).mean())
        v5_regime_rows.append(dict(
            terrain_type=rec['terrain_type'],environment=rec['environment'],run=rec['run'],
            test_id=test_no,ttest_s=actual,regime_accuracy=regime_acc
        ))

        print('\nV5 RESULT SAMPLE @ 0.1s')
        display(result[np.isclose(result.horizon_s,0.1)][[
            'signal','baseline_pred','short_native_pred','long_direct_pred','prediction','ground_truth','abs_error'
        ]].head(20))
        print('4-regime accuracy across all horizons/feet:',regime_acc)

        del sm,lm
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

v5_all=pd.concat(v5_results,ignore_index=True)
v5_grp=pd.concat(v5_grouped,ignore_index=True)
v5_reg=pd.DataFrame(v5_regime_rows)

v5_all.to_csv(V5_OUT/'V5_ALL_16_TESTS_ALL_SENSOR_VALUES.csv',index=False)
v5_grp.to_csv(V5_OUT/'V5_ALL_16_TESTS_GROUPED_METRICS.csv',index=False)
v5_reg.to_csv(V5_OUT/'V5_CONTACT_REGIME_ACCURACY.csv',index=False)

v5_mean=(v5_grp.groupby(['signal_group','horizon_s'],as_index=False)
         .agg(baseline_MAE=('baseline_MAE','mean'),long_MAE=('long_MAE','mean'),final_MAE=('final_MAE','mean')))
v5_mean.to_csv(V5_OUT/'V5_MEAN_METRICS_ACROSS_16_TESTS.csv',index=False)

print('\n'+'='*100)
print('V5 MEAN ACROSS ALL 16 TESTS')
print('='*100)
display(v5_mean)
print('\nMean 4-regime accuracy:',v5_reg['regime_accuracy'].mean())
print('\nFILES:')
for p in sorted(V5_OUT.iterdir()):
    if p.is_file(): print(' ',p)
print('\nDONE — V5 physics-informed/contact-regime experiment complete.')

# Optional direct V4 -> V5 comparison if the V4 summary CSV is still present.
v4_mean_path = Path('/content/v4_residual_prior_direct_results/MEAN_METRICS_ACROSS_16_TESTS.csv')
if v4_mean_path.exists():
    v4m = pd.read_csv(v4_mean_path).rename(columns={'final_MAE':'V4_final_MAE'})
    cmp = v5_mean.merge(
        v4m[['signal_group','horizon_s','V4_final_MAE']],
        on=['signal_group','horizon_s'],how='left'
    )
    cmp['V5_vs_V4_percent_change'] = 100.0*(cmp['final_MAE']-cmp['V4_final_MAE'])/(cmp['V4_final_MAE']+1e-12)
    cmp.to_csv(V5_OUT/'V4_VS_V5_MEAN_COMPARISON.csv',index=False)
    print('\nV4 -> V5 comparison (negative % = V5 improved):')
    display(cmp[cmp['signal_group'].isin(['tau','force','contact'])])

# %% notebook cell 4
# ============================================================================
# V6 PATCH — CONTINUOUS CONTACT LOAD + HARD PHYSICS TAU RESIDUAL + CLEAN FUSION
#
# Paste this SINGLE CELL at the END of the completed V5 cell and run it.
# It REUSES run_records / visual features already in RAM. No re-download.
#
# WHAT CHANGED FROM V5
# ---------------------------------------------------------------------------
# 1) NO short proxy after +0.1 s.
#       h = 0.1 s : short + long are fused (same reliability weighting concept)
#       h >= 0.2 s: final prediction = direct LONG prediction exactly.
#    This is directly motivated by V5: long was much better than the fused
#    short proxy at 0.2--0.6 s for force/tau/dq.
#
# 2) REMOVE the 4-class light/medium/heavy force discretization.
#    Replace it with three continuous quantities per foot:
#       p_contact       = sigmoid(binary-contact logit)
#       force magnitude = exp(log-magnitude prediction)
#       force direction = normalized 3-vector
#    Final force is HARD physically coupled:
#       F_hat = p_contact * |F|_hat * direction_hat
#    This preserves continuous load information instead of putting 20 N and
#    200 N in one broad class.
#
# 3) TORQUE is no longer an unconstrained independent sensor head.
#    A causal continuous inverse-dynamics model is identified from pre-test data:
#       tau_phys = Phi(q,dq,ddq,F,p_contact) W
#    and the Transformer predicts ONLY a bounded correction:
#       tau_hat = tau_phys + Delta_tau_NN
#    Therefore predicted q,dq,F/contact directly enter the torque prediction.
#
#    Phi is a low-order rigid-body-inspired basis:
#       [ddq, dq, sin(q), cos(q), F, |F|, contact, 1]
#    It is fitted causally by ridge regression and never uses post-ttest truth.
#
# IMPORTANT SCIENTIFIC NOTE ABOUT "EXACT" M,C,g,J:
# The extracted TartanGround signals in this notebook do NOT include the floating-
# base pose/orientation required to build the full generalized q,v state. Therefore
# this cell does NOT pretend a URDF alone is enough for exact floating-base inverse
# dynamics. It scans for an ANYmal URDF and reports it, but uses the causal identified
# continuous dynamics above unless the dataset is later extended with base pose.
# ============================================================================

import copy, math, gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import display

# ---------------------------------------------------------------------------
# 0) SANITY / CONFIG
# ---------------------------------------------------------------------------
_required_v6 = [
    'run_records','CausalHistoryEncoder','TerrainEncoder','make_direct_sample',
    'make_range_windows','make_local_windows','make_stats_between',
    'make_local_stats','normalize_arrays','nearest_index','horizon_offsets',
    'seed_everything','SHORT_CONTEXT_SEC','LONG_CONTEXT_SEC','SHORT_HOP_SEC',
    'LONG_HOP_SEC','SHORT_HORIZONS','LONG_HORIZONS','TRAIN_LOOKBACK_SEC',
    'SHORT_RECENCY_HALF_LIFE_SEC','LONG_RECENCY_HALF_LIFE_SEC','DEVICE','SEED',
    'WEIGHT_DECAY','GRAD_CLIP','LR_PRIOR','LR_LOCAL','EPS','CONTACT_THRESHOLD'
]
_missing_v6 = [x for x in _required_v6 if x not in globals()]
if _missing_v6:
    raise RuntimeError(
        'V6 patch must be run after the completed V5/V4 experiment in the SAME runtime. '
        f'Missing: {_missing_v6}'
    )

V6_OUT = Path('/content/v6_continuous_contact_physics_results')
V6_OUT.mkdir(parents=True, exist_ok=True)

# Same amount of training as V5 for a fair ablation.
V6_PRIOR_EPOCHS_SHORT = 2
V6_PRIOR_EPOCHS_LONG  = 3
V6_LOCAL_EPOCHS_SHORT = 4
V6_LOCAL_EPOCHS_LONG  = 4

# Data / physics loss weights.
W_Q_DATA          = 1.00
W_DQ_DATA         = 1.00
W_TAU_DATA        = 1.75
W_FORCE_VEC       = 2.25
W_BODY_DATA       = 0.50
W_CONTACT_BIN     = 0.55
W_FORCE_LOGMAG    = 0.60
W_FORCE_DIR       = 0.35
W_KIN             = 0.20
W_NEWTON          = 0.10
W_TAU_RESID_PRIOR = 0.025
RIDGE_V6          = 3e-3

# Clip only absurd numerical derivative spikes while fitting physics.
DDQ_QUANTILE_CLIP = 0.995

# Predicted normalized tau residual is bounded to +/- this many tau stds.
TAU_RESID_STD_LIMIT = 4.0

# Physics input per horizon:
# q_cv(12), tau_phys0(12), body_acc_phys(3), |F|scaled(4), contact(4), F_dir(12)
V6_PHYS_DIM = 47

print('\n' + '='*105)
print('V6: CONTINUOUS CONTACT LOAD + PHYSICS-RESIDUAL TORQUE + CLEAN LONG FUSION')
print('='*105)

# Scan only as a diagnostic. Full floating-base RBD needs base pose/orientation,
# which the current extraction does not provide.
_urdfs = []
try:
    for _root in [Path('/content/tartanground'), Path('/content')]:
        if _root.exists():
            _urdfs.extend([p for p in _root.rglob('*.urdf') if 'anymal' in p.name.lower()])
    _urdfs = sorted(set(_urdfs))
except Exception:
    _urdfs = []

if _urdfs:
    print('ANYmal-like URDF found:', _urdfs[0])
    print('Not using full floating-base M,C,g,J yet: extracted V4/V5 data has no base pose/orientation state.')
else:
    print('No ANYmal URDF found in current downloaded runtime.')
    print('Using causal continuous identified inverse dynamics; no fake mass/inertia/Jacobian constants inserted.')


# ---------------------------------------------------------------------------
# 1) CONTINUOUS CAUSAL PHYSICS IDENTIFICATION
# ---------------------------------------------------------------------------
def _ridge_fit_v6(X, Y, ridge=RIDGE_V6):
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    scale = np.sqrt((X*X).mean(axis=0) + 1e-8)
    Xs = X / scale[None,:]
    A = Xs.T @ Xs + float(ridge) * np.eye(Xs.shape[1])
    B = Xs.T @ Y
    try:
        W_scaled = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        W_scaled = np.linalg.pinv(A) @ B
    W = W_scaled / scale[:,None]
    return W.astype(np.float32)


def _compute_ddq_v6(data):
    t = np.asarray(data['time'], np.float64)
    dq = np.asarray(data['P'][:,:,3:6], np.float64)
    ddq = np.gradient(dq, t, axis=0)
    ddq[~np.isfinite(ddq)] = 0.0
    lim = np.quantile(np.abs(ddq), DDQ_QUANTILE_CLIP, axis=0, keepdims=True) + 1e-6
    return np.clip(ddq, -lim, lim).astype(np.float32)


def fit_continuous_physics_v6(data, start_time, end_time):
    """
    Causal continuous inverse-dynamics proxy using only [start_time,end_time].

    Per leg:
      tau = [ddq,dq,sin(q),cos(q),F,|F|,contact,1] W

    Body Newton proxy:
      body_acc = [sum_i F_i,1] W_body

    No light/medium/heavy classes are used.
    """
    t = np.asarray(data['time'], np.float64)
    P = np.asarray(data['P'], np.float32)
    B = np.asarray(data['B'], np.float32)
    ddq = _compute_ddq_v6(data)

    mask = (t >= float(start_time)) & (t <= float(end_time))
    idx = np.flatnonzero(mask)
    if len(idx) < 100:
        raise RuntimeError(f'V6 physics fit too short: {start_time:.2f}->{end_time:.2f}')

    invW = []
    for leg in range(4):
        q = P[idx,leg,0:3]
        dq_l = P[idx,leg,3:6]
        tau = P[idx,leg,6:9]
        force = P[idx,leg,9:12]
        ddq_l = ddq[idx,leg]
        mag = np.linalg.norm(force,axis=-1,keepdims=True).astype(np.float32)
        contact = (mag >= float(CONTACT_THRESHOLD)).astype(np.float32)

        # 3+3+3+3+3+1+1+1 = 18 features.
        X = np.concatenate([
            ddq_l, dq_l, np.sin(q), np.cos(q), force, mag, contact,
            np.ones((len(idx),1),np.float32)
        ],axis=1)
        invW.append(_ridge_fit_v6(X,tau))

    invW = np.stack(invW,axis=0).astype(np.float32)  # 4,18,3

    sumF = P[idx,:,9:12].sum(axis=1)
    Xb = np.concatenate([sumF,np.ones((len(idx),1),np.float32)],axis=1)
    bodyW = _ridge_fit_v6(Xb,B[idx,0:3])

    # Diagnostics / sensible scale caps learned causally.
    mag_all = np.linalg.norm(P[idx,:,9:12],axis=-1).reshape(-1)
    positive = mag_all[mag_all >= float(CONTACT_THRESHOLD)]
    if len(positive):
        mag_cap = float(max(np.quantile(positive,0.995)*1.35, CONTACT_THRESHOLD*2.0))
    else:
        mag_cap = 500.0

    tau_fit=P[idx,:,6:9]
    tau_lo=np.quantile(tau_fit,0.0025,axis=0).astype(np.float32)
    tau_hi=np.quantile(tau_fit,0.9975,axis=0).astype(np.float32)
    # Add a causal margin rather than allowing the identified model to explode.
    span=np.maximum(tau_hi-tau_lo,1.0)
    tau_lo=tau_lo-0.20*span
    tau_hi=tau_hi+0.20*span

    return dict(invW=invW,bodyW=bodyW,mag_cap=np.float32(mag_cap),tau_lo=tau_lo,tau_hi=tau_hi)


def physics_tau_cont_np_v6(physics,q,dq,ddq,force,p_contact):
    q=np.asarray(q,np.float32); dq=np.asarray(dq,np.float32); ddq=np.asarray(ddq,np.float32)
    force=np.asarray(force,np.float32); p_contact=np.asarray(p_contact,np.float32).reshape(4,1)
    mag=np.linalg.norm(force,axis=-1,keepdims=True).astype(np.float32)
    X=np.concatenate([
        ddq,dq,np.sin(q),np.cos(q),force,mag,p_contact,
        np.ones((4,1),np.float32)
    ],axis=1)
    tau=np.einsum('lf,lfj->lj',X,physics['invW']).astype(np.float32)
    if 'tau_lo' in physics and 'tau_hi' in physics:
        tau=np.clip(tau,physics['tau_lo'],physics['tau_hi'])
    return tau


def physics_acc_np_v6(physics,force):
    sf=np.asarray(force,np.float32).sum(axis=0)
    X=np.concatenate([sf,np.array([1.0],np.float32)])
    return (X @ physics['bodyW']).astype(np.float32)


def force_meta_v6(data,start_time,end_time):
    t=np.asarray(data['time'],np.float64)
    f=np.asarray(data['P'][:,:,9:12],np.float32)
    mask=(t>=float(start_time))&(t<=float(end_time))
    mag=np.linalg.norm(f[mask],axis=-1).reshape(-1)
    logm=np.log1p(np.maximum(mag,0.0))
    mean=float(logm.mean())
    std=float(logm.std()+1e-4)
    pos=mag[mag>=float(CONTACT_THRESHOLD)]
    cap=float(max(np.quantile(pos,0.995)*1.35,CONTACT_THRESHOLD*2.0)) if len(pos) else 500.0
    return dict(logmag_mean=np.float32(mean),logmag_std=np.float32(std),mag_cap=np.float32(cap))


for _rec in run_records:
    _start=float(_rec['data']['time'][0])
    _end=float(min(_rec['ttests']))
    _rec['physics_prior_v6']=fit_continuous_physics_v6(_rec['data'],_start,_end)
    _rec['force_meta_prior_v6']=force_meta_v6(_rec['data'],_start,_end)


# ---------------------------------------------------------------------------
# 2) MODEL — BINARY CONTACT + CONTINUOUS FORCE MAG/DIR + PHYSICS TAU RESIDUAL
# ---------------------------------------------------------------------------
class ContinuousContactPhysicsDynamicsV6(nn.Module):
    def __init__(self,history_steps,n_horizons,dino_dim=384,conv_dim=768):
        super().__init__()
        self.history_steps=int(history_steps)
        self.n_horizons=int(n_horizons)

        self.leg_encoder=CausalHistoryEncoder(24,history_steps,128,2,4)
        self.leg_identity=nn.Embedding(4,128)
        self.body_encoder=CausalHistoryEncoder(18,history_steps,128,2,4)
        self.terrain_encoder=TerrainEncoder(dino_dim,conv_dim)

        self.context_net=nn.Sequential(
            nn.LayerNorm(896),nn.Linear(896,512),nn.GELU(),nn.Dropout(0.1),
            nn.Linear(512,256),nn.GELU(),nn.LayerNorm(256)
        )
        self.control_proj=nn.Sequential(nn.Linear(36,128),nn.GELU(),nn.LayerNorm(128))
        self.horizon_proj=nn.Sequential(
            nn.Linear(1,64),nn.GELU(),nn.Linear(64,128),nn.GELU(),nn.LayerNorm(128)
        )
        self.physics_proj=nn.Sequential(
            nn.Linear(V6_PHYS_DIM,128),nn.GELU(),nn.Linear(128,128),nn.GELU(),nn.LayerNorm(128)
        )

        # z: 256 context + 128 controls + 128 horizon + 128 physics = 640
        self.contact_head=nn.Sequential(
            nn.Linear(640,256),nn.GELU(),nn.Dropout(0.1),nn.Linear(256,4)
        )
        self.contact_embed=nn.Sequential(
            nn.Linear(4,32),nn.GELU(),nn.LayerNorm(32)
        )

        # q,dq residuals (24) + body residual (12)
        self.state_body_head=nn.Sequential(
            nn.Linear(672,256),nn.GELU(),nn.Dropout(0.1),nn.Linear(256,36)
        )

        # Continuous force heads.
        self.force_mag_head=nn.Sequential(
            nn.Linear(672,256),nn.GELU(),nn.Dropout(0.1),nn.Linear(256,4)
        )
        self.force_dir_head=nn.Sequential(
            nn.Linear(672,256),nn.GELU(),nn.Dropout(0.1),nn.Linear(256,12)
        )

        # Torque residual is conditioned on the model's own predicted contact/load descriptor.
        self.force_desc_embed=nn.Sequential(
            nn.Linear(20,64),nn.GELU(),nn.LayerNorm(64)
        )
        self.tau_resid_head=nn.Sequential(
            nn.Linear(736,256),nn.GELU(),nn.Dropout(0.1),nn.Linear(256,12)
        )

    def encode_context(self,p_hist,b_hist,u_hist,l_hist,e_hist,dino,conv,depth):
        Bn,T=p_hist.shape[0],p_hist.shape[2]
        ulegs=u_hist.unsqueeze(1).expand(-1,4,-1,-1)
        leg_in=torch.cat([p_hist,ulegs,l_hist,e_hist],dim=-1).reshape(Bn*4,T,24)
        lf=self.leg_encoder(leg_in)
        ids=torch.arange(4,device=p_hist.device).repeat(Bn)
        lf=(lf+self.leg_identity(ids)).reshape(Bn,4,128).reshape(Bn,512)
        bf=self.body_encoder(torch.cat([b_hist,u_hist],dim=-1))
        terrain=self.terrain_encoder(dino,conv,depth)
        return self.context_net(torch.cat([lf,bf,terrain],dim=-1))

    def forward(self,p_hist,b_hist,u_hist,l_hist,e_hist,dino,conv,depth,
                future_control_features,horizon_values,physics_features):
        ctx=self.encode_context(p_hist,b_hist,u_hist,l_hist,e_hist,dino,conv,depth)
        Hn=future_control_features.shape[1]
        if Hn!=self.n_horizons:
            raise RuntimeError(f'V6 expected {self.n_horizons} horizons, got {Hn}')

        c=self.control_proj(future_control_features)
        h=self.horizon_proj(horizon_values)
        ph=self.physics_proj(physics_features)
        ctx=ctx.unsqueeze(1).expand(-1,Hn,-1)
        z=torch.cat([ctx,c,h,ph],dim=-1)

        contact_logits=self.contact_head(z)                    # B,H,4
        p_contact=torch.sigmoid(contact_logits)
        ce=self.contact_embed(p_contact)
        zc=torch.cat([z,ce],dim=-1)                            # B,H,672

        sb=self.state_body_head(zc)
        logmag_z=self.force_mag_head(zc)                       # B,H,4
        dir_raw=self.force_dir_head(zc).reshape(-1,Hn,4,3)
        dir_unit=F.normalize(dir_raw,dim=-1,eps=1e-6)

        force_desc=torch.cat([
            p_contact,logmag_z,dir_unit.reshape(-1,Hn,12)
        ],dim=-1)                                              # B,H,20
        fe=self.force_desc_embed(force_desc)
        tau_raw=self.tau_resid_head(torch.cat([zc,fe],dim=-1)).reshape(-1,Hn,4,3)
        tau_resid_norm=TAU_RESID_STD_LIMIT*torch.tanh(tau_raw/TAU_RESID_STD_LIMIT)

        residual_qdq=torch.zeros((p_hist.shape[0],Hn,4,6),dtype=sb.dtype,device=sb.device)
        residual_qdq[:]=sb[...,:24].reshape(-1,Hn,4,6)
        residual_b=sb[...,24:36]

        return dict(
            residual_qdq=residual_qdq,
            residual_b=residual_b,
            contact_logits=contact_logits,
            p_contact=p_contact,
            logmag_z=logmag_z,
            dir_unit=dir_unit,
            tau_resid_norm=tau_resid_norm,
        )


# Optional warm-start of representation layers from the completed V5 GLOBAL priors.
def _warmstart_v6_from_v5(v6_model,v5_model):
    if v5_model is None:
        return 0
    allowed=('leg_encoder.','leg_identity.','body_encoder.','terrain_encoder.',
             'context_net.','control_proj.','horizon_proj.')
    src=v5_model.state_dict(); dst=v6_model.state_dict(); copied=0
    for k,v in src.items():
        if k.startswith(allowed) and k in dst and dst[k].shape==v.shape:
            dst[k]=v.detach().clone(); copied+=1
    v6_model.load_state_dict(dst,strict=False)
    return copied


# ---------------------------------------------------------------------------
# 3) SAMPLE + PHYSICS FEATURES
# ---------------------------------------------------------------------------
def make_v6_sample(data,visual,stats,physics,idx,history_steps,horizons,norm=None):
    base=make_direct_sample(data,visual,stats,idx,history_steps,horizons,norm=norm)
    current_p=np.asarray(data['P'][idx],np.float32)
    horizons_arr=np.asarray(horizons,np.float32)

    cur_force=current_p[:,9:12]
    cur_mag=np.linalg.norm(cur_force,axis=-1)
    cur_contact=(cur_mag>=float(CONTACT_THRESHOLD)).astype(np.float32)
    cur_dir=cur_force/(cur_mag[:,None]+1e-6)
    force_scale=max(float(physics['mag_cap']),1.0)

    phys=[]
    for h,bp in zip(horizons_arr,base['baseline_p_raw']):
        q_cv=bp[:,0:3]
        dq0=current_p[:,3:6]
        ddq0=np.zeros((4,3),np.float32)
        tau0=physics_tau_cont_np_v6(physics,q_cv,dq0,ddq0,cur_force,cur_contact)
        acc0=physics_acc_np_v6(physics,cur_force)

        qn=((q_cv-stats['P_mean'][0:3])/stats['P_std'][0:3]).reshape(-1)
        taun=((tau0-stats['P_mean'][6:9])/stats['P_std'][6:9]).reshape(-1)
        accn=(acc0-stats['B_mean'][0:3])/stats['B_std'][0:3]
        mn=(cur_mag/force_scale).reshape(-1)
        feat=np.concatenate([qn,taun,accn,mn,cur_contact.reshape(-1),cur_dir.reshape(-1)]).astype(np.float32)
        if feat.size!=V6_PHYS_DIM:
            raise RuntimeError(f'V6 physics feature dim {feat.size} != {V6_PHYS_DIM}')
        phys.append(feat)

    base['physics_features']=torch.as_tensor(np.asarray(phys,np.float32)[None],device=DEVICE)
    base['current_p_raw_v6']=current_p
    base['horizons_raw_v6']=horizons_arr
    return base


# ---------------------------------------------------------------------------
# 4) RAW PREDICTION CONSTRUCTION — same function used by loss and inference
# ---------------------------------------------------------------------------
def _torch_tau_phys_v6(physics,q,dq,ddq,force,p_contact):
    mag=torch.linalg.norm(force,dim=-1,keepdim=True)
    ones=torch.ones((*q.shape[:-1],1),dtype=q.dtype,device=q.device)
    X=torch.cat([ddq,dq,torch.sin(q),torch.cos(q),force,mag,p_contact.unsqueeze(-1),ones],dim=-1)
    W=torch.as_tensor(physics['invW'],dtype=q.dtype,device=q.device)
    tau=torch.einsum('bhlf,lfj->bhlj',X,W)
    if 'tau_lo' in physics and 'tau_hi' in physics:
        lo=torch.as_tensor(physics['tau_lo'],dtype=q.dtype,device=q.device).view(1,1,4,3)
        hi=torch.as_tensor(physics['tau_hi'],dtype=q.dtype,device=q.device).view(1,1,4,3)
        tau=torch.maximum(torch.minimum(tau,hi),lo)
    return tau


def _torch_body_acc_v6(physics,force):
    sf=force.sum(dim=2)
    ones=torch.ones((*sf.shape[:-1],1),dtype=sf.dtype,device=sf.device)
    X=torch.cat([sf,ones],dim=-1)
    W=torch.as_tensor(physics['bodyW'],dtype=sf.dtype,device=sf.device)
    return X @ W


def construct_v6_raw(out,sample,stats,physics,fmeta):
    dtype=out['residual_qdq'].dtype; dev=out['residual_qdq'].device
    Pstd=torch.as_tensor(stats['P_std'],dtype=dtype,device=dev)
    Bstd=torch.as_tensor(stats['B_std'],dtype=dtype,device=dev)
    bp=torch.as_tensor(sample['baseline_p_raw'],dtype=dtype,device=dev).unsqueeze(0)
    bb=torch.as_tensor(sample['baseline_b_raw'],dtype=dtype,device=dev).unsqueeze(0)

    qdq=bp[:,:,:,0:6] + out['residual_qdq']*Pstd[0:6].view(1,1,1,6)
    q=qdq[:,:,:,0:3]; dq=qdq[:,:,:,3:6]

    # Continuous magnitude and direction.
    lm_mean=torch.as_tensor(float(fmeta['logmag_mean']),dtype=dtype,device=dev)
    lm_std=torch.as_tensor(float(fmeta['logmag_std']),dtype=dtype,device=dev)
    logmag=out['logmag_z']*lm_std+lm_mean
    logmag=torch.clamp(logmag,min=0.0,max=float(np.log1p(float(fmeta['mag_cap']))))
    mag=torch.expm1(logmag)
    p_contact=out['p_contact']
    force=p_contact.unsqueeze(-1)*mag.unsqueeze(-1)*out['dir_unit']

    # Physics tau + bounded learned correction.
    cur=torch.as_tensor(sample['current_p_raw_v6'],dtype=dtype,device=dev)
    dq0=cur[:,3:6].view(1,1,4,3)
    h=torch.as_tensor(sample['horizons_raw_v6'],dtype=dtype,device=dev).view(1,-1,1,1)
    ddq=(dq-dq0)/(h+1e-6)
    tau_phys=_torch_tau_phys_v6(physics,q,dq,ddq,force,p_contact)
    tau=tau_phys+out['tau_resid_norm']*Pstd[6:9].view(1,1,1,3)

    predP=torch.zeros_like(bp)
    predP[:,:,:,0:6]=qdq
    predP[:,:,:,6:9]=tau
    predP[:,:,:,9:12]=force
    predB=bb+out['residual_b']*Bstd.view(1,1,12)

    return predP,predB,tau_phys,mag,p_contact


# ---------------------------------------------------------------------------
# 5) V6 LOSS
# ---------------------------------------------------------------------------
def v6_loss(out,sample,stats,physics,fmeta):
    predP,predB,tau_phys,pred_mag,p_contact=construct_v6_raw(out,sample,stats,physics,fmeta)
    dtype=predP.dtype; dev=predP.device
    gtP=torch.as_tensor(sample['target_p_raw'],dtype=dtype,device=dev).unsqueeze(0)
    gtB=torch.as_tensor(sample['target_b_raw'],dtype=dtype,device=dev).unsqueeze(0)
    Pstd=torch.as_tensor(stats['P_std'],dtype=dtype,device=dev)
    Bstd=torch.as_tensor(stats['B_std'],dtype=dtype,device=dev)
    Hn=predP.shape[1]
    hw=torch.linspace(1.0,2.0,Hn,device=dev)

    def raw_norm_mse(a,b,scale):
        e=((a-b)/(scale+1e-6))**2
        dims=tuple(range(2,e.ndim))
        return (e.mean(dim=dims)*hw[None,:]).mean()

    lq=raw_norm_mse(predP[:,:,:,0:3],gtP[:,:,:,0:3],Pstd[0:3].view(1,1,1,3))
    ldq=raw_norm_mse(predP[:,:,:,3:6],gtP[:,:,:,3:6],Pstd[3:6].view(1,1,1,3))
    ltau=raw_norm_mse(predP[:,:,:,6:9],gtP[:,:,:,6:9],Pstd[6:9].view(1,1,1,3))
    lforce=raw_norm_mse(predP[:,:,:,9:12],gtP[:,:,:,9:12],Pstd[9:12].view(1,1,1,3))
    lbody=raw_norm_mse(predB,gtB,Bstd.view(1,1,12))

    gt_force=gtP[:,:,:,9:12]
    gt_mag=torch.linalg.norm(gt_force,dim=-1)
    gt_contact=(gt_mag>=float(CONTACT_THRESHOLD)).float()

    # Binary contact BCE, mildly rebalanced from the current causal sample.
    pos=gt_contact.mean().detach()
    pos_weight=torch.clamp((1.0-pos)/(pos+1e-4),0.5,4.0)
    bce=F.binary_cross_entropy_with_logits(
        out['contact_logits'],gt_contact,
        pos_weight=pos_weight,reduction='none'
    )
    lcontact=(bce.mean(dim=2)*hw[None,:]).mean()

    # Continuous force magnitude in log space.
    lm_mean=torch.as_tensor(float(fmeta['logmag_mean']),dtype=dtype,device=dev)
    lm_std=torch.as_tensor(float(fmeta['logmag_std']),dtype=dtype,device=dev)
    gt_logz=(torch.log1p(gt_mag)-lm_mean)/(lm_std+1e-6)
    magerr=F.smooth_l1_loss(out['logmag_z'],gt_logz,reduction='none')
    # Contact feet matter most, but no-contact feet still teach magnitude near zero.
    magw=0.20+0.80*gt_contact
    lmag=(magerr*magw*hw.view(1,-1,1)).mean()

    # Force direction cosine loss only where true contact exists.
    gt_dir=gt_force/(gt_mag.unsqueeze(-1)+1e-6)
    cos=(out['dir_unit']*gt_dir).sum(dim=-1).clamp(-1.0,1.0)
    direrr=(1.0-cos)*gt_contact
    denom=(gt_contact*hw.view(1,-1,1)).sum()+1e-6
    ldir=(direrr*hw.view(1,-1,1)).sum()/denom

    # Trapezoidal kinematic consistency.
    cur=torch.as_tensor(sample['current_p_raw_v6'],dtype=dtype,device=dev)
    q0=cur[:,0:3].view(1,1,4,3)
    dq0=cur[:,3:6].view(1,1,4,3)
    h=torch.as_tensor(sample['horizons_raw_v6'],dtype=dtype,device=dev).view(1,-1,1,1)
    qkin=q0+0.5*h*(dq0+predP[:,:,:,3:6])
    lkin=raw_norm_mse(predP[:,:,:,0:3],qkin,Pstd[0:3].view(1,1,1,3))

    # Newton consistency only on body acceleration channels.
    acc_phys=_torch_body_acc_v6(physics,predP[:,:,:,9:12])
    acc_scale=Bstd[0:3].view(1,1,3)
    lnewton=((((predB[:,:,0:3]-acc_phys)/(acc_scale+1e-6))**2).mean(dim=2)*hw[None,:]).mean()

    # Keep the learned tau correction as a residual rather than letting it replace physics.
    ltaur=(out['tau_resid_norm']**2).mean()

    total=(
        W_Q_DATA*lq + W_DQ_DATA*ldq + W_TAU_DATA*ltau +
        W_FORCE_VEC*lforce + W_BODY_DATA*lbody +
        W_CONTACT_BIN*lcontact + W_FORCE_LOGMAG*lmag + W_FORCE_DIR*ldir +
        W_KIN*lkin + W_NEWTON*lnewton + W_TAU_RESID_PRIOR*ltaur
    )
    parts=dict(
        q=float(lq.detach()),dq=float(ldq.detach()),tau=float(ltau.detach()),
        force=float(lforce.detach()),contact=float(lcontact.detach()),
        mag=float(lmag.detach()),dir=float(ldir.detach()),kin=float(lkin.detach()),
        newton=float(lnewton.detach()),tau_resid=float(ltaur.detach())
    )
    return total,parts


def v6_model_step(model,opt,sample,stats,physics,fmeta,weight=1.0):
    opt.zero_grad(set_to_none=True)
    out=model(
        sample['p_hist'],sample['b_hist'],sample['u_hist'],sample['l_hist'],sample['e_hist'],
        sample['dino'],sample['conv'],sample['depth'],sample['future_control_features'],
        sample['horizon_values'],sample['physics_features']
    )
    raw,parts=v6_loss(out,sample,stats,physics,fmeta)
    loss=raw*float(weight)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP)
    opt.step()
    val=float(raw.detach().cpu())
    del out,raw,loss
    return val,parts


# ---------------------------------------------------------------------------
# 6) TRAINING — SAME STRICT CHRONOLOGICAL PER-WINDOW UPDATES
# ---------------------------------------------------------------------------
def train_v6_prior(run_records,context_sec,hop_sec,horizons,epochs,seed,label,v5_warm=None):
    seed_everything(seed)
    first=run_records[0]
    dt=float(np.median(np.diff(first['data']['time'])))
    H=int(round(context_sec/dt))
    model=ContinuousContactPhysicsDynamicsV6(
        H,len(horizons),first['visual']['DINO'].shape[1],
        first['visual']['CONV'].reshape(len(first['visual']['CONV']),-1).shape[1]
    ).to(DEVICE)
    copied=_warmstart_v6_from_v5(model,v5_warm)
    if copied:
        print(f'V6 {label}: warm-started {copied} compatible representation tensors from V5 global prior.')
    opt=torch.optim.AdamW(model.parameters(),lr=LR_PRIOR,weight_decay=WEIGHT_DECAY)

    print('\n'+'='*105)
    print('V6 GLOBAL PRIOR ::',label)
    print('binary contact + continuous force magnitude/direction + physics-residual tau')
    print('NO BATCH / NO SHUFFLE / one optimizer.step per chronological window')
    print('='*105)

    for ep in range(epochs):
        vals=[]
        for rec in run_records:
            data,visual=rec['data'],rec['visual']
            start=float(data['time'][0]); end=float(min(rec['ttests']))
            stats=make_stats_between(data,start,end)
            norm=normalize_arrays(data,stats)
            physics=rec['physics_prior_v6']; fmeta=rec['force_meta_prior_v6']
            windows,H2=make_range_windows(data,visual,start,end,context_sec,hop_sec,max(horizons))
            if H2!=H: raise RuntimeError('V6 prior history mismatch')
            print(f'  prior ep {ep+1}/{epochs} | {rec["terrain_type"]} | windows={len(windows)}')
            for wi,(idx,_,te) in enumerate(windows,1):
                s=make_v6_sample(data,visual,stats,physics,idx,H,horizons,norm=norm)
                v,parts=v6_model_step(model,opt,s,stats,physics,fmeta,1.0)
                vals.append(v)
                if wi in (1,len(windows)):
                    print(f'    {wi:03d}/{len(windows):03d} target_end={te:.3f} loss={v:.4f} '
                          f'tau={parts["tau"]:.3f} force={parts["force"]:.3f} '
                          f'contact={parts["contact"]:.3f} mag={parts["mag"]:.3f}')
                del s
        print(f'  V6 PRIOR {label} epoch {ep+1} mean loss={np.mean(vals):.6f}')
    return model


def adapt_v6_local(prior_model,data,visual,ttest,context_sec,hop_sec,horizons,
                   half_life,epochs,seed,label):
    seed_everything(seed)
    model=copy.deepcopy(prior_model).to(DEVICE)
    start=max(float(data['time'][0]),float(ttest)-TRAIN_LOOKBACK_SEC)
    stats=make_local_stats(data,ttest)
    norm=normalize_arrays(data,stats)
    physics=fit_continuous_physics_v6(data,start,float(ttest))
    fmeta=force_meta_v6(data,start,float(ttest))
    windows,H=make_local_windows(data,visual,ttest,context_sec,hop_sec,max(horizons),half_life)
    if H!=model.history_steps: raise RuntimeError('V6 local history mismatch')
    opt=torch.optim.AdamW(model.parameters(),lr=LR_LOCAL,weight_decay=WEIGHT_DECAY)

    print(f'\n  V6 LOCAL {label}: ttest={ttest:.3f}s windows={len(windows)} '
          f'oldest_w={windows[0][1]:.5f} newest_w={windows[-1][1]:.5f}')
    for ep in range(epochs):
        vals=[]
        for wi,(idx,w,_,te) in enumerate(windows,1):
            s=make_v6_sample(data,visual,stats,physics,idx,H,horizons,norm=norm)
            v,parts=v6_model_step(model,opt,s,stats,physics,fmeta,w)
            vals.append(v)
            if wi==1 or wi==len(windows):
                print(f'    {label} ep {ep+1:02d} {wi:03d}/{len(windows):03d} target_end={te:.3f} '
                      f'w={w:.4f} loss={v:.4f} tau={parts["tau"]:.3f} '
                      f'force={parts["force"]:.3f} contact={parts["contact"]:.3f}')
            del s
        print(f'    {label} epoch {ep+1}/{epochs} mean loss={np.mean(vals):.6f}')
    return model,stats,physics,fmeta,H


# ---------------------------------------------------------------------------
# 7) INFERENCE
# ---------------------------------------------------------------------------
def predict_v6_raw(model,sample,stats,physics,fmeta):
    model.eval()
    with torch.no_grad():
        out=model(
            sample['p_hist'],sample['b_hist'],sample['u_hist'],sample['l_hist'],sample['e_hist'],
            sample['dino'],sample['conv'],sample['depth'],sample['future_control_features'],
            sample['horizon_values'],sample['physics_features']
        )
        predP,predB,tau_phys,mag,p_contact=construct_v6_raw(out,sample,stats,physics,fmeta)
    return (
        predP[0].cpu().numpy(),predB[0].cpu().numpy(),
        p_contact[0].cpu().numpy(),mag[0].cpu().numpy(),
        tau_phys[0].cpu().numpy()
    )


def _v6_mix_weights(h):
    # Critical V6 change: short exists ONLY at its native +0.1 s horizon.
    if abs(float(h)-0.1)<1e-8:
        if 'gaussian_mix_weights' in globals():
            return gaussian_mix_weights(0.1)
        return 0.90,0.10
    return 0.0,1.0


def evaluate_v6_case(short_model,short_stats,short_physics,short_fmeta,Hs,
                     long_model,long_stats,long_physics,long_fmeta,Hl,
                     data,visual,ttest):
    idx=nearest_index(data['time'],ttest)
    snorm=normalize_arrays(data,short_stats); lnorm=normalize_arrays(data,long_stats)
    ss=make_v6_sample(data,visual,short_stats,short_physics,idx,Hs,SHORT_HORIZONS,norm=snorm)
    ls=make_v6_sample(data,visual,long_stats,long_physics,idx,Hl,LONG_HORIZONS,norm=lnorm)
    sp,sb,sc,smag,stauphys=predict_v6_raw(short_model,ss,short_stats,short_physics,short_fmeta)
    lp,lb,lc,lmag,ltauphys=predict_v6_raw(long_model,ls,long_stats,long_physics,long_fmeta)

    rows=[]
    leg_names=['FL','FR','RL','RR']
    feat=['q_HAA','q_HFE','q_KFE','dq_HAA','dq_HFE','dq_KFE',
          'tau_HAA','tau_HFE','tau_KFE','Fx','Fy','Fz']
    body=['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z',
          'body_vx','body_vy','body_vz','body_wx','body_wy','body_wz']

    for hi,h in enumerate(LONG_HORIZONS):
        ws,wl=_v6_mix_weights(h)
        bp=ls['baseline_p_raw'][hi]; bb=ls['baseline_b_raw'][hi]
        gt_p=ls['target_p_raw'][hi]; gt_b=ls['target_b_raw'][hi]
        gt_mag=np.linalg.norm(gt_p[:,9:12],axis=-1)
        gt_c=(gt_mag>=float(CONTACT_THRESHOLD)).astype(np.float32)
        base_mag=np.linalg.norm(data['P'][idx,:,9:12],axis=-1)
        base_c=(base_mag>=float(CONTACT_THRESHOLD)).astype(np.float32)

        if ws>0:
            final_p=ws*sp[0]+wl*lp[hi]
            final_b=ws*sb[0]+wl*lb[hi]
            final_c=ws*sc[0]+wl*lc[hi]
            final_mag=ws*smag[0]+wl*lmag[hi]
            short_proxy_p=sp[0]
            short_proxy_b=sb[0]
            short_proxy_c=sc[0]
        else:
            # NO short proxy beyond 0.1 s.
            final_p=lp[hi].copy(); final_b=lb[hi].copy(); final_c=lc[hi].copy(); final_mag=lmag[hi].copy()
            short_proxy_p=None; short_proxy_b=None; short_proxy_c=None

        for li,LN in enumerate(leg_names):
            for fi,FN in enumerate(feat):
                truth=float(gt_p[li,fi]); bpred=float(bp[li,fi]); lpred=float(lp[hi,li,fi]); pred=float(final_p[li,fi])
                snative=float(sp[0,li,fi]) if abs(float(h)-0.1)<1e-8 else np.nan
                sproxy=float(short_proxy_p[li,fi]) if short_proxy_p is not None else np.nan
                rows.append(dict(
                    horizon_s=float(h),signal=f'{LN}.{FN}',baseline_pred=bpred,
                    short_native_pred=snative,short_proxy_pred=sproxy,long_direct_pred=lpred,
                    lambda_short=float(ws),lambda_long=float(wl),prediction=pred,ground_truth=truth,
                    baseline_abs_error=abs(bpred-truth),
                    short_proxy_abs_error=abs(sproxy-truth) if np.isfinite(sproxy) else np.nan,
                    long_abs_error=abs(lpred-truth),signed_error=pred-truth,abs_error=abs(pred-truth),
                    contact_binary_pred=np.nan,contact_binary_truth=np.nan,
                    force_mag_pred=np.nan,force_mag_truth=np.nan
                ))

            truth=float(gt_c[li]); bpred=float(base_c[li]); lpred=float(lc[hi,li]); pred=float(final_c[li])
            snative=float(sc[0,li]) if abs(float(h)-0.1)<1e-8 else np.nan
            sproxy=float(short_proxy_c[li]) if short_proxy_c is not None else np.nan
            rows.append(dict(
                horizon_s=float(h),signal=f'{LN}.contact_probability',baseline_pred=bpred,
                short_native_pred=snative,short_proxy_pred=sproxy,long_direct_pred=lpred,
                lambda_short=float(ws),lambda_long=float(wl),prediction=pred,ground_truth=truth,
                baseline_abs_error=abs(bpred-truth),
                short_proxy_abs_error=abs(sproxy-truth) if np.isfinite(sproxy) else np.nan,
                long_abs_error=abs(lpred-truth),signed_error=pred-truth,abs_error=abs(pred-truth),
                contact_binary_pred=int(pred>=0.5),contact_binary_truth=int(truth),
                force_mag_pred=float(final_mag[li]),force_mag_truth=float(gt_mag[li])
            ))

        for bi,BN in enumerate(body):
            truth=float(gt_b[bi]); bpred=float(bb[bi]); lpred=float(lb[hi,bi]); pred=float(final_b[bi])
            snative=float(sb[0,bi]) if abs(float(h)-0.1)<1e-8 else np.nan
            sproxy=float(short_proxy_b[bi]) if short_proxy_b is not None else np.nan
            rows.append(dict(
                horizon_s=float(h),signal=BN,baseline_pred=bpred,
                short_native_pred=snative,short_proxy_pred=sproxy,long_direct_pred=lpred,
                lambda_short=float(ws),lambda_long=float(wl),prediction=pred,ground_truth=truth,
                baseline_abs_error=abs(bpred-truth),
                short_proxy_abs_error=abs(sproxy-truth) if np.isfinite(sproxy) else np.nan,
                long_abs_error=abs(lpred-truth),signed_error=pred-truth,abs_error=abs(pred-truth),
                contact_binary_pred=np.nan,contact_binary_truth=np.nan,
                force_mag_pred=np.nan,force_mag_truth=np.nan
            ))

    vi=int(visual['visual_index'][idx])
    return pd.DataFrame(rows),idx,vi


# ---------------------------------------------------------------------------
# 8) METRICS
# ---------------------------------------------------------------------------
def _signal_group_v6(s):
    s=str(s)
    if '.q_' in s: return 'q'
    if '.dq_' in s: return 'dq'
    if '.tau_' in s: return 'tau'
    if s.endswith('.Fx') or s.endswith('.Fy') or s.endswith('.Fz'): return 'force'
    if 'contact_probability' in s: return 'contact'
    return 'body'


def grouped_metrics_v6(df):
    d=df.copy(); d['signal_group']=d['signal'].map(_signal_group_v6)
    return d.groupby([
        'terrain_type','environment','run','test_id','ttest_s','horizon_s','signal_group'
    ],as_index=False).agg(
        baseline_MAE=('baseline_abs_error','mean'),
        long_MAE=('long_abs_error','mean'),
        final_MAE=('abs_error','mean')
    )


# ---------------------------------------------------------------------------
# 9) TRAIN GLOBAL PRIORS
# ---------------------------------------------------------------------------
seed_everything(SEED+6000)

_v5_short=globals().get('pi_prior_short',None)
_v5_long=globals().get('pi_prior_long',None)

v6_prior_short=train_v6_prior(
    run_records,SHORT_CONTEXT_SEC,SHORT_HOP_SEC,SHORT_HORIZONS,
    V6_PRIOR_EPOCHS_SHORT,SEED+6010,'SHORT continuous-contact',v5_warm=_v5_short
)
v6_prior_long=train_v6_prior(
    run_records,LONG_CONTEXT_SEC,LONG_HOP_SEC,LONG_HORIZONS,
    V6_PRIOR_EPOCHS_LONG,SEED+6020,'LONG continuous-contact',v5_warm=_v5_long
)

torch.save(v6_prior_short.state_dict(),V6_OUT/'V6_PRIOR_SHORT.pt')
torch.save(v6_prior_long.state_dict(),V6_OUT/'V6_PRIOR_LONG.pt')


# ---------------------------------------------------------------------------
# 10) RUN ALL 16 TESTS
# ---------------------------------------------------------------------------
v6_results=[]; v6_contact_rows=[]

for rec_i,rec in enumerate(run_records):
    terrain=rec['terrain_type']; env=rec['environment']; run=rec['run']
    data=rec['data']; visual=rec['visual']
    print('\n'+'#'*105)
    print('V6',terrain.upper(),env,run)
    print('#'*105)

    for test_no,tt in enumerate(rec['ttests'],1):
        print('\n'+'-'*100)
        print(f'V6 TEST {test_no}/4 | ttest={float(tt):.3f}s')
        print('-'*100)

        sm,ss,sphy,sfm,Hs=adapt_v6_local(
            v6_prior_short,data,visual,float(tt),SHORT_CONTEXT_SEC,SHORT_HOP_SEC,
            SHORT_HORIZONS,SHORT_RECENCY_HALF_LIFE_SEC,V6_LOCAL_EPOCHS_SHORT,
            SEED+6100+rec_i*100+test_no,'SHORT'
        )
        lm,ls,lphy,lfm,Hl=adapt_v6_local(
            v6_prior_long,data,visual,float(tt),LONG_CONTEXT_SEC,LONG_HOP_SEC,
            LONG_HORIZONS,LONG_RECENCY_HALF_LIFE_SEC,V6_LOCAL_EPOCHS_LONG,
            SEED+6200+rec_i*100+test_no,'LONG'
        )

        result,idx,vi=evaluate_v6_case(
            sm,ss,sphy,sfm,Hs,lm,ls,lphy,lfm,Hl,data,visual,float(tt)
        )
        actual=float(data['time'][idx])
        result.insert(0,'ttest_s',actual); result.insert(0,'test_id',test_no)
        result.insert(0,'run',run); result.insert(0,'environment',env); result.insert(0,'terrain_type',terrain)
        v6_results.append(result)

        cr=result.dropna(subset=['contact_binary_pred']).copy()
        acc=float((cr['contact_binary_pred'].astype(int)==cr['contact_binary_truth'].astype(int)).mean())
        mag_mae=float(np.mean(np.abs(cr['force_mag_pred']-cr['force_mag_truth'])))
        v6_contact_rows.append(dict(
            terrain_type=terrain,environment=env,run=run,test_id=test_no,ttest_s=actual,
            binary_contact_accuracy=acc,force_magnitude_MAE_N=mag_mae
        ))

        print('\nV6 RESULT SAMPLE @ 0.1s')
        display(result[np.isclose(result['horizon_s'],0.1)][[
            'signal','baseline_pred','short_native_pred','long_direct_pred','prediction','ground_truth','abs_error'
        ]].head(20))
        print('binary contact accuracy across all horizons/feet:',acc)
        print('force magnitude MAE across all horizons/feet:',mag_mae,'N')

        outcsv=V6_OUT/f'{terrain}_{env}_{run}_test{test_no}_t{actual:.2f}.csv'
        result.to_csv(outcsv,index=False)
        del sm,lm,result; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


v6_all=pd.concat(v6_results,ignore_index=True)
v6_group=grouped_metrics_v6(v6_all)
v6_mean=v6_group.groupby(['signal_group','horizon_s'],as_index=False)[
    ['baseline_MAE','long_MAE','final_MAE']
].mean()
v6_contact=pd.DataFrame(v6_contact_rows)

v6_all.to_csv(V6_OUT/'V6_ALL_16_TESTS_ALL_SENSOR_VALUES.csv',index=False)
v6_group.to_csv(V6_OUT/'V6_ALL_16_TESTS_GROUPED_METRICS.csv',index=False)
v6_mean.to_csv(V6_OUT/'V6_MEAN_METRICS_ACROSS_16_TESTS.csv',index=False)
v6_contact.to_csv(V6_OUT/'V6_CONTACT_AND_FORCE_MAG_METRICS.csv',index=False)

print('\n'+'='*105)
print('V6 MEAN ACROSS ALL 16 TESTS')
print('='*105)
display(v6_mean)
print('\nMean binary contact accuracy:',v6_contact['binary_contact_accuracy'].mean())
print('Mean force magnitude MAE:',v6_contact['force_magnitude_MAE_N'].mean(),'N')

# ---------------------------------------------------------------------------
# 11) V5 -> V6 COMPARISON
# ---------------------------------------------------------------------------
_v5_path=Path('/content/v5_physics_contact_results/V5_MEAN_METRICS_ACROSS_16_TESTS.csv')
if _v5_path.exists():
    _v5=pd.read_csv(_v5_path).rename(columns={
        'long_MAE':'V5_long_MAE','final_MAE':'V5_final_MAE'
    })
    comp=v6_mean.merge(_v5[['signal_group','horizon_s','V5_long_MAE','V5_final_MAE']],
                       on=['signal_group','horizon_s'],how='left')
    comp['V6_vs_V5_final_percent_change']=100.0*(comp['final_MAE']-comp['V5_final_MAE'])/(comp['V5_final_MAE']+1e-12)
    comp['V6_long_vs_V5_long_percent_change']=100.0*(comp['long_MAE']-comp['V5_long_MAE'])/(comp['V5_long_MAE']+1e-12)
    comp.to_csv(V6_OUT/'V5_VS_V6_MEAN_COMPARISON.csv',index=False)
    print('\nV5 -> V6 comparison (negative % = V6 improved):')
    display(comp[comp['signal_group'].isin(['force','tau','contact','dq'])])

print('\nFILES:')
for _p in sorted(V6_OUT.glob('*')):
    if _p.is_file(): print(' ',_p)
print('\nDONE — V6 continuous-contact / physics-residual / clean-fusion experiment complete.')

# %% notebook cell 5

# ======================================================================
# V6.1 POST-PROCESS PATCH — NO RETRAINING
#
# You ALREADY ran the previous V6 cell.
#
# This patch changes ONLY the final expert selection:
#
#   +0.1 s  -> SHORT ONLY
#   +0.2...1.0 s -> LONG ONLY
#
# It DOES NOT retrain either Transformer.
# It rewrites the final prediction/error metrics from the already-saved V6 CSV.
#
# It also recomputes:
#   - signed / absolute errors
#   - grouped MAEs
#   - mean MAEs across all 16 tests
#   - binary contact accuracy
#   - force-magnitude MAE
#
# Original V6 outputs are preserved.
# New outputs are written with prefix V6_1_.
# ======================================================================

from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display

V6_OUT = Path("/content/v6_continuous_contact_physics_results")
SRC = V6_OUT / "V6_ALL_16_TESTS_ALL_SENSOR_VALUES.csv"

if not SRC.exists():
    raise FileNotFoundError(
        f"Could not find {SRC}\n"
        "The previous V6 experiment must finish first."
    )

df = pd.read_csv(SRC)

# ----------------------------------------------------------------------
# 1) HARD EXPERT SWITCH
# ----------------------------------------------------------------------
m_short = np.isclose(df["horizon_s"].to_numpy(float), 0.1)
m_long  = ~m_short

# +0.1 s = SHORT ONLY
if df.loc[m_short, "short_native_pred"].isna().any():
    bad = df.loc[m_short & df["short_native_pred"].isna(),
                 ["terrain_type","test_id","horizon_s","signal"]]
    raise RuntimeError(
        "Some +0.1 s rows are missing short_native_pred:\n"
        + bad.head(20).to_string(index=False)
    )

df.loc[m_short, "lambda_short"] = 1.0
df.loc[m_short, "lambda_long"] = 0.0
df.loc[m_short, "prediction"] = df.loc[m_short, "short_native_pred"]

# >= +0.2 s = LONG ONLY
df.loc[m_long, "lambda_short"] = 0.0
df.loc[m_long, "lambda_long"] = 1.0
df.loc[m_long, "prediction"] = df.loc[m_long, "long_direct_pred"]

# ----------------------------------------------------------------------
# 2) RECOMPUTE ERRORS
# ----------------------------------------------------------------------
df["signed_error"] = df["prediction"] - df["ground_truth"]
df["abs_error"] = df["signed_error"].abs()

# Existing long error remains valid, but recompute for safety.
df["long_abs_error"] = (
    df["long_direct_pred"] - df["ground_truth"]
).abs()

# ----------------------------------------------------------------------
# 3) RECOMPUTE CONTACT BINARY OUTPUT AT +0.1 FROM SHORT
# ----------------------------------------------------------------------
contact_mask = df["signal"].astype(str).str.contains(
    r"\.contact_probability$", regex=True
)

df.loc[contact_mask, "contact_binary_pred"] = (
    df.loc[contact_mask, "prediction"] >= 0.5
).astype(int)

df.loc[contact_mask, "contact_binary_truth"] = (
    df.loc[contact_mask, "ground_truth"] >= 0.5
).astype(int)

# ----------------------------------------------------------------------
# 4) RECOMPUTE FORCE MAGNITUDE FROM THE ACTUAL FINAL Fx,Fy,Fz
#
# At +0.1 this now means SHORT force magnitude.
# At >=0.2 this means LONG force magnitude.
# ----------------------------------------------------------------------

key_cols = [
    "terrain_type","environment","run","test_id","ttest_s","horizon_s"
]

# initialize only on contact rows; other rows can stay NaN
if "force_mag_pred" not in df.columns:
    df["force_mag_pred"] = np.nan
if "force_mag_truth" not in df.columns:
    df["force_mag_truth"] = np.nan

legs = ["FL","FR","RL","RR"]

for keys, g in df.groupby(key_cols, sort=False):
    idx_map = {str(s): i for i, s in zip(g.index, g["signal"])}
    for leg in legs:
        fx = idx_map.get(f"{leg}.Fx")
        fy = idx_map.get(f"{leg}.Fy")
        fz = idx_map.get(f"{leg}.Fz")
        cc = idx_map.get(f"{leg}.contact_probability")

        if None in (fx, fy, fz, cc):
            continue

        pred_vec = np.array([
            float(df.at[fx, "prediction"]),
            float(df.at[fy, "prediction"]),
            float(df.at[fz, "prediction"]),
        ], dtype=float)

        truth_vec = np.array([
            float(df.at[fx, "ground_truth"]),
            float(df.at[fy, "ground_truth"]),
            float(df.at[fz, "ground_truth"]),
        ], dtype=float)

        df.at[cc, "force_mag_pred"] = float(np.linalg.norm(pred_vec))
        df.at[cc, "force_mag_truth"] = float(np.linalg.norm(truth_vec))

# ----------------------------------------------------------------------
# 5) GROUPED METRICS
# ----------------------------------------------------------------------

def signal_group_v61(s):
    s = str(s)
    if ".q_" in s:
        return "q"
    if ".dq_" in s:
        return "dq"
    if ".tau_" in s:
        return "tau"
    if s.endswith(".Fx") or s.endswith(".Fy") or s.endswith(".Fz"):
        return "force"
    if "contact_probability" in s:
        return "contact"
    return "body"

metric_df = df.copy()
metric_df["signal_group"] = metric_df["signal"].map(signal_group_v61)

grouped = (
    metric_df.groupby(
        key_cols + ["signal_group"],
        as_index=False
    )
    .agg(
        baseline_MAE=("baseline_abs_error", "mean"),
        long_MAE=("long_abs_error", "mean"),
        final_MAE=("abs_error", "mean"),
    )
)

mean16 = (
    grouped.groupby(["signal_group","horizon_s"], as_index=False)[
        ["baseline_MAE","long_MAE","final_MAE"]
    ]
    .mean()
)

# ----------------------------------------------------------------------
# 6) CONTACT + FORCE-MAGNITUDE METRICS
# ----------------------------------------------------------------------

contact_rows = df[contact_mask].copy()

contact_summary = (
    contact_rows.groupby(
        ["terrain_type","environment","run","test_id","ttest_s"],
        as_index=False
    )
    .apply(
        lambda x: pd.Series({
            "binary_contact_accuracy":
                float(
                    (
                        x["contact_binary_pred"].astype(int)
                        == x["contact_binary_truth"].astype(int)
                    ).mean()
                ),
            "force_magnitude_MAE_N":
                float(
                    np.mean(
                        np.abs(
                            x["force_mag_pred"].to_numpy(float)
                            - x["force_mag_truth"].to_numpy(float)
                        )
                    )
                ),
        }),
        include_groups=False
    )
    .reset_index(drop=True)
)

# ----------------------------------------------------------------------
# 7) SAVE NEW V6.1 OUTPUTS
# ----------------------------------------------------------------------

OUT_ALL = V6_OUT / "V6_1_ALL_16_TESTS_ALL_SENSOR_VALUES.csv"
OUT_GROUP = V6_OUT / "V6_1_ALL_16_TESTS_GROUPED_METRICS.csv"
OUT_MEAN = V6_OUT / "V6_1_MEAN_METRICS_ACROSS_16_TESTS.csv"
OUT_CONTACT = V6_OUT / "V6_1_CONTACT_AND_FORCE_MAG_METRICS.csv"

df.to_csv(OUT_ALL, index=False)
grouped.to_csv(OUT_GROUP, index=False)
mean16.to_csv(OUT_MEAN, index=False)
contact_summary.to_csv(OUT_CONTACT, index=False)

# Save one V6.1 CSV per test too.
for keys, g in df.groupby(
    ["terrain_type","environment","run","test_id","ttest_s"],
    sort=False
):
    terrain, env, run, test_id, ttest = keys
    p = V6_OUT / (
        f"V6_1_{terrain}_{env}_{run}_"
        f"test{int(test_id)}_t{float(ttest):.2f}.csv"
    )
    g.to_csv(p, index=False)

# ----------------------------------------------------------------------
# 8) OPTIONAL V6 -> V6.1 COMPARISON
# ----------------------------------------------------------------------

old_mean_path = V6_OUT / "V6_MEAN_METRICS_ACROSS_16_TESTS.csv"

if old_mean_path.exists():
    old = pd.read_csv(old_mean_path).rename(
        columns={"final_MAE":"V6_old_final_MAE"}
    )
    comp = mean16.merge(
        old[["signal_group","horizon_s","V6_old_final_MAE"]],
        on=["signal_group","horizon_s"],
        how="left"
    )
    comp["V6_1_vs_old_V6_percent_change"] = (
        100.0
        * (comp["final_MAE"] - comp["V6_old_final_MAE"])
        / (comp["V6_old_final_MAE"] + 1e-12)
    )
    comp.to_csv(
        V6_OUT / "V6_1_VS_OLD_V6_MEAN_COMPARISON.csv",
        index=False
    )
else:
    comp = None

# ----------------------------------------------------------------------
# 9) PRINT CHECKS
# ----------------------------------------------------------------------

print("\n" + "="*100)
print("V6.1 HARD EXPERT SWITCH APPLIED — NO RETRAINING")
print("="*100)
print("+0.1 s      : SHORT ONLY")
print("+0.2..1.0 s : LONG ONLY")

# Sanity checks.
a = df[np.isclose(df["horizon_s"], 0.1)]
b = df[df["horizon_s"] >= 0.2 - 1e-9]

assert np.allclose(
    a["prediction"].to_numpy(float),
    a["short_native_pred"].to_numpy(float),
    equal_nan=False
)
assert np.allclose(
    b["prediction"].to_numpy(float),
    b["long_direct_pred"].to_numpy(float),
    equal_nan=False
)

print("\nSanity check passed:")
print("  all +0.1 s final predictions == short_native_pred")
print("  all >=+0.2 s final predictions == long_direct_pred")

print("\nMEAN ACROSS ALL 16 TESTS:")
display(mean16)

print("\nCONTACT/FORCE-MAGNITUDE SUMMARY:")
print("Mean binary contact accuracy:",
      contact_summary["binary_contact_accuracy"].mean())
print("Mean force magnitude MAE:",
      contact_summary["force_magnitude_MAE_N"].mean(), "N")

if comp is not None:
    print("\nOLD V6 -> V6.1 comparison")
    print("(negative % = V6.1 is better)")
    display(
        comp[comp["signal_group"].isin(
            ["force","tau","contact","dq","q","body"]
        )]
    )

print("\nFILES:")
print(OUT_ALL)
print(OUT_GROUP)
print(OUT_MEAN)
print(OUT_CONTACT)

print("\nDONE.")
