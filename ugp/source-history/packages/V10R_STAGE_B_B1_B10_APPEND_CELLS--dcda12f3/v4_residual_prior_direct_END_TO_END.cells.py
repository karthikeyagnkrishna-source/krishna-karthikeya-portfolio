# Source inspection export of the adjacent notebook.
# Notebook execution order and dependencies still apply.
# IPython magic, shell lines and prose-only conclusion cells are comments.

# %% notebook cell 0

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
