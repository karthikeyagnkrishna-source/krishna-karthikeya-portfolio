# Source inspection export of the adjacent notebook.
# Notebook execution order and dependencies still apply.
# IPython magic, shell lines and prose-only conclusion cells are comments.

# %% notebook cell 1

# CELL 1 — INSTALL + CONFIG
# IPYTHON: !pip -q install "rosbags>=0.10.0" boto3 botocore opencv-python-headless scipy scikit-learn tqdm

import os, sys, gc, math, json, time, random, re, warnings
from pathlib import Path
from dataclasses import dataclass, asdict, fields, is_dataclass
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
from scipy.spatial.transform import Rotation, Slerp
from sklearn.metrics import roc_auc_score, r2_score
from tqdm.auto import tqdm
from IPython.display import display

from google.colab import drive
drive.mount('/content/drive', force_remount=False)

SEED = 170926
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type != 'cuda':
    raise RuntimeError('Switch Colab to a GPU runtime before running this notebook.')

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---------- PUBLISHED FEEL-WM CONSTANTS ----------
DINO_DIM = 768
DINO_PATCHES = 256
CONTEXT_FRAMES = 4
PROP_WINDOW_S = 0.5
PROP_SAMPLES = 50
PROP_EMBED = 128
TX_LAYERS = 6
TX_HEADS = 8
TX_MLP_RATIO = 4
TX_DROPOUT = 0.1
ROLLOUT_H = 8
PHYS_SUPERVISION_S = 1.0
DT_CHOICES = (0.1, 0.2, 0.5, 1.0)

@dataclass
class FastConfig:
    # Official public TartanDrive2 object keys.
    train_key: str = 'bags/2023-11-14-14-24-21_gupta/gupta_0.bag'
    test_key: str = 'bags/2023-11-14-15-11-49_rough_rider_turnpike_loop/rough_rider_turnpike_loop_0.bag'

    # Only bounded sequences are processed from each downloaded bag.
    max_sequence_s: float = 60.0

    # Small diagnostic set; architecture remains unchanged.
    max_train_samples: int = 14
    max_val_samples: int = 3
    max_test_samples: int = 8
    origin_spacing_s: float = 2.0

    # Hard runtime limit for the expensive autoregressive world model.
    epochs: int = 2
    max_optimizer_steps: int = 8
    max_train_minutes: float = 13.0
    batch_size: int = 1
    lr: float = 6e-4
    weight_decay: float = 0.05
    grad_clip: float = 1.0

    # Paper loss structure. Physical targets are standardized, so no unreported
    # per-channel preference is introduced.
    lambda_prop: float = 0.5
    lambda_fail: float = 1.0
    visual_cosine_weight: float = 0.1

    raw_root: str = '/content/feelwm_td2_raw'
    output_root: str = '/content/drive/MyDrive/FeelWM_TD2_FAST'

CFG = FastConfig()
RAW_ROOT = Path(CFG.raw_root)
OUT_ROOT = Path(CFG.output_root)
CACHE_ROOT = OUT_ROOT/'cache'
for p in [RAW_ROOT,OUT_ROOT,CACHE_ROOT]: p.mkdir(parents=True,exist_ok=True)

T_ALL = time.time()
TIMING = {}

print(json.dumps(asdict(CFG), indent=2))
print('GPU:', torch.cuda.get_device_name(0))

# %% notebook cell 2

# CELL 2 — OFFICIAL TARTANDRIVE2 DOWNLOAD: TWO ROS BAGS ONLY
#
# This endpoint/bucket is the same unsigned-S3 path used by the official
# tartandrive_gui.py in castacks/tartan_drive_2.0.

import boto3
from botocore import UNSIGNED
from botocore.client import Config

TD2_ENDPOINT = "https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/AUTH_ac8533a83cff4d48bc8c608ad222d330"
TD2_BUCKET = "tartandrive2"
s3 = boto3.client("s3", endpoint_url=TD2_ENDPOINT, config=Config(signature_version=UNSIGNED))

def download_object(key, dst):
    dst = Path(dst)
    if dst.is_file() and dst.stat().st_size > 1024:
        print('reuse:', dst, f'({dst.stat().st_size/1024**3:.2f} GiB)')
        return dst

    dst.parent.mkdir(parents=True, exist_ok=True)
    head = s3.head_object(Bucket=TD2_BUCKET, Key=key)
    total = int(head['ContentLength'])
    print(f'downloading {key} -> {dst.name} ({total/1024**3:.2f} GiB)')
    resp = s3.get_object(Bucket=TD2_BUCKET, Key=key)
    with open(dst, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc=dst.name) as bar:
        for chunk in resp['Body'].iter_chunks(chunk_size=8*1024*1024):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
    if dst.stat().st_size != total:
        raise RuntimeError(f'incomplete download: {dst.stat().st_size} != {total}')
    return dst

t0=time.time()
TRAIN_BAG = download_object(CFG.train_key, RAW_ROOT/'train_gupta_0.bag')
TEST_BAG  = download_object(CFG.test_key,  RAW_ROOT/'test_rough_rider_0.bag')
TIMING['data_download_s'] = time.time()-t0

print(f'DATA DOWNLOAD = {TIMING["data_download_s"]/60:.2f} min')

# %% notebook cell 3

# CELL 3 — ROS BAG TOPIC AUDIT + ROBUST STREAM EXTRACTION
#
# We use explicit ROS-bag topics, not the ambiguous KITTI control/cmd columns.

from rosbags.highlevel import AnyReader

IMAGE_EXACT = [
    '/multisense/left/image_rect_color',
    '/multisense/left/image_rect_color/compressed',
]
IMU_EXACT = [
    '/multisense/imu/imu_data',
    '/multisense/imu/imu_data_raw',
    '/novatel/imu/data',
]
ODOM_EXACT = [
    '/odometry/filtered_odom',
    '/odometry/filtered',
]
WHEEL_EXACT = [
    '/wheel_rpm',
    '/racepak/wheel_rpm',
]
HAZARD_EXACT = [
    '/mux/intervention',
    '/intervention',
    '/human_intervention',
    '/hazard',
]

def choose_topic(topics, exact, keyword_groups, label):
    for x in exact:
        if x in topics: return x
    low={t:t.lower() for t in topics}
    for kws in keyword_groups:
        hits=[t for t,s in low.items() if all(k in s for k in kws)]
        if hits:
            hits.sort(key=len)
            return hits[0]
    raise RuntimeError(
        f'Could not identify required {label} topic.\n'
        f'Bag topics are:\n' + '\n'.join(sorted(topics))
    )

def flatten_numeric(obj, prefix=''):
    out=[]
    if obj is None: return out
    if isinstance(obj,(bool,np.bool_)):
        return [(prefix,float(obj))]
    if isinstance(obj,(int,float,np.integer,np.floating)):
        return [(prefix,float(obj))]
    if isinstance(obj,np.ndarray):
        if np.issubdtype(obj.dtype,np.number):
            for j,v in enumerate(obj.reshape(-1)):
                out.append((f'{prefix}[{j}]',float(v)))
        return out
    if isinstance(obj,(list,tuple)):
        for j,v in enumerate(obj):
            out.extend(flatten_numeric(v,f'{prefix}[{j}]'))
        return out
    if is_dataclass(obj):
        for f in fields(obj):
            name=f'{prefix}.{f.name}' if prefix else f.name
            out.extend(flatten_numeric(getattr(obj,f.name),name))
        return out
    slots=getattr(obj,'__slots__',[])
    for name in slots:
        if name.startswith('_'): continue
        full=f'{prefix}.{name}' if prefix else name
        try: out.extend(flatten_numeric(getattr(obj,name),full))
        except Exception: pass
    return out

def xyz(v):
    return np.array([float(v.x),float(v.y),float(v.z)],np.float64)

def quat_xyzw(q):
    return np.array([float(q.x),float(q.y),float(q.z),float(q.w)],np.float64)

def parse_imu(msg):
    return np.r_[xyz(msg.linear_acceleration), xyz(msg.angular_velocity), quat_xyzw(msg.orientation)]

def parse_odom(msg):
    p=xyz(msg.pose.pose.position)
    q=quat_xyzw(msg.pose.pose.orientation)
    lin=xyz(msg.twist.twist.linear)
    ang=xyz(msg.twist.twist.angular)
    return np.r_[p,q,lin,ang]

def select_wheel_values(msg):
    vals=flatten_numeric(msg)
    filtered=[]
    for name,v in vals:
        n=name.lower()
        if any(k in n for k in ['stamp','secs','nsecs','seq']):
            continue
        if any(k in n for k in ['rpm','wheel','fl','fr','rl','rr']):
            filtered.append((name,v))
    if len(filtered)<4:
        filtered=[x for x in vals if not any(k in x[0].lower() for k in ['stamp','secs','nsecs','seq'])]
    if len(filtered)<4:
        raise RuntimeError('Wheel message did not expose >=4 numeric wheel/RPM values: '+str(vals[:30]))
    return np.array([v for _,v in filtered[:4]],np.float64)

def select_hazard_value(msg):
    vals=flatten_numeric(msg)
    vals=[x for x in vals if not any(k in x[0].lower() for k in ['stamp','secs','nsecs','seq'])]
    if not vals:
        raise RuntimeError('Hazard/intervention message contains no numeric/bool value.')
    preferred=[
        x for x in vals
        if any(k in x[0].lower() for k in ['data','intervention','hazard','value','active'])
    ]
    return float((preferred or vals)[0][1])

def inspect_and_extract(bag_path, max_sequence_s):
    bag_path=Path(bag_path)
    data={k:[] for k in ['imu_t','imu','odom_t','odom','wheel_t','wheel','haz_t','haz']}
    cam_t=[]

    with AnyReader([bag_path]) as reader:
        topics=sorted({c.topic for c in reader.connections})
        image_topic=choose_topic(topics,IMAGE_EXACT,[('multisense','image'),('image','rect','color')],'RGB image')
        imu_topic=choose_topic(topics,IMU_EXACT,[('imu','data'),('imu',)],'IMU')
        odom_topic=choose_topic(topics,ODOM_EXACT,[('odometry','filtered'),('odom',)],'odometry')
        wheel_topic=choose_topic(topics,WHEEL_EXACT,[('wheel','rpm'),('rpm',)],'wheel RPM')
        hazard_topic=choose_topic(
            topics,HAZARD_EXACT,
            [('intervention',),('hazard',),('mux','intervention')],
            'recorded hazard/intervention'
        )

        chosen=dict(image=image_topic,imu=imu_topic,odom=odom_topic,wheel=wheel_topic,hazard=hazard_topic)
        print('chosen topics:',json.dumps(chosen,indent=2))

        cmap={c.topic:c for c in reader.connections}
        conns=[cmap[x] for x in chosen.values()]

        t_first=None
        for conn,ts,raw in reader.messages(connections=conns):
            t=ts*1e-9
            if t_first is None: t_first=t
            if t-t_first > max_sequence_s+10.0:
                break

            try:
                if conn.topic==image_topic:
                    cam_t.append(t)
                    continue
                msg=reader.deserialize(raw,conn.msgtype)
                if conn.topic==imu_topic:
                    data['imu_t'].append(t); data['imu'].append(parse_imu(msg))
                elif conn.topic==odom_topic:
                    data['odom_t'].append(t); data['odom'].append(parse_odom(msg))
                elif conn.topic==wheel_topic:
                    data['wheel_t'].append(t); data['wheel'].append(select_wheel_values(msg))
                elif conn.topic==hazard_topic:
                    data['haz_t'].append(t); data['haz'].append(select_hazard_value(msg))
            except Exception as exc:
                # One malformed message must not hide the rest of the recorded stream.
                if len(data.get('parse_errors',[]))<5:
                    data.setdefault('parse_errors',[]).append((conn.topic,type(exc).__name__,str(exc)[:200]))

    for k in ['imu_t','odom_t','wheel_t','haz_t']:
        data[k]=np.asarray(data[k],np.float64)
    for k in ['imu','odom','wheel','haz']:
        data[k]=np.asarray(data[k],np.float64)
    data['cam_t']=np.asarray(cam_t,np.float64)
    data['topics']=chosen

    for name,tkey,vkey in [
        ('IMU','imu_t','imu'),('odom','odom_t','odom'),('wheel','wheel_t','wheel'),('hazard','haz_t','haz')
    ]:
        if len(data[tkey])<10:
            raise RuntimeError(f'{name} extraction yielded only {len(data[tkey])} samples.')
        print(name,len(data[tkey]),data[vkey].shape,
              'span',round(float(data[tkey][-1]-data[tkey][0]),2),'s')
    if len(data['cam_t'])<8:
        raise RuntimeError('RGB stream has too few camera frames.')
    if data.get('parse_errors'):
        print('first parse warnings:',data['parse_errors'])
    return data

t0=time.time()
TRAIN_RAW=inspect_and_extract(TRAIN_BAG,CFG.max_sequence_s)
TEST_RAW=inspect_and_extract(TEST_BAG,CFG.max_sequence_s)
TIMING['bag_parse_s']=time.time()-t0
print(f'BAG PARSE = {TIMING["bag_parse_s"]/60:.2f} min')

# %% notebook cell 4

# CELL 4 — 100 Hz ALIGNMENT + PAPER PHYSICAL TARGETS
#
# Proprio input categories match the paper text:
# inertial acceleration, angular rates, body velocities, attitude, actuator/wheel readings.
#
# p_raw = [accel3, gyro3, body_linear_velocity3, roll/pitch/yaw3, wheel_rpm4] = 16 dims

GRID_HZ=100
GRID_DT=1.0/GRID_HZ
HALF_TARGET=int(round(0.25*GRID_HZ))

def interp_cols(t_src,x_src,tq):
    t_src=np.asarray(t_src,float); x_src=np.asarray(x_src,float)
    return np.stack([np.interp(tq,t_src,x_src[:,j]) for j in range(x_src.shape[1])],1)

def slerp_quat(t_src,q_src,tq):
    # Remove duplicate times before Slerp.
    keep=np.r_[True,np.diff(t_src)>1e-9]
    t=np.asarray(t_src)[keep]; q=np.asarray(q_src)[keep]
    rot=Rotation.from_quat(q)
    return Slerp(t,rot)(tq).as_quat()

def binarize_recorded_hazard(h):
    h=np.asarray(h,float)
    rounded=np.unique(np.round(h[np.isfinite(h)],6))
    if len(rounded)<=3 and np.all(np.isin(rounded,[0.,1.])):
        return (h>0.5).astype(np.float32)
    raise RuntimeError(
        'The selected hazard/intervention stream is continuous rather than bool/0-1. '
        'The Feel-WM paper states a fixed hazard threshold but does not publish a numeric '
        'threshold in the accessible implementation description. This notebook will not invent one. '
        f'Observed values include: {rounded[:20]}'
    )

def build_aligned(raw,max_s):
    start=max(raw['imu_t'][0],raw['odom_t'][0],raw['wheel_t'][0],raw['haz_t'][0],raw['cam_t'][0])
    end=min(raw['imu_t'][-1],raw['odom_t'][-1],raw['wheel_t'][-1],raw['haz_t'][-1],raw['cam_t'][-1],start+max_s)
    if end-start<12:
        raise RuntimeError(f'overlapping usable interval is only {end-start:.2f}s')
    grid=np.arange(start,end,GRID_DT,dtype=np.float64)

    imu=interp_cols(raw['imu_t'],raw['imu'],grid)  # acc3 gyro3 quat4
    odom=interp_cols(raw['odom_t'],raw['odom'],grid) # p3 q4 lin3 ang3
    wheel=interp_cols(raw['wheel_t'],raw['wheel'],grid)

    q_odom=slerp_quat(raw['odom_t'],raw['odom'][:,3:7],grid)
    euler=Rotation.from_quat(q_odom).as_euler('xyz',degrees=False)
    odom[:,3:7]=q_odom

    # Hazard is held/nearest; it is a recorded discrete intervention label.
    hbin=binarize_recorded_hazard(raw['haz'])
    idx=np.searchsorted(raw['haz_t'],grid,side='right')-1
    idx=np.clip(idx,0,len(hbin)-1)
    hazard=hbin[idx]

    prop_raw=np.c_[imu[:,:3],imu[:,3:6],odom[:,7:10],euler,wheel].astype(np.float32)
    ground_speed=np.linalg.norm(odom[:,7:9],axis=1).astype(np.float32)

    # Camera times restricted to this aligned clock interval.
    cam=raw['cam_t']
    cam=cam[(cam>=grid[0])&(cam<=grid[-1])]

    return dict(
        t=grid,imu=imu.astype(np.float32),odom=odom.astype(np.float32),
        wheel=wheel.astype(np.float32),hazard=hazard.astype(np.float32),
        prop_raw=prop_raw,ground_speed=ground_speed,cam_t=cam,
        topics=raw['topics']
    )

TRAIN=build_aligned(TRAIN_RAW,CFG.max_sequence_s)
TEST=build_aligned(TEST_RAW,CFG.max_sequence_s)

# The public bag stream is wheel RPM while the paper defines slip with encoder-derived
# wheel SPEED. The paper does not publish its wheel radius/calibration constant.
# Estimate one scalar using TRAINING DATA ONLY; no test information.
rpm_train=np.mean(np.abs(TRAIN['wheel']),axis=1)
mask=(TRAIN['ground_speed']>0.5)&(rpm_train>5)&np.isfinite(rpm_train)
if mask.sum()<50:
    raise RuntimeError('Not enough moving training samples to calibrate RPM -> wheel speed.')
RPM_TO_MPS=float(np.median(TRAIN['ground_speed'][mask]/rpm_train[mask]))
print('training-only RPM_TO_MPS =',RPM_TO_MPS)

def window_targets(seq):
    n=len(seq['t'])
    y=np.full((n,4),np.nan,np.float32)
    gyro=seq['imu'][:,3:6]
    az=seq['imu'][:,2]
    pitch=Rotation.from_quat(seq['odom'][:,3:7]).as_euler('xyz')[:,1]
    vg=seq['ground_speed']
    vw=np.mean(np.abs(seq['wheel']),axis=1)*RPM_TO_MPS

    for i in range(HALF_TARGET,n-HALF_TARGET):
        sl=slice(i-HALF_TARGET,i+HALF_TARGET)
        rough=float(np.std(az[sl]))
        angular=float(np.var(gyro[sl,0])+np.var(gyro[sl,1]))
        vwm=float(np.mean(vw[sl])); vgm=float(np.mean(vg[sl]))
        slip=float((vwm-vgm)/max(vwm,vgm,1e-3))
        slope=float(np.mean(vg[sl]*np.sin(pitch[sl])))
        y[i]=[rough,angular,slip,slope]
    return y

TRAIN['phys']=window_targets(TRAIN)
TEST['phys']=window_targets(TEST)

PHYS_NAMES=['roughness','angular_instability','wheel_slip','slope_speed']
display(pd.DataFrame(TRAIN['phys'],columns=PHYS_NAMES).describe())

# %% notebook cell 5

# CELL 5 — BUILD SMALL TRAIN/VAL/TEST SAMPLES + FIX Δt PER SAMPLE
#
# Each sample has one Δt for all H=8 rollout steps, exactly as in the paper.

def wrap_angle(a):
    return (a+np.pi)%(2*np.pi)-np.pi

def action_at(seq,i0,i1):
    p0=seq['odom'][i0,:3]; p1=seq['odom'][i1,:3]
    yaw0=Rotation.from_quat(seq['odom'][i0,3:7]).as_euler('xyz')[2]
    yaw1=Rotation.from_quat(seq['odom'][i1,3:7]).as_euler('xyz')[2]
    c,s=np.cos(yaw0),np.sin(yaw0)
    dp=p1-p0
    dx=c*dp[0]+s*dp[1]
    dy=-s*dp[0]+c*dp[1]
    return np.array([dx,dy,wrap_angle(yaw1-yaw0),float(seq['t'][i1]-seq['t'][i0])],np.float32)

def camera_index_before(seq,t):
    j=int(np.searchsorted(seq['cam_t'],float(t),side='right')-1)
    return j

def nearest_camera_index(seq,t):
    ct=seq['cam_t']; j=int(np.searchsorted(ct,float(t)))
    if j<=0: return 0
    if j>=len(ct): return len(ct)-1
    return j-1 if abs(ct[j-1]-t)<=abs(ct[j]-t) else j

def build_records(seq,max_count,start_fraction,end_fraction,dt_cycle):
    n=len(seq['t'])
    lo=max(PROP_SAMPLES-1,HALF_TARGET+1,int(start_fraction*n))
    hi=min(n-HALF_TARGET-2,int(end_fraction*n))
    spacing=max(1,int(round(CFG.origin_spacing_s*GRID_HZ)))
    cand=np.arange(lo,hi,spacing,dtype=int)
    rec=[]
    ci=0
    for i in cand:
        dt=float(dt_cycle[ci%len(dt_cycle)]); ci+=1
        step=int(round(dt*GRID_HZ))
        jlast=i+ROLLOUT_H*step
        if jlast+HALF_TARGET>=n: continue
        vi=camera_index_before(seq,seq['t'][i])
        if vi<3: continue
        if np.isnan(seq['phys'][jlast]).any(): continue
        target_cams=[nearest_camera_index(seq,seq['t'][i+(h+1)*step]) for h in range(ROLLOUT_H)]
        rec.append(dict(origin=int(i),dt=dt,step=step,ctx=list(range(vi-3,vi+1)),targets=target_cams))
        if len(rec)>=max_count: break
    return rec

# Chronological separation inside train bag; rough-rider bag is fully held out.
TRAIN_REC=build_records(TRAIN,CFG.max_train_samples,0.08,0.65,DT_CHOICES)
VAL_REC=build_records(TRAIN,CFG.max_val_samples,0.72,0.90,DT_CHOICES[::-1])
TEST_REC=build_records(TEST,CFG.max_test_samples,0.08,0.85,DT_CHOICES)

if len(TRAIN_REC)<4 or len(VAL_REC)<1 or len(TEST_REC)<1:
    raise RuntimeError(f'not enough valid records train={len(TRAIN_REC)} val={len(VAL_REC)} test={len(TEST_REC)}')

print('train samples:',len(TRAIN_REC),[r['dt'] for r in TRAIN_REC])
print('val samples:',len(VAL_REC),[r['dt'] for r in VAL_REC])
print('test samples:',len(TEST_REC),[r['dt'] for r in TEST_REC])

def needed_cam_indices(records):
    s=set()
    for r in records:
        s.update(r['ctx']); s.update(r['targets'])
    return sorted(s)

TRAIN_CAM_NEED=needed_cam_indices(TRAIN_REC+VAL_REC)
TEST_CAM_NEED=needed_cam_indices(TEST_REC)
print('DINO frames needed:',len(TRAIN_CAM_NEED),'train-bag,',len(TEST_CAM_NEED),'test-bag')

# %% notebook cell 6

# CELL 6 — EXTRACT EXACT NEEDED RGB FRAMES FROM BAG AND RUN FROZEN DINOV2 ViT-B/14
#
# Standard DINO image normalization is used. DINO is frozen.
# Output assertion is exactly 256 x 768 patch tokens.

import cv2
from PIL import Image
import torchvision.transforms as T

def decode_ros_image(msg):
    data=np.asarray(msg.data,dtype=np.uint8).tobytes() if not isinstance(msg.data,(bytes,bytearray)) else bytes(msg.data)

    name=type(msg).__name__.lower()
    if 'compressed' in name or hasattr(msg,'format'):
        im=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
        if im is None: raise ValueError('cv2.imdecode failed')
        return cv2.cvtColor(im,cv2.COLOR_BGR2RGB)

    h=int(msg.height); w=int(msg.width); enc=str(msg.encoding).lower()
    arr=np.frombuffer(data,np.uint8)
    step=int(getattr(msg,'step',0))
    if enc in ('rgb8','bgr8'):
        rowbytes=step if step else w*3
        arr=arr.reshape(h,rowbytes)[:,:w*3].reshape(h,w,3)
        return arr if enc=='rgb8' else cv2.cvtColor(arr,cv2.COLOR_BGR2RGB)
    if enc in ('rgba8','bgra8'):
        rowbytes=step if step else w*4
        arr=arr.reshape(h,rowbytes)[:,:w*4].reshape(h,w,4)
        code=cv2.COLOR_RGBA2RGB if enc=='rgba8' else cv2.COLOR_BGRA2RGB
        return cv2.cvtColor(arr,code)
    if enc in ('mono8','8uc1'):
        arr=arr.reshape(h,-1)[:,:w]
        return np.repeat(arr[...,None],3,-1)
    raise RuntimeError(f'unsupported image encoding: {enc}')

DINO_TFM=T.Compose([
    T.Resize(256,interpolation=T.InterpolationMode.BICUBIC,antialias=True),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225)),
])

def extract_needed_images(bag_path,image_topic,needed_indices):
    needed=set(map(int,needed_indices)); got={}
    with AnyReader([Path(bag_path)]) as reader:
        conn=next(c for c in reader.connections if c.topic==image_topic)
        k=0
        for c,ts,raw in reader.messages(connections=[conn]):
            if k in needed:
                msg=reader.deserialize(raw,c.msgtype)
                got[k]=decode_ros_image(msg)
                if len(got)==len(needed): break
            k+=1
    miss=needed-set(got)
    if miss: raise RuntimeError(f'missing selected image indices: {sorted(miss)[:20]}')
    return got

t0=time.time()
TRAIN_IMAGES=extract_needed_images(TRAIN_BAG,TRAIN_RAW['topics']['image'],TRAIN_CAM_NEED)
TEST_IMAGES=extract_needed_images(TEST_BAG,TEST_RAW['topics']['image'],TEST_CAM_NEED)

dino=torch.hub.load('facebookresearch/dinov2','dinov2_vitb14').to(DEVICE).eval()
for p in dino.parameters(): p.requires_grad_(False)

@torch.no_grad()
def make_dino_cache(images,cache_file,batch=8):
    cache_file=Path(cache_file)
    wanted=sorted(images)
    old={}
    if cache_file.is_file():
        try:
            with np.load(cache_file,allow_pickle=False) as z:
                ids=z['ids'].astype(int); tok=z['tokens']
            old={int(i):tok[j] for j,i in enumerate(ids)}
        except Exception: old={}
    todo=[i for i in wanted if i not in old]
    for a in tqdm(range(0,len(todo),batch),desc='DINO batches'):
        ids=todo[a:a+batch]
        x=torch.stack([DINO_TFM(Image.fromarray(images[i])) for i in ids]).to(DEVICE)
        with torch.amp.autocast('cuda',dtype=torch.float16):
            f=dino.forward_features(x)['x_norm_patchtokens']
        if f.shape[1:]!=(256,768):
            raise AssertionError(f'expected 256x768, got {tuple(f.shape)}')
        f=f.float().cpu().numpy().astype(np.float16)
        for j,i in enumerate(ids): old[int(i)]=f[j]
    ids=np.array(sorted(old),np.int32)
    tok=np.stack([old[int(i)] for i in ids])
    np.savez_compressed(cache_file,ids=ids,tokens=tok)
    return {int(i):np.asarray(old[int(i)],np.float32) for i in wanted}

TRAIN_DINO=make_dino_cache(TRAIN_IMAGES,CACHE_ROOT/'train_dino_vitb14.npz')
TEST_DINO=make_dino_cache(TEST_IMAGES,CACHE_ROOT/'test_dino_vitb14.npz')

del dino,TRAIN_IMAGES,TEST_IMAGES
gc.collect(); torch.cuda.empty_cache()

TIMING['dino_s']=time.time()-t0
print(f'RGB read + frozen DINO = {TIMING["dino_s"]/60:.2f} min')

# %% notebook cell 7

# CELL 7 — TRAIN-ONLY NORMALIZATION + DATASET

# Proprio input normalization: training sequence only.
PROP_MEAN=TRAIN['prop_raw'].mean(0).astype(np.float32)
PROP_STD=TRAIN['prop_raw'].std(0).astype(np.float32)
PROP_STD[PROP_STD<1e-6]=1.0

# Physical target normalization: only finite train targets.
finite=np.isfinite(TRAIN['phys']).all(1)
PHYS_MEAN=TRAIN['phys'][finite].mean(0).astype(np.float32)
PHYS_STD=TRAIN['phys'][finite].std(0).astype(np.float32)
PHYS_STD[PHYS_STD<1e-6]=1.0

# Action normalization from actual training records only.
train_actions=[]
for r in TRAIN_REC:
    i=r['origin']; s=r['step']
    for h in range(ROLLOUT_H):
        train_actions.append(action_at(TRAIN,i+h*s,i+(h+1)*s))
train_actions=np.stack(train_actions)
ACTION_MEAN=train_actions.mean(0).astype(np.float32)
ACTION_STD=train_actions.std(0).astype(np.float32)
ACTION_STD[ACTION_STD<1e-6]=1.0

np.savez_compressed(
    OUT_ROOT/'normalization.npz',
    PROP_MEAN=PROP_MEAN,PROP_STD=PROP_STD,
    PHYS_MEAN=PHYS_MEAN,PHYS_STD=PHYS_STD,
    ACTION_MEAN=ACTION_MEAN,ACTION_STD=ACTION_STD,
    RPM_TO_MPS=np.array(RPM_TO_MPS,np.float32),
)

def nprop(x): return ((x-PROP_MEAN)/PROP_STD).astype(np.float32)
def nphys(x): return ((x-PHYS_MEAN)/PHYS_STD).astype(np.float32)
def naction(x): return ((x-ACTION_MEAN)/ACTION_STD).astype(np.float32)

class FeelDataset(Dataset):
    def __init__(self,seq,records,dino):
        self.seq=seq; self.records=records; self.dino=dino
    def __len__(self): return len(self.records)
    def __getitem__(self,k):
        r=self.records[k]; i=r['origin']; s=r['step']
        p0=nprop(self.seq['prop_raw'][i-PROP_SAMPLES+1:i+1])
        ctx=np.stack([self.dino[j] for j in r['ctx']]).astype(np.float32)

        actions=[]; yvis=[]; yphys=[]; yfail=[]; phist=[]
        for h in range(ROLLOUT_H):
            j=i+(h+1)*s
            actions.append(naction(action_at(self.seq,i+h*s,j)))
            yvis.append(self.dino[r['targets'][h]])
            yphys.append(nphys(self.seq['phys'][j]))
            yfail.append(self.seq['hazard'][j])
            phist.append(nprop(self.seq['prop_raw'][j-PROP_SAMPLES+1:j+1]))

        return dict(
            origin=np.int64(i),dt=np.float32(r['dt']),
            prop=p0,visual=ctx,actions=np.stack(actions),
            target_visual=np.stack(yvis).astype(np.float32),
            target_phys=np.stack(yphys).astype(np.float32),
            target_fail=np.asarray(yfail,np.float32),
            future_prop_hist=np.stack(phist).astype(np.float32),
        )

train_ds=FeelDataset(TRAIN,TRAIN_REC,TRAIN_DINO)
val_ds=FeelDataset(TRAIN,VAL_REC,TRAIN_DINO)
test_ds=FeelDataset(TEST,TEST_REC,TEST_DINO)

train_loader=DataLoader(train_ds,batch_size=CFG.batch_size,shuffle=True,num_workers=0,pin_memory=True)
val_loader=DataLoader(val_ds,batch_size=1,shuffle=False,num_workers=0,pin_memory=True)
test_loader=DataLoader(test_ds,batch_size=1,shuffle=False,num_workers=0,pin_memory=True)

b=next(iter(train_loader))
for k,v in b.items():
    if torch.is_tensor(v): print(k,tuple(v.shape))

# %% notebook cell 8

# CELL 8 — FEEL-WM MODEL
#
# Paper-specified:
#   DINO 256x768, p_t 128, 6 Tx layers, 8 heads, MLP ratio 4, dropout .1.
#
# The paper does not publish layer-by-layer widths for the "small 1-D CNN",
# action encoder, or FiLM generator. The definitions below are minimal
# reproducible implementations of those published interfaces; they are not
# claimed to be unreleased author code.

class ProprioCNN(nn.Module):
    def __init__(self,in_ch=16):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv1d(in_ch,64,5,2,2),nn.GroupNorm(8,64),nn.GELU(),
            nn.Conv1d(64,128,5,2,2),nn.GroupNorm(8,128),nn.GELU(),
            nn.Conv1d(128,128,3,1,1),nn.GroupNorm(8,128),nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.norm=nn.LayerNorm(128)
    def forward(self,x):
        return self.norm(self.net(x.transpose(1,2)).squeeze(-1))

class FiLMGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk=nn.Sequential(nn.Linear(128,256),nn.GELU(),nn.LayerNorm(256))
        self.gamma=nn.Linear(256,768)
        self.beta=nn.Linear(256,768)
        nn.init.zeros_(self.gamma.weight); nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight); nn.init.zeros_(self.beta.bias)
    def forward(self,z,c):
        r=self.trunk(c)
        dg=self.gamma(r)[:,None,:]; b=self.beta(r)[:,None,:]
        return z*(1.0+dg)+b

class DynamicsTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embed=nn.Parameter(torch.randn(4,768)*.02)
        self.pos_embed=nn.Parameter(torch.randn(256,768)*.02)
        self.layers=nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=768,nhead=8,dim_feedforward=3072,
                dropout=.1,activation='gelu',batch_first=True,norm_first=True
            ) for _ in range(6)
        ])
        self.norm=nn.LayerNorm(768)

        time_ids=np.repeat(np.arange(4),256)
        # token i may attend tokens from the same or an earlier frame only
        mask=time_ids[None,:] > time_ids[:,None]
        self.register_buffer('mask',torch.as_tensor(mask,dtype=torch.bool),persistent=False)

    def forward(self,x):
        B=x.shape[0]
        x=x.reshape(B,4,256,768)
        x=x+self.time_embed[None,:,None,:]+self.pos_embed[None,None,:,:]
        x=x.reshape(B,1024,768)
        for layer in self.layers:
            if self.training:
                x=checkpoint(lambda y,layer=layer: layer(y,src_mask=self.mask),x,use_reentrant=False)
            else:
                x=layer(x,src_mask=self.mask)
        return self.norm(x)

class FeelWM(nn.Module):
    def __init__(self):
        super().__init__()
        self.prop_encoder=ProprioCNN(16)
        self.action_encoder=nn.Sequential(
            nn.Linear(4,128),nn.GELU(),nn.Linear(128,128),nn.LayerNorm(128)
        )
        self.film=FiLMGenerator()
        self.tx=DynamicsTransformer()

        self.visual_head=nn.Linear(768,768)
        self.prop_head=nn.Sequential(nn.Linear(768,256),nn.GELU(),nn.Linear(256,4))
        self.failure_head=nn.Sequential(nn.Linear(768,256),nn.GELU(),nn.Linear(256,1))
        self.prop_feedback=nn.Sequential(nn.Linear(4,128),nn.GELU(),nn.LayerNorm(128))

    def encode_prop(self,x):
        return self.prop_encoder(x)

    def step(self,visual_ctx,pcond,action):
        # action (+) p_t exactly at the conditioning interface
        cond=self.action_encoder(action)+pcond

        ctx=visual_ctx.clone()
        ctx[:,3]=self.film(ctx[:,3],cond)

        h=self.tx(ctx)
        hcur=h[:,3*256:4*256]
        pooled=hcur.mean(1)

        znext=self.visual_head(hcur)
        pnext=self.prop_head(pooled)
        cnext=self.failure_head(pooled).squeeze(-1)
        return dict(visual=znext,phys=pnext,fail_logit=cnext)

model=FeelWM().to(DEVICE)
NPAR=sum(p.numel() for p in model.parameters())
print(f'Feel-WM parameters in trainable world-model implementation: {NPAR/1e6:.2f} M')
print('Transformer: 6 layers | 8 heads | 768 width | 3072 FFN | dropout 0.1')

# %% notebook cell 9

# CELL 9 — H=8 AUTOREGRESSIVE TRAINING WITH PHYSICAL LOSS MASK H_p=1s
#
# One teacher-forcing decision is made for the whole rollout/sample.
# With the deliberately short two-epoch diagnostic run:
#   epoch 1 -> teacher forcing
#   epoch 2 -> 50% teacher forcing
# This preserves the paper's rollout-level scheduled-sampling mechanism while
# keeping the diagnostic run bounded.

# Recorded failure imbalance -> class-weighted BCE, as described by the paper.
train_fail=[]
for r in TRAIN_REC:
    i=r['origin']; s=r['step']
    train_fail.extend([TRAIN['hazard'][i+(h+1)*s] for h in range(ROLLOUT_H)
                       if (h+1)*r['dt']<=PHYS_SUPERVISION_S+1e-9])
train_fail=np.asarray(train_fail)
npos=float(train_fail.sum()); nneg=float(len(train_fail)-npos)
POS_WEIGHT=torch.tensor(nneg/max(npos,1.0),device=DEVICE)
print('failure positives:',int(npos),'/',len(train_fail),'pos_weight=',float(POS_WEIGHT))

def move(batch):
    return {k:(v.to(DEVICE,non_blocking=True) if torch.is_tensor(v) else v) for k,v in batch.items()}

def visual_loss(pred,target):
    mse=F.mse_loss(pred,target)
    cos=1.0-F.cosine_similarity(pred,target,dim=-1).mean()
    return mse+CFG.visual_cosine_weight*cos

def tf_prob(epoch):
    return 1.0 if epoch==1 else 0.5

def rollout_loss(batch,epoch,training=True,do_backward=False):
    b=move(batch)
    pcond=model.encode_prop(b['prop'].float())
    ctx=b['visual'].float()
    B=ctx.shape[0]

    tfmask=(torch.rand(B,device=DEVICE)<tf_prob(epoch)) if training else \
           torch.zeros(B,dtype=torch.bool,device=DEVICE)

    total=0.0
    components=dict(vis=0.,prop=0.,fail=0.)
    if do_backward: optimizer.zero_grad(set_to_none=True)

    for h in range(ROLLOUT_H):
        with torch.amp.autocast('cuda',dtype=torch.float16):
            o=model.step(ctx,pcond,b['actions'][:,h].float())
            Lv=visual_loss(o['visual'],b['target_visual'][:,h].float())

            phys_on=((h+1)*b['dt']<=PHYS_SUPERVISION_S+1e-6).float()
            if phys_on.sum()>0:
                per_prop=F.mse_loss(
                    o['phys'],b['target_phys'][:,h].float(),reduction='none'
                ).mean(-1)
                Lp=(per_prop*phys_on).sum()/phys_on.sum()

                per_fail=F.binary_cross_entropy_with_logits(
                    o['fail_logit'],b['target_fail'][:,h].float(),
                    pos_weight=POS_WEIGHT,reduction='none'
                )
                Lc=(per_fail*phys_on).sum()/phys_on.sum()
            else:
                Lp=o['phys'].sum()*0
                Lc=o['fail_logit'].sum()*0

            loss=(Lv+CFG.lambda_prop*Lp+CFG.lambda_fail*Lc)/ROLLOUT_H

        total+=float(loss.detach())
        components['vis']+=float(Lv.detach())/ROLLOUT_H
        components['prop']+=float(Lp.detach())/ROLLOUT_H
        components['fail']+=float(Lc.detach())/ROLLOUT_H

        if do_backward:
            scaler.scale(loss).backward()

        if h<ROLLOUT_H-1:
            # Ground-truth next p_t for teacher forcing comes from the future
            # 50-sample high-rate window. In autoregressive mode, predicted
            # physical state is projected back into the 128-D conditioning input.
            with torch.no_grad():
                ptrue=model.encode_prop(b['future_prop_hist'][:,h].float())
            pred_pcond=model.prop_feedback(o['phys'])
            m=tfmask[:,None,None]
            next_vis=torch.where(m,b['target_visual'][:,h].float(),o['visual'].detach())
            ctx=torch.cat([ctx[:,1:],next_vis[:,None]],1)
            pcond=torch.where(tfmask[:,None],ptrue,pred_pcond.detach())

    if do_backward:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(),CFG.grad_clip)
        scaler.step(optimizer); scaler.update()

    return total,components

optimizer=torch.optim.AdamW(model.parameters(),lr=CFG.lr,weight_decay=CFG.weight_decay)
scaler=torch.amp.GradScaler('cuda',enabled=True)

@torch.no_grad()
def val_loss(epoch):
    model.eval(); vals=[]
    for b in val_loader:
        x,_=rollout_loss(b,epoch,training=False,do_backward=False)
        vals.append(x)
    return float(np.mean(vals))

t0=time.time()
history=[]; best=np.inf; best_state=None; steps=0
stop=False

for epoch in range(1,CFG.epochs+1):
    model.train()
    vals=[]
    for batch in train_loader:
        if steps>=CFG.max_optimizer_steps or (time.time()-t0)/60>=CFG.max_train_minutes:
            stop=True; break
        loss,parts=rollout_loss(batch,epoch,training=True,do_backward=True)
        steps+=1; vals.append(loss)
        print(
            f'epoch={epoch} step={steps} loss={loss:.4f} '
            f'vis={parts["vis"]:.3f} prop={parts["prop"]:.3f} fail={parts["fail"]:.3f} '
            f'elapsed={(time.time()-t0)/60:.2f} min'
        )

    vl=val_loss(epoch)
    rec=dict(epoch=epoch,steps=steps,train=float(np.mean(vals)) if vals else np.nan,
             val=vl,elapsed_min=(time.time()-t0)/60)
    history.append(rec); print(rec)

    if np.isfinite(vl) and vl<best:
        best=vl
        best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if stop: break

if best_state is None:
    best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

model.load_state_dict(best_state)
torch.save(
    {'model':model.state_dict(),'config':asdict(CFG),'best_val':best,
     'paper_architecture':dict(dino=[256,768],context=4,prop_window_samples=50,
                               prop_embed=128,layers=6,heads=8,mlp_ratio=4,
                               dropout=.1,H=8,Hp_s=1.0,dt_choices=DT_CHOICES)},
    OUT_ROOT/'feelwm_td2_fast.pt'
)
pd.DataFrame(history).to_csv(OUT_ROOT/'training_history.csv',index=False)

TIMING['training_s']=time.time()-t0
print(f'TRAINING TIME = {TIMING["training_s"]/60:.2f} min')

# %% notebook cell 10

# CELL 10 — FULLY AUTOREGRESSIVE HELD-OUT ROUGH-RIDER TEST

def unphys_t(x):
    m=torch.as_tensor(PHYS_MEAN,device=x.device,dtype=x.dtype)
    s=torch.as_tensor(PHYS_STD,device=x.device,dtype=x.dtype)
    return x*s+m

@torch.no_grad()
def evaluate_test():
    model.eval()
    rows=[dict(pred=[],true=[],fail_p=[],fail_y=[],visual_cos=[]) for _ in range(ROLLOUT_H)]

    for batch in test_loader:
        b=move(batch)
        pcond=model.encode_prop(b['prop'].float())
        ctx=b['visual'].float()

        for h in range(ROLLOUT_H):
            o=model.step(ctx,pcond,b['actions'][:,h].float())
            pred=unphys_t(o['phys']).cpu().numpy()
            true=unphys_t(b['target_phys'][:,h].float()).cpu().numpy()

            rows[h]['pred'].append(pred)
            rows[h]['true'].append(true)
            rows[h]['fail_p'].append(torch.sigmoid(o['fail_logit']).cpu().numpy())
            rows[h]['fail_y'].append(b['target_fail'][:,h].cpu().numpy())
            rows[h]['visual_cos'].append(
                F.cosine_similarity(o['visual'],b['target_visual'][:,h].float(),dim=-1)
                .cpu().numpy().reshape(-1)
            )

            # Fully autoregressive: no future measured visual/proprio input.
            ctx=torch.cat([ctx[:,1:],o['visual'].detach()[:,None]],1)
            pcond=model.prop_feedback(o['phys']).detach()

    table=[]
    detailed={}
    for h,r in enumerate(rows,1):
        pred=np.concatenate(r['pred'],0)
        true=np.concatenate(r['true'],0)
        fp=np.concatenate(r['fail_p'],0)
        fy=np.concatenate(r['fail_y'],0)

        item={'horizon_step':h}
        for j,name in enumerate(PHYS_NAMES):
            item[f'{name}_MAE']=float(np.mean(np.abs(pred[:,j]-true[:,j])))
            item[f'{name}_R2']=float(r2_score(true[:,j],pred[:,j])) if len(pred)>=2 else np.nan
        item['DINO_cosine']=float(np.concatenate(r['visual_cos']).mean())
        item['failure_AUC']=float(roc_auc_score(fy,fp)) if len(np.unique(fy))==2 else np.nan
        table.append(item)
        detailed[h]=(pred,true,fp,fy)
    return pd.DataFrame(table),detailed

t0=time.time()
TEST_METRICS,TEST_DETAIL=evaluate_test()
TIMING['testing_s']=time.time()-t0
display(TEST_METRICS)
TEST_METRICS.to_csv(OUT_ROOT/'heldout_rough_rider_metrics.csv',index=False)
print(f'TEST TIME = {TIMING["testing_s"]/60:.2f} min')

# %% notebook cell 11

# CELL 11 — PAPER ROUGHNESS / YAW-VIBRATION COMPARISON ON THIS EXACT TEST SEQUENCE
#
# Feel-WM paper definition:
#   realized roughness = std(vertical acceleration)
#   yaw vibration      = mean(abs(heading rate from logged poses))
#
# Isaac Sim "Rough", Table II:
#   Visual-only WM      6.46 m/s^2, 0.73 rad/s
#   Feel-WM w/o prop.   6.53 m/s^2, 0.77 rad/s
#   Feel-WM             6.51 m/s^2, 0.65 rad/s

def exact_test_interval(records):
    starts=[]; ends=[]
    for r in records:
        i=r['origin']; starts.append(i)
        ends.append(i+ROLLOUT_H*r['step'])
    return min(starts),max(ends)

i0,i1=exact_test_interval(TEST_REC)

# Vertical acceleration from recorded IMU.
az=TEST['imu'][i0:i1+1,2]
ours_rough=float(np.std(az))

# Heading rate from logged odometry poses.
tt=TEST['t'][i0:i1+1]
yaw=np.unwrap(Rotation.from_quat(TEST['odom'][i0:i1+1,3:7]).as_euler('xyz')[:,2])
heading_rate=np.gradient(yaw,tt)
ours_yaw=float(np.mean(np.abs(heading_rate)))

t0_test=float(TEST['t'][i0]-TEST['t'][0])
t1_test=float(TEST['t'][i1]-TEST['t'][0])

COMPARE=pd.DataFrame([
    {'source':'Feel-WM paper — Isaac Sim Rough — Visual-only WM',
     'roughness_m_s2':6.46,'yaw_vibration_rad_s':0.73},
    {'source':'Feel-WM paper — Isaac Sim Rough — Feel-WM w/o proprio',
     'roughness_m_s2':6.53,'yaw_vibration_rad_s':0.77},
    {'source':'Feel-WM paper — Isaac Sim Rough — Feel-WM',
     'roughness_m_s2':6.51,'yaw_vibration_rad_s':0.65},
    {'source':f'OURS — TartanDrive2 rough_rider_turnpike_loop_0 [{t0_test:.2f},{t1_test:.2f}] s',
     'roughness_m_s2':ours_rough,'yaw_vibration_rad_s':ours_yaw},
])
display(COMPARE)
COMPARE.to_csv(OUT_ROOT/'roughness_yaw_comparison.csv',index=False)

print('HELD-OUT TEST BAG:',CFG.test_key)
print(f'exact evaluated interval relative to aligned start: {t0_test:.3f}–{t1_test:.3f} s')
print(f'OUR measured roughness    = {ours_rough:.5f} m/s^2')
print(f'OUR measured yaw vibration= {ours_yaw:.5f} rad/s')
print(
    '\nThese are not same-domain closed-loop scores: the three paper rows are Isaac-Sim '
    'closed-loop navigation aggregates, while our row is one fixed recorded TartanDrive2 sequence.'
)

# %% notebook cell 12

# CELL 12 — RUNTIME REPORT + EXPORT
TIMING['total_s']=time.time()-T_ALL

runtime=pd.DataFrame([
    ['official TD2 bag download',TIMING.get('data_download_s',np.nan)],
    ['bag parse / data import',TIMING.get('bag_parse_s',np.nan)],
    ['selected RGB + frozen DINO',TIMING.get('dino_s',np.nan)],
    ['training',TIMING.get('training_s',np.nan)],
    ['testing',TIMING.get('testing_s',np.nan)],
    ['TOTAL',TIMING.get('total_s',np.nan)],
],columns=['stage','seconds'])
runtime['minutes']=runtime.seconds/60
display(runtime)
runtime.to_csv(OUT_ROOT/'runtime_report.csv',index=False)

summary={
    'data_source':'TartanDrive2 only',
    'train_bag':CFG.train_key,
    'test_bag':CFG.test_key,
    'architecture':{
        'DINO':'ViT-B/14 frozen; 256x768',
        'context_frames':4,
        'proprio_window':'50 samples / 0.5 s',
        'proprio_embedding':128,
        'transformer_layers':6,
        'attention_heads':8,
        'MLP_ratio':4,
        'dropout':0.1,
        'rollout_H':8,
        'physical_supervision_s':1.0,
        'dt_choices':list(DT_CHOICES),
    },
    'training_only_rpm_to_mps':RPM_TO_MPS,
    'runtime_minutes':{k:float(v/60) for k,v in TIMING.items()},
    'test_metrics':TEST_METRICS.to_dict(orient='records'),
    'ride_comparison':COMPARE.to_dict(orient='records'),
}
(OUT_ROOT/'run_summary.json').write_text(json.dumps(summary,indent=2))

print('Saved outputs:',OUT_ROOT)
for p in sorted(OUT_ROOT.iterdir()): print(' ',p.name)
print(f'\nTOTAL WALL TIME = {TIMING["total_s"]/60:.2f} min')
