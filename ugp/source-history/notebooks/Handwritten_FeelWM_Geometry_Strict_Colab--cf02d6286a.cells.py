# Source inspection export of the adjacent notebook.
# Notebook execution order and dependencies still apply.
# IPython magic, shell lines and prose-only conclusion cells are comments.

# %% notebook cell 1

# CELL 1 — INSTALL, IMPORTS, AND LOCKED ARCHITECTURE
# IPYTHON: !pip -q install "rosbags>=0.10.0" boto3 botocore opencv-python-headless scipy scikit-learn tqdm

import os, sys, gc, math, json, time, random, re
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
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
from IPython.display import display

from google.colab import drive
drive.mount('/content/drive', force_remount=False)

SEED = 230923
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type != 'cuda':
    raise RuntimeError('Use a Colab GPU runtime.')

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---------- ARCHITECTURE: LOCKED ----------
DINO_PATCHES = 256
DINO_DIM = 768
P_DIM = 128
TX_LAYERS = 8
TX_HEADS = 8
TX_FFN = 3072
TX_DROPOUT = 0.1
VISUAL_STREAMS = 9
TX_TOKENS = VISUAL_STREAMS * DINO_PATCHES + 4  # 2308
ROLLOUT_H = 8
PROP_SAMPLES = 50
GRID_HZ = 100
GRID_DT = 1.0 / GRID_HZ
CORNER_NAMES = ('FL','FR','RL','RR')

assert TX_TOKENS == 2308

@dataclass
class Config:
    endpoint: str = "https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/AUTH_ac8533a83cff4d48bc8c608ad222d330"
    bucket: str = "tartandrive2"

    # resolved against the official public bucket at runtime
    train_hint: str = "gupta"
    test_hint: str = "rough_rider_turnpike_loop"

    max_sequence_s: float = 60.0

    # Runtime is controlled ONLY by data/optimization count.
    max_train_origins: int = 6
    max_val_origins: int = 2
    max_test_origins: int = 4
    origin_stride_frames: int = 2

    epochs: int = 4
    max_optimizer_steps: int = 6
    max_train_minutes: float = 15.0
    batch_size: int = 1
    lr: float = 3e-4
    weight_decay: float = 2e-4
    grad_clip: float = 1.0

    lambda_prop: float = 0.5
    lambda_fail: float = 1.0

    # handwritten λ_every_head was symbolic; 1.0 means all drawn auxiliary
    # heads are included without inventing an extra preference.
    lambda_every_head: float = 1.0
    visual_cosine_weight: float = 0.1

    raw_root: str = '/content/handwritten_td2_raw'
    output_root: str = '/content/drive/MyDrive/HANDWRITTEN_FEELWM_STRICT'

CFG = Config()
RAW_ROOT = Path(CFG.raw_root)
OUT_ROOT = Path(CFG.output_root)
CACHE_ROOT = OUT_ROOT / 'cache'
for p in (RAW_ROOT, OUT_ROOT, CACHE_ROOT):
    p.mkdir(parents=True, exist_ok=True)

TIMING = {}
T_ALL = time.time()

print(json.dumps(asdict(CFG), indent=2))
print('GPU:', torch.cuda.get_device_name(0))
print('LOCKED TRANSFORMER INPUT:', TX_TOKENS, 'x', DINO_DIM)

# %% notebook cell 2

# CELL 2 — RESOLVE + DOWNLOAD TWO OFFICIAL TARTANDRIVE 2 BAGS

import boto3
from botocore import UNSIGNED
from botocore.client import Config as BotoConfig

s3 = boto3.client(
    's3',
    endpoint_url=CFG.endpoint,
    config=BotoConfig(signature_version=UNSIGNED),
)

def find_bag_by_hint(hint):
    hint = hint.lower()
    token = None
    matches = []
    while True:
        kwargs = dict(Bucket=CFG.bucket, Prefix='bags/', MaxKeys=1000)
        if token is not None:
            kwargs['ContinuationToken'] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get('Contents', []):
            k = obj['Key']
            if k.endswith('.bag') and hint in k.lower():
                matches.append((k, int(obj['Size'])))
        if matches:
            break
        if not resp.get('IsTruncated', False):
            break
        token = resp.get('NextContinuationToken')

    if not matches:
        raise FileNotFoundError(
            f'No public TartanDrive2 .bag containing hint={hint!r}. '
            'Edit train_hint/test_hint only if the public object name changed.'
        )
    matches.sort(key=lambda x: (len(x[0]), x[0]))
    print('resolved', hint, '->', matches[0][0], f'{matches[0][1]/1024**3:.2f} GiB')
    return matches[0]

def download_key(key, size, dst):
    dst = Path(dst)
    if dst.is_file() and dst.stat().st_size == size:
        print('reuse:', dst, f'{size/1024**3:.2f} GiB')
        return dst

    print('download:', key, f'{size/1024**3:.2f} GiB')
    body = s3.get_object(Bucket=CFG.bucket, Key=key)['Body']
    with open(dst, 'wb') as f, tqdm(total=size, unit='B', unit_scale=True, desc=dst.name) as bar:
        for chunk in body.iter_chunks(chunk_size=8*1024*1024):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
    if dst.stat().st_size != size:
        raise RuntimeError(f'incomplete download {dst.stat().st_size} != {size}')
    return dst

t0 = time.time()
TRAIN_KEY, TRAIN_BYTES = find_bag_by_hint(CFG.train_hint)
TEST_KEY, TEST_BYTES = find_bag_by_hint(CFG.test_hint)

print(f'TOTAL REQUIRED DOWNLOAD IF NOT CACHED = {(TRAIN_BYTES+TEST_BYTES)/1024**3:.2f} GiB')

TRAIN_BAG = download_key(TRAIN_KEY, TRAIN_BYTES, RAW_ROOT/'train.bag')
TEST_BAG = download_key(TEST_KEY, TEST_BYTES, RAW_ROOT/'test_rough.bag')

TIMING['download_s'] = time.time() - t0
print(f'DOWNLOAD TIME = {TIMING["download_s"]/60:.2f} min')

# %% notebook cell 3

# CELL 3 — TOPIC AUDIT + COMMON REAL GEOMETRY SOURCE
from rosbags.highlevel import AnyReader

def topic_table(bag):
    with AnyReader([Path(bag)]) as reader:
        return sorted((c.topic, c.msgtype) for c in reader.connections)

TRAIN_TOPICS = topic_table(TRAIN_BAG)
TEST_TOPICS = topic_table(TEST_BAG)

print('TRAIN TOPICS')
display(pd.DataFrame(TRAIN_TOPICS, columns=['topic','msgtype']))
print('TEST TOPICS')
display(pd.DataFrame(TEST_TOPICS, columns=['topic','msgtype']))

def select_topic(topics, exact=(), groups=(), required=True, label='topic'):
    names = [x[0] for x in topics]
    for x in exact:
        if x in names:
            return x
    low = {t:t.lower() for t in names}
    for kws in groups:
        hits = [t for t,s in low.items() if all(k in s for k in kws)]
        if hits:
            return sorted(hits, key=len)[0]
    if required:
        raise RuntimeError(f'Could not locate {label}.\nAvailable topics:\n' + '\n'.join(names))
    return None

def candidate_topics(topics):
    return dict(
        rgb=select_topic(
            topics,
            exact=('/multisense/left/image_rect_color','/multisense/left/image_rect'),
            groups=(('multisense','left','image','rect'),),
            label='left RGB'
        ),
        depth=select_topic(
            topics,
            exact=('/multisense/left/depth','/multisense/left/openni_depth'),
            groups=(('multisense','left','depth'),('openni','depth')),
            required=False, label='depth'
        ),
        caminfo=select_topic(
            topics,
            exact=('/multisense/left/depth/camera_info','/multisense/left/image_rect/camera_info'),
            groups=(('multisense','left','camera_info'),),
            required=False, label='camera info'
        ),
        lidar=select_topic(
            topics,
            exact=('/velodyne_1/velodyne_points','/velodyne_2/velodyne_points'),
            groups=(('velodyne','points'),('points','raw')),
            required=False, label='LiDAR point cloud'
        ),
        imu=select_topic(
            topics,
            exact=('/multisense/imu/imu_data','/novatel/imu/data'),
            groups=(('imu','data'),('imu',)),
            label='IMU'
        ),
        odom=select_topic(
            topics,
            exact=('/odometry/filtered_odom','/odometry/filtered'),
            groups=(('odometry','filtered'),('odom',)),
            label='odometry'
        ),
        wheel=select_topic(
            topics, exact=('/wheel_rpm',), groups=(('wheel','rpm'),('rpm',)),
            label='wheel RPM'
        ),
        shock=select_topic(
            topics, exact=('/shock_pos',), groups=(('shock','pos'),('shock',)),
            label='shock position'
        ),
        hazard=select_topic(
            topics, exact=('/mux/intervention','/intervention'),
            groups=(('intervention',),('hazard',)),
            label='intervention/failure'
        ),
    )

TRAIN_SEL = candidate_topics(TRAIN_TOPICS)
TEST_SEL = candidate_topics(TEST_TOPICS)

# Use depth only if BOTH runs contain depth and camera-info.
# Otherwise use real LiDAR for BOTH runs. Never mix train/test geometry modalities.
if TRAIN_SEL['depth'] and TEST_SEL['depth'] and TRAIN_SEL['caminfo'] and TEST_SEL['caminfo']:
    GEOMETRY_SOURCE = 'depth'
elif TRAIN_SEL['lidar'] and TEST_SEL['lidar']:
    GEOMETRY_SOURCE = 'lidar'
else:
    raise RuntimeError(
        'The handwritten model requires one consistent real geometry modality. '
        'Neither verified depth+CameraInfo nor LiDAR exists in both selected runs.'
    )

TRAIN_SEL['geometry_source'] = GEOMETRY_SOURCE
TEST_SEL['geometry_source'] = GEOMETRY_SOURCE

print('COMMON GEOMETRY SOURCE =', GEOMETRY_SOURCE)
print('TRAIN:', json.dumps(TRAIN_SEL, indent=2))
print('TEST :', json.dumps(TEST_SEL, indent=2))

# %% notebook cell 4

# CELL 4 — EXTRACT NUMERIC STREAMS + ALIGN TO 100 Hz

def flatten_numeric(obj, prefix=''):
    out = []
    if obj is None:
        return out
    if isinstance(obj, (bool, np.bool_)):
        return [(prefix, float(obj))]
    if isinstance(obj, (int,float,np.integer,np.floating)):
        return [(prefix, float(obj))]
    if isinstance(obj, np.ndarray) and np.issubdtype(obj.dtype, np.number):
        return [(f'{prefix}[{i}]', float(v)) for i,v in enumerate(obj.reshape(-1))]
    if isinstance(obj, (list,tuple)):
        for i,v in enumerate(obj):
            out.extend(flatten_numeric(v, f'{prefix}[{i}]'))
        return out
    if is_dataclass(obj):
        for f in fields(obj):
            name = f'{prefix}.{f.name}' if prefix else f.name
            out.extend(flatten_numeric(getattr(obj,f.name), name))
        return out
    for name in getattr(obj, '__slots__', []):
        if str(name).startswith('_'):
            continue
        try:
            full = f'{prefix}.{name}' if prefix else name
            out.extend(flatten_numeric(getattr(obj,name), full))
        except Exception:
            pass
    return out

def xyz(v):
    return np.array([float(v.x),float(v.y),float(v.z)], np.float64)

def quat(v):
    return np.array([float(v.x),float(v.y),float(v.z),float(v.w)], np.float64)

def four_numeric(msg, label):
    vals = [
        (n,v) for n,v in flatten_numeric(msg)
        if not any(k in n.lower() for k in ('stamp','secs','nsecs','seq'))
    ]
    named = [
        x for x in vals
        if label in x[0].lower() or any(k.lower() in x[0].lower() for k in CORNER_NAMES)
    ]
    use = named if len(named) >= 4 else vals
    if len(use) < 4:
        raise RuntimeError(f'{label}: expected at least 4 numeric values; saw {use}')
    return np.array([v for _,v in use[:4]], np.float64)

def binary_hazard(msg):
    vals = [
        (n,v) for n,v in flatten_numeric(msg)
        if not any(k in n.lower() for k in ('stamp','secs','nsecs','seq'))
    ]
    if not vals:
        raise RuntimeError('intervention/failure message contains no numeric/bool value')
    v = float(vals[0][1])
    if abs(v-round(v)) < 1e-6 and int(round(v)) in (0,1):
        return float(int(round(v)))
    raise RuntimeError(
        f'Intervention stream is not binary (saw {v}). '
        'No arbitrary failure threshold is invented.'
    )

def extract_streams(bag, sel, max_s):
    out = {k:[] for k in (
        'imu_t','imu','odom_t','odom','wheel_t','wheel',
        'shock_t','shock','haz_t','haz','rgb_t','geom_t'
    )}
    K = None

    wanted = {sel['imu'],sel['odom'],sel['wheel'],sel['shock'],sel['hazard'],sel['rgb']}
    if sel['geometry_source'] == 'depth':
        wanted.add(sel['depth']); wanted.add(sel['caminfo'])
    else:
        wanted.add(sel['lidar'])

    with AnyReader([Path(bag)]) as reader:
        conns = [c for c in reader.connections if c.topic in wanted]
        first = None
        for c,ts,raw in reader.messages(connections=conns):
            t = ts * 1e-9
            if first is None:
                first = t
            if t-first > max_s + 5.0:
                break

            if c.topic == sel['rgb']:
                out['rgb_t'].append(t)
                continue
            if c.topic == (sel['depth'] if sel['geometry_source']=='depth' else sel['lidar']):
                out['geom_t'].append(t)
                continue

            msg = reader.deserialize(raw, c.msgtype)
            if c.topic == sel['imu']:
                out['imu_t'].append(t)
                out['imu'].append(np.r_[xyz(msg.linear_acceleration), xyz(msg.angular_velocity)])
            elif c.topic == sel['odom']:
                out['odom_t'].append(t)
                out['odom'].append(np.r_[
                    xyz(msg.pose.pose.position),
                    quat(msg.pose.pose.orientation),
                    xyz(msg.twist.twist.linear),
                    xyz(msg.twist.twist.angular)
                ])
            elif c.topic == sel['wheel']:
                out['wheel_t'].append(t)
                out['wheel'].append(four_numeric(msg,'rpm'))
            elif c.topic == sel['shock']:
                out['shock_t'].append(t)
                out['shock'].append(four_numeric(msg,'shock'))
            elif c.topic == sel['hazard']:
                out['haz_t'].append(t)
                out['haz'].append(binary_hazard(msg))
            elif sel['geometry_source']=='depth' and c.topic == sel['caminfo']:
                kval = getattr(msg,'k',None)
                if kval is None:
                    kval = getattr(msg,'K',None)
                K = np.asarray(kval,dtype=np.float64).reshape(3,3)

    for k in ('imu_t','odom_t','wheel_t','shock_t','haz_t','rgb_t','geom_t'):
        out[k] = np.asarray(out[k], np.float64)
    for k in ('imu','odom','wheel','shock','haz'):
        out[k] = np.asarray(out[k], np.float64)
    out['K'] = K

    for name in ('imu','odom','wheel','shock','haz'):
        if len(out[name]) < 10:
            raise RuntimeError(f'{name}: only {len(out[name])} usable samples')
    if len(out['rgb_t']) < 16 or len(out['geom_t']) < 16:
        raise RuntimeError('Too few RGB/geometry frames in selected sequence.')
    if sel['geometry_source']=='depth' and K is None:
        raise RuntimeError(
            'Depth is present but no recorded CameraInfo/intrinsic matrix was found. '
            'The notebook will not invent intrinsics.'
        )
    return out

def interp_cols(t, x, tq):
    return np.stack([np.interp(tq,t,x[:,j]) for j in range(x.shape[1])],1)

def align_100hz(raw, max_s):
    start = max(
        raw['imu_t'][0],raw['odom_t'][0],raw['wheel_t'][0],
        raw['shock_t'][0],raw['haz_t'][0],raw['rgb_t'][0],raw['geom_t'][0]
    )
    end = min(
        raw['imu_t'][-1],raw['odom_t'][-1],raw['wheel_t'][-1],
        raw['shock_t'][-1],raw['haz_t'][-1],raw['rgb_t'][-1],raw['geom_t'][-1],
        start+max_s
    )
    if end-start < 10:
        raise RuntimeError(f'Usable overlap is only {end-start:.2f}s.')

    tq = np.arange(start,end,GRID_DT)
    imu = interp_cols(raw['imu_t'],raw['imu'],tq)
    odom = interp_cols(raw['odom_t'],raw['odom'],tq)
    wheel = interp_cols(raw['wheel_t'],raw['wheel'],tq)
    shock = interp_cols(raw['shock_t'],raw['shock'],tq)

    hi = np.searchsorted(raw['haz_t'],tq,side='right')-1
    hi = np.clip(hi,0,len(raw['haz'])-1)
    hazard = raw['haz'][hi].astype(np.float32)

    q = odom[:,3:7]
    q = q/np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-9)
    euler = Rotation.from_quat(q).as_euler('xyz').astype(np.float32)

    # shared body: accel3 + gyro3 + linear velocity3 + roll/pitch/yaw3 = 12
    body = np.c_[imu[:,:3],imu[:,3:6],odom[:,7:10],euler].astype(np.float32)

    # four real corner streams: shared body + local RPM + local shock = 14
    corner = np.stack([
        np.c_[body,wheel[:,i],shock[:,i]] for i in range(4)
    ],axis=1).astype(np.float32)

    rgb_t = raw['rgb_t'][(raw['rgb_t']>=tq[0])&(raw['rgb_t']<=tq[-1])]
    geom_t = raw['geom_t'][(raw['geom_t']>=tq[0])&(raw['geom_t']<=tq[-1])]

    return dict(
        t=tq,imu=imu.astype(np.float32),odom=odom.astype(np.float32),
        wheel=wheel.astype(np.float32),shock=shock.astype(np.float32),
        hazard=hazard,body=body,corner=corner,rgb_t=rgb_t,geom_t=geom_t,K=raw['K']
    )

t0 = time.time()
TRAIN_RAW = extract_streams(TRAIN_BAG, TRAIN_SEL, CFG.max_sequence_s)
TEST_RAW = extract_streams(TEST_BAG, TEST_SEL, CFG.max_sequence_s)
TRAIN = align_100hz(TRAIN_RAW, CFG.max_sequence_s)
TEST = align_100hz(TEST_RAW, CFG.max_sequence_s)
TIMING['stream_parse_s'] = time.time()-t0

print('TRAIN corner:', TRAIN['corner'].shape, 'RGB:', len(TRAIN['rgb_t']), 'geometry:', len(TRAIN['geom_t']))
print('TEST  corner:', TEST['corner'].shape, 'RGB:', len(TEST['rgb_t']), 'geometry:', len(TEST['geom_t']))
print(f'STREAM PARSE = {TIMING["stream_parse_s"]/60:.2f} min')

# %% notebook cell 5

# CELL 5 — CHRONOLOGICAL CAMERA-FRAME ORIGINS

def nearest_grid(seq,t):
    return int(np.argmin(np.abs(seq['t']-float(t))))

def nearest_geom_index(seq,rgb_index):
    t = seq['rgb_t'][int(rgb_index)]
    return int(np.argmin(np.abs(seq['geom_t']-t)))

def action_between_frames(seq,fi0,fi1):
    i0 = nearest_grid(seq,seq['rgb_t'][fi0])
    i1 = nearest_grid(seq,seq['rgb_t'][fi1])

    p0 = seq['odom'][i0,:3]
    p1 = seq['odom'][i1,:3]
    yaw0 = Rotation.from_quat(seq['odom'][i0,3:7]).as_euler('xyz')[2]
    yaw1 = Rotation.from_quat(seq['odom'][i1,3:7]).as_euler('xyz')[2]

    dp = p1-p0
    c,s = np.cos(yaw0),np.sin(yaw0)
    dx = c*dp[0] + s*dp[1]
    dy = -s*dp[0] + c*dp[1]
    dyaw = (yaw1-yaw0+np.pi)%(2*np.pi)-np.pi
    dt = float(seq['rgb_t'][fi1]-seq['rgb_t'][fi0])
    return np.array([dx,dy,dyaw,dt],np.float32)

def valid_origins(seq):
    out = []
    for fi in range(3,len(seq['rgb_t'])-ROLLOUT_H,CFG.origin_stride_frames):
        gi = nearest_grid(seq,seq['rgb_t'][fi])
        gj = nearest_grid(seq,seq['rgb_t'][fi+ROLLOUT_H])
        if gi < PROP_SAMPLES-1 or gj < PROP_SAMPLES-1:
            continue
        out.append(fi)
    return np.asarray(out,int)

tr_all = valid_origins(TRAIN)
te_all = valid_origins(TEST)
if len(tr_all) < 4 or len(te_all) < 1:
    raise RuntimeError(f'Not enough origins train={len(tr_all)} test={len(te_all)}.')

nval = min(CFG.max_val_origins, max(1,len(tr_all)//5))
TRAIN_ORIGINS = tr_all[:min(CFG.max_train_origins,max(1,len(tr_all)-nval))]
VAL_ORIGINS = tr_all[len(TRAIN_ORIGINS):len(TRAIN_ORIGINS)+nval]
if len(VAL_ORIGINS)==0:
    VAL_ORIGINS = tr_all[-1:]
TEST_ORIGINS = te_all[:CFG.max_test_origins]

print('TRAIN origins:',TRAIN_ORIGINS.tolist())
print('VAL origins  :',VAL_ORIGINS.tolist())
print('TEST origins :',TEST_ORIGINS.tolist())

def needed_frames(origins):
    ids=set()
    for t in origins:
        ids.update(range(int(t)-3,int(t)+ROLLOUT_H+1))
    return sorted(ids)

TRAIN_RGB_NEED = needed_frames(np.unique(np.r_[TRAIN_ORIGINS,VAL_ORIGINS]))
TEST_RGB_NEED = needed_frames(TEST_ORIGINS)

print('needed RGB frames:',len(TRAIN_RGB_NEED),'train/val,',len(TEST_RGB_NEED),'test')

# %% notebook cell 6

# CELL 6 — EXTRACT ONLY REQUIRED RGB + DEPTH/LIDAR MESSAGES

import cv2
from PIL import Image

def decode_image(msg):
    data = bytes(msg.data)
    if hasattr(msg,'format') and not hasattr(msg,'height'):
        im = cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
        if im is None:
            raise RuntimeError('compressed image decode failed')
        return cv2.cvtColor(im,cv2.COLOR_BGR2RGB)

    h,w = int(msg.height),int(msg.width)
    enc = str(msg.encoding).lower()
    step = int(getattr(msg,'step',0))
    arr = np.frombuffer(data,np.uint8)

    if enc in ('rgb8','bgr8'):
        row = step if step else w*3
        im = arr.reshape(h,row)[:,:w*3].reshape(h,w,3)
        return im if enc=='rgb8' else cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    if enc in ('rgba8','bgra8'):
        row = step if step else w*4
        im = arr.reshape(h,row)[:,:w*4].reshape(h,w,4)
        code = cv2.COLOR_RGBA2RGB if enc=='rgba8' else cv2.COLOR_BGRA2RGB
        return cv2.cvtColor(im,code)
    if enc in ('mono8','8uc1'):
        im = arr.reshape(h,-1)[:,:w]
        return np.repeat(im[...,None],3,-1)
    raise RuntimeError(f'unsupported RGB encoding {enc!r}')

def decode_depth(msg):
    data = bytes(msg.data)
    h,w = int(msg.height),int(msg.width)
    enc = str(msg.encoding).lower()
    step = int(getattr(msg,'step',0))

    if enc in ('32fc1','32fc'):
        row_elems = (step//4) if step else w
        d = np.frombuffer(data,dtype='<f4').reshape(h,row_elems)[:,:w].astype(np.float32)
    elif enc in ('16uc1','mono16'):
        row_elems = (step//2) if step else w
        d = np.frombuffer(data,dtype='<u2').reshape(h,row_elems)[:,:w].astype(np.float32)/1000.0
    else:
        raise RuntimeError(f'unsupported depth encoding {enc!r}')
    d[~np.isfinite(d)] = 0
    return d

POINTFIELD_DT = {7:'<f4',8:'<f8'}

def pointcloud_xyz(msg):
    fmap = {f.name:f for f in msg.fields}
    if not all(k in fmap for k in ('x','y','z')):
        raise RuntimeError('PointCloud2 does not contain x/y/z fields.')

    n = int(msg.width)*int(msg.height)
    buf = bytes(msg.data)
    step = int(msg.point_step)
    xyz = np.empty((n,3),np.float32)

    for j,name in enumerate(('x','y','z')):
        f = fmap[name]
        code = int(f.datatype)
        if code not in POINTFIELD_DT:
            raise RuntimeError(f'PointCloud2 {name} datatype {code} is unsupported.')
        xyz[:,j] = np.ndarray(
            shape=(n,), dtype=np.dtype(POINTFIELD_DT[code]),
            buffer=buf, offset=int(f.offset), strides=(step,)
        ).astype(np.float32)

    return xyz[np.isfinite(xyz).all(1)]

def extract_selected(bag,sel,seq,needed_rgb):
    rgb_needed = set(map(int,needed_rgb))
    geom_needed = {nearest_geom_index(seq,i) for i in rgb_needed}
    rgb,geom = {},{}
    rgb_k=0; geom_k=0
    geom_topic = sel['depth'] if sel['geometry_source']=='depth' else sel['lidar']

    with AnyReader([Path(bag)]) as reader:
        conns = [c for c in reader.connections if c.topic in {sel['rgb'],geom_topic}]
        for c,ts,raw in reader.messages(connections=conns):
            if c.topic == sel['rgb']:
                if rgb_k in rgb_needed:
                    msg = reader.deserialize(raw,c.msgtype)
                    rgb[rgb_k] = decode_image(msg)
                rgb_k += 1
            elif c.topic == geom_topic:
                if geom_k in geom_needed:
                    msg = reader.deserialize(raw,c.msgtype)
                    geom[geom_k] = decode_depth(msg) if sel['geometry_source']=='depth' else pointcloud_xyz(msg)
                geom_k += 1

            if len(rgb)==len(rgb_needed) and len(geom)==len(geom_needed):
                break

    mr = rgb_needed-set(rgb)
    mg = geom_needed-set(geom)
    if mr:
        raise RuntimeError(f'Missing selected RGB frame IDs: {sorted(mr)[:20]}')
    if mg:
        raise RuntimeError(f'Missing selected geometry frame IDs: {sorted(mg)[:20]}')
    return rgb,geom

t0=time.time()
TRAIN_RGB,TRAIN_GEOM_RAW = extract_selected(TRAIN_BAG,TRAIN_SEL,TRAIN,TRAIN_RGB_NEED)
TEST_RGB,TEST_GEOM_RAW = extract_selected(TEST_BAG,TEST_SEL,TEST,TEST_RGB_NEED)
TIMING['selected_modalities_s'] = time.time()-t0

print('geometry source:',GEOMETRY_SOURCE)
print(f'SELECTED RGB/GEOMETRY READ = {TIMING["selected_modalities_s"]/60:.2f} min')

# %% notebook cell 7

# CELL 7 — FROZEN DINOV2 ViT-B/14: 256 x 768 TOKENS

import torchvision.transforms as T

DINO_TFM = T.Compose([
    T.Resize(256,interpolation=T.InterpolationMode.BICUBIC,antialias=True),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=(.485,.456,.406),std=(.229,.224,.225)),
])

dino = torch.hub.load('facebookresearch/dinov2','dinov2_vitb14').to(DEVICE).eval()
for p in dino.parameters():
    p.requires_grad_(False)

@torch.no_grad()
def build_dino_cache(images,cache_path):
    ids = sorted(images)
    old = {}
    cache_path = Path(cache_path)

    if cache_path.is_file():
        try:
            with np.load(cache_path,allow_pickle=False) as z:
                old = {int(i):z['tokens'][j] for j,i in enumerate(z['ids'])}
        except Exception:
            old = {}

    todo = [i for i in ids if i not in old]
    for a in tqdm(range(0,len(todo),8),desc='DINO'):
        bid = todo[a:a+8]
        x = torch.stack([DINO_TFM(Image.fromarray(images[i])) for i in bid]).to(DEVICE)
        with torch.amp.autocast('cuda',dtype=torch.float16):
            tok = dino.forward_features(x)['x_norm_patchtokens']
        if tok.shape[1:] != (256,768):
            raise AssertionError(f'Expected 256x768 DINO tokens, got {tuple(tok.shape)}')
        tok = tok.float().cpu().numpy().astype(np.float16)
        for j,i in enumerate(bid):
            old[int(i)] = tok[j]

    all_ids = np.asarray(sorted(old),np.int32)
    all_tok = np.stack([old[int(i)] for i in all_ids])
    np.savez_compressed(cache_path,ids=all_ids,tokens=all_tok)

    return {int(i):np.asarray(old[int(i)],np.float32) for i in ids}

t0=time.time()
TRAIN_Z = build_dino_cache(TRAIN_RGB,CACHE_ROOT/'train_dino_vitb14.npz')
TEST_Z = build_dino_cache(TEST_RGB,CACHE_ROOT/'test_dino_vitb14.npz')
del dino
gc.collect(); torch.cuda.empty_cache()
TIMING['dino_s'] = time.time()-t0

print(f'DINO TIME = {TIMING["dino_s"]/60:.2f} min')

# %% notebook cell 8

# CELL 8 — REAL GEOMETRIC ENCODINGS FROM DEPTH IF PRESENT, OTHERWISE LIDAR
#
# Geometry target g_t is pure geometry: 32 points x [x,y,z,nx,ny,nz] + 18 stats.
# No RGB/material vector is inserted into g_t.

def geometry_stats18(points,normals):
    if len(points)==0:
        return np.zeros(18,np.float32)

    p=np.asarray(points,float)
    n=np.asarray(normals,float)
    center=p.mean(0)
    pc=p-center
    z=p[:,2]

    if len(p)>=3:
        cov=np.cov(pc.T)
        ev=np.sort(np.maximum(np.linalg.eigvalsh(cov),0))
        rough=float(np.sqrt(ev[0]))
        curvature=float(ev[0]/max(ev.sum(),1e-8))
    else:
        ev=np.zeros(3)
        rough=curvature=0.0

    nmean=n.mean(0) if len(n) else np.array([0,0,1.])
    nmean=nmean/max(np.linalg.norm(nmean),1e-8)
    radial=np.linalg.norm(pc[:,:2],axis=1)

    return np.asarray([
        1.0,
        min(len(p)/512.0,1.0),
        float(np.min(np.linalg.norm(p,axis=1))),
        float(np.mean(z)),
        float(np.std(z)),
        float(np.ptp(z)),
        float(nmean[0]),float(nmean[1]),float(nmean[2]),
        float(np.arccos(np.clip(abs(nmean[2]),0,1))),
        rough,
        curvature,
        float(ev[-1]),
        float(np.quantile(z,.95)-np.quantile(z,.05)),
        float(np.std(p[:,0])),
        float(np.std(p[:,1])),
        float(np.mean(radial)),
        float(np.max(radial)),
    ],np.float32)

def depth_geometry(depth,K):
    d=np.asarray(depth,np.float32)
    H,W=d.shape
    fx,fy=float(K[0,0]),float(K[1,1])
    cx,cy=float(K[0,2]),float(K[1,2])

    vv,uu=np.indices((H,W))
    good=(d>.15)&(d<80)&np.isfinite(d)
    Z=d[good]
    pts=np.c_[
        (uu[good]-cx)*Z/fx,
        (vv[good]-cy)*Z/fy,
        Z
    ].astype(np.float32)

    if len(pts)<8:
        return np.zeros((32,6),np.float32),np.zeros(18,np.float32)

    # 32 deterministic image-grid query positions.
    us=np.linspace(.15*W,.85*W,8).astype(int)
    vs=np.linspace(.30*H,.90*H,4).astype(int)

    # Local normals from neighboring deprojected pixels.
    grid=np.full((H,W,3),np.nan,np.float32)
    grid[...,0][good]=(uu[good]-cx)*Z/fx
    grid[...,1][good]=(vv[good]-cy)*Z/fy
    grid[...,2][good]=Z

    desc=[]
    normals_valid=[]
    for v in vs:
        for u in us:
            if not good[v,u] or v<1 or v>=H-1 or u<1 or u>=W-1:
                desc.append(np.zeros(6,np.float32))
                continue
            px=grid[v,u+1]-grid[v,u-1]
            py=grid[v+1,u]-grid[v-1,u]
            if not np.isfinite(px).all() or not np.isfinite(py).all():
                n=np.zeros(3,np.float32)
            else:
                n=np.cross(px,py)
                n=n/max(np.linalg.norm(n),1e-8)
            desc.append(np.r_[grid[v,u],n].astype(np.float32))
            normals_valid.append(n)

    desc=np.asarray(desc,np.float32)
    # Stats use the full valid depth point cloud.
    # For its normal summary, use the normals available at sampled valid pixels.
    nstats=np.asarray(normals_valid,np.float32)
    if len(nstats)==0:
        nstats=np.zeros((1,3),np.float32)
    stats=geometry_stats18(pts,nstats)
    return desc,stats

def lidar_geometry(points):
    p=np.asarray(points,np.float32)
    p=p[np.isfinite(p).all(1)]
    if len(p)==0:
        return np.zeros((32,6),np.float32),np.zeros(18,np.float32)

    radial=np.linalg.norm(p[:,:2],axis=1)
    p=p[(radial<20)&(np.abs(p[:,2])<5)]
    if len(p)<8:
        return np.zeros((32,6),np.float32),np.zeros(18,np.float32)

    # Even azimuth coverage.
    ang=np.arctan2(p[:,1],p[:,0])
    order=np.argsort(ang)
    p=p[order]
    take=np.linspace(0,len(p)-1,32).astype(int)
    samp=p[take]

    tree=cKDTree(p)
    normals=np.zeros((32,3),np.float32)
    for i,x in enumerate(samp):
        _,idx=tree.query(x,k=min(12,len(p)))
        nn=p[np.atleast_1d(idx)]
        C=np.cov((nn-nn.mean(0)).T)
        w,V=np.linalg.eigh(C)
        n=V[:,np.argmin(w)]
        normals[i]=n/max(np.linalg.norm(n),1e-8)

    stats=geometry_stats18(p,normals)
    return np.c_[samp,normals].astype(np.float32),stats

def build_geometry_cache(seq,sel,rawgeom,needed_rgb):
    out={}
    for fi in needed_rgb:
        gi=nearest_geom_index(seq,fi)
        raw=rawgeom[gi]
        if sel['geometry_source']=='depth':
            out[int(fi)]=depth_geometry(raw,seq['K'])
        else:
            out[int(fi)]=lidar_geometry(raw)
    return out

t0=time.time()
TRAIN_GRAW=build_geometry_cache(TRAIN,TRAIN_SEL,TRAIN_GEOM_RAW,TRAIN_RGB_NEED)
TEST_GRAW=build_geometry_cache(TEST,TEST_SEL,TEST_GEOM_RAW,TEST_RGB_NEED)
TIMING['geometry_s']=time.time()-t0

ex=TRAIN_GRAW[TRAIN_RGB_NEED[0]]
print('geometry descriptor:',ex[0].shape,'stats:',ex[1].shape)
print(f'GEOMETRY PREP = {TIMING["geometry_s"]/60:.2f} min')

# %% notebook cell 9

# CELL 9 — TRAIN-ONLY NORMALIZATION + DATASET

CORNER_MEAN=TRAIN['corner'].reshape(-1,14).mean(0).astype(np.float32)
CORNER_STD=TRAIN['corner'].reshape(-1,14).std(0).astype(np.float32)
CORNER_STD[CORNER_STD<1e-6]=1.0

acts=[]
for t in TRAIN_ORIGINS:
    for s in range(ROLLOUT_H):
        acts.append(action_between_frames(TRAIN,int(t)+s,int(t)+s+1))
acts=np.stack(acts)
ACTION_MEAN=acts.mean(0).astype(np.float32)
ACTION_STD=acts.std(0).astype(np.float32)
ACTION_STD[ACTION_STD<1e-6]=1.0

np.savez_compressed(
    OUT_ROOT/'normalization.npz',
    CORNER_MEAN=CORNER_MEAN,CORNER_STD=CORNER_STD,
    ACTION_MEAN=ACTION_MEAN,ACTION_STD=ACTION_STD
)

def ncorner(x):
    return ((x-CORNER_MEAN)/CORNER_STD).astype(np.float32)

def naction(x):
    return ((x-ACTION_MEAN)/ACTION_STD).astype(np.float32)

def corner_history(seq,fi):
    gi=nearest_grid(seq,seq['rgb_t'][int(fi)])
    if gi<PROP_SAMPLES-1:
        raise RuntimeError('Insufficient 0.5-s proprio history.')
    return ncorner(seq['corner'][gi-PROP_SAMPLES+1:gi+1].transpose(1,0,2))

class HandwrittenDataset(Dataset):
    def __init__(self,seq,origins,Z,G):
        self.seq=seq
        self.origins=np.asarray(origins,int)
        self.Z=Z
        self.G=G

    def __len__(self):
        return len(self.origins)

    def __getitem__(self,k):
        t=int(self.origins[k])
        # relative frames -3 ... +H inclusive
        frame_ids=list(range(t-3,t+ROLLOUT_H+1))

        Zseq=np.stack([self.Z[i] for i in frame_ids]).astype(np.float32)
        Gpts=np.stack([self.G[i][0] for i in frame_ids]).astype(np.float32)
        Gstats=np.stack([self.G[i][1] for i in frame_ids]).astype(np.float32)

        actions=np.stack([
            naction(action_between_frames(self.seq,t+s,t+s+1))
            for s in range(ROLLOUT_H)
        ])

        phist=np.stack([
            corner_history(self.seq,t+s)
            for s in range(ROLLOUT_H+1)
        ]).astype(np.float32)

        fail=[]
        for s in range(ROLLOUT_H):
            gi=nearest_grid(self.seq,self.seq['rgb_t'][t+s+1])
            fail.append(self.seq['hazard'][gi])

        return dict(
            origin=np.int64(t),
            Zseq=Zseq,
            Gpts=Gpts,
            Gstats=Gstats,
            actions=actions,
            phist=phist,
            fail=np.asarray(fail,np.float32)
        )

train_ds=HandwrittenDataset(TRAIN,TRAIN_ORIGINS,TRAIN_Z,TRAIN_GRAW)
val_ds=HandwrittenDataset(TRAIN,VAL_ORIGINS,TRAIN_Z,TRAIN_GRAW)
test_ds=HandwrittenDataset(TEST,TEST_ORIGINS,TEST_Z,TEST_GRAW)

train_loader=DataLoader(train_ds,batch_size=1,shuffle=True,num_workers=0,pin_memory=True)
val_loader=DataLoader(val_ds,batch_size=1,shuffle=False,num_workers=0,pin_memory=True)
test_loader=DataLoader(test_ds,batch_size=1,shuffle=False,num_workers=0,pin_memory=True)

b=next(iter(train_loader))
for k,v in b.items():
    if torch.is_tensor(v):
        print(k,tuple(v.shape))

# %% notebook cell 10

# CELL 10 — EXACT HANDWRITTEN MODEL

class SharedCornerCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv1d(14,64,5,2,2),nn.GroupNorm(8,64),nn.GELU(),
            nn.Conv1d(64,128,5,2,2),nn.GroupNorm(8,128),nn.GELU(),
            nn.Conv1d(128,128,3,1,1),nn.GroupNorm(8,128),nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.norm=nn.LayerNorm(128)

    def forward(self,x):
        return self.norm(self.net(x.transpose(1,2)).squeeze(-1))

class GeometryEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.point=nn.Sequential(
            nn.Linear(6,64),nn.GELU(),
            nn.Linear(64,96),nn.GELU()
        )
        self.stats=nn.Sequential(
            nn.LayerNorm(18),nn.Linear(18,64),nn.GELU()
        )
        self.fuse=nn.Sequential(
            nn.Linear(160,128),nn.GELU(),nn.LayerNorm(128)
        )
        self.to768=nn.Linear(128,768)

    def forward(self,pts,stats):
        hp=self.point(pts).amax(-2)
        hs=self.stats(stats)
        g=self.fuse(torch.cat([hp,hs],-1))
        return g,self.to768(g)

class SharedFiLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk=nn.Sequential(
            nn.Linear(128,256),nn.GELU(),nn.LayerNorm(256)
        )
        self.v_gamma=nn.Linear(256,768)
        self.v_beta=nn.Linear(256,768)
        self.g_gamma=nn.Linear(256,768)
        self.g_beta=nn.Linear(256,768)

        # identity initialization: (1+delta_gamma), beta=0
        for m in (self.v_gamma,self.v_beta,self.g_gamma,self.g_beta):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def apply(self,x,c,kind):
        r=self.trunk(c)
        if kind=='visual':
            dg,b=self.v_gamma(r),self.v_beta(r)
        else:
            dg,b=self.g_gamma(r),self.g_beta(r)
        while dg.ndim<x.ndim:
            dg=dg.unsqueeze(-2)
            b=b.unsqueeze(-2)
        return x*(1.0+dg)+b

class BigDynamicsTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=768,nhead=8,dim_feedforward=3072,
                dropout=.1,activation='gelu',
                batch_first=True,norm_first=True
            )
            for _ in range(8)
        ])
        self.norm=nn.LayerNorm(768)

    def forward(self,x):
        # Every input to a rollout step is present/current/past.
        # The handwritten diagram did not add future tokens inside the step.
        for layer in self.layers:
            if self.training:
                x=checkpoint(lambda y,layer=layer: layer(y),x,use_reentrant=False)
            else:
                x=layer(x)
        return self.norm(x)

class HandwrittenWorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.corner=SharedCornerCNN()
        self.corner_id_p=nn.Parameter(torch.randn(4,128)*.02)

        self.global_p=nn.Sequential(
            nn.Linear(4*128,256),nn.GELU(),nn.LayerNorm(256),
            nn.Linear(256,128),nn.LayerNorm(128)
        )
        self.action=nn.Sequential(
            nn.Linear(4,128),nn.GELU(),
            nn.Linear(128,128),nn.LayerNorm(128)
        )

        self.film=SharedFiLM()
        self.geom=GeometryEncoder()

        self.visual_pos=nn.Parameter(torch.randn(256,768)*.02)
        self.stream_embed=nn.Parameter(torch.randn(13,768)*.02)
        self.corner_embed=nn.Parameter(torch.randn(4,768)*.02)

        self.tx=BigDynamicsTransformer()

        # Exactly the handwritten head families.
        self.visual_head1=nn.Linear(768,768)
        self.visual_head2=nn.Linear(768,768)  # shared by front 1/2
        self.visual_head3=nn.Linear(768,768)  # shared by rear 3/4

        self.geometry_head1=nn.Sequential(
            nn.Linear(768,256),nn.GELU(),nn.Linear(256,128)
        )
        self.geometry_head2=nn.Sequential(
            nn.Linear(768,256),nn.GELU(),nn.Linear(256,128)
        )

        self.proprio_head=nn.Sequential(
            nn.Linear(768,1024),nn.GELU(),nn.LayerNorm(1024),
            nn.Linear(1024,5*128)
        )
        self.failure_head=nn.Sequential(
            nn.Linear(768,256),nn.GELU(),nn.Linear(256,1)
        )

    def encode_p(self,hist):
        # hist [B,4,50,14]; same CNN for all four corners
        pi=torch.stack([self.corner(hist[:,i]) for i in range(4)],1)
        pg=self.global_p(pi.reshape(hist.shape[0],-1))
        return pg,pi

    def encode_g(self,pts,stats):
        return self.geom(pts,stats)

    def forward_step(
        self,
        raw4,                 # [B,4,256,768]
        front_base,           # [B,2,256,768]
        rear_base,            # [B,2,256,768]
        gfront_base,          # tuple([B,2,128],[B,2,768])
        grear_base,           # tuple([B,2,128],[B,2,768])
        pg,pi,action
    ):
        B=raw4.shape[0]
        ea=self.action(action)
        cG=ea+pg
        ci=ea[:,None,:]+pi+self.corner_id_p[None,:,:]

        streams=[]

        # 4 raw visual streams Z_{c-3:c}
        for j in range(4):
            streams.append(
                raw4[:,j]+self.visual_pos[None]+self.stream_embed[j]
            )

        # global FiLM current stream
        zg=self.film.apply(raw4[:,3],cG,'visual')
        zg=zg+self.visual_pos[None]+self.stream_embed[4]
        streams.append(zg)

        # front: exactly delayed t-2 stream, one recursive stream per corner
        for local_i,corner_i in enumerate((0,1)):
            z=self.film.apply(front_base[:,local_i],ci[:,corner_i],'visual')
            z=z+self.visual_pos[None]+self.stream_embed[5+local_i]+self.corner_embed[corner_i]
            streams.append(z)

        # rear: exactly delayed t-3 stream, one recursive stream per corner
        for local_i,corner_i in enumerate((2,3)):
            z=self.film.apply(rear_base[:,local_i],ci[:,corner_i],'visual')
            z=z+self.visual_pos[None]+self.stream_embed[7+local_i]+self.corner_embed[corner_i]
            streams.append(z)

        gfront768=gfront_base[1]
        grear768=grear_base[1]
        geom_tokens=[]

        for local_i,corner_i in enumerate((0,1)):
            g=self.film.apply(gfront768[:,local_i],ci[:,corner_i],'geometry')
            g=g+self.stream_embed[9+local_i]+self.corner_embed[corner_i]
            geom_tokens.append(g)

        for local_i,corner_i in enumerate((2,3)):
            g=self.film.apply(grear768[:,local_i],ci[:,corner_i],'geometry')
            g=g+self.stream_embed[11+local_i]+self.corner_embed[corner_i]
            geom_tokens.append(g)

        x=torch.cat(streams+[torch.stack(geom_tokens,1)],1)
        if x.shape[1:]!=(2308,768):
            raise AssertionError(x.shape)

        h=self.tx(x)

        s=0
        raw_h=[]
        for _ in range(4):
            raw_h.append(h[:,s:s+256]); s+=256

        h_global=h[:,s:s+256]; s+=256
        h_front1=h[:,s:s+256]; s+=256
        h_front2=h[:,s:s+256]; s+=256
        h_rear3=h[:,s:s+256]; s+=256
        h_rear4=h[:,s:s+256]; s+=256
        h_geom=h[:,s:s+4]

        pooled=h_global.mean(1)

        return dict(
            # visual head 1 -> Z_{c+1}
            main=self.visual_head1(h_global),

            # visual head 2 -> Z_{c-1} for front 1/2
            front1=self.visual_head2(h_front1),
            front2=self.visual_head2(h_front2),

            # visual head 3 -> Z_{c-2} for rear 3/4
            rear3=self.visual_head3(h_rear3),
            rear4=self.visual_head3(h_rear4),

            # geometry head 1/2
            gf1=self.geometry_head1(h_geom[:,0]),
            gf2=self.geometry_head1(h_geom[:,1]),
            gr3=self.geometry_head2(h_geom[:,2]),
            gr4=self.geometry_head2(h_geom[:,3]),

            # one proprioception head -> [pG,p1,p2,p3,p4]_{c+1}
            pnext=self.proprio_head(pooled).reshape(B,5,128),

            # failure c_{c+1}
            fail_logit=self.failure_head(pooled).squeeze(-1)
        )

model=HandwrittenWorldModel().to(DEVICE)
NPAR=sum(p.numel() for p in model.parameters())

print(f'Parameters: {NPAR/1e6:.2f}M')
print('Transformer input: 2308 x 768')
print('Transformer: 8 layers | 8 heads | FFN 3072 | dropout 0.1')

# %% notebook cell 11

# CELL 11 — EXACT MULTI-HEAD RECURSIVE + TEACHER-FORCED ROLLOUT

def move(batch):
    return {
        k:(v.to(DEVICE,non_blocking=True) if torch.is_tensor(v) else v)
        for k,v in batch.items()
    }

def visual_loss(pred,target):
    return (
        F.mse_loss(pred,target)
        + CFG.visual_cosine_weight *
          (1.0-F.cosine_similarity(pred,target,dim=-1).mean())
    )

def teacher_probability(epoch):
    # early = 1.0, middle decay, final = 0.5
    if CFG.epochs<=1:
        return .5
    f=(epoch-1)/max(CFG.epochs-1,1)
    if f<=.25:
        return 1.0
    if f>=.75:
        return .5
    return 1.0-(f-.25)/.5*.5

def encode_geometry_sequence(Gpts,Gstats):
    B,T=Gpts.shape[:2]
    g128,g768=model.encode_g(
        Gpts.reshape(B*T,32,6),
        Gstats.reshape(B*T,18)
    )
    return g128.reshape(B,T,128),g768.reshape(B,T,768)

def rollout(batch,epoch,training=True,backward=False):
    b=move(batch)
    Z=b['Zseq'].float()       # rel -3...+H
    Gpts=b['Gpts'].float()
    Gstats=b['Gstats'].float()
    actions=b['actions'].float()
    phist=b['phist'].float()

    # Real geometry encoding.
    # Only t-3/t-2 enter with gradients initially; all supervision targets are stop-grad.
    g128_all,g768_all=encode_geometry_sequence(Gpts,Gstats)

    def zi(rel):
        return Z[:,rel+3]

    def g128i(rel):
        return g128_all[:,rel+3]

    # current p(t) is trainable input
    pg,pi=model.encode_p(phist[:,0])

    # true future latent targets use the SAME encoder, stop-gradient
    p_future=[]
    with torch.no_grad():
        for j in range(1,ROLLOUT_H+1):
            pgj,pij=model.encode_p(phist[:,j])
            p_future.append(torch.cat([pgj[:,None],pij],1))
    p_future=torch.stack(p_future,1)

    # Exact initial delayed streams:
    raw4=torch.stack([zi(-3),zi(-2),zi(-1),zi(0)],1)

    # Each head owns its own recursive branch — no averaging.
    front_base=torch.stack([zi(-2),zi(-2)],1)
    rear_base=torch.stack([zi(-3),zi(-3)],1)

    gfront128=torch.stack([g128i(-2),g128i(-2)],1)
    grear128=torch.stack([g128i(-3),g128i(-3)],1)
    gfront768=torch.stack([g768_all[:,1],g768_all[:,1]],1) # rel -2
    grear768=torch.stack([g768_all[:,0],g768_all[:,0]],1)  # rel -3

    tfp=teacher_probability(epoch)
    total=0.0
    logs=defaultdict(float)

    if backward:
        optimizer.zero_grad(set_to_none=True)

    for s in range(ROLLOUT_H):
        with torch.amp.autocast('cuda',dtype=torch.float16):
            o=model.forward_step(
                raw4,
                front_base,
                rear_base,
                (gfront128,gfront768),
                (grear128,grear768),
                pg,pi,
                actions[:,s]
            )

            # EXACT handwritten targets at current time c=t+s
            z_main=zi(s+1)       # visual head 1 -> Z_{c+1}
            z_front=zi(s-1)      # visual head 2 -> Z_{c-1}
            z_rear=zi(s-2)       # visual head 3 -> Z_{c-2}
            g_front=g128i(s-1).detach()
            g_rear=g128i(s-2).detach()
            p_target=p_future[:,s]
            c_target=b['fail'][:,s].float()

            Lmain=visual_loss(o['main'],z_main)
            Lfront=(visual_loss(o['front1'],z_front)+visual_loss(o['front2'],z_front))/2
            Lrear=(visual_loss(o['rear3'],z_rear)+visual_loss(o['rear4'],z_rear))/2

            Lgf=(F.mse_loss(o['gf1'],g_front)+F.mse_loss(o['gf2'],g_front))/2
            Lgr=(F.mse_loss(o['gr3'],g_rear)+F.mse_loss(o['gr4'],g_rear))/2

            Lprop=F.mse_loss(o['pnext'],p_target)
            Lfail=F.binary_cross_entropy_with_logits(o['fail_logit'],c_target)

            loss=(
                Lmain
                + CFG.lambda_every_head*(Lfront+Lrear+Lgf+Lgr)
                + CFG.lambda_prop*Lprop
                + CFG.lambda_fail*Lfail
            )/ROLLOUT_H

        total += float(loss.detach())
        for name,val in (
            ('main_visual',Lmain),('front_visual',Lfront),('rear_visual',Lrear),
            ('front_geometry',Lgf),('rear_geometry',Lgr),
            ('proprio',Lprop),('failure',Lfail)
        ):
            logs[name] += float(val.detach())/ROLLOUT_H

        if backward:
            scaler.scale(loss).backward()

        # One coherent teacher-forcing decision for ALL streams at this step.
        use_gt=(torch.rand(raw4.shape[0],device=DEVICE)<tfp) if training \
               else torch.zeros(raw4.shape[0],dtype=torch.bool,device=DEVICE)

        def choose(gt,pred):
            m=use_gt
            while m.ndim<gt.ndim:
                m=m.unsqueeze(-1)
            return torch.where(m,gt,pred.detach())

        # Main stream
        next_main=choose(z_main,o['main'])
        raw4=torch.cat([raw4[:,1:],next_main[:,None]],1)

        # Front visual head 2: each output feeds its own branch
        front_base=torch.stack([
            choose(z_front,o['front1']),
            choose(z_front,o['front2'])
        ],1)

        # Rear visual head 3: each output feeds its own branch
        rear_base=torch.stack([
            choose(z_rear,o['rear3']),
            choose(z_rear,o['rear4'])
        ],1)

        # Front geometry head 1: each output feeds its own branch
        gfront128=torch.stack([
            choose(g_front,o['gf1']),
            choose(g_front,o['gf2'])
        ],1)
        gfront768=model.geom.to768(gfront128)

        # Rear geometry head 2
        grear128=torch.stack([
            choose(g_rear,o['gr3']),
            choose(g_rear,o['gr4'])
        ],1)
        grear768=model.geom.to768(grear128)

        # Proprioception head
        pchosen=choose(p_target,o['pnext'])
        pg,pi=pchosen[:,0],pchosen[:,1:]

    if backward:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(),CFG.grad_clip)
        scaler.step(optimizer)
        scaler.update()

    return total,dict(logs)

# %% notebook cell 12

# CELL 12 — TRAINING WITH HARD WALL-CLOCK CAP

optimizer=torch.optim.AdamW(
    model.parameters(),
    lr=CFG.lr,
    weight_decay=CFG.weight_decay
)
scaler=torch.amp.GradScaler('cuda',enabled=True)

@torch.no_grad()
def evaluate_loss(loader,epoch):
    model.eval()
    vals=[]
    for b in loader:
        L,_=rollout(b,epoch,training=False,backward=False)
        vals.append(L)
    return float(np.mean(vals)) if vals else np.nan

t0=time.time()
history=[]
best=np.inf
best_state=None
steps=0
stop=False

for epoch in range(1,CFG.epochs+1):
    model.train()
    trvals=[]

    for b in train_loader:
        if steps>=CFG.max_optimizer_steps or (time.time()-t0)/60>=CFG.max_train_minutes:
            stop=True
            break

        L,parts=rollout(b,epoch,training=True,backward=True)
        steps+=1
        trvals.append(L)

        print(
            f'epoch={epoch} step={steps} total={L:.4f} '
            f'main={parts["main_visual"]:.3f} '
            f'frontV={parts["front_visual"]:.3f} rearV={parts["rear_visual"]:.3f} '
            f'frontG={parts["front_geometry"]:.3f} rearG={parts["rear_geometry"]:.3f} '
            f'prop={parts["proprio"]:.3f} fail={parts["failure"]:.3f} '
            f'TF={teacher_probability(epoch):.2f} '
            f'elapsed={(time.time()-t0)/60:.2f}m'
        )

    vl=evaluate_loss(val_loader,epoch)
    rec=dict(
        epoch=epoch,
        optimizer_steps=steps,
        train=float(np.mean(trvals)) if trvals else np.nan,
        val=vl,
        teacher_probability=teacher_probability(epoch),
        elapsed_min=(time.time()-t0)/60
    )
    history.append(rec)
    print(rec)

    if np.isfinite(vl) and vl<best:
        best=vl
        best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

    if stop:
        break

if best_state is None:
    best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

model.load_state_dict(best_state)

torch.save({
    'model':model.state_dict(),
    'architecture':'HANDWRITTEN_FRONT_Tm2_REAR_Tm3_GEOMETRY_MULTIHEAD',
    'transformer_input':[2308,768],
    'layers':8,
    'heads':8,
    'rollout_H':8,
    'train_key':TRAIN_KEY,
    'test_key':TEST_KEY,
    'geometry_source':GEOMETRY_SOURCE,
    'best_val':best,
},OUT_ROOT/'best_handwritten_model.pt')

pd.DataFrame(history).to_csv(OUT_ROOT/'training_history.csv',index=False)

TIMING['train_s']=time.time()-t0
print(f'TRAINING TIME = {TIMING["train_s"]/60:.2f} min')

# %% notebook cell 13

# CELL 13 — FULLY RECURSIVE HELD-OUT TEST
# No teacher forcing. Every head feeds its own stream back.

@torch.no_grad()
def recursive_test(loader):
    model.eval()
    rows=[defaultdict(list) for _ in range(ROLLOUT_H)]

    for batch in loader:
        b=move(batch)
        Z=b['Zseq'].float()
        Gpts=b['Gpts'].float()
        Gstats=b['Gstats'].float()

        g128_all,g768_all=encode_geometry_sequence(Gpts,Gstats)

        def zi(rel):
            return Z[:,rel+3]

        def g128i(rel):
            return g128_all[:,rel+3]

        pg,pi=model.encode_p(b['phist'][:,0].float())

        p_future=[]
        for j in range(1,ROLLOUT_H+1):
            pgj,pij=model.encode_p(b['phist'][:,j].float())
            p_future.append(torch.cat([pgj[:,None],pij],1))
        p_future=torch.stack(p_future,1)

        raw4=torch.stack([zi(-3),zi(-2),zi(-1),zi(0)],1)
        front_base=torch.stack([zi(-2),zi(-2)],1)
        rear_base=torch.stack([zi(-3),zi(-3)],1)

        gfront128=torch.stack([g128i(-2),g128i(-2)],1)
        grear128=torch.stack([g128i(-3),g128i(-3)],1)
        gfront768=torch.stack([g768_all[:,1],g768_all[:,1]],1)
        grear768=torch.stack([g768_all[:,0],g768_all[:,0]],1)

        for s in range(ROLLOUT_H):
            o=model.forward_step(
                raw4,front_base,rear_base,
                (gfront128,gfront768),(grear128,grear768),
                pg,pi,b['actions'][:,s].float()
            )

            zmain=zi(s+1)
            zfront=zi(s-1)
            zrear=zi(s-2)
            gf=g128i(s-1)
            gr=g128i(s-2)
            pt=p_future[:,s]
            ct=b['fail'][:,s].float()

            r=rows[s]
            r['main_cos'].append(F.cosine_similarity(o['main'],zmain,dim=-1).cpu().numpy().reshape(-1))
            r['front_cos'].append(np.concatenate([
                F.cosine_similarity(o['front1'],zfront,dim=-1).cpu().numpy().reshape(-1),
                F.cosine_similarity(o['front2'],zfront,dim=-1).cpu().numpy().reshape(-1)
            ]))
            r['rear_cos'].append(np.concatenate([
                F.cosine_similarity(o['rear3'],zrear,dim=-1).cpu().numpy().reshape(-1),
                F.cosine_similarity(o['rear4'],zrear,dim=-1).cpu().numpy().reshape(-1)
            ]))
            r['gf_mae'].append(torch.abs(o['gf1']-gf).cpu().numpy().reshape(-1))
            r['gf_mae'].append(torch.abs(o['gf2']-gf).cpu().numpy().reshape(-1))
            r['gr_mae'].append(torch.abs(o['gr3']-gr).cpu().numpy().reshape(-1))
            r['gr_mae'].append(torch.abs(o['gr4']-gr).cpu().numpy().reshape(-1))
            r['p_mae'].append(torch.abs(o['pnext']-pt).cpu().numpy().reshape(-1))
            r['fail_p'].append(torch.sigmoid(o['fail_logit']).cpu().numpy())
            r['fail_y'].append(ct.cpu().numpy())

            # FULL recursive feedback, branch by branch.
            raw4=torch.cat([raw4[:,1:],o['main'][:,None]],1)

            front_base=torch.stack([o['front1'],o['front2']],1)
            rear_base=torch.stack([o['rear3'],o['rear4']],1)

            gfront128=torch.stack([o['gf1'],o['gf2']],1)
            grear128=torch.stack([o['gr3'],o['gr4']],1)
            gfront768=model.geom.to768(gfront128)
            grear768=model.geom.to768(grear128)

            pg,pi=o['pnext'][:,0],o['pnext'][:,1:]

    table=[]
    for s,r in enumerate(rows,1):
        fy=np.concatenate(r['fail_y'])
        fp=np.concatenate(r['fail_p'])
        auc=float(roc_auc_score(fy,fp)) if len(np.unique(fy))==2 else np.nan

        table.append(dict(
            rollout_step=s,
            main_visual_cos=float(np.concatenate(r['main_cos']).mean()),
            front_visual_cos=float(np.concatenate(r['front_cos']).mean()),
            rear_visual_cos=float(np.concatenate(r['rear_cos']).mean()),
            front_geometry_MAE=float(np.concatenate(r['gf_mae']).mean()),
            rear_geometry_MAE=float(np.concatenate(r['gr_mae']).mean()),
            proprio_latent_MAE=float(np.concatenate(r['p_mae']).mean()),
            failure_AUC=auc
        ))

    return pd.DataFrame(table)

t0=time.time()
TEST_METRICS=recursive_test(test_loader)
TIMING['test_s']=time.time()-t0

display(TEST_METRICS)
TEST_METRICS.to_csv(OUT_ROOT/'heldout_recursive_metrics.csv',index=False)
print(f'TEST TIME = {TIMING["test_s"]/60:.2f} min')

# %% notebook cell 14

# CELL 14 — ROUGHNESS / YAW-VIBRATION COMPARISON ON THE EXACT ROUGH TEST SEQUENCE

def evaluated_interval(seq,origins):
    f0=int(np.min(origins)-3)
    f1=int(np.max(origins)+ROLLOUT_H)
    t0=float(seq['rgb_t'][f0])
    t1=float(seq['rgb_t'][f1])
    i0=int(np.searchsorted(seq['t'],t0,side='left'))
    i1=int(np.searchsorted(seq['t'],t1,side='right'))
    return i0,i1,t0,t1

i0,i1,t_start,t_end=evaluated_interval(TEST,TEST_ORIGINS)

# Requested comparison definitions:
# roughness = std(vertical acceleration)
# yaw vibration = mean(abs(heading rate))
az=TEST['imu'][i0:i1,2]
rough=float(np.std(az))

yaw=np.unwrap(
    Rotation.from_quat(TEST['odom'][i0:i1,3:7]).as_euler('xyz')[:,2]
)
yaw_rate=np.gradient(yaw,TEST['t'][i0:i1])
yawvib=float(np.mean(np.abs(yaw_rate)))

COMPARE=pd.DataFrame([
    {
        'source':'Feel-WM paper — Isaac Sim Rough — Visual-only WM',
        'roughness_m_s2':6.46,
        'yaw_vibration_rad_s':0.73
    },
    {
        'source':'Feel-WM paper — Isaac Sim Rough — Feel-WM w/o prop.',
        'roughness_m_s2':6.53,
        'yaw_vibration_rad_s':0.77
    },
    {
        'source':'Feel-WM paper — Isaac Sim Rough — Feel-WM',
        'roughness_m_s2':6.51,
        'yaw_vibration_rad_s':0.65
    },
    {
        'source':f'OURS — TD2 rough test [{t_start-TEST["t"][0]:.2f},{t_end-TEST["t"][0]:.2f}] s',
        'roughness_m_s2':rough,
        'yaw_vibration_rad_s':yawvib
    }
])

display(COMPARE)
COMPARE.to_csv(OUT_ROOT/'rough_yaw_vs_feelwm.csv',index=False)

print('test bag:',TEST_KEY)
print('geometry source:',GEOMETRY_SOURCE)
print(f'roughness = {rough:.5f} m/s^2')
print(f'yaw vibration = {yawvib:.5f} rad/s')
print(
    'This is a scale comparison only: the paper rows are Isaac-Sim closed-loop '
    'aggregate values; ours is one fixed recorded TartanDrive2 sequence.'
)

# %% notebook cell 15

# CELL 15 — RUNTIME + ARCHITECTURE AUDIT
TIMING['total_s']=time.time()-T_ALL

runtime=pd.DataFrame([
    ['TD2 bag download',TIMING.get('download_s',np.nan)],
    ['sensor/data import',TIMING.get('stream_parse_s',np.nan)],
    ['selected RGB/geometry extraction',TIMING.get('selected_modalities_s',np.nan)],
    ['DINO extraction',TIMING.get('dino_s',np.nan)],
    ['geometry encoding prep',TIMING.get('geometry_s',np.nan)],
    ['training',TIMING.get('train_s',np.nan)],
    ['testing',TIMING.get('test_s',np.nan)],
    ['TOTAL',TIMING.get('total_s',np.nan)],
],columns=['stage','seconds'])
runtime['minutes']=runtime.seconds/60
display(runtime)
runtime.to_csv(OUT_ROOT/'runtime_report.csv',index=False)

audit={
    'architecture':'HANDWRITTEN STRICT',
    'transformer_input':[2308,768],
    'transformer_layers':8,
    'transformer_heads':8,
    'ffn':3072,
    'rollout_H':8,
    'geometry_source':GEOMETRY_SOURCE,
    'visual_streams':[
        'Z_t-3','Z_t-2','Z_t-1','Z_t',
        'FiLM(Z_t, action+pG)',
        'FiLM(Z_t-2, action+p1)',
        'FiLM(Z_t-2, action+p2)',
        'FiLM(Z_t-3, action+p3)',
        'FiLM(Z_t-3, action+p4)'
    ],
    'geometry_streams':[
        'FiLM(g_t-2, action+p1)',
        'FiLM(g_t-2, action+p2)',
        'FiLM(g_t-3, action+p3)',
        'FiLM(g_t-3, action+p4)'
    ],
    'heads':[
        'visual_head1 -> Z_t+1',
        'visual_head2 -> Z_t-1, front branches',
        'visual_head3 -> Z_t-2, rear branches',
        'geometry_head1 -> g_t-1, front branches',
        'geometry_head2 -> g_t-2, rear branches',
        'proprioception_head -> [pG,p1,p2,p3,p4]_t+1',
        'failure_head -> c_t+1'
    ],
    'recursive_feedback':'every individual head output feeds its own stream',
    'train_key':TRAIN_KEY,
    'test_key':TEST_KEY,
    'runtime_minutes':{k:float(v/60) for k,v in TIMING.items()}
}

(OUT_ROOT/'architecture_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2))
print('saved to:',OUT_ROOT)
