# Source inspection export of the adjacent notebook.
# Notebook execution order and dependencies still apply.
# IPython magic, shell lines and prose-only conclusion cells are comments.

# %% notebook cell 2
from pathlib import Path

def find_exact_bag(rec):
    expected = Path(rec["base"]) / "rosbags" / f'{rec["run"]}.bag'
    if expected.is_file():
        return expected

    roots = [
        Path("/content/tartanground_v10"),
        Path("/content/drive/MyDrive/UGP_V7_FULL4B"),
        Path("/content/drive/MyDrive/UGP_V7_FULL4B/raw"),
        Path("/content/drive/MyDrive/tartanground_v10"),
    ]

    for root in roots:
        if not root.exists():
            continue
        for bag in root.rglob(expected.name):
            if (
                bag.is_file()
                and bag.parent.name == "rosbags"
                and bag.parent.parent.name == rec["run"]
                and rec["environment"] in bag.parts
            ):
                return bag
    return None

missing = []
for rec in run_records:
    bag = find_exact_bag(rec)
    if bag is None:
        missing.append(f'{rec["environment"]}/{rec["run"]}')
    else:
        rec["base"] = bag.parents[1]
        print(rec["run_id"], "→", bag)

if missing:
    print("Missing native bags:", missing)
else:
    print("All native bags found. Rerun the native force-audit cell.")

# %% notebook cell 3
import sys, subprocess

try:
    import tartanair as ta
except Exception:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "git+https://github.com/castacks/tartanairpy.git"
    ])
    import tartanair as ta

ta.init(str(RAW_ROOT))

for e in EXPERIMENTS:
    ta.download_ground(
        env=[e["environment"]],
        version=["anymal"],
        traj=[e["run"]],
        modality=["imu", "rosbag", "image", "depth", "meta"],
        camera_name=list(CFG.geometry_cameras),
        unzip=True,
    )

# %% notebook cell 4
from __future__ import annotations

import copy, dataclasses, hashlib, importlib, json, math, os, random, shutil, subprocess, sys, time, warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

def ensure_package(import_name, pip_spec):
    try:
        return importlib.import_module(import_name)
    except Exception:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pip_spec])
        return importlib.import_module(import_name)

ensure_package('rosbags', 'rosbags>=0.10,<0.11')
ensure_package('plotly', 'plotly>=5.20,<7')
ensure_package('kaleido', 'kaleido==0.2.1')
ensure_package('joblib', 'joblib>=1.3')

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

from IPython.display import display, Markdown
from PIL import Image
from rosbags.highlevel import AnyReader
from scipy.optimize import least_squares
from scipy.spatial import cKDTree, ConvexHull
from scipy.spatial.transform import Rotation, Slerp
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    from google.colab import drive
    if not Path('/content/drive/MyDrive').exists():
        drive.mount('/content/drive')
except ImportError:
    pass

@dataclass(frozen=True)
class Config:
    implementation_version: str = 'V10R_end_to_end_20260911_v2'
    seed: int = 108781
    project_root: str = '/content/drive/MyDrive/UGP_V7_FULL4B'
    raw_root: str = '/content/tartanground_v10'
    experiment_name: str = 'V10R_REAL_LOTO_V1'
    history_s: float = 1.0
    history_dt_s: float = 0.02
    sample_hop_s: float = 0.10
    rollout_dt_s: float = 0.05
    max_horizon_s: float = 1.0
    voxel_m: float = 0.02
    pixel_stride: int = 4
    max_points_per_frame: int = 5500
    max_geometry_points_per_run: int = 1500000
    patch_radius_m: float = 0.18
    patch_points: int = 16
    patch_stats_dim: int = 32
    material_dim: int = 56
    physical_dim: int = 24
    min_geometry_confidence: float = 0.08
    proposal_batch: int = 48
    proposal_lr: float = 3e-4
    proposal_epochs: int = 120
    force_batch: int = 32
    force_lr: float = 4e-4
    force_epochs: int = 160
    patience: int = 20
    weight_decay: float = 2e-4
    force_target: str = 'mean10'
    action_mode: str = 'hold_current'
    require_native_force: bool = True
    rebuild_geometry: bool = False
    build_geometry_if_missing: bool = True
    geometry_cameras: Tuple[str, ...] = (
        'lcam_front','lcam_back','lcam_left','lcam_right','lcam_top','lcam_bottom'
    )
    outer_folds: Tuple[int, ...] = (0,1,2,3)
    fast_debug: bool = False
    allow_cpu: bool = False
    run_exact_state_ceiling: bool = True
    run_shadow_planner: bool = True

CFG = Config()

def stable_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), default=str)

CONFIG_HASH = hashlib.sha256(stable_json(asdict(CFG)).encode()).hexdigest()[:12]
PROJECT_ROOT = Path(CFG.project_root)
CACHE_ROOT = PROJECT_ROOT / 'cache'
RAW_ROOT = Path(CFG.raw_root)
V10_ROOT = PROJECT_ROOT / 'V10_geometry_coupled_force'
RUN_ROOT = PROJECT_ROOT / 'V10R_real_end_to_end' / f'{CFG.experiment_name}_{CONFIG_HASH}'
for p in [RUN_ROOT, RUN_ROOT/'audits', RUN_ROOT/'shared', RUN_ROOT/'folds', RUN_ROOT/'figures', RUN_ROOT/'tables', RUN_ROOT/'planner']:
    p.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type != 'cuda' and not CFG.allow_cpu:
    raise RuntimeError('Enable a Colab GPU runtime, or explicitly set allow_cpu=True for a very slow diagnostic run.')

random.seed(CFG.seed); np.random.seed(CFG.seed); torch.manual_seed(CFG.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(CFG.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

ROLLOUT_HORIZONS = np.round(np.arange(CFG.rollout_dt_s, CFG.max_horizon_s + 1e-9, CFG.rollout_dt_s), 3).astype(np.float32)
EVAL_HORIZONS = np.round(np.arange(0.10, CFG.max_horizon_s + 1e-9, 0.10), 2).astype(np.float32)
EVAL_HORIZON_IDS = np.array([int(np.argmin(np.abs(ROLLOUT_HORIZONS-h))) for h in EVAL_HORIZONS], np.int64)
LEG_NAMES = ('FL','FR','RL','RR')
AXIS_NAMES = ('Fx','Fy','Fz')
PHASE_NAMES = ('swing','touchdown','stance','liftoff')

(RUN_ROOT/'config.json').write_text(json.dumps({**asdict(CFG), 'config_hash':CONFIG_HASH,
    'rollout_horizons_s':ROLLOUT_HORIZONS.tolist(), 'evaluation_horizons_s':EVAL_HORIZONS.tolist()}, indent=2))
print('device       :', DEVICE)
print('config hash  :', CONFIG_HASH)
print('output root  :', RUN_ROOT)
print('rollout grid :', ROLLOUT_HORIZONS)
print('report grid  :', EVAL_HORIZONS)

# %% notebook cell 6
EXPERIMENTS = [
    dict(terrain='forest', environment='ForestEnv', run='P2000'),
    dict(terrain='swamp_marsh', environment='GreatMarsh', run='P2023'),
    dict(terrain='rough_uneven', environment='SeasonalForestWinter', run='P2000'),
    dict(terrain='wet_uneven_proxy', environment='OldTownFall', run='P2000'),
]
REQUIRED_KEYS = ('time','P','B','U','L_LEG','E','contact')

def npz_dict(path):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k] for k in z.files}

def sha256_file(path, chunk=8*1024*1024):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while True:
            b=f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

run_records=[]; provenance=[]; contract=[]
for ri,e in enumerate(EXPERIMENTS):
    pp=CACHE_ROOT/f"{e['environment']}_{e['run']}_physics.npz"
    mp=CACHE_ROOT/f"{e['environment']}_{e['run']}_frame_manifest.npz"
    if not pp.exists(): raise FileNotFoundError(f'Missing mandatory physics cache: {pp}')
    if not mp.exists(): raise FileNotFoundError(f'Missing mandatory frame manifest: {mp}')
    d=npz_dict(pp); man=npz_dict(mp)
    # Frame manifests made by the earlier V10 extractor are already on the
    # cropped physics clock.  If a user supplies an older absolute-clock
    # manifest, normalize only that obvious mismatch; never apply an arbitrary
    # offset to a manifest that is already aligned.
    _pt=np.asarray(d['time'],float); _ct=np.asarray(man.get('camera_times',[]),float)
    if len(_ct) and abs(_ct[0]-_pt[0])>10.0:
        man=dict(man); man['camera_times']=(_ct-_ct[0]+_pt[0]).astype(np.float64)
        man['time_normalization_offset_s']=float(_ct[0]-_pt[0])
    missing=[k for k in REQUIRED_KEYS if k not in d]
    if missing: raise RuntimeError(f'{pp.name} missing {missing}')
    n=len(d['time'])
    expected={'P':(n,4,12),'B':(n,12),'U':(n,6),'L_LEG':(n,4,3),'E':(n,4,3),'contact':(n,4)}
    for k,shape in expected.items():
        if d[k].shape!=shape: raise RuntimeError(f'{pp.name}:{k} expected {shape}, got {d[k].shape}')
        if not np.isfinite(d[k]).all(): raise RuntimeError(f'{pp.name}:{k} contains non-finite values')
    if np.any(np.diff(d['time'])<=0): raise RuntimeError(f'{pp.name}: time is not strictly increasing')
    for k in ('frame_ids','camera_times','poses','visual_index'):
        if k not in man: raise RuntimeError(f'{mp.name} missing {k}')
    visual={}
    for key,suffix in [('DINO','v7_dinov3'),('CONV','v7_convnext'),('FLOW','v7_waft')]:
        vp=CACHE_ROOT/f"{e['environment']}_{e['run']}_{suffix}.npy"
        if vp.exists():
            visual[key]=np.load(vp,mmap_mode='r'); provenance.append(dict(kind=key,run=ri,path=str(vp),sha256=sha256_file(vp)))
    base=RAW_ROOT/e['environment']/'Data_anymal'/e['run']
    rec={**e,'ri':ri,'run_id':f"{e['environment']}_{e['run']}",'data':d,'manifest':man,'visual':visual,'base':base}
    run_records.append(rec)
    dt=float(np.median(np.diff(d['time'])))
    contract.append(dict(ri=ri,run_id=rec['run_id'],terrain=e['terrain'],samples=n,dt_ms=1000*dt,
                         duration_s=float(d['time'][-1]-d['time'][0]),contact_fraction=float(np.mean(d['contact'])),
                         camera_frames=len(man['camera_times']),visual_streams=','.join(sorted(visual))))
    provenance += [dict(kind='physics',run=ri,path=str(pp),sha256=sha256_file(pp)),
                   dict(kind='manifest',run=ri,path=str(mp),sha256=sha256_file(mp))]

CONTRACT_DF=pd.DataFrame(contract); PROVENANCE_DF=pd.DataFrame(provenance)
display(CONTRACT_DF)
PROVENANCE_DF.to_csv(RUN_ROOT/'audits'/'input_sha256.csv',index=False)
CONTRACT_DF.to_csv(RUN_ROOT/'audits'/'cache_contract.csv',index=False)
if len(run_records)!=4: raise RuntimeError('Exactly four runs are required for this registered protocol.')
print('Physics/manifest contract passed. Visual streams are optional but their absence is recorded.')

# %% notebook cell 7


# %% notebook cell 8
from pathlib import Path
from google.colab import drive
import numpy as np

try:
    drive.flush_and_unmount()
except Exception as e:
    print("Unmount notice:", e)

drive.mount("/content/drive", force_remount=True)

p = Path("/content/drive/MyDrive/UGP_V7_FULL4B/cache/ForestEnv_P2000_physics.npz")
print("Exists:", p.exists())
print("Size:", p.stat().st_size / 1024**2, "MB")

with np.load(p, allow_pickle=False) as z:
    print("Readable. Keys:", z.files)

# %% notebook cell 10
import xml.etree.ElementTree as ET

EXT=Path('/content/v10r_external')
DESC=EXT/'anymal_d_simple_description'
URDF=DESC/'urdf'/'anymal.urdf'
if not URDF.exists():
    EXT.mkdir(parents=True,exist_ok=True)
    subprocess.check_call(['git','clone','--depth','1','https://github.com/ANYbotics/anymal_d_simple_description.git',str(DESC)])
if not URDF.exists(): raise FileNotFoundError(URDF)

def rpy_R(rpy):
    r,p,y=map(float,rpy); cr,sr=np.cos(r),np.sin(r); cp,sp=np.cos(p),np.sin(p); cy,sy=np.cos(y),np.sin(y)
    Rx=np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]],float)
    Ry=np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]],float)
    Rz=np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]],float)
    return Rz@Ry@Rx

def axis_R(axis,angle):
    a=np.asarray(axis,float); a/=np.linalg.norm(a)+1e-12; x,y,z=a; c=np.cos(angle); s=np.sin(angle); C=1-c
    return np.array([[c+x*x*C,x*y*C-z*s,x*z*C+y*s],
                     [y*x*C+z*s,c+y*y*C,y*z*C-x*s],
                     [z*x*C-y*s,z*y*C+x*s,c+z*z*C]],float)

class StrictURDFFK:
    def __init__(self,path,prefixes=('LF','RF','LH','RH')):
        root=ET.parse(path).getroot(); links={x.attrib['name'] for x in root.findall('link')}; child={}; joints={}
        for je in root.findall('joint'):
            org=je.find('origin'); ae=je.find('axis')
            d=dict(name=je.attrib['name'],type=je.attrib.get('type','fixed'),
                   parent=je.find('parent').attrib['link'],child=je.find('child').attrib['link'],
                   xyz=np.fromstring(org.attrib.get('xyz','0 0 0'),sep=' ') if org is not None else np.zeros(3),
                   rpy=np.fromstring(org.attrib.get('rpy','0 0 0'),sep=' ') if org is not None else np.zeros(3),
                   axis=np.fromstring(ae.attrib.get('xyz','1 0 0'),sep=' ') if ae is not None else np.array([1.,0,0]))
            child[d['child']]=d; joints[d['name']]=d
        roots=list(links-set(child));
        if len(roots)!=1: raise RuntimeError(f'URDF root ambiguity: {roots}')
        self.root=roots[0]; self.prefixes=prefixes; self.chains=[]; self.foot_links=[]
        for pref in prefixes:
            valid=[]
            for link in links:
                if not (link.upper().startswith(pref+'_') and 'FOOT' in link.upper()): continue
                chain=[]; cur=link
                while cur!=self.root:
                    if cur not in child: break
                    chain.append(child[cur]); cur=child[cur]['parent']
                chain=list(reversed(chain))
                names={j['name'] for j in chain}
                if all(f'{pref}_{j}' in names for j in ('HAA','HFE','KFE')): valid.append((len(chain),link,chain))
            if not valid: raise RuntimeError(f'No verified foot chain for {pref}')
            _,link,chain=max(valid); self.foot_links.append(link); self.chains.append(chain)
    @staticmethod
    def T(R=None,t=None):
        T=np.eye(4); T[:3,:3]=np.eye(3) if R is None else R; T[:3,3]=0 if t is None else t; return T
    def foot_transforms(self,q4):
        q4=np.asarray(q4,float)
        if q4.shape!=(4,3): raise ValueError(q4.shape)
        qmap={f'{p}_{j}':float(q4[i,k]) for i,p in enumerate(self.prefixes) for k,j in enumerate(('HAA','HFE','KFE'))}
        out=[]
        for chain in self.chains:
            T=np.eye(4)
            for j in chain:
                T=T@self.T(rpy_R(j['rpy']),j['xyz'])
                if j['type'] in ('revolute','continuous'): T=T@self.T(axis_R(j['axis'],qmap.get(j['name'],0.)))
                elif j['type']=='prismatic': T=T@self.T(t=j['axis']*qmap.get(j['name'],0.))
                elif j['type']!='fixed': raise RuntimeError(j['type'])
            out.append(T)
        return np.asarray(out)
    def foot_positions(self,q4): return self.foot_transforms(q4)[:,:3,3].astype(np.float32)
    def numerical_jacobian(self,q4,leg,eps=1e-4):
        q4=np.asarray(q4,float); p0=self.foot_positions(q4)[leg].astype(float); J=np.zeros((3,3))
        for k in range(3):
            q=q4.copy(); q[leg,k]+=eps; J[:,k]=(self.foot_positions(q)[leg]-p0)/eps
        return J.astype(np.float32)

FK=StrictURDFFK(URDF)
print('URDF:',URDF)
print('feet:',dict(zip(LEG_NAMES,FK.foot_links)))

R_CB_AXIS=np.diag([1.,-1.,-1.])  # base x-forward/y-left/z-up -> camera x-forward/y-right/z-down

def pose_Rwc(pose7): return Rotation.from_quat(np.asarray(pose7[3:7],float)).as_matrix()

def interp_camera_pose(rec,tq):
    tm=np.asarray(rec['manifest']['camera_times'],float); poses=np.asarray(rec['manifest']['poses'],float)
    tq=np.asarray(tq,float); tc=np.clip(tq,tm[0],tm[-1]); pos=np.column_stack([np.interp(tc,tm,poses[:,k]) for k in range(3)])
    rot=Slerp(tm,Rotation.from_quat(poses[:,3:7]))(tc).as_matrix()
    return pos,rot

def heading_Rwh(R_WB):
    # Columns are heading axes expressed in Tartan NED world coordinates.
    x=np.asarray(R_WB)[..., :,0].copy(); x[...,2]=0.; x/=np.linalg.norm(x,axis=-1,keepdims=True)+1e-12
    z=np.zeros_like(x); z[...,2]=-1.0
    y=np.cross(z,x); y/=np.linalg.norm(y,axis=-1,keepdims=True)+1e-12
    z=np.cross(x,y); z/=np.linalg.norm(z,axis=-1,keepdims=True)+1e-12
    return np.stack([x,y,z],axis=-1)

def Tcb_from_x(x): return Rotation.from_rotvec(np.asarray(x[:3],float)).as_matrix()@R_CB_AXIS,np.asarray(x[3:6],float)

def calibrate_tcb(train_ris,max_pairs=4000):
    pairs=[]
    for ri in train_ris:
        rec=run_records[ri]; d=rec['data']; tm=np.asarray(rec['manifest']['camera_times'],float); poses=np.asarray(rec['manifest']['poses'],float)
        pidx=np.clip(np.searchsorted(d['time'],tm,side='left'),0,len(d['time'])-1)
        for k in range(1,len(pidx)):
            i0,i1=int(pidx[k-1]),int(pidx[k])
            for leg in range(4):
                if d['contact'][i0,leg]>.5 and d['contact'][i1,leg]>.5:
                    pairs.append((poses[k-1],poses[k],FK.foot_positions(d['P'][i0,:,:3])[leg],FK.foot_positions(d['P'][i1,:,:3])[leg]))
    if len(pairs)>max_pairs:
        rng=np.random.default_rng(CFG.seed+sum(train_ris)); pairs=[pairs[i] for i in rng.choice(len(pairs),max_pairs,False)]
    if len(pairs)<200: raise RuntimeError(f'Only {len(pairs)} stance pairs for calibration')
    def residual(x):
        Rcb,tcb=Tcb_from_x(x); out=[]
        for p0,p1,b0,b1 in pairs:
            w0=pose_Rwc(p0)@(Rcb@b0+tcb)+p0[:3]; w1=pose_Rwc(p1)@(Rcb@b1+tcb)+p1[:3]
            out.extend((w1-w0).tolist())
        out.extend((.03*np.asarray(x[:3])).tolist()); out.extend((.01*np.asarray(x[3:])).tolist())
        return np.asarray(out)
    sol=least_squares(residual,np.zeros(6),loss='soft_l1',f_scale=.03,
                      bounds=([-0.6]*3+[-1.5]*3,[0.6]*3+[1.5]*3),max_nfev=250)
    before=np.median(np.linalg.norm(residual(np.zeros(6))[:-6].reshape(-1,3),axis=1))
    after=np.median(np.linalg.norm(residual(sol.x)[:-6].reshape(-1,3),axis=1))
    return sol.x.astype(np.float64),dict(pairs=len(pairs),median_stationarity_before_m=float(before),median_stationarity_after_m=float(after),cost=float(sol.cost))

# A numerical handedness test: x cross y must equal z for every constructed heading frame.
Rtest=heading_Rwh(np.eye(3)); handed=float(np.dot(np.cross(Rtest[:,0],Rtest[:,1]),Rtest[:,2]))
if handed<0.999: raise RuntimeError(f'Heading frame is not right-handed: {handed}')
print('heading handedness:',handed,'(x forward, y left, z up)')

# %% notebook cell 12
from pathlib import Path

# Preserve the complete four-run list for later restoration
if "ALL_RUN_RECORDS" not in globals():
    ALL_RUN_RECORDS = list(run_records)

# Process only terrains whose raw ROS bag still exists
run_records = [
    rec for rec in ALL_RUN_RECORDS
    if (Path(rec["base"]) / "rosbags" / f'{rec["run"]}.bag').exists()
]

print("Runs available for this staged pass:")
for rec in run_records:
    print(" ", rec["run_id"])

# %% notebook cell 13
FORCE_TOPICS = {
    'FL':'/state_estimator/contact_force_lf_foot',
    'FR':'/state_estimator/contact_force_rf_foot',
    'RL':'/state_estimator/contact_force_lh_foot',
    'RR':'/state_estimator/contact_force_rh_foot',
}

def stamp_seconds(st):
    if st is None: return np.nan
    if hasattr(st,'sec'): return float(st.sec)+1e-9*float(getattr(st,'nanosec',getattr(st,'nsec',0)))
    if hasattr(st,'secs'): return float(st.secs)+1e-9*float(getattr(st,'nsecs',0))
    return np.nan

def force_topic_audit(rec):
    bag=rec['base']/ 'rosbags' / f"{rec['run']}.bag"
    if not bag.exists(): raise FileNotFoundError(f'Raw ROS bag required for native force audit: {bag}')
    rows=[]; wanted=set(FORCE_TOPICS.values())
    with AnyReader([bag]) as reader:
        conns=[c for c in reader.connections if c.topic in wanted]
        for conn,bns,raw in reader.messages(connections=conns):
            msg=reader.deserialize(raw,conn.msgtype); h=getattr(msg,'header',None)
            rows.append(dict(topic=conn.topic,frame=str(getattr(h,'frame_id','')),bag_t=float(bns)*1e-9,
                             header_t=stamp_seconds(getattr(h,'stamp',None))))
    x=pd.DataFrame(rows)
    if x.empty: raise RuntimeError(f'No force messages found in {bag}')
    out=[]
    for leg,topic in FORCE_TOPICS.items():
        z=x[x.topic==topic]; frames=sorted(set(f for f in z.frame if f))
        bt=z.bag_t.to_numpy(); ht=z.header_t.to_numpy(); off=ht[np.isfinite(ht)]-bt[np.isfinite(ht)]
        out.append(dict(run_id=rec['run_id'],leg=leg,samples=len(z),frame_ids='|'.join(frames),
                        median_dt_ms=float(1000*np.median(np.diff(bt))) if len(bt)>1 else np.nan,
                        header_minus_bag_ms=float(1000*np.median(off)) if len(off) else np.nan,
                        header_bag_jitter_ms=float(1000*np.std(off)) if len(off) else np.nan))
    return pd.DataFrame(out)

FORCE_FRAME_AUDIT=pd.concat([force_topic_audit(r) for r in run_records],ignore_index=True)
display(FORCE_FRAME_AUDIT)
FORCE_FRAME_AUDIT.to_csv(RUN_ROOT/'audits'/'force_frame_audit.csv',index=False)

def frame_mode(frame):
    s=str(frame).lower().strip()
    if not s: return 'unknown'
    if any(k in s for k in ('foot','lf_','rf_','lh_','rh_','fl_','fr_','rl_','rr_')): return 'foot'
    if any(k in s for k in ('camera','cam','lcam','rcam')): return 'camera'
    if any(k in s for k in ('odom','world','map','ned')): return 'world'
    if any(k in s for k in ('base','body','imu')): return 'base'
    return 'unknown'

def interp_q(rec,t_original):
    d=rec['data']; t=d['time']-float(d['time'][0])
    tq=np.asarray(t_original,float); j=np.searchsorted(t,tq,side='right')-1; j=np.clip(j,0,len(t)-1); return d['P'][j,:,:3]

def trailing_mean(ts,x,grid,window=0.010):
    ts=np.asarray(ts,float); x=np.asarray(x,float); order=np.argsort(ts); ts=ts[order]; x=x[order]
    cs=np.concatenate([np.zeros((1,x.shape[1])),np.cumsum(x,axis=0)],axis=0)
    r=np.searchsorted(ts,grid,side='right'); l=np.searchsorted(ts,grid-window,side='right'); n=r-l
    out=np.zeros((len(grid),x.shape[1]),np.float32); good=n>0
    if good.any(): out[good]=((cs[r[good]]-cs[l[good]])/n[good,None]).astype(np.float32)
    if (~good).any():
        ii=np.clip(np.searchsorted(ts,grid[~good],side='right')-1,0,len(ts)-1); out[~good]=x[ii]
    return out

def load_native_force(rec):
    bag=rec['base']/ 'rosbags' / f"{rec['run']}.bag"
    physical=set(FORCE_TOPICS.values())|{'/state_estimator/joint_states','/state_estimator/twist'}
    with AnyReader([bag]) as reader:
        conns=[c for c in reader.connections if c.topic in physical]
        first=next(reader.messages(connections=conns)); t0=float(first[1])*1e-9
    streams={leg:dict(t=[],f=[],frame=[]) for leg in LEG_NAMES}
    topic_to_leg={v:k for k,v in FORCE_TOPICS.items()}
    with AnyReader([bag]) as reader:
        conns=[c for c in reader.connections if c.topic in set(FORCE_TOPICS.values())]
        for conn,bns,raw in reader.messages(connections=conns):
            msg=reader.deserialize(raw,conn.msgtype); leg=topic_to_leg[conn.topic]; h=getattr(msg,'header',None)
            bt=float(bns)*1e-9; ht=stamp_seconds(getattr(h,'stamp',None))
            # Prefer the bag clock for alignment when a producer stamped the
            # message with a different epoch.  Header time is used only when
            # it agrees with the bag clock to 5 ms, and the audit records the
            # discrepancy for publication.
            tt=((ht if abs(ht-bt)<=.005 else bt) if np.isfinite(ht) else bt)-t0
            w=msg.wrench.force; streams[leg]['t'].append(tt); streams[leg]['f'].append([w.x,w.y,w.z]); streams[leg]['frame'].append(str(getattr(h,'frame_id','')))
    for leg in LEG_NAMES:
        order=np.argsort(streams[leg]['t']); streams[leg]['t']=np.asarray(streams[leg]['t'],float)[order]; streams[leg]['f']=np.asarray(streams[leg]['f'],np.float32)[order]; streams[leg]['frame']=np.asarray(streams[leg]['frame'],str)[order]
        if len(streams[leg]['t'])<10: raise RuntimeError(f'{rec["run_id"]}/{leg}: too few native force samples')
        if not np.isfinite(streams[leg]['t']).all() or not np.isfinite(streams[leg]['f']).all(): raise RuntimeError(f'{rec["run_id"]}/{leg}: non-finite native force sample')
    return streams

def build_force_targets(rec):
    d=rec['data']; grid=np.asarray(d['time'],float)-float(d['time'][0])
    streams=load_native_force(rec); raw=[]; mean10=[]; modes=[]; frames=[]
    for li,leg in enumerate(LEG_NAMES):
        s=streams[leg]; dominant=pd.Series(s['frame']).replace('',np.nan).dropna().mode()
        fr=str(dominant.iloc[0]) if len(dominant) else ''; mode=frame_mode(fr)
        if mode=='unknown': raise RuntimeError(f'{rec["run_id"]}/{leg}: unresolved force frame {fr!r}')
        observed_modes={frame_mode(x) for x in s['frame'] if str(x).strip()}
        observed_modes.discard('unknown')
        if len(observed_modes)>1:
            raise RuntimeError(f'{rec["run_id"]}/{leg}: mixed native force frames {sorted(observed_modes)}')
        # Rotate foot-local vectors into the instantaneous base frame before averaging.
        if mode=='foot':
            qn=interp_q(rec,s['t']); fb=[]
            # StrictURDFFK returns the homogeneous transform ^B T_F (the
            # foot-link frame expressed in the base frame).  A force row
            # vector therefore maps F_F -> F_B with R_BF.T.
            for k in range(len(s['t'])): fb.append(s['f'][k] @ FK.foot_transforms(qn[k])[li,:3,:3].T)
            src=np.asarray(fb,np.float32)
        else: src=s['f']
        raw.append(src[np.clip(np.searchsorted(s['t'],grid,side='right')-1,0,len(src)-1)])
        mean10.append(trailing_mean(s['t'],src,grid))
        modes.append('base' if mode=='foot' else mode); frames.append(fr)
    out=dict(F_raw=np.stack(raw,1).astype(np.float32),F_mean10=np.stack(mean10,1).astype(np.float32),
             source_modes=np.asarray(modes),source_frames=np.asarray(frames),
             label_version=np.asarray('V10R_FORCE_V2',dtype='<U16'))
    # Contact-positive sign check is physical, not selected by test error. NED world z is down.
    c=np.asarray(d['contact'])>.5
    for li,mode in enumerate(modes):
        if mode=='base': vertical=float(np.median(out['F_mean10'][c[:,li],li,2]))
        elif mode in ('world','camera'): vertical=float(np.median(-out['F_mean10'][c[:,li],li,2]))
        else: vertical=np.nan
        if np.isfinite(vertical) and vertical < -1.0:
            raise RuntimeError(f'{rec["run_id"]}/{LEG_NAMES[li]} force sign is inconsistent with contact-positive convention: median vertical {vertical:.3f} N')
    return out

FORCE_LABELS={}; force_rows=[]
for rec in run_records:
    fp=RUN_ROOT/'shared'/f"{rec['run_id']}_force_labels.npz"
    if fp.exists():
        z=np.load(fp,allow_pickle=False); out={k:z[k] for k in z.files}
        if str(out.get('label_version',''))!='V10R_FORCE_V2':
            out=build_force_targets(rec); np.savez_compressed(fp,**out)
    else: out=build_force_targets(rec); np.savez_compressed(fp,**out)
    FORCE_LABELS[rec['ri']]=out
    e=np.abs(out['F_raw']-rec['data']['P'][:,:,9:12]); step=np.linalg.norm(np.diff(out['F_raw'],axis=0),axis=-1).ravel(); step10=np.linalg.norm(np.diff(out['F_mean10'],axis=0),axis=-1).ravel()
    # The legacy P[:,:,9:12] cache is documented in the base frame.  Only
    # compare against it after all four native streams have been established in
    # that frame; otherwise a frame mismatch would masquerade as label noise.
    e=np.abs(out['F_raw']-rec['data']['P'][:,:,9:12]) if all(str(m)=='base' for m in out['source_modes']) else np.full(1,np.nan)
    force_rows.append(dict(run_id=rec['run_id'],raw_vs_cached_mae_N=float(np.nanmean(e)),raw_adjacent_p95_N=float(np.quantile(step,.95)),mean10_adjacent_p95_N=float(np.quantile(step10,.95)),mean10_adjacent_max_N=float(step10.max()),source_modes='|'.join(out['source_modes']),legacy_cache_comparison='base-frame only'))
FORCE_AUDIT=pd.DataFrame(force_rows); display(FORCE_AUDIT); FORCE_AUDIT.to_csv(RUN_ROOT/'audits'/'force_label_audit.csv',index=False)
print('Native force target ready. Primary:',CFG.force_target,'raw stress target retained.')

# %% notebook cell 14
# Read-only schema check; does not modify labels
for rec in run_records:
    bag = rec['base'] / 'rosbags' / f"{rec['run']}.bag"
    seen = set()

    with AnyReader([bag]) as reader:
        conns = [
            c for c in reader.connections
            if c.topic in set(FORCE_TOPICS.values())
        ]

        for conn, bag_ns, raw in reader.messages(connections=conns):
            if conn.topic in seen:
                continue

            msg = reader.deserialize(raw, conn.msgtype)
            h = getattr(msg, 'header', None)

            print("\nRUN:", rec['run_id'])
            print("TOPIC:", conn.topic)
            print("TYPE:", conn.msgtype)
            print("header.frame_id:", repr(getattr(h, 'frame_id', None)))
            print("direct frame_id:", repr(getattr(msg, 'frame_id', None)))
            print("child_frame_id:", repr(getattr(msg, 'child_frame_id', None)))
            print("message fields:", getattr(msg, '__slots__', None))

            seen.add(conn.topic)
            if len(seen) == len(conns):
                break

# %% notebook cell 15
# DEBUG ONLY — NOT A NATIVE-FORCE RESULT
# The notebook documents P[:,:,9:12] as base-frame cache data.

FORCE_LABELS = {}
force_rows = []

for rec in run_records:
    d = rec['data']
    F_raw = np.asarray(d['P'][:, :, 9:12], np.float32)
    t = np.asarray(d['time'], float)
    t = t - t[0]

    F_mean10 = np.stack([
        trailing_mean(t, F_raw[:, li], t, window=0.010)
        for li in range(4)
    ], axis=1).astype(np.float32)

    out = {
        'F_raw': F_raw,
        'F_mean10': F_mean10,
        'source_modes': np.full(4, 'base', dtype='<U16'),
        'source_frames': np.full(
            4, 'CACHE_P_BASE_DECLARED', dtype='<U64'
        ),
        'label_version': np.asarray(
            'V10R_CACHE_DEBUG_V1', dtype='<U32'
        ),
    }

    FORCE_LABELS[rec['ri']] = out
    force_rows.append({
        'run_id': rec['run_id'],
        'native_force': False,
        'frame_status': 'missing_in_ros_header',
        'target': 'P[:,:,9:12]',
    })

FORCE_AUDIT = pd.DataFrame(force_rows)
display(FORCE_AUDIT)

print("DEBUG cache labels loaded.")
print("Run cell 12 next, then continue from cell 14.")
print("Do not use these outputs as native-force paper results.")

# %% notebook cell 16
from pathlib import Path
import shutil, subprocess

DRIVE_ROOT = Path("/content/drive/MyDrive/UGP_V7_FULL4B")
LOCAL_RAW = Path("/content/tartanground_v10")
LOCAL_EXTERNAL = Path("/content/v10r_external")

def size_gb(p):
    if not p.exists():
        return 0.0
    if p.is_file():
        return p.stat().st_size / 1024**3
    return sum(x.stat().st_size for x in p.rglob("*") if x.is_file()) / 1024**3

print("Storage:")
subprocess.run(["du", "-sh", "/content", "/content/drive/MyDrive"], check=False)

print("\nLikely downloaded locations:")
for p in [LOCAL_RAW, LOCAL_EXTERNAL, DRIVE_ROOT / "V10R_real_end_to_end"]:
    print(f"{size_gb(p):8.3f} GB   {p}")

print("\nLargest Drive directories:")
subprocess.run(
    f"du -h --max-depth=2 '{DRIVE_ROOT}' 2>/dev/null | sort -h | tail -30",
    shell=True,
)

# Keep False for inspection. Change to True only if you want to remove
# the temporary deviated download.
DELETE_TEMPORARY_RAW = False

if DELETE_TEMPORARY_RAW:
    targets = [p for p in [LOCAL_RAW, LOCAL_EXTERNAL] if p.exists()]

    for p in targets:
        if p.is_symlink():
            raise RuntimeError(f"Refusing to delete symlink: {p}")

    print("\nDeleting:")
    for p in targets:
        print(f"{size_gb(p):.3f} GB   {p}")
        shutil.rmtree(p)

    print("Temporary deviated downloads deleted.")
else:
    print("\nNothing deleted. Set DELETE_TEMPORARY_RAW=True after checking the sizes.")

# %% notebook cell 17
from pathlib import Path
import shutil

RAW_ROOT = Path("/content/tartanground_v10")
PROJECT_ROOT = Path("/content/drive/MyDrive/UGP_V7_FULL4B")

CONFIRM_DELETE = False
DELETE_CHECKPOINTED_RAW = False

RUNS = [
    ("ForestEnv", "P2000"),
    ("GreatMarsh", "P2023"),
    ("SeasonalForestWinter", "P2000"),
    ("OldTownFall", "P2000"),
]

def size_gb(p):
    if p.is_file():
        return p.stat().st_size / 1024**3
    return sum(x.stat().st_size for x in p.rglob("*") if x.is_file()) / 1024**3

targets = []

# Temporary download leftovers only
for p in RAW_ROOT.rglob("*"):
    if p.is_file() and p.suffix.lower() in {".part", ".tmp", ".download"}:
        targets.append(p)

# IMU is not read by deviated_run after the cached physics/manifest files exist
for env, run in RUNS:
    physics = PROJECT_ROOT / "cache" / f"{env}_{run}_physics.npz"
    manifest = PROJECT_ROOT / "cache" / f"{env}_{run}_frame_manifest.npz"
    imu = RAW_ROOT / env / "Data_anymal" / run / "imu"

    if physics.exists() and manifest.exists() and imu.exists():
        targets.append(imu)

# After cells 10 and 12 finish, remove raw data whose checkpoints are complete
if DELETE_CHECKPOINTED_RAW:
    run_roots = list((PROJECT_ROOT / "V10R_real_end_to_end").glob("V10R_REAL_LOTO_*"))

    for env, run in RUNS:
        run_id = f"{env}_{run}"
        force_done = any(
            (r / "shared" / f"{run_id}_force_labels.npz").exists()
            for r in run_roots
        )
        geometry_done = any(
            all((r / "shared" / "geometry_events" / run_id / f"{k}.npy").exists()
                for k in ("xyz", "normal", "rgb", "grid", "frame", "time", "camera"))
            for r in run_roots
        )

        if force_done and geometry_done:
            raw_run = RAW_ROOT / env / "Data_anymal" / run
            targets.extend(raw_run.glob("image_*"))
            targets.extend(raw_run.glob("depth_*"))
            targets.append(raw_run / "rosbags")
            targets.extend(raw_run.glob("pose_*.txt"))

print("Deletion candidates:")
total = 0.0
for p in targets:
    if p.exists():
        s = size_gb(p)
        total += s
        print(f"{s:8.3f} GB   {p}")

print(f"\nTotal: {total:.3f} GB")

if not CONFIRM_DELETE:
    print("\nDRY RUN — nothing deleted.")
    print("Set CONFIRM_DELETE=True and rerun.")
else:
    for p in targets:
        if not p.exists():
            continue
        if p.is_symlink():
            raise RuntimeError(f"Refusing symlink: {p}")
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()

    print("Cleanup completed.")

# %% notebook cell 19
# SAFE CONTINUATION — no deletion and no fake checkpoints

from pathlib import Path

if "ALL_RUN_RECORDS" not in globals():
    ALL_RUN_RECORDS = list(run_records)

run_records = [
    rec for rec in ALL_RUN_RECORDS
    if (
        Path(rec["base"])
        / "rosbags"
        / f'{rec["run"]}.bag'
    ).exists()
]

print("Skipping deletion guard.")
print("Available raw runs:")
for rec in run_records:
    print(" ", rec["run_id"])

print("ForestEnv/P2000 is omitted because its raw data is already absent.")

# %% notebook cell 20
# SAFE CONTINUATION — no deletion and no fake checkpoints

from pathlib import Path

if "ALL_RUN_RECORDS" not in globals():
    ALL_RUN_RECORDS = list(run_records)

run_records = [
    rec for rec in ALL_RUN_RECORDS
    if (
        Path(rec["base"])
        / "rosbags"
        / f'{rec["run"]}.bag'
    ).exists()
]

print("Skipping deletion guard.")
print("Available raw runs:")
for rec in run_records:
    print(" ", rec["run_id"])

print("ForestEnv/P2000 is omitted because its raw data is already absent.")

# %% notebook cell 21
CAM_W=CAM_H=640; FX=FY=CX=CY=320.0

def read_depth(path):
    raw=np.ascontiguousarray(cv2.imread(str(path),cv2.IMREAD_UNCHANGED))
    if raw is None: raise RuntimeError(f'Cannot read depth {path}')
    if raw.dtype==np.uint8 and raw.ndim==3 and raw.shape[2]==4: d=raw.view('<f4').reshape(raw.shape[:2])
    elif raw.dtype==np.float32 and raw.ndim==2: d=raw
    else: raise RuntimeError(f'Unexpected depth encoding {raw.dtype} {raw.shape}')
    d=d.astype(np.float32); d[~np.isfinite(d)]=0.; return d

def manifest_frame_path(base,kind,cam,fid):
    if kind=='depth': return base/f'depth_{cam}'/f'{fid}_{cam}_depth.png'
    return base/f'image_{cam}'/f'{fid}_{cam}.png'

def sample_rgbd_events(rec,frame_idx,cam):
    man=rec['manifest']; base=rec['base']; fid=str(man['frame_ids'][frame_idx]); fn=int(man.get('frame_nums',man['frame_ids'])[frame_idx])
    dp=manifest_frame_path(base,'depth',cam,fid); ip=manifest_frame_path(base,'image',cam,fid); pp=base/f'pose_{cam}.txt'
    if not (dp.exists() and ip.exists() and pp.exists()): return None
    depth=read_depth(dp); rgb=np.asarray(Image.open(ip).convert('RGB'))
    pose_cache=rec.setdefault('_pose_cache',{})
    if cam not in pose_cache: pose_cache[cam]=np.loadtxt(pp).astype(np.float64)
    poses=pose_cache[cam]
    if fn>=len(poses): return None
    pose=poses[fn]; Rwc=pose_Rwc(pose); yy=np.arange(1,depth.shape[0]-1,CFG.pixel_stride); xx=np.arange(1,depth.shape[1]-1,CFG.pixel_stride); v,u=np.meshgrid(yy,xx,indexing='ij'); Z=depth[v,u]
    valid=np.isfinite(Z)&(Z>.25)&(Z<8.0); pc=np.stack([Z,(u-CX)*Z/FX,(v-CY)*Z/FY],-1).astype(np.float32)
    dx=np.zeros_like(pc); dy=np.zeros_like(pc); dx[:,1:-1]=pc[:,2:]-pc[:,:-2]; dy[1:-1]=pc[2:]-pc[:-2]; nc=np.cross(dx,dy); nc/=np.linalg.norm(nc,axis=-1,keepdims=True)+1e-8
    world=pc.reshape(-1,3)@Rwc.T+pose[:3]; normals=nc.reshape(-1,3)@Rwc.T; normals/=np.linalg.norm(normals,axis=1,keepdims=True)+1e-8; cols=rgb[v,u].reshape(-1,3)
    # Keep a robot-local causal map volume around the current pose.
    p0=pose[:3]; r=np.linalg.norm(world-p0[None,:],axis=1); valid &= (r.reshape(Z.shape)<8.0)
    ids=np.flatnonzero(valid.ravel());
    if len(ids)==0: return None
    if len(ids)>CFG.max_points_per_frame:
        rng=np.random.default_rng(CFG.seed+frame_idx+sum(map(ord,cam))); ids=rng.choice(ids,CFG.max_points_per_frame,replace=False)
    # grid index links front-camera surfels to cached visual tokens.
    if cam=='lcam_front':
        gx=np.clip((u.ravel()[ids].astype(np.float32)/CAM_W*8).astype(np.int16),0,7); gy=np.clip((v.ravel()[ids].astype(np.float32)/CAM_H*8).astype(np.int16),0,7); grid=(gy*8+gx).astype(np.int16)
    else: grid=np.full(len(ids),-1,np.int16)
    return world[ids].astype(np.float32),normals[ids].astype(np.float32),cols[ids].astype(np.uint8),grid,np.full(len(ids),frame_idx,np.int32),np.full(len(ids),float(man['camera_times'][frame_idx]),np.float32),np.full(len(ids),sum(map(ord,cam)),np.int16)

def compact_geometry_events(root):
    '''Bound the raw event store without breaking causality.

    The six-camera teacher can contain tens of millions of surfels.  Keeping
    the first observation in each 2-cm world voxel preserves a valid causal
    timestamp (and static-ground appearance) while making the four-fold KD
    trees feasible in a Colab RAM budget.  If the voxel count is still above
    the registered cap, a deterministic subset is retained.
    '''
    meta_path=Path(root)/'meta.json'
    try: meta=json.loads(meta_path.read_text()) if meta_path.exists() else {}
    except Exception: meta={}
    if meta.get('compaction')=='first_observation_per_voxel' and int(meta.get('compacted_points',0))<=CFG.max_geometry_points_per_run: return
    xyz=np.load(Path(root)/'xyz.npy',mmap_mode='r'); n=len(xyz)
    if n==0: return
    vox=np.floor(np.asarray(xyz,np.float32)/CFG.voxel_m).astype(np.int64)
    tm=np.asarray(np.load(Path(root)/'time.npy',mmap_mode='r'),np.float32)
    order=np.lexsort((np.arange(n,dtype=np.int64),tm,vox[:,2],vox[:,1],vox[:,0])); sv=vox[order]; keep=np.r_[True,np.any(sv[1:]!=sv[:-1],axis=1)]; ids=order[keep]
    if len(ids)>CFG.max_geometry_points_per_run:
        rng=np.random.default_rng(CFG.seed+len(ids)); ids=np.sort(rng.choice(ids,CFG.max_geometry_points_per_run,replace=False))
    ids=np.asarray(ids,np.int64)
    for k in ('xyz','normal','rgb','grid','frame','time','camera'):
        src=Path(root)/f'{k}.npy'; arr=np.load(src,mmap_mode='r'); tmp=Path(root)/f'{k}.compact.npy'; np.save(tmp,np.asarray(arr[ids])); os.replace(tmp,src)
    meta.update(compaction='first_observation_per_voxel',raw_points=int(n),compacted_points=int(len(ids)),voxel_m=CFG.voxel_m,causal_timestamp='earliest observation retained')
    meta_path.write_text(json.dumps(meta,indent=2))

def build_geometry_events(rec):
    root=RUN_ROOT/'shared'/'geometry_events'/rec['run_id']; root.mkdir(parents=True,exist_ok=True)
    required=('xyz','normal','rgb','grid','frame','time','camera')
    if not CFG.rebuild_geometry and all((root/f'{k}.npy').exists() for k in required): compact_geometry_events(root); return root
    # Reuse the earlier V10 RGB-D event store when available; it was generated from the same 640px depth contract.
    legacy=V10_ROOT/'geometry_memory'/rec['run_id']
    if not CFG.rebuild_geometry and all((legacy/f'{k}.npy').exists() for k in ('xyz','normal','rgb','grid','frame','time')):
        for k in ('xyz','normal','rgb','grid','frame','time'): shutil.copy2(legacy/f'{k}.npy',root/f'{k}.npy')
        np.save(root/'camera.npy',np.full(len(np.load(root/'xyz.npy',mmap_mode='r')),sum(map(ord,'legacy')),np.int16))
        (root/'meta.json').write_text(json.dumps(dict(source='legacy_v10_rgbd_events',voxel_m=CFG.voxel_m),indent=2)); compact_geometry_events(root); return root
    if not CFG.build_geometry_if_missing: raise FileNotFoundError(f'No geometry event store and build_geometry_if_missing=False for {rec["run_id"]}')
    arrays={k:[] for k in required}; n=len(rec['manifest']['frame_ids']); cams=CFG.geometry_cameras
    print('Building RGB-D event store',rec['run_id'],'frames',n,'cameras',cams)
    for j in range(n):
        for cam in cams:
            out=sample_rgbd_events(rec,j,cam)
            if out is None: continue
            xyz,norm,rgb,grid,frame,tm,camera=out
            for k,v in zip(required,(xyz,norm,rgb,grid,frame,tm,camera)): arrays[k].append(v)
        if (j+1)%100==0: print(' ',j+1,'/',n)
    if not arrays['xyz']: raise RuntimeError(f'No RGB-D events built for {rec["run_id"]}; check raw RGB/depth/pose paths.')
    for k in required: np.save(root/f'{k}.npy',np.concatenate(arrays[k],axis=0))
    (root/'meta.json').write_text(json.dumps(dict(source='rgbd_deprojection',voxel_m=CFG.voxel_m,pixel_stride=CFG.pixel_stride,points=int(len(np.load(root/'xyz.npy',mmap_mode='r')))),indent=2)); compact_geometry_events(root); return root

class CausalVoxelSurfaceMap:
    def __init__(self,root):
        self.root=Path(root); self.xyz=np.load(self.root/'xyz.npy',mmap_mode='r'); self.normal=np.load(self.root/'normal.npy',mmap_mode='r'); self.rgb=np.load(self.root/'rgb.npy',mmap_mode='r'); self.grid=np.load(self.root/'grid.npy',mmap_mode='r'); self.frame=np.load(self.root/'frame.npy',mmap_mode='r'); self.time=np.load(self.root/'time.npy',mmap_mode='r'); self.camera=np.load(self.root/'camera.npy',mmap_mode='r') if (self.root/'camera.npy').exists() else np.zeros(len(self.xyz),np.int16); self.tree=cKDTree(np.asarray(self.xyz[:,:2],np.float32)); self.voxel=CFG.voxel_m
    def _fuse(self,ids):
        if len(ids)==0: return dict(xyz=np.zeros((0,3),np.float32),normal=np.zeros((0,3),np.float32),rgb=np.zeros((0,3),np.float32),grid=np.zeros(0,np.int16),frame=np.zeros(0,np.int32),time=np.zeros(0,np.float32),count=np.zeros(0,np.float32))
        xyz=np.asarray(self.xyz[ids],np.float32); vox=np.floor(xyz/self.voxel).astype(np.int64); uv,inv=np.unique(vox,axis=0,return_inverse=True); m=len(uv); cnt=np.bincount(inv,minlength=m).astype(np.float32); out={}
        for name,arr in [('xyz',xyz),('normal',np.asarray(self.normal[ids],np.float32)),('rgb',np.asarray(self.rgb[ids],np.float32)/255.)]: out[name]=np.asarray([arr[inv==k].mean(0) for k in range(m)],np.float32)
        out['normal']/=np.linalg.norm(out['normal'],axis=1,keepdims=True)+1e-8; out['count']=cnt; out['time']=np.asarray([np.max(self.time[ids][inv==k]) for k in range(m)],np.float32); out['frame']=np.asarray([self.frame[ids][np.flatnonzero(inv==k)[0]] for k in range(m)],np.int32); out['grid']=np.asarray([self.grid[ids][np.flatnonzero(inv==k)[0]] for k in range(m)],np.int16); return out
    def query(self,mean_world,cov_world,origin_t,R_WH):
        mean=np.asarray(mean_world,float); cov=np.asarray(cov_world,float); rad=float(CFG.patch_radius_m+3*np.sqrt(max(np.linalg.eigvalsh(cov).max(),1e-8)))
        ids=np.asarray(self.tree.query_ball_point(mean[:2],r=rad),np.int64); ids=ids[np.asarray(self.time[ids])<=float(origin_t)+1e-9]
        if len(ids)==0: return self.empty()
        d3=np.linalg.norm(np.asarray(self.xyz[ids],float)-mean[None,:],axis=1); ids=ids[np.argsort(d3)[:2048]]; f=self._fuse(ids); xyz=f['xyz']; normal=f['normal']; rgb=f['rgb'];
        local=(xyz-mean[None,:])@np.asarray(R_WH,float); nl=normal@np.asarray(R_WH,float); order=np.argsort(np.linalg.norm(local,axis=1)); local=local[order]; nl=nl[order]; rgb=rgb[order]; cnt=f['count'][order]; ages=np.maximum(0,float(origin_t)-f['time'][order]);
        # Surface statistics are computed in the query heading frame.
        z=local[:,2]; xy=local[:,:2]; normal_mean=nl[:min(len(nl),64)].mean(0); normal_mean/=np.linalg.norm(normal_mean)+1e-8
        # Contact support is represented by an upward normal in the canonical
        # heading frame.  Depth cross-products have an arbitrary orientation;
        # leave the geometry unchanged but orient the support normal explicitly.
        if normal_mean[2] < 0.: normal_mean=-normal_mean; nl=-nl
        slope=float(np.arccos(np.clip(abs(normal_mean[2]),0,1)))
        rough=float(np.std(z)); curvature=np.zeros(2); residual=rough;
        if len(xy)>=8:
            X=np.c_[xy[:,0]**2,xy[:,1]**2,xy[:,0]*xy[:,1],xy[:,0],xy[:,1],np.ones(len(xy))]
            try:
                co=np.linalg.lstsq(X,z,rcond=None)[0]; residual=float(np.std(z-X@co)); curvature=np.linalg.eigvalsh([[2*co[0],co[2]],[co[2],2*co[1]]]);
            except Exception: pass
        k=min(CFG.patch_points,len(local)); pts=np.c_[local[:k],nl[:k],rgb[:k]].astype(np.float32); pts=np.pad(pts,((0,CFG.patch_points-k),(0,0)))
        unique_frames=len(np.unique(f['frame'][order][:min(len(order),256)])); confidence=float(np.clip((1-np.exp(-len(local)/32.))*(1-np.exp(-unique_frames/3.))*np.exp(-np.mean(ages)/3.),0,1))
        stats=np.zeros(CFG.patch_stats_dim,np.float32); vals=[1.,np.log1p(len(ids))/6.,np.log1p(len(local))/6.,unique_frames/12.,float(d3.min()),float(np.mean(ages)),float(z.mean()),rough,float(np.ptp(z)),slope,*normal_mean,float(abs(curvature[0])),float(abs(curvature[1])),float(np.std(xy[:,0])),float(np.std(xy[:,1])),float(np.mean(cnt)),float(confidence),float(rgb[:,0].mean()),float(rgb[:,1].mean()),float(rgb[:,2].mean()),float(rgb[:,0].std()),float(rgb[:,1].std()),float(rgb[:,2].std()),float(z.min()),float(z.max()),float(np.trace(cov)),float(np.linalg.det(cov)**(1/3)),float(np.linalg.norm(mean[:2])),float(residual)]
        stats[:len(vals)]=np.asarray(vals[:CFG.patch_stats_dim],np.float32)
        mu=float(np.clip(.85-1.6*slope-2.5*rough, .25, 1.1))
        pairs=np.c_[f['frame'][order][:min(len(order),64)],f['grid'][order][:min(len(order),64)]]
        return dict(points=pts,stats=stats,normal=normal_mean.astype(np.float32),mu=np.float32(mu),confidence=np.float32(confidence),pairs=pairs)
    def empty(self): return dict(points=np.zeros((CFG.patch_points,9),np.float32),stats=np.zeros(CFG.patch_stats_dim,np.float32),normal=np.array([0,0,1],np.float32),mu=np.float32(.5),confidence=np.float32(0),pairs=np.zeros((0,2),np.int32))
    def snapshot(self,center_world,origin_t,radius=2.0,max_points=10000):
        ids=np.asarray(self.tree.query_ball_point(np.asarray(center_world)[:2],r=float(radius)),np.int64); ids=ids[np.asarray(self.time[ids])<=float(origin_t)+1e-9]
        if len(ids)>max_points: ids=np.random.default_rng(CFG.seed).choice(ids,max_points,replace=False)
        return self._fuse(ids)

GEOMETRY={}
for rec in run_records:
    GEOMETRY[rec['ri']]=CausalVoxelSurfaceMap(build_geometry_events(rec))
    print(rec['run_id'],'geometry events:',len(GEOMETRY[rec['ri']].xyz))

# %% notebook cell 22
# REPLACEMENT FOR THE DELETION-GUARD CELL
# Do not delete anything and do not raise an exception.

from pathlib import Path

if "ALL_RUN_RECORDS" not in globals():
    ALL_RUN_RECORDS = list(run_records)

run_records = [
    rec for rec in ALL_RUN_RECORDS
    if (
        Path(rec["base"])
        / "rosbags"
        / f'{rec["run"]}.bag'
    ).exists()
]

print("Deletion step skipped.")
print("Runs available for geometry:")

for rec in run_records:
    print(" ", rec["run_id"])

# %% notebook cell 23
from pathlib import Path

raw_path = Path('/content/tartanground_v10/ForestEnv/Data_anymal/P2000')

if not raw_path.exists():
    print("ForestEnv raw path is already absent. Nothing to delete.")
else:
    print("ForestEnv raw path still exists; deletion guard remains active.")

# %% notebook cell 24
from pathlib import Path
import shutil

ENVIRONMENT = "ForestEnv"   # change this
RUN_NAME = "P2000"          # change this

RAW_ROOT = Path("/content/tartanground_v10")
PROJECT_ROOT = Path("/content/drive/MyDrive/UGP_V7_FULL4B")

RUN_ID = f"{ENVIRONMENT}_{RUN_NAME}"
RAW_RUN = RAW_ROOT / ENVIRONMENT / "Data_anymal" / RUN_NAME

RUN_ROOTS = list((PROJECT_ROOT / "V10R_real_end_to_end").glob("V10R_REAL_LOTO_*"))

force_done = any(
    (r / "shared" / f"{RUN_ID}_force_labels.npz").exists()
    for r in RUN_ROOTS
)

geometry_done = any(
    all(
        (r / "shared" / "geometry_events" / RUN_ID / f"{k}.npy").exists()
        for k in ("xyz", "normal", "rgb", "grid", "frame", "time", "camera")
    )
    for r in RUN_ROOTS
)

print("force checkpoint:", force_done)
print("geometry checkpoint:", geometry_done)
print("raw path:", RAW_RUN)

if not (force_done and geometry_done):
    raise RuntimeError(
        "Refusing deletion: force labels and geometry checkpoints are not both complete."
    )

if RAW_RUN.exists():
    print("Deleting:", RAW_RUN)
    shutil.rmtree(RAW_RUN)
    print("Deleted successfully.")
else:
    print("Raw run already absent.")

import subprocess
subprocess.run(["df", "-h", "/content"], check=False)

# %% notebook cell 26
def compute_fold_kinematics(fold,train_ris):
    fold_root=RUN_ROOT/'folds'/f'fold_{fold}'; fold_root.mkdir(parents=True,exist_ok=True)
    xpath=fold_root/'camera_base_calibration.json'
    if xpath.exists(): cal=json.loads(xpath.read_text()); x=np.asarray(cal['x'],float); cal_stats=cal.get('stats',{})
    else:
        x,cal_stats=calibrate_tcb(train_ris); xpath.write_text(json.dumps(dict(x=x.tolist(),stats=cal_stats,train_runs=[run_records[i]['run_id'] for i in train_ris]),indent=2))
    R_CB,t_CB=Tcb_from_x(x); kin={}
    for rec in run_records:
        ri=rec['ri']; d=rec['data']; t=np.asarray(d['time'],float); p_wc,R_wc=interp_camera_pose(rec,t-float(t[0]))
        R_wb=np.einsum('nij,jk->nik',R_wc,R_CB); p_wb=p_wc+np.einsum('nij,j->ni',R_wc,t_CB); R_wh=heading_Rwh(R_wb)
        fp=fold_root/f'{rec["run_id"]}_kinematics.npz'
        if fp.exists():
            z=np.load(fp,allow_pickle=False); k={q:z[q] for q in z.files}
            use_cache=str(k.get('kin_version',''))=='V10R_KIN_V2' and all(q in k for q in ('time','p_wb','R_wb','R_wh','foot_base','foot_world','foot_vel_world','force_world10','force_world_raw'))
        else: use_cache=False
        if not use_cache:
            foot_b=np.empty((len(t),4,3),np.float32); foot_w=np.empty_like(foot_b)
            for j in range(len(t)):
                Tb=FK.foot_transforms(d['P'][j,:,:3]); foot_b[j]=Tb[:,:3,3]
                foot_c=foot_b[j]@R_CB.T+t_CB
                foot_w[j]=foot_c@R_wc[j].T+p_wc[j]
            dt=np.maximum(np.diff(t),1e-5); foot_v=np.zeros_like(foot_w); foot_v[1:]=(foot_w[1:]-foot_w[:-1])/dt[:,None,None]; foot_v[0]=foot_v[1]
            fw=[]
            for mode in FORCE_LABELS[ri]['source_modes']:
                fw.append(mode)
            Fsrc=FORCE_LABELS[ri]['F_mean10']; Fraw=FORCE_LABELS[ri]['F_raw']; Fw10=np.empty_like(Fsrc); Fwr=np.empty_like(Fraw)
            for li,mode in enumerate(FORCE_LABELS[ri]['source_modes']):
                if mode=='base': Fw10[:,li]=Fsrc[:,li]@R_wb.transpose(0,2,1); Fwr[:,li]=Fraw[:,li]@R_wb.transpose(0,2,1)
                elif mode=='camera': Fw10[:,li]=Fsrc[:,li]@R_wc.transpose(0,2,1); Fwr[:,li]=Fraw[:,li]@R_wc.transpose(0,2,1)
                else: Fw10[:,li]=Fsrc[:,li]; Fwr[:,li]=Fraw[:,li]
            k=dict(time=t.astype(np.float64),p_wb=p_wb.astype(np.float32),R_wb=R_wb.astype(np.float32),R_wh=R_wh.astype(np.float32),foot_base=foot_b,foot_world=foot_w,foot_vel_world=foot_v,force_world10=Fw10.astype(np.float32),force_world_raw=Fwr.astype(np.float32),kin_version=np.asarray('V10R_KIN_V2',dtype='<U16'))
            np.savez_compressed(fp,**k)
        kin[ri]=k
    return kin,dict(x=x,R_CB=R_CB,t_CB=t_CB,stats=cal_stats)

def phase_at(contact,idx,window=2):
    c=np.asarray(contact)>.5; idx=int(idx); cur=bool(c[idx]); lo=max(0,idx-window); hi=min(len(c),idx+window+1)
    recent=c[lo:idx+1]; prev=bool(c[idx-1]) if idx else cur; nxt=bool(c[idx+1]) if idx+1<len(c) else cur
    if (not prev) and cur: return 1
    if prev and (not cur): return 3
    if cur: return 2
    return 0

def make_fold_samples(fold):
    test_ri=fold; train_ris=[i for i in range(4) if i!=test_ri]; rows=[]
    for ri,rec in enumerate(run_records):
        t=np.asarray(rec['data']['time'],float); n=len(t); dt=np.median(np.diff(t));
        # Build indices from timestamps, rather than assuming exactly 100 Hz.
        # The cache contract records the actual cadence and searchsorted keeps
        # every horizon causal if a logger dropped or duplicated a sample.
        history_start=float(t[0]+CFG.history_s); last_origin=float(t[-1]-ROLLOUT_HORIZONS[-1]-dt)
        origin_times=np.arange(history_start,last_origin+1e-9,CFG.sample_hop_s)
        idxs=np.searchsorted(t,origin_times,side='left'); idxs=np.clip(idxs,0,n-1); idxs=np.unique(idxs)
        frac=(t-t[0])/max(t[-1]-t[0],1e-6)
        for idx in idxs:
            if ri==test_ri: split='test'
            elif frac[idx]<=.70: split='train'
            elif .80<=frac[idx]<=.90: split='val'
            else: continue
            # one-second exclusion gap around the train/validation boundary
            rows.append(dict(fold=fold,ri=ri,run_id=rec['run_id'],terrain=rec['terrain'],idx=idx,time=float(t[idx]),split=split))
    df=pd.DataFrame(rows)
    if CFG.fast_debug:
        df=pd.concat([g.head(160) for _,g in df.groupby(['split','ri'])],ignore_index=True)
    return df

FOLD_META={}; FOLD_KIN={}; FOLD_CAL={}
for fold in CFG.outer_folds:
    train_ris=[i for i in range(4) if i!=fold]; FOLD_KIN[fold],FOLD_CAL[fold]=compute_fold_kinematics(fold,train_ris); FOLD_META[fold]=make_fold_samples(fold)
    FOLD_META[fold].to_csv(RUN_ROOT/'folds'/f'fold_{fold}'/'sample_index.csv',index=False)
    print('fold',fold,'train/val/test',*(int((FOLD_META[fold].split==s).sum()) for s in ('train','val','test')))

def causal_rate(x,t,clip=None):
    x=np.asarray(x,np.float32); out=np.zeros_like(x); dt=np.maximum(np.diff(t),1e-5); out[1:]=np.diff(x,axis=0)/dt.reshape((-1,)+(1,)*(x.ndim-1)); out[0]=out[1]
    if clip is not None: out=np.clip(out,-clip,clip)
    return out

def build_histories(ri,kin):
    rec=run_records[ri]; d=rec['data']; t=d['time']; Fh=kin['force_world10']; R=kin['R_wh']; F_heading=np.einsum('nlj,njk->nlk',Fh,R); dF=causal_rate(F_heading,t,20000.); dE=causal_rate(d['E'],t,50.)
    foot_vb=causal_rate(kin['foot_base'],t,50.)
    leg=np.concatenate([d['P'][:,:,:9],F_heading,d['contact'][...,None],d['L_LEG'],d['E'],dE,dF,kin['foot_base'],foot_vb],axis=-1).astype(np.float32)
    body=np.concatenate([d['B'],d['U']],axis=-1).astype(np.float32); return leg,body,F_heading

for fold in CFG.outer_folds:
    for ri in range(4):
        leg,body,Fh=build_histories(ri,FOLD_KIN[fold][ri]); FOLD_KIN[fold][ri]['leg_hist']=leg; FOLD_KIN[fold][ri]['body_hist']=body; FOLD_KIN[fold][ri]['force_heading']=Fh

# %% notebook cell 28
class HapticMemory:
    def __init__(self,ri,kin):
        d=run_records[ri]['data']; fw=kin['foot_world']; t=kin['time']; Fw=kin['force_world10']; xyz=[]; tm=[]; ff=[]; ee=[]; slip=[]; legs=[]
        for j in range(2,len(t),2):
            for l in range(4):
                if d['contact'][j,l]>.5 and np.isfinite(fw[j,l]).all(): xyz.append(fw[j,l]); tm.append(t[j]); ff.append(Fw[j,l]); ee.append(d['E'][j,l]); slip.append(float(np.linalg.norm(kin['foot_vel_world'][j,l,:2]))); legs.append(l)
        self.xyz=np.asarray(xyz,np.float32); self.time=np.asarray(tm,np.float32); self.force=np.asarray(ff,np.float32); self.err=np.asarray(ee,np.float32); self.slip=np.asarray(slip,np.float32); self.leg=np.asarray(legs,np.int8); self.tree=cKDTree(self.xyz[:,:2]) if len(self.xyz) else None
    def query(self,world_xyz,t0,R_WH,enabled=True,radius=.25):
        if not enabled or self.tree is None: return np.zeros(CFG.physical_dim,np.float32)
        ids=np.asarray(self.tree.query_ball_point(np.asarray(world_xyz)[:2],r=radius),np.int64); ids=ids[self.time[ids]<float(t0)-1e-6]
        if len(ids)==0: return np.zeros(CFG.physical_dim,np.float32)
        dist=np.linalg.norm(self.xyz[ids]-np.asarray(world_xyz)[None,:],axis=1); age=np.maximum(0,float(t0)-self.time[ids]); w=np.exp(-dist/.10)*np.exp(-age/4.); w/=w.sum()+1e-8
        # Stored forces are world-frame rows; expose the haptic feature in the
        # same forecast-origin heading frame as the decoder targets.
        fh=self.force[ids]@np.asarray(R_WH,float); fm=(fh*w[:,None]).sum(0); fs=np.sqrt(((fh-fm)**2*w[:,None]).sum(0)+1e-6); em=(self.err[ids]*w[:,None]).sum(0); es=np.sqrt(((self.err[ids]-em)**2*w[:,None]).sum(0)+1e-6); sl=float((self.slip[ids]*w).sum()); ratio=np.linalg.norm(fm[:2])/(abs(float(fm[2]))+1e-3); hist=np.bincount(self.leg[ids].astype(int),weights=w,minlength=4).astype(np.float32)
        out=np.r_[fm,fs,em,es,sl,float(np.std(self.slip[ids])),ratio,min(len(ids)/32.,1.),float((age*w).sum()),float((dist*w).sum()),hist,1.].astype(np.float32); return np.pad(out,(0,max(0,CFG.physical_dim-len(out))))[:CFG.physical_dim]

HAPTIC={}
for fold in CFG.outer_folds:
    HAPTIC[fold]={ri:HapticMemory(ri,FOLD_KIN[fold][ri]) for ri in range(4)}

def fit_visual_projector(fold,train_ris):
    path=RUN_ROOT/'folds'/f'fold_{fold}'/'visual_pca.joblib'
    if path.exists(): return joblib.load(path)
    rng=np.random.default_rng(CFG.seed+fold); result={'DINO':None,'CONV':None,'FLOW':None}
    for key,ncomp in [('DINO',24),('CONV',24),('FLOW',8)]:
        chunks=[]
        for ri in train_ris:
            v=run_records[ri]['visual'].get(key)
            if v is None: continue
            a=np.asarray(v)
            if key=='FLOW': a=np.concatenate([a.mean(1),a.std(1)],axis=-1)
            a=a.reshape(-1,a.shape[-1]); take=min(len(a),12000//max(1,len(train_ris))); ids=rng.choice(len(a),take,replace=False); chunks.append(a[ids].astype(np.float32))
        if chunks:
            x=np.concatenate(chunks)
            if len(x)>=2:
                p=PCA(n_components=min(ncomp,x.shape[1],len(x)),whiten=True,random_state=CFG.seed+fold); p.fit(x); result[key]=p
    joblib.dump(result,path); return result

VIS_PCA={fold:fit_visual_projector(fold,[i for i in range(4) if i!=fold]) for fold in CFG.outer_folds}

def material_from_pairs(ri,pairs,projector):
    out=[]
    for fr,gr in np.asarray(pairs,np.int64):
        if gr<0: continue
        vals=[]
        for key in ('DINO','CONV'):
            p=projector.get(key); v=run_records[ri]['visual'].get(key)
            if p is None or v is None or v.ndim<3 or fr>=v.shape[0] or gr>=v.shape[1]: continue
            vals.append(p.transform(np.asarray(v[fr,gr],np.float32)[None])[0])
        p=projector.get('FLOW'); v=run_records[ri]['visual'].get('FLOW')
        if p is not None and v is not None and v.ndim>=4 and fr<v.shape[0] and gr<v.shape[2]:
            fv=np.asarray(v[fr,:,gr,:],np.float32); vals.append(p.transform(np.r_[fv.mean(0),fv.std(0)][None])[0])
        if vals: out.append(np.concatenate(vals))
    if not out: return np.zeros(CFG.material_dim,np.float32)
    z=np.mean(np.asarray(out),axis=0); return np.pad(z,(0,max(0,CFG.material_dim-len(z))))[:CFG.material_dim].astype(np.float32)

def origin_frame(kin,i): return kin['p_wb'][i],kin['R_wh'][i]

def make_sample_arrays(fold,row,mode='prequential'):
    ri=int(row.ri); i=int(row.idx); kin=FOLD_KIN[fold][ri]; rec=run_records[ri]; d=rec['data']; t=np.asarray(d['time'],float); leg=kin['leg_hist']; body=kin['body_hist']; hsteps=int(round(CFG.history_s/CFG.history_dt_s)); hist_times=t[i]-CFG.history_s+np.arange(hsteps,dtype=float)*CFG.history_dt_s; hist_ids=np.searchsorted(t,hist_times,side='left'); hist_ids=np.clip(hist_ids,0,len(t)-1); js=np.searchsorted(t,t[i]+ROLLOUT_HORIZONS.astype(float),side='left'); js=np.clip(js,0,len(t)-1); p0,R0=origin_frame(kin,i);
    leg_hist=leg[hist_ids].transpose(1,0,2).astype(np.float32); body_hist=body[hist_ids].astype(np.float32); action=np.repeat(np.r_[d['U'][i],d['L_LEG'][i].reshape(-1)][None],len(ROLLOUT_HORIZONS),axis=0).astype(np.float32)
    foot_local=[]; f10=[]; fr=[]; st=[]; bd=[]; ee=[]; slip=[]; phases=[]; wrench=[]
    for j in js:
        foot_local.append((kin['foot_world'][j]-p0[None])@R0); f10.append(kin['force_world10'][j]@R0); fr.append(kin['force_world_raw'][j]@R0); st.append(np.c_[d['P'][j,:,:9],d['E'][j]]); bd.append(d['B'][j]); ee.append(d['E'][j]); slip.append(np.linalg.norm(kin['foot_vel_world'][j,:,:2],axis=-1)); phases.append([phase_at(d['contact'][:,l],j,2) for l in range(4)]); wrench.append(np.r_[f10[-1].sum(0),np.cross(np.asarray(foot_local[-1]),np.asarray(f10[-1])).sum(0)])
    foot_now=((kin['foot_world'][i]-p0[None])@R0).astype(np.float32)
    return dict(leg_hist=leg_hist,body_hist=body_hist,action=action,foot_now=foot_now,origin_t=np.float32(d['time'][i]),origin_world=p0.astype(np.float32),R_origin=R0.astype(np.float32),
                y_foot=np.asarray(foot_local,np.float32),y_force10=np.asarray(f10,np.float32),y_force_raw=np.asarray(fr,np.float32),y_state=np.asarray(st,np.float32),y_body=np.asarray(bd,np.float32),y_E=np.asarray(ee,np.float32),y_slip=np.asarray(slip,np.float32),y_contact=d['contact'][js].astype(np.float32),phase=np.asarray(phases,np.int64),y_wrench=np.asarray(wrench,np.float32),ri=ri,idx=i)

print('Causal haptic memories and fold PCA projectors ready.')

# %% notebook cell 30
def make_sample_arrays(fold,row,mode='prequential'):
    ri=int(row.ri); i=int(row.idx); kin=FOLD_KIN[fold][ri]; rec=run_records[ri]; d=rec['data']; t=np.asarray(d['time'],float); leg=kin['leg_hist']; body=kin['body_hist']; hsteps=int(round(CFG.history_s/CFG.history_dt_s)); hist_times=t[i]-CFG.history_s+np.arange(hsteps,dtype=float)*CFG.history_dt_s; hist_ids=np.searchsorted(t,hist_times,side='left'); hist_ids=np.clip(hist_ids,0,len(t)-1); js=np.searchsorted(t,t[i]+ROLLOUT_HORIZONS.astype(float),side='left'); js=np.clip(js,0,len(t)-1); p0,R0=origin_frame(kin,i)
    leg_hist=leg[hist_ids].transpose(1,0,2).astype(np.float32); body_hist=body[hist_ids].astype(np.float32); action=np.repeat(np.r_[d['U'][i],d['L_LEG'][i].reshape(-1)][None],len(ROLLOUT_HORIZONS),axis=0).astype(np.float32)
    foot_local=[]; f10=[]; fr=[]; st=[]; bd=[]; ee=[]; slip=[]; phases=[]; wrench=[]
    for j in js:
        fl=(kin['foot_world'][j]-p0[None])@R0; ff10=kin['force_world10'][j]@R0; ffr=kin['force_world_raw'][j]@R0
        foot_local.append(fl); f10.append(ff10); fr.append(ffr); st.append(np.c_[d['P'][j,:,:9],d['E'][j]]); bd.append(d['B'][j]); ee.append(d['E'][j]); slip.append(np.linalg.norm(kin['foot_vel_world'][j,:,:2],axis=-1)); phases.append([phase_at(d['contact'][:,l],j,2) for l in range(4)]); wrench.append(np.r_[ff10.sum(0),np.cross(fl,ff10).sum(0)])
    enabled=(mode!='zero_shot')
    patch_phys=[]
    # Haptic features are queried later at predicted feet; this list is intentionally empty here.
    foot_now=((kin['foot_world'][i]-p0[None])@R0).astype(np.float32)
    return dict(leg_hist=leg_hist,body_hist=body_hist,action=action,foot_now=foot_now,origin_t=np.float32(d['time'][i]),origin_world=p0.astype(np.float32),R_origin=R0.astype(np.float32),
                y_foot=np.asarray(foot_local,np.float32),y_force10=np.asarray(f10,np.float32),y_force_raw=np.asarray(fr,np.float32),y_state=np.asarray(st,np.float32),y_body=np.asarray(bd,np.float32),y_E=np.asarray(ee,np.float32),y_slip=np.asarray(slip,np.float32),y_contact=d['contact'][js].astype(np.float32),phase=np.asarray(phases,np.int64),y_wrench=np.asarray(wrench,np.float32),ri=ri,idx=i)

@dataclass
class FoldNorm:
    leg_mean: np.ndarray; leg_std: np.ndarray; body_mean: np.ndarray; body_std: np.ndarray; action_mean: np.ndarray; action_std: np.ndarray; foot_mean: np.ndarray; foot_std: np.ndarray; state_mean: np.ndarray; state_std: np.ndarray; target_body_mean: np.ndarray; target_body_std: np.ndarray

def fit_norm(fold,df):
    p=RUN_ROOT/'folds'/f'fold_{fold}'/'normalization.npz'
    if p.exists():
        z=np.load(p); return FoldNorm(*(z[k] for k in ('leg_mean','leg_std','body_mean','body_std','action_mean','action_std','foot_mean','foot_std','state_mean','state_std','target_body_mean','target_body_std')))
    ids=df[df.split=='train']; sums=[None]*12; sq=[None]*12; ns=[0]*12
    def acc(k,x,axes):
        a=np.asarray(x,np.float64); sm=a.sum(axis=axes); ss=(a*a).sum(axis=axes); n=int(np.prod([a.shape[q] for q in axes]));
        if sums[k] is None: sums[k]=sm; sq[k]=ss
        else: sums[k]+=sm; sq[k]+=ss
        ns[k]+=n
    for _,row in ids.iterrows():
        s=make_sample_arrays(fold,row)
        acc(0,s['leg_hist'],(0,1)); acc(1,s['body_hist'],(0,)); acc(2,s['action'],(0,)); acc(3,s['y_foot'],(0,1)); acc(4,s['y_state'],(0,1)); acc(5,s['y_body'],(0,))
    means=[sums[k]/max(ns[k],1) for k in range(6)]; stds=[np.sqrt(np.maximum(sq[k]/max(ns[k],1)-means[k]**2,1e-6)) for k in range(6)]
    norm=FoldNorm(means[0].astype(np.float32),stds[0].astype(np.float32),means[1].astype(np.float32),stds[1].astype(np.float32),means[2].astype(np.float32),stds[2].astype(np.float32),means[3].astype(np.float32),stds[3].astype(np.float32),means[4].astype(np.float32),stds[4].astype(np.float32),means[5].astype(np.float32),stds[5].astype(np.float32))
    np.savez_compressed(p,**{f.name:getattr(norm,f.name) for f in dataclasses.fields(norm.__class__)})
    return norm

class ProposalDataset(Dataset):
    def __init__(self,fold,df,norm): self.fold=fold; self.df=df.reset_index(drop=True); self.norm=norm
    def __len__(self): return len(self.df)
    def __getitem__(self,k):
        s=make_sample_arrays(self.fold,self.df.iloc[k]); n=self.norm
        return dict(leg=torch.from_numpy((s['leg_hist']-n.leg_mean)/n.leg_std),body=torch.from_numpy((s['body_hist']-n.body_mean)/n.body_std),action=torch.from_numpy((s['action']-n.action_mean)/n.action_std),foot_now=torch.from_numpy(s['foot_now']),y_foot=torch.from_numpy((s['y_foot']-n.foot_mean)/n.foot_std),y_state=torch.from_numpy((s['y_state']-n.state_mean)/n.state_std),y_body=torch.from_numpy((s['y_body']-n.target_body_mean)/n.target_body_std),y_contact=torch.from_numpy(s['y_contact']),phase=torch.from_numpy(s['phase']),ri=torch.tensor(s['ri']),idx=torch.tensor(s['idx']))

class CausalEncoder(nn.Module):
    def __init__(self,din,d=128,layers=2,heads=4):
        super().__init__(); self.proj=nn.Linear(din,d); self.pos=nn.Parameter(torch.zeros(1,64,d)); layer=nn.TransformerEncoderLayer(d,heads,4*d,.1,batch_first=True,norm_first=True,activation='gelu'); self.enc=nn.TransformerEncoder(layer,layers); self.norm=nn.LayerNorm(d)
    def forward(self,x):
        T=x.shape[1]; z=self.proj(x)+self.pos[:,:T]; mask=torch.triu(torch.ones(T,T,dtype=torch.bool,device=x.device),1); return self.norm(self.enc(z,mask=mask)[:,-1])

class RelationalCoupler(nn.Module):
    def __init__(self,d=128):
        super().__init__(); self.edge=nn.Sequential(nn.Linear(2*d+3,d),nn.GELU(),nn.LayerNorm(d)); self.node=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.LayerNorm(d)); self.body=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.LayerNorm(d))
    def forward(self,leg,body,foot):
        msgs=[]
        for i in range(4):
            ms=[]
            for j in range(4):
                if i!=j: ms.append(self.edge(torch.cat([leg[:,i],leg[:,j],foot[:,j]-foot[:,i]],-1)))
            msgs.append(torch.stack(ms,1).mean(1))
        m=torch.stack(msgs,1); leg=leg+self.node(torch.cat([leg,m],-1))+body[:,None]; body=body+self.body(torch.cat([body,leg.mean(1)],-1)); return leg,body

class FourLegProposal(nn.Module):
    def __init__(self,leg_dim,body_dim,action_dim,H):
        super().__init__(); d=128; self.H=H; self.le= CausalEncoder(leg_dim,d); self.be=CausalEncoder(body_dim,d); self.id=nn.Embedding(4,d); self.rel=RelationalCoupler(d); self.hz=nn.Sequential(nn.Linear(1,d),nn.GELU(),nn.LayerNorm(d)); self.a=nn.Sequential(nn.Linear(action_dim,d),nn.GELU(),nn.LayerNorm(d)); self.cell=nn.GRUCell(2*d,d); self.bcell=nn.GRUCell(2*d,d); self.head=nn.Sequential(nn.Linear(3*d,d),nn.GELU(),nn.LayerNorm(d)); self.mode=nn.Linear(d,4); self.foot=nn.Linear(d,6); self.state=nn.Linear(d,24); self.body=nn.Linear(d,24)
    def encode(self,leg_hist,body_hist,foot_now):
        B,L,T,C=leg_hist.shape; leg=torch.stack([self.le(leg_hist[:,l]) for l in range(4)],1)+self.id(torch.arange(4,device=leg_hist.device))[None]; body=self.be(body_hist); return self.rel(leg,body,foot_now)
    def forward(self,leg_hist,body_hist,action,foot_now):
        leg,body=self.encode(leg_hist,body_hist,foot_now); outs=[]; bl=[]; ll=[]
        for k in range(self.H):
            ah=self.a(action[:,k]); zh=self.hz(torch.full((action.shape[0],1),float((k+1)*CFG.rollout_dt_s),device=action.device)); leg,body=self.rel(leg,body,foot_now); leg=self.cell(torch.cat([ah[:,None].expand(-1,4,-1),body[:,None].expand(-1,4,-1)],-1).reshape(-1,256),leg.reshape(-1,128)).reshape(-1,4,128); body=self.bcell(torch.cat([ah,leg.mean(1)],-1),body); h=self.head(leg+body[:,None]+zh[:,None]); outs.append(h); bl.append(body); ll.append(leg)
        h=torch.stack(outs,1); bodyh=torch.stack(bl,1); legh=torch.stack(ll,1); mode=self.mode(h); foot=self.foot(h).reshape(h.shape[0],h.shape[1],4,6); state=self.state(h).reshape(h.shape[0],h.shape[1],4,24); bo=self.body(bodyh).reshape(h.shape[0],h.shape[1],24)
        return dict(mode_logits=mode,contact_prob=1-torch.softmax(mode,-1)[...,0],foot_mean=foot[...,:3],foot_logscale=foot[...,3:].clamp(-5,3),state_mean=state[...,:12],state_logscale=state[...,12:].clamp(-5,3),body_mean=bo[...,:12],body_logscale=bo[...,12:].clamp(-5,3),latent_leg=legh,latent_body=bodyh)

def gaussian_nll(mu,logscale,y):
    s=torch.exp(logscale).clamp_min(1e-4); return (.5*((y-mu)/s)**2+logscale).mean()

def proposal_loss(o,b):
    phase=b['phase']; contact=b['y_contact']; posw=torch.clamp((contact.numel()-contact.sum())/(contact.sum()+1e-6),1.,5.); mode=F.cross_entropy(o['mode_logits'].reshape(-1,4),phase.reshape(-1),weight=torch.tensor([1.,2.,1.,2.],device=phase.device)); bc=F.binary_cross_entropy_with_logits(torch.logit(o['contact_prob'].clamp(1e-5,1-1e-5)),contact,pos_weight=posw); lf=gaussian_nll(o['foot_mean'],b['y_foot'],o['foot_logscale']); ls=gaussian_nll(o['state_mean'],b['y_state'],o['state_logscale']); lb=gaussian_nll(o['body_mean'],b['y_body'],o['body_logscale']); smooth=((o['foot_mean'][:,1:]-o['foot_mean'][:,:-1])**2).mean(); total=2.*lf+.8*ls+.5*lb+.8*mode+.4*bc+.05*smooth; return total,dict(foot=lf,state=ls,body=lb,mode=mode,contact=bc)

def move_batch(b,keys):
    return {k:(v.to(DEVICE,non_blocking=True) if torch.is_tensor(v) and k in keys else v) for k,v in b.items()}

def train_proposal_fold(fold):
    df=FOLD_META[fold]; norm=fit_norm(fold,df); tr=df[df.split=='train']; va=df[df.split=='val']; ds_tr=ProposalDataset(fold,tr,norm); ds_va=ProposalDataset(fold,va,norm); weights=np.ones(len(tr),np.float64); weights[(tr.reset_index(drop=True).apply(lambda r: bool(make_sample_arrays(fold,r)['y_contact'][:, :].sum()>0),axis=1)).to_numpy()]=2.; sampler=WeightedRandomSampler(torch.as_tensor(weights,dtype=torch.double),len(weights),replacement=True); dl_tr=DataLoader(ds_tr,batch_size=CFG.proposal_batch,sampler=sampler,num_workers=0,pin_memory=DEVICE.type=='cuda'); dl_va=DataLoader(ds_va,batch_size=64,shuffle=False,num_workers=0,pin_memory=DEVICE.type=='cuda'); model=FourLegProposal(ds_tr[0]['leg'].shape[-1],ds_tr[0]['body'].shape[-1],ds_tr[0]['action'].shape[-1],len(ROLLOUT_HORIZONS)).to(DEVICE); opt=torch.optim.AdamW(model.parameters(),lr=CFG.proposal_lr,weight_decay=CFG.weight_decay); scaler=torch.cuda.amp.GradScaler(enabled=DEVICE.type=='cuda'); ck=RUN_ROOT/'folds'/f'fold_{fold}'/'proposal.pt'; hp=RUN_ROOT/'folds'/f'fold_{fold}'/'proposal_history.csv'
    if ck.exists(): model.load_state_dict(torch.load(ck,map_location=DEVICE)); print('reused proposal',fold); return model,norm
    best=None; bestscore=1e99; wait=0; hist=[]
    for ep in range(1,CFG.proposal_epochs+1):
        model.train(); train_losses=[]
        for b in dl_tr:
            b=move_batch(b,{'leg','body','action','foot_now','y_foot','y_state','y_body','y_contact','phase'}); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=DEVICE.type=='cuda'): o=model(b['leg'],b['body'],b['action'],b['foot_now']); loss,_=proposal_loss(o,b)
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); scaler.step(opt); scaler.update(); train_losses.append(float(loss.detach().cpu()))
        model.eval(); vl=[]; fe=[]
        with torch.inference_mode():
            for b in dl_va:
                b=move_batch(b,{'leg','body','action','foot_now','y_foot','y_state','y_body','y_contact','phase'}); o=model(b['leg'],b['body'],b['action'],b['foot_now']); l,_=proposal_loss(o,b); vl.append(float(l.cpu())); fe.append(torch.abs(o['foot_mean']-b['y_foot']).cpu().numpy().reshape(-1))
        e=np.concatenate(fe) if fe else np.array([999.]); score=float(np.mean(vl)+.35*np.quantile(e,.95)); hist.append(dict(epoch=ep,train=float(np.mean(train_losses)),val_loss=float(np.mean(vl)),val_foot_p95_norm=float(np.quantile(e,.95)),score=score));
        if score<bestscore-1e-4: bestscore=score; best=copy.deepcopy(model.state_dict()); wait=0
        else: wait+=1
        if ep==1 or ep%10==0 or wait==0: print(f'fold {fold} proposal ep {ep:03d} train {np.mean(train_losses):.4f} val {np.mean(vl):.4f} foot95n {np.quantile(e,.95):.3f}')
        if wait>=CFG.patience: break
    model.load_state_dict(best); pd.DataFrame(hist).to_csv(hp,index=False); torch.save(model.state_dict(),ck); return model,norm

PROPOSAL={}; PROPOSAL_NORM={}
for fold in CFG.outer_folds:
    PROPOSAL[fold],PROPOSAL_NORM[fold]=train_proposal_fold(fold)

# %% notebook cell 32
@torch.inference_mode()
def proposal_predict_rows(fold):
    df=FOLD_META[fold]; norm=PROPOSAL_NORM[fold]; ds=ProposalDataset(fold,df,norm); dl=DataLoader(ds,batch_size=64,shuffle=False,num_workers=0,pin_memory=DEVICE.type=='cuda'); out={k:[] for k in ('foot_mean','foot_logscale','state_mean','body_mean','contact_prob','latent_leg','latent_body')}
    PROPOSAL[fold].eval()
    for b in dl:
        b=move_batch(b,{'leg','body','action','foot_now'}); o=PROPOSAL[fold](b['leg'],b['body'],b['action'],b['foot_now']); n=norm
        out['foot_mean'].append((o['foot_mean'].cpu().numpy()*n.foot_std+n.foot_mean).astype(np.float32)); out['foot_logscale'].append(o['foot_logscale'].cpu().numpy()); out['state_mean'].append((o['state_mean'].cpu().numpy()*n.state_std+n.state_mean).astype(np.float32)); out['body_mean'].append((o['body_mean'].cpu().numpy()*n.target_body_std+n.target_body_mean).astype(np.float32)); out['contact_prob'].append(o['contact_prob'].cpu().numpy().astype(np.float32)); out['latent_leg'].append(o['latent_leg'].cpu().numpy().astype(np.float32)); out['latent_body'].append(o['latent_body'].cpu().numpy().astype(np.float32))
    return {k:np.concatenate(v,0) for k,v in out.items()}

PROP_PRED={fold:proposal_predict_rows(fold) for fold in CFG.outer_folds}

def fit_scaled_ridge(X,Y,alpha=10.):
    xmu=X.mean(0); xstd=np.maximum(X.std(0),1e-6); model=Ridge(alpha=alpha).fit((X-xmu)/xstd,Y)
    return dict(mean=xmu.astype(np.float32),std=xstd.astype(np.float32),coef=np.asarray(model.coef_,np.float32),intercept=np.asarray(model.intercept_,np.float32))

def ridge_eval(prior,features):
    x=(features-prior['mean'])/prior['std']; return x@prior['coef'].T+prior['intercept']

def fit_fold_priors(fold):
    path=RUN_ROOT/'folds'/f'fold_{fold}'/'calibrated_priors.joblib'
    if path.exists(): return joblib.load(path)
    df=FOLD_META[fold]; train=df[df.split=='train']; Xf=[]; Yf=[]; Xw=[]; Yw=[]
    for _,row in train.iterrows():
        s=make_sample_arrays(fold,row);
        for h,hs in enumerate(ROLLOUT_HORIZONS):
            for l in range(4):
                one=np.eye(4,dtype=np.float32)[l]; Xf.append(np.r_[s['y_state'][h,l],s['y_body'][h],s['y_foot'][h,l],s['y_contact'][h,l],one,hs]); Yf.append(s['y_force10'][h,l])
            Xw.append(np.r_[s['y_body'][h],s['y_foot'][h].reshape(-1),s['y_contact'][h],hs]); Yw.append(s['y_wrench'][h])
    pri=dict(force=fit_scaled_ridge(np.asarray(Xf,np.float32),np.asarray(Yf,np.float32),alpha=25.),wrench=fit_scaled_ridge(np.asarray(Xw,np.float32),np.asarray(Yw,np.float32),alpha=25.)); joblib.dump(pri,path); return pri

PRIORS={fold:fit_fold_priors(fold) for fold in CFG.outer_folds}

def force_prior_from_prediction(prior,state,body,foot,contact):
    B,H,L,_=state.shape; one=np.eye(4,dtype=np.float32)[None,None].repeat(B,0).repeat(H,1); hs=np.broadcast_to(ROLLOUT_HORIZONS[None,:,None,None],(B,H,L,1)); x=np.concatenate([state,body[:,:,None].repeat(L,2),foot,contact[...,None],one,hs],-1); return ridge_eval(prior['force'],x).astype(np.float32)

def wrench_prior_from_prediction(prior,body,foot,contact):
    B,H,_,_=foot.shape; hs=np.broadcast_to(ROLLOUT_HORIZONS[None,:,None],(B,H,1)); x=np.concatenate([body,foot.reshape(B,H,-1),contact,hs],-1); return ridge_eval(prior['wrench'],x).astype(np.float32)

print('Proposal predictions:',{f:PROP_PRED[f]['foot_mean'].shape for f in PROP_PRED})

# %% notebook cell 34
def build_patch_cache(fold):
    root=RUN_ROOT/'folds'/f'fold_{fold}'/'patch_cache'; root.mkdir(parents=True,exist_ok=True); n=len(FOLD_META[fold]); H=len(ROLLOUT_HORIZONS); shapes={'points':(n,H,4,CFG.patch_points,9),'stats':(n,H,4,CFG.patch_stats_dim),'material':(n,H,4,CFG.material_dim),'physical':(n,H,4,CFG.physical_dim),'normal':(n,H,4,3),'mu':(n,H,4),'confidence':(n,H,4),'prior':(n,H,4,3),'wrench':(n,H,6)}; mm={}
    complete=root/'COMPLETE.json'
    if complete.exists() and all((root/f'{k}.npy').exists() for k in shapes):
        try: cache_meta=json.loads(complete.read_text())
        except Exception: cache_meta={}
        if cache_meta.get('cache_version')=='V10R_PATCH_V2': return {k:np.load(root/f'{k}.npy',mmap_mode='r') for k in shapes}
    for k,shape in shapes.items(): mm[k]=np.lib.format.open_memmap(root/f'{k}.npy',mode='w+',dtype=np.float16 if k in ('points','stats','material','physical') else np.float32,shape=shape)
    pred=PROP_PRED[fold]; prior=PRIORS[fold]; pca=VIS_PCA[fold]
    for rirow,row in FOLD_META[fold].iterrows():
        s=make_sample_arrays(fold,row); ri=int(s['ri']); kin=FOLD_KIN[fold][ri]; p0=np.asarray(s['origin_world']); R0=np.asarray(s['R_origin']); foot=pred['foot_mean'][rirow]; state=pred['state_mean'][rirow]; body=pred['body_mean'][rirow]; contact=pred['contact_prob'][rirow]; prior_f=force_prior_from_prediction(prior,state[None],body[None],foot[None],contact[None])[0]; prior_w=wrench_prior_from_prediction(prior,body[None],foot[None],contact[None])[0]
        for h in range(H):
            for l in range(4):
                local=foot[h,l]; world=p0+local@R0.T; sig=np.exp(pred['foot_logscale'][rirow,h,l])*PROPOSAL_NORM[fold].foot_std; cov=R0.T@np.diag(sig**2)@R0; q=GEOMETRY[ri].query(world,cov,float(s['origin_t']),R0); hp=HAPTIC[fold][ri].query(world,float(s['origin_t']),R0,enabled=True); mat=material_from_pairs(ri,q['pairs'],pca); mm['points'][rirow,h,l]=q['points']; mm['stats'][rirow,h,l]=q['stats']; mm['material'][rirow,h,l]=mat; mm['physical'][rirow,h,l]=hp; mm['normal'][rirow,h,l]=q['normal']; mm['mu'][rirow,h,l]=q['mu']; mm['confidence'][rirow,h,l]=q['confidence'];
            mm['prior'][rirow,h]=prior_f[h]; mm['wrench'][rirow,h]=prior_w[h]
        if (rirow+1)%100==0: print('fold',fold,'patch rows',rirow+1,'/',n)
    for v in mm.values(): v.flush()
    complete.write_text(json.dumps(dict(cache_version='V10R_PATCH_V2',rows=n,horizons=ROLLOUT_HORIZONS.tolist(),causal_timestamp='origin_t',geometry_source='RGB-D event store',haptic_mode='prequential-cache'),indent=2)); return {k:np.load(root/f'{k}.npy',mmap_mode='r') for k in shapes}

PATCH={fold:build_patch_cache(fold) for fold in CFG.outer_folds}
print('Patch caches ready.')

# %% notebook cell 36
class PatchEncoder(nn.Module):
    def __init__(self):
        super().__init__(); self.point=nn.Sequential(nn.Linear(9,64),nn.GELU(),nn.Linear(64,96),nn.GELU()); self.stats=nn.Sequential(nn.LayerNorm(CFG.patch_stats_dim),nn.Linear(CFG.patch_stats_dim,64),nn.GELU()); self.mat=nn.Sequential(nn.LayerNorm(CFG.material_dim),nn.Linear(CFG.material_dim,64),nn.GELU()); self.phys=nn.Sequential(nn.LayerNorm(CFG.physical_dim),nn.Linear(CFG.physical_dim,64),nn.GELU()); self.out=nn.Sequential(nn.Linear(96+64+64+64+5,128),nn.GELU(),nn.LayerNorm(128))
    def forward(self,points,stats,material,physical,normal,mu,confidence):
        p=self.point(points).amax(-2); x=torch.cat([p,self.stats(stats),self.mat(material),self.phys(physical),normal,mu.unsqueeze(-1),confidence.unsqueeze(-1)],-1); return self.out(x)

def cross_matrix_t(r):
    z=torch.zeros_like(r[...,0]); x,y,zz=r.unbind(-1); return torch.stack([z,-zz,y,zz,z,-x,-y,x,z],-1).reshape(*r.shape[:-1],3,3)

class JointContactProjection(nn.Module):
    def __init__(self,iterations=4,eps=1e-3): super().__init__(); self.iterations=iterations; self.eps=eps
    def forward(self,f_free,foot,normal,mu,contact,wrench):
        B,H,L,_=f_free.shape; I=torch.eye(3,device=f_free.device,dtype=f_free.dtype).reshape(1,1,1,3,3).expand(B,H,L,-1,-1); blocks=[torch.cat([I[:,:,l],cross_matrix_t(foot[:,:,l])],-2) for l in range(4)]; A=torch.cat(blocks,-1); eye=torch.eye(6,device=f_free.device,dtype=f_free.dtype).reshape(1,1,6,6).expand(B,H,-1,-1); f=f_free*contact.unsqueeze(-1)
        n=F.normalize(normal,dim=-1,eps=1e-5)
        for _ in range(self.iterations):
            err=wrench-(A@f.reshape(B,H,12,1)).squeeze(-1); corr=A.transpose(-1,-2)@torch.linalg.solve(A@A.transpose(-1,-2)+self.eps*eye,err.unsqueeze(-1)); f=f.reshape(B,H,12,1)+corr; f=f.reshape(B,H,4,3); fn=(f*n).sum(-1,keepdim=True); fn=F.softplus(fn,beta=8.)*contact.unsqueeze(-1); ft=f-fn*n; scale=torch.clamp(mu.unsqueeze(-1)*fn/(torch.linalg.vector_norm(ft,dim=-1,keepdim=True)+1e-6),max=1.); f=fn*n+scale*ft
        residual=wrench-(A@f.reshape(B,H,12,1)).squeeze(-1); return f,residual

class CoupledForceDecoder(nn.Module):
    def __init__(self,force_scale):
        super().__init__(); self.patch=PatchEncoder(); self.a=nn.Sequential(nn.Linear(18,64),nn.GELU()); self.state=nn.Sequential(nn.Linear(12,64),nn.GELU()); self.body=nn.Sequential(nn.Linear(12,64),nn.GELU()); self.foot=nn.Sequential(nn.Linear(4,32),nn.GELU()); self.fuse=nn.Sequential(nn.Linear(128+128+64+64+64+32+128,256),nn.GELU(),nn.Dropout(.10),nn.Linear(256,160),nn.GELU(),nn.LayerNorm(160)); self.head=nn.Linear(160,15); self.register_buffer('force_scale',torch.as_tensor(force_scale,dtype=torch.float32)); self.project=JointContactProjection()
        with torch.no_grad(): nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias); nn.init.constant_(self.head.bias[3:6],math.log(25.))
    def forward(self,leg_lat,body_lat,action,prop_state,prop_body,prop_foot,prop_contact,points,stats,material,physical,normal,mu,confidence,prior,wrench):
        B,H,L,_=leg_lat.shape; patch=self.patch(points,stats,material,physical,normal,mu,confidence); aa=self.a(action).unsqueeze(2).expand(-1,-1,4,-1); st=self.state(prop_state); bd=self.body(prop_body).unsqueeze(2).expand(-1,-1,4,-1); ft=self.foot(torch.cat([prop_foot,prop_contact.unsqueeze(-1)],-1)); z=self.fuse(torch.cat([leg_lat,body_lat.unsqueeze(2).expand(-1,-1,4,-1),aa,st,bd,ft,patch],-1)); h=self.head(z); delta=h[...,:3]*self.force_scale; ffree=prior+delta; fproj,res=self.project(ffree,prop_foot,normal,mu,prop_contact,wrench); raw=prior+h[...,9:12]*self.force_scale; return dict(force10=fproj,force_raw=raw,logscale=h[...,3:6].clamp(-2,6),raw_logscale=h[...,12:15].clamp(-2,6),wrench_residual=res,free=ffree,latent=z)

class ForceDataset(Dataset):
    def __init__(self,fold,df,norm,mode='prequential'):
        self.fold=fold; self.row_indices=np.asarray(df.index,np.int64); self.df=df.reset_index(drop=True); self.norm=norm; self.mode=mode; self.patch=PATCH[fold]; self.pred=PROP_PRED[fold]
    def __len__(self): return len(self.df)
    def __getitem__(self,k):
        row=self.df.iloc[k]; row_source=int(self.row_indices[k]); s=make_sample_arrays(self.fold,row,mode=self.mode); n=self.norm; p=self.pred[row_source]; pp=self.patch; phys=np.asarray(pp['physical'][row_source],np.float32).copy();
        if self.mode=='zero_shot' and row.split=='test': phys.fill(0.)
        return dict(leg_lat=torch.from_numpy(p['latent_leg']),body_lat=torch.from_numpy(p['latent_body']),action=torch.from_numpy((s['action']-n.action_mean)/n.action_std),prop_state=torch.from_numpy(p['state_mean']),prop_body=torch.from_numpy(p['body_mean']),prop_foot=torch.from_numpy(p['foot_mean']),prop_contact=torch.from_numpy(p['contact_prob']),points=torch.from_numpy(np.asarray(pp['points'][row_source],np.float32)),stats=torch.from_numpy(np.asarray(pp['stats'][row_source],np.float32)),material=torch.from_numpy(np.asarray(pp['material'][row_source],np.float32)),physical=torch.from_numpy(phys),normal=torch.from_numpy(np.asarray(pp['normal'][row_source],np.float32)),mu=torch.from_numpy(np.asarray(pp['mu'][row_source],np.float32)),confidence=torch.from_numpy(np.asarray(pp['confidence'][row_source],np.float32)),prior=torch.from_numpy(np.asarray(pp['prior'][row_source],np.float32)),wrench=torch.from_numpy(np.asarray(pp['wrench'][row_source],np.float32)),y_force10=torch.from_numpy(s['y_force10']),y_force_raw=torch.from_numpy(s['y_force_raw']),y_contact=torch.from_numpy(s['y_contact']),y_wrench=torch.from_numpy(s['y_wrench']),ri=torch.tensor(s['ri']),idx=torch.tensor(s['idx']))

def force_loss(o,b):
    c=b['y_contact']; w=(.30+1.70*c).unsqueeze(-1); e=o['force10']-b['y_force10']; data=(F.smooth_l1_loss(o['force10'],b['y_force10'],beta=5.,reduction='none')*w).mean(); vec=(torch.linalg.vector_norm(e,dim=-1)*(0.30+1.70*c)).mean(); raw=F.smooth_l1_loss(o['force_raw'],b['y_force_raw'],beta=8.); zero=((1-c)*torch.linalg.vector_norm(o['force10'],dim=-1)).mean(); sigma=torch.exp(o['logscale']); nll=(.5*((e/sigma)**2)+o['logscale']).mean(); wrench=(o['wrench_residual']**2).mean(); total=data+.15*vec+.08*raw+.02*zero+.02*nll+.03*wrench; return total,dict(data=data,vector=vec,raw=raw,zero=zero,nll=nll,wrench=wrench)

def collect_force_scale(fold):
    p=RUN_ROOT/'folds'/f'fold_{fold}'/'force_scale.npy'
    if p.exists(): return np.load(p)
    vals=[]
    for _,row in FOLD_META[fold][FOLD_META[fold].split=='train'].iterrows(): vals.append(make_sample_arrays(fold,row)['y_force10'])
    scale=np.maximum(np.concatenate(vals).std(axis=(0,1,2)),5.).astype(np.float32); np.save(p,scale); return scale

def train_force_fold(fold):
    df=FOLD_META[fold]; norm=PROPOSAL_NORM[fold]; scale=collect_force_scale(fold); tr=df[df.split=='train']; va=df[df.split=='val']; ds_tr=ForceDataset(fold,tr,norm,'prequential'); ds_va=ForceDataset(fold,va,norm,'prequential'); dl_tr=DataLoader(ds_tr,batch_size=CFG.force_batch,shuffle=True,num_workers=0,pin_memory=DEVICE.type=='cuda'); dl_va=DataLoader(ds_va,batch_size=64,shuffle=False,num_workers=0,pin_memory=DEVICE.type=='cuda'); model=CoupledForceDecoder(scale).to(DEVICE); opt=torch.optim.AdamW(model.parameters(),lr=CFG.force_lr,weight_decay=CFG.weight_decay); scaler=torch.cuda.amp.GradScaler(enabled=DEVICE.type=='cuda'); ck=RUN_ROOT/'folds'/f'fold_{fold}'/'force.pt'; hp=RUN_ROOT/'folds'/f'fold_{fold}'/'force_history.csv'
    if ck.exists(): model.load_state_dict(torch.load(ck,map_location=DEVICE)); print('reused force',fold); return model
    best=None; bestscore=1e99; wait=0; hist=[]
    for ep in range(1,CFG.force_epochs+1):
        model.train(); tl=[]
        for b in dl_tr:
            b=move_batch(b,{'leg_lat','body_lat','action','prop_state','prop_body','prop_foot','prop_contact','points','stats','material','physical','normal','mu','confidence','prior','wrench','y_force10','y_force_raw','y_contact','y_wrench'}); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=DEVICE.type=='cuda'): o=model(b['leg_lat'],b['body_lat'],b['action'],b['prop_state'],b['prop_body'],b['prop_foot'],b['prop_contact'],b['points'],b['stats'],b['material'],b['physical'],b['normal'],b['mu'],b['confidence'],b['prior'],b['wrench']); loss,_=force_loss(o,b)
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); scaler.step(opt); scaler.update(); tl.append(float(loss.detach().cpu()))
        model.eval(); vl=[]; ae=[]
        with torch.inference_mode():
            for b in dl_va:
                b=move_batch(b,{'leg_lat','body_lat','action','prop_state','prop_body','prop_foot','prop_contact','points','stats','material','physical','normal','mu','confidence','prior','wrench','y_force10','y_force_raw','y_contact','y_wrench'}); o=model(b['leg_lat'],b['body_lat'],b['action'],b['prop_state'],b['prop_body'],b['prop_foot'],b['prop_contact'],b['points'],b['stats'],b['material'],b['physical'],b['normal'],b['mu'],b['confidence'],b['prior'],b['wrench']); l,_=force_loss(o,b); vl.append(float(l.cpu())); ae.append(torch.abs(o['force10']-b['y_force10']).cpu().numpy().reshape(-1))
        e=np.concatenate(ae) if ae else np.array([999.]); score=float(e.mean()+.40*np.quantile(e,.95)); hist.append(dict(epoch=ep,train=float(np.mean(tl)),val_loss=float(np.mean(vl)),val_component_mae=float(e.mean()),val_component_p95=float(np.quantile(e,.95)),score=score));
        if score<bestscore-1e-3: bestscore=score; best=copy.deepcopy(model.state_dict()); wait=0
        else: wait+=1
        if ep==1 or ep%10==0 or wait==0: print(f'fold {fold} force ep {ep:03d} train {np.mean(tl):.3f} val {e.mean():.3f} p95 {np.quantile(e,.95):.3f}')
        if wait>=CFG.patience: break
    model.load_state_dict(best); pd.DataFrame(hist).to_csv(hp,index=False); torch.save(model.state_dict(),ck); return model

FORCE_MODEL={fold:train_force_fold(fold) for fold in CFG.outer_folds}

# %% notebook cell 38
@torch.inference_mode()
def force_predict_fold(fold,mode):
    df=FOLD_META[fold]; ds=ForceDataset(fold,df,PROPOSAL_NORM[fold],mode); dl=DataLoader(ds,batch_size=48,shuffle=False,num_workers=0,pin_memory=DEVICE.type=='cuda'); out={k:[] for k in ('force10','force_raw','logscale','raw_logscale','contact','ri','idx')}; model=FORCE_MODEL[fold].eval()
    for b in dl:
        b=move_batch(b,{'leg_lat','body_lat','action','prop_state','prop_body','prop_foot','prop_contact','points','stats','material','physical','normal','mu','confidence','prior','wrench'}); o=model(b['leg_lat'],b['body_lat'],b['action'],b['prop_state'],b['prop_body'],b['prop_foot'],b['prop_contact'],b['points'],b['stats'],b['material'],b['physical'],b['normal'],b['mu'],b['confidence'],b['prior'],b['wrench']); out['force10'].append(o['force10'].cpu().numpy()); out['force_raw'].append(o['force_raw'].cpu().numpy()); out['logscale'].append(o['logscale'].cpu().numpy()); out['raw_logscale'].append(o['raw_logscale'].cpu().numpy()); out['contact'].append(b['prop_contact'].cpu().numpy()); out['ri'].append(b['ri'].numpy()); out['idx'].append(b['idx'].numpy())
    return {k:np.concatenate(v,0) for k,v in out.items()}

FORCE_PRED={fold:{mode:force_predict_fold(fold,mode) for mode in ('prequential','zero_shot')} for fold in CFG.outer_folds}

def conformal_scale(fold):
    path=RUN_ROOT/'folds'/f'fold_{fold}'/'conformal_scale.npy'; pred=FORCE_PRED[fold]['prequential']; df=FOLD_META[fold]; val=np.flatnonzero((df.split=='val').to_numpy());
    if path.exists(): return np.load(path)
    rows=[]
    for k in val:
        s=make_sample_arrays(fold,df.iloc[k]); rows.append(np.abs(pred['force10'][k]-s['y_force10']))
    e=np.asarray(rows) if rows else np.ones((1,len(ROLLOUT_HORIZONS),4,3),np.float32); scale=np.maximum(np.quantile(e,.90,axis=0),2.).astype(np.float32); np.save(path,scale); return scale

CONFORMAL={fold:conformal_scale(fold) for fold in CFG.outer_folds}

def phase_mean_baseline(fold,df):
    train=df[df.split=='train']; arr=np.full((len(ROLLOUT_HORIZONS),4,4,3),np.nan,np.float32)
    buckets={(h,l,p,a):[] for h in range(len(ROLLOUT_HORIZONS)) for l in range(4) for p in range(4) for a in range(3)}
    for _,row in train.iterrows():
        s=make_sample_arrays(fold,row)
        for h in range(len(ROLLOUT_HORIZONS)):
            for l in range(4):
                for a in range(3): buckets[(h,l,int(s['phase'][h,l]),a)].append(float(s['y_force10'][h,l,a]))
    for key,vals in buckets.items(): arr[key]=float(np.mean(vals)) if vals else 0.
    return arr

PHASE_MEAN={fold:phase_mean_baseline(fold,FOLD_META[fold]) for fold in CFG.outer_folds}

def build_oof_tables():
    force_rows=[]; sensor_rows=[]; contact_rows=[]; event_rows=[]
    for fold in CFG.outer_folds:
        df=FOLD_META[fold]; test_ids=np.flatnonzero((df.split=='test').to_numpy()); norm=PROPOSAL_NORM[fold]; pp=PROP_PRED[fold]
        # Proposal and contact metrics use every test origin; force modes are intentionally separate.
        for k in test_ids:
            row=df.iloc[k]; s=make_sample_arrays(fold,row); p=pp[k]
            for hidx in EVAL_HORIZON_IDS:
                h=int(hidx); hs=float(ROLLOUT_HORIZONS[h]);
                for l in range(4):
                    for a in range(3):
                        for offset,name,kind in [(0,f'q{a}','q'),(3,f'dq{a}','dq'),(6,f'tau{a}','tau')]:
                            sensor_rows.append(dict(fold=fold,run_id=row.run_id,terrain=row.terrain,origin_t=float(row.time),horizon_s=hs,leg=LEG_NAMES[l],sensor=name,target=float(s['y_state'][h,l,offset+a]),prediction=float(p['state_mean'][h,l,offset+a]),kind=kind))
                        sensor_rows.append(dict(fold=fold,run_id=row.run_id,terrain=row.terrain,origin_t=float(row.time),horizon_s=hs,leg=LEG_NAMES[l],sensor=f'E{a}',target=float(s['y_E'][h,l,a]),prediction=float(p['state_mean'][h,l,9+a]),kind='E'))
                    contact_rows.append(dict(fold=fold,run_id=row.run_id,terrain=row.terrain,origin_t=float(row.time),horizon_s=hs,leg=LEG_NAMES[l],true=float(s['y_contact'][h,l]),prob=float(p['contact_prob'][h,l])))
                for b in range(12): sensor_rows.append(dict(fold=fold,run_id=row.run_id,terrain=row.terrain,origin_t=float(row.time),horizon_s=hs,leg='BODY',sensor=f'B{b}',target=float(s['y_body'][h,b]),prediction=float(p['body_mean'][h,b]),kind='B'))
            for mode in ('prequential','zero_shot'):
                fp=FORCE_PRED[fold][mode]; qscale=CONFORMAL[fold]
                for hidx in EVAL_HORIZON_IDS:
                    h=int(hidx); hs=float(ROLLOUT_HORIZONS[h]);
                    for l in range(4):
                        for a,name in enumerate(AXIS_NAMES):
                            mu=float(fp['force10'][k,h,l,a]); sig=float(np.exp(fp['logscale'][k,h,l,a])*qscale[h,l,a]); y=float(s['y_force10'][h,l,a]); yr=float(s['y_force_raw'][h,l,a]); ph=PHASE_NAMES[int(s['phase'][h,l])];
                            force_rows.append(dict(fold=fold,run_id=row.run_id,terrain=row.terrain,origin_t=float(row.time),horizon_s=hs,leg=LEG_NAMES[l],axis=name,phase=ph,haptic_mode=mode,true_mean10=y,true_raw=yr,pred_mean10=mu,pred_raw=float(fp['force_raw'][k,h,l,a]),sigma90=sig,covered90=float(abs(y-mu)<=sig),contact_true=float(s['y_contact'][h,l]),contact_prob=float(fp['contact'][k,h,l]),foot_error_m=float(np.linalg.norm(p['foot_mean'][h,l]-s['y_foot'][h,l])),geometry_confidence=float(PATCH[fold]['confidence'][k,h,l])))
                true_c=s['y_contact'][:,0:4]; prob=fp['contact'][k];
                for l in range(4):
                    ti=np.flatnonzero((true_c[1:,l]>.5)&(true_c[:-1]<=.5)); pi=np.flatnonzero((prob[1:,l]>=.5)&(prob[:-1]<.5));
                    if len(ti) and len(pi):
                        ttrue=float(ROLLOUT_HORIZONS[ti[0]+1]); tpred=float(ROLLOUT_HORIZONS[pi[0]+1]); j0=ti[0]+1; j1=min(len(ROLLOUT_HORIZONS),j0+3); imp_true=s['y_force10'][j0:j1,l].sum(0)*CFG.rollout_dt_s; imp_pred=fp['force10'][k,j0:j1,l].sum(0)*CFG.rollout_dt_s; event_rows.append(dict(fold=fold,run_id=row.run_id,terrain=row.terrain,origin_t=float(row.time),haptic_mode=mode,leg=LEG_NAMES[l],true_touchdown_s=ttrue,pred_touchdown_s=tpred,event_error_ms=1000*(tpred-ttrue),impulse_error_Ns=float(np.linalg.norm(imp_pred-imp_true)),true_impulse_Ns=float(np.linalg.norm(imp_true)),pred_impulse_Ns=float(np.linalg.norm(imp_pred))))
    force_df=pd.DataFrame(force_rows); sensor_df=pd.DataFrame(sensor_rows); contact_df=pd.DataFrame(contact_rows); event_df=pd.DataFrame(event_rows)
    return force_df,sensor_df,contact_df,event_df

FORCE_OOF,SENSOR_OOF,CONTACT_OOF,EVENT_OOF=build_oof_tables()
FORCE_OOF.to_csv(RUN_ROOT/'tables'/'force_predictions_oof.csv',index=False); SENSOR_OOF.to_csv(RUN_ROOT/'tables'/'sensor_predictions_oof.csv',index=False); CONTACT_OOF.to_csv(RUN_ROOT/'tables'/'contact_predictions_oof.csv',index=False); EVENT_OOF.to_csv(RUN_ROOT/'tables'/'touchdown_event_oof.csv',index=False)

def summarize_force(df,model_name='V10R'):
    rows=[]
    for keys,g in df.groupby(['haptic_mode','terrain','horizon_s','leg','axis','phase'],dropna=False):
        e=np.abs(g.pred_mean10-g.true_mean10); v=np.sqrt((g.pred_mean10-g.true_mean10)**2); rows.append(dict(model=model_name,haptic_mode=keys[0],terrain=keys[1],horizon_s=keys[2],leg=keys[3],axis=keys[4],phase=keys[5],n=len(g),mae_N=float(e.mean()),rmse_N=float(np.sqrt(np.mean(v**2))),p95_N=float(np.quantile(e,.95)),p99_N=float(np.quantile(e,.99)),max_N=float(e.max()),within3=float((e<=3).mean()),within4=float((e<=4).mean()),within5=float((e<=5).mean()),coverage90=float(g.covered90.mean()),interval_width_N=float((2*g.sigma90).mean())))
    return pd.DataFrame(rows)

FORCE_METRICS=summarize_force(FORCE_OOF); FORCE_METRICS.to_csv(RUN_ROOT/'tables'/'force_metrics_by_phase.csv',index=False)

def summarize_force_vector(df,model_name='V10R'):
    rows=[]; keys=['haptic_mode','terrain','horizon_s','leg','phase']
    for group,g in df.groupby(keys,dropna=False):
        piv=g.pivot_table(index=['fold','run_id','origin_t'],columns='axis',values=['pred_mean10','true_mean10'],aggfunc='first')
        if len(piv)==0 or not isinstance(piv.columns,pd.MultiIndex) or 'pred_mean10' not in piv.columns.get_level_values(0): continue
        if not all(a in piv['pred_mean10'].columns for a in AXIS_NAMES): continue
        pred=piv['pred_mean10'][list(AXIS_NAMES)].to_numpy(float); true=piv['true_mean10'][list(AXIS_NAMES)].to_numpy(float); e=np.linalg.norm(pred-true,axis=1)
        rows.append(dict(model=model_name,haptic_mode=group[0],terrain=group[1],horizon_s=group[2],leg=group[3],phase=group[4],n=len(e),vector_mae_N=float(e.mean()),vector_p95_N=float(np.quantile(e,.95)),vector_p99_N=float(np.quantile(e,.99)),vector_max_N=float(e.max()),within3=float((e<=3).mean()),within4=float((e<=4).mean()),within5=float((e<=5).mean())))
    return pd.DataFrame(rows)

FORCE_VECTOR_METRICS=summarize_force_vector(FORCE_OOF); FORCE_VECTOR_METRICS.to_csv(RUN_ROOT/'tables'/'force_vector_metrics.csv',index=False)

def summarize_sensor(df):
    rows=[]
    for keys,g in df.groupby(['horizon_s','sensor','kind']):
        e=np.abs(g.prediction-g.target); rows.append(dict(horizon_s=keys[0],sensor=keys[1],kind=keys[2],n=len(g),mae=float(e.mean()),rmse=float(np.sqrt(np.mean(e**2))),p95=float(np.quantile(e,.95))))
    return pd.DataFrame(rows)

SENSOR_METRICS=summarize_sensor(SENSOR_OOF); SENSOR_METRICS.to_csv(RUN_ROOT/'tables'/'sensor_metrics.csv',index=False)
CONTACT_METRICS=[]
for keys,g in CONTACT_OOF.groupby(['horizon_s','leg']):
    y=g.true.to_numpy(); pr=g.prob.to_numpy(); yh=(pr>=.5).astype(int); pp,rr,f1,_=precision_recall_fscore_support(y.astype(int),yh,average='binary',zero_division=0); CONTACT_METRICS.append(dict(horizon_s=keys[0],leg=keys[1],f1=float(f1),precision=float(pp),recall=float(rr),brier=float(brier_score_loss(y,pr)),auprc=float(average_precision_score(y,pr))))
CONTACT_METRICS=pd.DataFrame(CONTACT_METRICS); CONTACT_METRICS.to_csv(RUN_ROOT/'tables'/'contact_metrics.csv',index=False)

primary=FORCE_METRICS[(FORCE_METRICS.haptic_mode=='prequential')]; primary_vec=FORCE_VECTOR_METRICS[(FORCE_VECTOR_METRICS.haptic_mode=='prequential')]; worst=float(primary.p95_N.max()) if len(primary) else np.nan; overall=float(np.average(primary.mae_N,weights=primary.n)) if len(primary) else np.nan; worst_vec=float(primary_vec.vector_p95_N.max()) if len(primary_vec) else np.nan; gate=bool(np.isfinite(worst) and worst<=4.0 and overall<=4.0 and len(EVENT_OOF)>0)
GATE=dict(v10r_measured=True,primary_force_target=CFG.force_target,overall_component_mae_N=overall,worst_phase_leg_axis_horizon_p95_N=worst,worst_vector_p95_N=worst_vec,within4_fraction=float(np.average(primary.within4,weights=primary.n)) if len(primary) else np.nan,hard_3_to_4N_gate=gate,warning='A failed gate is an empirical result; it is not softened by removing touchdown rows.')
(RUN_ROOT/'tables'/'claim_gate.json').write_text(json.dumps(GATE,indent=2)); print(json.dumps(GATE,indent=2)); display(FORCE_METRICS.sort_values('p95_N',ascending=False).head(20))

# %% notebook cell 40
def baseline_force_table():
    rows=[]
    for fold in CFG.outer_folds:
        df=FOLD_META[fold]; test_ids=np.flatnonzero((df.split=='test').to_numpy()); means=PHASE_MEAN[fold]
        for k in test_ids:
            s=make_sample_arrays(fold,df.iloc[k]); current=s['leg_hist'][:,-1,9:12]
            for hidx in EVAL_HORIZON_IDS:
                h=int(hidx)
                for l in range(4):
                    for a,axis in enumerate(AXIS_NAMES):
                        y=float(s['y_force10'][h,l,a]); ph=int(s['phase'][h,l]); rows += [dict(model='persistence',fold=fold,terrain=df.iloc[k].terrain,horizon_s=float(ROLLOUT_HORIZONS[h]),leg=LEG_NAMES[l],axis=axis,phase=PHASE_NAMES[ph],target=y,pred=float(current[l,a])),dict(model='zero',fold=fold,terrain=df.iloc[k].terrain,horizon_s=float(ROLLOUT_HORIZONS[h]),leg=LEG_NAMES[l],axis=axis,phase=PHASE_NAMES[ph],target=y,pred=0.),dict(model='phase_mean',fold=fold,terrain=df.iloc[k].terrain,horizon_s=float(ROLLOUT_HORIZONS[h]),leg=LEG_NAMES[l],axis=axis,phase=PHASE_NAMES[ph],target=y,pred=float(means[h,l,ph,a]))]
    return pd.DataFrame(rows)

BASELINE_OOF=baseline_force_table(); BASELINE_OOF.to_csv(RUN_ROOT/'tables'/'baseline_force_predictions.csv',index=False)
base_rows=[]
for keys,g in BASELINE_OOF.groupby(['model','horizon_s','leg','axis','phase']):
    e=np.abs(g.pred-g.target); base_rows.append(dict(model=keys[0],horizon_s=keys[1],leg=keys[2],axis=keys[3],phase=keys[4],n=len(g),mae_N=float(e.mean()),p95_N=float(np.quantile(e,.95)),within4=float((e<=4).mean())))
BASELINE_METRICS=pd.DataFrame(base_rows); BASELINE_METRICS.to_csv(RUN_ROOT/'tables'/'baseline_force_metrics.csv',index=False)

def exact_state_ceiling():
    if not CFG.run_exact_state_ceiling: return pd.DataFrame()
    rows=[]
    for fold in CFG.outer_folds:
        df=FOLD_META[fold]; train=df[df.split=='train']; test=df[df.split=='test']; Xtr=[]; Ytr=[]; Xte=[]; Yte=[]
        def add(df0,X,Y):
            for _,row in df0.iterrows():
                s=make_sample_arrays(fold,row)
                for h,hs in enumerate(ROLLOUT_HORIZONS):
                    for l in range(4):
                        X.append(np.r_[s['y_state'][h,l],s['y_body'][h],s['y_foot'][h,l],s['y_contact'][h,l],np.eye(4)[l],hs]); Y.append(s['y_force10'][h,l])
        add(train,Xtr,Ytr); add(test,Xte,Yte); Xtr=np.asarray(Xtr,np.float32); Ytr=np.asarray(Ytr,np.float32); Xte=np.asarray(Xte,np.float32); Yte=np.asarray(Yte,np.float32)
        model=ExtraTreesRegressor(n_estimators=200,min_samples_leaf=2,max_features=.8,n_jobs=-1,random_state=CFG.seed+fold); model.fit(Xtr,Ytr); pred=model.predict(Xte).astype(np.float32); e=np.abs(pred-Yte); rows.append(dict(fold=fold,test_terrain=run_records[fold]['terrain'],n=len(Yte),mae_N=float(e.mean()),p95_N=float(np.quantile(e,.95)),within4=float((e<=4).mean()),max_N=float(e.max()))); np.savez_compressed(RUN_ROOT/'folds'/f'fold_{fold}'/'exact_state_predictions.npz',pred=pred,true=Yte)
    out=pd.DataFrame(rows); out.to_csv(RUN_ROOT/'tables'/'exact_state_ceiling.csv',index=False); return out

EXACT_STATE_METRICS=exact_state_ceiling(); display(EXACT_STATE_METRICS)

# %% notebook cell 42
if FORCE_OOF.empty: raise RuntimeError('No out-of-fold force rows exist; training/evaluation did not complete.')
plt.rcParams.update({'figure.dpi':150,'axes.grid':True,'font.size':9})

fig,axs=plt.subplots(1,3,figsize=(16,4))
for fold in CFG.outer_folds:
    fp=RUN_ROOT/'folds'/f'fold_{fold}'/'proposal_history.csv'; ff=RUN_ROOT/'folds'/f'fold_{fold}'/'force_history.csv'
    if fp.exists():
        q=pd.read_csv(fp); axs[0].plot(q.epoch,q.train,alpha=.65,label=f'fold {fold} train'); axs[0].plot(q.epoch,q.val_loss,linestyle='--',alpha=.8,label=f'fold {fold} val')
    if ff.exists():
        q=pd.read_csv(ff); axs[1].plot(q.epoch,q.train,alpha=.65,label=f'fold {fold} train'); axs[1].plot(q.epoch,q.val_loss,linestyle='--',alpha=.8,label=f'fold {fold} val'); axs[2].plot(q.epoch,q.val_component_p95,label=f'fold {fold}')
axs[0].set_title('proposal loss'); axs[1].set_title('force loss'); axs[2].set_title('validation force p95 (N)'); axs[0].set_xlabel('epoch'); axs[1].set_xlabel('epoch'); axs[2].set_xlabel('epoch'); axs[0].legend(fontsize=7,ncol=2); axs[1].legend(fontsize=7,ncol=2); axs[2].legend(fontsize=7)
fig.tight_layout(); fig.savefig(RUN_ROOT/'figures'/'training_curves.png'); plt.show()

agg=[]
for mode,g in FORCE_OOF.groupby(['haptic_mode','horizon_s']):
    e=np.abs(g.pred_mean10-g.true_mean10); agg.append(dict(haptic_mode=mode[0],horizon_s=mode[1],mae=float(e.mean()),p95=float(np.quantile(e,.95)),within4=float((e<=4).mean())))
agg=pd.DataFrame(agg)
fig,axs=plt.subplots(1,3,figsize=(15,4))
for mode,g in agg.groupby('haptic_mode'):
    axs[0].plot(g.horizon_s,g.mae,marker='o',label=mode); axs[1].plot(g.horizon_s,g.p95,marker='o',label=mode); axs[2].plot(g.horizon_s,g.within4,marker='o',label=mode)
for ax,title,y in zip(axs,['force component MAE','force component p95','within 4 N'],['N','N','fraction']): ax.set_title(title); ax.set_xlabel('horizon (s)'); ax.set_ylabel(y); ax.legend()
fig.tight_layout(); fig.savefig(RUN_ROOT/'figures'/'force_horizon_trends.png'); plt.show()

vg=FORCE_VECTOR_METRICS[FORCE_VECTOR_METRICS.haptic_mode=='prequential'].groupby('horizon_s',as_index=False).agg(vector_mae=('vector_mae_N','mean'),vector_p95=('vector_p95_N','max'),within4=('within4','mean'))
fig,axs=plt.subplots(1,3,figsize=(15,4)); axs[0].plot(vg.horizon_s,vg.vector_mae,marker='o'); axs[1].plot(vg.horizon_s,vg.vector_p95,marker='o',color='tab:red'); axs[2].plot(vg.horizon_s,vg.within4,marker='o',color='tab:green'); axs[0].set_title('3-D force vector MAE'); axs[1].set_title('worst 3-D force vector p95'); axs[2].set_title('3-D vectors within 4 N');
for ax,y in zip(axs,['N','N','fraction']): ax.set_xlabel('horizon (s)'); ax.set_ylabel(y)
fig.tight_layout(); fig.savefig(RUN_ROOT/'figures'/'force_vector_horizon_trends.png'); plt.show()

phase=FORCE_METRICS[FORCE_METRICS.haptic_mode=='prequential'].copy(); phase['cell']=phase.leg+'-'+phase.axis
pv=phase.pivot_table(index='cell',columns='phase',values='p95_N',aggfunc='max')
fig,ax=plt.subplots(figsize=(10,7)); sns.heatmap(pv,annot=True,fmt='.1f',cmap='magma',ax=ax); ax.set_title('Prequential worst p95 by leg/axis and phase (N)'); fig.tight_layout(); fig.savefig(RUN_ROOT/'figures'/'force_phase_heatmap.png'); plt.show()

fig,axs=plt.subplots(1,2,figsize=(13,4))
for kind,g in SENSOR_METRICS.groupby('kind'):
    gg=g.groupby('horizon_s',as_index=False).apply(lambda z: pd.Series({'mae':np.average(z.mae,weights=z.n),'p95':np.max(z.p95)}),include_groups=False).reset_index()
    axs[0].plot(gg.horizon_s,gg.mae,marker='o',label=kind); axs[1].plot(gg.horizon_s,gg.p95,marker='o',label=kind)
axs[0].set_title('sensor MAE vs horizon'); axs[1].set_title('sensor p95 vs horizon'); axs[0].set_xlabel('horizon (s)'); axs[1].set_xlabel('horizon (s)'); axs[0].legend(); axs[1].legend(); fig.tight_layout(); fig.savefig(RUN_ROOT/'figures'/'sensor_horizon_trends.png'); plt.show()

cal=FORCE_OOF[FORCE_OOF.haptic_mode=='prequential'].groupby('horizon_s',as_index=False).agg(coverage90=('covered90','mean'),width=('sigma90',lambda x:2*x.mean()))
fig,ax1=plt.subplots(figsize=(8,4)); ax1.plot(cal.horizon_s,cal.coverage90,marker='o',label='empirical coverage'); ax1.axhline(.9,color='k',linestyle='--',label='nominal 90%'); ax1.set_xlabel('horizon (s)'); ax1.set_ylabel('coverage'); ax2=ax1.twinx(); ax2.plot(cal.horizon_s,cal.width,color='tab:orange',marker='s',label='interval width'); ax2.set_ylabel('mean interval width (N)'); fig.tight_layout(); fig.savefig(RUN_ROOT/'figures'/'uncertainty_calibration.png'); plt.show(); cal.to_csv(RUN_ROOT/'tables'/'uncertainty_calibration.csv',index=False)

for fold in CFG.outer_folds:
    df=FOLD_META[fold]; test_ids=np.flatnonzero((df.split=='test').to_numpy());
    if not len(test_ids): continue
    k=int(test_ids[len(test_ids)//2]); row=df.iloc[k]; sub=FORCE_OOF[(FORCE_OOF.fold==fold)&(FORCE_OOF.origin_t==float(row.time))&(FORCE_OOF.haptic_mode=='prequential')&(FORCE_OOF.horizon_s==.1)]
    fig,axs=plt.subplots(4,3,figsize=(14,11),sharex=True); s=make_sample_arrays(fold,row)
    # Plot a local window of origins around the selected point at 100 ms horizon.
    ids=test_ids[max(0,len(test_ids)//2-80):min(len(test_ids),len(test_ids)//2+80)]; tt=df.iloc[ids].time.to_numpy()
    for l in range(4):
        for a,axis in enumerate(AXIS_NAMES):
            q=FORCE_OOF[(FORCE_OOF.fold==fold)&(FORCE_OOF.haptic_mode=='prequential')&(FORCE_OOF.horizon_s==.1)&(FORCE_OOF.leg==LEG_NAMES[l])&(FORCE_OOF.axis==axis)&(FORCE_OOF.origin_t.isin(tt))].sort_values('origin_t'); axs[l,a].plot(q.origin_t,q.true_mean10,label='true',lw=1.2); axs[l,a].plot(q.origin_t,q.pred_mean10,label='V10-R',lw=1.0); axs[l,a].set_title(f'{LEG_NAMES[l]} {axis}');
    axs[0,0].legend(); fig.suptitle(f'{run_records[fold]["terrain"]}: 0.10 s forecast over test trajectory'); fig.tight_layout(); fig.savefig(RUN_ROOT/'figures'/f'timeseries_{run_records[fold]["terrain"]}.png'); plt.show()

# %% notebook cell 44
def save_plotly(fig,path):
    path=Path(path); fig.write_html(str(path),include_plotlyjs='cdn')
    try: fig.write_image(str(path.with_suffix('.png')),width=1200,height=850,scale=1)
    except Exception as e: print('static Plotly export skipped:',repr(e))

def selected_test_row(fold):
    df=FOLD_META[fold]; ids=np.flatnonzero((df.split=='test').to_numpy()); return int(ids[len(ids)//2])

def make_geometry_figure(fold):
    k=selected_test_row(fold); row=FOLD_META[fold].iloc[k]; s=make_sample_arrays(fold,row); ri=int(row.ri); gm=GEOMETRY[ri]; p0=np.asarray(s['origin_world']); R0=np.asarray(s['R_origin']); snap=gm.snapshot(p0,float(s['origin_t']),radius=2.0,max_points=12000); xyz=(snap['xyz']-p0[None])@R0; rgb=np.clip(snap['rgb'],0,1); fig=go.Figure(); fig.add_trace(go.Scatter3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],mode='markers',marker=dict(size=2,color=['rgb(%d,%d,%d)'%tuple((c*255).astype(int)) for c in rgb]),name='causal fused voxels',text=snap['count']))
    true=s['y_foot'][EVAL_HORIZON_IDS[-1]]; pred=PROP_PRED[fold]['foot_mean'][k,EVAL_HORIZON_IDS[-1]]; fig.add_trace(go.Scatter3d(x=true[:,0],y=true[:,1],z=true[:,2],mode='markers',marker=dict(size=7,color='green',symbol='diamond'),name='true feet @1 s')); fig.add_trace(go.Scatter3d(x=pred[:,0],y=pred[:,1],z=pred[:,2],mode='markers',marker=dict(size=6,color='red',symbol='x'),name='predicted feet @1 s'))
    for l in range(4): fig.add_trace(go.Scatter3d(x=[s['foot_now'][l,0],pred[l,0]],y=[s['foot_now'][l,1],pred[l,1]],z=[s['foot_now'][l,2],pred[l,2]],mode='lines',line=dict(color='red',width=4),name=f'{LEG_NAMES[l]} predicted path',showlegend=l==0))
    fig.update_layout(title=f'{row.terrain}: causal 2 cm voxel-surface reconstruction at t={row.time:.2f}s',scene=dict(xaxis_title='x forward (m)',yaxis_title='y left (m)',zaxis_title='z up (m)',aspectmode='data'),legend=dict(orientation='h'))
    save_plotly(fig,RUN_ROOT/'figures'/f'geometry_reconstruction_{row.terrain}.html')

def make_force_vector_figure(fold):
    k=selected_test_row(fold); row=FOLD_META[fold].iloc[k]; s=make_sample_arrays(fold,row); h=int(EVAL_HORIZON_IDS[-1]); true=s['y_force10'][h]; pred=FORCE_PRED[fold]['prequential']['force10'][k,h]; pos=s['y_foot'][h]; fig=go.Figure(); fig.add_trace(go.Scatter3d(x=pos[:,0],y=pos[:,1],z=pos[:,2],mode='markers+text',text=list(LEG_NAMES),marker=dict(size=6,color='black'),name='feet'))
    scale=.0015
    for l in range(4):
        for vec,color,label in ((true[l],'green','true'),(pred[l],'red','pred')):
            end=pos[l]+scale*vec; fig.add_trace(go.Scatter3d(x=[pos[l,0],end[0]],y=[pos[l,1],end[1]],z=[pos[l,2],end[2]],mode='lines+markers',line=dict(color=color,width=6),marker=dict(size=3),name=f'{label} {LEG_NAMES[l]}',showlegend=(l==0)))
    fig.update_layout(title=f'{row.terrain}: true vs predicted 3-D force vectors at 1.0 s',scene=dict(xaxis_title='x forward (m)',yaxis_title='y left (m)',zaxis_title='z up (m)',aspectmode='data'),legend=dict(orientation='h'))
    save_plotly(fig,RUN_ROOT/'figures'/f'force_vectors_{row.terrain}.html')

for fold in CFG.outer_folds:
    make_geometry_figure(fold); make_force_vector_figure(fold)

# %% notebook cell 46
def shadow_candidates(fold,k,leg,hidx=9):
    df=FOLD_META[fold]; row=df.iloc[k]; s=make_sample_arrays(fold,row); p=PROP_PRED[fold]; nominal=p['foot_mean'][k,hidx,leg].copy(); p0=np.asarray(s['origin_world']); R0=np.asarray(s['R_origin']); gm=GEOMETRY[int(row.ri)]; grid=np.linspace(-.12,.12,9); cand=[]; base_p=PATCH[fold]
    for dx in grid:
        for dy in grid:
            local=nominal.copy(); local[:2]+=np.array([dx,dy],np.float32); world=p0+local@R0.T; q=gm.query(world,np.eye(3,dtype=float)*.002**2,float(s['origin_t']),R0); local[2]-=float(q['stats'][6]); world=p0+local@R0.T; hp=HAPTIC[fold][int(row.ri)].query(world,float(s['origin_t']),R0,enabled=True); mat=material_from_pairs(int(row.ri),q['pairs'],VIS_PCA[fold]); cand.append(dict(dx=dx,dy=dy,foot=local,world=world,points=q['points'],stats=q['stats'],material=mat,physical=hp,normal=q['normal'],mu=q['mu'],confidence=q['confidence']))
    C=len(cand); points=np.repeat(np.asarray(base_p['points'][k:k+1],np.float32),C,axis=0); stats=np.repeat(np.asarray(base_p['stats'][k:k+1],np.float32),C,axis=0); material=np.repeat(np.asarray(base_p['material'][k:k+1],np.float32),C,axis=0); physical=np.repeat(np.asarray(base_p['physical'][k:k+1],np.float32),C,axis=0); normal=np.repeat(np.asarray(base_p['normal'][k:k+1],np.float32),C,axis=0); mu=np.repeat(np.asarray(base_p['mu'][k:k+1],np.float32),C,axis=0); conf=np.repeat(np.asarray(base_p['confidence'][k:k+1],np.float32),C,axis=0); foot=np.repeat(p['foot_mean'][k:k+1],C,axis=0); prior=np.repeat(np.asarray(base_p['prior'][k:k+1],np.float32),C,axis=0); wrench=np.repeat(np.asarray(base_p['wrench'][k:k+1],np.float32),C,axis=0); leg_lat=np.repeat(p['latent_leg'][k:k+1],C,axis=0); body_lat=np.repeat(p['latent_body'][k:k+1],C,axis=0); state=np.repeat(p['state_mean'][k:k+1],C,axis=0); body=np.repeat(p['body_mean'][k:k+1],C,axis=0); contact=np.repeat(p['contact_prob'][k:k+1],C,axis=0); action=np.repeat(((s['action']-PROPOSAL_NORM[fold].action_mean)/PROPOSAL_NORM[fold].action_std)[None],C,axis=0)
    for c,x in enumerate(cand):
        points[c,hidx,leg]=x['points']; stats[c,hidx,leg]=x['stats']; material[c,hidx,leg]=x['material']; physical[c,hidx,leg]=x['physical']; normal[c,hidx,leg]=x['normal']; mu[c,hidx,leg]=x['mu']; conf[c,hidx,leg]=x['confidence']; foot[c,hidx,leg]=x['foot']
    # Recompute the train-only calibrated force/wrench priors for the altered
    # foothold.  Reusing the nominal prior would make this counterfactual
    # planner score a geometry change with a stale dynamics input.
    prior[:]=force_prior_from_prediction(PRIORS[fold],state,body,foot,contact)
    wrench[:]=wrench_prior_from_prediction(PRIORS[fold],body,foot,contact)
    model=FORCE_MODEL[fold].eval()
    with torch.inference_mode():
        args=[torch.from_numpy(x).to(DEVICE) for x in (leg_lat,body_lat,action,state,body,foot,contact,points,stats,material,physical,normal,mu,conf,prior,wrench)]; out=model(*args); f=out['force10'][:,hidx,leg].cpu().numpy(); sig=np.exp(out['logscale'][:,hidx,leg].cpu().numpy());
    rows=[]
    for c,x in enumerate(cand):
        st=x['stats']; geom=float(8*(1-st[18])+4*st[9]+4*st[7]+3*st[8]+2*max(0,.08-st[18])); force_risk=float(np.linalg.norm(f[c])+1.64*np.linalg.norm(sig[c])); reach=float(np.linalg.norm(x['foot']-s['foot_now'][leg])); feasible=bool(st[18]>=CFG.min_geometry_confidence and st[9]<.9 and st[8]<.25 and reach<.30); risk=geom+.025*force_risk+2.*reach+.8*np.linalg.norm([x['dx'],x['dy']]); rows.append(dict(fold=fold,run_id=row.run_id,terrain=row.terrain,origin_t=float(row.time),leg=LEG_NAMES[leg],horizon_s=float(ROLLOUT_HORIZONS[hidx]),dx=float(x['dx']),dy=float(x['dy']),x=float(x['foot'][0]),y=float(x['foot'][1]),z=float(x['foot'][2]),geometry_confidence=float(st[18]),slope_rad=float(st[9]),roughness_m=float(st[7]),step_m=float(st[8]),mu=float(x['mu']),reach_m=reach,force_norm_N=float(np.linalg.norm(f[c])),force_uncertainty_N=float(np.linalg.norm(sig[c])),risk=risk,feasible=feasible))
    return pd.DataFrame(rows)

def run_shadow_planner():
    if not CFG.run_shadow_planner: return pd.DataFrame()
    all_rows=[]
    for fold in CFG.outer_folds:
        k=selected_test_row(fold)
        for leg in range(4):
            out=shadow_candidates(fold,k,leg,hidx=min(len(ROLLOUT_HORIZONS)-1,9)); all_rows.append(out); best=out[out.feasible].sort_values('risk').head(1); print('fold',fold,'terrain',run_records[fold]['terrain'],'leg',LEG_NAMES[leg],'best',best[['x','y','z','risk']].to_dict('records'))
    out=pd.concat(all_rows,ignore_index=True); out.to_csv(RUN_ROOT/'planner'/'shadow_candidates.csv',index=False); decisions=out[out.feasible].sort_values('risk').groupby(['fold','leg'],as_index=False).head(1); decisions.to_csv(RUN_ROOT/'planner'/'shadow_decisions.csv',index=False); return out

SHADOW_CANDIDATES=run_shadow_planner()

# %% notebook cell 48
def export_bundle():
    readme=f'''# V10-R measured-results bundle\n\nConfiguration hash: {CONFIG_HASH}\nOuter folds: {list(CFG.outer_folds)}\nPrimary target: causal trailing 10 ms mean force\nEvaluation horizons: 0.10--1.00 s\n\nMeasured OOF force rows: {len(FORCE_OOF)}\nMeasured OOF sensor rows: {len(SENSOR_OOF)}\nMeasured touchdown-event rows: {len(EVENT_OOF)}\n\nOverall component MAE (prequential): {GATE["overall_component_mae_N"]:.6f} N\nWorst phase/leg/axis/horizon component p95 (prequential): {GATE["worst_phase_leg_axis_horizon_p95_N"]:.6f} N\nWorst 3-D vector p95 (prequential): {GATE["worst_vector_p95_N"]:.6f} N\nWithin 4 N fraction: {GATE["within4_fraction"]:.6f}\nHard 3--4 N gate: {GATE["hard_3_to_4N_gate"]}\n\nA failed gate is an empirical result and is not softened by removing touchdown samples.\n'''
    (RUN_ROOT/'RESULTS_README.md').write_text(readme)
    manifest=dict(config=asdict(CFG),config_hash=CONFIG_HASH,contract=CONTRACT_DF.to_dict('records'),force_audit=FORCE_AUDIT.to_dict('records'),gate=GATE,files=sorted(str(p.relative_to(RUN_ROOT)) for p in RUN_ROOT.rglob('*') if p.is_file()))
    (RUN_ROOT/'RESULTS_MANIFEST.json').write_text(json.dumps(manifest,indent=2,default=str))
    bundle=PROJECT_ROOT/f'{CFG.experiment_name}_{CONFIG_HASH}_RESULTS.zip'
    with shutil.make_archive(str(bundle.with_suffix('')),'zip',root_dir=RUN_ROOT): pass
    print('RUN COMPLETE')
    print('RESULTS_ROOT:',RUN_ROOT)
    print('RESULTS_ZIP :',bundle)
    print('CLAIM_GATE  :',json.dumps(GATE))
    return bundle

RESULTS_ZIP=export_bundle()
