# Source inspection export of the adjacent notebook.
# Notebook execution order and dependencies still apply.
# IPython magic, shell lines and prose-only conclusion cells are comments.

# %% notebook cell 1

# CELL 1 — SETUP. ONLY DATA/TIME LIMITS ARE TUNABLE; ARCHITECTURE IS LOCKED.
import os, sys, gc, re, json, math, time, random, warnings, subprocess, importlib
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
from IPython.display import display

from google.colab import drive
drive.mount('/content/drive', force_remount=False)

SEED = 230923
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type != 'cuda':
    raise RuntimeError('Use a Colab GPU runtime. The 20–30 minute profile is not a CPU profile.')

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---------------- ARCHITECTURE: DO NOT CHANGE ----------------
HISTORY_S = 1.0
SAMPLE_HZ = 100
HIST = 100
DT = 0.10
STEP = 10
H = 10
LEG_D = 37
P_DIM = 128
D_MODEL = 768
N_HEAD = 8
N_LAYER = 8
FFN_DIM = 3072
DROPOUT = 0.10
DINO_PATCHES = 256
DINO_DIM = 768
GEOM_POINTS = 32
GEOM_POINT_D = 9
GEOM_STATS_D = 18
PATCH_RADIUS_M = 0.07
PATCH_VOXEL_M = 0.015
VISUAL_MAX_RANGE_M = 4.0

assert (HIST, H, D_MODEL, N_HEAD, N_LAYER, FFN_DIM, DINO_PATCHES, DINO_DIM) == \
       (100,10,768,8,8,3072,256,768)

# ---------------- ONLY THESE ARE RUNTIME/DATA LIMITS ----------------
@dataclass
class FastBudget:
    project_root: str = '/content/drive/MyDrive/UGP_V7_FULL4B'
    raw_root: str = '/content/tartanground_v10'

    train_run_id: str = 'swamp_marsh_P2023'
    test_run_id: str = 'rough_uneven_P2000'

    # Main model: deliberately tiny diagnostic subset, architecture unchanged.
    max_train_origins: int = 18
    max_val_origins: int = 4
    max_test_origins: int = 12
    origin_stride_samples: int = 30
    train_segment_seconds: float = 7.0
    test_segment_seconds: float = 7.0

    # Frozen V10 foothold proposal is cheap and gets a slightly larger subset.
    proposal_train_origins: int = 96
    proposal_val_origins: int = 20
    proposal_epochs: int = 5
    proposal_train_minutes: float = 2.5

    # Exact 1032x768x8 model is expensive; hard cap keeps the session short.
    main_epochs: int = 2
    max_optimizer_steps: int = 10
    main_train_minutes: float = 13.0
    batch_size: int = 1
    grad_accum: int = 1
    lr: float = 2e-4
    weight_decay: float = 2e-4
    grad_clip: float = 1.0

    # Raw front RGB-D is only used for exact DINO/contact-visual mapping.
    # If it already exists locally, nothing is downloaded.
    allow_front_rgbd_download: bool = True
    dino_batch: int = 8
    dino_cache_dtype: str = 'float16'

BUDGET = FastBudget()
PROJECT_ROOT = Path(BUDGET.project_root)
RAW_ROOT = Path(BUDGET.raw_root)
STRICT_ROOT = PROJECT_ROOT/'V10R_STRICT_STAGED'
PREPARED_ROOTS = [
    STRICT_ROOT/'prepared_runs',
    Path('/content/V10R_STRICT_STAGED_RUNTIME/prepared_runs'),
]
CONTRACT_ROOT = STRICT_ROOT/'contracts'
C_ROOT = STRICT_ROOT/'stage_C_rgbd_incremental_v1'
C7R_FEATURE_ROOT = C_ROOT/'spatial_temporal_repair_v1'/'frame_features'
OUT_ROOT = PROJECT_ROOT/'V10_FEELWM_FROZEN_EXACT_FAST'
OUT_ROOT.mkdir(parents=True, exist_ok=True)

RUN_SPECS = {
    'swamp_marsh_P2023': dict(environment='GreatMarsh', run='P2023'),
    'rough_uneven_P2000': dict(environment='SeasonalForestWinter', run='P2000'),
}

T_ALL = time.time()
TIMING = {}
print(json.dumps(asdict(BUDGET), indent=2))
print('GPU:', torch.cuda.get_device_name(0))

# %% notebook cell 2

# CELL 2 — FAIL-FAST SCIENTIFIC PREFLIGHT + LOAD VERIFIED COMPACT RUNS

def find_prepared_root(run_id):
    for root in PREPARED_ROOTS:
        p=root/run_id
        if (p/'run_core.npz').is_file() and (p/'geometry'/'COMPLETE.json').is_file():
            return p
    return None

def load_json(path):
    with open(path,'r') as f: return json.load(f)

FORCE_CONTRACT = CONTRACT_ROOT/'force_frame_contract.json'
CALIB_CONTRACT = CONTRACT_ROOT/'calibration_contract.json'
SIGNAL_CONTRACT = CONTRACT_ROOT/'signal_semantics_contract.json'

for p in [FORCE_CONTRACT,CALIB_CONTRACT,SIGNAL_CONTRACT]:
    if not p.is_file():
        raise FileNotFoundError(f'Missing verified scientific contract: {p}')

fc, cc, sc = map(load_json,[FORCE_CONTRACT,CALIB_CONTRACT,SIGNAL_CONTRACT])
contract_errors=[]
if fc.get('verified') is not True: contract_errors.append('force-frame contract is not verified')
if cc.get('verified') is not True: contract_errors.append('calibration contract is not verified')
if sc.get('verified') is not True: contract_errors.append('signal contract is not verified')
if sc.get('imu_acceleration_semantics')!='specific_force_body':
    contract_errors.append('IMU acceleration is not verified as specific_force_body')
if sc.get('imu_gyro_frame')!='base':
    contract_errors.append('gyro frame is not verified as base')
if contract_errors:
    raise RuntimeError('Exact frozen implementation refuses unresolved contracts:\n- '+'\n- '.join(contract_errors))

def load_compact_run(run_id):
    root=find_prepared_root(run_id)
    if root is None:
        raise FileNotFoundError(
            f'{run_id}: verified run_core + geometry checkpoint not found in {PREPARED_ROOTS}. '
            'This exact notebook will not substitute a weaker cache.'
        )
    with np.load(root/'run_core.npz',allow_pickle=False) as z:
        d={k:np.asarray(z[k]) for k in z.files}
    required={
        'time','P','force_input_H','B','U','L_LEG','E','contact',
        'foot_world','R_HB','p_HB','touchdown_impulse'
    }
    miss=required-set(d)
    if miss:
        raise RuntimeError(f'{run_id}: strict core missing {sorted(miss)}')
    n=len(d['time'])
    if d['P'].shape!=(n,4,12) or d['force_input_H'].shape!=(n,4,3):
        raise RuntimeError(f'{run_id}: strict causal force/state shape contract failed')
    if d['foot_world'].shape!=(n,4,3) or d['R_HB'].shape!=(n,3,3):
        raise RuntimeError(f'{run_id}: strict mechanics shape contract failed')
    if not np.all(np.diff(np.asarray(d['time'],float))>0):
        raise RuntimeError(f'{run_id}: non-monotone strict timeline')

    events_path=root/'touchdown_events.csv'
    events=pd.read_csv(events_path) if events_path.is_file() else pd.DataFrame()

    geom_meta=load_json(root/'geometry'/'COMPLETE.json')
    if abs(float(geom_meta.get('voxel_m',PATCH_VOXEL_M))-PATCH_VOXEL_M)>1e-9:
        raise RuntimeError(
            f'{run_id}: geometry voxel is {geom_meta.get("voxel_m")}, frozen spec requires {PATCH_VOXEL_M}'
        )
    return dict(run_id=run_id, root=root, data=d, events=events, geom_meta=geom_meta)

t0=time.time()
RUNS={rid:load_compact_run(rid) for rid in [BUDGET.train_run_id,BUDGET.test_run_id]}
TIMING['compact_data_import_s']=time.time()-t0

print('Verified compact inputs:')
for rid,r in RUNS.items():
    print(rid, 'N=',len(r['data']['time']), 'root=',r['root'])
print(f'Compact data import: {TIMING["compact_data_import_s"]:.2f}s')

# %% notebook cell 3

# CELL 3 — EXACT INPUT CONSTRUCTION: 100x37 MAIN HISTORY + V10 PROPOSAL HISTORY + BODY ACTION

def causal_rate(x,t,clip=None):
    x=np.asarray(x,np.float32); t=np.asarray(t,np.float64)
    out=np.zeros_like(x,np.float32)
    if len(x)>1:
        dt=np.maximum(np.diff(t),1e-6)
        out[1:]=np.diff(x,axis=0)/dt.reshape((-1,)+(1,)*(x.ndim-1))
        out[0]=out[1]
    if clip is not None: out=np.clip(out,-clip,clip)
    return out

def build_main_leg37(d):
    # Frozen 37-D contract:
    # q3,dq3,tau3, causal force3, contact1, L3, E3, dE3, dF3, B12
    t=np.asarray(d['time'],np.float64)
    P=np.asarray(d['P'],np.float32)
    Fcausal=np.asarray(d['force_input_H'],np.float32)
    B=np.asarray(d['B'],np.float32)
    L=np.asarray(d['L_LEG'],np.float32)
    E=np.asarray(d['E'],np.float32)
    c=np.asarray(d['contact'],np.float32)
    dE=causal_rate(E,t,20.0)
    dF=causal_rate(Fcausal,t,2e4)
    B4=np.repeat(B[:,None,:],4,axis=1)
    x=np.concatenate([P[:,:,:9],Fcausal,c[...,None],L,E,dE,dF,B4],axis=-1)
    if x.shape[1:]!=(4,37): raise AssertionError(x.shape)
    return x.astype(np.float32)

def build_v10_prop_histories(d):
    # V10 history contract from the frozen V10 proposal branch:
    # 25 = P12 + L3 + E3 + U6 + contact1, 50 tokens over the previous 1 s at 50 Hz
    P=np.asarray(d['P'],np.float32).copy()
    P[:,:,9:12]=np.asarray(d['force_input_H'],np.float32)
    L=np.asarray(d['L_LEG'],np.float32)
    E=np.asarray(d['E'],np.float32)
    U=np.asarray(d['U'],np.float32)
    c=np.asarray(d['contact'],np.float32)
    U4=np.repeat(U[:,None,:],4,axis=1)
    leg25=np.concatenate([P,L,E,U4,c[...,None]],axis=-1)
    body18=np.concatenate([np.asarray(d['B'],np.float32),U],axis=-1)
    assert leg25.shape[1:]==(4,25) and body18.shape[1]==18
    return leg25.astype(np.float32),body18.astype(np.float32)

def body_action(d,i0,i1):
    R0=np.asarray(d['R_HB'][i0],float); R1=np.asarray(d['R_HB'][i1],float)
    dpH=np.asarray(d['p_HB'][i1]-d['p_HB'][i0],float)
    dpB=R0.T@dpH
    Rrel=R0.T@R1
    dyaw=math.atan2(float(Rrel[1,0]),float(Rrel[0,0]))
    dt=float(d['time'][i1]-d['time'][i0])
    return np.asarray([dpB[0],dpB[1],dyaw,dt],np.float32)

def contact_phase(contact):
    # 0 swing, 1 touchdown, 2 stance, 3 liftoff — derived only from the measured contact stream.
    c=np.asarray(contact)>0.5
    prev=np.vstack([c[:1],c[:-1]])
    y=np.zeros(c.shape,np.int64)
    y[(~prev)&c]=1
    y[prev&c]=2
    y[prev&(~c)]=3
    return y

for rid,r in RUNS.items():
    d=r['data']
    r['main37']=build_main_leg37(d)
    r['prop25'],r['prop_body18']=build_v10_prop_histories(d)
    r['phase']=contact_phase(d['contact'])
    print(rid,'main37',r['main37'].shape,'prop25',r['prop25'].shape)

# %% notebook cell 4

# CELL 4 — CAMERA-TIME IMPORT + FRONT RGB-D ONLY (REUSED IF ALREADY LOCAL)

def camera_times_path(run_id):
    p=C7R_FEATURE_ROOT/f'{run_id}_spatial_frame_features.npz'
    if p.is_file(): return p
    hits=list(PROJECT_ROOT.rglob(f'{run_id}*spatial_frame_features.npz'))
    return hits[0] if hits else None

def load_camera_times(run_id,d):
    p=camera_times_path(run_id)
    if p is None:
        raise FileNotFoundError(
            f'{run_id}: camera_times cache not found. Exact RGB/DINO alignment requires real camera timestamps.'
        )
    with np.load(p,allow_pickle=False) as z:
        if 'camera_times' not in z.files:
            raise RuntimeError(f'{p}: camera_times missing')
        t=np.asarray(z['camera_times'],np.float64)

    # Accept only an explicitly checkable alignment.
    phys=np.asarray(d['time'],float)
    overlap=lambda x: max(0.0,min(x.max(),phys.max())-max(x.min(),phys.min()))
    if overlap(t)>0.5*(phys.max()-phys.min()):
        return t
    crop=float(np.asarray(d.get('crop_start_original',0.0)).reshape(-1)[0])
    tc=t-crop
    if overlap(tc)>0.5*(phys.max()-phys.min()):
        return tc
    raise RuntimeError(
        f'{run_id}: camera/physics clocks do not overlap under direct or crop-start alignment.'
    )

def raw_run_root(run_id):
    s=RUN_SPECS[run_id]
    return RAW_ROOT/s['environment']/'Data_anymal'/s['run']

def raw_front_ready(run_id):
    root=raw_run_root(run_id)
    return (
        (root/'image_lcam_front').is_dir() and
        (root/'depth_lcam_front').is_dir() and
        (root/'pose_lcam_front.txt').is_file()
    )

def install_tartanair():
    if importlib.util.find_spec('tartanair') is None:
        subprocess.check_call([
            sys.executable,'-m','pip','install','-q',
            'git+https://github.com/castacks/tartanairpy.git'
        ])
        importlib.invalidate_caches()
    import tartanair as ta
    return ta

def ensure_front_rgbd(run_id):
    if raw_front_ready(run_id):
        print('Reusing local front RGB-D:',run_id)
        return
    if not BUDGET.allow_front_rgbd_download:
        raise FileNotFoundError(f'{run_id}: raw front RGB-D/pose missing and download disabled')
    s=RUN_SPECS[run_id]
    ta=install_tartanair()
    RAW_ROOT.mkdir(parents=True,exist_ok=True)
    ta.init(str(RAW_ROOT))
    print('Downloading ONLY front RGB + depth + metadata:',s)
    ta.download_ground(
        env=[s['environment']],version=['anymal'],traj=[s['run']],
        modality=['image','depth','meta'],
        camera_name=['lcam_front'],unzip=True,
    )
    if not raw_front_ready(run_id):
        raise RuntimeError(f'{run_id}: front RGB-D download completed but required files are still missing')

t0=time.time()
for rid,r in RUNS.items():
    r['camera_times']=load_camera_times(rid,r['data'])
    ensure_front_rgbd(rid)
TIMING['front_rgbd_import_s']=time.time()-t0
print(f'Front RGB-D import/download: {TIMING["front_rgbd_import_s"]/60:.2f} min')

# %% notebook cell 5

# CELL 5 — CHOOSE SMALL CONTIGUOUS TRAIN / VAL / FIXED ROUGH TEST SEQUENCES

def latest_camera_index(r,t):
    return int(np.searchsorted(r['camera_times'],float(t),side='right')-1)

def candidate_origins(r,start_s,duration_s,stride):
    d=r['data']; t=np.asarray(d['time'],float)
    lo=max(HIST-1,np.searchsorted(t,start_s,side='left'))
    hi=min(len(t)-H*STEP-1,np.searchsorted(t,start_s+duration_s,side='right')-1)
    out=[]
    for i in range(lo,hi+1,stride):
        vi=latest_camera_index(r,t[i])
        if vi<3: continue
        good=True
        for h in range(1,H+1):
            if latest_camera_index(r,t[i+h*STEP])<0:
                good=False; break
        if good: out.append(i)
    return np.asarray(out,np.int64)

def earliest_valid_start(r):
    t=np.asarray(r['data']['time'],float)
    return float(t[HIST-1]+0.5)

tr=RUNS[BUDGET.train_run_id]; te=RUNS[BUDGET.test_run_id]
train_start=earliest_valid_start(tr)
train_all=candidate_origins(tr,train_start,BUDGET.train_segment_seconds,BUDGET.origin_stride_samples)
if len(train_all)<6: raise RuntimeError('Too few valid training origins in the fast segment.')

# Train/val are temporally separated inside the training trajectory.
cut=max(1,int(round(.80*len(train_all))))
TRAIN_ORIGINS=train_all[:cut][:BUDGET.max_train_origins]
VAL_ORIGINS=train_all[cut:][:BUDGET.max_val_origins]
if len(VAL_ORIGINS)==0:
    VAL_ORIGINS=train_all[-min(BUDGET.max_val_origins,2):]
    TRAIN_ORIGINS=train_all[:-len(VAL_ORIGINS)][:BUDGET.max_train_origins]

test_start=earliest_valid_start(te)
test_all=candidate_origins(te,test_start,BUDGET.test_segment_seconds,BUDGET.origin_stride_samples)
TEST_ORIGINS=test_all[:BUDGET.max_test_origins]
if len(TEST_ORIGINS)==0: raise RuntimeError('No valid origins in the fixed rough test sequence.')

# Proposal gets more origins but from train trajectory only.
prop_all=np.arange(HIST-1,len(tr['data']['time'])-H*STEP-1,10,dtype=np.int64)
PROP_TRAIN=prop_all[:min(BUDGET.proposal_train_origins,len(prop_all))]
PROP_VAL=prop_all[
    min(BUDGET.proposal_train_origins,len(prop_all)):
    min(BUDGET.proposal_train_origins+BUDGET.proposal_val_origins,len(prop_all))
]
if len(PROP_VAL)==0: PROP_VAL=PROP_TRAIN[-min(8,len(PROP_TRAIN)):]

print('TRAIN RUN:',BUDGET.train_run_id, 'origins=',len(TRAIN_ORIGINS))
print('VAL RUN  :',BUDGET.train_run_id, 'origins=',len(VAL_ORIGINS))
print('TEST RUN :',BUDGET.test_run_id, 'origins=',len(TEST_ORIGINS))
print('TEST interval [s]:',
      float(te['data']['time'][TEST_ORIGINS.min()]),
      float(te['data']['time'][TEST_ORIGINS.max()+H*STEP]))

# %% notebook cell 6

# CELL 6 — TRAIN-ONLY NORMALIZATION

def moments(arrs,last_dim):
    x=np.concatenate([np.asarray(a).reshape(-1,last_dim) for a in arrs],0).astype(np.float64)
    m=x.mean(0).astype(np.float32); s=x.std(0).astype(np.float32)
    s[s<1e-6]=1.0
    return m,s

dtr=tr['data']
NORM={}
NORM['leg_mean'],NORM['leg_std']=moments([tr['main37']],37)
NORM['prop_leg_mean'],NORM['prop_leg_std']=moments([tr['prop25']],25)
NORM['prop_body_mean'],NORM['prop_body_std']=moments([tr['prop_body18']],18)
NORM['state_mean'],NORM['state_std']=moments([dtr['P'][:,:,:9]],9)
NORM['force_mean'],NORM['force_std']=moments([dtr['P'][:,:,9:12]],3)
NORM['body_mean'],NORM['body_std']=moments([dtr['B']],12)
NORM['foot_mean'],NORM['foot_std']=moments([dtr['foot_world']],3)

acts=[]
for i in PROP_TRAIN:
    for h in range(H):
        acts.append(body_action(dtr,i+h*STEP,i+(h+1)*STEP))
acts=np.stack(acts)
NORM['action_mean']=acts.mean(0).astype(np.float32)
NORM['action_std']=acts.std(0).astype(np.float32)
NORM['action_std'][NORM['action_std']<1e-6]=1.0

def norm_np(x,name):
    return ((np.asarray(x,np.float32)-NORM[name+'_mean'])/NORM[name+'_std']).astype(np.float32)
def unnorm_t(x,name):
    m=torch.as_tensor(NORM[name+'_mean'],device=x.device,dtype=x.dtype)
    s=torch.as_tensor(NORM[name+'_std'],device=x.device,dtype=x.dtype)
    return x*s+m

np.savez_compressed(OUT_ROOT/'normalization.npz',**NORM)
print('Normalization fit on training trajectory only.')

# %% notebook cell 7

# CELL 7 — FROZEN V10-STYLE FOOTHOLD PROPOSAL: TRAIN QUICKLY, THEN FREEZE

class CausalHistoryEncoder(nn.Module):
    def __init__(self,in_dim,d=128):
        super().__init__()
        self.inp=nn.Linear(in_dim,d)
        self.pos=nn.Parameter(torch.randn(50,d)*0.02)
        layer=nn.TransformerEncoderLayer(
            d_model=d,nhead=4,dim_feedforward=4*d,
            dropout=.1,activation='gelu',batch_first=True,norm_first=True
        )
        self.tx=nn.TransformerEncoder(layer,3)
        self.norm=nn.LayerNorm(d)
        mask=torch.triu(torch.ones(50,50,dtype=torch.bool),diagonal=1)
        self.register_buffer('mask',mask,persistent=False)
    def forward(self,x):
        h=self.inp(x)+self.pos[None]
        h=self.tx(h,mask=self.mask)
        return self.norm(h[:,-1])

class FrozenV10FootProposal(nn.Module):
    def __init__(self):
        super().__init__(); d=128
        self.leg_enc=CausalHistoryEncoder(25,d)
        self.body_enc=CausalHistoryEncoder(18,d)
        self.leg_id=nn.Parameter(torch.randn(4,d)*0.02)
        layer=nn.TransformerEncoderLayer(
            d_model=d,nhead=4,dim_feedforward=4*d,
            dropout=.1,activation='gelu',batch_first=True,norm_first=True
        )
        self.couple=nn.TransformerEncoder(layer,3)
        self.horizon=nn.Parameter(torch.randn(H,d)*0.02)
        self.action=nn.Sequential(nn.Linear(4,128),nn.GELU(),nn.LayerNorm(128))
        self.query=nn.Sequential(
            nn.Linear(4*d,192),nn.GELU(),nn.LayerNorm(192)
        )
        self.foot=nn.Linear(192,3)
        self.contact=nn.Linear(192,1)
        self.state=nn.Linear(192,9)
        self.body=nn.Sequential(nn.Linear(3*d,192),nn.GELU(),nn.Linear(192,12))

    def forward(self,leg50,body50,actions):
        B=leg50.shape[0]
        zl=torch.stack([self.leg_enc(leg50[:,i]) for i in range(4)],1)
        zb=self.body_enc(body50)
        toks=torch.cat([zb[:,None],zl+self.leg_id[None]],1)
        c=self.couple(toks)
        cb,cl=c[:,0],c[:,1:]
        foot=[]; contact=[]; state=[]; body=[]
        for h in range(H):
            ah=self.action(actions[:,h])
            ph=self.horizon[h].expand(B,-1)
            q=torch.cat([
                cl,
                cb[:,None].expand(-1,4,-1),
                ph[:,None].expand(-1,4,-1),
                ah[:,None].expand(-1,4,-1)
            ],-1)
            q=self.query(q)
            foot.append(self.foot(q))
            contact.append(self.contact(q).squeeze(-1))
            state.append(self.state(q))
            body.append(self.body(torch.cat([cb,ph,ah],-1)))
        return dict(
            foot=torch.stack(foot,1),
            contact_logit=torch.stack(contact,1),
            state=torch.stack(state,1),
            body=torch.stack(body,1),
        )

def proposal_item(r,i):
    d=r['data']
    ids=np.arange(i-98,i+1,2)  # 50 samples over 1 s
    if len(ids)!=50: raise AssertionError(len(ids))
    leg=norm_np(r['prop25'][ids].transpose(1,0,2),'prop_leg')
    body=norm_np(r['prop_body18'][ids],'prop_body')
    actions=np.stack([norm_np(body_action(d,i+h*STEP,i+(h+1)*STEP),'action') for h in range(H)])
    js=i+np.arange(1,H+1)*STEP
    return (
        torch.from_numpy(leg),torch.from_numpy(body),torch.from_numpy(actions),
        torch.from_numpy(norm_np(d['foot_world'][js],'foot')),
        torch.from_numpy(np.asarray(d['contact'][js],np.float32)),
        torch.from_numpy(norm_np(d['P'][js,:,:9],'state')),
        torch.from_numpy(norm_np(d['B'][js],'body')),
    )

proposal=FrozenV10FootProposal().to(DEVICE)
opt=torch.optim.AdamW(proposal.parameters(),lr=3e-4,weight_decay=2e-4)
scaler=torch.amp.GradScaler('cuda',enabled=True)

def proposal_epoch(origins,train=True):
    proposal.train(train); vals=[]
    seq=list(origins)
    if train: random.shuffle(seq)
    for i in seq:
        leg,body,act,yf,yc,ys,yb=proposal_item(tr,int(i))
        leg=leg[None].to(DEVICE); body=body[None].to(DEVICE); act=act[None].to(DEVICE)
        yf=yf[None].to(DEVICE); yc=yc[None].to(DEVICE); ys=ys[None].to(DEVICE); yb=yb[None].to(DEVICE)
        if train: opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda',dtype=torch.float16):
            o=proposal(leg,body,act)
            Lfoot=F.smooth_l1_loss(o['foot'],yf,beta=.25)
            Lstate=F.smooth_l1_loss(o['state'],ys,beta=.35)
            Lbody=F.smooth_l1_loss(o['body'],yb,beta=.35)
            Lcontact=F.binary_cross_entropy_with_logits(o['contact_logit'],yc)
            loss=2.5*Lfoot+1.0*Lstate+0.5*Lbody+0.6*Lcontact
        if train:
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(proposal.parameters(),2.0)
            scaler.step(opt); scaler.update()
        vals.append(float(loss.detach()))
    return float(np.mean(vals))

t0=time.time(); best=np.inf; best_state=None
for ep in range(1,BUDGET.proposal_epochs+1):
    if (time.time()-t0)/60>BUDGET.proposal_train_minutes: break
    trl=proposal_epoch(PROP_TRAIN,True)
    with torch.no_grad(): vl=proposal_epoch(PROP_VAL,False)
    print('proposal',ep,'train',round(trl,4),'val',round(vl,4))
    if vl<best:
        best=vl
        best_state={k:v.detach().cpu().clone() for k,v in proposal.state_dict().items()}
if best_state is None:
    best_state={k:v.detach().cpu().clone() for k,v in proposal.state_dict().items()}
proposal.load_state_dict(best_state)
proposal.eval()
for p in proposal.parameters(): p.requires_grad_(False)
torch.save({'model':proposal.state_dict(),'val':best},OUT_ROOT/'frozen_v10_foothold_proposal.pt')
TIMING['proposal_train_s']=time.time()-t0
print(f'Proposal training: {TIMING["proposal_train_s"]/60:.2f} min; now frozen.')

# %% notebook cell 8

# CELL 8 — EXACT DINOv2 ViT-B/14 256x768 TOKEN CACHE FOR ONLY THE NEEDED FRAMES

def index_files(root):
    out={}
    for p in Path(root).iterdir():
        if not p.is_file(): continue
        m=re.match(r'^(\d+)_',p.name)
        if m: out[int(m.group(1))]=p
    return out

def needed_frame_ids(r,origins):
    ids=set()
    t=r['data']['time']; ct=r['camera_times']
    if len(origins)==0: return ids
    # Exact causal visual memory: include all front frames from the beginning through
    # the last forecast origin in the selected small sequence.
    max_origin_t=float(t[max(origins)])
    max_vi=int(np.searchsorted(ct,max_origin_t,side='right')-1)
    ids.update(range(max_vi+1))
    # Plus true future target frames needed for the H=10 training/evaluation targets.
    for i in origins:
        vi=int(np.searchsorted(ct,float(t[i]),side='right')-1)
        ids.update(range(max(0,vi-3),vi+1))
        for h in range(1,H+1):
            v=int(np.searchsorted(ct,float(t[i+h*STEP]),side='right')-1)
            if v>=0: ids.add(v)
    return sorted(ids)

FRAME_IDS={}
FRAME_IDS[BUDGET.train_run_id]=needed_frame_ids(
    tr, np.unique(np.r_[TRAIN_ORIGINS,VAL_ORIGINS])
)
FRAME_IDS[BUDGET.test_run_id]=needed_frame_ids(te,TEST_ORIGINS)

print({k:len(v) for k,v in FRAME_IDS.items()})

# DINOv2 exact frozen backbone used by the locked architecture.
dino=torch.hub.load('facebookresearch/dinov2','dinov2_vitb14').to(DEVICE).eval()
for p in dino.parameters(): p.requires_grad_(False)

from PIL import Image
import torchvision.transforms as T
tfm=T.Compose([
    T.Resize(256,interpolation=T.InterpolationMode.BICUBIC,antialias=True),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225)),
])

DINO_CACHE={}
IMAGE_MAPS={}
DEPTH_MAPS={}

t0=time.time()
for rid,r in RUNS.items():
    raw=raw_run_root(rid)
    imap=index_files(raw/'image_lcam_front')
    dmap=index_files(raw/'depth_lcam_front')
    IMAGE_MAPS[rid]=imap; DEPTH_MAPS[rid]=dmap
    ids=[i for i in FRAME_IDS[rid] if i in imap]
    missing=[i for i in FRAME_IDS[rid] if i not in imap or i not in dmap]
    if missing:
        raise RuntimeError(f'{rid}: raw front RGB-D missing needed frame IDs, first={missing[:8]}')

    cache_file=Path('/content')/f'{rid}_DINOv2_vitb14_256x768.npz'
    cache={}
    if cache_file.is_file():
        with np.load(cache_file,allow_pickle=False) as z:
            old_ids=np.asarray(z['frame_ids'],int)
            old_tok=np.asarray(z['tokens'],np.float16)
        cache={int(i):old_tok[j] for j,i in enumerate(old_ids)}

    todo=[i for i in ids if i not in cache]
    print(rid,'DINO needed',len(ids),'new',len(todo))
    for a in range(0,len(todo),BUDGET.dino_batch):
        batch_ids=todo[a:a+BUDGET.dino_batch]
        x=torch.stack([tfm(Image.open(imap[i]).convert('RGB')) for i in batch_ids]).to(DEVICE)
        with torch.no_grad(), torch.amp.autocast('cuda',dtype=torch.float16):
            feat=dino.forward_features(x)['x_norm_patchtokens']
        if feat.shape[1:]!=(256,768): raise AssertionError(feat.shape)
        feat=feat.float().cpu().numpy().astype(np.float16)
        for j,i in enumerate(batch_ids): cache[int(i)]=feat[j]
    ordered=np.asarray(sorted(cache),np.int32)
    toks=np.stack([cache[int(i)] for i in ordered])
    np.savez_compressed(cache_file,frame_ids=ordered,tokens=toks)
    DINO_CACHE[rid]={int(i):cache[int(i)] for i in ids}

del dino
torch.cuda.empty_cache(); gc.collect()
TIMING['dino_extract_s']=time.time()-t0
print(f'Exact DINO token extraction/cache: {TIMING["dino_extract_s"]/60:.2f} min')

# %% notebook cell 9

# CELL 9 — CAUSAL METRIC GEOMETRY QUERY: EXACT 32x9 + 18 STATS

class CausalGeometry32x9_18:
    def __init__(self,run):
        root=run['root']/'geometry'
        meta=load_json(root/'COMPLETE.json')
        self.crop_start=float(np.asarray(run['data'].get('crop_start_original',0.0)).reshape(-1)[0])
        self.voxel=float(meta['voxel_m'])
        fields=('key','xyz','normal','rgb','xyz_var','count','camera_mask','time')
        pieces={k:[] for k in fields}
        for p in sorted((root/'chunks').glob('chunk_*.npz')):
            with np.load(p,allow_pickle=False) as z:
                for k in fields: pieces[k].append(np.asarray(z[k]))
        if not pieces['key']: raise RuntimeError(f'No geometry chunks: {root}')
        x={k:np.concatenate(v,0) for k,v in pieces.items()}
        order=np.lexsort((x['time'],x['key'][:,2],x['key'][:,1],x['key'][:,0]))
        for k in x: setattr(self,k,x[k][order])
        change=np.r_[True,np.any(np.diff(self.key,axis=0),axis=1)]
        self.starts=np.where(change)[0]
        self.ends=np.r_[self.starts[1:],len(self.key)]
        self.unique_key=self.key[self.starts]
        self.columns=defaultdict(list)
        for ui,k in enumerate(self.unique_key):
            self.columns[(int(k[0]),int(k[1]))].append(ui)

    def latest_candidates(self,q,t_cropped,radius=PATCH_RADIUS_M):
        q=np.asarray(q,float); t_original=float(t_cropped)+self.crop_start
        kr=int(math.ceil(radius/self.voxel)); center=np.floor(q/self.voxel).astype(int)
        ids=[]
        for dx in range(-kr,kr+1):
            for dy in range(-kr,kr+1):
                for ui in self.columns.get((center[0]+dx,center[1]+dy),[]):
                    a,b=self.starts[ui],self.ends[ui]
                    j=a+np.searchsorted(self.time[a:b],t_original,side='right')-1
                    if j>=a: ids.append(j)
        if not ids: return np.empty(0,int)
        ids=np.asarray(ids,int)
        dxy=np.linalg.norm(self.xyz[ids,:2]-q[None,:2],axis=1)
        ids=ids[dxy<=radius]
        if len(ids):
            dz=np.abs(self.xyz[ids,2]-q[2])
            near=dz<=max(.20,3*self.voxel)
            if near.any(): ids=ids[near]
        return ids

    @staticmethod
    def quadratic_shape(rel_xy,z):
        if len(z)<6:
            return 0.0,0.0,float(np.std(z)) if len(z) else 0.0
        x,y=rel_xy[:,0],rel_xy[:,1]
        A=np.c_[x*x,y*y,x*y,x,y,np.ones_like(x)]
        coef=np.linalg.lstsq(A,z,rcond=None)[0]
        pred=A@coef
        rough=float(np.std(z-pred))
        Hs=np.array([[2*coef[0],coef[2]],[coef[2],2*coef[1]]],float)
        ev=np.linalg.eigvalsh(Hs)
        return float(np.mean(np.abs(ev))),float(np.max(np.abs(ev))),rough

    def query(self,q,t_cropped):
        q=np.asarray(q,float)
        ids=self.latest_candidates(q,t_cropped,PATCH_RADIUS_M)
        pts=np.zeros((GEOM_POINTS,9),np.float32)
        stats=np.zeros(GEOM_STATS_D,np.float32)
        if len(ids)==0:
            return pts,stats

        dist3=np.linalg.norm(self.xyz[ids]-q[None],axis=1)
        order=np.argsort(dist3)
        ids=ids[order]; dist3=dist3[order]
        take=ids[:GEOM_POINTS]
        rel=self.xyz[take]-q[None]
        desc=np.c_[rel,self.normal[take],self.rgb[take]].astype(np.float32)
        pts[:len(take)]=desc

        xyz=self.xyz[ids]; relall=xyz-q[None]; z=relall[:,2]
        nmean=self.normal[ids].mean(0)
        nmean=nmean/max(np.linalg.norm(nmean),1e-8)
        curv,ridge,rough=self.quadratic_shape(relall[:,:2],z)
        age=(float(t_cropped)+self.crop_start-self.time[ids])
        stats[:]=[
            1.0,
            min(len(ids)/128.0,1.0),
            float(dist3.min()),
            float(np.mean(age)),
            float(np.mean(z)),
            float(np.std(z)),
            float(np.ptp(z)),
            float(nmean[0]),float(nmean[1]),float(nmean[2]),
            float(np.arccos(np.clip(abs(nmean[2]),0,1))),
            rough,
            curv,
            ridge,
            float(np.quantile(z,.95)-np.quantile(z,.05)),
            float(np.std(relall[:,0])),
            float(np.std(relall[:,1])),
            float(np.mean(np.linalg.norm(relall[:,:2],axis=1))),
        ]
        return pts,stats

print('Exact V10 geometry query class ready.')

# %% notebook cell 10

# CELL 10 — CAUSAL DINO SURFEL MEMORY: ATTACH EACH 256-DINO PATCH TOKEN TO REAL 3-D RGB-D POSITION

def read_depth(path):
    import cv2
    raw=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if raw is None: raise ValueError(path)
    if raw.dtype==np.uint8 and raw.ndim==3 and raw.shape[2]==4:
        d=np.ascontiguousarray(raw).view('<f4').reshape(raw.shape[:2])
    elif raw.dtype==np.float32 and raw.ndim==2:
        d=raw
    elif np.issubdtype(raw.dtype,np.integer):
        d=raw.astype(np.float32)/1000.0
    else:
        raise ValueError(f'unsupported depth {raw.dtype} {raw.shape}')
    d=np.asarray(d,np.float32); d[~np.isfinite(d)]=0
    return d

def patch_centers_original(width,height):
    # Standard frozen DINO preprocessing for 640x640:
    # resize square to 256, center-crop 224, 16x16 patches of size 14.
    # Mapping is exact for square TartanGround frames.
    if width!=height:
        raise RuntimeError('Frozen mapping currently requires the verified square TartanGround frame.')
    scale=width/256.0
    c=[]
    for iy in range(16):
        for ix in range(16):
            u=(16.0+(ix+0.5)*14.0)*scale
            v=(16.0+(iy+0.5)*14.0)*scale
            c.append((u,v))
    return np.asarray(c,np.float32)

class DINOContactMemory:
    def __init__(self,run_id,run):
        self.run_id=run_id; self.run=run
        meta=run['geom_meta']
        self.R_H_W=np.asarray(meta['R_H_W'],np.float64)
        self.p_W0=np.asarray(meta['p_W0'],np.float64)
        raw=raw_run_root(run_id)
        self.pose=np.loadtxt(raw/'pose_lcam_front.txt',dtype=np.float64)
        intr=cc['camera_intrinsics']['lcam_front']
        if intr.get('verified') is not True:
            raise RuntimeError('Front camera intrinsics are not verified')
        self.intr=intr
        self.centers=patch_centers_original(int(intr['width']),int(intr['height']))
        self.xyz=[]; self.t=[]; self.token=[]
        self._build()

    def _build(self):
        from scipy.spatial.transform import Rotation
        ct=self.run['camera_times']
        for fid in FRAME_IDS[self.run_id]:
            if fid not in DINO_CACHE[self.run_id]: continue
            if fid>=len(self.pose) or fid>=len(ct): continue
            d=read_depth(DEPTH_MAPS[self.run_id][fid])
            pose=self.pose[fid]
            R_WC=Rotation.from_quat(pose[3:7]).as_matrix()
            p_WC=pose[:3]
            xyz=[]; tok=[]
            tokens=np.asarray(DINO_CACHE[self.run_id][fid],np.float32)
            for j,(u,v) in enumerate(self.centers):
                ui=int(np.clip(round(float(u)),0,d.shape[1]-1))
                vi=int(np.clip(round(float(v)),0,d.shape[0]-1))
                Z=float(d[vi,ui])
                if not (.25<Z<VISUAL_MAX_RANGE_M): continue
                pC=np.array([
                    Z,
                    (u-float(self.intr['cx']))*Z/float(self.intr['fx']),
                    (v-float(self.intr['cy']))*Z/float(self.intr['fy'])
                ],float)
                pW=p_WC+R_WC@pC
                pH=self.R_H_W@(pW-self.p_W0)
                xyz.append(pH); tok.append(tokens[j])
            if xyz:
                self.xyz.append(np.asarray(xyz,np.float32))
                self.token.append(np.asarray(tok,np.float16))
                self.t.append(np.full(len(xyz),float(ct[fid]),np.float64))
        if self.xyz:
            self.xyz=np.concatenate(self.xyz,0)
            self.token=np.concatenate(self.token,0)
            self.t=np.concatenate(self.t,0)
        else:
            self.xyz=np.empty((0,3),np.float32)
            self.token=np.empty((0,768),np.float16)
            self.t=np.empty((0,),np.float64)

    def query(self,q,origin_t):
        q=np.asarray(q,float)
        if len(self.xyz)==0: return np.zeros(768,np.float32)
        ok=self.t<=float(origin_t)+1e-9
        ids=np.where(ok)[0]
        if not len(ids): return np.zeros(768,np.float32)
        dxy=np.linalg.norm(self.xyz[ids,:2]-q[None,:2],axis=1)
        dz=np.abs(self.xyz[ids,2]-q[2])
        ids=ids[(dxy<=PATCH_RADIUS_M)&(dz<=.20)]
        if not len(ids): return np.zeros(768,np.float32)
        dist=np.linalg.norm(self.xyz[ids]-q[None],axis=1)
        w=np.exp(-(dist**2)/(2*(PATCH_RADIUS_M/2)**2))
        w/=max(w.sum(),1e-12)
        return (np.asarray(self.token[ids],np.float32)*w[:,None]).sum(0).astype(np.float32)

t0=time.time()
VIS_MEM={rid:DINOContactMemory(rid,RUNS[rid]) for rid in RUNS}
TIMING['dino_contact_map_s']=time.time()-t0
print({rid:len(mem.xyz) for rid,mem in VIS_MEM.items()})
print(f'Causal DINO-contact map build: {TIMING["dino_contact_map_s"]/60:.2f} min')

# %% notebook cell 11

# CELL 11 — PRECOMPUTE FROZEN-PROPOSAL TERRAIN QUERIES FOR THE TINY MAIN-MODEL SUBSET

@torch.no_grad()
def proposal_predict_run(r,origins):
    proposal.eval()
    out={}
    for i in origins:
        # Same 50-token V10 history used during proposal training.
        ids=np.arange(int(i)-98,int(i)+1,2)
        leg=norm_np(r['prop25'][ids].transpose(1,0,2),'prop_leg')
        body=norm_np(r['prop_body18'][ids],'prop_body')
        acts=np.stack([
            norm_np(body_action(r['data'],int(i)+h*STEP,int(i)+(h+1)*STEP),'action')
            for h in range(H)
        ])
        o=proposal(
            torch.from_numpy(leg)[None].to(DEVICE),
            torch.from_numpy(body)[None].to(DEVICE),
            torch.from_numpy(acts)[None].to(DEVICE)
        )
        foot=unnorm_t(o['foot'],'foot')[0].float().cpu().numpy()
        out[int(i)]=foot
    return out

t0=time.time()
PRED_FOOT={
    BUDGET.train_run_id: proposal_predict_run(tr,np.unique(np.r_[TRAIN_ORIGINS,VAL_ORIGINS])),
    BUDGET.test_run_id: proposal_predict_run(te,TEST_ORIGINS),
}

PATCH_CACHE={}
for rid,origins in [
    (BUDGET.train_run_id,np.unique(np.r_[TRAIN_ORIGINS,VAL_ORIGINS])),
    (BUDGET.test_run_id,TEST_ORIGINS),
]:
    r=RUNS[rid]
    geom=CausalGeometry32x9_18(r)
    vmem=VIS_MEM[rid]
    for i in origins:
        origin_t=float(r['data']['time'][int(i)])
        pts=np.zeros((H,4,32,9),np.float32)
        stats=np.zeros((H,4,18),np.float32)
        vtok=np.zeros((H,4,768),np.float32)
        for h in range(H):
            for leg in range(4):
                q=PRED_FOOT[rid][int(i)][h,leg]
                pts[h,leg],stats[h,leg]=geom.query(q,origin_t)
                vtok[h,leg]=vmem.query(q,origin_t)
        PATCH_CACHE[(rid,int(i))]=dict(points=pts,stats=stats,visual=vtok)
    del geom; gc.collect()

TIMING['patch_precompute_s']=time.time()-t0
print(f'Frozen predicted-foot causal terrain query: {TIMING["patch_precompute_s"]/60:.2f} min')

# %% notebook cell 12

# CELL 12 — MAIN DATASET: EXACT FOUR DINO FRAMES + H=10 FUTURE TARGETS

def frame_tokens(r,t):
    vi=int(np.searchsorted(r['camera_times'],float(t),side='right')-1)
    if vi not in DINO_CACHE[r['run_id']]:
        raise KeyError(f'{r["run_id"]}: DINO frame {vi} not cached')
    return np.asarray(DINO_CACHE[r['run_id']][vi],np.float32)

def event_targets(r,i):
    # True touchdown occurrence, continuous offset within each 0.1-s bin,
    # and measured touchdown impulse. No synthetic event target is invented.
    d=r['data']; t=np.asarray(d['time'],float)
    hazard=np.zeros((H,4),np.float32)
    offset=np.zeros((H,4),np.float32)
    impulse=np.zeros((H,4,3),np.float32)
    mask=np.zeros((H,4),np.float32)

    ev=r['events']
    if len(ev) and 'leg_index' in ev.columns:
        time_col = 't_touchdown_cropped' if 't_touchdown_cropped' in ev.columns else \
                   ('time_cropped' if 'time_cropped' in ev.columns else None)
        if time_col is not None:
            t0=float(t[i])
            for _,row in ev.iterrows():
                li=int(row['leg_index'])
                rel=float(row[time_col])-t0
                if 0<rel<=H*DT and 0<=li<4:
                    h=min(H-1,max(0,int(math.ceil(rel/DT))-1))
                    if mask[h,li]: continue
                    center=(h+.5)*DT
                    hazard[h,li]=1.; offset[h,li]=rel-center; mask[h,li]=1.
                    j=i+(h+1)*STEP
                    impulse[h,li]=np.asarray(d['touchdown_impulse'][j,li],np.float32)
            return hazard,offset,impulse,mask

    # If exact event timestamps are unavailable, use measured touchdown bins only;
    # offset loss is masked rather than fabricated.
    c=np.asarray(d['contact'])>0.5
    prev=np.vstack([c[:1],c[:-1]])
    td=(~prev)&c
    for h in range(H):
        j=i+(h+1)*STEP
        hazard[h]=td[j].astype(np.float32)
        impulse[h]=np.asarray(d['touchdown_impulse'][j],np.float32)
    return hazard,offset,impulse,mask

class FrozenWorldDataset(Dataset):
    def __init__(self,rid,origins):
        self.rid=rid; self.r=RUNS[rid]; self.origins=np.asarray(origins,int)
    def __len__(self): return len(self.origins)
    def __getitem__(self,k):
        r=self.r; d=r['data']; i=int(self.origins[k])
        leg=norm_np(r['main37'][i-HIST+1:i+1].transpose(1,0,2),'leg')

        vi=int(np.searchsorted(r['camera_times'],float(d['time'][i]),side='right')-1)
        ctx=np.stack([np.asarray(DINO_CACHE[self.rid][j],np.float32) for j in range(vi-3,vi+1)])

        actions=[]; fvis=[]; fhist=[]; state=[]; force=[]; body=[]; phase=[]; foot=[]
        for h in range(H):
            j=i+(h+1)*STEP
            actions.append(norm_np(body_action(d,i+h*STEP,j),'action'))
            fvis.append(frame_tokens(r,d['time'][j]))
            fhist.append(norm_np(r['main37'][j-HIST+1:j+1].transpose(1,0,2),'leg'))
            state.append(norm_np(d['P'][j,:,:9],'state'))
            force.append(norm_np(d['P'][j,:,9:12],'force'))
            body.append(norm_np(d['B'][j],'body'))
            phase.append(r['phase'][j])
            foot.append(norm_np(d['foot_world'][j],'foot'))

        hazard,offset,impulse,evmask=event_targets(r,i)
        patch=PATCH_CACHE[(self.rid,i)]

        # Genuine failure labels are optional. If not present, the failure head exists
        # but its loss is masked to zero — no fake label.
        if 'failure' in d:
            failure=np.asarray([d['failure'][i+(h+1)*STEP] for h in range(H)],np.float32)
            failure_mask=np.ones(H,np.float32)
        else:
            failure=np.zeros(H,np.float32); failure_mask=np.zeros(H,np.float32)

        return dict(
            rid=self.rid,origin=np.int64(i),
            leg_hist=leg,visual_ctx=ctx,actions=np.stack(actions),
            future_visual=np.stack(fvis),future_hist=np.stack(fhist),
            geom_points=patch['points'],geom_stats=patch['stats'],contact_visual=patch['visual'],
            state=np.stack(state),force=np.stack(force),body=np.stack(body),
            phase=np.stack(phase),foot=np.stack(foot),
            event_hazard=hazard,event_offset=offset,event_impulse=impulse,event_mask=evmask,
            failure=failure,failure_mask=failure_mask,
        )

train_ds=FrozenWorldDataset(BUDGET.train_run_id,TRAIN_ORIGINS)
val_ds=FrozenWorldDataset(BUDGET.train_run_id,VAL_ORIGINS)
test_ds=FrozenWorldDataset(BUDGET.test_run_id,TEST_ORIGINS)

train_loader=DataLoader(train_ds,batch_size=1,shuffle=True,num_workers=0,pin_memory=True)
val_loader=DataLoader(val_ds,batch_size=1,shuffle=False,num_workers=0,pin_memory=True)
test_loader=DataLoader(test_ds,batch_size=1,shuffle=False,num_workers=0,pin_memory=True)

sample=next(iter(train_loader))
for k,v in sample.items():
    if torch.is_tensor(v): print(k,tuple(v.shape))

# %% notebook cell 13

# CELL 13 — EXACT FROZEN MAIN ARCHITECTURE

class LegHistoryCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1=nn.Conv1d(37,64,7,2,3)
        self.g1=nn.GroupNorm(8,64)
        self.c2=nn.Conv1d(64,128,5,2,2)
        self.g2=nn.GroupNorm(8,128)
        self.c3=nn.Conv1d(128,128,3,2,1)
        self.g3=nn.GroupNorm(8,128)
        self.c4=nn.Conv1d(128,128,3,1,1)
        self.g4=nn.GroupNorm(8,128)
        self.pool=nn.AdaptiveAvgPool1d(1)
        self.norm=nn.LayerNorm(128)
    def forward(self,x):
        x=x.transpose(1,2)
        x=F.gelu(self.g1(self.c1(x)))
        x=F.gelu(self.g2(self.c2(x)))
        x=F.gelu(self.g3(self.c3(x)))
        x=F.gelu(self.g4(self.c4(x)))
        return self.norm(self.pool(x).squeeze(-1))

class GeometryEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.point=nn.Sequential(
            nn.Linear(9,64),nn.GELU(),
            nn.Linear(64,96),nn.GELU()
        )
        self.stats=nn.Sequential(
            nn.LayerNorm(18),nn.Linear(18,64),nn.GELU()
        )
        self.fuse=nn.Sequential(
            nn.Linear(160,128),nn.GELU(),nn.LayerNorm(128)
        )
        self.project=nn.Linear(128,768)
    def forward(self,points,stats):
        hp=self.point(points).amax(dim=-2)
        hs=self.stats(stats)
        g=self.fuse(torch.cat([hp,hs],-1))
        return g,self.project(g)

class SharedFiLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk=nn.Sequential(nn.Linear(128,256),nn.GELU(),nn.LayerNorm(256))
        self.v_gamma=nn.Linear(256,768); self.v_beta=nn.Linear(256,768)
        self.g_gamma=nn.Linear(256,768); self.g_beta=nn.Linear(256,768)
        for m in [self.v_gamma,self.v_beta,self.g_gamma,self.g_beta]:
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
    def forward(self,x,c,kind):
        r=self.trunk(c)
        if kind=='visual':
            dg,b=self.v_gamma(r),self.v_beta(r)
        else:
            dg,b=self.g_gamma(r),self.g_beta(r)
        while dg.ndim<x.ndim:
            dg=dg.unsqueeze(-2); b=b.unsqueeze(-2)
        return x*(1.0+dg)+b

class ExactTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=768,nhead=8,dim_feedforward=3072,
                dropout=.1,activation='gelu',batch_first=True,norm_first=True
            ) for _ in range(8)
        ])
        self.norm=nn.LayerNorm(768)

        # 4 full visual frames + 4 contact visual + 4 geometry.
        time_ids=np.concatenate([
            np.full(256,0),np.full(256,1),np.full(256,2),np.full(256,3),
            np.full(4,3),np.full(4,3)
        ])
        assert len(time_ids)==1032
        # Block-causal: a token may attend any token at the same or an earlier time.
        mask=time_ids[None,:] > time_ids[:,None]
        self.register_buffer('causal_mask',torch.as_tensor(mask,dtype=torch.bool),persistent=False)

    def forward(self,x):
        for layer in self.layers:
            if self.training:
                mask=self.causal_mask
                x=checkpoint(
                    lambda y,layer=layer,mask=mask: layer(y,src_mask=mask),
                    x,use_reentrant=False
                )
            else:
                x=layer(x,src_mask=self.causal_mask)
        return self.norm(x)

class FrozenV10FeelWM(nn.Module):
    def __init__(self):
        super().__init__()
        self.leg_hist=LegHistoryCNN()
        self.leg_id_p=nn.Parameter(torch.randn(4,128)*.02)
        self.global_p=nn.Sequential(
            nn.Linear(512,256),nn.GELU(),nn.LayerNorm(256),
            nn.Linear(256,128),nn.LayerNorm(128)
        )
        self.action=nn.Sequential(
            nn.Linear(4,128),nn.GELU(),
            nn.Linear(128,128),nn.LayerNorm(128)
        )
        self.film=SharedFiLM()
        self.geom=GeometryEncoder()

        self.time_embed=nn.Parameter(torch.randn(4,768)*.02)
        self.visual_pos=nn.Parameter(torch.randn(256,768)*.02)
        self.modality=nn.Parameter(torch.randn(3,768)*.02)  # visual/contact/geometry
        self.leg_id_token=nn.Parameter(torch.randn(4,768)*.02)

        self.tx=ExactTransformer()
        self.visual_head=nn.Linear(768,768)

        self.leg_fuse=nn.Sequential(
            nn.Linear(2304,1024),nn.GELU(),nn.LayerNorm(1024),
            nn.Linear(1024,768),nn.GELU(),nn.LayerNorm(768)
        )
        self.p_head=nn.Linear(768,128)
        self.state_head=nn.Linear(768,9)
        self.force_head=nn.Linear(768,6) # mean3, log-sigma3
        self.phase_head=nn.Linear(768,4)
        self.foot_head=nn.Linear(768,3)
        self.touch_head=nn.Linear(768,1)
        self.event_offset_head=nn.Linear(768,1)
        self.impulse_head=nn.Linear(768,3)
        self.body_head=nn.Sequential(nn.Linear(768,768),nn.GELU(),nn.Linear(768,12))
        self.failure_head=nn.Linear(768,1)

    def encode_history(self,hist):
        # hist [B,4,100,37], shared weights.
        return torch.stack([self.leg_hist(hist[:,i]) for i in range(4)],1)

    def forward_step(self,visual_ctx,p_leg,action,contact_visual,geom_points,geom_stats):
        B=visual_ctx.shape[0]
        pG=self.global_p(p_leg.reshape(B,512))
        ea=self.action(action)
        cG=ea+pG
        ci=ea[:,None,:]+p_leg+self.leg_id_p[None,:,:]

        # Z_{t-3},Z_{t-2},Z_{t-1},tilde Z_t
        visual=[]
        for t in range(4):
            z=visual_ctx[:,t]+self.visual_pos[None]+self.time_embed[t][None,None]+self.modality[0]
            if t==3:
                z=self.film(z,cG,'visual')
            visual.append(z)
        visual=torch.cat(visual,1) # 1024x768

        # Four leg contact-visual tokens.
        vv=[]
        for i in range(4):
            x=self.film(contact_visual[:,i],ci[:,i],'visual')
            x=x+self.modality[1]+self.leg_id_token[i]
            vv.append(x)
        vv=torch.stack(vv,1)

        # Four V10 32x9+18 geometry tokens.
        _,gproj=self.geom(geom_points,geom_stats)
        gg=[]
        for i in range(4):
            x=self.film(gproj[:,i],ci[:,i],'geometry')
            x=x+self.modality[2]+self.leg_id_token[i]
            gg.append(x)
        gg=torch.stack(gg,1)

        x=torch.cat([visual,vv,gg],1)
        if x.shape[1:]!=(1032,768): raise AssertionError(x.shape)
        h=self.tx(x)

        # Frozen specification: current visual output + per-leg visual + per-leg geometry.
        cur=h[:,3*256:4*256]
        Hg=cur.mean(1)
        Hv=h[:,1024:1028]
        Hgeom=h[:,1028:1032]
        hi=self.leg_fuse(torch.cat([
            Hg[:,None,:].expand(-1,4,-1),Hv,Hgeom
        ],-1))

        f=self.force_head(hi)
        body=self.body_head(Hg)
        return dict(
            visual=self.visual_head(cur),
            p=self.p_head(hi),
            state=self.state_head(hi),
            force_mean=f[...,:3],
            force_logsigma=torch.clamp(f[...,3:],-4,4),
            phase_logits=self.phase_head(hi),
            foot=self.foot_head(hi),
            touchdown_logit=self.touch_head(hi).squeeze(-1),
            event_offset_s=.05*torch.tanh(self.event_offset_head(hi).squeeze(-1)),
            impulse=self.impulse_head(hi),
            body=body,
            failure_logit=self.failure_head(Hg).squeeze(-1),
        )

model=FrozenV10FeelWM().to(DEVICE)
npar=sum(p.numel() for p in model.parameters())
print(f'Frozen exact model parameters: {npar/1e6:.2f}M')
print('Locked transformer input:',1032,'x',768)

# %% notebook cell 14

# CELL 14 — EXACT LOSSES + H=10 RECURSIVE TRAINING, WITH ONLY A WALL-CLOCK/ORIGIN LIMIT

def to_device(batch):
    out={}
    for k,v in batch.items():
        out[k]=v.to(DEVICE,non_blocking=True) if torch.is_tensor(v) else v
    return out

def visual_loss(pred,target):
    return F.mse_loss(pred,target)+0.1*(1-F.cosine_similarity(pred,target,dim=-1).mean())

def gaussian_nll(mu,logsigma,target):
    return (0.5*((target-mu)*torch.exp(-logsigma))**2+logsigma).mean()

def tf_probability(epoch):
    # One rollout-level scheduled-sampling decision, as in the Feel-WM training description.
    if BUDGET.main_epochs<=1: return .5
    return 1.0 if epoch==1 else .5

optimizer=torch.optim.AdamW(
    model.parameters(),lr=BUDGET.lr,weight_decay=BUDGET.weight_decay
)
scaler=torch.amp.GradScaler('cuda',enabled=True)

def one_rollout(batch,epoch,train=True):
    b=to_device(batch)
    pcur=model.encode_history(b['leg_hist'].float())
    vis=b['visual_ctx'].float()
    tfmask=(torch.rand(vis.shape[0],device=DEVICE)<tf_probability(epoch)) if train \
           else torch.zeros(vis.shape[0],dtype=torch.bool,device=DEVICE)
    total_value=0.0

    if train: optimizer.zero_grad(set_to_none=True)

    for h in range(H):
        with torch.amp.autocast('cuda',dtype=torch.float16):
            o=model.forward_step(
                vis,pcur,b['actions'][:,h].float(),
                b['contact_visual'][:,h].float(),
                b['geom_points'][:,h].float(),
                b['geom_stats'][:,h].float()
            )
            with torch.no_grad():
                pgt=model.encode_history(b['future_hist'][:,h].float())

            Lv=visual_loss(o['visual'],b['future_visual'][:,h].float())
            Lp=F.mse_loss(o['p'],pgt)
            Lstate=F.smooth_l1_loss(o['state'],b['state'][:,h].float(),beta=.5)
            Lforce=gaussian_nll(o['force_mean'],o['force_logsigma'],b['force'][:,h].float())
            Lphase=F.cross_entropy(o['phase_logits'].reshape(-1,4),b['phase'][:,h].long().reshape(-1))
            Lfoot=F.smooth_l1_loss(o['foot'],b['foot'][:,h].float(),beta=.5)
            Ltouch=F.binary_cross_entropy_with_logits(
                o['touchdown_logit'],b['event_hazard'][:,h].float()
            )

            mask=b['event_mask'][:,h].float()
            if mask.sum()>0:
                Loff=((o['event_offset_s']-b['event_offset'][:,h].float()).abs()*mask).sum()/mask.sum()
                Limp=(F.smooth_l1_loss(
                    o['impulse'],b['event_impulse'][:,h].float(),reduction='none',beta=.5
                ).mean(-1)*mask).sum()/mask.sum()
            else:
                Loff=o['event_offset_s'].sum()*0
                Limp=o['impulse'].sum()*0

            Lbody=F.smooth_l1_loss(o['body'],b['body'][:,h].float(),beta=.5)

            fmask=b['failure_mask'][:,h].float()
            if fmask.sum()>0:
                Lfail=(F.binary_cross_entropy_with_logits(
                    o['failure_logit'],b['failure'][:,h].float(),reduction='none'
                )*fmask).sum()/fmask.sum()
            else:
                Lfail=o['failure_logit'].sum()*0

            # Frozen design specified these heads/loss families but not unequal lambdas;
            # use unit weighting rather than inventing a preference between heads.
            loss=(Lv+Lp+Lstate+Lforce+Lphase+Lfoot+Ltouch+Loff+Limp+Lbody+Lfail)/H

        total_value+=float(loss.detach())
        if train:
            scaler.scale(loss).backward()

        if h<H-1:
            # Recursive rollout; detach between world-model steps to control memory.
            # The next-state VALUE is predicted; only cross-step backpropagation is truncated.
            m=tfmask[:,None,None]
            nextv=torch.where(m,b['future_visual'][:,h].float(),o['visual'].detach())
            vis=torch.cat([vis[:,1:],nextv[:,None]],1)
            pcur=torch.where(m,pgt,o['p'].detach())

    if train:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(),BUDGET.grad_clip)
        scaler.step(optimizer); scaler.update()

    return total_value

@torch.no_grad()
def eval_loss(loader,epoch):
    model.eval(); vals=[]
    for batch in loader: vals.append(one_rollout(batch,epoch,False))
    return float(np.mean(vals)) if vals else np.nan

t0=time.time()
history=[]; best=np.inf; best_state=None; opt_steps=0
hard_stop=False

for epoch in range(1,BUDGET.main_epochs+1):
    model.train()
    vals=[]
    for batch in train_loader:
        if opt_steps>=BUDGET.max_optimizer_steps or (time.time()-t0)/60>=BUDGET.main_train_minutes:
            hard_stop=True; break
        v=one_rollout(batch,epoch,True)
        vals.append(v); opt_steps+=1
        print(f'epoch={epoch} step={opt_steps} loss={v:.4f} elapsed={(time.time()-t0)/60:.2f}m')
    vl=eval_loss(val_loader,epoch)
    rec=dict(epoch=epoch,optimizer_steps=opt_steps,train=np.mean(vals) if vals else np.nan,
             val=vl,elapsed_min=(time.time()-t0)/60)
    history.append(rec); print(rec)
    if np.isfinite(vl) and vl<best:
        best=vl
        best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if hard_stop: break

if best_state is None:
    best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
model.load_state_dict(best_state)
torch.save({
    'model':model.state_dict(),
    'architecture':'FROZEN_V10_FEELWM_V1_1032x768_8L8H_H10',
    'train_run':BUDGET.train_run_id,'test_run':BUDGET.test_run_id,
    'best_val':best,'budget':asdict(BUDGET)
},OUT_ROOT/'best_frozen_exact.pt')
pd.DataFrame(history).to_csv(OUT_ROOT/'training_history.csv',index=False)

TIMING['main_train_s']=time.time()-t0
print(f'MAIN TRAINING TIME = {TIMING["main_train_s"]/60:.2f} min')

# %% notebook cell 15

# CELL 15 — FULL H=10 AUTOREGRESSIVE HELD-OUT TEST

@torch.no_grad()
def test_model():
    model.eval()
    rows=[dict(force=[],q=[],dq=[],tau=[],foot=[],phase_y=[],phase_p=[],
               touch_y=[],touch_p=[],visual_cos=[],body=[]) for _ in range(H)]
    body_predictions=[]; body_targets=[]; origins=[]

    for batch in test_loader:
        b=to_device(batch)
        pcur=model.encode_history(b['leg_hist'].float())
        vis=b['visual_ctx'].float()
        origins.extend(batch['origin'].cpu().numpy().tolist())

        for h in range(H):
            o=model.forward_step(
                vis,pcur,b['actions'][:,h].float(),
                b['contact_visual'][:,h].float(),
                b['geom_points'][:,h].float(),
                b['geom_stats'][:,h].float()
            )
            r=rows[h]
            pf=unnorm_t(o['force_mean'],'force')
            tf=unnorm_t(b['force'][:,h].float(),'force')
            r['force'].append((pf-tf).abs().cpu().numpy().reshape(-1))

            ps=unnorm_t(o['state'],'state')
            ts=unnorm_t(b['state'][:,h].float(),'state')
            e=(ps-ts).abs().cpu().numpy()
            r['q'].append(e[...,:3].reshape(-1))
            r['dq'].append(e[...,3:6].reshape(-1))
            r['tau'].append(e[...,6:9].reshape(-1))

            pfoot=unnorm_t(o['foot'],'foot')
            tfoot=unnorm_t(b['foot'][:,h].float(),'foot')
            r['foot'].append(torch.linalg.norm(pfoot-tfoot,dim=-1).cpu().numpy().reshape(-1))

            r['phase_y'].append(b['phase'][:,h].cpu().numpy().reshape(-1))
            r['phase_p'].append(o['phase_logits'].argmax(-1).cpu().numpy().reshape(-1))
            r['touch_y'].append(b['event_hazard'][:,h].cpu().numpy().reshape(-1))
            r['touch_p'].append(torch.sigmoid(o['touchdown_logit']).cpu().numpy().reshape(-1))

            r['visual_cos'].append(
                F.cosine_similarity(o['visual'],b['future_visual'][:,h].float(),dim=-1)
                .cpu().numpy().reshape(-1)
            )
            pb=unnorm_t(o['body'],'body')
            tb=unnorm_t(b['body'][:,h].float(),'body')
            r['body'].append((pb-tb).abs().cpu().numpy().reshape(-1))

            if h==0:
                body_predictions.append(pb.cpu().numpy())
                body_targets.append(tb.cpu().numpy())

            # Test is fully autoregressive: no teacher forcing.
            vis=torch.cat([vis[:,1:],o['visual'].detach()[:,None]],1)
            pcur=o['p'].detach()

    def binary_f1(y,p):
        y=np.concatenate(y)>0.5; p=np.concatenate(p)>0.5
        tp=np.sum(y&p); fp=np.sum((~y)&p); fn=np.sum(y&(~p))
        return float(2*tp/max(2*tp+fp+fn,1))

    out=[]
    for h,r in enumerate(rows,1):
        fe=np.concatenate(r['force'])
        out.append(dict(
            horizon_s=h*.1,
            force_component_MAE_N=float(fe.mean()),
            force_component_p95_N=float(np.quantile(fe,.95)),
            q_MAE_rad=float(np.concatenate(r['q']).mean()),
            dq_MAE_rad_s=float(np.concatenate(r['dq']).mean()),
            tau_MAE=float(np.concatenate(r['tau']).mean()),
            foot_error_m=float(np.concatenate(r['foot']).mean()),
            contact_phase_accuracy=float(
                (np.concatenate(r['phase_y'])==np.concatenate(r['phase_p'])).mean()
            ),
            touchdown_F1=binary_f1(r['touch_y'],r['touch_p']),
            DINO_cosine=float(np.concatenate(r['visual_cos']).mean()),
            body_MAE=float(np.concatenate(r['body']).mean()),
        ))
    return pd.DataFrame(out),np.concatenate(body_predictions),np.concatenate(body_targets),np.asarray(origins,int)

t0=time.time()
TEST_METRICS,BODY_PRED_H1,BODY_TRUE_H1,TEST_ORIGIN_INDEX=test_model()
TIMING['test_s']=time.time()-t0
display(TEST_METRICS)
TEST_METRICS.to_csv(OUT_ROOT/'heldout_H10_metrics.csv',index=False)
print(f'TEST TIME = {TIMING["test_s"]/60:.2f} min')

# %% notebook cell 16

# CELL 16 — APPENDED FEEL-WM TABLE-II ROUGHNESS / YAW-VIBRATION COMPARISON
#
# Paper values are copied exactly from the user-supplied table image.
# Paper metric direction: lower is better.
#
# IMPORTANT: Feel-WM values are closed-loop Isaac-Sim navigation aggregates.
# This notebook's number is the realized ride quality of one fixed logged
# TartanGround/ANYmal test sequence. It is a scale comparison, not a method ranking.

def realized_ride_quality(run,origins):
    d=run['data']; t=np.asarray(d['time'],float)
    i0=int(np.min(origins))
    i1=int(np.max(origins)+H*STEP)
    idx=np.arange(i0,i1+1)

    R=np.asarray(d['R_HB'][idx],np.float64)
    aB=np.asarray(d['B'][idx,:3],np.float64)
    aH=np.einsum('nij,nj->ni',R,aB)
    rough=float(np.std(aH[:,2]))

    yaw=np.unwrap(np.arctan2(R[:,1,0],R[:,0,0]))
    yaw_rate=np.gradient(yaw,t[idx])
    yaw_vib=float(np.mean(np.abs(yaw_rate)))
    return rough,yaw_vib,float(t[i0]),float(t[i1])

rough,yawvib,t_start,t_end=realized_ride_quality(te,TEST_ORIGIN_INDEX)

COMPARE=pd.DataFrame([
    {'source':'Feel-WM paper — Isaac Sim Rough — Visual-only WM',
     'roughness_m_s2':6.46,'yaw_vibration_rad_s':0.73},
    {'source':'Feel-WM paper — Isaac Sim Rough — Feel-WM w/o prop.',
     'roughness_m_s2':6.53,'yaw_vibration_rad_s':0.77},
    {'source':'Feel-WM paper — Isaac Sim Rough — Feel-WM',
     'roughness_m_s2':6.51,'yaw_vibration_rad_s':0.65},
    {'source':f'OURS — measured fixed sequence — {BUDGET.test_run_id} [{t_start:.2f},{t_end:.2f}] s',
     'roughness_m_s2':rough,'yaw_vibration_rad_s':yawvib},
])
display(COMPARE)
COMPARE.to_csv(OUT_ROOT/'feelwm_rough_yaw_comparison.csv',index=False)

print('Specific test sequence:',BUDGET.test_run_id)
print(f't = {t_start:.3f} ... {t_end:.3f} s')
print(f'OURS realized roughness = {rough:.5f} m/s^2')
print(f'OURS realized yaw vibration = {yawvib:.5f} rad/s')
print(
    '\nInterpretation rule: do not call lower numbers here a win over Feel-WM; '
    'the domains and evaluation protocols are different.'
)

# %% notebook cell 17

# CELL 17 — EXACT IMPORT/TRAIN/TEST/TOTAL RUNTIME REPORT + SAVE RUN SUMMARY

TIMING['total_s']=time.time()-T_ALL

runtime=pd.DataFrame([
    ['verified compact data import',TIMING.get('compact_data_import_s',np.nan)],
    ['front RGB-D import/download',TIMING.get('front_rgbd_import_s',np.nan)],
    ['frozen V10 proposal training',TIMING.get('proposal_train_s',np.nan)],
    ['exact DINO extraction/cache',TIMING.get('dino_extract_s',np.nan)],
    ['causal DINO contact-map build',TIMING.get('dino_contact_map_s',np.nan)],
    ['predicted-foot patch precompute',TIMING.get('patch_precompute_s',np.nan)],
    ['main exact model training',TIMING.get('main_train_s',np.nan)],
    ['held-out H=10 testing',TIMING.get('test_s',np.nan)],
    ['TOTAL',TIMING.get('total_s',np.nan)],
],columns=['stage','seconds'])
runtime['minutes']=runtime['seconds']/60
display(runtime)
runtime.to_csv(OUT_ROOT/'runtime_report.csv',index=False)

summary={
    'architecture':{
        'leg_history':[4,100,37],
        'p_dim':128,
        'DINO_per_frame':[256,768],
        'transformer_input':[1032,768],
        'transformer_layers':8,
        'transformer_heads':8,
        'ffn_dim':3072,
        'rollout_steps':10,
        'rollout_dt_s':0.1,
        'geometry_points':[32,9],
        'geometry_stats':18,
        'patch_radius_m':0.07,
        'patch_voxel_m':0.015,
    },
    'budget':asdict(BUDGET),
    'timing_minutes':{k:float(v/60) for k,v in TIMING.items()},
    'train_origins':TRAIN_ORIGINS.tolist(),
    'val_origins':VAL_ORIGINS.tolist(),
    'test_origins':TEST_ORIGINS.tolist(),
    'test_metrics':TEST_METRICS.to_dict(orient='records'),
    'ride_quality_comparison':COMPARE.to_dict(orient='records'),
}
(OUT_ROOT/'run_summary.json').write_text(json.dumps(summary,indent=2))

print('Saved to:',OUT_ROOT)
for p in sorted(OUT_ROOT.iterdir()): print(' ',p.name)
print(f'\nTOTAL WALL TIME = {TIMING["total_s"]/60:.2f} min')
