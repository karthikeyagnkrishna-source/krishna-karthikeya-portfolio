# Source inspection export of the adjacent notebook.
# Notebook execution order and dependencies still apply.
# IPython magic, shell lines and prose-only conclusion cells are comments.

# %% notebook cell 1

# CELL 1 — PROMPT 2/3 ASSISTANT-PROPOSED ARCHITECTURE: SETUP + FAST-DATA BUDGET
# IPYTHON: !pip -q install tartanair opencv-python-headless

import os, time, math, json, random, gc, re, warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict

import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint

from scipy.spatial.transform import Rotation
from IPython.display import display
from google.colab import drive

drive.mount('/content/drive', force_remount=False)

SEED=230926
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type!='cuda':
    raise RuntimeError('This exact 1032×768, 8-layer architecture is intended for a Colab GPU.')
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32=True

T_GLOBAL=time.time()
TIMES={}

@dataclass
class CFG:
    project_root: str='/content/drive/MyDrive/UGP_V7_FULL4B'
    raw_root: str='/content/tartanground_v10'

    # Fixed held-out test sequence requested in the chat.
    test_run_id: str='C_rough_uneven_P2006'

    # Dataset-size limit only. Architecture is unchanged.
    max_train_windows: int=6
    max_val_windows: int=1
    max_test_windows: int=4
    max_train_runs: int=2

    # Test origins are non-overlapping 1-s rollouts.
    test_origin_spacing_s: float=1.0

    history_s: float=1.0
    sample_hz: int=100
    step_s: float=0.1
    H: int=10

    patch_radius_m: float=0.07
    patch_voxel_m: float=0.015
    max_contact_range_m: float=4.0

    p_dim: int=128
    d_model: int=768
    heads: int=8
    layers: int=8
    mlp_ratio: int=4
    dropout: float=0.1

    dino_image_size: int=224
    dino_patch: int=14
    dino_tokens: int=256
    dino_dim: int=768
    dino_batch: int=16

    train_epochs: int=1
    batch_size: int=1
    lr: float=2e-4
    weight_decay: float=1e-4
    grad_clip: float=1.0
    max_train_minutes: float=16.0

    download_raw_if_missing: bool=True
    download_workers: int=4

cfg=CFG()
HIST=int(round(cfg.history_s*cfg.sample_hz))
STEP=int(round(cfg.step_s*cfg.sample_hz))
assert HIST==100
assert STEP==10
assert cfg.H==10
assert cfg.dino_image_size//cfg.dino_patch==16
assert 16*16==cfg.dino_tokens
assert cfg.d_model==768 and cfg.layers==8 and cfg.heads==8 and cfg.mlp_ratio==4

print(json.dumps(asdict(cfg),indent=2))
print('GPU:',torch.cuda.get_device_name(0))

# %% notebook cell 2

# CELL 2 — PATH CONTRACTS + PREPARED-RUN AUDIT + FIXED TRAIN/VAL/TEST SELECTION
t0=time.time()

PROJECT_ROOT=Path(cfg.project_root)
RAW_ROOT=Path(cfg.raw_root)
STRICT_ROOT=PROJECT_ROOT/'V10R_STRICT_STAGED'
PREPARED_ROOT=STRICT_ROOT/'prepared_runs'
CONTRACT_ROOT=STRICT_ROOT/'contracts'
C_ROOT=STRICT_ROOT/'stage_C_rgbd_incremental_v1'
FEATURE_ROOT=C_ROOT/'spatial_temporal_repair_v1'/'frame_features'
SPLIT_PATH=C_ROOT/'C_split_manifest.csv'
OUT_ROOT=PROJECT_ROOT/'PROMPT2_3_ASSISTANT_PROPOSED_V10_FEELWM'
OUT_ROOT.mkdir(parents=True,exist_ok=True)

FORCE_CONTRACT_PATH=CONTRACT_ROOT/'force_frame_contract.json'
CALIBRATION_CONTRACT_PATH=CONTRACT_ROOT/'calibration_contract.json'
SIGNAL_CONTRACT_PATH=CONTRACT_ROOT/'signal_semantics_contract.json'

for p in [PREPARED_ROOT,SPLIT_PATH,FEATURE_ROOT,FORCE_CONTRACT_PATH,CALIBRATION_CONTRACT_PATH,SIGNAL_CONTRACT_PATH]:
    if not p.exists():
        raise FileNotFoundError(f'Missing required frozen-V10 artifact: {p}')

def load_json(p):
    with open(p) as f: return json.load(f)

force_contract=load_json(FORCE_CONTRACT_PATH)
cal_contract=load_json(CALIBRATION_CONTRACT_PATH)
signal_contract=load_json(SIGNAL_CONTRACT_PATH)

# Do not silently use unresolved signal/frame conventions.
if force_contract.get('verified') is not True:
    raise RuntimeError('force_frame_contract.json is not verified. Frozen architecture forbids guessing force axes/signs.')
if cal_contract.get('verified') is not True:
    raise RuntimeError('calibration_contract.json is not verified. Frozen architecture forbids guessing camera/base extrinsics.')
if signal_contract.get('verified') is not True:
    raise RuntimeError('signal_contract.json is not verified. Frozen architecture forbids guessing body/IMU semantics.')
if signal_contract.get('imu_acceleration_semantics')!='specific_force_body':
    raise RuntimeError('Expected verified IMU acceleration semantics = specific_force_body.')
if signal_contract.get('imu_gyro_frame')!='base':
    raise RuntimeError('Expected verified gyro frame = base.')

manifest=pd.read_csv(SPLIT_PATH)
required={'run_id','environment','run','terrain','role'}
if not required.issubset(manifest.columns):
    raise RuntimeError(f'C split manifest missing {sorted(required-set(manifest.columns))}')

def prepared_paths(rid):
    root=PREPARED_ROOT/rid
    return dict(
        root=root,
        core=root/'run_core.npz',
        events=root/'touchdown_events.csv',
        geometry=root/'geometry',
        geometry_complete=root/'geometry'/'COMPLETE.json',
    )

def feature_path(rid):
    return FEATURE_ROOT/f'{rid}_spatial_frame_features.npz'

audit=[]
for r in manifest.itertuples(index=False):
    pp=prepared_paths(str(r.run_id))
    audit.append(dict(
        run_id=str(r.run_id),terrain=str(r.terrain),role=str(r.role),
        environment=str(r.environment),run=str(r.run),
        core=pp['core'].is_file(),
        events=pp['events'].is_file(),
        geometry=pp['geometry_complete'].is_file(),
        frame_times=feature_path(str(r.run_id)).is_file(),
    ))
audit=pd.DataFrame(audit)
audit['ready']=audit[['core','events','geometry','frame_times']].all(axis=1)
display(audit)

ready_ids=set(audit.loc[audit.ready,'run_id'])
if cfg.test_run_id not in ready_ids:
    raise RuntimeError(
        f'Fixed test run {cfg.test_run_id} is not completely prepared. '
        'This notebook will not silently substitute another test sequence.'
    )

ready=manifest[manifest.run_id.astype(str).isin(ready_ids)].copy()
train_pool=ready[ready.role.astype(str).str.lower()=='train'].copy()
val_pool=ready[ready.role.astype(str).str.lower()=='val'].copy()

# Keep the frozen C test out of train/val.
train_pool=train_pool[train_pool.run_id.astype(str)!=cfg.test_run_id]
val_pool=val_pool[val_pool.run_id.astype(str)!=cfg.test_run_id]

if len(train_pool)==0 or len(val_pool)==0:
    raise RuntimeError('Need at least one prepared train run and one prepared validation run.')

# The only selection policy: take up to two existing TRAIN runs, preferring one rough and one swamp.
def terrain_text(df):
    return (df.run_id.astype(str)+' '+df.terrain.astype(str)+' '+df.environment.astype(str)).str.lower()

picked=[]
rough=train_pool[terrain_text(train_pool).str.contains('rough|uneven',regex=True)]
swamp=train_pool[terrain_text(train_pool).str.contains('swamp|marsh|mud',regex=True)]
if len(rough): picked.append(str(rough.iloc[0].run_id))
if len(swamp) and str(swamp.iloc[0].run_id) not in picked: picked.append(str(swamp.iloc[0].run_id))
for rid in train_pool.run_id.astype(str):
    if rid not in picked: picked.append(rid)
    if len(picked)>=cfg.max_train_runs: break
TRAIN_RUN_IDS=tuple(picked[:cfg.max_train_runs])

# Prefer rough validation because the held-out comparison is a rough test.
vrough=val_pool[terrain_text(val_pool).str.contains('rough|uneven',regex=True)]
VAL_RUN_ID=str((vrough if len(vrough) else val_pool).iloc[0].run_id)
TEST_RUN_ID=cfg.test_run_id

SELECTED_RUN_IDS=tuple(dict.fromkeys([*TRAIN_RUN_IDS,VAL_RUN_ID,TEST_RUN_ID]))
row_by_id={str(r.run_id):r for r in manifest.itertuples(index=False)}

assert TEST_RUN_ID not in TRAIN_RUN_IDS
assert TEST_RUN_ID!=VAL_RUN_ID

print('TRAIN:',TRAIN_RUN_IDS)
print('VAL  :',VAL_RUN_ID)
print('TEST :',TEST_RUN_ID)

TIMES['audit_split_s']=time.time()-t0

# %% notebook cell 3

# CELL 3 — LOAD STRICT PREPARED RUNS; BUILD EXACT 25-D V10-PROPOSAL INPUT AND 37-D NEW p_i INPUT
t0=time.time()

def causal_rate(x,t,clip=None):
    x=np.asarray(x,np.float32); t=np.asarray(t,np.float64)
    out=np.zeros_like(x,dtype=np.float32)
    if len(x)>1:
        dt=np.maximum(np.diff(t),1e-6)
        out[1:]=np.diff(x,axis=0)/dt.reshape((-1,)+(1,)*(x.ndim-1))
        out[0]=out[1]
    if clip is not None: out=np.clip(out,-clip,clip)
    return out

def load_run(rid):
    pp=prepared_paths(rid)
    with np.load(pp['core'],allow_pickle=False) as z:
        d={k:np.asarray(z[k]) for k in z.files}

    needed={'time','P','force_input_H','B','U','L_LEG','E','contact',
            'foot_world','R_HB','p_HB','crop_start_original'}
    miss=needed-set(d)
    if miss:
        raise RuntimeError(f'{rid}: prepared core missing {sorted(miss)}')

    t=np.asarray(d['time'],np.float64)
    P=np.asarray(d['P'],np.float32).copy()
    P_input=P.copy()
    P_input[...,9:12]=np.asarray(d['force_input_H'],np.float32)
    B=np.asarray(d['B'],np.float32)
    U=np.asarray(d['U'],np.float32)
    L=np.asarray(d['L_LEG'],np.float32)
    E=np.asarray(d['E'],np.float32)
    contact=np.asarray(d['contact'],np.float32)
    foot=np.asarray(d['foot_world'],np.float32)
    R_HB=np.asarray(d['R_HB'],np.float32)
    p_HB=np.asarray(d['p_HB'],np.float32)

    n=len(t)
    if P.shape!=(n,4,12): raise RuntimeError(f'{rid}: bad P shape {P.shape}')
    if L.shape!=(n,4,3) or E.shape!=(n,4,3) or contact.shape!=(n,4):
        raise RuntimeError(f'{rid}: bad leg-array shapes')

    dE=causal_rate(E,t,20.0)
    dF=causal_rate(P_input[...,9:12],t,2e4)

    # Exact strict V10-R proposal input:
    # 25 = P12 (with causal force input) + contact1 + L3 + E3 + dE3 + dF3
    leg25=np.concatenate([P_input,contact[...,None],L,E,dE,dF],axis=-1).astype(np.float32)
    if leg25.shape!=(n,4,25): raise AssertionError(leg25.shape)

    # Frozen merged architecture p_i input:
    # 37 = q3,dq3,tau3,Fcausal3,contact1,L3,E3,dE3,dF3,B12
    B4=np.repeat(B[:,None,:],4,axis=1)
    leg37=np.concatenate([P_input,contact[...,None],L,E,dE,dF,B4],axis=-1).astype(np.float32)
    if leg37.shape!=(n,4,37): raise AssertionError(leg37.shape)

    body18=np.concatenate([B,U],axis=-1).astype(np.float32)
    if body18.shape!=(n,18): raise AssertionError(body18.shape)

    # Camera time index comes from the already-built C-series frame cache.
    with np.load(feature_path(rid),allow_pickle=False) as z:
        if 'camera_times' not in z.files:
            raise RuntimeError(f'{rid}: camera_times missing from Stage-C frame cache.')
        camera_times=np.asarray(z['camera_times'],np.float64)
        frame_ids=None
        for k in ('camera_frame_ids','frame_ids','frame_id'):
            if k in z.files:
                frame_ids=np.asarray(z[k]).astype(int)
                break

    vi=np.searchsorted(camera_times,t,side='right')-1

    events=pd.read_csv(pp['events'])
    required_event={'leg_index','t_touchdown_cropped','Jx','Jy','Jz'}
    if not required_event.issubset(events.columns):
        raise RuntimeError(f'{rid}: touchdown_events.csv missing {sorted(required_event-set(events.columns))}')

    return dict(
        rid=rid,time=t,P=P,P_input=P_input,B=B,U=U,L=L,E=E,contact=contact,
        foot=foot,R_HB=R_HB,p_HB=p_HB,leg25=leg25,leg37=leg37,body18=body18,
        crop_start=float(np.asarray(d['crop_start_original']).item()),
        camera_times=camera_times,camera_frame_ids=frame_ids,visual_index=vi,
        events=events,failure=(np.asarray(d['failure'],np.float32) if 'failure' in d else None),
    )

RUNS={rid:load_run(rid) for rid in SELECTED_RUN_IDS}
print({rid:len(d['time']) for rid,d in RUNS.items()})

TIMES['prepared_load_s']=time.time()-t0
print(f'prepared-data load: {TIMES["prepared_load_s"]:.1f}s')

# %% notebook cell 4

# CELL 4 — LOAD THE EXISTING FROZEN V10-R FOOT PROPOSAL + ITS TRAIN-ONLY NORMALIZER
t0=time.time()

class MLP(nn.Module):
    def __init__(self,din,dout,hidden=128,depth=2):
        super().__init__(); layers=[]
        for i in range(depth-1):
            layers += [nn.Linear(din if i==0 else hidden,hidden),nn.SiLU(),nn.LayerNorm(hidden)]
        layers += [nn.Linear(din if depth==1 else hidden,dout)]
        self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

class CausalHistoryEncoder(nn.Module):
    def __init__(self,din,d=128,layers=3,heads=4,max_t=128):
        super().__init__()
        self.proj=nn.Linear(din,d)
        self.pos=nn.Parameter(torch.zeros(1,max_t,d))
        block=nn.TransformerEncoderLayer(d,heads,4*d,.1,batch_first=True,norm_first=True,activation='gelu')
        self.enc=nn.TransformerEncoder(block,layers)
        self.norm=nn.LayerNorm(d)
    def forward(self,x):
        z=self.proj(x)+self.pos[:,:x.shape[1]]
        mask=torch.triu(torch.ones(x.shape[1],x.shape[1],device=x.device,dtype=torch.bool),1)
        return self.norm(self.enc(z,mask=mask)[:,-1])

class FourLegRelationalBlock(nn.Module):
    def __init__(self,d=128):
        super().__init__()
        self.edge=MLP(2*d+3,d); self.node=MLP(2*d,d); self.body=MLP(2*d,d)
    def forward(self,leg,body,foot_xyz):
        msgs=[]
        for i in range(4):
            m=[]
            for j in range(4):
                if i==j: continue
                m.append(self.edge(torch.cat([leg[:,i],leg[:,j],foot_xyz[:,j]-foot_xyz[:,i]],-1)))
            msgs.append(torch.stack(m,1).mean(1))
        msg=torch.stack(msgs,1)
        leg=leg+self.node(torch.cat([leg,msg],-1))+body[:,None]
        body=body+self.body(torch.cat([body,leg.mean(1)],-1))
        return leg,body

class FrozenV10Proposal(nn.Module):
    def __init__(self,d=128,H=10):
        super().__init__(); self.H=H
        self.leg_hist=CausalHistoryEncoder(25,d)
        self.body_hist=CausalHistoryEncoder(18,d)
        self.rel=FourLegRelationalBlock(d)
        self.action=MLP(18,d)
        self.step_embed=nn.Parameter(torch.randn(H,d)*.01)
        self.leg_step=nn.GRUCell(2*d,d)
        self.body_step=nn.GRUCell(2*d,d)
        self.proposal_head=MLP(3*d,4+1+3+6+18)
        self.body_out=MLP(3*d,24)

    def propose(self,leg_hist,body_hist,action_plan,foot_now):
        b=leg_hist.shape[0]
        leg=torch.stack([self.leg_hist(leg_hist[:,i]) for i in range(4)],1)
        body=self.body_hist(body_hist)
        actions=self.action(action_plan)
        prop=[]; foot_state=foot_now
        for k in range(self.H):
            a=actions[:,k]+self.step_embed[k]
            leg,body=self.rel(leg,body,foot_state)
            leg=self.leg_step(
                torch.cat([a[:,None].expand(-1,4,-1),body[:,None].expand(-1,4,-1)],-1).reshape(b*4,-1),
                leg.reshape(b*4,-1),
            ).reshape(b,4,-1)
            body=self.body_step(torch.cat([a,leg.mean(1)],-1),body)
            shared=torch.cat([leg,body[:,None].expand(-1,4,-1),a[:,None].expand(-1,4,-1)],-1)
            pk=self.proposal_head(shared)
            foot_state=foot_state+pk[...,5:8]
            prop.append(pk)
        prop=torch.stack(prop,1)
        # Preserve the implementation used in V10-R.
        foot_mean=foot_now[:,None]+torch.cumsum(prop[...,5:8],dim=1)
        return foot_mean

# Locate a previously trained strict V10-R checkpoint. Do NOT train a replacement here.
model_root=STRICT_ROOT/'models'
ckpts=list(model_root.glob('**/best_G4.pt'))
if not ckpts: ckpts=list(model_root.glob('**/best_G2.pt'))
if not ckpts:
    raise FileNotFoundError(
        f'No existing V10-R best_G4.pt / best_G2.pt under {model_root}. '
        'The frozen architecture requires the existing V10 proposal; this notebook will not replace it with a new MLP.'
    )
PROPOSAL_CKPT=max(ckpts,key=lambda p:p.stat().st_mtime)
NORM_PATH=PROPOSAL_CKPT.parent/'input_normalizer.npz'
if not NORM_PATH.is_file():
    raise FileNotFoundError(f'Frozen V10 proposal normalizer missing: {NORM_PATH}')

v10proposal=FrozenV10Proposal(H=cfg.H).to(DEVICE)
ck=torch.load(PROPOSAL_CKPT,map_location=DEVICE,weights_only=False)
sd=ck['model'] if isinstance(ck,dict) and 'model' in ck else ck

# StrictV10R checkpoints store the proposal inside "core.*".
if any(k.startswith('core.') for k in sd):
    sd={k[5:]:v for k,v in sd.items() if k.startswith('core.')}

wanted_prefixes=('leg_hist.','body_hist.','rel.','action.','step_embed',
                 'leg_step.','body_step.','proposal_head.','body_out.')
proposal_sd={k:v for k,v in sd.items() if any(k.startswith(p) for p in wanted_prefixes)}
missing,unexpected=v10proposal.load_state_dict(proposal_sd,strict=False)

essential=('leg_hist','body_hist','rel','action','step_embed','leg_step','body_step','proposal_head')
bad=[x for x in missing if any(x.startswith(e) for e in essential)]
if bad:
    raise RuntimeError(f'Existing checkpoint does not contain the complete frozen V10 proposal: {bad[:20]}')

with np.load(NORM_PATH,allow_pickle=False) as z:
    PN={k:np.asarray(z[k],np.float32) for k in z.files}
for k in ('leg_mean','leg_std','body_mean','body_std','action_mean','action_std'):
    if k not in PN: raise RuntimeError(f'Frozen proposal normalizer missing {k}')

v10proposal.eval()
for p in v10proposal.parameters(): p.requires_grad_(False)

print('Frozen V10 proposal checkpoint:',PROPOSAL_CKPT)
TIMES['proposal_load_s']=time.time()-t0

# %% notebook cell 5

# CELL 5 — SELECT ONLY A FEW ORIGINS; TEST IS A CONTIGUOUS NON-OVERLAPPING SEQUENCE
t0=time.time()

def valid_origins(d):
    n=len(d['time'])
    lo=HIST-1
    hi=n-cfg.H*STEP-1
    out=[]
    for i in range(lo,hi+1,10):  # candidate origin every 0.1 s
        vi=int(d['visual_index'][i])
        if vi<3: continue
        ok=True
        for h in range(1,cfg.H+1):
            j=i+h*STEP
            if int(d['visual_index'][j])<0:
                ok=False; break
        if ok: out.append(i)
    return np.asarray(out,np.int64)

VALID={rid:valid_origins(d) for rid,d in RUNS.items()}
print({rid:len(v) for rid,v in VALID.items()})
if any(len(v)==0 for v in VALID.values()):
    raise RuntimeError('A selected run has no valid 1-s history + 1-s future visual windows.')

rng=np.random.default_rng(SEED)
train_samples=[]
per=max(1,math.ceil(cfg.max_train_windows/max(len(TRAIN_RUN_IDS),1)))
for rid in TRAIN_RUN_IDS:
    v=VALID[rid]
    take=min(per,len(v))
    ids=np.sort(rng.choice(v,take,replace=False))
    train_samples.extend((rid,int(i)) for i in ids)
train_samples=train_samples[:cfg.max_train_windows]

vv=VALID[VAL_RUN_ID]
val_samples=[(VAL_RUN_ID,int(vv[len(vv)//2]))][:cfg.max_val_windows]

# Fixed specific test sequence: earliest valid origins, 1.0 s apart, so rollouts do not overlap.
tv=VALID[TEST_RUN_ID]
spacing=max(1,int(round(cfg.test_origin_spacing_s/cfg.step_s)))
test_origins=[]
last=-10**9
for i in tv:
    if not test_origins or i-last>=spacing*STEP:
        test_origins.append(int(i)); last=int(i)
    if len(test_origins)>=cfg.max_test_windows: break
if len(test_origins)<cfg.max_test_windows:
    raise RuntimeError('Not enough non-overlapping test origins.')
test_samples=[(TEST_RUN_ID,i) for i in test_origins]

TRAIN_SAMPLES=train_samples
VAL_SAMPLES=val_samples
TEST_SAMPLES=test_samples
ALL_SAMPLES=TRAIN_SAMPLES+VAL_SAMPLES+TEST_SAMPLES

print('TRAIN_SAMPLES:',TRAIN_SAMPLES)
print('VAL_SAMPLES  :',VAL_SAMPLES)
print('TEST_SAMPLES :',TEST_SAMPLES)
assert all(r!=TEST_RUN_ID for r,_ in TRAIN_SAMPLES)
assert all(r!=TEST_RUN_ID for r,_ in VAL_SAMPLES)

TIMES['origin_selection_s']=time.time()-t0

# %% notebook cell 6

# CELL 6 — TRAIN-ONLY NORMALIZATION FOR THE NEW 37-D p_i / OUTPUT TARGETS / 4-D FEEL-WM ACTION
t0=time.time()

def mean_std(xs,dim):
    x=np.concatenate([np.asarray(a).reshape(-1,dim) for a in xs],0).astype(np.float64)
    m=x.mean(0).astype(np.float32); s=x.std(0).astype(np.float32)
    s[s<1e-6]=1.0
    return m,s

def body_action(d,i0,i1):
    R0=d['R_HB'][i0]; R1=d['R_HB'][i1]
    dp_H=d['p_HB'][i1]-d['p_HB'][i0]
    dp_B=R0.T@dp_H
    Rrel=R0.T@R1
    dyaw=math.atan2(float(Rrel[1,0]),float(Rrel[0,0]))
    dt=float(d['time'][i1]-d['time'][i0])
    return np.array([dp_B[0],dp_B[1],dyaw,dt],np.float32)

train_runs=[RUNS[r] for r in TRAIN_RUN_IDS]
NORM={}
NORM['leg37_mean'],NORM['leg37_std']=mean_std([d['leg37'] for d in train_runs],37)
NORM['state_mean'],NORM['state_std']=mean_std([d['P'][...,:9] for d in train_runs],9)
NORM['force_mean'],NORM['force_std']=mean_std([d['P'][...,9:12] for d in train_runs],3)
NORM['body_mean'],NORM['body_std']=mean_std([d['B'] for d in train_runs],12)
NORM['foot_mean'],NORM['foot_std']=mean_std([d['foot'] for d in train_runs],3)

actions=[]
for rid,i in TRAIN_SAMPLES:
    d=RUNS[rid]
    for h in range(cfg.H):
        actions.append(body_action(d,i+h*STEP,i+(h+1)*STEP))
actions=np.stack(actions)
NORM['action_mean']=actions.mean(0).astype(np.float32)
NORM['action_std']=actions.std(0).astype(np.float32)
NORM['action_std'][NORM['action_std']<1e-6]=1.0

# Normalize touchdown impulse only from genuine train events.
train_imp=[]
for rid in TRAIN_RUN_IDS:
    e=RUNS[rid]['events']
    if len(e): train_imp.append(e[['Jx','Jy','Jz']].to_numpy(np.float32))
if train_imp:
    NORM['impulse_mean'],NORM['impulse_std']=mean_std(train_imp,3)
else:
    NORM['impulse_mean']=np.zeros(3,np.float32); NORM['impulse_std']=np.ones(3,np.float32)

np.savez_compressed(OUT_ROOT/'new_model_normalization.npz',**NORM)
TIMES['normalization_s']=time.time()-t0

# %% notebook cell 7

# CELL 7 — DOWNLOAD ONLY FRONT RGB FOR THE SELECTED RUNS IF NEEDED
# This is data import, not a model change.
t0=time.time()
import tartanair as ta

RAW_ROOT.mkdir(parents=True,exist_ok=True)
ta.init(str(RAW_ROOT))

def raw_run_root(rid):
    r=row_by_id[rid]
    return RAW_ROOT/str(r.environment)/'Data_anymal'/str(r.run)

def front_images(rid):
    root=raw_run_root(rid)/'image_lcam_front'
    if not root.exists(): return []
    files=[p for p in root.iterdir() if p.is_file() and p.suffix.lower() in ('.png','.jpg','.jpeg')]
    def key(p):
        m=re.match(r'^(\d+)',p.name)
        return int(m.group(1)) if m else 10**12
    return sorted(files,key=key)

def ensure_front_rgb(rid):
    files=front_images(rid)
    if files: return files
    if not cfg.download_raw_if_missing:
        raise FileNotFoundError(f'Front RGB missing for {rid}: {raw_run_root(rid)}')
    r=row_by_id[rid]
    print('Downloading ONLY front RGB/meta for',rid,str(r.environment),str(r.run))
    kwargs=dict(
        env=[str(r.environment)],version=['anymal'],traj=[str(r.run)],
        modality=['image','meta'],camera_name=['lcam_front'],
        unzip=True,delete_zip=True,num_workers=cfg.download_workers
    )
    try:
        ta.download_ground(data_source='huggingface',**kwargs)
    except TypeError:
        ta.download_ground(**kwargs)
    files=front_images(rid)
    if not files:
        raise RuntimeError(f'TartanAir download finished but front RGB was not found for {rid}.')
    return files

RGB_FILES={rid:ensure_front_rgb(rid) for rid in SELECTED_RUN_IDS}
for rid,files in RGB_FILES.items():
    print(rid,len(files),'front RGB frames')

TIMES['rgb_import_s']=time.time()-t0
print(f'FRONT-RGB IMPORT/DOWNLOAD = {TIMES["rgb_import_s"]/60:.2f} min')

# %% notebook cell 8

# CELL 8 — FROZEN DINOv2 ViT-B/14; CACHE CONTEXT/FUTURE FRAMES NOW AND SOURCE FRAMES ON DEMAND
t0=time.time()

print('Loading frozen DINOv2 ViT-B/14...')
dino=torch.hub.load('facebookresearch/dinov2','dinov2_vitb14',pretrained=True).to(DEVICE).eval()
for p in dino.parameters():
    p.requires_grad_(False)

IM_MEAN=torch.tensor([0.485,0.456,0.406],device=DEVICE).view(1,3,1,1)
IM_STD=torch.tensor([0.229,0.224,0.225],device=DEVICE).view(1,3,1,1)

def frame_row_to_path(rid,row_idx):
    d=RUNS[rid]
    files=RGB_FILES[rid]
    fids=d['camera_frame_ids']

    if fids is None:
        if len(files)!=len(d['camera_times']):
            raise RuntimeError(
                f'{rid}: no camera-frame-id array and raw image count '
                f'{len(files)} != camera_times {len(d["camera_times"])}.'
            )
        return files[int(row_idx)]

    wanted=int(fids[int(row_idx)])
    byid={}
    for p in files:
        m=re.match(r'^(\d+)',p.name)
        if m:
            byid[int(m.group(1))]=p
    if wanted not in byid:
        raise KeyError(f'{rid}: raw RGB frame id {wanted} is missing.')
    return byid[wanted]

def needed_camera_rows(rid,samples):
    d=RUNS[rid]
    rows=set()
    for sample_rid,i in samples:
        if sample_rid!=rid:
            continue
        vi=int(d['visual_index'][i])
        rows.update(range(vi-3,vi+1))
        for h in range(1,cfg.H+1):
            rows.add(int(d['visual_index'][i+h*STEP]))
    return sorted(rows)

DINO_CACHE={rid:{} for rid in SELECTED_RUN_IDS}

def ensure_dino_rows(rid,rows):
    rows=sorted(set(int(x) for x in rows if int(x)>=0))
    missing=[r for r in rows if r not in DINO_CACHE[rid]]
    if not missing:
        return

    for a in range(0,len(missing),cfg.dino_batch):
        rr=missing[a:a+cfg.dino_batch]
        ims=[]
        for row in rr:
            p=frame_row_to_path(rid,row)
            bgr=cv2.imread(str(p),cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f'Cannot read {p}')
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            # The source-pixel -> DINO-patch mapping below assumes this exact direct resize.
            rgb=cv2.resize(
                rgb,(cfg.dino_image_size,cfg.dino_image_size),
                interpolation=cv2.INTER_CUBIC
            )
            x=torch.from_numpy(rgb).permute(2,0,1).float()/255.
            ims.append(x)

        x=torch.stack(ims).to(DEVICE,non_blocking=True)
        x=(x-IM_MEAN)/IM_STD

        with torch.inference_mode(), torch.autocast('cuda',dtype=torch.float16):
            feat=dino.forward_features(x)['x_norm_patchtokens']

        if feat.shape[1:]!=(cfg.dino_tokens,cfg.dino_dim):
            raise RuntimeError(f'Expected DINO [B,256,768], got {tuple(feat.shape)}')

        feat=feat.detach().cpu().half().numpy()
        for j,row in enumerate(rr):
            DINO_CACHE[rid][int(row)]=feat[j]

for rid in SELECTED_RUN_IDS:
    rows=needed_camera_rows(rid,[x for x in ALL_SAMPLES if x[0]==rid])
    ensure_dino_rows(rid,rows)
    print(rid,'initial DINO frames:',len(DINO_CACHE[rid]))

TIMES['dino_precompute_initial_s']=time.time()-t0
print(f'INITIAL DINO PRECOMPUTE = {TIMES["dino_precompute_initial_s"]/60:.2f} min')

# %% notebook cell 9

# CELL 9 — EXACT CAUSAL V10 TERRAIN QUERY: 32×9 + 18 STATS + SOURCE-TIME PROVENANCE
class V10ContactMap:
    def __init__(self,geometry_root,crop_start,voxel=0.015):
        root=Path(geometry_root)
        meta=load_json(root/'COMPLETE.json')

        self.voxel=float(voxel)
        self.crop_start=float(crop_start)
        self.R_H_W=np.asarray(meta['R_H_W'],np.float64)
        self.p_W0=np.asarray(meta['p_W0'],np.float64)

        fields=('key','xyz','normal','rgb','xyz_var','count','camera_mask','time')
        pieces={k:[] for k in fields}

        for x in meta['chunks']:
            p=root/'chunks'/x['name']
            with np.load(p,allow_pickle=False) as z:
                for k in fields:
                    pieces[k].append(np.asarray(z[k]))

        x={k:np.concatenate(v,0) for k,v in pieces.items()}
        order=np.lexsort((x['time'],x['key'][:,2],x['key'][:,1],x['key'][:,0]))
        for k in x:
            setattr(self,k,x[k][order])

        change=np.r_[True,np.any(np.diff(self.key,axis=0),axis=1)]
        self.starts=np.where(change)[0]
        self.ends=np.r_[self.starts[1:],len(self.key)]
        self.unique_key=self.key[self.starts]

        self.columns=defaultdict(list)
        for ui,k in enumerate(self.unique_key):
            self.columns[(int(k[0]),int(k[1]))].append(ui)

    def _latest_candidates(self,q,t_cropped,radius):
        q=np.asarray(q,float)
        t_original=float(t_cropped)+self.crop_start
        kr=int(math.ceil(radius/self.voxel))
        c=np.floor(q/self.voxel).astype(int)

        ids=[]
        for dx in range(-kr,kr+1):
            for dy in range(-kr,kr+1):
                for ui in self.columns.get((c[0]+dx,c[1]+dy),[]):
                    a,b=self.starts[ui],self.ends[ui]
                    j=a+np.searchsorted(self.time[a:b],t_original,side='right')-1
                    if j>=a:
                        ids.append(j)

        if not ids:
            return np.empty(0,int)

        ids=np.asarray(ids,int)
        dxy=np.linalg.norm(self.xyz[ids,:2]-q[None,:2],axis=1)
        return ids[dxy<=radius]

    def query(self,q,t_cropped,radius=0.07):
        q=np.asarray(q,np.float32)
        ids=self._latest_candidates(q,t_cropped,radius)

        pts=np.zeros((32,9),np.float32)
        stats=np.zeros(18,np.float32)
        abs_H=np.zeros((32,3),np.float32)
        src_time_original=np.full(32,np.nan,np.float64)

        if len(ids)==0:
            return pts,stats,0,abs_H,src_time_original

        dz=np.abs(self.xyz[ids,2]-q[2])
        near=dz<=max(.20,3*self.voxel)
        if near.any():
            ids=ids[near]
        if len(ids)==0:
            return pts,stats,0,abs_H,src_time_original

        rel=self.xyz[ids]-q[None,:]
        dist3=np.linalg.norm(rel,axis=1)
        order=np.argsort(dist3)
        use=ids[order[:32]]

        rel_use=self.xyz[use]-q[None,:]
        desc=np.concatenate(
            [rel_use,self.normal[use],self.rgb[use]],axis=-1
        ).astype(np.float32)

        nvalid=len(desc)
        pts[:nvalid]=desc
        abs_H[:nvalid]=self.xyz[use].astype(np.float32)
        src_time_original[:nvalid]=self.time[use].astype(np.float64)

        rel_all=self.xyz[ids]-q[None,:]
        z=rel_all[:,2]
        nmean=self.normal[ids].mean(0)
        nmean=nmean/max(np.linalg.norm(nmean),1e-8)

        query_original=float(t_cropped)+self.crop_start
        ages=np.maximum(query_original-self.time[ids],0.0)

        x=rel_all[:,0]
        y=rel_all[:,1]
        curvature=0.0
        ridge=0.0
        residual_std=float(np.std(z))

        if len(ids)>=6:
            A=np.c_[x*x,y*y,x*y,x,y,np.ones_like(x)]
            coef,*_=np.linalg.lstsq(A,z,rcond=None)
            pred=A@coef
            residual_std=float(np.std(z-pred))
            H2=np.array(
                [[2*coef[0],coef[2]],[coef[2],2*coef[1]]],
                float
            )
            eig=np.linalg.eigvalsh(H2)
            curvature=float(np.mean(np.abs(eig)))
            ridge=float(np.max(np.abs(eig)))

        slope=float(np.arccos(np.clip(abs(float(nmean[2])),0.0,1.0)))

        stats=np.array([
            1.0,
            min(len(ids)/128.0,1.0),
            float(dist3.min()),
            float(ages.mean()),
            float(z.mean()),
            float(z.std()),
            float(z.max()-z.min()),
            float(nmean[0]),float(nmean[1]),float(nmean[2]),
            slope,
            residual_std,
            curvature,
            ridge,
            float(np.quantile(z,.95)-np.quantile(z,.05)),
            float(x.std()),
            float(y.std()),
            float(np.linalg.norm(rel_all[:,:2],axis=1).mean()),
        ],np.float32)

        return pts,stats,nvalid,abs_H,src_time_original

print('V10ContactMap ready with source-time provenance.')

# %% notebook cell 10

# CELL 10 — FROZEN V10 PREDICTED FEET → CAUSAL MAP → SOURCE-FRAME DINO CONTACT TOKENS
t0=time.time()

intr=cal_contract['camera_intrinsics']['lcam_front']
W=int(intr['width'])
HIMG=int(intr['height'])
fx=float(intr['fx']); fy=float(intr['fy'])
cx=float(intr['cx']); cy=float(intr['cy'])

def pnorm(x,key):
    return (x-PN[key+'_mean'])/PN[key+'_std']

def frozen_v10_foot_predictions(rid,i):
    d=RUNS[rid]

    leg=pnorm(
        d['leg25'][i-HIST+1:i+1].transpose(1,0,2),
        'leg'
    ).astype(np.float32)

    body=pnorm(
        d['body18'][i-HIST+1:i+1],
        'body'
    ).astype(np.float32)

    # Frozen deployable V10 proposal uses only action available at origin.
    now=np.r_[d['U'][i],d['L'][i].reshape(-1)].astype(np.float32)
    plan=np.repeat(now[None,:],cfg.H,axis=0)
    plan=pnorm(plan,'action').astype(np.float32)

    foot=d['foot'][i].astype(np.float32)

    with torch.inference_mode():
        q=v10proposal.propose(
            torch.from_numpy(leg[None]).to(DEVICE),
            torch.from_numpy(body[None]).to(DEVICE),
            torch.from_numpy(plan[None]).to(DEVICE),
            torch.from_numpy(foot[None]).to(DEVICE),
        )[0]

    return q.cpu().numpy().astype(np.float32)

def load_front_pose_array(rid):
    p=raw_run_root(rid)/'pose_lcam_front.txt'
    if not p.is_file():
        raise FileNotFoundError(
            f'{rid}: {p} missing. Source-pixel DINO association requires the real front-camera pose file.'
        )
    pose=np.loadtxt(p,dtype=np.float64)
    if pose.ndim!=2 or pose.shape[1]<7:
        raise RuntimeError(f'{rid}: malformed front-camera pose file {pose.shape}')
    return pose

FRONT_POSE={rid:load_front_pose_array(rid) for rid in SELECTED_RUN_IDS}

def source_row_from_original_time(d,t_original):
    # Stage-C camera_times may be stored on the cropped or original clock.
    ct=np.asarray(d['camera_times'],np.float64)
    crop=float(d['crop_start'])

    cand=[
        (np.abs(ct-t_original),ct),
        (np.abs((ct+crop)-t_original),ct+crop),
    ]
    errors,aligned=min(cand,key=lambda x:float(np.min(x[0])))
    row=int(np.argmin(errors))
    return row,float(errors[row])

def raw_frame_id_for_row(d,row):
    fids=d['camera_frame_ids']
    return int(row) if fids is None else int(fids[int(row)])

def source_patch_id(rid,d,mem,xyz_H,t_original):
    row,terr=source_row_from_original_time(d,float(t_original))
    if terr>0.10:
        return None,None

    raw_id=raw_frame_id_for_row(d,row)
    pose_arr=FRONT_POSE[rid]
    if raw_id<0 or raw_id>=len(pose_arr):
        return None,None

    pose=pose_arr[raw_id]
    p_WC=pose[:3]
    R_WC=Rotation.from_quat(pose[3:7]).as_matrix()

    # Geometry construction used:
    # xyz_H = (xyz_W - p_W0) @ R_H_W.T
    # therefore xyz_W = p_W0 + xyz_H @ R_H_W
    xyz_W=mem.p_W0 + np.asarray(xyz_H,np.float64) @ mem.R_H_W

    # Camera pose maps C->W, so row-vector inverse is:
    # p_C = (p_W - p_WC) @ R_WC
    p_C=(xyz_W-p_WC) @ R_WC

    # TartanGround camera convention used by the prepared geometry:
    # x forward, y right, z down.
    X=float(p_C[0])
    if X<=0.05 or X>cfg.max_contact_range_m:
        return None,None

    u=cx+fx*float(p_C[1])/X
    v=cy+fy*float(p_C[2])/X
    if not (0<=u<W and 0<=v<HIMG):
        return None,None

    # Cell 8 resizes the full RGB frame directly to 224x224.
    col=int(np.clip(math.floor((u/W)*16),0,15))
    row_patch=int(np.clip(math.floor((v/HIMG)*16),0,15))
    patch=row_patch*16+col
    return row,patch

# Phase A: frozen proposal + causal geometry query.
# Keep source-point provenance, but do not compute DINO contact tokens until
# every source camera row required by those surfels has been identified.
QUERY_PRE={}
SOURCE_ROWS=defaultdict(set)

for rid in SELECTED_RUN_IDS:
    samples=[x for x in ALL_SAMPLES if x[0]==rid]
    if not samples:
        continue

    d=RUNS[rid]
    print('Loading causal geometry memory:',rid)
    mem=V10ContactMap(
        prepared_paths(rid)['geometry'],
        d['crop_start'],
        cfg.patch_voxel_m
    )

    for _,i in samples:
        predfeet=frozen_v10_foot_predictions(rid,i)

        gp=np.zeros((cfg.H,4,32,9),np.float32)
        gs=np.zeros((cfg.H,4,18),np.float32)
        nvalid=np.zeros((cfg.H,4),np.int16)
        absH=np.zeros((cfg.H,4,32,3),np.float32)
        srct=np.full((cfg.H,4,32),np.nan,np.float64)

        origin_t=float(d['time'][i])

        for h in range(cfg.H):
            for leg in range(4):
                q=predfeet[h,leg]
                pts,st,nv,xyzH,srcT=mem.query(
                    q,origin_t,cfg.patch_radius_m
                )
                gp[h,leg]=pts
                gs[h,leg]=st
                nvalid[h,leg]=nv
                absH[h,leg]=xyzH
                srct[h,leg]=srcT

                for j in range(nv):
                    source_row,patch=source_patch_id(
                        rid,d,mem,xyzH[j],srcT[j]
                    )
                    if source_row is not None:
                        SOURCE_ROWS[rid].add(int(source_row))

        QUERY_PRE[(rid,i)]=dict(
            predfeet=predfeet,
            geom_points=gp,
            geom_stats=gs,
            nvalid=nvalid,
            absH=absH,
            srcT=srct,
        )

    # Save transforms needed by the source-pixel projection.
    QUERY_PRE[(rid,'_mem_meta')]=dict(
        R_H_W=mem.R_H_W.copy(),
        p_W0=mem.p_W0.copy(),
    )
    del mem
    gc.collect()

# Phase B: add ONLY the source frames actually used by the queried surfels.
for rid,rows in SOURCE_ROWS.items():
    before=len(DINO_CACHE[rid])
    ensure_dino_rows(rid,rows)
    print(
        rid,
        'source-contact DINO rows:',
        len(rows),
        'newly added:',
        len(DINO_CACHE[rid])-before
    )

def contact_dino_from_source_surfels(rid,d,mem_meta,q_H,xyzH,srcT,nvalid):
    if nvalid<=0:
        return np.zeros(cfg.dino_dim,np.float32),0

    # Small stand-in object containing the exact H<->W transform.
    class Meta: pass
    mem=Meta()
    mem.R_H_W=mem_meta['R_H_W']
    mem.p_W0=mem_meta['p_W0']

    tok=[]
    dist=[]
    for j in range(int(nvalid)):
        source_row,patch=source_patch_id(
            rid,d,mem,xyzH[j],srcT[j]
        )
        if source_row is None:
            continue
        if source_row not in DINO_CACHE[rid]:
            continue
        tok.append(
            np.asarray(DINO_CACHE[rid][source_row][patch],np.float32)
        )
        dist.append(
            float(np.linalg.norm(np.asarray(xyzH[j,:2])-np.asarray(q_H[:2])))
        )

    if not tok:
        return np.zeros(cfg.dino_dim,np.float32),0

    dist=np.asarray(dist,np.float32)
    sigma=max(cfg.patch_radius_m/2,1e-6)
    w=np.exp(-0.5*(dist/sigma)**2)
    w=w/max(float(w.sum()),1e-12)

    tok=np.stack(tok)
    return (tok*w[:,None]).sum(0).astype(np.float32),len(tok)

# Phase C: exact per-leg source-frame DINO contact token.
PRE={}
for rid in SELECTED_RUN_IDS:
    d=RUNS[rid]
    mem_meta=QUERY_PRE[(rid,'_mem_meta')]

    for sample_rid,i in [x for x in ALL_SAMPLES if x[0]==rid]:
        qd=QUERY_PRE[(rid,i)]
        cv=np.zeros((cfg.H,4,cfg.dino_dim),np.float32)
        visible=np.zeros((cfg.H,4),np.int16)

        for h in range(cfg.H):
            for leg in range(4):
                cv[h,leg],visible[h,leg]=contact_dino_from_source_surfels(
                    rid,d,mem_meta,
                    qd['predfeet'][h,leg],
                    qd['absH'][h,leg],
                    qd['srcT'][h,leg],
                    qd['nvalid'][h,leg],
                )

        PRE[(rid,i)]=dict(
            predfeet=qd['predfeet'],
            geom_points=qd['geom_points'].astype(np.float16),
            geom_stats=qd['geom_stats'].astype(np.float32),
            contact_visual=cv.astype(np.float16),
            visible_points=visible,
        )

# DINO is frozen and no longer needed after all global/contact tokens are cached.
del dino
torch.cuda.empty_cache()
gc.collect()

TIMES['v10_query_precompute_s']=time.time()-t0
TIMES['dino_precompute_s']=(
    TIMES.get('dino_precompute_initial_s',0.0)
    + TIMES['v10_query_precompute_s']
)

print(
    f'FROZEN V10 FOOT + CAUSAL MAP + SOURCE-DINO PRECOMPUTE = '
    f'{TIMES["v10_query_precompute_s"]/60:.2f} min'
)

# %% notebook cell 11

# CELL 11 — EVENT LABELS + DATASET FOR H=10 RECURSIVE WORLD-MODEL TRAINING
def norm_new(x,key):
    return ((x-NORM[key+'_mean'])/NORM[key+'_std']).astype(np.float32)

def phase_at(contact,j):
    cur=(contact[j]>0.5)
    prev=(contact[max(j-1,0)]>0.5)
    phase=np.where(cur,2,0).astype(np.int64)
    phase[(~prev)&cur]=1
    phase[prev&(~cur)]=3
    return phase

def event_target(d,origin_t,h):
    lo=origin_t+h*cfg.step_s
    hi=origin_t+(h+1)*cfg.step_s
    center=(lo+hi)/2
    hazard=np.zeros(4,np.float32)
    offset=np.zeros(4,np.float32)
    impulse=np.zeros((4,3),np.float32)
    mask=np.zeros(4,np.float32)

    e=d['events']
    q=e[(e.t_touchdown_cropped>lo-1e-12)&(e.t_touchdown_cropped<=hi+1e-12)]
    for leg in range(4):
        z=q[q.leg_index.astype(int)==leg]
        if len(z)==0: continue
        row=z.iloc[0]
        hazard[leg]=1.; mask[leg]=1.
        offset[leg]=float(row.t_touchdown_cropped-center)
        impulse[leg]=row[['Jx','Jy','Jz']].to_numpy(np.float32)
    return hazard,offset,norm_new(impulse,'impulse'),mask

class FrozenArchDataset(Dataset):
    def __init__(self,samples): self.samples=list(samples)
    def __len__(self): return len(self.samples)

    def __getitem__(self,k):
        rid,i=self.samples[k]; d=RUNS[rid]; vi=int(d['visual_index'][i])

        zctx=np.stack([DINO_CACHE[rid][r] for r in range(vi-3,vi+1)]).astype(np.float16)
        zfuture=[]
        histfuture=[]
        actions=[]
        state=[]; force=[]; phase=[]; foot=[]; body=[]
        td=[]; tdoff=[]; impulse=[]; eventmask=[]
        failure=[]; failuremask=[]

        for h in range(cfg.H):
            j0=i+h*STEP; j1=i+(h+1)*STEP
            vj=int(d['visual_index'][j1])
            zfuture.append(DINO_CACHE[rid][vj].astype(np.float16))
            histfuture.append(norm_new(d['leg37'][j1-HIST+1:j1+1].transpose(1,0,2),'leg37'))
            actions.append(norm_new(body_action(d,j0,j1),'action'))
            state.append(norm_new(d['P'][j1,:,:9],'state'))
            force.append(norm_new(d['P'][j1,:,9:12],'force'))
            phase.append(phase_at(d['contact'],j1))
            foot.append(norm_new(d['foot'][j1],'foot'))
            body.append(norm_new(d['B'][j1],'body'))

            a,b,c,m=event_target(d,float(d['time'][i]),h)
            td.append(a); tdoff.append(b); impulse.append(c); eventmask.append(m)

            if d['failure'] is None:
                failure.append(0.0); failuremask.append(0.0)
            else:
                failure.append(float(d['failure'][j1])); failuremask.append(1.0)

        pc=PRE[(rid,i)]
        return dict(
            rid=rid,origin=np.int64(i),
            zctx=zctx,
            leg_hist=norm_new(d['leg37'][i-HIST+1:i+1].transpose(1,0,2),'leg37'),
            actions=np.stack(actions),
            future_z=np.stack(zfuture),
            future_hist=np.stack(histfuture),
            geom_points=pc['geom_points'],
            geom_stats=pc['geom_stats'],
            contact_visual=pc['contact_visual'],
            state=np.stack(state),force=np.stack(force),phase=np.stack(phase),
            foot=np.stack(foot),body=np.stack(body),
            touchdown=np.stack(td),event_offset=np.stack(tdoff),
            impulse=np.stack(impulse),event_mask=np.stack(eventmask),
            failure=np.asarray(failure,np.float32),
            failure_mask=np.asarray(failuremask,np.float32),
        )

train_ds=FrozenArchDataset(TRAIN_SAMPLES)
val_ds=FrozenArchDataset(VAL_SAMPLES)
test_ds=FrozenArchDataset(TEST_SAMPLES)

train_loader=DataLoader(train_ds,batch_size=1,shuffle=True,num_workers=0,pin_memory=True)
val_loader=DataLoader(val_ds,batch_size=1,shuffle=False,num_workers=0,pin_memory=True)
test_loader=DataLoader(test_ds,batch_size=1,shuffle=False,num_workers=0,pin_memory=True)

b=next(iter(train_loader))
for k,v in b.items():
    if torch.is_tensor(v): print(k,tuple(v.shape),v.dtype)

# %% notebook cell 12

# CELL 12 — THE FROZEN 1032×768 ARCHITECTURE (NO REDUCED-TOKEN SUBSTITUTE)
class LegCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv1d(37,64,7,2,3),nn.GroupNorm(8,64),nn.GELU(),
            nn.Conv1d(64,128,5,2,2),nn.GroupNorm(8,128),nn.GELU(),
            nn.Conv1d(128,128,3,2,1),nn.GroupNorm(8,128),nn.GELU(),
            nn.Conv1d(128,128,3,1,1),nn.GroupNorm(8,128),nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.norm=nn.LayerNorm(128)
    def forward(self,x):
        return self.norm(self.net(x.transpose(1,2)).squeeze(-1))

class GeometryEncoder(nn.Module):
    # Exact frozen V10 geometry shape:
    # 32×9 -> point MLP -> 96 max pool; 18 stats -> 64; 160 -> 128.
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
    def forward(self,points,stats):
        p=self.point(points).max(dim=-2).values
        s=self.stats(stats)
        return self.fuse(torch.cat([p,s],-1))

class SharedFiLM(nn.Module):
    # Shared 128->256 conditioning trunk; separate visual and geometry gamma/beta projections.
    def __init__(self):
        super().__init__()
        self.trunk=nn.Sequential(nn.Linear(128,256),nn.GELU(),nn.LayerNorm(256))
        self.vis=nn.Linear(256,2*768)
        self.geo=nn.Linear(256,2*768)
        # identity initialization: gamma = 1 + delta_gamma; beta = 0
        nn.init.zeros_(self.vis.weight); nn.init.zeros_(self.vis.bias)
        nn.init.zeros_(self.geo.weight); nn.init.zeros_(self.geo.bias)

    def params(self,c,kind):
        r=self.trunk(c)
        x=self.vis(r) if kind=='visual' else self.geo(r)
        dg,beta=x.chunk(2,-1)
        return 1.0+dg,beta

    def apply(self,x,c,kind):
        gamma,beta=self.params(c,kind)
        while gamma.ndim<x.ndim:
            gamma=gamma.unsqueeze(-2); beta=beta.unsqueeze(-2)
        return gamma*x+beta

class TemporalBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1=nn.LayerNorm(768)
        self.attn=nn.MultiheadAttention(768,8,dropout=.1,batch_first=True)
        self.ln2=nn.LayerNorm(768)
        self.ff=nn.Sequential(
            nn.Linear(768,3072),nn.GELU(),nn.Dropout(.1),
            nn.Linear(3072,768),nn.Dropout(.1)
        )
    def forward(self,x,mask):
        q=self.ln1(x)
        y=self.attn(q,q,q,attn_mask=mask,need_weights=False)[0]
        x=x+y
        x=x+self.ff(self.ln2(x))
        return x

class FrozenV10FeelWM(nn.Module):
    def __init__(self):
        super().__init__()
        self.leg_cnn=LegCNN()
        self.leg_id=nn.Parameter(torch.randn(4,128)*.02)

        self.global_p=nn.Sequential(
            nn.Linear(512,256),nn.GELU(),nn.LayerNorm(256),
            nn.Linear(256,128),nn.LayerNorm(128)
        )
        self.action_enc=nn.Sequential(
            nn.Linear(4,128),nn.GELU(),nn.LayerNorm(128)
        )

        self.geom_enc=GeometryEncoder()
        self.geom_to_768=nn.Linear(128,768)
        self.film=SharedFiLM()

        self.time_embed=nn.Parameter(torch.randn(4,768)*.02)
        self.mod_embed=nn.Parameter(torch.randn(3,768)*.02) # visual, contact visual, geometry
        self.leg_embed=nn.Parameter(torch.randn(4,768)*.02)

        self.blocks=nn.ModuleList([TemporalBlock() for _ in range(8)])
        self.norm=nn.LayerNorm(768)

        # Heads from the frozen design.
        self.visual_head=nn.Linear(768,768)
        self.leg_fuse=nn.Sequential(
            nn.Linear(2304,1024),nn.GELU(),nn.LayerNorm(1024),
            nn.Linear(1024,768),nn.GELU(),nn.LayerNorm(768)
        )
        self.p_head=nn.Linear(768,128)
        self.state_head=nn.Linear(768,9)
        self.force_head=nn.Linear(768,6)     # force mean 3 + log variance 3
        self.phase_head=nn.Linear(768,4)
        self.foot_head=nn.Linear(768,3)
        self.event_head=nn.Linear(768,5)     # hazard, time offset, Jx,Jy,Jz
        self.body_head=nn.Linear(768,12)
        self.failure_head=nn.Linear(768,1)

        # 4×256 visual tokens + 4 contact + 4 geometry = 1032.
        time_id=torch.tensor(
            [0]*256+[1]*256+[2]*256+[3]*256+[3]*4+[3]*4,
            dtype=torch.long
        )
        assert len(time_id)==1032
        # True means "do not attend": block future time groups, allow full attention within the same time.
        causal=time_id[None,:] > time_id[:,None]
        self.register_buffer('causal_mask',causal,persistent=False)

    def encode_p(self,hist):
        # hist [B,4,100,37]
        return torch.stack([self.leg_cnn(hist[:,i]) for i in range(4)],1)

    def tx(self,x):
        for block in self.blocks:
            if self.training:
                x=checkpoint(block,x,self.causal_mask,use_reentrant=False)
            else:
                x=block(x,self.causal_mask)
        return self.norm(x)

    def forward_step(self,zctx,p_leg,action,contact_visual,geom_points,geom_stats):
        B=zctx.shape[0]
        assert zctx.shape[1:]==(4,256,768),zctx.shape
        assert contact_visual.shape[1:]==(4,768)
        assert geom_points.shape[1:]==(4,32,9)
        assert geom_stats.shape[1:]==(4,18)

        pG=self.global_p(p_leg.reshape(B,512))
        ea=self.action_enc(action)
        cG=ea+pG
        ci=ea[:,None,:]+p_leg+self.leg_id[None,:,:]

        # Only current frame is FiLM-conditioned globally.
        z=[]
        for t in range(4):
            zz=zctx[:,t]+self.time_embed[t][None,None,:]+self.mod_embed[0][None,None,:]
            if t==3:
                zz=self.film.apply(zz,cG,'visual')
            z.append(zz)
        z=torch.cat(z,1) # [B,1024,768]

        # Per-leg contact visual FiLM.
        v=self.film.apply(contact_visual,ci,'visual')
        v=v+self.mod_embed[1][None,None,:]+self.leg_embed[None,:,:]

        # Per-leg V10 geometry MLP then FiLM.
        g128=self.geom_enc(geom_points,geom_stats)
        g768=self.geom_to_768(g128)
        g=self.film.apply(g768,ci,'geometry')
        g=g+self.mod_embed[2][None,None,:]+self.leg_embed[None,:,:]

        X=torch.cat([z,v,g],1)
        assert X.shape[1:]==(1032,768),X.shape
        H=self.tx(X)

        current=H[:,3*256:4*256]
        HG=current.mean(1)
        Hv=H[:,1024:1028]
        Hg=H[:,1028:1032]
        hi=self.leg_fuse(torch.cat([HG[:,None,:].expand(-1,4,-1),Hv,Hg],-1))

        fv=self.force_head(hi)
        ev=self.event_head(hi)
        return dict(
            visual=self.visual_head(current),        # [B,256,768]
            p=self.p_head(hi),                       # [B,4,128]
            state=self.state_head(hi),               # q,dq,tau
            force_mu=fv[...,:3],
            force_logvar=fv[...,3:].clamp(-8,8),
            phase_logits=self.phase_head(hi),
            foot=self.foot_head(hi),
            touchdown_logit=ev[...,0],
            event_offset_s=.05*torch.tanh(ev[...,1]),
            impulse=ev[...,2:5],
            body=self.body_head(HG),
            failure_logit=self.failure_head(HG).squeeze(-1),
        )

model=FrozenV10FeelWM().to(DEVICE)
PARAMS=sum(p.numel() for p in model.parameters())
print(f'New model parameters: {PARAMS/1e6:.2f} M')

# %% notebook cell 13

# CELL 13 — HARD ARCHITECTURE SELF-CHECK: FAIL IF ANY FROZEN SHAPE WAS REDUCED
assert HIST==100
assert cfg.H==10 and abs(cfg.step_s-.1)<1e-12
assert cfg.dino_tokens==256 and cfg.dino_dim==768
assert cfg.d_model==768 and cfg.layers==8 and cfg.heads==8 and cfg.mlp_ratio==4
assert model.causal_mask.shape==(1032,1032)
assert len(model.blocks)==8
assert model.blocks[0].attn.num_heads==8
assert model.blocks[0].ff[0].in_features==768
assert model.blocks[0].ff[0].out_features==3072

with torch.inference_mode():
    x=torch.zeros(1,4,100,37,device=DEVICE)
    p=model.encode_p(x)
assert p.shape==(1,4,128)

print('ARCHITECTURE LOCK PASSED')
print('  per-leg history : 100×37 -> 128')
print('  DINO/frame      : 256×768')
print('  geometry/leg    : 32×9 + 18 -> 128 -> 768')
print('  Transformer     : 1032×768, 8 layers, 8 heads, FFN 3072')
print('  rollout         : H=10 at 0.1 s = 1.0 s')
print('  frozen V10 foot proposal loaded:',PROPOSAL_CKPT)

# %% notebook cell 14

# CELL 14 — LOSSES FOR ONLY THE LOCKED HEADS; NO INVENTED FAILURE LABEL
def dev(batch):
    return {k:(v.to(DEVICE,non_blocking=True) if torch.is_tensor(v) else v) for k,v in batch.items()}

def unnorm_t(x,key):
    m=torch.as_tensor(NORM[key+'_mean'],device=x.device,dtype=x.dtype)
    s=torch.as_tensor(NORM[key+'_std'],device=x.device,dtype=x.dtype)
    return x*s+m

def visual_loss(pred,target):
    mse=F.mse_loss(pred,target)
    cos=1-F.cosine_similarity(pred,target,dim=-1).mean()
    return mse+cos

def force_nll(mu,logvar,y):
    return .5*(logvar+(y-mu).pow(2)*torch.exp(-logvar)).mean()

def rollout_loss(batch,autoregressive=True):
    b=dev(batch)
    hist=b['leg_hist'].float()
    z=b['zctx'].float()
    pcur=model.encode_p(hist)
    total=0.0
    parts=defaultdict(float)

    for h in range(cfg.H):
        out=model.forward_step(
            z,pcur,b['actions'][:,h].float(),
            b['contact_visual'][:,h].float(),
            b['geom_points'][:,h].float(),
            b['geom_stats'][:,h].float(),
        )
        with torch.no_grad():
            pstar=model.encode_p(b['future_hist'][:,h].float())

        losses={}
        losses['visual']=visual_loss(out['visual'],b['future_z'][:,h].float())
        losses['p']=F.mse_loss(out['p'],pstar)
        losses['state']=F.smooth_l1_loss(out['state'],b['state'][:,h].float(),beta=.5)
        losses['force']=force_nll(out['force_mu'],out['force_logvar'],b['force'][:,h].float())
        losses['phase']=F.cross_entropy(
            out['phase_logits'].reshape(-1,4),
            b['phase'][:,h].long().reshape(-1)
        )
        losses['foot']=F.smooth_l1_loss(out['foot'],b['foot'][:,h].float(),beta=.5)
        losses['touchdown']=F.binary_cross_entropy_with_logits(
            out['touchdown_logit'],b['touchdown'][:,h].float()
        )
        losses['body']=F.smooth_l1_loss(out['body'],b['body'][:,h].float(),beta=.5)

        m=b['event_mask'][:,h].float()
        if m.sum()>0:
            losses['event_time']=(
                F.smooth_l1_loss(out['event_offset_s'],b['event_offset'][:,h].float(),reduction='none',beta=.01)*m
            ).sum()/m.sum()
            losses['impulse']=(
                F.smooth_l1_loss(out['impulse'],b['impulse'][:,h].float(),reduction='none',beta=.5)*m[...,None]
            ).sum()/max(float(m.sum().item())*3,1.0)
        else:
            losses['event_time']=out['event_offset_s'].sum()*0
            losses['impulse']=out['impulse'].sum()*0

        fm=b['failure_mask'][:,h].float()
        if fm.sum()>0:
            fl=F.binary_cross_entropy_with_logits(
                out['failure_logit'],b['failure'][:,h].float(),reduction='none'
            )
            losses['failure']=(fl*fm).sum()/fm.sum()
        else:
            # No fake label. Head exists, but no loss if dataset has no legitimate failure annotation.
            losses['failure']=out['failure_logit'].sum()*0

        step=sum(losses.values())/cfg.H
        total=total+step
        for k,v in losses.items(): parts[k]+=float(v.detach())/cfg.H

        # Exact recursive state transition used at deployment.
        if h<cfg.H-1:
            if autoregressive:
                z=torch.cat([z[:,1:],out['visual'][:,None]],1)
                pcur=out['p']
            else:
                z=torch.cat([z[:,1:],b['future_z'][:,h,None].float()],1)
                pcur=pstar

    return total,dict(parts)

# %% notebook cell 15

# CELL 15 — FAST-DATA TRAINING OF THE FULL FROZEN ARCHITECTURE
# Architecture is full-sized; only the number of rollout origins is tiny.
t0=time.time()

opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
USE_AMP=True
amp_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
scaler=torch.amp.GradScaler('cuda',enabled=(amp_dtype==torch.float16))

best_path=OUT_ROOT/'best_frozen_arch_fast.pt'
best=np.inf
history=[]
train_start=time.time()
stop=False

for epoch in range(cfg.train_epochs):
    model.train()
    for bi,batch in enumerate(train_loader):
        if (time.time()-train_start)/60 >= cfg.max_train_minutes:
            stop=True
            print('Hard training-time budget reached.')
            break

        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=amp_dtype):
            loss,parts=rollout_loss(batch,autoregressive=True)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip)
            opt.step()

        rec={'epoch':epoch,'batch':bi,'loss':float(loss.detach()),
             'elapsed_min':(time.time()-train_start)/60,**parts}
        history.append(rec)
        print(rec)

    # One validation rollout. Full autoregressive.
    model.eval()
    vals=[]
    with torch.inference_mode():
        for vb in val_loader:
            with torch.autocast('cuda',dtype=amp_dtype):
                vl,_=rollout_loss(vb,autoregressive=True)
            vals.append(float(vl))
    score=float(np.mean(vals)) if vals else float(history[-1]['loss'])
    print('validation:',score)
    if score<best:
        best=score
        torch.save(dict(
            model=model.state_dict(),cfg=asdict(cfg),best_val=best,
            train_runs=TRAIN_RUN_IDS,val_run=VAL_RUN_ID,test_run=TEST_RUN_ID,
            proposal_checkpoint=str(PROPOSAL_CKPT),
        ),best_path)
    if stop: break

TIMES['training_s']=time.time()-t0
pd.DataFrame(history).to_csv(OUT_ROOT/'training_history.csv',index=False)

if not best_path.is_file():
    raise RuntimeError('No new-model checkpoint was saved.')
ck=torch.load(best_path,map_location=DEVICE,weights_only=False)
model.load_state_dict(ck['model'])
print(f'TRAINING TIME = {TIMES["training_s"]/60:.2f} min')
print('best validation loss:',ck['best_val'])

# %% notebook cell 16

# CELL 16 — FULL AUTOREGRESSIVE HELD-OUT TEST ON C_rough_uneven_P2006
t0=time.time()

def macro_phase_acc(logits,y):
    p=logits.argmax(-1)
    return float((p==y).float().mean())

@torch.inference_mode()
def evaluate(loader):
    model.eval()
    rows=[]
    ride_records=[]

    for batch in loader:
        b=dev(batch)
        rid=batch['rid'][0]
        origin=int(batch['origin'][0])
        d=RUNS[rid]

        z=b['zctx'].float()
        pcur=model.encode_p(b['leg_hist'].float())

        # Ride reconstruction starts from the true forecast-origin orientation only.
        Rpred=torch.from_numpy(d['R_HB'][origin]).to(DEVICE).float()[None]
        prev_yaw=torch.atan2(Rpred[:,1,0],Rpred[:,0,0])

        for h in range(cfg.H):
            with torch.autocast('cuda',dtype=amp_dtype):
                o=model.forward_step(
                    z,pcur,b['actions'][:,h].float(),
                    b['contact_visual'][:,h].float(),
                    b['geom_points'][:,h].float(),
                    b['geom_stats'][:,h].float(),
                )

            true_force=unnorm_t(b['force'][:,h].float(),'force')
            pred_force=unnorm_t(o['force_mu'],'force')
            fe=(pred_force-true_force).abs()

            true_state=unnorm_t(b['state'][:,h].float(),'state')
            pred_state=unnorm_t(o['state'],'state')
            true_foot=unnorm_t(b['foot'][:,h].float(),'foot')
            pred_foot=unnorm_t(o['foot'],'foot')
            true_body=unnorm_t(b['body'][:,h].float(),'body')
            pred_body=unnorm_t(o['body'],'body')

            phase_acc=macro_phase_acc(o['phase_logits'],b['phase'][:,h].long())
            tdprob=torch.sigmoid(o['touchdown_logit'])
            tdy=b['touchdown'][:,h].float()
            td_pred=(tdprob>.5).float()
            td_acc=float((td_pred==tdy).float().mean())

            mask=b['event_mask'][:,h].float()
            if mask.sum()>0:
                event_time_mae=float(
                    (((o['event_offset_s']-b['event_offset'][:,h].float()).abs()*mask).sum()/mask.sum()).cpu()
                )
                predJ=unnorm_t(o['impulse'],'impulse')
                trueJ=unnorm_t(b['impulse'][:,h].float(),'impulse')
                impulse_mae=float(
                    (((predJ-trueJ).abs()*mask[...,None]).sum()/(mask.sum()*3)).cpu()
                )
            else:
                event_time_mae=np.nan; impulse_mae=np.nan

            rows.append(dict(
                run_id=rid,origin=origin,horizon_s=(h+1)*cfg.step_s,
                force_MAE_N=float(fe.mean().cpu()),
                force_p95_N=float(torch.quantile(fe.flatten(),.95).cpu()),
                q_MAE=float((pred_state[...,:3]-true_state[...,:3]).abs().mean().cpu()),
                dq_MAE=float((pred_state[...,3:6]-true_state[...,3:6]).abs().mean().cpu()),
                tau_MAE=float((pred_state[...,6:9]-true_state[...,6:9]).abs().mean().cpu()),
                foot_RMSE_m=float(torch.sqrt(((pred_foot-true_foot)**2).sum(-1).mean()).cpu()),
                phase_accuracy=phase_acc,
                touchdown_accuracy=td_acc,
                event_time_MAE_s=event_time_mae,
                impulse_MAE=impulse_mae,
                DINO_cosine=float(F.cosine_similarity(o['visual'],b['future_z'][:,h].float(),dim=-1).mean().cpu()),
                body_MAE=float((pred_body-true_body).abs().mean().cpu()),
            ))

            # ---- ride-quality post-processing from the EXISTING body head; no extra ride head ----
            # B[:3] = body specific-force/acceleration vector; B[3:6] = verified base-frame gyro.
            acc_B=pred_body[:,:3]
            omega_B=pred_body[:,3:6]
            theta=torch.linalg.norm(omega_B*cfg.step_s,dim=-1,keepdim=True).clamp_min(1e-9)
            axis=omega_B*cfg.step_s/theta
            K=torch.zeros((1,3,3),device=DEVICE,dtype=pred_body.dtype)
            ax,ay,az=axis[:,0],axis[:,1],axis[:,2]
            K[:,0,1]=-az; K[:,0,2]=ay
            K[:,1,0]=az; K[:,1,2]=-ax
            K[:,2,0]=-ay; K[:,2,1]=ax
            I=torch.eye(3,device=DEVICE,dtype=pred_body.dtype)[None]
            A=torch.sin(theta)[...,None]/theta[...,None]
            Bc=(1-torch.cos(theta))[...,None]/(theta[...,None]**2)
            dR=I+A*K+Bc*(K@K)
            Rpred=Rpred@dR

            aH=torch.einsum('bij,bj->bi',Rpred,acc_B)
            yaw=torch.atan2(Rpred[:,1,0],Rpred[:,0,0])
            dyaw=torch.atan2(torch.sin(yaw-prev_yaw),torch.cos(yaw-prev_yaw))
            pred_yaw_rate=dyaw/cfg.step_s
            prev_yaw=yaw

            j=origin+(h+1)*STEP
            Rtrue=torch.from_numpy(d['R_HB'][j]).to(DEVICE).float()[None]
            atrue=torch.from_numpy(d['B'][j,:3]).to(DEVICE).float()[None]
            true_aH=torch.einsum('bij,bj->bi',Rtrue,atrue)
            jprev=origin+h*STEP
            y0=math.atan2(float(d['R_HB'][jprev,1,0]),float(d['R_HB'][jprev,0,0]))
            y1=math.atan2(float(d['R_HB'][j,1,0]),float(d['R_HB'][j,0,0]))
            true_yaw_rate=math.atan2(math.sin(y1-y0),math.cos(y1-y0))/cfg.step_s

            ride_records.append(dict(
                run_id=rid,origin=origin,horizon_s=(h+1)*cfg.step_s,
                pred_vertical_acc=float(aH[0,2].cpu()),
                true_vertical_acc=float(true_aH[0,2].cpu()),
                pred_yaw_rate=float(pred_yaw_rate[0].cpu()),
                true_yaw_rate=float(true_yaw_rate),
            ))

            # fully autoregressive recursive state
            if h<cfg.H-1:
                z=torch.cat([z[:,1:],o['visual'][:,None]],1)
                pcur=o['p']

    return pd.DataFrame(rows),pd.DataFrame(ride_records)

TEST_METRICS,RIDE_SERIES=evaluate(test_loader)
display(TEST_METRICS)
TEST_METRICS.to_csv(OUT_ROOT/'test_metrics_long.csv',index=False)
RIDE_SERIES.to_csv(OUT_ROOT/'ride_series_specific_test_sequence.csv',index=False)

TIMES['testing_s']=time.time()-t0
print(f'TEST TIME = {TIMES["testing_s"]/60:.2f} min')

# %% notebook cell 17

# CELL 17 — APPENDED PAPER COMPARISON: ISAAC-SIM ROUGH VS THIS EXACT C_rough_uneven_P2006 TEST SEQUENCE
# Values are copied from the supplied Feel-WM Table-II image.
#
# Paper Rough row:
# Visual-only WM      rough=6.46 m/s², yaw vib=0.73 rad/s
# Feel-WM w/o prop.   rough=6.53 m/s², yaw vib=0.77 rad/s
# Feel-WM             rough=6.51 m/s², yaw vib=0.65 rad/s
#
# For this test sequence:
# roughness = std(vertical acceleration)
# yaw vibration = mean(abs(heading rate))

real_rough=float(RIDE_SERIES.true_vertical_acc.std(ddof=0))
real_yaw=float(np.mean(np.abs(RIDE_SERIES.true_yaw_rate)))
pred_rough=float(RIDE_SERIES.pred_vertical_acc.std(ddof=0))
pred_yaw=float(np.mean(np.abs(RIDE_SERIES.pred_yaw_rate)))

COMPARE=pd.DataFrame([
    dict(source='Feel-WM paper: Isaac Sim Rough / Visual-only WM',
         roughness_m_s2=6.46,yaw_vibration_rad_s=0.73,
         scope='paper closed-loop aggregate'),
    dict(source='Feel-WM paper: Isaac Sim Rough / Feel-WM w/o prop.',
         roughness_m_s2=6.53,yaw_vibration_rad_s=0.77,
         scope='paper closed-loop aggregate'),
    dict(source='Feel-WM paper: Isaac Sim Rough / Feel-WM',
         roughness_m_s2=6.51,yaw_vibration_rad_s=0.65,
         scope='paper closed-loop aggregate'),
    dict(source=f'OURS measured: {TEST_RUN_ID} exact test sequence',
         roughness_m_s2=real_rough,yaw_vibration_rad_s=real_yaw,
         scope='measured fixed logged TartanGround/ANYmal sequence'),
    dict(source=f'OURS predicted: {TEST_RUN_ID} exact test sequence',
         roughness_m_s2=pred_rough,yaw_vibration_rad_s=pred_yaw,
         scope='1-s autoregressive body predictions on same logged sequence'),
])
display(COMPARE)
COMPARE.to_csv(OUT_ROOT/'feelwm_rough_yaw_comparison.csv',index=False)

print('SPECIFIC TEST RUN:',TEST_RUN_ID)
print('test origins:',TEST_SAMPLES)
print(f'MEASURED roughness     = {real_rough:.4f} m/s²')
print(f'MEASURED yaw vibration = {real_yaw:.4f} rad/s')
print(f'PREDICTED roughness    = {pred_rough:.4f} m/s²')
print(f'PREDICTED yaw vibration= {pred_yaw:.4f} rad/s')
print(
    '\nInterpretation guard: Feel-WM values are Isaac-Sim closed-loop route aggregates. '
    'This is a fixed logged ANYmal/TartanGround sequence. A lower number here is not, by itself, '
    'evidence that this model outperforms Feel-WM.'
)

# %% notebook cell 18

# CELL 18 — EXACT DATA-IMPORT / PRECOMPUTE / TRAIN / TEST RUNTIME REPORT
TIMES['total_s']=time.time()-T_GLOBAL
runtime=pd.DataFrame([
    ('audit + split',TIMES.get('audit_split_s',np.nan)),
    ('prepared-data load',TIMES.get('prepared_load_s',np.nan)),
    ('frozen V10 proposal load',TIMES.get('proposal_load_s',np.nan)),
    ('origin selection',TIMES.get('origin_selection_s',np.nan)),
    ('normalization',TIMES.get('normalization_s',np.nan)),
    ('front RGB import/download',TIMES.get('rgb_import_s',np.nan)),
    ('DINOv2 precompute',TIMES.get('dino_precompute_s',np.nan)),
    ('V10 foot/map precompute',TIMES.get('v10_query_precompute_s',np.nan)),
    ('FULL architecture training',TIMES.get('training_s',np.nan)),
    ('held-out testing',TIMES.get('testing_s',np.nan)),
    ('TOTAL',TIMES.get('total_s',np.nan)),
],columns=['stage','seconds'])
runtime['minutes']=runtime.seconds/60
display(runtime)
runtime.to_csv(OUT_ROOT/'runtime_report.csv',index=False)

summary=dict(
    architecture=dict(
        history='4 × 100 × 37',
        p='4 × 128',
        dino='4 frames × 256 × 768',
        geometry='4 × (32×9 + 18)',
        transformer='1032 × 768, 8 layers, 8 heads, FFN 3072',
        rollout='10 × 0.1 s',
    ),
    parameters=int(PARAMS),
    frozen_proposal_checkpoint=str(PROPOSAL_CKPT),
    train_runs=list(TRAIN_RUN_IDS),val_run=VAL_RUN_ID,test_run=TEST_RUN_ID,
    train_samples=[list(x) for x in TRAIN_SAMPLES],
    val_samples=[list(x) for x in VAL_SAMPLES],
    test_samples=[list(x) for x in TEST_SAMPLES],
    best_val=float(best),
    runtime_minutes={k:float(v/60) for k,v in TIMES.items()},
    ride_comparison=COMPARE.to_dict(orient='records'),
)
(OUT_ROOT/'run_summary.json').write_text(json.dumps(summary,indent=2))

print('OUTPUT DIRECTORY:',OUT_ROOT)
print(f'TOTAL WALL TIME = {TIMES["total_s"]/60:.2f} min')
print(f'FULL-MODEL TRAINING TIME = {TIMES["training_s"]/60:.2f} min')
print(f'FRONT RGB IMPORT/DOWNLOAD = {TIMES["rgb_import_s"]/60:.2f} min')
print(f'DINO PRECOMPUTE = {TIMES["dino_precompute_s"]/60:.2f} min')


# Prompt-2/3 architecture identity audit.
architecture_audit={
    'name':'Prompt 2-3 assistant-proposed V10 x Feel-WM v1',
    'handwritten_hard_tminus2_tminus3':False,
    'leg_history':'4 x 100 x 37',
    'shared_leg_cnn':'37->64->128->128->128 -> 128',
    'global_p':'512->256->128',
    'action':'[dx,dy,dyaw,dt] -> 128',
    'terrain_query':'frozen V10 predicted foot -> causal RGB-D map',
    'contact_visual':'source-frame DINO pixels belonging to predicted metric contact patch',
    'geometry':'32x9 + 18 -> 128 -> 768',
    'transformer_input':'1032 x 768',
    'transformer':'8 layers, 8 heads, head_dim 96, FFN 3072, dropout .1',
    'leg_fusion':'2304->1024->768 shared',
    'rollout':'0.1 s x 10 = 1.0 s',
    'heads':[
        'Z_next','p_leg_next','q_dq_tau','force_mean_logvar',
        'contact_phase_4way','future_foot',
        'touchdown_hazard','touchdown_offset','touchdown_impulse',
        'body_next','failure_risk'
    ]
}
(OUT_ROOT/'prompt2_3_architecture_audit.json').write_text(
    json.dumps(architecture_audit,indent=2)
)
print(json.dumps(architecture_audit,indent=2))
