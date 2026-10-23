# Source inspection export of the adjacent notebook.
# Notebook execution order and dependencies still apply.
# IPython magic, shell lines and prose-only conclusion cells are comments.

# %% notebook cell 1

# CELL 1 — COLAB SETUP + FAST CONFIG + GLOBAL TIMER
import os, time, math, json, random, warnings, gc
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from IPython.display import display

from google.colab import drive
drive.mount('/content/drive', force_remount=False)

SEED = 230926
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision('high')
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

T_GLOBAL = time.time()
TIMES = {}

@dataclass
class Config:
    project_root: str = '/content/drive/MyDrive/UGP_V7_FULL4B'
    quick_test_run_id: str = ''
    use_fresh_c_test: bool = False
    fresh_c_test_run_id: str = 'C_rough_uneven_P2006'
    max_train_runs: int = 3

    sample_hz: int = 100
    history_s: float = 1.0
    step_s: float = 0.10
    rollout_steps: int = 3
    origin_stride: int = 10

    max_train_windows: int = 2400
    max_val_windows: int = 400
    max_test_windows: int = 600

    batch_size: int = 32
    epochs: int = 8
    max_train_minutes: float = 14.0
    patience: int = 2
    lr: float = 3e-4
    weight_decay: float = 2e-4
    grad_clip: float = 1.0

    p_dim: int = 128
    d_model: int = 256
    nhead: int = 8
    tx_layers: int = 4
    dropout: float = 0.10

    max_visual_age_s: float = 0.25
    num_workers: int = 0

CFG = Config()
HIST = int(round(CFG.history_s * CFG.sample_hz))
STEP = int(round(CFG.step_s * CFG.sample_hz))
assert HIST == 100 and STEP == 10

print(json.dumps(asdict(CFG), indent=2))
print('DEVICE:', DEVICE, torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
if DEVICE.type != 'cuda':
    warnings.warn('The 20–30 minute target assumes a Colab GPU.')

# %% notebook cell 2

# CELL 2 — PATHS, ARTIFACT AUDIT, AND AUTOMATIC NON-LEAKING SPLIT
t0 = time.time()

PROJECT_ROOT = Path(CFG.project_root)
STRICT_ROOT = PROJECT_ROOT / 'V10R_STRICT_STAGED'
B_ROOT = STRICT_ROOT / 'B_dynamics_repair_v1'
C_ROOT = STRICT_ROOT / 'stage_C_rgbd_incremental_v1'
C7R_ROOT = C_ROOT / 'spatial_temporal_repair_v1'
FEATURE_ROOT = C7R_ROOT / 'frame_features'
DERIVED_ROOT = STRICT_ROOT / 'derived_real_targets_v1'
CACHE_ROOT = PROJECT_ROOT / 'cache'
ROT_ROOT = B_ROOT / 'derived'
SPLIT_PATH = C_ROOT / 'C_split_manifest.csv'
OUT_ROOT = PROJECT_ROOT / 'V10_FEELWM_FAST_20_30MIN'
OUT_ROOT.mkdir(parents=True, exist_ok=True)

required_top = [SPLIT_PATH, FEATURE_ROOT, DERIVED_ROOT, CACHE_ROOT, ROT_ROOT]
missing = [str(p) for p in required_top if not p.exists()]
if missing:
    raise FileNotFoundError(
        'Missing persistent V10/C7R artifacts:\n  ' + '\n  '.join(missing) +
        '\nThis notebook intentionally does NOT redownload raw RGB-D.'
    )

manifest = pd.read_csv(SPLIT_PATH).copy()
required_cols = {'run_id','environment','run','terrain','role'}
miss_cols = required_cols.difference(manifest.columns)
if miss_cols:
    raise RuntimeError(f'C split manifest is missing columns: {sorted(miss_cols)}')

def paths_for_row(r):
    return {
        'physics': CACHE_ROOT / f'{r.environment}_{r.run}_physics.npz',
        'derived': DERIVED_ROOT / f'{r.run_id}_real_targets.npz',
        'rotation': ROT_ROOT / f'{r.run_id}_R_odom_from_base.npy',
        'feature': FEATURE_ROOT / f'{r.run_id}_spatial_frame_features.npz',
    }

audit = []
for r in manifest.itertuples(index=False):
    pp = paths_for_row(r)
    audit.append({
        'run_id': str(r.run_id), 'role': str(r.role), 'terrain': str(r.terrain),
        'environment': str(r.environment), 'run': str(r.run),
        **{k: p.is_file() for k,p in pp.items()}
    })
audit = pd.DataFrame(audit)
audit['all_ready'] = audit[['physics','derived','rotation','feature']].all(axis=1)
display(audit)

ready_ids = set(audit.loc[audit.all_ready,'run_id'].astype(str))
ready = manifest[manifest.run_id.astype(str).isin(ready_ids)].copy()
if len(ready) < 3:
    raise RuntimeError('Need at least 3 fully prepared runs for train / val / quick held-out test.')

def rough_mask(df):
    s = (
        df.run_id.astype(str) + ' ' +
        df.terrain.astype(str) + ' ' +
        df.environment.astype(str)
    ).str.lower()
    return s.str.contains('rough|uneven|seasonal|rock|hill', regex=True)

FRESH_IDS = {'C_swamp_marsh_P2027','C_rough_uneven_P2006'}

if CFG.use_fresh_c_test:
    if CFG.fresh_c_test_run_id not in ready_ids:
        raise RuntimeError(
            f'Fresh test {CFG.fresh_c_test_run_id} is not fully prepared in all required caches.'
        )
    TEST_RUN_ID = CFG.fresh_c_test_run_id
else:
    if CFG.quick_test_run_id:
        if CFG.quick_test_run_id not in ready_ids:
            raise RuntimeError(f'quick_test_run_id={CFG.quick_test_run_id} is not fully prepared.')
        TEST_RUN_ID = CFG.quick_test_run_id
    else:
        q = ready[(ready.role.astype(str).str.lower()=='val') & rough_mask(ready)]
        if len(q)==0: q = ready[ready.role.astype(str).str.lower()=='val']
        if len(q)==0: q = ready[rough_mask(ready) & ~ready.run_id.astype(str).isin(FRESH_IDS)]
        if len(q)==0:
            raise RuntimeError('Could not find a non-fresh rough/validation run for quick held-out testing.')
        TEST_RUN_ID = str(q.iloc[0].run_id)

val_pool = ready[
    (ready.role.astype(str).str.lower()=='val') &
    (ready.run_id.astype(str) != TEST_RUN_ID) &
    (~ready.run_id.astype(str).isin(FRESH_IDS))
]
if len(val_pool)==0:
    val_pool = ready[
        (ready.role.astype(str).str.lower()=='train') &
        (ready.run_id.astype(str) != TEST_RUN_ID)
    ]
if len(val_pool)==0:
    raise RuntimeError('No independent validation trajectory available.')
VAL_RUN_ID = str(val_pool.iloc[0].run_id)

train_pool = ready[
    (ready.role.astype(str).str.lower()=='train') &
    (~ready.run_id.astype(str).isin({TEST_RUN_ID,VAL_RUN_ID})) &
    (~ready.run_id.astype(str).isin(FRESH_IDS))
].copy()
if len(train_pool)==0:
    raise RuntimeError('No training trajectory remains after reserving val/test.')

picked = []
rp = train_pool[rough_mask(train_pool)]
if len(rp): picked.append(str(rp.iloc[0].run_id))

sp = train_pool[
    (train_pool.run_id.astype(str)+' '+train_pool.terrain.astype(str)).str.lower()
    .str.contains('swamp|marsh|mud', regex=True)
]
if len(sp) and str(sp.iloc[0].run_id) not in picked:
    picked.append(str(sp.iloc[0].run_id))

for rid in train_pool.run_id.astype(str):
    if rid not in picked: picked.append(rid)
    if len(picked) >= CFG.max_train_runs: break

TRAIN_RUN_IDS = tuple(picked[:CFG.max_train_runs])

assert TEST_RUN_ID not in TRAIN_RUN_IDS
assert VAL_RUN_ID not in TRAIN_RUN_IDS
assert TEST_RUN_ID != VAL_RUN_ID
if not CFG.use_fresh_c_test: assert TEST_RUN_ID not in FRESH_IDS

SELECTED = list(TRAIN_RUN_IDS) + [VAL_RUN_ID, TEST_RUN_ID]
rows = ready[ready.run_id.astype(str).isin(SELECTED)].copy()

print('\nLOCKED FAST SPLIT')
print('TRAIN:', TRAIN_RUN_IDS)
print('VAL  :', VAL_RUN_ID)
print('TEST :', TEST_RUN_ID)
print('fresh C test used:', CFG.use_fresh_c_test)

TIMES['audit_split_s'] = time.time()-t0
print(f'Audit + split: {TIMES["audit_split_s"]:.1f}s')

# %% notebook cell 3

# CELL 3 — LOAD CACHES + BUILD STRICTLY CAUSAL 37-D LEG HISTORIES
t0 = time.time()

def causal_rate(x,t,clip=None):
    x=np.asarray(x,np.float32); t=np.asarray(t,np.float64)
    out=np.zeros_like(x,dtype=np.float32)
    if len(x)>1:
        dt=np.maximum(np.diff(t),1e-6)
        out[1:]=np.diff(x,axis=0)/dt.reshape((-1,)+(1,)*(x.ndim-1))
        out[0]=out[1]
    if clip is not None: out=np.clip(out,-clip,clip)
    return out

def heading_signals(R, B, t):
    yaw=np.unwrap(np.arctan2(R[:,1,0],R[:,0,0])).astype(np.float64)
    yaw_rate=np.gradient(yaw,t).astype(np.float32)
    a_body=np.asarray(B[:,:3],np.float32)
    a_H=np.einsum('nij,nj->ni',R.astype(np.float32),a_body)
    vertical_accel=a_H[:,2].astype(np.float32)
    return vertical_accel,yaw_rate

def load_run(row):
    p=paths_for_row(row)
    for k,v in p.items():
        if not v.is_file(): raise FileNotFoundError(f'{row.run_id}: missing {k}: {v}')

    with np.load(p['physics'],allow_pickle=False) as z:
        time_s=np.asarray(z['time'],np.float64)
        P=np.asarray(z['P'],np.float32)
        B=np.asarray(z['B'],np.float32)
        U=np.asarray(z['U'],np.float32)
        L=np.asarray(z['L_LEG'],np.float32)
        E=np.asarray(z['E'],np.float32)

    with np.load(p['derived'],allow_pickle=False) as z:
        keys=set(z.files)
        need={'foot_base','force_odom_from_anymal_state','contact_from_anymal_state','base_position_odom'}
        if not need.issubset(keys):
            raise RuntimeError(f'{row.run_id}: derived targets missing {sorted(need-keys)}')
        foot=np.asarray(z['foot_base'],np.float32)
        force=np.asarray(z['force_odom_from_anymal_state'],np.float32)
        contact=np.asarray(z['contact_from_anymal_state'],np.float32)
        base_pos=np.asarray(z['base_position_odom'],np.float32)
        touchdown=np.asarray(z['touchdown_event'],np.float32) if 'touchdown_event' in keys else None

    R=np.asarray(np.load(p['rotation']),np.float32)

    with np.load(p['feature'],allow_pickle=False) as z:
        fkeys=set(z.files)
        need={'camera_times','rgb_spatial','depth_spatial'}
        if not need.issubset(fkeys):
            raise RuntimeError(f'{row.run_id}: Stage-C feature cache missing {sorted(need-fkeys)}')
        camera_times=np.asarray(z['camera_times'],np.float64)
        rgb_flat=np.asarray(z['rgb_spatial'],np.float32)
        depth_flat=np.asarray(z['depth_spatial'],np.float32)

    n=len(time_s)
    for name,a in [('P',P),('B',B),('U',U),('L',L),('E',E),('foot',foot),
                   ('force',force),('contact',contact),('base_pos',base_pos),('R',R)]:
        if len(a)!=n: raise RuntimeError(f'{row.run_id}: {name} length {len(a)} != {n}')

    if touchdown is None:
        c=contact>0.5
        prev=np.vstack([c[:1],c[:-1]])
        touchdown=((~prev)&c).astype(np.float32)
    if len(touchdown)!=n: raise RuntimeError(f'{row.run_id}: touchdown length mismatch')

    if rgb_flat.ndim!=2 or rgb_flat.shape[1] % 6:
        raise RuntimeError(f'{row.run_id}: unexpected rgb_spatial shape {rgb_flat.shape}')
    if depth_flat.ndim!=2 or depth_flat.shape[1] % 64:
        raise RuntimeError(f'{row.run_id}: unexpected depth_spatial shape {depth_flat.shape}')

    dino_dim=rgb_flat.shape[1]//6
    depth_dim=depth_flat.shape[1]//64
    rgb=rgb_flat.reshape(len(rgb_flat),6,dino_dim)
    depth=depth_flat.reshape(len(depth_flat),64,depth_dim)

    dE=causal_rate(E,time_s,20.0)
    dF=causal_rate(force,time_s,2e4)
    B4=np.repeat(B[:,None,:],4,axis=1)

    # 37 = q,dq,tau(9) + force3 + contact1 + L3 + E3 + dE3 + dF3 + B12
    leg_x=np.concatenate([
        P[:,:,:9], force, contact[...,None], L, E, dE, dF, B4
    ],axis=-1).astype(np.float32)
    assert leg_x.shape==(n,4,37),leg_x.shape

    vi_all=(np.searchsorted(camera_times,time_s,side='right')-1).astype(np.int64)
    ride_az,yaw_rate=heading_signals(R,B,time_s)

    return dict(
        run_id=str(row.run_id), terrain=str(row.terrain), role=str(row.role),
        time=time_s,P=P,B=B,U=U,L=L,E=E,force=force,contact=contact,touchdown=touchdown,
        foot=foot,base_pos=base_pos,R=R,leg_x=leg_x,
        camera_times=camera_times,rgb=rgb,depth=depth,vi_all=vi_all,
        ride_az=ride_az,yaw_rate=yaw_rate
    )

RUNS={}
for rid in SELECTED:
    row=rows[rows.run_id.astype(str)==rid]
    if len(row)!=1: raise RuntimeError(f'Expected one manifest row for {rid}, found {len(row)}')
    print('loading',rid)
    RUNS[rid]=load_run(row.iloc[0])

DINO_DIM=RUNS[TRAIN_RUN_IDS[0]]['rgb'].shape[-1]
DEPTH_CELL_DIM=RUNS[TRAIN_RUN_IDS[0]]['depth'].shape[-1]
VIS_PATCHES=RUNS[TRAIN_RUN_IDS[0]]['rgb'].shape[1]
DEPTH_CELLS=RUNS[TRAIN_RUN_IDS[0]]['depth'].shape[1]

for rid,d in RUNS.items():
    if d['rgb'].shape[-1]!=DINO_DIM or d['depth'].shape[-1]!=DEPTH_CELL_DIM:
        raise RuntimeError('Feature dimensions differ across selected runs.')
    print(rid,'N=',len(d['time']),'rgb=',d['rgb'].shape,'depth=',d['depth'].shape,'leg=',d['leg_x'].shape)

TIMES['data_load_s']=time.time()-t0
print({'VIS_PATCHES':VIS_PATCHES,'DINO_DIM':DINO_DIM,
       'DEPTH_CELLS':DEPTH_CELLS,'DEPTH_CELL_DIM':DEPTH_CELL_DIM})
print(f'DATA IMPORT / LOAD = {TIMES["data_load_s"]/60:.2f} min')

# %% notebook cell 4

# CELL 4 — ORIGINS + TRAIN-ONLY NORMALIZATION
t0=time.time()

def action_between(d,i0,i1):
    R0,R1=d['R'][i0],d['R'][i1]
    dp_H=d['base_pos'][i1]-d['base_pos'][i0]
    dp_B=R0.T@dp_H
    Rrel=R0.T@R1
    dyaw=math.atan2(float(Rrel[1,0]),float(Rrel[0,0]))
    dt=float(d['time'][i1]-d['time'][i0])
    return np.array([dp_B[0],dp_B[1],dyaw,dt],np.float32)

def valid_origins(d):
    lo=HIST-1
    hi=len(d['time'])-CFG.rollout_steps*STEP-1
    cand=np.arange(lo,hi+1,CFG.origin_stride,dtype=np.int64)
    keep=[]
    for i in cand:
        vi=int(d['vi_all'][i])
        if vi<3: continue
        age=d['time'][i]-d['camera_times'][vi]
        if age<-1e-9 or age>CFG.max_visual_age_s: continue
        ok=True
        for h in range(1,CFG.rollout_steps+1):
            j=i+h*STEP
            vj=int(d['vi_all'][j])
            if vj<0: ok=False; break
            agej=d['time'][j]-d['camera_times'][vj]
            if agej<-1e-9 or agej>CFG.max_visual_age_s:
                ok=False; break
        if ok: keep.append(i)
    return np.asarray(keep,np.int64)

ORIGINS={rid:valid_origins(d) for rid,d in RUNS.items()}
print('valid origins:',{rid:len(v) for rid,v in ORIGINS.items()})
if any(len(v)==0 for v in ORIGINS.values()):
    raise RuntimeError('At least one selected run has zero valid causal visual/physics origins.')

def cap_origins(ids,max_total,seed,randomize):
    chunks=[(rid,ORIGINS[rid]) for rid in ids]
    total=sum(len(x) for _,x in chunks)
    if total<=max_total:
        return [(rid,int(i)) for rid,x in chunks for i in x]
    rng=np.random.default_rng(seed)
    out=[]
    for rid,x in chunks:
        k=max(1,int(round(max_total*len(x)/total)))
        k=min(k,len(x))
        sel=np.sort(rng.choice(x,size=k,replace=False)) if randomize else x[np.linspace(0,len(x)-1,k).round().astype(int)]
        out.extend((rid,int(i)) for i in sel)
    if len(out)>max_total:
        if randomize:
            idx=np.sort(rng.choice(len(out),max_total,replace=False))
            out=[out[k] for k in idx]
        else:
            out=[out[k] for k in np.linspace(0,len(out)-1,max_total).round().astype(int)]
    return out

TRAIN_SAMPLES=cap_origins(TRAIN_RUN_IDS,CFG.max_train_windows,SEED,True)
VAL_SAMPLES=cap_origins((VAL_RUN_ID,),CFG.max_val_windows,SEED+1,False)
TEST_SAMPLES=cap_origins((TEST_RUN_ID,),CFG.max_test_windows,SEED+2,False)
print({'train':len(TRAIN_SAMPLES),'val':len(VAL_SAMPLES),'test':len(TEST_SAMPLES)})

def mean_std(arrays,last_dim):
    x=np.concatenate([np.asarray(a).reshape(-1,last_dim) for a in arrays],axis=0).astype(np.float64)
    m=x.mean(0).astype(np.float32)
    s=x.std(0).astype(np.float32)
    s[s<1e-5]=1.0
    return m,s

train_runs=[RUNS[rid] for rid in TRAIN_RUN_IDS]
leg_mean,leg_std=mean_std([d['leg_x'] for d in train_runs],37)
state_mean,state_std=mean_std([d['P'][:,:,:9] for d in train_runs],9)
force_mean,force_std=mean_std([d['force'] for d in train_runs],3)
body_mean,body_std=mean_std([d['B'] for d in train_runs],12)
foot_mean,foot_std=mean_std([d['foot'] for d in train_runs],3)
vis_mean,vis_std=mean_std([d['rgb'] for d in train_runs],DINO_DIM)
depth_mean,depth_std=mean_std([d['depth'] for d in train_runs],DEPTH_CELL_DIM)
rideaz_mean,rideaz_std=mean_std([d['ride_az'][:,None] for d in train_runs],1)
yawrate_mean,yawrate_std=mean_std([d['yaw_rate'][:,None] for d in train_runs],1)

all_actions=[]
for rid,i in TRAIN_SAMPLES:
    d=RUNS[rid]
    for h in range(CFG.rollout_steps):
        j0=i+h*STEP; j1=i+(h+1)*STEP
        all_actions.append(action_between(d,j0,j1))
all_actions=np.stack(all_actions)
action_mean=all_actions.mean(0).astype(np.float32)
action_std=all_actions.std(0).astype(np.float32)
action_std[action_std<1e-5]=1.0

c=np.concatenate([d['contact'].reshape(-1) for d in train_runs])
td=np.concatenate([d['touchdown'].reshape(-1) for d in train_runs])
contact_pos_weight=float(np.clip((len(c)-c.sum())/max(c.sum(),1),0.5,8.0))
td_pos_weight=float(np.clip((len(td)-td.sum())/max(td.sum(),1),1.0,20.0))

NORM=dict(
    leg_mean=leg_mean,leg_std=leg_std,state_mean=state_mean,state_std=state_std,
    force_mean=force_mean,force_std=force_std,body_mean=body_mean,body_std=body_std,
    foot_mean=foot_mean,foot_std=foot_std,vis_mean=vis_mean,vis_std=vis_std,
    depth_mean=depth_mean,depth_std=depth_std,action_mean=action_mean,action_std=action_std,
    rideaz_mean=rideaz_mean,rideaz_std=rideaz_std,
    yawrate_mean=yawrate_mean,yawrate_std=yawrate_std
)
np.savez_compressed(OUT_ROOT/'normalization_fast.npz',**NORM)

TIMES['prep_s']=time.time()-t0
print({'contact_pos_weight':contact_pos_weight,'touchdown_pos_weight':td_pos_weight})
print(f'INDEX + NORMALIZATION = {TIMES["prep_s"]/60:.2f} min')

# %% notebook cell 5

# CELL 5 — DATASET + DATALOADERS
def norm_np(x,key):
    return ((x-NORM[key+'_mean'])/NORM[key+'_std']).astype(np.float32)

class FastFeelV10Dataset(Dataset):
    def __init__(self,samples): self.samples=list(samples)
    def __len__(self): return len(self.samples)
    def __getitem__(self,k):
        rid,i=self.samples[k]; d=RUNS[rid]
        leg_hist=norm_np(d['leg_x'][i-HIST+1:i+1].transpose(1,0,2),'leg')

        vi=int(d['vi_all'][i])
        vis_ctx=norm_np(d['rgb'][np.arange(vi-3,vi+1)],'vis')
        depth0=norm_np(d['depth'][vi],'depth')

        actions=[]; future_vis=[]; future_hist=[]
        y_state=[]; y_force=[]; y_body=[]; y_contact=[]; y_td=[]; y_foot=[]
        y_rideaz=[]; y_yawrate=[]

        for h in range(CFG.rollout_steps):
            j0=i+h*STEP; j1=i+(h+1)*STEP
            actions.append(norm_np(action_between(d,j0,j1),'action'))
            vj=int(d['vi_all'][j1])
            future_vis.append(norm_np(d['rgb'][vj],'vis'))
            fh=d['leg_x'][j1-HIST+1:j1+1].transpose(1,0,2)
            future_hist.append(norm_np(fh,'leg'))
            y_state.append(norm_np(d['P'][j1,:,:9],'state'))
            y_force.append(norm_np(d['force'][j1],'force'))
            y_body.append(norm_np(d['B'][j1],'body'))
            y_contact.append(d['contact'][j1].astype(np.float32))
            y_td.append(d['touchdown'][j1].astype(np.float32))
            y_foot.append(norm_np(d['foot'][j1],'foot'))
            y_rideaz.append(norm_np(np.array([d['ride_az'][j1]],np.float32),'rideaz')[0])
            y_yawrate.append(norm_np(np.array([d['yaw_rate'][j1]],np.float32),'yawrate')[0])

        return dict(
            rid=rid,origin=np.int64(i),leg_hist=leg_hist,vis_ctx=vis_ctx,depth=depth0,
            actions=np.stack(actions),future_vis=np.stack(future_vis),future_hist=np.stack(future_hist),
            state=np.stack(y_state),force=np.stack(y_force),body=np.stack(y_body),
            contact=np.stack(y_contact),touchdown=np.stack(y_td),foot=np.stack(y_foot),
            ride_az=np.asarray(y_rideaz,np.float32),
            yaw_rate=np.asarray(y_yawrate,np.float32),
        )

train_ds=FastFeelV10Dataset(TRAIN_SAMPLES)
val_ds=FastFeelV10Dataset(VAL_SAMPLES)
test_ds=FastFeelV10Dataset(TEST_SAMPLES)

kwargs=dict(
    batch_size=CFG.batch_size,
    num_workers=CFG.num_workers,
    pin_memory=True,
    persistent_workers=(CFG.num_workers>0)
)
train_loader=DataLoader(train_ds,shuffle=True,drop_last=True,**kwargs)
val_loader=DataLoader(val_ds,shuffle=False,drop_last=False,**kwargs)
test_loader=DataLoader(test_ds,shuffle=False,drop_last=False,**kwargs)

b=next(iter(train_loader))
for k,v in b.items():
    if torch.is_tensor(v): print(k,tuple(v.shape))

# %% notebook cell 6

# CELL 6 — MODEL: 1-D CNN + ACTION/LEG FiLM + DINO/DEPTH TOKENS + SHARED TRANSFORMER
class LegHistoryCNN(nn.Module):
    def __init__(self,p_dim=128):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv1d(37,64,7,2,3),nn.GroupNorm(8,64),nn.GELU(),
            nn.Conv1d(64,128,5,2,2),nn.GroupNorm(8,128),nn.GELU(),
            nn.Conv1d(128,128,3,2,1),nn.GroupNorm(8,128),nn.GELU(),
            nn.Conv1d(128,p_dim,3,1,1),nn.GroupNorm(8,p_dim),nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.norm=nn.LayerNorm(p_dim)
    def forward(self,x):
        return self.norm(self.net(x.transpose(1,2)).squeeze(-1))

class CondFiLM(nn.Module):
    def __init__(self,cond_dim,d_model):
        super().__init__()
        self.trunk=nn.Sequential(nn.Linear(cond_dim,256),nn.GELU(),nn.LayerNorm(256))
        self.v_gamma=nn.Linear(256,d_model); self.v_beta=nn.Linear(256,d_model)
        self.g_gamma=nn.Linear(256,d_model); self.g_beta=nn.Linear(256,d_model)
        for m in [self.v_gamma,self.v_beta,self.g_gamma,self.g_beta]:
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
    def apply(self,x,c,kind):
        r=self.trunk(c)
        if kind=='visual': dg,b=self.v_gamma(r),self.v_beta(r)
        else: dg,b=self.g_gamma(r),self.g_beta(r)
        return x*(1.0+dg)+b

class QueryPool(nn.Module):
    def __init__(self,cond_dim,d_model):
        super().__init__()
        self.q=nn.Linear(cond_dim,d_model); self.k=nn.Linear(d_model,d_model); self.v=nn.Linear(d_model,d_model)
    def forward(self,cond,tokens):
        q=self.q(cond)[:,None,:]; k=self.k(tokens); v=self.v(tokens)
        w=torch.softmax((q*k).sum(-1)/math.sqrt(tokens.shape[-1]),dim=-1)
        return (w[...,None]*v).sum(1)

class V10FeelWMFast(nn.Module):
    def __init__(self):
        super().__init__()
        p=CFG.p_dim; d=CFG.d_model

        self.leg_hist=LegHistoryCNN(p)
        self.leg_id=nn.Parameter(torch.randn(4,p)*0.02)
        self.global_p=nn.Sequential(
            nn.Linear(4*p,256),nn.GELU(),nn.LayerNorm(256),
            nn.Linear(256,p),nn.LayerNorm(p)
        )
        self.action=nn.Sequential(
            nn.Linear(4,128),nn.GELU(),nn.LayerNorm(128),
            nn.Linear(128,p),nn.LayerNorm(p)
        )

        self.vis_in=nn.Sequential(nn.LayerNorm(DINO_DIM),nn.Linear(DINO_DIM,d))
        self.depth_in=nn.Sequential(
            nn.LayerNorm(DEPTH_CELL_DIM),nn.Linear(DEPTH_CELL_DIM,128),
            nn.GELU(),nn.Linear(128,d)
        )
        self.film=CondFiLM(p,d)

        self.prefoot=nn.Sequential(
            nn.Linear(p,128),nn.GELU(),nn.LayerNorm(128),nn.Linear(128,3)
        )
        self.query_cond=nn.Sequential(
            nn.Linear(p+3,p),nn.GELU(),nn.LayerNorm(p)
        )
        self.vis_pool=QueryPool(p,d)
        self.geom_pool=QueryPool(p,d)

        self.time_embed=nn.Parameter(torch.randn(4,d)*0.02)
        self.vis_pos=nn.Parameter(torch.randn(VIS_PATCHES,d)*0.02)
        self.depth_pos=nn.Parameter(torch.randn(DEPTH_CELLS,d)*0.02)
        self.type_embed=nn.Parameter(torch.randn(4,d)*0.02)
        self.leg_token_embed=nn.Parameter(torch.randn(4,d)*0.02)
        self.body_token=nn.Linear(p,d)

        layer=nn.TransformerEncoderLayer(
            d_model=d,nhead=CFG.nhead,dim_feedforward=4*d,
            dropout=CFG.dropout,activation='gelu',
            batch_first=True,norm_first=True
        )
        self.tx=nn.TransformerEncoder(layer,CFG.tx_layers)
        self.out_norm=nn.LayerNorm(d)

        self.visual_head=nn.Linear(d,DINO_DIM)
        self.global_fuse=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.LayerNorm(d))
        self.leg_fuse=nn.Sequential(
            nn.Linear(3*d,2*d),nn.GELU(),nn.LayerNorm(2*d),
            nn.Linear(2*d,d),nn.GELU(),nn.LayerNorm(d)
        )

        self.p_head=nn.Linear(d,p)
        self.state_head=nn.Linear(d,9)
        self.force_head=nn.Linear(d,6)
        self.contact_head=nn.Linear(d,1)
        self.touchdown_head=nn.Linear(d,1)
        self.foot_head=nn.Linear(d,3)
        self.body_head=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,12))
        self.ride_head=nn.Linear(d,2)

    def encode_history(self,hist):
        return torch.stack([self.leg_hist(hist[:,i]) for i in range(4)],1)

    def forward_step(self,vis_ctx,p_leg,depth_cells,action):
        B=vis_ctx.shape[0]; d=CFG.d_model

        pG=self.global_p(p_leg.reshape(B,-1))
        ea=self.action(action)
        cG=ea+pG
        ci=ea[:,None,:]+p_leg+self.leg_id[None,:,:]

        V=self.vis_in(vis_ctx)
        image_tokens=[]
        for t in range(4):
            x=(V[:,t]+self.time_embed[t][None,None,:]+
               self.vis_pos[None,:,:]+self.type_embed[0][None,None,:])
            if t==3:
                cg=cG[:,None,:].expand(-1,VIS_PATCHES,-1).reshape(-1,CFG.p_dim)
                x=self.film.apply(x.reshape(-1,d),cg,'visual').reshape(B,VIS_PATCHES,d)
            image_tokens.append(x)
        image_tokens=torch.cat(image_tokens,1)

        G=self.depth_in(depth_cells)+self.depth_pos[None,:,:]

        v_leg=[]; g_leg=[]; prefoot=[]
        cur=V[:,3]+self.vis_pos[None,:,:]
        for i in range(4):
            c=ci[:,i]
            fp=self.prefoot(c)
            qc=self.query_cond(torch.cat([c,fp],-1))
            prefoot.append(fp)

            vv=self.vis_pool(qc,cur)
            vv=self.film.apply(vv,c,'visual')+self.type_embed[1]+self.leg_token_embed[i]
            gg=self.geom_pool(qc,G)
            gg=self.film.apply(gg,c,'geometry')+self.type_embed[2]+self.leg_token_embed[i]
            v_leg.append(vv); g_leg.append(gg)

        v_leg=torch.stack(v_leg,1)
        g_leg=torch.stack(g_leg,1)
        prefoot=torch.stack(prefoot,1)

        body_tok=self.body_token(pG)+self.type_embed[3]
        x=torch.cat([image_tokens,body_tok[:,None,:],v_leg,g_leg],1)
        h=self.out_norm(self.tx(x))

        CUR0=3*VIS_PATCHES; CUR1=4*VIS_PATCHES
        BODY=CUR1; V0=BODY+1; G0=V0+4

        next_vis=self.visual_head(h[:,CUR0:CUR1])

        hcur=h[:,CUR0:CUR1].mean(1)
        hbody=h[:,BODY]
        hg=self.global_fuse(torch.cat([hcur,hbody],-1))
        hv=h[:,V0:V0+4]
        hgeom=h[:,G0:G0+4]

        hl=self.leg_fuse(torch.cat([
            hg[:,None,:].expand(-1,4,-1),hv,hgeom
        ],-1))

        fraw=self.force_head(hl)
        ride=self.ride_head(hg)
        return dict(
            visual=next_vis,p=self.p_head(hl),state=self.state_head(hl),
            force_mu=fraw[...,:3],force_logstd=torch.clamp(fraw[...,3:],-3,3),
            contact_logit=self.contact_head(hl).squeeze(-1),
            touchdown_logit=self.touchdown_head(hl).squeeze(-1),
            foot=self.foot_head(hl),foot_prior=prefoot,
            body=self.body_head(hg),ride_az=ride[:,0],yaw_rate=ride[:,1],
        )

model=V10FeelWMFast().to(DEVICE)
NPAR=sum(p.numel() for p in model.parameters())
print(f'Parameters: {NPAR/1e6:.2f} M')

# %% notebook cell 7

# CELL 7 — LOSSES + 3-STEP AUTOREGRESSIVE TRAINING WITH HARD TIME BUDGET
contact_pw=torch.tensor(contact_pos_weight,device=DEVICE)
td_pw=torch.tensor(td_pos_weight,device=DEVICE)

def to_dev(batch):
    return {k:(v.to(DEVICE,non_blocking=True) if torch.is_tensor(v) else v) for k,v in batch.items()}

def visual_loss(pred,target):
    return F.mse_loss(pred,target)+0.1*(1-F.cosine_similarity(pred,target,dim=-1).mean())

def gaussian_nll(mu,logstd,target):
    return (0.5*(target-mu).pow(2)*torch.exp(-2*logstd)+logstd).mean()

def tf_prob_for_epoch(ep):
    frac=(ep-1)/max(CFG.epochs-1,1)
    if frac<=.25: return 1.0
    if frac>=.75: return .5
    return 1.0-(frac-.25)/.50*.5

def batch_rollout_loss(batch,epoch,train=True):
    batch=to_dev(batch)
    hist=batch['leg_hist'].float()
    vis=batch['vis_ctx'].float()
    depth=batch['depth'].float()
    pcur=model.encode_history(hist)
    B=hist.shape[0]

    tfmask=(torch.rand(B,device=DEVICE)<tf_prob_for_epoch(epoch)
            if train else torch.zeros(B,dtype=torch.bool,device=DEVICE))

    total=0.0
    for h in range(CFG.rollout_steps):
        out=model.forward_step(vis,pcur,depth,batch['actions'][:,h].float())
        with torch.no_grad():
            pgt=model.encode_history(batch['future_hist'][:,h].float())

        Lv=visual_loss(out['visual'],batch['future_vis'][:,h].float())
        Lp=F.mse_loss(out['p'],pgt)
        Ls=F.smooth_l1_loss(out['state'],batch['state'][:,h].float(),beta=.5)
        Lf=gaussian_nll(out['force_mu'],out['force_logstd'],batch['force'][:,h].float())
        Lc=F.binary_cross_entropy_with_logits(out['contact_logit'],batch['contact'][:,h].float(),pos_weight=contact_pw)
        Ltd=F.binary_cross_entropy_with_logits(out['touchdown_logit'],batch['touchdown'][:,h].float(),pos_weight=td_pw)
        Lfoot=F.smooth_l1_loss(out['foot'],batch['foot'][:,h].float(),beta=.5)
        Lpre=F.smooth_l1_loss(out['foot_prior'],batch['foot'][:,h].float(),beta=.5)
        Lb=F.smooth_l1_loss(out['body'],batch['body'][:,h].float(),beta=.5)
        Laz=F.smooth_l1_loss(out['ride_az'],batch['ride_az'][:,h].float(),beta=.5)
        Lyaw=F.smooth_l1_loss(out['yaw_rate'],batch['yaw_rate'][:,h].float(),beta=.5)

        total=total+(
            1.00*Lv + .25*Lp + .50*Ls + 1.00*Lf +
            .20*Lc + .20*Ltd + .30*Lfoot + .12*Lpre +
            .45*Lb + .20*Laz + .20*Lyaw
        )/CFG.rollout_steps

        if h<CFG.rollout_steps-1:
            m=tfmask[:,None,None]
            next_vis=torch.where(m,batch['future_vis'][:,h].float(),out['visual'])
            vis=torch.cat([vis[:,1:],next_vis[:,None]],1)
            pcur=torch.where(m,pgt,out['p'])

    return total

@torch.no_grad()
def validation_loss(loader,epoch):
    model.eval(); vals=[]
    for batch in loader:
        vals.append(float(batch_rollout_loss(batch,epoch,train=False)))
    return float(np.mean(vals)) if vals else np.inf

optimizer=torch.optim.AdamW(model.parameters(),lr=CFG.lr,weight_decay=CFG.weight_decay)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,T_max=max(CFG.epochs,1),eta_min=CFG.lr*.08
)
USE_AMP=(DEVICE.type=='cuda')
scaler=torch.amp.GradScaler('cuda',enabled=USE_AMP)

best=np.inf
best_path=OUT_ROOT/'best_fast_20_30min.pt'
history=[]
train_start=time.time()
bad=0
hard_stop=False

for epoch in range(1,CFG.epochs+1):
    model.train(); losses=[]; ep0=time.time()

    for batch in train_loader:
        if (time.time()-train_start)/60.0 >= CFG.max_train_minutes:
            hard_stop=True
            print('Reached hard training wall-clock budget inside epoch.')
            break

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda',enabled=USE_AMP,dtype=torch.float16):
            loss=batch_rollout_loss(batch,epoch,train=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(),CFG.grad_clip)
        scaler.step(optimizer); scaler.update()
        losses.append(float(loss.detach()))

    if not losses: break

    val=validation_loss(val_loader,epoch)
    scheduler.step()
    rec=dict(
        epoch=epoch,train=float(np.mean(losses)),val=val,
        tf_prob=tf_prob_for_epoch(epoch),epoch_s=time.time()-ep0,
        elapsed_min=(time.time()-train_start)/60.0,
        lr=optimizer.param_groups[0]['lr']
    )
    history.append(rec); print(rec)

    if val < best-1e-5:
        best=val; bad=0
        torch.save({
            'model':model.state_dict(),'cfg':asdict(CFG),
            'dino_dim':DINO_DIM,'depth_cell_dim':DEPTH_CELL_DIM,
            'vis_patches':VIS_PATCHES,'depth_cells':DEPTH_CELLS,
            'val':best,'train_runs':TRAIN_RUN_IDS,
            'val_run':VAL_RUN_ID,'test_run':TEST_RUN_ID,
        },best_path)
    else:
        bad+=1

    if bad>=CFG.patience:
        print('Early stopping.'); break
    if hard_stop: break

TIMES['train_s']=time.time()-train_start
pd.DataFrame(history).to_csv(OUT_ROOT/'training_history.csv',index=False)

if not best_path.is_file(): raise RuntimeError('No checkpoint was produced.')
ckpt=torch.load(best_path,map_location=DEVICE)
model.load_state_dict(ckpt['model'])
print(f'BEST VAL={ckpt["val"]:.5f}')
print(f'TRAINING TIME={TIMES["train_s"]/60:.2f} min')

# %% notebook cell 8

# CELL 8 — HELD-OUT AUTOREGRESSIVE TEST
t0=time.time()

def unnorm_t(x,key):
    m=torch.as_tensor(NORM[key+'_mean'],device=x.device,dtype=x.dtype)
    s=torch.as_tensor(NORM[key+'_std'],device=x.device,dtype=x.dtype)
    return x*s+m

@torch.no_grad()
def evaluate(loader):
    model.eval()
    rows=[dict(
        force_err=[],q_err=[],dq_err=[],tau_err=[],foot_sq=[],
        contact_y=[],contact_p=[],td_y=[],td_p=[],vis_cos=[],
        body_err=[],rideaz_err=[],yawrate_err=[]
    ) for _ in range(CFG.rollout_steps)]

    for batch in loader:
        batch=to_dev(batch)
        pcur=model.encode_history(batch['leg_hist'].float())
        vis=batch['vis_ctx'].float()
        depth=batch['depth'].float()

        for h in range(CFG.rollout_steps):
            out=model.forward_step(vis,pcur,depth,batch['actions'][:,h].float())
            r=rows[h]

            pf=unnorm_t(out['force_mu'],'force')
            tf=unnorm_t(batch['force'][:,h].float(),'force')
            r['force_err'].append((pf-tf).abs().cpu().numpy().reshape(-1))

            ps=unnorm_t(out['state'],'state')
            ts=unnorm_t(batch['state'][:,h].float(),'state')
            se=(ps-ts).abs().cpu().numpy()
            r['q_err'].append(se[...,:3].reshape(-1))
            r['dq_err'].append(se[...,3:6].reshape(-1))
            r['tau_err'].append(se[...,6:9].reshape(-1))

            pfoot=unnorm_t(out['foot'],'foot')
            tfoot=unnorm_t(batch['foot'][:,h].float(),'foot')
            r['foot_sq'].append(((pfoot-tfoot)**2).sum(-1).cpu().numpy().reshape(-1))

            r['contact_y'].append(batch['contact'][:,h].cpu().numpy().reshape(-1))
            r['contact_p'].append(torch.sigmoid(out['contact_logit']).cpu().numpy().reshape(-1))
            r['td_y'].append(batch['touchdown'][:,h].cpu().numpy().reshape(-1))
            r['td_p'].append(torch.sigmoid(out['touchdown_logit']).cpu().numpy().reshape(-1))

            pv=unnorm_t(out['visual'],'vis')
            tv=unnorm_t(batch['future_vis'][:,h].float(),'vis')
            r['vis_cos'].append(F.cosine_similarity(pv,tv,dim=-1).cpu().numpy().reshape(-1))

            pb=unnorm_t(out['body'],'body')
            tb=unnorm_t(batch['body'][:,h].float(),'body')
            r['body_err'].append((pb-tb).abs().cpu().numpy().reshape(-1))

            paz=unnorm_t(out['ride_az'][:,None],'rideaz')[:,0]
            taz=unnorm_t(batch['ride_az'][:,h,None].float(),'rideaz')[:,0]
            pyaw=unnorm_t(out['yaw_rate'][:,None],'yawrate')[:,0]
            tyaw=unnorm_t(batch['yaw_rate'][:,h,None].float(),'yawrate')[:,0]
            r['rideaz_err'].append((paz-taz).abs().cpu().numpy())
            r['yawrate_err'].append((pyaw-tyaw).abs().cpu().numpy())

            vis=torch.cat([vis[:,1:],out['visual'][:,None]],1)
            pcur=out['p']

    def f1(y,p):
        y=np.concatenate(y)>0.5; p=np.concatenate(p)>0.5
        tp=np.sum(y&p); fp=np.sum((~y)&p); fn=np.sum(y&(~p))
        return float(2*tp/max(2*tp+fp+fn,1))

    out_rows=[]
    for h,r in enumerate(rows,1):
        fe=np.concatenate(r['force_err'])
        out_rows.append({
            'horizon_s':h*CFG.step_s,
            'force_component_MAE_N':float(fe.mean()),
            'force_component_p95_N':float(np.quantile(fe,.95)),
            'q_MAE_rad':float(np.concatenate(r['q_err']).mean()),
            'dq_MAE_rad_s':float(np.concatenate(r['dq_err']).mean()),
            'tau_MAE':float(np.concatenate(r['tau_err']).mean()),
            'foot_RMSE_m':float(np.sqrt(np.concatenate(r['foot_sq']).mean())),
            'contact_F1':f1(r['contact_y'],r['contact_p']),
            'touchdown_F1':f1(r['td_y'],r['td_p']),
            'DINO_cosine':float(np.concatenate(r['vis_cos']).mean()),
            'body_MAE':float(np.concatenate(r['body_err']).mean()),
            'vertical_accel_MAE_m_s2':float(np.concatenate(r['rideaz_err']).mean()),
            'heading_rate_MAE_rad_s':float(np.concatenate(r['yawrate_err']).mean()),
        })
    return pd.DataFrame(out_rows)

TEST_METRICS=evaluate(test_loader)
display(TEST_METRICS)
TEST_METRICS.to_csv(OUT_ROOT/'heldout_autoregressive_metrics.csv',index=False)

TIMES['test_s']=time.time()-t0
print(f'TEST TIME={TIMES["test_s"]:.1f}s')

# %% notebook cell 9

# CELL 9 — APPENDED FEEL-WM ROUGHNESS / YAW-VIBRATION COMPARISON
#
# Feel-WM definitions:
#   roughness     = std(vertical acceleration)
#   yaw vibration = mean(abs(heading rate)) computed from logged poses
#
# Table II, Isaac Sim "Rough":
#   Visual-only WM     6.46 m/s^2, 0.73 rad/s
#   Feel-WM w/o prop.  6.53 m/s^2, 0.77 rad/s
#   Feel-WM            6.51 m/s^2, 0.65 rad/s
#
# This cell reports both the REALIZED metric on the exact logged test interval
# and the model-predicted 0.1-s-ahead metric on the same interval.

@torch.no_grad()
def collect_h1_ride_predictions(loader):
    model.eval()
    pred_az=[]; pred_yaw=[]; origins=[]
    for batch in loader:
        b=to_dev(batch)
        pcur=model.encode_history(b['leg_hist'].float())
        out=model.forward_step(
            b['vis_ctx'].float(),pcur,b['depth'].float(),b['actions'][:,0].float()
        )
        paz=unnorm_t(out['ride_az'][:,None],'rideaz')[:,0]
        pyaw=unnorm_t(out['yaw_rate'][:,None],'yawrate')[:,0]
        pred_az.append(paz.cpu().numpy())
        pred_yaw.append(pyaw.cpu().numpy())
        origins.append(batch['origin'].numpy())
    return np.concatenate(pred_az),np.concatenate(pred_yaw),np.concatenate(origins)

PRED_AZ,PRED_YAWR,TEST_ORIGINS=collect_h1_ride_predictions(test_loader)
order=np.argsort(TEST_ORIGINS)
PRED_AZ=PRED_AZ[order]; PRED_YAWR=PRED_YAWR[order]; TEST_ORIGINS=TEST_ORIGINS[order]

pred_rough=float(np.std(PRED_AZ))
pred_yawvib=float(np.mean(np.abs(PRED_YAWR)))

d=RUNS[TEST_RUN_ID]
s0=int(TEST_ORIGINS.min())
s1=min(int(TEST_ORIGINS.max()+CFG.rollout_steps*STEP+1),len(d['time']))
true_az=d['ride_az'][s0:s1]
true_yaw=d['yaw_rate'][s0:s1]
real_rough=float(np.std(true_az))
real_yawvib=float(np.mean(np.abs(true_yaw)))

RIDE_COMPARE=pd.DataFrame([
    {'source':'Feel-WM paper / Isaac Sim Rough / Visual-only WM',
     'roughness_m_s2':6.46,'yaw_vibration_rad_s':0.73,'meaning':'paper closed-loop aggregate'},
    {'source':'Feel-WM paper / Isaac Sim Rough / Feel-WM w/o proprio',
     'roughness_m_s2':6.53,'yaw_vibration_rad_s':0.77,'meaning':'paper closed-loop aggregate'},
    {'source':'Feel-WM paper / Isaac Sim Rough / Feel-WM',
     'roughness_m_s2':6.51,'yaw_vibration_rad_s':0.65,'meaning':'paper closed-loop aggregate'},
    {'source':f'OURS measured / held-out {TEST_RUN_ID}',
     'roughness_m_s2':real_rough,'yaw_vibration_rad_s':real_yawvib,
     'meaning':'realized fixed logged test interval'},
    {'source':f'OURS predicted H=0.1s / held-out {TEST_RUN_ID}',
     'roughness_m_s2':pred_rough,'yaw_vibration_rad_s':pred_yawvib,
     'meaning':'offline model predictions on same logged interval'},
])

display(RIDE_COMPARE)
RIDE_COMPARE.to_csv(OUT_ROOT/'ride_quality_vs_feelwm_tableII.csv',index=False)

print('\nSpecific held-out sequence:',TEST_RUN_ID)
print(f'Measured roughness = {real_rough:.4f} m/s^2')
print(f'Measured yaw vibration = {real_yawvib:.4f} rad/s')
print(f'Predicted roughness = {pred_rough:.4f} m/s^2')
print(f'Predicted yaw vibration = {pred_yawvib:.4f} rad/s')
print(
    '\nDo NOT call a lower number than the paper a win: the paper values are closed-loop Isaac-Sim '
    'aggregate navigation metrics, while this is one fixed logged ANYmal/TartanGround sequence.'
)

# %% notebook cell 10

# CELL 10 — SAVE SUMMARY + EXACT RUNTIME REPORT
TIMES['total_s']=time.time()-T_GLOBAL

summary={
    'config':asdict(CFG),
    'device':str(DEVICE),
    'parameters':int(NPAR),
    'train_runs':list(TRAIN_RUN_IDS),
    'val_run':VAL_RUN_ID,
    'test_run':TEST_RUN_ID,
    'fresh_C_test_used':bool(CFG.use_fresh_c_test),
    'timings_minutes':{k:float(v/60) for k,v in TIMES.items()},
    'best_val_loss':float(best),
    'test_metrics':TEST_METRICS.to_dict(orient='records'),
    'ride_quality_reference':RIDE_COMPARE.to_dict(orient='records'),
}
(OUT_ROOT/'run_summary.json').write_text(json.dumps(summary,indent=2))

timing=pd.DataFrame([
    ['artifact audit + split',TIMES.get('audit_split_s',np.nan)],
    ['data import/load',TIMES.get('data_load_s',np.nan)],
    ['index + normalization',TIMES.get('prep_s',np.nan)],
    ['training',TIMES.get('train_s',np.nan)],
    ['testing',TIMES.get('test_s',np.nan)],
    ['TOTAL',TIMES.get('total_s',np.nan)],
],columns=['stage','seconds'])
timing['minutes']=timing.seconds/60
display(timing)
timing.to_csv(OUT_ROOT/'runtime_report.csv',index=False)

print('Saved outputs to:',OUT_ROOT)
for p in sorted(OUT_ROOT.iterdir()): print(' ',p.name)

print(f'\nTOTAL WALL TIME = {TIMES["total_s"]/60:.2f} min')
print(
    'If your Colab GPU is slower, change only max_train_windows=1600 and '
    'max_train_minutes=10.0 in CELL 1.'
)
