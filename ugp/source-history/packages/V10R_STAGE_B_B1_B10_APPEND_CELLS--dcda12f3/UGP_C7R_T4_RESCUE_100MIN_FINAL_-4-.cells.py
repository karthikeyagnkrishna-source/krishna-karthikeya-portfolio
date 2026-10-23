# Source inspection export of the adjacent notebook.
# Notebook execution order and dependencies still apply.
# IPython magic, shell lines and prose-only conclusion cells are comments.

# %% notebook cell 1
# ============================================================
# CELL 1 — MOUNT DRIVE + STRICT ARTIFACT AUDIT
# Expected: ~1-2 min. No expensive compute.
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
import json, time, random, gc, math, copy, hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import precision_recall_fscore_support
from IPython.display import display

T0 = time.time()
SEED = 1781
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

if not torch.cuda.is_available():
    raise RuntimeError('GPU is not enabled. In Colab select Runtime → Change runtime type → T4 GPU.')
print('GPU:', torch.cuda.get_device_name(0))
if 'T4' not in torch.cuda.get_device_name(0):
    print('Warning: notebook was optimized/tested conceptually for T4, but will still run on this CUDA GPU.')

torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

PROJECT_ROOT = Path('/content/drive/MyDrive/UGP_V7_FULL4B')
STRICT_ROOT  = PROJECT_ROOT / 'V10R_STRICT_STAGED'
B_ROOT       = STRICT_ROOT / 'B_dynamics_repair_v1'
C_ROOT       = STRICT_ROOT / 'stage_C_rgbd_incremental_v1'
C7R_ROOT     = C_ROOT / 'spatial_temporal_repair_v1'
C7R_FEATURE_ROOT = C7R_ROOT / 'frame_features'
C7R_MODEL_ROOT   = C7R_ROOT / 'models_fast_t4_resume'
C7R_TABLE_ROOT   = C7R_ROOT / 'tables_fast_t4_resume'
for p in [C7R_MODEL_ROOT, C7R_TABLE_ROOT]: p.mkdir(parents=True, exist_ok=True)

C_SPLIT_PATH = C_ROOT / 'C_split_manifest.csv'
B9_CKPT = B_ROOT / 'models' / 'B9_multitrajectory_dense100hz.pt'
B9_CAL_PATH = B_ROOT / 'tables' / 'B9_validation_calibration.json'
B10_MARKER = B_ROOT / 'B10_TEST_WAS_SCORED.json'
C8_MARKER = C_ROOT / 'C8_TEST_WAS_SCORED.json'
DERIVED_ROOT = STRICT_ROOT / 'derived_real_targets_v1'
CACHE_ROOT = PROJECT_ROOT / 'cache'

for p in [C_SPLIT_PATH, B9_CKPT, B9_CAL_PATH, B10_MARKER]:
    if not p.is_file(): raise FileNotFoundError(f'Missing required persistent artifact: {p}')
if C8_MARKER.exists():
    raise RuntimeError(f'Fresh Stage-C test has already been scored: {C8_MARKER}. Do not retune C7R.')

B10 = json.loads(B10_MARKER.read_text())
if not bool(B10.get('final_gate',{}).get('stage_B_ready_for_stage_C',False)):
    raise RuntimeError('Frozen B10 marker does not show Stage-B readiness.')

C_SPLIT_MANIFEST = pd.read_csv(C_SPLIT_PATH)
DEV_ROWS = C_SPLIT_MANIFEST[C_SPLIT_MANIFEST['role'].isin(['train','val'])].copy().reset_index(drop=True)
TEST_ROWS = C_SPLIT_MANIFEST[C_SPLIT_MANIFEST['role']=='test'].copy().reset_index(drop=True)
if len(DEV_ROWS) != 8 or len(TEST_ROWS) != 2:
    raise RuntimeError(f'Unexpected frozen C split: dev={len(DEV_ROWS)}, test={len(TEST_ROWS)}')

FEATURE_VERSION='dino_patch_lower60_2x3__depth_lower60_8x8x4__t3delta_v1'
missing=[]
rows=[]
for r in DEV_ROWS.itertuples(index=False):
    cache=CACHE_ROOT/f'{r.environment}_{r.run}_physics.npz'
    derived=DERIVED_ROOT/f'{r.run_id}_real_targets.npz'
    rot=B_ROOT/'derived'/f'{r.run_id}_R_odom_from_base.npy'
    feat=C7R_FEATURE_ROOT/f'{r.run_id}_spatial_frame_features.npz'
    checks={'physics':cache,'derived':derived,'rotation':rot,'C7R_feature':feat}
    for kind,p in checks.items():
        if not p.is_file(): missing.append((r.run_id,kind,str(p)))
    ver='MISSING'
    if feat.is_file():
        with np.load(feat,allow_pickle=False) as z:
            ver=str(z['feature_version'].reshape(())) if 'feature_version' in z.files else ''
        if ver != FEATURE_VERSION:
            missing.append((r.run_id,'feature_version',f'{ver!r} != {FEATURE_VERSION!r}'))
    rows.append({'run_id':r.run_id,'role':r.role,'cache':cache.is_file(),'derived':derived.is_file(),'rotation':rot.is_file(),'C7R_feature':feat.is_file(),'feature_version':ver})

AUDIT=pd.DataFrame(rows)
display(AUDIT)
print('Fresh C tests remain LOCKED:', TEST_ROWS['run_id'].tolist())
if missing:
    print('\nMISSING ARTIFACTS:')
    for x in missing: print(' ',x)
    raise RuntimeError('Persistent resume artifacts are incomplete. Stop here; do NOT waste the remaining T4 window.')
print('\nARTIFACT AUDIT: PASS — no download, DINO extraction, B9 retraining, or test scoring is required.')

# %% notebook cell 2
# ============================================================
# CELL 2 — RESTORE B9 + REBUILD ONLY THE 8 C TRAIN/VAL RUNS
# Entirely from persistent Drive caches; fresh C tests are not loaded.
# Expected: a few minutes.
# ============================================================
from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class Config:
    sample_hz: int = 100
    history_s: float = 1.0
    horizons_s: Tuple[float,...] = tuple(np.round(np.arange(.1,1.01,.1),2))
CFG=Config()

HISTORY_STEPS=int(round(CFG.history_s*CFG.sample_hz))
HISTORY_LAGS=np.unique(np.round(np.linspace(HISTORY_STEPS-1,0,10)).astype(np.int64))
B_H=len(CFG.horizons_s)
B_HORIZON_OFFSETS=np.asarray([int(round(h*CFG.sample_hz)) for h in CFG.horizons_s],np.int64)
B_MAX_FUTURE=int(B_HORIZON_OFFSETS.max())
B_DENSE_ACTION_OFFSETS=np.arange(1,B_MAX_FUTURE+1,dtype=np.int64)
B_ACTION_DIM=18
B_HISTORY_LAGS=HISTORY_LAGS
B_MODEL_DIM=320
B_ACTION_HIDDEN=160
B_HORIZON_EMBED=48
B_ACTION_GATHER=torch.as_tensor(B_HORIZON_OFFSETS-1,dtype=torch.long)
B_DEVICE=torch.device('cuda')
TAIL_NAMES=('force_peak','E_peak','tau_peak','body_peak')


def _layout(items):
    out={}; cur=0
    for name,w in items:
        out[name]=slice(cur,cur+int(w)); cur+=int(w)
    return out,cur
B_STEP_LAYOUT,B_STEP_WIDTH=_layout([
    ('q',12),('dq',12),('tau',12),('force_odom',12),('body_B',12),('E',12),('foot_base',12),('delta_position_base',3)
])

class BDenseActionEncoder(nn.Module):
    def __init__(self,action_dim=18,proj=96,hidden=160):
        super().__init__()
        self.input=nn.Sequential(nn.Linear(action_dim,proj),nn.GELU(),nn.LayerNorm(proj))
        self.gru=nn.GRU(proj,hidden,num_layers=2,batch_first=True,dropout=.08)
        self.norm=nn.LayerNorm(hidden)
    def forward(self,action_seq,gather_idx):
        z=self.input(action_seq); z,_=self.gru(z); return self.norm(z)[:,gather_idx,:]

class BResponseModel(nn.Module):
    def __init__(self,history_dim,continuous_step_dim,use_action=True,hidden=B_MODEL_DIM):
        super().__init__(); self.use_action=bool(use_action); self.H=B_H; self.D=int(continuous_step_dim)
        self.history=nn.Sequential(nn.Linear(history_dim,hidden),nn.GELU(),nn.LayerNorm(hidden),nn.Dropout(.08),nn.Linear(hidden,hidden),nn.GELU(),nn.LayerNorm(hidden),nn.Dropout(.08),nn.Linear(hidden,hidden),nn.GELU(),nn.LayerNorm(hidden))
        self.action=BDenseActionEncoder(B_ACTION_DIM,96,B_ACTION_HIDDEN)
        self.horizon_embed=nn.Parameter(torch.randn(B_H,B_HORIZON_EMBED)*.02)
        fused=hidden+B_ACTION_HIDDEN+B_HORIZON_EMBED
        self.fuse=nn.Sequential(nn.Linear(fused,hidden),nn.GELU(),nn.LayerNorm(hidden),nn.Dropout(.06),nn.Linear(hidden,hidden),nn.GELU(),nn.LayerNorm(hidden))
        self.mean_head=nn.Linear(hidden,self.D); self.log_scale_head=nn.Linear(hidden,self.D)
        self.contact_head=nn.Linear(hidden,4); self.touchdown_head=nn.Linear(hidden,4)
        self.impulse_mean_head=nn.Linear(hidden,12); self.impulse_log_scale_head=nn.Linear(hidden,12)
        self.tail_q90_head=nn.Sequential(nn.Linear(hidden+B_ACTION_HIDDEN,hidden//2),nn.GELU(),nn.Linear(hidden//2,4))
    def forward(self,history_flat,action_dense):
        n=len(history_flat); h0=self.history(history_flat)
        if self.use_action:
            ah=self.action(action_dense,B_ACTION_GATHER.to(action_dense.device)); afinal=ah[:,-1]
        else:
            ah=torch.zeros(n,B_H,B_ACTION_HIDDEN,device=history_flat.device,dtype=history_flat.dtype); afinal=torch.zeros(n,B_ACTION_HIDDEN,device=history_flat.device,dtype=history_flat.dtype)
        h=h0[:,None,:].expand(-1,B_H,-1); he=self.horizon_embed[None].expand(n,-1,-1); z=self.fuse(torch.cat([h,ah,he],-1))
        return {'mean_z':self.mean_head(z),'log_scale_z':torch.clamp(self.log_scale_head(z),-5.5,2.8),'contact_logits':self.contact_head(z),'touchdown_logits':self.touchdown_head(z),'impulse_mean_z':self.impulse_mean_head(z).reshape(n,B_H,4,3),'impulse_log_scale_z':torch.clamp(self.impulse_log_scale_head(z).reshape(n,B_H,4,3),-5.5,3.0),'tail_q90_log':self.tail_q90_head(torch.cat([h0,afinal],-1))}

ck=torch.load(B9_CKPT,map_location='cpu',weights_only=False)
B9_PREP=ck['prep']
B9_MODEL=BResponseModel(int(ck['history_dim']),int(ck['step_width']),bool(ck['use_action']))
B9_MODEL.load_state_dict(ck['state_dict'],strict=True); B9_MODEL.eval()
for p in B9_MODEL.parameters(): p.requires_grad_(False)
B9_DENSE={'model':B9_MODEL,'checkpoint':str(B9_CKPT),'prep':B9_PREP,'use_action':True}
B9_CALIBRATION=json.loads(B9_CAL_PATH.read_text())
print('Restored frozen B9 checkpoint:',B9_CKPT)


def _b_reshape_cont(y): return np.asarray(y,np.float32).reshape(-1,B_H,B_STEP_WIDTH)
def _b_group(step,name): return step[:,:,B_STEP_LAYOUT[name]]
def _b_tail_raw(step,body_med,body_mad):
    n=len(step); force=_b_group(step,'force_odom').reshape(n,B_H,4,3); E=_b_group(step,'E').reshape(n,B_H,4,3); tau=_b_group(step,'tau').reshape(n,B_H,12); body=_b_group(step,'body_B').reshape(n,B_H,12)
    bz=np.abs((body-body_med[None,None,:])/np.maximum(1.4826*body_mad[None,None,:],1e-5))
    return np.stack([np.max(np.linalg.norm(force,axis=-1),axis=(1,2)),np.max(np.linalg.norm(E,axis=-1),axis=(1,2)),np.max(np.abs(tau),axis=(1,2)),np.max(bz,axis=(1,2))],axis=1).astype(np.float32)

def build_run(row):
    cache=CACHE_ROOT/f'{row.environment}_{row.run}_physics.npz'; derived=DERIVED_ROOT/f'{row.run_id}_real_targets.npz'; rot=B_ROOT/'derived'/f'{row.run_id}_R_odom_from_base.npy'
    with np.load(cache,allow_pickle=False) as z:
        time_s=np.asarray(z['time'],np.float64); P=np.asarray(z['P'],np.float32); B=np.asarray(z['B'],np.float32); U=np.asarray(z['U'],np.float32); L=np.asarray(z['L_LEG'],np.float32); E=np.asarray(z['E'],np.float32)
    with np.load(derived,allow_pickle=False) as d:
        foot=np.asarray(d['foot_base'],np.float32); force=np.asarray(d['force_odom_from_anymal_state'],np.float32); contact=np.asarray(d['contact_from_anymal_state'],np.uint8).astype(bool); td=np.asarray(d['touchdown_event'],np.uint8).astype(bool); imp=np.asarray(d['touchdown_impulse_odom'],np.float32); base_pos=np.asarray(d['base_position_odom'],np.float32)
    R_ob=np.asarray(np.load(rot),np.float32); n=len(time_s)
    if any(len(x)!=n for x in [P,B,U,L,E,foot,force,contact,td,imp,base_pos,R_ob]): raise RuntimeError(f'{row.run_id}: persistent length mismatch')
    origins=np.arange(int(B_HISTORY_LAGS.max()),n-B_MAX_FUTURE,dtype=np.int64)
    hist_idx=origins[:,None]-B_HISTORY_LAGS[None,:]; fut_idx=origins[:,None]+B_HORIZON_OFFSETS[None,:]; dense_idx=origins[:,None]+B_DENSE_ACTION_OFFSETS[None,:]
    P_hist=np.concatenate([P[:,:,:9],force],axis=-1)
    hist_signal=np.concatenate([P_hist.reshape(n,-1),B.reshape(n,-1),E.reshape(n,-1),contact.astype(np.float32),U.reshape(n,-1),L.reshape(n,-1),foot.reshape(n,-1)],axis=1).astype(np.float32)
    X=hist_signal[hist_idx].reshape(len(origins),-1)
    A=np.concatenate([U,L.reshape(n,-1)],axis=1).astype(np.float32)[dense_idx]
    q=P[fut_idx,:,:3].reshape(len(origins),B_H,12); dq=P[fut_idx,:,3:6].reshape(len(origins),B_H,12); tau=P[fut_idx,:,6:9].reshape(len(origins),B_H,12); ff=force[fut_idx].reshape(len(origins),B_H,12); body=B[fut_idx].reshape(len(origins),B_H,12); ee=E[fut_idx].reshape(len(origins),B_H,12); ft=foot[fut_idx].reshape(len(origins),B_H,12)
    delta_o=base_pos[fut_idx]-base_pos[origins][:,None,:]; R_bo=np.transpose(R_ob[origins],(0,2,1)); delta_b=np.einsum('nij,nhj->nhi',R_bo,delta_o).astype(np.float32)
    Y=np.concatenate([q,dq,tau,ff,body,ee,ft,delta_b],axis=-1).astype(np.float32).reshape(len(origins),-1)
    split=np.full(len(origins),str(row.role),dtype='<U8')
    return {'terrain':str(row.terrain),'time':time_s,'origins':origins,'X_core':X.astype(np.float32),'X_action_dense':A.astype(np.float32),'Y_cont':Y,'Y_contact':contact[fut_idx].astype(np.float32),'Y_touchdown':td[fut_idx].astype(np.float32),'Y_impulse':imp[fut_idx].astype(np.float32),'split':split,'derived_path':str(derived),'cache_path':str(cache)}

C7R_FRAME_FEATURES={}; C7R_DATA={}; C_MAX_VISUAL_AGE_S=.25
for row in DEV_ROWS.itertuples(index=False):
    d=build_run(row)
    fp=C7R_FEATURE_ROOT/f'{row.run_id}_spatial_frame_features.npz'
    with np.load(fp,allow_pickle=False) as z:
        tab={k:np.asarray(z[k]) for k in z.files}
    C7R_FRAME_FEATURES[str(row.run_id)]=tab
    ct=np.asarray(tab['camera_times'],np.float64); ot=d['time'][d['origins']]
    vi=np.searchsorted(ct,ot,side='right')-1; valid=vi>=0; age=np.full(len(vi),np.inf); age[valid]=ot[valid]-ct[vi[valid]]; valid &= (age>=-1e-9)&(age<=C_MAX_VISUAL_AGE_S)
    if valid.mean()<.90: raise RuntimeError(f'{row.run_id}: causal visual coverage {valid.mean():.2%}')
    n0=len(d['origins']); d2=dict(d)
    for k,v in d.items():
        if isinstance(v,np.ndarray) and len(v)==n0: d2[k]=v[valid]
    d2['visual_frame_index']=vi[valid].astype(np.int64); d2['visual_age_s']=age[valid].astype(np.float32)
    C7R_DATA[str(row.run_id)]=d2

C_HISTORY_QUIET_S=.20; C_MIN_EVENT_LEAD_S=.20
def compute_precontact(d):
    with np.load(d['cache_path'],allow_pickle=False) as z: Es=np.asarray(z['E'],np.float32)
    with np.load(d['derived_path'],allow_pickle=False) as z: Fs=np.asarray(z['force_odom_from_anymal_state'],np.float32)
    fscalar=np.max(np.linalg.norm(Fs,axis=-1),axis=1); escalar=np.max(np.linalg.norm(Es,axis=-1),axis=1); hist_n=max(2,int(round(C_HISTORY_QUIET_S*CFG.sample_hz)))
    quiet=np.zeros(len(d['origins']),bool)
    for j,o in enumerate(d['origins']):
        lo=max(0,int(o)-hist_n+1); quiet[j]=(np.max(fscalar[lo:int(o)+1])<=B9_PREP['force95'] and np.max(escalar[lo:int(o)+1])<=B9_PREP['E95'])
    step=_b_reshape_cont(d['Y_cont']); f=_b_group(step,'force_odom').reshape(-1,B_H,4,3); e=_b_group(step,'E').reshape(-1,B_H,4,3)
    bad=(np.max(np.linalg.norm(f,axis=-1),axis=2)>B9_PREP['force95']) | (np.max(np.linalg.norm(e,axis=-1),axis=2)>B9_PREP['E95'])
    imp=np.linalg.norm(d['Y_impulse'],axis=-1); bad |= np.max((d['Y_touchdown'].astype(bool)&(imp>float(B9_CALIBRATION['risk_reference']['touchdown_impulse_norm_p95_Ns']))),axis=2)
    has=bad.any(1); first=np.full(len(has),B_H+1,np.int32); first[has]=np.argmax(bad[has],axis=1); hs=np.asarray(CFG.horizons_s,np.float32); lead=np.full(len(has),np.nan,np.float32); lead[has]=hs[first[has]]
    d['C_precontact']=(quiet&has&(lead>=C_MIN_EVENT_LEAD_S-1e-6)); d['C_event_lead_s']=lead; return d

for rid in list(C7R_DATA): C7R_DATA[rid]=compute_precontact(C7R_DATA[rid])
summary=[]
for rid,d in C7R_DATA.items(): summary.append({'run_id':rid,'terrain':d['terrain'],'role':str(np.unique(d['split'])[0]),'origins':len(d['origins']),'precontact':int(d['C_precontact'].sum()),'visual_age_p95_s':float(np.quantile(d['visual_age_s'],.95))})
display(pd.DataFrame(summary))
print('C7R train/val dynamics restore: PASS')
print('Fresh tests were never loaded:', TEST_ROWS['run_id'].tolist())

# %% notebook cell 3
# ============================================================
# CELL 3 — VECTORIZED C7R TRAINING / RESUME ON T4
# Same frozen B9 model, targets, loss, feature version and gate protocol.
# Speed changes only: batched visual lookup + AMP + epoch checkpoints.
# ============================================================
C7R_LOWER_START=.40
C7R_TRAIN_BATCH=96               # optimizer micro-batch preserved from approved C7R protocol
C7R_FETCH_BATCH=384               # CPU/vectorized gather batch; split into 96-sample optimizer steps
C7R_EPOCHS=20
C7R_PATIENCE=5
C7R_LR=2e-4
C7R_WEIGHT_DECAY=1e-4
C7R_RESIDUAL_L2=2e-3
C_VIS_HIDDEN=128
USE_AMP=True                     # T4 tensor-core acceleration; model/loss/data unchanged


def c7r_origin_visual(rid,d,positions):
    positions=np.asarray(positions,np.int64); tab=C7R_FRAME_FEATURES[rid]; vi=np.asarray(d['visual_frame_index'][positions],np.int64); v1=np.maximum(vi-1,0); v2=np.maximum(vi-2,0)
    R=np.asarray(tab['rgb_spatial'],np.float32); D=np.asarray(tab['depth_spatial'],np.float32)
    r0,r1,r2=R[vi],R[v1],R[v2]; d0,d1,d2=D[vi],D[v1],D[v2]
    return np.concatenate([r0,r0-r1,r1-r2],1).astype(np.float32),np.concatenate([d0,d0-d1,d1-d2],1).astype(np.float32)

# Train-only visual normalization, vectorized in chunks.
def fit_vis_norm():
    sr=sr2=sd=sd2=None; n=0
    for rid,d in C7R_DATA.items():
        pos=np.flatnonzero(d['split']=='train')
        for st in range(0,len(pos),512):
            R,D=c7r_origin_visual(rid,d,pos[st:st+512])
            if sr is None:
                sr=np.zeros(R.shape[1],np.float64); sr2=np.zeros_like(sr); sd=np.zeros(D.shape[1],np.float64); sd2=np.zeros_like(sd)
            sr+=R.sum(0,dtype=np.float64); sr2+=np.square(R,dtype=np.float64).sum(0,dtype=np.float64); sd+=D.sum(0,dtype=np.float64); sd2+=np.square(D,dtype=np.float64).sum(0,dtype=np.float64); n+=len(R)
    rm=sr/n; dm=sd/n; rs=np.sqrt(np.maximum(sr2/n-rm*rm,0)); ds=np.sqrt(np.maximum(sd2/n-dm*dm,0)); rs[rs<1e-5]=1.; ds[ds<1e-5]=1.
    return {'rgb_mean':rm.astype(np.float32),'rgb_std':rs.astype(np.float32),'depth_mean':dm.astype(np.float32),'depth_std':ds.astype(np.float32),'train_count':int(n)}

VIS_NORM_PATH=C7R_ROOT/'fast_t4_visual_norm.npz'
if VIS_NORM_PATH.exists():
    with np.load(VIS_NORM_PATH,allow_pickle=False) as z: C7R_VIS_NORM={k:z[k] for k in z.files}; C7R_VIS_NORM['train_count']=int(np.asarray(C7R_VIS_NORM['train_count']).reshape(()))
    print('Reused visual normalization:',VIS_NORM_PATH)
else:
    C7R_VIS_NORM=fit_vis_norm(); np.savez_compressed(VIS_NORM_PATH,**C7R_VIS_NORM); print('Wrote visual normalization:',VIS_NORM_PATH)
print({'train_count':C7R_VIS_NORM['train_count'],'rgb_dim':len(C7R_VIS_NORM['rgb_mean']),'depth_dim':len(C7R_VIS_NORM['depth_mean'])})

NORM_T={k:torch.as_tensor(v,dtype=torch.float32,device=B_DEVICE) for k,v in {'x_mean':B9_PREP['x_mean'],'x_std':B9_PREP['x_std'],'action_mean':B9_PREP['action_mean'],'action_std':B9_PREP['action_std'],'y_mean':B9_PREP['y_mean'],'y_std':B9_PREP['y_std'],'imp_mean':B9_PREP['imp_mean'],'imp_std':B9_PREP['imp_std'],'rgb_mean':C7R_VIS_NORM['rgb_mean'],'rgb_std':C7R_VIS_NORM['rgb_std'],'depth_mean':C7R_VIS_NORM['depth_mean'],'depth_std':C7R_VIS_NORM['depth_std']}.items()}

class IndexDataset(Dataset):
    def __init__(self,data_dict,split,balanced=False):
        self.data_dict=data_dict; self.run_keys=[]; ro=[]; lo=[]; ftp=[]; etp=[]; tlp=[]; swp=[]; rec=[]; tc={}
        for rid,d in data_dict.items():
            idx=np.flatnonzero(d['split']==split).astype(np.int64)
            if len(idx): tc[d['terrain']]=tc.get(d['terrain'],0)+len(idx); rec.append((rid,d,idx,d['terrain']))
        total=sum(len(x[2]) for x in rec); tf={t:total/(len(tc)*c) for t,c in tc.items()}
        for rix,(rid,d,idx,terrain) in enumerate(rec):
            self.run_keys.append(rid); ro.append(np.full(len(idx),rix,np.int16)); lo.append(idx)
            step=np.asarray(d['Y_cont'][idx],np.float32).reshape(-1,B_H,B_STEP_WIDTH); td=np.asarray(d['Y_touchdown'][idx],np.float32); force=step[:,:,B_STEP_LAYOUT['force_odom']].reshape(-1,B_H,4,3); E=step[:,:,B_STEP_LAYOUT['E']].reshape(-1,B_H,4,3)
            ft=(np.linalg.norm(force,axis=-1)>B9_PREP['force95']); et=(np.linalg.norm(E,axis=-1)>B9_PREP['E95']); tail=np.log1p(np.maximum(_b_tail_raw(step,B9_PREP['body_med'],B9_PREP['body_mad']),0)).astype(np.float32)
            sw=1+2*ft.max((1,2)).astype(np.float32)+2*et.max((1,2)).astype(np.float32)+4*td.max((1,2)).astype(np.float32)
            if balanced: sw*=float(tf[terrain])
            ftp.append(ft.astype(np.uint8)); etp.append(et.astype(np.uint8)); tlp.append(tail); swp.append(np.clip(sw,1,12).astype(np.float32))
        self.run_of=np.concatenate(ro); self.local_of=np.concatenate(lo); self.force_tail=np.concatenate(ftp); self.E_tail=np.concatenate(etp); self.tail_log=np.concatenate(tlp); self.sample_weight=np.concatenate(swp)
    def __len__(self): return len(self.local_of)
    def __getitem__(self,i): return int(i)

# Batch collate: one vectorized NumPy gather per run, NOT one visual lookup per sample.
def make_collate(ds):
    rgb_dim=len(C7R_VIS_NORM['rgb_mean']); dep_dim=len(C7R_VIS_NORM['depth_mean'])
    def collate(items):
        gi=np.asarray(items,np.int64); Bn=len(gi); X=np.empty((Bn,1060),np.float32); A=np.empty((Bn,100,18),np.float32); Y=np.empty((Bn,B_H,B_STEP_WIDTH),np.float32); C=np.empty((Bn,B_H,4),np.float32); TD=np.empty_like(C); IMP=np.empty((Bn,B_H,4,3),np.float32); R=np.empty((Bn,rgb_dim),np.float32); D=np.empty((Bn,dep_dim),np.float32)
        runix=ds.run_of[gi]; local=ds.local_of[gi]
        for rix in np.unique(runix):
            outpos=np.flatnonzero(runix==rix); pos=local[outpos]; rid=ds.run_keys[int(rix)]; d=ds.data_dict[rid]
            X[outpos]=d['X_core'][pos]; A[outpos]=d['X_action_dense'][pos]; Y[outpos]=d['Y_cont'][pos].reshape(-1,B_H,B_STEP_WIDTH); C[outpos]=d['Y_contact'][pos]; TD[outpos]=d['Y_touchdown'][pos]; IMP[outpos]=d['Y_impulse'][pos]; rr,dd=c7r_origin_visual(rid,d,pos); R[outpos]=rr; D[outpos]=dd
        vals=[X,A,Y,C,TD,IMP,ds.force_tail[gi].astype(np.float32),ds.E_tail[gi].astype(np.float32),ds.tail_log[gi],R,D]
        return tuple(torch.from_numpy(v) for v in vals)
    return collate

def prep_batch(raw):
    raw=tuple(t.to(B_DEVICE,non_blocking=True) for t in raw); x,a,y,c,td,imp,ft,et,tail,rgb,dep=raw
    x=(x-NORM_T['x_mean'][None,:])/NORM_T['x_std'][None,:]; a=(a-NORM_T['action_mean'][None,None,:])/NORM_T['action_std'][None,None,:]; y=(y-NORM_T['y_mean'][None,:,:])/NORM_T['y_std'][None,:,:]; imp=(imp-NORM_T['imp_mean'][None,None,None,:])/NORM_T['imp_std'][None,None,None,:]; rgb=(rgb-NORM_T['rgb_mean'][None,:])/NORM_T['rgb_std'][None,:]; dep=(dep-NORM_T['depth_mean'][None,:])/NORM_T['depth_std'][None,:]
    return (x,a,y,c,td,imp,ft,et,tail),rgb,dep

def pinball(pred,target,q=.90):
    e=target-pred; return torch.maximum(q*e,(q-1)*e)
def b_loss(out,batch):
    x,a,y,contact,td,imp,f_tail,e_tail,tail=batch; mu=out['mean_z']; ls=out['log_scale_z']; nll=(.5*(y-mu).square()*torch.exp(-2*ls)+ls).mean(); fs=B_STEP_LAYOUT['force_odom']; es=B_STEP_LAYOUT['E']; fp=mu[:,:,fs].reshape(-1,B_H,4,3); ft=y[:,:,fs].reshape(-1,B_H,4,3); ep=mu[:,:,es].reshape(-1,B_H,4,3); et=y[:,:,es].reshape(-1,B_H,4,3)
    fl=(F.smooth_l1_loss(fp,ft,reduction='none',beta=.75)*(1+4*f_tail[...,None])).mean(); el=(F.smooth_l1_loss(ep,et,reduction='none',beta=.75)*(1+4*e_tail[...,None])).mean(); cl=F.binary_cross_entropy_with_logits(out['contact_logits'],contact); pw=torch.as_tensor(B9_PREP['td_pos_weight'],device=td.device,dtype=td.dtype); tdl=F.binary_cross_entropy_with_logits(out['touchdown_logits'],td,pos_weight=pw); imn=(.5*(imp-out['impulse_mean_z']).square()*torch.exp(-2*out['impulse_log_scale_z'])+out['impulse_log_scale_z']).mean(-1); mask=td>.5; il=imn[mask].mean() if torch.any(mask) else imn.sum()*0.; tl=pinball(out['tail_q90_log'],tail).mean(); return nll+.45*fl+.35*el+.20*cl+.45*tdl+.55*il+.15*tl

class CVisualResidualModel(nn.Module):
    def __init__(self,base,rgb_dim,depth_dim,mode):
        super().__init__(); self.mode=mode; self.base=base.eval();
        for p in self.base.parameters(): p.requires_grad_(False)
        self.rgb=nn.Sequential(nn.Linear(rgb_dim,C_VIS_HIDDEN),nn.GELU(),nn.LayerNorm(C_VIS_HIDDEN),nn.Linear(C_VIS_HIDDEN,C_VIS_HIDDEN),nn.GELU(),nn.LayerNorm(C_VIS_HIDDEN)); self.depth=nn.Sequential(nn.Linear(depth_dim,C_VIS_HIDDEN),nn.GELU(),nn.LayerNorm(C_VIS_HIDDEN),nn.Linear(C_VIS_HIDDEN,C_VIS_HIDDEN),nn.GELU(),nn.LayerNorm(C_VIS_HIDDEN)); vd=C_VIS_HIDDEN if mode in {'rgb','depth'} else 2*C_VIS_HIDDEN
        self.h_adapter=nn.Sequential(nn.Linear(B_MODEL_DIM+vd,B_MODEL_DIM),nn.GELU(),nn.LayerNorm(B_MODEL_DIM),nn.Dropout(.05),nn.Linear(B_MODEL_DIM,B_MODEL_DIM),nn.GELU(),nn.LayerNorm(B_MODEL_DIM)); self.seq_adapter=nn.Sequential(nn.Linear(B_MODEL_DIM+B_ACTION_HIDDEN+vd,B_MODEL_DIM),nn.GELU(),nn.LayerNorm(B_MODEL_DIM)); self.d_mean=nn.Linear(B_MODEL_DIM,B_STEP_WIDTH); self.d_log_scale=nn.Linear(B_MODEL_DIM,B_STEP_WIDTH); self.d_contact=nn.Linear(B_MODEL_DIM,4); self.d_touchdown=nn.Linear(B_MODEL_DIM,4); self.d_impulse_mean=nn.Linear(B_MODEL_DIM,12); self.d_impulse_log_scale=nn.Linear(B_MODEL_DIM,12); self.d_tail=nn.Linear(B_MODEL_DIM,4)
        for layer in [self.d_mean,self.d_log_scale,self.d_contact,self.d_touchdown,self.d_impulse_mean,self.d_impulse_log_scale,self.d_tail]: nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)
    def _v(self,r,d): return self.rgb(r) if self.mode=='rgb' else self.depth(d) if self.mode=='depth' else torch.cat([self.rgb(r),self.depth(d)],-1)
    def forward(self,x,a,r,d):
        self.base.eval()
        with torch.no_grad():
            h0=self.base.history(x); ah=self.base.action(a,B_ACTION_GATHER.to(a.device)); afinal=ah[:,-1]; h=h0[:,None,:].expand(-1,B_H,-1); he=self.base.horizon_embed[None].expand(len(x),-1,-1); z=self.base.fuse(torch.cat([h,ah,he],-1)); base={'mean_z':self.base.mean_head(z),'log_scale_z':torch.clamp(self.base.log_scale_head(z),-5.5,2.8),'contact_logits':self.base.contact_head(z),'touchdown_logits':self.base.touchdown_head(z),'impulse_mean_z':self.base.impulse_mean_head(z).reshape(-1,B_H,4,3),'impulse_log_scale_z':torch.clamp(self.base.impulse_log_scale_head(z).reshape(-1,B_H,4,3),-5.5,3.0),'tail_q90_log':self.base.tail_q90_head(torch.cat([h0,afinal],-1))}
        v=self._v(r,d); rz=self.h_adapter(torch.cat([z,v[:,None,:].expand(-1,B_H,-1)],-1)); seq=self.seq_adapter(torch.cat([h0,afinal,v],-1)); out={'mean_z':base['mean_z']+self.d_mean(rz),'log_scale_z':torch.clamp(base['log_scale_z']+self.d_log_scale(rz),-5.5,2.8),'contact_logits':base['contact_logits']+self.d_contact(rz),'touchdown_logits':base['touchdown_logits']+self.d_touchdown(rz),'impulse_mean_z':base['impulse_mean_z']+self.d_impulse_mean(rz).reshape(-1,B_H,4,3),'impulse_log_scale_z':torch.clamp(base['impulse_log_scale_z']+self.d_impulse_log_scale(rz).reshape(-1,B_H,4,3),-5.5,3.0),'tail_q90_log':base['tail_q90_log']+self.d_tail(seq)}
        out['_residual_l2']=self.d_mean(rz).square().mean()+0.25*self.d_impulse_mean(rz).reshape(-1,B_H,4,3).square().mean()+0.25*self.d_tail(seq).square().mean(); return out

def fresh_base():
    c=torch.load(B9_CKPT,map_location='cpu',weights_only=False); m=BResponseModel(int(c['history_dim']),int(c['step_width']),bool(c['use_action'])); m.load_state_dict(c['state_dict']); m.eval(); return m

def train_mode(mode):
    final=C7R_MODEL_ROOT/f'C7R_{mode}_adapter.pt'; resume=C7R_MODEL_ROOT/f'C7R_{mode}_resume.pt'
    if final.exists():
        s=torch.load(final,map_location='cpu',weights_only=False)
        if s.get('feature_version')==FEATURE_VERSION:
            print(f'C7R-{mode}: reusing completed checkpoint {final}'); return {'mode':mode,'checkpoint':str(final),'best_val_loss':float(s['best_val_loss'])}
    tr=IndexDataset(C7R_DATA,'train',True); va=IndexDataset(C7R_DATA,'val',False); sampler=WeightedRandomSampler(torch.as_tensor(tr.sample_weight,dtype=torch.double),len(tr),replacement=True); trl=DataLoader(tr,batch_size=C7R_FETCH_BATCH,sampler=sampler,num_workers=0,pin_memory=True,collate_fn=make_collate(tr)); val=DataLoader(va,batch_size=C7R_FETCH_BATCH,shuffle=False,num_workers=0,pin_memory=True,collate_fn=make_collate(va))
    model=CVisualResidualModel(fresh_base(),len(C7R_VIS_NORM['rgb_mean']),len(C7R_VIS_NORM['depth_mean']),mode).to(B_DEVICE); params=[p for p in model.parameters() if p.requires_grad]; opt=torch.optim.AdamW(params,lr=C7R_LR,weight_decay=C7R_WEIGHT_DECAY); scaler=torch.cuda.amp.GradScaler(enabled=USE_AMP); best=float('inf'); best_state=None; bad=0; hist=[]; start=1
    if resume.exists():
        rr=torch.load(resume,map_location='cpu',weights_only=False)
        if rr.get('feature_version')==FEATURE_VERSION and rr.get('mode')==mode:
            miss,unexp=model.load_state_dict(rr['adapter_state_dict'],strict=False); opt.load_state_dict(rr['optimizer_state']); best=float(rr['best']); best_state=rr['best_state']; bad=int(rr['bad']); hist=list(rr['history']); start=int(rr['epoch'])+1; print(f'C7R-{mode}: resuming at epoch {start}')
    @torch.no_grad()
    def eval_loss():
        model.eval(); total=n=0
        for raw in val:
            bb,r,d=prep_batch(raw)
            with torch.autocast('cuda',dtype=torch.float16,enabled=USE_AMP): out=model(bb[0],bb[1],r,d); loss=b_loss(out,bb)+C7R_RESIDUAL_L2*out['_residual_l2']
            total+=float(loss)*len(bb[0]); n+=len(bb[0])
        return total/max(n,1)
    for ep in range(start,C7R_EPOCHS+1):
        tic=time.time(); model.train(); total=n=0
        for raw_big in trl:
            # Vectorized CPU gather happens once for up to 384 samples.
            # Optimizer still steps on the original 96-sample micro-batches.
            for off in range(0,len(raw_big[0]),C7R_TRAIN_BATCH):
                raw=tuple(t[off:off+C7R_TRAIN_BATCH] for t in raw_big)
                bb,r,d=prep_batch(raw); opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16,enabled=USE_AMP): out=model(bb[0],bb[1],r,d); loss=b_loss(out,bb)+C7R_RESIDUAL_L2*out['_residual_l2']
                scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params,5.); scaler.step(opt); scaler.update(); total+=float(loss.detach())*len(bb[0]); n+=len(bb[0])
        tl=total/max(n,1); vl=eval_loss(); hist.append({'mode':mode,'epoch':ep,'train_loss':tl,'val_loss':vl,'seconds':time.time()-tic}); print(f'C7R-{mode:5s} ep={ep:02d} train={tl:.5f} val={vl:.5f} time={time.time()-tic:.1f}s total={(time.time()-T0)/60:.1f}m')
        if vl<best-1e-4: best=vl; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith('base.')}; bad=0
        else: bad+=1
        current={k:v.detach().cpu() for k,v in model.state_dict().items() if not k.startswith('base.')}; torch.save({'mode':mode,'feature_version':FEATURE_VERSION,'epoch':ep,'adapter_state_dict':current,'optimizer_state':opt.state_dict(),'best':best,'best_state':best_state,'bad':bad,'history':hist},resume)
        if bad>=C7R_PATIENCE: print(f'C7R-{mode}: early stop'); break
    if best_state is None: raise RuntimeError(f'C7R-{mode}: no best state')
    torch.save({'mode':mode,'adapter_state_dict':best_state,'best_val_loss':float(best),'vis_norm':C7R_VIS_NORM,'feature_version':FEATURE_VERSION,'B9_checkpoint':str(B9_CKPT),'uses_test':False,'training_history':hist,'amp_t4':USE_AMP},final)
    pd.DataFrame(hist).to_csv(C7R_TABLE_ROOT/f'C7R_{mode}_history.csv',index=False)
    del model,tr,va,trl,val,sampler,opt,params; gc.collect(); torch.cuda.empty_cache(); return {'mode':mode,'checkpoint':str(final),'best_val_loss':float(best)}

# Keep baseline model on CPU while each adapter carries its own frozen copy.
B9_DENSE['model'].to('cpu')
C7R_MODELS={}
for mode in ['rgb','depth','rgbd']:
    print('\n'+'='*80+'\nC7R MODE: '+mode+'\n'+'='*80); C7R_MODELS[mode]=train_mode(mode)
print('\nTRAINING COMPLETE / RESTORED')
display(pd.DataFrame([{'mode':k,'best_val_loss':v['best_val_loss'],'checkpoint':v['checkpoint']} for k,v in C7R_MODELS.items()]))

# %% notebook cell 4
# ============================================================
# CELL 4 — VALIDATION CALIBRATION + MODE SELECTION + ANTI-CHEATING GATE
# Uses validation trajectories only. Fresh C tests remain locked.
# ============================================================
from scipy.stats import spearmanr


def _b_ratio(x,ref):
    r=np.maximum(np.asarray(x,float),0)/max(float(ref),1e-8); return r/(1+r)
def actual_risk(step,contact,touchdown,impulse,prep,cal):
    step=np.asarray(step,float); n=len(step); refs=cal['risk_reference']; dt=np.diff(np.asarray(CFG.horizons_s,float)); f=_b_group(step,'force_odom').reshape(n,B_H,4,3); tau=_b_group(step,'tau').reshape(n,B_H,12); body=_b_group(step,'body_B').reshape(n,B_H,12); E=_b_group(step,'E').reshape(n,B_H,4,3); foot=_b_group(step,'foot_base').reshape(n,B_H,4,3)
    vals={'force':np.max(np.linalg.norm(f,axis=-1),axis=(1,2)),'torque':np.max(np.abs(tau),axis=(1,2)),'body':np.max(np.abs((body-prep['body_med'][None,None,:])/np.maximum(1.4826*prep['body_mad'][None,None,:],1e-5)),axis=(1,2)),'E':np.max(np.linalg.norm(E,axis=-1),axis=(1,2)),'foot':np.max(np.linalg.norm(np.diff(foot,axis=1)/dt[None,:,None,None],axis=-1),axis=(1,2)),'impulse':np.max(np.linalg.norm(impulse,axis=-1)*touchdown,axis=(1,2))}
    support=np.sum(contact,axis=-1); vals['support']=np.max(np.clip((2-support)/2,0,1),axis=1); comp={'force':_b_ratio(vals['force'],refs['force_norm_p95_N']),'torque':_b_ratio(vals['torque'],refs['abs_tau_p95_Nm']),'body':_b_ratio(vals['body'],refs['body_robust_z_p95']),'E':_b_ratio(vals['E'],refs['E_norm_p95']),'foot':_b_ratio(vals['foot'],refs['foot_speed_p95_mps']),'impulse':_b_ratio(vals['impulse'],refs['touchdown_impulse_norm_p95_Ns']),'support':vals['support']}; physical=.20*comp['force']+.14*comp['torque']+.14*comp['body']+.14*comp['E']+.10*comp['foot']+.18*comp['impulse']+.10*comp['support']; return {**comp,'physical_risk':np.clip(physical,0,1)}
def predicted_risk(pred,prep,cal):
    mu=np.asarray(pred['continuous_mean'],float); scale=np.asarray(pred['continuous_scale'],float); n=len(mu); refs=cal['risk_reference']; z90=1.645; dt=np.diff(np.asarray(CFG.horizons_s,float)); gs=lambda g:_b_group(scale,g)*float(cal['scale_factor'][g]); fmu=_b_group(mu,'force_odom').reshape(n,B_H,4,3); fsc=gs('force_odom').reshape(n,B_H,4,3); Emu=_b_group(mu,'E').reshape(n,B_H,4,3); Esc=gs('E').reshape(n,B_H,4,3); tmu=_b_group(mu,'tau').reshape(n,B_H,12); tsc=gs('tau').reshape(n,B_H,12); bmu=_b_group(mu,'body_B').reshape(n,B_H,12); bsc=gs('body_B').reshape(n,B_H,12); foot=_b_group(mu,'foot_base').reshape(n,B_H,4,3)
    fup=np.max(np.linalg.norm(fmu,axis=-1)+z90*np.linalg.norm(fsc,axis=-1),axis=(1,2)); Eup=np.max(np.linalg.norm(Emu,axis=-1)+z90*np.linalg.norm(Esc,axis=-1),axis=(1,2)); tup=np.max(np.abs(tmu)+z90*tsc,axis=(1,2)); bup=np.max((np.abs(bmu-prep['body_med'][None,None,:])+z90*bsc)/np.maximum(1.4826*prep['body_mad'][None,None,:],1e-5),axis=(1,2)); q=np.asarray(pred['tail_q90'],float).copy()
    for j,name in enumerate(TAIL_NAMES): q[:,j]+=cal['tail_add'][name]
    ft=np.maximum(fup,q[:,0]); Et=np.maximum(Eup,q[:,1]); tt=np.maximum(tup,q[:,2]); bt=np.maximum(bup,q[:,3]); footp=np.max(np.linalg.norm(np.diff(foot,axis=1)/dt[None,:,None,None],axis=-1),axis=(1,2)); iup=np.linalg.norm(pred['impulse_mean'],axis=-1)+z90*float(cal['impulse_scale_factor'])*np.linalg.norm(pred['impulse_scale'],axis=-1); iv=np.max(pred['touchdown_probability']*iup,axis=(1,2)); sup=np.max(np.clip((2-np.sum(pred['contact_probability'],axis=-1))/2,0,1),axis=1); comp={'force':_b_ratio(ft,refs['force_norm_p95_N']),'torque':_b_ratio(tt,refs['abs_tau_p95_Nm']),'body':_b_ratio(bt,refs['body_robust_z_p95']),'E':_b_ratio(Et,refs['E_norm_p95']),'foot':_b_ratio(footp,refs['foot_speed_p95_mps']),'impulse':_b_ratio(iv,refs['touchdown_impulse_norm_p95_Ns']),'support':sup}; physical=.20*comp['force']+.14*comp['torque']+.14*comp['body']+.14*comp['E']+.10*comp['foot']+.18*comp['impulse']+.10*comp['support']; return {**comp,'physical_risk':np.clip(physical,0,1),'force_upper':ft,'E_upper':Et}

@torch.no_grad()
def predict_B_fast(d,split='val',batch=512):
    idx=np.flatnonzero(d['split']==split); stores={k:[] for k in ['mz','sz','c','td','im','isc','tail']}; m=B9_DENSE['model'].to(B_DEVICE).eval()
    for st in range(0,len(idx),batch):
        p=idx[st:st+batch]; x=torch.from_numpy(np.asarray(d['X_core'][p],np.float32)).to(B_DEVICE); a=torch.from_numpy(np.asarray(d['X_action_dense'][p],np.float32)).to(B_DEVICE); x=(x-NORM_T['x_mean'][None,:])/NORM_T['x_std'][None,:]; a=(a-NORM_T['action_mean'][None,None,:])/NORM_T['action_std'][None,None,:]
        with torch.autocast('cuda',dtype=torch.float16,enabled=USE_AMP): o=m(x,a)
        stores['mz'].append(o['mean_z'].float().cpu().numpy()); stores['sz'].append(torch.exp(o['log_scale_z'].float()).cpu().numpy()); stores['c'].append(torch.sigmoid(o['contact_logits'].float()).cpu().numpy()); stores['td'].append(torch.sigmoid(o['touchdown_logits'].float()).cpu().numpy()); stores['im'].append(o['impulse_mean_z'].float().cpu().numpy()); stores['isc'].append(torch.exp(o['impulse_log_scale_z'].float()).cpu().numpy()); stores['tail'].append(o['tail_q90_log'].float().cpu().numpy())
    q={k:np.concatenate(v) for k,v in stores.items()}; return {'continuous_mean':(q['mz']*B9_PREP['y_std'][None]+B9_PREP['y_mean'][None]).astype(np.float32),'continuous_scale':(q['sz']*B9_PREP['y_std'][None]).astype(np.float32),'contact_probability':q['c'].astype(np.float32),'touchdown_probability':q['td'].astype(np.float32),'impulse_mean':(q['im']*B9_PREP['imp_std'][None,None,None,:]+B9_PREP['imp_mean'][None,None,None,:]).astype(np.float32),'impulse_scale':(q['isc']*B9_PREP['imp_std'][None,None,None,:]).astype(np.float32),'tail_q90':np.maximum(np.expm1(q['tail']),0).astype(np.float32),'truth':d['Y_cont'][idx],'contact_truth':d['Y_contact'][idx],'touchdown_truth':d['Y_touchdown'][idx],'impulse_truth':d['Y_impulse'][idx],'origins':d['origins'][idx]}

ACTIVE={'ck':None,'m':None}
def get_model(bundle):
    if ACTIVE['ck']==bundle['checkpoint'] and ACTIVE['m'] is not None: return ACTIVE['m']
    if ACTIVE['m'] is not None: ACTIVE['m'].to('cpu')
    s=torch.load(bundle['checkpoint'],map_location='cpu',weights_only=False); m=CVisualResidualModel(fresh_base(),len(C7R_VIS_NORM['rgb_mean']),len(C7R_VIS_NORM['depth_mean']),s['mode']); miss,un=m.load_state_dict(s['adapter_state_dict'],strict=False); bad=[x for x in miss if not x.startswith('base.')]
    if bad or un: raise RuntimeError((bad,un))
    ACTIVE.update({'ck':bundle['checkpoint'],'m':m.to(B_DEVICE).eval()}); return ACTIVE['m']

@torch.no_grad()
def predict_C7R(bundle,rid,d,split='val',control='true',seed=0,batch=256):
    idx=np.flatnonzero(d['split']==split).astype(np.int64); perm=np.random.default_rng(seed).permutation(len(idx)) if control=='shuffled' else None
    if control=='run_mean':
        sr=sd=None;n=0
        for st in range(0,len(idx),512):
            R,D=c7r_origin_visual(rid,d,idx[st:st+512]); sr=R.sum(0,dtype=np.float64) if sr is None else sr+R.sum(0,dtype=np.float64); sd=D.sum(0,dtype=np.float64) if sd is None else sd+D.sum(0,dtype=np.float64); n+=len(R)
        mr=(sr/n).astype(np.float32); md=(sd/n).astype(np.float32)
    model=get_model(bundle); S={k:[] for k in ['mz','sz','c','td','im','isc','tail']}
    for st in range(0,len(idx),batch):
        ed=min(st+batch,len(idx)); p=idx[st:ed]
        if control=='true': R,D=c7r_origin_visual(rid,d,p)
        elif control=='shuffled': R,D=c7r_origin_visual(rid,d,idx[perm[st:ed]])
        elif control=='run_mean': R=np.repeat(mr[None],len(p),0); D=np.repeat(md[None],len(p),0)
        else: raise ValueError(control)
        x=torch.from_numpy(np.asarray(d['X_core'][p],np.float32)).to(B_DEVICE); a=torch.from_numpy(np.asarray(d['X_action_dense'][p],np.float32)).to(B_DEVICE); r=torch.from_numpy(R).to(B_DEVICE); dp=torch.from_numpy(D).to(B_DEVICE); x=(x-NORM_T['x_mean'][None,:])/NORM_T['x_std'][None,:]; a=(a-NORM_T['action_mean'][None,None,:])/NORM_T['action_std'][None,None,:]; r=(r-NORM_T['rgb_mean'][None,:])/NORM_T['rgb_std'][None,:]; dp=(dp-NORM_T['depth_mean'][None,:])/NORM_T['depth_std'][None,:]
        with torch.autocast('cuda',dtype=torch.float16,enabled=USE_AMP): o=model(x,a,r,dp)
        S['mz'].append(o['mean_z'].float().cpu().numpy()); S['sz'].append(torch.exp(o['log_scale_z'].float()).cpu().numpy()); S['c'].append(torch.sigmoid(o['contact_logits'].float()).cpu().numpy()); S['td'].append(torch.sigmoid(o['touchdown_logits'].float()).cpu().numpy()); S['im'].append(o['impulse_mean_z'].float().cpu().numpy()); S['isc'].append(torch.exp(o['impulse_log_scale_z'].float()).cpu().numpy()); S['tail'].append(o['tail_q90_log'].float().cpu().numpy())
    q={k:np.concatenate(v) for k,v in S.items()}; return {'continuous_mean':(q['mz']*B9_PREP['y_std'][None]+B9_PREP['y_mean'][None]).astype(np.float32),'continuous_scale':(q['sz']*B9_PREP['y_std'][None]).astype(np.float32),'contact_probability':q['c'].astype(np.float32),'touchdown_probability':q['td'].astype(np.float32),'impulse_mean':(q['im']*B9_PREP['imp_std'][None,None,None,:]+B9_PREP['imp_mean'][None,None,None,:]).astype(np.float32),'impulse_scale':(q['isc']*B9_PREP['imp_std'][None,None,None,:]).astype(np.float32),'tail_q90':np.maximum(np.expm1(q['tail']),0).astype(np.float32),'truth':d['Y_cont'][idx],'contact_truth':d['Y_contact'][idx],'touchdown_truth':d['Y_touchdown'][idx],'impulse_truth':d['Y_impulse'][idx],'origins':d['origins'][idx]}

def merge_preds(ps): return {k:np.concatenate([p[k] for p in ps],axis=0) for k in ps[0]}
def fit_cal(mode,preds):
    val=merge_preds(list(preds.values())); truth=_b_reshape_cont(val['truth']); mu=np.asarray(val['continuous_mean'],float); sc=np.maximum(np.asarray(val['continuous_scale'],float),1e-6); sf={}
    for g,sl in B_STEP_LAYOUT.items(): sf[g]=max(float(np.quantile(np.abs(truth[:,:,sl]-mu[:,:,sl])/sc[:,:,sl],.95)),1e-3)
    td=val['touchdown_truth'].astype(bool); impf=float(np.quantile(np.abs(val['impulse_truth']-val['impulse_mean'])[td]/np.maximum(val['impulse_scale'],1e-6)[td],.95)) if np.any(td) else 1.96; ta=_b_tail_raw(truth,B9_PREP['body_med'],B9_PREP['body_mad']); tp=np.asarray(val['tail_q90'],float); add={name:float(max(0,np.quantile(ta[:,j]-tp[:,j],.90))) for j,name in enumerate(TAIL_NAMES)}; y=val['touchdown_truth'].reshape(-1).astype(int); p=val['touchdown_probability'].reshape(-1); best=max(((*precision_recall_fscore_support(y,(p>=th).astype(int),average='binary',zero_division=0)[:3],float(th)) for th in np.linspace(.05,.95,37)),key=lambda x:x[2])
    return {'scale_factor':sf,'impulse_scale_factor':impf,'tail_add':add,'touchdown_threshold':best[3],'touchdown_val_precision':best[0],'touchdown_val_recall':best[1],'touchdown_val_f1':best[2],'risk_reference':dict(B9_CALIBRATION['risk_reference'])}
def subset(pred,mask): return {k:(v[mask] if isinstance(v,np.ndarray) and len(v)==len(mask) else v) for k,v in pred.items()}
def metrics(pred,d,cal,mask=None):
    n=len(pred['truth']); mask=np.ones(n,bool) if mask is None else np.asarray(mask,bool)
    if mask.sum()<20: return {'samples':int(mask.sum()),'risk_spearman':np.nan,'top10_overlap':np.nan,'force_peak_spearman':np.nan,'force_tail_component_mae':np.nan,'E_tail_component_mae':np.nan,'touchdown_f1':np.nan}
    p=subset(pred,mask); truth=_b_reshape_cont(p['truth']); mu=p['continuous_mean']; ft=_b_group(truth,'force_odom').reshape(-1,B_H,4,3); fp=_b_group(mu,'force_odom').reshape(-1,B_H,4,3); Et=_b_group(truth,'E').reshape(-1,B_H,4,3); Ep=_b_group(mu,'E').reshape(-1,B_H,4,3); fn=np.linalg.norm(ft,axis=-1); En=np.linalg.norm(Et,axis=-1); fm=fn>B9_PREP['force95']; em=En>B9_PREP['E95']; fma=float(np.abs(fp-ft)[fm].mean()) if np.any(fm) else np.nan; ema=float(np.abs(Ep-Et)[em].mean()) if np.any(em) else np.nan; fs=float(pd.Series(np.max(np.linalg.norm(fp,axis=-1),axis=(1,2))).corr(pd.Series(np.max(fn,axis=(1,2))),method='spearman')); y=p['touchdown_truth'].reshape(-1).astype(int); yp=(p['touchdown_probability'].reshape(-1)>=cal['touchdown_threshold']).astype(int); f1=precision_recall_fscore_support(y,yp,average='binary',zero_division=0)[2]; pr=predicted_risk(p,B9_PREP,cal); ar=actual_risk(truth,p['contact_truth'],p['touchdown_truth'],p['impulse_truth'],B9_PREP,cal); rs=float(pd.Series(pr['physical_risk']).corr(pd.Series(ar['physical_risk']),method='spearman')); k=max(1,int(round(.1*len(truth)))); ov=len(set(np.argpartition(pr['physical_risk'],-k)[-k:])&set(np.argpartition(ar['physical_risk'],-k)[-k:]))/float(k); return {'samples':int(mask.sum()),'risk_spearman':rs,'top10_overlap':ov,'force_peak_spearman':fs,'force_tail_component_mae':fma,'E_tail_component_mae':ema,'touchdown_f1':float(f1)}

# Cache true validation predictions once per model/run.
val_ids=[rid for rid,d in C7R_DATA.items() if np.any(d['split']=='val')]
TRUE={}; CAL={}; BASE={rid:predict_B_fast(C7R_DATA[rid],'val') for rid in val_ids}
for mode,b in C7R_MODELS.items():
    TRUE[mode]={rid:predict_C7R(b,rid,C7R_DATA[rid],'val','true',SEED) for rid in val_ids}; CAL[mode]=fit_cal(mode,TRUE[mode])
rows=[]
for mode in ['rgb','depth','rgbd']:
    for rid in val_ids:
        d=C7R_DATA[rid]; vm=d['split']=='val'; rows.append({'mode':mode,'run_id':rid,'terrain':d['terrain'],'subset':'all',**metrics(TRUE[mode][rid],d,CAL[mode])}); rows.append({'mode':mode,'run_id':rid,'terrain':d['terrain'],'subset':'precontact',**metrics(TRUE[mode][rid],d,CAL[mode],d['C_precontact'][vm])})
for rid in val_ids:
    d=C7R_DATA[rid]; vm=d['split']=='val'; rows.append({'mode':'B9_baseline','run_id':rid,'terrain':d['terrain'],'subset':'all',**metrics(BASE[rid],d,B9_CALIBRATION)}); rows.append({'mode':'B9_baseline','run_id':rid,'terrain':d['terrain'],'subset':'precontact',**metrics(BASE[rid],d,B9_CALIBRATION,d['C_precontact'][vm])})
VAL=pd.DataFrame(rows); display(VAL)
base=VAL[(VAL['mode']=='B9_baseline')&(VAL['subset']=='precontact')].mean(numeric_only=True); sels=[]
for mode in ['rgb','depth','rgbd']:
    g=VAL[(VAL['mode']==mode)&(VAL['subset']=='precontact')].mean(numeric_only=True); fg=(base['force_tail_component_mae']-g['force_tail_component_mae'])/base['force_tail_component_mae']; eg=(base['E_tail_component_mae']-g['E_tail_component_mae'])/base['E_tail_component_mae']; score=.35*g['risk_spearman']+.25*g['force_peak_spearman']+.20*fg+.10*eg+.10*g['touchdown_f1']; sels.append({'mode':mode,'validation_selection_score':float(score),'precontact_risk_spearman':float(g['risk_spearman']),'precontact_force_peak_spearman':float(g['force_peak_spearman']),'precontact_force_tail_gain_vs_B9':float(fg),'precontact_E_tail_gain_vs_B9':float(eg),'precontact_touchdown_f1':float(g['touchdown_f1'])})
SEL=pd.DataFrame(sels).sort_values('validation_selection_score',ascending=False).reset_index(drop=True); display(SEL); selected=str(SEL.iloc[0]['mode']); print('SELECTED:',selected)
control=[]
for rid in val_ids:
    d=C7R_DATA[rid]; vm=d['split']=='val'; mask=d['C_precontact'][vm]; control.append({'condition':'B9_baseline','run_id':rid,'terrain':d['terrain'],**metrics(BASE[rid],d,B9_CALIBRATION,mask)}); control.append({'condition':'true','run_id':rid,'terrain':d['terrain'],**metrics(TRUE[selected][rid],d,CAL[selected],mask)})
    seed=SEED+int(hashlib.sha256(rid.encode()).hexdigest()[:8],16)%100000
    for cond in ['shuffled','run_mean']:
        pp=predict_C7R(C7R_MODELS[selected],rid,d,'val',cond,seed); control.append({'condition':cond,'run_id':rid,'terrain':d['terrain'],**metrics(pp,d,CAL[selected],mask)})
CTRL=pd.DataFrame(control); display(CTRL); SUM=CTRL.groupby('condition').mean(numeric_only=True); display(SUM)
v=lambda c,m:float(SUM.loc[c,m]); tr=v('true','risk_spearman'); br=v('B9_baseline','risk_spearman'); sr=v('shuffled','risk_spearman'); rr=v('run_mean','risk_spearman'); tf=v('true','force_peak_spearman'); bf=v('B9_baseline','force_peak_spearman'); bt=v('B9_baseline','force_tail_component_mae'); tt=v('true','force_tail_component_mae')
GATE={'selected_mode':selected,'feature_version':FEATURE_VERSION,'fresh_test_ids_still_locked':TEST_ROWS['run_id'].tolist(),'validation_precontact_true_risk_spearman':tr,'validation_precontact_B9_risk_spearman':br,'validation_precontact_shuffled_risk_spearman':sr,'validation_precontact_runmean_risk_spearman':rr,'validation_precontact_true_force_peak_spearman':tf,'validation_precontact_B9_force_peak_spearman':bf,'validation_precontact_force_tail_improvement_fraction':(bt-tt)/max(bt,1e-9),'true_beats_B9_risk':bool(tr>br+.01),'true_beats_shuffled_risk':bool(tr>sr+.01),'true_beats_runmean_risk':bool(tr>rr+.005),'tail_or_peak_improves':bool(tf>bf+.01 or tt<bt)}; GATE['open_fresh_C_test']=bool(GATE['true_beats_B9_risk'] and GATE['true_beats_shuffled_risk'] and GATE['true_beats_runmean_risk'] and GATE['tail_or_peak_improves'])
print('\n'+'='*92+'\nC7R SPATIAL/TEMPORAL VALIDATION GATE\n'+'='*92); print(json.dumps(GATE,indent=2)); VAL.to_csv(C7R_TABLE_ROOT/'C6R_validation_metrics.csv',index=False); SEL.to_csv(C7R_TABLE_ROOT/'C6R_mode_selection.csv',index=False); CTRL.to_csv(C7R_TABLE_ROOT/'C7R_causality_controls.csv',index=False); (C7R_TABLE_ROOT/'C7R_gate.json').write_text(json.dumps(GATE,indent=2)); print(f'\nTotal notebook runtime: {(time.time()-T0)/60:.1f} min')
if GATE['open_fresh_C_test']:
    print('\nPASS. Fresh C tests are STILL UNSCORED. Stop here and send me this gate output for the one-time C8R-C10 run.')
else:
    print('\nNOT PASSED. Fresh C tests remain locked. Do not force the gate.')

# %% notebook cell 5

# ======================================================================
# C7S — VALIDATION-ONLY CAUSAL HIGH-PASS + ALIGNMENT REPAIR
#
# APPEND/RUN THIS AFTER UGP_C7R_T4_RESCUE_100MIN_FINAL has completed.
#
# Scientific purpose:
#   Previous C7R still let run/scene context compete with the true local
#   visual signal: true ~= shuffled ~= run_mean.
#
# This repair changes ONLY the visual branch while keeping frozen:
#   - B9 dynamics model/checkpoint
#   - B9 calibration/risk definition
#   - C train/val split
#   - physical targets
#   - optimizer micro-batch = 96
#   - validation-only model selection
#   - anti-cheating gate thresholds
#
# New visual representation is causal and "absolute-scene-free":
#   [ current - causal trailing mean(8 frames),
#     current - previous frame,
#     current - frame 4 steps ago ]
#
# This removes most trajectory/scene identity information and asks whether
# time-local visual CHANGE predicts future physical response.
#
# A second train-only regularizer presents a within-run mismatched visual
# observation and asks its residual correction to collapse toward frozen B9.
# This is NOT applied to validation labels and never touches fresh C tests.
#
# Fresh C tests remain locked throughout.
# ======================================================================

import gc, time, json, hashlib, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import precision_recall_fscore_support
from IPython.display import display

# ----------------------------------------------------------------------
# 0. HARD GUARDS — refuse to run in the wrong state
# ----------------------------------------------------------------------
_REQUIRED = [
    "C_ROOT", "C7R_FRAME_FEATURES", "C7R_DATA", "TEST_ROWS",
    "B9_CKPT", "B9_PREP", "B9_CALIBRATION", "B9_DENSE",
    "BResponseModel", "B_DEVICE", "B_H", "B_STEP_WIDTH",
    "B_STEP_LAYOUT", "B_MODEL_DIM", "B_ACTION_HIDDEN",
    "B_ACTION_GATHER", "TAIL_NAMES", "CFG",
    "_b_tail_raw", "_b_reshape_cont", "_b_group",
    "b_loss", "fresh_base", "predict_B_fast",
    "fit_cal", "metrics"
]
_missing = [x for x in _REQUIRED if x not in globals()]
if _missing:
    raise RuntimeError(
        "Run the completed T4 rescue notebook first. Missing globals: "
        + repr(_missing)
    )

C8_MARKER_V2 = C_ROOT / "C8_TEST_WAS_SCORED.json"
if C8_MARKER_V2.exists():
    raise RuntimeError(
        f"Fresh Stage-C test was already scored: {C8_MARKER_V2}. "
        "Do not use it for further visual tuning."
    )

_fresh_ids = TEST_ROWS["run_id"].astype(str).tolist()
if set(_fresh_ids) & set(C7R_DATA.keys()):
    raise RuntimeError(
        "Fresh C test trajectories unexpectedly exist in C7R_DATA. "
        "Refusing to continue."
    )

print("Fresh C tests remain LOCKED:", _fresh_ids)
print("C7S uses development runs only:", sorted(C7R_DATA.keys()))

# ----------------------------------------------------------------------
# 1. NEW NAMESPACE / CONSTANTS
# ----------------------------------------------------------------------
SEED2 = int(globals().get("SEED", 1781)) + 202
random.seed(SEED2)
np.random.seed(SEED2)
torch.manual_seed(SEED2)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED2)

C7S_ROOT = C_ROOT / "spatial_temporal_repair_v2_highpass_alignment"
C7S_MODEL_ROOT = C7S_ROOT / "models"
C7S_TABLE_ROOT = C7S_ROOT / "tables"
for _p in (C7S_ROOT, C7S_MODEL_ROOT, C7S_TABLE_ROOT):
    _p.mkdir(parents=True, exist_ok=True)

C7S_FEATURE_VERSION = (
    "C7S_causal_highpass8_d1_d4__"
    "dino_lower60_2x3_depth_lower60_8x8x4__"
    "withinrun_negative_null_v1"
)

# Keep approved optimization schedule.
C7S_TRAIN_BATCH = 96
C7S_FETCH_BATCH = 384
C7S_EPOCHS = 20
C7S_PATIENCE = 5
C7S_LR = 2e-4
C7S_WEIGHT_DECAY = 1e-4
C7S_RESIDUAL_L2 = 2e-3

# New repair-specific settings.
C7S_CAUSAL_WINDOW_FRAMES = 8
C7S_LONG_DELTA_FRAMES = 4
C7S_NEG_ORIGIN_OFFSET = 257      # ~2.57 s at the 100-Hz origin grid
C7S_NEG_NULL_WEIGHT = 0.05
C7S_PRECONTACT_EXTRA_WEIGHT = 3.0

C7S_HIDDEN = 128
C7S_USE_AMP = True

print({
    "feature_version": C7S_FEATURE_VERSION,
    "optimizer_batch": C7S_TRAIN_BATCH,
    "fetch_batch": C7S_FETCH_BATCH,
    "causal_window_frames": C7S_CAUSAL_WINDOW_FRAMES,
    "long_delta_frames": C7S_LONG_DELTA_FRAMES,
    "negative_offset_origins": C7S_NEG_ORIGIN_OFFSET,
    "fresh_tests_loaded": False,
})

# ----------------------------------------------------------------------
# 2. PRECOMPUTE CAUSAL HIGH-PASS FRAME TABLES
#
# No future image is used.
# trailing_mean[i] = mean(F[max(0,i-W):i]); current frame is excluded.
# ----------------------------------------------------------------------
def _causal_change_table(F, window=8, long_delta=4):
    F = np.asarray(F, dtype=np.float32)
    if F.ndim != 2 or len(F) < 2:
        raise ValueError(f"Unexpected frame feature shape: {F.shape}")

    n, d = F.shape
    # float64 cumulative sum only during this one-time preparation.
    cs = np.zeros((n + 1, d), dtype=np.float64)
    np.cumsum(F, axis=0, dtype=np.float64, out=cs[1:])

    i = np.arange(n, dtype=np.int64)
    lo = np.maximum(0, i - int(window))
    cnt = i - lo

    prev_mean = np.asarray(F, dtype=np.float32).copy()
    good = cnt > 0
    if np.any(good):
        prev_sum = cs[i[good]] - cs[lo[good]]
        prev_mean[good] = (
            prev_sum / cnt[good, None]
        ).astype(np.float32)

    p1 = np.maximum(i - 1, 0)
    pk = np.maximum(i - int(long_delta), 0)

    hp = F - prev_mean
    d1 = F - F[p1]
    dk = F - F[pk]

    # Frame 0 has no causal past.
    hp[0] = 0.0
    d1[0] = 0.0
    dk[0] = 0.0

    return np.concatenate([hp, d1, dk], axis=1).astype(np.float32)


C7S_FRAME = {}
for rid, tab in C7R_FRAME_FEATURES.items():
    C7S_FRAME[rid] = {
        "rgb": _causal_change_table(
            tab["rgb_spatial"],
            C7S_CAUSAL_WINDOW_FRAMES,
            C7S_LONG_DELTA_FRAMES,
        ),
        "depth": _causal_change_table(
            tab["depth_spatial"],
            C7S_CAUSAL_WINDOW_FRAMES,
            C7S_LONG_DELTA_FRAMES,
        ),
    }

print(
    "C7S causal frame tables:",
    {
        rid: (v["rgb"].shape, v["depth"].shape)
        for rid, v in C7S_FRAME.items()
    },
)

def c7s_origin_visual(rid, d, positions):
    positions = np.asarray(positions, dtype=np.int64)
    vi = np.asarray(d["visual_frame_index"][positions], dtype=np.int64)

    tab = C7S_FRAME[rid]
    return (
        np.asarray(tab["rgb"][vi], dtype=np.float32),
        np.asarray(tab["depth"][vi], dtype=np.float32),
    )

# ----------------------------------------------------------------------
# 3. TRAIN-ONLY NORMALIZATION
# ----------------------------------------------------------------------
def fit_c7s_vis_norm():
    sr = sr2 = sd = sd2 = None
    n = 0

    for rid, d in C7R_DATA.items():
        pos = np.flatnonzero(d["split"] == "train")
        for st in range(0, len(pos), 512):
            R, D = c7s_origin_visual(rid, d, pos[st:st+512])

            if sr is None:
                sr = np.zeros(R.shape[1], dtype=np.float64)
                sr2 = np.zeros_like(sr)
                sd = np.zeros(D.shape[1], dtype=np.float64)
                sd2 = np.zeros_like(sd)

            sr += R.sum(0, dtype=np.float64)
            sr2 += np.square(R, dtype=np.float64).sum(0, dtype=np.float64)
            sd += D.sum(0, dtype=np.float64)
            sd2 += np.square(D, dtype=np.float64).sum(0, dtype=np.float64)
            n += len(R)

    rm = sr / n
    dm = sd / n
    rs = np.sqrt(np.maximum(sr2 / n - rm * rm, 0.0))
    ds = np.sqrt(np.maximum(sd2 / n - dm * dm, 0.0))
    rs[rs < 1e-5] = 1.0
    ds[ds < 1e-5] = 1.0

    return {
        "rgb_mean": rm.astype(np.float32),
        "rgb_std": rs.astype(np.float32),
        "depth_mean": dm.astype(np.float32),
        "depth_std": ds.astype(np.float32),
        "train_count": int(n),
    }

C7S_NORM_PATH = C7S_ROOT / "visual_norm.npz"
if C7S_NORM_PATH.exists():
    with np.load(C7S_NORM_PATH, allow_pickle=False) as z:
        C7S_VIS_NORM = {k: z[k] for k in z.files}
    C7S_VIS_NORM["train_count"] = int(
        np.asarray(C7S_VIS_NORM["train_count"]).reshape(())
    )
    print("Reused:", C7S_NORM_PATH)
else:
    C7S_VIS_NORM = fit_c7s_vis_norm()
    np.savez_compressed(C7S_NORM_PATH, **C7S_VIS_NORM)
    print("Wrote:", C7S_NORM_PATH)

print({
    "train_count": C7S_VIS_NORM["train_count"],
    "rgb_dim": len(C7S_VIS_NORM["rgb_mean"]),
    "depth_dim": len(C7S_VIS_NORM["depth_mean"]),
})

C7S_NORM_T = {
    k: torch.as_tensor(v, dtype=torch.float32, device=B_DEVICE)
    for k, v in {
        "x_mean": B9_PREP["x_mean"],
        "x_std": B9_PREP["x_std"],
        "action_mean": B9_PREP["action_mean"],
        "action_std": B9_PREP["action_std"],
        "y_mean": B9_PREP["y_mean"],
        "y_std": B9_PREP["y_std"],
        "imp_mean": B9_PREP["imp_mean"],
        "imp_std": B9_PREP["imp_std"],
        "rgb_mean": C7S_VIS_NORM["rgb_mean"],
        "rgb_std": C7S_VIS_NORM["rgb_std"],
        "depth_mean": C7S_VIS_NORM["depth_mean"],
        "depth_std": C7S_VIS_NORM["depth_std"],
    }.items()
}

# ----------------------------------------------------------------------
# 4. DATASET / SAMPLER
#
# Same terrain/tail/event balancing as C7R, plus extra TRAIN-ONLY weight
# for the pre-contact subset because this stage explicitly tests visual
# anticipation before physical contact.
# ----------------------------------------------------------------------
class C7SIndexDataset(Dataset):
    def __init__(self, data_dict, split, balanced=False):
        self.data_dict = data_dict
        self.run_keys = []

        ro, lo, ftp, etp, tlp, swp = [], [], [], [], [], []
        rec = []
        terrain_counts = {}

        for rid, d in data_dict.items():
            idx = np.flatnonzero(d["split"] == split).astype(np.int64)
            if len(idx):
                terrain_counts[d["terrain"]] = (
                    terrain_counts.get(d["terrain"], 0) + len(idx)
                )
                rec.append((rid, d, idx, d["terrain"]))

        total = sum(len(x[2]) for x in rec)
        terrain_factor = {
            t: total / (len(terrain_counts) * count)
            for t, count in terrain_counts.items()
        }

        for rix, (rid, d, idx, terrain) in enumerate(rec):
            self.run_keys.append(rid)
            ro.append(np.full(len(idx), rix, dtype=np.int16))
            lo.append(idx)

            step = np.asarray(
                d["Y_cont"][idx], dtype=np.float32
            ).reshape(-1, B_H, B_STEP_WIDTH)

            td = np.asarray(d["Y_touchdown"][idx], dtype=np.float32)

            force = step[:, :, B_STEP_LAYOUT["force_odom"]].reshape(
                -1, B_H, 4, 3
            )
            E = step[:, :, B_STEP_LAYOUT["E"]].reshape(
                -1, B_H, 4, 3
            )

            ft = (
                np.linalg.norm(force, axis=-1) > B9_PREP["force95"]
            )
            et = (
                np.linalg.norm(E, axis=-1) > B9_PREP["E95"]
            )

            tail = np.log1p(
                np.maximum(
                    _b_tail_raw(
                        step,
                        B9_PREP["body_med"],
                        B9_PREP["body_mad"],
                    ),
                    0.0,
                )
            ).astype(np.float32)

            sw = (
                1.0
                + 2.0 * ft.max((1, 2)).astype(np.float32)
                + 2.0 * et.max((1, 2)).astype(np.float32)
                + 4.0 * td.max((1, 2)).astype(np.float32)
            )

            if split == "train":
                pre = np.asarray(
                    d["C_precontact"][idx], dtype=np.float32
                )
                sw += C7S_PRECONTACT_EXTRA_WEIGHT * pre

            if balanced:
                sw *= float(terrain_factor[terrain])

            ftp.append(ft.astype(np.uint8))
            etp.append(et.astype(np.uint8))
            tlp.append(tail)
            swp.append(np.clip(sw, 1.0, 16.0).astype(np.float32))

        self.run_of = np.concatenate(ro)
        self.local_of = np.concatenate(lo)
        self.force_tail = np.concatenate(ftp)
        self.E_tail = np.concatenate(etp)
        self.tail_log = np.concatenate(tlp)
        self.sample_weight = np.concatenate(swp)

    def __len__(self):
        return len(self.local_of)

    def __getitem__(self, i):
        return int(i)


def c7s_make_collate(ds):
    rgb_dim = len(C7S_VIS_NORM["rgb_mean"])
    dep_dim = len(C7S_VIS_NORM["depth_mean"])

    def collate(items):
        gi = np.asarray(items, dtype=np.int64)
        Bn = len(gi)

        X = np.empty((Bn, 1060), dtype=np.float32)
        A = np.empty((Bn, 100, 18), dtype=np.float32)
        Y = np.empty((Bn, B_H, B_STEP_WIDTH), dtype=np.float32)
        C = np.empty((Bn, B_H, 4), dtype=np.float32)
        TD = np.empty_like(C)
        IMP = np.empty((Bn, B_H, 4, 3), dtype=np.float32)

        R = np.empty((Bn, rgb_dim), dtype=np.float32)
        D = np.empty((Bn, dep_dim), dtype=np.float32)
        RN = np.empty_like(R)
        DN = np.empty_like(D)

        runix = ds.run_of[gi]
        local = ds.local_of[gi]

        for rix in np.unique(runix):
            outpos = np.flatnonzero(runix == rix)
            pos = local[outpos]
            rid = ds.run_keys[int(rix)]
            d = ds.data_dict[rid]

            X[outpos] = d["X_core"][pos]
            A[outpos] = d["X_action_dense"][pos]
            Y[outpos] = d["Y_cont"][pos].reshape(
                -1, B_H, B_STEP_WIDTH
            )
            C[outpos] = d["Y_contact"][pos]
            TD[outpos] = d["Y_touchdown"][pos]
            IMP[outpos] = d["Y_impulse"][pos]

            rr, dd = c7s_origin_visual(rid, d, pos)
            R[outpos] = rr
            D[outpos] = dd

            # Deterministic within-run mismatched visual pairing.
            # Same environment/run, wrong time.
            neg_pos = (
                pos + C7S_NEG_ORIGIN_OFFSET
            ) % len(d["origins"])
            rrn, ddn = c7s_origin_visual(rid, d, neg_pos)
            RN[outpos] = rrn
            DN[outpos] = ddn

        vals = [
            X, A, Y, C, TD, IMP,
            ds.force_tail[gi].astype(np.float32),
            ds.E_tail[gi].astype(np.float32),
            ds.tail_log[gi],
            R, D, RN, DN,
        ]
        return tuple(torch.from_numpy(v) for v in vals)

    return collate


def c7s_prep_batch(raw):
    raw = tuple(
        t.to(B_DEVICE, non_blocking=True)
        for t in raw
    )
    (
        x, a, y, c, td, imp,
        ft, et, tail,
        rgb, dep, rgb_neg, dep_neg,
    ) = raw

    x = (
        (x - C7S_NORM_T["x_mean"][None, :])
        / C7S_NORM_T["x_std"][None, :]
    )
    a = (
        (a - C7S_NORM_T["action_mean"][None, None, :])
        / C7S_NORM_T["action_std"][None, None, :]
    )
    y = (
        (y - C7S_NORM_T["y_mean"][None, :, :])
        / C7S_NORM_T["y_std"][None, :, :]
    )
    imp = (
        (imp - C7S_NORM_T["imp_mean"][None, None, None, :])
        / C7S_NORM_T["imp_std"][None, None, None, :]
    )

    rgb = (
        (rgb - C7S_NORM_T["rgb_mean"][None, :])
        / C7S_NORM_T["rgb_std"][None, :]
    )
    dep = (
        (dep - C7S_NORM_T["depth_mean"][None, :])
        / C7S_NORM_T["depth_std"][None, :]
    )

    rgb_neg = (
        (rgb_neg - C7S_NORM_T["rgb_mean"][None, :])
        / C7S_NORM_T["rgb_std"][None, :]
    )
    dep_neg = (
        (dep_neg - C7S_NORM_T["depth_mean"][None, :])
        / C7S_NORM_T["depth_std"][None, :]
    )

    return (
        (x, a, y, c, td, imp, ft, et, tail),
        rgb, dep,
        rgb_neg, dep_neg,
    )

# ----------------------------------------------------------------------
# 5. VISUAL RESIDUAL MODEL
#
# Same frozen B9 architecture. We only expose a fuller residual magnitude
# so mismatched visuals can be explicitly pushed back toward B9.
# ----------------------------------------------------------------------
class C7SVisualResidualModel(nn.Module):
    def __init__(self, base, rgb_dim, depth_dim, mode):
        super().__init__()
        self.mode = mode
        self.base = base.eval()

        for p in self.base.parameters():
            p.requires_grad_(False)

        self.rgb = nn.Sequential(
            nn.Linear(rgb_dim, C7S_HIDDEN),
            nn.GELU(),
            nn.LayerNorm(C7S_HIDDEN),
            nn.Linear(C7S_HIDDEN, C7S_HIDDEN),
            nn.GELU(),
            nn.LayerNorm(C7S_HIDDEN),
        )
        self.depth = nn.Sequential(
            nn.Linear(depth_dim, C7S_HIDDEN),
            nn.GELU(),
            nn.LayerNorm(C7S_HIDDEN),
            nn.Linear(C7S_HIDDEN, C7S_HIDDEN),
            nn.GELU(),
            nn.LayerNorm(C7S_HIDDEN),
        )

        vd = (
            C7S_HIDDEN
            if mode in {"rgb", "depth"}
            else 2 * C7S_HIDDEN
        )

        self.h_adapter = nn.Sequential(
            nn.Linear(B_MODEL_DIM + vd, B_MODEL_DIM),
            nn.GELU(),
            nn.LayerNorm(B_MODEL_DIM),
            nn.Dropout(0.05),
            nn.Linear(B_MODEL_DIM, B_MODEL_DIM),
            nn.GELU(),
            nn.LayerNorm(B_MODEL_DIM),
        )

        self.seq_adapter = nn.Sequential(
            nn.Linear(
                B_MODEL_DIM + B_ACTION_HIDDEN + vd,
                B_MODEL_DIM,
            ),
            nn.GELU(),
            nn.LayerNorm(B_MODEL_DIM),
        )

        self.d_mean = nn.Linear(B_MODEL_DIM, B_STEP_WIDTH)
        self.d_log_scale = nn.Linear(B_MODEL_DIM, B_STEP_WIDTH)
        self.d_contact = nn.Linear(B_MODEL_DIM, 4)
        self.d_touchdown = nn.Linear(B_MODEL_DIM, 4)
        self.d_impulse_mean = nn.Linear(B_MODEL_DIM, 12)
        self.d_impulse_log_scale = nn.Linear(B_MODEL_DIM, 12)
        self.d_tail = nn.Linear(B_MODEL_DIM, 4)

        for layer in [
            self.d_mean,
            self.d_log_scale,
            self.d_contact,
            self.d_touchdown,
            self.d_impulse_mean,
            self.d_impulse_log_scale,
            self.d_tail,
        ]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _v(self, r, d):
        if self.mode == "rgb":
            return self.rgb(r)
        if self.mode == "depth":
            return self.depth(d)
        return torch.cat([self.rgb(r), self.depth(d)], dim=-1)

    def forward(self, x, a, r, d):
        self.base.eval()

        with torch.no_grad():
            h0 = self.base.history(x)
            ah = self.base.action(
                a, B_ACTION_GATHER.to(a.device)
            )
            afinal = ah[:, -1]

            h = h0[:, None, :].expand(-1, B_H, -1)
            he = self.base.horizon_embed[None].expand(
                len(x), -1, -1
            )

            z = self.base.fuse(
                torch.cat([h, ah, he], dim=-1)
            )

            base = {
                "mean_z": self.base.mean_head(z),
                "log_scale_z": torch.clamp(
                    self.base.log_scale_head(z),
                    -5.5, 2.8,
                ),
                "contact_logits": self.base.contact_head(z),
                "touchdown_logits":
                    self.base.touchdown_head(z),
                "impulse_mean_z":
                    self.base.impulse_mean_head(z).reshape(
                        -1, B_H, 4, 3
                    ),
                "impulse_log_scale_z":
                    torch.clamp(
                        self.base.impulse_log_scale_head(z).reshape(
                            -1, B_H, 4, 3
                        ),
                        -5.5, 3.0,
                    ),
                "tail_q90_log":
                    self.base.tail_q90_head(
                        torch.cat([h0, afinal], dim=-1)
                    ),
            }

        v = self._v(r, d)

        rz = self.h_adapter(
            torch.cat(
                [
                    z,
                    v[:, None, :].expand(-1, B_H, -1),
                ],
                dim=-1,
            )
        )

        seq = self.seq_adapter(
            torch.cat([h0, afinal, v], dim=-1)
        )

        dm = self.d_mean(rz)
        dls = self.d_log_scale(rz)
        dc = self.d_contact(rz)
        dtd = self.d_touchdown(rz)
        dim = self.d_impulse_mean(rz).reshape(
            -1, B_H, 4, 3
        )
        dils = self.d_impulse_log_scale(rz).reshape(
            -1, B_H, 4, 3
        )
        dtail = self.d_tail(seq)

        out = {
            "mean_z": base["mean_z"] + dm,
            "log_scale_z": torch.clamp(
                base["log_scale_z"] + dls,
                -5.5, 2.8,
            ),
            "contact_logits":
                base["contact_logits"] + dc,
            "touchdown_logits":
                base["touchdown_logits"] + dtd,
            "impulse_mean_z":
                base["impulse_mean_z"] + dim,
            "impulse_log_scale_z":
                torch.clamp(
                    base["impulse_log_scale_z"] + dils,
                    -5.5, 3.0,
                ),
            "tail_q90_log":
                base["tail_q90_log"] + dtail,
        }

        out["_residual_full_l2"] = (
            dm.square().mean()
            + 0.10 * dls.square().mean()
            + 0.05 * dc.square().mean()
            + 0.10 * dtd.square().mean()
            + 0.25 * dim.square().mean()
            + 0.05 * dils.square().mean()
            + 0.25 * dtail.square().mean()
        )

        return out

# ----------------------------------------------------------------------
# 6. TRAIN / RESUME — TRAIN RUNS ONLY
# ----------------------------------------------------------------------
def c7s_fresh_base():
    c = torch.load(
        B9_CKPT,
        map_location="cpu",
        weights_only=False,
    )
    m = BResponseModel(
        int(c["history_dim"]),
        int(c["step_width"]),
        bool(c["use_action"]),
    )
    m.load_state_dict(c["state_dict"])
    return m.eval()


def c7s_train_mode(mode):
    final = C7S_MODEL_ROOT / f"C7S_{mode}_adapter.pt"
    resume = C7S_MODEL_ROOT / f"C7S_{mode}_resume.pt"

    if final.exists():
        s = torch.load(
            final,
            map_location="cpu",
            weights_only=False,
        )
        if s.get("feature_version") == C7S_FEATURE_VERSION:
            print(
                f"C7S-{mode}: reusing completed checkpoint "
                f"{final}"
            )
            return {
                "mode": mode,
                "checkpoint": str(final),
                "best_val_loss": float(s["best_val_loss"]),
            }

    tr = C7SIndexDataset(
        C7R_DATA, "train", balanced=True
    )
    va = C7SIndexDataset(
        C7R_DATA, "val", balanced=False
    )

    sampler = WeightedRandomSampler(
        torch.as_tensor(
            tr.sample_weight,
            dtype=torch.double,
        ),
        len(tr),
        replacement=True,
    )

    trl = DataLoader(
        tr,
        batch_size=C7S_FETCH_BATCH,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=c7s_make_collate(tr),
    )
    val = DataLoader(
        va,
        batch_size=C7S_FETCH_BATCH,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=c7s_make_collate(va),
    )

    model = C7SVisualResidualModel(
        c7s_fresh_base(),
        len(C7S_VIS_NORM["rgb_mean"]),
        len(C7S_VIS_NORM["depth_mean"]),
        mode,
    ).to(B_DEVICE)

    params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    opt = torch.optim.AdamW(
        params,
        lr=C7S_LR,
        weight_decay=C7S_WEIGHT_DECAY,
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=C7S_USE_AMP
    )

    best = float("inf")
    best_state = None
    bad = 0
    hist = []
    start_epoch = 1

    if resume.exists():
        rr = torch.load(
            resume,
            map_location="cpu",
            weights_only=False,
        )

        if (
            rr.get("feature_version") == C7S_FEATURE_VERSION
            and rr.get("mode") == mode
        ):
            model.load_state_dict(
                rr["adapter_state_dict"],
                strict=False,
            )
            opt.load_state_dict(rr["optimizer_state"])
            best = float(rr["best"])
            best_state = rr["best_state"]
            bad = int(rr["bad"])
            hist = list(rr["history"])
            start_epoch = int(rr["epoch"]) + 1
            print(
                f"C7S-{mode}: resuming at epoch "
                f"{start_epoch}"
            )

    @torch.no_grad()
    def eval_loss():
        model.eval()
        total = 0.0
        n = 0

        for raw in val:
            (
                bb, r, d, _rn, _dn
            ) = c7s_prep_batch(raw)

            with torch.autocast(
                "cuda",
                dtype=torch.float16,
                enabled=C7S_USE_AMP,
            ):
                out = model(bb[0], bb[1], r, d)
                loss = (
                    b_loss(out, bb)
                    + C7S_RESIDUAL_L2
                    * out["_residual_full_l2"]
                )

            total += float(loss) * len(bb[0])
            n += len(bb[0])

        return total / max(n, 1)

    for ep in range(start_epoch, C7S_EPOCHS + 1):
        tic = time.time()
        model.train()

        total = 0.0
        n = 0

        for raw_big in trl:
            for off in range(
                0,
                len(raw_big[0]),
                C7S_TRAIN_BATCH,
            ):
                raw = tuple(
                    t[off:off+C7S_TRAIN_BATCH]
                    for t in raw_big
                )

                (
                    bb,
                    r,
                    d,
                    rn,
                    dn,
                ) = c7s_prep_batch(raw)

                opt.zero_grad(set_to_none=True)

                with torch.autocast(
                    "cuda",
                    dtype=torch.float16,
                    enabled=C7S_USE_AMP,
                ):
                    out = model(
                        bb[0], bb[1], r, d
                    )
                    out_neg = model(
                        bb[0], bb[1], rn, dn
                    )

                    loss_true = (
                        b_loss(out, bb)
                        + C7S_RESIDUAL_L2
                        * out["_residual_full_l2"]
                    )

                    # Wrong-time same-run vision should not be
                    # allowed to move the frozen B9 prediction.
                    loss_null = (
                        C7S_NEG_NULL_WEIGHT
                        * out_neg["_residual_full_l2"]
                    )

                    loss = loss_true + loss_null

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    params, 5.0
                )
                scaler.step(opt)
                scaler.update()

                total += float(
                    loss.detach()
                ) * len(bb[0])
                n += len(bb[0])

        train_loss = total / max(n, 1)
        val_loss = eval_loss()

        hist.append({
            "mode": mode,
            "epoch": ep,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "seconds": time.time() - tic,
        })

        print(
            f"C7S-{mode:5s} ep={ep:02d} "
            f"train={train_loss:.5f} "
            f"val={val_loss:.5f} "
            f"time={time.time()-tic:.1f}s"
        )

        if val_loss < best - 1e-4:
            best = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
                if not k.startswith("base.")
            }
            bad = 0
        else:
            bad += 1

        current = {
            k: v.detach().cpu()
            for k, v in model.state_dict().items()
            if not k.startswith("base.")
        }

        torch.save({
            "mode": mode,
            "feature_version": C7S_FEATURE_VERSION,
            "epoch": ep,
            "adapter_state_dict": current,
            "optimizer_state": opt.state_dict(),
            "best": best,
            "best_state": best_state,
            "bad": bad,
            "history": hist,
        }, resume)

        if bad >= C7S_PATIENCE:
            print(f"C7S-{mode}: early stop")
            break

    if best_state is None:
        raise RuntimeError(
            f"C7S-{mode}: no best state found"
        )

    torch.save({
        "mode": mode,
        "adapter_state_dict": best_state,
        "best_val_loss": float(best),
        "vis_norm": C7S_VIS_NORM,
        "feature_version": C7S_FEATURE_VERSION,
        "B9_checkpoint": str(B9_CKPT),
        "uses_test": False,
        "training_history": hist,
        "negative_null_weight":
            C7S_NEG_NULL_WEIGHT,
        "precontact_extra_weight":
            C7S_PRECONTACT_EXTRA_WEIGHT,
    }, final)

    pd.DataFrame(hist).to_csv(
        C7S_TABLE_ROOT
        / f"C7S_{mode}_history.csv",
        index=False,
    )

    del (
        model, tr, va,
        trl, val, sampler,
        opt, params,
    )
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "mode": mode,
        "checkpoint": str(final),
        "best_val_loss": float(best),
    }


B9_DENSE["model"].to("cpu")
C7S_MODELS = {}

print("\nTraining C7S causal high-pass adapters...")

for mode in ("rgb", "depth", "rgbd"):
    print(
        "\n" + "=" * 80
        + f"\nC7S MODE: {mode}"
        + "\n" + "=" * 80
    )
    C7S_MODELS[mode] = c7s_train_mode(mode)

print("\nC7S TRAINING COMPLETE / RESTORED")
display(pd.DataFrame([
    {
        "mode": k,
        "best_val_loss": v["best_val_loss"],
        "checkpoint": v["checkpoint"],
    }
    for k, v in C7S_MODELS.items()
]))

# ----------------------------------------------------------------------
# 7. VALIDATION-ONLY PREDICTION / CALIBRATION
# ----------------------------------------------------------------------
C7S_ACTIVE = {
    "checkpoint": None,
    "model": None,
}

def c7s_get_model(bundle):
    if (
        C7S_ACTIVE["checkpoint"]
        == bundle["checkpoint"]
        and C7S_ACTIVE["model"] is not None
    ):
        return C7S_ACTIVE["model"]

    if C7S_ACTIVE["model"] is not None:
        C7S_ACTIVE["model"].to("cpu")

    s = torch.load(
        bundle["checkpoint"],
        map_location="cpu",
        weights_only=False,
    )

    m = C7SVisualResidualModel(
        c7s_fresh_base(),
        len(C7S_VIS_NORM["rgb_mean"]),
        len(C7S_VIS_NORM["depth_mean"]),
        s["mode"],
    )

    miss, unexpected = m.load_state_dict(
        s["adapter_state_dict"],
        strict=False,
    )

    bad_missing = [
        x for x in miss
        if not x.startswith("base.")
    ]
    if bad_missing or unexpected:
        raise RuntimeError(
            (bad_missing, unexpected)
        )

    m = m.to(B_DEVICE).eval()

    C7S_ACTIVE["checkpoint"] = bundle["checkpoint"]
    C7S_ACTIVE["model"] = m
    return m


@torch.no_grad()
def predict_C7S(
    bundle,
    rid,
    d,
    split="val",
    control="true",
    seed=0,
    batch=256,
):
    idx = np.flatnonzero(
        d["split"] == split
    ).astype(np.int64)

    if not len(idx):
        raise RuntimeError(
            f"{rid}: no samples for split={split}"
        )

    rng = np.random.default_rng(seed)
    perm = (
        rng.permutation(len(idx))
        if control == "shuffled"
        else None
    )

    if control == "run_mean":
        sr = sd = None
        n = 0
        for st in range(0, len(idx), 512):
            R, D = c7s_origin_visual(
                rid,
                d,
                idx[st:st+512],
            )
            sr = (
                R.sum(0, dtype=np.float64)
                if sr is None
                else sr + R.sum(0, dtype=np.float64)
            )
            sd = (
                D.sum(0, dtype=np.float64)
                if sd is None
                else sd + D.sum(0, dtype=np.float64)
            )
            n += len(R)

        mr = (sr / n).astype(np.float32)
        md = (sd / n).astype(np.float32)

    model = c7s_get_model(bundle)

    S = {
        k: []
        for k in [
            "mz", "sz", "c", "td",
            "im", "isc", "tail",
        ]
    }

    for st in range(0, len(idx), batch):
        ed = min(st + batch, len(idx))
        p = idx[st:ed]

        if control == "true":
            R, D = c7s_origin_visual(rid, d, p)

        elif control == "shuffled":
            R, D = c7s_origin_visual(
                rid,
                d,
                idx[perm[st:ed]],
            )

        elif control == "run_mean":
            R = np.repeat(
                mr[None, :],
                len(p),
                axis=0,
            )
            D = np.repeat(
                md[None, :],
                len(p),
                axis=0,
            )

        else:
            raise ValueError(control)

        x = torch.from_numpy(
            np.asarray(
                d["X_core"][p],
                dtype=np.float32,
            )
        ).to(B_DEVICE)

        a = torch.from_numpy(
            np.asarray(
                d["X_action_dense"][p],
                dtype=np.float32,
            )
        ).to(B_DEVICE)

        r = torch.from_numpy(R).to(B_DEVICE)
        dp = torch.from_numpy(D).to(B_DEVICE)

        x = (
            (x - C7S_NORM_T["x_mean"][None, :])
            / C7S_NORM_T["x_std"][None, :]
        )
        a = (
            (
                a
                - C7S_NORM_T[
                    "action_mean"
                ][None, None, :]
            )
            / C7S_NORM_T[
                "action_std"
            ][None, None, :]
        )
        r = (
            (r - C7S_NORM_T["rgb_mean"][None, :])
            / C7S_NORM_T["rgb_std"][None, :]
        )
        dp = (
            (
                dp
                - C7S_NORM_T[
                    "depth_mean"
                ][None, :]
            )
            / C7S_NORM_T[
                "depth_std"
            ][None, :]
        )

        with torch.autocast(
            "cuda",
            dtype=torch.float16,
            enabled=C7S_USE_AMP,
        ):
            o = model(x, a, r, dp)

        S["mz"].append(
            o["mean_z"].float().cpu().numpy()
        )
        S["sz"].append(
            torch.exp(
                o["log_scale_z"].float()
            ).cpu().numpy()
        )
        S["c"].append(
            torch.sigmoid(
                o["contact_logits"].float()
            ).cpu().numpy()
        )
        S["td"].append(
            torch.sigmoid(
                o["touchdown_logits"].float()
            ).cpu().numpy()
        )
        S["im"].append(
            o["impulse_mean_z"]
            .float()
            .cpu()
            .numpy()
        )
        S["isc"].append(
            torch.exp(
                o["impulse_log_scale_z"].float()
            ).cpu().numpy()
        )
        S["tail"].append(
            o["tail_q90_log"]
            .float()
            .cpu()
            .numpy()
        )

    q = {
        k: np.concatenate(v, axis=0)
        for k, v in S.items()
    }

    return {
        "continuous_mean":
            (
                q["mz"]
                * B9_PREP["y_std"][None]
                + B9_PREP["y_mean"][None]
            ).astype(np.float32),

        "continuous_scale":
            (
                q["sz"]
                * B9_PREP["y_std"][None]
            ).astype(np.float32),

        "contact_probability":
            q["c"].astype(np.float32),

        "touchdown_probability":
            q["td"].astype(np.float32),

        "impulse_mean":
            (
                q["im"]
                * B9_PREP[
                    "imp_std"
                ][None, None, None, :]
                + B9_PREP[
                    "imp_mean"
                ][None, None, None, :]
            ).astype(np.float32),

        "impulse_scale":
            (
                q["isc"]
                * B9_PREP[
                    "imp_std"
                ][None, None, None, :]
            ).astype(np.float32),

        "tail_q90":
            np.maximum(
                np.expm1(q["tail"]),
                0.0,
            ).astype(np.float32),

        "truth":
            d["Y_cont"][idx],

        "contact_truth":
            d["Y_contact"][idx],

        "touchdown_truth":
            d["Y_touchdown"][idx],

        "impulse_truth":
            d["Y_impulse"][idx],

        "origins":
            d["origins"][idx],
    }


# ----------------------------------------------------------------------
# 8. SAME VALIDATION-ONLY MODE SELECTION
# ----------------------------------------------------------------------
val_ids = [
    rid
    for rid, d in C7R_DATA.items()
    if np.any(d["split"] == "val")
]

BASE2 = {
    rid: predict_B_fast(
        C7R_DATA[rid],
        "val",
    )
    for rid in val_ids
}

TRUE2 = {}
CAL2 = {}

for mode, bundle in C7S_MODELS.items():
    TRUE2[mode] = {
        rid: predict_C7S(
            bundle,
            rid,
            C7R_DATA[rid],
            "val",
            "true",
            SEED2,
        )
        for rid in val_ids
    }
    CAL2[mode] = fit_cal(
        mode,
        TRUE2[mode],
    )

rows = []

for mode in ("rgb", "depth", "rgbd"):
    for rid in val_ids:
        d = C7R_DATA[rid]
        vm = d["split"] == "val"

        rows.append({
            "mode": mode,
            "run_id": rid,
            "terrain": d["terrain"],
            "subset": "all",
            **metrics(
                TRUE2[mode][rid],
                d,
                CAL2[mode],
            ),
        })

        rows.append({
            "mode": mode,
            "run_id": rid,
            "terrain": d["terrain"],
            "subset": "precontact",
            **metrics(
                TRUE2[mode][rid],
                d,
                CAL2[mode],
                d["C_precontact"][vm],
            ),
        })

for rid in val_ids:
    d = C7R_DATA[rid]
    vm = d["split"] == "val"

    rows.append({
        "mode": "B9_baseline",
        "run_id": rid,
        "terrain": d["terrain"],
        "subset": "all",
        **metrics(
            BASE2[rid],
            d,
            B9_CALIBRATION,
        ),
    })

    rows.append({
        "mode": "B9_baseline",
        "run_id": rid,
        "terrain": d["terrain"],
        "subset": "precontact",
        **metrics(
            BASE2[rid],
            d,
            B9_CALIBRATION,
            d["C_precontact"][vm],
        ),
    })

C7S_VALIDATION = pd.DataFrame(rows)
display(C7S_VALIDATION)

base_pre = C7S_VALIDATION[
    (C7S_VALIDATION["mode"] == "B9_baseline")
    & (C7S_VALIDATION["subset"] == "precontact")
].mean(numeric_only=True)

selection_rows = []

for mode in ("rgb", "depth", "rgbd"):
    g = C7S_VALIDATION[
        (C7S_VALIDATION["mode"] == mode)
        & (C7S_VALIDATION["subset"] == "precontact")
    ].mean(numeric_only=True)

    force_gain = (
        base_pre["force_tail_component_mae"]
        - g["force_tail_component_mae"]
    ) / max(
        base_pre["force_tail_component_mae"],
        1e-9,
    )

    E_gain = (
        base_pre["E_tail_component_mae"]
        - g["E_tail_component_mae"]
    ) / max(
        base_pre["E_tail_component_mae"],
        1e-9,
    )

    score = (
        0.35 * g["risk_spearman"]
        + 0.25 * g["force_peak_spearman"]
        + 0.20 * force_gain
        + 0.10 * E_gain
        + 0.10 * g["touchdown_f1"]
    )

    selection_rows.append({
        "mode": mode,
        "validation_selection_score": float(score),
        "precontact_risk_spearman":
            float(g["risk_spearman"]),
        "precontact_force_peak_spearman":
            float(g["force_peak_spearman"]),
        "precontact_force_tail_gain_vs_B9":
            float(force_gain),
        "precontact_E_tail_gain_vs_B9":
            float(E_gain),
        "precontact_touchdown_f1":
            float(g["touchdown_f1"]),
    })

C7S_SELECTION = (
    pd.DataFrame(selection_rows)
    .sort_values(
        "validation_selection_score",
        ascending=False,
    )
    .reset_index(drop=True)
)

display(C7S_SELECTION)

C7S_SELECTED_MODE = str(
    C7S_SELECTION.iloc[0]["mode"]
)

print("C7S SELECTED:", C7S_SELECTED_MODE)

# ----------------------------------------------------------------------
# 9. SAME TRUE / SHUFFLED / RUN-MEAN ANTI-CHEATING CONTROLS
# ----------------------------------------------------------------------
control_rows = []

for rid in val_ids:
    d = C7R_DATA[rid]
    vm = d["split"] == "val"
    mask = d["C_precontact"][vm]

    control_rows.append({
        "condition": "B9_baseline",
        "run_id": rid,
        "terrain": d["terrain"],
        **metrics(
            BASE2[rid],
            d,
            B9_CALIBRATION,
            mask,
        ),
    })

    control_rows.append({
        "condition": "true",
        "run_id": rid,
        "terrain": d["terrain"],
        **metrics(
            TRUE2[C7S_SELECTED_MODE][rid],
            d,
            CAL2[C7S_SELECTED_MODE],
            mask,
        ),
    })

    seed = (
        SEED2
        + int(
            hashlib.sha256(
                rid.encode()
            ).hexdigest()[:8],
            16,
        )
        % 100000
    )

    for cond in ("shuffled", "run_mean"):
        pp = predict_C7S(
            C7S_MODELS[C7S_SELECTED_MODE],
            rid,
            d,
            "val",
            cond,
            seed,
        )

        control_rows.append({
            "condition": cond,
            "run_id": rid,
            "terrain": d["terrain"],
            **metrics(
                pp,
                d,
                CAL2[C7S_SELECTED_MODE],
                mask,
            ),
        })

C7S_CONTROLS = pd.DataFrame(control_rows)
display(C7S_CONTROLS)

C7S_SUMMARY = (
    C7S_CONTROLS
    .groupby("condition")
    .mean(numeric_only=True)
)

display(C7S_SUMMARY)

def _v(cond, metric):
    return float(
        C7S_SUMMARY.loc[cond, metric]
    )

true_risk = _v("true", "risk_spearman")
base_risk = _v("B9_baseline", "risk_spearman")
shuf_risk = _v("shuffled", "risk_spearman")
mean_risk = _v("run_mean", "risk_spearman")

true_peak = _v("true", "force_peak_spearman")
base_peak = _v(
    "B9_baseline",
    "force_peak_spearman",
)

base_tail = _v(
    "B9_baseline",
    "force_tail_component_mae",
)
true_tail = _v(
    "true",
    "force_tail_component_mae",
)

# ----------------------------------------------------------------------
# 10. GATE — EXACT SAME THRESHOLDS AS C7R
# ----------------------------------------------------------------------
C7S_GATE = {
    "selected_mode": C7S_SELECTED_MODE,
    "feature_version": C7S_FEATURE_VERSION,

    "fresh_test_ids_still_locked":
        _fresh_ids,

    "validation_precontact_true_risk_spearman":
        true_risk,

    "validation_precontact_B9_risk_spearman":
        base_risk,

    "validation_precontact_shuffled_risk_spearman":
        shuf_risk,

    "validation_precontact_runmean_risk_spearman":
        mean_risk,

    "validation_precontact_true_force_peak_spearman":
        true_peak,

    "validation_precontact_B9_force_peak_spearman":
        base_peak,

    "validation_precontact_force_tail_improvement_fraction":
        (
            base_tail - true_tail
        ) / max(base_tail, 1e-9),

    # SAME thresholds as the previous scientific gate.
    "true_beats_B9_risk":
        bool(true_risk > base_risk + 0.01),

    "true_beats_shuffled_risk":
        bool(true_risk > shuf_risk + 0.01),

    "true_beats_runmean_risk":
        bool(true_risk > mean_risk + 0.005),

    "tail_or_peak_improves":
        bool(
            true_peak > base_peak + 0.01
            or true_tail < base_tail
        ),
}

C7S_GATE["open_fresh_C_test"] = bool(
    C7S_GATE["true_beats_B9_risk"]
    and C7S_GATE["true_beats_shuffled_risk"]
    and C7S_GATE["true_beats_runmean_risk"]
    and C7S_GATE["tail_or_peak_improves"]
)

print(
    "\n"
    + "=" * 92
    + "\nC7S CAUSAL HIGH-PASS / ALIGNMENT VALIDATION GATE"
    + "\n"
    + "=" * 92
)

print(
    json.dumps(
        C7S_GATE,
        indent=2,
    )
)

C7S_VALIDATION.to_csv(
    C7S_TABLE_ROOT / "validation_metrics.csv",
    index=False,
)
C7S_SELECTION.to_csv(
    C7S_TABLE_ROOT / "mode_selection.csv",
    index=False,
)
C7S_CONTROLS.to_csv(
    C7S_TABLE_ROOT / "causality_controls.csv",
    index=False,
)
(
    C7S_TABLE_ROOT / "gate.json"
).write_text(
    json.dumps(
        C7S_GATE,
        indent=2,
    )
)

if C7S_GATE["open_fresh_C_test"]:
    print(
        "\nPASS. Fresh C test trajectories are STILL UNSCORED. "
        "STOP HERE and send me the complete C7S gate output. "
        "We will then do the one-time C8-C10 evaluation."
    )
else:
    print(
        "\nNOT PASSED. Fresh C tests remain locked. "
        "Do not weaken or bypass the gate."
    )

# %% notebook cell 6

# ======================================================================
# C7T — STRICT VISION-DEPENDENT INTERACTION REPAIR (VALIDATION ONLY)
#
# Run this AFTER the completed C7R + C7S cells in the SAME Colab runtime.
#
# ORIGINAL STAGE-C DISCIPLINE IS PRESERVED:
#   frozen B9 world model
#   same real physics targets / horizons / losses
#   same six train + two validation trajectories
#   same RGB / depth / RGB-D mode selection on validation only
#   same true / shuffled / run-mean anti-cheating controls
#   same scientific gate thresholds
#   fresh C tests P2027/P2006 remain unopened
#
# ONLY SCIENTIFIC CHANGE FROM C7S:
#   remove the non-visual residual bypass.
#
# C7S allowed:
#       residual = F([B9_latent, vision])
# so F could improve B9 while mostly ignoring vision.
#
# C7T enforces:
#       q_s = S(B9_latent)
#       q_v = V(vision), with V(0)=0
#       interaction = q_s * q_v
#       residual = O(interaction), with O(0)=0
#
# Therefore, BY ARCHITECTURE:
#       normalized vision == 0  =>  residual == 0  => prediction == frozen B9
#
# We keep the C7S causal high-pass visual features unchanged so this experiment
# isolates the FUSION/RESIDUAL ARCHITECTURE as the changed variable.
# ======================================================================

import gc, time, json, hashlib, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from IPython.display import display

# ----------------------------------------------------------------------
# 0. HARD STATE / TEST-LEAKAGE GUARDS
# ----------------------------------------------------------------------
_REQUIRED_T = [
    # Frozen Stage-B / Stage-C infrastructure
    "C_ROOT", "C7R_DATA", "TEST_ROWS",
    "B9_CKPT", "B9_PREP", "B9_CALIBRATION", "B9_DENSE",
    "BResponseModel", "B_DEVICE", "B_H", "B_STEP_WIDTH",
    "B_STEP_LAYOUT", "B_MODEL_DIM", "B_ACTION_HIDDEN",
    "B_ACTION_GATHER", "TAIL_NAMES",
    "b_loss", "predict_B_fast", "fit_cal", "metrics",
    # C7S representation + exact data protocol
    "C7S_FRAME", "C7S_VIS_NORM", "C7S_NORM_T",
    "C7SIndexDataset", "c7s_make_collate", "c7s_prep_batch",
    "c7s_origin_visual", "c7s_fresh_base",
    "C7S_TRAIN_BATCH", "C7S_FETCH_BATCH", "C7S_EPOCHS",
    "C7S_PATIENCE", "C7S_LR", "C7S_WEIGHT_DECAY",
    "C7S_RESIDUAL_L2", "C7S_NEG_NULL_WEIGHT",
    "C7S_PRECONTACT_EXTRA_WEIGHT", "C7S_USE_AMP",
    "C7S_FEATURE_VERSION",
]

_missing_t = [x for x in _REQUIRED_T if x not in globals()]
if _missing_t:
    raise RuntimeError(
        "C7T must be appended after the completed C7R + C7S runtime. "
        "Missing globals: " + repr(_missing_t)
    )

C8_MARKER_T = C_ROOT / "C8_TEST_WAS_SCORED.json"
if C8_MARKER_T.exists():
    raise RuntimeError(
        f"Fresh Stage-C test marker already exists: {C8_MARKER_T}. "
        "Do not use C7T for further tuning."
    )

C7T_FRESH_IDS = TEST_ROWS["run_id"].astype(str).tolist()

if set(C7T_FRESH_IDS) & set(C7R_DATA.keys()):
    raise RuntimeError(
        "Fresh C test trajectories unexpectedly appear in C7R_DATA. "
        "Refusing to continue."
    )

# Development data must remain exactly 6 train + 2 validation runs.
_c7t_train_runs = [
    rid for rid, d in C7R_DATA.items()
    if np.any(np.asarray(d["split"]) == "train")
]
_c7t_val_runs = [
    rid for rid, d in C7R_DATA.items()
    if np.any(np.asarray(d["split"]) == "val")
]

if len(_c7t_train_runs) != 6 or len(_c7t_val_runs) != 2:
    raise RuntimeError(
        f"Unexpected C development split. train={_c7t_train_runs}, "
        f"val={_c7t_val_runs}"
    )

print("Fresh C tests remain LOCKED:", C7T_FRESH_IDS)
print("C7T train runs:", _c7t_train_runs)
print("C7T validation runs:", _c7t_val_runs)

# ----------------------------------------------------------------------
# 1. NEW EXPERIMENT NAMESPACE
# ----------------------------------------------------------------------
C7T_SEED = 78120260921
random.seed(C7T_SEED)
np.random.seed(C7T_SEED % (2**32))
torch.manual_seed(C7T_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(C7T_SEED)

C7T_ROOT = C_ROOT / "spatial_temporal_repair_v3_strict_visual_interaction"
C7T_MODEL_ROOT = C7T_ROOT / "models"
C7T_TABLE_ROOT = C7T_ROOT / "tables"
for _p in (C7T_ROOT, C7T_MODEL_ROOT, C7T_TABLE_ROOT):
    _p.mkdir(parents=True, exist_ok=True)

# Important: C7S features are deliberately reused unchanged.
C7T_FEATURE_VERSION = (
    "C7T_strict_multiplicative_visual_interaction_v1__"
    + str(C7S_FEATURE_VERSION)
)

# Keep the SAME optimizer/data protocol as C7S.
C7T_TRAIN_BATCH = int(C7S_TRAIN_BATCH)      # 96
C7T_FETCH_BATCH = int(C7S_FETCH_BATCH)      # 384; only I/O/vectorization
C7T_EPOCHS = int(C7S_EPOCHS)
C7T_PATIENCE = int(C7S_PATIENCE)
C7T_LR = float(C7S_LR)
C7T_WEIGHT_DECAY = float(C7S_WEIGHT_DECAY)
C7T_RESIDUAL_L2 = float(C7S_RESIDUAL_L2)
C7T_NEG_NULL_WEIGHT = float(C7S_NEG_NULL_WEIGHT)
C7T_USE_AMP = bool(C7S_USE_AMP)

# Match the B9 latent width rather than changing capacity arbitrarily.
C7T_INTER_DIM = int(B_MODEL_DIM)

print({
    "experiment": C7T_FEATURE_VERSION,
    "visual_features_changed_from_C7S": False,
    "fusion_architecture_changed": True,
    "optimizer_batch": C7T_TRAIN_BATCH,
    "fetch_batch": C7T_FETCH_BATCH,
    "max_epochs": C7T_EPOCHS,
    "patience": C7T_PATIENCE,
    "lr": C7T_LR,
    "weight_decay": C7T_WEIGHT_DECAY,
    "residual_l2": C7T_RESIDUAL_L2,
    "negative_null_weight": C7T_NEG_NULL_WEIGHT,
    "fresh_tests_loaded": False,
})

# ----------------------------------------------------------------------
# 2. STRICT-ZERO BUILDING BLOCKS
#
# Every transform on the VISUAL path and every transform AFTER the
# multiplicative interaction is bias-free. Thus:
#
#     visual == 0 -> visual code == 0
#     interaction == 0
#     residual hidden == 0
#     every residual head == 0
#
# State projections MAY have bias because they are multiplied by q_v.
# ----------------------------------------------------------------------
class C7TStrictZeroVisual(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        # GELU(0)=0 and tanh(0)=0.
        return torch.tanh(self.fc2(F.gelu(self.fc1(x))))


class C7TStrictZeroMixer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim, bias=False)
        self.fc2 = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class C7TStrictVisionResidualModel(nn.Module):
    """
    Frozen B9 + strictly vision-dependent multiplicative residual.

    Horizon branch:
        z_h        = frozen B9 horizon/action-conditioned latent
        q_s(h)     = tanh(S_h(z_h))
        q_v        = V(v), V(0)=0
        g_h        = q_s(h) * q_v
        r_h        = M_h(g_h), M_h(0)=0
        output_h   = B9_h + bias-free heads(r_h)

    Sequence/tail branch:
        s          = [history_latent, final_action_latent]
        q_s_seq    = tanh(S_seq(s))
        g_seq      = q_s_seq * q_v
        r_seq      = M_seq(g_seq)
        tail       = B9_tail + bias-free head(r_seq)

    Therefore the adapter CANNOT perform a state/action-only correction.
    """
    def __init__(self, base, rgb_dim, depth_dim, mode):
        super().__init__()
        if mode not in {"rgb", "depth", "rgbd"}:
            raise ValueError(mode)

        self.mode = mode
        self.base = base.eval()

        for p in self.base.parameters():
            p.requires_grad_(False)

        if mode == "rgb":
            visual_dim = int(rgb_dim)
        elif mode == "depth":
            visual_dim = int(depth_dim)
        else:
            visual_dim = int(rgb_dim + depth_dim)

        self.visual = C7TStrictZeroVisual(
            visual_dim, C7T_INTER_DIM
        )

        # State projections are allowed bias; no bypass is possible because
        # their outputs are multiplied by the strict-zero visual code.
        self.state_h = nn.Sequential(
            nn.Linear(B_MODEL_DIM, C7T_INTER_DIM),
            nn.GELU(),
            nn.Linear(C7T_INTER_DIM, C7T_INTER_DIM),
            nn.Tanh(),
        )

        self.state_seq = nn.Sequential(
            nn.Linear(
                B_MODEL_DIM + B_ACTION_HIDDEN,
                C7T_INTER_DIM,
            ),
            nn.GELU(),
            nn.Linear(C7T_INTER_DIM, C7T_INTER_DIM),
            nn.Tanh(),
        )

        # After interaction: all bias-free => zero remains exactly zero.
        self.mix_h = C7TStrictZeroMixer(C7T_INTER_DIM)
        self.mix_seq = C7TStrictZeroMixer(C7T_INTER_DIM)

        self.d_mean = nn.Linear(
            C7T_INTER_DIM, B_STEP_WIDTH, bias=False
        )
        self.d_log_scale = nn.Linear(
            C7T_INTER_DIM, B_STEP_WIDTH, bias=False
        )
        self.d_contact = nn.Linear(
            C7T_INTER_DIM, 4, bias=False
        )
        self.d_touchdown = nn.Linear(
            C7T_INTER_DIM, 4, bias=False
        )
        self.d_impulse_mean = nn.Linear(
            C7T_INTER_DIM, 12, bias=False
        )
        self.d_impulse_log_scale = nn.Linear(
            C7T_INTER_DIM, 12, bias=False
        )
        self.d_tail = nn.Linear(
            C7T_INTER_DIM, 4, bias=False
        )

        # Start EXACTLY at B9. This is the same residual-learning philosophy
        # as earlier C repairs, but now zero vision remains B9 forever.
        for layer in [
            self.d_mean,
            self.d_log_scale,
            self.d_contact,
            self.d_touchdown,
            self.d_impulse_mean,
            self.d_impulse_log_scale,
            self.d_tail,
        ]:
            nn.init.zeros_(layer.weight)

    def _visual_input(self, rgb, depth):
        if self.mode == "rgb":
            return rgb
        if self.mode == "depth":
            return depth
        return torch.cat([rgb, depth], dim=-1)

    def forward(self, x, a, rgb, depth):
        self.base.eval()

        # Reproduce the frozen B9 forward path while retaining its latent z.
        with torch.no_grad():
            h0 = self.base.history(x)

            if self.base.use_action:
                ah = self.base.action(
                    a, B_ACTION_GATHER.to(a.device)
                )
                afinal = ah[:, -1]
            else:
                n = len(x)
                ah = torch.zeros(
                    n, B_H, B_ACTION_HIDDEN,
                    device=x.device,
                    dtype=x.dtype,
                )
                afinal = torch.zeros(
                    n, B_ACTION_HIDDEN,
                    device=x.device,
                    dtype=x.dtype,
                )

            h = h0[:, None, :].expand(-1, B_H, -1)
            he = self.base.horizon_embed[None].expand(
                len(x), -1, -1
            )

            z = self.base.fuse(
                torch.cat([h, ah, he], dim=-1)
            )

            base = {
                "mean_z":
                    self.base.mean_head(z),
                "log_scale_z":
                    torch.clamp(
                        self.base.log_scale_head(z),
                        -5.5, 2.8,
                    ),
                "contact_logits":
                    self.base.contact_head(z),
                "touchdown_logits":
                    self.base.touchdown_head(z),
                "impulse_mean_z":
                    self.base.impulse_mean_head(z).reshape(
                        -1, B_H, 4, 3
                    ),
                "impulse_log_scale_z":
                    torch.clamp(
                        self.base.impulse_log_scale_head(z).reshape(
                            -1, B_H, 4, 3
                        ),
                        -5.5, 3.0,
                    ),
                "tail_q90_log":
                    self.base.tail_q90_head(
                        torch.cat([h0, afinal], dim=-1)
                    ),
            }

        # ---------------- STRICT VISUAL INTERACTION ----------------
        v_raw = self._visual_input(rgb, depth)
        qv = self.visual(v_raw)                  # [N, D], V(0)=0

        qs_h = self.state_h(z)                   # [N,H,D]
        interaction_h = (
            qs_h * qv[:, None, :]
        )
        rh = self.mix_h(interaction_h)           # M_h(0)=0

        seq_state = torch.cat([h0, afinal], dim=-1)
        qs_seq = self.state_seq(seq_state)
        interaction_seq = qs_seq * qv
        rs = self.mix_seq(interaction_seq)       # M_seq(0)=0

        dm = self.d_mean(rh)
        dls = self.d_log_scale(rh)
        dc = self.d_contact(rh)
        dtd = self.d_touchdown(rh)
        dimp = self.d_impulse_mean(rh).reshape(
            -1, B_H, 4, 3
        )
        dil = self.d_impulse_log_scale(rh).reshape(
            -1, B_H, 4, 3
        )
        dtail = self.d_tail(rs)

        out = {
            "mean_z":
                base["mean_z"] + dm,
            "log_scale_z":
                torch.clamp(
                    base["log_scale_z"] + dls,
                    -5.5, 2.8,
                ),
            "contact_logits":
                base["contact_logits"] + dc,
            "touchdown_logits":
                base["touchdown_logits"] + dtd,
            "impulse_mean_z":
                base["impulse_mean_z"] + dimp,
            "impulse_log_scale_z":
                torch.clamp(
                    base["impulse_log_scale_z"] + dil,
                    -5.5, 3.0,
                ),
            "tail_q90_log":
                base["tail_q90_log"] + dtail,
        }

        # Full residual energy: used for regularization and wrong-time null.
        residual_terms = [
            dm,
            0.316227766 * dls,  # sqrt(.10)
            0.223606798 * dc,   # sqrt(.05)
            0.316227766 * dtd,  # sqrt(.10)
            0.5 * dimp,         # sqrt(.25)
            0.223606798 * dil,  # sqrt(.05)
            0.5 * dtail,        # sqrt(.25)
        ]

        out["_residual_full_l2"] = sum(
            t.square().mean() for t in residual_terms
        )

        out["_residual_max_abs"] = torch.stack(
            [t.abs().max() for t in residual_terms]
        ).max()

        return out


# ----------------------------------------------------------------------
# 3. FORMAL ARCHITECTURE SANITY CHECK BEFORE TRAINING
# ----------------------------------------------------------------------
@torch.no_grad()
def c7t_zero_invariant_check(model, n=8):
    model.eval()

    rid = _c7t_train_runs[0]
    d = C7R_DATA[rid]
    pos = np.flatnonzero(
        np.asarray(d["split"]) == "train"
    )[:n]

    x = torch.from_numpy(
        np.asarray(d["X_core"][pos], dtype=np.float32)
    ).to(B_DEVICE)

    a = torch.from_numpy(
        np.asarray(
            d["X_action_dense"][pos],
            dtype=np.float32,
        )
    ).to(B_DEVICE)

    x = (
        (x - C7S_NORM_T["x_mean"][None, :])
        / C7S_NORM_T["x_std"][None, :]
    )
    a = (
        (a - C7S_NORM_T["action_mean"][None, None, :])
        / C7S_NORM_T["action_std"][None, None, :]
    )

    zr = torch.zeros(
        len(pos),
        len(C7S_VIS_NORM["rgb_mean"]),
        device=B_DEVICE,
        dtype=x.dtype,
    )
    zd = torch.zeros(
        len(pos),
        len(C7S_VIS_NORM["depth_mean"]),
        device=B_DEVICE,
        dtype=x.dtype,
    )

    out = model(x, a, zr, zd)

    # This is the strongest direct test: every learned residual tensor must
    # be identically zero when normalized visual input is zero.
    residual_max = float(
        out["_residual_max_abs"].float().cpu()
    )

    # Independent external comparison against the frozen base.
    base = model.base(x, a)
    keys = [
        "mean_z",
        "log_scale_z",
        "contact_logits",
        "touchdown_logits",
        "impulse_mean_z",
        "impulse_log_scale_z",
        "tail_q90_log",
    ]
    max_diff = 0.0
    for k in keys:
        diff = float(
            (out[k] - base[k]).abs().max().float().cpu()
        )
        max_diff = max(max_diff, diff)

    return {
        "zero_visual_residual_max_abs": residual_max,
        "zero_visual_vs_B9_max_abs": max_diff,
        "pass": bool(
            residual_max <= 1e-8
            and max_diff <= 1e-6
        ),
    }


for _mode in ("rgb", "depth", "rgbd"):
    _m = C7TStrictVisionResidualModel(
        c7s_fresh_base(),
        len(C7S_VIS_NORM["rgb_mean"]),
        len(C7S_VIS_NORM["depth_mean"]),
        _mode,
    ).to(B_DEVICE)

    _chk = c7t_zero_invariant_check(_m)
    print("PRETRAIN ZERO-INVARIANT", _mode, _chk)
    if not _chk["pass"]:
        raise RuntimeError(
            f"C7T zero-vision architecture invariant failed "
            f"before training for {_mode}: {_chk}"
        )

    del _m
    gc.collect()
    torch.cuda.empty_cache()

print("C7T ARCHITECTURE ZERO-VISION INVARIANT: PASS")

# ----------------------------------------------------------------------
# 4. TRAIN / RESUME
#
# Uses EXACTLY the C7S dataset/collate/normalization:
#   same train/val samples
#   same terrain/tail/event sampling
#   same pre-contact extra TRAIN weight
#   same deterministic wrong-time same-run visual negatives
#
# The only changed variable is the residual fusion architecture above.
# ----------------------------------------------------------------------
def c7t_adapter_state(model):
    return {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if not k.startswith("base.")
    }


def c7t_train_mode(mode):
    final = C7T_MODEL_ROOT / f"C7T_{mode}_adapter.pt"
    resume = C7T_MODEL_ROOT / f"C7T_{mode}_resume.pt"

    if final.exists():
        s = torch.load(
            final,
            map_location="cpu",
            weights_only=False,
        )
        if (
            s.get("feature_version")
            == C7T_FEATURE_VERSION
        ):
            print(
                f"C7T-{mode}: reusing completed checkpoint "
                f"{final}"
            )
            return {
                "mode": mode,
                "checkpoint": str(final),
                "best_val_loss":
                    float(s["best_val_loss"]),
                "zero_invariant":
                    s["zero_invariant"],
            }

    tr = C7SIndexDataset(
        C7R_DATA, "train", balanced=True
    )
    va = C7SIndexDataset(
        C7R_DATA, "val", balanced=False
    )

    sampler = WeightedRandomSampler(
        torch.as_tensor(
            tr.sample_weight,
            dtype=torch.double,
        ),
        len(tr),
        replacement=True,
    )

    trl = DataLoader(
        tr,
        batch_size=C7T_FETCH_BATCH,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=c7s_make_collate(tr),
    )

    val = DataLoader(
        va,
        batch_size=C7T_FETCH_BATCH,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=c7s_make_collate(va),
    )

    model = C7TStrictVisionResidualModel(
        c7s_fresh_base(),
        len(C7S_VIS_NORM["rgb_mean"]),
        len(C7S_VIS_NORM["depth_mean"]),
        mode,
    ).to(B_DEVICE)

    precheck = c7t_zero_invariant_check(model)
    if not precheck["pass"]:
        raise RuntimeError(
            f"{mode}: zero invariant failed before training: "
            f"{precheck}"
        )

    params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    opt = torch.optim.AdamW(
        params,
        lr=C7T_LR,
        weight_decay=C7T_WEIGHT_DECAY,
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=C7T_USE_AMP
    )

    best = float("inf")
    best_state = None
    bad = 0
    hist = []
    start_epoch = 1

    if resume.exists():
        rr = torch.load(
            resume,
            map_location="cpu",
            weights_only=False,
        )
        if (
            rr.get("feature_version")
            == C7T_FEATURE_VERSION
            and rr.get("mode") == mode
        ):
            miss, un = model.load_state_dict(
                rr["adapter_state_dict"],
                strict=False,
            )
            bad_missing = [
                x for x in miss
                if not x.startswith("base.")
            ]
            if bad_missing or un:
                raise RuntimeError(
                    (bad_missing, un)
                )

            opt.load_state_dict(
                rr["optimizer_state"]
            )
            best = float(rr["best"])
            best_state = rr["best_state"]
            bad = int(rr["bad"])
            hist = list(rr["history"])
            start_epoch = int(rr["epoch"]) + 1

            print(
                f"C7T-{mode}: resuming at epoch "
                f"{start_epoch}"
            )

    @torch.no_grad()
    def eval_loss():
        model.eval()
        total = 0.0
        n = 0

        for raw in val:
            (
                bb, r, d, _rn, _dn
            ) = c7s_prep_batch(raw)

            with torch.autocast(
                "cuda",
                dtype=torch.float16,
                enabled=C7T_USE_AMP,
            ):
                out = model(
                    bb[0], bb[1], r, d
                )

                loss = (
                    b_loss(out, bb)
                    + C7T_RESIDUAL_L2
                    * out["_residual_full_l2"]
                )

            total += (
                float(loss.detach()) * len(bb[0])
            )
            n += len(bb[0])

        return total / max(n, 1)

    for ep in range(
        start_epoch,
        C7T_EPOCHS + 1,
    ):
        tic = time.time()
        model.train()

        total = 0.0
        n = 0

        for raw_big in trl:
            # IMPORTANT:
            # fetch batch 384 is only vectorized I/O.
            # Optimizer semantics remain the original 96 samples/step.
            for off in range(
                0,
                len(raw_big[0]),
                C7T_TRAIN_BATCH,
            ):
                raw = tuple(
                    t[off:off+C7T_TRAIN_BATCH]
                    for t in raw_big
                )

                (
                    bb,
                    r,
                    d,
                    rn,
                    dn,
                ) = c7s_prep_batch(raw)

                opt.zero_grad(set_to_none=True)

                with torch.autocast(
                    "cuda",
                    dtype=torch.float16,
                    enabled=C7T_USE_AMP,
                ):
                    out_true = model(
                        bb[0], bb[1], r, d
                    )

                    # Same C7S train-only wrong-time same-run
                    # negative. It is NOT a validation/test label.
                    out_neg = model(
                        bb[0], bb[1], rn, dn
                    )

                    loss_true = (
                        b_loss(out_true, bb)
                        + C7T_RESIDUAL_L2
                        * out_true[
                            "_residual_full_l2"
                        ]
                    )

                    # Mismatched visual evidence should not be
                    # allowed to move frozen B9.
                    loss_null = (
                        C7T_NEG_NULL_WEIGHT
                        * out_neg[
                            "_residual_full_l2"
                        ]
                    )

                    loss = (
                        loss_true + loss_null
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(opt)

                torch.nn.utils.clip_grad_norm_(
                    params, 5.0
                )

                scaler.step(opt)
                scaler.update()

                total += (
                    float(loss.detach())
                    * len(bb[0])
                )
                n += len(bb[0])

        train_loss = total / max(n, 1)
        val_loss = eval_loss()

        zchk = c7t_zero_invariant_check(model)
        if not zchk["pass"]:
            raise RuntimeError(
                f"C7T-{mode}: zero invariant was violated "
                f"after epoch {ep}: {zchk}"
            )

        hist.append({
            "mode": mode,
            "epoch": ep,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "seconds": time.time() - tic,
            **zchk,
        })

        print(
            f"C7T-{mode:5s} "
            f"ep={ep:02d} "
            f"train={train_loss:.5f} "
            f"val={val_loss:.5f} "
            f"zeroΔ={zchk['zero_visual_vs_B9_max_abs']:.2e} "
            f"time={time.time()-tic:.1f}s"
        )

        if val_loss < best - 1e-4:
            best = val_loss
            best_state = {
                k: v.clone()
                for k, v in c7t_adapter_state(
                    model
                ).items()
            }
            bad = 0
        else:
            bad += 1

        current = c7t_adapter_state(model)

        torch.save({
            "mode": mode,
            "feature_version":
                C7T_FEATURE_VERSION,
            "epoch": ep,
            "adapter_state_dict":
                current,
            "optimizer_state":
                opt.state_dict(),
            "best": best,
            "best_state":
                best_state,
            "bad": bad,
            "history": hist,
        }, resume)

        if bad >= C7T_PATIENCE:
            print(
                f"C7T-{mode}: early stop"
            )
            break

    if best_state is None:
        raise RuntimeError(
            f"C7T-{mode}: no best state found"
        )

    # Load best state before final architecture audit.
    miss, un = model.load_state_dict(
        best_state,
        strict=False,
    )
    bad_missing = [
        x for x in miss
        if not x.startswith("base.")
    ]
    if bad_missing or un:
        raise RuntimeError(
            (bad_missing, un)
        )

    postcheck = c7t_zero_invariant_check(
        model
    )
    if not postcheck["pass"]:
        raise RuntimeError(
            f"C7T-{mode}: best checkpoint violates "
            f"zero-vision invariant: {postcheck}"
        )

    torch.save({
        "mode": mode,
        "adapter_state_dict":
            best_state,
        "best_val_loss": float(best),
        "vis_norm": C7S_VIS_NORM,
        "feature_version":
            C7T_FEATURE_VERSION,
        "source_visual_feature_version":
            C7S_FEATURE_VERSION,
        "B9_checkpoint":
            str(B9_CKPT),
        "uses_test": False,
        "training_history": hist,
        "zero_invariant": postcheck,
        "train_protocol": {
            "optimizer_batch":
                C7T_TRAIN_BATCH,
            "fetch_batch":
                C7T_FETCH_BATCH,
            "negative_null_weight":
                C7T_NEG_NULL_WEIGHT,
            "residual_l2":
                C7T_RESIDUAL_L2,
            "precontact_extra_weight":
                C7S_PRECONTACT_EXTRA_WEIGHT,
        },
    }, final)

    pd.DataFrame(hist).to_csv(
        C7T_TABLE_ROOT
        / f"C7T_{mode}_history.csv",
        index=False,
    )

    del (
        model, tr, va,
        trl, val, sampler,
        opt, params,
    )
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "mode": mode,
        "checkpoint": str(final),
        "best_val_loss": float(best),
        "zero_invariant": postcheck,
    }


# Keep only one large model resident when possible.
B9_DENSE["model"].to("cpu")
if "C7S_ACTIVE" in globals():
    if C7S_ACTIVE.get("model") is not None:
        C7S_ACTIVE["model"].to("cpu")
        C7S_ACTIVE["model"] = None
        C7S_ACTIVE["checkpoint"] = None

gc.collect()
torch.cuda.empty_cache()

C7T_MODELS = {}

print(
    "\nTraining strict vision-dependent C7T adapters..."
)

for mode in ("rgb", "depth", "rgbd"):
    print(
        "\n" + "=" * 80
        + f"\nC7T MODE: {mode}"
        + "\n" + "=" * 80
    )
    C7T_MODELS[mode] = c7t_train_mode(
        mode
    )

print(
    "\nC7T TRAINING COMPLETE / RESTORED"
)

display(pd.DataFrame([
    {
        "mode": mode,
        "best_val_loss":
            bundle["best_val_loss"],
        "zero_invariant_pass":
            bundle["zero_invariant"]["pass"],
        "checkpoint":
            bundle["checkpoint"],
    }
    for mode, bundle
    in C7T_MODELS.items()
]))

# ----------------------------------------------------------------------
# 5. MODEL LOADER / PREDICTOR
# ----------------------------------------------------------------------
C7T_ACTIVE = {
    "checkpoint": None,
    "model": None,
}


def c7t_get_model(bundle):
    if (
        C7T_ACTIVE["checkpoint"]
        == bundle["checkpoint"]
        and C7T_ACTIVE["model"] is not None
    ):
        return C7T_ACTIVE["model"]

    if C7T_ACTIVE["model"] is not None:
        C7T_ACTIVE["model"].to("cpu")

    s = torch.load(
        bundle["checkpoint"],
        map_location="cpu",
        weights_only=False,
    )

    if (
        s.get("feature_version")
        != C7T_FEATURE_VERSION
    ):
        raise RuntimeError(
            "Wrong C7T checkpoint version."
        )

    m = C7TStrictVisionResidualModel(
        c7s_fresh_base(),
        len(C7S_VIS_NORM["rgb_mean"]),
        len(C7S_VIS_NORM["depth_mean"]),
        s["mode"],
    )

    miss, un = m.load_state_dict(
        s["adapter_state_dict"],
        strict=False,
    )

    bad_missing = [
        x for x in miss
        if not x.startswith("base.")
    ]

    if bad_missing or un:
        raise RuntimeError(
            (bad_missing, un)
        )

    m = m.to(B_DEVICE).eval()

    zchk = c7t_zero_invariant_check(m)
    if not zchk["pass"]:
        raise RuntimeError(
            f"Loaded C7T checkpoint violated "
            f"zero invariant: {zchk}"
        )

    C7T_ACTIVE["checkpoint"] = (
        bundle["checkpoint"]
    )
    C7T_ACTIVE["model"] = m
    return m


@torch.no_grad()
def predict_C7T(
    bundle,
    rid,
    d,
    split="val",
    control="true",
    seed=0,
    batch=256,
):
    """
    Controls:
      true      : correctly time-aligned C7S causal high-pass visual feature
      shuffled  : wrong time, same validation run
      run_mean  : one constant mean visual feature for the whole run
      zero      : normalized visual vector exactly zero

    'zero' is an architecture validity control, not used for model selection.
    """
    idx = np.flatnonzero(
        np.asarray(d["split"]) == split
    ).astype(np.int64)

    if not len(idx):
        raise RuntimeError(
            f"{rid}: no samples for split={split}"
        )

    if control not in {
        "true", "shuffled",
        "run_mean", "zero"
    }:
        raise ValueError(control)

    rng = np.random.default_rng(seed)
    perm = (
        rng.permutation(len(idx))
        if control == "shuffled"
        else None
    )

    if control == "run_mean":
        sr = sd = None
        nmean = 0

        for st in range(0, len(idx), 512):
            R, D = c7s_origin_visual(
                rid,
                d,
                idx[st:st+512],
            )

            sr = (
                R.sum(0, dtype=np.float64)
                if sr is None
                else sr
                + R.sum(0, dtype=np.float64)
            )

            sd = (
                D.sum(0, dtype=np.float64)
                if sd is None
                else sd
                + D.sum(0, dtype=np.float64)
            )

            nmean += len(R)

        mr = (sr / nmean).astype(
            np.float32
        )
        md = (sd / nmean).astype(
            np.float32
        )

    model = c7t_get_model(bundle)

    stores = {
        k: []
        for k in [
            "mz", "sz", "c", "td",
            "im", "isc", "tail",
            "resmax",
        ]
    }

    for st in range(0, len(idx), batch):
        ed = min(st + batch, len(idx))
        p = idx[st:ed]

        x = torch.from_numpy(
            np.asarray(
                d["X_core"][p],
                dtype=np.float32,
            )
        ).to(B_DEVICE)

        a = torch.from_numpy(
            np.asarray(
                d["X_action_dense"][p],
                dtype=np.float32,
            )
        ).to(B_DEVICE)

        x = (
            (x - C7S_NORM_T["x_mean"][None, :])
            / C7S_NORM_T["x_std"][None, :]
        )

        a = (
            (
                a
                - C7S_NORM_T[
                    "action_mean"
                ][None, None, :]
            )
            / C7S_NORM_T[
                "action_std"
            ][None, None, :]
        )

        # ----------------------------------------------------------
        # Visual control construction
        # ----------------------------------------------------------
        if control == "zero":
            # IMPORTANT: zero is in NORMALIZED visual coordinates.
            # We intentionally do NOT construct raw zeros and then normalize.
            r = torch.zeros(
                len(p),
                len(C7S_VIS_NORM["rgb_mean"]),
                device=B_DEVICE,
                dtype=x.dtype,
            )
            dp = torch.zeros(
                len(p),
                len(C7S_VIS_NORM["depth_mean"]),
                device=B_DEVICE,
                dtype=x.dtype,
            )

        else:
            if control == "true":
                R, D = c7s_origin_visual(
                    rid, d, p
                )

            elif control == "shuffled":
                R, D = c7s_origin_visual(
                    rid,
                    d,
                    idx[perm[st:ed]],
                )

            elif control == "run_mean":
                R = np.repeat(
                    mr[None, :],
                    len(p),
                    axis=0,
                )
                D = np.repeat(
                    md[None, :],
                    len(p),
                    axis=0,
                )

            r = torch.from_numpy(
                np.asarray(
                    R, dtype=np.float32
                )
            ).to(B_DEVICE)

            dp = torch.from_numpy(
                np.asarray(
                    D, dtype=np.float32
                )
            ).to(B_DEVICE)

            r = (
                (
                    r
                    - C7S_NORM_T[
                        "rgb_mean"
                    ][None, :]
                )
                / C7S_NORM_T[
                    "rgb_std"
                ][None, :]
            )

            dp = (
                (
                    dp
                    - C7S_NORM_T[
                        "depth_mean"
                    ][None, :]
                )
                / C7S_NORM_T[
                    "depth_std"
                ][None, :]
            )

        with torch.autocast(
            "cuda",
            dtype=torch.float16,
            enabled=C7T_USE_AMP,
        ):
            out = model(
                x, a, r, dp
            )

        stores["mz"].append(
            out["mean_z"]
            .float()
            .cpu()
            .numpy()
        )
        stores["sz"].append(
            torch.exp(
                out["log_scale_z"].float()
            ).cpu().numpy()
        )
        stores["c"].append(
            torch.sigmoid(
                out["contact_logits"].float()
            ).cpu().numpy()
        )
        stores["td"].append(
            torch.sigmoid(
                out["touchdown_logits"].float()
            ).cpu().numpy()
        )
        stores["im"].append(
            out["impulse_mean_z"]
            .float()
            .cpu()
            .numpy()
        )
        stores["isc"].append(
            torch.exp(
                out[
                    "impulse_log_scale_z"
                ].float()
            ).cpu().numpy()
        )
        stores["tail"].append(
            out["tail_q90_log"]
            .float()
            .cpu()
            .numpy()
        )
        stores["resmax"].append(
            np.asarray(
                [
                    float(
                        out[
                            "_residual_max_abs"
                        ].float().cpu()
                    )
                ],
                dtype=np.float32,
            )
        )

    q = {
        k: np.concatenate(v, axis=0)
        for k, v in stores.items()
    }

    return {
        "continuous_mean":
            (
                q["mz"]
                * B9_PREP[
                    "y_std"
                ][None, :, :]
                + B9_PREP[
                    "y_mean"
                ][None, :, :]
            ).astype(np.float32),

        "continuous_scale":
            (
                q["sz"]
                * B9_PREP[
                    "y_std"
                ][None, :, :]
            ).astype(np.float32),

        "contact_probability":
            q["c"].astype(np.float32),

        "touchdown_probability":
            q["td"].astype(np.float32),

        "impulse_mean":
            (
                q["im"]
                * B9_PREP[
                    "imp_std"
                ][None, None, None, :]
                + B9_PREP[
                    "imp_mean"
                ][None, None, None, :]
            ).astype(np.float32),

        "impulse_scale":
            (
                q["isc"]
                * B9_PREP[
                    "imp_std"
                ][None, None, None, :]
            ).astype(np.float32),

        "tail_q90":
            np.maximum(
                np.expm1(q["tail"]),
                0.0,
            ).astype(np.float32),

        "truth":
            d["Y_cont"][idx],

        "contact_truth":
            d["Y_contact"][idx],

        "touchdown_truth":
            d["Y_touchdown"][idx],

        "impulse_truth":
            d["Y_impulse"][idx],

        "origins":
            d["origins"][idx],

        "_residual_max_abs":
            float(np.max(q["resmax"])),
    }


# ----------------------------------------------------------------------
# 6. VALIDATION-ONLY CALIBRATION + MODE SELECTION
#
# Exactly the original C-series logic:
#   select RGB/depth/RGB-D from TRUE validation predictions only.
#   Anti-cheating controls do NOT choose the model.
# ----------------------------------------------------------------------
C7T_VAL_IDS = list(_c7t_val_runs)

C7T_BASE = {
    rid: predict_B_fast(
        C7R_DATA[rid], "val"
    )
    for rid in C7T_VAL_IDS
}

C7T_TRUE = {}
C7T_CAL = {}

for mode, bundle in C7T_MODELS.items():

    C7T_TRUE[mode] = {
        rid: predict_C7T(
            bundle,
            rid,
            C7R_DATA[rid],
            split="val",
            control="true",
            seed=C7T_SEED,
        )
        for rid in C7T_VAL_IDS
    }

    cal_preds = {
        rid: {
            k: v
            for k, v in pred.items()
            if not k.startswith("_")
        }
        for rid, pred in C7T_TRUE[mode].items()
    }

    C7T_CAL[mode] = fit_cal(
        mode,
        cal_preds,
    )

    print(f"Calibration ready: {mode}")

assert set(C7T_CAL.keys()) == {"rgb", "depth", "rgbd"}
print("Calibration keys:", C7T_CAL.keys())
    # Calibration must only receive ordinary prediction arrays.
# C7T diagnostic keys beginning with "_" are excluded.
_cal_preds = {
    rid: {
        k: v
        for k, v in pred.items()
        if not k.startswith("_")
    }
    for rid, pred in C7T_TRUE[mode].items()
}

C7T_CAL[mode] = fit_cal(
    mode,
    _cal_preds,
)
rows = []

for mode in ("rgb", "depth", "rgbd"):
    for rid in C7T_VAL_IDS:
        d = C7R_DATA[rid]
        vm = np.asarray(
            d["split"]
        ) == "val"

        rows.append({
            "mode": mode,
            "run_id": rid,
            "terrain": d["terrain"],
            "subset": "all",
            **metrics(
                C7T_TRUE[mode][rid],
                d,
                C7T_CAL[mode],
            ),
        })

        rows.append({
            "mode": mode,
            "run_id": rid,
            "terrain": d["terrain"],
            "subset": "precontact",
            **metrics(
                C7T_TRUE[mode][rid],
                d,
                C7T_CAL[mode],
                d["C_precontact"][vm],
            ),
        })

for rid in C7T_VAL_IDS:
    d = C7R_DATA[rid]
    vm = np.asarray(
        d["split"]
    ) == "val"

    rows.append({
        "mode": "B9_baseline",
        "run_id": rid,
        "terrain": d["terrain"],
        "subset": "all",
        **metrics(
            C7T_BASE[rid],
            d,
            B9_CALIBRATION,
        ),
    })

    rows.append({
        "mode": "B9_baseline",
        "run_id": rid,
        "terrain": d["terrain"],
        "subset": "precontact",
        **metrics(
            C7T_BASE[rid],
            d,
            B9_CALIBRATION,
            d["C_precontact"][vm],
        ),
    })

C7T_VALIDATION = pd.DataFrame(rows)

print(
    "\nC7T TRUE VALIDATION METRICS"
)
display(C7T_VALIDATION)

base_pre = C7T_VALIDATION[
    (
        C7T_VALIDATION["mode"]
        == "B9_baseline"
    )
    & (
        C7T_VALIDATION["subset"]
        == "precontact"
    )
].mean(numeric_only=True)

selection_rows = []

for mode in ("rgb", "depth", "rgbd"):
    g = C7T_VALIDATION[
        (
            C7T_VALIDATION["mode"]
            == mode
        )
        & (
            C7T_VALIDATION["subset"]
            == "precontact"
        )
    ].mean(numeric_only=True)

    force_gain = (
        base_pre[
            "force_tail_component_mae"
        ]
        - g[
            "force_tail_component_mae"
        ]
    ) / max(
        base_pre[
            "force_tail_component_mae"
        ],
        1e-9,
    )

    E_gain = (
        base_pre[
            "E_tail_component_mae"
        ]
        - g[
            "E_tail_component_mae"
        ]
    ) / max(
        base_pre[
            "E_tail_component_mae"
        ],
        1e-9,
    )

    # EXACT SAME selection formula as C7R/C7S.
    score = (
        0.35 * g["risk_spearman"]
        + 0.25
        * g["force_peak_spearman"]
        + 0.20 * force_gain
        + 0.10 * E_gain
        + 0.10 * g["touchdown_f1"]
    )

    selection_rows.append({
        "mode": mode,
        "validation_selection_score":
            float(score),
        "precontact_risk_spearman":
            float(g["risk_spearman"]),
        "precontact_force_peak_spearman":
            float(
                g[
                    "force_peak_spearman"
                ]
            ),
        "precontact_force_tail_gain_vs_B9":
            float(force_gain),
        "precontact_E_tail_gain_vs_B9":
            float(E_gain),
        "precontact_touchdown_f1":
            float(g["touchdown_f1"]),
    })

C7T_SELECTION = (
    pd.DataFrame(selection_rows)
    .sort_values(
        "validation_selection_score",
        ascending=False,
    )
    .reset_index(drop=True)
)

print(
    "\nC7T VALIDATION-ONLY MODE SELECTION"
)
display(C7T_SELECTION)

C7T_SELECTED_MODE = str(
    C7T_SELECTION.iloc[0]["mode"]
)

print(
    "C7T SELECTED MODE:",
    C7T_SELECTED_MODE,
)

# ----------------------------------------------------------------------
# 7. ANTI-CHEATING CONTROLS + ZERO-VISION ARCHITECTURE CONTROL
# ----------------------------------------------------------------------
control_rows = []
C7T_ZERO_RAW_MAX_DIFF = 0.0
C7T_ZERO_RESIDUAL_MAX = 0.0

_RAW_KEYS = [
    "continuous_mean",
    "continuous_scale",
    "contact_probability",
    "touchdown_probability",
    "impulse_mean",
    "impulse_scale",
    "tail_q90",
]

for rid in C7T_VAL_IDS:
    d = C7R_DATA[rid]
    vm = np.asarray(
        d["split"]
    ) == "val"
    mask = d["C_precontact"][vm]

    # Frozen baseline.
    control_rows.append({
        "condition": "B9_baseline",
        "run_id": rid,
        "terrain": d["terrain"],
        **metrics(
            C7T_BASE[rid],
            d,
            B9_CALIBRATION,
            mask,
        ),
    })

    # Correctly aligned visual observation.
    control_rows.append({
        "condition": "true",
        "run_id": rid,
        "terrain": d["terrain"],
        **metrics(
            C7T_TRUE[
                C7T_SELECTED_MODE
            ][rid],
            d,
            C7T_CAL[
                C7T_SELECTED_MODE
            ],
            mask,
        ),
    })

    seed = (
        C7T_SEED
        + int(
            hashlib.sha256(
                rid.encode()
            ).hexdigest()[:8],
            16,
        )
        % 100000
    )

    # Existing original anti-cheating controls.
    for cond in (
        "shuffled",
        "run_mean",
    ):
        pp = predict_C7T(
            C7T_MODELS[
                C7T_SELECTED_MODE
            ],
            rid,
            d,
            split="val",
            control=cond,
            seed=seed,
        )

        control_rows.append({
            "condition": cond,
            "run_id": rid,
            "terrain": d["terrain"],
            **metrics(
                pp,
                d,
                C7T_CAL[
                    C7T_SELECTED_MODE
                ],
                mask,
            ),
        })

    # New validity control: normalized vision exactly zero.
    pzero = predict_C7T(
        C7T_MODELS[
            C7T_SELECTED_MODE
        ],
        rid,
        d,
        split="val",
        control="zero",
        seed=seed,
    )

    C7T_ZERO_RESIDUAL_MAX = max(
        C7T_ZERO_RESIDUAL_MAX,
        float(
            pzero[
                "_residual_max_abs"
            ]
        ),
    )

    for k in _RAW_KEYS:
        aa = np.asarray(
            pzero[k],
            dtype=np.float64,
        )
        bb = np.asarray(
            C7T_BASE[rid][k],
            dtype=np.float64,
        )

        C7T_ZERO_RAW_MAX_DIFF = max(
            C7T_ZERO_RAW_MAX_DIFF,
            float(
                np.max(
                    np.abs(aa - bb)
                )
            ),
        )

    # Use B9 calibration for this validity control because the raw
    # predictions are required to equal B9; visual calibration is not
    # part of the architecture equality claim.
    control_rows.append({
        "condition": "zero_vision",
        "run_id": rid,
        "terrain": d["terrain"],
        **metrics(
            pzero,
            d,
            B9_CALIBRATION,
            mask,
        ),
    })


C7T_CONTROLS = pd.DataFrame(
    control_rows
)

print(
    "\nC7T PRE-CONTACT CAUSALITY CONTROLS"
)
display(C7T_CONTROLS)

C7T_SUMMARY = (
    C7T_CONTROLS
    .groupby("condition")
    .mean(numeric_only=True)
)

display(C7T_SUMMARY)

print({
    "zero_visual_residual_max_abs":
        C7T_ZERO_RESIDUAL_MAX,
    "zero_visual_raw_prediction_max_abs_diff_vs_B9":
        C7T_ZERO_RAW_MAX_DIFF,
})

# Tight architecture criterion. Raw denormalized arrays may show tiny AMP
# roundoff across separately executed B9 forward passes, so the raw check
# allows 1e-4 while the learned residual itself must be essentially zero.
C7T_ZERO_ARCH_PASS = bool(
    C7T_ZERO_RESIDUAL_MAX <= 1e-8
)
C7T_ZERO_RAW_PASS = bool(
    C7T_ZERO_RAW_MAX_DIFF <= 1e-4
)

if not C7T_ZERO_ARCH_PASS:
    raise RuntimeError(
        "C7T architecture invalid: zero vision generated "
        "a non-zero learned residual."
    )

if not C7T_ZERO_RAW_PASS:
    raise RuntimeError(
        "C7T architecture invalid: zero-vision predictions "
        "do not reproduce frozen B9 within numerical tolerance."
    )

print(
    "C7T ZERO-VISION FALLBACK CHECK: PASS"
)

# ----------------------------------------------------------------------
# 8. SAME SCIENTIFIC GATE AS C7R/C7S
#
# We do NOT weaken thresholds after observing R/S failures.
# Zero-vision validity is an extra architecture precondition.
# ----------------------------------------------------------------------
def _c7t_v(cond, metric):
    return float(
        C7T_SUMMARY.loc[
            cond, metric
        ]
    )


true_risk = _c7t_v(
    "true", "risk_spearman"
)
base_risk = _c7t_v(
    "B9_baseline", "risk_spearman"
)
shuf_risk = _c7t_v(
    "shuffled", "risk_spearman"
)
runmean_risk = _c7t_v(
    "run_mean", "risk_spearman"
)

true_peak = _c7t_v(
    "true", "force_peak_spearman"
)
base_peak = _c7t_v(
    "B9_baseline",
    "force_peak_spearman",
)

true_tail = _c7t_v(
    "true",
    "force_tail_component_mae",
)
base_tail = _c7t_v(
    "B9_baseline",
    "force_tail_component_mae",
)

C7T_GATE = {
    "selected_mode":
        C7T_SELECTED_MODE,

    "feature_version":
        C7T_FEATURE_VERSION,

    "source_visual_feature_version":
        C7S_FEATURE_VERSION,

    "fresh_test_ids_still_locked":
        C7T_FRESH_IDS,

    "zero_visual_residual_max_abs":
        C7T_ZERO_RESIDUAL_MAX,

    "zero_visual_raw_prediction_max_abs_diff_vs_B9":
        C7T_ZERO_RAW_MAX_DIFF,

    "zero_visual_architecture_invariant":
        C7T_ZERO_ARCH_PASS,

    "zero_visual_reproduces_B9":
        C7T_ZERO_RAW_PASS,

    "validation_precontact_true_risk_spearman":
        true_risk,

    "validation_precontact_B9_risk_spearman":
        base_risk,

    "validation_precontact_shuffled_risk_spearman":
        shuf_risk,

    "validation_precontact_runmean_risk_spearman":
        runmean_risk,

    "validation_precontact_true_force_peak_spearman":
        true_peak,

    "validation_precontact_B9_force_peak_spearman":
        base_peak,

    "validation_precontact_force_tail_improvement_fraction":
        (
            base_tail - true_tail
        ) / max(base_tail, 1e-9),

    # EXACT SAME CAUSAL THRESHOLDS.
    "true_beats_B9_risk":
        bool(
            true_risk
            > base_risk + 0.01
        ),

    "true_beats_shuffled_risk":
        bool(
            true_risk
            > shuf_risk + 0.01
        ),

    "true_beats_runmean_risk":
        bool(
            true_risk
            > runmean_risk + 0.005
        ),

    "tail_or_peak_improves":
        bool(
            true_peak
            > base_peak + 0.01
            or true_tail < base_tail
        ),
}

C7T_GATE["open_fresh_C_test"] = bool(
    C7T_GATE[
        "zero_visual_architecture_invariant"
    ]
    and C7T_GATE[
        "zero_visual_reproduces_B9"
    ]
    and C7T_GATE[
        "true_beats_B9_risk"
    ]
    and C7T_GATE[
        "true_beats_shuffled_risk"
    ]
    and C7T_GATE[
        "true_beats_runmean_risk"
    ]
    and C7T_GATE[
        "tail_or_peak_improves"
    ]
)

print(
    "\n"
    + "=" * 96
    + "\nC7T STRICT VISION-INTERACTION VALIDATION GATE"
    + "\n"
    + "=" * 96
)

print(
    json.dumps(
        C7T_GATE,
        indent=2,
    )
)

# ----------------------------------------------------------------------
# 9. PERSIST EVERYTHING NEEDED FOR AUDIT / RESUME
# ----------------------------------------------------------------------
C7T_VALIDATION.to_csv(
    C7T_TABLE_ROOT
    / "validation_metrics.csv",
    index=False,
)

C7T_SELECTION.to_csv(
    C7T_TABLE_ROOT
    / "mode_selection.csv",
    index=False,
)

C7T_CONTROLS.to_csv(
    C7T_TABLE_ROOT
    / "causality_controls.csv",
    index=False,
)

(
    C7T_TABLE_ROOT / "gate.json"
).write_text(
    json.dumps(
        C7T_GATE,
        indent=2,
    )
)

# Save selected calibration separately without touching any test data.
def _jsonable_t(obj):
    if isinstance(obj, dict):
        return {
            str(k): _jsonable_t(v)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [
            _jsonable_t(v) for v in obj
        ]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(
        obj,
        (np.floating, np.integer),
    ):
        return obj.item()
    return obj

(
    C7T_TABLE_ROOT
    / "selected_validation_calibration.json"
).write_text(
    json.dumps(
        _jsonable_t(
            C7T_CAL[
                C7T_SELECTED_MODE
            ]
        ),
        indent=2,
    )
)

(
    C7T_TABLE_ROOT
    / "selected_model.json"
).write_text(
    json.dumps(
        {
            "selected_mode":
                C7T_SELECTED_MODE,
            "checkpoint":
                C7T_MODELS[
                    C7T_SELECTED_MODE
                ]["checkpoint"],
            "feature_version":
                C7T_FEATURE_VERSION,
            "B9_checkpoint":
                str(B9_CKPT),
            "fresh_tests_scored":
                False,
        },
        indent=2,
    )
)

# ----------------------------------------------------------------------
# 10. STOP HERE — NEVER AUTO-OPEN THE TEST SET
# ----------------------------------------------------------------------
if C7T_GATE["open_fresh_C_test"]:
    print(
        "\nC7T VALIDATION GATE PASSED.\n"
        "Fresh C tests are STILL UNSCORED and were NOT loaded here.\n"
        "STOP NOW and send the complete C7T gate output. "
        "The next step is the one-time C8-C10 evaluation."
    )
else:
    print(
        "\nC7T VALIDATION GATE NOT PASSED.\n"
        "Fresh C tests remain locked. "
        "Do not weaken, bypass, or inspect C8 labels."
    )

# %% notebook cell 7
# ================================================================
# C7T FINAL ZERO-VISION AUDIT + FINAL GATE
# No training. No fresh-test loading.
# ================================================================

import json
import numpy as np
import torch

@torch.no_grad()
def c7t_full_zero_same_model_audit():

    bundle = C7T_MODELS[C7T_SELECTED_MODE]
    model = c7t_get_model(bundle).eval()

    keys = [
        "mean_z",
        "log_scale_z",
        "contact_logits",
        "touchdown_logits",
        "impulse_mean_z",
        "impulse_log_scale_z",
        "tail_q90_log",
    ]

    max_diff = {k: 0.0 for k in keys}
    residual_max = 0.0

    for rid in C7T_VAL_IDS:

        d = C7R_DATA[rid]
        idx = np.flatnonzero(
            np.asarray(d["split"]) == "val"
        ).astype(np.int64)

        for st in range(0, len(idx), 256):

            p = idx[st:st+256]

            x = torch.from_numpy(
                np.asarray(
                    d["X_core"][p],
                    dtype=np.float32
                )
            ).to(B_DEVICE)

            a = torch.from_numpy(
                np.asarray(
                    d["X_action_dense"][p],
                    dtype=np.float32
                )
            ).to(B_DEVICE)

            x = (
                x
                - C7S_NORM_T["x_mean"][None, :]
            ) / C7S_NORM_T["x_std"][None, :]

            a = (
                a
                - C7S_NORM_T[
                    "action_mean"
                ][None, None, :]
            ) / C7S_NORM_T[
                "action_std"
            ][None, None, :]

            zr = torch.zeros(
                len(p),
                len(C7S_VIS_NORM["rgb_mean"]),
                device=B_DEVICE,
                dtype=x.dtype,
            )

            zd = torch.zeros(
                len(p),
                len(C7S_VIS_NORM["depth_mean"]),
                device=B_DEVICE,
                dtype=x.dtype,
            )

            with torch.autocast(
                "cuda",
                dtype=torch.float16,
                enabled=C7T_USE_AMP,
            ):

                out = model(
                    x, a, zr, zd
                )

                base = model.base(
                    x, a
                )

            residual_max = max(
                residual_max,
                float(
                    out[
                        "_residual_max_abs"
                    ].float().cpu()
                )
            )

            for k in keys:

                md = float(
                    (
                        out[k]
                        - base[k]
                    )
                    .abs()
                    .max()
                    .float()
                    .cpu()
                )

                max_diff[k] = max(
                    max_diff[k], md
                )

    overall = max(
        max_diff.values()
    )

    return {
        "residual_max_abs":
            residual_max,

        "per_output_max_abs_diff":
            max_diff,

        "overall_same_model_max_abs_diff":
            overall,

        "pass":
            bool(
                residual_max <= 1e-8
                and overall <= 1e-6
            ),
    }


ZERO_AUDIT = (
    c7t_full_zero_same_model_audit()
)

print(
    "\nC7T FULL ZERO-VISION SAME-MODEL AUDIT"
)
print(
    json.dumps(
        ZERO_AUDIT,
        indent=2,
    )
)

if not ZERO_AUDIT["pass"]:
    raise RuntimeError(
        "True C7T zero-vision invariant failed."
    )


def V(cond, metric):
    return float(
        C7T_SUMMARY.loc[
            cond, metric
        ]
    )


true_risk = V(
    "true",
    "risk_spearman"
)

base_risk = V(
    "B9_baseline",
    "risk_spearman"
)

shuf_risk = V(
    "shuffled",
    "risk_spearman"
)

runmean_risk = V(
    "run_mean",
    "risk_spearman"
)

true_peak = V(
    "true",
    "force_peak_spearman"
)

base_peak = V(
    "B9_baseline",
    "force_peak_spearman"
)

true_tail = V(
    "true",
    "force_tail_component_mae"
)

base_tail = V(
    "B9_baseline",
    "force_tail_component_mae"
)


C7T_GATE = {

    "selected_mode":
        C7T_SELECTED_MODE,

    "fresh_test_ids_still_locked":
        C7T_FRESH_IDS,

    "zero_visual_residual_max_abs":
        ZERO_AUDIT[
            "residual_max_abs"
        ],

    "zero_visual_same_model_max_abs_diff":
        ZERO_AUDIT[
            "overall_same_model_max_abs_diff"
        ],

    "zero_visual_architecture_invariant":
        ZERO_AUDIT["pass"],

    "validation_precontact_true_risk_spearman":
        true_risk,

    "validation_precontact_B9_risk_spearman":
        base_risk,

    "validation_precontact_shuffled_risk_spearman":
        shuf_risk,

    "validation_precontact_runmean_risk_spearman":
        runmean_risk,

    "validation_precontact_true_force_peak_spearman":
        true_peak,

    "validation_precontact_B9_force_peak_spearman":
        base_peak,

    "validation_precontact_force_tail_improvement_fraction":
        (
            base_tail - true_tail
        ) / max(
            base_tail, 1e-9
        ),

    # SAME ORIGINAL GATE
    "true_beats_B9_risk":
        bool(
            true_risk
            > base_risk + 0.01
        ),

    "true_beats_shuffled_risk":
        bool(
            true_risk
            > shuf_risk + 0.01
        ),

    "true_beats_runmean_risk":
        bool(
            true_risk
            > runmean_risk + 0.005
        ),

    "tail_or_peak_improves":
        bool(
            true_peak
            > base_peak + 0.01
            or true_tail < base_tail
        ),
}


C7T_GATE[
    "open_fresh_C_test"
] = bool(

    C7T_GATE[
        "zero_visual_architecture_invariant"
    ]

    and C7T_GATE[
        "true_beats_B9_risk"
    ]

    and C7T_GATE[
        "true_beats_shuffled_risk"
    ]

    and C7T_GATE[
        "true_beats_runmean_risk"
    ]

    and C7T_GATE[
        "tail_or_peak_improves"
    ]
)


print(
    "\n"
    + "=" * 96
)

print(
    "C7T STRICT VISION-INTERACTION "
    "VALIDATION GATE"
)

print(
    "=" * 96
)

print(
    json.dumps(
        C7T_GATE,
        indent=2,
    )
)


(
    C7T_TABLE_ROOT
    / "gate.json"
).write_text(
    json.dumps(
        C7T_GATE,
        indent=2,
    )
)


if C7T_GATE[
    "open_fresh_C_test"
]:

    print(
        "\nPASS — fresh tests remain "
        "unscored until explicit C8."
    )

else:

    print(
        "\nNOT PASSED — P2027/P2006 "
        "remain locked."
    )
