# Source inspection export of the adjacent notebook.
# Notebook execution order and dependencies still apply.
# IPython magic, shell lines and prose-only conclusion cells are comments.

# %% notebook cell 1
# ======================================================================
# 0. FRESH-COLAB SETUP — LIGHTWEIGHT, NO FOUNDATION MODEL DOWNLOADS
# ======================================================================
import sys, subprocess, os
from pathlib import Path

def pip_install(*pkgs):
    subprocess.check_call([sys.executable,'-m','pip','install','-q',*pkgs])

# Colab usually already has these; this is intentionally far lighter than V7 setup.
pip_install('numpy','pandas','scipy','scikit-learn','matplotlib','pillow')

# Official ANYmal-D description is needed only for verified leg FK.
EXT=Path('/content/v8_external'); EXT.mkdir(parents=True,exist_ok=True)
ANYMAL_REPO=EXT/'anymal_d_simple_description'
if not ANYMAL_REPO.exists():
    subprocess.check_call(['git','clone','--depth','1','https://github.com/ANYbotics/anymal_d_simple_description.git',str(ANYMAL_REPO)])
ANYMAL_D_URDF=ANYMAL_REPO/'urdf'/'anymal.urdf'
assert ANYMAL_D_URDF.exists(), ANYMAL_D_URDF

print('Setup complete. No DINO/ConvNeXt/Utonia/WAFT model was downloaded.')
print('ANYmal-D URDF:',ANYMAL_D_URDF)

# %% notebook cell 2
# ======================================================================
# 1. CONFIGURATION + DRIVE + ARTIFACT DISCOVERY
# ======================================================================
import os, re, gc, math, random, copy, time, hashlib, warnings, contextlib, json, zipfile
from pathlib import Path
from collections import defaultdict
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from scipy.spatial import cKDTree, ConvexHull
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from sklearn.metrics import precision_recall_fscore_support

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from google.colab import drive
drive.mount('/content/drive')

DRIVE_PROJECT_DIR='MyDrive/UGP_V7_FULL4B'
PERSIST_ROOT=Path('/content/drive')/DRIVE_PROJECT_DIR
CACHE_ROOT=PERSIST_ROOT/'cache'
OUT_ROOT=PERSIST_ROOT/'results'
V8_ROOT=PERSIST_ROOT/'V8_contactmap_standalone'
V8_ROOT.mkdir(parents=True,exist_ok=True)

DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type!='cuda':
    raise RuntimeError('Use a GPU Colab runtime. V8-A re-encodes cached four-stream tokens through the trained V7 adapters.')

SEED=781
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# Exact V7 architectural constants required only to reconstruct the trained terrain encoder.
SHORT_CONTEXT_SEC=0.5
RAW_GRID=8; RAW_TOKENS=64; LATENTS_PER_BRANCH=16; VIS_DIM=256
FLOW_HISTORY=4; GEOM_REP_POINTS=64; PI_PHYS_DIM=47; N_REGIMES=4
FX=320.0; FY=320.0; CX=320.0; CY=320.0

EXPERIMENTS=[
    dict(terrain_type='forest',environment='ForestEnv',run='P2000',ttests=[28.0,45.0,75.0,90.0]),
    dict(terrain_type='swamp_marsh',environment='GreatMarsh',run='P2023',ttests=[47.0,61.0,73.0,87.0]),
    dict(terrain_type='rough_uneven',environment='SeasonalForestWinter',run='P2000',ttests=[29.0,46.0,64.0,92.0]),
    dict(terrain_type='wet_uneven_proxy',environment='OldTownFall',run='P2000',ttests=[32.0,84.0,100.0,115.0]),
]

# User-level science targets. These are GOALS, not claims about current performance.
FORCE_VECTOR_TARGET_N=5.0
TORQUE_COMPONENT_TARGET_NM=2.0
ENFORCE_V7F_MASTER_GATE=False  # True = stop before V8-C/E/F if V7-F strict force gate is missed.

print('DEVICE:',DEVICE,torch.cuda.get_device_name(0))
print('PERSIST_ROOT:',PERSIST_ROOT)
print('V8_ROOT:',V8_ROOT)

# %% notebook cell 3
# ======================================================================
# 2. CACHE / CHECKPOINT PREFLIGHT + OPTIONAL ZIP RECOVERY
# ======================================================================
def _extract_member_from_zip(zip_path, member_suffix, dest_dir):
    zip_path=Path(zip_path); dest_dir=Path(dest_dir); dest_dir.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        matches=[n for n in z.namelist() if n.endswith(member_suffix)]
        if not matches: return None
        n=matches[0]
        out=dest_dir/Path(n).name
        with z.open(n) as src, open(out,'wb') as dst: dst.write(src.read())
        return out

def _find_zip(pattern):
    cands=list(PERSIST_ROOT.glob(pattern))+list(PERSIST_ROOT.parent.glob(pattern))+list(Path('/content').glob(pattern))
    return cands[0] if cands else None

# Check mandatory cache contracts.
missing=[]
for e in EXPERIMENTS:
    env,run=e['environment'],e['run']
    req=[
        CACHE_ROOT/f'{env}_{run}_physics.npz',
        CACHE_ROOT/f'{env}_{run}_frame_manifest.npz',
        CACHE_ROOT/f'{env}_{run}_v7_dinov3.npy',
        CACHE_ROOT/f'{env}_{run}_v7_convnext.npy',
        CACHE_ROOT/f'{env}_{run}_v7_utonia.npy',
        CACHE_ROOT/f'{env}_{run}_v7_waft.npy',
    ]
    missing += [p for p in req if not p.exists()]

if missing:
    print('MISSING mandatory V7 persisted caches:')
    for p in missing: print(' ',p)
    raise RuntimeError('Fresh V8 cannot truthfully recreate the four-stream terrain memory without these persisted caches. Do not substitute a fake branch.')

V7_SHORT_CKPT=OUT_ROOT/'V7_PRIOR_SHORT.pt'
if not V7_SHORT_CKPT.exists():
    z=_find_zip('V7_RESULTS_ONLY*.zip')
    if z:
        print('Recovering V7 short prior from',z)
        p=_extract_member_from_zip(z,'V7_PRIOR_SHORT.pt','/content/v8_restore')
        if p: V7_SHORT_CKPT=Path(p)
if not V7_SHORT_CKPT.exists():
    raise RuntimeError('V7_PRIOR_SHORT.pt missing. Put V7_RESULTS_ONLY*.zip in MyDrive/UGP_V7_FULL4B or restore the results folder.')

print('All 4x4 visual caches + physics/manifests found.')
print('V7 short checkpoint:',V7_SHORT_CKPT)

# %% notebook cell 5
# ======================================================================
# 3. V7-F MASTERY AUDIT FROM SAVED RESULTS
# ======================================================================
def locate_v7f_csv():
    direct=PERSIST_ROOT/'V7F_force_specialist'/'V7F_ALL_16_TESTS_ALL_SENSOR_VALUES.csv'
    if direct.exists(): return direct
    for root in [PERSIST_ROOT,PERSIST_ROOT.parent,Path('/content')]:
        for z in root.glob('V7F_FORCE_SPECIALIST_RESULTS*.zip'):
            out=_extract_member_from_zip(z,'V7F_ALL_16_TESTS_ALL_SENSOR_VALUES.csv','/content/v8_restore/v7f')
            if out: return Path(out)
    return None

V7F_CSV=locate_v7f_csv()
V7F_MASTER_FORCE_GATE=False
V7F_AUDIT=None
if V7F_CSV is None:
    print('V7-F result CSV not found. V8 can still run, but Stage-0 audit is skipped.')
else:
    d=pd.read_csv(V7F_CSV)
    force=d[d.signal.str.match(r'^(FL|FR|RL|RR)\\.F[xyz]$')].copy()
    force['leg']=force.signal.str.split('.').str[0]
    force['axis']=force.signal.str[-1]

    # Component metrics.
    comp=force.groupby('horizon_s').agg(
        component_MAE_N=('abs_error','mean'),
        component_p95_N=('abs_error',lambda s:float(np.quantile(s,.95))),
        component_max_N=('abs_error','max'),
        component_fraction_within5=('abs_error',lambda s:float((s<=FORCE_VECTOR_TARGET_N).mean()))
    ).reset_index()

    # Assemble one 3-D vector per test/horizon/leg.
    keys=['terrain_type','test_id','ttest_s','horizon_s','leg']
    pp=force.pivot_table(index=keys,columns='axis',values='prediction').reset_index()
    gg=force.pivot_table(index=keys,columns='axis',values='ground_truth').reset_index()
    v=pp[keys].copy()
    pred=pp[['x','y','z']].to_numpy(float); gt=gg[['x','y','z']].to_numpy(float)
    v['vector_error_N']=np.linalg.norm(pred-gt,axis=1)
    v['magnitude_error_N']=np.abs(np.linalg.norm(pred,axis=1)-np.linalg.norm(gt,axis=1))
    pm=np.linalg.norm(pred,axis=1); gm=np.linalg.norm(gt,axis=1)
    loaded=(pm>=10)&(gm>=10)
    cos=np.full(len(v),np.nan); cos[loaded]=np.sum(pred[loaded]*gt[loaded],axis=1)/(pm[loaded]*gm[loaded]+1e-9)
    cos=np.clip(cos,-1,1); v['direction_cosine']=cos; v['direction_angle_deg']=np.degrees(np.arccos(cos))
    v['vector_within5']=v.vector_error_N<=FORCE_VECTOR_TARGET_N
    vec=v.groupby('horizon_s').agg(
        vector_mean_N=('vector_error_N','mean'),vector_p95_N=('vector_error_N',lambda s:float(np.quantile(s,.95))),vector_max_N=('vector_error_N','max'),
        vector_fraction_within5=('vector_within5','mean'),magnitude_MAE_N=('magnitude_error_N','mean'),
        direction_angle_median_deg=('direction_angle_deg','median'),direction_angle_p95_deg=('direction_angle_deg',lambda s:float(np.nanquantile(s,.95)))
    ).reset_index()

    tau=d[d.signal.str.match(r'^(FL|FR|RL|RR)\\.tau_')].copy()
    tau_summary=tau.groupby('horizon_s').agg(tau_MAE_Nm=('abs_error','mean'),tau_p95_Nm=('abs_error',lambda s:float(np.quantile(s,.95))),tau_max_Nm=('abs_error','max')).reset_index()
    audit=comp.merge(vec,on='horizon_s').merge(tau_summary,on='horizon_s')
    V7F_AUDIT=audit
    display(audit)

    focus=audit[audit.horizon_s.isin([0.1,1.0])]
    # Literal interpretation of "maximum ±5N per leg": every evaluated 3-D leg-force vector <=5 N error.
    V7F_MASTER_FORCE_GATE=bool((focus.vector_max_N<=FORCE_VECTOR_TARGET_N).all())
    print('\nStrict ±5 N vector-error mastery gate:',V7F_MASTER_FORCE_GATE)
    if not V7F_MASTER_FORCE_GATE:
        print('Important: V7-F beats V5 in mean MAE, but it has NOT yet mastered the much stricter ±5 N-per-leg maximum-error goal.')
        print('V8-A/B can still proceed. Treat V8-E as shadow research until the contact-response gate improves.')

# %% notebook cell 7
# ======================================================================
# 4. MINIMAL V7 ARCHITECTURE + VERIFIED ANYMAL-D LEG FK
# ======================================================================
from scipy.spatial.transform import Rotation

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

# %% notebook cell 8
# ======================================================================
# 5. CACHE-ONLY V7 RESTORE — NO ROSBAG, NO RAW RGB-D, NO TRAINING
# ======================================================================
def _npz_dict(path):
    z=np.load(path,allow_pickle=False)
    return {k:z[k] for k in z.files}

def _visual_bundle(env,run,manifest):
    paths={
        'DINO':CACHE_ROOT/f'{env}_{run}_v7_dinov3.npy',
        'CONV':CACHE_ROOT/f'{env}_{run}_v7_convnext.npy',
        'GEOM':CACHE_ROOT/f'{env}_{run}_v7_utonia.npy',
        'FLOW':CACHE_ROOT/f'{env}_{run}_v7_waft.npy',
    }
    out={k:np.load(p,mmap_mode='r') for k,p in paths.items()}
    out.update(manifest)
    N=len(manifest['frame_ids'])
    for k in ['DINO','CONV','GEOM','FLOW']:
        assert len(out[k])==N, (k,len(out[k]),N)
        assert np.isfinite(np.asarray(out[k][0],np.float32)).all(), k
    return out

run_records=[]
for e in EXPERIMENTS:
    env,run=e['environment'],e['run']
    data=_npz_dict(CACHE_ROOT/f'{env}_{run}_physics.npz')
    manifest=_npz_dict(CACHE_ROOT/f'{env}_{run}_frame_manifest.npz')
    visual=_visual_bundle(env,run,manifest)
    rec=dict(**e,data=data,manifest=manifest,visual=visual)
    run_records.append(rec)
    print(e['terrain_type'], 'P',data['P'].shape,'B',data['B'].shape,'U',data['U'].shape,
          '| DINO',visual['DINO'].shape,'CONV',visual['CONV'].shape,'GEOM',visual['GEOM'].shape,'FLOW',visual['FLOW'].shape)

rec0=run_records[0]
dt=float(np.median(np.diff(rec0['data']['time'])))
Hs=int(round(SHORT_CONTEXT_SEC/dt))
prior_short=V7FullDynamics(Hs,1,rec0['visual']['DINO'].shape[-1],rec0['visual']['CONV'].shape[-1],rec0['visual']['GEOM'].shape[-1]).to(DEVICE)
try:
    sd=torch.load(V7_SHORT_CKPT,map_location=DEVICE,weights_only=True)
except TypeError:
    sd=torch.load(V7_SHORT_CKPT,map_location=DEVICE)
prior_short.load_state_dict(sd,strict=True)
prior_short.eval()
for p in prior_short.parameters(): p.requires_grad_(False)

print('\nV7 short prior restored, frozen, and ready for V8-A.')
print('history steps:',Hs,'parameters:',sum(p.numel() for p in prior_short.parameters()))

# %% notebook cell 10
# ======================================================================
# V8-A — PERSISTENT FOUR-STREAM CONTACT MEMORY
# STANDALONE CELL. Uses the V7 state restored earlier in this notebook.
#
# What it does:
#   1) Reuses the trained V7 four-stream adapters (DINOv3/ConvNeXt/Utonia/WAFT).
#   2) Converts the 64 Utonia metric anchors in every frame to world coordinates.
#   3) Associates each 3-D anchor with the matching 8x8 DINO/Conv/WAFT token.
#   4) Builds a CAUSAL event memory: (world cell, time, 256-D token, geometry, confidence).
#   5) Benchmarks whether future/contact foot locations were already observed earlier.
#
# Important scientific note:
#   TartanGround provides ground-truth camera poses. For the offline foot-location
#   proxy below, the front-camera optical center is used as the robot reference
#   origin and ANYmal-D FK is expressed in the camera-heading frame. This is good
#   for the dataset memory experiment, but exact hardware deployment MUST replace
#   this with the verified camera<->base extrinsic.
# ======================================================================

import os, json, math, gc, hashlib, itertools
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

# --------------------------- preflight ---------------------------
_req = [
    'run_records','prior_short','ANYMAL_D_FK','heading_frame_from_pose',
    'CACHE_ROOT','DEVICE','RAW_GRID','FX','FY','CX','CY'
]
_missing=[x for x in _req if x not in globals()]
if _missing:
    raise RuntimeError('V8-A standalone restore is incomplete. Missing: '+repr(_missing))

# V8_ROOT is defined by the standalone configuration cell.
V8A_ROOT = V8_ROOT/'A_memory'
V8A_ROOT.mkdir(parents=True,exist_ok=True)

V8_CELL_RES_M = 0.05
V8_MEMORY_QUERY_RADIUS_M = 0.18
V8_MEMORY_KNN = 48
V8A_BATCH = 24
V8A_TOKEN_DIM = 256
V8A_MIN_CONF = 0.02

# Camera-origin -> ANYmal base-origin translation in camera-heading coordinates.
# Keep ZERO for TartanGround offline research prototype unless a verified extrinsic
# is supplied. Exact robot deployment must replace this value.
V8_BASE_IN_CAM_HEADING = np.array([0.0,0.0,0.0],np.float32)
V8_ALIGNMENT_MODE = 'camera_origin_proxy__replace_with_verified_T_C_B_for_hardware'


def _safe_norm(x,axis=-1,eps=1e-8):
    n=np.linalg.norm(x,axis=axis,keepdims=True)
    return x/np.maximum(n,eps)


def _data_camera_pose(rec,idx):
    vi=int(rec['manifest']['visual_index'][idx])
    if vi<0: return None,None
    return rec['manifest']['poses'][vi].astype(np.float64),vi


def _world_from_heading_local(local_xyz,pose7):
    # heading_frame_from_pose returns R_lw with rows = local basis in world.
    Rlw,_=heading_frame_from_pose(pose7)
    local=np.asarray(local_xyz,np.float64)
    return np.asarray(pose7[:3],np.float64) + local @ Rlw.astype(np.float64)


def _heading_local_from_world(world_xyz,pose7):
    Rlw,_=heading_frame_from_pose(pose7)
    dv=np.asarray(world_xyz,np.float64)-np.asarray(pose7[:3],np.float64)
    return dv @ Rlw.astype(np.float64).T


def v8_foot_world_proxy(rec,idx):
    """Offline dataset proxy for four foot world positions, shape (4,3)."""
    pose,vi=_data_camera_pose(rec,idx)
    if pose is None: return None
    pB=ANYMAL_D_FK.foot_positions(rec['data']['P'][idx,:,0:3]).astype(np.float32)
    # URDF: x forward, y left, z up. Camera-heading local: x forward, y right, z up.
    pl=np.stack([pB[:,0],-pB[:,1],pB[:,2]],axis=-1)+V8_BASE_IN_CAM_HEADING[None]
    return _world_from_heading_local(pl,pose).astype(np.float32)


def _project_heading_local_to_grid(local_xyz,pose7,crop_box):
    """Map Utonia heading-frame anchors to the 8x8 image-token grid."""
    local=np.asarray(local_xyz,np.float64)
    Rlw,Rwc=heading_frame_from_pose(pose7)
    world_vec=local @ Rlw.astype(np.float64)
    cam=world_vec @ Rwc.astype(np.float64)   # because world_vec = cam @ Rwc.T
    x=cam[:,0]; y=cam[:,1]; z=cam[:,2]
    u=CX + FX*y/np.maximum(x,1e-5)
    v=CY + FY*z/np.maximum(x,1e-5)
    x0,y0,x1,y1=[float(a) for a in crop_box]
    gx=np.floor((u-x0)/max(x1-x0,1.0)*RAW_GRID).astype(np.int64)
    gy=np.floor((v-y0)/max(y1-y0,1.0)*RAW_GRID).astype(np.int64)
    valid=(x>0)&(gx>=0)&(gx<RAW_GRID)&(gy>=0)&(gy<RAW_GRID)
    gx=np.clip(gx,0,RAW_GRID-1); gy=np.clip(gy,0,RAW_GRID-1)
    return gy*RAW_GRID+gx,valid


@torch.inference_mode()
def _encode_local_fourstream_batch(visual,manifest,start,end):
    """Return Bx64x256 cell-local tokens using the TRAINED V7 branch adapters."""
    enc=prior_short.terrain_encoder
    d=torch.as_tensor(np.asarray(visual['DINO'][start:end],np.float32),device=DEVICE)
    c=torch.as_tensor(np.asarray(visual['CONV'][start:end],np.float32),device=DEVICE)
    g=torch.as_tensor(np.asarray(visual['GEOM'][start:end],np.float32),device=DEVICE)
    fl=torch.as_tensor(np.asarray(visual['FLOW'][start:end],np.float32),device=DEVICE)
    de=enc.dino.adapter(d)+enc.modality[:,0]
    ce=enc.conv.adapter(c)+enc.modality[:,1]
    ge=enc.geom.adapter(g)+enc.modality[:,2]
    fe=enc.flow.adapter(enc.flow_temporal(fl))+enc.modality[:,3]
    return de,ce,ge,fe


def _build_memory_events_for_run(rec):
    env,run=rec['environment'],rec['run']
    out=V8A_ROOT/f'{env}_{run}_V8A_MEMORY_EVENTS.npz'
    meta=V8A_ROOT/f'{env}_{run}_V8A_MEMORY_META.json'
    if out.exists() and meta.exists():
        z=np.load(out,mmap_mode='r',allow_pickle=False)
        print('Reusing V8-A memory',env,run,'events=',len(z['time']))
        return dict(path=out,meta=json.loads(meta.read_text()))

    vis=rec['visual']; man=rec['manifest']; N=len(man['frame_ids'])
    all_xy=[]; all_z=[]; all_t=[]; all_tok=[]; all_n=[]; all_r=[]; all_conf=[]; all_frame=[]; all_grid=[]
    print(f'Building V8-A causal world memory {env}/{run}: {N} frames')

    for s in range(0,N,V8A_BATCH):
        e=min(N,s+V8A_BATCH)
        de,ce,ge,fe=_encode_local_fourstream_batch(vis,man,s,e)
        de=de.detach().cpu().numpy(); ce=ce.detach().cpu().numpy(); ge=ge.detach().cpu().numpy(); fe=fe.detach().cpu().numpy()
        rawg=np.asarray(vis['GEOM'][s:e],np.float32)
        rawd=np.asarray(vis['DINO'][s:e],np.float32)
        rawc=np.asarray(vis['CONV'][s:e],np.float32)
        rawf=np.asarray(vis['FLOW'][s:e],np.float32)

        for bi,j in enumerate(range(s,e)):
            # Utonia cache contract: [... learned feat ..., coord3, normal3, color3, rough1]
            local=rawg[bi,:,-10:-7]
            normal_local=rawg[bi,:,-7:-4]
            rough=rawg[bi,:,-1]
            pose=man['poses'][j]
            grid,valid=_project_heading_local_to_grid(local,pose,man['crop_boxes'][j])
            if not np.any(valid): continue
            k=np.where(valid)[0]
            gi=grid[k]
            # Spatially matched four-stream token. Geometry is point-specific; image/motion use grid token.
            tok=(de[bi,gi]+ce[bi,gi]+ge[bi,k]+fe[bi,gi])/4.0
            tok=_safe_norm(tok,axis=-1).astype(np.float16)
            world=_world_from_heading_local(local[k],pose).astype(np.float32)
            # Rotate local normals into world using heading basis.
            Rlw,_=heading_frame_from_pose(pose)
            nw=(normal_local[k] @ Rlw).astype(np.float32)
            nw=_safe_norm(nw,axis=-1).astype(np.float32)
            # confidence = average explicit validity channels + flow finite confidence proxy
            # DINO/Conv append mask-validity as their final raw channel in V7 cache.
            dc=np.clip(rawd[bi,gi,-1],0,1)
            cc=np.clip(rawc[bi,gi,-1],0,1)
            fc=np.isfinite(rawf[bi,-1,gi]).all(-1).astype(np.float32)
            conf=((dc+cc+fc)/3.0).astype(np.float32)
            keep=conf>=V8A_MIN_CONF
            if not np.any(keep): continue
            kk=np.where(keep)[0]
            all_xy.append(world[kk,:2]); all_z.append(world[kk,2]); all_t.append(np.full(len(kk),float(man['camera_times'][j]),np.float32))
            all_tok.append(tok[kk]); all_n.append(nw[kk]); all_r.append(rough[k][kk].astype(np.float32)); all_conf.append(conf[kk])
            all_frame.append(np.full(len(kk),j,np.int32)); all_grid.append(gi[kk].astype(np.int16))

        if e%120==0 or e==N:
            print(f'  {e:04d}/{N:04d}')
        del de,ce,ge,fe
        torch.cuda.empty_cache()

    xy=np.concatenate(all_xy).astype(np.float32); z=np.concatenate(all_z).astype(np.float32)
    tt=np.concatenate(all_t).astype(np.float32); tok=np.concatenate(all_tok).astype(np.float16)
    nn=np.concatenate(all_n).astype(np.float32); rr=np.concatenate(all_r).astype(np.float32)
    cf=np.concatenate(all_conf).astype(np.float32); fr=np.concatenate(all_frame).astype(np.int32); gr=np.concatenate(all_grid).astype(np.int16)
    cell=np.floor(xy/V8_CELL_RES_M).astype(np.int32)
    # Sort by time so every query can enforce strict causal lookup.
    order=np.argsort(tt,kind='stable')
    np.savez_compressed(out,xy=xy[order],z=z[order],time=tt[order],token=tok[order],normal=nn[order],rough=rr[order],
                        confidence=cf[order],frame=fr[order],grid=gr[order],cell=cell[order])
    info=dict(environment=env,run=run,n_events=int(len(order)),cell_resolution_m=V8_CELL_RES_M,
              token_dim=V8A_TOKEN_DIM,alignment_mode=V8_ALIGNMENT_MODE)
    meta.write_text(json.dumps(info,indent=2))
    print(' Saved',out,'events=',len(order))
    return dict(path=out,meta=info)


class V8MemoryIndex:
    def __init__(self,path):
        z=np.load(path,mmap_mode='r',allow_pickle=False)
        self.path=Path(path); self.xy=np.asarray(z['xy']); self.z=np.asarray(z['z']); self.time=np.asarray(z['time'])
        self.token=np.asarray(z['token']); self.normal=np.asarray(z['normal']); self.rough=np.asarray(z['rough']); self.conf=np.asarray(z['confidence'])
        self.cell=np.asarray(z['cell']); self.tree=cKDTree(self.xy)
    def query(self,xy,t,radius=V8_MEMORY_QUERY_RADIUS_M,k=V8_MEMORY_KNN):
        xy=np.asarray(xy,np.float32).reshape(2)
        ids=np.asarray(self.tree.query_ball_point(xy,r=float(radius)),dtype=np.int64)
        if ids.size==0: return None
        causal=ids[self.time[ids] <= float(t)+1e-8]
        if causal.size==0: return None
        pts=self.xy[causal]
        dd=np.linalg.norm(pts-xy[None,:],axis=1)
        age=np.maximum(0,float(t)-self.time[causal])
        # Freshness and distance both matter; confidence weakly breaks ties.
        score=age + 1.5*dd - 0.05*np.clip(self.conf[causal],0,1)
        j=int(np.argmin(score)); q=int(causal[j])
        return dict(index=q,xy=self.xy[q].astype(np.float32),z=float(self.z[q]),time=float(self.time[q]),age_s=float(age[j]),
                    distance_m=float(dd[j]),token=self.token[q].astype(np.float32),normal=self.normal[q].astype(np.float32),
                    rough=float(self.rough[q]),confidence=float(self.conf[q]),cell=self.cell[q].copy())
    def query_many(self,xy,t,radius=V8_MEMORY_QUERY_RADIUS_M):
        return [self.query(p,t,radius=radius) for p in np.asarray(xy)]


# --------------------------- build all four memories ---------------------------
v8a_memory_files={}
v8_memory={}
for rec in run_records:
    key=(rec['terrain_type'],rec['environment'],rec['run'])
    info=_build_memory_events_for_run(rec)
    v8a_memory_files[key]=info['path']
    v8_memory[key]=V8MemoryIndex(info['path'])

# --------------------------- causal memory coverage benchmark ---------------------------
rows=[]
for rec in run_records:
    key=(rec['terrain_type'],rec['environment'],rec['run']); mem=v8_memory[key]; data=rec['data']
    # 10 Hz diagnostic sampling is enough; only actual-contact legs are evaluated.
    dt=float(np.median(np.diff(data['time']))); step=max(1,int(round(0.10/dt)))
    for idx in range(0,len(data['time']),step):
        feet=v8_foot_world_proxy(rec,idx)
        if feet is None: continue
        t=float(data['time'][idx])
        for leg in range(4):
            if data['contact'][idx,leg] < 0.5: continue
            q=mem.query(feet[leg,:2],t)
            rows.append(dict(terrain_type=rec['terrain_type'],environment=rec['environment'],run=rec['run'],time=t,leg=leg,
                             found=q is not None,distance_m=np.nan if q is None else q['distance_m'],age_s=np.nan if q is None else q['age_s'],
                             seen_0p2s_earlier=False if q is None else q['age_s']>=0.2,
                             seen_0p5s_earlier=False if q is None else q['age_s']>=0.5,
                             seen_1p0s_earlier=False if q is None else q['age_s']>=1.0,
                             seen_2p0s_earlier=False if q is None else q['age_s']>=2.0))

v8a_coverage=pd.DataFrame(rows)
v8a_summary=v8a_coverage.groupby(['terrain_type']).agg(
    contact_queries=('found','size'),memory_hit_rate=('found','mean'),median_distance_m=('distance_m','median'),median_age_s=('age_s','median'),
    seen_0p5s_earlier=('seen_0p5s_earlier','mean'),seen_1p0s_earlier=('seen_1p0s_earlier','mean'),seen_2p0s_earlier=('seen_2p0s_earlier','mean')
).reset_index()

v8a_coverage.to_csv(V8A_ROOT/'V8A_CAUSAL_MEMORY_CONTACT_COVERAGE_RAW.csv',index=False)
v8a_summary.to_csv(V8A_ROOT/'V8A_CAUSAL_MEMORY_CONTACT_COVERAGE_SUMMARY.csv',index=False)

print('\nV8-A MEMORY COVERAGE')
display(v8a_summary)
print('Alignment mode:',V8_ALIGNMENT_MODE)
print('V8-A COMPLETE:',V8A_ROOT)

# %% notebook cell 11
# ======================================================================
# V8-A+ — EPISODIC KEYFRAME MEMORY + FRONT/HIND COVERAGE GATE
# ======================================================================
class V8KeyframeMemory:
    def __init__(self,rec):
        self.rec=rec
        self.time=np.asarray(rec['manifest']['camera_times'],np.float64)
        self.pose=np.asarray(rec['manifest']['poses'],np.float64)
        self.crop=np.asarray(rec['manifest']['crop_boxes'],np.int32)
    def prior_frames(self,t,max_age_s=3.0,max_frames=8):
        ids=np.where((self.time<=float(t)+1e-9)&(self.time>=float(t)-max_age_s))[0]
        if len(ids)==0: return []
        ids=ids[-max_frames:][::-1]
        return [dict(frame_index=int(i),time=float(self.time[i]),age_s=float(t-self.time[i]),pose=self.pose[i].copy(),crop_box=self.crop[i].copy()) for i in ids]

v8_keyframes={(r['terrain_type'],r['environment'],r['run']):V8KeyframeMemory(r) for r in run_records}

leg_name={0:'FL',1:'FR',2:'RL',3:'RR'}
v8a_coverage['leg_name']=v8a_coverage.leg.map(leg_name)
v8a_coverage['leg_group']=v8a_coverage.leg.map(lambda x:'front' if x<2 else 'hind')
leg_summary=v8a_coverage.groupby(['leg_group','leg_name']).agg(
    queries=('found','size'),hit_rate=('found','mean'),seen_0p5=('seen_0p5s_earlier','mean'),seen_1p0=('seen_1p0s_earlier','mean'),seen_2p0=('seen_2p0s_earlier','mean'),median_age_s=('age_s','median')
).reset_index()
display(leg_summary)

V8A_HIT=float(v8a_coverage.found.mean())
V8A_HIND_05=float(v8a_coverage[v8a_coverage.leg>=2].seen_0p5s_earlier.mean())
V8A_GATE=bool(V8A_HIT>=0.60 and V8A_HIND_05>=0.20)
print('V8-A gate:',V8A_GATE,'overall hit=',V8A_HIT,'hind seen >=0.5s earlier=',V8A_HIND_05)
print('This threshold is an engineering go/no-go diagnostic, not a published benchmark.')

# %% notebook cell 13
# ======================================================================
# V8-B — FUTURE PER-LEG TOUCHDOWN DISTRIBUTION
# Run AFTER V8-A in the SAME live Colab session.
#
# Learns, from proprioception/body/command history only:
#   p(contact_i(t+h)), mean future foot XYZ, and XYZ uncertainty
# for FL/FR/RL/RR over 0.2...2.0 s.
#
# Output coordinates are in the CURRENT camera-heading frame
# (x forward, y right, z up), so they can be transformed directly into
# the V8-A world memory. No future image or future sensor is used as input.
# ======================================================================

import json, math, copy, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

_req=['run_records','v8_foot_world_proxy','_data_camera_pose','_heading_local_from_world','V8_ROOT','DEVICE']
_missing=[x for x in _req if x not in globals()]
if _missing: raise RuntimeError('Run V8-A first. Missing: '+repr(_missing))

V8B_ROOT=V8_ROOT/'B_touchdown'
V8B_ROOT.mkdir(parents=True,exist_ok=True)
V8B_HORIZONS=np.asarray([0.2,0.4,0.6,0.8,1.0,1.2,1.5,2.0],np.float32)
V8B_CONTEXT_SEC=0.5
V8B_HOP_SEC=0.10
V8B_BATCH=32
V8B_EPOCHS=18
V8B_LR=3e-4
V8B_SEED=int(globals().get('SEED',1337))+8200

def _seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
_seed(V8B_SEED)


def _raw_leg_history(data,idx,H):
    h0=idx-H+1
    if h0<0: return None
    # Per leg: P12 + U6 + body B12 = 30 channels.
    P=data['P'][h0:idx+1].transpose(1,0,2)                    # 4,T,12
    U=np.broadcast_to(data['U'][h0:idx+1][None],(4,H,6))     # 4,T,6
    B=np.broadcast_to(data['B'][h0:idx+1][None],(4,H,12))    # 4,T,12
    return np.concatenate([P,U,B],axis=-1).astype(np.float32)


def _touch_targets(rec,idx,horizons):
    data=rec['data']; t0=float(data['time'][idx]); pose0,_=_data_camera_pose(rec,idx)
    if pose0 is None: return None
    dt=float(np.median(np.diff(data['time'])))
    C=[]; POS=[]
    for h in horizons:
        j=idx+int(round(float(h)/dt))
        if j>=len(data['time']): return None
        fw=v8_foot_world_proxy(rec,j)
        if fw is None: return None
        loc=_heading_local_from_world(fw,pose0).astype(np.float32)
        C.append(data['contact'][j].astype(np.float32)); POS.append(loc)
    return np.asarray(C,np.float32),np.asarray(POS,np.float32)  # H,4 ; H,4,3


def _build_touch_dataset():
    X=[]; C=[]; Y=[]; meta=[]; all_feat=[]
    for ri,rec in enumerate(run_records):
        data=rec['data']; dt=float(np.median(np.diff(data['time']))); H=int(round(V8B_CONTEXT_SEC/dt))
        stop=float(min(rec['ttests']))-float(V8B_HORIZONS.max())-0.05
        start=float(data['time'][0])+V8B_CONTEXT_SEC
        tt=start
        seen=set()
        while tt<=stop+1e-8:
            idx=int(np.argmin(np.abs(data['time']-tt)))
            if idx in seen: tt+=V8B_HOP_SEC; continue
            seen.add(idx)
            x=_raw_leg_history(data,idx,H); tar=_touch_targets(rec,idx,V8B_HORIZONS)
            if x is not None and tar is not None:
                c,y=tar; X.append(x); C.append(c); Y.append(y)
                meta.append((ri,idx,float(data['time'][idx])))
                all_feat.append(x.reshape(-1,30))
            tt+=V8B_HOP_SEC
    if not X: raise RuntimeError('No V8-B training windows.')
    feat=np.concatenate(all_feat,axis=0)
    mean=feat.mean(0).astype(np.float32); std=feat.std(0).astype(np.float32); std=np.maximum(std,1e-4)
    return np.asarray(X,np.float32),np.asarray(C,np.float32),np.asarray(Y,np.float32),meta,mean,std


Xb,Cb,Yb,v8b_meta,V8B_MEAN,V8B_STD=_build_touch_dataset()
print('V8-B training windows:',len(Xb),'history shape=',Xb.shape[1:],'targets=',Yb.shape)


class _TouchDS(Dataset):
    def __init__(self,X,C,Y,mean,std): self.X=X; self.C=C; self.Y=Y; self.mean=mean; self.std=std
    def __len__(self): return len(self.X)
    def __getitem__(self,i):
        x=(self.X[i]-self.mean[None,None,:])/self.std[None,None,:]
        return torch.from_numpy(x),torch.from_numpy(self.C[i]),torch.from_numpy(self.Y[i])


class V8TouchdownTransformer(nn.Module):
    def __init__(self,horizons):
        super().__init__(); self.register_buffer('horizons',torch.as_tensor(horizons,dtype=torch.float32))
        d=128
        self.inp=nn.Linear(30,d); self.leg=nn.Embedding(4,d); self.pos=nn.Parameter(torch.randn(1,128,d)*0.01)
        layer=nn.TransformerEncoderLayer(d_model=d,nhead=4,dim_feedforward=512,dropout=.1,batch_first=True,norm_first=True,activation='gelu')
        self.tx=nn.TransformerEncoder(layer,num_layers=3); self.norm=nn.LayerNorm(d)
        self.hproj=nn.Sequential(nn.Linear(1,64),nn.GELU(),nn.Linear(64,d),nn.LayerNorm(d))
        self.fuse=nn.Sequential(nn.LayerNorm(2*d),nn.Linear(2*d,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,128),nn.GELU())
        self.contact=nn.Linear(128,1); self.mu=nn.Linear(128,3); self.logstd=nn.Linear(128,3)
    def forward(self,x):
        # x B,4,T,30
        B,L,T,C=x.shape; xx=x.reshape(B*L,T,C); z=self.inp(xx)+self.pos[:,:T]
        mask=torch.triu(torch.ones(T,T,device=x.device,dtype=torch.bool),diagonal=1)
        z=self.norm(self.tx(z,mask=mask)[:,-1]).reshape(B,L,-1)
        ids=torch.arange(4,device=x.device)[None].expand(B,-1); z=z+self.leg(ids)
        H=len(self.horizons); hz=self.hproj(self.horizons.view(1,H,1).expand(B,-1,-1))
        zz=torch.cat([z[:,None].expand(-1,H,-1,-1),hz[:,:,None].expand(-1,-1,4,-1)],dim=-1)
        h=self.fuse(zz)
        return dict(contact_logits=self.contact(h).squeeze(-1),mu=self.mu(h),logstd=self.logstd(h).clamp(-4.0,1.5))


def _touch_loss(out,c,y):
    logits=out['contact_logits']; mu=out['mu']; ls=out['logstd']; var=torch.exp(2*ls)
    # Give contact positives extra weight without making swing invisible.
    posw=torch.tensor(1.6,device=logits.device)
    lc=F.binary_cross_entropy_with_logits(logits,c,pos_weight=posw)
    mask=c[...,None]
    nll=(0.5*((y-mu)**2/var + 2*ls)).sum(-1)
    lpos=(nll*c).sum()/(c.sum()+1e-6)
    # Weak all-phase trajectory supervision stabilizes the future foot cloud.
    ltrack=F.smooth_l1_loss(mu,y,beta=.08)
    return lc+0.75*lpos+0.20*ltrack,dict(contact=float(lc.detach()),pos=float(lpos.detach()),track=float(ltrack.detach()))

# Grouped temporal validation: last 15% of PRE-TEST windows from each run.
idx=np.arange(len(Xb)); val_mask=np.zeros(len(Xb),dtype=bool)
for ri in sorted(set(m[0] for m in v8b_meta)):
    ids=np.array([i for i,m in enumerate(v8b_meta) if m[0]==ri],dtype=int)
    times=np.array([v8b_meta[i][2] for i in ids],np.float64)
    cut=np.quantile(times,0.85)
    val_mask[ids[times>=cut]]=True
tr=np.where(~val_mask)[0]; va=np.where(val_mask)[0]
train_ds=_TouchDS(Xb[tr],Cb[tr],Yb[tr],V8B_MEAN,V8B_STD); val_ds=_TouchDS(Xb[va],Cb[va],Yb[va],V8B_MEAN,V8B_STD)
train_dl=DataLoader(train_ds,batch_size=V8B_BATCH,shuffle=True,num_workers=0); val_dl=DataLoader(val_ds,batch_size=64,shuffle=False,num_workers=0)

v8_touchdown_model=V8TouchdownTransformer(V8B_HORIZONS).to(DEVICE)
opt=torch.optim.AdamW(v8_touchdown_model.parameters(),lr=V8B_LR,weight_decay=1e-4)
best=None; bestv=float('inf')
for ep in range(V8B_EPOCHS):
    v8_touchdown_model.train(); tl=[]
    for x,c,y in train_dl:
        x=x.to(DEVICE); c=c.to(DEVICE); y=y.to(DEVICE); opt.zero_grad(set_to_none=True)
        o=v8_touchdown_model(x); loss,_=_touch_loss(o,c,y); loss.backward(); torch.nn.utils.clip_grad_norm_(v8_touchdown_model.parameters(),1.0); opt.step(); tl.append(float(loss.detach().cpu()))
    v8_touchdown_model.eval(); vl=[]
    with torch.inference_mode():
        for x,c,y in val_dl:
            x=x.to(DEVICE); c=c.to(DEVICE); y=y.to(DEVICE); o=v8_touchdown_model(x); loss,_=_touch_loss(o,c,y); vl.append(float(loss.cpu()))
    vm=float(np.mean(vl)); print(f'V8-B epoch {ep+1:02d}/{V8B_EPOCHS} train={np.mean(tl):.4f} val={vm:.4f}')
    if vm<bestv: bestv=vm; best=copy.deepcopy(v8_touchdown_model.state_dict())
v8_touchdown_model.load_state_dict(best); v8_touchdown_model.eval()


def v8b_predict(rec,idx):
    data=rec['data']; dt=float(np.median(np.diff(data['time']))); H=int(round(V8B_CONTEXT_SEC/dt))
    x=_raw_leg_history(data,idx,H)
    if x is None: return None
    xn=(x-V8B_MEAN[None,None,:])/V8B_STD[None,None,:]
    with torch.inference_mode():
        o=v8_touchdown_model(torch.as_tensor(xn[None],dtype=torch.float32,device=DEVICE))
    return dict(contact_prob=torch.sigmoid(o['contact_logits'])[0].cpu().numpy(),mu=o['mu'][0].cpu().numpy(),sigma=torch.exp(o['logstd'])[0].cpu().numpy())

# --------------------------- same 16 held-out diagnostic times ---------------------------
rows=[]; timing=[]
for rec in run_records:
    data=rec['data']; dt=float(np.median(np.diff(data['time'])))
    for ti,tt in enumerate(rec['ttests'],1):
        idx0=int(np.argmin(np.abs(data['time']-float(tt))))
        pred=v8b_predict(rec,idx0); tar=_touch_targets(rec,idx0,V8B_HORIZONS)
        if pred is None or tar is None: continue
        c,y=tar; p=pred['contact_prob']; mu=pred['mu']; sig=pred['sigma']
        for hi,h in enumerate(V8B_HORIZONS):
            for leg in range(4):
                err=float(np.linalg.norm(mu[hi,leg]-y[hi,leg]))
                rows.append(dict(terrain_type=rec['terrain_type'],test_index=ti,ttest=float(tt),horizon_s=float(h),leg=leg,
                                 contact_true=float(c[hi,leg]),contact_prob=float(p[hi,leg]),position_error_m=err,
                                 sigma_mean_m=float(sig[hi,leg].mean()),within_5cm=err<=.05,within_10cm=err<=.10,within_20cm=err<=.20))
        for leg in range(4):
            gt=np.where(c[:,leg]>0.5)[0]; pr=np.where(p[:,leg]>0.5)[0]
            gt_t=np.nan if len(gt)==0 else float(V8B_HORIZONS[gt[0]]); pr_t=np.nan if len(pr)==0 else float(V8B_HORIZONS[pr[0]])
            timing.append(dict(terrain_type=rec['terrain_type'],test_index=ti,leg=leg,gt_first_contact_s=gt_t,pred_first_contact_s=pr_t,
                               timing_abs_error_s=np.nan if not(np.isfinite(gt_t) and np.isfinite(pr_t)) else abs(gt_t-pr_t)))

v8b_eval=pd.DataFrame(rows); v8b_timing=pd.DataFrame(timing)
summary=[]
for h,g in v8b_eval.groupby('horizon_s'):
    yt=g.contact_true.values.astype(int); yp=(g.contact_prob.values>=.5).astype(int)
    pr,re,f1,_=precision_recall_fscore_support(yt,yp,average='binary',zero_division=0)
    contact_rows=g[g.contact_true>0.5]
    summary.append(dict(horizon_s=h,contact_precision=pr,contact_recall=re,contact_F1=f1,
                        contact_position_MAE_m=contact_rows.position_error_m.mean(),
                        Recall_5cm=contact_rows.within_5cm.mean(),Recall_10cm=contact_rows.within_10cm.mean(),Recall_20cm=contact_rows.within_20cm.mean(),
                        mean_sigma_m=contact_rows.sigma_mean_m.mean()))
v8b_summary=pd.DataFrame(summary)

v8b_eval.to_csv(V8B_ROOT/'V8B_16TEST_TOUCHDOWN_RAW.csv',index=False)
v8b_summary.to_csv(V8B_ROOT/'V8B_TOUCHDOWN_METRICS_BY_HORIZON.csv',index=False)
v8b_timing.to_csv(V8B_ROOT/'V8B_CONTACT_TIMING.csv',index=False)
torch.save(dict(state_dict=v8_touchdown_model.state_dict(),mean=V8B_MEAN,std=V8B_STD,horizons=V8B_HORIZONS),V8B_ROOT/'V8B_TOUCHDOWN_MODEL.pt')

print('\nV8-B HELD-OUT TOUCHDOWN METRICS')
display(v8b_summary)
print('Median first-contact timing error [s]:',float(v8b_timing.timing_abs_error_s.median()))
print('V8-B COMPLETE:',V8B_ROOT)

# %% notebook cell 14
# ======================================================================
# V8-B SCIENCE GATE
# ======================================================================
contact_b=v8b_summary.copy()
V8B_RECALL10=float(contact_b.Recall_10cm.mean()) if len(contact_b) else 0.0
V8B_F1=float(contact_b.contact_F1.mean()) if len(contact_b) else 0.0
V8B_GATE=bool(V8B_RECALL10>=0.50 and V8B_F1>=0.80)
print('V8-B gate:',V8B_GATE,'mean Recall@10cm=',V8B_RECALL10,'mean contact F1=',V8B_F1)
print('If this fails, do not trust precise foothold optimization yet; improve touchdown prediction first.')

# %% notebook cell 16
# ======================================================================
# V8-C — QUERYABLE LOCAL CONTACT-RESPONSE FIELD
# Run AFTER V8-A and V8-B.
#
# Learns a CELL-LOCAL probabilistic law:
#   p([tau,F]_{t+h} | terrain_cell_token, geometry, current foot/body state,
#                         command, kinematics, horizon)
# using only terrain evidence that was already observed by the current time.
#
# This is the first V8 module that can be queried at an ALTERNATIVE terrain cell.
# ======================================================================

import json, math, copy, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

_req=['run_records','v8_memory','v8_foot_world_proxy','V8_ROOT','ANYMAL_D_FK','DEVICE']
_missing=[x for x in _req if x not in globals()]
if _missing: raise RuntimeError('Run V8-A/B first. Missing: '+repr(_missing))

V8C_ROOT=V8_ROOT/'C_contact_field'; V8C_ROOT.mkdir(parents=True,exist_ok=True)
V8C_HORIZONS=np.asarray([0.1,0.3,0.5,0.7,1.0],np.float32)
V8C_HOP_SEC=0.15
V8C_BATCH=128
V8C_EPOCHS=22
V8C_LR=3e-4
V8C_SEED=int(globals().get('SEED',1337))+8300
V8C_N_EXPERTS=4

def _seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
_seed(V8C_SEED)


def v8_numeric_leg_jacobian(q3,leg,eps=1e-4):
    q3=np.asarray(q3,np.float64); J=np.zeros((3,3),np.float64)
    q4=np.zeros((4,3),np.float64); q4[leg]=q3
    p0=ANYMAL_D_FK.foot_positions(q4)[leg].astype(np.float64)
    for j in range(3):
        qq=q4.copy(); qq[leg,j]+=eps
        p=ANYMAL_D_FK.foot_positions(qq)[leg].astype(np.float64)
        J[:,j]=(p-p0)/eps
    return J.astype(np.float32)


def _current_foot_kin(rec,idx,leg):
    p=rec['data']['P'][idx,leg]
    q=p[0:3]; dq=p[3:6]
    q4=np.zeros((4,3),np.float32); q4[leg]=q
    pos=ANYMAL_D_FK.foot_positions(q4)[leg].astype(np.float32)
    J=v8_numeric_leg_jacobian(q,leg); vel=(J@dq.astype(np.float32)).astype(np.float32)
    return pos,vel


def v8c_feature_from_memory_query(rec,idx,leg,horizon,mq):
    """297-D candidate-cell feature. mq is result of V8MemoryIndex.query()."""
    data=rec['data']; p=data['P'][idx,leg]; pos,vel=_current_foot_kin(rec,idx,leg)
    geom=np.concatenate([
        np.array([mq['z']],np.float32),mq['normal'].astype(np.float32),
        np.array([mq['rough'],mq['confidence'],min(mq['age_s'],5.0)/5.0],np.float32)
    ]) # 7
    state=np.concatenate([
        p[0:9].astype(np.float32),            # q,dq,tau (NO measured force target leakage)
        data['U'][idx].astype(np.float32),    # 6
        data['B'][idx].astype(np.float32),    # 12
        pos,vel,                              # 6
        np.array([float(horizon)],np.float32)
    ])
    x=np.concatenate([mq['token'].astype(np.float32),geom,state]).astype(np.float32)
    return x


def _build_v8c_dataset():
    X=[]; Y=[]; M=[]
    for ri,rec in enumerate(run_records):
        data=rec['data']; mem=v8_memory[(rec['terrain_type'],rec['environment'],rec['run'])]
        dt=float(np.median(np.diff(data['time']))); stop=float(min(rec['ttests']))-float(V8C_HORIZONS.max())-.05
        tt=float(data['time'][0])+.6
        while tt<=stop+1e-9:
            idx=int(np.argmin(np.abs(data['time']-tt)))
            for h in V8C_HORIZONS:
                j=idx+int(round(float(h)/dt))
                if j>=len(data['time']): continue
                fw=v8_foot_world_proxy(rec,j)
                if fw is None: continue
                for leg in range(4):
                    mq=mem.query(fw[leg,:2],float(data['time'][idx]))
                    if mq is None: continue
                    x=v8c_feature_from_memory_query(rec,idx,leg,float(h),mq)
                    y=np.concatenate([data['P'][j,leg,6:9],data['P'][j,leg,9:12]]).astype(np.float32)
                    X.append(x); Y.append(y); M.append((ri,idx,j,leg,float(h),mq['age_s'],mq['distance_m']))
            tt+=V8C_HOP_SEC
    if not X: raise RuntimeError('No V8-C causal-memory training samples. V8-A map alignment may be too sparse.')
    X=np.asarray(X,np.float32); Y=np.asarray(Y,np.float32)
    xm=X.mean(0).astype(np.float32); xs=np.maximum(X.std(0),1e-4).astype(np.float32)
    ym=Y.mean(0).astype(np.float32); ys=np.maximum(Y.std(0),1e-3).astype(np.float32)
    return X,Y,M,xm,xs,ym,ys


V8C_X,V8C_Y,v8c_meta,V8C_XMEAN,V8C_XSTD,V8C_YMEAN,V8C_YSTD=_build_v8c_dataset()
print('V8-C samples:',len(V8C_X),'feature_dim=',V8C_X.shape[1])


class _V8CDS(Dataset):
    def __init__(self,idx): self.idx=np.asarray(idx)
    def __len__(self): return len(self.idx)
    def __getitem__(self,k):
        i=int(self.idx[k]); x=(V8C_X[i]-V8C_XMEAN)/V8C_XSTD; y=(V8C_Y[i]-V8C_YMEAN)/V8C_YSTD
        return torch.from_numpy(x),torch.from_numpy(y)


class V8ContactFieldMoE(nn.Module):
    def __init__(self,din,nexp=4):
        super().__init__(); self.nexp=nexp
        self.trunk=nn.Sequential(nn.LayerNorm(din),nn.Linear(din,512),nn.GELU(),nn.Dropout(.1),nn.Linear(512,256),nn.GELU(),nn.LayerNorm(256))
        self.gate=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Linear(128,nexp))
        self.experts=nn.ModuleList([nn.Sequential(nn.Linear(256,256),nn.GELU(),nn.Linear(256,12)) for _ in range(nexp)])
    def forward(self,x):
        z=self.trunk(x); logits=self.gate(z); pi=logits.softmax(-1)
        eo=torch.stack([e(z) for e in self.experts],dim=1)   # B,E,12
        mu=eo[...,:6]; ls=eo[...,6:].clamp(-4,2)
        mix=(pi[...,None]*mu).sum(1)
        # Law of total variance in normalized target units.
        var=(pi[...,None]*(torch.exp(2*ls)+mu**2)).sum(1)-mix**2
        return dict(gate_logits=logits,pi=pi,expert_mu=mu,expert_logstd=ls,mean=mix,var=var.clamp_min(1e-6))


def _v8c_loss(o,y):
    # True diagonal Gaussian mixture NLL.
    mu=o['expert_mu']; ls=o['expert_logstd']; yy=y[:,None,:]
    logp=-0.5*(((yy-mu)/torch.exp(ls))**2 + 2*ls + math.log(2*math.pi)).sum(-1)
    logmix=torch.log_softmax(o['gate_logits'],dim=-1)+logp
    nll=-torch.logsumexp(logmix,dim=-1).mean()
    # Force is the critical quantity; add point losses on mixture mean.
    hub_tau=F.smooth_l1_loss(o['mean'][:,:3],y[:,:3],beta=.4)
    hub_force=F.smooth_l1_loss(o['mean'][:,3:],y[:,3:],beta=.4)
    # Encourage non-collapsed experts without forcing uniform gating per sample.
    pbar=o['pi'].mean(0); bal=((pbar-1.0/o['pi'].shape[1])**2).mean()
    return nll+0.25*hub_tau+0.65*hub_force+0.10*bal

# Grouped temporal validation: last 15% of PRE-TEST source times within each run.
mi=np.arange(len(V8C_X)); val=np.zeros(len(V8C_X),dtype=bool)
for ri in sorted(set(m[0] for m in v8c_meta)):
    ids=np.array([i for i,m in enumerate(v8c_meta) if m[0]==ri],dtype=int)
    times=np.array([run_records[ri]['data']['time'][v8c_meta[i][1]] for i in ids],np.float64)
    cut=np.quantile(times,0.85)
    val[ids[times>=cut]]=True
tr=mi[~val]; va=mi[val]
trdl=DataLoader(_V8CDS(tr),batch_size=V8C_BATCH,shuffle=True,num_workers=0); vadl=DataLoader(_V8CDS(va),batch_size=256,shuffle=False,num_workers=0)
v8_contact_field=V8ContactFieldMoE(V8C_X.shape[1],V8C_N_EXPERTS).to(DEVICE)
opt=torch.optim.AdamW(v8_contact_field.parameters(),lr=V8C_LR,weight_decay=1e-4)
bestv=float('inf'); best=None
for ep in range(V8C_EPOCHS):
    v8_contact_field.train(); tl=[]
    for x,y in trdl:
        x=x.to(DEVICE); y=y.to(DEVICE); opt.zero_grad(set_to_none=True); o=v8_contact_field(x); l=_v8c_loss(o,y); l.backward(); torch.nn.utils.clip_grad_norm_(v8_contact_field.parameters(),1.0); opt.step(); tl.append(float(l.detach().cpu()))
    v8_contact_field.eval(); vl=[]
    with torch.inference_mode():
        for x,y in vadl:
            x=x.to(DEVICE); y=y.to(DEVICE); vl.append(float(_v8c_loss(v8_contact_field(x),y).cpu()))
    vm=float(np.mean(vl)); print(f'V8-C epoch {ep+1:02d}/{V8C_EPOCHS} train={np.mean(tl):.4f} val={vm:.4f}')
    if vm<bestv: bestv=vm; best=copy.deepcopy(v8_contact_field.state_dict())
v8_contact_field.load_state_dict(best); v8_contact_field.eval()


def v8c_predict_feature(x):
    xn=(np.asarray(x,np.float32)-V8C_XMEAN)/V8C_XSTD
    with torch.inference_mode():
        o=v8_contact_field(torch.as_tensor(xn[None],dtype=torch.float32,device=DEVICE))
    mu=(o['mean'][0].cpu().numpy()*V8C_YSTD+V8C_YMEAN).astype(np.float32)
    sd=(np.sqrt(o['var'][0].cpu().numpy())*V8C_YSTD).astype(np.float32)
    pi=o['pi'][0].cpu().numpy().astype(np.float32)
    return dict(tau=mu[:3],force=mu[3:],tau_sigma=sd[:3],force_sigma=sd[3:],expert_prob=pi)


def v8c_predict_cell(rec,idx,leg,horizon,mq):
    return v8c_predict_feature(v8c_feature_from_memory_query(rec,idx,leg,horizon,mq))

# --------------------------- held-out 16-test evaluation ---------------------------
rows=[]
for rec in run_records:
    data=rec['data']; mem=v8_memory[(rec['terrain_type'],rec['environment'],rec['run'])]; dt=float(np.median(np.diff(data['time'])))
    for ti,tt in enumerate(rec['ttests'],1):
        idx=int(np.argmin(np.abs(data['time']-float(tt))))
        for h in V8C_HORIZONS:
            j=idx+int(round(float(h)/dt))
            if j>=len(data['time']): continue
            fw=v8_foot_world_proxy(rec,j)
            if fw is None: continue
            for leg in range(4):
                mq=mem.query(fw[leg,:2],float(data['time'][idx]))
                if mq is None: continue
                pr=v8c_predict_cell(rec,idx,leg,float(h),mq); gt_tau=data['P'][j,leg,6:9]; gt_f=data['P'][j,leg,9:12]
                pred_f=np.asarray(pr['force'],np.float32); gt_f=np.asarray(gt_f,np.float32)
                pred_tau=np.asarray(pr['tau'],np.float32); gt_tau=np.asarray(gt_tau,np.float32)
                pf=float(np.linalg.norm(pred_f)); gf=float(np.linalg.norm(gt_f))
                cos=np.nan if min(pf,gf)<10.0 else float(np.clip(np.dot(pred_f,gt_f)/(pf*gf+1e-9),-1.0,1.0))
                ang=np.nan if not np.isfinite(cos) else float(np.degrees(np.arccos(cos)))
                fve=float(np.linalg.norm(pred_f-gt_f)); tve=float(np.linalg.norm(pred_tau-gt_tau))
                rows.append(dict(terrain_type=rec['terrain_type'],test_index=ti,ttest=float(tt),horizon_s=float(h),leg=leg,
                                 memory_age_s=mq['age_s'],memory_distance_m=mq['distance_m'],
                                 tau_MAE=float(np.mean(np.abs(pred_tau-gt_tau))),tau_vector_error=tve,
                                 force_MAE=float(np.mean(np.abs(pred_f-gt_f))),force_vector_error=fve,
                                 force_magnitude_error=float(abs(pf-gf)),force_direction_cosine=cos,force_direction_angle_deg=ang,
                                 force_vector_within_5N=bool(fve<=5.0),force_component_mean_within_5N=bool(np.mean(np.abs(pred_f-gt_f))<=5.0),
                                 force_sigma_mean=float(pr['force_sigma'].mean()),
                                 pred_Fx=float(pred_f[0]),pred_Fy=float(pred_f[1]),pred_Fz=float(pred_f[2]),
                                 gt_Fx=float(gt_f[0]),gt_Fy=float(gt_f[1]),gt_Fz=float(gt_f[2])))

v8c_eval=pd.DataFrame(rows)
v8c_summary=v8c_eval.groupby('horizon_s').agg(samples=('force_MAE','size'),tau_MAE=('tau_MAE','mean'),force_MAE=('force_MAE','mean'),
                                               force_vector_error=('force_vector_error','mean'),force_sigma_mean=('force_sigma_mean','mean')).reset_index()
v8c_eval.to_csv(V8C_ROOT/'V8C_16TEST_CONTACT_FIELD_RAW.csv',index=False)
v8c_summary.to_csv(V8C_ROOT/'V8C_CONTACT_FIELD_METRICS_BY_HORIZON.csv',index=False)
torch.save(dict(state_dict=v8_contact_field.state_dict(),xmean=V8C_XMEAN,xstd=V8C_XSTD,ymean=V8C_YMEAN,ystd=V8C_YSTD,horizons=V8C_HORIZONS),V8C_ROOT/'V8C_CONTACT_FIELD.pt')

print('\nV8-C HELD-OUT QUERYABLE FIELD METRICS')
display(v8c_summary)
print('V8-C COMPLETE:',V8C_ROOT)

# %% notebook cell 17
# ======================================================================
# V8-C STRICT PHYSICAL GATE: FORCE MAGNITUDE, DIRECTION, VECTOR + TORQUE
# ======================================================================
if len(v8c_eval):
    cgate=v8c_eval.groupby('horizon_s').agg(
        samples=('force_vector_error','size'),
        force_component_MAE_N=('force_MAE','mean'),
        force_vector_mean_N=('force_vector_error','mean'),
        force_vector_p95_N=('force_vector_error',lambda s:float(np.quantile(s,.95))),
        force_vector_max_N=('force_vector_error','max'),
        force_fraction_within5=('force_vector_within_5N','mean'),
        force_magnitude_MAE_N=('force_magnitude_error','mean'),
        force_direction_angle_median_deg=('force_direction_angle_deg','median'),
        force_direction_angle_p95_deg=('force_direction_angle_deg',lambda s:float(np.nanquantile(s,.95))),
        tau_component_MAE_Nm=('tau_MAE','mean'),tau_vector_mean_Nm=('tau_vector_error','mean')
    ).reset_index()
    display(cgate)
    focus=cgate[cgate.horizon_s.isin([0.1,1.0])]
    V8C_STRICT_GATE=bool(len(focus)==2 and (focus.force_vector_p95_N<=FORCE_VECTOR_TARGET_N).all())
    print('V8-C strict 95%-within-5N-vector gate:',V8C_STRICT_GATE)
else:
    cgate=pd.DataFrame(); V8C_STRICT_GATE=False

if ENFORCE_V7F_MASTER_GATE and not V7F_MASTER_FORCE_GATE:
    raise RuntimeError('ENFORCE_V7F_MASTER_GATE=True and the Stage-0 V7-F ±5 N mastery gate failed.')

# %% notebook cell 19
# ======================================================================
# V8-D — MATCHED-STATE COUNTERFACTUAL TERRAIN CHANGE HEAD
# Run AFTER V8-C.
#
# Learns the proposal's Siamese relation:
#   same/similar robot state + same leg + same horizon, different terrain token
#        -> change in [tau,F]
#
# This does NOT manufacture arbitrary counterfactual ground truth. Pairs are formed
# only from observed samples whose state/action contexts are nearest matches across
# different terrain runs.
# ======================================================================

import copy, random, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from scipy.spatial import cKDTree

_req=['V8C_X','V8C_Y','v8c_meta','V8C_XMEAN','V8C_XSTD','V8C_YSTD','run_records','V8_ROOT','DEVICE']
_missing=[x for x in _req if x not in globals()]
if _missing: raise RuntimeError('Run V8-C first. Missing: '+repr(_missing))

V8D_ROOT=V8_ROOT/'D_counterfactual'; V8D_ROOT.mkdir(parents=True,exist_ok=True)
V8D_SEED=int(globals().get('SEED',1337))+8400
V8D_MAX_PAIRS=12000
V8D_BATCH=128
V8D_EPOCHS=20
V8D_LR=3e-4
random.seed(V8D_SEED); np.random.seed(V8D_SEED); torch.manual_seed(V8D_SEED)

# First 256 dimensions are the persistent terrain token by V8-C contract.
Z=V8C_X[:,:256].astype(np.float32)
S=((V8C_X[:,256:]-V8C_XMEAN[256:])/V8C_XSTD[256:]).astype(np.float32)
Y=V8C_Y.astype(np.float32)
terrain=np.asarray([run_records[m[0]]['terrain_type'] for m in v8c_meta])
leg=np.asarray([m[3] for m in v8c_meta],np.int64)
hor=np.asarray([m[4] for m in v8c_meta],np.float32)


def _build_pairs():
    rng=np.random.default_rng(V8D_SEED); pairs=[]
    for lg in range(4):
        for h in sorted(np.unique(hor)):
            ids=np.where((leg==lg)&(np.isclose(hor,h)))[0]
            if len(ids)<20: continue
            terrs=np.unique(terrain[ids])
            trees={}
            for t in terrs:
                jj=ids[terrain[ids]==t]
                if len(jj)>=3: trees[t]=(jj,cKDTree(S[jj]))
            for ta in terrs:
                ia=ids[terrain[ids]==ta]
                if len(ia)==0: continue
                # Limit adjacent/correlated anchors per group.
                if len(ia)>350: ia=rng.choice(ia,350,replace=False)
                for a in ia:
                    best=None
                    for tb,(jj,tree) in trees.items():
                        if tb==ta: continue
                        d,k=tree.query(S[a],k=1)
                        if np.isfinite(d) and (best is None or d<best[0]): best=(float(d),int(jj[int(k)]))
                    if best is not None:
                        pairs.append((int(a),best[1],best[0]))
    if not pairs: raise RuntimeError('No cross-terrain matched-state pairs.')
    # Keep best matches first, then subsample to control size.
    pairs=sorted(pairs,key=lambda x:x[2])
    # Reject the worst half of state matches; this makes the causal claim stricter.
    cutoff=np.quantile([p[2] for p in pairs],0.55)
    pairs=[p for p in pairs if p[2]<=cutoff]
    if len(pairs)>V8D_MAX_PAIRS:
        sel=rng.choice(len(pairs),V8D_MAX_PAIRS,replace=False); pairs=[pairs[i] for i in sel]
    return pairs

V8D_PAIRS=_build_pairs()
print('V8-D matched cross-terrain pairs:',len(V8D_PAIRS),'median state distance=',float(np.median([p[2] for p in V8D_PAIRS])))


def _descriptor(a,b):
    za,zb=Z[a],Z[b]
    siam=np.concatenate([za,zb,zb-za,np.abs(zb-za),za*zb],axis=0)
    shared=.5*(S[a]+S[b])
    return np.concatenate([siam,shared],axis=0).astype(np.float32)

DX=np.asarray([_descriptor(a,b) for a,b,_ in V8D_PAIRS],np.float32)
DY=np.asarray([Y[b]-Y[a] for a,b,_ in V8D_PAIRS],np.float32)
# Normalize response delta only; token/state descriptor is already well behaved.
DYM=DY.mean(0).astype(np.float32); DYS=np.maximum(DY.std(0),1e-3).astype(np.float32)
DYN=(DY-DYM)/DYS


class _PairDS(Dataset):
    def __init__(self,ids): self.ids=np.asarray(ids)
    def __len__(self): return len(self.ids)
    def __getitem__(self,k):
        i=int(self.ids[k]); return torch.from_numpy(DX[i]),torch.from_numpy(DYN[i])


class V8SiameseDelta(nn.Module):
    def __init__(self,din):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(din),nn.Linear(din,768),nn.GELU(),nn.Dropout(.1),nn.Linear(768,384),nn.GELU(),nn.Dropout(.1),nn.Linear(384,192),nn.GELU())
        self.out=nn.Linear(192,12)
    def forward(self,x):
        y=self.out(self.net(x)); return dict(mean=y[:,:6],logstd=y[:,6:].clamp(-4,2))


def _loss(o,y):
    ls=o['logstd']; mu=o['mean']; nll=(.5*((y-mu)/torch.exp(ls))**2+ls).mean()
    # emphasize force change over torque change
    return nll+.2*F.smooth_l1_loss(mu[:,:3],y[:,:3],beta=.4)+.6*F.smooth_l1_loss(mu[:,3:],y[:,3:],beta=.4)

ids=np.arange(len(DX)); rng=np.random.default_rng(V8D_SEED); rng.shuffle(ids); cut=int(.8*len(ids)); tr,va=ids[:cut],ids[cut:]
trdl=DataLoader(_PairDS(tr),batch_size=V8D_BATCH,shuffle=True); vadl=DataLoader(_PairDS(va),batch_size=256,shuffle=False)
v8_delta_model=V8SiameseDelta(DX.shape[1]).to(DEVICE); opt=torch.optim.AdamW(v8_delta_model.parameters(),lr=V8D_LR,weight_decay=1e-4)
bestv=float('inf'); best=None
for ep in range(V8D_EPOCHS):
    v8_delta_model.train(); tl=[]
    for x,y in trdl:
        x=x.to(DEVICE); y=y.to(DEVICE); opt.zero_grad(set_to_none=True); o=v8_delta_model(x); l=_loss(o,y); l.backward(); torch.nn.utils.clip_grad_norm_(v8_delta_model.parameters(),1.0); opt.step(); tl.append(float(l.detach().cpu()))
    v8_delta_model.eval(); vl=[]
    with torch.inference_mode():
        for x,y in vadl: vl.append(float(_loss(v8_delta_model(x.to(DEVICE)),y.to(DEVICE)).cpu()))
    vm=float(np.mean(vl)); print(f'V8-D epoch {ep+1:02d}/{V8D_EPOCHS} train={np.mean(tl):.4f} val={vm:.4f}')
    if vm<bestv: bestv=vm; best=copy.deepcopy(v8_delta_model.state_dict())
v8_delta_model.load_state_dict(best); v8_delta_model.eval()


def v8d_predict_delta(token_a,token_b,state_raw_41):
    za=np.asarray(token_a,np.float32); zb=np.asarray(token_b,np.float32)
    sr=(np.asarray(state_raw_41,np.float32)-V8C_XMEAN[256:])/V8C_XSTD[256:]
    x=np.concatenate([za,zb,zb-za,np.abs(zb-za),za*zb,sr]).astype(np.float32)
    with torch.inference_mode(): o=v8_delta_model(torch.as_tensor(x[None],dtype=torch.float32,device=DEVICE))
    mu=o['mean'][0].cpu().numpy()*DYS+DYM; sd=np.exp(o['logstd'][0].cpu().numpy())*DYS
    return dict(delta_tau=mu[:3],delta_force=mu[3:],tau_sigma=sd[:3],force_sigma=sd[3:])

# --------------------------- held-out pair diagnostics ---------------------------
rows=[]
v8_delta_model.eval()
with torch.inference_mode():
    for ii in va:
        a,b,d=V8D_PAIRS[int(ii)]; x=torch.as_tensor(DX[int(ii)][None],dtype=torch.float32,device=DEVICE); o=v8_delta_model(x)
        pm=o['mean'][0].cpu().numpy()*DYS+DYM; gt=DY[int(ii)]
        fa=Y[a,3:]; fb=Y[b,3:]; pred_fb=fa+pm[3:]
        rows.append(dict(state_distance=d,delta_tau_MAE=float(np.mean(np.abs(pm[:3]-gt[:3]))),delta_force_MAE=float(np.mean(np.abs(pm[3:]-gt[3:]))),
                         delta_force_vector_error=float(np.linalg.norm(pm[3:]-gt[3:])),
                         force_magnitude_order_correct=bool(np.sign(np.linalg.norm(pred_fb)-np.linalg.norm(fa))==np.sign(np.linalg.norm(fb)-np.linalg.norm(fa)))))
v8d_eval=pd.DataFrame(rows)
v8d_summary=pd.DataFrame([dict(pairs=len(v8d_eval),median_state_distance=v8d_eval.state_distance.median(),delta_tau_MAE=v8d_eval.delta_tau_MAE.mean(),
                                  delta_force_MAE=v8d_eval.delta_force_MAE.mean(),delta_force_vector_error=v8d_eval.delta_force_vector_error.mean(),
                                  force_magnitude_ranking_accuracy=v8d_eval.force_magnitude_order_correct.mean())])
v8d_eval.to_csv(V8D_ROOT/'V8D_MATCHED_PAIR_RAW.csv',index=False); v8d_summary.to_csv(V8D_ROOT/'V8D_MATCHED_PAIR_SUMMARY.csv',index=False)
torch.save(dict(state_dict=v8_delta_model.state_dict(),dy_mean=DYM,dy_std=DYS,input_dim=DX.shape[1]),V8D_ROOT/'V8D_SIAMESE_DELTA.pt')

print('\nV8-D MATCHED-STATE COUNTERFACTUAL METRICS')
display(v8d_summary)
print('V8-D COMPLETE:',V8D_ROOT)

# %% notebook cell 21
# ======================================================================
# V8-E — CAUSAL MICRO-FOOTHOLD PLANNER (OFFLINE SHADOW MODE)
# Run AFTER V8-D.
#
# Pipeline per future leg touchdown:
#   V8-B touchdown distribution -> local candidate grid -> causal V8-A memory
#   -> geometry + numerical-Jacobian reachability filter
#   -> V8-C physical response -> V8-D terrain-change refinement
#   -> uncertainty/risk score -> top foothold(s)
#
# This cell DOES NOT command the robot. TartanGround logs contain only the executed
# trajectory, so alternative footholds are counterfactual. Results are therefore
# reported as SHADOW-PLANNER predicted risk reduction, not real navigation success.
# ======================================================================

import itertools, math, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

_req=['run_records','v8_memory','v8b_predict','v8c_predict_cell','v8c_feature_from_memory_query','v8d_predict_delta',
      '_data_camera_pose','_world_from_heading_local','_heading_local_from_world','v8_numeric_leg_jacobian','V8_ROOT']
_missing=[x for x in _req if x not in globals()]
if _missing: raise RuntimeError('Run V8-A through V8-D first. Missing: '+repr(_missing))

V8E_ROOT=V8_ROOT/'E_microplanner'; V8E_ROOT.mkdir(parents=True,exist_ok=True)
V8E_GRID_RADIUS_M=0.22
V8E_GRID_STEP_M=0.05
V8E_MAX_SLOPE_DEG=32.0
V8E_MAX_ROUGH=0.60
V8E_MAX_Q_DELTA_RAD=0.48
V8E_TOPK_PER_LEG=4

# Dimensionless planner weights. They are engineering starting values, not claimed optimal.
W_FORCE=1.00; W_TAU=0.45; W_UNC=0.40; W_SLOPE=0.30; W_ROUGH=0.25; W_OFFSET=0.20; W_MAP_AGE=0.10


def _slope_deg(normal_world):
    n=np.asarray(normal_world,np.float64); nz=np.clip(abs(n[2])/(np.linalg.norm(n)+1e-9),0,1)
    return float(np.degrees(np.arccos(nz)))


def _candidate_reachable(rec,idx,leg,nom_local,cand_world,pose0):
    cl=_heading_local_from_world(np.asarray(cand_world,np.float32),pose0)
    dl=(cl-np.asarray(nom_local,np.float32)).astype(np.float32)
    # heading-local y is RIGHT; URDF base y is LEFT.
    db=np.array([dl[0],-dl[1],dl[2]],np.float32)
    q=rec['data']['P'][idx,leg,0:3]
    J=v8_numeric_leg_jacobian(q,leg)
    dq=np.linalg.pinv(J,rcond=1e-3)@db
    return bool(np.max(np.abs(dq))<=V8E_MAX_Q_DELTA_RAD and np.linalg.norm(dq)<=0.75),dq.astype(np.float32)


def _risk(pr,mq,offset_m):
    fn=float(np.linalg.norm(pr['force'])); tn=float(np.linalg.norm(pr['tau']))
    unc=float(np.linalg.norm(pr['force_sigma']))
    slope=_slope_deg(mq['normal']); rough=float(mq['rough']); age=min(float(mq['age_s']),5.0)
    # Normalize to approximately order-one terms.
    score=(W_FORCE*fn/300.0 + W_TAU*tn/30.0 + W_UNC*unc/80.0 + W_SLOPE*slope/35.0 +
           W_ROUGH*rough + W_OFFSET*offset_m/max(V8E_GRID_RADIUS_M,1e-6) + W_MAP_AGE*age/5.0)
    return float(score),dict(force_norm=fn,tau_norm=tn,unc_norm=unc,slope_deg=slope,rough=rough,age_s=age)


def _choose_touch_horizon(pred,leg):
    hs=np.asarray(V8B_HORIZONS); p=pred['contact_prob'][:,leg]
    eligible=np.where((hs>=0.4)&(hs<=1.5)&(p>=0.50))[0]
    if len(eligible): return int(eligible[0])
    ids=np.where((hs>=0.4)&(hs<=1.5))[0]
    return int(ids[np.argmax(p[ids])])


def _plan_leg(rec,idx,leg,pred):
    pose0,_=_data_camera_pose(rec,idx); t=float(rec['data']['time'][idx]); mem=v8_memory[(rec['terrain_type'],rec['environment'],rec['run'])]
    hi=_choose_touch_horizon(pred,leg); h=float(V8B_HORIZONS[hi]); cp=float(pred['contact_prob'][hi,leg]); nom_local=pred['mu'][hi,leg]
    nom_world=_world_from_heading_local(nom_local,pose0).astype(np.float32)
    nom_mq=mem.query(nom_world[:2],t)
    if nom_mq is None: return [],dict(horizon_s=h,contact_prob=cp,reason='nominal_not_in_causal_memory')
    nom_pr=v8c_predict_cell(rec,idx,leg,h,nom_mq)
    nom_score,nom_diag=_risk(nom_pr,nom_mq,0.0)
    state_raw=v8c_feature_from_memory_query(rec,idx,leg,h,nom_mq)[256:]

    cand=[]; seen=set()
    offs=np.arange(-V8E_GRID_RADIUS_M,V8E_GRID_RADIUS_M+1e-6,V8E_GRID_STEP_M)
    for dx in offs:
        for dy in offs:
            if dx*dx+dy*dy>V8E_GRID_RADIUS_M**2+1e-9: continue
            loc=nom_local.copy(); loc[0]+=dx; loc[1]+=dy
            w=_world_from_heading_local(loc,pose0).astype(np.float32)
            mq=mem.query(w[:2],t)
            if mq is None: continue
            cell=tuple(int(x) for x in mq['cell'])
            if cell in seen: continue
            seen.add(cell)
            slope=_slope_deg(mq['normal'])
            if slope>V8E_MAX_SLOPE_DEG or mq['rough']>V8E_MAX_ROUGH: continue
            cand_world=np.array([mq['xy'][0],mq['xy'][1],mq['z']],np.float32)
            reach,dq=_candidate_reachable(rec,idx,leg,nom_local,cand_world,pose0)
            if not reach: continue
            pr=v8c_predict_cell(rec,idx,leg,h,mq)
            # Siamese terrain-change refinement around nominal terrain.
            try:
                dd=v8d_predict_delta(nom_mq['token'],mq['token'],state_raw)
                ref_tau=nom_pr['tau']+dd['delta_tau']; ref_force=nom_pr['force']+dd['delta_force']
                pr_tau=.75*pr['tau']+.25*ref_tau; pr_force=.75*pr['force']+.25*ref_force
                pr=dict(**pr); pr['tau']=pr_tau.astype(np.float32); pr['force']=pr_force.astype(np.float32)
                pr['tau_sigma']=np.sqrt(pr['tau_sigma']**2+.25**2*dd['tau_sigma']**2)
                pr['force_sigma']=np.sqrt(pr['force_sigma']**2+.25**2*dd['force_sigma']**2)
            except Exception:
                pass
            offset=float(np.linalg.norm(cand_world[:2]-nom_world[:2])); score,diag=_risk(pr,mq,offset)
            cand.append(dict(leg=leg,horizon_s=h,contact_prob=cp,world_x=float(cand_world[0]),world_y=float(cand_world[1]),world_z=float(cand_world[2]),
                             offset_m=offset,score=score,nominal_score=nom_score,predicted_score_improvement=nom_score-score,
                             force_norm=diag['force_norm'],tau_norm=diag['tau_norm'],unc_norm=diag['unc_norm'],slope_deg=diag['slope_deg'],rough=diag['rough'],
                             map_age_s=diag['age_s'],dq_norm=float(np.linalg.norm(dq)),cell_x=cell[0],cell_y=cell[1]))
    cand=sorted(cand,key=lambda r:r['score'])
    return cand[:max(V8E_TOPK_PER_LEG,1)],dict(horizon_s=h,contact_prob=cp,nominal_score=nom_score,nominal_force_norm=nom_diag['force_norm'],
                                              nominal_tau_norm=nom_diag['tau_norm'],nominal_world=nom_world)


def _support_area(points_xy):
    p=np.asarray(points_xy,np.float64)
    if len(p)<3: return 0.0
    try: return float(ConvexHull(p).volume)   # area in 2-D
    except Exception: return 0.0

# --------------------------- 16 test shadow plans ---------------------------
cand_rows=[]; plan_rows=[]
for rec in run_records:
    for ti,tt in enumerate(rec['ttests'],1):
        idx=int(np.argmin(np.abs(rec['data']['time']-float(tt))))
        pred=v8b_predict(rec,idx)
        if pred is None: continue
        tops={}; infos={}
        for leg in range(4):
            cc,inf=_plan_leg(rec,idx,leg,pred); tops[leg]=cc; infos[leg]=inf
            for rank,r in enumerate(cc,1): cand_rows.append(dict(terrain_type=rec['terrain_type'],test_index=ti,ttest=float(tt),rank=rank,**r))
        active=[lg for lg in range(4) if len(tops[lg])>0]
        if not active: continue
        # Joint beam over top candidates. Penalize collapsed support polygons and foot crowding.
        candidate_lists=[tops[lg][:3] for lg in active]
        best=None
        for combo in itertools.product(*candidate_lists):
            pts=np.array([[r['world_x'],r['world_y']] for r in combo],np.float64)
            score=sum(r['score'] for r in combo)
            # pairwise crowding penalty
            crowd=0.0
            for a,b in itertools.combinations(range(len(pts)),2):
                d=np.linalg.norm(pts[a]-pts[b]); crowd+=max(0,0.14-d)/0.14
            area=_support_area(pts); area_pen=0.0 if len(pts)<3 else max(0,0.035-area)/0.035
            joint=score+0.35*crowd+0.35*area_pen
            if best is None or joint<best[0]: best=(joint,combo,area,crowd,area_pen)
        joint,combo,area,crowd,area_pen=best
        nom=sum(infos[lg].get('nominal_score',0.0) for lg in active)
        row=dict(terrain_type=rec['terrain_type'],test_index=ti,ttest=float(tt),active_legs=len(active),joint_score=float(joint),
                 nominal_independent_score=float(nom),predicted_risk_reduction=float(nom-joint),support_area_m2=float(area),crowding_penalty=float(crowd))
        for lg,r in zip(active,combo):
            row[f'leg{lg}_x']=r['world_x']; row[f'leg{lg}_y']=r['world_y']; row[f'leg{lg}_horizon']=r['horizon_s']; row[f'leg{lg}_score']=r['score']
        plan_rows.append(row)

v8e_candidates=pd.DataFrame(cand_rows); v8e_plan=pd.DataFrame(plan_rows)
v8e_candidates.to_csv(V8E_ROOT/'V8E_ALL_CANDIDATES_TOPK.csv',index=False)
v8e_plan.to_csv(V8E_ROOT/'V8E_16TEST_SHADOW_PLANS.csv',index=False)

print('\nV8-E SHADOW-PLANNER SUMMARY')
if len(v8e_plan):
    display(pd.DataFrame([dict(tests=len(v8e_plan),mean_predicted_risk_reduction=v8e_plan.predicted_risk_reduction.mean(),
                               median_predicted_risk_reduction=v8e_plan.predicted_risk_reduction.median(),mean_support_area_m2=v8e_plan.support_area_m2.mean())]))
else:
    print('No complete shadow plans: memory/touchdown coverage gate failed. Inspect V8-A/B metrics before changing thresholds.')
print('NOTE: predicted counterfactual improvement only — not executed navigation evidence.')
print('V8-E COMPLETE:',V8E_ROOT)

# %% notebook cell 23
# ======================================================================
# V8-F — BOUNDED CONTACT POSTERIOR / RESIDUAL MEMORY + SAFE CURIOSITY
# Run AFTER V8-E.
#
# Part 1: strict causal replay of real contacts.
#   predict -> observe real [tau,F] -> update ONLY local sparse memory -> next prediction
# No backbone/contact-field weights are changed online.
#
# Part 2: value-of-information overlay on V8-E shadow foothold candidates.
# It identifies low-risk candidates that are informative enough for a hypothetical
# low-load exploratory touch. It does NOT claim those touches were executed.
# ======================================================================

import math, json, zipfile, hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

_req=['run_records','v8_memory','v8_foot_world_proxy','v8c_predict_cell','V8_ROOT','v8e_candidates']
_missing=[x for x in _req if x not in globals()]
if _missing: raise RuntimeError('Run V8-A through V8-E first. Missing: '+repr(_missing))

V8F_ROOT=V8_ROOT/'F_active_adaptation'; V8F_ROOT.mkdir(parents=True,exist_ok=True)
V8F_HORIZON=0.1
V8F_REPLAY_HOP=0.10
V8F_KERNEL_RADIUS_M=0.30
V8F_KERNEL_L_M=0.12
V8F_TOKEN_COS_MIN=0.45
V8F_MEMORY_MAX_CELLS=800
V8F_TAU_CORR_CAP=8.0
V8F_FORCE_CORR_CAP=40.0
V8F_INFO_LAMBDA=0.10


def _cos(a,b):
    a=np.asarray(a,np.float64); b=np.asarray(b,np.float64)
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))


class V8SparseResidualMemory:
    def __init__(self,max_cells=800):
        self.max_cells=max_cells; self.cells={}; self.clock=0
    def _key(self,mq): return tuple(int(x) for x in mq['cell'])
    def query(self,mq,t):
        if not self.cells: return np.zeros(6,np.float32),0.0,0
        xy=np.asarray(mq['xy']); tok=mq['token']; num=np.zeros(6,np.float64); den=0.0; nused=0; totaln=0
        for k,s in self.cells.items():
            d=float(np.linalg.norm(xy-s['xy']))
            if d>V8F_KERNEL_RADIUS_M: continue
            c=_cos(tok,s['token'])
            if c<V8F_TOKEN_COS_MIN: continue
            age=max(0,float(t)-s['last_t'])
            w=math.exp(-.5*(d/V8F_KERNEL_L_M)**2)*(max(c,0.0)**2)*math.exp(-age/12.0)
            if w<=1e-5: continue
            num+=w*s['mean']; den+=w; nused+=1; totaln+=s['n']
        if den<=1e-9: return np.zeros(6,np.float32),0.0,0
        corr=(num/den).astype(np.float32)
        # Conservative physical caps prevent a neighbor from rewriting prediction arbitrarily.
        tn=float(np.linalg.norm(corr[:3])); fn=float(np.linalg.norm(corr[3:]))
        if tn>V8F_TAU_CORR_CAP: corr[:3]*=V8F_TAU_CORR_CAP/(tn+1e-9)
        if fn>V8F_FORCE_CORR_CAP: corr[3:]*=V8F_FORCE_CORR_CAP/(fn+1e-9)
        return corr,float(den),int(totaln)
    def update(self,mq,residual,t):
        self.clock+=1; k=self._key(mq); r=np.asarray(residual,np.float64)
        if k not in self.cells:
            self.cells[k]=dict(mean=r.copy(),m2=np.zeros(6),n=1,xy=np.asarray(mq['xy'],np.float64),token=np.asarray(mq['token'],np.float32),last_t=float(t),clock=self.clock)
        else:
            s=self.cells[k]; s['n']+=1; d=r-s['mean']; s['mean']+=d/s['n']; s['m2']+=d*(r-s['mean']);
            # slow token update keeps the cell identity stable
            s['token']=(.9*s['token']+.1*np.asarray(mq['token'],np.float32)); s['last_t']=float(t); s['clock']=self.clock
        if len(self.cells)>self.max_cells:
            # evict least-recently updated cell
            old=min(self.cells.items(),key=lambda kv:kv[1]['clock'])[0]; del self.cells[old]


# --------------------------- strict causal replay ---------------------------
rows=[]
for rec in run_records:
    data=rec['data']; mem=v8_memory[(rec['terrain_type'],rec['environment'],rec['run'])]; rm=V8SparseResidualMemory(V8F_MEMORY_MAX_CELLS)
    dt=float(np.median(np.diff(data['time']))); off=int(round(V8F_HORIZON/dt)); step=max(1,int(round(V8F_REPLAY_HOP/dt)))
    # Replay starts before first diagnostic test so later tests can benefit from prior real contacts.
    for idx in range(0,len(data['time'])-off,step):
        j=idx+off; t=float(data['time'][idx]); fw=v8_foot_world_proxy(rec,j)
        if fw is None: continue
        for leg in range(4):
            # update/score only physically loaded targets; swing zero-force is already easy and would dominate statistics
            if data['contact'][j,leg] < .5: continue
            mq=mem.query(fw[leg,:2],t)
            if mq is None: continue
            base=v8c_predict_cell(rec,idx,leg,V8F_HORIZON,mq)
            y=np.concatenate([data['P'][j,leg,6:9],data['P'][j,leg,9:12]]).astype(np.float32)
            b=np.concatenate([base['tau'],base['force']]).astype(np.float32)
            corr,w,nprev=rm.query(mq,t); adapted=b+corr
            eb=np.abs(b-y); ea=np.abs(adapted-y)
            rows.append(dict(terrain_type=rec['terrain_type'],environment=rec['environment'],run=rec['run'],time=t,leg=leg,
                             prior_neighbor_contacts=nprev,kernel_weight=w,memory_age_s=mq['age_s'],
                             base_tau_MAE=float(eb[:3].mean()),adapt_tau_MAE=float(ea[:3].mean()),
                             base_force_MAE=float(eb[3:].mean()),adapt_force_MAE=float(ea[3:].mean()),
                             improved_force=bool(ea[3:].mean()<eb[3:].mean()),improved_tau=bool(ea[:3].mean()<eb[:3].mean())))
            # Strict causality: update only AFTER scoring the prediction whose target has now been observed.
            residual=y-b
            rm.update(mq,residual,float(data['time'][j]))

v8f_replay=pd.DataFrame(rows)
if len(v8f_replay)==0: raise RuntimeError('V8-F replay produced no contact samples.')

def _bucket(n):
    if n<=0:return '0'
    if n==1:return '1'
    if n<=4:return '2-4'
    return '5+'
v8f_replay['prior_contact_bucket']=v8f_replay.prior_neighbor_contacts.map(_bucket)
v8f_curve=v8f_replay.groupby('prior_contact_bucket').agg(samples=('base_force_MAE','size'),base_force_MAE=('base_force_MAE','mean'),adapt_force_MAE=('adapt_force_MAE','mean'),
                                                            base_tau_MAE=('base_tau_MAE','mean'),adapt_tau_MAE=('adapt_tau_MAE','mean'),
                                                            force_improvement_rate=('improved_force','mean'),tau_improvement_rate=('improved_tau','mean')).reset_index()
v8f_curve['force_improvement_percent']=100*(v8f_curve.base_force_MAE-v8f_curve.adapt_force_MAE)/(v8f_curve.base_force_MAE+1e-9)
v8f_curve['tau_improvement_percent']=100*(v8f_curve.base_tau_MAE-v8f_curve.adapt_tau_MAE)/(v8f_curve.base_tau_MAE+1e-9)

# --------------------------- safe-curiosity overlay ---------------------------
voi_rows=[]
if len(v8e_candidates):
    for keys,g in v8e_candidates.groupby(['terrain_type','test_index','ttest','leg','horizon_s']):
        g=g.copy().sort_values('score'); safest=g.iloc[0]
        # Approximate information gain from predictive uncertainty, map age, and low observation confidence.
        g['info_gain']=0.5*np.log1p((g['unc_norm'].clip(lower=0)/15.0)**2)*(1.0+g['map_age_s'].clip(0,5)/5.0)
        # Safe exploration gate: never explore a candidate much riskier than the current safest one.
        safe=g[g['score']<=float(safest.score)*1.12+0.02].copy()
        safe['active_score']=safe['score']-V8F_INFO_LAMBDA*safe['info_gain']
        active=safe.sort_values('active_score').iloc[0]
        voi_rows.append(dict(terrain_type=keys[0],test_index=keys[1],ttest=keys[2],leg=keys[3],horizon_s=keys[4],
                             safest_score=float(safest.score),active_score=float(active.score),
                             safest_info_gain=float(safest.info_gain),active_info_gain=float(active.info_gain),
                             exploration_offset_m=float(active.offset_m),same_as_safest=bool(active.name==safest.name)))
v8f_voi=pd.DataFrame(voi_rows)

v8f_replay.to_csv(V8F_ROOT/'V8F_CAUSAL_CONTACT_REPLAY_RAW.csv',index=False)
v8f_curve.to_csv(V8F_ROOT/'V8F_ADAPTATION_CURVE.csv',index=False)
v8f_voi.to_csv(V8F_ROOT/'V8F_SAFE_CURIOSITY_SHADOW.csv',index=False)

print('\nV8-F CONTACT ADAPTATION CURVE')
display(v8f_curve)
print('Overall force negative-transfer rate:',float((~v8f_replay.improved_force).mean()))
if len(v8f_voi):
    print('\nV8-F SAFE-CURIOSITY SHADOW SUMMARY')
    display(pd.DataFrame([dict(groups=len(v8f_voi),fraction_different_from_safest=float((~v8f_voi.same_as_safest).mean()),
                               mean_extra_information=float((v8f_voi.active_info_gain-v8f_voi.safest_info_gain).mean()))]))

# --------------------------- package every V8 result ---------------------------
readme=V8_ROOT/'README_V8_A_TO_F.txt'
readme.write_text('''V8 A-F standalone research prototype (fresh Colab restore from persisted V7 caches)\n\nA: persistent causal four-stream terrain memory\nB: probabilistic future per-leg touchdown predictor\nC: queryable local contact-response field\nD: matched-state Siamese terrain-change head\nE: causal micro-foothold planner in OFFLINE SHADOW MODE\nF: bounded causal contact adaptation + safe-curiosity shadow policy\n\nScientific limitation: TartanGround logs do not contain executed alternative footholds. E/F counterfactual planner outputs are predictions and offline replay diagnostics, not real closed-loop navigation evidence. Exact robot deployment also requires a verified camera-to-base extrinsic.\n''')

manifest=[]
for p in sorted(V8_ROOT.rglob('*')):
    if p.is_file():
        h=hashlib.sha256(p.read_bytes()).hexdigest(); manifest.append(dict(file=str(p.relative_to(V8_ROOT)),sha256=h,size_bytes=p.stat().st_size))
pd.DataFrame(manifest).to_csv(V8_ROOT/'V8_SHA256_MANIFEST.csv',index=False)
zip_path=Path('/content/V8_A_TO_F_RESULTS.zip')
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED,allowZip64=True) as z:
    for p in sorted(V8_ROOT.rglob('*')):
        if p.is_file(): z.write(p,arcname=str(p.relative_to(V8_ROOT.parent)))

print('\n'+'='*100)
print('V8 A-F EXPERIMENT STACK COMPLETE')
print('Results:',V8_ROOT)
print('ZIP    :',zip_path)
print('Interpret E as shadow counterfactual planning; do not claim executed improvement.')
print('='*100)
try:
    from google.colab import files
    files.download(str(zip_path))
except Exception as e:
    print('Automatic download skipped:',e)

# %% notebook cell 24
# ======================================================================
# FINAL RESEARCH DASHBOARD — WHAT IS ACTUALLY VALIDATED?
# ======================================================================
rows=[]
rows.append(dict(stage='V7-F strict force mastery',pass_gate=bool(V7F_MASTER_FORCE_GATE),metric='max 3D leg-force vector error <= 5 N at 0.1 & 1.0 s'))
rows.append(dict(stage='V8-A memory',pass_gate=bool(globals().get('V8A_GATE',False)),metric=f'hit={globals().get("V8A_HIT",np.nan):.3f}, hind>=0.5s={globals().get("V8A_HIND_05",np.nan):.3f}'))
rows.append(dict(stage='V8-B touchdown',pass_gate=bool(globals().get('V8B_GATE',False)),metric=f'Recall@10cm={globals().get("V8B_RECALL10",np.nan):.3f}, F1={globals().get("V8B_F1",np.nan):.3f}'))
rows.append(dict(stage='V8-C contact field strict',pass_gate=bool(globals().get('V8C_STRICT_GATE',False)),metric='p95 3D force-vector error <= 5 N at 0.1 & 1.0 s'))
if 'v8d_summary' in globals() and len(v8d_summary):
    rows.append(dict(stage='V8-D counterfactual ranking',pass_gate=bool(v8d_summary.force_magnitude_ranking_accuracy.iloc[0]>=0.65),metric=f'ranking acc={v8d_summary.force_magnitude_ranking_accuracy.iloc[0]:.3f}'))
if 'v8e_plan' in globals() and len(v8e_plan):
    rows.append(dict(stage='V8-E shadow planner',pass_gate=bool((v8e_plan.predicted_risk_reduction>0).mean()>=0.5),metric=f'positive predicted reduction fraction={(v8e_plan.predicted_risk_reduction>0).mean():.3f}'))
if 'v8f_curve' in globals() and len(v8f_curve):
    rows.append(dict(stage='V8-F adaptation',pass_gate=bool((v8f_curve.force_improvement_percent>0).any()),metric=f'best force improvement={v8f_curve.force_improvement_percent.max():.2f}%'))

dash=pd.DataFrame(rows)
display(dash)
dash.to_csv(V8_ROOT/'V8_FINAL_RESEARCH_GATE_DASHBOARD.csv',index=False)

print('\nINTERPRETATION RULE:')
if not V7F_MASTER_FORCE_GATE:
    print('- V7-F is the strongest tested V5/V7 family in mean MAE, but your ±5 N maximum-error goal is NOT yet mastered.')
if not globals().get('V8C_STRICT_GATE',False):
    print('- V8-E/F are useful research/shadow diagnostics, not a validated precision foothold controller yet.')
else:
    print('- V8-C passed the strict force gate; next evidence needed is executed simulator/hardware counterfactual footholds.')
print('- Exact robot claims still require verified camera↔base extrinsics and executed alternative foothold tests.')
