# Source inspection export of the adjacent notebook.
# Notebook execution order and dependencies still apply.
# IPython magic, shell lines and prose-only conclusion cells are comments.

# %% notebook cell 2
# Colab already ships NumPy/Pandas/Matplotlib/PyTorch/sklearn.
# If cvxpylayers is desired for the final QP, install it explicitly after
# the fixed-iteration differentiable projector has passed all tests.
from __future__ import annotations
import dataclasses, hashlib, json, math, os, random, warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error

SEED = 731
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass(frozen=True)
class Config:
    project_root: Path = Path('/content/drive/MyDrive/UGP_V7_FULL4B')
    sample_hz: int = 100
    native_force_hz: int = 400
    history_s: float = 1.0
    horizons_s: Tuple[float, ...] = tuple(np.round(np.arange(.1, 1.01, .1), 2))
    patch_radius_m: float = .07
    patch_voxel_m: float = .015
    rollout_dt_s: float = .02
    d_model: int = 128
    dropout: float = .1
    action_mode: Literal['deployable_plan','causal_policy','oracle_logged'] = 'causal_policy'
    smoke_test: bool = True
    strict_paper_gate: bool = True
    min_independent_runs: int = 12
    target_stance_p95_n: float = 4.0
    target_touchdown_impulse_rel: float = .10

CFG = Config()
print(dataclasses.asdict(CFG)); print('device:', DEVICE)

# %% notebook cell 4
REQUIRED_KEYS = {'time','P','B','U','L_LEG','E','contact'}
FORCE_SLICE = slice(9, 12)
Q_SLICE, DQ_SLICE, TAU_SLICE = slice(0,3), slice(3,6), slice(6,9)

@dataclass
class RunRecord:
    run_id: str
    terrain: str
    data: Dict[str, np.ndarray]
    force_frame: str
    base_frame: str
    trajectory_id: str
    controller_plan_available: bool = False

def load_physics_npz(path: Path, terrain: str, force_frame: str, base_frame: str) -> RunRecord:
    z = np.load(path, allow_pickle=False)
    missing = REQUIRED_KEYS.difference(z.files)
    if missing: raise ValueError(f'{path.name}: missing {sorted(missing)}')
    data = {k: np.asarray(z[k]) for k in z.files}
    n = len(data['time'])
    assert data['P'].shape == (n,4,12), data['P'].shape
    assert data['B'].shape[0] == n and data['U'].shape[0] == n
    assert data['L_LEG'].shape == (n,4,3)
    assert data['E'].shape == (n,4,3) and data['contact'].shape == (n,4)
    if not force_frame or not base_frame:
        raise ValueError('Force/base frame IDs are mandatory; V10 had empty force frame IDs.')
    return RunRecord(path.stem, terrain, data, force_frame, base_frame, path.stem)

def sha256(path: Path, chunk=1<<20):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(chunk), b''): h.update(b)
    return h.hexdigest()

def verify_monotonic_time(run: RunRecord):
    t=np.asarray(run.data['time'], float)
    dt=np.diff(t)
    return {'run':run.run_id,'n':len(t),'dt_med_ms':1000*np.median(dt),
            'dt_p99_ms':1000*np.quantile(dt,.99),'nonpositive':int((dt<=0).sum())}

# %% notebook cell 6
def robust_force_audit(time_s, force, contact=None, target_n=4.0):
    time_s=np.asarray(time_s,float); force=np.asarray(force,float)
    if force.ndim != 3 or force.shape[1:] != (4,3): raise ValueError(force.shape)
    dt=np.diff(time_s)
    d=np.linalg.norm(np.diff(force,axis=0),axis=-1)
    slope=d/np.maximum(dt[:,None],1e-6)
    out={'dt_med_ms':1000*np.median(dt),
         'delta_vector_p95_n':np.quantile(d,.95),
         'slope_p95_n_per_s':np.quantile(slope,.95),
         'timing_for_target_ms':1000*target_n/max(np.quantile(slope,.95),1e-9)}
    if contact is not None:
        c=np.asarray(contact)>0.5
        td=(~c[:-1])&c[1:]
        out['touchdowns']=int(td.sum())
        out['touchdown_delta_p95_n']=float(np.quantile(d[td],.95)) if td.any() else np.nan
    return out

def bandlimit_then_decimate(x, native_hz=400, target_hz=100, taps=17):
    # Windowed-sinc low-pass; no SciPy dependency. Apply to labels only after
    # validating clock alignment and preserving raw/native event targets.
    x=np.asarray(x,float); ratio=native_hz//target_hz
    if native_hz % target_hz: raise ValueError('integer rate ratio required')
    fc=.45/ratio; n=np.arange(taps)-(taps-1)/2
    h=2*fc*np.sinc(2*fc*n)*np.hamming(taps); h/=h.sum()
    y=np.stack([np.convolve(x[:,i,j],h,mode='same') for i in range(4) for j in range(3)],axis=1)
    return y.reshape(len(x),4,3)[::ratio]

def touchdown_targets(time_s, force, contact, pre_ms=5, post_ms=25):
    t=np.asarray(time_s,float); f=np.asarray(force,float); c=np.asarray(contact)>0.5
    rows=[]
    for leg in range(4):
        ids=np.where((~c[:-1,leg]) & c[1:,leg])[0]+1
        for j in ids:
            lo=np.searchsorted(t,t[j]-pre_ms/1000); hi=np.searchsorted(t,t[j]+post_ms/1000)
            impulse=np.trapz(f[lo:hi,leg],t[lo:hi],axis=0)
            rows.append((leg,j,t[j],*impulse))
    return pd.DataFrame(rows,columns=['leg','index','t_touchdown','Jx','Jy','Jz'])

# %% notebook cell 8
def assign_group_splits(records: Sequence[RunRecord], test_ids: Sequence[str], val_ids: Sequence[str]):
    test_ids,val_ids=set(test_ids),set(val_ids)
    if test_ids & val_ids: raise ValueError('test and validation trajectories overlap')
    rows=[]
    for r in records:
        split='test' if r.trajectory_id in test_ids else ('val' if r.trajectory_id in val_ids else 'train')
        rows.append({'run_id':r.run_id,'trajectory_id':r.trajectory_id,'terrain':r.terrain,'split':split})
    frame=pd.DataFrame(rows)
    assert frame.groupby('trajectory_id').split.nunique().max()==1
    return frame

def paper_data_gate(split_manifest: pd.DataFrame, min_runs=12):
    problems=[]
    if split_manifest.trajectory_id.nunique()<min_runs:
        problems.append(f'only {split_manifest.trajectory_id.nunique()} independent trajectories; require >= {min_runs}')
    if set(split_manifest.split)!={'train','val','test'}:
        problems.append('train/val/test are not all populated')
    for terrain,g in split_manifest.groupby('terrain'):
        if g.trajectory_id.nunique()<2: problems.append(f'{terrain}: only one trajectory')
    return {'pass':not problems,'problems':problems}

# %% notebook cell 10
class CausalPatchMemory:
    def __init__(self, xyz, normal, geom, timestamp, voxel=.015):
        self.xyz=np.asarray(xyz,np.float32); self.normal=np.asarray(normal,np.float32)
        self.geom=np.asarray(geom,np.float32); self.timestamp=np.asarray(timestamp,np.float64)
        self.voxel=float(voxel)
        if not (len(self.xyz)==len(self.normal)==len(self.geom)==len(self.timestamp)): raise ValueError('length mismatch')

    def query_one(self, mean_xyz, cov_xyz, origin_t, radius=.07, sigma_points=13):
        mean=np.asarray(mean_xyz,float); cov=np.asarray(cov_xyz,float)+np.eye(3)*1e-8
        vals,vecs=np.linalg.eigh(cov); vals=np.maximum(vals,1e-8)
        pts=[mean]
        for k in range(3):
            d=math.sqrt(3*vals[k])*vecs[:,k]; pts += [mean+d,mean-d]
        rng=np.random.default_rng(0)
        while len(pts)<sigma_points: pts.append(rng.multivariate_normal(mean,cov))
        pooled=[]
        for q in pts:
            past=self.timestamp<=origin_t
            dist=np.linalg.norm(self.xyz[:,:2]-q[None,:2],axis=1)
            ids=np.where(past & (dist<=radius))[0]
            if len(ids)==0:
                pooled.append(np.r_[np.zeros(self.geom.shape[1]+6),1.0]); continue
            w=np.exp(-.5*(dist[ids]/max(radius/2,1e-6))**2); w/=w.sum()
            z=self.xyz[ids,2]; n=(self.normal[ids]*w[:,None]).sum(0); n/=max(np.linalg.norm(n),1e-6)
            g=(self.geom[ids]*w[:,None]).sum(0)
            pooled.append(np.r_[g,z.mean(),z.std(),n,len(ids)/100,0.0])
        pooled=np.asarray(pooled)
        return np.r_[pooled.mean(0),pooled.std(0)]

# %% notebook cell 12
class MLP(nn.Module):
    def __init__(self, din, dout, hidden=128, depth=2):
        super().__init__(); layers=[]
        for i in range(depth-1): layers += [nn.Linear(din if i==0 else hidden,hidden),nn.SiLU(),nn.LayerNorm(hidden)]
        layers += [nn.Linear(din if depth==1 else hidden,dout)]; self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

class CausalHistoryEncoder(nn.Module):
    def __init__(self,din,d=128,layers=3,heads=4,max_t=128):
        super().__init__(); self.proj=nn.Linear(din,d); self.pos=nn.Parameter(torch.zeros(1,max_t,d))
        block=nn.TransformerEncoderLayer(d,heads,4*d,.1,batch_first=True,norm_first=True,activation='gelu')
        self.enc=nn.TransformerEncoder(block,layers); self.norm=nn.LayerNorm(d)
    def forward(self,x):
        z=self.proj(x)+self.pos[:,:x.shape[1]]
        mask=torch.triu(torch.ones(x.shape[1],x.shape[1],device=x.device,dtype=torch.bool),1)
        return self.norm(self.enc(z,mask=mask)[:,-1])

class FourLegRelationalBlock(nn.Module):
    def __init__(self,d=128):
        super().__init__(); self.edge=MLP(2*d+3,d); self.node=MLP(2*d,d); self.body=MLP(2*d,d)
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

def cross_matrix(r):
    z=torch.zeros_like(r[...,0]); x,y,zz=r.unbind(-1)
    return torch.stack([z,-zz,y,zz,z,-x,-y,x,z],-1).reshape(*r.shape[:-1],3,3)

class JointContactProjector(nn.Module):
    def __init__(self,iterations=3,eps=1e-3): super().__init__(); self.iterations=iterations; self.eps=eps
    def forward(self,f_free,foot_r,normal,mu,contact_prob,wrench):
        # Inputs: [B,H,4,3], wrench [B,H,6]. The fixed-iteration solve is
        # differentiable and deterministic; replace/verify with cvxpylayers for publication.
        b,h,l,_=f_free.shape; eye=torch.eye(6,device=f_free.device,dtype=f_free.dtype).expand(b,h,6,6)
        I3=torch.eye(3,device=f_free.device,dtype=f_free.dtype).expand(b,h,l,3,3)
        A=torch.cat([I3,cross_matrix(foot_r)],-2).permute(0,1,3,2,4).reshape(b,h,6,12)
        f=f_free
        for _ in range(self.iterations):
            flat=f.reshape(b,h,12,1); err=wrench[...,None]-A@flat
            corr=A.transpose(-1,-2)@torch.linalg.solve(A@A.transpose(-1,-2)+self.eps*eye,err)
            f=(flat+corr).reshape(b,h,l,3)
            n=F.normalize(normal,dim=-1); fn=(f*n).sum(-1,keepdim=True)
            fn=F.softplus(fn, beta=8.0)*contact_prob[...,None]
            ft=f-(f*n).sum(-1,keepdim=True)*n
            scale=torch.clamp(mu[...,None]*fn/(torch.linalg.norm(ft,dim=-1,keepdim=True)+1e-6),max=1.0)
            f=fn*n+scale*ft
        residual=wrench-(A@f.reshape(b,h,12,1)).squeeze(-1)
        return f,residual

class V10R(nn.Module):
    def __init__(self,leg_dim=25,body_dim=18,action_dim=18,patch_dim=64,d=128,horizons=10):
        super().__init__(); self.H=horizons
        self.leg_hist=CausalHistoryEncoder(leg_dim,d); self.body_hist=CausalHistoryEncoder(body_dim,d)
        self.rel=FourLegRelationalBlock(d); self.action=MLP(action_dim,d)
        self.step_embed=nn.Parameter(torch.randn(horizons,d)*.01)
        self.leg_step=nn.GRUCell(2*d,d); self.body_step=nn.GRUCell(2*d,d)
        # mode logits, event hazard, foot delta, foot Cholesky parameters,
        # and mean/log-scale for q,dq,tau (9 physical values).
        self.proposal_head=MLP(3*d,4+1+3+6+18)
        self.patch=MLP(patch_dim,d); self.force_free=MLP(4*d,9) # mean, log-scale, impulse per leg
        self.project=JointContactProjector()
        self.body_out=MLP(3*d,24) # mean/log-scale for the 12-D body/IMU target

    def propose(self,leg_hist,body_hist,action_plan,foot_now):
        b=leg_hist.shape[0]
        leg=torch.stack([self.leg_hist(leg_hist[:,i]) for i in range(4)],1)
        body=self.body_hist(body_hist)
        actions=self.action(action_plan)
        prop=[]; body_pred=[]; leg_lat=[]; body_lat=[]
        foot_state=foot_now
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
            prop.append(pk); body_pred.append(self.body_out(torch.cat([body,a,leg.mean(1)],-1)))
            leg_lat.append(leg); body_lat.append(body)
        prop=torch.stack(prop,1); body_pred=torch.stack(body_pred,1)
        leg_lat=torch.stack(leg_lat,1); body_lat=torch.stack(body_lat,1)
        mode_logits=prop[...,:4]; contact_prob=1-torch.softmax(mode_logits,-1)[...,0]
        # Proposal deltas are recurrently accumulated to preserve trajectory consistency.
        foot_mean=foot_now[:,None]+torch.cumsum(prop[...,5:8],dim=1)
        return {'mode_logits':mode_logits,'event_hazard_logit':prop[...,4],
                'contact_prob':contact_prob,'foot_mean':foot_mean,
                'foot_chol_raw':prop[...,8:14],'leg_sensor_raw':prop[...,14:],
                'body_raw':body_pred,'latent_leg':leg_lat,'latent_body':body_lat,
                'latent_action':actions}

    def decode_force(self,proposal,patch_features,normal,mu,desired_wrench):
        patch_lat=self.patch(patch_features)
        dec=torch.cat([proposal['latent_leg'],
                       proposal['latent_body'][:,:,None].expand(-1,-1,4,-1),
                       proposal['latent_action'][:,:,None].expand(-1,-1,4,-1),patch_lat],-1)
        force_raw=self.force_free(dec); f_free=force_raw[...,:3]
        f_proj,wrench_resid=self.project(f_free,proposal['foot_mean'],normal,mu,
                                         proposal['contact_prob'],desired_wrench)
        return {'force_mean':f_proj,'force_log_scale':force_raw[...,3:6],
                'impulse':force_raw[...,6:9],'wrench_residual':wrench_resid}

    def forward(self,leg_hist,body_hist,action_plan,foot_now,patch_features,normal,mu,desired_wrench):
        # In a real run, first call propose(), query the causal map at its
        # foot distribution, then call decode_force(). This wrapper is for
        # precomputed patch tensors and deterministic smoke tests.
        proposal=self.propose(leg_hist,body_hist,action_plan,foot_now)
        return {**proposal,**self.decode_force(proposal,patch_features,normal,mu,desired_wrench)}

# %% notebook cell 14
def pinball(pred,y,q):
    e=y-pred; return torch.maximum(q*e,(q-1)*e).mean()

def diagonal_gaussian_nll(raw,target,clamp=(-7.,5.)):
    mean,log_scale=raw.chunk(2,dim=-1)
    log_scale=log_scale.clamp(*clamp); scale=torch.exp(log_scale)
    return (.5*((target-mean)/scale).square()+log_scale).mean()

def v10r_loss(out,target,weights=None):
    w=weights or {'force':1.,'event':.5,'foot':.4,'sensor':.5,'impulse':.8,'wrench':.2}
    phase=target['phase'].long()
    event=F.cross_entropy(out['mode_logits'].reshape(-1,4),phase.reshape(-1))
    force=F.smooth_l1_loss(out['force_mean'],target['force'],beta=4.0,reduction='none')
    # Predeclared phase weights; do not tune on test.
    phase_w=torch.tensor([.35,2.5,1.0,1.5],device=force.device)[phase][...,None]
    force=(force*phase_w).mean()
    foot=F.smooth_l1_loss(out['foot_mean'],target['foot'],beta=.02)
    impulse=F.smooth_l1_loss(out['impulse'],target['impulse'],beta=.25)
    sensors=diagonal_gaussian_nll(out['leg_sensor_raw'],target['leg_sensor'])
    sensors=sensors+diagonal_gaussian_nll(out['body_raw'],target['body_sensor'])
    wrench=torch.linalg.norm(out['wrench_residual'],dim=-1).mean()
    total=(w['force']*force+w['event']*event+w['foot']*foot+w['impulse']*impulse+
           w['sensor']*sensors+w['wrench']*wrench)
    return {'total':total,'force':force,'event':event,'foot':foot,
            'impulse':impulse,'sensor':sensors,'wrench':wrench}

TRAINING_ORDER = [
    'G0 timestamp/frame/extrinsic/inertia audit; freeze calibration',
    'G1 exact-state force ceiling and label identifiability',
    'G2 causal state/contact/foot distribution rollout',
    'G3 TSDF/elevation patch mapper and cross-modal haptic posterior',
    'G4 constrained all-leg force + event impulse decoder',
    'G5 blocked uncertainty calibration; freeze model and thresholds',
    'G6 zero-shot, prequential, simulator counterfactual and hardware shadow tests',
]
print('\n'.join(TRAINING_ORDER))

# %% notebook cell 16
def synthetic_batch(b=3,hist=50,H=10,patch_dim=64):
    leg_hist=torch.randn(b,4,hist,25,device=DEVICE)
    body_hist=torch.randn(b,hist,18,device=DEVICE)
    action=torch.randn(b,H,18,device=DEVICE)
    foot=torch.tensor([[[-.35,.22,-.48],[.35,.22,-.48],[-.35,-.22,-.48],[.35,-.22,-.48]]],device=DEVICE).expand(b,-1,-1).float()
    patch=torch.randn(b,H,4,patch_dim,device=DEVICE)
    normal=torch.zeros(b,H,4,3,device=DEVICE); normal[...,2]=1
    mu=torch.full((b,H,4),.65,device=DEVICE)
    wrench=torch.zeros(b,H,6,device=DEVICE); wrench[...,2]=420
    return leg_hist,body_hist,action,foot,patch,normal,mu,wrench

if CFG.smoke_test:
    model=V10R(d=CFG.d_model,horizons=len(CFG.horizons_s)).to(DEVICE)
    args=synthetic_batch(H=len(CFG.horizons_s)); out=model(*args)
    assert out['force_mean'].shape==(3,10,4,3)
    assert out['foot_mean'].shape==(3,10,4,3)
    target={'phase':torch.randint(0,4,(3,10,4),device=DEVICE),
            'force':torch.randn(3,10,4,3,device=DEVICE),
            'foot':torch.randn(3,10,4,3,device=DEVICE),
            'impulse':torch.randn(3,10,4,3,device=DEVICE),
            'leg_sensor':torch.randn(3,10,4,9,device=DEVICE),
            'body_sensor':torch.randn(3,10,12,device=DEVICE)}
    losses=v10r_loss(out,target); losses['total'].backward()
    finite=all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    print({k:tuple(v.shape) for k,v in out.items() if torch.is_tensor(v)})
    print({k:float(v.detach()) for k,v in losses.items()}); print('finite gradients:',finite)

# %% notebook cell 18
def exact_state_features(run: RunRecord):
    d=run.data; P=np.asarray(d['P'],np.float32); B=np.asarray(d['B'],np.float32)
    q,dq,tau=P[...,Q_SLICE],P[...,DQ_SLICE],P[...,TAU_SLICE]
    # Only present-time values; add verified URDF-derived J, M, h and base acceleration in the final experiment.
    X=np.c_[q.reshape(len(P),-1),dq.reshape(len(P),-1),tau.reshape(len(P),-1),B,
            np.asarray(d['U'],np.float32),np.asarray(d['L_LEG'],np.float32).reshape(len(P),-1),
            np.asarray(d['E'],np.float32).reshape(len(P),-1),np.asarray(d['contact'],np.float32)]
    y=P[...,FORCE_SLICE].reshape(len(P),-1)
    return X,y

def blocked_exact_state_ceiling(train_runs, test_run, max_train=120_000):
    Xs,ys=zip(*(exact_state_features(r) for r in train_runs)); X=np.concatenate(Xs); y=np.concatenate(ys)
    if len(X)>max_train:
        ids=np.random.default_rng(SEED).choice(len(X),max_train,replace=False); X,y=X[ids],y[ids]
    Xt,yt=exact_state_features(test_run)
    model=ExtraTreesRegressor(n_estimators=256,min_samples_leaf=3,max_features=.8,n_jobs=-1,random_state=SEED)
    model.fit(X,y); pred=model.predict(Xt); err=np.abs(pred-yt)
    return {'component_mae_n':err.mean(),'component_p95_n':np.quantile(err,.95),
            'within_4n':(err<=4).mean(),'max_n':err.max()},model

# %% notebook cell 20
def future_action_sequence(run: RunRecord, i: int, horizon_idx: Sequence[int], mode: str,
                           controller_plan: Optional[np.ndarray]=None, causal_policy=None):
    d=run.data; now=np.r_[d['U'][i],d['L_LEG'][i].reshape(-1)]
    if mode=='deployable_plan':
        if controller_plan is None or not run.controller_plan_available:
            raise ValueError('No plan was available at forecast origin')
        return np.asarray(controller_plan,np.float32)
    if mode=='causal_policy':
        if causal_policy is None: return np.repeat(now[None],len(horizon_idx),axis=0) # declared hold-last baseline
        return np.asarray(causal_policy(run,i,horizon_idx),np.float32)
    if mode=='oracle_logged':
        warnings.warn('ORACLE-ACTION BENCHMARK: not a deployable forecast')
        return np.stack([np.r_[d['U'][j],d['L_LEG'][j].reshape(-1)] for j in horizon_idx])
    raise ValueError(mode)

# %% notebook cell 22
def causal_rate(x,time_s,clip=None):
    x=np.asarray(x,np.float32); t=np.asarray(time_s,float); out=np.zeros_like(x)
    dt=np.maximum(np.diff(t),1e-6)
    out[1:]=np.diff(x,axis=0)/dt.reshape((-1,)+(1,)*(x.ndim-1))
    out[0]=out[1]
    return np.clip(out,-clip,clip) if clip is not None else out

def leg_history_features(run: RunRecord):
    d=run.data; P=np.asarray(d['P'],np.float32); t=d['time']
    E=np.asarray(d['E'],np.float32); Fm=P[...,FORCE_SLICE]
    dE=causal_rate(E,t,clip=20.); dF=causal_rate(Fm,t,clip=20_000.)
    # 25 = P(12) + contact(1) + local command(3) + E(3) + dE(3) + dF(3).
    x=np.concatenate([P,np.asarray(d['contact'])[...,None],np.asarray(d['L_LEG']),E,dE,dF],axis=-1)
    assert x.shape[1:]==(4,25),x.shape
    return x

def body_history_features(run: RunRecord):
    x=np.c_[np.asarray(run.data['B'],np.float32),np.asarray(run.data['U'],np.float32)]
    if x.shape[1]!=18: raise ValueError(f'expected B(12)+U(6)=18, got {x.shape}')
    return x

def contact_phase_at(contact,indices):
    c=np.asarray(contact)>0.5; indices=np.asarray(indices,int); prev=c[np.maximum(indices-1,0)]; cur=c[indices]
    phase=np.where(cur,2,0).astype(np.int64)
    phase[(~prev)&cur]=1; phase[prev&(~cur)]=3
    return phase

def valid_origins(run: RunRecord,cfg=CFG):
    n=len(run.data['time']); hist=int(round(cfg.history_s*cfg.sample_hz))
    max_future=int(round(max(cfg.horizons_s)*cfg.sample_hz))
    return np.arange(hist-1,n-max_future,dtype=int)

class TrajectoryWindowDataset(torch.utils.data.Dataset):
    def __init__(self,run: RunRecord,origins: Sequence[int],cfg=CFG):
        self.run,self.origins,self.cfg=run,np.asarray(origins,int),cfg
        needed={'foot_world','desired_wrench','touchdown_impulse'}
        missing=needed.difference(run.data)
        if missing: raise ValueError('audited preprocessing missing: '+', '.join(sorted(missing)))
        self.leg_x=leg_history_features(run); self.body_x=body_history_features(run)
        self.hist=int(round(cfg.history_s*cfg.sample_hz))
        self.offset=np.array([int(round(h*cfg.sample_hz)) for h in cfg.horizons_s])

    def __len__(self): return len(self.origins)
    def __getitem__(self,k):
        i=int(self.origins[k]); js=i+self.offset; d=self.run.data
        plan=None
        if self.cfg.action_mode=='deployable_plan':
            plan=np.asarray(d['controller_plan'][i],np.float32)
        action=future_action_sequence(self.run,i,js,self.cfg.action_mode,controller_plan=plan)
        phase=contact_phase_at(d['contact'],js)
        item={'leg_hist':self.leg_x[i-self.hist+1:i+1].transpose(1,0,2),
              'body_hist':self.body_x[i-self.hist+1:i+1], 'action_plan':action,
              'foot_now':np.asarray(d['foot_world'][i],np.float32),
              'origin_t':np.float32(d['time'][i]),
              'desired_wrench':np.asarray(d['desired_wrench'][js],np.float32),
              'target':{'phase':phase,'force':np.asarray(d['P'][js,:,FORCE_SLICE],np.float32),
                        'foot':np.asarray(d['foot_world'][js],np.float32),
                        'impulse':np.asarray(d['touchdown_impulse'][js],np.float32),
                        'leg_sensor':np.asarray(d['P'][js,:,:9],np.float32),
                        'body_sensor':np.asarray(d['B'][js],np.float32)}}
        return torch.utils.data.default_convert(item)

def foot_covariance(raw):
    # Raw order: L00,L10,L11,L20,L21,L22. Softplus diagonal enforces SPD.
    L=torch.zeros(*raw.shape[:-1],3,3,device=raw.device,dtype=raw.dtype)
    L[...,0,0]=F.softplus(raw[...,0])+.002
    L[...,1,0]=raw[...,1]; L[...,1,1]=F.softplus(raw[...,2])+.002
    L[...,2,0]=raw[...,3]; L[...,2,1]=raw[...,4]; L[...,2,2]=F.softplus(raw[...,5])+.002
    return L@L.transpose(-1,-2)

def train_step(model,batch,patch_provider,optimizer):
    # patch_provider(mean,cov,origin_t) must return feature[B,H,4,D],
    # normal[B,H,4,3], mu[B,H,4] using only map observations <= origin_t.
    for key in ('leg_hist','body_hist','action_plan','foot_now','origin_t','desired_wrench'):
        batch[key]=batch[key].to(DEVICE)
    target={k:v.to(DEVICE) for k,v in batch['target'].items()}
    proposal=model.propose(batch['leg_hist'],batch['body_hist'],batch['action_plan'],batch['foot_now'])
    cov=foot_covariance(proposal['foot_chol_raw'])
    patch,normal,mu=patch_provider(proposal['foot_mean'].detach(),cov.detach(),batch['origin_t'])
    out={**proposal,**model.decode_force(proposal,patch.to(DEVICE),normal.to(DEVICE),mu.to(DEVICE),batch['desired_wrench'])}
    loss=v10r_loss(out,target); optimizer.zero_grad(set_to_none=True); loss['total'].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
    return {k:float(v.detach()) for k,v in loss.items()}

# %% notebook cell 24
@dataclass(frozen=True)
class PlannerWeights:
    collision: float=8.; reach: float=6.; support: float=5.; slip: float=4.
    force_cvar: float=3.; impulse_cvar: float=3.; epistemic: float=2.; energy: float=.5; deviation: float=.8

def cvar(values,alpha=.9,axis=-1):
    values=np.sort(np.asarray(values),axis=axis); k=max(1,int(math.ceil((1-alpha)*values.shape[axis])))
    return np.take(values,range(values.shape[axis]-k,values.shape[axis]),axis=axis).mean(axis=axis)

def score_candidates(candidate_table: pd.DataFrame, w=PlannerWeights()):
    c=candidate_table.copy()
    c['hard_reject']=(c.reach_violation>0)|(c.collision_prob>.2)|(c.map_confidence<.6)
    c['score']=(w.collision*c.collision_prob+w.reach*c.reach_violation+w.support*c.support_risk+
                w.slip*c.slip_prob+w.force_cvar*c.force_cvar+w.impulse_cvar*c.impulse_cvar+
                w.epistemic*c.epistemic+w.energy*c.energy+w.deviation*c.nominal_offset)
    c.loc[c.hard_reject,'score']=np.inf
    return c.sort_values(['leg','score'])

def make_xy_candidates(nominal_xyz, half_width=.08, spacing=.02):
    nominal_xyz=np.asarray(nominal_xyz,float); d=np.arange(-half_width,half_width+1e-9,spacing)
    return np.array([nominal_xyz+np.array([dx,dy,0]) for dx in d for dy in d])

def plot_leg_candidate_map(scored: pd.DataFrame,leg='FL'):
    g=scored[scored.leg==leg]
    fig,ax=plt.subplots(figsize=(6,5)); ok=np.isfinite(g.score)
    pts=ax.scatter(g.x,g.y,c=np.where(ok,g.score,np.nan),cmap='RdYlGn_r',s=70,edgecolor='white')
    if ok.any():
        best=g.loc[g.score.idxmin()]; ax.scatter(best.x,best.y,marker='*',s=280,c='#17365D',label='selected')
    ax.scatter(g.loc[~ok,'x'],g.loc[~ok,'y'],marker='x',c='#A23B3B',label='hard reject')
    ax.set_aspect('equal'); ax.set_xlabel('world x [m]'); ax.set_ylabel('world y [m]')
    ax.set_title(f'{leg}: joint-support-aware contact-patch risk'); ax.legend(); fig.colorbar(pts,ax=ax,label='fixed risk score')
    plt.show()

if CFG.smoke_test:
    xyz=make_xy_candidates(np.array([-.35,.22,0.])); demo=pd.DataFrame({'leg':'FL','x':xyz[:,0],'y':xyz[:,1]})
    demo['collision_prob']=np.exp(-((demo.x+.30)**2+(demo.y-.26)**2)/.0008)
    demo['reach_violation']=np.maximum(np.hypot(demo.x+.35,demo.y-.22)-.10,0)
    demo['map_confidence']=.9; demo['support_risk']=.2+2*(demo.y-.22)**2
    demo['slip_prob']=.1; demo['force_cvar']=.2+4*(demo.x+.37)**2; demo['impulse_cvar']=.15
    demo['epistemic']=.1; demo['energy']=.2; demo['nominal_offset']=np.hypot(demo.x+.35,demo.y-.22)
    plot_leg_candidate_map(score_candidates(demo))

# %% notebook cell 26
def force_metric_frame(y,p,phase,leg_names=('FL','FR','RL','RR'),horizons=CFG.horizons_s):
    y,p=np.asarray(y),np.asarray(p); e=np.abs(p-y); rows=[]
    for h,hs in enumerate(horizons):
        for l,name in enumerate(leg_names):
            for a,axis in enumerate('xyz'):
                for ph,phname in enumerate(('swing','touchdown','stance','liftoff')):
                    m=np.asarray(phase)[:,h,l]==ph
                    if not m.any(): continue
                    q=e[m,h,l,a]
                    rows.append([hs,name,axis,phname,len(q),q.mean(),np.quantile(q,.95),(q<=3).mean(),(q<=4).mean(),q.max()])
    return pd.DataFrame(rows,columns=['horizon_s','leg','axis','phase','n','mae_n','p95_n','within_3n','within_4n','max_n'])

PAPER_GATES = {
    'stable_stance_force': 'component p95 <= 4 N with block-bootstrap 95% upper bound <= 5 N',
    'touchdown': 'median |time error| <= 2 ms AND impulse relative error <= 10%; raw-force p95 reported',
    'uncertainty': '80/90/95% intervals within ±3 percentage points per phase on untouched runs',
    'all_sensors': 'beats persistence and V7F at every horizon with corrected multiple comparisons',
    'planner': 'lower executed/simulator-reset CVaR without higher fall/slip/collision rate',
}
pd.Series(PAPER_GATES,name='pre-registered criterion')

# %% notebook cell 29
STATUS = {
    'architecture_scaffold':'implemented',
    'synthetic_interface_test':'run above when enabled',
    'current_v10_overall_component_mae_n':5.7934,
    'current_v10_overall_component_p95_n':23.306,
    'current_v10_touchdown_component_p95_n':60.845,
    'all_regime_3_to_4n_claim':'NOT ESTABLISHED',
    'planner_execution_claim':'NOT ESTABLISHED',
}
print(json.dumps(STATUS,indent=2))

# %% notebook cell 30


# %% notebook cell 31
# ============================================================
# V10-R G0 — real-data and scientific-contract preflight
# Read-only: no downloads, deletion, training, or file changes
# ============================================================
from pathlib import Path
import json
import numpy as np
import pandas as pd
from IPython.display import display

PROJECT_ROOT = Path("/content/drive/MyDrive/UGP_V7_FULL4B")
CACHE_ROOT = PROJECT_ROOT / "cache"
RAW_ROOT = Path("/content/tartanground_v10")
CONTRACT_ROOT = PROJECT_ROOT / "V10R_STRICT_STAGED" / "contracts"

RUN_SPECS = [
    {
        "run_id": "ForestEnv_P2000",
        "terrain": "forest",
        "environment": "ForestEnv",
        "run": "P2000",
    },
    {
        "run_id": "GreatMarsh_P2023",
        "terrain": "swamp_marsh",
        "environment": "GreatMarsh",
        "run": "P2023",
    },
    {
        "run_id": "SeasonalForestWinter_P2000",
        "terrain": "rough_uneven",
        "environment": "SeasonalForestWinter",
        "run": "P2000",
    },
    {
        "run_id": "OldTownFall_P2000",
        "terrain": "wet_uneven_proxy",
        "environment": "OldTownFall",
        "run": "P2000",
    },
]

def read_json_if_present(path):
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)

def force_contract_pass(contract):
    if not (
        contract.get("verified") is True
        and contract.get("verified_before_split") is True
    ):
        return False

    per_leg = contract.get("per_leg", {})
    for leg in ("FL", "FR", "RL", "RR"):
        item = per_leg.get(leg, {})
        if item.get("source_frame") not in {
            "world", "odom", "base", "sensor", "foot"
        }:
            return False
        if item.get("direction") not in {
            "ground_on_robot", "robot_on_ground"
        }:
            return False
        if item.get("static_tf_test_passed") is not True:
            return False
        if item.get("direction_sign_test_passed") is not True:
            return False
    return True

def calibration_contract_pass(contract):
    if contract.get("verified") is not True:
        return False
    if contract.get("base_pose_adapter_verified") is not True:
        return False
    if not Path(contract.get("urdf_path", "")).exists():
        return False

    cameras = contract.get("cameras", {})
    required_cameras = (
        "lcam_front", "lcam_back", "lcam_left",
        "lcam_right", "lcam_top", "lcam_bottom",
    )
    for camera in required_cameras:
        c = cameras.get(camera, {})
        if c.get("verified") is not True:
            return False
        R = np.asarray(c.get("R_C_B", []), dtype=float)
        t = np.asarray(c.get("t_C_B_m", []), dtype=float)
        if R.shape != (3, 3) or t.shape != (3,):
            return False
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-3):
            return False
        if not np.isclose(np.linalg.det(R), 1.0, atol=1e-3):
            return False
    return True

def signal_contract_pass(contract):
    return bool(
        contract.get("verified") is True
        and contract.get("semantics_evidence_verified") is True
        and contract.get("imu_specific_force_frame") == "base"
        and contract.get("imu_angular_velocity_frame") == "base"
        and contract.get("twist_frame") == "base"
    )

force_contract = read_json_if_present(
    CONTRACT_ROOT / "force_contract.json"
)
calibration_contract = read_json_if_present(
    CONTRACT_ROOT / "calibration_contract.json"
)
signal_contract = read_json_if_present(
    CONTRACT_ROOT / "signal_contract.json"
)

V10R_CONTRACT_STATUS = {
    "force": force_contract_pass(force_contract),
    "calibration": calibration_contract_pass(calibration_contract),
    "signal": signal_contract_pass(signal_contract),
}
V10R_G0_PASS = all(V10R_CONTRACT_STATUS.values())

rows = []
V10R_CACHE_PATHS = {}

for spec in RUN_SPECS:
    run_id = spec["run_id"]
    cache = CACHE_ROOT / f"{run_id}_physics.npz"
    raw_run = (
        RAW_ROOT
        / spec["environment"]
        / "Data_anymal"
        / spec["run"]
    )
    bag = raw_run / "rosbags" / f"{spec['run']}.bag"

    row = {
        **spec,
        "cache_path": str(cache),
        "cache_exists": cache.exists(),
        "raw_run_exists": raw_run.exists(),
        "rosbag_exists": bag.exists(),
        "rgb_exists": (raw_run / "image_lcam_front").exists(),
        "depth_exists": (raw_run / "depth_lcam_front").exists(),
        "pose_exists": (raw_run / "pose_lcam_front").exists(),
        "cache_valid": False,
        "n": 0,
        "median_dt_ms": np.nan,
        "nonpositive_dt": np.nan,
        "force_finite": False,
        "has_foot_world": False,
        "has_desired_wrench": False,
        "has_touchdown_impulse": False,
    }

    if cache.exists():
        try:
            with np.load(cache, allow_pickle=False) as z:
                keys = set(z.files)
                required_keys = {
                    "time", "P", "B", "U", "L_LEG", "E", "contact"
                }
                row["keys"] = ",".join(sorted(keys))
                row["has_foot_world"] = "foot_world" in keys
                row["has_desired_wrench"] = "desired_wrench" in keys
                row["has_touchdown_impulse"] = (
                    "touchdown_impulse" in keys
                )

                if required_keys.issubset(keys):
                    time = np.asarray(z["time"], dtype=np.float64)
                    P = np.asarray(z["P"], dtype=np.float32)
                    B = np.asarray(z["B"], dtype=np.float32)
                    U = np.asarray(z["U"], dtype=np.float32)
                    L = np.asarray(z["L_LEG"], dtype=np.float32)
                    E = np.asarray(z["E"], dtype=np.float32)
                    contact = np.asarray(z["contact"])

                    n = len(time)
                    shapes_ok = (
                        P.shape == (n, 4, 12)
                        and B.shape[0] == n
                        and U.shape[0] == n
                        and L.shape == (n, 4, 3)
                        and E.shape == (n, 4, 3)
                        and contact.shape == (n, 4)
                    )

                    dt = np.diff(time)
                    row["n"] = n
                    row["median_dt_ms"] = (
                        float(np.median(dt) * 1000.0)
                        if len(dt) else np.nan
                    )
                    row["nonpositive_dt"] = int(np.sum(dt <= 0))
                    row["force_finite"] = bool(
                        np.isfinite(P[:, :, 9:12]).all()
                    )
                    row["cache_valid"] = bool(
                        shapes_ok
                        and row["nonpositive_dt"] == 0
                        and row["force_finite"]
                    )

            V10R_CACHE_PATHS[run_id] = cache

        except Exception as exc:
            row["read_error"] = repr(exc)

    rows.append(row)

V10R_DATA_PREFLIGHT = pd.DataFrame(rows)

display(
    V10R_DATA_PREFLIGHT[
        [
            "run_id", "cache_exists", "cache_valid", "n",
            "median_dt_ms", "raw_run_exists", "rosbag_exists",
            "rgb_exists", "depth_exists",
            "has_foot_world", "has_desired_wrench",
            "has_touchdown_impulse",
        ]
    ]
)

print("\nScientific contract status:")
for name, passed in V10R_CONTRACT_STATUS.items():
    print(f"  {name:12s}: {passed}")

print("\nG0 scientific contract pass:", V10R_G0_PASS)
print(
    "Independent trajectories:",
    len(V10R_DATA_PREFLIGHT),
    "/ required for paper:",
    CFG.min_independent_runs,
)

cache_pass = bool(V10R_DATA_PREFLIGHT["cache_valid"].all())
prepared_pass = bool(
    V10R_DATA_PREFLIGHT[
        [
            "has_foot_world",
            "has_desired_wrench",
            "has_touchdown_impulse",
        ]
    ].all().all()
)

print("All physics caches valid:", cache_pass)
print("All model-specific targets prepared:", prepared_pass)

if not V10R_G0_PASS:
    print(
        "\nSTOP: force frames, camera/base calibration, and/or signal "
        "semantics are not yet evidence-verified. Transformer training "
        "would not be physically interpretable."
    )
elif not cache_pass:
    print("\nSTOP: at least one physics cache is missing or invalid.")
elif not prepared_pass:
    print(
        "\nNEXT: caches are valid, but foot_world, desired_wrench, and "
        "touchdown_impulse still need to be generated."
    )
else:
    print("\nREADY: run the exact-state ceiling cell next.")

preflight_path = "/content/v10r_real_data_preflight.csv"
V10R_DATA_PREFLIGHT.to_csv(preflight_path, index=False)
print("\nSaved:", preflight_path)

# %% notebook cell 32
# ============================================================
# V10-R G1 — exact-state LOTO diagnostic ceiling
# This is not transformer training.
# ============================================================
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from IPython.display import display

if "V10R_G0_PASS" not in globals():
    raise RuntimeError("Run the G0 real-data preflight cell first.")

if not V10R_G0_PASS:
    raise RuntimeError(
        "G0 failed. Resolve force-frame, calibration, and signal "
        "contracts before interpreting Fx/Fy/Fz."
    )

if not V10R_DATA_PREFLIGHT["cache_valid"].all():
    raise RuntimeError("At least one required physics cache is invalid.")

def load_numeric_run(spec, path):
    with np.load(path, allow_pickle=False) as z:
        return {
            "spec": spec,
            "time": np.asarray(z["time"], dtype=np.float64),
            "P": np.asarray(z["P"], dtype=np.float32),
            "B": np.asarray(z["B"], dtype=np.float32),
            "U": np.asarray(z["U"], dtype=np.float32),
            "L_LEG": np.asarray(z["L_LEG"], dtype=np.float32),
            "E": np.asarray(z["E"], dtype=np.float32),
            "contact": np.asarray(z["contact"], dtype=bool),
        }

def exact_features(run):
    P = run["P"]
    return np.concatenate(
        [
            P[:, :, 0:9].reshape(len(P), -1),
            run["B"].reshape(len(P), -1),
            run["U"].reshape(len(P), -1),
            run["L_LEG"].reshape(len(P), -1),
            run["E"].reshape(len(P), -1),
            run["contact"].astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)

def force_target(run):
    return run["P"][:, :, 9:12].reshape(len(run["P"]), -1)

def contact_phase(contact):
    contact = np.asarray(contact, dtype=bool)
    previous = np.vstack([contact[:1], contact[:-1]])

    phase = np.zeros(contact.shape, dtype=np.int64)  # swing
    phase[contact] = 2                               # stance
    phase[(~previous) & contact] = 1                 # touchdown
    phase[previous & (~contact)] = 3                 # liftoff
    return phase

def block_bootstrap_p95_upper(
    error,
    phase_mask,
    block_samples=500,
    repetitions=200,
    seed=1234,
):
    blocks = []
    n = len(error)

    for start in range(0, n, block_samples):
        stop = min(start + block_samples, n)
        block_error = error[start:stop]
        block_mask = phase_mask[start:stop]
        selected = block_error[block_mask].reshape(-1)
        if selected.size:
            blocks.append(selected)

    if len(blocks) < 2:
        return np.nan

    rng = np.random.default_rng(seed)
    estimates = []

    for _ in range(repetitions):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        sample = np.concatenate([blocks[i] for i in chosen])
        estimates.append(np.quantile(sample, 0.95))

    return float(np.quantile(estimates, 0.95))

runs = []
for spec in RUN_SPECS:
    path = V10R_CACHE_PATHS.get(spec["run_id"])
    if path is None or not Path(path).exists():
        raise FileNotFoundError(
            f"Missing cache for {spec['run_id']}"
        )
    runs.append(load_numeric_run(spec, path))

metric_rows = []
fold_rows = []

for test_index, test_run in enumerate(runs):
    train_runs = [
        run for i, run in enumerate(runs)
        if i != test_index
    ]

    X_train = np.concatenate(
        [exact_features(run) for run in train_runs],
        axis=0,
    )
    y_train = np.concatenate(
        [force_target(run) for run in train_runs],
        axis=0,
    )

    X_test = exact_features(test_run)
    y_test = force_target(test_run)

    model_ceiling = ExtraTreesRegressor(
        n_estimators=256,
        min_samples_leaf=2,
        max_features=1.0,
        n_jobs=-1,
        random_state=CFG.seed + test_index,
    )
    model_ceiling.fit(X_train, y_train)

    prediction = model_ceiling.predict(X_test).reshape(-1, 4, 3)
    truth = y_test.reshape(-1, 4, 3)
    error = np.abs(prediction - truth)
    phase = contact_phase(test_run["contact"])

    phase_definitions = {
        "all": None,
        "swing": 0,
        "touchdown": 1,
        "stable_stance": 2,
        "liftoff": 3,
    }

    for phase_name, phase_code in phase_definitions.items():
        for leg_index, leg in enumerate(("FL", "FR", "RL", "RR")):
            if phase_code is None:
                sample_mask = np.ones(
                    len(error), dtype=bool
                )
            else:
                sample_mask = phase[:, leg_index] == phase_code

            for axis_index, axis in enumerate(("Fx", "Fy", "Fz")):
                values = error[
                    sample_mask, leg_index, axis_index
                ]

                if len(values) == 0:
                    continue

                metric_rows.append({
                    "test_run": test_run["spec"]["run_id"],
                    "phase": phase_name,
                    "leg": leg,
                    "axis": axis,
                    "n": len(values),
                    "mae_N": float(np.mean(values)),
                    "median_N": float(np.median(values)),
                    "p95_N": float(np.quantile(values, 0.95)),
                    "max_N": float(np.max(values)),
                })

    stance_mask = phase == 2
    touchdown_mask = phase == 1

    stance_values = error[stance_mask].reshape(-1)
    touchdown_values = error[touchdown_mask].reshape(-1)

    fold_rows.append({
        "test_run": test_run["spec"]["run_id"],
        "all_component_mae_N": float(error.mean()),
        "all_component_p95_N": float(np.quantile(error, 0.95)),
        "stance_component_p95_N": (
            float(np.quantile(stance_values, 0.95))
            if stance_values.size else np.nan
        ),
        "touchdown_component_p95_N": (
            float(np.quantile(touchdown_values, 0.95))
            if touchdown_values.size else np.nan
        ),
        "stance_p95_bootstrap_upper_N":
            block_bootstrap_p95_upper(
                error,
                stance_mask,
                block_samples=int(5 * CFG.sample_hz),
                repetitions=200,
                seed=CFG.seed + test_index,
            ),
    })

V10R_CEILING_METRICS = pd.DataFrame(metric_rows)
V10R_CEILING_FOLDS = pd.DataFrame(fold_rows)

print("Exact-state leave-one-terrain-out summary:")
display(V10R_CEILING_FOLDS)

stance_table = V10R_CEILING_METRICS[
    V10R_CEILING_METRICS["phase"] == "stable_stance"
].copy()

print("\nWorst stable-stance leg/axis cases:")
display(
    stance_table.sort_values("p95_N", ascending=False).head(20)
)

ceiling_path = "/content/v10r_exact_state_loto_metrics.csv"
fold_path = "/content/v10r_exact_state_loto_fold_summary.csv"

V10R_CEILING_METRICS.to_csv(ceiling_path, index=False)
V10R_CEILING_FOLDS.to_csv(fold_path, index=False)

print("Saved:", ceiling_path)
print("Saved:", fold_path)

worst_stance_p95 = float(stance_table["p95_N"].max())
worst_stance_upper = float(
    V10R_CEILING_FOLDS["stance_p95_bootstrap_upper_N"].max()
)

print("\nWorst leg/axis stable-stance p95:", worst_stance_p95, "N")
print("Worst fold bootstrap upper bound:", worst_stance_upper, "N")

if worst_stance_p95 > 4.0:
    print(
        "\nG1 RESULT: the exact-state ceiling misses the 4 N target. "
        "Do not expect the causal vision model to solve this merely by "
        "training longer. Investigate synchronization, force labels, "
        "trajectory diversity, and state/force observability."
    )
else:
    print(
        "\nG1 PRELIMINARY PASS: the exact-state state-to-force mapping "
        "reaches the numerical target. This permits proceeding to "
        "causal-history and geometry ablations, but is not itself a "
        "paper result."
    )

print(
    "\nPaper limitation: only four independent trajectories are present; "
    f"the configured paper gate requires {CFG.min_independent_runs}."
)

# %% notebook cell 33
# ============================================================
# REPLACEMENT R0 — corrected V10-R scaffold audit
# ============================================================
import inspect
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from IPython.display import display

required = [
    "CFG", "DEVICE", "V10R", "synthetic_batch",
    "v10r_loss", "foot_covariance"
]
missing = [name for name in required if name not in globals()]
if missing:
    raise RuntimeError(
        "Run the original V10-R notebook cells first. Missing: "
        + ", ".join(missing)
    )

AUDIT_SEED = int(globals().get("SEED", 731)) + 101
torch.manual_seed(AUDIT_SEED)
np.random.seed(AUDIT_SEED)

H = len(CFG.horizons_s)
history_steps = int(round(CFG.history_s * CFG.sample_hz))
PATCH_DIM = 64

# Correct constructor names from the original V10-R class.
audit_model = V10R(
    d=CFG.d_model,
    horizons=H,
    patch_dim=PATCH_DIM,
).to(DEVICE)

args = synthetic_batch(
    b=2,
    hist=history_steps,
    H=H,
    patch_dim=PATCH_DIM,
)

rows = []

def record(test, status, value, criterion):
    rows.append({
        "test": test,
        "status": status,
        "value": value,
        "criterion": criterion,
    })

# ------------------------------------------------------------
# 1. Forward pass, dimensions and finite values
# ------------------------------------------------------------
audit_model.eval()

with torch.no_grad():
    out = audit_model(*args)

expected_shapes = {
    "mode_logits": (2, H, 4, 4),
    "event_hazard_logit": (2, H, 4),
    "contact_prob": (2, H, 4),
    "foot_mean": (2, H, 4, 3),
    "foot_chol_raw": (2, H, 4, 6),
    "leg_sensor_raw": (2, H, 4, 18),
    "body_raw": (2, H, 24),
    "force_mean": (2, H, 4, 3),
    "force_log_scale": (2, H, 4, 3),
    "impulse": (2, H, 4, 3),
    "wrench_residual": (2, H, 6),
}

for name, expected in expected_shapes.items():
    actual = tuple(out[name].shape)
    record(
        f"shape/{name}",
        "PASS" if actual == expected else "FAIL",
        str(actual),
        str(expected),
    )

all_finite = all(
    torch.isfinite(value).all().item()
    for value in out.values()
    if torch.is_tensor(value)
)
record(
    "all outputs finite",
    "PASS" if all_finite else "FAIL",
    str(all_finite),
    "True",
)

# ------------------------------------------------------------
# 2. Positive-definite predicted foot covariance
# ------------------------------------------------------------
with torch.no_grad():
    covariance = foot_covariance(out["foot_chol_raw"])
    minimum_eigenvalue = float(
        torch.linalg.eigvalsh(covariance).min().item()
    )

record(
    "foot covariance positive definite",
    "PASS" if minimum_eigenvalue > 0.0 else "FAIL",
    f"{minimum_eigenvalue:.6e}",
    "> 0",
)

# ------------------------------------------------------------
# 3. Four-leg permutation equivariance
# ------------------------------------------------------------
permutation = torch.tensor([2, 0, 3, 1], device=DEVICE)

permuted_args = (
    args[0][:, permutation],
    args[1],
    args[2],
    args[3][:, permutation],
    args[4][:, :, permutation],
    args[5][:, :, permutation],
    args[6][:, :, permutation],
    args[7],
)

with torch.no_grad():
    permuted_out = audit_model(*permuted_args)

leg_outputs = [
    "mode_logits",
    "event_hazard_logit",
    "contact_prob",
    "foot_mean",
    "foot_chol_raw",
    "leg_sensor_raw",
    "force_mean",
    "force_log_scale",
    "impulse",
]

permutation_errors = []

for name in leg_outputs:
    reference = out[name][:, :, permutation]
    permutation_errors.append(
        float((permuted_out[name] - reference).abs().max().item())
    )

permutation_errors.extend([
    float(
        (permuted_out["body_raw"] - out["body_raw"])
        .abs().max().item()
    ),
    float(
        (
            permuted_out["wrench_residual"]
            - out["wrench_residual"]
        ).abs().max().item()
    ),
])

maximum_permutation_error = max(permutation_errors)

record(
    "four-leg permutation equivariance",
    "PASS" if maximum_permutation_error < 1e-4 else "FAIL",
    f"{maximum_permutation_error:.6e}",
    "< 1e-4",
)

# ------------------------------------------------------------
# 4. Contact constraints
# ------------------------------------------------------------
normal = F.normalize(args[5], dim=-1)
force = out["force_mean"]
mu = args[6]

normal_force = (force * normal).sum(dim=-1)
tangent = force - normal_force.unsqueeze(-1) * normal
tangent_norm = torch.linalg.vector_norm(tangent, dim=-1)

minimum_normal_force = float(normal_force.min().item())
friction_excess = float(
    (
        tangent_norm
        - mu * normal_force.clamp_min(0.0)
    ).clamp_min(0.0).max().item()
)

record(
    "unilateral force constraint",
    "PASS" if minimum_normal_force >= -1e-5 else "FAIL",
    f"{minimum_normal_force:.6e} N",
    "minimum normal force >= -1e-5 N",
)

record(
    "friction-cone constraint",
    "PASS" if friction_excess < 1e-4 else "FAIL",
    f"{friction_excess:.6e} N",
    "maximum violation < 1e-4 N",
)

with torch.no_grad():
    zero_contact_force, _ = audit_model.project(
        torch.randn_like(force),
        args[3][:, None].expand(-1, H, -1, -1),
        args[5],
        args[6],
        torch.zeros_like(out["contact_prob"]),
        torch.zeros_like(args[7]),
    )

zero_contact_error = float(zero_contact_force.abs().max().item())

record(
    "zero-contact force suppression",
    "PASS" if zero_contact_error < 1e-6 else "FAIL",
    f"{zero_contact_error:.6e} N",
    "< 1e-6 N",
)

# ------------------------------------------------------------
# 5. Determine which output heads receive loss gradients
# ------------------------------------------------------------
audit_model.train()
gradient_out = audit_model(*args)

gradient_names = [
    "mode_logits",
    "event_hazard_logit",
    "foot_mean",
    "foot_chol_raw",
    "force_mean",
    "force_log_scale",
    "impulse",
]

for name in gradient_names:
    gradient_out[name].retain_grad()

# Correct targets:
# leg_sensor_raw = mean(9) + log_scale(9)
# body_raw       = mean(12) + log_scale(12)
target = {
    "phase": torch.randint(
        0, 4, (2, H, 4), device=DEVICE
    ),
    "force": torch.randn(
        2, H, 4, 3, device=DEVICE
    ) * 80.0,
    "foot": args[3][:, None].expand(
        -1, H, -1, -1
    ) + 0.02 * torch.randn(
        2, H, 4, 3, device=DEVICE
    ),
    "impulse": torch.randn(
        2, H, 4, 3, device=DEVICE
    ),
    "leg_sensor": torch.randn(
        2, H, 4, 9, device=DEVICE
    ),
    "body_sensor": torch.randn(
        2, H, 12, device=DEVICE
    ),
}

audit_model.zero_grad(set_to_none=True)
losses = v10r_loss(gradient_out, target)
losses["total"].backward()

def gradient_sum(tensor):
    if tensor.grad is None:
        return 0.0
    return float(tensor.grad.abs().sum().item())

gradient_results = {
    name: gradient_sum(gradient_out[name])
    for name in gradient_names
}

for name, value in gradient_results.items():
    record(
        f"loss supervises {name}",
        "PASS" if value > 0.0 else "FAIL",
        f"{value:.6e}",
        "nonzero gradient",
    )

parameter_gradients_finite = all(
    parameter.grad is None
    or torch.isfinite(parameter.grad).all().item()
    for parameter in audit_model.parameters()
)

record(
    "parameter gradients finite",
    "PASS" if parameter_gradients_finite else "FAIL",
    str(parameter_gradients_finite),
    "True",
)

# ------------------------------------------------------------
# 6. Paper/code consistency
# ------------------------------------------------------------
required_internal_steps = int(
    round(max(CFG.horizons_s) / CFG.rollout_dt_s)
)
implemented_steps = int(audit_model.H)

record(
    "20 ms internal recurrent rollout",
    "PASS" if implemented_steps == required_internal_steps else "FAIL",
    f"implemented={implemented_steps}, required={required_internal_steps}",
    "50 internal steps for 1.0 s",
)

decode_signature = str(
    inspect.signature(audit_model.decode_force)
)
has_com_input = "com" in decode_signature.lower()

record(
    "COM-relative moment arm",
    "PASS" if has_com_input else "BLOCKED",
    decode_signature,
    "decode_force must receive COM or precomputed moment arms",
)

V10R_SCAFFOLD_AUDIT = pd.DataFrame(rows)
display(V10R_SCAFFOLD_AUDIT)

print("\nLoss values are synthetic and are NOT accuracy results:")
print({
    name: float(value.detach().cpu())
    for name, value in losses.items()
})

print("\nOutput-gradient sums:")
for name, value in gradient_results.items():
    print(f"{name:24s}: {value:.6e}")

V10R_SCAFFOLD_AUDIT.to_csv(
    "/content/v10r_scaffold_audit_corrected.csv",
    index=False,
)

# %% notebook cell 34
# ============================================================
# REPLACEMENT ENVIRONMENT RECOVERY — mount and bounded discovery
# ============================================================
from pathlib import Path
import importlib.util
import pandas as pd
from IPython.display import display

if importlib.util.find_spec("google.colab") is not None:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)

PROJECT_ROOT = Path("/content/drive/MyDrive/UGP_V7_FULL4B")
RAW_ROOT = Path("/content/tartanground_v10")
CONTRACT_ROOT = (
    PROJECT_ROOT / "V10R_STRICT_STAGED" / "contracts"
)

RUN_SPECS = [
    {
        "run_id": "ForestEnv_P2000",
        "environment": "ForestEnv",
        "run": "P2000",
        "terrain": "forest",
    },
    {
        "run_id": "GreatMarsh_P2023",
        "environment": "GreatMarsh",
        "run": "P2023",
        "terrain": "swamp_marsh",
    },
    {
        "run_id": "SeasonalForestWinter_P2000",
        "environment": "SeasonalForestWinter",
        "run": "P2000",
        "terrain": "rough_uneven",
    },
    {
        "run_id": "OldTownFall_P2000",
        "environment": "OldTownFall",
        "run": "P2000",
        "terrain": "wet_uneven_proxy",
    },
]

CACHE_ROOTS = [
    PROJECT_ROOT / "cache",
    PROJECT_ROOT / "V10_CACHE",
    PROJECT_ROOT / "V7_FULL4B" / "cache",
    PROJECT_ROOT,
]

def locate_exact_file(filename):
    for root in CACHE_ROOTS:
        candidate = root / filename
        if candidate.is_file():
            return candidate

    # Bounded fallback—never search the whole Drive.
    for root in CACHE_ROOTS[:3]:
        if root.is_dir():
            matches = sorted(root.rglob(filename))
            if matches:
                return matches[0]

    return None

DISCOVERED_CACHE_PATHS = {}
discovery_rows = []

for spec in RUN_SPECS:
    run_id = spec["run_id"]

    physics = locate_exact_file(
        f"{run_id}_physics.npz"
    )
    manifest = locate_exact_file(
        f"{run_id}_frame_manifest.npz"
    )

    raw_run = (
        RAW_ROOT
        / spec["environment"]
        / "Data_anymal"
        / spec["run"]
    )
    bag = raw_run / "rosbags" / f"{spec['run']}.bag"

    if physics is not None:
        DISCOVERED_CACHE_PATHS[run_id] = physics

    discovery_rows.append({
        "run_id": run_id,
        "physics_cache": str(physics or ""),
        "physics_exists": physics is not None,
        "physics_MiB": (
            round(physics.stat().st_size / 1024**2, 3)
            if physics is not None else 0.0
        ),
        "frame_manifest": str(manifest or ""),
        "manifest_exists": manifest is not None,
        "raw_run_exists": raw_run.is_dir(),
        "rosbag_exists": bag.is_file(),
    })

DISCOVERY_TABLE = pd.DataFrame(discovery_rows)
display(DISCOVERY_TABLE)

CONTRACT_PATHS = {
    "force": CONTRACT_ROOT / "force_frame_contract.json",
    "calibration": CONTRACT_ROOT / "calibration_contract.json",
    "signals": CONTRACT_ROOT / "signal_semantics_contract.json",
}

print("\nCanonical contract files:")
for name, path in CONTRACT_PATHS.items():
    print(f"{name:12s}: exists={path.is_file()}  {path}")

print("\nImportant:")
print("- Drive caches should persist across runtimes.")
print("- /content/tartanground_v10 is temporary and may correctly be absent.")
print("- Absence of raw data does not imply the Drive physics caches are absent.")

# %% notebook cell 35
# ============================================================
# REPLACEMENT G0 — corrected real-data and contract preflight
# ============================================================
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from IPython.display import display

LEG_NAMES = ("FL", "FR", "RL", "RR")
CAMERA_NAMES = (
    "lcam_front", "lcam_back", "lcam_left",
    "lcam_right", "lcam_top", "lcam_bottom",
)
CANONICAL_JOINT_NAMES = (
    "LF_HAA", "LF_HFE", "LF_KFE",
    "RF_HAA", "RF_HFE", "RF_KFE",
    "LH_HAA", "LH_HFE", "LH_KFE",
    "RH_HAA", "RH_HFE", "RH_KFE",
)

def read_json(path):
    path = Path(path)
    if not path.is_file():
        return {}
    with path.open("r") as file:
        return json.load(file)

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def validate_force_contract(contract):
    errors = []

    if contract.get("schema") != "V10R_FORCE_FRAME_CONTRACT_V1":
        errors.append("wrong or missing force schema")
    if contract.get("verified") is not True:
        errors.append("verified must be true")
    if contract.get("verified_before_split") is not True:
        errors.append("verified_before_split must be true")

    evidence = str(contract.get("evidence", ""))
    if len(evidence.strip()) < 12 or "REPLACE" in evidence:
        errors.append("auditable force-frame evidence missing")

    if contract.get("force_direction") not in {
        "environment_on_robot",
        "robot_on_environment",
    }:
        errors.append("force direction unresolved")

    per_leg = contract.get("per_leg", {})

    for leg in LEG_NAMES:
        item = per_leg.get(leg, {})

        if item.get("source_frame") not in {
            "base", "world_ned", "foot"
        }:
            errors.append(f"{leg}: source frame unresolved")

        permutation = item.get("axis_permutation")
        if sorted(permutation or []) != [0, 1, 2]:
            errors.append(f"{leg}: invalid axis permutation")

        signs = item.get("axis_sign")
        if (
            len(signs or []) != 3
            or any(value not in (-1, 1) for value in signs)
        ):
            errors.append(f"{leg}: invalid axis signs")

        if item.get("static_test_passed") is not True:
            errors.append(f"{leg}: static/sign test not passed")

    return errors

def validate_calibration_contract(contract):
    errors = []

    if contract.get("schema") != "V10R_CALIBRATION_CONTRACT_V1":
        errors.append("wrong or missing calibration schema")
    if contract.get("verified") is not True:
        errors.append("calibration verified must be true")

    evidence = str(contract.get("evidence", ""))
    if len(evidence.strip()) < 12 or "REPLACE" in evidence:
        errors.append("auditable calibration evidence missing")

    rotation = np.asarray(
        contract.get("R_C_B", []), dtype=float
    )
    translation = np.asarray(
        contract.get("t_C_B_m", []), dtype=float
    )

    if (
        rotation.shape != (3, 3)
        or not np.allclose(
            rotation.T @ rotation,
            np.eye(3),
            atol=1e-4,
        )
        or not np.isclose(
            np.linalg.det(rotation),
            1.0,
            atol=1e-4,
        )
    ):
        errors.append("R_C_B is not a verified proper rotation")

    if (
        translation.shape != (3,)
        or not np.isfinite(translation).all()
    ):
        errors.append("t_C_B_m is invalid")

    urdf_path = Path(str(contract.get("urdf_path", "")))

    if not urdf_path.is_file():
        errors.append(f"URDF unavailable in this runtime: {urdf_path}")
    elif contract.get("urdf_sha256") != sha256(urdf_path):
        errors.append("URDF SHA-256 does not match contract")

    joint_names = contract.get("joint_names", {})
    if set(joint_names) != set(CANONICAL_JOINT_NAMES):
        errors.append("canonical joint mapping incomplete")

    foot_frames = contract.get("foot_frame_names", {})
    if (
        set(foot_frames) != set(LEG_NAMES)
        or any(
            not str(value).strip()
            or "REPLACE" in str(value)
            for value in foot_frames.values()
        )
    ):
        errors.append("foot-frame mapping incomplete")

    for camera in CAMERA_NAMES:
        intrinsics = (
            contract.get("camera_intrinsics", {})
            .get(camera, {})
        )
        if intrinsics.get("verified") is not True:
            errors.append(f"{camera}: intrinsics unverified")

    return errors

def validate_signal_contract(contract):
    errors = []

    if contract.get("schema") != "V10R_SIGNAL_CONTRACT_V1":
        errors.append("wrong or missing signal schema")
    if contract.get("verified") is not True:
        errors.append("signal contract is not verified")
    if contract.get("world_frame") != "NED":
        errors.append("world frame unresolved")
    if contract.get("camera_axes") != "x_forward_y_right_z_down":
        errors.append("camera axes unresolved")
    if contract.get("pose_format") != "tx_ty_tz_qx_qy_qz_qw":
        errors.append("pose format unresolved")
    if (
        contract.get("imu_acceleration_semantics")
        != "specific_force_body"
    ):
        errors.append("IMU acceleration semantics unresolved")
    if contract.get("imu_gyro_frame") != "base":
        errors.append("IMU gyro frame unresolved")
    if contract.get("twist_frame") != "base":
        errors.append("twist frame unresolved")

    evidence = str(contract.get("evidence", ""))
    if len(evidence.strip()) < 12 or "REPLACE" in evidence:
        errors.append("auditable signal evidence missing")

    return errors

cache_rows = []

for spec in RUN_SPECS:
    run_id = spec["run_id"]
    path = DISCOVERED_CACHE_PATHS.get(run_id)

    row = {
        "run_id": run_id,
        "path": str(path or ""),
        "cache_valid": False,
        "n": 0,
        "median_dt_ms": np.nan,
        "clock_strictly_increasing": False,
        "has_foot_world": False,
        "has_desired_wrench": False,
        "has_touchdown_impulse": False,
        "error": "",
    }

    if path is not None:
        try:
            with np.load(path, allow_pickle=False) as archive:
                keys = set(archive.files)
                required = {
                    "time", "P", "B", "U",
                    "L_LEG", "E", "contact",
                }

                row["has_foot_world"] = "foot_world" in keys
                row["has_desired_wrench"] = (
                    "desired_wrench" in keys
                )
                row["has_touchdown_impulse"] = (
                    "touchdown_impulse" in keys
                )

                if not required.issubset(keys):
                    row["error"] = (
                        "missing keys: "
                        + ", ".join(sorted(required - keys))
                    )
                else:
                    time = np.asarray(
                        archive["time"], dtype=np.float64
                    )
                    P = np.asarray(
                        archive["P"], dtype=np.float32
                    )
                    B = np.asarray(
                        archive["B"], dtype=np.float32
                    )
                    U = np.asarray(
                        archive["U"], dtype=np.float32
                    )
                    L = np.asarray(
                        archive["L_LEG"], dtype=np.float32
                    )
                    E = np.asarray(
                        archive["E"], dtype=np.float32
                    )
                    contact = np.asarray(archive["contact"])

                    n = len(time)
                    dt = np.diff(time)

                    shapes_ok = (
                        P.shape == (n, 4, 12)
                        and B.shape[0] == n
                        and U.shape[0] == n
                        and L.shape == (n, 4, 3)
                        and E.shape == (n, 4, 3)
                        and contact.shape == (n, 4)
                    )

                    finite = all([
                        np.isfinite(time).all(),
                        np.isfinite(P).all(),
                        np.isfinite(B).all(),
                        np.isfinite(U).all(),
                        np.isfinite(L).all(),
                        np.isfinite(E).all(),
                    ])

                    row["n"] = n
                    row["median_dt_ms"] = (
                        float(np.median(dt) * 1000.0)
                        if len(dt) else np.nan
                    )
                    row["clock_strictly_increasing"] = bool(
                        len(dt) and np.all(dt > 0)
                    )
                    row["cache_valid"] = bool(
                        shapes_ok
                        and finite
                        and row["clock_strictly_increasing"]
                    )

        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"

    cache_rows.append(row)

V10R_CACHE_AUDIT = pd.DataFrame(cache_rows)
display(V10R_CACHE_AUDIT)

force_contract = read_json(CONTRACT_PATHS["force"])
calibration_contract = read_json(
    CONTRACT_PATHS["calibration"]
)
signal_contract = read_json(CONTRACT_PATHS["signals"])

V10R_CONTRACT_ERRORS = {
    "force": validate_force_contract(force_contract),
    "calibration": validate_calibration_contract(
        calibration_contract
    ),
    "signals": validate_signal_contract(signal_contract),
}

contract_table = pd.DataFrame([
    {
        "contract": name,
        "pass": len(errors) == 0,
        "problems": " | ".join(errors),
    }
    for name, errors in V10R_CONTRACT_ERRORS.items()
])
display(contract_table)

V10R_CACHE_PASS = bool(
    len(V10R_CACHE_AUDIT) == 4
    and V10R_CACHE_AUDIT["cache_valid"].all()
)
V10R_G0_PASS = not any(V10R_CONTRACT_ERRORS.values())
V10R_PREPARED_TARGETS_PASS = bool(
    V10R_CACHE_AUDIT[
        [
            "has_foot_world",
            "has_desired_wrench",
            "has_touchdown_impulse",
        ]
    ].all().all()
)

print("\nFINAL PREFLIGHT DECISION")
print("Physics caches valid:", V10R_CACHE_PASS)
print("Physical G0 contracts pass:", V10R_G0_PASS)
print(
    "Model-specific targets prepared:",
    V10R_PREPARED_TARGETS_PASS,
)
print(
    "Independent trajectories:",
    len(V10R_CACHE_AUDIT),
    f"/ {CFG.min_independent_runs} required for paper mode",
)

if not V10R_CACHE_PASS:
    print("\nSTOP: the persistent cache layer is unavailable or invalid.")
elif not V10R_G0_PASS:
    print(
        "\nPHYSICAL TRAINING REMAINS BLOCKED. "
        "A numerical channel-only diagnostic may run, but its channels "
        "must not be called physical Fx/Fy/Fz."
    )
elif not V10R_PREPARED_TARGETS_PASS:
    print(
        "\nG0 PASSED, but geometry/contact preprocessing is still "
        "required before transformer training."
    )
else:
    print("\nReady for the physical exact-state ceiling.")

V10R_CACHE_AUDIT.to_csv(
    "/content/v10r_corrected_cache_audit.csv",
    index=False,
)
contract_table.to_csv(
    "/content/v10r_corrected_contract_audit.csv",
    index=False,
)

# %% notebook cell 36
# ============================================================
# V10-R G0 — real-data and scientific-contract preflight
# Read-only: no downloads, deletion, training, or file changes
# ============================================================
from pathlib import Path
import json
import numpy as np
import pandas as pd
from IPython.display import display

PROJECT_ROOT = Path("/content/drive/MyDrive/UGP_V7_FULL4B")
CACHE_ROOT = PROJECT_ROOT / "cache"
RAW_ROOT = Path("/content/tartanground_v10")
CONTRACT_ROOT = PROJECT_ROOT / "V10R_STRICT_STAGED" / "contracts"

RUN_SPECS = [
    {
        "run_id": "ForestEnv_P2000",
        "terrain": "forest",
        "environment": "ForestEnv",
        "run": "P2000",
    },
    {
        "run_id": "GreatMarsh_P2023",
        "terrain": "swamp_marsh",
        "environment": "GreatMarsh",
        "run": "P2023",
    },
    {
        "run_id": "SeasonalForestWinter_P2000",
        "terrain": "rough_uneven",
        "environment": "SeasonalForestWinter",
        "run": "P2000",
    },
    {
        "run_id": "OldTownFall_P2000",
        "terrain": "wet_uneven_proxy",
        "environment": "OldTownFall",
        "run": "P2000",
    },
]

def read_json_if_present(path):
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)

def force_contract_pass(contract):
    if not (
        contract.get("verified") is True
        and contract.get("verified_before_split") is True
    ):
        return False

    per_leg = contract.get("per_leg", {})
    for leg in ("FL", "FR", "RL", "RR"):
        item = per_leg.get(leg, {})
        if item.get("source_frame") not in {
            "world", "odom", "base", "sensor", "foot"
        }:
            return False
        if item.get("direction") not in {
            "ground_on_robot", "robot_on_ground"
        }:
            return False
        if item.get("static_tf_test_passed") is not True:
            return False
        if item.get("direction_sign_test_passed") is not True:
            return False
    return True

def calibration_contract_pass(contract):
    if contract.get("verified") is not True:
        return False
    if contract.get("base_pose_adapter_verified") is not True:
        return False
    if not Path(contract.get("urdf_path", "")).exists():
        return False

    cameras = contract.get("cameras", {})
    required_cameras = (
        "lcam_front", "lcam_back", "lcam_left",
        "lcam_right", "lcam_top", "lcam_bottom",
    )
    for camera in required_cameras:
        c = cameras.get(camera, {})
        if c.get("verified") is not True:
            return False
        R = np.asarray(c.get("R_C_B", []), dtype=float)
        t = np.asarray(c.get("t_C_B_m", []), dtype=float)
        if R.shape != (3, 3) or t.shape != (3,):
            return False
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-3):
            return False
        if not np.isclose(np.linalg.det(R), 1.0, atol=1e-3):
            return False
    return True

def signal_contract_pass(contract):
    return bool(
        contract.get("verified") is True
        and contract.get("semantics_evidence_verified") is True
        and contract.get("imu_specific_force_frame") == "base"
        and contract.get("imu_angular_velocity_frame") == "base"
        and contract.get("twist_frame") == "base"
    )

force_contract = read_json_if_present(
    CONTRACT_ROOT / "force_contract.json"
)
calibration_contract = read_json_if_present(
    CONTRACT_ROOT / "calibration_contract.json"
)
signal_contract = read_json_if_present(
    CONTRACT_ROOT / "signal_contract.json"
)

V10R_CONTRACT_STATUS = {
    "force": force_contract_pass(force_contract),
    "calibration": calibration_contract_pass(calibration_contract),
    "signal": signal_contract_pass(signal_contract),
}
V10R_G0_PASS = all(V10R_CONTRACT_STATUS.values())

rows = []
V10R_CACHE_PATHS = {}

for spec in RUN_SPECS:
    run_id = spec["run_id"]
    cache = CACHE_ROOT / f"{run_id}_physics.npz"
    raw_run = (
        RAW_ROOT
        / spec["environment"]
        / "Data_anymal"
        / spec["run"]
    )
    bag = raw_run / "rosbags" / f"{spec['run']}.bag"

    row = {
        **spec,
        "cache_path": str(cache),
        "cache_exists": cache.exists(),
        "raw_run_exists": raw_run.exists(),
        "rosbag_exists": bag.exists(),
        "rgb_exists": (raw_run / "image_lcam_front").exists(),
        "depth_exists": (raw_run / "depth_lcam_front").exists(),
        "pose_exists": (raw_run / "pose_lcam_front").exists(),
        "cache_valid": False,
        "n": 0,
        "median_dt_ms": np.nan,
        "nonpositive_dt": np.nan,
        "force_finite": False,
        "has_foot_world": False,
        "has_desired_wrench": False,
        "has_touchdown_impulse": False,
    }

    if cache.exists():
        try:
            with np.load(cache, allow_pickle=False) as z:
                keys = set(z.files)
                required_keys = {
                    "time", "P", "B", "U", "L_LEG", "E", "contact"
                }
                row["keys"] = ",".join(sorted(keys))
                row["has_foot_world"] = "foot_world" in keys
                row["has_desired_wrench"] = "desired_wrench" in keys
                row["has_touchdown_impulse"] = (
                    "touchdown_impulse" in keys
                )

                if required_keys.issubset(keys):
                    time = np.asarray(z["time"], dtype=np.float64)
                    P = np.asarray(z["P"], dtype=np.float32)
                    B = np.asarray(z["B"], dtype=np.float32)
                    U = np.asarray(z["U"], dtype=np.float32)
                    L = np.asarray(z["L_LEG"], dtype=np.float32)
                    E = np.asarray(z["E"], dtype=np.float32)
                    contact = np.asarray(z["contact"])

                    n = len(time)
                    shapes_ok = (
                        P.shape == (n, 4, 12)
                        and B.shape[0] == n
                        and U.shape[0] == n
                        and L.shape == (n, 4, 3)
                        and E.shape == (n, 4, 3)
                        and contact.shape == (n, 4)
                    )

                    dt = np.diff(time)
                    row["n"] = n
                    row["median_dt_ms"] = (
                        float(np.median(dt) * 1000.0)
                        if len(dt) else np.nan
                    )
                    row["nonpositive_dt"] = int(np.sum(dt <= 0))
                    row["force_finite"] = bool(
                        np.isfinite(P[:, :, 9:12]).all()
                    )
                    row["cache_valid"] = bool(
                        shapes_ok
                        and row["nonpositive_dt"] == 0
                        and row["force_finite"]
                    )

            V10R_CACHE_PATHS[run_id] = cache

        except Exception as exc:
            row["read_error"] = repr(exc)

    rows.append(row)

V10R_DATA_PREFLIGHT = pd.DataFrame(rows)

display(
    V10R_DATA_PREFLIGHT[
        [
            "run_id", "cache_exists", "cache_valid", "n",
            "median_dt_ms", "raw_run_exists", "rosbag_exists",
            "rgb_exists", "depth_exists",
            "has_foot_world", "has_desired_wrench",
            "has_touchdown_impulse",
        ]
    ]
)

print("\nScientific contract status:")
for name, passed in V10R_CONTRACT_STATUS.items():
    print(f"  {name:12s}: {passed}")

print("\nG0 scientific contract pass:", V10R_G0_PASS)
print(
    "Independent trajectories:",
    len(V10R_DATA_PREFLIGHT),
    "/ required for paper:",
    CFG.min_independent_runs,
)

cache_pass = bool(V10R_DATA_PREFLIGHT["cache_valid"].all())
prepared_pass = bool(
    V10R_DATA_PREFLIGHT[
        [
            "has_foot_world",
            "has_desired_wrench",
            "has_touchdown_impulse",
        ]
    ].all().all()
)

print("All physics caches valid:", cache_pass)
print("All model-specific targets prepared:", prepared_pass)

if not V10R_G0_PASS:
    print(
        "\nSTOP: force frames, camera/base calibration, and/or signal "
        "semantics are not yet evidence-verified. Transformer training "
        "would not be physically interpretable."
    )
elif not cache_pass:
    print("\nSTOP: at least one physics cache is missing or invalid.")
elif not prepared_pass:
    print(
        "\nNEXT: caches are valid, but foot_world, desired_wrench, and "
        "touchdown_impulse still need to be generated."
    )
else:
    print("\nREADY: run the exact-state ceiling cell next.")

preflight_path = "/content/v10r_real_data_preflight.csv"
V10R_DATA_PREFLIGHT.to_csv(preflight_path, index=False)
print("\nSaved:", preflight_path)

# %% notebook cell 37
# ============================================================
# V10-R G1 — exact-state LOTO diagnostic ceiling
# This is not transformer training.
# ============================================================
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from IPython.display import display

if "V10R_G0_PASS" not in globals():
    raise RuntimeError("Run the G0 real-data preflight cell first.")

if not V10R_G0_PASS:
    raise RuntimeError(
        "G0 failed. Resolve force-frame, calibration, and signal "
        "contracts before interpreting Fx/Fy/Fz."
    )

if not V10R_DATA_PREFLIGHT["cache_valid"].all():
    raise RuntimeError("At least one required physics cache is invalid.")

def load_numeric_run(spec, path):
    with np.load(path, allow_pickle=False) as z:
        return {
            "spec": spec,
            "time": np.asarray(z["time"], dtype=np.float64),
            "P": np.asarray(z["P"], dtype=np.float32),
            "B": np.asarray(z["B"], dtype=np.float32),
            "U": np.asarray(z["U"], dtype=np.float32),
            "L_LEG": np.asarray(z["L_LEG"], dtype=np.float32),
            "E": np.asarray(z["E"], dtype=np.float32),
            "contact": np.asarray(z["contact"], dtype=bool),
        }

def exact_features(run):
    P = run["P"]
    return np.concatenate(
        [
            P[:, :, 0:9].reshape(len(P), -1),
            run["B"].reshape(len(P), -1),
            run["U"].reshape(len(P), -1),
            run["L_LEG"].reshape(len(P), -1),
            run["E"].reshape(len(P), -1),
            run["contact"].astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)

def force_target(run):
    return run["P"][:, :, 9:12].reshape(len(run["P"]), -1)

def contact_phase(contact):
    contact = np.asarray(contact, dtype=bool)
    previous = np.vstack([contact[:1], contact[:-1]])

    phase = np.zeros(contact.shape, dtype=np.int64)  # swing
    phase[contact] = 2                               # stance
    phase[(~previous) & contact] = 1                 # touchdown
    phase[previous & (~contact)] = 3                 # liftoff
    return phase

def block_bootstrap_p95_upper(
    error,
    phase_mask,
    block_samples=500,
    repetitions=200,
    seed=1234,
):
    blocks = []
    n = len(error)

    for start in range(0, n, block_samples):
        stop = min(start + block_samples, n)
        block_error = error[start:stop]
        block_mask = phase_mask[start:stop]
        selected = block_error[block_mask].reshape(-1)
        if selected.size:
            blocks.append(selected)

    if len(blocks) < 2:
        return np.nan

    rng = np.random.default_rng(seed)
    estimates = []

    for _ in range(repetitions):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        sample = np.concatenate([blocks[i] for i in chosen])
        estimates.append(np.quantile(sample, 0.95))

    return float(np.quantile(estimates, 0.95))

runs = []
for spec in RUN_SPECS:
    path = V10R_CACHE_PATHS.get(spec["run_id"])
    if path is None or not Path(path).exists():
        raise FileNotFoundError(
            f"Missing cache for {spec['run_id']}"
        )
    runs.append(load_numeric_run(spec, path))

metric_rows = []
fold_rows = []

for test_index, test_run in enumerate(runs):
    train_runs = [
        run for i, run in enumerate(runs)
        if i != test_index
    ]

    X_train = np.concatenate(
        [exact_features(run) for run in train_runs],
        axis=0,
    )
    y_train = np.concatenate(
        [force_target(run) for run in train_runs],
        axis=0,
    )

    X_test = exact_features(test_run)
    y_test = force_target(test_run)

    model_ceiling = ExtraTreesRegressor(
        n_estimators=256,
        min_samples_leaf=2,
        max_features=1.0,
        n_jobs=-1,
        random_state=CFG.seed + test_index,
    )
    model_ceiling.fit(X_train, y_train)

    prediction = model_ceiling.predict(X_test).reshape(-1, 4, 3)
    truth = y_test.reshape(-1, 4, 3)
    error = np.abs(prediction - truth)
    phase = contact_phase(test_run["contact"])

    phase_definitions = {
        "all": None,
        "swing": 0,
        "touchdown": 1,
        "stable_stance": 2,
        "liftoff": 3,
    }

    for phase_name, phase_code in phase_definitions.items():
        for leg_index, leg in enumerate(("FL", "FR", "RL", "RR")):
            if phase_code is None:
                sample_mask = np.ones(
                    len(error), dtype=bool
                )
            else:
                sample_mask = phase[:, leg_index] == phase_code

            for axis_index, axis in enumerate(("Fx", "Fy", "Fz")):
                values = error[
                    sample_mask, leg_index, axis_index
                ]

                if len(values) == 0:
                    continue

                metric_rows.append({
                    "test_run": test_run["spec"]["run_id"],
                    "phase": phase_name,
                    "leg": leg,
                    "axis": axis,
                    "n": len(values),
                    "mae_N": float(np.mean(values)),
                    "median_N": float(np.median(values)),
                    "p95_N": float(np.quantile(values, 0.95)),
                    "max_N": float(np.max(values)),
                })

    stance_mask = phase == 2
    touchdown_mask = phase == 1

    stance_values = error[stance_mask].reshape(-1)
    touchdown_values = error[touchdown_mask].reshape(-1)

    fold_rows.append({
        "test_run": test_run["spec"]["run_id"],
        "all_component_mae_N": float(error.mean()),
        "all_component_p95_N": float(np.quantile(error, 0.95)),
        "stance_component_p95_N": (
            float(np.quantile(stance_values, 0.95))
            if stance_values.size else np.nan
        ),
        "touchdown_component_p95_N": (
            float(np.quantile(touchdown_values, 0.95))
            if touchdown_values.size else np.nan
        ),
        "stance_p95_bootstrap_upper_N":
            block_bootstrap_p95_upper(
                error,
                stance_mask,
                block_samples=int(5 * CFG.sample_hz),
                repetitions=200,
                seed=CFG.seed + test_index,
            ),
    })

V10R_CEILING_METRICS = pd.DataFrame(metric_rows)
V10R_CEILING_FOLDS = pd.DataFrame(fold_rows)

print("Exact-state leave-one-terrain-out summary:")
display(V10R_CEILING_FOLDS)

stance_table = V10R_CEILING_METRICS[
    V10R_CEILING_METRICS["phase"] == "stable_stance"
].copy()

print("\nWorst stable-stance leg/axis cases:")
display(
    stance_table.sort_values("p95_N", ascending=False).head(20)
)

ceiling_path = "/content/v10r_exact_state_loto_metrics.csv"
fold_path = "/content/v10r_exact_state_loto_fold_summary.csv"

V10R_CEILING_METRICS.to_csv(ceiling_path, index=False)
V10R_CEILING_FOLDS.to_csv(fold_path, index=False)

print("Saved:", ceiling_path)
print("Saved:", fold_path)

worst_stance_p95 = float(stance_table["p95_N"].max())
worst_stance_upper = float(
    V10R_CEILING_FOLDS["stance_p95_bootstrap_upper_N"].max()
)

print("\nWorst leg/axis stable-stance p95:", worst_stance_p95, "N")
print("Worst fold bootstrap upper bound:", worst_stance_upper, "N")

if worst_stance_p95 > 4.0:
    print(
        "\nG1 RESULT: the exact-state ceiling misses the 4 N target. "
        "Do not expect the causal vision model to solve this merely by "
        "training longer. Investigate synchronization, force labels, "
        "trajectory diversity, and state/force observability."
    )
else:
    print(
        "\nG1 PRELIMINARY PASS: the exact-state state-to-force mapping "
        "reaches the numerical target. This permits proceeding to "
        "causal-history and geometry ablations, but is not itself a "
        "paper result."
    )

print(
    "\nPaper limitation: only four independent trajectories are present; "
    f"the configured paper gate requires {CFG.min_independent_runs}."
)
