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

# %% notebook cell 38
# ============================================================
# CELL D — G1 numerical exact-state LOTO ceiling
# Coordinate channels only: NOT yet physical Fx/Fy/Fz
# ============================================================
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from IPython.display import display

if not globals().get("V10R_CACHE_PASS", False):
    raise RuntimeError(
        "Corrected G0 did not validate all four physics caches."
    )

BASE_SEED = int(globals().get("SEED", 731))

def load_numeric_run(spec):
    path = DISCOVERED_CACHE_PATHS[spec["run_id"]]

    with np.load(path, allow_pickle=False) as archive:
        run = {
            "spec": spec,
            "time": np.asarray(
                archive["time"], dtype=np.float64
            ),
            "P": np.asarray(
                archive["P"], dtype=np.float32
            ),
            "B": np.asarray(
                archive["B"], dtype=np.float32
            ),
            "U": np.asarray(
                archive["U"], dtype=np.float32
            ),
            "L_LEG": np.asarray(
                archive["L_LEG"], dtype=np.float32
            ),
            "E": np.asarray(
                archive["E"], dtype=np.float32
            ),
            "contact": np.asarray(
                archive["contact"], dtype=bool
            ),
        }

    return run

def exact_state_features_numeric(run):
    P = run["P"]

    # Force channels P[..., 9:12] are explicitly excluded.
    X = np.concatenate([
        P[:, :, 0:9].reshape(len(P), -1),
        run["B"].reshape(len(P), -1),
        run["U"].reshape(len(P), -1),
        run["L_LEG"].reshape(len(P), -1),
        run["E"].reshape(len(P), -1),
        run["contact"].astype(np.float32),
    ], axis=1).astype(np.float32)

    y = P[:, :, 9:12].reshape(len(P), -1).astype(
        np.float32
    )

    if not np.isfinite(X).all():
        raise RuntimeError(
            f"{run['spec']['run_id']}: non-finite features"
        )
    if not np.isfinite(y).all():
        raise RuntimeError(
            f"{run['spec']['run_id']}: non-finite targets"
        )

    return X, y

def contact_phase_numeric(contact):
    contact = np.asarray(contact, dtype=bool)
    previous = np.vstack([contact[:1], contact[:-1]])

    phase = np.zeros(contact.shape, dtype=np.int64)
    phase[contact] = 2
    phase[(~previous) & contact] = 1
    phase[previous & (~contact)] = 3

    return phase

def block_bootstrap_p95_upper(
    error,
    phase_mask,
    block_samples=500,
    repetitions=300,
    seed=731,
):
    blocks = []

    for start in range(0, len(error), block_samples):
        stop = min(start + block_samples, len(error))

        selected = error[start:stop][
            phase_mask[start:stop]
        ].reshape(-1)

        if selected.size:
            blocks.append(selected)

    if len(blocks) < 2:
        return np.nan

    rng = np.random.default_rng(seed)
    estimates = []

    for _ in range(repetitions):
        selected_ids = rng.integers(
            0, len(blocks), size=len(blocks)
        )
        sample = np.concatenate([
            blocks[index] for index in selected_ids
        ])
        estimates.append(
            float(np.quantile(sample, 0.95))
        )

    return float(np.quantile(estimates, 0.95))

numeric_runs = [
    load_numeric_run(spec)
    for spec in RUN_SPECS
]

# Verify feature width is identical before fitting.
feature_widths = []

for run in numeric_runs:
    X, y = exact_state_features_numeric(run)
    feature_widths.append(X.shape[1])

    print(
        run["spec"]["run_id"],
        "X:", X.shape,
        "y:", y.shape,
    )

if len(set(feature_widths)) != 1:
    raise RuntimeError(
        f"Inconsistent feature widths: {feature_widths}"
    )

print("Exact-state feature width:", feature_widths[0])

metric_rows = []
fold_rows = []

for test_index, test_run in enumerate(numeric_runs):
    train_runs = [
        run
        for index, run in enumerate(numeric_runs)
        if index != test_index
    ]

    train_pairs = [
        exact_state_features_numeric(run)
        for run in train_runs
    ]
    X_train = np.concatenate(
        [pair[0] for pair in train_pairs],
        axis=0,
    )
    y_train = np.concatenate(
        [pair[1] for pair in train_pairs],
        axis=0,
    )

    X_test, y_test = exact_state_features_numeric(
        test_run
    )

    model_ceiling = ExtraTreesRegressor(
        n_estimators=256,
        min_samples_leaf=3,
        max_features=0.8,
        n_jobs=-1,
        random_state=BASE_SEED + test_index,
    )
    model_ceiling.fit(X_train, y_train)

    prediction = model_ceiling.predict(
        X_test
    ).reshape(-1, 4, 3)
    truth = y_test.reshape(-1, 4, 3)
    absolute_error = np.abs(prediction - truth)

    phase = contact_phase_numeric(
        test_run["contact"]
    )

    phase_codes = {
        "all": None,
        "swing": 0,
        "touchdown": 1,
        "stable_stance": 2,
        "liftoff": 3,
    }

    for phase_name, phase_code in phase_codes.items():
        for leg_index, leg in enumerate(
            ("FL", "FR", "RL", "RR")
        ):
            if phase_code is None:
                mask = np.ones(
                    len(absolute_error), dtype=bool
                )
            else:
                mask = (
                    phase[:, leg_index] == phase_code
                )

            for channel_index in range(3):
                values = absolute_error[
                    mask,
                    leg_index,
                    channel_index,
                ]

                if values.size == 0:
                    continue

                metric_rows.append({
                    "interpretation":
                        "NUMERIC_ONLY_FORCE_FRAME_UNRESOLVED",
                    "test_run":
                        test_run["spec"]["run_id"],
                    "phase": phase_name,
                    "leg": leg,
                    "channel":
                        f"channel_{channel_index}",
                    "n": int(values.size),
                    "mae_N": float(values.mean()),
                    "median_N": float(
                        np.median(values)
                    ),
                    "p95_N": float(
                        np.quantile(values, 0.95)
                    ),
                    "within_4N_fraction": float(
                        np.mean(values <= 4.0)
                    ),
                    "max_N": float(values.max()),
                })

    stance_mask = phase == 2
    touchdown_mask = phase == 1

    stance_error = absolute_error[
        stance_mask
    ].reshape(-1)
    touchdown_error = absolute_error[
        touchdown_mask
    ].reshape(-1)

    fold_rows.append({
        "interpretation":
            "NUMERIC_ONLY_FORCE_FRAME_UNRESOLVED",
        "test_run": test_run["spec"]["run_id"],
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "overall_component_mae_N": float(
            absolute_error.mean()
        ),
        "overall_component_p95_N": float(
            np.quantile(absolute_error, 0.95)
        ),
        "stable_stance_component_p95_N": (
            float(np.quantile(stance_error, 0.95))
            if stance_error.size else np.nan
        ),
        "touchdown_component_p95_N": (
            float(np.quantile(touchdown_error, 0.95))
            if touchdown_error.size else np.nan
        ),
        "stable_stance_p95_bootstrap_upper_N":
            block_bootstrap_p95_upper(
                absolute_error,
                stance_mask,
                block_samples=int(
                    5 * CFG.sample_hz
                ),
                repetitions=300,
                seed=BASE_SEED + test_index,
            ),
    })

V10R_NUMERIC_CEILING_METRICS = pd.DataFrame(
    metric_rows
)
V10R_NUMERIC_CEILING_FOLDS = pd.DataFrame(
    fold_rows
)

print(
    "\nNUMERICAL LOTO CEILING — "
    "FORCE FRAME NOT PHYSICALLY RESOLVED"
)
display(V10R_NUMERIC_CEILING_FOLDS)

stable_rows = V10R_NUMERIC_CEILING_METRICS[
    V10R_NUMERIC_CEILING_METRICS["phase"]
    == "stable_stance"
].sort_values("p95_N", ascending=False)

print("\nWorst stable-stance leg/channel cases:")
display(stable_rows.head(20))

V10R_NUMERIC_CEILING_METRICS.to_csv(
    "/content/v10r_numeric_ceiling_metrics.csv",
    index=False,
)
V10R_NUMERIC_CEILING_FOLDS.to_csv(
    "/content/v10r_numeric_ceiling_folds.csv",
    index=False,
)

worst_channel_p95 = float(
    stable_rows["p95_N"].max()
)
worst_fold_upper = float(
    V10R_NUMERIC_CEILING_FOLDS[
        "stable_stance_p95_bootstrap_upper_N"
    ].max()
)

print("\nWorst stable-stance channel p95:",
      worst_channel_p95, "N")
print("Worst fold bootstrap upper bound:",
      worst_fold_upper, "N")

if worst_channel_p95 > 4.0:
    print(
        "\nNUMERICAL CEILING FAILS 4 N. "
        "The transformer should not be expected to reach 4 N "
        "using these inputs and labels without resolving data quality, "
        "synchronization, or observability."
    )
else:
    print(
        "\nNUMERICAL CEILING REACHES 4 N. "
        "This supports continuing implementation, but remains "
        "a coordinate-unresolved four-run diagnostic."
    )

# %% notebook cell 39
# ============================================================
# CELL E — localize permutation-equivariance error
# No architecture modification
# ============================================================
import numpy as np
import pandas as pd
import torch
from IPython.display import display

DIAGNOSTIC_SEED = int(
    globals().get("SEED", 731)
) + 202

torch.manual_seed(DIAGNOSTIC_SEED)
np.random.seed(DIAGNOSTIC_SEED)

H = len(CFG.horizons_s)
history_steps = int(
    round(CFG.history_s * CFG.sample_hz)
)

equiv_model = V10R(
    d=CFG.d_model,
    horizons=H,
    patch_dim=64,
).to(DEVICE)
equiv_model.eval()

batch = synthetic_batch(
    b=2,
    hist=history_steps,
    H=H,
    patch_dim=64,
)

permutation = torch.tensor(
    [2, 0, 3, 1],
    device=DEVICE,
)

permuted_batch = (
    batch[0][:, permutation],
    batch[1],
    batch[2],
    batch[3][:, permutation],
    batch[4][:, :, permutation],
    batch[5][:, :, permutation],
    batch[6][:, :, permutation],
    batch[7],
)

with torch.no_grad():
    proposal = equiv_model.propose(
        batch[0],
        batch[1],
        batch[2],
        batch[3],
    )
    proposal_permuted = equiv_model.propose(
        permuted_batch[0],
        permuted_batch[1],
        permuted_batch[2],
        permuted_batch[3],
    )

    decoded = equiv_model.decode_force(
        proposal,
        batch[4],
        batch[5],
        batch[6],
        batch[7],
    )
    decoded_permuted = equiv_model.decode_force(
        proposal_permuted,
        permuted_batch[4],
        permuted_batch[5],
        permuted_batch[6],
        permuted_batch[7],
    )

def compute_free_force(model, proposal, patches):
    patch_latent = model.patch(patches)

    body = proposal["latent_body"][
        :, :, None
    ].expand(-1, -1, 4, -1)

    action = proposal["latent_action"][
        :, :, None
    ].expand(-1, -1, 4, -1)

    decoder_input = torch.cat([
        proposal["latent_leg"],
        body,
        action,
        patch_latent,
    ], dim=-1)

    return model.force_free(
        decoder_input
    )[..., :3]

with torch.no_grad():
    free_force = compute_free_force(
        equiv_model,
        proposal,
        batch[4],
    )
    free_force_permuted = compute_free_force(
        equiv_model,
        proposal_permuted,
        permuted_batch[4],
    )

diagnostic_rows = []

def add_leg_comparison(stage, name, value, permuted_value):
    reference = value[:, :, permutation]
    difference = (
        permuted_value - reference
    ).abs()

    scale = float(
        reference.abs().max().item()
    )

    diagnostic_rows.append({
        "stage": stage,
        "output": name,
        "max_abs_error": float(
            difference.max().item()
        ),
        "mean_abs_error": float(
            difference.mean().item()
        ),
        "reference_max_abs": scale,
        "relative_max_error": float(
            difference.max().item()
            / max(scale, 1e-12)
        ),
    })

def add_body_comparison(stage, name, value, permuted_value):
    difference = (
        permuted_value - value
    ).abs()

    scale = float(value.abs().max().item())

    diagnostic_rows.append({
        "stage": stage,
        "output": name,
        "max_abs_error": float(
            difference.max().item()
        ),
        "mean_abs_error": float(
            difference.mean().item()
        ),
        "reference_max_abs": scale,
        "relative_max_error": float(
            difference.max().item()
            / max(scale, 1e-12)
        ),
    })

for name in (
    "mode_logits",
    "event_hazard_logit",
    "contact_prob",
    "foot_mean",
    "foot_chol_raw",
    "leg_sensor_raw",
    "latent_leg",
):
    add_leg_comparison(
        "proposal",
        name,
        proposal[name],
        proposal_permuted[name],
    )

for name in (
    "body_raw",
    "latent_body",
    "latent_action",
):
    add_body_comparison(
        "proposal",
        name,
        proposal[name],
        proposal_permuted[name],
    )

add_leg_comparison(
    "pre_projector",
    "free_force",
    free_force,
    free_force_permuted,
)

for name in (
    "force_mean",
    "force_log_scale",
    "impulse",
):
    add_leg_comparison(
        "decoder",
        name,
        decoded[name],
        decoded_permuted[name],
    )

add_body_comparison(
    "decoder",
    "wrench_residual",
    decoded["wrench_residual"],
    decoded_permuted["wrench_residual"],
)

V10R_EQUIVARIANCE_LOCALIZATION = pd.DataFrame(
    diagnostic_rows
).sort_values(
    "max_abs_error",
    ascending=False,
)

display(V10R_EQUIVARIANCE_LOCALIZATION)

V10R_EQUIVARIANCE_LOCALIZATION.to_csv(
    "/content/v10r_equivariance_localization.csv",
    index=False,
)

proposal_max = float(
    V10R_EQUIVARIANCE_LOCALIZATION.loc[
        V10R_EQUIVARIANCE_LOCALIZATION["stage"]
        == "proposal",
        "max_abs_error",
    ].max()
)

free_force_error = float(
    V10R_EQUIVARIANCE_LOCALIZATION.loc[
        V10R_EQUIVARIANCE_LOCALIZATION["output"]
        == "free_force",
        "max_abs_error",
    ].iloc[0]
)

projected_force_error = float(
    V10R_EQUIVARIANCE_LOCALIZATION.loc[
        V10R_EQUIVARIANCE_LOCALIZATION["output"]
        == "force_mean",
        "max_abs_error",
    ].iloc[0]
)

print("\nMaximum proposal error:", proposal_max)
print("Free-force error:", free_force_error)
print("Projected-force error:", projected_force_error)

if proposal_max < 1e-4 and projected_force_error >= 1e-4:
    print(
        "\nDiagnosis: the shared neural proposal is permutation "
        "equivariant; the discrepancy is introduced by numerical "
        "ordering in the joint contact projector."
    )
elif proposal_max >= 1e-4:
    print(
        "\nDiagnosis: the permutation discrepancy already exists "
        "inside the recurrent relational proposal and is not only "
        "a projector-rounding issue."
    )
else:
    print(
        "\nDiagnosis: both proposal and decoder satisfy the strict "
        "1e-4 test in this isolated run."
    )

# %% notebook cell 40
# ============================================================
# CELL F — exact cache/manifest semantic inventory
# Read-only
# ============================================================
import numpy as np
import pandas as pd
from IPython.display import display

required_preprocessing_fields = (
    "foot_world",
    "com_H",
    "foot_r_H",
    "R_HB",
    "J_leg_H",
    "desired_wrench",
    "desired_wrench_target",
    "touchdown_impulse",
    "force_input_H",
)

inventory_rows = []
required_rows = []

for spec in RUN_SPECS:
    run_id = spec["run_id"]
    physics_path = DISCOVERED_CACHE_PATHS[run_id]

    manifest_name = (
        f"{run_id}_frame_manifest.npz"
    )
    manifest_path = locate_exact_file(
        manifest_name
    )

    with np.load(
        physics_path,
        allow_pickle=False,
    ) as archive:
        keys = set(archive.files)

        for field in required_preprocessing_fields:
            required_rows.append({
                "run_id": run_id,
                "field": field,
                "present": field in keys,
            })

        for key in sorted(keys):
            array = np.asarray(archive[key])

            row = {
                "run_id": run_id,
                "source": "physics_cache",
                "key": key,
                "shape": str(array.shape),
                "dtype": str(array.dtype),
                "finite": "",
                "minimum": np.nan,
                "median": np.nan,
                "maximum": np.nan,
                "text_value": "",
            }

            if np.issubdtype(
                array.dtype,
                np.number,
            ):
                row["finite"] = bool(
                    np.isfinite(array).all()
                )

                if array.size:
                    sample = array.reshape(-1)

                    if sample.size > 200000:
                        indices = np.linspace(
                            0,
                            sample.size - 1,
                            200000,
                        ).astype(int)
                        sample = sample[indices]

                    row["minimum"] = float(
                        np.min(sample)
                    )
                    row["median"] = float(
                        np.median(sample)
                    )
                    row["maximum"] = float(
                        np.max(sample)
                    )
            else:
                flat = array.reshape(-1)
                row["text_value"] = " | ".join(
                    map(str, flat[:10])
                )

            inventory_rows.append(row)

    if manifest_path is not None:
        with np.load(
            manifest_path,
            allow_pickle=False,
        ) as manifest:
            for key in sorted(manifest.files):
                array = np.asarray(manifest[key])

                row = {
                    "run_id": run_id,
                    "source": "frame_manifest",
                    "key": key,
                    "shape": str(array.shape),
                    "dtype": str(array.dtype),
                    "finite": "",
                    "minimum": np.nan,
                    "median": np.nan,
                    "maximum": np.nan,
                    "text_value": "",
                }

                if np.issubdtype(
                    array.dtype,
                    np.number,
                ):
                    row["finite"] = bool(
                        np.isfinite(array).all()
                    )

                    if array.size:
                        sample = array.reshape(-1)

                        if sample.size > 200000:
                            indices = np.linspace(
                                0,
                                sample.size - 1,
                                200000,
                            ).astype(int)
                            sample = sample[indices]

                        row["minimum"] = float(
                            np.min(sample)
                        )
                        row["median"] = float(
                            np.median(sample)
                        )
                        row["maximum"] = float(
                            np.max(sample)
                        )
                else:
                    flat = array.reshape(-1)
                    row["text_value"] = " | ".join(
                        map(str, flat[:10])
                    )

                inventory_rows.append(row)

V10R_CACHE_KEY_INVENTORY = pd.DataFrame(
    inventory_rows
)
V10R_REQUIRED_FIELD_MATRIX = pd.DataFrame(
    required_rows
).pivot(
    index="run_id",
    columns="field",
    values="present",
)

print("Required V10-R preprocessing fields:")
display(V10R_REQUIRED_FIELD_MATRIX)

print("\nAll cache and frame-manifest keys:")
display(V10R_CACHE_KEY_INVENTORY)

V10R_CACHE_KEY_INVENTORY.to_csv(
    "/content/v10r_cache_key_inventory.csv",
    index=False,
)
V10R_REQUIRED_FIELD_MATRIX.to_csv(
    "/content/v10r_required_field_matrix.csv"
)

if not V10R_REQUIRED_FIELD_MATRIX.all().all():
    print(
        "\nConfirmed: the current caches are physics-input caches, "
        "not complete V10-R training caches. Missing fields must be "
        "constructed from verified kinematics, transforms, native "
        "force timing and geometry; no existing ambiguous key will "
        "be silently renamed."
    )

# %% notebook cell 41
# ============================================================
# CELL G — actual causal information-set dataset and baselines
# Uses only information available at or before forecast origin
# Coordinate channels remain physically unresolved.
# ============================================================
import numpy as np
import pandas as pd
from IPython.display import display

if not globals().get("V10R_CACHE_PASS", False):
    raise RuntimeError("Four physics caches have not passed validation.")

HISTORY_STEPS = int(round(CFG.history_s * CFG.sample_hz))
HORIZON_OFFSETS = np.asarray([
    int(round(h * CFG.sample_hz))
    for h in CFG.horizons_s
], dtype=np.int64)

# Force/contact samples included from the past only.
FORCE_LAGS = np.asarray(
    [0, 1, 2, 5, 10, 20, 50, 99],
    dtype=np.int64,
)
SLOPE_LAGS = np.asarray(
    [1, 5, 10],
    dtype=np.int64,
)

def load_causal_run(spec):
    path = DISCOVERED_CACHE_PATHS[spec["run_id"]]

    with np.load(path, allow_pickle=False) as archive:
        return {
            "spec": spec,
            "time": np.asarray(archive["time"], np.float64),
            "P": np.asarray(archive["P"], np.float32),
            "B": np.asarray(archive["B"], np.float32),
            "U": np.asarray(archive["U"], np.float32),
            "L_LEG": np.asarray(
                archive["L_LEG"], np.float32
            ),
            "E": np.asarray(archive["E"], np.float32),
            "contact": np.asarray(
                archive["contact"], bool
            ),
        }

def future_contact_phase(contact, future_indices):
    current = contact[future_indices]
    previous = contact[future_indices - 1]

    phase = np.zeros(current.shape, dtype=np.int64)
    phase[current] = 2
    phase[(~previous) & current] = 1
    phase[previous & (~current)] = 3

    return phase

def build_causal_forecast_data(run):
    P = run["P"]
    force = P[:, :, 9:12]
    contact = run["contact"]
    n = len(P)

    first_origin = max(
        HISTORY_STEPS - 1,
        int(FORCE_LAGS.max()),
        int(SLOPE_LAGS.max()),
    )
    last_offset = int(HORIZON_OFFSETS.max())

    origins = np.arange(
        first_origin,
        n - last_offset,
        dtype=np.int64,
    )
    future_indices = (
        origins[:, None]
        + HORIZON_OFFSETS[None, :]
    )

    # 82 present-state features, still excluding present force.
    present_state = np.concatenate([
        P[origins, :, 0:9].reshape(len(origins), -1),
        run["B"][origins].reshape(len(origins), -1),
        run["U"][origins].reshape(len(origins), -1),
        run["L_LEG"][origins].reshape(len(origins), -1),
        run["E"][origins].reshape(len(origins), -1),
        contact[origins].astype(np.float32),
    ], axis=1)

    force_indices = (
        origins[:, None] - FORCE_LAGS[None, :]
    )
    force_history = force[
        force_indices
    ].reshape(len(origins), -1)

    contact_history = contact[
        force_indices
    ].astype(np.float32).reshape(len(origins), -1)

    force_slopes = []

    for lag in SLOPE_LAGS:
        seconds = lag / float(CFG.sample_hz)
        slope = (
            force[origins] - force[origins - lag]
        ) / seconds
        force_slopes.append(
            slope.reshape(len(origins), -1)
        )

    force_slopes = np.concatenate(
        force_slopes,
        axis=1,
    )

    X = np.concatenate([
        present_state,
        force_history,
        contact_history,
        force_slopes,
    ], axis=1).astype(np.float32)

    target_force = force[
        future_indices
    ].astype(np.float32)

    target_phase = future_contact_phase(
        contact,
        future_indices,
    )

    current_force = force[origins].astype(np.float32)

    recent_indices = (
        origins[:, None]
        - np.arange(5, dtype=np.int64)[None, :]
    )
    recent_force_mean = force[
        recent_indices
    ].mean(axis=1).astype(np.float32)

    if not np.isfinite(X).all():
        raise RuntimeError(
            f"{run['spec']['run_id']}: non-finite causal features"
        )
    if not np.isfinite(target_force).all():
        raise RuntimeError(
            f"{run['spec']['run_id']}: non-finite targets"
        )

    return {
        "run": run,
        "origins": origins,
        "X": X,
        "Y": target_force,
        "phase": target_phase,
        "current_force": current_force,
        "recent_force_mean": recent_force_mean,
    }

def forecast_metric_rows(
    truth,
    prediction,
    phase,
    run_id,
    model_name,
    protocol,
):
    error = np.abs(prediction - truth)
    rows = []

    phase_codes = {
        "all": None,
        "swing": 0,
        "touchdown": 1,
        "stable_stance": 2,
        "liftoff": 3,
    }

    for horizon_index, horizon_s in enumerate(
        CFG.horizons_s
    ):
        for phase_name, phase_code in phase_codes.items():
            for leg_index, leg in enumerate(
                ("FL", "FR", "RL", "RR")
            ):
                if phase_code is None:
                    mask = np.ones(
                        len(error), dtype=bool
                    )
                else:
                    mask = (
                        phase[
                            :, horizon_index, leg_index
                        ] == phase_code
                    )

                for channel_index in range(3):
                    values = error[
                        mask,
                        horizon_index,
                        leg_index,
                        channel_index,
                    ]

                    if values.size == 0:
                        continue

                    rows.append({
                        "interpretation":
                            "NUMERIC_ONLY_FORCE_FRAME_UNRESOLVED",
                        "protocol": protocol,
                        "model": model_name,
                        "run_id": run_id,
                        "horizon_s": float(horizon_s),
                        "phase": phase_name,
                        "leg": leg,
                        "channel":
                            f"channel_{channel_index}",
                        "n": int(values.size),
                        "mae_N": float(values.mean()),
                        "median_N": float(
                            np.median(values)
                        ),
                        "p95_N": float(
                            np.quantile(values, 0.95)
                        ),
                        "within_4N_fraction": float(
                            np.mean(values <= 4.0)
                        ),
                        "max_N": float(values.max()),
                    })

    return rows

CAUSAL_RUNS = [
    load_causal_run(spec)
    for spec in RUN_SPECS
]
CAUSAL_DATA = {
    run["spec"]["run_id"]:
        build_causal_forecast_data(run)
    for run in CAUSAL_RUNS
}

feature_widths = {
    run_id: data["X"].shape
    for run_id, data in CAUSAL_DATA.items()
}
print("Causal feature shapes:")
print(feature_widths)

if len({
    data["X"].shape[1]
    for data in CAUSAL_DATA.values()
}) != 1:
    raise RuntimeError("Causal feature widths differ.")

baseline_rows = []

for run_id, data in CAUSAL_DATA.items():
    truth = data["Y"]
    phase = data["phase"]
    H = truth.shape[1]

    persistence = np.repeat(
        data["current_force"][:, None],
        H,
        axis=1,
    )

    recent_mean = np.repeat(
        data["recent_force_mean"][:, None],
        H,
        axis=1,
    )

    baseline_rows.extend(
        forecast_metric_rows(
            truth,
            persistence,
            phase,
            run_id,
            "force_persistence",
            "no_fit_causal",
        )
    )
    baseline_rows.extend(
        forecast_metric_rows(
            truth,
            recent_mean,
            phase,
            run_id,
            "recent_50ms_mean",
            "no_fit_causal",
        )
    )

V10R_CAUSAL_BASELINE_METRICS = pd.DataFrame(
    baseline_rows
)

baseline_stance_summary = (
    V10R_CAUSAL_BASELINE_METRICS[
        V10R_CAUSAL_BASELINE_METRICS["phase"]
        == "stable_stance"
    ]
    .groupby(
        ["model", "horizon_s"],
        as_index=False,
    )
    .agg(
        worst_p95_N=("p95_N", "max"),
        median_p95_N=("p95_N", "median"),
        mean_mae_N=("mae_N", "mean"),
        minimum_within_4N=(
            "within_4N_fraction", "min"
        ),
    )
)

print("\nCausal no-fit stable-stance baselines:")
display(baseline_stance_summary)

V10R_CAUSAL_BASELINE_METRICS.to_csv(
    "/content/v10r_causal_baseline_metrics.csv",
    index=False,
)
baseline_stance_summary.to_csv(
    "/content/v10r_causal_baseline_summary.csv",
    index=False,
)

# %% notebook cell 42
# ============================================================
# CELL H — learned causal-history LOTO baseline
# Four trajectory folds; no random sample split
# ============================================================
import gc
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from IPython.display import display

if "CAUSAL_DATA" not in globals():
    raise RuntimeError("Run Cell G first.")

HISTORY_MODEL_SEED = int(
    globals().get("SEED", 731)
)

loto_rows = []
loto_fold_rows = []

for test_index, test_spec in enumerate(RUN_SPECS):
    test_id = test_spec["run_id"]
    test_data = CAUSAL_DATA[test_id]

    training_ids = [
        spec["run_id"]
        for spec in RUN_SPECS
        if spec["run_id"] != test_id
    ]

    X_train = np.concatenate([
        CAUSAL_DATA[run_id]["X"]
        for run_id in training_ids
    ], axis=0)

    Y_train = np.concatenate([
        CAUSAL_DATA[run_id]["Y"].reshape(
            len(CAUSAL_DATA[run_id]["Y"]), -1
        )
        for run_id in training_ids
    ], axis=0)

    X_test = test_data["X"]
    truth = test_data["Y"]
    phase = test_data["phase"]

    print(
        f"\nLOTO fold {test_index + 1}/4: "
        f"test={test_id}, "
        f"train={training_ids}, "
        f"X_train={X_train.shape}"
    )

    started = time.time()

    history_model = ExtraTreesRegressor(
        n_estimators=128,
        min_samples_leaf=5,
        max_features=0.5,
        n_jobs=-1,
        random_state=HISTORY_MODEL_SEED + test_index,
    )
    history_model.fit(X_train, Y_train)

    prediction = history_model.predict(
        X_test
    ).reshape(truth.shape)

    elapsed = time.time() - started

    loto_rows.extend(
        forecast_metric_rows(
            truth,
            prediction,
            phase,
            test_id,
            "causal_history_extratrees",
            "trajectory_LOTO",
        )
    )

    stance_mask = phase == 2
    touchdown_mask = phase == 1
    error = np.abs(prediction - truth)

    loto_fold_rows.append({
        "test_run": test_id,
        "train_runs": ",".join(training_ids),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "runtime_s": elapsed,
        "overall_mae_N": float(error.mean()),
        "overall_p95_N": float(
            np.quantile(error, 0.95)
        ),
        "stable_stance_p95_N": float(
            np.quantile(
                error[stance_mask],
                0.95,
            )
        ),
        "touchdown_p95_N": float(
            np.quantile(
                error[touchdown_mask],
                0.95,
            )
        ),
    })

    del history_model, X_train, Y_train
    gc.collect()

V10R_HISTORY_LOTO_METRICS = pd.DataFrame(
    loto_rows
)
V10R_HISTORY_LOTO_FOLDS = pd.DataFrame(
    loto_fold_rows
)

display(V10R_HISTORY_LOTO_FOLDS)

loto_stance_summary = (
    V10R_HISTORY_LOTO_METRICS[
        V10R_HISTORY_LOTO_METRICS["phase"]
        == "stable_stance"
    ]
    .groupby("horizon_s", as_index=False)
    .agg(
        worst_p95_N=("p95_N", "max"),
        median_p95_N=("p95_N", "median"),
        mean_mae_N=("mae_N", "mean"),
        minimum_within_4N=(
            "within_4N_fraction", "min"
        ),
    )
)

print("\nCausal-history LOTO stable-stance summary:")
display(loto_stance_summary)

V10R_HISTORY_LOTO_METRICS.to_csv(
    "/content/v10r_history_loto_metrics.csv",
    index=False,
)
V10R_HISTORY_LOTO_FOLDS.to_csv(
    "/content/v10r_history_loto_folds.csv",
    index=False,
)
loto_stance_summary.to_csv(
    "/content/v10r_history_loto_summary.csv",
    index=False,
)

# %% notebook cell 43
# ============================================================
# CELL I — chronological within-trajectory diagnostic
# Train first 60%, purge middle 20%, test final 20%
# ============================================================
import gc
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from IPython.display import display

if "CAUSAL_DATA" not in globals():
    raise RuntimeError("Run Cell G first.")

within_rows = []
within_fold_rows = []

for run_index, spec in enumerate(RUN_SPECS):
    run_id = spec["run_id"]
    data = CAUSAL_DATA[run_id]

    n = len(data["X"])
    train_stop = int(0.60 * n)
    test_start = int(0.80 * n)

    if test_start - train_stop <= int(
        HORIZON_OFFSETS.max()
    ):
        raise RuntimeError(
            f"{run_id}: purge interval is too short."
        )

    X_train = data["X"][:train_stop]
    Y_train = data["Y"][:train_stop].reshape(
        train_stop, -1
    )

    X_test = data["X"][test_start:]
    truth = data["Y"][test_start:]
    phase = data["phase"][test_start:]

    print(
        f"\nWithin-run {run_index + 1}/4: "
        f"{run_id}, train={len(X_train)}, "
        f"purged={test_start - train_stop}, "
        f"test={len(X_test)}"
    )

    started = time.time()

    within_model = ExtraTreesRegressor(
        n_estimators=128,
        min_samples_leaf=5,
        max_features=0.5,
        n_jobs=-1,
        random_state=HISTORY_MODEL_SEED + 100 + run_index,
    )
    within_model.fit(X_train, Y_train)

    prediction = within_model.predict(
        X_test
    ).reshape(truth.shape)

    elapsed = time.time() - started
    error = np.abs(prediction - truth)

    within_rows.extend(
        forecast_metric_rows(
            truth,
            prediction,
            phase,
            run_id,
            "causal_history_extratrees",
            "chronological_within_run",
        )
    )

    stance_mask = phase == 2
    touchdown_mask = phase == 1

    within_fold_rows.append({
        "run_id": run_id,
        "train_samples": len(X_train),
        "purged_origins": test_start - train_stop,
        "test_samples": len(X_test),
        "runtime_s": elapsed,
        "overall_mae_N": float(error.mean()),
        "overall_p95_N": float(
            np.quantile(error, 0.95)
        ),
        "stable_stance_p95_N": float(
            np.quantile(
                error[stance_mask],
                0.95,
            )
        ),
        "touchdown_p95_N": float(
            np.quantile(
                error[touchdown_mask],
                0.95,
            )
        ),
    })

    del within_model, X_train, Y_train
    gc.collect()

V10R_HISTORY_WITHIN_METRICS = pd.DataFrame(
    within_rows
)
V10R_HISTORY_WITHIN_FOLDS = pd.DataFrame(
    within_fold_rows
)

display(V10R_HISTORY_WITHIN_FOLDS)

within_stance_summary = (
    V10R_HISTORY_WITHIN_METRICS[
        V10R_HISTORY_WITHIN_METRICS["phase"]
        == "stable_stance"
    ]
    .groupby("horizon_s", as_index=False)
    .agg(
        worst_p95_N=("p95_N", "max"),
        median_p95_N=("p95_N", "median"),
        mean_mae_N=("mae_N", "mean"),
        minimum_within_4N=(
            "within_4N_fraction", "min"
        ),
    )
)

print(
    "\nChronological within-run stable-stance summary:"
)
display(within_stance_summary)

V10R_HISTORY_WITHIN_METRICS.to_csv(
    "/content/v10r_history_within_metrics.csv",
    index=False,
)
V10R_HISTORY_WITHIN_FOLDS.to_csv(
    "/content/v10r_history_within_folds.csv",
    index=False,
)
within_stance_summary.to_csv(
    "/content/v10r_history_within_summary.csv",
    index=False,
)

# %% notebook cell 44
# ============================================================
# CELL J — horizon curves and evidence-based decision
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

required = [
    "V10R_CAUSAL_BASELINE_METRICS",
    "V10R_HISTORY_LOTO_METRICS",
    "V10R_HISTORY_WITHIN_METRICS",
]
missing = [
    name for name in required
    if name not in globals()
]
if missing:
    raise RuntimeError(
        "Run Cells G, H and I first. Missing: "
        + ", ".join(missing)
    )

all_forecast_metrics = pd.concat([
    V10R_CAUSAL_BASELINE_METRICS,
    V10R_HISTORY_LOTO_METRICS,
    V10R_HISTORY_WITHIN_METRICS,
], ignore_index=True)

stable = all_forecast_metrics[
    all_forecast_metrics["phase"]
    == "stable_stance"
].copy()

V10R_FORECAST_COMPARISON = (
    stable
    .groupby(
        ["protocol", "model", "horizon_s"],
        as_index=False,
    )
    .agg(
        worst_p95_N=("p95_N", "max"),
        median_p95_N=("p95_N", "median"),
        mean_mae_N=("mae_N", "mean"),
        minimum_within_4N=(
            "within_4N_fraction", "min"
        ),
    )
)

display(V10R_FORECAST_COMPARISON)

fig, axes = plt.subplots(
    1, 2,
    figsize=(13, 4.8),
)

for (protocol, model_name), group in (
    V10R_FORECAST_COMPARISON.groupby(
        ["protocol", "model"]
    )
):
    group = group.sort_values("horizon_s")
    label = f"{protocol}: {model_name}"

    axes[0].plot(
        group["horizon_s"],
        group["worst_p95_N"],
        marker="o",
        label=label,
    )
    axes[1].plot(
        group["horizon_s"],
        group["median_p95_N"],
        marker="o",
        label=label,
    )

for axis in axes:
    axis.axhline(
        4.0,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="4 N target",
    )
    axis.set_xlabel("Forecast horizon [s]")
    axis.set_ylabel("Stable-stance component p95 [N]")
    axis.grid(alpha=0.25)

axes[0].set_title("Worst run/leg/channel")
axes[1].set_title("Median run/leg/channel")

handles, labels = axes[0].get_legend_handles_labels()
unique = dict(zip(labels, handles))
fig.legend(
    unique.values(),
    unique.keys(),
    loc="upper center",
    ncol=2,
    bbox_to_anchor=(0.5, 1.08),
)

fig.tight_layout()
figure_path = (
    "/content/v10r_causal_forecast_comparison.png"
)
fig.savefig(
    figure_path,
    dpi=180,
    bbox_inches="tight",
)
plt.show()

V10R_FORECAST_COMPARISON.to_csv(
    "/content/v10r_causal_forecast_comparison.csv",
    index=False,
)

loto = V10R_FORECAST_COMPARISON[
    V10R_FORECAST_COMPARISON["protocol"]
    == "trajectory_LOTO"
]
within = V10R_FORECAST_COMPARISON[
    V10R_FORECAST_COMPARISON["protocol"]
    == "chronological_within_run"
]

loto_worst_all = float(
    loto["worst_p95_N"].max()
)
within_worst_all = float(
    within["worst_p95_N"].max()
)

loto_01 = float(
    loto.loc[
        np.isclose(loto["horizon_s"], 0.1),
        "worst_p95_N",
    ].max()
)
within_01 = float(
    within.loc[
        np.isclose(within["horizon_s"], 0.1),
        "worst_p95_N",
    ].max()
)

print("\nDECISION STATISTICS")
print("LOTO worst p95 across all horizons:", loto_worst_all)
print("Within-run worst p95 across all horizons:", within_worst_all)
print("LOTO worst p95 at 0.1 s:", loto_01)
print("Within-run worst p95 at 0.1 s:", within_01)

if within_worst_all <= 4.0 and loto_worst_all > 4.0:
    print(
        "\nDIAGNOSIS: within-run predictability is sufficient, "
        "but cross-terrain generalization fails. The next work is "
        "terrain geometry/haptic conditioning and more independent "
        "training trajectories."
    )
elif within_worst_all > 4.0:
    print(
        "\nDIAGNOSIS: even same-trajectory chronological prediction "
        "does not meet 4 N. Before training V10-R, investigate label "
        "timing, force bandwidth, missing controller actions and "
        "intrinsic future uncertainty."
    )
else:
    print(
        "\nDIAGNOSIS: both within-run and LOTO causal-history "
        "baselines reach the numerical target. Proceed to repair "
        "the V10-R loss heads and construct verified geometry/"
        "mechanics targets."
    )

print("\nSaved figure:", figure_path)

# %% notebook cell 45
# ============================================================
# CELL K — guarded phase labels and all available targets
# ============================================================
import numpy as np
import pandas as pd
from IPython.display import display

if "CAUSAL_DATA" not in globals():
    raise RuntimeError("Run Cell G first.")

GUARD_SAMPLES = int(round(0.05 * CFG.sample_hz))  # ±50 ms

def guarded_phase_full(contact, guard=GUARD_SAMPLES):
    """
    Codes:
      -1 = ambiguous/transition-adjacent
       0 = guarded swing
       1 = touchdown window
       2 = guarded stable stance
       3 = liftoff window
    """
    contact = np.asarray(contact, dtype=bool)
    n, legs = contact.shape
    phase = np.full((n, legs), -1, dtype=np.int64)

    kernel = np.ones(2 * guard + 1, dtype=np.int32)
    full_window = len(kernel)

    for leg in range(legs):
        c = contact[:, leg]

        touchdown = np.zeros(n, dtype=np.int32)
        liftoff = np.zeros(n, dtype=np.int32)

        touchdown[1:] = ((~c[:-1]) & c[1:]).astype(np.int32)
        liftoff[1:] = (c[:-1] & (~c[1:])).astype(np.int32)

        contact_count = np.convolve(
            c.astype(np.int32), kernel, mode="same"
        )
        valid_count = np.convolve(
            np.ones(n, dtype=np.int32), kernel, mode="same"
        )

        valid_window = valid_count == full_window
        stable = valid_window & (contact_count == full_window)
        swing = valid_window & (contact_count == 0)

        touchdown_near = (
            np.convolve(touchdown, kernel, mode="same") > 0
        )
        liftoff_near = (
            np.convolve(liftoff, kernel, mode="same") > 0
        )

        phase[swing, leg] = 0
        phase[stable, leg] = 2
        phase[touchdown_near & (~liftoff_near), leg] = 1
        phase[liftoff_near & (~touchdown_near), leg] = 3

    return phase

# Continuous flattened-target layout.
# 10 horizons:
# force     = 10 × 4 × 3 = 120
# leg state = 10 × 4 × 9 = 360
# body      = 10 × 12    = 120
# E         = 10 × 4 × 3 = 120
TARGET_SLICES = {
    "force": slice(0, 120),
    "leg_state": slice(120, 480),
    "body": slice(480, 600),
    "E": slice(600, 720),
}
CONTINUOUS_TARGET_WIDTH = 720
CONTACT_TARGET_WIDTH = 40

for run_id, data in CAUSAL_DATA.items():
    run = data["run"]
    origins = data["origins"]

    future_indices = (
        origins[:, None]
        + HORIZON_OFFSETS[None, :]
    )

    force_target = run["P"][
        future_indices, :, 9:12
    ].astype(np.float32)

    leg_state_target = run["P"][
        future_indices, :, 0:9
    ].astype(np.float32)

    body_target = run["B"][
        future_indices
    ].astype(np.float32)

    E_target = run["E"][
        future_indices
    ].astype(np.float32)

    contact_target = run["contact"][
        future_indices
    ].astype(np.float32)

    complete_phase = guarded_phase_full(
        run["contact"]
    )
    guarded_phase = complete_phase[
        future_indices
    ]

    continuous_target = np.concatenate([
        force_target.reshape(len(origins), -1),
        leg_state_target.reshape(len(origins), -1),
        body_target.reshape(len(origins), -1),
        E_target.reshape(len(origins), -1),
    ], axis=1).astype(np.float32)

    if continuous_target.shape[1] != CONTINUOUS_TARGET_WIDTH:
        raise RuntimeError(continuous_target.shape)

    data["Y_continuous"] = continuous_target
    data["Y_contact"] = contact_target.reshape(
        len(origins), -1
    )
    data["phase_guarded"] = guarded_phase
    data["future_indices"] = future_indices

target_table = pd.DataFrame([
    {
        "run_id": run_id,
        "samples": len(data["X"]),
        "input_width": data["X"].shape[1],
        "continuous_target_width":
            data["Y_continuous"].shape[1],
        "contact_target_width":
            data["Y_contact"].shape[1],
        "guarded_stance_targets": int(
            np.sum(data["phase_guarded"] == 2)
        ),
        "touchdown_window_targets": int(
            np.sum(data["phase_guarded"] == 1)
        ),
        "ambiguous_targets": int(
            np.sum(data["phase_guarded"] == -1)
        ),
    }
    for run_id, data in CAUSAL_DATA.items()
])

display(target_table)

# %% notebook cell 46
# ============================================================
# CELL L — multi-output probabilistic diagnostic model
# This does not replace the V10-R architecture.
# ============================================================
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset

MULTITASK_ROOT = Path(
    "/content/v10r_multitask_diagnostic"
)
MULTITASK_ROOT.mkdir(parents=True, exist_ok=True)

MULTITASK_SEED = int(globals().get("SEED", 731))
MULTITASK_BATCH_SIZE = 512
MULTITASK_EPOCHS = 30
MULTITASK_PATIENCE = 6

class MultiTaskCausalDiagnostic(nn.Module):
    def __init__(
        self,
        input_width=246,
        continuous_width=720,
        contact_width=40,
        hidden=512,
    ):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(input_width, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(0.10),

            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(0.10),

            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )

        self.continuous_mean = nn.Linear(
            hidden, continuous_width
        )
        self.continuous_log_scale = nn.Linear(
            hidden, continuous_width
        )
        self.contact_logits = nn.Linear(
            hidden, contact_width
        )

    def forward(self, x):
        latent = self.trunk(x)

        return {
            "continuous_mean_z":
                self.continuous_mean(latent),
            "continuous_log_scale_z":
                self.continuous_log_scale(latent).clamp(-6, 3),
            "contact_logits":
                self.contact_logits(latent),
        }

def concatenate_runs(run_ids, field):
    return np.concatenate([
        CAUSAL_DATA[run_id][field]
        for run_id in run_ids
    ], axis=0)

def fit_normalizers(train_ids):
    X = concatenate_runs(train_ids, "X").astype(np.float32)
    Y = concatenate_runs(
        train_ids, "Y_continuous"
    ).astype(np.float32)

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std = np.maximum(X_std, 1e-5)

    Y_mean = Y.mean(axis=0)
    Y_std = Y.std(axis=0)
    Y_std = np.maximum(Y_std, 1e-5)

    return {
        "X_mean": X_mean.astype(np.float32),
        "X_std": X_std.astype(np.float32),
        "Y_mean": Y_mean.astype(np.float32),
        "Y_std": Y_std.astype(np.float32),
    }

def build_loader(
    run_ids,
    normalizer,
    shuffle,
    batch_size=MULTITASK_BATCH_SIZE,
):
    X = concatenate_runs(run_ids, "X")
    Y = concatenate_runs(run_ids, "Y_continuous")
    C = concatenate_runs(run_ids, "Y_contact")

    X_z = (
        (X - normalizer["X_mean"])
        / normalizer["X_std"]
    ).astype(np.float32)

    Y_z = (
        (Y - normalizer["Y_mean"])
        / normalizer["Y_std"]
    ).astype(np.float32)

    dataset = TensorDataset(
        torch.from_numpy(X_z),
        torch.from_numpy(Y_z),
        torch.from_numpy(C.astype(np.float32)),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=False,
    )

def multitask_loss(output, continuous_target_z, contact_target):
    mean = output["continuous_mean_z"]
    log_scale = output["continuous_log_scale_z"]
    inverse_variance = torch.exp(-2.0 * log_scale)

    element_nll = (
        0.5
        * (continuous_target_z - mean).square()
        * inverse_variance
        + log_scale
    )

    group_losses = {
        name: element_nll[:, target_slice].mean()
        for name, target_slice in TARGET_SLICES.items()
    }

    contact_loss = F.binary_cross_entropy_with_logits(
        output["contact_logits"],
        contact_target,
    )

    # Equal group weights avoid the 360-dimensional leg head
    # overwhelming the smaller heads.
    total = (
        group_losses["force"]
        + group_losses["leg_state"]
        + group_losses["body"]
        + 0.5 * group_losses["E"]
        + contact_loss
    )

    return {
        "total": total,
        "force_nll": group_losses["force"],
        "leg_state_nll": group_losses["leg_state"],
        "body_nll": group_losses["body"],
        "E_nll": group_losses["E"],
        "contact_bce": contact_loss,
    }

@torch.no_grad()
def validation_loss(model, loader):
    model.eval()
    total = 0.0
    samples = 0

    for X, Y, C in loader:
        X = X.to(DEVICE, non_blocking=True)
        Y = Y.to(DEVICE, non_blocking=True)
        C = C.to(DEVICE, non_blocking=True)

        output = model(X)
        loss = multitask_loss(output, Y, C)["total"]

        total += float(loss) * len(X)
        samples += len(X)

    return total / max(samples, 1)

def train_multitask_fold(
    train_ids,
    validation_id,
    test_id,
    fold_seed,
):
    random.seed(fold_seed)
    np.random.seed(fold_seed)
    torch.manual_seed(fold_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(fold_seed)

    normalizer = fit_normalizers(train_ids)

    train_loader = build_loader(
        train_ids,
        normalizer,
        shuffle=True,
    )
    validation_loader = build_loader(
        [validation_id],
        normalizer,
        shuffle=False,
    )

    model = MultiTaskCausalDiagnostic(
        input_width=next(iter(CAUSAL_DATA.values()))[
            "X"
        ].shape[1],
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    best_validation = np.inf
    best_state = None
    remaining_patience = MULTITASK_PATIENCE
    history = []

    for epoch in range(1, MULTITASK_EPOCHS + 1):
        model.train()
        train_total = 0.0
        train_samples = 0

        for X, Y, C in train_loader:
            X = X.to(DEVICE, non_blocking=True)
            Y = Y.to(DEVICE, non_blocking=True)
            C = C.to(DEVICE, non_blocking=True)

            output = model(X)
            losses = multitask_loss(output, Y, C)

            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )
            optimizer.step()

            train_total += float(
                losses["total"].detach()
            ) * len(X)
            train_samples += len(X)

        train_mean = train_total / max(train_samples, 1)
        validation_mean = validation_loss(
            model, validation_loader
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_mean,
            "validation_loss": validation_mean,
        })

        print(
            f"epoch {epoch:02d} | "
            f"train={train_mean:.5f} | "
            f"validation={validation_mean:.5f}"
        )

        if validation_mean < best_validation - 1e-5:
            best_validation = validation_mean
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            remaining_patience = MULTITASK_PATIENCE
        else:
            remaining_patience -= 1

        if remaining_patience == 0:
            print("Early stopping.")
            break

    if best_state is None:
        raise RuntimeError("No valid checkpoint was produced.")

    model.load_state_dict(best_state)

    fold_name = f"test_{test_id}"
    fold_root = MULTITASK_ROOT / fold_name
    fold_root.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state": best_state,
        "train_ids": list(train_ids),
        "validation_id": validation_id,
        "test_id": test_id,
        "seed": fold_seed,
        "best_validation_loss": best_validation,
    }, fold_root / "best_model.pt")

    np.savez_compressed(
        fold_root / "normalizer.npz",
        **normalizer,
    )
    pd.DataFrame(history).to_csv(
        fold_root / "training_history.csv",
        index=False,
    )

    return model, normalizer, pd.DataFrame(history)

@torch.no_grad()
def predict_multitask(model, normalizer, run_id):
    model.eval()

    X = CAUSAL_DATA[run_id]["X"]
    X_z = (
        (X - normalizer["X_mean"])
        / normalizer["X_std"]
    ).astype(np.float32)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_z)),
        batch_size=1024,
        shuffle=False,
        num_workers=0,
    )

    means = []
    scales = []
    contacts = []

    for (batch,) in loader:
        batch = batch.to(DEVICE)
        output = model(batch)

        mean_z = output[
            "continuous_mean_z"
        ].cpu().numpy()

        scale_z = torch.exp(
            output["continuous_log_scale_z"]
        ).cpu().numpy()

        contact_probability = torch.sigmoid(
            output["contact_logits"]
        ).cpu().numpy()

        means.append(mean_z)
        scales.append(scale_z)
        contacts.append(contact_probability)

    mean_z = np.concatenate(means)
    scale_z = np.concatenate(scales)
    contact_probability = np.concatenate(contacts)

    mean = (
        mean_z * normalizer["Y_std"]
        + normalizer["Y_mean"]
    )
    scale = (
        scale_z * normalizer["Y_std"]
    )

    return {
        "continuous_mean": mean.astype(np.float32),
        "continuous_scale": scale.astype(np.float32),
        "contact_probability":
            contact_probability.astype(np.float32),
    }

# %% notebook cell 47
# ============================================================
# CELL M — train/test every available output with LOTO folds
# ============================================================
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

continuous_rows = []
contact_rows = []
uncertainty_rows = []
fold_rows = []

def error_statistics(values):
    values = np.asarray(values).reshape(-1)

    return {
        "mae": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
        "within_4": float(np.mean(values <= 4.0)),
    }

def add_continuous_metrics(
    rows,
    run_id,
    truth,
    prediction,
    phase,
):
    force_truth = truth[
        :, TARGET_SLICES["force"]
    ].reshape(-1, 10, 4, 3)
    force_prediction = prediction[
        :, TARGET_SLICES["force"]
    ].reshape(-1, 10, 4, 3)

    leg_truth = truth[
        :, TARGET_SLICES["leg_state"]
    ].reshape(-1, 10, 4, 9)
    leg_prediction = prediction[
        :, TARGET_SLICES["leg_state"]
    ].reshape(-1, 10, 4, 9)

    body_truth = truth[
        :, TARGET_SLICES["body"]
    ].reshape(-1, 10, 12)
    body_prediction = prediction[
        :, TARGET_SLICES["body"]
    ].reshape(-1, 10, 12)

    E_truth = truth[
        :, TARGET_SLICES["E"]
    ].reshape(-1, 10, 4, 3)
    E_prediction = prediction[
        :, TARGET_SLICES["E"]
    ].reshape(-1, 10, 4, 3)

    for h, horizon_s in enumerate(CFG.horizons_s):
        # Force: detailed leg/channel guarded-stable evaluation.
        for leg in range(4):
            stable_mask = phase[:, h, leg] == 2

            for channel in range(3):
                if not stable_mask.any():
                    continue

                values = np.abs(
                    force_prediction[
                        stable_mask, h, leg, channel
                    ]
                    - force_truth[
                        stable_mask, h, leg, channel
                    ]
                )
                stats = error_statistics(values)

                rows.append({
                    "run_id": run_id,
                    "group": "force_guarded_stance",
                    "variable":
                        f"{('FL','FR','RL','RR')[leg]}_channel_{channel}",
                    "horizon_s": float(horizon_s),
                    "unit": "N_unresolved_axis",
                    **stats,
                })

        # q, dq and tau.
        variable_groups = {
            "joint_angle": (slice(0, 3), "rad"),
            "joint_velocity": (slice(3, 6), "rad_per_s"),
            "joint_torque": (slice(6, 9), "N_m"),
        }

        for group_name, (variable_slice, unit) in (
            variable_groups.items()
        ):
            values = np.abs(
                leg_prediction[:, h, :, variable_slice]
                - leg_truth[:, h, :, variable_slice]
            )
            stats = error_statistics(values)

            rows.append({
                "run_id": run_id,
                "group": group_name,
                "variable": "all_legs_components",
                "horizon_s": float(horizon_s),
                "unit": unit,
                **stats,
            })

        # Each body channel separately because semantics remain unresolved.
        for channel in range(12):
            values = np.abs(
                body_prediction[:, h, channel]
                - body_truth[:, h, channel]
            )
            stats = error_statistics(values)

            rows.append({
                "run_id": run_id,
                "group": "body_sensor",
                "variable": f"B{channel}",
                "horizon_s": float(horizon_s),
                "unit": "native_cache_unit",
                **stats,
            })

        # Tracking/error channels E.
        for channel in range(3):
            values = np.abs(
                E_prediction[:, h, :, channel]
                - E_truth[:, h, :, channel]
            )
            stats = error_statistics(values)

            rows.append({
                "run_id": run_id,
                "group": "E",
                "variable": f"E_channel_{channel}",
                "horizon_s": float(horizon_s),
                "unit": "native_cache_unit",
                **stats,
            })

def contact_metrics(
    run_id,
    truth,
    probability,
):
    truth = truth.reshape(-1, 10, 4)
    probability = probability.reshape(-1, 10, 4)
    rows = []
    epsilon = 1e-7

    for h, horizon_s in enumerate(CFG.horizons_s):
        y = truth[:, h].reshape(-1)
        p = np.clip(
            probability[:, h].reshape(-1),
            epsilon,
            1.0 - epsilon,
        )
        prediction = p >= 0.5

        true_positive = np.sum(prediction & (y == 1))
        false_positive = np.sum(prediction & (y == 0))
        false_negative = np.sum((~prediction) & (y == 1))

        rows.append({
            "run_id": run_id,
            "horizon_s": float(horizon_s),
            "brier_score": float(np.mean((p - y) ** 2)),
            "binary_cross_entropy": float(
                -np.mean(
                    y * np.log(p)
                    + (1 - y) * np.log(1 - p)
                )
            ),
            "accuracy": float(np.mean(prediction == y)),
            "precision": float(
                true_positive
                / max(true_positive + false_positive, 1)
            ),
            "recall": float(
                true_positive
                / max(true_positive + false_negative, 1)
            ),
            "mean_predicted_probability": float(p.mean()),
            "contact_fraction": float(y.mean()),
        })

    return rows

def calibration_factors(
    truth,
    mean,
    scale,
    coverage_levels=(0.80, 0.90, 0.95),
):
    factors = {}
    scale = np.maximum(scale, 1e-6)

    for group, target_slice in TARGET_SLICES.items():
        ratio = (
            np.abs(
                truth[:, target_slice]
                - mean[:, target_slice]
            )
            / scale[:, target_slice]
        ).reshape(-1)

        for coverage in coverage_levels:
            factors[(group, coverage)] = float(
                np.quantile(ratio, coverage)
            )

    return factors

run_ids = [spec["run_id"] for spec in RUN_SPECS]

for fold, test_id in enumerate(run_ids):
    validation_id = run_ids[(fold + 1) % len(run_ids)]
    train_ids = [
        run_id
        for run_id in run_ids
        if run_id not in {test_id, validation_id}
    ]

    print("\n" + "=" * 80)
    print("Fold:", fold + 1)
    print("Train:", train_ids)
    print("Validation:", validation_id)
    print("Test:", test_id)

    model, normalizer, history = train_multitask_fold(
        train_ids=train_ids,
        validation_id=validation_id,
        test_id=test_id,
        fold_seed=MULTITASK_SEED + fold,
    )

    validation_prediction = predict_multitask(
        model,
        normalizer,
        validation_id,
    )
    test_prediction = predict_multitask(
        model,
        normalizer,
        test_id,
    )

    validation_truth = CAUSAL_DATA[
        validation_id
    ]["Y_continuous"]
    test_truth = CAUSAL_DATA[
        test_id
    ]["Y_continuous"]

    factors = calibration_factors(
        validation_truth,
        validation_prediction["continuous_mean"],
        validation_prediction["continuous_scale"],
    )

    for group, target_slice in TARGET_SLICES.items():
        absolute_error = np.abs(
            test_truth[:, target_slice]
            - test_prediction["continuous_mean"][
                :, target_slice
            ]
        )
        predicted_scale = np.maximum(
            test_prediction["continuous_scale"][
                :, target_slice
            ],
            1e-6,
        )

        for coverage in (0.80, 0.90, 0.95):
            factor = factors[(group, coverage)]
            empirical = float(
                np.mean(
                    absolute_error
                    <= factor * predicted_scale
                )
            )

            uncertainty_rows.append({
                "test_run": test_id,
                "group": group,
                "nominal_coverage": coverage,
                "empirical_coverage": empirical,
                "coverage_error": empirical - coverage,
                "calibration_factor": factor,
            })

    add_continuous_metrics(
        continuous_rows,
        test_id,
        test_truth,
        test_prediction["continuous_mean"],
        CAUSAL_DATA[test_id]["phase_guarded"],
    )

    contact_rows.extend(
        contact_metrics(
            test_id,
            CAUSAL_DATA[test_id]["Y_contact"],
            test_prediction["contact_probability"],
        )
    )

    force_error = np.abs(
        test_truth[:, TARGET_SLICES["force"]]
        - test_prediction["continuous_mean"][
            :, TARGET_SLICES["force"]
        ]
    )

    fold_rows.append({
        "test_run": test_id,
        "validation_run": validation_id,
        "train_runs": ",".join(train_ids),
        "best_validation_loss": float(
            history["validation_loss"].min()
        ),
        "force_overall_mae_N": float(force_error.mean()),
        "force_overall_p95_N": float(
            np.quantile(force_error, 0.95)
        ),
    })

    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

V10R_MULTITASK_CONTINUOUS_METRICS = pd.DataFrame(
    continuous_rows
)
V10R_MULTITASK_CONTACT_METRICS = pd.DataFrame(
    contact_rows
)
V10R_MULTITASK_UNCERTAINTY = pd.DataFrame(
    uncertainty_rows
)
V10R_MULTITASK_FOLDS = pd.DataFrame(
    fold_rows
)

print("\nFold summary:")
display(V10R_MULTITASK_FOLDS)

print("\nContact probability metrics:")
display(V10R_MULTITASK_CONTACT_METRICS)

print("\nCalibrated uncertainty coverage:")
display(V10R_MULTITASK_UNCERTAINTY)

print("\nWorst guarded stable-stance forces:")
display(
    V10R_MULTITASK_CONTINUOUS_METRICS[
        V10R_MULTITASK_CONTINUOUS_METRICS["group"]
        == "force_guarded_stance"
    ]
    .sort_values("p95", ascending=False)
    .head(30)
)

V10R_MULTITASK_CONTINUOUS_METRICS.to_csv(
    MULTITASK_ROOT / "continuous_metrics.csv",
    index=False,
)
V10R_MULTITASK_CONTACT_METRICS.to_csv(
    MULTITASK_ROOT / "contact_metrics.csv",
    index=False,
)
V10R_MULTITASK_UNCERTAINTY.to_csv(
    MULTITASK_ROOT / "uncertainty_metrics.csv",
    index=False,
)
V10R_MULTITASK_FOLDS.to_csv(
    MULTITASK_ROOT / "fold_summary.csv",
    index=False,
)

# ------------------------------------------------------------
# Summary plots
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

force_table = (
    V10R_MULTITASK_CONTINUOUS_METRICS[
        V10R_MULTITASK_CONTINUOUS_METRICS["group"]
        == "force_guarded_stance"
    ]
    .groupby("horizon_s", as_index=False)
    .agg(
        worst_p95=("p95", "max"),
        median_p95=("p95", "median"),
    )
)

axes[0, 0].plot(
    force_table["horizon_s"],
    force_table["worst_p95"],
    marker="o",
    label="worst p95",
)
axes[0, 0].plot(
    force_table["horizon_s"],
    force_table["median_p95"],
    marker="o",
    label="median p95",
)
axes[0, 0].axhline(4, color="red", linestyle="--")
axes[0, 0].set_title("Guarded stable-stance force")
axes[0, 0].set_ylabel("Absolute error [N]")
axes[0, 0].legend()

for group in (
    "joint_angle",
    "joint_velocity",
    "joint_torque",
):
    table = (
        V10R_MULTITASK_CONTINUOUS_METRICS[
            V10R_MULTITASK_CONTINUOUS_METRICS["group"]
            == group
        ]
        .groupby("horizon_s", as_index=False)
        .agg(mae=("mae", "mean"))
    )
    axes[0, 1].plot(
        table["horizon_s"],
        table["mae"],
        marker="o",
        label=group,
    )

axes[0, 1].set_title("Joint-state prediction MAE")
axes[0, 1].legend()

body_table = (
    V10R_MULTITASK_CONTINUOUS_METRICS[
        V10R_MULTITASK_CONTINUOUS_METRICS["group"]
        == "body_sensor"
    ]
    .groupby("horizon_s", as_index=False)
    .agg(
        median_mae=("mae", "median"),
        worst_mae=("mae", "max"),
    )
)
axes[1, 0].plot(
    body_table["horizon_s"],
    body_table["median_mae"],
    marker="o",
    label="median B-channel",
)
axes[1, 0].plot(
    body_table["horizon_s"],
    body_table["worst_mae"],
    marker="o",
    label="worst B-channel",
)
axes[1, 0].set_title("Body/IMU-cache channel prediction")
axes[1, 0].legend()

contact_table = (
    V10R_MULTITASK_CONTACT_METRICS
    .groupby("horizon_s", as_index=False)
    .agg(
        brier=("brier_score", "mean"),
        accuracy=("accuracy", "mean"),
        recall=("recall", "mean"),
    )
)
axes[1, 1].plot(
    contact_table["horizon_s"],
    contact_table["accuracy"],
    marker="o",
    label="accuracy",
)
axes[1, 1].plot(
    contact_table["horizon_s"],
    contact_table["recall"],
    marker="o",
    label="recall",
)
axes[1, 1].plot(
    contact_table["horizon_s"],
    contact_table["brier"],
    marker="o",
    label="Brier score",
)
axes[1, 1].set_title("Contact probability")
axes[1, 1].legend()

for axis in axes.flat:
    axis.set_xlabel("Forecast horizon [s]")
    axis.grid(alpha=0.25)

fig.tight_layout()
plot_path = MULTITASK_ROOT / "multitask_summary.png"
fig.savefig(plot_path, dpi=180, bbox_inches="tight")
plt.show()

print("Results:", MULTITASK_ROOT)

# %% notebook cell 48
# ============================================================
# CELL N — full V10-R readiness gate
# Refuses fabricated foot/impulse/geometry targets
# ============================================================
import pandas as pd
from IPython.display import display

FULL_V10R_REQUIREMENTS = {
    "foot positions": (
        "foot_world",
        "Verified URDF FK + verified base/camera/world transforms",
    ),
    "COM-relative feet": (
        "foot_r_H",
        "foot_world - com_H in the same verified frame",
    ),
    "touchdown impulse": (
        "touchdown_impulse",
        "Native 400 Hz force integration around audited touchdown time",
    ),
    "desired wrench": (
        "desired_wrench_target",
        "Verified centroidal/RNEA mechanics and inertial parameters",
    ),
    "causal force history": (
        "force_input_H",
        "Native force transformed into the verified heading frame",
    ),
    "geometry": (
        "causal_geometry_checkpoint",
        "RGB-D, intrinsics, extrinsics, timestamps and causal fusion",
    ),
}

readiness_rows = []

for description, (field, evidence) in (
    FULL_V10R_REQUIREMENTS.items()
):
    present_everywhere = all(
        field in CAUSAL_DATA[run_id]["run"]
        for run_id in CAUSAL_DATA
    )

    readiness_rows.append({
        "requirement": description,
        "required_field": field,
        "present_for_all_runs": present_everywhere,
        "required_evidence": evidence,
    })

V10R_FULL_READINESS = pd.DataFrame(readiness_rows)
display(V10R_FULL_READINESS)

FULL_V10R_READY = bool(
    V10R_FULL_READINESS["present_for_all_runs"].all()
    and globals().get("V10R_G0_PASS", False)
)

print("Full V10-R training ready:", FULL_V10R_READY)

if not FULL_V10R_READY:
    print(
        "\nCorrectly blocked: current caches can train joint/body/"
        "contact/force/E diagnostics, but cannot scientifically train "
        "foot position, impulse, geometry coupling or foothold planning."
    )
    print(
        "The next data stage must reconstruct these targets from the "
        "raw ROS bags, RGB-D streams, verified URDF and calibration. "
        "No zeros, aliases or guessed frames will be substituted."
    )

# %% notebook cell 49
# ======================================================================
# V10-R G0 RUNTIME BOOTSTRAP
# Exact recovery of STRICT S1/S2/S3 prerequisites needed by G0B.
#
# SAFE:
#   - no terrain download
#   - no preprocessing
#   - no training
#   - no geometry construction
#   - no force-frame assumptions
#   - NEVER overwrites an existing contract
#   - NEVER sets verified=True
#
# After this cell succeeds, rerun G0B UNCHANGED.
# ======================================================================

import os
import sys
import gc
import json
import shutil
import hashlib
import tempfile
import contextlib
import subprocess
import importlib.util

from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except ImportError:
    display = print


# ======================================================================
# 0. RED CHECK — confirm what state was actually missing
# ======================================================================

_G0_REQUIRED_BEFORE_REPAIR = [
    "RUN_REGISTRY",
    "RAW_ROOT",
    "PROJECT_ROOT",
    "STRICT_ROOT",
    "AUDIT_ROOT",
    "CONTRACT_ROOT",
    "raw_run_path",
    "locate_physics_cache",
    "utc_now",
    "atomic_write_json",
    "load_and_validate_contracts",
    "LEG_NAMES",
    "CONTACT_ON_N",
]

_missing_before = [
    name for name in _G0_REQUIRED_BEFORE_REPAIR
    if name not in globals()
]

print("============================================================")
print("G0 RUNTIME STATE BEFORE REPAIR")
print("============================================================")
print("Missing:", _missing_before)


# ======================================================================
# 1. Colab Drive mount
# ======================================================================

if (
    Path("/content").exists()
    and importlib.util.find_spec("google.colab") is not None
):
    from google.colab import drive

    if not Path("/content/drive/MyDrive").exists():
        print("\nMounting Google Drive...")
        drive.mount("/content/drive")


# ======================================================================
# 2. Package prerequisite used by G0B
# ======================================================================

def _ensure_package(import_name, pip_name=None):
    pip_name = pip_name or import_name

    if importlib.util.find_spec(import_name) is None:
        print("Installing:", pip_name)
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            pip_name,
        ])


_ensure_package("rosbags")
_ensure_package("scipy")

from rosbags.highlevel import AnyReader
from scipy.spatial.transform import Rotation as SciRot


# ======================================================================
# 3. EXACT STRICT S1 PATHS / CONSTANTS
# ======================================================================

STRICT_PIPELINE_VERSION = "V10R_STRICT_STAGED_V1"

ANALYSIS_TIER = "paper"
ALLOW_DIAGNOSTIC_TRAINING = False

# Exact paths used by the final V10-R notebook.
RAW_ROOT = Path("/content/tartanground_v10")
PROJECT_ROOT = Path("/content/drive/MyDrive/UGP_V7_FULL4B")

STRICT_ROOT = PROJECT_ROOT / "V10R_STRICT_STAGED"
PREPARED_ROOT = STRICT_ROOT / "prepared_runs"
AUDIT_ROOT = STRICT_ROOT / "audits"
CONTRACT_ROOT = STRICT_ROOT / "contracts"

MODEL_ROOT = STRICT_ROOT / "models" / ANALYSIS_TIER
RESULT_ROOT = STRICT_ROOT / "results" / ANALYSIS_TIER
FIGURE_ROOT = STRICT_ROOT / "figures" / ANALYSIS_TIER

for _p in (
    STRICT_ROOT,
    PREPARED_ROOT,
    AUDIT_ROOT,
    CONTRACT_ROOT,
    MODEL_ROOT,
    RESULT_ROOT,
    FIGURE_ROOT,
):
    _p.mkdir(parents=True, exist_ok=True)


# Force/contact constants relevant to G0B.
FORCE_UNIFORM_HZ = 400
FORCE_LOWPASS_HZ = 45.0
CONTACT_ON_N = 12.0
CONTACT_OFF_N = 7.0


# ======================================================================
# 4. EXACT RUN REGISTRY + CANONICAL ROBOT CONSTANTS
# ======================================================================

RUN_REGISTRY = [
    {
        "run_id": "forest_P2000",
        "terrain": "forest",
        "environment": "ForestEnv",
        "run": "P2000",
        "trajectory_id": "ForestEnv_P2000",
        "gait": "unreported",
        "policy": "unreported",
        "speed_bin": "unreported",
    },
    {
        "run_id": "swamp_marsh_P2023",
        "terrain": "swamp_marsh",
        "environment": "GreatMarsh",
        "run": "P2023",
        "trajectory_id": "GreatMarsh_P2023",
        "gait": "unreported",
        "policy": "unreported",
        "speed_bin": "unreported",
    },
    {
        "run_id": "rough_uneven_P2000",
        "terrain": "rough_uneven",
        "environment": "SeasonalForestWinter",
        "run": "P2000",
        "trajectory_id": "SeasonalForestWinter_P2000",
        "gait": "unreported",
        "policy": "unreported",
        "speed_bin": "unreported",
    },
    {
        "run_id": "wet_uneven_proxy_P2000",
        "terrain": "wet_uneven_proxy",
        "environment": "OldTownFall",
        "run": "P2000",
        "trajectory_id": "OldTownFall_P2000",
        "gait": "unreported",
        "policy": "unreported",
        "speed_bin": "unreported",
    },
]

FORCE_TOPICS = {
    "FL": "/state_estimator/contact_force_lf_foot",
    "FR": "/state_estimator/contact_force_rf_foot",
    "RL": "/state_estimator/contact_force_lh_foot",
    "RR": "/state_estimator/contact_force_rh_foot",
}

CANONICAL_JOINT_NAMES = (
    "LF_HAA", "LF_HFE", "LF_KFE",
    "RF_HAA", "RF_HFE", "RF_KFE",
    "LH_HAA", "LH_HFE", "LH_KFE",
    "RH_HAA", "RH_HFE", "RH_KFE",
)

LEG_NAMES = ("FL", "FR", "RL", "RR")

CAMERA_NAMES = (
    "lcam_front",
    "lcam_back",
    "lcam_left",
    "lcam_right",
    "lcam_top",
    "lcam_bottom",
)


# ======================================================================
# 5. EXACT CORE S2 HELPERS
# ======================================================================

def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)

    return h.hexdigest()


def atomic_write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )

    try:
        with os.fdopen(fd, "w") as f:
            json.dump(
                obj,
                f,
                indent=2,
                sort_keys=True,
            )
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp, path)

    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def _bounded_first(candidates):
    for p in map(Path, candidates):
        if p.exists():
            return p

    return None


def raw_run_path(spec):
    return (
        RAW_ROOT
        / spec["environment"]
        / "Data_anymal"
        / spec["run"]
    )


def locate_physics_cache(spec):
    name = (
        f"{spec['environment']}_"
        f"{spec['run']}_physics.npz"
    )

    roots = [
        PROJECT_ROOT / "cache",
        PROJECT_ROOT / "V10_CACHE",
        PROJECT_ROOT / "V7_FULL4B" / "cache",
        PROJECT_ROOT,
    ]

    # First use exact known locations.
    exact = _bounded_first([
        r / name for r in roots
    ])

    if exact is not None:
        return exact

    # Exact filename search, but ONLY inside the bounded known cache trees.
    for root in roots[:3]:
        if root.exists():
            hits = sorted(root.rglob(name))

            if hits:
                return hits[0]

    return None


# ======================================================================
# 6. STRICT S3 CONTRACT PATHS
# ======================================================================

FORCE_CONTRACT_PATH = (
    CONTRACT_ROOT / "force_frame_contract.json"
)

CALIBRATION_CONTRACT_PATH = (
    CONTRACT_ROOT / "calibration_contract.json"
)

SIGNAL_CONTRACT_PATH = (
    CONTRACT_ROOT / "signal_semantics_contract.json"
)

SPLIT_CONTRACT_PATH = (
    CONTRACT_ROOT / "trajectory_split.csv"
)


# ======================================================================
# 7. Restore UNVERIFIED templates ONLY IF files do not already exist.
#
# Existing scientific contracts are NEVER overwritten.
# This is exactly the important property we need.
# ======================================================================

def write_contract_template_once(path, obj):
    path = Path(path)

    if not path.exists():
        atomic_write_json(path, obj)

        print(
            "Created missing UNVERIFIED contract template:",
            path,
        )


write_contract_template_once(
    FORCE_CONTRACT_PATH,
    {
        "schema": "V10R_FORCE_FRAME_CONTRACT_V1",
        "verified": False,
        "verified_before_split": False,
        "verification_method":
            "publisher/source-code statement AND static known-load/sign test",
        "evidence": "REPLACE_WITH_AUDITABLE_REFERENCE",
        "force_direction": "environment_on_robot",
        "per_leg": {
            leg: {
                "source_frame": "UNRESOLVED",
                "axis_permutation": [0, 1, 2],
                "axis_sign": [1, 1, 1],
                "static_test_passed": False,
            }
            for leg in LEG_NAMES
        },
    },
)


write_contract_template_once(
    CALIBRATION_CONTRACT_PATH,
    {
        "schema": "V10R_CALIBRATION_CONTRACT_V1",
        "verified": False,
        "source":
            "official transform or independent calibration trajectory "
            "excluded from every model split",
        "evidence": "REPLACE_WITH_AUDITABLE_REFERENCE",
        "calibration_trajectory_id":
            "REPLACE; must not be a model train/val/test trajectory",
        "rotation_convention":
            "p_C = R_C_B @ p_B + t_C_B",
        "R_C_B": [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        "t_C_B_m": [0, 0, 0],
        "urdf_path": "REPLACE_WITH_ABSOLUTE_COLAB_PATH",
        "urdf_sha256": "REPLACE",
        "joint_names": {
            name: name
            for name in CANONICAL_JOINT_NAMES
        },
        "foot_frame_names": {
            "FL": "REPLACE",
            "FR": "REPLACE",
            "RL": "REPLACE",
            "RR": "REPLACE",
        },
        "camera_intrinsics": {
            cam: {
                "width": 640,
                "height": 640,
                "fx": 320.0,
                "fy": 320.0,
                "cx": 320.0,
                "cy": 320.0,
                "depth_sigma_a_m": 0.002,
                "depth_sigma_b_per_m": 0.001,
                "verified": False,
                "source": "REPLACE",
            }
            for cam in CAMERA_NAMES
        },
    },
)


write_contract_template_once(
    SIGNAL_CONTRACT_PATH,
    {
        "schema": "V10R_SIGNAL_CONTRACT_V1",
        "verified": False,
        "world_frame": "NED",
        "camera_axes": "x_forward_y_right_z_down",
        "pose_format": "tx_ty_tz_qx_qy_qz_qw",

        "imu_acceleration_semantics":
            "UNRESOLVED; required: specific_force_body",

        "imu_gyro_frame":
            "UNRESOLVED; required: base",

        "twist_frame":
            "UNRESOLVED; required: base",

        "body_B_layout": [
            "acc_x", "acc_y", "acc_z",
            "gyro_x", "gyro_y", "gyro_z",
            "v_x", "v_y", "v_z",
            "omega_x", "omega_y", "omega_z",
        ],

        "evidence":
            "REPLACE_WITH_AUDITABLE_REFERENCE",
    },
)


# ======================================================================
# 8. EXACT CONTRACT VALIDATION LOGIC
# ======================================================================

def load_json_contract(path):
    with Path(path).open() as f:
        return json.load(f)


def validate_force_contract(c):
    errors = []

    if c.get("schema") != "V10R_FORCE_FRAME_CONTRACT_V1":
        errors.append("wrong schema")

    if c.get("verified") is not True:
        errors.append("verified must be true")

    if c.get("verified_before_split") is not True:
        errors.append("verified_before_split must be true")

    if (
        len(str(c.get("evidence", "")).strip()) < 12
        or "REPLACE" in str(c.get("evidence", ""))
    ):
        errors.append("auditable evidence missing")

    if c.get("force_direction") not in {
        "environment_on_robot",
        "robot_on_environment",
    }:
        errors.append("force_direction invalid")

    allowed = {
        "base",
        "world_ned",
        "foot",
    }

    for leg in LEG_NAMES:
        x = c.get("per_leg", {}).get(leg, {})

        if x.get("source_frame") not in allowed:
            errors.append(
                f"{leg}: source_frame must be one of "
                f"{sorted(allowed)}"
            )

        perm = x.get("axis_permutation")
        sign = x.get("axis_sign")

        if sorted(perm or []) != [0, 1, 2]:
            errors.append(
                f"{leg}: invalid axis_permutation"
            )

        if (
            len(sign or []) != 3
            or any(v not in (-1, 1) for v in sign)
        ):
            errors.append(
                f"{leg}: invalid axis_sign"
            )

        if x.get("static_test_passed") is not True:
            errors.append(
                f"{leg}: static_test_passed is not true"
            )

    return errors


def validate_calibration_contract(
    c,
    model_trajectory_ids=(),
):
    errors = []

    if c.get("schema") != "V10R_CALIBRATION_CONTRACT_V1":
        errors.append("wrong schema")

    if c.get("verified") is not True:
        errors.append("verified must be true")

    if (
        len(str(c.get("evidence", "")).strip()) < 12
        or "REPLACE" in str(c.get("evidence", ""))
    ):
        errors.append("auditable evidence missing")

    R = np.asarray(
        c.get("R_C_B", []),
        float,
    )

    t = np.asarray(
        c.get("t_C_B_m", []),
        float,
    )

    if (
        R.shape != (3, 3)
        or not np.allclose(
            R.T @ R,
            np.eye(3),
            atol=1e-4,
        )
        or np.linalg.det(R) < 0.999
    ):
        errors.append(
            "R_C_B is not a proper rotation"
        )

    if (
        t.shape != (3,)
        or not np.isfinite(t).all()
    ):
        errors.append(
            "t_C_B_m invalid"
        )

    cal_id = str(
        c.get(
            "calibration_trajectory_id",
            "",
        )
    )

    if cal_id in set(model_trajectory_ids):
        errors.append(
            "calibration trajectory contaminates a model split"
        )

    up = Path(
        str(
            c.get(
                "urdf_path",
                "",
            )
        )
    )

    if not up.exists():
        errors.append(
            f"URDF missing: {up}"
        )

    elif c.get("urdf_sha256") != file_sha256(up):
        errors.append(
            "URDF SHA-256 does not match contract"
        )

    joint_names = c.get(
        "joint_names",
        {},
    )

    foot_names = c.get(
        "foot_frame_names",
        {},
    )

    if (
        set(joint_names)
        != set(CANONICAL_JOINT_NAMES)
        or any(
            not str(v).strip()
            for v in joint_names.values()
        )
    ):
        errors.append(
            "explicit canonical joint map missing"
        )

    if (
        set(foot_names)
        != set(LEG_NAMES)
        or any(
            (
                not str(v).strip()
                or "REPLACE" in str(v)
            )
            for v in foot_names.values()
        )
    ):
        errors.append(
            "explicit foot frame map missing"
        )

    for cam in CAMERA_NAMES:
        x = (
            c.get(
                "camera_intrinsics",
                {},
            )
            .get(
                cam,
                {},
            )
        )

        if x.get("verified") is not True:
            errors.append(
                f"{cam}: intrinsics not verified"
            )

        if min(
            float(x.get(k, 0))
            for k in (
                "width",
                "height",
                "fx",
                "fy",
            )
        ) <= 0:
            errors.append(
                f"{cam}: invalid intrinsics"
            )

        if min(
            float(x.get(k, -1))
            for k in (
                "depth_sigma_a_m",
                "depth_sigma_b_per_m",
            )
        ) < 0:
            errors.append(
                f"{cam}: depth noise model invalid"
            )

    return errors


def validate_signal_contract(c):
    errors = []

    if (
        c.get("schema")
        != "V10R_SIGNAL_CONTRACT_V1"
        or c.get("verified") is not True
    ):
        errors.append(
            "signal contract is not verified"
        )

    if c.get("world_frame") != "NED":
        errors.append(
            "this adapter requires the verified "
            "TartanGround NED convention"
        )

    if (
        c.get("camera_axes")
        != "x_forward_y_right_z_down"
    ):
        errors.append(
            "camera axes unresolved"
        )

    if (
        c.get("pose_format")
        != "tx_ty_tz_qx_qy_qz_qw"
    ):
        errors.append(
            "pose format unresolved"
        )

    if (
        c.get("imu_acceleration_semantics")
        != "specific_force_body"
    ):
        errors.append(
            "IMU acceleration must be verified "
            "as specific_force_body"
        )

    if c.get("imu_gyro_frame") != "base":
        errors.append(
            "gyro frame must be verified as base"
        )

    if c.get("twist_frame") != "base":
        errors.append(
            "twist frame must be verified as base"
        )

    if (
        len(str(c.get("evidence", "")).strip()) < 12
        or "REPLACE" in str(c.get("evidence", ""))
    ):
        errors.append(
            "signal evidence missing"
        )

    return errors


def load_and_validate_contracts(
    require_approved=True,
):
    fc, cc, sc = map(
        load_json_contract,
        (
            FORCE_CONTRACT_PATH,
            CALIBRATION_CONTRACT_PATH,
            SIGNAL_CONTRACT_PATH,
        ),
    )

    errors = {
        "force":
            validate_force_contract(fc),

        "calibration":
            validate_calibration_contract(
                cc,
                [
                    x["trajectory_id"]
                    for x in RUN_REGISTRY
                ],
            ),

        "signals":
            validate_signal_contract(sc),
    }

    if (
        require_approved
        and any(errors.values())
    ):
        message = "\n".join(
            f"{k}: "
            + (
                "; ".join(v)
                if v
                else "PASS"
            )
            for k, v
            in errors.items()
        )

        raise RuntimeError(
            "G0 CONTRACT GATE FAILED — "
            "no frame/calibration/semantics may be guessed.\n"
            + message
        )

    return fc, cc, sc, errors


# ======================================================================
# 9. GREEN / STATE VERIFICATION
#
# This does NOT assert scientific contracts are valid.
# It verifies only that the Python runtime needed for G0B now exists.
# ======================================================================

_G0_REQUIRED_AFTER_REPAIR = [
    "RUN_REGISTRY",
    "RAW_ROOT",
    "PROJECT_ROOT",
    "STRICT_ROOT",
    "AUDIT_ROOT",
    "CONTRACT_ROOT",
    "raw_run_path",
    "locate_physics_cache",
    "utc_now",
    "atomic_write_json",
    "load_and_validate_contracts",
    "LEG_NAMES",
    "CONTACT_ON_N",
]

_missing_after = [
    name
    for name in _G0_REQUIRED_AFTER_REPAIR
    if name not in globals()
]

assert not _missing_after, (
    "Bootstrap incomplete. Missing: "
    + repr(_missing_after)
)

assert len(RUN_REGISTRY) == 4
assert CONTACT_ON_N == 12.0
assert LEG_NAMES == (
    "FL",
    "FR",
    "RL",
    "RR",
)


print("\n============================================================")
print("G0 BOOTSTRAP STATE")
print("============================================================")

print(
    "RAW_ROOT     :",
    RAW_ROOT,
    "| exists =",
    RAW_ROOT.exists(),
)

print(
    "PROJECT_ROOT :",
    PROJECT_ROOT,
    "| exists =",
    PROJECT_ROOT.exists(),
)

print(
    "AUDIT_ROOT   :",
    AUDIT_ROOT,
)

print(
    "CONTRACT_ROOT:",
    CONTRACT_ROOT,
)


# Show exactly what data G0B will see.
rows = []

for spec in RUN_REGISTRY:
    raw = raw_run_path(spec)
    cache = locate_physics_cache(spec)

    bag = (
        raw
        / "rosbags"
        / f"{spec['run']}.bag"
    )

    rows.append({
        "run_id":
            spec["run_id"],

        "raw_present":
            raw.exists(),

        "bag_present":
            bag.exists(),

        "physics_cache_present":
            cache is not None,

        "raw_path":
            str(raw),

        "physics_cache":
            (
                str(cache)
                if cache is not None
                else ""
            ),
    })


G0_BOOTSTRAP_INVENTORY = pd.DataFrame(rows)

display(
    G0_BOOTSTRAP_INVENTORY
)


# Contract loader must execute, but contracts are EXPECTED
# to remain unresolved at this stage.
fc, cc, sc, bootstrap_contract_errors = (
    load_and_validate_contracts(
        require_approved=False
    )
)

print("\nCurrent contract blockers:")
print(
    json.dumps(
        bootstrap_contract_errors,
        indent=2,
    )
)


usable = G0_BOOTSTRAP_INVENTORY[
    G0_BOOTSTRAP_INVENTORY["raw_present"]
    &
    G0_BOOTSTRAP_INVENTORY["bag_present"]
    &
    G0_BOOTSTRAP_INVENTORY["physics_cache_present"]
]

print("\n============================================================")
print("BOOTSTRAP PREREQUISITES RESTORED")
print("============================================================")

print(
    "Usable raw+bag+cache runs:",
    usable["run_id"].tolist(),
)

print(
    "\nNext action: rerun the ORIGINAL G0B cell unchanged."
)

# %% notebook cell 51
# ============================================================
# V10-R DEPENDENCY FIX — run once, then rerun G0B unchanged
# ============================================================

import sys
import subprocess
import importlib.util

def ensure_package(import_name, pip_name=None):
    pip_name = pip_name or import_name

    if importlib.util.find_spec(import_name) is None:
        print(f"Installing missing dependency: {pip_name}")
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            pip_name
        ])
    else:
        print(f"{import_name}: already installed")


ensure_package("rosbags")
ensure_package("scipy")

# hard verification
from rosbags.highlevel import AnyReader
from scipy.spatial.transform import Rotation as SciRot

import rosbags
import scipy

print("\nDEPENDENCY CHECK PASSED")
print("rosbags:", getattr(rosbags, "__version__", "installed"))
print("scipy:", scipy.__version__)
print("AnyReader import: OK")

# %% notebook cell 52
# ============================================================
# V10-R RUNTIME STATE REPAIR / CHECK
# Does NOT download, preprocess, train, or modify contracts.
# ============================================================

# Restore the EXACT registry from the V10-R notebook
if "RUN_REGISTRY" not in globals():
    RUN_REGISTRY = [
        {
            "run_id": "forest_P2000",
            "terrain": "forest",
            "environment": "ForestEnv",
            "run": "P2000",
            "trajectory_id": "ForestEnv_P2000",
            "gait": "unreported",
            "policy": "unreported",
            "speed_bin": "unreported",
        },
        {
            "run_id": "swamp_marsh_P2023",
            "terrain": "swamp_marsh",
            "environment": "GreatMarsh",
            "run": "P2023",
            "trajectory_id": "GreatMarsh_P2023",
            "gait": "unreported",
            "policy": "unreported",
            "speed_bin": "unreported",
        },
        {
            "run_id": "rough_uneven_P2000",
            "terrain": "rough_uneven",
            "environment": "SeasonalForestWinter",
            "run": "P2000",
            "trajectory_id": "SeasonalForestWinter_P2000",
            "gait": "unreported",
            "policy": "unreported",
            "speed_bin": "unreported",
        },
        {
            "run_id": "wet_uneven_proxy_P2000",
            "terrain": "wet_uneven_proxy",
            "environment": "OldTownFall",
            "run": "P2000",
            "trajectory_id": "OldTownFall_P2000",
            "gait": "unreported",
            "policy": "unreported",
            "speed_bin": "unreported",
        },
    ]

# G0B also needs these existing notebook objects.
needed = [
    "raw_run_path",
    "locate_physics_cache",
    "CONTACT_ON_N",
    "AUDIT_ROOT",
    "utc_now",
    "atomic_write_json",
    "load_and_validate_contracts",
]

missing = [x for x in needed if x not in globals()]

print("RUN_REGISTRY: OK")
print("runs:", [x["run_id"] for x in RUN_REGISTRY])

if missing:
    print("\n⚠️ Earlier notebook state is incomplete.")
    print("Missing:", missing)
    print(
        "\nDo NOT patch these one-by-one.\n"
        "Rerun the notebook's earlier setup/config/helper cells "
        "up to the contract/audit definitions, then rerun G0B."
    )
else:
    print("\nAll G0B prerequisites are present.")
    print("Now rerun the full G0B cell.")

# %% notebook cell 53
# ======================================================================
# V10-R G0 RUNTIME BOOTSTRAP
# Exact recovery of STRICT S1/S2/S3 prerequisites needed by G0B.
#
# SAFE:
#   - no terrain download
#   - no preprocessing
#   - no training
#   - no geometry construction
#   - no force-frame assumptions
#   - NEVER overwrites an existing contract
#   - NEVER sets verified=True
#
# After this cell succeeds, rerun G0B UNCHANGED.
# ======================================================================

import os
import sys
import gc
import json
import shutil
import hashlib
import tempfile
import contextlib
import subprocess
import importlib.util

from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except ImportError:
    display = print


# ======================================================================
# 0. RED CHECK — confirm what state was actually missing
# ======================================================================

_G0_REQUIRED_BEFORE_REPAIR = [
    "RUN_REGISTRY",
    "RAW_ROOT",
    "PROJECT_ROOT",
    "STRICT_ROOT",
    "AUDIT_ROOT",
    "CONTRACT_ROOT",
    "raw_run_path",
    "locate_physics_cache",
    "utc_now",
    "atomic_write_json",
    "load_and_validate_contracts",
    "LEG_NAMES",
    "CONTACT_ON_N",
]

_missing_before = [
    name for name in _G0_REQUIRED_BEFORE_REPAIR
    if name not in globals()
]

print("============================================================")
print("G0 RUNTIME STATE BEFORE REPAIR")
print("============================================================")
print("Missing:", _missing_before)


# ======================================================================
# 1. Colab Drive mount
# ======================================================================

if (
    Path("/content").exists()
    and importlib.util.find_spec("google.colab") is not None
):
    from google.colab import drive

    if not Path("/content/drive/MyDrive").exists():
        print("\nMounting Google Drive...")
        drive.mount("/content/drive")


# ======================================================================
# 2. Package prerequisite used by G0B
# ======================================================================

def _ensure_package(import_name, pip_name=None):
    pip_name = pip_name or import_name

    if importlib.util.find_spec(import_name) is None:
        print("Installing:", pip_name)
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            pip_name,
        ])


_ensure_package("rosbags")
_ensure_package("scipy")

from rosbags.highlevel import AnyReader
from scipy.spatial.transform import Rotation as SciRot


# ======================================================================
# 3. EXACT STRICT S1 PATHS / CONSTANTS
# ======================================================================

STRICT_PIPELINE_VERSION = "V10R_STRICT_STAGED_V1"

ANALYSIS_TIER = "paper"
ALLOW_DIAGNOSTIC_TRAINING = False

# Exact paths used by the final V10-R notebook.
RAW_ROOT = Path("/content/tartanground_v10")
PROJECT_ROOT = Path("/content/drive/MyDrive/UGP_V7_FULL4B")

STRICT_ROOT = PROJECT_ROOT / "V10R_STRICT_STAGED"
PREPARED_ROOT = STRICT_ROOT / "prepared_runs"
AUDIT_ROOT = STRICT_ROOT / "audits"
CONTRACT_ROOT = STRICT_ROOT / "contracts"

MODEL_ROOT = STRICT_ROOT / "models" / ANALYSIS_TIER
RESULT_ROOT = STRICT_ROOT / "results" / ANALYSIS_TIER
FIGURE_ROOT = STRICT_ROOT / "figures" / ANALYSIS_TIER

for _p in (
    STRICT_ROOT,
    PREPARED_ROOT,
    AUDIT_ROOT,
    CONTRACT_ROOT,
    MODEL_ROOT,
    RESULT_ROOT,
    FIGURE_ROOT,
):
    _p.mkdir(parents=True, exist_ok=True)


# Force/contact constants relevant to G0B.
FORCE_UNIFORM_HZ = 400
FORCE_LOWPASS_HZ = 45.0
CONTACT_ON_N = 12.0
CONTACT_OFF_N = 7.0


# ======================================================================
# 4. EXACT RUN REGISTRY + CANONICAL ROBOT CONSTANTS
# ======================================================================

RUN_REGISTRY = [
    {
        "run_id": "forest_P2000",
        "terrain": "forest",
        "environment": "ForestEnv",
        "run": "P2000",
        "trajectory_id": "ForestEnv_P2000",
        "gait": "unreported",
        "policy": "unreported",
        "speed_bin": "unreported",
    },
    {
        "run_id": "swamp_marsh_P2023",
        "terrain": "swamp_marsh",
        "environment": "GreatMarsh",
        "run": "P2023",
        "trajectory_id": "GreatMarsh_P2023",
        "gait": "unreported",
        "policy": "unreported",
        "speed_bin": "unreported",
    },
    {
        "run_id": "rough_uneven_P2000",
        "terrain": "rough_uneven",
        "environment": "SeasonalForestWinter",
        "run": "P2000",
        "trajectory_id": "SeasonalForestWinter_P2000",
        "gait": "unreported",
        "policy": "unreported",
        "speed_bin": "unreported",
    },
    {
        "run_id": "wet_uneven_proxy_P2000",
        "terrain": "wet_uneven_proxy",
        "environment": "OldTownFall",
        "run": "P2000",
        "trajectory_id": "OldTownFall_P2000",
        "gait": "unreported",
        "policy": "unreported",
        "speed_bin": "unreported",
    },
]

FORCE_TOPICS = {
    "FL": "/state_estimator/contact_force_lf_foot",
    "FR": "/state_estimator/contact_force_rf_foot",
    "RL": "/state_estimator/contact_force_lh_foot",
    "RR": "/state_estimator/contact_force_rh_foot",
}

CANONICAL_JOINT_NAMES = (
    "LF_HAA", "LF_HFE", "LF_KFE",
    "RF_HAA", "RF_HFE", "RF_KFE",
    "LH_HAA", "LH_HFE", "LH_KFE",
    "RH_HAA", "RH_HFE", "RH_KFE",
)

LEG_NAMES = ("FL", "FR", "RL", "RR")

CAMERA_NAMES = (
    "lcam_front",
    "lcam_back",
    "lcam_left",
    "lcam_right",
    "lcam_top",
    "lcam_bottom",
)


# ======================================================================
# 5. EXACT CORE S2 HELPERS
# ======================================================================

def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)

    return h.hexdigest()


def atomic_write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )

    try:
        with os.fdopen(fd, "w") as f:
            json.dump(
                obj,
                f,
                indent=2,
                sort_keys=True,
            )
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp, path)

    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def _bounded_first(candidates):
    for p in map(Path, candidates):
        if p.exists():
            return p

    return None


def raw_run_path(spec):
    return (
        RAW_ROOT
        / spec["environment"]
        / "Data_anymal"
        / spec["run"]
    )


def locate_physics_cache(spec):
    name = (
        f"{spec['environment']}_"
        f"{spec['run']}_physics.npz"
    )

    roots = [
        PROJECT_ROOT / "cache",
        PROJECT_ROOT / "V10_CACHE",
        PROJECT_ROOT / "V7_FULL4B" / "cache",
        PROJECT_ROOT,
    ]

    # First use exact known locations.
    exact = _bounded_first([
        r / name for r in roots
    ])

    if exact is not None:
        return exact

    # Exact filename search, but ONLY inside the bounded known cache trees.
    for root in roots[:3]:
        if root.exists():
            hits = sorted(root.rglob(name))

            if hits:
                return hits[0]

    return None


# ======================================================================
# 6. STRICT S3 CONTRACT PATHS
# ======================================================================

FORCE_CONTRACT_PATH = (
    CONTRACT_ROOT / "force_frame_contract.json"
)

CALIBRATION_CONTRACT_PATH = (
    CONTRACT_ROOT / "calibration_contract.json"
)

SIGNAL_CONTRACT_PATH = (
    CONTRACT_ROOT / "signal_semantics_contract.json"
)

SPLIT_CONTRACT_PATH = (
    CONTRACT_ROOT / "trajectory_split.csv"
)


# ======================================================================
# 7. Restore UNVERIFIED templates ONLY IF files do not already exist.
#
# Existing scientific contracts are NEVER overwritten.
# This is exactly the important property we need.
# ======================================================================

def write_contract_template_once(path, obj):
    path = Path(path)

    if not path.exists():
        atomic_write_json(path, obj)

        print(
            "Created missing UNVERIFIED contract template:",
            path,
        )


write_contract_template_once(
    FORCE_CONTRACT_PATH,
    {
        "schema": "V10R_FORCE_FRAME_CONTRACT_V1",
        "verified": False,
        "verified_before_split": False,
        "verification_method":
            "publisher/source-code statement AND static known-load/sign test",
        "evidence": "REPLACE_WITH_AUDITABLE_REFERENCE",
        "force_direction": "environment_on_robot",
        "per_leg": {
            leg: {
                "source_frame": "UNRESOLVED",
                "axis_permutation": [0, 1, 2],
                "axis_sign": [1, 1, 1],
                "static_test_passed": False,
            }
            for leg in LEG_NAMES
        },
    },
)


write_contract_template_once(
    CALIBRATION_CONTRACT_PATH,
    {
        "schema": "V10R_CALIBRATION_CONTRACT_V1",
        "verified": False,
        "source":
            "official transform or independent calibration trajectory "
            "excluded from every model split",
        "evidence": "REPLACE_WITH_AUDITABLE_REFERENCE",
        "calibration_trajectory_id":
            "REPLACE; must not be a model train/val/test trajectory",
        "rotation_convention":
            "p_C = R_C_B @ p_B + t_C_B",
        "R_C_B": [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        "t_C_B_m": [0, 0, 0],
        "urdf_path": "REPLACE_WITH_ABSOLUTE_COLAB_PATH",
        "urdf_sha256": "REPLACE",
        "joint_names": {
            name: name
            for name in CANONICAL_JOINT_NAMES
        },
        "foot_frame_names": {
            "FL": "REPLACE",
            "FR": "REPLACE",
            "RL": "REPLACE",
            "RR": "REPLACE",
        },
        "camera_intrinsics": {
            cam: {
                "width": 640,
                "height": 640,
                "fx": 320.0,
                "fy": 320.0,
                "cx": 320.0,
                "cy": 320.0,
                "depth_sigma_a_m": 0.002,
                "depth_sigma_b_per_m": 0.001,
                "verified": False,
                "source": "REPLACE",
            }
            for cam in CAMERA_NAMES
        },
    },
)


write_contract_template_once(
    SIGNAL_CONTRACT_PATH,
    {
        "schema": "V10R_SIGNAL_CONTRACT_V1",
        "verified": False,
        "world_frame": "NED",
        "camera_axes": "x_forward_y_right_z_down",
        "pose_format": "tx_ty_tz_qx_qy_qz_qw",

        "imu_acceleration_semantics":
            "UNRESOLVED; required: specific_force_body",

        "imu_gyro_frame":
            "UNRESOLVED; required: base",

        "twist_frame":
            "UNRESOLVED; required: base",

        "body_B_layout": [
            "acc_x", "acc_y", "acc_z",
            "gyro_x", "gyro_y", "gyro_z",
            "v_x", "v_y", "v_z",
            "omega_x", "omega_y", "omega_z",
        ],

        "evidence":
            "REPLACE_WITH_AUDITABLE_REFERENCE",
    },
)


# ======================================================================
# 8. EXACT CONTRACT VALIDATION LOGIC
# ======================================================================

def load_json_contract(path):
    with Path(path).open() as f:
        return json.load(f)


def validate_force_contract(c):
    errors = []

    if c.get("schema") != "V10R_FORCE_FRAME_CONTRACT_V1":
        errors.append("wrong schema")

    if c.get("verified") is not True:
        errors.append("verified must be true")

    if c.get("verified_before_split") is not True:
        errors.append("verified_before_split must be true")

    if (
        len(str(c.get("evidence", "")).strip()) < 12
        or "REPLACE" in str(c.get("evidence", ""))
    ):
        errors.append("auditable evidence missing")

    if c.get("force_direction") not in {
        "environment_on_robot",
        "robot_on_environment",
    }:
        errors.append("force_direction invalid")

    allowed = {
        "base",
        "world_ned",
        "foot",
    }

    for leg in LEG_NAMES:
        x = c.get("per_leg", {}).get(leg, {})

        if x.get("source_frame") not in allowed:
            errors.append(
                f"{leg}: source_frame must be one of "
                f"{sorted(allowed)}"
            )

        perm = x.get("axis_permutation")
        sign = x.get("axis_sign")

        if sorted(perm or []) != [0, 1, 2]:
            errors.append(
                f"{leg}: invalid axis_permutation"
            )

        if (
            len(sign or []) != 3
            or any(v not in (-1, 1) for v in sign)
        ):
            errors.append(
                f"{leg}: invalid axis_sign"
            )

        if x.get("static_test_passed") is not True:
            errors.append(
                f"{leg}: static_test_passed is not true"
            )

    return errors


def validate_calibration_contract(
    c,
    model_trajectory_ids=(),
):
    errors = []

    if c.get("schema") != "V10R_CALIBRATION_CONTRACT_V1":
        errors.append("wrong schema")

    if c.get("verified") is not True:
        errors.append("verified must be true")

    if (
        len(str(c.get("evidence", "")).strip()) < 12
        or "REPLACE" in str(c.get("evidence", ""))
    ):
        errors.append("auditable evidence missing")

    R = np.asarray(
        c.get("R_C_B", []),
        float,
    )

    t = np.asarray(
        c.get("t_C_B_m", []),
        float,
    )

    if (
        R.shape != (3, 3)
        or not np.allclose(
            R.T @ R,
            np.eye(3),
            atol=1e-4,
        )
        or np.linalg.det(R) < 0.999
    ):
        errors.append(
            "R_C_B is not a proper rotation"
        )

    if (
        t.shape != (3,)
        or not np.isfinite(t).all()
    ):
        errors.append(
            "t_C_B_m invalid"
        )

    cal_id = str(
        c.get(
            "calibration_trajectory_id",
            "",
        )
    )

    if cal_id in set(model_trajectory_ids):
        errors.append(
            "calibration trajectory contaminates a model split"
        )

    up = Path(
        str(
            c.get(
                "urdf_path",
                "",
            )
        )
    )

    if not up.exists():
        errors.append(
            f"URDF missing: {up}"
        )

    elif c.get("urdf_sha256") != file_sha256(up):
        errors.append(
            "URDF SHA-256 does not match contract"
        )

    joint_names = c.get(
        "joint_names",
        {},
    )

    foot_names = c.get(
        "foot_frame_names",
        {},
    )

    if (
        set(joint_names)
        != set(CANONICAL_JOINT_NAMES)
        or any(
            not str(v).strip()
            for v in joint_names.values()
        )
    ):
        errors.append(
            "explicit canonical joint map missing"
        )

    if (
        set(foot_names)
        != set(LEG_NAMES)
        or any(
            (
                not str(v).strip()
                or "REPLACE" in str(v)
            )
            for v in foot_names.values()
        )
    ):
        errors.append(
            "explicit foot frame map missing"
        )

    for cam in CAMERA_NAMES:
        x = (
            c.get(
                "camera_intrinsics",
                {},
            )
            .get(
                cam,
                {},
            )
        )

        if x.get("verified") is not True:
            errors.append(
                f"{cam}: intrinsics not verified"
            )

        if min(
            float(x.get(k, 0))
            for k in (
                "width",
                "height",
                "fx",
                "fy",
            )
        ) <= 0:
            errors.append(
                f"{cam}: invalid intrinsics"
            )

        if min(
            float(x.get(k, -1))
            for k in (
                "depth_sigma_a_m",
                "depth_sigma_b_per_m",
            )
        ) < 0:
            errors.append(
                f"{cam}: depth noise model invalid"
            )

    return errors


def validate_signal_contract(c):
    errors = []

    if (
        c.get("schema")
        != "V10R_SIGNAL_CONTRACT_V1"
        or c.get("verified") is not True
    ):
        errors.append(
            "signal contract is not verified"
        )

    if c.get("world_frame") != "NED":
        errors.append(
            "this adapter requires the verified "
            "TartanGround NED convention"
        )

    if (
        c.get("camera_axes")
        != "x_forward_y_right_z_down"
    ):
        errors.append(
            "camera axes unresolved"
        )

    if (
        c.get("pose_format")
        != "tx_ty_tz_qx_qy_qz_qw"
    ):
        errors.append(
            "pose format unresolved"
        )

    if (
        c.get("imu_acceleration_semantics")
        != "specific_force_body"
    ):
        errors.append(
            "IMU acceleration must be verified "
            "as specific_force_body"
        )

    if c.get("imu_gyro_frame") != "base":
        errors.append(
            "gyro frame must be verified as base"
        )

    if c.get("twist_frame") != "base":
        errors.append(
            "twist frame must be verified as base"
        )

    if (
        len(str(c.get("evidence", "")).strip()) < 12
        or "REPLACE" in str(c.get("evidence", ""))
    ):
        errors.append(
            "signal evidence missing"
        )

    return errors


def load_and_validate_contracts(
    require_approved=True,
):
    fc, cc, sc = map(
        load_json_contract,
        (
            FORCE_CONTRACT_PATH,
            CALIBRATION_CONTRACT_PATH,
            SIGNAL_CONTRACT_PATH,
        ),
    )

    errors = {
        "force":
            validate_force_contract(fc),

        "calibration":
            validate_calibration_contract(
                cc,
                [
                    x["trajectory_id"]
                    for x in RUN_REGISTRY
                ],
            ),

        "signals":
            validate_signal_contract(sc),
    }

    if (
        require_approved
        and any(errors.values())
    ):
        message = "\n".join(
            f"{k}: "
            + (
                "; ".join(v)
                if v
                else "PASS"
            )
            for k, v
            in errors.items()
        )

        raise RuntimeError(
            "G0 CONTRACT GATE FAILED — "
            "no frame/calibration/semantics may be guessed.\n"
            + message
        )

    return fc, cc, sc, errors


# ======================================================================
# 9. GREEN / STATE VERIFICATION
#
# This does NOT assert scientific contracts are valid.
# It verifies only that the Python runtime needed for G0B now exists.
# ======================================================================

_G0_REQUIRED_AFTER_REPAIR = [
    "RUN_REGISTRY",
    "RAW_ROOT",
    "PROJECT_ROOT",
    "STRICT_ROOT",
    "AUDIT_ROOT",
    "CONTRACT_ROOT",
    "raw_run_path",
    "locate_physics_cache",
    "utc_now",
    "atomic_write_json",
    "load_and_validate_contracts",
    "LEG_NAMES",
    "CONTACT_ON_N",
]

_missing_after = [
    name
    for name in _G0_REQUIRED_AFTER_REPAIR
    if name not in globals()
]

assert not _missing_after, (
    "Bootstrap incomplete. Missing: "
    + repr(_missing_after)
)

assert len(RUN_REGISTRY) == 4
assert CONTACT_ON_N == 12.0
assert LEG_NAMES == (
    "FL",
    "FR",
    "RL",
    "RR",
)


print("\n============================================================")
print("G0 BOOTSTRAP STATE")
print("============================================================")

print(
    "RAW_ROOT     :",
    RAW_ROOT,
    "| exists =",
    RAW_ROOT.exists(),
)

print(
    "PROJECT_ROOT :",
    PROJECT_ROOT,
    "| exists =",
    PROJECT_ROOT.exists(),
)

print(
    "AUDIT_ROOT   :",
    AUDIT_ROOT,
)

print(
    "CONTRACT_ROOT:",
    CONTRACT_ROOT,
)


# Show exactly what data G0B will see.
rows = []

for spec in RUN_REGISTRY:
    raw = raw_run_path(spec)
    cache = locate_physics_cache(spec)

    bag = (
        raw
        / "rosbags"
        / f"{spec['run']}.bag"
    )

    rows.append({
        "run_id":
            spec["run_id"],

        "raw_present":
            raw.exists(),

        "bag_present":
            bag.exists(),

        "physics_cache_present":
            cache is not None,

        "raw_path":
            str(raw),

        "physics_cache":
            (
                str(cache)
                if cache is not None
                else ""
            ),
    })


G0_BOOTSTRAP_INVENTORY = pd.DataFrame(rows)

display(
    G0_BOOTSTRAP_INVENTORY
)


# Contract loader must execute, but contracts are EXPECTED
# to remain unresolved at this stage.
fc, cc, sc, bootstrap_contract_errors = (
    load_and_validate_contracts(
        require_approved=False
    )
)

print("\nCurrent contract blockers:")
print(
    json.dumps(
        bootstrap_contract_errors,
        indent=2,
    )
)


usable = G0_BOOTSTRAP_INVENTORY[
    G0_BOOTSTRAP_INVENTORY["raw_present"]
    &
    G0_BOOTSTRAP_INVENTORY["bag_present"]
    &
    G0_BOOTSTRAP_INVENTORY["physics_cache_present"]
]

print("\n============================================================")
print("BOOTSTRAP PREREQUISITES RESTORED")
print("============================================================")

print(
    "Usable raw+bag+cache runs:",
    usable["run_id"].tolist(),
)

print(
    "\nNext action: rerun the ORIGINAL G0B cell unchanged."
)

# %% notebook cell 54
# ======================================================================
# V10-R CONTINUATION G0B
# REAL-DATA FRAME / SIGNAL / FORCE EVIDENCE RESOLVER
#
# APPEND ONLY.  DOES NOT:
#   - alter any contract
#   - fabricate any target
#   - alter the first 30 cells
#   - start training
#   - download 4 terrains
#
# It answers the exact blockers left by the last executed cell.
# ======================================================================

from pathlib import Path
from itertools import permutations, product
import json, math, dataclasses
import numpy as np
import pandas as pd

from rosbags.highlevel import AnyReader
from scipy.spatial.transform import Rotation as SciRot


# ----------------------------------------------------------------------
# 0. EXACT TWO MODEL RUNS WE WILL USE AFTER G0 PASSES
# ----------------------------------------------------------------------

TWO_RUN_IDS = (
    "swamp_marsh_P2023",       # GreatMarsh / P2023
    "rough_uneven_P2000",      # SeasonalForestWinter / P2000
)

TWO_SPECS = [
    s for s in RUN_REGISTRY
    if s["run_id"] in TWO_RUN_IDS
]

if set(s["run_id"] for s in TWO_SPECS) != set(TWO_RUN_IDS):
    raise RuntimeError(
        "Could not resolve exact two target runs from RUN_REGISTRY.\n"
        f"wanted={TWO_RUN_IDS}\n"
        f"found={[s['run_id'] for s in TWO_SPECS]}"
    )

print("MODEL RUNS FIXED TO:")
for s in TWO_SPECS:
    print(" ", s["run_id"], "->", s["environment"], s["run"])


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _stamp_seconds(stamp):
    try:
        return float(stamp.sec) + 1e-9 * float(stamp.nanosec)
    except Exception:
        try:
            return float(stamp.sec) + 1e-9 * float(stamp.nsec)
        except Exception:
            return np.nan


def _msg_header_time(msg):
    try:
        return _stamp_seconds(msg.header.stamp)
    except Exception:
        return np.nan


def _causal_hold_local(t_src, x_src, t_query):
    t_src = np.asarray(t_src, np.float64)
    x_src = np.asarray(x_src)
    t_query = np.asarray(t_query, np.float64)

    idx = np.searchsorted(t_src, t_query, side="right") - 1
    if np.any(idx < 0):
        raise RuntimeError("query precedes beginning of source stream")
    idx = np.clip(idx, 0, len(t_src) - 1)
    return x_src[idx]


def _proper_signed_permutations():
    """
    All 24 orientation-preserving signed permutation matrices.
    These cover axis reorderings + sign flips without permitting reflection.
    """
    mats = []
    for p in permutations(range(3)):
        for signs in product((-1.0, 1.0), repeat=3):
            R = np.zeros((3, 3), np.float64)
            for row, col in enumerate(p):
                R[row, col] = signs[row]
            if np.linalg.det(R) > 0.999:
                mats.append((p, signs, R))
    assert len(mats) == 24
    return mats


SIGNED_ROTATIONS = _proper_signed_permutations()


def _safe_corr(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 20:
        return np.nan
    a = a[good]
    b = b[good]
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


# ----------------------------------------------------------------------
# 1. Cache-level IMU <-> BASE axis identifiability
#
# B layout from the existing notebook:
#   0:3   accel
#   3:6   gyro
#   6:9   base linear twist
#   9:12  base angular twist
#
# We DO NOT assume the first six are already base-frame.
# Instead, find the proper axis mapping that makes raw gyro agree with the
# independently recorded base-frame angular velocity.
# ----------------------------------------------------------------------

def audit_signal_axes(spec):
    cache = locate_physics_cache(spec)
    if cache is None:
        raise FileNotFoundError(
            f"{spec['run_id']}: physics cache missing"
        )

    with np.load(cache, allow_pickle=False) as z:
        if "B" not in z.files:
            raise RuntimeError(f"{cache}: B missing")
        B = np.asarray(z["B"], np.float64)

    if B.ndim != 2 or B.shape[1] < 12:
        raise RuntimeError(f"{cache}: unexpected B shape {B.shape}")

    acc_raw   = B[:, 0:3]
    gyro_raw  = B[:, 3:6]
    vel_base  = B[:, 6:9]
    omega_base = B[:, 9:12]

    # Avoid nearly static samples dominating the axis search.
    excited = (
        np.linalg.norm(omega_base, axis=1) >
        max(0.025, np.quantile(np.linalg.norm(omega_base, axis=1), .35))
    )
    if excited.sum() < 100:
        excited = np.ones(len(B), bool)

    candidates = []

    for perm, signs, R_B_FROM_RAW in SIGNED_ROTATIONS:
        gyro_B = gyro_raw @ R_B_FROM_RAW.T

        e = gyro_B[excited] - omega_base[excited]

        rmse = float(np.sqrt(np.mean(e**2)))
        mae  = float(np.mean(np.abs(e)))

        corr_axis = [
            _safe_corr(
                gyro_B[excited, k],
                omega_base[excited, k]
            )
            for k in range(3)
        ]

        finite_corr = [c for c in corr_axis if np.isfinite(c)]
        mean_corr = (
            float(np.mean(finite_corr))
            if finite_corr else -1.0
        )

        # Lower = better.
        score = rmse + 0.05 * (1.0 - mean_corr)

        candidates.append({
            "perm": tuple(int(x) for x in perm),
            "signs": tuple(int(x) for x in signs),
            "R_B_FROM_RAW": R_B_FROM_RAW,
            "rmse_rad_s": rmse,
            "mae_rad_s": mae,
            "corr_x": corr_axis[0],
            "corr_y": corr_axis[1],
            "corr_z": corr_axis[2],
            "mean_corr": mean_corr,
            "score": score,
        })

    candidates = sorted(candidates, key=lambda x: x["score"])
    best = candidates[0]
    second = candidates[1]

    R = best["R_B_FROM_RAW"]
    acc_B = acc_raw @ R.T
    gyro_B = gyro_raw @ R.T

    acc_norm = np.linalg.norm(acc_raw, axis=1)

    # Quasi-static samples judged using signals independent of acceleration.
    quasi = (
        (np.linalg.norm(vel_base, axis=1) < 0.20)
        &
        (np.linalg.norm(omega_base, axis=1) < 0.18)
    )

    if quasi.sum() < 100:
        quasi = np.ones(len(B), bool)

    result = {
        "run_id": spec["run_id"],
        "cache": str(cache),

        "best_perm": best["perm"],
        "best_signs": best["signs"],
        "R_B_FROM_RAW": R.tolist(),

        "gyro_rmse_rad_s": best["rmse_rad_s"],
        "gyro_mae_rad_s": best["mae_rad_s"],
        "gyro_mean_corr": best["mean_corr"],

        "second_best_score": float(second["score"]),
        "best_score": float(best["score"]),
        "score_margin": float(second["score"] - best["score"]),

        "acc_norm_q05_q50_q95": (
            np.quantile(acc_norm, [.05, .50, .95]).tolist()
        ),

        "quasi_static_samples": int(quasi.sum()),
        "quasi_acc_raw_median": (
            np.median(acc_raw[quasi], axis=0).tolist()
        ),
        "quasi_acc_base_candidate_median": (
            np.median(acc_B[quasi], axis=0).tolist()
        ),

        "gyro_raw_std": np.std(gyro_raw, axis=0).tolist(),
        "gyro_base_twist_std": np.std(omega_base, axis=0).tolist(),
    }

    print("\n================ SIGNAL AXIS AUDIT =", spec["run_id"], "================")
    print("cache:", cache)

    show = pd.DataFrame([
        {
            "rank": i + 1,
            "perm": c["perm"],
            "signs": c["signs"],
            "gyro_rmse": c["rmse_rad_s"],
            "mean_corr": c["mean_corr"],
            "score": c["score"],
        }
        for i, c in enumerate(candidates[:6])
    ])

    display(show)

    print("BEST R_B_FROM_RAW:")
    print(R)

    print(
        "acc norm q05/q50/q95:",
        result["acc_norm_q05_q50_q95"]
    )
    print(
        "quasi raw accel median:",
        result["quasi_acc_raw_median"]
    )
    print(
        "quasi candidate BASE accel median:",
        result["quasi_acc_base_candidate_median"]
    )

    return result


# ----------------------------------------------------------------------
# 2. Inspect ALL raw ROS frame evidence that the previous diagnostic did
#    not fully expose.
#
# Specifically:
#   /tf
#   /tf_static
#   AnymalState.frame_transforms
#   AnymalState.contacts
#   force-message header frames
#
# This may resolve base<->camera and contact-force provenance directly,
# which is much stronger than guessing from force magnitude.
# ----------------------------------------------------------------------

FORCE_TOPICS = {
    "FL": "/state_estimator/contact_force_lf_foot",
    "FR": "/state_estimator/contact_force_rf_foot",
    "RL": "/state_estimator/contact_force_lh_foot",
    "RR": "/state_estimator/contact_force_rh_foot",
}

POSE_TOPIC = "/state_estimator/pose_in_odom"
STATE_TOPIC = "/state_estimator/anymal_state"


def _simple_value(x):
    if isinstance(x, (str, int, float, bool, np.number)):
        return x
    return None


def _field_names(obj):
    # rosbags generated message classes expose slots / dataclass fields.
    names = []

    try:
        names += [
            f.name for f in dataclasses.fields(obj)
        ]
    except Exception:
        pass

    if not names:
        try:
            names += [
                n for n in obj.__slots__
                if not n.startswith("_")
            ]
        except Exception:
            pass

    if not names:
        names = [
            n for n in dir(obj)
            if not n.startswith("_")
            and not callable(getattr(obj, n, None))
        ]

    # Stable unique order.
    return list(dict.fromkeys(names))


def _describe_object(obj, depth=0, max_depth=2, max_seq=3):
    if depth > max_depth:
        return f"<{type(obj).__name__}>"

    if obj is None:
        return None

    simple = _simple_value(obj)
    if simple is not None:
        return simple

    if isinstance(obj, np.ndarray):
        return {
            "type": "ndarray",
            "shape": tuple(obj.shape),
            "sample": obj.reshape(-1)[:6].tolist(),
        }

    if isinstance(obj, (list, tuple)):
        return [
            _describe_object(v, depth+1, max_depth, max_seq)
            for v in list(obj)[:max_seq]
        ]

    out = {"__type__": type(obj).__name__}
    for name in _field_names(obj):
        try:
            value = getattr(obj, name)
        except Exception:
            continue

        if name.startswith("STATE_ERROR"):
            continue

        out[name] = _describe_object(
            value,
            depth+1,
            max_depth,
            max_seq
        )

    return out


def audit_ros_frame_evidence(spec, max_messages=12000):
    root = raw_run_path(spec)
    bag = root / "rosbags" / f"{spec['run']}.bag"

    if not bag.exists():
        raise FileNotFoundError(bag)

    force_frames = {leg: set() for leg in LEG_NAMES}
    tf_pairs = set()
    state_frame_pairs = set()

    state_sample = None
    contacts_sample = None
    transforms_sample = None

    connection_rows = []

    print("\n================ ROS FRAME AUDIT =", spec["run_id"], "================")
    print("bag:", bag)

    with AnyReader([bag]) as reader:
        for c in reader.connections:
            ext = getattr(c, "ext", None)

            connection_rows.append({
                "topic": c.topic,
                "msgtype": c.msgtype,
                "callerid": str(
                    getattr(ext, "callerid", "")
                ) if ext is not None else "",
                "latching": str(
                    getattr(ext, "latching", "")
                ) if ext is not None else "",
            })

        wanted = [
            c for c in reader.connections
            if (
                c.topic in set(FORCE_TOPICS.values())
                or c.topic in {"/tf", "/tf_static", STATE_TOPIC}
            )
        ]

        nread = 0

        for c, bag_ns, raw in reader.messages(connections=wanted):
            msg = reader.deserialize(raw, c.msgtype)
            nread += 1

            if c.topic in FORCE_TOPICS.values():
                leg = next(
                    k for k, v in FORCE_TOPICS.items()
                    if v == c.topic
                )

                fid = str(
                    getattr(
                        getattr(msg, "header", None),
                        "frame_id",
                        ""
                    ) or ""
                )
                force_frames[leg].add(fid)

            elif c.topic in {"/tf", "/tf_static"}:
                for tr in getattr(msg, "transforms", []):
                    parent = str(
                        getattr(
                            getattr(tr, "header", None),
                            "frame_id",
                            ""
                        ) or ""
                    )
                    child = str(
                        getattr(tr, "child_frame_id", "") or ""
                    )

                    if parent or child:
                        tf_pairs.add(
                            (c.topic, parent, child)
                        )

            elif c.topic == STATE_TOPIC:
                if state_sample is None:
                    state_sample = _describe_object(
                        msg,
                        max_depth=2,
                        max_seq=2
                    )

                    contacts = getattr(msg, "contacts", None)
                    if contacts is not None:
                        contacts_sample = _describe_object(
                            contacts,
                            max_depth=4,
                            max_seq=4
                        )

                    fts = getattr(msg, "frame_transforms", None)
                    if fts is not None:
                        transforms_sample = _describe_object(
                            fts,
                            max_depth=4,
                            max_seq=12
                        )

                # Collect every named frame-transform pair.
                for tr in getattr(msg, "frame_transforms", []):
                    parent = str(
                        getattr(
                            getattr(tr, "header", None),
                            "frame_id",
                            ""
                        ) or ""
                    )
                    child = str(
                        getattr(tr, "child_frame_id", "") or ""
                    )

                    if parent or child:
                        state_frame_pairs.add((parent, child))

            if nread >= max_messages:
                break

    conn = pd.DataFrame(connection_rows)

    # Only interesting connection rows.
    mask = conn.topic.str.contains(
        "force|state|tf|imu|pose|twist|camera",
        case=False,
        regex=True
    )

    display(conn[mask].reset_index(drop=True))

    print("\nForce topic header frames:")
    print({
        leg: sorted(v)
        for leg, v in force_frames.items()
    })

    print("\n/tf + /tf_static pairs:")
    for row in sorted(tf_pairs):
        print(" ", row)

    print("\nAnymalState.frame_transforms pairs:")
    for row in sorted(state_frame_pairs):
        print(" ", row)

    print("\nAnymalState.contacts SAMPLE:")
    print(json.dumps(contacts_sample, indent=2, default=str)[:12000])

    print("\nAnymalState.frame_transforms SAMPLE:")
    print(json.dumps(transforms_sample, indent=2, default=str)[:12000])

    return {
        "run_id": spec["run_id"],
        "bag": str(bag),
        "force_frames": {
            leg: sorted(v)
            for leg, v in force_frames.items()
        },
        "tf_pairs": [list(x) for x in sorted(tf_pairs)],
        "state_frame_pairs": [
            list(x) for x in sorted(state_frame_pairs)
        ],
        "contacts_sample": contacts_sample,
        "transforms_sample": transforms_sample,
    }


# ----------------------------------------------------------------------
# 3. Force orientation test using the REAL measured native vectors.
#
# We do NOT call this a proof of the frame.
#
# It tests whether the raw force coordinates behave like a base-frame
# forward/left/up convention under load:
#   - stance feet should carry positive vertical load
#   - swing load should be small
#   - sum(Fz) should be physically plausible
#
# Publisher/TF/source evidence must still agree before the contract is
# marked verified.
# ----------------------------------------------------------------------

def audit_force_static_consistency(spec):
    cache = locate_physics_cache(spec)
    if cache is None:
        raise FileNotFoundError(
            f"{spec['run_id']}: physics cache missing"
        )

    with np.load(cache, allow_pickle=False) as z:
        P = np.asarray(z["P"], np.float64)
        B = np.asarray(z["B"], np.float64)

        if "contact" in z.files:
            contact = np.asarray(z["contact"], np.float64) > 0.5
        else:
            # Only diagnostic fallback; does NOT create target labels.
            contact = np.linalg.norm(
                P[:, :, 9:12], axis=-1
            ) > CONTACT_ON_N

    F = P[:, :, 9:12]
    dq = P[:, :, 3:6]

    # Use independent base twist / joint motion to isolate calm intervals.
    calm = (
        (np.linalg.norm(B[:, 6:9], axis=1) < 0.25)
        &
        (np.linalg.norm(B[:, 9:12], axis=1) < 0.20)
        &
        (np.max(np.abs(dq), axis=(1, 2)) < 1.5)
        &
        (contact.sum(axis=1) >= 3)
    )

    if calm.sum() < 100:
        calm = contact.sum(axis=1) >= 3

    rows = []

    for perm, signs, R in SIGNED_ROTATIONS:
        Fc = F @ R.T

        summed = Fc.sum(axis=1)
        z = summed[calm, 2]
        xy = np.linalg.norm(summed[calm, :2], axis=1)

        stance_fz = Fc[:, :, 2][contact]
        swing_mag = np.linalg.norm(Fc, axis=-1)[~contact]

        vertical = float(np.median(z))
        horiz = float(np.median(xy))

        vertical_positive = float(
            np.mean(stance_fz > 0)
        ) if len(stance_fz) else np.nan

        horiz_ratio = float(
            horiz / max(abs(vertical), 1.0)
        )

        rows.append({
            "perm": tuple(int(x) for x in perm),
            "signs": tuple(int(x) for x in signs),
            "vertical_sum_median_N": vertical,
            "horizontal_sum_median_N": horiz,
            "horizontal_to_vertical": horiz_ratio,
            "stance_Fz_positive_fraction": vertical_positive,
            "swing_force_median_N": (
                float(np.median(swing_mag))
                if len(swing_mag) else np.nan
            ),
            "mass_equivalent_kg_if_static": (
                vertical / 9.80665
            ),
        })

    table = pd.DataFrame(rows)

    # We expect environment-on-robot support to be positive vertical
    # in a standard robot base x-forward,y-left,z-up representation.
    plausible = table[
        table["vertical_sum_median_N"] > 0
    ].copy()

    plausible["score"] = (
        plausible["horizontal_to_vertical"]
        + 0.5 * (
            1.0 -
            plausible["stance_Fz_positive_fraction"]
        )
    )

    plausible = plausible.sort_values("score")

    print("\n================ FORCE STATIC AUDIT =", spec["run_id"], "================")
    print("calm samples:", int(calm.sum()))

    display(plausible.head(8))

    identity = table[
        (table["perm"] == (0, 1, 2))
        &
        (table["signs"] == (1, 1, 1))
    ]

    print("\nIDENTITY / AS-RECORDED FORCE COORDINATES:")
    display(identity)

    return {
        "run_id": spec["run_id"],
        "calm_samples": int(calm.sum()),
        "identity": (
            identity.iloc[0].to_dict()
            if len(identity) else None
        ),
        "best_static_candidate": (
            plausible.iloc[0].to_dict()
            if len(plausible) else None
        )
    }


# ----------------------------------------------------------------------
# 4. Run against every raw run CURRENTLY present.
#
# We use what is already on disk first.  No extra 60 GB download here.
# ----------------------------------------------------------------------

present_specs = [
    s for s in RUN_REGISTRY
    if raw_run_path(s).exists()
    and locate_physics_cache(s) is not None
]

if not present_specs:
    raise RuntimeError(
        "No complete raw run + physics cache is currently present. "
        "Do not fabricate evidence."
    )

print("\nRaw runs available for G0 evidence:")
print([s["run_id"] for s in present_specs])


G0_SIGNAL_EVIDENCE = {}
G0_ROS_EVIDENCE = {}
G0_FORCE_EVIDENCE = {}

for spec in present_specs:
    try:
        G0_SIGNAL_EVIDENCE[spec["run_id"]] = (
            audit_signal_axes(spec)
        )
    except Exception as e:
        print(
            f"\nSIGNAL AUDIT FAILED {spec['run_id']}:",
            type(e).__name__, e
        )

    try:
        G0_ROS_EVIDENCE[spec["run_id"]] = (
            audit_ros_frame_evidence(spec)
        )
    except Exception as e:
        print(
            f"\nROS AUDIT FAILED {spec['run_id']}:",
            type(e).__name__, e
        )

    try:
        G0_FORCE_EVIDENCE[spec["run_id"]] = (
            audit_force_static_consistency(spec)
        )
    except Exception as e:
        print(
            f"\nFORCE AUDIT FAILED {spec['run_id']}:",
            type(e).__name__, e
        )


# ----------------------------------------------------------------------
# 5. Persistent audit artifact.
# This is EVIDENCE ONLY.  Contracts remain untouched.
# ----------------------------------------------------------------------

def _json_clean(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {
            str(k): _json_clean(v)
            for k, v in x.items()
        }
    if isinstance(x, (list, tuple)):
        return [_json_clean(v) for v in x]
    return x


G0_CONTINUATION = {
    "schema": "V10R_G0_CONTINUATION_EVIDENCE_V1",
    "created_utc": utc_now(),
    "model_run_ids_fixed": list(TWO_RUN_IDS),
    "signal_axis_evidence": G0_SIGNAL_EVIDENCE,
    "ros_frame_evidence": G0_ROS_EVIDENCE,
    "force_static_evidence": G0_FORCE_EVIDENCE,
    "contracts_modified": False,
    "scientific_rule": (
        "No contract may be approved solely from a numerical best fit. "
        "Frame provenance and independent physical consistency must agree."
    ),
}

G0_PATH = (
    AUDIT_ROOT /
    "G0_continuation_frame_signal_force_evidence.json"
)

atomic_write_json(
    G0_PATH,
    _json_clean(G0_CONTINUATION)
)

print("\n==============================================================")
print("G0 CONTINUATION AUDIT COMPLETE")
print("==============================================================")
print("saved:", G0_PATH)

fc, cc, sc, errors = load_and_validate_contracts(
    require_approved=False
)

print("\nCONTRACTS HAVE NOT BEEN MODIFIED.")
print("\nCURRENT BLOCKERS:")
print(json.dumps(errors, indent=2))

# %% notebook cell 55
# ======================================================================
# V10-R CONTINUATION G0C
# EXACT TWO-RUN RAW -> PREPARED PIPELINE
#
# SAFE TO APPEND NOW.
# IT WILL REFUSE TO EXECUTE UNTIL THE THREE SCIENTIFIC CONTRACTS PASS.
# ======================================================================

TWO_RUN_AUTO_DELETE = False   # KEEP FALSE on first successful pass


def validate_or_download_exact_two():
    rows = []

    for spec in TWO_SPECS:
        state = final_validate_raw(spec)

        if not state["pass"]:
            print(
                "\nDownloading ONLY:",
                spec["environment"],
                spec["run"]
            )
            final_download_one(spec)

        state = final_validate_raw(spec)

        if not state["pass"]:
            raise RuntimeError(
                f"{spec['run_id']}: download still incomplete:\n"
                + "\n".join(state["missing"])
            )

        rows.append(state)

    table = pd.DataFrame(rows)
    display(table)

    if len(table) != 2 or not table["pass"].all():
        raise RuntimeError(
            "The exact two raw runs are not both ready."
        )

    return table


def process_exact_two_after_verified_G0(
    auto_delete=False
):
    # --------------------------------------------------------------
    # ABSOLUTE SCIENTIFIC GATE
    # --------------------------------------------------------------
    fc, cc, sc, errors = load_and_validate_contracts(
        require_approved=False
    )

    remaining = {
        k: v for k, v in errors.items()
        if len(v)
    }

    if remaining:
        raise RuntimeError(
            "G0 IS STILL UNRESOLVED.\n"
            "No foot trajectory, touchdown impulse, geometry, "
            "haptic memory or training product will be generated.\n\n"
            + json.dumps(remaining, indent=2)
        )

    print("G0 contracts verified.")

    # --------------------------------------------------------------
    # Download EXACTLY the two requested model trajectories.
    # --------------------------------------------------------------
    raw_table = validate_or_download_exact_two()

    status_rows = []

    for number, spec in enumerate(TWO_SPECS, 1):
        rid = spec["run_id"]

        print("\n")
        print("=" * 76)
        print(
            f"TWO-RUN STRICT PREPARATION {number}/2:",
            rid
        )
        print("=" * 76)

        # 1. Native 400-Hz force timing/frame audit.
        native, force_table = run_force_audit(spec)

        # 2. TRUE multi-view RGB-D geometry from raw images/depth/poses.
        build_geometry_checkpoint(spec, cc)

        geometry_report = (
            validate_geometry_against_heldout_frames(spec)
        )

        if not geometry_report["pass"]:
            raise RuntimeError(
                f"{rid}: held-out geometry validation failed:\n"
                f"{geometry_report}"
            )

        # 3. TRUE prepared products.
        #
        # prepare_strict_run produces, from the real preprocessing:
        #   q, dq, tau
        #   corrected force
        #   body/sensor arrays
        #   FK foot trajectories
        #   COM
        #   Jacobians
        #   contact
        #   touchdown impulse
        #   desired wrench
        #   geometry
        #   causal haptic event memory
        #
        # No E->foot or zero-filled fake target is allowed.
        prep = prepare_strict_run(spec)

        # 4. Full checkpoint integrity validation.
        verified = verify_prepared_run(
            rid,
            raise_on_error=True
        )

        if verified["pass"] is not True:
            raise RuntimeError(
                f"{rid}: strict prepared validation failed"
            )

        status_rows.append({
            **verified,
            "environment": spec["environment"],
            "run": spec["run"],
            "raw_path": str(raw_run_path(spec)),
        })

        print("\nVERIFIED:", rid)

        # Only after verification is raw deletion even offered.
        token = deletion_token(rid)
        print("Safe deletion token:")
        print(token)

        if auto_delete:
            delete_one_verified_raw_run(
                rid,
                token
            )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    table = pd.DataFrame(status_rows)
    display(table)

    if (
        len(table) != 2
        or not table["pass"].all()
        or set(table["run_id"]) != set(TWO_RUN_IDS)
    ):
        raise RuntimeError(
            "Exact two-run preparation did not finish cleanly."
        )

    print("\n==============================================================")
    print("TRUE PREPROCESSING PRODUCTS READY FOR BOTH MODEL RUNS")
    print("==============================================================")

    print(
        "\nThese are diagnostic two-terrain data only. "
        "They are NOT the notebook's >=12-trajectory paper protocol."
    )

    return table


# --------------------------------------------------------------
# OFF BY DEFAULT.
# G0 resolver output must be interpreted first.
# --------------------------------------------------------------

RUN_TWO_AFTER_G0 = False

if RUN_TWO_AFTER_G0:
    TWO_RUN_PREPARATION = process_exact_two_after_verified_G0(
        auto_delete=TWO_RUN_AUTO_DELETE
    )
else:
    print(
        "Two-run execution is OFF.\n"
        "First inspect the G0B output above. "
        "Do not set RUN_TWO_AFTER_G0=True until the contracts are "
        "verified from real evidence."
    )

# %% notebook cell 56
# ======================================================================
# V10-R EMERGENCY FULL RESCUE
#
# RUN THIS BEFORE THE COLAB RUNTIME DIES.
#
# 1. Saves ephemeral /content raw evidence to Drive FIRST.
# 2. Packages everything needed to resume the current V10-R state.
# 3. Splits large archive into ~1.2 GB pieces.
# 4. Downloads every piece to your LOCAL COMPUTER.
#
# NO MODEL/CONTRACT/DATA MODIFICATION.
# ======================================================================

from pathlib import Path
from datetime import datetime, timezone
import os
import sys
import json
import shutil
import tarfile
import subprocess
import time

from google.colab import files


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

RAW_ROOT = Path(
    globals().get(
        "RAW_ROOT",
        "/content/tartanground_v10"
    )
)

PROJECT_ROOT = Path(
    globals().get(
        "PROJECT_ROOT",
        "/content/drive/MyDrive/UGP_V7_FULL4B"
    )
)

STRICT_ROOT = Path(
    globals().get(
        "STRICT_ROOT",
        PROJECT_ROOT / "V10R_STRICT_STAGED"
    )
)

CACHE_ROOT = PROJECT_ROOT / "cache"

EXTERNAL_ROOT = Path("/content/v10r_external")

timestamp = datetime.now(
    timezone.utc
).strftime("%Y%m%d_%H%M%S")

DRIVE_RESCUE = (
    PROJECT_ROOT
    / "V10R_RUNTIME_RESCUE"
    / timestamp
)

DRIVE_RESCUE.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 72)
print("STEP 1 — SAVE EPHEMERAL RAW DATA TO DRIVE")
print("=" * 72)


# ----------------------------------------------------------------------
# 1. CRITICAL: copy /content raw data to Drive.
#
# If the runtime dies after THIS finishes, the downloaded ROS bags/IMU
# data are no longer lost.
# ----------------------------------------------------------------------

RAW_DRIVE_BACKUP = (
    DRIVE_RESCUE
    / "tartanground_v10"
)

if RAW_ROOT.exists():
    print(
        "Copying:",
        RAW_ROOT
    )
    print(
        "     to:",
        RAW_DRIVE_BACKUP
    )

    shutil.copytree(
        RAW_ROOT,
        RAW_DRIVE_BACKUP,
        dirs_exist_ok=True,
        copy_function=shutil.copy2,
    )

    print(
        "RAW DATA DRIVE BACKUP COMPLETE."
    )

else:
    print(
        "WARNING: RAW_ROOT does not exist:",
        RAW_ROOT
    )


# ----------------------------------------------------------------------
# 2. Save runtime/state metadata.
# ----------------------------------------------------------------------

state = {
    "created_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "python":
        sys.version,

    "RAW_ROOT":
        str(RAW_ROOT),

    "PROJECT_ROOT":
        str(PROJECT_ROOT),

    "STRICT_ROOT":
        str(STRICT_ROOT),

    "raw_drive_backup":
        str(RAW_DRIVE_BACKUP),

    "run_registry":
        globals().get(
            "RUN_REGISTRY",
            None
        ),

    "two_run_ids":
        list(
            globals().get(
                "TWO_RUN_IDS",
                []
            )
        ),

    "current_contract_errors":
        globals().get(
            "bootstrap_contract_errors",
            None
        ),
}


with (
    DRIVE_RESCUE
    / "runtime_state.json"
).open("w") as f:

    json.dump(
        state,
        f,
        indent=2,
        default=str,
    )


# pip environment
with (
    DRIVE_RESCUE
    / "pip_freeze.txt"
).open("w") as f:

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "freeze",
        ],
        stdout=f,
        stderr=subprocess.STDOUT,
        text=True,
    )


# Save live G0 evidence object too, if still in RAM.
if "G0_CONTINUATION" in globals():

    with (
        DRIVE_RESCUE
        / "G0_CONTinuation_live_copy.json"
    ).open("w") as f:

        json.dump(
            G0_CONTINUATION,
            f,
            indent=2,
            default=lambda x:
                x.tolist()
                if hasattr(x, "tolist")
                else str(x),
        )


# ----------------------------------------------------------------------
# 3. Collect the important sources.
#
# We include:
#   - raw evidence
#   - ALL existing physics/frame caches
#   - STRICT contracts/audits/prepared/results/models/etc.
#   - external robot-description files if any
#   - rescue metadata
# ----------------------------------------------------------------------

sources = []


def add_source(
    path,
    archive_name
):
    path = Path(path)

    if path.exists():
        sources.append(
            (
                path,
                archive_name
            )
        )

        print(
            "INCLUDE:",
            path
        )

    else:
        print(
            "MISSING/SKIP:",
            path
        )


# Use LOCAL raw copy because it is faster to read while creating tar.
add_source(
    RAW_ROOT,
    "raw/tartanground_v10"
)

# Entire cache directory contains the valuable physics .npz files
# and any frame manifests.
add_source(
    CACHE_ROOT,
    "project/cache"
)

# Everything produced by the strict V10-R pipeline.
add_source(
    STRICT_ROOT,
    "project/V10R_STRICT_STAGED"
)

# Anything such as cloned/downloaded robot description.
add_source(
    EXTERNAL_ROOT,
    "runtime/v10r_external"
)

# Metadata generated above.
add_source(
    DRIVE_RESCUE,
    "resume_metadata"
)


# ----------------------------------------------------------------------
# 4. Build an UNCOMPRESSED tar.
#
# Uncompressed is deliberate:
#   NPZ/rosbag data often compress poorly, and your runtime is expiring.
#   This is much faster and preserves exact bytes.
# ----------------------------------------------------------------------

ARCHIVE = Path(
    f"/content/"
    f"V10R_RESCUE_{timestamp}.tar"
)

print("\n" + "=" * 72)
print("STEP 2 — BUILD LOCAL RESCUE ARCHIVE")
print("=" * 72)

print(
    "Archive:",
    ARCHIVE
)


with tarfile.open(
    ARCHIVE,
    mode="w"
) as tar:

    for src, arcname in sources:

        print(
            "\nAdding:",
            src
        )

        tar.add(
            src,
            arcname=arcname,
            recursive=True,
        )


archive_bytes = (
    ARCHIVE.stat().st_size
)

archive_gb = (
    archive_bytes
    / 1024**3
)

print(
    f"\nArchive complete: "
    f"{archive_gb:.3f} GB"
)


# ----------------------------------------------------------------------
# 5. Write SHA-256 checksum.
# ----------------------------------------------------------------------

print(
    "\nComputing archive SHA-256..."
)

import hashlib

h = hashlib.sha256()

with ARCHIVE.open("rb") as f:

    while True:

        block = f.read(
            16 * 1024 * 1024
        )

        if not block:
            break

        h.update(block)


sha256 = h.hexdigest()

checksum_file = Path(
    str(ARCHIVE)
    + ".sha256.txt"
)

checksum_file.write_text(
    f"{sha256}  {ARCHIVE.name}\n"
)

print(
    "SHA256:",
    sha256
)


# ----------------------------------------------------------------------
# 6. Split if archive is large.
#
# ~1200 MB is much safer for Colab/browser downloading than one
# multi-gigabyte response.
# ----------------------------------------------------------------------

PART_SIZE_MB = 1200

part_prefix = Path(
    str(ARCHIVE)
    + ".part-"
)


if archive_bytes > (
    PART_SIZE_MB
    * 1024**2
):

    print("\n" + "=" * 72)
    print("STEP 3 — SPLITTING ARCHIVE")
    print("=" * 72)

    subprocess.run(
        [
            "split",
            "-b",
            f"{PART_SIZE_MB}M",
            "-d",
            "-a",
            "3",
            str(ARCHIVE),
            str(part_prefix),
        ],
        check=True,
    )

    parts = sorted(
        ARCHIVE.parent.glob(
            ARCHIVE.name
            + ".part-*"
        )
    )

else:

    parts = [
        ARCHIVE
    ]


print("\nFiles to download:")

for p in parts:

    print(
        f"  {p.name}  "
        f"{p.stat().st_size / 1024**3:.3f} GB"
    )


# ----------------------------------------------------------------------
# 7. Download checksum first.
# ----------------------------------------------------------------------

print("\n" + "=" * 72)
print("STEP 4 — START LOCAL DOWNLOADS")
print("=" * 72)

print(
    "\nDownloading checksum..."
)

files.download(
    str(checksum_file)
)

time.sleep(1)


# ----------------------------------------------------------------------
# 8. Trigger download of every archive/part.
#
# Chrome may ask:
#     "Allow multiple downloads?"
# Choose ALLOW.
# ----------------------------------------------------------------------

for i, p in enumerate(
    parts,
    1
):

    print(
        f"\nStarting download "
        f"{i}/{len(parts)}:",
        p.name
    )

    files.download(
        str(p)
    )

    time.sleep(2)


# ----------------------------------------------------------------------
# 9. Print recovery instructions.
# ----------------------------------------------------------------------

print("\n" + "=" * 72)
print("RESCUE COMPLETE / DOWNLOADS STARTED")
print("=" * 72)

print(
    "\nPersistent emergency Drive backup:"
)

print(
    DRIVE_RESCUE
)

if len(parts) == 1:

    print(
        "\nAfter downloading, extract:"
    )

    print(
        ARCHIVE.name
    )

else:

    print(
        "\nThe TAR was split into "
        f"{len(parts)} pieces."
    )

    print(
        "\nOn your LOCAL COMPUTER, "
        "put all .part-* files in one folder."
    )

    print(
        "\nLinux/macOS rejoin:"
    )

    print(
        f"cat "
        f"{ARCHIVE.name}.part-* "
        f"> {ARCHIVE.name}"
    )

    print(
        "\nWindows PowerShell rejoin:"
    )

    print(
        "$parts = Get-ChildItem "
        f"'{ARCHIVE.name}.part-*' "
        "| Sort-Object Name"
    )

    print(
        f"$out = "
        "[System.IO.File]::Create("
        f"'{ARCHIVE.name}')"
    )

    print(
        "foreach ($p in $parts) { "
        "$b=[System.IO.File]::ReadAllBytes($p.FullName); "
        "$out.Write($b,0,$b.Length) }"
    )

    print(
        "$out.Close()"
    )

print(
    "\nThen extract the resulting .tar."
)
