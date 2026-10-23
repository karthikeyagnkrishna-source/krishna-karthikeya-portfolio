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

# %% notebook cell 31
# --------------------------- run controls ---------------------------
# SHA-256 of the supplied ground-truth notebook; this is filled by the builder
# and is checked again by the final status cell.
CANONICAL_SOURCE_SHA256 = 'b491b669b0421d14c0841322f23c00a21a057684f5608f2212e58401c34b68c2'
# The default is safe to execute top-to-bottom on a fresh Colab runtime.
RUN_REAL = False
RUN_RESTORE_TARTANAIR = False
RUN_EXACT_STATE_CEILING = False
RUN_TRAINING = False
RUN_EVALUATION = False
RUN_PLANNER = False
RUN_PLOTS = False

# Keep the canonical CFG unchanged. These are orchestration controls only.
REAL_PROJECT_ROOT = Path(str(globals().get('PROJECT_ROOT', CFG.project_root)))
RAW_ROOT = Path('/content/tartanground_v10')
IMPLEMENTATION_ROOT = REAL_PROJECT_ROOT / 'v10r_canonical_real_outputs'
IMPLEMENTATION_ROOT.mkdir(parents=True, exist_ok=True)
TRAIN_EPOCHS = 40
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
MAX_ORIGINS_PER_RUN = None       # set an integer for a short debugging run
MAX_CEILING_TRAIN = 120_000
GEOM_FRAME_STRIDE = 2
GEOM_PIXEL_STRIDE = 4
GEOM_MAX_POINTS = 500_000
FORCE_TOPIC_BY_LEG = {
    'FL': '/state_estimator/contact_force_lf_foot',
    'FR': '/state_estimator/contact_force_rf_foot',
    'RL': '/state_estimator/contact_force_lh_foot',
    'RR': '/state_estimator/contact_force_rh_foot',
}
LEG_NAMES = ('FL', 'FR', 'RL', 'RR')
CAMERA_NAMES = ('lcam_front', 'lcam_back', 'lcam_left', 'lcam_right',
                 'lcam_top', 'lcam_bottom')
RUN_SPECS = [
    {'run_id': 'forest_P2000', 'terrain': 'forest', 'environment': 'ForestEnv', 'run': 'P2000'},
    {'run_id': 'swamp_marsh_P2023', 'terrain': 'swamp_marsh', 'environment': 'GreatMarsh', 'run': 'P2023'},
    {'run_id': 'rough_uneven_P2000', 'terrain': 'rough_uneven', 'environment': 'SeasonalForestWinter', 'run': 'P2000'},
    {'run_id': 'wet_uneven_proxy_P2000', 'terrain': 'wet_uneven_proxy', 'environment': 'OldTownFall', 'run': 'P2000'},
]
print('RUN_REAL =', RUN_REAL, '| project root =', REAL_PROJECT_ROOT)

# %% notebook cell 32
# --------------------------- cache discovery ---------------------------
def _first_existing(root, names):
    root = Path(root)
    for name in names:
        p = root / name
        if p.exists():
            return p
    # Keep discovery bounded to the project root; do not scan the whole drive.
    for name in names:
        hits = sorted(root.rglob(name))
        if hits:
            return hits[0]
    return None


def _npz_dict(path):
    if path is None:
        return {}
    with np.load(path, allow_pickle=True) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def _locate_physics(spec):
    names = [
        f"{spec['environment']}_{spec['run']}_physics.npz",
        f"{spec['run_id']}_physics.npz",
        f"{spec['run']}_physics.npz",
    ]
    p = _first_existing(REAL_PROJECT_ROOT, names)
    if p is None:
        raise FileNotFoundError(
            f"physics cache not found for {spec['run_id']}; expected one of {names} under {REAL_PROJECT_ROOT}"
        )
    return p


def _locate_manifest(spec):
    return _first_existing(REAL_PROJECT_ROOT, [
        f"{spec['environment']}_{spec['run']}_frame_manifest.npz",
        f"{spec['run_id']}_frame_manifest.npz",
        f"{spec['run']}_frame_manifest.npz",
    ])


def _locate_geometry(spec):
    return _first_existing(REAL_PROJECT_ROOT, [
        f"{spec['environment']}_{spec['run']}_geometry.npz",
        f"{spec['run_id']}_geometry.npz",
        f"{spec['run']}_geometry.npz",
    ])


def _locate_urdf():
    explicit = globals().get('ANYMAL_URDF', None)
    if explicit is not None and Path(explicit).exists():
        return Path(explicit)
    candidates = []
    for root in (RAW_ROOT, REAL_PROJECT_ROOT):
        if root.exists():
            candidates.extend(sorted(root.rglob('*.urdf')))
    # Prefer files whose name identifies ANYmal; never silently select a random
    # robot description when a named one is available.
    named = [p for p in candidates if 'anymal' in p.name.lower() or 'anymal' in str(p).lower()]
    if named:
        return named[0]
    return candidates[0] if candidates else None


def restore_anymal_runs(target_root=RAW_ROOT):
    """Optional explicit TartanAir restore; this function is never called implicitly."""
    import tartanair as ta
    target_root = Path(target_root)
    target_root.mkdir(parents=True, exist_ok=True)
    ta.init(str(target_root))
    for spec in RUN_SPECS:
        ta.download_ground(
            env=[spec['environment']], version=['anymal'], traj=[spec['run']],
            modality=['imu', 'rosbag', 'image', 'depth', 'meta'],
            camera_name=list(CAMERA_NAMES), unzip=True,
        )


REAL_RUNS = {}
REAL_AUX = {}
if RUN_RESTORE_TARTANAIR:
    restore_anymal_runs()
if RUN_REAL:
    for spec in RUN_SPECS:
        physics_path = _locate_physics(spec)
        rec = load_physics_npz(physics_path, spec['terrain'],
                               'native_rosbag_frame_pending_audit', 'base_link')
        # RunRecord is deliberately the canonical class; only its identifiers
        # are made human-readable for blocked group splitting.
        rec.run_id = spec['run_id']
        rec.trajectory_id = spec['run_id']
        manifest_path = _locate_manifest(spec)
        geometry_path = _locate_geometry(spec)
        REAL_RUNS[spec['run_id']] = rec
        REAL_AUX[spec['run_id']] = {
            'spec': spec,
            'physics_path': physics_path,
            'manifest_path': manifest_path,
            'manifest': _npz_dict(manifest_path),
            'geometry_path': geometry_path,
            'run_root': RAW_ROOT / spec['environment'] / 'Data_anymal' / spec['run'],
        }
    print('loaded canonical physics runs:', list(REAL_RUNS))
else:
    print('Real-data cells are gated. Set RUN_REAL=True and rerun from this cell.')

# %% notebook cell 33
# --------------------------- native ROS force audit ---------------------------
def _stamp_seconds(header):
    if header is None or not hasattr(header, 'stamp'):
        return None
    stamp = header.stamp
    sec = getattr(stamp, 'sec', getattr(stamp, 'secs', 0))
    nsec = getattr(stamp, 'nanosec', getattr(stamp, 'nsecs', 0))
    try:
        return float(sec) + 1e-9 * float(nsec)
    except Exception:
        return None


def _force_vector(msg):
    obj = msg.wrench if hasattr(msg, 'wrench') else msg
    obj = obj.force if hasattr(obj, 'force') else obj
    vals = [getattr(obj, k, None) for k in ('x', 'y', 'z')]
    if any(v is None for v in vals):
        raise ValueError(f'force message has no x/y/z vector: {type(msg)}')
    return np.asarray(vals, np.float64)


def read_native_force_bag(bag_path):
    """Read only the four named native force topics and retain both clocks."""
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as exc:
        raise ImportError('Install rosbags in Colab before the native force audit.') from exc
    bag_path = Path(bag_path)
    if not bag_path.exists():
        raise FileNotFoundError(f'Raw ROS bag required for native force audit: {bag_path}')
    streams = {leg: {'time': [], 'force': [], 'frame_id': [], 'header_time': []}
               for leg in LEG_NAMES}
    topic_to_leg = {topic: leg for leg, topic in FORCE_TOPIC_BY_LEG.items()}
    with AnyReader([bag_path]) as reader:
        connections = [c for c in reader.connections if c.topic in topic_to_leg]
        missing = set(topic_to_leg) - {c.topic for c in connections}
        if missing:
            raise ValueError(f'{bag_path}: missing native force topics {sorted(missing)}')
        for connection, bag_ns, raw in reader.messages(connections=connections):
            msg = reader.deserialize(raw, connection.msgtype)
            leg = topic_to_leg[connection.topic]
            streams[leg]['time'].append(float(bag_ns) * 1e-9)
            streams[leg]['force'].append(_force_vector(msg))
            header = getattr(msg, 'header', None)
            streams[leg]['frame_id'].append(str(getattr(header, 'frame_id', '') or ''))
            streams[leg]['header_time'].append(_stamp_seconds(header))
    for leg, s in streams.items():
        if not s['time']:
            raise ValueError(f'{bag_path}: no messages on {FORCE_TOPIC_BY_LEG[leg]}')
        order = np.argsort(np.asarray(s['time']))
        for k in ('time', 'force', 'frame_id', 'header_time'):
            s[k] = np.asarray(s[k], dtype=object if k in ('frame_id', 'header_time') else None)[order]
        s['time'] = s['time'].astype(float)
        s['force'] = np.stack(s['force']).astype(np.float64)
    return streams


def classify_force_frame(frame_ids):
    labels = []
    for raw in np.asarray(frame_ids, dtype=object):
        s = str(raw).lower().lstrip('/')
        if not s:
            labels.append('unknown')
        elif any(x in s for x in ('world', 'odom', 'map', 'ned')):
            labels.append('world')
        elif any(x in s for x in ('camera', 'cam', 'lcam', 'rgb', 'depth')):
            labels.append('camera')
        elif any(x in s for x in ('base', 'body', 'imu', 'trunk')):
            labels.append('base')
        elif any(x in s for x in ('foot', 'lf_', 'rf_', 'lh_', 'rh_', 'fl_', 'fr_', 'rl_', 'rr_')):
            labels.append('foot')
        else:
            labels.append('unknown')
    known = sorted(set(labels) - {'unknown'})
    if 'unknown' in labels:
        raise ValueError(f'unknown native force frame IDs: {sorted(set(np.asarray(frame_ids, dtype=object)))}')
    return known[0] if len(known) == 1 else 'mixed:' + ','.join(known)


def native_force_audit_table(streams):
    rows = []
    for leg in LEG_NAMES:
        s = streams[leg]
        t = s['time'].astype(float)
        dt = np.diff(t)
        rows.append({
            'leg': leg, 'n_native': len(t), 'frame_class': classify_force_frame(s['frame_id']),
            'frame_ids': '|'.join(sorted(set(map(str, s['frame_id'])))),
            'dt_med_ms': 1000 * float(np.median(dt)) if len(dt) else np.nan,
            'dt_p99_ms': 1000 * float(np.quantile(dt, .99)) if len(dt) else np.nan,
            'nonpositive_dt': int((dt <= 0).sum()),
            'header_clock_available': int(np.isfinite(s['header_time'].astype(float)).sum()),
        })
    return pd.DataFrame(rows)


def align_native_force(streams, t_physics, native_hz=CFG.native_force_hz,
                       target_hz=CFG.sample_hz):
    """Uniformly resample the native clock, call the canonical anti-aliaser,
    then interpolate its 100-Hz output onto the physics timestamps."""
    t_physics = np.asarray(t_physics, float)
    starts = [float(streams[l]['time'][0]) for l in LEG_NAMES]
    ends = [float(streams[l]['time'][-1]) for l in LEG_NAMES]
    duration = min(ends) - max(starts)
    if duration <= 0:
        raise ValueError('native force streams do not overlap in time')
    origin = max(starts)
    # The cache clock is relative to the cropped trajectory; the raw bag clock
    # is absolute. This explicit alignment is logged, never hidden.
    native_t = np.arange(0., duration + .5 / native_hz, 1. / native_hz)
    raw = np.zeros((len(native_t), 4, 3), np.float64)
    for li, leg in enumerate(LEG_NAMES):
        s = streams[leg]
        tl = s['time'].astype(float) - origin
        for a in range(3):
            raw[:, li, a] = np.interp(native_t, tl, s['force'][:, a])
    filtered = bandlimit_then_decimate(raw, native_hz=native_hz, target_hz=target_hz)
    filtered_t = native_t[::native_hz // target_hz][:len(filtered)]
    # Map the relative native stream to the physics clock by duration and start.
    t_local = t_physics - t_physics[0]
    aligned = np.empty((len(t_physics), 4, 3), np.float64)
    for li in range(4):
        for a in range(3):
            aligned[:, li, a] = np.interp(t_local, filtered_t, filtered[:, li, a])
    return {
        'raw_native_t': native_t + origin,
        'raw_native_force': raw,
        'filtered_t': filtered_t + t_physics[0],
        'filtered_force': filtered,
        'aligned_force': aligned,
        'alignment_mode': 'bag_relative_duration_to_cropped_physics_clock',
    }

# %% notebook cell 34
# --------------------------- official URDF and camera calibration ---------------------------
class PinocchioAnyMalKinematics:
    """Minimal adapter; all kinematic numbers come from the selected URDF."""
    def __init__(self, urdf_path):
        try:
            import pinocchio as pin
        except ImportError as exc:
            raise ImportError('Install pinocchio or provide cached official FK before a real run.') from exc
        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        names = {i: str(n) for i, n in enumerate(self.model.names)}
        self.joint_ids = {}
        urdf_tokens = {'FL': ('FL', 'LF'), 'FR': ('FR', 'RF'),
                       'RL': ('RL', 'LH'), 'RR': ('RR', 'RH')}
        for leg in LEG_NAMES:
            tokens = urdf_tokens[leg]
            ids = []
            for jid, name in names.items():
                u = name.upper().replace('-', '_')
                if any(token in u for token in tokens) and jid > 0 and self.model.joints[jid].nq >= 1:
                    ids.append(jid)
            ids = sorted(ids, key=lambda j: names[j])
            if len(ids) != 3:
                raise ValueError(f'URDF joint mapping for {leg} has {len(ids)} joints: {[names[j] for j in ids]}')
            self.joint_ids[leg] = ids
        frame_names = {i: str(f.name) for i, f in enumerate(self.model.frames)}
        self.frame_ids = {}
        for leg in LEG_NAMES:
            tokens = urdf_tokens[leg]
            candidates = [i for i, name in frame_names.items()
                          if any(token in name.upper().replace('-', '_') for token in tokens)
                          and 'foot' in name.lower()]
            if not candidates:
                candidates = [i for i, name in frame_names.items()
                              if any(token in name.upper().replace('-', '_') for token in tokens)
                              and 'end' in name.lower()]
            if not candidates:
                raise ValueError(f'URDF has no foot/end frame for {leg}')
            self.frame_ids[leg] = candidates[0]

    def foot_transforms(self, q_leg):
        q_leg = np.asarray(q_leg, float).reshape(4, 3)
        q = np.zeros(self.model.nq, dtype=float)
        for li, leg in enumerate(LEG_NAMES):
            for k, jid in enumerate(self.joint_ids[leg]):
                iq = self.model.joints[jid].idx_q
                q[iq] = q_leg[li, k]
        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)
        R = np.stack([self.data.oMf[self.frame_ids[leg]].rotation for leg in LEG_NAMES])
        p = np.stack([self.data.oMf[self.frame_ids[leg]].translation for leg in LEG_NAMES])
        return R.astype(np.float32), p.astype(np.float32)


def _load_cached_fk(aux):
    path = _first_existing(Path(aux['physics_path']).parent, [
        f"{aux['spec']['run_id']}_fk.npz", f"{aux['spec']['run']}_fk.npz",
    ])
    if path is None:
        return None
    d = _npz_dict(path)
    if not {'foot_rot', 'foot_pos'}.issubset(d):
        raise ValueError(f'{path} must contain foot_rot and foot_pos')
    class CachedFK:
        def foot_transforms(self, q_leg):
            n = len(np.asarray(q_leg).reshape(-1, 4, 3))
            R = np.asarray(d['foot_rot'])
            p = np.asarray(d['foot_pos'])
            if R.ndim == 4:
                R = np.repeat(R[None], n, axis=0)
            if p.ndim == 3:
                p = np.repeat(p[None], n, axis=0)
            return R[0].astype(np.float32), p[0].astype(np.float32)
    return CachedFK()


def _manifest_pose_arrays(manifest):
    pose_key = next((k for k in ('poses', 'pose', 'camera_poses') if k in manifest), None)
    time_key = next((k for k in ('camera_times', 'pose_times', 'times') if k in manifest), None)
    if pose_key is None or time_key is None:
        return None, None
    poses = np.asarray(manifest[pose_key], float)
    times = np.asarray(manifest[time_key], float).reshape(-1)
    if poses.ndim != 2 or poses.shape[1] < 7 or len(poses) != len(times):
        raise ValueError(f'invalid pose manifest shapes: {poses.shape}, {times.shape}')
    return times, poses[:, :7]


def _align_external_clock(times, reference):
    """Put an external relative/absolute clock on the reference trajectory clock."""
    times = np.asarray(times, float).reshape(-1)
    reference = np.asarray(reference, float).reshape(-1)
    if len(times) == 0 or len(reference) == 0:
        return times
    if abs(times[0] - reference[0]) > 5.:
        return times - times[0] + reference[0]
    return times


def _pose_rotations(poses):
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat(np.asarray(poses)[:, 3:7]).as_matrix()


def calibrate_base_camera_from_stance(run, manifest, fk):
    """Estimate T_CB from stance-foot stationarity and then freeze it."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    pose_t, poses = _manifest_pose_arrays(manifest)
    if pose_t is None:
        raise ValueError('camera pose/timestamp manifest is required for extrinsic calibration')
    pose_t = _align_external_clock(pose_t, run.data['time'])
    R_WC = _pose_rotations(poses)
    t_WC = poses[:, :3]
    t_phys = np.asarray(run.data['time'], float)
    ids = np.clip(np.searchsorted(t_phys, pose_t), 0, len(t_phys) - 1)
    q = np.asarray(run.data['P'])[ids, :, Q_SLICE]
    foot_R = []; foot_p = []
    for qk in q:
        rk, pk = fk.foot_transforms(qk)
        foot_R.append(rk); foot_p.append(pk)
    foot_p = np.asarray(foot_p)
    contact = np.asarray(run.data['contact'])[ids] > .5
    pairs = []
    for li in range(4):
        good = contact[1:, li] & contact[:-1, li]
        for j in np.where(good)[0]:
            pairs.append((j, j + 1, li))
    if len(pairs) < 20:
        raise ValueError(f'not enough stance pairs ({len(pairs)}) to calibrate T_CB')
    R0 = np.diag([1., -1., -1.])
    x0 = np.r_[Rotation.from_matrix(R0).as_rotvec(), np.zeros(3)]

    def residual(x):
        R_CB = Rotation.from_rotvec(x[:3]).as_matrix()
        t_CB = x[3:]
        out = []
        for j0, j1, li in pairs:
            p0 = t_WC[j0] + R_WC[j0] @ (t_CB + R_CB @ foot_p[j0, li])
            p1 = t_WC[j1] + R_WC[j1] @ (t_CB + R_CB @ foot_p[j1, li])
            out.extend(p1 - p0)
        return np.asarray(out)

    fit = least_squares(residual, x0, loss='soft_l1', f_scale=.01, max_nfev=300)
    R_CB = Rotation.from_rotvec(fit.x[:3]).as_matrix()
    t_CB = fit.x[3:]
    rms = float(np.sqrt(np.mean(np.square(residual(fit.x)))))
    if not np.isfinite(rms) or rms > .05:
        raise ValueError(f'camera/base calibration residual {rms:.4f} m exceeds 5 cm')
    return {'R_CB': R_CB.astype(np.float32), 't_CB': t_CB.astype(np.float32),
            'stance_rms_m': rms, 'source': 'stance_foot_stationarity'}


def _poses_at_physics(manifest, t_phys):
    pose_t, poses = _manifest_pose_arrays(manifest)
    if pose_t is None:
        raise ValueError('camera pose/timestamp manifest is required')
    pose_t = _align_external_clock(pose_t, t_phys)
    R = _pose_rotations(poses); t = poses[:, :3]
    idx = np.clip(np.searchsorted(pose_t, t_phys), 0, len(pose_t) - 1)
    # nearest-neighbour timestamp association is explicit and auditable.
    left = np.maximum(idx - 1, 0)
    use_left = np.abs(pose_t[left] - t_phys) < np.abs(pose_t[idx] - t_phys)
    idx[use_left] = left[use_left]
    return R[idx], t[idx], pose_t[idx]


def build_foot_world(run, manifest, fk, calibration, return_origin=False):
    R_WC, t_WC, _ = _poses_at_physics(manifest, np.asarray(run.data['time'], float))
    q = np.asarray(run.data['P'])[..., Q_SLICE]
    p_B = np.stack([fk.foot_transforms(qi)[1] for qi in q])
    R_CB, t_CB = calibration['R_CB'], calibration['t_CB']
    p_C = t_CB[None, None, :] + np.einsum('ab,nlb->nla', R_CB, p_B)
    p_W = t_WC[:, None, :] + np.einsum('nij,nlj->nli', R_WC, p_C)
    p_W0 = p_W[0, 0].astype(np.float32)
    rel = (p_W - p_W0[None, None, :]).astype(np.float32)
    return (rel, p_W0) if return_origin else rel


def _urdf_total_mass(urdf_path):
    import xml.etree.ElementTree as ET
    root = ET.parse(str(urdf_path)).getroot()
    masses = []
    for node in root.findall('.//inertial/mass'):
        if node.get('value') is not None:
            masses.append(float(node.get('value')))
    if not masses or not np.isfinite(masses).all() or sum(masses) <= 0:
        raise ValueError(f'no positive inertial mass found in {urdf_path}')
    return float(sum(masses))


def build_causal_desired_wrench(run, manifest, calibration, mass):
    """Deployable static gravity request; no future measured state is used."""
    R_WC, _, _ = _poses_at_physics(manifest, np.asarray(run.data['time'], float))
    R_WB = np.einsum('nij,jk->nik', R_WC, np.asarray(calibration['R_CB']))
    # TartanAir uses NED world convention; gravity is +z in that frame.
    g_W = np.array([0., 0., 9.81], dtype=float)
    g_B = np.einsum('nij,j->ni', np.swapaxes(R_WB, 1, 2), g_W)
    out = np.zeros((len(g_B), 6), dtype=np.float32)
    out[:, :3] = mass * g_B
    return out

# %% notebook cell 35
# --------------------------- force-frame conversion ---------------------------
def convert_force_to_base(aligned_force, streams, run, manifest, fk, calibration):
    aligned_force = np.asarray(aligned_force, float)
    q = np.asarray(run.data['P'])[..., Q_SLICE]
    foot_R = np.stack([fk.foot_transforms(qi)[0] for qi in q])
    R_WC, _, _ = _poses_at_physics(manifest, np.asarray(run.data['time'], float))
    R_CB = np.asarray(calibration['R_CB'], float)
    R_WB = np.einsum('nij,jk->nik', R_WC, R_CB)
    classes = [classify_force_frame(streams[leg]['frame_id']) for leg in LEG_NAMES]
    out = np.zeros_like(aligned_force)
    for li, cls in enumerate(classes):
        if cls == 'base':
            out[:, li] = aligned_force[:, li]
        elif cls == 'foot':
            # R_BF maps foot-frame column vectors into base-frame columns.
            out[:, li] = np.einsum('ni,nji->nj', aligned_force[:, li], foot_R[:, li])
        elif cls == 'camera':
            # R_CB maps base columns into camera columns; row f_C @ R_CB is f_B.
            out[:, li] = np.einsum('ni,nij->nj', aligned_force[:, li],
                                   np.broadcast_to(R_CB, (len(out), 3, 3)))
        elif cls == 'world':
            # R_WB maps base columns into world columns; row f_W @ R_WB is f_B.
            out[:, li] = np.einsum('ni,nij->nj', aligned_force[:, li], R_WB)
        else:
            raise ValueError(f'unsupported native force frame class: {cls}')
    return out.astype(np.float32), classes


def _touchdown_impulse_array(run, force_base):
    td = touchdown_targets(run.data['time'], force_base, run.data['contact'])
    impulse = np.zeros((len(run.data['time']), 4, 3), dtype=np.float32)
    for row in td.itertuples(index=False):
        impulse[int(row.index), int(row.leg)] = np.asarray([row.Jx, row.Jy, row.Jz], np.float32)
    return impulse, td

# %% notebook cell 36
# --------------------------- RGB-D causal patch construction ---------------------------
def _read_depth(path):
    path = Path(path)
    if path.suffix.lower() == '.npy':
        arr = np.load(path)
    else:
        try:
            import cv2
            arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        except ImportError as exc:
            raise ImportError('Install opencv-python for RGB-D geometry extraction.') from exc
    if arr is None:
        raise ValueError(f'cannot read depth {path}')
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] == 4 and arr.dtype == np.uint8:
        # TartanAir packed float32 depth RGBA.
        arr = arr.view('<f4').reshape(arr.shape[:2])
    elif np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / (1000. if arr.dtype.itemsize <= 2 else 1.)
    return arr.astype(np.float32)


def _read_rgb(path, shape):
    if path is None or not Path(path).exists():
        return np.zeros((*shape, 3), dtype=np.float32)
    try:
        import cv2
        im = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    except Exception:
        from PIL import Image
        im = np.asarray(Image.open(path).convert('RGB'))
    if tuple(im.shape[:2]) != tuple(shape):
        try:
            import cv2
            im = cv2.resize(im, (int(shape[1]), int(shape[0])), interpolation=cv2.INTER_AREA)
        except Exception:
            from PIL import Image
            im = np.asarray(Image.fromarray(im).resize((int(shape[1]), int(shape[0]))))
    return im.astype(np.float32) / 255.


def _resolve_frame_file(run_root, cam, frame_id, modality):
    run_root = Path(run_root)
    stem = str(frame_id)
    roots = [run_root / modality / cam, run_root / cam / modality,
             run_root / f'{modality}_{cam}', run_root / cam]
    suffixes = ('.npy', '.png', '.jpg', '.jpeg', '.tif', '.tiff')
    for root in roots:
        for suffix in suffixes:
            p = root / (stem + suffix)
            if p.exists():
                return p
    # Fallback is still scoped to this run and camera; the manifest order is
    # retained so timestamps cannot be silently shuffled.
    hits = []
    for p in run_root.rglob('*'):
        if p.is_file() and cam.lower() in str(p).lower() and modality.lower() in str(p).lower() \
                and p.suffix.lower() in suffixes:
            hits.append(p)
    hits = sorted(hits)
    if not hits:
        return None
    return hits[int(frame_id) % len(hits)] if str(frame_id).isdigit() else hits[0]


def _geom25(xyz, normal, rgb, depth, camera_index):
    scale = max(float(np.nanstd(xyz)), 1e-3)
    xn = np.clip(xyz / scale, -10., 10.)
    rgb = np.clip(rgb, 0., 1.)
    onehot = np.zeros((len(xyz), len(CAMERA_NAMES)), np.float32)
    onehot[:, int(camera_index)] = 1.
    g = np.c_[xn, normal, rgb, depth[:, None] / max(scale, 1e-3),
              xn * xn, rgb * rgb, onehot, normal * normal]
    if g.shape[1] != 25:
        raise AssertionError(g.shape)
    return g.astype(np.float32)


def _voxel_downsample(xyz, normal, geom, timestamp, voxel):
    vox = np.floor(np.asarray(xyz) / float(voxel)).astype(np.int64)
    _, keep = np.unique(vox, axis=0, return_index=True)
    keep = np.sort(keep)
    return xyz[keep], normal[keep], geom[keep], timestamp[keep]


def build_patch_memory(aux, manifest, origin_world=None):
    """Use a verified geometry cache, or deproject the raw Tartan RGB-D frames."""
    geometry_path = aux.get('geometry_path')
    if geometry_path is not None:
        d = _npz_dict(geometry_path)
        keys = {'xyz', 'normal', 'geom', 'timestamp'}
        if not keys.issubset(d):
            raise ValueError(f'{geometry_path} must contain {sorted(keys)}')
        xyz, normal, geom, ts = (d['xyz'], d['normal'], d['geom'], d['timestamp'])
        ref_t = REAL_RUNS[aux['spec']['run_id']].data['time']
        ts = _align_external_clock(ts, ref_t)
    else:
        pose_t, poses = _manifest_pose_arrays(manifest)
        if pose_t is None:
            raise ValueError('geometry build requires camera poses and timestamps')
        pose_t = _align_external_clock(pose_t, REAL_RUNS[aux['spec']['run_id']].data['time'])
        R_WC = _pose_rotations(poses); t_WC = poses[:, :3]
        frame_ids = manifest.get('frame_ids', manifest.get('frame_nums', np.arange(len(pose_t))))
        frame_ids = np.asarray(frame_ids).reshape(-1)
        xs = []; ns = []; gs = []; tt = []
        for ci, cam in enumerate(CAMERA_NAMES):
            for fi in range(0, len(frame_ids), GEOM_FRAME_STRIDE):
                depth_path = _resolve_frame_file(aux['run_root'], cam, frame_ids[fi], 'depth')
                if depth_path is None:
                    continue
                depth = _read_depth(depth_path)
                if depth.ndim != 2:
                    continue
                depth = depth[::GEOM_PIXEL_STRIDE, ::GEOM_PIXEL_STRIDE]
                h, w = depth.shape
                yy, xx = np.mgrid[0:h, 0:w]
                # Tartan camera convention: x forward, y right, z down.
                fx = fy = 320. / GEOM_PIXEL_STRIDE
                cx = cy = 320. / GEOM_PIXEL_STRIDE
                z = depth.reshape(-1)
                valid = np.isfinite(z) & (z > .05) & (z < 80.)
                if not valid.any():
                    continue
                xxv = xx.reshape(-1)[valid]; yyv = yy.reshape(-1)[valid]; zv = z[valid]
                pc = np.c_[(xxv - cx) * zv / fx, (yyv - cy) * zv / fy, zv]
                gx = np.gradient(depth, axis=1); gy = np.gradient(depth, axis=0)
                nn = np.c_[-gx.reshape(-1)[valid], -gy.reshape(-1)[valid], np.ones(valid.sum())]
                nn /= np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-6)
                rgb_path = _resolve_frame_file(aux['run_root'], cam, frame_ids[fi], 'image')
                rgb = _read_rgb(rgb_path, (h, w)).reshape(-1, 3)[valid]
                xyz_w = t_WC[fi][None] + pc @ R_WC[fi].T
                if origin_world is not None:
                    xyz_w = xyz_w - np.asarray(origin_world)[None]
                xs.append(xyz_w.astype(np.float32)); ns.append((nn @ R_WC[fi].T).astype(np.float32))
                gs.append(_geom25(xyz_w, ns[-1], rgb, zv, ci)); tt.append(np.full(len(xyz_w), pose_t[fi]))
                if sum(map(len, xs)) >= GEOM_MAX_POINTS:
                    break
            if sum(map(len, xs)) >= GEOM_MAX_POINTS:
                break
        if not xs:
            raise FileNotFoundError(f'no RGB-D frames found under {aux["run_root"]}')
        xyz, normal, geom, ts = map(np.concatenate, (xs, ns, gs, tt))
    xyz, normal, geom, ts = _voxel_downsample(np.asarray(xyz, np.float32),
                                               np.asarray(normal, np.float32),
                                               np.asarray(geom, np.float32),
                                               np.asarray(ts, float), CFG.patch_voxel_m)
    if len(xyz) == 0 or geom.shape[1] != 25:
        raise ValueError(f'empty or invalid causal geometry: xyz={xyz.shape}, geom={geom.shape}')
    return CausalPatchMemory(xyz, normal, geom, ts, voxel=CFG.patch_voxel_m)

# %% notebook cell 37
# --------------------------- exact canonical RunRecord preparation ---------------------------
def prepare_canonical_run(run_id, fk, calibration, mass):
    run = REAL_RUNS[run_id]
    aux = REAL_AUX[run_id]
    bag = Path(aux['run_root']) / 'rosbags' / f"{aux['spec']['run']}.bag"
    # The missing-bag error intentionally names the exact target instead of
    # synthesizing forces or falling back to the cached force slice.
    streams = read_native_force_bag(bag)
    native_audit = native_force_audit_table(streams)
    aligned = align_native_force(streams, run.data['time'])
    manifest = aux['manifest']
    force_base, frame_classes = convert_force_to_base(
        aligned['aligned_force'], streams, run, manifest, fk, calibration)
    force_audit = robust_force_audit(run.data['time'], force_base, run.data['contact'])
    impulse, touchdown = _touchdown_impulse_array(run, force_base)
    foot_world, world_origin = build_foot_world(run, manifest, fk, calibration, return_origin=True)
    desired_wrench = build_causal_desired_wrench(run, manifest, calibration, mass)

    # Adapter-only replacement: TrajectoryWindowDataset reads this exact P
    # force slice as its supervised target. The original P is retained in AUX.
    P_original = np.asarray(run.data['P']).copy()
    run.data['P'] = P_original.copy()
    run.data['P'][..., FORCE_SLICE] = force_base
    run.data['foot_world'] = foot_world
    run.data['desired_wrench'] = desired_wrench
    run.data['touchdown_impulse'] = impulse
    run.force_frame = 'base_link (native ROS frame audited and converted)'
    run.base_frame = 'base_link'
    run.controller_plan_available = 'controller_plan' in run.data

    aux.update({
        'bag_path': bag, 'force_streams': streams, 'native_force_audit': native_audit,
        'force_alignment': aligned, 'force_frame_classes': frame_classes,
        'force_audit': force_audit, 'touchdown_table': touchdown,
        'P_original_force': P_original[..., FORCE_SLICE].copy(),
        'mass_kg': mass, 'calibration': calibration,
    })
    PATCH_MEMORIES[run_id] = build_patch_memory(aux, manifest, origin_world=world_origin)
    return run


PATCH_MEMORIES = {}
FK_MODEL = None
CAMERA_CALIBRATION = None
if RUN_REAL:
    urdf_path = _locate_urdf()
    if urdf_path is None:
        raise FileNotFoundError('official ANYmal URDF not found; set ANYMAL_URDF explicitly')
    FK_MODEL = _load_cached_fk(next(iter(REAL_AUX.values())))
    if FK_MODEL is None:
        FK_MODEL = PinocchioAnyMalKinematics(urdf_path)
    mass = _urdf_total_mass(urdf_path)
    # Calibrate once on the first run and freeze the result for every terrain.
    first_id = next(iter(REAL_RUNS))
    first_aux = REAL_AUX[first_id]
    CAMERA_CALIBRATION = calibrate_base_camera_from_stance(
        REAL_RUNS[first_id], first_aux['manifest'], FK_MODEL)
    (IMPLEMENTATION_ROOT / 'camera_calibration.json').write_text(json.dumps({
        k: (v.tolist() if isinstance(v, np.ndarray) else v)
        for k, v in CAMERA_CALIBRATION.items()}, indent=2))
    for rid in REAL_RUNS:
        prepare_canonical_run(rid, FK_MODEL, CAMERA_CALIBRATION, mass)
    audits = []
    for rid, aux in REAL_AUX.items():
        a = dict(aux['force_audit']); a.update({'run_id': rid, 'frame_classes': '|'.join(aux['force_frame_classes'])})
        audits.append(a)
        aux['native_force_audit'].to_csv(IMPLEMENTATION_ROOT / f'{rid}_native_force_audit.csv', index=False)
    pd.DataFrame(audits).to_csv(IMPLEMENTATION_ROOT / 'force_audit_summary.csv', index=False)
    print(pd.DataFrame(audits))

# %% notebook cell 38
# --------------------------- causal split, gate, and exact-state ceiling ---------------------------
if RUN_REAL:
    RUN_LIST = list(REAL_RUNS.values())
    # This is a transparent blocked split for the four supplied trajectories.
    # The paper gate below must fail until independent trajectories are added.
    TEST_IDS = [RUN_LIST[-1].trajectory_id]
    VAL_IDS = [RUN_LIST[-2].trajectory_id]
    SPLIT_MANIFEST = assign_group_splits(RUN_LIST, TEST_IDS, VAL_IDS)
    SPLIT_MANIFEST.to_csv(IMPLEMENTATION_ROOT / 'split_manifest.csv', index=False)
    DATA_GATE = paper_data_gate(SPLIT_MANIFEST, min_runs=CFG.min_independent_runs)
    (IMPLEMENTATION_ROOT / 'paper_data_gate.json').write_text(json.dumps(DATA_GATE, indent=2))
    print('paper data gate:', json.dumps(DATA_GATE, indent=2))
    if RUN_EXACT_STATE_CEILING:
        train_ids = list(SPLIT_MANIFEST.query("split == 'train'").trajectory_id)
        test_id = TEST_IDS[0]
        ceiling, ceiling_model = blocked_exact_state_ceiling(
            [REAL_RUNS[x] for x in train_ids], REAL_RUNS[test_id], max_train=MAX_CEILING_TRAIN)
        (IMPLEMENTATION_ROOT / 'exact_state_ceiling.json').write_text(json.dumps(ceiling, indent=2))
        print('exact-state ceiling:', ceiling)
else:
    print('No split or paper gate was run because RUN_REAL=False.')

# %% notebook cell 39
# --------------------------- exact CausalPatchMemory provider ---------------------------
def _surface_at(memory, mean, origin_t, radius=CFG.patch_radius_m):
    xyz = memory.xyz
    past = memory.timestamp <= float(origin_t)
    dist = np.linalg.norm(xyz[:, :2] - np.asarray(mean)[None, :2], axis=1)
    ids = np.where(past & (dist <= radius))[0]
    if len(ids) == 0:
        return np.array([0., 0., 1.], np.float32), .5, 0.
    w = np.exp(-.5 * (dist[ids] / max(radius / 2, 1e-6)) ** 2)
    w /= max(w.sum(), 1e-8)
    normal = (memory.normal[ids] * w[:, None]).sum(0)
    normal /= max(np.linalg.norm(normal), 1e-6)
    rough = float(np.sqrt(np.sum(w * (memory.xyz[ids, 2] - np.sum(w * memory.xyz[ids, 2])) ** 2)))
    mu = float(np.clip(.85 - 2.0 * rough, .20, .90))
    return normal.astype(np.float32), mu, float(min(1., len(ids) / 100.))


def canonical_patch_provider(run_id, mean, cov, origin_t):
    """Call the original sigma-point patch query for every forecast foot."""
    if run_id not in PATCH_MEMORIES:
        raise KeyError(f'no causal patch memory for {run_id}')
    memory = PATCH_MEMORIES[run_id]
    mean_np = mean.detach().cpu().numpy() if torch.is_tensor(mean) else np.asarray(mean)
    cov_np = cov.detach().cpu().numpy() if torch.is_tensor(cov) else np.asarray(cov)
    time_np = origin_t.detach().cpu().numpy().reshape(-1) if torch.is_tensor(origin_t) else np.asarray(origin_t).reshape(-1)
    B, H, L, _ = mean_np.shape
    feat = np.zeros((B, H, L, 64), np.float32)
    normal = np.zeros((B, H, L, 3), np.float32)
    mu = np.zeros((B, H, L), np.float32)
    for b in range(B):
        for h in range(H):
            for l in range(L):
                feat[b, h, l] = memory.query_one(
                    mean_np[b, h, l], cov_np[b, h, l], time_np[b],
                    radius=CFG.patch_radius_m, sigma_points=13)
                normal[b, h, l], mu[b, h, l], _ = _surface_at(
                    memory, mean_np[b, h, l], time_np[b], CFG.patch_radius_m)
    return (torch.from_numpy(feat), torch.from_numpy(normal), torch.from_numpy(mu))


def _limited_origins(run):
    ids = valid_origins(run)
    if MAX_ORIGINS_PER_RUN is not None:
        ids = ids[:int(MAX_ORIGINS_PER_RUN)]
    return ids


def _loader_for_run(run, shuffle):
    ds = TrajectoryWindowDataset(run, _limited_origins(run), CFG)
    return torch.utils.data.DataLoader(ds, batch_size=1, shuffle=shuffle,
                                       num_workers=0, pin_memory=torch.cuda.is_available())


def _batch_forward(model, batch, run_id):
    for key in ('leg_hist', 'body_hist', 'action_plan', 'foot_now', 'origin_t', 'desired_wrench'):
        batch[key] = batch[key].to(DEVICE)
    target = {k: v.to(DEVICE) for k, v in batch['target'].items()}
    proposal = model.propose(batch['leg_hist'], batch['body_hist'], batch['action_plan'], batch['foot_now'])
    cov = foot_covariance(proposal['foot_chol_raw'])
    patch, normal, mu = canonical_patch_provider(run_id, proposal['foot_mean'].detach(),
                                                 cov.detach(), batch['origin_t'])
    out = {**proposal, **model.decode_force(proposal, patch.to(DEVICE), normal.to(DEVICE),
                                             mu.to(DEVICE), batch['desired_wrench'])}
    return out, target


def train_canonical_model(train_ids, val_ids=(), epochs=TRAIN_EPOCHS):
    """Train the exact V10R with the exact train_step and causal map provider."""
    model = V10R(d=CFG.d_model, horizons=len(CFG.horizons_s)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    history = []
    for epoch in range(int(epochs)):
        model.train(); train_rows = []
        for rid in train_ids:
            for batch in _loader_for_run(REAL_RUNS[rid], shuffle=True):
                train_rows.append(train_step(model, batch, lambda m, c, t, rid=rid:
                                             canonical_patch_provider(rid, m, c, t), optimizer))
        row = {'epoch': epoch, 'train_total': float(np.mean([x['total'] for x in train_rows]))}
        if val_ids:
            model.eval(); val_rows = []
            with torch.no_grad():
                for rid in val_ids:
                    for batch in _loader_for_run(REAL_RUNS[rid], shuffle=False):
                        out, target = _batch_forward(model, batch, rid)
                        val_rows.append({k: float(v.detach()) for k, v in v10r_loss(out, target).items()})
            row['val_total'] = float(np.mean([x['total'] for x in val_rows])) if val_rows else np.nan
        history.append(row)
        print(row)
    history_df = pd.DataFrame(history)
    history_df.to_csv(IMPLEMENTATION_ROOT / 'training_history.csv', index=False)
    torch.save({'model': model.state_dict(), 'cfg': dataclasses.asdict(CFG),
                'train_ids': list(train_ids), 'val_ids': list(val_ids)},
               IMPLEMENTATION_ROOT / 'v10r_canonical.pt')
    return model, history_df


MODEL = None
TRAINING_HISTORY = None
if RUN_REAL and RUN_TRAINING:
    train_ids = list(SPLIT_MANIFEST.query("split == 'train'").trajectory_id)
    val_ids = list(SPLIT_MANIFEST.query("split == 'val'").trajectory_id)
    MODEL, TRAINING_HISTORY = train_canonical_model(train_ids, val_ids)
else:
    print('Training is gated. Set RUN_TRAINING=True after a successful real-data preparation.')

# %% notebook cell 40
# --------------------------- blocked evaluation: all predicted sensors ---------------------------
def evaluate_canonical_run(model, run_id):
    run = REAL_RUNS[run_id]
    loader = _loader_for_run(run, shuffle=False)
    rows_force = []; rows_phase = []; rows_foot = []; rows_impulse = []
    rows_leg_sensor = []; rows_body_sensor = []; rows_event = []
    prediction_store = {k: [] for k in ('force_mean', 'foot_mean', 'impulse',
                                        'contact_prob', 'mode_logits', 'leg_sensor_mean', 'body_sensor_mean')}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            out, target = _batch_forward(model, batch, run_id)
            prediction_store['force_mean'].append(out['force_mean'].cpu().numpy())
            prediction_store['foot_mean'].append(out['foot_mean'].cpu().numpy())
            prediction_store['impulse'].append(out['impulse'].cpu().numpy())
            prediction_store['contact_prob'].append(out['contact_prob'].cpu().numpy())
            prediction_store['mode_logits'].append(out['mode_logits'].cpu().numpy())
            prediction_store['leg_sensor_mean'].append(out['leg_sensor_raw'][..., :9].cpu().numpy())
            prediction_store['body_sensor_mean'].append(out['body_raw'][..., :12].cpu().numpy())
            rows_phase.append(target['phase'].cpu().numpy())
            rows_force.append(target['force'].cpu().numpy())
            rows_foot.append(target['foot'].cpu().numpy())
            rows_impulse.append(target['impulse'].cpu().numpy())
            rows_leg_sensor.append(target['leg_sensor'].cpu().numpy())
            rows_body_sensor.append(target['body_sensor'].cpu().numpy())
    pred = {k: np.concatenate(v, axis=0) for k, v in prediction_store.items()}
    y_force = np.concatenate(rows_force, axis=0); phase = np.concatenate(rows_phase, axis=0)
    y_foot = np.concatenate(rows_foot, axis=0); y_impulse = np.concatenate(rows_impulse, axis=0)
    y_leg = np.concatenate(rows_leg_sensor, axis=0); y_body = np.concatenate(rows_body_sensor, axis=0)
    force_frame = force_metric_frame(y_force, pred['force_mean'], phase)
    force_frame.insert(0, 'run_id', run_id)
    sensor_frame = pd.DataFrame([
        {'run_id': run_id, 'target': 'leg_sensor', 'mae': float(np.abs(pred['leg_sensor_mean'] - y_leg).mean()),
         'p95': float(np.quantile(np.abs(pred['leg_sensor_mean'] - y_leg), .95))},
        {'run_id': run_id, 'target': 'body_sensor', 'mae': float(np.abs(pred['body_sensor_mean'] - y_body).mean()),
         'p95': float(np.quantile(np.abs(pred['body_sensor_mean'] - y_body), .95))},
        {'run_id': run_id, 'target': 'foot_position', 'mae_m': float(np.abs(pred['foot_mean'] - y_foot).mean()),
         'p95_m': float(np.quantile(np.abs(pred['foot_mean'] - y_foot), .95))},
        {'run_id': run_id, 'target': 'impulse', 'mae_Ns': float(np.abs(pred['impulse'] - y_impulse).mean()),
         'p95_Ns': float(np.quantile(np.abs(pred['impulse'] - y_impulse), .95))},
    ])
    event_target = (phase == 1).astype(float)
    event_pred = pred['contact_prob']
    event_frame = pd.DataFrame([{
        'run_id': run_id, 'event_brier': float(np.mean((event_pred - (phase >= 1)) ** 2)),
        'touchdown_probability_mae': float(np.abs(event_pred - event_target).mean()),
    }])
    out_dir = IMPLEMENTATION_ROOT / 'evaluation'; out_dir.mkdir(exist_ok=True)
    force_frame.to_csv(out_dir / f'{run_id}_force_metrics.csv', index=False)
    sensor_frame.to_csv(out_dir / f'{run_id}_sensor_metrics.csv', index=False)
    event_frame.to_csv(out_dir / f'{run_id}_event_metrics.csv', index=False)
    np.savez_compressed(out_dir / f'{run_id}_predictions.npz', **pred,
                        force_target=y_force, foot_target=y_foot,
                        impulse_target=y_impulse, phase=phase)
    return {'force': force_frame, 'sensor': sensor_frame, 'event': event_frame, 'pred': pred,
            'force_target': y_force, 'foot_target': y_foot,
            'impulse_target': y_impulse, 'phase': phase}


REAL_EVAL = {}
if RUN_REAL and RUN_EVALUATION:
    if MODEL is None:
        checkpoint = IMPLEMENTATION_ROOT / 'v10r_canonical.pt'
        if not checkpoint.exists():
            raise FileNotFoundError('no canonical checkpoint; set RUN_TRAINING=True first')
        MODEL = V10R(d=CFG.d_model, horizons=len(CFG.horizons_s)).to(DEVICE)
        MODEL.load_state_dict(torch.load(checkpoint, map_location=DEVICE)['model'])
    for rid in list(SPLIT_MANIFEST.query("split == 'test'").trajectory_id):
        REAL_EVAL[rid] = evaluate_canonical_run(MODEL, rid)
    print('evaluated runs:', list(REAL_EVAL))
else:
    print('Evaluation is gated. Set RUN_EVALUATION=True after training/checkpoint restore.')

# %% notebook cell 41
# --------------------------- plots and 3-D reconstruction ---------------------------
def save_canonical_plots(run_id, eval_result=None):
    out_dir = IMPLEMENTATION_ROOT / 'figures'; out_dir.mkdir(exist_ok=True)
    figs = []
    if eval_result is not None:
        f = eval_result['force']
        if len(f):
            fig, ax = plt.subplots(figsize=(7, 4))
            for leg, g in f.groupby('leg'):
                h = g.groupby('horizon_s').mae_n.mean()
                ax.plot(h.index, h.values, marker='o', label=leg)
            ax.set(xlabel='forecast horizon [s]', ylabel='component force MAE [N]',
                   title=f'{run_id}: causal V10-R force forecast')
            ax.grid(alpha=.25); ax.legend(ncol=2); fig.tight_layout()
            fig.savefig(out_dir / f'{run_id}_force_trend.png', dpi=180); figs.append(fig)
    if run_id in PATCH_MEMORIES:
        mem = PATCH_MEMORIES[run_id]
        ids = np.arange(len(mem.xyz))
        if len(ids) > 20_000:
            ids = np.linspace(0, len(ids) - 1, 20_000).astype(int)
        fig = plt.figure(figsize=(7, 6)); ax = fig.add_subplot(111, projection='3d')
        pts = mem.xyz[ids]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, alpha=.25, c=pts[:, 2], cmap='viridis')
        ax.set(xlabel='world x [m]', ylabel='world y [m]', zlabel='world z [m]',
               title=f'{run_id}: causal RGB-D patch geometry')
        fig.tight_layout(); fig.savefig(out_dir / f'{run_id}_geometry_3d.png', dpi=180); figs.append(fig)
    for fig in figs:
        plt.close(fig)
    return [str(out_dir / p) for p in sorted(out_dir.glob(f'{run_id}_*.png'))]


if RUN_REAL and RUN_PLOTS:
    for rid in REAL_RUNS:
        save_canonical_plots(rid, REAL_EVAL.get(rid))
    print('figures:', sorted(str(p) for p in (IMPLEMENTATION_ROOT / 'figures').glob('*.png')))

# %% notebook cell 42
# --------------------------- per-leg shadow foothold planning ---------------------------
def _candidate_support_features(memory, xyz, origin_t):
    rows = []
    for p in np.asarray(xyz):
        normal, mu, conf = _surface_at(memory, p, origin_t, CFG.patch_radius_m)
        slope = float(np.linalg.norm(normal[:2]) / max(abs(normal[2]), 1e-5))
        rows.append((normal, mu, conf, slope))
    return rows


def shadow_plan_at_origin(model, run_id, origin_position, horizon_index=0):
    """Select one candidate independently for each leg using the canonical scorer."""
    run = REAL_RUNS[run_id]
    origins = _limited_origins(run)
    i = int(origins[int(origin_position)])
    ds = TrajectoryWindowDataset(run, [i], CFG)
    batch = torch.utils.data.default_collate([ds[0]])
    with torch.no_grad():
        out, _ = _batch_forward(model, batch, run_id)
    pred_foot = out['foot_mean'][0].cpu().numpy()[int(horizon_index)]
    pred_force = out['force_mean'][0].cpu().numpy()[int(horizon_index)]
    pred_impulse = out['impulse'][0].cpu().numpy()[int(horizon_index)]
    origin_t = float(batch['origin_t'].reshape(-1)[0])
    memory = PATCH_MEMORIES[run_id]
    tables = []
    for li, leg in enumerate(LEG_NAMES):
        cand = make_xy_candidates(pred_foot[li])
        support = _candidate_support_features(memory, cand, origin_t)
        rows = []
        for p, (n, mu, conf, slope) in zip(cand, support):
            offset = float(np.linalg.norm(p[:2] - pred_foot[li, :2]))
            reach = max(offset - .10, 0.)
            collision = float(np.exp(-offset ** 2 / (.025 ** 2))) if conf > 0 else 1.
            slip = float(np.clip(slope / max(mu, .1), 0., 1.))
            rows.append({
                'leg': leg, 'x': p[0], 'y': p[1], 'z': p[2],
                'collision_prob': collision, 'reach_violation': reach,
                'map_confidence': conf, 'support_risk': slope,
                'slip_prob': slip, 'force_cvar': float(np.linalg.norm(pred_force[li])),
                'impulse_cvar': float(np.linalg.norm(pred_impulse[li])),
                'epistemic': 1. - conf, 'energy': offset, 'nominal_offset': offset,
            })
        tables.append(score_candidates(pd.DataFrame(rows)))
    scored = pd.concat(tables, ignore_index=True)
    selected = scored.loc[scored.groupby('leg').score.idxmin()].sort_values('leg')
    selected.to_csv(IMPLEMENTATION_ROOT / f'{run_id}_shadow_plan.csv', index=False)
    if RUN_PLOTS:
        for leg in LEG_NAMES:
            plot_leg_candidate_map(scored, leg)
            plt.gcf().savefig(IMPLEMENTATION_ROOT / 'figures' / f'{run_id}_{leg}_candidate_map.png', dpi=180)
            plt.close(plt.gcf())
    return selected, scored


if RUN_REAL and RUN_PLANNER:
    if MODEL is None:
        raise RuntimeError('train or restore MODEL before shadow planning')
    for rid in SPLIT_MANIFEST.query("split == 'test'").trajectory_id:
        selected, _ = shadow_plan_at_origin(MODEL, rid, origin_position=0, horizon_index=0)
        print(rid, 'selected footholds:\n', selected[['leg', 'x', 'y', 'z', 'score']])

# %% notebook cell 43
# --------------------------- final auditable status ---------------------------
IMPLEMENTATION_STATUS = {
    'architecture_source': 'V10R_Causal_Contact_Patch_World_Model(3).ipynb',
    'source_sha256': CANONICAL_SOURCE_SHA256,
    'original_cells_preserved': True,
    'canonical_functions_called': [
        'load_physics_npz', 'bandlimit_then_decimate', 'touchdown_targets',
        'CausalPatchMemory.query_one', 'V10R.propose', 'V10R.decode_force',
        'v10r_loss', 'TrajectoryWindowDataset', 'train_step',
        'force_metric_frame', 'make_xy_candidates', 'score_candidates',
    ],
    'real_run_requested': bool(RUN_REAL),
    'real_runs_loaded': list(REAL_RUNS),
    'training_ran': bool(MODEL is not None and RUN_TRAINING),
    'evaluation_ran': bool(REAL_EVAL),
    'planner_ran': bool(RUN_REAL and RUN_PLANNER),
    'three_to_four_newton_claim': 'not asserted; report only measured force_metric_frame outputs',
}
(IMPLEMENTATION_ROOT / 'implementation_status.json').write_text(json.dumps(IMPLEMENTATION_STATUS, indent=2))
print(json.dumps(IMPLEMENTATION_STATUS, indent=2))
