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
# STRICT S1 — controls, run registry, and dependency policy
from __future__ import annotations
import contextlib, csv, gc, importlib.util, io, platform, re, shutil, subprocess, sys, tempfile, time
from collections import defaultdict
from datetime import datetime, timezone
try:
    from IPython.display import display
except ImportError:
    display = print

STRICT_PIPELINE_VERSION = 'V10R_STRICT_STAGED_V1'

# Scientific output namespace.  Diagnostic and paper checkpoints/results are
# physically separated so a four-run diagnostic can never overwrite a paper run.
ANALYSIS_TIER = 'paper'                 # 'paper' or explicitly watermarked 'diagnostic'
ALLOW_DIAGNOSTIC_TRAINING = False

if Path('/content').exists() and importlib.util.find_spec('google.colab') is not None:
    from google.colab import drive
    if not Path('/content/drive/MyDrive').exists(): drive.mount('/content/drive')

# Change only this value to execute a stage. Safe values are documented below.
PIPELINE_STAGE = 'inventory'
VALID_STAGES = {
    'inventory', 'audit', 'geometry', 'prepare', 'verify', 'delete',
    'exact_state', 'train', 'ablations', 'evaluate', 'plots', 'planner', 'status', 'all_safe',
}
if PIPELINE_STAGE not in VALID_STAGES:
    raise ValueError(f'PIPELINE_STAGE must be one of {sorted(VALID_STAGES)}')

RAW_ROOT = Path('/content/tartanground_v10')
PROJECT_ROOT = Path('/content/drive/MyDrive/UGP_V7_FULL4B')
STRICT_ROOT = PROJECT_ROOT / 'V10R_STRICT_STAGED'
PREPARED_ROOT = STRICT_ROOT / 'prepared_runs'
AUDIT_ROOT = STRICT_ROOT / 'audits'
CONTRACT_ROOT = STRICT_ROOT / 'contracts'
MODEL_ROOT = STRICT_ROOT / 'models' / ANALYSIS_TIER
RESULT_ROOT = STRICT_ROOT / 'results' / ANALYSIS_TIER
FIGURE_ROOT = STRICT_ROOT / 'figures' / ANALYSIS_TIER
for _p in (STRICT_ROOT, PREPARED_ROOT, AUDIT_ROOT, CONTRACT_ROOT, MODEL_ROOT, RESULT_ROOT, FIGURE_ROOT):
    _p.mkdir(parents=True, exist_ok=True)

# Empty means: operate on every raw run currently present, up to the hard limit.
ACTIVE_RUN_IDS = []
MAX_SIMULTANEOUS_RAW_RUNS = 2

# Geometry is deterministic, multi-view, causal, and cross-view voxel fused.
GEOMETRY_PIXEL_STRIDE = 4
GEOMETRY_FRAME_STRIDE = 1
GEOMETRY_MAX_RANGE_M = 10.0
GEOMETRY_LOCAL_RADIUS_M = 6.0
GEOMETRY_MIN_UP_NORMAL = 0.25
GEOMETRY_CHUNK_FRAMES = 25
GEOMETRY_VALIDATION_FRAME_STRIDE = 25
GEOMETRY_VALIDATION_POINTS_PER_FRAME = 200
GEOMETRY_MIN_HELDOUT_COVERAGE = 0.70
GEOMETRY_MAX_HELDOUT_P95_M = 0.10

# Force/event label contract.
FORCE_UNIFORM_HZ = 400
FORCE_LOWPASS_HZ = 45.0
CONTACT_ON_N = 12.0
CONTACT_OFF_N = 7.0
IMPACT_PRE_S = 0.005
IMPACT_POST_S = 0.025
IMPACT_WAVEFORM_HZ = 400
HAPTIC_STRIDE = 5                       # 20 Hz causal posterior updates during contact
HAPTIC_ZERO_SHOT_TRAIN_FRACTION = 0.5
MAP_SNAPSHOT_DT_S = 0.02                # cache uses floor(t/dt), so it cannot see the future

# Training/evaluation contract. Paper mode never bypasses the data gate.
TRAIN_SEEDS = (731, 1731, 2731, 3731, 4731)
TRAIN_EPOCHS_G2 = 20
TRAIN_EPOCHS_G4 = 40
TRAIN_BATCH_SIZE = 4
TRAIN_ORIGINS_PER_RUN = None
TRAIN_ORIGIN_STRIDE = 5                 # 20 Hz origins; native event labels are unchanged
STRICT_ACTION_MODE = 'hold_last'         # deployable; logged future commands are oracle-only
MAX_CEILING_TRAIN = 120_000
EVAL_ORIGIN_STRIDE = 10                 # 10 Hz origins; outputs still span 0.1–1.0 s
BOOTSTRAP_BLOCK_S = 5.0
BOOTSTRAP_REPLICATES = 1000

# Destructive action remains off. The verify stage prints the only accepted token.
DELETE_CONFIRMATION = ''

RUN_REGISTRY = [
    {'run_id':'forest_P2000', 'terrain':'forest', 'environment':'ForestEnv', 'run':'P2000',
     'trajectory_id':'ForestEnv_P2000', 'gait':'unreported', 'policy':'unreported', 'speed_bin':'unreported'},
    {'run_id':'swamp_marsh_P2023', 'terrain':'swamp_marsh', 'environment':'GreatMarsh', 'run':'P2023',
     'trajectory_id':'GreatMarsh_P2023', 'gait':'unreported', 'policy':'unreported', 'speed_bin':'unreported'},
    {'run_id':'rough_uneven_P2000', 'terrain':'rough_uneven', 'environment':'SeasonalForestWinter', 'run':'P2000',
     'trajectory_id':'SeasonalForestWinter_P2000', 'gait':'unreported', 'policy':'unreported', 'speed_bin':'unreported'},
    {'run_id':'wet_uneven_proxy_P2000', 'terrain':'wet_uneven_proxy', 'environment':'OldTownFall', 'run':'P2000',
     'trajectory_id':'OldTownFall_P2000', 'gait':'unreported', 'policy':'unreported', 'speed_bin':'unreported'},
]

FORCE_TOPICS = {
    'FL':'/state_estimator/contact_force_lf_foot',
    'FR':'/state_estimator/contact_force_rf_foot',
    'RL':'/state_estimator/contact_force_lh_foot',
    'RR':'/state_estimator/contact_force_rh_foot',
}
PHYSICAL_CLOCK_TOPICS = {
    '/state_estimator/joint_states', *FORCE_TOPICS.values(), '/state_estimator/twist',
}
CANONICAL_JOINT_NAMES = (
    'LF_HAA','LF_HFE','LF_KFE','RF_HAA','RF_HFE','RF_KFE',
    'LH_HAA','LH_HFE','LH_KFE','RH_HAA','RH_HFE','RH_KFE',
)
LEG_NAMES = ('FL','FR','RL','RR')
CAMERA_NAMES = ('lcam_front','lcam_back','lcam_left','lcam_right','lcam_top','lcam_bottom')

print('STRICT pipeline:', STRICT_PIPELINE_VERSION)
print('stage:', PIPELINE_STAGE, '| tier:', ANALYSIS_TIER, '| raw-run limit:', MAX_SIMULTANEOUS_RAW_RUNS)
print('outputs:', STRICT_ROOT)

# %% notebook cell 32
# STRICT S2 — atomic artifacts, bounded discovery, inventory, and storage guards
def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def canonical_json_hash(obj):
    raw = json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def atomic_write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(obj, f, indent=2, sort_keys=True)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError): os.unlink(tmp)
        raise


def atomic_savez(path, compressed=True, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.stem + '.', suffix='.npz', dir=str(path.parent))
    os.close(fd)
    try:
        (np.savez_compressed if compressed else np.savez)(tmp, **arrays)
        # np.savez respects the existing .npz suffix above.
        with open(tmp, 'rb') as f: os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError): os.unlink(tmp)
        raise


def directory_bytes(path):
    path = Path(path)
    if not path.exists(): return 0
    if path.is_file(): return path.stat().st_size
    total = 0
    for p in path.rglob('*'):
        if p.is_file():
            with contextlib.suppress(OSError): total += p.stat().st_size
    return total


def gib(n): return float(n) / 1024**3


def storage_row(path):
    path = Path(path)
    probe = path if path.exists() else next((p for p in path.parents if p.exists()), Path('/'))
    use = shutil.disk_usage(probe)
    return {'path':str(path), 'exists':path.exists(), 'content_gib':gib(directory_bytes(path)),
            'free_gib':gib(use.free), 'capacity_gib':gib(use.total)}


def require_local_headroom(min_free_gib=4.0):
    probe = Path('/content') if Path('/content').exists() else Path(tempfile.gettempdir())
    free = gib(shutil.disk_usage(probe).free)
    if free < min_free_gib:
        raise RuntimeError(
            f'Only {free:.2f} GiB free at {probe}. Free local Colab disk before reading Drive/FUSE files; '
            f'buying Drive storage does not enlarge the Colab VM disk.'
        )
    return free


def _bounded_first(candidates):
    for p in map(Path, candidates):
        if p.exists(): return p
    return None


def raw_run_path(spec):
    return RAW_ROOT / spec['environment'] / 'Data_anymal' / spec['run']


def locate_physics_cache(spec):
    name = f"{spec['environment']}_{spec['run']}_physics.npz"
    roots = [PROJECT_ROOT/'cache', PROJECT_ROOT/'V10_CACHE', PROJECT_ROOT/'V7_FULL4B'/'cache', PROJECT_ROOT]
    exact = _bounded_first([r/name for r in roots])
    if exact is not None: return exact
    # Bounded fallback: search only known project subtrees and the exact filename.
    for root in roots[:3]:
        if root.exists():
            hits = sorted(root.rglob(name))
            if hits: return hits[0]
    return None


def locate_frame_manifest(spec):
    name = f"{spec['environment']}_{spec['run']}_frame_manifest.npz"
    roots = [PROJECT_ROOT/'cache', PROJECT_ROOT/'V10_CACHE', PROJECT_ROOT/'V7_FULL4B'/'cache']
    return _bounded_first([r/name for r in roots])


def inspect_physics_clock(path):
    if path is None: return {'clock_ok':False, 'clock_reason':'physics cache missing'}
    try:
        with np.load(path, allow_pickle=False) as z:
            keys = set(z.files)
            if 'grid_original' in keys:
                grid = np.asarray(z['grid_original'], np.float64).reshape(-1)
                source = 'grid_original'
            elif {'time','crop_start_original'} <= keys:
                grid = np.asarray(z['time'], np.float64).reshape(-1) + float(z['crop_start_original'])
                source = 'time_plus_crop_start_original'
            else:
                return {'clock_ok':False, 'clock_reason':'need grid_original or time+crop_start_original'}
            dt = np.diff(grid)
            return {'clock_ok':bool(len(grid)>1 and np.all(dt>0)), 'clock_source':source,
                    'n':len(grid), 'start_s':float(grid[0]), 'end_s':float(grid[-1]),
                    'median_dt_ms':float(1000*np.median(dt)), 'nonpositive':int((dt<=0).sum())}
    except Exception as e:
        return {'clock_ok':False, 'clock_reason':f'{type(e).__name__}: {e}'}


def inventory_table():
    rows=[]
    for spec in RUN_REGISTRY:
        rr=raw_run_path(spec); bag=rr/'rosbags'/f"{spec['run']}.bag"
        pc=locate_physics_cache(spec); fm=locate_frame_manifest(spec)
        ck=PREPARED_ROOT/spec['run_id']/'prepared_manifest.json'
        clock=inspect_physics_clock(pc)
        rows.append({**spec, 'raw_present':rr.exists(), 'raw_gib':round(gib(directory_bytes(rr)),3),
                     'bag_present':bag.exists(), 'physics_cache':str(pc) if pc else '',
                     'frame_manifest':str(fm) if fm else '', 'prepared_manifest':ck.exists(),
                     **clock})
    return pd.DataFrame(rows)


INVENTORY = inventory_table()
display(INVENTORY)
display(pd.DataFrame([storage_row('/content'), storage_row(RAW_ROOT), storage_row(PROJECT_ROOT)]))

present = INVENTORY.loc[INVENTORY.raw_present, 'run_id'].tolist()
selected = list(ACTIVE_RUN_IDS) if ACTIVE_RUN_IDS else present
unknown = sorted(set(selected) - set(INVENTORY.run_id))
if unknown: raise ValueError(f'unknown ACTIVE_RUN_IDS: {unknown}')
if len(selected) > MAX_SIMULTANEOUS_RAW_RUNS and PIPELINE_STAGE in {'audit','geometry','prepare','all_safe'}:
    raise RuntimeError(
        f'{len(selected)} raw runs selected, but the staged contract allows at most '
        f'{MAX_SIMULTANEOUS_RAW_RUNS}. Set ACTIVE_RUN_IDS to one or two run IDs.'
    )
ACTIVE_SPECS = [s for s in RUN_REGISTRY if s['run_id'] in selected]
print('active raw runs:', [s['run_id'] for s in ACTIVE_SPECS])

# %% notebook cell 33
# STRICT S3 — immutable scientific contracts (templates are never silently approved)
FORCE_CONTRACT_PATH = CONTRACT_ROOT / 'force_frame_contract.json'
CALIBRATION_CONTRACT_PATH = CONTRACT_ROOT / 'calibration_contract.json'
SIGNAL_CONTRACT_PATH = CONTRACT_ROOT / 'signal_semantics_contract.json'
SPLIT_CONTRACT_PATH = CONTRACT_ROOT / 'trajectory_split.csv'


def write_contract_template_once(path, obj):
    path=Path(path)
    if not path.exists():
        atomic_write_json(path, obj)
        print('Wrote UNVERIFIED template:', path)


write_contract_template_once(FORCE_CONTRACT_PATH, {
    'schema':'V10R_FORCE_FRAME_CONTRACT_V1', 'verified':False,
    'verified_before_split':False,
    'verification_method':'publisher/source-code statement AND static known-load/sign test',
    'evidence':'REPLACE_WITH_AUDITABLE_REFERENCE',
    'force_direction':'environment_on_robot',
    'per_leg':{leg:{'source_frame':'UNRESOLVED', 'axis_permutation':[0,1,2],
                    'axis_sign':[1,1,1], 'static_test_passed':False} for leg in LEG_NAMES},
})
write_contract_template_once(CALIBRATION_CONTRACT_PATH, {
    'schema':'V10R_CALIBRATION_CONTRACT_V1', 'verified':False,
    'source':'official transform or independent calibration trajectory excluded from every model split',
    'evidence':'REPLACE_WITH_AUDITABLE_REFERENCE',
    'calibration_trajectory_id':'REPLACE; must not be a model train/val/test trajectory',
    'rotation_convention':'p_C = R_C_B @ p_B + t_C_B',
    'R_C_B':[[1,0,0],[0,1,0],[0,0,1]], 't_C_B_m':[0,0,0],
    'urdf_path':'REPLACE_WITH_ABSOLUTE_COLAB_PATH', 'urdf_sha256':'REPLACE',
    'joint_names':{name:name for name in CANONICAL_JOINT_NAMES},
    'foot_frame_names':{'FL':'REPLACE','FR':'REPLACE','RL':'REPLACE','RR':'REPLACE'},
    'camera_intrinsics':{cam:{'width':640,'height':640,'fx':320.0,'fy':320.0,'cx':320.0,'cy':320.0,
                              'depth_sigma_a_m':0.002,'depth_sigma_b_per_m':0.001,
                              'verified':False,'source':'REPLACE'} for cam in CAMERA_NAMES},
})
write_contract_template_once(SIGNAL_CONTRACT_PATH, {
    'schema':'V10R_SIGNAL_CONTRACT_V1', 'verified':False,
    'world_frame':'NED', 'camera_axes':'x_forward_y_right_z_down',
    'pose_format':'tx_ty_tz_qx_qy_qz_qw',
    'imu_acceleration_semantics':'UNRESOLVED; required: specific_force_body',
    'imu_gyro_frame':'UNRESOLVED; required: base',
    'twist_frame':'UNRESOLVED; required: base',
    'body_B_layout':['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z',
                     'v_x','v_y','v_z','omega_x','omega_y','omega_z'],
    'evidence':'REPLACE_WITH_AUDITABLE_REFERENCE',
})


def load_json_contract(path):
    with Path(path).open() as f: return json.load(f)


def validate_force_contract(c):
    errors=[]
    if c.get('schema')!='V10R_FORCE_FRAME_CONTRACT_V1': errors.append('wrong schema')
    if c.get('verified') is not True: errors.append('verified must be true')
    if c.get('verified_before_split') is not True: errors.append('verified_before_split must be true')
    if len(str(c.get('evidence','')).strip())<12 or 'REPLACE' in str(c.get('evidence','')): errors.append('auditable evidence missing')
    if c.get('force_direction') not in {'environment_on_robot','robot_on_environment'}: errors.append('force_direction invalid')
    allowed={'base','world_ned','foot'}
    for leg in LEG_NAMES:
        x=c.get('per_leg',{}).get(leg,{})
        if x.get('source_frame') not in allowed: errors.append(f'{leg}: source_frame must be one of {sorted(allowed)}')
        perm=x.get('axis_permutation'); sign=x.get('axis_sign')
        if sorted(perm or []) != [0,1,2]: errors.append(f'{leg}: invalid axis_permutation')
        if len(sign or [])!=3 or any(v not in (-1,1) for v in sign): errors.append(f'{leg}: invalid axis_sign')
        if x.get('static_test_passed') is not True: errors.append(f'{leg}: static_test_passed is not true')
    return errors


def validate_calibration_contract(c, model_trajectory_ids=()):
    errors=[]
    if c.get('schema')!='V10R_CALIBRATION_CONTRACT_V1': errors.append('wrong schema')
    if c.get('verified') is not True: errors.append('verified must be true')
    if len(str(c.get('evidence','')).strip())<12 or 'REPLACE' in str(c.get('evidence','')): errors.append('auditable evidence missing')
    R=np.asarray(c.get('R_C_B',[]),float); t=np.asarray(c.get('t_C_B_m',[]),float)
    if R.shape!=(3,3) or not np.allclose(R.T@R,np.eye(3),atol=1e-4) or np.linalg.det(R)<.999:
        errors.append('R_C_B is not a proper rotation')
    if t.shape!=(3,) or not np.isfinite(t).all(): errors.append('t_C_B_m invalid')
    cal_id=str(c.get('calibration_trajectory_id',''))
    if cal_id in set(model_trajectory_ids): errors.append('calibration trajectory contaminates a model split')
    up=Path(str(c.get('urdf_path','')))
    if not up.exists(): errors.append(f'URDF missing: {up}')
    elif c.get('urdf_sha256')!=file_sha256(up): errors.append('URDF SHA-256 does not match contract')
    joint_names=c.get('joint_names',{}); foot_names=c.get('foot_frame_names',{})
    if set(joint_names)!=set(CANONICAL_JOINT_NAMES) or any(not str(v).strip() for v in joint_names.values()): errors.append('explicit canonical joint map missing')
    if set(foot_names)!=set(LEG_NAMES) or any(not str(v).strip() or 'REPLACE' in str(v) for v in foot_names.values()): errors.append('explicit foot frame map missing')
    for cam in CAMERA_NAMES:
        x=c.get('camera_intrinsics',{}).get(cam,{})
        if x.get('verified') is not True: errors.append(f'{cam}: intrinsics not verified')
        if min(float(x.get(k,0)) for k in ('width','height','fx','fy'))<=0: errors.append(f'{cam}: invalid intrinsics')
        if min(float(x.get(k,-1)) for k in ('depth_sigma_a_m','depth_sigma_b_per_m'))<0: errors.append(f'{cam}: depth noise model invalid')
    return errors


def validate_signal_contract(c):
    errors=[]
    if c.get('schema')!='V10R_SIGNAL_CONTRACT_V1' or c.get('verified') is not True: errors.append('signal contract is not verified')
    if c.get('world_frame')!='NED': errors.append('this adapter requires the verified TartanGround NED convention')
    if c.get('camera_axes')!='x_forward_y_right_z_down': errors.append('camera axes unresolved')
    if c.get('pose_format')!='tx_ty_tz_qx_qy_qz_qw': errors.append('pose format unresolved')
    if c.get('imu_acceleration_semantics')!='specific_force_body': errors.append('IMU acceleration must be verified as specific_force_body')
    if c.get('imu_gyro_frame')!='base': errors.append('gyro frame must be verified as base')
    if c.get('twist_frame')!='base': errors.append('twist frame must be verified as base')
    if len(str(c.get('evidence','')).strip())<12 or 'REPLACE' in str(c.get('evidence','')): errors.append('signal evidence missing')
    return errors


def load_and_validate_contracts(require_approved=True):
    fc,cc,sc=map(load_json_contract,(FORCE_CONTRACT_PATH,CALIBRATION_CONTRACT_PATH,SIGNAL_CONTRACT_PATH))
    errors={'force':validate_force_contract(fc),
            'calibration':validate_calibration_contract(cc,[x['trajectory_id'] for x in RUN_REGISTRY]),
            'signals':validate_signal_contract(sc)}
    if require_approved and any(errors.values()):
        message='\n'.join(f'{k}: '+('; '.join(v) if v else 'PASS') for k,v in errors.items())
        raise RuntimeError('G0 CONTRACT GATE FAILED — no frame/calibration/semantics may be guessed.\n'+message)
    return fc,cc,sc,errors


def inspect_urdf_names(path):
    import xml.etree.ElementTree as ET
    root=ET.parse(str(path)).getroot()
    joints=[x.get('name') for x in root.findall('.//joint')]
    links=[x.get('name') for x in root.findall('.//link')]
    print('1-DoF joint candidates:'); print('\n'.join(x for x in joints if x))
    print('\nfoot/contact link candidates:'); print('\n'.join(x for x in links if x and any(k in x.lower() for k in ('foot','toe','contact'))))


_,_,_,CONTRACT_ERRORS=load_and_validate_contracts(require_approved=False)
print(json.dumps(CONTRACT_ERRORS,indent=2))

# %% notebook cell 35
# STRICT S4 — exact native ROS force reader and original clock reconstruction
def ensure_python_packages(packages):
    missing=[pip_name for module_name,pip_name in packages if importlib.util.find_spec(module_name) is None]
    if missing:
        print('Installing required packages:', missing)
        subprocess.check_call([sys.executable,'-m','pip','install','-q',*missing])


def stamp_seconds(header):
    if header is None or not hasattr(header,'stamp'): return np.nan
    s=header.stamp
    sec=getattr(s,'sec',getattr(s,'secs',None)); ns=getattr(s,'nanosec',getattr(s,'nsecs',None))
    if sec is None or ns is None: return np.nan
    out=float(sec)+1e-9*float(ns)
    return out if np.isfinite(out) and out>0 else np.nan


def wrench_force_xyz(msg):
    x=msg.wrench if hasattr(msg,'wrench') else msg
    x=x.force if hasattr(x,'force') else x
    try: return np.array([x.x,x.y,x.z],np.float64)
    except Exception as e: raise ValueError(f'force payload has no x/y/z: {type(msg)}') from e


def physics_grid_from_cache(path):
    with np.load(path,allow_pickle=False) as z:
        d={k:np.asarray(z[k]) for k in z.files}
    if 'grid_original' in d:
        grid=np.asarray(d['grid_original'],np.float64).reshape(-1)
    elif 'time' in d and 'crop_start_original' in d:
        grid=np.asarray(d['time'],np.float64).reshape(-1)+float(d['crop_start_original'])
    else:
        raise ValueError(f'{path}: exact clock unavailable; need grid_original or time+crop_start_original')
    if len(grid)<2 or not np.all(np.diff(grid)>0): raise ValueError(f'{path}: non-monotone original grid')
    if 'time' not in d: d['time']=grid-grid[0]
    if len(d['time'])!=len(grid): raise ValueError(f'{path}: clock/data length mismatch')
    return d,grid


def read_native_force_exact(bag_path):
    ensure_python_packages([('rosbags','rosbags')])
    from rosbags.highlevel import AnyReader
    bag_path=Path(bag_path)
    if not bag_path.exists(): raise FileNotFoundError(f'raw ROS bag missing: {bag_path}')
    topic_to_leg={v:k for k,v in FORCE_TOPICS.items()}
    with AnyReader([bag_path]) as reader:
        physical=[c for c in reader.connections if c.topic in PHYSICAL_CLOCK_TOPICS]
        present={c.topic for c in physical}
        missing=set(FORCE_TOPICS.values())-present
        if missing: raise ValueError(f'{bag_path}: missing force topics {sorted(missing)}')
        try: _,t0_ns,_=next(reader.messages(connections=physical))
        except StopIteration: raise ValueError(f'{bag_path}: no physical messages')
        t0_sec=float(t0_ns)*1e-9
    streams={leg:defaultdict(list) for leg in LEG_NAMES}
    with AnyReader([bag_path]) as reader:
        conns=[c for c in reader.connections if c.topic in topic_to_leg]
        msgtypes={c.topic:c.msgtype for c in conns}
        for c,bag_ns,raw in reader.messages(connections=conns):
            msg=reader.deserialize(raw,c.msgtype); leg=topic_to_leg[c.topic]
            bag_abs=float(bag_ns)*1e-9
            header=getattr(msg,'header',None); h_abs=stamp_seconds(header)
            effective=h_abs if np.isfinite(h_abs) else bag_abs
            streams[leg]['bag_time_abs'].append(bag_abs)
            streams[leg]['header_time_abs'].append(h_abs)
            streams[leg]['time_original'].append(effective-t0_sec)
            streams[leg]['force_native'].append(wrench_force_xyz(msg))
            streams[leg]['frame_id'].append(str(getattr(header,'frame_id','') or ''))
            streams[leg]['clock_source'].append('header' if np.isfinite(h_abs) else 'bag')
    out={}
    for leg in LEG_NAMES:
        s=streams[leg]
        if not s['time_original']: raise ValueError(f'{bag_path}: empty {leg} force stream')
        order=np.argsort(np.asarray(s['time_original'],float),kind='stable')
        out[leg]={
            'bag_time_abs':np.asarray(s['bag_time_abs'],np.float64)[order],
            'header_time_abs':np.asarray(s['header_time_abs'],np.float64)[order],
            'time_original':np.asarray(s['time_original'],np.float64)[order],
            'force_native':np.asarray(s['force_native'],np.float64)[order],
            'frame_id':np.asarray(s['frame_id'],dtype='<U128')[order],
            'clock_source':np.asarray(s['clock_source'],dtype='<U8')[order],
        }
    return {'t0_sec':t0_sec,'bag_path':str(bag_path),'topic_types':msgtypes,'legs':out}


def native_force_audit(native,grid_original,run_id):
    rows=[]
    for leg,s in native['legs'].items():
        t=s['time_original']; dt=np.diff(t); h=s['header_time_abs']; b=s['bag_time_abs']
        finite=np.isfinite(h); delta=1000*(h[finite]-b[finite])
        frames=sorted(set(map(str,s['frame_id'])))
        rows.append({'run_id':run_id,'leg':leg,'samples':len(t),
                     'frame_ids':'|'.join(repr(x) for x in frames),
                     'empty_frame_fraction':float(np.mean(s['frame_id']=='')),
                     'median_dt_ms':float(1000*np.median(dt)),
                     'p99_dt_ms':float(1000*np.quantile(dt,.99)),
                     'nonpositive_dt':int((dt<=0).sum()),
                     'header_fraction':float(finite.mean()),
                     'header_minus_bag_ms':float(np.median(delta)) if len(delta) else np.nan,
                     'header_bag_jitter_ms':float(np.std(delta)) if len(delta) else np.nan,
                     'grid_start_margin_ms':float(1000*(grid_original[0]-t[0])),
                     'grid_end_margin_ms':float(1000*(t[-1]-grid_original[-1]))})
    return pd.DataFrame(rows)


def run_force_audit(spec):
    require_local_headroom()
    cache=locate_physics_cache(spec)
    if cache is None: raise FileNotFoundError(f"{spec['run_id']}: physics cache missing")
    _,grid=physics_grid_from_cache(cache)
    bag=raw_run_path(spec)/'rosbags'/f"{spec['run']}.bag"
    native=read_native_force_exact(bag)
    table=native_force_audit(native,grid,spec['run_id'])
    out=AUDIT_ROOT/spec['run_id']; out.mkdir(parents=True,exist_ok=True)
    table.to_csv(out/'native_force_clock_frame.csv',index=False)
    atomic_write_json(out/'native_force_metadata.json',{
        'pipeline_version':STRICT_PIPELINE_VERSION,'run_id':spec['run_id'],'created_utc':utc_now(),
        'bag_path':str(bag),'bag_sha256':file_sha256(bag),'physics_cache':str(cache),
        'physics_cache_sha256':file_sha256(cache),'t0_sec':native['t0_sec'],
        'topic_types':native['topic_types'],'audit_pass_for_clock':bool(
            (table.nonpositive_dt==0).all() and (table.grid_start_margin_ms>=0).all() and
            (table.grid_end_margin_ms>=0).all()),
        'force_frame_resolved_from_message':bool((table.empty_frame_fraction==0).all()),
    })
    display(table)
    return native,table


NATIVE_FORCE_CACHE={}

# %% notebook cell 36
# STRICT S5 — run-local heading frame, pose interpolation, frame conversion, and anti-aliased labels
def load_camera_pose_stream(run_root, camera='lcam_front'):
    ensure_python_packages([('scipy','scipy')])
    run_root=Path(run_root)
    times=np.loadtxt(run_root/'imu'/'cam_time.txt',dtype=np.float64).reshape(-1)
    poses=np.loadtxt(run_root/f'pose_{camera}.txt',dtype=np.float64)
    if poses.ndim!=2 or poses.shape[1]<7: raise ValueError(f'invalid {camera} pose array {poses.shape}')
    n=min(len(times),len(poses)); times=times[:n]; poses=poses[:n,:7]
    if not np.all(np.diff(times)>0): raise ValueError('camera clock is not strictly monotone')
    q=poses[:,3:7]; qnorm=np.linalg.norm(q,axis=1)
    if np.max(np.abs(qnorm-1))>1e-3: raise ValueError('camera quaternion is not normalized')
    return times,poses


def interp_pose(query_t,pose_t,poses):
    from scipy.spatial.transform import Rotation,Slerp
    query_t=np.asarray(query_t,np.float64); pose_t=np.asarray(pose_t,np.float64)
    if query_t.min()<pose_t[0] or query_t.max()>pose_t[-1]:
        raise ValueError(f'pose does not cover query clock: query [{query_t.min():.6f},{query_t.max():.6f}], '
                         f'pose [{pose_t[0]:.6f},{pose_t[-1]:.6f}]')
    pos=np.stack([np.interp(query_t,pose_t,poses[:,a]) for a in range(3)],axis=1)
    rot=Slerp(pose_t,Rotation.from_quat(poses[:,3:7]))(query_t).as_matrix()
    return rot.astype(np.float64),pos.astype(np.float64)


def heading_transform(first_R_WC):
    forward=np.asarray(first_R_WC[:,0],float); forward[2]=0
    if np.linalg.norm(forward)<1e-8: raise ValueError('cannot define heading from vertical forward axis')
    forward/=np.linalg.norm(forward)
    # With world NED and H-z up, H-y must point left for a right-handed frame.
    left=np.array([forward[1],-forward[0],0.],float)
    up=np.array([0.,0.,-1.],float)  # verified world NED -> run-local up
    R_H_W=np.stack([forward,left,up],axis=0)
    if (not np.allclose(R_H_W@R_H_W.T,np.eye(3),atol=1e-7)
            or not np.isclose(np.linalg.det(R_H_W),1.,atol=1e-7)):
        raise AssertionError('heading frame must be orthonormal and right-handed')
    return R_H_W


def pose_context(run_root,calibration,query_t,fixed_frame=None):
    pose_t,poses=load_camera_pose_stream(run_root,'lcam_front')
    R_WC,p_WC=interp_pose(query_t,pose_t,poses)
    R_C_B=np.asarray(calibration['R_C_B'],float); t_C_B=np.asarray(calibration['t_C_B_m'],float)
    R_WB=np.einsum('nij,jk->nik',R_WC,R_C_B)
    p_WB=p_WC+np.einsum('nij,j->ni',R_WC,t_C_B)
    if fixed_frame is None:
        R_H_W=heading_transform(interp_pose(np.array([query_t[0]]),pose_t,poses)[0][0])
        p_W0=p_WB[0].copy()
    else:
        R_H_W=np.asarray(fixed_frame['R_H_W'],float)
        p_W0=np.asarray(fixed_frame['p_W0'],float)
    R_HB=np.einsum('ij,njk->nik',R_H_W,R_WB)
    p_HB=np.einsum('ij,nj->ni',R_H_W,p_WB-p_W0)
    return {'R_WB':R_WB,'p_WB':p_WB,'R_H_W':R_H_W,'R_HB':R_HB,'p_HB':p_HB,
            'p_W0':p_W0,'pose_t':pose_t,'poses':poses}


def apply_declared_axis_map(force,leg_contract):
    force=np.asarray(force,float)
    p=np.asarray(leg_contract['axis_permutation'],int); s=np.asarray(leg_contract['axis_sign'],float)
    return force[:,p]*s[None,:]


def semantic_frame_class(name):
    s=str(name).lower().lstrip('/')
    if not s: return 'empty'
    if any(x in s for x in ('world','odom','map','ned')): return 'world_ned'
    if any(x in s for x in ('base','body','trunk','imu')): return 'base'
    if any(x in s for x in ('foot','toe','contact','lf_','rf_','lh_','rh_','fl_','fr_','rl_','rr_')): return 'foot'
    return 'unrecognized'


def assert_message_frames_consistent(native,force_contract):
    for leg in LEG_NAMES:
        ids=sorted(set(map(str,native['legs'][leg]['frame_id']))-set(['']))
        if not ids: continue  # external contract is mandatory for the supplied empty-header case.
        classes={semantic_frame_class(x) for x in ids}; declared=force_contract['per_leg'][leg]['source_frame']
        if classes!={declared}: raise RuntimeError(f'{leg}: message frame IDs {ids} imply {classes}, contract declares {declared}')


def transform_force_leg_to_H(force,t_original,leg,run_root,force_contract,calibration,
                             robot=None,q_interp=None,fixed_frame=None):
    c=force_contract['per_leg'][leg]
    f=apply_declared_axis_map(force,c)
    if force_contract['force_direction']=='robot_on_environment': f=-f
    ctx=pose_context(run_root,calibration,np.asarray(t_original,float),fixed_frame=fixed_frame)
    frame=c['source_frame']
    if frame=='world_ned':
        return np.einsum('ij,nj->ni',ctx['R_H_W'],f),ctx
    if frame=='base':
        return np.einsum('nij,nj->ni',ctx['R_HB'],f),ctx
    if frame=='foot':
        if robot is None or q_interp is None: raise ValueError('foot-frame force requires verified FK at native timestamps')
        li=LEG_NAMES.index(leg); R_BF=np.stack([robot.foot_transforms(q)[0][li] for q in q_interp])
        R_HF=np.einsum('nij,njk->nik',ctx['R_HB'],R_BF)
        return np.einsum('nij,nj->ni',R_HF,f),ctx
    raise ValueError(f'{leg}: unsupported declared frame {frame!r}')


def hysteretic_contact(force,on_n=CONTACT_ON_N,off_n=CONTACT_OFF_N):
    mag=np.linalg.norm(force,axis=-1); out=np.zeros(len(mag),bool); state=False
    for i,x in enumerate(mag):
        state = (x>=off_n) if state else (x>=on_n)
        out[i]=state
    return out


def lowpass_to_grid(native,grid_original):
    from scipy.signal import butter,sosfiltfilt
    starts=[native[leg]['time_original'][0] for leg in LEG_NAMES]
    ends=[native[leg]['time_original'][-1] for leg in LEG_NAMES]
    lo=max(max(starts),float(grid_original[0])); hi=min(min(ends),float(grid_original[-1]))
    if lo>grid_original[0]+1e-6 or hi<grid_original[-1]-1e-6:
        raise ValueError('native force does not cover every cached grid_original timestamp; no extrapolation allowed')
    uniform_t=np.arange(grid_original[0],grid_original[-1]+0.5/FORCE_UNIFORM_HZ,1/FORCE_UNIFORM_HZ)
    raw=np.empty((len(uniform_t),4,3),np.float64)
    for li,leg in enumerate(LEG_NAMES):
        s=native[leg]
        for a in range(3): raw[:,li,a]=np.interp(uniform_t,s['time_original'],s['force_H'][:,a])
    sos=butter(8,FORCE_LOWPASS_HZ,btype='low',fs=FORCE_UNIFORM_HZ,output='sos')
    filt=sosfiltfilt(sos,raw,axis=0,padtype='odd')
    aligned=np.empty((len(grid_original),4,3),np.float64)
    causal=np.empty((len(grid_original),4,3),np.float64)
    contact_grid=np.empty((len(grid_original),4),bool)
    for li in range(4):
        for a in range(3): aligned[:,li,a]=np.interp(grid_original,uniform_t,filt[:,li,a])
        # Deployable history channel: latest native observation at or before the
        # grid timestamp.  Never feed the symmetric zero-phase target filter to
        # the causal history encoder.
        s=native[LEG_NAMES[li]]; ids_causal=np.searchsorted(s['time_original'],grid_original,side='right')-1
        if np.any(ids_causal<0): raise ValueError(f'{LEG_NAMES[li]}: causal force history begins after the physics grid')
        causal[:,li]=s['force_H'][ids_causal]
        contact_grid[:,li]=hysteretic_contact(s['force_H'])[ids_causal]
    contact_native=np.stack([hysteretic_contact(raw[:,li]) for li in range(4)],axis=1)
    return {'uniform_t':uniform_t,'force_raw_H':raw,'force_lowpass_H':filt,
            'force_grid_H':aligned,'force_causal_grid_H':causal,
            'contact_native':contact_native,'contact_grid':contact_grid}


def event_targets_from_native(force_pack,grid_original):
    t=force_pack['uniform_t']; f=force_pack['force_raw_H']; c=force_pack['contact_native']
    n_grid=len(grid_original); impulse_grid=np.zeros((n_grid,4,3),np.float32)
    events=[]; waveform_n=int(round((IMPACT_PRE_S+IMPACT_POST_S)*IMPACT_WAVEFORM_HZ))+1
    wave_t=np.linspace(-IMPACT_PRE_S,IMPACT_POST_S,waveform_n)
    waves=[]
    for li,leg in enumerate(LEG_NAMES):
        ids=np.where((~c[:-1,li])&c[1:,li])[0]+1
        for event_id,j in enumerate(ids):
            te=float(t[j]); sample_t=te+wave_t
            if sample_t[0]<t[0] or sample_t[-1]>t[-1]: continue
            wave=np.stack([np.interp(sample_t,t,f[:,li,a]) for a in range(3)],axis=1)
            J=np.trapz(wave,sample_t,axis=0)
            k=int(np.argmin(np.abs(grid_original-te)))
            if abs(grid_original[k]-te)<=0.5/CFG.sample_hz+1e-9: impulse_grid[k,li]=J
            events.append({'leg':leg,'leg_index':li,'event_id':event_id,'waveform_row':len(waves),
                           't_touchdown_original':te,
                           't_touchdown_cropped':te-float(grid_original[0]),'grid_index':k,
                           'Jx':J[0],'Jy':J[1],'Jz':J[2],'J_norm':np.linalg.norm(J)})
            waves.append(wave.astype(np.float32))
    columns=['leg','leg_index','event_id','waveform_row','t_touchdown_original','t_touchdown_cropped',
             'grid_index','Jx','Jy','Jz','J_norm']
    table=pd.DataFrame(events,columns=columns)
    wave_arr=np.stack(waves) if waves else np.empty((0,waveform_n,3),np.float32)
    return impulse_grid,table,wave_t.astype(np.float32),wave_arr


def force_label_audits(force_pack,grid_original,contact_grid,events):
    f=force_pack['force_grid_H']; dt=np.diff(grid_original); delta=np.abs(np.diff(f,axis=0))
    rows=[]
    for li,leg in enumerate(LEG_NAMES):
        for a,axis in enumerate('xyz'):
            x=delta[:,li,a]
            rows.append({'leg':leg,'axis':axis,'adjacent_mae_n':float(x.mean()),
                         'adjacent_p95_n':float(np.quantile(x,.95)),'adjacent_max_n':float(x.max()),
                         'slope_p95_n_s':float(np.quantile(x/dt,.95)),
                         'timing_for_4n_ms':float(4000/max(np.quantile(x/dt,.95),1e-9))})
    # Filtering must approximately conserve registered impact impulse.
    impulse_rows=[]
    for _,e in events.iterrows():
        li=int(e.leg_index); te=float(e.t_touchdown_original)
        lo=te-IMPACT_PRE_S; hi=te+IMPACT_POST_S
        raw=np.array([np.trapz(np.interp(np.linspace(lo,hi,25),force_pack['uniform_t'],force_pack['force_raw_H'][:,li,a]),
                                 np.linspace(lo,hi,25)) for a in range(3)])
        fil=np.array([np.trapz(np.interp(np.linspace(lo,hi,25),force_pack['uniform_t'],force_pack['force_lowpass_H'][:,li,a]),
                                 np.linspace(lo,hi,25)) for a in range(3)])
        impulse_rows.append({'leg':e.leg,'t':te,'raw_norm':np.linalg.norm(raw),'filtered_norm':np.linalg.norm(fil),
                             'relative_vector_change':np.linalg.norm(fil-raw)/max(np.linalg.norm(raw),1e-6)})
    return pd.DataFrame(rows),pd.DataFrame(impulse_rows)

# %% notebook cell 38
# STRICT S6 — Pinocchio FK/Jacobians, COM, and auditable mechanics arrays
class VerifiedAnyMalModel:
    def __init__(self,urdf_path,calibration_contract):
        ensure_python_packages([('pinocchio','pin')])
        import pinocchio as pin
        self.pin=pin; self.urdf_path=Path(urdf_path)
        self.fixed=pin.buildModelFromUrdf(str(self.urdf_path))
        self.fixed_data=self.fixed.createData()
        self.mass=float(sum(i.mass for i in self.fixed.inertias))
        if not np.isfinite(self.mass) or self.mass<=0: raise ValueError('URDF mass invalid')

        # Resolve each canonical 1-DoF joint by exact name; aliases only exchange FL/LF and RL/LH prefixes.
        available={str(n):i for i,n in enumerate(self.fixed.names)}
        self.jids=[]; self.qidx=[]; self.vidx=[]; resolved=[]
        for canonical in CANONICAL_JOINT_NAMES:
            name=str(calibration_contract['joint_names'][canonical])
            if name not in available: raise ValueError(f'{canonical}: contracted URDF joint {name!r} missing')
            jid=available[name]; j=self.fixed.joints[jid]
            if j.nq!=1 or j.nv!=1: raise ValueError(f'{name} is not a 1-DoF joint')
            self.jids.append(jid); self.qidx.append(j.idx_q); self.vidx.append(j.idx_v); resolved.append(name)
        self.resolved_joint_names=tuple(resolved)

        frame_names={str(f.name):i for i,f in enumerate(self.fixed.frames)}
        self.foot_frame_ids={}
        for leg in LEG_NAMES:
            name=str(calibration_contract['foot_frame_names'][leg])
            if name not in frame_names: raise ValueError(f'{leg}: contracted foot frame {name!r} missing')
            self.foot_frame_ids[leg]=frame_names[name]

        q0=self.q_fixed(np.zeros(12)); v0=np.zeros(self.fixed.nv)
        pin.forwardKinematics(self.fixed,self.fixed_data,q0,v0)
        pin.updateFramePlacements(self.fixed,self.fixed_data)
        pin.centerOfMass(self.fixed,self.fixed_data,q0,v0)
        self.com_neutral_B=np.asarray(self.fixed_data.com[0],float).copy()
        pin.ccrba(self.fixed,self.fixed_data,q0,v0)
        self.inertia_neutral_B=np.asarray(self.fixed_data.Ig.inertia,float).copy()

    def q_fixed(self,q12):
        q=self.pin.neutral(self.fixed)
        for x,idx in zip(np.asarray(q12,float),self.qidx): q[idx]=x
        return q

    def v_fixed(self,dq12=None):
        v=np.zeros(self.fixed.nv)
        if dq12 is not None:
            for x,idx in zip(np.asarray(dq12,float),self.vidx): v[idx]=x
        return v

    def foot_transforms(self,q_leg):
        q12=np.asarray(q_leg,float).reshape(12); q=self.q_fixed(q12)
        self.pin.forwardKinematics(self.fixed,self.fixed_data,q)
        self.pin.updateFramePlacements(self.fixed,self.fixed_data)
        R=[]; p=[]
        for leg in LEG_NAMES:
            M=self.fixed_data.oMf[self.foot_frame_ids[leg]]
            R.append(np.asarray(M.rotation,float)); p.append(np.asarray(M.translation,float))
        return np.stack(R),np.stack(p)

    def sample_mechanics(self,q12,dq12=None):
        q=self.q_fixed(q12); v=self.v_fixed(dq12)
        self.pin.forwardKinematics(self.fixed,self.fixed_data,q,v)
        self.pin.computeJointJacobians(self.fixed,self.fixed_data,q)
        self.pin.updateFramePlacements(self.fixed,self.fixed_data)
        self.pin.centerOfMass(self.fixed,self.fixed_data,q,v)
        com=np.asarray(self.fixed_data.com[0],float).copy()
        feet=[]; rotations=[]; Jlegs=[]
        for li,leg in enumerate(LEG_NAMES):
            fid=self.foot_frame_ids[leg]; M=self.fixed_data.oMf[fid]
            J6=self.pin.getFrameJacobian(self.fixed,self.fixed_data,fid,self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            cols=[self.vidx[3*li+k] for k in range(3)]
            feet.append(np.asarray(M.translation,float)); rotations.append(np.asarray(M.rotation,float))
            Jlegs.append(np.asarray(J6[:3,cols],float))
        return com,np.stack(feet),np.stack(rotations),np.stack(Jlegs)


def interpolate_q_to_times(grid,q_leg,query_t):
    q=np.asarray(q_leg,float).reshape(len(grid),12)
    flat=np.stack([np.interp(query_t,grid,q[:,j]) for j in range(12)],axis=1)
    return flat.reshape(len(query_t),4,3)


def build_mechanics_arrays(data,grid_original,run_root,calibration,robot):
    P=np.asarray(data['P'],np.float32); n=len(P)
    pose=pose_context(run_root,calibration,grid_original)
    q=P[:,:,:3].reshape(n,12); dq=P[:,:,3:6].reshape(n,12)
    com_B=np.empty((n,3),np.float32); foot_B=np.empty((n,4,3),np.float32)
    foot_R_B=np.empty((n,4,3,3),np.float32); J_leg_B=np.empty((n,4,3,3),np.float32)
    for i in range(n):
        c,p,R,J=robot.sample_mechanics(q[i],dq[i])
        com_B[i]=c; foot_B[i]=p; foot_R_B[i]=R; J_leg_B[i]=J
        if (i+1)%5000==0: print(f'  mechanics {i+1}/{n}')
    com_H=pose['p_HB']+np.einsum('nij,nj->ni',pose['R_HB'],com_B)
    foot_H=pose['p_HB'][:,None,:]+np.einsum('nij,nlj->nli',pose['R_HB'],foot_B)
    foot_r_H=foot_H-com_H[:,None,:]
    # Jacobians map leg joint rates to foot linear velocity in B; rotate rows into H.
    J_leg_H=np.einsum('nij,nljk->nlik',pose['R_HB'],J_leg_B)
    return {'R_H_W':pose['R_H_W'].astype(np.float32),'p_W0':pose['p_W0'].astype(np.float64),
            'R_HB':pose['R_HB'].astype(np.float32),'p_HB':pose['p_HB'].astype(np.float32),
            'com_H':com_H.astype(np.float32),'foot_world':foot_H.astype(np.float32),
            'foot_r_H':foot_r_H.astype(np.float32),'J_leg_H':J_leg_H.astype(np.float32),
            'mass_kg':np.array(robot.mass,np.float32),
            'inertia_neutral_B':robot.inertia_neutral_B.astype(np.float32)}


def observed_centroidal_wrench_target(data,mechanics,signal_contract):
    # Under the verified contract, an accelerometer measures specific force in B:
    # m * specific_force is the net non-gravitational force, dominated here by contacts.
    B=np.asarray(data['B'],np.float32); t=np.asarray(data['time'],float)
    specific_B=B[:,:3]; omega_B=B[:,3:6]
    alpha_B=causal_rate(omega_B,t,clip=300.)
    I=np.asarray(mechanics['inertia_neutral_B'],np.float32)
    force_B=float(mechanics['mass_kg'])*specific_B
    moment_B=alpha_B@I.T+np.cross(omega_B,omega_B@I.T)
    R=np.asarray(mechanics['R_HB'],np.float32)
    force_H=np.einsum('nij,nj->ni',R,force_B)
    moment_H=np.einsum('nij,nj->ni',R,moment_B)
    return np.c_[force_H,moment_H].astype(np.float32)

# %% notebook cell 40
# STRICT S7 — RGB-D deprojection and resumable cross-view voxel-fusion events
def read_metric_depth(path):
    ensure_python_packages([('cv2','opencv-python-headless')])
    import cv2
    raw=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if raw is None: raise ValueError(f'cannot read depth: {path}')
    if raw.dtype==np.uint8 and raw.ndim==3 and raw.shape[2]==4:
        depth=np.ascontiguousarray(raw).view('<f4').reshape(raw.shape[:2])
    elif raw.dtype==np.float32 and raw.ndim==2: depth=raw
    elif np.issubdtype(raw.dtype,np.integer): depth=raw.astype(np.float32)/1000.
    else: raise ValueError(f'unrecognized depth encoding {raw.dtype} {raw.shape}')
    depth=np.asarray(depth,np.float32); depth[~np.isfinite(depth)]=0
    return depth


def read_rgb(path):
    ensure_python_packages([('cv2','opencv-python-headless')])
    import cv2
    bgr=cv2.imread(str(path),cv2.IMREAD_COLOR)
    if bgr is None: raise ValueError(f'cannot read RGB: {path}')
    return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB).astype(np.float32)/255.


def camera_file_map(run_root,cam,modality):
    root=Path(run_root)/(f'image_{cam}' if modality=='image' else f'depth_{cam}')
    if not root.exists(): return {}
    out={}
    for p in root.iterdir():
        if not p.is_file(): continue
        m=re.match(r'^(\d+)_',p.name)
        if m: out[int(m.group(1))]=p
    return out


def deproject_camera(depth,intr,pixel_stride=GEOMETRY_PIXEL_STRIDE):
    s=int(pixel_stride); rows=np.arange(0,depth.shape[0],s); cols=np.arange(0,depth.shape[1],s)
    v,u=np.meshgrid(rows,cols,indexing='ij'); Z=depth[v,u]
    good=np.isfinite(Z)&(Z>.08)&(Z<GEOMETRY_MAX_RANGE_M)
    Z=Z[good]; u=u[good]; v=v[good]
    # Verified Tartan camera convention: x forward, y right, z down.
    p=np.c_[Z,(u-float(intr['cx']))*Z/float(intr['fx']),(v-float(intr['cy']))*Z/float(intr['fy'])]
    return p.astype(np.float32),good,(v,u)


def point_normals_from_depth(depth,intr,pixel_stride=GEOMETRY_PIXEL_STRIDE):
    # Normals are computed from actual 3-D neighboring rays, not depth gradients alone.
    s=int(pixel_stride); d=depth[::s,::s]
    vv,uu=np.mgrid[0:d.shape[0],0:d.shape[1]]; u=uu*s; v=vv*s
    p=np.stack([d,(u-intr['cx'])*d/intr['fx'],(v-intr['cy'])*d/intr['fy']],axis=-1)
    dx=np.zeros_like(p); dy=np.zeros_like(p)
    dx[:,1:-1]=p[:,2:]-p[:,:-2]; dy[1:-1]=p[2:]-p[:-2]
    n=np.cross(dx,dy); n/=np.maximum(np.linalg.norm(n,axis=-1,keepdims=True),1e-8)
    return n.astype(np.float32)


def aggregate_frame_voxels(xyz,normal,rgb,camera_index,measurement_var=None,voxel=CFG.patch_voxel_m):
    key=np.floor(xyz/float(voxel)).astype(np.int32)
    order=np.lexsort((key[:,2],key[:,1],key[:,0])); key=key[order]
    xyz=xyz[order]; normal=normal[order]; rgb=rgb[order]
    measurement_var=np.zeros_like(xyz) if measurement_var is None else np.asarray(measurement_var,np.float32)[order]
    start=np.r_[0,1+np.where(np.any(np.diff(key,axis=0),axis=1))[0]]; end=np.r_[start[1:],len(key)]
    n=len(start); out_key=key[start]; out_xyz=np.empty((n,3),np.float32); out_n=np.empty((n,3),np.float32)
    out_rgb=np.empty((n,3),np.float32); out_var=np.empty((n,3),np.float32); out_count=np.empty(n,np.int32)
    for j,(a,b) in enumerate(zip(start,end)):
        out_xyz[j]=xyz[a:b].mean(0); nn=normal[a:b].mean(0); out_n[j]=nn/max(np.linalg.norm(nn),1e-8)
        out_rgb[j]=rgb[a:b].mean(0); out_var[j]=xyz[a:b].var(0)+measurement_var[a:b].mean(0); out_count[j]=b-a
    camera_mask=np.full(n,1<<int(camera_index),np.uint8)
    return {'key':out_key,'xyz':out_xyz,'normal':out_n,'rgb':out_rgb,'xyz_var':out_var,
            'sample_count':out_count,'camera_mask':camera_mask}


def merge_multiview_voxels(parts):
    if not parts: return None
    key=np.concatenate([x['key'] for x in parts]); xyz=np.concatenate([x['xyz'] for x in parts])
    normal=np.concatenate([x['normal'] for x in parts]); rgb=np.concatenate([x['rgb'] for x in parts])
    var=np.concatenate([x['xyz_var'] for x in parts]); count=np.concatenate([x['sample_count'] for x in parts])
    masks=np.concatenate([x['camera_mask'] for x in parts])
    order=np.lexsort((key[:,2],key[:,1],key[:,0])); key=key[order]
    xyz=xyz[order]; normal=normal[order]; rgb=rgb[order]; var=var[order]; count=count[order]; masks=masks[order]
    start=np.r_[0,1+np.where(np.any(np.diff(key,axis=0),axis=1))[0]]; end=np.r_[start[1:],len(key)]
    rows={k:[] for k in ('key','xyz','normal','rgb','xyz_var','sample_count','camera_mask')}
    for a,b in zip(start,end):
        w=count[a:b].astype(float); w/=w.sum(); mu=(xyz[a:b]*w[:,None]).sum(0)
        total_var=(var[a:b]*w[:,None]).sum(0)+((xyz[a:b]-mu)**2*w[:,None]).sum(0)
        nn=(normal[a:b]*w[:,None]).sum(0); nn/=max(np.linalg.norm(nn),1e-8)
        rows['key'].append(key[a]); rows['xyz'].append(mu); rows['normal'].append(nn)
        rows['rgb'].append((rgb[a:b]*w[:,None]).sum(0)); rows['xyz_var'].append(total_var)
        rows['sample_count'].append(int(count[a:b].sum())); rows['camera_mask'].append(int(np.bitwise_or.reduce(masks[a:b])))
    return {k:np.asarray(v,dtype=np.int32 if k=='key' else np.int64 if k=='sample_count' else np.uint8 if k=='camera_mask' else np.float32)
            for k,v in rows.items()}


def latest_state_from_geometry_chunks(chunk_paths):
    state={}
    for p in sorted(map(Path,chunk_paths)):
        with np.load(p,allow_pickle=False) as z:
            for i,k in enumerate(z['key']):
                state[tuple(map(int,k))]={name:np.asarray(z[name][i]).copy() for name in
                    ('xyz','normal','rgb','xyz_var','count','camera_mask','time')}
    return state


def fuse_voxel_observation(state,obs,timestamp):
    events={k:[] for k in ('key','xyz','normal','rgb','xyz_var','count','camera_mask','time')}
    for i,key_arr in enumerate(obs['key']):
        key=tuple(map(int,key_arr)); m=int(obs['sample_count'][i]); x=obs['xyz'][i]; v=obs['xyz_var'][i]
        if key not in state:
            new={'xyz':x.astype(np.float32),'normal':obs['normal'][i].astype(np.float32),
                 'rgb':obs['rgb'][i].astype(np.float32),'xyz_var':v.astype(np.float32),
                 'count':np.int64(m),'camera_mask':np.uint8(obs['camera_mask'][i]),'time':np.float64(timestamp)}
        else:
            old=state[key]; n=int(old['count']); total=n+m
            delta=x-old['xyz']; mean=old['xyz']+delta*(m/total)
            pooled=(n*old['xyz_var']+m*v+(delta**2)*(n*m/total))/total
            nn=old['normal']*n+obs['normal'][i]*m; nn/=max(np.linalg.norm(nn),1e-8)
            rgb=(old['rgb']*n+obs['rgb'][i]*m)/total
            new={'xyz':mean.astype(np.float32),'normal':nn.astype(np.float32),'rgb':rgb.astype(np.float32),
                 'xyz_var':pooled.astype(np.float32),'count':np.int64(total),
                 'camera_mask':np.uint8(int(old['camera_mask'])|int(obs['camera_mask'][i])),
                 'time':np.float64(timestamp)}
        state[key]=new; events['key'].append(key_arr)
        for name in ('xyz','normal','rgb','xyz_var','count','camera_mask','time'): events[name].append(new[name])
    return {k:np.asarray(v,dtype=np.int32 if k=='key' else None) for k,v in events.items()}


def geometry_frame_groups(run_root,calibration):
    pose_t=np.loadtxt(Path(run_root)/'imu'/'cam_time.txt',dtype=np.float64).reshape(-1)
    groups=[]
    file_maps={cam:{m:camera_file_map(run_root,cam,m) for m in ('image','depth')} for cam in CAMERA_NAMES}
    common_ids=sorted(set().union(*(set(file_maps[c]['depth']) for c in CAMERA_NAMES)))
    for fid in common_ids[::GEOMETRY_FRAME_STRIDE]:
        if fid>=len(pose_t): continue
        cams=[c for c in CAMERA_NAMES if fid in file_maps[c]['depth'] and fid in file_maps[c]['image']]
        if cams: groups.append({'frame_id':fid,'time_original':float(pose_t[fid]),'cameras':cams,'files':file_maps})
    return groups


def build_geometry_checkpoint(spec,calibration):
    run_root=raw_run_path(spec); out=PREPARED_ROOT/spec['run_id']/'geometry'
    chunks=out/'chunks'; chunks.mkdir(parents=True,exist_ok=True)
    complete=out/'COMPLETE.json'
    if complete.exists() and verify_geometry_checkpoint(out,raise_on_error=False)['pass']:
        print(spec['run_id'],'geometry already verified; reusing'); return out
    cache=locate_physics_cache(spec); data,grid=physics_grid_from_cache(cache)
    frame_ctx=pose_context(run_root,calibration,np.array([grid[0]],float))
    R_H_W=frame_ctx['R_H_W']; p_W0=frame_ctx['p_W0']
    groups=[g for g in geometry_frame_groups(run_root,calibration) if grid[0]<=g['time_original']<=grid[-1]]
    if not groups: raise FileNotFoundError(f"{spec['run_id']}: no synchronized RGB-D frame groups")
    existing=sorted(chunks.glob('chunk_*.npz')); done_frames=0
    if existing:
        meta=json.load(open(out/'progress.json')); done_frames=int(meta['frames_complete'])
        state=latest_state_from_geometry_chunks(existing); print('resuming geometry at frame',done_frames)
    else: state={}
    chunk_events=[]; chunk_index=len(existing)
    for gi,g in enumerate(groups[done_frames:],start=done_frames):
        parts=[]
        base_ctx=pose_context(run_root,calibration,np.array([g['time_original']],float),fixed_frame=frame_ctx)
        p_HB=base_ctx['p_HB'][0]
        for ci,cam in enumerate(CAMERA_NAMES):
            if cam not in g['cameras']: continue
            pose_file=run_root/f'pose_{cam}.txt'; poses=np.loadtxt(pose_file,dtype=np.float64)
            if g['frame_id']>=len(poses): continue
            pose=poses[g['frame_id']]; from scipy.spatial.transform import Rotation
            R_WC=Rotation.from_quat(pose[3:7]).as_matrix(); p_WC=pose[:3]
            depth=read_metric_depth(g['files'][cam]['depth'][g['frame_id']]); rgb_im=read_rgb(g['files'][cam]['image'][g['frame_id']])
            intr=calibration['camera_intrinsics'][cam]
            expected=(int(intr['height']),int(intr['width']))
            if depth.shape!=expected or rgb_im.shape[:2]!=expected:
                raise ValueError(f'{cam} frame {g["frame_id"]}: image shape does not match verified intrinsics: '
                                 f'depth={depth.shape}, rgb={rgb_im.shape[:2]}, expected={expected}')
            pc,good,_=deproject_camera(depth,intr); normals_grid=point_normals_from_depth(depth,intr)
            nc=normals_grid.reshape(-1,3)[good.reshape(-1)]
            rgb=rgb_im[::GEOMETRY_PIXEL_STRIDE,::GEOMETRY_PIXEL_STRIDE].reshape(-1,3)[good.reshape(-1)]
            xyz_W=p_WC[None]+pc@R_WC.T; xyz_H=(xyz_W-p_W0[None])@R_H_W.T
            normal_H=nc@R_WC.T@R_H_W.T
            valid=(np.linalg.norm(xyz_H[:,:2]-p_HB[None,:2],axis=1)<=GEOMETRY_LOCAL_RADIUS_M)&(normal_H[:,2]>=GEOMETRY_MIN_UP_NORMAL)
            if valid.any():
                sigma=float(intr['depth_sigma_a_m'])+float(intr['depth_sigma_b_per_m'])*pc[valid,0]**2
                measurement_var=np.repeat((sigma*sigma)[:,None],3,axis=1)
                parts.append(aggregate_frame_voxels(xyz_H[valid],normal_H[valid],rgb[valid],ci,measurement_var))
        obs=merge_multiview_voxels(parts)
        if obs is not None and len(obs['key']): chunk_events.append(fuse_voxel_observation(state,obs,g['time_original']))
        flush=(len(chunk_events)>=GEOMETRY_CHUNK_FRAMES) or (gi==len(groups)-1)
        if flush and chunk_events:
            merged={k:np.concatenate([e[k] for e in chunk_events],axis=0) for k in chunk_events[0]}
            cp=chunks/f'chunk_{chunk_index:05d}.npz'; atomic_savez(cp,**merged); chunk_index+=1; chunk_events=[]
            atomic_write_json(out/'progress.json',{'frames_complete':gi+1,'frames_total':len(groups),'updated_utc':utc_now()})
            print(f"  {spec['run_id']} geometry {gi+1}/{len(groups)} | voxels={len(state):,}")
            gc.collect()
    chunk_files=sorted(chunks.glob('chunk_*.npz'))
    atomic_write_json(complete,{'schema':'V10R_CAUSAL_VOXEL_FUSION_V1','run_id':spec['run_id'],
        'created_utc':utc_now(),'frames_total':len(groups),'final_voxels':len(state),
        'voxel_m':CFG.patch_voxel_m,'pixel_stride':GEOMETRY_PIXEL_STRIDE,'frame_stride':GEOMETRY_FRAME_STRIDE,
        'R_H_W':R_H_W.tolist(),'p_W0':p_W0.tolist(),
        'chunks':[{'name':p.name,'bytes':p.stat().st_size,'sha256':file_sha256(p)} for p in chunk_files]})
    verify_geometry_checkpoint(out,raise_on_error=True)
    return out


def verify_geometry_checkpoint(path,raise_on_error=True):
    path=Path(path); errors=[]; cp=path/'COMPLETE.json'
    if not cp.exists(): errors.append('COMPLETE.json missing')
    else:
        meta=json.load(open(cp))
        for x in meta.get('chunks',[]):
            p=path/'chunks'/x['name']
            if not p.exists(): errors.append(f"missing {x['name']}")
            elif p.stat().st_size!=x['bytes'] or file_sha256(p)!=x['sha256']: errors.append(f"hash/size mismatch {x['name']}")
    out={'pass':not errors,'errors':errors}
    if errors and raise_on_error: raise RuntimeError('geometry checkpoint invalid: '+'; '.join(errors))
    return out

# %% notebook cell 42
# STRICT S8 — assemble one strict run and write an immutable prepared checkpoint
def force_balance_diagnostic(force_pack,robot_mass):
    f=force_pack['force_grid_H']; c=force_pack['contact_grid']; all_stance=c.all(axis=1)
    if all_stance.sum()<10: all_stance=c.sum(axis=1)>=3
    total=f.sum(axis=1); vert=total[:,2]
    return {'samples':int(all_stance.sum()),
            'positive_vertical_fraction':float(np.mean(vert[all_stance]>0)) if all_stance.any() else np.nan,
            'median_vertical_n':float(np.median(vert[all_stance])) if all_stance.any() else np.nan,
            'median_vertical_over_mg':float(np.median(vert[all_stance])/(robot_mass*9.81)) if all_stance.any() else np.nan,
            'median_lateral_over_vertical':float(np.median(np.linalg.norm(total[all_stance,:2],axis=1)/
                np.maximum(np.abs(vert[all_stance]),1e-6))) if all_stance.any() else np.nan}


def transform_native_force_run(native,data,grid,run_root,fc,cc,robot):
    fixed=pose_context(run_root,cc,np.array([grid[0]],float))
    need_foot=any(fc['per_leg'][leg]['source_frame']=='foot' for leg in LEG_NAMES)
    q_grid=np.asarray(data['P'])[:,:,:3]
    transformed={}
    for leg in LEG_NAMES:
        source=native['legs'][leg]
        tt=np.asarray(source['time_original']); a=max(0,np.searchsorted(tt,grid[0],side='right')-65)
        b=min(len(tt),np.searchsorted(tt,grid[-1],side='left')+66)
        if b-a<100: raise ValueError(f'{leg}: too few native force samples on the cropped physics interval')
        s={k:np.asarray(v)[a:b] for k,v in source.items()}; q_native=None
        if need_foot: q_native=interpolate_q_to_times(grid,q_grid,s['time_original'])
        f_H,_=transform_force_leg_to_H(s['force_native'],s['time_original'],leg,run_root,fc,cc,
                                        robot=robot,q_interp=q_native,fixed_frame=fixed)
        transformed[leg]={**s,'force_H':f_H}
    return transformed,fixed


def save_native_force_checkpoint(path,native,transformed,force_pack,wave_t,waves):
    arrays={'t0_sec':np.array(native['t0_sec'],np.float64),
            'uniform_t':force_pack['uniform_t'].astype(np.float64),
            'force_raw_H':force_pack['force_raw_H'].astype(np.float32),
            'force_lowpass_H':force_pack['force_lowpass_H'].astype(np.float32),
            'force_causal_grid_H':force_pack['force_causal_grid_H'].astype(np.float32),
            'contact_native':force_pack['contact_native'].astype(np.uint8),
            'impact_wave_time_s':wave_t.astype(np.float32),'impact_waveforms_H':waves.astype(np.float32)}
    for leg in LEG_NAMES:
        s=transformed[leg]
        for key in ('bag_time_abs','header_time_abs','time_original','force_native','force_H','frame_id','clock_source'):
            arrays[f'{leg}_{key}']=np.asarray(s[key])
    atomic_savez(path,**arrays)


def prepared_paths(run_id):
    root=PREPARED_ROOT/run_id
    return {'root':root,'core':root/'run_core.npz','native':root/'force_native_and_events.npz',
            'haptic':root/'haptic_events.npz','geometry_validation':root/'geometry_validation.json',
            'geometry_validation_samples':root/'geometry_validation_samples.csv',
            'contracts_snapshot':root/'contracts_snapshot.json',
            'events':root/'touchdown_events.csv','force_audit':root/'force_label_audit.csv',
            'impulse_audit':root/'impulse_filter_audit.csv','manifest':root/'prepared_manifest.json',
            'geometry':root/'geometry'}


def build_haptic_events(grid,data,force_causal_grid,contact,mechanics,events):
    t=np.asarray(grid,float); foot=np.asarray(mechanics['foot_world']); vel=causal_rate(foot,t,clip=8.)
    E=np.asarray(data['E']); rows={k:[] for k in ('time_original','leg_index','xyz','force','impulse','slip','torque_error','contact_duration')}
    duration=np.zeros(4,float); duration_hist=np.zeros((len(t),4),np.float32)
    for i in range(len(t)):
        if i: duration=np.where(contact[i],np.where(contact[i-1],duration+(t[i]-t[i-1]),0.),0.)
        duration_hist[i]=duration
        if i%HAPTIC_STRIDE: continue
        for l in range(4):
            if not contact[i,l]: continue
            rows['time_original'].append(t[i]); rows['leg_index'].append(l); rows['xyz'].append(foot[i,l])
            rows['force'].append(force_causal_grid[i,l]); rows['impulse'].append(np.zeros(3,np.float32))
            rows['slip'].append(np.linalg.norm(vel[i,l])); rows['torque_error'].append(np.linalg.norm(E[i,l]))
            rows['contact_duration'].append(duration[l])
    # A touchdown impulse integrates through t_touchdown + IMPACT_POST_S and is
    # therefore inserted only when that complete outcome would be observable.
    for _,event in events.iterrows():
        available=float(event.t_touchdown_original)+IMPACT_POST_S
        i=int(np.searchsorted(t,available,side='left')); l=int(event.leg_index)
        if i>=len(t): continue
        touchdown_i=int(np.clip(event.grid_index,0,len(t)-1))
        rows['time_original'].append(available); rows['leg_index'].append(l)
        rows['xyz'].append(foot[touchdown_i,l]); rows['force'].append(force_causal_grid[i,l])
        rows['impulse'].append(np.array([event.Jx,event.Jy,event.Jz],np.float32))
        rows['slip'].append(np.linalg.norm(vel[i,l])); rows['torque_error'].append(np.linalg.norm(E[i,l]))
        rows['contact_duration'].append(duration_hist[i,l])
    return {k:np.asarray(v,dtype=np.int16 if k=='leg_index' else np.float64 if k=='time_original' else np.float32) for k,v in rows.items()}


def prepare_strict_run(spec):
    require_local_headroom(); fc,cc,sc,_=load_and_validate_contracts(require_approved=True)
    paths=prepared_paths(spec['run_id']); paths['root'].mkdir(parents=True,exist_ok=True)
    cache=locate_physics_cache(spec)
    if cache is None: raise FileNotFoundError(f"{spec['run_id']}: physics cache missing")
    data,grid=physics_grid_from_cache(cache); run_root=raw_run_path(spec)
    bag=run_root/'rosbags'/f"{spec['run']}.bag"
    if not bag.exists(): raise FileNotFoundError(f'raw bag required: {bag}')
    geom=verify_geometry_checkpoint(paths['geometry'],raise_on_error=False)
    if not geom['pass']:
        raise RuntimeError(f"{spec['run_id']}: geometry is incomplete. Run PIPELINE_STAGE='geometry' first: {geom['errors']}")
    geometry_validation=validate_geometry_against_heldout_frames(spec)
    if not geometry_validation['pass']:
        raise RuntimeError(f"{spec['run_id']}: held-out causal geometry validation failed: {geometry_validation}")

    robot=VerifiedAnyMalModel(cc['urdf_path'],cc)
    native=read_native_force_exact(bag)
    assert_message_frames_consistent(native,fc)
    audit=native_force_audit(native,grid,spec['run_id'])
    if not ((audit.nonpositive_dt==0).all() and (audit.grid_start_margin_ms>=-1e-3).all() and
            (audit.grid_end_margin_ms>=-1e-3).all()):
        raise RuntimeError('native clock audit failed; no extrapolation or reordered force labels allowed')
    transformed,fixed=transform_native_force_run(native,data,grid,run_root,fc,cc,robot)
    force_pack=lowpass_to_grid(transformed,grid)
    impulse_grid,events,wave_t,waves=event_targets_from_native(force_pack,grid)
    force_audit,impulse_audit=force_label_audits(force_pack,grid,force_pack['contact_grid'],events)
    mechanics=build_mechanics_arrays(data,grid,run_root,cc,robot)
    # Exact identity test: force and geometry must use the same fixed H transform.
    geom_meta=json.load(open(paths['geometry']/'COMPLETE.json'))
    if not np.allclose(mechanics['R_H_W'],np.asarray(geom_meta['R_H_W']),atol=1e-6):
        raise RuntimeError('force/mechanics and geometry heading rotations differ')
    if not np.allclose(mechanics['p_W0'],np.asarray(geom_meta['p_W0']),atol=1e-5):
        raise RuntimeError('force/mechanics and geometry origins differ')
    desired=observed_centroidal_wrench_target(data,mechanics,sc)
    haptic=build_haptic_events(grid,data,force_pack['force_causal_grid_H'],force_pack['contact_grid'],mechanics,events)

    P=np.asarray(data['P'],np.float32).copy(); legacy=P[:,:,9:12].copy()
    P[:,:,9:12]=force_pack['force_grid_H'].astype(np.float32)
    contact=force_pack['contact_grid'].astype(np.float32)
    required={'B','U','L_LEG','E'}
    missing=required-set(data)
    if missing: raise ValueError(f'{cache}: missing {sorted(missing)}')
    arrays={
        'time':(grid-grid[0]).astype(np.float64),'grid_original':grid.astype(np.float64),
        'crop_start_original':np.array(grid[0],np.float64),'P':P,'P_legacy_force':legacy,
        'force_input_H':force_pack['force_causal_grid_H'].astype(np.float32),
        'B':np.asarray(data['B'],np.float32),'U':np.asarray(data['U'],np.float32),
        'L_LEG':np.asarray(data['L_LEG'],np.float32),'E':np.asarray(data['E'],np.float32),
        'contact':contact,'touchdown_impulse':impulse_grid,'desired_wrench_target':desired,
        **mechanics,
        'force_frame':np.array('run_local_heading_H_forward_left_up'),
        'base_frame':np.array('base_link'),'trajectory_id':np.array(spec['trajectory_id']),
        'terrain':np.array(spec['terrain']),'run_id':np.array(spec['run_id']),
    }
    n=len(grid)
    for key in ('P','force_input_H','B','U','L_LEG','E','contact','touchdown_impulse','desired_wrench_target',
                'R_HB','p_HB','com_H','foot_world','foot_r_H','J_leg_H'):
        if len(arrays[key])!=n: raise ValueError(f'{key}: length {len(arrays[key])} != {n}')
        if not np.isfinite(arrays[key]).all(): raise ValueError(f'{key}: non-finite values')
    if arrays['P'].shape!=(n,4,12) or arrays['foot_world'].shape!=(n,4,3):
        raise ValueError('strict core shapes invalid')

    atomic_savez(paths['core'],**arrays)
    save_native_force_checkpoint(paths['native'],native,transformed,force_pack,wave_t,waves)
    atomic_savez(paths['haptic'],**haptic)
    events.to_csv(paths['events'],index=False); force_audit.to_csv(paths['force_audit'],index=False)
    impulse_audit.to_csv(paths['impulse_audit'],index=False)
    audit.to_csv(paths['root']/'native_clock_frame_audit.csv',index=False)
    balance=force_balance_diagnostic(force_pack,robot.mass)
    atomic_write_json(paths['root']/'force_balance_diagnostic.json',balance)
    atomic_write_json(paths['contracts_snapshot'],{'force':fc,'calibration':cc,'signals':sc,
        'source_hashes':{str(p.name):file_sha256(p) for p in (FORCE_CONTRACT_PATH,CALIBRATION_CONTRACT_PATH,SIGNAL_CONTRACT_PATH)}})
    files=[paths['core'],paths['native'],paths['haptic'],paths['events'],paths['force_audit'],paths['impulse_audit'],paths['geometry_validation'],
           paths['geometry_validation_samples'],paths['contracts_snapshot'],
           paths['root']/'native_clock_frame_audit.csv',paths['root']/'force_balance_diagnostic.json',
           paths['geometry']/'COMPLETE.json']
    manifest={
        'schema':'V10R_STRICT_PREPARED_RUN_V1','pipeline_version':STRICT_PIPELINE_VERSION,
        'created_utc':utc_now(),'run':spec,'coordinate_frame':'H: fixed right-handed run-local forward/left/up',
        'force_target':'native WrenchStamped -> verified frame/sign -> H; 45 Hz zero-phase LP -> exact grid_original',
        'legacy_force_used_as_target':False,'zero_phase_force_used_as_history_input':False,
        'future_measured_state_used_as_model_input':False,
        'source':{'raw_bag':str(bag),'raw_bag_sha256':file_sha256(bag),
                  'physics_cache':str(cache),'physics_cache_sha256':file_sha256(cache)},
        'contracts':{str(p.name):file_sha256(p) for p in (FORCE_CONTRACT_PATH,CALIBRATION_CONTRACT_PATH,SIGNAL_CONTRACT_PATH)},
        'files':[{'relative':str(p.relative_to(paths['root'])),'bytes':p.stat().st_size,'sha256':file_sha256(p)} for p in files],
        'geometry_chunks':json.load(open(paths['geometry']/'COMPLETE.json'))['chunks'],
        'counts':{'grid_samples':n,'native_uniform_samples':len(force_pack['uniform_t']),
                  'touchdowns':len(events),'haptic_events':len(haptic['time_original']),'geometry_voxels':geom_meta['final_voxels']},
        'diagnostics':{'force_balance':balance,
            'median_impulse_filter_relative_change':float(impulse_audit.relative_vector_change.median()) if len(impulse_audit) else None},
        'gates':{'clock':True,'force_frame_contract':True,'static_sign_contract':True,
                 'calibration_contract':True,'signal_contract':True,'geometry_checkpoint':True,
                 'heldout_geometry_validation':True,
                 'array_invariants':True},
    }
    atomic_write_json(paths['manifest'],manifest)
    result=verify_prepared_run(spec['run_id'],raise_on_error=True)
    print(spec['run_id'],'prepared and verified:',result)
    return result


def verify_prepared_run(run_id,raise_on_error=True):
    p=prepared_paths(run_id); errors=[]
    if not p['manifest'].exists(): errors.append('prepared_manifest.json missing')
    else:
        m=json.load(open(p['manifest']))
        if m.get('schema')!='V10R_STRICT_PREPARED_RUN_V1': errors.append('manifest schema mismatch')
        if m.get('legacy_force_used_as_target') is not False: errors.append('legacy force target flag invalid')
        if not all(m.get('gates',{}).values()): errors.append('one or more G0 preparation gates failed')
        for x in m.get('files',[]):
            q=p['root']/x['relative']
            if not q.exists(): errors.append(f"missing {x['relative']}")
            elif q.stat().st_size!=x['bytes'] or file_sha256(q)!=x['sha256']: errors.append(f"hash mismatch {x['relative']}")
        g=verify_geometry_checkpoint(p['geometry'],raise_on_error=False)
        if not g['pass']: errors.extend('geometry: '+x for x in g['errors'])
        if not p['haptic'].exists(): errors.append('haptic event checkpoint missing')
        if not p['geometry_validation'].exists(): errors.append('held-out geometry validation missing')
        elif not json.load(open(p['geometry_validation'])).get('pass',False): errors.append('held-out geometry validation failed')
        if p['core'].exists():
            try:
                with np.load(p['core'],allow_pickle=False) as z:
                    n=len(z['time']); required={'P','force_input_H','B','U','L_LEG','E','contact','foot_world','com_H','R_HB',
                        'desired_wrench_target','touchdown_impulse','J_leg_H','P_legacy_force'}
                    miss=required-set(z.files)
                    if miss: errors.append('core missing '+','.join(sorted(miss)))
                    elif z['P'].shape!=(n,4,12) or z['foot_world'].shape!=(n,4,3): errors.append('core shape mismatch')
                    if len(z['grid_original'])!=n or not np.all(np.diff(z['grid_original'])>0): errors.append('core clock invalid')
                    if np.array_equal(z['P'][:,:,9:12],z['P_legacy_force']): errors.append('strict force equals legacy cache byte-for-byte')
            except Exception as e: errors.append(f'core unreadable: {type(e).__name__}: {e}')
    result={'run_id':run_id,'pass':not errors,'errors':errors,
            'manifest_sha256':file_sha256(p['manifest']) if p['manifest'].exists() else None}
    if errors and raise_on_error: raise RuntimeError(f'{run_id} verification failed: '+'; '.join(errors))
    return result


def deletion_token(run_id):
    v=verify_prepared_run(run_id,raise_on_error=True)
    return f"DELETE_VERIFIED_RAW::{run_id}::{v['manifest_sha256'][:12]}"


def delete_one_verified_raw_run(run_id,confirmation):
    spec=next((s for s in RUN_REGISTRY if s['run_id']==run_id),None)
    if spec is None: raise ValueError(run_id)
    expected=deletion_token(run_id)
    if confirmation!=expected:
        raise RuntimeError('Deletion refused. Exact required token:\n'+expected)
    target=raw_run_path(spec).resolve(); root=RAW_ROOT.resolve()
    if target==root or root not in target.parents or target.name!=spec['run']:
        raise RuntimeError(f'unsafe deletion target rejected: {target}')
    if not target.exists(): print('raw run already absent:',target); return
    before=directory_bytes(target); print('Deleting verified raw run only:',target,f'({gib(before):.2f} GiB)')
    shutil.rmtree(target)
    print('Deleted. Prepared checkpoint remains at',prepared_paths(run_id)['root'])

# %% notebook cell 44
# STRICT S9 — load prepared runs and query latest causal fused voxel states
def scan_verified_prepared():
    rows=[]
    for p in sorted(PREPARED_ROOT.glob('*/prepared_manifest.json')):
        rid=p.parent.name; v=verify_prepared_run(rid,raise_on_error=False)
        spec=next((s for s in RUN_REGISTRY if s['run_id']==rid),{})
        rows.append({**spec,**v,'root':str(p.parent)})
    return pd.DataFrame(rows)


class CausalHapticMemory:
    def __init__(self,path,crop_start_original):
        with np.load(path,allow_pickle=False) as z:
            for k in z.files: setattr(self,k,np.asarray(z[k]))
        self.crop_start=float(crop_start_original)

    def query_one(self,mean_xyz,cov_xyz,origin_t,radius=CFG.patch_radius_m,enabled=True):
        if not enabled or len(self.time_original)==0:
            out=np.zeros(16,np.float32); out[-1]=1.; return out
        q=np.asarray(mean_xyz,float); t=float(origin_t)+self.crop_start
        d=np.linalg.norm(self.xyz[:,:2]-q[None,:2],axis=1)
        ids=np.where((self.time_original<t-1e-9)&(d<=radius))[0]
        if len(ids)==0:
            out=np.zeros(16,np.float32); out[-1]=1.; return out
        # Marginalize location uncertainty by inflating the spatial kernel.
        sigma=math.sqrt(max(float(np.trace(np.asarray(cov_xyz,float)[:2,:2])),1e-8))
        scale=max(radius/2,sigma); w=np.exp(-.5*(d[ids]/scale)**2); w/=max(w.sum(),1e-12)
        fm=(w[:,None]*self.force[ids]).sum(0); fs=np.sqrt((w[:,None]*(self.force[ids]-fm)**2).sum(0))
        jm=(w[:,None]*self.impulse[ids]).sum(0); slip=float(np.sum(w*self.slip[ids])); slips=float(np.sqrt(np.sum(w*(self.slip[ids]-slip)**2)))
        te=float(np.sum(w*self.torque_error[ids])); tes=float(np.sqrt(np.sum(w*(self.torque_error[ids]-te)**2)))
        dur=float(np.sum(w*self.contact_duration[ids])); conf=float(np.clip(np.log1p(len(ids))/5.,0,1))
        out=np.r_[fm,fs,jm,slip,slips,te,tes,dur,conf,0.]
        if out.shape!=(16,): raise AssertionError(out.shape)
        return out.astype(np.float32)


class CausalFusedVoxelMemory:
    """Latest cumulative state per 3-D voxel at a query time; returns 64 patch features."""
    def __init__(self,geometry_root,crop_start_original,voxel=CFG.patch_voxel_m):
        root=Path(geometry_root); paths=sorted((root/'chunks').glob('chunk_*.npz'))
        verify_geometry_checkpoint(root,raise_on_error=True)
        fields=('key','xyz','normal','rgb','xyz_var','count','camera_mask','time')
        pieces={k:[] for k in fields}
        for p in paths:
            with np.load(p,allow_pickle=False) as z:
                for k in fields: pieces[k].append(np.asarray(z[k]))
        x={k:np.concatenate(v,axis=0) for k,v in pieces.items()}
        # Contiguous histories sorted by voxel key then update time.
        order=np.lexsort((x['time'],x['key'][:,2],x['key'][:,1],x['key'][:,0]))
        for k in x: setattr(self,k,x[k][order])
        change=np.r_[True,np.any(np.diff(self.key,axis=0),axis=1)]
        self.starts=np.where(change)[0]; self.ends=np.r_[self.starts[1:],len(self.key)]
        self.unique_key=self.key[self.starts]; self.lookup={tuple(map(int,k)):i for i,k in enumerate(self.unique_key)}
        self.columns=defaultdict(list)
        for i,k in enumerate(self.unique_key): self.columns[(int(k[0]),int(k[1]))].append(i)
        self.crop_start=float(crop_start_original); self.voxel=float(voxel)

    def _latest_candidates(self,q,t_cropped,radius):
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
        ids=np.asarray(ids,int); dxy=np.linalg.norm(self.xyz[ids,:2]-q[None,:2],axis=1)
        return ids[dxy<=radius]

    def _feature32(self,q,t_cropped,radius,candidate_ids=None):
        if candidate_ids is None:
            ids=self._latest_candidates(q,t_cropped,radius)
        else:
            ids=np.asarray(candidate_ids,int)
            d0=np.linalg.norm(self.xyz[ids,:2]-np.asarray(q)[None,:2],axis=1)
            ids=ids[d0<=radius]
        if len(ids)==0:
            f=np.zeros(32,np.float32); f[-1]=1.; return f
        q=np.asarray(q,float); z=self.xyz[ids,2]
        # Prefer support near the predicted foot height; retain ambiguity as a feature.
        dz=np.abs(z-q[2]); near=dz<=max(.20,3*self.voxel)
        if near.any(): ids=ids[near]; z=z[near]
        dxy=np.linalg.norm(self.xyz[ids,:2]-q[None,:2],axis=1)
        w=np.exp(-.5*(dxy/max(radius/2,1e-6))**2)/np.maximum(np.sqrt(self.xyz_var[ids].sum(1)),1e-3)
        w/=max(w.sum(),1e-12); rel=self.xyz[ids]-q[None]
        centroid=(w[:,None]*rel).sum(0); var=(w[:,None]*(rel-centroid)**2).sum(0)
        nmean=(w[:,None]*self.normal[ids]).sum(0); nmean/=max(np.linalg.norm(nmean),1e-8)
        nstd=np.sqrt((w[:,None]*(self.normal[ids]-nmean)**2).sum(0))
        rgbmean=(w[:,None]*self.rgb[ids]).sum(0); rgbstd=np.sqrt((w[:,None]*(self.rgb[ids]-rgbmean)**2).sum(0))
        zrange=float(z.max()-z.min()); rough=float(np.sqrt(np.sum(w*(z-np.sum(w*z))**2)))
        slope=float(np.linalg.norm(nmean[:2])/max(abs(nmean[2]),1e-5))
        age=np.clip((float(t_cropped)+self.crop_start-self.time[ids]).min()/5.,0,1)
        geom_sigma=float(np.sqrt(np.sum(w*self.xyz_var[ids].sum(1))))
        cameras=np.array([int(x).bit_count() for x in self.camera_mask[ids]])
        camera_div=float(np.sum(w*cameras)/len(CAMERA_NAMES))
        count=float(np.log1p(np.sum(self.count[ids]))/12.)
        visibility=float(np.clip(np.sum(w*np.minimum(self.count[ids]/20.,1.)),0,1))
        edge=float(np.clip(zrange/.12,0,1)); ambiguity=float(np.clip((zrange-.08)/.20,0,1))
        occupancy=float(np.clip(len(ids)/(math.pi*(radius/self.voxel)**2+1e-6),0,1))
        f=np.r_[centroid,var,z.min()-q[2],z.max()-q[2],zrange,nmean,nstd,slope,rough,
                rgbmean,rgbstd,count,age,geom_sigma,visibility,edge,ambiguity,camera_div,occupancy,0.]
        if f.shape!=(32,): raise AssertionError(f.shape)
        return f.astype(np.float32)

    def query_one(self,mean_xyz,cov_xyz,origin_t,radius=CFG.patch_radius_m,sigma_points=13):
        mean=np.asarray(mean_xyz,float); cov=np.asarray(cov_xyz,float)+np.eye(3)*1e-8
        eig,vec=np.linalg.eigh(cov); eig=np.maximum(eig,1e-8); pts=[mean]
        for k in range(3):
            d=math.sqrt(3*eig[k])*vec[:,k]; pts.extend((mean+d,mean-d))
        # Deterministic symmetric sigma points; no stochastic test-time map query.
        scales=(.5,1.5,2.0)
        for s in scales:
            if len(pts)>=sigma_points: break
            for k in range(3):
                if len(pts)>=sigma_points: break
                pts.append(mean+s*math.sqrt(eig[k])*vec[:,k])
                if len(pts)<sigma_points: pts.append(mean-s*math.sqrt(eig[k])*vec[:,k])
        search_radius=radius+max(np.linalg.norm(p[:2]-mean[:2]) for p in pts[:sigma_points])
        candidates=self._latest_candidates(mean,origin_t,search_radius)
        features=np.stack([self._feature32(p,origin_t,radius,candidates) for p in pts[:sigma_points]])
        # 64-D contract: distribution-marginalized mean and dispersion.
        return np.r_[features.mean(0),features.std(0)].astype(np.float32)

    def surface_at(self,mean_xyz,origin_t,radius=CFG.patch_radius_m):
        q=np.asarray(mean_xyz,float); ids=self._latest_candidates(q,origin_t,radius)
        if len(ids)==0: return np.array([0.,0.,1.],np.float32),0.25,0.,1.
        dz=np.abs(self.xyz[ids,2]-q[2]); near=dz<=max(.20,3*self.voxel)
        if near.any(): ids=ids[near]
        d=np.linalg.norm(self.xyz[ids,:2]-q[None,:2],axis=1)
        w=np.exp(-.5*(d/max(radius/2,1e-6))**2); w/=max(w.sum(),1e-12)
        n=(self.normal[ids]*w[:,None]).sum(0); n/=max(np.linalg.norm(n),1e-8)
        rough=float(np.sqrt(np.sum(w*(self.xyz[ids,2]-np.sum(w*self.xyz[ids,2]))**2)))
        mu=float(np.clip(.8-2.5*rough,.15,.9))
        confidence=float(np.clip(np.sum(w*np.minimum(self.count[ids]/20.,1.)),0,1))
        uncertainty=float(np.sqrt(np.sum(w*self.xyz_var[ids].sum(1))))
        return n.astype(np.float32),mu,confidence,uncertainty

    def raycast_support(self,query_xyz,origin_t,radius=CFG.patch_radius_m):
        q=np.asarray(query_xyz,float); ids=self._latest_candidates(q,origin_t,radius)
        if len(ids)==0: return float(q[2]),0.,1.
        dz=np.abs(self.xyz[ids,2]-q[2]); near=dz<=max(.30,4*self.voxel)
        if near.any(): ids=ids[near]
        d=np.linalg.norm(self.xyz[ids,:2]-q[None,:2],axis=1)
        w=np.exp(-.5*(d/max(radius/2,1e-6))**2); w/=max(w.sum(),1e-12)
        z=float(np.sum(w*self.xyz[ids,2])); zrange=float(self.xyz[ids,2].max()-self.xyz[ids,2].min())
        confidence=float(np.clip(np.sum(w*np.minimum(self.count[ids]/20.,1.)),0,1))
        ambiguity=float(np.clip((zrange-.08)/.20,0,1))
        return z,confidence,ambiguity

    def latest_surface(self,t_cropped):
        ids=[]; t_original=float(t_cropped)+self.crop_start
        for a,b in zip(self.starts,self.ends):
            j=a+np.searchsorted(self.time[a:b],t_original,side='right')-1
            if j>=a: ids.append(j)
        ids=np.asarray(ids,int)
        return {k:getattr(self,k)[ids] for k in ('key','xyz','normal','rgb','xyz_var','count','camera_mask','time')}


def load_prepared_run(run_id,load_memory=True):
    verify_prepared_run(run_id,raise_on_error=True); p=prepared_paths(run_id)
    with np.load(p['core'],allow_pickle=False) as z: d={k:np.asarray(z[k]) for k in z.files}
    spec=next(s for s in RUN_REGISTRY if s['run_id']==run_id)
    # Preserve the canonical class and keys; desired_wrench here is a target/diagnostic only.
    d['desired_wrench']=d['desired_wrench_target']
    rec=RunRecord(run_id,spec['terrain'],d,str(d['force_frame'].item()),str(d['base_frame'].item()),spec['trajectory_id'])
    memory=CausalFusedVoxelMemory(p['geometry'],float(d['crop_start_original'])) if load_memory else None
    if memory is not None: memory.haptic=CausalHapticMemory(p['haptic'],float(d['crop_start_original']))
    events=pd.read_csv(p['events'])
    with np.load(p['native'],allow_pickle=False) as z:
        event_wave_time=np.asarray(z['impact_wave_time_s']); event_waveforms=np.asarray(z['impact_waveforms_H'])
    return rec,memory,events,event_wave_time,event_waveforms


PREPARED_TABLE=scan_verified_prepared()
display(PREPARED_TABLE)

# %% notebook cell 45
# STRICT S10 — frozen trajectory split and data-eligibility gate
def write_split_contract_once(prepared_table):
    columns=['run_id','trajectory_id','terrain','gait','policy','speed_bin','split','frozen_before_training','note']
    if SPLIT_CONTRACT_PATH.exists():
        try: existing=pd.read_csv(SPLIT_CONTRACT_PATH)
        except pd.errors.EmptyDataError: existing=pd.DataFrame(columns=columns)
    else: existing=pd.DataFrame(columns=columns)
    # Diagnostic proposal only. Review/freeze this file before any training.
    default={'forest_P2000':'train','swamp_marsh_P2023':'train',
             'rough_uneven_P2000':'val','wet_uneven_proxy_P2000':'test'}
    rows=existing.to_dict('records'); known=set(existing.run_id) if 'run_id' in existing else set()
    for _,r in prepared_table.iterrows():
        if r.run_id in known: continue
        rows.append({'run_id':r.run_id,'trajectory_id':r.trajectory_id,'terrain':r.terrain,
                     'gait':r.get('gait','unreported'),'policy':r.get('policy','unreported'),
                     'speed_bin':r.get('speed_bin','unreported'),'split':default.get(r.run_id,'UNASSIGNED'),
                     'frozen_before_training':False,'note':'REVIEW AND FREEZE'})
    frame=pd.DataFrame(rows,columns=columns)
    if len(frame)!=len(existing) or not SPLIT_CONTRACT_PATH.exists():
        frame.to_csv(SPLIT_CONTRACT_PATH,index=False)
        print('Updated split proposal; review it and set frozen_before_training=True:',SPLIT_CONTRACT_PATH)


def paper_data_gate_strict(frame,min_runs=CFG.min_independent_runs):
    problems=[]; f=frame.copy()
    required={'run_id','trajectory_id','terrain','gait','policy','speed_bin','split','frozen_before_training'}
    missing=required-set(f.columns)
    if missing: return {'pass':False,'problems':['split contract missing '+','.join(sorted(missing))]}
    if f.trajectory_id.nunique()<min_runs: problems.append(f'only {f.trajectory_id.nunique()} independent trajectories; require >= {min_runs}')
    if f.run_id.duplicated().any() or f.trajectory_id.duplicated().any(): problems.append('run/trajectory IDs are not unique')
    if set(f.split)!={'train','val','test'}: problems.append('train/val/test are not all populated or UNASSIGNED remains')
    frozen=f.frozen_before_training.astype(str).str.lower().isin({'true','1','yes'})
    if not frozen.all(): problems.append('split contract is not frozen before training')
    for col in ('terrain','gait','policy','speed_bin'):
        if f[col].astype(str).str.lower().isin({'','unknown','unreported','replace','nan'}).any(): problems.append(f'{col} metadata unresolved')
    if not any('unreported' in x for x in problems):
        counts=f.groupby(['terrain','gait','policy','speed_bin']).trajectory_id.nunique()
        small=counts[counts<3]
        for key,n in small.items(): problems.append(f'experimental cell {key} has {n} trajectories; require >=3')
    prepared=set(PREPARED_TABLE.loc[PREPARED_TABLE['pass']==True,'run_id']) if len(PREPARED_TABLE) else set()
    absent=sorted(set(f.run_id)-prepared)
    if absent: problems.append('unverified/missing prepared runs: '+','.join(absent))
    return {'pass':not problems,'problems':problems,'independent_trajectories':int(f.trajectory_id.nunique())}


write_split_contract_once(PREPARED_TABLE)
SPLIT_FRAME=pd.read_csv(SPLIT_CONTRACT_PATH) if SPLIT_CONTRACT_PATH.exists() else pd.DataFrame()
DATA_GATE=paper_data_gate_strict(SPLIT_FRAME)
atomic_write_json(STRICT_ROOT/'paper_data_gate.json',DATA_GATE)
display(SPLIT_FRAME); print(json.dumps(DATA_GATE,indent=2))


def assert_training_authorized():
    if DATA_GATE['pass']: return
    if ANALYSIS_TIER=='diagnostic' and ALLOW_DIAGNOSTIC_TRAINING:
        warnings.warn('DIAGNOSTIC ONLY — paper data gate failed; every result is ineligible for paper claims')
        return
    raise RuntimeError('TRAINING BLOCKED BY PAPER DATA GATE:\n'+'\n'.join('- '+x for x in DATA_GATE['problems']))

# %% notebook cell 47
# STRICT S11 — train-only input normalization and causal event-labelled windows
@dataclass
class InputNormalizer:
    leg_mean: np.ndarray; leg_std: np.ndarray
    body_mean: np.ndarray; body_std: np.ndarray
    action_mean: np.ndarray; action_std: np.ndarray

    def as_dict(self): return {k:np.asarray(v) for k,v in dataclasses.asdict(self).items()}


def _moments(arrays):
    total=0; sx=None; sx2=None
    for x in arrays:
        x=np.asarray(x,np.float64).reshape(-1,x.shape[-1]); n=len(x)
        total+=n; sx=x.sum(0) if sx is None else sx+x.sum(0); sx2=(x*x).sum(0) if sx2 is None else sx2+(x*x).sum(0)
    mean=sx/max(total,1); var=np.maximum(sx2/max(total,1)-mean*mean,1e-8)
    return mean.astype(np.float32),np.sqrt(var).astype(np.float32)


def strict_leg_history_features(run):
    """Canonical 25-D leg features with a strictly causal force-history channel."""
    d=run.data
    if 'force_input_H' not in d: raise ValueError('prepared run lacks causal force_input_H')
    P=np.asarray(d['P'],np.float32).copy(); P[...,FORCE_SLICE]=np.asarray(d['force_input_H'],np.float32)
    t=np.asarray(d['time'],float); E=np.asarray(d['E'],np.float32)
    dE=causal_rate(E,t,clip=20.); dF=causal_rate(P[...,FORCE_SLICE],t,clip=20_000.)
    x=np.concatenate([P,np.asarray(d['contact'])[...,None],np.asarray(d['L_LEG']),E,dE,dF],axis=-1)
    if x.shape[1:]!=(4,25): raise AssertionError(x.shape)
    return x.astype(np.float32)


def fit_input_normalizer(train_runs):
    leg=[]; body=[]; action=[]
    for r in train_runs:
        leg.append(strict_leg_history_features(r)); body.append(body_history_features(r))
        action.append(np.c_[r.data['U'],r.data['L_LEG'].reshape(len(r.data['time']),-1)])
    lm,ls=_moments(leg); bm,bs=_moments(body); am,ass=_moments(action)
    return InputNormalizer(lm,ls,bm,bs,am,ass)


def event_labels_for_origin(events,waveforms,origin_t,horizons=CFG.horizons_s):
    H=len(horizons); hazard=np.zeros((H,4),np.float32); offset=np.zeros((H,4),np.float32)
    mask=np.zeros((H,4),bool); impulse=np.zeros((H,4,3),np.float32)
    W=waveforms.shape[1] if waveforms.ndim==3 else int(round((IMPACT_PRE_S+IMPACT_POST_S)*IMPACT_WAVEFORM_HZ))+1
    wave=np.zeros((H,4,W,3),np.float32)
    edges=np.r_[0.,np.asarray(horizons,float)]
    for li in range(4):
        g=events[events.leg_index==li] if len(events) else events
        if not len(g): continue
        future=g[(g.t_touchdown_cropped>origin_t)&(g.t_touchdown_cropped<=origin_t+horizons[-1]+1e-9)]
        for _,e in future.iterrows():
            rel=float(e.t_touchdown_cropped-origin_t); k=int(np.searchsorted(edges[1:],rel,side='left'))
            if k<0 or k>=H or mask[k,li]: continue
            center=.5*(edges[k]+edges[k+1]); hazard[k,li]=1.; offset[k,li]=rel-center; mask[k,li]=True
            impulse[k,li]=[e.Jx,e.Jy,e.Jz]; wi=int(e.waveform_row)
            if 0<=wi<len(waveforms): wave[k,li]=waveforms[wi]
    return {'event_hazard':hazard,'event_offset_s':offset,'event_mask':mask,
            'impulse':impulse,'impact_waveform':wave,'impact_mask':mask.copy()}


class StrictTrajectoryWindowDataset(torch.utils.data.Dataset):
    def __init__(self,run,events,waveforms,normalizer,origins=None,cfg=CFG,action_mode=STRICT_ACTION_MODE):
        self.run=run; self.events=events; self.waveforms=np.asarray(waveforms); self.norm=normalizer; self.cfg=cfg
        self.hist=int(round(cfg.history_s*cfg.sample_hz)); self.offset=np.asarray([int(round(h*cfg.sample_hz)) for h in cfg.horizons_s])
        self.origins=np.asarray(valid_origins(run,cfg) if origins is None else origins,int)
        self.leg_x=strict_leg_history_features(run); self.body_x=body_history_features(run); self.action_mode=action_mode
        if action_mode not in {'hold_last','controller_plan','oracle_logged'}: raise ValueError(action_mode)
        if action_mode=='oracle_logged' and ANALYSIS_TIER=='paper': raise ValueError('oracle_logged is forbidden as the paper headline model')

    def __len__(self): return len(self.origins)

    def __getitem__(self,k):
        i=int(self.origins[k]); js=i+self.offset; d=self.run.data; origin_t=float(d['time'][i])
        now=np.r_[d['U'][i],d['L_LEG'][i].reshape(-1)].astype(np.float32)
        if self.action_mode=='hold_last': action=np.repeat(now[None],len(js),axis=0)
        elif self.action_mode=='controller_plan':
            if 'controller_plan' not in d: raise ValueError('timestamped controller_plan is unavailable')
            action=np.asarray(d['controller_plan'][i],np.float32)
        else: action=np.stack([np.r_[d['U'][j],d['L_LEG'][j].reshape(-1)] for j in js]).astype(np.float32)
        ev=event_labels_for_origin(self.events,self.waveforms,origin_t,self.cfg.horizons_s)
        phase=contact_phase_at(d['contact'],js)
        leg=(self.leg_x[i-self.hist+1:i+1].transpose(1,0,2)-self.norm.leg_mean)/self.norm.leg_std
        body=(self.body_x[i-self.hist+1:i+1]-self.norm.body_mean)/self.norm.body_std
        action=(action-self.norm.action_mean)/self.norm.action_std
        target={'phase':phase.astype(np.int64),'force':np.asarray(d['P'][js,:,9:12],np.float32),
                'foot':np.asarray(d['foot_world'][js],np.float32),'impulse':ev['impulse'],
                'leg_sensor':np.asarray(d['P'][js,:,:9],np.float32),
                'body_sensor':np.asarray(d['B'][js],np.float32),
                'wrench_target':np.asarray(d['desired_wrench_target'][js],np.float32),
                'event_hazard':ev['event_hazard'],'event_offset_s':ev['event_offset_s'],
                'event_mask':ev['event_mask'],'impact_waveform':ev['impact_waveform'],'impact_mask':ev['impact_mask']}
        item={'leg_hist':leg.astype(np.float32),'body_hist':body.astype(np.float32),
              'action_plan':action.astype(np.float32),'foot_now':np.asarray(d['foot_world'][i],np.float32),
              'origin_t':np.float32(origin_t),'origin_index':np.int64(i),
              'origin_R_HB':np.asarray(d['R_HB'][i],np.float32),'origin_com_H':np.asarray(d['com_H'][i],np.float32),
              'origin_gyro_B':np.asarray(d['B'][i,3:6],np.float32),
              'mass_kg':np.float32(d['mass_kg']),'inertia_B':np.asarray(d['inertia_neutral_B'],np.float32),
              'target':target}
        return torch.utils.data.default_convert(item)


def make_origins(run,split,seed=SEED):
    ids=valid_origins(run)
    if split in {'val','test'}: return ids[::EVAL_ORIGIN_STRIDE]
    ids=ids[::TRAIN_ORIGIN_STRIDE]
    if TRAIN_ORIGINS_PER_RUN is None or len(ids)<=TRAIN_ORIGINS_PER_RUN: return ids
    # Training-only fixed event enrichment; no validation/test frequencies are consulted.
    c=np.asarray(run.data['contact'])>0.5; event=np.any(c[1:]!=c[:-1],axis=1); event_ids=np.where(event)[0]+1
    near=np.zeros(len(c),bool)
    for j in event_ids: near[max(0,j-20):min(len(c),j+21)]=True
    w=np.where(near[ids],5.,1.); w/=w.sum()
    rng=np.random.default_rng(seed); return np.sort(rng.choice(ids,TRAIN_ORIGINS_PER_RUN,replace=False,p=w))

# %% notebook cell 48
# STRICT S12 — faithful V10R wrapper with non-leaking centroidal wrench and impact timing/waveform heads
def skew_torch(v):
    z=torch.zeros_like(v[...,0]); x,y,w=v.unbind(-1)
    return torch.stack([z,-w,y,w,z,-x,-y,x,z],-1).reshape(*v.shape[:-1],3,3)


def so3_exp_torch(phi):
    theta=torch.linalg.norm(phi,dim=-1,keepdim=True); K=skew_torch(phi)
    t2=theta.square(); safe=theta.clamp_min(1e-7); safe2=safe.square()
    A=torch.where(theta>1e-4,torch.sin(theta)/safe,1-t2/6+t2*t2/120)
    B=torch.where(theta>1e-4,(1-torch.cos(theta))/safe2,.5-t2/24+t2*t2/720)
    I=torch.eye(3,device=phi.device,dtype=phi.dtype).expand(*phi.shape[:-1],3,3)
    return I+A[...,None]*K+B[...,None]*(K@K)


def predicted_centroidal_context(proposal,batch):
    bmean=proposal['body_raw'][...,:12]; B,H,_=bmean.shape
    horizons=torch.as_tensor(CFG.horizons_s,device=bmean.device,dtype=bmean.dtype)
    dts=torch.diff(torch.cat([torch.zeros(1,device=bmean.device,dtype=bmean.dtype),horizons]))
    R=batch['origin_R_HB']; com=batch['origin_com_H']; prev_omega=batch['origin_gyro_B']
    mass=batch['mass_kg'].reshape(B,1); inertia=batch['inertia_B']
    Rs=[]; coms=[]; wrench=[]
    for k in range(H):
        dt=dts[k]; omega=bmean[:,k,3:6]
        R=R@so3_exp_torch(omega*dt)
        v_H=torch.einsum('bij,bj->bi',R,bmean[:,k,6:9]); com=com+v_H*dt
        alpha=(omega-prev_omega)/dt.clamp_min(1e-4); Iw=torch.einsum('bij,bj->bi',inertia,omega)
        moment_B=torch.einsum('bij,bj->bi',inertia,alpha)+torch.cross(omega,Iw,dim=-1)
        force_B=mass*bmean[:,k,:3]
        force_H=torch.einsum('bij,bj->bi',R,force_B); moment_H=torch.einsum('bij,bj->bi',R,moment_B)
        wrench.append(torch.cat([force_H,moment_H],-1)); Rs.append(R); coms.append(com); prev_omega=omega
    return torch.stack(wrench,1),torch.stack(coms,1),torch.stack(Rs,1)


class StrictV10R(nn.Module):
    def __init__(self,d=CFG.d_model,horizons=len(CFG.horizons_s),waveform_samples=13):
        super().__init__(); self.core=V10R(d=d,horizons=horizons); self.waveform_samples=int(waveform_samples)
        self.patch_fusion=MLP(80,64)
        self.event_time_head=MLP(3*d,2)
        self.waveform_head=MLP(4*d,self.waveform_samples*3*2)

    def propose(self,*args,**kwargs): return self.core.propose(*args,**kwargs)

    def decode_strict(self,proposal,patch_features,normal,mu,batch):
        patch_lat=self.core.patch(self.patch_fusion(patch_features))
        body=proposal['latent_body'][:,:,None].expand(-1,-1,4,-1)
        action=proposal['latent_action'][:,:,None].expand(-1,-1,4,-1)
        force_input=torch.cat([proposal['latent_leg'],body,action,patch_lat],-1)
        raw=self.core.force_free(force_input); f_free=raw[...,:3]
        desired,com_pred,R_pred=predicted_centroidal_context(proposal,batch)
        foot_r=proposal['foot_mean']-com_pred[:,:,None,:]
        f_proj,residual=self.core.project(f_free,foot_r,normal,mu,proposal['contact_prob'],desired)
        f_dynamics,dynamics_residual=self.core.project(torch.zeros_like(f_free),foot_r,normal,mu,
                                                       proposal['contact_prob'],desired)
        timing=self.event_time_head(torch.cat([proposal['latent_leg'],body,action],-1))
        event_offset=.05*torch.tanh(timing[...,0]); event_log_scale=timing[...,1].clamp(-8.,-1.)
        wave=self.waveform_head(force_input).reshape(*force_input.shape[:-1],self.waveform_samples,3,2)
        return {'force_mean':f_proj,'free_force':f_free,'dynamics_only_force':f_dynamics,
                'force_log_scale':raw[...,3:6].clamp(-7.,5.),'impulse':raw[...,6:9],
                'wrench_residual':residual,'desired_wrench_pred':desired,'com_pred':com_pred,'R_HB_pred':R_pred,
                'dynamics_only_wrench_residual':dynamics_residual,
                'event_offset_s':event_offset,'event_offset_log_scale':event_log_scale,
                'impact_waveform_mean':wave[...,0], 'impact_waveform_log_scale':wave[...,1].clamp(-7.,5.),
                'surface_normal':normal,'surface_mu':mu,'foot_r':foot_r}


class StrictPatchProvider:
    def __init__(self,memory,haptic_mode='causal',geometry_mode='uncertainty_patch'):
        if haptic_mode not in {'causal','zero_shot'}: raise ValueError(haptic_mode)
        if geometry_mode not in {'no_map','deterministic_point','deterministic_patch','uncertainty_patch'}: raise ValueError(geometry_mode)
        self.memory=memory; self.haptic_mode=haptic_mode; self.geometry_mode=geometry_mode; self.cache={}
    def __call__(self,mean,cov,origin_t):
        mn=mean.detach().cpu().numpy(); cv=cov.detach().cpu().numpy(); tt=origin_t.detach().cpu().numpy().reshape(-1)
        B,H,L,_=mn.shape; feat=np.empty((B,H,L,80),np.float32); normal=np.empty((B,H,L,3),np.float32)
        mu=np.empty((B,H,L),np.float32); confidence=np.empty((B,H,L),np.float32)
        for b in range(B):
            for h in range(H):
                for l in range(L):
                    voxel=self.memory.voxel; qkey=tuple(np.floor(mn[b,h,l]/voxel).astype(int))
                    q=(np.asarray(qkey,float)+.5)*voxel
                    cvkey=tuple(np.round(cv[b,h,l].reshape(-1),5))
                    safe_t=math.floor(float(tt[b])/MAP_SNAPSHOT_DT_S)*MAP_SNAPSHOT_DT_S
                    key=(qkey,cvkey,round(safe_t,6),l)
                    if key in self.cache:
                        feat[b,h,l],normal[b,h,l],mu[b,h,l],confidence[b,h,l]=self.cache[key]
                        continue
                    if self.geometry_mode=='no_map':
                        geom=np.zeros(64,np.float32); geom[31]=1.
                    elif self.geometry_mode=='deterministic_point':
                        geom=self.memory.query_one(q,np.eye(3)*1e-8,safe_t,radius=1.5*self.memory.voxel,sigma_points=1)
                    elif self.geometry_mode=='deterministic_patch':
                        geom=self.memory.query_one(q,np.eye(3)*1e-8,safe_t,radius=CFG.patch_radius_m,sigma_points=1)
                    else:
                        geom=self.memory.query_one(q,cv[b,h,l],safe_t)
                    haptic=self.memory.haptic.query_one(q,cv[b,h,l],safe_t,enabled=self.haptic_mode=='causal')
                    feat[b,h,l]=np.r_[geom,haptic]
                    if self.geometry_mode=='no_map':
                        normal[b,h,l]=[0,0,1]; mu[b,h,l]=.5; confidence[b,h,l]=0
                    else: normal[b,h,l],mu[b,h,l],confidence[b,h,l],_=self.memory.surface_at(q,safe_t)
                    self.cache[key]=(feat[b,h,l].copy(),normal[b,h,l].copy(),float(mu[b,h,l]),float(confidence[b,h,l]))
                    if len(self.cache)>200_000: self.cache.clear()
        return torch.from_numpy(feat),torch.from_numpy(normal),torch.from_numpy(mu),torch.from_numpy(confidence)


def batch_to_device(batch):
    keys=('leg_hist','body_hist','action_plan','foot_now','origin_t','origin_R_HB','origin_com_H',
          'origin_gyro_B','mass_kg','inertia_B')
    for k in keys: batch[k]=batch[k].to(DEVICE)
    target={k:v.to(DEVICE) for k,v in batch['target'].items()}
    return batch,target


def strict_forward(model,batch,provider):
    batch,target=batch_to_device(batch)
    proposal=model.propose(batch['leg_hist'],batch['body_hist'],batch['action_plan'],batch['foot_now'])
    cov=foot_covariance(proposal['foot_chol_raw'])
    patch,normal,mu,confidence=provider(proposal['foot_mean'].detach(),cov.detach(),batch['origin_t'])
    decoded=model.decode_strict(proposal,patch.to(DEVICE),normal.to(DEVICE),mu.to(DEVICE),batch)
    return {**proposal,**decoded,'map_confidence':confidence.to(DEVICE),'foot_covariance':cov},target,batch


def masked_gaussian_nll(mean,log_scale,target,mask):
    mask=mask.bool();
    if not mask.any(): return mean.sum()*0
    while mask.ndim<mean.ndim: mask=mask.unsqueeze(-1)
    scale=torch.exp(log_scale); loss=.5*((target-mean)/scale).square()+log_scale
    return loss[mask.expand_as(loss)].mean()


def strict_v10r_loss(out,target,stage='G4'):
    if stage=='G2':
        phase=F.cross_entropy(out['mode_logits'].reshape(-1,4),target['phase'].reshape(-1).long())
        foot=F.smooth_l1_loss(out['foot_mean'],target['foot'],beta=.02)
        sensors=diagonal_gaussian_nll(out['leg_sensor_raw'],target['leg_sensor'])+diagonal_gaussian_nll(out['body_raw'],target['body_sensor'])
        hazard=F.binary_cross_entropy_with_logits(out['event_hazard_logit'],target['event_hazard'])
        total=phase+.5*foot+.5*sensors+.5*hazard
        return {'total':total,'phase':phase,'foot':foot,'sensor':sensors,'hazard':hazard}
    base=v10r_loss(out,target)
    hazard=F.binary_cross_entropy_with_logits(out['event_hazard_logit'],target['event_hazard'])
    timing=masked_gaussian_nll(out['event_offset_s'],out['event_offset_log_scale'],target['event_offset_s'],target['event_mask'])
    wave=masked_gaussian_nll(out['impact_waveform_mean'],out['impact_waveform_log_scale'],target['impact_waveform'],target['impact_mask'])
    event_imp=F.smooth_l1_loss(out['impulse'][target['event_mask']],target['impulse'][target['event_mask']],beta=.05) \
        if target['event_mask'].any() else out['impulse'].sum()*0
    fscale=torch.exp(out['force_log_scale']); ferr=.5*((target['force']-out['force_mean'])/fscale).square()+out['force_log_scale']
    phase_w=torch.tensor([.35,2.5,1.,1.5],device=ferr.device)[target['phase'].long()][...,None]
    force_nll=(ferr*phase_w).mean()
    total=base['total']+.5*hazard+.5*timing+.4*wave+.8*event_imp+.25*force_nll
    return {**base,'total':total,'hazard':hazard,'timing':timing,'waveform':wave,
            'event_impulse':event_imp,'force_nll':force_nll}

# %% notebook cell 50
# STRICT S13 — leakage-free exact-state ceiling and moving-block confidence intervals
def strict_exact_state_features(run):
    d=run.data; P=np.asarray(d['P'],np.float32); n=len(P)
    X=np.c_[P[:,:,:9].reshape(n,-1),np.asarray(d['B']),np.asarray(d['U']),
            np.asarray(d['L_LEG']).reshape(n,-1),np.asarray(d['E']).reshape(n,-1),
            np.asarray(d['contact']),np.asarray(d['foot_r_H']).reshape(n,-1),
            np.asarray(d['J_leg_H']).reshape(n,-1),np.asarray(d['desired_wrench_target'])]
    y=P[:,:,9:12].reshape(n,-1)
    if X.shape[1]!=136: raise AssertionError(f'exact-state feature width changed: {X.shape}')
    return X.astype(np.float32),y.astype(np.float32)


def jacobian_torque_baseline(run,damping=1e-3):
    d=run.data; tau=np.asarray(d['P'])[:,:,6:9]; J=np.asarray(d['J_leg_H']); out=np.empty_like(tau)
    eye=np.eye(3)
    for i in range(len(tau)):
        for l in range(4):
            # tau_contact ≈ J^T f; sign/orientation is evaluated, never silently tuned on test.
            out[i,l]=np.linalg.solve(J[i,l]@J[i,l].T+damping*eye,J[i,l]@tau[i,l])
    return out


def phase_vector(contact):
    idx=np.arange(len(contact)); return contact_phase_at(contact,idx)


def moving_block_ci(values,times,stat=np.mean,block_s=BOOTSTRAP_BLOCK_S,reps=BOOTSTRAP_REPLICATES,seed=SEED):
    values=np.asarray(values); times=np.asarray(times,float)
    if len(values)==0: return (np.nan,np.nan)
    block=np.floor((times-times.min())/float(block_s)).astype(int); unique=np.unique(block)
    grouped=[values[block==b] for b in unique]; rng=np.random.default_rng(seed); draws=[]
    for _ in range(int(reps)):
        ids=rng.integers(0,len(grouped),len(grouped)); sample=np.concatenate([grouped[j] for j in ids])
        draws.append(float(stat(sample)))
    return tuple(np.quantile(draws,[.025,.975]))


def exact_state_metric_table(run_id,y,p,phase,times,model_name):
    e=np.abs(np.asarray(p)-np.asarray(y)); rows=[]
    for l,leg in enumerate(LEG_NAMES):
        for a,axis in enumerate('xyz'):
            for ph,name in enumerate(('swing','touchdown','stance','liftoff')):
                m=phase[:,l]==ph
                if not m.any(): continue
                x=e[m,l,a]; tt=np.asarray(times)[m]
                lo,hi=moving_block_ci(x,tt,stat=lambda q:np.quantile(q,.95),seed=SEED+31*l+7*a+ph)
                rows.append({'run_id':run_id,'model':model_name,'leg':leg,'axis':axis,'phase':name,'n':len(x),
                             'mae_n':x.mean(),'median_n':np.median(x),'p95_n':np.quantile(x,.95),
                             'p99_n':np.quantile(x,.99),'within_3n':np.mean(x<=3),'within_4n':np.mean(x<=4),
                             'max_n':x.max(),'p95_ci_low_n':lo,'p95_ci_high_n':hi})
    return pd.DataFrame(rows)


def run_exact_state_ceiling():
    if len(SPLIT_FRAME)==0: raise RuntimeError('split contract missing')
    # G1 is allowed as a diagnostic with >=2 independent verified runs, but can never override DATA_GATE.
    ids=SPLIT_FRAME.run_id.tolist()
    if len(ids)<2: raise RuntimeError('G1 requires at least two independently prepared trajectories')
    bundle={rid:load_prepared_run(rid,load_memory=False)[0] for rid in ids}
    train_ids=SPLIT_FRAME.query("split == 'train'").run_id.tolist(); test_ids=SPLIT_FRAME.query("split == 'test'").run_id.tolist()
    if not train_ids or not test_ids: raise RuntimeError('G1 needs nonempty train and test trajectories')
    Xs,ys=zip(*(strict_exact_state_features(bundle[x]) for x in train_ids)); X=np.concatenate(Xs); y=np.concatenate(ys)
    if len(X)>MAX_CEILING_TRAIN:
        rng=np.random.default_rng(SEED); take=np.sort(rng.choice(len(X),MAX_CEILING_TRAIN,replace=False)); X=X[take]; y=y[take]
    model=ExtraTreesRegressor(n_estimators=512,min_samples_leaf=3,max_features=.8,n_jobs=-1,random_state=SEED)
    model.fit(X,y); tables=[]
    out=RESULT_ROOT/'G1_exact_state'; out.mkdir(parents=True,exist_ok=True)
    for rid in test_ids:
        run=bundle[rid]; Xt,yt=strict_exact_state_features(run); pred=model.predict(Xt).reshape(-1,4,3)
        truth=yt.reshape(-1,4,3); phase=phase_vector(run.data['contact']); t=run.data['time']
        tables.append(exact_state_metric_table(rid,truth,pred,phase,t,'ExtraTrees_exact_state'))
        dyn=jacobian_torque_baseline(run); tables.append(exact_state_metric_table(rid,truth,dyn,phase,t,'Jacobian_torque'))
        np.savez_compressed(out/f'{rid}_predictions.npz',time=t,truth=truth,extra_trees=pred,jacobian_torque=dyn,phase=phase)
    metrics=pd.concat(tables,ignore_index=True); metrics.to_csv(out/'metrics_long.csv',index=False)
    stance=metrics[(metrics.model=='ExtraTrees_exact_state')&(metrics.phase=='stance')]
    gate={'paper_data_gate_pass':bool(DATA_GATE['pass']),'diagnostic_only':not bool(DATA_GATE['pass']),
          'all_stance_cells_p95_le_4n':bool(len(stance)>0 and (stance.p95_n<=4).all()),
          'all_stance_cells_bootstrap_ucb_le_5n':bool(len(stance)>0 and (stance.p95_ci_high_n<=5).all()),
          'max_stance_p95_n':float(stance.p95_n.max()) if len(stance) else None,
          'max_stance_p95_ucb_n':float(stance.p95_ci_high_n.max()) if len(stance) else None}
    gate['pass']=gate['paper_data_gate_pass'] and gate['all_stance_cells_p95_le_4n'] and gate['all_stance_cells_bootstrap_ucb_le_5n']
    atomic_write_json(out/'gate.json',gate); display(metrics); print(json.dumps(gate,indent=2))
    return metrics,gate,model

# %% notebook cell 52
# STRICT S14 — staged five-seed training, checkpointing, and bounded map residency
class PreparedStore:
    def __init__(self,split_frame,max_memories=1):
        self.split=split_frame.copy(); self.max_memories=max_memories; self.runs={}; self.events={}; self.waves={}; self.memories={}; self.order=[]
    def load_core(self,rid):
        if rid not in self.runs:
            r,_,e,wt,w=load_prepared_run(rid,load_memory=False)
            self.runs[rid]=r; self.events[rid]=e; self.waves[rid]=w
        return self.runs[rid]
    def memory(self,rid):
        if rid in self.memories:
            self.order.remove(rid); self.order.append(rid); return self.memories[rid]
        while len(self.memories)>=self.max_memories:
            old=self.order.pop(0); del self.memories[old]; gc.collect()
        p=prepared_paths(rid); run=self.load_core(rid)
        mem=CausalFusedVoxelMemory(p['geometry'],float(run.data['crop_start_original']))
        mem.haptic=CausalHapticMemory(p['haptic'],float(run.data['crop_start_original']))
        self.memories[rid]=mem; self.order.append(rid); return mem


def dataloader_for(store,rid,split,normalizer,shuffle,seed):
    run=store.load_core(rid); origins=make_origins(run,split,seed)
    ds=StrictTrajectoryWindowDataset(run,store.events[rid],store.waves[rid],normalizer,origins=origins)
    gen=torch.Generator(); gen.manual_seed(seed)
    return torch.utils.data.DataLoader(ds,batch_size=TRAIN_BATCH_SIZE,shuffle=shuffle,num_workers=0,
                                       pin_memory=torch.cuda.is_available(),generator=gen,drop_last=False)


def proposal_forward(model,batch):
    batch,target=batch_to_device(batch)
    out=model.propose(batch['leg_hist'],batch['body_hist'],batch['action_plan'],batch['foot_now'])
    return out,target,batch


def epoch_pass(model,store,run_ids,split,normalizer,optimizer=None,stage='G4',seed=SEED,max_batches=None,
               geometry_mode='uncertainty_patch',haptic_enabled=True):
    training=optimizer is not None; model.train(training); rows=[]
    for rid in run_ids:
        loader=dataloader_for(store,rid,split,normalizer,training,seed)
        memory=None if stage=='G2' else store.memory(rid)
        providers={} if memory is None else {
            'zero_shot':StrictPatchProvider(memory,'zero_shot',geometry_mode=geometry_mode),
            'causal':StrictPatchProvider(memory,'causal',geometry_mode=geometry_mode)}
        for bi,batch in enumerate(loader):
            if max_batches is not None and bi>=max_batches: break
            with torch.set_grad_enabled(training):
                if stage=='G2':
                    out,target,_=proposal_forward(model,batch)
                else:
                    use_zero=(not training) or (not haptic_enabled) or (((bi+seed)%100)/100.<HAPTIC_ZERO_SHOT_TRAIN_FRACTION)
                    provider=providers['zero_shot' if use_zero else 'causal']
                    out,target,_=strict_forward(model,batch,provider)
                loss=strict_v10r_loss(out,target,stage=stage)
                if training:
                    optimizer.zero_grad(set_to_none=True); loss['total'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
            rows.append({k:float(v.detach().cpu()) for k,v in loss.items()})
        if memory is not None: del providers,memory
    return {k:float(np.mean([r[k] for r in rows])) for k in rows[0]} if rows else {'total':np.nan}


def save_normalizer(path,norm): atomic_savez(path,**norm.as_dict())


def load_normalizer(path):
    with np.load(path,allow_pickle=False) as z: return InputNormalizer(**{k:np.asarray(z[k]) for k in z.files})


def train_one_seed(seed,store,train_ids,val_ids,experiment_name='primary',
                   geometry_mode='uncertainty_patch',haptic_enabled=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    outdir=(MODEL_ROOT/f'seed_{seed}' if experiment_name=='primary' else MODEL_ROOT/'ablations'/experiment_name/f'seed_{seed}')
    outdir.mkdir(parents=True,exist_ok=True)
    norm_path=outdir/'input_normalizer.npz'
    normalizer=load_normalizer(norm_path) if norm_path.exists() else fit_input_normalizer([store.load_core(x) for x in train_ids])
    if not norm_path.exists(): save_normalizer(norm_path,normalizer)
    waveform_samples=int(round((IMPACT_PRE_S+IMPACT_POST_S)*IMPACT_WAVEFORM_HZ))+1
    model=StrictV10R(waveform_samples=waveform_samples).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    history=[]
    stages=[('G2',TRAIN_EPOCHS_G2),('G4',TRAIN_EPOCHS_G4)]
    for stage,epochs in stages:
        best=np.inf; patience=0; best_path=outdir/f'best_{stage}.pt'
        for epoch in range(int(epochs)):
            tr=epoch_pass(model,store,train_ids,'train',normalizer,opt,stage,seed+epoch,
                          geometry_mode=geometry_mode,haptic_enabled=haptic_enabled)
            va=epoch_pass(model,store,val_ids,'val',normalizer,None,stage,seed,max_batches=None,
                          geometry_mode=geometry_mode,haptic_enabled=haptic_enabled) if val_ids else {'total':np.nan}
            row={'seed':seed,'experiment':experiment_name,'geometry_mode':geometry_mode,'haptic_enabled':haptic_enabled,
                 'stage':stage,'epoch':epoch,**{'train_'+k:v for k,v in tr.items()},**{'val_'+k:v for k,v in va.items()}}
            history.append(row); print(row)
            score=va.get('total',tr['total']) if np.isfinite(va.get('total',np.nan)) else tr['total']
            payload={'pipeline_version':STRICT_PIPELINE_VERSION,'seed':seed,'stage':stage,'epoch':epoch,
                     'model':model.state_dict(),'optimizer':opt.state_dict(),'cfg':dataclasses.asdict(CFG),
                     'split_sha256':file_sha256(SPLIT_CONTRACT_PATH),'normalizer_sha256':file_sha256(norm_path),
                     'analysis_tier':ANALYSIS_TIER,'paper_data_gate':DATA_GATE,
                     'experiment':{'name':experiment_name,'geometry_mode':geometry_mode,'haptic_enabled':haptic_enabled}}
            torch.save(payload,outdir/'last.pt')
            if score<best:
                best=score; patience=0; torch.save(payload,best_path)
            else: patience+=1
            pd.DataFrame(history).to_csv(outdir/'training_history.csv',index=False)
            if stage=='G4' and patience>=8: print('early stopping',seed,stage,epoch); break
        ck=torch.load(best_path,map_location=DEVICE,weights_only=False); model.load_state_dict(ck['model'])
    # The returned model is the best G4 checkpoint.
    return model,pd.DataFrame(history),normalizer


def train_all_seeds():
    assert_training_authorized()
    train_ids=SPLIT_FRAME.query("split == 'train'").run_id.tolist(); val_ids=SPLIT_FRAME.query("split == 'val'").run_id.tolist()
    if not train_ids or not val_ids: raise RuntimeError('training requires nonempty train and validation trajectories')
    store=PreparedStore(SPLIT_FRAME,max_memories=1); outputs={}
    seeds=TRAIN_SEEDS if ANALYSIS_TIER=='paper' else TRAIN_SEEDS[:1]
    for seed in seeds: outputs[seed]=train_one_seed(seed,store,train_ids,val_ids)
    return outputs


GEOMETRY_ABLATIONS=(
    ('no_map','no_map',False),
    ('deterministic_point','deterministic_point',False),
    ('deterministic_patch','deterministic_patch',False),
    ('uncertainty_patch_no_haptic','uncertainty_patch',False),
)


def train_required_geometry_ablations():
    assert_training_authorized(); store=PreparedStore(SPLIT_FRAME,max_memories=1)
    train_ids=SPLIT_FRAME.query("split == 'train'").run_id.tolist(); val_ids=SPLIT_FRAME.query("split == 'val'").run_id.tolist()
    seeds=TRAIN_SEEDS if ANALYSIS_TIER=='paper' else TRAIN_SEEDS[:1]; outputs={}
    for name,mode,haptic in GEOMETRY_ABLATIONS:
        outputs[name]={}
        for seed in seeds:
            outputs[name][seed]=train_one_seed(seed,store,train_ids,val_ids,name,mode,haptic)
    return outputs

# %% notebook cell 54
# STRICT S15 — prediction collection, conformal calibration, and complete metric tables
LEG_SENSOR_NAMES=('q1','q2','q3','dq1','dq2','dq3','tau1','tau2','tau3')
LEG_SENSOR_UNITS=('rad','rad','rad','rad/s','rad/s','rad/s','N m','N m','N m')
BODY_SENSOR_NAMES=('acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z','v_x','v_y','v_z','omega_x','omega_y','omega_z')
BODY_SENSOR_UNITS=('m/s^2','m/s^2','m/s^2','rad/s','rad/s','rad/s','m/s','m/s','m/s','rad/s','rad/s','rad/s')


def load_trained_seed(seed,experiment_name='primary'):
    out=(MODEL_ROOT/f'seed_{seed}' if experiment_name=='primary' else MODEL_ROOT/'ablations'/experiment_name/f'seed_{seed}')
    ck=out/'best_G4.pt'; norm=out/'input_normalizer.npz'
    if not ck.exists() or not norm.exists(): raise FileNotFoundError(f'seed {seed}: missing trained checkpoint/normalizer')
    payload=torch.load(ck,map_location=DEVICE,weights_only=False)
    if payload.get('split_sha256')!=file_sha256(SPLIT_CONTRACT_PATH): raise RuntimeError(f'seed {seed}: split hash changed')
    W=int(round((IMPACT_PRE_S+IMPACT_POST_S)*IMPACT_WAVEFORM_HZ))+1
    model=StrictV10R(waveform_samples=W).to(DEVICE); model.load_state_dict(payload['model']); model.eval()
    return model,load_normalizer(norm),payload


def collect_predictions(model,normalizer,store,rid,split,haptic_mode='zero_shot',geometry_mode='uncertainty_patch'):
    loader=dataloader_for(store,rid,split,normalizer,False,SEED)
    provider=StrictPatchProvider(store.memory(rid),haptic_mode=haptic_mode,geometry_mode=geometry_mode)
    keys=('force_mean','free_force','dynamics_only_force','force_log_scale','foot_mean','foot_covariance','impulse','contact_prob','mode_logits',
          'event_hazard_logit','event_offset_s','event_offset_log_scale','impact_waveform_mean','impact_waveform_log_scale',
          'wrench_residual','dynamics_only_wrench_residual','desired_wrench_pred','surface_normal','surface_mu','foot_r',
          'map_confidence','leg_sensor_raw','body_raw')
    target_keys=('phase','force','foot','impulse','leg_sensor','body_sensor','wrench_target','event_hazard',
                 'event_offset_s','event_mask','impact_waveform','impact_mask')
    outbuf={k:[] for k in keys}; ybuf={k:[] for k in target_keys}; origins=[]; indices=[]
    with torch.no_grad():
        for batch in loader:
            idx=batch['origin_index'].numpy(); out,target,batch=strict_forward(model,batch,provider)
            origins.append(batch['origin_t'].detach().cpu().numpy()); indices.append(idx)
            for k in keys: outbuf[k].append(out[k].detach().cpu().numpy())
            for k in target_keys: ybuf[k].append(target[k].detach().cpu().numpy())
    pred={k:np.concatenate(v,axis=0) for k,v in outbuf.items()}
    target={k:np.concatenate(v,axis=0) for k,v in ybuf.items()}
    pred['force_scale']=np.exp(pred['force_log_scale']); pred['leg_sensor_mean']=pred['leg_sensor_raw'][...,:9]
    pred['leg_sensor_scale']=np.exp(pred['leg_sensor_raw'][...,9:]); pred['body_sensor_mean']=pred['body_raw'][...,:12]
    pred['body_sensor_scale']=np.exp(pred['body_raw'][...,12:])
    return {'run_id':rid,'haptic_mode':haptic_mode,'geometry_mode':geometry_mode,
            'origin_t':np.concatenate(origins),'origin_index':np.concatenate(indices),
            'pred':pred,'target':target}


def finite_sample_quantile(x,coverage):
    x=np.sort(np.asarray(x,float)); n=len(x)
    if n==0: return np.nan
    k=min(n,int(math.ceil((n+1)*coverage))); return float(x[k-1])


def fit_force_scale_calibrator(validation_predictions,coverages=(.80,.90,.95),min_group=30):
    rows=[]
    for h,hs in enumerate(CFG.horizons_s):
        for l,leg in enumerate(LEG_NAMES):
            for a,axis in enumerate('xyz'):
                for ph,phase_name in enumerate(('swing','touchdown','stance','liftoff')):
                    scores=[]
                    for pack in validation_predictions:
                        m=pack['target']['phase'][:,h,l]==ph
                        e=np.abs(pack['target']['force'][m,h,l,a]-pack['pred']['force_mean'][m,h,l,a])
                        s=np.maximum(pack['pred']['force_scale'][m,h,l,a],1e-5); scores.extend((e/s).tolist())
                    for cov in coverages:
                        rows.append({'horizon_s':hs,'leg':leg,'axis':axis,'phase':phase_name,
                                     'coverage_nominal':cov,'n_calibration':len(scores),
                                     'scale_quantile':finite_sample_quantile(scores,cov) if len(scores)>=min_group else np.nan,
                                     'adequate_group':len(scores)>=min_group})
    return pd.DataFrame(rows)


def calibrator_lookup(frame):
    return {(float(r.horizon_s),r.leg,r.axis,r.phase,float(r.coverage_nominal)):float(r.scale_quantile)
            for _,r in frame.iterrows() if np.isfinite(r.scale_quantile)}


def force_metrics(pack,model_name,calibrator=None):
    y=pack['target']['force']; p=pack['pred']['force_mean']; s=pack['pred']['force_scale']; phase=pack['target']['phase']
    times=pack['origin_t']; lookup=calibrator_lookup(calibrator) if calibrator is not None else {}; rows=[]; vector_rows=[]
    for h,hs in enumerate(CFG.horizons_s):
        for l,leg in enumerate(LEG_NAMES):
            for ph,pname in enumerate(('swing','touchdown','stance','liftoff')):
                m=phase[:,h,l]==ph
                if not m.any(): continue
                ve=np.linalg.norm(p[m,h,l]-y[m,h,l],axis=1)
                vlo,vhi=moving_block_ci(ve,times[m],stat=lambda q:np.quantile(q,.95),seed=SEED+1000+h*31+l*7+ph)
                vector_rows.append({'run_id':pack['run_id'],'model':model_name,'horizon_s':hs,'leg':leg,'phase':pname,
                                    'n':len(ve),'vector_mae_n':ve.mean(),'vector_p95_n':np.quantile(ve,.95),
                                    'vector_p99_n':np.quantile(ve,.99),'vector_max_n':ve.max(),
                                    'vector_p95_ci_low_n':vlo,'vector_p95_ci_high_n':vhi})
                for a,axis in enumerate('xyz'):
                    e=np.abs(p[m,h,l,a]-y[m,h,l,a]); lo,hi=moving_block_ci(e,times[m],stat=lambda q:np.quantile(q,.95),
                        seed=SEED+h*101+l*17+a*3+ph)
                    row={'run_id':pack['run_id'],'model':model_name,'horizon_s':hs,'leg':leg,'axis':axis,'phase':pname,'n':len(e),
                         'mae_n':e.mean(),'median_n':np.median(e),'p95_n':np.quantile(e,.95),'p99_n':np.quantile(e,.99),
                         'within_3n':np.mean(e<=3),'within_4n':np.mean(e<=4),'max_n':e.max(),
                         'p95_ci_low_n':lo,'p95_ci_high_n':hi}
                    for cov in (.80,.90,.95):
                        q=lookup.get((float(hs),leg,axis,pname,cov),np.nan); width=2*q*s[m,h,l,a]
                        row[f'coverage_{int(cov*100)}']=float(np.mean(e<=q*s[m,h,l,a])) if np.isfinite(q) else np.nan
                        row[f'width_{int(cov*100)}_n']=float(np.mean(width)) if np.isfinite(q) else np.nan
                    rows.append(row)
    return pd.DataFrame(rows),pd.DataFrame(vector_rows)


def persistence_prediction(pack,run):
    idx=pack['origin_index']; H=len(CFG.horizons_s); d=run.data
    def repeat(x): return np.repeat(np.asarray(x)[:,None],H,axis=1)
    return {'force_mean':repeat(d['force_input_H'][idx]),'force_scale':np.ones((len(idx),H,4,3),np.float32),
            'leg_sensor_mean':repeat(d['P'][idx,:,:9]),'body_sensor_mean':repeat(d['B'][idx]),
            'foot_mean':repeat(d['foot_world'][idx]),'contact_prob':repeat(d['contact'][idx])}


def sensor_metrics(pack,model_name,pred_override=None):
    pred=pack['pred'] if pred_override is None else pred_override; target=pack['target']; rows=[]
    for h,hs in enumerate(CFG.horizons_s):
        for l,leg in enumerate(LEG_NAMES):
            for j,(name,unit) in enumerate(zip(LEG_SENSOR_NAMES,LEG_SENSOR_UNITS)):
                e=np.abs(pred['leg_sensor_mean'][:,h,l,j]-target['leg_sensor'][:,h,l,j])
                rows.append({'run_id':pack['run_id'],'model':model_name,'family':'leg','leg':leg,'sensor':name,'unit':unit,
                             'horizon_s':hs,'n':len(e),'mae':e.mean(),'median':np.median(e),'p95':np.quantile(e,.95),'p99':np.quantile(e,.99),'max':e.max()})
        for j,(name,unit) in enumerate(zip(BODY_SENSOR_NAMES,BODY_SENSOR_UNITS)):
            e=np.abs(pred['body_sensor_mean'][:,h,j]-target['body_sensor'][:,h,j])
            rows.append({'run_id':pack['run_id'],'model':model_name,'family':'body','leg':'BODY','sensor':name,'unit':unit,
                         'horizon_s':hs,'n':len(e),'mae':e.mean(),'median':np.median(e),'p95':np.quantile(e,.95),'p99':np.quantile(e,.99),'max':e.max()})
        fe=np.linalg.norm(pred['foot_mean'][:,h]-target['foot'][:,h],axis=-1)
        for l,leg in enumerate(LEG_NAMES):
            e=fe[:,l]; rows.append({'run_id':pack['run_id'],'model':model_name,'family':'kinematic','leg':leg,'sensor':'foot_position_3d',
                'unit':'m','horizon_s':hs,'n':len(e),'mae':e.mean(),'median':np.median(e),'p95':np.quantile(e,.95),'p99':np.quantile(e,.99),'max':e.max()})
    return pd.DataFrame(rows)


def event_impulse_waveform_metrics(pack,model_name):
    p=pack['pred']; y=pack['target']; rows=[]
    prob=1/(1+np.exp(-p['event_hazard_logit'])); pred_event=prob>=.5; truth=y['event_hazard']>.5
    tp=(pred_event&truth).sum(); fp=(pred_event&~truth).sum(); fn=(~pred_event&truth).sum()
    precision=tp/max(tp+fp,1); recall=tp/max(tp+fn,1); f1=2*precision*recall/max(precision+recall,1e-12)
    mask=y['event_mask'].astype(bool)
    time_error=np.abs(p['event_offset_s'][mask]-y['event_offset_s'][mask])
    imp_err=np.linalg.norm(p['impulse'][mask]-y['impulse'][mask],axis=-1) if mask.any() else np.array([])
    imp_rel=imp_err/np.maximum(np.linalg.norm(y['impulse'][mask],axis=-1),1e-6) if mask.any() else np.array([])
    wave_err=np.abs(p['impact_waveform_mean'][mask]-y['impact_waveform'][mask]).reshape(-1) if mask.any() else np.array([])
    rows.append({'run_id':pack['run_id'],'model':model_name,'events':int(mask.sum()),'event_precision':precision,'event_recall':recall,'event_f1':f1,
                 'event_time_mae_ms':float(1000*time_error.mean()) if len(time_error) else np.nan,
                 'event_time_median_abs_ms':float(1000*np.median(time_error)) if len(time_error) else np.nan,
                 'event_time_p95_abs_ms':float(1000*np.quantile(time_error,.95)) if len(time_error) else np.nan,
                 'impulse_vector_mae_ns':float(imp_err.mean()) if len(imp_err) else np.nan,
                 'impulse_relative_median':float(np.median(imp_rel)) if len(imp_rel) else np.nan,
                 'impulse_relative_mean':float(imp_rel.mean()) if len(imp_rel) else np.nan,
                 'waveform_component_mae_n':float(wave_err.mean()) if len(wave_err) else np.nan,
                 'waveform_component_p95_n':float(np.quantile(wave_err,.95)) if len(wave_err) else np.nan})
    return pd.DataFrame(rows)


def force_prediction_rows(pack,model_name):
    rows=[]; y=pack['target']; p=pack['pred']
    for n,(origin_t,origin_i) in enumerate(zip(pack['origin_t'],pack['origin_index'])):
        for h,hs in enumerate(CFG.horizons_s):
            for l,leg in enumerate(LEG_NAMES):
                ph=('swing','touchdown','stance','liftoff')[int(y['phase'][n,h,l])]
                for a,axis in enumerate('xyz'):
                    rows.append((pack['run_id'],model_name,origin_t,int(origin_i),hs,leg,axis,ph,
                                 y['force'][n,h,l,a],p['force_mean'][n,h,l,a],p['force_scale'][n,h,l,a],
                                 p['map_confidence'][n,h,l]))
    return pd.DataFrame(rows,columns=['run_id','model','origin_t','origin_index','horizon_s','leg','axis','phase',
                                     'target_n','prediction_n','predicted_scale_n','map_confidence'])

# %% notebook cell 55
# STRICT S16 — zero-shot/prequential seed and ensemble evaluation with claim gates
def _ensemble_pack(packs):
    base=packs[0]; pred=base['pred']; means=np.stack([x['pred']['force_mean'] for x in packs])
    aleatoric=np.stack([x['pred']['force_scale']**2 for x in packs]).mean(0)
    averaged=('foot_mean','impulse','contact_prob','event_hazard_logit','event_offset_s',
              'impact_waveform_mean','leg_sensor_mean','body_sensor_mean','map_confidence')
    ensemble={k:np.mean(np.stack([x['pred'][k] for x in packs]),axis=0) for k in averaged}
    ensemble['force_mean']=means.mean(0); ensemble['force_scale']=np.sqrt(aleatoric+means.var(0))
    for k in ('event_offset_log_scale','impact_waveform_log_scale','mode_logits','wrench_residual'):
        ensemble[k]=np.mean(np.stack([x['pred'][k] for x in packs]),axis=0)
    return {**base,'pred':{**pred,**ensemble}}


def evaluate_trained_models():
    assert_training_authorized(); store=PreparedStore(SPLIT_FRAME,max_memories=1)
    val_ids=SPLIT_FRAME.query("split == 'val'").run_id.tolist(); test_ids=SPLIT_FRAME.query("split == 'test'").run_id.tolist()
    if not val_ids or not test_ids: raise RuntimeError('evaluation needs validation and test trajectories')
    seeds=TRAIN_SEEDS if ANALYSIS_TIER=='paper' else TRAIN_SEEDS[:1]
    all_force=[]; all_vector=[]; all_sensor=[]; all_event=[]; prediction_index=[]
    test_packs={mode:defaultdict(dict) for mode in ('zero_shot','prequential')}
    val_packs={mode:defaultdict(dict) for mode in ('zero_shot','prequential')}
    persistence_done=set()
    for seed in seeds:
        model,norm,_=load_trained_seed(seed); cdir=RESULT_ROOT/f'seed_{seed}'; cdir.mkdir(parents=True,exist_ok=True)
        for mode in ('zero_shot','prequential'):
            vals=[collect_predictions(model,norm,store,rid,'val',mode) for rid in val_ids]
            for x in vals: val_packs[mode][x['run_id']][seed]=x
            calibrator=fit_force_scale_calibrator(vals); calibrator.to_csv(cdir/f'force_calibrator_{mode}.csv',index=False)
            if not calibrator.adequate_group.all(): warnings.warn(f'seed {seed}/{mode}: some conformal groups have <30 validation samples')
            for rid in test_ids:
                pack=collect_predictions(model,norm,store,rid,'test',mode); test_packs[mode][rid][seed]=pack
                label=f'V10R_seed_{seed}_{mode}'; fm,vm=force_metrics(pack,label,calibrator)
                sm=sensor_metrics(pack,label); em=event_impulse_waveform_metrics(pack,label)
                all_force.append(fm); all_vector.append(vm); all_sensor.append(sm); all_event.append(em)
                if rid not in persistence_done:
                    run=store.load_core(rid); base=persistence_prediction(pack,run); bm=sensor_metrics(pack,'persistence',base)
                    base_pack={**pack,'pred':{**pack['pred'],**base}}; bfm,bvm=force_metrics(base_pack,'persistence',None)
                    dyn_pack={**pack,'pred':{**pack['pred'],'force_mean':pack['pred']['dynamics_only_force']}}
                    dfm,dvm=force_metrics(dyn_pack,'dynamics_only_projector',None)
                    all_force.extend((bfm,dfm)); all_vector.extend((bvm,dvm)); all_sensor.append(bm); persistence_done.add(rid)
                rows=force_prediction_rows(pack,label)
                try:
                    ensure_python_packages([('pyarrow','pyarrow')]); pred_path=cdir/f'{rid}_{mode}_force_predictions.parquet'; rows.to_parquet(pred_path,index=False)
                except Exception:
                    pred_path=cdir/f'{rid}_{mode}_force_predictions.csv.gz'; rows.to_csv(pred_path,index=False,compression='gzip')
                complete=cdir/f'{rid}_{mode}_predictions_complete.npz'
                np.savez_compressed(complete,origin_t=pack['origin_t'],origin_index=pack['origin_index'],
                    **{f'pred_{k}':v for k,v in pack['pred'].items()},**{f'target_{k}':v for k,v in pack['target'].items()})
                prediction_index.append({'seed':seed,'run_id':rid,'haptic_mode':mode,
                                         'force_rows':str(pred_path),'complete_npz':str(complete)})
        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Fit one validation-only calibrator and evaluate one ensemble for each declared haptic regime.
    for mode in ('zero_shot','prequential'):
        ensemble_val=[_ensemble_pack([val_packs[mode][rid][s] for s in seeds]) for rid in val_ids]
        cal=fit_force_scale_calibrator(ensemble_val); cal.to_csv(RESULT_ROOT/f'ensemble_force_calibrator_{mode}.csv',index=False)
        for rid in test_ids:
            epack=_ensemble_pack([test_packs[mode][rid][s] for s in seeds]); label=f'V10R_ensemble_{mode}'
            fm,vm=force_metrics(epack,label,cal); sm=sensor_metrics(epack,label); em=event_impulse_waveform_metrics(epack,label)
            all_force.append(fm); all_vector.append(vm); all_sensor.append(sm); all_event.append(em)

    force=pd.concat(all_force,ignore_index=True); vector=pd.concat(all_vector,ignore_index=True)
    sensor=pd.concat(all_sensor,ignore_index=True); event=pd.concat(all_event,ignore_index=True)
    force.to_csv(RESULT_ROOT/'force_metrics_long.csv',index=False); vector.to_csv(RESULT_ROOT/'force_vector_metrics_long.csv',index=False)
    sensor.to_csv(RESULT_ROOT/'all_sensor_metrics_long.csv',index=False); event.to_csv(RESULT_ROOT/'event_impulse_waveform_metrics.csv',index=False)
    pd.DataFrame(prediction_index).to_csv(RESULT_ROOT/'prediction_artifact_index.csv',index=False)
    gate=compute_claim_gates(force,sensor,event); atomic_write_json(RESULT_ROOT/'claim_gates.json',gate)
    write_results_summary(force,sensor,event,gate)
    display(force); display(sensor); display(event); print(json.dumps(gate,indent=2))
    return {'force':force,'vector':vector,'sensor':sensor,'event':event,'gates':gate,'packs':test_packs}


def write_results_summary(force,sensor,event,gates):
    primary='V10R_ensemble_zero_shot'; stance=force[(force.model==primary)&(force.phase=='stance')]
    ev=event[event.model==primary]; lines=['# V10-R strict real-data results','',
        f"Generated: {utc_now()}",f"Paper data gate: **{'PASS' if gates['paper_data_gate'] else 'FAIL'}**",'',
        '## Primary zero-shot force result','']
    if len(stance):
        lines += [f"- Worst stable-stance component p95: {stance.p95_n.max():.4f} N",
                  f"- Worst stable-stance block-bootstrap p95 upper bound: {stance.p95_ci_high_n.max():.4f} N"]
    else: lines += ['- No primary stable-stance rows were produced.']
    lines += ['','## Touchdown','']
    if len(ev):
        lines += [f"- Median absolute event-time error (worst test run): {ev.event_time_median_abs_ms.max():.4f} ms",
                  f"- Median relative impulse error (worst test run): {ev.impulse_relative_median.max():.4%}",
                  f"- Event recall (worst test run): {ev.event_recall.min():.4%}"]
    else: lines += ['- No primary touchdown rows were produced.']
    lines += ['','## Claim decision','']+[f"- {k}: **{'PASS' if v else 'FAIL'}**" for k,v in gates.items() if isinstance(v,bool)]
    lines += ['','No result is eligible for an abstract claim unless its complete pre-registered gate is PASS.']
    (RESULT_ROOT/'RESULTS_SUMMARY.md').write_text('\n'.join(lines)+'\n')


def compute_claim_gates(force,sensor,event):
    primary='V10R_ensemble_zero_shot'
    headline=force[(force.model==primary)&(force.phase=='stance')]
    stance_p95=bool(len(headline)>0 and (headline.p95_n<=CFG.target_stance_p95_n).all())
    stance_ucb=bool(len(headline)>0 and (headline.p95_ci_high_n<=5.).all())
    ev=event[event.model==primary]
    timing=bool(len(ev)>0 and (ev.event_time_median_abs_ms<=2.).all())
    impulse=bool(len(ev)>0 and (ev.impulse_relative_median<=CFG.target_touchdown_impulse_rel).all())
    event_detect=bool(len(ev)>0 and (ev.event_recall>=.90).all())
    calibrated=True
    for c in (80,90,95):
        x=headline[f'coverage_{c}'].dropna(); calibrated &= bool(len(x)>0 and (np.abs(x-c/100)<=.03).all())
    model_s=sensor[sensor.model==primary].groupby('horizon_s').mae.mean()
    base_s=sensor[sensor.model=='persistence'].groupby('horizon_s').mae.mean()
    persistence_win=bool(len(model_s)==len(CFG.horizons_s) and all(model_s.get(h,np.inf)<base_s.get(h,-np.inf) for h in CFG.horizons_s))
    ablation_status=RESULT_ROOT/'required_geometry_ablation_status.json'
    ablation_complete=bool(json.load(open(ablation_status)).get('complete')) if ablation_status.exists() else False
    gates={'primary_regime':'zero_shot','paper_data_gate':bool(DATA_GATE['pass']),
           'stable_stance_component_p95_le_4n':stance_p95,'stable_stance_block_bootstrap_ucb_le_5n':stance_ucb,
           'touchdown_event_recall_ge_90pct':event_detect,'touchdown_median_time_le_2ms':timing,
           'touchdown_impulse_relative_le_10pct':impulse,'phase_interval_calibration_within_3pct':calibrated,
           'beats_persistence_every_horizon':persistence_win,'v7f_baseline_supplied':False,
           'dynamics_only_baseline_supplied':bool((force.model=='dynamics_only_projector').any()),
           'required_geometry_ablation_complete':ablation_complete,
           'zero_shot_prequential_reported_separately':bool(
               {'V10R_ensemble_zero_shot','V10R_ensemble_prequential'}<=set(force.model)),
           'reset_matched_planner_evidence':False}
    gates['force_claim_allowed']=all(gates[k] for k in ('paper_data_gate','stable_stance_component_p95_le_4n',
        'stable_stance_block_bootstrap_ucb_le_5n','touchdown_event_recall_ge_90pct','touchdown_median_time_le_2ms',
        'touchdown_impulse_relative_le_10pct','phase_interval_calibration_within_3pct'))
    gates['all_sensor_superiority_claim_allowed']=gates['paper_data_gate'] and persistence_win and gates['v7f_baseline_supplied'] and gates['dynamics_only_baseline_supplied']
    gates['geometry_interface_claim_allowed']=gates['paper_data_gate'] and gates['required_geometry_ablation_complete']
    gates['planner_claim_allowed']=gates['reset_matched_planner_evidence']
    return gates

# %% notebook cell 57
# STRICT S17 — publication figures from saved artifacts only
def savefig(fig,name):
    FIGURE_ROOT.mkdir(parents=True,exist_ok=True); path=FIGURE_ROOT/name
    fig.tight_layout(); fig.savefig(path,dpi=220,bbox_inches='tight'); plt.close(fig); return path


def plot_training_history():
    paths=sorted(MODEL_ROOT.glob('seed_*/training_history.csv'))
    if not paths: raise FileNotFoundError('no real training histories')
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=False)
    for p in paths:
        d=pd.read_csv(p); seed=p.parent.name.replace('seed_','')
        for ax,stage in zip(axes,('G2','G4')):
            g=d[d.stage==stage]; ax.plot(np.arange(len(g)),g.train_total,label=f'{seed} train',alpha=.8)
            if 'val_total' in g: ax.plot(np.arange(len(g)),g.val_total,'--',label=f'{seed} val',alpha=.8)
            ax.set(title=stage,xlabel='epoch',ylabel='loss'); ax.grid(alpha=.25)
    axes[1].legend(fontsize=7,ncol=2,bbox_to_anchor=(1.02,1),loc='upper left')
    return savefig(fig,'training_curves_all_seeds.png')


def plot_force_horizon_metrics(force):
    g=force[force.model=='V10R_ensemble_zero_shot']
    if len(g)==0: g=force[force.model.str.startswith('V10R_seed_')]
    if len(g)==0: raise ValueError('no V10-R force rows')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for phase,c in zip(('swing','touchdown','stance','liftoff'),('#4C78A8','#E45756','#54A24B','#B279A2')):
        x=g[g.phase==phase].groupby('horizon_s').agg(mae=('mae_n','mean'),p95=('p95_n','max'))
        axes[0].plot(x.index,x.mae,marker='o',label=phase,color=c); axes[1].plot(x.index,x.p95,marker='o',label=phase,color=c)
    axes[0].set(ylabel='component MAE [N]'); axes[1].set(ylabel='worst leg/axis component p95 [N]')
    for ax in axes: ax.set(xlabel='forecast horizon [s]'); ax.grid(alpha=.25); ax.legend()
    axes[1].axhline(4,color='black',ls=':',label='4 N gate')
    return savefig(fig,'force_error_vs_horizon_by_phase.png')


def plot_stance_p95_heatmap(force):
    g=force[(force.model=='V10R_ensemble_zero_shot')&(force.phase=='stance')]
    if len(g)==0: g=force[(force.model.str.startswith('V10R_seed_'))&(force.phase=='stance')]
    table=g.groupby(['leg','axis','horizon_s']).p95_n.max().unstack('horizon_s')
    fig,ax=plt.subplots(figsize=(10,4)); im=ax.imshow(table.values,aspect='auto',cmap='magma',vmin=0)
    ax.set_yticks(np.arange(len(table))); ax.set_yticklabels([f'{a}/{b}' for a,b in table.index])
    ax.set_xticks(np.arange(len(table.columns))); ax.set_xticklabels([f'{x:.1f}' for x in table.columns])
    ax.set(xlabel='horizon [s]',title='Stable-stance component p95 [N] (worst seed/run aggregation)')
    for i in range(table.shape[0]):
        for j in range(table.shape[1]): ax.text(j,i,f'{table.iloc[i,j]:.1f}',ha='center',va='center',fontsize=7,color='white')
    fig.colorbar(im,ax=ax,label='N'); return savefig(fig,'stable_stance_p95_heatmap.png')


def plot_calibration(force):
    g=force[force.model=='V10R_ensemble_zero_shot']; rows=[]
    for c in (80,90,95):
        x=g[f'coverage_{c}'].dropna(); rows.append((c/100,float(x.mean()) if len(x) else np.nan))
    fig,ax=plt.subplots(figsize=(5,5)); x=np.array([r[0] for r in rows]); y=np.array([r[1] for r in rows])
    ax.plot([.75,1],[.75,1],'k--'); ax.plot(x,y,'o-',lw=2); ax.set(xlim=(.75,1),ylim=(.75,1),
        xlabel='nominal coverage',ylabel='empirical coverage',title='Phase-conditioned force intervals'); ax.grid(alpha=.25)
    return savefig(fig,'force_interval_calibration.png')


def load_complete_prediction(seed,run_id,haptic_mode='zero_shot'):
    p=RESULT_ROOT/f'seed_{seed}'/f'{run_id}_{haptic_mode}_predictions_complete.npz'
    if not p.exists(): raise FileNotFoundError(p)
    with np.load(p,allow_pickle=False) as z: return {k:np.asarray(z[k]) for k in z.files}


def plot_force_trace(seed,run_id,horizon_index=0,leg='FL'):
    d=load_complete_prediction(seed,run_id); li=LEG_NAMES.index(leg); t=d['origin_t'];
    fig,axes=plt.subplots(3,1,figsize=(11,7),sharex=True)
    for a,(axis,ax) in enumerate(zip('xyz',axes)):
        y=d['target_force'][:,horizon_index,li,a]; p=d['pred_force_mean'][:,horizon_index,li,a]
        s=np.exp(d['pred_force_log_scale'][:,horizon_index,li,a])
        ax.plot(t,y,color='black',lw=.8,label='target'); ax.plot(t,p,color='#4C78A8',lw=.8,label='prediction')
        ax.fill_between(t,p-1.96*s,p+1.96*s,color='#4C78A8',alpha=.18,label='raw 95% scale' if a==0 else None)
        ax.set_ylabel(f'F{axis} [N]'); ax.grid(alpha=.2)
    axes[0].legend(ncol=3); axes[-1].set_xlabel('forecast origin time [s]')
    fig.suptitle(f'{run_id} · {leg} · h={CFG.horizons_s[horizon_index]:.1f}s · seed {seed}')
    return savefig(fig,f'{run_id}_{leg}_h{horizon_index+1:02d}_force_trace.png')


def plot_impact_waveform(seed,run_id,max_events=20):
    d=load_complete_prediction(seed,run_id); mask=d['target_impact_mask'].astype(bool)
    y=d['target_impact_waveform'][mask]; p=d['pred_impact_waveform_mean'][mask]
    if len(y)==0: raise ValueError('no touchdown waveform targets in evaluated origins')
    y=y[:max_events]; p=p[:max_events]; wt=np.linspace(-IMPACT_PRE_S,IMPACT_POST_S,y.shape[1])*1000
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),sharex=True)
    for a,ax in enumerate(axes):
        ax.plot(wt,y[:,:,a].T,color='black',alpha=.12); ax.plot(wt,p[:,:,a].T,color='#E45756',alpha=.12)
        ax.plot(wt,y[:,:,a].mean(0),color='black',lw=2,label='target mean'); ax.plot(wt,p[:,:,a].mean(0),color='#E45756',lw=2,label='pred mean')
        ax.axvline(0,color='gray',ls=':'); ax.set(title=f'F{"xyz"[a]}',xlabel='registered time [ms]',ylabel='N'); ax.grid(alpha=.2)
    axes[0].legend(); return savefig(fig,f'{run_id}_registered_touchdown_waveforms.png')


def plot_geometry_3d(run_id,max_points=60000):
    run,mem,_,_,_=load_prepared_run(run_id,load_memory=True); surf=mem.latest_surface(float(run.data['time'][-1]))
    ids=np.arange(len(surf['xyz']))
    if len(ids)>max_points: ids=np.linspace(0,len(ids)-1,max_points).astype(int)
    xyz=surf['xyz'][ids]; rgb=np.clip(surf['rgb'][ids],0,1)
    fig=plt.figure(figsize=(10,7)); ax=fig.add_subplot(111,projection='3d')
    ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c=rgb,s=.7,alpha=.35,rasterized=True)
    com=run.data['com_H'][::50]; ax.plot(com[:,0],com[:,1],com[:,2],color='black',lw=2,label='COM path')
    colors=('#1f77b4','#ff7f0e','#2ca02c','#d62728')
    for l,(leg,c) in enumerate(zip(LEG_NAMES,colors)):
        p=run.data['foot_world'][::50,l]; ax.plot(p[:,0],p[:,1],p[:,2],color=c,lw=1,label=leg)
    ax.set(xlabel='H forward [m]',ylabel='H left [m]',zlabel='H up [m]',title=f'{run_id}: fused causal support geometry')
    ax.legend(ncol=3); return savefig(fig,f'{run_id}_fused_geometry_3d.png')


def generate_all_result_figures():
    fpath=RESULT_ROOT/'force_metrics_long.csv'
    if not fpath.exists(): raise FileNotFoundError('real evaluation metrics missing; run evaluate first')
    force=pd.read_csv(fpath); paths=[plot_training_history(),plot_force_horizon_metrics(force),
        plot_stance_p95_heatmap(force),plot_calibration(force)]
    test_ids=SPLIT_FRAME.query("split == 'test'").run_id.tolist(); seed=(TRAIN_SEEDS if ANALYSIS_TIER=='paper' else TRAIN_SEEDS[:1])[0]
    for rid in test_ids:
        paths.extend([plot_force_trace(seed,rid),plot_impact_waveform(seed,rid),plot_geometry_3d(rid)])
    print('\n'.join(map(str,paths))); return paths

# %% notebook cell 59
# STRICT S18 — counterfactual candidate decoding and joint-support shadow selection
def signed_support_margin(points_xy,com_xy):
    points=np.unique(np.asarray(points_xy,float),axis=0); com=np.asarray(com_xy,float)
    if len(points)<3: return -np.inf
    from scipy.spatial import ConvexHull
    try: hull=ConvexHull(points); poly=points[hull.vertices]
    except Exception: return -np.inf
    # CCW convex polygon: positive minimum inward distance means COM is inside.
    area=np.sum(poly[:,0]*np.roll(poly[:,1],-1)-poly[:,1]*np.roll(poly[:,0],-1))
    if area<0: poly=poly[::-1]
    margins=[]
    for a,b in zip(poly,np.roll(poly,-1,axis=0)):
        edge=b-a; inward=np.array([-edge[1],edge[0]])/max(np.linalg.norm(edge),1e-9)
        margins.append(np.dot(com-a,inward))
    return float(min(margins))


def repeat_proposal(proposal,n):
    return {k:(v.detach().repeat((n,)+(1,)*(v.ndim-1)) if torch.is_tensor(v) else v) for k,v in proposal.items()}


def candidate_table_for_leg(model,provider,batch,proposal,leg_index,horizon_index,spacing=.02,half_width=.08):
    nominal=proposal['foot_mean'][0,horizon_index,leg_index].detach().cpu().numpy()
    candidates=make_xy_candidates(nominal,half_width=half_width,spacing=spacing)
    origin_t=float(batch['origin_t'][0].detach().cpu())
    for j in range(len(candidates)):
        candidates[j,2]=provider.memory.raycast_support(candidates[j],origin_t)[0]
    n=len(candidates)
    prop=repeat_proposal(proposal,n); delta=torch.as_tensor(candidates-nominal,device=DEVICE,dtype=prop['foot_mean'].dtype)
    prop['foot_mean'][:,horizon_index:,leg_index]+=delta[:,None,:]
    cov=foot_covariance(prop['foot_chol_raw']); origin=batch['origin_t'].repeat(n)
    patch,normal,mu,conf=provider(prop['foot_mean'].detach(),cov.detach(),origin)
    b={k:(v.repeat((n,)+(1,)*(v.ndim-1)) if torch.is_tensor(v) and v.shape[0]==1 else v)
       for k,v in batch.items() if k!='target'}
    with torch.no_grad(): dec=model.decode_strict(prop,patch.to(DEVICE),normal.to(DEVICE),mu.to(DEVICE),b)
    h=horizon_index; l=leg_index; force=dec['force_mean'][:,h,l].cpu().numpy(); scale=np.exp(dec['force_log_scale'][:,h,l].cpu().numpy())
    impulse=dec['impulse'][:,h,l].cpu().numpy(); normals=normal[:,h,l].numpy(); mus=mu[:,h,l].numpy(); confidence=conf[:,h,l].numpy()
    fn=np.sum(force*normals,axis=1); ft=np.linalg.norm(force-fn[:,None]*normals,axis=1)
    force_cvar=np.linalg.norm(force,axis=1)+1.645*np.linalg.norm(scale,axis=1)
    impulse_cvar=np.linalg.norm(impulse,axis=1)
    run_foot=batch['_run_foot_now'][0].cpu().numpy(); contact_now=batch['_contact_now'][0].cpu().numpy()>0.5
    com=dec['com_pred'][:,h].cpu().numpy(); rows=[]
    for j,p in enumerate(candidates):
        support=np.vstack([p[:2] if k==l else run_foot[k,:2] for k in range(4) if contact_now[k] or k==l])
        margin=signed_support_margin(support,com[j,:2]); feat=patch[j,h,l].numpy()
        offset=float(np.linalg.norm(p[:2]-nominal[:2])); reach=max(offset-.10,0.)
        edge=float(np.clip(feat[27],0,1)); ambiguity=float(np.clip(feat[28],0,1)); collision=max(edge,ambiguity)
        rows.append({'leg':LEG_NAMES[l],'candidate_id':j,'x':p[0],'y':p[1],'z':p[2],
                     'collision_prob':collision,'reach_violation':reach,'map_confidence':confidence[j],
                     'support_margin_m':margin,'support_risk':max(-margin,0.)/.05,
                     'slip_prob':float(np.clip(ft[j]/max(mus[j]*max(fn[j],0.),1e-6),0,2)/2),
                     'force_cvar':force_cvar[j]/300.,'impulse_cvar':impulse_cvar[j]/5.,
                     'epistemic':1-confidence[j],'energy':offset/.10,'nominal_offset':offset/.10,
                     'force_cvar_n':force_cvar[j],'impulse_cvar_ns':impulse_cvar[j],
                     'friction_utilization':ft[j]/max(mus[j]*max(fn[j],0.),1e-6)})
    table=pd.DataFrame(rows); table['hard_reject_extra']=table.support_margin_m<0
    scored=score_candidates(table); scored.loc[scored.hard_reject_extra,'score']=np.inf
    return scored.sort_values('score')


def shadow_plan_one_origin(seed,run_id,origin_position=0):
    model,norm,_=load_trained_seed(seed); store=PreparedStore(SPLIT_FRAME,max_memories=1); run=store.load_core(run_id)
    origins=make_origins(run,'test',seed); i=int(origins[origin_position])
    ds=StrictTrajectoryWindowDataset(run,store.events[run_id],store.waves[run_id],norm,origins=[i])
    batch=torch.utils.data.default_collate([ds[0]]); contact_now=torch.as_tensor(run.data['contact'][i:i+1])
    foot_now=torch.as_tensor(run.data['foot_world'][i:i+1]); batch['_contact_now']=contact_now; batch['_run_foot_now']=foot_now
    # Move only canonical model inputs; keep planner context tensors available.
    batch,target=batch_to_device(batch); batch['_contact_now']=batch['_contact_now'].to(DEVICE); batch['_run_foot_now']=batch['_run_foot_now'].to(DEVICE)
    with torch.no_grad(): proposal=model.propose(batch['leg_hist'],batch['body_hist'],batch['action_plan'],batch['foot_now'])
    provider=StrictPatchProvider(store.memory(run_id)); hazard=torch.sigmoid(proposal['event_hazard_logit'][0]).cpu().numpy()
    swing=np.where(~(run.data['contact'][i]>.5))[0].tolist()
    if not swing: swing=[int(np.argmax(hazard.max(0)))]
    tables=[]
    for l in swing:
        h=int(np.argmax(hazard[:,l])); tables.append(candidate_table_for_leg(model,provider,batch,proposal,l,h))
    scored=pd.concat(tables,ignore_index=True); selected=[]
    # Joint combination search over each leg's best five finite candidates.
    import itertools
    tops=[g[np.isfinite(g.score)].nsmallest(5,'score') for _,g in scored.groupby('leg',sort=False)]
    if any(len(x)==0 for x in tops): raise RuntimeError('no feasible candidate for one or more swing legs')
    best=None
    for combo in itertools.product(*[list(g.index) for g in tops]):
        rows=scored.loc[list(combo)]; feet=run.data['foot_world'][i].copy(); contact=run.data['contact'][i]>.5
        for _,r in rows.iterrows(): li=LEG_NAMES.index(r.leg); feet[li]=[r.x,r.y,r.z]; contact[li]=True
        margin=signed_support_margin(feet[contact,:2],run.data['com_H'][i,:2]); joint=float(rows.score.sum()+10*max(-margin,0.)/.05)
        if best is None or joint<best[0]: best=(joint,margin,list(combo))
    selected=scored.loc[best[2]].copy(); selected['joint_score']=best[0]; selected['joint_support_margin_m']=best[1]
    out=RESULT_ROOT/'planner'; out.mkdir(parents=True,exist_ok=True)
    scored.to_csv(out/f'{run_id}_origin_{i}_all_candidates.csv',index=False)
    selected.to_csv(out/f'{run_id}_origin_{i}_selected.csv',index=False)
    for leg,g in scored.groupby('leg'):
        fig,ax=plt.subplots(figsize=(6,5)); ok=np.isfinite(g.score)
        sc=ax.scatter(g.x,g.y,c=np.where(ok,g.score,np.nan),cmap='RdYlGn_r',s=55); ax.scatter(g.loc[~ok,'x'],g.loc[~ok,'y'],marker='x',c='black')
        pick=selected[selected.leg==leg]
        if len(pick): ax.scatter(pick.x,pick.y,marker='*',s=260,c='#17365D',label='joint selection')
        ax.set_aspect('equal'); ax.set(xlabel='H forward [m]',ylabel='H left [m]',title=f'{run_id} {leg}: shadow contact-patch candidates')
        ax.legend(); fig.colorbar(sc,ax=ax,label='fixed normalized risk score'); savefig(fig,f'{run_id}_{leg}_foothold_candidates.png')
    display(selected); return selected,scored

# %% notebook cell 61
# STRICT S19 — evaluate independently retrained geometry/haptic ablations
def evaluate_required_geometry_ablations():
    assert_training_authorized(); store=PreparedStore(SPLIT_FRAME,max_memories=1)
    val_ids=SPLIT_FRAME.query("split == 'val'").run_id.tolist(); test_ids=SPLIT_FRAME.query("split == 'test'").run_id.tolist()
    seeds=TRAIN_SEEDS if ANALYSIS_TIER=='paper' else TRAIN_SEEDS[:1]; rows=[]
    for name,geometry_mode,haptic_enabled in GEOMETRY_ABLATIONS:
        for seed in seeds:
            model,norm,_=load_trained_seed(seed,name)
            vals=[collect_predictions(model,norm,store,rid,'val','zero_shot',geometry_mode) for rid in val_ids]
            cal=fit_force_scale_calibrator(vals)
            for rid in test_ids:
                pack=collect_predictions(model,norm,store,rid,'test','zero_shot',geometry_mode)
                fm,_=force_metrics(pack,f'{name}_seed_{seed}',cal); fm['experiment']=name
                fm['seed']=seed; fm['geometry_mode']=geometry_mode; fm['haptic_enabled']=haptic_enabled; rows.append(fm)
            del model; gc.collect()
    out=pd.concat(rows,ignore_index=True); out.to_csv(RESULT_ROOT/'required_geometry_ablation_metrics.csv',index=False)
    expected=len(GEOMETRY_ABLATIONS)*len(seeds); observed=out[['experiment','seed']].drop_duplicates().shape[0]
    status={'complete':observed==expected,'expected_experiment_seeds':expected,'observed_experiment_seeds':observed,
            'same_split_sha256':file_sha256(SPLIT_CONTRACT_PATH),'paper_data_gate':bool(DATA_GATE['pass'])}
    atomic_write_json(RESULT_ROOT/'required_geometry_ablation_status.json',status)
    gate_path=RESULT_ROOT/'claim_gates.json'
    if gate_path.exists():
        gates=json.load(open(gate_path)); gates['required_geometry_ablation_complete']=bool(status['complete'])
        gates['geometry_interface_claim_allowed']=bool(gates.get('paper_data_gate')) and bool(status['complete'])
        atomic_write_json(gate_path,gates)
    print(json.dumps(status,indent=2)); display(out); return out,status

# %% notebook cell 63
# STRICT S20 — held-out RGB-D-to-prior-map consistency before raw deletion
def validate_geometry_against_heldout_frames(spec):
    paths=prepared_paths(spec['run_id']); run_root=raw_run_path(spec)
    if paths['geometry_validation'].exists():
        old=json.load(open(paths['geometry_validation']))
        if old.get('geometry_complete_sha256')==file_sha256(paths['geometry']/'COMPLETE.json'): return old
    _,cc,_,errors=load_and_validate_contracts(require_approved=False)
    if errors['calibration']: raise RuntimeError('calibration contract must pass before geometry validation: '+'; '.join(errors['calibration']))
    data,grid=physics_grid_from_cache(locate_physics_cache(spec)); mem=CausalFusedVoxelMemory(paths['geometry'],float(grid[0]))
    frame0=pose_context(run_root,cc,np.array([grid[0]],float)); R_H_W=frame0['R_H_W']; p_W0=frame0['p_W0']
    groups=[g for g in geometry_frame_groups(run_root,cc) if grid[0]<=g['time_original']<=grid[-1]]
    rows=[]; rng=np.random.default_rng(SEED)
    for g in groups[::GEOMETRY_VALIDATION_FRAME_STRIDE]:
        cam='lcam_front' if 'lcam_front' in g['cameras'] else g['cameras'][0]
        pose=np.loadtxt(run_root/f'pose_{cam}.txt',dtype=np.float64)[g['frame_id']]
        from scipy.spatial.transform import Rotation
        R_WC=Rotation.from_quat(pose[3:7]).as_matrix(); p_WC=pose[:3]
        depth=read_metric_depth(g['files'][cam]['depth'][g['frame_id']]); intr=cc['camera_intrinsics'][cam]
        pc,good,_=deproject_camera(depth,intr,pixel_stride=max(GEOMETRY_PIXEL_STRIDE,8))
        normals=point_normals_from_depth(depth,intr,pixel_stride=max(GEOMETRY_PIXEL_STRIDE,8)).reshape(-1,3)[good.reshape(-1)]
        xyz_H=(p_WC[None]+pc@R_WC.T-p_W0[None])@R_H_W.T; normal_H=normals@R_WC.T@R_H_W.T
        valid=normal_H[:,2]>=GEOMETRY_MIN_UP_NORMAL; xyz_H=xyz_H[valid]
        if len(xyz_H)>GEOMETRY_VALIDATION_POINTS_PER_FRAME:
            take=rng.choice(len(xyz_H),GEOMETRY_VALIDATION_POINTS_PER_FRAME,replace=False); xyz_H=xyz_H[take]
        t_crop=float(g['time_original']-grid[0]-1e-6)
        for q in xyz_H:
            ids=mem._latest_candidates(q,t_crop,2.5*CFG.patch_voxel_m)
            if not len(ids): rows.append((g['frame_id'],g['time_original'],False,np.nan,np.nan)); continue
            j=ids[np.argmin(np.linalg.norm(mem.xyz[ids,:2]-q[None,:2],axis=1))]
            err=abs(float(mem.xyz[j,2]-q[2])); sigma=math.sqrt(max(float(mem.xyz_var[j,2]),1e-8))
            rows.append((g['frame_id'],g['time_original'],True,err,sigma))
    frame=pd.DataFrame(rows,columns=['frame_id','time_original','mapped','height_abs_error_m','predicted_sigma_z_m'])
    if not len(frame): raise RuntimeError('no held-out geometry validation samples')
    mapped=frame[frame.mapped]; coverage=float(frame.mapped.mean())
    result={'schema':'V10R_HELDOUT_GEOMETRY_V1','run_id':spec['run_id'],'samples':len(frame),
            'mapped_samples':len(mapped),'coverage':coverage,
            'height_mae_m':float(mapped.height_abs_error_m.mean()) if len(mapped) else None,
            'height_p95_m':float(mapped.height_abs_error_m.quantile(.95)) if len(mapped) else None,
            'nominal_95_uncertainty_coverage':float((mapped.height_abs_error_m<=1.96*mapped.predicted_sigma_z_m).mean()) if len(mapped) else None,
            'geometry_complete_sha256':file_sha256(paths['geometry']/'COMPLETE.json')}
    result['pass']=bool(coverage>=GEOMETRY_MIN_HELDOUT_COVERAGE and len(mapped)>0 and result['height_p95_m']<=GEOMETRY_MAX_HELDOUT_P95_M)
    frame.to_csv(paths['root']/'geometry_validation_samples.csv',index=False); atomic_write_json(paths['geometry_validation'],result)
    print(json.dumps(result,indent=2)); return result

# %% notebook cell 65
# STRICT S21 — single explicit dispatcher
def require_active_specs():
    if not ACTIVE_SPECS: raise RuntimeError('no active raw runs; download one or two runs or set ACTIVE_RUN_IDS correctly')


def verify_selected_or_all():
    ids=[s['run_id'] for s in ACTIVE_SPECS] or (PREPARED_TABLE.run_id.tolist() if len(PREPARED_TABLE) else [])
    rows=[verify_prepared_run(rid,raise_on_error=False) for rid in ids]
    frame=pd.DataFrame(rows); display(frame)
    for r in rows:
        if r['pass']: print(r['run_id'],'deletion token:',deletion_token(r['run_id']))
    return frame


STAGE_RESULT=None
if PIPELINE_STAGE=='inventory':
    print('Inventory only. No data changed.')
elif PIPELINE_STAGE=='audit':
    require_active_specs(); STAGE_RESULT={s['run_id']:run_force_audit(s)[1] for s in ACTIVE_SPECS}
elif PIPELINE_STAGE=='geometry':
    require_active_specs(); _,cc,sc,errors=load_and_validate_contracts(require_approved=False)
    if errors['calibration'] or errors['signals']:
        raise RuntimeError('geometry requires verified calibration/signal contracts:\n'+'\n'.join(errors['calibration']+errors['signals']))
    STAGE_RESULT={}
    for s in ACTIVE_SPECS:
        build_geometry_checkpoint(s,cc); STAGE_RESULT[s['run_id']]=validate_geometry_against_heldout_frames(s)
elif PIPELINE_STAGE=='prepare':
    require_active_specs(); STAGE_RESULT={s['run_id']:prepare_strict_run(s) for s in ACTIVE_SPECS}
elif PIPELINE_STAGE=='verify':
    STAGE_RESULT=verify_selected_or_all()
elif PIPELINE_STAGE=='delete':
    matches=[]
    for s in RUN_REGISTRY:
        with contextlib.suppress(Exception):
            if DELETE_CONFIRMATION==deletion_token(s['run_id']): matches.append(s['run_id'])
    if len(matches)!=1:
        print('No deletion performed. Run verify, then paste exactly one printed token into DELETE_CONFIRMATION.')
    else: STAGE_RESULT=delete_one_verified_raw_run(matches[0],DELETE_CONFIRMATION)
elif PIPELINE_STAGE=='exact_state':
    STAGE_RESULT=run_exact_state_ceiling()
elif PIPELINE_STAGE=='train':
    STAGE_RESULT=train_all_seeds()
elif PIPELINE_STAGE=='ablations':
    STAGE_RESULT={'training':train_required_geometry_ablations()}
    STAGE_RESULT['evaluation']=evaluate_required_geometry_ablations()
elif PIPELINE_STAGE=='evaluate':
    STAGE_RESULT=evaluate_trained_models()
elif PIPELINE_STAGE=='plots':
    STAGE_RESULT=generate_all_result_figures()
elif PIPELINE_STAGE=='planner':
    test_ids=SPLIT_FRAME.query("split == 'test'").run_id.tolist()
    if not test_ids: raise RuntimeError('no test run in frozen split')
    seed=(TRAIN_SEEDS if ANALYSIS_TIER=='paper' else TRAIN_SEEDS[:1])[0]
    STAGE_RESULT=shadow_plan_one_origin(seed,test_ids[0],origin_position=0)
elif PIPELINE_STAGE=='all_safe':
    require_active_specs(); _,cc,sc,errors=load_and_validate_contracts(require_approved=False)
    for s in ACTIVE_SPECS: run_force_audit(s)
    if errors['calibration'] or errors['signals']:
        raise RuntimeError('audit completed; geometry stopped at calibration/signal gate:\n'+'\n'.join(errors['calibration']+errors['signals']))
    for s in ACTIVE_SPECS: build_geometry_checkpoint(s,cc); validate_geometry_against_heldout_frames(s)
    STAGE_RESULT={s['run_id']:prepare_strict_run(s) for s in ACTIVE_SPECS}; verify_selected_or_all()
elif PIPELINE_STAGE=='status':
    print('Status requested; see the final status object below.')


STRICT_STATUS={
    'pipeline_version':STRICT_PIPELINE_VERSION,'updated_utc':utc_now(),'stage':PIPELINE_STAGE,
    'analysis_tier':ANALYSIS_TIER,'first_30_cells_preserved':True,
    'first_30_cells_sha256':'a1526713238e52f93474d665088efb57251377fdb1197e4771a9eff05585a7cb',
    'verified_prepared_runs':scan_verified_prepared().query("`pass` == True").run_id.tolist() if len(scan_verified_prepared()) else [],
    'paper_data_gate':DATA_GATE,
    'exact_state_result_present':(RESULT_ROOT/'G1_exact_state'/'gate.json').exists(),
    'trained_seed_checkpoints':len(list(MODEL_ROOT.glob('seed_*/best_G4.pt'))),
    'evaluation_result_present':(RESULT_ROOT/'claim_gates.json').exists(),
    'claim_gates':json.load(open(RESULT_ROOT/'claim_gates.json')) if (RESULT_ROOT/'claim_gates.json').exists() else {},
    'new_v10r_results_claimed_by_notebook':False,
}
if STRICT_STATUS['evaluation_result_present']:
    STRICT_STATUS['new_v10r_results_claimed_by_notebook']=bool(STRICT_STATUS['claim_gates'].get('force_claim_allowed',False))
atomic_write_json(STRICT_ROOT/'strict_pipeline_status.json',STRICT_STATUS)
print(json.dumps(STRICT_STATUS,indent=2))

# %% notebook cell 67
# MANUAL T0 — refresh prepared runs/splits and inspect readiness
def configure_manual_analysis_mode(mode):
    global ANALYSIS_TIER, ALLOW_DIAGNOSTIC_TRAINING, MODEL_ROOT, RESULT_ROOT, FIGURE_ROOT
    if mode not in {'strict_paper','diagnostic_four_run'}: raise ValueError(mode)
    ANALYSIS_TIER='paper' if mode=='strict_paper' else 'diagnostic'
    ALLOW_DIAGNOSTIC_TRAINING=(mode=='diagnostic_four_run')
    MODEL_ROOT=STRICT_ROOT/'models'/ANALYSIS_TIER
    RESULT_ROOT=STRICT_ROOT/'results'/ANALYSIS_TIER
    FIGURE_ROOT=STRICT_ROOT/'figures'/ANALYSIS_TIER
    for p in (MODEL_ROOT,RESULT_ROOT,FIGURE_ROOT): p.mkdir(parents=True,exist_ok=True)
    return {'mode':mode,'analysis_tier':ANALYSIS_TIER,'model_root':str(MODEL_ROOT),
            'result_root':str(RESULT_ROOT),'figure_root':str(FIGURE_ROOT)}


def refresh_manual_training_state():
    global PREPARED_TABLE, SPLIT_FRAME, DATA_GATE
    PREPARED_TABLE=scan_verified_prepared(); write_split_contract_once(PREPARED_TABLE)
    SPLIT_FRAME=pd.read_csv(SPLIT_CONTRACT_PATH) if SPLIT_CONTRACT_PATH.exists() else pd.DataFrame()
    DATA_GATE=paper_data_gate_strict(SPLIT_FRAME)
    atomic_write_json(STRICT_ROOT/'paper_data_gate.json',DATA_GATE)
    return PREPARED_TABLE,SPLIT_FRAME,DATA_GATE


def assert_diagnostic_split_ready():
    if len(SPLIT_FRAME)<4: raise RuntimeError('diagnostic training requires four independently prepared trajectories')
    if set(SPLIT_FRAME.split)!={'train','val','test'}: raise RuntimeError('diagnostic split must contain train, val, and test')
    frozen=SPLIT_FRAME.frozen_before_training.astype(str).str.lower().isin({'true','1','yes'})
    if not frozen.all(): raise RuntimeError('freeze every split row before diagnostic training')
    if SPLIT_FRAME.run_id.duplicated().any() or SPLIT_FRAME.trajectory_id.duplicated().any():
        raise RuntimeError('run_id and trajectory_id must be unique')
    bad=[rid for rid in SPLIT_FRAME.run_id if not verify_prepared_run(rid,raise_on_error=False)['pass']]
    if bad: raise RuntimeError('unverified prepared runs: '+','.join(bad))
    return True


PREPARED_TABLE,SPLIT_FRAME,DATA_GATE=refresh_manual_training_state()
display(PREPARED_TABLE); display(SPLIT_FRAME); print(json.dumps(DATA_GATE,indent=2))


# %% notebook cell 68
# MANUAL T1 — TRAINING CELL (change only the two controls below)
TRAINING_MODE = 'strict_paper'          # 'strict_paper' or 'diagnostic_four_run'
START_TRAINING_NOW = False              # change to True, then run this cell only

TRAINING_OUTPUTS=None
if START_TRAINING_NOW:
    print(json.dumps(configure_manual_analysis_mode(TRAINING_MODE),indent=2))
    PREPARED_TABLE,SPLIT_FRAME,DATA_GATE=refresh_manual_training_state()
    if TRAINING_MODE=='diagnostic_four_run':
        assert_diagnostic_split_ready()
        print('DIAGNOSTIC ONLY: one seed; outputs cannot support a paper claim.')
    else:
        assert_training_authorized()
        print('STRICT PAPER TRAINING: five seeds; frozen paper data gate passed.')
    TRAINING_OUTPUTS=train_all_seeds()
    print('Training complete. Checkpoints:',MODEL_ROOT)
else:
    print("Training is armed but OFF. Set START_TRAINING_NOW=True and run this cell only.")


# %% notebook cell 69
# MANUAL T2 — TESTING/EVALUATION CELL (run only after T1 completes)
TESTING_MODE = 'strict_paper'           # must match the training mode
START_TESTING_NOW = False               # change to True, then run this cell only

TEST_RESULTS=None
if START_TESTING_NOW:
    print(json.dumps(configure_manual_analysis_mode(TESTING_MODE),indent=2))
    PREPARED_TABLE,SPLIT_FRAME,DATA_GATE=refresh_manual_training_state()
    if TESTING_MODE=='diagnostic_four_run': assert_diagnostic_split_ready()
    TEST_RESULTS=evaluate_trained_models()
    print('Testing complete.')
    print('Metrics:',RESULT_ROOT)
    print('Claim gates:',RESULT_ROOT/'claim_gates.json')
else:
    print("Testing is OFF. Set START_TESTING_NOW=True only after training finishes.")


# %% notebook cell 70
# MANUAL T3 — REQUIRED ABLATIONS (independently retrains every declared map variant)
ABLATION_MODE = 'strict_paper'          # must match the training mode
START_ABLATIONS_NOW = False

ABLATION_TRAINING_OUTPUTS=None; ABLATION_TEST_RESULTS=None
if START_ABLATIONS_NOW:
    print(json.dumps(configure_manual_analysis_mode(ABLATION_MODE),indent=2))
    PREPARED_TABLE,SPLIT_FRAME,DATA_GATE=refresh_manual_training_state()
    if ABLATION_MODE=='diagnostic_four_run': assert_diagnostic_split_ready()
    ABLATION_TRAINING_OUTPUTS=train_required_geometry_ablations()
    ABLATION_TEST_RESULTS=evaluate_required_geometry_ablations()
    print('Ablations complete:',RESULT_ROOT/'required_geometry_ablation_metrics.csv')
else:
    print("Ablations are OFF. Set START_ABLATIONS_NOW=True after primary testing.")


# %% notebook cell 71
# MANUAL T4 — GENERATE REAL FIGURES AND OPTIONAL RESULTS ZIP
PLOT_EXPORT_MODE = 'strict_paper'       # must match training/testing mode
GENERATE_PLOTS_NOW = False
BUILD_RESULTS_ZIP_NOW = False
INCLUDE_MODEL_CHECKPOINTS_IN_ZIP = False
DOWNLOAD_ZIP_NOW = False

if GENERATE_PLOTS_NOW or BUILD_RESULTS_ZIP_NOW:
    print(json.dumps(configure_manual_analysis_mode(PLOT_EXPORT_MODE),indent=2))
if GENERATE_PLOTS_NOW:
    GENERATED_FIGURES=generate_all_result_figures()
if BUILD_RESULTS_ZIP_NOW:
    import zipfile
    bundle=STRICT_ROOT/f'V10R_{ANALYSIS_TIER.upper()}_RESULTS.zip'
    roots=[RESULT_ROOT,FIGURE_ROOT]+([MODEL_ROOT] if INCLUDE_MODEL_CHECKPOINTS_IN_ZIP else [])
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for root in roots:
            if root.exists():
                for p in sorted(root.rglob('*')):
                    if p.is_file(): z.write(p,p.relative_to(STRICT_ROOT))
    print('Results ZIP:',bundle,'| MiB:',round(bundle.stat().st_size/1024**2,2))
    if DOWNLOAD_ZIP_NOW:
        try:
            from google.colab import files
            files.download(str(bundle))
        except ImportError:
            print('Not running inside Colab; download manually:',bundle)
else:
    print('Plot/export cell is OFF. Enable only the operation you want.')


# %% notebook cell 73
# FINAL F0 — execution controls
# The defaults execute the complete four-run diagnostic experiment with one real seed.
FINAL_EXECUTE = True
FINAL_MODE = 'diagnostic_four_run'
FINAL_AUTO_DELETE_VERIFIED_RAW = True
FINAL_RUN_EXACT_STATE_CEILING = True
FINAL_RUN_ABLATIONS = False          # True adds four independently retrained geometry ablations
FINAL_FORCE_RETRAIN = False
FINAL_FORCE_REEVALUATE = False
FINAL_INCLUDE_MODELS_IN_ZIP = True
FINAL_DOWNLOAD_RESULTS_ZIP = True

if FINAL_MODE != 'diagnostic_four_run':
    raise ValueError('This four-trajectory runner is diagnostic_four_run only. Paper mode requires the paper data gate.')

print('Final runner armed:', FINAL_EXECUTE)
print('Mode:', FINAL_MODE)
print('Raw deletion after cryptographic verification:', FINAL_AUTO_DELETE_VERIFIED_RAW)
print('Ablations:', FINAL_RUN_ABLATIONS)

# %% notebook cell 74
# FINAL F1 — official downloader and strict one-run validation
def final_install_tartanair():
    if importlib.util.find_spec('tartanair') is None:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-q',
            'git+https://github.com/castacks/tartanairpy.git'
        ])
        importlib.invalidate_caches()
    import tartanair as ta
    return ta


def final_raw_requirements(spec):
    root = raw_run_path(spec)
    files = [root/'rosbags'/f"{spec['run']}.bag"]
    files += [root/f'pose_{cam}.txt' for cam in CAMERA_NAMES]
    directories = [root/'imu']
    directories += [root/f'image_{cam}' for cam in CAMERA_NAMES]
    directories += [root/f'depth_{cam}' for cam in CAMERA_NAMES]
    return root, files, directories


def final_validate_raw(spec):
    root, files, directories = final_raw_requirements(spec)
    missing = [str(p) for p in files if not p.is_file() or p.stat().st_size == 0]
    missing += [str(p) for p in directories if not p.is_dir() or not any(p.iterdir())]
    return {'run_id': spec['run_id'], 'root': str(root), 'pass': not missing, 'missing': missing}


def final_download_one(spec):
    state = final_validate_raw(spec)
    if state['pass']:
        print('Reusing complete raw run:', spec['run_id'])
        return Path(state['root'])
    require_local_headroom(8.0)
    ta = final_install_tartanair()
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    ta.init(str(RAW_ROOT))
    print('Downloading:', spec['environment'], spec['run'])
    ta.download_ground(
        env=[spec['environment']], version=['anymal'], traj=[spec['run']],
        modality=['imu', 'rosbag', 'image', 'depth', 'meta'],
        camera_name=list(CAMERA_NAMES), unzip=True,
    )
    state = final_validate_raw(spec)
    if not state['pass']:
        raise RuntimeError('Incomplete TartanGround download:' + chr(10) + chr(10).join(state['missing']))
    print('Download verified:', state['root'])
    return Path(state['root'])


print('Downloader and raw-data verifier ready.')

# %% notebook cell 75


# %% notebook cell 76
# PATCH P1 — correct execution order and inspect real metadata

from pathlib import Path
import contextlib
import gc
import importlib
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

# Prevent the old broken F2–F8 runner from being used again.
FINAL_EXECUTE = False

PROJECT_ROOT = Path("/content/drive/MyDrive/UGP_V7_FULL4B")
RAW_ROOT = Path("/content/tartanground_v10")

# Large outputs go to Colab local storage because Drive has only ~1.48 GiB free.
STRICT_ROOT = Path("/content/V10R_STRICT_STAGED_RUNTIME")
PREPARED_ROOT = STRICT_ROOT / "prepared_runs"
AUDIT_ROOT = STRICT_ROOT / "audits"
MODEL_ROOT = STRICT_ROOT / "models" / "diagnostic"
RESULT_ROOT = STRICT_ROOT / "results" / "diagnostic"
FIGURE_ROOT = STRICT_ROOT / "figures" / "diagnostic"

# Keep only the small scientific contracts persistently on Drive.
CONTRACT_ROOT = (
    PROJECT_ROOT / "V10R_STRICT_STAGED" / "contracts"
)

FORCE_CONTRACT_PATH = CONTRACT_ROOT / "force_frame_contract.json"
CALIBRATION_CONTRACT_PATH = CONTRACT_ROOT / "calibration_contract.json"
SIGNAL_CONTRACT_PATH = CONTRACT_ROOT / "signal_semantics_contract.json"
SPLIT_CONTRACT_PATH = CONTRACT_ROOT / "trajectory_split.csv"

ANALYSIS_TIER = "diagnostic"
ALLOW_DIAGNOSTIC_TRAINING = True

for path in (
    STRICT_ROOT,
    PREPARED_ROOT,
    AUDIT_ROOT,
    MODEL_ROOT,
    RESULT_ROOT,
    FIGURE_ROOT,
    CONTRACT_ROOT,
):
    path.mkdir(parents=True, exist_ok=True)

local_free = shutil.disk_usage("/content").free / 1024**3
drive_free = shutil.disk_usage(PROJECT_ROOT).free / 1024**3

print(f"Local free storage: {local_free:.2f} GiB")
print(f"Drive free storage: {drive_free:.2f} GiB")
print("Large runtime outputs:", STRICT_ROOT)
print("Persistent contracts:", CONTRACT_ROOT)

if local_free < 25:
    raise RuntimeError(
        f"Only {local_free:.2f} GiB local space remains. "
        "At least 25 GiB is required before starting reconstruction."
    )

# Download ForestEnv/P2000 NOW, before checking contracts.
PATCH_SPEC = next(
    spec for spec in RUN_REGISTRY
    if spec["run_id"] == "forest_P2000"
)

PATCH_RAW_RUN = final_download_one(PATCH_SPEC)
PATCH_RAW_STATE = final_validate_raw(PATCH_SPEC)

if not PATCH_RAW_STATE["pass"]:
    raise RuntimeError(PATCH_RAW_STATE)

print("\nRAW DOWNLOAD VERIFIED")
print(PATCH_RAW_RUN)

# Refresh the inventory after the actual download.
INVENTORY = inventory_table()
display(INVENTORY)

# Native 400 Hz force/timestamp/frame audit.
PATCH_NATIVE_FORCE, PATCH_FORCE_AUDIT = run_force_audit(PATCH_SPEC)
display(PATCH_FORCE_AUDIT)

# Inspect frames of relevant ROS messages.
ensure_python_packages([("rosbags", "rosbags")])
from rosbags.highlevel import AnyReader

bag_path = (
    PATCH_RAW_RUN / "rosbags" / f"{PATCH_SPEC['run']}.bag"
)

wanted_topics = set(FORCE_TOPICS.values()) | {
    "/state_estimator/joint_states",
    "/state_estimator/twist",
    "/state_estimator/actuator_readings",
    "/path_planning_and_following/twist",
}

frame_rows = []
seen = set()

with AnyReader([bag_path]) as reader:
    connections = [
        connection
        for connection in reader.connections
        if connection.topic in wanted_topics
    ]

    for connection, timestamp, raw in reader.messages(
        connections=connections
    ):
        if connection.topic in seen:
            continue

        message = reader.deserialize(raw, connection.msgtype)
        header = getattr(message, "header", None)

        frame_rows.append({
            "topic": connection.topic,
            "message_type": connection.msgtype,
            "header_frame_id": str(
                getattr(header, "frame_id", "") or ""
            ),
            "child_frame_id": str(
                getattr(message, "child_frame_id", "") or ""
            ),
        })

        seen.add(connection.topic)

        if len(seen) == len({c.topic for c in connections}):
            break

PATCH_ROS_FRAME_TABLE = pd.DataFrame(frame_rows)
display(PATCH_ROS_FRAME_TABLE)

PATCH_ROS_FRAME_TABLE.to_csv(
    AUDIT_ROOT / "ForestEnv_P2000_ros_frame_inventory.csv",
    index=False,
)

# Install the official ANYmal-D URDF and extract only names actually present.
description_root = (
    Path("/content/v10r_external")
    / "anymal_d_simple_description"
)

if not description_root.exists():
    description_root.parent.mkdir(parents=True, exist_ok=True)

    subprocess.check_call([
        "git",
        "clone",
        "--depth",
        "1",
        "https://github.com/ANYbotics/"
        "anymal_d_simple_description.git",
        str(description_root),
    ])

PATCH_URDF = description_root / "urdf" / "anymal.urdf"

if not PATCH_URDF.is_file():
    raise FileNotFoundError(PATCH_URDF)

urdf_xml = ET.parse(PATCH_URDF).getroot()

urdf_joint_names = {
    element.attrib["name"]
    for element in urdf_xml.findall("joint")
}

urdf_link_names = {
    element.attrib["name"]
    for element in urdf_xml.findall("link")
}

missing_joints = [
    name
    for name in CANONICAL_JOINT_NAMES
    if name not in urdf_joint_names
]

prefix_by_leg = {
    "FL": "LF",
    "FR": "RF",
    "RL": "LH",
    "RR": "RH",
}

foot_frames = {}

for leg, prefix in prefix_by_leg.items():
    exact = f"{prefix}_FOOT"

    candidates = sorted(
        name
        for name in urdf_link_names
        if name.upper().startswith(prefix)
        and any(token in name.upper() for token in ("FOOT", "TOE"))
    )

    if exact in urdf_link_names:
        foot_frames[leg] = exact
    elif len(candidates) == 1:
        foot_frames[leg] = candidates[0]
    else:
        foot_frames[leg] = ""

print("URDF:", PATCH_URDF)
print("Missing canonical joints:", missing_joints)
print("Detected foot frames:", foot_frames)

# Safely fill only URDF fields that are directly verified from the file.
if CALIBRATION_CONTRACT_PATH.exists():
    calibration_contract = load_json_contract(
        CALIBRATION_CONTRACT_PATH
    )

    if calibration_contract.get("verified") is not True:
        calibration_contract["urdf_path"] = str(PATCH_URDF)
        calibration_contract["urdf_sha256"] = file_sha256(
            PATCH_URDF
        )

        if not missing_joints:
            calibration_contract["joint_names"] = {
                name: name
                for name in CANONICAL_JOINT_NAMES
            }

        if all(foot_frames.values()):
            calibration_contract["foot_frame_names"] = foot_frames

        atomic_write_json(
            CALIBRATION_CONTRACT_PATH,
            calibration_contract,
        )

# Report remaining genuine blockers without launching downstream cells.
_, _, _, PATCH_CONTRACT_ERRORS = load_and_validate_contracts(
    require_approved=False
)

PATCH_BLOCKER_TABLE = pd.DataFrame([
    {
        "contract": name,
        "pass": not problems,
        "problems": " | ".join(problems),
    }
    for name, problems in PATCH_CONTRACT_ERRORS.items()
])

display(PATCH_BLOCKER_TABLE)

print("\nContract files requiring verified evidence:")
print(FORCE_CONTRACT_PATH)
print(CALIBRATION_CONTRACT_PATH)
print(SIGNAL_CONTRACT_PATH)

# %% notebook cell 77
# PATCH P2 — guarded end-to-end continuation

PATCH_FORCE_RETRAIN = False
PATCH_FORCE_REEVALUATE = False
PATCH_RUN_ABLATIONS = False
PATCH_DELETE_VERIFIED_RAW = True
PATCH_DOWNLOAD_FINAL_ZIP = True


def patched_contract_status():
    _, _, _, errors = load_and_validate_contracts(
        require_approved=False
    )

    table = pd.DataFrame([
        {
            "contract": name,
            "pass": not problems,
            "problems": " | ".join(problems),
        }
        for name, problems in errors.items()
    ])

    return table, errors


def patched_end_to_end():
    global PREPARED_TABLE, SPLIT_FRAME, DATA_GATE

    contract_table, errors = patched_contract_status()
    display(contract_table)

    if any(errors.values()):
        print(
            "\nSTATUS: BLOCKED BEFORE PREPARATION\n"
            "The remaining items are missing scientific metadata, "
            "not Python errors. No training or export was started."
        )

        return {
            "status": "BLOCKED",
            "stage": "G0_contracts",
            "contract_paths": [
                str(FORCE_CONTRACT_PATH),
                str(CALIBRATION_CONTRACT_PATH),
                str(SIGNAL_CONTRACT_PATH),
            ],
        }

    # Downloads each remaining trajectory one at a time, reconstructs it,
    # verifies it, then deletes only its verified raw directory.
    preparation = final_process_all_runs(
        auto_delete=PATCH_DELETE_VERIFIED_RAW
    )

    if (
        len(preparation) != 4
        or not preparation["pass"].all()
    ):
        raise RuntimeError(
            "All four prepared runs did not pass verification."
        )

    # Freeze the explicit diagnostic trajectory split.
    final_freeze_diagnostic_split()

    configure_manual_analysis_mode(
        "diagnostic_four_run"
    )

    PREPARED_TABLE, SPLIT_FRAME, DATA_GATE = (
        refresh_manual_training_state()
    )

    assert_diagnostic_split_ready()

    print("\n===== EXACT-STATE CEILING =====")

    ceiling_gate = (
        RESULT_ROOT / "G1_exact_state" / "gate.json"
    )

    if PATCH_FORCE_RETRAIN or not ceiling_gate.exists():
        exact_state_results = run_exact_state_ceiling()
    else:
        exact_state_results = None
        print("Reusing:", ceiling_gate)

    print("\n===== TRAINING =====")

    seed = TRAIN_SEEDS[0]
    checkpoint = (
        MODEL_ROOT / f"seed_{seed}" / "best_G4.pt"
    )

    if PATCH_FORCE_RETRAIN or not checkpoint.exists():
        training_outputs = train_all_seeds()
    else:
        training_outputs = None
        print("Reusing:", checkpoint)

    if not checkpoint.is_file():
        raise RuntimeError(
            f"Training did not create {checkpoint}"
        )

    print("\n===== TESTING =====")

    claim_file = RESULT_ROOT / "claim_gates.json"

    if PATCH_FORCE_REEVALUATE or not claim_file.exists():
        test_results = evaluate_trained_models()
    else:
        test_results = None
        print("Reusing:", claim_file)

    if PATCH_RUN_ABLATIONS:
        train_required_geometry_ablations()
        evaluate_required_geometry_ablations()

    required_results = [
        RESULT_ROOT / "force_metrics_long.csv",
        RESULT_ROOT / "force_vector_metrics_long.csv",
        RESULT_ROOT / "all_sensor_metrics_long.csv",
        RESULT_ROOT / "event_impulse_waveform_metrics.csv",
        RESULT_ROOT / "prediction_artifact_index.csv",
        RESULT_ROOT / "claim_gates.json",
    ]

    missing_results = [
        str(path)
        for path in required_results
        if not path.is_file()
    ]

    if missing_results:
        raise RuntimeError(
            "Evaluation finished without mandatory artifacts:\n"
            + "\n".join(missing_results)
        )

    print("\n===== FIGURES =====")
    figure_paths = generate_all_result_figures()

    print("\n===== FULL METRIC TABLES =====")

    for path in required_results:
        if path.suffix == ".csv":
            table = pd.read_csv(path)

            print(f"\n{path.name}: {len(table)} rows")

            with pd.option_context(
                "display.max_rows", None,
                "display.max_columns", None,
                "display.width", 220,
            ):
                display(table)

    claim_gates = json.load(open(claim_file))

    print("\n===== CLAIM GATES =====")
    print(json.dumps(claim_gates, indent=2))

    try:
        from IPython.display import (
            Image as IPyImage,
            display as ipy_display,
        )

        for path in figure_paths:
            print(Path(path).name)
            ipy_display(IPyImage(filename=str(path)))

    except Exception as error:
        print("Inline image display failed:", error)

    print("\n===== EXPORT =====")

    import zipfile

    bundle = Path(
        "/content/"
        "V10R_FINAL_DIAGNOSTIC_RESULTS_AND_MODEL.zip"
    )

    export_sources = [
        (RESULT_ROOT, "results"),
        (FIGURE_ROOT, "figures"),
        (MODEL_ROOT, "models"),
        (CONTRACT_ROOT, "contracts"),
        (AUDIT_ROOT, "audits"),
    ]

    csv_rows = []

    for root, label in export_sources:
        if not root.exists():
            continue

        for path in root.rglob("*.csv"):
            csv_rows.append({
                "group": label,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            })

    csv_manifest = pd.DataFrame(csv_rows)
    csv_manifest_path = (
        STRICT_ROOT / "csv_manifest.csv"
    )

    csv_manifest.to_csv(
        csv_manifest_path,
        index=False,
    )

    with zipfile.ZipFile(
        bundle,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:

        for root, label in export_sources:
            if not root.exists():
                continue

            for path in sorted(root.rglob("*")):
                if path.is_file():
                    archive.write(
                        path,
                        Path(label) / path.relative_to(root),
                    )

        archive.write(
            csv_manifest_path,
            "csv_manifest.csv",
        )

    if bundle.stat().st_size == 0:
        raise RuntimeError(
            "Refusing to export an empty results ZIP."
        )

    print("Final model:", checkpoint)
    print("Final results ZIP:", bundle)
    print(
        "ZIP size MiB:",
        round(bundle.stat().st_size / 1024**2, 2),
    )
    print("CSV files:", len(csv_manifest))

    if PATCH_DOWNLOAD_FINAL_ZIP:
        from google.colab import files
        files.download(str(bundle))

    return {
        "status": "COMPLETE",
        "checkpoint": str(checkpoint),
        "results_zip": str(bundle),
        "claim_gates": claim_gates,
        "csv_count": len(csv_manifest),
    }


PATCH_RESULT = patched_end_to_end()
display(PATCH_RESULT)

# %% notebook cell 78
# PATCH P2 — guarded end-to-end continuation

PATCH_FORCE_RETRAIN = False
PATCH_FORCE_REEVALUATE = False
PATCH_RUN_ABLATIONS = False
PATCH_DELETE_VERIFIED_RAW = True
PATCH_DOWNLOAD_FINAL_ZIP = True


def patched_contract_status():
    _, _, _, errors = load_and_validate_contracts(
        require_approved=False
    )

    table = pd.DataFrame([
        {
            "contract": name,
            "pass": not problems,
            "problems": " | ".join(problems),
        }
        for name, problems in errors.items()
    ])

    return table, errors


def patched_end_to_end():
    global PREPARED_TABLE, SPLIT_FRAME, DATA_GATE

    contract_table, errors = patched_contract_status()
    display(contract_table)

    if any(errors.values()):
        print(
            "\nSTATUS: BLOCKED BEFORE PREPARATION\n"
            "The remaining items are missing scientific metadata, "
            "not Python errors. No training or export was started."
        )

        return {
            "status": "BLOCKED",
            "stage": "G0_contracts",
            "contract_paths": [
                str(FORCE_CONTRACT_PATH),
                str(CALIBRATION_CONTRACT_PATH),
                str(SIGNAL_CONTRACT_PATH),
            ],
        }

    # Downloads each remaining trajectory one at a time, reconstructs it,
    # verifies it, then deletes only its verified raw directory.
    preparation = final_process_all_runs(
        auto_delete=PATCH_DELETE_VERIFIED_RAW
    )

    if (
        len(preparation) != 4
        or not preparation["pass"].all()
    ):
        raise RuntimeError(
            "All four prepared runs did not pass verification."
        )

    # Freeze the explicit diagnostic trajectory split.
    final_freeze_diagnostic_split()

    configure_manual_analysis_mode(
        "diagnostic_four_run"
    )

    PREPARED_TABLE, SPLIT_FRAME, DATA_GATE = (
        refresh_manual_training_state()
    )

    assert_diagnostic_split_ready()

    print("\n===== EXACT-STATE CEILING =====")

    ceiling_gate = (
        RESULT_ROOT / "G1_exact_state" / "gate.json"
    )

    if PATCH_FORCE_RETRAIN or not ceiling_gate.exists():
        exact_state_results = run_exact_state_ceiling()
    else:
        exact_state_results = None
        print("Reusing:", ceiling_gate)

    print("\n===== TRAINING =====")

    seed = TRAIN_SEEDS[0]
    checkpoint = (
        MODEL_ROOT / f"seed_{seed}" / "best_G4.pt"
    )

    if PATCH_FORCE_RETRAIN or not checkpoint.exists():
        training_outputs = train_all_seeds()
    else:
        training_outputs = None
        print("Reusing:", checkpoint)

    if not checkpoint.is_file():
        raise RuntimeError(
            f"Training did not create {checkpoint}"
        )

    print("\n===== TESTING =====")

    claim_file = RESULT_ROOT / "claim_gates.json"

    if PATCH_FORCE_REEVALUATE or not claim_file.exists():
        test_results = evaluate_trained_models()
    else:
        test_results = None
        print("Reusing:", claim_file)

    if PATCH_RUN_ABLATIONS:
        train_required_geometry_ablations()
        evaluate_required_geometry_ablations()

    required_results = [
        RESULT_ROOT / "force_metrics_long.csv",
        RESULT_ROOT / "force_vector_metrics_long.csv",
        RESULT_ROOT / "all_sensor_metrics_long.csv",
        RESULT_ROOT / "event_impulse_waveform_metrics.csv",
        RESULT_ROOT / "prediction_artifact_index.csv",
        RESULT_ROOT / "claim_gates.json",
    ]

    missing_results = [
        str(path)
        for path in required_results
        if not path.is_file()
    ]

    if missing_results:
        raise RuntimeError(
            "Evaluation finished without mandatory artifacts:\n"
            + "\n".join(missing_results)
        )

    print("\n===== FIGURES =====")
    figure_paths = generate_all_result_figures()

    print("\n===== FULL METRIC TABLES =====")

    for path in required_results:
        if path.suffix == ".csv":
            table = pd.read_csv(path)

            print(f"\n{path.name}: {len(table)} rows")

            with pd.option_context(
                "display.max_rows", None,
                "display.max_columns", None,
                "display.width", 220,
            ):
                display(table)

    claim_gates = json.load(open(claim_file))

    print("\n===== CLAIM GATES =====")
    print(json.dumps(claim_gates, indent=2))

    try:
        from IPython.display import (
            Image as IPyImage,
            display as ipy_display,
        )

        for path in figure_paths:
            print(Path(path).name)
            ipy_display(IPyImage(filename=str(path)))

    except Exception as error:
        print("Inline image display failed:", error)

    print("\n===== EXPORT =====")

    import zipfile

    bundle = Path(
        "/content/"
        "V10R_FINAL_DIAGNOSTIC_RESULTS_AND_MODEL.zip"
    )

    export_sources = [
        (RESULT_ROOT, "results"),
        (FIGURE_ROOT, "figures"),
        (MODEL_ROOT, "models"),
        (CONTRACT_ROOT, "contracts"),
        (AUDIT_ROOT, "audits"),
    ]

    csv_rows = []

    for root, label in export_sources:
        if not root.exists():
            continue

        for path in root.rglob("*.csv"):
            csv_rows.append({
                "group": label,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            })

    csv_manifest = pd.DataFrame(csv_rows)
    csv_manifest_path = (
        STRICT_ROOT / "csv_manifest.csv"
    )

    csv_manifest.to_csv(
        csv_manifest_path,
        index=False,
    )

    with zipfile.ZipFile(
        bundle,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:

        for root, label in export_sources:
            if not root.exists():
                continue

            for path in sorted(root.rglob("*")):
                if path.is_file():
                    archive.write(
                        path,
                        Path(label) / path.relative_to(root),
                    )

        archive.write(
            csv_manifest_path,
            "csv_manifest.csv",
        )

    if bundle.stat().st_size == 0:
        raise RuntimeError(
            "Refusing to export an empty results ZIP."
        )

    print("Final model:", checkpoint)
    print("Final results ZIP:", bundle)
    print(
        "ZIP size MiB:",
        round(bundle.stat().st_size / 1024**2, 2),
    )
    print("CSV files:", len(csv_manifest))

    if PATCH_DOWNLOAD_FINAL_ZIP:
        from google.colab import files
        files.download(str(bundle))

    return {
        "status": "COMPLETE",
        "checkpoint": str(checkpoint),
        "results_zip": str(bundle),
        "claim_gates": claim_gates,
        "csv_count": len(csv_manifest),
    }


PATCH_RESULT = patched_end_to_end()
display(PATCH_RESULT)

# %% notebook cell 79
# FINAL F3 — sequential, resumable download → audit → geometry → prepare → verify → delete
def final_process_all_runs(auto_delete=True):
    fc, cc, sc = final_contract_preflight()
    status_rows = []
    for index, spec in enumerate(RUN_REGISTRY, start=1):
        print(f"\n===== RUN {index}/{len(RUN_REGISTRY)}: {spec['run_id']} =====")
        prior = verify_prepared_run(spec['run_id'], raise_on_error=False)
        if prior['pass']:
            print('Prepared checkpoint already verified; skipping reconstruction.')
            status_rows.append(prior)
            continue

        final_download_one(spec)
        run_force_audit(spec)
        build_geometry_checkpoint(spec, cc)
        geometry_report = validate_geometry_against_heldout_frames(spec)
        if not geometry_report['pass']:
            raise RuntimeError(f"{spec['run_id']}: held-out geometry validation failed: {geometry_report}")
        prepare_strict_run(spec)
        verified = verify_prepared_run(spec['run_id'], raise_on_error=True)
        status_rows.append(verified)

        if auto_delete:
            token = deletion_token(spec['run_id'])
            delete_one_verified_raw_run(spec['run_id'], token)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    table = pd.DataFrame(status_rows)
    display(table)
    expected = {s['run_id'] for s in RUN_REGISTRY}
    passed = set(table.loc[table['pass'] == True, 'run_id']) if len(table) else set()
    if passed != expected:
        raise RuntimeError(f'Prepared-run set incomplete. Expected {sorted(expected)}, passed {sorted(passed)}')
    return table


if FINAL_EXECUTE:
    FINAL_PREPARATION_STATUS = final_process_all_runs(FINAL_AUTO_DELETE_VERIFIED_RAW)
else:
    print('Final execution is OFF; raw runs were not processed.')

# %% notebook cell 80
# FINAL F4 — freeze the explicit four-run diagnostic split before any fitting
FINAL_DIAGNOSTIC_SPLIT = {
    'forest_P2000': 'train',
    'swamp_marsh_P2023': 'train',
    'rough_uneven_P2000': 'val',
    'wet_uneven_proxy_P2000': 'test',
}


def final_freeze_diagnostic_split():
    verified = scan_verified_prepared()
    good = set(verified.loc[verified['pass'] == True, 'run_id']) if len(verified) else set()
    required = set(FINAL_DIAGNOSTIC_SPLIT)
    if good != required:
        raise RuntimeError(f'Need all four verified runs. Verified={sorted(good)}')

    existing_models = list((STRICT_ROOT/'models').rglob('best_G4.pt'))
    if SPLIT_CONTRACT_PATH.exists() and existing_models:
        old = pd.read_csv(SPLIT_CONTRACT_PATH)
        old_map = dict(zip(old.run_id, old.split))
        if old_map != FINAL_DIAGNOSTIC_SPLIT:
            raise RuntimeError('A different split already has trained checkpoints; refusing to rewrite it.')

    rows = []
    for spec in RUN_REGISTRY:
        rows.append({
            'run_id': spec['run_id'], 'trajectory_id': spec['trajectory_id'],
            'terrain': spec['terrain'], 'gait': spec['gait'], 'policy': spec['policy'],
            'speed_bin': spec['speed_bin'], 'split': FINAL_DIAGNOSTIC_SPLIT[spec['run_id']],
            'frozen_before_training': True,
            'note': 'FROZEN FOUR-RUN DIAGNOSTIC SPLIT; NOT PAPER-ELIGIBLE',
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(SPLIT_CONTRACT_PATH, index=False)
    display(frame)
    return frame


if FINAL_EXECUTE:
    FINAL_SPLIT = final_freeze_diagnostic_split()
    print(json.dumps(configure_manual_analysis_mode(FINAL_MODE), indent=2))
    PREPARED_TABLE, SPLIT_FRAME, DATA_GATE = refresh_manual_training_state()
    assert_diagnostic_split_ready()
    print('Diagnostic split verified and frozen before fitting.')
else:
    print('Final execution is OFF; split was not frozen.')

# %% notebook cell 81
# FINAL F5 — exact-state ceiling and V10-R transformer training
if FINAL_EXECUTE:
    if FINAL_RUN_EXACT_STATE_CEILING:
        ceiling_gate = RESULT_ROOT/'G1_exact_state'/'gate.json'
        if FINAL_FORCE_RETRAIN or not ceiling_gate.exists():
            FINAL_EXACT_STATE_RESULTS = run_exact_state_ceiling()
        else:
            print('Reusing exact-state ceiling:', ceiling_gate)

    seed = TRAIN_SEEDS[0]
    final_checkpoint = MODEL_ROOT/f'seed_{seed}'/'best_G4.pt'
    if FINAL_FORCE_RETRAIN or not final_checkpoint.exists():
        FINAL_TRAINING_OUTPUTS = train_all_seeds()
    else:
        print('Reusing trained checkpoint:', final_checkpoint)
    if not final_checkpoint.exists():
        raise RuntimeError(f'Training ended without checkpoint: {final_checkpoint}')
    print('Training checkpoint:', final_checkpoint)
else:
    print('Final execution is OFF; training not started.')

# %% notebook cell 82
# FINAL F6 — untouched validation/test evaluation, optional ablations, and figures
if FINAL_EXECUTE:
    claim_path = RESULT_ROOT/'claim_gates.json'
    if FINAL_FORCE_REEVALUATE or not claim_path.exists():
        FINAL_TEST_RESULTS = evaluate_trained_models()
    else:
        print('Reusing evaluation:', claim_path)

    if FINAL_RUN_ABLATIONS:
        FINAL_ABLATION_TRAINING = train_required_geometry_ablations()
        FINAL_ABLATION_RESULTS = evaluate_required_geometry_ablations()

    FINAL_FIGURES = generate_all_result_figures()
    print('Generated figures:', len(FINAL_FIGURES))
else:
    print('Final execution is OFF; testing and figures not generated.')

# %% notebook cell 83
# FINAL F7 — display all metric tables and generated figures
if FINAL_EXECUTE:
    metric_files = [
        RESULT_ROOT/'force_metrics_long.csv',
        RESULT_ROOT/'force_vector_metrics_long.csv',
        RESULT_ROOT/'all_sensor_metrics_long.csv',
        RESULT_ROOT/'event_impulse_waveform_metrics.csv',
        RESULT_ROOT/'prediction_artifact_index.csv',
        RESULT_ROOT/'G1_exact_state'/'metrics_long.csv',
    ]
    if FINAL_RUN_ABLATIONS:
        metric_files.append(RESULT_ROOT/'required_geometry_ablation_metrics.csv')

    for path in metric_files:
        if path.exists():
            table = pd.read_csv(path)
            print(f'\n===== {path.name}: {len(table)} rows =====')
            with pd.option_context('display.max_rows', None, 'display.max_columns', None,
                                   'display.width', 220, 'display.max_colwidth', 80):
                display(table)

    if (RESULT_ROOT/'claim_gates.json').exists():
        print('\n===== CLAIM GATES =====')
        print(json.dumps(json.load(open(RESULT_ROOT/'claim_gates.json')), indent=2))

    try:
        from IPython.display import Image as IPyImage, display as ipy_display
        for path in sorted(FIGURE_ROOT.glob('*.png')):
            print(path.name)
            ipy_display(IPyImage(filename=str(path)))
    except Exception as exc:
        print('Inline figure display unavailable:', exc)

# %% notebook cell 84
# FINAL F8 — CSV manifest, model/results ZIP, and browser download
if FINAL_EXECUTE:
    import zipfile

    export_root = STRICT_ROOT/'final_export'
    export_root.mkdir(parents=True, exist_ok=True)
    csv_paths = sorted(set(RESULT_ROOT.rglob('*.csv')) | set(MODEL_ROOT.rglob('training_history.csv')))
    csv_manifest = pd.DataFrame([
        {'relative_path': str(p.relative_to(STRICT_ROOT)), 'bytes': p.stat().st_size,
         'sha256': file_sha256(p)} for p in csv_paths
    ])
    csv_manifest_path = export_root/'csv_manifest.csv'
    csv_manifest.to_csv(csv_manifest_path, index=False)
    display(csv_manifest)

    bundle = STRICT_ROOT/'V10R_FINAL_DIAGNOSTIC_RESULTS_AND_MODEL.zip'
    roots = [RESULT_ROOT, FIGURE_ROOT, CONTRACT_ROOT, export_root]
    if FINAL_INCLUDE_MODELS_IN_ZIP:
        roots.append(MODEL_ROOT)
    with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        seen = set()
        for root in roots:
            if not root.exists():
                continue
            for path in sorted(root.rglob('*')):
                if path.is_file() and path not in seen:
                    archive.write(path, path.relative_to(STRICT_ROOT))
                    seen.add(path)
        for extra in (STRICT_ROOT/'paper_data_gate.json', STRICT_ROOT/'strict_pipeline_status.json'):
            if extra.exists() and extra not in seen:
                archive.write(extra, extra.relative_to(STRICT_ROOT))

    print('Final ZIP:', bundle)
    print('ZIP size MiB:', round(bundle.stat().st_size/1024**2, 2))
    print('CSV files included:', len(csv_paths))
    if FINAL_DOWNLOAD_RESULTS_ZIP:
        try:
            from google.colab import files
            files.download(str(bundle))
        except Exception as exc:
            print('Automatic browser download unavailable:', exc)
            print('Download manually from:', bundle)
else:
    print('Final execution is OFF; nothing exported.')
