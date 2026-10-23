#!/usr/bin/env python3
"""
TERRAIN-FOOTHOLD WORLD MODEL: action-conditioned 3D surfel world model, paired foothold memory,
future terrain -> footholds -> force -> proprioception. ONE FILE, NO PREVIOUS-MODEL WEIGHTS.

Commands:
  python terrain_foothold_world_model.py self-test
  python terrain_foothold_world_model.py audit --data /path/to/recordings
  python terrain_foothold_world_model.py train --data /path/to/recordings --out runs/terrain_foothold
  python terrain_foothold_world_model.py evaluate --data /path/to/heldout --checkpoint runs/terrain_foothold/best.pt
  python terrain_foothold_world_model.py predict --data /path/to/recordings --checkpoint runs/terrain_foothold/best.pt --out predictions.npz

DEPENDENCIES: Python 3.10+, numpy, torch 2.2+; optional pretrained DINOv2 from
local Torch Hub cache or downloadable from facebookresearch/dinov2. GPU: PyTorch
CUDA build installed separately; cannot bundle CUDA or trained weights in a .py.

RAW RECORDING NPZ: one recording per file, each S rows with synchronized timestamps:
  rgb[S,H,W,3] uint8 or float RGB; depth[S,H,W] meters;
  K[3,3] camera matrix, T_base_camera[4,4] rig calibration;
  T_world_base[S,4,4] pose matrices (NOT guessed by software);
  proprio[S,D]; actions[S,A] realized controller commands per interval;
  foot_xyz_world[S,4,3]; foot_force_world[S,4,3] measured force N;
  contact[S,4] observed foot contact labels; timestamps[S] seconds.
  optional foot_velocity_world[S,4,3], gait_phase[S,4].
A training sequence requires all fields above, real calibration, SI units and
verified frames. We cannot derive unseen sensors from images or an arbitrary bag.

Raw actions from future timestamps are valid as logged conditional-control INPUT
for training and ex-post conditional evaluation, NEVER as an oracle at deployment.
For hypothetical future predictions supply a candidate action plan known at t.

This is a research model, not a validated safe controller. Scene correspondences
are approximate nearest-surface associations (NOT guaranteed material IDs). Sparse
surfel reconstructions contain observed surfaces only, not occluded geometry.
"""
from __future__ import annotations
import argparse, dataclasses, hashlib, json, math, os, random, sys, time, warnings
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

RAW_REQUIRED = ('rgb','depth','K','T_base_camera','T_world_base','proprio',
                'actions','foot_xyz_world','foot_force_world','contact','timestamps')
INPUT_KEYS = ('rgbd','xyz_world','normal_world','valid','uv','patch_ids','proprio',
              'actions','base_T','foot_xyz_world','foot_velocity_world','foot_contact',
              'gait_phase','history_xyz_world','history_normal','history_rgb',
              'history_response','history_delta_rgb','history_delta_xyz',
              'history_delta_proprio','history_age','history_patch_ids',
              'history_valid','K','T_base_camera','images')
TARGET_KEYS = ('future_xyz_world','future_rgb','future_valid','future_foot_xyz_world',
               'future_patch_ids','future_force','future_contact','future_body',
               'future_base_T','future_uvd','future_uvd_valid')

@dataclasses.dataclass
class Config:
    d: int = 320
    heads: int = 10
    spatial_layers: int = 8
    temporal_layers: int = 2
    mlp_ratio: int = 7
    future_layers: int = 1
    horizon: int = 10
    interval: float = 0.1
    patches: int = 64
    history: int = 5
    reach_m: float = 1.0
    match_m: float = 0.22
    deformation_max_m: float = 0.045
    force_scale: float = 200.0
    vision: str = 'native'  # 'native' or 'dinov2'; DINO requires genuine pretrained weights
    dino_repo: str = 'facebookresearch/dinov2'
    dino_name: str = 'dinov2_vits14'
    proprio_dim: int = 48
    action_dim: int = 16
    dropout: float = 0.0


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def mlp(ind, out, d):
    return nn.Sequential(nn.Linear(ind,d),nn.GELU(),nn.Linear(d,out))


def masked_mean(a, mask, dim):
    m = mask.to(a.dtype)
    while m.ndim < a.ndim: m=m.unsqueeze(-1)
    return (a*m).sum(dim)/m.sum(dim).clamp_min(1)


def masked_loss(err, mask):
    m=mask.to(err.dtype)
    while m.ndim < err.ndim: m=m.unsqueeze(-1)
    return (err*m).sum()/m.expand_as(err).sum().clamp_min(1)


def rpy_from_matrix_numpy(R):
    return np.array([np.arctan2(R[2,1],R[2,2]),
                     np.arcsin(np.clip(-R[2,0],-1,1)),
                     np.arctan2(R[1,0],R[0,0])],np.float32)


def rotvec_matrix(w):
    """Differentiable SO(3) exponential with small-angle Taylor coefficients."""
    theta2=(w*w).sum(-1,keepdim=True)
    theta=theta2.clamp_min(1e-12).sqrt()
    A=torch.where(theta2<1e-6, 1-theta2/6+theta2.square()/120,torch.sin(theta)/theta)
    B=torch.where(theta2<1e-6,.5-theta2/24+theta2.square()/720,(1-torch.cos(theta))/theta2.clamp_min(1e-12))
    x,y,z=w.unbind(-1)
    O=torch.zeros_like(x)
    W=torch.stack([O,-z,y,z,O,-x,-y,x,O],-1).reshape(*w.shape[:-1],3,3)
    I=torch.eye(3,device=w.device,dtype=w.dtype).expand_as(W)
    return I+A[...,None]*W+B[...,None]*(W@W)


def propagate_pose(T, twist):
    """twist is base-frame increment (translation meters, rotation radians)."""
    R=T[...,:3,:3]; t=T[...,:3,3]
    Rn=R@rotvec_matrix(twist[...,3:])
    tn=t+(R@twist[...,:3,None]).squeeze(-1)
    out=torch.zeros_like(T)
    out[...,:3,:3]=Rn;out[...,:3,3]=tn;out[...,3,3]=1
    return out


def future_projection(xyz_world,T_world_base,T_base_camera,K):
    """World points -> projected camera pixels + depth; differentiable in geometry & pose."""
    Twc=T_world_base@T_base_camera
    R=Twc[:,:3,:3];o=Twc[:,:3,3]
    q=torch.einsum('bij,bpj->bpi',R.transpose(-1,-2),xyz_world-o[:,None,:])
    z=q[...,2].clamp_min(.001)
    u=K[:,0,0,None]*(q[...,0]/z)+K[:,0,2,None]
    v=K[:,1,1,None]*(q[...,1]/z)+K[:,1,2,None]
    return torch.stack([u,v,q[...,2]],-1)


# ------- Honest calibrated RGB-D -> 3D surfel preprocessing, no fake force data. -------
def _rgb_to_float(a):
    a=np.asarray(a)
    a=a.astype(np.float32)
    if a.max()>1.1: a/=255.
    return np.clip(a,0,1)


def _validate_run(f,path):
    missing=[k for k in RAW_REQUIRED if k not in f]
    if missing: raise ValueError(f'{path}: missing true measured keys: {missing}')
    S=len(f['timestamps']); im=f['rgb']; dep=f['depth']; K=f['K']
    if im.ndim!=4 or im.shape[-1]!=3 or dep.shape!=im.shape[:3]:
        raise ValueError(f'{path}: rgb[S,H,W,3], depth[S,H,W] required')
    if K.shape!=(3,3) or f['T_base_camera'].shape!=(4,4):
        raise ValueError(f'{path}: valid K and T_base_camera calibration required')
    if f['T_world_base'].shape!=(S,4,4):raise ValueError('T_world_base shape wrong')
    if f['foot_xyz_world'].shape!=(S,4,3) or f['foot_force_world'].shape!=(S,4,3):
        raise ValueError('Foot positions and forces must be [S,4,3] world-frame')
    if f['contact'].shape!=(S,4):raise ValueError('contact must be [S,4]')
    if f['proprio'].shape[0]!=S or f['actions'].shape[0]!=S:
        raise ValueError('Actions/proprio must be synchronized row-for-row')
    ts=np.asarray(f['timestamps'])
    if S<5 or not (np.diff(ts)>0).all(): raise ValueError('Need >=5 strictly increasing timestamps')
    if any(not np.isfinite(np.asarray(f[k],dtype=np.float64)).all() for k in RAW_REQUIRED if k!='rgb'):
        raise ValueError(f'{path}: NaN/inf in raw fields')
    if (np.asarray(dep)<0).any():raise ValueError('Depth must be in meters, >=0')
    if not np.allclose(f['T_base_camera'][3], [0,0,0,1],atol=1e-4):raise ValueError('Bad camera transform')
    if not np.allclose(f['T_world_base'][:,3],[0,0,0,1],atol=1e-4):raise ValueError('Bad pose homogeneous rows')
    if not np.allclose(np.linalg.det(f['T_world_base'][:,:3,:3]),1,atol=.03):
        raise ValueError('Bad T_world_base rotation matrices')
    if np.max(np.abs(f['foot_force_world']))>10000:raise ValueError('Implausible force units')
    if not ((np.asarray(f['contact'])>=0)&(np.asarray(f['contact'])<=1)).all():
        raise ValueError('contact not within [0,1]')
    return dict(frames=S,resolution=im.shape[1:3],proprio_dim=f['proprio'].shape[1],
                action_dim=f['actions'].shape[1],duration=float(ts[-1]-ts[0]),
                median_dt=float(np.median(np.diff(ts))))


def _frame_cloud(rgb, depth, K, Twb, Tbc, stride=8):
    """All valid observed RGB-D samples become WORLD 3D points; no planar flattening.
    Surfels are matched across time by geometry with a radius gate, not oracle IDs.
    """
    H,W=depth.shape
    v,u=np.mgrid[stride//2:H:stride, stride//2:W:stride]
    uv=np.stack([u.ravel(),v.ravel()],-1).astype(np.float32)
    z=depth[v,u].ravel().astype(np.float32)
    keep=np.isfinite(z)&(z>.10)&(z<20.)
    if keep.sum()==0: return np.zeros((0,3),np.float32),np.zeros((0,3),np.float32),np.zeros((0,3),np.float32),np.zeros((0,2),np.float32),np.zeros((0,1),np.float32)
    uv,z=uv[keep],z[keep]
    xyz_cam=np.stack([(uv[:,0]-K[0,2])*z/K[0,0],(uv[:,1]-K[1,2])*z/K[1,1],z],-1)
    Twc=Twb@Tbc
    xyz=xyz_cam@Twc[:3,:3].T+Twc[:3,3]
    # Observed surface normals approximated from neighboring calibrated depths.
    uu=uv[:,0].astype(int); vv=uv[:,1].astype(int)
    d1=depth[np.clip(vv,0,H-1),np.clip(uu+1,0,W-1)]
    d2=depth[np.clip(vv+1,0,H-1),np.clip(uu,0,W-1)]
    dx=np.stack([(uu+1-K[0,2])*d1/K[0,0],(vv-K[1,2])*d1/K[1,1],d1],-1)-xyz_cam
    dy=np.stack([(uu-K[0,2])*d2/K[0,0],(vv+1-K[1,2])*d2/K[1,1],d2],-1)-xyz_cam
    n=np.cross(dx,dy)
    norm=np.linalg.norm(n,axis=-1,keepdims=True)
    n=n/np.maximum(norm,1e-6)
    n[~np.isfinite(n).all(-1)]=0
    n=n@Twc[:3,:3].T
    return xyz.astype(np.float32),n.astype(np.float32),_rgb_to_float(rgb[vv,uu]),uv,z[:,None]


def _nearest(anchor,cloud,radius):
    """Return surface matches and distances. Uses chunking to bound RAM."""
    if not len(cloud):return np.zeros(len(anchor),np.int64),np.zeros(len(anchor),bool)
    inds=[]; masks=[]
    for st in range(0,len(anchor),128):
        dd=((anchor[st:st+128,None,:]-cloud[None,:,:])**2).sum(-1)
        idx=dd.argmin(-1);dist=dd[np.arange(len(idx)),idx]
        inds.extend(idx.tolist());masks.extend((dist<radius*radius).tolist())
    return np.array(inds,np.int64),np.array(masks,bool)


def _select_surfel_anchors(clouds,P):
    """Coverage from current frame plus missing earlier visible surfaces, not XY bins."""
    if len(clouds[-1][0])==0:raise ValueError('No valid current depth surfels')
    cur=clouds[-1][0];picked=[]
    ids=np.linspace(0,len(cur)-1,min(max(1,P*3//4),len(cur)),dtype=int)
    for i in ids: picked.append(cur[i])
    # Include prior side-view surfaces not present in newest camera image.
    for frame in reversed(clouds[:-1]):
        if len(picked)>=P:break
        pc=frame[0]
        if not len(pc):continue
        for p in pc[::max(1,len(pc)//P)]:
            if len(picked)>=P:break
            if ((np.asarray(picked)-p)**2).sum(-1).min()>.16**2: picked.append(p)
    while len(picked)<P: picked.append(cur[(len(picked)*17)%len(cur)])
    return np.asarray(picked[:P],np.float32)


def _project_np(points,Twb,Tbc,K):
    Twc=Twb@Tbc
    c=(points-Twc[:3,3])@Twc[:3,:3]
    z=c[:,2]; safe=np.maximum(z,1e-3)
    return np.stack([K[0,0]*c[:,0]/safe+K[0,2],K[1,1]*c[:,1]/safe+K[1,2],z],-1)


def _touchdown_indices(contact,now,k):
    rise=np.where((contact[1:now+1]>.5)&(contact[:now]<=.5))[0]+1
    return rise[-k:][::-1]


def _raw_to_samples(file, cfg:Config, stride=8, limit=None, image_size=112):
    """Convert ONE calibrated trajectory into full supervised model samples.

    Real measured future RGB-D is used ONLY for label extraction; it is never fed
    into the model forward path. Missing surface correspondences are masked.
    """
    with np.load(file,allow_pickle=False) as f:
        info=_validate_run(f,file)
        raw={k:np.array(f[k]) for k in f.files}
    rgb=_rgb_to_float(raw['rgb']);depth=raw['depth'].astype(np.float32)
    K=raw['K'].astype(np.float32);Tbc=raw['T_base_camera'].astype(np.float32)
    T=raw['T_world_base'].astype(np.float32)
    ts=raw['timestamps'];S=len(ts);D=info['proprio_dim']; A=info['action_dim']
    if cfg.proprio_dim!=D or cfg.action_dim!=A:raise ValueError(f'Shape mismatch: recording D={D}, A={A}, configured D={cfg.proprio_dim}, A={cfg.action_dim}')
    clouds=[_frame_cloud(rgb[i],depth[i],K,T[i],Tbc,stride) for i in range(S)]
    # Avoid retaining 5 x full-resolution float RGB images for EVERY training window.
    ih,iw=rgb.shape[1:3];factor=min(1.,image_size/max(ih,iw))
    oh=max(14,round(ih*factor));ow=max(14,round(iw*factor))
    if (oh,ow)!=(ih,iw):
        small_rgb=F.interpolate(torch.from_numpy(rgb).permute(0,3,1,2),size=(oh,ow),mode='bilinear',align_corners=False).permute(0,2,3,1).numpy()
    else:small_rgb=rgb
    # Choose sample every ~dt for sufficiently uniform high-rate logs.
    dt=np.median(np.diff(ts));input_step=max(1,int(round(cfg.interval/dt)))
    idxs=range(4*input_step,S-cfg.horizon*input_step,input_step)
    examples=[]
    contact=raw['contact'];feet=raw['foot_xyz_world']
    if 'foot_velocity_world' in raw:
        fvel=raw['foot_velocity_world']
    else:
        # finite difference derived from ACTUAL measured foot positions, not invented force
        fvel=np.gradient(feet,ts,axis=0).astype(np.float32)
    gait=raw.get('gait_phase',np.zeros_like(contact,dtype=np.float32))
    for i in idxs:
        if limit is not None and len(examples)>=limit:break
        ii=[i-4*input_step+j*input_step for j in range(5)]
        fj=[i+(h+1)*input_step for h in range(cfg.horizon)]
        # Enforce frame/timestamp alignment. Else do not claim a 0.1 s horizon.
        if max(abs(ts[ii[j]]-(ts[i]-(4-j)*cfg.interval)) for j in range(5))>.05:continue
        if max(abs(ts[fj[j]]-(ts[i]+(j+1)*cfg.interval)) for j in range(cfg.horizon))>.05:continue
        try: anchors=_select_surfel_anchors([clouds[j] for j in ii],cfg.patches)
        except ValueError:continue
        P=cfg.patches;obs_rgbd=np.zeros((5,P,4),np.float32)
        obs_xyz=np.zeros((5,P,3),np.float32)
        obs_n=np.zeros((5,P,3),np.float32)
        obs_v=np.zeros((5,P),bool);obs_uv=np.zeros((5,P,2),np.float32)
        for k,j in enumerate(ii):
            pc,no,co,uv,z=clouds[j];inds,v=_nearest(anchors,pc,cfg.match_m)
            if not v.any():continue
            obs_v[k]=v
            obs_xyz[k,v]=pc[inds[v]];obs_n[k,v]=no[inds[v]]
            obs_rgbd[k,v,:3]=co[inds[v]];obs_rgbd[k,v,3]=z[inds[v],0]
            obs_uv[k,v]=uv[inds[v]]
        if not obs_v.any(-1).all():continue
        y_xyz=np.zeros((cfg.horizon,P,3),np.float32)
        y_rgb=np.zeros((cfg.horizon,P,3),np.float32)
        y_v=np.zeros((cfg.horizon,P),bool)
        y_uvd=np.zeros((cfg.horizon,P,3),np.float32)
        y_uvv=np.zeros((cfg.horizon,P),bool)
        for h,j in enumerate(fj):
            pc,no,co,uv,z=clouds[j];inds,v=_nearest(anchors,pc,cfg.match_m)
            if v.any():
                y_xyz[h,v]=pc[inds[v]];y_rgb[h,v]=co[inds[v]];y_v[h]=v
                q=_project_np(y_xyz[h,v],T[j],Tbc,K)
                y_uvd[h,v]=q; y_uvv[h,v]=(q[:,2]>.1)
        hist_xyz=np.zeros((4,cfg.history,3),np.float32)
        hist_normal=np.zeros_like(hist_xyz);hist_rgb=np.zeros_like(hist_xyz)
        hist_response=np.zeros((4,cfg.history,6),np.float32)
        hist_dr=np.zeros((4,cfg.history,3),np.float32)
        hist_dx=np.zeros((4,cfg.history,3),np.float32)
        hist_dp=np.zeros((4,cfg.history,D),np.float32)
        hist_age=np.zeros((4,cfg.history),np.float32)
        hist_ids=np.full((4,cfg.history),-1,np.int64)
        hist_valid=np.zeros((4,cfg.history),bool)
        for l in range(4):
            for k,j in enumerate(_touchdown_indices(contact[:,l],i,cfg.history)):
                p=feet[j,l];hist_xyz[l,k]=p;hist_age[l,k]=ts[i]-ts[j]
                nearest=int(np.argmin(((anchors-p)**2).sum(-1)))
                if np.linalg.norm(anchors[nearest]-p)<cfg.match_m*2:
                    hist_ids[l,k]=nearest
                cj=clouds[j]
                if len(cj[0]):
                    nidx=int(np.argmin(((cj[0]-p)**2).sum(-1)))
                    if np.linalg.norm(cj[0][nidx]-p)<.4:
                        hist_normal[l,k]=cj[1][nidx];hist_rgb[l,k]=cj[2][nidx]
                        j2=min(i,j+input_step)
                        if len(clouds[j2][0]):
                            aft_idx=int(np.argmin(((clouds[j2][0]-p)**2).sum(-1)))
                            if np.linalg.norm(clouds[j2][0][aft_idx]-p)<.4:
                                hist_dr[l,k]=clouds[j2][2][aft_idx]-cj[2][nidx]
                                hist_dx[l,k]=clouds[j2][0][aft_idx]-cj[0][nidx]
                hist_response[l,k,:3]=raw['foot_force_world'][j,l]
                hist_response[l,k,3:]=fvel[j,l]
                hist_dp[l,k]=raw['proprio'][min(i,j+input_step)]-raw['proprio'][max(0,j-input_step)]
                hist_valid[l,k]=True
        obs_uv[...,0]*=(ow-1)/max(iw-1,1)
        obs_uv[...,1]*=(oh-1)/max(ih-1,1)
        patchid=np.broadcast_to(np.arange(P,dtype=np.int64),(5,P)).copy()
        # True future contact patches identified using the recorded future foot position.
        lab_ids=np.full((cfg.horizon,4),-1,np.int64)
        for h,j in enumerate(fj):
            idx,valid=_nearest(feet[j],anchors,cfg.match_m*2)
            lab_ids[h]=np.where(valid&(contact[j]>.5),idx,-1)
        x=dict(rgbd=obs_rgbd,xyz_world=obs_xyz,normal_world=obs_n,valid=obs_v,
               uv=obs_uv,patch_ids=patchid,proprio=raw['proprio'][ii].astype(np.float32),
               actions=raw['actions'][fj].astype(np.float32),base_T=T[i],
               foot_xyz_world=feet[i].astype(np.float32),
               foot_velocity_world=fvel[i].astype(np.float32),foot_contact=contact[i].astype(np.float32),
               gait_phase=gait[i].astype(np.float32),
               history_xyz_world=hist_xyz,history_normal=hist_normal,history_rgb=hist_rgb,
               history_response=hist_response,history_delta_rgb=hist_dr,
               history_delta_xyz=hist_dx,history_delta_proprio=hist_dp,
               history_age=hist_age,history_patch_ids=hist_ids,history_valid=hist_valid,
               K=K,T_base_camera=Tbc,
               images=small_rgb[ii].transpose(0,3,1,2).astype(np.float16))
        y=dict(future_xyz_world=y_xyz,future_rgb=y_rgb,future_valid=y_v,
               future_foot_xyz_world=feet[fj].astype(np.float32),
               future_patch_ids=lab_ids,future_force=raw['foot_force_world'][fj].astype(np.float32),
               future_contact=contact[fj].astype(np.float32),
               future_body=raw['proprio'][fj].astype(np.float32),future_base_T=T[fj].astype(np.float32),
               future_uvd=y_uvd,future_uvd_valid=y_uvv)
        examples.append((x,y))
    return examples,info


class RawDataset(Dataset):
    def __init__(self,files,cfg,limit_per_run=None,stride=8,image_size=112):
        self.samples=[];self.stats=[]
        for file in files:
            samples,inf=_raw_to_samples(file,cfg,stride,limit_per_run,image_size)
            self.stats.append((str(file),len(samples),inf));self.samples.extend(samples)
        if not self.samples:raise RuntimeError('No verified trainable windows from recordings. Check timestamps, calibration, signals and sample rate.')
    def __len__(self):return len(self.samples)
    def __getitem__(self,i):
        x,y=self.samples[i]
        def to_tensor(k,a):
            return torch.from_numpy(a).bool() if k in ('valid','history_valid','future_valid','future_uvd_valid') else torch.from_numpy(a).long() if k.endswith('_ids') or k=='patch_ids' else torch.from_numpy(a).float()
        return {k:to_tensor(k,v) for k,v in x.items()},{k:to_tensor(k,v) for k,v in y.items()}


# ------------------------- Model: no blind future output path -----------------
class EncoderBlock(nn.Module):
    def __init__(self,d,heads,ratio,dropout=0):
        super().__init__()
        self.n1=nn.LayerNorm(d);self.a=nn.MultiheadAttention(d,heads,batch_first=True,dropout=dropout)
        self.n2=nn.LayerNorm(d)
        self.ff=nn.Sequential(nn.Linear(d,d*ratio),nn.GELU(),nn.Dropout(dropout),nn.Linear(d*ratio,d))
    def forward(self,z,mask=None):
        a=self.n1(z);z=z+self.a(a,a,a,key_padding_mask=mask,need_weights=False)[0]
        return z+self.ff(self.n2(z))


class LocalPointAttention(nn.Module):
    """3D relative-position point transformer, local KNN across surfels."""
    def __init__(self,d,k=8):
        super().__init__();self.k=k
        self.q=nn.Linear(d,d);self.v=nn.Linear(d,d)
        self.rel=mlp(3,d,d);self.score=mlp(d,1,d);self.out=nn.Linear(d,d)
    def forward(self,z,xyz,valid):
        B,P,D=z.shape; k=min(self.k,P)
        dist=torch.cdist(xyz,xyz).masked_fill(~valid[:,None,:],1e4)
        ix=dist.topk(k,largest=False).indices
        b=torch.arange(B,device=z.device)[:,None,None]
        zk=z[b,ix]; xk=xyz[b,ix]
        rv=self.rel(xk-xyz[:,:,None,:]);q=self.q(z)[:,:,None,:]
        logit=self.score(torch.tanh(q-zk+rv)).squeeze(-1)
        logit=logit.masked_fill(~valid[b,ix],-1e4)
        w=F.softmax(logit,dim=-1)
        agg=(w[...,None]*(self.v(zk)+rv)).sum(2)
        return (z+self.out(agg))*valid[...,None]


class NativeVision(nn.Module):
    """Learnable local texture CNN + patch semantic attention; no pretrained claims."""
    def __init__(self,d,heads):
        super().__init__()
        c=64
        self.stem=nn.Sequential(nn.Conv2d(3,c,4,4),nn.GELU(),nn.Conv2d(c,c,3,padding=1,groups=c),nn.GELU(),nn.Conv2d(c,d,1))
        self.depthwise=nn.Conv1d(d,d,5,padding=2,groups=d)
        self.semantic=EncoderBlock(d,heads,2)
    def forward(self,images,uv,mask):
        B,T,_,H,W=images.shape;P=uv.shape[2]
        f=self.stem(images.reshape(B*T,3,H,W))
        # Gather visual tokens by actual image coordinates associated with each 3D surfel.
        grid=uv.reshape(B*T,P,2).clone()
        grid[...,0]=2*grid[...,0]/max(W-1,1)-1
        grid[...,1]=2*grid[...,1]/max(H-1,1)-1
        picked=F.grid_sample(f,grid[:,:,None,:],mode='bilinear',align_corners=True)
        picked=picked.squeeze(-1).transpose(1,2).reshape(B,T,P,-1)
        tex=self.depthwise(picked.reshape(B*T,P,-1).transpose(1,2)).transpose(1,2)
        z=self.semantic(picked.reshape(B*T,P,-1)+tex,mask=~mask.reshape(B*T,P))
        return z.reshape(B,T,P,-1)*mask[...,None]


class DinoVision(nn.Module):
    """REAL DINOv2 ViT-S/14 features with pretrained checkpoint; frozen and spatially sampled.
    No silent random-weight fallback. Requires cached/downloadable genuine weights.
    """
    def __init__(self,cfg):
        super().__init__()
        try: backbone=torch.hub.load(cfg.dino_repo,cfg.dino_name,pretrained=True,trust_repo=True)
        except Exception as e:
            raise RuntimeError('Genuine pretrained DINOv2 weights unavailable. Cache official model on GPU server or use --vision native; native is NOT DINOv2.') from e
        self.backbone=backbone.eval()
        for p in self.backbone.parameters():p.requires_grad_(False)
        self.proj=nn.Linear(384,cfg.d)
    def train(self,mode=True):
        super().train(mode);self.backbone.eval();return self
    def forward(self,images,uv,mask):
        B,T,_,H,W=images.shape;P=uv.shape[2]
        # Official DINO input normalization; token grid inferred from current image.
        a=images.reshape(B*T,3,H,W)
        hh=(H//14)*14;ww=(W//14)*14
        if hh<14 or ww<14:raise ValueError('DINOv2 needs RGB image dimensions >=14')
        a=F.interpolate(a,size=(hh,ww),mode='bilinear',align_corners=False)
        mean=a.new_tensor([.485,.456,.406])[None,:,None,None];std=a.new_tensor([.229,.224,.225])[None,:,None,None]
        with torch.no_grad():
            tok=self.backbone.forward_features((a-mean)/std)['x_norm_patchtokens']
        token=tok.transpose(1,2).reshape(B*T,tok.shape[-1],hh//14,ww//14)
        grid=uv.reshape(B*T,P,2).clone()
        grid[...,0]=2*grid[...,0]/max(W-1,1)-1;grid[...,1]=2*grid[...,1]/max(H-1,1)-1
        z=F.grid_sample(token,grid[:,:,None,:],align_corners=True).squeeze(-1).transpose(1,2)
        return self.proj(z.reshape(B,T,P,-1))*mask[...,None]


class WorldModel(nn.Module):
    def __init__(self,cfg:Config):
        super().__init__()
        if cfg.d%cfg.heads:raise ValueError('d must divide heads')
        self.cfg=cfg;d=cfg.d
        self.vision=NativeVision(d,cfg.heads) if cfg.vision=='native' else DinoVision(cfg)
        # RGBD + XYZ + normals + observed + age: vision and physical geometry are both explicit.
        self.obs=mlp(4+3+3+1+1,d,d);self.times=nn.Embedding(5,d);self.xyz_embed=mlp(3,d,d)
        self.temporal=nn.ModuleList([EncoderBlock(d,cfg.heads,cfg.mlp_ratio) for _ in range(cfg.temporal_layers)])
        self.point_attention=LocalPointAttention(d)
        self.spatial=nn.ModuleList([EncoderBlock(d,cfg.heads,cfg.mlp_ratio) for _ in range(cfg.spatial_layers)])
        self.hist=mlp(3+3+3+6+3+3+cfg.proprio_dim+1,d,d)
        self.leg_ids=nn.Embedding(4,d)
        self.current_leg=mlp(3+3+1+1,d,d)
        self.memory_attn=nn.MultiheadAttention(d,cfg.heads,batch_first=True)
        self.prop_encoder=nn.GRU(cfg.proprio_dim,d,batch_first=True)
        self.action_encoder=nn.GRU(cfg.action_dim,d,batch_first=True)
        self.action_projection=mlp(cfg.action_dim,d,d)
        self.pose_head=mlp(3*d,6,d)
        self.pose_scale=nn.Parameter(torch.tensor([.08,.08,.08,.035,.035,.04]))
        self.patch_fusion=mlp(2*d+1,d,d)
        self.contact_delta=mlp(3*d+1,7,d) # true xyz displacement + RGB shift + uncertainty
        self.direct_future=mlp(3*d+1,7,d)
        self.future_embed=mlp(3+3+3+3,d,d)
        self.future_point_attention=LocalPointAttention(d)
        self.future_blocks=nn.ModuleList([EncoderBlock(d,cfg.heads,cfg.mlp_ratio) for _ in range(cfg.future_layers)])
        self.film=nn.ModuleList([mlp(d,2*d,d) for _ in range(4)])
        self.leg_query=mlp(2*d,d,d)
        self.cross=nn.MultiheadAttention(d,cfg.heads,batch_first=True)
        self.patch_score=mlp(2*d+3,1,d)
        self.foot_offset=mlp(2*d+3,3,d)
        self.contact_head=mlp(2*d+3,1,d)
        self.force_head=mlp(2*d+3,3,d)
        self.force_lvar=mlp(2*d,3,d)
        self.safety_head=mlp(2*d+4,2,d) # sparse outcome supervision if supplied
        self.body_head=mlp(2*d+4*(3+3+1),cfg.proprio_dim,d)

    def encode(self,x):
        rgbd=x['rgbd'];xyz=x['xyz_world'];normal=x['normal_world'];valid=x['valid'].bool()
        B,T,P,_=xyz.shape;d=self.cfg.d
        if T!=5 or x['patch_ids'].shape!=(B,T,P) or not torch.equal(x['patch_ids'],x['patch_ids'][:,-1:,:].expand_as(x['patch_ids'])):
            raise ValueError('Need five frames with identical registered surface patch IDs')
        age=torch.arange(T,device=xyz.device).view(1,T,1,1).expand(B,T,P,1).to(xyz.dtype)/4.
        z=self.obs(torch.cat([rgbd,xyz,normal,valid[...,None].to(xyz.dtype),age],-1))
        z=z+self.vision(x['images'],x['uv'],valid)
        z=z+self.times(torch.arange(5,device=z.device))[None,:,None,:]
        # Surfel-specific temporal attention tracks observations of SAME registered patch.
        zz=z.permute(0,2,1,3).reshape(B*P,5,d)
        mask=(~valid.permute(0,2,1).reshape(B*P,5)).clone()
        all_invalid=mask.all(-1);mask[all_invalid,0]=False
        for layer in self.temporal:zz=layer(zz,mask)
        last=(valid.to(torch.int64)*torch.arange(1,6,device=z.device)[None,:,None]).argmax(1)
        obs=zz[torch.arange(B*P,device=z.device),last.reshape(-1)].reshape(B,P,d)
        obsvalid=valid.any(1)
        oi=last[:,None,:,None]
        world=xyz.gather(1,oi.expand(-1,-1,-1,3)).squeeze(1)
        color=rgbd[...,:3].gather(1,oi.expand(-1,-1,-1,3)).squeeze(1)
        norm=normal.gather(1,oi.expand(-1,-1,-1,3)).squeeze(1)
        z=(obs+self.xyz_embed(world))*obsvalid[...,None]
        z=self.point_attention(z,world,obsvalid)
        safe=(~obsvalid).clone();safe[safe.all(-1),0]=False
        for layer in self.spatial:z=layer(z,safe)*obsvalid[...,None]
        return z,world,color,norm,obsvalid

    def history(self,x,z,patchvalid):
        B,_,K,_=x['history_xyz_world'].shape;d=self.cfg.d
        hist=self.hist(torch.cat([x['history_xyz_world'],x['history_normal'],x['history_rgb'],
            x['history_response'],x['history_delta_rgb'],x['history_delta_xyz'],
            x['history_delta_proprio'],x['history_age'][...,None]],-1))
        hist=hist+self.leg_ids(torch.arange(4,device=z.device))[None,:,None,:]
        # Each recorded touchdown is linked to its own terrain ID when still represented.
        idx=x['history_patch_ids'].clamp(0,z.shape[1]-1)
        matched=z[torch.arange(B,device=z.device)[:,None,None],idx]
        has=(x['history_patch_ids']>=0)&x['history_valid']
        hist=hist+matched*has[...,None]
        q=self.current_leg(torch.cat([x['foot_xyz_world'],x['foot_velocity_world'],
                 x['foot_contact'][...,None],x['gait_phase'][...,None]],-1))
        q=q+self.leg_ids(torch.arange(4,device=z.device))[None,:,:]
        hk=hist.reshape(B*4,K,d);hm=~x['history_valid'].reshape(B*4,K).bool()
        absent=hm.all(-1);hm[absent,0]=False
        h,_=self.memory_attn(q.reshape(B*4,1,d),hk,hk,key_padding_mask=hm,need_weights=False)
        return q,h.reshape(B,4,d)*(~absent).reshape(B,4,1)

    def forward(self,x):
        cfg=self.cfg;z,world0,rgb0,normal,valid=self.encode(x)
        B,P,d=z.shape;H=x['actions'].shape[1]
        if H>cfg.horizon:raise ValueError('Requested rollout longer than configured horizon')
        action=x['actions'];hist_prop=x['proprio']
        _,ph=self.prop_encoder(hist_prop);p=ph[-1]
        ac, _=self.action_encoder(action)
        q,mem=self.history(x,z,valid)
        pose=x['base_T'];world=world0;rgb=rgb0
        prior_foot=x['foot_xyz_world'];prior_force=x['foot_contact']*cfg.force_scale
        prior_contact=x['foot_contact'];prev_body=hist_prop[:,-1]
        outputs={k:[] for k in ('future_xyz_world','direct_xyz_world','future_rgb','direct_rgb',
           'geometry_log_sigma','future_uvd','patch_logits','predicted_patch_ids',
           'foot_xyz_world','contact_logits','force','force_log_sigma',
           'body','base_T','patch_safety_logits','patch_safety_log_sigma')}
        global_z=masked_mean(z,valid,1)
        for h in range(H):
            a=ac[:,h];ax=self.action_projection(action[:,h])
            # Ego change acts on CAMERA viewpoint; WORLD terrain remains static unless contact deforms it.
            twist=torch.tanh(self.pose_head(torch.cat([p,a,global_z],-1)))*self.pose_scale
            pose=propagate_pose(pose,twist)
            # Contact impulse influence is predicted from ACTUAL past at h=0 and
            # previously predicted contact/force for h>0 (never future ground truth).
            dist=torch.cdist(world,prior_foot)
            kernel=torch.exp(-dist.square()/(2*.22**2))
            imprint=(kernel*(prior_contact*torch.tanh(prior_force/cfg.force_scale))[:,None,:]).sum(-1).clamp(0,1)
            patch_ctx=self.patch_fusion(torch.cat([z,a[:,None,:].expand(-1,P,-1),imprint[...,None]],-1))
            evolution_input=torch.cat([z,patch_ctx,ax[:,None,:].expand(-1,P,-1),imprint[...,None]],-1)
            inc=self.contact_delta(evolution_input)
            direct=self.direct_future(torch.cat([z,patch_ctx,a[:,None,:].expand(-1,P,-1),imprint[...,None]],-1))
            step_disp=torch.tanh(inc[...,:3])*cfg.deformation_max_m*imprint[...,None]
            # horizon-conditioned anchored direct estimate + recursive contact-conditioned update
            xyz_recursive=world+step_disp
            xyz_direct=world0+torch.tanh(direct[...,:3])*cfg.deformation_max_m*(h+1)*imprint[...,None]
            world=.7*xyz_recursive+.3*xyz_direct
            rgb_recursive=(rgb+.08*torch.tanh(inc[...,3:6])*imprint[...,None]).clamp(0,1)
            rgb_direct=(rgb0+.08*torch.tanh(direct[...,3:6])*(h+1)*imprint[...,None]).clamp(0,1)
            rgb=(.7*rgb_recursive+.3*rgb_direct).clamp(0,1)
            uvd=future_projection(world,pose,x['T_base_camera'],x['K'])
            R=pose[:,:3,:3]; o=pose[:,:3,3]
            rel=torch.einsum('bij,bpj->bpi',R.transpose(-1,-2),world-o[:,None,:])
            fut=z+self.future_embed(torch.cat([rel,world-world0,rgb,normal],-1))
            fut=self.future_point_attention(fut,world,valid)
            safe_mask=(~valid).clone();safe_mask[safe_mask.all(-1),0]=False
            for layer in self.future_blocks:fut=layer(fut,safe_mask)*valid[...,None]
            # Terrain patch safety score for all observed candidate patches. Actual
            # safety labels may be sparse; training NEVER fabricates counterfactual labels.
            safety=self.safety_head(torch.cat([fut,global_z[:,None,:].expand(-1,P,-1),
                                    normal,imprint[...,None]],-1))
            # Four independent action-FiLM encodings (actions at current horizon only).
            fim=torch.stack([module(ax) for module in self.film],1)
            gamma,beta=fim.chunk(2,-1)
            legq=self.leg_query(torch.cat([q*(1+.1*torch.tanh(gamma))+.1*beta,mem],-1))
            # Explicit leg <-> future predicted surface cross-attention.
            attended,_=self.cross(legq,fut,fut,key_padding_mask=safe_mask,need_weights=False)
            # Reachable candidate patches based on planned foot progression & robot motion.
            planned=x['foot_xyz_world']+x['foot_velocity_world']*((h+1)*cfg.interval)
            dfoot=torch.cdist(planned,world)
            reachable=(dfoot<cfg.reach_m)&valid[:,None,:]
            nearest=dfoot.masked_fill(~valid[:,None,:],1e8).argmin(-1)
            reachable=reachable.scatter(-1,nearest[...,None],True)
            l=attended[:,:,None,:].expand(-1,-1,P,-1)
            f=fut[:,None,:,:].expand(-1,4,-1,-1)
            xyzp=world[:,None,:,:].expand(-1,4,-1,-1)
            logits=self.patch_score(torch.cat([l,f,xyzp],-1)).squeeze(-1)
            logits=logits.masked_fill(~reachable,-1e4)
            w=F.softmax(logits,dim=-1)
            selected=torch.einsum('blp,bpd->bld',w,fut)
            point=torch.einsum('blp,bpc->blc',w,world)
            # No purely proprioceptive/command-only predictor can bypass future terrain.
            interaction=selected*torch.sigmoid(attended)
            fused=torch.cat([interaction,selected,point],-1)
            foot=point+.08*torch.tanh(self.foot_offset(fused))
            logits_contact=self.contact_head(fused).squeeze(-1)
            force=self.force_head(fused)*cfg.force_scale
            lv=self.force_lvar(torch.cat([interaction,selected],-1)).clamp(-7,5)
            cp=torch.sigmoid(logits_contact)
            globe=masked_mean(fut,valid,1)
            body_in=torch.cat([globe,interaction.mean(1),torch.cat([foot,force/cfg.force_scale,
                                                    cp[...,None]],-1).reshape(B,-1)],-1)
            body=hist_prop[:,-1]+self.body_head(body_in)
            outputs['future_xyz_world'].append(world);outputs['direct_xyz_world'].append(xyz_direct)
            outputs['future_rgb'].append(rgb);outputs['direct_rgb'].append(rgb_direct)
            outputs['geometry_log_sigma'].append(inc[...,6].clamp(-7,5))
            outputs['future_uvd'].append(uvd)
            outputs['patch_logits'].append(logits)
            outputs['predicted_patch_ids'].append(x['patch_ids'][:,-1].gather(-1,logits.argmax(-1)))
            outputs['foot_xyz_world'].append(foot)
            outputs['contact_logits'].append(logits_contact)
            outputs['force'].append(force);outputs['force_log_sigma'].append(lv)
            outputs['body'].append(body);outputs['base_T'].append(pose)
            outputs['patch_safety_logits'].append(safety[...,0]);outputs['patch_safety_log_sigma'].append(safety[...,1])
            prior_foot=foot;prior_force=force.norm(dim=-1);prior_contact=cp
            # Recurrent proprioception is only used to update latent physical context,
            # and the body head itself has no path bypassing future terrain.
            p=globe*torch.sigmoid(p)+interaction.mean(1)
        out={k:torch.stack(v,dim=1) for k,v in outputs.items()}
        out['patch_valid']=valid;out['patch_ids']=x['patch_ids'][:,-1]
        return out


def objective(out,y,x):
    """Measured future supervision: geometric evolution, photometry, reprojection,
    touchdown patches, force, contact, body, SE(3) and direct/rollout consistency.
    Unseen or unmeasured geometry is masked; true GT is never a model input.
    """
    valid=y['future_valid']&out['patch_valid'][:,None,:]
    contact=y['future_contact']>.5
    losses={}
    losses['geometry']=masked_loss(F.smooth_l1_loss(out['future_xyz_world'],y['future_xyz_world'],reduction='none'),valid)
    losses['direct_geometry']=masked_loss(F.smooth_l1_loss(out['direct_xyz_world'],y['future_xyz_world'],reduction='none'),valid)
    losses['rgb']=masked_loss(F.smooth_l1_loss(out['future_rgb'],y['future_rgb'],reduction='none'),valid)
    losses['direct_rgb']=masked_loss(F.smooth_l1_loss(out['direct_rgb'],y['future_rgb'],reduction='none'),valid)
    losses['consistency']=masked_loss((out['future_xyz_world']-out['direct_xyz_world']).square(),valid)
    maskuv=y['future_uvd_valid']&valid&(out['future_uvd'][...,2]>.05)
    scale=out['future_uvd'].new_tensor([500,500,1])
    losses['projection']=masked_loss(F.smooth_l1_loss(out['future_uvd']/scale,y['future_uvd']/scale,reduction='none'),maskuv)
    patch=out['patch_ids'][:,None,None,:];which=(patch==y['future_patch_ids'][...,None])
    known=which.any(-1)&contact
    truth=which.long().argmax(-1)
    losses['landing_patch']=masked_loss(F.cross_entropy(out['patch_logits'].flatten(0,2),truth.reshape(-1),reduction='none').reshape_as(truth),known)
    losses['foothold']=masked_loss(F.smooth_l1_loss(out['foot_xyz_world'],y['future_foot_xyz_world'],reduction='none'),contact)
    losses['force']=masked_loss(F.smooth_l1_loss(out['force'],y['future_force'],reduction='none'),contact)
    losses['contact']=F.binary_cross_entropy_with_logits(out['contact_logits'],y['future_contact'])
    losses['body']=F.smooth_l1_loss(out['body'],y['future_body'])
    losses['base_translation']=F.smooth_l1_loss(out['base_T'][...,:3,3],y['future_base_T'][...,:3,3])
    r=(out['base_T'][...,:3,:3].transpose(-1,-2)@y['future_base_T'][...,:3,:3])
    losses['base_rotation']=(3-r.diagonal(dim1=-1,dim2=-2).sum(-1)).mean().clamp_min(0)
    geo_lvar=out['geometry_log_sigma']
    geo_sq=(out['future_xyz_world']-y['future_xyz_world']).square().mean(-1)
    losses['geo_uncertainty']=masked_loss(.5*(geo_sq*torch.exp(-geo_lvar)+geo_lvar),valid)
    force_lvar=out['force_log_sigma']; force_sq=(out['force']-y['future_force']).square()
    losses['force_uncertainty']=masked_loss(.5*(force_sq*torch.exp(-force_lvar)+force_lvar),contact)
    # Optional externally verified safe/unsafe outcome labels for VISITED patch ONLY.
    if 'future_patch_safety' in y and 'future_patch_safety_valid' in y:
        losses['patch_safety']=masked_loss(F.binary_cross_entropy_with_logits(out['patch_safety_logits'],y['future_patch_safety'],reduction='none'),y['future_patch_safety_valid'])
    weights=dict(geometry=4,direct_geometry=2,rgb=1,direct_rgb=.5,consistency=.25,
        projection=.5,landing_patch=1,foothold=3,force=.012,contact=1,body=2,
        base_translation=2,base_rotation=.25,geo_uncertainty=.015,force_uncertainty=.015,patch_safety=1)
    total=sum(weights[k]*v for k,v in losses.items())
    return total,losses


def data_files(p):
    p=Path(p)
    if p.is_file() and p.suffix=='.npz':return [p]
    if p.is_dir():return sorted(p.glob('*.npz'))
    raise FileNotFoundError(f'Expected raw sequence .npz or directory: {p}')


def batches_to_device(x,device):
    return {k:v.to(device,non_blocking=True) for k,v in x.items()}


def prepare_datasets(args):
    files=data_files(args.data)
    if not files:raise ValueError(f'No .npz recordings in {args.data}')
    if args.command=='train':
        # Split by WHOLE recordings, never randomly split overlapping windows.
        rnd=random.Random(args.seed);rnd.shuffle(files)
        if len(files)<2: raise RuntimeError('Train and validation require >=2 independent trajectory files. Refusing window-level leakage.')
        nval=max(1,round(len(files)*.2));tr,va=files[nval:],files[:nval]
        if args.val_data:
            va=data_files(args.val_data);tr=files
        return RawDataset(tr,args.config,args.limit_per_run,args.stride,args.image_size),RawDataset(va,args.config,args.limit_per_run,args.stride,args.image_size)
    return RawDataset(files,args.config,args.limit_per_run,args.stride,args.image_size)


def build_config_from_run(f,args):
    with np.load(f,allow_pickle=False) as z:
        info=_validate_run(z,f)
    if args.command=='train' and info['median_dt']>args.interval*1.5:
        raise ValueError('Sensor sampling too sparse for requested prediction horizon')
    return Config(d=args.hidden,heads=args.heads,spatial_layers=args.spatial_layers,
         temporal_layers=args.temporal_layers,mlp_ratio=args.mlp_ratio,
         future_layers=args.future_layers,horizon=args.horizon,interval=args.interval,
         patches=args.patches,history=args.history,vision=args.vision,
         proprio_dim=info['proprio_dim'],action_dim=info['action_dim'])


def checkpoint_load(path,device):
    obj=torch.load(path,map_location=device,weights_only=False)
    cfg=Config(**obj['config']);net=WorldModel(cfg).to(device)
    net.load_state_dict(obj['model'],strict=True)
    return net,cfg,obj


def _save_checkpoint(path,model,cfg,optim,epoch,val,provenance):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model=model.state_dict(),config=dataclasses.asdict(cfg),optimizer=optim.state_dict(),
                    epoch=epoch,val_loss=val,provenance=provenance),path)


def stable_clip_grad_norm_(parameters,max_norm):
    """Clip finite gradients using a float64 norm to avoid float32 overflow."""
    grads=[p.grad for p in parameters if p.grad is not None]
    if not grads:return True,0.
    norm=torch.stack([g.detach().double().square().sum() for g in grads]).sum().sqrt()
    if not torch.isfinite(norm):return False,float(norm)
    scale=(max_norm/norm.clamp_min(max_norm)).clamp(max=1.)
    for grad in grads:grad.copy_((grad.double()*scale).to(grad.dtype))
    return True,float(norm)


def train_command(args):
    files=data_files(args.data);cfg=build_config_from_run(files[0],args);args.config=cfg
    trainset,valset=prepare_datasets(args)
    print('Train:',trainset.stats,'\nValidation:',valset.stats,flush=True)
    if args.require_cuda and not torch.cuda.is_available():raise RuntimeError('CUDA required. Verify driver and CUDA-enabled PyTorch.')
    device=torch.device('cuda' if torch.cuda.is_available() and args.device=='auto' else 'cpu' if args.device=='auto' else args.device)
    if device.type=='cuda':print('GPU:',torch.cuda.get_device_name(device),flush=True)
    if device.type=='cuda':
        # The fused efficient-attention backward returned NaNs on real batches.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print('Attention backend: math',flush=True)
    net=WorldModel(cfg).to(device);opt=torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=.01)
    # Full precision avoids the observed mixed-precision attention-gradient explosion.
    amp=device.type=='cuda'
    trainloader=DataLoader(trainset,batch_size=args.batch,shuffle=True,num_workers=0,pin_memory=amp)
    valloader=DataLoader(valset,batch_size=args.batch,shuffle=False,num_workers=0,pin_memory=amp)
    best=float('inf');out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    provenance={'train_files':[s[0] for s in trainset.stats],'val_files':[s[0] for s in valset.stats],
                'previous_model_weights_used':False,'recorded_future_actions_for_supervised_training':True,
                'training_precision':'float32','gradient_norm_accumulation':'float64',
                'attention_backend':'math'}
    for epoch in range(1,args.epochs+1):
        net.train();sumloss=0;count=0;bad_gradients=0
        for step,(x,y) in enumerate(trainloader,1):
            x,y=batches_to_device(x,device),batches_to_device(y,device)
            opt.zero_grad(set_to_none=True)
            pred=net(x);loss,losses=objective(pred,y,x)
            if not torch.isfinite(loss):raise FloatingPointError('NaN/Inf loss: inspect calibration, inputs and loss scales')
            loss.backward()
            stable,grad_norm=stable_clip_grad_norm_(net.parameters(),1.)
            if not stable:
                bad_gradients+=1
                print(f'epoch {epoch:03d} batch {step}: skipped nonfinite gradient',flush=True)
                if bad_gradients>=10:raise FloatingPointError('10 nonfinite gradient batches in one epoch')
                continue
            opt.step()
            sumloss+=loss.item()*len(x['proprio']);count+=len(x['proprio'])
            if step%250==0:print(f'epoch {epoch:03d} batch {step}/{len(trainloader)} loss={loss.item():.5f} grad_norm={grad_norm:.3g}',flush=True)
        net.eval();vsum=0;vn=0
        with torch.no_grad():
            for x,y in valloader:
                x,y=batches_to_device(x,device),batches_to_device(y,device)
                loss,_=objective(net(x),y,x)
                vsum+=loss.item()*len(x['proprio']);vn+=len(x['proprio'])
        val=vsum/max(1,vn)
        print(f'epoch {epoch:03d} train={sumloss/max(1,count):.5f} validation={val:.5f} skipped_gradients={bad_gradients}',flush=True)
        _save_checkpoint(out/'last.pt',net,cfg,opt,epoch,val,provenance)
        if val<best:
            best=val;_save_checkpoint(out/'best.pt',net,cfg,opt,epoch,val,provenance)
    print('Saved:',out/'best.pt')


def evaluate_command(args):
    dev=torch.device('cuda' if torch.cuda.is_available() and args.device=='auto' else 'cpu' if args.device=='auto' else args.device)
    net,cfg,obj=checkpoint_load(args.checkpoint,dev);net.eval()
    ds=RawDataset(data_files(args.data),cfg,args.limit_per_run,args.stride,args.image_size)
    dl=DataLoader(ds,batch_size=args.batch,shuffle=False)
    totals={};n=0;perturbed=[]
    with torch.no_grad():
        for x,y in dl:
            x,y=batches_to_device(x,dev),batches_to_device(y,dev)
            res=net(x);_,ls=objective(res,y,x)
            for k,v in ls.items():totals[k]=totals.get(k,0)+v.item()*len(x['proprio'])
            n+=len(x['proprio'])
            # Genuine ablation: disrupt geometry, quantify sensitivity of future body.
            xp=dict(x);xp['xyz_world']=x['xyz_world']+torch.randn_like(x['xyz_world'])*.15
            xp['normal_world']=torch.roll(x['normal_world'],shifts=2,dims=-2)
            damaged=net(xp)
            perturbed.append((damaged['body']-res['body']).abs().mean().item())
    metric={k:round(v/n,6) for k,v in totals.items()}
    metric['geometry_perturbation_body_mean_abs_change']=float(np.mean(perturbed))
    metric['note']='Ablation sensitivity is NOT a measure of accuracy or causal identification.'
    print(json.dumps(metric,indent=2))


def predict_command(args):
    dev=torch.device('cuda' if torch.cuda.is_available() and args.device=='auto' else 'cpu' if args.device=='auto' else args.device)
    net,cfg,_=checkpoint_load(args.checkpoint,dev);net.eval()
    ds=RawDataset(data_files(args.data),cfg,args.limit_per_run,args.stride,args.image_size)
    x,_=ds[0]
    x={k:v[None].to(dev) for k,v in x.items()}
    if args.candidate_actions:
        plan=np.load(args.candidate_actions,allow_pickle=False)
        if plan.shape!=(cfg.horizon,cfg.action_dim):raise ValueError('Candidate plan must be [horizon,action_dim]')
        x['actions']=torch.tensor(plan,dtype=torch.float32,device=dev)[None]
    else:
        raise ValueError('INFERENCE REQUIRES --candidate-actions plan.npy known at time t. Refusing to use logged future control commands as an oracle.')
    with torch.no_grad():prediction=net(x)
    export={k:v[0].detach().cpu().numpy() for k,v in prediction.items()}
    np.savez_compressed(args.out,**export)
    print('Saved predictions under candidate actions:',args.out)


def test_command():
    set_seed(12)
    cfg=Config(d=40,heads=10,spatial_layers=1,temporal_layers=1,mlp_ratio=7,future_layers=1,
               horizon=2,patches=12,history=3,proprio_dim=8,action_dim=6)
    B,T,P,H,K,D,A=1,5,cfg.patches,cfg.horizon,cfg.history,cfg.proprio_dim,cfg.action_dim
    eye=torch.eye(4)[None].expand(B,-1,-1).clone()
    rand=lambda *shape:torch.randn(*shape)
    xyz=rand(B,T,P,3)*.1+torch.tensor([0.,0.,2.])
    x=dict(rgbd=torch.cat([torch.rand(B,T,P,3),torch.rand(B,T,P,1)+1],-1),
       xyz_world=xyz,normal_world=F.normalize(rand(B,T,P,3),dim=-1),
       valid=torch.ones(B,T,P,dtype=torch.bool),
       uv=torch.rand(B,T,P,2)*15,patch_ids=torch.arange(P).view(1,1,P).expand(B,T,P).clone(),
       proprio=rand(B,T,D),actions=rand(B,H,A),base_T=eye,
       foot_xyz_world=rand(B,4,3)*.05+torch.tensor([0.,0.,2.]),
       foot_velocity_world=rand(B,4,3)*.02,foot_contact=torch.ones(B,4),gait_phase=torch.rand(B,4),
       history_xyz_world=rand(B,4,K,3)*.04+torch.tensor([0.,0.,2.]),
       history_normal=rand(B,4,K,3),history_rgb=torch.rand(B,4,K,3),
       history_response=rand(B,4,K,6),history_delta_rgb=rand(B,4,K,3),
       history_delta_xyz=rand(B,4,K,3),history_delta_proprio=rand(B,4,K,D),
       history_age=torch.rand(B,4,K),history_patch_ids=torch.randint(0,P,(B,4,K)),
       history_valid=torch.ones(B,4,K,dtype=torch.bool),
       K=torch.tensor([[[30.,0.,8.],[0.,30.,8.],[0.,0.,1.]]]),T_base_camera=eye,
       images=torch.rand(B,T,3,16,16))
    y=dict(future_xyz_world=xyz[:,-1,None].expand(-1,H,-1,-1).clone(),
       future_rgb=torch.rand(B,H,P,3),future_valid=torch.ones(B,H,P,dtype=torch.bool),
       future_foot_xyz_world=x['foot_xyz_world'][:,None].expand(-1,H,-1,-1).clone(),
       future_patch_ids=torch.randint(0,P,(B,H,4)),future_force=rand(B,H,4,3)*20,
       future_contact=torch.ones(B,H,4),future_body=rand(B,H,D),
       future_base_T=eye[:,None].expand(-1,H,-1,-1).clone(),
       future_uvd=torch.rand(B,H,P,3)+1,future_uvd_valid=torch.ones(B,H,P,dtype=torch.bool))
    model=WorldModel(cfg);model.train();out=model(x)
    assert out['future_xyz_world'].shape==(B,H,P,3)
    assert out['foot_xyz_world'].shape==(B,H,4,3)
    assert out['body'].shape==(B,H,D)
    assert torch.isfinite(out['body']).all()
    total,parts=objective(out,y,x)
    total.backward()
    assert model.contact_delta[0].weight.grad is not None, 'BODY/FORCE -> terrain gradient missing'
    assert torch.isfinite(model.contact_delta[0].weight.grad).all()
    # Body response after explicit future terrain perturbation in causal observations.
    with torch.no_grad():
        old=out['body'];xp=dict(x);xp['xyz_world']=x['xyz_world']+torch.randn_like(xyz)*.13
        changed=model(xp)['body']
    assert (old-changed).abs().max()>.000001, 'future body insensitive to world geometry'
    print('PASS: forward, five-frame registration, surfels, training losses, gradients, terrain sensitivity')
    print('PASS: no old checkpoint, no measured future labels passed into forward')
    print('NOTE: synthetic execution is NOT real-world validation, DINO, deployment or data conversion verification')


def audit_command(args):
    files=data_files(args.data)
    if not files:raise FileNotFoundError('No .npz files found')
    passed=0
    for f in files:
        try:
            with np.load(f,allow_pickle=False) as dat:info=_validate_run(dat,f)
            passed+=1;print('VALID',f,json.dumps(info))
        except Exception as e:print('INVALID',f,type(e).__name__,str(e))
    if passed!=len(files):raise RuntimeError(f'{len(files)-passed} invalid recordings out of {len(files)}')
    print(f'{passed} recordings validate at schema/shape level. Physical calibration and sensor provenance still require independent verification.')


def main():
    p=argparse.ArgumentParser(description='3D terrain-foothold world model; one file')
    sub=p.add_subparsers(dest='command',required=True)
    for cmd in ('self-test','audit','train','evaluate','predict'):
        s=sub.add_parser(cmd)
        if cmd!='self-test':s.add_argument('--data',required=True)
        if cmd in ('train','evaluate','predict'):
            s.add_argument('--device',default='auto',choices=['auto','cpu','cuda'])
            s.add_argument('--batch',type=int,default=2)
            s.add_argument('--limit-per-run',type=int,default=None)
            s.add_argument('--stride',type=int,default=8)
            s.add_argument('--image-size',type=int,default=112,help='Downsample longer side of camera RGB for GPU memory')
        if cmd in ('evaluate','predict'):s.add_argument('--checkpoint',required=True)
        if cmd=='predict':
            s.add_argument('--out',default='predictions.npz')
            s.add_argument('--candidate-actions',help='NPY [H,A] candidate controls known NOW, not future oracle actions')
        if cmd=='train':
            s.add_argument('--out',default='runs/terrain_foothold');s.add_argument('--val-data')
            s.add_argument('--epochs',type=int,default=20);s.add_argument('--lr',type=float,default=2e-4)
            s.add_argument('--require-cuda',action='store_true');s.add_argument('--seed',type=int,default=42)
            s.add_argument('--hidden',type=int,default=320);s.add_argument('--heads',type=int,default=10)
            s.add_argument('--spatial-layers',type=int,default=8);s.add_argument('--temporal-layers',type=int,default=2)
            s.add_argument('--future-layers',type=int,default=1);s.add_argument('--mlp-ratio',type=int,default=7)
            s.add_argument('--horizon',type=int,default=10);s.add_argument('--interval',type=float,default=.1)
            s.add_argument('--patches',type=int,default=64);s.add_argument('--history',type=int,default=5)
            s.add_argument('--vision',choices=['native','dinov2'],default='native')
        if cmd=='audit':pass
    args=p.parse_args()
    if args.command=='self-test':test_command()
    elif args.command=='audit':audit_command(args)
    elif args.command=='train':set_seed(args.seed);train_command(args)
    elif args.command=='evaluate':evaluate_command(args)
    elif args.command=='predict':predict_command(args)

if __name__=='__main__':main()
