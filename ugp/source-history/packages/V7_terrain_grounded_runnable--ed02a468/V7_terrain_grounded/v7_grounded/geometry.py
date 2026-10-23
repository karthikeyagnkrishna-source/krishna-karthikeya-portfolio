"""Calibrated geometry helpers. Coordinates: +Z up in world, camera +Z forward.
An RGB-D observation is NEVER supplied from the future during inference.
"""
from __future__ import annotations
import torch


def euler_to_rotation(rpy: torch.Tensor) -> torch.Tensor:
    """[... ,3] roll/pitch/yaw (rad) -> [...,3,3] rotation Rz(yaw)Ry(pitch)Rx(roll)."""
    r,p,y = rpy.unbind(-1)
    cr,sr,cp,sp,cy,sy = r.cos(),r.sin(),p.cos(),p.sin(),y.cos(),y.sin()
    a = torch.stack([cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],-1)
    b = torch.stack([sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],-1)
    c = torch.stack([-sp,cp*sr,cp*cr],-1)
    return torch.stack([a,b,c],-2)


def world_to_robot(xyz_world: torch.Tensor, base_pose: torch.Tensor) -> torch.Tensor:
    """xyz [B,P,3] or [B,L,3]; base pose [B,6] (xyz + rpy)."""
    R = euler_to_rotation(base_pose[:,3:])
    return torch.einsum('bpi,bij->bpj',xyz_world-base_pose[:,None,:3],R)


def project_future_patches(xyz_world: torch.Tensor, base_pose: torch.Tensor,
                           camera_from_base: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Differentiable sparse pinhole reprojection: return [B,P,(u,v,depth)].

    camera_from_base transforms robot-frame xyz to camera-frame xyz. All inputs are
    calibrated measured quantities except xyz_world and base_pose, which are predictions.
    """
    base_xyz = world_to_robot(xyz_world,base_pose)
    cam_xyz = torch.einsum('bij,bpj->bpi',camera_from_base[:,:3,:3],base_xyz) + camera_from_base[:,None,:3,3]
    z = cam_xyz[...,2]
    zn = z.clamp(min=1e-3)
    u = intrinsics[:,None,0,0]*cam_xyz[...,0]/zn + intrinsics[:,None,0,2]
    v = intrinsics[:,None,1,1]*cam_xyz[...,1]/zn + intrinsics[:,None,1,2]
    return torch.stack([u,v,z],-1)


def grid_from_rgbd(rgb: torch.Tensor, depth: torch.Tensor, intrinsics: torch.Tensor,
                   camera_to_world: torch.Tensor, *, origin_xy: tuple[float,float],
                   n_side: int=8, cell_m: float=0.25, min_depth:float=.15,
                   max_depth:float=12.) -> dict[str,torch.Tensor]:
    """World-register FIVE (or more) calibrated RGB-D frames to persistent XY cells.

    rgb: [T,3,H,W], depth [T,H,W] in METERS, K [3,3], camera_to_world
    [T,4,4]. origin_xy is a world-anchored lower grid corner; keep it fixed over
    observation and target frames. Returned patch IDs encode WORLD XY cells and are
    stable through egomotion. Absent patches stay masked; do not invent visibility.
    This is a height-field representation (one vertical surface per XY cell).
    """
    if rgb.ndim!=4 or rgb.shape[1]!=3 or depth.shape!=rgb.shape[:1]+rgb.shape[-2:]:
        raise ValueError('rgb [T,3,H,W], depth [T,H,W] expected')
    T,_,H,W=rgb.shape
    device=depth.device; dt=depth.dtype; P=n_side*n_side
    oy,ox=torch.meshgrid(torch.arange(n_side,device=device),torch.arange(n_side,device=device),indexing='ij')
    cx=torch.as_tensor(origin_xy[0],device=device,dtype=dt)+(ox.reshape(-1)+.5)*cell_m
    cy=torch.as_tensor(origin_xy[1],device=device,dtype=dt)+(oy.reshape(-1)+.5)*cell_m
    base_xyz=torch.stack([cx,cy,torch.zeros_like(cx)],-1)
    # Stable absolute (rather than camera-indexed) world-cell identifier.
    gix=torch.floor(cx/cell_m).long(); giy=torch.floor(cy/cell_m).long()
    ids=(gix+1_000_000)*2_000_001+(giy+1_000_000)
    vv,uu=torch.meshgrid(torch.arange(H,device=device,dtype=dt),torch.arange(W,device=device,dtype=dt),indexing='ij')
    cam_rays=torch.stack([(uu-intrinsics[0,2])/intrinsics[0,0],(vv-intrinsics[1,2])/intrinsics[1,1],torch.ones_like(uu)],-1).reshape(-1,3)
    out_rgb=[];out_xyz=[];out_valid=[]
    for k in range(T):
        z=depth[k].reshape(-1); valid=torch.isfinite(z)&(z>min_depth)&(z<max_depth)
        cam=cam_rays*z[:,None]
        xyz=cam@camera_to_world[k,:3,:3].T+camera_to_world[k,:3,3]
        ix=torch.floor((xyz[:,0]-origin_xy[0])/cell_m).long()
        iy=torch.floor((xyz[:,1]-origin_xy[1])/cell_m).long()
        inside=valid&(ix>=0)&(ix<n_side)&(iy>=0)&(iy<n_side)
        cell=(iy*n_side+ix)[inside]
        colors=rgb[k].permute(1,2,0).reshape(-1,3)[inside]
        sums=torch.zeros(P,7,device=device,dtype=dt)
        counts=torch.zeros(P,device=device,dtype=dt)
        val=torch.cat([colors,xyz[inside],z[inside,None]],-1)
        sums.index_add_(0,cell,val)
        counts.index_add_(0,cell,torch.ones_like(cell,dtype=dt))
        means=sums/counts.clamp(min=1)[:,None]
        mask=counts>0
        out_rgb.append(torch.cat([means[:,:3],means[:,6:7]],-1))
        out_xyz.append(torch.where(mask[:,None],means[:,3:6],base_xyz))
        out_valid.append(mask)
    return dict(rgbd=torch.stack(out_rgb),xyz_world=torch.stack(out_xyz),
                valid=torch.stack(out_valid),patch_ids=ids[None].expand(T,-1).clone())
