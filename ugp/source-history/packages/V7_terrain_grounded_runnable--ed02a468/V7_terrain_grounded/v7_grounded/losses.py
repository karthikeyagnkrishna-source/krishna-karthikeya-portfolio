"""Multi-horizon REAL supervision. Predicted footholds are never used as labels.
Missing supervision is an error instead of silently creating fake ground truth.
"""
from __future__ import annotations
import torch
from torch.nn import functional as F

MANDATORY_TARGETS=('future_xyz_world','future_rgb','future_foot_xyz_world',
                   'future_patch_ids','future_force','future_contact','future_body','future_base_pose',
                   'future_patch_valid')


def _masked(err:torch.Tensor,valid:torch.Tensor)->torch.Tensor:
    valid=valid.bool()
    while valid.ndim<err.ndim:valid=valid[...,None]
    valid=valid.expand_as(err)
    return (err*valid.to(err.dtype)).sum()/valid.sum().clamp(min=1)


def grounded_loss(out:dict[str,torch.Tensor],target:dict[str,torch.Tensor]):
    for k in MANDATORY_TARGETS:
        if k not in target:raise KeyError(f'Missing measured target {k}; refusing pseudo-ground-truth training')
    valid=target['future_patch_valid'].bool() & out['patch_valid'][:,None,:]
    contact=target['future_contact'].bool()
    geom=_masked(F.smooth_l1_loss(out['future_xyz_world'],target['future_xyz_world'],reduction='none'),valid)
    anchor=_masked(F.smooth_l1_loss(out['direct_xyz_world'],target['future_xyz_world'],reduction='none'),valid)
    rgb=_masked(F.smooth_l1_loss(out['future_rgb'],target['future_rgb'],reduction='none'),valid)
    anchor_rgb=_masked(F.smooth_l1_loss(out['direct_rgb'],target['future_rgb'],reduction='none'),valid)
    consistency=_masked((out['future_xyz_world']-out['direct_xyz_world']).square(),valid)
    # Optional calibrated re-projection: future observations provide measured UV/depth labels.
    reproj=geom.new_zeros(())
    if 'future_uvd' in target and 'projected_uvd' in out:
        uvd=out['projected_uvd'];gt=target['future_uvd']
        camera_valid=valid & (gt[...,2]>.01) & (uvd[...,2]>.01)
        reproj=_masked(F.smooth_l1_loss(torch.cat([uvd[...,:2]/500,uvd[...,2:]],-1),
                                             torch.cat([gt[...,:2]/500,gt[...,2:]],-1),reduction='none'),camera_valid)
    # Per-leg categorical ID supervision includes only IDs that exist in the fixed world grid.
    ids=out['patch_ids'][:,None,None,:]
    truth=target['future_patch_ids']
    match=ids==truth[...,None]
    idx=match.long().argmax(-1)
    present=match.any(-1)&contact
    logits=out['patch_logits']
    patch_ce=_masked(F.cross_entropy(logits.flatten(0,2),idx.flatten(),reduction='none').reshape_as(idx),present)
    foot=_masked(F.smooth_l1_loss(out['foot_xyz_world'],target['future_foot_xyz_world'],reduction='none'),contact)
    force=_masked(F.smooth_l1_loss(out['force'],target['future_force'],reduction='none'),contact)
    classification=F.binary_cross_entropy_with_logits(out['contact_logits'],target['future_contact'].float())
    body=F.smooth_l1_loss(out['body'],target['future_body'])
    pose=F.smooth_l1_loss(out['base_pose'],target['future_base_pose'])
    # Heteroscedastic Gaussian NLL calibrates reported geometry/force uncertainty.
    geo_lvar=out['geometry_log_sigma']
    geo_sq=(out['future_xyz_world']-target['future_xyz_world']).square().mean(-1)
    geo_nll=_masked(.5*(geo_sq*torch.exp(-geo_lvar)+geo_lvar),valid)
    force_lvar=out['force_log_sigma']
    force_sq=(out['force']-target['future_force']).square()
    force_nll=_masked(.5*(force_sq*torch.exp(-force_lvar)+force_lvar),contact)
    # Observed patch geometry/color + footholds MUST be supervised, never just latent MSE.
    losses=dict(geometry=geom,direct_geometry=anchor,rgb=rgb,direct_rgb=anchor_rgb,
                geometric_consistency=consistency,reprojection=reproj,landing_cell=patch_ce,
                foothold=foot,force=force,contact=classification,body=body,egomotion=pose,
                geometry_nll=geo_nll,force_nll=force_nll)
    weights=dict(geometry=4,direct_geometry=2,rgb=1,direct_rgb=.5,
                 geometric_consistency=.3,reprojection=1,landing_cell=1.5,foothold=4,
                 force=2,contact=1,body=2,egomotion=2,geometry_nll=.05,force_nll=.05)
    total=sum(weights[k]*v for k,v in losses.items())
    return total,losses
