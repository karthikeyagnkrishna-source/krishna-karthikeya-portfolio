"""SmoothL1 with the approved tolerance and worst-tail penalties."""
import math
import torch
import torch.nn.functional as F
from contracts import ToleranceConfig

def scalar_tolerance(target,kind,cfg=ToleranceConfig()):
    y=target.double()
    factor=180/math.pi if kind=='angle' else 1.
    reported=y*factor
    if kind=='contact': return torch.full_like(y,cfg.contact)
    return torch.where(reported.abs()<=cfg.zero_threshold,
                       torch.full_like(y,getattr(cfg,kind)/factor),cfg.relative*y.abs())

def tolerance_penalty(error,tolerance):
    if error.numel()==0: return error.sum()
    b=torch.log1p(torch.relu(error.double()/tolerance.double()-1)).flatten()
    return .2*b.mean()+.5*b.topk(max(1,math.ceil(.1*b.numel()))).values.mean()

def specialist_loss(predicted,target,kind,scale,cfg=ToleranceConfig(),warmup=False):
    if predicted.numel()==0: return predicted.sum()
    if not torch.isfinite(predicted).all() or not torch.isfinite(target).all():
        raise ValueError('Nonfinite specialist value')
    loss=F.smooth_l1_loss(predicted.float()/scale.float(),target.float()/scale.float())
    return loss if warmup else loss+tolerance_penalty((predicted.double()-target.double()).abs(),scalar_tolerance(target,kind,cfg))

def stage_loss(predicted,targets,scales,stage,fk,cfg=ToleranceConfig(),warmup=False):
    if stage=='force':
        loss=specialist_loss(predicted.force,targets.force,'force',scales['loss_force_std'],cfg,warmup)
        error=(predicted.force.double()-targets.force.double()).norm(dim=-1)
        magnitude=targets.force.double().norm(dim=-1)
        tolerance=torch.where(magnitude<=cfg.zero_threshold,torch.full_like(magnitude,cfg.force),cfg.relative*magnitude)
        vector=F.smooth_l1_loss(error/scales['loss_force_std'].double().norm(dim=-1),torch.zeros_like(error))
        if not warmup:vector=vector+tolerance_penalty(error,tolerance)
        contact=F.binary_cross_entropy(predicted.contact.clamp(1e-7,1-1e-7),targets.contact.float())
        if not warmup:contact=contact+tolerance_penalty((predicted.contact.double()-targets.contact.double()).abs(),scalar_tolerance(targets.contact,'contact',cfg))
        return loss+.5*vector+contact
    name=stage.replace('control_','').replace('retune_','');j=int(name[-1]);kind,offset={'Q':('angle',0),'D':('speed',3),'T':('torque',6)}[name[0]]
    loss=specialist_loss(predicted.leg[...,offset+j],targets.leg[...,offset+j],kind,scales['loss_leg_std'][:,offset+j],cfg,warmup)
    if kind=='angle':
        with torch.no_grad():true=fk(targets.leg[...,:3])[0]
        loss=loss+.2*F.smooth_l1_loss(predicted.r/scales['r_std'],true/scales['r_std'])
    return loss
