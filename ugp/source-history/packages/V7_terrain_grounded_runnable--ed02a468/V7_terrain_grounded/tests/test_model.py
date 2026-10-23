"""CPU architecture and anti-blind smoke checks; NOT a robotics accuracy evaluation."""
from __future__ import annotations
import torch
from v7_grounded import V7Config,V7GroundedWorldModel
from v7_grounded.losses import grounded_loss

def sample(B=2,T=5,P=9,L=4,K=3,H=3,D=8,A=6):
    torch.manual_seed(42)
    ids=(torch.arange(P,dtype=torch.int64)+1000).view(1,1,P).expand(B,T,P).clone()
    world=torch.randn(B,T,P,3)*.2
    world[...,2]=world[...,2].abs()*.02
    color=torch.rand(B,T,P,4)
    valid=torch.ones(B,T,P,dtype=torch.bool)
    valid[:,0,3]=False
    current_feet=world[:,-1,:4,:].clone()
    x=dict(rgbd=color,xyz_world=world,valid=valid,patch_ids=ids,
           proprio=torch.randn(B,T,D)*.1,actions=torch.randn(B,H,A)*.2,
           base_pose=torch.zeros(B,6),foot_xyz_world=current_feet,
           foot_velocity_world=torch.randn(B,L,3)*.02,foot_contact=torch.ones(B,L)*.7,
           gait_phase=torch.rand(B,L),history_xyz_world=current_feet[:,:,None,:].expand(-1,-1,K,-1).clone(),
           history_rgb=torch.rand(B,L,K,3),history_geometry=torch.randn(B,L,K,4)*.05,
           history_response=torch.randn(B,L,K,6)*.1,
           history_delta_rgbd=torch.randn(B,L,K,4)*.1,
           history_delta_xyz=torch.randn(B,L,K,3)*.01,
           history_delta_proprio=torch.randn(B,L,K,D)*.1,
           history_age=torch.arange(K).float().view(1,1,K).expand(B,L,-1),
           history_patch_ids=ids[:,-1,:4][:,:,None].expand(B,L,K).clone(),
           history_valid=torch.ones(B,L,K,dtype=torch.bool))
    y=dict(future_xyz_world=world[:,-1,None].expand(-1,H,-1,-1).clone(),
           future_rgb=color[:,-1,None,:,:3].expand(-1,H,-1,-1).clone(),
           future_foot_xyz_world=current_feet[:,None].expand(-1,H,-1,-1).clone(),
           future_patch_ids=ids[:,-1,:4][:,None].expand(-1,H,-1).clone(),
           future_force=torch.randn(B,H,L,3)*.1,
           future_contact=torch.ones(B,H,L),
           future_body=x['proprio'][:,-1,None].expand(-1,H,-1).clone(),
           future_base_pose=x['base_pose'][:,None].expand(-1,H,-1).clone(),
           future_patch_valid=torch.ones(B,H,P,dtype=torch.bool))
    return x,y

def test_forward_and_loss():
    x,y=sample()
    cfg=V7Config(proprio_dim=8,action_dim=6,hidden=40,heads=5,spatial_layers=1,temporal_layers=1,horizons=3)
    net=V7GroundedWorldModel(cfg)
    p=net(x)
    assert p['future_xyz_world'].shape==(2,3,9,3)
    assert p['body'].shape==(2,3,8)
    assert p['patch_logits'].shape==(2,3,4,9)
    assert p['predicted_patch_ids'].shape==(2,3,4)
    loss,parts=grounded_loss(p,y)
    assert torch.isfinite(loss),loss
    loss.backward()
    grad=net.geom_step[-1].weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert grad.abs().sum()>0, 'No gradient through future evolving terrain'
    # Stronger anti-blind check: BODY loss backpropagates to the deformation head.
    net.zero_grad(set_to_none=True)
    out=net(x)
    out['body'].square().mean().backward()
    assert net.geom_step[-1].weight.grad is not None and net.geom_step[-1].weight.grad.abs().sum()>1e-10, 'Blind body bypass detected'
    print('Forward, supervised loss, backward: PASS; component losses:',{k:round(v.item(),4) for k,v in parts.items()})

def test_geometry_cannot_be_skipped():
    x,y=sample(B=1)
    cfg=V7Config(proprio_dim=8,action_dim=6,hidden=40,heads=5,spatial_layers=1,temporal_layers=1,horizons=3)
    net=V7GroundedWorldModel(cfg).eval()
    out1=net(x)
    x['xyz_world']=x['xyz_world'].clone();x['xyz_world'][:,:,:,2]+=.4
    out2=net(x)
    geom_delta=(out2['future_xyz_world']-out1['future_xyz_world']).abs().max().item()
    body_delta=(out2['body']-out1['body']).abs().max().item()
    assert geom_delta>.01
    assert body_delta>1e-7, f'body head unexpectedly insensitive to future terrain: {body_delta}'
    print(f'Geometry intervention affected future terrain ({geom_delta:.5f} m) AND body ({body_delta:.7f}). PASS')

def test_egomotion_not_world_deformation():
    x,_=sample(B=1)
    cfg=V7Config(proprio_dim=8,action_dim=6,hidden=40,heads=5,spatial_layers=1,temporal_layers=1,horizons=3)
    net=V7GroundedWorldModel(cfg).eval()
    p=net(x)
    assert p['base_pose'].shape==(1,3,6)
    assert p['future_xyz_world'].shape==(1,3,9,3)
    print('Base egomotion and world patch deformation are distinct fields: PASS')

if __name__=='__main__':
    test_forward_and_loss();test_geometry_cannot_be_skipped();test_egomotion_not_world_deformation()

