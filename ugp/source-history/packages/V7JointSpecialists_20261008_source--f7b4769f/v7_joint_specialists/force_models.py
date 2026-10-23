"""Unbounded force residual using frozen corrected joints and seed body."""
import copy
import torch
from torch import nn
from contracts import Predictions
from models import zero_last

class ForceExtension(nn.Module):
    def __init__(self,prior):
        super().__init__();self.refiner=copy.deepcopy(prior.force_refiner)
        zero_last(self.refiner.output)
        self.representation=nn.Linear(9*256,256)
        with torch.no_grad():
            self.representation.weight.copy_(torch.cat([torch.eye(256)/9]*9,1));self.representation.bias.zero_()
        self.contact=nn.Linear(256,1);nn.init.zeros_(self.contact.weight);nn.init.zeros_(self.contact.bias)

    def forward(self,batch,scales,state,hidden,body_hidden,prior_force):
        f=self.refiner;b,h,l=state.leg.shape[:3]
        lh=self.representation(hidden.flatten(-2))
        identity=f.identity(torch.arange(l,device=state.leg.device)).view(1,1,l,16).expand(b,h,-1,-1)
        body=((state.body-scales['body_mean'])/scales['body_std'])[:,:,None].expand(-1,-1,l,-1)
        bh=body_hidden[:,:,None].expand(-1,-1,l,-1)
        horizon=batch['horizon'].view(1,h,1,1).expand(b,-1,l,-1)
        features=torch.cat([batch['x'].float(),lh,bh,(state.leg-scales['leg_mean'])/scales['leg_std'],body,
            (state.r-scales['r_mean'])/scales['r_std'],(state.fv-scales['fv_mean'])/scales['fv_std'],
            (prior_force-scales['target_force_mean'])/scales['target_force_std'],batch['contact'][...,None],identity,horizon],-1)
        token=f.projection(features)
        token=torch.cat([f.body_query(body_hidden)[:,:,None],token],2).reshape(b*h,l+1,256)
        token=f.leg_mixer(token)[:,1:].reshape(b,h,l,256)
        causal=torch.ones(h,h,device=token.device,dtype=torch.bool).triu(1)
        token=f.temporal(token.permute(0,2,1,3).reshape(b*l,h,256),mask=causal).reshape(b,l,h,256).permute(0,2,1,3)
        force=prior_force+f.output(token)*scales['target_force_std']
        contact=(torch.logit(batch['contact'].float().clamp(1e-6,1-1e-6))+self.contact(token).squeeze(-1)).sigmoid()
        return force,contact

class ForceContactSpecialist(nn.Module):
    def __init__(self,joints,force_prior):
        super().__init__();self.joints=joints;self.joints.configure(None)
        self.force_prior=copy.deepcopy(force_prior).eval().requires_grad_(False)
        self.extension=ForceExtension(force_prior);self.stage=None

    @property
    def refiner(self):return self.extension.refiner
    @property
    def fk(self):return self.joints.fk
    def configure_force(self):
        self.zero_grad(set_to_none=True)
        self.requires_grad_(False);self.extension.requires_grad_(True);self.stage='force';self.train(self.training)
    def active_parameters(self):return [p for p in self.extension.parameters() if p.requires_grad]
    def train(self,mode=True):
        super().train(mode);self.joints.eval();self.force_prior.eval();return self
    def forward(self,inputs,scales):
        with torch.no_grad():
            state=self.joints(inputs,scales)
            prior=self.force_prior(inputs.base,scales,'one_pass')['force']
            _,_,bh=self.joints.seed_state(inputs,scales)
        force,contact=self.extension(inputs.base,scales,state,state.hidden,bh,prior)
        return Predictions(state.leg,force,state.body,contact,state.r,state.hidden,state.fv)
