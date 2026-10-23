"""Independent per-joint warm starts with causal command/history adapters."""
import copy
import contextlib
import torch
from torch import nn
from contracts import Predictions
from joint_data import normalized_inputs
from feedback_models import leg_prediction, body_prediction

FAMILIES={'Q':('angle',0),'D':('speed',3),'T':('torque',6)}

class SequenceEncoder(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.project=nn.Linear(width,64)
        layer=nn.TransformerEncoderLayer(64,4,256,.1,batch_first=True,norm_first=True)
        self.temporal=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.query=nn.Linear(1,64)

    def encode(self,x,valid):
        b,l,s,_=x.shape
        v=valid.reshape(b*l,s);safe=v.clone();safe[:,0]=True
        z=self.project(x).reshape(b*l,s,64)
        causal=torch.ones(s,s,device=x.device,dtype=torch.bool).triu(1)
        z=self.temporal(z,mask=causal,src_key_padding_mask=~safe)
        return z.reshape(b,l,s,64)*valid[...,None]

    def forward(self,x,valid,horizons,prefix=None):
        z=self.encode(x,valid);b,l,s,_=z.shape;h=len(horizons)
        allowed=valid[:,None].expand(-1,h,-1,-1) if prefix is None else prefix
        q=self.query(horizons[:,None]).view(1,h,1,1,64)
        score=(q*z[:,None]).sum(-1)/8
        # A valid zero sentinel handles empty prefixes without a NaN softmax.
        empty=~allowed.any(-1,keepdim=True)
        scores=torch.cat([score.masked_fill(~allowed,-torch.inf),torch.zeros_like(score[...,:1]).masked_fill(~empty,-torch.inf)],-1)
        weights=scores.softmax(-1)[...,:s]
        return (weights[...,None]*z[:,None]).sum(-2)

class JointSpecialist(nn.Module):
    def __init__(self,parent_leg_head,kind,joint,output_only=False):
        super().__init__();self.kind=kind;self.joint=joint;self.output_only=output_only
        family={'angle':0,'speed':1,'torque':2}[kind]
        self.identity=copy.deepcopy(parent_leg_head.identity)
        self.projection=copy.deepcopy(parent_leg_head.projection)
        self.temporal=copy.deepcopy(parent_leg_head.temporal)
        original=parent_leg_head.outputs[family]
        self.output=nn.Sequential(copy.deepcopy(original[0]),nn.GELU(),nn.Linear(128,1))
        with torch.no_grad():
            self.output[-1].weight.copy_(original[-1].weight[joint:joint+1]);self.output[-1].bias.copy_(original[-1].bias[joint:joint+1])
        self.history_encoder=SequenceEncoder(83);self.command_encoder=SequenceEncoder(55)
        self.history_adapter=nn.Linear(64,256,bias=False);self.command_adapter=nn.Linear(64,256,bias=False)
        self.condition_adapter=nn.Linear(11,256,bias=False)
        for m in (self.history_adapter,self.command_adapter,self.condition_adapter):nn.init.zeros_(m.weight)

    def forward(self,inputs,scales,history,commands,predicted,seed):
        batch=inputs.base;x=batch['x'].float();b,h,l,_=x.shape
        identity=self.identity(torch.arange(l,device=x.device)).view(1,1,l,16).expand(b,h,-1,-1)
        base=(batch['base_leg'].float()-scales['leg_mean'])/scales['leg_std']
        fn=(batch['force'].float()-scales['force_mean'])/scales['force_std']
        horizon=batch['horizon'].view(1,h,1,1).expand(b,-1,l,-1)
        y=self.projection(torch.cat([x,identity,base,fn,horizon],-1))
        y=self.temporal(y.permute(0,2,1,3).reshape(b*l,h,256)).reshape(b,l,h,256).permute(0,2,1,3)
        if not self.output_only:
            hy=self.history_encoder(history,inputs.history_valid,batch['horizon'])
            co=self.command_encoder(commands,inputs.command_valid.any(1),batch['horizon'],inputs.command_valid)
            y=y+self.history_adapter(hy)+self.command_adapter(co)
            if self.kind=='torque':
                # Candidate is last valid planned torque, not measured future effort.
                valid=inputs.command_valid
                ix=torch.arange(valid.shape[-1],device=x.device).view(1,1,1,-1)
                last=torch.where(valid,ix,-1).amax(-1)
                raw=inputs.commands[...,self.joint*16]
                candidate=raw[:,None].expand(-1,h,-1,-1).gather(-1,last.clamp_min(0)[...,None]).squeeze(-1)
                candidate=torch.where(last>=0,candidate,0)
                cond=torch.cat([(predicted[...,:6].detach()-scales['leg_mean'][:6])/scales['leg_std'][:6],
                    ((seed[...,6+self.joint:self.joint+7]-scales['leg_mean'][6+self.joint])/scales['leg_std'][6+self.joint]),
                    fn,(candidate/scales['leg_std'][6+self.joint])[...,None]],-1)
                y=y+self.condition_adapter(cond)
        channel={'angle':0,'speed':3,'torque':6}[self.kind]+self.joint
        return batch['base_leg'][...,channel].float()+self.output(y).squeeze(-1)*scales['leg_std'][channel],y

class JointSpecialists(nn.Module):
    def __init__(self,seed):
        super().__init__();self.seed=copy.deepcopy(seed).eval().requires_grad_(False)
        self.heads=nn.ModuleDict({f'{f}{j}':JointSpecialist(seed.legs,kind,j) for f,(kind,_) in FAMILIES.items() for j in range(3)})
        self.controls=nn.ModuleDict({f'T{j}':JointSpecialist(seed.legs,'torque',j,True) for j in range(3)})
        self.stage=None

    @property
    def fk(self):return self.seed.fk

    def configure(self,stage):
        self.zero_grad(set_to_none=True)
        self.stage=stage;self.requires_grad_(False)
        if stage is not None:
            if stage.startswith('control_'):self.controls[stage.split('_')[-1]].output.requires_grad_(True)
            else:self.heads[stage.replace('retune_','')].requires_grad_(True)
        self.train(self.training)

    def active_parameters(self):return [p for p in self.parameters() if p.requires_grad]

    def train(self,mode=True):
        super().train(mode);self.seed.eval()
        for name,head in self.heads.items():head.train(mode and self.stage in (name,'retune_'+name))
        for name,head in self.controls.items():
            head.eval()
            if mode and self.stage=='control_'+name:head.output.train()
        return self

    def seed_state(self,inputs,scales):
        with torch.no_grad():
            leg,lh=leg_prediction(self.seed.legs,inputs.base,scales,inputs.base['force'])
            r,j=self.fk(leg[...,:3]);fv=(j@leg[...,3:6,None]).squeeze(-1)
            body,bh=body_prediction(self.seed.body,inputs.base,scales,leg,inputs.base['force'],r,fv)
        return dict(leg=leg,body=body,r=r,fv=fv),lh,bh

    def forward(self,inputs,scales,variant='joint_specialists'):
        if any(k.startswith('target') for k in inputs.base):raise ValueError('Targets forbidden in InputBatch.base')
        seed,_,_=self.seed_state(inputs,scales)
        history,commands=normalized_inputs(inputs,scales)
        values=[];hidden=[]
        for f in ('Q','D','T'):
            predicted=torch.stack(values,-1) if len(values)==6 else seed['leg']
            for j in range(3):
                name=f'{f}{j}';use_control=variant=='torque_control' and f=='T'
                if variant in ('seed','torque_control','torque_only') and f!='T' or variant=='seed':
                    values.append(seed['leg'][...,FAMILIES[f][1]+j]);hidden.append(torch.zeros(*seed['leg'].shape[:-1],256,device=seed['leg'].device));continue
                head=self.controls[name] if use_control else self.heads[name]
                active=self.stage==('control_'+name if use_control else name) or self.stage=='retune_'+name
                with contextlib.nullcontext() if active else torch.no_grad():
                    value,feature=head(inputs,scales,history,commands,predicted,seed['leg'])
                values.append(value);hidden.append(feature)
        leg=torch.stack(values,-1);r,j=self.fk(leg[...,:3]);fv=(j@leg[...,3:6,None]).squeeze(-1)
        return Predictions(leg,inputs.base['force'].float(),seed['body'],inputs.base['contact'].float(),r,torch.stack(hidden,-2),fv)
