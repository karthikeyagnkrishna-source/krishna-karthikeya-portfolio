"""Lazy windows over immutable parent latents and compact raw commands."""
import json
from pathlib import Path
import numpy as np
import torch
from contracts import InputBatch, Targets
from command_data import check_targets, window_arrays
from foundation import sha256

BASE_KEYS=('x','base_leg','base_body','force','contact')

class JointDataset(torch.utils.data.Dataset):
    def __init__(self,parent,feedback,command_root,branch,split):
        self.branch=branch;self.context=.5 if branch=='short' else 4.
        self.horizons=torch.tensor([.1] if branch=='short' else [i/10 for i in range(2,11)])
        self.parts=[];self.indices=[];self.trajectories=[];self.terrains=[];self.transitions=[];self.signatures={}
        manifest=json.loads((Path(parent)/f'{branch}_manifest.json').read_text())
        cache=Path(feedback)/'cache'/branch;command_root=Path(command_root)
        for ti,meta in enumerate(manifest['trajectories']):
            name=f"{meta['environment']}_{meta['run']}"
            path=cache/f'{name}.pt';packet=torch.load(path,map_location='cpu',weights_only=False);t=packet['tensors']
            rawpath=command_root/f'{name}_physics.npz'
            with np.load(rawpath) as z:p={k:z[k] for k in z.files}
            with np.load(command_root/f'{name}_commands.npz') as z:c={k:z[k] for k in z.files}
            old=torch.load(Path(parent)/'cache'/branch/f'{name}.pt',map_location='cpu',weights_only=False)['tensors']
            if not torch.equal(t['train'],old['train']) or not torch.equal(t['val'],old['val']):raise ValueError('Temporal masks changed')
            for i in range(len(t['x'])):
                check_targets(p,float(t['anchor'][i]),t['target_time'][i].numpy(),t['target_leg'][i].numpy(),t['target_force'][i].numpy())
            if not torch.allclose(t['target_time']-t['anchor'][:,None],self.horizons.double(),atol=1e-7,rtol=0): raise ValueError('Horizon target times changed')
            selected=torch.nonzero(t['train' if split=='train' else 'val']).flatten().tolist()
            self.parts.append((name,t,p,c))
            for i in selected:
                self.indices.append((ti,i));self.trajectories.append(ti);self.terrains.append(meta['terrain'])
                a=float(t['anchor'][i]);lo=np.searchsorted(p['time'],a-self.context);hi=np.searchsorted(p['time'],a+self.horizons[-1].item())
                trans=bool(np.any(np.diff(p['contact'][lo:hi],axis=0)!=0) or np.any(abs(np.diff(p['L_LEG'][lo:hi],axis=0))>5))
                self.transitions.append(trans)
            self.signatures[name]=dict(cache=sha256(path),parent=sha256(Path(parent)/'cache'/branch/f'{name}.pt'),
                                       raw=sha256(rawpath),commands=sha256(command_root/f'{name}_commands.npz'))
        if len(self.parts)!=28 or not self.indices:raise ValueError('Requires 28 trajectories and nonempty split')
        self.trajectories=np.asarray(self.trajectories);self.transitions=np.asarray(self.transitions)

    def __len__(self):return len(self.indices)
    def __getitem__(self,j):
        ti,i=self.indices[j];name,t,p,c=self.parts[ti]
        base={k:t[k][i] for k in BASE_KEYS};base['horizon']=self.horizons
        arrays=window_arrays(p,c,float(t['anchor'][i]),t['target_time'][i].numpy(),self.context)
        inp=InputBatch(base,*(torch.from_numpy(a) for a in arrays))
        target=Targets(t['target_leg'][i],t['target_force'][i],t['target_body'][i],t['target_contact'][i],t['target_time'][i],(name,i))
        return inp,target

def collate_windows(items):
    ins,targets=zip(*items)
    def pad(values,axis):
        n=max(v.shape[axis] for v in values);out=[]
        for v in values:
            shape=list(v.shape);shape[axis]=n
            z=torch.zeros(shape,dtype=v.dtype);sl=[slice(None)]*v.ndim;sl[axis]=slice(0,v.shape[axis]);z[tuple(sl)]=v;out.append(z)
        return torch.stack(out)
    base={k:ins[0].base[k] if k=='horizon' else torch.stack([x.base[k] for x in ins]) for k in ins[0].base}
    inp=InputBatch(base,torch.stack([x.history for x in ins]),torch.stack([x.history_valid for x in ins]),
                   pad([x.commands for x in ins],1),pad([x.command_valid for x in ins],2))
    tar=Targets(*(torch.stack([getattr(t,k) for t in targets]) for k in ('leg','force','body','contact','times')),[t.rows for t in targets])
    return inp,tar

def fit_input_scales(dataset):
    if any(dataset.parts[ti][1]['train'][i].item()==False for ti,i in dataset.indices):raise ValueError('Input scales require optimization windows only')
    stats={k:[torch.zeros(n,dtype=torch.float64),torch.zeros(n,dtype=torch.float64),torch.zeros(n,dtype=torch.float64)]
           for k,n in [('history',83),('commands',55)]}
    leg=[];force=[]
    for j in range(len(dataset)):
        inp,t=dataset[j];leg.append(t.leg);force.append(t.force)
        for name,x,valid in [('history',inp.history,inp.history_valid),('commands',inp.commands,inp.command_valid.any(0))]:
            x=x.double();mask=valid[...,None].expand_as(x).clone()
            if name=='commands':
                for ji in range(3):mask[...,ji*16:ji*16+6]&=x[...,ji*16+6:ji*16+12].bool()
            else:
                for ji in range(3):mask[...,34+ji*16:40+ji*16]&=x[...,40+ji*16:46+ji*16].bool()
            if not torch.isfinite(x[mask]).all():raise ValueError('Nonfinite active history')
            axes=tuple(range(x.ndim-1));s=stats[name]
            s[0]+=torch.where(mask,x,0).sum(axes);s[1]+=torch.where(mask,x*x,0).sum(axes);s[2]+=mask.sum(axes)
    out={}
    for name,(summ,sq,count) in stats.items():
        mean=summ/count.clamp_min(1);std=(sq/count.clamp_min(1)-mean*mean).clamp_min(0).sqrt().clamp_min(1e-3)
        keep=list(range(12,16))+list(range(6,12))
        for ji in range(3):
            start=ji*16 if name=='commands' else 34+ji*16
            for k in keep:mean[start+k]=0;std[start+k]=1
        mean[-1]=0;std[-1]=1
        if name=='history':mean[15]=0;std[15]=1
        out[name+'_mean']=mean.float();out[name+'_std']=std.float()
    out['loss_leg_std']=torch.cat(leg).std(0,unbiased=False).clamp_min(1e-3)
    out['loss_force_std']=torch.cat(force).std(0,unbiased=False).clamp_min(1.)
    return out

def normalized_inputs(inputs,scales):
    values=[]
    for name,x,valid in [('history',inputs.history,inputs.history_valid),('commands',inputs.commands,inputs.command_valid.any(1))]:
        z=(x.float()-scales[name+'_mean'])/scales[name+'_std']
        for ji in range(3):
            start=ji*16 if name=='commands' else 34+ji*16
            z[...,start:start+6]=z[...,start:start+6]*x[...,start+6:start+12]
        z=z*valid[...,None];values.append(z)
    return values
