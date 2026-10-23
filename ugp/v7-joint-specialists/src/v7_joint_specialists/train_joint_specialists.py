"""Torque-first specialist training; checkpoint selection uses temporal validation."""
import argparse
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
sys.path[:0]=[str(Path(__file__).resolve().parent.parent/'v7legs_body'),str(Path(__file__).resolve().parent.parent/'v7feedback')]
import numpy as np
import torch
from data import atomic_json,atomic_save
from foundation import load_runtime,sha256,verify_provenance
from models import Hybrid,TorchFK
if os.name=='nt':
    # Reuse only these definitions: importing the Linux launcher would import fcntl.
    import ast
    helper=Path(__file__).resolve().parent.parent/'v7legs_body/train.py'
    tree=ast.parse(helper.read_text())
    tree.body=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('Reporter','capture_rng','restore_rng')]
    exec(compile(tree,str(helper),'exec'),globals())
else:
    from train import Reporter,capture_rng,restore_rng
from provenance import canonical_record
from contracts import ToleranceConfig
from command_data import extract_commands,verified_command_cache
from recording_source import recording
from joint_data import JointDataset,collate_windows,fit_input_scales
from joint_models import JointSpecialists
from joint_objectives import stage_loss
from strict_metrics import score_scalar,score_vector,strict_summary

STAGES=tuple([f'control_T{j}' for j in range(3)]+[f'{f}{j}' for f in ('T','Q','D') for j in range(3)]+[f'retune_T{j}' for j in range(3)]+['force'])

def save_training_state(path,module,opt,signature,state):
    atomic_save(path,dict(signature=canonical_record(signature),model=module.state_dict(),optimizer=opt.state_dict(),rng=capture_rng(),state=state))

def restore_training_state(path,module,opt,signature):
    packet=torch.load(path,map_location='cpu',weights_only=False)
    if packet['signature']!=canonical_record(signature):raise ValueError('Resume dependency signature differs; preserve the old run')
    module.load_state_dict(packet['model']);opt.load_state_dict(packet['optimizer']);restore_rng(packet['rng'])
    return packet['state']

def balanced_indices(dataset,generator,hard):
    groups=np.unique(dataset.trajectories);quota=max(1,int(np.ceil(len(dataset)/len(groups))))
    pieces=[]
    for g in groups:
        ids=np.flatnonzero(dataset.trajectories==g)
        ordinary=ids[torch.randint(len(ids),(quota,),generator=generator).numpy()]
        transition=ids[dataset.transitions[ids]]
        tail=ids[np.asarray(hard)[ids]>=np.quantile(np.asarray(hard)[ids],.9)]
        pieces.append(ordinary)
        for extra in (transition,tail):
            if len(extra):pieces.append(extra[torch.randint(len(extra),(max(1,quota//8),),generator=generator).numpy()])
    order=torch.tensor(np.concatenate(pieces),dtype=torch.long)
    return order[torch.randperm(len(order),generator=generator)]

def selection_key(scored):
    return (int((~np.asarray(scored['pass'])).sum()),float(np.asarray(scored['ratio']).max()),float(np.asarray(scored['error']).mean()))

def active_scores(out,t,stage,cfg):
    def cpu(x):return x.detach().cpu().double().numpy()
    if stage=='force':
        scores=[score_scalar(cpu(t.force),cpu(out.force),'force',cfg),score_vector(cpu(t.force),cpu(out.force),'force',cfg),score_scalar(cpu(t.contact),cpu(out.contact),'contact',cfg)]
        return {k:np.concatenate([s[k].ravel() for s in scores]) for k in scores[0]}
    name=stage.replace('control_','').replace('retune_','');j=int(name[-1]);kind,off={'Q':('angle',0),'D':('speed',3),'T':('torque',6)}[name[0]]
    return score_scalar(cpu(t.leg[...,off+j]),cpu(out.leg[...,off+j]),kind,cfg)

def output_for(model,inp,scales,stage):
    return model(inp,scales,'torque_control' if stage.startswith('control_') else 'joint_specialists') if stage!='force' else model(inp,scales)

@torch.no_grad()
def validate_head(model,dataset,scales,stage,cfg):
    model.eval();bs=16 if dataset.branch=='short' else 8;parts=[]
    device=next(model.parameters()).device
    for start in range(0,len(dataset),bs):
        inp,t=collate_windows([dataset[j] for j in range(start,min(start+bs,len(dataset)))])
        inp,t=inp.to(device),t.to(device)
        parts.append(active_scores(output_for(model,inp,scales,stage),t,stage,cfg))
    combined={k:np.concatenate([p[k].ravel() for p in parts]) for k in parts[0]}
    return selection_key(combined),strict_summary(combined)

def active_module(model,stage):
    if stage=='force':return model.extension
    name=stage.replace('control_','').replace('retune_','')
    return model.controls[name] if stage.startswith('control_') else model.heads[name]

def train_head(model,datasets,scales,stage,output,signature,report):
    train,val=datasets;branch=train.branch;cfg=ToleranceConfig();device=next(model.parameters()).device
    if stage=='force':model.configure_force()
    else:model.configure(stage)
    module=active_module(model,stage)
    opt=torch.optim.AdamW(model.active_parameters(),lr=5e-5 if stage.startswith('retune') else 3e-4,weight_decay=1e-4)
    directory=Path(output)/'checkpoints'/branch;latest=directory/f'{stage}_latest.pt';bestpath=directory/f'{stage}_best.pt'
    max_epochs=15 if stage.startswith('retune') else 30;bs=16 if branch=='short' else 8
    generator=torch.Generator().manual_seed(2781+STAGES.index(stage))
    state=dict(epoch=0,cursor=0,steps=0,patience=0,best=None,order=None,hard=np.zeros(len(train)),generator=generator.get_state(),complete=False,running=0.,n=0)
    if latest.exists():
        state=restore_training_state(latest,module,opt,signature)
        if state['complete']:
            packet=torch.load(bestpath,map_location='cpu',weights_only=False)
            if packet['signature']!=canonical_record(signature):raise ValueError('Best checkpoint signature differs')
            module.load_state_dict(packet['model']);report('stage_reused',branch=branch,head=stage,steps=state['steps']);return bestpath
        generator.set_state(state['generator'])
    def save():
        state['generator']=generator.get_state();save_training_state(latest,module,opt,signature,state)
    start_clock=time.monotonic();start_steps=state['steps']
    for ep in range(state['epoch'],max_epochs):
        state['epoch']=ep
        if state['order'] is None:
            state.update(order=balanced_indices(train,generator,state['hard']),cursor=0,running=0.,n=0)
        model.train()
        order=state['order']
        for offset in range(state['cursor'],len(order),bs):
            ids=order[offset:offset+bs].tolist();inp,t=collate_windows([train[j] for j in ids]);inp,t=inp.to(device),t.to(device)
            opt.zero_grad(set_to_none=True);out=output_for(model,inp,scales,stage)
            loss=stage_loss(out,t,scales,stage,model.fk,cfg,ep<3)
            if not torch.isfinite(loss):raise ValueError('Nonfinite specialist loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.active_parameters(),1.,error_if_nonfinite=True);opt.step()
            if stage!='force' and any(p.grad is not None for p in model.seed.parameters()):raise RuntimeError('Frozen seed gradient')
            if stage=='force' and any(p.grad is not None for p in model.joints.parameters()):raise RuntimeError('Frozen joints gradient')
            scored=active_scores(out,t,stage,cfg)
            # Only optimization rows determine hard sampling; unseen data is not loaded.
            shape=(len(ids),-1);err=np.asarray(scored['ratio']).reshape(shape).mean(1) if stage!='force' else np.full(len(ids),float(np.mean(scored['ratio'])))
            for j,e in zip(ids,err):state['hard'][j]=float(e)
            state.update(cursor=offset+len(ids),steps=state['steps']+1,running=state['running']+float(loss.detach())*len(ids),n=state['n']+len(ids))
            if state['steps']==1 or state['steps']%50==0:
                elapsed=time.monotonic()-start_clock;done=max(1,state['steps']-start_steps)
                report('training',branch=branch,head=stage,epoch=ep+1,max_epochs=max_epochs,steps=state['steps'],
                    loss=float(loss.detach()),grad_norm=float(norm),batch=offset//bs+1,batches=(len(order)+bs-1)//bs,
                    training_trajectories=28,train_windows=len(train),validation_windows=len(val),seconds_per_step=elapsed/done,
                    stage_cap_remaining_seconds=(max_epochs-ep)*(len(order)/bs)*elapsed/done,gpu_memory_mb=torch.cuda.memory_allocated()/1024**2 if device.type=='cuda' else 0)
            if state['steps']==1 or state['steps']%250==0:save()
        score,summary=validate_head(model,val,scales,stage,cfg)
        if state['best'] is None or score<tuple(state['best']):
            state['best']=score;state['patience']=0
            atomic_save(bestpath,dict(signature=canonical_record(signature),model=module.state_dict(),validation=summary,selection=score,epoch=ep+1,steps=state['steps']))
        else:state['patience']+=1
        record=dict(branch=branch,head=stage,epoch=ep+1,steps=state['steps'],train_loss=state['running']/state['n'],selection=score,patience=state['patience'],**summary)
        with (Path(output)/'metrics.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        report('epoch_complete',**record)
        state.update(epoch=ep+1,order=None,cursor=0)
        if state['patience']>=5:break
    state['complete']=True;save()
    module.load_state_dict(torch.load(bestpath,map_location='cpu',weights_only=False)['model']);return bestpath

def audit_commands(rt,parent,output,manifest,report):
    root=Path(output)/'commands';root.mkdir(parents=True,exist_ok=True);mapping=None;signatures={}
    for n,(env,run) in enumerate(manifest['training_trajectories'],1):
        raw=rt['CACHE_ROOT']/f'{env}_{run}_physics.npz'
        with np.load(raw) as z:p={k:z[k] for k in z.files}
        m=p['actuator_for_joint'].astype(int)
        if mapping is None:mapping=m
        if not np.array_equal(m,mapping):raise ValueError('Hardware permutation inconsistent across training trajectories')
        report('command_audit',trajectory=f'{env}/{run}',trajectories_completed=n-1,trajectories_total=28)
        path=root/f'{env}_{run}_commands.npz'
        if not verified_command_cache(path,p,mapping):
            with recording(rt['ROOT'],env,run,Path(output)/'temporary_recordings') as bag:
                extract_commands(bag,p,mapping,path)
        target=root/f'{env}_{run}_physics.npz'
        if target.exists():
            if sha256(target)!=sha256(raw):raise ValueError('Physics snapshot changed')
        else:
            import shutil
            shutil.copyfile(raw,target)
        signatures[f'{env}/{run}']=json.loads(path.with_suffix('.json').read_text())
    atomic_json(root/'manifest.json',dict(mapping=mapping.tolist(),trajectories=signatures));return root

def load_branch(parent,feedback,branch,fk,device):
    from feedback_models import FeedbackModel
    seed=Hybrid(copy.deepcopy(fk));seed.load_state_dict(torch.load(Path(parent)/'checkpoints'/branch/'joint_best.pt',map_location='cpu',weights_only=False)['model'])
    joints=JointSpecialists(seed).to(device)
    prior=FeedbackModel(seed);prior.load_state_dict(torch.load(Path(feedback)/'checkpoints'/branch/'one_pass_best.pt',map_location='cpu',weights_only=False)['model'])
    return joints,prior.to(device).eval().requires_grad_(False)

def verify_real(joints,dataset,scales):
    device=next(joints.parameters()).device;inp,t=collate_windows([dataset[0],dataset[min(1,len(dataset)-1)]]);inp,t=inp.to(device),t.to(device)
    joints.configure(None);joints.eval()
    with torch.no_grad():
        expected=joints.seed(inp.base,scales,'hybrid');out=joints(inp,scales)
        torch.testing.assert_close(out.leg,expected['leg'],atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(out.body,expected['body'],atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(out.force,inp.base['force'],atol=0,rtol=0)
    for stage in ('T0','Q1','D2'):
        joints.configure(stage);joints.zero_grad(set_to_none=True);out=joints(inp,scales)
        loss=stage_loss(out,t,scales,stage,joints.fk,warmup=True);loss.backward()
        if not any(p.grad is not None and p.grad.abs().sum()>0 for p in joints.active_parameters()):raise ValueError('Real selected-head gradients missing')
        if any(p.grad is not None for p in joints.seed.parameters()):raise ValueError('Real frozen seed gradient')
    joints.zero_grad(set_to_none=True);joints.configure(None)
    return dict(train_windows=len(dataset),target_shape=list(t.leg.shape),initial_predictions_match=True)

def main():
    import fcntl
    parser=argparse.ArgumentParser()
    for name in ('root','parent','feedback','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--notebook',type=Path,default=Path(__file__).resolve().parent.parent/'v7legs_body/foundation.ipynb')
    for flag in ('audit-only','verify-only','train-only'):parser.add_argument('--'+flag,action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    lock=(args.output/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    report=Reporter(args.output)
    try:
        random.seed(2781);np.random.seed(2781);torch.manual_seed(2781);torch.set_num_threads(2)
        device=torch.device('cpu' if args.audit_only else 'cuda:0');rt=load_runtime(args.root,args.notebook,device)
        pm=json.loads((args.parent/'manifest.json').read_text());proven=verify_provenance(rt)
        for k in ('training_trajectories','heldout_trajectories','completed_foundation'):
            if canonical_record(proven[k])!=canonical_record(pm[k]):raise ValueError('Parent/foundation provenance changed')
        if pm['notebook_sha256']!=sha256(args.notebook):raise ValueError('Parent notebook changed')
        source={str(p.relative_to(Path(__file__).parent.parent)):sha256(p) for name in ('v7_joint_specialists','v7legs_body','v7feedback') for p in sorted((Path(__file__).parent.parent/name).glob('*.py'))}
        manifest=canonical_record(dict(proven,source=source,seed=2781,tolerances=asdict(ToleranceConfig()),
            parent_sha256=sha256(args.parent/'manifest.json'),feedback_sha256=sha256(args.feedback/'manifest.json'),
            weights={f'{b}_{n}':sha256(p) for b in ('short','long') for n,p in [('parent',args.parent/'checkpoints'/b/'joint_best.pt'),('force',args.feedback/'checkpoints'/b/'one_pass_best.pt')]},
            urdf_sha256=sha256(rt['ANYMAL_D_URDF']),input_expansion='Active commands and sub-grid raw command prefixes; measured future state forbidden',stages=STAGES))
        mp=args.output/'manifest.json'
        if mp.exists() and json.loads(mp.read_text())!=manifest:raise ValueError('Run inputs changed; use a new output identity')
        atomic_json(mp,manifest);command_root=audit_commands(rt,args.parent,args.output,manifest,report)
        if args.audit_only:report('audit_complete',training_trajectories=28);return
        fk=TorchFK(rt['ANYMAL_D_FK'],rt['_rpy_R'])
        for branch in ('short','long'):
            train=JointDataset(args.parent,args.feedback,command_root,branch,'train');val=JointDataset(args.parent,args.feedback,command_root,branch,'val')
            sp=args.output/f'{branch}_scales.pt'
            if not sp.exists():
                original=torch.load(args.feedback/f'{branch}_scales.pt',map_location='cpu',weights_only=False)
                atomic_save(sp,dict(original,**fit_input_scales(train)))
            scales={k:v.to(device) for k,v in torch.load(sp,map_location='cpu',weights_only=False).items()}
            joints,prior=load_branch(args.parent,args.feedback,branch,fk,device)
            report('real_data_verified',branch=branch,training_trajectories=28,validation_windows=len(val),**verify_real(joints,train,scales))
            sig=dict(manifest=sha256(mp),branch=branch,cache=train.signatures,scales=sha256(sp),commands=sha256(command_root/'manifest.json'))
            dependencies={}
            if not args.verify_only:
                for stage in STAGES:
                    if stage=='force':
                        from force_models import ForceContactSpecialist
                        model=ForceContactSpecialist(joints,prior).to(device)
                    else:model=joints
                    ck=train_head(model,(train,val),scales,stage,args.output,dict(sig,stage=stage,predecessors=dict(dependencies)),report)
                    dependencies[stage]=sha256(ck)
                atomic_json(args.output/f'{branch}_trained.json',dict(signature=sig,checkpoints=dependencies))
            del train,val,joints,prior
            if not args.verify_only:del model
            torch.cuda.empty_cache()
        if args.verify_only:report('verification_complete');return
        report('training_complete',training_trajectories=28,goal_achieved=False)
        if not args.train_only:
            from evaluate_joint_specialists import evaluate_all
            result=evaluate_all(rt,args.output,args.parent,args.feedback,fk,manifest,report)
            report('complete',**result)
    except Exception as exc:
        report('failed',error=f'{type(exc).__name__}: {exc}');traceback.print_exc();raise

if __name__=='__main__':main()
