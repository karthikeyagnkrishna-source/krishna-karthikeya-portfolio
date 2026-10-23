"""Original 96 test cases with seven locked variants and every strict failure."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import torch
from data import atomic_json,atomic_save,nearest
from foundation import load_record,load_models,sha256
from evaluate import export_case,prediction_record,fit_causal_physics
from provenance import input_digest
from contracts import InputBatch,Targets,Predictions,ToleranceConfig
from command_data import extract_commands,window_arrays,check_targets,verified_command_cache
from recording_source import recording
from strict_metrics import score_scalar,score_vector,strict_summary
from train_joint_specialists import load_branch,active_module,STAGES
from force_models import ForceContactSpecialist

VARIANTS=('base_v7','base_v7f','seed','torque_control','torque_only','joint_specialists','full')

def check_completion(case_ids,failures,points):
    if len(case_ids)!=96 or set(case_ids)!=set(range(96)):raise ValueError('Incomplete or duplicate evaluation cases')
    if points!=57600:raise ValueError('Missing required exact prediction points')
    return dict(complete=True,goal_achieved=failures==0,failures=int(failures),required_points=int(points),cases=96)

def finalize_protocol(path,signature,result):
    atomic_json(path,dict(signature=signature,variants=VARIANTS,training_trajectories=28,familiar=16,time_gap=64,unseen=16,
        note='Existing excluded trajectories are development holdouts, not fresh blind evidence. Known future controls; new fields/resolution explicitly expanded.',**result))

def export_variant(meta,variant,inputs,targets,predictions):
    if targets.leg.shape[0]!=1:raise ValueError('Export one exact case at a time')
    horizons=[round(float(h),1) for h in inputs.base['horizon'].detach().cpu().tolist()]
    times=targets.times
    batch=dict(inputs.base,target_leg=targets.leg,target_force=targets.force,target_body=targets.body,target_contact=targets.contact,
               anchor=times[:,0]-horizons[0],target_time=times,force=predictions.force)
    # Geometry of actual angles is calculated outside every prediction forward call.
    true_r=meta.pop('_target_r') if '_target_r' in meta else None
    if true_r is None:
        raise ValueError('Export requires actual FK from the caller')
    prediction=dict(leg=predictions.leg,body=predictions.body,r=predictions.r,target_r=true_r)
    rows=export_case(meta,variant,batch,prediction,horizons);rows['strict']=[]
    def cpu(x):return x.detach().double().cpu().numpy()[0]
    actual,pred=cpu(targets.leg),cpu(predictions.leg)
    scalar={k:score_scalar(actual[...,o:o+3],pred[...,o:o+3],k) for k,o in [('angle',0),('speed',3),('torque',6)]}
    def add(common,quantity,signal,s,index):
        point=dict(common,quantity=quantity,signal=signal,actual=float(s['actual'][index]),predicted=float(s['predicted'][index]),
                   error=float(s['error'][index]),percent=float(s['percent'][index]),tolerance=float(s['tolerance'][index]),
                   zero=bool(s['zero'][index]),ratio=float(s['ratio'][index]),**{'pass':bool(s['pass'][index])})
        rows['strict'].append(point)
        return point
    for n,row in enumerate(rows['joint']):
        hi=n//12;li=(n%12)//3;ji=n%3
        for name,s in scalar.items():
            pt=add({k:v for k,v in row.items() if k in meta or k in ('model','leg','joint','horizon_s','anchor_time_s','target_time_s')},name,row['joint'],s,(hi,li,ji))
            row.update({f'{name}_actual':pt['actual'],f'{name}_predicted':pt['predicted'],f'{name}_signed_error':pt['predicted']-pt['actual'],
                f'{name}_absolute_error':pt['error'],f'{name}_error_percent':pt['percent'],f'{name}_exact_zero':pt['zero'],
                f'{name}_tolerance':pt['tolerance'],f'{name}_strict_pass':pt['pass']})
    af,pf=cpu(targets.force),cpu(predictions.force)
    components=score_scalar(af,pf,'force');vector=score_vector(af,pf,'force')
    foot=score_vector(cpu(true_r),cpu(predictions.r),'foot');contact=score_scalar(cpu(targets.contact),cpu(predictions.contact),'contact')
    for n,row in enumerate(rows['force']):
        hi,li=divmod(n,4)
        common={k:v for k,v in row.items() if k in meta or k in ('model','leg','horizon_s','anchor_time_s','target_time_s')}
        for j,axis in enumerate(('Fx','Fy','Fz')):
            pt=add(common,'force_component',axis,components,(hi,li,j))
            row.update({f'{axis}_error_N':pt['error'],f'{axis}_error_percent':pt['percent'],f'{axis}_exact_zero':pt['zero'],f'{axis}_tolerance_N':pt['tolerance'],f'{axis}_strict_pass':pt['pass']})
        for name,s in [('force_vector',vector),('foot_position',foot),('contact',contact)]:
            pt=add(common,name,name,s,(hi,li))
            row.update({f'{name}_exact_zero':pt['zero'],f'{name}_tolerance':pt['tolerance'],f'{name}_strict_pass':pt['pass'],f'{name}_error_percent':pt['percent']})
        row.update(predicted_contact=float(cpu(predictions.contact)[hi,li]),contact_absolute_error=float(contact['error'][hi,li]),
                   contact_classification_error=int((cpu(predictions.contact)[hi,li]>=.5)!=(cpu(targets.contact)[hi,li]>=.5)))
        row.update(actual_magnitude_N=float(vector['actual'][hi,li]),predicted_magnitude_N=float(vector['predicted'][hi,li]),
                   vector_error_N=float(vector['error'][hi,li]),vector_error_percent=float(vector['percent'][hi,li]),
                   foot_position_error_m=float(foot['error'][hi,li]))
        denominator=np.linalg.norm(af[hi,li])*np.linalg.norm(pf[hi,li])
        row['cosine']=float(np.dot(af[hi,li],pf[hi,li])/denominator) if denominator>1e-16 else np.nan
        for j,axis in enumerate(('x','y','z')):
            row[f'actual_foot_{axis}_m']=float(cpu(true_r)[hi,li,j]);row[f'predicted_foot_{axis}_m']=float(cpu(predictions.r)[hi,li,j])
    return rows

def write_exports(combined,directory):
    full_result=None
    for variant,tables in combined.items():
        frames={k:pd.DataFrame(v) for k,v in tables.items()};strict=frames['strict']
        failures=strict.loc[~strict['pass']].copy();summaries=[]
        for keys,g in strict.groupby(['split','adaptation_gap_s','horizon_s','leg','quantity','signal'],dropna=False):
            data={k:g[k].to_numpy() for k in ('error','percent','pass','zero')}
            summaries.append(dict(zip(('split','adaptation_gap_s','horizon_s','leg','quantity','signal'),keys),**strict_summary(data)))
        summary=pd.DataFrame(summaries)
        for name,frame in frames.items():frame.to_csv(directory/f'{variant}_{name}_exact.csv',index=False)
        failures.to_csv(directory/f'{variant}_failures.csv',index=False);summary.to_csv(directory/f'{variant}_strict_summary.csv',index=False)
        with pd.ExcelWriter(directory/f'{variant}_all_tests_exact.xlsx',engine='openpyxl') as writer:
            sheets={'Joint exact':frames['joint'],'Body exact':frames['body'],'Force exact':frames['force'],'Strict summary':summary,'Failures':failures}
            notes=['All 28 original training trajectories; original temporal masks and four excluded development trajectories.',
                'Every row identifies its leg, joint/channel, horizon, exact anchor and target time. Degrees, rad/s, Nm, N and metres.',
                'Strict nonzero pass: absolute error < 2% of absolute actual value. Exactly 2% fails. Effective zero <=1e-8 reported units.',
                'Zero limits: angle .05 deg, speed .01 rad/s, torque .05 Nm, force 1 N. Contact probability error < .02.',
                'Foot is body-relative URDF base geometry. Zero actual norm has undefined percentage and requires zero error.',
                'Future recorded commands are assumed known. New specialists add active mode-valid fields and sub-grid command prefixes before target.',
                'V7 has four and V7-F five original local adaptation epochs. All new specialists have zero local adaptation epochs.',
                'Body stays exactly at parent seed predictions for all new specialists. No recurrent loop.',
                'Standalone V7 force uses its own residual, baseline and scales. It is not the V7-F prediction.',
                'Every failure is retained. Improved averages do not establish the every-point 2% goal.']
            sheets['How to read']=pd.DataFrame({'note':notes})
            for sheet,frame in sheets.items():
                frame.to_excel(writer,sheet_name=sheet,index=False);ws=writer.sheets[sheet];ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
                for column in ws.columns:
                    ws.column_dimensions[column[0].column_letter].width=min(34,max(14,len(str(column[0].value))+2))
                    if column[0].value=='cosine':
                        for cell in column[1:]:cell.number_format='0.000000'
        if variant=='full':full_result=check_completion(list(range(96)),len(failures),len(strict))
    return full_result

def make_jobs(rt,records):
    diagnostics=[r for r in records if len(r['ttests'])==4];by_terrain={r['terrain_type']:r for r in diagnostics}
    jobs=[(r,r,i,tt,'familiar' if gap==0 else 'time_gap',gap) for r in diagnostics for i,tt in enumerate(r['ttests'],1) for gap in (0,4,7,9,10)]
    for terrain,(env,run) in rt['STRICT_HOLDOUTS'].items():
        rec=load_record(rt,terrain,env,run,holdout=True);t=rec['data']['time'];dt=np.median(np.diff(t))
        valid=np.flatnonzero((np.arange(len(t))>=int(round(4/dt))-1)&(np.arange(len(t))<len(t)-int(round(1/dt)))&(rec['visual']['visual_index']>=0))
        start=max(float(t[valid[0]])+1,20)
        if start+26>float(t[valid[-1]]):start=float(t[valid[-1]])-26
        if start<float(t[valid[0]]):raise ValueError('Unseen recording too short')
        for i,delta in enumerate((0,7,16,26),1):
            tt=float(t[valid[np.argmin(abs(t[valid]-start-delta))]])
            jobs.append((rec,by_terrain[terrain],i,tt,'unseen',0))
    identities={(r['environment'],r['run'],i,split,gap) for r,_,i,_,split,gap in jobs}
    if len(jobs)!=96 or len(identities)!=96:raise ValueError('Expected 96 distinct test cases')
    return jobs

def load_trained(parent,feedback,output,branch,fk,device):
    joints,prior=load_branch(parent,feedback,branch,fk,device)
    checkpoints={}
    trained=json.loads((Path(output)/f'{branch}_trained.json').read_text())
    for stage in STAGES:
        path=Path(output)/'checkpoints'/branch/f'{stage}_best.pt'
        if sha256(path)!=trained['checkpoints'][stage]:raise ValueError('Trained checkpoint was changed')
        checkpoints[stage]=torch.load(path,map_location='cpu',weights_only=False)
        if stage!='force':active_module(joints,stage).load_state_dict(checkpoints[stage]['model'])
    early,unused=load_branch(parent,feedback,branch,fk,device);del unused
    for j in range(3):early.heads[f'T{j}'].load_state_dict(checkpoints[f'T{j}']['model'])
    joints.configure(None);early.configure(None)
    full=ForceContactSpecialist(joints,prior).to(device)
    full.extension.load_state_dict(checkpoints['force']['model']);full.eval().requires_grad_(False)
    return joints.eval(),early.eval().requires_grad_(False),full

@torch.no_grad()
def standalone_force(rt,rec,idx,base,stats,physics,horizons,x):
    sample=rt['make_v7_sample'](rec['data'],rec['visual'],stats,physics,idx,base.history_steps,horizons,rt['normalize_arrays'](rec['data'],stats))
    baseline=torch.as_tensor(sample['baseline_p_raw'][...,9:12],device=rt['DEVICE'])[None]
    residual=base.tau_force_head(x.float())[...,3:6]
    return baseline+residual*torch.as_tensor(stats['P_std'][9:12],device=rt['DEVICE'])

def evaluate_all(rt,output,parent,feedback,fk,manifest,report):
    output=Path(output);directory=output/'evaluation';directory.mkdir(exist_ok=True)
    records=[load_record(rt,t,e,r) for t,pairs in rt['EXPANDED_TRAIN_RUNS'].items() for e,r in pairs]
    rt['TRAIN_RECORDS']=records;rt['REGIME_THRESHOLDS']=rt['learn_global_regime_thresholds'](records)
    centers,weights,counts=rt['learn_regime_centers_and_weights'](records,rt['REGIME_THRESHOLDS'])
    rt.update(REGIME_CENTERS=centers,REGIME_CLASS_WEIGHTS=weights,REGIME_COUNTS=counts)
    rt['_legacy_fit_causal_physics']=rt['fit_causal_physics'];rt['fit_causal_physics']=lambda d,s,e,thresholds=None:fit_causal_physics(rt,d,s,e)
    for rec in records:rec['physics_prior_v7']=fit_causal_physics(rt,rec['data'],float(rec['data']['time'][0]),min(rec['ttests']))
    jobs=make_jobs(rt,records);commands={};root=output/'commands';mapping=np.array(json.loads((root/'manifest.json').read_text())['mapping'])
    input_paths=[rt['ANYMAL_D_URDF'],output/'manifest.json']
    input_paths.extend(rt['CACHE_ROOT']/f"{r['environment']}_{r['run']}_physics.npz" for r in records)
    for rec,_,_,_,_,_ in jobs:
        name=f"{rec['environment']}_{rec['run']}"
        if name in commands:continue
        path=root/f'{name}_commands.npz'
        if not verified_command_cache(path,rec['data'],mapping):
            with recording(rt['ROOT'],rec['environment'],rec['run'],output/'temporary_recordings') as bag:
                extract_commands(bag,rec['data'],mapping,path)
        with np.load(path) as z:commands[name]={k:z[k] for k in z.files}
        input_paths.extend([path,path.with_suffix('.json')])
        input_paths.extend(rt['CACHE_ROOT']/f'{name}_{suffix}' for suffix in ('physics.npz','frame_manifest.npz','v7_dinov3.npy','v7_convnext.npy','v7_utonia.npy','v7_waft.npy'))
    foundations={};models={};scales={};checkpoint_hashes={}
    for branch in ('short','long'):
        foundations[branch]=load_models(rt,records[0],branch)
        models[branch]=load_trained(parent,feedback,output,branch,fk,rt['DEVICE'])
        scales[branch]={k:v.to(rt['DEVICE']) for k,v in torch.load(output/f'{branch}_scales.pt',map_location='cpu',weights_only=False).items()}
        input_paths.append(output/f'{branch}_scales.pt')
        for stage in STAGES:checkpoint_hashes[f'{branch}_{stage}']=sha256(output/'checkpoints'/branch/f'{stage}_best.pt')
    sources=input_digest(input_paths);atomic_json(directory/'input_provenance.json',sources)
    signature=hashlib.sha256(json.dumps(dict(manifest=manifest,checkpoints=checkpoint_hashes,inputs=sources),sort_keys=True).encode()).hexdigest()
    for job_id,(rec,source,test_id,tt,split,gap) in enumerate(jobs):
        path=directory/f'case_{job_id:03d}.pt'
        identity=[rec['environment'],rec['run'],test_id,split,gap]
        if path.exists():
            pkt=torch.load(path,map_location='cpu',weights_only=False)
            if pkt['signature']!=signature or pkt['identity']!=identity:raise ValueError('Saved case belongs to different weights/inputs')
            continue
        meta=dict(environment=rec['environment'],trajectory=rec['run'],test_id=test_id,split=split,requested_test_time_s=tt,
                  adaptation_gap_s=gap,adaptation_cutoff_s=None if split=='unseen' else tt-gap,new_head_local_epochs=0)
        report('evaluating',case=job_id+1,cases=96,**meta)
        rows={v:dict(joint=[],body=[],force=[],strict=[]) for v in VARIANTS};saved={}
        for branch in ('short','long'):
            bg,fg,horizons=foundations[branch];fs=rt['make_stats_between'](source['data'],float(source['data']['time'][0]),min(source['ttests']));fp=source['physics_prior_v7']
            if split=='unseen':base,fh,stats,physics=bg,fg,fs,fp
            else:
                cutoff=tt-gap;context,hop,half_life=(.5,.3,5) if branch=='short' else (4,1,6)
                bc=copy.deepcopy(bg).requires_grad_(True)
                base,stats,physics,_=rt['adapt_v7'](bc,rec['data'],rec['visual'],cutoff,context,hop,horizons,half_life,4,781+job_id,branch)
                base.eval().requires_grad_(False);del bc
                fc=copy.deepcopy(fg).requires_grad_(True)
                fh,_,_,_=rt['adapt_v7f_head'](fc,bg,rec,cutoff,context,hop,horizons,half_life,1781+job_id,branch)
                fh.eval().requires_grad_(False);del fc
            idx=nearest(rec['data']['time'],tt)
            record=prediction_record(rt,rec,idx,base,bg,fh,stats,physics,fs,fp,horizons)
            check_targets(rec['data'],float(record['anchor']),record['target_time'].numpy(),record['target_leg'].numpy(),record['target_force'].numpy())
            arrays=window_arrays(rec['data'],commands[f"{rec['environment']}_{rec['run']}"],float(record['anchor']),record['target_time'].numpy(),.5 if branch=='short' else 4)
            inp=InputBatch({k:record[k][None].to(rt['DEVICE']) for k in ('x','base_leg','base_body','force','contact')},*(torch.from_numpy(a)[None].to(rt['DEVICE']) for a in arrays))
            inp.base['horizon']=torch.tensor(horizons,device=rt['DEVICE'])
            target=Targets(*(record[k][None].to(rt['DEVICE']) for k in ('target_leg','target_force','target_body','target_contact','target_time')),[identity])
            j,early,full=models[branch]
            with torch.no_grad():
                base_r,jac=fk.to(rt['DEVICE'])(inp.base['base_leg'][...,:3])
                bp=Predictions(inp.base['base_leg'],inp.base['force'],inp.base['base_body'],inp.base['contact'],base_r)
                original=standalone_force(rt,rec,idx,base,stats,physics,horizons,inp.base['x'])
                variants=dict(base_v7=Predictions(bp.leg,original,bp.body,bp.contact,bp.r),base_v7f=bp,seed=j(inp,scales[branch],'seed'),
                    torque_control=j(inp,scales[branch],'torque_control'),torque_only=early(inp,scales[branch],'torque_only'),
                    joint_specialists=j(inp,scales[branch]),full=full(inp,scales[branch]))
                target_r=fk(target.leg[...,:3])[0]
                for variant,pred in variants.items():
                    exported=export_variant(dict(meta,_target_r=target_r),variant,inp,target,pred)
                    for name in rows[variant]:rows[variant][name].extend(exported[name])
                saved[branch]=dict(base={k:v.detach().cpu() for k,v in inp.base.items()},targets={k:getattr(target,k).cpu() for k in ('leg','force','body','contact','times')},
                    predictions={v:{k:getattr(p,k).detach().cpu() for k in ('leg','force','body','contact','r')} for v,p in variants.items()})
            if split!='unseen':del base,fh
            torch.cuda.empty_cache()
        atomic_save(path,dict(signature=signature,identity=identity,rows=rows,tensors=saved))
    combined={v:dict(joint=[],body=[],force=[],strict=[]) for v in VARIANTS};case_ids=[]
    for job_id in range(96):
        pkt=torch.load(directory/f'case_{job_id:03d}.pt',map_location='cpu',weights_only=False)
        if pkt['signature']!=signature:raise ValueError('Mismatched case provenance')
        case_ids.append(job_id)
        for v in VARIANTS:
            for name in combined[v]:combined[v][name].extend(pkt['rows'][v][name])
    check_completion(case_ids,sum(not r['pass'] for r in combined['full']['strict']),len(combined['full']['strict']))
    result=write_exports(combined,directory)
    finalize_protocol(directory/'protocol.json',signature,result)
    return dict(result_directory=str(directory),**result)
