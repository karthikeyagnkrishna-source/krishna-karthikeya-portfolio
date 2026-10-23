"""ROS commanded fields only; future actuator state is never extracted."""
import json
from pathlib import Path
import numpy as np
from data import nearest
from foundation import sha256

SCHEMA='joint_commands_v2_original_axis_55_83'
FIELDS=('joint_torque','position','velocity','pid_gains_p','pid_gains_i','pid_gains_d')
PHYSICAL={'/state_estimator/joint_states','/state_estimator/twist',
          *('/state_estimator/contact_force_'+x+'_foot' for x in ('lf','rf','lh','rh'))}

def encode_commands(values,modes,mapping):
    mapping=np.asarray(mapping,dtype=int)
    if sorted(mapping.tolist())!=list(range(12)): raise ValueError('Invalid actuator permutation')
    v=np.asarray(values,dtype=np.float32)[:,mapping];m=np.asarray(modes)[:,mapping]
    active=np.zeros(v.shape,dtype=bool)
    active[...,0]=np.isin(m,[10,12,13]);active[...,1:3]=np.isin(m,[12,13])[...,None]
    active[...,3:]= (m==13)[...,None]
    if not np.isfinite(v[active]).all(): raise ValueError('Nonfinite active command field')
    v=np.where(active,v,0)
    mode=np.stack([m==10,m==12,m==13,~np.isin(m,[10,12,13])],-1)
    result=np.concatenate([v,active.astype(np.float32),mode.astype(np.float32)],-1).reshape(-1,4,48)
    unique,counts=np.unique(m,return_counts=True)
    return result,dict(modes={str(k):int(c) for k,c in zip(unique,counts)},
                       unexpected_joint_commands=int(mode[...,-1].sum()),active_counts=active.sum((0,1)).tolist())

def sort_commands(times,values):
    t=np.asarray(times,dtype=np.float64)
    if not np.isfinite(t).all(): raise ValueError('Invalid command timestamps')
    order=np.argsort(t,kind='stable')
    return t[order],np.asarray(values)[order],dict(duplicates=int((np.diff(t[order])==0).sum()),out_of_order=int((np.diff(t)<0).sum()))

def causal_indices(source,target):
    ids=np.searchsorted(source,target,side='right')-1
    if np.any(ids<0): raise ValueError('Cannot extrapolate a command backwards')
    return ids

def extract_commands(bag,physics,mapping,output):
    from rosbags.highlevel import AnyReader
    output=Path(output);bag=Path(bag)
    sig=dict(schema=SCHEMA,bag_sha256=sha256(bag),mapping=np.asarray(mapping).tolist(),
             physics_digest=physics_digest(physics),implementation=sha256(__file__))
    meta=output.with_suffix('.json')
    if output.exists() and meta.exists():
        if json.loads(meta.read_text())['signature']!=sig: raise ValueError('Command cache inputs changed')
        return output
    def stamp(s): return float(s.sec)+float(s.nanosec)*1e-9
    values=[];modes=[];times=[];currents=[];spread=[];skipped=0
    with AnyReader([bag]) as reader:
        physical=[c for c in reader.connections if c.topic in PHYSICAL]
        _,t0,_=next(reader.messages(connections=physical));origin=t0*1e-9
        act=[c for c in reader.connections if c.topic=='/state_estimator/actuator_readings']
        if not act: raise ValueError('Actuator stream missing')
        for conn,_,raw in reader.messages(connections=act):
            readings=reader.deserialize(raw,conn.msgtype).readings
            if len(readings)!=12: skipped+=1;continue
            cmd=[r.commanded for r in readings]
            stamps=np.array([stamp(r.header.stamp) for r in cmd])
            times.append(float(np.median(stamps))-origin)
            spread.append(float(np.ptp(stamps)))
            values.append([[float(getattr(r,f)) for f in FIELDS] for r in cmd])
            modes.append([int(r.mode) for r in cmd]);currents.append([float(r.current) for r in cmd])
    if not times: raise ValueError('No complete commanded records')
    encoded,audit=encode_commands(values,modes,mapping)
    t,e,order_diag=sort_commands(times,encoded)
    order=np.argsort(times,kind='stable')
    legacy=np.asarray(values,dtype=np.float32)[order][:,mapping,0].reshape(-1,4,3)
    reconstructed=legacy[causal_indices(t,physics['grid_original'])]
    if not np.allclose(reconstructed,physics['L_LEG'],atol=1e-5,rtol=0):
        raise ValueError('Raw command torque does not reconstruct the legacy command grid')
    audit.update(order_diag,raw_messages=len(t),skipped_incomplete_messages=skipped,
                 max_joint_timestamp_spread_s=max(spread),legacy_max_difference=float(abs(reconstructed-physics['L_LEG']).max()),
                 command_effort_MAE_Nm=abs(reconstructed-physics['P'][...,6:9]).mean((0,1)).tolist(),
                 label_min=physics['P'].min(0).tolist(),label_max=physics['P'].max(0).tolist())
    output.parent.mkdir(parents=True,exist_ok=True)
    temp=output.with_suffix('.tmp.npz')
    np.savez_compressed(temp,time=t-float(physics['crop_start_original']),time_original=t,encoded=e,legacy_torque=legacy,
                        raw_current=np.asarray(currents,dtype=np.float32)[order],raw_mode=np.asarray(modes)[order])
    temp.replace(output)
    from data import atomic_json
    atomic_json(meta,dict(signature=sig,audit=audit,cache_sha256=sha256(output)))
    return output

def verified_command_cache(path,physics,mapping):
    path=Path(path);meta=path.with_suffix('.json')
    if not path.exists() or not meta.exists():return False
    packet=json.loads(meta.read_text());sig=packet['signature']
    expected=dict(schema=SCHEMA,mapping=np.asarray(mapping).tolist(),physics_digest=physics_digest(physics),implementation=sha256(__file__))
    if any(sig.get(k)!=v for k,v in expected.items()) or packet.get('cache_sha256')!=sha256(path):
        raise ValueError('Existing command cache integrity/signature mismatch')
    return True

def physics_digest(p):
    import hashlib
    h=hashlib.sha256()
    for k in ('time','grid_original','crop_start_original','P','B','E','U','contact','L_LEG'):
        a=np.ascontiguousarray(p[k]);h.update(k.encode());h.update(a.tobytes())
    return h.hexdigest()

def check_targets(p,anchor,target_time,leg,force):
    a=nearest(p['time'],anchor)
    ids=np.array([nearest(p['time'],t) for t in target_time])
    if not np.isclose(p['time'][a],anchor,atol=1e-7,rtol=0) or not np.allclose(p['time'][ids],target_time,atol=1e-7,rtol=0):
        raise ValueError('Parent target/anchor timestamps are not exact grid samples')
    if not np.allclose(p['P'][ids,...,:9],leg,atol=1e-7,rtol=0) or not np.allclose(p['P'][ids,...,9:12],force,atol=1e-7,rtol=0):
        raise ValueError('Parent labels differ from their exact physical target samples')
    return a,ids

def window_arrays(p,c,anchor,targets,context):
    idx=nearest(p['time'],anchor);dt=float(np.median(np.diff(p['time'])))
    n=int(round(context/dt));ids=np.arange(idx-n+1,idx+1)
    valid=ids>=0;safe=ids.clip(0,len(p['time'])-1)
    original='time_original' in c and 'grid_original' in p
    source_time=c['time_original'] if original else c['time']
    grid=p['grid_original'] if original else p['time']
    anchor_axis=grid[idx] if original else anchor
    target_axis=np.asarray([grid[nearest(p['time'],t)] for t in targets]) if original else np.asarray(targets)
    held=c['encoded'][causal_indices(source_time,grid[safe])]
    b=np.broadcast_to(p['B'][safe,None],(n,4,12));u=np.broadcast_to(p['U'][safe,None],(n,4,6))
    relative=np.broadcast_to((p['time'][safe]-anchor)[:,None,None],(n,4,1))
    history=np.concatenate([p['P'][safe],p['E'][safe],p['contact'][safe,:,None],b,u,held,relative],-1)
    history[~valid]=0
    chosen=np.flatnonzero((source_time>=anchor_axis)&(source_time<float(np.max(target_axis))))
    # Keep a padded token even for an empty prefix; encoders explicitly mask it.
    s=max(1,len(chosen));commands=np.zeros((s,4,55),np.float32);mask=np.zeros((len(targets),4,s),bool)
    if len(chosen):
        ct=source_time[chosen];ui=causal_indices(grid,ct)
        uu=np.broadcast_to(p['U'][ui,None],(len(chosen),4,6))
        rt=np.broadcast_to((ct-anchor_axis)[:,None,None],(len(chosen),4,1))
        commands[:len(chosen)]=np.concatenate([c['encoded'][chosen],uu,rt],-1)
        mask[:,:,:len(chosen)]= (ct[None,None,:]<target_axis[:,None,None])
    return history.transpose(1,0,2).astype(np.float32),np.broadcast_to(valid,(4,n)).copy(),commands.transpose(1,0,2),mask
