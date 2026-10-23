import json
import os
import hashlib
import collections
from pathlib import Path
import subprocess
import datetime

R=Path('/path/to/ugp/merged_feelwm_v7_runs')
out={'snapshot_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':{},'checkpoint_metadata':{},'manifest_counts':{},'extra_inventory':[],'errors':[]}
def add(p):
    try:
        raw=p.read_bytes();out['files'][str(p)]={'size':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'text':raw.decode('utf-8',errors='replace')}
    except Exception as e:out['errors'].append([str(p),repr(e)])
for p in (R/'tartanground_k1_k5_leg_compare'/'source_backup').glob('*.py'):add(p)
for p in (R/'tartanground_k1_k5_leg_compare'/'prepared').glob('*.json'):add(p)
for p in (R/'tartanground_k1_k5_leg_compare'/'prepared').glob('*.jsonl'):
    c=collections.Counter();n=0
    for line in p.open():
        n+=1
        rec=json.loads(line)
        ident=str(rec.get('environment',''))+'/'+str(rec.get('trajectory',''))
        c[ident]+=1
    out['manifest_counts'][str(p)]={'rows':n,'by_trajectory_fields':dict(c)}
root=Path('/path/to/ugp/V7F_4080_Run')
exclude={'raw','raw_tartanground','cache','cache_4branch','.git','__pycache__','v7_external','relay_archives','hf_cache','utonia_py310','images'}
if root.exists():
    for dr,ds,fs in os.walk(root):
        if len(Path(dr).relative_to(root).parts)>=4:ds[:]=[]
        else:ds[:]=[x for x in ds if x not in exclude]
        for name in fs:
            p=Path(dr)/name;s=p.stat().st_size
            if p.suffix in {'.pt','.json','.csv','.log','.ipynb','.py','.zip'}:out['extra_inventory'].append({'path':str(p),'bytes':s})
            if p.suffix in {'.json','.log','.ipynb','.py','.csv'} and s<2000000:add(p)
try:
    import torch
    out['runtime']={'torch':torch.__version__,'cuda_build':torch.version.cuda}
    for stage in ['tartanground_k1_k5_leg_compare_10ep_test10','tartanground_k1_k5_leg_compare']:
        for arch in ['baseline','per_leg']:
            for k in ['k1','k5']:
                for fn in ['latest.pt','best.pt']:
                    p=R/stage/arch/k/'checkpoints'/fn
                    if not p.exists():continue
                    ck=torch.load(p,map_location='cpu',mmap=True,weights_only=False)
                    sd=ck.get('model',ck.get('model_state_dict',{}))
                    out['checkpoint_metadata'][str(p)]={'bytes':p.stat().st_size,'keys':list(ck),'epoch':ck.get('epoch'),'step':ck.get('step',ck.get('global_step')),'best_val':ck.get('best_val'),'num_state_tensors':len(sd),'state_numel':sum(t.numel() for t in sd.values() if isinstance(t,torch.Tensor)),'tensor_shapes':{n:list(t.shape) for n,t in sd.items() if isinstance(t,torch.Tensor)},'optimizer_groups':[{q:v for q,v in g.items() if q!='params'} for g in ck.get('optimizer',{}).get('param_groups',[])]}
                    del ck
except Exception as e:out['errors'].append(['torch_checkpoint_metadata',repr(e)])
print(json.dumps(out,default=str))
