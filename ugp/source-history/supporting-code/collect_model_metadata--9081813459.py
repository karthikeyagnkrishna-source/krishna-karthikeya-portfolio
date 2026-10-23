from pathlib import Path
import json
import math
import hashlib
import struct
import torch

base=Path('/path/to/ugp/V7F_4080_Run')
out={'models':{},'foundation_candidates':[],'configs':{},'errors':[]}
paths=[base/'storage/results'/f'V7_PRIOR_{x}.pt' for x in ['SHORT','LONG']]
paths += [base/'storage/V7F_force_specialist'/f'V7F_FORCE_HEAD_{x}.pt' for x in ['SHORT','LONG']]
paths += list((base/'storage/V8_contactmap').glob('*/*.pt'))
for p in paths:
    try:
        ck=torch.load(p,map_location='cpu',mmap=True,weights_only=True)
        sd=ck.get('model',ck.get('state_dict',ck))
        shapes={k:list(v.shape) for k,v in sd.items() if isinstance(v,torch.Tensor)}
        out['models'][str(p)]={'bytes':p.stat().st_size,'num_tensors':len(shapes),'state_numel':sum(math.prod(x) for x in shapes.values()),'tensor_shapes':shapes}
        del ck
    except Exception as e:out['errors'].append([str(p),repr(e)])
hf_roots=[Path('/path/to/ugp/hub'),base/'storage/.cache/huggingface/hub',base/'storage/.work/hf_cache/hub',base/'hf_cache/hub']
for root in hf_roots:
    if not root.exists():continue
    for dr in root.glob('models--timm--*dinov3*'):
        if not any(x in dr.name for x in ['vit_large_patch16','convnext_large']):continue
        for p in dr.glob('snapshots/*/*'):
            if p.suffix=='.json':out['configs'][str(p)]=p.read_text()
            if p.suffix=='.safetensors':
                with p.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];head=json.loads(f.read(n))
                shapes={k:v['shape'] for k,v in head.items() if k!='__metadata__'}
                out['models'][str(p)]={'bytes':p.stat().st_size,'num_tensors':len(shapes),'state_numel':sum(math.prod(x) for x in shapes.values()),'tensor_shapes':shapes,'kind':'serialized backbone tensor total, may include unused heads/buffers'}
for p in [base/'storage/.work/v7_external/WAFT/ckpts/tar-c-t.pth',base/'storage/.work/v7_external/WAFT/depth-anything-ckpts/depth_anything_v2_vits.pth',base/'storage/.cache/utonia/ckpt/utonia.pth',Path('/path/to/ugp/utonia.pth')]:
    if not p.exists():continue
    out['foundation_candidates'].append({'path':str(p),'bytes':p.stat().st_size})
    try:
        ck=torch.load(p,map_location='cpu',mmap=True,weights_only=True)
        sd=ck.get('state_dict',ck.get('model',ck))
        shapes={k:list(v.shape) for k,v in sd.items() if isinstance(v,torch.Tensor)}
        out['models'][str(p)]={'bytes':p.stat().st_size,'num_tensors':len(shapes),'state_numel':sum(math.prod(x) for x in shapes.values()),'tensor_shapes':shapes,'kind':'serialized backbone tensor total, may include unused heads/buffers'}
        del ck
    except Exception as e:out['errors'].append([str(p),repr(e)])
print(json.dumps(out))
