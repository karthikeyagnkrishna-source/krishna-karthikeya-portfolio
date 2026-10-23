"""Inference from a trained checkpoint and causal current-observation NPZ only.

The observation NPZ may contain ONLY current/past sensor data and candidate action plan:
no future labels are required or read.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from v7_grounded import V7Config,V7GroundedWorldModel
from v7_grounded.data import INPUT_KEYS

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ckpt',required=True);p.add_argument('--observations',required=True)
    p.add_argument('--output',default='predictions.npz');p.add_argument('--batch',type=int,default=4)
    p.add_argument('--cpu',action='store_true')
    args=p.parse_args()
    dev=torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    ck=torch.load(args.ckpt,map_location=dev,weights_only=False)
    model=V7GroundedWorldModel(V7Config(**ck['cfg'])).to(dev)
    model.load_state_dict(ck['model'],strict=True);model.eval()
    with np.load(args.observations,allow_pickle=False) as f:
        needed=set(INPUT_KEYS)|({'patch_features','history_token_delta'} if model.cfg.feature_dim else set())
        absent=needed-set(f.files)
        if absent:raise KeyError(f'Missing causal inference inputs: {sorted(absent)}')
        keys=set(INPUT_KEYS)|set(['intrinsics','camera_from_base'])
        if model.cfg.feature_dim: keys.update(['patch_features','history_token_delta'])
        data={k:np.array(f[k]) for k in f.files if k in keys}
    n=data['rgbd'].shape[0]
    res={}
    with torch.inference_mode():
        for j in range(0,n,args.batch):
            x={k:torch.as_tensor(v[j:j+args.batch],device=dev,
                 dtype=torch.long if 'ids' in k else torch.bool if k.endswith('valid') else torch.float32)
               for k,v in data.items()}
            p=model(x)
            for k,v in p.items():
                if k=='patch_logits':v=v.softmax(-1)
                res.setdefault(k,[]).append(v.detach().cpu().numpy())
    output={k:np.concatenate(v,axis=0) for k,v in res.items()}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,**output)
    print(f'Predicted {n} causal observations into {args.output}; output keys: {sorted(output)}')
if __name__=='__main__':main()
