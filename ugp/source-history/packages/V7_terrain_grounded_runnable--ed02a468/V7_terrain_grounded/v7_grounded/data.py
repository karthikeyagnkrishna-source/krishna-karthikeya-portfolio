"""Strict NPZ loader for calibrated, time-aligned real datasets.

Arrays have a leading sample N dimension. No train/test sampling occurs here:
create separate disjoint trajectories for train and validation.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .losses import MANDATORY_TARGETS

INPUT_KEYS=('rgbd','xyz_world','valid','patch_ids','proprio','actions','base_pose',
            'foot_xyz_world','foot_velocity_world','foot_contact','gait_phase',
            'history_xyz_world','history_rgb','history_geometry','history_response',
            'history_age','history_patch_ids','history_valid',
            'history_delta_rgbd','history_delta_xyz','history_delta_proprio')

class V7NPZDataset(Dataset):
    def __init__(self,filename:str|Path):
        self.path=Path(filename)
        if not self.path.is_file():raise FileNotFoundError(self.path)
        with np.load(self.path,allow_pickle=False) as file:
            keys=set(file.files)
            expected=set(INPUT_KEYS)|set(MANDATORY_TARGETS)
            absent=expected-keys
            if absent:raise KeyError(f'{self.path} missing mandatory keys: {sorted(absent)}')
            extra={'intrinsics','camera_from_base','future_uvd','patch_features','history_token_delta'}
            if ('patch_features' in keys) != ('history_token_delta' in keys):
                raise ValueError('Both patch_features and history_token_delta required to preserve contact visual changes')
            selected={k:np.array(file[k],copy=True) for k in keys if k in expected|extra}
        n=selected['proprio'].shape[0]
        for k,a in selected.items():
            if a.shape[0]!=n:raise ValueError(f'Inconsistent sample count for {k}: {a.shape[0]} vs {n}')
            if a.dtype.kind not in 'biuf':raise TypeError(f'{k}: unsupported dtype {a.dtype}')
            if a.dtype.kind=='f' and not np.isfinite(a).all():raise ValueError(f'{k}: NaN or inf values')
        if selected['rgbd'].shape[1]!=5:raise ValueError('five camera frames required')
        if selected['history_xyz_world'].shape[1]!=4:raise ValueError('four legs required')
        ids=selected['patch_ids']
        if not np.all(ids==ids[:,[-1],:]):raise ValueError('patch IDs must be world-registered across all five frames')
        self.x={k:torch.from_numpy(a) for k,a in selected.items() if k in INPUT_KEYS or k in ('intrinsics','camera_from_base','patch_features','history_token_delta')}
        self.y={k:torch.from_numpy(a) for k,a in selected.items() if k in MANDATORY_TARGETS or k=='future_uvd'}
        self.n=n
    def __len__(self):return self.n
    def __getitem__(self,i):
        def cast(d):
            return {k:(v[i].long() if 'ids' in k else v[i].bool() if k.endswith('valid') else v[i].float()) for k,v in d.items()}
        return cast(self.x),cast(self.y)
