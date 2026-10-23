"""Attach exact force targets to immutable parent caches; preserve their split."""
import json
from pathlib import Path

import numpy as np
import torch

from data import atomic_json, atomic_save
from foundation import sha256


def aligned_force(target_time, time, physics, saved_gt):
    idx = np.searchsorted(time, target_time)
    if np.any(idx >= len(time)) or not np.allclose(time[idx], target_time, atol=1e-7, rtol=0):
        raise ValueError('Force target time does not match a recorded sample')
    result = physics[idx, :, 9:12]
    if result.shape != saved_gt.shape or not np.allclose(result, saved_gt, atol=1e-3, rtol=1e-5):
        raise ValueError('Original force labels disagree with recorded target-time force')
    return np.array(result, dtype=np.float32, copy=True)


def load_training_cache(rt, parent, output, branch, report):
    parent, output = Path(parent), Path(output)
    pm = json.loads((parent / f'{branch}_manifest.json').read_text())
    wanted = sorted([list(p) for pairs in rt['EXPANDED_TRAIN_RUNS'].values() for p in pairs])
    if sorted(pm['training_trajectories']) != wanted or len(wanted) != 28 or len(set(map(tuple, wanted))) != 28:
        raise ValueError('Parent cache must contain the exact 28 unique training trajectories')
    holdouts = set(tuple(p) for p in rt['STRICT_HOLDOUTS'].values())
    if holdouts & set(map(tuple, wanted)):
        raise ValueError('Strict holdout entered training')
    source_root = rt['V7F_FEATURE_RECOVERY_ROOT'] / ('SHORT__0_1s' if branch == 'short' else 'LONG__0_2___1_0s')
    base_hash = sha256(rt['OUT_ROOT'] / ('V7_PRIOR_SHORT.pt' if branch == 'short' else 'V7_PRIOR_LONG.pt'))
    force_hash = sha256(rt['PERSIST_ROOT'] / 'V7F_force_specialist' / ('V7F_FORCE_HEAD_SHORT.pt' if branch == 'short' else 'V7F_FORCE_HEAD_LONG.pt'))
    parts, provenance = [], []
    for meta in pm['trajectories']:
        env, run = meta['environment'], meta['run']
        if [env, run] not in wanted:
            raise ValueError('Unexpected trajectory in parent cache')
        pp = parent / 'cache' / branch / f'{env}_{run}.pt'
        raw = rt['CACHE_ROOT'] / f'{env}_{run}_physics.npz'
        sources = list(source_root.glob(f'*_{env}_{run}.pt'))
        if len(sources) != 1:
            raise ValueError('Expected one original frozen force-feature cache')
        sig = dict(schema='v7feedback_cache_v1', parent_cache_sha256=sha256(pp),
                   raw_sha256=sha256(raw), force_labels_sha256=sha256(sources[0]),
                   base_sha256=base_hash, force_sha256=force_hash)
        path = output / 'cache' / branch / f'{env}_{run}.pt'
        packet = torch.load(path, weights_only=False) if path.exists() else None
        if packet is None or packet['signature'] != sig:
            prior = torch.load(pp, map_location='cpu', weights_only=False)
            if prior['signature']['raw_sha256'] != sig['raw_sha256']:
                raise ValueError('Recorded physics changed since parent cache generation')
            if prior['signature']['base_sha256'] != base_hash or prior['signature']['force_sha256'] != force_hash:
                raise ValueError('Foundation weights changed since parent cache generation')
            t = dict(prior['tensors'])
            source = torch.load(sources[0], map_location='cpu', weights_only=False)
            if len(source['records']) != len(t['x']):
                raise ValueError('Parent and original feature windows differ')
            saved_gt = torch.stack([r['gtF'] for r in source['records']]).numpy()
            with np.load(raw, allow_pickle=False) as data:
                force = aligned_force(t['target_time'].numpy(), data['time'], data['P'], saved_gt)
            t['target_force'] = torch.from_numpy(force)
            packet = dict(signature=sig, tensors=t)
            atomic_save(path, packet)
        t = packet['tensors']
        if not t['train'].any() or not t['val'].any() or (t['train'] & t['val']).any():
            raise ValueError('Invalid parent temporal split')
        if t['target_time'][t['train']].max() >= t['target_time'][t['val']].min():
            raise ValueError('Train and validation target times overlap')
        t = dict(t, trajectory=torch.full((len(t['x']),), len(parts), dtype=torch.long),
                 terrain=torch.full((len(t['x']),), list(rt['EXPANDED_TRAIN_RUNS']).index(meta['terrain']), dtype=torch.long))
        parts.append(t)
        provenance.append(dict(environment=env, run=run, train=int(t['train'].sum()),
                               validation=int(t['val'].sum()), **sig))
        report('cache', branch=branch, trajectories_completed=len(parts), trajectories_total=28, trajectory=f'{env}/{run}')
    if len(parts) != 28 or len({(p['environment'], p['run']) for p in provenance}) != 28:
        raise ValueError('Incomplete or repeated trajectory cache')
    tensors = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    tensors['horizon'] = torch.tensor([.1] if branch == 'short' else [round(i / 10, 1) for i in range(2, 11)])
    atomic_json(output / f'{branch}_manifest.json', dict(trajectories=provenance, training_trajectories=wanted,
        parent_manifest_sha256=sha256(parent / f'{branch}_manifest.json'),
        note='Parent temporal split preserved. The reused V7 foundation was trained on full original eligible ranges; strict trajectory holdouts remain excluded. Future recorded controls are assumed known.'))
    return tensors


def fit_force_scales(tensors, original_scales):
    scales = {k: v.detach().cpu().clone() for k, v in original_scales.items()}
    force = tensors['target_force'][tensors['train']].reshape(-1, 3)
    scales.update(target_force_mean=force.mean(0), target_force_std=force.std(0, unbiased=False).clamp_min(1.))
    return scales


def checked_checkpoint(path, signature, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck['signature'] != signature:
        raise ValueError('Checkpoint provenance changed; use a new output directory')
    return ck
