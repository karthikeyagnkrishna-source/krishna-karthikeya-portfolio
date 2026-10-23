"""Build leg/body training records from already-persisted V7-F latents."""
import json
from pathlib import Path

import numpy as np
import torch

from foundation import load_models, load_record, sha256


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def atomic_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temp)
    temp.replace(path)


def nearest(times, value):
    j = min(int(np.searchsorted(times, value)), len(times) - 1)
    return j - 1 if j and abs(times[j - 1] - value) < abs(times[j] - value) else j


@torch.inference_mode()
def reconstruct(base, force_head, features, data, stats, horizons, device):
    """Original baseline + original standardized residual, at saved anchor times."""
    x = torch.stack([r['x'] for r in features]).to(device).float()
    z = x[:, :, 0, :768]
    sb = base.state_body_head(z)
    tf = base.tau_force_head(x)
    offsets = [int(round(h / np.median(np.diff(data['time'])))) for h in horizons]
    idxs = [nearest(data['time'], f['target_end'] - max(horizons)) for f in features]
    target_idx = np.asarray(idxs)[:, None] + np.asarray(offsets)[None]
    current = torch.tensor(data['P'][idxs], device=device)
    bp = current[:, None].expand(-1, len(horizons), -1, -1).clone()
    h = torch.tensor(horizons, device=device).view(1, -1, 1, 1)
    bp[..., :3] += h * current[:, None, :, 3:6]
    pstd = torch.tensor(stats['P_std'], device=device)
    residual = torch.cat([sb[..., :24].reshape(-1, len(horizons), 4, 6), tf[..., :3]], -1)
    leg = bp[..., :9] + residual * pstd[:9]
    body = torch.tensor(data['B'][idxs], device=device)[:, None] + sb[..., 24:] * torch.tensor(stats['B_std'], device=device)
    bF = torch.stack([r['baseF'] for r in features]).to(device)
    contact = torch.stack([r['pcontact'] for r in features]).to(device)
    regime = torch.stack([r['regime'] for r in features]).to(device)
    ps = torch.stack([r['pstd_force'] for r in features]).to(device).view(-1, 1, 1, 3)
    force = force_head(x, bF, ps, contact, regime)['force']
    target = data['P'][target_idx]
    saved_gt = torch.stack([r['gtF'] for r in features]).numpy()
    if not np.allclose(saved_gt, target[..., 9:12], atol=1e-3, rtol=1e-5):
        raise RuntimeError('Saved latent target timestamps disagree with raw target force')
    return dict(x=x.half().cpu(), base_leg=leg.cpu(), base_body=body.cpu(), force=force.cpu(),
                contact=contact.cpu(), target_leg=torch.tensor(target[..., :9]),
                target_body=torch.tensor(data['B'][target_idx]),
                target_contact=torch.tensor(data['contact'][target_idx]),
                anchor=torch.tensor(data['time'][idxs], dtype=torch.float64),
                target_time=torch.tensor(data['time'][target_idx], dtype=torch.float64))


def build_cache(rt, output, branch, foundation_manifest, status):
    cache_root = Path(output) / 'cache' / branch
    source_root = rt['V7F_FEATURE_RECOVERY_ROOT'] / ('SHORT__0_1s' if branch == 'short' else 'LONG__0_2___1_0s')
    code_hash = {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}
    model_path = rt['OUT_ROOT'] / ('V7_PRIOR_SHORT.pt' if branch == 'short' else 'V7_PRIOR_LONG.pt')
    force_path = rt['PERSIST_ROOT'] / 'V7F_force_specialist' / ('V7F_FORCE_HEAD_SHORT.pt' if branch == 'short' else 'V7F_FORCE_HEAD_LONG.pt')
    signature = dict(schema='v7legs_body_cache_v1', branch=branch, base_sha256=sha256(model_path),
                     force_sha256=sha256(force_path), implementation=code_hash)
    # Code changes in training/evaluation do not invalidate already-correct feature reconstruction.
    signature['implementation'] = {k: v for k, v in code_hash.items() if k in ('foundation.py', 'data.py')}
    first = load_record(rt, 'forest', 'ForestEnv', 'P2000')
    base, force_head, horizons = load_models(rt, first, branch)
    base_fp = rt['model_state_fingerprint'](base)
    all_parts, meta = [], []
    for terrain, runs in rt['EXPANDED_TRAIN_RUNS'].items():
        for env, run in runs:
            files = list(source_root.glob(f'*_{env}_{run}.pt'))
            if len(files) != 1:
                raise RuntimeError(f'Expected one reusable source cache for {env}/{run}: {files}')
            source = torch.load(files[0], map_location='cpu', weights_only=False)
            if source['signature']['base_fp'] != base_fp:
                raise RuntimeError('Frozen feature cache belongs to a different V7 foundation')
            if source['signature']['environment'] != env or source['signature']['run'] != run:
                raise RuntimeError('Source feature identity mismatch')
            path = cache_root / f'{env}_{run}.pt'
            sig = dict(signature, source_signature=source['signature'],
                       raw_sha256=sha256(rt['CACHE_ROOT'] / f'{env}_{run}_physics.npz'))
            packet = torch.load(path, map_location='cpu', weights_only=False) if path.exists() else None
            if packet is None or packet['signature'] != sig:
                rec = load_record(rt, terrain, env, run)
                data = rec['data']
                end = float(source['signature']['cutoff'])
                stats = rt['make_stats_between'](data, float(data['time'][0]), end)
                parts = []
                for start in range(0, len(source['records']), 16):
                    parts.append(reconstruct(base, force_head, source['records'][start:start + 16], data, stats, horizons, rt['DEVICE']))
                tensors = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
                boundary = float(data['time'][0] + 0.8 * (end - data['time'][0]))
                train = tensors['target_time'][:, -1] <= boundary - 1.0 + 1e-9
                val = tensors['target_time'][:, 0] > boundary + 1e-9
                if not train.any() or not val.any():
                    raise RuntimeError(f'Empty causal train/validation split for {env}/{run}')
                assert tensors['target_time'][train].max() < tensors['target_time'][val].min()
                tensors.update(train=train, val=val)
                packet = dict(signature=sig, tensors=tensors, boundary=boundary)
                atomic_save(path, packet)
            t = packet['tensors']
            t = dict(t, trajectory=torch.full((len(t['x']),), len(meta), dtype=torch.long),
                     terrain=torch.full((len(t['x']),), list(rt['EXPANDED_TRAIN_RUNS']).index(terrain), dtype=torch.long))
            all_parts.append(t)
            meta.append(dict(environment=env, run=run, terrain=terrain, boundary=packet['boundary'],
                             windows=len(t['x']), train=int(t['train'].sum()), validation=int(t['val'].sum())))
            status('cache', branch=branch, trajectories_completed=len(meta), trajectories_total=28,
                   trajectory=f'{env}/{run}', train=meta[-1]['train'], validation=meta[-1]['validation'])
            del source
    tensors = {k: torch.cat([p[k] for p in all_parts]) for k in all_parts[0]}
    tensors['horizon'] = torch.tensor(horizons)
    del base, force_head
    torch.cuda.empty_cache()
    atomic_json(Path(output) / f'{branch}_manifest.json', dict(signature=signature, trajectories=meta,
                training_trajectories=foundation_manifest['training_trajectories'],
                note='Head validation is temporal; the reused foundation was trained on the original full eligible ranges. Strict unseen trajectories remain excluded from every stage. Frozen x is the original float16 cache. Recorded future controller commands are assumed known, as in the source protocol.'))
    return tensors


@torch.inference_mode()
def fit_scales(tensors, fk):
    ids = torch.nonzero(tensors['train']).flatten()
    samples = dict(leg=tensors['target_leg'][ids], body=tensors['target_body'][ids], force=tensors['force'][ids])
    feet, velocities, residuals = [], [], []
    for chunk in ids.split(256):
        leg, body = tensors['target_leg'][chunk], tensors['target_body'][chunk]
        r, J = fk(leg[..., :3])
        fv = (J @ leg[..., 3:6, None]).squeeze(-1)
        residual = body[..., None, 6:9] + torch.cross(body[..., None, 9:12].expand_as(r), r, dim=-1) + fv
        feet.append(r)
        velocities.append(fv)
        residuals.append(residual)
    samples.update(r=torch.cat(feet), fv=torch.cat(velocities), stance=torch.cat(residuals))
    floors = dict(leg=1e-3, body=1e-3, force=1.0, r=0.01, fv=0.05, stance=0.05)
    scales = {}
    for name, x in samples.items():
        flat = x.reshape(-1, x.shape[-1])
        scales[name + '_mean'] = flat.mean(0)
        scales[name + '_std'] = flat.std(0, unbiased=False).clamp_min(floors[name])
    return scales


def batch_at(tensors, indices, device):
    count = len(tensors['x'])
    return {k: (v[indices] if k != 'horizon' and v.ndim and len(v) == count else v).to(device)
            for k, v in tensors.items() if isinstance(v, torch.Tensor)}
