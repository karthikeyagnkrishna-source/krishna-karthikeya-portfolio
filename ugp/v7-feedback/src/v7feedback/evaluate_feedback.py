"""Matched 96-case rerun with exact force exports and separately reported rounds."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import atomic_json, atomic_save, nearest
from foundation import load_models, load_record, sha256
from models import Hybrid
from evaluate import export_case, fit_causal_physics, prediction_record
from feedback_models import FeedbackModel
from provenance import input_digest

VARIANTS = ('seed', 'one_pass', 'round1', 'round2', 'round3')


def export_feedback_case(meta, variant, batch, prediction, horizons):
    # Legacy export_case reads force from batch; replace only its prediction input.
    predicted_batch = dict(batch, force=prediction['force'])
    tensors = {k: v for k, v in prediction.items() if isinstance(v, torch.Tensor)}
    if 'target_r' not in tensors:
        raise ValueError('Exporter requires target foot geometry calculated outside model.forward')
    return export_case(meta, variant, predicted_batch, tensors, horizons)


def summarize(combined, directory):
    summaries = []
    for variant, tables in combined.items():
        frames = {k: pd.DataFrame(v) for k, v in tables.items()}
        for name, frame in frames.items():
            frame.to_csv(directory / f'{variant}_{name}_exact.csv', index=False)
        for quantity in ('angle', 'speed', 'torque'):
            for keys, g in frames['joint'].groupby(['split', 'adaptation_gap_s', 'horizon_s', 'leg', 'joint']):
                e = g[f'{quantity}_signed_error'].values
                summaries.append(dict(model=variant, split=keys[0], adaptation_gap_s=keys[1], horizon_s=keys[2],
                    leg=keys[3], signal=keys[4], quantity=quantity, MAE=float(np.mean(abs(e))), RMSE=float(np.sqrt(np.mean(e**2))), n=len(e)))
        for keys, g in frames['body'].groupby(['split', 'adaptation_gap_s', 'horizon_s', 'signal']):
            e = g['signed_error'].values
            summaries.append(dict(model=variant, split=keys[0], adaptation_gap_s=keys[1], horizon_s=keys[2], signal=keys[3],
                quantity='body', MAE=float(np.mean(abs(e))), RMSE=float(np.sqrt(np.mean(e**2))), n=len(e)))
        for keys, g in frames['force'].groupby(['split', 'adaptation_gap_s', 'horizon_s', 'leg']):
            for quantity, column in [('force_vector', 'vector_error_N'), ('foot_position', 'foot_position_error_m')]:
                e = g[column].values
                summaries.append(dict(model=variant, split=keys[0], adaptation_gap_s=keys[1], horizon_s=keys[2], leg=keys[3],
                    quantity=quantity, MAE=float(e.mean()), RMSE=float(np.sqrt((e**2).mean())),
                    mean_force_cosine=float(g['cosine'].mean()), valid_cosines=int(g['cosine'].count()), n=len(e)))
        per_trajectory, contact = [], []
        for quantity in ('angle', 'speed', 'torque'):
            for keys, g in frames['joint'].groupby(['split', 'adaptation_gap_s', 'environment', 'trajectory', 'leg', 'joint']):
                per_trajectory.append(dict(model=variant, split=keys[0], adaptation_gap_s=keys[1], environment=keys[2],
                    trajectory=keys[3], leg=keys[4], signal=keys[5], quantity=quantity, MAE=float(g[f'{quantity}_absolute_error'].mean())))
            for keys, g in frames['joint'].groupby(['split', 'horizon_s', 'leg', 'joint', 'actual_contact']):
                contact.append(dict(model=variant, split=keys[0], horizon_s=keys[1], leg=keys[2], joint=keys[3],
                    actual_contact=keys[4], quantity=quantity, MAE=float(g[f'{quantity}_absolute_error'].mean())))
        for keys, g in frames['body'].groupby(['split', 'adaptation_gap_s', 'environment', 'trajectory', 'signal']):
            per_trajectory.append(dict(model=variant, split=keys[0], adaptation_gap_s=keys[1], environment=keys[2],
                trajectory=keys[3], signal=keys[4], quantity='body', MAE=float(g['absolute_error'].mean())))
        for keys, g in frames['force'].groupby(['split', 'adaptation_gap_s', 'environment', 'trajectory', 'leg']):
            per_trajectory.append(dict(model=variant, split=keys[0], adaptation_gap_s=keys[1], environment=keys[2],
                trajectory=keys[3], leg=keys[4], quantity='force_vector', MAE=float(g['vector_error_N'].mean()),
                mean_force_cosine=float(g['cosine'].mean()), valid_cosines=int(g['cosine'].count())))
        pd.DataFrame(per_trajectory).to_csv(directory / f'{variant}_trajectory_summary.csv', index=False)
        pd.DataFrame(contact).to_csv(directory / f'{variant}_contact_swing_summary.csv', index=False)
        with pd.ExcelWriter(directory / f'{variant}_all_tests_exact.xlsx', engine='openpyxl') as writer:
            for name, frame in frames.items():
                sheet_name = name.capitalize() + ' exact'
                frame.to_excel(writer, sheet_name=sheet_name, index=False)
                ws = writer.sheets[sheet_name]
                ws.freeze_panes = 'A2'
                ws.auto_filter.ref = ws.dimensions
                for column in ws.columns:
                    ws.column_dimensions[column[0].column_letter].width = min(32, max(14, len(str(column[0].value)) + 2))
                    if column[0].value == 'cosine':
                        for cell in column[1:]:
                            cell.number_format = '0.000000'
            notes = ['Angles deg; speeds rad/s; torques Nm; forces N. Per-joint actual/predicted values are exact target-time samples.',
                     'Round means repeated revision of the SAME target time, not another time step.',
                     'Seed and one-pass have identical legs/body for the same input. One-pass is an independently saved Stage A checkpoint.',
                     'Matched original V7/V7-F causal local adaptation; all leg/body/feedback correction weights remain fixed during tests.',
                     'Gap limits adaptation recency; recent inference history is still present. Future recorded controls assumed known.',
                     'Strict unseen: no optimization and no heldout scaler/physics fit. Round choice uses temporal validation only.']
            pd.DataFrame({'note': notes}).to_excel(writer, sheet_name='How to read', index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(directory / 'summary.csv', index=False)
    summary.to_csv(directory / 'round_comparison.csv', index=False)


def evaluate_all(rt, output, parent, fk, manifest, report):
    output, parent = Path(output), Path(parent)
    directory = output / 'evaluation'
    directory.mkdir(exist_ok=True)
    checkpoint_hashes = {f'{b}_{s}': sha256(output / 'checkpoints' / b / f'{s}_best.pt')
                         for b in ('short', 'long') for s in ('one_pass', 'feedback')}
    records = [load_record(rt, t, e, r) for t, pairs in rt['EXPANDED_TRAIN_RUNS'].items() for e, r in pairs]
    rt['TRAIN_RECORDS'] = records
    rt['REGIME_THRESHOLDS'] = rt['learn_global_regime_thresholds'](records)
    centers, weights, counts = rt['learn_regime_centers_and_weights'](records, rt['REGIME_THRESHOLDS'])
    rt.update(REGIME_CENTERS=centers, REGIME_CLASS_WEIGHTS=weights, REGIME_COUNTS=counts)
    rt['_legacy_fit_causal_physics'] = rt['fit_causal_physics']
    rt['fit_causal_physics'] = lambda d, start, end, thresholds=None: fit_causal_physics(rt, d, start, end)
    for rec in records:
        rec['physics_prior_v7'] = fit_causal_physics(rt, rec['data'], float(rec['data']['time'][0]), min(rec['ttests']))
    diagnostics = [r for r in records if len(r['ttests']) == 4]
    by_terrain = {r['terrain_type']: r for r in diagnostics}
    foundations, models, scales, selected = {}, {}, {}, {}
    for branch in ('short', 'long'):
        foundations[branch] = load_models(rt, records[0], branch)
        scales[branch] = {k: v.to(rt['DEVICE']) for k, v in torch.load(output / f'{branch}_scales.pt', map_location='cpu', weights_only=False).items()}
        seed = Hybrid(copy.deepcopy(fk))
        seed.load_state_dict(torch.load(parent / 'checkpoints' / branch / 'joint_best.pt', map_location='cpu', weights_only=False)['model'])
        models[branch] = {}
        for stage in ('one_pass', 'feedback'):
            model = FeedbackModel(seed).to(rt['DEVICE'])
            model.load_state_dict(torch.load(output / 'checkpoints' / branch / f'{stage}_best.pt', map_location=rt['DEVICE'], weights_only=False)['model'])
            models[branch][stage] = model.eval().requires_grad_(False)
        selected[branch] = json.loads((output / f'{branch}_round_selection.json').read_text())['selected_round']
    jobs = [(r, r, i, tt, 'familiar' if gap == 0 else 'time_gap', gap)
            for r in diagnostics for i, tt in enumerate(r['ttests'], 1) for gap in (0, 4, 7, 9, 10)]
    for terrain, (env, run) in rt['STRICT_HOLDOUTS'].items():
        rec = load_record(rt, terrain, env, run, holdout=True)
        t = rec['data']['time']
        dt = np.median(np.diff(t))
        valid = np.flatnonzero((np.arange(len(t)) >= int(round(4 / dt)) - 1) &
                              (np.arange(len(t)) < len(t) - int(round(1 / dt))) & (rec['visual']['visual_index'] >= 0))
        start = max(float(t[valid[0]]) + 1, 20)
        if start + 26 > float(t[valid[-1]]):
            start = float(t[valid[-1]]) - 26
        if start < float(t[valid[0]]):
            raise ValueError('Unseen recording too short')
        for i, delta in enumerate((0, 7, 16, 26), 1):
            tt = float(t[valid[np.argmin(abs(t[valid] - start - delta))]])
            jobs.append((rec, by_terrain[terrain], i, tt, 'unseen', 0))
    if len(jobs) != 96:
        raise ValueError('Expected the original 96 evaluation cases')
    # Global force regimes use every train recording; actual visual queries use
    # diagnostic and heldout recordings. Bind both sets before accepting cases.
    inputs = [rt['ANYMAL_D_URDF']]
    inputs.extend(rt['CACHE_ROOT'] / f"{r['environment']}_{r['run']}_physics.npz" for r in records)
    seen = {(r['environment'], r['run']) for r, *_ in jobs}
    for env, run in sorted(seen):
        inputs.extend(rt['CACHE_ROOT'] / f'{env}_{run}_{suffix}' for suffix in
                      ('physics.npz', 'frame_manifest.npz', 'v7_dinov3.npy', 'v7_convnext.npy', 'v7_utonia.npy', 'v7_waft.npy'))
    sources = input_digest(inputs)
    atomic_json(directory / 'input_provenance.json', sources)
    signature = hashlib.sha256(json.dumps(dict(manifest=manifest, checkpoints=checkpoint_hashes, inputs=sources), sort_keys=True).encode()).hexdigest()
    for job_id, (rec, source, test_id, tt, split, gap) in enumerate(jobs):
        path = directory / f'case_{job_id:03d}.pt'
        if path.exists():
            if torch.load(path, weights_only=False)['signature'] != signature:
                raise ValueError('Saved evaluation belongs to different weights or code')
            continue
        meta = dict(environment=rec['environment'], trajectory=rec['run'], test_id=test_id, split=split,
                    requested_test_time_s=tt, adaptation_gap_s=gap, adaptation_cutoff_s=None if split == 'unseen' else tt - gap,
                    new_head_local_adaptation=False)
        report('evaluating', case=job_id + 1, cases=96, **meta)
        rows = {v: dict(joint=[], body=[], force=[]) for v in VARIANTS}
        for branch in ('short', 'long'):
            bg, fg, horizons = foundations[branch]
            fs = rt['make_stats_between'](source['data'], float(source['data']['time'][0]), min(source['ttests']))
            fp = source['physics_prior_v7']
            if split == 'unseen':
                base, fh, stats, physics = bg, fg, fs, fp
            else:
                cutoff = tt - gap
                context, hop, half_life = (.5, .3, 5) if branch == 'short' else (4, 1, 6)
                prior_copy = copy.deepcopy(bg).requires_grad_(True)
                base, stats, physics, _ = rt['adapt_v7'](prior_copy, rec['data'], rec['visual'], cutoff,
                    context, hop, horizons, half_life, 4, 781 + job_id, branch)
                base.eval().requires_grad_(False)
                del prior_copy
                fcopy = copy.deepcopy(fg).requires_grad_(True)
                fh, _, _, _ = rt['adapt_v7f_head'](fcopy, bg, rec, cutoff, context, hop, horizons, half_life, 1781 + job_id, branch)
                fh.eval().requires_grad_(False)
                del fcopy
            record = prediction_record(rt, rec, nearest(rec['data']['time'], tt), base, bg, fh, stats, physics, fs, fp, horizons)
            batch = {k: v[None].to(rt['DEVICE']) for k, v in record.items()}
            batch['horizon'] = torch.tensor(horizons, device=rt['DEVICE'])
            with torch.no_grad():
                seed_out = models[branch]['one_pass'](batch, scales[branch], 'seed')
                one = models[branch]['one_pass'](batch, scales[branch], 'one_pass')
                feedback = models[branch]['feedback'](batch, scales[branch], 'feedback')
                target_r = fk.to(rt['DEVICE'])(batch['target_leg'][..., :3])[0]
                torch.testing.assert_close(seed_out['leg'], one['leg'], atol=0, rtol=0)
                torch.testing.assert_close(seed_out['body'], one['body'], atol=0, rtol=0)
            predictions = dict(seed=seed_out, one_pass=one, **{f'round{i}': step for i, step in enumerate(feedback['rounds'], 1)})
            for variant, prediction in predictions.items():
                prediction['target_r'] = target_r
                exported = export_feedback_case(dict(meta, selected_round_validation=selected[branch],
                                                     refinement_round=int(variant[-1]) if variant.startswith('round') else 0 if variant == 'seed' else 1),
                                                variant, batch, prediction, horizons)
                for name in rows[variant]:
                    rows[variant][name].extend(exported[name])
            if split != 'unseen':
                del base, fh
            torch.cuda.empty_cache()
        atomic_save(path, dict(signature=signature, rows=rows))
    combined = {v: dict(joint=[], body=[], force=[]) for v in VARIANTS}
    for job_id in range(96):
        packet = torch.load(directory / f'case_{job_id:03d}.pt', weights_only=False)
        if packet['signature'] != signature:
            raise ValueError('Mismatched case provenance')
        for v in VARIANTS:
            for name in combined[v]:
                combined[v][name].extend(packet['rows'][v][name])
    summarize(combined, directory)
    atomic_json(directory / 'protocol.json', dict(jobs=96, variants=list(VARIANTS), holdouts=manifest['heldout_trajectories'],
        selected_round_validation=selected, checkpoint_hashes=checkpoint_hashes, signature=signature,
        input_provenance_sha256=sha256(directory / 'input_provenance.json'),
        local_correction_head_epochs=0, original_base_local_epochs=4, original_force_local_epochs=5,
        note='Matched global correction heads. This is a new matched rerun, not identical to parent Excel values with five local correction-head epochs.'))
