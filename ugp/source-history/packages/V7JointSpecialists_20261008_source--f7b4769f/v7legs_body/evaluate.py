"""Matched familiar, adaptation-gap, and strictly unseen evaluations."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import atomic_json, atomic_save, batch_at, nearest
from foundation import load_models, load_record
from models import Hybrid, objective

VARIANTS = ('base', 'leg_only', 'body_only', 'hybrid')
BODY = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'body_vx', 'body_vy', 'body_vz', 'body_wx', 'body_wy', 'body_wz']


@torch.no_grad()
def prediction_record(rt, rec, idx, base, force_base, force_head, stats, physics, force_stats, force_physics, horizons):
    """The original force specialist stays conditioned on its global V7 backbone."""
    sample = rt['make_v7_sample'](rec['data'], rec['visual'], stats, physics, idx,
                                base.history_steps, horizons, rt['normalize_arrays'](rec['data'], stats))
    ctx = rt['v7f_extract_base_context'](base, sample, stats)
    x = ctx['x']
    sb = base.state_body_head(x[:, :, 0, :768])
    tf = base.tau_force_head(x)
    ps = torch.tensor(stats['P_std'], device=rt['DEVICE'])
    bp = torch.tensor(sample['baseline_p_raw'], device=rt['DEVICE'])[None]
    bb = torch.tensor(sample['baseline_b_raw'], device=rt['DEVICE'])[None]
    leg = bp[..., :9] + torch.cat([sb[..., :24].reshape(1, len(horizons), 4, 6), tf[..., :3]], -1) * ps[:9]
    body = bb + sb[..., 24:] * torch.tensor(stats['B_std'], device=rt['DEVICE'])
    fs = rt['make_v7_sample'](rec['data'], rec['visual'], force_stats, force_physics, idx,
                             force_base.history_steps, horizons, rt['normalize_arrays'](rec['data'], force_stats))
    fc = rt['v7f_extract_base_context'](force_base, fs, force_stats)
    force_scale = torch.tensor(force_stats['P_std'][9:12], device=rt['DEVICE']).view(1, 1, 1, 3)
    force = force_head(fc['x'], fc['baseF'], force_scale, fc['pcontact'], fc['regime'])['force']
    target_index = idx + np.array(rt['horizon_offsets'](rec['data'], horizons))
    return {k: v.cpu().clone() for k, v in dict(x=x[0].half(), base_leg=leg[0], base_body=body[0],
            force=force[0], contact=ctx['pcontact'][0], target_leg=torch.tensor(sample['target_p_raw'][..., :9]),
            target_body=torch.tensor(sample['target_b_raw']), target_contact=torch.tensor(sample['target_contact_raw']),
            target_force=torch.tensor(sample['target_p_raw'][..., 9:12]),
            anchor=torch.tensor(rec['data']['time'][idx], dtype=torch.float64),
            target_time=torch.tensor(rec['data']['time'][target_index], dtype=torch.float64)).items()}


def fit_causal_physics(rt, data, start, cutoff):
    # Prevent the legacy derivative/clipping preprocessing from seeing post-cutoff observations.
    mask = data['time'] <= cutoff + 1e-9
    cropped = {k: v[mask] if np.ndim(v) and len(v) == len(mask) else v for k, v in data.items()}
    legacy = rt.get('_legacy_fit_causal_physics', rt['fit_causal_physics'])
    return legacy(cropped, start, cutoff)


def local_records(rt, rec, cutoff, branch, base, force_base, force_head, stats, physics, force_stats, force_physics):
    horizons = rt['SHORT_HORIZONS'] if branch == 'short' else rt['LONG_HORIZONS']
    context, hop, half_life = (0.5, 0.3, 5) if branch == 'short' else (4, 1, 6)
    windows, _ = rt['make_local_windows_v7'](rec['data'], rec['visual'], cutoff, context, hop, max(horizons), half_life)
    records = []
    for idx, w, _, target_end in windows:
        if target_end > cutoff + 1e-9:
            continue
        record = prediction_record(rt, rec, idx, base, force_base, force_head, stats, physics, force_stats, force_physics, horizons)
        record['weight'] = torch.tensor(w)
        records.append(record)
    if not records:
        return None
    tensors = {k: torch.stack([r[k] for r in records]) for k in records[0]}
    times = tensors['target_time']
    boundary = float(times[:, -1].min() + 0.8 * (times[:, -1].max() - times[:, -1].min()))
    tensors['train'] = times[:, -1] <= boundary - 1 + 1e-9
    tensors['val'] = times[:, 0] > boundary + 1e-9
    tensors['horizon'] = torch.tensor(horizons)
    return tensors


def adapt_head(model, variant, tensors, scales, device, stance):
    model = copy.deepcopy(model)
    if variant == 'base' or tensors is None or not tensors['train'].any() or not tensors['val'].any():
        return model.eval(), False
    model.requires_grad_(False)
    if variant in ('leg_only', 'hybrid'):
        model.legs.requires_grad_(True)
    if variant in ('body_only', 'hybrid'):
        model.body.requires_grad_(True)
    stage = {'leg_only': 'legs', 'body_only': 'body_only', 'hybrid': 'joint'}[variant]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=6e-5, weight_decay=1e-4)
    best, best_state = float('inf'), copy.deepcopy(model.state_dict())
    train = torch.nonzero(tensors['train']).flatten()
    val = torch.nonzero(tensors['val']).flatten()
    for epoch in range(5):
        model.train()
        for ids in train.split(8):
            batch = batch_at(tensors, ids, device)
            opt.zero_grad(set_to_none=True)
            out = model(batch, scales, variant)
            loss, _, _ = objective(out, batch, scales, model.fk, stage, stance)
            # Match the old causal recency weighting across the new comparison variants.
            loss = loss * batch['weight'].mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1, error_if_nonfinite=True)
            opt.step()
        model.eval()
        total, count = 0., 0
        with torch.no_grad():
            for ids in val.split(8):
                b = batch_at(tensors, ids, device)
                _, score, _ = objective(model(b, scales, variant), b, scales, model.fk, stage, stance)
                total += float(score) * len(ids)
                count += len(ids)
        if total / count < best:
            best, best_state = total / count, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model.eval(), True


def percent(error, actual):
    return float(100 * error / abs(actual)) if abs(actual) > 1e-8 else np.nan


def export_case(meta, variant, batch, prediction, horizons):
    out = {k: v.detach().cpu().numpy()[0] for k, v in prediction.items()}
    true_leg = batch['target_leg'].cpu().numpy()[0]
    true_body = batch['target_body'].cpu().numpy()[0]
    true_force = batch['target_force'].cpu().numpy()[0]
    pred_force = batch['force'].cpu().numpy()[0]
    joint_rows, body_rows, force_rows = [], [], []
    for hi, h in enumerate(horizons):
        common = dict(meta, model=variant, horizon_s=h, anchor_time_s=float(batch['anchor'][0]),
                      target_time_s=float(batch['target_time'][0, hi]))
        for li, leg in enumerate(('FL', 'FR', 'RL', 'RR')):
            for ji, joint in enumerate(('HAA', 'HFE', 'KFE')):
                row = dict(common, leg=leg, joint=joint,
                           actual_contact=float(batch['target_contact'][0, hi, li]))
                for name, offset, factor in [('angle', 0, 180 / np.pi), ('speed', 3, 1), ('torque', 6, 1)]:
                    actual = float(true_leg[hi, li, ji + offset] * factor)
                    predicted = float(out['leg'][hi, li, ji + offset] * factor)
                    err = predicted - actual
                    row.update({f'{name}_actual': actual, f'{name}_predicted': predicted,
                                f'{name}_signed_error': err, f'{name}_absolute_error': abs(err),
                                f'{name}_error_percent': percent(abs(err), actual),
                                f'{name}_near_zero_actual': abs(actual) < (0.1 if name == 'angle' else 0.01)})
                joint_rows.append(row)
            pf, tf = pred_force[hi, li], true_force[hi, li]
            pnorm, tnorm = np.linalg.norm(pf), np.linalg.norm(tf)
            cosine = float(np.dot(pf, tf) / (pnorm * tnorm)) if pnorm > 1e-8 and tnorm > 1e-8 else np.nan
            force_rows.append(dict(common, leg=leg, actual_Fx=float(tf[0]), actual_Fy=float(tf[1]), actual_Fz=float(tf[2]),
                    predicted_Fx=float(pf[0]), predicted_Fy=float(pf[1]), predicted_Fz=float(pf[2]),
                    actual_magnitude_N=float(tnorm), predicted_magnitude_N=float(pnorm),
                    vector_error_N=float(np.linalg.norm(pf - tf)),
                    vector_error_percent=percent(float(np.linalg.norm(pf - tf)), float(tnorm)), cosine=cosine,
                    foot_position_error_m=float(np.linalg.norm(out['r'][hi, li] - out['target_r'][hi, li])),
                    actual_contact=float(batch['target_contact'][0, hi, li])))
        for bi, name in enumerate(BODY):
            actual, pred = float(true_body[hi, bi]), float(out['body'][hi, bi])
            units = 'm/s^2' if bi < 3 else 'rad/s' if bi < 6 or bi >= 9 else 'm/s'
            body_rows.append(dict(common, signal=name, units=units, actual=actual, predicted=pred,
                                  signed_error=pred - actual, absolute_error=abs(pred - actual),
                                  error_percent=percent(abs(pred - actual), actual)))
    return dict(joint=joint_rows, body=body_rows, force=force_rows)


def evaluate_all(rt, output, fk, manifest, report):
    output = Path(output)
    evaldir = output / 'evaluation'
    evaldir.mkdir(exist_ok=True)
    records = [load_record(rt, t, e, r) for t, pairs in rt['EXPANDED_TRAIN_RUNS'].items() for e, r in pairs]
    rt['TRAIN_RECORDS'] = records
    # Exactly the original train-only global regimes; heldout recordings never enter this fit.
    rt['REGIME_THRESHOLDS'] = rt['learn_global_regime_thresholds'](records)
    centers, weights, counts = rt['learn_regime_centers_and_weights'](records, rt['REGIME_THRESHOLDS'])
    rt.update(REGIME_CENTERS=centers, REGIME_CLASS_WEIGHTS=weights, REGIME_COUNTS=counts)
    rt['_legacy_fit_causal_physics'] = rt['fit_causal_physics']
    rt['fit_causal_physics'] = lambda data, start, end, thresholds=None: fit_causal_physics(rt, data, start, end)
    for rec in records:
        rec['physics_prior_v7'] = fit_causal_physics(rt, rec['data'], float(rec['data']['time'][0]), min(rec['ttests']))
    diagnostic = [r for r in records if len(r['ttests']) == 4]
    by_terrain = {r['terrain_type']: r for r in diagnostic}
    global_models, heads, scales = {}, {}, {}
    for branch in ('short', 'long'):
        global_models[branch] = load_models(rt, records[0], branch)
        scales[branch] = {k: v.to(rt['DEVICE']) for k, v in torch.load(output / f'{branch}_scales.pt', weights_only=False).items()}
        heads[branch] = {}
        for variant in VARIANTS:
            model = Hybrid(copy.deepcopy(fk)).to(rt['DEVICE'])
            if variant != 'base':
                stage = {'leg_only': 'legs', 'body_only': 'body_only', 'hybrid': 'joint'}[variant]
                ck = torch.load(output / 'checkpoints' / branch / f'{stage}_best.pt', map_location=rt['DEVICE'], weights_only=False)
                model.load_state_dict(ck['model'])
            heads[branch][variant] = model.eval()
    jobs = []
    for rec in diagnostic:
        for test_id, tt in enumerate(rec['ttests'], 1):
            for gap in (0, 4, 7, 9, 10):
                jobs.append((rec, rec, test_id, tt, 'familiar' if gap == 0 else 'time_gap', gap))
    for terrain, (env, run) in rt['STRICT_HOLDOUTS'].items():
        rec = load_record(rt, terrain, env, run, holdout=True)
        t = rec['data']['time']
        dt = np.median(np.diff(t))
        valid = np.flatnonzero((np.arange(len(t)) >= int(round(4 / dt)) - 1) &
                              (np.arange(len(t)) < len(t) - int(round(1 / dt))) &
                              (rec['visual']['visual_index'] >= 0))
        start = max(float(t[valid[0]]) + 1, 20)
        if start + 26 > float(t[valid[-1]]):
            start = float(t[valid[-1]]) - 26
        if start < float(t[valid[0]]):
            raise RuntimeError('Unseen recording too short for the fixed anchor spacings')
        for test_id, delta in enumerate((0, 7, 16, 26), 1):
            tt = float(t[valid[np.argmin(abs(t[valid] - start - delta))]])
            jobs.append((rec, by_terrain[terrain], test_id, tt, 'unseen', 0))
    for job_id, (rec, source, test_id, tt, split, gap) in enumerate(jobs):
        case_path = evaldir / f'case_{job_id:03d}.pt'
        if case_path.exists():
            continue
        meta = dict(environment=rec['environment'], trajectory=rec['run'], test_id=test_id,
                    split=split, requested_test_time_s=tt, adaptation_gap_s=gap,
                    adaptation_cutoff_s=(None if split == 'unseen' else tt - gap))
        report('evaluating', case=job_id + 1, cases=len(jobs), **meta)
        rows = {v: dict(joint=[], body=[], force=[]) for v in VARIANTS}
        for branch in ('short', 'long'):
            base_global, f_global, horizons = global_models[branch]
            force_stats = rt['make_stats_between'](source['data'], float(source['data']['time'][0]), min(source['ttests']))
            force_physics = source['physics_prior_v7']
            if split == 'unseen':
                base, fh, stats, physics = base_global, f_global, force_stats, force_physics
                local = None
            else:
                cutoff = tt - gap
                context, hop, half_life = (0.5, 0.3, 5) if branch == 'short' else (4, 1, 6)
                # Use causal legacy base/force adaptation; no modification of global checkpoint objects.
                prior_copy = copy.deepcopy(base_global).requires_grad_(True)
                base, stats, physics, history_steps = rt['adapt_v7'](prior_copy, rec['data'], rec['visual'], cutoff,
                               context, hop, horizons, half_life, 4, 781 + job_id, branch)
                base.eval().requires_grad_(False)
                del prior_copy
                fcopy = copy.deepcopy(f_global).requires_grad_(True)
                fh, _, _, _ = rt['adapt_v7f_head'](fcopy, base_global, rec, cutoff, context, hop, horizons,
                                                  half_life, 1781 + job_id, branch)
                fh.eval().requires_grad_(False)
                del fcopy
                local = local_records(rt, rec, cutoff, branch, base, base_global, fh, stats, physics, force_stats, force_physics)
            idx = nearest(rec['data']['time'], tt)
            record = prediction_record(rt, rec, idx, base, base_global, fh, stats, physics, force_stats, force_physics, horizons)
            batch = {k: v[None].to(rt['DEVICE']) for k, v in record.items()}
            batch['horizon'] = torch.tensor(horizons, device=rt['DEVICE'])
            for variant in VARIANTS:
                torch.manual_seed(2781 + job_id)
                adapted, used = adapt_head(heads[branch][variant], variant, local, scales[branch], rt['DEVICE'], manifest['frame_evidence']['verified'])
                with torch.no_grad():
                    prediction = adapted(batch, scales[branch], variant)
                    prediction['target_r'] = adapted.fk(batch['target_leg'][..., :3])[0]
                exported = export_case(dict(meta, new_head_local_adaptation=used), variant, batch, prediction, horizons)
                for name in rows[variant]:
                    rows[variant][name].extend(exported[name])
                del adapted
            if split != 'unseen':
                del base, fh, local
            torch.cuda.empty_cache()
        atomic_save(case_path, rows)
    combined = {v: dict(joint=[], body=[], force=[]) for v in VARIANTS}
    for p in sorted(evaldir.glob('case_*.pt')):
        case = torch.load(p, weights_only=False)
        for variant in VARIANTS:
            for name in combined[variant]:
                combined[variant][name].extend(case[variant][name])
    summaries = []
    for variant, tables in combined.items():
        frames = {k: pd.DataFrame(v) for k, v in tables.items()}
        # Force predictions must be exactly identical across all four correction variants.
        for k in ('predicted_Fx', 'predicted_Fy', 'predicted_Fz'):
            np.testing.assert_array_equal(frames['force'][k].values,
                                          np.array([r[k] for r in combined['base']['force']]))
        for name, frame in frames.items():
            frame.to_csv(evaldir / f'{variant}_{name}_exact.csv', index=False)
        for quantity in ('angle', 'speed', 'torque'):
            for keys, group in frames['joint'].groupby(['split', 'adaptation_gap_s', 'horizon_s', 'leg', 'joint']):
                values = group[f'{quantity}_signed_error'].values
                summaries.append(dict(model=variant, split=keys[0], adaptation_gap_s=keys[1], horizon_s=keys[2], leg=keys[3],
                                      signal=keys[4], quantity=quantity, MAE=float(np.mean(abs(values))),
                                      RMSE=float(np.sqrt(np.mean(values**2))), n=len(values)))
        for keys, group in frames['body'].groupby(['split', 'adaptation_gap_s', 'horizon_s', 'signal']):
            e = group['signed_error'].values
            summaries.append(dict(model=variant, split=keys[0], adaptation_gap_s=keys[1], horizon_s=keys[2], signal=keys[3], quantity='body',
                                  MAE=float(np.mean(abs(e))), RMSE=float(np.sqrt(np.mean(e**2))), n=len(e)))
        for keys, group in frames['force'].groupby(['split', 'adaptation_gap_s', 'horizon_s', 'leg']):
            for quantity, column in [('force_vector', 'vector_error_N'), ('foot_position', 'foot_position_error_m')]:
                e = group[column].values
                summaries.append(dict(model=variant, split=keys[0], adaptation_gap_s=keys[1], horizon_s=keys[2], leg=keys[3],
                                      quantity=quantity, MAE=float(np.mean(e)), RMSE=float(np.sqrt(np.mean(e**2))),
                                      mean_force_cosine=float(group['cosine'].mean()), valid_cosines=int(group['cosine'].count()), n=len(e)))
        trajectory_rows, contact_rows = [], []
        for quantity in ('angle', 'speed', 'torque'):
            for keys, group in frames['joint'].groupby(['split', 'adaptation_gap_s', 'environment', 'trajectory', 'joint']):
                e = group[f'{quantity}_signed_error'].values
                trajectory_rows.append(dict(model=variant, split=keys[0], adaptation_gap_s=keys[1], environment=keys[2],
                                            trajectory=keys[3], joint=keys[4], quantity=quantity,
                                            MAE=float(np.mean(abs(e))), RMSE=float(np.sqrt(np.mean(e**2)))))
            for keys, group in frames['joint'].groupby(['split', 'adaptation_gap_s', 'horizon_s', 'joint', 'actual_contact']):
                contact_rows.append(dict(model=variant, split=keys[0], adaptation_gap_s=keys[1], horizon_s=keys[2], joint=keys[3],
                                         actual_contact=keys[4], quantity=quantity, MAE=float(group[f'{quantity}_absolute_error'].mean())))
        for keys, group in frames['body'].groupby(['split', 'adaptation_gap_s', 'environment', 'trajectory', 'signal']):
            e = group['signed_error'].values
            trajectory_rows.append(dict(model=variant, split=keys[0], adaptation_gap_s=keys[1], environment=keys[2],
                                        trajectory=keys[3], signal=keys[4], quantity='body',
                                        MAE=float(np.mean(abs(e))), RMSE=float(np.sqrt(np.mean(e**2)))))
        pd.DataFrame(trajectory_rows).to_csv(evaldir / f'{variant}_trajectory_summary.csv', index=False)
        pd.DataFrame(contact_rows).to_csv(evaldir / f'{variant}_contact_swing_summary.csv', index=False)
        with pd.ExcelWriter(evaldir / f'{variant}_all_tests_exact.xlsx', engine='openpyxl') as writer:
            for name, frame in frames.items():
                frame.to_excel(writer, sheet_name=name.capitalize() + ' exact', index=False)
                sheet = writer.sheets[name.capitalize() + ' exact']
                sheet.freeze_panes = 'A2'
                sheet.auto_filter.ref = sheet.dimensions
                for col in sheet.columns:
                    sheet.column_dimensions[col[0].column_letter].width = min(32, max(14, len(str(col[0].value)) + 2))
            pd.DataFrame([dict(note='Angles deg; speeds rad/s; torques Nm. Future recorded controller commands are assumed known.'),
                          dict(note='Time gap limits local adaptation recency; inference history still ends at the test anchor.'),
                          dict(note='Strict unseen: no local training and no heldout scaler/physics fit.'),
                          dict(note='Base is the matched rerun for this experiment, not a claim of identical historical aggregate scores.')]).to_excel(writer, sheet_name='How to read', index=False)
    pd.DataFrame(summaries).to_csv(evaldir / 'summary.csv', index=False)
    atomic_json(evaldir / 'protocol.json', dict(jobs=len(jobs), variants=list(VARIANTS),
                holdouts=manifest['heldout_trajectories'], matched_frozen_force_verified=True,
                primary='Per-joint degree MAE and per-body-channel MAE on strictly unseen trajectories.'))
