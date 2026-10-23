"""CPU verification by default; explicit --train reuses the original V7 trainer."""
import argparse
import json
import time
from pathlib import Path
from .runtime import (check_output_directory, install_visual_history, load_foundation_runtime,
                      record_manifest, sha256)


def atomic_json(path, value):
    temp = Path(path).with_suffix('.json.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--foundation-root', type=Path, required=True)
    parser.add_argument('--notebook', type=Path,
                        default=Path(__file__).resolve().parents[1]/'v7legs_body/foundation.ipynb')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--branch', choices=['short', 'long', 'both'], default='both')
    parser.add_argument('--frames', type=int, default=6)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', type=int)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--train', action='store_true')
    modes.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.frames < 2 or (args.epochs is not None and args.epochs < 1):
        parser.error('frames must be >=2 and epochs must be positive')
    check_output_directory(args.foundation_root, args.output)
    import numpy as np
    import torch
    torch.set_num_threads(2)
    rt = load_foundation_runtime(args.foundation_root, args.notebook, args.device)
    from foundation import load_record, verify_provenance
    evidence = verify_provenance(rt)
    branches = ['short', 'long'] if args.branch == 'both' else [args.branch]
    checkpoints = {b: sha256(args.foundation_root/'storage/results'/f'V7_PRIOR_{b.upper()}.pt') for b in ('short', 'long')}
    manifest = dict(schema='v7_visual_history_v1', frames=args.frames, selection='current_plus_previous_consecutive',
        alignment='timestamped_full_robot_memory_learned_attention', foundation_frozen=True,
        training_trajectories=evidence['training_trajectories'], heldout_trajectories=evidence['heldout_trajectories'],
        foundation_weights=checkpoints, notebook_sha256=sha256(args.notebook),
        implementation={p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        loss='original_v7_loss', controls='original_known_future_controller_signals',
        history_samples={'short': 50, 'long': 400},
        training_protocol='Original V7 eligible ranges and trainer; no new validation-based checkpoint selection')
    record_manifest(args.output, manifest)
    original = install_visual_history(rt, args.foundation_root, args.frames)
    # New checkpoint/context namespace; original cached raw image features are read-only.
    rt.update(OUT_ROOT=args.output, RECOVERY_ROOT=args.output/'recovery',
              V7F_FEATURE_RECOVERY_ROOT=args.output/'v7f_frozen_features')
    for name in ('OUT_ROOT', 'RECOVERY_ROOT'):
        rt[name].mkdir(parents=True, exist_ok=True)
    records = []
    pairs = [(terrain, env, run) for terrain, runs in rt['EXPANDED_TRAIN_RUNS'].items() for env, run in runs]
    selected = pairs if args.train else pairs[:1]
    for terrain, env, run in selected:
        rec = load_record(rt, terrain, env, run)
        dt = float(np.median(np.diff(rec['data']['time'])))
        if not np.isclose(dt, .01, atol=1e-7, rtol=0):
            raise ValueError('Robot timestamp interval changed from the verified 100 Hz contract')
        records.append(rec)
    rt['initialize_regimes_and_physics'](records)
    if args.train:
        # These calls already implement loss, AdamW, clipping, per-window resume and RNG restore.
        atomic_json(args.output/'status.json', dict(phase='training', updated_unix=time.time(),
                    training_trajectories=len(records), branches=branches))
        for branch in branches:
            short = branch == 'short'; context = .5 if short else 4.
            horizons = rt['SHORT_HORIZONS'] if short else rt['LONG_HORIZONS']
            epochs = args.epochs or rt['PRIOR_EPOCHS_SHORT' if short else 'PRIOR_EPOCHS_LONG']
            model = rt['train_v7_prior'](records, context, .3 if short else 1., horizons, epochs,
                                        rt['SEED']+(10 if short else 20), 'VISUAL HISTORY '+branch.upper())
            rt['atomic_torch_save'](model.state_dict(), args.output/f'VISUAL_HISTORY_{branch.upper()}.pt')
            del model
        atomic_json(args.output/'status.json', dict(phase='training_complete', updated_unix=time.time(),
                    training_trajectories=len(records), branches=branches, evaluated=False))
        return
    reports = []
    rec = records[0]; data = rec['data']
    stats = rt['make_stats_between'](data, float(data['time'][0]), float(min(rec['ttests'])))
    norm = rt['normalize_arrays'](data, stats)
    idx = int(np.searchsorted(data['time'], data['time'][0]+10))
    for branch in branches:
        short = branch == 'short'; steps = 50 if short else 400
        horizons = rt['SHORT_HORIZONS'] if short else rt['LONG_HORIZONS']
        sample = rt['make_v7_sample'](data, rec['visual'], stats, rec['physics_prior_v7'], idx, steps, horizons, norm)
        model = rt['make_model_for_record'](rec, steps, horizons).eval()
        with torch.no_grad():
            old = original['original_forward'](model.base, sample)
            new = rt['model_forward_v7'](model, sample)
            for key in old:
                torch.testing.assert_close(old[key], new[key], atol=0, rtol=0)
            features, diagnostic = model.features(sample, return_attention=True)
            force_context = rt['v7f_extract_base_context'](model, sample, stats)
            torch.testing.assert_close(features, force_context['x'], atol=0, rtol=0)
        model.train()
        loss, _ = rt['v7_loss'](model.forward_sample(sample), sample, stats, rec['physics_prior_v7'])
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite real-data loss')
        loss.backward()  # Gradient check only: there is no optimizer or parameter update.
        grad = float(model.adapter.output.weight.grad.abs().sum())
        if grad <= 0 or any(p.grad is not None for p in model.base.parameters()):
            raise RuntimeError('History gradient contract failed')
        history = sample['visual_history']
        reports.append(dict(branch=branch, robot_samples=steps, robot_span_s=float(-history.robot_times[0, 0]),
            image_frames=int(history.valid.sum()), image_span_s=float(history.visual_times[0, -1]-history.visual_times[0, 0]),
            memory_tokens=diagnostic['memory_tokens'], feature_shape=list(features.shape), warm_start_exact=True,
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            real_data_loss=float(loss.detach()), adapter_gradient_sum=grad, optimizer_steps=0))
        del model
    result = dict(phase='cpu_verified' if args.device == 'cpu' else 'verified', training_started=False,
                  trajectory=f"{rec['environment']}/{rec['run']}", branches=reports)
    atomic_json(args.output/'verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
