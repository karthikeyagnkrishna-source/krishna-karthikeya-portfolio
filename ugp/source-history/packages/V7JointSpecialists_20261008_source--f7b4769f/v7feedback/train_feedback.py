"""Resume one-pass force training, then three-round feedback, then evaluation."""
import argparse
import copy
import fcntl
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'v7legs_body'))
import numpy as np
import torch

from data import atomic_json, atomic_save, batch_at
from foundation import load_runtime, sha256, verify_provenance
from models import Hybrid, TorchFK
from train import Reporter, balanced_order, capture_rng, restore_rng
from feedback_data import checked_checkpoint, fit_force_scales, load_training_cache
from feedback_models import FeedbackModel, feedback_objective
from provenance import canonical_record, implementation_hashes


def make_model(rt, parent, branch, fk, device):
    path = Path(parent) / 'checkpoints' / branch / 'joint_best.pt'
    seed = Hybrid(copy.deepcopy(fk))
    seed.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model'])
    return FeedbackModel(seed).to(device)


@torch.no_grad()
def validate(model, tensors, scales, stage, batch_size, stance, threshold, device):
    model.eval()
    total, count, pieces = 0., 0, {}
    for ids in torch.nonzero(tensors['val']).flatten().split(batch_size):
        batch = batch_at(tensors, ids, device)
        out = model(batch, scales, stage)
        _, score, p = feedback_objective(out, batch, scales, model.fk, stage, stance, threshold)
        total += float(score) * len(ids)
        count += len(ids)
        for k, v in p.items():
            pieces[k] = pieces.get(k, 0.) + float(v) * len(ids)
    return total / count, {k: v / count for k, v in pieces.items()}


def train_stage(model, tensors, scales, stage, output, branch, signature, stance, threshold, report, device):
    directory = Path(output) / 'checkpoints' / branch
    latest, best_path = directory / f'{stage}_latest.pt', directory / f'{stage}_best.pt'
    model.configure(stage)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                           lr=3e-4 if stage == 'one_pass' else 5e-5, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(1781 + (stage == 'feedback'))
    max_epochs, batch_size = (30 if stage == 'one_pass' else 15), (16 if branch == 'short' else 8)
    epoch = cursor = steps = patience = 0
    best, order, running, n = float('inf'), None, 0., 0
    if latest.exists():
        ck = checked_checkpoint(latest, signature, device)
        model.load_state_dict(ck['model'])
        if ck['complete']:
            model.load_state_dict(checked_checkpoint(best_path, signature, device)['model'])
            report('stage_reused', branch=branch, stage=stage, best_validation=ck['best'])
            return
        opt.load_state_dict(ck['optimizer'])
        epoch, cursor, steps, patience = (ck[k] for k in ('epoch', 'cursor', 'steps', 'patience'))
        best, order, running, n = ck['best'], ck['order'].cpu() if ck['order'] is not None else None, ck['running'], ck['n']
        generator.set_state(ck['generator'].cpu())
        restore_rng({**ck['rng'], 'cpu': ck['rng']['cpu'].cpu(), 'cuda': [s.cpu() for s in ck['rng']['cuda']]})

    def save(complete=False):
        atomic_save(latest, dict(signature=signature, model=model.state_dict(), optimizer=opt.state_dict(),
                    epoch=epoch, cursor=cursor, steps=steps, patience=patience, best=best, order=order,
                    generator=generator.get_state(), rng=capture_rng(), running=running, n=n, complete=complete))

    for ep in range(epoch, max_epochs):
        epoch = ep
        if order is None:
            order, cursor, running, n = balanced_order(tensors, generator), 0, 0., 0
        model.train()
        for offset in range(cursor, len(order), batch_size):
            ids = order[offset:offset + batch_size]
            batch = batch_at(tensors, ids, device)
            opt.zero_grad(set_to_none=True)
            out = model(batch, scales, stage)
            loss, _, _ = feedback_objective(out, batch, scales, model.fk, stage, stance, threshold)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite feedback loss')
            loss.backward()
            if any(p.grad is not None for p in model.seed.parameters()):
                raise RuntimeError('Frozen seed received gradients')
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
            opt.step()
            cursor, steps = offset + len(ids), steps + 1
            running, n = running + float(loss.detach()) * len(ids), n + len(ids)
            if steps == 1 or steps % 50 == 0:
                report('training', branch=branch, stage=stage, epoch=ep + 1, max_epochs=max_epochs,
                       steps=steps, loss=float(loss.detach()), grad_norm=float(norm), batch=offset // batch_size + 1,
                       batches=(len(order) + batch_size - 1) // batch_size,
                       training_trajectories=28, rounds=1 if stage == 'one_pass' else 3,
                       gpu_memory_mb=torch.cuda.memory_allocated() / 1024**2)
            if steps == 1 or steps % 250 == 0:
                save()
        score, pieces = validate(model, tensors, scales, stage, batch_size, stance, threshold, device)
        if score < best * .999:
            best, patience = score, 0
            atomic_save(best_path, dict(signature=signature, model=model.state_dict(), validation=score,
                                       epoch=ep + 1, steps=steps, components=pieces))
        else:
            patience += 1
        record = dict(branch=branch, stage=stage, epoch=ep + 1, supervised_validation=score, best_validation=best,
                      patience=patience, train_loss=running / n, **pieces)
        with (Path(output) / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        report('epoch_complete', **record)
        epoch, cursor, order = ep + 1, 0, None
        save(complete=patience >= 5 or epoch >= max_epochs)
        if patience >= 5:
            break
    model.load_state_dict(checked_checkpoint(best_path, signature, device)['model'])
    report('stage_complete', branch=branch, stage=stage, epochs_completed=epoch, best_validation=best)


def verify_real(model, tensors, scales, threshold, stance, device):
    batch = batch_at(tensors, torch.nonzero(tensors['train']).flatten()[:2], device)
    model.eval()
    with torch.no_grad():
        reference = model.seed(batch, scales, 'hybrid')
        initial = model(batch, scales)
        for step in initial['rounds']:
            torch.testing.assert_close(step['leg'], reference['leg'], atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(step['body'], reference['body'], atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(step['force'], batch['force'], atol=0, rtol=0)
    model.configure('one_pass')
    out = model(batch, scales, 'one_pass')
    loss, _, _ = feedback_objective(out, batch, scales, model.fk, 'one_pass', stance, threshold)
    loss.backward()
    if not torch.isfinite(loss) or not any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.force_refiner.parameters()):
        raise RuntimeError('Real-data force gradient check failed')
    if any(p.grad is not None for p in model.seed.parameters()):
        raise RuntimeError('Seed gradient check failed')
    model.zero_grad(set_to_none=True)
    return dict(train_windows=int(tensors['train'].sum()), validation_windows=int(tensors['val'].sum()),
                target_force_shape=list(batch['target_force'].shape), loss=float(loss.detach()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--notebook', type=Path, default=Path(__file__).resolve().parent.parent / 'v7legs_body/foundation.ipynb')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--train-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    report = Reporter(args.output)
    try:
        random.seed(1781)
        np.random.seed(1781)
        torch.manual_seed(1781)
        torch.cuda.manual_seed_all(1781)
        torch.set_num_threads(2)
        device = torch.device('cuda:0')
        report('initializing', gpu=torch.cuda.get_device_name(), physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))
        rt = load_runtime(args.root, args.notebook, device)
        parent_manifest = json.loads((args.parent / 'manifest.json').read_text())
        proven = verify_provenance(rt)
        for k in ('training_trajectories', 'heldout_trajectories'):
            if sorted(proven[k]) != sorted(parent_manifest[k]):
                raise ValueError('Parent/foundation trajectory provenance differs')
        if json.loads(json.dumps(proven['completed_foundation'])) != parent_manifest['completed_foundation']:
            raise ValueError('Completed foundation differs from the parent run')
        if parent_manifest['notebook_sha256'] != sha256(args.notebook):
            raise ValueError('Foundation notebook differs from completed parent')
        manifest = dict(proven, parent_manifest_sha256=sha256(args.parent / 'manifest.json'),
                        parent_checkpoint_sha256={b: sha256(args.parent / 'checkpoints' / b / 'joint_best.pt') for b in ('short', 'long')},
                        implementation=implementation_hashes(Path(__file__).parent, Path(__file__).parent.parent / 'v7legs_body'),
                        urdf_sha256=sha256(rt['ANYMAL_D_URDF']),
                        frame_evidence=parent_manifest['frame_evidence'], seed=1781, rounds=3,
                        variants=['seed', 'one_pass', 'round1', 'round2', 'round3'])
        manifest = canonical_record(manifest)
        mp = args.output / 'manifest.json'
        if mp.exists() and json.loads(mp.read_text()) != manifest:
            raise ValueError('Run inputs changed; preserve existing output and use a new run directory')
        atomic_json(mp, manifest)
        fk = TorchFK(rt['ANYMAL_D_FK'], rt['_rpy_R'])
        for branch in ('short', 'long'):
            tensors = load_training_cache(rt, args.parent, args.output, branch, report)
            sp = args.output / f'{branch}_scales.pt'
            if not sp.exists():
                original = torch.load(args.parent / f'{branch}_scales.pt', map_location='cpu', weights_only=False)
                atomic_save(sp, fit_force_scales(tensors, original))
            scales = {k: v.to(device) for k, v in torch.load(sp, map_location='cpu', weights_only=False).items()}
            model = make_model(rt, args.parent, branch, fk, device)
            stance = manifest['frame_evidence']['verified']
            threshold = float(rt['CONTACT_THRESHOLD'])
            report('real_data_verified', branch=branch, training_trajectories=28,
                   **verify_real(model, tensors, scales, threshold, stance, device))
            signature = dict(manifest_sha256=sha256(mp), branch=branch,
                             cache_manifest_sha256=sha256(args.output / f'{branch}_manifest.json'), scales_sha256=sha256(sp))
            if not args.verify_only:
                for stage in ('one_pass', 'feedback'):
                    train_stage(model, tensors, scales, stage, args.output, branch, signature, stance, threshold, report, device)
                _, pieces = validate(model, tensors, scales, 'feedback', 16 if branch == 'short' else 8, stance, threshold, device)
                selected = min(range(1, 4), key=lambda k: pieces[f'round{k}_supervised'])
                atomic_json(args.output / f'{branch}_round_selection.json', dict(selected_round=selected, validation=pieces,
                            note='Round selected using head validation only; unseen labels are never used for selection.'))
            del model, tensors
            torch.cuda.empty_cache()
        if args.verify_only:
            report('verification_complete')
            return
        report('training_complete')
        if not args.train_only:
            from evaluate_feedback import evaluate_all
            evaluate_all(rt, args.output, args.parent, fk, manifest, report)
            report('complete', result_directory=str(args.output / 'evaluation'))
    except Exception as exc:
        report('failed', error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
