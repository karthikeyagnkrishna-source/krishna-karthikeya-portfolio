"""Resumable staged training, followed by matched evaluation of four variants."""
import argparse
import copy
import fcntl
import json
import os
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from data import atomic_json, atomic_save, batch_at, build_cache, fit_scales
from foundation import load_runtime, sha256, verify_frames, verify_provenance
from models import BodyHead, Hybrid, TorchFK, objective


class Reporter:
    def __init__(self, output):
        self.output = Path(output)
        self.started = time.time()

    def __call__(self, phase, **values):
        record = dict(phase=phase, updated_unix=time.time(), pid=os.getpid(), **values)
        atomic_json(self.output / 'status.json', record)
        print(json.dumps(record, allow_nan=False), flush=True)


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    torch.cuda.set_rng_state_all(state['cuda'])


def configure_stage(model, stage):
    model.requires_grad_(False)
    if stage in ('legs', 'joint'):
        model.legs.requires_grad_(True)
    if stage in ('body', 'body_only', 'joint'):
        model.body.requires_grad_(True)


def balanced_order(tensors, generator):
    groups = [torch.nonzero(tensors['train'] & (tensors['terrain'] == i)).flatten() for i in range(4)]
    size = max(len(g) for g in groups)
    samples = []
    for group in groups:
        perm = group[torch.randperm(len(group), generator=generator)]
        if len(perm) < size:
            perm = torch.cat([perm, group[torch.randint(len(group), (size - len(perm),), generator=generator)]])
        samples.append(perm)
    return torch.stack(samples, 1).flatten()  # terrain mixed even for small batches


@torch.no_grad()
def validate(model, tensors, scales, stage, batch_size, stance, device):
    model.eval()
    variant = 'leg_only' if stage == 'legs' else 'body_only' if stage == 'body_only' else 'hybrid'
    ids = torch.nonzero(tensors['val']).flatten()
    total, count = 0.0, 0
    components = {}
    for chunk in ids.split(batch_size):
        batch = batch_at(tensors, chunk, device)
        out = model(batch, scales, variant)
        _, selection, pieces = objective(out, batch, scales, model.fk, stage, stance)
        total += float(selection) * len(chunk)
        count += len(chunk)
        for key, value in pieces.items():
            components[key] = components.get(key, 0) + float(value) * len(chunk)
    return total / count, {k: v / count for k, v in components.items()}


def train_stage(model, tensors, scales, stage, output, branch, signature, stance, report, device):
    outdir = Path(output) / 'checkpoints' / branch
    latest_path, best_path = outdir / f'{stage}_latest.pt', outdir / f'{stage}_best.pt'
    configure_stage(model, stage)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=5e-5 if stage == 'joint' else 3e-4, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(781 + ['legs', 'body', 'joint', 'body_only'].index(stage))
    epochs = 15 if stage == 'joint' else 30
    batch_size = 16 if branch == 'short' else 8
    epoch, cursor, steps, patience = 0, 0, 0, 0
    order = None
    best = float('inf')
    running_sum, running_n = 0.0, 0
    if latest_path.exists():
        ck = torch.load(latest_path, map_location=device, weights_only=False)
        if ck['signature'] != signature:
            raise ValueError('Resume signature changed; preserve run and use a new output directory')
        model.load_state_dict(ck['model'])
        if ck['complete']:
            model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)['model'])
            report('stage_reused', branch=branch, stage=stage, best_validation=ck['best'])
            return
        opt.load_state_dict(ck['optimizer'])
        epoch, cursor, steps, patience = ck['epoch'], ck['cursor'], ck['steps'], ck['patience']
        best, order = ck['best'], ck['order'].cpu() if ck['order'] is not None else None
        running_sum, running_n = ck['running_sum'], ck['running_n']
        generator.set_state(ck['generator'].cpu())
        restore_rng({**ck['rng'], 'cpu': ck['rng']['cpu'].cpu(), 'cuda': [s.cpu() for s in ck['rng']['cuda']]})

    def save(complete=False):
        atomic_save(latest_path, dict(signature=signature, model=model.state_dict(), optimizer=opt.state_dict(),
                    epoch=epoch, cursor=cursor, steps=steps, patience=patience, best=best,
                    order=order, generator=generator.get_state(), rng=capture_rng(), complete=complete,
                    running_sum=running_sum, running_n=running_n))

    variant = 'leg_only' if stage == 'legs' else 'body_only' if stage == 'body_only' else 'hybrid'
    for ep in range(epoch, epochs):
        epoch = ep
        if order is None:
            order = balanced_order(tensors, generator)
            cursor, running_sum, running_n = 0, 0.0, 0
        model.train()
        # Frozen leg head remains deterministic while the body learns from its predictions.
        if stage in ('body', 'body_only'):
            model.legs.eval()
        for offset in range(cursor, len(order), batch_size):
            indices = order[offset:offset + batch_size]
            batch = batch_at(tensors, indices, device)
            opt.zero_grad(set_to_none=True)
            out = model(batch, scales, variant)
            loss, _, _ = objective(out, batch, scales, model.fk, stage, stance)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
            opt.step()
            cursor = offset + len(indices)
            steps += 1
            running_sum += float(loss.detach()) * len(indices)
            running_n += len(indices)
            if steps == 1 or steps % 50 == 0:
                report('training', branch=branch, stage=stage, epoch=ep + 1, max_epochs=epochs,
                       batch=(offset // batch_size) + 1, batches=(len(order) + batch_size - 1) // batch_size,
                       steps=steps, loss=float(loss.detach()), grad_norm=float(norm),
                       gpu_memory_mb=torch.cuda.memory_allocated() / 1024**2)
            if steps % 250 == 0:
                save()
        score, pieces = validate(model, tensors, scales, stage, batch_size, stance, device)
        improved = score < best * 0.999
        if improved:
            best, patience = score, 0
            atomic_save(best_path, dict(signature=signature, model=model.state_dict(), validation=score,
                                       epoch=ep + 1, steps=steps))
        else:
            patience += 1
        record = dict(branch=branch, stage=stage, epoch=ep + 1, train_loss=running_sum / running_n,
                      supervised_validation=score, best_validation=best, patience=patience, **pieces)
        with (Path(output) / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        report('epoch_complete', **record)
        epoch, cursor, order = ep + 1, 0, None
        save(complete=patience >= 5 or epoch >= epochs)
        if patience >= 5:
            break
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)['model'])
    report('stage_complete', branch=branch, stage=stage, best_validation=best, epochs_completed=epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--notebook', type=Path, default=Path(__file__).with_name('foundation.ipynb'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    report = Reporter(args.output)
    try:
        random.seed(781)
        np.random.seed(781)
        torch.manual_seed(781)
        torch.cuda.manual_seed_all(781)
        torch.set_num_threads(2)
        device = torch.device('cuda:0')
        report('initializing', physical_cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
               gpu=torch.cuda.get_device_name(), total_gpu_memory_mb=torch.cuda.get_device_properties(0).total_memory / 1024**2)
        rt = load_runtime(args.root, args.notebook, device)
        manifest = verify_provenance(rt)
        frame = verify_frames(rt)
        manifest.update(frame_evidence=frame, notebook_sha256=sha256(args.notebook),
                        implementation={p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')},
                        variants=['base', 'leg_only', 'body_only', 'hybrid'], seed=781)
        manifest_path = args.output / 'manifest.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError('Run provenance changed; choose a new output directory')
        atomic_json(manifest_path, manifest)
        fk = TorchFK(rt['ANYMAL_D_FK'], rt['_rpy_R'])
        for branch in ('short', 'long'):
            tensors = build_cache(rt, args.output, branch, manifest, report)
            scale_path = args.output / f'{branch}_scales.pt'
            if scale_path.exists():
                scales = torch.load(scale_path, map_location='cpu', weights_only=False)
            else:
                scales = {k: v.clone() for k, v in fit_scales(tensors, fk).items()}
                atomic_save(scale_path, scales)
            scales = {k: v.to(device) for k, v in scales.items()}
            model = Hybrid(copy.deepcopy(fk)).to(device)
            signature = dict(manifest_sha256=sha256(manifest_path), branch=branch,
                             cache_manifest_sha256=sha256(args.output / f'{branch}_manifest.json'),
                             scales_sha256=sha256(scale_path), stance_enabled=frame['verified'])
            for stage in ('legs', 'body', 'joint'):
                train_stage(model, tensors, scales, stage, args.output, branch, signature, frame['verified'], report, device)
            model.body = BodyHead().to(device)
            train_stage(model, tensors, scales, 'body_only', args.output, branch, signature, frame['verified'], report, device)
            del tensors, model
            torch.cuda.empty_cache()
        report('training_complete')
        if not args.train_only:
            from evaluate import evaluate_all
            evaluate_all(rt, args.output, fk, manifest, report)
            report('complete', result_directory=str(args.output / 'evaluation'))
    except Exception as exc:
        report('failed', error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
