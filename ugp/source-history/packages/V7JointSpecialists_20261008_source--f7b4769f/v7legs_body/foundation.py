"""Reuse notebook definitions without running installation/training/V8 cells."""
import ast
import contextlib
import copy
import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_runtime(root, notebook, device):
    root = Path(root)
    env = dict(globals())
    env.update(PERSIST_ROOT=root / 'storage', DEVICE=torch.device(device),
               AUX_DEVICE=torch.device(device), display=lambda *a: None)
    env.update(WORK_ROOT=env['PERSIST_ROOT'] / '.work',
               ROOT=env['PERSIST_ROOT'] / 'raw_tartanground',
               CACHE_ROOT=env['PERSIST_ROOT'] / 'cache',
               OUT_ROOT=env['PERSIST_ROOT'] / 'results',
               RECOVERY_ROOT=env['PERSIST_ROOT'] / 'recovery')
    env['V7F_FEATURE_RECOVERY_ROOT'] = env['RECOVERY_ROOT'] / 'v7f_frozen_features'
    env['ANYMAL_D_URDF'] = env['WORK_ROOT'] / 'v7_external/anymal_d_simple_description/urdf/anymal.urdf'
    cells = json.loads(Path(notebook).read_text())['cells']
    # Literal configuration only: no installs, downloads, launches, deletes, or probes.
    for i in (4, 15):
        for n in ast.parse(''.join(cells[i]['source'])).body:
            if isinstance(n, ast.Assign):
                try:
                    value = ast.literal_eval(n.value)
                except (ValueError, TypeError):
                    if any(isinstance(t, ast.Name) and t.id == 'EXPERIMENTS' for t in n.targets):
                        env['EXPERIMENTS'] = eval(compile(ast.Expression(n.value), 'diagnostic_config', 'eval'),
                                                  {'__builtins__': {}, 'dict': dict})
                    continue
                for t in n.targets:
                    if isinstance(t, ast.Name) and t.id.isupper() and value is not None:
                        if t.id not in env:
                            env[t.id] = value
    env.update(LONG_HORIZONS=[round(i / 10, 1) for i in range(2, 11)],
               ALL_HORIZONS=[round(i / 10, 1) for i in range(1, 11)],
               RAW_TOKENS=64, GEOM_INPUT_POINTS=4096,
               LEG_NAMES=['FL', 'FR', 'RL', 'RR'],
               V7F_LOCAL_HOP_SHORT=0.3, V7F_LOCAL_HOP_LONG=1.0,
               V7F_GRAD_CLIP=1.0, V7F_LOCAL_VAL_FRAC=0.2)
    for i in (4, 5, 6, 7, 8, 9, 10, 11, 12, 15):
        tree = ast.parse(''.join(cells[i]['source']))
        tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        exec(compile(tree, str(notebook) + f':cell{i}', 'exec'), env)
    env['ANYMAL_D_FK'] = env['StrictURDFFK'](env['ANYMAL_D_URDF'])
    return env


def load_record(rt, terrain, environment, run, holdout=False):
    prefix = rt['CACHE_ROOT'] / f'{environment}_{run}'
    with np.load(str(prefix) + '_physics.npz', allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    with np.load(str(prefix) + '_frame_manifest.npz', allow_pickle=False) as z:
        manifest = {k: z[k] for k in z.files}
    vis = rt['load_visual_bundle'](environment, run, manifest)
    diag = next((e for e in rt['EXPERIMENTS'] if e['environment'] == environment and e['run'] == run), None)
    cutoff = diag['ttests'][0] if diag else float(data['time'][-1] - 1.05)
    return dict(terrain_type=terrain, environment=environment, run=run,
                data=data, visual=vis, manifest=manifest,
                ttests=list(diag['ttests']) if diag else [cutoff], holdout=holdout)


def load_models(rt, rec, branch):
    short = branch == 'short'
    hs = rt['SHORT_HORIZONS'] if short else rt['LONG_HORIZONS']
    context = 0.5 if short else 4.0
    steps = int(round(context / np.median(np.diff(rec['data']['time']))))
    base = rt['make_model_for_record'](rec, steps, hs)
    bp = rt['OUT_ROOT'] / ('V7_PRIOR_SHORT.pt' if short else 'V7_PRIOR_LONG.pt')
    fp = rt['PERSIST_ROOT'] / 'V7F_force_specialist' / ('V7F_FORCE_HEAD_SHORT.pt' if short else 'V7F_FORCE_HEAD_LONG.pt')
    base.load_state_dict(torch.load(bp, map_location=rt['DEVICE'], weights_only=False))
    head = rt['V7ForceSpecialistHead']().to(rt['DEVICE'])
    payload = torch.load(fp, map_location=rt['DEVICE'], weights_only=False)
    head.load_state_dict(payload.get('model', payload))
    base.eval().requires_grad_(False)
    head.eval().requires_grad_(False)
    return base, head, hs


def verify_provenance(rt):
    train = sorted([list(pair) for runs in rt['EXPANDED_TRAIN_RUNS'].values() for pair in runs])
    holdout = sorted([list(pair) for pair in rt['STRICT_HOLDOUTS'].values()])
    assert len(train) == 28 and not {tuple(x) for x in train} & {tuple(x) for x in holdout}
    evidence = {}
    for branch, tag in [('short', '500ms_1h'), ('long', '4000ms_9h')]:
        p = rt['RECOVERY_ROOT'] / f'_RECOVERY_V7_PRIOR_{tag}_v2.pt'
        ck = torch.load(p, map_location='cpu', weights_only=False)
        sig = ck['signature']
        assert sorted([list(r[:2]) for r in sig['runs']]) == train
        assert ck['epoch'] == sig['epochs'], f'Incomplete foundation {branch}'
        evidence[branch] = dict(signature=sig, recovery_sha256=sha256(p))
        del ck
    for branch, p in [('short', '_RECOVERY_V7F_SHORT__0_1s_v2.pt'),
                      ('long', '_RECOVERY_V7F_LONG__0_2___1_0s_v2.pt')]:
        ck = torch.load(rt['RECOVERY_ROOT'] / p, map_location='cpu', weights_only=False)
        assert ck['epoch'] == ck['signature']['epochs']
        evidence[branch]['force_signature'] = ck['signature']
    return dict(training_trajectories=train, heldout_trajectories=holdout, completed_foundation=evidence)


def verify_frames(rt):
    from rosbags.highlevel import AnyReader
    frames = []
    for p in sorted(rt['ROOT'].glob('*/Data_anymal/*/rosbags/*.bag')):
        with AnyReader([p]) as reader:
            conns = [c for c in reader.connections if c.topic == '/state_estimator/twist']
            if not conns:
                continue
            c, t, raw = next(reader.messages(connections=conns))
            frame = reader.deserialize(raw, c.msgtype).header.frame_id
            frames.append(dict(bag=str(p), twist_frame=frame))
    base = rt['ANYMAL_D_FK'].root_link
    verified = bool(frames) and base == 'base' and all(x['twist_frame'] == base for x in frames)
    return dict(verified=verified, urdf_root=base, sampled_raw_recordings=frames,
                inference='Twist expressed at URDF base origin per matching base frame identifier.' if verified else None)
