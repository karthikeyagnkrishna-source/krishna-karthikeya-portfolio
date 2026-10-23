"""One focused check of geometry, real-data reconstruction, and gradient flow."""
import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from data import reconstruct
from foundation import load_models, load_record, load_runtime, verify_frames
from models import Hybrid, TorchFK, objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    rt = load_runtime(args.root, Path(__file__).with_name('foundation.ipynb'), 'cuda:0')
    fk = TorchFK(rt['ANYMAL_D_FK'], rt['_rpy_R']).double()
    rec = load_record(rt, 'forest', 'ForestEnv', 'P2000')
    q = torch.tensor(rec['data']['P'][100:104, :, :3], dtype=torch.float64, requires_grad=True)
    r, J = fk(q)
    reference = np.stack([rt['ANYMAL_D_FK'].foot_positions(v.detach().numpy()) for v in q])
    assert np.allclose(r.detach().numpy(), reference, atol=1e-6)
    for joint in range(3):
        plus, minus = q.detach().clone(), q.detach().clone()
        plus[..., joint] += 1e-5
        minus[..., joint] -= 1e-5
        numerical = (fk(plus)[0] - fk(minus)[0]) / 2e-5
        torch.testing.assert_close(J[..., joint], numerical, atol=1e-6, rtol=1e-5)
    r.square().sum().backward()
    assert q.grad is not None and q.grad.abs().sum() > 0
    frames = verify_frames(rt)
    print('GEOMETRY_AND_FRAME_CHECK', frames, flush=True)
    fk = fk.float().cuda()
    base, force, horizons = load_models(rt, rec, 'long')
    source_path = next((rt['V7F_FEATURE_RECOVERY_ROOT'] / 'LONG__0_2___1_0s').glob('*_ForestEnv_P2000.pt'))
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    stats = rt['make_stats_between'](rec['data'], 0, source['signature']['cutoff'])
    batch = reconstruct(base, force, source['records'][:2], rec['data'], stats, horizons, rt['DEVICE'])
    batch = {k: v.cuda().clone() for k, v in batch.items()}
    batch['horizon'] = torch.tensor(horizons, device='cuda')
    scales = {}
    for name, width in [('leg', 9), ('force', 3), ('body', 12), ('r', 3), ('fv', 3), ('stance', 3)]:
        scales[name + '_mean'] = torch.zeros(width, device='cuda')
        scales[name + '_std'] = torch.ones(width, device='cuda')
    model = Hybrid(fk).cuda()
    model.eval()
    out = model(batch, scales)
    torch.testing.assert_close(out['leg'], batch['base_leg'], rtol=0, atol=0)
    torch.testing.assert_close(out['body'], batch['base_body'], rtol=0, atol=0)
    # A real body update makes the body path sensitive to the legs. No fabricated targets.
    opt = torch.optim.AdamW(model.body.parameters(), lr=3e-4)
    loss, _, _ = objective(out, batch, scales, fk, 'body', frames['verified'])
    loss.backward()
    opt.step()
    model.zero_grad(set_to_none=True)
    out = model(batch, scales)
    _, _, parts = objective(out, batch, scales, fk, 'joint', frames['verified'])
    parts['body'].backward()
    grad = sum(float(p.grad.abs().sum()) for p in model.legs.parameters() if p.grad is not None)
    assert grad > 0, 'Body supervision cannot reach leg corrections'
    assert all(p.grad is None for p in base.parameters())
    assert all(p.grad is None for p in force.parameters())
    assert torch.isfinite(out['body']).all() and torch.isfinite(out['leg']).all()
    print('REAL_DATA_ZERO_CORRECTION_AND_BODY_TO_LEG_GRADIENT_CHECK_PASSED', grad, flush=True)


if __name__ == '__main__':
    main()
