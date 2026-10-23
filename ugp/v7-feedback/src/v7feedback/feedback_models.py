"""Pretrained leg/body updates and shared, bounded force-feedback rounds."""
import copy

import torch
from torch import nn
from torch.nn import functional as F

from models import objective, transformer, zero_last


def leg_prediction(head, batch, scales, force, previous=None, adapter=None):
    x = batch['x'].float()
    B, H, L, _ = x.shape
    identity = head.identity(torch.arange(L, device=x.device)).view(1, 1, L, 16).expand(B, H, -1, -1)
    base = (batch['base_leg'] - scales['leg_mean']) / scales['leg_std']
    fn = (force - scales['force_mean']) / scales['force_std']
    horizon = batch['horizon'].view(1, H, 1, 1).expand(B, -1, L, -1)
    y = head.projection(torch.cat([x, identity, base, fn, horizon], -1))
    if adapter is not None:
        old_leg = (previous['leg'] - scales['leg_mean']) / scales['leg_std']
        old_body = ((previous['body'] - scales['body_mean']) / scales['body_std'])[:, :, None].expand(-1, -1, L, -1)
        y = y + adapter(torch.cat([old_leg, old_body], -1))
    y = head.temporal(y.permute(0, 2, 1, 3).reshape(B * L, H, 256))
    hidden = y.reshape(B, L, H, 256).permute(0, 2, 1, 3)
    delta = torch.cat([part(hidden) for part in head.outputs], -1)
    return batch['base_leg'] + delta * scales['leg_std'], hidden


def body_prediction(head, batch, scales, leg, force, r, fv, previous=None, adapter=None):
    B, H, L = leg.shape[:3]
    z = batch['x'][:, :, 0, :768].float()
    identity = head.identity(torch.arange(L, device=leg.device)).view(1, 1, L, 16).expand(B, H, -1, -1)
    token = torch.cat([(leg - scales['leg_mean']) / scales['leg_std'],
                       (force - scales['force_mean']) / scales['force_std'], batch['contact'][..., None],
                       (r - scales['r_mean']) / scales['r_std'], (fv - scales['fv_mean']) / scales['fv_std'], identity], -1)
    token = head.leg_projection(token)
    query = head.query(z)
    if adapter is not None:
        query = query + adapter((previous['body'] - scales['body_mean']) / scales['body_std'])
    mixed = torch.cat([query[:, :, None], token], 2).reshape(B * H, L + 1, 256)
    hidden = head.leg_mixer(mixed)[:, 0].reshape(B, H, 256)
    hidden = head.temporal(hidden)
    base = (batch['base_body'] - scales['body_mean']) / scales['body_std']
    delta = head.output(torch.cat([hidden, z, base], -1))
    return batch['base_body'] + delta * scales['body_std'], hidden


class ForceRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.identity = nn.Embedding(4, 16)
        self.projection = nn.Sequential(nn.LayerNorm(1845), nn.Linear(1845, 256), nn.GELU())
        self.body_query = nn.LayerNorm(256)
        self.leg_mixer = transformer()
        self.temporal = transformer()
        self.output = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 3))
        zero_last(self.output)

    def forward(self, batch, scales, state, leg_hidden, body_hidden, previous_force):
        B, H, L = state['leg'].shape[:3]
        identity = self.identity(torch.arange(L, device=previous_force.device)).view(1, 1, L, 16).expand(B, H, -1, -1)
        body = ((state['body'] - scales['body_mean']) / scales['body_std'])[:, :, None].expand(-1, -1, L, -1)
        bh = body_hidden[:, :, None].expand(-1, -1, L, -1)
        horizon = batch['horizon'].view(1, H, 1, 1).expand(B, -1, L, -1)
        x = torch.cat([batch['x'].float(), leg_hidden, bh,
                       (state['leg'] - scales['leg_mean']) / scales['leg_std'], body,
                       (state['r'] - scales['r_mean']) / scales['r_std'],
                       (state['fv'] - scales['fv_mean']) / scales['fv_std'],
                       (previous_force - scales['target_force_mean']) / scales['target_force_std'],
                       batch['contact'][..., None], identity, horizon], -1)
        token = self.projection(x)
        token = torch.cat([self.body_query(body_hidden)[:, :, None], token], 2).reshape(B * H, L + 1, 256)
        token = self.leg_mixer(token)[:, 1:].reshape(B, H, L, 256)
        token = self.temporal(token.permute(0, 2, 1, 3).reshape(B * L, H, 256))
        token = token.reshape(B, L, H, 256).permute(0, 2, 1, 3)
        return 2 * torch.tanh(self.output(token) / 2)  # normalized, bounded seed-relative correction


class FeedbackModel(nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.seed = copy.deepcopy(seed).eval().requires_grad_(False)
        self.core = copy.deepcopy(seed)
        self.leg_adapter = nn.Sequential(nn.LayerNorm(21), nn.Linear(21, 64), nn.GELU(), nn.Linear(64, 256))
        self.body_adapter = nn.Sequential(nn.LayerNorm(12), nn.Linear(12, 64), nn.GELU(), nn.Linear(64, 256))
        zero_last(self.leg_adapter)
        zero_last(self.body_adapter)
        self.force_refiner = ForceRefiner()
        self.gates = nn.Parameter(torch.zeros(3))  # .5 * sigmoid(0) = .25
        self._stage = 'feedback'

    @property
    def fk(self):
        return self.core.fk

    def train(self, mode=True):
        super().train(mode)
        self.seed.eval()
        if self._stage == 'one_pass':
            self.core.eval()
        return self

    def configure(self, stage):
        if stage not in ('one_pass', 'feedback'):
            raise ValueError(stage)
        self._stage = stage
        self.requires_grad_(False)
        self.force_refiner.requires_grad_(True)
        self.gates.requires_grad_(True)
        if stage == 'feedback':
            self.core.legs.requires_grad_(True)
            self.core.body.requires_grad_(True)
            self.leg_adapter.requires_grad_(True)
            self.body_adapter.requires_grad_(True)
        self.seed.eval()

    def seed_state(self, batch, scales):
        with torch.no_grad():
            leg, lh = leg_prediction(self.seed.legs, batch, scales, batch['force'])
            r, J = self.seed.fk(leg[..., :3])
            fv = (J @ leg[..., 3:6, None]).squeeze(-1)
            body, bh = body_prediction(self.seed.body, batch, scales, leg, batch['force'], r, fv)
        return dict(leg=leg, body=body, force=batch['force'], r=r, fv=fv), lh, bh

    def forward(self, batch, scales, mode='feedback', rounds=3):
        if mode not in ('seed', 'one_pass', 'feedback') or not 1 <= rounds <= 3:
            raise ValueError('Use seed/one_pass/feedback with one to three rounds')
        seed, lh, bh = self.seed_state(batch, scales)
        previous = seed
        outputs = []
        alpha = .5 * self.gates.sigmoid()
        for _ in range(1 if mode != 'feedback' else rounds):
            if mode in ('seed', 'one_pass'):
                state = dict(seed)
            else:
                candidate, lh = leg_prediction(self.core.legs, batch, scales, previous['force'], previous, self.leg_adapter)
                leg = previous['leg'] + alpha[0] * (candidate - previous['leg'])
                r, J = self.fk(leg[..., :3])
                fv = (J @ leg[..., 3:6, None]).squeeze(-1)
                candidate_body, bh = body_prediction(self.core.body, batch, scales, leg, previous['force'], r, fv,
                                                     previous, self.body_adapter)
                body = previous['body'] + alpha[1] * (candidate_body - previous['body'])
                state = dict(leg=leg, body=body, r=r, fv=fv)
            if mode == 'seed':
                force = seed['force']
            else:
                delta = self.force_refiner(batch, scales, state, lh, bh, previous['force'])
                candidate_force = seed['force'] + delta * scales['target_force_std']
                force = previous['force'] + alpha[2] * (candidate_force - previous['force'])
            state.update(force=force,
                         leg_delta=(state['leg'] - seed['leg']) / scales['leg_std'],
                         body_delta=(state['body'] - seed['body']) / scales['body_std'],
                         force_delta=(force - seed['force']) / scales['target_force_std'])
            outputs.append(state)
            previous = state
        return dict(outputs[-1], rounds=outputs)


def force_objective(pred, batch, scales, threshold):
    gt = batch['target_force']
    std = scales['target_force_std']
    vec = F.smooth_l1_loss((pred - gt) / std, torch.zeros_like(pred), beta=.35)
    pm, gm = pred.norm(dim=-1), gt.norm(dim=-1)
    mag = F.smooth_l1_loss(torch.log1p(pm), torch.log1p(gm), beta=.20)
    cosine = ((pred / pm.clamp_min(1.)[..., None]) * (gt / gm.clamp_min(1.)[..., None])).sum(-1)
    mask = (gm >= max(float(threshold), 1.)).to(pred.dtype)
    direction = ((1 - cosine.clamp(-1, 1)) * mask).sum() / mask.sum().clamp_min(1)
    curve = pred.sum() * 0
    if pred.shape[1] > 1:
        curve = F.smooth_l1_loss((pred[:, 1:] - pred[:, :-1]) / std,
                                 (gt[:, 1:] - gt[:, :-1]) / std, beta=.35)
    return vec + .25 * mag + .10 * direction + .10 * curve, dict(force_vector=vec, force_magnitude=mag,
                                                               force_direction=direction, force_curve=curve)


def feedback_objective(out, batch, scales, fk, stage, stance, threshold):
    steps = out['rounds']
    weights = [.2, .3, .5] if len(steps) == 3 else [1 / len(steps)] * len(steps)
    total, selection = batch['force'].sum() * 0, batch['force'].sum() * 0
    pieces = {}
    for index, (weight, step) in enumerate(zip(weights, steps), 1):
        fl, fp = force_objective(step['force'], batch, scales, threshold)
        if stage == 'one_pass':
            loss, score = fl + .01 * step['force_delta'].square().mean(), fl
            sp = {}
        else:
            ll, ls, sp = objective(step, batch, scales, fk, 'joint', stance)
            loss, score = ll + fl + .01 * step['force_delta'].square().mean(), ls + fl
        total, selection = total + weight * loss, selection + weight * score
        pieces.update({f'round{index}_{k}': v for k, v in {**sp, **fp}.items()})
        pieces[f'round{index}_supervised'] = score
    return total, selection, pieces
