"""Residual leg/body specialists and URDF kinematics; foundation remains frozen."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def transformer(layers=2):
    layer = nn.TransformerEncoderLayer(256, 8, 1024, 0.1, 'gelu', batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


def zero_last(module):
    last = [m for m in module.modules() if isinstance(m, nn.Linear)][-1]
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)


class TorchFK(nn.Module):
    def __init__(self, reference, rpy_rotation):
        super().__init__()
        self.chains = []
        for leg, chain in enumerate(reference.chains):
            rows = []
            expected = [f'{reference.leg_prefixes[leg]}_{j}' for j in ('HAA', 'HFE', 'KFE')]
            for index, joint in enumerate(chain):
                tag = f'l{leg}j{index}'
                self.register_buffer(tag + 'R', torch.tensor(rpy_rotation(joint['rpy']), dtype=torch.float32))
                self.register_buffer(tag + 'p', torch.tensor(joint['xyz'], dtype=torch.float32))
                axis = np.asarray(joint['axis'], float)
                axis /= max(np.linalg.norm(axis), 1e-12)
                self.register_buffer(tag + 'a', torch.tensor(axis, dtype=torch.float32))
                moving = joint['type'] != 'fixed'
                if moving and joint['name'] not in expected:
                    raise ValueError(f'Unmapped moving joint {joint["name"]}')
                if moving and joint['type'] not in ('revolute', 'continuous'):
                    raise ValueError('Unsupported variable joint type')
                rows.append((tag, expected.index(joint['name']) if moving else None))
            self.chains.append(rows)

    def forward(self, q):
        positions, jacobians = [], []
        eye = torch.eye(3, device=q.device, dtype=q.dtype)
        for leg, rows in enumerate(self.chains):
            shape = q.shape[:-2]
            R = eye.expand(*shape, 3, 3)
            p = q.new_zeros(*shape, 3)
            axes, origins = {}, {}
            for tag, jid in rows:
                p = p + (R @ getattr(self, tag + 'p').to(q.dtype).unsqueeze(-1)).squeeze(-1)
                R = R @ getattr(self, tag + 'R').to(q.dtype)
                if jid is not None:
                    a = getattr(self, tag + 'a').to(q.dtype)
                    axes[jid] = (R @ a.unsqueeze(-1)).squeeze(-1)
                    origins[jid] = p
                    angle = q[..., leg, jid]
                    x, y, z = a.unbind()
                    zero = x * 0
                    K = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero]).reshape(3, 3)
                    c, s = angle.cos()[..., None, None], angle.sin()[..., None, None]
                    Ra = c * eye + (1 - c) * torch.outer(a, a) + s * K
                    R = R @ Ra
            positions.append(p)
            jacobians.append(torch.stack([torch.cross(axes[j], p - origins[j], dim=-1) for j in range(3)], -1))
        return torch.stack(positions, -2), torch.stack(jacobians, -3)


class LegHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.identity = nn.Embedding(4, 16)
        self.projection = nn.Sequential(nn.LayerNorm(1314), nn.Linear(1314, 256), nn.GELU())
        self.temporal = transformer()
        self.outputs = nn.ModuleList([nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 3)) for _ in range(3)])
        for head in self.outputs:
            zero_last(head)

    def forward(self, batch, scales):
        x = batch['x'].float()
        B, H, L, _ = x.shape
        identity = self.identity(torch.arange(4, device=x.device)).view(1, 1, 4, 16).expand(B, H, -1, -1)
        leg = (batch['base_leg'] - scales['leg_mean']) / scales['leg_std']
        force = (batch['force'] - scales['force_mean']) / scales['force_std']
        horizon = batch['horizon'].view(1, H, 1, 1).expand(B, -1, L, -1)
        y = self.projection(torch.cat([x, identity, leg, force, horizon], -1))
        y = self.temporal(y.permute(0, 2, 1, 3).reshape(B * L, H, 256))
        y = y.reshape(B, L, H, 256).permute(0, 2, 1, 3)
        delta = torch.cat([head(y) for head in self.outputs], -1)
        return batch['base_leg'] + delta * scales['leg_std'], delta


class BodyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.identity = nn.Embedding(4, 16)
        self.leg_projection = nn.Sequential(nn.LayerNorm(35), nn.Linear(35, 256), nn.GELU())
        self.query = nn.Linear(768, 256)
        self.leg_mixer = transformer()
        self.temporal = transformer()
        self.output = nn.Sequential(nn.LayerNorm(1036), nn.Linear(1036, 256), nn.GELU(), nn.Linear(256, 12))
        zero_last(self.output)

    def forward(self, batch, leg, r, foot_velocity, scales):
        B, H = leg.shape[:2]
        z = batch['x'][:, :, 0, :768].float()
        identity = self.identity(torch.arange(4, device=leg.device)).view(1, 1, 4, 16).expand(B, H, -1, -1)
        token = torch.cat([(leg - scales['leg_mean']) / scales['leg_std'],
                           (batch['force'] - scales['force_mean']) / scales['force_std'],
                           batch['contact'][..., None], (r - scales['r_mean']) / scales['r_std'],
                           (foot_velocity - scales['fv_mean']) / scales['fv_std'], identity], -1)
        token = self.leg_projection(token)
        mixed = torch.cat([self.query(z)[:, :, None], token], 2).reshape(B * H, 5, 256)
        body_token = self.leg_mixer(mixed)[:, 0].reshape(B, H, 256)
        body_token = self.temporal(body_token)
        base = (batch['base_body'] - scales['body_mean']) / scales['body_std']
        delta = self.output(torch.cat([body_token, z, base], -1))
        return batch['base_body'] + delta * scales['body_std'], delta


class Hybrid(nn.Module):
    def __init__(self, fk):
        super().__init__()
        self.legs = LegHead()
        self.body = BodyHead()
        self.fk = fk

    def forward(self, batch, scales, variant='hybrid'):
        if variant in ('base', 'body_only'):
            leg, ld = batch['base_leg'], torch.zeros_like(batch['base_leg'])
        else:
            leg, ld = self.legs(batch, scales)
        r, J = self.fk(leg[..., :3])
        fv = (J @ leg[..., 3:6, None]).squeeze(-1)
        if variant in ('base', 'leg_only'):
            body, bd = batch['base_body'], torch.zeros_like(batch['base_body'])
        else:
            body, bd = self.body(batch, leg, r, fv, scales)
        return dict(leg=leg, body=body, r=r, fv=fv, leg_delta=ld, body_delta=bd)


def objective(out, batch, scales, fk, stage, stance_enabled):
    def e(a, b, scale):
        return F.smooth_l1_loss((a - b) / scale, torch.zeros_like(a), beta=0.5)
    target_r, _ = fk(batch['target_leg'][..., :3])
    q = e(out['leg'][..., :3], batch['target_leg'][..., :3], scales['leg_std'][:3])
    dq = e(out['leg'][..., 3:6], batch['target_leg'][..., 3:6], scales['leg_std'][3:6])
    tau = e(out['leg'][..., 6:9], batch['target_leg'][..., 6:9], scales['leg_std'][6:9])
    foot = e(out['r'], target_r, scales['r_std'])
    body = e(out['body'], batch['target_body'], scales['body_std'])
    residual = out['body'][..., None, 6:9] + torch.cross(out['body'][..., None, 9:12].expand_as(out['r']), out['r'], dim=-1) + out['fv']
    raw = F.smooth_l1_loss(residual / scales['stance_std'], torch.zeros_like(residual), beta=0.5, reduction='none').mean(-1)
    mask = batch['target_contact']
    stance = (raw * mask).sum() / mask.sum().clamp_min(1)
    leg_supervised = q + dq + tau + 0.2 * foot
    leg_total = leg_supervised + 0.01 * out['leg_delta'].square().mean()
    body_total = body + 0.01 * out['body_delta'].square().mean()
    if stage == 'legs':
        total, selection = leg_total, leg_supervised
    elif stage in ('body', 'body_only'):
        total, selection = body_total, body
    else:
        total = leg_total + body_total + (0.1 * stance if stance_enabled else 0)
        selection = leg_supervised + body
    return total, selection, dict(q=q, dq=dq, tau=tau, foot=foot, body=body, stance=stance)
