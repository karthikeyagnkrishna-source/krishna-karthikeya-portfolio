"""Reuse V7 with timestamped visual/robot memory and a zero warm-start adapter."""
import torch
from torch import nn
from torch.nn import functional as F
from .data import HistoryBatch

V7_ARGS = ('p_hist', 'b_hist', 'u_hist', 'l_hist', 'e_hist', 'dino', 'conv', 'geom',
           'flow', 'foot_query_kin', 'future_control_features', 'horizon_values', 'physics_features')


def history_sequence(encoder, values):
    """The original CausalHistoryEncoder forward, retaining every time output."""
    steps = values.shape[1]
    x = encoder.input_proj(values)+encoder.pos[:, :steps]
    mask = torch.triu(torch.ones(steps, steps, device=x.device, dtype=torch.bool), diagonal=1)
    return encoder.norm(encoder.tx(x, mask=mask))


def initial_queries(terrain, p, u, feet):
    batch = p.shape[0]
    cond = torch.cat([p, u[:, None].expand(-1, 4, -1), feet], dim=-1)
    ids = torch.arange(4, device=p.device)[None].expand(batch, -1)
    return torch.cat([terrain.global_query.expand(batch, -1, -1),
                      terrain.leg_query(ids)+terrain.state_to_query(cond)], dim=1)


class HistoryAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.time = nn.Sequential(nn.Linear(1, 64), nn.GELU(), nn.Linear(64, 256))
        self.leg_project = nn.Linear(128, 256)
        self.body_project = nn.Linear(128, 256)
        self.stream = nn.Parameter(torch.randn(1, 1, 5, 256)*.02)
        self.output = nn.Linear(256, 256)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, terrain, history, legs, body, flow, queries, return_attention=False):
        batch, frames = history.valid.shape
        branches = []
        for i, (resampler, raw) in enumerate(zip((terrain.dino, terrain.conv, terrain.geom),
                                                (history.dino, history.conv, history.geom))):
            raw = torch.where(history.valid[:, :, None, None], raw, torch.zeros_like(raw))
            tokens = resampler(raw.flatten(0, 1)).reshape(batch, frames, -1, 256)
            branches.append(tokens+terrain.modality[:, i, None])
        images = torch.cat(branches, dim=2)
        per_image = images.shape[2]
        images = images+self.time(history.visual_times.to(images.dtype).unsqueeze(-1))[:, :, None]
        images = images.flatten(1, 2)
        image_mask = ~history.valid[:, :, None].expand(-1, -1, per_image).reshape(batch, -1)
        flow_tokens = terrain.flow(terrain.flow_temporal(flow))+terrain.modality[:, 3]
        flow_tokens = flow_tokens+self.time(history.visual_times[:, -1:].to(flow_tokens.dtype).unsqueeze(-1))
        robot = torch.cat([self.leg_project(legs.permute(0, 2, 1, 3)),
                           self.body_project(body)[:, :, None]], dim=2)
        robot = robot+self.stream+self.time(history.robot_times.to(robot.dtype).unsqueeze(-1))[:, :, None]
        robot = robot.flatten(1, 2)
        memory = torch.cat([images, flow_tokens, robot], dim=1)
        padding = torch.cat([image_mask, torch.zeros(batch, flow_tokens.shape[1]+robot.shape[1],
                                                   dtype=torch.bool, device=memory.device)], dim=1)
        q = queries+self.time(torch.zeros(batch, 1, 1, device=queries.device, dtype=queries.dtype))
        weights = None
        # Reuse the original blocks, adding a key-padding mask for missing images.
        # Self-attention is between five queries, never between all memory tokens.
        for index, block in enumerate(terrain.blocks):
            want_weights = return_attention and index == len(terrain.blocks)-1
            attention, weights = block.cross(block.qn(q), block.vn(memory), block.vn(memory),
                key_padding_mask=padding, need_weights=want_weights)
            q = q+attention
            attention, _ = block.selfa(block.sn(q), block.sn(q), block.sn(q), need_weights=False)
            q = q+attention
            q = q+block.ff(q)
        diagnostic = dict(memory_tokens=memory.shape[1], visual_tokens=images.shape[1]+flow_tokens.shape[1],
                          robot_tokens=robot.shape[1], padding_mask=padding,
                          visual_times=history.visual_times, robot_times=history.robot_times)
        if return_attention:
            diagnostic['attention'] = weights
        return self.output(terrain.norm(q)), diagnostic


class V7VisualHistory(nn.Module):
    """Frozen pretrained V7 plus trainable history adapters; original head contract."""
    def __init__(self, base, frames=6):
        super().__init__()
        if frames < 2:
            raise ValueError('Visual history requires a current and at least one earlier image')
        self.base = base.eval().requires_grad_(False)
        self.frames = int(frames)
        self.history_steps = base.history_steps
        self.n_horizons = base.n_horizons
        self.n_regimes = base.regime_head[-1].out_features//4
        self.adapter = HistoryAdapter().to(next(base.parameters()).device)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()  # Preserve dropout behavior of the frozen foundation.
        return self

    def encode_sample(self, sample, return_attention=False):
        history = sample.get('visual_history')
        if not isinstance(history, HistoryBatch):
            raise ValueError('Timestamped visual history is required; old context caches are incompatible')
        p, b, u, low, err = (sample[k] for k in V7_ARGS[:5])
        batch, _, steps, _ = p.shape
        if steps != self.history_steps or p.shape[1:] != (4, steps, 12):
            raise ValueError('Robot history does not match the saved V7 branch')
        if b.shape != (batch, steps, 12) or u.shape != (batch, steps, 6):
            raise ValueError('Invalid body/command history shape')
        if low.shape != (batch, 4, steps, 3) or err.shape != low.shape:
            raise ValueError('Invalid commanded-torque or tracking-error history shape')
        history.validate(batch, steps, self.frames)
        for name in ('dino', 'conv', 'geom'):
            if not torch.equal(sample[name], getattr(history, name)[:, -1]):
                raise ValueError('Current image encoding must match the last history frame')
        robot_input = torch.cat([p, u[:, None].expand(-1, 4, -1, -1), low, err], dim=-1)
        legs = history_sequence(self.base.leg_encoder, robot_input.reshape(batch*4, steps, 24))
        ids = torch.arange(4, device=p.device).repeat(batch)
        legs = (legs+self.base.leg_identity(ids)[:, None]).reshape(batch, 4, steps, 128)
        body = history_sequence(self.base.body_encoder, torch.cat([b, u], dim=-1))
        current_p, current_u = p[:, :, -1], u[:, -1]
        terrain = self.base.terrain_encoder
        vg, vf = terrain(sample['dino'], sample['conv'], sample['geom'], sample['flow'],
                         current_p, current_u, sample['foot_query_kin'])
        q = initial_queries(terrain, current_p, current_u, sample['foot_query_kin'])
        delta, diagnostic = self.adapter(terrain, history, legs, body, sample['flow'], q, return_attention)
        corrected = torch.cat([vg[:, None], vf], dim=1)+delta
        vg, vf = corrected[:, 0], corrected[:, 1:]
        legf = self.base.leg_visual_fuse(torch.cat([legs[:, :, -1], vf], dim=-1))
        ctx = self.base.context_net(torch.cat([legf.reshape(batch, -1), body[:, -1], vg], dim=-1))
        if return_attention:
            return ctx, legf, vf, diagnostic
        return ctx, legf, vf

    def _head_features(self, sample, encoded):
        ctx, legf, vf = encoded[:3]
        count = sample['future_control_features'].shape[1]
        if count != self.n_horizons:
            raise ValueError('Prediction horizon count does not match the V7 branch')
        z = torch.cat([ctx[:, None].expand(-1, count, -1),
                       self.base.control_proj(sample['future_control_features']),
                       self.base.horizon_proj(sample['horizon_values']),
                       self.base.physics_proj(sample['physics_features'])], dim=-1)
        binary = self.base.binary_contact_head(z)
        regime = self.base.regime_head(z).reshape(-1, count, 4, self.n_regimes)
        probs = regime.softmax(-1)
        x = torch.cat([z[:, :, None].expand(-1, -1, 4, -1),
                       legf[:, None].expand(-1, count, -1, -1),
                       vf[:, None].expand(-1, count, -1, -1),
                       binary.sigmoid().unsqueeze(-1), probs], dim=-1)
        return z, x, binary, regime, probs

    def features(self, sample, return_attention=False):
        encoded = self.encode_sample(sample, return_attention)
        x = self._head_features(sample, encoded)[1]
        return (x, encoded[3]) if return_attention else x

    def forward_sample(self, sample):
        # Same V7 decode and residual layout as the inspected V7FullDynamics.
        z, x, binary, regime, probs = self._head_features(sample, self.encode_sample(sample))
        count = self.n_horizons
        sb = self.base.state_body_head(z)
        residual = torch.zeros(z.shape[0], count, 4, 12, device=z.device, dtype=z.dtype)
        residual[..., :6] = sb[..., :24].reshape(-1, count, 4, 6)
        residual[..., 6:] = self.base.tau_force_head(x)
        aux = self.base.force_aux_head(x)
        return dict(residual_p=residual, residual_b=sb[..., 24:], binary_contact_logits=binary,
                    regime_logits=regime, regime_probs=probs, force_aux_logmag=aux[..., 0],
                    force_aux_dir=F.normalize(aux[..., 1:4], dim=-1, eps=1e-6))

    def forward(self, sample):
        return self.forward_sample(sample)
