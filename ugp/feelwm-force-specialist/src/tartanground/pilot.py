from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class PerceiverResampler(nn.Module):
    """Exact V7 branch resampler used by the approved reference notebook."""

    def __init__(self, in_dim: int, d: int = 256, n_latents: int = 16, heads: int = 8, layers: int = 2):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, d), nn.GELU(), nn.LayerNorm(d)
        )
        self.latents = nn.Parameter(torch.randn(1, n_latents, d) * 0.02)
        self.cross = nn.ModuleList(
            [nn.MultiheadAttention(d, heads, batch_first=True) for _ in range(layers)]
        )
        self.lnq = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.lnkv = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.ffn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(d),
                    nn.Linear(d, 4 * d),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(4 * d, d),
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.adapter(x)
        q = self.latents.expand(x.shape[0], -1, -1)
        for attn, lq, lkv, ffn in zip(self.cross, self.lnq, self.lnkv, self.ffn):
            kv = lkv(x)
            update, _ = attn(lq(q), kv, kv, need_weights=False)
            q = q + update
            q = q + ffn(q)
        return q


class FlowTemporalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(6, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.GELU(),
        )
        self.out = nn.Linear(128, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, history, spatial, channels = x.shape
        y = x.permute(0, 2, 3, 1).reshape(batch * spatial, channels, history)
        y = self.net(y)[:, :, -1]
        return self.out(y).reshape(batch, spatial, 256)


class FrozenVisualResamplers(nn.Module):
    """Four post-Perceiver blocks in fixed DINO/Conv/geometry/flow order."""

    def __init__(self, dino_dim: int = 1025, conv_dim: int = 2881, geom_dim: int = 1234):
        super().__init__()
        self.dino = PerceiverResampler(dino_dim)
        self.conv = PerceiverResampler(conv_dim)
        self.geom = PerceiverResampler(geom_dim)
        self.flow_temporal = FlowTemporalEncoder()
        self.flow = PerceiverResampler(256)

    def forward(
        self,
        dino: torch.Tensor,
        conv: torch.Tensor,
        geom: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        branches = (
            self.dino(dino),
            self.conv(conv),
            self.geom(geom),
            self.flow(self.flow_temporal(flow)),
        )
        return torch.cat(branches, dim=1)


def load_v7_resamplers(checkpoint: str | Path, device: torch.device) -> tuple[FrozenVisualResamplers, str]:
    checkpoint = Path(checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    prefixes = ("dino.", "conv.", "geom.", "flow_temporal.", "flow.")
    selected = {
        key.removeprefix("terrain_encoder."): value
        for key, value in state.items()
        if key.startswith("terrain_encoder.")
        and key.removeprefix("terrain_encoder.").startswith(prefixes)
    }
    model = FrozenVisualResamplers()
    missing, unexpected = model.load_state_dict(selected, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"V7 resampler checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.eval().requires_grad_(False).to(device)
    return model, digest


def build_body_action(u: np.ndarray, dt: float = 0.1) -> np.ndarray:
    u = np.asarray(u, np.float32)
    return np.asarray([u[0] * dt, u[1] * dt, u[5] * dt, dt], np.float32)


def build_leg_action(q_cmd_now: np.ndarray, q_cmd_next: np.ndarray, tau_cmd_now: np.ndarray) -> np.ndarray:
    q_cmd_now = np.asarray(q_cmd_now, np.float32)
    q_cmd_next = np.asarray(q_cmd_next, np.float32)
    tau_cmd_now = np.asarray(tau_cmd_now, np.float32)
    delta = q_cmd_next - q_cmd_now
    return np.stack(
        [delta[:, 1], delta[:, 2], tau_cmd_now[:, 1], tau_cmd_now[:, 2]], axis=-1
    ).astype(np.float32)


def contact_phase(current: bool, future: bool) -> int:
    if not current and not future:
        return 0
    if not current and future:
        return 1
    if current and future:
        return 2
    return 3


def phase_vector(current: np.ndarray, future: np.ndarray) -> np.ndarray:
    return np.asarray(
        [contact_phase(bool(a), bool(b)) for a, b in zip(current, future)], dtype=np.int64
    )
