# CELL 6


# CELL 8
from __future__ import annotations
import os
import random
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def rng_state() -> Dict[str, Any]:
    out = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        out["cuda"] = torch.cuda.get_rng_state_all()
    return out


def restore_rng_state(state: Dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].detach().to(device="cpu", dtype=torch.uint8))
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([item.detach().to(device="cpu", dtype=torch.uint8) for item in state["cuda"]])


def atomic_torch_save(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(path: str | Path, model, optimizer, scheduler, scaler, epoch: int,
                    global_step: int, best_val: float, config: Dict[str, Any]) -> None:
    pack = {
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_val": float(best_val),
        "rng": rng_state(),
        "config": config,
    }
    atomic_torch_save(pack, path)


def load_checkpoint(path: str | Path, model, optimizer=None, scheduler=None, scaler=None, map_location="cpu"):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    unwrap_model(model).load_state_dict(ck["model"], strict=True)
    if optimizer is not None and ck.get("optimizer") is not None:
        optimizer.load_state_dict(ck["optimizer"])
    if scheduler is not None and ck.get("scheduler") is not None:
        scheduler.load_state_dict(ck["scheduler"])
    if scaler is not None and ck.get("scaler") is not None:
        scaler.load_state_dict(ck["scaler"])
    restore_rng_state(ck.get("rng"))
    return ck


def resolve_resume(resume: str, output_dir: str | Path) -> Path | None:
    if resume.lower() == "none":
        return None
    if resume.lower() == "auto":
        p = Path(output_dir) / "checkpoints" / "latest.pt"
        return p if p.exists() else None
    p = Path(resume)
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def prune_step_checkpoints(checkpoint_dir: str | Path, keep: int) -> None:
    p = Path(checkpoint_dir)
    files = sorted(p.glob("step_*.pt"), key=lambda x: x.stat().st_mtime)
    for old in files[:-max(keep, 0)]:
        try:
            old.unlink()
        except OSError:
            pass


# CELL 10
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict
import yaml


@dataclass
class ModelConfig:
    # Merged visual latent state: DINO / ConvNeXt / geometry / flow.
    num_modalities: int = 4
    tokens_per_modality: int = 16
    latent_dim: int = 256
    num_visual_frames: int = 4

    # 9 Feel-WM groups:
    # Z(t-3), HL@Z(t-3), HR@Z(t-3), Z(t-2), FL@Z(t-2), FR@Z(t-2), Z(t-1), Z(t), Body@Z(t)
    num_groups: int = 9

    # Requested high-capacity world transformer.
    world_d_model: int = 800       # 800 / 10 = 80 dims/head
    world_heads: int = 10
    world_layers: int = 8
    world_ffn_ratio: int = 7

    # Optional independent leg-local processing between FiLM and the shared
    # world transformer. Defaults preserve the original baseline architecture.
    use_leg_pretransformers: bool = False
    leg_pre_d_model: int = 384
    leg_pre_heads: int = 8
    leg_pre_layers: int = 4
    leg_pre_ffn_ratio: int = 4
    leg_pre_residual_scale: float = 0.1

    # V7 history encoders.
    hist_d_model: int = 128
    hist_heads: int = 4
    hist_layers: int = 2
    hist_ffn_ratio: int = 4

    # V7 terrain query blocks.
    query_dim: int = 256
    query_heads: int = 8
    query_layers: int = 4
    query_ffn_ratio: int = 4

    # Inputs.
    leg_hist_dim: int = 24         # proprio12 + U6 + low-level3 + error3
    body_hist_dim: int = 18        # body12 + U6
    leg_query_dim: int = 24        # current proprio12 + current U6 + foot kinematics6
    body_film_action_dim: int = 4
    leg_film_action_dim: int = 4
    future_action_dim: int = 36
    physics_dim: int = 47

    # V7 / V7-F.
    num_legs: int = 4
    num_load_regimes: int = 4
    num_contact_phases: int = 4
    force_ctx_dim: int = 1285
    v7f_dim: int = 256
    v7f_heads: int = 8
    v7f_layers: int = 2
    v7f_experts: int = 4

    dropout: float = 0.10
    eps: float = 1e-6

    @property
    def tokens_per_frame(self) -> int:
        return self.num_modalities * self.tokens_per_modality

    @property
    def world_tokens(self) -> int:
        return self.num_groups * self.tokens_per_frame


@dataclass
class TrainConfig:
    train_manifest: str = "data/train_manifest.jsonl"
    val_manifest: str = "data/val_manifest.jsonl"
    output_dir: str = "runs/merged_feelwm_v7"

    seed: int = 781
    epochs: int = 30
    batch_size: int = 2
    grad_accum_steps: int = 4
    num_workers: int = 6
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True

    lr: float = 2e-4
    min_lr_ratio: float = 0.05
    weight_decay: float = 0.05
    warmup_steps: int = 1000
    grad_clip: float = 1.0

    # RTX 4080: bf16 is normally the best default.
    amp_dtype: str = "bf16"       # bf16 | fp16 | fp32
    tf32: bool = True
    compile: bool = False

    log_every: int = 10
    val_every_steps: int = 1000
    checkpoint_every_steps: int = 250
    keep_last_checkpoints: int = 4
    resume: str = "auto"          # auto | none | /path/to/checkpoint.pt
    max_steps: int | None = None   # exact optimizer-step cap; None keeps epoch-only behavior
    init_weights_only: str | None = None  # checkpoint path; fresh optimizer/scheduler
    teacher_forcing: bool = False
    horizon_lambdas: tuple[float, ...] = ()
    indexed_data: bool = False

    # The recurrent model step can be 0.1 s while proprio is sampled faster.
    # If your prepared histories are already on the model-step grid, keep 1.
    # If histories are 100 Hz and world steps are 10 Hz, use 10.
    history_shift_samples: int = 1

    # Loss weights.
    w_dino: float = 1.0
    w_conv: float = 1.0
    w_geom: float = 1.0
    w_flow: float = 1.0
    w_contact: float = 0.50
    w_load: float = 0.25
    w_jointbody: float = 1.0
    w_tau: float = 1.0
    w_base_force: float = 1.0
    w_v7f_force: float = 1.0

    # Base V7 magnitude/direction auxiliary head. These outputs are part of the
    # architecture and must be supervised so every DDP parameter participates.
    v7_w_logmag: float = 0.10
    v7_w_dir: float = 0.02

    # V7-F specialist weights retained from the source force-specialist design.
    v7f_w_vec: float = 2.00
    v7f_w_mag: float = 1.60
    v7f_w_dir: float = 0.10
    v7f_w_phase: float = 0.25
    v7f_w_temp: float = 0.45
    v7f_w_impulse: float = 0.35
    v7f_w_anchor: float = 0.015
    v7f_w_gate: float = 0.005
    force_contact_threshold: float = 20.0
    rollout_dt: float = 0.1


def load_config(path: str | Path) -> tuple[ModelConfig, TrainConfig, Dict[str, Any]]:
    path = Path(path)
    raw: Dict[str, Any] = yaml.safe_load(path.read_text())
    model_cfg = ModelConfig(**raw.get("model", {}))
    train_cfg = TrainConfig(**raw.get("train", {}))
    return model_cfg, train_cfg, raw


def config_to_dict(model_cfg: ModelConfig, train_cfg: TrainConfig) -> Dict[str, Any]:
    return {"model": asdict(model_cfg), "train": asdict(train_cfg)}


# CELL 28
from __future__ import annotations
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Any
import torch
from torch.utils.data import Dataset


REQUIRED_KEYS = {
    "z_window",                    # [4,64,256] post-Perceiver visual state
    "leg_hist_t",                  # [4,T,24]
    "leg_hist_tm1",                # [4,T,24]
    "leg_hist_tm2",                # [4,T,24]
    "body_hist_t",                 # [T,18]
    "body_action_seq",             # [K,4]
    "leg_action_seq",              # [K,4,4] -- CURRENT action for each rollout step
    "leg_query_features_seq",      # [K,4,24]
    "future_action_seq",           # [K,36]
    "physics_seq",                 # [K,47]
    "horizon_seq",                 # [K,1]
    "lowlevel_cmd_seq",            # [K,4,3]
    "err_seq",                     # [K,4,3]
    "jointbody_baseline_seq",      # [K,36]
    "jointbody_std",               # [36]
    "tau_baseline_seq",            # [K,4,3]
    "tau_std",                     # [3]
    "force_baseline_seq",          # [K,4,3]
    "force_std",                   # [3]
    "target_z_seq",                # [K,64,256] = DINO|Conv|Geom|Flow targets
    "target_contact_seq",          # [K,4] float 0/1
    "target_load_class_seq",       # [K,4] long 0..3
    "target_jointbody_seq",        # [K,36]
    "target_tau_seq",              # [K,4,3]
    "target_force_seq",            # [K,4,3]
    "target_phase_seq",            # [K,4] long 0..3
}


def validate_sample(sample: Dict[str, Any]) -> None:
    missing = REQUIRED_KEYS - set(sample)
    if missing:
        raise KeyError(f"Sample missing required keys: {sorted(missing)}")
    if tuple(sample["z_window"].shape[-3:]) != (4, 64, 256):
        raise ValueError(f"z_window must be [4,64,256], got {tuple(sample['z_window'].shape)}")
    if sample["leg_action_seq"].shape[-2:] != (4, 4):
        raise ValueError("leg_action_seq must be [K,4,4] in leg order FL,FR,HL,HR")
    if sample["target_z_seq"].shape[-2:] != (64, 256):
        raise ValueError("target_z_seq must be [K,64,256]")


class ShardedWorldModelDataset(Dataset):
    """
    Manifest JSONL format, one row per sample:
      {"path":"/abs/path/shard_000.pt", "index":17}

    Each shard is:
      torch.save({"samples": [sample0, sample1, ...]}, shard_path)

    A small per-worker LRU cache avoids reloading a shard for every sample.
    """
    def __init__(self, manifest: str | Path, cache_shards: int = 2, validate_first: int = 2):
        self.manifest = Path(manifest)
        if not self.manifest.exists():
            raise FileNotFoundError(self.manifest)
        self.rows = [json.loads(line) for line in self.manifest.read_text().splitlines() if line.strip()]
        if not self.rows:
            raise RuntimeError(f"Empty manifest: {self.manifest}")
        self.cache_shards = max(1, int(cache_shards))
        self._cache: OrderedDict[str, Any] = OrderedDict()
        for i in range(min(validate_first, len(self.rows))):
            validate_sample(self[i])

    def __len__(self) -> int:
        return len(self.rows)

    def _load_shard(self, path: str):
        if path in self._cache:
            obj = self._cache.pop(path)
            self._cache[path] = obj
            return obj
        obj = torch.load(path, map_location="cpu", weights_only=False)
        self._cache[path] = obj
        while len(self._cache) > self.cache_shards:
            self._cache.popitem(last=False)
        return obj

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        shard = self._load_shard(row["path"])
        sample = shard["samples"][int(row["index"])]
        # Convert all arrays/scalars to tensors once here.
        out: Dict[str, torch.Tensor] = {}
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                out[k] = v
            else:
                out[k] = torch.as_tensor(v)
        return out


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    long_keys = {"target_load_class_seq", "target_phase_seq"}
    for k, v in batch.items():
        if not isinstance(v, torch.Tensor):
            continue
        if k in long_keys:
            v = v.long()
        elif not torch.is_floating_point(v):
            v = v.float()
        out[k] = v.to(device, non_blocking=True)
    return out


# CELL 32
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F

from config import ModelConfig, TrainConfig


def cosine_latent_loss(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = F.normalize(x, dim=-1, eps=eps)
    y = F.normalize(y, dim=-1, eps=eps)
    return 1.0 - (x * y).sum(-1).mean()


def normalized_smooth_l1(pred: torch.Tensor, target: torch.Tensor, scale: torch.Tensor,
                         beta: float = 0.35) -> torch.Tensor:
    while scale.dim() < pred.dim():
        scale = scale.unsqueeze(1)
    err = (pred - target) / (scale + 1e-6)
    return F.smooth_l1_loss(err, torch.zeros_like(err), beta=beta)


def _horizon_weights(configured, horizons: int, device, dtype) -> torch.Tensor:
    if configured:
        values = torch.as_tensor(configured, device=device, dtype=dtype)
        if values.numel() != horizons:
            raise ValueError(f"expected {horizons} horizon lambdas, got {values.numel()}")
        if bool((values < 0).any()) or not torch.isclose(values.sum(), values.new_tensor(1.0), atol=1e-6):
            raise ValueError("horizon lambdas must be nonnegative and sum to one")
        return values
    return torch.full((horizons,), 1.0 / horizons, device=device, dtype=dtype)


def compute_losses(
    out: Dict[str, Any], batch: Dict[str, torch.Tensor], mcfg: ModelConfig, tcfg: TrainConfig
) -> Dict[str, torch.Tensor]:
    """Compute supervised loss per forecast horizon, then apply configured lambdas."""
    steps = out["steps"]
    K = len(steps)
    n = mcfg.tokens_per_modality
    spec = out["v7f"]
    pred_force = spec["force"]
    gt_force = batch["target_force_seq"]
    target_phase = batch["target_phase_seq"]
    force_scale = batch["force_std"][:, None, None, :]
    gt_mag = torch.linalg.norm(gt_force, dim=-1)
    pred_mag = torch.linalg.norm(pred_force, dim=-1)
    contact = (gt_mag >= tcfg.force_contact_threshold).float()
    direction_target = F.normalize(gt_force, dim=-1, eps=mcfg.eps)
    direction_pred = F.normalize(pred_force, dim=-1, eps=mcfg.eps)
    direction_cosine = (direction_target * direction_pred).sum(-1)
    scale = gt_mag.detach().median().clamp_min(1.0)
    phase_boost = ((target_phase == 1) | (target_phase == 3)).float()
    force_weight = (1.0 + 1.25 * (gt_mag / scale).clamp(0, 2.0) + 0.50 * phase_boost).detach()

    per_horizon: list[torch.Tensor] = []
    components: dict[str, list[torch.Tensor]] = {}

    for k, step in enumerate(steps):
        target_z = batch["target_z_seq"][:, k]
        zd = target_z[:, 0 * n : 1 * n]
        zc = target_z[:, 1 * n : 2 * n]
        zg = target_z[:, 2 * n : 3 * n]
        zf = target_z[:, 3 * n : 4 * n]
        terms: dict[str, torch.Tensor] = {
            "dino": F.mse_loss(step["dino"], zd) + 0.1 * cosine_latent_loss(step["dino"], zd),
            "conv": F.mse_loss(step["conv"], zc) + 0.1 * cosine_latent_loss(step["conv"], zc),
            "geom": F.smooth_l1_loss(step["geom"], zg) + 0.1 * cosine_latent_loss(step["geom"], zg),
            "flow": F.smooth_l1_loss(step["flow"], zf) + 0.1 * cosine_latent_loss(step["flow"], zf),
            "contact": F.binary_cross_entropy_with_logits(
                step["contact_logits"], batch["target_contact_seq"][:, k]
            ),
            "load": F.cross_entropy(
                step["regime_logits"].reshape(-1, mcfg.num_load_regimes),
                batch["target_load_class_seq"][:, k].reshape(-1),
            ),
            "jointbody": normalized_smooth_l1(
                step["jointbody_pred"], batch["target_jointbody_seq"][:, k], batch["jointbody_std"]
            ),
            "tau": normalized_smooth_l1(
                step["tau_pred"], batch["target_tau_seq"][:, k], batch["tau_std"]
            ),
            "base_force": normalized_smooth_l1(step["base_force"], gt_force[:, k], batch["force_std"]),
        }

        target_magnitude = gt_mag[:, k]
        aux_mag = F.smooth_l1_loss(
            step["force_aux_logmag"], torch.log1p(target_magnitude), beta=0.25
        )
        meaningful = (target_magnitude >= tcfg.force_contact_threshold).float()
        aux_cos = (step["force_aux_dir"] * direction_target[:, k]).sum(-1)
        aux_dir = ((1.0 - aux_cos) * meaningful).sum() / (meaningful.sum() + 1e-6)
        terms["base_force_aux"] = tcfg.v7_w_logmag * aux_mag + tcfg.v7_w_dir * aux_dir

        force_error = (pred_force[:, k] - gt_force[:, k]) / (batch["force_std"][:, None, :] + 1e-6)
        v7f_vec = (
            F.smooth_l1_loss(force_error, torch.zeros_like(force_error), reduction="none", beta=0.35).mean(-1)
            * force_weight[:, k]
        ).mean()
        v7f_mag = F.smooth_l1_loss(torch.log1p(pred_mag[:, k]), torch.log1p(gt_mag[:, k]), beta=0.20)
        v7f_dir = ((1.0 - direction_cosine[:, k]) * contact[:, k]).sum() / (contact[:, k].sum() + 1e-6)
        v7f_phase = F.cross_entropy(
            spec["phase_logits"][:, k].reshape(-1, mcfg.num_contact_phases), target_phase[:, k].reshape(-1)
        )
        v7f_anchor = (spec["corr_norm"][:, k] ** 2).mean()
        mean_gate = spec["expert_w"][:, k].mean(dim=(0, 1))
        uniform = torch.full_like(mean_gate, 1.0 / len(mean_gate))
        v7f_gate = ((mean_gate - uniform) ** 2).mean()
        terms.update({
            "v7f_vec": v7f_vec,
            "v7f_mag": v7f_mag,
            "v7f_dir": v7f_dir,
            "v7f_phase": v7f_phase,
            "v7f_anchor": v7f_anchor,
            "v7f_gate": v7f_gate,
        })
        terms["v7f"] = (
            tcfg.v7f_w_vec * v7f_vec + tcfg.v7f_w_mag * v7f_mag
            + tcfg.v7f_w_dir * v7f_dir + tcfg.v7f_w_phase * v7f_phase
            + tcfg.v7f_w_anchor * v7f_anchor + tcfg.v7f_w_gate * v7f_gate
        )

        horizon_loss = (
            tcfg.w_dino * terms["dino"] + tcfg.w_conv * terms["conv"]
            + tcfg.w_geom * terms["geom"] + tcfg.w_flow * terms["flow"]
            + tcfg.w_contact * terms["contact"] + tcfg.w_load * terms["load"]
            + tcfg.w_jointbody * terms["jointbody"] + tcfg.w_tau * terms["tau"]
            + tcfg.w_base_force * (terms["base_force"] + terms["base_force_aux"])
            + tcfg.w_v7f_force * terms["v7f"]
        )
        per_horizon.append(horizon_loss)
        for name, value in terms.items():
            components.setdefault(name, []).append(value)

    # Temporal smoothness and impulse preservation couple horizons. They are
    # explicit regularizers outside the lambda-weighted supervised L_k sum.
    if K > 1:
        dp = (pred_force[:, 1:] - pred_force[:, :-1]) / (force_scale + 1e-6)
        dg = (gt_force[:, 1:] - gt_force[:, :-1]) / (force_scale + 1e-6)
        temporal = F.smooth_l1_loss(dp, dg, beta=0.35)
        predicted_impulse = (pred_force * tcfg.rollout_dt).sum(1)
        target_impulse = (gt_force * tcfg.rollout_dt).sum(1)
        impulse = F.smooth_l1_loss(
            (predicted_impulse - target_impulse) / (batch["force_std"][:, None, :] + 1e-6),
            torch.zeros_like(predicted_impulse), beta=0.35,
        )
    else:
        temporal = pred_force.sum() * 0.0
        impulse = pred_force.sum() * 0.0
    temporal_regularizer = tcfg.w_v7f_force * tcfg.v7f_w_temp * temporal
    impulse_regularizer = tcfg.w_v7f_force * tcfg.v7f_w_impulse * impulse

    lambdas = _horizon_weights(tcfg.horizon_lambdas, K, pred_force.device, torch.float32)
    total = sum(lambdas[k] * per_horizon[k] for k in range(K))
    total = total + temporal_regularizer + impulse_regularizer
    result: Dict[str, torch.Tensor] = {"total": total}
    for name, values in components.items():
        result[name] = torch.stack(values).mean()
    for k, value in enumerate(per_horizon):
        result[f"horizon_{k + 1}"] = value
        result[f"lambda_{k + 1}"] = lambdas[k]
    result["temporal_regularizer"] = temporal_regularizer
    result["impulse_regularizer"] = impulse_regularizer
    result["v7f_temp"] = temporal
    result["v7f_impulse"] = impulse
    return result


# CELL 34
from __future__ import annotations
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


# Leg order is fixed everywhere in this project.
FL, FR, HL, HR = 0, 1, 2, 3


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0, norm_in: bool = False):
        super().__init__()
        layers: List[nn.Module] = []
        if norm_in:
            layers.append(nn.LayerNorm(in_dim))
        layers += [nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ratio: int, dropout: float):
        super().__init__()
        hidden = d_model * ratio
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, ffn_ratio: int, dropout: float):
        super().__init__()
        assert d_model % heads == 0, f"d_model={d_model} must be divisible by heads={heads}"
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_ratio, dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + h
        x = x + self.ffn(self.ln2(x))
        return x


class CausalHistoryEncoder(nn.Module):
    """Faithful V7 history encoder: projection -> 2-layer causal Transformer -> last token."""
    def __init__(self, input_dim: int, cfg: ModelConfig, max_steps: int = 512):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, cfg.hist_d_model)
        self.pos = nn.Parameter(torch.randn(1, max_steps, cfg.hist_d_model) * 0.02)
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(cfg.hist_d_model, cfg.hist_heads, cfg.hist_ffn_ratio, cfg.dropout)
            for _ in range(cfg.hist_layers)
        ])
        self.norm = nn.LayerNorm(cfg.hist_d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [B,T,D]
        T = x.shape[1]
        if T > self.pos.shape[1]:
            raise ValueError(f"history length {T} exceeds max {self.pos.shape[1]}")
        h = self.input_proj(x) + self.pos[:, :T]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        for block in self.blocks:
            h = block(h, mask)
        return self.norm(h)[:, -1]


class QueryFusionBlockAligned(nn.Module):
    """
    Five V7 queries, but each query cross-attends to its temporally aligned 64-token bank:
      global -> Body@Z_t (G8)
      FL     -> FL@Z_t-2 (G4)
      FR     -> FR@Z_t-2 (G5)
      HL     -> HL@Z_t-3 (G1)
      HR     -> HR@Z_t-3 (G2)
    Then the 5 queries self-attend exactly as V7 does.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.query_dim
        self.qn = nn.LayerNorm(d)
        self.vn = nn.LayerNorm(d)
        self.cross = nn.MultiheadAttention(d, cfg.query_heads, dropout=cfg.dropout, batch_first=True)
        self.sn = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, cfg.query_heads, dropout=cfg.dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, cfg.query_ffn_ratio * d), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.query_ffn_ratio * d, d)
        )

    def forward(self, q: torch.Tensor, banks: torch.Tensor) -> torch.Tensor:
        # q [B,5,256], banks [B,5,64,256]
        B, Q, N, D = banks.shape
        q_flat = self.qn(q).reshape(B * Q, 1, D)
        v_flat = self.vn(banks).reshape(B * Q, N, D)
        a, _ = self.cross(q_flat, v_flat, v_flat, need_weights=False)
        q = q + a.reshape(B, Q, D)
        qn = self.sn(q)
        a, _ = self.self_attn(qn, qn, qn, need_weights=False)
        q = q + a
        return q + self.ffn(q)


def build_physical_time_mask(tokens_per_group: int, device: torch.device) -> torch.Tensor:
    # G0,G1,G2=t-3; G3,G4,G5=t-2; G6=t-1; G7,G8=t
    group_times = [0, 0, 0, 1, 1, 1, 2, 3, 3]
    times = torch.tensor([t for g in group_times for t in [g] * tokens_per_group], device=device)
    # True means forbidden in torch MultiheadAttention.
    return times[None, :] > times[:, None]


class FiLMGenerator(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 512), nn.GELU(), nn.Linear(512, 2 * cfg.world_d_model)
        )
        # Identity FiLM at initialization, like the original idea.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.net(c).chunk(2, dim=-1)


class LegPreTransformer(nn.Module):
    """Independent per-leg token transformer with a small residual update."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.input_proj = nn.Linear(cfg.world_d_model, cfg.leg_pre_d_model)
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(
                cfg.leg_pre_d_model, cfg.leg_pre_heads,
                cfg.leg_pre_ffn_ratio, cfg.dropout,
            )
            for _ in range(cfg.leg_pre_layers)
        ])
        self.output_proj = nn.Linear(cfg.leg_pre_d_model, cfg.world_d_model)
        self.residual_scale = cfg.leg_pre_residual_scale

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(tokens)
        for block in self.blocks:
            h = block(h)
        return tokens + self.residual_scale * self.output_proj(h)


def film(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + gamma[:, None]) + beta[:, None]


class V7ForceSpecialist(nn.Module):
    """V7-F specialist retained from the force-specialist notebook."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.v7f_dim
        self.cfg = cfg
        self.in_proj = nn.Sequential(
            nn.LayerNorm(cfg.force_ctx_dim), nn.Linear(cfg.force_ctx_dim, d), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d)
        )
        self.horizon_blocks = nn.ModuleList([
            SelfAttentionBlock(d, cfg.v7f_heads, 4, cfg.dropout) for _ in range(cfg.v7f_layers)
        ])
        self.phase_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 128), nn.GELU(), nn.Linear(128, cfg.num_contact_phases))
        self.experts = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 192), nn.GELU(), nn.Dropout(.08),
                          nn.Linear(192, 96), nn.GELU(), nn.Linear(96, 3))
            for _ in range(cfg.v7f_experts)
        ])
        gate_dim = d + cfg.num_contact_phases + cfg.num_load_regimes + 1
        self.expert_gate = nn.Sequential(nn.LayerNorm(gate_dim), nn.Linear(gate_dim, 128), nn.GELU(), nn.Linear(128, cfg.v7f_experts))
        self.direct_corr = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 192), nn.GELU(), nn.Linear(192, 3))
        self.magdir = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 192), nn.GELU(), nn.Linear(192, 4))
        self.fusion_gate = nn.Sequential(nn.LayerNorm(gate_dim), nn.Linear(gate_dim, 96), nn.GELU(), nn.Linear(96, 1))

        # Start exactly at the base V7 force prediction.
        for mod in list(self.experts) + [self.direct_corr, self.magdir]:
            last = [m for m in mod.modules() if isinstance(m, nn.Linear)][-1]
            nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
        last = [m for m in self.fusion_gate.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor, base_force: torch.Tensor, force_std: torch.Tensor,
                pcontact: torch.Tensor, regime: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x [B,K,4,1285]
        B, K, L, _ = x.shape
        h = self.in_proj(x)
        ht = h.permute(0, 2, 1, 3).reshape(B * L, K, self.cfg.v7f_dim)
        causal = torch.triu(torch.ones(K, K, dtype=torch.bool, device=x.device), diagonal=1)
        for block in self.horizon_blocks:
            ht = block(ht, causal)
        h = ht.reshape(B, L, K, self.cfg.v7f_dim).permute(0, 2, 1, 3).contiguous()

        phase_logits = self.phase_head(h)
        phase_prob = phase_logits.softmax(-1)
        gate_in = torch.cat([h, phase_prob, regime, pcontact.unsqueeze(-1)], dim=-1)
        expert_w = self.expert_gate(gate_in).softmax(-1)
        exp = torch.stack([e(h) for e in self.experts], dim=-2)
        moe_corr = (expert_w.unsqueeze(-1) * exp).sum(-2)
        direct_corr = self.direct_corr(h)
        corr_norm = moe_corr + direct_corr

        # Normalized specialist correction -> physical force units.
        direct_force = base_force + corr_norm * force_std[:, None, None, :]

        md = self.magdir(h)
        base_mag = torch.linalg.norm(base_force, dim=-1).clamp_min(1e-5)
        base_dir = F.normalize(base_force, dim=-1, eps=self.cfg.eps)
        mag = torch.expm1(torch.log1p(base_mag) + 0.75 * torch.tanh(md[..., 0])).clamp_min(0.0)
        direction = F.normalize(base_dir + 0.20 * torch.tanh(md[..., 1:4]), dim=-1, eps=self.cfg.eps)
        magdir_force = mag.unsqueeze(-1) * direction

        alpha = torch.sigmoid(self.fusion_gate(gate_in))
        final_force = alpha * direct_force + (1.0 - alpha) * magdir_force
        return {
            "force": final_force,
            "phase_logits": phase_logits,
            "phase_prob": phase_prob,
            "expert_w": expert_w,
            "corr_norm": corr_norm,
            "direct_force": direct_force,
            "magdir_force": magdir_force,
            "fusion_alpha": alpha.squeeze(-1),
        }


class MergedFeelWMV7(nn.Module):
    """
    Input visual latents are the frozen/stable post-Perceiver V7 branch latents:
      [DINO 16x256, ConvNeXt 16x256, Geometry 16x256, Flow 16x256] => 64x256 / frame.

    Critical timing rule implemented literally:
      * ALL four leg FiLM branches use CURRENT rollout-step action a_t.
      * FL/FR FiLM condition Z_(t-2).
      * HL/HR FiLM condition Z_(t-3).
      * Body FiLM conditions Z_t.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        assert cfg.world_d_model % cfg.world_heads == 0
        assert cfg.tokens_per_frame == 64, "Current architecture expects 4 modalities x 16 tokens"
        assert cfg.num_groups == 9

        # V7 causal history encoders.
        self.leg_encoder = CausalHistoryEncoder(cfg.leg_hist_dim, cfg)
        self.body_encoder = CausalHistoryEncoder(cfg.body_hist_dim, cfg)
        self.leg_identity = nn.Embedding(cfg.num_legs, cfg.hist_d_model)

        # Feel-WM action encoders. The current action a_t is used for all 4 legs.
        self.leg_action_encoder = MLP(cfg.leg_film_action_dim, 128, 128, cfg.dropout)
        self.body_action_encoder = MLP(cfg.body_film_action_dim, 128, 128, cfg.dropout)
        self.leg_cond_proj = nn.Linear(cfg.hist_d_model, 128)
        self.body_cond_proj = nn.Linear(cfg.hist_d_model, 128)

        self.film_fl = FiLMGenerator(cfg)
        self.film_fr = FiLMGenerator(cfg)
        self.film_hl = FiLMGenerator(cfg)
        self.film_hr = FiLMGenerator(cfg)
        self.film_body = FiLMGenerator(cfg)

        # Stable modality identity in the 64x256 V7 latent bank.
        self.modality_embedding = nn.Parameter(torch.randn(1, cfg.num_modalities, 1, cfg.latent_dim) * 0.02)
        self.visual_to_world = nn.Linear(cfg.latent_dim, cfg.world_d_model)
        self.group_embedding = nn.Parameter(torch.randn(1, cfg.num_groups, 1, cfg.world_d_model) * 0.02)
        self.slot_embedding = nn.Parameter(torch.randn(1, 1, cfg.tokens_per_frame, cfg.world_d_model) * 0.02)
        self.time_embedding = nn.Parameter(torch.randn(1, cfg.num_visual_frames, cfg.world_d_model) * 0.02)

        self.world_blocks = nn.ModuleList([
            SelfAttentionBlock(cfg.world_d_model, cfg.world_heads, cfg.world_ffn_ratio, cfg.dropout)
            for _ in range(cfg.world_layers)
        ])
        self.world_norm = nn.LayerNorm(cfg.world_d_model)

        # Four independent t+1 visual-latent heads.
        self.dino_head = MLP(cfg.world_d_model, 512, cfg.latent_dim, cfg.dropout)
        self.conv_head = MLP(cfg.world_d_model, 512, cfg.latent_dim, cfg.dropout)
        self.geom_head = MLP(cfg.world_d_model, 512, cfg.latent_dim, cfg.dropout)
        self.flow_head = MLP(cfg.world_d_model, 512, cfg.latent_dim, cfg.dropout)

        # World group -> V7 terrain bank.
        self.world_to_query_bank = nn.Sequential(nn.LayerNorm(cfg.world_d_model), nn.Linear(cfg.world_d_model, cfg.query_dim))
        self.global_query = nn.Parameter(torch.randn(1, 1, cfg.query_dim) * 0.02)
        self.global_query_from_body = nn.Linear(cfg.hist_d_model, cfg.query_dim)
        self.leg_query_embed = nn.Embedding(cfg.num_legs, cfg.query_dim)
        self.state_to_query = nn.Sequential(nn.LayerNorm(cfg.leg_query_dim), nn.Linear(cfg.leg_query_dim, cfg.query_dim), nn.GELU(), nn.Linear(cfg.query_dim, cfg.query_dim))
        self.query_blocks = nn.ModuleList([QueryFusionBlockAligned(cfg) for _ in range(cfg.query_layers)])
        self.query_norm = nn.LayerNorm(cfg.query_dim)

        # V7 leg fusion and global context.
        self.leg_visual_fuse = nn.Sequential(
            nn.LayerNorm(128 + 256), nn.Linear(384, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.LayerNorm(256)
        )
        self.context_net = nn.Sequential(
            nn.LayerNorm(4 * 256 + 128 + 256), nn.Linear(1408, 768), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(768, 384), nn.GELU(), nn.LayerNorm(384)
        )
        self.control_proj = nn.Sequential(nn.Linear(cfg.future_action_dim, 128), nn.GELU(), nn.LayerNorm(128))
        self.horizon_proj = nn.Sequential(nn.Linear(1, 64), nn.GELU(), nn.Linear(64, 128), nn.GELU(), nn.LayerNorm(128))
        self.physics_proj = nn.Sequential(nn.Linear(cfg.physics_dim, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.LayerNorm(128))

        self.contact_head = nn.Sequential(nn.Linear(768, 256), nn.GELU(), nn.Linear(256, 4))
        self.regime_head = nn.Sequential(nn.Linear(768, 256), nn.GELU(), nn.Linear(256, 4 * cfg.num_load_regimes))
        self.state_body_head = nn.Sequential(nn.Linear(768, 384), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(384, 36))
        self.tau_force_head = nn.Sequential(
            nn.LayerNorm(cfg.force_ctx_dim), nn.Linear(cfg.force_ctx_dim, 512), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 6)
        )
        self.force_aux_head = nn.Sequential(nn.LayerNorm(cfg.force_ctx_dim), nn.Linear(cfg.force_ctx_dim, 256), nn.GELU(), nn.Linear(256, 4))

        self.v7f = V7ForceSpecialist(cfg)
        # Construct these last: with a shared seed this keeps every baseline
        # parameter initialized identically in the baseline and per-leg models.
        self.leg_pre_transformers = (
            nn.ModuleDict({name: LegPreTransformer(cfg) for name in ("fl", "fr", "hl", "hr")})
            if cfg.use_leg_pretransformers else None
        )

    def add_modality_identity(self, z: torch.Tensor) -> torch.Tensor:
        # z [...,64,256] in fixed branch order [DINO, Conv, Geom, Flow].
        shape = z.shape
        x = z.view(*shape[:-2], self.cfg.num_modalities, self.cfg.tokens_per_modality, self.cfg.latent_dim)
        emb = self.modality_embedding
        while emb.dim() < x.dim():
            emb = emb.unsqueeze(1)
        x = x + emb
        return x.reshape(*shape)

    def encode_leg_histories(self, hist: torch.Tensor) -> torch.Tensor:
        # hist [B,4,T,24] -> [B,4,128]
        B, L, T, D = hist.shape
        h = self.leg_encoder(hist.reshape(B * L, T, D)).reshape(B, L, -1)
        ids = torch.arange(L, device=hist.device)[None].expand(B, -1)
        return h + self.leg_identity(ids)

    def build_world_groups(self, z_window: torch.Tensor, h_tm1: torch.Tensor, h_tm2: torch.Tensor,
                           h_body: torch.Tensor, leg_action_t: torch.Tensor, body_action_t: torch.Tensor) -> torch.Tensor:
        # z_window [B,4,64,256] = [t-3,t-2,t-1,t]
        z_window = self.add_modality_identity(z_window)
        z3, z2, z1, z0 = [self.visual_to_world(z_window[:, i]) for i in range(4)]

        # *** LOCKED RULE: every leg uses action encoding a_t ONLY. ***
        a = self.leg_action_encoder(leg_action_t)  # [B,4,128], current rollout-step action
        a_body = self.body_action_encoder(body_action_t)

        c_fl = self.leg_cond_proj(h_tm1[:, FL]) + a[:, FL]
        c_fr = self.leg_cond_proj(h_tm1[:, FR]) + a[:, FR]
        c_hl = self.leg_cond_proj(h_tm2[:, HL]) + a[:, HL]
        c_hr = self.leg_cond_proj(h_tm2[:, HR]) + a[:, HR]
        c_body = self.body_cond_proj(h_body) + a_body

        g_hl, b_hl = self.film_hl(c_hl)
        g_hr, b_hr = self.film_hr(c_hr)
        g_fl, b_fl = self.film_fl(c_fl)
        g_fr, b_fr = self.film_fr(c_fr)
        g_b, b_b = self.film_body(c_body)

        groups = torch.stack([
            z3,                    # G0 raw Z(t-3)
            film(z3, g_hl, b_hl), # G1 HL uses a_t on terrain t-3
            film(z3, g_hr, b_hr), # G2 HR uses a_t on terrain t-3
            z2,                    # G3 raw Z(t-2)
            film(z2, g_fl, b_fl), # G4 FL uses a_t on terrain t-2
            film(z2, g_fr, b_fr), # G5 FR uses a_t on terrain t-2
            z1,                    # G6 raw Z(t-1)
            z0,                    # G7 raw Z(t)
            film(z0, g_b, b_b),   # G8 Body uses current action on current terrain
        ], dim=1)                  # [B,9,64,800]

        time_ids = torch.tensor([0, 0, 0, 1, 1, 1, 2, 3, 3], device=z_window.device)
        groups = groups + self.group_embedding + self.slot_embedding
        groups = groups + self.time_embedding[:, time_ids].unsqueeze(2)
        return groups

    def world_forward(self, groups: torch.Tensor) -> torch.Tensor:
        B = groups.shape[0]
        groups = self.apply_leg_pretransformers(groups)
        x = groups.reshape(B, self.cfg.world_tokens, self.cfg.world_d_model)
        mask = build_physical_time_mask(self.cfg.tokens_per_frame, x.device)
        for block in self.world_blocks:
            x = block(x, mask)
        return self.world_norm(x).reshape(B, self.cfg.num_groups, self.cfg.tokens_per_frame, self.cfg.world_d_model)

    def apply_leg_pretransformers(self, groups: torch.Tensor) -> torch.Tensor:
        """Process FiLM groups FL/FR/HL/HR immediately before shared attention."""
        blocks = self.leg_pre_transformers
        if blocks is None:
            return groups
        # G4=FL, G5=FR, G1=HL, G2=HR. Clone before indexed writes so the
        # untouched raw/body groups remain bit-identical and autograd is safe.
        groups = groups.clone()
        for name, group_index in (("fl", 4), ("fr", 5), ("hl", 1), ("hr", 2)):
            groups[:, group_index] = blocks[name](groups[:, group_index])
        return groups

    def predict_next_visual(self, H: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Use body-conditioned current group G8; split its 64 slots by modality.
        h = H[:, 8]
        n = self.cfg.tokens_per_modality
        dino = self.dino_head(h[:, 0*n:1*n])
        conv = self.conv_head(h[:, 1*n:2*n])
        geom = self.geom_head(h[:, 2*n:3*n])
        flow = self.flow_head(h[:, 3*n:4*n])
        return {"dino": dino, "conv": conv, "geom": geom, "flow": flow,
                "z_next": torch.cat([dino, conv, geom, flow], dim=1)}

    def aligned_v7_queries(self, H: torch.Tensor, h_body: torch.Tensor, leg_query_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Query order: global, FL, FR, HL, HR.
        B = H.shape[0]
        global_q = self.global_query.expand(B, -1, -1) + self.global_query_from_body(h_body)[:, None]
        ids = torch.arange(4, device=H.device)[None].expand(B, -1)
        foot_q = self.leg_query_embed(ids) + self.state_to_query(leg_query_features)
        q = torch.cat([global_q, foot_q], dim=1)

        # Temporally aligned visual banks from the world transformer.
        # G8 current/body, G4 FL@t-2, G5 FR@t-2, G1 HL@t-3, G2 HR@t-3.
        bank_indices = [8, 4, 5, 1, 2]
        banks = torch.stack([self.world_to_query_bank(H[:, i]) for i in bank_indices], dim=1)  # [B,5,64,256]
        for block in self.query_blocks:
            q = block(q, banks)
        q = self.query_norm(q)
        return q[:, 0], q[:, 1:]

    def forward_step(self, z_window: torch.Tensor, leg_hist_t: torch.Tensor, leg_hist_tm1: torch.Tensor,
                     leg_hist_tm2: torch.Tensor, body_hist_t: torch.Tensor, body_action_t: torch.Tensor,
                     leg_action_t: torch.Tensor, leg_query_features_t: torch.Tensor, future_action: torch.Tensor,
                     physics: torch.Tensor, horizon: torch.Tensor, jointbody_baseline: torch.Tensor,
                     jointbody_std: torch.Tensor, tau_baseline: torch.Tensor, tau_std: torch.Tensor,
                     force_baseline: torch.Tensor, force_std: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = z_window.shape[0]
        h_t = self.encode_leg_histories(leg_hist_t)
        h_tm1 = self.encode_leg_histories(leg_hist_tm1)
        h_tm2 = self.encode_leg_histories(leg_hist_tm2)
        h_body = self.body_encoder(body_hist_t)

        groups = self.build_world_groups(z_window, h_tm1, h_tm2, h_body, leg_action_t, body_action_t)
        H = self.world_forward(groups)
        vis = self.predict_next_visual(H)

        v_global, v_legs = self.aligned_v7_queries(H, h_body, leg_query_features_t)
        legf = self.leg_visual_fuse(torch.cat([h_t, v_legs], dim=-1))
        ctx = self.context_net(torch.cat([legf.reshape(B, -1), h_body, v_global], dim=-1))
        dyn = torch.cat([ctx, self.control_proj(future_action), self.horizon_proj(horizon), self.physics_proj(physics)], dim=-1)

        contact_logits = self.contact_head(dyn)
        contact_prob = torch.sigmoid(contact_logits)
        regime_logits = self.regime_head(dyn).reshape(B, 4, self.cfg.num_load_regimes)
        regime_prob = regime_logits.softmax(-1)

        # V7 state/body residual prediction -> physical target space.
        state_residual = self.state_body_head(dyn)  # [B,36]
        jointbody_pred = jointbody_baseline + state_residual * jointbody_std

        # Per-leg 1285-D V7 force context.
        zh = dyn[:, None].expand(-1, 4, -1)
        force_ctx = torch.cat([zh, legf, v_legs, contact_prob.unsqueeze(-1), regime_prob], dim=-1)
        tf_residual = self.tau_force_head(force_ctx)  # [B,4,6]
        tau_pred = tau_baseline + tf_residual[..., :3] * tau_std[:, None, :]
        base_force = force_baseline + tf_residual[..., 3:] * force_std[:, None, :]
        aux = self.force_aux_head(force_ctx)

        return {
            "H": H,
            **vis,
            "v_global": v_global,
            "v_legs": v_legs,
            "legf": legf,
            "global_context": ctx,
            "dynamics": dyn,
            "contact_logits": contact_logits,
            "contact_prob": contact_prob,
            "regime_logits": regime_logits,
            "regime_prob": regime_prob,
            "state_residual": state_residual,
            "jointbody_pred": jointbody_pred,
            "force_ctx": force_ctx,
            "tf_residual": tf_residual,
            "tau_pred": tau_pred,
            "base_force": base_force,
            "force_aux_logmag": aux[..., 0],
            "force_aux_dir": F.normalize(aux[..., 1:4], dim=-1, eps=self.cfg.eps),
        }

    @staticmethod
    def _append_repeated(hist: torch.Tensor, next_step: torch.Tensor, shift: int) -> torch.Tensor:
        # hist [B,4,T,D] / [B,T,D], next_step [B,4,D] / [B,D]
        shift = max(1, min(int(shift), hist.shape[-2]))
        if hist.dim() == 4:
            rep = next_step.unsqueeze(2).expand(-1, -1, shift, -1)
            return torch.cat([hist[:, :, shift:], rep], dim=2)
        rep = next_step.unsqueeze(1).expand(-1, shift, -1)
        return torch.cat([hist[:, shift:], rep], dim=1)

    @staticmethod
    def _make_leg_history_step(jointbody: torch.Tensor, tau: torch.Tensor, force: torch.Tensor,
                               future_action: torch.Tensor, lowlevel: torch.Tensor, err: torch.Tensor) -> torch.Tensor:
        # jointbody [B,36]; first 24 = 4*(q3,dq3), final 12 = body.
        B = jointbody.shape[0]
        qdq = jointbody[:, :24].reshape(B, 4, 6)
        prop12 = torch.cat([qdq[..., :3], qdq[..., 3:], tau, force], dim=-1)
        u6 = future_action[:, :6][:, None].expand(-1, 4, -1)
        return torch.cat([prop12, u6, lowlevel, err], dim=-1)

    @staticmethod
    def _make_body_history_step(jointbody: torch.Tensor, future_action: torch.Tensor) -> torch.Tensor:
        return torch.cat([jointbody[:, 24:], future_action[:, :6]], dim=-1)

    @staticmethod
    def _batch_stat(batch: Dict[str, torch.Tensor], name: str, width: int, like: torch.Tensor,
                    default: float) -> torch.Tensor:
        value = batch.get(name)
        if value is None:
            return like.new_full((like.shape[0], width), default)
        value = value.to(device=like.device, dtype=like.dtype)
        if value.dim() == 1:
            value = value.unsqueeze(0).expand(like.shape[0], -1)
        return value

    def _recurrent_query(self, jointbody: torch.Tensor, tau: torch.Tensor, force: torch.Tensor,
                         leg_query_exogenous: torch.Tensor, foot_position: torch.Tensor,
                         batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        B = jointbody.shape[0]
        qdq = jointbody[:, :24].reshape(B, 4, 6)
        proprio = torch.cat([qdq, tau, force], dim=-1)
        mean = self._batch_stat(batch, "proprio_mean", 12, jointbody, 0.0)[:, None]
        std = self._batch_stat(batch, "proprio_std", 12, jointbody, 1.0)[:, None].clamp_min(1e-6)
        normalized = (proprio - mean) / std
        # The 12:18 controller features and final planned displacement are
        # exogenous. Foot position is updated from predicted q and a cached
        # local FK Jacobian, never copied from a future measured state.
        return torch.cat([
            normalized,
            leg_query_exogenous[..., 12:18],
            foot_position,
            leg_query_exogenous[..., 21:24],
        ], dim=-1)

    def _recurrent_physics(self, jointbody: torch.Tensor, tau: torch.Tensor, force: torch.Tensor,
                           contact_prob: torch.Tensor, regime_prob: torch.Tensor,
                           batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        B = jointbody.shape[0]
        qdq = jointbody[:, :24].reshape(B, 4, 6)
        p_mean = self._batch_stat(batch, "proprio_mean", 12, jointbody, 0.0)
        p_std = self._batch_stat(batch, "proprio_std", 12, jointbody, 1.0).clamp_min(1e-6)
        b_mean = self._batch_stat(batch, "body_mean", 12, jointbody, 0.0)
        b_std = self._batch_stat(batch, "body_std", 12, jointbody, 1.0).clamp_min(1e-6)
        baseline_q = qdq[..., :3] + 0.1 * qdq[..., 3:]
        q_norm = (baseline_q - p_mean[:, None, :3]) / p_std[:, None, :3]
        tau_norm = (tau - p_mean[:, None, 6:9]) / p_std[:, None, 6:9]
        body_norm = (jointbody[:, 24:27] - b_mean[:, :3]) / b_std[:, :3]
        thresholds = batch.get("load_thresholds")
        if thresholds is None:
            threshold = jointbody.new_ones((B, 1))
        else:
            thresholds = thresholds.to(device=jointbody.device, dtype=jointbody.dtype)
            if thresholds.dim() == 1:
                thresholds = thresholds.unsqueeze(0).expand(B, -1)
            threshold = thresholds[:, -1:].clamp_min(1.0)
        force_magnitude = torch.linalg.norm(force, dim=-1) / threshold
        # The existing 16 load channels carry both predicted contact and load:
        # each leg's four entries sum to its contact probability.
        regime_contact = regime_prob * contact_prob[..., None]
        return torch.cat([
            q_norm.reshape(B, -1), tau_norm.reshape(B, -1), body_norm,
            force_magnitude, regime_contact.reshape(B, -1),
        ], dim=-1)

    def forward(self, batch: Dict[str, torch.Tensor], history_shift: int = 1,
                teacher_force_mask: torch.Tensor | None = None):
        return self.rollout(batch, history_shift=history_shift, teacher_force_mask=teacher_force_mask)

    def rollout(self, batch: Dict[str, torch.Tensor], history_shift: int = 1,
                teacher_force_mask: torch.Tensor | None = None) -> Dict[str, torch.Tensor | List[Dict[str, torch.Tensor]]]:
        """Autoregressive K-step rollout. Every FiLM branch uses current action a_t at that rollout step."""
        z_window = batch["z_window"]
        leg_t = batch["leg_hist_t"]
        leg_tm1 = batch["leg_hist_tm1"]
        leg_tm2 = batch["leg_hist_tm2"]
        body_t = batch["body_hist_t"]
        K = batch["future_action_seq"].shape[1]
        B = z_window.shape[0]
        if teacher_force_mask is None:
            teacher_force_mask = torch.zeros((B, K), dtype=torch.bool, device=z_window.device)
        else:
            teacher_force_mask = teacher_force_mask.to(device=z_window.device, dtype=torch.bool)
            if tuple(teacher_force_mask.shape) != (B, K):
                raise ValueError(f"teacher_force_mask must be [{B},{K}]")

        steps: List[Dict[str, torch.Tensor]] = []
        force_ctx_seq, base_force_seq, contact_seq, regime_seq = [], [], [], []
        refined_force_seq: List[torch.Tensor] = []
        feedback_force_seq: List[torch.Tensor] = []
        jointbody_state = tau_state = force_state = regime_state = contact_state = None
        foot_position = batch["leg_query_features_seq"][:, 0, :, 18:21]
        previous_q = leg_t[:, :, -1, :3]
        foot_jacobian = batch.get("foot_jacobian")
        if foot_jacobian is not None:
            foot_jacobian = foot_jacobian.to(device=z_window.device, dtype=z_window.dtype)

        for k in range(K):
            if k == 0:
                query_t = batch["leg_query_features_seq"][:, k]
                physics_t = batch["physics_seq"][:, k]
                jointbody_baseline_t = batch["jointbody_baseline_seq"][:, k]
                tau_baseline_t = batch["tau_baseline_seq"][:, k]
                force_baseline_t = batch["force_baseline_seq"][:, k]
            else:
                query_exogenous = batch["leg_query_features_seq"][:, k]
                query_t = self._recurrent_query(
                    jointbody_state, tau_state, force_state, query_exogenous,
                    foot_position, batch,
                )
                physics_t = self._recurrent_physics(
                    jointbody_state, tau_state, force_state, contact_state, regime_state, batch,
                )
                jointbody_baseline_t = jointbody_state
                tau_baseline_t = tau_state
                force_baseline_t = force_state
            out = self.forward_step(
                z_window=z_window,
                leg_hist_t=leg_t, leg_hist_tm1=leg_tm1, leg_hist_tm2=leg_tm2,
                body_hist_t=body_t,
                body_action_t=batch["body_action_seq"][:, k],
                leg_action_t=batch["leg_action_seq"][:, k],   # CURRENT a_t ONLY
                leg_query_features_t=query_t,
                future_action=batch["future_action_seq"][:, k],
                physics=physics_t,
                horizon=batch["horizon_seq"][:, k],
                jointbody_baseline=jointbody_baseline_t,
                jointbody_std=batch["jointbody_std"],
                tau_baseline=tau_baseline_t,
                tau_std=batch["tau_std"],
                force_baseline=force_baseline_t,
                force_std=batch["force_std"],
            )
            steps.append(out)
            force_ctx_seq.append(out["force_ctx"])
            base_force_seq.append(out["base_force"])
            contact_seq.append(out["contact_prob"])
            regime_seq.append(out["regime_prob"])

            # V7-F is evaluated causally on this available prefix. Its refined
            # force is the force fed into the next autoregressive step.
            v7f_prefix = self.v7f(
                torch.stack(force_ctx_seq, dim=1),
                torch.stack(base_force_seq, dim=1), batch["force_std"],
                torch.stack(contact_seq, dim=1), torch.stack(regime_seq, dim=1),
            )
            refined_force = v7f_prefix["force"][:, -1]
            refined_force_seq.append(refined_force)

            # Recurrent state update.
            next_jointbody = out["jointbody_pred"]
            next_tau = out["tau_pred"]
            next_force = refined_force
            next_contact = out["contact_prob"]
            next_regime = out["regime_prob"]
            next_z = out["z_next"]
            if k + 1 < K:
                force_teacher = teacher_force_mask[:, k + 1, None, None]
                state_teacher = teacher_force_mask[:, k + 1, None]
                next_jointbody = torch.where(state_teacher, batch["target_jointbody_seq"][:, k], next_jointbody)
                next_tau = torch.where(force_teacher, batch["target_tau_seq"][:, k], next_tau)
                next_force = torch.where(force_teacher, batch["target_force_seq"][:, k], next_force)
                target_contact = batch["target_contact_seq"][:, k]
                next_contact = torch.where(state_teacher, target_contact, next_contact)
                target_regime = F.one_hot(
                    batch["target_load_class_seq"][:, k].long(), self.cfg.num_load_regimes
                ).to(next_regime.dtype)
                next_regime = torch.where(state_teacher[..., None], target_regime, next_regime)
                next_z = torch.where(
                    teacher_force_mask[:, k + 1, None, None], batch["target_z_seq"][:, k], next_z
                )

            jointbody_state, tau_state, force_state = next_jointbody, next_tau, next_force
            feedback_force_seq.append(next_force)
            regime_state, contact_state = next_regime, next_contact
            next_q = next_jointbody[:, :24].reshape(B, 4, 6)[..., :3]
            if foot_jacobian is not None:
                foot_position = foot_position + torch.einsum(
                    "blij,blj->bli", foot_jacobian, next_q - previous_q
                )
            previous_q = next_q
            next_leg = self._make_leg_history_step(
                next_jointbody, next_tau, next_force,
                batch["future_action_seq"][:, k], batch["lowlevel_cmd_seq"][:, k], batch["err_seq"][:, 0]
            )
            next_body = self._make_body_history_step(next_jointbody, batch["future_action_seq"][:, k])
            leg_tm2, leg_tm1 = leg_tm1, leg_t
            leg_t = self._append_repeated(leg_t, next_leg, history_shift)
            body_t = self._append_repeated(body_t, next_body, history_shift)
            z_window = torch.cat([z_window[:, 1:], next_z.unsqueeze(1)], dim=1)

        force_ctx_seq = torch.stack(force_ctx_seq, dim=1)
        base_force_seq = torch.stack(base_force_seq, dim=1)
        contact_seq = torch.stack(contact_seq, dim=1)
        regime_seq = torch.stack(regime_seq, dim=1)
        v7f = v7f_prefix
        return {
            "steps": steps,
            "force_ctx_seq": force_ctx_seq,
            "base_force_seq": base_force_seq,
            "contact_seq": contact_seq,
            "regime_seq": regime_seq,
            "v7f": v7f,
            "refined_force_seq": torch.stack(refined_force_seq, dim=1),
            "feedback_force_seq": torch.stack(feedback_force_seq, dim=1),
        }


# CELL 38
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    "uniform": [0.20, 0.20, 0.20, 0.20, 0.20],
    "mild_decay": [0.30, 0.25, 0.20, 0.15, 0.10],
    "strong_decay": [0.516, 0.258, 0.129, 0.065, 0.032],
}


def write_selected_configs(weights: list[float], root: Path) -> None:
    for name in ("tartanground_baseline_k5.yaml", "tartanground_leg_k5.yaml"):
        path = root / "configs" / name
        config = yaml.safe_load(path.read_text())
        config["train"]["horizon_lambdas"] = weights
        tmp = path.with_suffix(".tmp")
        tmp.write_text(yaml.safe_dump(config, sort_keys=False))
        os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/tartanground_baseline_k5.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--set-configs", action="store_true")
    args = parser.parse_args()
    base = yaml.safe_load(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    for profile, weights in PROFILES.items():
        out_dir = args.output / profile
        metrics_path = out_dir / "metrics.jsonl"
        checkpoint_path = out_dir / "checkpoints" / "latest.pt"
        score = None
        if metrics_path.exists() and checkpoint_path.exists():
            for line in metrics_path.read_text().splitlines():
                row = json.loads(line)
                if row.get("type") == "val" and int(row.get("step", -1)) >= 20:
                    score = float(row["physical_score"])
        if score is None:
            config = json.loads(json.dumps(base))
            config["train"].update({
                "output_dir": str(out_dir),
                "max_steps": 20,
                "val_every_steps": 20,
                "checkpoint_every_steps": 20,
                "keep_last_checkpoints": 1,
                "resume": "none",
                "teacher_forcing": True,
                "horizon_lambdas": weights,
            })
            config_path = out_dir / "calibration.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = "0"
            env["OMP_NUM_THREADS"] = "4"
            env["MKL_NUM_THREADS"] = "4"
            subprocess.run(
                [sys.executable, "-u", str(ROOT / "train.py"), "--config", str(config_path)],
                cwd=ROOT, env=env, check=True,
            )
            rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
            validations = [row for row in rows if row.get("type") == "val" and int(row.get("step", -1)) >= 20]
            if not validations:
                raise RuntimeError(f"No step-20 development validation recorded for {profile}")
            score = float(validations[-1]["physical_score"])
        results[profile] = {"weights": weights, "development_physical_score": score}
        print(json.dumps({"lambda_candidate": profile, **results[profile]}), flush=True)
    selected = min(results, key=lambda key: results[key]["development_physical_score"])
    report = {"selected": selected, "weights": results[selected]["weights"], "candidates": results}
    report_path = args.output / "lambda_calibration.json"
    report_path.write_text(json.dumps(report, indent=2))
    if args.set_configs:
        write_selected_configs(results[selected]["weights"], ROOT)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()


# CELL 40
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from model import MergedFeelWMV7
from data import move_batch_to_device
from tartanground.evaluate import physical_metrics, validate_strict_split
from tartanground.indexed import IndexedTartanGroundDataset


def plain_metrics(pred: dict[str, np.ndarray], target: dict[str, np.ndarray]) -> dict[str, float | None]:
    result = physical_metrics(pred, target)
    load = np.asarray(pred["load"]).argmax(-1) == np.asarray(target["load"])
    result["load_accuracy"] = float(load.mean())
    return {key: (value if np.isfinite(value) else None) for key, value in result.items()}


def collect(model, dataset, device, amp_dtype: str):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    case_rows = []
    fields = {
        key: [] for key in (
            "q_pred", "q_true", "tau_pred", "tau_true", "force_pred", "force_true",
            "contact_pred", "contact_true", "load_pred", "load_true",
        )
    }
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        enabled = device.type == "cuda" and amp_dtype in ("bf16", "fp16")
        dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
            output = model(batch, history_shift=10)
        pred_q = torch.stack([step["jointbody_pred"][:, :24].reshape(1, 4, 6)[..., :3] for step in output["steps"]], dim=1)
        true_q = batch["target_jointbody_seq"][:, :, :24].reshape(1, len(output["steps"]), 4, 6)[..., :3]
        values = {
            "q_pred": pred_q,
            "q_true": true_q,
            "tau_pred": torch.stack([step["tau_pred"] for step in output["steps"]], dim=1),
            "tau_true": batch["target_tau_seq"],
            "force_pred": output["v7f"]["force"],
            "force_true": batch["target_force_seq"],
            "contact_pred": torch.stack([step["contact_prob"] for step in output["steps"]], dim=1),
            "contact_true": batch["target_contact_seq"],
            "load_pred": torch.stack([step["regime_prob"] for step in output["steps"]], dim=1),
            "load_true": batch["target_load_class_seq"],
        }
        for key, value in values.items():
            fields[key].append(value.detach().float().cpu().numpy()[0])
        # Dataset order is trajectory-balanced only during training; manifests
        # retain per-sample identity for strict-test grouping.
        row = dataset.rows[len(case_rows)]
        camera_index = int(row.get("camera_index", len(case_rows)))
        case_rows.append({
            "case_id": row.get("test_case_id", f"{row['trajectory_id']}@camera_{camera_index:04d}"),
            "trajectory_id": row["trajectory_id"],
            "camera_index": camera_index,
            "current_indices": row.get("current_indices", []),
            "target_visual_indices": row.get("target_visual_indices", []),
        })
    result = {key: np.stack(values, axis=0) for key, values in fields.items()}
    return case_rows, result


def evaluate(prediction: dict[str, np.ndarray], trajectory_ids: list[str]) -> dict:
    target = {name: prediction[f"{name}_true"] for name in ("q", "tau", "force", "contact", "load")}
    pred = {name: prediction[f"{name}_pred"] for name in ("q", "tau", "force", "contact", "load")}
    metrics = {"global": plain_metrics(pred, target), "by_trajectory": {}, "by_horizon": {}, "by_leg": {}}
    ids = np.asarray(trajectory_ids)
    for trajectory in sorted(set(trajectory_ids)):
        mask = ids == trajectory
        metrics["by_trajectory"][trajectory] = plain_metrics(
            {key: value[mask] for key, value in pred.items()},
            {key: value[mask] for key, value in target.items()},
        )
    horizons = pred["q"].shape[1]
    for k in range(horizons):
        metrics["by_horizon"][f"+{(k + 1) / 10:.1f}s"] = plain_metrics(
            {key: value[:, k : k + 1] for key, value in pred.items()},
            {key: value[:, k : k + 1] for key, value in target.items()},
        )
    for leg, name in enumerate(("FL", "FR", "HL", "HR")):
        metrics["by_leg"][name] = plain_metrics(
            {key: value[:, :, leg : leg + 1] for key, value in pred.items()},
            {key: value[:, :, leg : leg + 1] for key, value in target.items()},
        )
    return metrics


def write_test_prediction_csv(path: Path, case_rows: list[dict], prediction: dict[str, np.ndarray]) -> None:
    axes = ("x", "y", "z")
    fields = ["case_id", "trajectory_id", "camera_index", "horizon_step", "horizon_seconds", "leg"]
    fields += [f"q_joint{i}_{kind}_rad" for i in range(3) for kind in ("pred", "true")]
    fields += [f"tau_{axis}_{kind}_nm" for axis in axes for kind in ("pred", "true")]
    fields += [f"force_{axis}_{kind}_n" for axis in axes for kind in ("pred", "true")]
    fields += ["force_magnitude_pred_n", "force_magnitude_true_n", "contact_probability", "contact_pred", "contact_true"]
    fields += ["load_pred_class", "load_true_class"] + [f"load_probability_{i}" for i in range(4)]
    leg_names = ("FL", "FR", "HL", "HR")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        horizons = prediction["q_pred"].shape[1]
        for sample_i, case in enumerate(case_rows):
            for horizon_i in range(horizons):
                for leg_i, leg_name in enumerate(leg_names):
                    q_pred = prediction["q_pred"][sample_i, horizon_i, leg_i]
                    q_true = prediction["q_true"][sample_i, horizon_i, leg_i]
                    tau_pred = prediction["tau_pred"][sample_i, horizon_i, leg_i]
                    tau_true = prediction["tau_true"][sample_i, horizon_i, leg_i]
                    force_pred = prediction["force_pred"][sample_i, horizon_i, leg_i]
                    force_true = prediction["force_true"][sample_i, horizon_i, leg_i]
                    load_prob = prediction["load_pred"][sample_i, horizon_i, leg_i]
                    contact_prob = float(prediction["contact_pred"][sample_i, horizon_i, leg_i])
                    row = {
                        "case_id": case["case_id"], "trajectory_id": case["trajectory_id"],
                        "camera_index": case["camera_index"], "horizon_step": horizon_i + 1,
                        "horizon_seconds": (horizon_i + 1) / 10.0, "leg": leg_name,
                        "force_magnitude_pred_n": float(np.linalg.norm(force_pred)),
                        "force_magnitude_true_n": float(np.linalg.norm(force_true)),
                        "contact_probability": contact_prob, "contact_pred": int(contact_prob >= 0.5),
                        "contact_true": int(prediction["contact_true"][sample_i, horizon_i, leg_i] >= 0.5),
                        "load_pred_class": int(np.argmax(load_prob)),
                        "load_true_class": int(prediction["load_true"][sample_i, horizon_i, leg_i]),
                    }
                    for channel in range(3):
                        row[f"q_joint{channel}_pred_rad"] = float(q_pred[channel])
                        row[f"q_joint{channel}_true_rad"] = float(q_true[channel])
                        row[f"tau_{axes[channel]}_pred_nm"] = float(tau_pred[channel])
                        row[f"tau_{axes[channel]}_true_nm"] = float(tau_true[channel])
                        row[f"force_{axes[channel]}_pred_n"] = float(force_pred[channel])
                        row[f"force_{axes[channel]}_true_n"] = float(force_true[channel])
                    for class_i, probability in enumerate(load_prob):
                        row[f"load_probability_{class_i}"] = float(probability)
                    writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cases-per-trajectory", type=int, default=None,
                        help="Require this many selected strict-test cases for each held-out trajectory")
    args = parser.parse_args()
    validate_strict_split(args.train_manifest, args.dev_manifest, args.test_manifest)
    mcfg, tcfg, _ = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = MergedFeelWMV7(mcfg).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    dataset = IndexedTartanGroundDataset(args.test_manifest)
    counts = Counter(row["trajectory_id"] for row in dataset.rows)
    if args.cases_per_trajectory is not None and any(value != args.cases_per_trajectory for value in counts.values()):
        raise ValueError(f"Expected {args.cases_per_trajectory} cases per held-out trajectory; found {dict(counts)}")
    case_rows, prediction = collect(model, dataset, device, tcfg.amp_dtype)
    trajectory_ids = [row["trajectory_id"] for row in case_rows]
    metrics = evaluate(prediction, trajectory_ids)
    metrics["sample_count"] = len(case_rows)
    metrics["sample_count_by_trajectory"] = dict(sorted(Counter(trajectory_ids).items()))
    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "strict_test_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=False))
    selection = {
        "sample_cases_per_trajectory": args.cases_per_trajectory,
        "total_cases": len(case_rows),
        "cases_by_trajectory": dict(sorted(Counter(trajectory_ids).items())),
        "cases": case_rows,
    }
    (args.output / "strict_test_selection.json").write_text(json.dumps(selection, indent=2, allow_nan=False) + "\n")
    pred_path = args.output / "strict_test_predictions.npz"
    np.savez_compressed(
        pred_path, case_id=np.asarray([row["case_id"] for row in case_rows]),
        trajectory_id=np.asarray(trajectory_ids),
        camera_index=np.asarray([row["camera_index"] for row in case_rows], dtype=np.int64),
        current_indices=np.asarray([row["current_indices"] for row in case_rows], dtype=np.int64),
        target_visual_indices=np.asarray([row["target_visual_indices"] for row in case_rows], dtype=np.int64),
        **{f"{key}_{kind}": value for key in ("q", "tau", "force", "contact", "load")
           for kind, value in (("pred", prediction[f"{key}_pred"]), ("true", prediction[f"{key}_true"]))},
    )
    csv_path = args.output / "strict_test_predictions.csv"
    write_test_prediction_csv(csv_path, case_rows, prediction)
    print(json.dumps({"samples": len(trajectory_ids), "cases_per_trajectory": dict(counts),
                      "metrics": metrics, "predictions": str(pred_path), "prediction_csv": str(csv_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()


# CELL 42
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import validate_sample
from tartanground.pilot import (
    build_body_action,
    build_leg_action,
    load_v7_resamplers,
    phase_vector,
)


DEFAULT_SHARED = Path(
    "/path/to/ugp/V7F_4080_Run/storage"
)
DEFAULT_OUTPUT = Path(
    "/path/to/ugp/merged_feelwm_v7_runs/"
    "tartanground_pilot/prepared_k1"
)


def causal_hold(t_src: np.ndarray, x_src: np.ndarray, t_target: np.ndarray) -> np.ndarray:
    index = np.searchsorted(t_src, t_target, side="right") - 1
    if np.any(index < 0):
        raise RuntimeError("Target precedes the first commanded-position sample")
    return x_src[index]


def stamp_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def extract_commanded_positions(
    bag_path: Path,
    grid_original: np.ndarray,
    actuator_for_joint: np.ndarray,
    cache_path: Path,
) -> np.ndarray:
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        if cached.shape == (len(grid_original), 4, 3) and np.isfinite(cached).all():
            return cached.astype(np.float32)

    from rosbags.highlevel import AnyReader

    physical = {
        "/state_estimator/joint_states",
        "/state_estimator/contact_force_lf_foot",
        "/state_estimator/contact_force_rf_foot",
        "/state_estimator/contact_force_lh_foot",
        "/state_estimator/contact_force_rh_foot",
        "/state_estimator/twist",
    }
    with AnyReader([bag_path]) as reader:
        selected = [c for c in reader.connections if c.topic in physical]
        if not selected:
            raise RuntimeError("No physical reference topic found in ROS bag")
        _, first_ns, _ = next(reader.messages(connections=selected))
    t0 = first_ns * 1e-9

    times: list[float] = []
    positions: list[list[float]] = []
    topic = "/state_estimator/actuator_readings"
    with AnyReader([bag_path]) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            raise RuntimeError(f"Required topic missing: {topic}")
        for connection, _, raw in reader.messages(connections=connections):
            message = reader.deserialize(raw, connection.msgtype)
            readings = list(message.readings)
            if len(readings) != 12:
                continue
            stamps = [stamp_seconds(r.commanded.header.stamp) for r in readings]
            values = [float(r.commanded.position) for r in readings]
            times.append(float(np.median(stamps)) - t0)
            positions.append(values)
    if not times:
        raise RuntimeError("No valid commanded-position records found")
    cmd_t = np.asarray(times, np.float64)
    cmd_q = np.asarray(positions, np.float32)
    order = np.argsort(cmd_t)
    cmd_t, cmd_q = cmd_t[order], cmd_q[order]
    aligned = causal_hold(cmd_t, cmd_q, grid_original)
    aligned = aligned[:, np.asarray(actuator_for_joint, np.int64)].reshape(-1, 4, 3)
    if not np.isfinite(aligned).all():
        raise RuntimeError("Non-finite commanded joint positions")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp.npy")
    np.save(temporary, aligned.astype(np.float32), allow_pickle=False)
    os.replace(temporary, cache_path)
    return aligned.astype(np.float32)


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in rpy]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], np.float64)
    return rz @ ry @ rx


def axis_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = np.asarray(axis, np.float64) / (np.linalg.norm(axis) + 1e-12)
    c, s, one_minus = np.cos(angle), np.sin(angle), 1.0 - np.cos(angle)
    return np.array(
        [
            [c + x * x * one_minus, x * y * one_minus - z * s, x * z * one_minus + y * s],
            [y * x * one_minus + z * s, c + y * y * one_minus, y * z * one_minus - x * s],
            [z * x * one_minus - y * s, z * y * one_minus + x * s, c + z * z * one_minus],
        ],
        np.float64,
    )


class StrictURDFFK:
    def __init__(self, urdf_path: Path):
        root = ET.parse(urdf_path).getroot()
        links = {node.attrib["name"] for node in root.findall("link")}
        children: dict[str, dict] = {}
        all_children = set()
        for element in root.findall("joint"):
            origin = element.find("origin")
            axis_element = element.find("axis")
            joint = {
                "name": element.attrib["name"],
                "type": element.attrib.get("type", "fixed"),
                "parent": element.find("parent").attrib["link"],
                "child": element.find("child").attrib["link"],
                "xyz": np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3),
                "rpy": np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3),
                "axis": np.fromstring(axis_element.attrib.get("xyz", "1 0 0"), sep=" ") if axis_element is not None else np.array([1.0, 0.0, 0.0]),
            }
            children[joint["child"]] = joint
            all_children.add(joint["child"])
        roots = list(links - all_children)
        if len(roots) != 1:
            raise RuntimeError(f"URDF must have one root link, got {roots}")
        self.root = roots[0]
        self.prefixes = ("LF", "RF", "LH", "RH")
        self.chains = []
        for prefix in self.prefixes:
            candidates = [link for link in links if link.upper().startswith(prefix + "_") and "FOOT" in link.upper()]
            valid = []
            for link in candidates:
                chain = []
                current = link
                while current != self.root and current in children:
                    chain.append(children[current])
                    current = children[current]["parent"]
                chain.reverse()
                names = {joint["name"] for joint in chain}
                if all(f"{prefix}_{name}" in names for name in ("HAA", "HFE", "KFE")):
                    valid.append(chain)
            if not valid:
                raise RuntimeError(f"No verified {prefix} foot chain in {urdf_path}")
            self.chains.append(max(valid, key=len))

    @staticmethod
    def transform(rotation=None, translation=None) -> np.ndarray:
        value = np.eye(4, dtype=np.float64)
        if rotation is not None:
            value[:3, :3] = rotation
        if translation is not None:
            value[:3, 3] = translation
        return value

    def foot_positions(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, np.float64)
        qmap = {
            f"{prefix}_{joint}": float(q[leg, index])
            for leg, prefix in enumerate(self.prefixes)
            for index, joint in enumerate(("HAA", "HFE", "KFE"))
        }
        result = []
        for chain in self.chains:
            transform = np.eye(4, dtype=np.float64)
            for joint in chain:
                transform = transform @ self.transform(rpy_matrix(joint["rpy"]), joint["xyz"])
                if joint["type"] in ("revolute", "continuous"):
                    transform = transform @ self.transform(axis_matrix(joint["axis"], qmap.get(joint["name"], 0.0)))
                elif joint["type"] == "prismatic":
                    transform = transform @ self.transform(translation=joint["axis"] * qmap.get(joint["name"], 0.0))
                elif joint["type"] != "fixed":
                    raise RuntimeError(f"Unsupported URDF joint type: {joint['type']}")
            result.append(transform[:3, 3])
        result = np.asarray(result, np.float32)
        if not np.isfinite(result).all() or np.max(np.linalg.norm(result, axis=1)) > 2.0:
            raise RuntimeError("Implausible ANYmal foot positions")
        return result


def atomic_torch_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def export_latents(cache_root: Path, checkpoint: Path, output: Path, max_frames: int) -> tuple[torch.Tensor, str]:
    latent_path = output / "post_perceiver_latents.pt"
    if latent_path.exists():
        cached = torch.load(latent_path, map_location="cpu", weights_only=True)
        if cached["latents"].shape == (max_frames, 64, 256):
            return cached["latents"], str(cached["checkpoint_sha256"])

    arrays = {
        "dino": np.load(cache_root / "ForestEnv_P2000_v7_dinov3.npy", mmap_mode="r", allow_pickle=False),
        "conv": np.load(cache_root / "ForestEnv_P2000_v7_convnext.npy", mmap_mode="r", allow_pickle=False),
        "geom": np.load(cache_root / "ForestEnv_P2000_v7_utonia.npy", mmap_mode="r", allow_pickle=False),
        "flow": np.load(cache_root / "ForestEnv_P2000_v7_waft.npy", mmap_mode="r", allow_pickle=False),
    }
    for name, array in arrays.items():
        if len(array) < max_frames or not np.isfinite(np.asarray(array[[0, max_frames - 1]], np.float32)).all():
            raise RuntimeError(f"Invalid {name} cache")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, digest = load_v7_resamplers(checkpoint, device)
    result = torch.empty((max_frames, 64, 256), dtype=torch.float16)
    for start in range(0, max_frames, 8):
        stop = min(max_frames, start + 8)
        tensors = [
            torch.as_tensor(np.asarray(arrays[name][start:stop], np.float32), device=device)
            for name in ("dino", "conv", "geom", "flow")
        ]
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            value = model(*tensors)
        result[start:stop] = value.float().cpu().half()
        if stop % 100 == 0 or stop == max_frames:
            print(f"LATENTS {stop}/{max_frames}", flush=True)
    if not torch.isfinite(result).all():
        raise RuntimeError("Non-finite post-Perceiver latents")
    atomic_torch_save({"latents": result, "checkpoint_sha256": digest}, latent_path)
    return result, digest


def write_split(samples: list[dict], split: str, output: Path, digest: str, shard_size: int = 16) -> Path:
    split_dir = output / split
    split_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for start in range(0, len(samples), shard_size):
        block = samples[start : start + shard_size]
        shard = split_dir / f"shard_{start // shard_size:04d}.pt"
        atomic_torch_save(
            {"samples": block, "metadata": {"split": split, "resampler_sha256": digest}},
            shard,
        )
        rows.extend({"path": str(shard.resolve()), "index": index} for index in range(len(block)))
    manifest = output / f"{split}_manifest.jsonl"
    temporary = manifest.with_suffix(".tmp")
    temporary.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    os.replace(temporary, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export capped ForestEnv/P2000 K=1 shards from validated shared V7 caches")
    parser.add_argument("--shared-root", type=Path, default=DEFAULT_SHARED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-frames", type=int, default=1200)
    args = parser.parse_args()
    if not 10 <= args.max_frames <= 1200:
        raise ValueError("--max-frames must be between 10 and 1200")
    args.output.mkdir(parents=True, exist_ok=True)
    if os.statvfs(args.output).f_bavail * os.statvfs(args.output).f_frsize < 50 * 1024**3:
        raise RuntimeError("Less than 50 GB free before export")

    cache_root = args.shared_root / "cache"
    raw_base = args.shared_root / "raw_tartanground/ForestEnv/Data_anymal/P2000"
    checkpoint = args.shared_root / "results/V7_PRIOR_SHORT.pt"
    urdf = args.shared_root / ".work/v7_external/anymal_d_simple_description/urdf/anymal.urdf"
    required = [cache_root, raw_base / "rosbags/P2000.bag", checkpoint, urdf]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing shared pilot inputs: {missing}")

    physics_file = np.load(cache_root / "ForestEnv_P2000_physics.npz", allow_pickle=False)
    data = {name: physics_file[name] for name in physics_file.files}
    manifest_file = np.load(cache_root / "ForestEnv_P2000_frame_manifest.npz", allow_pickle=False)
    manifest = {name: manifest_file[name] for name in manifest_file.files}
    frame_count = min(args.max_frames, len(manifest["camera_times"]))
    latents, digest = export_latents(cache_root, checkpoint, args.output, frame_count)

    commanded_q = extract_commanded_positions(
        raw_base / "rosbags/P2000.bag",
        data["grid_original"],
        data["actuator_for_joint"],
        args.output / "commanded_positions.npy",
    )
    fk = StrictURDFFK(urdf)
    time = np.asarray(data["time"], np.float64)
    camera_times = np.asarray(manifest["camera_times"][:frame_count], np.float64)
    anchors = []
    for visual_index in range(3, frame_count - 1):
        current = int(np.searchsorted(time, camera_times[visual_index], side="right") - 1)
        future = int(np.searchsorted(time, camera_times[visual_index + 1], side="right") - 1)
        if current - 20 - 49 >= 0 and future > current and future < len(time):
            anchors.append((visual_index, current, future))
    if len(anchors) < 100:
        raise RuntimeError(f"Too few eligible anchors: {len(anchors)}")

    train_count = int(0.70 * len(anchors))
    validation_count = int(0.20 * len(anchors))
    train_anchors = anchors[:train_count]
    purge_end = time[train_anchors[-1][1]] + 5.0
    validation_pool = [row for row in anchors[train_count:] if time[row[1]] >= purge_end]
    validation_anchors = validation_pool[:validation_count]
    if not validation_anchors:
        raise RuntimeError("Five-second purge left no validation anchors")
    train_end = time[train_anchors[-1][1]]
    train_mask = time <= train_end

    P = np.asarray(data["P"], np.float32)
    B = np.asarray(data["B"], np.float32)
    U = np.asarray(data["U"], np.float32)
    L = np.asarray(data["L_LEG"], np.float32)
    E = np.asarray(data["E"], np.float32)
    contact = np.asarray(data["contact"], np.float32)
    eps = 1e-6
    p_mean, p_std = P[train_mask].mean((0, 1)), P[train_mask].std((0, 1)) + eps
    b_mean, b_std = B[train_mask].mean(0), B[train_mask].std(0) + eps
    u_mean, u_std = U[train_mask].mean(0), U[train_mask].std(0) + eps
    Pn = (P - p_mean) / p_std
    Bn = (B - b_mean) / b_std
    Un = (U - u_mean) / u_std
    Ln = (L - p_mean[6:9]) / p_std[6:9]
    En = E / p_std[6:9]
    leg_history = np.concatenate(
        [Pn, np.repeat(Un[:, None, :], 4, axis=1), Ln, En], axis=-1
    ).astype(np.float32)
    body_history = np.concatenate([Bn, Un], axis=-1).astype(np.float32)
    jointbody = np.concatenate([P[:, :, :6].reshape(len(P), 24), B], axis=-1)
    jointbody_std = jointbody[train_mask].std(0).astype(np.float32) + eps
    tau_std = P[train_mask, :, 6:9].std((0, 1)).astype(np.float32) + eps
    force_std = P[train_mask, :, 9:12].std((0, 1)).astype(np.float32) + eps
    train_magnitude = np.linalg.norm(P[train_mask, :, 9:12], axis=-1).reshape(-1)
    positive = train_magnitude[train_magnitude > 10.0]
    thresholds = np.quantile(positive, [0.25, 0.50, 0.75]).astype(np.float32)

    def regime(force: np.ndarray) -> np.ndarray:
        return np.digitize(np.linalg.norm(force, axis=-1), thresholds).astype(np.int64)

    def history(end: int) -> np.ndarray:
        return leg_history[end - 49 : end + 1].transpose(1, 0, 2).copy()

    def make_sample(row: tuple[int, int, int]) -> dict:
        visual_index, current, future = row
        interval_u = Un[current:future]
        interval_l = Ln[current:future]
        if len(interval_u) == 0:
            interval_u, interval_l = Un[current : current + 1], Ln[current : current + 1]
        future_action = np.concatenate(
            [interval_u[-1], interval_l[-1].reshape(-1), interval_u.mean(0), interval_l.mean(0).reshape(-1)]
        ).astype(np.float32)
        current_p = P[current]
        current_b = B[current]
        baseline_q = current_p[:, :3] + 0.1 * current_p[:, 3:6]
        baseline_jointbody = np.concatenate(
            [np.concatenate([baseline_q, current_p[:, 3:6]], axis=-1).reshape(-1), current_b]
        ).astype(np.float32)
        current_regime = regime(current_p[:, 9:12])
        one_hot = np.eye(4, dtype=np.float32)[current_regime]
        physics = np.concatenate(
            [
                ((baseline_q - p_mean[:3]) / p_std[:3]).reshape(-1),
                ((current_p[:, 6:9] - p_mean[6:9]) / p_std[6:9]).reshape(-1),
                (current_b[:3] - b_mean[:3]) / b_std[:3],
                np.linalg.norm(current_p[:, 9:12], axis=-1) / max(float(thresholds[-1]), 1.0),
                one_hot.reshape(-1),
            ]
        ).astype(np.float32)
        foot_position = fk.foot_positions(current_p[:, :3])
        planned_delta = np.tile((U[current, :3] * 0.30)[None, :], (4, 1)).astype(np.float32)
        query = np.concatenate(
            [Pn[current], np.tile(Un[current][None, :], (4, 1)), np.concatenate([foot_position, planned_delta], axis=-1)],
            axis=-1,
        ).astype(np.float32)
        sample = {
            "z_window": latents[visual_index - 3 : visual_index + 1].clone(),
            "leg_hist_t": torch.from_numpy(history(current)),
            "leg_hist_tm1": torch.from_numpy(history(current - 10)),
            "leg_hist_tm2": torch.from_numpy(history(current - 20)),
            "body_hist_t": torch.from_numpy(body_history[current - 49 : current + 1].copy()),
            "body_action_seq": torch.from_numpy(build_body_action(U[current])[None]),
            "leg_action_seq": torch.from_numpy(build_leg_action(commanded_q[current], commanded_q[future], L[current])[None]),
            "leg_query_features_seq": torch.from_numpy(query[None]),
            "future_action_seq": torch.from_numpy(future_action[None]),
            "physics_seq": torch.from_numpy(physics[None]),
            "horizon_seq": torch.tensor([[0.1]], dtype=torch.float32),
            "lowlevel_cmd_seq": torch.from_numpy(Ln[current][None].copy()),
            "err_seq": torch.from_numpy(En[current][None].copy()),
            "jointbody_baseline_seq": torch.from_numpy(baseline_jointbody[None]),
            "jointbody_std": torch.from_numpy(jointbody_std.copy()),
            "tau_baseline_seq": torch.from_numpy(current_p[None, :, 6:9].copy()),
            "tau_std": torch.from_numpy(tau_std.copy()),
            "force_baseline_seq": torch.from_numpy(current_p[None, :, 9:12].copy()),
            "force_std": torch.from_numpy(force_std.copy()),
            "target_z_seq": latents[visual_index + 1 : visual_index + 2].clone(),
            "target_contact_seq": torch.from_numpy(contact[future][None].copy()),
            "target_load_class_seq": torch.from_numpy(regime(P[future, :, 9:12])[None]),
            "target_jointbody_seq": torch.from_numpy(jointbody[future][None].copy()),
            "target_tau_seq": torch.from_numpy(P[future : future + 1, :, 6:9].copy()),
            "target_force_seq": torch.from_numpy(P[future : future + 1, :, 9:12].copy()),
            "target_phase_seq": torch.from_numpy(phase_vector(contact[current], contact[future])[None]),
        }
        validate_sample(sample)
        for key, value in sample.items():
            if torch.is_floating_point(value) and not torch.isfinite(value).all():
                raise RuntimeError(f"Non-finite sample tensor: {key}")
        return sample

    train_samples = [make_sample(row) for row in train_anchors]
    validation_samples = [make_sample(row) for row in validation_anchors]
    train_manifest = write_split(train_samples, "train", args.output, digest)
    validation_manifest = write_split(validation_samples, "val", args.output, digest)
    metadata = {
        "environment": "ForestEnv",
        "trajectory": "P2000",
        "horizon": 1,
        "max_frames": frame_count,
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "purge_seconds": 5.0,
        "resampler_checkpoint": str(checkpoint),
        "resampler_sha256": digest,
        "load_thresholds": thresholds.tolist(),
        "train_manifest": str(train_manifest),
        "val_manifest": str(validation_manifest),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    np.savez(
        args.output / "training_stats.npz",
        p_mean=p_mean,
        p_std=p_std,
        b_mean=b_mean,
        b_std=b_std,
        u_mean=u_mean,
        u_std=u_std,
        jointbody_std=jointbody_std,
        tau_std=tau_std,
        force_std=force_std,
        load_thresholds=thresholds,
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()


# CELL 44
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch


def make_sample(K=3, T=8):
    return {
        "z_window": torch.randn(4,64,256),
        "leg_hist_t": torch.randn(4,T,24),
        "leg_hist_tm1": torch.randn(4,T,24),
        "leg_hist_tm2": torch.randn(4,T,24),
        "body_hist_t": torch.randn(T,18),
        "body_action_seq": torch.randn(K,4),
        "leg_action_seq": torch.randn(K,4,4),
        "leg_query_features_seq": torch.randn(K,4,24),
        "future_action_seq": torch.randn(K,36),
        "physics_seq": torch.randn(K,47),
        "horizon_seq": torch.full((K,1),0.1),
        "lowlevel_cmd_seq": torch.randn(K,4,3),
        "err_seq": torch.randn(K,4,3),
        "jointbody_baseline_seq": torch.randn(K,36),
        "jointbody_std": torch.rand(36)+0.5,
        "tau_baseline_seq": torch.randn(K,4,3),
        "tau_std": torch.rand(3)+0.5,
        "force_baseline_seq": torch.randn(K,4,3)*20,
        "force_std": torch.rand(3)*10+10,
        "target_z_seq": torch.randn(K,64,256),
        "target_contact_seq": torch.randint(0,2,(K,4)).float(),
        "target_load_class_seq": torch.randint(0,4,(K,4)),
        "target_jointbody_seq": torch.randn(K,36),
        "target_tau_seq": torch.randn(K,4,3),
        "target_force_seq": torch.randn(K,4,3)*40,
        "target_phase_seq": torch.randint(0,4,(K,4)),
    }


def write_split(root: Path, name: str, n: int, shard_size: int):
    rows=[]
    split=root/name; split.mkdir(parents=True,exist_ok=True)
    for s0 in range(0,n,shard_size):
        samples=[make_sample() for _ in range(min(shard_size,n-s0))]
        path=(split/f"shard_{s0//shard_size:03d}.pt").resolve()
        torch.save({"samples":samples},path)
        for i in range(len(samples)): rows.append({"path":str(path),"index":i})
    manifest=root/f"{name}_manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(x) for x in rows)+"\n")
    print(manifest)


if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default="dummy_data"); ap.add_argument("--train",type=int,default=16); ap.add_argument("--val",type=int,default=4)
    a=ap.parse_args(); root=Path(a.out).resolve(); root.mkdir(parents=True,exist_ok=True)
    write_split(root,"train",a.train,4); write_split(root,"val",a.val,4)


# CELL 52
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def add_file(archive: zipfile.ZipFile, source: Path, arcname: str, included: dict[str, str]) -> None:
    if source.is_file():
        archive.write(source, arcname)
        included[arcname] = digest(source)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    required = []
    for architecture in ("baseline", "per_leg"):
        for horizon in (1, 5):
            checkpoint_dir = args.run_root / architecture / f"k{horizon}" / "checkpoints"
            required.extend([checkpoint_dir / "best.pt", checkpoint_dir / "latest.pt"])
            required.extend([
                args.run_root / architecture / f"k{horizon}" / "metrics.jsonl",
                args.run_root / architecture / f"k{horizon}" / "train.log",
                args.run_root / architecture / f"k{horizon}" / "resolved_config.json",
                args.run_root / architecture / f"k{horizon}" / "strict_test" / "strict_test_metrics.json",
                args.run_root / architecture / f"k{horizon}" / "strict_test" / "strict_test_predictions.npz",
            ])
    required.extend([
        args.run_root / "prepared" / "preparation_metadata.json",
        args.run_root / "prepared" / "training_stats.npz",
        args.run_root / "lambda_calibration" / "lambda_calibration.json",
        args.run_root / "phase.txt",
    ])
    required.extend(args.run_root / "prepared" / f"{split}_k{horizon}.jsonl"
                    for split in ("train", "dev", "test") for horizon in (1, 5))
    required.extend((args.run_root / "prepared" / "cache").glob("*/metadata.json"))
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required result files: {missing}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    included: dict[str, str] = {}
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in required:
            add_file(archive, path, str(path.relative_to(args.run_root)), included)
        for path in (args.run_root / "logs").rglob("*"):
            if path.is_file():
                add_file(archive, path, str(path.relative_to(args.run_root)), included)
        for path in (args.run_root / "lambda_calibration").glob("*/metrics.jsonl"):
            add_file(archive, path, str(path.relative_to(args.run_root)), included)
        for path in (args.run_root / "lambda_calibration").glob("*/calibration.yaml"):
            add_file(archive, path, str(path.relative_to(args.run_root)), included)
        for path in (args.root / "configs").glob("tartanground_*_k*.yaml"):
            add_file(archive, path, f"source/configs/{path.name}", included)
        for dirname in ("tartanground", "scripts", "tests", "docs/superpowers/specs", "docs/superpowers/plans"):
            directory = args.root / dirname
            if directory.exists():
                for path in directory.rglob("*"):
                    if path.is_file() and "__pycache__" not in path.parts:
                        add_file(archive, path, f"source/{path.relative_to(args.root)}", included)
        calibration_report = json.loads((args.run_root / "lambda_calibration" / "lambda_calibration.json").read_text())
        prep_report = json.loads((args.run_root / "prepared" / "preparation_metadata.json").read_text())
        result_readme = (
            "# TartanGround K=1/K=5 comparison results\n\n"
            f"Selected development-only lambda profile: `{calibration_report['selected']}` "
            f"with weights `{calibration_report['weights']}`.\n\n"
            f"Training trajectories: {len(prep_report['training_trajectories'])}; "
            f"strict unseen test trajectories: {', '.join(prep_report['holdout_trajectories'])}.\n\n"
            "Both architectures were trained first at K=1, then K=5 initialized from the corresponding K=1 checkpoint. "
            "K=5 training used a deterministic 50/50 teacher-forced/autoregressive schedule; development and strict testing were fully autoregressive. "
            "Each trajectory used two 700-frame windows centered as close as possible to 25% and 75%, with overlapping frames cached once.\n\n"
            "The archive contains best/latest checkpoints, training logs and metrics, development lambda calibration, strict-test metrics and per-sample predictions, "
            "split/window manifests, source code, tests, and SHA-256 checksums. Raw sensor and foundation-feature caches are excluded.\n\n"
            "Results describe only the four held-out trajectories and do not establish broader generalization. Per-horizon physical metrics are in each architecture/checkpoint `strict_test` directory.\n"
        )
        archive.writestr("README_RESULTS.md", result_readme)
        for name in ("model.py", "data.py", "losses.py", "config.py", "train.py", "checkpointing.py", "README.md", "requirements.txt"):
            add_file(archive, args.root / name, f"source/{name}", included)
        checksum_text = "\n".join(f"{value}  {name}" for name, value in sorted(included.items())) + "\n"
        archive.writestr("SHA256SUMS.txt", checksum_text)
    os.replace(temp, args.output)
    final_sha = digest(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(f"{final_sha}  {args.output.name}\n")
    print(json.dumps({
        "zip": str(args.output), "bytes": args.output.stat().st_size,
        "sha256": final_sha, "entries": len(included),
    }, indent=2))


if __name__ == "__main__":
    main()


# CELL 54
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path


DEFAULT_PROJECT = Path("/path/to/ugp/merged_feelwm_v7")
DEFAULT_RUN = Path(
    "/path/to/ugp/merged_feelwm_v7_runs/"
    "tartanground_pilot"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_files(project: Path) -> list[Path]:
    files: set[Path] = set()
    for pattern in ("*.py", "*.md", "*.txt", "*.sh"):
        files.update(path for path in project.glob(pattern) if path.is_file())
    allowed = {".py", ".yaml", ".yml", ".sh", ".md", ".txt", ".json"}
    for directory in ("configs", "scripts", "tartanground", "tests"):
        root = project / directory
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in allowed and "__pycache__" not in path.parts:
                files.add(path)
    return sorted(files)


def add_file(archive: zipfile.ZipFile, path: Path, arcname: str) -> None:
    compression = zipfile.ZIP_STORED if path.suffix.lower() in {".pt", ".npz"} else zipfile.ZIP_DEFLATED
    archive.write(path, arcname, compress_type=compression)


def main() -> None:
    parser = argparse.ArgumentParser(description="Package the verified TartanGround P2000 K=1 pilot results")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    project = args.project.resolve()
    run = args.run.resolve()
    output = args.output or run / "exports" / "tartanground_p2000_k1_results_step200.zip"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    paths = {
        "checkpoint/latest.pt": run / "k1_train/checkpoints/latest.pt",
        "run/resolved_config.json": run / "k1_train/resolved_config.json",
        "run/metrics.jsonl": run / "k1_train/metrics.jsonl",
        "run/k1_pipeline.log": run / "logs/k1_pipeline.log",
        "data/metadata.json": run / "prepared_k1/metadata.json",
        "data/training_stats.npz": run / "prepared_k1/training_stats.npz",
        "data/train_manifest.jsonl": run / "prepared_k1/train_manifest.jsonl",
        "data/val_manifest.jsonl": run / "prepared_k1/val_manifest.jsonl",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing package inputs: {missing}")

    metadata = json.loads(paths["data/metadata.json"].read_text())
    records = [json.loads(line) for line in paths["run/metrics.jsonl"].read_text().splitlines() if line.strip()]
    validations = [record for record in records if record.get("type") == "val"]
    if [record.get("step") for record in validations] != [50, 100, 150, 200]:
        raise RuntimeError("Expected validation records at steps 50, 100, 150, and 200")

    test = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=project,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if test.returncode != 0:
        raise RuntimeError(f"Unit verification failed:\n{test.stdout}")

    columns = [
        "step", "total", "dino", "conv", "geom", "flow", "contact", "load",
        "jointbody", "tau", "base_force", "base_force_aux", "v7f", "v7f_vec",
        "v7f_mag", "v7f_dir", "v7f_phase", "v7f_temp", "v7f_impulse",
    ]
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(validations)

    checkpoint_hash = sha256(paths["checkpoint/latest.pt"])
    summary = {
        "experiment": "TartanGround ForestEnv/P2000 K=1 pilot",
        "status": "completed",
        "global_step": 200,
        "effective_epochs_approx": 1.91,
        "trajectory_count": 1,
        "trajectory": metadata["trajectory"],
        "frames": metadata["max_frames"],
        "trajectory_duration_seconds": 119.89999866485596,
        "train_samples": metadata["train_samples"],
        "validation_manifest_samples": metadata["validation_samples"],
        "validation_samples_per_event": 100,
        "purge_seconds": metadata["purge_seconds"],
        "pipeline_wall_seconds": 592,
        "checkpoint_sha256": checkpoint_hash,
        "validation": validations,
        "limitations": [
            "Validation uses a later chronological segment of P2000, not an unseen trajectory.",
            "P2001 was unavailable and was not tested.",
            "This pilot exports loss components, not physical-unit RMSE or classification accuracy/F1.",
        ],
    }

    readme = f"""# TartanGround P2000 K=1 pilot results

This package contains the verified result of the credit-capped merged Feel-WM V7 pilot.

## Run summary

- Status: completed at global optimizer step 200
- Effective training: approximately 1.91 epochs
- Data: one ForestEnv/P2000 trajectory, 1,200 frames spanning 119.9 seconds
- Split: 834 training samples and 238 validation-manifest samples, with a 5-second purge gap
- Compute: two RTX 4080 GPUs, DDP, BF16, TF32, effective batch approximately 8
- Trainable parameters: 110.53 million
- End-to-end successful pipeline time: approximately 592 seconds
- Final/best validation total: {validations[-1]['total']:.8f}
- Final checkpoint SHA-256: {checkpoint_hash}

## Evaluation scope

Validation ran at steps 50, 100, 150, and 200. Each event covered 100 distributed
validation samples because the evaluator was capped at 50 batches per GPU. The raw
records are in `run/metrics.jsonl`; compact copies are in `run/validation_summary.csv`
and `run/validation_summary.json`.

This is not an unseen-trajectory test. Validation came from a later chronological
segment of P2000 after a 5-second purge. P2001 was unavailable and was not evaluated.
No physical-unit RMSE, accuracy, precision, recall, or F1 values were produced.

## Checkpoint

`checkpoint/latest.pt` is a resumable PyTorch checkpoint containing model, optimizer,
scheduler, RNG, configuration, epoch index, global step, and best-validation state.
Use only with the included trusted source. Loading the full resumable checkpoint uses
PyTorch pickle semantics.

The raw dataset, prepared sample shards, shared frozen V7 resampler checkpoint, and
redundant intermediate checkpoints are intentionally excluded to keep this package
small. Their provenance and the frozen-resampler SHA-256 are recorded in metadata.

## Verification

The source unit suite was rerun while packaging and passed. See
`verification/tests.txt`. `SHA256SUMS.txt` records hashes for packaged files.
"""

    source = source_files(project)
    checksums: dict[str, str] = {"checkpoint/latest.pt": checkpoint_hash}
    for arcname, path in paths.items():
        if arcname not in checksums:
            checksums[arcname] = sha256(path)
    for path in source:
        arcname = f"source/{path.relative_to(project).as_posix()}"
        checksums[arcname] = sha256(path)

    generated = {
        "README_RESULTS.md": readme,
        "run/validation_summary.csv": csv_buffer.getvalue(),
        "run/validation_summary.json": json.dumps(summary, indent=2) + "\n",
        "verification/tests.txt": test.stdout,
    }
    for arcname, value in generated.items():
        checksums[arcname] = hashlib.sha256(value.encode()).hexdigest()
    checksum_text = "".join(f"{digest}  {arcname}\n" for arcname, digest in sorted(checksums.items()))

    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
            for arcname, path in paths.items():
                add_file(archive, path, arcname)
            for path in source:
                add_file(archive, path, f"source/{path.relative_to(project).as_posix()}")
            for arcname, value in generated.items():
                archive.writestr(arcname, value, compress_type=zipfile.ZIP_DEFLATED)
            archive.writestr("SHA256SUMS.txt", checksum_text, compress_type=zipfile.ZIP_DEFLATED)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP integrity check failed at {bad}")

    print(json.dumps({
        "output": str(output),
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "files": len(checksums) + 1,
        "unit_tests": "passed",
    }, indent=2))


if __name__ == "__main__":
    main()


# CELL 56
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tartanground.evaluate import STRICT_HOLDOUTS

CASES_PER_TRAJECTORY = 10


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Package strict unseen-test outputs only; exclude training artifacts.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    files: dict[str, Path] = {}
    summary: dict[str, dict] = {}
    for architecture in ("baseline", "per_leg"):
        summary[architecture] = {}
        for horizon in (1, 5):
            prefix = Path(architecture) / f"k{horizon}" / "strict_test"
            result_dir = args.run_root / prefix
            metrics_path = result_dir / "strict_test_metrics.json"
            predictions_path = result_dir / "strict_test_predictions.npz"
            selection_path = result_dir / "strict_test_selection.json"
            csv_path = result_dir / "strict_test_predictions.csv"
            for path in (metrics_path, predictions_path, selection_path, csv_path):
                if not path.is_file():
                    raise FileNotFoundError(f"Missing strict-test artifact: {path}")
            metrics = json.loads(metrics_path.read_text())
            selection = json.loads(selection_path.read_text())
            counts = Counter(selection["cases_by_trajectory"])
            if set(counts) != STRICT_HOLDOUTS or any(count != CASES_PER_TRAJECTORY for count in counts.values()):
                raise ValueError(f"Expected {CASES_PER_TRAJECTORY} cases for each strict holdout, found {dict(counts)}")
            if selection.get("total_cases") != CASES_PER_TRAJECTORY * len(STRICT_HOLDOUTS):
                raise ValueError(f"Unexpected strict-test sample count in {selection_path}")
            if len(selection.get("cases", [])) != selection["total_cases"]:
                raise ValueError(f"Case selection list is inconsistent in {selection_path}")
            case_ids = [case["case_id"] for case in selection["cases"]]
            if len(set(case_ids)) != len(case_ids):
                raise ValueError(f"Duplicate case IDs in {selection_path}")
            with np.load(predictions_path, allow_pickle=False) as predictions:
                if len(predictions["case_id"]) != selection["total_cases"]:
                    raise ValueError(f"Prediction count does not match selected cases in {predictions_path}")
                if predictions["case_id"].tolist() != case_ids:
                    raise ValueError(f"Prediction case IDs do not match selection in {predictions_path}")
            if metrics.get("sample_count") != selection["total_cases"]:
                raise ValueError(f"Metric sample count does not match selection in {metrics_path}")
            summary[architecture][f"k{horizon}"] = {
                "test_cases": selection["total_cases"],
                "cases_by_trajectory": dict(sorted(selection["cases_by_trajectory"].items())),
                "global_metrics": metrics["global"],
                "by_trajectory": metrics["by_trajectory"],
                "by_horizon": metrics["by_horizon"],
                "by_leg": metrics["by_leg"],
            }
            for source in (metrics_path, predictions_path, selection_path, csv_path):
                files[str(source.relative_to(args.run_root))] = source

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    included: dict[str, str] = {}
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for name, source in sorted(files.items()):
            archive.write(source, name)
            included[name] = digest(source)
        summary_text = json.dumps({"strict_unseen_trajectories": sorted(STRICT_HOLDOUTS), "models": summary}, indent=2, allow_nan=False) + "\n"
        archive.writestr("test_summary.json", summary_text)
        included["test_summary.json"] = hashlib.sha256(summary_text.encode("utf-8")).hexdigest()
        readme = (
            "# Strict unseen trajectory test results\n\n"
            "This archive contains test metrics and predictions only. It contains no training checkpoints, training logs, training/development examples, source code, or raw sensor caches.\n\n"
            f"The strict test set has {len(STRICT_HOLDOUTS)} trajectories with exactly {CASES_PER_TRAJECTORY} shared, evenly spaced test cases per trajectory ({CASES_PER_TRAJECTORY * len(STRICT_HOLDOUTS)} cases for each model/horizon evaluation).\n\n"
            "The four trajectories are ForestEnv/P2001, GreatMarsh/P2027, OldTownFall/P2001, and SeasonalForestWinter/P2006. K=1 and K=5 use the same selected camera anchors. The CSV files give per-case, per-horizon, per-leg predicted and actual values; NPZ files retain the complete numeric arrays; JSON files provide aggregate metrics and case selection details.\n"
        )
        archive.writestr("README_TEST_RESULTS.md", readme)
        included["README_TEST_RESULTS.md"] = hashlib.sha256(readme.encode("utf-8")).hexdigest()
        checksums = "".join(f"{value}  {name}\n" for name, value in sorted(included.items()))
        archive.writestr("SHA256SUMS.txt", checksums)

    os.replace(temp, args.output)
    archive_sha = digest(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(f"{archive_sha}  {args.output.name}\n")
    print(json.dumps({"zip": str(args.output), "bytes": args.output.stat().st_size,
                      "sha256": archive_sha, "test_artifact_files": len(files)}, indent=2), flush=True)


if __name__ == "__main__":
    main()


# CELL 58
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "scripts"))
from export_shared_p2000_k1 import StrictURDFFK
from tartanground.pilot import load_v7_resamplers
from tartanground.windows import encode_cache_native_action, quarter_windows, split_anchors, union_with_membership


DEFAULT_SHARED = Path("/path/to/ugp/V7F_4080_Run/storage")
DEFAULT_OUTPUT = Path("/path/to/ugp/merged_feelwm_v7_runs/tartanground_k1_k5_leg_compare/prepared")
HOLDOUTS = {
    "ForestEnv/P2001", "GreatMarsh/P2027", "OldTownFall/P2001", "SeasonalForestWinter/P2006",
}


def trajectory_name(environment: str, trajectory: str) -> str:
    return f"{environment}_{trajectory}"


def read_train_list(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty trajectory list: {path}")
    columns = {str(key).strip().lower(): key for key in rows[0]}
    env_key = next((columns[key] for key in ("environment", "env", "world") if key in columns), None)
    traj_key = next((columns[key] for key in ("trajectory", "trajectory_id", "id", "bag") if key in columns), None)
    names = []
    for row in rows:
        if env_key and traj_key:
            env = str(row[env_key]).strip().rstrip("/")
            raw_traj = str(row[traj_key]).strip().rstrip("/")
            if "/" in raw_traj and not env:
                env, raw_traj = raw_traj.rsplit("/", 1)
            traj = raw_traj.split("/")[-1]
            if not env or not traj:
                continue
            names.append(f"{env}/{traj}")
        else:
            joined = "/".join(str(value) for value in row.values())
            match = re.search(r"(ForestEnv|GreatMarsh|OldTownFall|SeasonalForestWinter)[/_-](P\d+)", joined)
            if match:
                names.append(f"{match.group(1)}/{match.group(2)}")
    names = sorted(set(names))
    if len(names) != 28:
        raise ValueError(f"Expected 28 unique training trajectories from {path}, found {len(names)}: {names}")
    if set(names) & HOLDOUTS:
        raise ValueError("The authoritative training list overlaps a strict holdout")
    return names


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + ".tmp.npy")
    np.save(temp, np.asarray(value), allow_pickle=False)
    os.replace(temp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def camera_to_physics(time: np.ndarray, camera_times: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(time, camera_times, side="right") - 1
    if np.any(indices < 0) or np.any(indices >= len(time)):
        raise ValueError("Camera timestamps are outside the physics grid")
    return indices.astype(np.int64)


def load_source(shared: Path, name: str) -> dict[str, np.ndarray]:
    env, traj = name.split("/")
    cache = shared / "cache"
    with np.load(cache / f"{env}_{traj}_physics.npz", allow_pickle=False) as source:
        required = ("P", "B", "U", "L_LEG", "E", "contact", "time")
        data = {key: np.asarray(source[key], np.float32) for key in required}
    with np.load(cache / f"{env}_{traj}_frame_manifest.npz", allow_pickle=False) as source:
        camera_times = np.asarray(source["camera_times"], np.float64)
    if len(camera_times) < 700 or len(data["P"]) != len(data["time"]):
        raise ValueError(f"Invalid cache dimensions for {name}")
    return {**data, "camera_times": camera_times}


def prepare_trajectory(
    shared: Path, output: Path, name: str, data: dict[str, np.ndarray], windows,
    resampler, device: torch.device, resampler_sha: str, fk: StrictURDFFK,
    batch_size: int,
) -> tuple[Path, list[dict]]:
    env, traj = name.split("/")
    cache_dir = output / "cache" / trajectory_name(env, traj)
    cache_dir.mkdir(parents=True, exist_ok=True)
    selected, membership = union_with_membership(windows)
    selected_array = np.asarray(selected, dtype=np.int64)
    cache_meta = cache_dir / "metadata.json"
    source_paths = [
        shared / "cache" / f"{env}_{traj}_{suffix}"
        for suffix in ("physics.npz", "frame_manifest.npz", "v7_dinov3.npy", "v7_convnext.npy", "v7_utonia.npy", "v7_waft.npy")
    ]
    fingerprint = hashlib.sha256(json.dumps([
        [path.name, path.stat().st_size, path.stat().st_mtime_ns] for path in source_paths
    ], sort_keys=True).encode()).hexdigest()
    old = json.loads(cache_meta.read_text()) if cache_meta.exists() else {}
    cached_files = [cache_dir / f"{key}.npy" for key in (
        "P", "B", "U", "L_LEG", "E", "contact", "time", "camera_times",
        "latents", "foot_position", "foot_jacobian", "action_features",
    )]
    if (old.get("fingerprint") == fingerprint and old.get("resampler_sha256") == resampler_sha
            and old.get("selected_frames") == selected and all(path.exists() for path in cached_files)):
        print(f"CACHE REUSE {name} frames={len(selected)}", flush=True)
    else:
        arrays = {
            "dino": np.load(shared / "cache" / f"{env}_{traj}_v7_dinov3.npy", mmap_mode="r", allow_pickle=False),
            "conv": np.load(shared / "cache" / f"{env}_{traj}_v7_convnext.npy", mmap_mode="r", allow_pickle=False),
            "geom": np.load(shared / "cache" / f"{env}_{traj}_v7_utonia.npy", mmap_mode="r", allow_pickle=False),
            "flow": np.load(shared / "cache" / f"{env}_{traj}_v7_waft.npy", mmap_mode="r", allow_pickle=False),
        }
        if min(len(value) for value in arrays.values()) < len(data["camera_times"]):
            raise ValueError(f"Incomplete visual cache for {name}")
        for key, value in data.items():
            if key != "camera_times":
                atomic_npy(cache_dir / f"{key}.npy", value)
        atomic_npy(cache_dir / "camera_times.npy", data["camera_times"])

        latent_tmp = cache_dir / "latents.tmp.npy"
        latents = np.lib.format.open_memmap(
            latent_tmp, mode="w+", dtype=np.float16, shape=(len(data["camera_times"]), 64, 256)
        )
        latents[:] = 0
        resampler.eval()
        for start in range(0, len(selected), batch_size):
            camera_ids = selected_array[start : start + batch_size]
            inputs = [torch.as_tensor(np.asarray(arrays[key][camera_ids], np.float32), device=device) for key in ("dino", "conv", "geom", "flow")]
            with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                encoded = resampler(*inputs)
            values = encoded.float().cpu().numpy().astype(np.float16)
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite post-Perceiver values for {name}")
            latents[camera_ids] = values
            done = start + len(camera_ids)
            if done % 128 == 0 or done == len(selected):
                print(f"PREP {name} latents {done}/{len(selected)}", flush=True)
        latents.flush()
        del latents
        os.replace(latent_tmp, cache_dir / "latents.npy")

        current_indices = camera_to_physics(data["time"], data["camera_times"])
        foot = np.zeros((len(current_indices), 4, 3), np.float32)
        jacobian = np.zeros((len(current_indices), 4, 3, 3), np.float32)
        for camera_index in selected:
            q = data["P"][current_indices[camera_index], :, :3]
            foot[camera_index] = fk.foot_positions(q)
            epsilon = 1e-4
            for joint in range(3):
                shifted = q.copy()
                shifted[:, joint] += epsilon
                jacobian[camera_index, :, :, joint] = (fk.foot_positions(shifted) - foot[camera_index]) / epsilon
        atomic_npy(cache_dir / "foot_position.npy", foot)
        atomic_npy(cache_dir / "foot_jacobian.npy", jacobian)

        command_at_camera = data["L_LEG"][current_indices]
        action_features = encode_cache_native_action(command_at_camera)
        atomic_npy(cache_dir / "action_features.npy", action_features)
        cache_meta.write_text(json.dumps({
            "trajectory": name, "fingerprint": fingerprint, "resampler_sha256": resampler_sha,
            "selected_frames": selected, "window_membership": {str(k): list(v) for k, v in membership.items()},
            "windows": [{
                "start": window.start, "stop_exclusive": window.stop,
                "requested_center": window.requested_center, "clamped_center": window.center,
                "start_time": float(data["camera_times"][window.start]),
                "end_time": float(data["camera_times"][window.stop - 1]),
            } for window in windows],
        }, indent=2))

    current_indices = camera_to_physics(data["time"], data["camera_times"])
    max_horizon = 5
    anchors: dict[int, tuple[int, ...]] = {}
    for window_id, window in enumerate(windows):
        first = window.start + 3
        stop = window.stop - max_horizon
        for camera_index in range(first, stop):
            sequence = current_indices[camera_index - 20] if camera_index >= 20 else -1
            if sequence < 69:
                continue
            anchors.setdefault(camera_index, tuple())
            anchors[camera_index] = tuple(sorted(set(anchors[camera_index] + (window_id,))))
    records = []
    for camera_index, window_ids in sorted(anchors.items()):
        current = current_indices[camera_index : camera_index + max_horizon + 1]
        if len(current) != max_horizon + 1 or np.any(np.diff(current) <= 0):
            continue
        records.append({
            "trajectory_id": name, "cache_dir": str(cache_dir.resolve()),
            "camera_index": camera_index,
            "current_indices": current.tolist(),
            "target_visual_indices": list(range(camera_index + 1, camera_index + max_horizon + 1)),
            "window_membership": list(window_ids),
        })
    if not records:
        raise ValueError(f"No valid five-horizon anchors for {name}")
    return cache_dir, records


def fit_training_stats(output: Path, train_records: list[dict]) -> Path:
    grouped: dict[str, set[int]] = {}
    action_camera: dict[str, set[int]] = {}
    for row in train_records:
        root = row["cache_dir"]
        camera_index = int(row["camera_index"])
        current = row["current_indices"]
        indices = grouped.setdefault(root, set())
        indices.update(range(max(0, int(current[0]) - 69), int(current[-1]) + 1))
        action_camera.setdefault(root, set()).update(range(camera_index, camera_index + 6))

    def collect(field: str, roots_and_indices):
        chunks = []
        for root, indices in roots_and_indices.items():
            chunks.append(np.load(Path(root) / f"{field}.npy", mmap_mode="r", allow_pickle=False)[sorted(indices)])
        return np.concatenate(chunks, axis=0)

    p = collect("P", grouped)
    body = collect("B", grouped)
    control = collect("U", grouped)
    command = collect("L_LEG", grouped)
    error = collect("E", grouped)
    jointbody = np.concatenate([p[:, :, :6].reshape(len(p), 24), body], axis=-1)
    magnitudes = np.linalg.norm(p[:, :, 9:12], axis=-1).reshape(-1)
    positive = magnitudes[magnitudes > 10.0]
    if len(positive) < 100:
        raise ValueError("Insufficient training contacts to calibrate load thresholds")
    thresholds = np.quantile(positive, [0.25, 0.50, 0.75]).astype(np.float32)

    action = np.concatenate([
        np.load(Path(root) / "action_features.npy", mmap_mode="r", allow_pickle=False)[sorted(indices)]
        for root, indices in action_camera.items()
    ], axis=0)
    stats_path = output / "training_stats.npz"
    np.savez(
        stats_path,
        proprio_mean=p.mean((0, 1)).astype(np.float32), proprio_std=(p.std((0, 1)) + 1e-6).astype(np.float32),
        body_mean=body.mean(0).astype(np.float32), body_std=(body.std(0) + 1e-6).astype(np.float32),
        control_mean=control.mean(0).astype(np.float32), control_std=(control.std(0) + 1e-6).astype(np.float32),
        command_mean=command.mean((0, 1)).astype(np.float32), command_std=(command.std((0, 1)) + 1e-6).astype(np.float32),
        error_std=(error.std((0, 1)) + 1e-6).astype(np.float32),
        jointbody_std=(jointbody.std(0) + 1e-6).astype(np.float32),
        tau_std=(p[:, :, 6:9].std((0, 1)) + 1e-6).astype(np.float32),
        force_std=(p[:, :, 9:12].std((0, 1)) + 1e-6).astype(np.float32),
        load_thresholds=thresholds,
        command_delta_mean=np.asarray(action[:, :, 3].mean(), np.float32),
        command_delta_std=np.asarray(action[:, :, 3].std() + 1e-6, np.float32),
    )
    return stats_path


def manifest_row(record: dict, stats_path: Path, horizon: int, split: str) -> dict:
    row = dict(record)
    row["current_indices"] = row["current_indices"][: horizon + 1]
    row["target_visual_indices"] = row["target_visual_indices"][:horizon]
    row["stats_path"] = str(stats_path.resolve())
    row["split"] = split
    return row


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n")
    os.replace(tmp, path)


def interleave_by_trajectory(records: list[dict]) -> list[dict]:
    buckets: dict[str, list[dict]] = {}
    for row in records:
        buckets.setdefault(row["trajectory_id"], []).append(row)
    result = []
    trajectories = sorted(buckets)
    for offset in range(max((len(rows) for rows in buckets.values()), default=0)):
        for trajectory in trajectories:
            if offset < len(buckets[trajectory]):
                result.append(buckets[trajectory][offset])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-root", type=Path, default=DEFAULT_SHARED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    train_csv = args.shared_root / "results/V7_EXPANDED_TRAIN_TRAJECTORIES.csv"
    names = read_train_list(train_csv)
    all_names = names + sorted(HOLDOUTS)
    checkpoint = args.shared_root / "results/V7_PRIOR_SHORT.pt"
    urdf = args.shared_root / ".work/v7_external/anymal_d_simple_description/urdf/anymal.urdf"
    missing = [str(path) for path in (train_csv, checkpoint, urdf) if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    args.output.mkdir(parents=True, exist_ok=True)
    if shutil_free(args.output) < 20 * 1024**3:
        raise RuntimeError("Less than 20 GB free before shared preparation")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    resampler, resampler_sha = load_v7_resamplers(checkpoint, device)
    fk = StrictURDFFK(urdf)
    trajectory_records: dict[str, list[dict]] = {}
    for name in all_names:
        data = load_source(args.shared_root, name)
        windows = quarter_windows(len(data["camera_times"]))
        _, records = prepare_trajectory(
            args.shared_root, args.output, name, data, windows, resampler, device,
            resampler_sha, fk, args.batch_size,
        )
        trajectory_records[name] = records
        print(f"INDEX {name} anchors={len(records)} windows={[w.start for w in windows]}", flush=True)

    split_records: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    for name in names:
        records = trajectory_records[name]
        split = split_anchors((row["camera_index"] for row in records), purge_frames=50, train_fraction=0.8)
        for key in ("train", "dev"):
            selected = set(split[key])
            split_records[key].extend(row for row in records if row["camera_index"] in selected)
        print(f"SPLIT {name} train={len(split['train'])} purge={len(split['purge'])} dev={len(split['dev'])}", flush=True)
    for name in sorted(HOLDOUTS):
        split_records["test"].extend(trajectory_records[name])
    stats_path = fit_training_stats(args.output, split_records["train"])
    metadata = {
        "training_trajectories": names, "holdout_trajectories": sorted(HOLDOUTS),
        "window_frames": 700, "window_center_fractions": [0.25, 0.75],
        "window_semantics": "centers rounded and starts clamped to valid range; overlapping frames cached once with both memberships",
        "purge_camera_frames": 50, "training_stats": str(stats_path.resolve()),
        "resampler_sha256": resampler_sha,
        "counts": {key: len(value) for key, value in split_records.items()},
    }
    (args.output / "preparation_metadata.json").write_text(json.dumps(metadata, indent=2))
    for horizon in (1, 5):
        for split, records in split_records.items():
            if split == "dev":
                records = interleave_by_trajectory(records)
            rows = [manifest_row(row, stats_path, horizon, split) for row in records]
            write_manifest(args.output / f"{split}_k{horizon}.jsonl", rows)
    print(json.dumps(metadata, indent=2), flush=True)


def shutil_free(path: Path) -> int:
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize


if __name__ == "__main__":
    main()


# CELL 64
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tartanground.evaluate import STRICT_HOLDOUTS, validate_strict_split


def read_manifest(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(content)
    os.replace(temp, path)


def select_shared_rows(k1_rows: list[dict], k5_rows: list[dict], count: int):
    grouped = {1: defaultdict(dict), 5: defaultdict(dict)}
    for row in k1_rows:
        grouped[1][row["trajectory_id"]][int(row["camera_index"])] = row
    for row in k5_rows:
        grouped[5][row["trajectory_id"]][int(row["camera_index"])] = row

    if set(grouped[1]) != STRICT_HOLDOUTS or set(grouped[5]) != STRICT_HOLDOUTS:
        raise ValueError("Both strict-test manifests must contain exactly the four approved unseen trajectories")

    selected = {1: [], 5: []}
    report = {"requested_cases_per_trajectory": count,
              "sampling": f"{count} evenly spaced common camera anchors", "trajectories": {}}
    for trajectory in sorted(STRICT_HOLDOUTS):
        common = sorted(set(grouped[1][trajectory]) & set(grouped[5][trajectory]))
        if len(common) < count:
            raise ValueError(
                f"{trajectory} has only {len(common)} shared valid test anchors; {count} requested"
            )
        if count == 1:
            positions = [len(common) // 2]
        else:
            positions = [round(i * (len(common) - 1) / (count - 1)) for i in range(count)]
        anchors = [common[position] for position in positions]
        if len(set(anchors)) != count:
            raise RuntimeError(f"Anchor selection was not unique for {trajectory}")
        case_ids = []
        for camera_index in anchors:
            case_id = f"{trajectory}@camera_{camera_index:04d}"
            case_ids.append(case_id)
            for horizon in (1, 5):
                row = dict(grouped[horizon][trajectory][camera_index])
                row["test_case_id"] = case_id
                selected[horizon].append(row)
        report["trajectories"][trajectory] = {
            "common_valid_anchors": len(common),
            "selected_cases": count,
            "camera_indices": anchors,
            "case_ids": case_ids,
        }

    return selected, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Select shared, evenly spaced strict-unseen test anchors for K=1 and K=5.")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--k1-manifest", type=Path, required=True)
    parser.add_argument("--k5-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases-per-trajectory", type=int, default=10)
    args = parser.parse_args()
    if args.cases_per_trajectory < 1:
        raise ValueError("--cases-per-trajectory must be positive")
    validate_strict_split(args.train_manifest, args.dev_manifest, args.k1_manifest)
    validate_strict_split(args.train_manifest, args.dev_manifest, args.k5_manifest)
    selected, report = select_shared_rows(read_manifest(args.k1_manifest), read_manifest(args.k5_manifest), args.cases_per_trajectory)
    args.output.mkdir(parents=True, exist_ok=True)
    for horizon in (1, 5):
        text = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in selected[horizon])
        atomic_write(args.output / f"test_k{horizon}_cases{args.cases_per_trajectory}.jsonl", text)
    atomic_write(args.output / "selection.json", json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"total_cases_per_horizon": len(selected[1]), "selection": report}, indent=2), flush=True)


if __name__ == "__main__":
    main()


# CELL 66
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import ShardedWorldModelDataset, validate_sample

if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("manifest"); ap.add_argument("--n",type=int,default=10); a=ap.parse_args()
    ds=ShardedWorldModelDataset(a.manifest,validate_first=0)
    for i in range(min(a.n,len(ds))): validate_sample(ds[i])
    print(f"OK: validated {min(a.n,len(ds))}/{len(ds)} samples from {a.manifest}")


# CELL 68
from __future__ import annotations
import torch
from config import ModelConfig, TrainConfig
from model import MergedFeelWMV7
from losses import compute_losses


def sample(B=1,K=2,T=5):
    dev="cuda" if torch.cuda.is_available() else "cpu"
    def r(*s): return torch.randn(*s,device=dev)
    b={
        "z_window":r(B,4,64,256), "leg_hist_t":r(B,4,T,24), "leg_hist_tm1":r(B,4,T,24), "leg_hist_tm2":r(B,4,T,24),
        "body_hist_t":r(B,T,18), "body_action_seq":r(B,K,4), "leg_action_seq":r(B,K,4,4),
        "leg_query_features_seq":r(B,K,4,24), "future_action_seq":r(B,K,36), "physics_seq":r(B,K,47),
        "horizon_seq":torch.full((B,K,1),0.1,device=dev), "lowlevel_cmd_seq":r(B,K,4,3), "err_seq":r(B,K,4,3),
        "jointbody_baseline_seq":r(B,K,36), "jointbody_std":torch.ones(B,36,device=dev),
        "tau_baseline_seq":r(B,K,4,3), "tau_std":torch.ones(B,3,device=dev),
        "force_baseline_seq":r(B,K,4,3), "force_std":torch.ones(B,3,device=dev)*10,
        "target_z_seq":r(B,K,64,256), "target_contact_seq":torch.randint(0,2,(B,K,4),device=dev).float(),
        "target_load_class_seq":torch.randint(0,4,(B,K,4),device=dev), "target_jointbody_seq":r(B,K,36),
        "target_tau_seq":r(B,K,4,3), "target_force_seq":r(B,K,4,3), "target_phase_seq":torch.randint(0,4,(B,K,4),device=dev),
    }
    return b


if __name__ == "__main__":
    # Tiny-capacity test keeps CPU/GPU verification quick while preserving all tensor wiring and 10-head divisibility.
    cfg=ModelConfig(world_d_model=80,world_heads=10,world_layers=2,world_ffn_ratio=2,query_layers=1,hist_layers=1,v7f_layers=1)
    tcfg=TrainConfig(amp_dtype="fp32")
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m=MergedFeelWMV7(cfg).to(dev)
    b=sample()
    o=m.rollout(b,history_shift=1)
    losses=compute_losses(o,b,cfg,tcfg)
    assert o["v7f"]["force"].shape==(1,2,4,3)
    assert o["steps"][0]["z_next"].shape==(1,64,256)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    print("SMOKE TEST PASSED")
    print("force",tuple(o["v7f"]["force"].shape),"z_next",tuple(o["steps"][0]["z_next"].shape),"loss",float(losses["total"].detach().cpu()))


# CELL 70
"""TartanGround pilot preparation helpers."""



# CELL 72
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


STRICT_HOLDOUTS = {
    "ForestEnv/P2001", "GreatMarsh/P2027", "OldTownFall/P2001", "SeasonalForestWinter/P2006",
}


def _manifest_trajectories(path: str | Path) -> set[str]:
    return {
        json.loads(line)["trajectory_id"]
        for line in Path(path).read_text().splitlines()
        if line.strip()
    }


def validate_strict_split(train_manifest: str | Path, dev_manifest: str | Path,
                          test_manifest: str | Path) -> None:
    train = _manifest_trajectories(train_manifest)
    dev = _manifest_trajectories(dev_manifest)
    test = _manifest_trajectories(test_manifest)
    overlap = test & (train | dev)
    if overlap:
        raise ValueError(
            "strict test trajectories must be unseen in both training and development; "
            f"overlap: {sorted(overlap)}"
        )
    if test != STRICT_HOLDOUTS:
        raise ValueError(f"strict test manifest must contain exactly the four holdouts; found {sorted(test)}")


def normalized_physical_score(out: dict[str, Any], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Equal mean of normalized q, tau, force, and contact errors across horizons."""
    steps = out["steps"]
    K = len(steps)
    batch_size = len(steps[0]["jointbody_pred"])
    joint_scale = batch["jointbody_std"]
    tau_scale = batch["tau_std"]
    force_scale = batch["force_std"]
    if joint_scale.dim() == 1:
        joint_scale = joint_scale[None].expand(batch_size, -1)
    if tau_scale.dim() == 1:
        tau_scale = tau_scale[None].expand(batch_size, -1)
    if force_scale.dim() == 1:
        force_scale = force_scale[None].expand(batch_size, -1)
    q_scale = joint_scale[:, :24].reshape(batch_size, 4, 6)[:, :, :3].clamp_min(1e-6)
    tau_scale = tau_scale.reshape(batch_size, 1, 3).clamp_min(1e-6)
    force_scale = force_scale.reshape(batch_size, 1, 3).clamp_min(1e-6)
    q_errors, tau_errors, force_errors, contact_errors = [], [], [], []
    for k, step in enumerate(steps):
        target_joint = batch["target_jointbody_seq"][:, k, :24].reshape(batch_size, 4, 6)[..., :3]
        predicted_joint = step["jointbody_pred"][:, :24].reshape(batch_size, 4, 6)[..., :3]
        q_errors.append(((predicted_joint - target_joint).abs() / q_scale).mean())
        tau_errors.append(((step["tau_pred"] - batch["target_tau_seq"][:, k]).abs() / tau_scale).mean())
        force_errors.append(((out["v7f"]["force"][:, k] - batch["target_force_seq"][:, k]).abs() / force_scale).mean())
        contact_errors.append((step["contact_prob"] - batch["target_contact_seq"][:, k]).abs().mean())
    return torch.stack([
        torch.stack(q_errors).mean(), torch.stack(tau_errors).mean(),
        torch.stack(force_errors).mean(), torch.stack(contact_errors).mean(),
    ]).mean()


def physical_metrics(prediction: dict[str, np.ndarray], target: dict[str, np.ndarray],
                     contact_threshold: float = 0.5, force_threshold: float = 20.0) -> dict[str, float]:
    """Return physical-unit metrics from [sample,horizon,leg,channel] arrays."""
    q_error = np.asarray(prediction["q"]) - np.asarray(target["q"])
    tau_error = np.asarray(prediction["tau"]) - np.asarray(target["tau"])
    force_pred = np.asarray(prediction["force"])
    force_true = np.asarray(target["force"])
    force_error = force_pred - force_true
    contact_pred = np.asarray(prediction["contact"]) >= contact_threshold
    contact_true = np.asarray(target["contact"]) >= 0.5
    tp = int(np.sum(contact_pred & contact_true))
    fp = int(np.sum(contact_pred & ~contact_true))
    fn = int(np.sum(~contact_pred & contact_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    magnitude_pred = np.linalg.norm(force_pred, axis=-1)
    magnitude_true = np.linalg.norm(force_true, axis=-1)
    contacting = magnitude_true >= force_threshold
    cosine = np.sum(force_pred * force_true, axis=-1) / np.maximum(magnitude_pred * magnitude_true, 1e-8)
    return {
        "q_mae_rad": float(np.mean(np.abs(q_error))),
        "q_rmse_rad": float(np.sqrt(np.mean(q_error ** 2))),
        "q_mae_deg": float(np.degrees(np.mean(np.abs(q_error)))),
        "q_rmse_deg": float(np.degrees(np.sqrt(np.mean(q_error ** 2)))),
        "tau_mae_nm": float(np.mean(np.abs(tau_error))),
        "tau_rmse_nm": float(np.sqrt(np.mean(tau_error ** 2))),
        "force_vector_mae": float(np.mean(np.abs(force_error))),
        "force_vector_rmse": float(np.sqrt(np.mean(force_error ** 2))),
        "force_magnitude_mae": float(np.mean(np.abs(magnitude_pred - magnitude_true))),
        "force_direction_cosine_contact": float(np.mean(cosine[contacting])) if contacting.any() else float("nan"),
        "contact_accuracy": float(np.mean(contact_pred == contact_true)),
        "contact_precision": precision,
        "contact_recall": recall,
        "contact_f1": 2 * precision * recall / max(1e-8, precision + recall),
    }


# CELL 74
from __future__ import annotations

import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from data import validate_sample
from tartanground.pilot import build_body_action, phase_vector


def _tensor(value: np.ndarray | list | float, dtype=None) -> torch.Tensor:
    result = torch.as_tensor(np.asarray(value), dtype=dtype)
    return result.contiguous()


class IndexedTartanGroundDataset(Dataset):
    """Build model samples lazily from per-trajectory memory-mapped cache arrays."""

    def __init__(self, manifest: str | Path, cache_trajectories: int = 4):
        self.manifest = Path(manifest)
        self.rows = [json.loads(line) for line in self.manifest.read_text().splitlines() if line.strip()]
        if not self.rows:
            raise RuntimeError(f"Empty indexed manifest: {self.manifest}")
        self.cache_trajectories = max(1, int(cache_trajectories))
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.trajectory_indices: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(self.rows):
            self.trajectory_indices[row["trajectory_id"]].append(index)
        self.stats_path = Path(self.rows[0]["stats_path"])
        with np.load(self.stats_path, allow_pickle=False) as stats:
            self.stats = {key: np.asarray(stats[key], np.float32) for key in stats.files}

    def __len__(self) -> int:
        return len(self.rows)

    def _trajectory(self, row: dict[str, Any]) -> dict[str, Any]:
        path = row["cache_dir"]
        if path in self._cache:
            value = self._cache.pop(path)
            self._cache[path] = value
            return value
        root = Path(path)
        names = (
            "P", "B", "U", "L_LEG", "E", "contact", "time",
            "camera_times", "latents", "foot_position", "foot_jacobian", "action_features",
        )
        value = {name: np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False) for name in names}
        self._cache[path] = value
        while len(self._cache) > self.cache_trajectories:
            self._cache.popitem(last=False)
        return value

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        arrays = self._trajectory(row)
        stats = self.stats
        visual_index = int(row["camera_index"])
        current = np.asarray(row["current_indices"], dtype=np.int64)
        targets_visual = np.asarray(row["target_visual_indices"], dtype=np.int64)
        K = len(targets_visual)
        p = arrays["P"]
        body = arrays["B"]
        control = arrays["U"]
        lowlevel = arrays["L_LEG"]
        error = arrays["E"]
        contact = arrays["contact"]
        time = arrays["time"]

        p_mean, p_std = stats["proprio_mean"], np.maximum(stats["proprio_std"], 1e-6)
        b_mean, b_std = stats["body_mean"], np.maximum(stats["body_std"], 1e-6)
        u_mean, u_std = stats["control_mean"], np.maximum(stats["control_std"], 1e-6)
        l_mean, l_std = stats["command_mean"], np.maximum(stats["command_std"], 1e-6)
        e_std = np.maximum(stats["error_std"], 1e-6)
        Pn = ((p - p_mean) / p_std).astype(np.float32, copy=False)
        Bn = ((body - b_mean) / b_std).astype(np.float32, copy=False)
        Un = ((control - u_mean) / u_std).astype(np.float32, copy=False)
        Ln = ((lowlevel - l_mean) / l_std).astype(np.float32, copy=False)
        En = (error / e_std).astype(np.float32, copy=False)

        def leg_history(end: int) -> np.ndarray:
            start = end - 49
            return np.concatenate([
                Pn[start : end + 1],
                np.repeat(Un[start : end + 1, None, :], 4, axis=1),
                Ln[start : end + 1], En[start : end + 1],
            ], axis=-1).transpose(1, 0, 2).copy()

        body_history = np.concatenate([Bn, Un], axis=-1)
        future_actions = []
        leg_actions = []
        body_actions = []
        queries = []
        physics = []
        jointbody_baselines = []
        tau_baselines = []
        force_baselines = []
        lowlevel_seq = []
        error_seq = []
        target_z = []
        target_contact = []
        target_load = []
        target_jointbody = []
        target_tau = []
        target_force = []
        target_phase = []
        thresholds = stats["load_thresholds"]

        for k in range(K):
            now, nxt = int(current[k]), int(current[k + 1])
            target_now = int(np.searchsorted(time, arrays["camera_times"][targets_visual[k]], side="right") - 1)
            if target_now != nxt:
                nxt = target_now
            u_interval = Un[now : nxt + 1]
            l_interval = Ln[now : nxt + 1]
            if not len(u_interval):
                u_interval, l_interval = Un[now : now + 1], Ln[now : now + 1]
            future_actions.append(np.concatenate([
                u_interval[-1], l_interval[-1].reshape(-1),
                u_interval.mean(0), l_interval.mean(0).reshape(-1),
            ]).astype(np.float32))
            body_actions.append(build_body_action(control[now]))
            action = np.asarray(arrays["action_features"][visual_index + k], np.float32)
            delta_mean = float(stats["command_delta_mean"])
            delta_std = max(float(stats["command_delta_std"]), 1e-6)
            leg_actions.append(np.concatenate([
                (action[:, :3] - l_mean) / l_std,
                ((action[:, 3:4] - delta_mean) / delta_std),
            ], axis=-1))
            lowlevel_seq.append(Ln[now])
            error_seq.append(En[now])

            current_p = np.asarray(p[now], np.float32)
            current_b = np.asarray(body[now], np.float32)
            current_contact = np.asarray(contact[now], np.float32)
            baseline_q = current_p[:, :3] + 0.1 * current_p[:, 3:6]
            baseline = np.concatenate([
                np.concatenate([baseline_q, current_p[:, 3:6]], axis=-1).reshape(-1), current_b
            ]).astype(np.float32)
            jointbody_baselines.append(baseline)
            tau_baselines.append(current_p[:, 6:9])
            force_baselines.append(current_p[:, 9:12])

            force_norm = np.linalg.norm(current_p[:, 9:12], axis=-1)
            regime = np.digitize(force_norm, thresholds).astype(np.int64)
            regime_onehot = np.eye(4, dtype=np.float32)[regime]
            physics.append(np.concatenate([
                ((baseline_q - p_mean[:3]) / p_std[:3]).reshape(-1),
                ((current_p[:, 6:9] - p_mean[6:9]) / p_std[6:9]).reshape(-1),
                (current_b[:3] - b_mean[:3]) / b_std[:3],
                force_norm / max(float(thresholds[-1]), 1.0),
                (regime_onehot * current_contact[:, None]).reshape(-1),
            ]).astype(np.float32))

            foot = np.asarray(arrays["foot_position"][visual_index + k], np.float32)
            planned_delta = np.tile((control[now, :3] * 0.30)[None], (4, 1)).astype(np.float32)
            queries.append(np.concatenate([Pn[now], np.tile(Un[now][None], (4, 1)), foot, planned_delta], axis=-1))

            target_p = np.asarray(p[nxt], np.float32)
            target_contact_k = np.asarray(contact[nxt], np.float32)
            target_force_k = target_p[:, 9:12]
            target_z.append(np.asarray(arrays["latents"][targets_visual[k]], np.float16))
            target_contact.append(target_contact_k)
            target_load.append(np.digitize(np.linalg.norm(target_force_k, axis=-1), thresholds).astype(np.int64))
            target_jointbody.append(np.concatenate([target_p[:, :6].reshape(-1), np.asarray(body[nxt], np.float32)]))
            target_tau.append(target_p[:, 6:9])
            target_force.append(target_force_k)
            target_phase.append(phase_vector(current_contact, target_contact_k > 0.5))

        sample = {
            "z_window": _tensor(np.asarray(arrays["latents"][visual_index - 3 : visual_index + 1], np.float16)),
            "leg_hist_t": _tensor(leg_history(int(current[0]))),
            "leg_hist_tm1": _tensor(leg_history(int(current[0]) - 10)),
            "leg_hist_tm2": _tensor(leg_history(int(current[0]) - 20)),
            "body_hist_t": _tensor(body_history[int(current[0]) - 49 : int(current[0]) + 1].copy()),
            "body_action_seq": _tensor(body_actions),
            "leg_action_seq": _tensor(leg_actions),
            "leg_query_features_seq": _tensor(queries),
            "future_action_seq": _tensor(future_actions),
            "physics_seq": _tensor(physics),
            "horizon_seq": _tensor(np.arange(1, K + 1, dtype=np.float32)[:, None] * 0.1),
            "lowlevel_cmd_seq": _tensor(lowlevel_seq),
            "err_seq": _tensor(error_seq),
            "jointbody_baseline_seq": _tensor(jointbody_baselines),
            "jointbody_std": _tensor(stats["jointbody_std"]),
            "tau_baseline_seq": _tensor(tau_baselines),
            "tau_std": _tensor(stats["tau_std"]),
            "force_baseline_seq": _tensor(force_baselines),
            "force_std": _tensor(stats["force_std"]),
            "target_z_seq": _tensor(target_z),
            "target_contact_seq": _tensor(target_contact),
            "target_load_class_seq": _tensor(target_load, torch.long),
            "target_jointbody_seq": _tensor(target_jointbody),
            "target_tau_seq": _tensor(target_tau),
            "target_force_seq": _tensor(target_force),
            "target_phase_seq": _tensor(target_phase, torch.long),
            "proprio_mean": _tensor(stats["proprio_mean"]),
            "proprio_std": _tensor(stats["proprio_std"]),
            "body_mean": _tensor(stats["body_mean"]),
            "body_std": _tensor(stats["body_std"]),
            "load_thresholds": _tensor(stats["load_thresholds"]),
            "foot_jacobian": _tensor(arrays["foot_jacobian"][visual_index]),
        }
        validate_sample(sample)
        return sample


class BalancedTrajectorySampler(Sampler[int]):
    """Shuffle short within-trajectory blocks and cycle through all trajectories."""

    def __init__(self, dataset: IndexedTartanGroundDataset, seed: int = 781, block_size: int = 64):
        self.dataset = dataset
        self.seed = int(seed)
        self.block_size = max(1, int(block_size))
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        per_trajectory = {}
        blocks = []
        for trajectory, indices in self.dataset.trajectory_indices.items():
            order = torch.randperm(len(indices), generator=generator).tolist()
            shuffled = [indices[i] for i in order]
            per_trajectory[trajectory] = [
                shuffled[start : start + self.block_size]
                for start in range(0, len(shuffled), self.block_size)
            ]
            blocks.extend((trajectory, i) for i in range(len(per_trajectory[trajectory])))
        if not blocks:
            return iter(())
        order = torch.randperm(len(blocks), generator=generator).tolist()
        result = []
        for block_index in order:
            trajectory, block = blocks[block_index]
            result.extend(per_trajectory[trajectory][block])
        return iter(result)


# CELL 76
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


# CELL 78
from __future__ import annotations

import torch


def teacher_forcing_mask(
    batch_size: int,
    horizons: int,
    sample_offset: int = 0,
    seed: int = 781,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Choose one deterministic mode per example, alternating exactly 50/50."""
    if batch_size < 0 or horizons < 1:
        raise ValueError("batch_size must be nonnegative and horizons must be positive")
    ids = torch.arange(batch_size, device=device, dtype=torch.long) + int(sample_offset)
    forced = (ids + int(seed)) % 2 == 0
    mask = forced[:, None].expand(batch_size, horizons).clone()
    mask[:, 0] = False  # the initial observation is always the same for both modes
    return mask


# CELL 80
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Window:
    start: int
    stop: int
    requested_center: float

    @property
    def center(self) -> float:
        return 0.5 * (self.start + self.stop - 1)

    @property
    def indices(self) -> range:
        return range(self.start, self.stop)


def quarter_windows(frame_count: int, length: int = 700, min_center_gap: int = 50) -> tuple[Window, Window]:
    frame_count = int(frame_count)
    length = int(length)
    if frame_count < length:
        raise ValueError(f"trajectory has {frame_count} frames but each window requires {length}")
    maximum_start = frame_count - length
    half_span = 0.5 * (length - 1)
    result = []
    for fraction in (0.25, 0.75):
        requested = fraction * (frame_count - 1)
        start = int(round(requested - half_span))
        start = min(max(start, 0), maximum_start)
        result.append(Window(start=start, stop=start + length, requested_center=requested))
    if result[1].center - result[0].center < int(min_center_gap):
        raise ValueError("the two 700-frame window centers cannot be separated by five seconds")
    return result[0], result[1]


def union_with_membership(windows: Sequence[Window]) -> tuple[list[int], dict[int, tuple[int, ...]]]:
    memberships: dict[int, list[int]] = {}
    for window_id, window in enumerate(windows):
        for index in window.indices:
            memberships.setdefault(index, []).append(window_id)
    indices = sorted(memberships)
    return indices, {index: tuple(memberships[index]) for index in indices}


def split_anchors(
    anchor_indices: Iterable[int], purge_frames: int = 50, train_fraction: float = 0.8
) -> dict[str, list[int]]:
    anchors = sorted(set(int(index) for index in anchor_indices))
    if not anchors:
        return {"train": [], "purge": [], "dev": []}
    cut = int(len(anchors) * float(train_fraction))
    cut = min(max(cut, 1), len(anchors))
    purge_until = anchors[cut - 1] + 1 + int(purge_frames)
    purge_stop = cut
    while purge_stop < len(anchors) and anchors[purge_stop] < purge_until:
        purge_stop += 1
    return {
        "train": anchors[:cut],
        "purge": anchors[cut:purge_stop],
        "dev": anchors[purge_stop:],
    }


def encode_cache_native_action(commands: np.ndarray) -> np.ndarray:
    commands = np.asarray(commands, dtype=np.float32)
    if commands.ndim < 3 or commands.shape[-1] != 3:
        raise ValueError("commands must have shape [time, leg, 3]")
    delta = np.zeros(commands.shape[:-1] + (1,), dtype=np.float32)
    if len(commands) > 1:
        delta[:-1, ..., 0] = np.linalg.norm(commands[1:] - commands[:-1], axis=-1)
    return np.concatenate([commands, delta], axis=-1)


# CELL 82
import subprocess
import sys
import unittest


class ExportP2000EntrypointTest(unittest.TestCase):
    def test_help_runs_from_project_root(self):
        result = subprocess.run(
            [sys.executable, "scripts/export_shared_p2000_k1.py", "--help"],
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--max-frames", result.stdout)
        self.assertIn("--output", result.stdout)


if __name__ == "__main__":
    unittest.main()


# CELL 84
import unittest

import torch

from tartanground.rollout import teacher_forcing_mask


class TeacherForcingScheduleTests(unittest.TestCase):
    def test_schedule_is_deterministic_and_balanced_over_even_examples(self):
        first = teacher_forcing_mask(8, 5, sample_offset=0, seed=781)
        again = teacher_forcing_mask(8, 5, sample_offset=0, seed=781)
        torch.testing.assert_close(first, again)
        self.assertEqual(int(first[:, 1].sum()), 4)
        self.assertTrue(torch.equal(first[:, 1], first[:, 4]))
        self.assertFalse(bool(first[:, 0].any()))

    def test_adjacent_microbatches_split_even_when_batch_size_is_one(self):
        first = teacher_forcing_mask(1, 5, sample_offset=0, seed=781)
        second = teacher_forcing_mask(1, 5, sample_offset=1, seed=781)
        self.assertNotEqual(bool(first[0, 1]), bool(second[0, 1]))

    def test_k1_has_no_teacher_forcing(self):
        mask = teacher_forcing_mask(3, 1, sample_offset=0, seed=781)
        self.assertEqual(tuple(mask.shape), (3, 1))
        self.assertFalse(bool(mask.any()))

    def test_k5_autoregressive_outputs_ignore_future_measured_state_fields(self):
        from config import ModelConfig
        from model import MergedFeelWMV7
        from smoke_test import sample

        torch.manual_seed(17)
        cfg = ModelConfig(
            world_d_model=80, world_heads=10, world_layers=1, world_ffn_ratio=2,
            query_layers=1, hist_layers=1, v7f_layers=1,
        )
        model = MergedFeelWMV7(cfg).eval()
        batch = {key: value.cpu() for key, value in sample(B=1, K=5, T=5).items()}
        changed = {key: value.clone() for key, value in batch.items()}
        for key in (
            "physics_seq", "jointbody_baseline_seq", "tau_baseline_seq", "force_baseline_seq",
            "target_z_seq", "target_contact_seq",
            "target_jointbody_seq", "target_tau_seq", "target_force_seq", "target_phase_seq",
            "err_seq",
        ):
            changed[key][:, 1:] = changed[key][:, 1:] + 123.0
        changed["target_load_class_seq"][:, 1:] = (changed["target_load_class_seq"][:, 1:] + 1) % 4
        changed["leg_query_features_seq"][:, 1:, :, :12] += 123.0
        changed["leg_query_features_seq"][:, 1:, :, 18:21] += 123.0
        with torch.no_grad():
            first = model(batch)
            second = model(changed)
        for horizon in range(5):
            for key in ("z_next", "jointbody_pred", "tau_pred", "base_force", "contact_prob", "regime_prob"):
                torch.testing.assert_close(first["steps"][horizon][key], second["steps"][horizon][key])
        torch.testing.assert_close(first["v7f"]["force"], second["v7f"]["force"])

    def test_autoregressive_feedback_is_refined_v7f_force(self):
        from config import ModelConfig
        from model import MergedFeelWMV7
        from smoke_test import sample

        torch.manual_seed(19)
        cfg = ModelConfig(
            world_d_model=80, world_heads=10, world_layers=1, world_ffn_ratio=2,
            query_layers=1, hist_layers=1, v7f_layers=1,
        )
        model = MergedFeelWMV7(cfg).eval()
        batch = {key: value.cpu() for key, value in sample(B=1, K=3, T=5).items()}
        with torch.no_grad():
            output = model(batch)
        torch.testing.assert_close(output["feedback_force_seq"][:, :-1], output["refined_force_seq"][:, :-1])

    def test_k5_loss_uses_all_five_configured_horizon_lambdas(self):
        from config import ModelConfig, TrainConfig
        from losses import compute_losses
        from smoke_test import sample

        torch.manual_seed(23)
        batch = {key: value.cpu() for key, value in sample(B=1, K=5, T=5).items()}
        batch["target_z_seq"].zero_()
        batch["target_contact_seq"].zero_()
        batch["target_load_class_seq"].zero_()
        batch["target_jointbody_seq"].zero_()
        batch["target_tau_seq"].zero_()
        batch["target_force_seq"].zero_()
        steps = []
        for horizon in range(5):
            value = float(horizon + 1)
            steps.append({
                "dino": torch.full((1, 16, 256), value),
                "conv": torch.full((1, 16, 256), value),
                "geom": torch.full((1, 16, 256), value),
                "flow": torch.full((1, 16, 256), value),
                "contact_logits": torch.full((1, 4), value),
                "regime_logits": torch.full((1, 4, 4), value),
                "jointbody_pred": torch.full((1, 36), value),
                "tau_pred": torch.full((1, 4, 3), value),
                "base_force": torch.full((1, 4, 3), value),
                "force_aux_logmag": torch.full((1, 4), value),
                "force_aux_dir": torch.full((1, 4, 3), value / 10.0),
            })
        spec = {
            "force": torch.stack([torch.full((1, 4, 3), float(k + 1)) for k in range(5)], dim=1),
            "phase_logits": torch.zeros(1, 5, 4, 4),
            "corr_norm": torch.zeros(1, 5, 4, 3),
            "expert_w": torch.full((1, 5, 4, 4), 0.25),
        }
        out = {"steps": steps, "v7f": spec}
        weights = [0.10, 0.15, 0.20, 0.25, 0.30]
        losses = compute_losses(out, batch, ModelConfig(), TrainConfig(horizon_lambdas=tuple(weights)))
        self.assertTrue(all(f"horizon_{k}" in losses for k in range(1, 6)))
        expected = sum(weights[k] * losses[f"horizon_{k + 1}"] for k in range(5))
        expected = expected + losses["temporal_regularizer"]
        expected = expected + losses["impulse_regularizer"]
        torch.testing.assert_close(losses["total"], expected)
        self.assertGreater(float(losses["horizon_5"]), float(losses["horizon_1"]))


if __name__ == "__main__":
    unittest.main()


# CELL 86
import unittest

import torch

from config import ModelConfig
from model import MergedFeelWMV7


class LegPreTransformerTests(unittest.TestCase):
    def config(self, enabled):
        return ModelConfig(
            world_d_model=80,
            world_heads=10,
            world_layers=1,
            world_ffn_ratio=2,
            hist_layers=1,
            query_layers=1,
            v7f_layers=1,
            use_leg_pretransformers=enabled,
            leg_pre_d_model=32,
            leg_pre_heads=4,
            leg_pre_layers=1,
        )

    def test_disabled_path_has_no_extra_parameters_and_preserves_groups(self):
        torch.manual_seed(13)
        model = MergedFeelWMV7(self.config(False))
        groups = torch.randn(2, 9, 64, 80)
        self.assertEqual(model.leg_pre_transformers, None)
        torch.testing.assert_close(model.apply_leg_pretransformers(groups), groups)
        self.assertFalse(any("leg_pre_transformers" in name for name, _ in model.named_parameters()))

    def test_same_seed_keeps_all_shared_weights_identical(self):
        torch.manual_seed(29)
        baseline = MergedFeelWMV7(self.config(False))
        torch.manual_seed(29)
        per_leg = MergedFeelWMV7(self.config(True))
        baseline_state = baseline.state_dict()
        leg_state = per_leg.state_dict()
        for name, value in baseline_state.items():
            torch.testing.assert_close(value, leg_state[name])

    def test_enabled_path_uses_four_independent_blocks_on_only_the_leg_groups(self):
        torch.manual_seed(13)
        model = MergedFeelWMV7(self.config(True))
        self.assertEqual(set(model.leg_pre_transformers.keys()), {"fl", "fr", "hl", "hr"})
        groups = torch.randn(2, 9, 64, 80, requires_grad=True)
        transformed = model.apply_leg_pretransformers(groups)
        self.assertEqual(tuple(transformed.shape), (2, 9, 64, 80))
        for group in (0, 3, 6, 7, 8):
            torch.testing.assert_close(transformed[:, group], groups[:, group])
        for group in (1, 2, 4, 5):
            self.assertFalse(torch.equal(transformed[:, group], groups[:, group]))
        transformed.square().mean().backward()
        for name, block in model.leg_pre_transformers.items():
            for parameter_name, parameter in block.named_parameters():
                self.assertIsNotNone(parameter.grad, f"{name}.{parameter_name} received no gradient")
                self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertFalse(torch.equal(
            next(model.leg_pre_transformers["fl"].parameters()),
            next(model.leg_pre_transformers["fr"].parameters()),
        ))


if __name__ == "__main__":
    unittest.main()


# CELL 88
import unittest

from config import TrainConfig
from train import max_steps_reached


class MaxStepsTest(unittest.TestCase):
    def test_max_steps_stops_after_exact_optimizer_step(self):
        cfg = TrainConfig(max_steps=3)

        self.assertFalse(max_steps_reached(0, cfg.max_steps))
        self.assertFalse(max_steps_reached(2, cfg.max_steps))
        self.assertTrue(max_steps_reached(3, cfg.max_steps))

    def test_resumed_run_does_not_repeat_completed_step(self):
        cfg = TrainConfig(max_steps=200)

        resumed_global_step = 200
        self.assertTrue(max_steps_reached(resumed_global_step, cfg.max_steps))

    def test_unbounded_training_preserves_existing_behavior(self):
        self.assertFalse(max_steps_reached(10_000, None))


if __name__ == "__main__":
    unittest.main()


# CELL 90
from __future__ import annotations

import unittest

import torch

from config import ModelConfig, TrainConfig
from losses import compute_losses
from model import MergedFeelWMV7
from smoke_test import sample


class ModelForwardTest(unittest.TestCase):
    def test_module_call_runs_autoregressive_rollout(self) -> None:
        cfg = ModelConfig(
            world_d_model=80,
            world_heads=10,
            world_layers=1,
            world_ffn_ratio=2,
            query_layers=1,
            hist_layers=1,
            v7f_layers=1,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MergedFeelWMV7(cfg).to(device)
        batch = sample(B=1, K=1, T=5)

        try:
            output = model(batch, history_shift=1)
        except TypeError as exc:
            self.fail(f"module call must run the rollout instead of raising TypeError: {exc}")

        self.assertEqual(tuple(output["steps"][0]["z_next"].shape), (1, 64, 256))
        self.assertEqual(tuple(output["v7f"]["force"].shape), (1, 1, 4, 3))

    def test_force_auxiliary_head_receives_gradient(self) -> None:
        cfg = ModelConfig(
            world_d_model=80,
            world_heads=10,
            world_layers=1,
            world_ffn_ratio=2,
            query_layers=1,
            hist_layers=1,
            v7f_layers=1,
        )
        model = MergedFeelWMV7(cfg).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        batch = sample(B=1, K=1, T=5)
        output = model(batch, history_shift=1)
        loss = compute_losses(output, batch, cfg, TrainConfig(amp_dtype="fp32"))["total"]
        loss.backward()

        for name, parameter in model.force_aux_head.named_parameters():
            self.assertIsNotNone(parameter.grad, f"force_aux_head.{name} did not receive a gradient")
            self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()


# CELL 92
import unittest

import numpy as np

from tartanground.windows import (
    encode_cache_native_action,
    quarter_windows,
    split_anchors,
    union_with_membership,
)


class QuarterWindowTests(unittest.TestCase):
    def test_two_700_windows_are_quarter_centered_and_overlap_is_deduplicated(self):
        first, second = quarter_windows(1290)
        self.assertEqual((first.start, first.stop), (0, 700))
        self.assertEqual((second.start, second.stop), (590, 1290))
        self.assertAlmostEqual(first.requested_center, 0.25 * 1289)
        self.assertAlmostEqual(second.requested_center, 0.75 * 1289)

        indices, membership = union_with_membership((first, second))
        self.assertEqual(len(indices), 1290)
        self.assertEqual(membership[100], (0,))
        self.assertEqual(membership[650], (0, 1))
        self.assertEqual(membership[1200], (1,))

    def test_rejects_trajectories_that_cannot_separate_window_centers_by_five_seconds(self):
        with self.assertRaisesRegex(ValueError, "five seconds"):
            quarter_windows(740)

    def test_chronological_split_has_a_50_frame_purge(self):
        anchors = list(range(100, 1100))
        split = split_anchors(anchors)
        self.assertEqual(split["train"][0], 100)
        self.assertEqual(split["train"][-1], 899)
        self.assertEqual(split["purge"], list(range(900, 950)))
        self.assertEqual(split["dev"][0], 950)
        self.assertTrue(set(split["train"]).isdisjoint(split["dev"]))

    def test_purge_uses_camera_time_even_when_anchor_indices_are_sparse(self):
        split = split_anchors([0, 1, 2, 3, 4, 5, 6, 100, 101, 102], train_fraction=0.5)
        self.assertEqual(split["train"], [0, 1, 2, 3, 4])
        self.assertEqual(split["purge"], [5, 6])
        self.assertEqual(split["dev"], [100, 101, 102])


class CacheNativeActionTests(unittest.TestCase):
    def test_encodes_three_low_level_commands_and_next_change_magnitude(self):
        commands = np.array(
            [
                [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]],
                [[4.0, 6.0, 3.0], [0.0, 3.0, 4.0]],
                [[4.0, 6.0, 3.0], [0.0, 3.0, 4.0]],
            ],
            dtype=np.float32,
        )
        encoded = encode_cache_native_action(commands)
        np.testing.assert_allclose(encoded[..., :3], commands)
        np.testing.assert_allclose(encoded[:, :, 3], [[5.0, 5.0], [0.0, 0.0], [0.0, 0.0]])


if __name__ == "__main__":
    unittest.main()


# CELL 94
import unittest

import numpy as np
import torch

from tartanground.pilot import (
    FrozenVisualResamplers,
    build_body_action,
    build_leg_action,
    contact_phase,
)


class TartanGroundPilotTest(unittest.TestCase):
    def test_body_and_leg_actions_use_commands(self):
        body = build_body_action(np.array([2.0, -1.0, 9.0, 4.0, 5.0, 0.5], np.float32))
        np.testing.assert_allclose(body, [0.2, -0.1, 0.05, 0.1])

        q0 = np.array([[0.0, 1.0, 2.0]] * 4, np.float32)
        q1 = q0 + np.array([[9.0, 0.2, -0.3]] * 4, np.float32)
        tau = np.array([[8.0, 4.0, -5.0]] * 4, np.float32)
        got = build_leg_action(q0, q1, tau)
        np.testing.assert_allclose(got[0], [0.2, -0.3, 4.0, -5.0], atol=1e-6)

    def test_contact_phase_labels_four_transitions(self):
        self.assertEqual(contact_phase(False, False), 0)
        self.assertEqual(contact_phase(False, True), 1)
        self.assertEqual(contact_phase(True, True), 2)
        self.assertEqual(contact_phase(True, False), 3)

    def test_resamplers_export_fixed_branch_order_without_modality_embedding(self):
        torch.manual_seed(7)
        model = FrozenVisualResamplers(dino_dim=5, conv_dim=6, geom_dim=7).eval()
        dino = torch.randn(2, 64, 5)
        conv = torch.randn(2, 64, 6)
        geom = torch.randn(2, 64, 7)
        flow = torch.randn(2, 4, 64, 6)
        with torch.inference_mode():
            out = model(dino, conv, geom, flow)
            expected = torch.cat([
                model.dino(dino),
                model.conv(conv),
                model.geom(geom),
                model.flow(model.flow_temporal(flow)),
            ], dim=1)
        self.assertEqual(tuple(out.shape), (2, 64, 256))
        torch.testing.assert_close(out, expected)


if __name__ == "__main__":
    unittest.main()


# CELL 96
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ValidateManifestEntrypointTest(unittest.TestCase):
    def test_help_runs_from_project_root(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/validate_manifest.py", "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("usage:", result.stdout.lower())


if __name__ == "__main__":
    unittest.main()


# CELL 98
from __future__ import annotations
import argparse
import json
import math
import os
import random
import signal
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from config import load_config, config_to_dict
from model import MergedFeelWMV7
from data import ShardedWorldModelDataset, move_batch_to_device
from losses import compute_losses
from checkpointing import save_checkpoint, load_checkpoint, resolve_resume, prune_step_checkpoints, unwrap_model
from tartanground.rollout import teacher_forcing_mask
from tartanground.evaluate import normalized_physical_score


STOP_REQUESTED = False


def on_signal(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def is_main(rank):
    return rank == 0


def seed_everything(seed: int, rank: int):
    s = seed + rank * 100003
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def make_optimizer(model, lr: float, wd: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or name.endswith("bias") or "norm" in name.lower() or "embedding" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": wd},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=lr, betas=(0.9, 0.95))


def make_scheduler(optimizer, warmup: int, total_steps: int, min_ratio: float):
    def f(step: int):
        if step < warmup:
            return max(1e-8, step / max(1, warmup))
        progress = (step - warmup) / max(1, total_steps - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return min_ratio + (1.0 - min_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def amp_context(dtype_name: str):
    if dtype_name == "fp32" or not torch.cuda.is_available():
        return nullcontext()
    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def reduce_scalar(x: torch.Tensor, world_size: int):
    y = x.detach().float()
    if world_size > 1:
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
        y /= world_size
    return float(y.cpu())


def validate(model, loader, device, mcfg, tcfg, world_size, max_batches=50):
    model.eval()
    sums = {}
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            with amp_context(tcfg.amp_dtype):
                out = model(batch, history_shift=tcfg.history_shift_samples)
                losses = compute_losses(out, batch, mcfg, tcfg)
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + reduce_scalar(v, world_size)
            score = normalized_physical_score(out, batch)
            sums["physical_score"] = sums.get("physical_score", 0.0) + reduce_scalar(score, world_size)
            count += 1
            if count >= max_batches:
                break
    model.train()
    return {k: v / max(count, 1) for k, v in sums.items()}


def max_steps_reached(global_step: int, max_steps: int | None) -> bool:
    return max_steps is not None and global_step >= max_steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_2x4080.yaml")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    distributed, rank, world_size, local_rank = init_distributed()
    mcfg, tcfg, _ = load_config(args.config)
    seed_everything(tcfg.seed, rank)

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.backends.cuda.matmul.allow_tf32 = tcfg.tf32
        torch.backends.cudnn.allow_tf32 = tcfg.tf32
    else:
        device = torch.device("cpu")

    out_dir = Path(tcfg.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    if is_main(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "resolved_config.json").write_text(json.dumps(config_to_dict(mcfg, tcfg), indent=2))

    if tcfg.indexed_data:
        from tartanground.indexed import IndexedTartanGroundDataset, BalancedTrajectorySampler
        train_ds = IndexedTartanGroundDataset(tcfg.train_manifest)
        val_ds = IndexedTartanGroundDataset(tcfg.val_manifest)
        balanced_sampler = BalancedTrajectorySampler(train_ds, seed=tcfg.seed)
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=tcfg.seed
        ) if distributed else balanced_sampler
    else:
        train_ds = ShardedWorldModelDataset(tcfg.train_manifest)
        val_ds = ShardedWorldModelDataset(tcfg.val_manifest)
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=tcfg.seed) if distributed else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None

    loader_kwargs = dict(
        batch_size=tcfg.batch_size,
        num_workers=tcfg.num_workers,
        pin_memory=tcfg.pin_memory,
        persistent_workers=tcfg.persistent_workers and tcfg.num_workers > 0,
    )
    if tcfg.num_workers > 0:
        loader_kwargs["prefetch_factor"] = tcfg.prefetch_factor
    train_loader = DataLoader(train_ds, shuffle=train_sampler is None, sampler=train_sampler, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, sampler=val_sampler, drop_last=False, **loader_kwargs)

    model = MergedFeelWMV7(mcfg).to(device)
    if tcfg.init_weights_only:
        source = Path(tcfg.init_weights_only)
        if not source.exists():
            raise FileNotFoundError(source)
        initial = torch.load(source, map_location="cpu", weights_only=False)
        model.load_state_dict(initial["model"], strict=True)
        if is_main(rank):
            print(f"INITIALIZED MODEL WEIGHTS ONLY: {source}; optimizer and scheduler are fresh")
    if tcfg.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=False)

    optimizer = make_optimizer(model, tcfg.lr, tcfg.weight_decay)
    optim_steps_per_epoch = math.ceil(len(train_loader) / tcfg.grad_accum_steps)
    total_optim_steps = max(1, tcfg.epochs * optim_steps_per_epoch)
    if tcfg.max_steps is not None:
        if tcfg.max_steps < 1:
            raise ValueError("train.max_steps must be a positive integer or null")
        total_optim_steps = min(total_optim_steps, tcfg.max_steps)
    scheduler = make_scheduler(optimizer, tcfg.warmup_steps, total_optim_steps, tcfg.min_lr_ratio)
    scaler = torch.amp.GradScaler("cuda", enabled=(tcfg.amp_dtype == "fp16" and torch.cuda.is_available()))

    start_epoch, global_step, best_val = 0, 0, float("inf")
    resume_path = resolve_resume(tcfg.resume, tcfg.output_dir)
    if resume_path is not None:
        ck = load_checkpoint(resume_path, model, optimizer, scheduler, scaler, map_location=device)
        start_epoch = int(ck["epoch"])
        global_step = int(ck["global_step"])
        best_val = float(ck.get("best_val", best_val))
        if is_main(rank): print(f"RESUMED: {resume_path} epoch={start_epoch} step={global_step}")

    if is_main(rank):
        params = sum(p.numel() for p in unwrap_model(model).parameters() if p.requires_grad)
        print(f"Trainable parameters: {params/1e6:.2f} M")
        print(f"Distributed={distributed}, world_size={world_size}, device={device}, AMP={tcfg.amp_dtype}")
        print("Locked FiLM timing: all 4 legs use CURRENT a_t; FL/FR terrain=t-2; HL/HR terrain=t-3.")

    metrics_path = out_dir / "metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)

    if max_steps_reached(global_step, tcfg.max_steps):
        if is_main(rank):
            save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, scheduler, scaler, start_epoch, global_step, best_val, config_to_dict(mcfg, tcfg))
            print(f"Step cap already reached at step={global_step}; durable latest.pt written.")
        if distributed:
            dist.barrier()
            dist.destroy_process_group()
        return

    for epoch in range(start_epoch, tcfg.epochs):
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        model.train()
        running = {}
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            batch = move_batch_to_device(batch, device)
            sync_now = ((batch_idx + 1) % tcfg.grad_accum_steps == 0) or (batch_idx + 1 == len(train_loader))
            ddp_ctx = nullcontext() if (not distributed or sync_now) else model.no_sync()
            with ddp_ctx:
                with amp_context(tcfg.amp_dtype):
                    horizons = batch["future_action_seq"].shape[1]
                    teacher_mask = None
                    if tcfg.teacher_forcing and horizons > 1:
                        offset = epoch * len(train_ds) + batch_idx * batch["future_action_seq"].shape[0]
                        teacher_mask = teacher_forcing_mask(
                            batch_size=batch["future_action_seq"].shape[0],
                            horizons=horizons,
                            sample_offset=offset,
                            seed=tcfg.seed,
                            device=device,
                        )
                    out = model(
                        batch,
                        history_shift=tcfg.history_shift_samples,
                        teacher_force_mask=teacher_mask,
                    )
                    losses = compute_losses(out, batch, mcfg, tcfg)
                    if teacher_mask is not None:
                        losses["teacher_forcing_fraction"] = teacher_mask[:, 1:].any(-1).float().mean()
                    loss = losses["total"] / tcfg.grad_accum_steps
                scaler.scale(loss).backward()

            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + reduce_scalar(v, world_size)

            if sync_now:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                if is_main(rank) and global_step % tcfg.log_every == 0:
                    denom = tcfg.log_every * tcfg.grad_accum_steps
                    log = {k: v / max(1, denom) for k, v in running.items()}
                    log.update({"epoch": epoch, "step": global_step, "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - t0})
                    print(json.dumps(log))
                    with metrics_path.open("a") as f: f.write(json.dumps(log) + "\n")
                    running = {}; t0 = time.time()

                if is_main(rank) and global_step % tcfg.checkpoint_every_steps == 0:
                    step_path = ckpt_dir / f"step_{global_step:09d}.pt"
                    cfg_dict = config_to_dict(mcfg, tcfg)
                    save_checkpoint(step_path, model, optimizer, scheduler, scaler, epoch, global_step, best_val, cfg_dict)
                    save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, scheduler, scaler, epoch, global_step, best_val, cfg_dict)
                    prune_step_checkpoints(ckpt_dir, tcfg.keep_last_checkpoints)

                if global_step % tcfg.val_every_steps == 0:
                    val = validate(model, val_loader, device, mcfg, tcfg, world_size)
                    if is_main(rank):
                        print("VAL", json.dumps({"step": global_step, **val}))
                        with metrics_path.open("a") as f: f.write(json.dumps({"type": "val", "step": global_step, **val}) + "\n")
                        if val["total"] < best_val:
                            best_val = val["total"]
                            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, scheduler, scaler, epoch, global_step, best_val, config_to_dict(mcfg, tcfg))

                if STOP_REQUESTED:
                    if is_main(rank):
                        save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, scheduler, scaler, epoch, global_step, best_val, config_to_dict(mcfg, tcfg))
                        print("Stop signal received; durable latest.pt written.")
                    if distributed: dist.barrier()
                    return

                if max_steps_reached(global_step, tcfg.max_steps):
                    if is_main(rank):
                        save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, scheduler, scaler, epoch, global_step, best_val, config_to_dict(mcfg, tcfg))
                        print(f"Step cap reached at step={global_step}; durable latest.pt written.")
                    if distributed:
                        dist.barrier()
                        dist.destroy_process_group()
                    return

        # End-of-epoch durable checkpoint. Epoch+1 means resume at next epoch.
        if is_main(rank):
            save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, scheduler, scaler, epoch + 1, global_step, best_val, config_to_dict(mcfg, tcfg))
        if distributed: dist.barrier()

    if is_main(rank): print("Training complete.")
    if distributed: dist.destroy_process_group()


if __name__ == "__main__":
    main()
