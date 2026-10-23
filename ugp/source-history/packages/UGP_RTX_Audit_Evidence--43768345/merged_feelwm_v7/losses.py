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
