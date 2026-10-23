from __future__ import annotations
from typing import Dict, Any
import torch
import torch.nn.functional as F

from config import ModelConfig, TrainConfig


def cosine_latent_loss(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = F.normalize(x, dim=-1, eps=eps)
    y = F.normalize(y, dim=-1, eps=eps)
    return 1.0 - (x * y).sum(-1).mean()


def normalized_smooth_l1(pred: torch.Tensor, target: torch.Tensor, scale: torch.Tensor, beta: float = 0.35) -> torch.Tensor:
    while scale.dim() < pred.dim():
        scale = scale.unsqueeze(1)
    err = (pred - target) / (scale + 1e-6)
    return F.smooth_l1_loss(err, torch.zeros_like(err), beta=beta)


def compute_losses(out: Dict[str, Any], batch: Dict[str, torch.Tensor], mcfg: ModelConfig, tcfg: TrainConfig) -> Dict[str, torch.Tensor]:
    steps = out["steps"]
    K = len(steps)
    n = mcfg.tokens_per_modality

    Ld = Lc = Lg = Lf = 0.0
    Lcontact = Lload = Lstate = Ltau = LbaseF = LbaseAux = 0.0

    for k, s in enumerate(steps):
        zt = batch["target_z_seq"][:, k]
        zd, zc, zg, zf = zt[:, 0*n:1*n], zt[:, 1*n:2*n], zt[:, 2*n:3*n], zt[:, 3*n:4*n]
        Ld = Ld + F.mse_loss(s["dino"], zd) + 0.1 * cosine_latent_loss(s["dino"], zd)
        Lc = Lc + F.mse_loss(s["conv"], zc) + 0.1 * cosine_latent_loss(s["conv"], zc)
        Lg = Lg + F.smooth_l1_loss(s["geom"], zg) + 0.1 * cosine_latent_loss(s["geom"], zg)
        Lf = Lf + F.smooth_l1_loss(s["flow"], zf) + 0.1 * cosine_latent_loss(s["flow"], zf)

        Lcontact = Lcontact + F.binary_cross_entropy_with_logits(s["contact_logits"], batch["target_contact_seq"][:, k])
        Lload = Lload + F.cross_entropy(s["regime_logits"].reshape(-1, mcfg.num_load_regimes), batch["target_load_class_seq"][:, k].reshape(-1))
        Lstate = Lstate + normalized_smooth_l1(s["jointbody_pred"], batch["target_jointbody_seq"][:, k], batch["jointbody_std"])
        Ltau = Ltau + normalized_smooth_l1(s["tau_pred"], batch["target_tau_seq"][:, k], batch["tau_std"])
        LbaseF = LbaseF + normalized_smooth_l1(s["base_force"], batch["target_force_seq"][:, k], batch["force_std"])

        target_force = batch["target_force_seq"][:, k]
        target_magnitude = torch.linalg.norm(target_force, dim=-1)
        auxiliary_magnitude = F.smooth_l1_loss(
            s["force_aux_logmag"], torch.log1p(target_magnitude), beta=.25
        )
        target_direction = F.normalize(target_force, dim=-1, eps=mcfg.eps)
        direction_cosine = (s["force_aux_dir"] * target_direction).sum(-1)
        meaningful_contact = (target_magnitude >= tcfg.force_contact_threshold).float()
        auxiliary_direction = ((1.0 - direction_cosine) * meaningful_contact).sum() / (meaningful_contact.sum() + 1e-6)
        LbaseAux = LbaseAux + tcfg.v7_w_logmag * auxiliary_magnitude + tcfg.v7_w_dir * auxiliary_direction

    # Normalize step-wise auxiliary objectives by rollout length.
    Ld, Lc, Lg, Lf = [x / K for x in (Ld, Lc, Lg, Lf)]
    Lcontact, Lload, Lstate, Ltau, LbaseF, LbaseAux = [
        x / K for x in (Lcontact, Lload, Lstate, Ltau, LbaseF, LbaseAux)
    ]

    # V7-F force specialist loss, faithful to the specialist notebook.
    spec = out["v7f"]
    pred = spec["force"]
    gt = batch["target_force_seq"]
    phase = batch["target_phase_seq"]
    pstd = batch["force_std"][:, None, None, :]

    gtmag = torch.linalg.norm(gt, dim=-1)
    predmag = torch.linalg.norm(pred, dim=-1)
    # Large-load emphasis without unbounded domination.
    scale = gtmag.detach().median().clamp_min(1.0)
    phase_boost = ((phase == 1) | (phase == 3)).float()
    w = (1.0 + 1.25 * (gtmag / scale).clamp(0, 2.0) + 0.50 * phase_boost).detach()

    e = (pred - gt) / (pstd + 1e-6)
    lvec = (F.smooth_l1_loss(e, torch.zeros_like(e), reduction="none", beta=.35).mean(-1) * w).mean()
    lmag = F.smooth_l1_loss(torch.log1p(predmag), torch.log1p(gtmag), beta=.20)

    gtdir = F.normalize(gt, dim=-1, eps=mcfg.eps)
    pdir = F.normalize(pred, dim=-1, eps=mcfg.eps)
    contact = (gtmag >= tcfg.force_contact_threshold).float()
    cos = (gtdir * pdir).sum(-1)
    ldir = ((1.0 - cos) * contact).sum() / (contact.sum() + 1e-6)
    lphase = F.cross_entropy(spec["phase_logits"].reshape(-1, mcfg.num_contact_phases), phase.reshape(-1))

    if K > 1:
        dp = (pred[:, 1:] - pred[:, :-1]) / (pstd + 1e-6)
        dg = (gt[:, 1:] - gt[:, :-1]) / (pstd + 1e-6)
        ltemp = F.smooth_l1_loss(dp, dg, beta=.35)
        ip = (pred * tcfg.rollout_dt).sum(1)
        ig = (gt * tcfg.rollout_dt).sum(1)
        limp = F.smooth_l1_loss((ip - ig) / (batch["force_std"][:, None, :] + 1e-6), torch.zeros_like(ip), beta=.35)
    else:
        ltemp = pred.sum() * 0.0
        limp = pred.sum() * 0.0

    lanchor = (spec["corr_norm"] ** 2).mean()
    mean_gate = spec["expert_w"].mean(dim=(0, 1, 2))
    uniform = torch.full_like(mean_gate, 1.0 / len(mean_gate))
    lgate = ((mean_gate - uniform) ** 2).mean()

    Lv7f = (tcfg.v7f_w_vec * lvec + tcfg.v7f_w_mag * lmag + tcfg.v7f_w_dir * ldir +
            tcfg.v7f_w_phase * lphase + tcfg.v7f_w_temp * ltemp + tcfg.v7f_w_impulse * limp +
            tcfg.v7f_w_anchor * lanchor + tcfg.v7f_w_gate * lgate)

    total = (tcfg.w_dino * Ld + tcfg.w_conv * Lc + tcfg.w_geom * Lg + tcfg.w_flow * Lf +
             tcfg.w_contact * Lcontact + tcfg.w_load * Lload + tcfg.w_jointbody * Lstate +
             tcfg.w_tau * Ltau + tcfg.w_base_force * (LbaseF + LbaseAux) + tcfg.w_v7f_force * Lv7f)

    return {
        "total": total,
        "dino": Ld, "conv": Lc, "geom": Lg, "flow": Lf,
        "contact": Lcontact, "load": Lload, "jointbody": Lstate,
        "tau": Ltau, "base_force": LbaseF, "base_force_aux": LbaseAux, "v7f": Lv7f,
        "v7f_vec": lvec, "v7f_mag": lmag, "v7f_dir": ldir,
        "v7f_phase": lphase, "v7f_temp": ltemp, "v7f_impulse": limp,
    }
