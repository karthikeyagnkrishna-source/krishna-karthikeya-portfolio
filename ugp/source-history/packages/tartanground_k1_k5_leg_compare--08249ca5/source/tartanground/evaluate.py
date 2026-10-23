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
