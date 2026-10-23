#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
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
    rows = []
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
        rows.append(dataset.rows[len(rows)]["trajectory_id"])
    result = {key: np.stack(values, axis=0) for key, values in fields.items()}
    return rows, result


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    validate_strict_split(args.train_manifest, args.dev_manifest, args.test_manifest)
    mcfg, tcfg, _ = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = MergedFeelWMV7(mcfg).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    dataset = IndexedTartanGroundDataset(args.test_manifest)
    trajectory_ids, prediction = collect(model, dataset, device, tcfg.amp_dtype)
    metrics = evaluate(prediction, trajectory_ids)
    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "strict_test_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=False))
    pred_path = args.output / "strict_test_predictions.npz"
    np.savez_compressed(
        pred_path, trajectory_id=np.asarray(trajectory_ids),
        **{f"{key}_{kind}": value for key in ("q", "tau", "force", "contact", "load")
           for kind, value in (("pred", prediction[f"{key}_pred"]), ("true", prediction[f"{key}_true"]))},
    )
    print(json.dumps({"samples": len(trajectory_ids), "metrics": metrics, "predictions": str(pred_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
