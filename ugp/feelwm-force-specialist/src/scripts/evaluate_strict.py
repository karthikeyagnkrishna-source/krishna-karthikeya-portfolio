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
