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
