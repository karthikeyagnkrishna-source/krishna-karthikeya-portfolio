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
