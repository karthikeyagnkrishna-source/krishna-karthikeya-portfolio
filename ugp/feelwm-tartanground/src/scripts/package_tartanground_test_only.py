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
