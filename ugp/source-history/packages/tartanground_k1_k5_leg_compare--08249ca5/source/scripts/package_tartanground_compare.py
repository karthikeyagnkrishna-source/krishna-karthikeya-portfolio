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
