#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = [
        args.run_root / "phase.txt",
        args.run_root / "checkpoint_selection.json",
        args.run_root / "train.log",
        args.run_root / "metrics.jsonl",
        args.run_root / "resolved_config.json",
        args.run_root / "strict_test" / "strict_test_metrics.json",
        args.run_root / "strict_test" / "strict_test_selection.json",
        args.run_root / "strict_test" / "strict_test_predictions.npz",
        args.run_root / "strict_test" / "strict_test_predictions.csv",
        args.project_root / "configs" / "tartanground_leg_k5_v7f_finetune.yaml",
        args.project_root / "docs" / "V7F_FORCE_FINETUNE_ARCHITECTURE.md",
    ]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing result files: {missing}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    checksums = []
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for path in files:
            if path.is_relative_to(args.run_root):
                arcname = Path("results") / path.relative_to(args.run_root)
            else:
                arcname = Path("source") / path.relative_to(args.project_root)
            archive.write(path, arcname.as_posix())
            checksums.append(f"{sha256(path)}  {arcname.as_posix()}")
        archive.writestr("SHA256SUMS.txt", "\n".join(checksums) + "\n")
    os.replace(temporary, args.output)
    print(f"{sha256(args.output)}  {args.output}", flush=True)


if __name__ == "__main__":
    main()
