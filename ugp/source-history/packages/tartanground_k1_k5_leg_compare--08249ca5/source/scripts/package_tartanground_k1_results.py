from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path


DEFAULT_PROJECT = Path("/path/to/ugp/merged_feelwm_v7")
DEFAULT_RUN = Path(
    "/path/to/ugp/merged_feelwm_v7_runs/"
    "tartanground_pilot"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_files(project: Path) -> list[Path]:
    files: set[Path] = set()
    for pattern in ("*.py", "*.md", "*.txt", "*.sh"):
        files.update(path for path in project.glob(pattern) if path.is_file())
    allowed = {".py", ".yaml", ".yml", ".sh", ".md", ".txt", ".json"}
    for directory in ("configs", "scripts", "tartanground", "tests"):
        root = project / directory
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in allowed and "__pycache__" not in path.parts:
                files.add(path)
    return sorted(files)


def add_file(archive: zipfile.ZipFile, path: Path, arcname: str) -> None:
    compression = zipfile.ZIP_STORED if path.suffix.lower() in {".pt", ".npz"} else zipfile.ZIP_DEFLATED
    archive.write(path, arcname, compress_type=compression)


def main() -> None:
    parser = argparse.ArgumentParser(description="Package the verified TartanGround P2000 K=1 pilot results")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    project = args.project.resolve()
    run = args.run.resolve()
    output = args.output or run / "exports" / "tartanground_p2000_k1_results_step200.zip"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    paths = {
        "checkpoint/latest.pt": run / "k1_train/checkpoints/latest.pt",
        "run/resolved_config.json": run / "k1_train/resolved_config.json",
        "run/metrics.jsonl": run / "k1_train/metrics.jsonl",
        "run/k1_pipeline.log": run / "logs/k1_pipeline.log",
        "data/metadata.json": run / "prepared_k1/metadata.json",
        "data/training_stats.npz": run / "prepared_k1/training_stats.npz",
        "data/train_manifest.jsonl": run / "prepared_k1/train_manifest.jsonl",
        "data/val_manifest.jsonl": run / "prepared_k1/val_manifest.jsonl",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing package inputs: {missing}")

    metadata = json.loads(paths["data/metadata.json"].read_text())
    records = [json.loads(line) for line in paths["run/metrics.jsonl"].read_text().splitlines() if line.strip()]
    validations = [record for record in records if record.get("type") == "val"]
    if [record.get("step") for record in validations] != [50, 100, 150, 200]:
        raise RuntimeError("Expected validation records at steps 50, 100, 150, and 200")

    test = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=project,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if test.returncode != 0:
        raise RuntimeError(f"Unit verification failed:\n{test.stdout}")

    columns = [
        "step", "total", "dino", "conv", "geom", "flow", "contact", "load",
        "jointbody", "tau", "base_force", "base_force_aux", "v7f", "v7f_vec",
        "v7f_mag", "v7f_dir", "v7f_phase", "v7f_temp", "v7f_impulse",
    ]
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(validations)

    checkpoint_hash = sha256(paths["checkpoint/latest.pt"])
    summary = {
        "experiment": "TartanGround ForestEnv/P2000 K=1 pilot",
        "status": "completed",
        "global_step": 200,
        "effective_epochs_approx": 1.91,
        "trajectory_count": 1,
        "trajectory": metadata["trajectory"],
        "frames": metadata["max_frames"],
        "trajectory_duration_seconds": 119.89999866485596,
        "train_samples": metadata["train_samples"],
        "validation_manifest_samples": metadata["validation_samples"],
        "validation_samples_per_event": 100,
        "purge_seconds": metadata["purge_seconds"],
        "pipeline_wall_seconds": 592,
        "checkpoint_sha256": checkpoint_hash,
        "validation": validations,
        "limitations": [
            "Validation uses a later chronological segment of P2000, not an unseen trajectory.",
            "P2001 was unavailable and was not tested.",
            "This pilot exports loss components, not physical-unit RMSE or classification accuracy/F1.",
        ],
    }

    readme = f"""# TartanGround P2000 K=1 pilot results

This package contains the verified result of the credit-capped merged Feel-WM V7 pilot.

## Run summary

- Status: completed at global optimizer step 200
- Effective training: approximately 1.91 epochs
- Data: one ForestEnv/P2000 trajectory, 1,200 frames spanning 119.9 seconds
- Split: 834 training samples and 238 validation-manifest samples, with a 5-second purge gap
- Compute: two RTX 4080 GPUs, DDP, BF16, TF32, effective batch approximately 8
- Trainable parameters: 110.53 million
- End-to-end successful pipeline time: approximately 592 seconds
- Final/best validation total: {validations[-1]['total']:.8f}
- Final checkpoint SHA-256: {checkpoint_hash}

## Evaluation scope

Validation ran at steps 50, 100, 150, and 200. Each event covered 100 distributed
validation samples because the evaluator was capped at 50 batches per GPU. The raw
records are in `run/metrics.jsonl`; compact copies are in `run/validation_summary.csv`
and `run/validation_summary.json`.

This is not an unseen-trajectory test. Validation came from a later chronological
segment of P2000 after a 5-second purge. P2001 was unavailable and was not evaluated.
No physical-unit RMSE, accuracy, precision, recall, or F1 values were produced.

## Checkpoint

`checkpoint/latest.pt` is a resumable PyTorch checkpoint containing model, optimizer,
scheduler, RNG, configuration, epoch index, global step, and best-validation state.
Use only with the included trusted source. Loading the full resumable checkpoint uses
PyTorch pickle semantics.

The raw dataset, prepared sample shards, shared frozen V7 resampler checkpoint, and
redundant intermediate checkpoints are intentionally excluded to keep this package
small. Their provenance and the frozen-resampler SHA-256 are recorded in metadata.

## Verification

The source unit suite was rerun while packaging and passed. See
`verification/tests.txt`. `SHA256SUMS.txt` records hashes for packaged files.
"""

    source = source_files(project)
    checksums: dict[str, str] = {"checkpoint/latest.pt": checkpoint_hash}
    for arcname, path in paths.items():
        if arcname not in checksums:
            checksums[arcname] = sha256(path)
    for path in source:
        arcname = f"source/{path.relative_to(project).as_posix()}"
        checksums[arcname] = sha256(path)

    generated = {
        "README_RESULTS.md": readme,
        "run/validation_summary.csv": csv_buffer.getvalue(),
        "run/validation_summary.json": json.dumps(summary, indent=2) + "\n",
        "verification/tests.txt": test.stdout,
    }
    for arcname, value in generated.items():
        checksums[arcname] = hashlib.sha256(value.encode()).hexdigest()
    checksum_text = "".join(f"{digest}  {arcname}\n" for arcname, digest in sorted(checksums.items()))

    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
            for arcname, path in paths.items():
                add_file(archive, path, arcname)
            for path in source:
                add_file(archive, path, f"source/{path.relative_to(project).as_posix()}")
            for arcname, value in generated.items():
                archive.writestr(arcname, value, compress_type=zipfile.ZIP_DEFLATED)
            archive.writestr("SHA256SUMS.txt", checksum_text, compress_type=zipfile.ZIP_DEFLATED)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP integrity check failed at {bad}")

    print(json.dumps({
        "output": str(output),
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "files": len(checksums) + 1,
        "unit_tests": "passed",
    }, indent=2))


if __name__ == "__main__":
    main()
