#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "scripts"))
from export_shared_p2000_k1 import StrictURDFFK
from tartanground.pilot import load_v7_resamplers
from tartanground.windows import encode_cache_native_action, quarter_windows, split_anchors, union_with_membership


DEFAULT_SHARED = Path("/path/to/ugp/V7F_4080_Run/storage")
DEFAULT_OUTPUT = Path("/path/to/ugp/merged_feelwm_v7_runs/tartanground_k1_k5_leg_compare/prepared")
HOLDOUTS = {
    "ForestEnv/P2001", "GreatMarsh/P2027", "OldTownFall/P2001", "SeasonalForestWinter/P2006",
}


def trajectory_name(environment: str, trajectory: str) -> str:
    return f"{environment}_{trajectory}"


def read_train_list(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty trajectory list: {path}")
    columns = {str(key).strip().lower(): key for key in rows[0]}
    env_key = next((columns[key] for key in ("environment", "env", "world") if key in columns), None)
    traj_key = next((columns[key] for key in ("trajectory", "trajectory_id", "id", "bag") if key in columns), None)
    names = []
    for row in rows:
        if env_key and traj_key:
            env = str(row[env_key]).strip().rstrip("/")
            raw_traj = str(row[traj_key]).strip().rstrip("/")
            if "/" in raw_traj and not env:
                env, raw_traj = raw_traj.rsplit("/", 1)
            traj = raw_traj.split("/")[-1]
            if not env or not traj:
                continue
            names.append(f"{env}/{traj}")
        else:
            joined = "/".join(str(value) for value in row.values())
            match = re.search(r"(ForestEnv|GreatMarsh|OldTownFall|SeasonalForestWinter)[/_-](P\d+)", joined)
            if match:
                names.append(f"{match.group(1)}/{match.group(2)}")
    names = sorted(set(names))
    if len(names) != 28:
        raise ValueError(f"Expected 28 unique training trajectories from {path}, found {len(names)}: {names}")
    if set(names) & HOLDOUTS:
        raise ValueError("The authoritative training list overlaps a strict holdout")
    return names


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + ".tmp.npy")
    np.save(temp, np.asarray(value), allow_pickle=False)
    os.replace(temp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def camera_to_physics(time: np.ndarray, camera_times: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(time, camera_times, side="right") - 1
    if np.any(indices < 0) or np.any(indices >= len(time)):
        raise ValueError("Camera timestamps are outside the physics grid")
    return indices.astype(np.int64)


def load_source(shared: Path, name: str) -> dict[str, np.ndarray]:
    env, traj = name.split("/")
    cache = shared / "cache"
    with np.load(cache / f"{env}_{traj}_physics.npz", allow_pickle=False) as source:
        required = ("P", "B", "U", "L_LEG", "E", "contact", "time")
        data = {key: np.asarray(source[key], np.float32) for key in required}
    with np.load(cache / f"{env}_{traj}_frame_manifest.npz", allow_pickle=False) as source:
        camera_times = np.asarray(source["camera_times"], np.float64)
    if len(camera_times) < 700 or len(data["P"]) != len(data["time"]):
        raise ValueError(f"Invalid cache dimensions for {name}")
    return {**data, "camera_times": camera_times}


def prepare_trajectory(
    shared: Path, output: Path, name: str, data: dict[str, np.ndarray], windows,
    resampler, device: torch.device, resampler_sha: str, fk: StrictURDFFK,
    batch_size: int,
) -> tuple[Path, list[dict]]:
    env, traj = name.split("/")
    cache_dir = output / "cache" / trajectory_name(env, traj)
    cache_dir.mkdir(parents=True, exist_ok=True)
    selected, membership = union_with_membership(windows)
    selected_array = np.asarray(selected, dtype=np.int64)
    cache_meta = cache_dir / "metadata.json"
    source_paths = [
        shared / "cache" / f"{env}_{traj}_{suffix}"
        for suffix in ("physics.npz", "frame_manifest.npz", "v7_dinov3.npy", "v7_convnext.npy", "v7_utonia.npy", "v7_waft.npy")
    ]
    fingerprint = hashlib.sha256(json.dumps([
        [path.name, path.stat().st_size, path.stat().st_mtime_ns] for path in source_paths
    ], sort_keys=True).encode()).hexdigest()
    old = json.loads(cache_meta.read_text()) if cache_meta.exists() else {}
    cached_files = [cache_dir / f"{key}.npy" for key in (
        "P", "B", "U", "L_LEG", "E", "contact", "time", "camera_times",
        "latents", "foot_position", "foot_jacobian", "action_features",
    )]
    if (old.get("fingerprint") == fingerprint and old.get("resampler_sha256") == resampler_sha
            and old.get("selected_frames") == selected and all(path.exists() for path in cached_files)):
        print(f"CACHE REUSE {name} frames={len(selected)}", flush=True)
    else:
        arrays = {
            "dino": np.load(shared / "cache" / f"{env}_{traj}_v7_dinov3.npy", mmap_mode="r", allow_pickle=False),
            "conv": np.load(shared / "cache" / f"{env}_{traj}_v7_convnext.npy", mmap_mode="r", allow_pickle=False),
            "geom": np.load(shared / "cache" / f"{env}_{traj}_v7_utonia.npy", mmap_mode="r", allow_pickle=False),
            "flow": np.load(shared / "cache" / f"{env}_{traj}_v7_waft.npy", mmap_mode="r", allow_pickle=False),
        }
        if min(len(value) for value in arrays.values()) < len(data["camera_times"]):
            raise ValueError(f"Incomplete visual cache for {name}")
        for key, value in data.items():
            if key != "camera_times":
                atomic_npy(cache_dir / f"{key}.npy", value)
        atomic_npy(cache_dir / "camera_times.npy", data["camera_times"])

        latent_tmp = cache_dir / "latents.tmp.npy"
        latents = np.lib.format.open_memmap(
            latent_tmp, mode="w+", dtype=np.float16, shape=(len(data["camera_times"]), 64, 256)
        )
        latents[:] = 0
        resampler.eval()
        for start in range(0, len(selected), batch_size):
            camera_ids = selected_array[start : start + batch_size]
            inputs = [torch.as_tensor(np.asarray(arrays[key][camera_ids], np.float32), device=device) for key in ("dino", "conv", "geom", "flow")]
            with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                encoded = resampler(*inputs)
            values = encoded.float().cpu().numpy().astype(np.float16)
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite post-Perceiver values for {name}")
            latents[camera_ids] = values
            done = start + len(camera_ids)
            if done % 128 == 0 or done == len(selected):
                print(f"PREP {name} latents {done}/{len(selected)}", flush=True)
        latents.flush()
        del latents
        os.replace(latent_tmp, cache_dir / "latents.npy")

        current_indices = camera_to_physics(data["time"], data["camera_times"])
        foot = np.zeros((len(current_indices), 4, 3), np.float32)
        jacobian = np.zeros((len(current_indices), 4, 3, 3), np.float32)
        for camera_index in selected:
            q = data["P"][current_indices[camera_index], :, :3]
            foot[camera_index] = fk.foot_positions(q)
            epsilon = 1e-4
            for joint in range(3):
                shifted = q.copy()
                shifted[:, joint] += epsilon
                jacobian[camera_index, :, :, joint] = (fk.foot_positions(shifted) - foot[camera_index]) / epsilon
        atomic_npy(cache_dir / "foot_position.npy", foot)
        atomic_npy(cache_dir / "foot_jacobian.npy", jacobian)

        command_at_camera = data["L_LEG"][current_indices]
        action_features = encode_cache_native_action(command_at_camera)
        atomic_npy(cache_dir / "action_features.npy", action_features)
        cache_meta.write_text(json.dumps({
            "trajectory": name, "fingerprint": fingerprint, "resampler_sha256": resampler_sha,
            "selected_frames": selected, "window_membership": {str(k): list(v) for k, v in membership.items()},
            "windows": [{
                "start": window.start, "stop_exclusive": window.stop,
                "requested_center": window.requested_center, "clamped_center": window.center,
                "start_time": float(data["camera_times"][window.start]),
                "end_time": float(data["camera_times"][window.stop - 1]),
            } for window in windows],
        }, indent=2))

    current_indices = camera_to_physics(data["time"], data["camera_times"])
    max_horizon = 5
    anchors: dict[int, tuple[int, ...]] = {}
    for window_id, window in enumerate(windows):
        first = window.start + 3
        stop = window.stop - max_horizon
        for camera_index in range(first, stop):
            sequence = current_indices[camera_index - 20] if camera_index >= 20 else -1
            if sequence < 69:
                continue
            anchors.setdefault(camera_index, tuple())
            anchors[camera_index] = tuple(sorted(set(anchors[camera_index] + (window_id,))))
    records = []
    for camera_index, window_ids in sorted(anchors.items()):
        current = current_indices[camera_index : camera_index + max_horizon + 1]
        if len(current) != max_horizon + 1 or np.any(np.diff(current) <= 0):
            continue
        records.append({
            "trajectory_id": name, "cache_dir": str(cache_dir.resolve()),
            "camera_index": camera_index,
            "current_indices": current.tolist(),
            "target_visual_indices": list(range(camera_index + 1, camera_index + max_horizon + 1)),
            "window_membership": list(window_ids),
        })
    if not records:
        raise ValueError(f"No valid five-horizon anchors for {name}")
    return cache_dir, records


def fit_training_stats(output: Path, train_records: list[dict]) -> Path:
    grouped: dict[str, set[int]] = {}
    action_camera: dict[str, set[int]] = {}
    for row in train_records:
        root = row["cache_dir"]
        camera_index = int(row["camera_index"])
        current = row["current_indices"]
        indices = grouped.setdefault(root, set())
        indices.update(range(max(0, int(current[0]) - 69), int(current[-1]) + 1))
        action_camera.setdefault(root, set()).update(range(camera_index, camera_index + 6))

    def collect(field: str, roots_and_indices):
        chunks = []
        for root, indices in roots_and_indices.items():
            chunks.append(np.load(Path(root) / f"{field}.npy", mmap_mode="r", allow_pickle=False)[sorted(indices)])
        return np.concatenate(chunks, axis=0)

    p = collect("P", grouped)
    body = collect("B", grouped)
    control = collect("U", grouped)
    command = collect("L_LEG", grouped)
    error = collect("E", grouped)
    jointbody = np.concatenate([p[:, :, :6].reshape(len(p), 24), body], axis=-1)
    magnitudes = np.linalg.norm(p[:, :, 9:12], axis=-1).reshape(-1)
    positive = magnitudes[magnitudes > 10.0]
    if len(positive) < 100:
        raise ValueError("Insufficient training contacts to calibrate load thresholds")
    thresholds = np.quantile(positive, [0.25, 0.50, 0.75]).astype(np.float32)

    action = np.concatenate([
        np.load(Path(root) / "action_features.npy", mmap_mode="r", allow_pickle=False)[sorted(indices)]
        for root, indices in action_camera.items()
    ], axis=0)
    stats_path = output / "training_stats.npz"
    np.savez(
        stats_path,
        proprio_mean=p.mean((0, 1)).astype(np.float32), proprio_std=(p.std((0, 1)) + 1e-6).astype(np.float32),
        body_mean=body.mean(0).astype(np.float32), body_std=(body.std(0) + 1e-6).astype(np.float32),
        control_mean=control.mean(0).astype(np.float32), control_std=(control.std(0) + 1e-6).astype(np.float32),
        command_mean=command.mean((0, 1)).astype(np.float32), command_std=(command.std((0, 1)) + 1e-6).astype(np.float32),
        error_std=(error.std((0, 1)) + 1e-6).astype(np.float32),
        jointbody_std=(jointbody.std(0) + 1e-6).astype(np.float32),
        tau_std=(p[:, :, 6:9].std((0, 1)) + 1e-6).astype(np.float32),
        force_std=(p[:, :, 9:12].std((0, 1)) + 1e-6).astype(np.float32),
        load_thresholds=thresholds,
        command_delta_mean=np.asarray(action[:, :, 3].mean(), np.float32),
        command_delta_std=np.asarray(action[:, :, 3].std() + 1e-6, np.float32),
    )
    return stats_path


def manifest_row(record: dict, stats_path: Path, horizon: int, split: str) -> dict:
    row = dict(record)
    row["current_indices"] = row["current_indices"][: horizon + 1]
    row["target_visual_indices"] = row["target_visual_indices"][:horizon]
    row["stats_path"] = str(stats_path.resolve())
    row["split"] = split
    return row


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n")
    os.replace(tmp, path)


def interleave_by_trajectory(records: list[dict]) -> list[dict]:
    buckets: dict[str, list[dict]] = {}
    for row in records:
        buckets.setdefault(row["trajectory_id"], []).append(row)
    result = []
    trajectories = sorted(buckets)
    for offset in range(max((len(rows) for rows in buckets.values()), default=0)):
        for trajectory in trajectories:
            if offset < len(buckets[trajectory]):
                result.append(buckets[trajectory][offset])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-root", type=Path, default=DEFAULT_SHARED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    train_csv = args.shared_root / "results/V7_EXPANDED_TRAIN_TRAJECTORIES.csv"
    names = read_train_list(train_csv)
    all_names = names + sorted(HOLDOUTS)
    checkpoint = args.shared_root / "results/V7_PRIOR_SHORT.pt"
    urdf = args.shared_root / ".work/v7_external/anymal_d_simple_description/urdf/anymal.urdf"
    missing = [str(path) for path in (train_csv, checkpoint, urdf) if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    args.output.mkdir(parents=True, exist_ok=True)
    if shutil_free(args.output) < 20 * 1024**3:
        raise RuntimeError("Less than 20 GB free before shared preparation")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    resampler, resampler_sha = load_v7_resamplers(checkpoint, device)
    fk = StrictURDFFK(urdf)
    trajectory_records: dict[str, list[dict]] = {}
    for name in all_names:
        data = load_source(args.shared_root, name)
        windows = quarter_windows(len(data["camera_times"]))
        _, records = prepare_trajectory(
            args.shared_root, args.output, name, data, windows, resampler, device,
            resampler_sha, fk, args.batch_size,
        )
        trajectory_records[name] = records
        print(f"INDEX {name} anchors={len(records)} windows={[w.start for w in windows]}", flush=True)

    split_records: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    for name in names:
        records = trajectory_records[name]
        split = split_anchors((row["camera_index"] for row in records), purge_frames=50, train_fraction=0.8)
        for key in ("train", "dev"):
            selected = set(split[key])
            split_records[key].extend(row for row in records if row["camera_index"] in selected)
        print(f"SPLIT {name} train={len(split['train'])} purge={len(split['purge'])} dev={len(split['dev'])}", flush=True)
    for name in sorted(HOLDOUTS):
        split_records["test"].extend(trajectory_records[name])
    stats_path = fit_training_stats(args.output, split_records["train"])
    metadata = {
        "training_trajectories": names, "holdout_trajectories": sorted(HOLDOUTS),
        "window_frames": 700, "window_center_fractions": [0.25, 0.75],
        "window_semantics": "centers rounded and starts clamped to valid range; overlapping frames cached once with both memberships",
        "purge_camera_frames": 50, "training_stats": str(stats_path.resolve()),
        "resampler_sha256": resampler_sha,
        "counts": {key: len(value) for key, value in split_records.items()},
    }
    (args.output / "preparation_metadata.json").write_text(json.dumps(metadata, indent=2))
    for horizon in (1, 5):
        for split, records in split_records.items():
            if split == "dev":
                records = interleave_by_trajectory(records)
            rows = [manifest_row(row, stats_path, horizon, split) for row in records]
            write_manifest(args.output / f"{split}_k{horizon}.jsonl", rows)
    print(json.dumps(metadata, indent=2), flush=True)


def shutil_free(path: Path) -> int:
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize


if __name__ == "__main__":
    main()
