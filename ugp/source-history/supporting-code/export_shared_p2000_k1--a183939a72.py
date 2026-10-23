#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import validate_sample
from tartanground.pilot import (
    build_body_action,
    build_leg_action,
    load_v7_resamplers,
    phase_vector,
)


DEFAULT_SHARED = Path(
    "/path/to/ugp/V7F_4080_Run/storage"
)
DEFAULT_OUTPUT = Path(
    "/path/to/ugp/merged_feelwm_v7_runs/"
    "tartanground_pilot/prepared_k1"
)


def causal_hold(t_src: np.ndarray, x_src: np.ndarray, t_target: np.ndarray) -> np.ndarray:
    index = np.searchsorted(t_src, t_target, side="right") - 1
    if np.any(index < 0):
        raise RuntimeError("Target precedes the first commanded-position sample")
    return x_src[index]


def stamp_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def extract_commanded_positions(
    bag_path: Path,
    grid_original: np.ndarray,
    actuator_for_joint: np.ndarray,
    cache_path: Path,
) -> np.ndarray:
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        if cached.shape == (len(grid_original), 4, 3) and np.isfinite(cached).all():
            return cached.astype(np.float32)

    from rosbags.highlevel import AnyReader

    physical = {
        "/state_estimator/joint_states",
        "/state_estimator/contact_force_lf_foot",
        "/state_estimator/contact_force_rf_foot",
        "/state_estimator/contact_force_lh_foot",
        "/state_estimator/contact_force_rh_foot",
        "/state_estimator/twist",
    }
    with AnyReader([bag_path]) as reader:
        selected = [c for c in reader.connections if c.topic in physical]
        if not selected:
            raise RuntimeError("No physical reference topic found in ROS bag")
        _, first_ns, _ = next(reader.messages(connections=selected))
    t0 = first_ns * 1e-9

    times: list[float] = []
    positions: list[list[float]] = []
    topic = "/state_estimator/actuator_readings"
    with AnyReader([bag_path]) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            raise RuntimeError(f"Required topic missing: {topic}")
        for connection, _, raw in reader.messages(connections=connections):
            message = reader.deserialize(raw, connection.msgtype)
            readings = list(message.readings)
            if len(readings) != 12:
                continue
            stamps = [stamp_seconds(r.commanded.header.stamp) for r in readings]
            values = [float(r.commanded.position) for r in readings]
            times.append(float(np.median(stamps)) - t0)
            positions.append(values)
    if not times:
        raise RuntimeError("No valid commanded-position records found")
    cmd_t = np.asarray(times, np.float64)
    cmd_q = np.asarray(positions, np.float32)
    order = np.argsort(cmd_t)
    cmd_t, cmd_q = cmd_t[order], cmd_q[order]
    aligned = causal_hold(cmd_t, cmd_q, grid_original)
    aligned = aligned[:, np.asarray(actuator_for_joint, np.int64)].reshape(-1, 4, 3)
    if not np.isfinite(aligned).all():
        raise RuntimeError("Non-finite commanded joint positions")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp.npy")
    np.save(temporary, aligned.astype(np.float32), allow_pickle=False)
    os.replace(temporary, cache_path)
    return aligned.astype(np.float32)


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in rpy]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], np.float64)
    return rz @ ry @ rx


def axis_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = np.asarray(axis, np.float64) / (np.linalg.norm(axis) + 1e-12)
    c, s, one_minus = np.cos(angle), np.sin(angle), 1.0 - np.cos(angle)
    return np.array(
        [
            [c + x * x * one_minus, x * y * one_minus - z * s, x * z * one_minus + y * s],
            [y * x * one_minus + z * s, c + y * y * one_minus, y * z * one_minus - x * s],
            [z * x * one_minus - y * s, z * y * one_minus + x * s, c + z * z * one_minus],
        ],
        np.float64,
    )


class StrictURDFFK:
    def __init__(self, urdf_path: Path):
        root = ET.parse(urdf_path).getroot()
        links = {node.attrib["name"] for node in root.findall("link")}
        children: dict[str, dict] = {}
        all_children = set()
        for element in root.findall("joint"):
            origin = element.find("origin")
            axis_element = element.find("axis")
            joint = {
                "name": element.attrib["name"],
                "type": element.attrib.get("type", "fixed"),
                "parent": element.find("parent").attrib["link"],
                "child": element.find("child").attrib["link"],
                "xyz": np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3),
                "rpy": np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3),
                "axis": np.fromstring(axis_element.attrib.get("xyz", "1 0 0"), sep=" ") if axis_element is not None else np.array([1.0, 0.0, 0.0]),
            }
            children[joint["child"]] = joint
            all_children.add(joint["child"])
        roots = list(links - all_children)
        if len(roots) != 1:
            raise RuntimeError(f"URDF must have one root link, got {roots}")
        self.root = roots[0]
        self.prefixes = ("LF", "RF", "LH", "RH")
        self.chains = []
        for prefix in self.prefixes:
            candidates = [link for link in links if link.upper().startswith(prefix + "_") and "FOOT" in link.upper()]
            valid = []
            for link in candidates:
                chain = []
                current = link
                while current != self.root and current in children:
                    chain.append(children[current])
                    current = children[current]["parent"]
                chain.reverse()
                names = {joint["name"] for joint in chain}
                if all(f"{prefix}_{name}" in names for name in ("HAA", "HFE", "KFE")):
                    valid.append(chain)
            if not valid:
                raise RuntimeError(f"No verified {prefix} foot chain in {urdf_path}")
            self.chains.append(max(valid, key=len))

    @staticmethod
    def transform(rotation=None, translation=None) -> np.ndarray:
        value = np.eye(4, dtype=np.float64)
        if rotation is not None:
            value[:3, :3] = rotation
        if translation is not None:
            value[:3, 3] = translation
        return value

    def foot_positions(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, np.float64)
        qmap = {
            f"{prefix}_{joint}": float(q[leg, index])
            for leg, prefix in enumerate(self.prefixes)
            for index, joint in enumerate(("HAA", "HFE", "KFE"))
        }
        result = []
        for chain in self.chains:
            transform = np.eye(4, dtype=np.float64)
            for joint in chain:
                transform = transform @ self.transform(rpy_matrix(joint["rpy"]), joint["xyz"])
                if joint["type"] in ("revolute", "continuous"):
                    transform = transform @ self.transform(axis_matrix(joint["axis"], qmap.get(joint["name"], 0.0)))
                elif joint["type"] == "prismatic":
                    transform = transform @ self.transform(translation=joint["axis"] * qmap.get(joint["name"], 0.0))
                elif joint["type"] != "fixed":
                    raise RuntimeError(f"Unsupported URDF joint type: {joint['type']}")
            result.append(transform[:3, 3])
        result = np.asarray(result, np.float32)
        if not np.isfinite(result).all() or np.max(np.linalg.norm(result, axis=1)) > 2.0:
            raise RuntimeError("Implausible ANYmal foot positions")
        return result


def atomic_torch_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def export_latents(cache_root: Path, checkpoint: Path, output: Path, max_frames: int) -> tuple[torch.Tensor, str]:
    latent_path = output / "post_perceiver_latents.pt"
    if latent_path.exists():
        cached = torch.load(latent_path, map_location="cpu", weights_only=True)
        if cached["latents"].shape == (max_frames, 64, 256):
            return cached["latents"], str(cached["checkpoint_sha256"])

    arrays = {
        "dino": np.load(cache_root / "ForestEnv_P2000_v7_dinov3.npy", mmap_mode="r", allow_pickle=False),
        "conv": np.load(cache_root / "ForestEnv_P2000_v7_convnext.npy", mmap_mode="r", allow_pickle=False),
        "geom": np.load(cache_root / "ForestEnv_P2000_v7_utonia.npy", mmap_mode="r", allow_pickle=False),
        "flow": np.load(cache_root / "ForestEnv_P2000_v7_waft.npy", mmap_mode="r", allow_pickle=False),
    }
    for name, array in arrays.items():
        if len(array) < max_frames or not np.isfinite(np.asarray(array[[0, max_frames - 1]], np.float32)).all():
            raise RuntimeError(f"Invalid {name} cache")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, digest = load_v7_resamplers(checkpoint, device)
    result = torch.empty((max_frames, 64, 256), dtype=torch.float16)
    for start in range(0, max_frames, 8):
        stop = min(max_frames, start + 8)
        tensors = [
            torch.as_tensor(np.asarray(arrays[name][start:stop], np.float32), device=device)
            for name in ("dino", "conv", "geom", "flow")
        ]
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            value = model(*tensors)
        result[start:stop] = value.float().cpu().half()
        if stop % 100 == 0 or stop == max_frames:
            print(f"LATENTS {stop}/{max_frames}", flush=True)
    if not torch.isfinite(result).all():
        raise RuntimeError("Non-finite post-Perceiver latents")
    atomic_torch_save({"latents": result, "checkpoint_sha256": digest}, latent_path)
    return result, digest


def write_split(samples: list[dict], split: str, output: Path, digest: str, shard_size: int = 16) -> Path:
    split_dir = output / split
    split_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for start in range(0, len(samples), shard_size):
        block = samples[start : start + shard_size]
        shard = split_dir / f"shard_{start // shard_size:04d}.pt"
        atomic_torch_save(
            {"samples": block, "metadata": {"split": split, "resampler_sha256": digest}},
            shard,
        )
        rows.extend({"path": str(shard.resolve()), "index": index} for index in range(len(block)))
    manifest = output / f"{split}_manifest.jsonl"
    temporary = manifest.with_suffix(".tmp")
    temporary.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    os.replace(temporary, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export capped ForestEnv/P2000 K=1 shards from validated shared V7 caches")
    parser.add_argument("--shared-root", type=Path, default=DEFAULT_SHARED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-frames", type=int, default=1200)
    args = parser.parse_args()
    if not 10 <= args.max_frames <= 1200:
        raise ValueError("--max-frames must be between 10 and 1200")
    args.output.mkdir(parents=True, exist_ok=True)
    if os.statvfs(args.output).f_bavail * os.statvfs(args.output).f_frsize < 50 * 1024**3:
        raise RuntimeError("Less than 50 GB free before export")

    cache_root = args.shared_root / "cache"
    raw_base = args.shared_root / "raw_tartanground/ForestEnv/Data_anymal/P2000"
    checkpoint = args.shared_root / "results/V7_PRIOR_SHORT.pt"
    urdf = args.shared_root / ".work/v7_external/anymal_d_simple_description/urdf/anymal.urdf"
    required = [cache_root, raw_base / "rosbags/P2000.bag", checkpoint, urdf]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing shared pilot inputs: {missing}")

    physics_file = np.load(cache_root / "ForestEnv_P2000_physics.npz", allow_pickle=False)
    data = {name: physics_file[name] for name in physics_file.files}
    manifest_file = np.load(cache_root / "ForestEnv_P2000_frame_manifest.npz", allow_pickle=False)
    manifest = {name: manifest_file[name] for name in manifest_file.files}
    frame_count = min(args.max_frames, len(manifest["camera_times"]))
    latents, digest = export_latents(cache_root, checkpoint, args.output, frame_count)

    commanded_q = extract_commanded_positions(
        raw_base / "rosbags/P2000.bag",
        data["grid_original"],
        data["actuator_for_joint"],
        args.output / "commanded_positions.npy",
    )
    fk = StrictURDFFK(urdf)
    time = np.asarray(data["time"], np.float64)
    camera_times = np.asarray(manifest["camera_times"][:frame_count], np.float64)
    anchors = []
    for visual_index in range(3, frame_count - 1):
        current = int(np.searchsorted(time, camera_times[visual_index], side="right") - 1)
        future = int(np.searchsorted(time, camera_times[visual_index + 1], side="right") - 1)
        if current - 20 - 49 >= 0 and future > current and future < len(time):
            anchors.append((visual_index, current, future))
    if len(anchors) < 100:
        raise RuntimeError(f"Too few eligible anchors: {len(anchors)}")

    train_count = int(0.70 * len(anchors))
    validation_count = int(0.20 * len(anchors))
    train_anchors = anchors[:train_count]
    purge_end = time[train_anchors[-1][1]] + 5.0
    validation_pool = [row for row in anchors[train_count:] if time[row[1]] >= purge_end]
    validation_anchors = validation_pool[:validation_count]
    if not validation_anchors:
        raise RuntimeError("Five-second purge left no validation anchors")
    train_end = time[train_anchors[-1][1]]
    train_mask = time <= train_end

    P = np.asarray(data["P"], np.float32)
    B = np.asarray(data["B"], np.float32)
    U = np.asarray(data["U"], np.float32)
    L = np.asarray(data["L_LEG"], np.float32)
    E = np.asarray(data["E"], np.float32)
    contact = np.asarray(data["contact"], np.float32)
    eps = 1e-6
    p_mean, p_std = P[train_mask].mean((0, 1)), P[train_mask].std((0, 1)) + eps
    b_mean, b_std = B[train_mask].mean(0), B[train_mask].std(0) + eps
    u_mean, u_std = U[train_mask].mean(0), U[train_mask].std(0) + eps
    Pn = (P - p_mean) / p_std
    Bn = (B - b_mean) / b_std
    Un = (U - u_mean) / u_std
    Ln = (L - p_mean[6:9]) / p_std[6:9]
    En = E / p_std[6:9]
    leg_history = np.concatenate(
        [Pn, np.repeat(Un[:, None, :], 4, axis=1), Ln, En], axis=-1
    ).astype(np.float32)
    body_history = np.concatenate([Bn, Un], axis=-1).astype(np.float32)
    jointbody = np.concatenate([P[:, :, :6].reshape(len(P), 24), B], axis=-1)
    jointbody_std = jointbody[train_mask].std(0).astype(np.float32) + eps
    tau_std = P[train_mask, :, 6:9].std((0, 1)).astype(np.float32) + eps
    force_std = P[train_mask, :, 9:12].std((0, 1)).astype(np.float32) + eps
    train_magnitude = np.linalg.norm(P[train_mask, :, 9:12], axis=-1).reshape(-1)
    positive = train_magnitude[train_magnitude > 10.0]
    thresholds = np.quantile(positive, [0.25, 0.50, 0.75]).astype(np.float32)

    def regime(force: np.ndarray) -> np.ndarray:
        return np.digitize(np.linalg.norm(force, axis=-1), thresholds).astype(np.int64)

    def history(end: int) -> np.ndarray:
        return leg_history[end - 49 : end + 1].transpose(1, 0, 2).copy()

    def make_sample(row: tuple[int, int, int]) -> dict:
        visual_index, current, future = row
        interval_u = Un[current:future]
        interval_l = Ln[current:future]
        if len(interval_u) == 0:
            interval_u, interval_l = Un[current : current + 1], Ln[current : current + 1]
        future_action = np.concatenate(
            [interval_u[-1], interval_l[-1].reshape(-1), interval_u.mean(0), interval_l.mean(0).reshape(-1)]
        ).astype(np.float32)
        current_p = P[current]
        current_b = B[current]
        baseline_q = current_p[:, :3] + 0.1 * current_p[:, 3:6]
        baseline_jointbody = np.concatenate(
            [np.concatenate([baseline_q, current_p[:, 3:6]], axis=-1).reshape(-1), current_b]
        ).astype(np.float32)
        current_regime = regime(current_p[:, 9:12])
        one_hot = np.eye(4, dtype=np.float32)[current_regime]
        physics = np.concatenate(
            [
                ((baseline_q - p_mean[:3]) / p_std[:3]).reshape(-1),
                ((current_p[:, 6:9] - p_mean[6:9]) / p_std[6:9]).reshape(-1),
                (current_b[:3] - b_mean[:3]) / b_std[:3],
                np.linalg.norm(current_p[:, 9:12], axis=-1) / max(float(thresholds[-1]), 1.0),
                one_hot.reshape(-1),
            ]
        ).astype(np.float32)
        foot_position = fk.foot_positions(current_p[:, :3])
        planned_delta = np.tile((U[current, :3] * 0.30)[None, :], (4, 1)).astype(np.float32)
        query = np.concatenate(
            [Pn[current], np.tile(Un[current][None, :], (4, 1)), np.concatenate([foot_position, planned_delta], axis=-1)],
            axis=-1,
        ).astype(np.float32)
        sample = {
            "z_window": latents[visual_index - 3 : visual_index + 1].clone(),
            "leg_hist_t": torch.from_numpy(history(current)),
            "leg_hist_tm1": torch.from_numpy(history(current - 10)),
            "leg_hist_tm2": torch.from_numpy(history(current - 20)),
            "body_hist_t": torch.from_numpy(body_history[current - 49 : current + 1].copy()),
            "body_action_seq": torch.from_numpy(build_body_action(U[current])[None]),
            "leg_action_seq": torch.from_numpy(build_leg_action(commanded_q[current], commanded_q[future], L[current])[None]),
            "leg_query_features_seq": torch.from_numpy(query[None]),
            "future_action_seq": torch.from_numpy(future_action[None]),
            "physics_seq": torch.from_numpy(physics[None]),
            "horizon_seq": torch.tensor([[0.1]], dtype=torch.float32),
            "lowlevel_cmd_seq": torch.from_numpy(Ln[current][None].copy()),
            "err_seq": torch.from_numpy(En[current][None].copy()),
            "jointbody_baseline_seq": torch.from_numpy(baseline_jointbody[None]),
            "jointbody_std": torch.from_numpy(jointbody_std.copy()),
            "tau_baseline_seq": torch.from_numpy(current_p[None, :, 6:9].copy()),
            "tau_std": torch.from_numpy(tau_std.copy()),
            "force_baseline_seq": torch.from_numpy(current_p[None, :, 9:12].copy()),
            "force_std": torch.from_numpy(force_std.copy()),
            "target_z_seq": latents[visual_index + 1 : visual_index + 2].clone(),
            "target_contact_seq": torch.from_numpy(contact[future][None].copy()),
            "target_load_class_seq": torch.from_numpy(regime(P[future, :, 9:12])[None]),
            "target_jointbody_seq": torch.from_numpy(jointbody[future][None].copy()),
            "target_tau_seq": torch.from_numpy(P[future : future + 1, :, 6:9].copy()),
            "target_force_seq": torch.from_numpy(P[future : future + 1, :, 9:12].copy()),
            "target_phase_seq": torch.from_numpy(phase_vector(contact[current], contact[future])[None]),
        }
        validate_sample(sample)
        for key, value in sample.items():
            if torch.is_floating_point(value) and not torch.isfinite(value).all():
                raise RuntimeError(f"Non-finite sample tensor: {key}")
        return sample

    train_samples = [make_sample(row) for row in train_anchors]
    validation_samples = [make_sample(row) for row in validation_anchors]
    train_manifest = write_split(train_samples, "train", args.output, digest)
    validation_manifest = write_split(validation_samples, "val", args.output, digest)
    metadata = {
        "environment": "ForestEnv",
        "trajectory": "P2000",
        "horizon": 1,
        "max_frames": frame_count,
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "purge_seconds": 5.0,
        "resampler_checkpoint": str(checkpoint),
        "resampler_sha256": digest,
        "load_thresholds": thresholds.tolist(),
        "train_manifest": str(train_manifest),
        "val_manifest": str(validation_manifest),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    np.savez(
        args.output / "training_stats.npz",
        p_mean=p_mean,
        p_std=p_std,
        b_mean=b_mean,
        b_std=b_std,
        u_mean=u_mean,
        u_std=u_std,
        jointbody_std=jointbody_std,
        tau_std=tau_std,
        force_std=force_std,
        load_thresholds=thresholds,
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
