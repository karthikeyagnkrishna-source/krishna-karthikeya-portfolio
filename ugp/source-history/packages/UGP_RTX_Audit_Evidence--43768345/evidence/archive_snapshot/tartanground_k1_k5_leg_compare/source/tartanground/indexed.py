from __future__ import annotations

import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from data import validate_sample
from tartanground.pilot import build_body_action, phase_vector


def _tensor(value: np.ndarray | list | float, dtype=None) -> torch.Tensor:
    result = torch.as_tensor(np.asarray(value), dtype=dtype)
    return result.contiguous()


class IndexedTartanGroundDataset(Dataset):
    """Build model samples lazily from per-trajectory memory-mapped cache arrays."""

    def __init__(self, manifest: str | Path, cache_trajectories: int = 4):
        self.manifest = Path(manifest)
        self.rows = [json.loads(line) for line in self.manifest.read_text().splitlines() if line.strip()]
        if not self.rows:
            raise RuntimeError(f"Empty indexed manifest: {self.manifest}")
        self.cache_trajectories = max(1, int(cache_trajectories))
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.trajectory_indices: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(self.rows):
            self.trajectory_indices[row["trajectory_id"]].append(index)
        self.stats_path = Path(self.rows[0]["stats_path"])
        with np.load(self.stats_path, allow_pickle=False) as stats:
            self.stats = {key: np.asarray(stats[key], np.float32) for key in stats.files}

    def __len__(self) -> int:
        return len(self.rows)

    def _trajectory(self, row: dict[str, Any]) -> dict[str, Any]:
        path = row["cache_dir"]
        if path in self._cache:
            value = self._cache.pop(path)
            self._cache[path] = value
            return value
        root = Path(path)
        names = (
            "P", "B", "U", "L_LEG", "E", "contact", "time",
            "camera_times", "latents", "foot_position", "foot_jacobian", "action_features",
        )
        value = {name: np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False) for name in names}
        self._cache[path] = value
        while len(self._cache) > self.cache_trajectories:
            self._cache.popitem(last=False)
        return value

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        arrays = self._trajectory(row)
        stats = self.stats
        visual_index = int(row["camera_index"])
        current = np.asarray(row["current_indices"], dtype=np.int64)
        targets_visual = np.asarray(row["target_visual_indices"], dtype=np.int64)
        K = len(targets_visual)
        p = arrays["P"]
        body = arrays["B"]
        control = arrays["U"]
        lowlevel = arrays["L_LEG"]
        error = arrays["E"]
        contact = arrays["contact"]
        time = arrays["time"]

        p_mean, p_std = stats["proprio_mean"], np.maximum(stats["proprio_std"], 1e-6)
        b_mean, b_std = stats["body_mean"], np.maximum(stats["body_std"], 1e-6)
        u_mean, u_std = stats["control_mean"], np.maximum(stats["control_std"], 1e-6)
        l_mean, l_std = stats["command_mean"], np.maximum(stats["command_std"], 1e-6)
        e_std = np.maximum(stats["error_std"], 1e-6)
        Pn = ((p - p_mean) / p_std).astype(np.float32, copy=False)
        Bn = ((body - b_mean) / b_std).astype(np.float32, copy=False)
        Un = ((control - u_mean) / u_std).astype(np.float32, copy=False)
        Ln = ((lowlevel - l_mean) / l_std).astype(np.float32, copy=False)
        En = (error / e_std).astype(np.float32, copy=False)

        def leg_history(end: int) -> np.ndarray:
            start = end - 49
            return np.concatenate([
                Pn[start : end + 1],
                np.repeat(Un[start : end + 1, None, :], 4, axis=1),
                Ln[start : end + 1], En[start : end + 1],
            ], axis=-1).transpose(1, 0, 2).copy()

        body_history = np.concatenate([Bn, Un], axis=-1)
        future_actions = []
        leg_actions = []
        body_actions = []
        queries = []
        physics = []
        jointbody_baselines = []
        tau_baselines = []
        force_baselines = []
        lowlevel_seq = []
        error_seq = []
        target_z = []
        target_contact = []
        target_load = []
        target_jointbody = []
        target_tau = []
        target_force = []
        target_phase = []
        thresholds = stats["load_thresholds"]

        for k in range(K):
            now, nxt = int(current[k]), int(current[k + 1])
            target_now = int(np.searchsorted(time, arrays["camera_times"][targets_visual[k]], side="right") - 1)
            if target_now != nxt:
                nxt = target_now
            u_interval = Un[now : nxt + 1]
            l_interval = Ln[now : nxt + 1]
            if not len(u_interval):
                u_interval, l_interval = Un[now : now + 1], Ln[now : now + 1]
            future_actions.append(np.concatenate([
                u_interval[-1], l_interval[-1].reshape(-1),
                u_interval.mean(0), l_interval.mean(0).reshape(-1),
            ]).astype(np.float32))
            body_actions.append(build_body_action(control[now]))
            action = np.asarray(arrays["action_features"][visual_index + k], np.float32)
            delta_mean = float(stats["command_delta_mean"])
            delta_std = max(float(stats["command_delta_std"]), 1e-6)
            leg_actions.append(np.concatenate([
                (action[:, :3] - l_mean) / l_std,
                ((action[:, 3:4] - delta_mean) / delta_std),
            ], axis=-1))
            lowlevel_seq.append(Ln[now])
            error_seq.append(En[now])

            current_p = np.asarray(p[now], np.float32)
            current_b = np.asarray(body[now], np.float32)
            current_contact = np.asarray(contact[now], np.float32)
            baseline_q = current_p[:, :3] + 0.1 * current_p[:, 3:6]
            baseline = np.concatenate([
                np.concatenate([baseline_q, current_p[:, 3:6]], axis=-1).reshape(-1), current_b
            ]).astype(np.float32)
            jointbody_baselines.append(baseline)
            tau_baselines.append(current_p[:, 6:9])
            force_baselines.append(current_p[:, 9:12])

            force_norm = np.linalg.norm(current_p[:, 9:12], axis=-1)
            regime = np.digitize(force_norm, thresholds).astype(np.int64)
            regime_onehot = np.eye(4, dtype=np.float32)[regime]
            physics.append(np.concatenate([
                ((baseline_q - p_mean[:3]) / p_std[:3]).reshape(-1),
                ((current_p[:, 6:9] - p_mean[6:9]) / p_std[6:9]).reshape(-1),
                (current_b[:3] - b_mean[:3]) / b_std[:3],
                force_norm / max(float(thresholds[-1]), 1.0),
                (regime_onehot * current_contact[:, None]).reshape(-1),
            ]).astype(np.float32))

            foot = np.asarray(arrays["foot_position"][visual_index + k], np.float32)
            planned_delta = np.tile((control[now, :3] * 0.30)[None], (4, 1)).astype(np.float32)
            queries.append(np.concatenate([Pn[now], np.tile(Un[now][None], (4, 1)), foot, planned_delta], axis=-1))

            target_p = np.asarray(p[nxt], np.float32)
            target_contact_k = np.asarray(contact[nxt], np.float32)
            target_force_k = target_p[:, 9:12]
            target_z.append(np.asarray(arrays["latents"][targets_visual[k]], np.float16))
            target_contact.append(target_contact_k)
            target_load.append(np.digitize(np.linalg.norm(target_force_k, axis=-1), thresholds).astype(np.int64))
            target_jointbody.append(np.concatenate([target_p[:, :6].reshape(-1), np.asarray(body[nxt], np.float32)]))
            target_tau.append(target_p[:, 6:9])
            target_force.append(target_force_k)
            target_phase.append(phase_vector(current_contact, target_contact_k > 0.5))

        sample = {
            "z_window": _tensor(np.asarray(arrays["latents"][visual_index - 3 : visual_index + 1], np.float16)),
            "leg_hist_t": _tensor(leg_history(int(current[0]))),
            "leg_hist_tm1": _tensor(leg_history(int(current[0]) - 10)),
            "leg_hist_tm2": _tensor(leg_history(int(current[0]) - 20)),
            "body_hist_t": _tensor(body_history[int(current[0]) - 49 : int(current[0]) + 1].copy()),
            "body_action_seq": _tensor(body_actions),
            "leg_action_seq": _tensor(leg_actions),
            "leg_query_features_seq": _tensor(queries),
            "future_action_seq": _tensor(future_actions),
            "physics_seq": _tensor(physics),
            "horizon_seq": _tensor(np.arange(1, K + 1, dtype=np.float32)[:, None] * 0.1),
            "lowlevel_cmd_seq": _tensor(lowlevel_seq),
            "err_seq": _tensor(error_seq),
            "jointbody_baseline_seq": _tensor(jointbody_baselines),
            "jointbody_std": _tensor(stats["jointbody_std"]),
            "tau_baseline_seq": _tensor(tau_baselines),
            "tau_std": _tensor(stats["tau_std"]),
            "force_baseline_seq": _tensor(force_baselines),
            "force_std": _tensor(stats["force_std"]),
            "target_z_seq": _tensor(target_z),
            "target_contact_seq": _tensor(target_contact),
            "target_load_class_seq": _tensor(target_load, torch.long),
            "target_jointbody_seq": _tensor(target_jointbody),
            "target_tau_seq": _tensor(target_tau),
            "target_force_seq": _tensor(target_force),
            "target_phase_seq": _tensor(target_phase, torch.long),
            "proprio_mean": _tensor(stats["proprio_mean"]),
            "proprio_std": _tensor(stats["proprio_std"]),
            "body_mean": _tensor(stats["body_mean"]),
            "body_std": _tensor(stats["body_std"]),
            "load_thresholds": _tensor(stats["load_thresholds"]),
            "foot_jacobian": _tensor(arrays["foot_jacobian"][visual_index]),
        }
        validate_sample(sample)
        return sample


class BalancedTrajectorySampler(Sampler[int]):
    """Shuffle short within-trajectory blocks and cycle through all trajectories."""

    def __init__(self, dataset: IndexedTartanGroundDataset, seed: int = 781, block_size: int = 64):
        self.dataset = dataset
        self.seed = int(seed)
        self.block_size = max(1, int(block_size))
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        per_trajectory = {}
        blocks = []
        for trajectory, indices in self.dataset.trajectory_indices.items():
            order = torch.randperm(len(indices), generator=generator).tolist()
            shuffled = [indices[i] for i in order]
            per_trajectory[trajectory] = [
                shuffled[start : start + self.block_size]
                for start in range(0, len(shuffled), self.block_size)
            ]
            blocks.extend((trajectory, i) for i in range(len(per_trajectory[trajectory])))
        if not blocks:
            return iter(())
        order = torch.randperm(len(blocks), generator=generator).tolist()
        result = []
        for block_index in order:
            trajectory, block = blocks[block_index]
            result.extend(per_trajectory[trajectory][block])
        return iter(result)
