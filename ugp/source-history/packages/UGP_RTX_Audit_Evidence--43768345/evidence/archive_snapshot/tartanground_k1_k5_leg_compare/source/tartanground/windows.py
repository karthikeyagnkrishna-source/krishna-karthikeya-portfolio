from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Window:
    start: int
    stop: int
    requested_center: float

    @property
    def center(self) -> float:
        return 0.5 * (self.start + self.stop - 1)

    @property
    def indices(self) -> range:
        return range(self.start, self.stop)


def quarter_windows(frame_count: int, length: int = 700, min_center_gap: int = 50) -> tuple[Window, Window]:
    frame_count = int(frame_count)
    length = int(length)
    if frame_count < length:
        raise ValueError(f"trajectory has {frame_count} frames but each window requires {length}")
    maximum_start = frame_count - length
    half_span = 0.5 * (length - 1)
    result = []
    for fraction in (0.25, 0.75):
        requested = fraction * (frame_count - 1)
        start = int(round(requested - half_span))
        start = min(max(start, 0), maximum_start)
        result.append(Window(start=start, stop=start + length, requested_center=requested))
    if result[1].center - result[0].center < int(min_center_gap):
        raise ValueError("the two 700-frame window centers cannot be separated by five seconds")
    return result[0], result[1]


def union_with_membership(windows: Sequence[Window]) -> tuple[list[int], dict[int, tuple[int, ...]]]:
    memberships: dict[int, list[int]] = {}
    for window_id, window in enumerate(windows):
        for index in window.indices:
            memberships.setdefault(index, []).append(window_id)
    indices = sorted(memberships)
    return indices, {index: tuple(memberships[index]) for index in indices}


def split_anchors(
    anchor_indices: Iterable[int], purge_frames: int = 50, train_fraction: float = 0.8
) -> dict[str, list[int]]:
    anchors = sorted(set(int(index) for index in anchor_indices))
    if not anchors:
        return {"train": [], "purge": [], "dev": []}
    cut = int(len(anchors) * float(train_fraction))
    cut = min(max(cut, 1), len(anchors))
    purge_until = anchors[cut - 1] + 1 + int(purge_frames)
    purge_stop = cut
    while purge_stop < len(anchors) and anchors[purge_stop] < purge_until:
        purge_stop += 1
    return {
        "train": anchors[:cut],
        "purge": anchors[cut:purge_stop],
        "dev": anchors[purge_stop:],
    }


def encode_cache_native_action(commands: np.ndarray) -> np.ndarray:
    commands = np.asarray(commands, dtype=np.float32)
    if commands.ndim < 3 or commands.shape[-1] != 3:
        raise ValueError("commands must have shape [time, leg, 3]")
    delta = np.zeros(commands.shape[:-1] + (1,), dtype=np.float32)
    if len(commands) > 1:
        delta[:-1, ..., 0] = np.linalg.norm(commands[1:] - commands[:-1], axis=-1)
    return np.concatenate([commands, delta], axis=-1)
