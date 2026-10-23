"""Input-only histories selected by recorded timestamps, never target values."""
from dataclasses import dataclass, fields
from typing import Any
import numpy as np


def select_camera_history(camera_times, anchor, count=6):
    times = np.asarray(camera_times, dtype=np.float64)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
        raise ValueError('Camera times must be a finite nonempty vector')
    if np.any(np.diff(times) <= 0):
        raise ValueError('Camera timestamps must be strictly increasing')
    if not isinstance(count, int) or count < 2 or not np.isfinite(anchor):
        raise ValueError('History needs at least two slots and a finite anchor')
    last = int(np.searchsorted(times, anchor, side='right')) - 1
    if last < 0:
        raise ValueError('No observed camera frame at or before the anchor')
    ids = np.arange(last-count+1, last+1, dtype=np.int64)
    return ids, ids >= 0


@dataclass(frozen=True)
class HistoryBatch:
    dino: Any
    conv: Any
    geom: Any
    visual_times: Any
    robot_times: Any
    valid: Any
    camera_indices: Any

    def to(self, device):
        return type(self)(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})

    def validate(self, batch, steps, frames):
        import torch
        if self.valid.dtype != torch.bool or self.valid.shape != (batch, frames):
            raise ValueError('Invalid visual history mask')
        if self.visual_times.shape != self.valid.shape or self.robot_times.shape != (batch, steps):
            raise ValueError('History timestamp shape does not match samples')
        if self.camera_indices.shape != self.valid.shape or self.camera_indices.dtype != torch.long:
            raise ValueError('Invalid camera indices')
        if not self.valid[:, -1].all():
            raise ValueError('The current observed image is required')
        for value in (self.dino, self.conv, self.geom):
            if value.ndim != 4 or value.shape[:2] != (batch, frames):
                raise ValueError('Visual history must have batch, frame, token, feature axes')
            if value.device != self.valid.device:
                raise ValueError('History tensors must be on the same device')
            if not torch.isfinite(value[self.valid]).all():
                raise ValueError('Nonfinite observed image encodings')
        if not torch.isfinite(self.robot_times).all() or not torch.isfinite(self.visual_times[self.valid]).all():
            raise ValueError('Nonfinite observed timestamps')
        if (self.visual_times[self.valid] > 1e-9).any() or (self.robot_times > 1e-9).any():
            raise ValueError('Future observations are forbidden')
        if self.robot_times[:, -1].abs().max() > 1e-9:
            raise ValueError('Robot history must end at the prediction anchor')
        if (self.robot_times[:, 1:] <= self.robot_times[:, :-1]).any():
            raise ValueError('Robot timestamps must increase')
        for row in range(batch):
            t = self.visual_times[row, self.valid[row]]
            ids = self.camera_indices[row, self.valid[row]]
            if (t[1:] <= t[:-1]).any() or (ids[1:] <= ids[:-1]).any():
                raise ValueError('Observed image times and indices must increase')


def build_history(data, visual, idx, steps, device, count=6):
    import torch
    robot = np.asarray(data['time'], dtype=np.float64)
    if not isinstance(idx, (int, np.integer)) or idx < steps-1 or idx >= len(robot):
        raise ValueError('Insufficient observed robot history')
    anchor = float(robot[idx])
    camera = np.asarray(visual['camera_times'], dtype=np.float64)
    ids, valid = select_camera_history(camera, anchor, count)
    if int(visual['visual_index'][idx]) != int(ids[-1]):
        raise ValueError('Current visual index disagrees with the causal camera clock')
    safe_ids = np.maximum(ids, 0)
    values = {}
    for field, cache in [('dino', 'DINO'), ('conv', 'CONV'), ('geom', 'GEOM')]:
        raw = visual[cache]
        if len(raw) != len(camera):
            raise ValueError('Image encoding cache length disagrees with camera timestamps')
        array = np.array(raw[safe_ids], dtype=np.float32, copy=True)
        array[~valid] = 0
        values[field] = torch.tensor(array[None], device=device)
    visual_times = camera[safe_ids]-anchor
    visual_times[~valid] = 0
    history = HistoryBatch(**values,
        visual_times=torch.tensor(visual_times[None], dtype=torch.float64, device=device),
        robot_times=torch.tensor((robot[idx-steps+1:idx+1]-anchor)[None], dtype=torch.float64, device=device),
        valid=torch.tensor(valid[None], dtype=torch.bool, device=device),
        camera_indices=torch.tensor(ids[None], dtype=torch.long, device=device))
    history.validate(1, steps, count)
    return history
