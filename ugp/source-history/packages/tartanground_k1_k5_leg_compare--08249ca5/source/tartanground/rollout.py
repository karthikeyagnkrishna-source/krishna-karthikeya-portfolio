from __future__ import annotations

import torch


def teacher_forcing_mask(
    batch_size: int,
    horizons: int,
    sample_offset: int = 0,
    seed: int = 781,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Choose one deterministic mode per example, alternating exactly 50/50."""
    if batch_size < 0 or horizons < 1:
        raise ValueError("batch_size must be nonnegative and horizons must be positive")
    ids = torch.arange(batch_size, device=device, dtype=torch.long) + int(sample_offset)
    forced = (ids + int(seed)) % 2 == 0
    mask = forced[:, None].expand(batch_size, horizons).clone()
    mask[:, 0] = False  # the initial observation is always the same for both modes
    return mask
