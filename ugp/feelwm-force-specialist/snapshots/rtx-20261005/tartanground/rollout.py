from __future__ import annotations

import torch


def teacher_forcing_probability(
    epoch: int,
    total_epochs: int,
    start_probability: float,
    end_probability: float,
) -> float:
    """Linearly anneal scheduled sampling probability across complete epochs."""
    start = float(start_probability)
    end = float(end_probability)
    if not (0.0 <= start <= 1.0 and 0.0 <= end <= 1.0):
        raise ValueError("teacher-forcing probabilities must be between zero and one")
    if total_epochs < 1:
        raise ValueError("total_epochs must be positive")
    if total_epochs == 1:
        return end
    progress = min(max(int(epoch), 0), total_epochs - 1) / float(total_epochs - 1)
    return start + (end - start) * progress


def teacher_forcing_mask(
    batch_size: int,
    horizons: int,
    sample_offset: int = 0,
    seed: int = 781,
    device: torch.device | str | None = None,
    probability: float = 0.5,
) -> torch.Tensor:
    """Choose one deterministic scheduled-sampling mode per example."""
    if batch_size < 0 or horizons < 1:
        raise ValueError("batch_size must be nonnegative and horizons must be positive")
    probability = float(probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    ids = torch.arange(batch_size, device=device, dtype=torch.long) + int(sample_offset)
    if probability == 0.5:
        # Preserve the original exactly balanced schedule for existing configs.
        forced = (ids + int(seed)) % 2 == 0
    elif probability <= 0.0:
        forced = torch.zeros_like(ids, dtype=torch.bool)
    elif probability >= 1.0:
        forced = torch.ones_like(ids, dtype=torch.bool)
    else:
        # Multiplicative hashing gives a deterministic, well-distributed sample
        # without mutable RNG state, including across resumed runs.
        buckets = ((ids + int(seed)) * 2654435761) % 1_000_003
        forced = buckets < int(round(probability * 1_000_003))
    mask = forced[:, None].expand(batch_size, horizons).clone()
    mask[:, 0] = False  # the initial observation is always the same for both modes
    return mask
