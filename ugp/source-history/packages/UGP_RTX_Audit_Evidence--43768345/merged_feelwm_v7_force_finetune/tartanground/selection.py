from __future__ import annotations

from typing import Any, Iterable


def choose_best_candidate(candidates: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Return the checkpoint with the lowest finite development force score."""
    valid = []
    for candidate in candidates:
        score = float(candidate["force_score"])
        if score != score or score in (float("inf"), float("-inf")):
            continue
        valid.append(candidate)
    if not valid:
        raise ValueError("no candidate has a finite force_score")
    return min(valid, key=lambda candidate: (float(candidate["force_score"]), str(candidate["path"])))
