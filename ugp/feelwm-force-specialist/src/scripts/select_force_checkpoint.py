#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from data import move_batch_to_device
from model import MergedFeelWMV7
from tartanground.evaluate import normalized_force_score
from tartanground.indexed import IndexedTartanGroundDataset
from tartanground.selection import choose_best_candidate


def score_checkpoint(model, checkpoint_path: Path, loader, device, amp_dtype: str, history_shift: int) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    total = 0.0
    samples = 0
    enabled = device.type == "cuda" and amp_dtype in ("bf16", "fp16")
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
                output = model(batch, history_shift=history_shift)
                score = normalized_force_score(output, batch)
            batch_size = int(batch["target_force_seq"].shape[0])
            total += float(score.detach().float().cpu()) * batch_size
            samples += batch_size
    stat = checkpoint_path.stat()
    return {
        "path": str(checkpoint_path.resolve()),
        "force_score": total / max(samples, 1),
        "samples": samples,
        "epoch": int(checkpoint.get("epoch", -1)),
        "global_step": int(checkpoint.get("global_step", -1)),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-link", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    missing = [str(path) for path in args.candidates if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing candidate checkpoints: {missing}")
    mcfg, tcfg, _ = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = IndexedTartanGroundDataset(args.manifest)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = MergedFeelWMV7(mcfg).to(device)
    scores = [
        score_checkpoint(model, path, loader, device, tcfg.amp_dtype, tcfg.history_shift_samples)
        for path in dict.fromkeys(path.resolve() for path in args.candidates)
    ]
    selected = choose_best_candidate(scores)
    report = {"selection_metric": "normalized_force_mae", "candidates": scores, "selected": selected}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, args.output)

    args.selected_link.parent.mkdir(parents=True, exist_ok=True)
    if args.selected_link.exists() or args.selected_link.is_symlink():
        args.selected_link.unlink()
    args.selected_link.symlink_to(Path(selected["path"]))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
