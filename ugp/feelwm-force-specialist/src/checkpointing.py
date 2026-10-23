from __future__ import annotations
import os
import random
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def rng_state() -> Dict[str, Any]:
    out = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        out["cuda"] = torch.cuda.get_rng_state_all()
    return out


def restore_rng_state(state: Dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch_state = state["torch"].detach().to(device="cpu", dtype=torch.uint8)
    torch.set_rng_state(torch_state)
    if torch.cuda.is_available() and "cuda" in state:
        cuda_states = [item.detach().to(device="cpu", dtype=torch.uint8) for item in state["cuda"]]
        torch.cuda.set_rng_state_all(cuda_states)


def atomic_torch_save(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(path: str | Path, model, optimizer, scheduler, scaler, epoch: int,
                    global_step: int, best_val: float, config: Dict[str, Any]) -> None:
    pack = {
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_val": float(best_val),
        "rng": rng_state(),
        "config": config,
    }
    atomic_torch_save(pack, path)


def load_checkpoint(path: str | Path, model, optimizer=None, scheduler=None, scaler=None, map_location="cpu"):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    unwrap_model(model).load_state_dict(ck["model"], strict=True)
    if optimizer is not None and ck.get("optimizer") is not None:
        optimizer.load_state_dict(ck["optimizer"])
    if scheduler is not None and ck.get("scheduler") is not None:
        scheduler.load_state_dict(ck["scheduler"])
    if scaler is not None and ck.get("scaler") is not None:
        scaler.load_state_dict(ck["scaler"])
    restore_rng_state(ck.get("rng"))
    return ck


def resolve_resume(resume: str, output_dir: str | Path) -> Path | None:
    if resume.lower() == "none":
        return None
    if resume.lower() == "auto":
        p = Path(output_dir) / "checkpoints" / "latest.pt"
        return p if p.exists() else None
    p = Path(resume)
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def prune_step_checkpoints(checkpoint_dir: str | Path, keep: int) -> None:
    p = Path(checkpoint_dir)
    files = sorted(p.glob("step_*.pt"), key=lambda x: x.stat().st_mtime)
    for old in files[:-max(keep, 0)]:
        try:
            old.unlink()
        except OSError:
            pass
