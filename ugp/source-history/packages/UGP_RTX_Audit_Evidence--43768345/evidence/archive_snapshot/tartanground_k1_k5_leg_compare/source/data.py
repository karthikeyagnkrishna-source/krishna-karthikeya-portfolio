from __future__ import annotations
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Any
import torch
from torch.utils.data import Dataset


REQUIRED_KEYS = {
    "z_window",                    # [4,64,256] post-Perceiver visual state
    "leg_hist_t",                  # [4,T,24]
    "leg_hist_tm1",                # [4,T,24]
    "leg_hist_tm2",                # [4,T,24]
    "body_hist_t",                 # [T,18]
    "body_action_seq",             # [K,4]
    "leg_action_seq",              # [K,4,4] -- CURRENT action for each rollout step
    "leg_query_features_seq",      # [K,4,24]
    "future_action_seq",           # [K,36]
    "physics_seq",                 # [K,47]
    "horizon_seq",                 # [K,1]
    "lowlevel_cmd_seq",            # [K,4,3]
    "err_seq",                     # [K,4,3]
    "jointbody_baseline_seq",      # [K,36]
    "jointbody_std",               # [36]
    "tau_baseline_seq",            # [K,4,3]
    "tau_std",                     # [3]
    "force_baseline_seq",          # [K,4,3]
    "force_std",                   # [3]
    "target_z_seq",                # [K,64,256] = DINO|Conv|Geom|Flow targets
    "target_contact_seq",          # [K,4] float 0/1
    "target_load_class_seq",       # [K,4] long 0..3
    "target_jointbody_seq",        # [K,36]
    "target_tau_seq",              # [K,4,3]
    "target_force_seq",            # [K,4,3]
    "target_phase_seq",            # [K,4] long 0..3
}


def validate_sample(sample: Dict[str, Any]) -> None:
    missing = REQUIRED_KEYS - set(sample)
    if missing:
        raise KeyError(f"Sample missing required keys: {sorted(missing)}")
    if tuple(sample["z_window"].shape[-3:]) != (4, 64, 256):
        raise ValueError(f"z_window must be [4,64,256], got {tuple(sample['z_window'].shape)}")
    if sample["leg_action_seq"].shape[-2:] != (4, 4):
        raise ValueError("leg_action_seq must be [K,4,4] in leg order FL,FR,HL,HR")
    if sample["target_z_seq"].shape[-2:] != (64, 256):
        raise ValueError("target_z_seq must be [K,64,256]")


class ShardedWorldModelDataset(Dataset):
    """
    Manifest JSONL format, one row per sample:
      {"path":"/abs/path/shard_000.pt", "index":17}

    Each shard is:
      torch.save({"samples": [sample0, sample1, ...]}, shard_path)

    A small per-worker LRU cache avoids reloading a shard for every sample.
    """
    def __init__(self, manifest: str | Path, cache_shards: int = 2, validate_first: int = 2):
        self.manifest = Path(manifest)
        if not self.manifest.exists():
            raise FileNotFoundError(self.manifest)
        self.rows = [json.loads(line) for line in self.manifest.read_text().splitlines() if line.strip()]
        if not self.rows:
            raise RuntimeError(f"Empty manifest: {self.manifest}")
        self.cache_shards = max(1, int(cache_shards))
        self._cache: OrderedDict[str, Any] = OrderedDict()
        for i in range(min(validate_first, len(self.rows))):
            validate_sample(self[i])

    def __len__(self) -> int:
        return len(self.rows)

    def _load_shard(self, path: str):
        if path in self._cache:
            obj = self._cache.pop(path)
            self._cache[path] = obj
            return obj
        obj = torch.load(path, map_location="cpu", weights_only=False)
        self._cache[path] = obj
        while len(self._cache) > self.cache_shards:
            self._cache.popitem(last=False)
        return obj

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        shard = self._load_shard(row["path"])
        sample = shard["samples"][int(row["index"])]
        # Convert all arrays/scalars to tensors once here.
        out: Dict[str, torch.Tensor] = {}
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                out[k] = v
            else:
                out[k] = torch.as_tensor(v)
        return out


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    long_keys = {"target_load_class_seq", "target_phase_seq"}
    for k, v in batch.items():
        if not isinstance(v, torch.Tensor):
            continue
        if k in long_keys:
            v = v.long()
        elif not torch.is_floating_point(v):
            v = v.float()
        out[k] = v.to(device, non_blocking=True)
    return out
