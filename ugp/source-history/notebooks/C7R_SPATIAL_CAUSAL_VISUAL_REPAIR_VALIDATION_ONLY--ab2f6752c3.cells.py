# Source inspection export of the adjacent notebook.
# Notebook execution order and dependencies still apply.
# IPython magic, shell lines and prose-only conclusion cells are comments.

# %% notebook cell 0
# ============================================================================
# C7R — SPATIAL + TEMPORAL RGB-D REPAIR, VALIDATION ONLY
#
# Run AFTER the original C7 gate failed.
#
# Scientific rules preserved:
#   * Frozen Stage-B B9 dynamics model.
#   * Same train/validation trajectories and same physical targets/losses.
#   * NO VLM and NO planner.
#   * NO fresh C-test scoring and NO test-label inspection in this cell.
#   * Model selection + anti-cheating controls use validation only.
#
# Repair:
#   RGB  : DINOv2 PATCH tokens pooled over 6 lower/front terrain regions.
#   Depth: metric lower/front crop retaining an 8x8 spatial grid.
#   Time : current visual descriptor + two causal frame-to-frame deltas
#          (current, current-prev1, prev1-prev2).
#   RAM  : frame-level features + lazy origin lookup + batch normalization.
#
# IMPORTANT: Do not run C8-C10 unless C7R_GATE['open_fresh_C_test'] is True.
# ============================================================================

import gc, json, math, hashlib, random
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from IPython.display import display
from sklearn.metrics import precision_recall_fscore_support

# -----------------------------------------------------------------------------
# 0) Hard prerequisites and protocol lock
# -----------------------------------------------------------------------------
_required = [
    "C_DATA", "C_SPLIT_MANIFEST", "C_ROOT", "RAW_ROOT",
    "B9_DENSE", "B9_PREP", "B9_CALIBRATION", "BResponseModel",
    "B_DEVICE", "B_H", "B_STEP_WIDTH", "B_STEP_LAYOUT",
    "B_ACTION_GATHER", "B_ACTION_HIDDEN", "B_MODEL_DIM",
    "_b_loss", "_b_tail_raw", "_b_reshape_cont", "_b_group",
    "predict_B", "c_metrics_from_prediction", "CVisualResidualModel",
]
_missing = [x for x in _required if x not in globals()]
if _missing:
    raise RuntimeError("C7R missing prerequisites: " + repr(_missing))

_c8_existing_marker = Path(C_ROOT) / "C8_TEST_WAS_SCORED.json"
if _c8_existing_marker.exists():
    raise RuntimeError(
        "A C8 test marker already exists. Do not retune Stage C after opening the fresh test set. "
        f"Marker: {_c8_existing_marker}"
    )

# Extra protection: use ONLY train/val rows here.
C7R_DEV_ROWS = C_SPLIT_MANIFEST[
    C_SPLIT_MANIFEST["role"].isin(["train", "val"])
].copy().reset_index(drop=True)

if len(C7R_DEV_ROWS) != 8:
    print("Warning: expected 8 Stage-C development trajectories; found", len(C7R_DEV_ROWS))

C7R_TEST_IDS_LOCKED = C_SPLIT_MANIFEST.loc[
    C_SPLIT_MANIFEST["role"] == "test", "run_id"
].tolist()

print("Fresh C test trajectories remain LOCKED:", C7R_TEST_IDS_LOCKED)
print("C7R will read RGB-D only from these roles:", sorted(C7R_DEV_ROWS["role"].unique()))

SEED_C7R = 1781
random.seed(SEED_C7R)
np.random.seed(SEED_C7R)
torch.manual_seed(SEED_C7R)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED_C7R)

C7R_ROOT = Path(C_ROOT) / "spatial_temporal_repair_v1"
C7R_FEATURE_ROOT = C7R_ROOT / "frame_features"
C7R_MODEL_ROOT = C7R_ROOT / "models"
C7R_TABLE_ROOT = C7R_ROOT / "tables"
for _p in [C7R_ROOT, C7R_FEATURE_ROOT, C7R_MODEL_ROOT, C7R_TABLE_ROOT]:
    _p.mkdir(parents=True, exist_ok=True)

# Keep original failed C7 result untouched; use a new namespace/files.
C7R_SPEC = {
    "rgb_backbone": "vit_small_patch14_dinov2.lvd142m",
    "rgb_regions": "lower_60pct_2x3_patch_token_pool",
    "depth_region": "lower_60pct_metric_8x8x4",
    "temporal_encoding": "current_delta1_delta2",
    "temporal_frames": 3,
    "uses_test_trajectories": False,
}
(C7R_ROOT / "protocol.json").write_text(json.dumps(C7R_SPEC, indent=2))

# Release any old visual inference model before loading DINO again.
if "_c_release_active_model" in globals():
    try:
        _c_release_active_model()
    except Exception:
        pass
B9_DENSE["model"].to("cpu").eval()
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# -----------------------------------------------------------------------------
# 1) Reload frozen DINOv2 and extract SPATIAL patch-token features
#    ONLY for train/val trajectories.
# -----------------------------------------------------------------------------
import timm
from timm.data import resolve_model_data_config, create_transform
import cv2

C7R_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
C7R_DINO_NAME = "vit_small_patch14_dinov2.lvd142m"
C7R_BATCH = 16
C7R_LOWER_START = 0.40  # top 40% excluded; lower 60% is terrain-focused
C7R_DEPTH_GRID = 8
C7R_DEPTH_MIN_M = 0.20
C7R_DEPTH_MAX_M = 8.0
C7R_FEATURE_VERSION = "dino_patch_lower60_2x3__depth_lower60_8x8x4__t3delta_v1"

print("Loading frozen DINOv2 for spatial patch tokens:", C7R_DINO_NAME)
C7R_RGB_MODEL = timm.create_model(
    C7R_DINO_NAME,
    pretrained=True,
    num_classes=0,
).to(C7R_DEVICE).eval()
for _p in C7R_RGB_MODEL.parameters():
    _p.requires_grad_(False)
_C7R_CFG = resolve_model_data_config(C7R_RGB_MODEL)
C7R_RGB_TRANSFORM = create_transform(**_C7R_CFG, is_training=False)


def c7r_read_metric_depth(path):
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError(f"Could not read depth {path}")
    if raw.dtype == np.uint8 and raw.ndim == 3 and raw.shape[2] == 4:
        raw = np.ascontiguousarray(raw)
        d = raw.view("<f4").reshape(raw.shape[0], raw.shape[1])
    elif raw.dtype == np.float32 and raw.ndim == 2:
        d = raw
    else:
        raise RuntimeError(f"Unexpected depth format {raw.dtype} {raw.shape}: {path}")
    d = d.astype(np.float32)
    d[~np.isfinite(d)] = 0.0
    return d


def c7r_depth_terrain_descriptor(depth):
    """Lower/front crop, but retain explicit spatial structure."""
    d = np.asarray(depth, np.float32)
    h = d.shape[0]
    y0 = int(round(C7R_LOWER_START * h))
    d = d[y0:, :]

    valid = (d >= C7R_DEPTH_MIN_M) & (d <= C7R_DEPTH_MAX_M)
    fill = np.where(valid, d, C7R_DEPTH_MAX_M).astype(np.float32)

    small = cv2.resize(fill, (64, 64), interpolation=cv2.INTER_AREA)
    vsmall = cv2.resize(valid.astype(np.float32), (64, 64), interpolation=cv2.INTER_AREA)

    logd = np.log1p(np.clip(small, C7R_DEPTH_MIN_M, C7R_DEPTH_MAX_M))
    logd /= np.log1p(C7R_DEPTH_MAX_M)

    inv = 1.0 / np.maximum(small, C7R_DEPTH_MIN_M)
    inv = np.clip(
        (inv - 1.0 / C7R_DEPTH_MAX_M)
        / (1.0 / C7R_DEPTH_MIN_M - 1.0 / C7R_DEPTH_MAX_M),
        0.0, 1.0,
    )

    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    grad = np.clip(grad / 2.0, 0.0, 1.0)

    pooled = []
    for c in [logd, inv, grad, vsmall]:
        pooled.append(
            cv2.resize(
                c.astype(np.float32),
                (C7R_DEPTH_GRID, C7R_DEPTH_GRID),
                interpolation=cv2.INTER_AREA,
            )
        )
    return np.stack(pooled, axis=0).reshape(-1).astype(np.float32)


@torch.no_grad()
def c7r_encode_spatial_rgb(images):
    """Return 6 pooled lower/front DINO patch-token regions per frame."""
    x = torch.stack([C7R_RGB_TRANSFORM(im) for im in images], dim=0).to(C7R_DEVICE)
    ff = C7R_RGB_MODEL.forward_features(x)

    if isinstance(ff, dict):
        if "x_norm_patchtokens" in ff:
            patch = ff["x_norm_patchtokens"]
        elif "x_prenorm" in ff:
            patch = ff["x_prenorm"]
        else:
            raise RuntimeError("Unsupported DINO forward_features keys: " + repr(list(ff.keys())))
    else:
        patch = ff

    if patch.ndim != 3:
        raise RuntimeError(f"Unexpected DINO feature shape: {tuple(patch.shape)}")

    grid = getattr(getattr(C7R_RGB_MODEL, "patch_embed", None), "grid_size", None)
    if grid is not None:
        gh, gw = int(grid[0]), int(grid[1])
        npatch = gh * gw
        if patch.shape[1] >= npatch:
            # Robust to CLS/register prefix tokens: spatial patches are the final npatch tokens.
            patch = patch[:, -npatch:, :]
        else:
            grid = None

    if grid is None or patch.shape[1] != gh * gw:
        n = int(patch.shape[1])
        s = int(round(math.sqrt(n)))
        if s * s != n:
            raise RuntimeError(
                f"Cannot infer square DINO patch grid from {n} tokens; "
                f"model grid={grid}"
            )
        gh = gw = s

    patch = patch.reshape(len(images), gh, gw, -1)
    r0 = max(0, min(gh - 1, int(math.floor(C7R_LOWER_START * gh))))
    rmid = max(r0 + 1, min(gh - 1, int(round(0.70 * gh))))

    c1 = max(1, gw // 3)
    c2 = max(c1 + 1, (2 * gw) // 3)
    col_ranges = [(0, c1), (c1, c2), (c2, gw)]
    row_ranges = [(r0, rmid), (rmid, gh)]

    regs = []
    for ra, rb in row_ranges:
        for ca, cb in col_ranges:
            z = patch[:, ra:rb, ca:cb, :].mean(dim=(1, 2))
            z = F.normalize(z.float(), dim=-1)
            regs.append(z)

    desc = torch.cat(regs, dim=-1)
    return desc.cpu().numpy().astype(np.float32)


def c7r_load_manifest(rid):
    p = Path(C_ROOT) / "manifests" / f"{rid}_front_manifest.npz"
    if not p.exists():
        # Fall back to C_MANIFEST_ROOT if it is still defined.
        if "C_MANIFEST_ROOT" in globals():
            p = Path(C_MANIFEST_ROOT) / f"{rid}_front_manifest.npz"
    if not p.exists():
        raise FileNotFoundError(f"Missing frozen C manifest for {rid}: {p}")
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def c7r_extract_dev_frame_features(row):
    rid = str(row.run_id)
    out = C7R_FEATURE_ROOT / f"{rid}_spatial_frame_features.npz"

    if out.exists():
        with np.load(out, allow_pickle=False) as z:
            ver = str(z["feature_version"].reshape(())) if "feature_version" in z.files else ""
            if ver == C7R_FEATURE_VERSION:
                return {k: z[k] for k in z.files}

    manifest = c7r_load_manifest(rid)
    ids = np.asarray(manifest["frame_ids"])
    base = Path(RAW_ROOT) / str(row.environment) / "Data_anymal" / str(row.run)

    rgb_parts, dep_parts = [], []
    for st in range(0, len(ids), C7R_BATCH):
        batch_ids = ids[st:st + C7R_BATCH]
        ims, deps = [], []
        for fid0 in batch_ids:
            fid = str(fid0)
            rgb_path = base / "image_lcam_front" / f"{fid}_lcam_front.png"
            dep_path = base / "depth_lcam_front" / f"{fid}_lcam_front_depth.png"

            with Image.open(rgb_path) as im:
                ims.append(im.convert("RGB"))
            deps.append(c7r_depth_terrain_descriptor(c7r_read_metric_depth(dep_path)))

        rgb_parts.append(c7r_encode_spatial_rgb(ims))
        dep_parts.append(np.stack(deps, axis=0).astype(np.float32))

        if (st // C7R_BATCH) % 20 == 0:
            print(f"{rid}: spatial terrain features {min(st+C7R_BATCH,len(ids))}/{len(ids)}")

    rgb = np.concatenate(rgb_parts, axis=0).astype(np.float32)
    dep = np.concatenate(dep_parts, axis=0).astype(np.float32)

    if len(rgb) != len(ids) or len(dep) != len(ids):
        raise RuntimeError(f"{rid}: spatial feature length mismatch")
    if not np.isfinite(rgb).all() or not np.isfinite(dep).all():
        raise RuntimeError(f"{rid}: non-finite spatial features")

    np.savez_compressed(
        out,
        frame_ids=ids,
        camera_times=np.asarray(manifest["camera_times"], np.float64),
        rgb_spatial=rgb.astype(np.float16),
        depth_spatial=dep.astype(np.float16),
        feature_version=np.asarray(C7R_FEATURE_VERSION),
    )
    return {
        "frame_ids": ids,
        "camera_times": np.asarray(manifest["camera_times"], np.float64),
        "rgb_spatial": rgb.astype(np.float16),
        "depth_spatial": dep.astype(np.float16),
        "feature_version": np.asarray(C7R_FEATURE_VERSION),
    }


C7R_FRAME_FEATURES = {}
for _row in C7R_DEV_ROWS.itertuples(index=False):
    # Hard gate: this loop can never include fresh test rows.
    if str(_row.role) == "test":
        raise RuntimeError("Internal error: C7R attempted to extract fresh test visual features")
    C7R_FRAME_FEATURES[str(_row.run_id)] = c7r_extract_dev_frame_features(_row)

print("C7R development frame features ready:", list(C7R_FRAME_FEATURES))

# DINO is no longer needed; free GPU before adapter training.
del C7R_RGB_MODEL, C7R_RGB_TRANSFORM
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# -----------------------------------------------------------------------------
# 2) Lazy causal 3-frame origin descriptor
# -----------------------------------------------------------------------------
def c7r_origin_visual(rid, d, positions):
    positions = np.asarray(positions, dtype=np.int64)
    if rid not in C7R_FRAME_FEATURES:
        raise RuntimeError(
            f"{rid}: no C7R frame features. Fresh C test is intentionally locked."
        )

    tab = C7R_FRAME_FEATURES[rid]
    vi = np.asarray(d["visual_frame_index"][positions], dtype=np.int64)
    if np.any(vi < 0) or np.any(vi >= len(tab["rgb_spatial"])):
        raise RuntimeError(f"{rid}: visual frame index out of range")

    v1 = np.maximum(vi - 1, 0)
    v2 = np.maximum(vi - 2, 0)

    # Explicit causality: only current/latest causal frame and earlier frames.
    if np.any(v1 > vi) or np.any(v2 > v1):
        raise RuntimeError(f"{rid}: temporal visual causality violation")

    R = np.asarray(tab["rgb_spatial"], np.float32)
    D = np.asarray(tab["depth_spatial"], np.float32)

    r0, r1, r2 = R[vi], R[v1], R[v2]
    d0, d1, d2 = D[vi], D[v1], D[v2]

    # Current appearance/geometry plus causal motion/change cues.
    rgb = np.concatenate([r0, r0-r1, r1-r2], axis=1).astype(np.float32)
    dep = np.concatenate([d0, d0-d1, d1-d2], axis=1).astype(np.float32)
    return rgb, dep


# Development-only dictionary; shallow references to already-built dynamics arrays.
C7R_DATA = {
    rid: d for rid, d in C_DATA.items()
    if rid not in set(C7R_TEST_IDS_LOCKED)
}

if set(C7R_DATA) != set(C7R_DEV_ROWS["run_id"].astype(str)):
    raise RuntimeError("C7R development data IDs do not match the frozen train/val manifest")

# -----------------------------------------------------------------------------
# 3) Streaming train-only visual normalization
# -----------------------------------------------------------------------------
C7R_STAT_CHUNK = 256


def c7r_fit_visual_norm():
    sums = {"r": None, "r2": None, "d": None, "d2": None}
    n = 0

    for rid, dat in C7R_DATA.items():
        pos = np.flatnonzero(dat["split"] == "train")
        if len(pos) == 0:
            continue
        for st in range(0, len(pos), C7R_STAT_CHUNK):
            pp = pos[st:st+C7R_STAT_CHUNK]
            R, D = c7r_origin_visual(rid, dat, pp)
            if sums["r"] is None:
                sums["r"] = np.zeros(R.shape[1], np.float64)
                sums["r2"] = np.zeros(R.shape[1], np.float64)
                sums["d"] = np.zeros(D.shape[1], np.float64)
                sums["d2"] = np.zeros(D.shape[1], np.float64)
            sums["r"] += R.sum(0, dtype=np.float64)
            sums["r2"] += np.square(R, dtype=np.float64).sum(0, dtype=np.float64)
            sums["d"] += D.sum(0, dtype=np.float64)
            sums["d2"] += np.square(D, dtype=np.float64).sum(0, dtype=np.float64)
            n += len(R)
            del R, D

    if n == 0:
        raise RuntimeError("C7R found zero training visual samples")

    rm = sums["r"] / n
    dm = sums["d"] / n
    rv = np.maximum(sums["r2"] / n - rm*rm, 0.0)
    dv = np.maximum(sums["d2"] / n - dm*dm, 0.0)
    rs = np.sqrt(rv)
    ds = np.sqrt(dv)
    rs[rs < 1e-5] = 1.0
    ds[ds < 1e-5] = 1.0

    return {
        "rgb_mean": rm.astype(np.float32),
        "rgb_std": rs.astype(np.float32),
        "depth_mean": dm.astype(np.float32),
        "depth_std": ds.astype(np.float32),
        "train_count": int(n),
    }


C7R_VIS_NORM = c7r_fit_visual_norm()
print(
    "C7R visual normalization:",
    {
        "train_count": C7R_VIS_NORM["train_count"],
        "rgb_dim": len(C7R_VIS_NORM["rgb_mean"]),
        "depth_dim": len(C7R_VIS_NORM["depth_mean"]),
    }
)

# -----------------------------------------------------------------------------
# 4) Lazy dataset: same B9 targets/tail balancing, new visual lookup only
# -----------------------------------------------------------------------------
C7R_TRAIN_BATCH = 96
C7R_EPOCHS = 20
C7R_PATIENCE = 5
C7R_LR = 2e-4
C7R_WEIGHT_DECAY = 1e-4
C7R_RESIDUAL_L2 = globals().get("C_RESIDUAL_L2", 2e-3)


class C7RDataset(Dataset):
    def __init__(self, data_dict, split, balanced=False):
        self.data_dict = data_dict
        self.run_keys = []
        self.run_of_parts = []
        self.local_of_parts = []
        self.ft_parts = []
        self.et_parts = []
        self.tail_parts = []
        self.sw_parts = []

        terrain_counts = {}
        records = []
        for rid, d in data_dict.items():
            idx = np.flatnonzero(d["split"] == split).astype(np.int64)
            if len(idx) == 0:
                continue
            terrain = str(d["terrain"])
            terrain_counts[terrain] = terrain_counts.get(terrain, 0) + len(idx)
            records.append((rid, d, idx, terrain))

        if not records:
            raise RuntimeError(f"No C7R samples for split={split}")

        total = sum(len(x[2]) for x in records)
        nterr = len(terrain_counts)
        terrain_factor = {t: total/(nterr*c) for t, c in terrain_counts.items()}

        for rix, (rid, d, idx, terrain) in enumerate(records):
            self.run_keys.append(rid)
            self.run_of_parts.append(np.full(len(idx), rix, np.int16))
            self.local_of_parts.append(idx)

            ft = np.empty((len(idx), B_H, 4), np.uint8)
            et = np.empty((len(idx), B_H, 4), np.uint8)
            tail = np.empty((len(idx), 4), np.float32)
            sw = np.empty(len(idx), np.float32)

            chunk = 512
            for st in range(0, len(idx), chunk):
                sel = idx[st:st+chunk]
                step = np.asarray(d["Y_cont"][sel], np.float32).reshape(-1, B_H, B_STEP_WIDTH)
                td = np.asarray(d["Y_touchdown"][sel], np.float32)
                force = step[:, :, B_STEP_LAYOUT["force_odom"]].reshape(-1, B_H, 4, 3)
                E = step[:, :, B_STEP_LAYOUT["E"]].reshape(-1, B_H, 4, 3)
                ft0 = np.linalg.norm(force, axis=-1) > B9_PREP["force95"]
                et0 = np.linalg.norm(E, axis=-1) > B9_PREP["E95"]
                tail0 = np.log1p(np.maximum(
                    _b_tail_raw(step, B9_PREP["body_med"], B9_PREP["body_mad"]), 0.0
                )).astype(np.float32)
                sw0 = (
                    1.0
                    + 2.0*ft0.max(axis=(1,2)).astype(np.float32)
                    + 2.0*et0.max(axis=(1,2)).astype(np.float32)
                    + 4.0*td.max(axis=(1,2)).astype(np.float32)
                )
                if balanced:
                    sw0 *= float(terrain_factor[terrain])
                ft[st:st+len(sel)] = ft0.astype(np.uint8)
                et[st:st+len(sel)] = et0.astype(np.uint8)
                tail[st:st+len(sel)] = tail0
                sw[st:st+len(sel)] = np.clip(sw0, 1.0, 12.0)

            self.ft_parts.append(ft)
            self.et_parts.append(et)
            self.tail_parts.append(tail)
            self.sw_parts.append(sw)

        self.run_of = np.concatenate(self.run_of_parts)
        self.local_of = np.concatenate(self.local_of_parts)
        self.force_tail = np.concatenate(self.ft_parts)
        self.E_tail = np.concatenate(self.et_parts)
        self.tail_log = np.concatenate(self.tail_parts)
        self.sample_weight = np.concatenate(self.sw_parts)

        self.run_of_parts = self.local_of_parts = None
        self.ft_parts = self.et_parts = self.tail_parts = self.sw_parts = None

    def __len__(self):
        return len(self.local_of)

    def __getitem__(self, i):
        rix = int(self.run_of[i])
        j = int(self.local_of[i])
        rid = self.run_keys[rix]
        d = self.data_dict[rid]

        R, DP = c7r_origin_visual(rid, d, np.asarray([j], np.int64))
        rgb = R[0]
        dep = DP[0]

        x = np.asarray(d["X_core"][j], np.float32)
        a = np.asarray(d["X_action_dense"][j], np.float32)
        y = np.asarray(d["Y_cont"][j], np.float32).reshape(B_H, B_STEP_WIDTH)
        c = np.asarray(d["Y_contact"][j], np.float32)
        td = np.asarray(d["Y_touchdown"][j], np.float32)
        imp = np.asarray(d["Y_impulse"][j], np.float32)

        return (
            torch.from_numpy(x), torch.from_numpy(a), torch.from_numpy(y),
            torch.from_numpy(c), torch.from_numpy(td), torch.from_numpy(imp),
            torch.from_numpy(self.force_tail[i].astype(np.float32)),
            torch.from_numpy(self.E_tail[i].astype(np.float32)),
            torch.from_numpy(self.tail_log[i]),
            torch.from_numpy(rgb), torch.from_numpy(dep),
        )


def _c7r_t(x):
    return torch.as_tensor(x, dtype=torch.float32, device=B_DEVICE)


C7R_NORM_T = {
    "x_mean": _c7r_t(B9_PREP["x_mean"]),
    "x_std": _c7r_t(B9_PREP["x_std"]),
    "action_mean": _c7r_t(B9_PREP["action_mean"]),
    "action_std": _c7r_t(B9_PREP["action_std"]),
    "y_mean": _c7r_t(B9_PREP["y_mean"]),
    "y_std": _c7r_t(B9_PREP["y_std"]),
    "imp_mean": _c7r_t(B9_PREP["imp_mean"]),
    "imp_std": _c7r_t(B9_PREP["imp_std"]),
    "rgb_mean": _c7r_t(C7R_VIS_NORM["rgb_mean"]),
    "rgb_std": _c7r_t(C7R_VIS_NORM["rgb_std"]),
    "depth_mean": _c7r_t(C7R_VIS_NORM["depth_mean"]),
    "depth_std": _c7r_t(C7R_VIS_NORM["depth_std"]),
}


def c7r_prepare_batch(raw):
    raw = tuple(t.to(B_DEVICE, non_blocking=False) for t in raw)
    x, a, y, c, td, imp, ft, et, tail, rgb, dep = raw

    x = (x - C7R_NORM_T["x_mean"][None, :]) / C7R_NORM_T["x_std"][None, :]
    a = (
        (a - C7R_NORM_T["action_mean"][None, None, :])
        / C7R_NORM_T["action_std"][None, None, :]
    )
    # Correct shape: y is [B,H,87], y_mean/std are [H,87].
    y = (
        (y - C7R_NORM_T["y_mean"][None, :, :])
        / C7R_NORM_T["y_std"][None, :, :]
    )
    imp = (
        (imp - C7R_NORM_T["imp_mean"][None, None, None, :])
        / C7R_NORM_T["imp_std"][None, None, None, :]
    )
    rgb = (rgb - C7R_NORM_T["rgb_mean"][None, :]) / C7R_NORM_T["rgb_std"][None, :]
    dep = (dep - C7R_NORM_T["depth_mean"][None, :]) / C7R_NORM_T["depth_std"][None, :]

    if y.ndim != 3 or y.shape[1:] != (B_H, B_STEP_WIDTH):
        raise RuntimeError(f"C7R normalized target has bad shape {tuple(y.shape)}")
    return (x, a, y, c, td, imp, ft, et, tail), rgb, dep


# -----------------------------------------------------------------------------
# 5) Train three adapters; frozen B9 base, same objective
# -----------------------------------------------------------------------------
def c7r_load_b9_base_cpu():
    ck = torch.load(B9_DENSE["checkpoint"], map_location="cpu", weights_only=False)
    base = BResponseModel(
        history_dim=int(ck["history_dim"]),
        continuous_step_dim=int(ck["step_width"]),
        use_action=bool(ck["use_action"]),
    )
    base.load_state_dict(ck["state_dict"], strict=True)
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    return base


def c7r_train_mode(mode):
    tr = C7RDataset(C7R_DATA, "train", balanced=True)
    va = C7RDataset(C7R_DATA, "val", balanced=False)
    sampler = WeightedRandomSampler(
        torch.as_tensor(tr.sample_weight, dtype=torch.double),
        num_samples=len(tr), replacement=True,
    )
    tr_loader = DataLoader(tr, batch_size=C7R_TRAIN_BATCH, sampler=sampler,
                           num_workers=0, pin_memory=False)
    va_loader = DataLoader(va, batch_size=C7R_TRAIN_BATCH, shuffle=False,
                           num_workers=0, pin_memory=False)

    base = c7r_load_b9_base_cpu()
    model = CVisualResidualModel(
        base,
        rgb_dim=len(C7R_VIS_NORM["rgb_mean"]),
        depth_dim=len(C7R_VIS_NORM["depth_mean"]),
        mode=mode,
    ).to(B_DEVICE)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=C7R_LR, weight_decay=C7R_WEIGHT_DECAY)

    best = float("inf")
    best_state = None
    bad = 0
    history = []

    @torch.no_grad()
    def eval_loss():
        model.eval()
        total, n = 0.0, 0
        for raw in va_loader:
            bb, rgb, dep = c7r_prepare_batch(raw)
            out = model(bb[0], bb[1], rgb, dep)
            loss = _b_loss(out, bb, B9_PREP)["total"] + C7R_RESIDUAL_L2*out["_residual_l2"]
            total += float(loss) * len(bb[0])
            n += len(bb[0])
            del raw, bb, rgb, dep, out, loss
        return total / max(n, 1)

    for ep in range(1, C7R_EPOCHS + 1):
        model.train()
        total, n = 0.0, 0
        for raw in tr_loader:
            bb, rgb, dep = c7r_prepare_batch(raw)
            opt.zero_grad(set_to_none=True)
            out = model(bb[0], bb[1], rgb, dep)
            loss = _b_loss(out, bb, B9_PREP)["total"] + C7R_RESIDUAL_L2*out["_residual_l2"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"C7R-{mode}: non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            opt.step()
            total += float(loss.detach()) * len(bb[0])
            n += len(bb[0])
            del raw, bb, rgb, dep, out, loss

        tr_loss = total / max(n, 1)
        va_loss = eval_loss()
        history.append({"mode": mode, "epoch": ep, "train_loss": tr_loss, "val_loss": va_loss})
        print(f"C7R-{mode:5s} ep={ep:02d} train={tr_loss:.5f} val={va_loss:.5f}")

        if va_loss < best - 1e-4:
            best = va_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
                if not k.startswith("base.")
            }
            bad = 0
        else:
            bad += 1
        if bad >= C7R_PATIENCE:
            print(f"C7R-{mode}: early stop")
            break

    if best_state is None:
        raise RuntimeError(f"C7R-{mode}: no checkpoint")

    path = C7R_MODEL_ROOT / f"C7R_{mode}_adapter.pt"
    torch.save({
        "mode": mode,
        "adapter_state_dict": best_state,
        "best_val_loss": float(best),
        "vis_norm": C7R_VIS_NORM,
        "feature_version": C7R_FEATURE_VERSION,
        "B9_checkpoint": B9_DENSE["checkpoint"],
        "uses_test": False,
    }, path)

    result = {
        "mode": mode,
        "checkpoint": str(path),
        "best_val_loss": float(best),
        "history": pd.DataFrame(history),
    }

    del model, base, trainable, opt, tr_loader, va_loader, tr, va, sampler, best_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


print("\nTraining validation-only spatial/temporal adapters...")
C7R_MODELS = {}
for _mode in ["rgb", "depth", "rgbd"]:
    print("\n" + "="*80)
    print("C7R MODE:", _mode)
    print("="*80)
    C7R_MODELS[_mode] = c7r_train_mode(_mode)

C7R_TRAINING_SUMMARY = pd.DataFrame([
    {"mode": k, "best_val_loss": v["best_val_loss"], "checkpoint": v["checkpoint"]}
    for k, v in C7R_MODELS.items()
])
display(C7R_TRAINING_SUMMARY)
C7R_TRAINING_SUMMARY.to_csv(C7R_TABLE_ROOT / "training_summary.csv", index=False)

# -----------------------------------------------------------------------------
# 6) Lazy inference for repair models
# -----------------------------------------------------------------------------
C7R_ACTIVE = {"checkpoint": None, "model": None}


def c7r_release_model():
    m = C7R_ACTIVE.get("model")
    if m is not None:
        try:
            m.to("cpu")
        except Exception:
            pass
    C7R_ACTIVE["checkpoint"] = None
    C7R_ACTIVE["model"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def c7r_get_model(bundle):
    p = str(bundle["checkpoint"])
    if C7R_ACTIVE["checkpoint"] == p and C7R_ACTIVE["model"] is not None:
        return C7R_ACTIVE["model"]
    c7r_release_model()
    saved = torch.load(p, map_location="cpu", weights_only=False)
    base = c7r_load_b9_base_cpu()
    model = CVisualResidualModel(
        base,
        rgb_dim=len(C7R_VIS_NORM["rgb_mean"]),
        depth_dim=len(C7R_VIS_NORM["depth_mean"]),
        mode=saved["mode"],
    )
    missing, unexpected = model.load_state_dict(saved["adapter_state_dict"], strict=False)
    bad_missing = [k for k in missing if not k.startswith("base.")]
    if bad_missing or unexpected:
        raise RuntimeError(f"C7R checkpoint mismatch: missing={bad_missing}, unexpected={unexpected}")
    model = model.to(B_DEVICE).eval()
    C7R_ACTIVE.update({"checkpoint": p, "model": model})
    return model


def c7r_runmean_features(rid, d, split):
    idx = np.flatnonzero(d["split"] == split)
    sr = sd = None
    n = 0
    for st in range(0, len(idx), 256):
        pp = idx[st:st+256]
        R, D = c7r_origin_visual(rid, d, pp)
        if sr is None:
            sr = np.zeros(R.shape[1], np.float64)
            sd = np.zeros(D.shape[1], np.float64)
        sr += R.sum(0, dtype=np.float64)
        sd += D.sum(0, dtype=np.float64)
        n += len(R)
    return (sr/n).astype(np.float32), (sd/n).astype(np.float32)


@torch.no_grad()
def c7r_predict(bundle, rid, d, split="val", control="true", seed=0, batch_size=256):
    idx = np.flatnonzero(d["split"] == split).astype(np.int64)
    if len(idx) == 0:
        raise RuntimeError(f"{rid}: no samples for {split}")
    if rid not in C7R_FRAME_FEATURES:
        raise RuntimeError(f"{rid}: C7R has no features; fresh test remains locked")

    if control == "shuffled":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(idx))
    else:
        perm = None
    if control == "run_mean":
        meanR, meanD = c7r_runmean_features(rid, d, split)
    else:
        meanR = meanD = None
    if control not in {"true", "shuffled", "run_mean"}:
        raise ValueError(control)

    model = c7r_get_model(bundle)
    store = {k: [] for k in [
        "mean_z", "scale_z", "contact", "touchdown",
        "impulse_mean_z", "impulse_scale_z", "tail_q90_log",
        "truth", "contact_truth", "touchdown_truth", "impulse_truth", "origins"
    ]}

    for st in range(0, len(idx), batch_size):
        ed = min(st+batch_size, len(idx))
        target_pos = idx[st:ed]

        if control == "true":
            R, DP = c7r_origin_visual(rid, d, target_pos)
        elif control == "shuffled":
            source_pos = idx[perm[st:ed]]
            R, DP = c7r_origin_visual(rid, d, source_pos)
        else:
            R = np.repeat(meanR[None, :], len(target_pos), axis=0)
            DP = np.repeat(meanD[None, :], len(target_pos), axis=0)

        x = torch.from_numpy(np.asarray(d["X_core"][target_pos], np.float32)).to(B_DEVICE)
        a = torch.from_numpy(np.asarray(d["X_action_dense"][target_pos], np.float32)).to(B_DEVICE)
        rgb = torch.from_numpy(R).to(B_DEVICE)
        dep = torch.from_numpy(DP).to(B_DEVICE)

        x = (x - C7R_NORM_T["x_mean"][None, :]) / C7R_NORM_T["x_std"][None, :]
        a = (a - C7R_NORM_T["action_mean"][None,None,:]) / C7R_NORM_T["action_std"][None,None,:]
        rgb = (rgb - C7R_NORM_T["rgb_mean"][None,:]) / C7R_NORM_T["rgb_std"][None,:]
        dep = (dep - C7R_NORM_T["depth_mean"][None,:]) / C7R_NORM_T["depth_std"][None,:]

        out = model(x, a, rgb, dep)
        store["mean_z"].append(out["mean_z"].cpu().numpy())
        store["scale_z"].append(np.exp(out["log_scale_z"].cpu().numpy()))
        store["contact"].append(torch.sigmoid(out["contact_logits"]).cpu().numpy())
        store["touchdown"].append(torch.sigmoid(out["touchdown_logits"]).cpu().numpy())
        store["impulse_mean_z"].append(out["impulse_mean_z"].cpu().numpy())
        store["impulse_scale_z"].append(np.exp(out["impulse_log_scale_z"].cpu().numpy()))
        store["tail_q90_log"].append(out["tail_q90_log"].cpu().numpy())
        store["truth"].append(np.asarray(d["Y_cont"][target_pos], np.float32))
        store["contact_truth"].append(np.asarray(d["Y_contact"][target_pos], np.float32))
        store["touchdown_truth"].append(np.asarray(d["Y_touchdown"][target_pos], np.float32))
        store["impulse_truth"].append(np.asarray(d["Y_impulse"][target_pos], np.float32))
        store["origins"].append(np.asarray(d["origins"][target_pos], np.int64))

        del x, a, rgb, dep, out, R, DP

    raw = {k: np.concatenate(v, axis=0) for k, v in store.items()}

    mean = raw["mean_z"] * B9_PREP["y_std"][None,:,:] + B9_PREP["y_mean"][None,:,:]
    scale = raw["scale_z"] * B9_PREP["y_std"][None,:,:]
    imp_mean = raw["impulse_mean_z"] * B9_PREP["imp_std"][None,None,None,:] + B9_PREP["imp_mean"][None,None,None,:]
    imp_scale = raw["impulse_scale_z"] * B9_PREP["imp_std"][None,None,None,:]

    result = {
        "continuous_mean": mean.astype(np.float32),
        "continuous_scale": scale.astype(np.float32),
        "contact_probability": raw["contact"].astype(np.float32),
        "touchdown_probability": raw["touchdown"].astype(np.float32),
        "impulse_mean": imp_mean.astype(np.float32),
        "impulse_scale": imp_scale.astype(np.float32),
        "tail_q90": np.maximum(np.expm1(raw["tail_q90_log"]), 0.0).astype(np.float32),
        "truth": raw["truth"].astype(np.float32),
        "contact_truth": raw["contact_truth"].astype(np.float32),
        "touchdown_truth": raw["touchdown_truth"].astype(np.float32),
        "impulse_truth": raw["impulse_truth"].astype(np.float32),
        "origins": raw["origins"],
    }

    n = len(idx)
    assert result["continuous_mean"].shape == (n, B_H, B_STEP_WIDTH)
    assert result["continuous_scale"].shape == (n, B_H, B_STEP_WIDTH)
    return result


# Restore B9 baseline for validation metrics.
B9_DENSE["model"].to(B_DEVICE).eval()

# -----------------------------------------------------------------------------
# 7) Validation-only calibration for each repair mode
# -----------------------------------------------------------------------------
def c7r_concat_predictions(bundle, split="val", control="true"):
    outs, rids, terrains = [], [], []
    for rid, d in C7R_DATA.items():
        if not np.any(d["split"] == split):
            continue
        p = c7r_predict(bundle, rid, d, split=split, control=control,
                        seed=SEED_C7R + int(hashlib.sha256(rid.encode()).hexdigest()[:8],16)%100000)
        outs.append(p)
        rids.extend([rid]*len(p["truth"]))
        terrains.extend([d["terrain"]]*len(p["truth"]))
    if not outs:
        raise RuntimeError(f"C7R no predictions for split={split}")
    merged = {k: np.concatenate([o[k] for o in outs], axis=0) for k in outs[0]}
    merged["run_ids"] = np.asarray(rids)
    merged["terrains"] = np.asarray(terrains)
    return merged


def c7r_fit_calibration(bundle):
    val = c7r_concat_predictions(bundle, "val", "true")
    truth = _b_reshape_cont(val["truth"])
    mu = np.asarray(val["continuous_mean"], float)
    scale = np.maximum(np.asarray(val["continuous_scale"], float), 1e-6)

    scale_factor, cov_rows = {}, []
    for group, sl in B_STEP_LAYOUT.items():
        err = np.abs(truth[:,:,sl] - mu[:,:,sl])
        sc = scale[:,:,sl]
        factor = max(float(np.quantile(err/sc, .95)), 1e-3)
        scale_factor[group] = factor
        cov_rows.append({
            "group": group,
            "raw_gaussian_95_coverage": float(np.mean(err <= 1.96*sc)),
            "calibrated_marginal_coverage": float(np.mean(err <= factor*sc)),
            "scale_factor": factor,
        })

    td = val["touchdown_truth"].astype(bool)
    if np.any(td):
        ie = np.abs(val["impulse_truth"] - val["impulse_mean"])
        isc = np.maximum(val["impulse_scale"], 1e-6)
        impulse_factor = float(np.quantile(ie[td]/isc[td], .95))
    else:
        impulse_factor = 1.96

    tail_actual = _b_tail_raw(truth, B9_PREP["body_med"], B9_PREP["body_mad"])
    tail_pred = np.asarray(val["tail_q90"], float)
    tail_add, tail_rows = {}, []
    for j, name in enumerate(["force_peak", "E_peak", "tau_peak", "body_peak"]):
        add = float(max(0.0, np.quantile(tail_actual[:,j]-tail_pred[:,j], .90)))
        tail_add[name] = add
        tail_rows.append({
            "metric": name,
            "raw_q90_coverage": float(np.mean(tail_actual[:,j] <= tail_pred[:,j])),
            "calibrated_q90_coverage": float(np.mean(tail_actual[:,j] <= tail_pred[:,j] + add)),
            "additive_calibration": add,
        })

    y = val["touchdown_truth"].reshape(-1).astype(int)
    p = val["touchdown_probability"].reshape(-1)
    best = None
    for th in np.linspace(.05, .95, 37):
        yp = (p >= th).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(y, yp, average="binary", zero_division=0)
        row = (float(f1), float(pr), float(rc), float(th))
        if best is None or row[0] > best[0]:
            best = row

    return {
        "scale_factor": scale_factor,
        "impulse_scale_factor": impulse_factor,
        "tail_add": tail_add,
        "touchdown_threshold": best[3],
        "touchdown_val_precision": best[1],
        "touchdown_val_recall": best[2],
        "touchdown_val_f1": best[0],
        "risk_reference": dict(B9_CALIBRATION["risk_reference"]),
        "coverage_table": pd.DataFrame(cov_rows),
        "tail_table": pd.DataFrame(tail_rows),
    }


C7R_CALIBRATIONS = {mode: c7r_fit_calibration(bundle) for mode, bundle in C7R_MODELS.items()}

# -----------------------------------------------------------------------------
# 8) C6R: validation model selection, same score as original C6
# -----------------------------------------------------------------------------
_val_rows = []
for mode, bundle in C7R_MODELS.items():
    cal = C7R_CALIBRATIONS[mode]
    for rid, d in C7R_DATA.items():
        if not np.any(d["split"] == "val"):
            continue
        pred = c7r_predict(bundle, rid, d, "val", "true")
        vm = d["split"] == "val"
        for subset, mask in [
            ("all", np.ones(vm.sum(), dtype=bool)),
            ("precontact", d["C_precontact"][vm]),
        ]:
            met = c_metrics_from_prediction(pred, d, cal, mask=mask)
            _val_rows.append({"mode": mode, "run_id": rid, "terrain": d["terrain"], "subset": subset, **met})

for rid, d in C7R_DATA.items():
    if not np.any(d["split"] == "val"):
        continue
    pred = predict_B(B9_DENSE, d, "val")
    vm = d["split"] == "val"
    for subset, mask in [
        ("all", np.ones(vm.sum(), dtype=bool)),
        ("precontact", d["C_precontact"][vm]),
    ]:
        met = c_metrics_from_prediction(pred, d, B9_CALIBRATION, mask=mask)
        _val_rows.append({"mode": "B9_baseline", "run_id": rid, "terrain": d["terrain"], "subset": subset, **met})

C6R_VALIDATION = pd.DataFrame(_val_rows)
display(C6R_VALIDATION)

base_pre = C6R_VALIDATION[
    (C6R_VALIDATION["mode"] == "B9_baseline")
    & (C6R_VALIDATION["subset"] == "precontact")
].mean(numeric_only=True)

_select = []
for mode in ["rgb", "depth", "rgbd"]:
    g = C6R_VALIDATION[
        (C6R_VALIDATION["mode"] == mode)
        & (C6R_VALIDATION["subset"] == "precontact")
    ].mean(numeric_only=True)
    force_gain = (base_pre["force_tail_component_mae"] - g["force_tail_component_mae"]) / max(base_pre["force_tail_component_mae"], 1e-9)
    E_gain = (base_pre["E_tail_component_mae"] - g["E_tail_component_mae"]) / max(base_pre["E_tail_component_mae"], 1e-9)
    score = (
        0.35*g["risk_spearman"]
        + 0.25*g["force_peak_spearman"]
        + 0.20*force_gain
        + 0.10*E_gain
        + 0.10*g["touchdown_f1"]
    )
    _select.append({
        "mode": mode,
        "validation_selection_score": float(score),
        "precontact_risk_spearman": float(g["risk_spearman"]),
        "precontact_force_peak_spearman": float(g["force_peak_spearman"]),
        "precontact_force_tail_gain_vs_B9": float(force_gain),
        "precontact_E_tail_gain_vs_B9": float(E_gain),
        "precontact_touchdown_f1": float(g["touchdown_f1"]),
    })

C6R_MODE_SELECTION = pd.DataFrame(_select).sort_values(
    "validation_selection_score", ascending=False
).reset_index(drop=True)
C7R_SELECTED_MODE = str(C6R_MODE_SELECTION.iloc[0]["mode"])
C7R_SELECTED_MODEL = C7R_MODELS[C7R_SELECTED_MODE]
C7R_SELECTED_CAL = C7R_CALIBRATIONS[C7R_SELECTED_MODE]

print("\nC6R MODE SELECTION")
display(C6R_MODE_SELECTION)
print("SELECTED:", C7R_SELECTED_MODE)

C6R_VALIDATION.to_csv(C7R_TABLE_ROOT / "C6R_validation_metrics.csv", index=False)
C6R_MODE_SELECTION.to_csv(C7R_TABLE_ROOT / "C6R_mode_selection.csv", index=False)

# -----------------------------------------------------------------------------
# 9) C7R: exact anti-cheating controls on validation only
# -----------------------------------------------------------------------------
_control_rows = []

# Frozen B9 baseline.
for rid, d in C7R_DATA.items():
    if not np.any(d["split"] == "val"):
        continue
    vm = d["split"] == "val"
    mask = d["C_precontact"][vm]
    bp = predict_B(B9_DENSE, d, "val")
    met = c_metrics_from_prediction(bp, d, B9_CALIBRATION, mask=mask)
    _control_rows.append({"condition": "B9_baseline", "run_id": rid, "terrain": d["terrain"], **met})

for cond in ["true", "shuffled", "run_mean"]:
    for rid, d in C7R_DATA.items():
        if not np.any(d["split"] == "val"):
            continue
        seed = SEED_C7R + int(hashlib.sha256(rid.encode()).hexdigest()[:8],16) % 100000
        pred = c7r_predict(C7R_SELECTED_MODEL, rid, d, "val", control=cond, seed=seed)
        vm = d["split"] == "val"
        mask = d["C_precontact"][vm]
        met = c_metrics_from_prediction(pred, d, C7R_SELECTED_CAL, mask=mask)
        _control_rows.append({"condition": cond, "run_id": rid, "terrain": d["terrain"], **met})

C7R_CONTROLS = pd.DataFrame(_control_rows)
print("\nC7R VALIDATION CAUSALITY CONTROLS")
display(C7R_CONTROLS)
C7R_SUMMARY = C7R_CONTROLS.groupby("condition").mean(numeric_only=True)
display(C7R_SUMMARY)


def _c7rv(cond, metric):
    return float(C7R_SUMMARY.loc[cond, metric])


true_risk = _c7rv("true", "risk_spearman")
base_risk = _c7rv("B9_baseline", "risk_spearman")
shuf_risk = _c7rv("shuffled", "risk_spearman")
mean_risk = _c7rv("run_mean", "risk_spearman")
true_force = _c7rv("true", "force_peak_spearman")
base_force = _c7rv("B9_baseline", "force_peak_spearman")
base_tail = _c7rv("B9_baseline", "force_tail_component_mae")
true_tail = _c7rv("true", "force_tail_component_mae")

C7R_GATE = {
    "selected_mode": C7R_SELECTED_MODE,
    "feature_version": C7R_FEATURE_VERSION,
    "fresh_test_ids_still_locked": C7R_TEST_IDS_LOCKED,
    "validation_precontact_true_risk_spearman": true_risk,
    "validation_precontact_B9_risk_spearman": base_risk,
    "validation_precontact_shuffled_risk_spearman": shuf_risk,
    "validation_precontact_runmean_risk_spearman": mean_risk,
    "validation_precontact_true_force_peak_spearman": true_force,
    "validation_precontact_B9_force_peak_spearman": base_force,
    "validation_precontact_force_tail_improvement_fraction": (base_tail-true_tail)/max(base_tail,1e-9),
    # EXACT original C7 thresholds:
    "true_beats_B9_risk": bool(true_risk > base_risk + 0.01),
    "true_beats_shuffled_risk": bool(true_risk > shuf_risk + 0.01),
    "true_beats_runmean_risk": bool(true_risk > mean_risk + 0.005),
    "tail_or_peak_improves": bool(true_force > base_force + 0.01 or true_tail < base_tail),
}
C7R_GATE["open_fresh_C_test"] = bool(
    C7R_GATE["true_beats_B9_risk"]
    and C7R_GATE["true_beats_shuffled_risk"]
    and C7R_GATE["true_beats_runmean_risk"]
    and C7R_GATE["tail_or_peak_improves"]
)

print("\n" + "="*92)
print("C7R SPATIAL/TEMPORAL VALIDATION GATE")
print("="*92)
print(json.dumps(C7R_GATE, indent=2))

C7R_CONTROLS.to_csv(C7R_TABLE_ROOT / "C7R_causality_controls.csv", index=False)
(C7R_TABLE_ROOT / "C7R_gate.json").write_text(json.dumps(C7R_GATE, indent=2))

# Deliberately DO NOT score test here, even on PASS.
if C7R_GATE["open_fresh_C_test"]:
    print("\nC7R PASS: time-local spatial/temporal vision beat the validation controls.")
    print("Fresh C tests are STILL UNSCORED by this cell.")
    print("Send me this output; next we run the one-time C8R-C10 test cell.")
else:
    print("\nC7R DID NOT PASS. Fresh C tests remain locked and unscored.")
    print("Do NOT force the gate and do NOT run C8-C10.")

c7r_release_model()
B9_DENSE["model"].to(B_DEVICE).eval()
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\nC7R validation-only repair cell complete.")
