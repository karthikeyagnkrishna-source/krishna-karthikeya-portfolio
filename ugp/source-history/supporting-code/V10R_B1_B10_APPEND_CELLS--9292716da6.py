

####################################################################################################
# B1
####################################################################################################


# ======================================================================
# B1 — CORRECTED TARGETS: RELATIVE/HEADING-FRAME MOTION + DENSE ACTIONS
#
# Replaces the bad absolute-odom target with displacement expressed in the
# robot base frame at forecast origin. Builds the 100 Hz future action
# sequence used later by the action encoder.
#
# Requires: A1 + A2.
# Does NOT train.
# ======================================================================

from pathlib import Path
import json, math, gc
import numpy as np
import pandas as pd
from IPython.display import display

_REQUIRED_B1 = [
    "TWO_SPECS",
    "V10R_DERIVED_PATHS",
    "collect_full_tf_graph",
    "_find_directed_path",
    "_compose_path_series",
    "locate_physics_cache",
]
_missing = [x for x in _REQUIRED_B1 if x not in globals()]
if _missing:
    raise RuntimeError("Run through A2 first. Missing: " + repr(_missing))

B_ROOT = STRICT_ROOT / "b_tail_rgbd_world_model_v2"
B_DATA_ROOT = B_ROOT / "datasets"
B_MODEL_ROOT = B_ROOT / "models"
B_EVAL_ROOT = B_ROOT / "evaluation"
B_VIS_ROOT = B_ROOT / "visual"
B_MULTI_ROOT = B_ROOT / "multi_trajectory"
for p in (B_ROOT, B_DATA_ROOT, B_MODEL_ROOT, B_EVAL_ROOT, B_VIS_ROOT, B_MULTI_ROOT):
    p.mkdir(parents=True, exist_ok=True)

B_HORIZONS = np.asarray(CFG.horizons_s, dtype=np.float64)
B_H = len(B_HORIZONS)
B_HORIZON_OFFSETS = np.asarray(
    [int(round(float(h) * CFG.sample_hz)) for h in B_HORIZONS],
    dtype=np.int64,
)
B_MAX_OFFSET = int(B_HORIZON_OFFSETS.max())
B_DENSE_ACTION_OFFSETS = np.arange(1, B_MAX_OFFSET + 1, dtype=np.int64)

B_HISTORY_SNAPSHOTS = 10
B_HISTORY_STEPS = int(round(CFG.history_s * CFG.sample_hz))
B_HISTORY_LAGS = np.unique(
    np.round(
        np.linspace(B_HISTORY_STEPS - 1, 0, B_HISTORY_SNAPSHOTS)
    ).astype(np.int64)
)
if len(B_HISTORY_LAGS) != B_HISTORY_SNAPSHOTS:
    raise RuntimeError("History-lag construction produced duplicates.")

def _b_layout(items):
    out = {}
    c = 0
    for name, width in items:
        out[name] = slice(c, c + width)
        c += width
    return out, c

# IMPORTANT: touchdown impulse is no longer buried in the continuous target.
# It will be modeled conditionally in B2/B4.
B_STEP_LAYOUT, B_STEP_WIDTH = _b_layout([
    ("q", 12),
    ("dq", 12),
    ("tau", 12),
    ("force_odom", 12),
    ("body_B", 12),
    ("E", 12),
    ("foot_base", 12),
    ("delta_base_heading", 3),
])

B_HISTORY_WIDTH = 106   # same causal state signal used by A3
B_ACTION_WIDTH = 18     # U(6) + L_LEG(12)

def b_base_pose_series(spec, time_s, derived_path):
    """Return T_odom_base at every cache sample using the registered TF clock."""
    with np.load(derived_path, allow_pickle=False) as d:
        offset_s = float(np.asarray(d["cache_to_ros_offset_s"]).reshape(()))
    graph = collect_full_tf_graph(spec)
    cache_rel = np.asarray(time_s, float) - float(time_s[0])
    ros_query_rel = cache_rel + offset_s
    path = _find_directed_path(
        graph["static"], graph["dynamic"], "odom", "base"
    )
    return _compose_path_series(graph, path, ros_query_rel)

def b_build_run_dataset(spec, derived_path, split_role=None):
    cache_path = locate_physics_cache(spec)
    if cache_path is None:
        raise FileNotFoundError(f"{spec['run_id']}: physics cache missing.")

    derived_path = Path(derived_path)
    if not derived_path.is_file():
        raise FileNotFoundError(derived_path)

    with np.load(cache_path, allow_pickle=False) as z:
        time_s = np.asarray(z["time"], np.float64)
        P = np.asarray(z["P"], np.float32)
        B = np.asarray(z["B"], np.float32)
        U = np.asarray(z["U"], np.float32)
        L = np.asarray(z["L_LEG"], np.float32)
        E = np.asarray(z["E"], np.float32)

    with np.load(derived_path, allow_pickle=False) as d:
        foot_base = np.asarray(d["foot_base"], np.float32)
        force_odom = np.asarray(d["force_odom_from_anymal_state"], np.float32)
        contact = np.asarray(d["contact_from_anymal_state"], np.uint8).astype(bool)
        base_position_odom = np.asarray(d["base_position_odom"], np.float32)
        touchdown_event_raw = np.asarray(d["touchdown_event"], np.uint8).astype(bool)
        touchdown_impulse_raw = np.asarray(d["touchdown_impulse_odom"], np.float32)
        visual_index = np.asarray(d["visual_index"], np.int64)

    n = len(time_s)
    for name, arr in {
        "P": P, "B": B, "U": U, "L": L, "E": E,
        "foot_base": foot_base, "force_odom": force_odom,
        "contact": contact, "base_position_odom": base_position_odom,
        "touchdown_event": touchdown_event_raw,
        "touchdown_impulse": touchdown_impulse_raw,
        "visual_index": visual_index,
    }.items():
        if len(arr) != n:
            raise RuntimeError(f"{spec['run_id']}: {name} length {len(arr)} != {n}")

    if P.shape[1:] != (4, 12):
        raise RuntimeError(f"{spec['run_id']}: P shape {P.shape}")
    if B.ndim != 2 or B.shape[1] != 12:
        raise RuntimeError(f"{spec['run_id']}: B shape {B.shape}")
    if U.ndim != 2 or U.shape[1] != 6:
        raise RuntimeError(f"{spec['run_id']}: U shape {U.shape}")
    if L.shape[1:] != (4, 3):
        raise RuntimeError(f"{spec['run_id']}: L_LEG shape {L.shape}")

    first_origin = int(B_HISTORY_LAGS.max())
    origins = np.arange(
        first_origin,
        n - B_MAX_OFFSET,
        dtype=np.int64,
    )
    history_idx = origins[:, None] - B_HISTORY_LAGS[None, :]
    future_idx = origins[:, None] + B_HORIZON_OFFSETS[None, :]
    dense_action_idx = origins[:, None] + B_DENSE_ACTION_OFFSETS[None, :]

    # Causal state history. Force is the explicitly resolved odom-frame force.
    P_hist = np.concatenate([P[:, :, :9], force_odom], axis=-1)
    hist_signal = np.concatenate([
        P_hist.reshape(n, -1),          # 48
        B,                              # 12
        E.reshape(n, -1),               # 12
        contact.astype(np.float32),     # 4
        U,                              # 6
        L.reshape(n, -1),               # 12
        foot_base.reshape(n, -1),       # 12
    ], axis=1).astype(np.float32)

    if hist_signal.shape[1] != B_HISTORY_WIDTH:
        raise RuntimeError(
            f"{spec['run_id']}: history width {hist_signal.shape[1]} != {B_HISTORY_WIDTH}"
        )

    X_history = hist_signal[history_idx].astype(np.float32)

    dense_action = np.concatenate([
        U[dense_action_idx],
        L[dense_action_idx].reshape(len(origins), B_MAX_OFFSET, -1),
    ], axis=-1).astype(np.float32)

    if dense_action.shape[1:] != (B_MAX_OFFSET, B_ACTION_WIDTH):
        raise RuntimeError(
            f"{spec['run_id']}: dense action shape {dense_action.shape}"
        )

    # Reconstruct the robot's base orientation in odom and express future
    # displacement in the base frame AT THE FORECAST ORIGIN.
    T_odom_base = b_base_pose_series(spec, time_s, derived_path)
    R_odom_base_now = T_odom_base[origins, :3, :3]
    delta_odom = (
        base_position_odom[future_idx]
        - base_position_odom[origins, None, :]
    )
    delta_heading = np.einsum(
        "nji,nhj->nhi",
        R_odom_base_now,
        delta_odom,
    ).astype(np.float32)

    q = P[future_idx, :, 0:3].reshape(len(origins), B_H, 12)
    dq = P[future_idx, :, 3:6].reshape(len(origins), B_H, 12)
    tau = P[future_idx, :, 6:9].reshape(len(origins), B_H, 12)
    f = force_odom[future_idx].reshape(len(origins), B_H, 12)
    body = B[future_idx].reshape(len(origins), B_H, 12)
    e = E[future_idx].reshape(len(origins), B_H, 12)
    foot = foot_base[future_idx].reshape(len(origins), B_H, 12)

    Y_cont = np.concatenate(
        [q, dq, tau, f, body, e, foot, delta_heading],
        axis=-1,
    ).astype(np.float32)

    if Y_cont.shape[-1] != B_STEP_WIDTH:
        raise RuntimeError(
            f"{spec['run_id']}: target width {Y_cont.shape[-1]} != {B_STEP_WIDTH}"
        )

    Y_contact = contact[future_idx].astype(np.float32)

    # For two-run development we retain chronology + purge. For B9/B10 the
    # whole trajectory receives a trajectory-level role.
    if split_role is None:
        frac = origins / float(n - 1)
        split = np.full(len(origins), "purge", dtype="<U8")
        split[frac < 0.60] = "train"
        split[(frac >= 0.65) & (frac < 0.75)] = "val"
        split[frac >= 0.80] = "test"

        train_last = origins[split == "train"].max()
        val_first = origins[split == "val"].min()
        val_last = origins[split == "val"].max()
        test_first = origins[split == "test"].min()
        if (
            (val_first - train_last) <= B_MAX_OFFSET
            or (test_first - val_last) <= B_MAX_OFFSET
        ):
            raise RuntimeError(f"{spec['run_id']}: purge gap <= max horizon.")
    else:
        if split_role not in {"train", "val", "test"}:
            raise ValueError(split_role)
        split = np.full(len(origins), split_role, dtype="<U8")

    for name, arr in {
        "X_history": X_history,
        "dense_action": dense_action,
        "Y_cont": Y_cont,
        "Y_contact": Y_contact,
        "delta_heading": delta_heading,
    }.items():
        if not np.isfinite(arr).all():
            raise RuntimeError(f"{spec['run_id']}: nonfinite {name}")

    return {
        "spec": dict(spec),
        "cache_path": str(cache_path),
        "derived_path": str(derived_path),
        "time": time_s,
        "origins": origins,
        "history_idx": history_idx,
        "future_idx": future_idx,
        "dense_action_idx": dense_action_idx,
        "X_history": X_history,
        "dense_action": dense_action,
        "Y_cont": Y_cont,
        "Y_contact": Y_contact,
        "split": split,
        "visual_index_full": visual_index,
        "raw_touchdown_event": touchdown_event_raw,
        "raw_touchdown_impulse": touchdown_impulse_raw,
        "base_position_odom": base_position_odom,
    }

B_DEV_DATA = {
    spec["run_id"]: b_build_run_dataset(
        spec,
        V10R_DERIVED_PATHS[spec["run_id"]],
        split_role=None,
    )
    for spec in TWO_SPECS
}

rows = []
for rid, d in B_DEV_DATA.items():
    for s in ("train", "val", "test", "purge"):
        rows.append({
            "run_id": rid,
            "split": s,
            "samples": int(np.sum(d["split"] == s)),
            "history_shape": str(d["X_history"].shape[1:]),
            "dense_action_shape": str(d["dense_action"].shape[1:]),
            "target_shape": str(d["Y_cont"].shape[1:]),
        })
B1_DATASET_TABLE = pd.DataFrame(rows)
display(B1_DATASET_TABLE)

print("\nB1 target layout:")
for k, s in B_STEP_LAYOUT.items():
    print(f"  {k:24s} {s.start}:{s.stop}")

print("\nKEY CORRECTION:")
print("  old: absolute p_odom(t+h)")
print("  new: R_odom_base(t)^T [p_odom(t+h) - p_odom(t)]")
print("Dense future action:", B_MAX_OFFSET, "samples @", CFG.sample_hz, "Hz")


####################################################################################################
# B2
####################################################################################################


# ======================================================================
# B2 — TOUCHDOWN AS EVENT CLASSIFICATION + CONDITIONAL IMPULSE
#
# For each forecast bin (0-.1, .1-.2, ... .9-1.0 s), predict:
#   P(touchdown in bin)
# and, ONLY IF a touchdown occurs:
#   3-D impulse for that event window.
#
# This removes the misleading "mostly-zero impulse MAE".
# ======================================================================

import numpy as np
import pandas as pd
from IPython.display import display

if "B_DEV_DATA" not in globals():
    raise RuntimeError("Run B1 first.")

def b_add_touchdown_targets(d):
    origins = d["origins"]
    raw_td = np.asarray(d["raw_touchdown_event"], bool)
    raw_imp = np.asarray(d["raw_touchdown_impulse"], np.float32)

    td = np.zeros((len(origins), B_H, 4), dtype=np.float32)
    imp = np.zeros((len(origins), B_H, 4, 3), dtype=np.float32)

    prev = 0
    for h, off in enumerate(B_HORIZON_OFFSETS):
        lo = origins + prev + 1
        hi = origins + int(off) + 1

        # Vectorized per sample would be awkward because intervals move;
        # H=10 and runs are small enough for this explicit, auditable loop.
        for i in range(len(origins)):
            ev = raw_td[lo[i]:hi[i]]
            im = raw_imp[lo[i]:hi[i]]
            td[i, h] = ev.any(axis=0).astype(np.float32)

            for leg in range(4):
                ids = np.flatnonzero(ev[:, leg])
                if len(ids):
                    # More than one touchdown in a 100-ms bin is unusual.
                    # If it happens, total impulse across the bin is the
                    # physically meaningful aggregate.
                    imp[i, h, leg] = im[ids, leg].sum(axis=0)

        prev = int(off)

    d["Y_touchdown"] = td
    d["Y_impulse_cond"] = imp
    return d

for rid in list(B_DEV_DATA):
    B_DEV_DATA[rid] = b_add_touchdown_targets(B_DEV_DATA[rid])

event_rows = []
for rid, d in B_DEV_DATA.items():
    for split_name in ("train", "val", "test"):
        m = d["split"] == split_name
        td = d["Y_touchdown"][m]
        imp = d["Y_impulse_cond"][m]
        ev = td > 0.5
        norms = np.linalg.norm(imp, axis=-1)[ev]
        event_rows.append({
            "run_id": rid,
            "split": split_name,
            "forecast_bins": int(td.size),
            "positive_touchdown_bins": int(ev.sum()),
            "positive_fraction": float(ev.mean()),
            "impulse_norm_median_Ns": (
                float(np.median(norms)) if len(norms) else np.nan
            ),
            "impulse_norm_p95_Ns": (
                float(np.quantile(norms, .95)) if len(norms) else np.nan
            ),
        })

B2_TOUCHDOWN_TABLE = pd.DataFrame(event_rows)
display(B2_TOUCHDOWN_TABLE)

# Hard proof that the regression labels are zero outside events and evaluated
# conditionally later.
for rid, d in B_DEV_DATA.items():
    outside = d["Y_touchdown"] <= 0.5
    leak = np.linalg.norm(d["Y_impulse_cond"], axis=-1)[outside]
    if np.any(np.abs(leak) > 1e-8):
        raise RuntimeError(f"{rid}: nonzero conditional impulse outside touchdown.")

print("\nB2 ready:")
print("  touchdown head -> binary event in each 100-ms bin")
print("  impulse head   -> trained/evaluated only where touchdown==1")


####################################################################################################
# B3
####################################################################################################


# ======================================================================
# B3 — ACTUAL CAUSAL RGB-D INPUT
#
# A1-A6 deliberately tested robot dynamics WITHOUT learned RGB-D input.
# B3 fixes that for the final direction:
#
#   latest RGB frame at/before t  -> frozen visual semantic feature
#   synchronized metric depth     -> explicit 8x8 geometry/depth descriptor
#
# Only the frame available at the forecast origin is used. No future image.
# Front RGB-D is the DEPLOYABLE visual input for this experiment.
# ======================================================================

import os, sys, subprocess, importlib, importlib.util, inspect, gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from IPython.display import display

if "B_DEV_DATA" not in globals():
    raise RuntimeError("Run B1-B2 first.")

def _b_ensure_pkg(import_name, pip_name=None):
    if importlib.util.find_spec(import_name) is None:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", pip_name or import_name
        ])
        importlib.invalidate_caches()

_b_ensure_pkg("cv2", "opencv-python-headless")
import cv2

def b_read_depth(path):
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError("Could not read depth: " + str(path))
    if raw.dtype == np.uint8 and raw.ndim == 3 and raw.shape[2] == 4:
        raw = np.ascontiguousarray(raw)
        depth = raw.view("<f4").reshape(raw.shape[0], raw.shape[1])
    elif raw.dtype == np.float32 and raw.ndim == 2:
        depth = raw
    else:
        raise RuntimeError(
            f"Unexpected TartanGround depth encoding {raw.dtype} {raw.shape}"
        )
    depth = depth.astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    return depth

def b_download_ground_compat(**kwargs):
    """Prefer HuggingFace when supported; retry the older project API."""
    import tartanair as ta
    ta.init(str(RAW_ROOT))
    try:
        return ta.download_ground(data_source="huggingface", **kwargs)
    except TypeError:
        return ta.download_ground(**kwargs)

def b_ensure_front_rgbd(spec):
    base = raw_run_path(spec)
    required = [
        base / "image_lcam_front",
        base / "depth_lcam_front",
        base / "pose_lcam_front.txt",
        base / "imu" / "cam_time.txt",
    ]
    if all(p.exists() for p in required):
        return base

    if importlib.util.find_spec("tartanair") is None:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "git+https://github.com/castacks/tartanairpy.git"
        ])
        importlib.invalidate_caches()

    print("Downloading FRONT RGB-D + meta only:", spec["environment"], spec["run"])
    b_download_ground_compat(
        env=[spec["environment"]],
        version=["anymal"],
        traj=[spec["run"]],
        modality=["image", "depth", "meta"],
        camera_name=["lcam_front"],
        unzip=True,
    )

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError("Front RGB-D restore incomplete: " + repr(missing))
    return base

def b_manifest_for_spec(spec, time_s, cache_path):
    trajectory_id = str(spec.get("trajectory_id", f"{spec['environment']}_{spec['run']}"))
    old = (
        locate_exact_file(f"{trajectory_id}_frame_manifest.npz")
        if "locate_exact_file" in globals() else None
    )
    if old is not None:
        with np.load(old, allow_pickle=False) as z:
            out = {k: np.asarray(z[k]) for k in z.files}
        if (
            "frame_ids" in out
            and "camera_times" in out
            and "visual_index" in out
            and len(out["visual_index"]) == len(time_s)
        ):
            return out

    base = b_ensure_front_rgbd(spec)
    rgb_dir = base / "image_lcam_front"
    dep_dir = base / "depth_lcam_front"

    rgb_ids = {p.name.split("_")[0] for p in rgb_dir.glob("*_lcam_front.png")}
    dep_ids = {p.name.split("_")[0] for p in dep_dir.glob("*_lcam_front_depth.png")}
    ids = sorted(rgb_ids & dep_ids, key=lambda x: int(x))
    if not ids:
        raise RuntimeError(f"{spec['run_id']}: no paired front RGB-D frames.")

    nums = np.asarray([int(x) for x in ids], dtype=np.int64)
    cam_orig = np.loadtxt(base / "imu" / "cam_time.txt").reshape(-1).astype(np.float64)

    with np.load(cache_path, allow_pickle=False) as z:
        if "crop_start_original" in z.files:
            crop_start = float(np.asarray(z["crop_start_original"]).reshape(()))
        elif "grid_original" in z.files:
            crop_start = float(np.asarray(z["grid_original"])[0])
        else:
            raise RuntimeError(
                f"{spec['run_id']}: cache lacks crop_start_original/grid_original."
            )

    if nums.max() >= len(cam_orig):
        raise RuntimeError(f"{spec['run_id']}: frame id exceeds cam_time rows.")

    cam_t = cam_orig[nums] - crop_start
    keep = (
        (cam_t >= float(time_s[0]) - 1e-8)
        & (cam_t <= float(time_s[-1]) + 1e-8)
    )
    ids = np.asarray(ids)[keep]
    nums = nums[keep]
    cam_t = cam_t[keep]

    vi = np.searchsorted(cam_t, time_s, side="right") - 1
    if np.any((vi >= 0) & (cam_t[np.maximum(vi, 0)] > time_s + 1e-9)):
        raise RuntimeError(f"{spec['run_id']}: causal visual-index test failed.")

    return {
        "frame_ids": ids.astype("<U16"),
        "frame_nums": nums,
        "camera_times": cam_t.astype(np.float64),
        "visual_index": vi.astype(np.int64),
    }

# ----------------------------------------------------------------------
# Frozen RGB encoder.
# Prefer DINOv2-small because this project already used DINO-family features.
# Fall back to ImageNet ConvNeXt-Tiny if transformers/model retrieval fails.
# ----------------------------------------------------------------------
B_VIS_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_B_RGB_ENCODER = None

def b_get_rgb_encoder():
    global _B_RGB_ENCODER
    if _B_RGB_ENCODER is not None:
        return _B_RGB_ENCODER

    try:
        _b_ensure_pkg("transformers")
        from transformers import AutoImageProcessor, AutoModel
        proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        model = AutoModel.from_pretrained("facebook/dinov2-small").to(B_VIS_DEVICE).eval()
        for p in model.parameters():
            p.requires_grad_(False)

        def encode(images):
            x = proc(images=images, return_tensors="pt").to(B_VIS_DEVICE)
            with torch.inference_mode():
                out = model(**x)
                feat = out.last_hidden_state[:, 0]
                feat = torch.nn.functional.normalize(feat, dim=-1)
            return feat.cpu().numpy().astype(np.float32)

        dim = int(model.config.hidden_size)
        _B_RGB_ENCODER = ("dinov2-small", encode, dim)
        return _B_RGB_ENCODER

    except Exception as exc:
        print("DINOv2 load failed; using ConvNeXt-Tiny fallback:", repr(exc)[:240])
        from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        model = convnext_tiny(weights=weights).to(B_VIS_DEVICE).eval()
        model.classifier = torch.nn.Identity()
        preprocess = weights.transforms()
        for p in model.parameters():
            p.requires_grad_(False)

        def encode(images):
            xs = torch.stack([preprocess(im) for im in images]).to(B_VIS_DEVICE)
            with torch.inference_mode():
                feat = model(xs)
                feat = torch.nn.functional.normalize(feat, dim=-1)
            return feat.cpu().numpy().astype(np.float32)

        _B_RGB_ENCODER = ("convnext_tiny", encode, 768)
        return _B_RGB_ENCODER

def b_depth_descriptor(depth):
    """
    Explicit metric-depth descriptor:
      8x8 median depth = 64
      valid fraction, q10,q50,q90,std, rough-grad median/p95 = 7
    total 71.
    """
    d = np.asarray(depth, np.float32)
    valid = np.isfinite(d) & (d > 0.20) & (d < 20.0)
    clean = np.where(valid, d, np.nan)

    H0, W0 = clean.shape
    if H0 % 8 != 0 or W0 % 8 != 0:
        # deterministic resize only for unusual image size
        small = cv2.resize(
            np.where(valid, d, 0.0), (8, 8), interpolation=cv2.INTER_AREA
        )
        grid = small.astype(np.float32)
    else:
        bh, bw = H0 // 8, W0 // 8
        grid = np.nanmedian(
            clean.reshape(8, bh, 8, bw),
            axis=(1, 3),
        ).astype(np.float32)
        med = float(np.nanmedian(clean)) if np.any(valid) else 0.0
        grid[~np.isfinite(grid)] = med

    vals = d[valid]
    if len(vals):
        q10, q50, q90 = np.quantile(vals, [0.10, 0.50, 0.90])
        std = float(np.std(vals))
    else:
        q10 = q50 = q90 = std = 0.0

    gx = np.diff(d, axis=1)
    gy = np.diff(d, axis=0)
    g = np.concatenate([
        np.abs(gx[np.isfinite(gx)]).reshape(-1),
        np.abs(gy[np.isfinite(gy)]).reshape(-1),
    ])
    if len(g):
        gmed = float(np.median(g))
        gp95 = float(np.quantile(g, .95))
    else:
        gmed = gp95 = 0.0

    stats = np.asarray([
        float(valid.mean()), float(q10), float(q50), float(q90),
        float(std), gmed, gp95,
    ], dtype=np.float32)
    return np.concatenate([grid.reshape(-1), stats], axis=0)

def b_build_rgbd_feature_cache(spec, d):
    cache_path = B_VIS_ROOT / f"{spec['environment']}_{spec['run']}_front_rgbd_features.npz"
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as z:
            return {k: np.asarray(z[k]) for k in z.files}

    base = b_ensure_front_rgbd(spec)
    man = b_manifest_for_spec(spec, d["time"], Path(d["cache_path"]))
    frame_ids = np.asarray(man["frame_ids"]).astype(str)

    encoder_name, encode_rgb, rgb_dim = b_get_rgb_encoder()
    feats = []
    batch_images = []
    batch_depth = []
    batch_ids = []
    BATCH = 16

    def flush():
        if not batch_images:
            return
        rgb_feat = encode_rgb(batch_images)
        for j in range(len(batch_images)):
            dep = b_depth_descriptor(batch_depth[j])
            feats.append(np.concatenate([rgb_feat[j], dep], axis=0).astype(np.float32))
        batch_images.clear()
        batch_depth.clear()
        batch_ids.clear()

    for j, fid in enumerate(frame_ids):
        rgb_path = base / "image_lcam_front" / f"{fid}_lcam_front.png"
        dep_path = base / "depth_lcam_front" / f"{fid}_lcam_front_depth.png"
        if not rgb_path.is_file() or not dep_path.is_file():
            raise FileNotFoundError(f"Missing RGB-D pair {rgb_path} / {dep_path}")

        batch_images.append(Image.open(rgb_path).convert("RGB"))
        batch_depth.append(b_read_depth(dep_path))
        batch_ids.append(fid)

        if len(batch_images) >= BATCH:
            flush()
        if (j + 1) % 200 == 0:
            print(spec["run_id"], "RGB-D features", j + 1, "/", len(frame_ids))

    flush()
    frame_features = np.asarray(feats, np.float32)

    if len(frame_features) != len(frame_ids):
        raise RuntimeError("RGB-D feature count mismatch.")
    if not np.isfinite(frame_features).all():
        raise RuntimeError("Nonfinite RGB-D features.")

    np.savez_compressed(
        cache_path,
        frame_ids=frame_ids.astype("<U16"),
        camera_times=np.asarray(man["camera_times"], np.float64),
        visual_index=np.asarray(man["visual_index"], np.int64),
        frame_features=frame_features,
        encoder=np.asarray(encoder_name),
        rgb_dim=np.asarray(rgb_dim, np.int32),
        depth_dim=np.asarray(frame_features.shape[1] - rgb_dim, np.int32),
    )
    return {
        "frame_ids": frame_ids,
        "camera_times": np.asarray(man["camera_times"], np.float64),
        "visual_index": np.asarray(man["visual_index"], np.int64),
        "frame_features": frame_features,
        "encoder": np.asarray(encoder_name),
        "rgb_dim": np.asarray(rgb_dim, np.int32),
        "depth_dim": np.asarray(frame_features.shape[1] - rgb_dim, np.int32),
    }

def b_attach_visual_to_dataset(spec, d):
    z = b_build_rgbd_feature_cache(spec, d)
    vi_full = np.asarray(z["visual_index"], np.int64)
    if len(vi_full) != len(d["time"]):
        raise RuntimeError(f"{spec['run_id']}: visual index/time mismatch.")

    vi = vi_full[d["origins"]]
    frame_feat = np.asarray(z["frame_features"], np.float32)
    valid = (vi >= 0) & (vi < len(frame_feat))

    Xv = np.zeros((len(vi), frame_feat.shape[1] + 1), np.float32)
    Xv[valid, :-1] = frame_feat[vi[valid]]
    Xv[:, -1] = (~valid).astype(np.float32)  # explicit missingness indicator

    d["X_visual"] = Xv
    d["origin_visual_index"] = vi
    d["visual_feature_cache"] = str(
        B_VIS_ROOT / f"{spec['environment']}_{spec['run']}_front_rgbd_features.npz"
    )
    d["rgb_encoder"] = str(np.asarray(z["encoder"]).reshape(()))
    return d

for spec in TWO_SPECS:
    rid = spec["run_id"]
    B_DEV_DATA[rid] = b_attach_visual_to_dataset(spec, B_DEV_DATA[rid])

B3_VISUAL_TABLE = pd.DataFrame([
    {
        "run_id": rid,
        "encoder": d["rgb_encoder"],
        "X_visual_shape": str(d["X_visual"].shape),
        "missing_origin_fraction": float(d["X_visual"][:, -1].mean()),
        "feature_cache": d["visual_feature_cache"],
    }
    for rid, d in B_DEV_DATA.items()
])
display(B3_VISUAL_TABLE)

print("\nB3 visual contract:")
print("  MODEL INPUT = causal FRONT RGB + synchronized metric FRONT DEPTH")
print("  RGB -> frozen semantic encoder")
print("  depth -> explicit metric 8x8 geometry/depth descriptor")
print("  no future RGB-D; no six-camera privileged teacher in this model input")


####################################################################################################
# B4
####################################################################################################


# ======================================================================
# B4 — TAIL-AWARE MULTI-HEAD RGB-D WORLD MODEL
#
# Three controlled models:
#   state_only        : history only
#   dense_action      : history + 100 Hz logged future action
#   dense_action_rgbd : history + dense future action + causal RGB-D
#
# Heads:
#   continuous state distribution
#   contact probability
#   touchdown-bin probability
#   conditional touchdown impulse distribution
#
# Tail-aware weighting emphasizes high-force/high-E futures.
# ======================================================================

import copy, random, gc, json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from IPython.display import display

if "B_DEV_DATA" not in globals() or any("X_visual" not in d for d in B_DEV_DATA.values()):
    raise RuntimeError("Run B1-B3 first.")

B_SEED = 260919
random.seed(B_SEED)
np.random.seed(B_SEED)
torch.manual_seed(B_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(B_SEED)

B_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
B_BATCH = 256
B_EPOCHS = 24
B_PATIENCE = 5
B_LR = 3e-4
B_HIDDEN = 320

def b_indices_for_split(d, split_name):
    return np.flatnonzero(np.asarray(d["split"]) == split_name)

def b_stack(data_map, field, split_name):
    parts = []
    for d in data_map.values():
        ids = b_indices_for_split(d, split_name)
        if len(ids):
            parts.append(np.asarray(d[field])[ids])
    if not parts:
        raise RuntimeError(f"No {split_name} samples for field {field}")
    return np.concatenate(parts, axis=0)

def b_fit_stats(data_map):
    Xh = b_stack(data_map, "X_history", "train").astype(np.float32)
    Xa = b_stack(data_map, "dense_action", "train").astype(np.float32)
    Xv = b_stack(data_map, "X_visual", "train").astype(np.float32)
    Y = b_stack(data_map, "Y_cont", "train").astype(np.float32)
    TD = b_stack(data_map, "Y_touchdown", "train").astype(np.float32)
    IMP = b_stack(data_map, "Y_impulse_cond", "train").astype(np.float32)

    stats = {
        "hist_mean": Xh.reshape(-1, Xh.shape[-1]).mean(0).astype(np.float32),
        "hist_std": Xh.reshape(-1, Xh.shape[-1]).std(0).astype(np.float32),
        "action_mean": Xa.reshape(-1, Xa.shape[-1]).mean(0).astype(np.float32),
        "action_std": Xa.reshape(-1, Xa.shape[-1]).std(0).astype(np.float32),
        "visual_mean": Xv.mean(0).astype(np.float32),
        "visual_std": Xv.std(0).astype(np.float32),
        "y_mean": Y.reshape(-1, B_STEP_WIDTH).mean(0).astype(np.float32),
        "y_std": Y.reshape(-1, B_STEP_WIDTH).std(0).astype(np.float32),
    }
    for k in ("hist_std", "action_std", "visual_std", "y_std"):
        stats[k] = np.where(stats[k] < 1e-5, 1.0, stats[k]).astype(np.float32)

    # Conditional impulse normalization is fit only on actual touchdown bins.
    ev = TD > 0.5
    imp_vals = IMP[ev]
    if len(imp_vals) < 100:
        raise RuntimeError("Too few training touchdown impulses.")
    stats["imp_mean"] = imp_vals.mean(0).astype(np.float32)
    stats["imp_std"] = imp_vals.std(0).astype(np.float32)
    stats["imp_std"] = np.where(stats["imp_std"] < 1e-5, 1.0, stats["imp_std"]).astype(np.float32)

    # Tail thresholds are TRAIN-ONLY.
    Ystep = Y.reshape(-1, B_H, B_STEP_WIDTH)
    f = Ystep[:, :, B_STEP_LAYOUT["force_odom"]].reshape(-1, B_H, 4, 3)
    e = Ystep[:, :, B_STEP_LAYOUT["E"]].reshape(-1, B_H, 4, 3)
    stats["force_tail_N"] = float(np.quantile(np.linalg.norm(f, axis=-1), .95))
    stats["E_tail"] = float(np.quantile(np.linalg.norm(e, axis=-1), .95))

    # Positive weights for touchdown head, clipped to avoid unstable extremes.
    pos = TD.reshape(-1, 4).sum(0)
    neg = TD.reshape(-1, 4).shape[0] - pos
    stats["td_pos_weight"] = np.clip(neg / np.maximum(pos, 1.0), 1.0, 12.0).astype(np.float32)
    return stats

class BRunDataset(Dataset):
    def __init__(self, data_map, split_name, stats, balance_by_terrain=False):
        self.rows = []
        counts_by_terrain = {}
        for rid, d in data_map.items():
            ids = b_indices_for_split(d, split_name)
            terrain = str(d["spec"].get("terrain", d["spec"].get("environment", "unknown")))
            counts_by_terrain[terrain] = counts_by_terrain.get(terrain, 0) + len(ids)
            for i in ids:
                self.rows.append((rid, int(i), terrain))

        if not self.rows:
            raise RuntimeError(f"No {split_name} rows.")

        self.data_map = data_map
        self.stats = stats
        self.balance_by_terrain = bool(balance_by_terrain)
        total = sum(counts_by_terrain.values())
        nterr = max(len(counts_by_terrain), 1)
        self.terrain_weight = {
            t: total / max(nterr * c, 1)
            for t, c in counts_by_terrain.items()
        }

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, j):
        rid, i, terrain = self.rows[j]
        d = self.data_map[rid]
        s = self.stats

        xh = (d["X_history"][i] - s["hist_mean"]) / s["hist_std"]
        xa = (d["dense_action"][i] - s["action_mean"]) / s["action_std"]
        xv = (d["X_visual"][i] - s["visual_mean"]) / s["visual_std"]
        yz = (d["Y_cont"][i] - s["y_mean"]) / s["y_std"]
        c = d["Y_contact"][i].astype(np.float32)
        td = d["Y_touchdown"][i].astype(np.float32)
        impz = (d["Y_impulse_cond"][i] - s["imp_mean"]) / s["imp_std"]

        # TRAIN-ONLY thresholds, applied to this target for event/tail emphasis.
        y = d["Y_cont"][i]
        f = y[:, B_STEP_LAYOUT["force_odom"]].reshape(B_H, 4, 3)
        e = y[:, B_STEP_LAYOUT["E"]].reshape(B_H, 4, 3)
        high_f = float(np.max(np.linalg.norm(f, axis=-1)) > s["force_tail_N"])
        high_e = float(np.max(np.linalg.norm(e, axis=-1)) > s["E_tail"])
        tail_weight = 1.0 + 2.0 * high_f + 1.5 * high_e
        balance = self.terrain_weight[terrain] if self.balance_by_terrain else 1.0

        return (
            torch.from_numpy(xh.astype(np.float32)),
            torch.from_numpy(xa.astype(np.float32)),
            torch.from_numpy(xv.astype(np.float32)),
            torch.from_numpy(yz.astype(np.float32)),
            torch.from_numpy(c),
            torch.from_numpy(td),
            torch.from_numpy(impz.astype(np.float32)),
            torch.tensor(tail_weight * balance, dtype=torch.float32),
        )

class BWorldModel(nn.Module):
    def __init__(self, hist_dim, visual_dim, use_action=True, use_visual=True, hidden=B_HIDDEN):
        super().__init__()
        self.use_action = bool(use_action)
        self.use_visual = bool(use_visual)

        self.state_gru = nn.GRU(
            input_size=hist_dim,
            hidden_size=192,
            num_layers=2,
            batch_first=True,
            dropout=0.08,
        )

        if self.use_action:
            self.action_gru = nn.GRU(
                input_size=B_ACTION_WIDTH,
                hidden_size=128,
                num_layers=2,
                batch_first=True,
                dropout=0.08,
            )
        else:
            self.action_gru = None

        if self.use_visual:
            self.visual_net = nn.Sequential(
                nn.Linear(visual_dim, 256),
                nn.GELU(),
                nn.LayerNorm(256),
                nn.Dropout(.08),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.LayerNorm(128),
            )
        else:
            self.visual_net = None

        fusion_in = 192 + (128 if self.use_action else 0) + (128 if self.use_visual else 0)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(.10),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )

        self.horizon_embed = nn.Parameter(torch.randn(B_H, 64) * .02)
        self.horizon_fuse = nn.Sequential(
            nn.Linear(hidden + 64, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )

        self.mean_head = nn.Linear(hidden, B_STEP_WIDTH)
        self.log_scale_head = nn.Linear(hidden, B_STEP_WIDTH)
        self.contact_head = nn.Linear(hidden, 4)
        self.touchdown_head = nn.Linear(hidden, 4)
        self.impulse_mean_head = nn.Linear(hidden, 12)
        self.impulse_log_scale_head = nn.Linear(hidden, 12)

    def forward(self, x_hist, x_action, x_visual):
        _, hs = self.state_gru(x_hist)
        pieces = [hs[-1]]

        if self.use_action:
            _, ha = self.action_gru(x_action)
            pieces.append(ha[-1])

        if self.use_visual:
            pieces.append(self.visual_net(x_visual))

        z = self.fusion(torch.cat(pieces, dim=-1))
        zh = torch.cat([
            z[:, None, :].expand(-1, B_H, -1),
            self.horizon_embed[None, :, :].expand(len(z), -1, -1),
        ], dim=-1)
        h = self.horizon_fuse(zh)

        return {
            "mean_z": self.mean_head(h),
            "log_scale_z": torch.clamp(self.log_scale_head(h), -5.0, 2.5),
            "contact_logits": self.contact_head(h),
            "touchdown_logits": self.touchdown_head(h),
            "impulse_mean_z": self.impulse_mean_head(h).reshape(-1, B_H, 4, 3),
            "impulse_log_scale_z": torch.clamp(
                self.impulse_log_scale_head(h).reshape(-1, B_H, 4, 3),
                -5.0, 2.5
            ),
        }

def b_dimension_weights(device):
    w = torch.ones(B_STEP_WIDTH, device=device)
    w[B_STEP_LAYOUT["tau"]] = 1.20
    w[B_STEP_LAYOUT["force_odom"]] = 2.00
    w[B_STEP_LAYOUT["E"]] = 1.60
    w[B_STEP_LAYOUT["delta_base_heading"]] = 1.20
    return w

def b_loss(out, yz, contact, touchdown, impulse_z, sample_weight, td_pos_weight):
    mu = out["mean_z"]
    log_s = out["log_scale_z"]
    nll = 0.5 * (yz - mu) ** 2 * torch.exp(-2.0 * log_s) + log_s
    dimw = b_dimension_weights(yz.device)[None, None, :]
    cont_per = (nll * dimw).mean(dim=(1, 2))
    cont_loss = (cont_per * sample_weight).mean()

    contact_loss = F.binary_cross_entropy_with_logits(
        out["contact_logits"], contact
    )

    td_loss = F.binary_cross_entropy_with_logits(
        out["touchdown_logits"],
        touchdown,
        pos_weight=td_pos_weight[None, None, :],
    )

    im_mu = out["impulse_mean_z"]
    im_log_s = out["impulse_log_scale_z"]
    im_nll = (
        0.5 * (impulse_z - im_mu) ** 2 * torch.exp(-2.0 * im_log_s)
        + im_log_s
    ).mean(dim=-1)  # [B,H,4]

    mask = touchdown > 0.5
    if mask.any():
        impulse_loss = im_nll[mask].mean()
    else:
        impulse_loss = torch.zeros((), device=yz.device)

    total = (
        cont_loss
        + 0.30 * contact_loss
        + 0.45 * td_loss
        + 0.35 * impulse_loss
    )
    return total, {
        "continuous": cont_loss.detach(),
        "contact": contact_loss.detach(),
        "touchdown": td_loss.detach(),
        "impulse": impulse_loss.detach(),
    }

@torch.no_grad()
def b_eval_loss(model, loader, td_pos_weight):
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        xh, xa, xv, yz, c, td, impz, sw = [x.to(B_DEVICE) for x in batch]
        out = model(xh, xa, xv)
        loss, _ = b_loss(out, yz, c, td, impz, sw, td_pos_weight)
        total += float(loss) * len(xh)
        n += len(xh)
    return total / max(n, 1)

def b_train_one(data_map, label, use_action, use_visual, prefix="dev", balance_by_terrain=False):
    stats = b_fit_stats(data_map)
    train_ds = BRunDataset(data_map, "train", stats, balance_by_terrain=balance_by_terrain)
    val_ds = BRunDataset(data_map, "val", stats, balance_by_terrain=False)

    train_loader = DataLoader(
        train_ds, batch_size=B_BATCH, shuffle=True, num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=B_BATCH, shuffle=False, num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    sample = next(iter(train_loader))
    hist_dim = int(sample[0].shape[-1])
    visual_dim = int(sample[2].shape[-1])

    model = BWorldModel(
        hist_dim=hist_dim,
        visual_dim=visual_dim,
        use_action=use_action,
        use_visual=use_visual,
    ).to(B_DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=B_LR, weight_decay=1e-4)
    td_pos_weight = torch.as_tensor(
        stats["td_pos_weight"], dtype=torch.float32, device=B_DEVICE
    )

    best = float("inf")
    best_state = None
    bad = 0
    hist = []

    for epoch in range(1, B_EPOCHS + 1):
        model.train()
        vals = []
        for batch in train_loader:
            xh, xa, xv, yz, c, td, impz, sw = [x.to(B_DEVICE) for x in batch]
            opt.zero_grad(set_to_none=True)
            out = model(xh, xa, xv)
            loss, parts = b_loss(out, yz, c, td, impz, sw, td_pos_weight)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{label}: non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            vals.append(float(loss.detach()))

        tr = float(np.mean(vals))
        va = b_eval_loss(model, val_loader, td_pos_weight)
        hist.append({"model": label, "epoch": epoch, "train_loss": tr, "val_loss": va})
        print(f"{prefix}/{label:18s} epoch {epoch:02d} train={tr:.5f} val={va:.5f}")

        if va < best - 1e-4:
            best = va
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            bad = 0
        else:
            bad += 1
        if bad >= B_PATIENCE:
            break

    if best_state is None:
        raise RuntimeError(f"{label}: no checkpoint")

    model.load_state_dict(best_state)
    model.to(B_DEVICE).eval()

    bundle = {
        "label": label,
        "model": model,
        "use_action": bool(use_action),
        "use_visual": bool(use_visual),
        "stats": stats,
        "history": pd.DataFrame(hist),
        "best_val_loss": float(best),
    }

    ckpt = B_MODEL_ROOT / f"{prefix}_{label}.pt"
    torch.save({
        "label": label,
        "use_action": bool(use_action),
        "use_visual": bool(use_visual),
        "state_dict": best_state,
        "stats": stats,
        "hist_dim": hist_dim,
        "visual_dim": visual_dim,
        "step_width": B_STEP_WIDTH,
        "horizons_s": B_HORIZONS,
    }, ckpt)
    bundle["checkpoint"] = str(ckpt)
    return bundle

def b_train_suite(data_map, prefix="dev", balance_by_terrain=False):
    print("device:", B_DEVICE)
    suite = {}
    configs = [
        ("state_only", False, False),
        ("dense_action", True, False),
        ("dense_action_rgbd", True, True),
    ]
    for label, ua, uv in configs:
        print("\nTRAIN:", label)
        suite[label] = b_train_one(
            data_map, label, ua, uv,
            prefix=prefix,
            balance_by_terrain=balance_by_terrain,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return suite

@torch.no_grad()
def b_predict(bundle, d, split_name="test", batch_size=1024):
    ids = b_indices_for_split(d, split_name)
    if not len(ids):
        return None

    s = bundle["stats"]
    xh = ((d["X_history"][ids] - s["hist_mean"]) / s["hist_std"]).astype(np.float32)
    xa = ((d["dense_action"][ids] - s["action_mean"]) / s["action_std"]).astype(np.float32)
    xv = ((d["X_visual"][ids] - s["visual_mean"]) / s["visual_std"]).astype(np.float32)

    means, scales, cp, tdp, im, ims = [], [], [], [], [], []
    model = bundle["model"].to(B_DEVICE).eval()

    for st in range(0, len(ids), batch_size):
        xb_h = torch.from_numpy(xh[st:st+batch_size]).to(B_DEVICE)
        xb_a = torch.from_numpy(xa[st:st+batch_size]).to(B_DEVICE)
        xb_v = torch.from_numpy(xv[st:st+batch_size]).to(B_DEVICE)
        out = model(xb_h, xb_a, xb_v)

        mu_z = out["mean_z"].cpu().numpy()
        sig_z = np.exp(out["log_scale_z"].cpu().numpy())
        means.append(mu_z * s["y_std"][None, None, :] + s["y_mean"][None, None, :])
        scales.append(sig_z * s["y_std"][None, None, :])
        cp.append(torch.sigmoid(out["contact_logits"]).cpu().numpy())
        tdp.append(torch.sigmoid(out["touchdown_logits"]).cpu().numpy())

        imp_mu_z = out["impulse_mean_z"].cpu().numpy()
        imp_sig_z = np.exp(out["impulse_log_scale_z"].cpu().numpy())
        im.append(imp_mu_z * s["imp_std"][None, None, None, :] + s["imp_mean"][None, None, None, :])
        ims.append(imp_sig_z * s["imp_std"][None, None, None, :])

    return {
        "indices": ids,
        "origins": d["origins"][ids],
        "mean": np.concatenate(means),
        "scale": np.concatenate(scales),
        "contact_probability": np.concatenate(cp),
        "touchdown_probability": np.concatenate(tdp),
        "impulse_mean": np.concatenate(im),
        "impulse_scale": np.concatenate(ims),
        "truth": d["Y_cont"][ids],
        "contact_truth": d["Y_contact"][ids],
        "touchdown_truth": d["Y_touchdown"][ids],
        "impulse_truth": d["Y_impulse_cond"][ids],
    }

B_DEV_MODELS = b_train_suite(B_DEV_DATA, prefix="dev", balance_by_terrain=False)

B4_TRAINING_SUMMARY = pd.DataFrame([
    {
        "model": k,
        "best_val_loss": v["best_val_loss"],
        "checkpoint": v["checkpoint"],
    }
    for k, v in B_DEV_MODELS.items()
])
display(B4_TRAINING_SUMMARY)
B4_TRAINING_SUMMARY.to_csv(B_EVAL_ROOT / "B4_training_summary.csv", index=False)

print("\nB4 complete.")
print("Interpretation:")
print(" state_only        -> nominal extrapolation baseline")
print(" dense_action      -> value of 100 Hz future commands")
print(" dense_action_rgbd -> added value of causal RGB-D")


####################################################################################################
# B5
####################################################################################################


# ======================================================================
# B5 — TWO-RUN ABLATION + TAIL/EVENT METRICS + UNCERTAINTY CALIBRATION
#
# This is the controlled "same data, fixed formulation" experiment.
# It asks:
#   - do dense future actions help?
#   - does causal RGB-D help beyond actions?
#   - did tail/event prediction improve?
# ======================================================================

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from IPython.display import display

if "B_DEV_MODELS" not in globals():
    raise RuntimeError("Run B4 first.")

def b_group(x, name):
    return np.asarray(x)[..., B_STEP_LAYOUT[name]]

def b_mae_p95(y, p):
    e = np.abs(np.asarray(y) - np.asarray(p)).reshape(-1)
    return float(e.mean()), float(np.quantile(e, .95))

def b_binary_metrics(y, p, threshold=.5):
    y = np.asarray(y).astype(bool).reshape(-1)
    p = np.asarray(p).reshape(-1)
    pred = p >= threshold
    tp = int(np.sum(pred & y))
    fp = int(np.sum(pred & ~y))
    fn = int(np.sum(~pred & y))
    tn = int(np.sum(~pred & ~y))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    brier = float(np.mean((p - y.astype(float)) ** 2))
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "brier": brier,
        "accuracy": float((pred == y).mean()),
        "positives": int(y.sum()),
    }

B_DEV_PRED = {}
rows = []
tail_rows = []
event_rows = []

# TRAIN-ONLY thresholds shared from the RGB-D model's training stats.
ref_stats = B_DEV_MODELS["dense_action_rgbd"]["stats"]
F_TAIL = float(ref_stats["force_tail_N"])
E_TAIL = float(ref_stats["E_tail"])

for rid, d in B_DEV_DATA.items():
    B_DEV_PRED[rid] = {}
    for label, bundle in B_DEV_MODELS.items():
        pred = b_predict(bundle, d, "test")
        B_DEV_PRED[rid][label] = pred

        y = pred["truth"]
        mu = pred["mean"]

        for h, hs in enumerate(B_HORIZONS):
            for group in B_STEP_LAYOUT:
                mae, p95 = b_mae_p95(
                    b_group(y[:, h], group),
                    b_group(mu[:, h], group),
                )
                rows.append({
                    "run_id": rid,
                    "model": label,
                    "horizon_s": float(hs),
                    "group": group,
                    "mae": mae,
                    "p95": p95,
                })

        # Tail-specific force metric: condition on TRUE high-force leg/horizon.
        yf = b_group(y, "force_odom").reshape(len(y), B_H, 4, 3)
        pf = b_group(mu, "force_odom").reshape(len(y), B_H, 4, 3)
        fmask = np.linalg.norm(yf, axis=-1) >= F_TAIL
        if fmask.any():
            fe = np.abs(yf - pf)
            tail_force_mae = float(fe[fmask].mean())
            tail_force_p95 = float(np.quantile(fe[fmask], .95))
            true_peak = np.max(np.linalg.norm(yf, axis=-1), axis=(1, 2))
            pred_peak = np.max(np.linalg.norm(pf, axis=-1), axis=(1, 2))
            peak_mae = float(np.mean(np.abs(pred_peak - true_peak)))
        else:
            tail_force_mae = tail_force_p95 = peak_mae = np.nan

        ye = b_group(y, "E").reshape(len(y), B_H, 4, 3)
        pe = b_group(mu, "E").reshape(len(y), B_H, 4, 3)
        emask = np.linalg.norm(ye, axis=-1) >= E_TAIL
        if emask.any():
            ee = np.abs(ye - pe)
            tail_E_mae = float(ee[emask].mean())
            tail_E_p95 = float(np.quantile(ee[emask], .95))
        else:
            tail_E_mae = tail_E_p95 = np.nan

        tail_rows.append({
            "run_id": rid,
            "model": label,
            "force_tail_threshold_N": F_TAIL,
            "force_tail_component_mae_N": tail_force_mae,
            "force_tail_component_p95_N": tail_force_p95,
            "force_peak_norm_mae_N": peak_mae,
            "E_tail_threshold": E_TAIL,
            "E_tail_component_mae": tail_E_mae,
            "E_tail_component_p95": tail_E_p95,
        })

        tdmet = b_binary_metrics(
            pred["touchdown_truth"],
            pred["touchdown_probability"],
        )
        ev = pred["touchdown_truth"] > .5
        if ev.any():
            imp_err = np.abs(
                pred["impulse_mean"] - pred["impulse_truth"]
            )[ev]
            imp_mae = float(imp_err.mean())
            imp_p95 = float(np.quantile(imp_err, .95))

            true_norm = np.linalg.norm(pred["impulse_truth"], axis=-1)[ev]
            pred_norm = np.linalg.norm(pred["impulse_mean"], axis=-1)[ev]
            imp_norm_mae = float(np.mean(np.abs(pred_norm - true_norm)))
            imp_rel = float(np.median(
                np.abs(pred_norm - true_norm) / np.maximum(true_norm, .25)
            ))
        else:
            imp_mae = imp_p95 = imp_norm_mae = imp_rel = np.nan

        event_rows.append({
            "run_id": rid,
            "model": label,
            **{f"touchdown_{k}": v for k, v in tdmet.items()},
            "event_impulse_component_mae_Ns": imp_mae,
            "event_impulse_component_p95_Ns": imp_p95,
            "event_impulse_norm_mae_Ns": imp_norm_mae,
            "event_impulse_median_relative_error": imp_rel,
        })

B5_METRICS = pd.DataFrame(rows)
B5_TAIL_METRICS = pd.DataFrame(tail_rows)
B5_EVENT_METRICS = pd.DataFrame(event_rows)

# -------------------------- command / visual benefit --------------------------
avg = (
    B5_METRICS
    .groupby(["model", "group"], as_index=False)
    .agg(mae=("mae", "mean"), p95=("p95", "mean"))
)

pivot = avg.pivot(index="group", columns="model", values="mae")
benefit_rows = []
for group in pivot.index:
    state = float(pivot.loc[group, "state_only"])
    action = float(pivot.loc[group, "dense_action"])
    rgbd = float(pivot.loc[group, "dense_action_rgbd"])
    benefit_rows.append({
        "group": group,
        "state_only_mae": state,
        "dense_action_mae": action,
        "dense_action_rgbd_mae": rgbd,
        "dense_action_improvement_vs_state": (state - action) / max(state, 1e-12),
        "rgbd_improvement_vs_action": (action - rgbd) / max(action, 1e-12),
        "full_improvement_vs_state": (state - rgbd) / max(state, 1e-12),
    })
B5_INFORMATION_BENEFIT = pd.DataFrame(benefit_rows)

print("\nB5 nominal metrics:")
display(avg.sort_values(["group", "model"]))
print("\nCommand and RGB-D information benefit:")
display(B5_INFORMATION_BENEFIT.sort_values("full_improvement_vs_state", ascending=False))
print("\nTail metrics:")
display(B5_TAIL_METRICS)
print("\nTouchdown/event-conditioned metrics:")
display(B5_EVENT_METRICS)

# ------------------------- validation calibration -----------------------------
# Calibrate standardized |error|/predicted-scale multipliers on VALIDATION,
# never on test. This does not retrain the mean model.
rgbd_bundle = B_DEV_MODELS["dense_action_rgbd"]

val_pred = []
for rid, d in B_DEV_DATA.items():
    pr = b_predict(rgbd_bundle, d, "val")
    if pr is not None:
        val_pred.append(pr)

cal_rows = []
B_CALIBRATION_MULTIPLIER = {}

for group, sl in B_STEP_LAYOUT.items():
    ratios = []
    for pr in val_pred:
        err = np.abs(pr["truth"][..., sl] - pr["mean"][..., sl])
        sc = np.maximum(pr["scale"][..., sl], 1e-6)
        ratios.append((err / sc).reshape(-1))
    r = np.concatenate(ratios)
    r = r[np.isfinite(r)]
    B_CALIBRATION_MULTIPLIER[group] = {
        "q50": float(np.quantile(r, .50)),
        "q80": float(np.quantile(r, .80)),
        "q90": float(np.quantile(r, .90)),
        "q95": float(np.quantile(r, .95)),
    }

# Test empirical coverages using only validation-fitted multipliers.
coverage_rows = []
for rid, models in B_DEV_PRED.items():
    pr = models["dense_action_rgbd"]
    for group, sl in B_STEP_LAYOUT.items():
        y = pr["truth"][..., sl]
        mu = pr["mean"][..., sl]
        sc = np.maximum(pr["scale"][..., sl], 1e-6)
        for level, qkey in [(0.50, "q50"), (0.80, "q80"), (0.90, "q90"), (0.95, "q95")]:
            k = B_CALIBRATION_MULTIPLIER[group][qkey]
            cover = float(np.mean(np.abs(y - mu) <= k * sc))
            coverage_rows.append({
                "run_id": rid,
                "group": group,
                "nominal_coverage": level,
                "empirical_coverage": cover,
                "val_multiplier": k,
            })

B5_COVERAGE = pd.DataFrame(coverage_rows)
print("\nValidation-calibrated uncertainty coverage on TEST:")
display(
    B5_COVERAGE.groupby(["group", "nominal_coverage"], as_index=False)
    .agg(empirical_coverage=("empirical_coverage", "mean"))
)

for name, frame in {
    "B5_nominal_metrics.csv": B5_METRICS,
    "B5_tail_metrics.csv": B5_TAIL_METRICS,
    "B5_event_metrics.csv": B5_EVENT_METRICS,
    "B5_information_benefit.csv": B5_INFORMATION_BENEFIT,
    "B5_calibrated_coverage.csv": B5_COVERAGE,
}.items():
    frame.to_csv(B_EVAL_ROOT / name, index=False)

with open(B_EVAL_ROOT / "B5_calibration_multipliers.json", "w") as f:
    json.dump(B_CALIBRATION_MULTIPLIER, f, indent=2)

print("\nB5 complete. The impulse numbers above are EVENT-CONDITIONED, not zero-dominated.")


####################################################################################################
# B6
####################################################################################################


# ======================================================================
# B6 — CALIBRATED TAIL RISK, MONOTONIC UNCERTAINTY, TWO-RUN GATE
#
# Fixes A6's two main problems:
#   1) risk uses calibrated upper envelopes, not only predicted means;
#   2) uncertainty can only INCREASE risk, never make a state look safer.
# ======================================================================

import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from IPython.display import display

if "B5_COVERAGE" not in globals():
    raise RuntimeError("Run B5 first.")

def _b_bound_ratio(x, ref):
    r = np.maximum(np.asarray(x, float), 0.0) / max(float(ref), 1e-8)
    return r / (1.0 + r)

# Training-only risk reference values.
Ytr = b_stack(B_DEV_DATA, "Y_cont", "train")
TDtr = b_stack(B_DEV_DATA, "Y_touchdown", "train")
IMPtr = b_stack(B_DEV_DATA, "Y_impulse_cond", "train")

ftr = b_group(Ytr, "force_odom").reshape(-1, B_H, 4, 3)
tautr = b_group(Ytr, "tau").reshape(-1, B_H, 12)
btr = b_group(Ytr, "body_B").reshape(-1, B_H, 12)
etr = b_group(Ytr, "E").reshape(-1, B_H, 4, 3)
foottr = b_group(Ytr, "foot_base").reshape(-1, B_H, 4, 3)

B_BODY_MED = np.median(btr.reshape(-1, 12), axis=0)
B_BODY_MAD = np.median(
    np.abs(btr.reshape(-1, 12) - B_BODY_MED[None, :]), axis=0
)
B_BODY_MAD = np.maximum(B_BODY_MAD, 1e-4)

if B_H > 1:
    dh = np.diff(B_HORIZONS)
    vfoot = np.linalg.norm(
        np.diff(foottr, axis=1) / dh[None, :, None, None],
        axis=-1,
    )
else:
    vfoot = np.zeros((len(foottr), 1, 4))

impnorm = np.linalg.norm(IMPtr, axis=-1)[TDtr > .5]

B_RISK_REF = {
    "force_norm_p95_N": float(np.quantile(np.linalg.norm(ftr, axis=-1), .95)),
    "abs_tau_p95_Nm": float(np.quantile(np.abs(tautr), .95)),
    "body_robust_z_p95": float(np.quantile(
        np.abs((btr.reshape(-1, 12) - B_BODY_MED) / (1.4826 * B_BODY_MAD)), .95
    )),
    "E_norm_p95": float(np.quantile(np.linalg.norm(etr, axis=-1), .95)),
    "foot_speed_p95_mps": float(np.quantile(vfoot, .95)),
    "touchdown_impulse_norm_p95_Ns": float(
        np.quantile(impnorm, .95) if len(impnorm) else 1.0
    ),
}

def b_q95(group):
    return float(B_CALIBRATION_MULTIPLIER[group]["q95"])

def b_response_risk(
    mean, scale, contact_prob, touchdown_prob,
    impulse_mean, impulse_scale,
    truth_mode=False,
):
    mean = np.asarray(mean, float)
    scale = np.zeros_like(mean) if scale is None else np.asarray(scale, float)
    N = len(mean)

    fmu = b_group(mean, "force_odom").reshape(N, B_H, 4, 3)
    fs = b_group(scale, "force_odom").reshape(N, B_H, 4, 3)
    tmu = b_group(mean, "tau").reshape(N, B_H, 12)
    ts = b_group(scale, "tau").reshape(N, B_H, 12)
    bmu = b_group(mean, "body_B").reshape(N, B_H, 12)
    bs = b_group(scale, "body_B").reshape(N, B_H, 12)
    emu = b_group(mean, "E").reshape(N, B_H, 4, 3)
    es = b_group(scale, "E").reshape(N, B_H, 4, 3)
    fpmu = b_group(mean, "foot_base").reshape(N, B_H, 4, 3)
    fps = b_group(scale, "foot_base").reshape(N, B_H, 4, 3)

    if truth_mode:
        force_hi = np.linalg.norm(fmu, axis=-1)
        tau_hi = np.abs(tmu)
        body_hi = np.abs((bmu - B_BODY_MED[None, None, :]) / (1.4826 * B_BODY_MAD[None, None, :]))
        E_hi = np.linalg.norm(emu, axis=-1)
    else:
        force_hi = np.linalg.norm(fmu, axis=-1) + b_q95("force_odom") * np.linalg.norm(fs, axis=-1)
        tau_hi = np.abs(tmu) + b_q95("tau") * ts
        body_hi = (
            np.abs(bmu - B_BODY_MED[None, None, :])
            + b_q95("body_B") * bs
        ) / (1.4826 * B_BODY_MAD[None, None, :])
        E_hi = np.linalg.norm(emu, axis=-1) + b_q95("E") * np.linalg.norm(es, axis=-1)

    force_peak = force_hi.max(axis=(1, 2))
    tau_peak = tau_hi.max(axis=(1, 2))
    body_peak = body_hi.max(axis=(1, 2))
    E_peak = E_hi.max(axis=(1, 2))

    if B_H > 1:
        dh = np.diff(B_HORIZONS)
        vmu = np.linalg.norm(
            np.diff(fpmu, axis=1) / dh[None, :, None, None],
            axis=-1,
        )
        if truth_mode:
            foot_peak = vmu.max(axis=(1, 2))
        else:
            # Conservative uncertainty propagation for a finite difference.
            vs = (
                np.linalg.norm(fps[:, 1:], axis=-1)
                + np.linalg.norm(fps[:, :-1], axis=-1)
            ) / dh[None, :, None]
            foot_peak = (vmu + b_q95("foot_base") * vs).max(axis=(1, 2))
    else:
        foot_peak = np.zeros(N)

    im_mu_norm = np.linalg.norm(impulse_mean, axis=-1)
    if truth_mode:
        imp_upper = im_mu_norm
    else:
        imp_upper = im_mu_norm + 1.645 * np.linalg.norm(impulse_scale, axis=-1)

    # Expected severe touchdown load: event probability times upper impulse.
    imp_event_risk = np.max(touchdown_prob * imp_upper, axis=(1, 2))

    support = np.sum(contact_prob, axis=-1)
    support_shortfall = np.max(
        np.clip((2.0 - support) / 2.0, 0.0, 1.0),
        axis=1,
    )

    comp = {
        "force": _b_bound_ratio(force_peak, B_RISK_REF["force_norm_p95_N"]),
        "torque": _b_bound_ratio(tau_peak, B_RISK_REF["abs_tau_p95_Nm"]),
        "body": _b_bound_ratio(body_peak, B_RISK_REF["body_robust_z_p95"]),
        "E_response": _b_bound_ratio(E_peak, B_RISK_REF["E_norm_p95"]),
        "foot_motion": _b_bound_ratio(foot_peak, B_RISK_REF["foot_speed_p95_mps"]),
        "touchdown_impulse": _b_bound_ratio(
            imp_event_risk, B_RISK_REF["touchdown_impulse_norm_p95_Ns"]
        ),
        "support_shortfall": support_shortfall,
    }

    physical = (
        .20 * comp["force"]
        + .14 * comp["torque"]
        + .14 * comp["body"]
        + .14 * comp["E_response"]
        + .10 * comp["foot_motion"]
        + .18 * comp["touchdown_impulse"]
        + .10 * comp["support_shortfall"]
    )

    if truth_mode:
        uncertainty = np.zeros(N)
    else:
        # Dimensionless epistemic/aleatoric proxy from relative predictive scale.
        ystd = B_DEV_MODELS["dense_action_rgbd"]["stats"]["y_std"]
        rel = scale / np.maximum(ystd[None, None, :], 1e-6)
        uncertainty = np.mean(np.clip(rel, 0.0, 5.0), axis=(1, 2))
        uncertainty = uncertainty / (1.0 + uncertainty)

    # MONOTONIC: uncertainty can only add risk.
    total = np.clip(physical + 0.15 * uncertainty, 0.0, 1.0)

    return {
        **comp,
        "physical_risk": physical,
        "uncertainty_risk": uncertainty,
        "total_risk": total,
    }

B6_RISK_ROWS = []
B6_RISK_SERIES = {}

for rid, d in B_DEV_DATA.items():
    pr = B_DEV_PRED[rid]["dense_action_rgbd"]

    pred_r = b_response_risk(
        pr["mean"], pr["scale"],
        pr["contact_probability"], pr["touchdown_probability"],
        pr["impulse_mean"], pr["impulse_scale"],
        truth_mode=False,
    )
    true_r = b_response_risk(
        pr["truth"], None,
        pr["contact_truth"], pr["touchdown_truth"],
        pr["impulse_truth"], np.zeros_like(pr["impulse_truth"]),
        truth_mode=True,
    )

    rp = pred_r["total_risk"]
    ra = true_r["total_risk"]
    rho = float(spearmanr(rp, ra, nan_policy="omit").statistic)

    n_top = max(1, int(round(.10 * len(rp))))
    top_p = set(np.argpartition(rp, -n_top)[-n_top:].tolist())
    top_a = set(np.argpartition(ra, -n_top)[-n_top:].tolist())
    overlap = len(top_p & top_a) / float(n_top)

    B6_RISK_ROWS.append({
        "run_id": rid,
        "samples": len(rp),
        "spearman_pred_actual": rho,
        "top10_overlap_fraction": float(overlap),
        "pred_risk_median": float(np.median(rp)),
        "actual_risk_median": float(np.median(ra)),
    })
    B6_RISK_SERIES[rid] = {
        "pred": pred_r,
        "actual": true_r,
        "origins": pr["origins"],
    }

B6_RISK_DIAGNOSTIC = pd.DataFrame(B6_RISK_ROWS)
display(B6_RISK_DIAGNOSTIC)

mean_rho = float(B6_RISK_DIAGNOSTIC["spearman_pred_actual"].mean())
mean_top = float(B6_RISK_DIAGNOSTIC["top10_overlap_fraction"].mean())

core = {"q", "tau", "force_odom", "body_B", "foot_base"}
benefit_core = B5_INFORMATION_BENEFIT[B5_INFORMATION_BENEFIT.group.isin(core)]

B6_GATE = {
    "mean_risk_spearman": mean_rho,
    "mean_top10_overlap": mean_top,
    "mean_dense_action_benefit_core": float(
        benefit_core["dense_action_improvement_vs_state"].mean()
    ),
    "mean_rgbd_benefit_core": float(
        benefit_core["rgbd_improvement_vs_action"].mean()
    ),
    "risk_trend_gate": bool(mean_rho >= .40 and mean_top >= .20),
    "command_information_gate": bool(
        benefit_core["dense_action_improvement_vs_state"].mean() > 0.0
    ),
    "rgbd_information_gate": bool(
        benefit_core["rgbd_improvement_vs_action"].mean() > 0.0
    ),
}

print("\nB6 gate:")
print(json.dumps(B6_GATE, indent=2))

if "V10R_RISK_DIAGNOSTIC" in globals():
    print("\nA6 -> B6 comparison:")
    try:
        old_rho = float(V10R_RISK_DIAGNOSTIC["spearman_pred_actual"].mean())
        old_top = float(V10R_RISK_DIAGNOSTIC["top10_overlap_fraction"].mean())
        print(" old mean Spearman:", old_rho, " new:", mean_rho)
        print(" old top10 overlap:", old_top, " new:", mean_top)
    except Exception as exc:
        print("Could not summarize A6 comparison:", exc)

B6_RISK_DIAGNOSTIC.to_csv(B_EVAL_ROOT / "B6_risk_diagnostic.csv", index=False)
with open(B_EVAL_ROOT / "B6_gate.json", "w") as f:
    json.dump(B6_GATE, f, indent=2)

# Simple risk traces for inspection.
for rid, s in B6_RISK_SERIES.items():
    n = min(1200, len(s["pred"]["total_risk"]))
    plt.figure(figsize=(14, 4))
    plt.plot(s["actual"]["total_risk"][:n], label="actual future risk")
    plt.plot(s["pred"]["physical_risk"][:n], label="predicted physical risk")
    plt.plot(s["pred"]["total_risk"][:n], label="predicted + monotonic uncertainty")
    plt.title(f"{rid}: B6 calibrated tail-risk trace")
    plt.xlabel("test origin")
    plt.ylabel("dimensionless risk")
    plt.legend()
    plt.show()

print("\nB6 complete. Decide whether the corrected two-run formulation improved tails/risk.")


####################################################################################################
# B7
####################################################################################################


# ======================================================================
# B7 — ENUMERATE REAL ADDITIONAL ANYMAL TRAJECTORIES + DOWNLOAD FRONT RGB-D
#
# We do NOT invent P20xx IDs.
# We query the TartanGround HuggingFace repository and select only paths that
# actually exist under <environment>/Data_anymal/.
#
# Visual contract for B7-B10:
#   FRONT RGB + FRONT DEPTH + meta + IMU + ROS bag.
# We intentionally do NOT download six-camera RGB-D here.
# ======================================================================

import os, sys, re, subprocess, importlib, importlib.util, inspect, json
from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display

_b_ensure_pkg("huggingface_hub")
from huggingface_hub import HfApi

if importlib.util.find_spec("tartanair") is None:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "git+https://github.com/castacks/tartanairpy.git"
    ])
    importlib.invalidate_caches()
import tartanair as ta

B_REPO_ID = "theairlabcmu/TartanGround"
B_TARGET_ENVIRONMENTS = {
    "GreatMarsh": {
        "terrain": "swamp_marsh",
        "dev_run": "P2023",
    },
    "SeasonalForestWinter": {
        "terrain": "rough_uneven",
        "dev_run": "P2000",
    },
}
B_MAX_RUNS_PER_TERRAIN = 4
B7_DO_DOWNLOAD = True

def b_list_anymal_runs(env):
    api = HfApi()
    path = f"{env}/Data_anymal"
    runs = set()

    try:
        for entry in api.list_repo_tree(
            repo_id=B_REPO_ID,
            repo_type="dataset",
            path_in_repo=path,
            recursive=False,
            expand=False,
        ):
            name = str(entry.path).rstrip("/").split("/")[-1]
            if re.fullmatch(r"P2\d{3}", name):
                runs.add(name)
    except Exception as exc:
        print("list_repo_tree failed; fallback to list_repo_files:", repr(exc)[:240])
        files = api.list_repo_files(B_REPO_ID, repo_type="dataset")
        prefix = path + "/"
        for f in files:
            if f.startswith(prefix):
                rest = f[len(prefix):].split("/")[0]
                if re.fullmatch(r"P2\d{3}", rest):
                    runs.add(rest)

    return sorted(runs, key=lambda x: int(x[1:]))

def b_choose_trajectory_roles(env, runs, dev_run, max_runs):
    if dev_run not in runs:
        raise RuntimeError(
            f"{env}: development run {dev_run} not found in actual repository listing."
        )
    if len(runs) < 3:
        raise RuntimeError(
            f"{env}: need at least 3 real ANYmal trajectories for train/val/test; found {runs}"
        )

    others = [r for r in runs if r != dev_run]
    chosen = [dev_run] + others[:max(0, max_runs - 1)]
    if len(chosen) < 3:
        chosen = [dev_run] + others[:2]

    # Permanently untouched test/validation are NOT the already-inspected dev run.
    test_run = chosen[-1]
    val_run = chosen[-2]
    if test_run == dev_run or val_run == dev_run:
        raise RuntimeError("Split construction accidentally used dev run as val/test.")

    roles = {}
    for r in chosen:
        if r == test_run:
            roles[r] = "test"
        elif r == val_run:
            roles[r] = "val"
        else:
            roles[r] = "train"
    return roles

rows = []
B_MULTI_SPECS = []

for env, meta in B_TARGET_ENVIRONMENTS.items():
    runs = b_list_anymal_runs(env)
    print(env, "real ANYmal trajectories:", runs)

    roles = b_choose_trajectory_roles(
        env, runs, meta["dev_run"], B_MAX_RUNS_PER_TERRAIN
    )

    for run, role in roles.items():
        spec = {
            "run_id": f"B_{env}_{run}",
            "terrain": meta["terrain"],
            "environment": env,
            "run": run,
            "trajectory_id": f"{env}_{run}",
            "gait": "unreported",
            "policy": "unreported",
            "speed_bin": "unreported",
            "trajectory_role": role,
        }
        B_MULTI_SPECS.append(spec)
        rows.append({
            "terrain": meta["terrain"],
            "environment": env,
            "run": run,
            "role": role,
            "previously_inspected_dev_run": bool(run == meta["dev_run"]),
        })

B7_SPLIT_MANIFEST = pd.DataFrame(rows)

# Hard leakage guards.
if B7_SPLIT_MANIFEST.groupby(["environment", "run"]).role.nunique().max() != 1:
    raise RuntimeError("A trajectory appears in multiple roles.")
for env, g in B7_SPLIT_MANIFEST.groupby("environment"):
    if set(g.role) != {"train", "val", "test"}:
        raise RuntimeError(f"{env}: train/val/test not all populated.")
    if g.loc[g.role == "test", "previously_inspected_dev_run"].any():
        raise RuntimeError(f"{env}: inspected dev run leaked into final test.")

display(B7_SPLIT_MANIFEST)
B7_SPLIT_MANIFEST.to_csv(B_MULTI_ROOT / "trajectory_split_manifest.csv", index=False)

def b_download_selected_run(spec):
    base = raw_run_path(spec)
    required = [
        base / "imu",
        base / "rosbags" / f"{spec['run']}.bag",
        base / "image_lcam_front",
        base / "depth_lcam_front",
        base / "pose_lcam_front.txt",
    ]
    if all(p.exists() for p in required):
        print("REUSE:", spec["environment"], spec["run"])
        return base

    if not B7_DO_DOWNLOAD:
        raise RuntimeError(
            f"Missing raw run {spec['environment']}/{spec['run']} and B7_DO_DOWNLOAD=False"
        )

    ta.init(str(RAW_ROOT))
    kwargs = dict(
        env=[spec["environment"]],
        version=["anymal"],
        traj=[spec["run"]],
        modality=["imu", "rosbag", "image", "depth", "meta"],
        camera_name=["lcam_front"],
        unzip=True,
    )

    print(
        "\nDOWNLOAD:",
        spec["environment"], spec["run"], spec["trajectory_role"],
        "| front RGB-D + IMU + rosbag + meta"
    )

    # Prefer HuggingFace source when the installed tartanairpy exposes it.
    done = False
    if hasattr(ta, "download_ground_multi_thread"):
        try:
            ta.download_ground_multi_thread(
                **kwargs,
                num_workers=4,
                data_source="huggingface",
            )
            done = True
        except TypeError:
            pass
        except Exception as exc:
            print("multi-thread HuggingFace attempt failed:", repr(exc)[:240])

    if not done:
        try:
            ta.download_ground(data_source="huggingface", **kwargs)
            done = True
        except TypeError:
            ta.download_ground(**kwargs)
            done = True

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError("TartanGround download incomplete: " + repr(missing))
    return base

B7_RAW_PATHS = {}
for spec in B_MULTI_SPECS:
    B7_RAW_PATHS[spec["run_id"]] = str(b_download_selected_run(spec))

print("\nB7 complete.")
print("Selected independent trajectories:", len(B_MULTI_SPECS))
print("Final test trajectories have not been used for model-selection metrics.")


####################################################################################################
# B8
####################################################################################################


# ======================================================================
# B8 — PREPARE EVERY SELECTED TRAJECTORY FROM REAL RAW DATA
#
# Reuses the exact V7-era signal extraction contract for P/B/U/L_LEG/E and
# then re-runs the A1/A2 physical evidence path per trajectory:
#   force-topic equivalence -> TF graph -> foot trajectories -> touchdown impulse.
#
# Original raw data and original caches are never overwritten except for
# creating the normal <env>_<run>_physics.npz cache when it does not exist.
# ======================================================================

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation as SciRot
from rosbags.highlevel import AnyReader
from IPython.display import display

if "B_MULTI_SPECS" not in globals():
    raise RuntimeError("Run B7 first.")

B_CACHE_ROOT = PROJECT_ROOT / "cache"
B_CACHE_ROOT.mkdir(parents=True, exist_ok=True)

def b_stamp_to_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9

def b_causal_hold(t_src, x_src, t_target):
    t_src = np.asarray(t_src, np.float64)
    x_src = np.asarray(x_src)
    t_target = np.asarray(t_target, np.float64)
    idx = np.searchsorted(t_src, t_target, side="right") - 1
    if np.any(idx < 0):
        raise RuntimeError("Target occurs before first source sample.")
    idx = np.clip(idx, 0, len(t_src) - 1)
    return x_src[idx]

def b_extract_run_physics(spec):
    """
    Code-faithful reconstruction of the project's established V7 extractor.
    """
    env, run = spec["environment"], spec["run"]
    base = raw_run_path(spec)
    cache = B_CACHE_ROOT / f"{env}_{run}_physics.npz"

    if cache.is_file():
        with np.load(cache, allow_pickle=False) as z:
            required = {"time", "P", "B", "U", "L_LEG", "E", "contact"}
            missing = required - set(z.files)
            if not missing:
                print("REUSE physics cache:", cache)
                return cache
            print("Existing cache incomplete, rebuilding:", missing)

    imu_dir = base / "imu"
    bag_path = base / "rosbags" / f"{run}.bag"
    if not bag_path.is_file():
        raise FileNotFoundError(bag_path)

    imu_time = np.load(imu_dir / "imu_time.npy").astype(np.float64)
    acc = np.load(imu_dir / "acc.npy").astype(np.float32)
    gyro = np.loadtxt(imu_dir / "gyro.txt").astype(np.float32)

    topics = {
        "joints": "/state_estimator/joint_states",
        "lf": "/state_estimator/contact_force_lf_foot",
        "rf": "/state_estimator/contact_force_rf_foot",
        "lh": "/state_estimator/contact_force_lh_foot",
        "rh": "/state_estimator/contact_force_rh_foot",
        "twist": "/state_estimator/twist",
        "command": "/path_planning_and_following/twist",
    }

    physical_topics = {topics[k] for k in ["joints", "lf", "rf", "lh", "rh", "twist"]}
    with AnyReader([bag_path]) as reader:
        selected = [c for c in reader.connections if c.topic in physical_topics]
        if not selected:
            raise RuntimeError(f"{spec['run_id']}: no physical ROS topics.")
        _, t0_ns, _ = next(reader.messages(connections=selected))
    t0_sec = float(t0_ns) * 1e-9

    def msg_time(msg, bag_ns):
        try:
            absolute = b_stamp_to_seconds(msg.header.stamp)
            if not np.isfinite(absolute) or absolute <= 0:
                raise ValueError
        except Exception:
            absolute = float(bag_ns) * 1e-9
        return absolute - t0_sec

    data = {k: {"t": [], "x": []} for k in topics}
    expected_joint_names = [
        "LF_HAA", "LF_HFE", "LF_KFE",
        "RF_HAA", "RF_HFE", "RF_KFE",
        "LH_HAA", "LH_HFE", "LH_KFE",
        "RH_HAA", "RH_HFE", "RH_KFE",
    ]
    joint_reorder = None

    with AnyReader([bag_path]) as reader:
        selected = [c for c in reader.connections if c.topic in set(topics.values())]
        for connection, timestamp, rawdata in reader.messages(connections=selected):
            msg = reader.deserialize(rawdata, connection.msgtype)
            topic = connection.topic
            t_msg = msg_time(msg, timestamp)

            if topic == topics["joints"]:
                if joint_reorder is None:
                    names = list(msg.name)
                    missing = [n for n in expected_joint_names if n not in names]
                    if missing:
                        raise RuntimeError(
                            f"{spec['run_id']}: missing expected joints {missing}"
                        )
                    joint_reorder = np.asarray(
                        [names.index(n) for n in expected_joint_names], dtype=int
                    )
                pos = np.asarray(msg.position, np.float32)[joint_reorder]
                vel = np.asarray(msg.velocity, np.float32)[joint_reorder]
                eff = np.asarray(msg.effort, np.float32)[joint_reorder]
                x = np.concatenate([pos, vel, eff])
                key = "joints"

            elif topic in {topics["lf"], topics["rf"], topics["lh"], topics["rh"]}:
                x = np.asarray([
                    msg.wrench.force.x,
                    msg.wrench.force.y,
                    msg.wrench.force.z,
                ], np.float32)
                key = (
                    "lf" if topic == topics["lf"] else
                    "rf" if topic == topics["rf"] else
                    "lh" if topic == topics["lh"] else "rh"
                )

            elif topic == topics["twist"]:
                tw = msg.twist.twist
                x = np.asarray([
                    tw.linear.x, tw.linear.y, tw.linear.z,
                    tw.angular.x, tw.angular.y, tw.angular.z,
                ], np.float32)
                key = "twist"

            elif topic == topics["command"]:
                tw = msg.twist
                x = np.asarray([
                    tw.linear.x, tw.linear.y, tw.linear.z,
                    tw.angular.x, tw.angular.y, tw.angular.z,
                ], np.float32)
                key = "command"
            else:
                continue

            data[key]["t"].append(t_msg)
            data[key]["x"].append(x)

    for k in data:
        data[k]["t"] = np.asarray(data[k]["t"], np.float64)
        data[k]["x"] = np.asarray(data[k]["x"], np.float32)
        if len(data[k]["t"]) == 0:
            raise RuntimeError(f"{spec['run_id']}: empty stream {k}")
        o = np.argsort(data[k]["t"])
        data[k]["t"] = data[k]["t"][o]
        data[k]["x"] = data[k]["x"][o]

    stream_keys = ["joints", "lf", "rf", "lh", "rh", "twist", "command"]
    common_start = max(imu_time[0], *[data[k]["t"][0] for k in stream_keys])
    common_end = min(imu_time[-1], *[data[k]["t"][-1] for k in stream_keys])
    grid_mask = (imu_time >= common_start) & (imu_time <= common_end)
    grid_original = imu_time[grid_mask]

    if len(grid_original) < 1000:
        raise RuntimeError(f"{spec['run_id']}: too little common data.")

    acc_sync = acc[grid_mask]
    gyro_sync = gyro[grid_mask]
    joint_sync = b_causal_hold(data["joints"]["t"], data["joints"]["x"], grid_original)
    lf = b_causal_hold(data["lf"]["t"], data["lf"]["x"], grid_original)
    rf = b_causal_hold(data["rf"]["t"], data["rf"]["x"], grid_original)
    lh = b_causal_hold(data["lh"]["t"], data["lh"]["x"], grid_original)
    rh = b_causal_hold(data["rh"]["t"], data["rh"]["x"], grid_original)
    body_twist = b_causal_hold(data["twist"]["t"], data["twist"]["x"], grid_original)
    U = b_causal_hold(data["command"]["t"], data["command"]["x"], grid_original).astype(np.float32)

    q = joint_sync[:, 0:12]
    dq = joint_sync[:, 12:24]
    tau = joint_sync[:, 24:36]
    legs = [[0,1,2], [3,4,5], [6,7,8], [9,10,11]]
    force_list = [lf, rf, lh, rh]
    P = np.stack([
        np.concatenate(
            [q[:, idx], dq[:, idx], tau[:, idx], force_list[j]], axis=1
        )
        for j, idx in enumerate(legs)
    ], axis=1).astype(np.float32)
    B = np.concatenate([acc_sync, gyro_sync, body_twist], axis=1).astype(np.float32)

    # Low-level actuator commands: exact field contract used by the project.
    ACT = "/state_estimator/actuator_readings"
    cmd_t, cmd_tau, cmd_mode, st_t, st_pos = [], [], [], [], []

    with AnyReader([bag_path]) as reader:
        conns = [c for c in reader.connections if c.topic == ACT]
        if not conns:
            raise RuntimeError(f"{spec['run_id']}: {ACT} missing.")
        for c, bag_ns, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, c.msgtype)
            readings = list(msg.readings)
            if len(readings) != 12:
                continue
            cs, ct, cm, ss, sp = [], [], [], [], []
            for r in readings:
                cs.append(b_stamp_to_seconds(r.commanded.header.stamp))
                ct.append(float(r.commanded.joint_torque))
                cm.append(int(r.commanded.mode))
                ss.append(b_stamp_to_seconds(r.state.header.stamp))
                sp.append(float(r.state.joint_position))
            cmd_t.append(float(np.median(cs)) - t0_sec)
            cmd_tau.append(ct)
            cmd_mode.append(cm)
            st_t.append(float(np.median(ss)) - t0_sec)
            st_pos.append(sp)

    cmd_t = np.asarray(cmd_t, np.float64)
    cmd_tau = np.asarray(cmd_tau, np.float32)
    cmd_mode = np.asarray(cmd_mode, np.int32)
    st_t = np.asarray(st_t, np.float64)
    st_pos = np.asarray(st_pos, np.float32)

    if min(len(cmd_t), len(st_t)) == 0:
        raise RuntimeError(f"{spec['run_id']}: actuator stream empty.")

    o = np.argsort(cmd_t)
    cmd_t, cmd_tau, cmd_mode = cmd_t[o], cmd_tau[o], cmd_mode[o]
    o = np.argsort(st_t)
    st_t, st_pos = st_t[o], st_pos[o]

    valid_start = max(grid_original[0], cmd_t[0], st_t[0])
    keep = grid_original >= valid_start
    grid_original = grid_original[keep]
    P, B, U = P[keep], B[keep], U[keep]

    act_pos_sync = b_causal_hold(st_t, st_pos, grid_original)
    qcan = P[:, :, 0:3].reshape(len(P), 12)
    m = min(5000, len(P))
    cost = np.zeros((12, 12), np.float64)
    for ai in range(12):
        for ji in range(12):
            diff = act_pos_sync[:m, ai] - qcan[:m, ji]
            scale = np.std(qcan[:m, ji]) + 1e-6
            cost[ai, ji] = np.sqrt(np.mean(diff * diff)) / scale

    rows, cols = linear_sum_assignment(cost)
    actuator_for_joint = np.empty(12, dtype=int)
    for ai, ji in zip(rows, cols):
        actuator_for_joint[ji] = ai

    cmd_tau_sync = b_causal_hold(cmd_t, cmd_tau, grid_original)
    cmd_mode_sync = b_causal_hold(cmd_t, cmd_mode, grid_original)
    L = cmd_tau_sync[:, actuator_for_joint].astype(np.float32)
    L_MODE = cmd_mode_sync[:, actuator_for_joint]
    unsupported = ~np.isin(L_MODE, [10, 12, 13])
    if float(unsupported.mean()) > .01:
        raise RuntimeError(
            f"{spec['run_id']}: >1% actuator samples outside torque-containing modes."
        )

    L_LEG = L.reshape(len(L), 4, 3)
    measured_tau = P[:, :, 6:9]
    E = np.zeros_like(measured_tau, np.float32)
    E[0] = measured_tau[0] - L_LEG[0]
    E[1:] = measured_tau[1:] - L_LEG[:-1]

    crop_start_original = float(grid_original[0])
    time_s = (grid_original - crop_start_original).astype(np.float64)
    contact = (
        np.linalg.norm(P[:, :, 9:12], axis=-1) > float(CONTACT_ON_N)
    ).astype(np.float32)

    out = dict(
        time=time_s,
        grid_original=grid_original.astype(np.float64),
        crop_start_original=np.asarray(crop_start_original),
        P=P, B=B, U=U, L_LEG=L_LEG, E=E, contact=contact,
        actuator_for_joint=actuator_for_joint,
    )

    for k in ["P", "B", "U", "L_LEG", "E", "contact"]:
        if not np.isfinite(out[k]).all():
            raise RuntimeError(f"{spec['run_id']}: nonfinite {k}")

    np.savez_compressed(cache, **out)
    print("WROTE physics cache:", cache, "N=", len(time_s))
    return cache

def b_build_simple_manifest(spec, cache_path):
    """
    Causal frame manifest. No guessed ground mask is added.
    B3 derives its own actual depth descriptor from the raw metric depth.
    """
    env, run = spec["environment"], spec["run"]
    out_path = B_CACHE_ROOT / f"{env}_{run}_frame_manifest.npz"

    with np.load(cache_path, allow_pickle=False) as z:
        time_s = np.asarray(z["time"], np.float64)
        crop_start = float(np.asarray(z["crop_start_original"]).reshape(()))

    base = raw_run_path(spec)
    rgb_dir = base / "image_lcam_front"
    dep_dir = base / "depth_lcam_front"
    rgb_ids = {p.name.split("_")[0] for p in rgb_dir.glob("*_lcam_front.png")}
    dep_ids = {p.name.split("_")[0] for p in dep_dir.glob("*_lcam_front_depth.png")}
    ids = sorted(rgb_ids & dep_ids, key=lambda x: int(x))
    if not ids:
        raise RuntimeError(f"{spec['run_id']}: no paired RGB-D.")

    nums = np.asarray([int(x) for x in ids], np.int64)
    cam_orig = np.loadtxt(base / "imu" / "cam_time.txt").reshape(-1).astype(np.float64)
    poses = np.loadtxt(base / "pose_lcam_front.txt").astype(np.float64)

    if nums.max() >= len(cam_orig) or nums.max() >= len(poses):
        raise RuntimeError(f"{spec['run_id']}: frame id outside cam_time/pose rows.")

    times = cam_orig[nums] - crop_start
    keep = (times >= time_s[0] - 1e-8) & (times <= time_s[-1] + 1e-8)
    ids = np.asarray(ids)[keep]
    nums = nums[keep]
    times = times[keep]
    poses = poses[nums]
    vi = np.searchsorted(times, time_s, side="right") - 1

    if np.any((vi >= 0) & (times[np.maximum(vi, 0)] > time_s + 1e-9)):
        raise RuntimeError(f"{spec['run_id']}: causal manifest failure.")

    np.savez_compressed(
        out_path,
        frame_ids=ids.astype("<U16"),
        frame_nums=nums,
        camera_times=times.astype(np.float64),
        poses=poses.astype(np.float64),
        visual_index=vi.astype(np.int64),
        ground_fraction=np.ones(len(ids), np.float32),
    )
    return out_path

B_MULTI_DERIVED_PATHS = {}
B_MULTI_PREP_ROWS = []

# Keep the original A1 native-stream map and extend it with the new runs.
if "V10R_NATIVE_STREAMS" not in globals():
    V10R_NATIVE_STREAMS = {}

for spec in B_MULTI_SPECS:
    print("\n" + "=" * 90)
    print("B8 PREP:", spec["run_id"], "| role:", spec["trajectory_role"])
    print("=" * 90)

    cache = b_extract_run_physics(spec)
    manifest = b_build_simple_manifest(spec, cache)

    native = collect_native_contract_streams(spec)
    force_table = force_equivalence_table(native)
    winners, resolution = summarize_force_resolution(force_table)

    if (
        not resolution["verified_for_derived_adapter"]
        or resolution["source_frame_supported_by_equivalence"] != "odom"
    ):
        display(winners)
        raise RuntimeError(
            f"{spec['run_id']}: force semantics failed the same evidence gate."
        )

    # Static robot calibration must remain the same physical base->IMU transform.
    T = np.asarray(native["T_base_imu"], float)
    Tref = np.asarray(V10R_IMU_TF_RESOLUTION["T_base_imu"], float)
    dR = Tref[:3, :3].T @ T[:3, :3]
    angle_deg = float(np.degrees(SciRot.from_matrix(dR).magnitude()))
    trans_mm = float(np.linalg.norm(Tref[:3, 3] - T[:3, 3]) * 1000.0)
    if angle_deg > 1e-3 or trans_mm > 1e-3:
        raise RuntimeError(
            f"{spec['run_id']}: base->imu static TF changed "
            f"({angle_deg} deg, {trans_mm} mm)."
        )

    V10R_NATIVE_STREAMS[spec["run_id"]] = native

    report = build_real_targets_for_run(spec)
    B_MULTI_DERIVED_PATHS[spec["run_id"]] = Path(report["derived_path"])

    B_MULTI_PREP_ROWS.append({
        "run_id": spec["run_id"],
        "terrain": spec["terrain"],
        "role": spec["trajectory_role"],
        "physics_cache": str(cache),
        "manifest": str(manifest),
        "derived_path": report["derived_path"],
        "force_corr": report["state_vs_cache_force_component_correlation"],
        "force_mae_N": report["state_vs_cache_force_component_mae_N"],
        "touchdown_events": report["touchdown_events"],
    })

B8_PREP_REPORT = pd.DataFrame(B_MULTI_PREP_ROWS)
display(B8_PREP_REPORT)
B8_PREP_REPORT.to_csv(B_MULTI_ROOT / "B8_preparation_report.csv", index=False)

print("\nB8 complete. Every selected trajectory has real physics + TF/contact targets.")


####################################################################################################
# B9
####################################################################################################


# ======================================================================
# B9 — TRAJECTORY-POOLED, TERRAIN-BALANCED TRAINING
#
# Build the SAME corrected target/action/RGB-D representation for every
# selected trajectory, but assign entire trajectories to train/val/test.
#
# CRITICAL:
#   test trajectories are PREPARED here but their prediction metrics are NOT
#   computed until B10. B9 optimization touches train + validation only.
# ======================================================================

import gc, json
import numpy as np
import pandas as pd
from IPython.display import display

if "B_MULTI_DERIVED_PATHS" not in globals():
    raise RuntimeError("Run B7-B8 first.")

B_MULTI_DATA = {}

for spec in B_MULTI_SPECS:
    rid = spec["run_id"]
    role = spec["trajectory_role"]

    print("\nBUILD MULTI-TRAJECTORY DATA:", rid, "role=", role)

    d = b_build_run_dataset(
        spec,
        B_MULTI_DERIVED_PATHS[rid],
        split_role=role,
    )
    d = b_add_touchdown_targets(d)
    d = b_attach_visual_to_dataset(spec, d)
    B_MULTI_DATA[rid] = d

    # Persist the expensive aligned tensors so a Colab reset need not repeat
    # the target construction. This is a derived artifact, not a replacement
    # for raw evidence.
    save_path = B_MULTI_ROOT / f"{rid}_aligned_model_data.npz"
    np.savez_compressed(
        save_path,
        origins=d["origins"],
        X_history=d["X_history"],
        dense_action=d["dense_action"],
        X_visual=d["X_visual"],
        Y_cont=d["Y_cont"],
        Y_contact=d["Y_contact"],
        Y_touchdown=d["Y_touchdown"],
        Y_impulse_cond=d["Y_impulse_cond"],
        split=d["split"],
        origin_visual_index=d["origin_visual_index"],
    )
    d["aligned_cache"] = str(save_path)

summary_rows = []
for rid, d in B_MULTI_DATA.items():
    summary_rows.append({
        "run_id": rid,
        "terrain": d["spec"]["terrain"],
        "environment": d["spec"]["environment"],
        "run": d["spec"]["run"],
        "role": d["spec"]["trajectory_role"],
        "samples": len(d["origins"]),
        "history": str(d["X_history"].shape[1:]),
        "dense_action": str(d["dense_action"].shape[1:]),
        "rgbd": str(d["X_visual"].shape[1:]),
        "touchdown_positive_fraction": float(d["Y_touchdown"].mean()),
    })

B9_DATASET_MANIFEST = pd.DataFrame(summary_rows)
display(B9_DATASET_MANIFEST)
B9_DATASET_MANIFEST.to_csv(B_MULTI_ROOT / "B9_dataset_manifest.csv", index=False)

# Leakage proof: one role per trajectory; test runs never appear in train/val.
for rid, d in B_MULTI_DATA.items():
    roles = set(np.unique(d["split"]).tolist())
    expected = {d["spec"]["trajectory_role"]}
    if roles != expected:
        raise RuntimeError(f"{rid}: role contamination {roles} != {expected}")

print("\nTraining trajectory-pooled suite with terrain balancing...")
B_MULTI_MODELS = b_train_suite(
    B_MULTI_DATA,
    prefix="multi",
    balance_by_terrain=True,
)

B9_TRAINING_SUMMARY = pd.DataFrame([
    {
        "model": k,
        "best_val_loss": v["best_val_loss"],
        "checkpoint": v["checkpoint"],
    }
    for k, v in B_MULTI_MODELS.items()
])
display(B9_TRAINING_SUMMARY)
B9_TRAINING_SUMMARY.to_csv(B_MULTI_ROOT / "B9_training_summary.csv", index=False)

print("\nB9 complete.")
print("No B10 test metric has been computed yet.")


####################################################################################################
# B10
####################################################################################################


# ======================================================================
# B10 — FIRST EVALUATION ON PERMANENTLY UNTOUCHED TRAJECTORIES
#
# This cell is the first place where held-out trajectory predictions are
# scored. It reports nominal dynamics, tail dynamics, touchdown/impulse,
# command benefit, RGB-D benefit, calibration, and calibrated risk ranking.
# ======================================================================

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from IPython.display import display

if "B_MULTI_MODELS" not in globals():
    raise RuntimeError("Run B9 first.")

# ----------------------------------------------------------------------
# 1. Fit uncertainty multipliers on trajectory-level VALIDATION only.
# ----------------------------------------------------------------------

def b_fit_calibration_for_suite(bundle, data_map):
    vals = [
        b_predict(bundle, d, "val")
        for d in data_map.values()
        if np.any(np.asarray(d["split"]) == "val")
    ]
    out = {}
    for group, sl in B_STEP_LAYOUT.items():
        rr = []
        for pr in vals:
            err = np.abs(pr["truth"][..., sl] - pr["mean"][..., sl])
            sc = np.maximum(pr["scale"][..., sl], 1e-6)
            rr.append((err / sc).reshape(-1))
        r = np.concatenate(rr)
        r = r[np.isfinite(r)]
        out[group] = {
            "q50": float(np.quantile(r, .50)),
            "q80": float(np.quantile(r, .80)),
            "q90": float(np.quantile(r, .90)),
            "q95": float(np.quantile(r, .95)),
        }
    return out

B10_CAL = b_fit_calibration_for_suite(
    B_MULTI_MODELS["dense_action_rgbd"],
    B_MULTI_DATA,
)

# ----------------------------------------------------------------------
# 2. First test predictions.
# ----------------------------------------------------------------------

B10_TEST_PRED = {}
nom_rows, tail_rows, event_rows = [], [], []

multi_stats = B_MULTI_MODELS["dense_action_rgbd"]["stats"]
F_TAIL_MULTI = float(multi_stats["force_tail_N"])
E_TAIL_MULTI = float(multi_stats["E_tail"])

for rid, d in B_MULTI_DATA.items():
    if d["spec"]["trajectory_role"] != "test":
        continue

    B10_TEST_PRED[rid] = {}

    for label, bundle in B_MULTI_MODELS.items():
        pr = b_predict(bundle, d, "test")
        B10_TEST_PRED[rid][label] = pr

        y, mu = pr["truth"], pr["mean"]

        for h, hs in enumerate(B_HORIZONS):
            for group in B_STEP_LAYOUT:
                mae, p95 = b_mae_p95(
                    b_group(y[:, h], group),
                    b_group(mu[:, h], group),
                )
                nom_rows.append({
                    "run_id": rid,
                    "terrain": d["spec"]["terrain"],
                    "model": label,
                    "horizon_s": float(hs),
                    "group": group,
                    "mae": mae,
                    "p95": p95,
                })

        yf = b_group(y, "force_odom").reshape(len(y), B_H, 4, 3)
        pf = b_group(mu, "force_odom").reshape(len(y), B_H, 4, 3)
        fmask = np.linalg.norm(yf, axis=-1) >= F_TAIL_MULTI
        if fmask.any():
            fe = np.abs(yf - pf)
            ft_mae = float(fe[fmask].mean())
            ft_p95 = float(np.quantile(fe[fmask], .95))
            true_peak = np.max(np.linalg.norm(yf, axis=-1), axis=(1, 2))
            pred_peak = np.max(np.linalg.norm(pf, axis=-1), axis=(1, 2))
            peak_mae = float(np.mean(np.abs(true_peak - pred_peak)))
        else:
            ft_mae = ft_p95 = peak_mae = np.nan

        ye = b_group(y, "E").reshape(len(y), B_H, 4, 3)
        pe = b_group(mu, "E").reshape(len(y), B_H, 4, 3)
        emask = np.linalg.norm(ye, axis=-1) >= E_TAIL_MULTI
        if emask.any():
            ee = np.abs(ye - pe)
            et_mae = float(ee[emask].mean())
            et_p95 = float(np.quantile(ee[emask], .95))
        else:
            et_mae = et_p95 = np.nan

        tail_rows.append({
            "run_id": rid,
            "terrain": d["spec"]["terrain"],
            "model": label,
            "force_tail_threshold_N": F_TAIL_MULTI,
            "force_tail_component_mae_N": ft_mae,
            "force_tail_component_p95_N": ft_p95,
            "force_peak_norm_mae_N": peak_mae,
            "E_tail_threshold": E_TAIL_MULTI,
            "E_tail_component_mae": et_mae,
            "E_tail_component_p95": et_p95,
        })

        tdmet = b_binary_metrics(
            pr["touchdown_truth"],
            pr["touchdown_probability"],
        )
        ev = pr["touchdown_truth"] > .5
        if ev.any():
            ie = np.abs(pr["impulse_mean"] - pr["impulse_truth"])[ev]
            imae = float(ie.mean())
            ip95 = float(np.quantile(ie, .95))
            tn = np.linalg.norm(pr["impulse_truth"], axis=-1)[ev]
            pn = np.linalg.norm(pr["impulse_mean"], axis=-1)[ev]
            inorm = float(np.mean(np.abs(pn - tn)))
            irel = float(np.median(np.abs(pn - tn) / np.maximum(tn, .25)))
        else:
            imae = ip95 = inorm = irel = np.nan

        event_rows.append({
            "run_id": rid,
            "terrain": d["spec"]["terrain"],
            "model": label,
            **{f"touchdown_{k}": v for k, v in tdmet.items()},
            "event_impulse_component_mae_Ns": imae,
            "event_impulse_component_p95_Ns": ip95,
            "event_impulse_norm_mae_Ns": inorm,
            "event_impulse_median_relative_error": irel,
        })

B10_NOMINAL = pd.DataFrame(nom_rows)
B10_TAIL = pd.DataFrame(tail_rows)
B10_EVENTS = pd.DataFrame(event_rows)

# Information benefit on independent trajectories.
avg = (
    B10_NOMINAL
    .groupby(["model", "group"], as_index=False)
    .agg(mae=("mae", "mean"), p95=("p95", "mean"))
)
pv = avg.pivot(index="group", columns="model", values="mae")
benefit = []
for g in pv.index:
    state = float(pv.loc[g, "state_only"])
    action = float(pv.loc[g, "dense_action"])
    rgbd = float(pv.loc[g, "dense_action_rgbd"])
    benefit.append({
        "group": g,
        "state_only_mae": state,
        "dense_action_mae": action,
        "dense_action_rgbd_mae": rgbd,
        "command_improvement": (state - action) / max(state, 1e-12),
        "rgbd_improvement": (action - rgbd) / max(action, 1e-12),
        "full_improvement": (state - rgbd) / max(state, 1e-12),
    })
B10_INFORMATION_BENEFIT = pd.DataFrame(benefit)

print("\nB10 independent-trajectory nominal metrics:")
display(avg.sort_values(["group", "model"]))
print("\nB10 command/RGB-D benefit:")
display(B10_INFORMATION_BENEFIT.sort_values("full_improvement", ascending=False))
print("\nB10 tail metrics:")
display(B10_TAIL)
print("\nB10 event-conditioned metrics:")
display(B10_EVENTS)

# ----------------------------------------------------------------------
# 3. Test uncertainty coverage.
# ----------------------------------------------------------------------
coverage = []
for rid, models in B10_TEST_PRED.items():
    pr = models["dense_action_rgbd"]
    for group, sl in B_STEP_LAYOUT.items():
        y, mu, sc = pr["truth"][..., sl], pr["mean"][..., sl], np.maximum(pr["scale"][..., sl], 1e-6)
        for level, key in [(0.50, "q50"), (0.80, "q80"), (0.90, "q90"), (0.95, "q95")]:
            k = B10_CAL[group][key]
            coverage.append({
                "run_id": rid,
                "group": group,
                "nominal": level,
                "empirical": float(np.mean(np.abs(y - mu) <= k * sc)),
            })
B10_COVERAGE = pd.DataFrame(coverage)
print("\nB10 calibrated test coverage:")
display(
    B10_COVERAGE.groupby(["group", "nominal"], as_index=False)
    .agg(empirical=("empirical", "mean"))
)

# ----------------------------------------------------------------------
# 4. MULTI-TRAJECTORY risk references from TRAIN ONLY.
# ----------------------------------------------------------------------
Ytr = b_stack(B_MULTI_DATA, "Y_cont", "train")
TDtr = b_stack(B_MULTI_DATA, "Y_touchdown", "train")
IMPtr = b_stack(B_MULTI_DATA, "Y_impulse_cond", "train")

ftr = b_group(Ytr, "force_odom").reshape(-1, B_H, 4, 3)
tautr = b_group(Ytr, "tau").reshape(-1, B_H, 12)
btr = b_group(Ytr, "body_B").reshape(-1, B_H, 12)
etr = b_group(Ytr, "E").reshape(-1, B_H, 4, 3)
foottr = b_group(Ytr, "foot_base").reshape(-1, B_H, 4, 3)

M_BODY_MED = np.median(btr.reshape(-1, 12), axis=0)
M_BODY_MAD = np.median(
    np.abs(btr.reshape(-1, 12) - M_BODY_MED[None, :]), axis=0
)
M_BODY_MAD = np.maximum(M_BODY_MAD, 1e-4)

if B_H > 1:
    dh = np.diff(B_HORIZONS)
    fv = np.linalg.norm(
        np.diff(foottr, axis=1) / dh[None, :, None, None], axis=-1
    )
else:
    fv = np.zeros((len(foottr), 1, 4))

impnorm = np.linalg.norm(IMPtr, axis=-1)[TDtr > .5]

M_RISK_REF = {
    "force": float(np.quantile(np.linalg.norm(ftr, axis=-1), .95)),
    "tau": float(np.quantile(np.abs(tautr), .95)),
    "body": float(np.quantile(
        np.abs((btr.reshape(-1, 12) - M_BODY_MED) / (1.4826 * M_BODY_MAD)), .95
    )),
    "E": float(np.quantile(np.linalg.norm(etr, axis=-1), .95)),
    "foot": float(np.quantile(fv, .95)),
    "impulse": float(np.quantile(impnorm, .95) if len(impnorm) else 1.0),
}

def b10_risk(pr, truth_mode=False):
    mean = pr["truth"] if truth_mode else pr["mean"]
    scale = np.zeros_like(mean) if truth_mode else pr["scale"]
    contact = pr["contact_truth"] if truth_mode else pr["contact_probability"]
    td = pr["touchdown_truth"] if truth_mode else pr["touchdown_probability"]
    im = pr["impulse_truth"] if truth_mode else pr["impulse_mean"]
    ims = np.zeros_like(im) if truth_mode else pr["impulse_scale"]

    N = len(mean)
    fm = b_group(mean, "force_odom").reshape(N, B_H, 4, 3)
    fs = b_group(scale, "force_odom").reshape(N, B_H, 4, 3)
    tm = b_group(mean, "tau").reshape(N, B_H, 12)
    ts = b_group(scale, "tau").reshape(N, B_H, 12)
    bm = b_group(mean, "body_B").reshape(N, B_H, 12)
    bs = b_group(scale, "body_B").reshape(N, B_H, 12)
    em = b_group(mean, "E").reshape(N, B_H, 4, 3)
    es = b_group(scale, "E").reshape(N, B_H, 4, 3)
    footm = b_group(mean, "foot_base").reshape(N, B_H, 4, 3)
    foots = b_group(scale, "foot_base").reshape(N, B_H, 4, 3)

    def k(g):
        return 0.0 if truth_mode else float(B10_CAL[g]["q95"])

    force = (
        np.linalg.norm(fm, axis=-1)
        + k("force_odom") * np.linalg.norm(fs, axis=-1)
    ).max(axis=(1, 2))
    tau = (
        np.abs(tm) + k("tau") * ts
    ).max(axis=(1, 2))
    body = (
        (
            np.abs(bm - M_BODY_MED[None, None, :])
            + k("body_B") * bs
        )
        / (1.4826 * M_BODY_MAD[None, None, :])
    ).max(axis=(1, 2))
    Ee = (
        np.linalg.norm(em, axis=-1)
        + k("E") * np.linalg.norm(es, axis=-1)
    ).max(axis=(1, 2))

    if B_H > 1:
        dh = np.diff(B_HORIZONS)
        vf = np.linalg.norm(
            np.diff(footm, axis=1) / dh[None, :, None, None], axis=-1
        )
        vf_unc = (
            np.linalg.norm(foots[:, 1:], axis=-1)
            + np.linalg.norm(foots[:, :-1], axis=-1)
        ) / dh[None, :, None]
        foot = (vf + k("foot_base") * vf_unc).max(axis=(1, 2))
    else:
        foot = np.zeros(N)

    iupper = np.linalg.norm(im, axis=-1)
    if not truth_mode:
        iupper = iupper + 1.645 * np.linalg.norm(ims, axis=-1)
    impulse = np.max(td * iupper, axis=(1, 2))

    support = np.sum(contact, axis=-1)
    support_short = np.max(
        np.clip((2.0 - support) / 2.0, 0.0, 1.0), axis=1
    )

    comp = {
        "force": _b_bound_ratio(force, M_RISK_REF["force"]),
        "torque": _b_bound_ratio(tau, M_RISK_REF["tau"]),
        "body": _b_bound_ratio(body, M_RISK_REF["body"]),
        "E_response": _b_bound_ratio(Ee, M_RISK_REF["E"]),
        "foot_motion": _b_bound_ratio(foot, M_RISK_REF["foot"]),
        "touchdown_impulse": _b_bound_ratio(impulse, M_RISK_REF["impulse"]),
        "support_shortfall": support_short,
    }
    physical = (
        .20*comp["force"] + .14*comp["torque"] + .14*comp["body"]
        + .14*comp["E_response"] + .10*comp["foot_motion"]
        + .18*comp["touchdown_impulse"] + .10*comp["support_shortfall"]
    )

    if truth_mode:
        unc = np.zeros(N)
    else:
        ystd = B_MULTI_MODELS["dense_action_rgbd"]["stats"]["y_std"]
        rel = scale / np.maximum(ystd[None, None, :], 1e-6)
        unc = np.mean(np.clip(rel, 0, 5), axis=(1, 2))
        unc = unc / (1.0 + unc)

    return {
        **comp,
        "physical_risk": physical,
        "uncertainty_risk": unc,
        "total_risk": np.clip(physical + .15 * unc, 0, 1),
    }

risk_rows = []
B10_RISK_SERIES = {}

for rid, models in B10_TEST_PRED.items():
    pr = models["dense_action_rgbd"]
    rp = b10_risk(pr, False)
    ra = b10_risk(pr, True)

    rho = float(spearmanr(rp["total_risk"], ra["total_risk"], nan_policy="omit").statistic)
    ntop = max(1, int(round(.10 * len(pr["truth"]))))
    ip = set(np.argpartition(rp["total_risk"], -ntop)[-ntop:].tolist())
    ia = set(np.argpartition(ra["total_risk"], -ntop)[-ntop:].tolist())
    overlap = len(ip & ia) / float(ntop)

    risk_rows.append({
        "run_id": rid,
        "terrain": B_MULTI_DATA[rid]["spec"]["terrain"],
        "spearman_pred_actual": rho,
        "top10_overlap_fraction": float(overlap),
        "pred_median": float(np.median(rp["total_risk"])),
        "actual_median": float(np.median(ra["total_risk"])),
    })
    B10_RISK_SERIES[rid] = {"pred": rp, "actual": ra}

B10_RISK = pd.DataFrame(risk_rows)
print("\nB10 independent-trajectory risk ranking:")
display(B10_RISK)

# ----------------------------------------------------------------------
# 5. Final diagnostic gate.
# ----------------------------------------------------------------------
core_groups = {"q", "tau", "force_odom", "body_B", "foot_base", "delta_base_heading"}
core_ben = B10_INFORMATION_BENEFIT[
    B10_INFORMATION_BENEFIT.group.isin(core_groups)
]

rgbd_tail = B10_TAIL[B10_TAIL.model == "dense_action_rgbd"]
action_tail = B10_TAIL[B10_TAIL.model == "dense_action"]

tail_force_gain = float(
    (
        action_tail["force_tail_component_mae_N"].mean()
        - rgbd_tail["force_tail_component_mae_N"].mean()
    )
    / max(action_tail["force_tail_component_mae_N"].mean(), 1e-12)
)

rgbd_events = B10_EVENTS[B10_EVENTS.model == "dense_action_rgbd"]

B10_FINAL_GATE = {
    "independent_test_runs": int(len(B10_TEST_PRED)),
    "mean_command_benefit_core": float(core_ben["command_improvement"].mean()),
    "mean_rgbd_benefit_core": float(core_ben["rgbd_improvement"].mean()),
    "rgbd_tail_force_gain_vs_action_only": tail_force_gain,
    "touchdown_f1_mean": float(rgbd_events["touchdown_f1"].mean()),
    "event_impulse_median_relative_error_mean": float(
        rgbd_events["event_impulse_median_relative_error"].mean()
    ),
    "risk_spearman_mean": float(B10_RISK["spearman_pred_actual"].mean()),
    "risk_top10_overlap_mean": float(B10_RISK["top10_overlap_fraction"].mean()),
}
B10_FINAL_GATE["command_information_pass"] = bool(
    B10_FINAL_GATE["mean_command_benefit_core"] > 0
)
B10_FINAL_GATE["rgbd_information_pass"] = bool(
    B10_FINAL_GATE["mean_rgbd_benefit_core"] > 0
)
B10_FINAL_GATE["risk_trend_pass"] = bool(
    B10_FINAL_GATE["risk_spearman_mean"] >= .40
    and B10_FINAL_GATE["risk_top10_overlap_mean"] >= .20
)
B10_FINAL_GATE["event_model_pass"] = bool(
    B10_FINAL_GATE["touchdown_f1_mean"] >= .75
    and B10_FINAL_GATE["event_impulse_median_relative_error_mean"] <= .25
)

print("\n" + "=" * 80)
print("B10 FINAL UNTOUCHED-TRAJECTORY DIAGNOSTIC")
print("=" * 80)
print(json.dumps(B10_FINAL_GATE, indent=2))

# ----------------------------------------------------------------------
# 6. Representative plots from each untouched trajectory.
# ----------------------------------------------------------------------
for rid, models in B10_TEST_PRED.items():
    pr = models["dense_action_rgbd"]
    y, mu = pr["truth"], pr["mean"]
    yf = b_group(y, "force_odom").reshape(len(y), B_H, 4, 3)
    pf = b_group(mu, "force_odom").reshape(len(y), B_H, 4, 3)
    ye = b_group(y, "E").reshape(len(y), B_H, 4, 3)
    pe = b_group(mu, "E").reshape(len(y), B_H, 4, 3)
    ft = b_group(y, "foot_base").reshape(len(y), B_H, 4, 3)
    fp = b_group(mu, "foot_base").reshape(len(y), B_H, 4, 3)
    rr = B10_RISK_SERIES[rid]

    n = min(800, len(y))
    # Stress window: choose largest actual 0.1-s FL Fz variation.
    fz = yf[:, 0, 0, 2]
    if len(fz) > n:
        score = np.convolve(
            np.abs(np.diff(fz, prepend=fz[0])),
            np.ones(n), mode="valid"
        )
        start = int(np.argmax(score))
    else:
        start = 0
    sl = slice(start, start+n)

    fig, ax = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(
        f"{rid} — FIRST untouched-trajectory RGB-D/action-conditioned test"
    )
    ax[0,0].plot(yf[sl,0,0,2], label="actual FL Fz")
    ax[0,0].plot(pf[sl,0,0,2], label="predicted FL Fz")
    ax[0,0].set_title("0.1 s force")
    ax[0,0].legend()

    ax[0,1].plot(np.linalg.norm(ye[sl,0], axis=(1,2)), label="actual E")
    ax[0,1].plot(np.linalg.norm(pe[sl,0], axis=(1,2)), label="predicted E")
    ax[0,1].set_title("0.1 s E response")
    ax[0,1].legend()

    ax[1,0].plot(ft[sl,0,0,2], label="actual foot z")
    ax[1,0].plot(fp[sl,0,0,2], label="predicted foot z")
    ax[1,0].set_title("0.1 s FL foot z")
    ax[1,0].legend()

    ax[1,1].plot(rr["actual"]["total_risk"][sl], label="actual future risk")
    ax[1,1].plot(rr["pred"]["total_risk"][sl], label="predicted risk")
    ax[1,1].set_title("calibrated risk ranking")
    ax[1,1].legend()
    plt.tight_layout()
    plt.show()

# Save everything.
for name, frame in {
    "B10_nominal.csv": B10_NOMINAL,
    "B10_tail.csv": B10_TAIL,
    "B10_events.csv": B10_EVENTS,
    "B10_information_benefit.csv": B10_INFORMATION_BENEFIT,
    "B10_coverage.csv": B10_COVERAGE,
    "B10_risk.csv": B10_RISK,
}.items():
    frame.to_csv(B_MULTI_ROOT / name, index=False)

with open(B_MULTI_ROOT / "B10_final_gate.json", "w") as f:
    json.dump(B10_FINAL_GATE, f, indent=2)
with open(B_MULTI_ROOT / "B10_calibration.json", "w") as f:
    json.dump(B10_CAL, f, indent=2)

print("\nB10 complete.")
print("If risk_trend_pass and event_model_pass are true, the next step is VLM semantic")
print("terrain expectation + counterfactual CEM/MPPI. If not, fix the failing tail/event head first.")
