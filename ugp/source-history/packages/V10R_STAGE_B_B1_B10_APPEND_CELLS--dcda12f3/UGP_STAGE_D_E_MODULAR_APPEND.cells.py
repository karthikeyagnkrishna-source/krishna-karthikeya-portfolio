# Source inspection export of the adjacent notebook.
# Notebook execution order and dependencies still apply.
# IPython magic, shell lines and prose-only conclusion cells are comments.

# %% notebook cell 1
# ================================================================
# D0 — FREEZE SCIENTIFIC STATE / CREATE MODULAR D-E NAMESPACE
# No training. No fresh-test loading.
# ================================================================
import gc, json, math, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch

REQ = [
    "C_ROOT", "C7R_DATA", "C7R_FRAME_FEATURES",
    "B9_DENSE", "B9_PREP", "B9_CALIBRATION",
    "B_STEP_LAYOUT", "B_STEP_WIDTH", "B_H",
    "CFG", "B_DEVICE", "TAIL_NAMES",
]
missing = [x for x in REQ if x not in globals()]
if missing:
    raise RuntimeError(
        "Run the rescue notebook through the B9/C7R restore first. "
        "Missing: " + repr(missing)
    )

if "TEST_ROWS" in globals():
    DE_LOCKED_TEST_IDS = TEST_ROWS["run_id"].astype(str).tolist()
else:
    DE_LOCKED_TEST_IDS = [
        "C_swamp_marsh_P2027",
        "C_rough_uneven_P2006",
    ]

if set(DE_LOCKED_TEST_IDS) & set(C7R_DATA):
    raise RuntimeError("Fresh C test runs leaked into C7R_DATA.")

if set(DE_LOCKED_TEST_IDS) & set(C7R_FRAME_FEATURES):
    raise RuntimeError("Fresh C test visual features leaked into D/E development data.")

# Load final C7T gate if current runtime has it; otherwise use persisted audit if present.
DE_C7T_GATE = globals().get("C7T_GATE", None)
if DE_C7T_GATE is None:
    c7t_gate_path = (
        Path(C_ROOT)
        / "spatial_temporal_repair_v3_strict_visual_interaction"
        / "tables"
        / "gate.json"
    )
    if c7t_gate_path.exists():
        DE_C7T_GATE = json.loads(c7t_gate_path.read_text())

if DE_C7T_GATE is not None:
    if bool(DE_C7T_GATE.get("open_fresh_C_test", False)):
        raise RuntimeError(
            "This D/E append assumes the final C7T causal gate did NOT pass. "
            "Unexpected persisted state."
        )

DE_ROOT = Path(C_ROOT) / "stage_DE_modular_physical_visual_v1"
D_ROOT = DE_ROOT / "D_visual_context"
E_ROOT = DE_ROOT / "E_support_constrained_planner"
for p in (DE_ROOT, D_ROOT, E_ROOT):
    p.mkdir(parents=True, exist_ok=True)

DE_TRAIN_IDS = [
    rid for rid, d in C7R_DATA.items()
    if np.any(np.asarray(d["split"]) == "train")
]
DE_VAL_IDS = [
    rid for rid, d in C7R_DATA.items()
    if np.any(np.asarray(d["split"]) == "val")
]

if len(DE_TRAIN_IDS) != 6 or len(DE_VAL_IDS) != 2:
    raise RuntimeError(
        f"Unexpected D/E development split: "
        f"train={DE_TRAIN_IDS}, val={DE_VAL_IDS}"
    )

# Keep B9 frozen.
B9_MODEL_DE = B9_DENSE["model"].eval()
for p in B9_MODEL_DE.parameters():
    p.requires_grad_(False)

# We do NOT assume image-left/right == robot-left/right yet.
D_CAMERA_BODY_HORIZONTAL_ALIGNMENT_VERIFIED = False

DE_ARCH = {
    "version": "DE_modular_physical_visual_v1",
    "physical_model": "frozen_B9",
    "visual_geometry": "existing_metric_depth_8x8x4_descriptor",
    "visual_novelty": "existing_DINOv2_lower_front_spatial_tokens",
    "semantic_model": "event_triggered_external_VLM_interface",
    "candidate_actions": "logged_training_sequences_only",
    "action_support": "nearest_B9_history_latent",
    "directional_visual_cost_enabled": False,
    "fresh_C_tests_loaded": False,
    "fresh_C_test_ids": DE_LOCKED_TEST_IDS,
    "notes": (
        "Stage C did not establish time-local RGB-D value inside the physical "
        "dynamics predictor. D/E therefore keeps B9 as the physical model and "
        "uses RGB-D only as modular geometry/novelty/semantic context."
    ),
}
(DE_ROOT / "architecture.json").write_text(json.dumps(DE_ARCH, indent=2))

print("D/E architecture frozen.")
print("Train runs:", DE_TRAIN_IDS)
print("Validation runs:", DE_VAL_IDS)
print("Fresh C tests remain LOCKED:", DE_LOCKED_TEST_IDS)
print("Directional camera/body visual cost:", D_CAMERA_BODY_HORIZONTAL_ALIGNMENT_VERIFIED)

# %% notebook cell 2
# ================================================================
# D1 — INTERPRETABLE DEPTH GEOMETRY + DINO VISUAL NOVELTY
#
# Reuses existing C7R frame features only.
# No new encoder inference. No labels. No training.
#
# depth_spatial is exactly:
#   channel 0: normalized log-depth
#   channel 1: normalized inverse-depth / proximity
#   channel 2: normalized depth gradient / roughness-edge proxy
#   channel 3: valid-depth fraction
# on the lower/front 8x8 grid.
# ================================================================
D_GRID = 8
D_RGB_REF_PER_TRAIN_RUN = 96
D_RGB_CAL_PER_TRAIN_RUN = 48
D_SEED = 78120260921 % (2**32)

def _stable_seed(text, base=D_SEED):
    h = int(hashlib.sha256(str(text).encode()).hexdigest()[:8], 16)
    return (int(base) + h) % (2**32)

def d_depth_maps_from_tab(tab):
    raw = np.asarray(tab["depth_spatial"], np.float32)
    if raw.ndim != 2 or raw.shape[1] != 4 * D_GRID * D_GRID:
        raise RuntimeError(f"Unexpected depth_spatial shape {raw.shape}")
    z = raw.reshape(-1, 4, D_GRID, D_GRID)
    return {
        "logdepth": z[:, 0],
        "proximity": z[:, 1],
        "roughness": z[:, 2],
        "validity": z[:, 3],
        "invalidity": 1.0 - z[:, 3],
    }

def d_depth_summary(tab):
    m = d_depth_maps_from_tab(tab)
    prox = m["proximity"]
    rough = m["roughness"]
    inv = m["invalidity"]

    # Three image sectors are stored for later use, but NOT used for
    # candidate ranking until camera/body horizontal alignment is verified.
    col_slices = {
        "left_img": slice(0, 3),
        "center_img": slice(3, 5),
        "right_img": slice(5, 8),
    }

    out = {
        "proximity_mean": prox.mean((1, 2)).astype(np.float32),
        "proximity_p90": np.quantile(prox, 0.90, axis=(1, 2)).astype(np.float32),
        "roughness_mean": rough.mean((1, 2)).astype(np.float32),
        "roughness_p90": np.quantile(rough, 0.90, axis=(1, 2)).astype(np.float32),
        "invalid_fraction": inv.mean((1, 2)).astype(np.float32),
    }

    for name, sl in col_slices.items():
        out[f"{name}_proximity"] = prox[:, :, sl].mean((1, 2)).astype(np.float32)
        out[f"{name}_roughness"] = rough[:, :, sl].mean((1, 2)).astype(np.float32)
        out[f"{name}_invalid"] = inv[:, :, sl].mean((1, 2)).astype(np.float32)

    return out

def _normalize_rows(x):
    x = np.asarray(x, np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-8)

# ---------------- RGB novelty reference bank: TRAIN ONLY ----------------
rgb_ref_parts = []
rgb_cal_parts = []

for rid in DE_TRAIN_IDS:
    R = _normalize_rows(
        np.asarray(C7R_FRAME_FEATURES[rid]["rgb_spatial"], np.float32)
    )
    rng = np.random.default_rng(_stable_seed(rid))
    order = rng.permutation(len(R))

    nr = min(D_RGB_REF_PER_TRAIN_RUN, max(1, len(R) // 3))
    nc = min(D_RGB_CAL_PER_TRAIN_RUN, max(1, len(R) // 6))

    rgb_ref_parts.append(R[order[:nr]])
    rgb_cal_parts.append(R[order[nr:nr+nc]])

D_RGB_REF = _normalize_rows(np.concatenate(rgb_ref_parts, axis=0))
D_RGB_CAL = _normalize_rows(np.concatenate(rgb_cal_parts, axis=0))

def d_rgb_novelty(features, ref=D_RGB_REF, batch=256):
    """
    1 - maximum cosine similarity to frozen TRAIN visual reference bank.
    Higher = more visually novel relative to development training data.
    """
    q = _normalize_rows(features)
    ref_t = torch.as_tensor(ref, dtype=torch.float32, device=B_DEVICE)
    vals = []

    for st in range(0, len(q), batch):
        qt = torch.as_tensor(q[st:st+batch], dtype=torch.float32, device=B_DEVICE)
        sim = qt @ ref_t.T
        vals.append((1.0 - sim.max(dim=1).values).cpu().numpy())

    return np.concatenate(vals).astype(np.float32)

D_RGB_NOVELTY_THR = float(
    np.quantile(d_rgb_novelty(D_RGB_CAL), 0.95)
)

# ---------------- Geometry outlier thresholds: TRAIN ONLY ----------------
_train_geo = []
for rid in DE_TRAIN_IDS:
    s = d_depth_summary(C7R_FRAME_FEATURES[rid])
    _train_geo.append(
        np.stack(
            [
                s["proximity_p90"],
                s["roughness_p90"],
                s["invalid_fraction"],
            ],
            axis=1,
        )
    )

_train_geo = np.concatenate(_train_geo, axis=0)
D_GEO_THRESHOLDS = {
    "proximity_p90": float(np.quantile(_train_geo[:, 0], 0.95)),
    "roughness_p90": float(np.quantile(_train_geo[:, 1], 0.95)),
    "invalid_fraction": float(np.quantile(_train_geo[:, 2], 0.95)),
}

# ---------------- Build cached D visual state for DEV ONLY ----------------
D_VISUAL_STATE = {}

for rid in list(C7R_FRAME_FEATURES):
    if rid in DE_LOCKED_TEST_IDS:
        raise RuntimeError("Locked test reached D visual loop.")

    tab = C7R_FRAME_FEATURES[rid]
    geo = d_depth_summary(tab)
    novelty = d_rgb_novelty(
        np.asarray(tab["rgb_spatial"], np.float32)
    )

    trigger = (
        (novelty > D_RGB_NOVELTY_THR)
        | (geo["proximity_p90"] > D_GEO_THRESHOLDS["proximity_p90"])
        | (geo["roughness_p90"] > D_GEO_THRESHOLDS["roughness_p90"])
        | (geo["invalid_fraction"] > D_GEO_THRESHOLDS["invalid_fraction"])
    )

    state = {
        "frame_ids": np.asarray(tab["frame_ids"]).astype(str),
        "camera_times": np.asarray(tab["camera_times"], np.float64),
        **geo,
        "rgb_novelty": novelty,
        "semantic_trigger": trigger.astype(np.uint8),
    }
    D_VISUAL_STATE[rid] = state

    np.savez_compressed(
        D_ROOT / f"{rid}_visual_context.npz",
        **state,
    )

D_THRESH = {
    "rgb_novelty_p95_train": D_RGB_NOVELTY_THR,
    **{f"{k}_p95_train": v for k, v in D_GEO_THRESHOLDS.items()},
}
(D_ROOT / "visual_thresholds.json").write_text(
    json.dumps(D_THRESH, indent=2)
)

def d_origin_visual_context(rid, origin_positions):
    d = C7R_DATA[rid]
    p = np.asarray(origin_positions, np.int64)
    vi = np.asarray(d["visual_frame_index"][p], np.int64)
    s = D_VISUAL_STATE[rid]

    return pd.DataFrame({
        "origin_pos": p,
        "origin_index": d["origins"][p],
        "frame_index": vi,
        "frame_id": s["frame_ids"][vi],
        "rgb_novelty": s["rgb_novelty"][vi],
        "proximity_p90": s["proximity_p90"][vi],
        "roughness_p90": s["roughness_p90"][vi],
        "invalid_fraction": s["invalid_fraction"][vi],
        "semantic_trigger": s["semantic_trigger"][vi].astype(bool),
    })

print("D1 complete.")
print("Visual novelty threshold:", D_RGB_NOVELTY_THR)
print("Geometry thresholds:", D_GEO_THRESHOLDS)
print("No labels or fresh tests were used.")

# %% notebook cell 3
# ================================================================
# D2 — EVENT-TRIGGERED VLM SEMANTIC INTERFACE
#
# This does NOT run a heavy VLM in the main loop.
# It creates a strict structured request only when D1 says the
# scene is novel / geometrically suspicious.
#
# You can later connect ANY chosen VLM and return the JSON schema
# below. The planner will consume the semantic risk only after
# camera/body directional alignment is audited.
# ================================================================
D_SEMANTIC_SCHEMA = {
    "terrain_material": "short string",
    "water_probability": "float 0..1",
    "mud_soft_probability": "float 0..1",
    "snow_ice_probability": "float 0..1",
    "loose_gravel_probability": "float 0..1",
    "rock_obstacle_probability": "float 0..1",
    "dense_vegetation_probability": "float 0..1",
    "slipperiness": "float 0..1",
    "deformability": "float 0..1",
    "obstacle_severity": "float 0..1",
    "confidence": "float 0..1",
    "reason_short": "one short sentence",
}

D_VLM_PROMPT = """
You are a terrain-risk perception module for an outdoor legged robot.
Use ONLY what is visible in the supplied front RGB image.

Return JSON only with exactly these fields:
terrain_material,
water_probability,
mud_soft_probability,
snow_ice_probability,
loose_gravel_probability,
rock_obstacle_probability,
dense_vegetation_probability,
slipperiness,
deformability,
obstacle_severity,
confidence,
reason_short.

All probabilities/severity values must be in [0,1].
Do not infer hidden terrain properties with certainty from appearance alone.
Use lower confidence when appearance is ambiguous.
"""

def d_make_vlm_request(rid, origin_pos):
    row = d_origin_visual_context(rid, [int(origin_pos)]).iloc[0].to_dict()
    request = {
        "run_id": rid,
        "origin_pos": int(origin_pos),
        "origin_index": int(row["origin_index"]),
        "frame_id": str(row["frame_id"]),
        "triggered": bool(row["semantic_trigger"]),
        "visual_diagnostics": {
            "rgb_novelty": float(row["rgb_novelty"]),
            "proximity_p90": float(row["proximity_p90"]),
            "roughness_p90": float(row["roughness_p90"]),
            "invalid_fraction": float(row["invalid_fraction"]),
        },
        "prompt": D_VLM_PROMPT.strip(),
        "schema": D_SEMANTIC_SCHEMA,
    }

    out = D_ROOT / f"vlm_request_{rid}_origin{int(origin_pos)}.json"
    out.write_text(json.dumps(request, indent=2))
    return request, out

def d_validate_vlm_response(resp):
    numeric = [
        "water_probability",
        "mud_soft_probability",
        "snow_ice_probability",
        "loose_gravel_probability",
        "rock_obstacle_probability",
        "dense_vegetation_probability",
        "slipperiness",
        "deformability",
        "obstacle_severity",
        "confidence",
    ]
    for k in numeric:
        if k not in resp:
            raise ValueError(f"Missing VLM field: {k}")
        v = float(resp[k])
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{k} outside [0,1]: {v}")

    # Conservative scalar semantic hazard; confidence does not erase hazard.
    hazard = max(
        float(resp["water_probability"]),
        float(resp["mud_soft_probability"]),
        float(resp["snow_ice_probability"]),
        float(resp["loose_gravel_probability"]),
        float(resp["rock_obstacle_probability"]),
        float(resp["dense_vegetation_probability"]),
        float(resp["slipperiness"]),
        float(resp["deformability"]),
        float(resp["obstacle_severity"]),
    )
    return float(np.clip(hazard, 0.0, 1.0))

print("D2 semantic interface ready.")
print("No VLM inference has been run automatically.")

# %% notebook cell 4
# ================================================================
# E0 — SUPPORT-CONSTRAINED CANDIDATE BANK + FROZEN B9 ROLLOUT
#
# Candidate actions are NOT invented by Gaussian/CEM perturbations yet.
# Every candidate is a REAL 1-second dense future action sequence from
# the six training runs, retrieved near the current B9 history latent.
#
# This avoids pretending B9 is valid far outside logged action support.
# ================================================================
E_SEED = 78120260921 % (2**32)
E_BANK_PER_RUN = 256
E_CAL_PER_RUN = 64
E_DEFAULT_K = 64

E_DEVICE = B_DEVICE
E_USE_AMP = torch.cuda.is_available()

E_NORM_T = {
    "x_mean": torch.as_tensor(B9_PREP["x_mean"], dtype=torch.float32, device=E_DEVICE),
    "x_std": torch.as_tensor(B9_PREP["x_std"], dtype=torch.float32, device=E_DEVICE),
    "action_mean": torch.as_tensor(B9_PREP["action_mean"], dtype=torch.float32, device=E_DEVICE),
    "action_std": torch.as_tensor(B9_PREP["action_std"], dtype=torch.float32, device=E_DEVICE),
}

def e_encode_history(X, batch=512):
    X = np.asarray(X, np.float32)
    m = B9_MODEL_DE.to(E_DEVICE).eval()
    out = []
    with torch.no_grad():
        for st in range(0, len(X), batch):
            x = torch.from_numpy(X[st:st+batch]).to(E_DEVICE)
            x = (x - E_NORM_T["x_mean"][None, :]) / E_NORM_T["x_std"][None, :]
            h = m.history(x)
            h = torch.nn.functional.normalize(h.float(), dim=-1)
            out.append(h.cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)

@torch.no_grad()
def e_predict_arbitrary(X_core, action_dense, batch=256):
    X_core = np.asarray(X_core, np.float32)
    action_dense = np.asarray(action_dense, np.float32)

    if len(X_core) != len(action_dense):
        raise ValueError("X_core/action_dense length mismatch")
    if action_dense.shape[1:] != (100, 18):
        raise ValueError(f"Expected candidate actions [N,100,18], got {action_dense.shape}")

    m = B9_MODEL_DE.to(E_DEVICE).eval()
    S = {k: [] for k in ["mz", "sz", "c", "td", "im", "isc", "tail"]}

    for st in range(0, len(X_core), batch):
        x = torch.from_numpy(X_core[st:st+batch]).to(E_DEVICE)
        a = torch.from_numpy(action_dense[st:st+batch]).to(E_DEVICE)

        x = (x - E_NORM_T["x_mean"][None, :]) / E_NORM_T["x_std"][None, :]
        a = (
            a - E_NORM_T["action_mean"][None, None, :]
        ) / E_NORM_T["action_std"][None, None, :]

        with torch.autocast(
            "cuda",
            dtype=torch.float16,
            enabled=E_USE_AMP,
        ):
            o = m(x, a)

        S["mz"].append(o["mean_z"].float().cpu().numpy())
        S["sz"].append(torch.exp(o["log_scale_z"].float()).cpu().numpy())
        S["c"].append(torch.sigmoid(o["contact_logits"].float()).cpu().numpy())
        S["td"].append(torch.sigmoid(o["touchdown_logits"].float()).cpu().numpy())
        S["im"].append(o["impulse_mean_z"].float().cpu().numpy())
        S["isc"].append(torch.exp(o["impulse_log_scale_z"].float()).cpu().numpy())
        S["tail"].append(o["tail_q90_log"].float().cpu().numpy())

    q = {k: np.concatenate(v, axis=0) for k, v in S.items()}

    return {
        "continuous_mean": (
            q["mz"] * B9_PREP["y_std"][None, :, :]
            + B9_PREP["y_mean"][None, :, :]
        ).astype(np.float32),
        "continuous_scale": (
            q["sz"] * B9_PREP["y_std"][None, :, :]
        ).astype(np.float32),
        "contact_probability": q["c"].astype(np.float32),
        "touchdown_probability": q["td"].astype(np.float32),
        "impulse_mean": (
            q["im"] * B9_PREP["imp_std"][None, None, None, :]
            + B9_PREP["imp_mean"][None, None, None, :]
        ).astype(np.float32),
        "impulse_scale": (
            q["isc"] * B9_PREP["imp_std"][None, None, None, :]
        ).astype(np.float32),
        "tail_q90": np.maximum(np.expm1(q["tail"]), 0.0).astype(np.float32),
    }

def _de_ratio(x, ref):
    r = np.maximum(np.asarray(x, float), 0.0) / max(float(ref), 1e-8)
    return r / (1.0 + r)

def e_predicted_risk(pred):
    """
    Exact B9 calibrated physical-risk construction used in the C-series,
    copied locally so D/E does not depend on a mutable C helper.
    """
    mu = np.asarray(pred["continuous_mean"], float)
    scale = np.asarray(pred["continuous_scale"], float)
    n = len(mu)
    refs = B9_CALIBRATION["risk_reference"]
    z90 = 1.645
    dt = np.diff(np.asarray(CFG.horizons_s, float))

    def group(x, name):
        return x[:, :, B_STEP_LAYOUT[name]]

    def gs(name):
        return group(scale, name) * float(B9_CALIBRATION["scale_factor"][name])

    fmu = group(mu, "force_odom").reshape(n, B_H, 4, 3)
    fsc = gs("force_odom").reshape(n, B_H, 4, 3)
    Emu = group(mu, "E").reshape(n, B_H, 4, 3)
    Esc = gs("E").reshape(n, B_H, 4, 3)
    tmu = group(mu, "tau").reshape(n, B_H, 12)
    tsc = gs("tau").reshape(n, B_H, 12)
    bmu = group(mu, "body_B").reshape(n, B_H, 12)
    bsc = gs("body_B").reshape(n, B_H, 12)
    foot = group(mu, "foot_base").reshape(n, B_H, 4, 3)

    fup = np.max(
        np.linalg.norm(fmu, axis=-1)
        + z90 * np.linalg.norm(fsc, axis=-1),
        axis=(1, 2),
    )
    Eup = np.max(
        np.linalg.norm(Emu, axis=-1)
        + z90 * np.linalg.norm(Esc, axis=-1),
        axis=(1, 2),
    )
    tup = np.max(np.abs(tmu) + z90 * tsc, axis=(1, 2))
    bup = np.max(
        (
            np.abs(bmu - B9_PREP["body_med"][None, None, :])
            + z90 * bsc
        )
        / np.maximum(
            1.4826 * B9_PREP["body_mad"][None, None, :],
            1e-5,
        ),
        axis=(1, 2),
    )

    q = np.asarray(pred["tail_q90"], float).copy()
    for j, name in enumerate(TAIL_NAMES):
        q[:, j] += B9_CALIBRATION["tail_add"][name]

    ft = np.maximum(fup, q[:, 0])
    Et = np.maximum(Eup, q[:, 1])
    tt = np.maximum(tup, q[:, 2])
    bt = np.maximum(bup, q[:, 3])

    footp = np.max(
        np.linalg.norm(
            np.diff(foot, axis=1) / dt[None, :, None, None],
            axis=-1,
        ),
        axis=(1, 2),
    )

    iup = (
        np.linalg.norm(pred["impulse_mean"], axis=-1)
        + z90
        * float(B9_CALIBRATION["impulse_scale_factor"])
        * np.linalg.norm(pred["impulse_scale"], axis=-1)
    )
    iv = np.max(
        pred["touchdown_probability"] * iup,
        axis=(1, 2),
    )

    sup = np.max(
        np.clip(
            (2.0 - np.sum(pred["contact_probability"], axis=-1)) / 2.0,
            0.0,
            1.0,
        ),
        axis=1,
    )

    comp = {
        "force": _de_ratio(ft, refs["force_norm_p95_N"]),
        "torque": _de_ratio(tt, refs["abs_tau_p95_Nm"]),
        "body": _de_ratio(bt, refs["body_robust_z_p95"]),
        "E": _de_ratio(Et, refs["E_norm_p95"]),
        "foot": _de_ratio(footp, refs["foot_speed_p95_mps"]),
        "impulse": _de_ratio(iv, refs["touchdown_impulse_norm_p95_Ns"]),
        "support": sup,
    }

    physical = (
        0.20 * comp["force"]
        + 0.14 * comp["torque"]
        + 0.14 * comp["body"]
        + 0.14 * comp["E"]
        + 0.10 * comp["foot"]
        + 0.18 * comp["impulse"]
        + 0.10 * comp["support"]
    )

    return {
        **comp,
        "physical_risk": np.clip(physical, 0.0, 1.0),
        "force_upper": ft,
        "E_upper": Et,
    }

# ---------------- Build deterministic TRAIN-ONLY candidate bank ----------------
bank_X, bank_A, bank_H = [], [], []
bank_rid, bank_pos = [], []
cal_X = []

for rid in DE_TRAIN_IDS:
    d = C7R_DATA[rid]
    n = len(d["origins"])
    rng = np.random.default_rng(_stable_seed("E_" + rid))
    order = rng.permutation(n)

    nb = min(E_BANK_PER_RUN, max(32, n // 4))
    nc = min(E_CAL_PER_RUN, max(16, n // 10))

    bp = order[:nb]
    cp = order[nb:nb+nc]

    Xb = np.asarray(d["X_core"][bp], np.float32)
    Ab = np.asarray(d["X_action_dense"][bp], np.float32)

    bank_X.append(Xb)
    bank_A.append(Ab)
    bank_H.append(e_encode_history(Xb))
    bank_rid.extend([rid] * len(bp))
    bank_pos.extend(bp.tolist())
    cal_X.append(np.asarray(d["X_core"][cp], np.float32))

E_BANK_X = np.concatenate(bank_X, axis=0)
E_BANK_A = np.concatenate(bank_A, axis=0)
E_BANK_H = _normalize_rows(np.concatenate(bank_H, axis=0))
E_BANK_RID = np.asarray(bank_rid, dtype="<U64")
E_BANK_POS = np.asarray(bank_pos, np.int64)

# ---------------- Support-distance threshold from held-out TRAIN origins ----------------
E_SUPPORT_CAL = _normalize_rows(e_encode_history(np.concatenate(cal_X, axis=0)))
_ref_t = torch.as_tensor(E_BANK_H, dtype=torch.float32, device=E_DEVICE)

support_cal_vals = []
for st in range(0, len(E_SUPPORT_CAL), 256):
    q = torch.as_tensor(
        E_SUPPORT_CAL[st:st+256],
        dtype=torch.float32,
        device=E_DEVICE,
    )
    sim = q @ _ref_t.T
    support_cal_vals.append((1.0 - sim.max(dim=1).values).cpu().numpy())

E_SUPPORT_CAL = np.concatenate(support_cal_vals)
E_SUPPORT_VETO_THRESHOLD = float(np.quantile(E_SUPPORT_CAL, 0.95))

# ---------------- Physical-risk threshold from logged TRAIN state/action pairs ----------------
E_BANK_PRED = e_predict_arbitrary(E_BANK_X, E_BANK_A)
E_BANK_RISK = e_predicted_risk(E_BANK_PRED)
E_PHYS_VETO_THRESHOLD = float(
    np.quantile(E_BANK_RISK["physical_risk"], 0.95)
)

np.savez_compressed(
    E_ROOT / "candidate_bank_train_only.npz",
    history_latent=E_BANK_H.astype(np.float16),
    action_dense=E_BANK_A.astype(np.float16),
    source_run=E_BANK_RID,
    source_pos=E_BANK_POS,
)

(E_ROOT / "support_and_veto_thresholds.json").write_text(
    json.dumps(
        {
            "support_distance_p95_train": E_SUPPORT_VETO_THRESHOLD,
            "physical_risk_p95_train": E_PHYS_VETO_THRESHOLD,
            "candidate_bank_count": int(len(E_BANK_A)),
            "candidate_sequences_are_logged_train_actions": True,
            "fresh_tests_used": False,
        },
        indent=2,
    )
)

print("E0 complete.")
print("Candidate bank:", E_BANK_A.shape)
print("Support-distance veto threshold:", E_SUPPORT_VETO_THRESHOLD)
print("Physical-risk veto threshold:", E_PHYS_VETO_THRESHOLD)
print("All candidate actions come from TRAIN logs only.")

# %% notebook cell 5
# ================================================================
# E1 — COUNTERFACTUAL PHYSICAL RANKER
#
# Input at deployment:
#   current proprioceptive history X_core
#   local waypoint direction in robot/base horizontal frame
#
# Candidate sequences:
#   nearest real logged TRAIN action sequences in B9 history-latent space
#
# Score:
#   progress toward local waypoint
#   - physical B9 risk
#   - state/action support distance
#   - optional semantic penalty
#
# Hard veto:
#   physical risk above TRAIN p95
#   OR latent support distance above TRAIN p95
#
# RGB-D is NOT forced into B9.
# D visual state decides whether a VLM/semantic review is needed.
# ================================================================
def _zscore_pool(x):
    x = np.asarray(x, float)
    s = float(np.std(x))
    if s < 1e-8:
        return np.zeros_like(x)
    return (x - float(np.mean(x))) / s

def e_query_latent(x_core):
    return _normalize_rows(
        e_encode_history(np.asarray(x_core, np.float32)[None, :])
    )[0]

def e_retrieve_candidates(x_core, K=E_DEFAULT_K):
    q = e_query_latent(x_core)
    sim = E_BANK_H @ q
    K = min(int(K), len(sim))
    idx = np.argpartition(-sim, K-1)[:K]
    idx = idx[np.argsort(-sim[idx])]

    return {
        "bank_index": idx,
        "actions": np.asarray(E_BANK_A[idx], np.float32),
        "support_distance": (1.0 - sim[idx]).astype(np.float32),
        "source_run": E_BANK_RID[idx],
        "source_pos": E_BANK_POS[idx],
    }

def e_rank_candidates(
    rid,
    origin_pos,
    goal_dir_body=(1.0, 0.0),
    K=E_DEFAULT_K,
    lambda_phys=1.0,
    lambda_support=0.25,
    semantic_cost=None,
    lambda_semantic=0.50,
):
    if rid in DE_LOCKED_TEST_IDS:
        raise RuntimeError("Fresh C test access is forbidden in D/E development.")
    if rid not in C7R_DATA:
        raise KeyError(rid)

    d = C7R_DATA[rid]
    p = int(origin_pos)
    if not (0 <= p < len(d["origins"])):
        raise IndexError(p)

    x = np.asarray(d["X_core"][p], np.float32)
    cand = e_retrieve_candidates(x, K=K)

    Xrep = np.repeat(x[None, :], len(cand["actions"]), axis=0)
    pred = e_predict_arbitrary(Xrep, cand["actions"])
    risk = e_predicted_risk(pred)

    # Predicted displacement is already in the robot/base frame at the origin.
    delta = (
        np.asarray(pred["continuous_mean"], float)
        [:, :, B_STEP_LAYOUT["delta_position_base"]]
    )
    final_xy = delta[:, -1, :2]

    gd = np.asarray(goal_dir_body, float).reshape(2)
    gn = np.linalg.norm(gd)
    if gn < 1e-8:
        raise ValueError("goal_dir_body cannot be zero.")
    gd = gd / gn

    progress = final_xy @ gd

    support = np.asarray(cand["support_distance"], float)
    phys = np.asarray(risk["physical_risk"], float)

    # Semantic penalty is optional until a VLM response exists.
    if semantic_cost is None:
        sem = np.zeros(len(progress), dtype=float)
    else:
        sem = np.asarray(semantic_cost, float)
        if sem.ndim == 0:
            sem = np.full(len(progress), float(sem))
        if len(sem) != len(progress):
            raise ValueError("semantic_cost must be scalar or length K.")

    score = (
        _zscore_pool(progress)
        - float(lambda_phys) * _zscore_pool(phys)
        - float(lambda_support) * _zscore_pool(support)
        - float(lambda_semantic) * _zscore_pool(sem)
    )

    veto = (
        (phys > E_PHYS_VETO_THRESHOLD)
        | (support > E_SUPPORT_VETO_THRESHOLD)
    )

    safe_score = score.copy()
    safe_score[veto] = -np.inf

    # D visual context is scene-level until camera/body alignment is verified.
    vctx = d_origin_visual_context(rid, [p]).iloc[0].to_dict()
    needs_vlm = bool(vctx["semantic_trigger"])

    table = pd.DataFrame({
        "candidate_rank_raw": np.arange(len(progress)),
        "source_run": cand["source_run"],
        "source_pos": cand["source_pos"],
        "support_distance": support,
        "physical_risk": phys,
        "force_component": risk["force"],
        "impulse_component": risk["impulse"],
        "support_component": risk["support"],
        "progress_m": progress,
        "pred_final_dx_m": final_xy[:, 0],
        "pred_final_dy_m": final_xy[:, 1],
        "semantic_cost": sem,
        "veto": veto,
        "score": safe_score,
    })

    table = table.sort_values(
        ["veto", "score"],
        ascending=[True, False],
    ).reset_index(drop=True)

    selected = None
    if np.any(~veto):
        selected = table.iloc[0].to_dict()

    return {
        "run_id": rid,
        "origin_pos": p,
        "origin_index": int(d["origins"][p]),
        "goal_dir_body": gd.tolist(),
        "visual_context": vctx,
        "needs_vlm_semantic_review": needs_vlm,
        "selected": selected,
        "all_candidates_vetoed": bool(np.all(veto)),
        "table": table,
        "predictions": pred,
        "risk": risk,
    }

# ---------------- Smoke test: validation input, NO future truth read ----------------
E_SMOKE_RID = DE_VAL_IDS[0]
E_SMOKE_POS = len(C7R_DATA[E_SMOKE_RID]["origins"]) // 2

E_SMOKE = e_rank_candidates(
    E_SMOKE_RID,
    E_SMOKE_POS,
    goal_dir_body=(1.0, 0.0),  # placeholder local waypoint direction for smoke test only
    K=64,
)

print("E1 smoke test:")
print("run:", E_SMOKE_RID, "origin_pos:", E_SMOKE_POS)
print("needs VLM semantic review:", E_SMOKE["needs_vlm_semantic_review"])
print("all candidates vetoed:", E_SMOKE["all_candidates_vetoed"])
print("selected:", E_SMOKE["selected"])
display(E_SMOKE["table"].head(10))

if E_SMOKE["needs_vlm_semantic_review"]:
    req, path = d_make_vlm_request(E_SMOKE_RID, E_SMOKE_POS)
    print("Semantic request written:", path)

# %% notebook cell 6
# ================================================================
# E2 — FREEZE PROTOTYPE STATUS / DO NOT JUMP TO MPPI YET
#
# The next scientific steps are:
#   1) verify camera-to-body horizontal alignment from recorded poses;
#   2) then make depth/VLM costs candidate-direction-specific;
#   3) obtain NEW development/validation trajectories;
#   4) validate candidate ranking / safety veto there;
#   5) only then add CEM/MPPI continuous candidate optimization.
#
# P2027/P2006 remain sealed.
# ================================================================
DE_STATUS = {
    "version": "DE_modular_physical_visual_v1",
    "B9_frozen": True,
    "C_visual_residual_used_in_dynamics": False,
    "depth_geometry_ready": True,
    "DINO_visual_novelty_ready": True,
    "event_triggered_VLM_interface_ready": True,
    "support_constrained_candidate_bank_ready": True,
    "candidate_actions_are_real_training_sequences": True,
    "physical_counterfactual_ranker_ready": True,
    "directional_visual_cost_enabled": bool(
        D_CAMERA_BODY_HORIZONTAL_ALIGNMENT_VERIFIED
    ),
    "MPPI_or_CEM_enabled": False,
    "fresh_C_tests_scored": False,
    "fresh_C_tests_locked": DE_LOCKED_TEST_IDS,
    "next_step": (
        "Audit camera/body alignment, add direction-specific geometry/semantic "
        "cost, and validate on NEW trajectories before enabling CEM/MPPI."
    ),
}

(DE_ROOT / "DE_STATUS.json").write_text(
    json.dumps(DE_STATUS, indent=2)
)

print(json.dumps(DE_STATUS, indent=2))
print("\nSaved:", DE_ROOT / "DE_STATUS.json")
print("\nSTOP HERE. Do not open P2027/P2006 and do not enable MPPI/CEM yet.")
