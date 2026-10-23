# V10-R STAGE B: dynamics repair only. NO RGB-D/VLM/planner.


####################################################################################################
# B1
####################################################################################################


# ======================================================================
# B1 — CORRECTED DYNAMICS DATASET
#
# Fixes:
#   1) absolute odom position -> origin-relative displacement in the
#      CURRENT BASE FRAME;
#   2) 10 future command snapshots -> full 100 Hz future command sequence.
#
# NO RGB / DEPTH / VLM is used anywhere in B1-B10.
# ======================================================================

from pathlib import Path
import json, math, hashlib
import numpy as np
import pandas as pd
from IPython.display import display

for _name in (
    "V10R_ACTION_DATA",
    "V10R_DERIVED_PATHS",
    "TWO_SPECS",
    "collect_full_tf_graph",
    "_find_directed_path",
    "_compose_path_series",
):
    if _name not in globals():
        raise RuntimeError(f"B1 requires A1-A6 baseline state; missing {_name}")

B_ROOT = STRICT_ROOT / "B_dynamics_repair_v1"
B_ROOT.mkdir(parents=True, exist_ok=True)
(B_ROOT / "models").mkdir(exist_ok=True)
(B_ROOT / "tables").mkdir(exist_ok=True)
(B_ROOT / "figures").mkdir(exist_ok=True)
(B_ROOT / "derived").mkdir(exist_ok=True)

B_H = len(CFG.horizons_s)
B_HORIZON_OFFSETS = np.asarray(
    [int(round(float(h) * CFG.sample_hz)) for h in CFG.horizons_s],
    dtype=np.int64,
)
B_MAX_FUTURE = int(B_HORIZON_OFFSETS.max())

# Full future control sequence at the cache's native 100 Hz:
# samples t+0.01 ... t+1.00 for a 1 s forecast.
B_DENSE_ACTION_OFFSETS = np.arange(1, B_MAX_FUTURE + 1, dtype=np.int64)
B_ACTION_DIM = 18  # U:6 + L_LEG:12

# Keep the same history representation that produced the strong A5 curves.
B_HISTORY_LAGS = np.asarray(HISTORY_LAGS, dtype=np.int64)
B_HISTORY_SNAPSHOTS = len(B_HISTORY_LAGS)

def _b_layout(items):
    layout, cursor = {}, 0
    for name, width in items:
        layout[name] = slice(cursor, cursor + int(width))
        cursor += int(width)
    return layout, cursor

# Impulse is NOT in this continuous target anymore. B2 handles it
# conditionally on touchdown.
B_STEP_LAYOUT, B_STEP_WIDTH = _b_layout([
    ("q", 12),
    ("dq", 12),
    ("tau", 12),
    ("force_odom", 12),
    ("body_B", 12),
    ("E", 12),
    ("foot_base", 12),
    ("delta_position_base", 3),
])

def _b_base_rotation_cache(spec, derived_path):
    """
    Return R_odom_from_base on the exact cache clock.
    Persist it because TF parsing is comparatively expensive.
    """
    out = B_ROOT / "derived" / f"{spec['run_id']}_R_odom_from_base.npy"
    if out.exists():
        R = np.load(out)
        if R.ndim == 3 and R.shape[1:] == (3, 3):
            return R.astype(np.float32)

    cache = locate_physics_cache(spec)
    if cache is None:
        raise FileNotFoundError(f"{spec['run_id']}: physics cache missing")

    with np.load(cache, allow_pickle=False) as z:
        time_s = np.asarray(z["time"], np.float64)

    with np.load(derived_path, allow_pickle=False) as d:
        offset_s = float(np.asarray(d["cache_to_ros_offset_s"]))

    graph = collect_full_tf_graph(spec)
    path = _find_directed_path(
        graph["static"], graph["dynamic"], "odom", "base"
    )
    query_rel = (time_s - time_s[0]) + offset_s
    T = _compose_path_series(graph, path, query_rel)
    R = T[:, :3, :3].astype(np.float32)

    if len(R) != len(time_s) or not np.isfinite(R).all():
        raise RuntimeError(f"{spec['run_id']}: invalid odom<-base rotations")

    # Proper-rotation check on a sparse set.
    probe = R[::max(1, len(R)//200)]
    err = np.max(np.abs(
        np.einsum("nij,nkj->nik", probe, probe) - np.eye(3)[None]
    ))
    det = np.linalg.det(probe)
    if err > 3e-3 or np.min(det) < 0.995:
        raise RuntimeError(
            f"{spec['run_id']}: TF rotation validation failed "
            f"(orth_err={err}, det_min={np.min(det)})"
        )

    np.save(out, R)
    return R

def build_B_run_data(spec, split_policy="chronological", trajectory_role=None,
                     derived_path=None):
    """
    Build one causal dynamics run.

    split_policy:
      - chronological: same 60/5/10/5/20 development split as A3
      - trajectory: every origin receives the provided trajectory_role
    """
    cache_path = locate_physics_cache(spec)
    if cache_path is None:
        raise FileNotFoundError(f"{spec['run_id']}: physics cache missing")

    if derived_path is None:
        derived_path = V10R_DERIVED_PATHS.get(spec["run_id"])
    if derived_path is None or not Path(derived_path).exists():
        raise FileNotFoundError(
            f"{spec['run_id']}: real derived target file missing"
        )
    derived_path = Path(derived_path)

    with np.load(cache_path, allow_pickle=False) as z:
        time_s = np.asarray(z["time"], np.float64)
        P = np.asarray(z["P"], np.float32)
        B = np.asarray(z["B"], np.float32)
        U = np.asarray(z["U"], np.float32)
        L = np.asarray(z["L_LEG"], np.float32)
        E = np.asarray(z["E"], np.float32)

    with np.load(derived_path, allow_pickle=False) as d:
        foot_base = np.asarray(d["foot_base"], np.float32)
        force_odom = np.asarray(
            d["force_odom_from_anymal_state"], np.float32
        )
        contact = np.asarray(
            d["contact_from_anymal_state"], np.uint8
        ).astype(bool)
        touchdown = np.asarray(
            d["touchdown_event"], np.uint8
        ).astype(bool)
        impulse = np.asarray(
            d["touchdown_impulse_odom"], np.float32
        )
        base_pos = np.asarray(
            d["base_position_odom"], np.float32
        )

    n = len(time_s)
    for name, x in {
        "P":P, "B":B, "U":U, "L":L, "E":E,
        "foot_base":foot_base, "force_odom":force_odom,
        "contact":contact, "touchdown":touchdown,
        "impulse":impulse, "base_pos":base_pos,
    }.items():
        if len(x) != n:
            raise RuntimeError(
                f"{spec['run_id']}: {name} length {len(x)} != {n}"
            )

    R_ob = _b_base_rotation_cache(spec, derived_path)
    if len(R_ob) != n:
        raise RuntimeError(f"{spec['run_id']}: base rotation length mismatch")

    first_origin = int(B_HISTORY_LAGS.max())
    origins = np.arange(
        first_origin,
        n - B_MAX_FUTURE,
        dtype=np.int64,
    )
    hist_idx = origins[:, None] - B_HISTORY_LAGS[None, :]
    fut_idx = origins[:, None] + B_HORIZON_OFFSETS[None, :]
    dense_idx = origins[:, None] + B_DENSE_ACTION_OFFSETS[None, :]

    # History exactly follows the successful A3 representation.
    P_hist = np.concatenate([P[:, :, 0:9], force_odom], axis=-1)
    hist_signal = np.concatenate([
        P_hist.reshape(n, -1),          # 48
        B.reshape(n, -1),               # 12
        E.reshape(n, -1),               # 12
        contact.astype(np.float32),     # 4
        U.reshape(n, -1),               # 6
        L.reshape(n, -1),               # 12
        foot_base.reshape(n, -1),       # 12
    ], axis=1).astype(np.float32)

    X_core = hist_signal[hist_idx].reshape(len(origins), -1)

    # FULL 100 Hz logged future control path.
    action_all = np.concatenate(
        [U, L.reshape(n, -1)], axis=1
    ).astype(np.float32)
    X_action_dense = action_all[dense_idx].astype(np.float32)

    # Future continuous targets.
    q = P[fut_idx, :, 0:3].reshape(len(origins), B_H, 12)
    dq = P[fut_idx, :, 3:6].reshape(len(origins), B_H, 12)
    tau = P[fut_idx, :, 6:9].reshape(len(origins), B_H, 12)
    f = force_odom[fut_idx].reshape(len(origins), B_H, 12)
    body = B[fut_idx].reshape(len(origins), B_H, 12)
    e = E[fut_idx].reshape(len(origins), B_H, 12)
    foot = foot_base[fut_idx].reshape(len(origins), B_H, 12)

    # Correct displacement target:
    # delta expressed in the BASE frame at the forecast origin.
    delta_o = (
        base_pos[fut_idx]
        - base_pos[origins][:, None, :]
    )
    R_bo_origin = np.transpose(R_ob[origins], (0, 2, 1))
    delta_b = np.einsum(
        "nij,nhj->nhi", R_bo_origin, delta_o
    ).astype(np.float32)

    Y_step = np.concatenate(
        [q, dq, tau, f, body, e, foot, delta_b],
        axis=-1,
    ).astype(np.float32)

    if Y_step.shape[-1] != B_STEP_WIDTH:
        raise RuntimeError(
            f"{spec['run_id']}: B step width mismatch {Y_step.shape}"
        )

    Y_cont = Y_step.reshape(len(origins), -1)
    Y_contact = contact[fut_idx].astype(np.float32)
    Y_touchdown = touchdown[fut_idx].astype(np.float32)
    Y_impulse = impulse[fut_idx].astype(np.float32)

    if split_policy == "chronological":
        frac = origins / float(n - 1)
        split = np.full(len(origins), "purge", dtype="<U8")
        split[frac < 0.60] = "train"
        split[(frac >= 0.65) & (frac < 0.75)] = "val"
        split[frac >= 0.80] = "test"

        # Leakage guard: purge must exceed the complete forecast horizon.
        tr = origins[split == "train"]
        va = origins[split == "val"]
        te = origins[split == "test"]
        if min(len(tr), len(va), len(te)) == 0:
            raise RuntimeError(f"{spec['run_id']}: empty chronological split")
        if (va.min() - tr.max()) <= B_MAX_FUTURE:
            raise RuntimeError(f"{spec['run_id']}: train-val purge too short")
        if (te.min() - va.max()) <= B_MAX_FUTURE:
            raise RuntimeError(f"{spec['run_id']}: val-test purge too short")

    elif split_policy == "trajectory":
        if trajectory_role not in {"train", "val", "test"}:
            raise ValueError("trajectory_role must be train/val/test")
        split = np.full(len(origins), trajectory_role, dtype="<U8")
    else:
        raise ValueError(split_policy)

    for name, arr in {
        "X_core":X_core,
        "X_action_dense":X_action_dense,
        "Y_cont":Y_cont,
        "Y_contact":Y_contact,
        "Y_touchdown":Y_touchdown,
        "Y_impulse":Y_impulse,
    }.items():
        if not np.isfinite(arr).all():
            raise RuntimeError(
                f"{spec['run_id']}: nonfinite values in {name}"
            )

    return {
        "spec": spec,
        "terrain": spec["terrain"],
        "time": time_s,
        "origins": origins,
        "future_idx": fut_idx,
        "dense_idx": dense_idx,
        "X_core": X_core.astype(np.float32),
        "X_action_dense": X_action_dense.astype(np.float32),
        "Y_cont": Y_cont.astype(np.float32),
        "Y_contact": Y_contact.astype(np.float32),
        "Y_touchdown": Y_touchdown.astype(np.float32),
        "Y_impulse": Y_impulse.astype(np.float32),
        "split": split,
        "derived_path": str(derived_path),
        "cache_path": str(cache_path),
    }

B_DEV_DATA = {
    spec["run_id"]: build_B_run_data(
        spec, split_policy="chronological"
    )
    for spec in TWO_SPECS
}

rows = []
for rid, d in B_DEV_DATA.items():
    for sp in ("train", "val", "test", "purge"):
        rows.append({
            "run_id":rid,
            "split":sp,
            "samples":int(np.sum(d["split"] == sp)),
            "history_width":int(d["X_core"].shape[1]),
            "dense_action_shape":str(d["X_action_dense"].shape[1:]),
            "continuous_target_width":int(d["Y_cont"].shape[1]),
        })
B1_DATASET_TABLE = pd.DataFrame(rows)
display(B1_DATASET_TABLE)

print("\nB1 per-horizon continuous layout:")
for k, s in B_STEP_LAYOUT.items():
    print(f"  {k:24s} {s.start}:{s.stop}")
print(
    "\nB1 complete: absolute odom target removed; "
    "100 Hz future-control sequence retained."
)


####################################################################################################
# B2
####################################################################################################


# ======================================================================
# B2 — TOUCHDOWN AS EVENT + CONDITIONAL IMPULSE
#
# Replaces the misleading mostly-zero impulse regression metric with:
#   P(touchdown at horizon, leg)
#   p(impulse | touchdown)
#
# Also computes train-only event prevalence/threshold references.
# ======================================================================

import numpy as np
import pandas as pd
from IPython.display import display

if "B_DEV_DATA" not in globals():
    raise RuntimeError("Run B1 first.")

def _b_stack(data_dict, field, split_name):
    xs = []
    for d in data_dict.values():
        m = d["split"] == split_name
        if np.any(m):
            xs.append(d[field][m])
    if not xs:
        raise RuntimeError(f"No {split_name} samples for {field}")
    return np.concatenate(xs, axis=0)

B2_TD_TRAIN = _b_stack(B_DEV_DATA, "Y_touchdown", "train")
B2_IMP_TRAIN = _b_stack(B_DEV_DATA, "Y_impulse", "train")

td_count = int(B2_TD_TRAIN.sum())
td_total = int(B2_TD_TRAIN.size)
td_prev = td_count / max(td_total, 1)

# Conditional impulse statistics: only real touchdown labels.
event_imp = B2_IMP_TRAIN[B2_TD_TRAIN.astype(bool)]
if len(event_imp) == 0:
    raise RuntimeError("B2 found zero touchdown events in training data.")

imp_norm = np.linalg.norm(event_imp, axis=-1)
B2_EVENT_REFERENCE = {
    "touchdown_positive_count": td_count,
    "touchdown_label_count": td_total,
    "touchdown_prevalence": float(td_prev),
    "impulse_component_abs_p50_Ns": float(
        np.quantile(np.abs(event_imp), 0.50)
    ),
    "impulse_component_abs_p95_Ns": float(
        np.quantile(np.abs(event_imp), 0.95)
    ),
    "impulse_norm_p50_Ns": float(np.quantile(imp_norm, 0.50)),
    "impulse_norm_p95_Ns": float(np.quantile(imp_norm, 0.95)),
}

# Positive-class balancing for touchdown BCE.
# Clamp to avoid pathological gradients on an extremely rare event.
B2_TD_POS_WEIGHT = float(np.clip(
    (td_total - td_count) / max(td_count, 1),
    1.0,
    40.0,
))

print("B2 event reference:")
print(B2_EVENT_REFERENCE)
print("touchdown BCE pos_weight:", B2_TD_POS_WEIGHT)

# Sanity: non-event impulse labels should be zero by construction.
non_event = B2_IMP_TRAIN[~B2_TD_TRAIN.astype(bool)]
non_event_max = float(np.max(np.abs(non_event))) if len(non_event) else 0.0
if non_event_max > 1e-5:
    raise RuntimeError(
        "B2 expected impulse labels to be zero outside touchdown events, "
        f"but max non-event magnitude is {non_event_max}"
    )

rows = []
for rid, d in B_DEV_DATA.items():
    for sp in ("train", "val", "test"):
        m = d["split"] == sp
        td = d["Y_touchdown"][m]
        imp = d["Y_impulse"][m]
        event = td.astype(bool)
        rows.append({
            "run_id": rid,
            "split": sp,
            "origins": int(m.sum()),
            "touchdown_labels": int(td.sum()),
            "touchdown_prevalence": float(td.mean()) if td.size else np.nan,
            "event_impulse_norm_median_Ns": float(
                np.median(np.linalg.norm(imp[event], axis=-1))
            ) if np.any(event) else np.nan,
        })

B2_EVENT_TABLE = pd.DataFrame(rows)
display(B2_EVENT_TABLE)

print(
    "\nB2 complete. From here on, impulse loss/MAE is computed only "
    "where Y_touchdown == 1."
)


####################################################################################################
# B3
####################################################################################################


# ======================================================================
# B3 — DENSE 100 Hz FUTURE-COMMAND ENCODER
#
# The action-conditioned model now consumes every logged future command
# available between origin and each prediction horizon.
#
# IMPORTANT:
# - This remains the oracle_logged dynamics-identifiability experiment.
# - For the 0.1 s prediction, the action encoder sees only commands up to
#   0.1 s; it cannot use commands from 0.2-1.0 s.
# - NO RGB-D enters this model.
# ======================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

if "B_DEV_DATA" not in globals():
    raise RuntimeError("Run B1-B2 first.")

B_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
B_MODEL_DIM = 320
B_ACTION_HIDDEN = 160
B_HORIZON_EMBED = 48

# Dense sequence indexes corresponding exactly to 0.1, ..., 1.0 s.
B_ACTION_GATHER = torch.as_tensor(
    B_HORIZON_OFFSETS - 1,
    dtype=torch.long,
)

class BDenseActionEncoder(nn.Module):
    """
    Causal GRU over the full 100 Hz command sequence.
    Output at horizon h has seen only commands <= h.
    """
    def __init__(self, action_dim=18, proj=96, hidden=160):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(action_dim, proj),
            nn.GELU(),
            nn.LayerNorm(proj),
        )
        self.gru = nn.GRU(
            input_size=proj,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.08,
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, action_seq, gather_idx):
        z = self.input(action_seq)
        z, _ = self.gru(z)
        z = self.norm(z)
        return z[:, gather_idx, :]

class BResponseModel(nn.Module):
    """
    Horizon-factorized probabilistic dynamics model.

    Outputs:
      continuous mean/log-scale : [B,H,D]
      contact logits            : [B,H,4]
      touchdown logits          : [B,H,4]
      conditional impulse mean/log-scale : [B,H,4,3]
      direct log-space q90 tail metrics  : [B,4]
    """
    def __init__(
        self,
        history_dim,
        continuous_step_dim,
        use_action=True,
        hidden=B_MODEL_DIM,
    ):
        super().__init__()
        self.use_action = bool(use_action)
        self.H = int(B_H)
        self.D = int(continuous_step_dim)

        self.history = nn.Sequential(
            nn.Linear(history_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(0.08),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(0.08),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )

        self.action = BDenseActionEncoder(
            action_dim=B_ACTION_DIM,
            hidden=B_ACTION_HIDDEN,
        )

        self.horizon_embed = nn.Parameter(
            torch.randn(B_H, B_HORIZON_EMBED) * 0.02
        )

        fused_dim = hidden + B_ACTION_HIDDEN + B_HORIZON_EMBED
        self.fuse = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(0.06),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )

        self.mean_head = nn.Linear(hidden, self.D)
        self.log_scale_head = nn.Linear(hidden, self.D)
        self.contact_head = nn.Linear(hidden, 4)
        self.touchdown_head = nn.Linear(hidden, 4)

        self.impulse_mean_head = nn.Linear(hidden, 12)
        self.impulse_log_scale_head = nn.Linear(hidden, 12)

        # Sequence-level direct upper-tail predictors:
        # [peak force norm, peak E norm, peak |tau|, peak body deviation]
        self.tail_q90_head = nn.Sequential(
            nn.Linear(hidden + B_ACTION_HIDDEN, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 4),
        )

    def forward(self, history_flat, action_dense):
        n = history_flat.shape[0]
        h0 = self.history(history_flat)

        if self.use_action:
            ah = self.action(
                action_dense,
                B_ACTION_GATHER.to(action_dense.device),
            )
            a_final = ah[:, -1]
        else:
            ah = torch.zeros(
                n, self.H, B_ACTION_HIDDEN,
                device=history_flat.device,
                dtype=history_flat.dtype,
            )
            a_final = torch.zeros(
                n, B_ACTION_HIDDEN,
                device=history_flat.device,
                dtype=history_flat.dtype,
            )

        h = h0[:, None, :].expand(-1, self.H, -1)
        he = self.horizon_embed[None].expand(n, -1, -1)

        z = self.fuse(torch.cat([h, ah, he], dim=-1))

        mean = self.mean_head(z)
        log_scale = torch.clamp(
            self.log_scale_head(z),
            -5.5, 2.8,
        )

        impulse_mean = self.impulse_mean_head(z).reshape(
            n, self.H, 4, 3
        )
        impulse_log_scale = torch.clamp(
            self.impulse_log_scale_head(z).reshape(n, self.H, 4, 3),
            -5.5, 3.0,
        )

        tail_q90_log = self.tail_q90_head(
            torch.cat([h0, a_final], dim=-1)
        )

        return {
            "mean_z": mean,
            "log_scale_z": log_scale,
            "contact_logits": self.contact_head(z),
            "touchdown_logits": self.touchdown_head(z),
            "impulse_mean_z": impulse_mean,
            "impulse_log_scale_z": impulse_log_scale,
            "tail_q90_log": tail_q90_log,
        }

# Shape smoke test using actual B1 dimensions.
_rid0 = next(iter(B_DEV_DATA))
_d0 = B_DEV_DATA[_rid0]
_smoke = BResponseModel(
    history_dim=_d0["X_core"].shape[1],
    continuous_step_dim=B_STEP_WIDTH,
    use_action=True,
).to(B_DEVICE)

with torch.no_grad():
    _n = 3
    _out = _smoke(
        torch.zeros(_n, _d0["X_core"].shape[1], device=B_DEVICE),
        torch.zeros(
            _n,
            len(B_DENSE_ACTION_OFFSETS),
            B_ACTION_DIM,
            device=B_DEVICE,
        ),
    )

assert _out["mean_z"].shape == (_n, B_H, B_STEP_WIDTH)
assert _out["contact_logits"].shape == (_n, B_H, 4)
assert _out["touchdown_logits"].shape == (_n, B_H, 4)
assert _out["impulse_mean_z"].shape == (_n, B_H, 4, 3)
assert _out["tail_q90_log"].shape == (_n, 4)

del _smoke, _out
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("B3 dense-action model shape test: PASS")
print("device:", B_DEVICE)
print(
    "Each horizon uses the causal prefix of the 100 Hz future command sequence."
)


####################################################################################################
# B4
####################################################################################################


# ======================================================================
# B4 — TAIL-AWARE / EVENT-BALANCED TRAINING
#
# Trains on the same two development trajectories as A1-A6, but fixes the
# objective:
#   - event-balanced sampling;
#   - extra force-tail and E-tail loss;
#   - explicit touchdown BCE;
#   - impulse NLL ONLY at true touchdown labels;
#   - direct 90th-quantile prediction of sequence-level tail observables.
#
# Models:
#   B4_NOACTION : history only
#   B4_DENSE    : history + full 100 Hz logged future commands
# ======================================================================

import copy, gc, random, math, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from IPython.display import display

if "BResponseModel" not in globals():
    raise RuntimeError("Run B1-B3 first.")

B_SEED = 2409
random.seed(B_SEED)
np.random.seed(B_SEED)
torch.manual_seed(B_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(B_SEED)

B_BATCH = 384
B_EPOCHS = 30
B_PATIENCE = 6
B_LR = 2.5e-4
B_WEIGHT_DECAY = 1e-4

def _b_concat(data_dict, field, split):
    out = []
    run_ids = []
    terrains = []
    for rid, d in data_dict.items():
        m = d["split"] == split
        if not np.any(m):
            continue
        out.append(d[field][m])
        run_ids.extend([rid] * int(m.sum()))
        terrains.extend([d["terrain"]] * int(m.sum()))
    if not out:
        raise RuntimeError(f"No '{split}' data for {field}")
    return np.concatenate(out, axis=0), np.asarray(run_ids), np.asarray(terrains)

def _b_reshape_cont(y):
    return np.asarray(y, np.float32).reshape(-1, B_H, B_STEP_WIDTH)

def _b_group(step, name):
    return step[:, :, B_STEP_LAYOUT[name]]

def _b_tail_raw(step, body_med, body_mad):
    n = len(step)
    force = _b_group(step, "force_odom").reshape(n, B_H, 4, 3)
    E = _b_group(step, "E").reshape(n, B_H, 4, 3)
    tau = _b_group(step, "tau").reshape(n, B_H, 12)
    body = _b_group(step, "body_B").reshape(n, B_H, 12)

    force_peak = np.max(np.linalg.norm(force, axis=-1), axis=(1, 2))
    E_peak = np.max(np.linalg.norm(E, axis=-1), axis=(1, 2))
    tau_peak = np.max(np.abs(tau), axis=(1, 2))

    bz = np.abs(
        (body - body_med[None, None, :])
        / np.maximum(1.4826 * body_mad[None, None, :], 1e-5)
    )
    body_peak = np.max(bz, axis=(1, 2))
    return np.stack([force_peak, E_peak, tau_peak, body_peak], axis=1).astype(np.float32)

def fit_B_preprocessor(data_dict):
    Xtr, rid, terrain = _b_concat(data_dict, "X_core", "train")
    Atr, _, _ = _b_concat(data_dict, "X_action_dense", "train")
    Ytr, _, _ = _b_concat(data_dict, "Y_cont", "train")
    Ctr, _, _ = _b_concat(data_dict, "Y_contact", "train")
    TDtr, _, _ = _b_concat(data_dict, "Y_touchdown", "train")
    Itr, _, _ = _b_concat(data_dict, "Y_impulse", "train")

    Ystep = _b_reshape_cont(Ytr)

    x_mean = Xtr.mean(axis=0).astype(np.float32)
    x_std = Xtr.std(axis=0).astype(np.float32)
    x_std = np.where(x_std < 1e-5, 1.0, x_std).astype(np.float32)

    action_mean = Atr.mean(axis=(0, 1)).astype(np.float32)
    action_std = Atr.std(axis=(0, 1)).astype(np.float32)
    action_std = np.where(action_std < 1e-5, 1.0, action_std).astype(np.float32)

    y_mean = Ystep.mean(axis=0).astype(np.float32)
    y_std = Ystep.std(axis=0).astype(np.float32)
    y_std = np.where(y_std < 1e-5, 1.0, y_std).astype(np.float32)

    body = _b_group(Ystep, "body_B").reshape(-1, 12)
    body_med = np.median(body, axis=0).astype(np.float32)
    body_mad = np.median(
        np.abs(body - body_med[None]), axis=0
    ).astype(np.float32)
    body_mad = np.maximum(body_mad, 1e-4).astype(np.float32)

    force = _b_group(Ystep, "force_odom").reshape(-1, 4, 3)
    E = _b_group(Ystep, "E").reshape(-1, 4, 3)

    force95 = float(np.quantile(np.linalg.norm(force, axis=-1), 0.95))
    E95 = float(np.quantile(np.linalg.norm(E, axis=-1), 0.95))

    event_imp = Itr[TDtr.astype(bool)]
    if len(event_imp) < 20:
        raise RuntimeError(
            f"Too few touchdown impulse events in train split: {len(event_imp)}"
        )
    imp_mean = event_imp.mean(axis=0).astype(np.float32)
    imp_std = event_imp.std(axis=0).astype(np.float32)
    imp_std = np.where(imp_std < 1e-4, 1.0, imp_std).astype(np.float32)

    tail_train = _b_tail_raw(Ystep, body_med, body_mad)
    tail_log_train = np.log1p(np.maximum(tail_train, 0.0))
    tail_log_med = np.median(tail_log_train, axis=0).astype(np.float32)
    tail_log_scale = np.median(
        np.abs(tail_log_train - tail_log_med[None]), axis=0
    ).astype(np.float32)
    tail_log_scale = np.maximum(1.4826 * tail_log_scale, 0.05).astype(np.float32)

    # Per-output touchdown positive weight.
    td_pos = TDtr.sum(axis=0)
    td_neg = len(TDtr) - td_pos
    td_pos_weight = np.clip(
        td_neg / np.maximum(td_pos, 1.0),
        1.0, 40.0,
    ).astype(np.float32)

    return {
        "x_mean":x_mean,
        "x_std":x_std,
        "action_mean":action_mean,
        "action_std":action_std,
        "y_mean":y_mean,
        "y_std":y_std,
        "imp_mean":imp_mean,
        "imp_std":imp_std,
        "body_med":body_med,
        "body_mad":body_mad,
        "force95":force95,
        "E95":E95,
        "tail_log_med":tail_log_med,
        "tail_log_scale":tail_log_scale,
        "td_pos_weight":td_pos_weight,
    }

class BArrayDataset(Dataset):
    def __init__(self, data_dict, split, prep, balanced=False):
        X, run_ids, terrains = _b_concat(data_dict, "X_core", split)
        A, _, _ = _b_concat(data_dict, "X_action_dense", split)
        Y, _, _ = _b_concat(data_dict, "Y_cont", split)
        C, _, _ = _b_concat(data_dict, "Y_contact", split)
        TD, _, _ = _b_concat(data_dict, "Y_touchdown", split)
        IMP, _, _ = _b_concat(data_dict, "Y_impulse", split)

        step = _b_reshape_cont(Y)
        force = _b_group(step, "force_odom").reshape(len(Y), B_H, 4, 3)
        E = _b_group(step, "E").reshape(len(Y), B_H, 4, 3)

        force_tail = (
            np.linalg.norm(force, axis=-1) > prep["force95"]
        ).astype(np.float32)
        E_tail = (
            np.linalg.norm(E, axis=-1) > prep["E95"]
        ).astype(np.float32)

        tail_raw = _b_tail_raw(step, prep["body_med"], prep["body_mad"])
        tail_log = np.log1p(np.maximum(tail_raw, 0.0)).astype(np.float32)

        self.X = ((X - prep["x_mean"]) / prep["x_std"]).astype(np.float32)
        self.A = (
            (A - prep["action_mean"][None, None])
            / prep["action_std"][None, None]
        ).astype(np.float32)
        self.Y = (
            (step - prep["y_mean"][None])
            / prep["y_std"][None]
        ).astype(np.float32)
        self.C = C.astype(np.float32)
        self.TD = TD.astype(np.float32)
        self.IMP = (
            (IMP - prep["imp_mean"][None, None, None, :])
            / prep["imp_std"][None, None, None, :]
        ).astype(np.float32)
        self.force_tail = force_tail
        self.E_tail = E_tail
        self.tail_log = tail_log
        self.run_ids = run_ids
        self.terrains = terrains

        # Event-balanced + tail-balanced sampler weights.
        sw = (
            1.0
            + 2.0 * force_tail.max(axis=(1, 2))
            + 2.0 * E_tail.max(axis=(1, 2))
            + 4.0 * TD.max(axis=(1, 2))
        )

        if balanced:
            unique, counts = np.unique(terrains, return_counts=True)
            terrain_factor = {
                t: len(terrains) / (len(unique) * c)
                for t, c in zip(unique, counts)
            }
            sw *= np.asarray([terrain_factor[t] for t in terrains], float)

        self.sample_weight = np.clip(sw, 1.0, 12.0).astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.X[i]),
            torch.from_numpy(self.A[i]),
            torch.from_numpy(self.Y[i]),
            torch.from_numpy(self.C[i]),
            torch.from_numpy(self.TD[i]),
            torch.from_numpy(self.IMP[i]),
            torch.from_numpy(self.force_tail[i]),
            torch.from_numpy(self.E_tail[i]),
            torch.from_numpy(self.tail_log[i]),
        )

def _b_pinball(pred, target, q=0.90):
    e = target - pred
    return torch.maximum(q * e, (q - 1.0) * e)

def _b_loss(out, batch, prep):
    x, a, y, contact, td, imp, f_tail, e_tail, tail_log = batch

    mu = out["mean_z"]
    log_s = out["log_scale_z"]
    inv_var = torch.exp(-2.0 * log_s)
    nll_elem = 0.5 * (y - mu).square() * inv_var + log_s
    base_nll = nll_elem.mean()

    fs = B_STEP_LAYOUT["force_odom"]
    es = B_STEP_LAYOUT["E"]

    force_pred = mu[:, :, fs].reshape(-1, B_H, 4, 3)
    force_true = y[:, :, fs].reshape(-1, B_H, 4, 3)
    E_pred = mu[:, :, es].reshape(-1, B_H, 4, 3)
    E_true = y[:, :, es].reshape(-1, B_H, 4, 3)

    force_component_weight = 1.0 + 4.0 * f_tail[..., None]
    E_component_weight = 1.0 + 4.0 * e_tail[..., None]

    force_tail_loss = (
        F.smooth_l1_loss(
            force_pred, force_true,
            reduction="none", beta=0.75,
        ) * force_component_weight
    ).mean()

    E_tail_loss = (
        F.smooth_l1_loss(
            E_pred, E_true,
            reduction="none", beta=0.75,
        ) * E_component_weight
    ).mean()

    contact_loss = F.binary_cross_entropy_with_logits(
        out["contact_logits"], contact
    )

    pos_w = torch.as_tensor(
        prep["td_pos_weight"],
        device=td.device,
        dtype=td.dtype,
    )
    td_loss = F.binary_cross_entropy_with_logits(
        out["touchdown_logits"],
        td,
        pos_weight=pos_w,
    )

    # Conditional impulse NLL ONLY at true touchdown labels.
    im_mu = out["impulse_mean_z"]
    im_log = out["impulse_log_scale_z"]
    im_nll = (
        0.5 * (imp - im_mu).square() * torch.exp(-2.0 * im_log)
        + im_log
    ).mean(dim=-1)  # [B,H,4]
    td_mask = td > 0.5
    if torch.any(td_mask):
        impulse_loss = im_nll[td_mask].mean()
    else:
        impulse_loss = im_nll.sum() * 0.0

    # Direct upper-tail target is log1p(metric); pinball q=.90.
    tail_loss = _b_pinball(
        out["tail_q90_log"], tail_log, q=0.90
    ).mean()

    total = (
        base_nll
        + 0.45 * force_tail_loss
        + 0.35 * E_tail_loss
        + 0.20 * contact_loss
        + 0.45 * td_loss
        + 0.55 * impulse_loss
        + 0.15 * tail_loss
    )

    return {
        "total": total,
        "base_nll": base_nll.detach(),
        "force_tail": force_tail_loss.detach(),
        "E_tail": E_tail_loss.detach(),
        "contact": contact_loss.detach(),
        "touchdown": td_loss.detach(),
        "impulse": impulse_loss.detach(),
        "tail_q90": tail_loss.detach(),
    }

@torch.no_grad()
def _b_eval_loss(model, loader, prep):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = tuple(t.to(B_DEVICE, non_blocking=True) for t in batch)
        out = model(batch[0], batch[1])
        loss = _b_loss(out, batch, prep)["total"]
        total += float(loss) * len(batch[0])
        n += len(batch[0])
    return total / max(n, 1)

def train_B_pair(
    data_dict,
    prefix,
    balanced_terrain=False,
    epochs=B_EPOCHS,
):
    prep = fit_B_preprocessor(data_dict)

    tr = BArrayDataset(
        data_dict, "train", prep,
        balanced=balanced_terrain,
    )
    va = BArrayDataset(
        data_dict, "val", prep,
        balanced=False,
    )

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(tr.sample_weight, dtype=torch.double),
        num_samples=len(tr),
        replacement=True,
    )
    tr_loader = DataLoader(
        tr,
        batch_size=B_BATCH,
        sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    va_loader = DataLoader(
        va,
        batch_size=B_BATCH,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    def fit_one(use_action, label):
        model = BResponseModel(
            history_dim=tr.X.shape[1],
            continuous_step_dim=B_STEP_WIDTH,
            use_action=use_action,
        ).to(B_DEVICE)

        opt = torch.optim.AdamW(
            model.parameters(),
            lr=B_LR,
            weight_decay=B_WEIGHT_DECAY,
        )
        best = float("inf")
        best_state = None
        bad = 0
        hist = []

        for ep in range(1, int(epochs) + 1):
            model.train()
            acc, count = 0.0, 0

            for batch in tr_loader:
                batch = tuple(
                    t.to(B_DEVICE, non_blocking=True)
                    for t in batch
                )
                opt.zero_grad(set_to_none=True)
                out = model(batch[0], batch[1])
                losses = _b_loss(out, batch, prep)
                loss = losses["total"]

                if not torch.isfinite(loss):
                    raise RuntimeError(f"{label}: nonfinite loss")

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 5.0
                )
                opt.step()

                acc += float(loss.detach()) * len(batch[0])
                count += len(batch[0])

            train_loss = acc / max(count, 1)
            val_loss = _b_eval_loss(model, va_loader, prep)
            hist.append({
                "model":label,
                "epoch":ep,
                "train_loss":train_loss,
                "val_loss":val_loss,
            })

            print(
                f"{label:18s} ep={ep:02d} "
                f"train={train_loss:.5f} val={val_loss:.5f}"
            )

            if val_loss < best - 1e-4:
                best = val_loss
                best_state = copy.deepcopy(
                    {k:v.detach().cpu() for k,v in model.state_dict().items()}
                )
                bad = 0
            else:
                bad += 1

            if bad >= B_PATIENCE:
                print(label, "early stop")
                break

        if best_state is None:
            raise RuntimeError(f"{label}: no valid checkpoint")

        model.load_state_dict(best_state)
        model.to(B_DEVICE).eval()

        ckpt = {
            "label":label,
            "use_action":bool(use_action),
            "state_dict":best_state,
            "history_dim":int(tr.X.shape[1]),
            "step_width":int(B_STEP_WIDTH),
            "prep":prep,
            "horizons_s":tuple(float(x) for x in CFG.horizons_s),
            "dense_action_hz":float(CFG.sample_hz),
        }
        path = B_ROOT / "models" / f"{label}.pt"
        torch.save(ckpt, path)

        return {
            "model":model,
            "use_action":bool(use_action),
            "prep":prep,
            "best_val_loss":float(best),
            "history":pd.DataFrame(hist),
            "checkpoint":str(path),
        }

    noaction = fit_one(False, f"{prefix}_noaction")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    dense = fit_one(True, f"{prefix}_dense100hz")

    return noaction, dense, prep

@torch.no_grad()
def predict_B(bundle, d, split="test", batch_size=1024):
    prep = bundle["prep"]
    m = d["split"] == split
    X = d["X_core"][m]
    A = d["X_action_dense"][m]

    Xz = ((X - prep["x_mean"]) / prep["x_std"]).astype(np.float32)
    Az = (
        (A - prep["action_mean"][None, None])
        / prep["action_std"][None, None]
    ).astype(np.float32)

    outputs = {
        "mean_z":[], "scale_z":[],
        "contact":[], "touchdown":[],
        "impulse_mean_z":[], "impulse_scale_z":[],
        "tail_q90_log":[],
    }

    model = bundle["model"]
    model.eval()

    for st in range(0, len(Xz), batch_size):
        xb = torch.from_numpy(Xz[st:st+batch_size]).to(B_DEVICE)
        ab = torch.from_numpy(Az[st:st+batch_size]).to(B_DEVICE)
        out = model(xb, ab)

        outputs["mean_z"].append(out["mean_z"].cpu().numpy())
        outputs["scale_z"].append(
            np.exp(out["log_scale_z"].cpu().numpy())
        )
        outputs["contact"].append(
            torch.sigmoid(out["contact_logits"]).cpu().numpy()
        )
        outputs["touchdown"].append(
            torch.sigmoid(out["touchdown_logits"]).cpu().numpy()
        )
        outputs["impulse_mean_z"].append(
            out["impulse_mean_z"].cpu().numpy()
        )
        outputs["impulse_scale_z"].append(
            np.exp(out["impulse_log_scale_z"].cpu().numpy())
        )
        outputs["tail_q90_log"].append(
            out["tail_q90_log"].cpu().numpy()
        )

    out = {k:np.concatenate(v, axis=0) for k,v in outputs.items()}

    mean = (
        out["mean_z"] * prep["y_std"][None]
        + prep["y_mean"][None]
    )
    scale = out["scale_z"] * prep["y_std"][None]

    imp_mean = (
        out["impulse_mean_z"]
        * prep["imp_std"][None, None, None, :]
        + prep["imp_mean"][None, None, None, :]
    )
    imp_scale = (
        out["impulse_scale_z"]
        * prep["imp_std"][None, None, None, :]
    )

    tail_q90 = np.expm1(out["tail_q90_log"])
    tail_q90 = np.maximum(tail_q90, 0.0)

    return {
        "continuous_mean":mean.astype(np.float32),
        "continuous_scale":scale.astype(np.float32),
        "contact_probability":out["contact"].astype(np.float32),
        "touchdown_probability":out["touchdown"].astype(np.float32),
        "impulse_mean":imp_mean.astype(np.float32),
        "impulse_scale":imp_scale.astype(np.float32),
        "tail_q90":tail_q90.astype(np.float32),
        "truth":d["Y_cont"][m],
        "contact_truth":d["Y_contact"][m],
        "touchdown_truth":d["Y_touchdown"][m],
        "impulse_truth":d["Y_impulse"][m],
        "origins":d["origins"][m],
    }

print("Training B4 on the SAME two development trajectories...")
B4_NOACTION, B4_DENSE, B4_PREP = train_B_pair(
    B_DEV_DATA,
    prefix="B4_dev2",
    balanced_terrain=True,
)

B4_HISTORY = pd.concat(
    [B4_NOACTION["history"], B4_DENSE["history"]],
    ignore_index=True,
)
display(
    B4_HISTORY.groupby("model", as_index=False).agg(
        best_val=("val_loss","min"),
        epochs=("epoch","max"),
    )
)
B4_HISTORY.to_csv(
    B_ROOT / "tables" / "B4_training_history.csv",
    index=False,
)

print("\nB4 complete.")


####################################################################################################
# B5
####################################################################################################


# ======================================================================
# B5 — CALIBRATED UPPER-TAIL RISK + MONOTONIC UNCERTAINTY
#
# Validation data only is used to:
#   - calibrate continuous predictive scales to ~95% marginal coverage;
#   - calibrate direct q90 tail heads;
#   - choose ONE touchdown probability threshold by validation F1.
#
# Risk uses upper envelopes / quantiles, not only predicted means.
# Uncertainty is ADDITIVE: it can never make a prediction look safer.
# ======================================================================

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from IPython.display import display

if "B4_DENSE" not in globals():
    raise RuntimeError("Run B4 first.")

def _concat_B_predictions(bundle, data_dict, split):
    parts = []
    for rid, d in data_dict.items():
        if np.any(d["split"] == split):
            p = predict_B(bundle, d, split=split)
            p["run_id"] = rid
            parts.append(p)
    if not parts:
        raise RuntimeError(f"No {split} predictions")

    keys = [
        "continuous_mean","continuous_scale","contact_probability",
        "touchdown_probability","impulse_mean","impulse_scale",
        "tail_q90","truth","contact_truth","touchdown_truth",
        "impulse_truth",
    ]
    return {
        k:np.concatenate([p[k] for p in parts], axis=0)
        for k in keys
    }

B5_VAL = _concat_B_predictions(B4_DENSE, B_DEV_DATA, "val")
_val_true = _b_reshape_cont(B5_VAL["truth"])
_val_mu = np.asarray(B5_VAL["continuous_mean"], float)
_val_scale = np.asarray(B5_VAL["continuous_scale"], float)

# ----------------------------------------------------------------------
# 1. Group-wise 95% scale calibration on validation residuals.
# ----------------------------------------------------------------------
B5_SCALE_FACTOR = {}
coverage_rows = []

for group, sl in B_STEP_LAYOUT.items():
    err = np.abs(_val_true[:, :, sl] - _val_mu[:, :, sl])
    sc = np.maximum(_val_scale[:, :, sl], 1e-6)
    ratio = err / sc

    factor = float(np.quantile(ratio[np.isfinite(ratio)], 0.95))
    factor = max(factor, 1e-3)
    B5_SCALE_FACTOR[group] = factor

    raw_cov = float(np.mean(err <= 1.96 * sc))
    cal_cov = float(np.mean(err <= factor * sc))

    coverage_rows.append({
        "group":group,
        "raw_95_gaussian_coverage":raw_cov,
        "calibrated_marginal_coverage":cal_cov,
        "scale_factor_q95_abs_error_over_sigma":factor,
    })

# Conditional impulse scale calibration.
tdv = B5_VAL["touchdown_truth"].astype(bool)
if np.any(tdv):
    ie = np.abs(
        B5_VAL["impulse_truth"] - B5_VAL["impulse_mean"]
    )
    isc = np.maximum(B5_VAL["impulse_scale"], 1e-6)
    iratio = ie[tdv] / isc[tdv]
    B5_IMPULSE_SCALE_FACTOR = float(np.quantile(iratio, 0.95))
else:
    B5_IMPULSE_SCALE_FACTOR = 1.96

B5_COVERAGE = pd.DataFrame(coverage_rows)
display(B5_COVERAGE)
print("conditional impulse scale factor:", B5_IMPULSE_SCALE_FACTOR)

# ----------------------------------------------------------------------
# 2. Direct q90 tail-head calibration on validation only.
# ----------------------------------------------------------------------
val_tail_actual = _b_tail_raw(
    _val_true,
    B4_PREP["body_med"],
    B4_PREP["body_mad"],
)
val_tail_pred = np.asarray(B5_VAL["tail_q90"], float)

TAIL_NAMES = ("force_peak","E_peak","tau_peak","body_peak")
B5_TAIL_ADD = {}
tail_rows = []

for j, name in enumerate(TAIL_NAMES):
    residual = val_tail_actual[:, j] - val_tail_pred[:, j]

    # One-sided split-conformal style additive correction.
    add = float(max(0.0, np.quantile(residual, 0.90)))
    B5_TAIL_ADD[name] = add
    calibrated = val_tail_pred[:, j] + add

    tail_rows.append({
        "metric":name,
        "raw_q90_coverage":float(
            np.mean(val_tail_actual[:, j] <= val_tail_pred[:, j])
        ),
        "calibrated_q90_coverage":float(
            np.mean(val_tail_actual[:, j] <= calibrated)
        ),
        "additive_calibration":add,
    })

B5_TAIL_CALIBRATION = pd.DataFrame(tail_rows)
display(B5_TAIL_CALIBRATION)

# ----------------------------------------------------------------------
# 3. Touchdown threshold selected on VALIDATION only.
# ----------------------------------------------------------------------
td_true = B5_VAL["touchdown_truth"].reshape(-1).astype(int)
td_prob = B5_VAL["touchdown_probability"].reshape(-1)

best = None
for threshold in np.linspace(0.05, 0.95, 37):
    td_pred = (td_prob >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        td_true, td_pred,
        average="binary",
        zero_division=0,
    )
    row = (float(f1), float(p), float(r), float(threshold))
    if best is None or row[0] > best[0]:
        best = row

B5_TOUCHDOWN_THRESHOLD = best[3]
print(
    "validation-selected touchdown threshold:",
    B5_TOUCHDOWN_THRESHOLD,
    "precision:",best[1],
    "recall:",best[2],
    "F1:",best[0],
)

# ----------------------------------------------------------------------
# 4. Train-distribution physical reference scales.
# ----------------------------------------------------------------------
Ytr, _, _ = _b_concat(B_DEV_DATA, "Y_cont", "train")
TDtr, _, _ = _b_concat(B_DEV_DATA, "Y_touchdown", "train")
Itr, _, _ = _b_concat(B_DEV_DATA, "Y_impulse", "train")
tr_step = _b_reshape_cont(Ytr)

def _q95_positive(x, floor=1e-6):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(max(np.quantile(x, .95), floor))

tr_force = _b_group(tr_step, "force_odom").reshape(-1, B_H, 4, 3)
tr_tau = _b_group(tr_step, "tau").reshape(-1, B_H, 12)
tr_body = _b_group(tr_step, "body_B").reshape(-1, B_H, 12)
tr_E = _b_group(tr_step, "E").reshape(-1, B_H, 4, 3)
tr_foot = _b_group(tr_step, "foot_base").reshape(-1, B_H, 4, 3)

dt_h = np.diff(np.asarray(CFG.horizons_s, float))
tr_foot_speed = np.linalg.norm(
    np.diff(tr_foot, axis=1)
    / dt_h[None, :, None, None],
    axis=-1,
)

body_z = np.abs(
    (tr_body - B4_PREP["body_med"][None, None, :])
    / np.maximum(
        1.4826 * B4_PREP["body_mad"][None, None, :],
        1e-5,
    )
)

event_imp = Itr[TDtr.astype(bool)]
event_imp_norm = np.linalg.norm(event_imp, axis=-1)

B5_RISK_REFERENCE = {
    "force_norm_p95_N":_q95_positive(np.linalg.norm(tr_force, axis=-1)),
    "abs_tau_p95_Nm":_q95_positive(np.abs(tr_tau)),
    "body_robust_z_p95":_q95_positive(body_z),
    "E_norm_p95":_q95_positive(np.linalg.norm(tr_E, axis=-1)),
    "foot_speed_p95_mps":_q95_positive(tr_foot_speed),
    "touchdown_impulse_norm_p95_Ns":_q95_positive(event_imp_norm),
}
print("B5 risk references:", B5_RISK_REFERENCE)

def _b_ratio(x, ref):
    r = np.maximum(np.asarray(x, float), 0.0) / max(float(ref), 1e-8)
    return r / (1.0 + r)

def _calibrated_group_scale(scale, group):
    return np.asarray(scale, float) * float(B5_SCALE_FACTOR[group])

def B5_actual_risk(step, contact, touchdown, impulse):
    """Physical risk from actual future response."""
    step = np.asarray(step, float)
    n = len(step)

    f = _b_group(step, "force_odom").reshape(n, B_H, 4, 3)
    tau = _b_group(step, "tau").reshape(n, B_H, 12)
    body = _b_group(step, "body_B").reshape(n, B_H, 12)
    E = _b_group(step, "E").reshape(n, B_H, 4, 3)
    foot = _b_group(step, "foot_base").reshape(n, B_H, 4, 3)

    force_peak = np.max(np.linalg.norm(f, axis=-1), axis=(1,2))
    tau_peak = np.max(np.abs(tau), axis=(1,2))
    body_peak = np.max(np.abs(
        (body - B4_PREP["body_med"][None,None,:])
        / np.maximum(
            1.4826 * B4_PREP["body_mad"][None,None,:], 1e-5
        )
    ), axis=(1,2))
    E_peak = np.max(np.linalg.norm(E, axis=-1), axis=(1,2))
    foot_speed = np.linalg.norm(
        np.diff(foot, axis=1)
        / dt_h[None,:,None,None],
        axis=-1,
    )
    foot_peak = np.max(foot_speed, axis=(1,2))

    imp_norm = np.linalg.norm(np.asarray(impulse,float), axis=-1)
    impulse_peak = np.max(
        imp_norm * np.asarray(touchdown,float),
        axis=(1,2),
    )

    support = np.sum(np.asarray(contact,float), axis=-1)
    support_short = np.max(
        np.clip((2.0 - support) / 2.0, 0.0, 1.0),
        axis=1,
    )

    comp = {
        "force":_b_ratio(force_peak, B5_RISK_REFERENCE["force_norm_p95_N"]),
        "torque":_b_ratio(tau_peak, B5_RISK_REFERENCE["abs_tau_p95_Nm"]),
        "body":_b_ratio(body_peak, B5_RISK_REFERENCE["body_robust_z_p95"]),
        "E":_b_ratio(E_peak, B5_RISK_REFERENCE["E_norm_p95"]),
        "foot":_b_ratio(foot_peak, B5_RISK_REFERENCE["foot_speed_p95_mps"]),
        "impulse":_b_ratio(
            impulse_peak,
            B5_RISK_REFERENCE["touchdown_impulse_norm_p95_Ns"],
        ),
        "support":support_short,
    }
    physical = (
        .20*comp["force"] + .14*comp["torque"] + .14*comp["body"]
        + .14*comp["E"] + .10*comp["foot"] + .18*comp["impulse"]
        + .10*comp["support"]
    )
    return {**comp, "physical_risk":np.clip(physical,0,1)}

def B5_predicted_risk(pred):
    """
    Conservative risk from:
      calibrated direct q90 tail heads,
      calibrated Gaussian scale envelopes,
      touchdown probability + conditional impulse uncertainty.
    """
    mu = np.asarray(pred["continuous_mean"], float)
    scale = np.asarray(pred["continuous_scale"], float)
    n = len(mu)

    # Mean geometry/state.
    fmu = _b_group(mu, "force_odom").reshape(n,B_H,4,3)
    fsc = _b_group(scale, "force_odom").reshape(n,B_H,4,3)
    tmu = _b_group(mu, "tau").reshape(n,B_H,12)
    tsc = _b_group(scale, "tau").reshape(n,B_H,12)
    bmu = _b_group(mu, "body_B").reshape(n,B_H,12)
    bsc = _b_group(scale, "body_B").reshape(n,B_H,12)
    emu = _b_group(mu, "E").reshape(n,B_H,4,3)
    esc = _b_group(scale, "E").reshape(n,B_H,4,3)
    foot = _b_group(mu, "foot_base").reshape(n,B_H,4,3)

    z90 = 1.645

    # Calibrated component-scale envelopes.
    f_upper = np.max(
        np.linalg.norm(fmu,axis=-1)
        + z90*np.linalg.norm(
            _calibrated_group_scale(fsc,"force_odom"),axis=-1
        ),
        axis=(1,2),
    )
    E_upper = np.max(
        np.linalg.norm(emu,axis=-1)
        + z90*np.linalg.norm(
            _calibrated_group_scale(esc,"E"),axis=-1
        ),
        axis=(1,2),
    )
    tau_upper = np.max(
        np.abs(tmu)
        + z90*_calibrated_group_scale(tsc,"tau"),
        axis=(1,2),
    )

    body_scale = _calibrated_group_scale(bsc,"body_B")
    body_upper = np.max(
        (
            np.abs(bmu - B4_PREP["body_med"][None,None,:])
            + z90*body_scale
        )
        / np.maximum(
            1.4826*B4_PREP["body_mad"][None,None,:],1e-5
        ),
        axis=(1,2),
    )

    # Direct calibrated q90 sequence-level heads.
    qtail = np.asarray(pred["tail_q90"], float).copy()
    for j,name in enumerate(TAIL_NAMES):
        qtail[:,j] += B5_TAIL_ADD[name]

    force_tail = np.maximum(f_upper, qtail[:,0])
    E_tail = np.maximum(E_upper, qtail[:,1])
    tau_tail = np.maximum(tau_upper, qtail[:,2])
    body_tail = np.maximum(body_upper, qtail[:,3])

    foot_speed = np.linalg.norm(
        np.diff(foot,axis=1) / dt_h[None,:,None,None],
        axis=-1,
    )
    foot_peak = np.max(foot_speed,axis=(1,2))

    # Event-conditioned impulse upper envelope, weighted by touchdown hazard.
    imp_mu = np.asarray(pred["impulse_mean"],float)
    imp_sc = np.asarray(pred["impulse_scale"],float)
    imp_upper = (
        np.linalg.norm(imp_mu,axis=-1)
        + z90 * B5_IMPULSE_SCALE_FACTOR
        * np.linalg.norm(imp_sc,axis=-1)
    )
    td_prob = np.asarray(pred["touchdown_probability"],float)
    impulse_risk_value = np.max(td_prob * imp_upper, axis=(1,2))

    contact_prob = np.asarray(pred["contact_probability"],float)
    support = np.sum(contact_prob,axis=-1)
    support_short = np.max(
        np.clip((2.0-support)/2.0,0.0,1.0),
        axis=1,
    )

    comp = {
        "force":_b_ratio(force_tail,B5_RISK_REFERENCE["force_norm_p95_N"]),
        "torque":_b_ratio(tau_tail,B5_RISK_REFERENCE["abs_tau_p95_Nm"]),
        "body":_b_ratio(body_tail,B5_RISK_REFERENCE["body_robust_z_p95"]),
        "E":_b_ratio(E_tail,B5_RISK_REFERENCE["E_norm_p95"]),
        "foot":_b_ratio(foot_peak,B5_RISK_REFERENCE["foot_speed_p95_mps"]),
        "impulse":_b_ratio(
            impulse_risk_value,
            B5_RISK_REFERENCE["touchdown_impulse_norm_p95_Ns"],
        ),
        "support":support_short,
    }

    physical = (
        .20*comp["force"] + .14*comp["torque"] + .14*comp["body"]
        + .14*comp["E"] + .10*comp["foot"] + .18*comp["impulse"]
        + .10*comp["support"]
    )

    # Dimensionless uncertainty summary.
    rel_parts = []
    for group in ("force_odom","tau","body_B","E","foot_base"):
        s = _calibrated_group_scale(
            _b_group(scale,group),
            group,
        )
        ref = np.maximum(
            B4_PREP["y_std"][:, B_STEP_LAYOUT[group]],
            1e-5,
        )
        rel_parts.append(
            np.mean(
                np.clip(s/ref[None],0,5),
                axis=(1,2),
            )
        )
    uncertainty = np.mean(np.stack(rel_parts,axis=1),axis=1)
    uncertainty = uncertainty/(1.0+uncertainty)

    # MONOTONIC: uncertainty only ADDS risk.
    total = np.clip(physical + 0.15*uncertainty, 0.0, 1.0)

    return {
        **comp,
        "physical_risk":np.clip(physical,0,1),
        "uncertainty_risk":uncertainty,
        "total_risk":total,
        "force_upper":force_tail,
        "E_upper":E_tail,
        "tau_upper":tau_tail,
        "body_upper":body_tail,
    }

B5_CALIBRATION_SUMMARY = {
    "scale_factors":B5_SCALE_FACTOR,
    "impulse_scale_factor":B5_IMPULSE_SCALE_FACTOR,
    "tail_additive":B5_TAIL_ADD,
    "touchdown_threshold":B5_TOUCHDOWN_THRESHOLD,
    "risk_reference":B5_RISK_REFERENCE,
}
with open(B_ROOT/"tables"/"B5_calibration.json","w") as f:
    json.dump(B5_CALIBRATION_SUMMARY,f,indent=2)

print("\nB5 complete: validation-only calibration frozen.")


####################################################################################################
# B6
####################################################################################################


# ======================================================================
# B6 — CONTROLLED RE-RUN ON THE SAME TWO DEVELOPMENT TRAJECTORIES
#
# This is the clean A-vs-B ablation:
#   same runs, same held-out chronological tails,
#   but corrected targets/action encoder/loss/risk.
#
# We inspect:
#   - nominal group MAE
#   - high-force / high-E tails
#   - peak-force ordering
#   - touchdown F1
#   - conditional touchdown-impulse MAE
#   - predicted-vs-actual risk Spearman + top-decile overlap
#   - command-information benefit
# ======================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from IPython.display import display

if "B5_predicted_risk" not in globals():
    raise RuntimeError("Run B5 first.")

def _mae(a,b):
    return float(np.mean(np.abs(np.asarray(a)-np.asarray(b))))

B6_PREDICTIONS = {}
metric_rows = []
tail_rows = []
event_rows = []
risk_rows = []
benefit_rows = []

core_groups = ("q","dq","tau","force_odom","body_B","E","foot_base",
               "delta_position_base")

for rid,d in B_DEV_DATA.items():
    dense = predict_B(B4_DENSE,d,"test")
    noact = predict_B(B4_NOACTION,d,"test")
    B6_PREDICTIONS[rid] = {"dense":dense,"noaction":noact}

    truth = _b_reshape_cont(dense["truth"])
    mu = dense["continuous_mean"]
    mu0 = noact["continuous_mean"]

    # --------------------------------------------------------------
    # Nominal per-group MAE + future-command benefit.
    # --------------------------------------------------------------
    for g in core_groups:
        sl = B_STEP_LAYOUT[g]
        e1 = _mae(mu[:,:,sl],truth[:,:,sl])
        e0 = _mae(mu0[:,:,sl],truth[:,:,sl])
        improvement = (e0-e1)/max(e0,1e-12)
        metric_rows.append({
            "run_id":rid,"group":g,
            "dense100hz_mae":e1,
            "noaction_mae":e0,
            "improvement_fraction":improvement,
        })

    # --------------------------------------------------------------
    # Tail metrics.
    # --------------------------------------------------------------
    ftrue = _b_group(truth,"force_odom").reshape(-1,B_H,4,3)
    fpred = _b_group(mu,"force_odom").reshape(-1,B_H,4,3)
    f0 = _b_group(mu0,"force_odom").reshape(-1,B_H,4,3)

    fnorm = np.linalg.norm(ftrue,axis=-1)
    fmask = fnorm > B4_PREP["force95"]

    if np.any(fmask):
        dense_comp = np.abs(fpred-ftrue)[fmask]
        noact_comp = np.abs(f0-ftrue)[fmask]
        dense_vec = np.linalg.norm(fpred-ftrue,axis=-1)[fmask]
        noact_vec = np.linalg.norm(f0-ftrue,axis=-1)[fmask]

        tail_rows.append({
            "run_id":rid,
            "metric":"force_above_train_p95",
            "events":int(fmask.sum()),
            "dense_component_mae":float(dense_comp.mean()),
            "dense_component_p95":float(np.quantile(dense_comp,.95)),
            "dense_vector_mae":float(dense_vec.mean()),
            "noaction_component_mae":float(noact_comp.mean()),
            "noaction_vector_mae":float(noact_vec.mean()),
        })

    Etrue = _b_group(truth,"E").reshape(-1,B_H,4,3)
    Epred = _b_group(mu,"E").reshape(-1,B_H,4,3)
    E0 = _b_group(mu0,"E").reshape(-1,B_H,4,3)
    Enorm = np.linalg.norm(Etrue,axis=-1)
    Emask = Enorm > B4_PREP["E95"]

    if np.any(Emask):
        tail_rows.append({
            "run_id":rid,
            "metric":"E_above_train_p95",
            "events":int(Emask.sum()),
            "dense_component_mae":float(
                np.abs(Epred-Etrue)[Emask].mean()
            ),
            "dense_component_p95":float(
                np.quantile(np.abs(Epred-Etrue)[Emask],.95)
            ),
            "dense_vector_mae":float(
                np.linalg.norm(Epred-Etrue,axis=-1)[Emask].mean()
            ),
            "noaction_component_mae":float(
                np.abs(E0-Etrue)[Emask].mean()
            ),
            "noaction_vector_mae":float(
                np.linalg.norm(E0-Etrue,axis=-1)[Emask].mean()
            ),
        })

    actual_peak = np.max(fnorm,axis=(1,2))
    pred_peak = np.max(np.linalg.norm(fpred,axis=-1),axis=(1,2))
    noact_peak = np.max(np.linalg.norm(f0,axis=-1),axis=(1,2))

    peak_spear = float(pd.Series(pred_peak).corr(
        pd.Series(actual_peak), method="spearman"
    ))
    peak_spear0 = float(pd.Series(noact_peak).corr(
        pd.Series(actual_peak), method="spearman"
    ))

    tail_rows.append({
        "run_id":rid,
        "metric":"force_peak_per_origin",
        "events":len(actual_peak),
        "dense_component_mae":float(np.mean(np.abs(pred_peak-actual_peak))),
        "dense_component_p95":float(np.quantile(np.abs(pred_peak-actual_peak),.95)),
        "dense_vector_mae":peak_spear,  # explicit label in metric column
        "noaction_component_mae":float(np.mean(np.abs(noact_peak-actual_peak))),
        "noaction_vector_mae":peak_spear0,
    })

    # --------------------------------------------------------------
    # Touchdown + conditional impulse.
    # --------------------------------------------------------------
    td_true = dense["touchdown_truth"].reshape(-1).astype(int)
    td_pred = (
        dense["touchdown_probability"].reshape(-1)
        >= B5_TOUCHDOWN_THRESHOLD
    ).astype(int)
    pr,rc,f1,_ = precision_recall_fscore_support(
        td_true,td_pred,average="binary",zero_division=0
    )

    event_mask = dense["touchdown_truth"].astype(bool)
    if np.any(event_mask):
        imp_err = np.abs(
            dense["impulse_mean"] - dense["impulse_truth"]
        )[event_mask]
        imp_vec = np.linalg.norm(
            dense["impulse_mean"]-dense["impulse_truth"],
            axis=-1,
        )[event_mask]
        impulse_component_mae = float(imp_err.mean())
        impulse_vector_mae = float(imp_vec.mean())
        impulse_vector_p95 = float(np.quantile(imp_vec,.95))
    else:
        impulse_component_mae = np.nan
        impulse_vector_mae = np.nan
        impulse_vector_p95 = np.nan

    event_rows.append({
        "run_id":rid,
        "threshold":B5_TOUCHDOWN_THRESHOLD,
        "touchdown_precision":float(pr),
        "touchdown_recall":float(rc),
        "touchdown_f1":float(f1),
        "true_touchdowns":int(td_true.sum()),
        "conditional_impulse_component_mae_Ns":impulse_component_mae,
        "conditional_impulse_vector_mae_Ns":impulse_vector_mae,
        "conditional_impulse_vector_p95_Ns":impulse_vector_p95,
    })

    # --------------------------------------------------------------
    # Corrected upper-tail risk.
    # --------------------------------------------------------------
    pred_r = B5_predicted_risk(dense)
    act_r = B5_actual_risk(
        truth,
        dense["contact_truth"],
        dense["touchdown_truth"],
        dense["impulse_truth"],
    )

    spear = float(pd.Series(pred_r["physical_risk"]).corr(
        pd.Series(act_r["physical_risk"]),method="spearman"
    ))
    k = max(1,int(round(.10*len(truth))))
    ptop = set(np.argpartition(pred_r["physical_risk"],-k)[-k:])
    atop = set(np.argpartition(act_r["physical_risk"],-k)[-k:])
    overlap = len(ptop & atop)/float(k)

    risk_rows.append({
        "run_id":rid,
        "samples":len(truth),
        "risk_spearman":spear,
        "top10_overlap":overlap,
        "median_predicted_physical":float(np.median(pred_r["physical_risk"])),
        "median_predicted_total":float(np.median(pred_r["total_risk"])),
        "median_actual":float(np.median(act_r["physical_risk"])),
        "force_peak_spearman":peak_spear,
    })

    # --------------------------------------------------------------
    # Stress plot around the largest actual first-horizon FL/Fz/force region.
    # --------------------------------------------------------------
    origins = dense["origins"]
    t = d["time"][origins]
    fh0 = np.linalg.norm(ftrue[:,0],axis=-1).max(axis=1)
    center = int(np.argmax(fh0))
    half = min(300, max(60,len(t)//8))
    lo=max(0,center-half); hi=min(len(t),center+half)
    slc=slice(lo,hi)
    rel_t=t[slc]-t[lo]

    fscale = _b_group(
        dense["continuous_scale"],"force_odom"
    ).reshape(-1,B_H,4,3)
    scale_factor = B5_SCALE_FACTOR["force_odom"]
    upper_fz = (
        fpred[:,0,0,2]
        + 1.645*scale_factor*fscale[:,0,0,2]
    )

    fig,ax=plt.subplots(figsize=(14,4.5))
    ax.plot(rel_t,ftrue[slc,0,0,2],label="actual FL Fz @0.1s")
    ax.plot(rel_t,fpred[slc,0,0,2],label="B mean")
    ax.plot(
        rel_t,upper_fz[slc],
        label="B calibrated upper envelope",
        alpha=.75,
    )
    ax.set_title(f"{rid}: B6 high-force stress window")
    ax.set_xlabel("relative held-out time [s]")
    ax.set_ylabel("N")
    ax.grid(alpha=.2); ax.legend()
    plt.tight_layout()
    plt.savefig(
        B_ROOT/"figures"/f"B6_{rid}_force_stress.png",
        dpi=160,bbox_inches="tight",
    )
    plt.show()

B6_NOMINAL = pd.DataFrame(metric_rows)
B6_TAIL = pd.DataFrame(tail_rows)
B6_EVENTS = pd.DataFrame(event_rows)
B6_RISK_DIAGNOSTIC = pd.DataFrame(risk_rows)

B6_COMMAND_BENEFIT = (
    B6_NOMINAL.groupby("group",as_index=False)
    .agg(
        dense_mae=("dense100hz_mae","mean"),
        noaction_mae=("noaction_mae","mean"),
        mean_improvement_fraction=("improvement_fraction","mean"),
    )
)

print("\nB6 NOMINAL / COMMAND BENEFIT")
display(B6_COMMAND_BENEFIT)
print("\nB6 TAIL METRICS")
display(B6_TAIL)
print("\nB6 EVENT METRICS")
display(B6_EVENTS)
print("\nB6 RISK")
display(B6_RISK_DIAGNOSTIC)

for name,tab in {
    "B6_nominal.csv":B6_NOMINAL,
    "B6_command_benefit.csv":B6_COMMAND_BENEFIT,
    "B6_tail.csv":B6_TAIL,
    "B6_events.csv":B6_EVENTS,
    "B6_risk.csv":B6_RISK_DIAGNOSTIC,
}.items():
    tab.to_csv(B_ROOT/"tables"/name,index=False)

old_risk = np.nan
old_top = np.nan
if "V10R_RISK_DIAGNOSTIC" in globals():
    old_risk = float(
        V10R_RISK_DIAGNOSTIC[
            "predicted_vs_actual_risk_spearman"
        ].mean()
    )
    old_top = float(V10R_RISK_DIAGNOSTIC["top_decile_overlap"].mean())

new_risk = float(B6_RISK_DIAGNOSTIC["risk_spearman"].mean())
new_top = float(B6_RISK_DIAGNOSTIC["top10_overlap"].mean())

core = {"force_odom","tau","body_B","foot_base"}
cmd_benefit = float(
    B6_COMMAND_BENEFIT[
        B6_COMMAND_BENEFIT["group"].isin(core)
    ]["mean_improvement_fraction"].mean()
)
td_f1 = float(B6_EVENTS["touchdown_f1"].mean())
peak_spear = float(B6_RISK_DIAGNOSTIC["force_peak_spearman"].mean())

B6_GATE = {
    "old_A_mean_risk_spearman":old_risk,
    "new_B_mean_risk_spearman":new_risk,
    "risk_spearman_change":(
        new_risk-old_risk if np.isfinite(old_risk) else np.nan
    ),
    "old_A_mean_top10_overlap":old_top,
    "new_B_mean_top10_overlap":new_top,
    "top10_change":(
        new_top-old_top if np.isfinite(old_top) else np.nan
    ),
    "mean_core_command_benefit":cmd_benefit,
    "mean_touchdown_f1":td_f1,
    "mean_force_peak_spearman":peak_spear,
    "repair_improved_risk":bool(
        (not np.isfinite(old_risk)) or new_risk > old_risk
    ),
    "original_trend_gate_pass":bool(
        new_risk >= .40 and new_top >= .20
    ),
    "tail_event_readiness_gate":bool(
        peak_spear >= .40 and td_f1 >= .50
    ),
    "note":(
        "Development-run diagnostic only. B7-B10 are required for "
        "independent-trajectory evidence."
    ),
}

print("\n"+"="*80)
print("B6 DEVELOPMENT-TRAJECTORY REPAIR GATE")
print("="*80)
print(B6_GATE)

with open(B_ROOT/"tables"/"B6_gate.json","w") as f:
    import json
    json.dump(B6_GATE,f,indent=2)

print("\nB6 complete. B7 may now enumerate independent trajectories.")


####################################################################################################
# B7
####################################################################################################


# ======================================================================
# B7 — DISCOVER REAL ADDITIONAL TRAJECTORIES + DOWNLOAD IMU/ROSBAG ONLY
#
# NO RGB-D is downloaded in Stage B.
#
# Selection rules:
#   - never invent Pxxxx IDs;
#   - enumerate actual ANYmal run folders from the public repository;
#   - previously inspected dev run is TRAIN only;
#   - persist the split manifest once so val/test identities never drift;
#   - target 5 trajectories/terrain when available:
#       3 train (including old dev), 1 val, 1 untouched test.
# ======================================================================

import sys, subprocess, importlib.util, hashlib, re, json
from pathlib import Path
import pandas as pd

if "B6_GATE" not in globals():
    raise RuntimeError("Run B1-B6 first.")

def _b7_ensure(import_name, pip_name=None):
    if importlib.util.find_spec(import_name) is None:
        subprocess.check_call([
            sys.executable,"-m","pip","install","-q",
            pip_name or import_name
        ])

_b7_ensure("huggingface_hub")
if importlib.util.find_spec("tartanair") is None:
    subprocess.check_call([
        sys.executable,"-m","pip","install","-q",
        "git+https://github.com/castacks/tartanairpy.git"
    ])

from huggingface_hub import HfApi
import tartanair as ta

B7_REPO_ID = "theairlabcmu/TartanGround"
B7_TARGET_PER_TERRAIN = 5
B7_MIN_PER_TERRAIN = 4
B7_SPLIT_SEED = "V10R_B_SPLIT_V1"
B7_SPLIT_PATH = B_ROOT / "multitrajectory_split_manifest.csv"

B7_ENVIRONMENTS = {
    "swamp_marsh":{
        "environment":"GreatMarsh",
        "development_run":"P2023",
        "development_run_id":"swamp_marsh_P2023",
    },
    "rough_uneven":{
        "environment":"SeasonalForestWinter",
        "development_run":"P2000",
        "development_run_id":"rough_uneven_P2000",
    },
}

def _discover_anymal_runs(environment):
    api = HfApi()
    errors = []
    candidate_paths = [
        f"{environment}/Data_anymal",
        f"data/{environment}/Data_anymal",
    ]

    for path_in_repo in candidate_paths:
        try:
            items = list(api.list_repo_tree(
                repo_id=B7_REPO_ID,
                repo_type="dataset",
                path_in_repo=path_in_repo,
                recursive=False,
            ))
            runs = sorted({
                Path(getattr(x,"path","")).name
                for x in items
                if re.fullmatch(
                    r"P\d{4}",
                    Path(getattr(x,"path","")).name,
                )
            })
            if runs:
                return runs, path_in_repo
        except Exception as exc:
            errors.append(f"{path_in_repo}: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        f"Could not enumerate {environment} ANYmal runs from "
        f"{B7_REPO_ID}. Tried: {errors}"
    )

def _stable_rank(environment, run):
    s = f"{B7_SPLIT_SEED}|{environment}|{run}".encode()
    return hashlib.sha256(s).hexdigest()

if B7_SPLIT_PATH.exists():
    B7_SPLIT_MANIFEST = pd.read_csv(B7_SPLIT_PATH)
    print("Reusing frozen B7 split manifest:",B7_SPLIT_PATH)
else:
    split_rows = []
    discovery_rows = []

    for terrain,cfg in B7_ENVIRONMENTS.items():
        env = cfg["environment"]
        dev = cfg["development_run"]

        runs, repo_path = _discover_anymal_runs(env)
        discovery_rows.append({
            "terrain":terrain,
            "environment":env,
            "repo_path":repo_path,
            "available_runs":len(runs),
            "runs_preview":",".join(runs[:20]),
        })

        if dev not in runs:
            raise RuntimeError(
                f"Known development run {env}/{dev} was not found "
                "in repository enumeration."
            )

        extras = [r for r in runs if r != dev]
        extras = sorted(extras,key=lambda r:_stable_rank(env,r))

        need_extra = B7_TARGET_PER_TERRAIN - 1
        if len(extras) < (B7_MIN_PER_TERRAIN - 1):
            raise RuntimeError(
                f"{env}: only {len(extras)+1} usable trajectories found; "
                f"need at least {B7_MIN_PER_TERRAIN}."
            )

        chosen_extra = extras[:min(need_extra,len(extras))]
        chosen = [dev] + chosen_extra

        # Need two held-out trajectories: one validation and one test.
        if len(chosen) < B7_MIN_PER_TERRAIN:
            raise RuntimeError(f"{env}: insufficient selected runs {chosen}")

        train_runs = chosen[:-2]
        val_run = chosen[-2]
        test_run = chosen[-1]

        for run in chosen:
            role = (
                "train" if run in train_runs
                else "val" if run == val_run
                else "test"
            )
            run_id = (
                cfg["development_run_id"]
                if run == dev
                else f"{terrain}_{run}"
            )
            split_rows.append({
                "terrain":terrain,
                "environment":env,
                "run":run,
                "run_id":run_id,
                "trajectory_id":f"{env}_{run}",
                "role":role,
                "previously_inspected_dev":bool(run==dev),
            })

    B7_DISCOVERY = pd.DataFrame(discovery_rows)
    display(B7_DISCOVERY)

    B7_SPLIT_MANIFEST = pd.DataFrame(split_rows)

    # Hard split guards.
    for terrain,g in B7_SPLIT_MANIFEST.groupby("terrain"):
        if g["run"].nunique() < B7_MIN_PER_TERRAIN:
            raise RuntimeError(f"{terrain}: too few independent runs")
        if set(g["role"]) != {"train","val","test"}:
            raise RuntimeError(f"{terrain}: incomplete train/val/test roles")
        if g.loc[g["previously_inspected_dev"],"role"].ne("train").any():
            raise RuntimeError(
                f"{terrain}: inspected development trajectory entered val/test"
            )
        if g.loc[g["role"]=="test","previously_inspected_dev"].any():
            raise RuntimeError("Inspected trajectory assigned to test")

    B7_SPLIT_MANIFEST.to_csv(B7_SPLIT_PATH,index=False)
    print("Frozen split manifest written:",B7_SPLIT_PATH)

display(
    B7_SPLIT_MANIFEST.sort_values(["terrain","role","run"])
)

# ----------------------------------------------------------------------
# Download ONLY IMU + ROS bag.
# ----------------------------------------------------------------------
ta.init(str(RAW_ROOT))

def _minimal_raw_ready(row):
    base = (
        RAW_ROOT
        / row.environment
        / "Data_anymal"
        / row.run
    )
    required = [
        base/"imu"/"imu_time.npy",
        base/"imu"/"acc.npy",
        base/"imu"/"gyro.txt",
        base/"rosbags"/f"{row.run}.bag",
    ]
    return base, all(
        p.is_file() and p.stat().st_size>0
        for p in required
    ), required

download_rows = []

for row in B7_SPLIT_MANIFEST.itertuples(index=False):
    base,ready,required = _minimal_raw_ready(row)

    if not ready:
        print(
            "\nDownloading physics-only evidence:",
            row.environment,row.run,
            "(imu + rosbag)"
        )
        ta.download_ground(
            env=[row.environment],
            version=["anymal"],
            traj=[row.run],
            modality=["imu","rosbag"],
            camera_name=["lcam_front"],
            unzip=True,
        )
        base,ready,required = _minimal_raw_ready(row)

    missing = [str(p) for p in required if not p.is_file()]
    download_rows.append({
        "run_id":row.run_id,
        "terrain":row.terrain,
        "role":row.role,
        "ready":bool(ready),
        "raw_path":str(base),
        "missing":" | ".join(missing),
    })

B7_DOWNLOAD_STATUS = pd.DataFrame(download_rows)
display(B7_DOWNLOAD_STATUS)

if not B7_DOWNLOAD_STATUS["ready"].all():
    raise RuntimeError(
        "B7 download incomplete for: "
        + ", ".join(
            B7_DOWNLOAD_STATUS.loc[
                ~B7_DOWNLOAD_STATUS["ready"],"run_id"
            ].tolist()
        )
    )

print(
    "\nB7 complete. No RGB or depth was downloaded. "
    "Test trajectory identities are now frozen; do not change them."
)


####################################################################################################
# B8
####################################################################################################


# ======================================================================
# B8 — BUILD REAL PHYSICS / TF / CONTACT / IMPULSE TARGETS FOR ALL RUNS
#
# Reuses the physics-extraction logic that produced the original caches:
#   time, P[q,dq,tau,F], B, U, L_LEG, E, contact
#
# Then reuses the proven A1/A2 native evidence path:
#   force topic <-> AnymalState equivalence
#   recorded TF graph -> foot trajectories
#   native contact transitions -> 50 ms touchdown impulse
#
# NO RGB-D is required.
# ======================================================================

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from rosbags.highlevel import AnyReader
from IPython.display import display

if "B7_SPLIT_MANIFEST" not in globals():
    raise RuntimeError("Run B7 first.")

B8_CACHE_ROOT = PROJECT_ROOT / "cache"
B8_CACHE_ROOT.mkdir(parents=True,exist_ok=True)
B8_CONTACT_THRESHOLD_N = 10.0

def b8_causal_hold(t_src,x_src,t_target):
    t_src=np.asarray(t_src,float)
    x_src=np.asarray(x_src)
    t_target=np.asarray(t_target,float)
    idx=np.searchsorted(t_src,t_target,side="right")-1
    if np.any(idx<0):
        raise RuntimeError("Target precedes first source measurement")
    idx=np.clip(idx,0,len(t_src)-1)
    return x_src[idx]

def b8_stamp_s(stamp):
    return float(stamp.sec)+float(stamp.nanosec)*1e-9

def b8_extract_physics(spec):
    """
    Build the exact persistent physics cache needed by B1/B9.
    Existing valid caches are reused.
    """
    env=spec["environment"]; run=spec["run"]
    cache=B8_CACHE_ROOT/f"{env}_{run}_physics.npz"

    required_keys={"time","P","B","U","L_LEG","E","contact"}

    if cache.exists():
        try:
            with np.load(cache,allow_pickle=False) as z:
                if required_keys.issubset(z.files):
                    n=len(z["time"])
                    if (
                        np.asarray(z["P"]).shape==(n,4,12)
                        and np.asarray(z["L_LEG"]).shape==(n,4,3)
                        and np.asarray(z["E"]).shape==(n,4,3)
                        and np.asarray(z["contact"]).shape==(n,4)
                    ):
                        print("Reusing physics cache:",cache)
                        return cache
        except Exception:
            pass

    base=raw_run_path(spec)
    imu_dir=base/"imu"
    bag_path=base/"rosbags"/f"{run}.bag"

    if not bag_path.is_file():
        raise FileNotFoundError(bag_path)

    imu_time=np.load(imu_dir/"imu_time.npy").astype(np.float64)
    acc=np.load(imu_dir/"acc.npy").astype(np.float32)
    gyro=np.loadtxt(imu_dir/"gyro.txt").astype(np.float32)
    if gyro.ndim==1:
        gyro=gyro.reshape(-1,3)

    if not (len(imu_time)==len(acc)==len(gyro)):
        raise RuntimeError(
            f"{spec['run_id']}: IMU length mismatch "
            f"{len(imu_time)}, {len(acc)}, {len(gyro)}"
        )

    topics={
        "joints":"/state_estimator/joint_states",
        "lf":"/state_estimator/contact_force_lf_foot",
        "rf":"/state_estimator/contact_force_rf_foot",
        "lh":"/state_estimator/contact_force_lh_foot",
        "rh":"/state_estimator/contact_force_rh_foot",
        "twist":"/state_estimator/twist",
        "command":"/path_planning_and_following/twist",
    }

    physical_topics={topics[k] for k in ["joints","lf","rf","lh","rh","twist"]}

    with AnyReader([bag_path]) as reader:
        selected=[c for c in reader.connections if c.topic in physical_topics]
        if not selected:
            raise RuntimeError(f"{spec['run_id']}: physical ROS topics missing")
        _,t0_ns,_=next(reader.messages(connections=selected))
    t0_sec=t0_ns*1e-9

    def msg_time(msg,bag_ns):
        try:
            return b8_stamp_s(msg.header.stamp)-t0_sec
        except Exception:
            return bag_ns*1e-9-t0_sec

    data={k:{"t":[],"x":[]} for k in topics}
    expected=[
        "LF_HAA","LF_HFE","LF_KFE",
        "RF_HAA","RF_HFE","RF_KFE",
        "LH_HAA","LH_HFE","LH_KFE",
        "RH_HAA","RH_HFE","RH_KFE",
    ]
    joint_reorder=None

    with AnyReader([bag_path]) as reader:
        selected=[
            c for c in reader.connections
            if c.topic in set(topics.values())
        ]
        found_topics={c.topic for c in selected}
        missing=set(topics.values())-found_topics
        if missing:
            raise RuntimeError(
                f"{spec['run_id']}: required topics missing {sorted(missing)}"
            )

        for c,bag_ns,raw in reader.messages(connections=selected):
            msg=reader.deserialize(raw,c.msgtype)
            topic=c.topic
            tm=msg_time(msg,bag_ns)

            if topic==topics["joints"]:
                if joint_reorder is None:
                    names=list(msg.name)
                    miss=[x for x in expected if x not in names]
                    if miss:
                        raise RuntimeError(
                            f"{spec['run_id']}: missing joints {miss}"
                        )
                    joint_reorder=np.asarray(
                        [names.index(x) for x in expected],dtype=int
                    )
                pos=np.asarray(msg.position,np.float32)[joint_reorder]
                vel=np.asarray(msg.velocity,np.float32)[joint_reorder]
                eff=np.asarray(msg.effort,np.float32)[joint_reorder]
                x=np.concatenate([pos,vel,eff])
                key="joints"

            elif topic in {
                topics["lf"],topics["rf"],topics["lh"],topics["rh"]
            }:
                x=np.asarray([
                    msg.wrench.force.x,
                    msg.wrench.force.y,
                    msg.wrench.force.z,
                ],np.float32)
                key=(
                    "lf" if topic==topics["lf"] else
                    "rf" if topic==topics["rf"] else
                    "lh" if topic==topics["lh"] else "rh"
                )

            elif topic==topics["twist"]:
                tw=msg.twist.twist
                x=np.asarray([
                    tw.linear.x,tw.linear.y,tw.linear.z,
                    tw.angular.x,tw.angular.y,tw.angular.z,
                ],np.float32)
                key="twist"

            elif topic==topics["command"]:
                tw=msg.twist
                x=np.asarray([
                    tw.linear.x,tw.linear.y,tw.linear.z,
                    tw.angular.x,tw.angular.y,tw.angular.z,
                ],np.float32)
                key="command"

            else:
                continue

            data[key]["t"].append(tm)
            data[key]["x"].append(x)

    for k in data:
        data[k]["t"]=np.asarray(data[k]["t"],np.float64)
        data[k]["x"]=np.asarray(data[k]["x"],np.float32)
        if len(data[k]["t"])==0:
            raise RuntimeError(f"{spec['run_id']}: empty stream {k}")
        order=np.argsort(data[k]["t"])
        data[k]["t"]=data[k]["t"][order]
        data[k]["x"]=data[k]["x"][order]

    stream_keys=["joints","lf","rf","lh","rh","twist","command"]
    common_start=max(
        float(imu_time[0]),
        *[float(data[k]["t"][0]) for k in stream_keys],
    )
    common_end=min(
        float(imu_time[-1]),
        *[float(data[k]["t"][-1]) for k in stream_keys],
    )
    mask=(imu_time>=common_start)&(imu_time<=common_end)
    grid=imu_time[mask]

    if len(grid)<2000:
        raise RuntimeError(
            f"{spec['run_id']}: common synchronized window too short ({len(grid)})"
        )

    acc_sync=acc[mask]
    gyro_sync=gyro[mask]
    joints=b8_causal_hold(data["joints"]["t"],data["joints"]["x"],grid)
    lf=b8_causal_hold(data["lf"]["t"],data["lf"]["x"],grid)
    rf=b8_causal_hold(data["rf"]["t"],data["rf"]["x"],grid)
    lh=b8_causal_hold(data["lh"]["t"],data["lh"]["x"],grid)
    rh=b8_causal_hold(data["rh"]["t"],data["rh"]["x"],grid)
    twist=b8_causal_hold(data["twist"]["t"],data["twist"]["x"],grid)
    U=b8_causal_hold(data["command"]["t"],data["command"]["x"],grid).astype(np.float32)

    q=joints[:,0:12]
    dq=joints[:,12:24]
    tau=joints[:,24:36]
    legs=[[0,1,2],[3,4,5],[6,7,8],[9,10,11]]
    flist=[lf,rf,lh,rh]
    P=np.stack([
        np.concatenate(
            [q[:,idx],dq[:,idx],tau[:,idx],flist[j]],axis=1
        )
        for j,idx in enumerate(legs)
    ],axis=1).astype(np.float32)

    B=np.concatenate(
        [acc_sync,gyro_sync,twist],axis=1
    ).astype(np.float32)

    # --------------------------------------------------------------
    # TRUE low-level commanded joint torques.
    # --------------------------------------------------------------
    ACT="/state_estimator/actuator_readings"
    cmd_t=[]; cmd_tau=[]; cmd_mode=[]
    st_t=[]; st_pos=[]

    with AnyReader([bag_path]) as reader:
        conns=[c for c in reader.connections if c.topic==ACT]
        if not conns:
            raise RuntimeError(f"{spec['run_id']}: {ACT} missing")

        for c,bag_ns,raw in reader.messages(connections=conns):
            msg=reader.deserialize(raw,c.msgtype)
            readings=list(msg.readings)
            if len(readings)!=12:
                continue

            cs=[]; ct=[]; cm=[]; ss=[]; sp=[]
            for r in readings:
                cs.append(b8_stamp_s(r.commanded.header.stamp))
                ct.append(float(r.commanded.joint_torque))
                cm.append(int(r.commanded.mode))
                ss.append(b8_stamp_s(r.state.header.stamp))
                sp.append(float(r.state.joint_position))

            cmd_t.append(float(np.median(cs))-t0_sec)
            cmd_tau.append(ct)
            cmd_mode.append(cm)
            st_t.append(float(np.median(ss))-t0_sec)
            st_pos.append(sp)

    cmd_t=np.asarray(cmd_t,np.float64)
    cmd_tau=np.asarray(cmd_tau,np.float32)
    cmd_mode=np.asarray(cmd_mode,np.int32)
    st_t=np.asarray(st_t,np.float64)
    st_pos=np.asarray(st_pos,np.float32)

    if min(len(cmd_t),len(st_t))<100:
        raise RuntimeError(f"{spec['run_id']}: insufficient actuator readings")

    o=np.argsort(cmd_t)
    cmd_t=cmd_t[o]; cmd_tau=cmd_tau[o]; cmd_mode=cmd_mode[o]
    o=np.argsort(st_t)
    st_t=st_t[o]; st_pos=st_pos[o]

    # Avoid backward extrapolation if actuator stream begins later.
    valid_start=max(float(grid[0]),float(cmd_t[0]),float(st_t[0]))
    keep=grid>=valid_start
    grid=grid[keep]; P=P[keep]; B=B[keep]; U=U[keep]

    act_pos=b8_causal_hold(st_t,st_pos,grid)
    qcan=P[:,:,0:3].reshape(len(P),12)

    # Determine the bag's actuator ordering from actual joint positions.
    m=min(5000,len(P))
    cost=np.zeros((12,12),np.float64)
    for ai in range(12):
        for ji in range(12):
            diff=act_pos[:m,ai]-qcan[:m,ji]
            scale=np.std(qcan[:m,ji])+1e-6
            cost[ai,ji]=np.sqrt(np.mean(diff*diff))/scale

    rows,cols=linear_sum_assignment(cost)
    actuator_for_joint=np.empty(12,dtype=int)
    for ai,ji in zip(rows,cols):
        actuator_for_joint[ji]=ai

    assignment_cost=np.asarray([
        cost[actuator_for_joint[j],j] for j in range(12)
    ])
    if not np.isfinite(assignment_cost).all():
        raise RuntimeError(f"{spec['run_id']}: actuator mapping nonfinite")

    cmd_tau_sync=b8_causal_hold(cmd_t,cmd_tau,grid)
    cmd_mode_sync=b8_causal_hold(cmd_t,cmd_mode,grid)

    L=cmd_tau_sync[:,actuator_for_joint].astype(np.float32)
    L_MODE=cmd_mode_sync[:,actuator_for_joint]
    unsupported=~np.isin(L_MODE,[10,12,13])
    unsupported_fraction=float(unsupported.mean())
    if unsupported_fraction>0.01:
        raise RuntimeError(
            f"{spec['run_id']}: {unsupported_fraction:.3%} actuator "
            "samples outside torque-containing modes"
        )

    L_LEG=L.reshape(len(L),4,3)
    measured_tau=P[:,:,6:9]

    # Tracking residual is causal: measured tau(t) - command(t-1).
    E=np.zeros_like(measured_tau,np.float32)
    E[0]=measured_tau[0]-L_LEG[0]
    E[1:]=measured_tau[1:]-L_LEG[:-1]

    crop_start=float(grid[0])
    time=(grid-crop_start).astype(np.float64)
    contact=(
        np.linalg.norm(P[:,:,9:12],axis=-1)
        > B8_CONTACT_THRESHOLD_N
    ).astype(np.float32)

    if (
        P.shape!=(len(time),4,12)
        or B.shape!=(len(time),12)
        or U.shape!=(len(time),6)
        or L_LEG.shape!=(len(time),4,3)
        or E.shape!=(len(time),4,3)
    ):
        raise RuntimeError(f"{spec['run_id']}: final cache shape failure")

    if not all(np.isfinite(x).all() for x in [time,P,B,U,L_LEG,E]):
        raise RuntimeError(f"{spec['run_id']}: nonfinite physics cache")

    np.savez_compressed(
        cache,
        time=time,
        grid_original=grid.astype(np.float64),
        crop_start_original=np.array(crop_start),
        P=P,B=B,U=U,L_LEG=L_LEG,E=E,contact=contact,
        actuator_for_joint=actuator_for_joint,
        actuator_assignment_cost=assignment_cost.astype(np.float32),
        unsupported_command_mode_fraction=np.array(
            unsupported_fraction,dtype=np.float32
        ),
    )
    print("Wrote physics cache:",cache)
    return cache

# ----------------------------------------------------------------------
# Build exact run specifications from the frozen B7 manifest.
# ----------------------------------------------------------------------
_dev_lookup={
    (s["environment"],s["run"]):s
    for s in TWO_SPECS
}

B_ALL_SPECS=[]
for row in B7_SPLIT_MANIFEST.itertuples(index=False):
    key=(row.environment,row.run)
    if key in _dev_lookup:
        spec=dict(_dev_lookup[key])
    else:
        spec={
            "run_id":row.run_id,
            "terrain":row.terrain,
            "environment":row.environment,
            "run":row.run,
            "trajectory_id":row.trajectory_id,
            "gait":"unreported",
            "policy":"unreported",
            "speed_bin":"unreported",
        }
    spec["trajectory_role"]=row.role
    B_ALL_SPECS.append(spec)

# Extract/reuse caches first.
B8_CACHE_PATHS={}
for spec in B_ALL_SPECS:
    print("\n"+"="*80)
    print("B8 PHYSICS:",spec["run_id"],spec["trajectory_role"])
    print("="*80)
    B8_CACHE_PATHS[spec["run_id"]]=Path(b8_extract_physics(spec))

# Verify force equivalence on EVERY selected trajectory before derived targets.
B8_FORCE_TABLES=[]
B8_FORCE_WINNERS=[]
B8_NATIVE_STREAMS={}
if "V10R_NATIVE_STREAMS" not in globals():
    V10R_NATIVE_STREAMS={}

for spec in B_ALL_SPECS:
    print("\n"+"="*80)
    print("B8 FORCE-FRAME EQUIVALENCE:",spec["run_id"])
    print("="*80)

    native=collect_native_contract_streams(spec)
    table=force_equivalence_table(native)
    winners,resolution=summarize_force_resolution(table)

    winners=winners.copy()
    winners["trajectory_role"]=spec["trajectory_role"]
    B8_FORCE_TABLES.append(table)
    B8_FORCE_WINNERS.append(winners)

    if not resolution["verified_for_derived_adapter"]:
        raise RuntimeError(
            f"{spec['run_id']}: force equivalence did not pass"
        )
    if resolution["source_frame_supported_by_equivalence"]!="odom":
        raise RuntimeError(
            f"{spec['run_id']}: force frame is not the expected odom "
            f"equivalence: {resolution}"
        )

    B8_NATIVE_STREAMS[spec["run_id"]]=native
    V10R_NATIVE_STREAMS[spec["run_id"]]=native

B8_FORCE_EQUIVALENCE=pd.concat(B8_FORCE_TABLES,ignore_index=True)
B8_FORCE_WINNERS=pd.concat(B8_FORCE_WINNERS,ignore_index=True)
display(B8_FORCE_WINNERS)

# Build/rebuild the real TF/contact/impulse target products.
B_DERIVED_PATHS={}
derived_rows=[]

for spec in B_ALL_SPECS:
    print("\n"+"="*80)
    print("B8 REAL TARGETS:",spec["run_id"],spec["trajectory_role"])
    print("="*80)

    report=build_real_targets_for_run(spec)
    report["trajectory_role"]=spec["trajectory_role"]
    report["terrain"]=spec["terrain"]
    derived_rows.append(report)
    B_DERIVED_PATHS[spec["run_id"]]=Path(report["derived_path"])

B8_DERIVED_REPORT=pd.DataFrame(derived_rows)
display(B8_DERIVED_REPORT)

B8_FORCE_EQUIVALENCE.to_csv(
    B_ROOT/"tables"/"B8_force_equivalence_all_runs.csv",index=False
)
B8_FORCE_WINNERS.to_csv(
    B_ROOT/"tables"/"B8_force_winners_all_runs.csv",index=False
)
B8_DERIVED_REPORT.to_csv(
    B_ROOT/"tables"/"B8_derived_report_all_runs.csv",index=False
)

print("\nB8 complete: all selected trajectories have real physics + TF/contact/impulse targets.")


####################################################################################################
# B9
####################################################################################################


# ======================================================================
# B9 — TRAJECTORY-POOLED, TERRAIN-BALANCED TRAINING
#
# Entire trajectories are the split unit.
#   train trajectories -> optimization
#   val trajectories   -> early stopping + calibration
#   test trajectories  -> NEVER SCORED HERE
#
# Same B model as B4:
#   proprio/history (+ optional dense 100 Hz future commands)
#   -> future physical response
#
# NO RGB-D / VLM.
# ======================================================================

import json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from IPython.display import display

if "B_DERIVED_PATHS" not in globals():
    raise RuntimeError("Run B8 first.")

role_by_id=dict(zip(
    B7_SPLIT_MANIFEST["run_id"],
    B7_SPLIT_MANIFEST["role"],
))

B_ALL_DATA={}
for spec in B_ALL_SPECS:
    role=role_by_id[spec["run_id"]]
    B_ALL_DATA[spec["run_id"]]=build_B_run_data(
        spec,
        split_policy="trajectory",
        trajectory_role=role,
        derived_path=B_DERIVED_PATHS[spec["run_id"]],
    )

# Hard trajectory isolation checks.
split_rows=[]
for rid,d in B_ALL_DATA.items():
    roles=np.unique(d["split"])
    if len(roles)!=1:
        raise RuntimeError(f"{rid}: more than one trajectory role {roles}")
    split_rows.append({
        "run_id":rid,
        "terrain":d["terrain"],
        "role":roles[0],
        "origins":len(d["origins"]),
    })

B9_DATASET_TABLE=pd.DataFrame(split_rows)
display(B9_DATASET_TABLE)

for terrain,g in B9_DATASET_TABLE.groupby("terrain"):
    if set(g["role"])!={"train","val","test"}:
        raise RuntimeError(f"{terrain}: incomplete trajectory roles")
    if int((g["role"]=="test").sum())!=1:
        raise RuntimeError(f"{terrain}: expected exactly one untouched test run")
    if int((g["role"]=="val").sum())!=1:
        raise RuntimeError(f"{terrain}: expected exactly one validation run")

# Prove inspected A trajectories are train-only.
for row in B7_SPLIT_MANIFEST.itertuples(index=False):
    if bool(row.previously_inspected_dev) and row.role!="train":
        raise RuntimeError(
            f"Previously inspected {row.run_id} leaked into {row.role}"
        )

print("\nTraining pooled B9 models. TEST trajectories are not loaded by train_B_pair.")
B9_NOACTION,B9_DENSE,B9_PREP=train_B_pair(
    B_ALL_DATA,
    prefix="B9_multitrajectory",
    balanced_terrain=True,
    epochs=35,
)

B9_HISTORY=pd.concat(
    [B9_NOACTION["history"],B9_DENSE["history"]],
    ignore_index=True,
)
display(
    B9_HISTORY.groupby("model",as_index=False).agg(
        best_val=("val_loss","min"),
        epochs=("epoch","max"),
    )
)
B9_HISTORY.to_csv(
    B_ROOT/"tables"/"B9_training_history.csv",index=False
)

# ----------------------------------------------------------------------
# Fit ALL calibration choices from whole validation trajectories only.
# No test prediction is made here.
# ----------------------------------------------------------------------
def fit_B9_calibration(bundle,data_dict):
    prep=bundle["prep"]
    val=_concat_B_predictions(bundle,data_dict,"val")
    truth=_b_reshape_cont(val["truth"])
    mu=np.asarray(val["continuous_mean"],float)
    scale=np.asarray(val["continuous_scale"],float)

    scale_factor={}
    cov_rows=[]
    for group,sl in B_STEP_LAYOUT.items():
        err=np.abs(truth[:,:,sl]-mu[:,:,sl])
        sc=np.maximum(scale[:,:,sl],1e-6)
        ratio=err/sc
        factor=float(np.quantile(ratio[np.isfinite(ratio)],.95))
        factor=max(factor,1e-3)
        scale_factor[group]=factor
        cov_rows.append({
            "group":group,
            "raw_gaussian_95_coverage":float(np.mean(err<=1.96*sc)),
            "calibrated_marginal_coverage":float(np.mean(err<=factor*sc)),
            "scale_factor":factor,
        })

    td=val["touchdown_truth"].astype(bool)
    if np.any(td):
        ie=np.abs(val["impulse_truth"]-val["impulse_mean"])
        isc=np.maximum(val["impulse_scale"],1e-6)
        impulse_factor=float(np.quantile(ie[td]/isc[td],.95))
    else:
        impulse_factor=1.96

    tail_actual=_b_tail_raw(
        truth,prep["body_med"],prep["body_mad"]
    )
    tail_pred=np.asarray(val["tail_q90"],float)
    tail_add={}
    tail_rows=[]
    for j,name in enumerate(TAIL_NAMES):
        add=float(max(
            0.0,
            np.quantile(tail_actual[:,j]-tail_pred[:,j],.90),
        ))
        tail_add[name]=add
        tail_rows.append({
            "metric":name,
            "raw_q90_coverage":float(
                np.mean(tail_actual[:,j]<=tail_pred[:,j])
            ),
            "calibrated_q90_coverage":float(
                np.mean(tail_actual[:,j]<=tail_pred[:,j]+add)
            ),
            "additive_calibration":add,
        })

    # Validation-only touchdown threshold.
    y=val["touchdown_truth"].reshape(-1).astype(int)
    p=val["touchdown_probability"].reshape(-1)
    best=None
    for th in np.linspace(.05,.95,37):
        yp=(p>=th).astype(int)
        pr,rc,f1,_=precision_recall_fscore_support(
            y,yp,average="binary",zero_division=0
        )
        row=(float(f1),float(pr),float(rc),float(th))
        if best is None or row[0]>best[0]:
            best=row

    # Train-only physical references.
    Ytr,_,_=_b_concat(data_dict,"Y_cont","train")
    TDtr,_,_=_b_concat(data_dict,"Y_touchdown","train")
    Itr,_,_=_b_concat(data_dict,"Y_impulse","train")
    st=_b_reshape_cont(Ytr)

    f=_b_group(st,"force_odom").reshape(-1,B_H,4,3)
    tau=_b_group(st,"tau").reshape(-1,B_H,12)
    body=_b_group(st,"body_B").reshape(-1,B_H,12)
    E=_b_group(st,"E").reshape(-1,B_H,4,3)
    foot=_b_group(st,"foot_base").reshape(-1,B_H,4,3)

    body_z=np.abs(
        (body-prep["body_med"][None,None,:])
        / np.maximum(1.4826*prep["body_mad"][None,None,:],1e-5)
    )
    foot_speed=np.linalg.norm(
        np.diff(foot,axis=1)
        / np.diff(np.asarray(CFG.horizons_s,float))[None,:,None,None],
        axis=-1,
    )
    event_imp=Itr[TDtr.astype(bool)]
    if len(event_imp)==0:
        raise RuntimeError("No train touchdown impulses in B9")

    refs={
        "force_norm_p95_N":_q95_positive(np.linalg.norm(f,axis=-1)),
        "abs_tau_p95_Nm":_q95_positive(np.abs(tau)),
        "body_robust_z_p95":_q95_positive(body_z),
        "E_norm_p95":_q95_positive(np.linalg.norm(E,axis=-1)),
        "foot_speed_p95_mps":_q95_positive(foot_speed),
        "touchdown_impulse_norm_p95_Ns":_q95_positive(
            np.linalg.norm(event_imp,axis=-1)
        ),
    }

    return {
        "scale_factor":scale_factor,
        "impulse_scale_factor":impulse_factor,
        "tail_add":tail_add,
        "touchdown_threshold":best[3],
        "touchdown_val_precision":best[1],
        "touchdown_val_recall":best[2],
        "touchdown_val_f1":best[0],
        "risk_reference":refs,
        "coverage_table":pd.DataFrame(cov_rows),
        "tail_table":pd.DataFrame(tail_rows),
    }

B9_CALIBRATION=fit_B9_calibration(B9_DENSE,B_ALL_DATA)
print("\nB9 validation calibration:")
display(B9_CALIBRATION["coverage_table"])
display(B9_CALIBRATION["tail_table"])
print(
    "touchdown validation threshold/F1:",
    B9_CALIBRATION["touchdown_threshold"],
    B9_CALIBRATION["touchdown_val_f1"],
)

# ----------------------------------------------------------------------
# Generic pooled risk functions, parameterized by the frozen B9 calibration.
# ----------------------------------------------------------------------
def B9_actual_risk(step,contact,touchdown,impulse,prep,cal):
    step=np.asarray(step,float)
    n=len(step)
    refs=cal["risk_reference"]
    dt=np.diff(np.asarray(CFG.horizons_s,float))

    f=_b_group(step,"force_odom").reshape(n,B_H,4,3)
    tau=_b_group(step,"tau").reshape(n,B_H,12)
    body=_b_group(step,"body_B").reshape(n,B_H,12)
    E=_b_group(step,"E").reshape(n,B_H,4,3)
    foot=_b_group(step,"foot_base").reshape(n,B_H,4,3)

    vals={}
    vals["force"]=np.max(np.linalg.norm(f,axis=-1),axis=(1,2))
    vals["torque"]=np.max(np.abs(tau),axis=(1,2))
    vals["body"]=np.max(np.abs(
        (body-prep["body_med"][None,None,:])
        / np.maximum(1.4826*prep["body_mad"][None,None,:],1e-5)
    ),axis=(1,2))
    vals["E"]=np.max(np.linalg.norm(E,axis=-1),axis=(1,2))
    vals["foot"]=np.max(np.linalg.norm(
        np.diff(foot,axis=1)/dt[None,:,None,None],axis=-1
    ),axis=(1,2))
    vals["impulse"]=np.max(
        np.linalg.norm(impulse,axis=-1)*touchdown,
        axis=(1,2),
    )
    support=np.sum(contact,axis=-1)
    vals["support"]=np.max(
        np.clip((2.0-support)/2.0,0,1),axis=1
    )

    comp={
        "force":_b_ratio(vals["force"],refs["force_norm_p95_N"]),
        "torque":_b_ratio(vals["torque"],refs["abs_tau_p95_Nm"]),
        "body":_b_ratio(vals["body"],refs["body_robust_z_p95"]),
        "E":_b_ratio(vals["E"],refs["E_norm_p95"]),
        "foot":_b_ratio(vals["foot"],refs["foot_speed_p95_mps"]),
        "impulse":_b_ratio(
            vals["impulse"],refs["touchdown_impulse_norm_p95_Ns"]
        ),
        "support":vals["support"],
    }
    physical=(
        .20*comp["force"]+.14*comp["torque"]+.14*comp["body"]
        +.14*comp["E"]+.10*comp["foot"]+.18*comp["impulse"]
        +.10*comp["support"]
    )
    return {**comp,"physical_risk":np.clip(physical,0,1)}

def B9_predicted_risk(pred,prep,cal):
    mu=np.asarray(pred["continuous_mean"],float)
    scale=np.asarray(pred["continuous_scale"],float)
    n=len(mu)
    refs=cal["risk_reference"]
    z90=1.645
    dt=np.diff(np.asarray(CFG.horizons_s,float))

    def gs(group):
        return (
            _b_group(scale,group)
            * float(cal["scale_factor"][group])
        )

    fmu=_b_group(mu,"force_odom").reshape(n,B_H,4,3)
    fsc=gs("force_odom").reshape(n,B_H,4,3)
    Emu=_b_group(mu,"E").reshape(n,B_H,4,3)
    Esc=gs("E").reshape(n,B_H,4,3)
    tmu=_b_group(mu,"tau").reshape(n,B_H,12)
    tsc=gs("tau").reshape(n,B_H,12)
    bmu=_b_group(mu,"body_B").reshape(n,B_H,12)
    bsc=gs("body_B").reshape(n,B_H,12)
    foot=_b_group(mu,"foot_base").reshape(n,B_H,4,3)

    f_up=np.max(
        np.linalg.norm(fmu,axis=-1)+z90*np.linalg.norm(fsc,axis=-1),
        axis=(1,2),
    )
    E_up=np.max(
        np.linalg.norm(Emu,axis=-1)+z90*np.linalg.norm(Esc,axis=-1),
        axis=(1,2),
    )
    tau_up=np.max(np.abs(tmu)+z90*tsc,axis=(1,2))
    body_up=np.max(
        (
            np.abs(bmu-prep["body_med"][None,None,:])+z90*bsc
        )/np.maximum(1.4826*prep["body_mad"][None,None,:],1e-5),
        axis=(1,2),
    )

    q=np.asarray(pred["tail_q90"],float).copy()
    for j,name in enumerate(TAIL_NAMES):
        q[:,j]+=cal["tail_add"][name]

    f_tail=np.maximum(f_up,q[:,0])
    E_tail=np.maximum(E_up,q[:,1])
    tau_tail=np.maximum(tau_up,q[:,2])
    body_tail=np.maximum(body_up,q[:,3])

    foot_peak=np.max(np.linalg.norm(
        np.diff(foot,axis=1)/dt[None,:,None,None],axis=-1
    ),axis=(1,2))

    imp_up=(
        np.linalg.norm(pred["impulse_mean"],axis=-1)
        +z90*float(cal["impulse_scale_factor"])
        *np.linalg.norm(pred["impulse_scale"],axis=-1)
    )
    imp_val=np.max(
        pred["touchdown_probability"]*imp_up,
        axis=(1,2),
    )

    support=np.sum(pred["contact_probability"],axis=-1)
    support_short=np.max(
        np.clip((2.0-support)/2.0,0,1),axis=1
    )

    comp={
        "force":_b_ratio(f_tail,refs["force_norm_p95_N"]),
        "torque":_b_ratio(tau_tail,refs["abs_tau_p95_Nm"]),
        "body":_b_ratio(body_tail,refs["body_robust_z_p95"]),
        "E":_b_ratio(E_tail,refs["E_norm_p95"]),
        "foot":_b_ratio(foot_peak,refs["foot_speed_p95_mps"]),
        "impulse":_b_ratio(
            imp_val,refs["touchdown_impulse_norm_p95_Ns"]
        ),
        "support":support_short,
    }
    physical=(
        .20*comp["force"]+.14*comp["torque"]+.14*comp["body"]
        +.14*comp["E"]+.10*comp["foot"]+.18*comp["impulse"]
        +.10*comp["support"]
    )

    # Monotonic additive uncertainty.
    rel=[]
    for group in ("force_odom","tau","body_B","E","foot_base"):
        s=gs(group)
        ref=np.maximum(prep["y_std"][:,B_STEP_LAYOUT[group]],1e-5)
        rel.append(np.mean(np.clip(s/ref[None],0,5),axis=(1,2)))
    unc=np.mean(np.stack(rel,axis=1),axis=1)
    unc=unc/(1+unc)

    total=np.clip(physical+.15*unc,0,1)
    return {
        **comp,
        "physical_risk":np.clip(physical,0,1),
        "uncertainty_risk":unc,
        "total_risk":total,
        "force_upper":f_tail,
        "E_upper":E_tail,
    }

# Freeze calibration to disk BEFORE B10.
serial={
    k:v for k,v in B9_CALIBRATION.items()
    if k not in {"coverage_table","tail_table"}
}
with open(B_ROOT/"tables"/"B9_validation_calibration.json","w") as f:
    json.dump(serial,f,indent=2)

B9_CALIBRATION["coverage_table"].to_csv(
    B_ROOT/"tables"/"B9_validation_coverage.csv",index=False
)
B9_CALIBRATION["tail_table"].to_csv(
    B_ROOT/"tables"/"B9_validation_tail_calibration.csv",index=False
)

print(
    "\nB9 complete. No test trajectory has been scored. "
    "B10 is the first test evaluation."
)


####################################################################################################
# B10
####################################################################################################


# ======================================================================
# B10 — FIRST SCORING OF PERMANENTLY UNTOUCHED TEST TRAJECTORIES
#
# This is intentionally the first model-evaluation cell that touches the
# B7 test trajectories.
#
# After a successful run a marker is written. If you later tune B1-B9 after
# seeing these results, these trajectories are no longer "untouched" and a
# new test pair must be frozen for a publication-grade claim.
# ======================================================================

import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from IPython.display import display

if "B9_CALIBRATION" not in globals():
    raise RuntimeError("Run B9 first.")

B10_MARKER=B_ROOT/"B10_TEST_WAS_SCORED.json"
ALLOW_B10_RESCORE=False

if B10_MARKER.exists() and not ALLOW_B10_RESCORE:
    previous=json.loads(B10_MARKER.read_text())
    raise RuntimeError(
        "B10 test trajectories have already been scored in a prior run. "
        "They are no longer untouched for further model tuning. "
        "Read B10_TEST_WAS_SCORED.json or explicitly set "
        "ALLOW_B10_RESCORE=True only for reproducibility, not for retuning. "
        f"Previous marker: {previous}"
    )

test_ids=B9_DATASET_TABLE.loc[
    B9_DATASET_TABLE["role"]=="test","run_id"
].tolist()

if len(test_ids)!=len(B7_ENVIRONMENTS):
    raise RuntimeError(
        f"Expected one test trajectory per terrain, got {test_ids}"
    )

# Strong guard: no inspected dev run may be in this set.
bad=set(
    B7_SPLIT_MANIFEST.loc[
        B7_SPLIT_MANIFEST["previously_inspected_dev"],"run_id"
    ]
)&set(test_ids)
if bad:
    raise RuntimeError(f"Inspected development run leaked into B10: {bad}")

nominal_rows=[]
tail_rows=[]
event_rows=[]
coverage_rows=[]
risk_rows=[]
prediction_store={}

for rid in test_ids:
    d=B_ALL_DATA[rid]
    dense=predict_B(B9_DENSE,d,"test")
    noact=predict_B(B9_NOACTION,d,"test")
    prediction_store[rid]={"dense":dense,"noaction":noact}

    truth=_b_reshape_cont(dense["truth"])
    mu=dense["continuous_mean"]
    mu0=noact["continuous_mean"]
    scale=dense["continuous_scale"]

    # --------------------------------------------------------------
    # Nominal dynamics and command-information benefit.
    # --------------------------------------------------------------
    for g in core_groups:
        sl=B_STEP_LAYOUT[g]
        de=_mae(mu[:,:,sl],truth[:,:,sl])
        ne=_mae(mu0[:,:,sl],truth[:,:,sl])
        nominal_rows.append({
            "run_id":rid,
            "terrain":d["terrain"],
            "group":g,
            "dense100hz_mae":de,
            "noaction_mae":ne,
            "improvement_fraction":(ne-de)/max(ne,1e-12),
        })

    # --------------------------------------------------------------
    # Tail force / E + peak ordering.
    # --------------------------------------------------------------
    ftrue=_b_group(truth,"force_odom").reshape(-1,B_H,4,3)
    fpred=_b_group(mu,"force_odom").reshape(-1,B_H,4,3)
    f0=_b_group(mu0,"force_odom").reshape(-1,B_H,4,3)
    fnorm=np.linalg.norm(ftrue,axis=-1)
    fmask=fnorm>B9_PREP["force95"]

    if np.any(fmask):
        derr=np.abs(fpred-ftrue)[fmask]
        nerr=np.abs(f0-ftrue)[fmask]
        tail_rows.append({
            "run_id":rid,"terrain":d["terrain"],
            "metric":"force_above_train_p95",
            "events":int(fmask.sum()),
            "dense_component_mae":float(derr.mean()),
            "dense_component_p95":float(np.quantile(derr,.95)),
            "dense_vector_mae":float(
                np.linalg.norm(fpred-ftrue,axis=-1)[fmask].mean()
            ),
            "noaction_component_mae":float(nerr.mean()),
            "noaction_vector_mae":float(
                np.linalg.norm(f0-ftrue,axis=-1)[fmask].mean()
            ),
        })

    Etrue=_b_group(truth,"E").reshape(-1,B_H,4,3)
    Epred=_b_group(mu,"E").reshape(-1,B_H,4,3)
    E0=_b_group(mu0,"E").reshape(-1,B_H,4,3)
    Emask=np.linalg.norm(Etrue,axis=-1)>B9_PREP["E95"]

    if np.any(Emask):
        derr=np.abs(Epred-Etrue)[Emask]
        nerr=np.abs(E0-Etrue)[Emask]
        tail_rows.append({
            "run_id":rid,"terrain":d["terrain"],
            "metric":"E_above_train_p95",
            "events":int(Emask.sum()),
            "dense_component_mae":float(derr.mean()),
            "dense_component_p95":float(np.quantile(derr,.95)),
            "dense_vector_mae":float(
                np.linalg.norm(Epred-Etrue,axis=-1)[Emask].mean()
            ),
            "noaction_component_mae":float(nerr.mean()),
            "noaction_vector_mae":float(
                np.linalg.norm(E0-Etrue,axis=-1)[Emask].mean()
            ),
        })

    actual_peak=np.max(fnorm,axis=(1,2))
    pred_peak=np.max(np.linalg.norm(fpred,axis=-1),axis=(1,2))
    noact_peak=np.max(np.linalg.norm(f0,axis=-1),axis=(1,2))
    peak_spear=float(pd.Series(pred_peak).corr(
        pd.Series(actual_peak),method="spearman"
    ))
    peak_spear0=float(pd.Series(noact_peak).corr(
        pd.Series(actual_peak),method="spearman"
    ))

    # --------------------------------------------------------------
    # Event + conditional impulse.
    # --------------------------------------------------------------
    td_true=dense["touchdown_truth"].reshape(-1).astype(int)
    td_pred=(
        dense["touchdown_probability"].reshape(-1)
        >=B9_CALIBRATION["touchdown_threshold"]
    ).astype(int)
    pr,rc,f1,_=precision_recall_fscore_support(
        td_true,td_pred,average="binary",zero_division=0
    )

    ev=dense["touchdown_truth"].astype(bool)
    if np.any(ev):
        imp_diff=dense["impulse_mean"]-dense["impulse_truth"]
        imp_comp=float(np.abs(imp_diff)[ev].mean())
        imp_vec=np.linalg.norm(imp_diff,axis=-1)[ev]
        imp_vec_mae=float(imp_vec.mean())
        imp_vec_p95=float(np.quantile(imp_vec,.95))
    else:
        imp_comp=imp_vec_mae=imp_vec_p95=np.nan

    event_rows.append({
        "run_id":rid,"terrain":d["terrain"],
        "touchdown_threshold":B9_CALIBRATION["touchdown_threshold"],
        "precision":float(pr),"recall":float(rc),"f1":float(f1),
        "true_touchdowns":int(td_true.sum()),
        "conditional_impulse_component_mae_Ns":imp_comp,
        "conditional_impulse_vector_mae_Ns":imp_vec_mae,
        "conditional_impulse_vector_p95_Ns":imp_vec_p95,
    })

    # --------------------------------------------------------------
    # Test coverage of VALIDATION-FROZEN calibration.
    # --------------------------------------------------------------
    for g,sl in B_STEP_LAYOUT.items():
        err=np.abs(truth[:,:,sl]-mu[:,:,sl])
        sc=np.maximum(scale[:,:,sl],1e-6)
        factor=float(B9_CALIBRATION["scale_factor"][g])
        coverage_rows.append({
            "run_id":rid,"terrain":d["terrain"],"group":g,
            "test_calibrated_marginal_coverage":float(
                np.mean(err<=factor*sc)
            ),
            "mean_abs_error":float(err.mean()),
            "mean_predicted_scale":float(sc.mean()),
        })

    # --------------------------------------------------------------
    # Risk on untouched trajectory.
    # --------------------------------------------------------------
    prisk=B9_predicted_risk(dense,B9_PREP,B9_CALIBRATION)
    arisk=B9_actual_risk(
        truth,
        dense["contact_truth"],
        dense["touchdown_truth"],
        dense["impulse_truth"],
        B9_PREP,
        B9_CALIBRATION,
    )

    spear=float(pd.Series(prisk["physical_risk"]).corr(
        pd.Series(arisk["physical_risk"]),method="spearman"
    ))
    k=max(1,int(round(.10*len(truth))))
    ptop=set(np.argpartition(prisk["physical_risk"],-k)[-k:])
    atop=set(np.argpartition(arisk["physical_risk"],-k)[-k:])
    overlap=len(ptop&atop)/float(k)

    upper_force_coverage=float(
        np.mean(actual_peak<=prisk["force_upper"])
    )

    risk_rows.append({
        "run_id":rid,
        "terrain":d["terrain"],
        "samples":len(truth),
        "risk_spearman":spear,
        "top10_overlap":overlap,
        "force_peak_spearman":peak_spear,
        "noaction_force_peak_spearman":peak_spear0,
        "force_upper_envelope_coverage":upper_force_coverage,
        "median_predicted_physical_risk":float(
            np.median(prisk["physical_risk"])
        ),
        "median_predicted_total_risk":float(
            np.median(prisk["total_risk"])
        ),
        "median_actual_risk":float(
            np.median(arisk["physical_risk"])
        ),
    })

    # --------------------------------------------------------------
    # Plot: force peak stress region + risk trend.
    # --------------------------------------------------------------
    origins=dense["origins"]
    t=d["time"][origins]
    center=int(np.argmax(actual_peak))
    half=min(300,max(80,len(t)//10))
    lo=max(0,center-half); hi=min(len(t),center+half)
    slc=slice(lo,hi)
    rel=t[slc]-t[lo]

    fig,ax=plt.subplots(figsize=(14,4.5))
    ax.plot(
        rel,actual_peak[slc],
        label="actual max future force norm"
    )
    ax.plot(
        rel,pred_peak[slc],
        label="predicted mean max force norm"
    )
    ax.plot(
        rel,prisk["force_upper"][slc],
        label="calibrated upper force envelope"
    )
    ax.axhline(
        B9_PREP["force95"],linestyle="--",alpha=.5,
        label="train force p95 threshold"
    )
    ax.set_title(f"B10 untouched {rid}: force-tail stress window")
    ax.set_xlabel("relative time [s]"); ax.set_ylabel("N")
    ax.grid(alpha=.2); ax.legend()
    plt.tight_layout()
    plt.savefig(
        B_ROOT/"figures"/f"B10_{rid}_force_tail.png",
        dpi=160,bbox_inches="tight",
    )
    plt.show()

    fig,ax=plt.subplots(figsize=(14,4.3))
    rt=t-t[0]
    ax.plot(rt,arisk["physical_risk"],label="actual future risk",alpha=.8)
    ax.plot(rt,prisk["physical_risk"],label="predicted physical risk",alpha=.9)
    ax.plot(rt,prisk["total_risk"],label="predicted + monotonic uncertainty",alpha=.7)
    ax.set_title(
        f"B10 untouched {rid}: risk "
        f"(Spearman={spear:.3f}, top10={overlap:.3f})"
    )
    ax.set_xlabel("relative time [s]")
    ax.set_ylabel("dimensionless risk")
    ax.grid(alpha=.2); ax.legend()
    plt.tight_layout()
    plt.savefig(
        B_ROOT/"figures"/f"B10_{rid}_risk.png",
        dpi=160,bbox_inches="tight",
    )
    plt.show()

B10_NOMINAL=pd.DataFrame(nominal_rows)
B10_TAIL=pd.DataFrame(tail_rows)
B10_EVENTS=pd.DataFrame(event_rows)
B10_COVERAGE=pd.DataFrame(coverage_rows)
B10_RISK=pd.DataFrame(risk_rows)

B10_INFORMATION_BENEFIT=(
    B10_NOMINAL.groupby("group",as_index=False)
    .agg(
        dense_mae=("dense100hz_mae","mean"),
        noaction_mae=("noaction_mae","mean"),
        mean_improvement_fraction=("improvement_fraction","mean"),
    )
)

print("\nB10 INFORMATION BENEFIT")
display(B10_INFORMATION_BENEFIT)
print("\nB10 TAIL")
display(B10_TAIL)
print("\nB10 EVENTS")
display(B10_EVENTS)
print("\nB10 CALIBRATION COVERAGE")
display(B10_COVERAGE)
print("\nB10 RISK")
display(B10_RISK)

for name,tab in {
    "B10_nominal.csv":B10_NOMINAL,
    "B10_information_benefit.csv":B10_INFORMATION_BENEFIT,
    "B10_tail.csv":B10_TAIL,
    "B10_events.csv":B10_EVENTS,
    "B10_coverage.csv":B10_COVERAGE,
    "B10_risk.csv":B10_RISK,
}.items():
    tab.to_csv(B_ROOT/"tables"/name,index=False)

core={"force_odom","tau","body_B","foot_base"}
mean_cmd=float(
    B10_INFORMATION_BENEFIT[
        B10_INFORMATION_BENEFIT["group"].isin(core)
    ]["mean_improvement_fraction"].mean()
)
mean_risk=float(B10_RISK["risk_spearman"].mean())
mean_top=float(B10_RISK["top10_overlap"].mean())
mean_peak=float(B10_RISK["force_peak_spearman"].mean())
mean_f1=float(B10_EVENTS["f1"].mean())
mean_force_upper_cov=float(
    B10_RISK["force_upper_envelope_coverage"].mean()
)

# Tail force command benefit.
force_tail_rows=B10_TAIL[
    B10_TAIL["metric"]=="force_above_train_p95"
]
if len(force_tail_rows):
    tail_force_improvement=float(np.mean(
        (
            force_tail_rows["noaction_component_mae"]
            -force_tail_rows["dense_component_mae"]
        )
        /np.maximum(
            force_tail_rows["noaction_component_mae"],1e-12
        )
    ))
else:
    tail_force_improvement=np.nan

B10_FINAL_GATE={
    "test_run_ids":test_ids,
    "mean_core_command_benefit_fraction":mean_cmd,
    "mean_tail_force_command_benefit_fraction":tail_force_improvement,
    "mean_risk_spearman":mean_risk,
    "mean_top10_overlap":mean_top,
    "mean_force_peak_spearman":mean_peak,
    "mean_touchdown_f1":mean_f1,
    "mean_force_upper_envelope_coverage":mean_force_upper_cov,
    "command_information_gate":bool(mean_cmd>0),
    "tail_force_command_gate":bool(
        np.isfinite(tail_force_improvement)
        and tail_force_improvement>0
    ),
    "risk_trend_gate":bool(
        mean_risk>=.40 and mean_top>=.20
    ),
    "tail_event_gate":bool(
        mean_peak>=.40 and mean_f1>=.50
    ),
    "stage_B_ready_for_stage_C":bool(
        mean_cmd>0
        and np.isfinite(tail_force_improvement)
        and tail_force_improvement>0
        and mean_risk>=.40
        and mean_top>=.20
        and mean_peak>=.40
        and mean_f1>=.50
    ),
    "interpretation":(
        "Diagnostic readiness gate on two independently held-out trajectories "
        "(one per terrain). Passing is not yet a paper-level terrain-generalization "
        "claim; Stage C may then measure incremental RGB-D value."
    ),
}

print("\n"+"="*88)
print("B10 FINAL UNTOUCHED-TRAJECTORY GATE")
print("="*88)
print(json.dumps(B10_FINAL_GATE,indent=2))

# Only mark test as seen AFTER all scoring/artifact writes succeeded.
marker={
    "scored_utc":datetime.now(timezone.utc).isoformat(),
    "test_run_ids":test_ids,
    "dense_checkpoint":B9_DENSE["checkpoint"],
    "noaction_checkpoint":B9_NOACTION["checkpoint"],
    "final_gate":B10_FINAL_GATE,
    "warning":(
        "If B1-B9 are retuned after reading these results, freeze new test "
        "trajectories before making another untouched-test claim."
    ),
}
B10_MARKER.write_text(json.dumps(marker,indent=2))

print("\nB10 marker written:",B10_MARKER)
print("Stage B is complete. Do not add RGB-D/VLM/planning until these results are interpreted.")
