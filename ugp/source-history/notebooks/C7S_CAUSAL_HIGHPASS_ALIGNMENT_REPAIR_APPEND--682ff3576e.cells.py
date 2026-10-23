# Source inspection export of the adjacent notebook.
# Notebook execution order and dependencies still apply.
# IPython magic, shell lines and prose-only conclusion cells are comments.

# %% notebook cell 1

# ======================================================================
# C7S — VALIDATION-ONLY CAUSAL HIGH-PASS + ALIGNMENT REPAIR
#
# APPEND/RUN THIS AFTER UGP_C7R_T4_RESCUE_100MIN_FINAL has completed.
#
# Scientific purpose:
#   Previous C7R still let run/scene context compete with the true local
#   visual signal: true ~= shuffled ~= run_mean.
#
# This repair changes ONLY the visual branch while keeping frozen:
#   - B9 dynamics model/checkpoint
#   - B9 calibration/risk definition
#   - C train/val split
#   - physical targets
#   - optimizer micro-batch = 96
#   - validation-only model selection
#   - anti-cheating gate thresholds
#
# New visual representation is causal and "absolute-scene-free":
#   [ current - causal trailing mean(8 frames),
#     current - previous frame,
#     current - frame 4 steps ago ]
#
# This removes most trajectory/scene identity information and asks whether
# time-local visual CHANGE predicts future physical response.
#
# A second train-only regularizer presents a within-run mismatched visual
# observation and asks its residual correction to collapse toward frozen B9.
# This is NOT applied to validation labels and never touches fresh C tests.
#
# Fresh C tests remain locked throughout.
# ======================================================================

import gc, time, json, hashlib, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import precision_recall_fscore_support
from IPython.display import display

# ----------------------------------------------------------------------
# 0. HARD GUARDS — refuse to run in the wrong state
# ----------------------------------------------------------------------
_REQUIRED = [
    "C_ROOT", "C7R_FRAME_FEATURES", "C7R_DATA", "TEST_ROWS",
    "B9_CKPT", "B9_PREP", "B9_CALIBRATION", "B9_DENSE",
    "BResponseModel", "B_DEVICE", "B_H", "B_STEP_WIDTH",
    "B_STEP_LAYOUT", "B_MODEL_DIM", "B_ACTION_HIDDEN",
    "B_ACTION_GATHER", "TAIL_NAMES", "CFG",
    "_b_tail_raw", "_b_reshape_cont", "_b_group",
    "b_loss", "fresh_base", "predict_B_fast",
    "fit_cal", "metrics"
]
_missing = [x for x in _REQUIRED if x not in globals()]
if _missing:
    raise RuntimeError(
        "Run the completed T4 rescue notebook first. Missing globals: "
        + repr(_missing)
    )

C8_MARKER_V2 = C_ROOT / "C8_TEST_WAS_SCORED.json"
if C8_MARKER_V2.exists():
    raise RuntimeError(
        f"Fresh Stage-C test was already scored: {C8_MARKER_V2}. "
        "Do not use it for further visual tuning."
    )

_fresh_ids = TEST_ROWS["run_id"].astype(str).tolist()
if set(_fresh_ids) & set(C7R_DATA.keys()):
    raise RuntimeError(
        "Fresh C test trajectories unexpectedly exist in C7R_DATA. "
        "Refusing to continue."
    )

print("Fresh C tests remain LOCKED:", _fresh_ids)
print("C7S uses development runs only:", sorted(C7R_DATA.keys()))

# ----------------------------------------------------------------------
# 1. NEW NAMESPACE / CONSTANTS
# ----------------------------------------------------------------------
SEED2 = int(globals().get("SEED", 1781)) + 202
random.seed(SEED2)
np.random.seed(SEED2)
torch.manual_seed(SEED2)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED2)

C7S_ROOT = C_ROOT / "spatial_temporal_repair_v2_highpass_alignment"
C7S_MODEL_ROOT = C7S_ROOT / "models"
C7S_TABLE_ROOT = C7S_ROOT / "tables"
for _p in (C7S_ROOT, C7S_MODEL_ROOT, C7S_TABLE_ROOT):
    _p.mkdir(parents=True, exist_ok=True)

C7S_FEATURE_VERSION = (
    "C7S_causal_highpass8_d1_d4__"
    "dino_lower60_2x3_depth_lower60_8x8x4__"
    "withinrun_negative_null_v1"
)

# Keep approved optimization schedule.
C7S_TRAIN_BATCH = 96
C7S_FETCH_BATCH = 384
C7S_EPOCHS = 20
C7S_PATIENCE = 5
C7S_LR = 2e-4
C7S_WEIGHT_DECAY = 1e-4
C7S_RESIDUAL_L2 = 2e-3

# New repair-specific settings.
C7S_CAUSAL_WINDOW_FRAMES = 8
C7S_LONG_DELTA_FRAMES = 4
C7S_NEG_ORIGIN_OFFSET = 257      # ~2.57 s at the 100-Hz origin grid
C7S_NEG_NULL_WEIGHT = 0.05
C7S_PRECONTACT_EXTRA_WEIGHT = 3.0

C7S_HIDDEN = 128
C7S_USE_AMP = True

print({
    "feature_version": C7S_FEATURE_VERSION,
    "optimizer_batch": C7S_TRAIN_BATCH,
    "fetch_batch": C7S_FETCH_BATCH,
    "causal_window_frames": C7S_CAUSAL_WINDOW_FRAMES,
    "long_delta_frames": C7S_LONG_DELTA_FRAMES,
    "negative_offset_origins": C7S_NEG_ORIGIN_OFFSET,
    "fresh_tests_loaded": False,
})

# ----------------------------------------------------------------------
# 2. PRECOMPUTE CAUSAL HIGH-PASS FRAME TABLES
#
# No future image is used.
# trailing_mean[i] = mean(F[max(0,i-W):i]); current frame is excluded.
# ----------------------------------------------------------------------
def _causal_change_table(F, window=8, long_delta=4):
    F = np.asarray(F, dtype=np.float32)
    if F.ndim != 2 or len(F) < 2:
        raise ValueError(f"Unexpected frame feature shape: {F.shape}")

    n, d = F.shape
    # float64 cumulative sum only during this one-time preparation.
    cs = np.zeros((n + 1, d), dtype=np.float64)
    np.cumsum(F, axis=0, dtype=np.float64, out=cs[1:])

    i = np.arange(n, dtype=np.int64)
    lo = np.maximum(0, i - int(window))
    cnt = i - lo

    prev_mean = np.asarray(F, dtype=np.float32).copy()
    good = cnt > 0
    if np.any(good):
        prev_sum = cs[i[good]] - cs[lo[good]]
        prev_mean[good] = (
            prev_sum / cnt[good, None]
        ).astype(np.float32)

    p1 = np.maximum(i - 1, 0)
    pk = np.maximum(i - int(long_delta), 0)

    hp = F - prev_mean
    d1 = F - F[p1]
    dk = F - F[pk]

    # Frame 0 has no causal past.
    hp[0] = 0.0
    d1[0] = 0.0
    dk[0] = 0.0

    return np.concatenate([hp, d1, dk], axis=1).astype(np.float32)


C7S_FRAME = {}
for rid, tab in C7R_FRAME_FEATURES.items():
    C7S_FRAME[rid] = {
        "rgb": _causal_change_table(
            tab["rgb_spatial"],
            C7S_CAUSAL_WINDOW_FRAMES,
            C7S_LONG_DELTA_FRAMES,
        ),
        "depth": _causal_change_table(
            tab["depth_spatial"],
            C7S_CAUSAL_WINDOW_FRAMES,
            C7S_LONG_DELTA_FRAMES,
        ),
    }

print(
    "C7S causal frame tables:",
    {
        rid: (v["rgb"].shape, v["depth"].shape)
        for rid, v in C7S_FRAME.items()
    },
)

def c7s_origin_visual(rid, d, positions):
    positions = np.asarray(positions, dtype=np.int64)
    vi = np.asarray(d["visual_frame_index"][positions], dtype=np.int64)

    tab = C7S_FRAME[rid]
    return (
        np.asarray(tab["rgb"][vi], dtype=np.float32),
        np.asarray(tab["depth"][vi], dtype=np.float32),
    )

# ----------------------------------------------------------------------
# 3. TRAIN-ONLY NORMALIZATION
# ----------------------------------------------------------------------
def fit_c7s_vis_norm():
    sr = sr2 = sd = sd2 = None
    n = 0

    for rid, d in C7R_DATA.items():
        pos = np.flatnonzero(d["split"] == "train")
        for st in range(0, len(pos), 512):
            R, D = c7s_origin_visual(rid, d, pos[st:st+512])

            if sr is None:
                sr = np.zeros(R.shape[1], dtype=np.float64)
                sr2 = np.zeros_like(sr)
                sd = np.zeros(D.shape[1], dtype=np.float64)
                sd2 = np.zeros_like(sd)

            sr += R.sum(0, dtype=np.float64)
            sr2 += np.square(R, dtype=np.float64).sum(0, dtype=np.float64)
            sd += D.sum(0, dtype=np.float64)
            sd2 += np.square(D, dtype=np.float64).sum(0, dtype=np.float64)
            n += len(R)

    rm = sr / n
    dm = sd / n
    rs = np.sqrt(np.maximum(sr2 / n - rm * rm, 0.0))
    ds = np.sqrt(np.maximum(sd2 / n - dm * dm, 0.0))
    rs[rs < 1e-5] = 1.0
    ds[ds < 1e-5] = 1.0

    return {
        "rgb_mean": rm.astype(np.float32),
        "rgb_std": rs.astype(np.float32),
        "depth_mean": dm.astype(np.float32),
        "depth_std": ds.astype(np.float32),
        "train_count": int(n),
    }

C7S_NORM_PATH = C7S_ROOT / "visual_norm.npz"
if C7S_NORM_PATH.exists():
    with np.load(C7S_NORM_PATH, allow_pickle=False) as z:
        C7S_VIS_NORM = {k: z[k] for k in z.files}
    C7S_VIS_NORM["train_count"] = int(
        np.asarray(C7S_VIS_NORM["train_count"]).reshape(())
    )
    print("Reused:", C7S_NORM_PATH)
else:
    C7S_VIS_NORM = fit_c7s_vis_norm()
    np.savez_compressed(C7S_NORM_PATH, **C7S_VIS_NORM)
    print("Wrote:", C7S_NORM_PATH)

print({
    "train_count": C7S_VIS_NORM["train_count"],
    "rgb_dim": len(C7S_VIS_NORM["rgb_mean"]),
    "depth_dim": len(C7S_VIS_NORM["depth_mean"]),
})

C7S_NORM_T = {
    k: torch.as_tensor(v, dtype=torch.float32, device=B_DEVICE)
    for k, v in {
        "x_mean": B9_PREP["x_mean"],
        "x_std": B9_PREP["x_std"],
        "action_mean": B9_PREP["action_mean"],
        "action_std": B9_PREP["action_std"],
        "y_mean": B9_PREP["y_mean"],
        "y_std": B9_PREP["y_std"],
        "imp_mean": B9_PREP["imp_mean"],
        "imp_std": B9_PREP["imp_std"],
        "rgb_mean": C7S_VIS_NORM["rgb_mean"],
        "rgb_std": C7S_VIS_NORM["rgb_std"],
        "depth_mean": C7S_VIS_NORM["depth_mean"],
        "depth_std": C7S_VIS_NORM["depth_std"],
    }.items()
}

# ----------------------------------------------------------------------
# 4. DATASET / SAMPLER
#
# Same terrain/tail/event balancing as C7R, plus extra TRAIN-ONLY weight
# for the pre-contact subset because this stage explicitly tests visual
# anticipation before physical contact.
# ----------------------------------------------------------------------
class C7SIndexDataset(Dataset):
    def __init__(self, data_dict, split, balanced=False):
        self.data_dict = data_dict
        self.run_keys = []

        ro, lo, ftp, etp, tlp, swp = [], [], [], [], [], []
        rec = []
        terrain_counts = {}

        for rid, d in data_dict.items():
            idx = np.flatnonzero(d["split"] == split).astype(np.int64)
            if len(idx):
                terrain_counts[d["terrain"]] = (
                    terrain_counts.get(d["terrain"], 0) + len(idx)
                )
                rec.append((rid, d, idx, d["terrain"]))

        total = sum(len(x[2]) for x in rec)
        terrain_factor = {
            t: total / (len(terrain_counts) * count)
            for t, count in terrain_counts.items()
        }

        for rix, (rid, d, idx, terrain) in enumerate(rec):
            self.run_keys.append(rid)
            ro.append(np.full(len(idx), rix, dtype=np.int16))
            lo.append(idx)

            step = np.asarray(
                d["Y_cont"][idx], dtype=np.float32
            ).reshape(-1, B_H, B_STEP_WIDTH)

            td = np.asarray(d["Y_touchdown"][idx], dtype=np.float32)

            force = step[:, :, B_STEP_LAYOUT["force_odom"]].reshape(
                -1, B_H, 4, 3
            )
            E = step[:, :, B_STEP_LAYOUT["E"]].reshape(
                -1, B_H, 4, 3
            )

            ft = (
                np.linalg.norm(force, axis=-1) > B9_PREP["force95"]
            )
            et = (
                np.linalg.norm(E, axis=-1) > B9_PREP["E95"]
            )

            tail = np.log1p(
                np.maximum(
                    _b_tail_raw(
                        step,
                        B9_PREP["body_med"],
                        B9_PREP["body_mad"],
                    ),
                    0.0,
                )
            ).astype(np.float32)

            sw = (
                1.0
                + 2.0 * ft.max((1, 2)).astype(np.float32)
                + 2.0 * et.max((1, 2)).astype(np.float32)
                + 4.0 * td.max((1, 2)).astype(np.float32)
            )

            if split == "train":
                pre = np.asarray(
                    d["C_precontact"][idx], dtype=np.float32
                )
                sw += C7S_PRECONTACT_EXTRA_WEIGHT * pre

            if balanced:
                sw *= float(terrain_factor[terrain])

            ftp.append(ft.astype(np.uint8))
            etp.append(et.astype(np.uint8))
            tlp.append(tail)
            swp.append(np.clip(sw, 1.0, 16.0).astype(np.float32))

        self.run_of = np.concatenate(ro)
        self.local_of = np.concatenate(lo)
        self.force_tail = np.concatenate(ftp)
        self.E_tail = np.concatenate(etp)
        self.tail_log = np.concatenate(tlp)
        self.sample_weight = np.concatenate(swp)

    def __len__(self):
        return len(self.local_of)

    def __getitem__(self, i):
        return int(i)


def c7s_make_collate(ds):
    rgb_dim = len(C7S_VIS_NORM["rgb_mean"])
    dep_dim = len(C7S_VIS_NORM["depth_mean"])

    def collate(items):
        gi = np.asarray(items, dtype=np.int64)
        Bn = len(gi)

        X = np.empty((Bn, 1060), dtype=np.float32)
        A = np.empty((Bn, 100, 18), dtype=np.float32)
        Y = np.empty((Bn, B_H, B_STEP_WIDTH), dtype=np.float32)
        C = np.empty((Bn, B_H, 4), dtype=np.float32)
        TD = np.empty_like(C)
        IMP = np.empty((Bn, B_H, 4, 3), dtype=np.float32)

        R = np.empty((Bn, rgb_dim), dtype=np.float32)
        D = np.empty((Bn, dep_dim), dtype=np.float32)
        RN = np.empty_like(R)
        DN = np.empty_like(D)

        runix = ds.run_of[gi]
        local = ds.local_of[gi]

        for rix in np.unique(runix):
            outpos = np.flatnonzero(runix == rix)
            pos = local[outpos]
            rid = ds.run_keys[int(rix)]
            d = ds.data_dict[rid]

            X[outpos] = d["X_core"][pos]
            A[outpos] = d["X_action_dense"][pos]
            Y[outpos] = d["Y_cont"][pos].reshape(
                -1, B_H, B_STEP_WIDTH
            )
            C[outpos] = d["Y_contact"][pos]
            TD[outpos] = d["Y_touchdown"][pos]
            IMP[outpos] = d["Y_impulse"][pos]

            rr, dd = c7s_origin_visual(rid, d, pos)
            R[outpos] = rr
            D[outpos] = dd

            # Deterministic within-run mismatched visual pairing.
            # Same environment/run, wrong time.
            neg_pos = (
                pos + C7S_NEG_ORIGIN_OFFSET
            ) % len(d["origins"])
            rrn, ddn = c7s_origin_visual(rid, d, neg_pos)
            RN[outpos] = rrn
            DN[outpos] = ddn

        vals = [
            X, A, Y, C, TD, IMP,
            ds.force_tail[gi].astype(np.float32),
            ds.E_tail[gi].astype(np.float32),
            ds.tail_log[gi],
            R, D, RN, DN,
        ]
        return tuple(torch.from_numpy(v) for v in vals)

    return collate


def c7s_prep_batch(raw):
    raw = tuple(
        t.to(B_DEVICE, non_blocking=True)
        for t in raw
    )
    (
        x, a, y, c, td, imp,
        ft, et, tail,
        rgb, dep, rgb_neg, dep_neg,
    ) = raw

    x = (
        (x - C7S_NORM_T["x_mean"][None, :])
        / C7S_NORM_T["x_std"][None, :]
    )
    a = (
        (a - C7S_NORM_T["action_mean"][None, None, :])
        / C7S_NORM_T["action_std"][None, None, :]
    )
    y = (
        (y - C7S_NORM_T["y_mean"][None, :, :])
        / C7S_NORM_T["y_std"][None, :, :]
    )
    imp = (
        (imp - C7S_NORM_T["imp_mean"][None, None, None, :])
        / C7S_NORM_T["imp_std"][None, None, None, :]
    )

    rgb = (
        (rgb - C7S_NORM_T["rgb_mean"][None, :])
        / C7S_NORM_T["rgb_std"][None, :]
    )
    dep = (
        (dep - C7S_NORM_T["depth_mean"][None, :])
        / C7S_NORM_T["depth_std"][None, :]
    )

    rgb_neg = (
        (rgb_neg - C7S_NORM_T["rgb_mean"][None, :])
        / C7S_NORM_T["rgb_std"][None, :]
    )
    dep_neg = (
        (dep_neg - C7S_NORM_T["depth_mean"][None, :])
        / C7S_NORM_T["depth_std"][None, :]
    )

    return (
        (x, a, y, c, td, imp, ft, et, tail),
        rgb, dep,
        rgb_neg, dep_neg,
    )

# ----------------------------------------------------------------------
# 5. VISUAL RESIDUAL MODEL
#
# Same frozen B9 architecture. We only expose a fuller residual magnitude
# so mismatched visuals can be explicitly pushed back toward B9.
# ----------------------------------------------------------------------
class C7SVisualResidualModel(nn.Module):
    def __init__(self, base, rgb_dim, depth_dim, mode):
        super().__init__()
        self.mode = mode
        self.base = base.eval()

        for p in self.base.parameters():
            p.requires_grad_(False)

        self.rgb = nn.Sequential(
            nn.Linear(rgb_dim, C7S_HIDDEN),
            nn.GELU(),
            nn.LayerNorm(C7S_HIDDEN),
            nn.Linear(C7S_HIDDEN, C7S_HIDDEN),
            nn.GELU(),
            nn.LayerNorm(C7S_HIDDEN),
        )
        self.depth = nn.Sequential(
            nn.Linear(depth_dim, C7S_HIDDEN),
            nn.GELU(),
            nn.LayerNorm(C7S_HIDDEN),
            nn.Linear(C7S_HIDDEN, C7S_HIDDEN),
            nn.GELU(),
            nn.LayerNorm(C7S_HIDDEN),
        )

        vd = (
            C7S_HIDDEN
            if mode in {"rgb", "depth"}
            else 2 * C7S_HIDDEN
        )

        self.h_adapter = nn.Sequential(
            nn.Linear(B_MODEL_DIM + vd, B_MODEL_DIM),
            nn.GELU(),
            nn.LayerNorm(B_MODEL_DIM),
            nn.Dropout(0.05),
            nn.Linear(B_MODEL_DIM, B_MODEL_DIM),
            nn.GELU(),
            nn.LayerNorm(B_MODEL_DIM),
        )

        self.seq_adapter = nn.Sequential(
            nn.Linear(
                B_MODEL_DIM + B_ACTION_HIDDEN + vd,
                B_MODEL_DIM,
            ),
            nn.GELU(),
            nn.LayerNorm(B_MODEL_DIM),
        )

        self.d_mean = nn.Linear(B_MODEL_DIM, B_STEP_WIDTH)
        self.d_log_scale = nn.Linear(B_MODEL_DIM, B_STEP_WIDTH)
        self.d_contact = nn.Linear(B_MODEL_DIM, 4)
        self.d_touchdown = nn.Linear(B_MODEL_DIM, 4)
        self.d_impulse_mean = nn.Linear(B_MODEL_DIM, 12)
        self.d_impulse_log_scale = nn.Linear(B_MODEL_DIM, 12)
        self.d_tail = nn.Linear(B_MODEL_DIM, 4)

        for layer in [
            self.d_mean,
            self.d_log_scale,
            self.d_contact,
            self.d_touchdown,
            self.d_impulse_mean,
            self.d_impulse_log_scale,
            self.d_tail,
        ]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _v(self, r, d):
        if self.mode == "rgb":
            return self.rgb(r)
        if self.mode == "depth":
            return self.depth(d)
        return torch.cat([self.rgb(r), self.depth(d)], dim=-1)

    def forward(self, x, a, r, d):
        self.base.eval()

        with torch.no_grad():
            h0 = self.base.history(x)
            ah = self.base.action(
                a, B_ACTION_GATHER.to(a.device)
            )
            afinal = ah[:, -1]

            h = h0[:, None, :].expand(-1, B_H, -1)
            he = self.base.horizon_embed[None].expand(
                len(x), -1, -1
            )

            z = self.base.fuse(
                torch.cat([h, ah, he], dim=-1)
            )

            base = {
                "mean_z": self.base.mean_head(z),
                "log_scale_z": torch.clamp(
                    self.base.log_scale_head(z),
                    -5.5, 2.8,
                ),
                "contact_logits": self.base.contact_head(z),
                "touchdown_logits":
                    self.base.touchdown_head(z),
                "impulse_mean_z":
                    self.base.impulse_mean_head(z).reshape(
                        -1, B_H, 4, 3
                    ),
                "impulse_log_scale_z":
                    torch.clamp(
                        self.base.impulse_log_scale_head(z).reshape(
                            -1, B_H, 4, 3
                        ),
                        -5.5, 3.0,
                    ),
                "tail_q90_log":
                    self.base.tail_q90_head(
                        torch.cat([h0, afinal], dim=-1)
                    ),
            }

        v = self._v(r, d)

        rz = self.h_adapter(
            torch.cat(
                [
                    z,
                    v[:, None, :].expand(-1, B_H, -1),
                ],
                dim=-1,
            )
        )

        seq = self.seq_adapter(
            torch.cat([h0, afinal, v], dim=-1)
        )

        dm = self.d_mean(rz)
        dls = self.d_log_scale(rz)
        dc = self.d_contact(rz)
        dtd = self.d_touchdown(rz)
        dim = self.d_impulse_mean(rz).reshape(
            -1, B_H, 4, 3
        )
        dils = self.d_impulse_log_scale(rz).reshape(
            -1, B_H, 4, 3
        )
        dtail = self.d_tail(seq)

        out = {
            "mean_z": base["mean_z"] + dm,
            "log_scale_z": torch.clamp(
                base["log_scale_z"] + dls,
                -5.5, 2.8,
            ),
            "contact_logits":
                base["contact_logits"] + dc,
            "touchdown_logits":
                base["touchdown_logits"] + dtd,
            "impulse_mean_z":
                base["impulse_mean_z"] + dim,
            "impulse_log_scale_z":
                torch.clamp(
                    base["impulse_log_scale_z"] + dils,
                    -5.5, 3.0,
                ),
            "tail_q90_log":
                base["tail_q90_log"] + dtail,
        }

        out["_residual_full_l2"] = (
            dm.square().mean()
            + 0.10 * dls.square().mean()
            + 0.05 * dc.square().mean()
            + 0.10 * dtd.square().mean()
            + 0.25 * dim.square().mean()
            + 0.05 * dils.square().mean()
            + 0.25 * dtail.square().mean()
        )

        return out

# ----------------------------------------------------------------------
# 6. TRAIN / RESUME — TRAIN RUNS ONLY
# ----------------------------------------------------------------------
def c7s_fresh_base():
    c = torch.load(
        B9_CKPT,
        map_location="cpu",
        weights_only=False,
    )
    m = BResponseModel(
        int(c["history_dim"]),
        int(c["step_width"]),
        bool(c["use_action"]),
    )
    m.load_state_dict(c["state_dict"])
    return m.eval()


def c7s_train_mode(mode):
    final = C7S_MODEL_ROOT / f"C7S_{mode}_adapter.pt"
    resume = C7S_MODEL_ROOT / f"C7S_{mode}_resume.pt"

    if final.exists():
        s = torch.load(
            final,
            map_location="cpu",
            weights_only=False,
        )
        if s.get("feature_version") == C7S_FEATURE_VERSION:
            print(
                f"C7S-{mode}: reusing completed checkpoint "
                f"{final}"
            )
            return {
                "mode": mode,
                "checkpoint": str(final),
                "best_val_loss": float(s["best_val_loss"]),
            }

    tr = C7SIndexDataset(
        C7R_DATA, "train", balanced=True
    )
    va = C7SIndexDataset(
        C7R_DATA, "val", balanced=False
    )

    sampler = WeightedRandomSampler(
        torch.as_tensor(
            tr.sample_weight,
            dtype=torch.double,
        ),
        len(tr),
        replacement=True,
    )

    trl = DataLoader(
        tr,
        batch_size=C7S_FETCH_BATCH,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=c7s_make_collate(tr),
    )
    val = DataLoader(
        va,
        batch_size=C7S_FETCH_BATCH,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=c7s_make_collate(va),
    )

    model = C7SVisualResidualModel(
        c7s_fresh_base(),
        len(C7S_VIS_NORM["rgb_mean"]),
        len(C7S_VIS_NORM["depth_mean"]),
        mode,
    ).to(B_DEVICE)

    params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    opt = torch.optim.AdamW(
        params,
        lr=C7S_LR,
        weight_decay=C7S_WEIGHT_DECAY,
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=C7S_USE_AMP
    )

    best = float("inf")
    best_state = None
    bad = 0
    hist = []
    start_epoch = 1

    if resume.exists():
        rr = torch.load(
            resume,
            map_location="cpu",
            weights_only=False,
        )

        if (
            rr.get("feature_version") == C7S_FEATURE_VERSION
            and rr.get("mode") == mode
        ):
            model.load_state_dict(
                rr["adapter_state_dict"],
                strict=False,
            )
            opt.load_state_dict(rr["optimizer_state"])
            best = float(rr["best"])
            best_state = rr["best_state"]
            bad = int(rr["bad"])
            hist = list(rr["history"])
            start_epoch = int(rr["epoch"]) + 1
            print(
                f"C7S-{mode}: resuming at epoch "
                f"{start_epoch}"
            )

    @torch.no_grad()
    def eval_loss():
        model.eval()
        total = 0.0
        n = 0

        for raw in val:
            (
                bb, r, d, _rn, _dn
            ) = c7s_prep_batch(raw)

            with torch.autocast(
                "cuda",
                dtype=torch.float16,
                enabled=C7S_USE_AMP,
            ):
                out = model(bb[0], bb[1], r, d)
                loss = (
                    b_loss(out, bb)
                    + C7S_RESIDUAL_L2
                    * out["_residual_full_l2"]
                )

            total += float(loss) * len(bb[0])
            n += len(bb[0])

        return total / max(n, 1)

    for ep in range(start_epoch, C7S_EPOCHS + 1):
        tic = time.time()
        model.train()

        total = 0.0
        n = 0

        for raw_big in trl:
            for off in range(
                0,
                len(raw_big[0]),
                C7S_TRAIN_BATCH,
            ):
                raw = tuple(
                    t[off:off+C7S_TRAIN_BATCH]
                    for t in raw_big
                )

                (
                    bb,
                    r,
                    d,
                    rn,
                    dn,
                ) = c7s_prep_batch(raw)

                opt.zero_grad(set_to_none=True)

                with torch.autocast(
                    "cuda",
                    dtype=torch.float16,
                    enabled=C7S_USE_AMP,
                ):
                    out = model(
                        bb[0], bb[1], r, d
                    )
                    out_neg = model(
                        bb[0], bb[1], rn, dn
                    )

                    loss_true = (
                        b_loss(out, bb)
                        + C7S_RESIDUAL_L2
                        * out["_residual_full_l2"]
                    )

                    # Wrong-time same-run vision should not be
                    # allowed to move the frozen B9 prediction.
                    loss_null = (
                        C7S_NEG_NULL_WEIGHT
                        * out_neg["_residual_full_l2"]
                    )

                    loss = loss_true + loss_null

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    params, 5.0
                )
                scaler.step(opt)
                scaler.update()

                total += float(
                    loss.detach()
                ) * len(bb[0])
                n += len(bb[0])

        train_loss = total / max(n, 1)
        val_loss = eval_loss()

        hist.append({
            "mode": mode,
            "epoch": ep,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "seconds": time.time() - tic,
        })

        print(
            f"C7S-{mode:5s} ep={ep:02d} "
            f"train={train_loss:.5f} "
            f"val={val_loss:.5f} "
            f"time={time.time()-tic:.1f}s"
        )

        if val_loss < best - 1e-4:
            best = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
                if not k.startswith("base.")
            }
            bad = 0
        else:
            bad += 1

        current = {
            k: v.detach().cpu()
            for k, v in model.state_dict().items()
            if not k.startswith("base.")
        }

        torch.save({
            "mode": mode,
            "feature_version": C7S_FEATURE_VERSION,
            "epoch": ep,
            "adapter_state_dict": current,
            "optimizer_state": opt.state_dict(),
            "best": best,
            "best_state": best_state,
            "bad": bad,
            "history": hist,
        }, resume)

        if bad >= C7S_PATIENCE:
            print(f"C7S-{mode}: early stop")
            break

    if best_state is None:
        raise RuntimeError(
            f"C7S-{mode}: no best state found"
        )

    torch.save({
        "mode": mode,
        "adapter_state_dict": best_state,
        "best_val_loss": float(best),
        "vis_norm": C7S_VIS_NORM,
        "feature_version": C7S_FEATURE_VERSION,
        "B9_checkpoint": str(B9_CKPT),
        "uses_test": False,
        "training_history": hist,
        "negative_null_weight":
            C7S_NEG_NULL_WEIGHT,
        "precontact_extra_weight":
            C7S_PRECONTACT_EXTRA_WEIGHT,
    }, final)

    pd.DataFrame(hist).to_csv(
        C7S_TABLE_ROOT
        / f"C7S_{mode}_history.csv",
        index=False,
    )

    del (
        model, tr, va,
        trl, val, sampler,
        opt, params,
    )
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "mode": mode,
        "checkpoint": str(final),
        "best_val_loss": float(best),
    }


B9_DENSE["model"].to("cpu")
C7S_MODELS = {}

print("\nTraining C7S causal high-pass adapters...")

for mode in ("rgb", "depth", "rgbd"):
    print(
        "\n" + "=" * 80
        + f"\nC7S MODE: {mode}"
        + "\n" + "=" * 80
    )
    C7S_MODELS[mode] = c7s_train_mode(mode)

print("\nC7S TRAINING COMPLETE / RESTORED")
display(pd.DataFrame([
    {
        "mode": k,
        "best_val_loss": v["best_val_loss"],
        "checkpoint": v["checkpoint"],
    }
    for k, v in C7S_MODELS.items()
]))

# ----------------------------------------------------------------------
# 7. VALIDATION-ONLY PREDICTION / CALIBRATION
# ----------------------------------------------------------------------
C7S_ACTIVE = {
    "checkpoint": None,
    "model": None,
}

def c7s_get_model(bundle):
    if (
        C7S_ACTIVE["checkpoint"]
        == bundle["checkpoint"]
        and C7S_ACTIVE["model"] is not None
    ):
        return C7S_ACTIVE["model"]

    if C7S_ACTIVE["model"] is not None:
        C7S_ACTIVE["model"].to("cpu")

    s = torch.load(
        bundle["checkpoint"],
        map_location="cpu",
        weights_only=False,
    )

    m = C7SVisualResidualModel(
        c7s_fresh_base(),
        len(C7S_VIS_NORM["rgb_mean"]),
        len(C7S_VIS_NORM["depth_mean"]),
        s["mode"],
    )

    miss, unexpected = m.load_state_dict(
        s["adapter_state_dict"],
        strict=False,
    )

    bad_missing = [
        x for x in miss
        if not x.startswith("base.")
    ]
    if bad_missing or unexpected:
        raise RuntimeError(
            (bad_missing, unexpected)
        )

    m = m.to(B_DEVICE).eval()

    C7S_ACTIVE["checkpoint"] = bundle["checkpoint"]
    C7S_ACTIVE["model"] = m
    return m


@torch.no_grad()
def predict_C7S(
    bundle,
    rid,
    d,
    split="val",
    control="true",
    seed=0,
    batch=256,
):
    idx = np.flatnonzero(
        d["split"] == split
    ).astype(np.int64)

    if not len(idx):
        raise RuntimeError(
            f"{rid}: no samples for split={split}"
        )

    rng = np.random.default_rng(seed)
    perm = (
        rng.permutation(len(idx))
        if control == "shuffled"
        else None
    )

    if control == "run_mean":
        sr = sd = None
        n = 0
        for st in range(0, len(idx), 512):
            R, D = c7s_origin_visual(
                rid,
                d,
                idx[st:st+512],
            )
            sr = (
                R.sum(0, dtype=np.float64)
                if sr is None
                else sr + R.sum(0, dtype=np.float64)
            )
            sd = (
                D.sum(0, dtype=np.float64)
                if sd is None
                else sd + D.sum(0, dtype=np.float64)
            )
            n += len(R)

        mr = (sr / n).astype(np.float32)
        md = (sd / n).astype(np.float32)

    model = c7s_get_model(bundle)

    S = {
        k: []
        for k in [
            "mz", "sz", "c", "td",
            "im", "isc", "tail",
        ]
    }

    for st in range(0, len(idx), batch):
        ed = min(st + batch, len(idx))
        p = idx[st:ed]

        if control == "true":
            R, D = c7s_origin_visual(rid, d, p)

        elif control == "shuffled":
            R, D = c7s_origin_visual(
                rid,
                d,
                idx[perm[st:ed]],
            )

        elif control == "run_mean":
            R = np.repeat(
                mr[None, :],
                len(p),
                axis=0,
            )
            D = np.repeat(
                md[None, :],
                len(p),
                axis=0,
            )

        else:
            raise ValueError(control)

        x = torch.from_numpy(
            np.asarray(
                d["X_core"][p],
                dtype=np.float32,
            )
        ).to(B_DEVICE)

        a = torch.from_numpy(
            np.asarray(
                d["X_action_dense"][p],
                dtype=np.float32,
            )
        ).to(B_DEVICE)

        r = torch.from_numpy(R).to(B_DEVICE)
        dp = torch.from_numpy(D).to(B_DEVICE)

        x = (
            (x - C7S_NORM_T["x_mean"][None, :])
            / C7S_NORM_T["x_std"][None, :]
        )
        a = (
            (
                a
                - C7S_NORM_T[
                    "action_mean"
                ][None, None, :]
            )
            / C7S_NORM_T[
                "action_std"
            ][None, None, :]
        )
        r = (
            (r - C7S_NORM_T["rgb_mean"][None, :])
            / C7S_NORM_T["rgb_std"][None, :]
        )
        dp = (
            (
                dp
                - C7S_NORM_T[
                    "depth_mean"
                ][None, :]
            )
            / C7S_NORM_T[
                "depth_std"
            ][None, :]
        )

        with torch.autocast(
            "cuda",
            dtype=torch.float16,
            enabled=C7S_USE_AMP,
        ):
            o = model(x, a, r, dp)

        S["mz"].append(
            o["mean_z"].float().cpu().numpy()
        )
        S["sz"].append(
            torch.exp(
                o["log_scale_z"].float()
            ).cpu().numpy()
        )
        S["c"].append(
            torch.sigmoid(
                o["contact_logits"].float()
            ).cpu().numpy()
        )
        S["td"].append(
            torch.sigmoid(
                o["touchdown_logits"].float()
            ).cpu().numpy()
        )
        S["im"].append(
            o["impulse_mean_z"]
            .float()
            .cpu()
            .numpy()
        )
        S["isc"].append(
            torch.exp(
                o["impulse_log_scale_z"].float()
            ).cpu().numpy()
        )
        S["tail"].append(
            o["tail_q90_log"]
            .float()
            .cpu()
            .numpy()
        )

    q = {
        k: np.concatenate(v, axis=0)
        for k, v in S.items()
    }

    return {
        "continuous_mean":
            (
                q["mz"]
                * B9_PREP["y_std"][None]
                + B9_PREP["y_mean"][None]
            ).astype(np.float32),

        "continuous_scale":
            (
                q["sz"]
                * B9_PREP["y_std"][None]
            ).astype(np.float32),

        "contact_probability":
            q["c"].astype(np.float32),

        "touchdown_probability":
            q["td"].astype(np.float32),

        "impulse_mean":
            (
                q["im"]
                * B9_PREP[
                    "imp_std"
                ][None, None, None, :]
                + B9_PREP[
                    "imp_mean"
                ][None, None, None, :]
            ).astype(np.float32),

        "impulse_scale":
            (
                q["isc"]
                * B9_PREP[
                    "imp_std"
                ][None, None, None, :]
            ).astype(np.float32),

        "tail_q90":
            np.maximum(
                np.expm1(q["tail"]),
                0.0,
            ).astype(np.float32),

        "truth":
            d["Y_cont"][idx],

        "contact_truth":
            d["Y_contact"][idx],

        "touchdown_truth":
            d["Y_touchdown"][idx],

        "impulse_truth":
            d["Y_impulse"][idx],

        "origins":
            d["origins"][idx],
    }


# ----------------------------------------------------------------------
# 8. SAME VALIDATION-ONLY MODE SELECTION
# ----------------------------------------------------------------------
val_ids = [
    rid
    for rid, d in C7R_DATA.items()
    if np.any(d["split"] == "val")
]

BASE2 = {
    rid: predict_B_fast(
        C7R_DATA[rid],
        "val",
    )
    for rid in val_ids
}

TRUE2 = {}
CAL2 = {}

for mode, bundle in C7S_MODELS.items():
    TRUE2[mode] = {
        rid: predict_C7S(
            bundle,
            rid,
            C7R_DATA[rid],
            "val",
            "true",
            SEED2,
        )
        for rid in val_ids
    }
    CAL2[mode] = fit_cal(
        mode,
        TRUE2[mode],
    )

rows = []

for mode in ("rgb", "depth", "rgbd"):
    for rid in val_ids:
        d = C7R_DATA[rid]
        vm = d["split"] == "val"

        rows.append({
            "mode": mode,
            "run_id": rid,
            "terrain": d["terrain"],
            "subset": "all",
            **metrics(
                TRUE2[mode][rid],
                d,
                CAL2[mode],
            ),
        })

        rows.append({
            "mode": mode,
            "run_id": rid,
            "terrain": d["terrain"],
            "subset": "precontact",
            **metrics(
                TRUE2[mode][rid],
                d,
                CAL2[mode],
                d["C_precontact"][vm],
            ),
        })

for rid in val_ids:
    d = C7R_DATA[rid]
    vm = d["split"] == "val"

    rows.append({
        "mode": "B9_baseline",
        "run_id": rid,
        "terrain": d["terrain"],
        "subset": "all",
        **metrics(
            BASE2[rid],
            d,
            B9_CALIBRATION,
        ),
    })

    rows.append({
        "mode": "B9_baseline",
        "run_id": rid,
        "terrain": d["terrain"],
        "subset": "precontact",
        **metrics(
            BASE2[rid],
            d,
            B9_CALIBRATION,
            d["C_precontact"][vm],
        ),
    })

C7S_VALIDATION = pd.DataFrame(rows)
display(C7S_VALIDATION)

base_pre = C7S_VALIDATION[
    (C7S_VALIDATION["mode"] == "B9_baseline")
    & (C7S_VALIDATION["subset"] == "precontact")
].mean(numeric_only=True)

selection_rows = []

for mode in ("rgb", "depth", "rgbd"):
    g = C7S_VALIDATION[
        (C7S_VALIDATION["mode"] == mode)
        & (C7S_VALIDATION["subset"] == "precontact")
    ].mean(numeric_only=True)

    force_gain = (
        base_pre["force_tail_component_mae"]
        - g["force_tail_component_mae"]
    ) / max(
        base_pre["force_tail_component_mae"],
        1e-9,
    )

    E_gain = (
        base_pre["E_tail_component_mae"]
        - g["E_tail_component_mae"]
    ) / max(
        base_pre["E_tail_component_mae"],
        1e-9,
    )

    score = (
        0.35 * g["risk_spearman"]
        + 0.25 * g["force_peak_spearman"]
        + 0.20 * force_gain
        + 0.10 * E_gain
        + 0.10 * g["touchdown_f1"]
    )

    selection_rows.append({
        "mode": mode,
        "validation_selection_score": float(score),
        "precontact_risk_spearman":
            float(g["risk_spearman"]),
        "precontact_force_peak_spearman":
            float(g["force_peak_spearman"]),
        "precontact_force_tail_gain_vs_B9":
            float(force_gain),
        "precontact_E_tail_gain_vs_B9":
            float(E_gain),
        "precontact_touchdown_f1":
            float(g["touchdown_f1"]),
    })

C7S_SELECTION = (
    pd.DataFrame(selection_rows)
    .sort_values(
        "validation_selection_score",
        ascending=False,
    )
    .reset_index(drop=True)
)

display(C7S_SELECTION)

C7S_SELECTED_MODE = str(
    C7S_SELECTION.iloc[0]["mode"]
)

print("C7S SELECTED:", C7S_SELECTED_MODE)

# ----------------------------------------------------------------------
# 9. SAME TRUE / SHUFFLED / RUN-MEAN ANTI-CHEATING CONTROLS
# ----------------------------------------------------------------------
control_rows = []

for rid in val_ids:
    d = C7R_DATA[rid]
    vm = d["split"] == "val"
    mask = d["C_precontact"][vm]

    control_rows.append({
        "condition": "B9_baseline",
        "run_id": rid,
        "terrain": d["terrain"],
        **metrics(
            BASE2[rid],
            d,
            B9_CALIBRATION,
            mask,
        ),
    })

    control_rows.append({
        "condition": "true",
        "run_id": rid,
        "terrain": d["terrain"],
        **metrics(
            TRUE2[C7S_SELECTED_MODE][rid],
            d,
            CAL2[C7S_SELECTED_MODE],
            mask,
        ),
    })

    seed = (
        SEED2
        + int(
            hashlib.sha256(
                rid.encode()
            ).hexdigest()[:8],
            16,
        )
        % 100000
    )

    for cond in ("shuffled", "run_mean"):
        pp = predict_C7S(
            C7S_MODELS[C7S_SELECTED_MODE],
            rid,
            d,
            "val",
            cond,
            seed,
        )

        control_rows.append({
            "condition": cond,
            "run_id": rid,
            "terrain": d["terrain"],
            **metrics(
                pp,
                d,
                CAL2[C7S_SELECTED_MODE],
                mask,
            ),
        })

C7S_CONTROLS = pd.DataFrame(control_rows)
display(C7S_CONTROLS)

C7S_SUMMARY = (
    C7S_CONTROLS
    .groupby("condition")
    .mean(numeric_only=True)
)

display(C7S_SUMMARY)

def _v(cond, metric):
    return float(
        C7S_SUMMARY.loc[cond, metric]
    )

true_risk = _v("true", "risk_spearman")
base_risk = _v("B9_baseline", "risk_spearman")
shuf_risk = _v("shuffled", "risk_spearman")
mean_risk = _v("run_mean", "risk_spearman")

true_peak = _v("true", "force_peak_spearman")
base_peak = _v(
    "B9_baseline",
    "force_peak_spearman",
)

base_tail = _v(
    "B9_baseline",
    "force_tail_component_mae",
)
true_tail = _v(
    "true",
    "force_tail_component_mae",
)

# ----------------------------------------------------------------------
# 10. GATE — EXACT SAME THRESHOLDS AS C7R
# ----------------------------------------------------------------------
C7S_GATE = {
    "selected_mode": C7S_SELECTED_MODE,
    "feature_version": C7S_FEATURE_VERSION,

    "fresh_test_ids_still_locked":
        _fresh_ids,

    "validation_precontact_true_risk_spearman":
        true_risk,

    "validation_precontact_B9_risk_spearman":
        base_risk,

    "validation_precontact_shuffled_risk_spearman":
        shuf_risk,

    "validation_precontact_runmean_risk_spearman":
        mean_risk,

    "validation_precontact_true_force_peak_spearman":
        true_peak,

    "validation_precontact_B9_force_peak_spearman":
        base_peak,

    "validation_precontact_force_tail_improvement_fraction":
        (
            base_tail - true_tail
        ) / max(base_tail, 1e-9),

    # SAME thresholds as the previous scientific gate.
    "true_beats_B9_risk":
        bool(true_risk > base_risk + 0.01),

    "true_beats_shuffled_risk":
        bool(true_risk > shuf_risk + 0.01),

    "true_beats_runmean_risk":
        bool(true_risk > mean_risk + 0.005),

    "tail_or_peak_improves":
        bool(
            true_peak > base_peak + 0.01
            or true_tail < base_tail
        ),
}

C7S_GATE["open_fresh_C_test"] = bool(
    C7S_GATE["true_beats_B9_risk"]
    and C7S_GATE["true_beats_shuffled_risk"]
    and C7S_GATE["true_beats_runmean_risk"]
    and C7S_GATE["tail_or_peak_improves"]
)

print(
    "\n"
    + "=" * 92
    + "\nC7S CAUSAL HIGH-PASS / ALIGNMENT VALIDATION GATE"
    + "\n"
    + "=" * 92
)

print(
    json.dumps(
        C7S_GATE,
        indent=2,
    )
)

C7S_VALIDATION.to_csv(
    C7S_TABLE_ROOT / "validation_metrics.csv",
    index=False,
)
C7S_SELECTION.to_csv(
    C7S_TABLE_ROOT / "mode_selection.csv",
    index=False,
)
C7S_CONTROLS.to_csv(
    C7S_TABLE_ROOT / "causality_controls.csv",
    index=False,
)
(
    C7S_TABLE_ROOT / "gate.json"
).write_text(
    json.dumps(
        C7S_GATE,
        indent=2,
    )
)

if C7S_GATE["open_fresh_C_test"]:
    print(
        "\nPASS. Fresh C test trajectories are STILL UNSCORED. "
        "STOP HERE and send me the complete C7S gate output. "
        "We will then do the one-time C8-C10 evaluation."
    )
else:
    print(
        "\nNOT PASSED. Fresh C tests remain locked. "
        "Do not weaken or bypass the gate."
    )
