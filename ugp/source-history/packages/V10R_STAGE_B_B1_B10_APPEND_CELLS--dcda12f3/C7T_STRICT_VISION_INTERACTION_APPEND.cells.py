# Source inspection export of the adjacent notebook.
# Notebook execution order and dependencies still apply.
# IPython magic, shell lines and prose-only conclusion cells are comments.

# %% notebook cell 1

# ======================================================================
# C7T — STRICT VISION-DEPENDENT INTERACTION REPAIR (VALIDATION ONLY)
#
# Run this AFTER the completed C7R + C7S cells in the SAME Colab runtime.
#
# ORIGINAL STAGE-C DISCIPLINE IS PRESERVED:
#   frozen B9 world model
#   same real physics targets / horizons / losses
#   same six train + two validation trajectories
#   same RGB / depth / RGB-D mode selection on validation only
#   same true / shuffled / run-mean anti-cheating controls
#   same scientific gate thresholds
#   fresh C tests P2027/P2006 remain unopened
#
# ONLY SCIENTIFIC CHANGE FROM C7S:
#   remove the non-visual residual bypass.
#
# C7S allowed:
#       residual = F([B9_latent, vision])
# so F could improve B9 while mostly ignoring vision.
#
# C7T enforces:
#       q_s = S(B9_latent)
#       q_v = V(vision), with V(0)=0
#       interaction = q_s * q_v
#       residual = O(interaction), with O(0)=0
#
# Therefore, BY ARCHITECTURE:
#       normalized vision == 0  =>  residual == 0  => prediction == frozen B9
#
# We keep the C7S causal high-pass visual features unchanged so this experiment
# isolates the FUSION/RESIDUAL ARCHITECTURE as the changed variable.
# ======================================================================

import gc, time, json, hashlib, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from IPython.display import display

# ----------------------------------------------------------------------
# 0. HARD STATE / TEST-LEAKAGE GUARDS
# ----------------------------------------------------------------------
_REQUIRED_T = [
    # Frozen Stage-B / Stage-C infrastructure
    "C_ROOT", "C7R_DATA", "TEST_ROWS",
    "B9_CKPT", "B9_PREP", "B9_CALIBRATION", "B9_DENSE",
    "BResponseModel", "B_DEVICE", "B_H", "B_STEP_WIDTH",
    "B_STEP_LAYOUT", "B_MODEL_DIM", "B_ACTION_HIDDEN",
    "B_ACTION_GATHER", "TAIL_NAMES",
    "b_loss", "predict_B_fast", "fit_cal", "metrics",
    # C7S representation + exact data protocol
    "C7S_FRAME", "C7S_VIS_NORM", "C7S_NORM_T",
    "C7SIndexDataset", "c7s_make_collate", "c7s_prep_batch",
    "c7s_origin_visual", "c7s_fresh_base",
    "C7S_TRAIN_BATCH", "C7S_FETCH_BATCH", "C7S_EPOCHS",
    "C7S_PATIENCE", "C7S_LR", "C7S_WEIGHT_DECAY",
    "C7S_RESIDUAL_L2", "C7S_NEG_NULL_WEIGHT",
    "C7S_PRECONTACT_EXTRA_WEIGHT", "C7S_USE_AMP",
    "C7S_FEATURE_VERSION",
]

_missing_t = [x for x in _REQUIRED_T if x not in globals()]
if _missing_t:
    raise RuntimeError(
        "C7T must be appended after the completed C7R + C7S runtime. "
        "Missing globals: " + repr(_missing_t)
    )

C8_MARKER_T = C_ROOT / "C8_TEST_WAS_SCORED.json"
if C8_MARKER_T.exists():
    raise RuntimeError(
        f"Fresh Stage-C test marker already exists: {C8_MARKER_T}. "
        "Do not use C7T for further tuning."
    )

C7T_FRESH_IDS = TEST_ROWS["run_id"].astype(str).tolist()

if set(C7T_FRESH_IDS) & set(C7R_DATA.keys()):
    raise RuntimeError(
        "Fresh C test trajectories unexpectedly appear in C7R_DATA. "
        "Refusing to continue."
    )

# Development data must remain exactly 6 train + 2 validation runs.
_c7t_train_runs = [
    rid for rid, d in C7R_DATA.items()
    if np.any(np.asarray(d["split"]) == "train")
]
_c7t_val_runs = [
    rid for rid, d in C7R_DATA.items()
    if np.any(np.asarray(d["split"]) == "val")
]

if len(_c7t_train_runs) != 6 or len(_c7t_val_runs) != 2:
    raise RuntimeError(
        f"Unexpected C development split. train={_c7t_train_runs}, "
        f"val={_c7t_val_runs}"
    )

print("Fresh C tests remain LOCKED:", C7T_FRESH_IDS)
print("C7T train runs:", _c7t_train_runs)
print("C7T validation runs:", _c7t_val_runs)

# ----------------------------------------------------------------------
# 1. NEW EXPERIMENT NAMESPACE
# ----------------------------------------------------------------------
C7T_SEED = 78120260921
random.seed(C7T_SEED)
np.random.seed(C7T_SEED)
torch.manual_seed(C7T_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(C7T_SEED)

C7T_ROOT = C_ROOT / "spatial_temporal_repair_v3_strict_visual_interaction"
C7T_MODEL_ROOT = C7T_ROOT / "models"
C7T_TABLE_ROOT = C7T_ROOT / "tables"
for _p in (C7T_ROOT, C7T_MODEL_ROOT, C7T_TABLE_ROOT):
    _p.mkdir(parents=True, exist_ok=True)

# Important: C7S features are deliberately reused unchanged.
C7T_FEATURE_VERSION = (
    "C7T_strict_multiplicative_visual_interaction_v1__"
    + str(C7S_FEATURE_VERSION)
)

# Keep the SAME optimizer/data protocol as C7S.
C7T_TRAIN_BATCH = int(C7S_TRAIN_BATCH)      # 96
C7T_FETCH_BATCH = int(C7S_FETCH_BATCH)      # 384; only I/O/vectorization
C7T_EPOCHS = int(C7S_EPOCHS)
C7T_PATIENCE = int(C7S_PATIENCE)
C7T_LR = float(C7S_LR)
C7T_WEIGHT_DECAY = float(C7S_WEIGHT_DECAY)
C7T_RESIDUAL_L2 = float(C7S_RESIDUAL_L2)
C7T_NEG_NULL_WEIGHT = float(C7S_NEG_NULL_WEIGHT)
C7T_USE_AMP = bool(C7S_USE_AMP)

# Match the B9 latent width rather than changing capacity arbitrarily.
C7T_INTER_DIM = int(B_MODEL_DIM)

print({
    "experiment": C7T_FEATURE_VERSION,
    "visual_features_changed_from_C7S": False,
    "fusion_architecture_changed": True,
    "optimizer_batch": C7T_TRAIN_BATCH,
    "fetch_batch": C7T_FETCH_BATCH,
    "max_epochs": C7T_EPOCHS,
    "patience": C7T_PATIENCE,
    "lr": C7T_LR,
    "weight_decay": C7T_WEIGHT_DECAY,
    "residual_l2": C7T_RESIDUAL_L2,
    "negative_null_weight": C7T_NEG_NULL_WEIGHT,
    "fresh_tests_loaded": False,
})

# ----------------------------------------------------------------------
# 2. STRICT-ZERO BUILDING BLOCKS
#
# Every transform on the VISUAL path and every transform AFTER the
# multiplicative interaction is bias-free. Thus:
#
#     visual == 0 -> visual code == 0
#     interaction == 0
#     residual hidden == 0
#     every residual head == 0
#
# State projections MAY have bias because they are multiplied by q_v.
# ----------------------------------------------------------------------
class C7TStrictZeroVisual(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        # GELU(0)=0 and tanh(0)=0.
        return torch.tanh(self.fc2(F.gelu(self.fc1(x))))


class C7TStrictZeroMixer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim, bias=False)
        self.fc2 = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class C7TStrictVisionResidualModel(nn.Module):
    """
    Frozen B9 + strictly vision-dependent multiplicative residual.

    Horizon branch:
        z_h        = frozen B9 horizon/action-conditioned latent
        q_s(h)     = tanh(S_h(z_h))
        q_v        = V(v), V(0)=0
        g_h        = q_s(h) * q_v
        r_h        = M_h(g_h), M_h(0)=0
        output_h   = B9_h + bias-free heads(r_h)

    Sequence/tail branch:
        s          = [history_latent, final_action_latent]
        q_s_seq    = tanh(S_seq(s))
        g_seq      = q_s_seq * q_v
        r_seq      = M_seq(g_seq)
        tail       = B9_tail + bias-free head(r_seq)

    Therefore the adapter CANNOT perform a state/action-only correction.
    """
    def __init__(self, base, rgb_dim, depth_dim, mode):
        super().__init__()
        if mode not in {"rgb", "depth", "rgbd"}:
            raise ValueError(mode)

        self.mode = mode
        self.base = base.eval()

        for p in self.base.parameters():
            p.requires_grad_(False)

        if mode == "rgb":
            visual_dim = int(rgb_dim)
        elif mode == "depth":
            visual_dim = int(depth_dim)
        else:
            visual_dim = int(rgb_dim + depth_dim)

        self.visual = C7TStrictZeroVisual(
            visual_dim, C7T_INTER_DIM
        )

        # State projections are allowed bias; no bypass is possible because
        # their outputs are multiplied by the strict-zero visual code.
        self.state_h = nn.Sequential(
            nn.Linear(B_MODEL_DIM, C7T_INTER_DIM),
            nn.GELU(),
            nn.Linear(C7T_INTER_DIM, C7T_INTER_DIM),
            nn.Tanh(),
        )

        self.state_seq = nn.Sequential(
            nn.Linear(
                B_MODEL_DIM + B_ACTION_HIDDEN,
                C7T_INTER_DIM,
            ),
            nn.GELU(),
            nn.Linear(C7T_INTER_DIM, C7T_INTER_DIM),
            nn.Tanh(),
        )

        # After interaction: all bias-free => zero remains exactly zero.
        self.mix_h = C7TStrictZeroMixer(C7T_INTER_DIM)
        self.mix_seq = C7TStrictZeroMixer(C7T_INTER_DIM)

        self.d_mean = nn.Linear(
            C7T_INTER_DIM, B_STEP_WIDTH, bias=False
        )
        self.d_log_scale = nn.Linear(
            C7T_INTER_DIM, B_STEP_WIDTH, bias=False
        )
        self.d_contact = nn.Linear(
            C7T_INTER_DIM, 4, bias=False
        )
        self.d_touchdown = nn.Linear(
            C7T_INTER_DIM, 4, bias=False
        )
        self.d_impulse_mean = nn.Linear(
            C7T_INTER_DIM, 12, bias=False
        )
        self.d_impulse_log_scale = nn.Linear(
            C7T_INTER_DIM, 12, bias=False
        )
        self.d_tail = nn.Linear(
            C7T_INTER_DIM, 4, bias=False
        )

        # Start EXACTLY at B9. This is the same residual-learning philosophy
        # as earlier C repairs, but now zero vision remains B9 forever.
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

    def _visual_input(self, rgb, depth):
        if self.mode == "rgb":
            return rgb
        if self.mode == "depth":
            return depth
        return torch.cat([rgb, depth], dim=-1)

    def forward(self, x, a, rgb, depth):
        self.base.eval()

        # Reproduce the frozen B9 forward path while retaining its latent z.
        with torch.no_grad():
            h0 = self.base.history(x)

            if self.base.use_action:
                ah = self.base.action(
                    a, B_ACTION_GATHER.to(a.device)
                )
                afinal = ah[:, -1]
            else:
                n = len(x)
                ah = torch.zeros(
                    n, B_H, B_ACTION_HIDDEN,
                    device=x.device,
                    dtype=x.dtype,
                )
                afinal = torch.zeros(
                    n, B_ACTION_HIDDEN,
                    device=x.device,
                    dtype=x.dtype,
                )

            h = h0[:, None, :].expand(-1, B_H, -1)
            he = self.base.horizon_embed[None].expand(
                len(x), -1, -1
            )

            z = self.base.fuse(
                torch.cat([h, ah, he], dim=-1)
            )

            base = {
                "mean_z":
                    self.base.mean_head(z),
                "log_scale_z":
                    torch.clamp(
                        self.base.log_scale_head(z),
                        -5.5, 2.8,
                    ),
                "contact_logits":
                    self.base.contact_head(z),
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

        # ---------------- STRICT VISUAL INTERACTION ----------------
        v_raw = self._visual_input(rgb, depth)
        qv = self.visual(v_raw)                  # [N, D], V(0)=0

        qs_h = self.state_h(z)                   # [N,H,D]
        interaction_h = (
            qs_h * qv[:, None, :]
        )
        rh = self.mix_h(interaction_h)           # M_h(0)=0

        seq_state = torch.cat([h0, afinal], dim=-1)
        qs_seq = self.state_seq(seq_state)
        interaction_seq = qs_seq * qv
        rs = self.mix_seq(interaction_seq)       # M_seq(0)=0

        dm = self.d_mean(rh)
        dls = self.d_log_scale(rh)
        dc = self.d_contact(rh)
        dtd = self.d_touchdown(rh)
        dimp = self.d_impulse_mean(rh).reshape(
            -1, B_H, 4, 3
        )
        dil = self.d_impulse_log_scale(rh).reshape(
            -1, B_H, 4, 3
        )
        dtail = self.d_tail(rs)

        out = {
            "mean_z":
                base["mean_z"] + dm,
            "log_scale_z":
                torch.clamp(
                    base["log_scale_z"] + dls,
                    -5.5, 2.8,
                ),
            "contact_logits":
                base["contact_logits"] + dc,
            "touchdown_logits":
                base["touchdown_logits"] + dtd,
            "impulse_mean_z":
                base["impulse_mean_z"] + dimp,
            "impulse_log_scale_z":
                torch.clamp(
                    base["impulse_log_scale_z"] + dil,
                    -5.5, 3.0,
                ),
            "tail_q90_log":
                base["tail_q90_log"] + dtail,
        }

        # Full residual energy: used for regularization and wrong-time null.
        residual_terms = [
            dm,
            0.316227766 * dls,  # sqrt(.10)
            0.223606798 * dc,   # sqrt(.05)
            0.316227766 * dtd,  # sqrt(.10)
            0.5 * dimp,         # sqrt(.25)
            0.223606798 * dil,  # sqrt(.05)
            0.5 * dtail,        # sqrt(.25)
        ]

        out["_residual_full_l2"] = sum(
            t.square().mean() for t in residual_terms
        )

        out["_residual_max_abs"] = torch.stack(
            [t.abs().max() for t in residual_terms]
        ).max()

        return out


# ----------------------------------------------------------------------
# 3. FORMAL ARCHITECTURE SANITY CHECK BEFORE TRAINING
# ----------------------------------------------------------------------
@torch.no_grad()
def c7t_zero_invariant_check(model, n=8):
    model.eval()

    rid = _c7t_train_runs[0]
    d = C7R_DATA[rid]
    pos = np.flatnonzero(
        np.asarray(d["split"]) == "train"
    )[:n]

    x = torch.from_numpy(
        np.asarray(d["X_core"][pos], dtype=np.float32)
    ).to(B_DEVICE)

    a = torch.from_numpy(
        np.asarray(
            d["X_action_dense"][pos],
            dtype=np.float32,
        )
    ).to(B_DEVICE)

    x = (
        (x - C7S_NORM_T["x_mean"][None, :])
        / C7S_NORM_T["x_std"][None, :]
    )
    a = (
        (a - C7S_NORM_T["action_mean"][None, None, :])
        / C7S_NORM_T["action_std"][None, None, :]
    )

    zr = torch.zeros(
        len(pos),
        len(C7S_VIS_NORM["rgb_mean"]),
        device=B_DEVICE,
        dtype=x.dtype,
    )
    zd = torch.zeros(
        len(pos),
        len(C7S_VIS_NORM["depth_mean"]),
        device=B_DEVICE,
        dtype=x.dtype,
    )

    out = model(x, a, zr, zd)

    # This is the strongest direct test: every learned residual tensor must
    # be identically zero when normalized visual input is zero.
    residual_max = float(
        out["_residual_max_abs"].float().cpu()
    )

    # Independent external comparison against the frozen base.
    base = model.base(x, a)
    keys = [
        "mean_z",
        "log_scale_z",
        "contact_logits",
        "touchdown_logits",
        "impulse_mean_z",
        "impulse_log_scale_z",
        "tail_q90_log",
    ]
    max_diff = 0.0
    for k in keys:
        diff = float(
            (out[k] - base[k]).abs().max().float().cpu()
        )
        max_diff = max(max_diff, diff)

    return {
        "zero_visual_residual_max_abs": residual_max,
        "zero_visual_vs_B9_max_abs": max_diff,
        "pass": bool(
            residual_max <= 1e-8
            and max_diff <= 1e-6
        ),
    }


for _mode in ("rgb", "depth", "rgbd"):
    _m = C7TStrictVisionResidualModel(
        c7s_fresh_base(),
        len(C7S_VIS_NORM["rgb_mean"]),
        len(C7S_VIS_NORM["depth_mean"]),
        _mode,
    ).to(B_DEVICE)

    _chk = c7t_zero_invariant_check(_m)
    print("PRETRAIN ZERO-INVARIANT", _mode, _chk)
    if not _chk["pass"]:
        raise RuntimeError(
            f"C7T zero-vision architecture invariant failed "
            f"before training for {_mode}: {_chk}"
        )

    del _m
    gc.collect()
    torch.cuda.empty_cache()

print("C7T ARCHITECTURE ZERO-VISION INVARIANT: PASS")

# ----------------------------------------------------------------------
# 4. TRAIN / RESUME
#
# Uses EXACTLY the C7S dataset/collate/normalization:
#   same train/val samples
#   same terrain/tail/event sampling
#   same pre-contact extra TRAIN weight
#   same deterministic wrong-time same-run visual negatives
#
# The only changed variable is the residual fusion architecture above.
# ----------------------------------------------------------------------
def c7t_adapter_state(model):
    return {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if not k.startswith("base.")
    }


def c7t_train_mode(mode):
    final = C7T_MODEL_ROOT / f"C7T_{mode}_adapter.pt"
    resume = C7T_MODEL_ROOT / f"C7T_{mode}_resume.pt"

    if final.exists():
        s = torch.load(
            final,
            map_location="cpu",
            weights_only=False,
        )
        if (
            s.get("feature_version")
            == C7T_FEATURE_VERSION
        ):
            print(
                f"C7T-{mode}: reusing completed checkpoint "
                f"{final}"
            )
            return {
                "mode": mode,
                "checkpoint": str(final),
                "best_val_loss":
                    float(s["best_val_loss"]),
                "zero_invariant":
                    s["zero_invariant"],
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
        batch_size=C7T_FETCH_BATCH,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=c7s_make_collate(tr),
    )

    val = DataLoader(
        va,
        batch_size=C7T_FETCH_BATCH,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=c7s_make_collate(va),
    )

    model = C7TStrictVisionResidualModel(
        c7s_fresh_base(),
        len(C7S_VIS_NORM["rgb_mean"]),
        len(C7S_VIS_NORM["depth_mean"]),
        mode,
    ).to(B_DEVICE)

    precheck = c7t_zero_invariant_check(model)
    if not precheck["pass"]:
        raise RuntimeError(
            f"{mode}: zero invariant failed before training: "
            f"{precheck}"
        )

    params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    opt = torch.optim.AdamW(
        params,
        lr=C7T_LR,
        weight_decay=C7T_WEIGHT_DECAY,
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=C7T_USE_AMP
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
            rr.get("feature_version")
            == C7T_FEATURE_VERSION
            and rr.get("mode") == mode
        ):
            miss, un = model.load_state_dict(
                rr["adapter_state_dict"],
                strict=False,
            )
            bad_missing = [
                x for x in miss
                if not x.startswith("base.")
            ]
            if bad_missing or un:
                raise RuntimeError(
                    (bad_missing, un)
                )

            opt.load_state_dict(
                rr["optimizer_state"]
            )
            best = float(rr["best"])
            best_state = rr["best_state"]
            bad = int(rr["bad"])
            hist = list(rr["history"])
            start_epoch = int(rr["epoch"]) + 1

            print(
                f"C7T-{mode}: resuming at epoch "
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
                enabled=C7T_USE_AMP,
            ):
                out = model(
                    bb[0], bb[1], r, d
                )

                loss = (
                    b_loss(out, bb)
                    + C7T_RESIDUAL_L2
                    * out["_residual_full_l2"]
                )

            total += (
                float(loss.detach()) * len(bb[0])
            )
            n += len(bb[0])

        return total / max(n, 1)

    for ep in range(
        start_epoch,
        C7T_EPOCHS + 1,
    ):
        tic = time.time()
        model.train()

        total = 0.0
        n = 0

        for raw_big in trl:
            # IMPORTANT:
            # fetch batch 384 is only vectorized I/O.
            # Optimizer semantics remain the original 96 samples/step.
            for off in range(
                0,
                len(raw_big[0]),
                C7T_TRAIN_BATCH,
            ):
                raw = tuple(
                    t[off:off+C7T_TRAIN_BATCH]
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
                    enabled=C7T_USE_AMP,
                ):
                    out_true = model(
                        bb[0], bb[1], r, d
                    )

                    # Same C7S train-only wrong-time same-run
                    # negative. It is NOT a validation/test label.
                    out_neg = model(
                        bb[0], bb[1], rn, dn
                    )

                    loss_true = (
                        b_loss(out_true, bb)
                        + C7T_RESIDUAL_L2
                        * out_true[
                            "_residual_full_l2"
                        ]
                    )

                    # Mismatched visual evidence should not be
                    # allowed to move frozen B9.
                    loss_null = (
                        C7T_NEG_NULL_WEIGHT
                        * out_neg[
                            "_residual_full_l2"
                        ]
                    )

                    loss = (
                        loss_true + loss_null
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(opt)

                torch.nn.utils.clip_grad_norm_(
                    params, 5.0
                )

                scaler.step(opt)
                scaler.update()

                total += (
                    float(loss.detach())
                    * len(bb[0])
                )
                n += len(bb[0])

        train_loss = total / max(n, 1)
        val_loss = eval_loss()

        zchk = c7t_zero_invariant_check(model)
        if not zchk["pass"]:
            raise RuntimeError(
                f"C7T-{mode}: zero invariant was violated "
                f"after epoch {ep}: {zchk}"
            )

        hist.append({
            "mode": mode,
            "epoch": ep,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "seconds": time.time() - tic,
            **zchk,
        })

        print(
            f"C7T-{mode:5s} "
            f"ep={ep:02d} "
            f"train={train_loss:.5f} "
            f"val={val_loss:.5f} "
            f"zeroΔ={zchk['zero_visual_vs_B9_max_abs']:.2e} "
            f"time={time.time()-tic:.1f}s"
        )

        if val_loss < best - 1e-4:
            best = val_loss
            best_state = {
                k: v.clone()
                for k, v in c7t_adapter_state(
                    model
                ).items()
            }
            bad = 0
        else:
            bad += 1

        current = c7t_adapter_state(model)

        torch.save({
            "mode": mode,
            "feature_version":
                C7T_FEATURE_VERSION,
            "epoch": ep,
            "adapter_state_dict":
                current,
            "optimizer_state":
                opt.state_dict(),
            "best": best,
            "best_state":
                best_state,
            "bad": bad,
            "history": hist,
        }, resume)

        if bad >= C7T_PATIENCE:
            print(
                f"C7T-{mode}: early stop"
            )
            break

    if best_state is None:
        raise RuntimeError(
            f"C7T-{mode}: no best state found"
        )

    # Load best state before final architecture audit.
    miss, un = model.load_state_dict(
        best_state,
        strict=False,
    )
    bad_missing = [
        x for x in miss
        if not x.startswith("base.")
    ]
    if bad_missing or un:
        raise RuntimeError(
            (bad_missing, un)
        )

    postcheck = c7t_zero_invariant_check(
        model
    )
    if not postcheck["pass"]:
        raise RuntimeError(
            f"C7T-{mode}: best checkpoint violates "
            f"zero-vision invariant: {postcheck}"
        )

    torch.save({
        "mode": mode,
        "adapter_state_dict":
            best_state,
        "best_val_loss": float(best),
        "vis_norm": C7S_VIS_NORM,
        "feature_version":
            C7T_FEATURE_VERSION,
        "source_visual_feature_version":
            C7S_FEATURE_VERSION,
        "B9_checkpoint":
            str(B9_CKPT),
        "uses_test": False,
        "training_history": hist,
        "zero_invariant": postcheck,
        "train_protocol": {
            "optimizer_batch":
                C7T_TRAIN_BATCH,
            "fetch_batch":
                C7T_FETCH_BATCH,
            "negative_null_weight":
                C7T_NEG_NULL_WEIGHT,
            "residual_l2":
                C7T_RESIDUAL_L2,
            "precontact_extra_weight":
                C7S_PRECONTACT_EXTRA_WEIGHT,
        },
    }, final)

    pd.DataFrame(hist).to_csv(
        C7T_TABLE_ROOT
        / f"C7T_{mode}_history.csv",
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
        "zero_invariant": postcheck,
    }


# Keep only one large model resident when possible.
B9_DENSE["model"].to("cpu")
if "C7S_ACTIVE" in globals():
    if C7S_ACTIVE.get("model") is not None:
        C7S_ACTIVE["model"].to("cpu")
        C7S_ACTIVE["model"] = None
        C7S_ACTIVE["checkpoint"] = None

gc.collect()
torch.cuda.empty_cache()

C7T_MODELS = {}

print(
    "\nTraining strict vision-dependent C7T adapters..."
)

for mode in ("rgb", "depth", "rgbd"):
    print(
        "\n" + "=" * 80
        + f"\nC7T MODE: {mode}"
        + "\n" + "=" * 80
    )
    C7T_MODELS[mode] = c7t_train_mode(
        mode
    )

print(
    "\nC7T TRAINING COMPLETE / RESTORED"
)

display(pd.DataFrame([
    {
        "mode": mode,
        "best_val_loss":
            bundle["best_val_loss"],
        "zero_invariant_pass":
            bundle["zero_invariant"]["pass"],
        "checkpoint":
            bundle["checkpoint"],
    }
    for mode, bundle
    in C7T_MODELS.items()
]))

# ----------------------------------------------------------------------
# 5. MODEL LOADER / PREDICTOR
# ----------------------------------------------------------------------
C7T_ACTIVE = {
    "checkpoint": None,
    "model": None,
}


def c7t_get_model(bundle):
    if (
        C7T_ACTIVE["checkpoint"]
        == bundle["checkpoint"]
        and C7T_ACTIVE["model"] is not None
    ):
        return C7T_ACTIVE["model"]

    if C7T_ACTIVE["model"] is not None:
        C7T_ACTIVE["model"].to("cpu")

    s = torch.load(
        bundle["checkpoint"],
        map_location="cpu",
        weights_only=False,
    )

    if (
        s.get("feature_version")
        != C7T_FEATURE_VERSION
    ):
        raise RuntimeError(
            "Wrong C7T checkpoint version."
        )

    m = C7TStrictVisionResidualModel(
        c7s_fresh_base(),
        len(C7S_VIS_NORM["rgb_mean"]),
        len(C7S_VIS_NORM["depth_mean"]),
        s["mode"],
    )

    miss, un = m.load_state_dict(
        s["adapter_state_dict"],
        strict=False,
    )

    bad_missing = [
        x for x in miss
        if not x.startswith("base.")
    ]

    if bad_missing or un:
        raise RuntimeError(
            (bad_missing, un)
        )

    m = m.to(B_DEVICE).eval()

    zchk = c7t_zero_invariant_check(m)
    if not zchk["pass"]:
        raise RuntimeError(
            f"Loaded C7T checkpoint violated "
            f"zero invariant: {zchk}"
        )

    C7T_ACTIVE["checkpoint"] = (
        bundle["checkpoint"]
    )
    C7T_ACTIVE["model"] = m
    return m


@torch.no_grad()
def predict_C7T(
    bundle,
    rid,
    d,
    split="val",
    control="true",
    seed=0,
    batch=256,
):
    """
    Controls:
      true      : correctly time-aligned C7S causal high-pass visual feature
      shuffled  : wrong time, same validation run
      run_mean  : one constant mean visual feature for the whole run
      zero      : normalized visual vector exactly zero

    'zero' is an architecture validity control, not used for model selection.
    """
    idx = np.flatnonzero(
        np.asarray(d["split"]) == split
    ).astype(np.int64)

    if not len(idx):
        raise RuntimeError(
            f"{rid}: no samples for split={split}"
        )

    if control not in {
        "true", "shuffled",
        "run_mean", "zero"
    }:
        raise ValueError(control)

    rng = np.random.default_rng(seed)
    perm = (
        rng.permutation(len(idx))
        if control == "shuffled"
        else None
    )

    if control == "run_mean":
        sr = sd = None
        nmean = 0

        for st in range(0, len(idx), 512):
            R, D = c7s_origin_visual(
                rid,
                d,
                idx[st:st+512],
            )

            sr = (
                R.sum(0, dtype=np.float64)
                if sr is None
                else sr
                + R.sum(0, dtype=np.float64)
            )

            sd = (
                D.sum(0, dtype=np.float64)
                if sd is None
                else sd
                + D.sum(0, dtype=np.float64)
            )

            nmean += len(R)

        mr = (sr / nmean).astype(
            np.float32
        )
        md = (sd / nmean).astype(
            np.float32
        )

    model = c7t_get_model(bundle)

    stores = {
        k: []
        for k in [
            "mz", "sz", "c", "td",
            "im", "isc", "tail",
            "resmax",
        ]
    }

    for st in range(0, len(idx), batch):
        ed = min(st + batch, len(idx))
        p = idx[st:ed]

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

        # ----------------------------------------------------------
        # Visual control construction
        # ----------------------------------------------------------
        if control == "zero":
            # IMPORTANT: zero is in NORMALIZED visual coordinates.
            # We intentionally do NOT construct raw zeros and then normalize.
            r = torch.zeros(
                len(p),
                len(C7S_VIS_NORM["rgb_mean"]),
                device=B_DEVICE,
                dtype=x.dtype,
            )
            dp = torch.zeros(
                len(p),
                len(C7S_VIS_NORM["depth_mean"]),
                device=B_DEVICE,
                dtype=x.dtype,
            )

        else:
            if control == "true":
                R, D = c7s_origin_visual(
                    rid, d, p
                )

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

            r = torch.from_numpy(
                np.asarray(
                    R, dtype=np.float32
                )
            ).to(B_DEVICE)

            dp = torch.from_numpy(
                np.asarray(
                    D, dtype=np.float32
                )
            ).to(B_DEVICE)

            r = (
                (
                    r
                    - C7S_NORM_T[
                        "rgb_mean"
                    ][None, :]
                )
                / C7S_NORM_T[
                    "rgb_std"
                ][None, :]
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
            enabled=C7T_USE_AMP,
        ):
            out = model(
                x, a, r, dp
            )

        stores["mz"].append(
            out["mean_z"]
            .float()
            .cpu()
            .numpy()
        )
        stores["sz"].append(
            torch.exp(
                out["log_scale_z"].float()
            ).cpu().numpy()
        )
        stores["c"].append(
            torch.sigmoid(
                out["contact_logits"].float()
            ).cpu().numpy()
        )
        stores["td"].append(
            torch.sigmoid(
                out["touchdown_logits"].float()
            ).cpu().numpy()
        )
        stores["im"].append(
            out["impulse_mean_z"]
            .float()
            .cpu()
            .numpy()
        )
        stores["isc"].append(
            torch.exp(
                out[
                    "impulse_log_scale_z"
                ].float()
            ).cpu().numpy()
        )
        stores["tail"].append(
            out["tail_q90_log"]
            .float()
            .cpu()
            .numpy()
        )
        stores["resmax"].append(
            np.asarray(
                [
                    float(
                        out[
                            "_residual_max_abs"
                        ].float().cpu()
                    )
                ],
                dtype=np.float32,
            )
        )

    q = {
        k: np.concatenate(v, axis=0)
        for k, v in stores.items()
    }

    return {
        "continuous_mean":
            (
                q["mz"]
                * B9_PREP[
                    "y_std"
                ][None, :, :]
                + B9_PREP[
                    "y_mean"
                ][None, :, :]
            ).astype(np.float32),

        "continuous_scale":
            (
                q["sz"]
                * B9_PREP[
                    "y_std"
                ][None, :, :]
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

        "_residual_max_abs":
            float(np.max(q["resmax"])),
    }


# ----------------------------------------------------------------------
# 6. VALIDATION-ONLY CALIBRATION + MODE SELECTION
#
# Exactly the original C-series logic:
#   select RGB/depth/RGB-D from TRUE validation predictions only.
#   Anti-cheating controls do NOT choose the model.
# ----------------------------------------------------------------------
C7T_VAL_IDS = list(_c7t_val_runs)

C7T_BASE = {
    rid: predict_B_fast(
        C7R_DATA[rid], "val"
    )
    for rid in C7T_VAL_IDS
}

C7T_TRUE = {}
C7T_CAL = {}

for mode, bundle in C7T_MODELS.items():
    C7T_TRUE[mode] = {
        rid: predict_C7T(
            bundle,
            rid,
            C7R_DATA[rid],
            split="val",
            control="true",
            seed=C7T_SEED,
        )
        for rid in C7T_VAL_IDS
    }

    C7T_CAL[mode] = fit_cal(
        mode,
        C7T_TRUE[mode],
    )

rows = []

for mode in ("rgb", "depth", "rgbd"):
    for rid in C7T_VAL_IDS:
        d = C7R_DATA[rid]
        vm = np.asarray(
            d["split"]
        ) == "val"

        rows.append({
            "mode": mode,
            "run_id": rid,
            "terrain": d["terrain"],
            "subset": "all",
            **metrics(
                C7T_TRUE[mode][rid],
                d,
                C7T_CAL[mode],
            ),
        })

        rows.append({
            "mode": mode,
            "run_id": rid,
            "terrain": d["terrain"],
            "subset": "precontact",
            **metrics(
                C7T_TRUE[mode][rid],
                d,
                C7T_CAL[mode],
                d["C_precontact"][vm],
            ),
        })

for rid in C7T_VAL_IDS:
    d = C7R_DATA[rid]
    vm = np.asarray(
        d["split"]
    ) == "val"

    rows.append({
        "mode": "B9_baseline",
        "run_id": rid,
        "terrain": d["terrain"],
        "subset": "all",
        **metrics(
            C7T_BASE[rid],
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
            C7T_BASE[rid],
            d,
            B9_CALIBRATION,
            d["C_precontact"][vm],
        ),
    })

C7T_VALIDATION = pd.DataFrame(rows)

print(
    "\nC7T TRUE VALIDATION METRICS"
)
display(C7T_VALIDATION)

base_pre = C7T_VALIDATION[
    (
        C7T_VALIDATION["mode"]
        == "B9_baseline"
    )
    & (
        C7T_VALIDATION["subset"]
        == "precontact"
    )
].mean(numeric_only=True)

selection_rows = []

for mode in ("rgb", "depth", "rgbd"):
    g = C7T_VALIDATION[
        (
            C7T_VALIDATION["mode"]
            == mode
        )
        & (
            C7T_VALIDATION["subset"]
            == "precontact"
        )
    ].mean(numeric_only=True)

    force_gain = (
        base_pre[
            "force_tail_component_mae"
        ]
        - g[
            "force_tail_component_mae"
        ]
    ) / max(
        base_pre[
            "force_tail_component_mae"
        ],
        1e-9,
    )

    E_gain = (
        base_pre[
            "E_tail_component_mae"
        ]
        - g[
            "E_tail_component_mae"
        ]
    ) / max(
        base_pre[
            "E_tail_component_mae"
        ],
        1e-9,
    )

    # EXACT SAME selection formula as C7R/C7S.
    score = (
        0.35 * g["risk_spearman"]
        + 0.25
        * g["force_peak_spearman"]
        + 0.20 * force_gain
        + 0.10 * E_gain
        + 0.10 * g["touchdown_f1"]
    )

    selection_rows.append({
        "mode": mode,
        "validation_selection_score":
            float(score),
        "precontact_risk_spearman":
            float(g["risk_spearman"]),
        "precontact_force_peak_spearman":
            float(
                g[
                    "force_peak_spearman"
                ]
            ),
        "precontact_force_tail_gain_vs_B9":
            float(force_gain),
        "precontact_E_tail_gain_vs_B9":
            float(E_gain),
        "precontact_touchdown_f1":
            float(g["touchdown_f1"]),
    })

C7T_SELECTION = (
    pd.DataFrame(selection_rows)
    .sort_values(
        "validation_selection_score",
        ascending=False,
    )
    .reset_index(drop=True)
)

print(
    "\nC7T VALIDATION-ONLY MODE SELECTION"
)
display(C7T_SELECTION)

C7T_SELECTED_MODE = str(
    C7T_SELECTION.iloc[0]["mode"]
)

print(
    "C7T SELECTED MODE:",
    C7T_SELECTED_MODE,
)

# ----------------------------------------------------------------------
# 7. ANTI-CHEATING CONTROLS + ZERO-VISION ARCHITECTURE CONTROL
# ----------------------------------------------------------------------
control_rows = []
C7T_ZERO_RAW_MAX_DIFF = 0.0
C7T_ZERO_RESIDUAL_MAX = 0.0

_RAW_KEYS = [
    "continuous_mean",
    "continuous_scale",
    "contact_probability",
    "touchdown_probability",
    "impulse_mean",
    "impulse_scale",
    "tail_q90",
]

for rid in C7T_VAL_IDS:
    d = C7R_DATA[rid]
    vm = np.asarray(
        d["split"]
    ) == "val"
    mask = d["C_precontact"][vm]

    # Frozen baseline.
    control_rows.append({
        "condition": "B9_baseline",
        "run_id": rid,
        "terrain": d["terrain"],
        **metrics(
            C7T_BASE[rid],
            d,
            B9_CALIBRATION,
            mask,
        ),
    })

    # Correctly aligned visual observation.
    control_rows.append({
        "condition": "true",
        "run_id": rid,
        "terrain": d["terrain"],
        **metrics(
            C7T_TRUE[
                C7T_SELECTED_MODE
            ][rid],
            d,
            C7T_CAL[
                C7T_SELECTED_MODE
            ],
            mask,
        ),
    })

    seed = (
        C7T_SEED
        + int(
            hashlib.sha256(
                rid.encode()
            ).hexdigest()[:8],
            16,
        )
        % 100000
    )

    # Existing original anti-cheating controls.
    for cond in (
        "shuffled",
        "run_mean",
    ):
        pp = predict_C7T(
            C7T_MODELS[
                C7T_SELECTED_MODE
            ],
            rid,
            d,
            split="val",
            control=cond,
            seed=seed,
        )

        control_rows.append({
            "condition": cond,
            "run_id": rid,
            "terrain": d["terrain"],
            **metrics(
                pp,
                d,
                C7T_CAL[
                    C7T_SELECTED_MODE
                ],
                mask,
            ),
        })

    # New validity control: normalized vision exactly zero.
    pzero = predict_C7T(
        C7T_MODELS[
            C7T_SELECTED_MODE
        ],
        rid,
        d,
        split="val",
        control="zero",
        seed=seed,
    )

    C7T_ZERO_RESIDUAL_MAX = max(
        C7T_ZERO_RESIDUAL_MAX,
        float(
            pzero[
                "_residual_max_abs"
            ]
        ),
    )

    for k in _RAW_KEYS:
        aa = np.asarray(
            pzero[k],
            dtype=np.float64,
        )
        bb = np.asarray(
            C7T_BASE[rid][k],
            dtype=np.float64,
        )

        C7T_ZERO_RAW_MAX_DIFF = max(
            C7T_ZERO_RAW_MAX_DIFF,
            float(
                np.max(
                    np.abs(aa - bb)
                )
            ),
        )

    # Use B9 calibration for this validity control because the raw
    # predictions are required to equal B9; visual calibration is not
    # part of the architecture equality claim.
    control_rows.append({
        "condition": "zero_vision",
        "run_id": rid,
        "terrain": d["terrain"],
        **metrics(
            pzero,
            d,
            B9_CALIBRATION,
            mask,
        ),
    })


C7T_CONTROLS = pd.DataFrame(
    control_rows
)

print(
    "\nC7T PRE-CONTACT CAUSALITY CONTROLS"
)
display(C7T_CONTROLS)

C7T_SUMMARY = (
    C7T_CONTROLS
    .groupby("condition")
    .mean(numeric_only=True)
)

display(C7T_SUMMARY)

print({
    "zero_visual_residual_max_abs":
        C7T_ZERO_RESIDUAL_MAX,
    "zero_visual_raw_prediction_max_abs_diff_vs_B9":
        C7T_ZERO_RAW_MAX_DIFF,
})

# Tight architecture criterion. Raw denormalized arrays may show tiny AMP
# roundoff across separately executed B9 forward passes, so the raw check
# allows 1e-4 while the learned residual itself must be essentially zero.
C7T_ZERO_ARCH_PASS = bool(
    C7T_ZERO_RESIDUAL_MAX <= 1e-8
)
C7T_ZERO_RAW_PASS = bool(
    C7T_ZERO_RAW_MAX_DIFF <= 1e-4
)

if not C7T_ZERO_ARCH_PASS:
    raise RuntimeError(
        "C7T architecture invalid: zero vision generated "
        "a non-zero learned residual."
    )

if not C7T_ZERO_RAW_PASS:
    raise RuntimeError(
        "C7T architecture invalid: zero-vision predictions "
        "do not reproduce frozen B9 within numerical tolerance."
    )

print(
    "C7T ZERO-VISION FALLBACK CHECK: PASS"
)

# ----------------------------------------------------------------------
# 8. SAME SCIENTIFIC GATE AS C7R/C7S
#
# We do NOT weaken thresholds after observing R/S failures.
# Zero-vision validity is an extra architecture precondition.
# ----------------------------------------------------------------------
def _c7t_v(cond, metric):
    return float(
        C7T_SUMMARY.loc[
            cond, metric
        ]
    )


true_risk = _c7t_v(
    "true", "risk_spearman"
)
base_risk = _c7t_v(
    "B9_baseline", "risk_spearman"
)
shuf_risk = _c7t_v(
    "shuffled", "risk_spearman"
)
runmean_risk = _c7t_v(
    "run_mean", "risk_spearman"
)

true_peak = _c7t_v(
    "true", "force_peak_spearman"
)
base_peak = _c7t_v(
    "B9_baseline",
    "force_peak_spearman",
)

true_tail = _c7t_v(
    "true",
    "force_tail_component_mae",
)
base_tail = _c7t_v(
    "B9_baseline",
    "force_tail_component_mae",
)

C7T_GATE = {
    "selected_mode":
        C7T_SELECTED_MODE,

    "feature_version":
        C7T_FEATURE_VERSION,

    "source_visual_feature_version":
        C7S_FEATURE_VERSION,

    "fresh_test_ids_still_locked":
        C7T_FRESH_IDS,

    "zero_visual_residual_max_abs":
        C7T_ZERO_RESIDUAL_MAX,

    "zero_visual_raw_prediction_max_abs_diff_vs_B9":
        C7T_ZERO_RAW_MAX_DIFF,

    "zero_visual_architecture_invariant":
        C7T_ZERO_ARCH_PASS,

    "zero_visual_reproduces_B9":
        C7T_ZERO_RAW_PASS,

    "validation_precontact_true_risk_spearman":
        true_risk,

    "validation_precontact_B9_risk_spearman":
        base_risk,

    "validation_precontact_shuffled_risk_spearman":
        shuf_risk,

    "validation_precontact_runmean_risk_spearman":
        runmean_risk,

    "validation_precontact_true_force_peak_spearman":
        true_peak,

    "validation_precontact_B9_force_peak_spearman":
        base_peak,

    "validation_precontact_force_tail_improvement_fraction":
        (
            base_tail - true_tail
        ) / max(base_tail, 1e-9),

    # EXACT SAME CAUSAL THRESHOLDS.
    "true_beats_B9_risk":
        bool(
            true_risk
            > base_risk + 0.01
        ),

    "true_beats_shuffled_risk":
        bool(
            true_risk
            > shuf_risk + 0.01
        ),

    "true_beats_runmean_risk":
        bool(
            true_risk
            > runmean_risk + 0.005
        ),

    "tail_or_peak_improves":
        bool(
            true_peak
            > base_peak + 0.01
            or true_tail < base_tail
        ),
}

C7T_GATE["open_fresh_C_test"] = bool(
    C7T_GATE[
        "zero_visual_architecture_invariant"
    ]
    and C7T_GATE[
        "zero_visual_reproduces_B9"
    ]
    and C7T_GATE[
        "true_beats_B9_risk"
    ]
    and C7T_GATE[
        "true_beats_shuffled_risk"
    ]
    and C7T_GATE[
        "true_beats_runmean_risk"
    ]
    and C7T_GATE[
        "tail_or_peak_improves"
    ]
)

print(
    "\n"
    + "=" * 96
    + "\nC7T STRICT VISION-INTERACTION VALIDATION GATE"
    + "\n"
    + "=" * 96
)

print(
    json.dumps(
        C7T_GATE,
        indent=2,
    )
)

# ----------------------------------------------------------------------
# 9. PERSIST EVERYTHING NEEDED FOR AUDIT / RESUME
# ----------------------------------------------------------------------
C7T_VALIDATION.to_csv(
    C7T_TABLE_ROOT
    / "validation_metrics.csv",
    index=False,
)

C7T_SELECTION.to_csv(
    C7T_TABLE_ROOT
    / "mode_selection.csv",
    index=False,
)

C7T_CONTROLS.to_csv(
    C7T_TABLE_ROOT
    / "causality_controls.csv",
    index=False,
)

(
    C7T_TABLE_ROOT / "gate.json"
).write_text(
    json.dumps(
        C7T_GATE,
        indent=2,
    )
)

# Save selected calibration separately without touching any test data.
def _jsonable_t(obj):
    if isinstance(obj, dict):
        return {
            str(k): _jsonable_t(v)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [
            _jsonable_t(v) for v in obj
        ]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(
        obj,
        (np.floating, np.integer),
    ):
        return obj.item()
    return obj

(
    C7T_TABLE_ROOT
    / "selected_validation_calibration.json"
).write_text(
    json.dumps(
        _jsonable_t(
            C7T_CAL[
                C7T_SELECTED_MODE
            ]
        ),
        indent=2,
    )
)

(
    C7T_TABLE_ROOT
    / "selected_model.json"
).write_text(
    json.dumps(
        {
            "selected_mode":
                C7T_SELECTED_MODE,
            "checkpoint":
                C7T_MODELS[
                    C7T_SELECTED_MODE
                ]["checkpoint"],
            "feature_version":
                C7T_FEATURE_VERSION,
            "B9_checkpoint":
                str(B9_CKPT),
            "fresh_tests_scored":
                False,
        },
        indent=2,
    )
)

# ----------------------------------------------------------------------
# 10. STOP HERE — NEVER AUTO-OPEN THE TEST SET
# ----------------------------------------------------------------------
if C7T_GATE["open_fresh_C_test"]:
    print(
        "\nC7T VALIDATION GATE PASSED.\n"
        "Fresh C tests are STILL UNSCORED and were NOT loaded here.\n"
        "STOP NOW and send the complete C7T gate output. "
        "The next step is the one-time C8-C10 evaluation."
    )
else:
    print(
        "\nC7T VALIDATION GATE NOT PASSED.\n"
        "Fresh C tests remain locked. "
        "Do not weaken, bypass, or inspect C8 labels."
    )
