# ======================================================================
# C5 LOW-RAM DROP-IN REPLACEMENT
#
# RUN THIS AFTER C4. SKIP THE ORIGINAL C5 CELL.
# Leave C6-C10 unchanged.
#
# Scientific model is unchanged:
#   frozen B9 dynamics + RGB/depth/RGB-D residual adapters.
# Only the memory implementation changes:
#   - no whole-dataset concatenation
#   - no persistent normalized copies of X/A/Y/RGB/depth
#   - normalization is per batch on GPU
#   - one C model is resident at a time
#   - large obsolete B/C globals are freed after C4
# ======================================================================

import copy, gc, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from IPython.display import display

if "C_DATA" not in globals():
    raise RuntimeError("Run C4 first. This cell REPLACES the original C5 cell.")
if "B9_DENSE" not in globals() or "B9_PREP" not in globals():
    raise RuntimeError("Stage-B B9 runtime/checkpoint is required.")

# ----------------------------------------------------------------------
# 0. Low-RAM settings. Model/scientific objective stays unchanged.
# ----------------------------------------------------------------------
C_BATCH = 128              # lower transient batch RAM; architecture unchanged
C_PRED_BATCH = 256
C_EPOCHS = 20
C_PATIENCE = 5
C_LR = 2e-4
C_WEIGHT_DECAY = 1e-4
C_VIS_HIDDEN = 128
C_RESIDUAL_L2 = 2e-3
C_STREAM_CHUNK = 2048


def _c_ram_gb():
    try:
        import psutil
        p = psutil.Process(os.getpid())
        return p.memory_info().rss / (1024**3)
    except Exception:
        return float("nan")

print(f"C5 low-RAM replacement starting. RSS={_c_ram_gb():.2f} GiB")

# C4 has already copied/filtered every array needed by C6-C10 into C_DATA.
# These objects are not referenced by any later C cell and can consume GBs.
for _name in [
    "C_ALL_DATA",
    "C_FRAME_FEATURES",
    "C_VISUAL_MANIFESTS",
    "B_ALL_DATA",
    "B_DEV_DATA",
    "B6_PREDICTIONS",
    "B5_VAL",
    "B4_NOACTION",
    "B4_DENSE",
    "B9_NOACTION",
]:
    if _name in globals():
        try:
            del globals()[_name]
        except Exception:
            pass

# DINO has finished its job in C4; C5-C10 use cached feature vectors only.
for _name in ["C_RGB_MODEL", "C_RGB_TRANSFORM"]:
    if _name in globals():
        try:
            del globals()[_name]
        except Exception:
            pass

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print(f"After C4 cleanup. RSS={_c_ram_gb():.2f} GiB")

# ----------------------------------------------------------------------
# 1. Streaming train-only visual normalization.
#    Computes exactly the same mean/std concept as old C5 without
#    concatenating ~76k visual rows into another large array.
# ----------------------------------------------------------------------
def _c_stream_feature_stats(data_dict, field, split="train", chunk=C_STREAM_CHUNK):
    dim = None
    total = None
    total2 = None
    n = 0

    for rid, d in data_dict.items():
        idx = np.flatnonzero(d["split"] == split)
        if len(idx) == 0:
            continue
        if dim is None:
            dim = int(d[field].shape[1])
            total = np.zeros(dim, dtype=np.float64)
            total2 = np.zeros(dim, dtype=np.float64)

        for st in range(0, len(idx), chunk):
            sel = idx[st:st+chunk]
            x = np.asarray(d[field][sel], dtype=np.float32)
            total += x.sum(axis=0, dtype=np.float64)
            total2 += np.square(x, dtype=np.float64).sum(axis=0, dtype=np.float64)
            n += len(x)
            del x

    if n == 0 or dim is None:
        raise RuntimeError(f"No {split} samples for {field}")

    mean = total / float(n)
    var = np.maximum(total2 / float(n) - mean * mean, 0.0)
    std = np.sqrt(var)
    std = np.where(std < 1e-5, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32), int(n)


def c_fit_visual_norm_streaming(data_dict):
    rm, rs, nr = _c_stream_feature_stats(data_dict, "X_rgb", "train")
    dm, ds, nd = _c_stream_feature_stats(data_dict, "X_depth", "train")
    if nr != nd:
        raise RuntimeError(f"RGB/depth train count mismatch: {nr} vs {nd}")
    return {
        "rgb_mean": rm,
        "rgb_std": rs,
        "depth_mean": dm,
        "depth_std": ds,
        "train_count": int(nr),
    }

C_VIS_NORM = c_fit_visual_norm_streaming(C_DATA)
print(
    "Streaming visual normalization ready:",
    {"train_count": C_VIS_NORM["train_count"],
     "rgb_dim": len(C_VIS_NORM["rgb_mean"]),
     "depth_dim": len(C_VIS_NORM["depth_mean"])}
)

# ----------------------------------------------------------------------
# 2. Lazy dataset.
#    Holds only small integer lookup tables + tail/event metadata.
#    X_core, dense actions, targets and visual vectors remain in C_DATA.
# ----------------------------------------------------------------------
class CStreamingDataset(Dataset):
    def __init__(self, data_dict, split, prep, balanced=False, chunk=C_STREAM_CHUNK):
        self.data_dict = data_dict
        self.prep = prep
        self.split = split
        self.run_keys = []
        self.run_of_parts = []
        self.local_of_parts = []
        self.ft_parts = []
        self.et_parts = []
        self.tail_parts = []
        self.sw_parts = []

        # Counts needed for exactly the same terrain balancing principle as B9.
        terrain_counts = {}
        run_records = []
        for rid, d in data_dict.items():
            idx = np.flatnonzero(d["split"] == split).astype(np.int64)
            if len(idx) == 0:
                continue
            terrain = str(d["terrain"])
            terrain_counts[terrain] = terrain_counts.get(terrain, 0) + len(idx)
            run_records.append((rid, d, idx, terrain))

        if not run_records:
            raise RuntimeError(f"No C samples for split={split}")

        total_n = int(sum(len(r[2]) for r in run_records))
        nterr = len(terrain_counts)
        terrain_factor = {
            t: total_n / (nterr * c)
            for t, c in terrain_counts.items()
        }

        for rix, (rid, d, idx, terrain) in enumerate(run_records):
            self.run_keys.append(rid)
            self.run_of_parts.append(np.full(len(idx), rix, dtype=np.int16))
            self.local_of_parts.append(idx)

            ft_run = np.empty((len(idx), B_H, 4), dtype=np.uint8)
            et_run = np.empty((len(idx), B_H, 4), dtype=np.uint8)
            tail_run = np.empty((len(idx), 4), dtype=np.float32)
            sw_run = np.empty(len(idx), dtype=np.float32)

            for st in range(0, len(idx), chunk):
                ed = min(st + chunk, len(idx))
                sel = idx[st:ed]

                step = np.asarray(
                    d["Y_cont"][sel], dtype=np.float32
                ).reshape(-1, B_H, B_STEP_WIDTH)
                td = np.asarray(d["Y_touchdown"][sel], dtype=np.float32)

                force = step[:, :, B_STEP_LAYOUT["force_odom"]].reshape(-1, B_H, 4, 3)
                earr = step[:, :, B_STEP_LAYOUT["E"]].reshape(-1, B_H, 4, 3)

                ft = (np.linalg.norm(force, axis=-1) > prep["force95"])
                et = (np.linalg.norm(earr, axis=-1) > prep["E95"])
                tail = np.log1p(
                    np.maximum(
                        _b_tail_raw(step, prep["body_med"], prep["body_mad"]),
                        0.0,
                    )
                ).astype(np.float32)

                sw = (
                    1.0
                    + 2.0 * ft.max(axis=(1, 2)).astype(np.float32)
                    + 2.0 * et.max(axis=(1, 2)).astype(np.float32)
                    + 4.0 * td.max(axis=(1, 2)).astype(np.float32)
                )
                if balanced:
                    sw *= float(terrain_factor[terrain])

                ft_run[st:ed] = ft.astype(np.uint8)
                et_run[st:ed] = et.astype(np.uint8)
                tail_run[st:ed] = tail
                sw_run[st:ed] = np.clip(sw, 1.0, 12.0).astype(np.float32)

                del step, td, force, earr, ft, et, tail, sw

            self.ft_parts.append(ft_run)
            self.et_parts.append(et_run)
            self.tail_parts.append(tail_run)
            self.sw_parts.append(sw_run)

        self.run_of = np.concatenate(self.run_of_parts, axis=0)
        self.local_of = np.concatenate(self.local_of_parts, axis=0)
        self.force_tail = np.concatenate(self.ft_parts, axis=0)
        self.E_tail = np.concatenate(self.et_parts, axis=0)
        self.tail_log = np.concatenate(self.tail_parts, axis=0)
        self.sample_weight = np.concatenate(self.sw_parts, axis=0)

        # Drop construction-only lists.
        self.run_of_parts = None
        self.local_of_parts = None
        self.ft_parts = None
        self.et_parts = None
        self.tail_parts = None
        self.sw_parts = None

    def __len__(self):
        return len(self.local_of)

    def __getitem__(self, i):
        rix = int(self.run_of[i])
        j = int(self.local_of[i])
        d = self.data_dict[self.run_keys[rix]]

        # Return RAW values. Batch normalization happens after collation,
        # so no full normalized dataset is ever materialized.
        x = np.asarray(d["X_core"][j], dtype=np.float32)
        a = np.asarray(d["X_action_dense"][j], dtype=np.float32)
        y = np.asarray(d["Y_cont"][j], dtype=np.float32).reshape(B_H, B_STEP_WIDTH)
        c = np.asarray(d["Y_contact"][j], dtype=np.float32)
        td = np.asarray(d["Y_touchdown"][j], dtype=np.float32)
        imp = np.asarray(d["Y_impulse"][j], dtype=np.float32)
        rgb = np.asarray(d["X_rgb"][j], dtype=np.float32)
        dep = np.asarray(d["X_depth"][j], dtype=np.float32)

        return (
            torch.from_numpy(x),
            torch.from_numpy(a),
            torch.from_numpy(y),
            torch.from_numpy(c),
            torch.from_numpy(td),
            torch.from_numpy(imp),
            torch.from_numpy(self.force_tail[i].astype(np.float32)),
            torch.from_numpy(self.E_tail[i].astype(np.float32)),
            torch.from_numpy(self.tail_log[i]),
            torch.from_numpy(rgb),
            torch.from_numpy(dep),
        )

# ----------------------------------------------------------------------
# 3. Device-side normalization constants + per-batch normalization.
# ----------------------------------------------------------------------
def _c_t(x):
    return torch.as_tensor(x, dtype=torch.float32, device=B_DEVICE)

C_NORM_T = {
    "x_mean": _c_t(B9_PREP["x_mean"]),
    "x_std": _c_t(B9_PREP["x_std"]),
    "action_mean": _c_t(B9_PREP["action_mean"]),
    "action_std": _c_t(B9_PREP["action_std"]),
    "y_mean": _c_t(B9_PREP["y_mean"]),
    "y_std": _c_t(B9_PREP["y_std"]),
    "imp_mean": _c_t(B9_PREP["imp_mean"]),
    "imp_std": _c_t(B9_PREP["imp_std"]),
    "rgb_mean": _c_t(C_VIS_NORM["rgb_mean"]),
    "rgb_std": _c_t(C_VIS_NORM["rgb_std"]),
    "depth_mean": _c_t(C_VIS_NORM["depth_mean"]),
    "depth_std": _c_t(C_VIS_NORM["depth_std"]),
}


def c_prepare_batch(raw_batch):
    # No pin_memory: prevents another large host-side staging pool.
    raw_batch = tuple(t.to(B_DEVICE, non_blocking=False) for t in raw_batch)
    x, a, y, c, td, imp, ft, et, tail, rgb, dep = raw_batch

    x = (x - C_NORM_T["x_mean"]) / C_NORM_T["x_std"]
    a = (
        (a - C_NORM_T["action_mean"][None, None, :])
        / C_NORM_T["action_std"][None, None, :]
    )
    y = (
        (y - C_NORM_T["y_mean"][None, None, :])
        / C_NORM_T["y_std"][None, None, :]
    )
    imp = (
        (imp - C_NORM_T["imp_mean"][None, None, None, :])
        / C_NORM_T["imp_std"][None, None, None, :]
    )
    rgb = (rgb - C_NORM_T["rgb_mean"][None, :]) / C_NORM_T["rgb_std"][None, :]
    dep = (dep - C_NORM_T["depth_mean"][None, :]) / C_NORM_T["depth_std"][None, :]

    base_batch = (x, a, y, c, td, imp, ft, et, tail)
    return base_batch, rgb, dep

# ----------------------------------------------------------------------
# 4. Same C visual residual architecture as original C5.
#    Difference: receives a freshly loaded CPU B9 base; no deepcopy needed.
# ----------------------------------------------------------------------
class CVisualResidualModel(nn.Module):
    def __init__(self, base_model, rgb_dim, depth_dim, mode):
        super().__init__()
        if mode not in {"rgb", "depth", "rgbd"}:
            raise ValueError(mode)
        self.mode = mode
        self.base = base_model.eval()
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.rgb = nn.Sequential(
            nn.Linear(rgb_dim, C_VIS_HIDDEN),
            nn.GELU(),
            nn.LayerNorm(C_VIS_HIDDEN),
            nn.Linear(C_VIS_HIDDEN, C_VIS_HIDDEN),
            nn.GELU(),
            nn.LayerNorm(C_VIS_HIDDEN),
        )
        self.depth = nn.Sequential(
            nn.Linear(depth_dim, C_VIS_HIDDEN),
            nn.GELU(),
            nn.LayerNorm(C_VIS_HIDDEN),
            nn.Linear(C_VIS_HIDDEN, C_VIS_HIDDEN),
            nn.GELU(),
            nn.LayerNorm(C_VIS_HIDDEN),
        )

        vis_dim = C_VIS_HIDDEN if mode in {"rgb", "depth"} else 2*C_VIS_HIDDEN

        self.h_adapter = nn.Sequential(
            nn.Linear(B_MODEL_DIM + vis_dim, B_MODEL_DIM),
            nn.GELU(),
            nn.LayerNorm(B_MODEL_DIM),
            nn.Dropout(0.05),
            nn.Linear(B_MODEL_DIM, B_MODEL_DIM),
            nn.GELU(),
            nn.LayerNorm(B_MODEL_DIM),
        )
        self.seq_adapter = nn.Sequential(
            nn.Linear(B_MODEL_DIM + B_ACTION_HIDDEN + vis_dim, B_MODEL_DIM),
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

        # Preserve original zero-residual initialization exactly.
        for layer in [
            self.d_mean, self.d_log_scale,
            self.d_contact, self.d_touchdown,
            self.d_impulse_mean, self.d_impulse_log_scale,
            self.d_tail,
        ]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _visual(self, rgb, depth):
        if self.mode == "rgb":
            return self.rgb(rgb)
        if self.mode == "depth":
            return self.depth(depth)
        return torch.cat([self.rgb(rgb), self.depth(depth)], dim=-1)

    def _base_latents_and_outputs(self, x, a):
        self.base.eval()
        with torch.no_grad():
            h0 = self.base.history(x)
            ah = self.base.action(a, B_ACTION_GATHER.to(a.device))
            a_final = ah[:, -1]
            h = h0[:, None, :].expand(-1, B_H, -1)
            he = self.base.horizon_embed[None].expand(len(x), -1, -1)
            z = self.base.fuse(torch.cat([h, ah, he], dim=-1))

            mean = self.base.mean_head(z)
            log_scale = torch.clamp(self.base.log_scale_head(z), -5.5, 2.8)
            contact = self.base.contact_head(z)
            touchdown = self.base.touchdown_head(z)
            imp_mean = self.base.impulse_mean_head(z).reshape(-1, B_H, 4, 3)
            imp_log = torch.clamp(
                self.base.impulse_log_scale_head(z).reshape(-1, B_H, 4, 3),
                -5.5, 3.0,
            )
            tail = self.base.tail_q90_head(torch.cat([h0, a_final], dim=-1))

        return h0, ah, a_final, z, {
            "mean_z": mean,
            "log_scale_z": log_scale,
            "contact_logits": contact,
            "touchdown_logits": touchdown,
            "impulse_mean_z": imp_mean,
            "impulse_log_scale_z": imp_log,
            "tail_q90_log": tail,
        }

    def forward(self, x, a, rgb, depth):
        h0, ah, a_final, z, base = self._base_latents_and_outputs(x, a)
        v = self._visual(rgb, depth)
        vh = v[:, None, :].expand(-1, B_H, -1)
        rz = self.h_adapter(torch.cat([z, vh], dim=-1))
        seq = self.seq_adapter(torch.cat([h0, a_final, v], dim=-1))

        dm = self.d_mean(rz)
        dls = self.d_log_scale(rz)
        dc = self.d_contact(rz)
        dtd = self.d_touchdown(rz)
        dimp = self.d_impulse_mean(rz).reshape(-1, B_H, 4, 3)
        dil = self.d_impulse_log_scale(rz).reshape(-1, B_H, 4, 3)
        dtail = self.d_tail(seq)

        return {
            "mean_z": base["mean_z"] + dm,
            "log_scale_z": torch.clamp(base["log_scale_z"] + dls, -5.5, 2.8),
            "contact_logits": base["contact_logits"] + dc,
            "touchdown_logits": base["touchdown_logits"] + dtd,
            "impulse_mean_z": base["impulse_mean_z"] + dimp,
            "impulse_log_scale_z": torch.clamp(
                base["impulse_log_scale_z"] + dil, -5.5, 3.0
            ),
            "tail_q90_log": base["tail_q90_log"] + dtail,
            "_residual_l2": (
                dm.square().mean()
                + 0.25*dimp.square().mean()
                + 0.25*dtail.square().mean()
            ),
        }

# ----------------------------------------------------------------------
# 5. Load frozen B9 base from checkpoint, not from a second live dataset/model.
# ----------------------------------------------------------------------
def _c_load_b9_base_cpu():
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
    del ck
    return base

# Temporarily move the already-live B9 copy off GPU while training C adapters.
_B9_WAS_DEVICE = next(B9_DENSE["model"].parameters()).device
B9_DENSE["model"].to("cpu").eval()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ----------------------------------------------------------------------
# 6. Low-RAM training: one visual mode/model at a time.
# ----------------------------------------------------------------------
def c_train_one_mode(mode):
    tr = CStreamingDataset(C_DATA, "train", B9_PREP, balanced=True)
    va = CStreamingDataset(C_DATA, "val", B9_PREP, balanced=False)

    sampler = WeightedRandomSampler(
        torch.as_tensor(tr.sample_weight, dtype=torch.double),
        num_samples=len(tr),
        replacement=True,
    )
    tr_loader = DataLoader(
        tr,
        batch_size=C_BATCH,
        sampler=sampler,
        num_workers=0,
        pin_memory=False,
    )
    va_loader = DataLoader(
        va,
        batch_size=C_BATCH,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    base = _c_load_b9_base_cpu()
    model = CVisualResidualModel(
        base,
        rgb_dim=len(C_VIS_NORM["rgb_mean"]),
        depth_dim=len(C_VIS_NORM["depth_mean"]),
        mode=mode,
    ).to(B_DEVICE)
    del base

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        trainable,
        lr=C_LR,
        weight_decay=C_WEIGHT_DECAY,
    )

    best = float("inf")
    best_state = None
    bad = 0
    hist = []

    @torch.no_grad()
    def eval_loss():
        model.eval()
        tot = 0.0
        n = 0
        for raw in va_loader:
            base_batch, rgb, depth = c_prepare_batch(raw)
            out = model(base_batch[0], base_batch[1], rgb, depth)
            loss = _b_loss(out, base_batch, B9_PREP)["total"]
            loss = loss + C_RESIDUAL_L2*out["_residual_l2"]
            tot += float(loss) * len(base_batch[0])
            n += len(base_batch[0])
            del raw, base_batch, rgb, depth, out, loss
        return tot / max(n, 1)

    for ep in range(1, C_EPOCHS + 1):
        model.train()
        tot = 0.0
        n = 0

        for raw in tr_loader:
            base_batch, rgb, depth = c_prepare_batch(raw)

            opt.zero_grad(set_to_none=True)
            out = model(base_batch[0], base_batch[1], rgb, depth)
            losses = _b_loss(out, base_batch, B9_PREP)
            loss = losses["total"] + C_RESIDUAL_L2*out["_residual_l2"]

            if not torch.isfinite(loss):
                raise RuntimeError(f"C-{mode}: nonfinite training loss")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            opt.step()

            tot += float(loss.detach()) * len(base_batch[0])
            n += len(base_batch[0])

            del raw, base_batch, rgb, depth, out, losses, loss

        tr_loss = tot / max(n, 1)
        va_loss = eval_loss()
        hist.append({
            "mode": mode,
            "epoch": ep,
            "train_loss": tr_loss,
            "val_loss": va_loss,
        })
        print(
            f"C-{mode:5s} ep={ep:02d} "
            f"train={tr_loss:.5f} val={va_loss:.5f} "
            f"RSS={_c_ram_gb():.2f}GiB"
        )

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

        if bad >= C_PATIENCE:
            print(f"C-{mode}: early stop")
            break

    if best_state is None:
        raise RuntimeError(f"C-{mode}: no valid checkpoint")

    ckpt = {
        "mode": mode,
        "adapter_state_dict": best_state,
        "best_val_loss": float(best),
        "rgb_backbone": C_RGB_BACKBONE,
        "vis_norm": C_VIS_NORM,
        "B9_checkpoint": B9_DENSE["checkpoint"],
        "split_manifest": str(C_SPLIT_PATH),
        "memory_impl": "streaming_batch_normalization_v1",
    }
    path = C_MODEL_ROOT / f"C5_{mode}_adapter.pt"
    torch.save(ckpt, path)

    result = {
        "mode": mode,
        "best_val_loss": float(best),
        "history": pd.DataFrame(hist),
        "checkpoint": str(path),
        "vis_norm": C_VIS_NORM,
        "prep": B9_PREP,
        # Intentionally no live `model`: predict_C below loads one mode lazily.
    }

    del opt, trainable, model, tr_loader, va_loader, sampler, tr, va, best_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"C-{mode}: checkpoint saved; resident training model released. RSS={_c_ram_gb():.2f} GiB")
    return result

# ----------------------------------------------------------------------
# 7. One-model-at-a-time lazy inference. Compatible with C6-C10.
# ----------------------------------------------------------------------
_C_ACTIVE_MODEL = {"checkpoint": None, "model": None}


def _c_release_active_model():
    old = _C_ACTIVE_MODEL.get("model")
    if old is not None:
        try:
            old.to("cpu")
        except Exception:
            pass
    _C_ACTIVE_MODEL["checkpoint"] = None
    _C_ACTIVE_MODEL["model"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _c_get_model(bundle):
    ckpt_path = str(bundle["checkpoint"])
    if (
        _C_ACTIVE_MODEL.get("checkpoint") == ckpt_path
        and _C_ACTIVE_MODEL.get("model") is not None
    ):
        return _C_ACTIVE_MODEL["model"]

    _c_release_active_model()

    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    base = _c_load_b9_base_cpu()
    model = CVisualResidualModel(
        base,
        rgb_dim=len(C_VIS_NORM["rgb_mean"]),
        depth_dim=len(C_VIS_NORM["depth_mean"]),
        mode=saved["mode"],
    )
    del base

    missing, unexpected = model.load_state_dict(
        saved["adapter_state_dict"], strict=False
    )
    bad_missing = [k for k in missing if not k.startswith("base.")]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"Adapter checkpoint mismatch. missing={bad_missing}, unexpected={unexpected}"
        )

    model = model.to(B_DEVICE).eval()
    _C_ACTIVE_MODEL["checkpoint"] = ckpt_path
    _C_ACTIVE_MODEL["model"] = model
    del saved
    return model


@torch.no_grad()
def predict_C(bundle, d, split="test", override_rgb=None, override_depth=None,
              batch_size=C_PRED_BATCH):
    idx = np.flatnonzero(d["split"] == split)
    if len(idx) == 0:
        raise RuntimeError(f"No C samples for split={split}")

    if override_rgb is not None and len(override_rgb) != len(idx):
        raise RuntimeError("override_rgb length mismatch")
    if override_depth is not None and len(override_depth) != len(idx):
        raise RuntimeError("override_depth length mismatch")

    outputs = {
        "mean_z": [], "scale_z": [],
        "contact": [], "touchdown": [],
        "impulse_mean_z": [], "impulse_scale_z": [],
        "tail_q90_log": [],
        "truth": [], "contact_truth": [],
        "touchdown_truth": [], "impulse_truth": [],
        "origins": [],
    }

    model = _c_get_model(bundle)

    for st in range(0, len(idx), batch_size):
        ed = min(st + batch_size, len(idx))
        sel = idx[st:ed]

        x = torch.from_numpy(np.asarray(d["X_core"][sel], np.float32)).to(B_DEVICE)
        a = torch.from_numpy(np.asarray(d["X_action_dense"][sel], np.float32)).to(B_DEVICE)

        if override_rgb is None:
            r_np = np.asarray(d["X_rgb"][sel], np.float32)
        else:
            r_np = np.asarray(override_rgb[st:ed], np.float32)
        if override_depth is None:
            dp_np = np.asarray(d["X_depth"][sel], np.float32)
        else:
            dp_np = np.asarray(override_depth[st:ed], np.float32)

        rgb = torch.from_numpy(r_np).to(B_DEVICE)
        dep = torch.from_numpy(dp_np).to(B_DEVICE)

        x = (x - C_NORM_T["x_mean"]) / C_NORM_T["x_std"]
        a = (
            (a - C_NORM_T["action_mean"][None, None, :])
            / C_NORM_T["action_std"][None, None, :]
        )
        rgb = (rgb - C_NORM_T["rgb_mean"][None, :]) / C_NORM_T["rgb_std"][None, :]
        dep = (dep - C_NORM_T["depth_mean"][None, :]) / C_NORM_T["depth_std"][None, :]

        out = model(x, a, rgb, dep)
        outputs["mean_z"].append(out["mean_z"].cpu().numpy())
        outputs["scale_z"].append(np.exp(out["log_scale_z"].cpu().numpy()))
        outputs["contact"].append(torch.sigmoid(out["contact_logits"]).cpu().numpy())
        outputs["touchdown"].append(torch.sigmoid(out["touchdown_logits"]).cpu().numpy())
        outputs["impulse_mean_z"].append(out["impulse_mean_z"].cpu().numpy())
        outputs["impulse_scale_z"].append(np.exp(out["impulse_log_scale_z"].cpu().numpy()))
        outputs["tail_q90_log"].append(out["tail_q90_log"].cpu().numpy())

        outputs["truth"].append(np.asarray(d["Y_cont"][sel], np.float32))
        outputs["contact_truth"].append(np.asarray(d["Y_contact"][sel], np.float32))
        outputs["touchdown_truth"].append(np.asarray(d["Y_touchdown"][sel], np.float32))
        outputs["impulse_truth"].append(np.asarray(d["Y_impulse"][sel], np.float32))
        outputs["origins"].append(np.asarray(d["origins"][sel], np.int64))

        del x, a, rgb, dep, r_np, dp_np, out

    out = {
        k: np.concatenate(v, axis=0)
        for k, v in outputs.items()
    }

    mean = (
        out["mean_z"] * B9_PREP["y_std"][None, None, :]
        + B9_PREP["y_mean"][None, None, :]
    )
    scale = out["scale_z"] * B9_PREP["y_std"][None, None, :]

    imp_mean = (
        out["impulse_mean_z"] * B9_PREP["imp_std"][None, None, None, :]
        + B9_PREP["imp_mean"][None, None, None, :]
    )
    imp_scale = (
        out["impulse_scale_z"] * B9_PREP["imp_std"][None, None, None, :]
    )
    tail_q90 = np.maximum(np.expm1(out["tail_q90_log"]), 0.0)

    result = {
        "continuous_mean": mean.astype(np.float32),
        "continuous_scale": scale.astype(np.float32),
        "contact_probability": out["contact"].astype(np.float32),
        "touchdown_probability": out["touchdown"].astype(np.float32),
        "impulse_mean": imp_mean.astype(np.float32),
        "impulse_scale": imp_scale.astype(np.float32),
        "tail_q90": tail_q90.astype(np.float32),
        "truth": out["truth"].astype(np.float32),
        "contact_truth": out["contact_truth"].astype(np.float32),
        "touchdown_truth": out["touchdown_truth"].astype(np.float32),
        "impulse_truth": out["impulse_truth"].astype(np.float32),
        "origins": out["origins"],
    }

    del out, mean, scale, imp_mean, imp_scale, tail_q90
    gc.collect()
    return result

# ----------------------------------------------------------------------
# 8. Train the same three controlled visual ablations sequentially.
# ----------------------------------------------------------------------
print("Training controlled visual residual adapters — LOW-RAM streaming path...")
C5_MODELS = {}

try:
    for mode in ("rgb", "depth", "rgbd"):
        print("\n" + "="*80)
        print(f"C5 LOW-RAM MODE: {mode}")
        print("="*80)
        C5_MODELS[mode] = c_train_one_mode(mode)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
finally:
    # C6 uses predict_B(B9_DENSE,...), so restore the frozen B9 model to device.
    B9_DENSE["model"].to(B_DEVICE).eval()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

C5_TRAINING_SUMMARY = pd.DataFrame([
    {
        "mode": mode,
        "best_val_loss": bundle["best_val_loss"],
        "checkpoint": bundle["checkpoint"],
        "memory_impl": "streaming_batch_normalization_v1",
    }
    for mode, bundle in C5_MODELS.items()
])

display(C5_TRAINING_SUMMARY)
C5_TRAINING_SUMMARY.to_csv(
    C_TABLE_ROOT / "C5_training_summary.csv", index=False
)

print(f"\nC5 LOW-RAM complete. RSS={_c_ram_gb():.2f} GiB")
print("Proceed to the ORIGINAL C6 cell. Do not run the old C5 cell.")
