####################################################################################################
# C6
####################################################################################################


# ======================================================================
# C6 — VALIDATION-ONLY CALIBRATION + RGB / DEPTH / RGB-D COMPARISON
#
# Select one visual mode using ONLY the C validation trajectories.
# C test trajectories remain unscored.
# ======================================================================

import json, numpy as np, pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from IPython.display import display

if "C5_MODELS" not in globals():
    raise RuntimeError("Run C5 first.")

def c_concat_predictions(bundle, data_dict, split):
    outs = []
    rids = []
    terrains = []
    for rid, d in data_dict.items():
        if not np.any(d["split"] == split):
            continue
        p = predict_C(bundle, d, split=split)
        outs.append(p)
        rids.extend([rid]*len(p["truth"]))
        terrains.extend([d["terrain"]]*len(p["truth"]))
    if not outs:
        raise RuntimeError(f"No C predictions for split={split}")
    keys = outs[0].keys()
    merged = {}
    for k in keys:
        if k == "origins":
            merged[k] = np.concatenate([o[k] for o in outs], axis=0)
        else:
            merged[k] = np.concatenate([o[k] for o in outs], axis=0)
    merged["run_ids"] = np.asarray(rids)
    merged["terrains"] = np.asarray(terrains)
    return merged

def fit_C_calibration(bundle, data_dict):
    val = c_concat_predictions(bundle, data_dict, "val")
    truth = _b_reshape_cont(val["truth"])
    mu = np.asarray(val["continuous_mean"], float)
    scale = np.maximum(np.asarray(val["continuous_scale"], float), 1e-6)

    scale_factor = {}
    cov_rows = []
    for group, sl in B_STEP_LAYOUT.items():
        err = np.abs(truth[:,:,sl] - mu[:,:,sl])
        sc = scale[:,:,sl]
        factor = float(np.quantile(err/sc, .95))
        factor = max(factor, 1e-3)
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

    tail_actual = _b_tail_raw(
        truth, B9_PREP["body_med"], B9_PREP["body_mad"]
    )
    tail_pred = np.asarray(val["tail_q90"], float)
    tail_add = {}
    tail_rows = []
    for j, name in enumerate(TAIL_NAMES):
        add = float(max(
            0.0,
            np.quantile(tail_actual[:,j] - tail_pred[:,j], .90),
        ))
        tail_add[name] = add
        tail_rows.append({
            "metric": name,
            "raw_q90_coverage": float(
                np.mean(tail_actual[:,j] <= tail_pred[:,j])
            ),
            "calibrated_q90_coverage": float(
                np.mean(tail_actual[:,j] <= tail_pred[:,j] + add)
            ),
            "additive_calibration": add,
        })

    y = val["touchdown_truth"].reshape(-1).astype(int)
    p = val["touchdown_probability"].reshape(-1)
    best = None
    for th in np.linspace(.05, .95, 37):
        yp = (p >= th).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(
            y, yp, average="binary", zero_division=0
        )
        row = (float(f1), float(pr), float(rc), float(th))
        if best is None or row[0] > best[0]:
            best = row

    # Keep the exact Stage-B physical references so risk has identical meaning.
    refs = dict(B9_CALIBRATION["risk_reference"])

    return {
        "scale_factor": scale_factor,
        "impulse_scale_factor": impulse_factor,
        "tail_add": tail_add,
        "touchdown_threshold": best[3],
        "touchdown_val_precision": best[1],
        "touchdown_val_recall": best[2],
        "touchdown_val_f1": best[0],
        "risk_reference": refs,
        "coverage_table": pd.DataFrame(cov_rows),
        "tail_table": pd.DataFrame(tail_rows),
    }

def c_subset_prediction(pred, mask):
    out = {}
    n = len(mask)
    for k, v in pred.items():
        if isinstance(v, np.ndarray) and len(v) == n:
            out[k] = v[mask]
        else:
            out[k] = v
    return out

def c_metrics_from_prediction(pred, d, cal, mask=None):
    n = len(pred["truth"])
    if mask is None:
        mask = np.ones(n, dtype=bool)
    mask = np.asarray(mask, bool)

    if mask.sum() < 20:
        return {
            "samples": int(mask.sum()),
            "risk_spearman": np.nan,
            "top10_overlap": np.nan,
            "force_peak_spearman": np.nan,
            "force_tail_component_mae": np.nan,
            "E_tail_component_mae": np.nan,
            "touchdown_f1": np.nan,
        }

    p = c_subset_prediction(pred, mask)
    truth = _b_reshape_cont(p["truth"])
    mu = p["continuous_mean"]

    ftrue = _b_group(truth, "force_odom").reshape(-1,B_H,4,3)
    fpred = _b_group(mu, "force_odom").reshape(-1,B_H,4,3)
    Etrue = _b_group(truth, "E").reshape(-1,B_H,4,3)
    Epred = _b_group(mu, "E").reshape(-1,B_H,4,3)

    fnorm = np.linalg.norm(ftrue, axis=-1)
    Enorm = np.linalg.norm(Etrue, axis=-1)
    fmask = fnorm > B9_PREP["force95"]
    Emask = Enorm > B9_PREP["E95"]

    fmae = float(np.abs(fpred-ftrue)[fmask].mean()) if np.any(fmask) else np.nan
    Emae = float(np.abs(Epred-Etrue)[Emask].mean()) if np.any(Emask) else np.nan

    actual_peak = np.max(fnorm, axis=(1,2))
    pred_peak = np.max(np.linalg.norm(fpred,axis=-1), axis=(1,2))
    force_spear = float(pd.Series(pred_peak).corr(
        pd.Series(actual_peak), method="spearman"
    ))

    y = p["touchdown_truth"].reshape(-1).astype(int)
    yp = (
        p["touchdown_probability"].reshape(-1)
        >= cal["touchdown_threshold"]
    ).astype(int)
    _, _, f1, _ = precision_recall_fscore_support(
        y, yp, average="binary", zero_division=0
    )

    prisk = B9_predicted_risk(p, B9_PREP, cal)
    arisk = B9_actual_risk(
        truth,
        p["contact_truth"],
        p["touchdown_truth"],
        p["impulse_truth"],
        B9_PREP,
        cal,
    )

    spear = float(pd.Series(prisk["physical_risk"]).corr(
        pd.Series(arisk["physical_risk"]), method="spearman"
    ))
    k = max(1, int(round(.10*len(truth))))
    ptop = set(np.argpartition(prisk["physical_risk"], -k)[-k:])
    atop = set(np.argpartition(arisk["physical_risk"], -k)[-k:])
    overlap = len(ptop & atop) / float(k)

    return {
        "samples": int(mask.sum()),
        "risk_spearman": spear,
        "top10_overlap": overlap,
        "force_peak_spearman": force_spear,
        "force_tail_component_mae": fmae,
        "E_tail_component_mae": Emae,
        "touchdown_f1": float(f1),
    }

# Baseline B9 validation predictions per run.
C6_CALIBRATIONS = {}
rows = []

for mode, bundle in C5_MODELS.items():
    cal = fit_C_calibration(bundle, C_DATA)
    C6_CALIBRATIONS[mode] = cal

    for rid, d in C_DATA.items():
        if not np.any(d["split"] == "val"):
            continue
        pred = predict_C(bundle, d, "val")
        m_all = c_metrics_from_prediction(pred, d, cal)
        m_pre = c_metrics_from_prediction(
            pred, d, cal,
            mask=d["C_precontact"][d["split"]=="val"],
        )
        rows.append({
            "mode": mode,
            "run_id": rid,
            "terrain": d["terrain"],
            "subset": "all",
            **m_all,
        })
        rows.append({
            "mode": mode,
            "run_id": rid,
            "terrain": d["terrain"],
            "subset": "precontact",
            **m_pre,
        })

# B9 baseline on the exact same val trajectories.
for rid, d in C_DATA.items():
    if not np.any(d["split"] == "val"):
        continue
    pred = predict_B(B9_DENSE, d, "val")
    m_all = c_metrics_from_prediction(pred, d, B9_CALIBRATION)
    m_pre = c_metrics_from_prediction(
        pred, d, B9_CALIBRATION,
        mask=d["C_precontact"][d["split"]=="val"],
    )
    rows.append({
        "mode": "B9_baseline",
        "run_id": rid,
        "terrain": d["terrain"],
        "subset": "all",
        **m_all,
    })
    rows.append({
        "mode": "B9_baseline",
        "run_id": rid,
        "terrain": d["terrain"],
        "subset": "precontact",
        **m_pre,
    })

C6_VALIDATION = pd.DataFrame(rows)
display(C6_VALIDATION)

# Select visual mode using a pre-specified validation-only score.
_base_pre_rows = C6_VALIDATION[
    (C6_VALIDATION["mode"] == "B9_baseline")
    & (C6_VALIDATION["subset"] == "precontact")
]
if _base_pre_rows.empty:
    raise RuntimeError("C6 baseline precontact validation rows are missing.")
base_pre = _base_pre_rows.mean(numeric_only=True)

select_rows = []
for mode in ("rgb", "depth", "rgbd"):
    _mode_rows = C6_VALIDATION[
        (C6_VALIDATION["mode"] == mode)
        & (C6_VALIDATION["subset"] == "precontact")
    ]
    if _mode_rows.empty:
        raise RuntimeError(f"C6 precontact validation rows are missing for mode={mode}.")
    g = _mode_rows.mean(numeric_only=True)

    force_gain = (
        base_pre["force_tail_component_mae"] - g["force_tail_component_mae"]
    ) / max(base_pre["force_tail_component_mae"], 1e-9)
    E_gain = (
        base_pre["E_tail_component_mae"] - g["E_tail_component_mae"]
    ) / max(base_pre["E_tail_component_mae"], 1e-9)

    score = (
        .35 * g["risk_spearman"]
        + .25 * g["force_peak_spearman"]
        + .20 * force_gain
        + .10 * E_gain
        + .10 * g["touchdown_f1"]
    )
    select_rows.append({
        "mode": mode,
        "validation_selection_score": float(score),
        "precontact_risk_spearman": float(g["risk_spearman"]),
        "precontact_force_peak_spearman": float(g["force_peak_spearman"]),
        "precontact_force_tail_gain_vs_B9": float(force_gain),
        "precontact_E_tail_gain_vs_B9": float(E_gain),
        "precontact_touchdown_f1": float(g["touchdown_f1"]),
    })

C6_MODE_SELECTION = pd.DataFrame(select_rows).sort_values(
    "validation_selection_score", ascending=False
)
C_SELECTED_MODE = str(C6_MODE_SELECTION.iloc[0]["mode"])
C_SELECTED_MODEL = C5_MODELS[C_SELECTED_MODE]
C_SELECTED_CAL = C6_CALIBRATIONS[C_SELECTED_MODE]

print("\nC6 mode selection:")
display(C6_MODE_SELECTION)
print("SELECTED:", C_SELECTED_MODE)

C6_VALIDATION.to_csv(C_TABLE_ROOT/"C6_validation_metrics.csv", index=False)
C6_MODE_SELECTION.to_csv(C_TABLE_ROOT/"C6_mode_selection.csv", index=False)

serial = {
    k:v for k,v in C_SELECTED_CAL.items()
    if k not in {"coverage_table","tail_table"}
}
(C_TABLE_ROOT/"C6_selected_calibration.json").write_text(
    json.dumps(serial, indent=2)
)
C_SELECTED_CAL["coverage_table"].to_csv(
    C_TABLE_ROOT/"C6_selected_coverage.csv", index=False
)
C_SELECTED_CAL["tail_table"].to_csv(
    C_TABLE_ROOT/"C6_selected_tail_calibration.csv", index=False
)

print("\nC6 complete. Fresh C test trajectories have not been scored.")


####################################################################################################
# C7
####################################################################################################


# ======================================================================
# C7 — VALIDATION CAUSALITY CONTROLS
#
# Three visual conditions on the same validation origins:
#   1) true time-local visual feature
#   2) shuffled visual feature within the same run
#   3) run-mean visual feature
#
# If true vision does not beat these controls on PRECONTACT events,
# do not open the fresh C test trajectories.
# ======================================================================

import numpy as np, pandas as pd, json, hashlib
from IPython.display import display

if "C_SELECTED_MODEL" not in globals():
    raise RuntimeError("Run C6 first.")

def _c_override_for_run(d, split, kind, seed):
    m = d["split"] == split
    R = np.asarray(d["X_rgb"][m], np.float32)
    D = np.asarray(d["X_depth"][m], np.float32)

    if kind == "true":
        return R, D
    if kind == "run_mean":
        return (
            np.repeat(R.mean(0, keepdims=True), len(R), axis=0),
            np.repeat(D.mean(0, keepdims=True), len(D), axis=0),
        )
    if kind == "shuffled":
        rng = np.random.default_rng(seed)
        p = rng.permutation(len(R))
        return R[p], D[p]
    raise ValueError(kind)

rows = []

# Baseline B9.
for rid, d in C_DATA.items():
    if not np.any(d["split"]=="val"):
        continue
    mask = d["C_precontact"][d["split"]=="val"]
    bpred = predict_B(B9_DENSE, d, "val")
    met = c_metrics_from_prediction(
        bpred, d, B9_CALIBRATION, mask=mask
    )
    rows.append({
        "condition":"B9_baseline",
        "run_id":rid,
        "terrain":d["terrain"],
        **met,
    })

for cond in ("true", "shuffled", "run_mean"):
    for rid, d in C_DATA.items():
        if not np.any(d["split"]=="val"):
            continue
        R, DP = _c_override_for_run(
            d, "val", cond,
            seed=781 + int(hashlib.sha256(rid.encode()).hexdigest()[:8],16) % 100000,
        )
        pred = predict_C(
            C_SELECTED_MODEL, d, "val",
            override_rgb=R,
            override_depth=DP,
        )
        mask = d["C_precontact"][d["split"]=="val"]
        met = c_metrics_from_prediction(
            pred, d, C_SELECTED_CAL, mask=mask
        )
        rows.append({
            "condition":cond,
            "run_id":rid,
            "terrain":d["terrain"],
            **met,
        })

C7_CONTROLS = pd.DataFrame(rows)
display(C7_CONTROLS)

summary = C7_CONTROLS.groupby("condition").mean(numeric_only=True)
display(summary)

def _v(cond, metric):
    return float(summary.loc[cond, metric])

true_risk = _v("true","risk_spearman")
base_risk = _v("B9_baseline","risk_spearman")
shuf_risk = _v("shuffled","risk_spearman")
mean_risk = _v("run_mean","risk_spearman")

true_force = _v("true","force_peak_spearman")
base_force = _v("B9_baseline","force_peak_spearman")

base_tail = _v("B9_baseline","force_tail_component_mae")
true_tail = _v("true","force_tail_component_mae")

C7_GATE = {
    "selected_mode": C_SELECTED_MODE,
    "validation_precontact_true_risk_spearman": true_risk,
    "validation_precontact_B9_risk_spearman": base_risk,
    "validation_precontact_shuffled_risk_spearman": shuf_risk,
    "validation_precontact_runmean_risk_spearman": mean_risk,
    "validation_precontact_true_force_peak_spearman": true_force,
    "validation_precontact_B9_force_peak_spearman": base_force,
    "validation_precontact_force_tail_improvement_fraction": (
        (base_tail-true_tail)/max(base_tail,1e-9)
    ),
    "true_beats_B9_risk": bool(true_risk > base_risk + 0.01),
    "true_beats_shuffled_risk": bool(true_risk > shuf_risk + 0.01),
    "true_beats_runmean_risk": bool(true_risk > mean_risk + 0.005),
    "tail_or_peak_improves": bool(
        true_force > base_force + 0.01
        or true_tail < base_tail
    ),
}
C7_GATE["open_fresh_C_test"] = bool(
    C7_GATE["true_beats_B9_risk"]
    and C7_GATE["true_beats_shuffled_risk"]
    and C7_GATE["true_beats_runmean_risk"]
    and C7_GATE["tail_or_peak_improves"]
)

print("\nC7 VALIDATION CAUSALITY GATE")
print(json.dumps(C7_GATE, indent=2))

C7_CONTROLS.to_csv(C_TABLE_ROOT/"C7_causality_controls.csv", index=False)
(C_TABLE_ROOT/"C7_gate.json").write_text(json.dumps(C7_GATE, indent=2))

if not C7_GATE["open_fresh_C_test"]:
    raise RuntimeError(
        "C7 validation controls do not yet prove useful time-local visual value. "
        "Fresh C test trajectories remain unopened."
    )

print("\nC7 passed. Fresh C test trajectories may now be evaluated ONCE.")


####################################################################################################
# C8
####################################################################################################


# ======================================================================
# C8 — FIRST EVALUATION ON FRESH, UNTOUCHED STAGE-C TEST TRAJECTORIES
#
# Compare:
#   frozen B9 dynamics baseline
#   selected C visual adapter
#
# Report both ALL origins and the stricter PRECONTACT anticipation subset.
# ======================================================================

import json
from datetime import datetime, timezone
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

if not bool(C7_GATE.get("open_fresh_C_test", False)):
    raise RuntimeError("C7 gate did not pass.")

C8_MARKER = C_ROOT / "C8_TEST_WAS_SCORED.json"
ALLOW_C8_RESCORE = False

if C8_MARKER.exists() and not ALLOW_C8_RESCORE:
    previous = json.loads(C8_MARKER.read_text())
    raise RuntimeError(
        "C8 test trajectories were already scored. "
        "They are no longer untouched for retuning. "
        "Set ALLOW_C8_RESCORE=True only for reproducibility. "
        f"Previous marker: {previous}"
    )

test_ids = C_SPLIT_MANIFEST.loc[
    C_SPLIT_MANIFEST["role"]=="test", "run_id"
].tolist()

rows = []
C8_PREDICTIONS = {}

for rid in test_ids:
    d = C_DATA[rid]
    # First label-derived anticipation mask computation for this fresh test run.
    d = c_compute_precontact(d)
    C_DATA[rid] = d

    bpred = predict_B(B9_DENSE, d, "test")
    cpred = predict_C(C_SELECTED_MODEL, d, "test")
    C8_PREDICTIONS[rid] = {"B9":bpred, "C":cpred}

    split_mask = d["split"]=="test"
    pre_mask = d["C_precontact"][split_mask]

    for model_name, pred, cal in [
        ("B9_baseline", bpred, B9_CALIBRATION),
        (f"C_{C_SELECTED_MODE}", cpred, C_SELECTED_CAL),
    ]:
        for subset, mask in [
            ("all", np.ones(len(pred["truth"]), dtype=bool)),
            ("precontact", pre_mask),
        ]:
            met = c_metrics_from_prediction(pred, d, cal, mask=mask)
            rows.append({
                "run_id":rid,
                "terrain":d["terrain"],
                "model":model_name,
                "subset":subset,
                **met,
            })

    # --------------------------------------------------------------
    # Visual comparison plot on the most stressful precontact window.
    # --------------------------------------------------------------
    truth = _b_reshape_cont(cpred["truth"])
    ftrue = _b_group(truth,"force_odom").reshape(-1,B_H,4,3)
    actual_peak = np.max(np.linalg.norm(ftrue,axis=-1),axis=(1,2))

    btruth = _b_reshape_cont(bpred["truth"])
    bmu = _b_group(bpred["continuous_mean"],"force_odom").reshape(-1,B_H,4,3)
    cmu = _b_group(cpred["continuous_mean"],"force_odom").reshape(-1,B_H,4,3)
    bpeak = np.max(np.linalg.norm(bmu,axis=-1),axis=(1,2))
    cpeak = np.max(np.linalg.norm(cmu,axis=-1),axis=(1,2))

    origins = cpred["origins"]
    t = d["time"][origins]
    candidate = np.flatnonzero(pre_mask)
    if len(candidate):
        center = int(candidate[np.argmax(actual_peak[candidate])])
    else:
        center = int(np.argmax(actual_peak))
    half = min(300, max(100, len(t)//12))
    lo=max(0,center-half); hi=min(len(t),center+half)
    sl=slice(lo,hi)
    rel=t[sl]-t[lo]

    fig,ax=plt.subplots(figsize=(14,4.5))
    ax.plot(rel,actual_peak[sl],label="actual future peak force")
    ax.plot(rel,bpeak[sl],label="B9 predicted mean peak")
    ax.plot(rel,cpeak[sl],label=f"C-{C_SELECTED_MODE} predicted mean peak")
    ax.axhline(B9_PREP["force95"],linestyle="--",alpha=.4,label="train force p95")
    ax.set_title(f"C8 {rid}: fresh test force response")
    ax.set_xlabel("relative time [s]"); ax.set_ylabel("N")
    ax.grid(alpha=.2); ax.legend()
    plt.tight_layout()
    plt.savefig(C_FIG_ROOT/f"C8_{rid}_force.png",dpi=160,bbox_inches="tight")
    plt.show()

    # Risk comparison.
    br = B9_predicted_risk(bpred,B9_PREP,B9_CALIBRATION)
    cr = B9_predicted_risk(cpred,B9_PREP,C_SELECTED_CAL)
    ar = B9_actual_risk(
        truth,
        cpred["contact_truth"],
        cpred["touchdown_truth"],
        cpred["impulse_truth"],
        B9_PREP,
        C_SELECTED_CAL,
    )

    fig,ax=plt.subplots(figsize=(14,4.5))
    rt=t-t[0]
    ax.plot(rt,ar["physical_risk"],label="actual future risk",alpha=.85)
    ax.plot(rt,br["physical_risk"],label="B9 predicted risk",alpha=.75)
    ax.plot(rt,cr["physical_risk"],label=f"C-{C_SELECTED_MODE} predicted risk",alpha=.9)
    ax.set_title(f"C8 {rid}: incremental visual risk prediction")
    ax.set_xlabel("relative time [s]"); ax.set_ylabel("risk")
    ax.grid(alpha=.2); ax.legend()
    plt.tight_layout()
    plt.savefig(C_FIG_ROOT/f"C8_{rid}_risk.png",dpi=160,bbox_inches="tight")
    plt.show()

C8_METRICS = pd.DataFrame(rows)
display(C8_METRICS)

# Paired visual benefit table.
pivot = C8_METRICS.pivot_table(
    index=["run_id","terrain","subset"],
    columns="model",
    values=[
        "risk_spearman",
        "top10_overlap",
        "force_peak_spearman",
        "force_tail_component_mae",
        "E_tail_component_mae",
        "touchdown_f1",
    ],
).reset_index()

C8_METRICS.to_csv(C_TABLE_ROOT/"C8_test_metrics.csv",index=False)
pivot.to_csv(C_TABLE_ROOT/"C8_test_paired.csv",index=False)

print("\nC8 paired test metrics:")
display(pivot)

marker = {
    "scored_utc": datetime.now(timezone.utc).isoformat(),
    "test_run_ids": test_ids,
    "selected_visual_mode": C_SELECTED_MODE,
    "C_checkpoint": C_SELECTED_MODEL["checkpoint"],
    "B9_checkpoint": B9_DENSE["checkpoint"],
    "warning": (
        "If C1-C7 are retuned after reading C8, freeze new test trajectories "
        "before making another untouched Stage-C claim."
    ),
}
C8_MARKER.write_text(json.dumps(marker, indent=2))

print("\nC8 marker written:", C8_MARKER)


####################################################################################################
# C9
####################################################################################################


# ======================================================================
# C9 — LEAD-TIME / HORIZON ANALYSIS OF VISUAL INFORMATION
#
# Does vision help more before the robot physically reaches the patch?
# Compare B9 vs C at every 0.1...1.0 s horizon on the fresh C test runs.
# ======================================================================

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss
from IPython.display import display

if "C8_PREDICTIONS" not in globals():
    raise RuntimeError("Run C8 first.")

rows = []

for rid, store in C8_PREDICTIONS.items():
    d = C_DATA[rid]
    bp = store["B9"]
    cp = store["C"]

    truth = _b_reshape_cont(cp["truth"])
    bmu = bp["continuous_mean"]
    cmu = cp["continuous_mean"]
    split_mask = d["split"]=="test"
    pre = d["C_precontact"][split_mask]

    for h_idx, h in enumerate(CFG.horizons_s):
        for subset, mask in [
            ("all", np.ones(len(truth), dtype=bool)),
            ("precontact", pre),
        ]:
            if mask.sum() < 20:
                continue

            ft = _b_group(truth,"force_odom").reshape(-1,B_H,4,3)[mask,h_idx]
            fb = _b_group(bmu,"force_odom").reshape(-1,B_H,4,3)[mask,h_idx]
            fc = _b_group(cmu,"force_odom").reshape(-1,B_H,4,3)[mask,h_idx]

            Et = _b_group(truth,"E").reshape(-1,B_H,4,3)[mask,h_idx]
            Eb = _b_group(bmu,"E").reshape(-1,B_H,4,3)[mask,h_idx]
            Ec = _b_group(cmu,"E").reshape(-1,B_H,4,3)[mask,h_idx]

            b_force = float(np.abs(fb-ft).mean())
            c_force = float(np.abs(fc-ft).mean())
            b_E = float(np.abs(Eb-Et).mean())
            c_E = float(np.abs(Ec-Et).mean())

            # Contact probability quality.
            y = cp["contact_truth"][mask,h_idx].reshape(-1)
            pb = bp["contact_probability"][mask,h_idx].reshape(-1)
            pc = cp["contact_probability"][mask,h_idx].reshape(-1)
            brier_b = float(np.mean((pb-y)**2))
            brier_c = float(np.mean((pc-y)**2))

            rows.append({
                "run_id":rid,
                "terrain":d["terrain"],
                "subset":subset,
                "horizon_s":float(h),
                "B9_force_mae":b_force,
                "C_force_mae":c_force,
                "force_improvement_fraction":(b_force-c_force)/max(b_force,1e-9),
                "B9_E_mae":b_E,
                "C_E_mae":c_E,
                "E_improvement_fraction":(b_E-c_E)/max(b_E,1e-9),
                "B9_contact_brier":brier_b,
                "C_contact_brier":brier_c,
                "contact_brier_improvement_fraction":(
                    (brier_b-brier_c)/max(brier_b,1e-9)
                ),
                "samples":int(mask.sum()),
            })

C9_HORIZON_BENEFIT = pd.DataFrame(rows)
display(C9_HORIZON_BENEFIT)

summary = C9_HORIZON_BENEFIT.groupby(
    ["subset","horizon_s"],as_index=False
).mean(numeric_only=True)

fig,ax=plt.subplots(figsize=(10,4.5))
for subset in ("all","precontact"):
    g=summary[summary["subset"]==subset]
    ax.plot(
        g["horizon_s"],
        100*g["force_improvement_fraction"],
        marker="o",label=f"{subset}: force"
    )
ax.axhline(0,linestyle="--",alpha=.4)
ax.set(
    xlabel="forecast horizon [s]",
    ylabel="visual improvement in force MAE [%]",
    title="C9: incremental RGB-D value vs forecast horizon",
)
ax.grid(alpha=.2); ax.legend()
plt.tight_layout()
plt.savefig(C_FIG_ROOT/"C9_force_benefit_vs_horizon.png",dpi=160,bbox_inches="tight")
plt.show()

C9_HORIZON_BENEFIT.to_csv(
    C_TABLE_ROOT/"C9_horizon_benefit.csv",index=False
)
print("\nC9 complete.")


####################################################################################################
# C10
####################################################################################################


# ======================================================================
# C10 — STAGE-C FINAL GATE + EXPORT FRAMES FOR THE NEXT VLM STAGE
#
# No VLM is run here.
# We only decide whether causal RGB-D added real predictive information and
# export the most informative visual cases for the later semantic analysis.
# ======================================================================

import json, numpy as np, pandas as pd
from pathlib import Path
from IPython.display import display

if "C9_HORIZON_BENEFIT" not in globals():
    raise RuntimeError("Run C9 first.")

c_name = f"C_{C_SELECTED_MODE}"

# Aggregate fresh-test metrics.
pre = C8_METRICS[C8_METRICS["subset"]=="precontact"]
allm = C8_METRICS[C8_METRICS["subset"]=="all"]

def _mean(model, subset_df, metric):
    g = subset_df[subset_df["model"]==model]
    return float(g[metric].mean())

base_risk = _mean("B9_baseline", pre, "risk_spearman")
vis_risk = _mean(c_name, pre, "risk_spearman")
base_peak = _mean("B9_baseline", pre, "force_peak_spearman")
vis_peak = _mean(c_name, pre, "force_peak_spearman")
base_tail = _mean("B9_baseline", pre, "force_tail_component_mae")
vis_tail = _mean(c_name, pre, "force_tail_component_mae")
base_E = _mean("B9_baseline", pre, "E_tail_component_mae")
vis_E = _mean(c_name, pre, "E_tail_component_mae")

pre_h = C9_HORIZON_BENEFIT[C9_HORIZON_BENEFIT["subset"]=="precontact"]
long_h = pre_h[pre_h["horizon_s"] >= 0.4]
mean_long_force_gain = float(long_h["force_improvement_fraction"].mean())
mean_long_E_gain = float(long_h["E_improvement_fraction"].mean())

C10_FINAL_GATE = {
    "selected_visual_mode": C_SELECTED_MODE,
    "fresh_test_run_ids": C_SPLIT_MANIFEST.loc[
        C_SPLIT_MANIFEST["role"]=="test","run_id"
    ].tolist(),
    "validation_causality_gate_passed": bool(C7_GATE["open_fresh_C_test"]),
    "precontact_risk_spearman_B9": base_risk,
    "precontact_risk_spearman_visual": vis_risk,
    "precontact_risk_spearman_delta": vis_risk-base_risk,
    "precontact_force_peak_spearman_B9": base_peak,
    "precontact_force_peak_spearman_visual": vis_peak,
    "precontact_force_peak_spearman_delta": vis_peak-base_peak,
    "precontact_force_tail_mae_improvement_fraction": (
        (base_tail-vis_tail)/max(base_tail,1e-9)
    ),
    "precontact_E_tail_mae_improvement_fraction": (
        (base_E-vis_E)/max(base_E,1e-9)
    ),
    "mean_precontact_force_gain_h_ge_0p4": mean_long_force_gain,
    "mean_precontact_E_gain_h_ge_0p4": mean_long_E_gain,
}

# Stage-C diagnostic readiness:
# visual information should improve at least TWO of the three core anticipation
# signals on fresh test data: risk ordering, force-tail/peak, long-horizon response.
signals = [
    C10_FINAL_GATE["precontact_risk_spearman_delta"] > 0.02,
    (
        C10_FINAL_GATE["precontact_force_peak_spearman_delta"] > 0.02
        or C10_FINAL_GATE["precontact_force_tail_mae_improvement_fraction"] > 0.03
    ),
    (
        mean_long_force_gain > 0.02
        or mean_long_E_gain > 0.02
    ),
]
C10_FINAL_GATE["positive_anticipation_signals"] = int(sum(bool(x) for x in signals))
C10_FINAL_GATE["stage_C_ready_for_VLM_semantic_analysis"] = bool(
    C7_GATE["open_fresh_C_test"]
    and sum(bool(x) for x in signals) >= 2
)

print("\n"+"="*92)
print("C10 FINAL STAGE-C GATE")
print("="*92)
print(json.dumps(C10_FINAL_GATE, indent=2))

# ----------------------------------------------------------------------
# Export informative visual cases for later VLM analysis.
# We prefer cases where:
#   - actual future risk is high
#   - C prediction is closer to actual risk than B9
#   - history was quiet (precontact)
# ----------------------------------------------------------------------
export_rows = []

for rid, store in C8_PREDICTIONS.items():
    d = C_DATA[rid]
    bp = store["B9"]
    cp = store["C"]

    truth = _b_reshape_cont(cp["truth"])
    br = B9_predicted_risk(bp,B9_PREP,B9_CALIBRATION)
    cr = B9_predicted_risk(cp,B9_PREP,C_SELECTED_CAL)
    ar = B9_actual_risk(
        truth,
        cp["contact_truth"],
        cp["touchdown_truth"],
        cp["impulse_truth"],
        B9_PREP,
        C_SELECTED_CAL,
    )

    m = d["split"]=="test"
    pre = d["C_precontact"][m]
    frame_ids = d["visual_frame_id"][m]
    origins = d["origins"][m]
    event_lead = d["C_event_lead_s"][m]

    actual = ar["physical_risk"]
    bpred = br["physical_risk"]
    cpred = cr["physical_risk"]

    correction = np.abs(actual-bpred)-np.abs(actual-cpred)
    high = actual >= np.quantile(actual, .75)
    cand = np.flatnonzero(pre & high)

    row_meta = C_SPLIT_MANIFEST[
        C_SPLIT_MANIFEST["run_id"]==rid
    ].iloc[0]
    base = RAW_ROOT / row_meta["environment"] / "Data_anymal" / row_meta["run"]

    for j in cand:
        fid = str(frame_ids[j])
        export_rows.append({
            "run_id":rid,
            "terrain":d["terrain"],
            "environment":row_meta["environment"],
            "run":row_meta["run"],
            "origin_index":int(origins[j]),
            "origin_time_s":float(d["time"][origins[j]]),
            "frame_id":fid,
            "event_lead_s":float(event_lead[j]) if np.isfinite(event_lead[j]) else np.nan,
            "actual_future_risk":float(actual[j]),
            "B9_predicted_risk":float(bpred[j]),
            "C_predicted_risk":float(cpred[j]),
            "visual_correction_gain":float(correction[j]),
            "image_path":str(
                base/"image_lcam_front"/f"{fid}_lcam_front.png"
            ),
            "depth_path":str(
                base/"depth_lcam_front"/f"{fid}_lcam_front_depth.png"
            ),
        })

C10_VLM_MANIFEST = pd.DataFrame(export_rows)
if len(C10_VLM_MANIFEST):
    C10_VLM_MANIFEST = C10_VLM_MANIFEST.sort_values(
        ["actual_future_risk","visual_correction_gain"],
        ascending=[False,False],
    ).head(200)

display(C10_VLM_MANIFEST.head(30))
C10_VLM_MANIFEST.to_csv(
    C_TABLE_ROOT/"C10_VLM_candidate_manifest.csv",index=False
)

(C_TABLE_ROOT/"C10_final_gate.json").write_text(
    json.dumps(C10_FINAL_GATE,indent=2)
)

print("\nC10 candidate VLM manifest:",
      C_TABLE_ROOT/"C10_VLM_candidate_manifest.csv")
print(
    "\nStage C complete. "
    "No VLM or planner was used in C1-C10."
)