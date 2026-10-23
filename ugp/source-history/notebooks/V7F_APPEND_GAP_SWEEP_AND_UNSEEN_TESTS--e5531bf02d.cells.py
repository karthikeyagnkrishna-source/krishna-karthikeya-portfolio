# Source inspection export of the adjacent notebook.
# Notebook execution order and dependencies still apply.
# IPython magic, shell lines and prose-only conclusion cells are comments.

# %% notebook cell 0
# ======================================================================
# APPEND CELL 1 — SAME-RUN TEMPORAL-GAP SWEEP
# Exact V7/V7-F model, but force the LAST GRADIENT-TRAINING ANCHOR to be
# 4 s / 7 s / 9 s / 10 s before each ttest.
#
# Important: there is NO validation data after the requested cutoff.
# This makes the temporal gap real: the specialist receives no training
# or model-selection information from (ttest-gap, ttest].
# ======================================================================
import os, gc, copy, json, math, random, hashlib, zipfile, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch

_required = [
    'run_records','prior_short','prior_long','v7f_head_short','v7f_head_long',
    'SHORT_CONTEXT_SEC','LONG_CONTEXT_SEC','SHORT_HORIZONS','LONG_HORIZONS',
    'V7F_LOCAL_HOP_SHORT','V7F_LOCAL_HOP_LONG',
    'SHORT_RECENCY_HALF_LIFE_SEC','LONG_RECENCY_HALF_LIFE_SEC',
    'TRAIN_LOOKBACK_SEC','V7F_LOCAL_EPOCHS','V7F_LR_LOCAL',
    'V7F_WEIGHT_DECAY','V7F_GRAD_CLIP','CONTACT_THRESHOLD',
    'make_stats_between','normalize_arrays','nearest_index','horizon_offsets',
    '_make_local_force_item','collate_v7f','v7f_force_loss',
    'predict_v7f_force','LEG_NAMES','DEVICE','V7F_SEED','_seed_v7f'
]
_missing = [x for x in _required if x not in globals()]
if _missing:
    raise RuntimeError("Run the completed V7 + V7-F notebook first. Missing: "+repr(_missing))

GAP_SWEEP_SEC = [4.0, 7.0, 9.0, 10.0]
GAP_ROOT = Path('/content/drive/MyDrive/UGP_V7_FULL4B/V7F_same_run_gap_sweep')
if GAP_ROOT.exists():
    shutil.rmtree(GAP_ROOT)
GAP_ROOT.mkdir(parents=True, exist_ok=True)

def _gap_windows_exact(data, visual, ttest, anchor_gap_s, context_sec, hop_sec,
                       max_horizon, half_life):
    """
    Build causal local-adaptation windows backwards from an exact desired
    final INPUT-ANCHOR time = ttest - anchor_gap_s.

    Each row = (idx, recency_weight, actual_anchor_time, actual_target_end).
    No window anchor later than ttest-gap is allowed.
    """
    dt = float(np.median(np.diff(data['time'])))
    H = int(round(float(context_sec)/dt))
    max_off = int(round(float(max_horizon)/dt))
    anchor_cut = float(ttest) - float(anchor_gap_s)

    if anchor_cut <= float(data['time'][0]) + float(context_sec):
        raise RuntimeError(
            f"Not enough history for ttest={ttest:.3f}, gap={anchor_gap_s:.3f}"
        )

    desired = []
    a = anchor_cut
    lower = max(float(data['time'][0]) + float(context_sec),
                anchor_cut - float(TRAIN_LOOKBACK_SEC))
    while a >= lower - 1e-9:
        desired.append(a)
        a -= float(hop_sec)

    rows, seen = [], set()
    for da in reversed(desired):
        idx = nearest_index(data['time'], da)
        if idx in seen:
            continue
        seen.add(idx)
        if idx-H+1 < 0 or idx+max_off >= len(data['time']):
            continue
        if int(visual['visual_index'][idx]) < 0:
            continue

        actual_anchor = float(data['time'][idx])
        target_end = float(data['time'][idx+max_off])

        if actual_anchor > anchor_cut + 0.015:
            continue
        if target_end >= float(ttest) - 1e-9:
            continue

        age = max(0.0, anchor_cut - actual_anchor)
        w = 0.5**(age/float(half_life))
        rows.append((idx, w, actual_anchor, target_end))

    if not rows:
        raise RuntimeError(
            f"No gap-sweep windows for ttest={ttest}, gap={anchor_gap_s}"
        )
    rows.sort(key=lambda x: x[2])
    wmax = max(r[1] for r in rows)
    rows = [(i,w/wmax,a,te) for i,w,a,te in rows]
    return rows, H, anchor_cut

def _adapt_v7f_at_fixed_gap(global_head, base, rec, ttest, anchor_gap_s,
                            context_sec, hop_sec, horizons, half_life,
                            seed, label):
    _seed_v7f(seed)
    head = copy.deepcopy(global_head).to(DEVICE)

    end0 = float(min(rec['ttests']))
    stats = make_stats_between(rec['data'], float(rec['data']['time'][0]), end0)
    norm = normalize_arrays(rec['data'], stats)
    physics = rec['physics_prior_v7']

    wins, H, anchor_cut = _gap_windows_exact(
        rec['data'], rec['visual'], ttest, anchor_gap_s,
        context_sec, hop_sec, max(horizons), half_life
    )

    items = []
    for idx,w,anchor_t,target_end in wins:
        it = _make_local_force_item(
            base, rec, idx, H, horizons, stats, norm, physics
        )
        it['w'] = float(w)
        it['anchor_t'] = float(anchor_t)
        it['target_end'] = float(target_end)
        items.append(it)

    opt = torch.optim.AdamW(
        head.parameters(), lr=V7F_LR_LOCAL, weight_decay=V7F_WEIGHT_DECAY
    )

    for ep in range(V7F_LOCAL_EPOCHS):
        head.train()
        losses = []
        for it in items:
            b = collate_v7f([it],[0])
            opt.zero_grad(set_to_none=True)
            o = head(
                b['x'], b['baseF'], b['pstd_force'],
                b['pcontact'], b['regime']
            )
            raw,_ = v7f_force_loss(o,b)
            loss = raw * float(it['w'])
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite gap-sweep loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), V7F_GRAD_CLIP)
            opt.step()
            losses.append(float(raw.detach().cpu()))
            del b,o,raw,loss

        print(
            f"    {label} gap={anchor_gap_s:g}s "
            f"epoch {ep+1}/{V7F_LOCAL_EPOCHS} "
            f"n={len(items)} mean_loss={np.mean(losses):.4f}"
        )

    audit = dict(
        n_train_windows=int(len(items)),
        requested_anchor_gap_s=float(anchor_gap_s),
        cutoff_anchor_s=float(anchor_cut),
        latest_train_anchor_s=float(items[-1]['anchor_t']),
        latest_train_target_end_s=float(items[-1]['target_end']),
        actual_anchor_gap_s=float(ttest-items[-1]['anchor_t']),
        actual_target_gap_s=float(ttest-items[-1]['target_end']),
    )
    return head, stats, physics, H, audit

def _append_force_rows(rows, rec, test_id, ttest, gap_s, branch,
                       horizons, predF, sample, audit):
    for hi,h in enumerate(horizons):
        for li,L in enumerate(LEG_NAMES):
            pred = np.asarray(predF[hi,li], np.float64)
            gt = np.asarray(sample['target_p_raw'][hi,li,9:12], np.float64)
            err = pred-gt
            ve = float(np.linalg.norm(err))
            pm = float(np.linalg.norm(pred))
            gm = float(np.linalg.norm(gt))
            cos = float(np.dot(pred,gt)/(pm*gm+1e-12)) if gm >= CONTACT_THRESHOLD and pm > 1e-9 else np.nan
            rows.append(dict(
                terrain_type=rec['terrain_type'],
                environment=rec['environment'],
                run=rec['run'],
                test_id=int(test_id),
                ttest_s=float(ttest),
                requested_anchor_gap_s=float(gap_s),
                branch=branch,
                horizon_s=float(h),
                leg=L,
                pred_Fx=float(pred[0]), pred_Fy=float(pred[1]), pred_Fz=float(pred[2]),
                gt_Fx=float(gt[0]), gt_Fy=float(gt[1]), gt_Fz=float(gt[2]),
                component_MAE_N=float(np.abs(err).mean()),
                force_vector_error_N=ve,
                pred_force_mag_N=pm,
                true_force_mag_N=gm,
                force_magnitude_abs_error_N=float(abs(pm-gm)),
                force_direction_cosine=cos,
                vector_error_percent=(100.0*ve/gm) if gm > 1e-9 else np.nan,
                latest_train_anchor_s=audit['latest_train_anchor_s'],
                latest_train_target_end_s=audit['latest_train_target_end_s'],
                actual_anchor_gap_s=audit['actual_anchor_gap_s'],
                actual_target_gap_s=audit['actual_target_gap_s'],
                n_train_windows=audit['n_train_windows'],
            ))

rows = []
audit_rows = []

for ri,rec in enumerate(run_records):
    for ti,tt in enumerate(rec['ttests'],1):
        print("\n"+"="*100)
        print(f"SAME-RUN GAP SWEEP | {rec['terrain_type']} {rec['environment']}/{rec['run']} "
              f"| test #{ti} t={tt:.1f}s")
        print("="*100)

        for gi,gap_s in enumerate(GAP_SWEEP_SEC):
            seed = int(V7F_SEED + 50000 + ri*10000 + ti*1000 + gi*10)

            sh,ss,sp,Hs,sa = _adapt_v7f_at_fixed_gap(
                v7f_head_short, prior_short, rec, float(tt), gap_s,
                SHORT_CONTEXT_SEC, V7F_LOCAL_HOP_SHORT, SHORT_HORIZONS,
                SHORT_RECENCY_HALF_LIFE_SEC, seed+1, 'SHORT'
            )
            lh,ls,lp,Hl,la = _adapt_v7f_at_fixed_gap(
                v7f_head_long, prior_long, rec, float(tt), gap_s,
                LONG_CONTEXT_SEC, V7F_LOCAL_HOP_LONG, LONG_HORIZONS,
                LONG_RECENCY_HALF_LIFE_SEC, seed+2, 'LONG'
            )

            sf,_,_,ssamp,_ = predict_v7f_force(
                prior_short, sh, rec, float(tt),
                SHORT_CONTEXT_SEC, SHORT_HORIZONS, ss, sp, Hs
            )
            lf,_,_,lsamp,_ = predict_v7f_force(
                prior_long, lh, rec, float(tt),
                LONG_CONTEXT_SEC, LONG_HORIZONS, ls, lp, Hl
            )

            _append_force_rows(
                rows, rec, ti, tt, gap_s, 'SHORT',
                SHORT_HORIZONS, sf, ssamp, sa
            )
            _append_force_rows(
                rows, rec, ti, tt, gap_s, 'LONG',
                LONG_HORIZONS, lf, lsamp, la
            )

            for branch,a in [('SHORT',sa),('LONG',la)]:
                audit_rows.append(dict(
                    terrain_type=rec['terrain_type'],
                    environment=rec['environment'],
                    run=rec['run'],
                    test_id=int(ti),
                    ttest_s=float(tt),
                    branch=branch,
                    **a
                ))

            del sh,lh,ssamp,lsamp
            gc.collect()
            torch.cuda.empty_cache()

raw = pd.DataFrame(rows)
audit = pd.DataFrame(audit_rows)

summary = (
    raw.groupby(
        ['terrain_type','environment','run',
         'requested_anchor_gap_s','horizon_s','leg'],
        as_index=False
    )
    .agg(
        n_tests=('test_id','count'),
        mean_component_MAE_N=('component_MAE_N','mean'),
        mean_vector_error_N=('force_vector_error_N','mean'),
        mean_true_force_mag_N=('true_force_mag_N','mean'),
        mean_force_magnitude_abs_error_N=('force_magnitude_abs_error_N','mean'),
        mean_direction_cosine=('force_direction_cosine','mean'),
        mean_actual_anchor_gap_s=('actual_anchor_gap_s','mean'),
        mean_actual_target_gap_s=('actual_target_gap_s','mean'),
    )
)
summary['vector_error_percent_of_mean_force'] = np.where(
    summary['mean_true_force_mag_N'] > 1e-9,
    100.0*summary['mean_vector_error_N']/summary['mean_true_force_mag_N'],
    np.nan
)

overall = (
    raw.groupby(['requested_anchor_gap_s','horizon_s'],as_index=False)
    .agg(
        n_vectors=('force_vector_error_N','count'),
        mean_component_MAE_N=('component_MAE_N','mean'),
        mean_vector_error_N=('force_vector_error_N','mean'),
        mean_true_force_mag_N=('true_force_mag_N','mean'),
        mean_force_magnitude_abs_error_N=('force_magnitude_abs_error_N','mean'),
        mean_direction_cosine=('force_direction_cosine','mean'),
    )
)
overall['vector_error_percent_of_mean_force'] = np.where(
    overall['mean_true_force_mag_N'] > 1e-9,
    100.0*overall['mean_vector_error_N']/overall['mean_true_force_mag_N'],
    np.nan
)

raw.to_csv(GAP_ROOT/'SAME_RUN_GAP_SWEEP_RAW.csv',index=False)
summary.to_csv(GAP_ROOT/'SAME_RUN_GAP_SWEEP_BY_TERRAIN_LEG.csv',index=False)
overall.to_csv(GAP_ROOT/'SAME_RUN_GAP_SWEEP_OVERALL.csv',index=False)
audit.to_csv(GAP_ROOT/'SAME_RUN_GAP_AUDIT.csv',index=False)

cfg = dict(
    experiment='same-run temporal-gap sweep',
    original_runs=[
        dict(terrain_type=r['terrain_type'],environment=r['environment'],run=r['run'])
        for r in run_records
    ],
    original_ttests={r['terrain_type']:[float(x) for x in r['ttests']] for r in run_records},
    requested_anchor_gaps_s=GAP_SWEEP_SEC,
    definition=(
        'gap = ttest - latest gradient-training INPUT anchor. '
        'No training or validation/model-selection data after that cutoff.'
    ),
    local_epochs=int(V7F_LOCAL_EPOCHS),
    lr_local=float(V7F_LR_LOCAL),
    note='Every gap condition starts from the same frozen global V7-F head.'
)
(GAP_ROOT/'CONFIG.json').write_text(json.dumps(cfg,indent=2))

manifest=[]
for p in sorted(GAP_ROOT.rglob('*')):
    if p.is_file():
        h=hashlib.sha256(p.read_bytes()).hexdigest()
        manifest.append(dict(file=str(p.relative_to(GAP_ROOT)),sha256=h,size_bytes=p.stat().st_size))
pd.DataFrame(manifest).to_csv(GAP_ROOT/'SHA256_MANIFEST.csv',index=False)

zip_path = Path('/content/V7F_SAME_RUN_GAP_SWEEP_RESULTS.zip')
if zip_path.exists():
    zip_path.unlink()
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED,allowZip64=True) as z:
    for p in sorted(GAP_ROOT.rglob('*')):
        if p.is_file():
            z.write(p,arcname=str(Path(GAP_ROOT.name)/p.relative_to(GAP_ROOT)))

print("\nSAME-RUN GAP SWEEP COMPLETE")
display(overall)
print("ZIP:",zip_path)
try:
    from google.colab import files
    files.download(str(zip_path))
except Exception as e:
    print("Auto-download skipped:",e)

# %% notebook cell 1
# ======================================================================
# APPEND CELL 2 — STRICT UNSEEN-TRAJECTORY ZERO-SHOT TEST
#
# Train source trajectories (already used above):
#   forest            ForestEnv/P2000
#   swamp_marsh       GreatMarsh/P2023
#   rough_uneven      SeasonalForestWinter/P2000
#   wet_uneven_proxy  OldTownFall/P2000
#
# Completely different held-out trajectories used here:
#   ForestEnv/P2001
#   GreatMarsh/P2027
#   SeasonalForestWinter/P2006
#   OldTownFall/P2001
#
# NO gradient updates, NO local adaptation, NO physics fitting and NO
# normalization fitting are allowed on the held-out trajectory.
# Training-run statistics + training-run physics prior are reused.
# ======================================================================
import os, gc, json, hashlib, zipfile, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch

_required = [
    'run_records','prior_short','prior_long','v7f_head_short','v7f_head_long',
    'download_run_v7','extract_run_physics','build_frame_manifest',
    'build_dinov3_cache','build_convnext_cache','build_utonia_cache',
    'build_waft_cache','load_visual_bundle',
    'make_stats_between','predict_v7f_force',
    'SHORT_CONTEXT_SEC','LONG_CONTEXT_SEC','SHORT_HORIZONS','LONG_HORIZONS',
    'CONTACT_THRESHOLD','LEG_NAMES','DEVICE','horizon_offsets'
]
_missing=[x for x in _required if x not in globals()]
if _missing:
    raise RuntimeError("Run the completed V7 + V7-F notebook first. Missing: "+repr(_missing))

UNSEEN_SPECS = [
    dict(terrain_type='forest',           environment='ForestEnv',            run='P2001'),
    dict(terrain_type='swamp_marsh',      environment='GreatMarsh',           run='P2027'),
    dict(terrain_type='rough_uneven',     environment='SeasonalForestWinter', run='P2006'),
    dict(terrain_type='wet_uneven_proxy', environment='OldTownFall',          run='P2001'),
]

TRAIN_KEYS={(r['environment'],r['run']) for r in run_records}
for s in UNSEEN_SPECS:
    if (s['environment'],s['run']) in TRAIN_KEYS:
        raise RuntimeError("Held-out trajectory accidentally overlaps training: "+repr(s))

ZERO_ROOT = Path('/content/drive/MyDrive/UGP_V7_FULL4B/V7F_unseen_trajectory_zero_shot')
if ZERO_ROOT.exists():
    shutil.rmtree(ZERO_ROOT)
ZERO_ROOT.mkdir(parents=True,exist_ok=True)

# ----------------------------------------------------------------------
# A) Prepare the 4 unseen trajectories and all exact V7 visual branches.
# This is inference preprocessing only. No model weights are changed.
# ----------------------------------------------------------------------
unseen_records=[]
for spec in UNSEEN_SPECS:
    env,run=spec['environment'],spec['run']
    print("\n"+"#"*100)
    print("UNSEEN PREP",spec['terrain_type'],env,run)
    print("#"*100)

    base=download_run_v7(env,run)
    data=extract_run_physics(base,run)
    manifest=build_frame_manifest(base,env,run,data)

    build_dinov3_cache(base,env,run,manifest)
    build_convnext_cache(base,env,run,manifest)
    build_utonia_cache(base,env,run,manifest)
    build_waft_cache(base,env,run,manifest)
    visual=load_visual_bundle(env,run,manifest)

    unseen_records.append(dict(
        **spec,base=base,data=data,manifest=manifest,visual=visual
    ))
    gc.collect()
    torch.cuda.empty_cache()

source_by_terrain={r['terrain_type']:r for r in run_records}

# ----------------------------------------------------------------------
# B) Pick four causal ttests per unseen run with successive spacing 7,9,10 s.
# No force/contact values are inspected to choose these test points.
# ----------------------------------------------------------------------
def _snap_valid_test_times(rec):
    data,vis=rec['data'],rec['visual']
    t=np.asarray(data['time'],np.float64)
    dt=float(np.median(np.diff(t)))
    Hlong=int(getattr(prior_long,'history_steps',round(LONG_CONTEXT_SEC/dt)))
    offmax=max(horizon_offsets(data,LONG_HORIZONS))

    valid=np.zeros(len(t),dtype=bool)
    lo=Hlong-1
    hi=len(t)-offmax
    valid[lo:hi]=True
    valid &= (np.asarray(vis['visual_index']) >= 0)
    valid_idx=np.flatnonzero(valid)
    if len(valid_idx)==0:
        raise RuntimeError("No valid test anchors for "+rec['environment']+'/'+rec['run'])

    valid_times=t[valid_idx]
    offsets=np.array([0.0,7.0,16.0,26.0],dtype=np.float64)

    earliest=float(valid_times[0])
    latest=float(valid_times[-1])
    base=max(earliest+1.0,20.0)
    if base+offsets[-1] > latest:
        base=latest-offsets[-1]
    if base < earliest:
        raise RuntimeError(
            f"Run too short for 4 unseen tests spaced 7/9/10 s: "
            f"{rec['environment']}/{rec['run']}"
        )

    picked=[]
    used=set()
    for desired in base+offsets:
        j=int(np.argmin(np.abs(valid_times-desired)))
        idx=int(valid_idx[j])
        if idx in used:
            raise RuntimeError("Duplicate snapped ttest; trajectory too short/dense issue.")
        used.add(idx)
        picked.append(float(t[idx]))
    return picked

# ----------------------------------------------------------------------
# C) Strict zero-shot inference.
#
# - use the global V7 prior directly;
# - use the global V7-F force head directly;
# - DO NOT call adapt_v7(...)
# - DO NOT call adapt_v7f_head(...)
# - DO NOT fit_causal_physics(...) on held-out trajectory;
# - DO NOT compute normalization statistics from held-out trajectory.
#
# Each held-out run uses the corresponding TRAINING RUN's normalization
# and physics prior. The held-out run contributes only causal inputs and
# post-hoc future ground truth for scoring.
# ----------------------------------------------------------------------
prior_short.eval(); prior_long.eval()
v7f_head_short.eval(); v7f_head_long.eval()

rows=[]
test_manifest=[]

for rec in unseen_records:
    src=source_by_terrain[rec['terrain_type']]

    src_end=float(min(src['ttests']))
    src_stats=make_stats_between(
        src['data'],float(src['data']['time'][0]),src_end
    )
    src_physics=src['physics_prior_v7']

    dt=float(np.median(np.diff(rec['data']['time'])))
    Hs=int(getattr(prior_short,'history_steps',round(SHORT_CONTEXT_SEC/dt)))
    Hl=int(getattr(prior_long,'history_steps',round(LONG_CONTEXT_SEC/dt)))

    ttests=_snap_valid_test_times(rec)
    print("\n"+"="*100)
    print(
        f"ZERO-SHOT {rec['terrain_type']} | TRAIN SOURCE "
        f"{src['environment']}/{src['run']} -> HELD OUT "
        f"{rec['environment']}/{rec['run']}"
    )
    print("ttests:",ttests)
    print("="*100)

    for ti,tt in enumerate(ttests,1):
        sf,_,_,ssamp,sidx = predict_v7f_force(
            prior_short,v7f_head_short,rec,tt,
            SHORT_CONTEXT_SEC,SHORT_HORIZONS,
            src_stats,src_physics,Hs
        )
        lf,_,_,lsamp,lidx = predict_v7f_force(
            prior_long,v7f_head_long,rec,tt,
            LONG_CONTEXT_SEC,LONG_HORIZONS,
            src_stats,src_physics,Hl
        )

        test_manifest.append(dict(
            terrain_type=rec['terrain_type'],
            training_environment=src['environment'],
            training_run=src['run'],
            unseen_environment=rec['environment'],
            unseen_run=rec['run'],
            test_id=int(ti),
            ttest_s=float(tt),
            short_robot_idx=int(sidx),
            long_robot_idx=int(lidx),
            zero_gradient_updates_on_test_run=True,
            zero_test_run_physics_fit=True,
            zero_test_run_normalization_fit=True,
        ))

        for branch,horizons,predF,samp in [
            ('SHORT',SHORT_HORIZONS,sf,ssamp),
            ('LONG',LONG_HORIZONS,lf,lsamp)
        ]:
            for hi,h in enumerate(horizons):
                for li,L in enumerate(LEG_NAMES):
                    pred=np.asarray(predF[hi,li],np.float64)
                    gt=np.asarray(samp['target_p_raw'][hi,li,9:12],np.float64)
                    err=pred-gt
                    ve=float(np.linalg.norm(err))
                    pm=float(np.linalg.norm(pred))
                    gm=float(np.linalg.norm(gt))
                    cos=(
                        float(np.dot(pred,gt)/(pm*gm+1e-12))
                        if gm>=CONTACT_THRESHOLD and pm>1e-9 else np.nan
                    )
                    rows.append(dict(
                        terrain_type=rec['terrain_type'],
                        training_environment=src['environment'],
                        training_run=src['run'],
                        unseen_environment=rec['environment'],
                        unseen_run=rec['run'],
                        test_id=int(ti),
                        ttest_s=float(tt),
                        branch=branch,
                        horizon_s=float(h),
                        leg=L,
                        pred_Fx=float(pred[0]), pred_Fy=float(pred[1]), pred_Fz=float(pred[2]),
                        gt_Fx=float(gt[0]), gt_Fy=float(gt[1]), gt_Fz=float(gt[2]),
                        component_MAE_N=float(np.abs(err).mean()),
                        force_vector_error_N=ve,
                        pred_force_mag_N=pm,
                        true_force_mag_N=gm,
                        force_magnitude_abs_error_N=float(abs(pm-gm)),
                        force_direction_cosine=cos,
                        vector_error_percent=(100.0*ve/gm) if gm>1e-9 else np.nan,
                    ))

        del ssamp,lsamp
        gc.collect()
        torch.cuda.empty_cache()

raw=pd.DataFrame(rows)
manifest_df=pd.DataFrame(test_manifest)

by_terrain_leg=(
    raw.groupby(
        ['terrain_type','training_environment','training_run',
         'unseen_environment','unseen_run','horizon_s','leg'],
        as_index=False
    )
    .agg(
        n_tests=('test_id','count'),
        mean_component_MAE_N=('component_MAE_N','mean'),
        mean_vector_error_N=('force_vector_error_N','mean'),
        mean_true_force_mag_N=('true_force_mag_N','mean'),
        mean_force_magnitude_abs_error_N=('force_magnitude_abs_error_N','mean'),
        mean_direction_cosine=('force_direction_cosine','mean'),
    )
)
by_terrain_leg['vector_error_percent_of_mean_force']=np.where(
    by_terrain_leg['mean_true_force_mag_N']>1e-9,
    100.0*by_terrain_leg['mean_vector_error_N']/by_terrain_leg['mean_true_force_mag_N'],
    np.nan
)

overall=(
    raw.groupby(['horizon_s'],as_index=False)
    .agg(
        n_vectors=('force_vector_error_N','count'),
        mean_component_MAE_N=('component_MAE_N','mean'),
        mean_vector_error_N=('force_vector_error_N','mean'),
        mean_true_force_mag_N=('true_force_mag_N','mean'),
        mean_force_magnitude_abs_error_N=('force_magnitude_abs_error_N','mean'),
        mean_direction_cosine=('force_direction_cosine','mean'),
    )
)
overall['vector_error_percent_of_mean_force']=np.where(
    overall['mean_true_force_mag_N']>1e-9,
    100.0*overall['mean_vector_error_N']/overall['mean_true_force_mag_N'],
    np.nan
)

raw.to_csv(ZERO_ROOT/'UNSEEN_TRAJECTORY_RAW.csv',index=False)
by_terrain_leg.to_csv(ZERO_ROOT/'UNSEEN_TRAJECTORY_BY_TERRAIN_LEG.csv',index=False)
overall.to_csv(ZERO_ROOT/'UNSEEN_TRAJECTORY_OVERALL.csv',index=False)
manifest_df.to_csv(ZERO_ROOT/'UNSEEN_TEST_MANIFEST.csv',index=False)

cfg=dict(
    experiment='strict unseen-trajectory zero-shot V7-F',
    training_runs=[
        dict(terrain_type=r['terrain_type'],environment=r['environment'],run=r['run'])
        for r in run_records
    ],
    held_out_runs=UNSEEN_SPECS,
    test_spacing_s=[7.0,9.0,10.0],
    local_adaptation=False,
    gradients_on_held_out_run=False,
    held_out_physics_fit=False,
    held_out_normalization_fit=False,
    calibration_policy=(
        'For each terrain, apply normalization statistics and physics prior '
        'from its original training trajectory to the unseen trajectory.'
    )
)
(ZERO_ROOT/'CONFIG.json').write_text(json.dumps(cfg,indent=2))

sha=[]
for p in sorted(ZERO_ROOT.rglob('*')):
    if p.is_file():
        sha.append(dict(
            file=str(p.relative_to(ZERO_ROOT)),
            sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            size_bytes=p.stat().st_size
        ))
pd.DataFrame(sha).to_csv(ZERO_ROOT/'SHA256_MANIFEST.csv',index=False)

zip_path=Path('/content/V7F_UNSEEN_TRAJECTORY_ZERO_SHOT_RESULTS.zip')
if zip_path.exists():
    zip_path.unlink()
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED,allowZip64=True) as z:
    for p in sorted(ZERO_ROOT.rglob('*')):
        if p.is_file():
            z.write(p,arcname=str(Path(ZERO_ROOT.name)/p.relative_to(ZERO_ROOT)))

print("\nSTRICT UNSEEN-TRAJECTORY TEST COMPLETE")
display(manifest_df)
display(overall)
print("ZIP:",zip_path)
try:
    from google.colab import files
    files.download(str(zip_path))
except Exception as e:
    print("Auto-download skipped:",e)
