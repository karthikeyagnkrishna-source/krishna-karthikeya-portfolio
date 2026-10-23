from __future__ import annotations

from pathlib import Path
import modal

APP_NAME = "ugp-v10r-h100"
WORK_VOLUME_NAME = "ugp-v10r-work"
HF_VOLUME_NAME = "ugp-v10r-hf-cache"

HERE = Path(__file__).resolve().parent
BASE_NOTEBOOK = HERE / "V10R_TONIGHT_RESPONSE_WORLD_MODEL1_1_2(1).ipynb"
C5_FILE = HERE / "C5_LOW_RAM_REPLACEMENT.py"
C6_FILE = HERE / "C6_TO_C10_CORRECTED_FULL.py"
C7R_FILE = HERE / "C7R_SPATIAL_CAUSAL_VISUAL_REPAIR_VALIDATION_ONLY.py"

for p in (BASE_NOTEBOOK, C5_FILE, C6_FILE, C7R_FILE):
    if not p.is_file():
        raise FileNotFoundError(f"Required local file missing: {p}")

work_vol = modal.Volume.from_name(WORK_VOLUME_NAME, create_if_missing=True)
hf_vol = modal.Volume.from_name(HF_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0", "ffmpeg")
    .pip_install(
        "torch>=2.5,<3",
        "torchvision>=0.20,<1",
        "numpy>=1.26,<3",
        "pandas>=2.2,<4",
        "scipy>=1.12,<2",
        "scikit-learn>=1.5,<2",
        "matplotlib>=3.8,<4",
        "timm>=1.0.20,<2",
        "opencv-python-headless>=4.10,<5",
        "pillow>=10,<13",
        "huggingface_hub>=0.30,<2",
        "rosbags>=0.10,<1",
        "psutil>=5.9,<8",
        "tqdm>=4.66,<5",
        "ipython>=8,<10",
        "git+https://github.com/castacks/tartanairpy.git",
    )
    .env(
        {
            "MPLBACKEND": "Agg",
            "HF_HOME": "/hf",
            "HF_HUB_CACHE": "/hf/hub",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_file(BASE_NOTEBOOK, "/opt/ugp/base.ipynb")
    .add_local_file(C5_FILE, "/opt/ugp/C5_LOW_RAM_REPLACEMENT.py")
    .add_local_file(C6_FILE, "/opt/ugp/C6_TO_C10_CORRECTED_FULL.py")
    .add_local_file(C7R_FILE, "/opt/ugp/C7R_SPATIAL_CAUSAL_VISUAL_REPAIR_VALIDATION_ONLY.py")
)

app = modal.App(APP_NAME, image=image)

# Known/frozen runs from the already-established B/C protocol.
# Prefetch happens on CPU so H100 credit is not spent waiting for network I/O.
PHYSICS_RUNS = {
    "GreatMarsh": ["P2023", "P2002", "P2025", "P2028", "P2012", "P2027"],
    "SeasonalForestWinter": ["P2000", "P2007", "P2001", "P2020", "P2005", "P2006"],
}

# C train/val + fresh C test need front RGB-D/meta for the original C1-C4 reconstruction.
VISUAL_RUNS = {
    "GreatMarsh": {"P2023", "P2002", "P2025", "P2028", "P2027"},
    "SeasonalForestWinter": {"P2000", "P2007", "P2001", "P2020", "P2006"},
}


def _ensure_layout():
    """Create Colab-compatible paths as symlinks into the persistent Modal volume."""
    import os
    from pathlib import Path

    work = Path("/work")
    project = work / "UGP_V7_FULL4B"
    raw = work / "tartanground_v10"
    project.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)

    Path("/content/drive/MyDrive").mkdir(parents=True, exist_ok=True)
    Path("/content").mkdir(parents=True, exist_ok=True)

    links = {
        Path("/content/drive/MyDrive/UGP_V7_FULL4B"): project,
        Path("/content/tartanground_v10"): raw,
    }
    for link, target in links.items():
        if link.is_symlink():
            if link.resolve() != target.resolve():
                link.unlink()
        elif link.exists():
            # A fresh Modal container should not have these, but fail safely if it does.
            if link.resolve() != target.resolve():
                raise RuntimeError(f"Refusing to replace existing path {link}")
        if not link.exists():
            os.symlink(target, link, target_is_directory=True)

    return project, raw


def _physics_ready(base: Path, run: str) -> bool:
    required = [
        base / "imu" / "imu_time.npy",
        base / "imu" / "acc.npy",
        base / "imu" / "gyro.txt",
        base / "rosbags" / f"{run}.bag",
    ]
    return all(p.is_file() and p.stat().st_size > 0 for p in required)


def _visual_ready(base: Path) -> bool:
    rgb = base / "image_lcam_front"
    dep = base / "depth_lcam_front"
    # Exact metadata files are verified later by C2/C4. Here we only avoid
    # needless redownloads when the large image/depth trees are clearly present.
    if not rgb.is_dir() or not dep.is_dir():
        return False
    try:
        return sum(1 for _ in rgb.glob("*.png")) > 100 and sum(1 for _ in dep.glob("*.png")) > 100
    except Exception:
        return False


@app.function(
    cpu=4,
    memory=16 * 1024,
    timeout=3 * 60 * 60,
    volumes={"/work": work_vol},
)
def prefetch_data() -> dict:
    """CPU-only TartanGround download into persistent Volume."""
    from pathlib import Path
    import tartanair as ta

    _, raw = _ensure_layout()
    ta.init(str(raw))

    rows = []
    for env, runs in PHYSICS_RUNS.items():
        for run in runs:
            base = raw / env / "Data_anymal" / run
            need_phys = not _physics_ready(base, run)
            need_vis = run in VISUAL_RUNS.get(env, set()) and not _visual_ready(base)

            modalities = []
            if need_vis:
                modalities += ["image", "depth", "meta"]
            if need_phys:
                modalities += ["imu", "rosbag"]

            # Deduplicate while preserving order.
            modalities = list(dict.fromkeys(modalities))
            if modalities:
                print(f"PREFETCH {env}/{run}: {modalities}", flush=True)
                ta.download_ground(
                    env=[env],
                    version=["anymal"],
                    traj=[run],
                    modality=modalities,
                    camera_name=["lcam_front"],
                    unzip=True,
                )
            else:
                print(f"PREFETCH reuse {env}/{run}", flush=True)

            rows.append(
                {
                    "environment": env,
                    "run": run,
                    "physics_ready": _physics_ready(base, run),
                    "visual_ready": (run not in VISUAL_RUNS.get(env, set())) or _visual_ready(base),
                }
            )

    bad = [r for r in rows if not r["physics_ready"] or not r["visual_ready"]]
    if bad:
        raise RuntimeError(f"Prefetch incomplete: {bad}")

    work_vol.commit()
    print("PREFETCH COMPLETE + VOLUME COMMITTED", flush=True)
    return {"runs": len(rows), "status": "ok"}


# Historical B10 result: already observed before this Modal migration.
# We restore the frozen readiness decision rather than rescoring/re-tuning on B10.
HISTORICAL_B10_GATE = {
    "test_run_ids": ["swamp_marsh_P2012", "rough_uneven_P2005"],
    "mean_core_command_benefit_fraction": 0.32667665415764946,
    "mean_tail_force_command_benefit_fraction": 0.27294685509279043,
    "mean_risk_spearman": 0.7845018615318589,
    "mean_top10_overlap": 0.4004625718799212,
    "mean_force_peak_spearman": 0.5366610704339385,
    "mean_touchdown_f1": 0.6653407923599876,
    "mean_force_upper_envelope_coverage": 0.9785168987761449,
    "command_information_gate": True,
    "tail_force_command_gate": True,
    "risk_trend_gate": True,
    "tail_event_gate": True,
    "stage_B_ready_for_stage_C": True,
    "interpretation": (
        "Restored historical B10 diagnostic readiness gate from the already-scored "
        "B10 trajectories; this Modal run does not use B10 for retuning."
    ),
}


def _exec_source(ns: dict, source: str, label: str):
    import time
    print("\n" + "=" * 96, flush=True)
    print(f"EXECUTING {label}", flush=True)
    print("=" * 96, flush=True)
    t0 = time.time()
    code = compile(source, f"<{label}>", "exec")
    exec(code, ns, ns)
    dt = (time.time() - t0) / 60.0
    print(f"FINISHED {label} in {dt:.2f} min", flush=True)


def _load_nb():
    import json
    with open("/opt/ugp/base.ipynb", "r", encoding="utf-8") as f:
        return json.load(f)


def _code_cell(nb, idx: int) -> str:
    c = nb["cells"][idx]
    if c["cell_type"] != "code":
        raise RuntimeError(f"Notebook cell {idx} is not code")
    return "".join(c["source"])


def _patch_c7r_for_h100(src: str) -> str:
    """Implementation-only speedups; no split/loss/gate/model change."""
    src = src.replace("C7R_BATCH = 16", "C7R_BATCH = 128")
    src = src.replace("C7R_TRAIN_BATCH = 96", "C7R_TRAIN_BATCH = 512")
    src = src.replace(
        'tr_loader = DataLoader(tr, batch_size=C7R_TRAIN_BATCH, sampler=sampler,\n'
        '                           num_workers=0, pin_memory=False)',
        'tr_loader = DataLoader(tr, batch_size=C7R_TRAIN_BATCH, sampler=sampler,\n'
        '                           num_workers=8, pin_memory=True,\n'
        '                           persistent_workers=True, prefetch_factor=4)',
    )
    src = src.replace(
        'va_loader = DataLoader(va, batch_size=C7R_TRAIN_BATCH, shuffle=False,\n'
        '                           num_workers=0, pin_memory=False)',
        'va_loader = DataLoader(va, batch_size=C7R_TRAIN_BATCH, shuffle=False,\n'
        '                           num_workers=8, pin_memory=True,\n'
        '                           persistent_workers=True, prefetch_factor=4)',
    )
    return src


@app.function(
    gpu="H100",
    cpu=8,
    memory=32 * 1024,
    ephemeral_disk=32 * 1024,
    timeout=6 * 60 * 60,
    volumes={"/work": work_vol, "/hf": hf_vol},
)
def run_h100_pipeline() -> dict:
    """
    Rebuild runtime state from the original successful notebook path, stop B at B9,
    restore the historical B10 gate, rebuild C1-C4, then run only the repaired C7R
    validation path. Fresh C tests are never scored here.
    """
    import gc
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    project, raw = _ensure_layout()
    print(subprocess.check_output(["nvidia-smi"], text=True), flush=True)

    # H100-friendly math; this does not alter the scientific model definition.
    import torch
    torch.set_float32_matmul_precision("high")
    print("torch:", torch.__version__, "cuda:", torch.version.cuda, flush=True)
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE", flush=True)

    nb = _load_nb()
    ns = {
        "__name__": "__main__",
        "__file__": "/opt/ugp/modal_reconstructed_runtime.py",
    }

    # ------------------------------------------------------------------
    # 1) Reconstruct the original baseline/A-series runtime.
    # Cells 4..70 had no recorded execution errors in the supplied notebook.
    # ------------------------------------------------------------------
    for idx in range(4, 71):
        c = nb["cells"][idx]
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if not src.strip():
            continue
        _exec_source(ns, src, f"BASE_NOTEBOOK_CELL_{idx}")

    # ------------------------------------------------------------------
    # 2) Stage B: execute B1-B9 only. Never rescore B10 in this migration.
    # ------------------------------------------------------------------
    bsrc = _code_cell(nb, 72)
    cut = bsrc.find("# B10", 100000)
    if cut < 0:
        raise RuntimeError("Could not locate B10 boundary in Stage-B cell")
    _exec_source(ns, bsrc[:cut], "STAGE_B_B1_TO_B9")

    ns["B10_FINAL_GATE"] = dict(HISTORICAL_B10_GATE)
    broot = Path(ns["B_ROOT"])
    marker = broot / "B10_TEST_WAS_SCORED.json"
    if not marker.exists():
        marker.write_text(
            json.dumps(
                {
                    "source": "historical_restore_for_modal_migration",
                    "final_gate": HISTORICAL_B10_GATE,
                    "warning": "Do not use historical B10 tests for retuning.",
                },
                indent=2,
            )
        )
    print("Restored historical B10 gate; B10 was NOT rescored.", flush=True)
    work_vol.commit()
    print("CHECKPOINT: B1-B9 committed", flush=True)

    # ------------------------------------------------------------------
    # 3) Rebuild C1-C4 exactly from the supplied combined cell.
    #    This creates C_DATA, causal visual alignment, manifests, precontact masks.
    # ------------------------------------------------------------------
    _exec_source(ns, _code_cell(nb, 74), "STAGE_C_C1_TO_C4")
    work_vol.commit()
    print("CHECKPOINT: C1-C4 committed", flush=True)

    # ------------------------------------------------------------------
    # 4) Define low-RAM C visual model class/helpers WITHOUT re-running the
    #    already-failed global C5 experiment. We go directly to approved C7R.
    # ------------------------------------------------------------------
    c5src = Path("/opt/ugp/C5_LOW_RAM_REPLACEMENT.py").read_text()
    # Fix the known y normalization broadcasting in case this helper is used.
    c5src = c5src.replace(
        'C_NORM_T["y_mean"][None, None, :]',
        'C_NORM_T["y_mean"][None, :, :]',
    ).replace(
        'C_NORM_T["y_std"][None, None, :]',
        'C_NORM_T["y_std"][None, :, :]',
    )
    train_marker = 'print("Training controlled visual residual adapters — LOW-RAM streaming path...")'
    c5cut = c5src.find(train_marker)
    if c5cut < 0:
        raise RuntimeError("Could not locate C5 training boundary")
    _exec_source(ns, c5src[:c5cut], "C5_LOW_RAM_DEFINITIONS_ONLY")

    # c_metrics_from_prediction is needed by C7R. Define C6 helpers only;
    # do not rerun the old global-feature C6/C7 branch.
    ns["C5_MODELS"] = {}
    c6src = Path("/opt/ugp/C6_TO_C10_CORRECTED_FULL.py").read_text()
    c6cut = c6src.find("# Baseline B9 validation predictions per run.")
    if c6cut < 0:
        raise RuntimeError("Could not locate C6 helper boundary")
    _exec_source(ns, c6src[:c6cut], "C6_METRIC_HELPERS_ONLY")

    # ------------------------------------------------------------------
    # 5) Approved spatial/temporal C7R repair, validation only.
    #    Split into feature extraction and training/gate so features persist
    #    even if a later training step is interrupted.
    # ------------------------------------------------------------------
    c7rsrc = Path("/opt/ugp/C7R_SPATIAL_CAUSAL_VISUAL_REPAIR_VALIDATION_ONLY.py").read_text()
    c7rsrc = _patch_c7r_for_h100(c7rsrc)

    boundary = 'print("C7R development frame features ready:", list(C7R_FRAME_FEATURES))'
    pos = c7rsrc.find(boundary)
    if pos < 0:
        raise RuntimeError("Could not locate C7R feature checkpoint boundary")
    endline = c7rsrc.find("\n", pos)
    feature_src = c7rsrc[: endline + 1]
    rest_src = c7rsrc[endline + 1 :]

    _exec_source(ns, feature_src, "C7R_SPATIAL_FEATURE_EXTRACTION")
    work_vol.commit()
    hf_vol.commit()
    print("CHECKPOINT: C7R spatial frame features committed", flush=True)

    _exec_source(ns, rest_src, "C7R_TRAIN_VALIDATION_GATE")
    work_vol.commit()
    hf_vol.commit()
    print("CHECKPOINT: C7R final validation outputs committed", flush=True)

    gate = ns.get("C7R_GATE")
    if gate is None:
        raise RuntimeError("C7R finished without C7R_GATE")

    result_path = (
        project
        / "V10R_STRICT_STAGED"
        / "stage_C_rgbd_incremental_v1"
        / "spatial_temporal_repair_v1"
        / "tables"
        / "C7R_gate.json"
    )
    print("FINAL C7R GATE:", json.dumps(gate, indent=2), flush=True)
    print("Gate file:", result_path, flush=True)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "status": "complete",
        "gate": gate,
        "gate_file": str(result_path),
        "fresh_test_scored": False,
    }


@app.function(
    cpu=1,
    memory=1024,
    timeout=3 * 60 * 60,
    volumes={"/work": work_vol},
)
def orchestrate() -> dict:
    """
    Durable remote orchestrator: prefetch on CPU, then spawn the H100 job.
    Because this function itself runs on Modal, a local laptop/Colab disconnect
    after launch does not prevent the H100 job from being scheduled.
    """
    from pathlib import Path
    import json

    print("Starting CPU prefetch...", flush=True)
    pre = prefetch_data.remote()
    print("Prefetch result:", pre, flush=True)

    print("Spawning H100 pipeline...", flush=True)
    call = run_h100_pipeline.spawn()
    call_id = call.object_id

    p = Path("/work/h100_call_id.txt")
    p.write_text(call_id + "\n")
    work_vol.commit()

    print("H100 FunctionCall ID:", call_id, flush=True)
    print("The H100 job is now running asynchronously on Modal.", flush=True)
    return {"h100_call_id": call_id, "prefetch": pre}


@app.local_entrypoint()
def main():
    """Launch the durable remote orchestrator."""
    result = orchestrate.remote()
    print(result)
    print("\nMonitor with:")
    print(f"  modal app logs {APP_NAME} -f")
    print("\nAfter completion, inspect:")
    print(
        f"  modal volume get {WORK_VOLUME_NAME} "
        "UGP_V7_FULL4B/V10R_STRICT_STAGED/stage_C_rgbd_incremental_v1/"
        "spatial_temporal_repair_v1/tables ./C7R_tables"
    )
