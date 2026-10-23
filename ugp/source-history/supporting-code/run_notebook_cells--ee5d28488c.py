#!/usr/bin/env python3
"""Execute a Jupyter notebook's code cells in one persistent Python process.

This runner is intentionally small and dependency-free.  It is used when the
scientific environment is complete but installing a Jupyter server would add
unnecessary moving parts.  Cell 2 of this specific notebook is a Colab-facing
online installer; on the server it is replaced by an equivalent offline
bootstrap that points at the assets already validated by preflight.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import types
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def install_display_shim() -> None:
    """Supply only the IPython.display API used by this notebook."""

    if "IPython" in sys.modules:
        return

    ipython = types.ModuleType("IPython")
    display_mod = types.ModuleType("IPython.display")

    def display(*objects, **_kwargs):
        for obj in objects:
            try:
                print(obj)
            except Exception:
                print(repr(obj))

    class HTML(str):
        pass

    ipython.get_ipython = lambda: None
    ipython.version_info = (0, 0, 0)
    display_mod.display = display
    display_mod.HTML = HTML
    ipython.display = display_mod
    sys.modules["IPython"] = ipython
    sys.modules["IPython.display"] = display_mod


SERVER_BOOTSTRAP = r'''
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

NOTEBOOK_WALL_T0 = time.perf_counter()
WORK_ROOT = Path(os.environ["V7_WORK_ROOT"]).expanduser()
EXT = WORK_ROOT / "v7_external"
UTONIA_PY = EXT / "utonia_py310" / "bin" / "python"
ANYMAL_D_URDF = EXT / "anymal_d_simple_description" / "urdf" / "anymal.urdf"
WAFT = EXT / "WAFT"
cfg = WAFT / "config" / "a1" / "tar-c-t.json"

def run(cmd, **kwargs):
    print(" ".join(map(str, cmd)), flush=True)
    subprocess.check_call(list(map(str, cmd)), **kwargs)

required = {
    "Utonia worker": UTONIA_PY,
    "ANYmal-D URDF": ANYMAL_D_URDF,
    "WAFT config": cfg,
    "WAFT checkpoint": WAFT / "ckpts" / "tar-c-t.pth",
    "Depth checkpoint": WAFT / "depth-anything-ckpts" / "depth_anything_v2_vits.pth",
}
missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
if missing:
    raise RuntimeError("Server bootstrap is incomplete:\n" + "\n".join(missing))

run([
    str(UTONIA_PY), "-c",
    "import sys, torch, spconv.pytorch, torch_scatter, utonia; "
    "print('Utonia worker Python:', sys.version.split()[0]); "
    "print('Utonia worker Torch :', torch.__version__); "
    "print('Utonia worker CUDA  :', torch.version.cuda); "
    "print('Utonia import OK')",
])
print("SERVER OFFLINE BOOTSTRAP COMPLETE", flush=True)
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()

    notebook_path = args.notebook.resolve()
    state_path = args.state.resolve()
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    os.environ.setdefault("MPLBACKEND", "Agg")
    install_display_shim()

    scope = {
        "__name__": "__main__",
        "__file__": str(notebook_path),
    }
    started = utc_now()
    state = {
        "status": "running",
        "pid": os.getpid(),
        "notebook": str(notebook_path),
        "started_utc": started,
        "current_cell": None,
        "completed_cells": [],
    }
    atomic_json(state_path, state)
    print(f"NOTEBOOK START {started}", flush=True)
    print(f"NOTEBOOK PATH {notebook_path}", flush=True)

    try:
        for index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue

            source = "".join(cell.get("source", []))
            if index == 2:
                source = SERVER_BOOTSTRAP
                print("CELL 2 uses validated server-offline bootstrap", flush=True)

            cell_started = time.time()
            state["current_cell"] = index
            state["cell_started_utc"] = utc_now()
            atomic_json(state_path, state)
            print(f"\n===== CELL {index} START {state['cell_started_utc']} =====", flush=True)
            exec(compile(source, f"{notebook_path.name}:cell_{index}", "exec"), scope, scope)
            elapsed = time.time() - cell_started
            state["completed_cells"].append(index)
            state["last_completed_cell"] = index
            state["last_completed_utc"] = utc_now()
            state["last_cell_elapsed_seconds"] = elapsed
            atomic_json(state_path, state)
            print(f"===== CELL {index} COMPLETE ({elapsed:.1f}s) =====", flush=True)

        state["status"] = "complete"
        state["current_cell"] = None
        state["completed_utc"] = utc_now()
        atomic_json(state_path, state)
        print(f"NOTEBOOK COMPLETE {state['completed_utc']}", flush=True)
        return 0
    except BaseException as exc:
        state["status"] = "failed"
        state["failed_utc"] = utc_now()
        state["error_type"] = type(exc).__name__
        state["error"] = str(exc)
        atomic_json(state_path, state)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
