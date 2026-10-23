"""Install only packages already copied into offline_packages; it never contacts the Internet."""
from __future__ import annotations

import subprocess
import sys
import shutil
from pathlib import Path

RUN_ROOT = Path(__file__).resolve().parent
BUNDLE = RUN_ROOT / "offline_packages"
WHEELHOUSE = BUNDLE / "wheelhouse"
REQUIREMENTS = BUNDLE / "requirements.txt"
SOURCE = BUNDLE / "source"
EXTERNAL = BUNDLE / "external" / "v7_external"
WORK_EXTERNAL = RUN_ROOT / "storage" / ".work" / "v7_external"
HF_CACHE = BUNDLE / "huggingface_cache" / "hub"
WORK_HF_CACHE = RUN_ROOT / "storage" / ".cache" / "huggingface" / "hub"

def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.check_call(command)

def main() -> int:
    if not REQUIREMENTS.exists():
        print("Missing offline_packages/requirements.txt")
        return 2
    if not any(WHEELHOUSE.glob("*.whl")):
        print("No wheel files found in offline_packages/wheelhouse")
        return 2

    # --no-index makes an accidental online package download impossible.
    run([
        sys.executable, "-m", "pip", "install", "--no-index",
        "--find-links", str(WHEELHOUSE), "-r", str(REQUIREMENTS),
    ])

    # Repositories copied into source/ are installed only from disk.
    for project in sorted(SOURCE.iterdir()) if SOURCE.exists() else []:
        if project.is_dir() and (project / "pyproject.toml").exists() or project.is_dir() and (project / "setup.py").exists():
            run([sys.executable, "-m", "pip", "install", "--no-index", "--no-build-isolation", str(project)])

    if EXTERNAL.exists():
        print(f"Copying offline model/source files to {WORK_EXTERNAL}")
        shutil.copytree(EXTERNAL, WORK_EXTERNAL, dirs_exist_ok=True)
        utonia = WORK_EXTERNAL / "Utonia"
        if (utonia / "setup.py").exists():
            run([sys.executable, "-m", "pip", "install", "--no-index", "--no-build-isolation", str(utonia)])
    else:
        print("No offline_packages/external/v7_external directory was supplied.")

    if HF_CACHE.exists():
        print(f"Copying Hugging Face model cache to {WORK_HF_CACHE}")
        shutil.copytree(HF_CACHE, WORK_HF_CACHE, dirs_exist_ok=True)
    else:
        print("No offline Hugging Face cache was supplied.")

    print("Offline installation completed. Next: python preflight.py")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
