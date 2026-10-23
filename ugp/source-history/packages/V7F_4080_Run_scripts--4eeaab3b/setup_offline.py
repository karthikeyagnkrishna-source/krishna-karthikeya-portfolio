"""Install only packages already copied into offline_packages; it never contacts the Internet."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

RUN_ROOT = Path(__file__).resolve().parent
BUNDLE = RUN_ROOT / "offline_packages"
WHEELHOUSE = BUNDLE / "wheelhouse"
REQUIREMENTS = BUNDLE / "requirements.txt"
SOURCE = BUNDLE / "source"

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

    print("Offline installation completed. Next: python preflight.py")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
