"""Check that this project can start without downloading anything."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

RUN_ROOT = Path(__file__).resolve().parent
STORAGE_ROOT = RUN_ROOT / "storage"
WORK_ROOT = STORAGE_ROOT / ".work"
EXT = WORK_ROOT / "v7_external"

REQUIRED_MODULES = {
    "torch": "PyTorch with CUDA",
    "torchvision": "TorchVision",
    "timm": "timm",
    "huggingface_hub": "Hugging Face Hub",
    "safetensors": "SafeTensors",
    "gdown": "gdown",
    "rosbags": "rosbags",
    "scipy": "SciPy",
    "sklearn": "scikit-learn",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "PIL": "Pillow",
    "cv2": "OpenCV",
    "einops": "einops",
    "addict": "addict",
    "psutil": "psutil",
    "natsort": "natsort",
    "tqdm": "tqdm",
    "xformers": "xformers",
    "tartanair": "tartanairpy",
}

def missing_modules() -> list[str]:
    return [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]

def main() -> int:
    print("Python:", sys.executable)
    print("Project:", RUN_ROOT)
    print("Storage:", STORAGE_ROOT)
    STORAGE_ROOT.mkdir(parents=True, exist_ok=True)

    missing = missing_modules()
    if missing:
        print("\nMissing main-environment packages:")
        for name in missing:
            print(f"  - {REQUIRED_MODULES[name]} ({name})")
    else:
        print("Main-environment Python packages: OK")

    try:
        import torch
        print("\nTorch:", torch.__version__)
        print("Torch CUDA build:", torch.version.cuda)
        print("CUDA available:", torch.cuda.is_available())
        print("Visible GPUs:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            missing.append("two CUDA-visible GPUs")
    except Exception as exc:
        print("\nPyTorch CUDA check failed:", repr(exc))
        missing.append("working PyTorch CUDA")

    required_paths = {
        "Utonia source": EXT / "Utonia",
        "Utonia Python": EXT / "utonia_py310" / "bin" / "python",
        "ANYmal-D URDF": EXT / "anymal_d_simple_description" / "urdf" / "anymal.urdf",
        "WAFT config": EXT / "WAFT" / "config" / "a1" / "tar-c-t.json",
        "WAFT checkpoint": EXT / "WAFT" / "ckpts" / "tar-c-t.pth",
        "Depth checkpoint": EXT / "WAFT" / "depth-anything-ckpts" / "depth_anything_v2_vits.pth",
    }
    absent = [label for label, path in required_paths.items() if not path.exists()]
    if absent:
        print("\nMissing copied model/source files:")
        for label in absent:
            print("  -", label)
    else:
        print("External source and checkpoints: OK")

    if missing or absent:
        print("\nPREFLIGHT FAILED. Run setup_offline.py after the offline bundle is copied here.")
        return 1
    print("\nPREFLIGHT PASSED. Start: python train.py")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
