"""Build the V7F offline bundle on a Linux x86_64 computer with Python 3.10 and working Internet.

It must not be run on the GPU server: that server has a broken package-download certificate route.
The output folder is copied to V7F_4080_Run/offline_packages on the server.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd() / "V7F_4080_OFFLINE_BUNDLE"
WHEELHOUSE = ROOT / "wheelhouse"
SOURCE = ROOT / "source"
EXTERNAL = ROOT / "external" / "v7_external"
HF_HOME = ROOT / "huggingface_cache"

MAIN_REQUIREMENTS = [
    "numpy==1.26.4", "timm>=1.0.20", "huggingface_hub>=0.33", "safetensors",
    "gdown", "rosbags", "scipy", "scikit-learn", "pandas", "matplotlib",
    "Pillow", "opencv-python-headless", "einops", "addict", "psutil", "natsort", "tqdm",
    "spconv-cu121==2.3.8",
]

def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(map(str, command)))
    subprocess.check_call(list(map(str, command)), env=env)

def clone(url: str, destination: Path, *extra: str) -> None:
    if destination.exists():
        print("Reusing", destination)
        return
    run(["git", "clone", "--depth", "1", *extra, url, str(destination)])

def main() -> int:
    if sys.version_info[:2] != (3, 10):
        raise SystemExit("Run this with Linux CPython 3.10, the same version as v7f4080.")
    if sys.platform != "linux":
        raise SystemExit("This bundle must contain Linux wheels, so run it on Linux.")

    for path in (WHEELHOUSE, SOURCE, EXTERNAL, HF_HOME):
        path.mkdir(parents=True, exist_ok=True)

    # `uv run --with huggingface_hub --with gdown` supplies these two tools in
    # a disposable virtual environment. Do not install into uv's managed
    # CPython installation: it correctly rejects that under PEP 668.
    run([sys.executable, "-m", "pip", "download", "--only-binary=:all:", "--dest", str(WHEELHOUSE), *MAIN_REQUIREMENTS])
    # xFormers 0.0.28.post3 is the CPython-3.10 build for torch 2.5.1. Do not let
    # pip replace the server's existing torch 2.5.1+cu121 during bundle creation.
    run([sys.executable, "-m", "pip", "download", "--no-deps", "--dest", str(WHEELHOUSE), "xformers==0.0.28.post3"])
    run([
        sys.executable, "-m", "pip", "download", "--no-deps", "--dest", str(WHEELHOUSE),
        "torch-scatter==2.1.2+pt25cu121", "--find-links",
        "https://data.pyg.org/whl/torch-2.5.0+cu121.html",
    ])

    (ROOT / "requirements.txt").write_text("\n".join([
        *MAIN_REQUIREMENTS,
        "xformers==0.0.28.post3",
        "torch-scatter==2.1.2+pt25cu121",
    ]) + "\n")

    clone("https://github.com/castacks/tartanairpy.git", SOURCE / "tartanairpy")
    clone("https://github.com/Pointcept/Utonia.git", EXTERNAL / "Utonia")
    clone("https://github.com/ANYbotics/anymal_d_simple_description.git", EXTERNAL / "anymal_d_simple_description")
    clone("https://github.com/princeton-vl/WAFT.git", EXTERNAL / "WAFT", "--recurse-submodules", "--branch", "waftv2")

    from gdown import download
    waft_ckpt = EXTERNAL / "WAFT" / "ckpts" / "tar-c-t.pth"
    waft_ckpt.parent.mkdir(parents=True, exist_ok=True)
    if not waft_ckpt.exists():
        download(id="1CxzBQx0iSg6AyIgt6MF0ROlF_cAeZLPC", output=str(waft_ckpt), quiet=False)

    depth = EXTERNAL / "WAFT" / "depth-anything-ckpts" / "depth_anything_v2_vits.pth"
    depth.parent.mkdir(parents=True, exist_ok=True)
    if not depth.exists():
        run(["curl", "--fail", "--location", "--output", str(depth),
             "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth"])

    from huggingface_hub import snapshot_download
    cache = HF_HOME / "hub"
    for repo in [
        "timm/vit_large_patch16_dinov3.lvd1689m",
        "timm/convnext_large.dinov3_lvd1689m",
        "Pointcept/Utonia",
    ]:
        print("Downloading Hugging Face snapshot:", repo)
        snapshot_download(repo_id=repo, cache_dir=str(cache))

    shutil.make_archive(str(ROOT), "zip", ROOT)
    print("\nBundle created:", ROOT.with_suffix(".zip"))
    print("Copy that ZIP to the server's V7F_4080_Run folder, then unzip it into offline_packages/.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
