from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import ShardedWorldModelDataset, validate_sample

if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("manifest"); ap.add_argument("--n",type=int,default=10); a=ap.parse_args()
    ds=ShardedWorldModelDataset(a.manifest,validate_first=0)
    for i in range(min(a.n,len(ds))): validate_sample(ds[i])
    print(f"OK: validated {min(a.n,len(ds))}/{len(ds)} samples from {a.manifest}")
