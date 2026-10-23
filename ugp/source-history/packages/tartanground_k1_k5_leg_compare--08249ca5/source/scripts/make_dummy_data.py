from __future__ import annotations
import argparse, json
from pathlib import Path
import torch


def make_sample(K=3, T=8):
    return {
        "z_window": torch.randn(4,64,256),
        "leg_hist_t": torch.randn(4,T,24),
        "leg_hist_tm1": torch.randn(4,T,24),
        "leg_hist_tm2": torch.randn(4,T,24),
        "body_hist_t": torch.randn(T,18),
        "body_action_seq": torch.randn(K,4),
        "leg_action_seq": torch.randn(K,4,4),
        "leg_query_features_seq": torch.randn(K,4,24),
        "future_action_seq": torch.randn(K,36),
        "physics_seq": torch.randn(K,47),
        "horizon_seq": torch.full((K,1),0.1),
        "lowlevel_cmd_seq": torch.randn(K,4,3),
        "err_seq": torch.randn(K,4,3),
        "jointbody_baseline_seq": torch.randn(K,36),
        "jointbody_std": torch.rand(36)+0.5,
        "tau_baseline_seq": torch.randn(K,4,3),
        "tau_std": torch.rand(3)+0.5,
        "force_baseline_seq": torch.randn(K,4,3)*20,
        "force_std": torch.rand(3)*10+10,
        "target_z_seq": torch.randn(K,64,256),
        "target_contact_seq": torch.randint(0,2,(K,4)).float(),
        "target_load_class_seq": torch.randint(0,4,(K,4)),
        "target_jointbody_seq": torch.randn(K,36),
        "target_tau_seq": torch.randn(K,4,3),
        "target_force_seq": torch.randn(K,4,3)*40,
        "target_phase_seq": torch.randint(0,4,(K,4)),
    }


def write_split(root: Path, name: str, n: int, shard_size: int):
    rows=[]
    split=root/name; split.mkdir(parents=True,exist_ok=True)
    for s0 in range(0,n,shard_size):
        samples=[make_sample() for _ in range(min(shard_size,n-s0))]
        path=(split/f"shard_{s0//shard_size:03d}.pt").resolve()
        torch.save({"samples":samples},path)
        for i in range(len(samples)): rows.append({"path":str(path),"index":i})
    manifest=root/f"{name}_manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(x) for x in rows)+"\n")
    print(manifest)


if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default="dummy_data"); ap.add_argument("--train",type=int,default=16); ap.add_argument("--val",type=int,default=4)
    a=ap.parse_args(); root=Path(a.out).resolve(); root.mkdir(parents=True,exist_ok=True)
    write_split(root,"train",a.train,4); write_split(root,"val",a.val,4)
