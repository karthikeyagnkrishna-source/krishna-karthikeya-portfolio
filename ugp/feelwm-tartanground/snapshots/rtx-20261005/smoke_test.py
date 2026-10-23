from __future__ import annotations
import torch
from config import ModelConfig, TrainConfig
from model import MergedFeelWMV7
from losses import compute_losses


def sample(B=1,K=2,T=5):
    dev="cuda" if torch.cuda.is_available() else "cpu"
    def r(*s): return torch.randn(*s,device=dev)
    b={
        "z_window":r(B,4,64,256), "leg_hist_t":r(B,4,T,24), "leg_hist_tm1":r(B,4,T,24), "leg_hist_tm2":r(B,4,T,24),
        "body_hist_t":r(B,T,18), "body_action_seq":r(B,K,4), "leg_action_seq":r(B,K,4,4),
        "leg_query_features_seq":r(B,K,4,24), "future_action_seq":r(B,K,36), "physics_seq":r(B,K,47),
        "horizon_seq":torch.full((B,K,1),0.1,device=dev), "lowlevel_cmd_seq":r(B,K,4,3), "err_seq":r(B,K,4,3),
        "jointbody_baseline_seq":r(B,K,36), "jointbody_std":torch.ones(B,36,device=dev),
        "tau_baseline_seq":r(B,K,4,3), "tau_std":torch.ones(B,3,device=dev),
        "force_baseline_seq":r(B,K,4,3), "force_std":torch.ones(B,3,device=dev)*10,
        "target_z_seq":r(B,K,64,256), "target_contact_seq":torch.randint(0,2,(B,K,4),device=dev).float(),
        "target_load_class_seq":torch.randint(0,4,(B,K,4),device=dev), "target_jointbody_seq":r(B,K,36),
        "target_tau_seq":r(B,K,4,3), "target_force_seq":r(B,K,4,3), "target_phase_seq":torch.randint(0,4,(B,K,4),device=dev),
    }
    return b


if __name__ == "__main__":
    # Tiny-capacity test keeps CPU/GPU verification quick while preserving all tensor wiring and 10-head divisibility.
    cfg=ModelConfig(world_d_model=80,world_heads=10,world_layers=2,world_ffn_ratio=2,query_layers=1,hist_layers=1,v7f_layers=1)
    tcfg=TrainConfig(amp_dtype="fp32")
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m=MergedFeelWMV7(cfg).to(dev)
    b=sample()
    o=m.rollout(b,history_shift=1)
    losses=compute_losses(o,b,cfg,tcfg)
    assert o["v7f"]["force"].shape==(1,2,4,3)
    assert o["steps"][0]["z_next"].shape==(1,64,256)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    print("SMOKE TEST PASSED")
    print("force",tuple(o["v7f"]["force"].shape),"z_next",tuple(o["steps"][0]["z_next"].shape),"loss",float(losses["total"].detach().cpu()))
