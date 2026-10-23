from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict
import yaml


@dataclass
class ModelConfig:
    # Merged visual latent state: DINO / ConvNeXt / geometry / flow.
    num_modalities: int = 4
    tokens_per_modality: int = 16
    latent_dim: int = 256
    num_visual_frames: int = 4

    # 9 Feel-WM groups:
    # Z(t-3), HL@Z(t-3), HR@Z(t-3), Z(t-2), FL@Z(t-2), FR@Z(t-2), Z(t-1), Z(t), Body@Z(t)
    num_groups: int = 9

    # Requested high-capacity world transformer.
    world_d_model: int = 800       # 800 / 10 = 80 dims/head
    world_heads: int = 10
    world_layers: int = 8
    world_ffn_ratio: int = 7

    # Optional independent leg-local processing between FiLM and the shared
    # world transformer. Defaults preserve the original baseline architecture.
    use_leg_pretransformers: bool = False
    leg_pre_d_model: int = 384
    leg_pre_heads: int = 8
    leg_pre_layers: int = 4
    leg_pre_ffn_ratio: int = 4
    leg_pre_residual_scale: float = 0.1

    # V7 history encoders.
    hist_d_model: int = 128
    hist_heads: int = 4
    hist_layers: int = 2
    hist_ffn_ratio: int = 4

    # V7 terrain query blocks.
    query_dim: int = 256
    query_heads: int = 8
    query_layers: int = 4
    query_ffn_ratio: int = 4

    # Inputs.
    leg_hist_dim: int = 24         # proprio12 + U6 + low-level3 + error3
    body_hist_dim: int = 18        # body12 + U6
    leg_query_dim: int = 24        # current proprio12 + current U6 + foot kinematics6
    body_film_action_dim: int = 4
    leg_film_action_dim: int = 4
    future_action_dim: int = 36
    physics_dim: int = 47

    # V7 / V7-F.
    num_legs: int = 4
    num_load_regimes: int = 4
    num_contact_phases: int = 4
    force_ctx_dim: int = 1285
    v7f_dim: int = 256
    v7f_heads: int = 8
    v7f_layers: int = 2
    v7f_experts: int = 4

    dropout: float = 0.10
    eps: float = 1e-6

    @property
    def tokens_per_frame(self) -> int:
        return self.num_modalities * self.tokens_per_modality

    @property
    def world_tokens(self) -> int:
        return self.num_groups * self.tokens_per_frame


@dataclass
class TrainConfig:
    train_manifest: str = "data/train_manifest.jsonl"
    val_manifest: str = "data/val_manifest.jsonl"
    output_dir: str = "runs/merged_feelwm_v7"

    seed: int = 781
    epochs: int = 30
    batch_size: int = 2
    grad_accum_steps: int = 4
    num_workers: int = 6
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True

    lr: float = 2e-4
    min_lr_ratio: float = 0.05
    weight_decay: float = 0.05
    warmup_steps: int = 1000
    grad_clip: float = 1.0

    # RTX 4080: bf16 is normally the best default.
    amp_dtype: str = "bf16"       # bf16 | fp16 | fp32
    tf32: bool = True
    compile: bool = False

    log_every: int = 10
    val_every_steps: int = 1000
    checkpoint_every_steps: int = 250
    keep_last_checkpoints: int = 4
    resume: str = "auto"          # auto | none | /path/to/checkpoint.pt
    max_steps: int | None = None   # exact optimizer-step cap; None keeps epoch-only behavior
    init_weights_only: str | None = None  # checkpoint path; fresh optimizer/scheduler
    teacher_forcing: bool = False
    teacher_forcing_start_probability: float = 0.5
    teacher_forcing_end_probability: float = 0.5
    horizon_lambdas: tuple[float, ...] = ()
    indexed_data: bool = False

    # Optional targeted continuation. Empty prefixes keep the original
    # end-to-end training behavior.
    trainable_prefixes: tuple[str, ...] = ()
    checkpoint_metric: str = "total"
    best_checkpoint_name: str = "best.pt"
    validation_max_batches: int | None = 50

    # The recurrent model step can be 0.1 s while proprio is sampled faster.
    # If your prepared histories are already on the model-step grid, keep 1.
    # If histories are 100 Hz and world steps are 10 Hz, use 10.
    history_shift_samples: int = 1

    # Loss weights.
    w_dino: float = 1.0
    w_conv: float = 1.0
    w_geom: float = 1.0
    w_flow: float = 1.0
    w_contact: float = 0.50
    w_load: float = 0.25
    w_jointbody: float = 1.0
    w_tau: float = 1.0
    w_base_force: float = 1.0
    w_v7f_force: float = 1.0

    # Base V7 magnitude/direction auxiliary head. These outputs are part of the
    # architecture and must be supervised so every DDP parameter participates.
    v7_w_logmag: float = 0.10
    v7_w_dir: float = 0.02

    # V7-F specialist weights retained from the source force-specialist design.
    v7f_w_vec: float = 2.00
    v7f_w_mag: float = 1.60
    v7f_w_dir: float = 0.10
    v7f_w_phase: float = 0.25
    v7f_w_temp: float = 0.45
    v7f_w_impulse: float = 0.35
    v7f_w_anchor: float = 0.015
    v7f_w_gate: float = 0.005
    force_contact_threshold: float = 20.0
    rollout_dt: float = 0.1


def load_config(path: str | Path) -> tuple[ModelConfig, TrainConfig, Dict[str, Any]]:
    path = Path(path)
    raw: Dict[str, Any] = yaml.safe_load(path.read_text())
    model_cfg = ModelConfig(**raw.get("model", {}))
    train_cfg = TrainConfig(**raw.get("train", {}))
    return model_cfg, train_cfg, raw


def config_to_dict(model_cfg: ModelConfig, train_cfg: TrainConfig) -> Dict[str, Any]:
    return {"model": asdict(model_cfg), "train": asdict(train_cfg)}
