from __future__ import annotations
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


# Leg order is fixed everywhere in this project.
FL, FR, HL, HR = 0, 1, 2, 3


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0, norm_in: bool = False):
        super().__init__()
        layers: List[nn.Module] = []
        if norm_in:
            layers.append(nn.LayerNorm(in_dim))
        layers += [nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ratio: int, dropout: float):
        super().__init__()
        hidden = d_model * ratio
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, ffn_ratio: int, dropout: float):
        super().__init__()
        assert d_model % heads == 0, f"d_model={d_model} must be divisible by heads={heads}"
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_ratio, dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + h
        x = x + self.ffn(self.ln2(x))
        return x


class CausalHistoryEncoder(nn.Module):
    """Faithful V7 history encoder: projection -> 2-layer causal Transformer -> last token."""
    def __init__(self, input_dim: int, cfg: ModelConfig, max_steps: int = 512):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, cfg.hist_d_model)
        self.pos = nn.Parameter(torch.randn(1, max_steps, cfg.hist_d_model) * 0.02)
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(cfg.hist_d_model, cfg.hist_heads, cfg.hist_ffn_ratio, cfg.dropout)
            for _ in range(cfg.hist_layers)
        ])
        self.norm = nn.LayerNorm(cfg.hist_d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [B,T,D]
        T = x.shape[1]
        if T > self.pos.shape[1]:
            raise ValueError(f"history length {T} exceeds max {self.pos.shape[1]}")
        h = self.input_proj(x) + self.pos[:, :T]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        for block in self.blocks:
            h = block(h, mask)
        return self.norm(h)[:, -1]


class QueryFusionBlockAligned(nn.Module):
    """
    Five V7 queries, but each query cross-attends to its temporally aligned 64-token bank:
      global -> Body@Z_t (G8)
      FL     -> FL@Z_t-2 (G4)
      FR     -> FR@Z_t-2 (G5)
      HL     -> HL@Z_t-3 (G1)
      HR     -> HR@Z_t-3 (G2)
    Then the 5 queries self-attend exactly as V7 does.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.query_dim
        self.qn = nn.LayerNorm(d)
        self.vn = nn.LayerNorm(d)
        self.cross = nn.MultiheadAttention(d, cfg.query_heads, dropout=cfg.dropout, batch_first=True)
        self.sn = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, cfg.query_heads, dropout=cfg.dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, cfg.query_ffn_ratio * d), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.query_ffn_ratio * d, d)
        )

    def forward(self, q: torch.Tensor, banks: torch.Tensor) -> torch.Tensor:
        # q [B,5,256], banks [B,5,64,256]
        B, Q, N, D = banks.shape
        q_flat = self.qn(q).reshape(B * Q, 1, D)
        v_flat = self.vn(banks).reshape(B * Q, N, D)
        a, _ = self.cross(q_flat, v_flat, v_flat, need_weights=False)
        q = q + a.reshape(B, Q, D)
        qn = self.sn(q)
        a, _ = self.self_attn(qn, qn, qn, need_weights=False)
        q = q + a
        return q + self.ffn(q)


def build_physical_time_mask(tokens_per_group: int, device: torch.device) -> torch.Tensor:
    # G0,G1,G2=t-3; G3,G4,G5=t-2; G6=t-1; G7,G8=t
    group_times = [0, 0, 0, 1, 1, 1, 2, 3, 3]
    times = torch.tensor([t for g in group_times for t in [g] * tokens_per_group], device=device)
    # True means forbidden in torch MultiheadAttention.
    return times[None, :] > times[:, None]


class FiLMGenerator(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 512), nn.GELU(), nn.Linear(512, 2 * cfg.world_d_model)
        )
        # Identity FiLM at initialization, like the original idea.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.net(c).chunk(2, dim=-1)


class LegPreTransformer(nn.Module):
    """Independent per-leg token transformer with a small residual update."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.input_proj = nn.Linear(cfg.world_d_model, cfg.leg_pre_d_model)
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(
                cfg.leg_pre_d_model, cfg.leg_pre_heads,
                cfg.leg_pre_ffn_ratio, cfg.dropout,
            )
            for _ in range(cfg.leg_pre_layers)
        ])
        self.output_proj = nn.Linear(cfg.leg_pre_d_model, cfg.world_d_model)
        self.residual_scale = cfg.leg_pre_residual_scale

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(tokens)
        for block in self.blocks:
            h = block(h)
        return tokens + self.residual_scale * self.output_proj(h)


def film(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + gamma[:, None]) + beta[:, None]


class V7ForceSpecialist(nn.Module):
    """V7-F specialist retained from the force-specialist notebook."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.v7f_dim
        self.cfg = cfg
        self.in_proj = nn.Sequential(
            nn.LayerNorm(cfg.force_ctx_dim), nn.Linear(cfg.force_ctx_dim, d), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d)
        )
        self.horizon_blocks = nn.ModuleList([
            SelfAttentionBlock(d, cfg.v7f_heads, 4, cfg.dropout) for _ in range(cfg.v7f_layers)
        ])
        self.phase_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 128), nn.GELU(), nn.Linear(128, cfg.num_contact_phases))
        self.experts = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 192), nn.GELU(), nn.Dropout(.08),
                          nn.Linear(192, 96), nn.GELU(), nn.Linear(96, 3))
            for _ in range(cfg.v7f_experts)
        ])
        gate_dim = d + cfg.num_contact_phases + cfg.num_load_regimes + 1
        self.expert_gate = nn.Sequential(nn.LayerNorm(gate_dim), nn.Linear(gate_dim, 128), nn.GELU(), nn.Linear(128, cfg.v7f_experts))
        self.direct_corr = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 192), nn.GELU(), nn.Linear(192, 3))
        self.magdir = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 192), nn.GELU(), nn.Linear(192, 4))
        self.fusion_gate = nn.Sequential(nn.LayerNorm(gate_dim), nn.Linear(gate_dim, 96), nn.GELU(), nn.Linear(96, 1))

        # Start exactly at the base V7 force prediction.
        for mod in list(self.experts) + [self.direct_corr, self.magdir]:
            last = [m for m in mod.modules() if isinstance(m, nn.Linear)][-1]
            nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
        last = [m for m in self.fusion_gate.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor, base_force: torch.Tensor, force_std: torch.Tensor,
                pcontact: torch.Tensor, regime: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x [B,K,4,1285]
        B, K, L, _ = x.shape
        h = self.in_proj(x)
        ht = h.permute(0, 2, 1, 3).reshape(B * L, K, self.cfg.v7f_dim)
        causal = torch.triu(torch.ones(K, K, dtype=torch.bool, device=x.device), diagonal=1)
        for block in self.horizon_blocks:
            ht = block(ht, causal)
        h = ht.reshape(B, L, K, self.cfg.v7f_dim).permute(0, 2, 1, 3).contiguous()

        phase_logits = self.phase_head(h)
        phase_prob = phase_logits.softmax(-1)
        gate_in = torch.cat([h, phase_prob, regime, pcontact.unsqueeze(-1)], dim=-1)
        expert_w = self.expert_gate(gate_in).softmax(-1)
        exp = torch.stack([e(h) for e in self.experts], dim=-2)
        moe_corr = (expert_w.unsqueeze(-1) * exp).sum(-2)
        direct_corr = self.direct_corr(h)
        corr_norm = moe_corr + direct_corr

        # Normalized specialist correction -> physical force units.
        direct_force = base_force + corr_norm * force_std[:, None, None, :]

        md = self.magdir(h)
        base_mag = torch.linalg.norm(base_force, dim=-1).clamp_min(1e-5)
        base_dir = F.normalize(base_force, dim=-1, eps=self.cfg.eps)
        mag = torch.expm1(torch.log1p(base_mag) + 0.75 * torch.tanh(md[..., 0])).clamp_min(0.0)
        direction = F.normalize(base_dir + 0.20 * torch.tanh(md[..., 1:4]), dim=-1, eps=self.cfg.eps)
        magdir_force = mag.unsqueeze(-1) * direction

        alpha = torch.sigmoid(self.fusion_gate(gate_in))
        final_force = alpha * direct_force + (1.0 - alpha) * magdir_force
        return {
            "force": final_force,
            "phase_logits": phase_logits,
            "phase_prob": phase_prob,
            "expert_w": expert_w,
            "corr_norm": corr_norm,
            "direct_force": direct_force,
            "magdir_force": magdir_force,
            "fusion_alpha": alpha.squeeze(-1),
        }


class MergedFeelWMV7(nn.Module):
    """
    Input visual latents are the frozen/stable post-Perceiver V7 branch latents:
      [DINO 16x256, ConvNeXt 16x256, Geometry 16x256, Flow 16x256] => 64x256 / frame.

    Critical timing rule implemented literally:
      * ALL four leg FiLM branches use CURRENT rollout-step action a_t.
      * FL/FR FiLM condition Z_(t-2).
      * HL/HR FiLM condition Z_(t-3).
      * Body FiLM conditions Z_t.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        assert cfg.world_d_model % cfg.world_heads == 0
        assert cfg.tokens_per_frame == 64, "Current architecture expects 4 modalities x 16 tokens"
        assert cfg.num_groups == 9

        # V7 causal history encoders.
        self.leg_encoder = CausalHistoryEncoder(cfg.leg_hist_dim, cfg)
        self.body_encoder = CausalHistoryEncoder(cfg.body_hist_dim, cfg)
        self.leg_identity = nn.Embedding(cfg.num_legs, cfg.hist_d_model)

        # Feel-WM action encoders. The current action a_t is used for all 4 legs.
        self.leg_action_encoder = MLP(cfg.leg_film_action_dim, 128, 128, cfg.dropout)
        self.body_action_encoder = MLP(cfg.body_film_action_dim, 128, 128, cfg.dropout)
        self.leg_cond_proj = nn.Linear(cfg.hist_d_model, 128)
        self.body_cond_proj = nn.Linear(cfg.hist_d_model, 128)

        self.film_fl = FiLMGenerator(cfg)
        self.film_fr = FiLMGenerator(cfg)
        self.film_hl = FiLMGenerator(cfg)
        self.film_hr = FiLMGenerator(cfg)
        self.film_body = FiLMGenerator(cfg)

        # Stable modality identity in the 64x256 V7 latent bank.
        self.modality_embedding = nn.Parameter(torch.randn(1, cfg.num_modalities, 1, cfg.latent_dim) * 0.02)
        self.visual_to_world = nn.Linear(cfg.latent_dim, cfg.world_d_model)
        self.group_embedding = nn.Parameter(torch.randn(1, cfg.num_groups, 1, cfg.world_d_model) * 0.02)
        self.slot_embedding = nn.Parameter(torch.randn(1, 1, cfg.tokens_per_frame, cfg.world_d_model) * 0.02)
        self.time_embedding = nn.Parameter(torch.randn(1, cfg.num_visual_frames, cfg.world_d_model) * 0.02)

        self.world_blocks = nn.ModuleList([
            SelfAttentionBlock(cfg.world_d_model, cfg.world_heads, cfg.world_ffn_ratio, cfg.dropout)
            for _ in range(cfg.world_layers)
        ])
        self.world_norm = nn.LayerNorm(cfg.world_d_model)

        # Four independent t+1 visual-latent heads.
        self.dino_head = MLP(cfg.world_d_model, 512, cfg.latent_dim, cfg.dropout)
        self.conv_head = MLP(cfg.world_d_model, 512, cfg.latent_dim, cfg.dropout)
        self.geom_head = MLP(cfg.world_d_model, 512, cfg.latent_dim, cfg.dropout)
        self.flow_head = MLP(cfg.world_d_model, 512, cfg.latent_dim, cfg.dropout)

        # World group -> V7 terrain bank.
        self.world_to_query_bank = nn.Sequential(nn.LayerNorm(cfg.world_d_model), nn.Linear(cfg.world_d_model, cfg.query_dim))
        self.global_query = nn.Parameter(torch.randn(1, 1, cfg.query_dim) * 0.02)
        self.global_query_from_body = nn.Linear(cfg.hist_d_model, cfg.query_dim)
        self.leg_query_embed = nn.Embedding(cfg.num_legs, cfg.query_dim)
        self.state_to_query = nn.Sequential(nn.LayerNorm(cfg.leg_query_dim), nn.Linear(cfg.leg_query_dim, cfg.query_dim), nn.GELU(), nn.Linear(cfg.query_dim, cfg.query_dim))
        self.query_blocks = nn.ModuleList([QueryFusionBlockAligned(cfg) for _ in range(cfg.query_layers)])
        self.query_norm = nn.LayerNorm(cfg.query_dim)

        # V7 leg fusion and global context.
        self.leg_visual_fuse = nn.Sequential(
            nn.LayerNorm(128 + 256), nn.Linear(384, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.LayerNorm(256)
        )
        self.context_net = nn.Sequential(
            nn.LayerNorm(4 * 256 + 128 + 256), nn.Linear(1408, 768), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(768, 384), nn.GELU(), nn.LayerNorm(384)
        )
        self.control_proj = nn.Sequential(nn.Linear(cfg.future_action_dim, 128), nn.GELU(), nn.LayerNorm(128))
        self.horizon_proj = nn.Sequential(nn.Linear(1, 64), nn.GELU(), nn.Linear(64, 128), nn.GELU(), nn.LayerNorm(128))
        self.physics_proj = nn.Sequential(nn.Linear(cfg.physics_dim, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.LayerNorm(128))

        self.contact_head = nn.Sequential(nn.Linear(768, 256), nn.GELU(), nn.Linear(256, 4))
        self.regime_head = nn.Sequential(nn.Linear(768, 256), nn.GELU(), nn.Linear(256, 4 * cfg.num_load_regimes))
        self.state_body_head = nn.Sequential(nn.Linear(768, 384), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(384, 36))
        self.tau_force_head = nn.Sequential(
            nn.LayerNorm(cfg.force_ctx_dim), nn.Linear(cfg.force_ctx_dim, 512), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 6)
        )
        self.force_aux_head = nn.Sequential(nn.LayerNorm(cfg.force_ctx_dim), nn.Linear(cfg.force_ctx_dim, 256), nn.GELU(), nn.Linear(256, 4))

        self.v7f = V7ForceSpecialist(cfg)
        # Construct these last: with a shared seed this keeps every baseline
        # parameter initialized identically in the baseline and per-leg models.
        self.leg_pre_transformers = (
            nn.ModuleDict({name: LegPreTransformer(cfg) for name in ("fl", "fr", "hl", "hr")})
            if cfg.use_leg_pretransformers else None
        )

    def add_modality_identity(self, z: torch.Tensor) -> torch.Tensor:
        # z [...,64,256] in fixed branch order [DINO, Conv, Geom, Flow].
        shape = z.shape
        x = z.view(*shape[:-2], self.cfg.num_modalities, self.cfg.tokens_per_modality, self.cfg.latent_dim)
        emb = self.modality_embedding
        while emb.dim() < x.dim():
            emb = emb.unsqueeze(1)
        x = x + emb
        return x.reshape(*shape)

    def encode_leg_histories(self, hist: torch.Tensor) -> torch.Tensor:
        # hist [B,4,T,24] -> [B,4,128]
        B, L, T, D = hist.shape
        h = self.leg_encoder(hist.reshape(B * L, T, D)).reshape(B, L, -1)
        ids = torch.arange(L, device=hist.device)[None].expand(B, -1)
        return h + self.leg_identity(ids)

    def build_world_groups(self, z_window: torch.Tensor, h_tm1: torch.Tensor, h_tm2: torch.Tensor,
                           h_body: torch.Tensor, leg_action_t: torch.Tensor, body_action_t: torch.Tensor) -> torch.Tensor:
        # z_window [B,4,64,256] = [t-3,t-2,t-1,t]
        z_window = self.add_modality_identity(z_window)
        z3, z2, z1, z0 = [self.visual_to_world(z_window[:, i]) for i in range(4)]

        # *** LOCKED RULE: every leg uses action encoding a_t ONLY. ***
        a = self.leg_action_encoder(leg_action_t)  # [B,4,128], current rollout-step action
        a_body = self.body_action_encoder(body_action_t)

        c_fl = self.leg_cond_proj(h_tm1[:, FL]) + a[:, FL]
        c_fr = self.leg_cond_proj(h_tm1[:, FR]) + a[:, FR]
        c_hl = self.leg_cond_proj(h_tm2[:, HL]) + a[:, HL]
        c_hr = self.leg_cond_proj(h_tm2[:, HR]) + a[:, HR]
        c_body = self.body_cond_proj(h_body) + a_body

        g_hl, b_hl = self.film_hl(c_hl)
        g_hr, b_hr = self.film_hr(c_hr)
        g_fl, b_fl = self.film_fl(c_fl)
        g_fr, b_fr = self.film_fr(c_fr)
        g_b, b_b = self.film_body(c_body)

        groups = torch.stack([
            z3,                    # G0 raw Z(t-3)
            film(z3, g_hl, b_hl), # G1 HL uses a_t on terrain t-3
            film(z3, g_hr, b_hr), # G2 HR uses a_t on terrain t-3
            z2,                    # G3 raw Z(t-2)
            film(z2, g_fl, b_fl), # G4 FL uses a_t on terrain t-2
            film(z2, g_fr, b_fr), # G5 FR uses a_t on terrain t-2
            z1,                    # G6 raw Z(t-1)
            z0,                    # G7 raw Z(t)
            film(z0, g_b, b_b),   # G8 Body uses current action on current terrain
        ], dim=1)                  # [B,9,64,800]

        time_ids = torch.tensor([0, 0, 0, 1, 1, 1, 2, 3, 3], device=z_window.device)
        groups = groups + self.group_embedding + self.slot_embedding
        groups = groups + self.time_embedding[:, time_ids].unsqueeze(2)
        return groups

    def world_forward(self, groups: torch.Tensor) -> torch.Tensor:
        B = groups.shape[0]
        groups = self.apply_leg_pretransformers(groups)
        x = groups.reshape(B, self.cfg.world_tokens, self.cfg.world_d_model)
        mask = build_physical_time_mask(self.cfg.tokens_per_frame, x.device)
        for block in self.world_blocks:
            x = block(x, mask)
        return self.world_norm(x).reshape(B, self.cfg.num_groups, self.cfg.tokens_per_frame, self.cfg.world_d_model)

    def apply_leg_pretransformers(self, groups: torch.Tensor) -> torch.Tensor:
        """Process FiLM groups FL/FR/HL/HR immediately before shared attention."""
        blocks = self.leg_pre_transformers
        if blocks is None:
            return groups
        # G4=FL, G5=FR, G1=HL, G2=HR. Clone before indexed writes so the
        # untouched raw/body groups remain bit-identical and autograd is safe.
        groups = groups.clone()
        for name, group_index in (("fl", 4), ("fr", 5), ("hl", 1), ("hr", 2)):
            groups[:, group_index] = blocks[name](groups[:, group_index])
        return groups

    def predict_next_visual(self, H: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Use body-conditioned current group G8; split its 64 slots by modality.
        h = H[:, 8]
        n = self.cfg.tokens_per_modality
        dino = self.dino_head(h[:, 0*n:1*n])
        conv = self.conv_head(h[:, 1*n:2*n])
        geom = self.geom_head(h[:, 2*n:3*n])
        flow = self.flow_head(h[:, 3*n:4*n])
        return {"dino": dino, "conv": conv, "geom": geom, "flow": flow,
                "z_next": torch.cat([dino, conv, geom, flow], dim=1)}

    def aligned_v7_queries(self, H: torch.Tensor, h_body: torch.Tensor, leg_query_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Query order: global, FL, FR, HL, HR.
        B = H.shape[0]
        global_q = self.global_query.expand(B, -1, -1) + self.global_query_from_body(h_body)[:, None]
        ids = torch.arange(4, device=H.device)[None].expand(B, -1)
        foot_q = self.leg_query_embed(ids) + self.state_to_query(leg_query_features)
        q = torch.cat([global_q, foot_q], dim=1)

        # Temporally aligned visual banks from the world transformer.
        # G8 current/body, G4 FL@t-2, G5 FR@t-2, G1 HL@t-3, G2 HR@t-3.
        bank_indices = [8, 4, 5, 1, 2]
        banks = torch.stack([self.world_to_query_bank(H[:, i]) for i in bank_indices], dim=1)  # [B,5,64,256]
        for block in self.query_blocks:
            q = block(q, banks)
        q = self.query_norm(q)
        return q[:, 0], q[:, 1:]

    def forward_step(self, z_window: torch.Tensor, leg_hist_t: torch.Tensor, leg_hist_tm1: torch.Tensor,
                     leg_hist_tm2: torch.Tensor, body_hist_t: torch.Tensor, body_action_t: torch.Tensor,
                     leg_action_t: torch.Tensor, leg_query_features_t: torch.Tensor, future_action: torch.Tensor,
                     physics: torch.Tensor, horizon: torch.Tensor, jointbody_baseline: torch.Tensor,
                     jointbody_std: torch.Tensor, tau_baseline: torch.Tensor, tau_std: torch.Tensor,
                     force_baseline: torch.Tensor, force_std: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = z_window.shape[0]
        h_t = self.encode_leg_histories(leg_hist_t)
        h_tm1 = self.encode_leg_histories(leg_hist_tm1)
        h_tm2 = self.encode_leg_histories(leg_hist_tm2)
        h_body = self.body_encoder(body_hist_t)

        groups = self.build_world_groups(z_window, h_tm1, h_tm2, h_body, leg_action_t, body_action_t)
        H = self.world_forward(groups)
        vis = self.predict_next_visual(H)

        v_global, v_legs = self.aligned_v7_queries(H, h_body, leg_query_features_t)
        legf = self.leg_visual_fuse(torch.cat([h_t, v_legs], dim=-1))
        ctx = self.context_net(torch.cat([legf.reshape(B, -1), h_body, v_global], dim=-1))
        dyn = torch.cat([ctx, self.control_proj(future_action), self.horizon_proj(horizon), self.physics_proj(physics)], dim=-1)

        contact_logits = self.contact_head(dyn)
        contact_prob = torch.sigmoid(contact_logits)
        regime_logits = self.regime_head(dyn).reshape(B, 4, self.cfg.num_load_regimes)
        regime_prob = regime_logits.softmax(-1)

        # V7 state/body residual prediction -> physical target space.
        state_residual = self.state_body_head(dyn)  # [B,36]
        jointbody_pred = jointbody_baseline + state_residual * jointbody_std

        # Per-leg 1285-D V7 force context.
        zh = dyn[:, None].expand(-1, 4, -1)
        force_ctx = torch.cat([zh, legf, v_legs, contact_prob.unsqueeze(-1), regime_prob], dim=-1)
        tf_residual = self.tau_force_head(force_ctx)  # [B,4,6]
        tau_pred = tau_baseline + tf_residual[..., :3] * tau_std[:, None, :]
        base_force = force_baseline + tf_residual[..., 3:] * force_std[:, None, :]
        aux = self.force_aux_head(force_ctx)

        return {
            "H": H,
            **vis,
            "v_global": v_global,
            "v_legs": v_legs,
            "legf": legf,
            "global_context": ctx,
            "dynamics": dyn,
            "contact_logits": contact_logits,
            "contact_prob": contact_prob,
            "regime_logits": regime_logits,
            "regime_prob": regime_prob,
            "state_residual": state_residual,
            "jointbody_pred": jointbody_pred,
            "force_ctx": force_ctx,
            "tf_residual": tf_residual,
            "tau_pred": tau_pred,
            "base_force": base_force,
            "force_aux_logmag": aux[..., 0],
            "force_aux_dir": F.normalize(aux[..., 1:4], dim=-1, eps=self.cfg.eps),
        }

    @staticmethod
    def _append_repeated(hist: torch.Tensor, next_step: torch.Tensor, shift: int) -> torch.Tensor:
        # hist [B,4,T,D] / [B,T,D], next_step [B,4,D] / [B,D]
        shift = max(1, min(int(shift), hist.shape[-2]))
        if hist.dim() == 4:
            rep = next_step.unsqueeze(2).expand(-1, -1, shift, -1)
            return torch.cat([hist[:, :, shift:], rep], dim=2)
        rep = next_step.unsqueeze(1).expand(-1, shift, -1)
        return torch.cat([hist[:, shift:], rep], dim=1)

    @staticmethod
    def _make_leg_history_step(jointbody: torch.Tensor, tau: torch.Tensor, force: torch.Tensor,
                               future_action: torch.Tensor, lowlevel: torch.Tensor, err: torch.Tensor) -> torch.Tensor:
        # jointbody [B,36]; first 24 = 4*(q3,dq3), final 12 = body.
        B = jointbody.shape[0]
        qdq = jointbody[:, :24].reshape(B, 4, 6)
        prop12 = torch.cat([qdq[..., :3], qdq[..., 3:], tau, force], dim=-1)
        u6 = future_action[:, :6][:, None].expand(-1, 4, -1)
        return torch.cat([prop12, u6, lowlevel, err], dim=-1)

    @staticmethod
    def _make_body_history_step(jointbody: torch.Tensor, future_action: torch.Tensor) -> torch.Tensor:
        return torch.cat([jointbody[:, 24:], future_action[:, :6]], dim=-1)

    @staticmethod
    def _batch_stat(batch: Dict[str, torch.Tensor], name: str, width: int, like: torch.Tensor,
                    default: float) -> torch.Tensor:
        value = batch.get(name)
        if value is None:
            return like.new_full((like.shape[0], width), default)
        value = value.to(device=like.device, dtype=like.dtype)
        if value.dim() == 1:
            value = value.unsqueeze(0).expand(like.shape[0], -1)
        return value

    def _recurrent_query(self, jointbody: torch.Tensor, tau: torch.Tensor, force: torch.Tensor,
                         leg_query_exogenous: torch.Tensor, foot_position: torch.Tensor,
                         batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        B = jointbody.shape[0]
        qdq = jointbody[:, :24].reshape(B, 4, 6)
        proprio = torch.cat([qdq, tau, force], dim=-1)
        mean = self._batch_stat(batch, "proprio_mean", 12, jointbody, 0.0)[:, None]
        std = self._batch_stat(batch, "proprio_std", 12, jointbody, 1.0)[:, None].clamp_min(1e-6)
        normalized = (proprio - mean) / std
        # The 12:18 controller features and final planned displacement are
        # exogenous. Foot position is updated from predicted q and a cached
        # local FK Jacobian, never copied from a future measured state.
        return torch.cat([
            normalized,
            leg_query_exogenous[..., 12:18],
            foot_position,
            leg_query_exogenous[..., 21:24],
        ], dim=-1)

    def _recurrent_physics(self, jointbody: torch.Tensor, tau: torch.Tensor, force: torch.Tensor,
                           contact_prob: torch.Tensor, regime_prob: torch.Tensor,
                           batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        B = jointbody.shape[0]
        qdq = jointbody[:, :24].reshape(B, 4, 6)
        p_mean = self._batch_stat(batch, "proprio_mean", 12, jointbody, 0.0)
        p_std = self._batch_stat(batch, "proprio_std", 12, jointbody, 1.0).clamp_min(1e-6)
        b_mean = self._batch_stat(batch, "body_mean", 12, jointbody, 0.0)
        b_std = self._batch_stat(batch, "body_std", 12, jointbody, 1.0).clamp_min(1e-6)
        baseline_q = qdq[..., :3] + 0.1 * qdq[..., 3:]
        q_norm = (baseline_q - p_mean[:, None, :3]) / p_std[:, None, :3]
        tau_norm = (tau - p_mean[:, None, 6:9]) / p_std[:, None, 6:9]
        body_norm = (jointbody[:, 24:27] - b_mean[:, :3]) / b_std[:, :3]
        thresholds = batch.get("load_thresholds")
        if thresholds is None:
            threshold = jointbody.new_ones((B, 1))
        else:
            thresholds = thresholds.to(device=jointbody.device, dtype=jointbody.dtype)
            if thresholds.dim() == 1:
                thresholds = thresholds.unsqueeze(0).expand(B, -1)
            threshold = thresholds[:, -1:].clamp_min(1.0)
        force_magnitude = torch.linalg.norm(force, dim=-1) / threshold
        # The existing 16 load channels carry both predicted contact and load:
        # each leg's four entries sum to its contact probability.
        regime_contact = regime_prob * contact_prob[..., None]
        return torch.cat([
            q_norm.reshape(B, -1), tau_norm.reshape(B, -1), body_norm,
            force_magnitude, regime_contact.reshape(B, -1),
        ], dim=-1)

    def forward(self, batch: Dict[str, torch.Tensor], history_shift: int = 1,
                teacher_force_mask: torch.Tensor | None = None):
        return self.rollout(batch, history_shift=history_shift, teacher_force_mask=teacher_force_mask)

    def rollout(self, batch: Dict[str, torch.Tensor], history_shift: int = 1,
                teacher_force_mask: torch.Tensor | None = None) -> Dict[str, torch.Tensor | List[Dict[str, torch.Tensor]]]:
        """Autoregressive K-step rollout. Every FiLM branch uses current action a_t at that rollout step."""
        z_window = batch["z_window"]
        leg_t = batch["leg_hist_t"]
        leg_tm1 = batch["leg_hist_tm1"]
        leg_tm2 = batch["leg_hist_tm2"]
        body_t = batch["body_hist_t"]
        K = batch["future_action_seq"].shape[1]
        B = z_window.shape[0]
        if teacher_force_mask is None:
            teacher_force_mask = torch.zeros((B, K), dtype=torch.bool, device=z_window.device)
        else:
            teacher_force_mask = teacher_force_mask.to(device=z_window.device, dtype=torch.bool)
            if tuple(teacher_force_mask.shape) != (B, K):
                raise ValueError(f"teacher_force_mask must be [{B},{K}]")

        steps: List[Dict[str, torch.Tensor]] = []
        force_ctx_seq, base_force_seq, contact_seq, regime_seq = [], [], [], []
        refined_force_seq: List[torch.Tensor] = []
        feedback_force_seq: List[torch.Tensor] = []
        jointbody_state = tau_state = force_state = regime_state = contact_state = None
        foot_position = batch["leg_query_features_seq"][:, 0, :, 18:21]
        previous_q = leg_t[:, :, -1, :3]
        foot_jacobian = batch.get("foot_jacobian")
        if foot_jacobian is not None:
            foot_jacobian = foot_jacobian.to(device=z_window.device, dtype=z_window.dtype)

        for k in range(K):
            if k == 0:
                query_t = batch["leg_query_features_seq"][:, k]
                physics_t = batch["physics_seq"][:, k]
                jointbody_baseline_t = batch["jointbody_baseline_seq"][:, k]
                tau_baseline_t = batch["tau_baseline_seq"][:, k]
                force_baseline_t = batch["force_baseline_seq"][:, k]
            else:
                query_exogenous = batch["leg_query_features_seq"][:, k]
                query_t = self._recurrent_query(
                    jointbody_state, tau_state, force_state, query_exogenous,
                    foot_position, batch,
                )
                physics_t = self._recurrent_physics(
                    jointbody_state, tau_state, force_state, contact_state, regime_state, batch,
                )
                jointbody_baseline_t = jointbody_state
                tau_baseline_t = tau_state
                force_baseline_t = force_state
            out = self.forward_step(
                z_window=z_window,
                leg_hist_t=leg_t, leg_hist_tm1=leg_tm1, leg_hist_tm2=leg_tm2,
                body_hist_t=body_t,
                body_action_t=batch["body_action_seq"][:, k],
                leg_action_t=batch["leg_action_seq"][:, k],   # CURRENT a_t ONLY
                leg_query_features_t=query_t,
                future_action=batch["future_action_seq"][:, k],
                physics=physics_t,
                horizon=batch["horizon_seq"][:, k],
                jointbody_baseline=jointbody_baseline_t,
                jointbody_std=batch["jointbody_std"],
                tau_baseline=tau_baseline_t,
                tau_std=batch["tau_std"],
                force_baseline=force_baseline_t,
                force_std=batch["force_std"],
            )
            steps.append(out)
            force_ctx_seq.append(out["force_ctx"])
            base_force_seq.append(out["base_force"])
            contact_seq.append(out["contact_prob"])
            regime_seq.append(out["regime_prob"])

            # V7-F is evaluated causally on this available prefix. Its refined
            # force is the force fed into the next autoregressive step.
            v7f_prefix = self.v7f(
                torch.stack(force_ctx_seq, dim=1),
                torch.stack(base_force_seq, dim=1), batch["force_std"],
                torch.stack(contact_seq, dim=1), torch.stack(regime_seq, dim=1),
            )
            refined_force = v7f_prefix["force"][:, -1]
            refined_force_seq.append(refined_force)

            # Recurrent state update.
            next_jointbody = out["jointbody_pred"]
            next_tau = out["tau_pred"]
            next_force = refined_force
            next_contact = out["contact_prob"]
            next_regime = out["regime_prob"]
            next_z = out["z_next"]
            if k + 1 < K:
                force_teacher = teacher_force_mask[:, k + 1, None, None]
                state_teacher = teacher_force_mask[:, k + 1, None]
                next_jointbody = torch.where(state_teacher, batch["target_jointbody_seq"][:, k], next_jointbody)
                next_tau = torch.where(force_teacher, batch["target_tau_seq"][:, k], next_tau)
                next_force = torch.where(force_teacher, batch["target_force_seq"][:, k], next_force)
                target_contact = batch["target_contact_seq"][:, k]
                next_contact = torch.where(state_teacher, target_contact, next_contact)
                target_regime = F.one_hot(
                    batch["target_load_class_seq"][:, k].long(), self.cfg.num_load_regimes
                ).to(next_regime.dtype)
                next_regime = torch.where(state_teacher[..., None], target_regime, next_regime)
                next_z = torch.where(
                    teacher_force_mask[:, k + 1, None, None], batch["target_z_seq"][:, k], next_z
                )

            jointbody_state, tau_state, force_state = next_jointbody, next_tau, next_force
            feedback_force_seq.append(next_force)
            regime_state, contact_state = next_regime, next_contact
            next_q = next_jointbody[:, :24].reshape(B, 4, 6)[..., :3]
            if foot_jacobian is not None:
                foot_position = foot_position + torch.einsum(
                    "blij,blj->bli", foot_jacobian, next_q - previous_q
                )
            previous_q = next_q
            next_leg = self._make_leg_history_step(
                next_jointbody, next_tau, next_force,
                batch["future_action_seq"][:, k], batch["lowlevel_cmd_seq"][:, k], batch["err_seq"][:, 0]
            )
            next_body = self._make_body_history_step(next_jointbody, batch["future_action_seq"][:, k])
            leg_tm2, leg_tm1 = leg_tm1, leg_t
            leg_t = self._append_repeated(leg_t, next_leg, history_shift)
            body_t = self._append_repeated(body_t, next_body, history_shift)
            z_window = torch.cat([z_window[:, 1:], next_z.unsqueeze(1)], dim=1)

        force_ctx_seq = torch.stack(force_ctx_seq, dim=1)
        base_force_seq = torch.stack(base_force_seq, dim=1)
        contact_seq = torch.stack(contact_seq, dim=1)
        regime_seq = torch.stack(regime_seq, dim=1)
        v7f = v7f_prefix
        return {
            "steps": steps,
            "force_ctx_seq": force_ctx_seq,
            "base_force_seq": base_force_seq,
            "contact_seq": contact_seq,
            "regime_seq": regime_seq,
            "v7f": v7f,
            "refined_force_seq": torch.stack(refined_force_seq, dim=1),
            "feedback_force_seq": torch.stack(feedback_force_seq, dim=1),
        }
