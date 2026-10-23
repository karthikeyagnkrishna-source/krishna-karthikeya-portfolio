# Source inspection export of the adjacent notebook.
# Notebook execution order and dependencies still apply.
# IPython magic, shell lines and prose-only conclusion cells are comments.

# %% notebook cell 0
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. TEMPORAL PROPRIOCEPTION ENCODER
#
# [B, 50, d_prop] -> [B, 128]
# ============================================================

class TemporalCNN(nn.Module):

    def __init__(self, in_dim, out_dim=128):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv1d(
                in_dim,
                64,
                kernel_size=7,
                stride=2,
                padding=3
            ),
            nn.GroupNorm(8, 64),
            nn.GELU(),

            nn.Conv1d(
                64,
                128,
                kernel_size=5,
                stride=2,
                padding=2
            ),
            nn.GroupNorm(8, 128),
            nn.GELU(),

            nn.Conv1d(
                128,
                out_dim,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.GroupNorm(8, out_dim),
            nn.GELU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.norm = nn.LayerNorm(out_dim)


    def forward(self, x):

        # x:
        # [B,50,d]

        x = x.transpose(1, 2)

        # [B,d,50]

        x = self.net(x).squeeze(-1)

        # [B,128]

        return self.norm(x)


# ============================================================
# 2. ACTION ENCODER
#
# R^4 -> R^128
# ============================================================

class ActionEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(4, 128),

            nn.GELU(),

            nn.Linear(128, 128),

            nn.LayerNorm(128)
        )


    def forward(self, action):

        return self.net(action)


# ============================================================
# 3. FiLM GENERATOR
#
# fused proprioception/action:
#
# R128
#   |
#   v
# R256
#   |
#   v
# R1536
#
# split:
#
# gamma R768
# beta  R768
# ============================================================

class FiLMGenerator(nn.Module):

    def __init__(self, d_model=768):
        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                128,
                256
            ),

            nn.GELU(),

            nn.Linear(
                256,
                2 * d_model
            )
        )

        # Start FiLM near identity:
        #
        # z_tilde = z * (1 + gamma) + beta
        #
        # gamma = beta = 0 initially

        nn.init.zeros_(
            self.net[-1].weight
        )

        nn.init.zeros_(
            self.net[-1].bias
        )


    def forward(self, condition):

        gamma, beta = (
            self.net(condition)
            .chunk(2, dim=-1)
        )

        return gamma, beta


# ============================================================
# 4. ONE TRANSFORMER DYNAMICS BLOCK
#
# IMPORTANT CORRECTION:
#
# EVERY TRANSFORMER BLOCK:
#
# ATTENTION:
# 768 -> 768
#
# MLP:
#
# 768
#   |
#   v
# 5*768 = 3840
#   |
# GELU
#   |
#   v
# 4*768 = 3072
#   |
# GELU
#   |
#   v
# 768
#
# So:
#
# d -> 5d -> 4d -> d
#
# IN ALL SIX BLOCKS.
# ============================================================

class DynamicsBlock(nn.Module):

    def __init__(
        self,
        d_model=768,
        n_heads=8,
        dropout=0.1
    ):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model

        self.n_heads = n_heads

        self.head_dim = (
            d_model // n_heads
        )

        self.dropout = dropout


        # ====================================================
        # SELF ATTENTION
        # ====================================================

        self.norm1 = nn.LayerNorm(
            d_model
        )


        # One projection generates:
        #
        # Q,K,V
        #
        # 768 -> 3*768 = 2304

        self.qkv = nn.Linear(
            d_model,
            3 * d_model
        )


        self.attn_proj = nn.Linear(
            d_model,
            d_model
        )


        # ====================================================
        # YOUR NEW MLP
        #
        # Original:
        #
        # 768 -> 3072 -> 768
        #
        #
        # YOUR ARCHITECTURE:
        #
        # 768
        #  ->
        # 3840 = 5*768
        #  ->
        # 3072 = 4*768
        #  ->
        # 768
        # ====================================================

        self.norm2 = nn.LayerNorm(
            d_model
        )


        self.mlp = nn.Sequential(

            # -------------------------------
            # EXPANSION 1
            #
            # d -> 5d
            #
            # 768 -> 3840
            # -------------------------------

            nn.Linear(
                d_model,
                5 * d_model
            ),

            nn.GELU(),

            nn.Dropout(dropout),


            # -------------------------------
            # EXPANSION/HIDDEN LAYER 2
            #
            # 5d -> 4d
            #
            # 3840 -> 3072
            # -------------------------------

            nn.Linear(
                5 * d_model,
                4 * d_model
            ),

            nn.GELU(),

            nn.Dropout(dropout),


            # -------------------------------
            # BACK TO TRANSFORMER WIDTH
            #
            # 4d -> d
            #
            # 3072 -> 768
            # -------------------------------

            nn.Linear(
                4 * d_model,
                d_model
            ),

            nn.Dropout(dropout)
        )


    def forward(
        self,
        x,
        attention_mask=None
    ):

        # x:
        #
        # [B,N,768]

        B, N, C = x.shape


        # ====================================================
        # ATTENTION
        # ====================================================

        residual = x


        h = self.norm1(x)


        qkv = self.qkv(h)

        # [B,N,2304]


        qkv = qkv.reshape(

            B,
            N,
            3,
            self.n_heads,
            self.head_dim

        )


        # ->
        #
        # [3,B,heads,N,head_dim]

        qkv = qkv.permute(
            2,
            0,
            3,
            1,
            4
        )


        q, k, v = qkv.unbind(0)


        # For d=768, heads=8:
        #
        # Q = [B,8,N,96]
        # K = [B,8,N,96]
        # V = [B,8,N,96]


        attn = F.scaled_dot_product_attention(

            q,
            k,
            v,

            attn_mask=attention_mask,

            dropout_p=(
                self.dropout
                if self.training
                else 0.0
            )
        )


        # [B,8,N,96]
        #
        # ->
        #
        # [B,N,768]

        attn = (
            attn
            .transpose(1, 2)
            .contiguous()
            .reshape(
                B,
                N,
                C
            )
        )


        attn = self.attn_proj(
            attn
        )


        x = residual + F.dropout(

            attn,

            p=self.dropout,

            training=self.training

        )


        # ====================================================
        # 5d -> 4d MLP
        # ====================================================

        residual = x


        h = self.norm2(x)


        # EXACTLY:
        #
        # [B,N,768]
        #
        # ->
        #
        # [B,N,3840]
        #
        # ->
        #
        # [B,N,3072]
        #
        # ->
        #
        # [B,N,768]

        h = self.mlp(h)


        x = residual + h


        # Output remains:
        #
        # [B,N,768]

        return x


# ============================================================
# 5. COMPLETE FIVE-BRANCH WORLD MODEL
# ============================================================

class FiveStreamFeelWM(nn.Module):

    """
    ==========================================================
    EXACT TRANSFORMER INPUT ORDER
    ==========================================================

    0:
        z_(t-3)

    1:
        z_(t-3)_tilde_leg3

    2:
        z_(t-3)_tilde_leg4

    3:
        z_(t-2)

    4:
        z_(t-2)_tilde_leg1

    5:
        z_(t-2)_tilde_leg2

    6:
        z_(t-1)

    7:
        z_t

    8:
        z_t_tilde_body


    If each visual representation contains 256 tokens:

        9 * 256 = 2304 tokens

    Transformer input:

        [B,2304,768]

    Transformer output:

        [B,2304,768]

    which is reshaped back into:

        [B,9,256,768]


    ==========================================================
    HEAD ASSIGNMENTS
    ==========================================================

    H group 8
        ->
        visual head
        body proprioception head


    H group 4
        ->
        leg 1 head


    H group 5
        ->
        leg 2 head


    H group 1
        ->
        leg 3 head


    H group 2
        ->
        leg 4 head
    """


    GROUP_NAMES = (

        "z_tm3",

        "z_tm3_tilde_leg3",

        "z_tm3_tilde_leg4",

        "z_tm2",

        "z_tm2_tilde_leg1",

        "z_tm2_tilde_leg2",

        "z_tm1",

        "z_t",

        "z_t_tilde_body"
    )


    # Physical temporal correspondence.
    #
    # Groups with the same value correspond to the same
    # physical visual time.

    GROUP_TIMES = (

        0,  # z t-3

        0,  # leg3 conditioned z t-3

        0,  # leg4 conditioned z t-3

        1,  # z t-2

        1,  # leg1 conditioned z t-2

        1,  # leg2 conditioned z t-2

        2,  # z t-1

        3,  # z t

        3   # body conditioned z t

    )


    def __init__(

        self,

        body_hist_dim,

        leg_hist_dims,

        patches=256,

        d_model=768,

        depth=6,

        n_heads=8,

        dropout=0.1

    ):
        super().__init__()


        assert (
            len(leg_hist_dims) == 4
        )


        self.patches = patches

        self.d_model = d_model

        self.depth = depth


        # ====================================================
        # WHOLE-BODY PROPRIOCEPTION
        #
        # [B,50,d_body]
        # ->
        # [B,128]
        # ====================================================

        self.body_prop_encoder = (
            TemporalCNN(
                body_hist_dim,
                128
            )
        )


        # ====================================================
        # FOUR LEG PROPRIOCEPTION ENCODERS
        #
        # each:
        #
        # [B,50,d_leg_i]
        #
        # ->
        #
        # [B,128]
        # ====================================================

        self.leg_prop_encoders = (
            nn.ModuleList([

                TemporalCNN(
                    leg_hist_dims[0],
                    128
                ),

                TemporalCNN(
                    leg_hist_dims[1],
                    128
                ),

                TemporalCNN(
                    leg_hist_dims[2],
                    128
                ),

                TemporalCNN(
                    leg_hist_dims[3],
                    128
                )
            ])
        )


        # ====================================================
        # BODY ACTION ENCODER
        #
        # a_t =
        #
        # [
        # delta_x_body,
        # delta_y_body,
        # delta_psi,
        # delta_t
        # ]
        #
        # R4 -> R128
        # ====================================================

        self.body_action_encoder = (
            ActionEncoder()
        )


        # ====================================================
        # FOUR LEG ACTION ENCODERS
        #
        # a_i =
        #
        # [
        # delta_q_HFE,
        # delta_q_KFE,
        # torque_HFE,
        # torque_KFE
        # ]
        #
        # each:
        #
        # R4 -> R128
        # ====================================================

        self.leg_action_encoders = (
            nn.ModuleList([

                ActionEncoder(),

                ActionEncoder(),

                ActionEncoder(),

                ActionEncoder()
            ])
        )


        # ====================================================
        # FIVE SEPARATE FiLM GENERATORS
        # ====================================================

        # Pair 1:
        #
        # body

        self.body_film = (
            FiLMGenerator(
                d_model
            )
        )


        # Pairs 2-5:
        #
        # leg 1
        # leg 2
        # leg 3
        # leg 4

        self.leg_films = (
            nn.ModuleList([

                FiLMGenerator(
                    d_model
                ),

                FiLMGenerator(
                    d_model
                ),

                FiLMGenerator(
                    d_model
                ),

                FiLMGenerator(
                    d_model
                )
            ])
        )


        # ====================================================
        # TIME EMBEDDINGS
        #
        # only four REAL visual times:
        #
        # t-3
        # t-2
        # t-1
        # t
        # ====================================================

        self.time_embedding = (
            nn.Embedding(
                4,
                d_model
            )
        )


        # ====================================================
        # SIX TRANSFORMER BLOCKS
        #
        # EACH ONE USES:
        #
        # 768 -> 3840 -> 3072 -> 768
        #
        # NOT only first block.
        #
        # ALL SIX.
        # ====================================================

        self.blocks = (
            nn.ModuleList([

                DynamicsBlock(

                    d_model=d_model,

                    n_heads=n_heads,

                    dropout=dropout

                )

                for _ in range(depth)

            ])
        )


        self.final_norm = (
            nn.LayerNorm(
                d_model
            )
        )


        # ====================================================
        # VISUAL HEAD
        #
        # H_body:
        #
        # [B,256,768]
        #
        # ->
        #
        # predicted z_(t+1):
        #
        # [B,256,768]
        # ====================================================

        self.visual_head = (
            nn.Linear(
                d_model,
                d_model
            )
        )


        # ====================================================
        # BODY PROPRIOCEPTION HEAD
        #
        # H_body
        #
        # [B,256,768]
        #
        # mean pool patches
        #
        # ->
        #
        # [B,768]
        #
        # ->
        #
        # [B,4]
        # ====================================================

        self.body_head = (
            nn.Sequential(

                nn.Linear(
                    d_model,
                    256
                ),

                nn.GELU(),

                nn.Linear(
                    256,
                    4
                )
            )
        )


        # ====================================================
        # FOUR LEG-SPECIFIC HEADS
        #
        # each:
        #
        # H_leg:
        #
        # [B,256,768]
        #
        # mean pool
        #
        # ->
        #
        # [B,768]
        #
        # ->
        #
        # [B,4]
        # ====================================================

        self.leg_heads = (
            nn.ModuleList([

                nn.Sequential(

                    nn.Linear(
                        d_model,
                        256
                    ),

                    nn.GELU(),

                    nn.Linear(
                        256,
                        4
                    )
                )

                for _ in range(4)
            ])
        )


        # ====================================================
        # PREDICTED STATE -> CONDITIONING VECTOR
        #
        # Used for recursive world-model rollout.
        #
        # predicted R4
        #
        # ->
        #
        # R128
        # ====================================================

        self.body_feedback = (
            nn.Sequential(

                nn.Linear(
                    4,
                    128
                ),

                nn.GELU(),

                nn.LayerNorm(
                    128
                )
            )
        )


        self.leg_feedback = (
            nn.ModuleList([

                nn.Sequential(

                    nn.Linear(
                        4,
                        128
                    ),

                    nn.GELU(),

                    nn.LayerNorm(
                        128
                    )
                )

                for _ in range(4)

            ])
        )


        # ====================================================
        # CAUSAL TEMPORAL ATTENTION MASK
        # ====================================================

        group_times = torch.tensor(

            self.GROUP_TIMES,

            dtype=torch.long
        )


        # Each of the 9 groups contains
        # 'patches' tokens.

        token_times = (
            torch.repeat_interleave(

                group_times,

                patches
            )
        )


        # query time >= key time
        #
        # means:
        #
        # current/present may look backwards,
        # past cannot look into future.
        #
        # Same physical time is allowed.

        allowed = (

            token_times[:, None]

            >=

            token_times[None, :]

        )


        self.register_buffer(

            "attention_mask",

            allowed,

            persistent=False
        )


    # ========================================================
    # FiLM APPLICATION
    # ========================================================

    @staticmethod
    def apply_film(
        z,
        gamma,
        beta
    ):

        # z:
        # [B,256,768]
        #
        # gamma:
        # [B,768]
        #
        # beta:
        # [B,768]

        return (

            z

            *

            (
                1.0
                +
                gamma[:, None, :]
            )

            +

            beta[:, None, :]

        )


    # ========================================================
    # ENCODE ALL PROPRIOCEPTION
    # ========================================================

    def encode_proprioception(

        self,

        body_hist_t,

        leg1_hist_tm1,

        leg2_hist_tm1,

        leg3_hist_tm2,

        leg4_hist_tm2

    ):


        # ----------------------------------------------------
        # BODY:
        #
        # history ending at t
        # ----------------------------------------------------

        p_body = (
            self.body_prop_encoder(
                body_hist_t
            )
        )


        histories = [

            leg1_hist_tm1,

            leg2_hist_tm1,

            leg3_hist_tm2,

            leg4_hist_tm2
        ]


        p_legs = []


        for encoder, history in zip(

            self.leg_prop_encoders,

            histories

        ):

            p_legs.append(

                encoder(
                    history
                )

            )


        return p_body, p_legs


    # ========================================================
    # CORE WORLD MODEL
    #
    # This function receives already encoded p vectors.
    # ========================================================

    def forward_from_embeddings(

        self,

        context,

        p_body_t,

        p_legs,

        a_body_t,

        a_legs

    ):

        """
        context:

        [
            z_(t-3),
            z_(t-2),
            z_(t-1),
            z_t
        ]

        Shape:

            [B,4,256,768]


        p_legs:

        [
            p1_(t-1),
            p2_(t-1),
            p3_(t-2),
            p4_(t-2)
        ]


        a_legs:

        [
            a1_(t-1),
            a2_(t-1),
            a3_(t-2),
            a4_(t-2)
        ]
        """


        assert (
            context.ndim == 4
        )


        assert (
            context.shape[1] == 4
        )


        assert (
            len(p_legs) == 4
        )


        assert (
            len(a_legs) == 4
        )


        # ----------------------------------------------------
        # Extract visual frames
        # ----------------------------------------------------

        (
            z_tm3,

            z_tm2,

            z_tm1,

            z_t

        ) = context.unbind(dim=1)


        # ====================================================
        # BRANCH 1:
        #
        # WHOLE BODY
        #
        # p_t + Ea(a_t)
        #
        # ->
        #
        # FiLM 1
        #
        # ->
        #
        # gamma1,beta1
        #
        # ->
        #
        # applied to z_t
        #
        # ->
        #
        # z_t_tilde
        # ====================================================

        E_body = (
            self.body_action_encoder(
                a_body_t
            )
        )


        body_condition = (

            p_body_t

            +

            E_body

        )


        gamma1, beta1 = (
            self.body_film(
                body_condition
            )
        )


        z_t_tilde = (
            self.apply_film(

                z_t,

                gamma1,

                beta1

            )
        )


        # ====================================================
        # BRANCH 2:
        #
        # LEG 1
        #
        # p1_(t-1) + Ea1(a1_(t-1))
        #
        # ->
        #
        # FiLM 2
        #
        # ->
        #
        # applied to z_(t-2)
        # ====================================================

        E_leg1 = (
            self.leg_action_encoders[0](
                a_legs[0]
            )
        )


        leg1_condition = (

            p_legs[0]

            +

            E_leg1

        )


        gamma2, beta2 = (
            self.leg_films[0](
                leg1_condition
            )
        )


        z_tm2_tilde_leg1 = (
            self.apply_film(

                z_tm2,

                gamma2,

                beta2

            )
        )


        # ====================================================
        # BRANCH 3:
        #
        # LEG 2
        # ====================================================

        E_leg2 = (
            self.leg_action_encoders[1](
                a_legs[1]
            )
        )


        leg2_condition = (

            p_legs[1]

            +

            E_leg2

        )


        gamma3, beta3 = (
            self.leg_films[1](
                leg2_condition
            )
        )


        z_tm2_tilde_leg2 = (
            self.apply_film(

                z_tm2,

                gamma3,

                beta3

            )
        )


        # ====================================================
        # BRANCH 4:
        #
        # LEG 3
        #
        # p3_(t-2) + Ea3(a3_(t-2))
        #
        # ->
        #
        # applied to z_(t-3)
        # ====================================================

        E_leg3 = (
            self.leg_action_encoders[2](
                a_legs[2]
            )
        )


        leg3_condition = (

            p_legs[2]

            +

            E_leg3

        )


        gamma4, beta4 = (
            self.leg_films[2](
                leg3_condition
            )
        )


        z_tm3_tilde_leg3 = (
            self.apply_film(

                z_tm3,

                gamma4,

                beta4

            )
        )


        # ====================================================
        # BRANCH 5:
        #
        # LEG 4
        #
        # p4_(t-2) + Ea4(a4_(t-2))
        #
        # ->
        #
        # applied to z_(t-3)
        # ====================================================

        E_leg4 = (
            self.leg_action_encoders[3](
                a_legs[3]
            )
        )


        leg4_condition = (

            p_legs[3]

            +

            E_leg4

        )


        gamma5, beta5 = (
            self.leg_films[3](
                leg4_condition
            )
        )


        z_tm3_tilde_leg4 = (
            self.apply_film(

                z_tm3,

                gamma5,

                beta5

            )
        )


        # ====================================================
        # EXACT 9-GROUP ORDER
        # ====================================================

        groups = torch.stack(

            [

                # GROUP 0
                z_tm3,


                # GROUP 1
                z_tm3_tilde_leg3,


                # GROUP 2
                z_tm3_tilde_leg4,


                # GROUP 3
                z_tm2,


                # GROUP 4
                z_tm2_tilde_leg1,


                # GROUP 5
                z_tm2_tilde_leg2,


                # GROUP 6
                z_tm1,


                # GROUP 7
                z_t,


                # GROUP 8
                z_t_tilde

            ],

            dim=1
        )


        # Shape:
        #
        # [B,9,256,768]


        # ====================================================
        # ADD TEMPORAL EMBEDDINGS
        # ====================================================

        time_ids = torch.tensor(

            self.GROUP_TIMES,

            device=groups.device,

            dtype=torch.long

        )


        temporal_embeddings = (
            self.time_embedding(
                time_ids
            )
        )


        # [9,768]


        groups = (

            groups

            +

            temporal_embeddings[
                None,
                :,
                None,
                :
            ]

        )


        # ====================================================
        # FLATTEN GROUPS INTO TRANSFORMER SEQUENCE
        # ====================================================

        B, G, P, C = groups.shape


        x = groups.reshape(

            B,

            G * P,

            C

        )


        # If P=256:
        #
        # [B,9*256,768]
        #
        # =
        #
        # [B,2304,768]


        # ====================================================
        # SIX TRANSFORMER LAYERS
        #
        # IMPORTANT:
        #
        # EVERY ONE:
        #
        # Attention
        #
        # +
        #
        # 768
        # ->
        # 3840
        # ->
        # 3072
        # ->
        # 768
        # ====================================================

        for block in self.blocks:

            x = block(

                x,

                attention_mask=(
                    self.attention_mask
                )

            )


        # Final normalization

        x = self.final_norm(x)


        # ====================================================
        # RESTORE THE 9 GROUPS
        # ====================================================

        H = x.reshape(

            B,

            G,

            P,

            C

        )


        # H:
        #
        # [B,9,256,768]


        H_groups = {

            name: H[:, index]

            for index, name
            in enumerate(
                self.GROUP_NAMES
            )

        }


        # ====================================================
        # BODY OUTPUT
        #
        # Take transformer output corresponding EXACTLY to:
        #
        # z_t_tilde
        #
        # GROUP 8
        # ====================================================

        H_body = (
            H_groups[
                "z_t_tilde_body"
            ]
        )


        # H_body:
        #
        # [B,256,768]


        # ====================================================
        # VISUAL HEAD
        #
        # Preserve 256 spatial tokens
        # ====================================================

        z_next = (
            self.visual_head(
                H_body
            )
        )


        # z_next:
        #
        # [B,256,768]


        # ====================================================
        # WHOLE BODY PHYSICAL HEAD
        # ====================================================

        H_body_global = (
            H_body.mean(
                dim=1
            )
        )


        # [B,768]


        p_body_next = (
            self.body_head(
                H_body_global
            )
        )


        # [B,4]


        # ====================================================
        # LEG 1
        #
        # Transformer output corresponding to:
        #
        # z_(t-2)_tilde_leg1
        #
        # GROUP 4
        # ====================================================

        H_leg1 = (
            H_groups[
                "z_tm2_tilde_leg1"
            ]
        )


        H_leg1_global = (
            H_leg1.mean(
                dim=1
            )
        )


        p_leg1_next = (
            self.leg_heads[0](
                H_leg1_global
            )
        )


        # ====================================================
        # LEG 2
        #
        # GROUP 5
        # ====================================================

        H_leg2 = (
            H_groups[
                "z_tm2_tilde_leg2"
            ]
        )


        H_leg2_global = (
            H_leg2.mean(
                dim=1
            )
        )


        p_leg2_next = (
            self.leg_heads[1](
                H_leg2_global
            )
        )


        # ====================================================
        # LEG 3
        #
        # GROUP 1
        # ====================================================

        H_leg3 = (
            H_groups[
                "z_tm3_tilde_leg3"
            ]
        )


        H_leg3_global = (
            H_leg3.mean(
                dim=1
            )
        )


        p_leg3_next = (
            self.leg_heads[2](
                H_leg3_global
            )
        )


        # ====================================================
        # LEG 4
        #
        # GROUP 2
        # ====================================================

        H_leg4 = (
            H_groups[
                "z_tm3_tilde_leg4"
            ]
        )


        H_leg4_global = (
            H_leg4.mean(
                dim=1
            )
        )


        p_leg4_next = (
            self.leg_heads[3](
                H_leg4_global
            )
        )


        return {

            # -----------------------------------------------
            # FUTURE VISUAL LATENT
            # -----------------------------------------------

            "z_next":
                z_next,


            # -----------------------------------------------
            # WHOLE-BODY FUTURE STATE
            # -----------------------------------------------

            "p_body_next":
                p_body_next,


            # -----------------------------------------------
            # FOUR LEG-SPECIFIC FUTURE STATES
            # -----------------------------------------------

            "p_leg1_next":
                p_leg1_next,

            "p_leg2_next":
                p_leg2_next,

            "p_leg3_next":
                p_leg3_next,

            "p_leg4_next":
                p_leg4_next,


            # -----------------------------------------------
            # COMPLETE TRANSFORMER OUTPUT
            # -----------------------------------------------

            "H":
                H,


            # -----------------------------------------------
            # NAMED OUTPUT GROUPS
            # -----------------------------------------------

            "H_groups":
                H_groups,


            # -----------------------------------------------
            # FiLM PARAMETERS
            #
            # useful for debugging/inspection
            # -----------------------------------------------

            "film": {

                "body":
                    (gamma1, beta1),

                "leg1":
                    (gamma2, beta2),

                "leg2":
                    (gamma3, beta3),

                "leg3":
                    (gamma4, beta4),

                "leg4":
                    (gamma5, beta5)
            }

        }


    # ========================================================
    # COMPLETE FORWARD FROM RAW 50-STEP PROPRIO HISTORY
    # ========================================================

    def forward(

        self,

        context,

        body_hist_t,

        leg1_hist_tm1,

        leg2_hist_tm1,

        leg3_hist_tm2,

        leg4_hist_tm2,

        a_body_t,

        a1_tm1,

        a2_tm1,

        a3_tm2,

        a4_tm2

    ):


        # ====================================================
        # Encode p_t, p1, p2, p3, p4
        # ====================================================

        p_body, p_legs = (
            self.encode_proprioception(

                body_hist_t,

                leg1_hist_tm1,

                leg2_hist_tm1,

                leg3_hist_tm2,

                leg4_hist_tm2

            )
        )


        # ====================================================
        # Run world model
        # ====================================================

        prediction = (
            self.forward_from_embeddings(

                context=context,

                p_body_t=p_body,

                p_legs=p_legs,

                a_body_t=a_body_t,

                a_legs=[

                    a1_tm1,

                    a2_tm1,

                    a3_tm2,

                    a4_tm2

                ]

            )
        )


        return prediction


    # ========================================================
    # PROJECT PREDICTIONS BACK INTO 128-D CONDITIONING SPACE
    #
    # FOR AUTOREGRESSIVE RECURSION
    # ========================================================

    def project_predictions_for_feedback(

        self,

        prediction

    ):


        p_body_feedback = (
            self.body_feedback(

                prediction[
                    "p_body_next"
                ]

            )
        )


        p_leg1_feedback = (
            self.leg_feedback[0](

                prediction[
                    "p_leg1_next"
                ]

            )
        )


        p_leg2_feedback = (
            self.leg_feedback[1](

                prediction[
                    "p_leg2_next"
                ]

            )
        )


        p_leg3_feedback = (
            self.leg_feedback[2](

                prediction[
                    "p_leg3_next"
                ]

            )
        )


        p_leg4_feedback = (
            self.leg_feedback[3](

                prediction[
                    "p_leg4_next"
                ]

            )
        )


        return {

            "p_body":
                p_body_feedback,

            "p_leg1":
                p_leg1_feedback,

            "p_leg2":
                p_leg2_feedback,

            "p_leg3":
                p_leg3_feedback,

            "p_leg4":
                p_leg4_feedback

        }

# %% notebook cell 1
# ======================================================================
# CELL 1 — STRICT ANYMAL-C DATASET -> YOUR FIVE-STREAM ARCHITECTURE
#
# IMPORTANT:
# Feel-WM's exact 815 ANYmal-C author trajectories are not publicly
# downloadable at present.
#
# Therefore this cell defines the EXACT data contract required by our
# architecture. It will REFUSE to run if real leg channels are absent.
#
# Expected trajectory .npz:
#
# z                [T, 256, 768]
#                  Frozen DINOv2-B/14 tokens for every camera frame.
#
# body_hist         [T, 50, D_body]
#                  0.5 s whole-body proprioception ending at each frame.
#
# leg_hist          [T, 4, 50, D_leg]
#                  Per-leg 0.5 s proprioception histories.
#
# a_body            [T, 4]
#                  [dx_body, dy_body, dpsi, dt]
#
# a_leg             [T, 4, 4]
#                  Per leg:
#                  [Δq_HFE_cmd, Δq_KFE_cmd,
#                   tau_HFE_cmd, tau_KFE_cmd]
#
# body_target       [T, 4]
#                  Whole-body physical future:
#                  [roughness, angular_instability,
#                   foot_slip, slope_proxy]
#
# leg_target        [T, 4, 4]
#                  Per leg:
#                  [q_HFE, q_KFE, tau_HFE, tau_KFE]
#
# -------------------------------------------------------------
# YOUR EXACT TEMPORAL CONNECTION:
#
# body branch:
#       p_body(t), a_body(t)          -> z_t
#
# leg 1:
#       p1(t-1), a1(t-1)             -> z_(t-2)
#
# leg 2:
#       p2(t-1), a2(t-1)             -> z_(t-2)
#
# leg 3:
#       p3(t-2), a3(t-2)             -> z_(t-3)
#
# leg 4:
#       p4(t-2), a4(t-2)             -> z_(t-3)
#
# ALL HEADS SUPERVISED AGAINST t+1.
# ======================================================================

import os
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ----------------------------------------------------------------------
# 0. CHECK THAT YOUR ARCHITECTURE CELL WAS ACTUALLY RUN
# ----------------------------------------------------------------------

assert "FiveStreamFeelWM" in globals(), (
    "Run the complete FiveStreamFeelWM architecture cell first."
)


# ----------------------------------------------------------------------
# 1. PATH
#
# Put the real/exported ANYmal-C trajectories here.
#
# Suggested layout:
#
# /content/drive/MyDrive/FeelWM_ANYmalC_815/
#     train/
#         traj_0000.npz
#         traj_0001.npz
#         ...
#
#     validation/
#         ...
#
# We deliberately do NOT touch test yet.
# ----------------------------------------------------------------------

DATA_ROOT = Path(
    "/content/drive/MyDrive/FeelWM_ANYmalC_815"
)

TRAIN_ROOT = DATA_ROOT / "train"
VAL_ROOT   = DATA_ROOT / "validation"


# ----------------------------------------------------------------------
# 2. REQUIRED FIELDS
# ----------------------------------------------------------------------

REQUIRED_KEYS = {
    "z",
    "body_hist",
    "leg_hist",
    "a_body",
    "a_leg",
    "body_target",
    "leg_target",
}


def inspect_trajectory(path):

    with np.load(path, mmap_mode="r") as x:

        missing = REQUIRED_KEYS - set(x.files)

        if missing:
            raise RuntimeError(
                f"\n{path}\n"
                f"is missing REAL required channels:\n"
                f"{sorted(missing)}\n\n"
                "No surrogate/fake channels will be generated."
            )

        shapes = {
            k: tuple(x[k].shape)
            for k in REQUIRED_KEYS
        }

    return shapes


# ----------------------------------------------------------------------
# 3. STRICT SHAPE VERIFICATION
# ----------------------------------------------------------------------

def verify_trajectory(path):

    with np.load(path, mmap_mode="r") as x:

        z           = x["z"]
        body_hist   = x["body_hist"]
        leg_hist    = x["leg_hist"]
        a_body      = x["a_body"]
        a_leg       = x["a_leg"]
        body_target = x["body_target"]
        leg_target  = x["leg_target"]

        T = len(z)

        # --------------------------------------------------------------
        # Visual representation from frozen DINOv2-B/14
        # --------------------------------------------------------------

        assert z.ndim == 3
        assert z.shape[1:] == (256, 768), (
            path,
            "z must be [T,256,768]"
        )


        # --------------------------------------------------------------
        # Whole-body history
        # --------------------------------------------------------------

        assert body_hist.ndim == 3
        assert body_hist.shape[0] == T
        assert body_hist.shape[1] == 50, (
            path,
            "body_hist must be [T,50,D_body]"
        )


        # --------------------------------------------------------------
        # Four independent leg histories
        # --------------------------------------------------------------

        assert leg_hist.ndim == 4
        assert leg_hist.shape[0] == T
        assert leg_hist.shape[1] == 4
        assert leg_hist.shape[2] == 50, (
            path,
            "leg_hist must be [T,4,50,D_leg]"
        )


        # --------------------------------------------------------------
        # Whole-body high-level action
        # --------------------------------------------------------------

        assert a_body.shape == (T, 4), (
            path,
            "a_body must be [T,4]"
        )


        # --------------------------------------------------------------
        # Four leg actions
        # --------------------------------------------------------------

        assert a_leg.shape == (T, 4, 4), (
            path,
            "a_leg must be [T,4,4]"
        )


        # --------------------------------------------------------------
        # Whole-body physical target
        # --------------------------------------------------------------

        assert body_target.shape == (T, 4), (
            path,
            "body_target must be [T,4]"
        )


        # --------------------------------------------------------------
        # Four independent 4-D leg targets
        # --------------------------------------------------------------

        assert leg_target.shape == (T, 4, 4), (
            path,
            "leg_target must be [T,4,4]"
        )


        if T < 6:
            raise RuntimeError(
                f"{path}: only {T} frames; insufficient."
            )

        return (
            body_hist.shape[-1],
            leg_hist.shape[-1],
            T
        )


# ----------------------------------------------------------------------
# 4. FIND REAL TRAJECTORIES
# ----------------------------------------------------------------------

train_files = sorted(TRAIN_ROOT.glob("*.npz"))
val_files   = sorted(VAL_ROOT.glob("*.npz"))


if not train_files:

    raise FileNotFoundError(
        "\nExact legged training data are not present.\n\n"
        f"Expected real ANYmal-C trajectory exports in:\n"
        f"    {TRAIN_ROOT}\n\n"
        "Do NOT point this at TartanDrive: TartanDrive has no four-leg "
        "joint/action streams required by this architecture."
    )


print("Training trajectories :", len(train_files))
print("Validation trajectories:", len(val_files))


# Verify EVERY file instead of discovering bad data halfway through training.

reference_body_dim = None
reference_leg_dim  = None
total_frames       = 0

for path in train_files + val_files:

    body_dim, leg_dim, T = verify_trajectory(path)

    if reference_body_dim is None:
        reference_body_dim = body_dim
        reference_leg_dim  = leg_dim

    if body_dim != reference_body_dim:
        raise RuntimeError(
            f"Inconsistent body dimension: {path}"
        )

    if leg_dim != reference_leg_dim:
        raise RuntimeError(
            f"Inconsistent leg dimension: {path}"
        )

    total_frames += T


BODY_DIM = reference_body_dim
LEG_DIM  = reference_leg_dim

print("BODY_DIM =", BODY_DIM)
print("LEG_DIM  =", LEG_DIM)
print("Total available frames =", total_frames)


# ======================================================================
# 5. TRAIN-ONLY TARGET NORMALIZATION
#
# Feel-WM standardizes its physical target channels.
#
# We do the same for body_target.
#
# Our four new per-leg targets are ALSO standardized using training
# statistics only; this is necessary because angle and torque scales
# are different.
# ======================================================================

body_sum  = np.zeros(4, dtype=np.float64)
body_sq   = np.zeros(4, dtype=np.float64)
body_n    = 0

leg_sum   = np.zeros((4, 4), dtype=np.float64)
leg_sq    = np.zeros((4, 4), dtype=np.float64)
leg_n     = 0


for path in train_files:

    with np.load(path, mmap_mode="r") as x:

        b = np.asarray(
            x["body_target"],
            dtype=np.float64
        )

        l = np.asarray(
            x["leg_target"],
            dtype=np.float64
        )

        good_b = np.isfinite(b).all(axis=1)
        good_l = np.isfinite(l).all(axis=(1, 2))

        b = b[good_b]
        l = l[good_l]

        body_sum += b.sum(axis=0)
        body_sq  += (b * b).sum(axis=0)
        body_n   += len(b)

        leg_sum += l.sum(axis=0)
        leg_sq  += (l * l).sum(axis=0)
        leg_n   += len(l)


if body_n == 0 or leg_n == 0:
    raise RuntimeError(
        "No finite physical targets in training set."
    )


BODY_MEAN = body_sum / body_n
BODY_VAR  = body_sq / body_n - BODY_MEAN**2
BODY_STD  = np.sqrt(np.maximum(BODY_VAR, 1e-8))


LEG_MEAN = leg_sum / leg_n
LEG_VAR  = leg_sq / leg_n - LEG_MEAN**2
LEG_STD  = np.sqrt(np.maximum(LEG_VAR, 1e-8))


print("\nTRAIN-ONLY normalizers")
print("BODY_MEAN =", BODY_MEAN)
print("BODY_STD  =", BODY_STD)

print("\nLEG_MEAN [4 legs x 4 quantities] =")
print(LEG_MEAN)

print("\nLEG_STD =")
print(LEG_STD)


# ======================================================================
# 6. INDEX EVERY VALID t
#
# Context:
#
#      t-3, t-2, t-1, t
#
# Target:
#
#      t+1
#
# Therefore:
#
#      t >= 3
#      t+1 < T
# ======================================================================

class FiveStreamAnymalDataset(Dataset):

    def __init__(
        self,
        files,
        body_mean,
        body_std,
        leg_mean,
        leg_std
    ):

        self.files = list(files)

        self.body_mean = torch.tensor(
            body_mean,
            dtype=torch.float32
        )

        self.body_std = torch.tensor(
            body_std,
            dtype=torch.float32
        )

        self.leg_mean = torch.tensor(
            leg_mean,
            dtype=torch.float32
        )

        self.leg_std = torch.tensor(
            leg_std,
            dtype=torch.float32
        )


        # Each entry:
        #
        # (trajectory number, reference t)

        self.index = []

        for file_id, path in enumerate(self.files):

            with np.load(path, mmap_mode="r") as x:

                T = len(x["z"])

                for t in range(3, T - 1):

                    # Reject non-finite supervision samples.
                    if not np.isfinite(
                        x["body_target"][t + 1]
                    ).all():
                        continue

                    if not np.isfinite(
                        x["leg_target"][t + 1]
                    ).all():
                        continue

                    self.index.append(
                        (file_id, t)
                    )


        if not self.index:
            raise RuntimeError(
                "No valid t -> t+1 training samples."
            )


        # Tiny one-file RAM cache.
        self._cache_id = None
        self._cache = None


    def __len__(self):
        return len(self.index)


    def _load(self, file_id):

        if self._cache_id != file_id:

            path = self.files[file_id]

            with np.load(path, allow_pickle=False) as x:

                self._cache = {
                    k: np.asarray(x[k])
                    for k in REQUIRED_KEYS
                }

            self._cache_id = file_id

        return self._cache


    def __getitem__(self, index):

        file_id, t = self.index[index]

        x = self._load(file_id)


        # ==============================================================
        # FOUR REAL VISUAL FRAMES
        #
        # z_(t-3), z_(t-2), z_(t-1), z_t
        # ==============================================================

        context = torch.from_numpy(

            np.asarray(
                x["z"][t-3:t+1],
                dtype=np.float32
            )

        )


        # ==============================================================
        # BODY CONDITIONING AT t
        # ==============================================================

        body_hist_t = torch.from_numpy(

            np.asarray(
                x["body_hist"][t],
                dtype=np.float32
            )

        )

        a_body_t = torch.from_numpy(

            np.asarray(
                x["a_body"][t],
                dtype=np.float32
            )

        )


        # ==============================================================
        # LEGS 1,2:
        #
        # conditioning state/action from t-1
        # ==============================================================

        leg1_hist_tm1 = torch.from_numpy(

            np.asarray(
                x["leg_hist"][t-1, 0],
                dtype=np.float32
            )

        )

        leg2_hist_tm1 = torch.from_numpy(

            np.asarray(
                x["leg_hist"][t-1, 1],
                dtype=np.float32
            )

        )

        a1_tm1 = torch.from_numpy(

            np.asarray(
                x["a_leg"][t-1, 0],
                dtype=np.float32
            )

        )

        a2_tm1 = torch.from_numpy(

            np.asarray(
                x["a_leg"][t-1, 1],
                dtype=np.float32
            )

        )


        # ==============================================================
        # LEGS 3,4:
        #
        # conditioning state/action from t-2
        # ==============================================================

        leg3_hist_tm2 = torch.from_numpy(

            np.asarray(
                x["leg_hist"][t-2, 2],
                dtype=np.float32
            )

        )

        leg4_hist_tm2 = torch.from_numpy(

            np.asarray(
                x["leg_hist"][t-2, 3],
                dtype=np.float32
            )

        )

        a3_tm2 = torch.from_numpy(

            np.asarray(
                x["a_leg"][t-2, 2],
                dtype=np.float32
            )

        )

        a4_tm2 = torch.from_numpy(

            np.asarray(
                x["a_leg"][t-2, 3],
                dtype=np.float32
            )

        )


        # ==============================================================
        # ALL SUPERVISION IS t+1
        # ==============================================================

        z_next = torch.from_numpy(

            np.asarray(
                x["z"][t+1],
                dtype=np.float32
            )

        )


        body_target = torch.from_numpy(

            np.asarray(
                x["body_target"][t+1],
                dtype=np.float32
            )

        )


        leg_target = torch.from_numpy(

            np.asarray(
                x["leg_target"][t+1],
                dtype=np.float32
            )

        )


        # Paper-style physical-channel z-score normalization.

        body_target = (
            body_target - self.body_mean
        ) / self.body_std


        # Independent normalization for our four new leg heads.

        leg_target = (
            leg_target - self.leg_mean
        ) / self.leg_std


        return {

            "context": context,

            "body_hist_t": body_hist_t,

            "leg1_hist_tm1": leg1_hist_tm1,
            "leg2_hist_tm1": leg2_hist_tm1,

            "leg3_hist_tm2": leg3_hist_tm2,
            "leg4_hist_tm2": leg4_hist_tm2,

            "a_body_t": a_body_t,

            "a1_tm1": a1_tm1,
            "a2_tm1": a2_tm1,

            "a3_tm2": a3_tm2,
            "a4_tm2": a4_tm2,

            "z_target": z_next,

            "body_target": body_target,

            "leg_target": leg_target,

            "trajectory_id": file_id,

            "t_index": t,
        }


# ======================================================================
# 7. BUILD DATASETS
# ======================================================================

train_dataset = FiveStreamAnymalDataset(

    train_files,

    BODY_MEAN,
    BODY_STD,

    LEG_MEAN,
    LEG_STD
)


val_dataset = (

    FiveStreamAnymalDataset(

        val_files,

        BODY_MEAN,
        BODY_STD,

        LEG_MEAN,
        LEG_STD

    )

    if val_files

    else None
)


print("\nTraining samples  :", len(train_dataset))

if val_dataset is not None:
    print("Validation samples:", len(val_dataset))


# ======================================================================
# 8. DATA LOADERS
#
# Start batch size 1 because:
#
# 9 * 256 = 2304 transformer tokens,
# d = 768,
# six layers,
# and every layer has:
#
# 768 -> 3840 -> 3072 -> 768
#
# This is substantially heavier than original Feel-WM.
# ======================================================================

NUM_WORKERS = min(
    4,
    os.cpu_count() or 1
)

train_loader = DataLoader(

    train_dataset,

    batch_size=1,

    shuffle=True,

    num_workers=NUM_WORKERS,

    pin_memory=True,

    drop_last=False
)


val_loader = (

    DataLoader(

        val_dataset,

        batch_size=1,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=True

    )

    if val_dataset is not None

    else None
)


# ======================================================================
# 9. INSTANTIATE YOUR EXACT MODEL
# ======================================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


model = FiveStreamFeelWM(

    body_hist_dim=BODY_DIM,

    leg_hist_dims=[
        LEG_DIM,
        LEG_DIM,
        LEG_DIM,
        LEG_DIM
    ],

    patches=256,

    d_model=768,

    depth=6,

    n_heads=8,

    dropout=0.1

).to(device)


print("\nDEVICE:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


n_params = sum(
    p.numel()
    for p in model.parameters()
)

n_trainable = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


print(
    f"Parameters: {n_params/1e6:.2f} M"
)

print(
    f"Trainable : {n_trainable/1e6:.2f} M"
)


# ======================================================================
# 10. SANITY CHECK ONE REAL SAMPLE
# ======================================================================

batch = next(
    iter(train_loader)
)


print("\nSHAPE AUDIT")

for key, value in batch.items():

    if torch.is_tensor(value):
        print(
            f"{key:20s}",
            tuple(value.shape)
        )


assert batch["context"].shape[-3:] == (
    4,
    256,
    768
)

assert batch["a_body_t"].shape[-1] == 4

assert batch["a1_tm1"].shape[-1] == 4
assert batch["a2_tm1"].shape[-1] == 4
assert batch["a3_tm2"].shape[-1] == 4
assert batch["a4_tm2"].shape[-1] == 4

assert batch["body_target"].shape[-1] == 4

assert batch["leg_target"].shape[-2:] == (
    4,
    4
)


print(
    "\nDataset is structurally compatible "
    "with FiveStreamFeelWM."
)

# %% notebook cell 2
# ======================================================================
# CELL 2 — TRAIN YOUR FIVE-STREAM MODEL
#
# Trains:
#
# H[z_t_tilde]
#       -> z_(t+1)
#       -> body p_(t+1)
#
# H[z_(t-2)_tilde_leg1]
#       -> leg1_(t+1)
#
# H[z_(t-2)_tilde_leg2]
#       -> leg2_(t+1)
#
# H[z_(t-3)_tilde_leg3]
#       -> leg3_(t+1)
#
# H[z_(t-3)_tilde_leg4]
#       -> leg4_(t+1)
#
# NO TESTING METRICS YET.
# NO FAKE FAILURE HEAD.
# ======================================================================

import time
from pathlib import Path

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------
# 1. TRAINING SETTINGS
# ----------------------------------------------------------------------

EPOCHS = 20

LR = 2e-4

WEIGHT_DECAY = 1e-4

GRAD_ACCUM_STEPS = 4

MAX_GRAD_NORM = 1.0


# We have introduced 4 NEW leg losses not present in original Feel-WM.
# Keep the weights explicit instead of hiding a design choice.

LAMBDA_VIS  = 1.0

LAMBDA_BODY = 1.0

LAMBDA_LEGS = 1.0


CHECKPOINT_DIR = Path(
    "/content/drive/MyDrive/FiveStreamFeelWM_checkpoints"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ----------------------------------------------------------------------
# 2. VISUAL LOSS
#
# Same form used by Feel-WM:
#
# squared feature error
# +
# cosine-distance term
# ----------------------------------------------------------------------

def visual_loss(
    prediction,
    target
):

    mse = F.mse_loss(
        prediction,
        target
    )


    cosine = F.cosine_similarity(

        prediction,

        target,

        dim=-1

    )


    cosine_loss = (
        1.0
        -
        cosine.mean()
    )


    return (
        mse
        +
        0.1 * cosine_loss
    )


# ----------------------------------------------------------------------
# 3. PHYSICAL REGRESSION
# ----------------------------------------------------------------------

def regression_loss(
    prediction,
    target
):

    return F.smooth_l1_loss(
        prediction,
        target
    )


# ----------------------------------------------------------------------
# 4. MOVE ONE DICTIONARY TO GPU
# ----------------------------------------------------------------------

def to_device(
    batch,
    device
):

    out = {}

    for k, v in batch.items():

        if torch.is_tensor(v):

            out[k] = v.to(
                device,
                non_blocking=True
            )

        else:

            out[k] = v

    return out


# ----------------------------------------------------------------------
# 5. AMP
# ----------------------------------------------------------------------

use_amp = (
    device.type == "cuda"
)


# bfloat16 where possible; otherwise float16.

if (
    use_amp
    and torch.cuda.is_bf16_supported()
):

    amp_dtype = torch.bfloat16

else:

    amp_dtype = torch.float16


scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(
        use_amp
        and amp_dtype == torch.float16
    )
)


# ----------------------------------------------------------------------
# 6. OPTIMIZER
# ----------------------------------------------------------------------

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LR,

    weight_decay=WEIGHT_DECAY,

    betas=(0.9, 0.95)

)


# ----------------------------------------------------------------------
# 7. TRAIN LOOP
# ----------------------------------------------------------------------

history = []

best_train_loss = float("inf")


for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )


    epoch_total = 0.0

    epoch_visual = 0.0

    epoch_body = 0.0

    epoch_leg = np.zeros(
        4,
        dtype=np.float64
    )


    n_batches = 0

    start = time.time()


    for step, batch in enumerate(
        train_loader,
        start=1
    ):

        batch = to_device(
            batch,
            device
        )


        # ==============================================================
        # FORWARD PASS THROUGH YOUR EXACT ARCHITECTURE
        # ==============================================================

        with torch.autocast(

            device_type=device.type,

            dtype=amp_dtype,

            enabled=use_amp

        ):

            pred = model(

                context=
                    batch["context"],


                body_hist_t=
                    batch["body_hist_t"],


                leg1_hist_tm1=
                    batch["leg1_hist_tm1"],

                leg2_hist_tm1=
                    batch["leg2_hist_tm1"],


                leg3_hist_tm2=
                    batch["leg3_hist_tm2"],

                leg4_hist_tm2=
                    batch["leg4_hist_tm2"],


                a_body_t=
                    batch["a_body_t"],


                a1_tm1=
                    batch["a1_tm1"],

                a2_tm1=
                    batch["a2_tm1"],


                a3_tm2=
                    batch["a3_tm2"],

                a4_tm2=
                    batch["a4_tm2"]

            )


            # ==========================================================
            # BODY VISUAL PREDICTION
            #
            # [B,256,768]
            # ==========================================================

            L_vis = visual_loss(

                pred["z_next"],

                batch["z_target"]

            )


            # ==========================================================
            # BODY PHYSICAL HEAD
            #
            # [B,4]
            # ==========================================================

            L_body = regression_loss(

                pred["p_body_next"],

                batch["body_target"]

            )


            # ==========================================================
            # FOUR LEG HEADS
            #
            # each [B,4]
            # ==========================================================

            L_leg1 = regression_loss(

                pred["p_leg1_next"],

                batch["leg_target"][:, 0]

            )


            L_leg2 = regression_loss(

                pred["p_leg2_next"],

                batch["leg_target"][:, 1]

            )


            L_leg3 = regression_loss(

                pred["p_leg3_next"],

                batch["leg_target"][:, 2]

            )


            L_leg4 = regression_loss(

                pred["p_leg4_next"],

                batch["leg_target"][:, 3]

            )


            # Mean keeps total leg contribution independent
            # of the number of leg heads.

            L_legs = (

                L_leg1
                +
                L_leg2
                +
                L_leg3
                +
                L_leg4

            ) / 4.0


            # ==========================================================
            # TOTAL LOSS
            # ==========================================================

            loss = (

                LAMBDA_VIS
                *
                L_vis

                +

                LAMBDA_BODY
                *
                L_body

                +

                LAMBDA_LEGS
                *
                L_legs

            )


            scaled_loss = (
                loss
                /
                GRAD_ACCUM_STEPS
            )


        # ==============================================================
        # BACKPROP
        # ==============================================================

        scaler.scale(
            scaled_loss
        ).backward()


        if (

            step % GRAD_ACCUM_STEPS == 0

            or

            step == len(train_loader)

        ):

            scaler.unscale_(
                optimizer
            )


            torch.nn.utils.clip_grad_norm_(

                model.parameters(),

                MAX_GRAD_NORM

            )


            scaler.step(
                optimizer
            )


            scaler.update()


            optimizer.zero_grad(
                set_to_none=True
            )


        # ==============================================================
        # LOGGING
        # ==============================================================

        epoch_total += float(
            loss.detach()
        )

        epoch_visual += float(
            L_vis.detach()
        )

        epoch_body += float(
            L_body.detach()
        )


        epoch_leg[0] += float(
            L_leg1.detach()
        )

        epoch_leg[1] += float(
            L_leg2.detach()
        )

        epoch_leg[2] += float(
            L_leg3.detach()
        )

        epoch_leg[3] += float(
            L_leg4.detach()
        )


        n_batches += 1


        if step % 25 == 0:

            print(

                f"epoch {epoch:02d} | "

                f"step {step:05d}/{len(train_loader):05d} | "

                f"L={loss.item():.5f} | "

                f"vis={L_vis.item():.5f} | "

                f"body={L_body.item():.5f} | "

                f"legs="
                f"{L_leg1.item():.4f},"
                f"{L_leg2.item():.4f},"
                f"{L_leg3.item():.4f},"
                f"{L_leg4.item():.4f}"

            )


    # ------------------------------------------------------------------
    # EPOCH RESULT
    # ------------------------------------------------------------------

    mean_total = (
        epoch_total
        /
        max(n_batches, 1)
    )

    mean_visual = (
        epoch_visual
        /
        max(n_batches, 1)
    )

    mean_body = (
        epoch_body
        /
        max(n_batches, 1)
    )

    mean_legs = (
        epoch_leg
        /
        max(n_batches, 1)
    )


    elapsed = (
        time.time()
        -
        start
    )


    row = {

        "epoch": epoch,

        "loss": mean_total,

        "visual": mean_visual,

        "body": mean_body,

        "leg1": mean_legs[0],

        "leg2": mean_legs[1],

        "leg3": mean_legs[2],

        "leg4": mean_legs[3],

        "seconds": elapsed

    }


    history.append(
        row
    )


    print(
        "\n"
        f"EPOCH {epoch:02d}/{EPOCHS}\n"
        f"  total  = {mean_total:.6f}\n"
        f"  visual = {mean_visual:.6f}\n"
        f"  body   = {mean_body:.6f}\n"
        f"  leg1   = {mean_legs[0]:.6f}\n"
        f"  leg2   = {mean_legs[1]:.6f}\n"
        f"  leg3   = {mean_legs[2]:.6f}\n"
        f"  leg4   = {mean_legs[3]:.6f}\n"
        f"  time   = {elapsed/60:.2f} min\n"
    )


    # ------------------------------------------------------------------
    # SAVE EVERY EPOCH
    # ------------------------------------------------------------------

    state = {

        "epoch":
            epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "history":
            history,

        "BODY_MEAN":
            BODY_MEAN,

        "BODY_STD":
            BODY_STD,

        "LEG_MEAN":
            LEG_MEAN,

        "LEG_STD":
            LEG_STD,

        "BODY_DIM":
            BODY_DIM,

        "LEG_DIM":
            LEG_DIM,

        "architecture":
            {
                "visual_groups": 9,
                "patches_per_group": 256,
                "transformer_tokens": 2304,
                "d_model": 768,
                "layers": 6,
                "heads": 8,
                "mlp": "768->3840->3072->768"
            }

    }


    torch.save(

        state,

        CHECKPOINT_DIR
        /
        "latest.pt"

    )


    if mean_total < best_train_loss:

        best_train_loss = mean_total

        torch.save(

            state,

            CHECKPOINT_DIR
            /
            "best_train.pt"

        )


print(
    "\nTRAINING COMPLETE."
)

print(
    "Latest checkpoint:",
    CHECKPOINT_DIR / "latest.pt"
)

print(
    "Best training-loss checkpoint:",
    CHECKPOINT_DIR / "best_train.pt"
)

# %% notebook cell 3
# ======================================================================
# CELL 3 — TESTING
# ======================================================================
#
# INTENTIONALLY NOT IMPLEMENTED YET.
#
# We still need to decide the actual scientific testing protocol:
#
#   - one-step body physical prediction
#   - per-leg prediction metrics
#   - multi-step autoregressive rollout
#   - temporal-error growth
#   - visual latent prediction
#   - terrain-held-out testing
#   - action sensitivity / counterfactual testing
#   - whether/how to compare against original Feel-WM
#
# Most importantly, we still have to define the exact recursive timing
# of the delayed leg streams:
#
#     legs 1/2 condition at t-1
#     legs 3/4 condition at t-2
#     all predict t+1
#
# I am NOT choosing that recurrence silently.
# ======================================================================

print(
    "Testing intentionally left unimplemented "
    "until the evaluation and recursive-rollout protocol is fixed."
)

# %% notebook cell 4
# ======================================================================
# GRANDTOUR / ANYMAL-D ACTUATOR AUDIT
#
# Goal:
#   Download ONE real anymal_state_actuator stream and determine whether
#   the dataset contains everything required for
#
#       a_i = [
#           Δq_HFE_cmd,
#           Δq_KFE_cmd,
#           tau_HFE_cmd,
#           tau_KFE_cmd
#       ]
#
# DO NOT TRAIN ANYTHING YET.
#
# This cell:
#   1. installs minimal dependencies
#   2. downloads ONE ~331 MB actuator stream
#   3. safely extracts it
#   4. discovers the Zarr group automatically
#   5. recursively prints every array + shape + dtype
#   6. prints all attrs / joint-name metadata
#   7. searches explicitly for:
#        position
#        velocity
#        torque
#        effort
#        current
#        command / desired / target
#   8. prints small REAL samples from candidate arrays
# ======================================================================


# ----------------------------------------------------------------------
# 0. INSTALL
# GrandTour currently uses Zarr-v2-style groups (.zgroup), so keep zarr<3
# for this audit.
# ----------------------------------------------------------------------

# IPYTHON: !pip -q install -U "huggingface_hub>=0.25" "zarr<3" numcodecs pyyaml


# ----------------------------------------------------------------------
# 1. IMPORTS
# ----------------------------------------------------------------------

import os
import re
import json
import tarfile
import shutil
from pathlib import Path

import numpy as np
import zarr

from huggingface_hub import hf_hub_download


# ----------------------------------------------------------------------
# 2. ONE KNOWN GRANDTOUR MISSION
#
# This mission currently contains:
#   anymal_state_actuator.tar  ~331 MB
#
# We are NOT downloading cameras / LiDAR / the entire mission.
# ----------------------------------------------------------------------

REPO_ID = "leggedrobotics/grand_tour_dataset"

MISSION = "2024-11-02-17-10-25"

HF_FILENAME = (
    f"{MISSION}/data/anymal_state_actuator.tar"
)

ROOT = Path("/content/grandtour_actuator_audit")

DOWNLOAD_DIR = ROOT / "download"
EXTRACT_DIR  = ROOT / "extracted"

DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Start extraction cleanly if rerunning.
if EXTRACT_DIR.exists():
    shutil.rmtree(EXTRACT_DIR)

EXTRACT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 80)
print("GRANDTOUR ACTUATOR AUDIT")
print("=" * 80)

print("Repository :", REPO_ID)
print("Mission    :", MISSION)
print("Stream     :", HF_FILENAME)
print()


# ----------------------------------------------------------------------
# 3. DOWNLOAD ONLY THE ACTUATOR TAR
# ----------------------------------------------------------------------

print("Downloading actuator stream only ...")

try:

    tar_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=HF_FILENAME,
        local_dir=DOWNLOAD_DIR,
    )

except Exception as e:

    print("\nDOWNLOAD FAILED")
    print(type(e).__name__, ":", e)

    print(
        "\nIf Hugging Face asks for access/authentication:\n"
        "1. Open the GrandTour Hugging Face page.\n"
        "2. Request/accept dataset access if required.\n"
        "3. In Colab run:\n\n"
        "   from huggingface_hub import notebook_login\n"
        "   notebook_login()\n\n"
        "4. Then rerun this cell."
    )

    raise


tar_path = Path(tar_path)

print("\nDownloaded:")
print(tar_path)

print(
    f"Size: {tar_path.stat().st_size / 1024**2:.1f} MB"
)


# ----------------------------------------------------------------------
# 4. INSPECT TAR CONTENTS BEFORE EXTRACTION
# ----------------------------------------------------------------------

print("\n" + "=" * 80)
print("TAR CONTENTS — FIRST 60 MEMBERS")
print("=" * 80)

with tarfile.open(tar_path, "r:*") as tf:

    members = tf.getmembers()

    print("Total members:", len(members))

    for m in members[:60]:
        print(m.name)

    if len(members) > 60:
        print(f"... {len(members)-60} more members")


# ----------------------------------------------------------------------
# 5. SAFE TAR EXTRACTION
# ----------------------------------------------------------------------

def safe_extract(tf, destination):

    destination = destination.resolve()

    for member in tf.getmembers():

        target = (
            destination / member.name
        ).resolve()

        if not str(target).startswith(
            str(destination)
        ):
            raise RuntimeError(
                f"Unsafe path in archive: {member.name}"
            )

    tf.extractall(destination)


print("\nExtracting ...")

with tarfile.open(tar_path, "r:*") as tf:
    safe_extract(tf, EXTRACT_DIR)

print("Extraction complete.")


# ----------------------------------------------------------------------
# 6. FIND ALL ZARR GROUP MARKERS
# ----------------------------------------------------------------------

zgroup_files = list(
    EXTRACT_DIR.rglob(".zgroup")
)

print("\n" + "=" * 80)
print("ZARR GROUP MARKERS FOUND")
print("=" * 80)

if not zgroup_files:

    print(
        "No .zgroup was found.\n"
        "Printing extracted tree so we can inspect the archive format:"
    )

    for p in list(EXTRACT_DIR.rglob("*"))[:200]:
        print(p.relative_to(EXTRACT_DIR))

    raise RuntimeError(
        "Could not automatically locate Zarr group."
    )


for p in zgroup_files[:40]:
    print(p.relative_to(EXTRACT_DIR))


# ----------------------------------------------------------------------
# 7. FIND THE MOST LIKELY ACTUATOR ROOT
#
# We choose a group that:
#   - is called anymal_state_actuator if present
#   - otherwise choose the shallowest .zgroup
# ----------------------------------------------------------------------

candidate_dirs = [
    p.parent
    for p in zgroup_files
]


named = [
    p
    for p in candidate_dirs
    if "anymal_state_actuator" in str(p)
]


if named:

    ACTUATOR_ROOT = min(
        named,
        key=lambda p: len(p.parts)
    )

else:

    ACTUATOR_ROOT = min(
        candidate_dirs,
        key=lambda p: len(p.parts)
    )


print("\nSelected actuator Zarr root:")
print(ACTUATOR_ROOT)


# ----------------------------------------------------------------------
# 8. OPEN THE ACTUAL ZARR GROUP
# ----------------------------------------------------------------------

root = zarr.open_group(
    str(ACTUATOR_ROOT),
    mode="r"
)


print("\n" + "=" * 80)
print("ROOT ATTRIBUTES")
print("=" * 80)

if len(root.attrs):

    for k, v in root.attrs.items():
        print(f"{k}:")
        print(v)
        print()

else:

    print("(no root attrs)")


# ----------------------------------------------------------------------
# 9. RECURSIVELY DISCOVER EVERY ARRAY
# ----------------------------------------------------------------------

arrays = {}


def walk_group(group, prefix=""):

    # Arrays directly below this group
    for key in group.array_keys():

        arr = group[key]

        full_name = (
            f"{prefix}/{key}"
            if prefix
            else key
        )

        arrays[full_name] = arr


    # Child groups
    for key in group.group_keys():

        child = group[key]

        child_prefix = (
            f"{prefix}/{key}"
            if prefix
            else key
        )

        walk_group(
            child,
            child_prefix
        )


walk_group(root)


print("\n" + "=" * 80)
print("EVERY ARRAY IN anymal_state_actuator")
print("=" * 80)

print(f"Total arrays discovered: {len(arrays)}\n")

for name in sorted(arrays):

    arr = arrays[name]

    print(
        f"{name:65s} "
        f"shape={str(arr.shape):24s} "
        f"dtype={arr.dtype}"
    )


# ----------------------------------------------------------------------
# 10. RECURSIVELY PRINT GROUP ATTRIBUTES TOO
#
# Joint names sometimes live in attrs rather than normal arrays.
# ----------------------------------------------------------------------

print("\n" + "=" * 80)
print("ALL GROUP ATTRIBUTES")
print("=" * 80)


def print_attrs(group, prefix="ROOT"):

    if len(group.attrs):

        print(f"\n[{prefix}]")

        for k, v in group.attrs.items():

            try:
                text = json.dumps(
                    v,
                    indent=2,
                    default=str
                )

            except Exception:
                text = str(v)

            print(f"{k} = {text}")


    for key in group.group_keys():

        child_prefix = (
            f"{prefix}/{key}"
        )

        print_attrs(
            group[key],
            child_prefix
        )


print_attrs(root)


# ----------------------------------------------------------------------
# 11. FIND IMPORTANT FIELDS BY NAME
# ----------------------------------------------------------------------

KEYWORDS = {

    "COMMAND":
        [
            "command",
            "cmd",
            "desired",
            "target",
            "reference",
            "setpoint"
        ],

    "POSITION":
        [
            "position",
            "pos",
            "angle",
            "q"
        ],

    "VELOCITY":
        [
            "velocity",
            "vel",
            "dq"
        ],

    "TORQUE_OR_EFFORT":
        [
            "torque",
            "effort",
            "tau"
        ],

    "CURRENT":
        [
            "current",
            "motor_current"
        ],

    "JOINT_NAMES":
        [
            "name",
            "joint",
            "actuator"
        ],

    "TIMESTAMP":
        [
            "timestamp",
            "time"
        ],
}


print("\n" + "=" * 80)
print("FIELD SEARCH")
print("=" * 80)


matches_by_category = {}


for category, words in KEYWORDS.items():

    matches = []

    for name in arrays:

        low = name.lower()

        if any(
            w in low
            for w in words
        ):

            matches.append(name)


    matches_by_category[category] = matches


    print(f"\n--- {category} ---")

    if matches:

        for name in matches:

            arr = arrays[name]

            print(
                f"{name}  "
                f"shape={arr.shape} "
                f"dtype={arr.dtype}"
            )

    else:

        print("NO FIELD NAME MATCH")


# ----------------------------------------------------------------------
# 12. PRINT SMALL REAL SAMPLES
#
# We deliberately print only a few rows so we can inspect actual values
# without loading a huge 400-Hz array into RAM.
# ----------------------------------------------------------------------

IMPORTANT_WORDS = (

    "command",
    "cmd",
    "desired",
    "target",

    "position",
    "velocity",

    "torque",
    "effort",
    "tau",

    "current",

    "joint",
    "name",

    "timestamp"
)


candidate_names = [

    name
    for name in arrays
    if any(
        word in name.lower()
        for word in IMPORTANT_WORDS
    )
]


print("\n" + "=" * 80)
print("SMALL REAL SAMPLES FROM CANDIDATE ARRAYS")
print("=" * 80)


def read_small_sample(arr):

    # scalar
    if arr.ndim == 0:
        return arr[()]

    # empty
    if arr.shape[0] == 0:
        return "<EMPTY>"

    n = min(
        3,
        arr.shape[0]
    )

    return arr[:n]


for name in sorted(candidate_names):

    arr = arrays[name]

    print("\n" + "-" * 80)

    print("FIELD :", name)
    print("SHAPE :", arr.shape)
    print("DTYPE :", arr.dtype)

    try:

        sample = read_small_sample(arr)

        print("SAMPLE:")
        print(sample)

    except Exception as e:

        print(
            "Could not display sample:",
            type(e).__name__,
            e
        )


# ----------------------------------------------------------------------
# 13. SEARCH METADATA VALUES FOR ACTUAL ANYMAL JOINT NAMES
#
# Expected ANYmal-style strings may include things like:
# LF_HAA, LF_HFE, LF_KFE, RF_..., LH_..., RH_...
#
# We SEARCH; we do not assume they are present.
# ----------------------------------------------------------------------

LEG_TOKENS = [
    "LF",
    "RF",
    "LH",
    "RH"
]

JOINT_TOKENS = [
    "HAA",
    "HFE",
    "KFE"
]


metadata_strings = []


def collect_attr_strings(group, path="ROOT"):

    for k, v in group.attrs.items():

        metadata_strings.append(
            (
                f"{path}:{k}",
                str(v)
            )
        )


    for key in group.group_keys():

        collect_attr_strings(
            group[key],
            f"{path}/{key}"
        )


collect_attr_strings(root)


# Also inspect tiny string/object arrays.

for name, arr in arrays.items():

    if arr.dtype.kind in (
        "U",
        "S",
        "O"
    ):

        try:

            vals = np.asarray(
                arr[:min(len(arr), 100)]
            )

            metadata_strings.append(
                (
                    name,
                    str(vals.tolist())
                )
            )

        except Exception:
            pass


print("\n" + "=" * 80)
print("SEARCH FOR LF/RF/LH/RH + HAA/HFE/KFE")
print("=" * 80)


found_joint_text = False


for source, text in metadata_strings:

    upper = text.upper()

    if (
        any(x in upper for x in LEG_TOKENS)
        or
        any(x in upper for x in JOINT_TOKENS)
    ):

        print(f"\n[{source}]")
        print(text[:5000])

        found_joint_text = True


if not found_joint_text:

    print(
        "No obvious ANYmal joint-name strings were found in attrs/string arrays."
    )


# ----------------------------------------------------------------------
# 14. AUTOMATIC ANSWER TO OUR MAIN QUESTION
# ----------------------------------------------------------------------

all_names_lower = [
    name.lower()
    for name in arrays
]


def contains_any(words):

    return [
        name
        for name in arrays
        if any(
            word in name.lower()
            for word in words
        )
    ]


command_candidates = contains_any(
    [
        "command",
        "cmd",
        "desired",
        "target",
        "reference",
        "setpoint"
    ]
)


torque_candidates = contains_any(
    [
        "torque",
        "effort",
        "tau"
    ]
)


position_candidates = contains_any(
    [
        "position",
        "angle"
    ]
)


print("\n" + "=" * 80)
print("AUDIT SUMMARY")
print("=" * 80)


print("\nPosition-like arrays:")
for x in position_candidates:
    print("  ", x)


print("\nTorque/effort-like arrays:")
for x in torque_candidates:
    print("  ", x)


print("\nCommand/desired/target-like arrays:")
for x in command_candidates:
    print("  ", x)


# Look specifically for fields whose NAME contains BOTH
# a command-like term and torque-like term.

explicit_command_torque = [

    name
    for name in arrays

    if (

        any(
            x in name.lower()
            for x in [
                "command",
                "cmd",
                "desired",
                "target",
                "reference",
                "setpoint"
            ]
        )

        and

        any(
            x in name.lower()
            for x in [
                "torque",
                "effort",
                "tau"
            ]
        )

    )
]


print("\n" + "=" * 80)

if explicit_command_torque:

    print(
        "GOOD: explicit command+torque field name(s) were found:"
    )

    for x in explicit_command_torque:
        print("   ", x)

else:

    print(
        "IMPORTANT: no field NAME containing both a command-like term "
        "and torque/effort-like term was automatically identified."
    )

    print(
        "\nThis does NOT yet prove commanded torque is absent — "
        "the actuator message may store command quantities as components "
        "inside another array. We must inspect the complete output above "
        "before deciding."
    )


print("\n" + "=" * 80)
print("AUDIT FINISHED")
print("=" * 80)

print(
    "\nSend me the output from:\n"
    "  1. EVERY ARRAY IN anymal_state_actuator\n"
    "  2. ALL GROUP ATTRIBUTES\n"
    "  3. FIELD SEARCH\n"
    "  4. AUDIT SUMMARY\n\n"
    "From those four sections I can map the REAL GrandTour channels to "
    "a1,a2,a3,a4 and p1,p2,p3,p4 without guessing."
)

# %% notebook cell 5
# ======================================================================
# GRANDTOUR ACTION VIABILITY AUDIT
#
# Determines whether our EXACT action
#
# a_i = [
#     Δq_HFE_cmd,
#     Δq_KFE_cmd,
#     tau_HFE_cmd,
#     tau_KFE_cmd
# ]
#
# is genuinely populated in GrandTour.
#
# Uses the `root` Zarr group produced by the previous audit cell.
# ======================================================================

import numpy as np
import zarr
from pathlib import Path


# ----------------------------------------------------------------------
# Recover Zarr group automatically if runtime variables disappeared
# ----------------------------------------------------------------------

if "root" not in globals():

    ACTUATOR_ROOT = Path(
        "/content/grandtour_actuator_audit/"
        "extracted/anymal_state_actuator"
    )

    if not ACTUATOR_ROOT.exists():
        raise RuntimeError(
            "The previous GrandTour audit data are not present. "
            "Run the actuator-download/audit cell first."
        )

    root = zarr.open_group(
        str(ACTUATOR_ROOT),
        mode="r"
    )


# ======================================================================
# EXACT JOINT MAPPING FROM GRANDTOUR METADATA
# ======================================================================

JOINTS = {
    0: "LF_HAA",
    1: "LF_HFE",
    2: "LF_KFE",

    3: "RF_HAA",
    4: "RF_HFE",
    5: "RF_KFE",

    6: "LH_HAA",
    7: "LH_HFE",
    8: "LH_KFE",

    9: "RH_HAA",
    10: "RH_HFE",
    11: "RH_KFE",
}


# Our architecture's leg numbering
LEGS = {
    "leg1_LF": {
        "HFE": 1,
        "KFE": 2
    },

    "leg2_RF": {
        "HFE": 4,
        "KFE": 5
    },

    "leg3_LH": {
        "HFE": 7,
        "KFE": 8
    },

    "leg4_RH": {
        "HFE": 10,
        "KFE": 11
    },
}


# ======================================================================
# TIMESTAMP / SAMPLE RATE
# ======================================================================

timestamp = np.asarray(
    root["timestamp"][:],
    dtype=np.float64
)

dt = np.diff(timestamp)

good_dt = dt[
    np.isfinite(dt) & (dt > 0)
]

median_dt = np.median(good_dt)

print("=" * 90)
print("TIMING")
print("=" * 90)

print("samples        :", len(timestamp))
print("duration [s]   :", timestamp[-1] - timestamp[0])
print("median dt [s]  :", median_dt)
print("median rate Hz :", 1.0 / median_dt)


# ======================================================================
# HELPER
# ======================================================================

def stats(x):

    x = np.asarray(
        x,
        dtype=np.float64
    )

    good = x[np.isfinite(x)]

    return {
        "min": float(np.min(good)),
        "max": float(np.max(good)),
        "mean": float(np.mean(good)),
        "std": float(np.std(good)),
        "abs_max": float(np.max(np.abs(good))),
        "nonzero_1e-9": float(
            np.mean(np.abs(good) > 1e-9)
        ),
        "nonzero_1e-6": float(
            np.mean(np.abs(good) > 1e-6)
        ),
        "nonzero_1e-3": float(
            np.mean(np.abs(good) > 1e-3)
        ),
    }


# ======================================================================
# AUDIT EACH REQUIRED HFE / KFE
# ======================================================================

all_actions = {}


for leg_name, ids in LEGS.items():

    hfe = ids["HFE"]
    kfe = ids["KFE"]

    hfe_prefix = f"{hfe:02d}"
    kfe_prefix = f"{kfe:02d}"


    # ------------------------------------------------------------------
    # commanded joint positions
    # ------------------------------------------------------------------

    qh = np.asarray(
        root[
            f"{hfe_prefix}_command_position"
        ][:],
        dtype=np.float64
    )

    qk = np.asarray(
        root[
            f"{kfe_prefix}_command_position"
        ][:],
        dtype=np.float64
    )


    # ------------------------------------------------------------------
    # commanded torques
    # ------------------------------------------------------------------

    tauh = np.asarray(
        root[
            f"{hfe_prefix}_command_joint_torque"
        ][:],
        dtype=np.float64
    )

    tauk = np.asarray(
        root[
            f"{kfe_prefix}_command_joint_torque"
        ][:],
        dtype=np.float64
    )


    # ------------------------------------------------------------------
    # measured torques -- AUDIT ONLY
    #
    # We are NOT substituting these into a_i.
    # ------------------------------------------------------------------

    tauh_state = np.asarray(
        root[
            f"{hfe_prefix}_state_joint_torque"
        ][:],
        dtype=np.float64
    )

    tauk_state = np.asarray(
        root[
            f"{kfe_prefix}_state_joint_torque"
        ][:],
        dtype=np.float64
    )


    # ------------------------------------------------------------------
    # Δq command
    #
    # One actuator-sample change:
    #
    # Δq(t) = q_cmd(t) - q_cmd(t-1)
    #
    # Later, after synchronization to our model time grid,
    # we will calculate Δq over the actual world-model action interval.
    # ------------------------------------------------------------------

    dqh = np.zeros_like(qh)
    dqk = np.zeros_like(qk)

    dqh[1:] = np.diff(qh)
    dqk[1:] = np.diff(qk)


    # ------------------------------------------------------------------
    # Build our EXACT candidate action
    # ------------------------------------------------------------------

    a = np.stack(
        [
            dqh,
            dqk,
            tauh,
            tauk
        ],
        axis=-1
    )

    all_actions[leg_name] = a


    # ------------------------------------------------------------------
    # command mode
    # ------------------------------------------------------------------

    mode_hfe = np.asarray(
        root[
            f"{hfe_prefix}_command_mode"
        ][:]
    )

    mode_kfe = np.asarray(
        root[
            f"{kfe_prefix}_command_mode"
        ][:]
    )


    # ==================================================================
    # PRINT
    # ==================================================================

    print("\n" + "=" * 90)

    print(
        leg_name,
        "|",
        JOINTS[hfe],
        "+",
        JOINTS[kfe]
    )

    print("=" * 90)


    print("\nHFE commanded position")
    print(stats(qh))

    print("\nKFE commanded position")
    print(stats(qk))


    print("\nΔq_HFE command")
    print(stats(dqh))

    print("\nΔq_KFE command")
    print(stats(dqk))


    print("\nHFE COMMANDED torque")
    print(stats(tauh))

    print("\nKFE COMMANDED torque")
    print(stats(tauk))


    print("\nHFE measured torque -- comparison only")
    print(stats(tauh_state))

    print("\nKFE measured torque -- comparison only")
    print(stats(tauk_state))


    print("\nUnique HFE command modes:")
    print(
        np.unique(
            mode_hfe,
            return_counts=True
        )
    )

    print("\nUnique KFE command modes:")
    print(
        np.unique(
            mode_kfe,
            return_counts=True
        )
    )


    print(
        "\nFirst 10 exact candidate a_i rows:"
    )

    print(
        "[Δq_HFE_cmd, Δq_KFE_cmd, "
        "tau_HFE_cmd, tau_KFE_cmd]"
    )

    print(a[:10])


# ======================================================================
# FINAL DECISION
# ======================================================================

print("\n\n" + "#" * 90)
print("FINAL ACTION VIABILITY CHECK")
print("#" * 90)


usable = True


for leg_name, a in all_actions.items():

    tau_hfe = a[:, 2]
    tau_kfe = a[:, 3]

    nz_hfe = np.mean(
        np.abs(tau_hfe) > 1e-6
    )

    nz_kfe = np.mean(
        np.abs(tau_kfe) > 1e-6
    )

    std_hfe = np.std(tau_hfe)
    std_kfe = np.std(tau_kfe)


    print(
        f"\n{leg_name}:"
    )

    print(
        f"  commanded HFE torque "
        f"nonzero fraction = {nz_hfe:.6f}, "
        f"std = {std_hfe:.6f}"
    )

    print(
        f"  commanded KFE torque "
        f"nonzero fraction = {nz_kfe:.6f}, "
        f"std = {std_kfe:.6f}"
    )


    if (
        std_hfe < 1e-8
        or
        std_kfe < 1e-8
    ):
        usable = False


print("\n" + "-" * 90)


if usable:

    print(
        "RESULT: commanded HFE/KFE torques vary in this run.\n"
        "GrandTour can directly support the torque components "
        "of our exact 4-D leg action."
    )

else:

    print(
        "RESULT: at least one required commanded-torque stream "
        "is effectively constant.\n\n"
        "Do NOT replace it with measured torque automatically.\n"
        "If this occurs for all missions, GrandTour cannot "
        "provide our exact chosen 4-D action definition."
    )


print("#" * 90)
