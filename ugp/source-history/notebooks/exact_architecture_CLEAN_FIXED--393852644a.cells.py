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

# %% notebook cell 2
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

# %% notebook cell 3
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

# Ensure the required Zarr v2 reader exists even in a fresh Colab runtime.
# IPYTHON: !pip -q install "zarr<3"

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

# %% notebook cell 4
# ======================================================================
# GRANDTOUR -> FIVE-STREAM-FEELWM PREPROCESSING
# STAGE A: NUMERICAL SYNCHRONIZATION + MODEL-READY INDEXING
#
# THIS CELL BUILDS:
#
# body_hist_t       [N, 50, D_body]
#
# leg1_hist_tm1     [N, 50, D_leg]
# leg2_hist_tm1     [N, 50, D_leg]
# leg3_hist_tm2     [N, 50, D_leg]
# leg4_hist_tm2     [N, 50, D_leg]
#
# a_body_t          [N, 4]
#
# a1_tm1            [N, 4]
# a2_tm1            [N, 4]
# a3_tm2            [N, 4]
# a4_tm2            [N, 4]
#
# leg_target_tp1    [N, 4, 4]
#
# RGB context indices corresponding to
#       z_(t-3), z_(t-2), z_(t-1), z_t
#
# and RGB target index corresponding to
#       z_(t+1)
#
#
# IMPORTANT:
#
# It DOES NOT yet:
#
#   - download the ~1.38 GB ZED2i RGB archive
#   - compute DINOv2 tokens z_t
#   - fabricate foot-slip
#   - fabricate body_target
#
# We first verify that ALL numerical streams line up correctly.
# ======================================================================


# ======================================================================
# 0. INSTALL
# ======================================================================

# IPYTHON: !pip -q install -U "huggingface_hub>=0.25" "zarr<3" scipy


# ======================================================================
# 1. IMPORTS
# ======================================================================

import os
import json
import math
import tarfile
import shutil

from pathlib import Path

import numpy as np
import zarr

from scipy.spatial.transform import Rotation

from huggingface_hub import hf_hub_download


# ======================================================================
# 2. CONFIGURATION
# ======================================================================

REPO_ID = "leggedrobotics/grand_tour_dataset"

MISSION = "2024-11-02-17-10-25"


# ----------------------------------------------------------
# WORLD-MODEL CLOCK
#
# One t -> t+1 transition = 0.1 s
#
# This can later be changed, but ALL streams use the same DT.
# ----------------------------------------------------------

MODEL_DT = 0.10


# ----------------------------------------------------------
# Proprioception:
#
# 50 measurements over 0.5 s
#
# therefore:
#
# 100 Hz * 0.5 s = 50 measurements
# ----------------------------------------------------------

PROP_HZ = 100.0

PROP_DT = 1.0 / PROP_HZ

PROP_HISTORY = 50


# ----------------------------------------------------------
# Camera synchronization tolerance.
#
# Automatically increased later if the actual camera rate
# requires a slightly larger nearest-frame window.
# ----------------------------------------------------------

CAMERA_TOL_MIN = 0.050


# ----------------------------------------------------------
# Storage
# ----------------------------------------------------------

PIPE_ROOT = Path(
    "/content/grandtour_fivestream_pipeline"
)

DOWNLOAD_ROOT = PIPE_ROOT / "downloads"

EXTRACT_ROOT = PIPE_ROOT / "extracted"

OUTPUT_ROOT = PIPE_ROOT / "processed"


for p in [
    DOWNLOAD_ROOT,
    EXTRACT_ROOT,
    OUTPUT_ROOT
]:
    p.mkdir(
        parents=True,
        exist_ok=True
    )


print("=" * 90)
print("GRANDTOUR -> FIVE-STREAM PREPROCESSING")
print("=" * 90)

print("Mission        :", MISSION)
print("Model dt       :", MODEL_DT)
print("Proprio rate   :", PROP_HZ)
print("History length :", PROP_HISTORY)


# ======================================================================
# 3. SAFE TAR EXTRACTION
# ======================================================================

def safe_extract(
    tar_path,
    destination
):

    destination.mkdir(
        parents=True,
        exist_ok=True
    )

    destination_resolved = (
        destination.resolve()
    )

    with tarfile.open(
        tar_path,
        "r:*"
    ) as tf:

        for member in tf.getmembers():

            target = (
                destination
                /
                member.name
            ).resolve()

            if not str(target).startswith(
                str(destination_resolved)
            ):

                raise RuntimeError(
                    f"Unsafe archive path: "
                    f"{member.name}"
                )

        tf.extractall(
            destination
        )


# ======================================================================
# 4. FIND A ZARR GROUP
# ======================================================================

def find_stream_zarr(
    folder,
    stream_name
):

    candidates = []

    for marker in folder.rglob(
        ".zgroup"
    ):

        parent = marker.parent

        if (
            parent.name == stream_name
            or
            stream_name in str(parent)
        ):

            candidates.append(
                parent
            )


    if not candidates:

        return None


    return min(
        candidates,
        key=lambda x: len(x.parts)
    )


# ======================================================================
# 5. DOWNLOAD / OPEN SMALL NUMERIC STREAMS
# ======================================================================

def get_stream(
    stream_name
):

    # ----------------------------------------------------------
    # Reuse the actuator stream from YOUR PREVIOUS AUDIT.
    # This avoids downloading the 331 MB file again.
    # ----------------------------------------------------------

    if (
        stream_name
        ==
        "anymal_state_actuator"
    ):

        # Previous audit variable still alive.
        if (
            "root" in globals()
            and
            isinstance(
                globals()["root"],
                zarr.hierarchy.Group
            )
            and
            globals()["root"].attrs.get(
                "topic",
                ""
            )
            ==
            "anymal_state_actuator"
        ):

            print(
                "Reusing actuator group "
                "already loaded in RAM."
            )

            return globals()["root"]


        previous_audit = Path(
            "/content/"
            "grandtour_actuator_audit/"
            "extracted"
        )

        if previous_audit.exists():

            found = find_stream_zarr(
                previous_audit,
                stream_name
            )

            if found is not None:

                print(
                    "Reusing previously "
                    "downloaded actuator:"
                )

                print(found)

                return zarr.open_group(
                    str(found),
                    mode="r"
                )


    # ----------------------------------------------------------
    # Check our new preprocessing folder.
    # ----------------------------------------------------------

    target_extract = (
        EXTRACT_ROOT
        /
        stream_name
    )

    found = (
        find_stream_zarr(
            target_extract,
            stream_name
        )
        if target_extract.exists()
        else None
    )


    if found is None:

        hf_file = (
            f"{MISSION}/data/"
            f"{stream_name}.tar"
        )

        print(
            f"\nDownloading {stream_name} ..."
        )

        tar_path = Path(

            hf_hub_download(

                repo_id=REPO_ID,

                repo_type="dataset",

                filename=hf_file,

                local_dir=
                    DOWNLOAD_ROOT
            )

        )


        print(
            f"Downloaded "
            f"{tar_path.stat().st_size/1024**2:.1f} MB"
        )


        if target_extract.exists():

            shutil.rmtree(
                target_extract
            )


        target_extract.mkdir(
            parents=True,
            exist_ok=True
        )


        print(
            f"Extracting {stream_name} ..."
        )


        safe_extract(
            tar_path,
            target_extract
        )


        found = find_stream_zarr(
            target_extract,
            stream_name
        )


    if found is None:

        raise RuntimeError(
            f"Could not locate Zarr group "
            f"for {stream_name}"
        )


    print(
        "Opened:",
        found
    )


    return zarr.open_group(
        str(found),
        mode="r"
    )


# ======================================================================
# 6. OPEN EXACT STREAMS REQUIRED FOR THIS STAGE
# ======================================================================

actuator = get_stream(
    "anymal_state_actuator"
)

imu = get_stream(
    "anymal_imu"
)

odom = get_stream(
    "anymal_state_odometry"
)

# Only the ~2 MB Zarr metadata/timestamps.
# NOT the 1.38 GB actual image archive.

camera = get_stream(
    "zed2i_left_images"
)


# ======================================================================
# 7. ARRAY DISCOVERY UTILITIES
# ======================================================================

def flatten_arrays(
    group,
    prefix=""
):

    out = {}


    for key in group.array_keys():

        name = (
            f"{prefix}/{key}"
            if prefix
            else key
        )

        out[name] = group[key]


    for key in group.group_keys():

        child_prefix = (
            f"{prefix}/{key}"
            if prefix
            else key
        )

        out.update(

            flatten_arrays(
                group[key],
                child_prefix
            )

        )


    return out


def print_stream_schema(
    name,
    group
):

    print(
        "\n"
        +
        "=" * 90
    )

    print(
        f"{name} SCHEMA"
    )

    print(
        "=" * 90
    )


    for k, arr in sorted(
        flatten_arrays(group).items()
    ):

        print(
            f"{k:45s}"
            f" shape={str(arr.shape):20s}"
            f" dtype={arr.dtype}"
        )


# Print only the newly introduced schemas.

print_stream_schema(
    "ANYMAL IMU",
    imu
)

print_stream_schema(
    "ANYMAL ODOMETRY",
    odom
)

print_stream_schema(
    "ZED2I LEFT METADATA",
    camera
)


# ======================================================================
# 8. ROBUST FIELD PICKER
# ======================================================================

def pick_array(
    group,
    token_alternatives,
    width=None,
    exclude=("cov",)
):

    """
    token_alternatives example:

        [
            ("linear", "acc"),
            ("lin", "acc")
        ]

    Every token in ONE tuple must be present.
    """

    arrays = flatten_arrays(
        group
    )

    matches = []


    for name, arr in arrays.items():

        low = name.lower()


        if any(
            x in low
            for x in exclude
        ):

            continue


        if not any(

            all(
                token in low
                for token in tokens
            )

            for tokens
            in token_alternatives

        ):

            continue


        if width is not None:

            if (
                arr.ndim < 2
                or
                arr.shape[-1]
                != width
            ):

                continue


        matches.append(
            (name, arr)
        )


    if len(matches) == 1:

        print(
            "Mapped:",
            matches[0][0]
        )

        return matches[0][1]


    if len(matches) == 0:

        raise RuntimeError(
            "No array matched tokens "
            f"{token_alternatives}"
        )


    print(
        "\nAMBIGUOUS FIELD SEARCH:"
    )

    for n, a in matches:

        print(
            n,
            a.shape
        )


    raise RuntimeError(
        f"Multiple arrays matched "
        f"{token_alternatives}. "
        "Do not guess."
    )


# ======================================================================
# 9. TIMESTAMPS
# ======================================================================

t_act = np.asarray(
    actuator["timestamp"][:],
    dtype=np.float64
)

t_imu = np.asarray(
    imu["timestamp"][:],
    dtype=np.float64
)

t_odom = np.asarray(
    odom["timestamp"][:],
    dtype=np.float64
)

t_cam = np.asarray(
    camera["timestamp"][:],
    dtype=np.float64
)


for name, t in [

    ("actuator", t_act),
    ("imu", t_imu),
    ("odometry", t_odom),
    ("camera", t_cam),

]:

    assert np.all(
        np.diff(t) > 0
    ), (
        f"{name} timestamps are "
        "not strictly increasing."
    )


    hz = (
        1.0
        /
        np.median(
            np.diff(t)
        )
    )


    print(
        f"{name:10s}: "
        f"N={len(t):7d}  "
        f"median rate={hz:8.2f} Hz"
    )


cam_dt = np.median(
    np.diff(t_cam)
)

CAMERA_TOL = max(
    CAMERA_TOL_MIN,
    0.75 * cam_dt
)


print(
    "\nCamera nearest-frame tolerance:",
    CAMERA_TOL,
    "s"
)


# ======================================================================
# 10. IDENTIFY BODY IMU CHANNELS
#
# Feel-WM's p_t uses:
#
#   inertial acceleration
#   angular rate
#   body velocity
#   attitude
#   actuator readings
#
# We construct those exact categories.
# ======================================================================

imu_acc_arr = pick_array(

    imu,

    [
        ("linear", "acc"),
        ("lin", "acc"),
    ],

    width=3
)


imu_gyro_arr = pick_array(

    imu,

    [
        ("angular", "vel"),
        ("ang", "vel"),
    ],

    width=3
)


# ======================================================================
# 11. ODOMETRY POSE
#
# GrandTour official examples expose:
#
#   pose_pos
#   pose_orien
#
# pose_orien is ROS quaternion order:
#
#   [qx, qy, qz, qw]
# ======================================================================

if (
    "pose_pos" not in odom
    or
    "pose_orien" not in odom
):

    raise RuntimeError(
        "Expected GrandTour odometry fields "
        "'pose_pos' and 'pose_orien' "
        "were not found."
    )


odom_pos = np.asarray(
    odom["pose_pos"][:],
    dtype=np.float64
)

odom_quat = np.asarray(
    odom["pose_orien"][:],
    dtype=np.float64
)


assert (
    odom_pos.ndim == 2
    and
    odom_pos.shape[1] == 3
)

assert (
    odom_quat.ndim == 2
    and
    odom_quat.shape[1] == 4
)


# Normalize quaternion.

odom_quat /= np.maximum(

    np.linalg.norm(
        odom_quat,
        axis=1,
        keepdims=True
    ),

    1e-12
)


# ======================================================================
# 12. NEAREST-SAMPLE INDEX
# ======================================================================

def nearest_indices(
    timestamps,
    query
):

    timestamps = np.asarray(
        timestamps,
        dtype=np.float64
    )

    q = np.asarray(
        query,
        dtype=np.float64
    )


    right = np.searchsorted(
        timestamps,
        q,
        side="left"
    )


    right = np.clip(
        right,
        1,
        len(timestamps) - 1
    )


    left = right - 1


    choose_right = (

        np.abs(
            timestamps[right]
            -
            q
        )

        <

        np.abs(
            q
            -
            timestamps[left]
        )

    )


    return np.where(
        choose_right,
        right,
        left
    )


# ======================================================================
# 13. BUILD UNIFORM 100-Hz PROPRIOCEPTION CLOCK
# ======================================================================

common_raw_start = max(

    t_act[0],
    t_imu[0],
    t_odom[0]

)

common_raw_end = min(

    t_act[-1],
    t_imu[-1],
    t_odom[-1]

)


t_prop = np.arange(

    common_raw_start,

    common_raw_end,

    PROP_DT,

    dtype=np.float64

)


print(
    "\n100-Hz proprio grid:",
    len(t_prop),
    "samples"
)


idx_act_prop = nearest_indices(
    t_act,
    t_prop
)

idx_imu_prop = nearest_indices(
    t_imu,
    t_prop
)

idx_odom_prop = nearest_indices(
    t_odom,
    t_prop
)


# ======================================================================
# 14. BODY PROPRIOCEPTION p_t
# ======================================================================

imu_acc = np.asarray(
    imu_acc_arr[:],
    dtype=np.float64
)[idx_imu_prop]


imu_gyro = np.asarray(
    imu_gyro_arr[:],
    dtype=np.float64
)[idx_imu_prop]


body_pos = (
    odom_pos[
        idx_odom_prop
    ]
)


body_quat = (
    odom_quat[
        idx_odom_prop
    ]
)


body_rpy = (

    Rotation
    .from_quat(
        body_quat
    )
    .as_euler(
        "xyz",
        degrees=False
    )

)


# ----------------------------------------------------------
# Ground/world velocity from synchronized odometry.
# ----------------------------------------------------------

vel_world = np.gradient(

    body_pos,

    PROP_DT,

    axis=0

)


# ----------------------------------------------------------
# Rotate XY velocity into body frame using yaw.
# ----------------------------------------------------------

yaw = body_rpy[:, 2]

cy = np.cos(yaw)
sy = np.sin(yaw)


vx_body = (
    cy * vel_world[:, 0]
    +
    sy * vel_world[:, 1]
)

vy_body = (
    -sy * vel_world[:, 0]
    +
    cy * vel_world[:, 1]
)

vz_body = vel_world[:, 2]


body_velocity = np.stack(

    [
        vx_body,
        vy_body,
        vz_body
    ],

    axis=1

)


# ======================================================================
# 15. GLOBAL ACTUATOR READINGS FOR p_t
#
# For the whole-body p_t we use:
#
#   all 12 measured joint positions
#   all 12 measured joint velocities
#   all 12 measured joint torques
#
# exactly as actuator readings, separate from commands.
# ======================================================================

joint_q = []
joint_dq = []
joint_tau = []


for j in range(12):

    prefix = f"{j:02d}"


    joint_q.append(

        np.asarray(
            actuator[
                f"{prefix}_state_joint_position"
            ][:]
        )[idx_act_prop]

    )


    joint_dq.append(

        np.asarray(
            actuator[
                f"{prefix}_state_joint_velocity"
            ][:]
        )[idx_act_prop]

    )


    joint_tau.append(

        np.asarray(
            actuator[
                f"{prefix}_state_joint_torque"
            ][:]
        )[idx_act_prop]

    )


joint_q = np.stack(
    joint_q,
    axis=1
)

joint_dq = np.stack(
    joint_dq,
    axis=1
)

joint_tau = np.stack(
    joint_tau,
    axis=1
)


# ----------------------------------------------------------
# Whole-body raw proprio vector:
#
# acc                 3
# gyro                3
# body velocity       3
# attitude r,p,y      3
# joint q            12
# joint dq           12
# joint tau          12
#
# TOTAL = 48
# ----------------------------------------------------------

body_prop_100hz = np.concatenate(

    [

        imu_acc,

        imu_gyro,

        body_velocity,

        body_rpy,

        joint_q,

        joint_dq,

        joint_tau,

    ],

    axis=1

).astype(
    np.float32
)


BODY_FEATURE_NAMES = (

    [
        "imu_ax",
        "imu_ay",
        "imu_az",

        "imu_wx",
        "imu_wy",
        "imu_wz",

        "body_vx",
        "body_vy",
        "body_vz",

        "roll",
        "pitch",
        "yaw",
    ]

    +

    [
        f"joint{j:02d}_q"
        for j in range(12)
    ]

    +

    [
        f"joint{j:02d}_dq"
        for j in range(12)
    ]

    +

    [
        f"joint{j:02d}_tau"
        for j in range(12)
    ]
)


assert (
    body_prop_100hz.shape[1]
    ==
    len(BODY_FEATURE_NAMES)
)


print(
    "\nBODY p_t raw dimension:",
    body_prop_100hz.shape[1]
)


# ======================================================================
# 16. PER-LEG PROPRIOCEPTION p1,p2,p3,p4
#
# We include ALL continuous actuator-state feedback
# corresponding to each leg while excluding:
#
#   covariance matrices
#   statusword bitmasks
#
# For each of the 3 actuators in a leg:
#
# state_current             1
# state_gear_position       1
# state_gear_velocity       1
# state_imu_ang_vel         3
# state_imu_lin_acc         3
# state_imu_orien           4
# state_joint_acceleration  1
# state_joint_position      1
# state_joint_torque        1
# state_joint_velocity      1
#
# = 17 values / actuator
#
# 3 actuators / leg
#
# => D_leg = 51
# ======================================================================

LEG_JOINTS = {

    0: [0, 1, 2],      # LF
    1: [3, 4, 5],      # RF
    2: [6, 7, 8],      # LH
    3: [9, 10, 11],    # RH

}


LEG_NAMES = [
    "LF",
    "RF",
    "LH",
    "RH"
]


LEG_STATE_SUFFIXES = [

    "state_current",

    "state_gear_position",

    "state_gear_velocity",

    "state_imu_ang_vel",

    "state_imu_lin_acc",

    "state_imu_orien",

    "state_joint_acceleration",

    "state_joint_position",

    "state_joint_torque",

    "state_joint_velocity",

]


def build_leg_prop(
    joint_ids
):

    features = []

    names = []


    for j in joint_ids:

        prefix = f"{j:02d}"


        for suffix in (
            LEG_STATE_SUFFIXES
        ):

            key = (
                f"{prefix}_{suffix}"
            )


            if key not in actuator:

                raise RuntimeError(
                    f"Missing required "
                    f"actuator field: {key}"
                )


            raw = np.asarray(
                actuator[key][:]
            )


            sampled = (
                raw[
                    idx_act_prop
                ]
            )


            if sampled.ndim == 1:

                sampled = (
                    sampled[:, None]
                )


            features.append(
                sampled.astype(
                    np.float32
                )
            )


            for k in range(
                sampled.shape[1]
            ):

                suffix_name = (

                    suffix

                    if sampled.shape[1]
                    ==
                    1

                    else

                    f"{suffix}_{k}"

                )


                names.append(

                    f"joint{j:02d}_"
                    f"{suffix_name}"

                )


    return (

        np.concatenate(
            features,
            axis=1
        ),

        names

    )


leg_prop_100hz = []

LEG_FEATURE_NAMES = []


for leg in range(4):

    x, names = build_leg_prop(
        LEG_JOINTS[leg]
    )

    leg_prop_100hz.append(
        x
    )

    LEG_FEATURE_NAMES.append(
        names
    )


leg_prop_100hz = np.stack(

    leg_prop_100hz,

    axis=1

)


# shape:
#
# [T100, 4, D_leg]


D_LEG = (
    leg_prop_100hz.shape[-1]
)


print(
    "LEG p_i raw dimension:",
    D_LEG
)


print(
    "leg_prop_100hz shape:",
    leg_prop_100hz.shape
)


# ======================================================================
# 17. EXACT PER-LEG ACTION MAPPING
#
# Locked definition:
#
# a_i =
#
# [
#   Δq_HFE_cmd,
#   Δq_KFE_cmd,
#   tau_HFE_cmd,
#   tau_KFE_cmd
# ]
#
# ======================================================================

LEG_HFE_KFE = {

    0: (1, 2),       # LF

    1: (4, 5),       # RF

    2: (7, 8),       # LH

    3: (10, 11),     # RH

}


def actuator_value_at(
    key,
    times
):

    idx = nearest_indices(
        t_act,
        times
    )

    return np.asarray(
        actuator[key][:]
    )[idx]


def leg_action_at(
    leg_id,
    start_times
):

    """
    Action over one world-model transition.

    Δq command is computed OVER MODEL_DT:

        q_cmd(t + ΔT) - q_cmd(t)

    Torque components are commanded torque at
    the beginning of that transition.
    """

    hfe, kfe = (
        LEG_HFE_KFE[
            leg_id
        ]
    )


    ph = f"{hfe:02d}"
    pk = f"{kfe:02d}"


    qh_0 = actuator_value_at(

        f"{ph}_command_position",

        start_times

    )


    qh_1 = actuator_value_at(

        f"{ph}_command_position",

        start_times
        +
        MODEL_DT

    )


    qk_0 = actuator_value_at(

        f"{pk}_command_position",

        start_times

    )


    qk_1 = actuator_value_at(

        f"{pk}_command_position",

        start_times
        +
        MODEL_DT

    )


    tau_h = actuator_value_at(

        f"{ph}_command_joint_torque",

        start_times

    )


    tau_k = actuator_value_at(

        f"{pk}_command_joint_torque",

        start_times

    )


    return np.stack(

        [

            qh_1 - qh_0,

            qk_1 - qk_0,

            tau_h,

            tau_k

        ],

        axis=-1

    ).astype(
        np.float32
    )


# ======================================================================
# 18. WHOLE-BODY ACTION a_t
#
# EXACT FORMAT:
#
# [
#   Δx_body,
#   Δy_body,
#   Δpsi,
#   Δt
# ]
#
# Computed from recorded odometry poses.
# ======================================================================

def wrap_angle(
    x
):

    return (
        x
        +
        np.pi
    ) % (
        2 * np.pi
    ) - np.pi


def odom_pose_at(
    times
):

    idx = nearest_indices(
        t_odom,
        times
    )


    p = (
        odom_pos[
            idx
        ]
    )


    q = (
        odom_quat[
            idx
        ]
    )


    rpy = (

        Rotation
        .from_quat(q)
        .as_euler(
            "xyz",
            degrees=False
        )

    )


    return (
        p,
        rpy
    )


def body_action_at(
    start_times
):

    p0, rpy0 = odom_pose_at(
        start_times
    )


    p1, rpy1 = odom_pose_at(

        start_times
        +
        MODEL_DT

    )


    d_world = (
        p1 - p0
    )


    yaw0 = rpy0[:, 2]


    c = np.cos(
        yaw0
    )

    s = np.sin(
        yaw0
    )


    # world delta -> start-body frame

    dx_body = (

        c * d_world[:, 0]

        +

        s * d_world[:, 1]

    )


    dy_body = (

        -s * d_world[:, 0]

        +

        c * d_world[:, 1]

    )


    dpsi = wrap_angle(

        rpy1[:, 2]

        -

        rpy0[:, 2]

    )


    dt = np.full_like(

        dx_body,

        MODEL_DT

    )


    return np.stack(

        [

            dx_body,

            dy_body,

            dpsi,

            dt

        ],

        axis=-1

    ).astype(
        np.float32
    )


# ======================================================================
# 19. LEG TARGET
#
# Exclusive 4-D future target for every leg:
#
# [
#   q_HFE_measured,
#   q_KFE_measured,
#   tau_HFE_measured,
#   tau_KFE_measured
# ]_(t+1)
# ======================================================================

def leg_target_at(
    target_times
):

    out = []


    for leg in range(4):

        hfe, kfe = (
            LEG_HFE_KFE[
                leg
            ]
        )


        ph = f"{hfe:02d}"
        pk = f"{kfe:02d}"


        qh = actuator_value_at(

            f"{ph}_state_joint_position",

            target_times

        )


        qk = actuator_value_at(

            f"{pk}_state_joint_position",

            target_times

        )


        th = actuator_value_at(

            f"{ph}_state_joint_torque",

            target_times

        )


        tk = actuator_value_at(

            f"{pk}_state_joint_torque",

            target_times

        )


        out.append(

            np.stack(

                [
                    qh,
                    qk,
                    th,
                    tk
                ],

                axis=-1

            )

        )


    return np.stack(

        out,

        axis=1

    ).astype(
        np.float32
    )


# ======================================================================
# 20. DETERMINE VALID MODEL TIME RANGE
#
# Need enough past data for:
#
# z_(t-3)
#
# and:
#
# leg3/leg4 history ending at t-2,
# containing 0.5 s of previous measurements.
#
# Also need t+1 target.
# ======================================================================

required_history_seconds = (

    (PROP_HISTORY - 1)
    *
    PROP_DT

)


start_numeric = max(

    common_raw_start
    +
    required_history_seconds
    +
    2 * MODEL_DT,

    t_cam[0]
    +
    3 * MODEL_DT

)


end_numeric = min(

    common_raw_end
    -
    MODEL_DT,

    t_cam[-1]
    -
    MODEL_DT

)


if (
    end_numeric
    <=
    start_numeric
):

    raise RuntimeError(
        "No valid overlap between "
        "camera and proprioception streams."
    )


t_model = np.arange(

    start_numeric,

    end_numeric,

    MODEL_DT,

    dtype=np.float64

)


print(
    "\nInitial model transitions:",
    len(t_model)
)


# ======================================================================
# 21. CAMERA INDEX MAPPING
#
# For every training transition t:
#
# context RGB:
#
#   t-3
#   t-2
#   t-1
#   t
#
# target RGB:
#
#   t+1
# ======================================================================

context_offsets = np.array(

    [
        -3,
        -2,
        -1,
        0
    ],

    dtype=np.float64

) * MODEL_DT


context_times = (

    t_model[:, None]

    +

    context_offsets[
        None, :
    ]

)


target_visual_times = (

    t_model
    +
    MODEL_DT

)


rgb_context_idx = nearest_indices(

    t_cam,

    context_times.reshape(-1)

).reshape(

    len(t_model),
    4

)


rgb_target_idx = nearest_indices(

    t_cam,

    target_visual_times

)


rgb_context_error = np.abs(

    t_cam[
        rgb_context_idx
    ]

    -

    context_times

)


rgb_target_error = np.abs(

    t_cam[
        rgb_target_idx
    ]

    -

    target_visual_times

)


camera_valid = (

    np.all(

        rgb_context_error
        <=
        CAMERA_TOL,

        axis=1

    )

    &

    (
        rgb_target_error
        <=
        CAMERA_TOL
    )

)


print(
    "Camera-valid transitions:",
    int(
        camera_valid.sum()
    ),
    "/",
    len(camera_valid)
)


print(
    "Worst retained context mismatch:",
    float(
        rgb_context_error[
            camera_valid
        ].max()
    )
    if camera_valid.any()
    else None
)


# Filter to actual camera-supported samples.

t_model = (
    t_model[
        camera_valid
    ]
)

rgb_context_idx = (
    rgb_context_idx[
        camera_valid
    ]
)

rgb_target_idx = (
    rgb_target_idx[
        camera_valid
    ]
)


# ======================================================================
# 22. HISTORY EXTRACTOR
# ======================================================================

def history_ending_at(
    signal,
    ending_times
):

    """
    signal is already on the 100-Hz uniform grid.

    Returns:
        [N,50,D]
    """

    end_idx = nearest_indices(

        t_prop,

        ending_times

    )


    all_hist = []


    for idx in end_idx:

        start = (
            idx
            -
            PROP_HISTORY
            +
            1
        )


        if start < 0:

            raise RuntimeError(
                "History starts before "
                "100-Hz data."
            )


        h = signal[
            start:
            idx + 1
        ]


        if (
            len(h)
            !=
            PROP_HISTORY
        ):

            raise RuntimeError(
                "History-length mismatch."
            )


        all_hist.append(
            h
        )


    return np.stack(
        all_hist,
        axis=0
    ).astype(
        np.float32
    )


# ======================================================================
# 23. EXACT TEMPORAL ALIGNMENT REQUESTED BY YOUR ARCHITECTURE
# ======================================================================

# ----------------------------------------------------------
# BODY:
#
# p_t
# ----------------------------------------------------------

body_hist_t = history_ending_at(

    body_prop_100hz,

    t_model

)


# ----------------------------------------------------------
# LEGS 1 / 2:
#
# p1_(t-1)
# p2_(t-1)
# ----------------------------------------------------------

leg1_hist_tm1 = history_ending_at(

    leg_prop_100hz[:, 0],

    t_model
    -
    MODEL_DT

)


leg2_hist_tm1 = history_ending_at(

    leg_prop_100hz[:, 1],

    t_model
    -
    MODEL_DT

)


# ----------------------------------------------------------
# LEGS 3 / 4:
#
# p3_(t-2)
# p4_(t-2)
# ----------------------------------------------------------

leg3_hist_tm2 = history_ending_at(

    leg_prop_100hz[:, 2],

    t_model
    -
    2 * MODEL_DT

)


leg4_hist_tm2 = history_ending_at(

    leg_prop_100hz[:, 3],

    t_model
    -
    2 * MODEL_DT

)


# ======================================================================
# 24. EXACT ACTION ALIGNMENT
# ======================================================================

a_body_t = body_action_at(
    t_model
)


a1_tm1 = leg_action_at(

    0,

    t_model
    -
    MODEL_DT

)


a2_tm1 = leg_action_at(

    1,

    t_model
    -
    MODEL_DT

)


a3_tm2 = leg_action_at(

    2,

    t_model
    -
    2 * MODEL_DT

)


a4_tm2 = leg_action_at(

    3,

    t_model
    -
    2 * MODEL_DT

)


# ======================================================================
# 25. t+1 LEG SUPERVISION
# ======================================================================

leg_target_tp1 = leg_target_at(

    t_model
    +
    MODEL_DT

)


# ======================================================================
# 26. FINITE-VALUE VALIDATION
# ======================================================================

numeric_objects = {

    "body_hist_t":
        body_hist_t,

    "leg1_hist_tm1":
        leg1_hist_tm1,

    "leg2_hist_tm1":
        leg2_hist_tm1,

    "leg3_hist_tm2":
        leg3_hist_tm2,

    "leg4_hist_tm2":
        leg4_hist_tm2,

    "a_body_t":
        a_body_t,

    "a1_tm1":
        a1_tm1,

    "a2_tm1":
        a2_tm1,

    "a3_tm2":
        a3_tm2,

    "a4_tm2":
        a4_tm2,

    "leg_target_tp1":
        leg_target_tp1,

}


valid = np.ones(

    len(t_model),

    dtype=bool

)


for name, x in (
    numeric_objects.items()
):

    axes = tuple(
        range(
            1,
            x.ndim
        )
    )


    finite = np.isfinite(
        x
    ).all(
        axis=axes
    )


    print(
        f"{name:20s}: "
        f"{int(finite.sum())}/"
        f"{len(finite)} finite"
    )


    valid &= finite


# Apply common valid mask.

t_model = (
    t_model[
        valid
    ]
)


rgb_context_idx = (
    rgb_context_idx[
        valid
    ]
)

rgb_target_idx = (
    rgb_target_idx[
        valid
    ]
)


for name in list(
    numeric_objects.keys()
):

    numeric_objects[name] = (
        numeric_objects[name][
            valid
        ]
    )


body_hist_t = (
    numeric_objects[
        "body_hist_t"
    ]
)

leg1_hist_tm1 = (
    numeric_objects[
        "leg1_hist_tm1"
    ]
)

leg2_hist_tm1 = (
    numeric_objects[
        "leg2_hist_tm1"
    ]
)

leg3_hist_tm2 = (
    numeric_objects[
        "leg3_hist_tm2"
    ]
)

leg4_hist_tm2 = (
    numeric_objects[
        "leg4_hist_tm2"
    ]
)

a_body_t = (
    numeric_objects[
        "a_body_t"
    ]
)

a1_tm1 = (
    numeric_objects[
        "a1_tm1"
    ]
)

a2_tm1 = (
    numeric_objects[
        "a2_tm1"
    ]
)

a3_tm2 = (
    numeric_objects[
        "a3_tm2"
    ]
)

a4_tm2 = (
    numeric_objects[
        "a4_tm2"
    ]
)

leg_target_tp1 = (
    numeric_objects[
        "leg_target_tp1"
    ]
)


# ======================================================================
# 27. EXACT SHAPE AUDIT
# ======================================================================

N = len(
    t_model
)


print(
    "\n"
    +
    "=" * 90
)

print(
    "FINAL MODEL-READY NUMERICAL SHAPES"
)

print(
    "=" * 90
)


print(
    "samples             ",
    N
)


print(
    "body_hist_t         ",
    body_hist_t.shape
)


print(
    "leg1_hist_tm1       ",
    leg1_hist_tm1.shape
)

print(
    "leg2_hist_tm1       ",
    leg2_hist_tm1.shape
)

print(
    "leg3_hist_tm2       ",
    leg3_hist_tm2.shape
)

print(
    "leg4_hist_tm2       ",
    leg4_hist_tm2.shape
)


print(
    "a_body_t            ",
    a_body_t.shape
)

print(
    "a1_tm1              ",
    a1_tm1.shape
)

print(
    "a2_tm1              ",
    a2_tm1.shape
)

print(
    "a3_tm2              ",
    a3_tm2.shape
)

print(
    "a4_tm2              ",
    a4_tm2.shape
)


print(
    "leg_target_tp1      ",
    leg_target_tp1.shape
)


print(
    "RGB context indices ",
    rgb_context_idx.shape
)

print(
    "RGB target indices  ",
    rgb_target_idx.shape
)


assert (
    body_hist_t.shape
    ==
    (
        N,
        50,
        len(BODY_FEATURE_NAMES)
    )
)


assert (
    leg1_hist_tm1.shape
    ==
    (
        N,
        50,
        D_LEG
    )
)


assert (
    leg2_hist_tm1.shape
    ==
    (
        N,
        50,
        D_LEG
    )
)


assert (
    leg3_hist_tm2.shape
    ==
    (
        N,
        50,
        D_LEG
    )
)


assert (
    leg4_hist_tm2.shape
    ==
    (
        N,
        50,
        D_LEG
    )
)


assert (
    a_body_t.shape
    ==
    (
        N,
        4
    )
)


assert (
    a1_tm1.shape
    ==
    (
        N,
        4
    )
)


assert (
    a2_tm1.shape
    ==
    (
        N,
        4
    )
)


assert (
    a3_tm2.shape
    ==
    (
        N,
        4
    )
)


assert (
    a4_tm2.shape
    ==
    (
        N,
        4
    )
)


assert (
    leg_target_tp1.shape
    ==
    (
        N,
        4,
        4
    )
)


assert (
    rgb_context_idx.shape
    ==
    (
        N,
        4
    )
)


assert (
    rgb_target_idx.shape
    ==
    (
        N,
    )
)


# ======================================================================
# 28. ACTION DIAGNOSTICS
# ======================================================================

print(
    "\n"
    +
    "=" * 90
)

print(
    "ACTION DIAGNOSTICS AFTER MODEL-TIME SAMPLING"
)

print(
    "=" * 90
)


for name, a in [

    ("leg1", a1_tm1),
    ("leg2", a2_tm1),
    ("leg3", a3_tm2),
    ("leg4", a4_tm2),

]:

    print(
        f"\n{name}"
    )


    print(
        "  Δq HFE std :",
        float(
            a[:, 0].std()
        )
    )

    print(
        "  Δq KFE std :",
        float(
            a[:, 1].std()
        )
    )

    print(
        "  tau HFE std:",
        float(
            a[:, 2].std()
        )
    )

    print(
        "  tau KFE std:",
        float(
            a[:, 3].std()
        )
    )

    print(
        "  tau HFE nonzero:",
        float(
            np.mean(
                np.abs(
                    a[:, 2]
                )
                >
                1e-6
            )
        )
    )

    print(
        "  tau KFE nonzero:",
        float(
            np.mean(
                np.abs(
                    a[:, 3]
                )
                >
                1e-6
            )
        )
    )


# ======================================================================
# 29. SAVE MODEL-READY NUMERICAL STAGE
#
# NO NORMALIZATION HERE.
#
# Normalization MUST be computed using TRAIN split only
# after multiple missions are collected.
# ======================================================================

OUT_FILE = (

    OUTPUT_ROOT

    /

    f"{MISSION}_"
    f"five_stream_numeric_stageA.npz"

)


np.savez(

    OUT_FILE,

    model_time=
        t_model.astype(
            np.float64
        ),


    body_hist_t=
        body_hist_t,


    leg1_hist_tm1=
        leg1_hist_tm1,

    leg2_hist_tm1=
        leg2_hist_tm1,

    leg3_hist_tm2=
        leg3_hist_tm2,

    leg4_hist_tm2=
        leg4_hist_tm2,


    a_body_t=
        a_body_t,

    a1_tm1=
        a1_tm1,

    a2_tm1=
        a2_tm1,

    a3_tm2=
        a3_tm2,

    a4_tm2=
        a4_tm2,


    leg_target_tp1=
        leg_target_tp1,


    rgb_context_idx=
        rgb_context_idx.astype(
            np.int32
        ),

    rgb_target_idx=
        rgb_target_idx.astype(
            np.int32
        ),


    camera_timestamps=
        t_cam,


    body_feature_names=
        np.asarray(
            BODY_FEATURE_NAMES
        ),

    leg_feature_names=
        np.asarray(
            LEG_FEATURE_NAMES[0]
        ),

)


# ======================================================================
# 30. SAVE HUMAN-READABLE METADATA
# ======================================================================

META_FILE = (

    OUTPUT_ROOT

    /

    f"{MISSION}_"
    f"five_stream_numeric_stageA.json"

)


metadata = {

    "mission":
        MISSION,

    "model_dt":
        MODEL_DT,

    "proprio_hz":
        PROP_HZ,

    "history_samples":
        PROP_HISTORY,

    "body_dim":
        int(
            body_hist_t.shape[-1]
        ),

    "leg_dim":
        int(
            D_LEG
        ),

    "num_samples":
        int(
            N
        ),

    "leg_mapping": {

        "leg1":
            "LF",

        "leg2":
            "RF",

        "leg3":
            "LH",

        "leg4":
            "RH",

    },

    "leg_action_definition": [

        "delta_q_HFE_command",

        "delta_q_KFE_command",

        "tau_HFE_command",

        "tau_KFE_command",

    ],

    "leg_target_definition": [

        "q_HFE_measured",

        "q_KFE_measured",

        "tau_HFE_measured",

        "tau_KFE_measured",

    ],

    "transformer_visual_times": [

        "t-3",
        "t-2",
        "t-1",
        "t"

    ],

    "conditioning_times": {

        "body":
            "t",

        "leg1":
            "t-1",

        "leg2":
            "t-1",

        "leg3":
            "t-2",

        "leg4":
            "t-2",

    },

    "body_target":
        "NOT YET CREATED",

    "visual_tokens":
        "NOT YET CREATED",

}


with open(
    META_FILE,
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=2
    )


print(
    "\n"
    +
    "=" * 90
)

print(
    "STAGE A COMPLETE"
)

print(
    "=" * 90
)


print(
    "Saved numerical dataset:"
)

print(
    OUT_FILE
)


print(
    "\nSaved metadata:"
)

print(
    META_FILE
)


print(
    "\nNEXT REQUIRED STAGE:"
)

print(
    "1. Download ZED2i left RGB images."
)

print(
    "2. Convert indexed RGB frames -> "
    "frozen DINOv2-B/14 "
    "[256,768] tokens."
)

print(
    "3. Build the four whole-body "
    "physical targets."
)

print(
    "4. Only then connect this "
    "dataset to FiveStreamFeelWM."
)

# %% notebook cell 5
# ======================================================================
# STAGE B — ZED2i RGB -> frozen DINOv2-B/14 patch tokens
#
# Input:
#   Stage A:
#       rgb_context_idx [N,4]
#       rgb_target_idx  [N]
#
# Output:
#   unique_rgb_indices.npy
#   dino_tokens_fp16.npy
#
# Each RGB frame:
#
#       image
#         |
#         v
#   frozen DINOv2-B/14
#         |
#         v
#      [256,768]
#
# No image decoder is trained.
# ======================================================================

# IPYTHON: !pip -q install -U huggingface_hub tqdm

import io
import os
import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from huggingface_hub import hf_hub_download


# ======================================================================
# 0. CONFIG
# ======================================================================

REPO_ID = "leggedrobotics/grand_tour_dataset"
MISSION = "2024-11-02-17-10-25"

ROOT = Path("/content/grandtour_fivestream_pipeline")
PROC = ROOT / "processed"
DOWNLOAD = ROOT / "downloads"

DOWNLOAD.mkdir(parents=True, exist_ok=True)

STAGE_A = (
    PROC /
    f"{MISSION}_five_stream_numeric_stageA.npz"
)

assert STAGE_A.exists(), (
    f"Stage A missing:\n{STAGE_A}"
)


# ======================================================================
# 1. LOAD ONLY INDEX INFORMATION FROM STAGE A
# ======================================================================

stageA = np.load(
    STAGE_A,
    allow_pickle=False
)

rgb_context_idx = stageA["rgb_context_idx"]
rgb_target_idx = stageA["rgb_target_idx"]

N = len(rgb_target_idx)

required_indices = np.unique(
    np.concatenate([
        rgb_context_idx.reshape(-1),
        rgb_target_idx.reshape(-1),
    ])
).astype(np.int64)

required_set = set(
    required_indices.tolist()
)

print("Training transitions :", N)
print("Total RGB references :", N * 5)
print("Unique RGB frames    :", len(required_indices))


# ======================================================================
# 2. DOWNLOAD ACTUAL IMAGE ARCHIVE
#
# NOTE:
# data/zed2i_left_images.tar contained metadata.
#
# Actual JPEG frames live under:
#
#     <MISSION>/images/zed2i_left_images.tar
# ======================================================================

HF_IMAGE_FILE = (
    f"{MISSION}/images/"
    f"zed2i_left_images.tar"
)

print("\nDownloading actual RGB archive...")
print(HF_IMAGE_FILE)

IMAGE_TAR = Path(
    hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=HF_IMAGE_FILE,
        local_dir=DOWNLOAD
    )
)

print(
    f"Archive size: "
    f"{IMAGE_TAR.stat().st_size / 1024**3:.2f} GB"
)


# ======================================================================
# 3. LOAD FROZEN DINOv2-B/14
# ======================================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("\nDevice:", DEVICE)

# facebookresearch official DINOv2 torch hub model.
dino = torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vitb14"
)

dino = dino.to(DEVICE)
dino.eval()

for p in dino.parameters():
    p.requires_grad_(False)

print("DINOv2-B/14 frozen.")


# ======================================================================
# 4. IMAGE PREPROCESSING
#
# 224x224 / patch 14 -> 16x16 = 256 tokens.
# ======================================================================

# Manual equivalent of:
#   Resize(shorter side=256, bicubic)
#   CenterCrop(224)
#   ToTensor()
#   ImageNet normalization
#
# This deliberately avoids importing torchvision because the previous
# runtime showed a torchvision/Pillow `_Ink` version conflict.

_IMAGENET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406],
    dtype=torch.float32
)[:, None, None]

_IMAGENET_STD = torch.tensor(
    [0.229, 0.224, 0.225],
    dtype=torch.float32
)[:, None, None]


def preprocess(img):

    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size

    if w <= h:
        new_w = 256
        new_h = int(round(h * 256.0 / w))
    else:
        new_h = 256
        new_w = int(round(w * 256.0 / h))

    img = img.resize(
        (new_w, new_h),
        resample=Image.Resampling.BICUBIC
    )

    left = (new_w - 224) // 2
    top = (new_h - 224) // 2

    img = img.crop(
        (left, top, left + 224, top + 224)
    )

    arr = np.asarray(
        img,
        dtype=np.float32
    ) / 255.0

    x = torch.from_numpy(
        arr
    ).permute(
        2, 0, 1
    ).contiguous()

    return (
        x - _IMAGENET_MEAN
    ) / _IMAGENET_STD


# ======================================================================
# 5. OUTPUT MEMORY-MAPPED TOKEN STORE
#
# Avoid creating context copies like:
#
# [3732,4,256,768]
#
# Instead store each unique RGB frame ONCE.
# ======================================================================

INDEX_FILE = (
    PROC /
    f"{MISSION}_unique_rgb_indices.npy"
)

TOKEN_FILE = (
    PROC /
    f"{MISSION}_dinov2_vitb14_tokens_fp16.npy"
)

np.save(
    INDEX_FILE,
    required_indices
)

tokens = np.lib.format.open_memmap(
    TOKEN_FILE,
    mode="w+",
    dtype=np.float16,
    shape=(
        len(required_indices),
        256,
        768
    )
)

index_to_row = {
    int(frame_idx): row
    for row, frame_idx
    in enumerate(required_indices)
}


# ======================================================================
# 6. ROBUST FRAME INDEX PARSER
#
# GrandTour examples use:
#
# images/<tag>/000123.jpeg
#
# Archive paths may contain additional parent folders.
# ======================================================================

def frame_index_from_member(name):

    stem = Path(name).stem

    try:
        return int(stem)
    except Exception:
        return None


# ======================================================================
# 7. STREAM ONLY REQUIRED JPEG FRAMES FROM TAR
# ======================================================================

BATCH = 16

batch_images = []
batch_rows = []
batch_frame_ids = []

found = set()


@torch.inference_mode()
def encode_batch():

    global batch_images
    global batch_rows
    global batch_frame_ids

    if not batch_images:
        return

    x = torch.stack(
        batch_images,
        dim=0
    ).to(
        DEVICE,
        non_blocking=True
    )

    with torch.autocast(
        device_type=DEVICE.type,
        dtype=torch.float16,
        enabled=(DEVICE.type == "cuda")
    ):

        features = dino.forward_features(x)

        patch_tokens = (
            features[
                "x_norm_patchtokens"
            ]
        )

    assert patch_tokens.shape[1:] == (
        256,
        768
    ), patch_tokens.shape

    patch_tokens = (
        patch_tokens
        .float()
        .cpu()
        .numpy()
        .astype(np.float16)
    )

    for j, row in enumerate(batch_rows):
        tokens[row] = patch_tokens[j]

    found.update(batch_frame_ids)

    batch_images = []
    batch_rows = []
    batch_frame_ids = []


print(
    "\nScanning image tar and "
    "encoding only required frames..."
)

with tarfile.open(
    IMAGE_TAR,
    "r:*"
) as tf:

    members = tf.getmembers()

    pbar = tqdm(
        total=len(required_indices),
        desc="DINO frames"
    )

    previous_found = 0

    for member in members:

        if not member.isfile():
            continue

        frame_idx = (
            frame_index_from_member(
                member.name
            )
        )

        if frame_idx is None:
            continue

        if frame_idx not in required_set:
            continue

        # avoid duplicate path/member
        if frame_idx in found:
            continue

        f = tf.extractfile(member)

        if f is None:
            continue

        try:

            img = Image.open(
                io.BytesIO(
                    f.read()
                )
            ).convert("RGB")

        except Exception as e:

            print(
                "Could not decode",
                member.name,
                e
            )

            continue

        batch_images.append(
            preprocess(img)
        )

        batch_rows.append(
            index_to_row[
                frame_idx
            ]
        )

        batch_frame_ids.append(
            frame_idx
        )

        if len(batch_images) >= BATCH:

            old = len(found)

            encode_batch()

            pbar.update(
                len(found) - old
            )

        if len(found) == len(required_indices):
            break


    if batch_images:

        old = len(found)

        encode_batch()

        pbar.update(
            len(found) - old
        )

    pbar.close()


tokens.flush()


# ======================================================================
# 8. VERIFY COMPLETE COVERAGE
# ======================================================================

missing = sorted(
    required_set - found
)

if missing:

    print(
        "\nFirst missing indices:",
        missing[:50]
    )

    raise RuntimeError(
        f"{len(missing)} required RGB frames "
        "were not found in the archive."
    )


# ======================================================================
# 9. TOKEN VALIDATION
# ======================================================================

del tokens

tokens_check = np.load(
    TOKEN_FILE,
    mmap_mode="r"
)

assert tokens_check.shape == (
    len(required_indices),
    256,
    768
)

assert np.isfinite(
    tokens_check[
        :min(
            50,
            len(tokens_check)
        )
    ]
).all()


print("\n" + "=" * 90)

print("STAGE B COMPLETE")

print("=" * 90)

print(
    "Unique frames:",
    len(required_indices)
)

print(
    "Token store :",
    TOKEN_FILE
)

print(
    "Token shape :",
    tokens_check.shape
)

print(
    "dtype       :",
    tokens_check.dtype
)

print(
    "disk size   :",
    f"{TOKEN_FILE.stat().st_size/1024**3:.2f} GB"
)

# %% notebook cell 6
# ======================================================================
# STAGE C — WHOLE-BODY PHYSICAL TARGETS
#
# Output at every model target time t+1:
#
# body_target =
#
# [
#    roughness,
#    angular_instability,
#    foot_slip,
#    slope_proxy
# ]
#
# Window:
#    0.5 s CENTERED on target time
#
# IMPORTANT:
#   foot_slip is the required legged adaptation.
# ======================================================================

# IPYTHON: !pip -q install -U "zarr<3" huggingface_hub scipy tqdm

import tarfile
import shutil
from pathlib import Path

import numpy as np
import zarr

from scipy.spatial.transform import Rotation
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm


# ======================================================================
# 0. PATHS / LOAD STAGE A
# ======================================================================

REPO_ID = "leggedrobotics/grand_tour_dataset"
MISSION = "2024-11-02-17-10-25"

ROOT = Path(
    "/content/grandtour_fivestream_pipeline"
)

PROC = ROOT / "processed"
DOWNLOAD = ROOT / "downloads"
EXTRACT = ROOT / "extracted"

STAGE_A = (
    PROC /
    f"{MISSION}_five_stream_numeric_stageA.npz"
)

A = np.load(
    STAGE_A,
    allow_pickle=False
)

model_time = A["model_time"]

target_time = (
    model_time + 0.10
)

N = len(target_time)

print("Target samples:", N)


# ======================================================================
# 1. HELPERS
# ======================================================================

def safe_extract(
    tar_path,
    destination
):

    destination.mkdir(
        parents=True,
        exist_ok=True
    )

    base = destination.resolve()

    with tarfile.open(
        tar_path,
        "r:*"
    ) as tf:

        for member in tf.getmembers():

            p = (
                destination /
                member.name
            ).resolve()

            if not str(p).startswith(
                str(base)
            ):
                raise RuntimeError(
                    "Unsafe tar member"
                )

        tf.extractall(
            destination
        )


def find_zarr(
    folder,
    name
):

    candidates = []

    for p in folder.rglob(
        ".zgroup"
    ):

        if name in str(
            p.parent
        ):

            candidates.append(
                p.parent
            )

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda p: len(p.parts)
    )


def get_stream(
    name
):

    folder = (
        EXTRACT /
        name
    )

    existing = (
        find_zarr(
            folder,
            name
        )
        if folder.exists()
        else None
    )

    if existing is None:

        filename = (
            f"{MISSION}/data/"
            f"{name}.tar"
        )

        print(
            "\nDownloading:",
            filename
        )

        tarpath = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=filename,
                local_dir=DOWNLOAD
            )
        )

        if folder.exists():
            shutil.rmtree(folder)

        safe_extract(
            tarpath,
            folder
        )

        existing = find_zarr(
            folder,
            name
        )

    if existing is None:

        raise RuntimeError(
            f"Could not open {name}"
        )

    return zarr.open_group(
        str(existing),
        mode="r"
    )


def flatten_arrays(
    group,
    prefix=""
):

    out = {}

    for k in group.array_keys():

        name = (
            f"{prefix}/{k}"
            if prefix
            else k
        )

        out[name] = group[k]

    for k in group.group_keys():

        child = (
            f"{prefix}/{k}"
            if prefix
            else k
        )

        out.update(
            flatten_arrays(
                group[k],
                child
            )
        )

    return out


# ======================================================================
# 2. LOAD BODY IMU + STATE ESTIMATOR
# ======================================================================

imu = get_stream(
    "anymal_imu"
)

state = get_stream(
    "anymal_state_state_estimator"
)


# ======================================================================
# 3. PRINT STATE-ESTIMATOR SCHEMA ONCE
#
# This is important because foot-state names differ across conversions.
# ======================================================================

state_arrays = flatten_arrays(
    state
)

print("\n" + "=" * 90)
print("STATE ESTIMATOR CANDIDATES")
print("=" * 90)

for name, arr in sorted(
    state_arrays.items()
):

    low = name.lower()

    if any(
        x in low
        for x in [
            "foot",
            "contact",
            "position",
            "velocity",
            "twist",
            "pose",
        ]
    ):

        print(
            f"{name:55s}",
            arr.shape
        )


# ======================================================================
# 4. IMU ARRAYS
# ======================================================================

imu_arrays = flatten_arrays(
    imu
)


def unique_match(
    arrays,
    alternatives,
    width=None
):

    matches = []

    for name, arr in arrays.items():

        low = name.lower()

        if "cov" in low:
            continue

        ok = any(
            all(
                token in low
                for token in alt
            )
            for alt in alternatives
        )

        if not ok:
            continue

        if width is not None:

            if (
                arr.ndim < 2
                or
                arr.shape[-1]
                != width
            ):

                continue

        matches.append(
            (name, arr)
        )

    if len(matches) != 1:

        print(
            "\nCandidates for",
            alternatives
        )

        for n, a in matches:
            print(n, a.shape)

        raise RuntimeError(
            "Could not uniquely map field."
        )

    print(
        "Mapped:",
        matches[0][0]
    )

    return matches[0][1]


imu_acc_arr = unique_match(
    imu_arrays,
    [
        ("linear", "acc"),
        ("lin", "acc"),
    ],
    width=3
)

imu_gyro_arr = unique_match(
    imu_arrays,
    [
        ("angular", "vel"),
        ("ang", "vel"),
    ],
    width=3
)

t_imu = np.asarray(
    imu["timestamp"][:],
    dtype=np.float64
)

acc = np.asarray(
    imu_acc_arr[:],
    dtype=np.float64
)

gyro = np.asarray(
    imu_gyro_arr[:],
    dtype=np.float64
)


# ======================================================================
# 5. GET BODY PITCH / GROUND VELOCITY FROM STAGE-A ODOMETRY STREAM
# ======================================================================

odom = get_stream(
    "anymal_state_odometry"
)

t_odom = np.asarray(
    odom["timestamp"][:],
    dtype=np.float64
)

pos = np.asarray(
    odom["pose_pos"][:],
    dtype=np.float64
)

quat = np.asarray(
    odom["pose_orien"][:],
    dtype=np.float64
)

quat /= np.maximum(
    np.linalg.norm(
        quat,
        axis=1,
        keepdims=True
    ),
    1e-12
)

rpy = (
    Rotation
    .from_quat(quat)
    .as_euler(
        "xyz",
        degrees=False
    )
)

dt_odom = np.gradient(
    t_odom
)

vel_world = np.gradient(
    pos,
    axis=0
) / dt_odom[:, None]

ground_speed = np.linalg.norm(
    vel_world[:, :2],
    axis=1
)

pitch = rpy[:, 1]


# ======================================================================
# 6. INTERPOLATION
# ======================================================================

def interp_scalar(
    t,
    x,
    q
):

    return np.interp(
        q,
        t,
        x
    )


def interp_vec(
    t,
    x,
    q
):

    return np.stack(
        [
            np.interp(
                q,
                t,
                x[:, j]
            )
            for j
            in range(
                x.shape[1]
            )
        ],
        axis=1
    )


# ======================================================================
# 7. CENTERED 0.5-s TARGET WINDOWS
#
# Feel-WM:
#
# target statistics are evaluated in a short window
# centered on target time.
# ======================================================================

WINDOW = 0.50

SAMPLE_HZ = 100.0

HALF = WINDOW / 2

relative = np.linspace(
    -HALF,
    HALF,
    int(
        WINDOW *
        SAMPLE_HZ
    ),
    endpoint=False
)

assert len(relative) == 50


body_target = np.zeros(
    (N, 4),
    dtype=np.float32
)


# ======================================================================
# 8. FOOT SLIP SUPPORT
#
# We do NOT guess field names.
#
# Try to identify explicit foot position and contact channels.
# ======================================================================

foot_position_candidates = []

contact_candidates = []


for name, arr in state_arrays.items():

    low = name.lower()

    if (
        "foot" in low
        and
        "position" in low
    ):
        foot_position_candidates.append(
            (name, arr)
        )

    if (
        "contact" in low
        and
        "foot" in low
    ):
        contact_candidates.append(
            (name, arr)
        )


print("\nFoot-position candidates:")

for n, a in foot_position_candidates:
    print(" ", n, a.shape)

print("\nFoot-contact candidates:")

for n, a in contact_candidates:
    print(" ", n, a.shape)


# ======================================================================
# 9. RESOLVE FOOT ARRAYS ONLY IF UNAMBIGUOUS
# ======================================================================

FOOT_SLIP_AVAILABLE = False


if (
    len(foot_position_candidates) == 1
    and
    len(contact_candidates) == 1
):

    foot_pos_name, foot_pos_arr = (
        foot_position_candidates[0]
    )

    contact_name, contact_arr = (
        contact_candidates[0]
    )

    t_state = np.asarray(
        state["timestamp"][:],
        dtype=np.float64
    )

    foot_pos = np.asarray(
        foot_pos_arr[:],
        dtype=np.float64
    )

    contact = np.asarray(
        contact_arr[:]
    )

    print(
        "\nUsing foot positions:",
        foot_pos_name
    )

    print(
        "Using contacts:",
        contact_name
    )

    FOOT_SLIP_AVAILABLE = True

else:

    print(
        "\nFoot-slip field mapping is not "
        "unambiguous."
    )

    print(
        "The cell will compute the other "
        "3 targets but will NOT fabricate "
        "foot slip."
    )


# ======================================================================
# 10. BODY TARGETS
# ======================================================================

for i, tc in enumerate(
    tqdm(
        target_time,
        desc="Body targets"
    )
):

    tw = tc + relative


    # ----------------------------------------------------------
    # 1. ROUGHNESS
    #
    # vertical acceleration variability
    # ----------------------------------------------------------

    a_win = interp_vec(
        t_imu,
        acc,
        tw
    )

    roughness = np.std(
        a_win[:, 2]
    )


    # ----------------------------------------------------------
    # 2. ANGULAR INSTABILITY
    #
    # variability in roll/pitch rates
    # ----------------------------------------------------------

    w_win = interp_vec(
        t_imu,
        gyro,
        tw
    )

    angular_instability = (
        np.var(
            w_win[:, 0]
        )
        +
        np.var(
            w_win[:, 1]
        )
    )


    # ----------------------------------------------------------
    # 3. SLOPE PROXY
    #
    # Feel-WM:
    #
    # mean(v_ground * sin(theta))
    # ----------------------------------------------------------

    vg_win = interp_scalar(
        t_odom,
        ground_speed,
        tw
    )

    pitch_win = interp_scalar(
        t_odom,
        pitch,
        tw
    )

    slope_proxy = np.mean(
        vg_win *
        np.sin(
            pitch_win
        )
    )


    # ----------------------------------------------------------
    # 4. FOOT SLIP
    #
    # legged adaptation:
    #
    # mean horizontal speed of feet
    # that are labelled in contact.
    #
    # This is NOT claimed to be the
    # wheeled Feel-WM slip equation.
    # ----------------------------------------------------------

    if FOOT_SLIP_AVAILABLE:

        # nearest state estimator samples
        ids = np.searchsorted(
            t_state,
            tw
        )

        ids = np.clip(
            ids,
            1,
            len(t_state)-1
        )

        fp = foot_pos[ids]

        ct = contact[ids]

        # expected canonical shapes:
        #
        # fp: [50,4,3]
        # ct: [50,4] or [50,4,1]

        if (
            fp.ndim == 3
            and
            fp.shape[1:] == (4,3)
        ):

            dt = 1.0 / SAMPLE_HZ

            fv = np.gradient(
                fp,
                dt,
                axis=0
            )

            horizontal = np.linalg.norm(
                fv[..., :2],
                axis=-1
            )

            ct = np.squeeze(
                ct
            )

            if (
                ct.ndim == 2
                and
                ct.shape[1] == 4
            ):

                mask = (
                    ct > 0
                )

                if mask.any():

                    foot_slip = float(
                        horizontal[
                            mask
                        ].mean()
                    )

                else:

                    foot_slip = 0.0

            else:

                raise RuntimeError(
                    "Unexpected contact shape: "
                    f"{ct.shape}"
                )

        else:

            raise RuntimeError(
                "Unexpected foot-position shape: "
                f"{fp.shape}"
            )

    else:

        foot_slip = np.nan


    body_target[i] = [

        roughness,

        angular_instability,

        foot_slip,

        slope_proxy,

    ]


# ======================================================================
# 11. SAVE
# ======================================================================

BODY_TARGET_FILE = (
    PROC /
    f"{MISSION}_body_target_stageC.npy"
)

np.save(
    BODY_TARGET_FILE,
    body_target
)


print("\n" + "=" * 90)
print("STAGE C COMPLETE")
print("=" * 90)

print(
    "body_target:",
    body_target.shape
)

print(
    "finite per channel:",
    np.isfinite(
        body_target
    ).sum(axis=0)
)

print(
    "mean:",
    np.nanmean(
        body_target,
        axis=0
    )
)

print(
    "std :",
    np.nanstd(
        body_target,
        axis=0
    )
)

print(
    "\nSaved:",
    BODY_TARGET_FILE
)

if not FOOT_SLIP_AVAILABLE:

    print(
        "\nSTOP before training:"
    )

    print(
        "The first, second and fourth "
        "body targets are built, but "
        "foot-slip still needs exact "
        "state-estimator field mapping."
    )

# %% notebook cell 7
# ======================================================================
# STAGE D — FINAL GRANDTOUR DATASET -> FiveStreamFeelWM
#
# After this cell:
#
# batch = next(iter(train_loader))
#
# pred = model(...)
#
# works directly.
# ======================================================================

import json
from pathlib import Path

import numpy as np
import torch

from torch.utils.data import (
    Dataset,
    DataLoader,
    Subset
)


# ======================================================================
# 0. CHECK ARCHITECTURE
# ======================================================================

assert "FiveStreamFeelWM" in globals(), (
    "Run your exact architecture cell first."
)


# ======================================================================
# 1. PATHS
# ======================================================================

MISSION = "2024-11-02-17-10-25"

ROOT = Path(
    "/content/grandtour_fivestream_pipeline"
)

PROC = ROOT / "processed"


STAGE_A = (
    PROC /
    f"{MISSION}_five_stream_numeric_stageA.npz"
)

RGB_INDEX_FILE = (
    PROC /
    f"{MISSION}_unique_rgb_indices.npy"
)

DINO_FILE = (
    PROC /
    f"{MISSION}_dinov2_vitb14_tokens_fp16.npy"
)

BODY_TARGET_FILE = (
    PROC /
    f"{MISSION}_body_target_stageC.npy"
)


for p in [
    STAGE_A,
    RGB_INDEX_FILE,
    DINO_FILE,
    BODY_TARGET_FILE
]:

    assert p.exists(), (
        f"Missing required stage:\n{p}"
    )


# ======================================================================
# 2. LOAD METADATA
# ======================================================================

A = np.load(
    STAGE_A,
    mmap_mode="r",
    allow_pickle=False
)

unique_rgb_indices = np.load(
    RGB_INDEX_FILE
)

dino_tokens = np.load(
    DINO_FILE,
    mmap_mode="r"
)

body_target = np.load(
    BODY_TARGET_FILE,
    mmap_mode="r"
)


# ======================================================================
# 3. REQUIRE ALL FOUR BODY TARGETS
# ======================================================================

finite_body = np.isfinite(
    body_target
).all(axis=1)

print(
    "Fully supervised samples:",
    finite_body.sum(),
    "/",
    len(finite_body)
)

if not finite_body.all():

    raise RuntimeError(
        "Body target still contains NaN. "
        "Do not train until foot-slip "
        "mapping is resolved."
    )


# ======================================================================
# 4. CAMERA INDEX -> DINO ROW LOOKUP
# ======================================================================

rgb_to_token_row = {

    int(frame):
        i

    for i, frame
    in enumerate(
        unique_rgb_indices
    )

}


# ======================================================================
# 5. RAW DATASET
# ======================================================================

class GrandTourFiveStreamDataset(
    Dataset
):

    def __init__(
        self,
        A,
        dino_tokens,
        body_target,
        rgb_lookup
    ):

        self.A = A

        self.dino = dino_tokens

        self.body_target = (
            body_target
        )

        self.rgb_lookup = (
            rgb_lookup
        )


    def __len__(self):

        return len(
            self.A[
                "model_time"
            ]
        )


    def _token(
        self,
        frame_idx
    ):

        row = self.rgb_lookup[
            int(frame_idx)
        ]

        # float16 disk storage ->
        # float32 CPU tensor.
        return torch.from_numpy(
            np.asarray(
                self.dino[row],
                dtype=np.float32
            ).copy()
        )


    def __getitem__(
        self,
        i
    ):

        rgb_idx = (
            self.A[
                "rgb_context_idx"
            ][i]
        )


        # --------------------------------------------------------------
        # EXACT:
        #
        # [z_t-3,z_t-2,z_t-1,z_t]
        # --------------------------------------------------------------

        context = torch.stack(
            [
                self._token(
                    rgb_idx[0]
                ),

                self._token(
                    rgb_idx[1]
                ),

                self._token(
                    rgb_idx[2]
                ),

                self._token(
                    rgb_idx[3]
                ),
            ],
            dim=0
        )


        # --------------------------------------------------------------
        # VISUAL t+1 TARGET
        # --------------------------------------------------------------

        z_target = self._token(

            self.A[
                "rgb_target_idx"
            ][i]

        )


        def tensor(key):

            return torch.from_numpy(
                np.asarray(
                    self.A[key][i],
                    dtype=np.float32
                ).copy()
            )


        return {

            "context":
                context,


            # BODY
            "body_hist_t":
                tensor(
                    "body_hist_t"
                ),

            "a_body_t":
                tensor(
                    "a_body_t"
                ),


            # LEGS 1,2 at t-1
            "leg1_hist_tm1":
                tensor(
                    "leg1_hist_tm1"
                ),

            "leg2_hist_tm1":
                tensor(
                    "leg2_hist_tm1"
                ),

            "a1_tm1":
                tensor(
                    "a1_tm1"
                ),

            "a2_tm1":
                tensor(
                    "a2_tm1"
                ),


            # LEGS 3,4 at t-2
            "leg3_hist_tm2":
                tensor(
                    "leg3_hist_tm2"
                ),

            "leg4_hist_tm2":
                tensor(
                    "leg4_hist_tm2"
                ),

            "a3_tm2":
                tensor(
                    "a3_tm2"
                ),

            "a4_tm2":
                tensor(
                    "a4_tm2"
                ),


            # TARGETS t+1
            "z_target":
                z_target,

            "body_target":
                torch.from_numpy(
                    np.asarray(
                        self.body_target[i],
                        dtype=np.float32
                    ).copy()
                ),

            "leg_target":
                tensor(
                    "leg_target_tp1"
                ),


            "time":
                float(
                    self.A[
                        "model_time"
                    ][i]
                ),
        }


dataset = (
    GrandTourFiveStreamDataset(

        A,

        dino_tokens,

        body_target,

        rgb_to_token_row

    )
)


# ======================================================================
# 6. TEMPORAL SPLIT
#
# DO NOT randomly shuffle samples into train/validation before splitting.
#
# Neighboring samples overlap heavily in time.
# ======================================================================

N = len(dataset)

train_end = int(
    0.70 * N
)

val_start = int(
    0.80 * N
)

val_end = int(
    0.90 * N
)


train_indices = np.arange(
    0,
    train_end
)

val_indices = np.arange(
    val_start,
    val_end
)


train_dataset = Subset(
    dataset,
    train_indices
)

val_dataset = Subset(
    dataset,
    val_indices
)


# ======================================================================
# 7. TRAIN-ONLY NORMALIZATION
#
# We calculate statistics only on TRAIN.
#
# Targets are normalized here.
# ======================================================================

train_body = np.asarray(
    body_target[
        train_indices
    ],
    dtype=np.float64
)

BODY_MEAN = (
    train_body.mean(
        axis=0
    )
)

BODY_STD = (
    train_body.std(
        axis=0
    )
    +
    1e-6
)


train_leg = np.asarray(
    A["leg_target_tp1"][
        train_indices
    ],
    dtype=np.float64
)

LEG_MEAN = (
    train_leg.mean(
        axis=0
    )
)

LEG_STD = (
    train_leg.std(
        axis=0
    )
    +
    1e-6
)


print(
    "BODY_MEAN:",
    BODY_MEAN
)

print(
    "BODY_STD :",
    BODY_STD
)

print(
    "\nLEG_MEAN:"
)

print(
    LEG_MEAN
)

print(
    "\nLEG_STD:"
)

print(
    LEG_STD
)


# ======================================================================
# 8. NORMALIZED WRAPPER
# ======================================================================

class NormalizeTargets(
    Dataset
):

    def __init__(
        self,
        base,
        body_mean,
        body_std,
        leg_mean,
        leg_std
    ):

        self.base = base

        self.body_mean = (
            torch.tensor(
                body_mean,
                dtype=torch.float32
            )
        )

        self.body_std = (
            torch.tensor(
                body_std,
                dtype=torch.float32
            )
        )

        self.leg_mean = (
            torch.tensor(
                leg_mean,
                dtype=torch.float32
            )
        )

        self.leg_std = (
            torch.tensor(
                leg_std,
                dtype=torch.float32
            )
        )


    def __len__(self):

        return len(
            self.base
        )


    def __getitem__(
        self,
        i
    ):

        x = self.base[i]

        x[
            "body_target"
        ] = (

            x["body_target"]
            -
            self.body_mean

        ) / self.body_std


        x[
            "leg_target"
        ] = (

            x["leg_target"]
            -
            self.leg_mean

        ) / self.leg_std


        return x


train_dataset = NormalizeTargets(

    train_dataset,

    BODY_MEAN,
    BODY_STD,

    LEG_MEAN,
    LEG_STD

)


val_dataset = NormalizeTargets(

    val_dataset,

    BODY_MEAN,
    BODY_STD,

    LEG_MEAN,
    LEG_STD

)


# ======================================================================
# 9. LOADERS
# ======================================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=1,

    shuffle=True,

    num_workers=2,

    pin_memory=True

)


val_loader = DataLoader(

    val_dataset,

    batch_size=1,

    shuffle=False,

    num_workers=2,

    pin_memory=True

)


# ======================================================================
# 10. INSTANTIATE EXACT MODEL
# ======================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


BODY_DIM = (
    A[
        "body_hist_t"
    ].shape[-1]
)

LEG_DIM = (
    A[
        "leg1_hist_tm1"
    ].shape[-1]
)


model = FiveStreamFeelWM(

    body_hist_dim=
        BODY_DIM,

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

).to(
    DEVICE
)


# ======================================================================
# 11. FULL FORWARD-PASS AUDIT
# ======================================================================

batch = next(
    iter(
        train_loader
    )
)


def gpu(x):

    return x.to(
        DEVICE,
        non_blocking=True
    )


model.eval()


with torch.inference_mode():

    pred = model(

        context=
            gpu(
                batch[
                    "context"
                ]
            ),


        body_hist_t=
            gpu(
                batch[
                    "body_hist_t"
                ]
            ),


        leg1_hist_tm1=
            gpu(
                batch[
                    "leg1_hist_tm1"
                ]
            ),

        leg2_hist_tm1=
            gpu(
                batch[
                    "leg2_hist_tm1"
                ]
            ),


        leg3_hist_tm2=
            gpu(
                batch[
                    "leg3_hist_tm2"
                ]
            ),

        leg4_hist_tm2=
            gpu(
                batch[
                    "leg4_hist_tm2"
                ]
            ),


        a_body_t=
            gpu(
                batch[
                    "a_body_t"
                ]
            ),


        a1_tm1=
            gpu(
                batch[
                    "a1_tm1"
                ]
            ),

        a2_tm1=
            gpu(
                batch[
                    "a2_tm1"
                ]
            ),


        a3_tm2=
            gpu(
                batch[
                    "a3_tm2"
                ]
            ),

        a4_tm2=
            gpu(
                batch[
                    "a4_tm2"
                ]
            )

    )


print("\n" + "=" * 90)

print("END-TO-END ARCHITECTURE AUDIT")

print("=" * 90)

print(
    "input context:",
    batch[
        "context"
    ].shape
)

print(
    "pred z_next:",
    pred[
        "z_next"
    ].shape
)

print(
    "target z_next:",
    batch[
        "z_target"
    ].shape
)

print(
    "pred body:",
    pred[
        "p_body_next"
    ].shape
)

print(
    "target body:",
    batch[
        "body_target"
    ].shape
)

print(
    "pred leg1:",
    pred[
        "p_leg1_next"
    ].shape
)

print(
    "pred leg2:",
    pred[
        "p_leg2_next"
    ].shape
)

print(
    "pred leg3:",
    pred[
        "p_leg3_next"
    ].shape
)

print(
    "pred leg4:",
    pred[
        "p_leg4_next"
    ].shape
)

print(
    "leg target:",
    batch[
        "leg_target"
    ].shape
)

print(
    "full H:",
    pred[
        "H"
    ].shape
)


assert pred[
    "z_next"
].shape == (
    1,
    256,
    768
)

assert pred[
    "p_body_next"
].shape == (
    1,
    4
)

for leg in range(
    1,
    5
):

    assert pred[
        f"p_leg{leg}_next"
    ].shape == (
        1,
        4
    )


assert pred[
    "H"
].shape == (
    1,
    9,
    256,
    768
)


print(
    "\nSUCCESS:"
)

print(
    "GrandTour -> preprocessing -> "
    "DINO -> five FiLM branches -> "
    "9-stream transformer -> "
    "body/leg heads is connected."
)

# %% notebook cell 9
# ======================================================================
# FINAL GRANDTOUR TRAINING CELL
#
# RUN THIS ONLY AFTER STAGE D has successfully created:
#   model, train_loader, val_loader,
#   BODY_MEAN, BODY_STD, LEG_MEAN, LEG_STD,
#   BODY_DIM, LEG_DIM
#
# This supersedes the earlier placeholder ANYmal training cell.
#
# Trains:
#   visual future z_(t+1)
#   body physical head
#   four separate leg future heads
#
# Selects BEST checkpoint using VALIDATION loss, not training loss.
# ======================================================================

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ----------------------------------------------------------------------
# 0. REQUIRED OBJECTS
# ----------------------------------------------------------------------

required = [
    "model",
    "train_loader",
    "val_loader",
    "BODY_MEAN",
    "BODY_STD",
    "LEG_MEAN",
    "LEG_STD",
    "BODY_DIM",
    "LEG_DIM",
]

missing = [x for x in required if x not in globals()]

if missing:
    raise RuntimeError(
        "Run Stage D first. Missing objects: "
        + ", ".join(missing)
    )

device = next(model.parameters()).device

# ----------------------------------------------------------------------
# 1. SETTINGS
# ----------------------------------------------------------------------

EPOCHS = 20
LR = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_ACCUM_STEPS = 4
MAX_GRAD_NORM = 1.0

LAMBDA_VIS = 1.0
LAMBDA_BODY = 1.0
LAMBDA_LEGS = 1.0

CHECKPOINT_DIR = Path(
    "/content/drive/MyDrive/FiveStreamFeelWM_GrandTour_checkpoints"
)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# 2. TRAINING LOSSES
#
# NOTE:
# Test-time L_z is calculated later using the EXACT requested formula:
#
#   (1/256) sum_j || zhat_j - z_j ||_2^2
#
# Here we keep the existing training objective:
# elementwise MSE + 0.1 cosine distance.
# ----------------------------------------------------------------------

def visual_training_loss(prediction, target):

    mse = F.mse_loss(
        prediction,
        target
    )

    cosine = F.cosine_similarity(
        prediction,
        target,
        dim=-1
    ).mean()

    return mse + 0.1 * (1.0 - cosine)


def regression_loss(prediction, target):

    return F.smooth_l1_loss(
        prediction,
        target
    )


def move_batch(batch, device):

    return {
        k: (
            v.to(device, non_blocking=True)
            if torch.is_tensor(v)
            else v
        )
        for k, v in batch.items()
    }


def forward_model(batch):

    return model(
        context=batch["context"],
        body_hist_t=batch["body_hist_t"],
        leg1_hist_tm1=batch["leg1_hist_tm1"],
        leg2_hist_tm1=batch["leg2_hist_tm1"],
        leg3_hist_tm2=batch["leg3_hist_tm2"],
        leg4_hist_tm2=batch["leg4_hist_tm2"],
        a_body_t=batch["a_body_t"],
        a1_tm1=batch["a1_tm1"],
        a2_tm1=batch["a2_tm1"],
        a3_tm2=batch["a3_tm2"],
        a4_tm2=batch["a4_tm2"],
    )


def compute_losses(pred, batch):

    L_vis = visual_training_loss(
        pred["z_next"],
        batch["z_target"]
    )

    L_body = regression_loss(
        pred["p_body_next"],
        batch["body_target"]
    )

    leg_losses = []

    for leg in range(4):

        L = regression_loss(
            pred[f"p_leg{leg+1}_next"],
            batch["leg_target"][:, leg]
        )

        leg_losses.append(L)

    L_legs = torch.stack(
        leg_losses
    ).mean()

    total = (
        LAMBDA_VIS * L_vis
        + LAMBDA_BODY * L_body
        + LAMBDA_LEGS * L_legs
    )

    return total, L_vis, L_body, leg_losses


# ----------------------------------------------------------------------
# 3. AMP + OPTIMIZER
# ----------------------------------------------------------------------

use_amp = (
    device.type == "cuda"
)

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

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.95)
)

# ----------------------------------------------------------------------
# 4. TRAIN + VALIDATE
# ----------------------------------------------------------------------

history = []
best_val_loss = float("inf")

for epoch in range(1, EPOCHS + 1):

    # ==============================================================
    # TRAIN
    # ==============================================================

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    tr_total = 0.0
    tr_visual = 0.0
    tr_body = 0.0
    tr_legs = np.zeros(
        4,
        dtype=np.float64
    )

    n_train = 0

    start = time.time()

    for step, batch in enumerate(
        train_loader,
        start=1
    ):

        batch = move_batch(
            batch,
            device
        )

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp
        ):

            pred = forward_model(
                batch
            )

            (
                loss,
                L_vis,
                L_body,
                leg_losses
            ) = compute_losses(
                pred,
                batch
            )

            scaled_loss = (
                loss
                / GRAD_ACCUM_STEPS
            )

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

        tr_total += loss.detach().item()
        tr_visual += L_vis.detach().item()
        tr_body += L_body.detach().item()

        for leg in range(4):
            tr_legs[leg] += (
                leg_losses[leg]
                .detach()
                .item()
            )

        n_train += 1

        if step % 25 == 0:

            print(
                f"epoch {epoch:02d} | "
                f"step {step:05d}/{len(train_loader):05d} | "
                f"train={loss.item():.5f}"
            )

    tr_total /= max(n_train, 1)
    tr_visual /= max(n_train, 1)
    tr_body /= max(n_train, 1)
    tr_legs /= max(n_train, 1)

    # ==============================================================
    # VALIDATION
    # ==============================================================

    model.eval()

    va_total = 0.0
    va_visual = 0.0
    va_body = 0.0
    va_legs = np.zeros(
        4,
        dtype=np.float64
    )

    n_val = 0

    with torch.inference_mode():

        for batch in val_loader:

            batch = move_batch(
                batch,
                device
            )

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp
            ):

                pred = forward_model(
                    batch
                )

                (
                    loss,
                    L_vis,
                    L_body,
                    leg_losses
                ) = compute_losses(
                    pred,
                    batch
                )

            va_total += loss.item()
            va_visual += L_vis.item()
            va_body += L_body.item()

            for leg in range(4):
                va_legs[leg] += (
                    leg_losses[leg]
                    .item()
                )

            n_val += 1

    va_total /= max(n_val, 1)
    va_visual /= max(n_val, 1)
    va_body /= max(n_val, 1)
    va_legs /= max(n_val, 1)

    elapsed = (
        time.time()
        - start
    )

    row = {
        "epoch": epoch,

        "train_total":
            float(tr_total),

        "train_visual":
            float(tr_visual),

        "train_body":
            float(tr_body),

        "train_leg1":
            float(tr_legs[0]),

        "train_leg2":
            float(tr_legs[1]),

        "train_leg3":
            float(tr_legs[2]),

        "train_leg4":
            float(tr_legs[3]),

        "val_total":
            float(va_total),

        "val_visual":
            float(va_visual),

        "val_body":
            float(va_body),

        "val_leg1":
            float(va_legs[0]),

        "val_leg2":
            float(va_legs[1]),

        "val_leg3":
            float(va_legs[2]),

        "val_leg4":
            float(va_legs[3]),

        "seconds":
            float(elapsed),
    }

    history.append(
        row
    )

    print(
        "\n"
        f"EPOCH {epoch:02d}/{EPOCHS}\n"
        f"  train total = {tr_total:.6f}\n"
        f"  val   total = {va_total:.6f}\n"
        f"  val visual  = {va_visual:.6f}\n"
        f"  val body    = {va_body:.6f}\n"
        f"  val legs    = "
        f"{va_legs[0]:.5f}, "
        f"{va_legs[1]:.5f}, "
        f"{va_legs[2]:.5f}, "
        f"{va_legs[3]:.5f}\n"
        f"  time        = {elapsed/60:.2f} min\n"
    )

    # ==============================================================
    # CHECKPOINT
    # ==============================================================

    state = {
        "epoch":
            epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "history":
            history,

        "best_val_loss":
            min(
                best_val_loss,
                va_total
            ),

        "BODY_MEAN":
            np.asarray(
                BODY_MEAN
            ),

        "BODY_STD":
            np.asarray(
                BODY_STD
            ),

        "LEG_MEAN":
            np.asarray(
                LEG_MEAN
            ),

        "LEG_STD":
            np.asarray(
                LEG_STD
            ),

        "BODY_DIM":
            int(BODY_DIM),

        "LEG_DIM":
            int(LEG_DIM),

        "architecture": {
            "visual_groups": 9,
            "patches_per_group": 256,
            "transformer_tokens": 2304,
            "d_model": 768,
            "layers": 6,
            "heads": 8,
            "mlp": "768->3840->3072->768",
        },
    }

    torch.save(
        state,
        CHECKPOINT_DIR / "latest.pt"
    )

    if va_total < best_val_loss:

        best_val_loss = va_total

        state[
            "best_val_loss"
        ] = best_val_loss

        torch.save(
            state,
            CHECKPOINT_DIR / "best_val.pt"
        )

        print(
            "  saved new best_val.pt"
        )

print(
    "\nTRAINING COMPLETE."
)

print(
    "Best validation checkpoint:",
    CHECKPOINT_DIR / "best_val.pt"
)

# %% notebook cell 10
# ======================================================================
# FINAL HELD-OUT TEST METRICS
#
# Metrics requested:
#
# VISUAL:
#
#   L_z =
#       (1/256) sum_j
#       || zhat_(t+1)^j - z_(t+1)^j ||_2^2
#
#   cosine similarity =
#       (1/256) sum_j
#       cos(zhat_(t+1)^j, z_(t+1)^j)
#
# PER LEG (reported separately for LF, RF, LH, RH):
#
#   MAE q_HFE       [rad]
#   MAE q_KFE       [rad]
#   MAE tau_HFE     [N m]
#   MAE tau_KFE     [N m]
#
# NO roughness metric.
# NO yaw-vibration metric.
#
# The per-leg predictions and targets are DE-NORMALIZED before MAE,
# so the reported numbers are in physical units.
# ======================================================================

from pathlib import Path
import json

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from torch.utils.data import (
    DataLoader,
    Subset
)

# ----------------------------------------------------------------------
# 0. REQUIRED OBJECTS
# ----------------------------------------------------------------------

required = [
    "model",
    "dataset",
    "NormalizeTargets",
    "BODY_MEAN",
    "BODY_STD",
    "LEG_MEAN",
    "LEG_STD",
]

missing = [
    x for x in required
    if x not in globals()
]

if missing:

    raise RuntimeError(
        "Run Stage D first. Missing: "
        + ", ".join(missing)
    )


device = next(
    model.parameters()
).device


# ----------------------------------------------------------------------
# 1. LOAD BEST VALIDATION CHECKPOINT
# ----------------------------------------------------------------------

CHECKPOINT_DIR = Path(
    "/content/drive/MyDrive/FiveStreamFeelWM_GrandTour_checkpoints"
)

BEST_CKPT = (
    CHECKPOINT_DIR
    /
    "best_val.pt"
)

if not BEST_CKPT.exists():

    raise FileNotFoundError(
        "best_val.pt not found. "
        "Run the FINAL GRANDTOUR TRAINING CELL first."
    )


checkpoint = torch.load(
    BEST_CKPT,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

model.eval()

print(
    "Loaded checkpoint:",
    BEST_CKPT
)

print(
    "epoch:",
    checkpoint[
        "epoch"
    ]
)

print(
    "best val loss:",
    checkpoint[
        "best_val_loss"
    ]
)


# ----------------------------------------------------------------------
# 2. STRICT HELD-OUT TEMPORAL TEST SPLIT
#
# Stage D:
#   train = [0%,70%)
#   validation = [80%,90%)
#
# We use:
#   small temporal guard after validation
#   test = remaining final segment.
#
# 10 model samples = 1.0 s at dt=0.1 s.
# ----------------------------------------------------------------------

N = len(
    dataset
)

VAL_END = int(
    0.90 * N
)

TEST_GUARD = 10

TEST_START = (
    VAL_END
    +
    TEST_GUARD
)

if TEST_START >= N:

    raise RuntimeError(
        "Not enough held-out samples "
        "after temporal guard."
    )


test_indices = np.arange(
    TEST_START,
    N
)


test_raw = Subset(
    dataset,
    test_indices
)


test_dataset = NormalizeTargets(
    test_raw,
    BODY_MEAN,
    BODY_STD,
    LEG_MEAN,
    LEG_STD
)


test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)


print(
    "\nTotal dataset samples:",
    N
)

print(
    "Held-out test samples:",
    len(test_dataset)
)

print(
    "Test index range:",
    int(test_indices[0]),
    "to",
    int(test_indices[-1])
)


# ----------------------------------------------------------------------
# 3. TRAIN-NORMALIZATION CONSTANTS FOR DE-NORMALIZATION
# ----------------------------------------------------------------------

LEG_MEAN_T = torch.as_tensor(
    LEG_MEAN,
    dtype=torch.float32,
    device=device
)

LEG_STD_T = torch.as_tensor(
    LEG_STD,
    dtype=torch.float32,
    device=device
)


# ----------------------------------------------------------------------
# 4. ACCUMULATORS
# ----------------------------------------------------------------------

# Exact visual metric:
# average squared L2 error PER PATCH TOKEN.

lz_sum = 0.0
visual_patch_count = 0

cos_sum = 0.0
cos_count = 0

# [4 legs, 4 physical variables]
leg_abs_error_sum = np.zeros(
    (4, 4),
    dtype=np.float64
)

leg_sample_count = np.zeros(
    4,
    dtype=np.int64
)


# ----------------------------------------------------------------------
# 5. FORWARD HELPER
# ----------------------------------------------------------------------

def move_test_batch(
    batch,
    device
):

    return {
        k: (
            v.to(
                device,
                non_blocking=True
            )
            if torch.is_tensor(v)
            else v
        )
        for k, v
        in batch.items()
    }


def test_forward(
    batch
):

    return model(
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
            batch["a4_tm2"],
    )


# ----------------------------------------------------------------------
# 6. HELD-OUT TEST
# ----------------------------------------------------------------------

with torch.inference_mode():

    for batch in test_loader:

        batch = move_test_batch(
            batch,
            device
        )

        pred = test_forward(
            batch
        )


        # ==============================================================
        # A. EXACT L_z
        #
        # z_pred,z_true: [B,256,768]
        #
        # squared L2 for each patch:
        #
        #   ||zhat_j-z_j||_2^2
        #
        # then mean over ALL patches / test samples.
        # ==============================================================

        z_pred = (
            pred["z_next"]
            .float()
        )

        z_true = (
            batch["z_target"]
            .float()
        )


        patch_squared_l2 = (

            (z_pred - z_true)
            .pow(2)
            .sum(dim=-1)

        )

        # [B,256]

        lz_sum += (
            patch_squared_l2
            .sum()
            .item()
        )

        visual_patch_count += (
            patch_squared_l2
            .numel()
        )


        # ==============================================================
        # B. PATCHWISE COSINE SIMILARITY
        # ==============================================================

        patch_cos = F.cosine_similarity(
            z_pred,
            z_true,
            dim=-1
        )

        # [B,256]

        cos_sum += (
            patch_cos
            .sum()
            .item()
        )

        cos_count += (
            patch_cos
            .numel()
        )


        # ==============================================================
        # C. FOUR LEG MAEs
        #
        # Network output is normalized because Stage D normalized
        # leg_target using TRAIN statistics.
        #
        # De-normalize BOTH prediction and target before MAE.
        # ==============================================================

        for leg in range(4):

            pred_norm = (
                pred[
                    f"p_leg{leg+1}_next"
                ]
                .float()
            )

            true_norm = (
                batch[
                    "leg_target"
                ][:, leg]
                .float()
            )


            pred_phys = (

                pred_norm
                *
                LEG_STD_T[leg]

                +
                LEG_MEAN_T[leg]

            )


            true_phys = (

                true_norm
                *
                LEG_STD_T[leg]

                +
                LEG_MEAN_T[leg]

            )


            abs_error = (

                pred_phys
                -
                true_phys

            ).abs()


            leg_abs_error_sum[
                leg
            ] += (

                abs_error
                .sum(dim=0)
                .cpu()
                .numpy()
            )


            leg_sample_count[
                leg
            ] += (
                abs_error.shape[0]
            )


# ----------------------------------------------------------------------
# 7. FINAL METRICS
# ----------------------------------------------------------------------

L_z = (
    lz_sum
    /
    max(
        visual_patch_count,
        1
    )
)

cosine_similarity = (
    cos_sum
    /
    max(
        cos_count,
        1
    )
)

leg_mae = (

    leg_abs_error_sum
    /
    np.maximum(
        leg_sample_count[:, None],
        1
    )

)


# ----------------------------------------------------------------------
# 8. PRINT EXACT RESULTS
# ----------------------------------------------------------------------

print(
    "\n"
    +
    "=" * 90
)

print(
    "HELD-OUT TEST RESULTS"
)

print(
    "=" * 90
)


print(
    f"\nVisual L_z ↓                : "
    f"{L_z:.8f}"
)

print(
    f"Patch cosine similarity ↑   : "
    f"{cosine_similarity:.8f}"
)


LEG_LABELS = [
    "Leg 1 — LF",
    "Leg 2 — RF",
    "Leg 3 — LH",
    "Leg 4 — RH",
]

COLUMN_NAMES = [
    "MAE q_HFE [rad]",
    "MAE q_KFE [rad]",
    "MAE tau_HFE [N m]",
    "MAE tau_KFE [N m]",
]


results_table = pd.DataFrame(
    leg_mae,
    index=LEG_LABELS,
    columns=COLUMN_NAMES
)


print(
    "\nPer-leg physical prediction MAE ↓"
)

display(
    results_table
)


# ----------------------------------------------------------------------
# 9. SAVE RESULTS
# ----------------------------------------------------------------------

PROC = Path(
    "/content/grandtour_fivestream_pipeline/processed"
)

CSV_OUT = (
    PROC
    /
    "heldout_test_per_leg_mae.csv"
)

JSON_OUT = (
    PROC
    /
    "heldout_test_metrics.json"
)


results_table.to_csv(
    CSV_OUT
)


json_metrics = {

    "checkpoint":
        str(BEST_CKPT),

    "checkpoint_epoch":
        int(
            checkpoint["epoch"]
        ),

    "num_test_samples":
        int(
            len(test_dataset)
        ),

    "visual": {

        "L_z":
            float(L_z),

        "patch_cosine_similarity":
            float(
                cosine_similarity
            ),
    },

    "per_leg_mae": {

        LEG_LABELS[i]: {

            COLUMN_NAMES[j]:
                float(
                    leg_mae[i, j]
                )

            for j in range(4)
        }

        for i in range(4)
    },
}


with open(
    JSON_OUT,
    "w"
) as f:

    json.dump(
        json_metrics,
        f,
        indent=2
    )


print(
    "\nSaved CSV :",
    CSV_OUT
)

print(
    "Saved JSON:",
    JSON_OUT
)

print(
    "\nInterpretation:"
)

print(
    "  lower L_z is better"
)

print(
    "  higher cosine similarity is better"
)

print(
    "  lower all four MAEs are better"
)
