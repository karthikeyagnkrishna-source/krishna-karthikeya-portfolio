# Source inspection export of the adjacent notebook.
# Notebook execution order and dependencies still apply.
# IPython magic, shell lines and prose-only conclusion cells are comments.

# %% notebook cell 0
# ======================================================================
# END-TO-END COLAB CELL
#
# TartanGround P2000
#
# GOAL:
#   past 50 samples ending at t
#   + command active at t
#           ↓
#   predict physical response at t+1
#
# 100 Hz:
#   50 samples = 0.5 s history
#   t+1 = 0.01 s = 10 ms into future
#
# TARGET (18 values):
#   FL force xyz   = 3
#   FR force xyz   = 3
#   RL force xyz   = 3
#   RR force xyz   = 3
#   body linear velocity xyz  = 3
#   body angular velocity xyz = 3
#
# TRAINING:
#   STRICT chronological order
#   batch size = 1
#   no shuffle
#   no repeated epochs
#
#   predict t+1
#       ↓
#   reveal actual t+1
#       ↓
#   MSE
#       ↓
#   update parameters
#       ↓
#   shift one sample
#
# TEST:
#   same chronological/prequential process
#   prediction is SAVED BEFORE actual t+1 is used for update
#
# COUPLING:
#   Flat MLP only
#   no graph attention
# ======================================================================


# ======================================================================
# 1. INSTALL
# ======================================================================

# IPYTHON: !pip install -q git+https://github.com/castacks/tartanairpy.git
# IPYTHON: !pip install -q rosbags


# ======================================================================
# 2. IMPORTS
# ======================================================================

import os
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import tartanair as ta
from rosbags.highlevel import AnyReader


# ======================================================================
# 3. REPRODUCIBILITY + GPU
# ======================================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ======================================================================
# 4. DOWNLOAD ONLY DATA REQUIRED FOR THIS EXPERIMENT
#
# We do NOT need RGB/depth for this one-step proprioceptive experiment.
# We need:
#   - IMU
#   - ROS bag
# ======================================================================

root = "/content/tartanground"

ta.init(root)

ta.download_ground(
    env=["OldTownFall"],
    version=["anymal"],
    traj=["P2000"],
    modality=[
        "imu",
        "rosbag",
    ],
    camera_name=["lcam_front"],
    unzip=True
)


# ======================================================================
# 5. PATHS
# ======================================================================

base = "/content/tartanground/OldTownFall/Data_anymal/P2000"

imu_dir = os.path.join(
    base,
    "imu"
)

bag_path = Path(
    os.path.join(
        base,
        "rosbags",
        "P2000.bag"
    )
)

assert os.path.exists(imu_dir)
assert bag_path.exists()

print("\nDataset ready:")
print("IMU:", imu_dir)
print("Bag:", bag_path)


# ======================================================================
# 6. LOAD IMU
#
# imu_time is already approximately:
# 0.00, 0.01, 0.02, ...
# ======================================================================

imu_time = np.load(
    os.path.join(
        imu_dir,
        "imu_time.npy"
    )
).astype(np.float64)

acc = np.load(
    os.path.join(
        imu_dir,
        "acc.npy"
    )
).astype(np.float32)

gyro = np.loadtxt(
    os.path.join(
        imu_dir,
        "gyro.txt"
    )
).astype(np.float32)

print("\nIMU")
print("time :", imu_time.shape)
print("acc  :", acc.shape)
print("gyro :", gyro.shape)

print(
    "median dt:",
    np.median(
        np.diff(imu_time)
    )
)

print(
    "approx frequency:",
    1.0 / np.median(np.diff(imu_time)),
    "Hz"
)


# ======================================================================
# 7. ROS TOPICS
# ======================================================================

topics = {

    "joints":
        "/state_estimator/joint_states",

    "lf":
        "/state_estimator/contact_force_lf_foot",

    "rf":
        "/state_estimator/contact_force_rf_foot",

    "lh":
        "/state_estimator/contact_force_lh_foot",

    "rh":
        "/state_estimator/contact_force_rh_foot",

    "twist":
        "/state_estimator/twist",

    "command":
        "/path_planning_and_following/twist",
}


# ======================================================================
# 8. FIND COMMON BAG TIME ORIGIN
#
# State streams start before the command stream.
# Use earliest physical-state ROS message as t = 0 reference.
# ======================================================================

physical_topics = {

    topics["joints"],
    topics["lf"],
    topics["rf"],
    topics["lh"],
    topics["rh"],
    topics["twist"],
}

with AnyReader([bag_path]) as reader:

    selected = [
        c
        for c in reader.connections
        if c.topic in physical_topics
    ]

    first_message = next(
        reader.messages(
            connections=selected
        )
    )

    _connection, t0_ns, _raw = first_message


t0_sec = t0_ns * 1e-9

print(
    "\nROS physical-state time origin:",
    t0_sec
)


# ======================================================================
# 9. HELPER FOR HEADER TIMESTAMP
#
# Prefer the message's sensor timestamp.
# Fall back to bag timestamp if necessary.
# ======================================================================

def get_message_time(
    msg,
    bag_timestamp_ns
):

    try:

        stamp = msg.header.stamp

        absolute = (
            float(stamp.sec)
            + float(stamp.nanosec) * 1e-9
        )

    except Exception:

        absolute = (
            bag_timestamp_ns * 1e-9
        )

    return (
        absolute - t0_sec
    )


# ======================================================================
# 10. EXTRACT ALL REQUIRED ROS DATA
# ======================================================================

data = {

    "joints": {
        "t": [],
        "x": []
    },

    "lf": {
        "t": [],
        "x": []
    },

    "rf": {
        "t": [],
        "x": []
    },

    "lh": {
        "t": [],
        "x": []
    },

    "rh": {
        "t": [],
        "x": []
    },

    "twist": {
        "t": [],
        "x": []
    },

    "command": {
        "t": [],
        "x": []
    },
}


# Canonical TartanGround ANYmal ordering
expected_joint_names = [

    "LF_HAA",
    "LF_HFE",
    "LF_KFE",

    "RF_HAA",
    "RF_HFE",
    "RF_KFE",

    "LH_HAA",
    "LH_HFE",
    "LH_KFE",

    "RH_HAA",
    "RH_HFE",
    "RH_KFE",
]

joint_reorder = None


with AnyReader([bag_path]) as reader:

    selected = [

        c
        for c in reader.connections

        if c.topic
        in set(topics.values())
    ]

    for (
        connection,
        timestamp,
        rawdata
    ) in reader.messages(
        connections=selected
    ):

        msg = reader.deserialize(
            rawdata,
            connection.msgtype
        )

        topic = connection.topic

        t_msg = get_message_time(
            msg,
            timestamp
        )


        # ==============================================================
        # JOINT STATES
        #
        # 12 q
        # 12 dq
        # 12 effort
        # ==============================================================

        if topic == topics["joints"]:

            if joint_reorder is None:

                names = list(
                    msg.name
                )

                missing = [
                    n
                    for n in expected_joint_names
                    if n not in names
                ]

                if missing:

                    raise RuntimeError(
                        "Missing expected joints: "
                        + str(missing)
                    )

                joint_reorder = np.array(
                    [
                        names.index(n)
                        for n
                        in expected_joint_names
                    ],
                    dtype=int
                )

                print(
                    "\nJoint order verified:"
                )

                print(
                    expected_joint_names
                )


            position = np.asarray(
                msg.position,
                dtype=np.float32
            )[joint_reorder]

            velocity = np.asarray(
                msg.velocity,
                dtype=np.float32
            )[joint_reorder]

            effort = np.asarray(
                msg.effort,
                dtype=np.float32
            )[joint_reorder]


            x = np.concatenate(
                [
                    position,
                    velocity,
                    effort
                ]
            )


            data["joints"]["t"].append(
                t_msg
            )

            data["joints"]["x"].append(
                x
            )


        # ==============================================================
        # CONTACT FORCES
        # ==============================================================

        elif topic in {

            topics["lf"],
            topics["rf"],
            topics["lh"],
            topics["rh"],

        }:

            force = np.array(
                [
                    msg.wrench.force.x,
                    msg.wrench.force.y,
                    msg.wrench.force.z,
                ],
                dtype=np.float32
            )


            if topic == topics["lf"]:

                key = "lf"

            elif topic == topics["rf"]:

                key = "rf"

            elif topic == topics["lh"]:

                key = "lh"

            else:

                key = "rh"


            data[key]["t"].append(
                t_msg
            )

            data[key]["x"].append(
                force
            )


        # ==============================================================
        # BODY TWIST
        #
        # vx vy vz wx wy wz
        # ==============================================================

        elif topic == topics["twist"]:

            tw = msg.twist.twist

            x = np.array(
                [
                    tw.linear.x,
                    tw.linear.y,
                    tw.linear.z,

                    tw.angular.x,
                    tw.angular.y,
                    tw.angular.z,
                ],
                dtype=np.float32
            )

            data["twist"]["t"].append(
                t_msg
            )

            data["twist"]["x"].append(
                x
            )


        # ==============================================================
        # HIGH-LEVEL TWIST COMMAND
        #
        # ux uy uz uwx uwy uwz
        # ==============================================================

        elif topic == topics["command"]:

            tw = msg.twist

            x = np.array(
                [
                    tw.linear.x,
                    tw.linear.y,
                    tw.linear.z,

                    tw.angular.x,
                    tw.angular.y,
                    tw.angular.z,
                ],
                dtype=np.float32
            )

            data["command"]["t"].append(
                t_msg
            )

            data["command"]["x"].append(
                x
            )


# Convert to NumPy
for key in data:

    data[key]["t"] = np.asarray(
        data[key]["t"],
        dtype=np.float64
    )

    data[key]["x"] = np.asarray(
        data[key]["x"],
        dtype=np.float32
    )

    # sort by timestamp just in case
    order = np.argsort(
        data[key]["t"]
    )

    data[key]["t"] = (
        data[key]["t"][order]
    )

    data[key]["x"] = (
        data[key]["x"][order]
    )

    print(
        f"{key:8s}",
        "t:",
        data[key]["t"].shape,
        "x:",
        data[key]["x"].shape,
        "range:",
        (
            data[key]["t"][0],
            data[key]["t"][-1]
        )
    )


# ======================================================================
# 11. CAUSAL SYNCHRONIZATION
#
# IMPORTANT:
# We do NOT linearly interpolate using a future sample.
#
# At time t:
# take the latest measurement whose timestamp <= t.
#
# This is zero-order hold / last observation carried forward.
# ======================================================================

def causal_hold(
    t_src,
    x_src,
    t_target
):

    idx = np.searchsorted(
        t_src,
        t_target,
        side="right"
    ) - 1

    if np.any(idx < 0):

        raise RuntimeError(
            "Target time occurs before "
            "first source measurement."
        )

    return x_src[idx]


# ======================================================================
# 12. FIND COMMON VALID TIME RANGE
# ======================================================================

stream_keys = [
    "joints",
    "lf",
    "rf",
    "lh",
    "rh",
    "twist",
    "command",
]

common_start = max(

    imu_time[0],

    *[
        data[k]["t"][0]
        for k in stream_keys
    ]
)

common_end = min(

    imu_time[-1],

    *[
        data[k]["t"][-1]
        for k in stream_keys
    ]
)


grid_mask = (

    (imu_time >= common_start)
    &
    (imu_time <= common_end)

)


grid_original = (
    imu_time[grid_mask]
)


acc_sync = (
    acc[grid_mask]
)

gyro_sync = (
    gyro[grid_mask]
)


print(
    "\nCommon range:",
    common_start,
    "to",
    common_end
)

print(
    "100 Hz samples:",
    len(grid_original)
)


# ======================================================================
# 13. SYNCHRONIZE STATE + COMMAND STREAMS
# ======================================================================

joint_sync = causal_hold(
    data["joints"]["t"],
    data["joints"]["x"],
    grid_original
)

lf_force = causal_hold(
    data["lf"]["t"],
    data["lf"]["x"],
    grid_original
)

rf_force = causal_hold(
    data["rf"]["t"],
    data["rf"]["x"],
    grid_original
)

lh_force = causal_hold(
    data["lh"]["t"],
    data["lh"]["x"],
    grid_original
)

rh_force = causal_hold(
    data["rh"]["t"],
    data["rh"]["x"],
    grid_original
)

body_twist = causal_hold(
    data["twist"]["t"],
    data["twist"]["x"],
    grid_original
)

command_sync = causal_hold(
    data["command"]["t"],
    data["command"]["x"],
    grid_original
)


# ======================================================================
# 14. RESET CROPPED TIME TO ZERO
# ======================================================================

crop_start_original = (
    grid_original[0]
)

time = (
    grid_original
    - crop_start_original
)

command_event_times = (
    data["command"]["t"]
    - crop_start_original
)


print(
    "\nCropped duration:",
    time[-1],
    "seconds"
)

print(
    "Median dt:",
    np.median(
        np.diff(time)
    )
)


# ======================================================================
# 15. CONSTRUCT P
#
# Per leg 12 features:
#
# q1 q2 q3
# dq1 dq2 dq3
# tau1 tau2 tau3
# Fx Fy Fz
# ======================================================================

q = joint_sync[:, 0:12]

dq = joint_sync[:, 12:24]

tau = joint_sync[:, 24:36]


leg_idx = {

    "FL": [0, 1, 2],

    "FR": [3, 4, 5],

    "RL": [6, 7, 8],

    "RR": [9, 10, 11],
}


forces = {

    "FL": lf_force,

    "FR": rf_force,

    "RL": lh_force,

    "RR": rh_force,
}


leg_data = {}


for leg, idx in leg_idx.items():

    leg_data[leg] = np.concatenate(
        [
            q[:, idx],
            dq[:, idx],
            tau[:, idx],
            forces[leg]
        ],
        axis=1
    )


P = np.stack(
    [
        leg_data["FL"],
        leg_data["FR"],
        leg_data["RL"],
        leg_data["RR"]
    ],
    axis=1
).astype(np.float32)


# ======================================================================
# 16. CONSTRUCT B
#
# B =
# acc xyz
# gyro xyz
# body twist:
# vx vy vz wx wy wz
#
# 3 + 3 + 6 = 12
# ======================================================================

B = np.concatenate(
    [
        acc_sync,
        gyro_sync,
        body_twist
    ],
    axis=1
).astype(np.float32)


# high-level command
U = command_sync.astype(
    np.float32
)


print("\nFinal synchronized data:")
print("time:", time.shape)
print("P   :", P.shape)
print("B   :", B.shape)
print("U   :", U.shape)


# ======================================================================
# 17. VERIFY COMMAND DOES NOT CHANGE INSIDE EACH 10-ms TRANSITION
#
# For transition:
#
#     time[t] -------- time[t+1]
#
# An event exactly at time[t+1] is okay:
# it belongs to the NEXT control interval.
#
# We reject a transition if a new command arrives strictly INSIDE it.
# ======================================================================

N = len(time)

no_mid_command_change = np.zeros(
    N - 1,
    dtype=bool
)


TIME_TOL = 1e-7


for i in range(N - 1):

    left = time[i]
    right = time[i + 1]

    j = np.searchsorted(
        command_event_times,
        left + TIME_TOL,
        side="right"
    )

    if j >= len(command_event_times):

        no_mid_command_change[i] = True

    else:

        next_event = (
            command_event_times[j]
        )

        no_mid_command_change[i] = (
            next_event
            >= right - TIME_TOL
        )


print(
    "\nTransitions with no command event "
    "inside t -> t+1:",
    int(
        no_mid_command_change.sum()
    ),
    "/",
    len(no_mid_command_change)
)

print(
    "Fraction:",
    100.0
    * no_mid_command_change.mean(),
    "%"
)


if no_mid_command_change.sum() < 500:

    raise RuntimeError(
        "\nToo few clean t -> t+1 transitions.\n"
        "Do not continue training.\n"
        "Send this output back so the command timing "
        "can be inspected first."
    )


# ======================================================================
# 18. EXPERIMENT SETTINGS
# ======================================================================

T = 50

# First few seconds are ONLY used to establish fixed scaling.
#
# This avoids using future test data in normalization.
# Training begins after this calibration period.
CALIBRATION_SEC = 5.0

# Sequential gradient updates stop here.
TRAIN_END_SEC = 87.0

D_MODEL = 128
NHEAD = 4
TEMPORAL_LAYERS = 2

LR = 1e-4
WEIGHT_DECAY = 1e-4

# No stochastic dropout in batch-size-1 sequential experiment.
DROPOUT = 0.0

# During the test trajectory:
#
# Predict first.
# Save prediction.
# Reveal actual.
# Score.
# THEN use actual to update before next prediction.
#
# This is prequential / walk-forward testing.
ONLINE_UPDATE_DURING_TEST = True


calibration_end_idx = (
    np.searchsorted(
        time,
        CALIBRATION_SEC,
        side="right"
    ) - 1
)

train_end_idx = (
    np.searchsorted(
        time,
        TRAIN_END_SEC,
        side="right"
    ) - 1
)


print(
    "\nCalibration ends:",
    calibration_end_idx,
    time[calibration_end_idx]
)

print(
    "Training ends:",
    train_end_idx,
    time[train_end_idx]
)


# ======================================================================
# 19. FIXED CAUSAL NORMALIZATION
#
# Statistics are computed ONLY from first 5 seconds.
#
# Everything after 5 s is unseen when scaling is defined.
# ======================================================================

eps = 1e-6


cal_P = P[
    :calibration_end_idx + 1
]

cal_B = B[
    :calibration_end_idx + 1
]

cal_U = U[
    :calibration_end_idx + 1
]


leg_mean = cal_P.mean(
    axis=(0, 1)
)

leg_std = cal_P.std(
    axis=(0, 1)
) + eps


body_mean = cal_B.mean(
    axis=0
)

body_std = cal_B.std(
    axis=0
) + eps


cmd_mean = cal_U.mean(
    axis=0
)

cmd_std = cal_U.std(
    axis=0
) + eps


cal_targets = np.concatenate(
    [
        cal_P[:, :, 9:12].reshape(
            -1,
            12
        ),

        cal_B[:, 6:12]
    ],
    axis=1
)


y_mean = cal_targets.mean(
    axis=0
)

y_std = cal_targets.std(
    axis=0
) + eps


print(
    "\nNormalization fixed using only",
    CALIBRATION_SEC,
    "seconds."
)


# ======================================================================
# 20. MAKE EXACTLY ONE CAUSAL t -> t+1 EXAMPLE
# ======================================================================

def make_one_step_example(t):

    # --------------------------------------------------------------
    # INPUT HISTORY:
    #
    # t-49 ... t
    #
    # exactly 50 samples
    # --------------------------------------------------------------

    leg_history = P[
        t - T + 1 :
        t + 1
    ]

    # 50 x 4 x 12
    #
    # Transformer expects:
    # 4 x 50 x 12

    leg_history = np.transpose(
        leg_history,
        (1, 0, 2)
    )


    body_history = B[
        t - T + 1 :
        t + 1
    ]

    # 50 x 12


    current_command = U[t]


    # --------------------------------------------------------------
    # ACTUAL NEXT PHYSICAL RESPONSE AT t+1
    # --------------------------------------------------------------

    next_forces = P[
        t + 1,
        :,
        9:12
    ].reshape(12)


    next_body_twist = B[
        t + 1,
        6:12
    ]


    actual_next = np.concatenate(
        [
            next_forces,
            next_body_twist
        ]
    ).astype(np.float32)


    # --------------------------------------------------------------
    # NORMALIZE INPUT
    # --------------------------------------------------------------

    leg_n = (
        leg_history
        - leg_mean[None, None, :]
    ) / leg_std[None, None, :]


    body_n = (
        body_history
        - body_mean[None, :]
    ) / body_std[None, :]


    cmd_n = (
        current_command
        - cmd_mean
    ) / cmd_std


    actual_n = (
        actual_next
        - y_mean
    ) / y_std


    # --------------------------------------------------------------
    # TORCH
    # batch size = 1
    # --------------------------------------------------------------

    leg_tensor = torch.tensor(
        leg_n,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)


    body_tensor = torch.tensor(
        body_n,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)


    cmd_tensor = torch.tensor(
        cmd_n,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)


    target_tensor = torch.tensor(
        actual_n,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)


    return (
        leg_tensor,
        body_tensor,
        cmd_tensor,
        target_tensor,
        actual_next
    )


# ======================================================================
# 21. SHARED LEG TEMPORAL TRANSFORMER
# ======================================================================

class SharedLegTransformer(nn.Module):

    def __init__(
        self,
        input_dim=12,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_ff=256,
        max_len=50,
        dropout=0.0
    ):

        super().__init__()

        self.d_model = d_model

        self.input_proj = nn.Linear(
            input_dim,
            d_model
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(
                1,
                max_len,
                d_model
            ) * 0.02
        )

        # FL FR RL RR
        self.leg_embedding = nn.Embedding(
            4,
            d_model
        )


        layer = nn.TransformerEncoderLayer(

            d_model=d_model,

            nhead=nhead,

            dim_feedforward=dim_ff,

            dropout=dropout,

            batch_first=True,

            norm_first=True,

            activation="gelu"
        )


        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(
            d_model
        )


    def forward(
        self,
        x
    ):

        # x:
        # batch x 4 x 50 x 12

        batch_size, legs, seq_len, features = (
            x.shape
        )


        # SAME temporal Transformer weights
        # for all four legs

        x = x.reshape(
            batch_size * legs,
            seq_len,
            features
        )


        x = self.input_proj(
            x
        )


        # chronological position within the 50-step history

        x = (
            x
            + self.pos_embedding[
                :,
                :seq_len
            ]
        )


        leg_ids = torch.arange(
            legs,
            device=x.device
        ).repeat(
            batch_size
        )


        x = (
            x
            + self.leg_embedding(
                leg_ids
            ).unsqueeze(1)
        )


        # no temporal token can look into future
        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=x.device
            ),
            diagonal=1
        )


        x = self.transformer(
            x,
            mask=causal_mask
        )


        x = self.norm(
            x
        )


        # representation at current time t
        h = x[
            :,
            -1,
            :
        ]


        h = h.reshape(
            batch_size,
            legs,
            self.d_model
        )


        return h


# ======================================================================
# 22. BODY TEMPORAL TRANSFORMER
# ======================================================================

class BodyTransformer(nn.Module):

    def __init__(
        self,
        input_dim=12,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_ff=256,
        max_len=50,
        dropout=0.0
    ):

        super().__init__()


        self.input_proj = nn.Linear(
            input_dim,
            d_model
        )


        self.pos_embedding = nn.Parameter(
            torch.randn(
                1,
                max_len,
                d_model
            ) * 0.02
        )


        layer = nn.TransformerEncoderLayer(

            d_model=d_model,

            nhead=nhead,

            dim_feedforward=dim_ff,

            dropout=dropout,

            batch_first=True,

            norm_first=True,

            activation="gelu"
        )


        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers
        )


        self.norm = nn.LayerNorm(
            d_model
        )


    def forward(
        self,
        x
    ):

        batch_size, seq_len, features = (
            x.shape
        )


        x = self.input_proj(
            x
        )


        x = (
            x
            + self.pos_embedding[
                :,
                :seq_len
            ]
        )


        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=x.device
            ),
            diagonal=1
        )


        x = self.transformer(
            x,
            mask=causal_mask
        )


        x = self.norm(
            x
        )


        return x[
            :,
            -1,
            :
        ]


# ======================================================================
# 23. FLAT MLP COUPLING MODEL
# ======================================================================

class OneStepFlatMLP(nn.Module):

    def __init__(
        self,
        d_model=128,
        nhead=4,
        temporal_layers=2,
        dropout=0.0,
        coupling_hidden=208
    ):

        super().__init__()


        self.leg_encoder = (
            SharedLegTransformer(

                input_dim=12,

                d_model=d_model,

                nhead=nhead,

                num_layers=temporal_layers,

                dim_ff=2 * d_model,

                max_len=T,

                dropout=dropout
            )
        )


        self.body_encoder = (
            BodyTransformer(

                input_dim=12,

                d_model=d_model,

                nhead=nhead,

                num_layers=temporal_layers,

                dim_ff=2 * d_model,

                max_len=T,

                dropout=dropout
            )
        )


        # command at current time t
        self.command_encoder = nn.Sequential(

            nn.Linear(
                6,
                64
            ),

            nn.GELU(),

            nn.Linear(
                64,
                d_model
            )
        )


        # combine each leg's temporal state
        # with current command
        self.leg_node_proj = nn.Sequential(

            nn.Linear(
                2 * d_model,
                d_model
            ),

            nn.GELU(),

            nn.LayerNorm(
                d_model
            )
        )


        # combine body temporal state
        # with current command
        self.body_node_proj = nn.Sequential(

            nn.Linear(
                2 * d_model,
                d_model
            ),

            nn.GELU(),

            nn.LayerNorm(
                d_model
            )
        )


        # ==============================================================
        # FLAT COUPLING
        #
        # BODY + FL + FR + RL + RR
        #
        # 5 x 128 = 640 dimensions
        # ==============================================================

        flat_dim = (
            5 * d_model
        )


        self.flat_norm = nn.LayerNorm(
            flat_dim
        )


        self.coupling_mlp = nn.Sequential(

            nn.Linear(
                flat_dim,
                coupling_hidden
            ),

            nn.GELU(),

            nn.Linear(
                coupling_hidden,
                flat_dim
            )
        )


        # same force decoder for all legs
        self.leg_head = nn.Sequential(

            nn.Linear(
                d_model,
                64
            ),

            nn.GELU(),

            nn.Linear(
                64,
                3
            )
        )


        self.body_head = nn.Sequential(

            nn.Linear(
                d_model,
                64
            ),

            nn.GELU(),

            nn.Linear(
                64,
                6
            )
        )


    def forward(
        self,
        x_leg,
        x_body,
        x_cmd
    ):

        batch_size = (
            x_leg.shape[0]
        )


        # --------------------------------------------------------------
        # TEMPORAL ENCODING
        # --------------------------------------------------------------

        leg_h = self.leg_encoder(
            x_leg
        )

        # batch x 4 x 128


        body_h = self.body_encoder(
            x_body
        )

        # batch x 128


        cmd_h = self.command_encoder(
            x_cmd
        )

        # batch x 128


        # --------------------------------------------------------------
        # LEG STATES
        # --------------------------------------------------------------

        cmd_for_legs = (
            cmd_h
            .unsqueeze(1)
            .expand(
                -1,
                4,
                -1
            )
        )


        leg_nodes = self.leg_node_proj(

            torch.cat(
                [
                    leg_h,
                    cmd_for_legs
                ],
                dim=-1
            )
        )


        # --------------------------------------------------------------
        # BODY STATE
        # --------------------------------------------------------------

        body_node = self.body_node_proj(

            torch.cat(
                [
                    body_h,
                    cmd_h
                ],
                dim=-1
            )
        ).unsqueeze(1)


        # --------------------------------------------------------------
        # BODY FL FR RL RR
        # --------------------------------------------------------------

        nodes = torch.cat(
            [
                body_node,
                leg_nodes
            ],
            dim=1
        )


        flat = nodes.reshape(
            batch_size,
            -1
        )


        # generic global coupling
        update = self.coupling_mlp(
            self.flat_norm(
                flat
            )
        )


        coupled = (
            flat + update
        )


        coupled_nodes = (
            coupled.reshape(
                batch_size,
                5,
                -1
            )
        )


        body_final = (
            coupled_nodes[
                :,
                0
            ]
        )


        leg_final = (
            coupled_nodes[
                :,
                1:5
            ]
        )


        # --------------------------------------------------------------
        # t+1 OUTPUT
        # --------------------------------------------------------------

        leg_force = (
            self.leg_head(
                leg_final
            )
            .reshape(
                batch_size,
                12
            )
        )


        body_twist_out = (
            self.body_head(
                body_final
            )
        )


        return torch.cat(
            [
                leg_force,
                body_twist_out
            ],
            dim=-1
        )


# ======================================================================
# 24. CREATE FRESH MODEL
# ======================================================================

model = OneStepFlatMLP(

    d_model=D_MODEL,

    nhead=NHEAD,

    temporal_layers=TEMPORAL_LAYERS,

    dropout=DROPOUT,

    coupling_hidden=208

).to(device)


parameter_count = sum(
    p.numel()
    for p in model.parameters()
)


print(
    "\nModel parameters:",
    f"{parameter_count:,}"
)


criterion = nn.MSELoss()


optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LR,

    weight_decay=WEIGHT_DECAY
)


# ======================================================================
# 25. STRICTLY CHRONOLOGICAL TRAINING
#
# No DataLoader.
# No shuffle.
# No batch of 64.
# No second epoch.
#
# Each physical transition gets exactly:
#
# predict t+1
# reveal actual
# update
# move to next t
# ======================================================================

first_train_t = max(

    T - 1,

    calibration_end_idx
)


train_losses = []

train_steps = 0

skipped_train_steps = 0


print(
    "\n===================================================="
)

print(
    "STARTING STRICT CHRONOLOGICAL TRAINING"
)

print(
    "===================================================="
)


for t in range(
    first_train_t,
    train_end_idx
):


    # Do NOT use a transition if command changes
    # somewhere strictly inside t -> t+1.

    if not no_mid_command_change[t]:

        skipped_train_steps += 1

        continue


    (
        leg_t,
        body_t,
        cmd_t,
        target_t,
        actual_raw
    ) = make_one_step_example(
        t
    )


    model.train()


    # --------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------

    optimizer.zero_grad()


    pred_t = model(
        leg_t,
        body_t,
        cmd_t
    )


    # --------------------------------------------------------------
    # Now compare with real t+1
    # --------------------------------------------------------------

    loss = criterion(
        pred_t,
        target_t
    )


    # --------------------------------------------------------------
    # Correct model
    # --------------------------------------------------------------

    loss.backward()


    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )


    optimizer.step()


    train_losses.append(
        float(
            loss.item()
        )
    )


    train_steps += 1


    if train_steps % 500 == 0:

        recent_mse = np.mean(
            train_losses[-500:]
        )


        print(

            f"update {train_steps:5d} | "

            f"{time[t]:7.2f}s -> "
            f"{time[t+1]:7.2f}s | "

            f"recent normalized MSE "
            f"{recent_mse:.6f}"
        )


print(
    "\nTraining finished."
)

print(
    "Gradient updates:",
    train_steps
)

print(
    "Skipped command-changing transitions:",
    skipped_train_steps
)

print(
    "Mean normalized training MSE:",
    float(
        np.mean(
            train_losses
        )
    )
)


# ======================================================================
# 26. WALK-FORWARD / PREQUENTIAL TEST
#
# Critical rule:
#
# At every test time:
#
#     FIRST  make prediction for t+1
#     THEN   save prediction
#     THEN   reveal actual t+1
#     THEN   calculate error
#     ONLY THEN may model update
#
# Therefore the target being scored was never available
# when its own prediction was generated.
# ======================================================================

predictions = []

actuals = []

persistence_predictions = []

prediction_times = []

source_times = []

source_indices = []

test_step_losses = []


skipped_test_steps = 0


print(
    "\n===================================================="
)

print(
    "STARTING WALK-FORWARD TEST"
)

print(
    "===================================================="
)


for t in range(
    train_end_idx,
    N - 1
):


    if not no_mid_command_change[t]:

        skipped_test_steps += 1

        continue


    (
        leg_t,
        body_t,
        cmd_t,
        target_t,
        actual_raw
    ) = make_one_step_example(
        t
    )


    # ==============================================================
    # A. PREDICT BEFORE SEEING t+1
    # ==============================================================

    model.eval()


    with torch.no_grad():

        pred_n = model(
            leg_t,
            body_t,
            cmd_t
        )


    pred_n = (
        pred_n
        .detach()
        .cpu()
        .numpy()[0]
    )


    predicted_raw = (

        pred_n
        * y_std
        + y_mean
    )


    # ==============================================================
    # B. SAVE PREDICTION
    # ==============================================================

    predictions.append(
        predicted_raw.copy()
    )


    actuals.append(
        actual_raw.copy()
    )


    source_times.append(
        time[t]
    )


    prediction_times.append(
        time[t + 1]
    )


    source_indices.append(
        t
    )


    # ==============================================================
    # C. PERSISTENCE BASELINE
    #
    # "next value = current value"
    # ==============================================================

    current_forces = P[
        t,
        :,
        9:12
    ].reshape(12)


    current_twist = B[
        t,
        6:12
    ]


    persistence_raw = np.concatenate(
        [
            current_forces,
            current_twist
        ]
    )


    persistence_predictions.append(
        persistence_raw
    )


    # ==============================================================
    # D. ACTUAL t+1 IS NOW REVEALED
    #
    # Only AFTER our prediction was stored.
    # ==============================================================

    scored_pred_n = torch.tensor(
        pred_n,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)


    test_loss = criterion(
        scored_pred_n,
        target_t
    )


    test_step_losses.append(
        float(
            test_loss.item()
        )
    )


    # ==============================================================
    # E. NOW UPDATE USING THE NEWLY OBSERVED t+1
    #
    # This update can only help future predictions:
    # t+2, t+3, ...
    #
    # It cannot change the already stored t+1 prediction.
    # ==============================================================

    if ONLINE_UPDATE_DURING_TEST:

        model.train()

        optimizer.zero_grad()


        fresh_pred = model(
            leg_t,
            body_t,
            cmd_t
        )


        update_loss = criterion(
            fresh_pred,
            target_t
        )


        update_loss.backward()


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )


        optimizer.step()


# NumPy arrays
predictions = np.asarray(
    predictions,
    dtype=np.float32
)

actuals = np.asarray(
    actuals,
    dtype=np.float32
)

persistence_predictions = np.asarray(
    persistence_predictions,
    dtype=np.float32
)

source_times = np.asarray(
    source_times,
    dtype=np.float64
)

prediction_times = np.asarray(
    prediction_times,
    dtype=np.float64
)


print(
    "\nWalk-forward test completed."
)

print(
    "Predictions:",
    predictions.shape
)

print(
    "Actuals:",
    actuals.shape
)

print(
    "Skipped test transitions:",
    skipped_test_steps
)


# ======================================================================
# 27. METRIC DEFINITIONS
# ======================================================================

output_groups = [

    (
        "FL force",
        0,
        3
    ),

    (
        "FR force",
        3,
        6
    ),

    (
        "RL force",
        6,
        9
    ),

    (
        "RR force",
        9,
        12
    ),

    (
        "Body linear velocity",
        12,
        15
    ),

    (
        "Body angular velocity",
        15,
        18
    ),
]


def rmse(
    prediction,
    actual
):

    return float(
        np.sqrt(
            np.mean(
                (
                    prediction
                    - actual
                ) ** 2
            )
        )
    )


def mae(
    prediction,
    actual
):

    return float(
        np.mean(
            np.abs(
                prediction
                - actual
            )
        )
    )


# ======================================================================
# 28. AGGREGATE MODEL RESULTS
# ======================================================================

print(
    "\n===================================================="
)

print(
    "ONE-STEP MODEL RESULTS"
)

print(
    "10 ms WALK-FORWARD PREDICTION"
)

print(
    "===================================================="
)


for (
    name,
    a,
    b
) in output_groups:

    value_rmse = rmse(

        predictions[
            :,
            a:b
        ],

        actuals[
            :,
            a:b
        ]
    )


    value_mae = mae(

        predictions[
            :,
            a:b
        ],

        actuals[
            :,
            a:b
        ]
    )


    print(

        f"{name:25s} | "

        f"RMSE {value_rmse:10.6f} | "

        f"MAE {value_mae:10.6f}"
    )


# ======================================================================
# 29. PERSISTENCE BASELINE
# ======================================================================

print(
    "\n===================================================="
)

print(
    "10-ms PERSISTENCE BASELINE"
)

print(
    "===================================================="
)


for (
    name,
    a,
    b
) in output_groups:

    value_rmse = rmse(

        persistence_predictions[
            :,
            a:b
        ],

        actuals[
            :,
            a:b
        ]
    )


    print(
        f"{name:25s}: "
        f"{value_rmse:.6f}"
    )


# ======================================================================
# 30. PRINT MULTIPLE REAL TEST EXAMPLES
#
# These are spread across the test trajectory.
#
# For each one we show:
#
#   input/current time t
#   predicted time t+1
#   predicted values
#   actual recorded values
#   prediction error
# ======================================================================

N_SHOW = min(
    10,
    len(predictions)
)


show_ids = np.linspace(

    0,

    len(predictions) - 1,

    N_SHOW,

    dtype=int
)


print(
    "\n===================================================="
)

print(
    "REAL t+1 TEST EXAMPLES"
)

print(
    "PREDICTION WAS MADE BEFORE TARGET WAS REVEALED"
)

print(
    "===================================================="
)


for example_number, result_idx in enumerate(
    show_ids,
    start=1
):

    original_t = (
        source_indices[
            result_idx
        ]
    )


    print(
        "\n"
        + "=" * 75
    )


    print(
        f"EXAMPLE {example_number}"
    )


    print(
        f"input history ends at t = "
        f"{source_times[result_idx]:.4f} s"
    )


    print(
        f"prediction is for t+1 = "
        f"{prediction_times[result_idx]:.4f} s"
    )


    print(
        f"prediction horizon = "
        f"{prediction_times[result_idx] - source_times[result_idx]:.4f} s"
    )


    print(
        "current high-level command U[t] =",
        np.round(
            U[original_t],
            5
        )
    )


    for (
        name,
        a,
        b
    ) in output_groups:

        p = predictions[
            result_idx,
            a:b
        ]

        y = actuals[
            result_idx,
            a:b
        ]

        error = (
            p - y
        )


        print(
            f"\n{name}"
        )

        print(
            " predicted :",
            np.round(
                p,
                5
            )
        )

        print(
            " actual    :",
            np.round(
                y,
                5
            )
        )

        print(
            " error     :",
            np.round(
                error,
                5
            )
        )


# ======================================================================
# 31. SAVE EVERYTHING
# ======================================================================

np.savez_compressed(

    "/content/one_step_tartanground_results.npz",

    time=time,

    P=P,

    B=B,

    U=U,

    predictions=predictions,

    actuals=actuals,

    persistence_predictions=persistence_predictions,

    source_times=source_times,

    prediction_times=prediction_times,

    train_losses=np.asarray(
        train_losses,
        dtype=np.float32
    ),

    test_step_losses=np.asarray(
        test_step_losses,
        dtype=np.float32
    )
)


torch.save(

    model.state_dict(),

    "/content/one_step_flat_mlp_model.pt"
)


print(
    "\n===================================================="
)

print(
    "DONE"
)

print(
    "===================================================="
)

print(
    "Saved results to:"
)

print(
    "/content/one_step_tartanground_results.npz"
)

print(
    "Saved model to:"
)

print(
    "/content/one_step_flat_mlp_model.pt"
)

# %% notebook cell 1
# ======================================================================
# VISUAL + PROPRIOCEPTION TERRAIN-CONDITIONED DYNAMICS MODEL
#
# ASSUMES THESE ALREADY EXIST:
#
# P    : (N, 4, 12)
#        per leg:
#        q3 + dq3 + effort3 + force_xyz3
#
# B    : (N, 12)
#        acc3 + gyro3 + body_twist6
#
# U    : (N, 6)
#        high-level twist command
#
# time : (N,)
#
# For your current preprocessing:
# time=0 corresponds to original trajectory ~4.04 s
#
#
# MODEL:
#
# lower-centre near-ground RGB-D
#        ↓
# DINOv2 semantic embedding
# ConvNeXt texture embedding
# Depth CNN geometry embedding
#        ↓
# visual Transformer fusion
#        ↓
# terrain token
#
# +
#
# 0.5 s proprio history
#        ↓
# causal Transformer
#
# +
#
# current U
#
#        ↓
# flat MLP coupling
#        ↓
# complete predicted next state t+1
#
# OUTPUT:
#
# P(t+1): 4 x 12 = 48
# B(t+1): 12
#
# total = 60 predicted values
# ======================================================================


# ======================================================================
# 1. INSTALL / IMPORT
# ======================================================================

# IPYTHON: !pip install -q transformers

import os
import gc
import math
import random

import numpy as np
import cv2

from PIL import Image

import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoImageProcessor,
    AutoModel
)

from torchvision.models import (
    convnext_tiny,
    ConvNeXt_Tiny_Weights
)


# ======================================================================
# 2. SETTINGS
# ======================================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


BASE = (
    "/content/tartanground/"
    "OldTownFall/"
    "Data_anymal/"
    "P2000"
)

RGB_DIR = os.path.join(
    BASE,
    "image_lcam_front"
)

DEPTH_DIR = os.path.join(
    BASE,
    "depth_lcam_front"
)

CAM_TIME_FILE = os.path.join(
    BASE,
    "imu",
    "cam_time.txt"
)


# Your previous preprocessing cropped the state trajectory
# beginning at original time 4.04 s and then reset it to zero.
STATE_START_IN_ORIGINAL_TIME = 4.04


# ------------------------------------------------------------
# "ground just ahead" heuristic ROI
#
# x: centre ~64% of image
# y: lower ~52% of image
#
# change these later after visual inspection.
# ------------------------------------------------------------

ROI_X_MIN = 0.18
ROI_X_MAX = 0.82

ROI_Y_MIN = 0.45
ROI_Y_MAX = 0.97


# Only retain visually nearby geometry.
#
# This prevents distant buildings / skyline from strongly
# affecting terrain embeddings.
DEPTH_MIN_METERS = 0.25
DEPTH_MAX_METERS = 4.0


DEPTH_SIZE = 64

HISTORY = 50

# chronological split
TRAIN_END_SEC = 90.0
VAL_END_SEC = 100.0
TEST_END_SEC = 120.0


BATCH_SIZE = 32
EPOCHS = 8

LR = 2e-4
WEIGHT_DECAY = 1e-4


# ======================================================================
# 3. MAKE SURE RGB + DEPTH EXIST
# ======================================================================

if (
    not os.path.exists(RGB_DIR)
    or
    not os.path.exists(DEPTH_DIR)
):

    print(
        "RGB/depth missing. Downloading them..."
    )

    import tartanair as ta

    ta.init(
        "/content/tartanground"
    )

    ta.download_ground(

        env=["OldTownFall"],

        version=["anymal"],

        traj=["P2000"],

        modality=[
            "image",
            "depth"
        ],

        camera_name=[
            "lcam_front"
        ],

        unzip=True
    )


assert os.path.isdir(RGB_DIR)
assert os.path.isdir(DEPTH_DIR)

print(
    "RGB/depth directories found."
)


# ======================================================================
# 4. FIND ALL MATCHED RGB + DEPTH FRAMES
# ======================================================================

rgb_ids = {

    f.split("_")[0]

    for f in os.listdir(RGB_DIR)

    if f.endswith(
        "_lcam_front.png"
    )
}


depth_ids = {

    f.split("_")[0]

    for f in os.listdir(DEPTH_DIR)

    if f.endswith(
        "_lcam_front_depth.png"
    )
}


frame_ids = sorted(
    rgb_ids
    & depth_ids,

    key=lambda x: int(x)
)


print(
    "\nRGB frames:",
    len(rgb_ids)
)

print(
    "Depth frames:",
    len(depth_ids)
)

print(
    "Matched RGB-D frames:",
    len(frame_ids)
)


assert len(frame_ids) > 0


# ======================================================================
# 5. CAMERA TIMESTAMPS
# ======================================================================

camera_time_original = np.loadtxt(
    CAM_TIME_FILE
).reshape(-1)


print(
    "\nCamera timestamps:",
    camera_time_original.shape
)

print(
    "Camera time range:",
    camera_time_original[0],
    "->",
    camera_time_original[-1]
)


# Frame ID i corresponds to camera timestamp i.
frame_numbers = np.array(
    [
        int(x)
        for x in frame_ids
    ],
    dtype=int
)


if frame_numbers.max() >= len(
    camera_time_original
):

    raise RuntimeError(
        "Camera frame IDs do not match "
        "cam_time.txt length."
    )


camera_times_original = (
    camera_time_original[
        frame_numbers
    ]
)


# convert to SAME zero as P/B/U/time
camera_times = (

    camera_times_original

    - STATE_START_IN_ORIGINAL_TIME
)


# only retain camera frames overlapping our state data

valid_camera = (

    (camera_times >= time[0])

    &

    (camera_times <= time[-1])
)


frame_numbers = frame_numbers[
    valid_camera
]

frame_ids = np.asarray(
    frame_ids
)[
    valid_camera
].tolist()

camera_times = camera_times[
    valid_camera
]


print(
    "\nUsable camera frames:",
    len(frame_ids)
)

print(
    "Aligned camera range:",
    camera_times[0],
    "->",
    camera_times[-1]
)


# ======================================================================
# 6. TARTANGROUND DEPTH DECODER
#
# Depth PNG = 4 uint8 bytes representing one little-endian
# float32 depth value in metres.
# ======================================================================

def load_metric_depth(
    path
):

    raw = cv2.imread(
        path,
        cv2.IMREAD_UNCHANGED
    )


    if raw is None:

        raise RuntimeError(
            f"Could not read {path}"
        )


    # TartanAir-V2 / TartanGround packed format

    if (

        raw.dtype == np.uint8

        and raw.ndim == 3

        and raw.shape[2] == 4

    ):

        raw = np.ascontiguousarray(
            raw
        )


        depth = raw.view(
            "<f4"
        ).reshape(
            raw.shape[0],
            raw.shape[1]
        )


    elif (
        raw.dtype == np.float32
        and raw.ndim == 2
    ):

        depth = raw


    else:

        raise RuntimeError(

            "Unexpected depth format: "

            f"dtype={raw.dtype}, "
            f"shape={raw.shape}"
        )


    depth = depth.astype(
        np.float32
    )


    depth[
        ~np.isfinite(depth)
    ] = 0.0


    return depth


# ======================================================================
# 7. LOAD ONE RGB-D FRAME AND TAKE ONLY NEAR-GROUND ROI
# ======================================================================

def load_ground_roi(
    frame_id
):

    rgb_path = os.path.join(

        RGB_DIR,

        f"{frame_id}_lcam_front.png"
    )


    depth_path = os.path.join(

        DEPTH_DIR,

        f"{frame_id}_lcam_front_depth.png"
    )


    # RGB
    rgb = np.asarray(
        Image.open(
            rgb_path
        ).convert(
            "RGB"
        )
    )


    # metric depth
    depth = load_metric_depth(
        depth_path
    )


    H, W = rgb.shape[:2]


    # --------------------------------------------------------------
    # lower-centre crop
    # --------------------------------------------------------------

    x0 = int(
        ROI_X_MIN * W
    )

    x1 = int(
        ROI_X_MAX * W
    )

    y0 = int(
        ROI_Y_MIN * H
    )

    y1 = int(
        ROI_Y_MAX * H
    )


    rgb_roi = rgb[
        y0:y1,
        x0:x1
    ].copy()


    depth_roi = depth[
        y0:y1,
        x0:x1
    ].copy()


    # --------------------------------------------------------------
    # near-ground / near-field mask
    #
    # removes distant scenery.
    # --------------------------------------------------------------

    valid_depth = (

        np.isfinite(
            depth_roi
        )

        &

        (
            depth_roi
            >= DEPTH_MIN_METERS
        )

        &

        (
            depth_roi
            <= DEPTH_MAX_METERS
        )
    )


    # RGB image actually sent to DINO/ConvNeXt.
    #
    # Far pixels become black.

    rgb_ground = rgb_roi.copy()


    rgb_ground[
        ~valid_depth
    ] = 0


    # depth input
    depth_ground = depth_roi.copy()


    depth_ground[
        ~valid_depth
    ] = 0.0


    return (

        rgb,

        rgb_ground,

        depth_ground,

        valid_depth
    )


# ======================================================================
# 8. VISUALIZE THE ROI BEFORE TRAINING
#
# IMPORTANT:
# Look at this plot.
# This tells us whether our crop actually captures terrain ahead.
# ======================================================================

example_id = frame_ids[
    len(frame_ids) // 2
]


(
    full_rgb,
    example_rgb_ground,
    example_depth,
    example_mask
) = load_ground_roi(
    example_id
)


plt.figure(
    figsize=(7, 7)
)

plt.imshow(
    full_rgb
)

plt.title(
    "Full front RGB"
)

plt.axis(
    "off"
)

plt.show()


plt.figure(
    figsize=(7, 5)
)

plt.imshow(
    example_rgb_ground
)

plt.title(
    "Near-ground RGB actually used by model"
)

plt.axis(
    "off"
)

plt.show()


plt.figure(
    figsize=(7, 5)
)

temp_depth = example_depth.copy()

temp_depth[
    temp_depth <= 0
] = np.nan


plt.imshow(
    temp_depth
)

plt.title(
    "Near-ground metric depth ROI"
)

plt.colorbar(
    label="metres"
)

plt.axis(
    "off"
)

plt.show()


print(
    "Valid near-depth pixels:",
    100.0 * example_mask.mean(),
    "%"
)


# ======================================================================
# 9. LOAD PRETRAINED VISUAL BACKBONES
#
# DINOv2:
# semantic / structure features
#
# ConvNeXt:
# local texture / appearance features
#
# Both are frozen initially.
# ======================================================================

print(
    "\nLoading DINOv2..."
)


DINO_NAME = (
    "facebook/dinov2-small"
)


dino_processor = (
    AutoImageProcessor.from_pretrained(
        DINO_NAME
    )
)


dino_model = (
    AutoModel.from_pretrained(
        DINO_NAME
    )
    .to(device)
)


dino_model.eval()


for p in dino_model.parameters():

    p.requires_grad = False


print(
    "DINO loaded."
)


print(
    "Loading ConvNeXt..."
)


conv_weights = (
    ConvNeXt_Tiny_Weights.DEFAULT
)


convnext = convnext_tiny(
    weights=conv_weights
).to(device)


# remove final classifier
convnext.classifier = nn.Identity()


convnext.eval()


for p in convnext.parameters():

    p.requires_grad = False


conv_transform = (
    conv_weights.transforms()
)


print(
    "ConvNeXt loaded."
)


# ======================================================================
# 10. PRECOMPUTE VISUAL FEATURES FOR EVERY CAMERA FRAME
#
# We DO THIS ONCE.
#
# DINO and ConvNeXt do not need to run every training epoch.
# ======================================================================

VIS_BATCH = 16


all_dino = []

all_conv = []

all_depth = []


print(
    "\nPrecomputing terrain visual features..."
)


for batch_start in range(
    0,
    len(frame_ids),
    VIS_BATCH
):


    batch_ids = frame_ids[

        batch_start:

        batch_start
        + VIS_BATCH
    ]


    pil_batch = []

    conv_batch = []

    depth_batch = []


    for frame_id in batch_ids:


        (
            _,
            rgb_ground,
            depth_ground,
            _
        ) = load_ground_roi(
            frame_id
        )


        pil_img = Image.fromarray(
            rgb_ground
        )


        pil_batch.append(
            pil_img
        )


        # ConvNeXt preprocessing

        conv_tensor = conv_transform(
            pil_img
        )


        conv_batch.append(
            conv_tensor
        )


        # ----------------------------------------------------------
        # depth to small 64x64 grid
        #
        # metric depth / MAX_DEPTH
        # ----------------------------------------------------------

        depth_small = cv2.resize(

            depth_ground,

            (
                DEPTH_SIZE,
                DEPTH_SIZE
            ),

            interpolation=cv2.INTER_AREA
        )


        depth_small = np.clip(

            depth_small,

            0.0,

            DEPTH_MAX_METERS
        )


        depth_small = (

            depth_small

            / DEPTH_MAX_METERS
        )


        depth_batch.append(

            depth_small[
                None,
                :,
                :
            ]
        )


    # --------------------------------------------------------------
    # DINO
    # --------------------------------------------------------------

    dino_inputs = dino_processor(

        images=pil_batch,

        return_tensors="pt"

    ).to(device)


    with torch.no_grad():


        dino_output = dino_model(
            **dino_inputs
        )


        # CLS token
        dino_feature = (

            dino_output
            .last_hidden_state[
                :,
                0,
                :
            ]
        )


    # --------------------------------------------------------------
    # ConvNeXt
    # --------------------------------------------------------------

    conv_tensor = torch.stack(
        conv_batch
    ).to(device)


    with torch.no_grad():

        conv_feature = convnext(
            conv_tensor
        )


    all_dino.append(

        dino_feature
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )


    all_conv.append(

        conv_feature
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )


    all_depth.append(

        np.asarray(
            depth_batch,
            dtype=np.float32
        )
    )


    print(

        f"\rVisual frames: "
        f"{min(batch_start + VIS_BATCH, len(frame_ids))}"
        f"/{len(frame_ids)}",

        end=""
    )


DINO_FEATURES = np.concatenate(
    all_dino,
    axis=0
)


CONV_FEATURES = np.concatenate(
    all_conv,
    axis=0
)


DEPTH_GRIDS = np.concatenate(
    all_depth,
    axis=0
)


print(
    "\n\nVisual features complete."
)


print(
    "DINO:",
    DINO_FEATURES.shape
)

print(
    "ConvNeXt:",
    CONV_FEATURES.shape
)

print(
    "Depth:",
    DEPTH_GRIDS.shape
)


# save because visual extraction is expensive

np.savez_compressed(

    "/content/"
    "P2000_ground_visual_features.npz",

    frame_numbers=frame_numbers,

    camera_times=camera_times,

    dino=DINO_FEATURES,

    convnext=CONV_FEATURES,

    depth=DEPTH_GRIDS
)


print(
    "Saved visual features."
)


# release frozen models from GPU

del dino_model
del convnext

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ======================================================================
# 11. CAUSALLY ASSIGN A CAMERA FRAME TO EVERY 100-Hz STATE SAMPLE
#
# At state time t:
#
# use LAST image whose camera timestamp <= t.
#
# NEVER use a future image.
# ======================================================================

visual_index_for_state = (

    np.searchsorted(

        camera_times,

        time,

        side="right"

    )

    - 1
)


visual_is_valid = (

    visual_index_for_state >= 0
)


print(
    "\nStates with causal RGB-D:",
    visual_is_valid.sum(),
    "/",
    len(time)
)


# ======================================================================
# 12. NORMALIZATION USING TRAIN REGION ONLY
# ======================================================================

train_state_mask = (

    (time < TRAIN_END_SEC)

    &

    visual_is_valid
)


train_P = P[
    train_state_mask
]


train_B = B[
    train_state_mask
]


train_U = U[
    train_state_mask
]


EPSILON = 1e-6


p_mean = train_P.mean(
    axis=(0, 1)
).astype(
    np.float32
)


p_std = (

    train_P.std(
        axis=(0, 1)
    )

    + EPSILON

).astype(
    np.float32
)


b_mean = train_B.mean(
    axis=0
).astype(
    np.float32
)


b_std = (

    train_B.std(
        axis=0
    )

    + EPSILON

).astype(
    np.float32
)


u_mean = train_U.mean(
    axis=0
).astype(
    np.float32
)


u_std = (

    train_U.std(
        axis=0
    )

    + EPSILON

).astype(
    np.float32
)


def norm_p(x):

    return (
        x - p_mean
    ) / p_std


def denorm_p(x):

    return (
        x * p_std
        + p_mean
    )


def norm_b(x):

    return (
        x - b_mean
    ) / b_std


def denorm_b(x):

    return (
        x * b_std
        + b_mean
    )


def norm_u(x):

    return (
        x - u_mean
    ) / u_std


# ======================================================================
# 13. SAMPLE INDICES
#
# Input:
# t-49 ... t
#
# Target:
# t+1
# ======================================================================

all_sample_t = []


for t in range(
    HISTORY - 1,
    len(P) - 1
):


    if not visual_is_valid[t]:

        continue


    all_sample_t.append(
        t
    )


all_sample_t = np.asarray(
    all_sample_t,
    dtype=int
)


train_indices = all_sample_t[

    time[
        all_sample_t
    ]

    < TRAIN_END_SEC
]


val_indices = all_sample_t[

    (
        time[
            all_sample_t
        ]

        >= TRAIN_END_SEC
    )

    &

    (
        time[
            all_sample_t
        ]

        < VAL_END_SEC
    )
]


test_indices = all_sample_t[

    (
        time[
            all_sample_t
        ]

        >= VAL_END_SEC
    )

    &

    (
        time[
            all_sample_t
        ]

        <= TEST_END_SEC
    )
]


print(
    "\nTrain examples:",
    len(train_indices)
)

print(
    "Validation examples:",
    len(val_indices)
)

print(
    "Test examples:",
    len(test_indices)
)


# ======================================================================
# 14. DATASET
# ======================================================================

class VisualProprioDataset(
    Dataset
):


    def __init__(
        self,
        indices
    ):

        self.indices = indices


    def __len__(
        self
    ):

        return len(
            self.indices
        )


    def __getitem__(
        self,
        index
    ):


        t = int(
            self.indices[
                index
            ]
        )


        # ----------------------------------------------------------
        # 0.5 s actual proprio history
        # ----------------------------------------------------------

        p_history = P[

            t-HISTORY+1:
            t+1

        ]


        # time x 4 x 12
        # -> 4 x time x 12

        p_history = np.transpose(

            p_history,

            (1, 0, 2)
        )


        b_history = B[

            t-HISTORY+1:
            t+1

        ]


        # ----------------------------------------------------------
        # command at current t
        # ----------------------------------------------------------

        u = U[
            t
        ]


        # ----------------------------------------------------------
        # causal image
        # ----------------------------------------------------------

        vi = int(

            visual_index_for_state[
                t
            ]
        )


        dino = DINO_FEATURES[
            vi
        ]


        conv = CONV_FEATURES[
            vi
        ]


        depth = DEPTH_GRIDS[
            vi
        ]


        # ----------------------------------------------------------
        # ACTUAL NEXT COMPLETE STATE
        # ----------------------------------------------------------

        target_p = P[
            t+1
        ]


        target_b = B[
            t+1
        ]


        # ----------------------------------------------------------
        # normalize
        # ----------------------------------------------------------

        p_history = norm_p(
            p_history
        )


        b_history = norm_b(
            b_history
        )


        u = norm_u(
            u
        )


        target_p = norm_p(
            target_p
        )


        target_b = norm_b(
            target_b
        )


        return {

            "p":
                torch.tensor(
                    p_history,
                    dtype=torch.float32
                ),

            "b":
                torch.tensor(
                    b_history,
                    dtype=torch.float32
                ),

            "u":
                torch.tensor(
                    u,
                    dtype=torch.float32
                ),

            "dino":
                torch.tensor(
                    dino,
                    dtype=torch.float32
                ),

            "conv":
                torch.tensor(
                    conv,
                    dtype=torch.float32
                ),

            "depth":
                torch.tensor(
                    depth,
                    dtype=torch.float32
                ),

            "target_p":
                torch.tensor(
                    target_p,
                    dtype=torch.float32
                ),

            "target_b":
                torch.tensor(
                    target_b,
                    dtype=torch.float32
                ),

            "t":
                t
        }


train_dataset = VisualProprioDataset(
    train_indices
)


val_dataset = VisualProprioDataset(
    val_indices
)


test_dataset = VisualProprioDataset(
    test_indices
)


train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=2
)


val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=2
)


test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=2
)


# ======================================================================
# 15. SHARED CAUSAL LEG TRANSFORMER
# ======================================================================

class SharedLegTransformer(
    nn.Module
):


    def __init__(
        self,
        d_model=128,
        nhead=4,
        layers=2
    ):

        super().__init__()


        self.d_model = d_model


        self.input_proj = nn.Linear(
            12,
            d_model
        )


        self.pos_embedding = nn.Parameter(

            torch.randn(
                1,
                HISTORY,
                d_model
            ) * 0.02
        )


        self.leg_embedding = nn.Embedding(
            4,
            d_model
        )


        layer = nn.TransformerEncoderLayer(

            d_model=d_model,

            nhead=nhead,

            dim_feedforward=256,

            dropout=0.1,

            batch_first=True,

            norm_first=True,

            activation="gelu"
        )


        self.encoder = nn.TransformerEncoder(

            layer,

            num_layers=layers
        )


        self.norm = nn.LayerNorm(
            d_model
        )


    def forward(
        self,
        x
    ):


        # B x 4 x 50 x 12

        batch, legs, T, features = (
            x.shape
        )


        x = x.reshape(

            batch * legs,

            T,

            features
        )


        x = self.input_proj(
            x
        )


        x = (

            x

            + self.pos_embedding[
                :,
                :T
            ]
        )


        leg_ids = torch.arange(

            legs,

            device=x.device

        ).repeat(
            batch
        )


        x = (

            x

            + self.leg_embedding(
                leg_ids
            ).unsqueeze(1)
        )


        causal_mask = torch.triu(

            torch.ones(

                T,
                T,

                dtype=torch.bool,

                device=x.device
            ),

            diagonal=1
        )


        x = self.encoder(

            x,

            mask=causal_mask
        )


        x = self.norm(
            x
        )


        x = x[
            :,
            -1,
            :
        ]


        return x.reshape(

            batch,

            4,

            self.d_model
        )


# ======================================================================
# 16. BODY TRANSFORMER
# ======================================================================

class BodyTransformer(
    nn.Module
):


    def __init__(
        self,
        d_model=128,
        nhead=4,
        layers=2
    ):

        super().__init__()


        self.input_proj = nn.Linear(
            12,
            d_model
        )


        self.pos_embedding = nn.Parameter(

            torch.randn(
                1,
                HISTORY,
                d_model
            ) * 0.02
        )


        layer = nn.TransformerEncoderLayer(

            d_model=d_model,

            nhead=nhead,

            dim_feedforward=256,

            dropout=0.1,

            batch_first=True,

            norm_first=True,

            activation="gelu"
        )


        self.encoder = nn.TransformerEncoder(

            layer,

            num_layers=layers
        )


        self.norm = nn.LayerNorm(
            d_model
        )


    def forward(
        self,
        x
    ):


        batch, T, features = (
            x.shape
        )


        x = self.input_proj(
            x
        )


        x = (

            x

            + self.pos_embedding[
                :,
                :T
            ]
        )


        mask = torch.triu(

            torch.ones(

                T,
                T,

                dtype=torch.bool,

                device=x.device
            ),

            diagonal=1
        )


        x = self.encoder(

            x,

            mask=mask
        )


        x = self.norm(
            x
        )


        return x[
            :,
            -1,
            :
        ]


# ======================================================================
# 17. DEPTH GEOMETRY ENCODER
#
# Trainable because depth already carries real metric geometry.
# ======================================================================

class DepthEncoder(
    nn.Module
):


    def __init__(
        self
    ):

        super().__init__()


        self.net = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=5,
                stride=2,
                padding=2
            ),

            nn.GELU(),


            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.GELU(),


            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.GELU(),


            nn.AdaptiveAvgPool2d(
                1
            )
        )


    def forward(
        self,
        x
    ):


        x = self.net(
            x
        )


        return x.flatten(
            1
        )


# ======================================================================
# 18. MULTIMODAL TERRAIN FUSION
#
# Three tokens:
#
# 1. DINO semantic token
# 2. ConvNeXt texture token
# 3. metric-depth geometry token
#
# Transformer allows modalities to interact.
# ======================================================================

class VisualTerrainFusion(
    nn.Module
):


    def __init__(
        self,
        dino_dim,
        conv_dim,
        d_model=256
    ):

        super().__init__()


        self.depth_encoder = (
            DepthEncoder()
        )


        self.dino_proj = nn.Linear(
            dino_dim,
            d_model
        )


        self.conv_proj = nn.Linear(
            conv_dim,
            d_model
        )


        self.depth_proj = nn.Linear(
            128,
            d_model
        )


        self.type_embedding = nn.Parameter(

            torch.randn(
                1,
                3,
                d_model
            ) * 0.02
        )


        layer = nn.TransformerEncoderLayer(

            d_model=d_model,

            nhead=4,

            dim_feedforward=512,

            dropout=0.1,

            batch_first=True,

            norm_first=True,

            activation="gelu"
        )


        self.fusion = nn.TransformerEncoder(

            layer,

            num_layers=2
        )


        self.norm = nn.LayerNorm(
            d_model
        )


    def forward(
        self,
        dino,
        conv,
        depth
    ):


        dino_token = self.dino_proj(
            dino
        )


        conv_token = self.conv_proj(
            conv
        )


        depth_feature = self.depth_encoder(
            depth
        )


        depth_token = self.depth_proj(
            depth_feature
        )


        tokens = torch.stack(

            [
                dino_token,
                conv_token,
                depth_token
            ],

            dim=1
        )


        tokens = (
            tokens
            + self.type_embedding
        )


        tokens = self.fusion(
            tokens
        )


        tokens = self.norm(
            tokens
        )


        # one terrain representation

        terrain = tokens.mean(
            dim=1
        )


        return terrain


# ======================================================================
# 19. COMPLETE VISUAL-PROPRIO MODEL
# ======================================================================

class VisualProprioDynamics(
    nn.Module
):


    def __init__(
        self,
        dino_dim,
        conv_dim
    ):

        super().__init__()


        self.leg_transformer = (
            SharedLegTransformer(
                d_model=128
            )
        )


        self.body_transformer = (
            BodyTransformer(
                d_model=128
            )
        )


        self.visual = (
            VisualTerrainFusion(

                dino_dim=dino_dim,

                conv_dim=conv_dim,

                d_model=256
            )
        )


        self.command_encoder = nn.Sequential(

            nn.Linear(
                6,
                64
            ),

            nn.GELU(),

            nn.Linear(
                64,
                64
            )
        )


        # --------------------------------------------------------------
        # Flat coupling
        #
        # four legs:
        # 4*128 = 512
        #
        # body:
        # 128
        #
        # terrain:
        # 256
        #
        # command:
        # 64
        #
        # total = 960
        # --------------------------------------------------------------

        fusion_input = (

            4*128

            + 128

            + 256

            + 64
        )


        self.coupling = nn.Sequential(

            nn.LayerNorm(
                fusion_input
            ),

            nn.Linear(
                fusion_input,
                512
            ),

            nn.GELU(),

            nn.Dropout(
                0.1
            ),

            nn.Linear(
                512,
                256
            ),

            nn.GELU()
        )


        # predict complete next-state delta

        self.output = nn.Linear(
            256,
            60
        )


    def forward(
        self,
        p_history,
        b_history,
        u,
        dino,
        conv,
        depth
    ):


        # --------------------------------------------------------------
        # proprio temporal representation
        # --------------------------------------------------------------

        leg_h = self.leg_transformer(
            p_history
        )


        body_h = self.body_transformer(
            b_history
        )


        # --------------------------------------------------------------
        # terrain visual representation
        # --------------------------------------------------------------

        terrain_h = self.visual(

            dino,

            conv,

            depth
        )


        # --------------------------------------------------------------
        # action representation
        # --------------------------------------------------------------

        command_h = self.command_encoder(
            u
        )


        # --------------------------------------------------------------
        # flat body/leg/terrain coupling
        # --------------------------------------------------------------

        x = torch.cat(

            [
                leg_h.flatten(
                    1
                ),

                body_h,

                terrain_h,

                command_h
            ],

            dim=1
        )


        x = self.coupling(
            x
        )


        delta = self.output(
            x
        )


        # --------------------------------------------------------------
        # Residual dynamics:
        #
        # next state =
        # current state + predicted change
        # --------------------------------------------------------------

        current_p = p_history[
            :,
            :,
            -1,
            :
        ]


        current_b = b_history[
            :,
            -1,
            :
        ]


        current_state = torch.cat(

            [
                current_p.reshape(
                    len(current_p),
                    48
                ),

                current_b
            ],

            dim=1
        )


        predicted_state = (

            current_state

            + delta
        )


        predicted_p = predicted_state[
            :,
            :48
        ].reshape(
            -1,
            4,
            12
        )


        predicted_b = predicted_state[
            :,
            48:60
        ]


        return (

            predicted_p,

            predicted_b
        )


# ======================================================================
# 20. CREATE MODEL
# ======================================================================

DINO_DIM = (
    DINO_FEATURES.shape[1]
)


CONV_DIM = (
    CONV_FEATURES.shape[1]
)


model = VisualProprioDynamics(

    dino_dim=DINO_DIM,

    conv_dim=CONV_DIM

).to(device)


print(
    "\nTrainable parameters:",
    f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
)


optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LR,

    weight_decay=WEIGHT_DECAY
)


mse = nn.MSELoss()


# ======================================================================
# 21. HELPER: MOVE BATCH TO GPU
# ======================================================================

def move_batch(
    batch
):


    return {

        key:
            value.to(device)

        if torch.is_tensor(
            value
        )

        else value

        for key, value in batch.items()
    }


# ======================================================================
# 22. TRAIN
# ======================================================================

train_loss_history = []

val_loss_history = []


best_val = float(
    "inf"
)


best_state = None


for epoch in range(
    1,
    EPOCHS + 1
):


    # --------------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------------

    model.train()


    epoch_losses = []


    for batch in train_loader:


        batch = move_batch(
            batch
        )


        optimizer.zero_grad()


        pred_p, pred_b = model(

            batch["p"],

            batch["b"],

            batch["u"],

            batch["dino"],

            batch["conv"],

            batch["depth"]
        )


        p_loss = mse(

            pred_p,

            batch[
                "target_p"
            ]
        )


        b_loss = mse(

            pred_b,

            batch[
                "target_b"
            ]
        )


        loss = (

            0.5 * p_loss

            +

            0.5 * b_loss
        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            max_norm=1.0
        )


        optimizer.step()


        epoch_losses.append(

            float(
                loss.item()
            )
        )


    train_loss = float(

        np.mean(
            epoch_losses
        )
    )


    # --------------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------------

    model.eval()


    val_losses = []


    with torch.no_grad():


        for batch in val_loader:


            batch = move_batch(
                batch
            )


            pred_p, pred_b = model(

                batch["p"],

                batch["b"],

                batch["u"],

                batch["dino"],

                batch["conv"],

                batch["depth"]
            )


            p_loss = mse(

                pred_p,

                batch[
                    "target_p"
                ]
            )


            b_loss = mse(

                pred_b,

                batch[
                    "target_b"
                ]
            )


            loss = (

                0.5*p_loss

                +

                0.5*b_loss
            )


            val_losses.append(

                float(
                    loss.item()
                )
            )


    val_loss = float(

        np.mean(
            val_losses
        )
    )


    train_loss_history.append(
        train_loss
    )


    val_loss_history.append(
        val_loss
    )


    print(

        f"Epoch {epoch:02d}"

        f" | train {train_loss:.6f}"

        f" | val {val_loss:.6f}"
    )


    if val_loss < best_val:


        best_val = val_loss


        best_state = {

            k:
                v.detach()
                .cpu()
                .clone()

            for k, v
            in model.state_dict().items()
        }


# restore best model

model.load_state_dict(
    best_state
)


print(
    "\nBest validation loss:",
    best_val
)


# ======================================================================
# 23. TEST
# ======================================================================

model.eval()


pred_p_all = []

pred_b_all = []

actual_p_all = []

actual_b_all = []

test_t_all = []


with torch.no_grad():


    for batch in test_loader:


        batch = move_batch(
            batch
        )


        pred_p, pred_b = model(

            batch["p"],

            batch["b"],

            batch["u"],

            batch["dino"],

            batch["conv"],

            batch["depth"]
        )


        # denormalize

        pred_p_np = denorm_p(

            pred_p
            .cpu()
            .numpy()
        )


        pred_b_np = denorm_b(

            pred_b
            .cpu()
            .numpy()
        )


        actual_p_np = denorm_p(

            batch[
                "target_p"
            ]
            .cpu()
            .numpy()
        )


        actual_b_np = denorm_b(

            batch[
                "target_b"
            ]
            .cpu()
            .numpy()
        )


        pred_p_all.append(
            pred_p_np
        )


        pred_b_all.append(
            pred_b_np
        )


        actual_p_all.append(
            actual_p_np
        )


        actual_b_all.append(
            actual_b_np
        )


        test_t_all.append(

            batch[
                "t"
            ]
            .cpu()
            .numpy()
        )


pred_p_all = np.concatenate(
    pred_p_all
)


pred_b_all = np.concatenate(
    pred_b_all
)


actual_p_all = np.concatenate(
    actual_p_all
)


actual_b_all = np.concatenate(
    actual_b_all
)


test_t_all = np.concatenate(
    test_t_all
)


# ======================================================================
# 24. REPORT CONTACT RESPONSE
# ======================================================================

def RMSE(
    pred,
    actual
):

    return float(

        np.sqrt(

            np.mean(

                (
                    pred
                    - actual
                ) ** 2
            )
        )
    )


legs = [
    "FL",
    "FR",
    "RL",
    "RR"
]


print(
    "\n"
    + "="*80
)

print(
    "TEST CONTACT-RESPONSE RESULTS"
)

print(
    "="*80
)


for leg in range(4):


    pred_force = pred_p_all[
        :,
        leg,
        9:12
    ]


    actual_force = actual_p_all[
        :,
        leg,
        9:12
    ]


    print(

        f"{legs[leg]} force RMSE: "

        f"{RMSE(pred_force, actual_force):.6f}"
    )


print(

    "Body linear velocity RMSE:",

    RMSE(

        pred_b_all[
            :,
            6:9
        ],

        actual_b_all[
            :,
            6:9
        ]
    )
)


print(

    "Body angular velocity RMSE:",

    RMSE(

        pred_b_all[
            :,
            9:12
        ],

        actual_b_all[
            :,
            9:12
        ]
    )
)


# ======================================================================
# 25. PLOT ACTUAL VS PREDICTED FOOT FORCE MAGNITUDE
# ======================================================================

plot_time = time[
    test_t_all + 1
]


for leg in range(4):


    pred_mag = np.linalg.norm(

        pred_p_all[
            :,
            leg,
            9:12
        ],

        axis=1
    )


    actual_mag = np.linalg.norm(

        actual_p_all[
            :,
            leg,
            9:12
        ],

        axis=1
    )


    plt.figure(
        figsize=(11, 4)
    )


    plt.plot(

        plot_time,

        actual_mag,

        label="Actual"
    )


    plt.plot(

        plot_time,

        pred_mag,

        label="Predicted"
    )


    plt.title(

        f"{legs[leg]} contact force: "
        f"visual + proprio prediction"
    )


    plt.xlabel(
        "Time (s)"
    )


    plt.ylabel(
        "Force magnitude"
    )


    plt.legend()

    plt.grid()

    plt.show()


# ======================================================================
# 26. SHOW FOUR INDIVIDUAL TEST EXAMPLES
# ======================================================================

requested_examples = [
    102.0,
    108.0,
    114.0,
    119.0
]


print(
    "\n"
    + "="*80
)

print(
    "INDIVIDUAL VISUAL-PROPRIO TEST EXAMPLES"
)

print(
    "="*80
)


for requested_time in requested_examples:


    j = int(

        np.argmin(

            np.abs(

                plot_time
                - requested_time
            )
        )
    )


    print(
        "\nTarget time:",
        plot_time[j],
        "s"
    )


    for leg in range(4):


        pred = pred_p_all[
            j,
            leg,
            9:12
        ]


        actual = actual_p_all[
            j,
            leg,
            9:12
        ]


        print(
            f"\n{legs[leg]} force"
        )


        print(
            "predicted:",
            np.round(
                pred,
                4
            )
        )


        print(
            "actual   :",
            np.round(
                actual,
                4
            )
        )


    print(
        "\nBody twist"
    )


    print(
        "predicted:",
        np.round(
            pred_b_all[
                j,
                6:12
            ],
            4
        )
    )


    print(
        "actual   :",
        np.round(
            actual_b_all[
                j,
                6:12
            ],
            4
        )
    )


# ======================================================================
# 27. SAVE MODEL
# ======================================================================

torch.save(

    {

        "model_state":
            model.state_dict(),

        "p_mean":
            p_mean,

        "p_std":
            p_std,

        "b_mean":
            b_mean,

        "b_std":
            b_std,

        "u_mean":
            u_mean,

        "u_std":
            u_std,

        "roi":
            (
                ROI_X_MIN,
                ROI_X_MAX,
                ROI_Y_MIN,
                ROI_Y_MAX
            ),

        "depth_range":
            (
                DEPTH_MIN_METERS,
                DEPTH_MAX_METERS
            ),

    },

    "/content/"
    "visual_proprio_terrain_model.pt"
)


print(
    "\nSaved:"
)

print(
    "/content/"
    "visual_proprio_terrain_model.pt"
)

# %% notebook cell 2
# ================================================================
# CELL 2 — FIX CONVNEXT FEATURES
# ================================================================

print("Before:", CONV_FEATURES.shape)

# (1251, 768, 1, 1) -> (1251, 768)
CONV_FEATURES = CONV_FEATURES.reshape(
    CONV_FEATURES.shape[0],
    -1
).astype(np.float32)

print("After :", CONV_FEATURES.shape)

assert CONV_FEATURES.shape[1] == 768

# overwrite cached file with corrected features
np.savez_compressed(
    "/content/P2000_ground_visual_features.npz",
    frame_numbers=frame_numbers,
    camera_times=camera_times,
    dino=DINO_FEATURES,
    convnext=CONV_FEATURES,
    depth=DEPTH_GRIDS
)

print("ConvNeXt feature shape fixed.")

# %% notebook cell 3
# ================================================================
# CELL 3 — RECREATE LOADERS + FRESH MODEL
# ================================================================

from torch.utils.data import DataLoader


train_dataset = VisualProprioDataset(
    train_indices
)

val_dataset = VisualProprioDataset(
    val_indices
)

test_dataset = VisualProprioDataset(
    test_indices
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)


DINO_DIM = DINO_FEATURES.shape[1]
CONV_DIM = CONV_FEATURES.shape[1]

print("DINO dimension:", DINO_DIM)
print("Conv dimension:", CONV_DIM)


# fresh model
model = VisualProprioDynamics(
    dino_dim=DINO_DIM,
    conv_dim=CONV_DIM
).to(device)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)


mse = nn.MSELoss()


print(
    "Trainable parameters:",
    f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
)

print("Fresh model created.")

# %% notebook cell 4
# ================================================================
# CELL 4 — DIMENSION / FORWARD-PASS SANITY CHECK
# ================================================================

batch = next(
    iter(train_loader)
)

batch = move_batch(
    batch
)


print("INPUT SHAPES")
print("P history :", batch["p"].shape)
print("B history :", batch["b"].shape)
print("U         :", batch["u"].shape)
print("DINO      :", batch["dino"].shape)
print("ConvNeXt  :", batch["conv"].shape)
print("Depth     :", batch["depth"].shape)


model.eval()

with torch.no_grad():

    pred_p_test, pred_b_test = model(

        batch["p"],
        batch["b"],
        batch["u"],
        batch["dino"],
        batch["conv"],
        batch["depth"]
    )


print("\nOUTPUT SHAPES")
print("Predicted P:", pred_p_test.shape)
print("Predicted B:", pred_b_test.shape)


assert pred_p_test.shape == (
    batch["p"].shape[0],
    4,
    12
)

assert pred_b_test.shape == (
    batch["p"].shape[0],
    12
)


print("\nEverything is dimensionally correct.")

# %% notebook cell 5
# ================================================================
# CELL 5 — TRAIN VISUAL + PROPRIO MODEL
# ================================================================

train_loss_history = []
val_loss_history = []

best_val = float("inf")
best_state = None


for epoch in range(
    1,
    EPOCHS + 1
):

    # ============================================================
    # TRAIN
    # ============================================================

    model.train()

    epoch_losses = []


    for batch_number, batch in enumerate(
        train_loader,
        start=1
    ):

        batch = move_batch(
            batch
        )

        optimizer.zero_grad()


        pred_p, pred_b = model(

            batch["p"],
            batch["b"],
            batch["u"],
            batch["dino"],
            batch["conv"],
            batch["depth"]
        )


        p_loss = mse(
            pred_p,
            batch["target_p"]
        )


        b_loss = mse(
            pred_b,
            batch["target_b"]
        )


        loss = (
            0.5 * p_loss
            +
            0.5 * b_loss
        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )


        optimizer.step()


        epoch_losses.append(
            float(loss.item())
        )


        if batch_number % 100 == 0:

            print(
                f"Epoch {epoch:02d} "
                f"| batch {batch_number:03d}/{len(train_loader)} "
                f"| loss {loss.item():.6f}"
            )


    train_loss = float(
        np.mean(epoch_losses)
    )


    # ============================================================
    # VALIDATION
    # ============================================================

    model.eval()

    validation_losses = []


    with torch.no_grad():

        for batch in val_loader:

            batch = move_batch(
                batch
            )


            pred_p, pred_b = model(

                batch["p"],
                batch["b"],
                batch["u"],
                batch["dino"],
                batch["conv"],
                batch["depth"]
            )


            p_loss = mse(
                pred_p,
                batch["target_p"]
            )


            b_loss = mse(
                pred_b,
                batch["target_b"]
            )


            loss = (
                0.5 * p_loss
                +
                0.5 * b_loss
            )


            validation_losses.append(
                float(loss.item())
            )


    val_loss = float(
        np.mean(validation_losses)
    )


    train_loss_history.append(
        train_loss
    )

    val_loss_history.append(
        val_loss
    )


    print(
        f"\nEpoch {epoch:02d}/{EPOCHS}"
        f" | Train = {train_loss:.6f}"
        f" | Val = {val_loss:.6f}\n"
    )


    # save best validation model

    if val_loss < best_val:

        best_val = val_loss

        best_state = {
            key: value.detach().cpu().clone()
            for key, value
            in model.state_dict().items()
        }


# restore best model
model.load_state_dict(
    best_state
)


print(
    "\nTraining complete."
)

print(
    "Best validation loss:",
    best_val
)

# %% notebook cell 6
# ================================================================
# CELL 6 — TRAIN / VALIDATION LOSS GRAPH
# ================================================================

epochs_axis = np.arange(
    1,
    len(train_loss_history) + 1
)


plt.figure(
    figsize=(8, 5)
)


plt.plot(
    epochs_axis,
    train_loss_history,
    marker="o",
    label="Training loss"
)


plt.plot(
    epochs_axis,
    val_loss_history,
    marker="o",
    label="Validation loss"
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Normalized MSE"
)

plt.title(
    "Visual-Proprio Model Learning Curve"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.show()

# %% notebook cell 7
# ================================================================
# CELL 7 — TEST SET PREDICTIONS
# ================================================================

model.eval()


pred_p_all = []
pred_b_all = []

actual_p_all = []
actual_b_all = []

test_t_all = []


with torch.no_grad():

    for batch in test_loader:

        batch = move_batch(
            batch
        )


        pred_p, pred_b = model(

            batch["p"],
            batch["b"],
            batch["u"],
            batch["dino"],
            batch["conv"],
            batch["depth"]
        )


        # ----------------------------------------------------------
        # DENORMALIZE PREDICTIONS
        # ----------------------------------------------------------

        pred_p_np = denorm_p(

            pred_p
            .cpu()
            .numpy()
        )


        pred_b_np = denorm_b(

            pred_b
            .cpu()
            .numpy()
        )


        # ----------------------------------------------------------
        # DENORMALIZE GROUND TRUTH
        # ----------------------------------------------------------

        actual_p_np = denorm_p(

            batch["target_p"]
            .cpu()
            .numpy()
        )


        actual_b_np = denorm_b(

            batch["target_b"]
            .cpu()
            .numpy()
        )


        pred_p_all.append(
            pred_p_np
        )

        pred_b_all.append(
            pred_b_np
        )

        actual_p_all.append(
            actual_p_np
        )

        actual_b_all.append(
            actual_b_np
        )


        test_t_all.append(

            batch["t"]
            .cpu()
            .numpy()
        )


pred_p_all = np.concatenate(
    pred_p_all,
    axis=0
)

pred_b_all = np.concatenate(
    pred_b_all,
    axis=0
)

actual_p_all = np.concatenate(
    actual_p_all,
    axis=0
)

actual_b_all = np.concatenate(
    actual_b_all,
    axis=0
)

test_t_all = np.concatenate(
    test_t_all,
    axis=0
)


print("TEST OUTPUTS")

print(
    "Pred P:",
    pred_p_all.shape
)

print(
    "Actual P:",
    actual_p_all.shape
)

print(
    "Pred B:",
    pred_b_all.shape
)

print(
    "Actual B:",
    actual_b_all.shape
)

# %% notebook cell 8
# ================================================================
# CELL 8 — MODEL VS PERSISTENCE BASELINE
# ================================================================

def rmse(
    prediction,
    actual
):

    return float(
        np.sqrt(
            np.mean(
                (
                    prediction
                    - actual
                ) ** 2
            )
        )
    )


LEG_NAMES = [
    "FL",
    "FR",
    "RL",
    "RR"
]


print(
    "\n"
    + "=" * 75
)

print(
    "FOOT FORCE TEST RESULTS"
)

print(
    "=" * 75
)


model_force_rmse = []
persistence_force_rmse = []


# Current measured state at t
# is the persistence prediction for t+1.

persistence_p = P[
    test_t_all
]


for leg in range(
    4
):


    model_force = pred_p_all[
        :,
        leg,
        9:12
    ]


    actual_force = actual_p_all[
        :,
        leg,
        9:12
    ]


    persistence_force = persistence_p[
        :,
        leg,
        9:12
    ]


    model_error = rmse(
        model_force,
        actual_force
    )


    persistence_error = rmse(
        persistence_force,
        actual_force
    )


    model_force_rmse.append(
        model_error
    )


    persistence_force_rmse.append(
        persistence_error
    )


    improvement = (

        100

        *

        (
            persistence_error
            - model_error
        )

        /

        persistence_error
    )


    print(
        f"\n{LEG_NAMES[leg]}"
    )

    print(
        "Model RMSE       :",
        round(model_error, 5)
    )

    print(
        "Persistence RMSE :",
        round(persistence_error, 5)
    )

    print(
        "Improvement      :",
        round(improvement, 2),
        "%"
    )

# %% notebook cell 9
# ================================================================
# CELL 9 — PREDICTED VS ACTUAL CONTACT FORCES
# ================================================================

target_times = time[
    test_t_all + 1
]


for leg in range(
    4
):


    predicted_magnitude = np.linalg.norm(

        pred_p_all[
            :,
            leg,
            9:12
        ],

        axis=1
    )


    actual_magnitude = np.linalg.norm(

        actual_p_all[
            :,
            leg,
            9:12
        ],

        axis=1
    )


    persistence_magnitude = np.linalg.norm(

        persistence_p[
            :,
            leg,
            9:12
        ],

        axis=1
    )


    plt.figure(
        figsize=(13, 4)
    )


    plt.plot(
        target_times,
        actual_magnitude,
        label="Ground truth"
    )


    plt.plot(
        target_times,
        predicted_magnitude,
        label="Visual + proprio model"
    )


    plt.plot(
        target_times,
        persistence_magnitude,
        label="Persistence",
        alpha=0.55
    )


    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Contact force magnitude"
    )


    plt.title(
        f"{LEG_NAMES[leg]} foot force"
    )


    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.show()

# %% notebook cell 10
# ================================================================
# CELL 10 — FOUR SPECIFIC UNSEEN TEST CASES
# ================================================================

CASE_TIMES = [
    102.0,
    108.0,
    114.0,
    119.0
]


for requested_time in CASE_TIMES:


    j = int(

        np.argmin(

            np.abs(
                target_times
                - requested_time
            )
        )
    )


    print(
        "\n\n"
        + "=" * 80
    )

    print(
        "TEST TARGET:",
        target_times[j],
        "seconds"
    )

    print(
        "=" * 80
    )


    predicted_force_magnitudes = []

    actual_force_magnitudes = []

    persistence_force_magnitudes = []


    for leg in range(
        4
    ):


        predicted_force = pred_p_all[
            j,
            leg,
            9:12
        ]


        actual_force = actual_p_all[
            j,
            leg,
            9:12
        ]


        current_force = persistence_p[
            j,
            leg,
            9:12
        ]


        print(
            f"\n{LEG_NAMES[leg]} FORCE"
        )


        print(
            "Predicted:",
            np.round(
                predicted_force,
                4
            )
        )


        print(
            "Actual   :",
            np.round(
                actual_force,
                4
            )
        )


        print(
            "Current  :",
            np.round(
                current_force,
                4
            )
        )


        predicted_force_magnitudes.append(

            np.linalg.norm(
                predicted_force
            )
        )


        actual_force_magnitudes.append(

            np.linalg.norm(
                actual_force
            )
        )


        persistence_force_magnitudes.append(

            np.linalg.norm(
                current_force
            )
        )


    # ------------------------------------------------------------
    # BAR GRAPH
    # ------------------------------------------------------------

    x = np.arange(
        4
    )

    width = 0.25


    plt.figure(
        figsize=(9, 5)
    )


    plt.bar(

        x - width,

        actual_force_magnitudes,

        width,

        label="Ground truth"
    )


    plt.bar(

        x,

        predicted_force_magnitudes,

        width,

        label="Prediction"
    )


    plt.bar(

        x + width,

        persistence_force_magnitudes,

        width,

        label="Persistence"
    )


    plt.xticks(
        x,
        LEG_NAMES
    )


    plt.ylabel(
        "Force magnitude"
    )


    plt.title(

        f"Prediction at "
        f"{target_times[j]:.2f} s"
    )


    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.show()

# %% notebook cell 11


# %% notebook cell 12
# ======================================================================
# FOUR TEST CASES:
#
# A) ONE-STEP t -> t+1
# B) RECURSIVE t -> t+10 = t+0.1 s
#
# Reports ALL 48 VALUES OF P:
#
# For EACH leg:
# q1 q2 q3
# dq1 dq2 dq3
# effort1 effort2 effort3
# Fx Fy Fz
#
# B is also recursively predicted internally because the next
# Transformer input requires body history.
# ======================================================================

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


# ======================================================================
# 1. SETTINGS
# ======================================================================

TEST_START_TIMES = [
    102.0,
    108.0,
    114.0,
    119.0
]

LEG_NAMES = [
    "FL",
    "FR",
    "RL",
    "RR"
]

P_VARIABLES = [
    "q1",
    "q2",
    "q3",

    "dq1",
    "dq2",
    "dq3",

    "effort1",
    "effort2",
    "effort3",

    "Fx",
    "Fy",
    "Fz"
]


SENSOR_GROUPS = {

    "Joint angle": (
        slice(0, 3),
        ["q1", "q2", "q3"]
    ),

    "Joint velocity": (
        slice(3, 6),
        ["dq1", "dq2", "dq3"]
    ),

    "Joint effort": (
        slice(6, 9),
        ["e1", "e2", "e3"]
    ),

    "Foot force": (
        slice(9, 12),
        ["Fx", "Fy", "Fz"]
    )
}


dt = float(
    np.median(
        np.diff(time)
    )
)


ROLLOUT_STEPS = int(
    round(
        0.1 / dt
    )
)


print(
    "dt:",
    dt
)

print(
    "Rollout steps:",
    ROLLOUT_STEPS
)

print(
    "Actual horizon:",
    ROLLOUT_STEPS * dt,
    "seconds"
)


assert ROLLOUT_STEPS == 10


# ======================================================================
# 2. GET VISUAL INPUT AT A PARTICULAR ROBOT STATE TIME
#
# IMPORTANT:
#
# visual_index_for_state was constructed causally:
#
# use latest camera image whose timestamp <= robot state time.
#
# Since camera is slower than 100 Hz, several consecutive robot
# states can legitimately use the same RGB-D frame.
# ======================================================================

def get_visual_input(
    state_idx
):

    visual_idx = int(
        visual_index_for_state[
            state_idx
        ]
    )


    if visual_idx < 0:

        raise RuntimeError(
            f"No causal visual frame for "
            f"state index {state_idx}"
        )


    # --------------------------------------------------------------
    # DINO
    # --------------------------------------------------------------

    dino = np.asarray(
        DINO_FEATURES[
            visual_idx
        ],
        dtype=np.float32
    ).reshape(-1)


    # --------------------------------------------------------------
    # ConvNeXt
    #
    # Robust even if old cached features are still:
    # 768 x 1 x 1
    # --------------------------------------------------------------

    conv = np.asarray(
        CONV_FEATURES[
            visual_idx
        ],
        dtype=np.float32
    ).reshape(-1)


    # --------------------------------------------------------------
    # Depth
    #
    # 1 x 64 x 64
    # --------------------------------------------------------------

    depth = np.asarray(
        DEPTH_GRIDS[
            visual_idx
        ],
        dtype=np.float32
    )


    dino_t = torch.tensor(
        dino,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)


    conv_t = torch.tensor(
        conv,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)


    depth_t = torch.tensor(
        depth,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)


    return (
        dino_t,
        conv_t,
        depth_t,
        visual_idx
    )


# ======================================================================
# 3. BUILD REAL INITIAL 0.5 SECOND HISTORY
#
# Input ending at t:
#
# P[t-49:t] -> 4 x 50 x 12
# B[t-49:t] -> 50 x 12
# ======================================================================

def make_initial_history(
    t
):

    p_history = P[
        t-HISTORY+1:
        t+1
    ]

    # 50 x 4 x 12
    # ->
    # 4 x 50 x 12

    p_history = np.transpose(
        p_history,
        (1, 0, 2)
    )


    b_history = B[
        t-HISTORY+1:
        t+1
    ]


    # normalized because model was trained
    # using normalized states

    p_history = norm_p(
        p_history
    )


    b_history = norm_b(
        b_history
    )


    p_history_t = torch.tensor(
        p_history,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)


    b_history_t = torch.tensor(
        b_history,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)


    return (
        p_history_t,
        b_history_t
    )


# ======================================================================
# 4. COMPLETE RECURSIVE ROLLOUT
#
# Step k:
#
# predicted state t+k
# +
# visual embedding belonging to time t+k
# +
# recorded U[t+k]
#
# -> predicted state t+k+1
#
#
# STATE is recursive.
#
# VISUAL and U are supplied from recorded trajectory.
# ======================================================================

def recursive_visual_proprio_rollout(
    start_t,
    steps=10
):

    model.eval()


    rolling_p, rolling_b = (
        make_initial_history(
            start_t
        )
    )


    predicted_p = []

    predicted_b = []

    used_visual_indices = []

    used_commands = []


    with torch.no_grad():


        for k in range(
            steps
        ):


            current_idx = (
                start_t
                + k
            )


            # ======================================================
            # U[t+k]
            #
            # transition:
            #
            # state(t+k)
            # ->
            # state(t+k+1)
            # ======================================================

            command = norm_u(
                U[
                    current_idx
                ]
            )


            command_t = torch.tensor(
                command,
                dtype=torch.float32,
                device=device
            ).unsqueeze(0)


            # ======================================================
            # RGB-D embedding at time t+k
            #
            # k=0:
            # image at t
            #
            # k=1:
            # image at t+1
            #
            # ...
            # ======================================================

            (
                dino_t,
                conv_t,
                depth_t,
                visual_idx
            ) = get_visual_input(
                current_idx
            )


            # ======================================================
            # PREDICT NEXT COMPLETE STATE
            # ======================================================

            next_p_n, next_b_n = model(

                rolling_p,
                rolling_b,

                command_t,

                dino_t,
                conv_t,
                depth_t
            )


            # ======================================================
            # SAVE PHYSICAL-UNIT PREDICTIONS
            # ======================================================

            next_p_physical = denorm_p(

                next_p_n[
                    0
                ]
                .cpu()
                .numpy()
            )


            next_b_physical = denorm_b(

                next_b_n[
                    0
                ]
                .cpu()
                .numpy()
            )


            predicted_p.append(
                next_p_physical
            )


            predicted_b.append(
                next_b_physical
            )


            used_visual_indices.append(
                visual_idx
            )


            used_commands.append(
                U[
                    current_idx
                ].copy()
            )


            # ======================================================
            # RECURSIVE UPDATE
            #
            # Remove oldest state.
            #
            # Append MODEL PREDICTION.
            #
            # IMPORTANT:
            # actual P/B at t+k+1 is NOT inserted.
            # ======================================================

            rolling_p = torch.cat(

                [

                    rolling_p[
                        :,
                        :,
                        1:,
                        :
                    ],

                    next_p_n.unsqueeze(
                        2
                    )

                ],

                dim=2
            )


            rolling_b = torch.cat(

                [

                    rolling_b[
                        :,
                        1:,
                        :
                    ],

                    next_b_n.unsqueeze(
                        1
                    )

                ],

                dim=1
            )


    predicted_p = np.asarray(
        predicted_p,
        dtype=np.float32
    )


    predicted_b = np.asarray(
        predicted_b,
        dtype=np.float32
    )


    # actual future states are read ONLY for evaluation

    actual_p = P[
        start_t+1:
        start_t+steps+1
    ]


    actual_b = B[
        start_t+1:
        start_t+steps+1
    ]


    return {

        "predicted_p":
            predicted_p,

        "predicted_b":
            predicted_b,

        "actual_p":
            actual_p,

        "actual_b":
            actual_b,

        "visual_indices":
            np.asarray(
                used_visual_indices
            ),

        "commands":
            np.asarray(
                used_commands
            )
    }


# ======================================================================
# 5. PRINT ALL 48 P VALUES
# ======================================================================

def print_complete_P(
    predicted,
    actual,
    title
):

    print(
        "\n"
        + "=" * 100
    )

    print(
        title
    )

    print(
        "=" * 100
    )


    for leg_idx, leg_name in enumerate(
        LEG_NAMES
    ):


        pred_leg = predicted[
            leg_idx
        ]


        actual_leg = actual[
            leg_idx
        ]


        error_leg = (
            pred_leg
            - actual_leg
        )


        table = pd.DataFrame({

            "Variable":
                P_VARIABLES,

            "Predicted":
                pred_leg,

            "Actual":
                actual_leg,

            "Error":
                error_leg,

            "Abs Error":
                np.abs(
                    error_leg
                )
        })


        print(
            f"\n{leg_name}"
        )


        print(
            table.to_string(
                index=False,
                float_format=lambda x:
                    f"{x: .6f}"
            )
        )


        leg_rmse = float(

            np.sqrt(

                np.mean(
                    error_leg ** 2
                )
            )
        )


        print(
            f"{leg_name} complete "
            f"12D RMSE = "
            f"{leg_rmse:.6f}"
        )


# ======================================================================
# 6. PLOT COMPLETE P STATE
#
# 4 rows = legs
#
# columns:
#
# joint angle
# joint velocity
# joint effort
# contact force
# ======================================================================

def plot_complete_P(
    predicted,
    actual,
    title
):

    fig, axes = plt.subplots(

        4,
        4,

        figsize=(
            17,
            13
        )
    )


    group_items = list(
        SENSOR_GROUPS.items()
    )


    for leg_idx in range(
        4
    ):


        for group_idx, (
            group_name,
            (
                sensor_slice,
                labels
            )
        ) in enumerate(
            group_items
        ):


            ax = axes[
                leg_idx,
                group_idx
            ]


            pred_values = predicted[
                leg_idx,
                sensor_slice
            ]


            actual_values = actual[
                leg_idx,
                sensor_slice
            ]


            x = np.arange(
                3
            )


            width = 0.36


            ax.bar(

                x - width/2,

                actual_values,

                width,

                label="Actual"
            )


            ax.bar(

                x + width/2,

                pred_values,

                width,

                label="Predicted"
            )


            ax.set_xticks(
                x
            )


            ax.set_xticklabels(
                labels
            )


            if leg_idx == 0:

                ax.set_title(
                    group_name
                )


            if group_idx == 0:

                ax.set_ylabel(
                    LEG_NAMES[
                        leg_idx
                    ]
                )


            ax.grid(
                axis="y",
                alpha=0.25
            )


            if (
                leg_idx == 0
                and
                group_idx == 0
            ):

                ax.legend()


    fig.suptitle(
        title,
        fontsize=15
    )


    plt.tight_layout()

    plt.show()


# ======================================================================
# 7. ROLLOUT ERROR GRAPH
#
# Shows how error accumulates when predicted states are fed back.
# ======================================================================

def plot_rollout_errors(
    predicted_p,
    actual_p,
    title
):


    horizon_ms = (

        np.arange(
            1,
            len(predicted_p) + 1
        )

        * dt
        * 1000.0
    )


    group_ranges = {

        "Joint angle":
            slice(
                0,
                3
            ),

        "Joint velocity":
            slice(
                3,
                6
            ),

        "Joint effort":
            slice(
                6,
                9
            ),

        "Foot force":
            slice(
                9,
                12
            )
    }


    plt.figure(
        figsize=(10, 6)
    )


    for name, sensor_slice in (
        group_ranges.items()
    ):


        errors = []


        for k in range(
            len(predicted_p)
        ):


            e = (

                predicted_p[
                    k,
                    :,
                    sensor_slice
                ]

                -

                actual_p[
                    k,
                    :,
                    sensor_slice
                ]
            )


            value = np.sqrt(
                np.mean(
                    e ** 2
                )
            )


            errors.append(
                value
            )


        plt.plot(

            horizon_ms,

            errors,

            marker="o",

            label=name
        )


    plt.xlabel(
        "Recursive prediction horizon (ms)"
    )


    plt.ylabel(
        "RMSE"
    )


    plt.title(
        title
    )


    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.show()


# ======================================================================
# 8. RUN FOUR TESTS
# ======================================================================

ALL_ROLLOUT_RESULTS = []


for case_number, requested_time in enumerate(
    TEST_START_TIMES,
    start=1
):


    # --------------------------------------------------------------
    # nearest test starting state
    # --------------------------------------------------------------

    start_t = int(

        np.argmin(

            np.abs(
                time
                - requested_time
            )
        )
    )


    # enough past history + future rollout

    if (
        start_t < HISTORY - 1
        or
        start_t + ROLLOUT_STEPS >= len(P)
    ):

        raise RuntimeError(
            f"Invalid test time "
            f"{requested_time}"
        )


    print(
        "\n\n"
        + "#" * 110
    )


    print(
        f"TEST CASE {case_number}"
    )


    print(
        f"START t = "
        f"{time[start_t]:.3f} s"
    )


    print(
        f"ONE STEP TARGET = "
        f"{time[start_t+1]:.3f} s"
    )


    print(
        f"0.1 s TARGET = "
        f"{time[start_t+ROLLOUT_STEPS]:.3f} s"
    )


    print(
        "#" * 110
    )


    result = recursive_visual_proprio_rollout(

        start_t,

        steps=ROLLOUT_STEPS
    )


    # ==================================================================
    # TEST A:
    # t -> t+1
    # ==================================================================

    predicted_t1 = result[
        "predicted_p"
    ][0]


    actual_t1 = result[
        "actual_p"
    ][0]


    print_complete_P(

        predicted_t1,

        actual_t1,

        (
            f"TEST {case_number}: "
            f"ONE-STEP PREDICTION "
            f"{time[start_t]:.3f}s "
            f"-> "
            f"{time[start_t+1]:.3f}s"
        )
    )


    plot_complete_P(

        predicted_t1,

        actual_t1,

        (
            f"Test {case_number} — "
            f"all 48 P values at t+1"
        )
    )


    # ==================================================================
    # TEST B:
    # recursive t -> t+10
    # ==================================================================

    predicted_t10 = result[
        "predicted_p"
    ][
        -1
    ]


    actual_t10 = result[
        "actual_p"
    ][
        -1
    ]


    print_complete_P(

        predicted_t10,

        actual_t10,

        (
            f"TEST {case_number}: "
            f"RECURSIVE 0.1s PREDICTION "
            f"{time[start_t]:.3f}s "
            f"-> "
            f"{time[start_t+ROLLOUT_STEPS]:.3f}s"
        )
    )


    plot_complete_P(

        predicted_t10,

        actual_t10,

        (
            f"Test {case_number} — "
            f"all 48 P values after "
            f"0.1 s recursive rollout"
        )
    )


    # ==================================================================
    # SHOW ERROR ACCUMULATION THROUGH ALL 10 STEPS
    # ==================================================================

    plot_rollout_errors(

        result[
            "predicted_p"
        ],

        result[
            "actual_p"
        ],

        (
            f"Test {case_number}: "
            f"error accumulation over "
            f"10 recursive steps"
        )
    )


    # ==================================================================
    # SUMMARY NUMBERS
    # ==================================================================

    one_step_error = (

        predicted_t1
        - actual_t1
    )


    final_error = (

        predicted_t10
        - actual_t10
    )


    one_step_rmse = float(

        np.sqrt(
            np.mean(
                one_step_error ** 2
            )
        )
    )


    final_rmse = float(

        np.sqrt(
            np.mean(
                final_error ** 2
            )
        )
    )


    print(
        "\nSUMMARY"
    )


    print(
        "Complete P[t+1] 48D RMSE:",
        one_step_rmse
    )


    print(
        "Complete P[t+10] 48D RMSE:",
        final_rmse
    )


    print(
        "\nCommands used during rollout:"
    )


    for k in range(
        ROLLOUT_STEPS
    ):


        print(

            f"step {k+1:02d}: "

            f"U[{start_t+k}] = ",

            np.round(

                result[
                    "commands"
                ][k],

                5
            )
        )


    ALL_ROLLOUT_RESULTS.append({

        "case":
            case_number,

        "start_idx":
            start_t,

        "start_time":
            time[start_t],

        "one_step_time":
            time[start_t+1],

        "final_time":
            time[
                start_t
                + ROLLOUT_STEPS
            ],

        "predicted_p":
            result[
                "predicted_p"
            ],

        "actual_p":
            result[
                "actual_p"
            ],

        "predicted_b":
            result[
                "predicted_b"
            ],

        "actual_b":
            result[
                "actual_b"
            ],

        "one_step_rmse":
            one_step_rmse,

        "final_rmse":
            final_rmse
    })


# ======================================================================
# 9. FINAL FOUR-CASE SUMMARY
# ======================================================================

print(
    "\n\n"
    + "=" * 100
)

print(
    "FOUR TEST CASE SUMMARY"
)

print(
    "=" * 100
)


summary_rows = []


for result in ALL_ROLLOUT_RESULTS:


    summary_rows.append({

        "Test":
            result["case"],

        "Start time":
            result["start_time"],

        "t+1 time":
            result["one_step_time"],

        "P t+1 RMSE":
            result["one_step_rmse"],

        "t+0.1 time":
            result["final_time"],

        "P t+0.1 RMSE":
            result["final_rmse"]
    })


summary_df = pd.DataFrame(
    summary_rows
)


print(
    summary_df.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}"
    )
)


# ======================================================================
# 10. SUMMARY GRAPH
# ======================================================================

case_labels = [

    f"Test {r['case']}\n"
    f"{r['start_time']:.1f}s"

    for r in ALL_ROLLOUT_RESULTS
]


one_step_values = [

    r[
        "one_step_rmse"
    ]

    for r in ALL_ROLLOUT_RESULTS
]


final_values = [

    r[
        "final_rmse"
    ]

    for r in ALL_ROLLOUT_RESULTS
]


x = np.arange(
    len(case_labels)
)


width = 0.36


plt.figure(
    figsize=(10, 5)
)


plt.bar(

    x - width/2,

    one_step_values,

    width,

    label="t+1 = 10 ms"
)


plt.bar(

    x + width/2,

    final_values,

    width,

    label="Recursive t+0.1 s"
)


plt.xticks(
    x,
    case_labels
)


plt.ylabel(
    "Complete 48D P-state RMSE"
)


plt.title(
    "One-step vs 0.1 s recursive prediction"
)


plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.show()


print(
    "\nDONE."
)

# %% notebook cell 13
# ================================================================
# V2 CELL 1 — CAUSAL LEG-PHASE PROXY
# ================================================================

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


CONTACT_THRESHOLD = 10.0

# P:
# N x 4 x 12
#
# [:,:,9:12] = foot force xyz

force_magnitude = np.linalg.norm(
    P[:, :, 9:12],
    axis=-1
)

contact_state = (
    force_magnitude > CONTACT_THRESHOLD
).astype(np.float32)


# ------------------------------------------------
# second phase feature:
#
# amount of time since last stance/swing transition
#
# This is CAUSAL.
# It never looks into the future.
# ------------------------------------------------

time_since_transition = np.zeros_like(
    contact_state,
    dtype=np.float32
)


MAX_PHASE_HISTORY = 50


for leg in range(4):

    previous = contact_state[0, leg]
    count = 0

    for t in range(len(P)):

        current = contact_state[t, leg]

        if t == 0 or current != previous:

            count = 0
            previous = current

        else:

            count += 1

        time_since_transition[t, leg] = (
            min(count, MAX_PHASE_HISTORY)
            /
            MAX_PHASE_HISTORY
        )


# N x 4 x 2
#
# feature 0 = contact state
# feature 1 = time since contact transition

PHASE_PROXY = np.stack(
    [
        contact_state,
        time_since_transition
    ],
    axis=-1
).astype(np.float32)


print("Phase proxy:", PHASE_PROXY.shape)

print(
    "Contact percentage:",
    100 * contact_state.mean()
)

# %% notebook cell 14
# ================================================================
# V2 CELL 2 — DATASET
# ================================================================

HISTORY_V2 = 50
HORIZON_V2 = 10


def normalize_u_sequence(x):

    return np.stack(
        [
            norm_u(row)
            for row in x
        ],
        axis=0
    ).astype(np.float32)


def normalize_p_sequence(x):

    return np.stack(
        [
            norm_p(row)
            for row in x
        ],
        axis=0
    ).astype(np.float32)


def normalize_b_sequence(x):

    return np.stack(
        [
            norm_b(row)
            for row in x
        ],
        axis=0
    ).astype(np.float32)


def make_split_indices(
    start_time,
    end_time
):

    indices = []

    for t in range(
        HISTORY_V2 - 1,
        len(P) - HORIZON_V2
    ):

        # input must lie inside split
        if time[t] < start_time:
            continue

        # complete future horizon must lie inside split
        if time[t + HORIZON_V2] >= end_time:
            continue

        # causal camera must exist
        if visual_index_for_state[t] < 0:
            continue

        indices.append(t)

    return np.asarray(
        indices,
        dtype=np.int64
    )


train_indices_v2 = make_split_indices(
    0.0,
    90.0
)

val_indices_v2 = make_split_indices(
    90.0,
    100.0
)

test_indices_v2 = make_split_indices(
    100.0,
    120.0
)


print(
    "Train:",
    len(train_indices_v2)
)

print(
    "Val  :",
    len(val_indices_v2)
)

print(
    "Test :",
    len(test_indices_v2)
)


class TerrainDynamicsDatasetV2(
    Dataset
):

    def __init__(
        self,
        indices
    ):

        self.indices = np.asarray(
            indices,
            dtype=np.int64
        )


    def __len__(self):

        return len(
            self.indices
        )


    def __getitem__(
        self,
        index
    ):

        t = int(
            self.indices[index]
        )


        # ========================================================
        # P HISTORY
        #
        # raw:
        # 50 x 4 x 12
        #
        # output:
        # 4 x 50 x 12
        # ========================================================

        p_hist = P[
            t-HISTORY_V2+1:
            t+1
        ]

        p_hist_n = normalize_p_sequence(
            p_hist
        )

        p_hist_n = np.transpose(
            p_hist_n,
            (
                1,
                0,
                2
            )
        )


        # ========================================================
        # B HISTORY
        #
        # 50 x 12
        # ========================================================

        b_hist = B[
            t-HISTORY_V2+1:
            t+1
        ]

        b_hist_n = normalize_b_sequence(
            b_hist
        )


        # ========================================================
        # COMMAND HISTORY
        #
        # THIS IS NEW
        #
        # 50 x 6
        # ========================================================

        u_hist = U[
            t-HISTORY_V2+1:
            t+1
        ]

        u_hist_n = normalize_u_sequence(
            u_hist
        )


        # ========================================================
        # LEG PHASE HISTORY
        #
        # raw:
        # 50 x 4 x 2
        #
        # output:
        # 4 x 50 x 2
        # ========================================================

        phase_hist = PHASE_PROXY[
            t-HISTORY_V2+1:
            t+1
        ]

        phase_hist = np.transpose(
            phase_hist,
            (
                1,
                0,
                2
            )
        ).astype(np.float32)


        # ========================================================
        # VISUAL INPUT AT CURRENT TIME t
        # ========================================================

        vi = int(
            visual_index_for_state[t]
        )


        dino = np.asarray(
            DINO_FEATURES[vi],
            dtype=np.float32
        ).reshape(-1)


        conv = np.asarray(
            CONV_FEATURES[vi],
            dtype=np.float32
        ).reshape(-1)


        depth = np.asarray(
            DEPTH_GRIDS[vi],
            dtype=np.float32
        )


        # ========================================================
        # FUTURE COMMAND SEQUENCE
        #
        # U[t] drives:
        # t -> t+1
        #
        # U[t+9] drives:
        # t+9 -> t+10
        # ========================================================

        future_u = U[
            t:
            t + HORIZON_V2
        ]

        future_u_n = normalize_u_sequence(
            future_u
        )


        # ========================================================
        # ONE STEP TARGET
        # ========================================================

        target_p1 = norm_p(
            P[t+1]
        ).astype(np.float32)


        target_b1 = norm_b(
            B[t+1]
        ).astype(np.float32)


        # ========================================================
        # DIRECT 10-STEP TARGET
        # ========================================================

        future_p = P[
            t+1:
            t+HORIZON_V2+1
        ]

        future_b = B[
            t+1:
            t+HORIZON_V2+1
        ]


        future_p_n = normalize_p_sequence(
            future_p
        )

        future_b_n = normalize_b_sequence(
            future_b
        )


        # ========================================================
        # CONTACT TARGET
        #
        # 10 x 4
        # ========================================================

        future_contact = contact_state[
            t+1:
            t+HORIZON_V2+1
        ].astype(np.float32)


        return {

            "p_hist":
                torch.tensor(p_hist_n),

            "b_hist":
                torch.tensor(b_hist_n),

            "u_hist":
                torch.tensor(u_hist_n),

            "phase_hist":
                torch.tensor(phase_hist),

            "dino":
                torch.tensor(dino),

            "conv":
                torch.tensor(conv),

            "depth":
                torch.tensor(depth),

            "future_u":
                torch.tensor(future_u_n),

            "target_p1":
                torch.tensor(target_p1),

            "target_b1":
                torch.tensor(target_b1),

            "future_p":
                torch.tensor(future_p_n),

            "future_b":
                torch.tensor(future_b_n),

            "future_contact":
                torch.tensor(future_contact),

            "t":
                torch.tensor(
                    t,
                    dtype=torch.long
                )
        }


train_dataset_v2 = TerrainDynamicsDatasetV2(
    train_indices_v2
)

val_dataset_v2 = TerrainDynamicsDatasetV2(
    val_indices_v2
)

test_dataset_v2 = TerrainDynamicsDatasetV2(
    test_indices_v2
)


BATCH_SIZE_V2 = 32


train_loader_v2 = DataLoader(
    train_dataset_v2,
    batch_size=BATCH_SIZE_V2,
    shuffle=True,
    num_workers=2
)

val_loader_v2 = DataLoader(
    val_dataset_v2,
    batch_size=BATCH_SIZE_V2,
    shuffle=False,
    num_workers=2
)

test_loader_v2 = DataLoader(
    test_dataset_v2,
    batch_size=BATCH_SIZE_V2,
    shuffle=False,
    num_workers=2
)

# %% notebook cell 15
# ================================================================
# V2 CELL 2 — DATASET
# ================================================================

HISTORY_V2 = 50
HORIZON_V2 = 10


def normalize_u_sequence(x):

    return np.stack(
        [
            norm_u(row)
            for row in x
        ],
        axis=0
    ).astype(np.float32)


def normalize_p_sequence(x):

    return np.stack(
        [
            norm_p(row)
            for row in x
        ],
        axis=0
    ).astype(np.float32)


def normalize_b_sequence(x):

    return np.stack(
        [
            norm_b(row)
            for row in x
        ],
        axis=0
    ).astype(np.float32)


def make_split_indices(
    start_time,
    end_time
):

    indices = []

    for t in range(
        HISTORY_V2 - 1,
        len(P) - HORIZON_V2
    ):

        # input must lie inside split
        if time[t] < start_time:
            continue

        # complete future horizon must lie inside split
        if time[t + HORIZON_V2] >= end_time:
            continue

        # causal camera must exist
        if visual_index_for_state[t] < 0:
            continue

        indices.append(t)

    return np.asarray(
        indices,
        dtype=np.int64
    )


train_indices_v2 = make_split_indices(
    0.0,
    90.0
)

val_indices_v2 = make_split_indices(
    90.0,
    100.0
)

test_indices_v2 = make_split_indices(
    100.0,
    120.0
)


print(
    "Train:",
    len(train_indices_v2)
)

print(
    "Val  :",
    len(val_indices_v2)
)

print(
    "Test :",
    len(test_indices_v2)
)


class TerrainDynamicsDatasetV2(
    Dataset
):

    def __init__(
        self,
        indices
    ):

        self.indices = np.asarray(
            indices,
            dtype=np.int64
        )


    def __len__(self):

        return len(
            self.indices
        )


    def __getitem__(
        self,
        index
    ):

        t = int(
            self.indices[index]
        )


        # ========================================================
        # P HISTORY
        #
        # raw:
        # 50 x 4 x 12
        #
        # output:
        # 4 x 50 x 12
        # ========================================================

        p_hist = P[
            t-HISTORY_V2+1:
            t+1
        ]

        p_hist_n = normalize_p_sequence(
            p_hist
        )

        p_hist_n = np.transpose(
            p_hist_n,
            (
                1,
                0,
                2
            )
        )


        # ========================================================
        # B HISTORY
        #
        # 50 x 12
        # ========================================================

        b_hist = B[
            t-HISTORY_V2+1:
            t+1
        ]

        b_hist_n = normalize_b_sequence(
            b_hist
        )


        # ========================================================
        # COMMAND HISTORY
        #
        # THIS IS NEW
        #
        # 50 x 6
        # ========================================================

        u_hist = U[
            t-HISTORY_V2+1:
            t+1
        ]

        u_hist_n = normalize_u_sequence(
            u_hist
        )


        # ========================================================
        # LEG PHASE HISTORY
        #
        # raw:
        # 50 x 4 x 2
        #
        # output:
        # 4 x 50 x 2
        # ========================================================

        phase_hist = PHASE_PROXY[
            t-HISTORY_V2+1:
            t+1
        ]

        phase_hist = np.transpose(
            phase_hist,
            (
                1,
                0,
                2
            )
        ).astype(np.float32)


        # ========================================================
        # VISUAL INPUT AT CURRENT TIME t
        # ========================================================

        vi = int(
            visual_index_for_state[t]
        )


        dino = np.asarray(
            DINO_FEATURES[vi],
            dtype=np.float32
        ).reshape(-1)


        conv = np.asarray(
            CONV_FEATURES[vi],
            dtype=np.float32
        ).reshape(-1)


        depth = np.asarray(
            DEPTH_GRIDS[vi],
            dtype=np.float32
        )


        # ========================================================
        # FUTURE COMMAND SEQUENCE
        #
        # U[t] drives:
        # t -> t+1
        #
        # U[t+9] drives:
        # t+9 -> t+10
        # ========================================================

        future_u = U[
            t:
            t + HORIZON_V2
        ]

        future_u_n = normalize_u_sequence(
            future_u
        )


        # ========================================================
        # ONE STEP TARGET
        # ========================================================

        target_p1 = norm_p(
            P[t+1]
        ).astype(np.float32)


        target_b1 = norm_b(
            B[t+1]
        ).astype(np.float32)


        # ========================================================
        # DIRECT 10-STEP TARGET
        # ========================================================

        future_p = P[
            t+1:
            t+HORIZON_V2+1
        ]

        future_b = B[
            t+1:
            t+HORIZON_V2+1
        ]


        future_p_n = normalize_p_sequence(
            future_p
        )

        future_b_n = normalize_b_sequence(
            future_b
        )


        # ========================================================
        # CONTACT TARGET
        #
        # 10 x 4
        # ========================================================

        future_contact = contact_state[
            t+1:
            t+HORIZON_V2+1
        ].astype(np.float32)


        return {

            "p_hist":
                torch.tensor(p_hist_n),

            "b_hist":
                torch.tensor(b_hist_n),

            "u_hist":
                torch.tensor(u_hist_n),

            "phase_hist":
                torch.tensor(phase_hist),

            "dino":
                torch.tensor(dino),

            "conv":
                torch.tensor(conv),

            "depth":
                torch.tensor(depth),

            "future_u":
                torch.tensor(future_u_n),

            "target_p1":
                torch.tensor(target_p1),

            "target_b1":
                torch.tensor(target_b1),

            "future_p":
                torch.tensor(future_p_n),

            "future_b":
                torch.tensor(future_b_n),

            "future_contact":
                torch.tensor(future_contact),

            "t":
                torch.tensor(
                    t,
                    dtype=torch.long
                )
        }


train_dataset_v2 = TerrainDynamicsDatasetV2(
    train_indices_v2
)

val_dataset_v2 = TerrainDynamicsDatasetV2(
    val_indices_v2
)

test_dataset_v2 = TerrainDynamicsDatasetV2(
    test_indices_v2
)


BATCH_SIZE_V2 = 32


train_loader_v2 = DataLoader(
    train_dataset_v2,
    batch_size=BATCH_SIZE_V2,
    shuffle=True,
    num_workers=2
)

val_loader_v2 = DataLoader(
    val_dataset_v2,
    batch_size=BATCH_SIZE_V2,
    shuffle=False,
    num_workers=2
)

test_loader_v2 = DataLoader(
    test_dataset_v2,
    batch_size=BATCH_SIZE_V2,
    shuffle=False,
    num_workers=2
)

# %% notebook cell 16
# ================================================================
# V2 CELL 3 — MODEL
# ================================================================

class CausalTransformerEncoder(
    nn.Module
):

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        layers=2,
        heads=4
    ):

        super().__init__()


        self.input_projection = nn.Linear(
            input_dim,
            hidden_dim
        )


        encoder_layer = nn.TransformerEncoderLayer(

            d_model=hidden_dim,

            nhead=heads,

            dim_feedforward=hidden_dim * 4,

            dropout=0.1,

            batch_first=True,

            norm_first=True
        )


        self.encoder = nn.TransformerEncoder(

            encoder_layer,

            num_layers=layers
        )


        self.norm = nn.LayerNorm(
            hidden_dim
        )


    def forward(
        self,
        x
    ):

        # B x T x D

        x = self.input_projection(
            x
        )


        T = x.shape[1]


        mask = torch.triu(

            torch.ones(
                T,
                T,
                device=x.device,
                dtype=torch.bool
            ),

            diagonal=1
        )


        x = self.encoder(
            x,
            mask=mask
        )


        x = self.norm(
            x
        )


        return x[:, -1]


# ================================================================


class TerrainVisualEncoderV2(
    nn.Module
):

    def __init__(
        self,
        dino_dim=384,
        conv_dim=768
    ):

        super().__init__()


        self.dino_proj = nn.Sequential(

            nn.Linear(
                dino_dim,
                256
            ),

            nn.GELU(),

            nn.LayerNorm(
                256
            )
        )


        self.conv_proj = nn.Sequential(

            nn.Linear(
                conv_dim,
                256
            ),

            nn.GELU(),

            nn.LayerNorm(
                256
            )
        )


        self.depth_encoder = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                5,
                stride=2,
                padding=2
            ),

            nn.GELU(),

            nn.Conv2d(
                32,
                64,
                3,
                stride=2,
                padding=1
            ),

            nn.GELU(),

            nn.Conv2d(
                64,
                128,
                3,
                stride=2,
                padding=1
            ),

            nn.GELU(),

            nn.AdaptiveAvgPool2d(
                1
            )
        )


        self.depth_proj = nn.Sequential(

            nn.Linear(
                128,
                256
            ),

            nn.GELU(),

            nn.LayerNorm(
                256
            )
        )


        layer = nn.TransformerEncoderLayer(

            d_model=256,

            nhead=4,

            dim_feedforward=512,

            dropout=0.1,

            batch_first=True,

            norm_first=True
        )


        self.fusion = nn.TransformerEncoder(

            layer,

            num_layers=2
        )


        self.type_embedding = nn.Parameter(

            torch.randn(
                1,
                3,
                256
            ) * 0.02
        )


        self.norm = nn.LayerNorm(
            256
        )


    def forward(
        self,
        dino,
        conv,
        depth
    ):

        # robust ConvNeXt shape handling
        conv = conv.flatten(
            start_dim=1
        )


        dino_token = self.dino_proj(
            dino
        )


        conv_token = self.conv_proj(
            conv
        )


        depth_feature = self.depth_encoder(
            depth
        ).flatten(1)


        depth_token = self.depth_proj(
            depth_feature
        )


        tokens = torch.stack(

            [
                dino_token,
                conv_token,
                depth_token
            ],

            dim=1
        )


        tokens = (
            tokens
            +
            self.type_embedding
        )


        tokens = self.fusion(
            tokens
        )


        return self.norm(
            tokens.mean(dim=1)
        )


# ================================================================
# COMPLETE MODEL
# ================================================================

class TerrainAwareDynamicsV2(
    nn.Module
):

    def __init__(
        self,
        dino_dim=384,
        conv_dim=768,
        horizon=10
    ):

        super().__init__()


        self.horizon = horizon


        # ========================================================
        # LEG TRANSFORMER
        #
        # per timestep:
        #
        # P     = 12
        # U     = 6
        # phase = 2
        #
        # total = 20
        # ========================================================

        self.leg_encoder = CausalTransformerEncoder(

            input_dim=20,

            hidden_dim=128,

            layers=2,

            heads=4
        )


        self.leg_embedding = nn.Embedding(
            4,
            128
        )


        # ========================================================
        # BODY TRANSFORMER
        #
        # body = 12
        # command = 6
        #
        # 18
        # ========================================================

        self.body_encoder = CausalTransformerEncoder(

            input_dim=18,

            hidden_dim=128,

            layers=2,

            heads=4
        )


        # ========================================================
        # VISION
        # ========================================================

        self.visual_encoder = TerrainVisualEncoderV2(

            dino_dim=dino_dim,

            conv_dim=conv_dim
        )


        # ========================================================
        # FOUR LEGS + BODY + TERRAIN
        #
        # 4*128 + 128 + 256
        # = 896
        # ========================================================

        self.context = nn.Sequential(

            nn.LayerNorm(
                896
            ),

            nn.Linear(
                896,
                512
            ),

            nn.GELU(),

            nn.Dropout(
                0.1
            ),

            nn.Linear(
                512,
                256
            ),

            nn.GELU(),

            nn.LayerNorm(
                256
            )
        )


        # ========================================================
        # IMMEDIATE t+1 HEAD
        #
        # output:
        #
        # P = 48
        # B = 12
        #
        # total = 60
        # ========================================================

        self.one_step_head = nn.Sequential(

            nn.Linear(
                256,
                256
            ),

            nn.GELU(),

            nn.Linear(
                256,
                60
            )
        )


        # explicit contact state
        self.one_contact_head = nn.Linear(
            256,
            4
        )


        # ========================================================
        # FUTURE COMMAND ENCODER
        # ========================================================

        self.future_command_encoder = nn.Sequential(

            nn.Linear(
                6,
                64
            ),

            nn.GELU(),

            nn.Linear(
                64,
                64
            )
        )


        # context + future command
        self.future_input = nn.Sequential(

            nn.Linear(
                256 + 64,
                128
            ),

            nn.GELU()
        )


        # ========================================================
        # DIRECT MULTI-STEP MODEL
        #
        # This replaces recursive state prediction.
        # ========================================================

        self.future_gru = nn.GRU(

            input_size=128,

            hidden_size=256,

            num_layers=2,

            batch_first=True,

            dropout=0.1
        )


        self.initial_hidden = nn.Linear(

            256,

            2 * 256
        )


        self.future_state_head = nn.Sequential(

            nn.Linear(
                256,
                256
            ),

            nn.GELU(),

            nn.Linear(
                256,
                60
            )
        )


        self.future_contact_head = nn.Linear(

            256,

            4
        )


    # ============================================================

    def forward(
        self,
        p_hist,
        b_hist,
        u_hist,
        phase_hist,
        dino,
        conv,
        depth,
        future_u
    ):


        batch_size = p_hist.shape[0]

        H = p_hist.shape[2]


        # ========================================================
        # PER-LEG TEMPORAL INPUT
        # ========================================================

        # B x H x 6
        # ->
        # B x 4 x H x 6

        u_leg = u_hist.unsqueeze(
            1
        ).expand(
            -1,
            4,
            -1,
            -1
        )


        # B x 4 x H x 20

        leg_input = torch.cat(

            [
                p_hist,
                u_leg,
                phase_hist
            ],

            dim=-1
        )


        # B*4 x H x 20

        leg_input = leg_input.reshape(

            batch_size * 4,

            H,

            20
        )


        leg_features = self.leg_encoder(
            leg_input
        )


        # add leg identity

        leg_ids = torch.arange(
            4,
            device=p_hist.device
        ).repeat(
            batch_size
        )


        leg_features = (

            leg_features
            +
            self.leg_embedding(
                leg_ids
            )
        )


        leg_features = leg_features.reshape(

            batch_size,

            4,

            128
        )


        leg_features_flat = leg_features.reshape(

            batch_size,

            4 * 128
        )


        # ========================================================
        # BODY + COMMAND HISTORY
        # ========================================================

        body_input = torch.cat(

            [
                b_hist,
                u_hist
            ],

            dim=-1
        )


        body_feature = self.body_encoder(
            body_input
        )


        # ========================================================
        # TERRAIN
        # ========================================================

        terrain_feature = self.visual_encoder(

            dino,
            conv,
            depth
        )


        # ========================================================
        # SHARED PHYSICAL CONTEXT
        # ========================================================

        context = torch.cat(

            [
                leg_features_flat,
                body_feature,
                terrain_feature
            ],

            dim=-1
        )


        context = self.context(
            context
        )


        # ========================================================
        # CURRENT NORMALIZED STATE
        # ========================================================

        current_p = p_hist[
            :,
            :,
            -1,
            :
        ].reshape(
            batch_size,
            48
        )


        current_b = b_hist[
            :,
            -1,
            :
        ]


        current_state = torch.cat(

            [
                current_p,
                current_b
            ],

            dim=-1
        )


        # ========================================================
        # ONE-STEP RESIDUAL
        # ========================================================

        one_delta = self.one_step_head(
            context
        )


        predicted_one = (
            current_state
            +
            one_delta
        )


        one_contact_logits = (
            self.one_contact_head(
                context
            )
        )


        # ========================================================
        # DIRECT MULTI-STEP PREDICTION
        #
        # Important:
        #
        # No predicted state is fed into the next step.
        #
        # Instead:
        # context + U[t:t+9]
        # -> entire horizon
        # ========================================================

        future_command_feature = (
            self.future_command_encoder(
                future_u
            )
        )


        context_sequence = (
            context
            .unsqueeze(1)
            .expand(
                -1,
                self.horizon,
                -1
            )
        )


        future_input = torch.cat(

            [
                context_sequence,
                future_command_feature
            ],

            dim=-1
        )


        future_input = self.future_input(
            future_input
        )


        hidden = self.initial_hidden(
            context
        )


        hidden = hidden.view(

            batch_size,

            2,

            256
        ).permute(
            1,
            0,
            2
        ).contiguous()


        future_latent, _ = self.future_gru(

            future_input,

            hidden
        )


        future_delta = self.future_state_head(
            future_latent
        )


        predicted_future = (

            current_state.unsqueeze(1)

            +

            future_delta
        )


        future_contact_logits = (
            self.future_contact_head(
                future_latent
            )
        )


        # ========================================================
        # SPLIT P/B
        # ========================================================

        one_p = predicted_one[
            :,
            :48
        ].reshape(
            batch_size,
            4,
            12
        )


        one_b = predicted_one[
            :,
            48:
        ]


        future_p = predicted_future[
            :,
            :,
            :48
        ].reshape(

            batch_size,

            self.horizon,

            4,

            12
        )


        future_b = predicted_future[
            :,
            :,
            48:
        ]


        return {

            "one_p":
                one_p,

            "one_b":
                one_b,

            "one_contact_logits":
                one_contact_logits,

            "future_p":
                future_p,

            "future_b":
                future_b,

            "future_contact_logits":
                future_contact_logits,

            "context":
                context
        }

# %% notebook cell 17
# ================================================================
# V2 CELL 4 — MODEL CHECK
# ================================================================

model_v2 = TerrainAwareDynamicsV2(

    dino_dim=DINO_FEATURES.shape[1],

    conv_dim=CONV_FEATURES.reshape(
        len(CONV_FEATURES),
        -1
    ).shape[1],

    horizon=HORIZON_V2

).to(device)


def move_batch_v2(
    batch
):

    return {

        key:
            value.to(
                device
            )

        for key, value in batch.items()
    }


batch = next(
    iter(
        train_loader_v2
    )
)


batch = move_batch_v2(
    batch
)


model_v2.eval()


with torch.no_grad():

    output = model_v2(

        batch["p_hist"],
        batch["b_hist"],
        batch["u_hist"],
        batch["phase_hist"],

        batch["dino"],
        batch["conv"],
        batch["depth"],

        batch["future_u"]
    )


print(
    "ONE STEP P:",
    output["one_p"].shape
)

print(
    "ONE STEP B:",
    output["one_b"].shape
)

print(
    "ONE CONTACT:",
    output["one_contact_logits"].shape
)

print(
    "FUTURE P:",
    output["future_p"].shape
)

print(
    "FUTURE B:",
    output["future_b"].shape
)

print(
    "FUTURE CONTACT:",
    output["future_contact_logits"].shape
)

# %% notebook cell 18
# ================================================================
# V2 CELL 5 — TRAIN
# ================================================================

optimizer_v2 = torch.optim.AdamW(

    model_v2.parameters(),

    lr=2e-4,

    weight_decay=1e-4
)


state_mse = nn.MSELoss()

contact_bce = nn.BCEWithLogitsLoss()


EPOCHS_V2 = 10


train_loss_v2 = []
val_loss_v2 = []


best_val = float("inf")
best_state = None


# later horizons get slightly more importance
horizon_weights = torch.linspace(

    1.0,
    2.0,
    HORIZON_V2,

    device=device

).view(
    1,
    HORIZON_V2,
    1
)


for epoch in range(
    1,
    EPOCHS_V2 + 1
):

    # ============================================================
    # TRAIN
    # ============================================================

    model_v2.train()

    train_losses = []


    for batch in train_loader_v2:

        batch = move_batch_v2(
            batch
        )


        optimizer_v2.zero_grad()


        out = model_v2(

            batch["p_hist"],
            batch["b_hist"],
            batch["u_hist"],
            batch["phase_hist"],

            batch["dino"],
            batch["conv"],
            batch["depth"],

            batch["future_u"]
        )


        # --------------------------------------------------------
        # pack t+1 target
        # --------------------------------------------------------

        target_one = torch.cat(

            [
                batch[
                    "target_p1"
                ].reshape(
                    len(batch["target_p1"]),
                    48
                ),

                batch[
                    "target_b1"
                ]
            ],

            dim=-1
        )


        predicted_one = torch.cat(

            [
                out[
                    "one_p"
                ].reshape(
                    len(out["one_p"]),
                    48
                ),

                out[
                    "one_b"
                ]
            ],

            dim=-1
        )


        # --------------------------------------------------------
        # direct horizon target
        # --------------------------------------------------------

        target_future = torch.cat(

            [
                batch[
                    "future_p"
                ].reshape(
                    len(batch["future_p"]),
                    HORIZON_V2,
                    48
                ),

                batch[
                    "future_b"
                ]
            ],

            dim=-1
        )


        predicted_future = torch.cat(

            [
                out[
                    "future_p"
                ].reshape(
                    len(out["future_p"]),
                    HORIZON_V2,
                    48
                ),

                out[
                    "future_b"
                ]
            ],

            dim=-1
        )


        # ========================================================
        # LOSSES
        # ========================================================

        loss_one = state_mse(

            predicted_one,

            target_one
        )


        horizon_squared_error = (

            predicted_future
            -
            target_future

        ) ** 2


        loss_horizon = (

            horizon_squared_error
            *
            horizon_weights

        ).mean()


        # contact state at t+1

        contact_one_target = (
            batch["future_contact"][
                :,
                0,
                :
            ]
        )


        loss_contact_one = contact_bce(

            out[
                "one_contact_logits"
            ],

            contact_one_target
        )


        loss_contact_future = contact_bce(

            out[
                "future_contact_logits"
            ],

            batch[
                "future_contact"
            ]
        )


        # --------------------------------------------------------
        # final loss
        # --------------------------------------------------------

        loss = (

            loss_one

            +

            loss_horizon

            +

            0.2 * loss_contact_one

            +

            0.2 * loss_contact_future
        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(

            model_v2.parameters(),

            1.0
        )


        optimizer_v2.step()


        train_losses.append(
            float(
                loss.item()
            )
        )


    # ============================================================
    # VALIDATION
    # ============================================================

    model_v2.eval()

    validation_losses = []


    with torch.no_grad():

        for batch in val_loader_v2:

            batch = move_batch_v2(
                batch
            )


            out = model_v2(

                batch["p_hist"],
                batch["b_hist"],
                batch["u_hist"],
                batch["phase_hist"],

                batch["dino"],
                batch["conv"],
                batch["depth"],

                batch["future_u"]
            )


            target_one = torch.cat(

                [
                    batch[
                        "target_p1"
                    ].reshape(
                        len(
                            batch[
                                "target_p1"
                            ]
                        ),
                        48
                    ),

                    batch[
                        "target_b1"
                    ]
                ],

                dim=-1
            )


            predicted_one = torch.cat(

                [
                    out[
                        "one_p"
                    ].reshape(
                        len(
                            out[
                                "one_p"
                            ]
                        ),
                        48
                    ),

                    out[
                        "one_b"
                    ]
                ],

                dim=-1
            )


            target_future = torch.cat(

                [
                    batch[
                        "future_p"
                    ].reshape(
                        len(
                            batch[
                                "future_p"
                            ]
                        ),
                        HORIZON_V2,
                        48
                    ),

                    batch[
                        "future_b"
                    ]
                ],

                dim=-1
            )


            predicted_future = torch.cat(

                [
                    out[
                        "future_p"
                    ].reshape(
                        len(
                            out[
                                "future_p"
                            ]
                        ),
                        HORIZON_V2,
                        48
                    ),

                    out[
                        "future_b"
                    ]
                ],

                dim=-1
            )


            loss_one = state_mse(

                predicted_one,
                target_one
            )


            loss_horizon = (

                (
                    predicted_future
                    -
                    target_future
                ) ** 2

                *
                horizon_weights

            ).mean()


            contact_one_target = (
                batch[
                    "future_contact"
                ][
                    :,
                    0,
                    :
                ]
            )


            loss_contact_one = contact_bce(

                out[
                    "one_contact_logits"
                ],

                contact_one_target
            )


            loss_contact_future = contact_bce(

                out[
                    "future_contact_logits"
                ],

                batch[
                    "future_contact"
                ]
            )


            loss = (

                loss_one

                +

                loss_horizon

                +

                0.2 * loss_contact_one

                +

                0.2 * loss_contact_future
            )


            validation_losses.append(
                float(
                    loss.item()
                )
            )


    train_epoch = np.mean(
        train_losses
    )

    val_epoch = np.mean(
        validation_losses
    )


    train_loss_v2.append(
        train_epoch
    )

    val_loss_v2.append(
        val_epoch
    )


    print(

        f"Epoch {epoch:02d}/{EPOCHS_V2}"

        f" | train = {train_epoch:.6f}"

        f" | val = {val_epoch:.6f}"
    )


    if val_epoch < best_val:

        best_val = val_epoch

        best_state = {

            key:
                value
                .detach()
                .cpu()
                .clone()

            for key, value
            in model_v2.state_dict().items()
        }


model_v2.load_state_dict(
    best_state
)


print(
    "\nBest validation loss:",
    best_val
)

# %% notebook cell 19
# ======================================================================
# V2 CELL 6
#
# SAME FOUR TEST CASES AS V1
#
# Compare:
#
# 1. t -> t+1      = 0.01 s = 10 ms
# 2. t -> t+10     = 0.10 s = 100 ms DIRECT prediction
#
# ALL 48 VALUES OF P ARE EVALUATED
# ======================================================================

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


TEST_CASE_TIMES = [
    102.0,
    108.0,
    114.0,
    119.0
]


LEG_NAMES = [
    "FL",
    "FR",
    "RL",
    "RR"
]


P_NAMES = [
    "q1",
    "q2",
    "q3",

    "dq1",
    "dq2",
    "dq3",

    "effort1",
    "effort2",
    "effort3",

    "Fx",
    "Fy",
    "Fz"
]


# ======================================================================
# HELPERS
# ======================================================================

def rmse(pred, actual):

    return float(
        np.sqrt(
            np.mean(
                (pred - actual) ** 2
            )
        )
    )


def denorm_p_safe(x):

    x = np.asarray(x)

    original_shape = x.shape

    flat = x.reshape(
        -1,
        4,
        12
    )

    result = np.stack(
        [
            denorm_p(v)
            for v in flat
        ],
        axis=0
    )

    return result.reshape(
        original_shape
    )


def denorm_b_safe(x):

    x = np.asarray(x)

    original_shape = x.shape

    flat = x.reshape(
        -1,
        12
    )

    result = np.stack(
        [
            denorm_b(v)
            for v in flat
        ],
        axis=0
    )

    return result.reshape(
        original_shape
    )


# ======================================================================
# FIND DATASET SAMPLE CORRESPONDING TO STATE INDEX t
# ======================================================================

def get_dataset_item_for_t(
    dataset,
    target_t
):

    matches = np.where(
        dataset.indices == target_t
    )[0]


    if len(matches) == 0:

        raise RuntimeError(
            f"State index {target_t} "
            f"not found in test dataset."
        )


    return dataset[
        int(matches[0])
    ]


# ======================================================================
# ADD BATCH DIMENSION
# ======================================================================

def batchify_case(
    sample
):

    result = {}

    for key, value in sample.items():

        if torch.is_tensor(value):

            result[key] = (
                value
                .unsqueeze(0)
                .to(device)
            )

        else:

            result[key] = value

    return result


# ======================================================================
# PRINT ALL 48 P VALUES IN READABLE FORMAT
# ======================================================================

def show_complete_p_table(
    predicted,
    actual,
    title
):

    print(
        "\n"
        + "=" * 95
    )

    print(title)

    print(
        "=" * 95
    )


    for leg in range(4):

        table = pd.DataFrame({

            "Variable":
                P_NAMES,

            "Predicted":
                predicted[
                    leg
                ],

            "Actual":
                actual[
                    leg
                ],

            "Error":
                predicted[
                    leg
                ]
                -
                actual[
                    leg
                ],

            "|Error|":
                np.abs(
                    predicted[
                        leg
                    ]
                    -
                    actual[
                        leg
                    ]
                )
        })


        print(
            f"\n--- {LEG_NAMES[leg]} ---"
        )


        print(
            table.to_string(
                index=False,
                float_format=lambda x:
                    f"{x: .5f}"
            )
        )


# ======================================================================
# SENSOR-GROUP RMSE
# ======================================================================

def get_group_rmse(
    predicted,
    actual
):

    return {

        "Joint angle":
            rmse(
                predicted[:, 0:3],
                actual[:, 0:3]
            ),

        "Joint velocity":
            rmse(
                predicted[:, 3:6],
                actual[:, 3:6]
            ),

        "Joint effort":
            rmse(
                predicted[:, 6:9],
                actual[:, 6:9]
            ),

        "Foot force":
            rmse(
                predicted[:, 9:12],
                actual[:, 9:12]
            ),

        "Complete P":
            rmse(
                predicted,
                actual
            )
    }


# ======================================================================
# RUN MODEL ON ONE TEST CASE
# ======================================================================

def evaluate_v2_case(
    requested_time
):

    # --------------------------------------------------------------
    # nearest state index
    # --------------------------------------------------------------

    t = int(
        np.argmin(
            np.abs(
                time - requested_time
            )
        )
    )


    sample = get_dataset_item_for_t(
        test_dataset_v2,
        t
    )


    batch = batchify_case(
        sample
    )


    model_v2.eval()


    with torch.no_grad():

        out = model_v2(

            batch["p_hist"],
            batch["b_hist"],
            batch["u_hist"],
            batch["phase_hist"],

            batch["dino"],
            batch["conv"],
            batch["depth"],

            batch["future_u"]
        )


    # ==============================================================
    # ONE STEP
    # ==============================================================

    pred_p_10ms_n = (
        out["one_p"][0]
        .cpu()
        .numpy()
    )


    pred_b_10ms_n = (
        out["one_b"][0]
        .cpu()
        .numpy()
    )


    actual_p_10ms_n = (
        batch["target_p1"][0]
        .cpu()
        .numpy()
    )


    actual_b_10ms_n = (
        batch["target_b1"][0]
        .cpu()
        .numpy()
    )


    pred_p_10ms = denorm_p_safe(
        pred_p_10ms_n
    )


    actual_p_10ms = denorm_p_safe(
        actual_p_10ms_n
    )


    pred_b_10ms = denorm_b_safe(
        pred_b_10ms_n
    )


    actual_b_10ms = denorm_b_safe(
        actual_b_10ms_n
    )


    # ==============================================================
    # DIRECT 100 ms
    #
    # index 9 corresponds to t+10
    # ==============================================================

    pred_p_100ms_n = (
        out["future_p"][
            0,
            -1
        ]
        .cpu()
        .numpy()
    )


    pred_b_100ms_n = (
        out["future_b"][
            0,
            -1
        ]
        .cpu()
        .numpy()
    )


    actual_p_100ms_n = (
        batch["future_p"][
            0,
            -1
        ]
        .cpu()
        .numpy()
    )


    actual_b_100ms_n = (
        batch["future_b"][
            0,
            -1
        ]
        .cpu()
        .numpy()
    )


    pred_p_100ms = denorm_p_safe(
        pred_p_100ms_n
    )


    actual_p_100ms = denorm_p_safe(
        actual_p_100ms_n
    )


    pred_b_100ms = denorm_b_safe(
        pred_b_100ms_n
    )


    actual_b_100ms = denorm_b_safe(
        actual_b_100ms_n
    )


    # ==============================================================
    # CONTACT PROBABILITIES
    # ==============================================================

    contact_prob_10ms = (

        torch.sigmoid(
            out[
                "one_contact_logits"
            ][0]
        )

        .cpu()
        .numpy()
    )


    contact_prob_100ms = (

        torch.sigmoid(
            out[
                "future_contact_logits"
            ][
                0,
                -1
            ]
        )

        .cpu()
        .numpy()
    )


    actual_contact_10ms = (

        batch[
            "future_contact"
        ][
            0,
            0
        ]

        .cpu()
        .numpy()
    )


    actual_contact_100ms = (

        batch[
            "future_contact"
        ][
            0,
            -1
        ]

        .cpu()
        .numpy()
    )


    return {

        "t_index":
            t,

        "start_time":
            float(
                time[t]
            ),

        "time_10ms":
            float(
                time[t + 1]
            ),

        "time_100ms":
            float(
                time[t + 10]
            ),

        "pred_p_10ms":
            pred_p_10ms,

        "actual_p_10ms":
            actual_p_10ms,

        "pred_b_10ms":
            pred_b_10ms,

        "actual_b_10ms":
            actual_b_10ms,

        "pred_p_100ms":
            pred_p_100ms,

        "actual_p_100ms":
            actual_p_100ms,

        "pred_b_100ms":
            pred_b_100ms,

        "actual_b_100ms":
            actual_b_100ms,

        "contact_prob_10ms":
            contact_prob_10ms,

        "contact_prob_100ms":
            contact_prob_100ms,

        "actual_contact_10ms":
            actual_contact_10ms,

        "actual_contact_100ms":
            actual_contact_100ms
    }

# %% notebook cell 20
# ======================================================================
# V2 CELL 7
# RUN ALL FOUR CASES
# ======================================================================

V2_RESULTS = []


for case_number, requested_time in enumerate(
    TEST_CASE_TIMES,
    start=1
):

    result = evaluate_v2_case(
        requested_time
    )


    V2_RESULTS.append(
        result
    )


    print(
        "\n\n"
        + "#" * 100
    )

    print(
        f"TEST CASE {case_number}"
    )

    print(
        f"Initial state        : "
        f"{result['start_time']:.3f} s"
    )

    print(
        f"One-step prediction  : "
        f"{result['time_10ms']:.3f} s "
        f"(10 ms later)"
    )

    print(
        f"Direct prediction    : "
        f"{result['time_100ms']:.3f} s "
        f"(100 ms later)"
    )

    print(
        "#" * 100
    )


    # ==============================================================
    # t+1
    # ==============================================================

    show_complete_p_table(

        result[
            "pred_p_10ms"
        ],

        result[
            "actual_p_10ms"
        ],

        (
            f"TEST {case_number} — "
            f"PREDICTED vs ACTUAL P at "
            f"t+1 = 10 ms"
        )
    )


    # ==============================================================
    # t+10
    # ==============================================================

    show_complete_p_table(

        result[
            "pred_p_100ms"
        ],

        result[
            "actual_p_100ms"
        ],

        (
            f"TEST {case_number} — "
            f"PREDICTED vs ACTUAL P at "
            f"t+10 = 100 ms DIRECT"
        )
    )


    # ==============================================================
    # READABLE ERROR SUMMARY
    # ==============================================================

    errors_10 = get_group_rmse(

        result[
            "pred_p_10ms"
        ],

        result[
            "actual_p_10ms"
        ]
    )


    errors_100 = get_group_rmse(

        result[
            "pred_p_100ms"
        ],

        result[
            "actual_p_100ms"
        ]
    )


    group_table = pd.DataFrame({

        "Sensor group":
            list(
                errors_10.keys()
            ),

        "10 ms RMSE":
            list(
                errors_10.values()
            ),

        "100 ms direct RMSE":
            list(
                errors_100.values()
            )
    })


    print(
        "\nSENSOR-GROUP SUMMARY"
    )


    print(
        group_table.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}"
        )
    )


    # ==============================================================
    # CONTACT HEAD
    # ==============================================================

    contact_table = pd.DataFrame({

        "Leg":
            LEG_NAMES,

        "P(contact) 10ms":
            result[
                "contact_prob_10ms"
            ],

        "Actual contact 10ms":
            result[
                "actual_contact_10ms"
            ],

        "P(contact) 100ms":
            result[
                "contact_prob_100ms"
            ],

        "Actual contact 100ms":
            result[
                "actual_contact_100ms"
            ]
    })


    print(
        "\nCONTACT-STATE PREDICTION"
    )


    print(
        contact_table.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}"
        )
    )

# %% notebook cell 21
# ======================================================================
# V2 CELL 8
#
# V1 vs V2
#
# V1:
# 10 ms = one-step
# 100 ms = RECURSIVE
#
# V2:
# 10 ms = one-step
# 100 ms = DIRECT MULTI-STEP
# ======================================================================


PREVIOUS_V1 = {

    102.0: {
        "10ms": 1.429556,
        "100ms": 15.823002
    },

    108.0: {
        "10ms": 1.468045,
        "100ms": 11.640738
    },

    114.0: {
        "10ms": 1.416657,
        "100ms": 23.024879
    },

    119.0: {
        "10ms": 1.295269,
        "100ms": 26.768475
    }
}


comparison_rows = []


for requested_time, result in zip(
    TEST_CASE_TIMES,
    V2_RESULTS
):


    v2_10 = rmse(

        result[
            "pred_p_10ms"
        ],

        result[
            "actual_p_10ms"
        ]
    )


    v2_100 = rmse(

        result[
            "pred_p_100ms"
        ],

        result[
            "actual_p_100ms"
        ]
    )


    v1_10 = PREVIOUS_V1[
        requested_time
    ][
        "10ms"
    ]


    v1_100 = PREVIOUS_V1[
        requested_time
    ][
        "100ms"
    ]


    improvement_10 = (

        100.0
        *
        (
            v1_10
            -
            v2_10
        )
        /
        v1_10
    )


    improvement_100 = (

        100.0
        *
        (
            v1_100
            -
            v2_100
        )
        /
        v1_100
    )


    comparison_rows.append({

        "Start":
            requested_time,

        "V1 10ms":
            v1_10,

        "V2 10ms":
            v2_10,

        "10ms improvement %":
            improvement_10,

        "V1 recursive 100ms":
            v1_100,

        "V2 direct 100ms":
            v2_100,

        "100ms improvement %":
            improvement_100
    })


comparison_df = pd.DataFrame(
    comparison_rows
)


print(
    "\n"
    + "=" * 115
)

print(
    "PREVIOUS MODEL V1 vs NEW MODEL V2"
)

print(
    "=" * 115
)


print(

    comparison_df.to_string(

        index=False,

        float_format=lambda x:
            f"{x:.4f}"
    )
)


print(
    "\nAVERAGES"
)


print(
    "V1 10 ms average:",
    comparison_df[
        "V1 10ms"
    ].mean()
)


print(
    "V2 10 ms average:",
    comparison_df[
        "V2 10ms"
    ].mean()
)


print(
    "\nV1 recursive 100 ms average:",
    comparison_df[
        "V1 recursive 100ms"
    ].mean()
)


print(
    "V2 direct 100 ms average:",
    comparison_df[
        "V2 direct 100ms"
    ].mean()
)

# %% notebook cell 22
# ======================================================================
# V2 CELL 9
# GRAPH: OLD vs NEW
# ======================================================================

labels = [
    "102 s",
    "108 s",
    "114 s",
    "119 s"
]


x = np.arange(
    4
)


width = 0.35


# ----------------------------------------------------------------------
# 10 ms
# ----------------------------------------------------------------------

plt.figure(
    figsize=(9, 5)
)


plt.bar(

    x - width/2,

    comparison_df[
        "V1 10ms"
    ],

    width,

    label="V1"
)


plt.bar(

    x + width/2,

    comparison_df[
        "V2 10ms"
    ],

    width,

    label="V2"
)


plt.xticks(
    x,
    labels
)


plt.ylabel(
    "48D P-state RMSE"
)


plt.title(
    "10 ms Prediction: V1 vs V2"
)


plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.show()


# ----------------------------------------------------------------------
# 100 ms
# ----------------------------------------------------------------------

plt.figure(
    figsize=(9, 5)
)


plt.bar(

    x - width/2,

    comparison_df[
        "V1 recursive 100ms"
    ],

    width,

    label="V1 recursive"
)


plt.bar(

    x + width/2,

    comparison_df[
        "V2 direct 100ms"
    ],

    width,

    label="V2 direct"
)


plt.xticks(
    x,
    labels
)


plt.ylabel(
    "48D P-state RMSE"
)


plt.title(
    "100 ms Prediction: Recursive V1 vs Direct V2"
)


plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.show()

# %% notebook cell 23
# ======================================================================
# V2 CELL 7
# RUN ALL FOUR CASES
# ======================================================================

V2_RESULTS = []


for case_number, requested_time in enumerate(
    TEST_CASE_TIMES,
    start=1
):

    result = evaluate_v2_case(
        requested_time
    )


    V2_RESULTS.append(
        result
    )


    print(
        "\n\n"
        + "#" * 100
    )

    print(
        f"TEST CASE {case_number}"
    )

    print(
        f"Initial state        : "
        f"{result['start_time']:.3f} s"
    )

    print(
        f"One-step prediction  : "
        f"{result['time_10ms']:.3f} s "
        f"(10 ms later)"
    )

    print(
        f"Direct prediction    : "
        f"{result['time_100ms']:.3f} s "
        f"(100 ms later)"
    )

    print(
        "#" * 100
    )


    # ==============================================================
    # t+1
    # ==============================================================

    show_complete_p_table(

        result[
            "pred_p_10ms"
        ],

        result[
            "actual_p_10ms"
        ],

        (
            f"TEST {case_number} — "
            f"PREDICTED vs ACTUAL P at "
            f"t+1 = 10 ms"
        )
    )


    # ==============================================================
    # t+10
    # ==============================================================

    show_complete_p_table(

        result[
            "pred_p_100ms"
        ],

        result[
            "actual_p_100ms"
        ],

        (
            f"TEST {case_number} — "
            f"PREDICTED vs ACTUAL P at "
            f"t+10 = 100 ms DIRECT"
        )
    )


    # ==============================================================
    # READABLE ERROR SUMMARY
    # ==============================================================

    errors_10 = get_group_rmse(

        result[
            "pred_p_10ms"
        ],

        result[
            "actual_p_10ms"
        ]
    )


    errors_100 = get_group_rmse(

        result[
            "pred_p_100ms"
        ],

        result[
            "actual_p_100ms"
        ]
    )


    group_table = pd.DataFrame({

        "Sensor group":
            list(
                errors_10.keys()
            ),

        "10 ms RMSE":
            list(
                errors_10.values()
            ),

        "100 ms direct RMSE":
            list(
                errors_100.values()
            )
    })


    print(
        "\nSENSOR-GROUP SUMMARY"
    )


    print(
        group_table.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}"
        )
    )


    # ==============================================================
    # CONTACT HEAD
    # ==============================================================

    contact_table = pd.DataFrame({

        "Leg":
            LEG_NAMES,

        "P(contact) 10ms":
            result[
                "contact_prob_10ms"
            ],

        "Actual contact 10ms":
            result[
                "actual_contact_10ms"
            ],

        "P(contact) 100ms":
            result[
                "contact_prob_100ms"
            ],

        "Actual contact 100ms":
            result[
                "actual_contact_100ms"
            ]
    })


    print(
        "\nCONTACT-STATE PREDICTION"
    )


    print(
        contact_table.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}"
        )
    )

# %% notebook cell 24
# ============================================================
# ONE-STEP CHRONOLOGICAL VISUO/PROPRIO PIPELINE
#
# Past 50 samples (0.5 s) ending at t
# + command U[t]
#          ↓
# predict physical response at t+1 = 10 ms later
#
# STRICTLY CHRONOLOGICAL:
# predict -> reveal actual -> MSE -> update -> shift by 1
#
# FLAT MLP COUPLING ONLY
# NO GRAPH
# NO SHUFFLING
# NO DATALOADER
# NO MULTIPLE EPOCHS
# ============================================================

import numpy as np
import torch
import torch.nn as nn
import random
import math

# ------------------------------------------------------------
# 0. Reproducibility + device
# ------------------------------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ------------------------------------------------------------
# 1. CHECK EXISTING RAW DATA
# ------------------------------------------------------------

assert "P" in globals(), "P is missing"
assert "B" in globals(), "B is missing"
assert "U" in globals(), "U is missing"
assert "time" in globals(), "time is missing"

P = np.asarray(P, dtype=np.float32)
B = np.asarray(B, dtype=np.float32)
U = np.asarray(U, dtype=np.float32)
time = np.asarray(time, dtype=np.float64)

assert len(P) == len(B) == len(U) == len(time)

print("\nRaw data:")
print("P   :", P.shape)
print("B   :", B.shape)
print("U   :", U.shape)
print("time:", time.shape)

dt = np.median(np.diff(time))

print("\nMedian dt:", dt, "seconds")
print("Sampling frequency:", 1.0 / dt, "Hz")

if not np.isclose(dt, 0.01, atol=1e-3):
    print("WARNING: dt is not approximately 0.01 s")


# ------------------------------------------------------------
# 2. EXPERIMENT SETTINGS
# ------------------------------------------------------------

T = 50                     # 50 samples = 0.5 s history
TRAIN_END_TIME = 87.0      # chronological training until 87 s

# We only accept transitions for which the high-level
# command is effectively unchanged between t and t+1.
CMD_TOL = 1e-6

# During testing:
#
# prediction at t+1 happens FIRST.
# Then actual t+1 is revealed.
# Then model is allowed to update from that result
# before predicting the next point.
#
ONLINE_UPDATE_DURING_TEST = True

LR = 1e-4
WEIGHT_DECAY = 1e-4

D_MODEL = 128
NHEAD = 4
NUM_TEMPORAL_LAYERS = 2

# dropout = 0 makes the sequential online experiment
# deterministic/stabler than stochastic dropout at batch size 1
DROPOUT = 0.0


# ------------------------------------------------------------
# 3. CHECK COMMAND STABILITY
# ------------------------------------------------------------

command_change = np.max(
    np.abs(U[1:] - U[:-1]),
    axis=1
)

stable_transition = command_change <= CMD_TOL

stable_fraction = stable_transition.mean()

print(
    "\nUnchanged-command transitions:",
    int(stable_transition.sum()),
    "/",
    len(stable_transition)
)

print(
    "Fraction unchanged:",
    f"{100*stable_fraction:.2f}%"
)

# This catches the old LINEAR interpolation version of U.
if stable_transition.sum() < 500:
    raise RuntimeError(
        "\nToo few unchanged-command transitions.\n"
        "Your U array may still be linearly interpolated.\n"
        "Use the ZOH command synchronization first, then rebuild/reload P,B,U,time."
    )


# ------------------------------------------------------------
# 4. CHRONOLOGICAL TRAIN / TEST BOUNDARY
# ------------------------------------------------------------

train_end_idx = np.searchsorted(
    time,
    TRAIN_END_TIME,
    side="right"
) - 1

print("\nTrain boundary:")
print("index:", train_end_idx)
print("time :", time[train_end_idx])

print(
    "\nTraining transitions will end at <= %.3f s"
    % time[train_end_idx]
)

print(
    "Walk-forward testing begins immediately after that."
)


# ------------------------------------------------------------
# 5. NORMALIZATION
#
# IMPORTANT:
# statistics use TRAINING REGION ONLY.
# Nothing from future test region enters normalization.
# ------------------------------------------------------------

eps = 1e-6

train_P = P[:train_end_idx + 1]
train_B = B[:train_end_idx + 1]
train_U = U[:train_end_idx + 1]

# P feature normalization:
# average over time + four legs
leg_mean = train_P.mean(
    axis=(0, 1)
)

leg_std = train_P.std(
    axis=(0, 1)
) + eps

body_mean = train_B.mean(
    axis=0
)

body_std = train_B.std(
    axis=0
) + eps

cmd_mean = train_U.mean(
    axis=0
)

cmd_std = train_U.std(
    axis=0
) + eps

# Target variables:
# 12 foot-force numbers + 6 body-twist numbers

train_target_state = np.concatenate(
    [
        train_P[:, :, 9:12].reshape(-1, 12),
        train_B[:, 6:12]
    ],
    axis=1
)

y_mean = train_target_state.mean(
    axis=0
)

y_std = train_target_state.std(
    axis=0
) + eps

print("\nNormalization ready.")


# ------------------------------------------------------------
# 6. HELPER:
# MAKE EXACTLY ONE t -> t+1 EXAMPLE
# ------------------------------------------------------------

def make_one_step_example(t):

    # ------------------------------------------------
    # history:
    # t-49, ..., t
    # exactly 50 chronological samples
    # ------------------------------------------------

    leg_history = P[
        t-T+1 : t+1
    ]

    # current shape:
    # 50 x 4 x 12

    leg_history = np.transpose(
        leg_history,
        (1, 0, 2)
    )

    # 4 x 50 x 12

    body_history = B[
        t-T+1 : t+1
    ]

    # 50 x 12

    command = U[t]

    # ------------------------------------------------
    # GROUND TRUTH:
    # EXACT NEXT SAMPLE ONLY
    # ------------------------------------------------

    next_forces = P[
        t+1,
        :,
        9:12
    ].reshape(-1)

    next_body_twist = B[
        t+1,
        6:12
    ]

    target = np.concatenate(
        [
            next_forces,
            next_body_twist
        ]
    ).astype(np.float32)

    # ------------------------------------------------
    # NORMALIZE
    # ------------------------------------------------

    leg_n = (
        leg_history - leg_mean[None, None, :]
    ) / leg_std[None, None, :]

    body_n = (
        body_history - body_mean[None, :]
    ) / body_std[None, :]

    cmd_n = (
        command - cmd_mean
    ) / cmd_std

    target_n = (
        target - y_mean
    ) / y_std

    # add batch dimension = 1

    leg_t = torch.tensor(
        leg_n,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)

    body_t = torch.tensor(
        body_n,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)

    cmd_t = torch.tensor(
        cmd_n,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)

    target_t = torch.tensor(
        target_n,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)

    return (
        leg_t,
        body_t,
        cmd_t,
        target_t,
        target
    )


# ============================================================
# 7. SHARED TEMPORAL TRANSFORMER FOR LEGS
# ============================================================

class OneStepLegTransformer(nn.Module):

    def __init__(
        self,
        input_dim=12,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_ff=256,
        max_len=50,
        dropout=0.0
    ):
        super().__init__()

        self.d_model = d_model

        self.input_proj = nn.Linear(
            input_dim,
            d_model
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(
                1,
                max_len,
                d_model
            ) * 0.02
        )

        # FL FR RL RR identities
        self.leg_embedding = nn.Embedding(
            4,
            d_model
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu"
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(
            d_model
        )

    def forward(self, x):

        # x:
        # BATCH x 4 x 50 x 12

        batch_size, legs, T_now, features = x.shape

        # same Transformer applied to all four legs
        x = x.reshape(
            batch_size * legs,
            T_now,
            features
        )

        x = self.input_proj(x)

        x = (
            x
            + self.pos_embedding[:, :T_now]
        )

        leg_ids = torch.arange(
            legs,
            device=x.device
        ).repeat(batch_size)

        leg_emb = self.leg_embedding(
            leg_ids
        ).unsqueeze(1)

        x = x + leg_emb

        # causal temporal attention:
        # a time point cannot look forward
        causal_mask = torch.triu(
            torch.ones(
                T_now,
                T_now,
                dtype=torch.bool,
                device=x.device
            ),
            diagonal=1
        )

        x = self.transformer(
            x,
            mask=causal_mask
        )

        x = self.norm(x)

        # use representation at latest observed time t
        h = x[:, -1, :]

        h = h.reshape(
            batch_size,
            legs,
            self.d_model
        )

        return h


# ============================================================
# 8. BODY TEMPORAL TRANSFORMER
# ============================================================

class OneStepBodyTransformer(nn.Module):

    def __init__(
        self,
        input_dim=12,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_ff=256,
        max_len=50,
        dropout=0.0
    ):
        super().__init__()

        self.input_proj = nn.Linear(
            input_dim,
            d_model
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(
                1,
                max_len,
                d_model
            ) * 0.02
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu"
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(
            d_model
        )

    def forward(self, x):

        # BATCH x 50 x 12

        _, T_now, _ = x.shape

        x = self.input_proj(x)

        x = (
            x
            + self.pos_embedding[:, :T_now]
        )

        causal_mask = torch.triu(
            torch.ones(
                T_now,
                T_now,
                dtype=torch.bool,
                device=x.device
            ),
            diagonal=1
        )

        x = self.transformer(
            x,
            mask=causal_mask
        )

        x = self.norm(x)

        return x[:, -1, :]


# ============================================================
# 9. FLAT MLP COUPLING MODEL
# ============================================================

class OneStepFlatMLP(nn.Module):

    def __init__(
        self,
        d_model=128,
        nhead=4,
        temporal_layers=2,
        dropout=0.0,
        coupling_hidden=208
    ):
        super().__init__()

        # SAME temporal model for every leg
        self.leg_encoder = OneStepLegTransformer(
            input_dim=12,
            d_model=d_model,
            nhead=nhead,
            num_layers=temporal_layers,
            dim_ff=2*d_model,
            max_len=T,
            dropout=dropout
        )

        self.body_encoder = OneStepBodyTransformer(
            input_dim=12,
            d_model=d_model,
            nhead=nhead,
            num_layers=temporal_layers,
            dim_ff=2*d_model,
            max_len=T,
            dropout=dropout
        )

        self.command_encoder = nn.Sequential(
            nn.Linear(6, 64),
            nn.GELU(),
            nn.Linear(64, d_model)
        )

        self.leg_node_proj = nn.Sequential(
            nn.Linear(
                2*d_model,
                d_model
            ),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )

        self.body_node_proj = nn.Sequential(
            nn.Linear(
                2*d_model,
                d_model
            ),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )

        # ----------------------------------------------------
        # FLAT COUPLING:
        #
        # BODY + FL + FR + RL + RR
        # 5 x 128 = 640
        # ----------------------------------------------------

        flat_dim = 5 * d_model

        self.flat_norm = nn.LayerNorm(
            flat_dim
        )

        self.coupling_mlp = nn.Sequential(
            nn.Linear(
                flat_dim,
                coupling_hidden
            ),
            nn.GELU(),
            nn.Linear(
                coupling_hidden,
                flat_dim
            )
        )

        # same force head for four legs
        self.leg_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 3)
        )

        self.body_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 6)
        )

    def forward(
        self,
        x_leg,
        x_body,
        x_cmd
    ):

        batch_size = x_leg.shape[0]

        # temporal history encoding
        leg_h = self.leg_encoder(
            x_leg
        )

        body_h = self.body_encoder(
            x_body
        )

        cmd_h = self.command_encoder(
            x_cmd
        )

        # command information for all legs
        cmd_legs = cmd_h.unsqueeze(1).expand(
            -1,
            4,
            -1
        )

        leg_nodes = self.leg_node_proj(
            torch.cat(
                [
                    leg_h,
                    cmd_legs
                ],
                dim=-1
            )
        )

        body_node = self.body_node_proj(
            torch.cat(
                [
                    body_h,
                    cmd_h
                ],
                dim=-1
            )
        ).unsqueeze(1)

        # order:
        # BODY FL FR RL RR

        nodes = torch.cat(
            [
                body_node,
                leg_nodes
            ],
            dim=1
        )

        # B x 5 x 128

        flat = nodes.reshape(
            batch_size,
            -1
        )

        # generic global inter-leg/body coupling
        update = self.coupling_mlp(
            self.flat_norm(flat)
        )

        coupled = flat + update

        coupled_nodes = coupled.reshape(
            batch_size,
            5,
            -1
        )

        body_final = coupled_nodes[:, 0]
        leg_final = coupled_nodes[:, 1:5]

        leg_force = self.leg_head(
            leg_final
        ).reshape(
            batch_size,
            12
        )

        body_twist = self.body_head(
            body_final
        )

        # 18 values:
        # FL3 FR3 RL3 RR3 body_linear3 body_angular3
        return torch.cat(
            [
                leg_force,
                body_twist
            ],
            dim=-1
        )


# ------------------------------------------------------------
# 10. CREATE A COMPLETELY NEW MODEL
# ------------------------------------------------------------

one_model = OneStepFlatMLP(
    d_model=D_MODEL,
    nhead=NHEAD,
    temporal_layers=NUM_TEMPORAL_LAYERS,
    dropout=DROPOUT,
    coupling_hidden=208
).to(device)

n_params = sum(
    p.numel()
    for p in one_model.parameters()
)

print(
    "\nOne-step Flat MLP parameters:",
    f"{n_params:,}"
)

criterion = nn.MSELoss()

optimizer = torch.optim.AdamW(
    one_model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)


# ------------------------------------------------------------
# 11. STRICT CHRONOLOGICAL TRAINING
#
# NO batches.
# NO shuffle.
# NO second epoch.
#
# t -> predict t+1 -> see truth -> update
# ------------------------------------------------------------

train_losses = []
trained_steps = 0

print("\nStarting chronological training...\n")

# Last training target must remain inside training region,
# therefore t <= train_end_idx - 1

for t in range(
    T - 1,
    train_end_idx
):

    # only use transitions where command did not change
    if not stable_transition[t]:
        continue

    (
        leg_t,
        body_t,
        cmd_t,
        target_t,
        target_raw
    ) = make_one_step_example(t)

    one_model.train()

    optimizer.zero_grad()

    pred = one_model(
        leg_t,
        body_t,
        cmd_t
    )

    loss = criterion(
        pred,
        target_t
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        one_model.parameters(),
        max_norm=1.0
    )

    optimizer.step()

    train_losses.append(
        loss.item()
    )

    trained_steps += 1

    if trained_steps % 500 == 0:

        recent = np.mean(
            train_losses[-500:]
        )

        print(
            f"Step {trained_steps:5d} | "
            f"time {time[t]:7.2f}s -> {time[t+1]:7.2f}s | "
            f"recent MSE {recent:.6f}"
        )


print("\nTraining finished.")

print(
    "Chronological parameter updates:",
    trained_steps
)

print(
    "Mean training normalized MSE:",
    np.mean(train_losses)
)


# ============================================================
# 12. WALK-FORWARD TESTING
#
# CRITICAL ORDER:
#
# 1. model sees information only through t
# 2. model predicts t+1
# 3. SAVE prediction
# 4. reveal actual t+1
# 5. calculate error
# 6. ONLY THEN update model
# 7. move forward one sample
# ============================================================

predictions = []
actuals = []
persistence_predictions = []

test_times = []
test_source_indices = []

online_losses = []

print("\nStarting walk-forward test...\n")

for t in range(
    train_end_idx,
    len(P) - 1
):

    if not stable_transition[t]:
        continue

    (
        leg_t,
        body_t,
        cmd_t,
        target_t,
        target_raw
    ) = make_one_step_example(t)

    # ------------------------------------------------
    # FIRST: PREDICT.
    # The model has NOT seen target t+1.
    # ------------------------------------------------

    one_model.eval()

    with torch.no_grad():

        pred_n = one_model(
            leg_t,
            body_t,
            cmd_t
        )

    pred_n_np = (
        pred_n
        .cpu()
        .numpy()[0]
    )

    # back into physical units
    pred_raw = (
        pred_n_np * y_std
        + y_mean
    )

    # ------------------------------------------------
    # NOW record actual t+1.
    # This happens AFTER prediction.
    # ------------------------------------------------

    actual_raw = target_raw.copy()

    predictions.append(
        pred_raw
    )

    actuals.append(
        actual_raw
    )

    test_times.append(
        time[t+1]
    )

    test_source_indices.append(
        t
    )

    # ------------------------------------------------
    # Persistence baseline:
    # "next state = current state"
    # ------------------------------------------------

    current_force = P[
        t,
        :,
        9:12
    ].reshape(-1)

    current_body_twist = B[
        t,
        6:12
    ]

    persistence = np.concatenate(
        [
            current_force,
            current_body_twist
        ]
    )

    persistence_predictions.append(
        persistence
    )

    # ------------------------------------------------
    # ONLY NOW, after scoring/revealing t+1,
    # optionally update using this newly observed truth.
    # ------------------------------------------------

    if ONLINE_UPDATE_DURING_TEST:

        one_model.train()

        optimizer.zero_grad()

        update_pred = one_model(
            leg_t,
            body_t,
            cmd_t
        )

        update_loss = criterion(
            update_pred,
            target_t
        )

        update_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            one_model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        online_losses.append(
            update_loss.item()
        )


predictions = np.asarray(
    predictions,
    dtype=np.float32
)

actuals = np.asarray(
    actuals,
    dtype=np.float32
)

persistence_predictions = np.asarray(
    persistence_predictions,
    dtype=np.float32
)

test_times = np.asarray(
    test_times
)

print(
    "Walk-forward predictions:",
    predictions.shape
)

print(
    "Actual values:",
    actuals.shape
)


# ------------------------------------------------------------
# 13. PRINT MULTIPLE REAL TEST EXAMPLES
# ------------------------------------------------------------

names_3d = [
    ("FL force", 0, 3),
    ("FR force", 3, 6),
    ("RL force", 6, 9),
    ("RR force", 9, 12),
    ("Body linear velocity", 12, 15),
    ("Body angular velocity", 15, 18),
]

N_SHOW = min(
    8,
    len(predictions)
)

show_ids = np.linspace(
    0,
    len(predictions) - 1,
    N_SHOW,
    dtype=int
)

print("\n")
print("=" * 80)
print("EXAMPLE NEXT-STEP PREDICTIONS")
print("=" * 80)

for j, rid in enumerate(
    show_ids,
    start=1
):

    source_t = test_source_indices[rid]

    print(
        f"\nEXAMPLE {j}"
    )

    print(
        f"Input ends at: {time[source_t]:.3f} s"
    )

    print(
        f"Predicting   : {test_times[rid]:.3f} s"
    )

    print(
        f"dt           : "
        f"{test_times[rid] - time[source_t]:.4f} s"
    )

    print(
        "max command change:",
        np.max(
            np.abs(
                U[source_t+1]
                - U[source_t]
            )
        )
    )

    for name, a, b in names_3d:

        p = predictions[
            rid,
            a:b
        ]

        y = actuals[
            rid,
            a:b
        ]

        print(
            f"\n{name}"
        )

        print(
            " predicted:",
            np.round(p, 5)
        )

        print(
            " actual   :",
            np.round(y, 5)
        )

        print(
            " error    :",
            np.round(p - y, 5)
        )

    print(
        "\n" + "-" * 80
    )


# ------------------------------------------------------------
# 14. AGGREGATE PREQUENTIAL RMSE
# ------------------------------------------------------------

def rmse(
    pred,
    true
):
    return np.sqrt(
        np.mean(
            (pred - true) ** 2
        )
    )


print("\n")
print("=" * 80)
print("ONE-STEP WALK-FORWARD MODEL RMSE")
print("=" * 80)

for name, a, b in names_3d:

    value = rmse(
        predictions[:, a:b],
        actuals[:, a:b]
    )

    print(
        f"{name:25s}: {value:.6f}"
    )


# ------------------------------------------------------------
# 15. IMPORTANT 10-ms PERSISTENCE BASELINE
#
# At only 10 ms, persistence may be VERY strong.
# Our learned model should beat it to be interesting.
# ------------------------------------------------------------

print("\n")
print("=" * 80)
print("ONE-STEP PERSISTENCE RMSE")
print("=" * 80)

for name, a, b in names_3d:

    value = rmse(
        persistence_predictions[:, a:b],
        actuals[:, a:b]
    )

    print(
        f"{name:25s}: {value:.6f}"
    )


print("\nDone.")

# %% notebook cell 25
# ======================================================================
# COMPLETE 0.1 s RECURSIVE ROLLOUT EXPERIMENT
#
# ASSUMES THESE ALREADY EXIST:
#
# P    : (N, 4, 12)
# B    : (N, 12)
# U    : (N, 6)
# time : (N,)
#
# ----------------------------------------------------------------------
# TRAINING EXAMPLE:
#
# actual states:
# [t-49 ... t]
#
# then:
#
# state history + U[t]   -> predicted t+1
# predicted t+1 + U[t+1] -> predicted t+2
# predicted t+2 + U[t+2] -> predicted t+3
# ...
# predicted t+9 + U[t+9] -> predicted t+10
#
# NO ACTUAL t+1 ... t+9 STATES ARE USED.
#
# ONLY after t+10:
#
# predicted full state at t+10
#          vs
# actual full state at t+10
#
# -> MSE
# -> backprop through ALL 10 rollout steps
#
# ----------------------------------------------------------------------
# TRAINING:
#
# For each requested test time:
# use ONLY the previous 20 seconds
#
# chronological order only
# no shuffling
# one epoch
# batches contain consecutive chronological examples
#
# ----------------------------------------------------------------------
# TEST TIMES:
#
# ~100 s
# ~106 s
# ~112 s
# ~118 s
#
# Each test gets a fresh model trained only on the previous 20 s.
# ======================================================================


# ======================================================================
# 1. IMPORTS
# ======================================================================

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import gc


# ======================================================================
# 2. CHECK DATA
# ======================================================================

P = np.asarray(P, dtype=np.float32)
B = np.asarray(B, dtype=np.float32)
U = np.asarray(U, dtype=np.float32)
time = np.asarray(time, dtype=np.float64)

assert P.ndim == 3
assert P.shape[1:] == (4, 12)

assert B.ndim == 2
assert B.shape[1] == 12

assert U.ndim == 2
assert U.shape[1] == 6

assert len(P) == len(B) == len(U) == len(time)


print("P   :", P.shape)
print("B   :", B.shape)
print("U   :", U.shape)
print("time:", time.shape)


dt = float(
    np.median(
        np.diff(time)
    )
)

print("dt:", dt)

print(
    "frequency:",
    1.0 / dt,
    "Hz"
)


# ======================================================================
# 3. SETTINGS
# ======================================================================

SEQUENCE_LENGTH = 50

# 10 x 0.01 s = 0.1 s
ROLLOUT_STEPS = 10

TRAIN_HISTORY_SECONDS = 20.0


TEST_TIMES = [
    100.0,
    106.0,
    112.0,
    118.0
]


# consecutive chronological examples per gradient update
BATCH_SIZE = 60


D_MODEL = 128

NHEAD = 4

TEMPORAL_LAYERS = 2

DROPOUT = 0.0


LR = 1e-4

WEIGHT_DECAY = 1e-4


EPS = 1e-6


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    "\nDevice:",
    device
)

print(
    "History:",
    SEQUENCE_LENGTH * dt,
    "seconds"
)

print(
    "Recursive horizon:",
    ROLLOUT_STEPS * dt,
    "seconds"
)


# ======================================================================
# 4. NORMALIZATION
#
# For each test experiment:
#
# normalization statistics come ONLY from its previous 20 s.
# ======================================================================

def make_stats(
    start_idx,
    end_idx
):

    # end_idx exclusive

    p_region = P[
        start_idx:end_idx
    ]

    b_region = B[
        start_idx:end_idx
    ]

    u_region = U[
        start_idx:end_idx
    ]


    # same feature type shared across four legs

    p_mean = p_region.mean(
        axis=(0, 1)
    )

    p_std = p_region.std(
        axis=(0, 1)
    ) + EPS


    b_mean = b_region.mean(
        axis=0
    )

    b_std = b_region.std(
        axis=0
    ) + EPS


    u_mean = u_region.mean(
        axis=0
    )

    u_std = u_region.std(
        axis=0
    ) + EPS


    return {

        "p_mean":
            p_mean.astype(np.float32),

        "p_std":
            p_std.astype(np.float32),

        "b_mean":
            b_mean.astype(np.float32),

        "b_std":
            b_std.astype(np.float32),

        "u_mean":
            u_mean.astype(np.float32),

        "u_std":
            u_std.astype(np.float32),
    }


def normalize_p(
    x,
    stats
):

    return (
        x
        - stats["p_mean"]
    ) / stats["p_std"]


def denormalize_p(
    x,
    stats
):

    return (
        x
        * stats["p_std"]
        + stats["p_mean"]
    )


def normalize_b(
    x,
    stats
):

    return (
        x
        - stats["b_mean"]
    ) / stats["b_std"]


def denormalize_b(
    x,
    stats
):

    return (
        x
        * stats["b_std"]
        + stats["b_mean"]
    )


def normalize_u(
    x,
    stats
):

    return (
        x
        - stats["u_mean"]
    ) / stats["u_std"]


# ======================================================================
# 5. SHARED LEG TEMPORAL TRANSFORMER
# ======================================================================

class RolloutLegTransformer(nn.Module):

    def __init__(
        self,
        input_dim=12,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_ff=256,
        max_len=50,
        dropout=0.0
    ):

        super().__init__()


        self.d_model = d_model


        self.input_proj = nn.Linear(
            input_dim,
            d_model
        )


        self.pos_embedding = nn.Parameter(

            torch.randn(
                1,
                max_len,
                d_model
            ) * 0.02
        )


        # FL FR RL RR identity

        self.leg_embedding = nn.Embedding(
            4,
            d_model
        )


        layer = nn.TransformerEncoderLayer(

            d_model=d_model,

            nhead=nhead,

            dim_feedforward=dim_ff,

            dropout=dropout,

            batch_first=True,

            norm_first=True,

            activation="gelu"
        )


        self.transformer = nn.TransformerEncoder(

            layer,

            num_layers=num_layers
        )


        self.norm = nn.LayerNorm(
            d_model
        )


    def forward(
        self,
        x
    ):

        # x:
        #
        # batch x 4 x 50 x 12

        batch_size, legs, T, F = (
            x.shape
        )


        # same Transformer weights
        # independently applied to each leg

        x = x.reshape(

            batch_size * legs,

            T,

            F
        )


        x = self.input_proj(
            x
        )


        # time positions inside the 50-step sequence

        x = (

            x

            + self.pos_embedding[
                :,
                :T
            ]
        )


        leg_ids = torch.arange(

            legs,

            device=x.device

        ).repeat(
            batch_size
        )


        leg_emb = (

            self.leg_embedding(
                leg_ids
            )

            .unsqueeze(1)
        )


        x = (
            x
            + leg_emb
        )


        # causal attention:
        #
        # timestep i cannot see i+1 ...

        causal_mask = torch.triu(

            torch.ones(

                T,
                T,

                dtype=torch.bool,

                device=x.device
            ),

            diagonal=1
        )


        x = self.transformer(

            x,

            mask=causal_mask
        )


        x = self.norm(
            x
        )


        # latest temporal representation

        h = x[
            :,
            -1,
            :
        ]


        h = h.reshape(

            batch_size,

            legs,

            self.d_model
        )


        return h


# ======================================================================
# 6. BODY TEMPORAL TRANSFORMER
# ======================================================================

class RolloutBodyTransformer(nn.Module):

    def __init__(
        self,
        input_dim=12,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_ff=256,
        max_len=50,
        dropout=0.0
    ):

        super().__init__()


        self.input_proj = nn.Linear(
            input_dim,
            d_model
        )


        self.pos_embedding = nn.Parameter(

            torch.randn(
                1,
                max_len,
                d_model
            ) * 0.02
        )


        layer = nn.TransformerEncoderLayer(

            d_model=d_model,

            nhead=nhead,

            dim_feedforward=dim_ff,

            dropout=dropout,

            batch_first=True,

            norm_first=True,

            activation="gelu"
        )


        self.transformer = nn.TransformerEncoder(

            layer,

            num_layers=num_layers
        )


        self.norm = nn.LayerNorm(
            d_model
        )


    def forward(
        self,
        x
    ):

        # x:
        # batch x 50 x 12

        batch_size, T, F = (
            x.shape
        )


        x = self.input_proj(
            x
        )


        x = (

            x

            + self.pos_embedding[
                :,
                :T
            ]
        )


        causal_mask = torch.triu(

            torch.ones(

                T,
                T,

                dtype=torch.bool,

                device=x.device
            ),

            diagonal=1
        )


        x = self.transformer(

            x,

            mask=causal_mask
        )


        x = self.norm(
            x
        )


        return x[
            :,
            -1,
            :
        ]


# ======================================================================
# 7. FLAT-MLP COUPLED ONE-STEP DYNAMICS MODEL
#
# IMPORTANT:
#
# To recursively predict t+2 after t+1,
# we need the COMPLETE predicted state.
#
# Therefore output:
#
# 4 legs x 12 = 48
# body          = 12
#
# total complete next state = 60 values.
#
# Model predicts DELTA:
#
# next_state =
# current_state + learned_change
# ======================================================================

class RecursiveFlatMLPModel(nn.Module):

    def __init__(
        self,
        d_model=128,
        nhead=4,
        temporal_layers=2,
        coupling_hidden=208,
        dropout=0.0
    ):

        super().__init__()


        self.leg_encoder = RolloutLegTransformer(

            input_dim=12,

            d_model=d_model,

            nhead=nhead,

            num_layers=temporal_layers,

            dim_ff=2*d_model,

            max_len=SEQUENCE_LENGTH,

            dropout=dropout
        )


        self.body_encoder = RolloutBodyTransformer(

            input_dim=12,

            d_model=d_model,

            nhead=nhead,

            num_layers=temporal_layers,

            dim_ff=2*d_model,

            max_len=SEQUENCE_LENGTH,

            dropout=dropout
        )


        self.command_encoder = nn.Sequential(

            nn.Linear(
                6,
                64
            ),

            nn.GELU(),

            nn.Linear(
                64,
                d_model
            )
        )


        # combine each leg token
        # with current command

        self.leg_node_proj = nn.Sequential(

            nn.Linear(
                2*d_model,
                d_model
            ),

            nn.GELU(),

            nn.LayerNorm(
                d_model
            )
        )


        # combine body token
        # with command

        self.body_node_proj = nn.Sequential(

            nn.Linear(
                2*d_model,
                d_model
            ),

            nn.GELU(),

            nn.LayerNorm(
                d_model
            )
        )


        # BODY FL FR RL RR
        #
        # 5 x 128 = 640

        flat_dim = (
            5 * d_model
        )


        self.flat_norm = nn.LayerNorm(
            flat_dim
        )


        self.coupling_mlp = nn.Sequential(

            nn.Linear(
                flat_dim,
                coupling_hidden
            ),

            nn.GELU(),

            nn.Linear(
                coupling_hidden,
                flat_dim
            )
        )


        # each leg:
        #
        # predict delta of
        # q3 dq3 tau3 force3 = 12

        self.leg_delta_head = nn.Sequential(

            nn.Linear(
                d_model,
                64
            ),

            nn.GELU(),

            nn.Linear(
                64,
                12
            )
        )


        # complete body:
        #
        # acc3 gyro3 twist6 = 12

        self.body_delta_head = nn.Sequential(

            nn.Linear(
                d_model,
                64
            ),

            nn.GELU(),

            nn.Linear(
                64,
                12
            )
        )


    def forward(
        self,
        p_history,
        b_history,
        command
    ):

        batch_size = (
            p_history.shape[0]
        )


        # temporal features

        leg_h = self.leg_encoder(
            p_history
        )


        body_h = self.body_encoder(
            b_history
        )


        cmd_h = self.command_encoder(
            command
        )


        cmd_for_legs = (

            cmd_h

            .unsqueeze(1)

            .expand(
                -1,
                4,
                -1
            )
        )


        leg_nodes = self.leg_node_proj(

            torch.cat(

                [
                    leg_h,
                    cmd_for_legs
                ],

                dim=-1
            )
        )


        body_node = (

            self.body_node_proj(

                torch.cat(

                    [
                        body_h,
                        cmd_h
                    ],

                    dim=-1
                )
            )

            .unsqueeze(1)
        )


        # BODY FL FR RL RR

        nodes = torch.cat(

            [
                body_node,
                leg_nodes
            ],

            dim=1
        )


        flat = nodes.reshape(

            batch_size,

            -1
        )


        # generic coupling across robot

        coupling_update = self.coupling_mlp(

            self.flat_norm(
                flat
            )
        )


        coupled_flat = (

            flat
            + coupling_update
        )


        coupled_nodes = (

            coupled_flat

            .reshape(
                batch_size,
                5,
                -1
            )
        )


        body_final = coupled_nodes[
            :,
            0
        ]


        leg_final = coupled_nodes[
            :,
            1:5
        ]


        delta_p = self.leg_delta_head(
            leg_final
        )


        delta_b = self.body_delta_head(
            body_final
        )


        # last physical state in history

        current_p = p_history[
            :,
            :,
            -1,
            :
        ]


        current_b = b_history[
            :,
            -1,
            :
        ]


        # residual transition

        next_p = (
            current_p
            + delta_p
        )


        next_b = (
            current_b
            + delta_b
        )


        return (
            next_p,
            next_b
        )


# ======================================================================
# 8. RECURSIVE ROLLOUT
#
# THIS IS THE CORE.
#
# Actual intermediate states are NEVER inserted.
#
# command_sequence contains:
#
# U[t]
# U[t+1]
# ...
# U[t+9]
#
# They are known conditioning inputs.
# ======================================================================

def recursive_rollout(
    model,
    p_history,
    b_history,
    command_sequence,
    steps=10,
    return_all=False
):

    p_hist = p_history

    b_hist = b_history


    predicted_p_steps = []

    predicted_b_steps = []


    for k in range(
        steps
    ):


        # --------------------------------------------------------------
        # command for THIS transition
        # --------------------------------------------------------------

        current_command = command_sequence[
            :,
            k,
            :
        ]


        # --------------------------------------------------------------
        # predict next COMPLETE state
        # --------------------------------------------------------------

        next_p, next_b = model(

            p_hist,

            b_hist,

            current_command
        )


        predicted_p_steps.append(
            next_p
        )


        predicted_b_steps.append(
            next_b
        )


        # --------------------------------------------------------------
        # SHIFT HISTORY
        #
        # remove oldest real/predicted state
        #
        # append PREDICTED next state
        #
        # NO TEACHER FORCING
        # --------------------------------------------------------------

        p_hist = torch.cat(

            [

                p_hist[
                    :,
                    :,
                    1:,
                    :
                ],

                next_p.unsqueeze(
                    2
                )

            ],

            dim=2
        )


        b_hist = torch.cat(

            [

                b_hist[
                    :,
                    1:,
                    :
                ],

                next_b.unsqueeze(
                    1
                )

            ],

            dim=1
        )


    predicted_p_steps = torch.stack(

        predicted_p_steps,

        dim=1
    )


    predicted_b_steps = torch.stack(

        predicted_b_steps,

        dim=1
    )


    # shapes:
    #
    # P:
    # batch x 10 x 4 x 12
    #
    # B:
    # batch x 10 x 12


    if return_all:

        return (

            next_p,

            next_b,

            predicted_p_steps,

            predicted_b_steps
        )


    return (
        next_p,
        next_b
    )


# ======================================================================
# 9. CREATE TRAINING ANCHORS
#
# Strict chronological sequence.
#
# Every possible start is used ONCE.
#
# No shuffle.
# ======================================================================

def make_training_anchors(
    train_start_idx,
    test_idx
):


    first_anchor = max(

        train_start_idx
        + SEQUENCE_LENGTH
        - 1,

        SEQUENCE_LENGTH
        - 1
    )


    # target t+10 must remain
    # before the test starting point

    final_anchor = (

        test_idx

        - ROLLOUT_STEPS

        - 1
    )


    if final_anchor < first_anchor:

        return np.array(
            [],
            dtype=int
        )


    anchors = np.arange(

        first_anchor,

        final_anchor + 1,

        dtype=int
    )


    return anchors


# ======================================================================
# 10. BUILD ONE CHRONOLOGICAL BATCH
#
# Example anchor = t
#
# INPUT:
#
# actual:
# t-49 ... t
#
# COMMANDS:
#
# U[t] ... U[t+9]
#
# TARGET:
#
# actual complete state t+10 ONLY
# ======================================================================

def make_batch(
    anchors,
    stats
):


    p_history_list = []

    b_history_list = []

    command_sequence_list = []

    final_p_target_list = []

    final_b_target_list = []


    for t in anchors:


        # --------------------------------------------------------------
        # past actual history
        # --------------------------------------------------------------

        p_hist = P[
            t-SEQUENCE_LENGTH+1:
            t+1
        ]

        # 50 x 4 x 12


        p_hist = np.transpose(

            p_hist,

            (1, 0, 2)
        )

        # 4 x 50 x 12


        b_hist = B[
            t-SEQUENCE_LENGTH+1:
            t+1
        ]


        # --------------------------------------------------------------
        # known control sequence
        #
        # 10 x 6
        # --------------------------------------------------------------

        command_sequence = U[
            t:
            t + ROLLOUT_STEPS
        ]


        # --------------------------------------------------------------
        # ONLY FINAL ACTUAL TARGET
        #
        # t + 10
        # --------------------------------------------------------------

        target_p = P[
            t + ROLLOUT_STEPS
        ]


        target_b = B[
            t + ROLLOUT_STEPS
        ]


        # --------------------------------------------------------------
        # normalize
        # --------------------------------------------------------------

        p_hist = normalize_p(
            p_hist,
            stats
        )


        b_hist = normalize_b(
            b_hist,
            stats
        )


        command_sequence = normalize_u(

            command_sequence,

            stats
        )


        target_p = normalize_p(
            target_p,
            stats
        )


        target_b = normalize_b(
            target_b,
            stats
        )


        p_history_list.append(
            p_hist
        )


        b_history_list.append(
            b_hist
        )


        command_sequence_list.append(
            command_sequence
        )


        final_p_target_list.append(
            target_p
        )


        final_b_target_list.append(
            target_b
        )


    p_history_tensor = torch.tensor(

        np.asarray(
            p_history_list
        ),

        dtype=torch.float32,

        device=device
    )


    b_history_tensor = torch.tensor(

        np.asarray(
            b_history_list
        ),

        dtype=torch.float32,

        device=device
    )


    command_sequence_tensor = torch.tensor(

        np.asarray(
            command_sequence_list
        ),

        dtype=torch.float32,

        device=device
    )


    final_p_target_tensor = torch.tensor(

        np.asarray(
            final_p_target_list
        ),

        dtype=torch.float32,

        device=device
    )


    final_b_target_tensor = torch.tensor(

        np.asarray(
            final_b_target_list
        ),

        dtype=torch.float32,

        device=device
    )


    return (

        p_history_tensor,

        b_history_tensor,

        command_sequence_tensor,

        final_p_target_tensor,

        final_b_target_tensor
    )


# ======================================================================
# 11. FIND TEST INDEX
# ======================================================================

def get_test_index(
    requested_time
):

    idx = int(

        np.argmin(

            np.abs(
                time
                - requested_time
            )
        )
    )


    # enough space for 10-step target

    idx = min(

        idx,

        len(P)
        - ROLLOUT_STEPS
        - 1
    )


    return idx


# ======================================================================
# 12. TRAIN FRESH MODEL FOR ONE TEST POINT
#
# Example:
#
# test = 100 s
#
# train only:
#
# roughly 80 s -> just before 100 s
#
# ONE EPOCH
# CHRONOLOGICAL BATCHES OF 60
# NO SHUFFLE
# ======================================================================

def train_model_before_test(
    test_idx,
    seed=42
):


    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )


    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


    test_time = time[
        test_idx
    ]


    train_start_time = (

        test_time

        - TRAIN_HISTORY_SECONDS
    )


    train_start_idx = int(

        np.searchsorted(

            time,

            train_start_time,

            side="left"
        )
    )


    train_start_idx = max(

        0,

        train_start_idx
    )


    # --------------------------------------------------------------
    # statistics only from historical training data
    # --------------------------------------------------------------

    stats = make_stats(

        train_start_idx,

        test_idx
    )


    anchors = make_training_anchors(

        train_start_idx,

        test_idx
    )


    if len(anchors) == 0:

        raise RuntimeError(
            "No training anchors available."
        )


    print(
        "\nTraining interval:"
    )


    print(

        f"{time[train_start_idx]:.3f} s"

        " -> "

        f"{time[test_idx]:.3f} s"
    )


    print(
        "Training rollout examples:",
        len(anchors)
    )


    print(
        "Chronological batch size:",
        BATCH_SIZE
    )


    print(
        "Number of gradient updates:",
        int(
            np.ceil(
                len(anchors)
                / BATCH_SIZE
            )
        )
    )


    # --------------------------------------------------------------
    # new model
    # --------------------------------------------------------------

    model = RecursiveFlatMLPModel(

        d_model=D_MODEL,

        nhead=NHEAD,

        temporal_layers=TEMPORAL_LAYERS,

        coupling_hidden=208,

        dropout=DROPOUT

    ).to(device)


    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LR,

        weight_decay=WEIGHT_DECAY
    )


    mse = nn.MSELoss()


    training_losses = []


    model.train()


    # ==============================================================
    # ONE SINGLE CHRONOLOGICAL EPOCH
    # ==============================================================

    for batch_start in range(

        0,

        len(anchors),

        BATCH_SIZE
    ):


        batch_anchors = anchors[

            batch_start:

            batch_start
            + BATCH_SIZE
        ]


        (

            p_history,

            b_history,

            command_sequence,

            target_p,

            target_b

        ) = make_batch(

            batch_anchors,

            stats
        )


        optimizer.zero_grad()


        # ----------------------------------------------------------
        # 10 predicted recursive states
        #
        # final prediction is t+10
        # ----------------------------------------------------------

        predicted_p_t10, predicted_b_t10 = (

            recursive_rollout(

                model,

                p_history,

                b_history,

                command_sequence,

                steps=ROLLOUT_STEPS,

                return_all=False
            )
        )


        # ----------------------------------------------------------
        # LOSS ONLY AT FINAL 0.1-s STATE
        #
        # NOT at intermediate predictions.
        # ----------------------------------------------------------

        p_loss = mse(

            predicted_p_t10,

            target_p
        )


        b_loss = mse(

            predicted_b_t10,

            target_b
        )


        final_loss = (

            0.5 * p_loss

            + 0.5 * b_loss
        )


        # ----------------------------------------------------------
        # Backprop goes through:
        #
        # t+10
        # ↑
        # t+9
        # ↑
        # ...
        # ↑
        # t+1
        #
        # all 10 model applications
        # ----------------------------------------------------------

        final_loss.backward()


        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            max_norm=1.0
        )


        optimizer.step()


        training_losses.append(

            float(
                final_loss.item()
            )
        )


        print(

            f"batch "
            f"{len(training_losses):02d}"
            f" | "

            f"anchors "
            f"{batch_anchors[0]}"
            f"-"
            f"{batch_anchors[-1]}"
            f" | "

            f"final 0.1s loss "
            f"{training_losses[-1]:.6f}"
        )


    print(
        "\nOne chronological epoch complete."
    )


    print(

        "Mean final-horizon loss:",

        np.mean(
            training_losses
        )
    )


    return (

        model,

        stats,

        training_losses
    )


# ======================================================================
# 13. TEST ONE 0.1-s ROLLOUT
# ======================================================================

def run_test_rollout(
    model,
    stats,
    t
):


    # --------------------------------------------------------------
    # actual observations ONLY through t
    # --------------------------------------------------------------

    p_hist = P[
        t-SEQUENCE_LENGTH+1:
        t+1
    ]


    p_hist = np.transpose(

        p_hist,

        (1, 0, 2)
    )


    b_hist = B[
        t-SEQUENCE_LENGTH+1:
        t+1
    ]


    # --------------------------------------------------------------
    # supplied future control sequence
    #
    # U[t] ... U[t+9]
    # --------------------------------------------------------------

    command_sequence = U[
        t:
        t + ROLLOUT_STEPS
    ]


    # normalize

    p_hist = normalize_p(
        p_hist,
        stats
    )


    b_hist = normalize_b(
        b_hist,
        stats
    )


    command_sequence = normalize_u(

        command_sequence,

        stats
    )


    # torch

    p_tensor = torch.tensor(

        p_hist,

        dtype=torch.float32,

        device=device

    ).unsqueeze(0)


    b_tensor = torch.tensor(

        b_hist,

        dtype=torch.float32,

        device=device

    ).unsqueeze(0)


    command_tensor = torch.tensor(

        command_sequence,

        dtype=torch.float32,

        device=device

    ).unsqueeze(0)


    # --------------------------------------------------------------
    # predictions
    # --------------------------------------------------------------

    model.eval()


    with torch.no_grad():

        (

            final_p_n,

            final_b_n,

            rollout_p_n,

            rollout_b_n

        ) = recursive_rollout(

            model,

            p_tensor,

            b_tensor,

            command_tensor,

            steps=ROLLOUT_STEPS,

            return_all=True
        )


    # remove batch dimension

    rollout_p_n = (

        rollout_p_n[0]

        .cpu()

        .numpy()
    )


    rollout_b_n = (

        rollout_b_n[0]

        .cpu()

        .numpy()
    )


    # denormalize

    rollout_p = denormalize_p(

        rollout_p_n,

        stats
    )


    rollout_b = denormalize_b(

        rollout_b_n,

        stats
    )


    # --------------------------------------------------------------
    # actual future trajectory
    #
    # ONLY used for evaluation
    # --------------------------------------------------------------

    actual_p = P[
        t+1:
        t+ROLLOUT_STEPS+1
    ]


    actual_b = B[
        t+1:
        t+ROLLOUT_STEPS+1
    ]


    return (

        rollout_p,

        rollout_b,

        actual_p,

        actual_b
    )


# ======================================================================
# 14. EXTRACT 18 PHYSICAL OUTPUTS WE CARE ABOUT
#
# 12 foot-force values:
#
# FL Fx Fy Fz
# FR Fx Fy Fz
# RL Fx Fy Fz
# RR Fx Fy Fz
#
# +
#
# body:
# vx vy vz
# wx wy wz
# ======================================================================

def extract_18(
    p,
    b
):


    # trajectory case:
    #
    # p = time x 4 x 12

    if p.ndim == 3:


        force = p[
            :,
            :,
            9:12
        ].reshape(
            len(p),
            12
        )


        twist = b[
            :,
            6:12
        ]


        return np.concatenate(

            [
                force,
                twist
            ],

            axis=1
        )


    # single-state case

    force = p[
        :,
        9:12
    ].reshape(12)


    twist = b[
        6:12
    ]


    return np.concatenate(
        [
            force,
            twist
        ]
    )


# ======================================================================
# 15. OUTPUT NAMES
# ======================================================================

groups = [

    (
        "FL force",
        0,
        3
    ),

    (
        "FR force",
        3,
        6
    ),

    (
        "RL force",
        6,
        9
    ),

    (
        "RR force",
        9,
        12
    ),

    (
        "Body linear velocity",
        12,
        15
    ),

    (
        "Body angular velocity",
        15,
        18
    ),
]


# ======================================================================
# 16. RUN FOUR TEST EXPERIMENTS
# ======================================================================

results = []


for experiment_id, requested_time in enumerate(

    TEST_TIMES,

    start=1
):


    print(
        "\n\n"
        + "=" * 90
    )


    print(
        f"TEST EXPERIMENT {experiment_id}"
    )


    print(
        "Requested start:",
        requested_time,
        "s"
    )


    print(
        "=" * 90
    )


    test_idx = get_test_index(
        requested_time
    )


    print(

        "Actual rollout start:",

        time[test_idx],
        "s"
    )


    print(

        "Actual target time:",

        time[
            test_idx
            + ROLLOUT_STEPS
        ],
        "s"
    )


    # --------------------------------------------------------------
    # fresh model trained only on previous 20 s
    # --------------------------------------------------------------

    model, stats, losses = (

        train_model_before_test(

            test_idx,

            seed=100 + experiment_id
        )
    )


    # --------------------------------------------------------------
    # recursive prediction
    # --------------------------------------------------------------

    (

        predicted_p,

        predicted_b,

        actual_p,

        actual_b

    ) = run_test_rollout(

        model,

        stats,

        test_idx
    )


    predicted_18 = extract_18(

        predicted_p,

        predicted_b
    )


    actual_18 = extract_18(

        actual_p,

        actual_b
    )


    final_pred = predicted_18[
        -1
    ]


    final_actual = actual_18[
        -1
    ]


    # --------------------------------------------------------------
    # PRINT FINAL 100-ms RESULT
    # --------------------------------------------------------------

    print(
        "\n"
        + "-" * 90
    )

    print(
        "FINAL PREDICTION AT t + 0.1 s"
    )

    print(
        "-" * 90
    )


    for name, a, b in groups:


        pred_value = final_pred[
            a:b
        ]


        actual_value = final_actual[
            a:b
        ]


        error = (

            pred_value

            - actual_value
        )


        group_rmse = np.sqrt(

            np.mean(
                error ** 2
            )
        )


        print(
            f"\n{name}"
        )


        print(

            "predicted :",

            np.round(
                pred_value,
                5
            )
        )


        print(

            "actual    :",

            np.round(
                actual_value,
                5
            )
        )


        print(

            "error     :",

            np.round(
                error,
                5
            )
        )


        print(

            "RMSE      :",

            round(
                float(group_rmse),
                6
            )
        )


    # ==============================================================
    # GRAPH 1
    #
    # FOUR FOOT FORCE MAGNITUDES
    # ==============================================================

    horizon_ms = (

        np.arange(
            1,
            ROLLOUT_STEPS + 1
        )

        * dt

        * 1000
    )


    predicted_force_mag = np.linalg.norm(

        predicted_p[
            :,
            :,
            9:12
        ],

        axis=2
    )


    actual_force_mag = np.linalg.norm(

        actual_p[
            :,
            :,
            9:12
        ],

        axis=2
    )


    plt.figure(
        figsize=(10, 6)
    )


    for leg_id, leg_name in enumerate(

        [
            "FL",
            "FR",
            "RL",
            "RR"
        ]
    ):


        plt.plot(

            horizon_ms,

            actual_force_mag[
                :,
                leg_id
            ],

            label=f"{leg_name} actual"
        )


        plt.plot(

            horizon_ms,

            predicted_force_mag[
                :,
                leg_id
            ],

            linestyle="--",

            label=f"{leg_name} predicted"
        )


    plt.xlabel(
        "Time into recursive rollout (ms)"
    )


    plt.ylabel(
        "Foot force magnitude"
    )


    plt.title(

        f"Foot-force recursive rollout\n"
        f"{time[test_idx]:.2f}s "
        f"→ "
        f"{time[test_idx + ROLLOUT_STEPS]:.2f}s"
    )


    plt.legend()

    plt.grid()

    plt.show()


    # ==============================================================
    # GRAPH 2
    #
    # BODY LINEAR SPEED
    # ==============================================================

    pred_linear_speed = np.linalg.norm(

        predicted_b[
            :,
            6:9
        ],

        axis=1
    )


    actual_linear_speed = np.linalg.norm(

        actual_b[
            :,
            6:9
        ],

        axis=1
    )


    plt.figure(
        figsize=(8, 5)
    )


    plt.plot(

        horizon_ms,

        actual_linear_speed,

        marker="o",

        label="Actual"
    )


    plt.plot(

        horizon_ms,

        pred_linear_speed,

        marker="o",

        label="Predicted"
    )


    plt.xlabel(
        "Time into recursive rollout (ms)"
    )


    plt.ylabel(
        "Body linear speed"
    )


    plt.title(

        f"Body-speed recursive rollout\n"
        f"{time[test_idx]:.2f}s "
        f"→ "
        f"{time[test_idx + ROLLOUT_STEPS]:.2f}s"
    )


    plt.legend()

    plt.grid()

    plt.show()


    # --------------------------------------------------------------
    # SAVE
    # --------------------------------------------------------------

    results.append({

        "requested_time":
            requested_time,

        "start_time":
            time[test_idx],

        "target_time":
            time[
                test_idx
                + ROLLOUT_STEPS
            ],

        "predicted":
            predicted_18,

        "actual":
            actual_18,

        "final_pred":
            final_pred,

        "final_actual":
            final_actual,

        "losses":
            np.asarray(
                losses
            )
    })


    # free GPU before next fresh model

    del model

    gc.collect()


    if torch.cuda.is_available():

        torch.cuda.empty_cache()


# ======================================================================
# 17. FINAL SUMMARY OF FOUR TESTS
# ======================================================================

print(
    "\n\n"
    + "=" * 90
)

print(
    "FINAL SUMMARY"
)

print(
    "=" * 90
)


for test_number, result in enumerate(

    results,

    start=1
):


    print(

        f"\nTEST {test_number}: "

        f"{result['start_time']:.3f} s"

        " -> "

        f"{result['target_time']:.3f} s"
    )


    for name, a, b in groups:


        predicted = result[
            "final_pred"
        ][
            a:b
        ]


        actual = result[
            "final_actual"
        ][
            a:b
        ]


        value = float(

            np.sqrt(

                np.mean(

                    (
                        predicted
                        - actual
                    ) ** 2
                )
            )
        )


        print(

            f"{name:25s}: "

            f"{value:.6f}"
        )


print(
    "\nDONE."
)
