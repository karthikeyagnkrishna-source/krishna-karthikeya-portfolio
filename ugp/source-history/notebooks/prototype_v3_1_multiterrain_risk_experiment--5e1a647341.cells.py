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
# V3 CELL 1
#
# EXTRACT + ALIGN TRUE LOW-LEVEL ACTUATOR COMMANDS
#
# Output:
#
#   L       : N x 12
#             commanded joint torques in canonical joint order
#
#   L_LEG   : N x 4 x 3
#
#   E_TAU   : N x 4 x 3
#             past actuator tracking residual
#
# Canonical order:
# FL HAA HFE KFE
# FR HAA HFE KFE
# RL HAA HFE KFE
# RR HAA HFE KFE
# ======================================================================

from rosbags.highlevel import AnyReader
from pathlib import Path
from scipy.optimize import linear_sum_assignment

import numpy as np


ACTUATOR_TOPIC = "/state_estimator/actuator_readings"


# ----------------------------------------------------------------------
# ROS timestamp helper
# ----------------------------------------------------------------------

def stamp_to_seconds(stamp):

    return (
        float(stamp.sec)
        +
        float(stamp.nanosec) * 1e-9
    )


# ----------------------------------------------------------------------
# Extract:
#
# command timestamp
# commanded joint torque
# command mode
#
# and actuator-state joint position
# used ONLY to determine actuator ordering
# ----------------------------------------------------------------------

command_times = []
command_torques = []
command_modes = []

actuator_state_times = []
actuator_state_positions = []


with AnyReader([Path(bag_path)]) as reader:

    connections = [

        c for c in reader.connections

        if c.topic == ACTUATOR_TOPIC
    ]


    assert len(connections) > 0, (
        f"{ACTUATOR_TOPIC} not found"
    )


    for connection, bag_timestamp, rawdata in reader.messages(
        connections=connections
    ):

        msg = reader.deserialize(
            rawdata,
            connection.msgtype
        )


        readings = list(msg.readings)


        if len(readings) != 12:
            continue


        # ==============================================================
        # LOW-LEVEL COMMAND
        # ==============================================================

        cmd_stamps = []

        cmd_tau = []

        cmd_mode = []


        for reading in readings:

            cmd = reading.commanded


            cmd_stamps.append(

                stamp_to_seconds(
                    cmd.header.stamp
                )
            )


            cmd_tau.append(

                float(
                    cmd.joint_torque
                )
            )


            cmd_mode.append(

                int(
                    cmd.mode
                )
            )


        cmd_stamps = np.asarray(
            cmd_stamps,
            dtype=np.float64
        )


        # all 12 should be essentially same timestamp
        command_time_abs = float(
            np.median(cmd_stamps)
        )


        command_times.append(
            command_time_abs - t0_sec
        )


        command_torques.append(
            cmd_tau
        )


        command_modes.append(
            cmd_mode
        )


        # ==============================================================
        # ACTUATOR MEASURED POSITION
        #
        # only for determining actuator index -> joint index
        # ==============================================================

        state_stamps = []

        state_positions = []


        for reading in readings:

            state = reading.state


            state_stamps.append(

                stamp_to_seconds(
                    state.header.stamp
                )
            )


            state_positions.append(

                float(
                    state.joint_position
                )
            )


        state_time_abs = float(
            np.median(state_stamps)
        )


        actuator_state_times.append(
            state_time_abs - t0_sec
        )


        actuator_state_positions.append(
            state_positions
        )


# ----------------------------------------------------------------------
# Convert to NumPy + sort
# ----------------------------------------------------------------------

command_times = np.asarray(
    command_times,
    dtype=np.float64
)

command_torques = np.asarray(
    command_torques,
    dtype=np.float32
)

command_modes = np.asarray(
    command_modes,
    dtype=np.int32
)


actuator_state_times = np.asarray(
    actuator_state_times,
    dtype=np.float64
)

actuator_state_positions = np.asarray(
    actuator_state_positions,
    dtype=np.float32
)


cmd_order = np.argsort(
    command_times
)

command_times = command_times[
    cmd_order
]

command_torques = command_torques[
    cmd_order
]

command_modes = command_modes[
    cmd_order
]


state_order = np.argsort(
    actuator_state_times
)

actuator_state_times = actuator_state_times[
    state_order
]

actuator_state_positions = actuator_state_positions[
    state_order
]


print(
    "Raw low-level command messages:",
    command_torques.shape
)

print(
    "Raw actuator states:",
    actuator_state_positions.shape
)


# ======================================================================
# VERIFY COMMAND MODES
# ======================================================================

unique_modes, mode_counts = np.unique(
    command_modes,
    return_counts=True
)


print("\nCommand mode distribution:")

for mode, count in zip(
    unique_modes,
    mode_counts
):

    print(
        f"mode {mode:2d}: "
        f"{count:8d} actuator samples"
    )


# torque is explicitly active in:
#
# 10 = torque
# 12 = pos + vel + torque
# 13 = pos + vel + torque + gains

TORQUE_COMMAND_MODES = {
    10,
    12,
    13
}


unsupported = ~np.isin(
    command_modes,
    list(TORQUE_COMMAND_MODES)
)


unsupported_fraction = float(
    unsupported.mean()
)


print(
    "\nFraction not containing commanded torque:",
    unsupported_fraction
)


if unsupported_fraction > 0.01:

    raise RuntimeError(

        "More than 1% of actuator samples are not in a "
        "torque-containing control mode. Inspect modes before training."
    )


# ======================================================================
# CAUSALLY ALIGN ACTUATOR STATE TO OUR 100-Hz STATE GRID
# ======================================================================

actuator_position_sync = causal_hold(

    actuator_state_times,
    actuator_state_positions,
    grid_original
)


# ======================================================================
# DETERMINE ACTUATOR ORDER AUTOMATICALLY
#
# P is already in:
#
# LF_HAA LF_HFE LF_KFE
# RF_HAA RF_HFE RF_KFE
# LH_HAA LH_HFE LH_KFE
# RH_HAA RH_HFE RH_KFE
#
# Compare actuator joint_position to existing canonical q.
# ======================================================================

q_canonical = P[
    :,
    :,
    0:3
].reshape(
    len(P),
    12
)


# use a long interval to make matching robust
match_count = min(
    5000,
    len(P)
)


cost = np.zeros(
    (12, 12),
    dtype=np.float64
)


for actuator_index in range(12):

    for joint_index in range(12):

        difference = (

            actuator_position_sync[
                :match_count,
                actuator_index
            ]

            -

            q_canonical[
                :match_count,
                joint_index
            ]
        )


        scale = (

            np.std(
                q_canonical[
                    :match_count,
                    joint_index
                ]
            )

            + 1e-6
        )


        cost[
            actuator_index,
            joint_index
        ] = (

            np.sqrt(
                np.mean(
                    difference ** 2
                )
            )

            /
            scale
        )


rows, cols = linear_sum_assignment(
    cost
)


# actuator_for_joint[k]
# gives actuator index corresponding to canonical joint k

actuator_for_joint = np.empty(
    12,
    dtype=int
)


for actuator_index, joint_index in zip(
    rows,
    cols
):

    actuator_for_joint[
        joint_index
    ] = actuator_index


EXPECTED_JOINTS = [

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


print(
    "\nRecovered actuator ordering:"
)


for joint_index, name in enumerate(
    EXPECTED_JOINTS
):

    actuator_index = actuator_for_joint[
        joint_index
    ]


    print(

        f"{name:8s}"
        f" <- actuator {actuator_index:2d}"
        f" | normalized match error = "
        f"{cost[actuator_index, joint_index]:.6f}"
    )


# ======================================================================
# CAUSALLY ALIGN LOW-LEVEL COMMAND TO 100-Hz GRID
# ======================================================================

command_tau_sync_actuator_order = causal_hold(

    command_times,
    command_torques,
    grid_original
)


command_mode_sync_actuator_order = causal_hold(

    command_times,
    command_modes,
    grid_original
)


# reorder to same order as P

L = command_tau_sync_actuator_order[
    :,
    actuator_for_joint
].astype(
    np.float32
)


L_MODE = command_mode_sync_actuator_order[
    :,
    actuator_for_joint
]


# N x 4 x 3

L_LEG = L.reshape(
    len(L),
    4,
    3
)


print(
    "\nFinal low-level command arrays:"
)

print(
    "L     :",
    L.shape
)

print(
    "L_LEG :",
    L_LEG.shape
)


# ======================================================================
# HISTORICAL COMMAND -> RESPONSE ERROR
#
# State at k is compared against command from previous interval k-1.
#
# E_TAU[k] =
#
# measured torque[k] - commanded torque[k-1]
# ======================================================================

measured_tau = P[
    :,
    :,
    6:9
]


E_TAU = np.zeros_like(
    measured_tau,
    dtype=np.float32
)


E_TAU[0] = (

    measured_tau[0]
    -
    L_LEG[0]
)


E_TAU[1:] = (

    measured_tau[1:]

    -

    L_LEG[:-1]
)


# ======================================================================
# DIAGNOSTICS
# ======================================================================

same_time_error = (

    measured_tau
    -
    L_LEG
)


previous_command_error = E_TAU


print(
    "\nTorque diagnostics:"
)


print(
    "RMSE measured(t) - command(t):",
    np.sqrt(
        np.mean(
            same_time_error ** 2
        )
    )
)


print(
    "RMSE measured(t) - command(t-1):",
    np.sqrt(
        np.mean(
            previous_command_error ** 2
        )
    )
)


print(
    "\nExample:"
)

print(
    "commanded torque:",
    L_LEG[100, 0]
)

print(
    "measured torque :",
    measured_tau[101, 0]
)

print(
    "response error  :",
    measured_tau[101, 0]
    -
    L_LEG[100, 0]
)


assert (
    len(L)
    ==
    len(P)
    ==
    len(B)
    ==
    len(U)
    ==
    len(time)
)


print(
    "\nLOW-LEVEL COMMAND EXTRACTION COMPLETE."
)

# %% notebook cell 2
# ======================================================================
# V3 FIX — LOCATE / DOWNLOAD RGB-D DATA
#
# Run this BEFORE the V3 visual-feature rebuild cell.
# ======================================================================

import os
from pathlib import Path


ROOT = Path("/content/tartanground")


# ======================================================================
# 1. LOOK FOR EXISTING P2000 DIRECTORY
# ======================================================================

candidate_bases = []


if ROOT.exists():

    for p in ROOT.rglob("P2000"):

        if p.is_dir():

            candidate_bases.append(p)


print("P2000 directories found:")

for p in candidate_bases:

    print(" ", p)


# Preferred known TartanGround location
preferred = (
    ROOT
    / "OldTownFall"
    / "Data_anymal"
    / "P2000"
)


if preferred.exists():

    BASE = preferred

elif len(candidate_bases) > 0:

    # choose candidate that looks most like the trajectory folder

    BASE = candidate_bases[0]

else:

    BASE = preferred


print("\nUsing BASE:")
print(BASE)


# ======================================================================
# 2. DEFINE EXPECTED VISUAL PATHS
# ======================================================================

RGB_DIR = BASE / "image_lcam_front"

DEPTH_DIR = BASE / "depth_lcam_front"

CAM_TIME_FILE = (
    BASE
    / "imu"
    / "cam_time.txt"
)


print("\nBefore download:")

print(
    "RGB   :",
    RGB_DIR.exists(),
    RGB_DIR
)

print(
    "Depth :",
    DEPTH_DIR.exists(),
    DEPTH_DIR
)

print(
    "Cam t :",
    CAM_TIME_FILE.exists(),
    CAM_TIME_FILE
)


# ======================================================================
# 3. DOWNLOAD VISUAL DATA IF NECESSARY
# ======================================================================

need_visual_download = (

    not RGB_DIR.exists()

    or

    not DEPTH_DIR.exists()

    or

    not CAM_TIME_FILE.exists()
)


if need_visual_download:

    print(
        "\nRGB/depth/camera timing incomplete."
    )

    print(
        "Downloading required P2000 visual data..."
    )


    import tartanair as ta


    ta.init(
        str(ROOT)
    )


    # image + depth are needed for terrain
    #
    # imu is included because cam_time.txt lives there

    ta.download_ground(

        env=[
            "OldTownFall"
        ],

        version=[
            "anymal"
        ],

        traj=[
            "P2000"
        ],

        modality=[
            "image",
            "depth",
            "imu"
        ],

        camera_name=[
            "lcam_front"
        ],

        unzip=True
    )


# ======================================================================
# 4. SEARCH AGAIN AFTER DOWNLOAD
# ======================================================================

candidate_bases = [

    p

    for p in ROOT.rglob("P2000")

    if p.is_dir()

    and
    (
        p / "image_lcam_front"
    ).exists()

]


if len(candidate_bases) == 0:

    raise RuntimeError(

        "\nCould not find a P2000 directory containing "
        "image_lcam_front after download.\n\n"

        "Run:\n"
        "!find /content/tartanground -maxdepth 6 -type d | head -200\n\n"

        "and send me that output."
    )


# Prefer exact known path if valid.

if (
    preferred / "image_lcam_front"
).exists():

    BASE = preferred

else:

    BASE = candidate_bases[0]


RGB_DIR = BASE / "image_lcam_front"

DEPTH_DIR = BASE / "depth_lcam_front"

CAM_TIME_FILE = (
    BASE
    / "imu"
    / "cam_time.txt"
)


# ======================================================================
# 5. FINAL CHECK
# ======================================================================

assert RGB_DIR.is_dir(), (
    f"RGB missing: {RGB_DIR}"
)

assert DEPTH_DIR.is_dir(), (
    f"Depth missing: {DEPTH_DIR}"
)

assert CAM_TIME_FILE.exists(), (
    f"cam_time.txt missing: {CAM_TIME_FILE}"
)


rgb_files = list(
    RGB_DIR.glob(
        "*_lcam_front.png"
    )
)

depth_files = list(
    DEPTH_DIR.glob(
        "*_lcam_front_depth.png"
    )
)


print("\n" + "=" * 70)

print("VISUAL DATA FOUND")

print("=" * 70)

print(
    "BASE:",
    BASE
)

print(
    "RGB frames:",
    len(rgb_files)
)

print(
    "Depth frames:",
    len(depth_files)
)

print(
    "cam_time.txt:",
    CAM_TIME_FILE
)


print(
    "\nNow run the V3 VISUAL REBUILD cell."
)

# %% notebook cell 3
# ======================================================================
# V3 VISUAL REBUILD — RUN ONCE BEFORE V3 CELL 2
#
# Recreates:
#   DINO_FEATURES
#   CONV_FEATURES
#   DEPTH_GRIDS
#   camera_times
#   visual_index_for_state
#
# Uses ONLY terrain immediately ahead.
# NO future image is ever assigned to a state.
# ======================================================================

import os
import gc
import cv2
import numpy as np
import torch
import torch.nn as nn

from PIL import Image

from transformers import (
    AutoImageProcessor,
    AutoModel
)

from torchvision.models import (
    convnext_tiny,
    ConvNeXt_Tiny_Weights
)


# ======================================================================
# 1. PATHS
# ======================================================================

BASE = "/content/tartanground/OldTownFall/Data_anymal/P2000"

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


assert os.path.isdir(RGB_DIR)
assert os.path.isdir(DEPTH_DIR)
assert os.path.exists(CAM_TIME_FILE)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ======================================================================
# 2. TERRAIN-ONLY REGION
#
# lower-centre view
#
# and only nearby geometry.
#
# Old model used 4 m.
# V3 uses 2 m because our objective is terrain immediately ahead.
# ======================================================================

ROI_X_MIN = 0.18
ROI_X_MAX = 0.82

ROI_Y_MIN = 0.45
ROI_Y_MAX = 0.97

DEPTH_MIN_METERS = 0.25
DEPTH_MAX_METERS = 2.0

DEPTH_SIZE = 64


# ======================================================================
# 3. MATCH RGB AND DEPTH
# ======================================================================

rgb_ids = {
    f.split("_")[0]
    for f in os.listdir(RGB_DIR)
    if f.endswith("_lcam_front.png")
}


depth_ids = {
    f.split("_")[0]
    for f in os.listdir(DEPTH_DIR)
    if f.endswith("_lcam_front_depth.png")
}


frame_ids_all = sorted(
    rgb_ids & depth_ids,
    key=lambda x: int(x)
)


frame_numbers_all = np.asarray(
    [int(x) for x in frame_ids_all],
    dtype=int
)


print(
    "Matched raw RGB-D frames:",
    len(frame_ids_all)
)


# ======================================================================
# 4. CAMERA TIMES
# ======================================================================

camera_time_original = np.loadtxt(
    CAM_TIME_FILE
).reshape(-1).astype(np.float64)


assert frame_numbers_all.max() < len(
    camera_time_original
)


camera_times_original_all = (
    camera_time_original[
        frame_numbers_all
    ]
)


# Use exact current crop offset if it exists.

if "crop_start_original" in globals():

    STATE_START_IN_ORIGINAL_TIME = float(
        crop_start_original
    )

else:

    # value used by the previous P2000 preprocessing

    STATE_START_IN_ORIGINAL_TIME = 4.04

    print(
        "WARNING: crop_start_original absent; "
        "using previous 4.04 s offset."
    )


camera_times_all = (
    camera_times_original_all
    -
    STATE_START_IN_ORIGINAL_TIME
)


# keep only camera frames overlapping current state trajectory

valid_camera = (
    (camera_times_all >= time[0] - 1e-8)
    &
    (camera_times_all <= time[-1] + 1e-8)
)


camera_times = camera_times_all[
    valid_camera
]


frame_numbers = frame_numbers_all[
    valid_camera
]


frame_ids = np.asarray(
    frame_ids_all
)[
    valid_camera
].tolist()


print(
    "Usable RGB-D frames:",
    len(frame_ids)
)

print(
    "Camera range:",
    camera_times[0],
    "->",
    camera_times[-1]
)


# ======================================================================
# 5. TARTANGROUND METRIC DEPTH DECODER
#
# PNG contains packed little-endian float32 depth.
# ======================================================================

def load_metric_depth(path):

    raw = cv2.imread(
        path,
        cv2.IMREAD_UNCHANGED
    )


    if raw is None:

        raise RuntimeError(
            f"Could not read {path}"
        )


    if (
        raw.dtype == np.uint8
        and
        raw.ndim == 3
        and
        raw.shape[2] == 4
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
        and
        raw.ndim == 2
    ):

        depth = raw


    else:

        raise RuntimeError(
            f"Unexpected depth format: "
            f"{raw.dtype}, {raw.shape}"
        )


    depth = depth.astype(
        np.float32
    )


    depth[
        ~np.isfinite(depth)
    ] = 0.0


    return depth


# ======================================================================
# 6. TAKE ONLY TERRAIN DIRECTLY AHEAD
# ======================================================================

def load_ground_roi(frame_id):

    rgb_path = os.path.join(
        RGB_DIR,
        f"{frame_id}_lcam_front.png"
    )

    depth_path = os.path.join(
        DEPTH_DIR,
        f"{frame_id}_lcam_front_depth.png"
    )


    rgb = np.asarray(
        Image.open(
            rgb_path
        ).convert("RGB")
    )


    depth = load_metric_depth(
        depth_path
    )


    H, W = rgb.shape[:2]


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
    # keep only nearby valid geometry
    # --------------------------------------------------------------

    valid_depth = (
        np.isfinite(depth_roi)
        &
        (depth_roi >= DEPTH_MIN_METERS)
        &
        (depth_roi <= DEPTH_MAX_METERS)
    )


    # DINO + ConvNeXt see only RGB associated with nearby depth

    rgb_ground = rgb_roi.copy()

    rgb_ground[
        ~valid_depth
    ] = 0


    # depth network sees corresponding geometry

    depth_ground = depth_roi.copy()

    depth_ground[
        ~valid_depth
    ] = 0.0


    return (
        rgb_ground,
        depth_ground
    )


# ======================================================================
# 7. LOAD FROZEN DINOv2
# ======================================================================

print("\nLoading DINOv2...")


DINO_NAME = "facebook/dinov2-small"


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


for parameter in dino_model.parameters():

    parameter.requires_grad = False


print("DINOv2 ready.")


# ======================================================================
# 8. LOAD FROZEN CONVNEXT
# ======================================================================

print("Loading ConvNeXt...")


conv_weights = (
    ConvNeXt_Tiny_Weights.DEFAULT
)


convnext = convnext_tiny(
    weights=conv_weights
).to(device)


convnext.classifier = nn.Identity()


convnext.eval()


for parameter in convnext.parameters():

    parameter.requires_grad = False


conv_transform = (
    conv_weights.transforms()
)


print("ConvNeXt ready.")


# ======================================================================
# 9. PRECOMPUTE TERRAIN FEATURES
#
# This is the expensive part.
# It runs ONCE.
# ======================================================================

VIS_BATCH = 16


all_dino = []
all_conv = []
all_depth = []


print(
    "\nPrecomputing V3 terrain features..."
)


for batch_start in range(
    0,
    len(frame_ids),
    VIS_BATCH
):

    batch_ids = frame_ids[
        batch_start:
        batch_start + VIS_BATCH
    ]


    pil_batch = []
    conv_batch = []
    depth_batch = []


    for frame_id in batch_ids:

        rgb_ground, depth_ground = (
            load_ground_roi(
                frame_id
            )
        )


        pil_img = Image.fromarray(
            rgb_ground
        )


        pil_batch.append(
            pil_img
        )


        # ConvNeXt input

        conv_batch.append(
            conv_transform(
                pil_img
            )
        )


        # ----------------------------------------------------------
        # Depth: 64 x 64
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
            /
            DEPTH_MAX_METERS
        )


        depth_batch.append(
            depth_small[
                None,
                :,
                :
            ]
        )


    # ==============================================================
    # DINO
    # ==============================================================

    dino_inputs = dino_processor(

        images=pil_batch,

        return_tensors="pt"

    ).to(device)


    with torch.no_grad():

        dino_output = dino_model(
            **dino_inputs
        )


        # global CLS representation

        dino_feature = (
            dino_output
            .last_hidden_state[
                :,
                0,
                :
            ]
        )


    # ==============================================================
    # ConvNeXt
    # ==============================================================

    conv_tensor = torch.stack(
        conv_batch
    ).to(device)


    with torch.no_grad():

        conv_feature = convnext(
            conv_tensor
        )


    # make it definitely B x 768

    conv_feature = conv_feature.flatten(
        1
    )


    all_dino.append(
        dino_feature
        .cpu()
        .numpy()
        .astype(np.float32)
    )


    all_conv.append(
        conv_feature
        .cpu()
        .numpy()
        .astype(np.float32)
    )


    all_depth.append(
        np.asarray(
            depth_batch,
            dtype=np.float32
        )
    )


    print(
        f"\rFrames: "
        f"{min(batch_start + VIS_BATCH, len(frame_ids))}"
        f"/{len(frame_ids)}",
        end=""
    )


# ======================================================================
# 10. FINAL FEATURE ARRAYS
# ======================================================================

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


print("\n\nVisual extraction complete.")

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


assert len(DINO_FEATURES) == len(
    camera_times
)

assert len(CONV_FEATURES) == len(
    camera_times
)

assert len(DEPTH_GRIDS) == len(
    camera_times
)


# ======================================================================
# 11. SAVE CACHE
#
# Therefore you do NOT need to regenerate it next time in this runtime.
# ======================================================================

VISUAL_CACHE = (
    "/content/"
    "P2000_V3_ground_visual_features.npz"
)


np.savez_compressed(

    VISUAL_CACHE,

    frame_numbers=frame_numbers,

    camera_times=camera_times,

    dino=DINO_FEATURES,

    convnext=CONV_FEATURES,

    depth=DEPTH_GRIDS
)


print(
    "Saved:",
    VISUAL_CACHE
)


# ======================================================================
# 12. CAUSALLY ASSIGN ONE IMAGE TO EVERY STATE
#
# state at t uses:
#
# latest image camera_time <= time[t]
#
# NEVER t+1 image.
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


valid_indices = np.where(
    visual_is_valid
)[0]


selected_camera_times = camera_times[

    visual_index_for_state[
        valid_indices
    ]
]


assert np.all(

    selected_camera_times

    <=

    time[
        valid_indices
    ]

    + 1e-8
)


print(
    "\nStates with causal terrain image:",
    int(
        visual_is_valid.sum()
    ),
    "/",
    len(time)
)


print(
    "visual_index_for_state:",
    visual_index_for_state.shape
)


print(
    "\nV3 VISUAL DATA READY."
)

print(
    "Now run V3 CELL 2."
)


# free frozen visual models

del dino_model
del convnext

gc.collect()


if torch.cuda.is_available():

    torch.cuda.empty_cache()

# %% notebook cell 4
# ======================================================================
# V3 CELL 2
#
# DATASET
#
# History:
#   P[t-49:t]      sensor response
#   B[t-49:t]      body response
#   U[t-49:t]      high-level commands
#   L[t-49:t]      low-level joint torque commands
#   E[t-49:t]      command -> measured torque residual
#
# Vision:
#   ONLY RGB-D / terrain embedding at t
#
# Training rollout commands:
#   U[t:t+9]
#   L[t:t+9]
#
# Targets:
#   P[t+1:t+10]
#   B[t+1:t+10]
#
# NO FUTURE RGB-D.
# ======================================================================

import torch
import torch.nn as nn

from torch.utils.data import (
    Dataset,
    DataLoader
)


HISTORY_V3 = 50
HORIZON_V3 = 10


TRAIN_END_V3 = 90.0
VAL_END_V3 = 100.0
TEST_END_V3 = 120.0


EPS_V3 = 1e-6


# ======================================================================
# NORMALIZATION
#
# TRAIN REGION ONLY
# ======================================================================

train_mask_v3 = (
    time < TRAIN_END_V3
)


P_MEAN_V3 = P[
    train_mask_v3
].mean(
    axis=(0, 1)
).astype(
    np.float32
)


P_STD_V3 = (

    P[
        train_mask_v3
    ].std(
        axis=(0, 1)
    )

    + EPS_V3

).astype(
    np.float32
)


B_MEAN_V3 = B[
    train_mask_v3
].mean(
    axis=0
).astype(
    np.float32
)


B_STD_V3 = (

    B[
        train_mask_v3
    ].std(
        axis=0
    )

    + EPS_V3

).astype(
    np.float32
)


U_MEAN_V3 = U[
    train_mask_v3
].mean(
    axis=0
).astype(
    np.float32
)


U_STD_V3 = (

    U[
        train_mask_v3
    ].std(
        axis=0
    )

    + EPS_V3

).astype(
    np.float32
)


def norm_p_v3(x):

    return (
        x - P_MEAN_V3
    ) / P_STD_V3


def denorm_p_v3(x):

    return (
        x * P_STD_V3
        + P_MEAN_V3
    )


def norm_b_v3(x):

    return (
        x - B_MEAN_V3
    ) / B_STD_V3


def denorm_b_v3(x):

    return (
        x * B_STD_V3
        + B_MEAN_V3
    )


def norm_u_v3(x):

    return (
        x - U_MEAN_V3
    ) / U_STD_V3


# ----------------------------------------------------------------------
# Normalize LOW-LEVEL torque in EXACTLY the same coordinate system
# as measured torque in P.
#
# This lets:
#
# predicted measured torque =
# commanded torque + learned tracking/contact residual
# ----------------------------------------------------------------------

L_PNORM = (

    L_LEG

    -

    P_MEAN_V3[
        6:9
    ]

) / P_STD_V3[
    6:9
]


# historical response residual normalization

E_TAU_NORM = (

    E_TAU

    /

    P_STD_V3[
        6:9
    ]
).astype(
    np.float32
)


# ======================================================================
# CONTACT LABEL
# ======================================================================

CONTACT_THRESHOLD_V3 = 10.0


CONTACT_STATE_V3 = (

    np.linalg.norm(
        P[
            :,
            :,
            9:12
        ],
        axis=-1
    )

    >
    CONTACT_THRESHOLD_V3

).astype(
    np.float32
)


# ======================================================================
# VALID SAMPLE INDICES
# ======================================================================

def make_v3_indices(
    start_time,
    end_time
):

    indices = []


    for t in range(

        HISTORY_V3 - 1,

        len(P) - HORIZON_V3
    ):


        if time[t] < start_time:
            continue


        # full t+1 ... t+10 target
        # must remain inside split

        if time[
            t + HORIZON_V3
        ] >= end_time:

            continue


        # only require image at CURRENT t

        if visual_index_for_state[t] < 0:
            continue


        indices.append(
            t
        )


    return np.asarray(
        indices,
        dtype=np.int64
    )


train_indices_v3 = make_v3_indices(
    0.0,
    TRAIN_END_V3
)


val_indices_v3 = make_v3_indices(
    TRAIN_END_V3,
    VAL_END_V3
)


test_indices_v3 = make_v3_indices(
    VAL_END_V3,
    TEST_END_V3
)


print(
    "Train:",
    len(train_indices_v3)
)

print(
    "Val  :",
    len(val_indices_v3)
)

print(
    "Test :",
    len(test_indices_v3)
)


# ======================================================================
# DATASET
# ======================================================================

class ControlledTerrainDatasetV3(
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
        dataset_index
    ):

        t = int(
            self.indices[
                dataset_index
            ]
        )


        h0 = (
            t
            -
            HISTORY_V3
            +
            1
        )

        h1 = t + 1


        # ==============================================================
        # PROPRIOCEPTION HISTORY
        #
        # 50 x 4 x 12
        # ->
        # 4 x 50 x 12
        # ==============================================================

        p_hist = norm_p_v3(

            P[
                h0:h1
            ]

        ).astype(
            np.float32
        )


        p_hist = np.transpose(

            p_hist,

            (1, 0, 2)
        )


        # ==============================================================
        # BODY HISTORY
        # ==============================================================

        b_hist = norm_b_v3(

            B[
                h0:h1
            ]

        ).astype(
            np.float32
        )


        # ==============================================================
        # HIGH-LEVEL COMMAND HISTORY
        # ==============================================================

        u_hist = norm_u_v3(

            U[
                h0:h1
            ]

        ).astype(
            np.float32
        )


        # ==============================================================
        # LOW-LEVEL COMMAND HISTORY
        #
        # 50 x 4 x 3
        # ->
        # 4 x 50 x 3
        # ==============================================================

        l_hist = L_PNORM[
            h0:h1
        ]


        l_hist = np.transpose(

            l_hist,

            (1, 0, 2)

        ).astype(
            np.float32
        )


        # ==============================================================
        # HISTORICAL COMMAND -> RESPONSE ERROR
        # ==============================================================

        e_hist = E_TAU_NORM[
            h0:h1
        ]


        e_hist = np.transpose(

            e_hist,

            (1, 0, 2)

        ).astype(
            np.float32
        )


        # ==============================================================
        # ONE AND ONLY VISUAL OBSERVATION
        #
        # at time t
        #
        # NO future image is loaded anywhere in this Dataset.
        # ==============================================================

        vi = int(
            visual_index_for_state[
                t
            ]
        )


        dino = np.asarray(

            DINO_FEATURES[
                vi
            ],

            dtype=np.float32

        ).reshape(-1)


        conv = np.asarray(

            CONV_FEATURES[
                vi
            ],

            dtype=np.float32

        ).reshape(-1)


        depth = np.asarray(

            DEPTH_GRIDS[
                vi
            ],

            dtype=np.float32
        )


        # ==============================================================
        # RECORDED FUTURE HIGH-LEVEL COMMANDS
        #
        # ONLY FOR TRAINING / OFFLINE ROLLOUT
        #
        # U[t] ... U[t+9]
        # ==============================================================

        future_u = norm_u_v3(

            U[
                t:
                t + HORIZON_V3
            ]

        ).astype(
            np.float32
        )


        # ==============================================================
        # RECORDED FUTURE LOW-LEVEL COMMANDS
        #
        # L[t] ... L[t+9]
        # ==============================================================

        future_l = L_PNORM[

            t:
            t + HORIZON_V3

        ].astype(
            np.float32
        )


        # ==============================================================
        # ACTUAL FUTURE SENSOR RESPONSES
        # ==============================================================

        target_p = norm_p_v3(

            P[
                t+1:
                t+HORIZON_V3+1
            ]

        ).astype(
            np.float32
        )


        target_b = norm_b_v3(

            B[
                t+1:
                t+HORIZON_V3+1
            ]

        ).astype(
            np.float32
        )


        target_contact = CONTACT_STATE_V3[

            t+1:
            t+HORIZON_V3+1

        ].astype(
            np.float32
        )


        return {

            "p_hist":
                torch.from_numpy(
                    p_hist
                ),

            "b_hist":
                torch.from_numpy(
                    b_hist
                ),

            "u_hist":
                torch.from_numpy(
                    u_hist
                ),

            "l_hist":
                torch.from_numpy(
                    l_hist
                ),

            "e_hist":
                torch.from_numpy(
                    e_hist
                ),

            "dino":
                torch.from_numpy(
                    dino
                ),

            "conv":
                torch.from_numpy(
                    conv
                ),

            "depth":
                torch.from_numpy(
                    depth
                ),

            "future_u":
                torch.from_numpy(
                    future_u
                ),

            "future_l":
                torch.from_numpy(
                    future_l
                ),

            "target_p":
                torch.from_numpy(
                    target_p
                ),

            "target_b":
                torch.from_numpy(
                    target_b
                ),

            "target_contact":
                torch.from_numpy(
                    target_contact
                ),

            "t":
                torch.tensor(
                    t,
                    dtype=torch.long
                )
        }


train_dataset_v3 = ControlledTerrainDatasetV3(
    train_indices_v3
)

val_dataset_v3 = ControlledTerrainDatasetV3(
    val_indices_v3
)

test_dataset_v3 = ControlledTerrainDatasetV3(
    test_indices_v3
)


BATCH_SIZE_V3 = 16


train_loader_v3 = DataLoader(

    train_dataset_v3,

    batch_size=BATCH_SIZE_V3,

    shuffle=True,

    num_workers=2,

    pin_memory=True
)


val_loader_v3 = DataLoader(

    val_dataset_v3,

    batch_size=BATCH_SIZE_V3,

    shuffle=False,

    num_workers=2,

    pin_memory=True
)


test_loader_v3 = DataLoader(

    test_dataset_v3,

    batch_size=BATCH_SIZE_V3,

    shuffle=False,

    num_workers=2,

    pin_memory=True
)


print(
    "\nV3 DATASET READY."
)

# %% notebook cell 5
# ======================================================================
# V3 FIX — RESTORE CAUSAL VISUAL -> STATE ALIGNMENT
#
# Run immediately BEFORE V3 CELL 2
# ======================================================================

import numpy as np
import os

VISUAL_CACHE = "/content/P2000_ground_visual_features.npz"


# ----------------------------------------------------------------------
# 1. Recover visual features / camera times if this Colab session
#    does not currently have them in memory.
# ----------------------------------------------------------------------

if "camera_times" not in globals():

    assert os.path.exists(VISUAL_CACHE), (
        f"Visual cache not found: {VISUAL_CACHE}"
    )

    cache = np.load(
        VISUAL_CACHE,
        allow_pickle=False
    )

    camera_times = cache[
        "camera_times"
    ].astype(np.float64)

    print(
        "Reloaded camera_times:",
        camera_times.shape
    )


    # also restore visual features if necessary

    if "DINO_FEATURES" not in globals():

        DINO_FEATURES = cache[
            "dino"
        ].astype(np.float32)


    if "CONV_FEATURES" not in globals():

        CONV_FEATURES = cache[
            "convnext"
        ].astype(np.float32)


    if "DEPTH_GRIDS" not in globals():

        DEPTH_GRIDS = cache[
            "depth"
        ].astype(np.float32)


# ----------------------------------------------------------------------
# 2. Basic consistency checks
# ----------------------------------------------------------------------

camera_times = np.asarray(
    camera_times,
    dtype=np.float64
)

time = np.asarray(
    time,
    dtype=np.float64
)


assert len(camera_times) == len(DINO_FEATURES), (
    "camera_times and DINO feature count mismatch"
)

assert len(camera_times) == len(CONV_FEATURES), (
    "camera_times and ConvNeXt feature count mismatch"
)

assert len(camera_times) == len(DEPTH_GRIDS), (
    "camera_times and depth feature count mismatch"
)


print("\nState time range:")
print(
    time[0],
    "->",
    time[-1]
)

print("\nCamera time range:")
print(
    camera_times[0],
    "->",
    camera_times[-1]
)


# ----------------------------------------------------------------------
# 3. CAUSAL image assignment
#
# For state time t:
#
# choose LAST camera frame whose timestamp <= t
#
# NO future RGB-D is ever selected.
# ----------------------------------------------------------------------

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
    "\nStates with a causal RGB-D frame:",
    int(
        visual_is_valid.sum()
    ),
    "/",
    len(time)
)


# ----------------------------------------------------------------------
# 4. Strong causal sanity check
# ----------------------------------------------------------------------

valid_state_indices = np.where(
    visual_is_valid
)[0]


used_visual_indices = visual_index_for_state[
    valid_state_indices
]


used_camera_times = camera_times[
    used_visual_indices
]


assert np.all(

    used_camera_times

    <=

    time[
        valid_state_indices
    ]

    + 1e-8

), "FUTURE IMAGE LEAKAGE DETECTED"


print(
    "\nCausal check passed:"
)

print(
    "Every selected RGB-D frame occurs at or before its robot state."
)


print(
    "\nvisual_index_for_state shape:",
    visual_index_for_state.shape
)

print(
    "\nREADY FOR V3 CELL 2."
)

# %% notebook cell 6
# ======================================================================
# V3 CELL 3
#
# CONTROLLED TERRAIN-CONDITIONED DYNAMICS MODEL
#
# Fundamental learned cell:
#
# history(P,B,U,L,E) + terrain(t)
#            +
#       current L_t
#
#              ->
#
# P[t+1], B[t+1], contact[t+1]
#
#
# Training:
#
# SAME CELL is unrolled 10 times.
#
# State predictions are fed back.
#
# Recorded U/L commands are advanced.
#
# TERRAIN TOKEN IS NEVER ADVANCED.
# ======================================================================


class CausalHistoryEncoderV3(
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


        self.input_proj = nn.Linear(
            input_dim,
            hidden_dim
        )


        self.pos_embedding = nn.Parameter(

            torch.randn(
                1,
                HISTORY_V3,
                hidden_dim
            )

            * 0.02
        )


        layer = nn.TransformerEncoderLayer(

            d_model=hidden_dim,

            nhead=heads,

            dim_feedforward=
                hidden_dim * 4,

            dropout=0.1,

            batch_first=True,

            norm_first=True,

            activation="gelu"
        )


        self.transformer = (
            nn.TransformerEncoder(
                layer,
                num_layers=layers
            )
        )


        self.norm = nn.LayerNorm(
            hidden_dim
        )


    def forward(
        self,
        x
    ):

        # B x T x input_dim

        T_now = x.shape[1]


        x = (
            self.input_proj(x)

            +

            self.pos_embedding[
                :,
                :T_now
            ]
        )


        causal_mask = torch.triu(

            torch.ones(
                T_now,
                T_now,
                device=x.device,
                dtype=torch.bool
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
            -1
        ]


# ======================================================================
# VISUAL TERRAIN ENCODER
#
# Uses existing precomputed:
# DINO
# ConvNeXt
# depth grid
#
# This module receives ONLY the visual information from initial t.
# ======================================================================

class TerrainEncoderV3(
    nn.Module
):

    def __init__(
        self,
        dino_dim,
        conv_dim
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


        self.depth_net = nn.Sequential(

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

            norm_first=True,

            activation="gelu"
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
            )

            * 0.02
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

        conv = conv.flatten(
            start_dim=1
        )


        dino_token = self.dino_proj(
            dino
        )


        conv_token = self.conv_proj(
            conv
        )


        depth_token = self.depth_net(
            depth
        ).flatten(1)


        depth_token = self.depth_proj(
            depth_token
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

            tokens.mean(
                dim=1
            )
        )


# ======================================================================
# COMPLETE V3
# ======================================================================

class ControlledTerrainDynamicsV3(
    nn.Module
):

    def __init__(
        self,
        dino_dim=384,
        conv_dim=768
    ):

        super().__init__()


        # --------------------------------------------------------------
        # PER-LEG:
        #
        # P = 12
        # U = 6
        # L = 3
        # E = 3
        #
        # total = 24
        # --------------------------------------------------------------

        self.leg_encoder = CausalHistoryEncoderV3(

            input_dim=24,

            hidden_dim=128,

            layers=2,

            heads=4
        )


        self.leg_identity = nn.Embedding(
            4,
            128
        )


        # body + high-level command

        self.body_encoder = CausalHistoryEncoderV3(

            input_dim=18,

            hidden_dim=128,

            layers=2,

            heads=4
        )


        self.terrain_encoder = TerrainEncoderV3(

            dino_dim=dino_dim,

            conv_dim=conv_dim
        )


        # four legs:
        # 4 x 128 = 512
        #
        # body:
        # 128
        #
        # terrain:
        # 256
        #
        # total:
        # 896

        self.context_net = nn.Sequential(

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


        # --------------------------------------------------------------
        # ONE CONTROLLED DYNAMICS CELL
        #
        # 256 physical context
        # ->
        # 60 normalized residual values
        #
        # P 48
        # B 12
        # --------------------------------------------------------------

        self.response_head = nn.Sequential(

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


        # four contact probabilities

        self.contact_head = nn.Sequential(

            nn.Linear(
                256,
                128
            ),

            nn.GELU(),

            nn.Linear(
                128,
                4
            )
        )


        # normalized raw-zero force location

        zero_force_norm = (

            (
                np.zeros(
                    3,
                    dtype=np.float32
                )

                -

                P_MEAN_V3[
                    9:12
                ]
            )

            /

            P_STD_V3[
                9:12
            ]
        )


        self.register_buffer(

            "zero_force_norm",

            torch.tensor(
                zero_force_norm,
                dtype=torch.float32
            )
        )


    # ==================================================================
    # CURRENT HISTORY -> PHYSICAL CONTEXT
    # ==================================================================

    def encode_context(
        self,
        p_hist,
        b_hist,
        u_hist,
        l_hist,
        e_hist,
        terrain_token
    ):

        batch_size = p_hist.shape[0]

        T_now = p_hist.shape[2]


        # --------------------------------------------------------------
        # replicate U history for each leg
        # --------------------------------------------------------------

        u_for_legs = (

            u_hist
            .unsqueeze(1)
            .expand(
                -1,
                4,
                -1,
                -1
            )
        )


        # B x 4 x 50 x 24

        leg_input = torch.cat(

            [
                p_hist,
                u_for_legs,
                l_hist,
                e_hist
            ],

            dim=-1
        )


        # apply SAME Transformer to all four legs

        leg_input = leg_input.reshape(

            batch_size * 4,

            T_now,

            24
        )


        leg_feature = self.leg_encoder(
            leg_input
        )


        # --------------------------------------------------------------
        # add explicit leg identity
        # --------------------------------------------------------------

        leg_ids = torch.arange(

            4,

            device=p_hist.device

        ).repeat(
            batch_size
        )


        leg_feature = (

            leg_feature

            +

            self.leg_identity(
                leg_ids
            )
        )


        leg_feature = leg_feature.reshape(

            batch_size,

            4,

            128
        )


        leg_feature_flat = leg_feature.reshape(

            batch_size,

            512
        )


        # --------------------------------------------------------------
        # body history
        # --------------------------------------------------------------

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


        # --------------------------------------------------------------
        # combine robot + terrain
        # --------------------------------------------------------------

        context = torch.cat(

            [
                leg_feature_flat,
                body_feature,
                terrain_token
            ],

            dim=-1
        )


        context = self.context_net(
            context
        )


        return context


    # ==================================================================
    # ONE 10-ms CONTROLLED TRANSITION
    # ==================================================================

    def one_step(
        self,
        p_hist,
        b_hist,
        u_hist,
        l_hist,
        e_hist,
        terrain_token
    ):

        batch_size = p_hist.shape[0]


        context = self.encode_context(

            p_hist,
            b_hist,
            u_hist,
            l_hist,
            e_hist,
            terrain_token
        )


        delta = self.response_head(
            context
        )


        # --------------------------------------------------------------
        # current state
        # --------------------------------------------------------------

        current_p = p_hist[
            :,
            :,
            -1,
            :
        ].clone()


        current_b = b_hist[
            :,
            -1,
            :
        ].clone()


        # --------------------------------------------------------------
        # PHYSICALLY MEANINGFUL BASELINE
        #
        # q,dq,F:
        # start from current state
        #
        # torque:
        # start from current commanded joint torque
        #
        # tau_measured(t+1)
        # =
        # tau_commanded(t)
        # +
        # learned response residual
        # --------------------------------------------------------------

        base_p = current_p.clone()


        current_l = l_hist[
            :,
            :,
            -1,
            :
        ]


        base_p[
            :,
            :,
            6:9
        ] = current_l


        base_state = torch.cat(

            [
                base_p.reshape(
                    batch_size,
                    48
                ),

                current_b
            ],

            dim=-1
        )


        predicted_state = (

            base_state

            +

            delta
        )


        predicted_p = predicted_state[
            :,
            :48
        ].reshape(

            batch_size,

            4,

            12
        )


        predicted_b = predicted_state[
            :,
            48:
        ]


        # --------------------------------------------------------------
        # CONTACT HEAD
        # --------------------------------------------------------------

        contact_logits = self.contact_head(
            context
        )


        contact_probability = torch.sigmoid(
            contact_logits
        )


        # --------------------------------------------------------------
        # CONTACT-CONDITIONED FORCE
        #
        # When p(contact) -> 0:
        # force -> physical zero.
        #
        # When p(contact) -> 1:
        # use learned force prediction.
        #
        # Work correctly in NORMALIZED coordinates.
        # --------------------------------------------------------------

        raw_force = predicted_p[
            :,
            :,
            9:12
        ]


        zero_force = (

            self.zero_force_norm
            .view(
                1,
                1,
                3
            )
        )


        gated_force = (

            contact_probability
            .unsqueeze(-1)

            *

            raw_force

            +

            (
                1.0

                -

                contact_probability
                .unsqueeze(-1)
            )

            *

            zero_force
        )


        predicted_p = predicted_p.clone()


        predicted_p[
            :,
            :,
            9:12
        ] = gated_force


        return (

            predicted_p,
            predicted_b,
            contact_logits
        )


    # ==================================================================
    # TRAINING ROLLOUT
    #
    # IMPORTANT:
    #
    # visual embedding is computed ONCE.
    #
    # predicted states are fed back.
    #
    # future recorded U and L advance the controls.
    #
    # absolutely NO future image is used.
    # ==================================================================

    def rollout(
        self,
        p_hist,
        b_hist,
        u_hist,
        l_hist,
        e_hist,
        dino,
        conv,
        depth,
        future_u,
        future_l
    ):


        # ==============================================================
        # COMPUTE TERRAIN ONCE AT INITIAL t
        # ==============================================================

        terrain_token = self.terrain_encoder(

            dino,
            conv,
            depth
        )


        rolling_p = p_hist
        rolling_b = b_hist
        rolling_u = u_hist
        rolling_l = l_hist
        rolling_e = e_hist


        predicted_p_sequence = []
        predicted_b_sequence = []
        contact_logit_sequence = []


        for k in range(
            HORIZON_V3
        ):


            # ==========================================================
            # current history already ends with:
            #
            # k=0 -> U[t],   L[t]
            #
            # k=1 -> U[t+1], L[t+1]
            #
            # etc.
            # ==========================================================

            pred_p, pred_b, contact_logits = self.one_step(

                rolling_p,
                rolling_b,
                rolling_u,
                rolling_l,
                rolling_e,
                terrain_token
            )


            predicted_p_sequence.append(
                pred_p
            )

            predicted_b_sequence.append(
                pred_b
            )

            contact_logit_sequence.append(
                contact_logits
            )


            if k == (
                HORIZON_V3 - 1
            ):

                break


            # ==========================================================
            # tracking residual for predicted next state
            #
            # tau_pred(t+k+1)
            # -
            # L(t+k)
            # ==========================================================

            command_used = rolling_l[
                :,
                :,
                -1,
                :
            ]


            e_next = (

                pred_p[
                    :,
                    :,
                    6:9
                ]

                -

                command_used
            )


            # ==========================================================
            # FEED BACK MODEL'S OWN STATE PREDICTION
            #
            # NOT ground-truth future state.
            # ==========================================================

            rolling_p = torch.cat(

                [
                    rolling_p[
                        :,
                        :,
                        1:,
                        :
                    ],

                    pred_p.unsqueeze(
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

                    pred_b.unsqueeze(
                        1
                    )
                ],

                dim=1
            )


            # ==========================================================
            # ADVANCE KNOWN TRAINING COMMANDS
            #
            # future_u[:, k+1]
            # future_l[:, k+1]
            # ==========================================================

            next_u = future_u[
                :,
                k+1,
                :
            ]


            rolling_u = torch.cat(

                [
                    rolling_u[
                        :,
                        1:,
                        :
                    ],

                    next_u.unsqueeze(
                        1
                    )
                ],

                dim=1
            )


            next_l = future_l[
                :,
                k+1,
                :,
                :
            ]


            # future_l:
            # B x H x 4 x 3
            #
            # history:
            # B x 4 x T x 3

            rolling_l = torch.cat(

                [
                    rolling_l[
                        :,
                        :,
                        1:,
                        :
                    ],

                    next_l.unsqueeze(
                        2
                    )
                ],

                dim=2
            )


            rolling_e = torch.cat(

                [
                    rolling_e[
                        :,
                        :,
                        1:,
                        :
                    ],

                    e_next.unsqueeze(
                        2
                    )
                ],

                dim=2
            )


        predicted_p_sequence = torch.stack(

            predicted_p_sequence,

            dim=1
        )


        predicted_b_sequence = torch.stack(

            predicted_b_sequence,

            dim=1
        )


        contact_logit_sequence = torch.stack(

            contact_logit_sequence,

            dim=1
        )


        return {

            "p":
                predicted_p_sequence,

            "b":
                predicted_b_sequence,

            "contact_logits":
                contact_logit_sequence,

            "terrain":
                terrain_token
        }

# %% notebook cell 7
# ======================================================================
# V3 CELL 4
# ======================================================================

device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"
)


model_v3 = ControlledTerrainDynamicsV3(

    dino_dim=
        DINO_FEATURES.shape[1],

    conv_dim=
        CONV_FEATURES.reshape(
            len(CONV_FEATURES),
            -1
        ).shape[1]

).to(
    device
)


def move_v3(
    batch
):

    return {

        key:
            value.to(
                device,
                non_blocking=True
            )

        for key, value
        in batch.items()
    }


batch = next(
    iter(
        train_loader_v3
    )
)


batch = move_v3(
    batch
)


model_v3.eval()


with torch.no_grad():

    out = model_v3.rollout(

        batch["p_hist"],
        batch["b_hist"],
        batch["u_hist"],
        batch["l_hist"],
        batch["e_hist"],

        batch["dino"],
        batch["conv"],
        batch["depth"],

        batch["future_u"],
        batch["future_l"]
    )


print(
    "Predicted P:",
    out["p"].shape
)

print(
    "Predicted B:",
    out["b"].shape
)

print(
    "Contact:",
    out["contact_logits"].shape
)

print(
    "Terrain token:",
    out["terrain"].shape
)

# %% notebook cell 8
# ======================================================================
# V3 CELL 5 — TRAIN
# ======================================================================

optimizer_v3 = torch.optim.AdamW(

    model_v3.parameters(),

    lr=2e-4,

    weight_decay=1e-4
)


contact_loss_fn = nn.BCEWithLogitsLoss(
    reduction="none"
)


EPOCHS_V3 = 8


# later horizons matter slightly more

HORIZON_WEIGHTS_V3 = torch.linspace(

    1.0,
    2.0,
    HORIZON_V3,

    device=device
)


def horizon_mse(
    predicted,
    target
):

    # predicted/target:
    #
    # B x H x ...

    error = (
        predicted - target
    ) ** 2


    # average every dimension except B,H

    dimensions = tuple(
        range(
            2,
            error.ndim
        )
    )


    error_per_step = error.mean(
        dim=dimensions
    )


    return (

        error_per_step

        *

        HORIZON_WEIGHTS_V3[
            None,
            :
        ]

    ).mean()


def calculate_v3_loss(
    out,
    batch
):

    pred_p = out["p"]
    pred_b = out["b"]


    true_p = batch[
        "target_p"
    ]


    true_b = batch[
        "target_b"
    ]


    # --------------------------------------------------------------
    # q
    # --------------------------------------------------------------

    loss_q = horizon_mse(

        pred_p[
            :,
            :,
            :,
            0:3
        ],

        true_p[
            :,
            :,
            :,
            0:3
        ]
    )


    # --------------------------------------------------------------
    # dq
    # --------------------------------------------------------------

    loss_dq = horizon_mse(

        pred_p[
            :,
            :,
            :,
            3:6
        ],

        true_p[
            :,
            :,
            :,
            3:6
        ]
    )


    # --------------------------------------------------------------
    # measured torque
    #
    # internally this is predicted as:
    #
    # commanded torque + learned residual
    # --------------------------------------------------------------

    loss_tau = horizon_mse(

        pred_p[
            :,
            :,
            :,
            6:9
        ],

        true_p[
            :,
            :,
            :,
            6:9
        ]
    )


    # --------------------------------------------------------------
    # contact forces
    # --------------------------------------------------------------

    loss_force = horizon_mse(

        pred_p[
            :,
            :,
            :,
            9:12
        ],

        true_p[
            :,
            :,
            :,
            9:12
        ]
    )


    # --------------------------------------------------------------
    # body state auxiliary loss
    # --------------------------------------------------------------

    loss_body = horizon_mse(

        pred_b,

        true_b
    )


    # --------------------------------------------------------------
    # contact classification
    # --------------------------------------------------------------

    contact_error = contact_loss_fn(

        out[
            "contact_logits"
        ],

        batch[
            "target_contact"
        ]
    )


    contact_per_step = contact_error.mean(
        dim=-1
    )


    loss_contact = (

        contact_per_step

        *

        HORIZON_WEIGHTS_V3[
            None,
            :
        ]

    ).mean()


    # equalized physical groups
    #
    # slightly stronger force emphasis
    # body/contact auxiliary

    total = (

        loss_q

        +

        loss_dq

        +

        loss_tau

        +

        1.25 * loss_force

        +

        0.5 * loss_body

        +

        0.25 * loss_contact
    )


    return {

        "total":
            total,

        "q":
            loss_q,

        "dq":
            loss_dq,

        "tau":
            loss_tau,

        "force":
            loss_force,

        "body":
            loss_body,

        "contact":
            loss_contact
    }


best_val_v3 = float(
    "inf"
)

best_state_v3 = None


for epoch in range(
    1,
    EPOCHS_V3 + 1
):

    # ==============================================================
    # TRAIN
    # ==============================================================

    model_v3.train()


    train_running = []


    for batch in train_loader_v3:

        batch = move_v3(
            batch
        )


        optimizer_v3.zero_grad(
            set_to_none=True
        )


        out = model_v3.rollout(

            batch["p_hist"],
            batch["b_hist"],
            batch["u_hist"],
            batch["l_hist"],
            batch["e_hist"],

            batch["dino"],
            batch["conv"],
            batch["depth"],

            batch["future_u"],
            batch["future_l"]
        )


        losses = calculate_v3_loss(

            out,
            batch
        )


        losses[
            "total"
        ].backward()


        torch.nn.utils.clip_grad_norm_(

            model_v3.parameters(),

            1.0
        )


        optimizer_v3.step()


        train_running.append(

            float(
                losses[
                    "total"
                ].item()
            )
        )


    # ==============================================================
    # VALIDATION
    # ==============================================================

    model_v3.eval()


    validation_running = []


    with torch.no_grad():

        for batch in val_loader_v3:

            batch = move_v3(
                batch
            )


            out = model_v3.rollout(

                batch["p_hist"],
                batch["b_hist"],
                batch["u_hist"],
                batch["l_hist"],
                batch["e_hist"],

                batch["dino"],
                batch["conv"],
                batch["depth"],

                batch["future_u"],
                batch["future_l"]
            )


            losses = calculate_v3_loss(

                out,
                batch
            )


            validation_running.append(

                float(
                    losses[
                        "total"
                    ].item()
                )
            )


    train_loss = float(
        np.mean(
            train_running
        )
    )


    val_loss = float(
        np.mean(
            validation_running
        )
    )


    print(

        f"Epoch {epoch:02d}/{EPOCHS_V3}"

        f" | train={train_loss:.6f}"

        f" | val={val_loss:.6f}"
    )


    if val_loss < best_val_v3:

        best_val_v3 = val_loss


        best_state_v3 = {

            key:
                value
                .detach()
                .cpu()
                .clone()

            for key, value
            in model_v3
            .state_dict()
            .items()
        }


model_v3.load_state_dict(
    best_state_v3
)


print(
    "\nTraining complete."
)

print(
    "Best validation loss:",
    best_val_v3
)

# %% notebook cell 9
# ======================================================================
# V3 TESTING — FOUR UNSEEN TEST TIMES
#
# ttest:
#   102 s
#   108 s
#   114 s
#   119 s
#
# Predict:
#   ttest + 0.1 s
#
# Compare against:
#   actual P and B at ttest + 0.1 s = index t + 10
#
# TESTING RULES:
#
#   ✓ use sensor history t-49 ... t
#   ✓ use U history t-49 ... t
#   ✓ use L history t-49 ... t
#   ✓ use tracking-error history
#   ✓ use ONLY RGB-D at t
#
#   ✗ NO RGB-D[t+1 : t+9]
#   ✗ NO actual future sensor values as model input
#   ✗ NO recorded future U[t+1:t+9]
#   ✗ NO recorded future L[t+1:t+9]
#
# For the 100-ms hypothetical rollout:
#
#   U_future = U[t] held constant
#   L_future = L[t] held constant
#
# ======================================================================

import numpy as np
import pandas as pd
import torch


# ======================================================================
# 1. TEST TIMES
# ======================================================================

TEST_TIMES_V3 = [
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

    "tau1",
    "tau2",
    "tau3",

    "Fx",
    "Fy",
    "Fz"
]


BODY_NAMES = [
    "acc_x",
    "acc_y",
    "acc_z",

    "gyro_x",
    "gyro_y",
    "gyro_z",

    "vx",
    "vy",
    "vz",

    "wx",
    "wy",
    "wz"
]


# ======================================================================
# 2. SANITY CHECK REQUIRED VARIABLES
# ======================================================================

required = [

    "model_v3",

    "P",
    "B",
    "U",

    "L_LEG",
    "L_PNORM",
    "E_TAU_NORM",

    "time",

    "DINO_FEATURES",
    "CONV_FEATURES",
    "DEPTH_GRIDS",

    "visual_index_for_state",

    "P_MEAN_V3",
    "P_STD_V3",

    "B_MEAN_V3",
    "B_STD_V3"
]


missing = [

    name
    for name in required

    if name not in globals()
]


if missing:

    raise RuntimeError(
        "Missing required V3 variables:\n"
        + str(missing)
    )


model_v3.eval()


# ======================================================================
# 3. HELPER — FIND STATE INDEX CLOSEST TO TEST TIME
# ======================================================================

def index_from_time(
    requested_time
):

    idx = int(

        np.argmin(
            np.abs(
                time - requested_time
            )
        )
    )


    if idx < HISTORY_V3 - 1:

        raise RuntimeError(
            "Not enough history."
        )


    if idx + HORIZON_V3 >= len(P):

        raise RuntimeError(
            "Not enough future ground truth."
        )


    return idx


# ======================================================================
# 4. CREATE ONE DEPLOYMENT-STYLE TEST SAMPLE
# ======================================================================

def make_test_input_v3(
    t
):

    h0 = (
        t
        -
        HISTORY_V3
        +
        1
    )

    h1 = t + 1


    # --------------------------------------------------------------
    # SENSOR HISTORY
    # --------------------------------------------------------------

    p_hist = norm_p_v3(

        P[
            h0:h1
        ]

    ).astype(
        np.float32
    )


    # 50 x 4 x 12
    # ->
    # 4 x 50 x 12

    p_hist = np.transpose(

        p_hist,

        (1, 0, 2)
    )


    # --------------------------------------------------------------
    # BODY HISTORY
    # --------------------------------------------------------------

    b_hist = norm_b_v3(

        B[
            h0:h1
        ]

    ).astype(
        np.float32
    )


    # --------------------------------------------------------------
    # HIGH-LEVEL COMMAND HISTORY
    # --------------------------------------------------------------

    u_hist = norm_u_v3(

        U[
            h0:h1
        ]

    ).astype(
        np.float32
    )


    # --------------------------------------------------------------
    # LOW-LEVEL COMMAND HISTORY
    # --------------------------------------------------------------

    l_hist = L_PNORM[
        h0:h1
    ]


    l_hist = np.transpose(

        l_hist,

        (1, 0, 2)

    ).astype(
        np.float32
    )


    # --------------------------------------------------------------
    # HISTORICAL COMMAND -> SENSOR RESPONSE ERROR
    # --------------------------------------------------------------

    e_hist = E_TAU_NORM[
        h0:h1
    ]


    e_hist = np.transpose(

        e_hist,

        (1, 0, 2)

    ).astype(
        np.float32
    )


    # ==============================================================
    # ONLY VISUAL INFORMATION AVAILABLE AT t
    # ==============================================================

    vi = int(

        visual_index_for_state[
            t
        ]
    )


    if vi < 0:

        raise RuntimeError(
            f"No causal RGB-D at state {t}"
        )


    dino = np.asarray(

        DINO_FEATURES[
            vi
        ],

        dtype=np.float32

    ).reshape(-1)


    conv = np.asarray(

        CONV_FEATURES[
            vi
        ],

        dtype=np.float32

    ).reshape(-1)


    depth = np.asarray(

        DEPTH_GRIDS[
            vi
        ],

        dtype=np.float32
    )


    # ==============================================================
    # DEPLOYMENT TESTING:
    #
    # Future recorded commands are UNKNOWN.
    #
    # Therefore hold CURRENT command constant.
    #
    # U[t], U[t], ..., U[t]
    # L[t], L[t], ..., L[t]
    # ==============================================================

    u_t_normalized = norm_u_v3(

        U[t]

    ).astype(
        np.float32
    )


    future_u = np.repeat(

        u_t_normalized[
            None,
            :
        ],

        HORIZON_V3,

        axis=0
    )


    l_t_normalized = L_PNORM[
        t
    ].astype(
        np.float32
    )


    future_l = np.repeat(

        l_t_normalized[
            None,
            :,
            :
        ],

        HORIZON_V3,

        axis=0
    )


    # ==================================================================
    # ADD BATCH DIMENSION
    # ==================================================================

    result = {

        "p_hist":
            torch.tensor(
                p_hist
            ).unsqueeze(0).to(device),

        "b_hist":
            torch.tensor(
                b_hist
            ).unsqueeze(0).to(device),

        "u_hist":
            torch.tensor(
                u_hist
            ).unsqueeze(0).to(device),

        "l_hist":
            torch.tensor(
                l_hist
            ).unsqueeze(0).to(device),

        "e_hist":
            torch.tensor(
                e_hist
            ).unsqueeze(0).to(device),

        "dino":
            torch.tensor(
                dino
            ).unsqueeze(0).to(device),

        "conv":
            torch.tensor(
                conv
            ).unsqueeze(0).to(device),

        "depth":
            torch.tensor(
                depth
            ).unsqueeze(0).to(device),

        "future_u":
            torch.tensor(
                future_u
            ).unsqueeze(0).to(device),

        "future_l":
            torch.tensor(
                future_l
            ).unsqueeze(0).to(device)
    }


    return result


# ======================================================================
# 5. RUN ONE TEST
# ======================================================================

def evaluate_one_test_v3(
    requested_time
):

    t = index_from_time(
        requested_time
    )


    test_input = make_test_input_v3(
        t
    )


    # ==============================================================
    # MODEL PREDICTION
    # ==============================================================

    with torch.no_grad():

        output = model_v3.rollout(

            test_input[
                "p_hist"
            ],

            test_input[
                "b_hist"
            ],

            test_input[
                "u_hist"
            ],

            test_input[
                "l_hist"
            ],

            test_input[
                "e_hist"
            ],

            test_input[
                "dino"
            ],

            test_input[
                "conv"
            ],

            test_input[
                "depth"
            ],

            test_input[
                "future_u"
            ],

            test_input[
                "future_l"
            ]
        )


    # ==============================================================
    # PREDICTION AT THE FINAL 100-ms POINT
    #
    # step index 9 corresponds to t+10
    # ==============================================================

    pred_p_n = (

        output[
            "p"
        ][
            0,
            -1
        ]
        .cpu()
        .numpy()
    )


    pred_b_n = (

        output[
            "b"
        ][
            0,
            -1
        ]
        .cpu()
        .numpy()
    )


    # --------------------------------------------------------------
    # convert prediction back to physical units
    # --------------------------------------------------------------

    pred_p = (

        pred_p_n

        *

        P_STD_V3[
            None,
            :
        ]

        +

        P_MEAN_V3[
            None,
            :
        ]
    )


    pred_b = (

        pred_b_n

        *

        B_STD_V3

        +

        B_MEAN_V3
    )


    # ==============================================================
    # TRUE SENSOR VALUES 100 ms LATER
    # ==============================================================

    target_index = (
        t
        +
        HORIZON_V3
    )


    true_p = P[
        target_index
    ].copy()


    true_b = B[
        target_index
    ].copy()


    # ==============================================================
    # CONTACT PREDICTION
    # ==============================================================

    contact_probability = (

        torch.sigmoid(

            output[
                "contact_logits"
            ][
                0,
                -1
            ]

        )

        .cpu()
        .numpy()
    )


    predicted_contact = (

        contact_probability
        >=
        0.5

    ).astype(int)


    actual_contact = (

        np.linalg.norm(

            true_p[
                :,
                9:12
            ],

            axis=-1
        )

        >

        CONTACT_THRESHOLD_V3

    ).astype(int)


    # ==============================================================
    # ERRORS
    # ==============================================================

    error_p = (
        pred_p
        -
        true_p
    )


    error_b = (
        pred_b
        -
        true_b
    )


    # --------------------------------------------------------------
    # physically meaningful group RMSEs
    # --------------------------------------------------------------

    rmse_q = np.sqrt(

        np.mean(

            error_p[
                :,
                0:3
            ] ** 2
        )
    )


    rmse_dq = np.sqrt(

        np.mean(

            error_p[
                :,
                3:6
            ] ** 2
        )
    )


    rmse_tau = np.sqrt(

        np.mean(

            error_p[
                :,
                6:9
            ] ** 2
        )
    )


    rmse_force = np.sqrt(

        np.mean(

            error_p[
                :,
                9:12
            ] ** 2
        )
    )


    rmse_body = np.sqrt(

        np.mean(
            error_b ** 2
        )
    )


    # ==============================================================
    # LONG-FORM SENSOR TABLE
    # ==============================================================

    sensor_rows = []


    for leg_idx, leg_name in enumerate(
        LEG_NAMES
    ):

        for variable_idx, variable_name in enumerate(
            P_NAMES
        ):

            sensor_rows.append(

                {
                    "test_time_s":
                        float(
                            time[t]
                        ),

                    "prediction_time_s":
                        float(
                            time[
                                target_index
                            ]
                        ),

                    "leg":
                        leg_name,

                    "sensor":
                        variable_name,

                    "predicted":
                        float(
                            pred_p[
                                leg_idx,
                                variable_idx
                            ]
                        ),

                    "actual":
                        float(
                            true_p[
                                leg_idx,
                                variable_idx
                            ]
                        ),

                    "error":
                        float(
                            error_p[
                                leg_idx,
                                variable_idx
                            ]
                        )
                }
            )


    sensor_table = pd.DataFrame(
        sensor_rows
    )


    # ==============================================================
    # BODY TABLE
    # ==============================================================

    body_rows = []


    for j, name in enumerate(
        BODY_NAMES
    ):

        body_rows.append(

            {
                "test_time_s":
                    float(
                        time[t]
                    ),

                "prediction_time_s":
                    float(
                        time[
                            target_index
                        ]
                    ),

                "sensor":
                    name,

                "predicted":
                    float(
                        pred_b[j]
                    ),

                "actual":
                    float(
                        true_b[j]
                    ),

                "error":
                    float(
                        error_b[j]
                    )
            }
        )


    body_table = pd.DataFrame(
        body_rows
    )


    # ==============================================================
    # CONTACT TABLE
    # ==============================================================

    contact_table = pd.DataFrame(

        {
            "leg":
                LEG_NAMES,

            "probability":
                contact_probability,

            "predicted_contact":
                predicted_contact,

            "actual_contact":
                actual_contact
        }
    )


    summary = {

        "requested_test_s":
            requested_time,

        "actual_test_s":
            float(
                time[t]
            ),

        "target_time_s":
            float(
                time[
                    target_index
                ]
            ),

        "actual_horizon_s":
            float(
                time[
                    target_index
                ]
                -
                time[t]
            ),

        "q_RMSE":
            float(
                rmse_q
            ),

        "dq_RMSE":
            float(
                rmse_dq
            ),

        "tau_RMSE":
            float(
                rmse_tau
            ),

        "force_RMSE":
            float(
                rmse_force
            ),

        "body_RMSE":
            float(
                rmse_body
            ),

        "contact_correct":
            int(
                (
                    predicted_contact
                    ==
                    actual_contact
                ).sum()
            ),

        "contact_total":
            4
    }


    return {

        "summary":
            summary,

        "sensors":
            sensor_table,

        "body":
            body_table,

        "contact":
            contact_table,

        "pred_p":
            pred_p,

        "true_p":
            true_p,

        "pred_b":
            pred_b,

        "true_b":
            true_b
    }


# ======================================================================
# 6. RUN ALL FOUR TESTS
# ======================================================================

results_v3 = []


for ttest in TEST_TIMES_V3:

    result = evaluate_one_test_v3(
        ttest
    )

    results_v3.append(
        result
    )


# ======================================================================
# 7. SUMMARY TABLE
# ======================================================================

summary_v3 = pd.DataFrame(

    [
        r["summary"]

        for r in results_v3
    ]
)


print(
    "\n"
    +
    "=" * 100
)

print(
    "V3 — FOUR 100-ms TEST RESULTS"
)

print(
    "=" * 100
)


display(
    summary_v3
)


# ======================================================================
# 8. PRINT FULL SENSOR RESULTS
# ======================================================================

for i, result in enumerate(
    results_v3
):

    ttest = TEST_TIMES_V3[i]


    print(
        "\n\n"
        +
        "=" * 100
    )

    print(
        f"TEST {i+1}: "
        f"ttest = {ttest:.1f} s"
    )

    print(
        "=" * 100
    )


    print(
        "\nALL FOUR-LEG SENSOR VALUES "
        "(prediction vs actual at +0.1 s)"
    )


    display(
        result[
            "sensors"
        ]
    )


    print(
        "\nBODY / IMU VALUES"
    )


    display(
        result[
            "body"
        ]
    )


    print(
        "\nCONTACT"
    )


    display(
        result[
            "contact"
        ]
    )


# ======================================================================
# 9. OPTIONAL SAVE
# ======================================================================

all_sensor_results_v3 = pd.concat(

    [
        r["sensors"]

        for r in results_v3
    ],

    ignore_index=True
)


all_body_results_v3 = pd.concat(

    [
        r["body"]

        for r in results_v3
    ],

    ignore_index=True
)


all_sensor_results_v3.to_csv(

    "/content/V3_four_test_sensor_predictions.csv",

    index=False
)


all_body_results_v3.to_csv(

    "/content/V3_four_test_body_predictions.csv",

    index=False
)


summary_v3.to_csv(

    "/content/V3_four_test_summary.csv",

    index=False
)


print(
    "\nSaved:"
)

print(
    "/content/V3_four_test_sensor_predictions.csv"
)

print(
    "/content/V3_four_test_body_predictions.csv"
)

print(
    "/content/V3_four_test_summary.csv"
)

# %% notebook cell 10
# NOTEBOOK: conclusion:_
# NOTEBOOK: with the V3 prototype tau and q values are predicted very nicely and the error of F x F y F z values for each leg has gone down
# NOTEBOOK: significantly compared to prototype_V2
# NOTEBOOK: 
# NOTEBOOK: next changes that we can work on is
# NOTEBOOK: less descretisation of C values can lead to more accurate foot force predictions I think

# %% notebook cell 11
# ======================================================================
# SHOW EXACT RGB + DEPTH USED FOR THE FOUR V3 TEST EXAMPLES
#
# Tests:
#   102 s
#   108 s
#   114 s
#   119 s
#
# For each test:
#   - find robot state index
#   - find exact CAUSAL camera frame used by model
#   - show RGB
#   - show metric depth
#
# Also saves all 8 images to /content/V3_test_visuals/
# ======================================================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


TEST_TIMES_V3 = [
    102.0,
    108.0,
    114.0,
    119.0
]


SAVE_DIR = "/content/V3_test_visuals"

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)


# ======================================================================
# 1. METRIC DEPTH DECODER
# ======================================================================

def read_metric_depth_for_test(path):

    raw = cv2.imread(
        path,
        cv2.IMREAD_UNCHANGED
    )


    if raw is None:

        raise RuntimeError(
            f"Could not read depth image:\n{path}"
        )


    # TartanGround packed little-endian float32 format

    if (
        raw.dtype == np.uint8
        and
        raw.ndim == 3
        and
        raw.shape[2] == 4
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
        and
        raw.ndim == 2
    ):

        depth = raw


    else:

        raise RuntimeError(
            f"Unexpected depth format: "
            f"dtype={raw.dtype}, "
            f"shape={raw.shape}"
        )


    depth = depth.astype(
        np.float32
    )


    depth[
        ~np.isfinite(depth)
    ] = np.nan


    return depth


# ======================================================================
# 2. FIND EXACT FRAME USED FOR EACH TEST
# ======================================================================

test_visual_info = []


for test_number, requested_time in enumerate(
    TEST_TIMES_V3,
    start=1
):

    # --------------------------------------------------------------
    # robot-state index closest to requested ttest
    # --------------------------------------------------------------

    t_idx = int(

        np.argmin(
            np.abs(
                time - requested_time
            )
        )
    )


    actual_robot_time = float(
        time[t_idx]
    )


    # --------------------------------------------------------------
    # EXACT causal visual frame used by Transformer
    # --------------------------------------------------------------

    vi = int(
        visual_index_for_state[
            t_idx
        ]
    )


    if vi < 0:

        raise RuntimeError(
            f"No causal camera frame for "
            f"test time {requested_time}"
        )


    camera_time_used = float(
        camera_times[
            vi
        ]
    )


    frame_id = str(
        frame_ids[
            vi
        ]
    )


    # --------------------------------------------------------------
    # file paths
    # --------------------------------------------------------------

    rgb_path = os.path.join(

        str(RGB_DIR),

        f"{frame_id}_lcam_front.png"
    )


    depth_path = os.path.join(

        str(DEPTH_DIR),

        f"{frame_id}_lcam_front_depth.png"
    )


    assert os.path.exists(
        rgb_path
    )

    assert os.path.exists(
        depth_path
    )


    # --------------------------------------------------------------
    # load RGB
    # --------------------------------------------------------------

    rgb = np.asarray(

        Image.open(
            rgb_path
        ).convert(
            "RGB"
        )
    )


    # --------------------------------------------------------------
    # load metric depth
    # --------------------------------------------------------------

    depth = read_metric_depth_for_test(
        depth_path
    )


    # --------------------------------------------------------------
    # save easy-to-view RGB copy
    # --------------------------------------------------------------

    rgb_save_path = os.path.join(

        SAVE_DIR,

        f"test_{test_number}_"
        f"{requested_time:.1f}s_RGB.png"
    )


    Image.fromarray(
        rgb
    ).save(
        rgb_save_path
    )


    # --------------------------------------------------------------
    # save depth visualization
    #
    # This is a visualization PNG.
    # Actual metric depth values remain in `depth`.
    # --------------------------------------------------------------

    depth_display = depth.copy()


    valid = np.isfinite(
        depth_display
    )


    # Limit display to nearby range so terrain is visible

    depth_display[
        ~valid
    ] = 0.0


    depth_display = np.clip(
        depth_display,
        0.0,
        4.0
    )


    depth_uint8 = (

        depth_display
        /
        4.0
        *
        255.0

    ).astype(
        np.uint8
    )


    depth_save_path = os.path.join(

        SAVE_DIR,

        f"test_{test_number}_"
        f"{requested_time:.1f}s_DEPTH.png"
    )


    cv2.imwrite(
        depth_save_path,
        depth_uint8
    )


    test_visual_info.append(

        {
            "test_number":
                test_number,

            "requested_test_time":
                requested_time,

            "robot_state_time":
                actual_robot_time,

            "camera_time_used":
                camera_time_used,

            "camera_age_ms":
                (
                    actual_robot_time
                    -
                    camera_time_used
                )
                * 1000.0,

            "visual_index":
                vi,

            "frame_id":
                frame_id,

            "rgb":
                rgb,

            "depth":
                depth,

            "rgb_path":
                rgb_save_path,

            "depth_path":
                depth_save_path
        }
    )


# ======================================================================
# 3. PRINT EXACT TIMING INFORMATION
# ======================================================================

print(
    "\nEXACT VISUAL FRAMES USED BY V3"
)

print(
    "=" * 90
)


for item in test_visual_info:

    print(
        f"\nTEST {item['test_number']}"
    )

    print(
        "Requested ttest :",
        item[
            "requested_test_time"
        ],
        "s"
    )

    print(
        "Robot state time:",
        item[
            "robot_state_time"
        ],
        "s"
    )

    print(
        "Camera time used:",
        item[
            "camera_time_used"
        ],
        "s"
    )

    print(
        "Camera age:",
        f"{item['camera_age_ms']:.2f}",
        "ms"
    )

    print(
        "Frame ID:",
        item[
            "frame_id"
        ]
    )


# ======================================================================
# 4. SHOW FOUR RGB IMAGES
# ======================================================================

fig, axes = plt.subplots(

    1,
    4,

    figsize=(
        20,
        5
    )
)


for ax, item in zip(
    axes,
    test_visual_info
):

    ax.imshow(
        item[
            "rgb"
        ]
    )

    ax.set_title(

        f"Test {item['test_number']}\n"
        f"ttest={item['requested_test_time']:.1f}s\n"
        f"cam={item['camera_time_used']:.3f}s"
    )

    ax.axis(
        "off"
    )


plt.tight_layout()
plt.show()


# ======================================================================
# 5. SHOW FOUR DEPTH IMAGES
# ======================================================================

fig, axes = plt.subplots(

    1,
    4,

    figsize=(
        20,
        5
    )
)


for ax, item in zip(
    axes,
    test_visual_info
):

    depth = item[
        "depth"
    ].copy()


    # focus visualization on nearby terrain

    depth[
        depth <= 0
    ] = np.nan


    im = ax.imshow(

        depth,

        vmin=0.25,
        vmax=4.0
    )


    ax.set_title(

        f"Test {item['test_number']}\n"
        f"ttest={item['requested_test_time']:.1f}s\n"
        f"metric depth"
    )

    ax.axis(
        "off"
    )


fig.colorbar(

    im,

    ax=axes.tolist(),

    label="Depth (metres)",

    shrink=0.75
)


plt.show()


print(
    "\nSaved all images in:"
)

print(
    SAVE_DIR
)

# %% notebook cell 12
# Install only the extra discovery dependency if needed.
# IPYTHON: !pip install -q -U huggingface_hub

# %% notebook cell 13
# V3 multi-terrain, multi-timescale experiment
# Paste/run AFTER the existing prototype_v3_1 notebook cells.
#
# 4 terrain families x 4 test cases.
# EACH test case creates TWO completely fresh models:
#   short expert: 0.5 s history, 0.3 s training-window hop, trained on 0.1 s rollout
#   long expert : 4.0 s history, 1.0 s training-window hop, trained on 1.0 s rollout
# Training uses ONLY the 20 s preceding each t_test.
# Window losses are exponentially recency-weighted: targets closer to t_test matter more.
# Final prediction at h=0.1,...,1.0 s is a Gaussian-weighted interpolation of the two experts.
#
# Offline dynamics benchmark uses the RECORDED future U/L command sequence; no future P/B/RGB-D is input.

import os, re, gc, math, random
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from rosbags.highlevel import AnyReader

import tartanair as ta
from huggingface_hub import HfApi
from transformers import AutoImageProcessor, AutoModel
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

# ----------------------------- CONFIG ---------------------------------
ROOT = Path('/content/tartanground')
CACHE_ROOT = Path('/content/v3_multiterrain_cache')
OUT_ROOT = Path('/content/v3_multiterrain_results')
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
OUT_ROOT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 781

TRAIN_LOOKBACK_SEC = 20.0
SHORT_CONTEXT_SEC = 0.5
SHORT_HOP_SEC = 0.3
SHORT_TRAIN_HORIZON_SEC = 0.1
LONG_CONTEXT_SEC = 4.0
LONG_HOP_SEC = 1.0
LONG_TRAIN_HORIZON_SEC = 1.0
TEST_HORIZON_SEC = 1.0
OUTPUT_INTERVAL_SEC = 0.1

# Recency weighting across training windows.
# A target this many seconds older gets half the weight.
SHORT_RECENCY_HALF_LIFE_SEC = 5.0
LONG_RECENCY_HALF_LIFE_SEC = 6.0

EPOCHS_SHORT = 8
EPOCHS_LONG = 4       # long 100-step rollouts are expensive; raise after first successful run
LR = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

# Gaussian expert reliability for final interpolation.
SHORT_NATIVE_HORIZON = 0.1
LONG_NATIVE_HORIZON = 1.0
SIGMA_SHORT = 0.28
SIGMA_LONG = 0.42

CONTACT_THRESHOLD = 10.0
EPS = 1e-6

ROI_X_MIN, ROI_X_MAX = 0.18, 0.82
ROI_Y_MIN, ROI_Y_MAX = 0.45, 0.97
DEPTH_MIN_METERS, DEPTH_MAX_METERS = 0.25, 2.0
DEPTH_SIZE = 64
VIS_BATCH = 16

# These are environment proxies. TartanGround does NOT provide physical material labels like "mud".
TERRAIN_CANDIDATES = {
    'forest': ['ForestEnv', 'SeasonalForestAutumn', 'OldScandinavia'],
    'swamp_marsh': ['GreatMarsh'],
    'rough_mixed': ['TerrainBlending', 'Ruins', 'CoalMine', 'GothicIsland'],
    'mud_wet_uneven_proxy': ['WaterMillDay', 'FactoryWeather', 'OldTownFall', 'SeasonalForestAutumn'],
}
PREFERRED_RUN = {'ForestEnv': 'P2000', 'GreatMarsh': 'P2023'}

# ------------------------- BASIC HELPERS -------------------------------
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def causal_hold(t_src, x_src, t_target):
    idx = np.searchsorted(t_src, t_target, side='right') - 1
    if np.any(idx < 0):
        raise RuntimeError('Target time occurs before first source measurement.')
    return x_src[idx]


def stamp_to_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def load_metric_depth(path):
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError(f'Could not read {path}')
    if raw.dtype == np.uint8 and raw.ndim == 3 and raw.shape[2] == 4:
        raw = np.ascontiguousarray(raw)
        depth = raw.view('<f4').reshape(raw.shape[0], raw.shape[1])
    elif raw.dtype == np.float32 and raw.ndim == 2:
        depth = raw
    else:
        raise RuntimeError(f'Unexpected depth format: {raw.dtype}, {raw.shape}')
    depth = depth.astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    return depth


def nearest_index(time, t):
    return int(np.argmin(np.abs(time - float(t))))

# ----------------------- TERRAIN/RUN DISCOVERY -------------------------
def list_anymal_runs(env):
    api = HfApi()
    try:
        items = list(api.list_repo_tree(
            repo_id='theairlabcmu/TartanGround',
            repo_type='dataset',
            path_in_repo=f'{env}/Data_anymal',
            recursive=False,
        ))
    except Exception:
        return []
    runs = []
    for item in items:
        name = Path(item.path).name
        if item.__class__.__name__ == 'RepoFolder' and re.fullmatch(r'P\d+', name):
            runs.append(name)
    return sorted(runs)


def build_terrain_run_map():
    used_envs = set()
    rows = []
    for terrain_type, candidates in TERRAIN_CANDIDATES.items():
        chosen = None
        for env in candidates:
            if env in used_envs:
                continue
            runs = list_anymal_runs(env)
            if runs:
                pref = PREFERRED_RUN.get(env)
                run = pref if pref in runs else ('P2000' if 'P2000' in runs else runs[0])
                chosen = (env, run, runs)
                break
        if chosen is None:
            raise RuntimeError(f'No ANYmal environment found for terrain family: {terrain_type}')
        env, run, runs = chosen
        used_envs.add(env)
        rows.append({
            'terrain_type': terrain_type,
            'environment': env,
            'chosen_run': run,
            'n_available_runs': len(runs),
            'available_runs': ','.join(runs),
        })
    df = pd.DataFrame(rows)
    print('\nTERRAIN / RUN MAP')
    display(df[['terrain_type','environment','chosen_run','n_available_runs']])
    print('NOTE: mud/wet/rough names are environment proxies; the 16 test images are the visual verification.')
    return df

# -------------------------- DOWNLOAD -----------------------------------
def download_run(env, run):
    ta.init(str(ROOT))
    ta.download_ground(
        env=[env], version=['anymal'], traj=[run],
        modality=['imu','rosbag','image','depth'],
        camera_name=['lcam_front'], unzip=True,
    )
    base = ROOT / env / 'Data_anymal' / run
    required = [
        base/'imu', base/'rosbags'/f'{run}.bag',
        base/'image_lcam_front', base/'depth_lcam_front'
    ]
    for p in required:
        if not p.exists():
            raise RuntimeError(f'Missing downloaded path: {p}')
    return base

# ------------------------ PHYSICS EXTRACTION ---------------------------
def extract_run_physics(base, run):
    cache = CACHE_ROOT / f'{base.parent.parent.name}_{run}_physics.npz'
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return {k: z[k] for k in z.files}

    imu_dir = base/'imu'
    bag_path = base/'rosbags'/f'{run}.bag'
    imu_time = np.load(imu_dir/'imu_time.npy').astype(np.float64)
    acc = np.load(imu_dir/'acc.npy').astype(np.float32)
    gyro = np.loadtxt(imu_dir/'gyro.txt').astype(np.float32)

    topics = {
        'joints':'/state_estimator/joint_states',
        'lf':'/state_estimator/contact_force_lf_foot',
        'rf':'/state_estimator/contact_force_rf_foot',
        'lh':'/state_estimator/contact_force_lh_foot',
        'rh':'/state_estimator/contact_force_rh_foot',
        'twist':'/state_estimator/twist',
        'command':'/path_planning_and_following/twist',
    }
    physical_topics = {topics[k] for k in ['joints','lf','rf','lh','rh','twist']}
    with AnyReader([bag_path]) as reader:
        selected = [c for c in reader.connections if c.topic in physical_topics]
        _, t0_ns, _ = next(reader.messages(connections=selected))
    t0_sec = t0_ns * 1e-9

    def msg_time(msg, bag_ns):
        try:
            absolute = stamp_to_seconds(msg.header.stamp)
        except Exception:
            absolute = bag_ns * 1e-9
        return absolute - t0_sec

    data = {k:{'t':[],'x':[]} for k in topics}
    expected_joint_names = [
        'LF_HAA','LF_HFE','LF_KFE','RF_HAA','RF_HFE','RF_KFE',
        'LH_HAA','LH_HFE','LH_KFE','RH_HAA','RH_HFE','RH_KFE'
    ]
    joint_reorder = None
    with AnyReader([bag_path]) as reader:
        selected = [c for c in reader.connections if c.topic in set(topics.values())]
        for connection, timestamp, rawdata in reader.messages(connections=selected):
            msg = reader.deserialize(rawdata, connection.msgtype)
            topic = connection.topic
            t_msg = msg_time(msg, timestamp)
            if topic == topics['joints']:
                if joint_reorder is None:
                    names = list(msg.name)
                    missing = [n for n in expected_joint_names if n not in names]
                    if missing:
                        raise RuntimeError(f'Missing expected joints: {missing}')
                    joint_reorder = np.array([names.index(n) for n in expected_joint_names], dtype=int)
                pos = np.asarray(msg.position, np.float32)[joint_reorder]
                vel = np.asarray(msg.velocity, np.float32)[joint_reorder]
                eff = np.asarray(msg.effort, np.float32)[joint_reorder]
                x = np.concatenate([pos,vel,eff])
                key='joints'
            elif topic in {topics['lf'],topics['rf'],topics['lh'],topics['rh']}:
                x = np.array([msg.wrench.force.x,msg.wrench.force.y,msg.wrench.force.z], np.float32)
                key = 'lf' if topic==topics['lf'] else 'rf' if topic==topics['rf'] else 'lh' if topic==topics['lh'] else 'rh'
            elif topic == topics['twist']:
                tw=msg.twist.twist
                x=np.array([tw.linear.x,tw.linear.y,tw.linear.z,tw.angular.x,tw.angular.y,tw.angular.z],np.float32)
                key='twist'
            elif topic == topics['command']:
                tw=msg.twist
                x=np.array([tw.linear.x,tw.linear.y,tw.linear.z,tw.angular.x,tw.angular.y,tw.angular.z],np.float32)
                key='command'
            else:
                continue
            data[key]['t'].append(t_msg); data[key]['x'].append(x)

    for k in data:
        data[k]['t']=np.asarray(data[k]['t'],np.float64)
        data[k]['x']=np.asarray(data[k]['x'],np.float32)
        o=np.argsort(data[k]['t']); data[k]['t']=data[k]['t'][o]; data[k]['x']=data[k]['x'][o]

    stream_keys=['joints','lf','rf','lh','rh','twist','command']
    common_start=max(imu_time[0], *[data[k]['t'][0] for k in stream_keys])
    common_end=min(imu_time[-1], *[data[k]['t'][-1] for k in stream_keys])
    grid_mask=(imu_time>=common_start)&(imu_time<=common_end)
    grid_original=imu_time[grid_mask]
    acc_sync=acc[grid_mask]; gyro_sync=gyro[grid_mask]
    joint_sync=causal_hold(data['joints']['t'],data['joints']['x'],grid_original)
    lf=causal_hold(data['lf']['t'],data['lf']['x'],grid_original)
    rf=causal_hold(data['rf']['t'],data['rf']['x'],grid_original)
    lh=causal_hold(data['lh']['t'],data['lh']['x'],grid_original)
    rh=causal_hold(data['rh']['t'],data['rh']['x'],grid_original)
    body_twist=causal_hold(data['twist']['t'],data['twist']['x'],grid_original)
    U=causal_hold(data['command']['t'],data['command']['x'],grid_original).astype(np.float32)

    q=joint_sync[:,0:12]; dq=joint_sync[:,12:24]; tau=joint_sync[:,24:36]
    legs=[[0,1,2],[3,4,5],[6,7,8],[9,10,11]]
    force_list=[lf,rf,lh,rh]
    P=np.stack([np.concatenate([q[:,idx],dq[:,idx],tau[:,idx],force_list[j]],axis=1) for j,idx in enumerate(legs)],axis=1).astype(np.float32)
    B=np.concatenate([acc_sync,gyro_sync,body_twist],axis=1).astype(np.float32)

    # Low-level actuator commands.
    ACT='/state_estimator/actuator_readings'
    cmd_t=[]; cmd_tau=[]; cmd_mode=[]; st_t=[]; st_pos=[]
    with AnyReader([bag_path]) as reader:
        conns=[c for c in reader.connections if c.topic==ACT]
        if not conns: raise RuntimeError(f'{ACT} missing in run {run}')
        for c,bag_ns,raw in reader.messages(connections=conns):
            msg=reader.deserialize(raw,c.msgtype); readings=list(msg.readings)
            if len(readings)!=12: continue
            cs=[]; ct=[]; cm=[]; ss=[]; sp=[]
            for r in readings:
                cs.append(stamp_to_seconds(r.commanded.header.stamp)); ct.append(float(r.commanded.joint_torque)); cm.append(int(r.commanded.mode))
                ss.append(stamp_to_seconds(r.state.header.stamp)); sp.append(float(r.state.joint_position))
            cmd_t.append(float(np.median(cs))-t0_sec); cmd_tau.append(ct); cmd_mode.append(cm)
            st_t.append(float(np.median(ss))-t0_sec); st_pos.append(sp)
    cmd_t=np.asarray(cmd_t,np.float64); cmd_tau=np.asarray(cmd_tau,np.float32); cmd_mode=np.asarray(cmd_mode,np.int32)
    st_t=np.asarray(st_t,np.float64); st_pos=np.asarray(st_pos,np.float32)
    o=np.argsort(cmd_t); cmd_t=cmd_t[o]; cmd_tau=cmd_tau[o]; cmd_mode=cmd_mode[o]
    o=np.argsort(st_t); st_t=st_t[o]; st_pos=st_pos[o]

    # Crop again if actuator stream begins later. This avoids backward extrapolation.
    valid_start=max(grid_original[0],cmd_t[0],st_t[0])
    keep=grid_original>=valid_start
    grid_original=grid_original[keep]; P=P[keep]; B=B[keep]; U=U[keep]

    act_pos_sync=causal_hold(st_t,st_pos,grid_original)
    qcan=P[:,:,0:3].reshape(len(P),12)
    m=min(5000,len(P)); cost=np.zeros((12,12),np.float64)
    for ai in range(12):
        for ji in range(12):
            d=act_pos_sync[:m,ai]-qcan[:m,ji]; scale=np.std(qcan[:m,ji])+1e-6
            cost[ai,ji]=np.sqrt(np.mean(d*d))/scale
    rows,cols=linear_sum_assignment(cost)
    actuator_for_joint=np.empty(12,dtype=int)
    for ai,ji in zip(rows,cols): actuator_for_joint[ji]=ai

    cmd_tau_sync=causal_hold(cmd_t,cmd_tau,grid_original)
    cmd_mode_sync=causal_hold(cmd_t,cmd_mode,grid_original)
    L=cmd_tau_sync[:,actuator_for_joint].astype(np.float32)
    L_MODE=cmd_mode_sync[:,actuator_for_joint]
    unsupported=~np.isin(L_MODE,[10,12,13])
    if float(unsupported.mean())>0.01:
        raise RuntimeError('>1% actuator samples are outside torque-containing modes.')
    L_LEG=L.reshape(len(L),4,3)
    measured_tau=P[:,:,6:9]
    E=np.zeros_like(measured_tau,np.float32)
    E[0]=measured_tau[0]-L_LEG[0]
    E[1:]=measured_tau[1:]-L_LEG[:-1]

    crop_start_original=float(grid_original[0])
    time=(grid_original-crop_start_original).astype(np.float64)
    contact=(np.linalg.norm(P[:,:,9:12],axis=-1)>CONTACT_THRESHOLD).astype(np.float32)

    out=dict(time=time,grid_original=grid_original.astype(np.float64),crop_start_original=np.array(crop_start_original),
             P=P,B=B,U=U,L_LEG=L_LEG,E=E,contact=contact,actuator_for_joint=actuator_for_joint)
    np.savez_compressed(cache,**out)
    return out

# ---------------------- FROZEN VISUAL FEATURES -------------------------
_VIS_MODELS=None

def get_visual_models():
    global _VIS_MODELS
    if _VIS_MODELS is not None: return _VIS_MODELS
    dino_name='facebook/dinov2-small'
    proc=AutoImageProcessor.from_pretrained(dino_name)
    dino=AutoModel.from_pretrained(dino_name).to(DEVICE).eval()
    for p in dino.parameters(): p.requires_grad=False
    weights=ConvNeXt_Tiny_Weights.DEFAULT
    conv=convnext_tiny(weights=weights).to(DEVICE); conv.classifier=nn.Identity(); conv.eval()
    for p in conv.parameters(): p.requires_grad=False
    _VIS_MODELS=(proc,dino,conv,weights.transforms())
    return _VIS_MODELS


def prepare_visual_features(base, env, run, physics):
    cache=CACHE_ROOT/f'{env}_{run}_visual_v3.npz'
    if cache.exists():
        z=np.load(cache,allow_pickle=False)
        return {k:z[k] for k in z.files}

    rgb_dir=base/'image_lcam_front'; depth_dir=base/'depth_lcam_front'; cam_file=base/'imu'/'cam_time.txt'
    rgb_ids={f.name.split('_')[0] for f in rgb_dir.glob('*_lcam_front.png')}
    dep_ids={f.name.split('_')[0] for f in depth_dir.glob('*_lcam_front_depth.png')}
    frame_ids_all=sorted(rgb_ids & dep_ids,key=lambda x:int(x))
    nums=np.asarray([int(x) for x in frame_ids_all],int)
    cam_orig=np.loadtxt(cam_file).reshape(-1).astype(np.float64)
    cam_times_all=cam_orig[nums]-float(physics['crop_start_original'])
    valid=(cam_times_all>=physics['time'][0]-1e-8)&(cam_times_all<=physics['time'][-1]+1e-8)
    frame_ids=np.asarray(frame_ids_all)[valid].tolist(); cam_times=cam_times_all[valid]

    proc,dino,conv,conv_transform=get_visual_models()
    all_dino=[]; all_conv=[]; all_depth=[]; rough=[]
    for bs in range(0,len(frame_ids),VIS_BATCH):
        batch_ids=frame_ids[bs:bs+VIS_BATCH]; pil_batch=[]; conv_batch=[]; depth_batch=[]; rough_batch=[]
        for fid in batch_ids:
            rgb=np.asarray(Image.open(rgb_dir/f'{fid}_lcam_front.png').convert('RGB'))
            depth=load_metric_depth(depth_dir/f'{fid}_lcam_front_depth.png')
            H,W=rgb.shape[:2]; x0=int(ROI_X_MIN*W); x1=int(ROI_X_MAX*W); y0=int(ROI_Y_MIN*H); y1=int(ROI_Y_MAX*H)
            rr=rgb[y0:y1,x0:x1].copy(); dd=depth[y0:y1,x0:x1].copy()
            valid_d=np.isfinite(dd)&(dd>=DEPTH_MIN_METERS)&(dd<=DEPTH_MAX_METERS)
            rr[~valid_d]=0; dd[~valid_d]=0.0
            pil=Image.fromarray(rr); pil_batch.append(pil); conv_batch.append(conv_transform(pil))
            small=cv2.resize(dd,(DEPTH_SIZE,DEPTH_SIZE),interpolation=cv2.INTER_AREA)
            small=np.clip(small,0,DEPTH_MAX_METERS)/DEPTH_MAX_METERS
            depth_batch.append(small[None,:,:])
            v=small[small>0]
            if len(v)<20: rough_batch.append(0.0)
            else:
                gy,gx=np.gradient(small); rough_batch.append(float(np.std(v)+np.mean(np.sqrt(gx*gx+gy*gy))))
        din_in=proc(images=pil_batch,return_tensors='pt').to(DEVICE)
        with torch.no_grad(): dfeat=dino(**din_in).last_hidden_state[:,0,:]
        ct=torch.stack(conv_batch).to(DEVICE)
        with torch.no_grad(): cfeat=conv(ct).flatten(1)
        all_dino.append(dfeat.cpu().numpy().astype(np.float32)); all_conv.append(cfeat.cpu().numpy().astype(np.float32))
        all_depth.append(np.asarray(depth_batch,np.float32)); rough.extend(rough_batch)
    D=np.concatenate(all_dino); C=np.concatenate(all_conv); G=np.concatenate(all_depth); rough=np.asarray(rough,np.float32)
    vi=np.searchsorted(cam_times,physics['time'],side='right')-1
    out=dict(DINO=D,CONV=C,DEPTH=G,camera_times=cam_times.astype(np.float64),visual_index=vi.astype(np.int64),
             frame_ids=np.asarray(frame_ids),roughness=rough)
    np.savez_compressed(cache,**out)
    return out

# ------------------------- NORMALIZATION -------------------------------
def make_stats(data, ttest):
    t=data['time']; mask=(t>=ttest-TRAIN_LOOKBACK_SEC)&(t<=ttest)
    if mask.sum()<100: raise RuntimeError('Insufficient pre-test data for normalization.')
    P=data['P'][mask]; B=data['B'][mask]; U=data['U'][mask]
    return {
        'P_mean':P.mean(axis=(0,1)).astype(np.float32), 'P_std':(P.std(axis=(0,1))+EPS).astype(np.float32),
        'B_mean':B.mean(axis=0).astype(np.float32), 'B_std':(B.std(axis=0)+EPS).astype(np.float32),
        'U_mean':U.mean(axis=0).astype(np.float32), 'U_std':(U.std(axis=0)+EPS).astype(np.float32),
    }


def normalize_arrays(data, stats):
    P=(data['P']-stats['P_mean'])/stats['P_std']
    B=(data['B']-stats['B_mean'])/stats['B_std']
    U=(data['U']-stats['U_mean'])/stats['U_std']
    L=(data['L_LEG']-stats['P_mean'][6:9])/stats['P_std'][6:9]
    E=data['E']/stats['P_std'][6:9]
    return P.astype(np.float32),B.astype(np.float32),U.astype(np.float32),L.astype(np.float32),E.astype(np.float32)

# --------------------------- MODEL -------------------------------------
class CausalHistoryEncoderFlex(nn.Module):
    def __init__(self,input_dim,history_steps,hidden=128,layers=2,heads=4):
        super().__init__(); self.history_steps=history_steps
        self.input_proj=nn.Linear(input_dim,hidden)
        self.pos=nn.Parameter(torch.randn(1,history_steps,hidden)*0.02)
        layer=nn.TransformerEncoderLayer(d_model=hidden,nhead=heads,dim_feedforward=hidden*4,dropout=0.1,
                                         batch_first=True,norm_first=True,activation='gelu')
        self.tx=nn.TransformerEncoder(layer,num_layers=layers); self.norm=nn.LayerNorm(hidden)
    def forward(self,x):
        T=x.shape[1]; x=self.input_proj(x)+self.pos[:,:T]
        mask=torch.triu(torch.ones(T,T,device=x.device,dtype=torch.bool),diagonal=1)
        x=self.tx(x,mask=mask); return self.norm(x)[:,-1]

class TerrainEncoderFlex(nn.Module):
    def __init__(self,dino_dim,conv_dim):
        super().__init__()
        self.dino_proj=nn.Sequential(nn.Linear(dino_dim,256),nn.GELU(),nn.LayerNorm(256))
        self.conv_proj=nn.Sequential(nn.Linear(conv_dim,256),nn.GELU(),nn.LayerNorm(256))
        self.depth_net=nn.Sequential(nn.Conv2d(1,32,5,2,2),nn.GELU(),nn.Conv2d(32,64,3,2,1),nn.GELU(),
                                     nn.Conv2d(64,128,3,2,1),nn.GELU(),nn.AdaptiveAvgPool2d(1))
        self.depth_proj=nn.Sequential(nn.Linear(128,256),nn.GELU(),nn.LayerNorm(256))
        layer=nn.TransformerEncoderLayer(d_model=256,nhead=4,dim_feedforward=512,dropout=0.1,batch_first=True,norm_first=True,activation='gelu')
        self.fusion=nn.TransformerEncoder(layer,num_layers=2)
        self.type_embedding=nn.Parameter(torch.randn(1,3,256)*0.02); self.norm=nn.LayerNorm(256)
    def forward(self,dino,conv,depth):
        toks=torch.stack([self.dino_proj(dino),self.conv_proj(conv.flatten(1)),self.depth_proj(self.depth_net(depth).flatten(1))],dim=1)
        toks=self.fusion(toks+self.type_embedding); return self.norm(toks.mean(1))

class ControlledTerrainDynamicsFlex(nn.Module):
    # Same V3 one-step controlled dynamics, but history length and rollout length are parameters.
    def __init__(self,history_steps,stats,dino_dim=384,conv_dim=768):
        super().__init__(); self.history_steps=history_steps
        self.leg_encoder=CausalHistoryEncoderFlex(24,history_steps,128,2,4)
        self.leg_identity=nn.Embedding(4,128)
        self.body_encoder=CausalHistoryEncoderFlex(18,history_steps,128,2,4)
        self.terrain_encoder=TerrainEncoderFlex(dino_dim,conv_dim)
        self.context_net=nn.Sequential(nn.LayerNorm(896),nn.Linear(896,512),nn.GELU(),nn.Dropout(0.1),nn.Linear(512,256),nn.GELU(),nn.LayerNorm(256))
        self.response_head=nn.Sequential(nn.Linear(256,256),nn.GELU(),nn.Linear(256,60))
        self.contact_head=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Linear(128,4))
        zf=(np.zeros(3,np.float32)-stats['P_mean'][9:12])/stats['P_std'][9:12]
        self.register_buffer('zero_force_norm',torch.tensor(zf,dtype=torch.float32))
    def encode_context(self,p_hist,b_hist,u_hist,l_hist,e_hist,terrain):
        Bn=p_hist.shape[0]; T=p_hist.shape[2]
        ulegs=u_hist.unsqueeze(1).expand(-1,4,-1,-1)
        li=torch.cat([p_hist,ulegs,l_hist,e_hist],dim=-1).reshape(Bn*4,T,24)
        lf=self.leg_encoder(li)
        ids=torch.arange(4,device=p_hist.device).repeat(Bn)
        lf=(lf+self.leg_identity(ids)).reshape(Bn,4,128).reshape(Bn,512)
        bf=self.body_encoder(torch.cat([b_hist,u_hist],dim=-1))
        return self.context_net(torch.cat([lf,bf,terrain],dim=-1))
    def one_step(self,p_hist,b_hist,u_hist,l_hist,e_hist,terrain):
        Bn=p_hist.shape[0]; ctx=self.encode_context(p_hist,b_hist,u_hist,l_hist,e_hist,terrain); delta=self.response_head(ctx)
        cp=p_hist[:,:,-1,:].clone(); cb=b_hist[:,-1,:].clone(); basep=cp.clone(); current_l=l_hist[:,:,-1,:]
        basep[:,:,6:9]=current_l
        base=torch.cat([basep.reshape(Bn,48),cb],dim=-1); state=base+delta
        pp=state[:,:48].reshape(Bn,4,12); pb=state[:,48:]; clog=self.contact_head(ctx); prob=torch.sigmoid(clog)
        rawf=pp[:,:,9:12]; zf=self.zero_force_norm.view(1,1,3)
        pp=pp.clone(); pp[:,:,9:12]=prob.unsqueeze(-1)*rawf+(1-prob.unsqueeze(-1))*zf
        return pp,pb,clog
    def rollout(self,p_hist,b_hist,u_hist,l_hist,e_hist,dino,conv,depth,future_u,future_l,steps):
        terrain=self.terrain_encoder(dino,conv,depth)
        rp,rb,ru,rl,re=p_hist,b_hist,u_hist,l_hist,e_hist
        ps=[]; bs=[]; cs=[]
        for k in range(steps):
            pp,pb,cl=self.one_step(rp,rb,ru,rl,re,terrain); ps.append(pp); bs.append(pb); cs.append(cl)
            if k==steps-1: break
            command_used=rl[:,:,-1,:]; enext=pp[:,:,6:9]-command_used
            rp=torch.cat([rp[:,:,1:,:],pp.unsqueeze(2)],dim=2)
            rb=torch.cat([rb[:,1:,:],pb.unsqueeze(1)],dim=1)
            nu=future_u[:,k+1,:]; nl=future_l[:,k+1,:,:]
            ru=torch.cat([ru[:,1:,:],nu.unsqueeze(1)],dim=1)
            rl=torch.cat([rl[:,:,1:,:],nl.unsqueeze(2)],dim=2)
            re=torch.cat([re[:,:,1:,:],enext.unsqueeze(2)],dim=2)
        return {'p':torch.stack(ps,1),'b':torch.stack(bs,1),'contact_logits':torch.stack(cs,1),'terrain':terrain}

# ------------------------- SAMPLE BUILDING -----------------------------
def tensor(x): return torch.as_tensor(x,dtype=torch.float32,device=DEVICE)

def make_sample(data,visual,stats,idx,history_steps,horizon_steps):
    Pn,Bn,Un,Ln,En=normalize_arrays(data,stats)
    h0=idx-history_steps+1
    if h0<0 or idx+horizon_steps>=len(data['time']): raise RuntimeError('Sample out of range')
    vi=int(visual['visual_index'][idx])
    if vi<0: raise RuntimeError('No causal image for sample')
    sample={
        'p_hist':tensor(Pn[h0:idx+1].transpose(1,0,2)[None,:,:,:]),
        'b_hist':tensor(Bn[h0:idx+1][None,:,:]),
        'u_hist':tensor(Un[h0:idx+1][None,:,:]),
        'l_hist':tensor(Ln[h0:idx+1].transpose(1,0,2)[None,:,:,:]),
        'e_hist':tensor(En[h0:idx+1].transpose(1,0,2)[None,:,:,:]),
        'dino':tensor(visual['DINO'][vi][None,:]),
        'conv':tensor(visual['CONV'][vi][None,:]),
        'depth':tensor(visual['DEPTH'][vi][None,:,:,:]),
        'future_u':tensor(Un[idx:idx+horizon_steps][None,:,:]),
        'future_l':tensor(Ln[idx:idx+horizon_steps][None,:,:,:]),
        'target_p':tensor(Pn[idx+1:idx+horizon_steps+1][None,:,:,:]),
        'target_b':tensor(Bn[idx+1:idx+horizon_steps+1][None,:,:]),
        'target_contact':tensor(data['contact'][idx+1:idx+horizon_steps+1][None,:,:]),
        'vi':vi,
    }
    return sample


def make_training_windows(data,visual,ttest,context_sec,hop_sec,horizon_sec,half_life):
    dt=float(np.median(np.diff(data['time']))); H=int(round(context_sec/dt)); K=int(round(horizon_sec/dt))
    # Build backward from t_test so the MOST RECENT training target ends exactly at t_test.
    # This preserves the immediate pre-test context instead of reserving it for validation.
    target_ends=[]; te=float(ttest)
    while (te-horizon_sec-context_sec) >= (ttest-TRAIN_LOOKBACK_SEC-1e-9):
        target_ends.append(te); te-=hop_sec
    rows=[]; seen=set()
    for desired_target_end in reversed(target_ends):
        current_end=desired_target_end-horizon_sec
        idx=nearest_index(data['time'],current_end)
        if idx in seen: continue
        seen.add(idx)
        if idx-H+1<0 or idx+K>=len(data['time']): continue
        if int(visual['visual_index'][idx])<0: continue
        actual_target_end=float(data['time'][idx+K])
        if actual_target_end>ttest+0.015: continue
        # Losses whose targets are closer to t_test receive more optimization weight.
        age=max(0.0,ttest-actual_target_end)
        w=0.5**(age/half_life)
        rows.append((idx,w,float(data['time'][idx]),actual_target_end))
    if not rows: raise RuntimeError('No training windows produced.')
    return rows,H,K

# ---------------------------- LOSS -------------------------------------
BCE=nn.BCEWithLogitsLoss(reduction='none')

def rollout_loss(out,sample):
    pp,pb=out['p'],out['b']; tp,tb=sample['target_p'],sample['target_b']; H=pp.shape[1]
    hw=torch.linspace(1.0,2.0,H,device=pp.device)
    def hmse(a,b):
        e=(a-b)**2; dims=tuple(range(2,e.ndim)); return (e.mean(dim=dims)*hw[None,:]).mean()
    lq=hmse(pp[:,:,:,0:3],tp[:,:,:,0:3]); ldq=hmse(pp[:,:,:,3:6],tp[:,:,:,3:6]); lt=hmse(pp[:,:,:,6:9],tp[:,:,:,6:9]); lf=hmse(pp[:,:,:,9:12],tp[:,:,:,9:12]); lb=hmse(pb,tb)
    ce=BCE(out['contact_logits'],sample['target_contact']).mean(-1); lc=(ce*hw[None,:]).mean()
    total=lq+ldq+lt+1.25*lf+0.5*lb+0.25*lc
    return total

# ---------------------- RECENCY-WEIGHTED TRAIN -------------------------
def train_fresh_expert(data,visual,ttest,context_sec,hop_sec,horizon_sec,half_life,epochs,seed):
    seed_everything(seed)
    stats=make_stats(data,ttest)
    windows,H,K=make_training_windows(data,visual,ttest,context_sec,hop_sec,horizon_sec,half_life)
    model=ControlledTerrainDynamicsFlex(H,stats,dino_dim=visual['DINO'].shape[1],conv_dim=visual['CONV'].reshape(len(visual['CONV']),-1).shape[1]).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    weights=np.asarray([w for _,w,_,_ in windows],np.float64); weights=weights/weights.sum()
    print(f'  train: context={context_sec}s hop={hop_sec}s horizon={horizon_sec}s windows={len(windows)} H={H} K={K}')
    for ep in range(epochs):
        model.train(); opt.zero_grad(set_to_none=True); running=0.0
        for (idx,_,_,_),wn in zip(windows,weights):
            smp=make_sample(data,visual,stats,idx,H,K)
            out=model.rollout(smp['p_hist'],smp['b_hist'],smp['u_hist'],smp['l_hist'],smp['e_hist'],smp['dino'],smp['conv'],smp['depth'],smp['future_u'],smp['future_l'],K)
            loss=rollout_loss(out,smp); (loss*float(wn)).backward(); running+=float(loss.detach().cpu())*float(wn)
            del out,smp,loss
        torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP); opt.step()
        print(f'    epoch {ep+1:02d}/{epochs}: weighted loss={running:.6f}')
    return model,stats,H

# -------------------------- TEST/FUSION --------------------------------
def gaussian_mix_weights(h):
    gs=math.exp(-0.5*((h-SHORT_NATIVE_HORIZON)/SIGMA_SHORT)**2)
    gl=math.exp(-0.5*((h-LONG_NATIVE_HORIZON)/SIGMA_LONG)**2)
    z=gs+gl+1e-12
    return gs/z,gl/z


def denorm_predictions(out,stats):
    p=out['p'][0].detach().cpu().numpy()*stats['P_std'][None,None,:]+stats['P_mean'][None,None,:]
    b=out['b'][0].detach().cpu().numpy()*stats['B_std'][None,:]+stats['B_mean'][None,:]
    c=torch.sigmoid(out['contact_logits'][0]).detach().cpu().numpy()
    return p,b,c


def evaluate_test_case(short_model,short_stats,Hs,long_model,long_stats,Hl,data,visual,ttest):
    dt=float(np.median(np.diff(data['time']))); Ktest=int(round(TEST_HORIZON_SEC/dt)); idx=nearest_index(data['time'],ttest)
    # Same raw test history, separately normalized for each expert (stats are both from same 20 s region but kept explicit).
    ss=make_sample(data,visual,short_stats,idx,Hs,Ktest)
    ls=make_sample(data,visual,long_stats,idx,Hl,Ktest)
    short_model.eval(); long_model.eval()
    with torch.no_grad():
        so=short_model.rollout(ss['p_hist'],ss['b_hist'],ss['u_hist'],ss['l_hist'],ss['e_hist'],ss['dino'],ss['conv'],ss['depth'],ss['future_u'],ss['future_l'],Ktest)
        lo=long_model.rollout(ls['p_hist'],ls['b_hist'],ls['u_hist'],ls['l_hist'],ls['e_hist'],ls['dino'],ls['conv'],ls['depth'],ls['future_u'],ls['future_l'],Ktest)
    sp,sb,sc=denorm_predictions(so,short_stats); lp,lb,lc=denorm_predictions(lo,long_stats)
    rows=[]
    leg_names=['FL','FR','RL','RR']; feat=['q_HAA','q_HFE','q_KFE','dq_HAA','dq_HFE','dq_KFE','tau_HAA','tau_HFE','tau_KFE','Fx','Fy','Fz']
    body=['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z','body_vx','body_vy','body_vz','body_wx','body_wy','body_wz']
    step_offsets=[int(round(h/dt)) for h in np.arange(OUTPUT_INTERVAL_SEC,TEST_HORIZON_SEC+1e-9,OUTPUT_INTERVAL_SEC)]
    for h,off in zip(np.arange(0.1,1.0001,0.1),step_offsets):
        k=off-1; gt_p=data['P'][idx+off]; gt_b=data['B'][idx+off]; gt_c=data['contact'][idx+off]
        ws,wl=gaussian_mix_weights(float(h)); fp=ws*sp[k]+wl*lp[k]; fb=ws*sb[k]+wl*lb[k]; fc=ws*sc[k]+wl*lc[k]
        for li,LN in enumerate(leg_names):
            for fi,FN in enumerate(feat):
                truth=float(gt_p[li,fi]); pred=float(fp[li,fi])
                rows.append(dict(horizon_s=round(float(h),1),signal=f'{LN}.{FN}',short_pred=float(sp[k,li,fi]),long_pred=float(lp[k,li,fi]),lambda_short=ws,lambda_long=wl,prediction=pred,ground_truth=truth,short_abs_error=abs(float(sp[k,li,fi])-truth),long_abs_error=abs(float(lp[k,li,fi])-truth),signed_error=pred-truth,abs_error=abs(pred-truth)))
            truth=float(gt_c[li]); pred=float(fc[li])
            rows.append(dict(horizon_s=round(float(h),1),signal=f'{LN}.contact_probability',short_pred=float(sc[k,li]),long_pred=float(lc[k,li]),lambda_short=ws,lambda_long=wl,prediction=pred,ground_truth=truth,short_abs_error=abs(float(sc[k,li])-truth),long_abs_error=abs(float(lc[k,li])-truth),signed_error=pred-truth,abs_error=abs(pred-truth)))
        for bi,BN in enumerate(body):
            truth=float(gt_b[bi]); pred=float(fb[bi])
            rows.append(dict(horizon_s=round(float(h),1),signal=BN,short_pred=float(sb[k,bi]),long_pred=float(lb[k,bi]),lambda_short=ws,lambda_long=wl,prediction=pred,ground_truth=truth,short_abs_error=abs(float(sb[k,bi])-truth),long_abs_error=abs(float(lb[k,bi])-truth),signed_error=pred-truth,abs_error=abs(pred-truth)))
    vi=int(visual['visual_index'][idx])
    return pd.DataFrame(rows),idx,vi

# ---------------------- TEST-TIME SELECTION ----------------------------
def choose_four_ttests(data,visual):
    t=data['time']; lo=max(25.0,float(t[0])+25.0); hi=float(t[-1])-1.2
    candidates=np.arange(lo,hi,1.0); scored=[]
    for tt in candidates:
        idx=nearest_index(t,tt); vi=int(visual['visual_index'][idx])
        if vi<0: continue
        scored.append((float(visual['roughness'][vi]),float(t[idx])))
    scored.sort(reverse=True)
    chosen=[]
    for score,tt in scored:
        if all(abs(tt-x)>=10.0 for x in chosen):
            chosen.append(tt)
        if len(chosen)==4: break
    if len(chosen)<4:
        chosen=list(np.linspace(lo,hi,4))
    return sorted(chosen)

# -------------------------- IMAGE OUTPUT -------------------------------
def get_test_images(base,visual,vi):
    fid=str(visual['frame_ids'][vi])
    rgb=np.asarray(Image.open(base/'image_lcam_front'/f'{fid}_lcam_front.png').convert('RGB'))
    depth=load_metric_depth(base/'depth_lcam_front'/f'{fid}_lcam_front_depth.png')
    H,W=rgb.shape[:2]; x0=int(ROI_X_MIN*W); x1=int(ROI_X_MAX*W); y0=int(ROI_Y_MIN*H); y1=int(ROI_Y_MAX*H)
    roi=rgb[y0:y1,x0:x1].copy(); dd=depth[y0:y1,x0:x1]
    valid=np.isfinite(dd)&(dd>=DEPTH_MIN_METERS)&(dd<=DEPTH_MAX_METERS); roi[~valid]=0
    return fid,rgb,roi

# --------------------------- MAIN DRIVER -------------------------------
def run_full_multiterrain_experiment():
    terrain_map=build_terrain_run_map(); all_results=[]; image_records=[]; summary=[]
    for terrain_i,row in terrain_map.iterrows():
        terrain_type=row.terrain_type; env=row.environment; run=row.chosen_run
        print('\n'+'='*100); print(f'{terrain_type.upper()} :: {env}/{run}'); print('='*100)
        base=download_run(env,run)
        data=extract_run_physics(base,run)
        visual=prepare_visual_features(base,env,run,data)
        ttests=choose_four_ttests(data,visual)
        print('Selected test times (high depth-roughness, >=10 s apart):',ttests)

        for test_no,ttest in enumerate(ttests,1):
            print('\n'+'-'*90); print(f'{terrain_type} | {env}/{run} | test {test_no}/4 | ttest={ttest:.3f}s')
            # Fresh random initialization PER TEST CASE, for BOTH experts.
            base_seed=SEED + terrain_i*1000 + test_no*100
            short_model,short_stats,Hs=train_fresh_expert(data,visual,ttest,SHORT_CONTEXT_SEC,SHORT_HOP_SEC,SHORT_TRAIN_HORIZON_SEC,SHORT_RECENCY_HALF_LIFE_SEC,EPOCHS_SHORT,base_seed+1)
            long_model,long_stats,Hl=train_fresh_expert(data,visual,ttest,LONG_CONTEXT_SEC,LONG_HOP_SEC,LONG_TRAIN_HORIZON_SEC,LONG_RECENCY_HALF_LIFE_SEC,EPOCHS_LONG,base_seed+2)
            result,idx,vi=evaluate_test_case(short_model,short_stats,Hs,long_model,long_stats,Hl,data,visual,ttest)
            result.insert(0,'test_id',test_no); result.insert(0,'run',run); result.insert(0,'environment',env); result.insert(0,'terrain_type',terrain_type); result.insert(4,'ttest_s',float(data['time'][idx]))
            all_results.append(result)
            perfile=OUT_ROOT/f'{terrain_type}_{env}_{run}_test{test_no}_ttest_{data["time"][idx]:.2f}.csv'; result.to_csv(perfile,index=False)
            physical=result[~result.signal.str.contains('contact_probability')]
            summary.append(dict(terrain_type=terrain_type,environment=env,run=run,test_id=test_no,ttest_s=float(data['time'][idx]),overall_RMSE=float(np.sqrt(np.mean(physical.signed_error.values**2))),overall_MAE=float(np.mean(physical.abs_error.values))))
            fid,rgb,roi=get_test_images(base,visual,vi)
            img_path=OUT_ROOT/f'{terrain_type}_{env}_{run}_test{test_no}_t{data["time"][idx]:.2f}_RGB.png'; Image.fromarray(rgb).save(img_path)
            roi_path=OUT_ROOT/f'{terrain_type}_{env}_{run}_test{test_no}_t{data["time"][idx]:.2f}_MODEL_ROI.png'; Image.fromarray(roi).save(roi_path)
            image_records.append(dict(terrain_type=terrain_type,environment=env,run=run,test_id=test_no,ttest=float(data['time'][idx]),frame_id=fid,rgb=rgb,roi=roi,rgb_path=str(img_path),roi_path=str(roi_path)))
            print(f'Saved table: {perfile}')
            display(result)
            del short_model,long_model,result
            gc.collect();
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        # release current run arrays before next several-GB terrain
        del data,visual
        gc.collect();
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    results=pd.concat(all_results,ignore_index=True) if all_results else pd.DataFrame()
    summary_df=pd.DataFrame(summary)
    results.to_csv(OUT_ROOT/'ALL_16_TESTS_ALL_SENSOR_VALUES.csv',index=False)
    summary_df.to_csv(OUT_ROOT/'ALL_16_TESTS_SUMMARY.csv',index=False)
    terrain_map.to_csv(OUT_ROOT/'TERRAIN_RUN_MAP.csv',index=False)

    # Exactly 16 t_test RGB images in one grid.
    fig,axes=plt.subplots(4,4,figsize=(20,18)); axes=axes.reshape(-1)
    for ax,rec in zip(axes,image_records):
        ax.imshow(rec['rgb']); ax.axis('off')
        ax.set_title(f"{rec['terrain_type']}\n{rec['environment']}/{rec['run']} | test {rec['test_id']} | t={rec['ttest']:.2f}s",fontsize=9)
    plt.tight_layout(); fig.savefig(OUT_ROOT/'ALL_16_TTEST_RGB_IMAGES.png',dpi=180,bbox_inches='tight'); plt.show()

    # Also show the exact masked near-ground ROI actually used by V3 vision.
    fig,axes=plt.subplots(4,4,figsize=(20,18)); axes=axes.reshape(-1)
    for ax,rec in zip(axes,image_records):
        ax.imshow(rec['roi']); ax.axis('off')
        ax.set_title(f"MODEL ROI: {rec['terrain_type']}\nt={rec['ttest']:.2f}s",fontsize=9)
    plt.tight_layout(); fig.savefig(OUT_ROOT/'ALL_16_TTEST_MODEL_ROIS.png',dpi=180,bbox_inches='tight'); plt.show()

    print('\nFINAL SUMMARY')
    display(summary_df)
    print('\nFull sensor table:',OUT_ROOT/'ALL_16_TESTS_ALL_SENSOR_VALUES.csv')
    print('16-image grid     :',OUT_ROOT/'ALL_16_TTEST_RGB_IMAGES.png')
    print('16 model ROIs     :',OUT_ROOT/'ALL_16_TTEST_MODEL_ROIS.png')
    return terrain_map,results,summary_df,image_records


# %% notebook cell 14
# RUN THE FULL 4-terrain x 4-test experiment.
# Two fresh experts per test case = 32 fresh models total.
terrain_run_map_v32, results_v32, summary_v32, test_images_v32 = run_full_multiterrain_experiment()
# IPYTHON: !cd /content && zip -qr v3_multiterrain_results.zip v3_multiterrain_results
print("ZIP: /content/v3_multiterrain_results.zip")
