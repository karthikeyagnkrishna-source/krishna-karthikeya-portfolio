# Source inspection export of the adjacent notebook.
# Notebook execution order and dependencies still apply.
# IPython magic, shell lines and prose-only conclusion cells are comments.

# %% notebook cell 0
# IPYTHON: !pip install -q git+https://github.com/castacks/tartanairpy.git
# IPYTHON: !pip install -q rosbags

# %% notebook cell 1
import tartanair as ta

root = "/content/tartanground"
ta.init(root)

# %% notebook cell 2
ta.download_ground(
    env=["OldTownFall"],
    version=["anymal"],
    traj=["P2000"],
    modality=[
        "image",
        "depth",
        "imu",
        "meta",
        "rosbag",
    ],
    camera_name=["lcam_front"],
    unzip=True
)

# %% notebook cell 3
import os

for root_dir, dirs, files in os.walk("/content/tartanground"):
    level = root_dir.replace("/content/tartanground", "").count(os.sep)

    if level <= 4:
        print("  " * level + os.path.basename(root_dir) + "/")
        for f in files[:10]:
            print("  " * (level + 1) + f)

# %% notebook cell 4
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

base = "/content/tartanground/OldTownFall/Data_anymal/P2000"

rgb_dir = os.path.join(base, "image_lcam_front")
depth_dir = os.path.join(base, "depth_lcam_front")

# Get frame IDs available in both RGB and depth
rgb_ids = {
    f.split("_")[0]
    for f in os.listdir(rgb_dir)
    if f.endswith(".png")
}

depth_ids = {
    f.split("_")[0]
    for f in os.listdir(depth_dir)
    if f.endswith(".png")
}

common_ids = sorted(rgb_ids & depth_ids)

print("RGB frames:", len(rgb_ids))
print("Depth frames:", len(depth_ids))
print("Matched frames:", len(common_ids))

frame_id = common_ids[0]

RGB_FILE = os.path.join(
    rgb_dir,
    f"{frame_id}_lcam_front.png"
)

DEPTH_FILE = os.path.join(
    depth_dir,
    f"{frame_id}_lcam_front_depth.png"
)

print("RGB:", RGB_FILE)
print("Depth:", DEPTH_FILE)

# %% notebook cell 5
rgb = np.array(Image.open(RGB_FILE))
depth = np.array(Image.open(DEPTH_FILE))

print("RGB shape:", rgb.shape)
print("RGB dtype:", rgb.dtype)

print("Depth shape:", depth.shape)
print("Depth dtype:", depth.dtype)
print("Depth min/max:", depth.min(), depth.max())

plt.figure(figsize=(7, 7))
plt.imshow(rgb)
plt.title("RGB frame")
plt.axis("off")
plt.show()

plt.figure(figsize=(7, 7))

# percentile clipping only makes visualization easier
valid_depth = depth[depth > 0]

if len(valid_depth) > 0:
    vmax = np.percentile(valid_depth, 98)
else:
    vmax = depth.max()

plt.imshow(depth, vmin=0, vmax=vmax)
plt.title("Depth frame")
plt.colorbar()
plt.axis("off")
plt.show()

# %% notebook cell 6
from pathlib import Path
from rosbags.rosbag1 import Reader

bag_path = Path(
    "/content/tartanground/OldTownFall/Data_anymal/P2000/rosbags/P2000.bag"
)

with Reader(bag_path) as reader:
    print("Number of connections:", len(reader.connections))
    print()

    for connection in reader.connections:
        print(
            f"{connection.topic:60s}",
            connection.msgtype
        )

# %% notebook cell 7
imu_dir = os.path.join(base, "imu")

acc = np.load(os.path.join(imu_dir, "acc.npy"))
imu_time = np.load(os.path.join(imu_dir, "imu_time.npy"))

print("Acceleration:", acc.shape)
print("IMU timestamps:", imu_time.shape)

print("\nFirst 5 acceleration values:")
print(acc[:5])

print("\nFirst 5 timestamps:")
print(imu_time[:5])

# %% notebook cell 8
gyro = np.loadtxt(os.path.join(imu_dir, "gyro.txt"))
vel_body = np.loadtxt(os.path.join(imu_dir, "vel_body.txt"))
pos_global = np.loadtxt(os.path.join(imu_dir, "pos_global.txt"))

print("Gyro:", gyro.shape)
print("Body velocity:", vel_body.shape)
print("Global position:", pos_global.shape)

# %% notebook cell 9
from pathlib import Path
from rosbags.rosbag1 import Reader

bag_path = Path(
    "/content/tartanground/OldTownFall/Data_anymal/P2000/rosbags/P2000.bag"
)

with Reader(bag_path) as reader:

    print("Number of connections:", len(reader.connections))
    print()

    for c in reader.connections:
        print(
            f"{c.topic:65s} "
            f"{c.msgtype:50s} "
            f"messages={c.msgcount}"
        )

# %% notebook cell 10
keywords = [
    "joint",
    "state",
    "motor",
    "actuator",
    "contact",
    "foot",
    "force",
    "torque",
    "command",
    "anymal"
]

with Reader(bag_path) as reader:

    for c in reader.connections:

        name = c.topic.lower()

        if any(k in name for k in keywords):
            print(
                f"{c.topic:65s} "
                f"{c.msgtype:50s} "
                f"messages={c.msgcount}"
            )

# %% notebook cell 11
from pathlib import Path
from rosbags.highlevel import AnyReader

bag_path = Path(
    "/content/tartanground/OldTownFall/Data_anymal/P2000/rosbags/P2000.bag"
)

topics_to_check = [
    "/state_estimator/joint_states",
    "/state_estimator/contact_force_lf_foot",
    "/state_estimator/twist",
    "/path_planning_and_following/twist",
]

with AnyReader([bag_path]) as reader:

    for topic in topics_to_check:

        conns = [c for c in reader.connections if c.topic == topic]

        print("\n" + "="*80)
        print(topic)
        print("="*80)

        for connection, timestamp, rawdata in reader.messages(
            connections=conns
        ):
            msg = reader.deserialize(rawdata, connection.msgtype)

            print("timestamp:", timestamp)
            print(msg)
            break

# %% notebook cell 12
from pathlib import Path
from rosbags.highlevel import AnyReader
import numpy as np

bag_path = Path(
    "/content/tartanground/OldTownFall/Data_anymal/P2000/rosbags/P2000.bag"
)

topics = {
    "joints": "/state_estimator/joint_states",
    "lf": "/state_estimator/contact_force_lf_foot",
    "rf": "/state_estimator/contact_force_rf_foot",
    "lh": "/state_estimator/contact_force_lh_foot",
    "rh": "/state_estimator/contact_force_rh_foot",
    "twist": "/state_estimator/twist",
}

data = {
    k: {"t": [], "x": []}
    for k in topics
}

with AnyReader([bag_path]) as reader:

    selected = [
        c for c in reader.connections
        if c.topic in topics.values()
    ]

    # common time origin for the complete bag
    timestamps = []

    for connection, timestamp, rawdata in reader.messages(
        connections=selected
    ):
        timestamps.append(timestamp)

    t0 = min(timestamps)

print("Common bag t0:", t0)

# %% notebook cell 13
with AnyReader([bag_path]) as reader:

    selected = [
        c for c in reader.connections
        if c.topic in topics.values()
    ]

    for connection, timestamp, rawdata in reader.messages(
        connections=selected
    ):

        msg = reader.deserialize(
            rawdata,
            connection.msgtype
        )

        t = (timestamp - t0) * 1e-9

        topic = connection.topic

        # -----------------------
        # JOINT STATES
        # -----------------------
        if topic == topics["joints"]:

            # 12 q + 12 dq + 12 effort
            x = np.concatenate([
                np.asarray(msg.position),
                np.asarray(msg.velocity),
                np.asarray(msg.effort),
            ])

            data["joints"]["t"].append(t)
            data["joints"]["x"].append(x)

        # -----------------------
        # FOOT CONTACT FORCES
        # -----------------------
        elif topic in [
            topics["lf"],
            topics["rf"],
            topics["lh"],
            topics["rh"],
        ]:

            force = np.array([
                msg.wrench.force.x,
                msg.wrench.force.y,
                msg.wrench.force.z
            ])

            if topic == topics["lf"]:
                key = "lf"
            elif topic == topics["rf"]:
                key = "rf"
            elif topic == topics["lh"]:
                key = "lh"
            else:
                key = "rh"

            data[key]["t"].append(t)
            data[key]["x"].append(force)

        # -----------------------
        # BODY TWIST
        # -----------------------
        elif topic == topics["twist"]:

            tw = msg.twist.twist

            x = np.array([
                tw.linear.x,
                tw.linear.y,
                tw.linear.z,
                tw.angular.x,
                tw.angular.y,
                tw.angular.z
            ])

            data["twist"]["t"].append(t)
            data["twist"]["x"].append(x)


# convert lists -> numpy arrays
for key in data:

    data[key]["t"] = np.asarray(data[key]["t"])
    data[key]["x"] = np.asarray(data[key]["x"])

    print(
        key,
        "time:", data[key]["t"].shape,
        "values:", data[key]["x"].shape
    )

# %% notebook cell 14
imu_dir = "/content/tartanground/OldTownFall/Data_anymal/P2000/imu"

imu_time = np.load(
    f"{imu_dir}/imu_time.npy"
)

acc = np.load(
    f"{imu_dir}/acc.npy"
)

gyro = np.loadtxt(
    f"{imu_dir}/gyro.txt"
)

vel_body = np.loadtxt(
    f"{imu_dir}/vel_body.txt"
)

# %% notebook cell 15
def interpolate_stream(t_src, x_src, t_target):

    x_target = np.zeros(
        (len(t_target), x_src.shape[1])
    )

    for d in range(x_src.shape[1]):

        x_target[:, d] = np.interp(
            t_target,
            t_src,
            x_src[:, d]
        )

    return x_target

# %% notebook cell 16
joint_sync = interpolate_stream(
    data["joints"]["t"],
    data["joints"]["x"],
    imu_time
)

lf_force = interpolate_stream(
    data["lf"]["t"],
    data["lf"]["x"],
    imu_time
)

rf_force = interpolate_stream(
    data["rf"]["t"],
    data["rf"]["x"],
    imu_time
)

lh_force = interpolate_stream(
    data["lh"]["t"],
    data["lh"]["x"],
    imu_time
)

rh_force = interpolate_stream(
    data["rh"]["t"],
    data["rh"]["x"],
    imu_time
)

body_twist = interpolate_stream(
    data["twist"]["t"],
    data["twist"]["x"],
    imu_time
)

# %% notebook cell 17
print(joint_sync.shape)
print(lf_force.shape)
print(rf_force.shape)
print(lh_force.shape)
print(rh_force.shape)
print(body_twist.shape)
print(acc.shape)
print(gyro.shape)

# %% notebook cell 18
q = joint_sync[:, 0:12]
dq = joint_sync[:, 12:24]
tau = joint_sync[:, 24:36]

# %% notebook cell 19
leg_idx = {
    "FL": [0, 1, 2],
    "FR": [3, 4, 5],
    "RL": [6, 7, 8],
    "RR": [9, 10, 11],
}

# %% notebook cell 20
forces = {
    "FL": lf_force,
    "FR": rf_force,
    "RL": lh_force,
    "RR": rh_force,
}

leg_data = {}

for leg, idx in leg_idx.items():

    leg_data[leg] = np.concatenate([
        q[:, idx],
        dq[:, idx],
        tau[:, idx],
        forces[leg],
    ], axis=1)

    print(
        leg,
        leg_data[leg].shape
    )

# %% notebook cell 21
P = np.stack([
    leg_data["FL"],
    leg_data["FR"],
    leg_data["RL"],
    leg_data["RR"]
], axis=1)

print(P.shape)

# %% notebook cell 22
B = np.concatenate([
    acc,
    gyro,
    body_twist
], axis=1)

print(B.shape)

# %% notebook cell 23
cmd_topic = "/path_planning_and_following/twist"

cmd_t = []
cmd_x = []

with AnyReader([bag_path]) as reader:

    conns = [
        c for c in reader.connections
        if c.topic == cmd_topic
    ]

    for connection, timestamp, rawdata in reader.messages(
        connections=conns
    ):

        msg = reader.deserialize(
            rawdata,
            connection.msgtype
        )

        # use header timestamp
        stamp = msg.header.stamp
        ts = stamp.sec + stamp.nanosec * 1e-9

        # same zero as our bag streams
        ts -= t0 * 1e-9

        tw = msg.twist

        u = np.array([
            tw.linear.x,
            tw.linear.y,
            tw.linear.z,
            tw.angular.x,
            tw.angular.y,
            tw.angular.z
        ])

        cmd_t.append(ts)
        cmd_x.append(u)

cmd_t = np.asarray(cmd_t)
cmd_x = np.asarray(cmd_x)

print(cmd_t.shape)
print(cmd_x.shape)
print("time range:", cmd_t[0], "to", cmd_t[-1])
print("first command:", cmd_x[0])

# %% notebook cell 24
cmd_sync = interpolate_stream(
    cmd_t,
    cmd_x,
    imu_time
)

print(cmd_sync.shape)

# %% notebook cell 25
np.savez_compressed(
    "/content/tartanground_processed_P2000.npz",

    time=imu_time,

    P=P,
    B=B,
    command=cmd_sync,

    force_FL=lf_force,
    force_FR=rf_force,
    force_RL=lh_force,
    force_RR=rh_force
)

print("Saved.")

# %% notebook cell 26
import numpy as np

data_npz = np.load(
    "/content/tartanground_processed_P2000.npz"
)

time = data_npz["time"]
P = data_npz["P"]                 # (12920, 4, 12)
B = data_npz["B"]                 # (12920, 12)
U = data_npz["command"]           # (12920, 6)

print("Before crop:")
print("P:", P.shape)
print("B:", B.shape)
print("U:", U.shape)

# command starts at 4.04 s
valid = time >= 4.04

time = time[valid]
P = P[valid]
B = B[valid]
U = U[valid]

# reset time origin
time = time - time[0]

print("\nAfter crop:")
print("time:", time.shape)
print("P:", P.shape)
print("B:", B.shape)
print("U:", U.shape)
print("duration:", time[-1])

# %% notebook cell 27
T = 50       # history = 0.5 s
H = 10       # prediction horizon = 0.1 s

X_leg = []
X_body = []
X_cmd = []
Y = []

for t in range(T - 1, len(P) - H):

    # Past 0.5 s
    leg_history = P[t-T+1:t+1]         # (50, 4, 12)

    # put legs first
    leg_history = np.transpose(
        leg_history,
        (1, 0, 2)
    )                                  # (4, 50, 12)

    body_history = B[t-T+1:t+1]        # (50, 12)

    command = U[t]                     # (6,)

    future = t + H

    # future forces
    # last 3 dimensions in each leg token = force xyz
    future_forces = P[future, :, 9:12].reshape(-1)

    # B = acc(3), gyro(3), body_twist(6)
    future_body_twist = B[future, 6:12]

    target = np.concatenate([
        future_forces,
        future_body_twist
    ])

    X_leg.append(leg_history)
    X_body.append(body_history)
    X_cmd.append(command)
    Y.append(target)

X_leg = np.asarray(X_leg, dtype=np.float32)
X_body = np.asarray(X_body, dtype=np.float32)
X_cmd = np.asarray(X_cmd, dtype=np.float32)
Y = np.asarray(Y, dtype=np.float32)

print("X_leg :", X_leg.shape)
print("X_body:", X_body.shape)
print("X_cmd :", X_cmd.shape)
print("Y     :", Y.shape)

# %% notebook cell 28
N = len(X_leg)

gap = T + H       # 60 samples

train_end = int(0.70 * N)
val_end   = int(0.85 * N)

train_ids = np.arange(0, train_end)

val_start = train_end + gap
val_ids = np.arange(val_start, val_end)

test_start = val_end + gap
test_ids = np.arange(test_start, N)

print("Train:", len(train_ids))
print("Val  :", len(val_ids))
print("Test :", len(test_ids))

print("Gap train->val:", val_start - train_end)
print("Gap val->test :", test_start - val_end)

# %% notebook cell 29
eps = 1e-6

leg_mean = X_leg[train_ids].mean(
    axis=(0, 1, 2),
    keepdims=True
)
leg_std = X_leg[train_ids].std(
    axis=(0, 1, 2),
    keepdims=True
) + eps

body_mean = X_body[train_ids].mean(
    axis=(0, 1),
    keepdims=True
)
body_std = X_body[train_ids].std(
    axis=(0, 1),
    keepdims=True
) + eps

cmd_mean = X_cmd[train_ids].mean(
    axis=0,
    keepdims=True
)
cmd_std = X_cmd[train_ids].std(
    axis=0,
    keepdims=True
) + eps

y_mean = Y[train_ids].mean(
    axis=0,
    keepdims=True
)
y_std = Y[train_ids].std(
    axis=0,
    keepdims=True
) + eps

X_leg_n  = (X_leg  - leg_mean)  / leg_std
X_body_n = (X_body - body_mean) / body_std
X_cmd_n  = (X_cmd  - cmd_mean)  / cmd_std
Y_n      = (Y      - y_mean)    / y_std

print("Normalization done")

# %% notebook cell 30
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset
import numpy as np
import copy
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Convert normalized arrays once
X_leg_t = torch.from_numpy(
    X_leg_n.astype(np.float32)
)

X_body_t = torch.from_numpy(
    X_body_n.astype(np.float32)
)

X_cmd_t = torch.from_numpy(
    X_cmd_n.astype(np.float32)
)

Y_t = torch.from_numpy(
    Y_n.astype(np.float32)
)

dataset = TensorDataset(
    X_leg_t,
    X_body_t,
    X_cmd_t,
    Y_t
)

train_set = Subset(dataset, train_ids.tolist())
val_set   = Subset(dataset, val_ids.tolist())
test_set  = Subset(dataset, test_ids.tolist())

train_loader = DataLoader(
    train_set,
    batch_size=64,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_set,
    batch_size=128,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    test_set,
    batch_size=128,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print("Train batches:", len(train_loader))
print("Val batches:", len(val_loader))
print("Test batches:", len(test_loader))

# %% notebook cell 31
class SharedLegTransformer(nn.Module):

    def __init__(
        self,
        input_dim=12,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_ff=256,
        max_len=50,
        dropout=0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len

        self.input_proj = nn.Linear(
            input_dim,
            d_model
        )

        # learnable temporal position embedding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_len, d_model) * 0.02
        )

        # FL, FR, RL, RR identity
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

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):

        # x:
        # B x 4 x T x 12

        B, L, T, F = x.shape

        # combine robot-leg dimension with batch
        x = x.reshape(
            B * L,
            T,
            F
        )

        x = self.input_proj(x)

        # temporal information
        x = x + self.pos_embedding[:, :T]

        # leg IDs:
        # FL FR RL RR FL FR RL RR ...
        leg_ids = torch.arange(
            L,
            device=x.device
        ).repeat(B)

        leg_emb = self.leg_embedding(
            leg_ids
        ).unsqueeze(1)

        x = x + leg_emb

        # causal mask
        # True = cannot attend
        causal_mask = torch.triu(
            torch.ones(
                T,
                T,
                device=x.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        x = self.transformer(
            x,
            mask=causal_mask
        )

        x = self.norm(x)

        # current state = latest temporal token
        h = x[:, -1, :]

        # restore four legs
        h = h.reshape(
            B,
            L,
            self.d_model
        )

        return h

# %% notebook cell 32
class BodyTransformer(nn.Module):

    def __init__(
        self,
        input_dim=12,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_ff=256,
        max_len=50,
        dropout=0.1
    ):
        super().__init__()

        self.input_proj = nn.Linear(
            input_dim,
            d_model
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_len, d_model) * 0.02
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

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):

        # B x T x 12

        B, T, F = x.shape

        x = self.input_proj(x)

        x = x + self.pos_embedding[:, :T]

        causal_mask = torch.triu(
            torch.ones(
                T,
                T,
                device=x.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        x = self.transformer(
            x,
            mask=causal_mask
        )

        x = self.norm(x)

        return x[:, -1, :]

# %% notebook cell 33
class GraphAttentionBlock(nn.Module):

    def __init__(
        self,
        d_model=128,
        nhead=4,
        dim_ff=256,
        dropout=0.1
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):

        h = self.norm1(x)

        attn_out, _ = self.attn(
            h,
            h,
            h,
            attn_mask=mask,
            need_weights=False
        )

        x = x + self.dropout(attn_out)

        x = x + self.dropout(
            self.ff(
                self.norm2(x)
            )
        )

        return x

# %% notebook cell 34
class MorphologyCouplingModel(nn.Module):

    def __init__(
        self,
        d_model=128,
        nhead=4,
        temporal_layers=2,
        graph_layers=2,
        dropout=0.1
    ):
        super().__init__()

        # same temporal encoder for all 4 legs
        self.leg_encoder = SharedLegTransformer(
            input_dim=12,
            d_model=d_model,
            nhead=nhead,
            num_layers=temporal_layers,
            dim_ff=2*d_model,
            max_len=50,
            dropout=dropout
        )

        self.body_encoder = BodyTransformer(
            input_dim=12,
            d_model=d_model,
            nhead=nhead,
            num_layers=temporal_layers,
            dim_ff=2*d_model,
            max_len=50,
            dropout=dropout
        )

        # global commanded motion
        self.command_encoder = nn.Sequential(
            nn.Linear(6, 64),
            nn.GELU(),
            nn.Linear(64, d_model)
        )

        # combine each leg with command
        self.leg_node_proj = nn.Sequential(
            nn.Linear(2*d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )

        # combine body state with command
        self.body_node_proj = nn.Sequential(
            nn.Linear(2*d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )

        self.graph_blocks = nn.ModuleList([
            GraphAttentionBlock(
                d_model=d_model,
                nhead=nhead,
                dim_ff=2*d_model,
                dropout=dropout
            )
            for _ in range(graph_layers)
        ])

        # same prediction head applied to every leg
        self.leg_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 3)
        )

        # body predicts 3 linear + 3 angular velocity
        self.body_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 6)
        )

    def graph_mask(self, device):

        # True means attention is forbidden

        mask = torch.ones(
            5,
            5,
            dtype=torch.bool,
            device=device
        )

        # self attention
        mask.fill_diagonal_(False)

        # BODY <-> each leg
        for leg in range(1, 5):
            mask[0, leg] = False
            mask[leg, 0] = False

        return mask

    def forward(
        self,
        x_leg,
        x_body,
        x_cmd
    ):

        B = x_leg.shape[0]

        # --------------------
        # Temporal encoding
        # --------------------

        leg_h = self.leg_encoder(
            x_leg
        )

        # B x 4 x d

        body_h = self.body_encoder(
            x_body
        )

        # B x d

        cmd_h = self.command_encoder(
            x_cmd
        )

        # B x d

        # --------------------
        # Create leg nodes
        # --------------------

        cmd_for_legs = cmd_h.unsqueeze(1).expand(
            -1,
            4,
            -1
        )

        leg_nodes = self.leg_node_proj(
            torch.cat([
                leg_h,
                cmd_for_legs
            ], dim=-1)
        )

        # --------------------
        # Create body node
        # --------------------

        body_node = self.body_node_proj(
            torch.cat([
                body_h,
                cmd_h
            ], dim=-1)
        )

        body_node = body_node.unsqueeze(1)

        # node order:
        # BODY FL FR RL RR

        nodes = torch.cat([
            body_node,
            leg_nodes
        ], dim=1)

        # B x 5 x d

        mask = self.graph_mask(
            nodes.device
        )

        for block in self.graph_blocks:
            nodes = block(
                nodes,
                mask
            )

        # --------------------
        # Outputs
        # --------------------

        body_final = nodes[:, 0]

        leg_final = nodes[:, 1:5]

        # SAME force head for all legs
        leg_force = self.leg_head(
            leg_final
        )

        # B x 4 x 3

        leg_force = leg_force.reshape(
            B,
            12
        )

        body_twist = self.body_head(
            body_final
        )

        # Final ordering matches Y exactly:
        #
        # FL force
        # FR force
        # RL force
        # RR force
        # body linear velocity
        # body angular velocity

        y = torch.cat([
            leg_force,
            body_twist
        ], dim=-1)

        return y

# %% notebook cell 35
model = MorphologyCouplingModel(
    d_model=128,
    nhead=4,
    temporal_layers=2,
    graph_layers=2,
    dropout=0.1
).to(device)

print(model)

n_params = sum(
    p.numel()
    for p in model.parameters()
)

print("\nParameters:", f"{n_params:,}")

# %% notebook cell 36
batch = next(iter(train_loader))

leg_b, body_b, cmd_b, y_b = [
    x.to(device)
    for x in batch
]

print("Leg input :", leg_b.shape)
print("Body input:", body_b.shape)
print("Command   :", cmd_b.shape)
print("Target    :", y_b.shape)

with torch.no_grad():

    pred = model(
        leg_b,
        body_b,
        cmd_b
    )

print("Prediction:", pred.shape)

# %% notebook cell 37
criterion = nn.MSELoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)

# %% notebook cell 38
def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device
):

    model.train()

    total_loss = 0.0
    n = 0

    for leg, body, cmd, target in loader:

        leg = leg.to(device, non_blocking=True)
        body = body.to(device, non_blocking=True)
        cmd = cmd.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad()

        pred = model(
            leg,
            body,
            cmd
        )

        loss = criterion(
            pred,
            target
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        batch_size = leg.shape[0]

        total_loss += loss.item() * batch_size
        n += batch_size

    return total_loss / n


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0.0
    n = 0

    for leg, body, cmd, target in loader:

        leg = leg.to(device, non_blocking=True)
        body = body.to(device, non_blocking=True)
        cmd = cmd.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        pred = model(
            leg,
            body,
            cmd
        )

        loss = criterion(
            pred,
            target
        )

        batch_size = leg.shape[0]

        total_loss += loss.item() * batch_size
        n += batch_size

    return total_loss / n

# %% notebook cell 39
num_epochs = 25
patience = 6

best_val = float("inf")
best_state = None
epochs_without_improvement = 0

history = {
    "train": [],
    "val": []
}

for epoch in range(
    1,
    num_epochs + 1
):

    start = time.time()

    train_loss = train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion,
        device
    )

    val_loss = evaluate(
        model,
        val_loader,
        criterion,
        device
    )

    scheduler.step(val_loss)

    history["train"].append(train_loss)
    history["val"].append(val_loss)

    elapsed = time.time() - start

    lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch:02d} | "
        f"train {train_loss:.5f} | "
        f"val {val_loss:.5f} | "
        f"lr {lr:.2e} | "
        f"{elapsed:.1f}s"
    )

    if val_loss < best_val:

        best_val = val_loss

        best_state = copy.deepcopy(
            model.state_dict()
        )

        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1

        if epochs_without_improvement >= patience:

            print("Early stopping.")
            break


model.load_state_dict(best_state)

print("\nBest validation loss:", best_val)

# %% notebook cell 40
test_loss = evaluate(
    model,
    test_loader,
    criterion,
    device
)

print("Normalized test MSE:", test_loss)

# %% notebook cell 41
@torch.no_grad()
def get_predictions(
    model,
    loader,
    device
):

    model.eval()

    preds = []
    targets = []

    for leg, body, cmd, target in loader:

        leg = leg.to(device)
        body = body.to(device)
        cmd = cmd.to(device)

        pred = model(
            leg,
            body,
            cmd
        )

        preds.append(
            pred.cpu().numpy()
        )

        targets.append(
            target.numpy()
        )

    return (
        np.concatenate(preds),
        np.concatenate(targets)
    )


pred_n, true_n = get_predictions(
    model,
    test_loader,
    device
)

# undo target normalization
pred = (
    pred_n * y_std.reshape(1, -1)
    + y_mean.reshape(1, -1)
)

true = (
    true_n * y_std.reshape(1, -1)
    + y_mean.reshape(1, -1)
)

print(pred.shape)
print(true.shape)

# %% notebook cell 42
def rmse(a, b):
    return np.sqrt(
        np.mean(
            (a - b) ** 2
        )
    )


names = [
    ("FL force", 0, 3),
    ("FR force", 3, 6),
    ("RL force", 6, 9),
    ("RR force", 9, 12),
    ("Body linear velocity", 12, 15),
    ("Body angular velocity", 15, 18),
]

for name, a, b in names:

    value = rmse(
        pred[:, a:b],
        true[:, a:b]
    )

    print(
        f"{name:25s}: {value:.5f}"
    )
