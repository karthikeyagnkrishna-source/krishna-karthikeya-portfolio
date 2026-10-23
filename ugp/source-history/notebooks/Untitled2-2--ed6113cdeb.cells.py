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

n_train = int(0.70 * N)
n_val   = int(0.15 * N)

train_idx = slice(0, n_train)
val_idx   = slice(n_train, n_train + n_val)
test_idx  = slice(n_train + n_val, N)

print("Train:", n_train)
print("Val  :", n_val)
print("Test :", N - n_train - n_val)

# %% notebook cell 29
eps = 1e-6

# Leg features
leg_mean = X_leg[train_idx].mean(
    axis=(0, 1, 2),
    keepdims=True
)

leg_std = X_leg[train_idx].std(
    axis=(0, 1, 2),
    keepdims=True
) + eps

# Body features
body_mean = X_body[train_idx].mean(
    axis=(0, 1),
    keepdims=True
)

body_std = X_body[train_idx].std(
    axis=(0, 1),
    keepdims=True
) + eps

# Command
cmd_mean = X_cmd[train_idx].mean(
    axis=0,
    keepdims=True
)

cmd_std = X_cmd[train_idx].std(
    axis=0,
    keepdims=True
) + eps

# Targets
y_mean = Y[train_idx].mean(
    axis=0,
    keepdims=True
)

y_std = Y[train_idx].std(
    axis=0,
    keepdims=True
) + eps


X_leg_n = (X_leg - leg_mean) / leg_std
X_body_n = (X_body - body_mean) / body_std
X_cmd_n = (X_cmd - cmd_mean) / cmd_std
Y_n = (Y - y_mean) / y_std

print("normalization done")

# %% notebook cell 30

