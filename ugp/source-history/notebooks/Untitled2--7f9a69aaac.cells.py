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
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

rgb = Image.open(RGB_FILE)
depth = np.load(DEPTH_FILE)

plt.figure(figsize=(6,6))
plt.imshow(rgb)
plt.axis("off")
plt.show()

plt.figure(figsize=(6,6))
plt.imshow(depth)
plt.colorbar(label="depth")
plt.show()

print(np.array(rgb).shape)
print(depth.shape)
print(depth.min(), depth.max())
