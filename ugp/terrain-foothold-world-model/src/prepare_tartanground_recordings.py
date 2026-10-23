#!/usr/bin/env python3
"""Prepare a bounded, synchronized TartanGround pilot for the supplied model.

Only logged RGB-D, camera poses, robot state, contact forces, leg TF and
torque-mode actuator commands are used. The output is simulated robot data,
not physical force-sensor recordings. No model code or weights are imported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from scipy.spatial.transform import Rotation


LEGS = ("LF", "RF", "LH", "RH")
WORLD_ENU_TO_NED = np.diag([1.0, -1.0, -1.0])
DEFAULT_RECORDINGS = (
    ("ForestEnv", "P2000"),
    ("OldTownFall", "P2000"),
    ("GreatMarsh", "P2023"),
    ("SeasonalForestWinter", "P2000"),
)


def matrix_from_pose(position, quaternion):
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    out[:3, 3] = position
    return out


def matrix_from_tf(transform):
    v = transform.translation
    q = transform.rotation
    return matrix_from_pose((v.x, v.y, v.z), (q.x, q.y, q.z, q.w))


def nearest_indices(times, query):
    idx = np.clip(np.searchsorted(times, query), 0, len(times) - 1)
    return np.where(
        (idx > 0) & (np.abs(times[idx - 1] - query) < np.abs(times[idx] - query)),
        idx - 1, idx,
    )


def resolve_tf(edges, child):
    chain = []
    seen = set()
    while child != "odom":
        if child in seen or child not in edges:
            raise ValueError(f"Incomplete or cyclic TF chain to {child}")
        seen.add(child)
        parent, matrix = edges[child]
        chain.append(matrix)
        child = parent
    result = np.eye(4)
    for edge in reversed(chain):
        result = result @ edge
    return result


def load_bag(path):
    states, actuators, tf_events, static = [], [], [], {}
    wanted = {
        "/state_estimator/anymal_state",
        "/state_estimator/actuator_readings",
        "/tf",
        "/tf_static",
    }
    with AnyReader([path]) as reader:
        conns = [conn for conn in reader.connections if conn.topic in wanted]
        for conn, stamp, raw in reader.messages(connections=conns):
            message = reader.deserialize(raw, conn.msgtype)
            time_s = stamp / 1e9
            if conn.topic == "/state_estimator/anymal_state":
                pose = message.pose.pose
                p = pose.position
                q = pose.orientation
                joint = message.joints
                if len(joint.position) != 12 or len(joint.velocity) != 12 or len(joint.effort) != 12:
                    raise ValueError("Expected 12 observed joints")
                observed_names = [f"{leg}_{joint_name}" for leg in LEGS for joint_name in ("HAA", "HFE", "KFE")]
                if list(joint.name) != observed_names:
                    raise ValueError(f"Unexpected joint order: {joint.name}")
                contacts = {x.name: x for x in message.contacts}
                if set(contacts) != {f"{leg}_FOOT" for leg in LEGS}:
                    raise ValueError(f"Unexpected contact names: {list(contacts)}")
                force = []
                contact = []
                contact_position = []
                for leg in LEGS:
                    x = contacts[f"{leg}_FOOT"]
                    if x.header.frame_id != "odom":
                        raise ValueError(f"Contact force frame is {x.header.frame_id!r}")
                    f = x.wrench.force
                    force.append(WORLD_ENU_TO_NED @ np.array([f.x, f.y, f.z]))
                    contact.append(float(x.state in (x.STATE_CLOSED, x.STATE_SLIPPING)))
                    pfoot = x.position
                    contact_position.append(WORLD_ENU_TO_NED @ np.array([pfoot.x, pfoot.y, pfoot.z]))
                twist = message.twist.twist
                body_twist = [twist.linear.x, twist.linear.y, twist.linear.z,
                              twist.angular.x, twist.angular.y, twist.angular.z]
                states.append(dict(
                    time=time_s,
                    position=WORLD_ENU_TO_NED @ np.array([p.x, p.y, p.z]),
                    rotation=WORLD_ENU_TO_NED @ Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix(),
                    proprio=np.r_[joint.position, joint.velocity, joint.effort, body_twist],
                    force=np.array(force), contact=np.array(contact),
                    contact_position=np.array(contact_position),
                ))
            elif conn.topic == "/state_estimator/actuator_readings":
                readings = message.readings
                if len(readings) != 12:
                    raise ValueError("Expected 12 actuator readings")
                if any(x.commanded.mode != x.commanded.MODE_JOINT_TORQUE for x in readings):
                    raise ValueError("Actuator not in logged torque command mode")
                actuators.append((time_s,
                                  np.array([x.commanded.joint_torque for x in readings]),
                                  np.array([x.state.joint_position for x in readings])))
            elif conn.topic in ("/tf", "/tf_static"):
                edges = []
                for transform in message.transforms:
                    edges.append((transform.child_frame_id,
                                  (transform.header.frame_id, matrix_from_tf(transform.transform))))
                if conn.topic == "/tf_static":
                    static.update(edges)
                else:
                    tf_events.append((time_s, edges))
    if not states or not actuators or not tf_events:
        raise ValueError("Missing state, actuator or dynamic TF stream")
    return states, actuators, tf_events, static


def estimate_camera_offset(camera_pose, camera_time, state):
    times = np.array([x["time"] for x in state])
    positions = np.array([x["position"] for x in state])
    early = np.where(times < times[0] + 12.0)[0]
    offsets = []
    for i in range(min(50, len(camera_pose))):
        idx = early[np.argmin(np.linalg.norm(positions[early] - camera_pose[i, :3], axis=1))]
        offsets.append(times[idx] - camera_time[i])
    offset = float(np.median(offsets))
    matched = nearest_indices(times, camera_time + offset)
    pos_error = np.linalg.norm(positions[matched] - camera_pose[:, :3], axis=1)
    cam_rot = Rotation.from_quat(camera_pose[:, 3:7]).as_matrix()
    body_rot = np.array([state[i]["rotation"] for i in matched])
    expected = body_rot @ WORLD_ENU_TO_NED
    rot_error = Rotation.from_matrix(np.transpose(expected, (0, 2, 1)) @ cam_rot).magnitude()
    time_error = np.abs(times[matched] - (camera_time + offset))
    stats = dict(offset_s=offset,
                 position_median_m=float(np.median(pos_error)),
                 position_p95_m=float(np.percentile(pos_error, 95)),
                 rotation_p95_deg=float(np.percentile(rot_error, 95) * 180 / np.pi),
                 time_p95_ms=float(np.percentile(time_error, 95) * 1000))
    if (stats["position_p95_m"] > 0.10 or stats["rotation_p95_deg"] > 2.0
            or stats["time_p95_ms"] > 60):
        raise ValueError(f"Camera/base alignment failed: {stats}")
    return offset, matched, stats


def foot_positions(camera_absolute_time, tf_events, static):
    edges = dict(static)
    result = []
    event_index = 0
    for target in camera_absolute_time:
        while event_index < len(tf_events) and tf_events[event_index][0] <= target:
            edges.update(tf_events[event_index][1])
            event_index += 1
        world = []
        for leg in LEGS:
            foot = resolve_tf(edges, f"{leg}_FOOT")
            world.append(WORLD_ENU_TO_NED @ foot[:3, 3])
        result.append(world)
    return np.asarray(result)


def prepare_one(root, calibration, output, environment, sequence, start, frames, side):
    source = root / environment / "Data_anymal" / sequence
    if not source.is_dir():
        raise FileNotFoundError(source)
    pose = np.loadtxt(source / "pose_lcam_front.txt")
    camera_time = np.loadtxt(source / "imu/cam_time.txt")
    if pose.shape != (len(camera_time), 7):
        raise ValueError("Camera pose/timestamp rows disagree")
    state, actuators, tf_events, static = load_bag(source / "rosbags" / f"{sequence}.bag")
    offset, state_idx, alignment = estimate_camera_offset(pose, camera_time, state)
    selection = np.arange(start, min(start + frames, len(camera_time)))
    if len(selection) < 15:
        raise ValueError("Too few selected frames")
    absolute_time = camera_time[selection] + offset
    # The first image has no earlier camera image. Its preceding TF sample is
    # still logged in the bag, so use one camera interval before that image.
    if selection[0] == 0:
        previous_time = absolute_time[0] - float(np.median(np.diff(camera_time)))
    else:
        previous_time = camera_time[selection[0] - 1] + offset
    all_foot_times = np.r_[previous_time, absolute_time]
    all_feet = foot_positions(all_foot_times, tf_events, static)
    feet = all_feet[1:]
    foot_velocity = np.diff(all_feet, axis=0) / np.diff(all_foot_times)[:, None, None]
    actuator_time = np.array([x[0] for x in actuators])
    actuator_idx = np.searchsorted(actuator_time, absolute_time, side="right") - 1
    if np.any(actuator_idx < 0):
        raise ValueError("No torque command exists for a selected frame")
    actuator_age = absolute_time - actuator_time[actuator_idx]
    if actuator_age.max() > 0.1:
        raise ValueError(f"Stale actuator commands: max age {actuator_age.max():.3f}s")
    command_order_error = np.array([
        np.abs(actuators[a][2] - state[s]["proprio"][:12]).mean()
        for a, s in zip(actuator_idx, state_idx[selection])
    ])
    if np.median(command_order_error) > 0.1:
        raise ValueError("Actuator ordering does not match named joint states")
    Tbc = np.asarray(calibration["T_base_camera"], dtype=np.float64)
    if Tbc.shape != (4, 4) or not np.allclose(Tbc[:3, :3], WORLD_ENU_TO_NED, atol=0.01):
        raise ValueError("Camera extrinsic is not the independently checked synthetic rig transform")
    T_world_base = []
    for row in pose[selection]:
        Twc = matrix_from_pose(row[:3], row[3:7])
        T_world_base.append(Twc @ np.linalg.inv(Tbc))
    rgb, depth = [], []
    for i in selection:
        stem = f"{i:06d}_lcam_front"
        rgb_path = source / "image_lcam_front" / f"{stem}.png"
        depth_path = source / "depth_lcam_front" / f"{stem}_depth.png"
        color_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        packed = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if color_bgr is None or packed is None or packed.shape != (640, 640, 4):
            raise ValueError(f"Missing or malformed RGB-D pair at {stem}")
        color = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        z = packed.copy().view("<f4").reshape(640, 640)
        # Rendered depth can use very large finite values for distant/sky
        # pixels. The supplied model ignores depths outside (0.1, 20) m, so
        # validate that useful surface pixels remain without dropping frames.
        usable_fraction = np.mean((z > .1) & (z < 20.))
        if not np.isfinite(z).all() or (z < 0).any() or usable_fraction < .01:
            raise ValueError(f"Implausible depth decoding at {stem}")
        rgb.append(cv2.resize(color, (side, side), interpolation=cv2.INTER_AREA))
        depth.append(cv2.resize(z, (side, side), interpolation=cv2.INTER_NEAREST))
    scale = side / 640.0
    K = np.array([[calibration["fx"] * scale, 0, (calibration["cx"] + .5) * scale - .5],
                  [0, calibration["fy"] * scale, (calibration["cy"] + .5) * scale - .5],
                  [0, 0, 1]], dtype=np.float32)
    contact = np.array([state[i]["contact"] for i in state_idx[selection]])
    contact_position = np.array([state[i]["contact_position"] for i in state_idx[selection]])
    closed_error = np.linalg.norm(feet - contact_position, axis=-1)[contact > .5]
    if len(closed_error) and np.median(closed_error) > .12:
        raise ValueError(f"TF foot/contact position mismatch: median {np.median(closed_error):.3f}m")
    arrays = dict(
        rgb=np.asarray(rgb, dtype=np.uint8),
        depth=np.asarray(depth, dtype=np.float32),
        K=K, T_base_camera=Tbc.astype(np.float32),
        T_world_base=np.asarray(T_world_base, dtype=np.float32),
        proprio=np.asarray([state[i]["proprio"] for i in state_idx[selection]], dtype=np.float32),
        actions=np.asarray([actuators[i][1] for i in actuator_idx], dtype=np.float32),
        foot_xyz_world=feet.astype(np.float32),
        foot_velocity_world=foot_velocity.astype(np.float32),
        foot_force_world=np.asarray([state[i]["force"] for i in state_idx[selection]], dtype=np.float32),
        contact=contact.astype(np.float32),
        timestamps=camera_time[selection].astype(np.float64),
    )
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"{environment}_{sequence}.npz"
    np.savez_compressed(destination, **arrays)
    report = dict(environment=environment, sequence=sequence, source=str(source),
                  bag=str(source / "rosbags" / f"{sequence}.bag"),
                  selected_frames=[int(selection[0]), int(selection[-1])],
                  arrays={k: list(v.shape) for k, v in arrays.items()},
                  alignment=alignment,
                  actuator_age_max_ms=float(actuator_age.max() * 1000),
                  actuator_joint_order_median_abs_error=float(np.median(command_order_error)),
                  closed_foot_tf_contact_median_error_m=float(np.median(closed_error)) if len(closed_error) else None,
                  foot_velocity_kind="causal backward difference of TF foot positions, including a preceding frame",
                  force_kind="logged simulated anymal_state contact wrench, not verified physical force sensor",
                  action_kind="logged torque-mode actuator command, 12 joints in verified state order",
                  camera_kind="synthetic lcam_front; RGB/depth resized with K updated")
    (output / f"{environment}_{sequence}.manifest.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(output=str(destination), **report), indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--start", type=int, default=50)
    p.add_argument("--frames", type=int, default=80)
    p.add_argument("--side", type=int, default=160)
    p.add_argument("--recording", action="append", help="Environment/trajectory; repeat as needed")
    args = p.parse_args()
    calibration = json.loads(args.calibration.read_text())
    if calibration.get("scope") != "TartanGround synthetic lcam_front only":
        raise ValueError("Calibration scope does not match camera stream")
    recordings = [x.split("/", 1) for x in args.recording] if args.recording else DEFAULT_RECORDINGS
    for environment, sequence in recordings:
        prepare_one(args.root, calibration, args.output, environment, sequence,
                    args.start, args.frames, args.side)


if __name__ == "__main__":
    main()
