"""Causal V7 SHORT foothold data utilities; embedded verbatim in the delivered notebook."""
import json
from pathlib import Path
import numpy as np


def visual_window_indices(camera_times, anchor_t, max_age=.12, max_gap=.20):
    t = np.asarray(camera_times, dtype=np.float64)
    if t.ndim != 1 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError('Camera times must be finite and strictly increasing')
    j = int(np.searchsorted(t, anchor_t, side='right') - 1)
    if j < 4 or anchor_t - t[j] > max_age or np.any(np.diff(t[j-4:j+1]) > max_gap):
        raise ValueError('Five recent synchronized RGB-D frames are unavailable')
    ids = np.arange(j-4, j+1, dtype=np.int64)
    if np.any(t[ids] > anchor_t + 1e-9):
        raise RuntimeError('Future camera frame entered causal window')
    return ids


def touchdown_events(time, contact, min_confirm_s=.02):
    """(onset, first causal confirmation time, leg); an incomplete run-end event is dropped."""
    t = np.asarray(time, dtype=np.float64)
    c = np.asarray(contact) > .5
    if c.shape != (len(t), 4) and c.shape != (len(t), 1):
        raise ValueError('Contact must have one or four leg columns')
    if np.any(np.diff(t) <= 0):
        raise ValueError('Robot times must increase')
    out = []
    for leg in range(c.shape[1]):
        for i in range(1, len(t)):
            if not c[i, leg] or c[i-1, leg]:
                continue
            k = i
            while k < len(t) and c[k, leg] and t[k] - t[i] < min_confirm_s - 1e-10:
                k += 1
            if k < len(t) and c[k, leg]:
                out.append((float(t[i]), float(t[k]), leg))
    return sorted(out)


def causal_events(events, anchor_t):
    return [e for e in events if e[1] <= anchor_t + 1e-10 and e[0] < anchor_t - 1e-10]


def first_future_touchdown(events, anchor_t, horizon_s, leg):
    candidates = [e for e in events if e[2] == leg and anchor_t < e[0] <= anchor_t + horizon_s + 1e-10]
    return min(candidates, default=None, key=lambda e: e[0])


def target_end_ok(anchor_t, horizon_s, cutoff_t):
    return anchor_t + horizon_s <= cutoff_t + 1e-10


def load_verified_calibration(path):
    """Require explicit measured calibration; never infer camera/base alignment."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f'Missing measured V7 SHORT calibration JSON: {p}')
    c = json.loads(p.read_text(encoding='utf-8'))
    required = ('verified_source', 'width', 'height', 'fx', 'fy', 'cx', 'cy',
                'depth_unit_m', 'camera_axes', 'pose_convention', 'T_base_camera')
    missing = [k for k in required if k not in c]
    if missing:
        raise ValueError(f'Calibration missing fields: {missing}')
    if not str(c['verified_source']).strip():
        raise ValueError('Calibration requires a real measured/documented source')
    expected = dict(width=640, height=640, fx=320., fy=320., cx=320., cy=320., depth_unit_m=1.)
    for k, v in expected.items():
        if not np.isclose(float(c[k]), v, rtol=0, atol=1e-5):
            raise ValueError(f'{k} conflicts with baseline RGB-D geometry; rebuild baseline caches')
    if c['camera_axes'] != 'forward_right_down' or c['pose_convention'] != 'T_world_camera_NED':
        raise ValueError('Camera axis or pose convention is incompatible with V7 geometry')
    T = np.asarray(c['T_base_camera'], dtype=np.float64)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        raise ValueError('T_base_camera must be a finite 4x4 matrix')
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError('Invalid homogeneous transform bottom row')
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-3) or not np.isclose(np.linalg.det(R), 1., atol=1e-3):
        raise ValueError('T_base_camera rotation must be proper and orthonormal')
    c['T_base_camera'] = T
    return c


def pose_matrix(pose7):
    p = np.asarray(pose7, dtype=np.float64)
    if p.shape != (7,) or not np.all(np.isfinite(p)):
        raise ValueError('Camera pose must be 7 finite numbers')
    x, y, z, w = p[3:]
    qn = np.linalg.norm([x, y, z, w])
    if not np.isclose(qn, 1., atol=1e-2):
        raise ValueError('Camera pose quaternion is not normalized')
    x, y, z, w = np.array([x, y, z, w]) / qn
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p[:3]
    return T


def camera_index_at(manifest, t, max_pose_age=.12):
    ct = np.asarray(manifest['camera_times'])
    j = int(np.searchsorted(ct, t, side='right') - 1)
    if j < 0 or t - ct[j] > max_pose_age or ct[j] > t + 1e-9:
        raise ValueError('No recent causal camera pose at robot event')
    return j


def foot_world_at(rec, robot_idx, leg, calibration, fk, max_pose_age=.12):
    data, man = rec['data'], rec['manifest']
    t = float(data['time'][robot_idx])
    j = camera_index_at(man, t, max_pose_age)
    T_wc = pose_matrix(man['poses'][j])
    T_wb = T_wc @ np.linalg.inv(calibration['T_base_camera'])
    foot_base = fk.foot_positions(data['P'][robot_idx, :, :3])[leg]
    return T_wb[:3, :3] @ foot_base + T_wb[:3, 3]


def event_records(rec, calibration, fk, min_confirm_s=.02):
    data = rec['data']
    events = touchdown_events(data['time'], data['contact'], min_confirm_s)
    rows = []
    for onset, confirmed, leg in events:
        i = int(np.searchsorted(data['time'], onset))
        try:
            pos = foot_world_at(rec, i, leg, calibration, fk)
        except ValueError:
            pos = None  # retain the event so it cannot become a false no-touch label
        force = float(np.linalg.norm(data['P'][i, leg, 9:12]))
        rows.append(dict(onset=onset, confirmed=confirmed, leg=leg, xyz=pos, force=force))
    return rows


def cell_for_point(xyz, grid_xyz, grid_valid, max_xy=.16, max_z=.25):
    valid = np.flatnonzero(grid_valid)
    if not len(valid):
        return None
    delta = grid_xyz[valid] - np.asarray(xyz)
    dist_xy = np.linalg.norm(delta[:, :2], axis=1)
    choices = np.where((dist_xy <= max_xy) & (np.abs(delta[:, 2]) <= max_z))[0]
    if not len(choices):
        return None
    k = choices[np.argmin(dist_xy[choices] + .2 * np.abs(delta[choices, 2]))]
    return int(valid[k])


def memory_at(anchor_t, rows, grid_xyz, grid_valid, camera_xyz):
    """Last five touchdown tokens per leg and prior physical experience per valid cell."""
    past = [r for r in rows if r['confirmed'] <= anchor_t + 1e-10 and r['onset'] < anchor_t - 1e-10 and r['xyz'] is not None]
    touches = np.zeros((4, 5, 6), np.float32)  # rel XYZ, force norm, age, validity
    cells = np.zeros((64, 4), np.float32)      # log count, mean force, age, validity
    buckets = [[] for _ in range(64)]
    for leg in range(4):
        recent = [r for r in past if r['leg'] == leg][-5:]
        for slot, r in enumerate(reversed(recent)):
            touches[leg, slot] = [*((r['xyz']-camera_xyz).tolist()), r['force'], anchor_t-r['onset'], 1.]
    for r in past:
        cell = cell_for_point(r['xyz'], grid_xyz, grid_valid)
        if cell is not None:
            buckets[cell].append(r)
    for c, b in enumerate(buckets):
        if b:
            cells[c] = [np.log1p(len(b)), np.mean([r['force'] for r in b]), anchor_t-b[-1]['onset'], 1.]
    return touches, cells


def target_at(anchor_t, horizon_s, rows, grid_xyz, grid_valid, camera_xyz):
    cell = np.full(4, 64, np.int64)  # 64 = observed no-touchdown within horizon
    xyz = np.zeros((4, 3), np.float32)
    tau = np.zeros(4, np.float32)
    valid = np.ones(4, np.float32)
    positive = np.zeros(4, np.float32)
    for leg in range(4):
        future = [r for r in rows if r['leg'] == leg and anchor_t < r['onset'] <= anchor_t+horizon_s+1e-10]
        if not future:
            continue
        r = min(future, key=lambda x: x['onset'])
        if r['confirmed'] > anchor_t+horizon_s+1e-10 or r['xyz'] is None:
            valid[leg] = 0.  # observed onset with unavailable verified label
            continue
        c = cell_for_point(r['xyz'], grid_xyz, grid_valid)
        if c is None:
            valid[leg] = 0.  # event occurred outside observed valid ground; exclude, not no-touch
            continue
        cell[leg], xyz[leg], tau[leg], positive[leg] = c, r['xyz']-camera_xyz, r['onset']-anchor_t, 1.
    return dict(cell=cell, xyz=xyz, tau=tau, valid=valid, positive=positive)
