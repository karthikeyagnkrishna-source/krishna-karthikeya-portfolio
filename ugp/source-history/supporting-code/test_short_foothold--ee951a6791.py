import unittest
import numpy as np
import json
import tempfile
from pathlib import Path
import ast

from short_foothold_numpy import (visual_window_indices, touchdown_events, causal_events,
    first_future_touchdown, target_end_ok, memory_at, target_at, cell_for_point,
    load_verified_calibration, pose_matrix)


class CausalFootholdTests(unittest.TestCase):
    def test_five_frames_and_four_pairs_end_at_anchor(self):
        times = np.arange(8, dtype=float) * .05
        ids = visual_window_indices(times, .255, max_age=.06, max_gap=.06)
        np.testing.assert_array_equal(ids, [1, 2, 3, 4, 5])
        self.assertTrue(np.all(times[ids] <= .255))
        with self.assertRaises(ValueError):
            visual_window_indices(times, .12, max_age=.06, max_gap=.06)

    def test_contact_event_requires_past_confirmation(self):
        t = np.arange(10) * .01
        c = np.array([[0], [0], [0], [1], [1], [1], [0], [0], [1], [1]], float)
        events = touchdown_events(t, c, min_confirm_s=.02)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0][0], .03)
        self.assertEqual(len(causal_events(events, .04)), 0)
        self.assertEqual(len(causal_events(events, .05)), 1)

    def test_future_label_and_cutoff(self):
        events = [(1.1, 1.12, 0), (1.3, 1.32, 0)]
        self.assertEqual(first_future_touchdown(events, 1.0, .2, 0)[0], 1.1)
        self.assertIsNone(first_future_touchdown(events, 1.0, .05, 0))
        self.assertFalse(target_end_ok(1.0, .2, 1.19))
        self.assertTrue(target_end_ok(1.0, .2, 1.20))

    def test_memory_never_uses_unconfirmed_future_event(self):
        grid = np.zeros((64, 3), np.float32)
        valid = np.zeros(64, bool); valid[0] = True
        rows = [dict(onset=.9, confirmed=1.01, leg=0, xyz=np.zeros(3), force=20.),
                dict(onset=1.2, confirmed=1.22, leg=1, xyz=np.zeros(3), force=30.)]
        touches, cells = memory_at(1., rows, grid, valid, np.zeros(3))
        self.assertEqual(float(touches.sum()), 0.)
        self.assertEqual(float(cells.sum()), 0.)
        touches, cells = memory_at(1.05, rows, grid, valid, np.zeros(3))
        self.assertEqual(float(touches[0, 0, 3]), 20.)
        self.assertEqual(float(cells[0, 1]), 20.)
        self.assertEqual(float(touches[1].sum()), 0.)

    def test_outside_grid_event_is_excluded_not_labeled_no_touch(self):
        grid = np.zeros((64, 3), np.float32)
        valid = np.zeros(64, bool); valid[0] = True
        rows = [dict(onset=1.1, confirmed=1.12, leg=0, xyz=np.array([10., 0., 0.]), force=20.)]
        y = target_at(1., .6, rows, grid, valid, np.zeros(3))
        self.assertEqual(y['valid'][0], 0.)
        self.assertEqual(y['cell'][0], 64)
        self.assertEqual(y['valid'][1], 1.)
        self.assertIsNone(cell_for_point(np.array([10., 0., 0.]), grid, valid))

    def test_calibration_must_exist(self):
        with self.assertRaises(FileNotFoundError):
            load_verified_calibration('definitely_missing_calibration.json')

    def test_calibration_rejects_mismatched_geometry(self):
        c = dict(verified_source='measured rig record', width=640, height=640,
                 fx=320., fy=320., cx=320., cy=320., depth_unit_m=1.,
                 camera_axes='forward_right_down', pose_convention='T_world_camera_NED',
                 T_base_camera=np.eye(4).tolist())
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path = Path(folder) / 'cal.json'
            path.write_text(json.dumps(c))
            self.assertEqual(load_verified_calibration(path)['T_base_camera'].shape, (4, 4))
            c['fx'] = 330.
            path.write_text(json.dumps(c))
            with self.assertRaises(ValueError):
                load_verified_calibration(path)

    def test_pose_transform_identity_rotation(self):
        T = pose_matrix([1., 2., 3., 0., 0., 0., 1.])
        np.testing.assert_allclose(T[:3, :3], np.eye(3))
        np.testing.assert_allclose(T[:3, 3], [1., 2., 3.])

    def test_metric_grid_uses_cached_rgbd_points(self):
        tree = ast.parse(Path(__file__).with_name('short_foothold_model.py').read_text())
        node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'short_fm_frame_grid')
        scope = dict(np=np, heading_frame_from_pose=lambda pose: (np.eye(3), np.eye(3)))
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'grid_test', 'exec'), scope)
        geom = np.zeros((1, 64, 20), np.float32)
        geom[0, 0, -10:-7] = [1., 0., 0.]
        rec = dict(manifest=dict(crop_boxes=np.array([[0, 0, 640, 640]]),
                                 poses=np.array([[0., 0., 0., 0., 0., 0., 1.]])),
                   visual=dict(GEOM=geom))
        xyz, valid = scope['short_fm_frame_grid'](rec, 0)
        self.assertEqual(int(valid.sum()), 1)
        np.testing.assert_allclose(xyz[36], [1., 0., 0.])


if __name__ == '__main__':
    unittest.main()
