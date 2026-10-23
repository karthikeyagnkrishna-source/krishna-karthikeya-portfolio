import importlib
import unittest
import numpy as np


def history_module(test):
    try:
        return importlib.import_module('v7_visual_history.data')
    except ModuleNotFoundError:
        test.fail('Timestamped visual-history selection is not implemented')


class CameraHistoryTests(unittest.TestCase):
    def test_selects_five_preceding_frames_not_arbitrary_spacing(self):
        m = history_module(self)
        times = np.array([0., .1, .2, .3, .4, .5, .6, .7])
        ids, valid = m.select_camera_history(times, .605, count=6)
        np.testing.assert_array_equal(ids, [1, 2, 3, 4, 5, 6])
        self.assertTrue(valid.all())
        self.assertLessEqual(times[ids].max(), .605)

    def test_early_missing_frames_are_left_padded(self):
        m = history_module(self)
        ids, valid = m.select_camera_history(np.array([0., .1, .2]), .11)
        np.testing.assert_array_equal(valid, [False, False, False, False, True, True])
        np.testing.assert_array_equal(ids[valid], [0, 1])

    def test_actual_missing_frame_gap_is_preserved(self):
        m = history_module(self)
        times = np.array([0., .1, .2, .5, .6, .7, .8])
        ids, valid = m.select_camera_history(times, .705)
        np.testing.assert_array_equal(times[ids[valid]], [0., .1, .2, .5, .6, .7])

    def test_no_observed_frame_is_an_error(self):
        m = history_module(self)
        with self.assertRaisesRegex(ValueError, 'observed'):
            m.select_camera_history(np.array([1., 1.1]), .9)

    def test_duplicate_camera_timestamps_are_rejected(self):
        m = history_module(self)
        with self.assertRaisesRegex(ValueError, 'increasing'):
            m.select_camera_history(np.array([0., .1, .1]), .2)


if __name__ == '__main__':
    unittest.main()
