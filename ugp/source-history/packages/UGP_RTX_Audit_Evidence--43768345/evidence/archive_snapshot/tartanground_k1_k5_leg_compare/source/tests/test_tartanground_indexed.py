import unittest

import numpy as np

from tartanground.windows import (
    encode_cache_native_action,
    quarter_windows,
    split_anchors,
    union_with_membership,
)


class QuarterWindowTests(unittest.TestCase):
    def test_two_700_windows_are_quarter_centered_and_overlap_is_deduplicated(self):
        first, second = quarter_windows(1290)
        self.assertEqual((first.start, first.stop), (0, 700))
        self.assertEqual((second.start, second.stop), (590, 1290))
        self.assertAlmostEqual(first.requested_center, 0.25 * 1289)
        self.assertAlmostEqual(second.requested_center, 0.75 * 1289)

        indices, membership = union_with_membership((first, second))
        self.assertEqual(len(indices), 1290)
        self.assertEqual(membership[100], (0,))
        self.assertEqual(membership[650], (0, 1))
        self.assertEqual(membership[1200], (1,))

    def test_rejects_trajectories_that_cannot_separate_window_centers_by_five_seconds(self):
        with self.assertRaisesRegex(ValueError, "five seconds"):
            quarter_windows(740)

    def test_chronological_split_has_a_50_frame_purge(self):
        anchors = list(range(100, 1100))
        split = split_anchors(anchors)
        self.assertEqual(split["train"][0], 100)
        self.assertEqual(split["train"][-1], 899)
        self.assertEqual(split["purge"], list(range(900, 950)))
        self.assertEqual(split["dev"][0], 950)
        self.assertTrue(set(split["train"]).isdisjoint(split["dev"]))

    def test_purge_uses_camera_time_even_when_anchor_indices_are_sparse(self):
        split = split_anchors([0, 1, 2, 3, 4, 5, 6, 100, 101, 102], train_fraction=0.5)
        self.assertEqual(split["train"], [0, 1, 2, 3, 4])
        self.assertEqual(split["purge"], [5, 6])
        self.assertEqual(split["dev"], [100, 101, 102])


class CacheNativeActionTests(unittest.TestCase):
    def test_encodes_three_low_level_commands_and_next_change_magnitude(self):
        commands = np.array(
            [
                [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]],
                [[4.0, 6.0, 3.0], [0.0, 3.0, 4.0]],
                [[4.0, 6.0, 3.0], [0.0, 3.0, 4.0]],
            ],
            dtype=np.float32,
        )
        encoded = encode_cache_native_action(commands)
        np.testing.assert_allclose(encoded[..., :3], commands)
        np.testing.assert_allclose(encoded[:, :, 3], [[5.0, 5.0], [0.0, 0.0], [0.0, 0.0]])


if __name__ == "__main__":
    unittest.main()
