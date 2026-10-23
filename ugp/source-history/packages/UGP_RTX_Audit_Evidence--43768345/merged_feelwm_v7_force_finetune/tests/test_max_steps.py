import unittest

from config import TrainConfig
from train import max_steps_reached


class MaxStepsTest(unittest.TestCase):
    def test_max_steps_stops_after_exact_optimizer_step(self):
        cfg = TrainConfig(max_steps=3)

        self.assertFalse(max_steps_reached(0, cfg.max_steps))
        self.assertFalse(max_steps_reached(2, cfg.max_steps))
        self.assertTrue(max_steps_reached(3, cfg.max_steps))

    def test_resumed_run_does_not_repeat_completed_step(self):
        cfg = TrainConfig(max_steps=200)

        resumed_global_step = 200
        self.assertTrue(max_steps_reached(resumed_global_step, cfg.max_steps))

    def test_unbounded_training_preserves_existing_behavior(self):
        self.assertFalse(max_steps_reached(10_000, None))


if __name__ == "__main__":
    unittest.main()
