import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from checkpointing import restore_rng_state


class RestoreRngStateTests(unittest.TestCase):
    def test_torch_rng_state_is_normalized_to_cpu_byte_tensor(self):
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.tensor([1, 2, 3], dtype=torch.int64),
        }

        with patch("checkpointing.torch.set_rng_state") as set_rng_state:
            restore_rng_state(state)

        restored = set_rng_state.call_args.args[0]
        self.assertEqual(restored.device.type, "cpu")
        self.assertEqual(restored.dtype, torch.uint8)
        self.assertEqual(restored.tolist(), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
