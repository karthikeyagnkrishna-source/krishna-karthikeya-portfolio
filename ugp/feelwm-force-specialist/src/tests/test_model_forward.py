from __future__ import annotations

import unittest

import torch

from config import ModelConfig, TrainConfig
from losses import compute_losses
from model import MergedFeelWMV7
from smoke_test import sample


class ModelForwardTest(unittest.TestCase):
    def test_module_call_runs_autoregressive_rollout(self) -> None:
        cfg = ModelConfig(
            world_d_model=80,
            world_heads=10,
            world_layers=1,
            world_ffn_ratio=2,
            query_layers=1,
            hist_layers=1,
            v7f_layers=1,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MergedFeelWMV7(cfg).to(device)
        batch = sample(B=1, K=1, T=5)

        try:
            output = model(batch, history_shift=1)
        except TypeError as exc:
            self.fail(f"module call must run the rollout instead of raising TypeError: {exc}")

        self.assertEqual(tuple(output["steps"][0]["z_next"].shape), (1, 64, 256))
        self.assertEqual(tuple(output["v7f"]["force"].shape), (1, 1, 4, 3))

    def test_force_auxiliary_head_receives_gradient(self) -> None:
        cfg = ModelConfig(
            world_d_model=80,
            world_heads=10,
            world_layers=1,
            world_ffn_ratio=2,
            query_layers=1,
            hist_layers=1,
            v7f_layers=1,
        )
        model = MergedFeelWMV7(cfg).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        batch = sample(B=1, K=1, T=5)
        output = model(batch, history_shift=1)
        loss = compute_losses(output, batch, cfg, TrainConfig(amp_dtype="fp32"))["total"]
        loss.backward()

        for name, parameter in model.force_aux_head.named_parameters():
            self.assertIsNotNone(parameter.grad, f"force_aux_head.{name} did not receive a gradient")
            self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
