import unittest

import torch

from config import ModelConfig
from model import MergedFeelWMV7


class LegPreTransformerTests(unittest.TestCase):
    def config(self, enabled):
        return ModelConfig(
            world_d_model=80,
            world_heads=10,
            world_layers=1,
            world_ffn_ratio=2,
            hist_layers=1,
            query_layers=1,
            v7f_layers=1,
            use_leg_pretransformers=enabled,
            leg_pre_d_model=32,
            leg_pre_heads=4,
            leg_pre_layers=1,
        )

    def test_disabled_path_has_no_extra_parameters_and_preserves_groups(self):
        torch.manual_seed(13)
        model = MergedFeelWMV7(self.config(False))
        groups = torch.randn(2, 9, 64, 80)
        self.assertEqual(model.leg_pre_transformers, None)
        torch.testing.assert_close(model.apply_leg_pretransformers(groups), groups)
        self.assertFalse(any("leg_pre_transformers" in name for name, _ in model.named_parameters()))

    def test_same_seed_keeps_all_shared_weights_identical(self):
        torch.manual_seed(29)
        baseline = MergedFeelWMV7(self.config(False))
        torch.manual_seed(29)
        per_leg = MergedFeelWMV7(self.config(True))
        baseline_state = baseline.state_dict()
        leg_state = per_leg.state_dict()
        for name, value in baseline_state.items():
            torch.testing.assert_close(value, leg_state[name])

    def test_enabled_path_uses_four_independent_blocks_on_only_the_leg_groups(self):
        torch.manual_seed(13)
        model = MergedFeelWMV7(self.config(True))
        self.assertEqual(set(model.leg_pre_transformers.keys()), {"fl", "fr", "hl", "hr"})
        groups = torch.randn(2, 9, 64, 80, requires_grad=True)
        transformed = model.apply_leg_pretransformers(groups)
        self.assertEqual(tuple(transformed.shape), (2, 9, 64, 80))
        for group in (0, 3, 6, 7, 8):
            torch.testing.assert_close(transformed[:, group], groups[:, group])
        for group in (1, 2, 4, 5):
            self.assertFalse(torch.equal(transformed[:, group], groups[:, group]))
        transformed.square().mean().backward()
        for name, block in model.leg_pre_transformers.items():
            for parameter_name, parameter in block.named_parameters():
                self.assertIsNotNone(parameter.grad, f"{name}.{parameter_name} received no gradient")
                self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertFalse(torch.equal(
            next(model.leg_pre_transformers["fl"].parameters()),
            next(model.leg_pre_transformers["fr"].parameters()),
        ))


if __name__ == "__main__":
    unittest.main()
