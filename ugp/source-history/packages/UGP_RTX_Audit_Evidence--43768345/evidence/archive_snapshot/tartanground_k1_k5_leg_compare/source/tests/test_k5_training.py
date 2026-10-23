import unittest

import torch

from tartanground.rollout import teacher_forcing_mask


class TeacherForcingScheduleTests(unittest.TestCase):
    def test_schedule_is_deterministic_and_balanced_over_even_examples(self):
        first = teacher_forcing_mask(8, 5, sample_offset=0, seed=781)
        again = teacher_forcing_mask(8, 5, sample_offset=0, seed=781)
        torch.testing.assert_close(first, again)
        self.assertEqual(int(first[:, 1].sum()), 4)
        self.assertTrue(torch.equal(first[:, 1], first[:, 4]))
        self.assertFalse(bool(first[:, 0].any()))

    def test_adjacent_microbatches_split_even_when_batch_size_is_one(self):
        first = teacher_forcing_mask(1, 5, sample_offset=0, seed=781)
        second = teacher_forcing_mask(1, 5, sample_offset=1, seed=781)
        self.assertNotEqual(bool(first[0, 1]), bool(second[0, 1]))

    def test_k1_has_no_teacher_forcing(self):
        mask = teacher_forcing_mask(3, 1, sample_offset=0, seed=781)
        self.assertEqual(tuple(mask.shape), (3, 1))
        self.assertFalse(bool(mask.any()))

    def test_k5_autoregressive_outputs_ignore_future_measured_state_fields(self):
        from config import ModelConfig
        from model import MergedFeelWMV7
        from smoke_test import sample

        torch.manual_seed(17)
        cfg = ModelConfig(
            world_d_model=80, world_heads=10, world_layers=1, world_ffn_ratio=2,
            query_layers=1, hist_layers=1, v7f_layers=1,
        )
        model = MergedFeelWMV7(cfg).eval()
        batch = {key: value.cpu() for key, value in sample(B=1, K=5, T=5).items()}
        changed = {key: value.clone() for key, value in batch.items()}
        for key in (
            "physics_seq", "jointbody_baseline_seq", "tau_baseline_seq", "force_baseline_seq",
            "target_z_seq", "target_contact_seq",
            "target_jointbody_seq", "target_tau_seq", "target_force_seq", "target_phase_seq",
            "err_seq",
        ):
            changed[key][:, 1:] = changed[key][:, 1:] + 123.0
        changed["target_load_class_seq"][:, 1:] = (changed["target_load_class_seq"][:, 1:] + 1) % 4
        changed["leg_query_features_seq"][:, 1:, :, :12] += 123.0
        changed["leg_query_features_seq"][:, 1:, :, 18:21] += 123.0
        with torch.no_grad():
            first = model(batch)
            second = model(changed)
        for horizon in range(5):
            for key in ("z_next", "jointbody_pred", "tau_pred", "base_force", "contact_prob", "regime_prob"):
                torch.testing.assert_close(first["steps"][horizon][key], second["steps"][horizon][key])
        torch.testing.assert_close(first["v7f"]["force"], second["v7f"]["force"])

    def test_autoregressive_feedback_is_refined_v7f_force(self):
        from config import ModelConfig
        from model import MergedFeelWMV7
        from smoke_test import sample

        torch.manual_seed(19)
        cfg = ModelConfig(
            world_d_model=80, world_heads=10, world_layers=1, world_ffn_ratio=2,
            query_layers=1, hist_layers=1, v7f_layers=1,
        )
        model = MergedFeelWMV7(cfg).eval()
        batch = {key: value.cpu() for key, value in sample(B=1, K=3, T=5).items()}
        with torch.no_grad():
            output = model(batch)
        torch.testing.assert_close(output["feedback_force_seq"][:, :-1], output["refined_force_seq"][:, :-1])

    def test_k5_loss_uses_all_five_configured_horizon_lambdas(self):
        from config import ModelConfig, TrainConfig
        from losses import compute_losses
        from smoke_test import sample

        torch.manual_seed(23)
        batch = {key: value.cpu() for key, value in sample(B=1, K=5, T=5).items()}
        batch["target_z_seq"].zero_()
        batch["target_contact_seq"].zero_()
        batch["target_load_class_seq"].zero_()
        batch["target_jointbody_seq"].zero_()
        batch["target_tau_seq"].zero_()
        batch["target_force_seq"].zero_()
        steps = []
        for horizon in range(5):
            value = float(horizon + 1)
            steps.append({
                "dino": torch.full((1, 16, 256), value),
                "conv": torch.full((1, 16, 256), value),
                "geom": torch.full((1, 16, 256), value),
                "flow": torch.full((1, 16, 256), value),
                "contact_logits": torch.full((1, 4), value),
                "regime_logits": torch.full((1, 4, 4), value),
                "jointbody_pred": torch.full((1, 36), value),
                "tau_pred": torch.full((1, 4, 3), value),
                "base_force": torch.full((1, 4, 3), value),
                "force_aux_logmag": torch.full((1, 4), value),
                "force_aux_dir": torch.full((1, 4, 3), value / 10.0),
            })
        spec = {
            "force": torch.stack([torch.full((1, 4, 3), float(k + 1)) for k in range(5)], dim=1),
            "phase_logits": torch.zeros(1, 5, 4, 4),
            "corr_norm": torch.zeros(1, 5, 4, 3),
            "expert_w": torch.full((1, 5, 4, 4), 0.25),
        }
        out = {"steps": steps, "v7f": spec}
        weights = [0.10, 0.15, 0.20, 0.25, 0.30]
        losses = compute_losses(out, batch, ModelConfig(), TrainConfig(horizon_lambdas=tuple(weights)))
        self.assertTrue(all(f"horizon_{k}" in losses for k in range(1, 6)))
        expected = sum(weights[k] * losses[f"horizon_{k + 1}"] for k in range(5))
        expected = expected + losses["temporal_regularizer"]
        expected = expected + losses["impulse_regularizer"]
        torch.testing.assert_close(losses["total"], expected)
        self.assertGreater(float(losses["horizon_5"]), float(losses["horizon_1"]))


if __name__ == "__main__":
    unittest.main()
