import unittest

import torch
from torch import nn

from tartanground.evaluate import normalized_force_score
from tartanground.rollout import teacher_forcing_probability
from tartanground.selection import choose_best_candidate
from train import configure_trainable_modules, validation_metric_value


class NormalizedForceScoreTests(unittest.TestCase):
    def test_score_is_mean_absolute_error_in_training_standard_deviations(self):
        output = {
            "v7f": {
                "force": torch.tensor(
                    [[[[2.0, 4.0, 8.0]], [[4.0, 8.0, 16.0]]]],
                    dtype=torch.float32,
                )
            }
        }
        batch = {
            "target_force_seq": torch.zeros(1, 2, 1, 3),
            "force_std": torch.tensor([[2.0, 4.0, 8.0]]),
        }

        score = normalized_force_score(output, batch)

        self.assertAlmostEqual(float(score), 1.5, places=6)


class TrainableModuleTests(unittest.TestCase):
    def test_only_requested_module_prefix_remains_trainable(self):
        model = nn.Module()
        model.world = nn.Linear(2, 2)
        model.v7f = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))

        selected = configure_trainable_modules(model, ("v7f",))

        self.assertTrue(selected)
        self.assertTrue(all(name.startswith("v7f.") for name in selected))
        self.assertTrue(all(parameter.requires_grad for name, parameter in model.named_parameters() if name.startswith("v7f.")))
        self.assertTrue(all(not parameter.requires_grad for name, parameter in model.named_parameters() if name.startswith("world.")))

    def test_unknown_prefix_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "matched no parameters"):
            configure_trainable_modules(nn.Linear(2, 2), ("v7f",))

    def test_empty_prefixes_preserve_full_training(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))

        selected = configure_trainable_modules(model, ())

        self.assertEqual(set(selected), {name for name, _ in model.named_parameters()})
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))


class ValidationSelectionTests(unittest.TestCase):
    def test_configured_force_metric_is_selected(self):
        metrics = {"total": 9.0, "force_score": 0.25}
        self.assertEqual(validation_metric_value(metrics, "force_score"), 0.25)

    def test_missing_metric_is_rejected(self):
        with self.assertRaisesRegex(KeyError, "selection metric"):
            validation_metric_value({"total": 1.0}, "force_score")

    def test_candidate_selection_uses_lowest_development_force_score(self):
        candidates = [
            {"path": "early.pt", "force_score": 0.31},
            {"path": "late.pt", "force_score": 0.24},
        ]
        self.assertEqual(choose_best_candidate(candidates)["path"], "late.pt")


class TeacherForcingProbabilityTests(unittest.TestCase):
    def test_linear_schedule_reaches_both_endpoints(self):
        self.assertEqual(teacher_forcing_probability(0, 7, 0.5, 0.0), 0.5)
        self.assertEqual(teacher_forcing_probability(6, 7, 0.5, 0.0), 0.0)
        self.assertAlmostEqual(teacher_forcing_probability(3, 7, 0.5, 0.0), 0.25)

    def test_probability_inputs_are_validated(self):
        with self.assertRaises(ValueError):
            teacher_forcing_probability(0, 7, 1.1, 0.0)


if __name__ == "__main__":
    unittest.main()
