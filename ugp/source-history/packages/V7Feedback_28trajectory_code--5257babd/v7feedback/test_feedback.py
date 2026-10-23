"""Checks for new feedback risks; no vision or foundation retraining."""
import copy
import importlib.util
import sys
import unittest
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / '.verification_deps'), str(HERE.parent / 'v7legs_body')]


class FeedbackChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)
        cls.torch = torch

    def fixture(self, horizons=3):
        self.assertIsNotNone(importlib.util.find_spec('feedback_models'),
                             'Approved recurrent model has not been implemented')
        import numpy as np
        from models import Hybrid, TorchFK
        from feedback_models import FeedbackModel
        t = self.torch
        t.manual_seed(781)
        class Robot:
            leg_prefixes = ['LF', 'RF', 'LH', 'RH']
        robot = Robot()
        robot.chains = [[dict(name=p + '_' + j, type='revolute', rpy=[0, 0, 0],
                            xyz=[0, 0, -.15], axis=a)
                         for j, a in [('HAA', [1, 0, 0]), ('HFE', [0, 1, 0]), ('KFE', [0, 1, 0])]]
                        + [dict(name=p + '_foot', type='fixed', rpy=[0, 0, 0], xyz=[0, 0, -.15], axis=[0, 0, 1])]
                        for p in robot.leg_prefixes]
        fk = TorchFK(robot, lambda rpy: np.eye(3))
        seed = Hybrid(fk)
        for head in seed.legs.outputs:
            t.nn.init.normal_(head[-1].weight, std=.015)
        t.nn.init.normal_(seed.body.output[-1].weight, std=.015)
        model = FeedbackModel(seed).eval()
        batch = dict(x=t.randn(2, horizons, 4, 1285).half(), base_leg=t.randn(2, horizons, 4, 9) * .2,
                     base_body=t.randn(2, horizons, 12) * .2, force=t.randn(2, horizons, 4, 3) * 20,
                     contact=t.ones(2, horizons, 4), horizon=t.arange(1, horizons + 1).float() / 10,
                     target_leg=t.randn(2, horizons, 4, 9) * .2, target_body=t.randn(2, horizons, 12) * .2,
                     target_force=t.randn(2, horizons, 4, 3) * 20, target_contact=t.ones(2, horizons, 4))
        scales = {}
        for name, width, scale in [('leg', 9, .2), ('body', 12, .5), ('force', 3, 100),
                                   ('r', 3, 1), ('fv', 3, 1), ('stance', 3, 1), ('target_force', 3, 100)]:
            scales[name + '_mean'] = t.zeros(width)
            scales[name + '_std'] = t.full((width,), float(scale))
        return model, batch, scales

    def test_zero_new_paths_preserve_seed_in_every_round(self):
        model, batch, scales = self.fixture()
        expected = model.seed(batch, scales, 'hybrid')
        out = model(batch, scales)
        for step in out['rounds']:
            self.torch.testing.assert_close(step['leg'], expected['leg'], atol=2e-6, rtol=1e-5)
            self.torch.testing.assert_close(step['body'], expected['body'], atol=2e-6, rtol=1e-5)
            self.torch.testing.assert_close(step['force'], batch['force'], atol=0, rtol=0)

    def test_future_labels_never_change_predictions(self):
        model, batch, scales = self.fixture()
        first = model(batch, scales)
        changed = {k: v + 1234 if k.startswith('target_') else v for k, v in batch.items()}
        second = model(changed, scales)
        for a, b in zip(first['rounds'], second['rounds']):
            for key in ('leg', 'body', 'force'):
                self.torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)

    def test_force_update_reaches_next_round_legs_and_body(self):
        model, batch, scales = self.fixture()
        self.torch.nn.init.constant_(model.force_refiner.output[-1].bias, .3)
        out = model(batch, scales)
        self.assertGreater(float((out['rounds'][1]['leg'] - out['rounds'][0]['leg']).abs().max()), 1e-7)
        self.assertGreater(float((out['rounds'][1]['body'] - out['rounds'][0]['body']).abs().max()), 1e-7)

    def test_force_gradient_reaches_reused_transformers_but_not_seed(self):
        model, batch, scales = self.fixture()
        model.configure('feedback')
        self.torch.nn.init.normal_(model.force_refiner.output[-1].weight, std=.03)
        out = model(batch, scales)
        loss = (out['force'] - batch['target_force']).square().mean()
        loss.backward()
        for module in (model.core.legs.temporal, model.core.body.temporal):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None), 0)
        self.assertTrue(all(p.grad is None for p in model.seed.parameters()))

    def test_zero_force_and_all_swing_loss_is_finite(self):
        model, batch, scales = self.fixture(horizons=1)
        from feedback_models import feedback_objective
        batch['force'].zero_()
        batch['target_force'].zero_()
        batch['target_contact'].zero_()
        out = model(batch, scales)
        loss, selection, _ = feedback_objective(out, batch, scales, model.fk, 'feedback', True, 1.)
        self.assertTrue(bool(self.torch.isfinite(loss)))
        self.assertTrue(bool(self.torch.isfinite(selection)))

    def test_force_export_uses_refined_prediction(self):
        self.assertIsNotNone(importlib.util.find_spec('evaluate_feedback'), 'Refined-force exporter is missing')
        from evaluate_feedback import export_feedback_case
        model, batch, scales = self.fixture(horizons=1)
        batch = {k: (v[:1] if k != 'horizon' else v) for k, v in batch.items()}
        batch.update(anchor=self.torch.tensor([10.], dtype=self.torch.float64),
                     target_time=self.torch.tensor([[10.1]], dtype=self.torch.float64))
        out = model(batch, scales)
        out['force'] = self.torch.full_like(batch['force'], 42.)
        out['target_r'] = model.fk(batch['target_leg'][..., :3])[0]
        rows = export_feedback_case({}, 'round3', batch, out, [.1])
        self.assertEqual(rows['force'][0]['predicted_Fx'], 42.)
        self.assertEqual(rows['force'][0]['predicted_Fz'], 42.)

    def test_target_alignment_rejects_shifted_time(self):
        self.assertIsNotNone(importlib.util.find_spec('feedback_data'), 'Force target alignment is missing')
        import numpy as np
        from feedback_data import aligned_force
        time = np.array([0., .1, .2])
        physics = np.zeros((3, 4, 12), dtype=np.float32)
        physics[2, :, 9:12] = 17.
        gt = np.full((1, 1, 4, 3), 17., dtype=np.float32)
        self.assertTrue(np.array_equal(aligned_force(np.array([[.2]]), time, physics, gt), gt))
        with self.assertRaises(ValueError):
            aligned_force(np.array([[.21]]), time, physics, gt)

    def test_reused_code_change_alters_provenance(self):
        self.assertIsNotNone(importlib.util.find_spec('provenance'), 'Reused-code provenance is missing')
        from provenance import implementation_hashes
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / 'new').mkdir()
            (p / 'reused').mkdir()
            (p / 'new/feedback.py').write_text('value=1')
            reused = p / 'reused/models.py'
            reused.write_text('value=2')
            first = implementation_hashes(p / 'new', p / 'reused')
            reused.write_text('value=3')
            self.assertNotEqual(first, implementation_hashes(p / 'new', p / 'reused'))

    def test_changed_evaluation_input_invalidates_saved_case(self):
        self.assertIsNotNone(importlib.util.find_spec('provenance'), 'Evaluation-input provenance is missing')
        from provenance import input_digest
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'heldout_physics.npz'
            p.write_bytes(b'original observations')
            first = input_digest([p])
            p.write_bytes(b'changed observations')
            self.assertNotEqual(first, input_digest([p]))

    def test_json_manifest_tuple_roundtrip_preserves_identity(self):
        import provenance
        self.assertTrue(hasattr(provenance, 'canonical_record'), 'Manifest JSON normalization is missing')
        source = {'signature': {'runs': [('ForestEnv', 'P2000', 28)]}}
        stored = {'signature': {'runs': [['ForestEnv', 'P2000', 28]]}}
        self.assertEqual(provenance.canonical_record(source), stored)
        changed = {'signature': {'runs': [('ForestEnv', 'P2001', 28)]}}
        self.assertNotEqual(provenance.canonical_record(changed), stored)


if __name__ == '__main__':
    unittest.main()
