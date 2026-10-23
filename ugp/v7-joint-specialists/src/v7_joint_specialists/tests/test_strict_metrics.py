import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'v7_joint_specialists'), str(ROOT / 'v7feedback/.verification_deps')]

class StrictChecks(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('strict_metrics'), 'Strict physical scoring is missing')
        from contracts import ToleranceConfig
        from strict_metrics import score_scalar, score_vector, strict_summary
        return ToleranceConfig(), score_scalar, score_vector, strict_summary

    def test_strict_boundary_and_tiny_nonzero(self):
        cfg, scalar, _, _ = self.api()
        import numpy as np
        s = scalar([100,100,0,.1,0,1e-7], [101.99,102,.04999,.10201,.05,1.03e-7], 'torque', cfg)
        np.testing.assert_array_equal(s['pass'], [True,False,True,False,False,False])
        self.assertFalse(s['zero'][-1])

    def test_angles_contact_nonfinite(self):
        cfg, scalar, _, _ = self.api()
        import numpy as np
        s = scalar([np.pi/2,0], [np.pi/2+.01,.001], 'angle', cfg)
        self.assertAlmostEqual(s['actual'][0],90)
        self.assertFalse(s['pass'][1])
        self.assertFalse(scalar([0],[.02],'contact',cfg)['pass'][0])
        with self.assertRaises(ValueError): scalar([1],[np.nan],'torque',cfg)

    def test_vector_and_summary_keep_zero_failures(self):
        cfg, scalar, vector, summary = self.api()
        v = vector([[100,0,0],[0,0,0]], [[100,1.1,0],[1,0,0]], 'force', cfg)
        self.assertTrue(v['pass'][0])
        self.assertFalse(scalar([0],[1.1],'force',cfg)['pass'][0])
        self.assertFalse(v['pass'][1])
        s = summary(scalar([100,0],[101,.06],'torque',cfg))
        self.assertEqual(s['failures'],1)
        self.assertEqual(s['zero_failures'],1)
        self.assertEqual(s['mean_percent'],1)

    def test_loss_finite_gradients_near_zero_and_empty(self):
        self.api()
        import torch
        from joint_objectives import specialist_loss
        from contracts import ToleranceConfig
        y = torch.tensor([0,1e-7,100.],dtype=torch.float32)
        p = torch.tensor([.1,1.1e-7,105.],requires_grad=True)
        loss = specialist_loss(p,y,'torque',torch.ones(3),ToleranceConfig(),False)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(p.grad).all())
        empty = p[:0]
        self.assertEqual(float(specialist_loss(empty,y[:0],'torque',torch.ones(1),ToleranceConfig(),False)),0)

if __name__ == '__main__': unittest.main()
