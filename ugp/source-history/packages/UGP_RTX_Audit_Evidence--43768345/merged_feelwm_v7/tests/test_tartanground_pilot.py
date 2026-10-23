import unittest

import numpy as np
import torch

from tartanground.pilot import (
    FrozenVisualResamplers,
    build_body_action,
    build_leg_action,
    contact_phase,
)


class TartanGroundPilotTest(unittest.TestCase):
    def test_body_and_leg_actions_use_commands(self):
        body = build_body_action(np.array([2.0, -1.0, 9.0, 4.0, 5.0, 0.5], np.float32))
        np.testing.assert_allclose(body, [0.2, -0.1, 0.05, 0.1])

        q0 = np.array([[0.0, 1.0, 2.0]] * 4, np.float32)
        q1 = q0 + np.array([[9.0, 0.2, -0.3]] * 4, np.float32)
        tau = np.array([[8.0, 4.0, -5.0]] * 4, np.float32)
        got = build_leg_action(q0, q1, tau)
        np.testing.assert_allclose(got[0], [0.2, -0.3, 4.0, -5.0], atol=1e-6)

    def test_contact_phase_labels_four_transitions(self):
        self.assertEqual(contact_phase(False, False), 0)
        self.assertEqual(contact_phase(False, True), 1)
        self.assertEqual(contact_phase(True, True), 2)
        self.assertEqual(contact_phase(True, False), 3)

    def test_resamplers_export_fixed_branch_order_without_modality_embedding(self):
        torch.manual_seed(7)
        model = FrozenVisualResamplers(dino_dim=5, conv_dim=6, geom_dim=7).eval()
        dino = torch.randn(2, 64, 5)
        conv = torch.randn(2, 64, 6)
        geom = torch.randn(2, 64, 7)
        flow = torch.randn(2, 4, 64, 6)
        with torch.inference_mode():
            out = model(dino, conv, geom, flow)
            expected = torch.cat([
                model.dino(dino),
                model.conv(conv),
                model.geom(geom),
                model.flow(model.flow_temporal(flow)),
            ], dim=1)
        self.assertEqual(tuple(out.shape), (2, 64, 256))
        torch.testing.assert_close(out, expected)


if __name__ == "__main__":
    unittest.main()
