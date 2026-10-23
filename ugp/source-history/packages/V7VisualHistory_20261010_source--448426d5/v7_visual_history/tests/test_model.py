"""Behavior checks with real V7 class definitions; no substitute Transformer."""
import ast
import copy
import importlib
import unittest
from dataclasses import replace
from pathlib import Path
try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ModuleNotFoundError:
    torch = None

ROOT = Path(__file__).resolve().parents[2]
if torch is not None:
    torch.set_num_threads(2)


def fixture(steps=50, horizons=1):
    source = ROOT/'outputs/architecture_atlas_20261009/source_snapshots/v7_rtx_notebook.py'
    names = {'CausalHistoryEncoder', 'PerceiverResampler', 'FlowTemporalEncoder',
             'QueryFusionBlock', 'FullFourBranchTerrainEncoder', 'V7FullDynamics'}
    tree = ast.parse(source.read_text(encoding='utf-8'))
    tree.body = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
    env = dict(torch=torch, nn=nn, F=F, VIS_DIM=256, LATENTS_PER_BRANCH=16, N_REGIMES=4, PI_PHYS_DIM=47)
    exec(compile(tree, str(source), 'exec'), env)
    torch.manual_seed(712)
    base = env['V7FullDynamics'](steps, horizons, 8, 10, 6).eval().requires_grad_(False)
    from v7_visual_history.data import HistoryBatch
    history = HistoryBatch(torch.randn(1, 6, 64, 8), torch.randn(1, 6, 64, 10), torch.randn(1, 6, 64, 6),
        torch.tensor([[-.5, -.4, -.3, -.2, -.1, 0]], dtype=torch.float64),
        -torch.arange(steps-1, -1, -1, dtype=torch.float64)[None]*.01,
        torch.ones(1, 6, dtype=torch.bool), torch.arange(6)[None])
    sample = dict(p_hist=torch.randn(1, 4, steps, 12), b_hist=torch.randn(1, steps, 12),
        u_hist=torch.randn(1, steps, 6), l_hist=torch.randn(1, 4, steps, 3), e_hist=torch.randn(1, 4, steps, 3),
        dino=history.dino[:, -1], conv=history.conv[:, -1], geom=history.geom[:, -1], flow=torch.randn(1, 4, 64, 6),
        foot_query_kin=torch.randn(1, 4, 6), future_control_features=torch.randn(1, horizons, 36),
        horizon_values=(torch.tensor([.1]) if horizons == 1 else torch.arange(2, horizons+2).float()/10)[None, :, None],
        physics_features=torch.randn(1, horizons, 47), visual_history=history)
    return base, sample


def implementation(test):
    try:
        return importlib.import_module('v7_visual_history.model')
    except ModuleNotFoundError:
        test.fail('V7 visual-history attention is not implemented')


class VisualHistoryModelTests(unittest.TestCase):
    def test_robot_tokens_remain_causal_before_memory_fusion(self):
        m = implementation(self)
        base, s = fixture()
        robot = torch.cat([s['b_hist'], s['u_hist']], dim=-1)
        a = m.history_sequence(base.body_encoder, robot)
        changed = robot.clone(); changed[:, 30:] += torch.randn_like(changed[:, 30:])*10
        b = m.history_sequence(base.body_encoder, changed)
        torch.testing.assert_close(a[:, :30], b[:, :30], atol=0, rtol=0)

    def test_zero_warm_start_matches_original_short_and_long(self):
        m = implementation(self)
        for steps, horizons in [(50, 1), (400, 9)]:
            base, s = fixture(steps, horizons)
            original = base(*(s[k] for k in m.V7_ARGS))
            model = m.V7VisualHistory(base).eval()
            actual = model.forward_sample(s)
            self.assertEqual(set(original), set(actual))
            for k in original:
                torch.testing.assert_close(actual[k], original[k], atol=0, rtol=0)
            features, diagnostic = model.features(s, return_attention=True)
            self.assertEqual(tuple(features.shape), (1, horizons, 4, 1285))
            self.assertEqual(diagnostic['memory_tokens'], 304+5*steps)
            self.assertEqual(tuple(diagnostic['attention'].shape), (1, 5, 304+5*steps))

    def test_past_images_affect_predictions_after_adapter_is_open(self):
        m = implementation(self)
        base, s = fixture()
        model = m.V7VisualHistory(base).eval()
        with torch.no_grad():
            model.adapter.output.weight.copy_(torch.eye(256)*.1)
        original = model.forward_sample(s)['residual_p']
        old = s['visual_history']
        changed = old.dino.clone(); changed[:, :-1] += torch.randn_like(changed[:, :-1])*3
        s = dict(s, visual_history=replace(old, dino=changed))
        actual = model.forward_sample(s)['residual_p']
        self.assertGreater(float((actual-original).abs().max()), 1e-7)

    def test_padded_past_images_cannot_affect_predictions(self):
        m = implementation(self)
        base, s = fixture()
        model = m.V7VisualHistory(base).eval()
        with torch.no_grad():
            model.adapter.output.weight.copy_(torch.eye(256)*.1)
        old = s['visual_history']; mask = old.valid.clone(); mask[:, :3] = False
        h = replace(old, valid=mask)
        a = model.forward_sample(dict(s, visual_history=h))
        changed = h.dino.clone(); changed[:, :3] = float('nan')
        b = model.forward_sample(dict(s, visual_history=replace(h, dino=changed)))
        for k in a:
            torch.testing.assert_close(a[k], b[k], atol=0, rtol=0)

    def test_future_images_and_wrong_robot_clock_are_rejected(self):
        m = implementation(self)
        base, s = fixture()
        model = m.V7VisualHistory(base).eval(); old = s['visual_history']
        times = old.visual_times.clone(); times[:, -1] = .01
        with self.assertRaisesRegex(ValueError, 'Future'):
            model.forward_sample(dict(s, visual_history=replace(old, visual_times=times)))
        with self.assertRaisesRegex(ValueError, 'anchor'):
            model.forward_sample(dict(s, visual_history=replace(old, robot_times=old.robot_times-.01)))

    def test_only_adapter_receives_gradients(self):
        m = implementation(self)
        base, s = fixture(); model = m.V7VisualHistory(base).train()
        self.assertFalse(model.base.training)
        out = model.forward_sample(s)
        out['residual_p'].square().mean().backward()
        self.assertGreater(float(model.adapter.output.weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in model.base.parameters()))
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            model.adapter.output.weight.copy_(torch.eye(256)*.1)
        model.forward_sample(s)['residual_p'].square().mean().backward()
        self.assertGreater(float(model.adapter.time[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.adapter.leg_project.weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in model.base.parameters()))

    def test_future_targets_are_not_inputs_and_missing_history_fails(self):
        m = implementation(self)
        base, s = fixture(); model = m.V7VisualHistory(base).eval()
        a = model.forward_sample(s)
        s['target_residual_p'] = torch.full((1, 1, 4, 12), float('nan'))
        b = model.forward_sample(s)
        for k in a:
            torch.testing.assert_close(a[k], b[k], atol=0, rtol=0)
        del s['visual_history']
        with self.assertRaisesRegex(ValueError, 'history'):
            model.forward_sample(s)


if __name__ == '__main__':
    unittest.main()
