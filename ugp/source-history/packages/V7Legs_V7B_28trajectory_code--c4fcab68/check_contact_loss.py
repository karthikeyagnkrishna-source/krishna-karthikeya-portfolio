"""Regression: swinging feet must not contribute to the planted-foot constraint."""
import torch
from models import objective

leg = torch.zeros(1, 1, 4, 9)
body = torch.zeros(1, 1, 12)
fv = torch.tensor([[[[0., 0., 0.], [10., 10., 10.], [20., 20., 20.], [30., 30., 30.]]]], requires_grad=True)
out = dict(leg=leg, body=body, r=torch.zeros(1, 1, 4, 3), fv=fv,
           leg_delta=torch.zeros_like(leg), body_delta=torch.zeros_like(body))
batch = dict(target_leg=leg, target_body=body, target_contact=torch.tensor([[[1., 0., 0., 0.]]]))
scales = dict(leg_std=torch.ones(9), body_std=torch.ones(12), r_std=torch.ones(3), stance_std=torch.ones(3))
fk = lambda q: (torch.zeros_like(q), None)
_, _, parts = objective(out, batch, scales, fk, 'joint', True)
assert float(parts['stance']) == 0., 'Swing feet incorrectly contribute to planted-foot penalty'
parts['stance'].backward()
assert torch.count_nonzero(fv.grad[..., 1:, :]) == 0
batch['target_contact'].zero_()
_, _, parts = objective(out, batch, scales, fk, 'joint', True)
assert float(parts['stance']) == 0.
print('CONTACT_MASK_AND_ALL_SWING_REGRESSION_PASSED')
