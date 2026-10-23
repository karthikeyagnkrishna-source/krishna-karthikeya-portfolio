import importlib.util
import unittest
from fixtures import fixture,torch

class ForceChecks(unittest.TestCase):
    def test_joint_to_force_transition_clears_predecessor_gradients(self):
        from force_models import ForceContactSpecialist
        from joint_models import JointSpecialists
        parent,inp,b,s=fixture();j=JointSpecialists(parent.seed);j.configure('T0')
        out=j(inp,s);out.leg[...,6].square().mean().backward()
        torch.optim.AdamW(j.active_parameters(),lr=.001).step()
        self.assertTrue(any(p.grad is not None for p in j.parameters()))
        model=ForceContactSpecialist(j,parent).eval();model.configure_force()
        self.assertTrue(all(p.grad is None for p in model.joints.parameters()))
        model(inp,s).force.square().mean().backward()
        self.assertTrue(all(p.grad is None for p in model.joints.parameters()))

    def test_force_and_contact_do_not_read_later_horizon_commands(self):
        from force_models import ForceContactSpecialist
        from joint_models import JointSpecialists
        parent,inp,b,s=fixture();j=JointSpecialists(parent.seed)
        for head in j.heads.values():torch.nn.init.normal_(head.command_adapter.weight,std=.03)
        model=ForceContactSpecialist(j,parent).eval()
        torch.nn.init.normal_(model.refiner.output[-1].weight,std=.03)
        torch.nn.init.normal_(model.extension.contact.weight,std=.03)
        inp.base['contact'].fill_(.5)
        a=model(inp,s);inp.commands[:,:,2:]+=100;b=model(inp,s)
        torch.testing.assert_close(a.force[:,0],b.force[:,0],atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(a.contact[:,0],b.contact[:,0],atol=1e-6,rtol=1e-5)
        self.assertGreater(float((a.force[:,-1]-b.force[:,-1]).abs().max()),1e-5)

    def test_uncapped_correction_preserves_joints_body_and_frozen_gradients(self):
        self.assertIsNotNone(importlib.util.find_spec('force_models'),'Fixed-state force/contact extension missing')
        from force_models import ForceContactSpecialist
        from joint_models import JointSpecialists
        parent,inp,b,s=fixture();j=JointSpecialists(parent.seed).eval()
        model=ForceContactSpecialist(j,parent).eval();before=model(inp,s)
        torch.testing.assert_close(before.force,parent(inp.base,s,'one_pass')['force'],atol=0,rtol=0)
        model.configure_force();torch.nn.init.constant_(model.refiner.output[-1].bias,3.)
        after=model(inp,s)
        torch.testing.assert_close(before.leg,after.leg,atol=0,rtol=0);torch.testing.assert_close(before.body,after.body,atol=0,rtol=0)
        self.assertTrue(torch.all(after.force-before.force>2*s['target_force_std']))
        after.force.square().mean().backward()
        self.assertTrue(all(p.grad is None for p in model.joints.parameters()))
        self.assertTrue(all(p.grad is None for p in model.force_prior.parameters()))
        self.assertTrue(torch.isfinite(after.contact).all())

if __name__=='__main__':unittest.main()
