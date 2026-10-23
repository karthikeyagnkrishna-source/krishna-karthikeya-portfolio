import importlib.util
import unittest
from fixtures import fixture,torch

class ModelChecks(unittest.TestCase):
    def model(self):
        self.assertIsNotNone(importlib.util.find_spec('joint_models'),'Nine warm-started specialists missing')
        from joint_models import JointSpecialists
        parent,inp,b,s=fixture()
        return JointSpecialists(parent.seed).eval(),parent,inp,b,s

    def test_initial_outputs_storage_and_no_target_inputs(self):
        model,parent,inp,b,s=self.model();out=model(inp,s);seed=parent(inp.base,s,'seed')
        torch.testing.assert_close(out.leg,seed['leg'],atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(out.body,seed['body'],atol=0,rtol=0)
        torch.testing.assert_close(out.force,inp.base['force'],atol=0,rtol=0)
        self.assertEqual(out.hidden.shape,(2,3,4,9,256))
        self.assertNotEqual(model.heads['Q0'].projection[1].weight.data_ptr(),model.heads['Q1'].projection[1].weight.data_ptr())
        self.assertNotEqual(model.heads['Q0'].projection[1].weight.data_ptr(),model.seed.legs.projection[1].weight.data_ptr())
        self.assertFalse(any(k.startswith('target') for k in inp.base))

    def test_padding_and_nontrivial_prefix_invariance(self):
        model,_,inp,_,s=self.model()
        for head in model.heads.values():torch.nn.init.normal_(head.command_adapter.weight,std=.04)
        a=model(inp,s)
        inp.commands[:,:,2:]+=100
        b=model(inp,s)
        torch.testing.assert_close(a.leg[:,0],b.leg[:,0],atol=1e-5,rtol=1e-5)
        self.assertGreater(float((a.leg[:,-1]-b.leg[:,-1]).abs().max()),1e-7)
        inp.command_valid.zero_();inp.history_valid.zero_()
        self.assertTrue(torch.isfinite(model(inp,s).leg).all())

    def test_only_selected_head_changes_and_inactive_eval(self):
        model,_,inp,_,s=self.model();model.configure('T1');model.train()
        a=model(inp,s);a.leg[...,7].square().mean().backward()
        self.assertTrue(all(p.grad is None for p in model.seed.parameters()))
        for name,head in model.heads.items():
            self.assertEqual(head.training,name=='T1')
            if name!='T1':self.assertTrue(all(p.grad is None for p in head.parameters()))
        opt=torch.optim.AdamW(model.active_parameters(),lr=.01)
        opt.step();model.eval();after=model(inp,s)
        torch.testing.assert_close(a.body,after.body,atol=0,rtol=0)
        self.assertGreater(float((a.leg[...,7]-after.leg[...,7]).abs().max()),0)
        torch.testing.assert_close(a.leg[...,:6],after.leg[...,:6],atol=0,rtol=0)

if __name__=='__main__':unittest.main()
