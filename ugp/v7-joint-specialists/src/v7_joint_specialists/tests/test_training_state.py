import importlib.util
import unittest
import tempfile
from pathlib import Path
from fixtures import torch

class StateChecks(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('train_joint_specialists'),'Resumable specialist trainer missing')
        from train_joint_specialists import save_training_state,restore_training_state,balanced_indices,selection_key
        return save_training_state,restore_training_state,balanced_indices,selection_key

    def test_resume_next_step_matches_and_rejects_changed_signature(self):
        save,restore,_,_=self.api();torch.manual_seed(19)
        model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Dropout(.2),torch.nn.Linear(4,1))
        opt=torch.optim.AdamW(model.parameters(),lr=.01);x=torch.ones(5,3)
        def step(m,o):o.zero_grad();m(x).square().mean().backward();o.step()
        step(model,opt)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'latest.pt';sig={'code':'a','scales':'b','inputs':[[1,2]]}
            save(path,model,opt,sig,{'cursor':1})
            step(model,opt);expected={k:v.clone() for k,v in model.state_dict().items()}
            other=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Dropout(.2),torch.nn.Linear(4,1));o=torch.optim.AdamW(other.parameters(),lr=.01)
            self.assertEqual(restore(path,other,o,sig)['cursor'],1);step(other,o)
            for k,v in expected.items():torch.testing.assert_close(other.state_dict()[k],v,atol=0,rtol=0)
            with self.assertRaises(ValueError):restore(path,other,o,dict(sig,code='changed'))

    def test_balanced_optimization_only_and_zero_failure_selection(self):
        _,_,balance,key=self.api()
        import numpy as np
        class Dataset:
            trajectories=np.array([0,0,1,1]);transitions=np.array([0,1,0,1],bool)
            def __len__(self):return 4
        order=balance(Dataset(),torch.Generator().manual_seed(2),np.array([0,1,0,9]))
        self.assertTrue(set(order.tolist())<=set(range(4)))
        self.assertEqual(key({'pass':np.array([True,False]),'ratio':np.array([.1,2]),'error':np.array([1,.06])})[0],1)

if __name__=='__main__':unittest.main()
