import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'v7_joint_specialists'),str(ROOT/'v7_joint_specialists/.verification_deps'),str(ROOT/'v7feedback'),str(ROOT/'v7feedback/.verification_deps'),str(ROOT/'v7legs_body')]
import torch
torch.set_num_threads(2)
def fixture(h=3):
    from test_feedback import FeedbackChecks
    FeedbackChecks.setUpClass()
    parent,b,s=FeedbackChecks().fixture(h)
    from contracts import InputBatch
    base={k:v for k,v in b.items() if not k.startswith('target_')}
    history=torch.randn(2,4,6,83);commands=torch.randn(2,4,5,55)
    for x,start in [(history,34),(commands,0)]:
        for j in range(3):x[...,start+j*16+6:start+j*16+12]=1
    mask=torch.zeros(2,h,4,5,dtype=torch.bool)
    for i in range(h):mask[:,i,:,:min(i+2,5)]=True
    inp=InputBatch(base,history,torch.ones(2,4,6,dtype=torch.bool),commands,mask)
    s.update(history_mean=torch.zeros(83),history_std=torch.ones(83),commands_mean=torch.zeros(55),commands_std=torch.ones(55),
             loss_leg_std=torch.ones(4,9),loss_force_std=torch.ones(4,3))
    return parent,inp,b,s
