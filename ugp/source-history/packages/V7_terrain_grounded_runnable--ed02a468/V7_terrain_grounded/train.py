"""Train on actual synchronized RGB-D/robot logs converted to the NPZ contract.

python train.py --train /path/real_train.npz --val /path/real_val.npz --out ckpts
"""
from __future__ import annotations
import argparse,random,time,json,os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from v7_grounded import V7Config,V7GroundedWorldModel
from v7_grounded.data import V7NPZDataset
from v7_grounded.losses import grounded_loss


def put(d,device):return {k:v.to(device,non_blocking=True) for k,v in d.items()}

def run_epoch(model,loader,optimizer,scaler,device,train,amp):
    model.train(train);tot=0.;n=0
    for x,y in loader:
        x=put(x,device);y=put(y,device)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type,enabled=amp,dtype=torch.float16 if device.type=='cuda' else torch.bfloat16):
                out=model(x);loss,parts=grounded_loss(out,y)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(optimizer);scaler.update()
        bs=x['rgbd'].shape[0];tot+=float(loss.detach().cpu())*bs;n+=bs
    return tot/max(n,1)

def main():
    p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--val',required=True)
    p.add_argument('--out',default='checkpoints');p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch',type=int,default=2);p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--num-workers',type=int,default=0);p.add_argument('--resume',default=None)
    p.add_argument('--hidden',type=int,default=320);p.add_argument('--heads',type=int,default=10)
    p.add_argument('--layers',type=int,default=8);p.add_argument('--horizons',type=int,default=10)
    p.add_argument('--seed',type=int,default=2026);p.add_argument('--cpu',action='store_true')
    args=p.parse_args()
    if Path(args.train).resolve()==Path(args.val).resolve():raise ValueError('Train and val cannot be the same file')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    train=V7NPZDataset(args.train);val=V7NPZDataset(args.val)
    xx,_=train[0];vx,_=val[0]
    if xx['rgbd'].shape[1:]!=vx['rgbd'].shape[1:]:raise ValueError('Train/val grid shapes differ')
    cfg=V7Config(proprio_dim=xx['proprio'].shape[-1],action_dim=xx['actions'].shape[-1],
                 feature_dim=xx['patch_features'].shape[-1] if 'patch_features' in xx else 0,
                 hidden=args.hidden,heads=args.heads,spatial_layers=args.layers,horizons=args.horizons)
    if xx['actions'].shape[0]!=cfg.horizons:raise ValueError('Horizon count mismatch with actions')
    device=torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    model=V7GroundedWorldModel(cfg).to(device)
    optim=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    lr_sched=torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=max(args.epochs,1),eta_min=args.lr*.1)
    start=0;best=float('inf');out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if args.resume:
        ck=torch.load(args.resume,map_location=device,weights_only=False)
        if ck['cfg']!=vars(cfg):raise ValueError('Checkpoint configuration differs')
        model.load_state_dict(ck['model']);optim.load_state_dict(ck['optim'])
        scaler.load_state_dict(ck['scaler']);lr_sched.load_state_dict(ck['sched']);start=ck['epoch']+1;best=ck['best']
    train_loader=DataLoader(train,batch_size=args.batch,shuffle=True,num_workers=args.num_workers,pin_memory=device.type=='cuda')
    val_loader=DataLoader(val,batch_size=args.batch,shuffle=False,num_workers=args.num_workers,pin_memory=device.type=='cuda')
    for epoch in range(start,args.epochs):
        then=time.perf_counter();tr=run_epoch(model,train_loader,optim,scaler,device,True,device.type=='cuda')
        va=run_epoch(model,val_loader,None,scaler,device,False,device.type=='cuda');lr_sched.step()
        isbest=va<best;best=min(best,va)
        ck=dict(model=model.state_dict(),optim=optim.state_dict(),scaler=scaler.state_dict(),
                sched=lr_sched.state_dict(),cfg=vars(cfg),epoch=epoch,best=best)
        temp=out/'last.pt.part';torch.save(ck,temp);os.replace(temp,out/'last.pt')
        if isbest:
            temp=out/'best.pt.part';torch.save(ck,temp);os.replace(temp,out/'best.pt')
        print(json.dumps(dict(epoch=epoch+1,train_loss=round(tr,5),val_loss=round(va,5),
                              best_val=round(best,5),seconds=round(time.perf_counter()-then,1))),flush=True)
if __name__=='__main__':main()
