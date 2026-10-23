from __future__ import annotations
import argparse
import json
import math
import os
import random
import signal
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from config import load_config, config_to_dict
from model import MergedFeelWMV7
from data import ShardedWorldModelDataset, move_batch_to_device
from losses import compute_losses
from checkpointing import save_checkpoint, load_checkpoint, resolve_resume, prune_step_checkpoints, unwrap_model


STOP_REQUESTED = False


def on_signal(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def is_main(rank):
    return rank == 0


def seed_everything(seed: int, rank: int):
    s = seed + rank * 100003
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def make_optimizer(model, lr: float, wd: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or name.endswith("bias") or "norm" in name.lower() or "embedding" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": wd},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=lr, betas=(0.9, 0.95))


def make_scheduler(optimizer, warmup: int, total_steps: int, min_ratio: float):
    def f(step: int):
        if step < warmup:
            return max(1e-8, step / max(1, warmup))
        progress = (step - warmup) / max(1, total_steps - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return min_ratio + (1.0 - min_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def amp_context(dtype_name: str):
    if dtype_name == "fp32" or not torch.cuda.is_available():
        return nullcontext()
    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def reduce_scalar(x: torch.Tensor, world_size: int):
    y = x.detach().float()
    if world_size > 1:
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
        y /= world_size
    return float(y.cpu())


def validate(model, loader, device, mcfg, tcfg, world_size, max_batches=50):
    model.eval()
    sums = {}
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            with amp_context(tcfg.amp_dtype):
                out = model(batch, history_shift=tcfg.history_shift_samples)
                losses = compute_losses(out, batch, mcfg, tcfg)
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + reduce_scalar(v, world_size)
            count += 1
            if count >= max_batches:
                break
    model.train()
    return {k: v / max(count, 1) for k, v in sums.items()}


def max_steps_reached(global_step: int, max_steps: int | None) -> bool:
    return max_steps is not None and global_step >= max_steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_2x4080.yaml")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    distributed, rank, world_size, local_rank = init_distributed()
    mcfg, tcfg, _ = load_config(args.config)
    seed_everything(tcfg.seed, rank)

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.backends.cuda.matmul.allow_tf32 = tcfg.tf32
        torch.backends.cudnn.allow_tf32 = tcfg.tf32
    else:
        device = torch.device("cpu")

    out_dir = Path(tcfg.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    if is_main(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "resolved_config.json").write_text(json.dumps(config_to_dict(mcfg, tcfg), indent=2))

    train_ds = ShardedWorldModelDataset(tcfg.train_manifest)
    val_ds = ShardedWorldModelDataset(tcfg.val_manifest)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=tcfg.seed) if distributed else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None

    loader_kwargs = dict(
        batch_size=tcfg.batch_size,
        num_workers=tcfg.num_workers,
        pin_memory=tcfg.pin_memory,
        persistent_workers=tcfg.persistent_workers and tcfg.num_workers > 0,
    )
    if tcfg.num_workers > 0:
        loader_kwargs["prefetch_factor"] = tcfg.prefetch_factor
    train_loader = DataLoader(train_ds, shuffle=train_sampler is None, sampler=train_sampler, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, sampler=val_sampler, drop_last=False, **loader_kwargs)

    model = MergedFeelWMV7(mcfg).to(device)
    if tcfg.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=False)

    optimizer = make_optimizer(model, tcfg.lr, tcfg.weight_decay)
    optim_steps_per_epoch = math.ceil(len(train_loader) / tcfg.grad_accum_steps)
    total_optim_steps = max(1, tcfg.epochs * optim_steps_per_epoch)
    if tcfg.max_steps is not None:
        if tcfg.max_steps < 1:
            raise ValueError("train.max_steps must be a positive integer or null")
        total_optim_steps = min(total_optim_steps, tcfg.max_steps)
    scheduler = make_scheduler(optimizer, tcfg.warmup_steps, total_optim_steps, tcfg.min_lr_ratio)
    scaler = torch.amp.GradScaler("cuda", enabled=(tcfg.amp_dtype == "fp16" and torch.cuda.is_available()))

    start_epoch, global_step, best_val = 0, 0, float("inf")
    resume_path = resolve_resume(tcfg.resume, tcfg.output_dir)
    if resume_path is not None:
        ck = load_checkpoint(resume_path, model, optimizer, scheduler, scaler, map_location=device)
        start_epoch = int(ck["epoch"])
        global_step = int(ck["global_step"])
        best_val = float(ck.get("best_val", best_val))
        if is_main(rank): print(f"RESUMED: {resume_path} epoch={start_epoch} step={global_step}")

    if is_main(rank):
        params = sum(p.numel() for p in unwrap_model(model).parameters() if p.requires_grad)
        print(f"Trainable parameters: {params/1e6:.2f} M")
        print(f"Distributed={distributed}, world_size={world_size}, device={device}, AMP={tcfg.amp_dtype}")
        print("Locked FiLM timing: all 4 legs use CURRENT a_t; FL/FR terrain=t-2; HL/HR terrain=t-3.")

    metrics_path = out_dir / "metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)

    if max_steps_reached(global_step, tcfg.max_steps):
        if is_main(rank):
            save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, scheduler, scaler, start_epoch, global_step, best_val, config_to_dict(mcfg, tcfg))
            print(f"Step cap already reached at step={global_step}; durable latest.pt written.")
        if distributed:
            dist.barrier()
            dist.destroy_process_group()
        return

    for epoch in range(start_epoch, tcfg.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        running = {}
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            batch = move_batch_to_device(batch, device)
            sync_now = ((batch_idx + 1) % tcfg.grad_accum_steps == 0) or (batch_idx + 1 == len(train_loader))
            ddp_ctx = nullcontext() if (not distributed or sync_now) else model.no_sync()
            with ddp_ctx:
                with amp_context(tcfg.amp_dtype):
                    out = model(batch, history_shift=tcfg.history_shift_samples)
                    losses = compute_losses(out, batch, mcfg, tcfg)
                    loss = losses["total"] / tcfg.grad_accum_steps
                scaler.scale(loss).backward()

            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + reduce_scalar(v, world_size)

            if sync_now:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                if is_main(rank) and global_step % tcfg.log_every == 0:
                    denom = tcfg.log_every * tcfg.grad_accum_steps
                    log = {k: v / max(1, denom) for k, v in running.items()}
                    log.update({"epoch": epoch, "step": global_step, "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - t0})
                    print(json.dumps(log))
                    with metrics_path.open("a") as f: f.write(json.dumps(log) + "\n")
                    running = {}; t0 = time.time()

                if is_main(rank) and global_step % tcfg.checkpoint_every_steps == 0:
                    step_path = ckpt_dir / f"step_{global_step:09d}.pt"
                    cfg_dict = config_to_dict(mcfg, tcfg)
                    save_checkpoint(step_path, model, optimizer, scheduler, scaler, epoch, global_step, best_val, cfg_dict)
                    save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, scheduler, scaler, epoch, global_step, best_val, cfg_dict)
                    prune_step_checkpoints(ckpt_dir, tcfg.keep_last_checkpoints)

                if global_step % tcfg.val_every_steps == 0:
                    val = validate(model, val_loader, device, mcfg, tcfg, world_size)
                    if is_main(rank):
                        print("VAL", json.dumps({"step": global_step, **val}))
                        with metrics_path.open("a") as f: f.write(json.dumps({"type": "val", "step": global_step, **val}) + "\n")
                        if val["total"] < best_val:
                            best_val = val["total"]
                            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, scheduler, scaler, epoch, global_step, best_val, config_to_dict(mcfg, tcfg))

                if STOP_REQUESTED:
                    if is_main(rank):
                        save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, scheduler, scaler, epoch, global_step, best_val, config_to_dict(mcfg, tcfg))
                        print("Stop signal received; durable latest.pt written.")
                    if distributed: dist.barrier()
                    return

                if max_steps_reached(global_step, tcfg.max_steps):
                    if is_main(rank):
                        save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, scheduler, scaler, epoch, global_step, best_val, config_to_dict(mcfg, tcfg))
                        print(f"Step cap reached at step={global_step}; durable latest.pt written.")
                    if distributed:
                        dist.barrier()
                        dist.destroy_process_group()
                    return

        # End-of-epoch durable checkpoint. Epoch+1 means resume at next epoch.
        if is_main(rank):
            save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, scheduler, scaler, epoch + 1, global_step, best_val, config_to_dict(mcfg, tcfg))
        if distributed: dist.barrier()

    if is_main(rank): print("Training complete.")
    if distributed: dist.destroy_process_group()


if __name__ == "__main__":
    main()
