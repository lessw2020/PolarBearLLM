import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
from training_config import TrainingConfig
from model_config import PolarBearConfig
from model_polarbear import PolarBearLLM
import time

# training loop
trcfg = TrainingConfig()
print(f"{trcfg.batch_size=}")

mcfg = PolarBearConfig()
print(f"{mcfg=}")

if trcfg.master_process:
    os.makedirs(trcfg.out_dir, exist_ok=True)
torch.manual_seed(1337 + trcfg.seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in trcfg.device else "cpu"


# poor man's data loader
import numpy as np

data_dir = os.path.join("../data", trcfg.dataset)
train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
seqmax = mcfg.max_seq_len
device = trcfg.device


# --------- replace with sampler -----------------
def get_batch(split="train"):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - mcfg.max_seq_len, (trcfg.batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + seqmax]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [torch.from_numpy((data[i + 1 : i + 1 + seqmax]).astype(np.int64)) for i in ix]
    )
    if device_type == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# build model
from contextlib import nullcontext

dtype = "bfloat16"
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

model = PolarBearLLM(mcfg)
if mcfg.model_to_bf16:
    model.to(torch.bfloat16)
model.to(device)

# optimizer
optimizer = model.configure_optimizers(
    weight_decay=trcfg.weight_decay,
    learning_rate=trcfg.learning_rate,
    betas=(trcfg.beta1, trcfg.beta2),
    device_type=trcfg.device,
)


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(trcfg.eval_iters)
        for k in range(trcfg.eval_iters):
            X, Y = get_batch(split)
            # with ctx:
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# reduce lookups
warmup_iters = trcfg.warmup_iters
lr_decay_iters = trcfg.lr_decay_iters
min_lr = trcfg.min_lr
learning_rate = trcfg.learning_rate


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# logging
if trcfg.wandb_log and trcfg.master_process:
    import wandb

    wandb.init(project=trcfg.wandb_project, name=trcfg.wandb_run_name, config=trcfg)

# training loop
X, Y = get_batch("train")  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if trcfg.use_ddp else model  # unwrap DDP container if needed
running_mfu = -1.0
out_dir = trcfg.out_dir
iter_num = 0
master_process = trcfg.master_process
gradient_accumulation_steps = 1
best_val_loss = 1e10

while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if trcfg.decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % trcfg.eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(
            f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
        if trcfg.wandb_log:
            wandb.log(
                {
                    "iter": iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "lr": lr,
                    "mfu": running_mfu * 100,  # convert to percentage
                }
            )
        if losses["val"] < best_val_loss or trcfg.always_save_checkpoint:
            best_val_loss = losses["val"]
            if iter_num > 100:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    # "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": trcfg,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
    if iter_num == 0 and trcfg.eval_only:
        break

    """# forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (
                micro_step == gradient_accumulation_steps - 1
            )
        with ctx:
            logits, loss = model(X, Y)
            loss = (
                loss / gradient_accumulation_steps
            )  # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch("train")
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)
    """
    # with ctx:
    logits, loss = model(X, Y)
    X, Y = get_batch()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % trcfg.log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(
                trcfg.batch_size * gradient_accumulation_steps, dt
            )
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(
            f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
        )
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > trcfg.max_iters:
        break
