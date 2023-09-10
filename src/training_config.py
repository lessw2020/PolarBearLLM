import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import torch.nn as nn


@dataclass
class TrainingConfig:
    out_dir = "out"
    eval_interval = 2000
    log_interval = 1
    eval_iters = 200
    eval_only = False
    wandb_log = False  # disabled by default
    wandb_project = "owt_flashalibi"
    wandb_run_name = "polarbear_start"
    dataset = "openwebtext"
    # gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
    batch_size = 12
    learning_rate = 6e-4  # max learning rate
    max_iters = 2  # 6000  # total number of training iterations
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr = True  # whether to decay the learning rate
    warmup_iters = 200  # how many steps to warm up for
    lr_decay_iters = 6000  # should be ~= max_iters per Chinchilla
    min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile = True  # use PyTorch 2.0
    master_process = True
    seed_offset = 2020
    ddp_world_size = 1
    use_ddp: bool = False
    always_save_checkpoint: bool = False
