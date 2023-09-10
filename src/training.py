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
