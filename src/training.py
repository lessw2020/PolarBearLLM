import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
from training_config import TrainingConfig

# training loop
tr = TrainingConfig
print(f"{tr.batch_size=}")

if tr.master_process:
    os.makedirs(tr.out_dir, exist_ok=True)
torch.manual_seed(1337 + tr.seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in tr.device else "cpu"
