import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import torch.nn as nn


@dataclass
class PolarBearConfig:
    vocab_size: int = 50_304
    emb_dim: int = 128
    num_heads: int = 2
    num_kv_heads: int = 0
    num_layers: int = 2
    mlp_expansion_factor: float = 4
    multiple_of: int = 256
    activation_fn: str = "gelu"
    p_dropout: float = 0.0
    max_seq_len: int = 1024
    pos_embeddings: str = "alibi"
    use_sdpa: bool = False
    use_triton_flash: bool = True
    use_learned_emb: bool = True
    tie_head_with_embedding_weights: bool = True
