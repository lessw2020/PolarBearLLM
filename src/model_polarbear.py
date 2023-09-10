import torch.nn.functional as F

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import torch.nn as nn

class PolarBearLLM(nn.Module):
  def __init__(self, cfg, ):
        super().__init__()
        assert cfg.vocab_size is not None
        assert cfg.max_seq_len is not None
        self.cfg = cfg

        self.transformer = nn.ModuleDict(dict(
            embeddings = nn.Embedding(cfg.vocab_size, cfg.emb_dim),
            pos_embeddings = nn.Embedding(cfg.max_seq_len, cfg.emb_dim),
            layers = nn.ModuleList([TransformerLayer(cfg) for _ in range(cfg.num_layers)]),
            norm_out = SimpleRMSNorm(cfg.emb_dim,),
        ))

        self.lm_head = nn.Linear(cfg.emb_dim, cfg.vocab_size)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.embeddings.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('out_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * cfg.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

  def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

  def _init_weights(self, module):
      if isinstance(module, nn.Linear):
          torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
          #if module.bias is not None:
          #    torch.nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
          torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
  
  def forward(self, x, targets=None):
        device = x.device
        b, t = x.size()
        assert t <= self.config.max_seq_len, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the model itself
        tok_emb = self.transformer.embeddings(x) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.pos_embeddings(pos) # if self. # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb
        for block in self.transformer.layers:
            x = block(x)
        x = self.transformer.norm_out(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

  def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):

    param_dict = {pn: p for pn, p in self.named_parameters() }
    grads_dict = {pn:p for pn, p in param_dict.items() if p.requires_grad}             
    decay_params = [p for pn, p in param_dict.items() if p.dim() >=2]
    no_decay_params = [p for pn, p in param_dict.items() if p.dim() < 2]
    optim_groups = [{'params': decay_params, 'weight_decay': weight_decay}, {'params':no_decay_params, 'weight_decay':0.0}]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_no_decay_params = sum(p.numel() for p in no_decay_params)

    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if fused_available else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    return optimizer
