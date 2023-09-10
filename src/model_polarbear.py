import torch.nn.functional as F

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
from normalizations import SimpleRMSNorm
import inspect

from triton_masked import attention as triton_attention

# from triton_flash2 import attention as triton_attention

# keeping attention local for a bit to test out flash/triton stuff


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.emb_dim = cfg.emb_dim
        self.num_heads = cfg.num_heads
        self.use_sdpa = cfg.use_sdpa
        self.use_triton = cfg.use_triton_flash
        assert self.emb_dim % self.num_heads == 0
        self.seq_len = cfg.max_seq_len
        self.config = cfg
        self.in_proj = nn.Linear(self.emb_dim, 3 * self.emb_dim)
        self.out_proj = nn.Linear(self.emb_dim, self.emb_dim)
        self.scale = 0
        if not self.use_sdpa:
            # causal_mask = torch.triu(torch.ones(self.seq_len, self.seq_len) * float("-inf"), diagonal=1)
            # causal mask
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(self.seq_len, self.seq_len)).view(
                    1, 1, self.seq_len, self.seq_len
                ),
            )

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.in_proj(x).split(self.emb_dim, dim=2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(
            1, 2
        )  # (B, nh, T, hs)

        if not self.scale:
            self.scale = math.sqrt(k.size(-1))
            print(f"{self.scale=}")
            print(f"{k.shape=}")
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.use_triton:
            y = triton_attention(
                q,
                k,
                v,
                True,
                self.scale,
            )
        elif self.use_sdpa:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0, is_causal=True
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            # att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.out_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.in_proj = nn.Linear(cfg.emb_dim, cfg.mlp_expansion_factor * cfg.emb_dim)
        self.act_fn = nn.GELU()
        self.out_proj = nn.Linear(cfg.mlp_expansion_factor * cfg.emb_dim, cfg.emb_dim)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.act_fn(x)
        x = self.out_proj(x)
        return x


class PolarBearLLM(nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        assert cfg.vocab_size is not None
        assert cfg.max_seq_len is not None
        self.cfg = cfg

        self.transformer = nn.ModuleDict(
            dict(
                embeddings=nn.Embedding(cfg.vocab_size, cfg.emb_dim),
                pos_embeddings=nn.Embedding(cfg.max_seq_len, cfg.emb_dim),
                layers=nn.ModuleList(
                    [TransformerLayer(cfg) for _ in range(cfg.num_layers)]
                ),
                norm_out=SimpleRMSNorm(
                    cfg.emb_dim,
                ),
            )
        )

        self.lm_head = nn.Linear(cfg.emb_dim, cfg.vocab_size)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        if cfg.tie_head_with_embedding_weights:
            self.transformer.embeddings.weight = (
                self.lm_head.weight
            )  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("out_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.num_layers)
                )

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.embeddings.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # if module.bias is not None:
            #    torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        device = x.device
        b, t = x.size()
        assert (
            t <= self.cfg.max_seq_len
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the model itself
        tok_emb = self.transformer.embeddings(
            x
        )  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.pos_embeddings(
            pos
        )  # if self. # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb
        for block in self.transformer.layers:
            x = block(x)
        x = self.transformer.norm_out(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        grads_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for pn, p in param_dict.items() if p.dim() >= 2]
        no_decay_params = [p for pn, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        extra_args = dict(fused=True) if fused_available else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.cfg
        L, H, Q, T = (
            cfg.num_layers,
            cfg.num_heads,
            cfg.emb_dim // cfg.num_heads,
            cfg.max_seq_len,
        )
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = (
            125  # A10  312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        )
        mfu = flops_achieved / flops_promised
        return mfu


class TransformerLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.pre_norm = SimpleRMSNorm(cfg.emb_dim)
        self.attn = CausalSelfAttention(cfg)
        self.post_norm = SimpleRMSNorm(cfg.emb_dim)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.pre_norm(x))
        x = x + self.mlp(self.post_norm(x))
        return x
