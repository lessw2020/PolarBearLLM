class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.emb_dim = cfg.emb_dim
        self.num_heads = cfg.num_heads
        self.use_flash = cfg.use_flash
        assert self.emb_dim % self.num_heads == 0
        self.seq_len = cfg.max_seq_len

        self.in_proj = nn.Linear(self.emb_dim, 3 * self.emb_dim)
        self.out_proj = nn.Linear(self.emb_dim, self.emb_dim) 
      
        
        if not self.flash:
            # causal_mask = torch.triu(torch.ones(self.seq_len, self.seq_len) * float("-inf"), diagonal=1)
            # causal mask
            self.register_buffer("bias", torch.tril(torch.ones(self.seq_len, self.seq_len))
                                        .view(1, 1, self.seq_len, self.seq_len))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.in_proj(x).split(self.emb_dim, dim=2)
        k = k.view(B, T, self.n_head, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.out_proj(y)
        return y
