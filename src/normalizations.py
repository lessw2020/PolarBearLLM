
class SimpleRMSNorm(nn.Module):
    """Simple RMSNorm
    SRMSNorm(x) = x / ∥x∥2/√d
    as proposed in:
    Scaling TransNormer to 175 Billion Parameters
    https://arxiv.org/abs/2307.14995
    """

    def __init__(self, dim: int, eps: float = 1e-12):
        super().__init__()
        self.scaling = dim**0.5
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        denom = x.norm(p=2, dim=-1, keepdim=True).clamp_min(self.eps).expand_as(x)
        return (x / denom) * self.scaling
