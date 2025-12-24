import torch
import torch.nn as nn
import einops
from jaxtyping import Float, Tensor


class LayerNormWithEinops(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(torch.ones(cfg.d_model))
        self.b = nn.Parameter(torch.zeros(cfg.d_model))

    def forward(
        self, residual: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:

        residual_mean = einops.reduce(
            residual, 'batch posn d_model -> batch posn 1', 'mean')

        def layernorm_variance(x, axis):
            """Variance for LayerNorm (uses N not N-1 denominator)"""
            return x.var(axis=axis, unbiased=False)

        residual_variance = einops.reduce(
            residual, 'batch posn d_model -> batch posn 1', layernorm_variance)

        residual_std = torch.sqrt(residual_variance + self.cfg.layer_norm_eps)

        residual = (residual - residual_mean) / residual_std
        return residual * self.w + self.b


class LayerNormWithoutEinops(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(torch.ones(cfg.d_model))
        self.b = nn.Parameter(torch.zeros(cfg.d_model))

    def forward(
        self, residual: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:

        residual_mean = residual.mean(dim=-1, keepdim=True)

        residual_variance = residual.var(dim=-1, keepdim=True, unbiased=False)

        residual_std = torch.sqrt(residual_variance + self.cfg.layer_norm_eps)

        residual = (residual - residual_mean) / residual_std

        return residual * self.w + self.b


class LayerNormWithTorch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ln = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)

    def forward(
        self, residual: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        return self.ln(residual)
