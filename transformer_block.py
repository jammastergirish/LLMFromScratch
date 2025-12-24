import torch
import torch.nn as nn
from jaxtyping import Float, Tensor
from attention import AttentionWithEinops, AttentionWithoutEinops
from mlp import MLPWithEinops, MLPWithoutEinops
from layernorm import LayerNormWithEinops, LayerNormWithoutEinops


class TransformerBlockWithEinops(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNormWithEinops(cfg)
        self.attn = AttentionWithEinops(cfg)
        self.ln2 = LayerNormWithEinops(cfg)
        self.mlp = MLPWithEinops(cfg)

    def forward(
        self, residual: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        # Pre-norm attention with residual connection
        residual = residual + self.attn(self.ln1(residual))

        # Pre-norm MLP with residual connection
        residual = residual + self.mlp(self.ln2(residual))

        return residual


class TransformerBlockWithoutEinops(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNormWithoutEinops(cfg)
        self.attn = AttentionWithoutEinops(cfg)
        self.ln2 = LayerNormWithoutEinops(cfg)
        self.mlp = MLPWithoutEinops(cfg)

    def forward(
        self, residual: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        # Pre-norm attention with residual connection
        residual = residual + self.attn(self.ln1(residual))

        # Pre-norm MLP with residual connection
        residual = residual + self.mlp(self.ln2(residual))

        return residual
