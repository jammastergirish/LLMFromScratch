from torch import nn
from jaxtyping import Float
from torch import Tensor
from typing import Optional, Tuple
from pretraining.attention.attention import AttentionWithEinops, AttentionWithoutEinops
from pretraining.mlp.mlp import create_mlp_layer
from pretraining.normalization.layernorm import create_norm_layer


def _extract_mlp_output_and_aux_loss(mlp_output):
    """Extract output and auxiliary loss from MLP (handles both MoE and standard MLPs)."""
    aux_loss = None
    if isinstance(mlp_output, tuple):
        mlp_output, aux_loss = mlp_output
    return mlp_output, aux_loss


class TransformerBlockWithEinops(nn.Module):
    def __init__(self, cfg, rope=None, alibi=None):
        super().__init__()
        self.cfg = cfg
        self.ln1 = create_norm_layer(cfg, use_einops=True)
        self.attn = AttentionWithEinops(cfg, rope=rope, alibi=alibi)
        self.ln2 = create_norm_layer(cfg, use_einops=True)
        self.mlp = create_mlp_layer(cfg, use_einops=True)

    def forward(
        self, 
        residual: Float[Tensor, "batch posn d_model"],
        cache: Optional[tuple[Float[Tensor, "batch cache_len n_heads d_head"], Float[Tensor, "batch cache_len n_heads d_head"]]] = None,
        start_pos: int = 0
    ) -> Tuple[Float[Tensor, "batch posn d_model"], tuple[Float[Tensor, "batch new_cache_len n_heads d_head"], Float[Tensor, "batch new_cache_len n_heads d_head"]], Optional[Float[Tensor, ""]]]:
        # residual: [batch, posn, d_model]

        # Pre-norm attention with residual connection
        # ln1(residual): [batch, posn, d_model]
        # attn(...): [batch, posn, d_model]
        # residual: [batch, posn, d_model]
        attn_output, new_cache = self.attn(self.ln1(residual), cache=cache, start_pos=start_pos)
        residual = residual + attn_output

        # Pre-norm MLP with residual connection
        # ln2(residual): [batch, posn, d_model]
        # mlp(...): [batch, posn, d_model] or (output, aux_loss) if MoE
        # residual: [batch, posn, d_model]
        mlp_output = self.mlp(self.ln2(residual))
        mlp_output, aux_loss = _extract_mlp_output_and_aux_loss(mlp_output)
        residual = residual + mlp_output

        return residual, new_cache, aux_loss


class TransformerBlockWithoutEinops(nn.Module):
    def __init__(self, cfg, rope=None, alibi=None):
        super().__init__()
        self.cfg = cfg
        self.ln1 = create_norm_layer(cfg, use_einops=False)
        self.attn = AttentionWithoutEinops(cfg, rope=rope, alibi=alibi)
        self.ln2 = create_norm_layer(cfg, use_einops=False)
        self.mlp = create_mlp_layer(cfg, use_einops=False)

    def forward(
        self, 
        residual: Float[Tensor, "batch posn d_model"],
        cache: Optional[tuple[Float[Tensor, "batch cache_len n_heads d_head"], Float[Tensor, "batch cache_len n_heads d_head"]]] = None,
        start_pos: int = 0
    ) -> Tuple[Float[Tensor, "batch posn d_model"], tuple[Float[Tensor, "batch new_cache_len n_heads d_head"], Float[Tensor, "batch new_cache_len n_heads d_head"]], Optional[Float[Tensor, ""]]]:
        # residual: [batch, posn, d_model]

        # Pre-norm attention with residual connection
        # ln1(residual): [batch, posn, d_model]
        # attn(...): [batch, posn, d_model]
        # residual: [batch, posn, d_model]
        attn_output, new_cache = self.attn(self.ln1(residual), cache=cache, start_pos=start_pos)
        residual = residual + attn_output

        # Pre-norm MLP with residual connection
        # ln2(residual): [batch, posn, d_model]
        # mlp(...): [batch, posn, d_model] or (output, aux_loss) if MoE
        # residual: [batch, posn, d_model]
        mlp_output = self.mlp(self.ln2(residual))
        mlp_output, aux_loss = _extract_mlp_output_and_aux_loss(mlp_output)
        residual = residual + mlp_output

        return residual, new_cache, aux_loss


def create_transformer_block(cfg, use_einops=True, rope=None, alibi=None):
    """Factory function to create transformer block"""
    if use_einops:
        return TransformerBlockWithEinops(cfg, rope=rope, alibi=alibi)
    return TransformerBlockWithoutEinops(cfg, rope=rope, alibi=alibi)
