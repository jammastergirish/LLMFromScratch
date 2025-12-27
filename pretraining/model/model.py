import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from typing import Optional, Union, Tuple
from config import Architecture, PositionalEncoding
from pretraining.embeddings.embed import EmbedWithoutTorch, EmbedWithTorch, UnembedWithoutTorch, UnembedWithTorch
from pretraining.positional_embeddings.positional_embedding import PosEmbedWithEinops, PosEmbedWithoutEinops
from pretraining.transformer_blocks.transformer_block import create_transformer_block
from pretraining.normalization.layernorm import create_norm_layer
from pretraining.positional_embeddings.rope import RoPE


def _aggregate_aux_losses(aux_losses: list) -> Optional[Float[Tensor, ""]]:
    """Aggregate auxiliary losses from multiple MoE layers."""
    if aux_losses:
        return sum(aux_losses)
    return None


class TransformerModelWithEinops(nn.Module):
    """Generic transformer model supporting both GPT and LLaMA architectures with Einops"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Token embeddings (same for both)
        self.embed = EmbedWithoutTorch(cfg)

        # Positional embeddings (based on positional_encoding config)
        if cfg.positional_encoding == PositionalEncoding.LEARNED:
            self.pos_embed = PosEmbedWithEinops(cfg)
        else:
            self.pos_embed = None

        # RoPE (for ROPE positional encoding)
        if cfg.positional_encoding == PositionalEncoding.ROPE:
            self.rope = RoPE(cfg)
        else:
            self.rope = None

        # ALiBi (for ALIBI positional encoding)
        if cfg.positional_encoding == PositionalEncoding.ALIBI:
            from pretraining.positional_embeddings.alibi import ALiBi
            self.alibi = ALiBi(cfg)
        else:
            self.alibi = None

        # Transformer blocks
        self.blocks = nn.ModuleList([
            create_transformer_block(
                cfg, use_einops=True, rope=self.rope, alibi=self.alibi)
            for _ in range(cfg.n_layers)
        ])

        # Final normalization
        self.ln_f = create_norm_layer(cfg, use_einops=True)

        # Unembedding
        self.unembed = UnembedWithoutTorch(cfg)

    def forward(
        self, 
        tokens: Int[Tensor, "batch position"],
        cache: Optional[list[tuple[Float[Tensor, "batch cache_len n_heads d_head"], Float[Tensor, "batch cache_len n_heads d_head"]]]] = None,
        start_pos: int = 0
    ) -> Union[
        Float[Tensor, "batch position d_vocab"],
        Tuple[Float[Tensor, "batch position d_vocab"], Optional[Float[Tensor, ""]]],
        Tuple[Float[Tensor, "batch position d_vocab"], Optional[list[tuple[Float[Tensor, "batch new_cache_len n_heads d_head"], Float[Tensor, "batch new_cache_len n_heads d_head"]]]],
        Tuple[Float[Tensor, "batch position d_vocab"], Optional[list[tuple[Float[Tensor, "batch new_cache_len n_heads d_head"], Float[Tensor, "batch new_cache_len n_heads d_head"]]], Optional[Float[Tensor, ""]]]
    ]:
        # tokens: [batch, position]

        # Token embeddings
        # residual: [batch, position, d_model]
        residual = self.embed(tokens)

        # Positional embeddings (GPT only)
        if self.pos_embed is not None:
            # pos_emb: [batch, position, d_model]
            # residual: [batch, position, d_model]
            residual = residual + self.pos_embed(tokens)

        # Transformer blocks
        # Each block: [batch, position, d_model] -> [batch, position, d_model]
        new_cache_list = []
        aux_losses = []
        for i, block in enumerate(self.blocks):
            block_cache = cache[i] if cache is not None else None
            residual, new_cache, aux_loss = block(residual, cache=block_cache, start_pos=start_pos)
            new_cache_list.append(new_cache)
            if aux_loss is not None:
                aux_losses.append(aux_loss)

        # Aggregate auxiliary losses from all MoE layers
        total_aux_loss = _aggregate_aux_losses(aux_losses)

        # Final layer norm
        # residual: [batch, position, d_model]
        residual = self.ln_f(residual)

        # Unembedding to logits
        # residual: [batch, position, d_model]
        # logits: [batch, position, d_vocab]
        logits = self.unembed(residual)

        # Return format: maintain backward compatibility
        # If MoE is enabled and training, return (logits, aux_loss)
        # If cache was provided, return (logits, cache) or (logits, cache, aux_loss)
        # Otherwise return just logits
        if cache is not None:
            if total_aux_loss is not None:
                return logits, new_cache_list, total_aux_loss
            else:
                return logits, new_cache_list
        else:
            if total_aux_loss is not None:
                return logits, total_aux_loss
            else:
                return logits


class TransformerModelWithoutEinops(nn.Module):
    """Generic transformer model supporting both GPT and LLaMA architectures without Einops"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Token embeddings (same for both)
        self.embed = EmbedWithTorch(cfg)

        # Positional embeddings (based on positional_encoding config)
        if cfg.positional_encoding == PositionalEncoding.LEARNED:
            self.pos_embed = PosEmbedWithoutEinops(cfg)
        else:
            self.pos_embed = None

        # RoPE (for ROPE positional encoding)
        if cfg.positional_encoding == PositionalEncoding.ROPE:
            self.rope = RoPE(cfg)
        else:
            self.rope = None

        # ALiBi (for ALIBI positional encoding)
        if cfg.positional_encoding == PositionalEncoding.ALIBI:
            from pretraining.positional_embeddings.alibi import ALiBi
            self.alibi = ALiBi(cfg)
        else:
            self.alibi = None

        # Transformer blocks
        self.blocks = nn.ModuleList([
            create_transformer_block(
                cfg, use_einops=False, rope=self.rope, alibi=self.alibi)
            for _ in range(cfg.n_layers)
        ])

        # Final normalization
        self.ln_f = create_norm_layer(cfg, use_einops=False)

        # Unembedding
        self.unembed = UnembedWithTorch(cfg)

    def forward(
        self, 
        tokens: Int[Tensor, "batch position"],
        cache: Optional[list[tuple[Float[Tensor, "batch cache_len n_heads d_head"], Float[Tensor, "batch cache_len n_heads d_head"]]]] = None,
        start_pos: int = 0
    ) -> Union[
        Float[Tensor, "batch position d_vocab"],
        Tuple[Float[Tensor, "batch position d_vocab"], Optional[Float[Tensor, ""]]],
        Tuple[Float[Tensor, "batch position d_vocab"], Optional[list[tuple[Float[Tensor, "batch new_cache_len n_heads d_head"], Float[Tensor, "batch new_cache_len n_heads d_head"]]]],
        Tuple[Float[Tensor, "batch position d_vocab"], Optional[list[tuple[Float[Tensor, "batch new_cache_len n_heads d_head"], Float[Tensor, "batch new_cache_len n_heads d_head"]]], Optional[Float[Tensor, ""]]]
    ]:
        # tokens: [batch, position]

        # Token embeddings
        # residual: [batch, position, d_model]
        residual = self.embed(tokens)

        # Positional embeddings (GPT only)
        if self.pos_embed is not None:
            # pos_emb: [batch, position, d_model]
            # residual: [batch, position, d_model]
            residual = residual + self.pos_embed(tokens)

        # Transformer blocks
        # Each block: [batch, position, d_model] -> [batch, position, d_model]
        new_cache_list = []
        aux_losses = []
        for i, block in enumerate(self.blocks):
            block_cache = cache[i] if cache is not None else None
            residual, new_cache, aux_loss = block(residual, cache=block_cache, start_pos=start_pos)
            new_cache_list.append(new_cache)
            if aux_loss is not None:
                aux_losses.append(aux_loss)

        # Aggregate auxiliary losses from all MoE layers
        total_aux_loss = _aggregate_aux_losses(aux_losses)

        # Final layer norm
        # residual: [batch, position, d_model]
        residual = self.ln_f(residual)

        # Unembedding to logits
        # residual: [batch, position, d_model]
        # logits: [batch, position, d_vocab]
        logits = self.unembed(residual)

        # Return format: maintain backward compatibility
        # If MoE is enabled and training, return (logits, aux_loss)
        # If cache was provided, return (logits, cache) or (logits, cache, aux_loss)
        # Otherwise return just logits
        if cache is not None:
            if total_aux_loss is not None:
                return logits, new_cache_list, total_aux_loss
            else:
                return logits, new_cache_list
        else:
            if total_aux_loss is not None:
                return logits, total_aux_loss
            else:
                return logits


# Backward compatibility aliases
GPTWithEinops = TransformerModelWithEinops
GPTWithoutEinops = TransformerModelWithoutEinops

