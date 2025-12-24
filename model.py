import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from config import Architecture
from embed import EmbedWithoutTorch, EmbedWithTorch
from positional_embedding import PosEmbedWithEinops, PosEmbedWithoutEinops
from transformer_block import create_transformer_block
from layernorm import create_norm_layer
from rope import RoPE


class TransformerModelWithEinops(nn.Module):
    """Generic transformer model supporting both GPT and LLaMA architectures with Einops"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Token embeddings (same for both)
        self.embed = EmbedWithoutTorch(cfg)

        # Positional embeddings (only for GPT)
        if cfg.architecture == Architecture.GPT:
            self.pos_embed = PosEmbedWithEinops(cfg)
        else:  # LLaMA/OLMo - no positional embedding layer
            self.pos_embed = None

        # RoPE (only for LLaMA)
        if cfg.architecture == Architecture.LLAMA:
            self.rope = RoPE(cfg)
        else:
            self.rope = None

        # ALiBi (only for OLMo)
        if cfg.architecture == Architecture.OLMO:
            from alibi import ALiBi
            self.alibi = ALiBi(cfg)
        else:
            self.alibi = None

        # Transformer blocks
        self.blocks = nn.ModuleList([
            create_transformer_block(cfg, use_einops=True, rope=self.rope, alibi=self.alibi)
            for _ in range(cfg.n_layers)
        ])

        # Final normalization
        self.ln_f = create_norm_layer(cfg, use_einops=True)

        # Unembedding
        self.unembed = nn.Parameter(torch.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.unembed, std=self.cfg.init_range)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_vocab"]:
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
        for block in self.blocks:
            residual = block(residual)

        # Final layer norm
        # residual: [batch, position, d_model]
        residual = self.ln_f(residual)

        # Unembedding to logits
        # residual: [batch, position, d_model]
        # unembed: [d_model, d_vocab]
        # logits: [batch, position, d_vocab]
        logits = torch.matmul(residual, self.unembed)

        return logits


class TransformerModelWithoutEinops(nn.Module):
    """Generic transformer model supporting both GPT and LLaMA architectures without Einops"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Token embeddings (same for both)
        self.embed = EmbedWithTorch(cfg)

        # Positional embeddings (only for GPT)
        if cfg.architecture == Architecture.GPT:
            self.pos_embed = PosEmbedWithoutEinops(cfg)
        else:  # LLaMA/OLMo - no positional embedding layer
            self.pos_embed = None

        # RoPE (only for LLaMA)
        if cfg.architecture == Architecture.LLAMA:
            self.rope = RoPE(cfg)
        else:
            self.rope = None

        # ALiBi (only for OLMo)
        if cfg.architecture == Architecture.OLMO:
            from alibi import ALiBi
            self.alibi = ALiBi(cfg)
        else:
            self.alibi = None

        # Transformer blocks
        self.blocks = nn.ModuleList([
            create_transformer_block(cfg, use_einops=False, rope=self.rope, alibi=self.alibi)
            for _ in range(cfg.n_layers)
        ])

        # Final normalization
        self.ln_f = create_norm_layer(cfg, use_einops=False)

        # Unembedding
        self.unembed = nn.Parameter(torch.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.unembed, std=self.cfg.init_range)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_vocab"]:
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
        for block in self.blocks:
            residual = block(residual)

        # Final layer norm
        # residual: [batch, position, d_model]
        residual = self.ln_f(residual)

        # Unembedding to logits
        # residual: [batch, position, d_model]
        # unembed: [d_model, d_vocab]
        # logits: [batch, position, d_vocab]
        logits = torch.matmul(residual, self.unembed)

        return logits


# Backward compatibility aliases
GPTWithEinops = TransformerModelWithEinops
GPTWithoutEinops = TransformerModelWithoutEinops

