from dataclasses import dataclass
from enum import Enum
from typing import Union


class Architecture(str, Enum):
    GPT = "gpt"
    LLAMA = "llama"
    OLMO = "olmo"


class PositionalEncoding(str, Enum):
    LEARNED = "learned"  # Learned positional embeddings (GPT style)
    ROPE = "rope"  # Rotary Position Embedding (LLaMA style)
    ALIBI = "alibi"  # Attention with Linear Biases (OLMo style)
    NONE = "none"  # No positional encoding


class Normalization(str, Enum):
    LAYERNORM = "layernorm"  # LayerNorm (GPT, OLMo style)
    RMSNORM = "rmsnorm"  # RMSNorm (LLaMA style)


class Activation(str, Enum):
    GELU = "gelu"  # GELU activation (GPT style)
    SWIGLU = "swiglu"  # SwiGLU activation (LLaMA, OLMo style)


@dataclass
class ModelConfig:
    # Architecture selection (required - no default)
    architecture: Architecture

    # Model dimensions
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12

    # Configurable model components (if None, will be set based on architecture)
    positional_encoding: Union[PositionalEncoding, None] = None
    normalization: Union[Normalization, None] = None
    activation: Union[Activation, None] = None

    # LLaMA-specific
    rope_theta: float = 10000.0  # Base frequency for RoPE

    def __post_init__(self):
        """Set defaults based on architecture if not explicitly provided"""
        if self.positional_encoding is None:
            if self.architecture == Architecture.GPT:
                self.positional_encoding = PositionalEncoding.LEARNED
            elif self.architecture == Architecture.LLAMA:
                self.positional_encoding = PositionalEncoding.ROPE
            elif self.architecture == Architecture.OLMO:
                self.positional_encoding = PositionalEncoding.ALIBI
            else:
                self.positional_encoding = PositionalEncoding.NONE

        if self.normalization is None:
            if self.architecture == Architecture.LLAMA:
                self.normalization = Normalization.RMSNORM
            else:  # GPT, OLMO
                self.normalization = Normalization.LAYERNORM

        if self.activation is None:
            if self.architecture == Architecture.GPT:
                self.activation = Activation.GELU
            else:  # LLaMA, OLMo
                self.activation = Activation.SWIGLU

    @classmethod
    def gpt_small(cls):
        """Small GPT config for faster training/testing (good for Mac)"""
        return cls(
            architecture=Architecture.GPT,
            d_model=256,
            n_heads=4,
            n_layers=4,
            n_ctx=256,
            d_head=64,
            d_mlp=1024,
            d_vocab=50257,  # Will be updated by tokenizer
        )

    @classmethod
    def gpt_medium(cls):
        """Medium GPT config (between small and full)"""
        return cls(
            architecture=Architecture.GPT,
            d_model=512,
            n_heads=8,
            n_layers=6,
            n_ctx=512,
            d_head=64,
            d_mlp=2048,
            d_vocab=50257,
        )

    @classmethod
    def gpt_full(cls):
        """Full GPT-2 size config"""
        return cls(
            architecture=Architecture.GPT,
            d_model=768,
            n_heads=12,
            n_layers=12,
            n_ctx=1024,
            d_head=64,
            d_mlp=3072,
            d_vocab=50257,
        )

    @classmethod
    def llama_small(cls):
        """Small LLaMA config for faster training/testing"""
        return cls(
            architecture=Architecture.LLAMA,
            d_model=256,
            n_heads=4,
            n_layers=4,
            n_ctx=256,
            d_head=64,
            d_mlp=1024,
            d_vocab=50257,  # Will be updated by tokenizer
            rope_theta=10000.0,
        )

    @classmethod
    def llama_medium(cls):
        """Medium LLaMA config (between small and full)"""
        return cls(
            architecture=Architecture.LLAMA,
            d_model=512,
            n_heads=8,
            n_layers=6,
            n_ctx=512,
            d_head=64,
            d_mlp=2048,
            d_vocab=50257,
            rope_theta=10000.0,
        )

    @classmethod
    def llama_full(cls):
        """Full LLaMA config"""
        return cls(
            architecture=Architecture.LLAMA,
            d_model=768,
            n_heads=12,
            n_layers=12,
            n_ctx=1024,
            d_head=64,
            d_mlp=3072,
            d_vocab=50257,
            rope_theta=10000.0,
        )

    @classmethod
    def olmo_small(cls):
        """Small OLMo config for faster training/testing"""
        return cls(
            architecture=Architecture.OLMO,
            d_model=256,
            n_heads=4,
            n_layers=4,
            n_ctx=256,
            d_head=64,
            d_mlp=1024,
            d_vocab=50257,  # Will be updated by tokenizer
        )

    @classmethod
    def olmo_medium(cls):
        """Medium OLMo config (between small and full)"""
        return cls(
            architecture=Architecture.OLMO,
            d_model=512,
            n_heads=8,
            n_layers=6,
            n_ctx=512,
            d_head=64,
            d_mlp=2048,
            d_vocab=50257,
        )

    @classmethod
    def olmo_full(cls):
        """Full OLMo config"""
        return cls(
            architecture=Architecture.OLMO,
            d_model=768,
            n_heads=12,
            n_layers=12,
            n_ctx=1024,
            d_head=64,
            d_mlp=3072,
            d_vocab=50257,
        )
