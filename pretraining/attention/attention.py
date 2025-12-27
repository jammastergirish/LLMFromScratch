import torch
import torch.nn as nn
import einops
from jaxtyping import Float
from torch import Tensor
from typing import Optional


class AttentionWithEinops(nn.Module):
    def __init__(self, cfg, rope=None, alibi=None):
        super().__init__()
        self.cfg = cfg
        self.rope = rope  # RoPE module (None for GPT, RoPE instance for LLaMA)
        # ALiBi module (None for GPT/LLaMA, ALiBi instance for OLMo)
        self.alibi = alibi
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads  # Number of KV heads (for GQA/MQA)
        # W_Q: [n_heads, d_head, d_model] - separate Q for each head
        self.W_Q = nn.Parameter(torch.empty(
            (cfg.n_heads, cfg.d_head, cfg.d_model)))
        # W_K, W_V: [n_kv_heads, d_head, d_model] - shared K/V heads for GQA/MQA
        self.W_K = nn.Parameter(torch.empty(
            (cfg.n_kv_heads, cfg.d_head, cfg.d_model)))
        self.W_V = nn.Parameter(torch.empty(
            (cfg.n_kv_heads, cfg.d_head, cfg.d_model)))
        self.W_O = nn.Parameter(torch.empty(
            (cfg.n_heads, cfg.d_head, cfg.d_model)))

        for param in [self.W_Q, self.W_K, self.W_V, self.W_O]:
            nn.init.normal_(param, std=self.cfg.init_range)

    def forward(
        self, 
        residual: Float[Tensor, "batch posn d_model"],
        cache: Optional[tuple[Float[Tensor, "batch cache_len n_kv_heads d_head"], Float[Tensor, "batch cache_len n_kv_heads d_head"]]] = None,
        start_pos: int = 0
    ) -> tuple[Float[Tensor, "batch posn d_model"], tuple[Float[Tensor, "batch new_cache_len n_kv_heads d_head"], Float[Tensor, "batch new_cache_len n_kv_heads d_head"]]]:
        # residual: [batch, posn, d_model]
        seq_len = residual.shape[1]

        # Compute Q, K, V
        # residual: [batch, posn, d_model]
        # W_Q: [n_heads, d_head, d_model]
        # q: [batch, posn, n_heads, d_head]
        q = einops.einsum(
            residual, self.W_Q,
            "batch posn d_model, n_heads d_head d_model -> batch posn n_heads d_head"
        )
        # W_K, W_V: [n_kv_heads, d_head, d_model]
        # k: [batch, posn, n_kv_heads, d_head]
        k = einops.einsum(
            residual, self.W_K,
            "batch posn d_model, n_kv_heads d_head d_model -> batch posn n_kv_heads d_head"
        )
        # v: [batch, posn, n_kv_heads, d_head]
        v = einops.einsum(
            residual, self.W_V,
            "batch posn d_model, n_kv_heads d_head d_model -> batch posn n_kv_heads d_head"
        )

        # Handle KV cache: concatenate cached K, V with new ones
        # Note: Cached K, V already have RoPE applied (if using RoPE)
        # Cache shape: [batch, cache_len, n_kv_heads, d_head] (smaller than MHA!)
        if cache is not None:
            k_cache, v_cache = cache
            # k_cache, v_cache: [batch, cache_len, n_kv_heads, d_head]
            # k, v: [batch, seq_len, n_kv_heads, d_head]
            # Concatenate along sequence dimension
            k = torch.cat([k_cache, k], dim=1)  # [batch, cache_len + seq_len, n_kv_heads, d_head]
            v = torch.cat([v_cache, v], dim=1)  # [batch, cache_len + seq_len, n_kv_heads, d_head]
            total_len = k.shape[1]
        else:
            total_len = seq_len

        # Apply RoPE if provided (LLaMA)
        # For cached K, RoPE was already applied, so we only apply to new positions
        # Note: RoPE is applied to K with n_kv_heads shape
        if self.rope is not None:
            positions = torch.arange(start_pos, start_pos + seq_len, device=residual.device)
            # Apply RoPE to q (new positions only) and k (new positions only)
            # q: [batch, seq_len, n_heads, d_head], k_new: [batch, seq_len, n_kv_heads, d_head]
            q, k_new = self.rope(q, k[:, -seq_len:, :, :], positions)
            # Concatenate cached k (already rotated) with new rotated k
            if cache is not None:
                k = torch.cat([k[:, :-seq_len, :, :], k_new], dim=1)
            else:
                k = k_new

        # Store original K/V for caching (before broadcasting)
        # We cache the original n_kv_heads version to save memory
        k_for_cache = k[:, -seq_len:, :, :]  # [batch, seq_len, n_kv_heads, d_head]
        v_for_cache = v[:, -seq_len:, :, :]  # [batch, seq_len, n_kv_heads, d_head]

        # Broadcast K/V to match Q heads for GQA/MQA
        # k, v: [batch, posn_k, n_kv_heads, d_head]
        # We need to repeat each KV head to match multiple Q heads
        # For GQA: n_heads / n_kv_heads groups, each group shares one KV head
        if self.n_kv_heads < self.n_heads:
            # Repeat each KV head (n_heads / n_kv_heads) times
            repeat_factor = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=2)  # [batch, posn_k, n_heads, d_head]
            v = v.repeat_interleave(repeat_factor, dim=2)  # [batch, posn_k, n_heads, d_head]

        # Scaled dot-product attention
        # q: [batch, posn_q, n_heads, d_head]
        # k: [batch, posn_k, n_heads, d_head] (broadcasted, may include cache)
        # attn_scores: [batch, n_heads, posn_q, posn_k]
        attn_scores = einops.einsum(
            q, k,
            "batch posn_q n_heads d_head, batch posn_k n_heads d_head -> batch n_heads posn_q posn_k"
        ) / (self.cfg.d_head ** 0.5)

        # Apply ALiBi bias if provided (OLMo)
        if self.alibi is not None:
            alibi_bias = self.alibi.get_bias(
                total_len, residual.device)  # [n_heads, total_len, total_len]
            # Add bias to attention scores
            # [batch, n_heads, posn_q, posn_k]
            attn_scores = attn_scores + alibi_bias.unsqueeze(0)[:, :, start_pos:start_pos+seq_len, :]

        # Causal mask: mask out future positions
        # mask: [seq_len, total_len] - lower triangular matrix for current sequence
        # We need to mask so that position i can only attend to positions <= i
        mask = torch.tril(torch.ones(
            (seq_len, total_len), device=residual.device))
        # attn_scores: [batch, n_heads, posn_q, posn_k] - masked
        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        # Softmax over last dimension (keys)
        # attn_pattern: [batch, n_heads, posn_q, posn_k]
        attn_pattern = torch.softmax(attn_scores, dim=-1)

        # Apply attention to values
        # attn_pattern: [batch, n_heads, posn_q, posn_k]
        # v: [batch, posn_k, n_heads, d_head]
        # attn_output: [batch, posn_q, n_heads, d_head]
        attn_output = einops.einsum(
            attn_pattern, v,
            "batch n_heads posn_q posn_k, batch posn_k n_heads d_head -> batch posn_q n_heads d_head"
        )

        # Project back to d_model
        # attn_output: [batch, posn, n_heads, d_head]
        # W_O: [n_heads, d_head, d_model]
        # output: [batch, posn, d_model]
        output = einops.einsum(
            attn_output, self.W_O,
            "batch posn n_heads d_head, n_heads d_head d_model -> batch posn d_model"
        )

        # Return cache: use the original (non-broadcasted) K/V to save memory
        # k_for_cache, v_for_cache: [batch, seq_len, n_kv_heads, d_head]
        return output, (k_for_cache, v_for_cache)


class AttentionWithoutEinops(nn.Module):
    def __init__(self, cfg, rope=None, alibi=None):
        super().__init__()
        self.cfg = cfg
        self.rope = rope  # RoPE module (None for GPT, RoPE instance for LLaMA)
        # ALiBi module (None for GPT/LLaMA, ALiBi instance for OLMo)
        self.alibi = alibi
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads  # Number of KV heads (for GQA/MQA)
        # W_Q: [n_heads, d_head, d_model] - per-head projection matrices
        self.W_Q = nn.Parameter(torch.empty(
            (cfg.n_heads, cfg.d_head, cfg.d_model)))
        # W_K, W_V: [n_kv_heads, d_head, d_model] - shared K/V heads for GQA/MQA
        self.W_K = nn.Parameter(torch.empty(
            (cfg.n_kv_heads, cfg.d_head, cfg.d_model)))
        self.W_V = nn.Parameter(torch.empty(
            (cfg.n_kv_heads, cfg.d_head, cfg.d_model)))
        # W_O: [n_heads, d_head, d_model] - output projection
        self.W_O = nn.Parameter(torch.empty(
            (cfg.n_heads, cfg.d_head, cfg.d_model)))

        for param in [self.W_Q, self.W_K, self.W_V, self.W_O]:
            nn.init.normal_(param, std=self.cfg.init_range)

    def forward(
        self, 
        residual: Float[Tensor, "batch posn d_model"],
        cache: Optional[tuple[Float[Tensor, "batch cache_len n_kv_heads d_head"], Float[Tensor, "batch cache_len n_kv_heads d_head"]]] = None,
        start_pos: int = 0
    ) -> tuple[Float[Tensor, "batch posn d_model"], tuple[Float[Tensor, "batch new_cache_len n_kv_heads d_head"], Float[Tensor, "batch new_cache_len n_kv_heads d_head"]]]:
        # residual: [batch, posn, d_model]
        seq_len = residual.shape[1]

        # Compute Q, K, V
        # residual: [batch, seq_len, d_model]
        # W_Q: [n_heads, d_head, d_model]
        # q: [batch, seq_len, n_heads, d_head]
        q = torch.einsum("bpd,nhd->bpnh", residual, self.W_Q)
        # W_K, W_V: [n_kv_heads, d_head, d_model]
        # k: [batch, seq_len, n_kv_heads, d_head]
        k = torch.einsum("bpd,nkd->bpnk", residual, self.W_K)
        # v: [batch, seq_len, n_kv_heads, d_head]
        v = torch.einsum("bpd,nkd->bpnk", residual, self.W_V)

        # Handle KV cache: concatenate cached K, V with new ones
        # Note: Cached K, V already have RoPE applied (if using RoPE)
        # Cache shape: [batch, cache_len, n_kv_heads, d_head] (smaller than MHA!)
        if cache is not None:
            k_cache, v_cache = cache
            # k_cache, v_cache: [batch, cache_len, n_kv_heads, d_head]
            # k, v: [batch, seq_len, n_kv_heads, d_head]
            # Concatenate along sequence dimension
            k = torch.cat([k_cache, k], dim=1)  # [batch, cache_len + seq_len, n_kv_heads, d_head]
            v = torch.cat([v_cache, v], dim=1)  # [batch, cache_len + seq_len, n_kv_heads, d_head]
            total_len = k.shape[1]
        else:
            total_len = seq_len

        # Apply RoPE if provided (LLaMA)
        # For cached K, RoPE was already applied, so we only apply to new positions
        # Note: RoPE is applied to K with n_kv_heads shape
        if self.rope is not None:
            positions = torch.arange(start_pos, start_pos + seq_len, device=residual.device)
            # Apply RoPE to q (new positions only) and k (new positions only)
            # q: [batch, seq_len, n_heads, d_head], k_new: [batch, seq_len, n_kv_heads, d_head]
            q, k_new = self.rope(q, k[:, -seq_len:, :, :], positions)
            # Concatenate cached k (already rotated) with new rotated k
            if cache is not None:
                k = torch.cat([k[:, :-seq_len, :, :], k_new], dim=1)
            else:
                k = k_new

        # Store original K/V for caching (before broadcasting)
        # We cache the original n_kv_heads version to save memory
        k_for_cache = k[:, -seq_len:, :, :]  # [batch, seq_len, n_kv_heads, d_head]
        v_for_cache = v[:, -seq_len:, :, :]  # [batch, seq_len, n_kv_heads, d_head]

        # Broadcast K/V to match Q heads for GQA/MQA
        # k, v: [batch, posn_k, n_kv_heads, d_head]
        # We need to repeat each KV head to match multiple Q heads
        if self.n_kv_heads < self.n_heads:
            # Repeat each KV head (n_heads / n_kv_heads) times
            repeat_factor = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=2)  # [batch, posn_k, n_heads, d_head]
            v = v.repeat_interleave(repeat_factor, dim=2)  # [batch, posn_k, n_heads, d_head]

        # Scaled dot-product attention
        # Transpose to [batch, n_heads, seq_len, d_head] for matmul
        # q: [batch, seq_len, n_heads, d_head] -> [batch, n_heads, seq_len, d_head]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # attn_scores: [batch, n_heads, seq_len, total_len]
        # q @ k^T: [batch, n_heads, seq_len, d_head] @ [batch, n_heads, d_head, total_len]
        attn_scores = torch.matmul(
            q, k.transpose(-2, -1)) / (self.cfg.d_head ** 0.5)

        # Apply ALiBi bias if provided (OLMo)
        if self.alibi is not None:
            alibi_bias = self.alibi.get_bias(
                total_len, residual.device)  # [n_heads, total_len, total_len]
            # Add bias to attention scores (only for current sequence positions)
            # [batch, n_heads, seq_len, total_len]
            attn_scores = attn_scores + alibi_bias.unsqueeze(0)[:, :, start_pos:start_pos+seq_len, :]

        # Causal mask: mask out future positions
        # mask: [seq_len, total_len] - lower triangular matrix for current sequence
        # We need to mask so that position i can only attend to positions <= i
        mask = torch.tril(torch.ones(
            (seq_len, total_len), device=residual.device))
        # attn_scores: [batch, n_heads, seq_len, total_len] - masked
        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        # Softmax over last dimension (keys)
        # attn_pattern: [batch, n_heads, seq_len, seq_len]
        attn_pattern = torch.softmax(attn_scores, dim=-1)

        # Apply attention to values
        # attn_pattern: [batch, n_heads, seq_len, seq_len]
        # v: [batch, n_heads, seq_len, d_head]
        # attn_output: [batch, n_heads, seq_len, d_head]
        attn_output = torch.matmul(attn_pattern, v)

        # Concatenate heads: [batch, n_heads, seq_len, d_head] -> [batch, seq_len, n_heads, d_head]
        attn_output = attn_output.transpose(1, 2)

        # Project back to d_model
        # attn_output: [batch, seq_len, n_heads, d_head]
        # W_O: [n_heads, d_head, d_model]
        # output: [batch, seq_len, d_model]
        output = torch.einsum("bpnh,nhd->bpd", attn_output, self.W_O)

        # Return cache: use the original (non-broadcasted) K/V to save memory
        # k_for_cache, v_for_cache: [batch, seq_len, n_kv_heads, d_head]
        return output, (k_for_cache, v_for_cache)
