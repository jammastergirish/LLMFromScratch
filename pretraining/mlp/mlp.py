import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from jaxtyping import Float
from torch import Tensor
from typing import Optional, Tuple
from config import RouterType


class MLPWithEinops(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # W_in: [d_model, d_mlp] - input projection
        self.W_in = nn.Parameter(torch.empty((cfg.d_model, cfg.d_mlp)))
        # b_in: [d_mlp] - input bias
        self.b_in = nn.Parameter(torch.zeros(cfg.d_mlp))
        # W_out: [d_mlp, d_model] - output projection
        self.W_out = nn.Parameter(torch.empty((cfg.d_mlp, cfg.d_model)))
        # b_out: [d_model] - output bias
        self.b_out = nn.Parameter(torch.zeros(cfg.d_model))

        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        nn.init.normal_(self.W_out, std=self.cfg.init_range)

    def forward(
        self, residual: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        # First linear layer: d_model -> d_mlp
        # residual: [batch, posn, d_model]
        # W_in: [d_model, d_mlp]
        # hidden: [batch, posn, d_mlp]
        hidden = einops.einsum(
            residual, self.W_in,
            "batch posn d_model, d_model d_mlp -> batch posn d_mlp"
        ) + self.b_in

        # GELU activation (element-wise)
        # hidden: [batch, posn, d_mlp]
        hidden = torch.nn.functional.gelu(hidden)

        # Second linear layer: d_mlp -> d_model
        # hidden: [batch, posn, d_mlp]
        # W_out: [d_mlp, d_model]
        # output: [batch, posn, d_model]
        output = einops.einsum(
            hidden, self.W_out,
            "batch posn d_mlp, d_mlp d_model -> batch posn d_model"
        ) + self.b_out

        return output


class MLPWithoutEinops(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # W_in: [d_model, d_mlp] - input projection
        self.W_in = nn.Parameter(torch.empty((cfg.d_model, cfg.d_mlp)))
        # b_in: [d_mlp] - input bias
        self.b_in = nn.Parameter(torch.zeros(cfg.d_mlp))
        # W_out: [d_mlp, d_model] - output projection
        self.W_out = nn.Parameter(torch.empty((cfg.d_mlp, cfg.d_model)))
        # b_out: [d_model] - output bias
        self.b_out = nn.Parameter(torch.zeros(cfg.d_model))

        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        nn.init.normal_(self.W_out, std=self.cfg.init_range)

    def forward(
        self, residual: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        # First linear layer: d_model -> d_mlp
        # residual: [batch, posn, d_model]
        # W_in: [d_model, d_mlp]
        # hidden: [batch, posn, d_mlp]
        hidden = torch.matmul(residual, self.W_in) + self.b_in

        # GELU activation (element-wise)
        # hidden: [batch, posn, d_mlp]
        hidden = torch.nn.functional.gelu(hidden)

        # Second linear layer: d_mlp -> d_model
        # hidden: [batch, posn, d_mlp]
        # W_out: [d_mlp, d_model]
        # output: [batch, posn, d_model]
        output = torch.matmul(hidden, self.W_out) + self.b_out

        return output


class MLPSwiGLUWithEinops(nn.Module):
    """MLP with SwiGLU activation (LLaMA style) using Einops"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # SwiGLU needs 3 weight matrices
        # W_gate: [d_model, d_mlp] - gate branch
        self.W_gate = nn.Parameter(torch.empty((cfg.d_model, cfg.d_mlp)))
        # W_up: [d_model, d_mlp] - up branch
        self.W_up = nn.Parameter(torch.empty((cfg.d_model, cfg.d_mlp)))
        # W_out: [d_mlp, d_model] - output projection
        self.W_out = nn.Parameter(torch.empty((cfg.d_mlp, cfg.d_model)))
        self.b_gate = nn.Parameter(torch.zeros(cfg.d_mlp))
        self.b_up = nn.Parameter(torch.zeros(cfg.d_mlp))
        self.b_out = nn.Parameter(torch.zeros(cfg.d_model))

        nn.init.normal_(self.W_gate, std=self.cfg.init_range)
        nn.init.normal_(self.W_up, std=self.cfg.init_range)
        nn.init.normal_(self.W_out, std=self.cfg.init_range)

    def forward(
        self, residual: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        # residual: [batch, posn, d_model]

        # Gate branch: Swish activation (SiLU)
        # gate: [batch, posn, d_mlp]
        gate = torch.nn.functional.silu(
            einops.einsum(
                residual, self.W_gate,
                "batch posn d_model, d_model d_mlp -> batch posn d_mlp"
            ) + self.b_gate
        )  # SiLU = Swish

        # Up branch: linear
        # up: [batch, posn, d_mlp]
        up = einops.einsum(
            residual, self.W_up,
            "batch posn d_model, d_model d_mlp -> batch posn d_mlp"
        ) + self.b_up

        # Element-wise multiply (gating)
        # hidden: [batch, posn, d_mlp]
        hidden = gate * up

        # Output projection
        # output: [batch, posn, d_model]
        output = einops.einsum(
            hidden, self.W_out,
            "batch posn d_mlp, d_mlp d_model -> batch posn d_model"
        ) + self.b_out

        return output


class MLPSwiGLUWithoutEinops(nn.Module):
    """MLP with SwiGLU activation (LLaMA style) without Einops"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_gate = nn.Parameter(torch.empty((cfg.d_model, cfg.d_mlp)))
        self.W_up = nn.Parameter(torch.empty((cfg.d_model, cfg.d_mlp)))
        self.W_out = nn.Parameter(torch.empty((cfg.d_mlp, cfg.d_model)))
        self.b_gate = nn.Parameter(torch.zeros(cfg.d_mlp))
        self.b_up = nn.Parameter(torch.zeros(cfg.d_mlp))
        self.b_out = nn.Parameter(torch.zeros(cfg.d_model))

        nn.init.normal_(self.W_gate, std=self.cfg.init_range)
        nn.init.normal_(self.W_up, std=self.cfg.init_range)
        nn.init.normal_(self.W_out, std=self.cfg.init_range)

    def forward(
        self, residual: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        # residual: [batch, posn, d_model]

        # Gate branch: Swish activation (SiLU)
        gate = torch.nn.functional.silu(
            torch.matmul(residual, self.W_gate) + self.b_gate
        )  # [batch, posn, d_mlp]

        # Up branch: linear
        up = torch.matmul(residual, self.W_up) + \
            self.b_up  # [batch, posn, d_mlp]

        # Element-wise multiply (gating)
        hidden = gate * up  # [batch, posn, d_mlp]

        # Output projection
        output = torch.matmul(hidden, self.W_out) + \
            self.b_out  # [batch, posn, d_model]

        return output


class MoEMLPBase(nn.Module):
    """Base class for Mixture of Experts MLP - contains shared forward logic"""
    
    def __init__(self, cfg, expert_class):
        super().__init__()
        self.cfg = cfg
        self.num_experts = cfg.num_experts
        self.num_experts_per_tok = cfg.num_experts_per_tok
        self.use_shared_experts = cfg.use_shared_experts if cfg.use_moe else False
        self.num_shared_experts = cfg.num_shared_experts if self.use_shared_experts else 0
        self.router_type = cfg.router_type if cfg.use_moe else RouterType.TOP_K
        
        # Router network: [d_model] -> [num_experts]
        self.router = nn.Linear(cfg.d_model, cfg.num_experts, bias=False)
        nn.init.normal_(self.router.weight, std=cfg.init_range)
        
        # Create expert MLPs
        self.experts = nn.ModuleList([
            expert_class(cfg) for _ in range(cfg.num_experts)
        ])
        
        # Shared experts (if enabled)
        if self.use_shared_experts:
            self.shared_experts = nn.ModuleList([
                expert_class(cfg) for _ in range(self.num_shared_experts)
            ])
        else:
            self.shared_experts = None
    
    def _compute_load_balancing_loss(
        self, router_probs: Float[Tensor, "batch seq_len num_experts"],
        top_k_indices: Float[Tensor, "batch seq_len num_experts_per_tok"],
        batch_size: int, seq_len: int
    ) -> Float[Tensor, ""]:
        """Compute load balancing auxiliary loss."""
        # Calculate fraction of tokens routed to each expert
        expert_usage = torch.zeros(self.num_experts, device=router_probs.device)
        for k in range(self.num_experts_per_tok):
            for expert_idx in range(self.num_experts):
                mask = (top_k_indices[:, :, k] == expert_idx)
                expert_usage[expert_idx] += mask.sum().float()
        expert_usage = expert_usage / (batch_size * seq_len * self.num_experts_per_tok)
        
        # Average routing probability
        avg_router_probs = router_probs.mean(dim=[0, 1])  # [num_experts]
        
        # Load balancing loss: num_experts * sum(P_i * f_i)
        return self.num_experts * torch.sum(avg_router_probs * expert_usage)
    
    def forward(
        self, residual: Float[Tensor, "batch posn d_model"]
    ) -> Tuple[Float[Tensor, "batch posn d_model"], Optional[Float[Tensor, ""]]]:
        """
        Forward pass through MoE MLP.
        Returns: (output, aux_loss)
        """
        batch_size, seq_len, d_model = residual.shape
        
        # Router logits: [batch, seq_len, num_experts]
        router_logits = self.router(residual)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts per token
        top_k_probs, top_k_indices = torch.topk(
            router_probs, k=self.num_experts_per_tok, dim=-1
        )  # [batch, seq_len, num_experts_per_tok]
        
        # Normalize top-k probabilities
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Initialize output
        output = torch.zeros_like(residual)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens that use this expert
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)  # [batch, seq_len]
            
            if expert_mask.any():
                # Get expert output
                expert_output = self.experts[expert_idx](residual)  # [batch, seq_len, d_model]
                
                # Get routing weights for this expert
                expert_weights = torch.zeros_like(router_probs[:, :, expert_idx])
                for k in range(self.num_experts_per_tok):
                    mask = (top_k_indices[:, :, k] == expert_idx)
                    expert_weights[mask] = top_k_probs[:, :, k][mask]
                
                # Weighted contribution
                output += expert_weights.unsqueeze(-1) * expert_output
        
        # Add shared experts contribution (if enabled)
        if self.use_shared_experts and self.shared_experts is not None:
            shared_output = torch.zeros_like(residual)
            for shared_expert in self.shared_experts:
                shared_output += shared_expert(residual)
            output += shared_output / len(self.shared_experts)
        
        # Calculate load balancing loss
        aux_loss = None
        if self.training:
            aux_loss = self._compute_load_balancing_loss(
                router_probs, top_k_indices, batch_size, seq_len
            )
        
        return output, aux_loss


class MoEMLPWithEinops(MoEMLPBase):
    """Mixture of Experts MLP with Einops"""
    
    def __init__(self, cfg):
        from config import Activation
        expert_class = MLPSwiGLUWithEinops if cfg.activation == Activation.SWIGLU else MLPWithEinops
        super().__init__(cfg, expert_class)


class MoEMLPWithoutEinops(MoEMLPBase):
    """Mixture of Experts MLP without Einops"""
    
    def __init__(self, cfg):
        from config import Activation
        expert_class = MLPSwiGLUWithoutEinops if cfg.activation == Activation.SWIGLU else MLPWithoutEinops
        super().__init__(cfg, expert_class)


def create_moe_mlp_layer(cfg, use_einops=True):
    """Factory function to create MoE MLP layer"""
    if not cfg.use_moe:
        return None
    
    if use_einops:
        return MoEMLPWithEinops(cfg)
    else:
        return MoEMLPWithoutEinops(cfg)


def create_mlp_layer(cfg, use_einops=True):
    """Factory function to create appropriate MLP layer based on activation config"""
    from config import Activation
    
    # Check if MoE is enabled
    if cfg.use_moe:
        return create_moe_mlp_layer(cfg, use_einops)

    if cfg.activation == Activation.SWIGLU:
        if use_einops:
            return MLPSwiGLUWithEinops(cfg)
        else:
            return MLPSwiGLUWithoutEinops(cfg)
    else:  # GELU
        if use_einops:
            return MLPWithEinops(cfg)
        else:
            return MLPWithoutEinops(cfg)
