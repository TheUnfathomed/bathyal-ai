"""LoRA adapters for OpenCLIP ViT vision encoders."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(slots=True)
class LoraConfig:
    rank: int = 8
    alpha: float = 16.0
    target_blocks: list[int] | None = None
    targets: list[str] = field(default_factory=lambda: ["q", "k", "v", "o"])


class LoraLinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.scaling = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (x @ self.lora_a.T) @ self.lora_b.T
        return base_out + lora_out * self.scaling

    def merge(self) -> nn.Linear:
        with torch.no_grad():
            merged_weight = self.base.weight + (self.lora_b @ self.lora_a) * self.scaling
            self.base.weight.copy_(merged_weight)
        return self.base


class LoraMultiheadAttention(nn.Module):
    def __init__(self, original: nn.MultiheadAttention, config: LoraConfig) -> None:
        super().__init__()
        embed_dim = original.embed_dim
        num_heads = original.num_heads
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.config = config

        in_proj_weight = original.in_proj_weight.data
        in_proj_bias = original.in_proj_bias.data if original.in_proj_bias is not None else None

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.q_proj.weight = nn.Parameter(in_proj_weight[:embed_dim].clone())
        self.k_proj.weight = nn.Parameter(in_proj_weight[embed_dim : 2 * embed_dim].clone())
        self.v_proj.weight = nn.Parameter(in_proj_weight[2 * embed_dim :].clone())

        if in_proj_bias is not None:
            self.q_proj.bias = nn.Parameter(in_proj_bias[:embed_dim].clone())
            self.k_proj.bias = nn.Parameter(in_proj_bias[embed_dim : 2 * embed_dim].clone())
            self.v_proj.bias = nn.Parameter(in_proj_bias[2 * embed_dim :].clone())
        else:
            self.q_proj.bias = None
            self.k_proj.bias = None
            self.v_proj.bias = None

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj.weight = nn.Parameter(original.out_proj.weight.data.clone())
        if original.out_proj.bias is not None:
            self.out_proj.bias = nn.Parameter(original.out_proj.bias.data.clone())
        else:
            self.out_proj.bias = None

        targets = set(config.targets)
        if "q" in targets:
            self.q_proj = LoraLinear(self.q_proj, config.rank, config.alpha)
        if "k" in targets:
            self.k_proj = LoraLinear(self.k_proj, config.rank, config.alpha)
        if "v" in targets:
            self.v_proj = LoraLinear(self.v_proj, config.rank, config.alpha)
        if "o" in targets:
            self.out_proj = LoraLinear(self.out_proj, config.rank, config.alpha)

        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            if not isinstance(proj, LoraLinear):
                proj.weight.requires_grad_(False)
                if proj.bias is not None:
                    proj.bias.requires_grad_(False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        need_weights: bool = False,
        attn_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        seq_len, batch_size, _ = query.shape

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(seq_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(k.size(0), batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(v.size(0), batch_size * self.num_heads, self.head_dim).transpose(0, 1)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0
        )

        attn_output = attn_output.transpose(0, 1).contiguous().view(seq_len, batch_size, self.embed_dim)
        output = self.out_proj(attn_output)
        return output, None


def _freeze_all_params(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad_(False)


def apply_lora_to_vision_encoder(model: nn.Module, config: LoraConfig) -> list[nn.Parameter]:
    _freeze_all_params(model)

    resblocks = model.visual.transformer.resblocks
    target_indices = set(config.target_blocks) if config.target_blocks is not None else set(range(len(resblocks)))

    for i, block in enumerate(resblocks):
        if i not in target_indices:
            continue
        original_attn = block.attn
        if not isinstance(original_attn, nn.MultiheadAttention):
            continue
        lora_attn = LoraMultiheadAttention(original_attn, config)
        block.attn = lora_attn

    lora_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            lora_params.append(param)
    return lora_params


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach()
        for name, param in model.named_parameters()
        if "lora_a" in name or "lora_b" in name
    }


def load_lora_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    model_state = model.state_dict()
    for key, value in state_dict.items():
        if key in model_state:
            model_state[key] = value
    model.load_state_dict(model_state, strict=False)


def merge_lora_weights(model: nn.Module) -> None:
    for block in model.visual.transformer.resblocks:
        if not isinstance(block.attn, LoraMultiheadAttention):
            continue
        lora_attn = block.attn
        embed_dim = lora_attn.embed_dim
        num_heads = lora_attn.num_heads

        q_linear = lora_attn.q_proj.merge() if isinstance(lora_attn.q_proj, LoraLinear) else lora_attn.q_proj
        k_linear = lora_attn.k_proj.merge() if isinstance(lora_attn.k_proj, LoraLinear) else lora_attn.k_proj
        v_linear = lora_attn.v_proj.merge() if isinstance(lora_attn.v_proj, LoraLinear) else lora_attn.v_proj
        o_linear = lora_attn.out_proj.merge() if isinstance(lora_attn.out_proj, LoraLinear) else lora_attn.out_proj

        restored = nn.MultiheadAttention(embed_dim, num_heads, bias=q_linear.bias is not None)
        restored = restored.to(q_linear.weight.device)
        with torch.no_grad():
            restored.in_proj_weight.copy_(torch.cat([q_linear.weight, k_linear.weight, v_linear.weight], dim=0))
            if restored.in_proj_bias is not None:
                restored.in_proj_bias.copy_(torch.cat([q_linear.bias, k_linear.bias, v_linear.bias], dim=0))
            restored.out_proj.weight.copy_(o_linear.weight)
            if restored.out_proj.bias is not None and o_linear.bias is not None:
                restored.out_proj.bias.copy_(o_linear.bias)

        block.attn = restored
