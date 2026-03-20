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
    dropout: float = 0.0

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> "LoraConfig":
        return cls(
            rank=int(d.get("rank", 8)),
            alpha=float(d.get("alpha", 16.0)),
            target_blocks=d.get("target_blocks"),
            targets=d.get("targets", ["q", "k", "v", "o"]),
            dropout=float(d.get("dropout", 0.0)),
        )


class LoraLinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.scaling = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        x_lora = self.lora_dropout(x)
        lora_out = (x_lora @ self.lora_a.T) @ self.lora_b.T
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
            self.q_proj = LoraLinear(self.q_proj, config.rank, config.alpha, config.dropout)
        if "k" in targets:
            self.k_proj = LoraLinear(self.k_proj, config.rank, config.alpha, config.dropout)
        if "v" in targets:
            self.v_proj = LoraLinear(self.v_proj, config.rank, config.alpha, config.dropout)
        if "o" in targets:
            self.out_proj = LoraLinear(self.out_proj, config.rank, config.alpha, config.dropout)

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
        batch_size, seq_len, _ = query.shape

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, k.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, v.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attn_output)
        return output, None


def _freeze_all_params(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad_(False)


_ATTN_TARGETS = {"q", "k", "v", "o"}
_MLP_TARGETS = {"mlp_fc", "mlp_proj"}


def apply_lora_to_vision_encoder(model: nn.Module, config: LoraConfig) -> list[nn.Parameter]:
    _freeze_all_params(model)

    resblocks = model.visual.transformer.resblocks
    target_indices = set(config.target_blocks) if config.target_blocks is not None else set(range(len(resblocks)))
    targets = set(config.targets)
    has_attn_targets = bool(targets & _ATTN_TARGETS)

    for i, block in enumerate(resblocks):
        if i not in target_indices:
            continue

        if has_attn_targets and isinstance(block.attn, nn.MultiheadAttention):
            block.attn = LoraMultiheadAttention(block.attn, config)

        if "mlp_fc" in targets and isinstance(block.mlp.c_fc, nn.Linear):
            block.mlp.c_fc = LoraLinear(block.mlp.c_fc, config.rank, config.alpha, config.dropout)

        if "mlp_proj" in targets and isinstance(block.mlp.c_proj, nn.Linear):
            block.mlp.c_proj = LoraLinear(block.mlp.c_proj, config.rank, config.alpha, config.dropout)

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
        if isinstance(block.attn, LoraMultiheadAttention):
            lora_attn = block.attn
            embed_dim = lora_attn.embed_dim
            num_heads = lora_attn.num_heads

            q_linear = lora_attn.q_proj.merge() if isinstance(lora_attn.q_proj, LoraLinear) else lora_attn.q_proj
            k_linear = lora_attn.k_proj.merge() if isinstance(lora_attn.k_proj, LoraLinear) else lora_attn.k_proj
            v_linear = lora_attn.v_proj.merge() if isinstance(lora_attn.v_proj, LoraLinear) else lora_attn.v_proj
            o_linear = lora_attn.out_proj.merge() if isinstance(lora_attn.out_proj, LoraLinear) else lora_attn.out_proj

            restored = nn.MultiheadAttention(embed_dim, num_heads, bias=q_linear.bias is not None, batch_first=True)
            restored = restored.to(q_linear.weight.device)
            with torch.no_grad():
                restored.in_proj_weight.copy_(torch.cat([q_linear.weight, k_linear.weight, v_linear.weight], dim=0))
                if restored.in_proj_bias is not None:
                    restored.in_proj_bias.copy_(torch.cat([q_linear.bias, k_linear.bias, v_linear.bias], dim=0))
                restored.out_proj.weight.copy_(o_linear.weight)
                if restored.out_proj.bias is not None and o_linear.bias is not None:
                    restored.out_proj.bias.copy_(o_linear.bias)

            block.attn = restored

        if isinstance(block.mlp.c_fc, LoraLinear):
            block.mlp.c_fc = block.mlp.c_fc.merge()

        if isinstance(block.mlp.c_proj, LoraLinear):
            block.mlp.c_proj = block.mlp.c_proj.merge()
