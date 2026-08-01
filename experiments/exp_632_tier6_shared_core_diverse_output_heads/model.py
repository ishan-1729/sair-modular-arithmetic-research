"""Exact-initialized diverse residual heads over a frozen cellular core."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SharedCoreDiverseHeads(nn.Module):
    """Reuse one recurrent core and majority-vote complementary residual heads."""

    def __init__(self, base: nn.Module, heads: int = 3, mask_seed: int = 0) -> None:
        super().__init__()
        if heads != 3:
            raise ValueError("the preregistered design requires exactly three heads")
        self.base = base
        self.width = int(base.width)
        self.hidden = int(base.hidden)
        self.heads = int(heads)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.residual_heads = nn.ModuleList(
            nn.Conv1d(self.hidden, 1, 1) for _ in range(self.heads)
        )
        for head in self.residual_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        generator = torch.Generator().manual_seed(mask_seed)
        groups = torch.arange(self.hidden) % self.heads
        groups = groups[torch.randperm(self.hidden, generator=generator)]
        masks = torch.stack(
            [(groups != omitted).float() for omitted in range(self.heads)], dim=0
        ).unsqueeze(-1)
        self.register_buffer("channel_masks", masks)

    def _shared_hidden(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.base.width = features.shape[-1]
        hidden = self.base.stem(features)
        for dilation in self.base.dilation_schedule(features.shape[-1]):
            hidden = self.base.recurrent_update(hidden, features, dilation)
        normalized = F.silu(self.base.output_norm(hidden))
        pooled = torch.cat(
            (normalized.mean(dim=-1), normalized.amax(dim=-1)), dim=1
        )
        return normalized, self.base.quotient_head(pooled)

    def forward_heads(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized, quotient_logits = self._shared_hidden(features)
        base_logits = self.base.local_head(normalized).squeeze(1)
        head_logits = torch.stack(
            [
                base_logits + head(normalized * self.channel_masks[index]).squeeze(1)
                for index, head in enumerate(self.residual_heads)
            ],
            dim=1,
        )
        return head_logits, quotient_logits

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        head_logits, quotient_logits = self.forward_heads(features)
        signed_vote = head_logits.ge(0).sum(dim=1).mul(2).sub(self.heads)
        return signed_vote.to(head_logits.dtype), quotient_logits

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

