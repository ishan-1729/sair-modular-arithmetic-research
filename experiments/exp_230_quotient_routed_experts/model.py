"""Learned quotient router over learned width-32 residue experts."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import Tensor, nn


class ResidualBlock(nn.Module):
    def __init__(self, hidden: int, dilation: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(1, hidden)
        self.conv1 = nn.Conv1d(hidden, hidden, 3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(1, hidden)
        self.conv2 = nn.Conv1d(hidden, hidden, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.conv2(F.silu(self.norm2(self.conv1(F.silu(self.norm1(inputs))))))


class QuotientRoutedExperts(nn.Module):
    """Shared learned body with a learned router and three learned output heads."""

    def __init__(self, width: int = 32, hidden: int = 48, blocks: int = 7, experts: int = 3) -> None:
        super().__init__()
        self.width = width
        self.hidden = hidden
        self.experts = experts
        self.routing_mode = "predicted"
        self.stem = nn.Conv1d(6, hidden, 1)
        self.position = nn.Parameter(torch.zeros(1, hidden, width))
        self.modulus_context = nn.Sequential(nn.Linear(width, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        cycle = max(1, math.ceil(math.log2(width)))
        self.body = nn.ModuleList(ResidualBlock(hidden, 1 << (index % cycle)) for index in range(blocks))
        self.head_norm = nn.GroupNorm(1, hidden)
        self.local_heads = nn.ModuleList(nn.Conv1d(hidden, 1, 1) for _ in range(experts))
        self.global_heads = nn.ModuleList(nn.Linear(hidden * width, width) for _ in range(experts))
        self.quotient_head = nn.Linear(hidden * width, experts)

    def encode(self, features: Tensor) -> Tensor:
        context = self.modulus_context(features[:, 2, :])[:, :, None]
        hidden = self.stem(features) + self.position + context
        for block in self.body:
            hidden = block(hidden)
        return F.silu(self.head_norm(hidden))

    def expert_logits(self, hidden: Tensor) -> Tensor:
        flat = hidden.flatten(1)
        return torch.stack([local(hidden).squeeze(1) + global_head(flat) for local, global_head in zip(self.local_heads, self.global_heads, strict=True)], dim=1)

    def forward(self, features: Tensor, teacher_route: Tensor | None = None) -> tuple[Tensor, Tensor]:
        hidden = self.encode(features)
        flat = hidden.flatten(1)
        router_logits = self.quotient_head(flat)
        candidates = self.expert_logits(hidden)
        if teacher_route is not None:
            route = teacher_route
        elif self.routing_mode == "predicted":
            route = router_logits.argmax(dim=1)
        elif self.routing_mode == "expert0":
            route = torch.zeros(len(features), dtype=torch.long, device=features.device)
        elif self.routing_mode == "rotated":
            route = (router_logits.argmax(dim=1) + 1) % self.experts
        elif self.routing_mode == "average":
            return candidates.mean(dim=1), router_logits
        else:
            raise ValueError(f"unknown routing mode {self.routing_mode!r}")
        selected = candidates.gather(1, route[:, None, None].expand(-1, 1, self.width)).squeeze(1)
        return selected, router_logits

    def load_single_head_checkpoint(self, path: Path) -> None:
        source = load_file(str(path))
        target = self.state_dict()
        for key, value in source.items():
            if key in target and target[key].shape == value.shape:
                target[key].copy_(value)
        for index in range(self.experts):
            target[f"local_heads.{index}.weight"].copy_(source["local_head.weight"])
            target[f"local_heads.{index}.bias"].copy_(source["local_head.bias"])
            target[f"global_heads.{index}.weight"].copy_(source["global_head.weight"])
            target[f"global_heads.{index}.bias"].copy_(source["global_head.bias"])
        self.load_state_dict(target)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def expert_pairwise_distance(self) -> dict[str, float]:
        distances = {}
        for left in range(self.experts):
            for right in range(left + 1, self.experts):
                local = (self.local_heads[left].weight - self.local_heads[right].weight).norm()
                global_distance = (self.global_heads[left].weight - self.global_heads[right].weight).norm()
                distances[f"{left}-{right}"] = float((local + global_distance).detach())
        return distances
