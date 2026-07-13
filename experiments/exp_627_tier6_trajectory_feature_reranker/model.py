"""Compact trained candidate scorer for conditional beam inference."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TrajectoryFeatureReranker(nn.Module):
    """Score beam candidates from learned-search and public-input features."""

    def __init__(self, feature_count: int, hidden: int = 64) -> None:
        super().__init__()
        self.register_buffer("feature_mean", torch.zeros(feature_count))
        self.register_buffer("feature_scale", torch.ones(feature_count))
        self.network = nn.Sequential(
            nn.Linear(feature_count, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.greedy_prior = nn.Parameter(torch.tensor(2.0))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    @torch.no_grad()
    def set_normalization(self, training_features: Tensor) -> None:
        flat = training_features.reshape(-1, training_features.shape[-1])
        self.feature_mean.copy_(flat.mean(dim=0))
        self.feature_scale.copy_(flat.std(dim=0).clamp_min(1e-5))

    def forward(self, features: Tensor) -> Tensor:
        standardized = (features - self.feature_mean) / self.feature_scale
        learned = self.network(standardized).squeeze(-1)
        equals_greedy = features[..., 3]
        return learned + self.greedy_prior * equals_greedy

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
