"""Bounded-faithful GRAM posterior-to-prior guidance for EXP-666."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Independent, Normal, kl_divergence

from experiments.exp_220_width_independent_cellular.model import SharedDilatedConvGRU


ROUTES = (
    "proposal_only",
    "gram_full",
    "deterministic_mean",
    "guidance_only_no_kl",
    "stochasticity_only_no_posterior",
    "direct_prediction",
    "randomized_guidance",
)
TRAINED_ROUTES = ROUTES[1:-1]


@dataclass
class GRAMOutput:
    logits: Tensor
    quotient_logits: Tensor
    state: Tensor
    proposal: Tensor
    relative_residual: Tensor
    iterations: Tensor
    kl: Tensor
    prior_mean: Tensor
    prior_scale: Tensor
    posterior_mean: Tensor | None
    posterior_scale: Tensor | None
    supervision_logits: tuple[Tensor, ...]


class ZeroResidualConv(nn.Module):
    """Proposal-injected deterministic low-state refinement."""

    def __init__(self, hidden: int, inner: int, scale: float) -> None:
        super().__init__()
        self.scale = float(scale)
        self.norm = nn.GroupNorm(1, 2 * hidden + 6)
        self.inner = nn.Conv1d(2 * hidden + 6, inner, 3, padding=1)
        self.output = nn.Conv1d(inner, hidden, 1)
        nn.init.kaiming_uniform_(self.inner.weight, nonlinearity="relu")
        nn.init.zeros_(self.inner.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, low: Tensor, proposal: Tensor, features: Tensor) -> Tensor:
        joined = self.norm(torch.cat((low, proposal, features), dim=1))
        return proposal + self.scale * torch.tanh(self.output(F.silu(self.inner(joined))))


class GRAMCell(nn.Module):
    """Frozen decoded proposal with deterministic low and Gaussian high state.

    Target access exists only in ``forward_train``.  ``forward`` constructs only
    the target-free learned prior and exactly one trajectory.
    """

    def __init__(
        self,
        *,
        hidden: int = 92,
        cycles: int = 2,
        high_dim: int = 32,
        low_inner: int = 64,
        low_scale: float = 0.10,
        guidance_scale: float = 0.10,
        low_updates: int = 2,
        transitions_per_supervision: int = 2,
        supervision_steps: int = 2,
        minimum_scale: float = 1e-3,
        initial_log_scale: float = -3.0,
    ) -> None:
        super().__init__()
        self.base = SharedDilatedConvGRU(width=256, hidden=hidden, cycles=cycles)
        self.low_refiner = ZeroResidualConv(hidden, low_inner, low_scale)
        pooled = 2 * hidden + high_dim
        self.high_update = nn.Sequential(nn.Linear(pooled, 2 * high_dim), nn.SiLU(), nn.Linear(2 * high_dim, high_dim))
        self.prior_mean_head = nn.Linear(high_dim, high_dim)
        self.prior_log_scale_head = nn.Linear(high_dim, high_dim)
        self.target_encoder = nn.Sequential(nn.Conv1d(1, 16, 3, padding=1), nn.SiLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.posterior_mean_delta = nn.Linear(high_dim + 16, high_dim)
        self.posterior_log_scale_delta = nn.Linear(high_dim + 16, high_dim)
        self.guidance_output = nn.Linear(high_dim, hidden)
        self.direct_output = nn.Conv1d(hidden, hidden, 1)
        self.guidance_scale = float(guidance_scale)
        self.low_updates = int(low_updates)
        self.transitions_per_supervision = int(transitions_per_supervision)
        self.supervision_steps = int(supervision_steps)
        self.minimum_scale = float(minimum_scale)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        for module in (self.high_update[-1], self.prior_mean_head, self.posterior_mean_delta, self.posterior_log_scale_delta, self.guidance_output, self.direct_output):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
        nn.init.zeros_(self.prior_log_scale_head.weight)
        nn.init.constant_(self.prior_log_scale_head.bias, float(initial_log_scale))

    def load_source(self, state: dict[str, Tensor]) -> None:
        self.base.load_state_dict(state, strict=True)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def proposal_state(self, features: Tensor) -> Tensor:
        self.base.width = features.shape[-1]
        hidden = self.base.stem(features)
        for dilation in self.base.dilation_schedule(features.shape[-1]):
            hidden = self.base.recurrent_update(hidden, features, dilation)
        return hidden

    def decode(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        normalized = F.silu(self.base.output_norm(hidden))
        logits = self.base.local_head(normalized).squeeze(1)
        pooled = torch.cat((normalized.mean(dim=-1), normalized.amax(dim=-1)), dim=1)
        return logits, self.base.quotient_head(pooled)

    def _prior(self, deterministic_high: Tensor) -> Independent:
        mean = self.prior_mean_head(deterministic_high)
        scale = self.prior_log_scale_head(deterministic_high).exp().clamp_min(self.minimum_scale)
        return Independent(Normal(mean, scale), 1)

    def _posterior(self, deterministic_high: Tensor, target_bits: Tensor, prior: Independent) -> Independent:
        encoded = self.target_encoder(target_bits.unsqueeze(1))
        joined = torch.cat((deterministic_high, encoded), dim=1)
        mean = prior.base_dist.loc + self.posterior_mean_delta(joined)
        log_scale = prior.base_dist.scale.log() + self.posterior_log_scale_delta(joined)
        return Independent(Normal(mean, log_scale.exp().clamp_min(self.minimum_scale)), 1)

    @staticmethod
    def _rsample(distribution: Independent, generator: torch.Generator | None) -> Tensor:
        if generator is None:
            return distribution.rsample()
        seed = int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item())
        devices = [] if distribution.base_dist.loc.device.type == "cpu" else [distribution.base_dist.loc.device]
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            return distribution.rsample()

    def _transition(
        self,
        low: Tensor,
        high: Tensor,
        proposal: Tensor,
        features: Tensor,
        route: str,
        generator: torch.Generator | None,
        target_bits: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Independent, Independent | None]:
        for _ in range(self.low_updates):
            low = self.low_refiner(low, proposal, features)
        pooled = torch.cat((low.mean(-1), low.amax(-1), high), dim=1)
        deterministic_high = torch.tanh(self.high_update(pooled))
        prior = self._prior(deterministic_high)
        posterior = None
        if target_bits is not None and route in {"gram_full", "deterministic_mean", "guidance_only_no_kl"}:
            posterior = self._posterior(deterministic_high, target_bits, prior)
        distribution = posterior if posterior is not None else prior
        if route in {"deterministic_mean", "guidance_only_no_kl"}:
            high = distribution.base_dist.loc
        elif route == "stochasticity_only_no_posterior":
            zero_mean = torch.zeros_like(prior.base_dist.loc)
            high = self._rsample(Independent(Normal(zero_mean, prior.base_dist.scale), 1), generator)
        else:
            high = self._rsample(distribution, generator)
        divergence = kl_divergence(posterior, prior) if posterior is not None else torch.zeros(high.shape[0], device=high.device)
        return low, high, divergence, prior, posterior

    def _run(
        self,
        features: Tensor,
        route: str,
        generator: torch.Generator | None,
        target_bits: Tensor | None,
    ) -> GRAMOutput:
        if route not in ROUTES:
            raise ValueError(f"unknown route: {route}")
        proposal = self.proposal_state(features)
        if route == "proposal_only":
            logits, quotient = self.decode(proposal)
            zero = torch.zeros(proposal.shape[0], device=proposal.device)
            empty = torch.zeros((proposal.shape[0], self.prior_mean_head.out_features), device=proposal.device)
            return GRAMOutput(logits, quotient, proposal, proposal, zero, zero.long(), zero, empty, empty, None, None, (logits,))
        if route == "direct_prediction":
            state = proposal + self.guidance_scale * torch.tanh(self.direct_output(proposal))
            logits, quotient = self.decode(state)
            zero = torch.zeros(proposal.shape[0], device=proposal.device)
            empty = torch.zeros((proposal.shape[0], self.prior_mean_head.out_features), device=proposal.device)
            return GRAMOutput(logits, quotient, state, proposal, (state-proposal).flatten(1).norm(dim=1), torch.ones_like(zero, dtype=torch.long), zero, empty, empty, None, None, (logits,) * self.supervision_steps)
        low = proposal
        high = torch.zeros((proposal.shape[0], self.prior_mean_head.out_features), device=proposal.device, dtype=proposal.dtype)
        all_logits: list[Tensor] = []
        all_kl: list[Tensor] = []
        prior = posterior = None
        for supervision in range(self.supervision_steps):
            for transition in range(self.transitions_per_supervision):
                low, high, divergence, prior, posterior = self._transition(low, high, proposal, features, route, generator, target_bits)
                all_kl.append(divergence)
                if transition + 1 < self.transitions_per_supervision:
                    low, high = low.detach(), high.detach()
            guided = low + self.guidance_scale * torch.tanh(self.guidance_output(high)).unsqueeze(-1)
            all_logits.append(self.decode(guided)[0])
            if supervision + 1 < self.supervision_steps:
                low, high = low.detach(), high.detach()
        state = low + self.guidance_scale * torch.tanh(self.guidance_output(high)).unsqueeze(-1)
        logits, quotient = self.decode(state)
        assert prior is not None
        prior_mean, prior_scale = prior.base_dist.loc, prior.base_dist.scale
        posterior_mean = None if posterior is None else posterior.base_dist.loc
        posterior_scale = None if posterior is None else posterior.base_dist.scale
        guidance_norm = (state - proposal).flatten(1).norm(dim=1) / proposal.flatten(1).norm(dim=1).clamp_min(1e-9)
        return GRAMOutput(logits, quotient, state, proposal, guidance_norm, torch.full_like(guidance_norm, self.supervision_steps * self.transitions_per_supervision, dtype=torch.long), torch.stack(all_kl).mean(0), prior_mean, prior_scale, posterior_mean, posterior_scale, tuple(all_logits))

    def forward(self, features: Tensor, *, route: str = "gram_full", generator: torch.Generator | None = None) -> GRAMOutput:
        """Run one target-free prior trajectory; posterior inputs are impossible."""
        return self._run(features, route, generator, None)

    def forward_train(self, features: Tensor, target_bits: Tensor, *, route: str = "gram_full", generator: torch.Generator | None = None) -> GRAMOutput:
        return self._run(features, route, generator, target_bits)

    def randomized_guidance(self, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                if name.startswith("base."):
                    continue
                scale = parameter.detach().std().clamp_min(1e-3)
                parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype, device=parameter.device) * scale)
