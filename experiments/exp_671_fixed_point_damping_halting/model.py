"""Bounded, exact-initialized Fixed-Point Reasoner transfer for EXP671."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from experiments.exp_220_width_independent_cellular.model import SharedDilatedConvGRU


ROUTES = (
    "full_fpr",
    "stabilization_fixed",
    "damped_fixed",
    "halt_without_damping",
    "plain_fixed",
    "proposal_only",
    "reset",
    "randomized",
)
TRAINED_ROUTES = ("full_fpr", "stabilization_fixed", "plain_fixed")


@dataclass
class FPROutput:
    logits: Tensor
    quotient_logits: Tensor
    state: Tensor
    proposal: Tensor
    relative_residual: Tensor
    iterations: Tensor
    final_damping: Tensor
    hard_max: Tensor
    deep_logits: tuple[Tensor, ...] = ()
    deep_quotient_logits: tuple[Tensor, ...] = ()


class PreNormSharedRefinement(nn.Module):
    """One shared PreNorm sublayer with per-channel learned residual scaling."""

    def __init__(self, hidden: int, inner: int) -> None:
        super().__init__()
        channels = 2 * hidden + 6
        self.norm = nn.GroupNorm(1, channels)
        self.inner = nn.Conv1d(channels, inner, 3, padding=1)
        self.output = nn.Conv1d(inner, hidden, 1)
        self.residual_scale = nn.Parameter(torch.zeros(1, hidden, 1))
        nn.init.kaiming_uniform_(self.inner.weight, nonlinearity="relu")
        nn.init.zeros_(self.inner.bias)
        nn.init.kaiming_uniform_(self.output.weight, nonlinearity="linear")
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        state: Tensor,
        proposal: Tensor,
        features: Tensor,
        *,
        plain: bool = False,
        reset_scale: bool = False,
    ) -> Tensor:
        if plain:
            joined = torch.cat((state, torch.zeros_like(proposal), torch.zeros_like(features)), 1)
        else:
            # The immutable proposal and original legal inputs are reinjected every iteration.
            joined = self.norm(torch.cat((state, proposal, features), 1))
        delta = torch.tanh(self.output(F.silu(self.inner(joined))))
        scale = torch.zeros_like(self.residual_scale) if reset_scale else torch.tanh(self.residual_scale)
        return state + scale * delta


class FixedPointReasonerCell(nn.Module):
    """Frozen EXP609 proposal followed by finite shared fixed-point refinement."""

    def __init__(
        self,
        *,
        hidden: int = 92,
        cycles: int = 2,
        operator_inner: int = 64,
        hard_max_iterations: int = 6,
        training_iterations: int = 6,
        tbptt_window: int = 2,
        initial_damping: float = 1.0,
        damping_decay: float = 0.5,
        patience: int = 2,
        residual_epsilon: float = 1e-9,
    ) -> None:
        super().__init__()
        self.base = SharedDilatedConvGRU(width=256, hidden=hidden, cycles=cycles)
        self.refinement = PreNormSharedRefinement(hidden, operator_inner)
        self.hard_max_iterations = int(hard_max_iterations)
        self.training_iterations = int(training_iterations)
        self.tbptt_window = int(tbptt_window)
        self.initial_damping = float(initial_damping)
        self.damping_decay = float(damping_decay)
        self.patience = int(patience)
        self.residual_epsilon = float(residual_epsilon)
        if self.hard_max_iterations < 1 or self.training_iterations < 1:
            raise ValueError("iteration bounds must be positive")
        if self.training_iterations % self.tbptt_window:
            raise ValueError("training iterations must divide into complete TBPTT windows")
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

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
        pooled = torch.cat((normalized.mean(-1), normalized.amax(-1)), 1)
        return logits, self.base.quotient_head(pooled)

    def operator(
        self,
        state: Tensor,
        proposal: Tensor,
        features: Tensor,
        *,
        plain: bool = False,
        reset_scale: bool = False,
    ) -> Tensor:
        return self.refinement(state, proposal, features, plain=plain, reset_scale=reset_scale)

    def residual(self, state: Tensor, proposed: Tensor) -> Tensor:
        numerator = (proposed - state).flatten(1).abs().amax(1)
        denominator = proposed.flatten(1).abs().amax(1).clamp_min(self.residual_epsilon)
        return numerator / denominator

    def _fixed(
        self,
        features: Tensor,
        route: str,
        *,
        training: bool,
    ) -> FPROutput:
        proposal = self.proposal_state(features)
        state = proposal
        count = self.training_iterations if training else self.hard_max_iterations
        damped = route in ("full_fpr", "damped_fixed")
        plain = route == "plain_fixed"
        reset_scale = route == "reset"
        eta = torch.full((state.shape[0],), self.initial_damping, device=state.device, dtype=state.dtype)
        best = torch.full_like(eta, torch.inf)
        stale = torch.zeros_like(eta, dtype=torch.long)
        deep_logits: list[Tensor] = []
        deep_quotient: list[Tensor] = []
        residual = torch.zeros_like(eta)
        for index in range(count):
            proposed = self.operator(state, proposal, features, plain=plain, reset_scale=reset_scale)
            residual = self.residual(state, proposed)
            if damped:
                state = eta[:, None, None] * proposed + (1.0 - eta[:, None, None]) * state
                improved = residual < best
                best = torch.minimum(best, residual)
                stale = torch.where(improved, torch.zeros_like(stale), stale + 1)
                decay = stale >= self.patience
                eta = torch.where(decay, eta * self.damping_decay, eta)
                stale = torch.where(decay, torch.zeros_like(stale), stale)
            else:
                state = proposed
            if training and (index + 1) % self.tbptt_window == 0:
                logits, quotient = self.decode(state)
                deep_logits.append(logits)
                deep_quotient.append(quotient)
                if index + 1 < count:
                    state = state.detach()
        logits, quotient = self.decode(state)
        iterations = torch.full_like(residual, count, dtype=torch.long)
        return FPROutput(
            logits,
            quotient,
            state,
            proposal,
            residual,
            iterations,
            eta,
            torch.ones_like(iterations, dtype=torch.bool),
            tuple(deep_logits),
            tuple(deep_quotient),
        )

    def _adaptive(
        self,
        features: Tensor,
        route: str,
        threshold: float,
        minimum_iterations: int,
    ) -> FPROutput:
        proposal = self.proposal_state(features)
        state = proposal.clone()
        batch = state.shape[0]
        eta = torch.full((batch,), self.initial_damping, device=state.device, dtype=state.dtype)
        best = torch.full_like(eta, torch.inf)
        residual = torch.full_like(eta, torch.inf)
        stale = torch.zeros(batch, device=state.device, dtype=torch.long)
        iterations = torch.zeros(batch, device=state.device, dtype=torch.long)
        active = torch.arange(batch, device=state.device)
        damped = route == "full_fpr"
        for index in range(self.hard_max_iterations):
            if active.numel() == 0:
                break
            current = state.index_select(0, active)
            active_proposal = proposal.index_select(0, active)
            active_features = features.index_select(0, active)
            proposed = self.operator(current, active_proposal, active_features)
            current_residual = self.residual(current, proposed)
            active_eta = eta.index_select(0, active)
            updated = (active_eta[:, None, None] * proposed
                       + (1.0 - active_eta[:, None, None]) * current) if damped else proposed
            state.index_copy_(0, active, updated)
            residual.index_copy_(0, active, current_residual)
            iterations.index_fill_(0, active, index + 1)
            if damped:
                active_best = best.index_select(0, active)
                active_stale = stale.index_select(0, active)
                improved = current_residual < active_best
                active_best = torch.minimum(active_best, current_residual)
                active_stale = torch.where(improved, torch.zeros_like(active_stale), active_stale + 1)
                decay = active_stale >= self.patience
                next_eta = torch.where(decay, active_eta * self.damping_decay, active_eta)
                active_stale = torch.where(decay, torch.zeros_like(active_stale), active_stale)
                best.index_copy_(0, active, active_best)
                stale.index_copy_(0, active, active_stale)
                eta.index_copy_(0, active, next_eta)
            converged = current_residual < float(threshold)
            if index + 1 < int(minimum_iterations):
                converged = torch.zeros_like(converged)
            active = active[~converged]
        logits, quotient = self.decode(state)
        hard_max = iterations.eq(self.hard_max_iterations)
        return FPROutput(logits, quotient, state, proposal, residual, iterations, eta, hard_max)

    def forward_train(self, features: Tensor, *, route: str) -> FPROutput:
        if route not in TRAINED_ROUTES:
            raise ValueError("training requires a frozen trained route")
        return self._fixed(features, route, training=True)

    def forward(
        self,
        features: Tensor,
        *,
        route: str = "full_fpr",
        threshold: float = 0.01,
        minimum_iterations: int = 2,
    ) -> FPROutput:
        if route not in ROUTES:
            raise ValueError(f"unknown route: {route}")
        if route == "proposal_only":
            proposal = self.proposal_state(features)
            logits, quotient = self.decode(proposal)
            zeros = torch.zeros(proposal.shape[0], device=proposal.device)
            return FPROutput(logits, quotient, proposal, proposal, zeros, zeros.long(), zeros, zeros.bool())
        if route in ("full_fpr", "halt_without_damping"):
            return self._adaptive(features, route, threshold, minimum_iterations)
        return self._fixed(features, route, training=False)

    def randomized_intervention(self, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for parameter in self.refinement.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator,
                                             dtype=parameter.dtype, device=parameter.device) * 0.02)

    def reset_intervention(self) -> None:
        with torch.no_grad():
            self.refinement.residual_scale.zero_()
