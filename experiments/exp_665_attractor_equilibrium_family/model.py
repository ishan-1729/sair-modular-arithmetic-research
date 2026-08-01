"""Exact-initialized proposal-conditioned attractor models for EXP-665."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchdeq import get_deq

from experiments.exp_220_width_independent_cellular.model import SharedDilatedConvGRU


ROUTES = (
    "proposal_only",
    "fixed_depth_shared",
    "plain_picard_phantom",
    "anderson_phantom",
    "anderson_implicit",
    "eqr_finite",
)
TRAINED_ROUTES = ROUTES[1:]


@dataclass
class AttractorOutput:
    logits: Tensor
    quotient_logits: Tensor
    state: Tensor
    proposal: Tensor
    relative_residual: Tensor
    iterations: Tensor


class ProposalCorrection(nn.Module):
    """Bounded learned correction with persistent proposal injection."""

    def __init__(self, hidden: int, correction_hidden: int, scale: float) -> None:
        super().__init__()
        self.scale = float(scale)
        self.input_norm = nn.GroupNorm(1, 2 * hidden + 6)
        self.inner = nn.Conv1d(2 * hidden + 6, correction_hidden, 3, padding=1)
        self.output = nn.Conv1d(correction_hidden, hidden, 1)
        nn.init.kaiming_uniform_(self.inner.weight, nonlinearity="relu")
        nn.init.zeros_(self.inner.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, state: Tensor, proposal: Tensor, features: Tensor) -> Tensor:
        joined = self.input_norm(torch.cat((state, proposal, features), dim=1))
        delta = self.output(F.silu(self.inner(joined)))
        return proposal + self.scale * torch.tanh(delta)


class AttractorCell(nn.Module):
    """Frozen EXP609 proposal plus one matched learned equilibrium operator."""

    def __init__(
        self,
        *,
        hidden: int = 92,
        cycles: int = 2,
        correction_hidden: int = 64,
        correction_scale: float = 0.25,
        fixed_depth: int = 4,
        solver_max_iterations: int = 6,
        solver_tolerance: float = 1e-3,
        anderson_history: int = 4,
        anderson_regularization: float = 1e-4,
        implicit_backward_iterations: int = 6,
        implicit_backward_tolerance: float = 1e-4,
        eqr_damping: float = 0.05,
        eqr_noise: float = 0.01,
        eqr_initialization_std: float = 0.05,
        eqr_training_breadth: int = 1,
        eqr_inference_breadth: int = 2,
    ) -> None:
        super().__init__()
        self.base = SharedDilatedConvGRU(width=256, hidden=hidden, cycles=cycles)
        self.correction = ProposalCorrection(hidden, correction_hidden, correction_scale)
        self.fixed_depth = int(fixed_depth)
        self.solver_max_iterations = int(solver_max_iterations)
        self.solver_tolerance = float(solver_tolerance)
        self.anderson_history = int(anderson_history)
        self.anderson_regularization = float(anderson_regularization)
        self.implicit_backward_iterations = int(implicit_backward_iterations)
        self.implicit_backward_tolerance = float(implicit_backward_tolerance)
        self.eqr_damping = float(eqr_damping)
        self.eqr_noise = float(eqr_noise)
        self.eqr_initialization_std = float(eqr_initialization_std)
        self.eqr_training_breadth = int(eqr_training_breadth)
        self.eqr_inference_breadth = int(eqr_inference_breadth)
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
        pooled = torch.cat((normalized.mean(dim=-1), normalized.amax(dim=-1)), dim=1)
        return logits, self.base.quotient_head(pooled)

    def operator(self, state: Tensor, proposal: Tensor, features: Tensor, *, damping: float = 0.0, noise: Tensor | None = None) -> Tensor:
        updated = self.correction(state, proposal, features)
        if damping:
            updated = state + (1.0 - float(damping)) * (updated - state)
        return updated if noise is None else updated + noise

    @staticmethod
    def residual(state: Tensor, updated: Tensor) -> Tensor:
        numerator = (updated - state).flatten(1).norm(dim=1)
        denominator = updated.flatten(1).norm(dim=1).clamp_min(1e-9)
        return numerator / denominator

    def _deq(self, route: str):
        solver = "anderson" if route.startswith("anderson") else "fixed_point_iter"
        implicit = route == "anderson_implicit"
        return get_deq({
            "core": "indexing",
            "f_solver": solver,
            "b_solver": "anderson",
            "f_max_iter": self.solver_max_iterations,
            "b_max_iter": self.implicit_backward_iterations,
            "f_tol": self.solver_tolerance,
            "b_tol": self.implicit_backward_tolerance,
            "f_stop_mode": "rel",
            "b_stop_mode": "rel",
            "grad": 1,
            "ift": implicit,
            "n_states": 1,
        })

    def _eqr(self, proposal: Tensor, features: Tensor, generator: torch.Generator, breadth: int) -> tuple[Tensor, Tensor, Tensor]:
        states, residuals = [], []
        for _ in range(breadth):
            state = torch.randn(proposal.shape, generator=generator, device=proposal.device, dtype=proposal.dtype) * self.eqr_initialization_std
            for _ in range(self.fixed_depth):
                noise = torch.randn(state.shape, generator=generator, device=state.device, dtype=state.dtype) * self.eqr_noise
                state = self.operator(state, proposal, features, damping=self.eqr_damping, noise=noise)
            updated = self.operator(state, proposal, features, damping=self.eqr_damping)
            states.append(state); residuals.append(self.residual(state, updated))
        stacked_state, stacked_residual = torch.stack(states), torch.stack(residuals)
        choice = stacked_residual.argmin(dim=0)
        batch = torch.arange(proposal.shape[0], device=proposal.device)
        selected = stacked_state[choice, batch]
        return selected, stacked_residual[choice, batch], torch.full_like(choice, self.fixed_depth)

    def forward(
        self,
        features: Tensor,
        *,
        route: str = "anderson_phantom",
        generator: torch.Generator | None = None,
        initial_state: Tensor | None = None,
    ) -> AttractorOutput:
        if route not in ROUTES:
            raise ValueError(f"unknown route: {route}")
        proposal = self.proposal_state(features)
        if route == "proposal_only":
            logits, quotient = self.decode(proposal)
            zeros = torch.zeros(proposal.shape[0], device=proposal.device)
            return AttractorOutput(logits, quotient, proposal, proposal, zeros, zeros.to(torch.long))
        if initial_state is not None:
            state = initial_state
            for _ in range(self.solver_max_iterations):
                state = self.operator(state, proposal, features)
            updated = self.operator(state, proposal, features)
            relative, iterations = self.residual(state, updated), torch.full((state.shape[0],), self.solver_max_iterations, dtype=torch.long, device=state.device)
        elif route == "fixed_depth_shared":
            state = proposal
            for _ in range(self.fixed_depth): state = self.operator(state, proposal, features)
            updated = self.operator(state, proposal, features)
            relative, iterations = self.residual(state, updated), torch.full((state.shape[0],), self.fixed_depth, dtype=torch.long, device=state.device)
        elif route == "eqr_finite":
            if generator is None: generator = torch.Generator(device=features.device).manual_seed(665000 + features.shape[-1])
            breadth = self.eqr_training_breadth if self.training else self.eqr_inference_breadth
            state, relative, iterations = self._eqr(proposal, features, generator, breadth)
        else:
            deq = self._deq(route); deq.train(self.training)
            solver_kwargs = {"m": self.anderson_history, "lam": self.anderson_regularization} if route.startswith("anderson") else {}
            states, info = deq(lambda z: self.operator(z, proposal, features), proposal, solver_kwargs=solver_kwargs)
            state = states[-1]
            updated = self.operator(state, proposal, features)
            relative = self.residual(state, updated)
            iterations = info["nstep"].to(torch.long)
        logits, quotient = self.decode(state)
        return AttractorOutput(logits, quotient, state, proposal, relative, iterations)

    def randomized_correction(self, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for parameter in self.correction.parameters():
                scale = parameter.detach().std().clamp_min(1e-3)
                parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype, device=parameter.device) * scale)
