"""Position-free shared residual refinement for EXP670."""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from experiments.exp_220_width_independent_cellular.model import SharedDilatedConvGRU

ROUTES = (
    "proposal_only", "clipped_poisson", "fixed_mean", "matched_histogram",
    "shuffled_count_independent", "randomized",
)
TRAINED_ROUTES = ROUTES[1:5]


@dataclass
class RefinementOutput:
    logits: Tensor
    quotient_logits: Tensor
    state: Tensor
    proposal: Tensor
    relative_residual: Tensor
    iterations: Tensor


class SharedResidualOperator(nn.Module):
    """One NoPE operator, with proposal and original input reinjected every loop."""

    def __init__(self, hidden: int, inner: int, scale: float) -> None:
        super().__init__()
        self.scale = float(scale)
        self.norm = nn.GroupNorm(1, 2 * hidden + 6)
        self.inner = nn.Conv1d(2 * hidden + 6, inner, 3, padding=1)
        self.output = nn.Conv1d(inner, hidden, 1)
        nn.init.kaiming_uniform_(self.inner.weight, nonlinearity="relu")
        nn.init.zeros_(self.inner.bias)
        # The complete residual branch is exactly zero at initialization.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, state: Tensor, proposal: Tensor, features: Tensor) -> Tensor:
        injected = self.norm(torch.cat((state, proposal, features), dim=1))
        delta = torch.tanh(self.output(F.silu(self.inner(injected))))
        return state + self.scale * delta


class LoopCurriculumCell(nn.Module):
    """Frozen EXP609 proposal followed by a shared, bounded refinement loop."""

    def __init__(self, *, hidden: int = 92, cycles: int = 2, operator_inner: int = 64,
                 transition_scale: float = .10, inference_count: int = 4) -> None:
        super().__init__()
        self.base = SharedDilatedConvGRU(width=256, hidden=hidden, cycles=cycles)
        self.operator = SharedResidualOperator(hidden, operator_inner, transition_scale)
        self.inference_count = int(inference_count)
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
        # This mandatory proposal scan is inherited and is never count-controlled.
        for dilation in self.base.dilation_schedule(features.shape[-1]):
            hidden = self.base.recurrent_update(hidden, features, dilation)
        return hidden

    def decode(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        normalized = F.silu(self.base.output_norm(hidden))
        logits = self.base.local_head(normalized).squeeze(1)
        pooled = torch.cat((normalized.mean(-1), normalized.amax(-1)), 1)
        return logits, self.base.quotient_head(pooled)

    def _run(self, features: Tensor, route: str, count: int) -> RefinementOutput:
        if route not in ROUTES:
            raise ValueError(f"unknown route: {route}")
        proposal = self.proposal_state(features)
        state = proposal
        calls = 0 if route == "proposal_only" else int(count)
        if not 0 <= calls <= 8:
            raise ValueError("refinement count outside frozen bound")
        for _ in range(calls):
            state = self.operator(state, proposal, features)
        logits, quotient = self.decode(state)
        residual = ((state - proposal).flatten(1).norm(dim=1) /
                    proposal.flatten(1).norm(dim=1).clamp_min(1e-9))
        iterations = torch.full_like(residual, calls, dtype=torch.long)
        return RefinementOutput(logits, quotient, state, proposal, residual, iterations)

    def forward(self, features: Tensor, *, route: str = "clipped_poisson") -> RefinementOutput:
        """Fixed target-free inference depth; training counts cannot reach this surface."""
        return self._run(features, route, 0 if route == "proposal_only" else self.inference_count)

    def forward_train(self, features: Tensor, *, route: str, count: int) -> RefinementOutput:
        return self._run(features, route, int(count))

    def randomized_intervention(self, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for parameter in self.operator.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator,
                                            dtype=parameter.dtype) * .02)


def frozen_count_schedules(config: dict, updates: int) -> dict[str, list[int]]:
    """Create replayable, sample-once schedules with exactly matched total work."""
    schedule = config["count_schedule"]
    total = int(schedule["frozen_schedule_updates"])
    if updates < 1 or updates > total:
        raise ValueError("updates outside frozen count schedule")
    generator = torch.Generator().manual_seed(int(schedule["seed"]))
    rate = torch.full((total,), float(schedule["poisson_mean"]), dtype=torch.float32)
    source = torch.poisson(rate, generator=generator).clamp(
        int(schedule["minimum"]), int(schedule["maximum"])).to(torch.long)
    expected_total = total * int(schedule["fixed_mean"])
    if int(source.sum()) != expected_total:
        raise RuntimeError("frozen Poisson stream no longer has preregistered mean")
    matched = source.sort().values
    shuffle_generator = torch.Generator().manual_seed(int(schedule["shuffle_seed"]))
    shuffled = source[torch.randperm(total, generator=shuffle_generator)]
    return {
        "clipped_poisson": source[:updates].tolist(),
        "fixed_mean": [int(schedule["fixed_mean"])] * updates,
        "matched_histogram": matched[:updates].tolist() if updates == total else source[:updates].sort().values.tolist(),
        "shuffled_count_independent": shuffled[:updates].tolist() if updates == total else source[:updates].flip(0).tolist(),
    }
