"""Bounded-faithful, exact-initialized LT2 hybrid transfer for EXP672."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from experiments.exp_220_width_independent_cellular.model import SharedDilatedConvGRU


ROUTES = (
    "full_hybrid", "gate_only", "mixer_fixed_gate", "shared_conv_control",
    "proposal_only", "reset", "randomized",
)
TRAINED_ROUTES = ROUTES[:4]


@dataclass
class LT2Output:
    logits: Tensor
    quotient_logits: Tensor
    state: Tensor
    proposal: Tensor
    relative_residual: Tensor
    iterations: Tensor
    loop_states: tuple[Tensor, ...]
    linear_norm: Tensor
    sparse_concentration: Tensor


class LT2HybridMixer(nn.Module):
    """Native linear-attention plus fixed-window sparse-attention hybrid."""

    def __init__(self, channels: int, hidden: int, width: int, rank: int,
                 sparse_window: int, epsilon: float) -> None:
        super().__init__()
        if sparse_window < 1 or sparse_window % 2 != 1:
            raise ValueError("sparse_window must be positive and odd")
        self.hidden = hidden
        self.width = width
        self.rank = rank
        self.sparse_window = sparse_window
        self.epsilon = epsilon
        self.norm = nn.GroupNorm(1, channels)
        self.linear_q = nn.Conv1d(channels, rank, 1)
        self.linear_k = nn.Conv1d(channels, rank, 1)
        self.linear_v = nn.Conv1d(channels, width, 1)
        self.linear_out = nn.Conv1d(width, hidden, 1)
        self.sparse_q = nn.Conv1d(channels, rank, 1)
        self.sparse_k = nn.Conv1d(channels, rank, 1)
        self.sparse_v = nn.Conv1d(channels, width, 1)
        self.sparse_gate = nn.Conv1d(channels, width, 1, bias=False)
        self.sparse_out = nn.Conv1d(width, hidden, 1)
        self.merge = nn.Parameter(torch.ones(2) * 0.5)
        nn.init.zeros_(self.sparse_gate.weight)

    def _linear(self, x: Tensor) -> tuple[Tensor, Tensor]:
        q = F.elu(self.linear_q(x)) + 1.0
        k = F.elu(self.linear_k(x)) + 1.0
        v = self.linear_v(x)
        # Prefix recurrent state S_t = sum_{i<=t} phi(k_i) v_i^T.
        state = torch.einsum("brw,bvw->brvw", k, v).cumsum(-1)
        normalizer = k.cumsum(-1)
        numerator = torch.einsum("brw,brvw->bvw", q, state)
        denominator = torch.einsum("brw,brw->bw", q, normalizer).unsqueeze(1)
        output = numerator / denominator.clamp_min(self.epsilon)
        return self.linear_out(output), state.flatten(1, 2).norm(dim=1).mean(-1)

    def _sparse(self, x: Tensor) -> tuple[Tensor, Tensor]:
        q, k, v = self.sparse_q(x), self.sparse_k(x), self.sparse_v(x)
        radius = self.sparse_window // 2
        k_window = F.pad(k, (radius, radius)).unfold(-1, self.sparse_window, 1)
        v_window = F.pad(v, (radius, radius)).unfold(-1, self.sparse_window, 1)
        scores = torch.einsum("brw,brws->bws", q, k_window) / self.rank ** 0.5
        attention = scores.softmax(-1)
        mixed = torch.einsum("bws,bvws->bvw", attention, v_window)
        # Official G1 shape: zero weights imply sigmoid(0)=0.5 before output projection.
        mixed = mixed * torch.sigmoid(self.sparse_gate(x))
        concentration = attention.amax(-1).mean(-1)
        return self.sparse_out(mixed), concentration

    def forward(self, state: Tensor, proposal: Tensor, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = self.norm(torch.cat((state, proposal, features), 1))
        linear, linear_norm = self._linear(x)
        sparse, concentration = self._sparse(x)
        weights = self.merge.softmax(0)
        return torch.tanh(weights[0] * linear + weights[1] * sparse), linear_norm, concentration


class MatchedSharedConv(nn.Module):
    """Simple shared convolutional control with similar parameters and four calls."""

    def __init__(self, channels: int, hidden: int, inner: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.inner = nn.Conv1d(channels, inner, 5, padding=2)
        self.output = nn.Conv1d(inner, hidden, 1)

    def forward(self, state: Tensor, proposal: Tensor, features: Tensor) -> Tensor:
        x = self.norm(torch.cat((state, proposal, features), 1))
        return torch.tanh(self.output(F.silu(self.inner(x))))


class LT2GatedHybridCell(nn.Module):
    """Frozen EXP609 proposal plus one genuinely shared bounded LT2 correction loop."""

    def __init__(self, *, hidden: int = 92, cycles: int = 2, loop_count: int = 4,
                 mixer_width: int = 24, linear_rank: int = 8, sparse_window: int = 5,
                 control_inner: int = 24, fixed_mixer_scale: float = 0.05,
                 epsilon: float = 1e-6) -> None:
        super().__init__()
        self.base = SharedDilatedConvGRU(width=256, hidden=hidden, cycles=cycles)
        channels = 2 * hidden + 6
        self.hybrid = LT2HybridMixer(channels, hidden, mixer_width, linear_rank,
                                     sparse_window, epsilon)
        self.control = MatchedSharedConv(channels, hidden, control_inner)
        self.loop_count = int(loop_count)
        self.fixed_mixer_scale = float(fixed_mixer_scale)
        self.loop_gate = nn.Parameter(torch.zeros(self.loop_count, hidden, 1))
        self.output_gate = nn.Parameter(torch.zeros(1, hidden, 1))
        self.epsilon = float(epsilon)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def load_source(self, state: dict[str, Tensor]) -> None:
        self.base.load_state_dict(state, strict=True)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def trainable_parameter_count(self, route: str | None = None) -> int:
        if route is None:
            return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return sum(parameter.numel() for parameter in self.route_parameters(route))

    def route_parameters(self, route: str) -> list[nn.Parameter]:
        if route == "full_hybrid":
            return [*self.hybrid.parameters(), self.loop_gate, self.output_gate]
        if route == "gate_only":
            return [self.loop_gate, self.output_gate]
        if route == "mixer_fixed_gate":
            return list(self.hybrid.parameters())
        if route == "shared_conv_control":
            return [*self.control.parameters(), self.loop_gate, self.output_gate]
        return []

    def proposal_state(self, features: Tensor) -> Tensor:
        self.base.width = features.shape[-1]
        hidden = self.base.stem(features)
        for dilation in self.base.dilation_schedule(features.shape[-1]):
            hidden = self.base.recurrent_update(hidden, features, dilation)
        return hidden

    def proposal_step(self, state: Tensor, features: Tensor, index: int) -> Tensor:
        schedule = self.base.dilation_schedule(features.shape[-1])
        return self.base.recurrent_update(state, features, schedule[index % len(schedule)])

    def decode(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        normalized = F.silu(self.base.output_norm(hidden))
        logits = self.base.local_head(normalized).squeeze(1)
        pooled = torch.cat((normalized.mean(-1), normalized.amax(-1)), 1)
        return logits, self.base.quotient_head(pooled)

    def _run(self, features: Tensor, route: str, loops: int | None = None) -> LT2Output:
        if route not in ROUTES:
            raise ValueError(f"unknown route: {route}")
        proposal = self.proposal_state(features)
        if route == "proposal_only":
            logits, quotient = self.decode(proposal)
            zeros = proposal.new_zeros(proposal.shape[0])
            return LT2Output(logits, quotient, proposal, proposal, zeros, zeros.long(), (), zeros, zeros)
        count = self.loop_count if loops is None else int(loops)
        if not 1 <= count <= self.loop_count:
            raise ValueError("loop count outside frozen diagnostic bound")
        state = proposal
        states: list[Tensor] = []
        linear_norms: list[Tensor] = []
        concentrations: list[Tensor] = []
        for index in range(count):
            if route == "gate_only":
                mixed = self.proposal_step(state, features, index)
                linear_norm = concentration = proposal.new_zeros(proposal.shape[0])
            elif route == "shared_conv_control":
                mixed = state + self.control(state, proposal, features)
                linear_norm = concentration = proposal.new_zeros(proposal.shape[0])
            else:
                delta, linear_norm, concentration = self.hybrid(state, proposal, features)
                mixed = state + delta
            if route == "mixer_fixed_gate":
                state = state + self.fixed_mixer_scale * (mixed - state)
            else:
                rho = torch.tanh(self.loop_gate[index])
                state = mixed + rho * state
            states.append(state)
            linear_norms.append(linear_norm)
            concentrations.append(concentration)
        if route == "mixer_fixed_gate":
            final = state
        else:
            final = proposal + torch.tanh(self.output_gate) * (state - proposal)
        logits, quotient = self.decode(final)
        residual = ((final - proposal).flatten(1).norm(dim=1) /
                    proposal.flatten(1).norm(dim=1).clamp_min(self.epsilon))
        iterations = torch.full_like(residual, count, dtype=torch.long)
        linear_mean = torch.stack(linear_norms).mean(0)
        concentration_mean = torch.stack(concentrations).mean(0)
        return LT2Output(logits, quotient, final, proposal, residual, iterations,
                         tuple(states), linear_mean, concentration_mean)

    def forward_train(self, features: Tensor, *, route: str) -> LT2Output:
        if route not in TRAINED_ROUTES:
            raise ValueError("route is not trainable")
        return self._run(features, route)

    def forward(self, features: Tensor, *, route: str = "full_hybrid", loops: int | None = None,
                **unused) -> LT2Output:
        return self._run(features, route, loops)

    def reset_gate_intervention(self) -> None:
        with torch.no_grad():
            self.loop_gate.zero_()
            self.output_gate.zero_()

    def randomized_mixer_intervention(self, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for parameter in self.hybrid.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator,
                                             dtype=parameter.dtype, device=parameter.device) * 0.02)
            self.hybrid.sparse_gate.weight.zero_()

    def diagnostics(self) -> dict[str, float | list[float]]:
        return {
            "loop_gate_tanh_mean_abs": float(torch.tanh(self.loop_gate).abs().mean()),
            "loop_gate_tanh_max_abs": float(torch.tanh(self.loop_gate).abs().max()),
            "output_gate_tanh_mean_abs": float(torch.tanh(self.output_gate).abs().mean()),
            "output_gate_tanh_max_abs": float(torch.tanh(self.output_gate).abs().max()),
            "hybrid_merge": self.hybrid.merge.softmax(0).detach().tolist(),
        }

    def analytical_mixer_flops(self, width: int, batch: int, route: str) -> int:
        channels = 2 * self.base.hidden + 6
        if route == "gate_only":
            return int(batch * self.loop_count * width * self.base.hidden * 9)
        if route == "shared_conv_control":
            inner = self.control.inner.out_channels
            return int(batch * self.loop_count * width *
                       (channels * inner * 5 + inner * self.base.hidden) * 2)
        if route in ("full_hybrid", "mixer_fixed_gate", "reset", "randomized"):
            r, v, s = self.hybrid.rank, self.hybrid.width, self.hybrid.sparse_window
            return int(batch * self.loop_count * width *
                       (channels * (4 * r + 3 * v) + 2 * r * v + s * (r + v) +
                        2 * v * self.base.hidden) * 2)
        return 0
