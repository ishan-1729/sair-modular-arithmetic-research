"""Bounded-faithful Universal Transformer memory-token transition models."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class DerfNorm(nn.Module):
    """Learned erf normalization-free transform from the source paper."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(hidden))
        self.shift = nn.Parameter(torch.zeros(hidden))

    def forward(self, value: Tensor) -> Tensor:
        return torch.erf(self.alpha * value + self.shift)


def apply_rope(value: Tensor) -> Tensor:
    """Apply generic indexed rotary position encoding to [B,H,N,D]."""
    dimension = value.shape[-1]
    if dimension % 2:
        raise ValueError("RoPE head dimension must be even")
    positions = torch.arange(value.shape[-2], device=value.device, dtype=value.dtype)
    frequencies = torch.exp(
        torch.arange(0, dimension, 2, device=value.device, dtype=value.dtype)
        * (-math.log(10_000.0) / dimension)
    )
    angles = positions[:, None] * frequencies[None]
    cosine, sine = angles.cos()[None, None], angles.sin()[None, None]
    even, odd = value[..., 0::2], value[..., 1::2]
    return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(-2)


class QKRoPEAttention(nn.Module):
    def __init__(self, hidden: int, heads: int) -> None:
        super().__init__()
        if hidden % heads or (hidden // heads) % 2:
            raise ValueError("hidden/head geometry must support paired RoPE")
        self.hidden, self.heads, self.head_dim = hidden, heads, hidden // heads
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.output = nn.Linear(hidden, hidden)

    def forward(self, value: Tensor, *, use_index: bool = True) -> Tensor:
        batch, tokens, _ = value.shape
        q, k, v = self.qkv(value).chunk(3, dim=-1)
        reshape = lambda item: item.view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = reshape(q), reshape(k), reshape(v)
        if use_index:
            q, k = apply_rope(q), apply_rope(k)
        q = q * torch.rsqrt(q.square().mean(dim=-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().mean(dim=-1, keepdim=True) + 1e-6)
        attention = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1)
        combined = torch.matmul(attention, v).transpose(1, 2).reshape(batch, tokens, self.hidden)
        return self.output(combined)


class SharedUTCell(nn.Module):
    """One pre-norm DerfNorm/attention/SwiGLU block shared over steps."""

    def __init__(self, hidden: int, heads: int, ff_hidden: int) -> None:
        super().__init__()
        self.attention_norm = DerfNorm(hidden)
        self.attention = QKRoPEAttention(hidden, heads)
        self.ffn_norm = DerfNorm(hidden)
        self.ffn_gate = nn.Linear(hidden, 2 * ff_hidden)
        self.ffn_output = nn.Linear(ff_hidden, hidden)

    def forward(self, value: Tensor, *, use_index: bool = True) -> Tensor:
        value = value + self.attention(self.attention_norm(value), use_index=use_index)
        gate, content = self.ffn_gate(self.ffn_norm(value)).chunk(2, dim=-1)
        return value + self.ffn_output(F.silu(gate) * content)


@dataclass
class TransitionOutput:
    logits: Tensor
    hidden: Tensor
    halt_probabilities: Tensor
    halt_steps: Tensor
    ponder_cost: Tensor
    block_calls: int
    analytical_flops: int


class UTMemoryTransition(nn.Module):
    """Legal bit-token transition with learned shared workspace and matched controls."""

    ABLATIONS = {"none", "reset", "zero", "randomize", "permute", "no_type", "no_index"}

    def __init__(
        self,
        hidden: int = 16,
        heads: int = 4,
        ff_hidden: int = 32,
        wide_hidden: int = 24,
        max_memory_tokens: int = 64,
        max_steps: int = 6,
        act_epsilon: float = 0.01,
        halt_bias: float = -3.0,
        architecture: str = "ut_memory",
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.heads = int(heads)
        self.ff_hidden = int(ff_hidden)
        self.wide_hidden = int(wide_hidden)
        self.max_memory_tokens = int(max_memory_tokens)
        self.max_steps = int(max_steps)
        self.act_epsilon = float(act_epsilon)
        self.architecture = architecture
        self.input_projection = nn.Linear(6, hidden)
        self.memory_tokens = nn.Parameter(torch.empty(max_memory_tokens, hidden))
        self.type_embedding = nn.Parameter(torch.empty(2, hidden))
        self.step_projection = nn.Sequential(nn.Linear(4, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.cell = SharedUTCell(hidden, heads, ff_hidden)
        self.router = nn.Linear(hidden, 1)
        self.output_norm = DerfNorm(hidden)
        self.output_head = nn.Linear(hidden, 1)
        # Superset control modules are stored by every arm for exact parameter identity.
        self.wide_up = nn.Linear(hidden, wide_hidden)
        self.wide_down = nn.Linear(wide_hidden, hidden)
        self.decoded_gate = nn.Conv1d(2 * hidden, 2 * hidden, 3, padding=1)
        self.decoded_candidate = nn.Conv1d(2 * hidden, hidden, 3, padding=1)
        self.reset_parameters(halt_bias)

    def reset_parameters(self, halt_bias: float) -> None:
        nn.init.normal_(self.memory_tokens, std=0.02)
        nn.init.normal_(self.type_embedding, std=0.02)
        nn.init.zeros_(self.router.weight)
        nn.init.constant_(self.router.bias, float(halt_bias))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def step_signal(step: int, total: int, batch: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        progress = float(step) / float(max(1, total - 1))
        base = torch.tensor([progress, 1.0 - progress, math.sin(math.pi * progress), math.cos(math.pi * progress)], device=device, dtype=dtype)
        return base[None].expand(batch, -1)

    def _initial_tokens(
        self,
        features: Tensor,
        memory_count: int,
        ablation: str,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, Tensor]:
        batch, _, width = features.shape
        sequence = self.input_projection(features.transpose(1, 2))
        memory = self.memory_tokens[:memory_count][None].expand(batch, -1, -1)
        if ablation == "zero":
            memory = torch.zeros_like(memory)
        elif ablation == "randomize":
            memory = torch.randn(memory.shape, device=memory.device, dtype=memory.dtype, generator=generator)
        elif ablation == "permute" and memory_count:
            memory = memory.flip(1)
        if ablation != "no_type":
            if memory_count:
                memory = memory + self.type_embedding[0]
            sequence = sequence + self.type_embedding[1]
        return memory, sequence

    def _decoded_step(self, hidden: Tensor, injected: Tensor) -> Tensor:
        combined = torch.cat((hidden, injected), dim=-1).transpose(1, 2)
        reset, update = torch.sigmoid(self.decoded_gate(combined)).chunk(2, dim=1)
        candidate_input = torch.cat((reset * hidden.transpose(1, 2), injected.transpose(1, 2)), dim=1)
        candidate = torch.tanh(self.decoded_candidate(candidate_input)).transpose(1, 2)
        return (1.0 - update.transpose(1, 2)) * hidden + update.transpose(1, 2) * candidate

    def forward(
        self,
        features: Tensor,
        *,
        memory_count: int,
        use_act: bool = True,
        max_steps: int | None = None,
        ablation: str = "none",
        generator: torch.Generator | None = None,
    ) -> TransitionOutput:
        if features.ndim != 3 or features.shape[1] != 6:
            raise ValueError("features must have shape [batch, 6, width]")
        if memory_count not in {0, 4, 8, 16, 32, 64} or memory_count > self.max_memory_tokens:
            raise ValueError("memory_count must be one frozen sweep value")
        if ablation not in self.ABLATIONS:
            raise ValueError(f"unknown ablation: {ablation}")
        steps = self.max_steps if max_steps is None else int(max_steps)
        memory, sequence = self._initial_tokens(features, memory_count, ablation, generator)
        initial_memory = memory
        hidden = torch.cat((memory, sequence), dim=1)
        injected = hidden
        weighted = torch.zeros_like(hidden)
        cumulative = torch.zeros(hidden.shape[:2], device=hidden.device, dtype=hidden.dtype)
        halt_steps = torch.full(hidden.shape[:2], steps, device=hidden.device, dtype=torch.long)
        probabilities = []
        for step in range(steps):
            signal = self.step_projection(self.step_signal(step, steps, hidden.shape[0], device=hidden.device, dtype=hidden.dtype))[:, None]
            if self.architecture == "decoded_recurrent":
                next_hidden = self._decoded_step(hidden, injected + signal)
            else:
                next_hidden = self.cell(hidden + signal, use_index=ablation != "no_index")
                if self.architecture == "wider_no_memory":
                    next_hidden = next_hidden + self.wide_down(F.silu(self.wide_up(next_hidden)))
            if ablation == "reset" and memory_count:
                next_hidden = torch.cat((initial_memory, next_hidden[:, memory_count:]), dim=1)
            probability = torch.sigmoid(self.router(next_hidden)).squeeze(-1)
            probabilities.append(probability)
            remaining = (1.0 - cumulative).clamp_min(0.0)
            if step == steps - 1:
                weight = remaining
            else:
                proposed = probability * cumulative.lt(1.0 - self.act_epsilon)
                weight = torch.minimum(proposed, remaining)
            newly_halted = cumulative.lt(1.0 - self.act_epsilon) & (cumulative + weight).ge(1.0 - self.act_epsilon)
            halt_steps = torch.where(newly_halted, torch.full_like(halt_steps, step + 1), halt_steps)
            weighted = weighted + weight.unsqueeze(-1) * next_hidden
            cumulative = cumulative + weight
            hidden = next_hidden
        chosen = weighted if use_act and self.architecture != "decoded_recurrent" else hidden
        sequence_hidden = chosen[:, memory_count:]
        logits = self.output_head(self.output_norm(sequence_hidden)).squeeze(-1)
        tokens = hidden.shape[1]
        attention_flops = 4 * tokens * self.hidden * self.hidden + 2 * tokens * tokens * self.hidden
        ffn_flops = 3 * tokens * self.hidden * self.ff_hidden
        per_step = attention_flops + ffn_flops
        if self.architecture == "wider_no_memory":
            per_step += 2 * tokens * self.hidden * self.wide_hidden
        elif self.architecture == "decoded_recurrent":
            per_step = 18 * features.shape[-1] * self.hidden * self.hidden
        return TransitionOutput(
            logits=logits,
            hidden=sequence_hidden,
            halt_probabilities=torch.stack(probabilities, dim=1),
            halt_steps=halt_steps[:, memory_count:],
            ponder_cost=halt_steps[:, memory_count:].float().mean(),
            block_calls=steps,
            analytical_flops=int(features.shape[0] * steps * per_step),
        )
