"""Ordinary-initialized addressable neural-RAM executor for modular prefixes."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class NeuralRAMExecutor(nn.Module):
    """Persistent spatial canvas plus learned registers and append-only trace.

    The architecture supplies generic differentiable memory operations only. No
    arithmetic update, instruction table, solver, or interpreter is encoded in
    its initialization or forward path.
    """

    ABLATIONS = {"none", "reset", "permute", "uniform"}

    def __init__(
        self,
        width: int,
        canvas_hidden: int = 48,
        controller_hidden: int = 48,
        memory_dim: int = 32,
        register_slots: int = 8,
        attention_heads: int = 4,
        canvas_microsteps: int = 2,
        condition_dim: int = 16,
    ) -> None:
        super().__init__()
        if memory_dim % attention_heads:
            raise ValueError("memory_dim must be divisible by attention_heads")
        if register_slots < 2 or canvas_microsteps < 1:
            raise ValueError("register_slots >= 2 and canvas_microsteps >= 1 required")
        self.width = int(width)
        self.canvas_hidden = int(canvas_hidden)
        self.controller_hidden = int(controller_hidden)
        self.memory_dim = int(memory_dim)
        self.register_slots = int(register_slots)
        self.canvas_microsteps = int(canvas_microsteps)

        self.static_stem = nn.Conv1d(4, canvas_hidden, 1)
        self.controller_init = nn.Linear(2 * canvas_hidden, controller_hidden)
        self.register_initial = nn.Parameter(torch.empty(register_slots, memory_dim))
        self.register_address = nn.Parameter(torch.empty(register_slots, memory_dim))
        self.address_mlp = nn.Sequential(
            nn.Linear(4, memory_dim), nn.SiLU(), nn.Linear(memory_dim, memory_dim)
        )
        self.read_query = nn.Linear(controller_hidden, memory_dim)
        self.read_attention = nn.MultiheadAttention(
            memory_dim, attention_heads, batch_first=True
        )
        controller_input = 1 + 4 + canvas_hidden + memory_dim
        self.controller = nn.GRUCell(controller_input, controller_hidden)
        self.register_key = nn.Linear(memory_dim, memory_dim, bias=False)
        self.write_query = nn.Linear(controller_hidden, memory_dim, bias=False)
        self.write_erase = nn.Linear(controller_hidden + memory_dim, memory_dim)
        self.write_add = nn.Linear(controller_hidden + memory_dim, memory_dim)
        self.trace_projection = nn.Linear(
            controller_hidden + memory_dim + canvas_hidden, memory_dim
        )

        self.condition = nn.Linear(
            controller_hidden + memory_dim + 1 + 4, condition_dim
        )
        canvas_input = canvas_hidden + 4 + condition_dim
        self.canvas_norm = nn.GroupNorm(1, canvas_hidden)
        self.canvas_gates = nn.Conv1d(canvas_input, 2 * canvas_hidden, 3, padding=1)
        self.canvas_candidate = nn.Conv1d(canvas_input, canvas_hidden, 3, padding=1)
        self.output_norm = nn.GroupNorm(1, canvas_hidden)
        self.local_head = nn.Conv1d(canvas_hidden, 1, 1)

        nn.init.xavier_uniform_(self.register_initial)
        nn.init.xavier_uniform_(self.register_address)

    @staticmethod
    def bits(values: list[int], width: int) -> Tensor:
        return torch.tensor(
            [[(value >> position) & 1 for position in range(width)] for value in values],
            dtype=torch.float32,
        )

    def static_features(
        self, multiplicand: list[int], modulus: list[int], width: int
    ) -> Tensor:
        batch = len(multiplicand)
        lsb = torch.zeros(batch, width)
        msb = torch.zeros(batch, width)
        lsb[:, 0] = 1.0
        msb[:, -1] = 1.0
        return torch.stack(
            (self.bits(multiplicand, width), self.bits(modulus, width), lsb, msb),
            dim=1,
        ).mul(2.0).sub(1.0)

    @staticmethod
    def position_features(
        position: int, total: int, *, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        progress = float(position + 1) / float(total)
        return torch.tensor(
            [
                progress,
                1.0 - progress,
                math.sin(math.pi * progress),
                math.cos(math.pi * progress),
            ],
            device=device,
            dtype=dtype,
        )

    def decode(self, canvas: Tensor) -> Tensor:
        return self.local_head(F.silu(self.output_norm(canvas))).squeeze(1)

    def _memory_read(
        self,
        controller: Tensor,
        registers: Tensor,
        trace: list[Tensor],
        trace_addresses: list[Tensor],
        ablation: str,
    ) -> tuple[Tensor, Tensor]:
        batch = controller.shape[0]
        register_addresses = self.register_address[None].expand(batch, -1, -1)
        memory_parts = [registers]
        address_parts = [register_addresses]
        if trace:
            memory_parts.append(torch.stack(trace, dim=1))
            address_parts.append(torch.stack(trace_addresses, dim=1))
        memory = torch.cat(memory_parts, dim=1)
        addresses = torch.cat(address_parts, dim=1)
        content = memory.flip(1) if ablation == "permute" else memory
        if ablation == "uniform":
            weights = torch.full(
                (batch, 1, memory.shape[1]),
                1.0 / memory.shape[1],
                device=memory.device,
                dtype=memory.dtype,
            )
            return content.mean(dim=1), weights
        query = self.read_query(controller)[:, None, :]
        read, weights = self.read_attention(
            query,
            content + addresses,
            content,
            need_weights=True,
            average_attn_weights=True,
        )
        return read[:, 0], weights

    def _write_registers(
        self,
        controller: Tensor,
        read: Tensor,
        registers: Tensor,
        ablation: str,
    ) -> tuple[Tensor, Tensor]:
        batch = controller.shape[0]
        if ablation == "uniform":
            weights = torch.full(
                (batch, self.register_slots),
                1.0 / self.register_slots,
                device=registers.device,
                dtype=registers.dtype,
            )
        else:
            query = self.write_query(controller)
            keys = self.register_key(registers) + self.register_address[None]
            weights = torch.softmax(
                torch.einsum("bd,brd->br", query, keys) / math.sqrt(self.memory_dim),
                dim=-1,
            )
        write_context = torch.cat((controller, read), dim=-1)
        erase = torch.sigmoid(self.write_erase(write_context))[:, None, :]
        add = torch.tanh(self.write_add(write_context))[:, None, :]
        allocation = weights[:, :, None]
        updated = registers * (1.0 - allocation * erase) + allocation * add
        return updated, weights

    def _update_canvas(
        self,
        canvas: Tensor,
        static_features: Tensor,
        controller: Tensor,
        read: Tensor,
        input_bit: Tensor,
        position: Tensor,
    ) -> Tensor:
        condition = torch.tanh(
            self.condition(
                torch.cat(
                    (controller, read, input_bit.float()[:, None], position), dim=-1
                )
            )
        )
        expanded = condition[:, :, None].expand(-1, -1, static_features.shape[-1])
        for _ in range(self.canvas_microsteps):
            normalized = self.canvas_norm(canvas)
            combined = torch.cat((normalized, static_features, expanded), dim=1)
            reset, update = torch.sigmoid(self.canvas_gates(combined)).chunk(2, dim=1)
            candidate_inputs = torch.cat(
                (reset * normalized, static_features, expanded), dim=1
            )
            candidate = torch.tanh(self.canvas_candidate(candidate_inputs))
            canvas = (1.0 - update) * canvas + update * candidate
        return canvas

    def forward(
        self,
        multiplier_bits: Tensor,
        static_features: Tensor,
        *,
        return_all: bool = False,
        return_diagnostics: bool = False,
        memory_ablation: str = "none",
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if memory_ablation not in self.ABLATIONS:
            raise ValueError(f"unknown memory ablation: {memory_ablation}")
        if multiplier_bits.ndim != 2 or static_features.shape[1] != 4:
            raise ValueError("expected [batch, steps] bits and four static channels")
        batch, steps = multiplier_bits.shape
        self.width = static_features.shape[-1]
        canvas = self.static_stem(static_features)
        pooled = torch.cat((canvas.mean(-1), canvas.amax(-1)), dim=-1)
        controller = torch.tanh(self.controller_init(pooled))
        initial_registers = self.register_initial[None].expand(batch, -1, -1)
        registers = initial_registers
        trace: list[Tensor] = []
        trace_addresses: list[Tensor] = []
        outputs = []
        diagnostic_values: dict[str, list[Tensor]] = {
            "read_entropy": [],
            "write_entropy": [],
            "maximum_write_probability": [],
            "register_delta_rms": [],
            "trace_rms": [],
        }
        for position_index in range(steps):
            if memory_ablation == "reset" and position_index:
                registers = initial_registers
                trace = []
                trace_addresses = []
            position_vector = self.position_features(
                position_index,
                steps,
                device=static_features.device,
                dtype=static_features.dtype,
            )
            position = position_vector[None].expand(batch, -1)
            read, read_weights = self._memory_read(
                controller, registers, trace, trace_addresses, memory_ablation
            )
            bit = multiplier_bits[:, position_index]
            controller = self.controller(
                torch.cat((bit[:, None], position, canvas.mean(-1), read), dim=-1),
                controller,
            )
            previous_registers = registers
            registers, write_weights = self._write_registers(
                controller, read, registers, memory_ablation
            )
            trace_token = torch.tanh(
                self.trace_projection(torch.cat((controller, read, canvas.mean(-1)), dim=-1))
            )
            trace.append(trace_token)
            address = self.address_mlp(position)
            trace_addresses.append(address)
            canvas = self._update_canvas(
                canvas, static_features, controller, read, bit, position
            )
            if return_all:
                outputs.append(self.decode(canvas))
            if return_diagnostics:
                read_probability = read_weights[:, 0].clamp_min(1e-12)
                read_normalizer = math.log(max(2, read_probability.shape[-1]))
                write_probability = write_weights.clamp_min(1e-12)
                diagnostic_values["read_entropy"].append(
                    -(read_probability * read_probability.log()).sum(-1) / read_normalizer
                )
                diagnostic_values["write_entropy"].append(
                    -(write_probability * write_probability.log()).sum(-1)
                    / math.log(self.register_slots)
                )
                diagnostic_values["maximum_write_probability"].append(
                    write_weights.amax(-1)
                )
                diagnostic_values["register_delta_rms"].append(
                    (registers - previous_registers).square().mean((1, 2)).sqrt()
                )
                diagnostic_values["trace_rms"].append(
                    trace_token.square().mean(-1).sqrt()
                )
        logits = torch.stack(outputs, dim=1) if return_all else self.decode(canvas)
        if not return_diagnostics:
            return logits
        diagnostics = {
            key: torch.stack(values, dim=1) for key, values in diagnostic_values.items()
        }
        return logits, diagnostics

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
