"""Explicit-device inputs for the frozen EXP654 neural-RAM recurrence."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from .runtime import module_device


@dataclass(frozen=True)
class NeuralRAMInputs:
    multiplier_bits: Tensor
    static_features: Tensor
    position_vectors: Tensor


def bits_on_device(
    values: list[int], width: int, device: torch.device
) -> Tensor:
    rows = [
        [(value >> position) & 1 for position in range(width)]
        for value in values
    ]
    return torch.tensor(rows, dtype=torch.float32, device=device)


def prepare_neural_ram_inputs(
    *,
    multiplier: list[int],
    multiplicand: list[int],
    modulus: list[int],
    width: int,
    device: torch.device,
) -> NeuralRAMInputs:
    if not (len(multiplier) == len(multiplicand) == len(modulus)):
        raise ValueError("neural-RAM input batches must have equal lengths")
    batch = len(multiplier)
    lsb = torch.zeros(batch, width, device=device)
    msb = torch.zeros(batch, width, device=device)
    lsb[:, 0] = 1.0
    msb[:, -1] = 1.0
    static = torch.stack(
        (
            bits_on_device(multiplicand, width, device),
            bits_on_device(modulus, width, device),
            lsb,
            msb,
        ),
        dim=1,
    ).mul(2.0).sub(1.0)
    positions = []
    for position in range(width):
        progress = float(position + 1) / float(width)
        positions.append(
            [
                progress,
                1.0 - progress,
                math.sin(math.pi * progress),
                math.cos(math.pi * progress),
            ]
        )
    return NeuralRAMInputs(
        multiplier_bits=bits_on_device(multiplier, width, device).flip(1),
        static_features=static,
        position_vectors=torch.tensor(
            positions, dtype=static.dtype, device=device
        ),
    )


def neural_ram_forward_prepared(
    model: nn.Module,
    inputs: NeuralRAMInputs,
    *,
    return_all: bool = False,
    return_diagnostics: bool = False,
    memory_ablation: str = "none",
):
    """Run frozen EXP654 layers with all Python-derived tensors prebuilt."""

    if memory_ablation not in model.ABLATIONS:
        raise ValueError(f"unknown memory ablation: {memory_ablation}")
    multiplier_bits = inputs.multiplier_bits
    static_features = inputs.static_features
    if multiplier_bits.ndim != 2 or static_features.shape[1] != 4:
        raise ValueError("expected [batch, steps] bits and four static channels")
    batch, steps = multiplier_bits.shape
    if inputs.position_vectors.shape != (steps, 4):
        raise ValueError("expected one prepared four-vector per recurrent step")
    model.width = static_features.shape[-1]
    canvas = model.static_stem(static_features)
    pooled = torch.cat((canvas.mean(-1), canvas.amax(-1)), dim=-1)
    controller = torch.tanh(model.controller_init(pooled))
    initial_registers = model.register_initial[None].expand(batch, -1, -1)
    registers = initial_registers
    trace: list[Tensor] = []
    trace_addresses: list[Tensor] = []
    outputs: list[Tensor] = []
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
        position = inputs.position_vectors[position_index][None].expand(batch, -1)
        read, read_weights = model._memory_read(
            controller, registers, trace, trace_addresses, memory_ablation
        )
        bit = multiplier_bits[:, position_index]
        controller = model.controller(
            torch.cat((bit[:, None], position, canvas.mean(-1), read), dim=-1),
            controller,
        )
        previous_registers = registers
        registers, write_weights = model._write_registers(
            controller, read, registers, memory_ablation
        )
        trace_token = torch.tanh(
            model.trace_projection(
                torch.cat((controller, read, canvas.mean(-1)), dim=-1)
            )
        )
        trace.append(trace_token)
        address = model.address_mlp(position)
        trace_addresses.append(address)
        canvas = model._update_canvas(
            canvas, static_features, controller, read, bit, position
        )
        if return_all:
            outputs.append(model.decode(canvas))
        if return_diagnostics:
            read_probability = read_weights[:, 0].clamp_min(1e-12)
            read_normalizer = math.log(max(2, read_probability.shape[-1]))
            write_probability = write_weights.clamp_min(1e-12)
            diagnostic_values["read_entropy"].append(
                -(read_probability * read_probability.log()).sum(-1)
                / read_normalizer
            )
            diagnostic_values["write_entropy"].append(
                -(write_probability * write_probability.log()).sum(-1)
                / math.log(model.register_slots)
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
    logits = torch.stack(outputs, dim=1) if return_all else model.decode(canvas)
    if not return_diagnostics:
        return logits
    diagnostics = {
        key: torch.stack(values, dim=1)
        for key, values in diagnostic_values.items()
    }
    return logits, diagnostics


class Exp654NeuralRAMAdapter(nn.Module):
    """Keep EXP654 inputs/model recurrence on one explicitly selected device."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @property
    def width(self) -> int:
        return int(self.model.width)

    def prepare(
        self,
        multiplier: list[int],
        multiplicand: list[int],
        modulus: list[int],
        width: int,
    ) -> NeuralRAMInputs:
        return prepare_neural_ram_inputs(
            multiplier=multiplier,
            multiplicand=multiplicand,
            modulus=modulus,
            width=width,
            device=module_device(self.model),
        )

    def forward(
        self,
        inputs: NeuralRAMInputs,
        *,
        return_all: bool = False,
        return_diagnostics: bool = False,
        memory_ablation: str = "none",
    ):
        device = module_device(self.model)
        if {
            inputs.multiplier_bits.device,
            inputs.static_features.device,
        } != {device}:
            raise RuntimeError("neural-RAM inputs and model are on different devices")
        return neural_ram_forward_prepared(
            self.model,
            inputs,
            return_all=return_all,
            return_diagnostics=return_diagnostics,
            memory_ablation=memory_ablation,
        )
