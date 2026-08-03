"""Device-resident adapters for the frozen EXP401/EXP510 PyG route."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .runtime import module_device


class DeviceGraphCache:
    """Prepare immutable graph topology once per batch/width/device."""

    def __init__(self, frozen_graph_module: Any) -> None:
        self.frozen = frozen_graph_module
        self._edges: dict[
            tuple[int, int, int, str], tuple[Tensor, Tensor]
        ] = {}

    def get(
        self,
        batch_size: int,
        width: int,
        dilation: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        key = (batch_size, width, dilation, str(device))
        if key not in self._edges:
            edge_index, edge_attr = self.frozen.cpu_edges(
                batch_size, width, dilation
            )
            self._edges[key] = (
                edge_index.to(device=device, non_blocking=True),
                edge_attr.to(device=device, non_blocking=True),
            )
        return self._edges[key]

    def prepare(
        self,
        *,
        batch_size: int,
        width: int,
        cycles: int,
        device: torch.device,
    ) -> list[tuple[Tensor, Tensor]]:
        return [
            self.get(batch_size, width, dilation, device)
            for dilation in self.frozen.dilation_schedule(width, cycles)
        ]

    def clear(self) -> None:
        self._edges.clear()


def typed_max_forward_prepared(
    model: nn.Module,
    features: Tensor,
    prepared_edges: list[tuple[Tensor, Tensor]],
) -> tuple[Tensor, Tensor]:
    """Run the EXP510 typed-max recurrence with no transfer in its loop."""

    batch_size, _, width = features.shape
    nodes = model.stem(features.transpose(1, 2)).reshape(
        batch_size * width, model.hidden
    )
    for edge_index, edge_attr in prepared_edges:
        message = model.conv(nodes, edge_index, edge_attr)
        nodes = model.update(message, nodes)
    return model.finish(nodes, batch_size, width)


class Exp510TypedMaxAdapter(nn.Module):
    """Wrap the frozen EXP510-selected PyG model with a device edge cache."""

    def __init__(self, model: nn.Module, frozen_graph_module: Any) -> None:
        super().__init__()
        self.model = model
        self.cache = DeviceGraphCache(frozen_graph_module)

    @property
    def width(self) -> int:
        return int(self.model.width)

    @width.setter
    def width(self, value: int) -> None:
        self.model.width = int(value)

    def parameter_count(self) -> int:
        return int(self.model.parameter_count())

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        device = module_device(self.model)
        if features.device != device:
            raise RuntimeError(
                f"graph features are on {features.device}, model is on {device}"
            )
        batch_size, _, width = features.shape
        prepared = self.cache.prepare(
            batch_size=batch_size,
            width=width,
            cycles=int(self.model.cycles),
            device=device,
        )
        return typed_max_forward_prepared(self.model, features, prepared)


def move_graph_batch_to_model(batch: Any, model: nn.Module) -> Any:
    """Move a PyG Data/Batch object once at the model invocation boundary."""

    device = module_device(model)
    moved = batch.to(device=device, non_blocking=True)
    tensor_devices = {
        value.device
        for value in moved.to_dict().values()
        if isinstance(value, Tensor)
    }
    if tensor_devices and tensor_devices != {device}:
        raise RuntimeError(f"graph batch has mixed devices: {tensor_devices}")
    return moved
