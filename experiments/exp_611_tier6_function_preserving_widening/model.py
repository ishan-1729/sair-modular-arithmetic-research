"""Exactly function-preserving residual widening for the cellular transition."""

from __future__ import annotations

from torch import Tensor, nn


class ResidualWidenedCell(nn.Module):
    """Retain a trained base cell and add a zero-output recurrent adapter."""

    def __init__(self, base: nn.Module, adapter: nn.Module) -> None:
        super().__init__()
        self.base = base
        self.adapter = adapter
        self.width = int(base.width)
        self._zero_adapter_outputs()

    def _zero_adapter_outputs(self) -> None:
        nn.init.zeros_(self.adapter.local_head.weight)
        nn.init.zeros_(self.adapter.local_head.bias)
        nn.init.zeros_(self.adapter.quotient_head[-1].weight)
        nn.init.zeros_(self.adapter.quotient_head[-1].bias)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        self.base.width = self.width
        self.adapter.width = self.width
        base_logits, base_quotient = self.base(features)
        adapter_logits, adapter_quotient = self.adapter(features)
        return base_logits + adapter_logits, base_quotient + adapter_quotient

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def base_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.base.parameters())

    def adapter_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.adapter.parameters())
