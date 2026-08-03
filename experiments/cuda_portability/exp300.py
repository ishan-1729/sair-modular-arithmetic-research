"""Device adapter for the frozen EXP300 bit-tensor interface.

Python big integers remain an offline sampler/label boundary.  The adapter
converts each prepared batch once, keeps model recurrence on the selected
device, and converts results back only after the recurrent loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.nn.functional as F


class DeadlineLike(Protocol):
    def check(self, stage: str) -> None: ...


@dataclass(frozen=True)
class PreparedTrainingBatch:
    teacher_features: torch.Tensor
    recurrent_static: torch.Tensor
    target_bits: torch.Tensor
    quotients: torch.Tensor


class DeviceExp300:
    """Explicit-device facade preserving EXP300 sampling/loss semantics."""

    def __init__(
        self,
        parent: Any,
        device: torch.device,
        deadline: DeadlineLike,
    ) -> None:
        self.parent = parent
        self.device = device
        self.deadline = deadline
        self.BigIntSampler = parent.BigIntSampler

    def bits(self, values: list[int], width: int) -> torch.Tensor:
        rows = [
            [(value >> position) & 1 for position in range(width)]
            for value in values
        ]
        return torch.tensor(rows, dtype=torch.float32, device=self.device)

    def features(
        self,
        state: list[int],
        multiplicand: list[int],
        modulus: list[int],
        input_bit: list[int],
        width: int,
    ) -> torch.Tensor:
        batch = len(state)
        bit_plane = torch.tensor(
            input_bit, dtype=torch.float32, device=self.device
        )[:, None].expand(batch, width)
        lsb = torch.zeros(batch, width, device=self.device)
        msb = torch.zeros(batch, width, device=self.device)
        lsb[:, 0] = 1.0
        msb[:, -1] = 1.0
        return torch.stack(
            (
                self.bits(state, width),
                self.bits(multiplicand, width),
                self.bits(modulus, width),
                bit_plane,
                lsb,
                msb,
            ),
            dim=1,
        ).mul(2.0).sub(1.0)

    def transition_loss_tensors(
        self,
        logits: torch.Tensor,
        quotient_logits: torch.Tensor,
        target: torch.Tensor,
        quotients: torch.Tensor,
        worst_weight: float,
        quotient_weight: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        per_bit = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        )
        mean = per_bit.mean()
        worst = per_bit.mean(0).max()
        quotient = F.cross_entropy(quotient_logits, quotients)
        loss = mean + worst_weight * worst + quotient_weight * quotient
        exact = (logits.ge(0) == target.bool()).all(1).float().mean()
        return loss, {
            "loss": loss.detach(),
            "batch_exact": exact.detach(),
            "mean_bce": mean.detach(),
            "worst_bit_bce": worst.detach(),
            "quotient_ce": quotient.detach(),
        }

    def transition_loss(
        self,
        logits: torch.Tensor,
        quotient_logits: torch.Tensor,
        targets: list[int],
        quotients: torch.Tensor,
        width: int,
        worst_weight: float,
        quotient_weight: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss, tensors = self.transition_loss_tensors(
            logits,
            quotient_logits,
            self.bits(targets, width),
            quotients.to(self.device),
            worst_weight,
            quotient_weight,
        )
        return loss, {name: float(value) for name, value in tensors.items()}

    def prepare_training_batch(
        self,
        *,
        parent_base: Any,
        left: list[int],
        right: list[int],
        modulus: list[int],
        starts: list[int],
        width: int,
        horizon: int,
    ) -> PreparedTrainingBatch:
        """Prepare exact labels before the device-resident recurrent loop."""

        teacher = parent_base.teacher_prefix(left, right, modulus, starts)
        teacher_states: list[list[int]] = []
        flags_by_step: list[list[int]] = []
        targets_by_step: list[list[int]] = []
        quotients_by_step: list[list[int]] = []
        for offset in range(horizon):
            flags = [
                (value >> (width - 1 - start - offset)) & 1
                for value, start in zip(left, starts, strict=True)
            ]
            raw = [
                2 * state + flag * other
                for state, flag, other in zip(teacher, flags, right, strict=True)
            ]
            target = [
                value % prime for value, prime in zip(raw, modulus, strict=True)
            ]
            teacher_states.append(teacher)
            flags_by_step.append(flags)
            targets_by_step.append(target)
            quotients_by_step.append(
                [value // prime for value, prime in zip(raw, modulus, strict=True)]
            )
            teacher = target

        teacher_features = torch.stack(
            [
                self.features(state, right, modulus, flags, width)
                for state, flags in zip(
                    teacher_states, flags_by_step, strict=True
                )
            ]
        )
        batch = len(left)
        right_bits = self.bits(right, width).mul(2.0).sub(1.0)
        modulus_bits = self.bits(modulus, width).mul(2.0).sub(1.0)
        flag_bits = torch.tensor(
            flags_by_step, dtype=torch.float32, device=self.device
        )[:, :, None].expand(horizon, batch, width).mul(2.0).sub(1.0)
        lsb = torch.full((batch, width), -1.0, device=self.device)
        msb = torch.full((batch, width), -1.0, device=self.device)
        lsb[:, 0] = 1.0
        msb[:, -1] = 1.0
        recurrent_static = torch.stack(
            (
                right_bits.expand(horizon, -1, -1),
                modulus_bits.expand(horizon, -1, -1),
                flag_bits,
                lsb.expand(horizon, -1, -1),
                msb.expand(horizon, -1, -1),
            ),
            dim=2,
        )
        flat_targets = [value for step in targets_by_step for value in step]
        target_bits = self.bits(flat_targets, width).view(horizon, batch, width)
        quotients = torch.tensor(
            quotients_by_step, dtype=torch.long, device=self.device
        )
        return PreparedTrainingBatch(
            teacher_features=teacher_features,
            recurrent_static=recurrent_static,
            target_bits=target_bits,
            quotients=quotients,
        )

    @staticmethod
    def recurrent_features(
        state_plane: torch.Tensor, recurrent_static: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat(
            (state_plane[:, None, :].mul(2.0).sub(1.0), recurrent_static), dim=1
        )

    @torch.inference_mode()
    def evaluate_transitions(self, model: Any, sampler: Any, cases: int) -> dict:
        seen = correct = valid = 0
        while seen < cases:
            self.deadline.check("transition-evaluation")
            size = min(512, cases - seen)
            state, multiplicand, modulus, input_bit, target, _ = sampler.transitions(
                size
            )
            logits, _ = model(
                self.features(state, multiplicand, modulus, input_bit, model.width)
            )
            predicted = logits.ge(0)
            target_tensor = self.bits(target, model.width).bool()
            correct += int((predicted == target_tensor).all(1).sum().item())
            rows = predicted.cpu().tolist()
            predicted_values = [
                sum(int(row[position]) << position for position in range(model.width))
                for row in rows
            ]
            valid += sum(
                value < prime
                for value, prime in zip(predicted_values, modulus, strict=True)
            )
            seen += size
        return {
            "cases": seen,
            "correct": correct,
            "exact_accuracy": correct / seen,
            "valid_residue_rate": valid / seen,
        }


def unsigned_bits_ge(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Compare arbitrary-width little-endian bit rows without host decoding."""

    different = left.ne(right).flip(1)
    any_different = different.any(dim=1)
    first_from_msb = different.to(torch.int64).argmax(dim=1)
    gather_index = left.shape[1] - 1 - first_from_msb
    left_first = left.gather(1, gather_index[:, None]).squeeze(1)
    right_first = right.gather(1, gather_index[:, None]).squeeze(1)
    return (~any_different) | (left_first & ~right_first)
