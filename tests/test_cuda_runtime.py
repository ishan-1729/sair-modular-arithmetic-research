from __future__ import annotations

import torch

from experiments.cuda_portability.exp300 import DeviceExp300, unsigned_bits_ge
from experiments.cuda_portability.runtime import resolve_device
from experiments.exp_699_h92_w256_backbone_factorial_complete import run as source


class _Deadline:
    def check(self, stage: str) -> None:
        assert stage


def _bool_bits(values: list[int], width: int) -> torch.Tensor:
    return torch.tensor(
        [
            [(value >> position) & 1 for position in range(width)]
            for value in values
        ],
        dtype=torch.bool,
    )


def test_auto_is_a_development_fallback_but_hardware_mode_fails_closed(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == torch.device("cpu")
    try:
        resolve_device("auto", require_accelerator=True)
    except RuntimeError as error:
        assert "requires an explicit CUDA device" in str(error)
    else:
        raise AssertionError("hardware acceptance silently fell back to CPU")


def test_arbitrary_width_unsigned_comparison_stays_tensor_native() -> None:
    width = 96
    left = [0, 7, (1 << 80) + 3, (1 << 95) - 1]
    right = [0, 8, (1 << 80) + 2, 1 << 95]
    actual = unsigned_bits_ge(_bool_bits(left, width), _bool_bits(right, width))
    assert actual.tolist() == [a >= b for a, b in zip(left, right, strict=True)]


def test_exp300_prepared_training_batch_matches_frozen_feature_semantics() -> None:
    parent = source.base.load_exp300()
    adapter = DeviceExp300(parent, torch.device("cpu"), _Deadline())
    width = 16
    left = [11, 7]
    right = [5, 9]
    modulus = [13, 17]
    starts = [2, 4]
    prepared = adapter.prepare_training_batch(
        parent_base=source.base,
        left=left,
        right=right,
        modulus=modulus,
        starts=starts,
        width=width,
        horizon=3,
    )
    teacher = source.base.teacher_prefix(left, right, modulus, starts)
    for offset in range(3):
        flags = [
            (value >> (width - 1 - start - offset)) & 1
            for value, start in zip(left, starts, strict=True)
        ]
        expected_features = parent.features(teacher, right, modulus, flags, width)
        assert torch.equal(prepared.teacher_features[offset], expected_features)
        raw = [
            2 * state + flag * other
            for state, flag, other in zip(teacher, flags, right, strict=True)
        ]
        target = [value % prime for value, prime in zip(raw, modulus, strict=True)]
        quotient = [value // prime for value, prime in zip(raw, modulus, strict=True)]
        assert torch.equal(
            prepared.target_bits[offset], parent.bits(target, width)
        )
        assert prepared.quotients[offset].tolist() == quotient
        teacher = target
