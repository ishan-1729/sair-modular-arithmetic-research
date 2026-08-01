"""Transfer common radix-2 cellular parameters into the radix-4 model."""

from __future__ import annotations

import torch


FEATURE_MAP = {0: ((0, 1.0),), 1: ((1, 1.0),), 2: ((2, 1.0),), 3: ((3, 1 / 3), (4, 2 / 3)), 4: ((5, 1.0),), 5: ((6, 1.0),)}


def transfer_radix2_state(
    target: torch.nn.Module, source_state: dict[str, torch.Tensor]
) -> dict[str, object]:
    """Copy compatible local computation and expand digit/quotient parameters."""
    target_state = target.state_dict()
    hidden = int(target.hidden)
    copied = []

    for key, value in source_state.items():
        if key in {"stem.weight", "gate_weight", "candidate_weight", "quotient_head.2.weight", "quotient_head.2.bias"}:
            continue
        if key in target_state and target_state[key].shape == value.shape:
            target_state[key].copy_(value)
            copied.append(key)

    target_state["stem.weight"].zero_()
    for old_feature, destinations in FEATURE_MAP.items():
        for new_feature, scale in destinations:
            target_state["stem.weight"][:, new_feature].add_(
                source_state["stem.weight"][:, old_feature] * scale
            )
    copied.append("stem.weight")

    for key in ("gate_weight", "candidate_weight"):
        target_state[key][:, :hidden].copy_(source_state[key][:, :hidden])
        target_state[key][:, hidden:].zero_()
        for old_feature, destinations in FEATURE_MAP.items():
            for new_feature, scale in destinations:
                target_state[key][:, hidden + new_feature].add_(
                    source_state[key][:, hidden + old_feature] * scale
                )
        copied.append(key)

    target_state["quotient_head.2.weight"][:3].copy_(
        source_state["quotient_head.2.weight"]
    )
    target_state["quotient_head.2.bias"][:3].copy_(
        source_state["quotient_head.2.bias"]
    )
    copied.extend(("quotient_head.2.weight[:3]", "quotient_head.2.bias[:3]"))
    target.load_state_dict(target_state)
    return {
        "copied_components": copied,
        "digit_initialization": "normalized low/3 plus 2*high/3 projection",
        "copied_quotient_classes": [0, 1, 2],
        "new_quotient_classes": [3, 4, 5, 6],
    }

