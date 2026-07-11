from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class POCConfig:
    run_id: str = "poc-001-horner-tcn-smoke"
    seed: int = 20260711
    width: int = 8
    min_prime_bits: int = 4
    max_prime_bits: int | None = 6
    holdout_fraction: float = 0.25
    augment_composite_moduli: bool = True
    hidden: int = 32
    blocks: int = 5
    steps: int = 1500
    batch_size: int = 256
    learning_rate: float = 0.0025
    weight_decay: float = 0.00002
    worst_bit_weight: float = 0.35
    one_bit_probability: float = 0.70
    train_operand_bits: int = 16
    transition_cases: int = 20_000
    rollout_cases: int = 4_000
    torch_threads: int = 4

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "POCConfig":
        if "model" in raw:
            run = raw.get("run", {})
            model = raw.get("model", {})
            data = raw.get("data", {})
            training = raw.get("training", {})
            runtime = raw.get("runtime", {})
            return cls(
                run_id=str(run.get("id", cls.run_id)),
                seed=int(run.get("seed", cls.seed)),
                width=int(model.get("width", cls.width)),
                min_prime_bits=int(data.get("min_prime_bits", cls.min_prime_bits)),
                max_prime_bits=(
                    int(data["max_prime_bits"])
                    if data.get("max_prime_bits") is not None
                    else None
                ),
                holdout_fraction=float(data.get("heldout_prime_fraction", cls.holdout_fraction)),
                augment_composite_moduli=bool(
                    data.get("augment_composite_moduli", cls.augment_composite_moduli)
                ),
                hidden=int(model.get("channels", cls.hidden)),
                blocks=int(model.get("residual_blocks", cls.blocks)),
                steps=int(training.get("steps", cls.steps)),
                batch_size=int(training.get("batch_size", cls.batch_size)),
                learning_rate=float(training.get("learning_rate", cls.learning_rate)),
                weight_decay=float(training.get("weight_decay", cls.weight_decay)),
                worst_bit_weight=float(training.get("worst_bit_weight", cls.worst_bit_weight)),
                one_bit_probability=float(data.get("one_bit_probability", cls.one_bit_probability)),
                train_operand_bits=int(data.get("train_operand_bits", cls.train_operand_bits)),
                transition_cases=int(data.get("transition_cases", cls.transition_cases)),
                rollout_cases=int(data.get("rollout_cases", cls.rollout_cases)),
                torch_threads=int(runtime.get("torch_threads", cls.torch_threads)),
            )

        # Backward-compatible with the flat config embedded in the reference checkpoint.
        aliases = {
            "holdout_fraction": raw.get("holdout_fraction", cls.holdout_fraction),
            "hidden": raw.get("hidden", cls.hidden),
            "blocks": raw.get("blocks", cls.blocks),
            "learning_rate": raw.get("lr", raw.get("learning_rate", cls.learning_rate)),
            "train_operand_bits": raw.get("train_a_bits", raw.get("train_operand_bits", cls.train_operand_bits)),
            "transition_cases": raw.get("eval_transitions", raw.get("transition_cases", cls.transition_cases)),
            "rollout_cases": raw.get("eval_rollouts", raw.get("rollout_cases", cls.rollout_cases)),
            "torch_threads": raw.get("threads", raw.get("torch_threads", cls.torch_threads)),
            "augment_composite_moduli": raw.get(
                "augment_composite_moduli", cls.augment_composite_moduli
            ),
        }
        values = {field: raw.get(field, getattr(cls, field)) for field in cls.__dataclass_fields__}
        values.update(aliases)
        return cls(**values)


def load_config(path: str | Path) -> POCConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return POCConfig.from_mapping(raw)
