"""Train and gate Candidate007's input-only Tier-7 expert router."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any

import torch
from safetensors.torch import save_file
from torch import Tensor, nn
import torch.nn.functional as F
import yaml


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "exp-707-candidate007-tier-ensemble"
CONFIG_PATH = ROOT / "configs/exp_707_candidate007_tier_ensemble.yaml"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class Tier7Router(nn.Module):
    """Small input-only gate; it never receives targets or expert outputs."""

    def __init__(self, input_features: int, hidden: int, experts: int) -> None:
        super().__init__()
        self.hidden = nn.Linear(input_features, hidden)
        self.output = nn.Linear(hidden, experts)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.output(F.silu(self.hidden(inputs)))


def bit_rows(values: list[int], width: int) -> Tensor:
    return torch.tensor(
        [[(value >> position) & 1 for position in range(width)] for value in values],
        dtype=torch.float32,
    )


def router_features(rows: list[list[Any]], width: int, chunk_bits: int) -> Tensor:
    multiplier = [int(row[0]) for row in rows]
    multiplicand = [int(row[1]) for row in rows]
    modulus = [int(row[2]) for row in rows]
    planes = (
        bit_rows(multiplier, width),
        bit_rows(multiplicand, width),
        bit_rows(modulus, width),
        bit_rows([a ^ b for a, b in zip(multiplier, multiplicand, strict=True)], width),
        bit_rows([a & b for a, b in zip(multiplier, multiplicand, strict=True)], width),
    )
    if width % chunk_bits:
        raise RuntimeError("feature width must be divisible by feature chunk size")
    pooled = torch.stack(planes, dim=1).reshape(
        len(rows), len(planes), width // chunk_bits, chunk_bits
    ).mean(dim=-1)
    return pooled.flatten(1).mul_(2.0).sub_(1.0)


def bucket(namespace: str, modulus: int, count: int) -> int:
    digest = hashlib.sha256(f"{namespace}:{modulus}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % count


def selected_experts(logits: Tensor, margin: float) -> Tensor:
    teacher = logits[:, 0]
    alternative_value, alternative_index = logits[:, 1:].max(dim=1)
    route_alternative = alternative_value > teacher + margin
    return torch.where(route_alternative, alternative_index + 1, torch.zeros_like(alternative_index))


def correct_count(logits: Tensor, correctness: Tensor, margin: float) -> tuple[int, list[int]]:
    selected = selected_experts(logits, margin)
    chosen = correctness.gather(1, selected[:, None]).squeeze(1)
    counts = torch.bincount(selected, minlength=correctness.shape[1]).tolist()
    return int(chosen.sum().item()), [int(value) for value in counts]


def train_router(
    features: Tensor,
    correctness: Tensor,
    indices: Tensor,
    *,
    seed: int,
    hidden: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[Tier7Router, list[dict[str, float]]]:
    torch.manual_seed(seed)
    model = Tier7Router(features.shape[1], hidden, correctness.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    curves: list[dict[str, float]] = []
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(features[indices])
        loss = F.binary_cross_entropy_with_logits(logits, correctness[indices])
        loss.backward()
        optimizer.step()
        if epoch in {0, epochs // 4, epochs // 2, 3 * epochs // 4, epochs - 1}:
            curves.append({"epoch": float(epoch + 1), "loss": float(loss.detach())})
    return model.eval(), curves


def load_evidence(config: dict[str, Any]) -> tuple[list[str], list[list[Any]], Tensor]:
    source = ROOT / config["source"]["paired_evidence"]
    if sha256(source) != config["source"]["paired_evidence_sha256"]:
        raise RuntimeError("paired evidence hash mismatch")
    summary = json.loads(source.read_text(encoding="utf-8"))
    arms = list(config["experts"]["order"])
    records = {
        arm: summary["evaluations"][arm][str(config["source"]["width"])]["paired_vectors"]["rollout"]
        for arm in arms
    }
    rows = records[arms[0]]
    if len(rows) != int(config["source"]["cases"]):
        raise RuntimeError("unexpected paired-evidence denominator")
    for arm in arms[1:]:
        if any(
            tuple(records[arm][index][:4]) != tuple(row[:4])
            for index, row in enumerate(rows)
        ):
            raise RuntimeError(f"paired evidence is not aligned for {arm}")
    correctness = torch.tensor(
        [
            [int(records[arm][index][4] == row[3]) for arm in arms]
            for index, row in enumerate(rows)
        ],
        dtype=torch.float32,
    )
    return arms, rows, correctness


def run(config_path: Path, output: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["run_id"] != RUN_ID:
        raise RuntimeError("unexpected run_id")
    for arm in config["experts"]["order"]:
        path = ROOT / config["experts"][arm]["path"]
        if not path.is_file() or sha256(path) != config["experts"][arm]["sha256"]:
            raise RuntimeError(f"expert checkpoint mismatch: {arm}")

    torch.set_num_threads(int(config["evaluation"]["torch_threads"]))
    try:
        torch.set_num_interop_threads(int(config["evaluation"]["interop_threads"]))
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)

    arms, rows, correctness = load_evidence(config)
    router = config["router"]
    features = router_features(
        rows,
        int(router["feature_width"]),
        int(router["feature_chunk_bits"]),
    )
    if features.shape[1] != int(router["input_features"]):
        raise RuntimeError("router feature count mismatch")
    buckets = torch.tensor(
        [
            bucket(router["split_namespace"], int(row[2]), int(router["bucket_count"]))
            for row in rows
        ],
        dtype=torch.long,
    )
    fit_indices = (buckets >= 2).nonzero().flatten()
    calibration_indices = (buckets == int(router["calibration_bucket"])).nonzero().flatten()
    sealed_indices = (buckets == int(router["sealed_bucket"])).nonzero().flatten()
    if min(len(fit_indices), len(calibration_indices), len(sealed_indices)) == 0:
        raise RuntimeError("empty grouped split")

    model, development_curve = train_router(
        features,
        correctness,
        fit_indices,
        seed=int(config["seed"]),
        hidden=int(router["hidden"]),
        epochs=int(router["epochs"]),
        learning_rate=float(router["learning_rate"]),
        weight_decay=float(router["weight_decay"]),
    )
    with torch.inference_mode():
        logits = model(features)
    margin_rows = []
    for margin in [float(value) for value in router["route_margin_grid"]]:
        correct, routes = correct_count(
            logits[calibration_indices], correctness[calibration_indices], margin
        )
        margin_rows.append({"margin": margin, "correct": correct, "routes": routes})
    selected_margin = max(
        margin_rows,
        key=lambda row: (row["correct"], row["margin"]),
    )["margin"]

    def split_result(name: str, indices: Tensor) -> dict[str, Any]:
        selected_correct, routes = correct_count(
            logits[indices], correctness[indices], float(selected_margin)
        )
        per_expert = {
            arm: int(correctness[indices, index].sum().item())
            for index, arm in enumerate(arms)
        }
        oracle = int(correctness[indices].amax(dim=1).sum().item())
        return {
            "name": name,
            "cases": len(indices),
            "router_correct": selected_correct,
            "router_accuracy": selected_correct / len(indices),
            "teacher_cosine_correct": per_expert["teacher_cosine"],
            "teacher_cosine_accuracy": per_expert["teacher_cosine"] / len(indices),
            "gain_over_teacher_cosine": selected_correct - per_expert["teacher_cosine"],
            "oracle_correct": oracle,
            "oracle_accuracy": oracle / len(indices),
            "per_expert_correct": per_expert,
            "route_counts": dict(zip(arms, routes, strict=True)),
            "moduli": len({int(rows[index][2]) for index in indices.tolist()}),
        }

    split_results = {
        "fit": split_result("fit", fit_indices),
        "calibration": split_result("calibration", calibration_indices),
        "sealed": split_result("sealed", sealed_indices),
    }
    sealed_gain = split_results["sealed"]["gain_over_teacher_cosine"]
    gate_passed = sealed_gain >= int(
        config["gates"]["minimum_sealed_correct_gain_over_teacher_cosine"]
    )

    output.mkdir(parents=True, exist_ok=True)
    final_curve: list[dict[str, float]] = []
    router_path: Path | None = None
    if gate_passed:
        final_model, final_curve = train_router(
            features,
            correctness,
            torch.arange(len(rows)),
            seed=int(config["seed"]) + 1,
            hidden=int(router["hidden"]),
            epochs=int(router["epochs"]),
            learning_rate=float(router["learning_rate"]),
            weight_decay=float(router["weight_decay"]),
        )
        router_path = output / "router.safetensors"
        save_file(
            {key: value.detach().cpu().contiguous() for key, value in final_model.state_dict().items()},
            str(router_path),
        )

    result = {
        "schema_version": "exp707-router-v1",
        "run_id": RUN_ID,
        "status": "router-gate-passed" if gate_passed else "completed-negative-router-gate",
        "created_at": utc_now(),
        "training_executed": True,
        "training_scope": "tier7-input-only-router",
        "expert_training_executed": False,
        "config_sha256": sha256(config_path),
        "paired_evidence_sha256": config["source"]["paired_evidence_sha256"],
        "feature_sha256": canonical_hash(features.tolist()),
        "correctness_sha256": canonical_hash(correctness.int().tolist()),
        "arms": arms,
        "split_counts": {
            "fit": len(fit_indices),
            "calibration": len(calibration_indices),
            "sealed": len(sealed_indices),
        },
        "development_curve": development_curve,
        "margin_calibration": margin_rows,
        "selected_margin": selected_margin,
        "splits": split_results,
        "gate": {
            "minimum_sealed_gain": int(config["gates"]["minimum_sealed_correct_gain_over_teacher_cosine"]),
            "actual_sealed_gain": sealed_gain,
            "passed": gate_passed,
        },
        "final_training_curve": final_curve,
        "router": None
        if router_path is None
        else {
            "path": str(router_path),
            "sha256": sha256(router_path),
            "parameters": sum(value.numel() for value in final_model.state_dict().values()),
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": "cpu",
            "deterministic_algorithms": True,
        },
        "limitations": [
            "The sealed gate estimates routing generalization but is not a private benchmark.",
            "Final Candidate007 promotion still requires frozen five-seed and official runtime gates.",
            "The incumbent pre-reduction path remains blocked by current upstream rules.",
        ],
    }
    write_json(output / "router-result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=ROOT / ".scratch/exp707-candidate007")
    args = parser.parse_args()
    result = run(args.config.resolve(), args.output.resolve())
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
