"""Measure a tensor-native cached rollout route against Candidate005 semantics."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import random
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Callable

import mlflow
import psutil
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "exp-656-tensor-native-rollout-route"
CHALLENGE_COMMIT = "82510bba00a1126649bd76dd1a451f14d0b3eb60"
sys.path.insert(0, str(ROOT))
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def process_cpu_seconds() -> float:
    times = psutil.Process().cpu_times()
    return float(times.user + times.system)


def peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_source(config: dict[str, Any]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for stem in ("candidate_model", "candidate_manifest", "checkpoint"):
        path = ROOT / config["source"][stem]
        expected = str(config["source"][f"{stem}_sha256"])
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"{stem} hash mismatch: {actual} != {expected}")
        resolved[f"{stem}_sha256"] = actual
    return resolved


def decode_boolean_bits(rows: torch.Tensor) -> list[int]:
    if rows.dtype is not torch.bool:
        raise TypeError("decode_boolean_bits requires a boolean tensor")
    return [
        sum(int(bit) << position for position, bit in enumerate(row))
        for row in rows.tolist()
    ]


@torch.inference_mode()
def legacy_rollout(
    candidate_module,
    network,
    multiplier: list[int],
    multiplicand: list[int],
    modulus: list[int],
    width: int,
) -> tuple[list[int], torch.Tensor]:
    """Reproduce the current wide Candidate005 Python-bigint route."""
    state = [0] * len(multiplier)
    minimum_margin = torch.full((len(multiplier),), math.inf)
    for bit_index in range(width - 1, -1, -1):
        input_bits = [(value >> bit_index) & 1 for value in multiplier]
        logits, _ = network(
            candidate_module._bigint_features(
                state, multiplicand, modulus, input_bits, width
            )
        )
        minimum_margin = torch.minimum(
            minimum_margin, logits.abs().amin(dim=1)
        )
        state = candidate_module._decode_bigint_logits(logits)
    return state, minimum_margin


@torch.inference_mode()
def tensor_native_rollout(
    candidate_module,
    network,
    multiplier: list[int],
    multiplicand: list[int],
    modulus: list[int],
    width: int,
) -> tuple[list[int], torch.Tensor]:
    """Cache invariant planes and retain thresholded state on the tensor route."""
    batch = len(multiplier)
    multiplier_bits = candidate_module._bigint_bits(multiplier, width)
    features = torch.empty((batch, 6, width), dtype=torch.float32)
    features[:, 1, :] = candidate_module._bigint_bits(
        multiplicand, width
    ).mul(2.0).sub(1.0)
    features[:, 2, :] = candidate_module._bigint_bits(modulus, width).mul(
        2.0
    ).sub(1.0)
    features[:, 4, :].fill_(-1.0)
    features[:, 4, 0] = 1.0
    features[:, 5, :].fill_(-1.0)
    features[:, 5, -1] = 1.0
    state_bits = torch.zeros((batch, width), dtype=torch.bool)
    minimum_margin = torch.full((batch,), math.inf)
    for bit_index in range(width - 1, -1, -1):
        features[:, 0, :] = state_bits.to(torch.float32).mul(2.0).sub(1.0)
        features[:, 3, :] = multiplier_bits[:, bit_index, None].mul(2.0).sub(1.0)
        logits, _ = network(features)
        minimum_margin = torch.minimum(
            minimum_margin, logits.abs().amin(dim=1)
        )
        state_bits = logits.ge(0)
    return decode_boolean_bits(state_bits), minimum_margin


def make_cases(width: int, batch_size: int, seed: int) -> tuple[list[int], ...]:
    """Generate input-only wide values; no answer or accuracy label is produced."""
    rng = random.Random(seed + width)
    moduli: list[int] = []
    multipliers: list[int] = []
    multiplicands: list[int] = []
    for _ in range(batch_size):
        modulus = (1 << (width - 1)) | rng.getrandbits(width - 1) | 1
        moduli.append(modulus)
        multipliers.append(rng.randrange(modulus))
        multiplicands.append(rng.randrange(modulus))
    return multipliers, multiplicands, moduli


def output_hash(outputs: list[int]) -> str:
    return sha256_text(canonical_json(outputs))


def margin_hash(margins: torch.Tensor) -> str:
    return hashlib.sha256(margins.contiguous().numpy().tobytes()).hexdigest()


def measured_call(function: Callable, *args) -> tuple[list[int], torch.Tensor, dict[str, float]]:
    gc.collect()
    wall_started = time.perf_counter()
    cpu_started = process_cpu_seconds()
    outputs, margins = function(*args)
    timing = {
        "wall_seconds": time.perf_counter() - wall_started,
        "aggregate_process_cpu_seconds": process_cpu_seconds() - cpu_started,
    }
    return outputs, margins, timing


def check_budget(started: float, cpu_started: float, config: dict[str, Any]) -> None:
    wall = time.perf_counter() - started
    cpu = process_cpu_seconds() - cpu_started
    budget = config["resource_budget"]
    if wall > float(budget["hard_wall_seconds"]):
        raise RuntimeError(f"hard wall ceiling exceeded: {wall:.3f}")
    if cpu > float(budget["hard_aggregate_process_cpu_seconds"]):
        raise RuntimeError(f"hard aggregate CPU ceiling exceeded: {cpu:.3f}")


def assert_parity(
    legacy: tuple[list[int], torch.Tensor],
    tensor_native: tuple[list[int], torch.Tensor],
) -> None:
    if legacy[0] != tensor_native[0]:
        raise RuntimeError("semantic mismatch: final outputs differ")
    if not torch.equal(legacy[1], tensor_native[1]):
        difference = float((legacy[1] - tensor_native[1]).abs().max())
        raise RuntimeError(f"semantic mismatch: minimum margins differ by {difference}")


def benchmark_width(
    candidate_module,
    network,
    width: int,
    config: dict[str, Any],
    started: float,
    cpu_started: float,
) -> dict[str, Any]:
    benchmark = config["benchmark"]
    cases = make_cases(width, int(benchmark["batch_size"]), int(config["seed"]))
    legacy_args = (candidate_module, network, *cases, width)
    native_args = (candidate_module, network, *cases, width)

    warm_legacy = legacy_rollout(*legacy_args)
    check_budget(started, cpu_started, config)
    warm_native = tensor_native_rollout(*native_args)
    assert_parity(warm_legacy, warm_native)
    check_budget(started, cpu_started, config)

    repeats: list[dict[str, Any]] = []
    count = int(benchmark["measured_repeats"])
    for repeat in range(count):
        order = ["legacy", "tensor_native"] if repeat % 2 == 0 else ["tensor_native", "legacy"]
        observed: dict[str, tuple[list[int], torch.Tensor, dict[str, float]]] = {}
        for route in order:
            function, args = (
                (legacy_rollout, legacy_args)
                if route == "legacy"
                else (tensor_native_rollout, native_args)
            )
            observed[route] = measured_call(function, *args)
            check_budget(started, cpu_started, config)
        assert_parity(
            (observed["legacy"][0], observed["legacy"][1]),
            (observed["tensor_native"][0], observed["tensor_native"][1]),
        )
        repeats.append(
            {
                "repeat": repeat + 1,
                "order": order,
                "legacy": {
                    **observed["legacy"][2],
                    "output_sha256": output_hash(observed["legacy"][0]),
                    "minimum_margin_sha256": margin_hash(observed["legacy"][1]),
                },
                "tensor_native": {
                    **observed["tensor_native"][2],
                    "output_sha256": output_hash(observed["tensor_native"][0]),
                    "minimum_margin_sha256": margin_hash(observed["tensor_native"][1]),
                },
                "exact_output_parity": True,
                "exact_minimum_margin_parity": True,
            }
        )

    legacy_wall = [item["legacy"]["wall_seconds"] for item in repeats]
    native_wall = [item["tensor_native"]["wall_seconds"] for item in repeats]
    legacy_cpu = [item["legacy"]["aggregate_process_cpu_seconds"] for item in repeats]
    native_cpu = [item["tensor_native"]["aggregate_process_cpu_seconds"] for item in repeats]
    legacy_wall_median = median(legacy_wall)
    native_wall_median = median(native_wall)
    legacy_cpu_median = median(legacy_cpu)
    native_cpu_median = median(native_cpu)
    return {
        "width": width,
        "batch_size": len(cases[0]),
        "input_sha256": sha256_text(canonical_json(cases)),
        "warmup": {
            "routes": ["legacy", "tensor_native"],
            "exact_output_parity": True,
            "exact_minimum_margin_parity": True,
            "output_sha256": output_hash(warm_legacy[0]),
            "minimum_margin_sha256": margin_hash(warm_legacy[1]),
        },
        "repeats": repeats,
        "summary": {
            "legacy_median_wall_seconds": legacy_wall_median,
            "tensor_native_median_wall_seconds": native_wall_median,
            "tensor_to_legacy_wall_ratio": native_wall_median / legacy_wall_median,
            "wall_reduction_fraction": 1.0 - native_wall_median / legacy_wall_median,
            "legacy_median_aggregate_process_cpu_seconds": legacy_cpu_median,
            "tensor_native_median_aggregate_process_cpu_seconds": native_cpu_median,
            "tensor_to_legacy_aggregate_cpu_ratio": native_cpu_median / legacy_cpu_median,
            "all_exact_output_parity": True,
            "all_exact_minimum_margin_parity": True,
        },
    }


def decide(widths: dict[str, dict[str, Any]], config: dict[str, Any]) -> tuple[str, dict[str, bool]]:
    gates = config["gates"]
    parity = all(
        record["summary"]["all_exact_output_parity"]
        and record["summary"]["all_exact_minimum_margin_parity"]
        for record in widths.values()
    )
    width128 = (
        widths["128"]["summary"]["tensor_to_legacy_wall_ratio"]
        <= float(gates["maximum_width128_tensor_to_legacy_wall_ratio"])
    )
    width256 = (
        widths["256"]["summary"]["tensor_to_legacy_wall_ratio"]
        <= float(gates["maximum_width256_tensor_to_legacy_wall_ratio"])
    )
    passed = parity and width128 and width256
    return (
        "advance-tensor-native-route" if passed else "reject-tensor-native-route",
        {
            "semantic_parity": parity,
            "width128_non_regression": width128,
            "width256_minimum_20_percent_wall_reduction": width256,
            "all_passed": passed,
        },
    )


def environment_payload(config: dict[str, Any]) -> dict[str, Any]:
    dependencies = {}
    for name in ("mlflow", "psutil", "safetensors", "torch", "PyYAML"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = "unavailable"
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "executable": sys.executable,
        "device": "cpu",
        "torch_threads": int(config["benchmark"]["torch_threads"]),
        "torch_interop_threads": 1,
        "deterministic_algorithms": True,
        "dependencies": dependencies,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/exp_656_tensor_native_rollout_route.yaml",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["run_id"] != RUN_ID:
        raise RuntimeError("unexpected run_id")
    source_hashes = validate_source(config)
    run_dir = ROOT / config["outputs"]["run_dir"]
    metrics_path = ROOT / config["outputs"]["metrics"]
    run_path = run_dir / "run.json"
    if run_path.exists() and not args.overwrite:
        raise RuntimeError(f"durable run already exists: {run_path}")

    torch.set_num_threads(int(config["benchmark"]["torch_threads"]))
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    candidate_module = load_module(
        "candidate005_for_exp656", ROOT / config["source"]["candidate_model"]
    )
    candidate = candidate_module.Tier2HornerModel()
    candidate.load(str(ROOT / "submission"))
    network = candidate.tier6_network
    if network is None:
        raise RuntimeError("Candidate005 Tier-6 network did not load")
    network.eval()

    experiment_id = configure_mlflow(config["tracking"]["experiment_name"])
    started = time.perf_counter()
    cpu_started = process_cpu_seconds()
    created_at = datetime.now(timezone.utc).isoformat()
    repository_commit = git_output("rev-parse", "HEAD")
    repository_dirty = bool(git_output("status", "--porcelain"))
    with mlflow.start_run(run_name=RUN_ID) as active:
        mlflow_run_id = active.info.run_id
        mlflow.set_tags(
            {
                "canonical_run_id": RUN_ID,
                "status": "running",
                "role": "input-only-route-ab",
                "repository_commit": repository_commit,
                "repository_dirty": str(repository_dirty).lower(),
                "scientific_accuracy_reported": "false",
            }
        )
        mlflow.log_params(
            {
                "seed": int(config["seed"]),
                "widths": ",".join(map(str, config["benchmark"]["widths"])),
                "batch_size": int(config["benchmark"]["batch_size"]),
                "measured_repeats": int(config["benchmark"]["measured_repeats"]),
                "torch_threads": int(config["benchmark"]["torch_threads"]),
                **source_hashes,
            }
        )
        widths = {
            str(width): benchmark_width(
                candidate_module,
                network,
                int(width),
                config,
                started,
                cpu_started,
            )
            for width in config["benchmark"]["widths"]
        }
        decision, gates = decide(widths, config)
        total_wall = time.perf_counter() - started
        total_cpu = process_cpu_seconds() - cpu_started
        resource_payload = {
            "wall_seconds": total_wall,
            "aggregate_process_cpu_seconds": total_cpu,
            "peak_rss_bytes": peak_rss_bytes(),
            "hard_wall_seconds": float(config["resource_budget"]["hard_wall_seconds"]),
            "hard_aggregate_process_cpu_seconds": float(
                config["resource_budget"]["hard_aggregate_process_cpu_seconds"]
            ),
        }
        metrics = {
            "run_id": RUN_ID,
            "created_at": created_at,
            "checkpoint_sha256": source_hashes["checkpoint_sha256"],
            "status": "completed",
            "experiment_class": "input-only-route-performance",
            "scientific_accuracy_reported": False,
            "source": source_hashes,
            "evaluation": {
                "not_performed": True,
                "reason": "Input-only route performance A/B; no answer labels or oracle were generated or read.",
                "labels_or_oracle_present": False,
            },
            "controls": {
                "semantic_and_performance_gates": gates,
                "matched_input_only_route_comparison": True,
            },
            "benchmark": widths,
            "gates": gates,
            "decision": decision,
            "resource": resource_payload,
            "runtime": resource_payload,
            "limitations": [
                "This input-only A/B reports no exact-answer accuracy and cannot promote a candidate.",
                "The width-256 network invocation is a research-route probe; Candidate005 still returns zero above its learned 128-bit submission boundary.",
                "Timings are local CPU measurements from a dirty research worktree and require confirmation before submission integration.",
                "Python big integers are used only to generate inputs and by the frozen legacy representation; neither route performs arithmetic repair.",
            ],
        }
        environment = environment_payload(config)
        mlflow_payload = {
            "tracking_uri": tracking_uri(),
            "experiment_name": config["tracking"]["experiment_name"],
            "experiment_id": experiment_id,
            "run_id": mlflow_run_id,
        }
        run_record = {
            "run_id": RUN_ID,
            "created_at": created_at,
            "status": "completed",
            "hypothesis": config["hypothesis"],
            "seed": int(config["seed"]),
            "source": {
                "repository_commit": repository_commit,
                "dirty": repository_dirty,
                "repository_dirty": repository_dirty,
                "challenge_commit": CHALLENGE_COMMIT,
                "runner_sha256": sha256(Path(__file__)),
                "config_sha256": sha256(config_path),
                **source_hashes,
            },
            "config": str(config_path.relative_to(ROOT)),
            "artifacts": {
                "checkpoint": config["source"]["checkpoint"],
                "metrics": str(metrics_path.relative_to(ROOT)),
            },
            "executed_config": str((run_dir / "executed-config.yaml").relative_to(ROOT)),
            "environment": str((run_dir / "environment.json").relative_to(ROOT)),
            "metrics": str(metrics_path.relative_to(ROOT)),
            "mlflow": mlflow_payload,
            "decision": decision,
            "limitations": metrics["limitations"],
            "notes": [
                "Input-only route performance evidence; no scientific evaluation was performed."
            ],
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "executed-config.yaml").write_text(
            config_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        atomic_json(run_dir / "environment.json", environment)
        atomic_json(run_dir / "mlflow.json", mlflow_payload)
        atomic_json(metrics_path, metrics)
        atomic_json(run_path, run_record)
        for width, record in widths.items():
            mlflow.log_metrics(
                {
                    f"width{width}_tensor_to_legacy_wall_ratio": record["summary"][
                        "tensor_to_legacy_wall_ratio"
                    ],
                    f"width{width}_wall_reduction_fraction": record["summary"][
                        "wall_reduction_fraction"
                    ],
                    f"width{width}_tensor_to_legacy_cpu_ratio": record["summary"][
                        "tensor_to_legacy_aggregate_cpu_ratio"
                    ],
                }
            )
        mlflow.log_metrics(
            {
                "total_wall_seconds": total_wall,
                "total_aggregate_process_cpu_seconds": total_cpu,
                "all_gates_passed": int(gates["all_passed"]),
            }
        )
        mlflow.log_artifacts(str(run_dir), artifact_path="canonical-run")
        mlflow.log_artifact(str(metrics_path), artifact_path="canonical-metrics")
        mlflow.log_artifact(str(config_path), artifact_path="preregistration")
        mlflow.log_artifact(
            str(ROOT / config["preregistration"]), artifact_path="preregistration"
        )
        mlflow.set_tags({"status": "completed", "decision": decision})
    print(
        json.dumps(
            {
                "run_id": RUN_ID,
                "decision": decision,
                "gates": gates,
                "ratios": {
                    width: record["summary"]["tensor_to_legacy_wall_ratio"]
                    for width, record in widths.items()
                },
                "resource": metrics["resource"],
                "mlflow_run_id": mlflow_run_id,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
