"""Evaluate target-free confidence routing into frozen width-4 learned search."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import mlflow
import psutil
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=15
    ).stdout.strip()


def decode(logits: torch.Tensor) -> list[int]:
    return [
        sum(int(bit) << position for position, bit in enumerate(row))
        for row in logits.ge(0).tolist()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/exp_625_tier6_confidence_routed_width4_search.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".scratch/exp-625-tier6-confidence-routed-width4-search",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    source = (ROOT / config["source_checkpoint"]).resolve()
    if sha256(source) != config["source_checkpoint_sha256"]:
        raise RuntimeError("source checkpoint hash mismatch")

    exp300 = load_module(
        "exp300_for_exp625",
        ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py",
    )
    cellular = load_module(
        "cellular_for_exp625",
        ROOT / "experiments/exp_220_width_independent_cellular/model.py",
    )
    beam_module = load_module(
        "beam_for_exp625",
        ROOT / "experiments/exp_621_tier6_learned_beam_decoding/decoder.py",
    )
    routing = load_module(
        "routing_for_exp625",
        ROOT / "experiments/exp_625_tier6_confidence_routed_width4_search/routing.py",
    )
    seed = int(config["seed"])
    width = int(config["evaluation"]["width"])
    split_config = {
        "data": {
            "prime_bits_min": width - config["data"]["prime_bit_span"] + 1,
            "prime_bits_max": width,
            "train_prime_identities": config["data"]["train_prime_identities"],
            "test_prime_identities": config["data"]["test_prime_identities"],
        }
    }
    split = exp300.bigint_prime_split(split_config, seed + 62500 + width)
    dataset_hash = sha256_text(json.dumps(split, sort_keys=True))
    torch.set_num_threads(config["evaluation"]["torch_threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()
    started = time.perf_counter()
    cpu_started = time.process_time()
    config_hash = sha256(config_path)
    runner_hash = sha256(Path(__file__))
    commit = git_output("rev-parse", "HEAD")
    experiment_id = configure_mlflow(config["tracking"]["experiment_name"])
    model = cellular.SharedDilatedConvGRU(
        width=width, hidden=config["model"]["hidden"], cycles=config["model"]["cycles"]
    )
    model.load_state_dict(load_file(source))
    model.eval()

    @torch.inference_mode()
    def greedy_confidence(left, right, modulus):
        batch = len(left)
        state = [0] * batch
        minimum = torch.full((batch,), math.inf)
        step_minimum_sum = torch.zeros(batch)
        last_minimum_sum = torch.zeros(batch)
        bit_nll_sum = torch.zeros(batch)
        quotient_entropy_sum = torch.zeros(batch)
        final_minimum = torch.zeros(batch)
        for position in range(width):
            bit = [(value >> (width - 1 - position)) & 1 for value in left]
            logits, quotient_logits = model(
                exp300.features(state, right, modulus, bit, width)
            )
            absolute = logits.abs()
            step_minimum = absolute.amin(dim=1)
            minimum = torch.minimum(minimum, step_minimum)
            step_minimum_sum += step_minimum
            if position >= width - 32:
                last_minimum_sum += step_minimum
            bit_nll_sum += (-F.logsigmoid(absolute)).mean(dim=1)
            quotient_probability = quotient_logits.softmax(dim=1)
            quotient_entropy_sum += -(
                quotient_probability * quotient_probability.clamp_min(1e-12).log()
            ).sum(dim=1)
            final_minimum = step_minimum
            state = decode(logits)
        scores = {
            "mean_bit_nll": (bit_nll_sum / width).tolist(),
            "negative_trajectory_min_margin": (-minimum).tolist(),
            "negative_mean_step_min_margin": (-step_minimum_sum / width).tolist(),
            "negative_last32_mean_min_margin": (-last_minimum_sum / 32).tolist(),
            "negative_final_min_margin": (-final_minimum).tolist(),
            "mean_quotient_entropy": (quotient_entropy_sum / width).tolist(),
        }
        return {"predictions": state, "confidence": scores}

    def evaluate(cases: int, sampler_seed: int):
        sampler = exp300.BigIntSampler(split["test"], sampler_seed)
        left, right, modulus = sampler.pairs(cases)
        expected = [
            (a * b) % prime for a, b, prime in zip(left, right, modulus)
        ]
        greedy_wall = time.perf_counter()
        greedy_cpu = time.process_time()
        greedy = greedy_confidence(left, right, modulus)
        greedy_runtime = {
            "wall_seconds": time.perf_counter() - greedy_wall,
            "aggregate_cpu_seconds": time.process_time() - greedy_cpu,
        }
        beam_wall = time.perf_counter()
        beam_cpu = time.process_time()
        beam = beam_module.LearnedLikelihoodBeamDecoder(model, width, 4).rollout(
            left, right, modulus, exp300.features
        )
        beam_runtime = {
            "wall_seconds": time.perf_counter() - beam_wall,
            "aggregate_cpu_seconds": time.process_time() - beam_cpu,
        }
        greedy_correct = [
            guess == truth for guess, truth in zip(greedy["predictions"], expected)
        ]
        beam_contains = [
            truth in states for truth, states in zip(expected, beam["beam_states"])
        ]
        beam_invalid = [
            guess >= prime for guess, prime in zip(beam["predictions"], modulus)
        ]
        route_count = round(cases * config["evaluation"]["routed_fraction"])
        confidence_routes = {
            name: routing.routed_coverage(
                greedy_correct,
                beam_contains,
                routing.ranked_indices(values, route_count),
            )
            for name, values in greedy["confidence"].items()
        }
        oracle_route = routing.oracle_allocation(
            greedy_correct, beam_contains, route_count
        )
        metrics = {
            "cases": cases,
            "greedy": {
                "correct": sum(greedy_correct),
                "exact_accuracy": sum(greedy_correct) / cases,
                "runtime": greedy_runtime,
            },
            "beam": {
                "selected_correct": sum(
                    guess == truth
                    for guess, truth in zip(beam["predictions"], expected)
                ),
                "oracle_contains_correct": sum(beam_contains),
                "oracle_contains_rate": sum(beam_contains) / cases,
                "selected_invalid_rate": sum(beam_invalid) / cases,
                "runtime": beam_runtime,
            },
            "routing": {
                "route_count": route_count,
                "unrestricted_union_correct": sum(
                    greedy_item or beam_item
                    for greedy_item, beam_item in zip(greedy_correct, beam_contains)
                ),
                "unrestricted_union_coverage": sum(
                    greedy_item or beam_item
                    for greedy_item, beam_item in zip(greedy_correct, beam_contains)
                )
                / cases,
                "oracle_allocation": routing.routed_coverage(
                    greedy_correct, beam_contains, oracle_route
                ),
                "confidence_allocations": confidence_routes,
            },
        }
        raw = {
            "left": left,
            "right": right,
            "modulus": modulus,
            "expected": expected,
            "greedy_predictions": greedy["predictions"],
            "confidence": greedy["confidence"],
            "beam_predictions": beam["predictions"],
            "beam_states": beam["beam_states"],
            "beam_scores": beam["beam_scores"],
            "oracle_route": oracle_route,
        }
        return metrics, raw

    evaluation_started = time.perf_counter()
    evaluation_cpu = time.process_time()
    metrics, raw = evaluate(
        config["evaluation"]["cases"], seed + 62600 + width
    )
    raw_path = output / "routing-predictions-confidence.json"
    raw_path.write_text(json.dumps(raw) + "\n")
    determinism = []
    for _ in range(2):
        repeat_metrics, repeat_raw = evaluate(
            config["evaluation"]["determinism_cases"], seed + 62700 + width
        )
        determinism.append({"metrics": repeat_metrics, "raw": repeat_raw})
    deterministic = (
        determinism[0]["raw"]["greedy_predictions"]
        == determinism[1]["raw"]["greedy_predictions"]
        and determinism[0]["raw"]["confidence"]
        == determinism[1]["raw"]["confidence"]
        and determinism[0]["raw"]["beam_predictions"]
        == determinism[1]["raw"]["beam_predictions"]
        and determinism[0]["raw"]["beam_states"]
        == determinism[1]["raw"]["beam_states"]
        and determinism[0]["metrics"]["routing"]
        == determinism[1]["metrics"]["routing"]
    )
    projection = config["runtime_projection"]
    gates = config["gates"]
    primary = config["evaluation"]["primary_confidence"]
    passed = (
        metrics["routing"]["unrestricted_union_coverage"]
        >= gates["minimum_unrestricted_union_coverage"]
        and metrics["routing"]["oracle_allocation"]["coverage"]
        >= gates["minimum_oracle_allocated_coverage"]
        and metrics["routing"]["confidence_allocations"][primary]["coverage"]
        >= gates["minimum_primary_confidence_allocated_coverage"]
        and projection["tier6_wall_seconds"] <= gates["maximum_tier6_wall_seconds"]
        and metrics["beam"]["selected_invalid_rate"]
        <= gates["maximum_beam_invalid_rate"]
        and deterministic
    )
    decision = config["decisions"]["advance" if passed else "reject"]
    evaluation_runtime = {
        "wall_seconds": time.perf_counter() - evaluation_started,
        "aggregate_cpu_seconds": time.process_time() - evaluation_cpu,
        "peak_rss_bytes": process.memory_info().rss,
    }
    with mlflow.start_run(run_name=config["run_id"]) as run:
        mlflow.set_tags(
            {
                "canonical_run_id": config["run_id"],
                "status": "completed",
                "decision": decision,
                "role": "tier6-confidence-routing-diagnostic",
                "repository_commit": commit,
                "repository_dirty": str(bool(git_output("status", "--porcelain"))).lower(),
            }
        )
        mlflow.log_params(
            {
                "seed": seed,
                "routed_fraction": config["evaluation"]["routed_fraction"],
                "primary_confidence": primary,
                "source_checkpoint_sha256": sha256(source),
                "config_sha256": config_hash,
                "runner_sha256": runner_hash,
                "dataset_sha256": dataset_hash,
            }
        )
        mlflow.log_metrics(
            {
                "greedy_accuracy": metrics["greedy"]["exact_accuracy"],
                "unrestricted_union_coverage": metrics["routing"]["unrestricted_union_coverage"],
                "oracle_allocated_coverage": metrics["routing"]["oracle_allocation"]["coverage"],
                "primary_confidence_coverage": metrics["routing"]["confidence_allocations"][primary]["coverage"],
                "tier6_projected_wall_seconds": projection["tier6_wall_seconds"],
                "wall_seconds": evaluation_runtime["wall_seconds"],
                "aggregate_cpu_seconds": evaluation_runtime["aggregate_cpu_seconds"],
            }
        )
        mlflow.log_artifact(str(config_path), artifact_path="preregistration")
        mlflow.log_artifact(str(ROOT / config["preregistration"]), artifact_path="preregistration")
        mlflow.log_artifact(str(raw_path), artifact_path="paired-evidence")
        summary = {
            "run_id": config["run_id"],
            "status": "completed",
            "hypothesis": config["hypothesis"],
            "mlflow": {
                "tracking_uri": tracking_uri(),
                "experiment_name": config["tracking"]["experiment_name"],
                "experiment_id": experiment_id,
                "run_id": run.info.run_id,
            },
            "source": {
                "repository_commit": commit,
                "dirty": bool(git_output("status", "--porcelain")),
                "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60",
                "config_sha256": config_hash,
                "runner_sha256": runner_hash,
                "routing_sha256": sha256(
                    ROOT
                    / "experiments/exp_625_tier6_confidence_routed_width4_search/routing.py"
                ),
                "dataset_sha256": dataset_hash,
                "source_checkpoint_sha256": sha256(source),
            },
            "metrics": metrics,
            "determinism": {"passed": deterministic},
            "runtime_projection": projection,
            "raw_evidence": {
                "path": str(raw_path.relative_to(ROOT)),
                "sha256": sha256(raw_path),
            },
            "gates": {**gates, "passed": passed},
            "decision": decision,
            "resource_budget": config["resource_budget"],
            "runtime": {
                "wall_seconds": time.perf_counter() - started,
                "aggregate_cpu_seconds": time.process_time() - cpu_started,
                "evaluation": evaluation_runtime,
                "torch_threads": config["evaluation"]["torch_threads"],
            },
            "limitations": [
                "Fresh generated diagnostic evidence only; checked and private answers were not read.",
                "Beam containment and oracle allocation are evaluation-only ceilings and never select inference outputs.",
                "This run does not train a router or reranker.",
                "Exact bigint arithmetic is confined to generated truth and post-inference scoring.",
            ],
        }
        summary_path = output / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        mlflow.log_artifact(str(summary_path), artifact_path="canonical-preview")
    print(
        json.dumps(
            {
                "decision": decision,
                "metrics": metrics,
                "determinism": deterministic,
                "runtime_projection": projection,
                "runtime": summary["runtime"],
                "mlflow": summary["mlflow"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
