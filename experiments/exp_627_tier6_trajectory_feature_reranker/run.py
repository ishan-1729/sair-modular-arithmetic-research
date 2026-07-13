"""Train and evaluate a compact reranker for confidence-routed beam states."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import psutil
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file, save_file


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
        default=ROOT / "configs/exp_627_tier6_trajectory_feature_reranker.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".scratch/exp-627-tier6-trajectory-feature-reranker",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    source = (ROOT / config["source_checkpoint"]).resolve()
    if sha256(source) != config["source_checkpoint_sha256"]:
        raise RuntimeError("source checkpoint hash mismatch")

    exp300 = load_module(
        "exp300_for_exp627",
        ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py",
    )
    cellular = load_module(
        "cellular_for_exp627",
        ROOT / "experiments/exp_220_width_independent_cellular/model.py",
    )
    beam_module = load_module(
        "beam_for_exp627",
        ROOT / "experiments/exp_621_tier6_learned_beam_decoding/decoder.py",
    )
    feature_module = load_module(
        "features_for_exp627",
        ROOT / "experiments/exp_627_tier6_trajectory_feature_reranker/features.py",
    )
    reranker_module = load_module(
        "model_for_exp627",
        ROOT / "experiments/exp_627_tier6_trajectory_feature_reranker/model.py",
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
    prime_split = exp300.bigint_prime_split(split_config, seed + 62700 + width)
    validation_count = config["data"]["validation_prime_identities"]
    primes = {
        "train": prime_split["train"],
        "validation": prime_split["test"][:validation_count],
        "selection": prime_split["test"][validation_count:],
    }
    dataset_hash = sha256_text(json.dumps(primes, sort_keys=True))
    torch.set_num_threads(config["training"]["torch_threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache_dir = output / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()
    started = time.perf_counter()
    cpu_started = time.process_time()
    config_hash = sha256(config_path)
    runner_hash = sha256(Path(__file__))
    feature_hash = sha256(
        ROOT / "experiments/exp_627_tier6_trajectory_feature_reranker/features.py"
    )
    model_hash = sha256(
        ROOT / "experiments/exp_627_tier6_trajectory_feature_reranker/model.py"
    )
    commit = git_output("rev-parse", "HEAD")
    experiment_id = configure_mlflow(config["tracking"]["experiment_name"])
    transition_model = cellular.SharedDilatedConvGRU(
        width=width,
        hidden=config["transition_model"]["hidden"],
        cycles=config["transition_model"]["cycles"],
    )
    transition_model.load_state_dict(load_file(source))
    transition_model.eval()

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
            logits, quotient_logits = transition_model(
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
        return {
            "predictions": state,
            "confidence": {
                "mean_bit_nll": (bit_nll_sum / width).tolist(),
                "negative_trajectory_min_margin": (-minimum).tolist(),
                "negative_mean_step_min_margin": (-step_minimum_sum / width).tolist(),
                "negative_last32_mean_min_margin": (-last_minimum_sum / 32).tolist(),
                "negative_final_min_margin": (-final_minimum).tolist(),
                "mean_quotient_entropy": (quotient_entropy_sum / width).tolist(),
            },
        }

    def generate_dataset(name: str, cases: int, sampler_seed: int):
        key = sha256_text(
            json.dumps(
                {
                    "name": name,
                    "cases": cases,
                    "sampler_seed": sampler_seed,
                    "primes": primes[name],
                    "checkpoint": sha256(source),
                    "runner": runner_hash,
                    "features": feature_hash,
                    "route_fraction": config["data_generation"]["routed_fraction"],
                },
                sort_keys=True,
            )
        )
        cache_path = cache_dir / f"{name}-{key}.pt"
        if cache_path.exists():
            return torch.load(cache_path, weights_only=False), True
        generated_started = time.perf_counter()
        generated_cpu = time.process_time()
        sampler = exp300.BigIntSampler(primes[name], sampler_seed)
        left, right, modulus = sampler.pairs(cases)
        expected = [(a * b) % prime for a, b, prime in zip(left, right, modulus)]
        batch_size = config["data_generation"]["batch_cases"]
        greedy_predictions = []
        confidence = {key: [] for key in feature_module.CONFIDENCE_KEYS}
        greedy_wall = 0.0
        greedy_cpu = 0.0
        for start in range(0, cases, batch_size):
            stop = min(cases, start + batch_size)
            wall = time.perf_counter()
            cpu = time.process_time()
            result = greedy_confidence(
                left[start:stop], right[start:stop], modulus[start:stop]
            )
            greedy_wall += time.perf_counter() - wall
            greedy_cpu += time.process_time() - cpu
            greedy_predictions.extend(result["predictions"])
            for key_name in confidence:
                confidence[key_name].extend(result["confidence"][key_name])
        route_count = round(cases * config["data_generation"]["routed_fraction"])
        route_score = confidence["negative_trajectory_min_margin"]
        routed_indices = sorted(
            range(cases), key=lambda index: (-route_score[index], index)
        )[:route_count]
        routed_left = [left[index] for index in routed_indices]
        routed_right = [right[index] for index in routed_indices]
        routed_modulus = [modulus[index] for index in routed_indices]
        beam_states = []
        beam_scores = []
        beam_predictions = []
        beam_wall = 0.0
        beam_cpu = 0.0
        for start in range(0, route_count, batch_size):
            stop = min(route_count, start + batch_size)
            wall = time.perf_counter()
            cpu = time.process_time()
            result = beam_module.LearnedLikelihoodBeamDecoder(
                transition_model, width, config["data_generation"]["beam_width"]
            ).rollout(
                routed_left[start:stop],
                routed_right[start:stop],
                routed_modulus[start:stop],
                exp300.features,
            )
            beam_wall += time.perf_counter() - wall
            beam_cpu += time.process_time() - cpu
            beam_predictions.extend(result["predictions"])
            beam_states.extend(result["beam_states"])
            beam_scores.extend(result["beam_scores"])
        routed_confidence = {
            key_name: [values[index] for index in routed_indices]
            for key_name, values in confidence.items()
        }
        routed_greedy = [greedy_predictions[index] for index in routed_indices]
        features = feature_module.build_candidate_features(
            routed_left,
            routed_right,
            routed_modulus,
            routed_greedy,
            routed_confidence,
            beam_states,
            beam_scores,
            width,
        )
        targets = []
        for index, states in zip(routed_indices, beam_states):
            targets.append(states.index(expected[index]) if expected[index] in states else -1)
        data = {
            "name": name,
            "cases": cases,
            "route_count": route_count,
            "left": left,
            "right": right,
            "modulus": modulus,
            "expected": expected,
            "greedy_predictions": greedy_predictions,
            "confidence": confidence,
            "routed_indices": routed_indices,
            "beam_predictions": beam_predictions,
            "beam_states": beam_states,
            "beam_scores": beam_scores,
            "features": features,
            "targets": torch.tensor(targets, dtype=torch.long),
            "runtime": {
                "wall_seconds": time.perf_counter() - generated_started,
                "aggregate_cpu_seconds": time.process_time() - generated_cpu,
                "greedy_wall_seconds": greedy_wall,
                "greedy_aggregate_cpu_seconds": greedy_cpu,
                "beam_wall_seconds": beam_wall,
                "beam_aggregate_cpu_seconds": beam_cpu,
            },
        }
        torch.save(data, cache_path)
        return data, False

    def evaluate_reranker(model, data):
        with torch.inference_mode():
            candidate_scores = model(data["features"])
            choices = candidate_scores.argmax(dim=1).tolist()
        routed = set(data["routed_indices"])
        selected = list(data["greedy_predictions"])
        likelihood_selected = list(data["greedy_predictions"])
        oracle_correct = [
            guess == truth
            for guess, truth in zip(data["greedy_predictions"], data["expected"])
        ]
        reranked_states = []
        for route_row, case_index in enumerate(data["routed_indices"]):
            selected[case_index] = data["beam_states"][route_row][choices[route_row]]
            likelihood_selected[case_index] = data["beam_predictions"][route_row]
            reranked_states.append(selected[case_index])
            oracle_correct[case_index] = (
                data["expected"][case_index] in data["beam_states"][route_row]
            )
        correct = sum(
            guess == truth for guess, truth in zip(selected, data["expected"])
        )
        likelihood_correct = sum(
            guess == truth
            for guess, truth in zip(likelihood_selected, data["expected"])
        )
        greedy_correct = sum(
            guess == truth
            for guess, truth in zip(data["greedy_predictions"], data["expected"])
        )
        routed_correct = sum(
            state == data["expected"][case_index]
            for state, case_index in zip(reranked_states, data["routed_indices"])
        )
        invalid = sum(
            state >= data["modulus"][case_index]
            for state, case_index in zip(reranked_states, data["routed_indices"])
        )
        return {
            "cases": data["cases"],
            "correct": correct,
            "exact_accuracy": correct / data["cases"],
            "greedy_correct": greedy_correct,
            "greedy_accuracy": greedy_correct / data["cases"],
            "gain_over_greedy": (correct - greedy_correct) / data["cases"],
            "likelihood_correct": likelihood_correct,
            "likelihood_accuracy": likelihood_correct / data["cases"],
            "oracle_correct": sum(oracle_correct),
            "oracle_accuracy": sum(oracle_correct) / data["cases"],
            "routed_correct": routed_correct,
            "routed_cases": data["route_count"],
            "routed_accuracy": routed_correct / data["route_count"],
            "routed_invalid_rate": invalid / data["route_count"],
            "choices": choices,
            "selected_predictions": selected,
            "candidate_scores": candidate_scores.tolist(),
        }

    generation = {}
    datasets = {}
    for name, cases, offset in (
        ("train", config["datasets"]["train_cases"], 1),
        ("validation", config["datasets"]["validation_cases"], 2),
        ("selection", config["datasets"]["selection_cases"], 3),
    ):
        data, cached = generate_dataset(name, cases, seed + 62800 + offset)
        datasets[name] = data
        generation[name] = {
            "cached": cached,
            "cases": cases,
            "routed_cases": data["route_count"],
            "contained_routed_cases": int(data["targets"].ge(0).sum()),
            "runtime": data["runtime"],
        }

    model = reranker_module.TrajectoryFeatureReranker(
        feature_count=len(feature_module.FEATURE_NAMES),
        hidden=config["reranker"]["hidden"],
    )
    model.set_normalization(datasets["train"]["features"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    eligible = datasets["train"]["targets"].ge(0).nonzero(as_tuple=False).squeeze(1)
    generator = torch.Generator().manual_seed(seed + 71)
    best_state = copy.deepcopy(model.state_dict())
    best_validation = evaluate_reranker(model, datasets["validation"])
    best_epoch = 0
    patience = 0
    history = []
    training_started = time.perf_counter()
    training_cpu = time.process_time()
    for epoch in range(1, config["training"]["epochs"] + 1):
        order = eligible[torch.randperm(len(eligible), generator=generator)]
        model.train()
        losses = []
        for start in range(0, len(order), config["training"]["batch_size"]):
            indices = order[start : start + config["training"]["batch_size"]]
            scores = model(datasets["train"]["features"][indices])
            loss = F.cross_entropy(scores, datasets["train"]["targets"][indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["clip_norm"])
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        validation = evaluate_reranker(model, datasets["validation"])
        history.append(
            {
                "epoch": epoch,
                "loss": sum(losses) / len(losses),
                "validation_exact_accuracy": validation["exact_accuracy"],
                "validation_routed_accuracy": validation["routed_accuracy"],
                "greedy_prior": float(model.greedy_prior.detach()),
            }
        )
        if validation["correct"] > best_validation["correct"]:
            best_validation = validation
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
        if patience >= config["training"]["patience"]:
            break
    model.load_state_dict(best_state)
    model.eval()
    training_runtime = {
        "wall_seconds": time.perf_counter() - training_started,
        "aggregate_cpu_seconds": time.process_time() - training_cpu,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
    }
    checkpoint = output / "reranker.safetensors"
    save_file(model.state_dict(), checkpoint)
    selection = evaluate_reranker(model, datasets["selection"])
    repeated = evaluate_reranker(model, datasets["selection"])
    deterministic = (
        selection["choices"] == repeated["choices"]
        and selection["selected_predictions"] == repeated["selected_predictions"]
        and selection["candidate_scores"] == repeated["candidate_scores"]
    )
    random_model = reranker_module.TrajectoryFeatureReranker(
        feature_count=len(feature_module.FEATURE_NAMES),
        hidden=config["reranker"]["hidden"],
    )
    random_model.feature_mean.copy_(model.feature_mean)
    random_model.feature_scale.copy_(model.feature_scale)
    random_model.eval()
    randomized = evaluate_reranker(random_model, datasets["selection"])
    inference_started = time.perf_counter()
    inference_cpu = time.process_time()
    for _ in range(config["runtime_projection"]["reranker_repeats"]):
        with torch.inference_mode():
            model(datasets["selection"]["features"][:40])
    reranker_wall = (
        time.perf_counter() - inference_started
    ) / config["runtime_projection"]["reranker_repeats"]
    reranker_cpu = (
        time.process_time() - inference_cpu
    ) / config["runtime_projection"]["reranker_repeats"]
    projected_integrated = (
        config["runtime_projection"]["base_integrated_wall_seconds"] + reranker_wall
    )
    gates = config["gates"]
    passed = (
        selection["exact_accuracy"] >= gates["minimum_selected_accuracy"]
        and selection["gain_over_greedy"] >= gates["minimum_gain_over_greedy"]
        and selection["routed_invalid_rate"] <= gates["maximum_invalid_rate"]
        and selection["exact_accuracy"] - randomized["exact_accuracy"]
        >= gates["minimum_randomized_weight_gain"]
        and projected_integrated <= gates["maximum_projected_integrated_seconds"]
        and deterministic
    )
    decision = config["decisions"]["advance" if passed else "reject"]
    selection_evidence = {
        "expected": datasets["selection"]["expected"],
        "greedy_predictions": datasets["selection"]["greedy_predictions"],
        "routed_indices": datasets["selection"]["routed_indices"],
        "beam_states": datasets["selection"]["beam_states"],
        "beam_scores": datasets["selection"]["beam_scores"],
        "reranker_scores": selection["candidate_scores"],
        "reranker_choices": selection["choices"],
        "selected_predictions": selection["selected_predictions"],
    }
    evidence_path = output / "selection-evidence.json"
    evidence_path.write_text(json.dumps(selection_evidence) + "\n")
    with mlflow.start_run(run_name=config["run_id"]) as run:
        mlflow.set_tags(
            {
                "canonical_run_id": config["run_id"],
                "status": "completed",
                "decision": decision,
                "role": "tier6-trajectory-feature-reranker",
                "repository_commit": commit,
                "repository_dirty": str(bool(git_output("status", "--porcelain"))).lower(),
            }
        )
        mlflow.log_params(
            {
                "seed": seed,
                "train_cases": config["datasets"]["train_cases"],
                "validation_cases": config["datasets"]["validation_cases"],
                "selection_cases": config["datasets"]["selection_cases"],
                "reranker_parameters": model.parameter_count(),
                "config_sha256": config_hash,
                "runner_sha256": runner_hash,
                "feature_sha256": feature_hash,
                "model_sha256": model_hash,
                "dataset_sha256": dataset_hash,
                "source_checkpoint_sha256": sha256(source),
            }
        )
        mlflow.log_metrics(
            {
                "selection_accuracy": selection["exact_accuracy"],
                "selection_gain_over_greedy": selection["gain_over_greedy"],
                "selection_oracle_accuracy": selection["oracle_accuracy"],
                "randomized_accuracy": randomized["exact_accuracy"],
                "projected_integrated_seconds": projected_integrated,
                "training_wall_seconds": training_runtime["wall_seconds"],
                "wall_seconds": time.perf_counter() - started,
                "aggregate_cpu_seconds": time.process_time() - cpu_started,
            }
        )
        mlflow.log_artifact(str(config_path), artifact_path="preregistration")
        mlflow.log_artifact(str(ROOT / config["preregistration"]), artifact_path="preregistration")
        mlflow.log_artifact(str(checkpoint), artifact_path="checkpoint")
        mlflow.log_artifact(str(evidence_path), artifact_path="selection-evidence")
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
                "feature_sha256": feature_hash,
                "model_sha256": model_hash,
                "dataset_sha256": dataset_hash,
                "source_checkpoint_sha256": sha256(source),
            },
            "generation": generation,
            "training": {
                "eligible_routed_cases": len(eligible),
                "history": history,
                "runtime": training_runtime,
                "best_validation": {
                    key: value
                    for key, value in best_validation.items()
                    if key not in {"choices", "selected_predictions", "candidate_scores"}
                },
            },
            "selection": {
                key: value
                for key, value in selection.items()
                if key not in {"choices", "selected_predictions", "candidate_scores"}
            },
            "randomized_control": {
                key: value
                for key, value in randomized.items()
                if key not in {"choices", "selected_predictions", "candidate_scores"}
            },
            "determinism": {"passed": deterministic},
            "checkpoint": {
                "path": str(checkpoint.relative_to(ROOT)),
                "sha256": sha256(checkpoint),
                "bytes": checkpoint.stat().st_size,
                "parameters": model.parameter_count(),
            },
            "selection_evidence": {
                "path": str(evidence_path.relative_to(ROOT)),
                "sha256": sha256(evidence_path),
            },
            "runtime_projection": {
                **config["runtime_projection"],
                "reranker_wall_seconds": reranker_wall,
                "reranker_aggregate_cpu_seconds": reranker_cpu,
                "projected_integrated_seconds": projected_integrated,
            },
            "gates": {**gates, "passed": passed},
            "decision": decision,
            "resource_budget": config["resource_budget"],
            "runtime": {
                "wall_seconds": time.perf_counter() - started,
                "aggregate_cpu_seconds": time.process_time() - cpu_started,
                "peak_rss_bytes": process.memory_info().rss,
                "torch_threads": config["training"]["torch_threads"],
            },
            "limitations": [
                "Training, validation, and selection use generated prime-disjoint partitions; checked and private answers were not read.",
                "Selection accuracy is fresh generated evidence, not independent five-seed or official evidence.",
                "Exact bigint arithmetic is confined to generated labels and evaluation scoring.",
                "Projected runtime is not a substitute for the pinned isolated sandbox gate.",
            ],
        }
        summary_path = output / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        mlflow.log_artifact(str(summary_path), artifact_path="canonical-preview")
    print(
        json.dumps(
            {
                "decision": decision,
                "generation": generation,
                "training_runtime": training_runtime,
                "best_validation": summary["training"]["best_validation"],
                "selection": summary["selection"],
                "randomized_control": summary["randomized_control"],
                "determinism": deterministic,
                "runtime_projection": summary["runtime_projection"],
                "runtime": summary["runtime"],
                "mlflow": summary["mlflow"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
