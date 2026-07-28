"""Preregister, calibrate, and execute EXP671's bounded FPR decision route."""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import sys
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import psutil
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.exp_664_sprm_boundary_perturbation.run import (  # noqa: E402
    Timer,
    aggregate_resource,
    bits,
    decode,
    features,
    git_output,
    json_hash,
    load_exp300,
    make_split,
    rollout_cases,
    sha256,
    transition_cases,
    write_json,
)
from experiments.exp_671_fixed_point_damping_halting.model import (  # noqa: E402
    ROUTES,
    TRAINED_ROUTES,
    FixedPointReasonerCell,
)
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402


CONFIG_PATH = ROOT / "configs/exp_671_fixed_point_damping_halting.yaml"


def build_model(config: dict[str, Any], source: Path) -> FixedPointReasonerCell:
    model = FixedPointReasonerCell(**config["model"])
    model.load_source(load_file(source))
    return model


def objective(output, target: list[int], quotient: torch.Tensor, width: int, config: dict[str, Any]):
    truth = bits(target, width)
    deep_bit_losses = [F.binary_cross_entropy_with_logits(logits, truth, reduction="none")
                       for logits in output.deep_logits]
    deep_quotient_losses = [F.cross_entropy(logits, quotient)
                            for logits in output.deep_quotient_logits]
    if not deep_bit_losses:
        raise RuntimeError("TBPTT/deep-supervision trajectory is empty")
    final = deep_bit_losses[-1]
    loss = float(config["training"]["deep_supervision_weight"]) * torch.stack(
        [item.mean() for item in deep_bit_losses]).mean()
    loss = loss + float(config["training"]["worst_bit_weight"]) * final.mean(0).max()
    loss = loss + float(config["training"]["quotient_weight"]) * torch.stack(
        deep_quotient_losses).mean()
    return loss, truth


def train_route(config, exp300, source, splits, root, route, steps, output, mode):
    random.seed(root + 671)
    np.random.seed(root + 671)
    torch.manual_seed(root + 671)
    directory = output / mode / f"root-{root}" / route
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "model.safetensors"
    cache = directory / "training-result.json"
    key = json_hash({"source": sha256(source), "root": root, "route": route,
                     "steps": steps, "model": config["model"],
                     "training": config["training"], "splits": splits})
    if config["resource_policy"]["cache_completed_roots"] and checkpoint.is_file() and cache.is_file():
        previous = json.loads(cache.read_text(encoding="utf-8"))
        if previous.get("cache_key") == key and sha256(checkpoint) == previous["result"]["checkpoint"]["sha256"]:
            model = build_model(config, source)
            model.load_state_dict(load_file(checkpoint))
            return model, {**previous["result"], "cached": True}
    model = build_model(config, source)
    timer = Timer()
    history: list[dict[str, Any]] = []
    examples = calls = 0
    if route in TRAINED_ROUTES:
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=steps,
            eta_min=float(config["training"]["minimum_learning_rate"]),
        )
        samplers = {
            width: exp300.BigIntSampler(
                splits[width]["train"], root + int(config["data"]["training_seed_offset"]) + width)
            for width in config["data"]["training_widths"]
        }
        model.train()
        for step in range(1, steps + 1):
            width = int(config["training"]["schedule"][(step - 1) % len(config["training"]["schedule"])])
            state, multiplier, modulus, input_bit, target, quotient = samplers[width].transitions(
                int(config["training"]["batch_size"]))
            tick = Timer()
            result = model.forward_train(features(state, multiplier, modulus, input_bit, width), route=route)
            loss, truth = objective(result, target, quotient, width, config)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(
                parameters, float(config["training"]["gradient_clip"])))
            optimizer.step()
            scheduler.step()
            examples += len(target)
            calls += len(target) * int(config["model"]["training_iterations"])
            record = {
                "step": step,
                "width": width,
                "loss": float(loss.detach()),
                "gradient_norm": gradient,
                "batch_exact": float((result.logits.ge(0) == truth.bool()).all(1).float().mean()),
                "relative_residual": float(result.relative_residual.mean()),
                "residual_scale_norm": float(model.refinement.residual_scale.detach().norm()),
                "deep_losses": len(result.deep_logits),
                "timing": tick.finish(),
            }
            history.append(record)
            if step == 1 or step % int(config["training"]["log_every_steps"]) == 0 or step == steps:
                print(json.dumps({"route": route, **{k: v for k, v in record.items() if k != "timing"}}), flush=True)
    save_file({name: value.detach().contiguous() for name, value in model.state_dict().items()}, checkpoint)
    reloaded = build_model(config, source)
    reloaded.load_state_dict(load_file(checkpoint))
    exact = all(torch.equal(left, right) for left, right in
                zip(model.state_dict().values(), reloaded.state_dict().values(), strict=True))
    path = str(checkpoint.relative_to(ROOT)) if checkpoint.is_relative_to(ROOT) else str(checkpoint)
    result_record = {
        "route": route,
        "cached": False,
        "parameters": model.parameter_count(),
        "trainable_parameters": model.trainable_parameter_count(),
        "checkpoint": {"path": path, "sha256": sha256(checkpoint), "bytes": checkpoint.stat().st_size},
        "checkpoint_reload_exact": exact,
        "training": {
            "updates": steps if route in TRAINED_ROUTES else 0,
            "examples": examples,
            "operator_calls": calls,
            "history": history,
            "timing": timer.finish(),
        },
    }
    write_json(cache, {"cache_key": key, "result": result_record})
    return model, result_record


def _model_for_route(config, source, route_records, route, intervention_seed):
    checkpoint_route = route if route in TRAINED_ROUTES else (
        "proposal_only" if route == "proposal_only" else "full_fpr")
    model = build_model(config, source)
    checkpoint_path = Path(route_records[checkpoint_route]["checkpoint"]["path"])
    model.load_state_dict(load_file(checkpoint_path if checkpoint_path.is_absolute() else ROOT / checkpoint_path))
    if route == "randomized":
        model.randomized_intervention(intervention_seed)
    elif route == "reset":
        model.reset_intervention()
    model.eval()
    return model


@torch.inference_mode()
def evaluate_transitions(model, route, cases, threshold, minimum_iterations):
    timer = Timer()
    width = int(cases["width"])
    correct: list[bool] = []
    predictions: list[int] = []
    residuals: list[float] = []
    iterations: list[int] = []
    hard_max: list[bool] = []
    bit_correct = invalid = 0
    for start in range(0, len(cases["target"]), 128):
        stop = start + 128
        result = model(
            features(cases["state"][start:stop], cases["multiplicand"][start:stop],
                     cases["modulus"][start:stop], cases["input_bit"][start:stop], width),
            route=route, threshold=threshold, minimum_iterations=minimum_iterations,
        )
        predicted = result.logits.ge(0)
        truth = bits(cases["target"][start:stop], width).bool()
        values = decode(predicted)
        batch_correct = (predicted == truth).all(1).tolist()
        correct.extend(batch_correct)
        predictions.extend(values)
        residuals.extend(result.relative_residual.tolist())
        iterations.extend(result.iterations.tolist())
        hard_max.extend(result.hard_max.tolist())
        bit_correct += int((predicted == truth).sum())
        invalid += sum(value >= prime for value, prime in
                       zip(values, cases["modulus"][start:stop], strict=True))
    count = len(correct)
    return {
        "cases": count,
        "correct": sum(correct),
        "exact_accuracy": sum(correct) / count,
        "bit_accuracy": bit_correct / (count * width),
        "invalid_residue_rate": invalid / count,
        "mean_relative_residual": float(np.mean(residuals)),
        "mean_iterations": float(np.mean(iterations)),
        "early_exit_rate": float(np.mean(np.asarray(iterations) < model.hard_max_iterations)),
        "hard_max_rate": float(np.mean(hard_max)),
        "prediction_sha256": json_hash(predictions),
        "records": {"correct": correct, "residual": residuals, "iterations": iterations},
        "timing": timer.finish(),
    }


@torch.inference_mode()
def evaluate_rollouts(model, route, cases, threshold, minimum_iterations):
    timer = Timer()
    width = int(cases["width"])
    count = len(cases["left"])
    endpoint = prefix = transition_exact = invalid = 0
    final_predictions: list[int] = []
    endpoint_correct: list[bool] = []
    endpoint_residual: list[float] = []
    all_iterations: list[int] = []
    all_hard_max: list[bool] = []
    for start in range(0, count, 32):
        stop = min(count, start + 32)
        left = cases["left"][start:stop]
        right = cases["right"][start:stop]
        modulus = cases["modulus"][start:stop]
        size = len(left)
        state = [0] * size
        teacher = [0] * size
        all_ok = [True] * size
        any_invalid = [False] * size
        last_residual = [0.0] * size
        for position in range(width):
            input_bit = [(value >> (width - 1 - position)) & 1 for value in left]
            result = model(features(state, right, modulus, input_bit, width), route=route,
                           threshold=threshold, minimum_iterations=minimum_iterations)
            state = decode(result.logits.ge(0))
            last_residual = result.relative_residual.tolist()
            all_iterations.extend(result.iterations.tolist())
            all_hard_max.extend(result.hard_max.tolist())
            teacher = [(2 * value + bit * other) % prime for value, bit, other, prime in
                       zip(teacher, input_bit, right, modulus, strict=True)]
            step_ok = [actual == expected for actual, expected in zip(state, teacher, strict=True)]
            transition_exact += sum(step_ok)
            for index, ok in enumerate(step_ok):
                all_ok[index] = all_ok[index] and ok
                any_invalid[index] = any_invalid[index] or state[index] >= modulus[index]
        expected = [(left_value * right_value) % prime for left_value, right_value, prime in
                    zip(left, right, modulus, strict=True)]
        current_endpoint = [actual == truth for actual, truth in zip(state, expected, strict=True)]
        endpoint += sum(current_endpoint)
        prefix += sum(all_ok)
        invalid += sum(any_invalid)
        endpoint_correct.extend(current_endpoint)
        endpoint_residual.extend(last_residual)
        final_predictions.extend(state)
    return {
        "cases": count,
        "correct": endpoint,
        "exact_accuracy": endpoint / count,
        "all_prefix_exact_accuracy": prefix / count,
        "transition_exact_accuracy": transition_exact / (count * width),
        "any_invalid_residue_rate": invalid / count,
        "mean_relative_residual": float(np.mean(endpoint_residual)),
        "mean_iterations": float(np.mean(all_iterations)),
        "early_exit_rate": float(np.mean(np.asarray(all_iterations) < model.hard_max_iterations)),
        "hard_max_rate": float(np.mean(all_hard_max)),
        "prediction_sha256": json_hash(final_predictions),
        "records": {"correct": endpoint_correct, "residual": endpoint_residual},
        "timing": timer.finish(),
    }


def evaluate_route(model, route, cases, threshold, minimum_iterations):
    return {
        "transition": evaluate_transitions(model, route, cases["transition"], threshold, minimum_iterations),
        "rollout": evaluate_rollouts(model, route, cases["rollout"], threshold, minimum_iterations),
    }


def calibrate_threshold(config, model, cases):
    minimum = int(config["threshold_calibration"]["minimum_iterations"])
    candidates = []
    for threshold in config["threshold_calibration"]["candidates"]:
        result = evaluate_rollouts(model, "full_fpr", cases, float(threshold), minimum)
        candidates.append({"threshold": float(threshold), "exact_accuracy": result["exact_accuracy"],
                           "mean_iterations": result["mean_iterations"], "result": result})
    best_accuracy = max(item["exact_accuracy"] for item in candidates)
    eligible = [item for item in candidates if item["exact_accuracy"] >= best_accuracy - 0.02]
    selected = min(eligible, key=lambda item: (item["mean_iterations"], -item["threshold"]))
    labels = np.asarray(selected["result"]["records"]["correct"], dtype=np.float64)
    residuals = np.asarray(selected["result"]["records"]["residual"], dtype=np.float64)
    low = residuals < selected["threshold"]
    smoothing = float(config["threshold_calibration"]["probability_smoothing"])
    def probability(mask):
        return float((labels[mask].sum() + smoothing) / (mask.sum() + 2.0 * smoothing))
    probabilities = {
        "below_threshold": probability(low),
        "at_or_above_threshold": probability(~low),
        "constant": float((labels.sum() + smoothing) / (len(labels) + 2.0 * smoothing)),
    }
    return {"selected_threshold": selected["threshold"], "minimum_iterations": minimum,
            "probabilities": probabilities, "candidates": [
                {k: v for k, v in item.items() if k != "result"} for item in candidates],
            "selection_labels": "root-20261234-validation-only"}


def reliability(records, calibration):
    labels = np.asarray(records["correct"], dtype=np.float64)
    residuals = np.asarray(records["residual"], dtype=np.float64)
    threshold = float(calibration["selected_threshold"])
    probabilities = calibration["probabilities"]
    prediction = np.where(residuals < threshold,
                          float(probabilities["below_threshold"]),
                          float(probabilities["at_or_above_threshold"]))
    constant = np.full_like(labels, float(probabilities["constant"]))
    return {"brier_residual": float(np.mean((prediction - labels) ** 2)),
            "brier_constant": float(np.mean((constant - labels) ** 2)),
            "better_than_constant": float(np.mean((prediction - labels) ** 2)) <
                                    float(np.mean((constant - labels) ** 2)),
            "cases": len(labels)}


class Experiment:
    def __init__(self, config_path, output, mode, root=None):
        self.config_path = Path(config_path).resolve()
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.output = Path(output).resolve()
        self.mode = mode
        expected = {"preregister": 20261234, "calibration": 20261234, "formal": 20261235}[mode]
        self.root = expected if root is None else int(root)
        allowed = {expected} if mode != "formal" else {20261235, 20261236, 20261237}
        if self.root not in allowed:
            raise ValueError(f"root {self.root} is not frozen for {mode}")
        self.source = (ROOT / self.config["source_checkpoint"]).resolve()
        if sha256(self.source) != self.config["source_checkpoint_sha256"]:
            raise RuntimeError("source checkpoint hash mismatch")
        if sha256(ROOT / self.config["paper"]["path"]) != self.config["paper"]["sha256"]:
            raise RuntimeError("paper hash mismatch")
        for name, digest in self.config["fixture_hashes"].items():
            fixture = ROOT / ".scratch/exp-658-joint-quotient-carry-transducer/pinned_sources" / name
            if sha256(fixture) != digest:
                raise RuntimeError(f"fixture hash mismatch: {name}")
        self.exp300 = load_exp300()
        self.splits = {width: make_split(self.config, self.exp300, width, self.root)
                       for width in self.config["data"]["widths"]}

    def preregistration(self):
        files = {
            "config": self.config_path,
            "mechanism": ROOT / self.config["mechanism_map"],
            "model": ROOT / "experiments/exp_671_fixed_point_damping_halting/model.py",
            "runner": Path(__file__),
            "tests": ROOT / "tests/test_exp_671_fixed_point_damping_halting.py",
        }
        return {
            "run_id": self.config["run_id"],
            "root": self.root,
            "mode": self.mode,
            "source": {
                "repository_commit": git_output("rev-parse", "HEAD"),
                "repository_branch": git_output("branch", "--show-current"),
                "checkpoint_sha256": sha256(self.source),
                "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60",
                "files": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
                          for name, path in files.items() if path.is_file()},
            },
            "paper": self.config["paper"],
            "fixtures": self.config["fixture_hashes"],
            "roots": self.config["roots"],
            "routes": self.config["routes"],
            "trained_routes": self.config["trained_routes"],
            "threshold_calibration": self.config["threshold_calibration"],
            "splits": {str(width): {name: json_hash(values) for name, values in split.items()}
                       for width, split in self.splits.items()},
            "gates": self.config["gates"],
            "prohibited": ["answer-dependent halt or selection", "validity filter or repair",
                           "programmed arithmetic", "unbounded loops", "test-label threshold fitting",
                           "checked/private/candidate/submission evaluation"],
        }

    def cases(self, transitions, rollouts, widths, split_names=("validation", "test")):
        records = {split: {} for split in split_names}
        for split in records:
            for width in widths:
                offset = int(self.config["data"]["validation_seed_offset"] if split == "validation"
                             else self.config["data"]["test_seed_offset"])
                records[split][str(width)] = {
                    "transition": transition_cases(self.exp300, self.splits[width][split], width,
                                                   self.root + offset + width, transitions),
                    "rollout": rollout_cases(self.exp300, self.splits[width][split], width,
                                             self.root + offset + width + 500, rollouts),
                }
        return records

    def _train(self, steps):
        records = {}
        for route in (*TRAINED_ROUTES, "proposal_only"):
            _, records[route] = train_route(self.config, self.exp300, self.source, self.splits,
                                             self.root, route, steps, self.output, self.mode)
        for route in ROUTES:
            if route not in records:
                records[route] = {"route": route, "derived_from": "full_fpr",
                                  "checkpoint": records["full_fpr"]["checkpoint"],
                                  "checkpoint_reload_exact": records["full_fpr"]["checkpoint_reload_exact"],
                                  "parameters": records["full_fpr"]["parameters"],
                                  "trainable_parameters": records["full_fpr"]["trainable_parameters"],
                                  "training": {"updates": 0, "examples": 0, "operator_calls": 0,
                                               "history": [], "timing": {"wall_seconds": 0.0}}}
        return records

    def _evaluate(self, records, route, cases, threshold, minimum, seed_offset=0):
        model = _model_for_route(self.config, self.source, records, route,
                                 self.root + 67191 + seed_offset)
        return evaluate_route(model, route, cases, threshold, minimum)

    def run_calibration(self, preregistration, tracking):
        cfg = self.config
        started = Timer()
        records = self._train(int(cfg["calibration"]["updates_per_trained_route"]))
        threshold_cases = self.cases(int(cfg["calibration"]["threshold_transition_cases"]),
                                     int(cfg["calibration"]["threshold_rollout_cases"]),
                                     [128], ("validation",))["validation"]["128"]
        full = _model_for_route(cfg, self.source, records, "full_fpr", self.root + 67191)
        threshold = calibrate_threshold(cfg, full, threshold_cases["rollout"])
        cases = self.cases(int(cfg["calibration"]["transition_cases_per_width"]),
                           int(cfg["calibration"]["rollout_cases_per_width"]),
                           cfg["data"]["widths"])
        for route in ROUTES:
            records[route]["validation"] = {}
            records[route]["test"] = {}
            for split in ("validation", "test"):
                for width in cfg["data"]["widths"]:
                    records[route][split][str(width)] = self._evaluate(
                        records, route, cases[split][str(width)], threshold["selected_threshold"],
                        threshold["minimum_iterations"])
        repeat = self._evaluate(records, "full_fpr", cases["test"]["128"],
                                threshold["selected_threshold"], threshold["minimum_iterations"])
        deterministic = repeat["rollout"]["prediction_sha256"] == \
            records["full_fpr"]["test"]["128"]["rollout"]["prediction_sha256"]
        update_walls = [entry["timing"]["wall_seconds"] for route in TRAINED_ROUTES
                        for entry in records[route]["training"]["history"]]
        repeats = int(cfg["calibration"]["repeats"])
        route_walls = []
        for _ in range(repeats):
            sample = self._evaluate(records, "full_fpr", cases["test"]["128"],
                                    threshold["selected_threshold"], threshold["minimum_iterations"])
            route_walls.append(sample["rollout"]["timing"]["wall_seconds"])
        variability = max(route_walls) / max(float(np.mean(route_walls)), 1e-12)
        headroom = min(float(cfg["resource_policy"]["variability_headroom_ceiling"]),
                       max(float(cfg["resource_policy"]["variability_headroom_floor"]), variability))
        training_wall = sum(records[route]["training"]["timing"]["wall_seconds"]
                            for route in TRAINED_ROUTES)
        small_t = int(cfg["calibration"]["transition_cases_per_width"])
        small_r = int(cfg["calibration"]["rollout_cases_per_width"])
        projected_evaluation = sum(
            records[route][split][str(width)][kind]["timing"]["wall_seconds"] *
            (int(cfg["evaluation"]["rollout_cases_per_width"] if kind == "rollout" else
                 cfg["evaluation"]["transition_cases_per_width"]) / (small_r if kind == "rollout" else small_t))
            for route in ROUTES for split in ("validation", "test")
            for width in cfg["data"]["widths"] for kind in ("transition", "rollout"))
        expected = training_wall + projected_evaluation
        primary_wall = records["full_fpr"]["test"]["128"]["rollout"]["timing"]["wall_seconds"]
        proposal_wall = records["proposal_only"]["test"]["128"]["rollout"]["timing"]["wall_seconds"]
        candidate = float(cfg["resource_policy"]["incumbent_candidate_wall_seconds"]) + \
            max(0.0, primary_wall - proposal_wall) * 100.0 / small_r
        measured = started.finish()
        complete = all(all(split in records[route] for split in ("validation", "test")) for route in ROUTES)
        checks = {
            "complete_routes": complete,
            "expected_value": expected <= float(cfg["resource_policy"]["maximum_expected_formal_wall_seconds"]),
            "memory": measured["peak_rss_bytes"] <= int(cfg["resource_policy"]["maximum_peak_rss_bytes"]),
            "candidate_inference": candidate < float(cfg["gates"]["maximum_candidate_inference_wall_seconds"])
                                   and candidate * headroom <= float(cfg["gates"]["maximum_candidate_headroom_wall_seconds"]),
            "determinism": deterministic,
            "reload": all(records[route]["checkpoint_reload_exact"] for route in ROUTES),
            "split_separation": all(set(split["train"]).isdisjoint(split["validation"])
                                    and set(split["train"]).isdisjoint(split["test"])
                                    and set(split["validation"]).isdisjoint(split["test"])
                                    for split in self.splits.values()),
            "target_free_threshold": threshold["selection_labels"] == "root-20261234-validation-only",
        }
        admitted = all(checks.values())
        admission = {
            "admitted": admitted,
            "checks": checks,
            "measured_training_wall_seconds": training_wall,
            "projected_evaluation_wall_seconds": projected_evaluation,
            "expected_wall_seconds": expected,
            "eta_wall_seconds": expected * headroom,
            "soft_guard_wall_seconds": math.ceil(expected * headroom *
                                                  float(cfg["resource_policy"]["soft_guard_multiplier"])),
            "hard_guard_wall_seconds": math.ceil(expected * headroom *
                                                  float(cfg["resource_policy"]["hard_guard_multiplier"])),
            "variability_ratio": variability,
            "headroom": headroom,
            "candidate_inference_wall_seconds": candidate,
            "candidate_inference_headroom_wall_seconds": candidate * headroom,
            "wall_samples_seconds": route_walls,
            "process_tree_cpu_seconds": measured["process_cpu_seconds"],
            "cgroup_cpu_seconds": measured["cgroup_cpu_seconds"],
            "peak_rss_bytes": measured["peak_rss_bytes"],
        }
        decision = "admit-root-20261235" if admitted else "park-resource-compliance-or-feasibility"
        result = {"run_id": cfg["run_id"], "status": "completed", "mode": "calibration",
                  "root": self.root, "decision": decision, "source": preregistration["source"],
                  "threshold_calibration": threshold, "routes": records, "admission": admission,
                  "resource": aggregate_resource(measured, [], 1),
                  "mlflow": {"tracking_uri": tracking_uri(), "parent_run_id": tracking.info.run_id},
                  "limitations": ["Calibration exactness selects only a disjoint-validation residual threshold.",
                                  "Formal test labels remain unopened.",
                                  "Bounded Conv1d SAIR transfer, not a language Transformer reproduction.",
                                  "No checked/private/candidate/submission surface opened."]}
        write_json(self.output / "calibration.json", result)
        return result

    def run_formal(self, preregistration, tracking, calibration):
        cfg = self.config
        started = Timer()
        threshold = calibration["threshold_calibration"]
        records = self._train(int(cfg["training"]["steps_per_trained_route"]))
        cases128 = self.cases(int(cfg["evaluation"]["transition_cases_per_width"]),
                              int(cfg["evaluation"]["rollout_cases_per_width"]), [128], ("test",))["test"]["128"]
        records["full_fpr"]["test"] = {"128": self._evaluate(
            records, "full_fpr", cases128, threshold["selected_threshold"], threshold["minimum_iterations"])}
        primary = records["full_fpr"]["test"]["128"]
        primary_reliability = reliability(primary["rollout"]["records"], threshold)
        basic = {
            "rollout_floor": primary["rollout"]["exact_accuracy"] >= float(cfg["gates"]["minimum_w128_primary_rollout"]),
            "zero_invalids": primary["rollout"]["any_invalid_residue_rate"] <= float(cfg["gates"]["maximum_invalid_rollout_rate"]),
            "early_exits": primary["rollout"]["early_exit_rate"] >= float(cfg["gates"]["minimum_early_exit_rate"]),
            "bounded_hard_max": primary["rollout"]["hard_max_rate"] <= float(cfg["gates"]["maximum_hard_max_rate"]),
            "residual_reliability": primary_reliability["better_than_constant"],
        }
        decision = "reject-w128-primary-basic"
        gates: dict[str, Any] = {"basic": basic, "residual_reliability": primary_reliability}
        if all(basic.values()):
            for route in ROUTES[1:]:
                records[route]["test"] = {"128": self._evaluate(
                    records, route, cases128, threshold["selected_threshold"], threshold["minimum_iterations"],
                    seed_offset=ROUTES.index(route))}
            full_accuracy = primary["rollout"]["exact_accuracy"]
            proposal_accuracy = records["proposal_only"]["test"]["128"]["rollout"]["exact_accuracy"]
            fixed_accuracy = records["stabilization_fixed"]["test"]["128"]["rollout"]["exact_accuracy"]
            reset_accuracy = records["reset"]["test"]["128"]["rollout"]["exact_accuracy"]
            randomized_accuracy = records["randomized"]["test"]["128"]["rollout"]["exact_accuracy"]
            halt_only_accuracy = records["halt_without_damping"]["test"]["128"]["rollout"]["exact_accuracy"]
            component = {
                "retention": full_accuracy >= max(proposal_accuracy, fixed_accuracy) -
                             float(cfg["gates"]["maximum_w128_loss_to_proposal_or_fixed"]),
                "trained_stabilization": full_accuracy - max(reset_accuracy, randomized_accuracy) >=
                                         float(cfg["gates"]["minimum_stabilization_drop"]),
                "damping_effect": abs(full_accuracy - halt_only_accuracy) >=
                                  float(cfg["gates"]["minimum_damping_accuracy_effect"]),
                "halting_effect": primary["rollout"]["early_exit_rate"] >=
                                  float(cfg["gates"]["minimum_early_exit_rate"]),
                "reload": all(records[route]["checkpoint_reload_exact"] for route in ROUTES),
            }
            gates["component"] = component
            if all(component.values()):
                cases256 = self.cases(int(cfg["evaluation"]["transition_cases_per_width"]),
                                      int(cfg["evaluation"]["rollout_cases_per_width"]), [256], ("test",))["test"]["256"]
                for route in ROUTES:
                    records[route]["test"]["256"] = self._evaluate(
                        records, route, cases256, threshold["selected_threshold"],
                        threshold["minimum_iterations"], seed_offset=100 + ROUTES.index(route))
                w256 = records["full_fpr"]["test"]["256"]["rollout"]["exact_accuracy"]
                comparator = max(records[route]["test"]["256"]["rollout"]["exact_accuracy"]
                                 for route in ("proposal_only", "plain_fixed", "damped_fixed"))
                width_gate = {"absolute": w256 >= float(cfg["gates"]["minimum_w256_primary_rollout"]),
                              "control_gain": w256 - comparator >=
                                              float(cfg["gates"]["minimum_w256_gain_over_controls"])}
                gates["w256"] = width_gate
                decision = "advance-confirmations" if all(width_gate.values()) else "reject-w256"
            else:
                decision = "reject-w128-component"
        measured = started.finish()
        result = {"run_id": cfg["run_id"], "status": "completed", "mode": "formal",
                  "root": self.root, "decision": decision, "source": preregistration["source"],
                  "threshold_calibration": threshold, "routes": records, "gates": gates,
                  "resource": aggregate_resource(measured, [], 1),
                  "mlflow": {"tracking_uri": tracking_uri(), "parent_run_id": tracking.info.run_id},
                  "limitations": ["Formal order stops when a frozen conjunct becomes impossible.",
                                  "Exact arithmetic is confined to offline labels and evaluation.",
                                  "No checked/private/candidate/submission surface opened."]}
        write_json(self.output / f"root-{self.root}-summary.json", result)
        return result

    def run(self):
        self.output.mkdir(parents=True, exist_ok=True)
        preregistration = self.preregistration()
        write_json(self.output / f"preregistration-root-{self.root}.json", preregistration)
        if self.mode == "preregister":
            return preregistration
        torch.set_num_threads(int(self.config["training"]["torch_threads"]))
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        configure_mlflow(self.config["tracking"]["experiment_name"])
        with mlflow.start_run(run_name=f"{self.config['run_id']}-{self.mode}-{self.root}") as tracking:
            if self.mode == "calibration":
                result = self.run_calibration(preregistration, tracking)
            else:
                calibration_path = self.output / "calibration.json"
                if not calibration_path.is_file():
                    raise RuntimeError("formal root requires calibration receipt")
                calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
                if not calibration["admission"]["admitted"]:
                    raise RuntimeError("formal root blocked by resource admission")
                if self.root > 20261235:
                    previous = self.output / f"root-{self.root - 1}-summary.json"
                    if not previous.is_file() or json.loads(previous.read_text(encoding="utf-8"))["decision"] != "advance-confirmations":
                        raise RuntimeError("confirmation root blocked by prior conjunctive gate")
                result = self.run_formal(preregistration, tracking, calibration)
            mlflow.set_tag("decision", result["decision"])
            return result


def environment_record():
    return {"python": platform.python_version(), "torch": torch.__version__,
            "platform": platform.platform(), "logical_cpu_count": psutil.cpu_count(),
            "torch_threads": torch.get_num_threads()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("preregister", "calibration", "formal"), required=True)
    parser.add_argument("--root", type=int)
    arguments = parser.parse_args()
    result = Experiment(arguments.config, arguments.output, arguments.mode, arguments.root).run()
    print(json.dumps({"decision": result.get("decision"), "root": result.get("root"),
                      "status": result.get("status", "preregistered")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
