"""Freeze, calibrate, and execute the EXP684 STARS × random-depth factorial."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.exp_220_width_independent_cellular.model import SharedDilatedConvGRU  # noqa: E402
from experiments.exp_664_sprm_boundary_perturbation.run import (  # noqa: E402
    Timer, aggregate_resource, bits, features, git_output, json_hash, load_exp300,
    make_split, rollout_cases, sha256, transition_cases, write_json,
)
from experiments.exp_667_rsm_shallow_train_deep_infer.run import (  # noqa: E402
    evaluate_rollouts, evaluate_transitions,
)
from experiments.exp_684_stars_random_depth_factorial.model import (  # noqa: E402
    JVP_ROUTES, ROUTES, TRAINED_ROUTES, FactorialCell, frozen_count_schedules,
)
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402


CONFIG_PATH = ROOT / "configs/exp_684_stars_random_depth_factorial.yaml"
OUTPUT_PATH = ROOT / ".scratch/exp-684-stars-random-depth-factorial"
PACKAGE_FILES = {
    "config": "configs/exp_684_stars_random_depth_factorial.yaml",
    "model": "experiments/exp_684_stars_random_depth_factorial/model.py",
    "runner": "experiments/exp_684_stars_random_depth_factorial/run.py",
    "test": "tests/test_exp_684_stars_random_depth_factorial.py",
    "preregistration": "knowledge/autoresearch/ablations/exp-684-stars-random-depth-factorial.md",
    "mechanism": "knowledge/autoresearch/research/2026-07-30-exp684-stars-random-depth-factorial.md",
}


def build_model(config: dict[str, Any], source: Path) -> FactorialCell:
    model = FactorialCell(**config["model"])
    model.load_source(load_file(source))
    return model


class InferenceView:
    """Every route uses the inherited fixed two-cycle Candidate005 forward graph."""

    def __init__(self, model: FactorialCell):
        self.model = model

    def __call__(self, batch: torch.Tensor, **unused):
        logits, quotient = self.model(batch)
        zero = torch.zeros(batch.shape[0], device=batch.device)
        calls = len(self.model.dilation_schedule(batch.shape[-1]))
        return SimpleNamespace(logits=logits, quotient_logits=quotient,
                               relative_residual=zero,
                               iterations=torch.full_like(zero, calls, dtype=torch.long))


def objective(logits, quotient_logits, target, quotient, width, config):
    truth = bits(target, width)
    per_bit = F.binary_cross_entropy_with_logits(logits, truth, reduction="none")
    loss = (per_bit.mean()
            + float(config["training"]["worst_bit_weight"]) * per_bit.mean(0).max()
            + float(config["training"]["quotient_weight"])
            * F.cross_entropy(quotient_logits, quotient))
    return loss, truth


def train_route(config, exp300, source, splits, root, route, steps, output, mode):
    random.seed(root + 684)
    np.random.seed(root + 684)
    torch.manual_seed(root + 684)
    directory = output / mode / f"root-{root}" / route
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "model.safetensors"
    cache = directory / "training-result.json"
    counts = frozen_count_schedules(config, steps)[route]
    key = json_hash({"source": sha256(source), "root": root, "route": route,
                     "steps": steps, "training": config["training"],
                     "regularization": config["regularization"], "counts": counts,
                     "splits": splits})
    if (config["resource_policy"]["cache_completed_roots"] and checkpoint.is_file()
            and cache.is_file()):
        previous = json.loads(cache.read_text())
        result = previous.get("result", {})
        if previous.get("cache_key") == key and sha256(checkpoint) == result.get("checkpoint", {}).get("sha256"):
            model = build_model(config, source)
            model.load_state_dict(load_file(checkpoint))
            return model, {**result, "cached": True}
    model = build_model(config, source)
    timer = Timer()
    history = []
    examples = recurrent_visits = jvp_calls = 0
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(parameters,
                                  lr=float(config["training"]["learning_rate"]),
                                  weight_decay=float(config["training"]["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=float(config["training"]["minimum_learning_rate"]))
    samplers = {width: exp300.BigIntSampler(
        splits[width]["train"], root + int(config["data"]["training_seed_offset"]) + width)
        for width in config["data"]["training_widths"]}
    regularization_weight = float(config["regularization"]["lambda"])
    model.train()
    for step, cycles in enumerate(counts, 1):
        width = int(config["training"]["schedule"][(step - 1) % 4])
        state, multiplier, modulus, input_bit, target, quotient = samplers[width].transitions(
            int(config["training"]["batch_size"]))
        batch = features(state, multiplier, modulus, input_bit, width)
        tick = Timer()
        prediction = model.forward_train(batch, cycles=cycles)
        base_loss, truth = objective(prediction.logits, prediction.quotient_logits,
                                     target, quotient, width, config)
        proxy = base_loss.new_zeros(())
        first_jvp_norm = vector_norm = state_norm = 0.0
        gradient_enabled = False
        if route in JVP_ROUTES:
            record = model.jvp_proxy(
                prediction.boundary, batch, prediction.final_dilation,
                seed=root + step + 68400, stop_gradient=route == "random_stop_gradient")
            proxy = record.proxy
            first_jvp_norm = float(record.first_jvp_norm)
            vector_norm = float(record.vector_norm)
            state_norm = float(record.state_norm)
            gradient_enabled = record.parameter_gradient_enabled
            loss = (1.0 - regularization_weight) * base_loss + regularization_weight * proxy
            jvp_calls += record.jvp_calls * len(target)
        else:
            loss = base_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(
            parameters, float(config["training"]["gradient_clip"])))
        optimizer.step()
        scheduler.step()
        examples += len(target)
        recurrent_visits += prediction.recurrent_visits * len(target)
        history.append({"step": step, "width": width, "cycles": cycles,
                        "recurrent_visits_per_example": prediction.recurrent_visits,
                        "loss": float(loss.detach()), "base_loss": float(base_loss.detach()),
                        "proxy": float(proxy.detach()), "gradient_norm": gradient,
                        "batch_exact": float((prediction.logits.ge(0) == truth.bool()).all(1).float().mean()),
                        "first_jvp_norm": first_jvp_norm, "vector_norm": vector_norm,
                        "state_norm": state_norm,
                        "proxy_parameter_gradient_enabled": gradient_enabled,
                        "timing": tick.finish()})
    save_file({name: value.detach().contiguous() for name, value in model.state_dict().items()}, checkpoint)
    reloaded = build_model(config, source)
    reloaded.load_state_dict(load_file(checkpoint))
    reload_exact = all(torch.equal(left, right) for left, right in
                       zip(model.state_dict().values(), reloaded.state_dict().values(), strict=True))
    checkpoint_path = (str(checkpoint.relative_to(ROOT))
                       if checkpoint.is_relative_to(ROOT) else str(checkpoint))
    histogram = {str(value): counts.count(value) for value in range(1, 5)}
    result = {
        "route": route,
        "cached": False,
        "parameters": model.parameter_count(),
        "checkpoint": {"path": checkpoint_path, "sha256": sha256(checkpoint),
                       "bytes": checkpoint.stat().st_size},
        "checkpoint_reload_exact": reload_exact,
        "inference_recurrent_visits_w128": len(model.dilation_schedule(128)),
        "training": {"updates": steps, "examples": examples,
                     "recurrent_visits": recurrent_visits, "jvp_calls": jvp_calls,
                     "jvp_gradient_compute_matched": False,
                     "count_sum": sum(counts), "count_histogram": histogram,
                     "counts_sha256": json_hash(counts), "history": history,
                     "resource": timer.finish()},
    }
    write_json(cache, {"cache_key": key, "result": result})
    return model, result


def make_cases(config, exp300, splits, root, widths, transitions, rollouts):
    records = {"validation": {}, "test": {}}
    for split in records:
        offset = int(config["data"]["validation_seed_offset"] if split == "validation"
                     else config["data"]["test_seed_offset"])
        for width in widths:
            records[split][str(width)] = {
                "transition": transition_cases(exp300, splits[width][split], width,
                                               root + offset + width, transitions),
                "rollout": rollout_cases(exp300, splits[width][split], width,
                                         root + offset + width + 500, rollouts),
            }
    return records


def evaluate(model, route, cases):
    view = InferenceView(model)
    return {"transition": evaluate_transitions(view, route, cases["transition"]),
            "rollout": evaluate_rollouts(view, route, cases["rollout"])}


def stability_diagnostic(model, cases, seed, epsilon):
    width = int(cases["width"])
    batch = features(cases["state"], cases["multiplicand"], cases["modulus"],
                     cases["input_bit"], width)
    prediction = model.forward_train(batch, cycles=2)
    record = model.jvp_proxy(prediction.boundary, batch, prediction.final_dilation,
                             seed=seed)
    generator = torch.Generator().manual_seed(seed + 1)
    noise = torch.randn(prediction.boundary.shape, generator=generator,
                        dtype=prediction.boundary.dtype)
    noise = noise / noise.flatten(1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-12) * epsilon
    first = model.recurrent_update(prediction.boundary, batch, prediction.final_dilation)
    second = model.recurrent_update(prediction.boundary + noise, batch,
                                    prediction.final_dilation)
    return {"one_step_jvp_squared_norm_proxy": float(record.proxy),
            "first_jvp_norm": float(record.first_jvp_norm),
            "vector_norm": float(record.vector_norm),
            "perturbation_amplification": float(
                (second - first).flatten(1).norm(dim=1).mean() / epsilon),
            "state_norm": float(record.state_norm), "jvp_calls": record.jvp_calls,
            "exact_spectral_radius": False}


class Experiment:
    def __init__(self, config_path=CONFIG_PATH, output=OUTPUT_PATH, mode="preflight", root=None):
        self.config_path = Path(config_path).resolve()
        self.config = yaml.safe_load(self.config_path.read_text())
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        expected = {"freeze": 20261286, "preflight": 20261286,
                    "calibration": 20261286, "formal": 20261287}[mode]
        self.root = expected if root is None else int(root)
        allowed = {expected} if mode != "formal" else {20261287, 20261288, 20261289}
        if self.root not in allowed:
            raise ValueError("unfrozen root")
        self.source = (ROOT / self.config["source"]["checkpoint"]).resolve()
        if sha256(self.source) != self.config["source"]["checkpoint_sha256"]:
            raise RuntimeError("checkpoint identity mismatch")
        for paper in ("stars_paper", "count_paper"):
            record = self.config["source"][paper]
            if sha256(ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"{paper} identity mismatch")
        for name, digest in self.config["source"]["fixtures"].items():
            path = ROOT / ".scratch/exp-658-joint-quotient-carry-transducer/pinned_sources" / name
            if sha256(path) != digest:
                raise RuntimeError("fixture identity mismatch")
        self.exp300 = load_exp300()
        self.splits = {width: make_split(self.config, self.exp300, width, self.root)
                       for width in self.config["data"]["widths"]}

    def identity(self):
        return {"repository_commit": git_output("rev-parse", "HEAD"),
                "repository_branch": git_output("branch", "--show-current"),
                "challenge_commit": git_output("-C", str(ROOT / "sources/github/modular-arithmetic-challenge"), "rev-parse", "HEAD"),
                "checkpoint_sha256": sha256(self.source),
                **{f"{name}_sha256": sha256(ROOT / path)
                   for name, path in PACKAGE_FILES.items()}}

    def protocol(self):
        schedules = frozen_count_schedules(self.config, 400)
        levels = [6, 7, 8, 7]
        work = {route: sum(count * levels[index % 4]
                           for index, count in enumerate(values))
                for route, values in schedules.items()}
        return {"schema_version": "exp684-frozen-v1", "source": self.identity(),
                "routes": self.config["routes"], "factorial": self.config["factorial"],
                "count_schedule": self.config["count_schedule"],
                "schedule_hashes": {route: json_hash(values)
                                    for route, values in schedules.items()},
                "recurrent_work_per_example": work, "regularization": self.config["regularization"],
                "inference": self.config["inference"], "training": self.config["training"],
                "data": self.config["data"], "roots": self.config["roots"],
                "gates": self.config["gates"], "resource_policy": self.config["resource_policy"],
                "splits": {str(width): {name: json_hash(values) for name, values in split.items()}
                           for width, split in self.splits.items()}}

    def freeze(self):
        protocol = self.protocol()
        write_json(self.output / "frozen-protocol.json", protocol)
        return {"status": "frozen", "sha256": sha256(self.output / "frozen-protocol.json")}

    def assert_frozen(self):
        path = self.output / "frozen-protocol.json"
        if not path.is_file() or json.loads(path.read_text()) != json.loads(json.dumps(self.protocol())):
            raise RuntimeError("package differs from frozen protocol")

    def preflight(self):
        self.assert_frozen()
        schedules = frozen_count_schedules(self.config, 400)
        source = SharedDilatedConvGRU(**self.config["model"])
        source.load_state_dict(load_file(self.source))
        model = build_model(self.config, self.source)
        batch = torch.randn(2, 6, 64)
        fixed = model.forward_train(batch, cycles=2)
        ordinary = model(batch)
        checks = {
            "routes": tuple(self.config["routes"]) == ROUTES == TRAINED_ROUTES,
            "roots": self.config["roots"] == {"calibration": 20261286,
                                               "viability": 20261287,
                                               "confirmations": [20261288, 20261289]},
            "schedule_hashes": (json_hash(schedules["random_stars"]) == self.config["count_schedule"]["random_sha256"]
                                and json_hash(schedules["shuffled_random_stars"]) == self.config["count_schedule"]["shuffled_sha256"]),
            "equal_work": len({record for record in self.protocol()["recurrent_work_per_example"].values()}) == 1,
            "exact_fixed_training": torch.equal(fixed.logits, ordinary[0]) and torch.equal(fixed.quotient_logits, ordinary[1]),
            "inherited_forward": FactorialCell.forward is SharedDilatedConvGRU.forward,
            "inference_identity": torch.equal(model(batch)[0], source(batch)[0])
                                  and torch.equal(model(batch)[1], source(batch)[1]),
            "parameters": model.parameter_count() == source.parameter_count()
                          and set(model.state_dict()) == set(source.state_dict()),
            "split_separation": all(set(split["train"]).isdisjoint(split["validation"])
                                    and set(split["train"]).isdisjoint(split["test"])
                                    and set(split["validation"]).isdisjoint(split["test"])
                                    for split in self.splits.values()),
            "schema": all(key in self.config for key in
                          ("run_id", "source", "routes", "factorial", "count_schedule",
                           "regularization", "inference", "roots", "gates")),
        }
        if not all(checks.values()):
            raise RuntimeError(checks)
        result = {"schema_version": "exp684-preflight-v1", "status": "passed",
                  "checks": checks, "completion_marker": "preflight-complete"}
        write_json(self.output / "preflight.json", result)
        return result

    def _load_routes(self, records):
        models = {}
        for route, record in records.items():
            model = build_model(self.config, self.source)
            model.load_state_dict(load_file(ROOT / record["checkpoint"]["path"]))
            model.eval()
            models[route] = model
        return models

    def calibration(self):
        self.assert_frozen()
        self.preflight()
        config = self.config
        torch.set_num_threads(int(config["training"]["torch_threads"]))
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        started = Timer()
        configure_mlflow(config["tracking"]["experiment_name"])
        routes = {}
        steps = int(config["calibration"]["representative_updates_per_trained_route"])
        transitions = int(config["calibration"]["transition_cases_per_width"])
        rollouts = int(config["calibration"]["rollout_cases_per_width"])
        widths = config["data"]["widths"]
        with mlflow.start_run(run_name=f"{config['run_id']}-calibration-{self.root}") as tracking:
            for route in ROUTES:
                _, routes[route] = train_route(config, self.exp300, self.source, self.splits,
                                               self.root, route, steps, self.output, "calibration")
            cases = make_cases(config, self.exp300, self.splits, self.root, widths,
                               transitions, rollouts)
            models = self._load_routes(routes)
            source_model = build_model(config, self.source).eval()
            for route, model in models.items():
                routes[route]["validation"] = {}
                routes[route]["test"] = {}
                for split in ("validation", "test"):
                    for width in widths:
                        routes[route][split][str(width)] = evaluate(
                            model, route, cases[split][str(width)])
            source_eval = {split: {str(width): evaluate(source_model, "source",
                                      cases[split][str(width)]) for width in widths}
                           for split in ("validation", "test")}
            diagnostics = {str(width): stability_diagnostic(
                models["random_stars"], cases["validation"][str(width)]["transition"],
                self.root + 68400 + width,
                float(config["calibration"]["perturbation_epsilon"])) for width in widths}
            repeat = evaluate(models["random_stars"], "random_stars",
                              cases["test"]["128"])
            deterministic = (repeat["rollout"]["prediction_sha256"] ==
                             routes["random_stars"]["test"]["128"]["rollout"]["prediction_sha256"])
            update_walls = [entry["timing"]["wall_seconds"] for route in ROUTES
                            for entry in routes[route]["training"]["history"]]
            projected_training = sum(update_walls) * int(config["training"]["steps_per_trained_route"]) / steps
            variability = max(update_walls) / max(float(np.mean(update_walls)), 1e-12)
            headroom = min(float(config["resource_policy"]["variability_headroom_ceiling"]),
                           max(float(config["resource_policy"]["variability_headroom_floor"]), variability))
            projected_evaluation = sum(
                routes[route][split][str(width)][kind]["timing"]["wall_seconds"]
                * (int(config["evaluation"]["rollout_cases_per_width"] if kind == "rollout"
                       else config["evaluation"]["transition_cases_per_width"])
                   / (rollouts if kind == "rollout" else transitions))
                for route in ROUTES for split in ("validation", "test") for width in widths
                for kind in ("transition", "rollout"))
            expected = projected_training + projected_evaluation
            candidate = float(config["resource_policy"]["incumbent_candidate_wall_seconds"])
            resource = aggregate_resource(started.finish(), [], 1)
            full_schedules = frozen_count_schedules(config, 400)
            levels = [6, 7, 8, 7]
            full_work = {route: sum(count * levels[index % 4]
                                    for index, count in enumerate(values))
                         for route, values in full_schedules.items()}
            source = SharedDilatedConvGRU(**config["model"])
            source.load_state_dict(load_file(self.source))
            probe = torch.randn(2, 6, 64)
            identity = (FactorialCell.forward is SharedDilatedConvGRU.forward
                        and source.parameter_count() == models["random_stars"].parameter_count()
                        and len(source.dilation_schedule(128)) == routes["random_stars"]["inference_recurrent_visits_w128"]
                        and torch.equal(build_model(config, self.source)(probe)[0], source(probe)[0]))
            checks = {
                "complete_routes": set(routes) == set(ROUTES),
                "expected_value": expected <= float(config["resource_policy"]["maximum_expected_formal_wall_seconds"]),
                "memory": resource["peak_rss_bytes"] <= int(config["resource_policy"]["maximum_peak_rss_bytes"]),
                "candidate_inference": candidate < float(config["gates"]["maximum_candidate_inference_wall_seconds"])
                                       and candidate * headroom <= float(config["gates"]["maximum_candidate_headroom_wall_seconds"]),
                "determinism": deterministic,
                "reload": all(record["checkpoint_reload_exact"] for record in routes.values()),
                "equal_full_work": len(set(full_work.values())) == 1 == (5600 in set(full_work.values())),
                "inference_identity": identity,
                "finite_jvp": all(math.isfinite(value[key]) for value in diagnostics.values()
                                  for key in ("one_step_jvp_squared_norm_proxy", "perturbation_amplification")),
                "split_separation": all(set(split["train"]).isdisjoint(split["validation"])
                                        and set(split["train"]).isdisjoint(split["test"])
                                        and set(split["validation"]).isdisjoint(split["test"])
                                        for split in self.splits.values()),
            }
            admitted = all(checks.values())
            admission = {"admitted": admitted, "checks": checks,
                         "projected_training_wall_seconds": projected_training,
                         "projected_evaluation_wall_seconds": projected_evaluation,
                         "expected_wall_seconds": expected,
                         "eta_wall_seconds": expected * headroom,
                         "soft_guard_wall_seconds": math.ceil(expected * headroom * 1.25),
                         "hard_guard_wall_seconds": math.ceil(expected * headroom * 2),
                         "variability_ratio": variability, "headroom": headroom,
                         "candidate_inference_wall_seconds": candidate,
                         "candidate_inference_headroom_wall_seconds": candidate * headroom,
                         "full_recurrent_work_per_example": full_work,
                         "resource": resource}
            decision = "admit-root-20261287" if admitted else "park-root-20261286-resource-admission"
            result = {"run_id": config["run_id"], "status": "completed",
                      "mode": "calibration", "root": self.root, "decision": decision,
                      "source": self.identity(), "routes": routes, "source_evaluation": source_eval,
                      "diagnostics": {"stability": diagnostics,
                                      "deterministic_repeat_equal": deterministic,
                                      "inference_identity": identity},
                      "admission": admission,
                      "mlflow": {"tracking_uri": tracking_uri(),
                                 "parent_run_id": tracking.info.run_id},
                      "limitations": ["Calibration is resource/integrity evidence, not accuracy science.",
                                      "JVP gradient compute is intentionally unmatched to no-JVP controls and disclosed.",
                                      "Formal inference is the inherited fixed Candidate005 graph.",
                                      "No checked/private/candidate/submission surface opened."]}
            write_json(self.output / "calibration.json", result)
            write_json(self.output / "calibration.complete", {"completion_marker": "calibration-complete",
                                                               "decision": decision})
            mlflow.set_tag("decision", decision)
            return result

    def formal(self):
        self.assert_frozen()
        calibration = json.loads((self.output / "calibration.json").read_text())
        if not calibration["admission"]["admitted"]:
            raise RuntimeError("formal blocked by calibration")
        if self.root > 20261287:
            previous = json.loads((self.output / f"root-{self.root - 1}-summary.json").read_text())
            if not previous["gates"].get("all_pass", False):
                raise RuntimeError("confirmation blocked by prior root")
        config = self.config
        torch.set_num_threads(int(config["training"]["torch_threads"]))
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        started = Timer()
        configure_mlflow(config["tracking"]["experiment_name"])
        routes = {}
        steps = int(config["training"]["steps_per_trained_route"])
        transitions = int(config["evaluation"]["transition_cases_per_width"])
        rollouts = int(config["evaluation"]["rollout_cases_per_width"])
        with mlflow.start_run(run_name=f"{config['run_id']}-formal-{self.root}") as tracking:
            for route in ROUTES:
                _, routes[route] = train_route(config, self.exp300, self.source, self.splits,
                                               self.root, route, steps, self.output, "formal")
            models = self._load_routes(routes)
            source_model = build_model(config, self.source).eval()
            cases = make_cases(config, self.exp300, self.splits, self.root, [64, 128, 256],
                               transitions, rollouts)
            for route, model in models.items():
                routes[route]["test"] = {}
                for width in (64, 128):
                    routes[route]["test"][str(width)] = evaluate(
                        model, route, cases["test"][str(width)])
            source_eval = {str(width): evaluate(source_model, "source",
                                                 cases["test"][str(width)])
                           for width in (64, 128)}
            primary = routes["random_stars"]["test"]
            joint = primary["128"]["rollout"]
            joint_correct = round(joint["exact_accuracy"] * joint["cases"])
            fixed_stars = routes["fixed_stars"]["test"]["128"]["rollout"]
            random_no_jvp = routes["random_no_jvp"]["test"]["128"]["rollout"]
            fixed_base = routes["fixed_no_jvp"]["test"]["128"]["rollout"]
            shuffled = routes["shuffled_random_stars"]["test"]["128"]["rollout"]
            stop_gradient = routes["random_stop_gradient"]["test"]["128"]["rollout"]
            randomized = copy.deepcopy(models["random_stars"])
            randomized.randomized_intervention(self.root + 68491)
            randomized.eval()
            randomized_eval = evaluate(randomized, "randomized_intervention",
                                       cases["test"]["128"])
            repeat = evaluate(models["random_stars"], "random_stars",
                              cases["test"]["128"])
            factorial_interaction = (joint["exact_accuracy"]
                                     - random_no_jvp["exact_accuracy"]
                                     - fixed_stars["exact_accuracy"]
                                     + fixed_base["exact_accuracy"])
            required_correct = int(config["gates"]["minimum_additional_w128_correct_over_each_component"])
            w128_checks = {
                "joint_rollout": joint["exact_accuracy"] >= float(config["gates"]["minimum_w128_joint_rollout"]),
                "joint_transition": primary["128"]["transition"]["exact_accuracy"] >= float(config["gates"]["minimum_w128_transition"]),
                "zero_invalid": joint["any_invalid_residue_rate"] <= float(config["gates"]["maximum_invalid_rollout_rate"]),
                "w64_retention": primary["64"]["rollout"]["exact_accuracy"] >= source_eval["64"]["rollout"]["exact_accuracy"] - float(config["gates"]["maximum_w64_retention_loss"]),
                "gain_over_fixed_stars": joint_correct >= round(fixed_stars["exact_accuracy"] * fixed_stars["cases"]) + required_correct,
                "gain_over_random_no_jvp": joint_correct >= round(random_no_jvp["exact_accuracy"] * random_no_jvp["cases"]) + required_correct,
                "positive_factorial_interaction": factorial_interaction > float(config["gates"]["minimum_factorial_interaction"]),
                "count_association_causal": joint_correct >= round(shuffled["exact_accuracy"] * shuffled["cases"]) + int(config["gates"]["minimum_causal_correct_gain"]),
                "jvp_gradient_causal": joint_correct >= round(stop_gradient["exact_accuracy"] * stop_gradient["cases"]) + int(config["gates"]["minimum_causal_correct_gain"]),
                "trained_weight_dependence": joint["exact_accuracy"] - randomized_eval["rollout"]["exact_accuracy"] >= float(config["gates"]["minimum_trained_weight_drop"]),
                "reload": all(record["checkpoint_reload_exact"] for record in routes.values()),
                "determinism": repeat["rollout"]["prediction_sha256"] == joint["prediction_sha256"],
                "equal_work": len({record["training"]["recurrent_visits"] for record in routes.values()}) == 1,
            }
            w128_pass = all(w128_checks.values())
            w256_checks = {"authorized": w128_pass}
            if w128_pass:
                for route, model in models.items():
                    routes[route]["test"]["256"] = evaluate(
                        model, route, cases["test"]["256"])
                source_eval["256"] = evaluate(source_model, "source", cases["test"]["256"])
                wide = routes["random_stars"]["test"]["256"]["rollout"]
                w256_checks.update({
                    "joint_rollout": wide["exact_accuracy"] >= float(config["gates"]["minimum_w256_joint_rollout"]),
                    "gain_over_fixed_stars": wide["exact_accuracy"] >= routes["fixed_stars"]["test"]["256"]["rollout"]["exact_accuracy"] + float(config["gates"]["minimum_w256_gain_over_each_component"]),
                    "gain_over_random_no_jvp": wide["exact_accuracy"] >= routes["random_no_jvp"]["test"]["256"]["rollout"]["exact_accuracy"] + float(config["gates"]["minimum_w256_gain_over_each_component"]),
                    "zero_invalid": wide["any_invalid_residue_rate"] == 0.0,
                })
            all_pass = w128_pass and all(w256_checks.values())
            if not w128_pass:
                decision = "reject-w128-factorial-or-causality-gate"
            elif not all(w256_checks.values()):
                decision = "reject-w256-transfer-or-component-gate"
            elif self.root == 20261287:
                decision = "advance-confirmation-root-20261288"
            elif self.root == 20261288:
                decision = "advance-confirmation-root-20261289"
            else:
                decision = "advance-intermediate-exp684-result"
            resource = aggregate_resource(started.finish(), [], 1)
            result = {"run_id": config["run_id"], "status": "completed", "mode": "formal",
                      "root": self.root, "decision": decision, "source": self.identity(),
                      "routes": routes, "source_evaluation": source_eval,
                      "interventions": {"randomized": randomized_eval,
                                        "factorial_interaction": factorial_interaction},
                      "gates": {"w128": w128_checks, "w128_pass": w128_pass,
                                "w256": w256_checks, "all_pass": all_pass},
                      "resource": resource,
                      "mlflow": {"tracking_uri": tracking_uri(),
                                 "parent_run_id": tracking.info.run_id},
                      "limitations": ["STARS uses a one-step JVP proxy, not exact spectral radius.",
                                      "JVP gradient compute is intentionally unmatched and disclosed.",
                                      "Formal inference is the inherited fixed Candidate005 graph.",
                                      "No checked/private/candidate/submission surface opened."]}
            write_json(self.output / f"root-{self.root}-summary.json", result)
            write_json(self.output / f"root-{self.root}.complete",
                       {"completion_marker": "formal-root-complete", "decision": decision})
            mlflow.set_tag("decision", decision)
            return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--mode", choices=("freeze", "preflight", "calibration", "formal"),
                        required=True)
    parser.add_argument("--root", type=int)
    arguments = parser.parse_args()
    experiment = Experiment(arguments.config, arguments.output, arguments.mode, arguments.root)
    if arguments.mode == "freeze":
        result = experiment.freeze()
    elif arguments.mode == "preflight":
        result = experiment.preflight()
    elif arguments.mode == "calibration":
        result = experiment.calibration()
    else:
        result = experiment.formal()
    print(json.dumps({"status": result.get("status"), "decision": result.get("decision")},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
