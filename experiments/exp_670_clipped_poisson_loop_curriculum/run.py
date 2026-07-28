"""Execute EXP670 calibration and gated clipped-Poisson curriculum science."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.exp_664_sprm_boundary_perturbation.run import (  # noqa: E402
    Timer, aggregate_resource, bits, features, git_output, json_hash, load_exp300,
    make_split, rollout_cases, sha256, transition_cases, write_json,
)
from experiments.exp_667_rsm_shallow_train_deep_infer.run import (  # noqa: E402
    evaluate_rollouts, evaluate_transitions,
)
from experiments.exp_670_clipped_poisson_loop_curriculum.model import (  # noqa: E402
    ROUTES, TRAINED_ROUTES, LoopCurriculumCell, frozen_count_schedules,
)
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402

CONFIG_PATH = ROOT / "configs/exp_670_clipped_poisson_loop_curriculum.yaml"


def build_model(config: dict[str, Any], source: Path) -> LoopCurriculumCell:
    model = LoopCurriculumCell(**config["model"])
    model.load_source(load_file(source))
    return model


def objective(output, target, quotient, width, config):
    truth = bits(target, width)
    per_bit = F.binary_cross_entropy_with_logits(output.logits, truth, reduction="none")
    loss = (per_bit.mean()
            + float(config["training"]["worst_bit_weight"]) * per_bit.mean(0).max()
            + float(config["training"]["quotient_weight"]) *
            F.cross_entropy(output.quotient_logits, quotient))
    return loss, truth


def train_route(config, exp300, source, splits, root, route, steps, output, mode):
    random.seed(root + 670)
    np.random.seed(root + 670)
    torch.manual_seed(root + 670)
    directory = output / mode / f"root-{root}" / route
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "model.safetensors"
    cache = directory / "training-result.json"
    schedules = frozen_count_schedules(config, steps)
    counts = schedules.get(route, [])
    key = json_hash({"source": sha256(source), "root": root, "route": route,
                     "steps": steps, "training": config["training"], "counts": counts,
                     "splits": splits})
    if checkpoint.is_file() and cache.is_file():
        old = json.loads(cache.read_text())
        if old.get("cache_key") == key and sha256(checkpoint) == old["result"]["checkpoint"]["sha256"]:
            model = build_model(config, source)
            model.load_state_dict(load_file(checkpoint))
            return model, {**old["result"], "cached": True}
    model = build_model(config, source)
    timer = Timer()
    history = []
    examples = calls = 0
    if route == "randomized":
        model.randomized_intervention(root + 67091)
    if route in TRAINED_ROUTES:
        parameters = list(model.operator.parameters())
        optimizer = torch.optim.AdamW(parameters,
                                      lr=float(config["training"]["learning_rate"]),
                                      weight_decay=float(config["training"]["weight_decay"]))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=steps,
            eta_min=float(config["training"]["minimum_learning_rate"]))
        samplers = {width: exp300.BigIntSampler(
            splits[width]["train"], root + int(config["data"]["training_seed_offset"]) + width)
            for width in config["data"]["training_widths"]}
        model.train()
        for step, count in enumerate(counts, 1):
            width = int(config["training"]["schedule"][(step - 1) % 4])
            state, multiplier, modulus, input_bit, target, quotient = samplers[width].transitions(
                int(config["training"]["batch_size"]))
            batch = features(state, multiplier, modulus, input_bit, width)
            tick = Timer()
            prediction = model.forward_train(batch, route=route, count=count)
            loss, truth = objective(prediction, target, quotient, width, config)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(
                parameters, float(config["training"]["gradient_clip"])))
            optimizer.step()
            scheduler.step()
            examples += len(target)
            calls += len(target) * count
            history.append({"step": step, "width": width, "count": count,
                            "loss": float(loss.detach()), "gradient_norm": gradient,
                            "batch_exact": float((prediction.logits.ge(0) == truth.bool()).all(1).float().mean()),
                            "relative_residual": float(prediction.relative_residual.mean()),
                            "timing": tick.finish()})
    save_file({name: value.detach().contiguous() for name, value in model.state_dict().items()}, checkpoint)
    reloaded = build_model(config, source)
    reloaded.load_state_dict(load_file(checkpoint))
    exact = all(torch.equal(left, right) for left, right in
                zip(model.state_dict().values(), reloaded.state_dict().values(), strict=True))
    path = str(checkpoint.relative_to(ROOT)) if checkpoint.is_relative_to(ROOT) else str(checkpoint)
    histogram = {str(value): counts.count(value) for value in range(2, 9)} if counts else {}
    result = {"route": route, "cached": False, "parameters": model.parameter_count(),
              "trainable_parameters": model.trainable_parameter_count(),
              "checkpoint": {"path": path, "sha256": sha256(checkpoint),
                             "bytes": checkpoint.stat().st_size},
              "checkpoint_reload_exact": exact,
              "inference_refinement_calls": int(config["inference"]["fixed_count"]) if route != "proposal_only" else 0,
              "training": {"updates": steps if route in TRAINED_ROUTES else 0,
                           "examples": examples, "refinement_calls": calls,
                           "count_sum": sum(counts), "count_mean": float(np.mean(counts)) if counts else 0.0,
                           "count_variance": float(np.var(counts)) if counts else 0.0,
                           "count_histogram": histogram, "counts_sha256": json_hash(counts),
                           "history": history, "timing": timer.finish()}}
    write_json(cache, {"cache_key": key, "result": result})
    return model, result


def evaluate_route(model, route, cases):
    class View:
        def __call__(self, batch, **unused):
            return model(batch, route=route)
    view = View()
    return {"transition": evaluate_transitions(view, route, cases["transition"]),
            "rollout": evaluate_rollouts(view, route, cases["rollout"])}


def depth_diagnostic(model, route, cases, counts):
    records = {}
    for count in counts:
        class View:
            def __call__(self, batch, **unused):
                return model._run(batch, route, int(count))
        records[str(count)] = evaluate_rollouts(View(), route, cases["rollout"])
    return records


class Experiment:
    def __init__(self, config_path, output, mode, root=None):
        self.config_path = Path(config_path).resolve()
        self.config = yaml.safe_load(self.config_path.read_text())
        self.output = Path(output).resolve()
        self.mode = mode
        expected = {"preregister": 20261230, "calibration": 20261230, "formal": 20261231}[mode]
        self.root = expected if root is None else int(root)
        allowed = {expected} if mode != "formal" else {20261231, 20261232, 20261233}
        if self.root not in allowed:
            raise ValueError("unfrozen root")
        self.source = (ROOT / self.config["source_checkpoint"]).resolve()
        if (sha256(self.source) != self.config["source_checkpoint_sha256"] or
                sha256(ROOT / self.config["paper"]["path"]) != self.config["paper"]["sha256"]):
            raise RuntimeError("source identity mismatch")
        for name, digest in self.config["fixture_hashes"].items():
            fixture = ROOT / ".scratch/exp-658-joint-quotient-carry-transducer/pinned_sources" / name
            if sha256(fixture) != digest:
                raise RuntimeError("fixture mismatch")
        self.exp300 = load_exp300()
        self.splits = {width: make_split(self.config, self.exp300, width, self.root)
                       for width in self.config["data"]["widths"]}

    def preregistration(self):
        schedules = frozen_count_schedules(self.config, 400)
        return {"run_id": self.config["run_id"], "root": self.root, "mode": self.mode,
                "source": {"repository_commit": git_output("rev-parse", "HEAD"),
                           "repository_branch": git_output("branch", "--show-current"),
                           "checkpoint_sha256": sha256(self.source),
                           "challenge_commit": "82510bba00a1126649bd76dd1a451f14d0b3eb60"},
                "paper": self.config["paper"], "fixtures": self.config["fixture_hashes"],
                "routes": self.config["routes"], "count_schedule": self.config["count_schedule"],
                "count_hashes": {name: json_hash(values) for name, values in schedules.items()},
                "inference": self.config["inference"], "roots": self.config["roots"],
                "splits": {str(width): {name: json_hash(values) for name, values in split.items()}
                           for width, split in self.splits.items()}, "gates": self.config["gates"],
                "adaptations": ["Conv1d residual operator replaces language attention/FFN",
                                "proposal and six legal input channels are reinjected each loop"],
                "closed": ["M018.b hop/KL/entropy halting curriculum", "adaptive inference stopping",
                           "language-only attention/FFN behavior", "checked/private/candidate/submission"]}

    def cases(self, transitions, rollouts, widths):
        records = {"validation": {}, "test": {}}
        for split in records:
            for width in widths:
                offset = int(self.config["data"]["validation_seed_offset"] if split == "validation"
                             else self.config["data"]["test_seed_offset"])
                records[split][str(width)] = {
                    "transition": transition_cases(self.exp300, self.splits[width][split], width,
                                                   self.root + offset + width, transitions),
                    "rollout": rollout_cases(self.exp300, self.splits[width][split], width,
                                             self.root + offset + width + 500, rollouts)}
        return records

    def run(self):
        self.output.mkdir(parents=True, exist_ok=True)
        preregistration = self.preregistration()
        write_json(self.output / f"preregistration-root-{self.root}.json", preregistration)
        if self.mode == "preregister":
            return preregistration
        calibration_path = self.output / "calibration.json"
        if self.mode == "formal" and (not calibration_path.is_file() or
                not json.loads(calibration_path.read_text())["admission"]["admitted"]):
            raise RuntimeError("formal blocked by calibration")
        config = self.config
        calibration = self.mode == "calibration"
        torch.set_num_threads(int(config["training"]["torch_threads"]))
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        started = Timer()
        configure_mlflow(config["tracking"]["experiment_name"])
        steps = int(config["calibration"]["representative_updates_per_trained_route"] if calibration
                    else config["training"]["steps_per_trained_route"])
        transitions = int(config["calibration"]["transition_cases_per_width"] if calibration
                          else config["evaluation"]["transition_cases_per_width"])
        rollouts = int(config["calibration"]["rollout_cases_per_width"] if calibration
                       else config["evaluation"]["rollout_cases_per_width"])
        widths = list(config["data"]["widths"]) if calibration else [128]
        routes = {}
        with mlflow.start_run(run_name=f"{config['run_id']}-{self.mode}-{self.root}") as tracking:
            for route in ROUTES:
                _, routes[route] = train_route(config, self.exp300, self.source, self.splits,
                                               self.root, route, steps, self.output, self.mode)
            cases = self.cases(transitions, rollouts, widths)
            for route in ROUTES:
                model = build_model(config, self.source)
                model.load_state_dict(load_file(ROOT / routes[route]["checkpoint"]["path"]))
                model.eval()
                routes[route]["validation"] = {}
                routes[route]["test"] = {}
                for split in ("validation", "test"):
                    for width in widths:
                        routes[route][split][str(width)] = evaluate_route(
                            model, route, cases[split][str(width)])
            primary = build_model(config, self.source)
            primary.load_state_dict(load_file(ROOT / routes["clipped_poisson"]["checkpoint"]["path"]))
            primary.eval()
            depth_width = "128"
            depth = depth_diagnostic(primary, "clipped_poisson", cases["test"][depth_width],
                                     config["calibration" if calibration else "evaluation"]["depth_counts"])
            repeat = evaluate_route(primary, "clipped_poisson", cases["test"][depth_width])
            deterministic = (repeat["rollout"]["prediction_sha256"] ==
                             routes["clipped_poisson"]["test"][depth_width]["rollout"]["prediction_sha256"])
            wall_samples = [entry["timing"]["wall_seconds"] for route in TRAINED_ROUTES
                            for entry in routes[route]["training"]["history"]]
            projected_training = sum(wall_samples) * 400 / max(steps, 1)
            variability = max(wall_samples) / max(float(np.mean(wall_samples)), 1e-12)
            headroom = min(2.0, max(1.25, variability))
            primary_eval = routes["clipped_poisson"]["test"][depth_width]["rollout"]["timing"]["wall_seconds"]
            proposal_eval = routes["proposal_only"]["test"][depth_width]["rollout"]["timing"]["wall_seconds"]
            candidate = float(config["resource_policy"]["incumbent_candidate_wall_seconds"]) + max(
                0.0, primary_eval - proposal_eval) * 100 / rollouts
            measured = started.finish()
            resource = aggregate_resource(measured, [], 1)
            projected_evaluation = sum(
                routes[route][split][str(width)][kind]["timing"]["wall_seconds"] *
                (int(config["evaluation"]["rollout_cases_per_width"] if kind == "rollout" else
                     config["evaluation"]["transition_cases_per_width"]) /
                 (rollouts if kind == "rollout" else transitions))
                for route in ROUTES for split in ("validation", "test") for width in widths
                for kind in ("transition", "rollout"))
            expected = projected_training + projected_evaluation
            complete_histogram = (steps != 400 or
                routes["clipped_poisson"]["training"]["count_sum"] == 1600 and
                all(routes[route]["training"]["count_sum"] == 1600 for route in TRAINED_ROUTES))
            checks = {"complete_routes": all(route in routes for route in ROUTES),
                      "expected_value": expected <= float(config["resource_policy"]["maximum_expected_formal_wall_seconds"]),
                      "memory": resource["peak_rss_bytes"] * 3 <= int(config["resource_policy"]["maximum_peak_rss_bytes"]),
                      "candidate_inference": candidate < 300 and candidate * headroom <= 285,
                      "determinism": deterministic,
                      "reload": all(record["checkpoint_reload_exact"] for record in routes.values()),
                      "count_histogram": complete_histogram,
                      "split_separation": all(set(split["train"]).isdisjoint(split["validation"])
                          and set(split["train"]).isdisjoint(split["test"])
                          and set(split["validation"]).isdisjoint(split["test"])
                          for split in self.splits.values())}
            admitted = all(checks.values())
            admission = {"admitted": admitted, "checks": checks,
                         "projected_training_wall_seconds": projected_training,
                         "projected_evaluation_wall_seconds": projected_evaluation,
                         "expected_wall_seconds": expected, "eta_wall_seconds": expected * headroom,
                         "soft_guard_wall_seconds": math.ceil(expected * headroom * 1.25),
                         "hard_guard_wall_seconds": math.ceil(expected * headroom * 2),
                         "variability_ratio": variability, "headroom": headroom,
                         "candidate_inference_wall_seconds": candidate,
                         "candidate_inference_headroom_wall_seconds": candidate * headroom,
                         "aggregate_peak_rss_upper_bound_bytes": resource["peak_rss_bytes"] * 3}
            decision = ("admit-root-20261231" if admitted else
                        "park-resource-compliance-or-feasibility") if calibration else "formal-w128-complete"
            result = {"run_id": config["run_id"], "status": "completed", "mode": self.mode,
                      "root": self.root, "decision": decision, "source": preregistration["source"],
                      "routes": routes, "diagnostics": {"depth": depth,
                          "deterministic_repeat_equal": deterministic}, "admission": admission,
                      "resource": resource, "mlflow": {"tracking_uri": tracking_uri(),
                          "parent_run_id": tracking.info.run_id},
                      "limitations": ["Bounded SAIR adaptation, not a language-model reproduction.",
                          "Input reinjection is an explicit SAIR adaptation; the source omits it.",
                          "M018.b hop/KL/entropy halting remains closed.",
                          "Calibration is non-scientific.",
                          "No checked/private/candidate/submission surface opened."]}
            write_json(calibration_path if calibration else self.output / f"root-{self.root}-summary.json", result)
            mlflow.set_tag("decision", decision)
            return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("preregister", "calibration", "formal"), required=True)
    parser.add_argument("--root", type=int)
    arguments = parser.parse_args()
    result = Experiment(arguments.config, arguments.output, arguments.mode, arguments.root).run()
    print(json.dumps({"decision": result.get("decision"), "root": result.get("root")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
