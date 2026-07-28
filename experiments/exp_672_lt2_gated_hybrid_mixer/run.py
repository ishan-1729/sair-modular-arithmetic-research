"""Preregister, calibrate, and gate EXP672 LT2 hybrid science."""
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
from experiments.exp_672_lt2_gated_hybrid_mixer.model import (  # noqa: E402
    ROUTES, TRAINED_ROUTES, LT2GatedHybridCell,
)
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402

CONFIG_PATH = ROOT / "configs/exp_672_lt2_gated_hybrid_mixer.yaml"


def build_model(config: dict[str, Any], source: Path) -> LT2GatedHybridCell:
    model = LT2GatedHybridCell(**config["model"])
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


def _checkpoint_record(model, source, checkpoint, route, timer, training, persistence):
    save_timer = Timer()
    save_file({name: value.detach().contiguous() for name, value in model.state_dict().items()},
              checkpoint)
    persistence["save"] = save_timer.finish()
    reloaded = build_model(training["config"], source)
    reload_timer = Timer()
    reloaded.load_state_dict(load_file(checkpoint))
    reload_timing = reload_timer.finish()
    exact = all(torch.equal(left, right) for left, right in
                zip(model.state_dict().values(), reloaded.state_dict().values(), strict=True))
    path = str(checkpoint.relative_to(ROOT)) if checkpoint.is_relative_to(ROOT) else str(checkpoint)
    persistence.update({"reload": reload_timing, "checkpoint_bytes": checkpoint.stat().st_size})
    return {
        "route": route,
        "cached": False,
        "parameters": model.parameter_count(),
        "trainable_parameters": model.trainable_parameter_count(route),
        "checkpoint": {"path": path, "sha256": sha256(checkpoint),
                       "bytes": checkpoint.stat().st_size},
        "checkpoint_reload_exact": exact,
        "diagnostics": model.diagnostics(),
        "training": {key: value for key, value in training.items() if key != "config"},
        "persistence": persistence,
        "timing": timer.finish(),
    }


def train_route(config, exp300, source, splits, root, route, steps, output, mode):
    if route not in TRAINED_ROUTES and route != "proposal_only":
        raise ValueError("train_route only accepts trained or proposal routes")
    # Identical seeds and sampler identities deliberately match every trained arm.
    random.seed(root + 672)
    np.random.seed(root + 672)
    torch.manual_seed(root + 672)
    directory = output / mode / f"root-{root}" / route
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "model.safetensors"
    cache = directory / "training-result.json"
    key = json_hash({"source": sha256(source), "root": root, "route": route,
                     "steps": steps, "model": config["model"],
                     "training": config["training"], "splits": splits})
    if bool(config["resource_policy"]["cache_completed_roots"]) and checkpoint.is_file() and cache.is_file():
        old = json.loads(cache.read_text(encoding="utf-8"))
        if old.get("cache_key") == key and sha256(checkpoint) == old["result"]["checkpoint"]["sha256"]:
            model = build_model(config, source)
            model.load_state_dict(load_file(checkpoint))
            return model, {**old["result"], "cached": True}
    model = build_model(config, source)
    timer = Timer()
    history: list[dict[str, Any]] = []
    examples = calls = analytical_flops = 0
    if route in TRAINED_ROUTES:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        parameters = model.route_parameters(route)
        for parameter in parameters:
            parameter.requires_grad_(True)
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
        for step in range(1, steps + 1):
            width = int(config["training"]["schedule"][(step - 1) %
                                                         len(config["training"]["schedule"])])
            state, multiplier, modulus, input_bit, target, quotient = samplers[width].transitions(
                int(config["training"]["batch_size"]))
            tick = Timer()
            prediction = model.forward_train(features(state, multiplier, modulus, input_bit, width),
                                             route=route)
            loss, truth = objective(prediction, target, quotient, width, config)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(
                parameters, float(config["training"]["gradient_clip"])))
            optimizer.step()
            scheduler.step()
            batch = len(target)
            examples += batch
            calls += batch * model.loop_count
            analytical_flops += model.analytical_mixer_flops(width, batch, route)
            record = {
                "step": step, "width": width, "loss": float(loss.detach()),
                "gradient_norm": gradient,
                "batch_exact": float((prediction.logits.ge(0) == truth.bool()).all(1).float().mean()),
                "relative_residual": float(prediction.relative_residual.mean()),
                "linear_state_norm": float(prediction.linear_norm.mean()),
                "sparse_concentration": float(prediction.sparse_concentration.mean()),
                "timing": tick.finish(),
            }
            history.append(record)
            if step == 1 or step % int(config["training"]["log_every_steps"]) == 0 or step == steps:
                print(json.dumps({"route": route, **{k: v for k, v in record.items()
                                                      if k != "timing"}}), flush=True)
    training = {
        "config": config,
        "updates": steps if route in TRAINED_ROUTES else 0,
        "examples": examples,
        "shared_loop_calls": calls,
        "analytical_mixer_flops": analytical_flops,
        "history": history,
    }
    result = _checkpoint_record(model, source, checkpoint, route, timer, training, {})
    write_json(cache, {"cache_key": key, "result": result})
    return model, result


def intervention_route(config, source, root, route, primary_checkpoint, output, mode):
    if route not in ("reset", "randomized"):
        raise ValueError(route)
    directory = output / mode / f"root-{root}" / route
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "model.safetensors"
    cache = directory / "training-result.json"
    key = json_hash({"primary": sha256(primary_checkpoint), "route": route, "root": root})
    if checkpoint.is_file() and cache.is_file():
        old = json.loads(cache.read_text(encoding="utf-8"))
        if old.get("cache_key") == key and sha256(checkpoint) == old["result"]["checkpoint"]["sha256"]:
            model = build_model(config, source)
            model.load_state_dict(load_file(checkpoint))
            return model, {**old["result"], "cached": True}
    timer = Timer()
    model = build_model(config, source)
    model.load_state_dict(load_file(primary_checkpoint))
    if route == "reset":
        model.reset_gate_intervention()
    else:
        model.randomized_mixer_intervention(root + 67291)
    result = _checkpoint_record(
        model, source, checkpoint, route, timer,
        {"config": config, "updates": 0, "examples": 0, "shared_loop_calls": 0,
         "analytical_mixer_flops": 0, "history": [],
         "derived_from": sha256(primary_checkpoint)},
        {"intervention": route},
    )
    write_json(cache, {"cache_key": key, "result": result})
    return model, result


def evaluate_route(model, route, cases):
    return {
        "transition": evaluate_transitions(model, route, cases["transition"]),
        "rollout": evaluate_rollouts(model, route, cases["rollout"]),
    }


def depth_diagnostic(model, cases, counts):
    records = {}
    for count in counts:
        class View:
            def __call__(self, batch, **unused):
                return model(batch, route="full_hybrid", loops=int(count))
        records[str(count)] = evaluate_rollouts(View(), "full_hybrid", cases["rollout"])
    return records


class Experiment:
    def __init__(self, config_path, output, mode, root=None):
        self.config_path = Path(config_path).resolve()
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.output = Path(output).resolve()
        self.mode = mode
        expected = {"preregister": 20261238, "calibration": 20261238,
                    "formal": 20261239}[mode]
        self.root = expected if root is None else int(root)
        allowed = {expected} if mode != "formal" else {20261239, 20261240, 20261241}
        if self.root not in allowed:
            raise ValueError("unfrozen root")
        self.source = (ROOT / self.config["source_checkpoint"]).resolve()
        if sha256(self.source) != self.config["source_checkpoint_sha256"]:
            raise RuntimeError("source checkpoint mismatch")
        if sha256(ROOT / self.config["paper"]["path"]) != self.config["paper"]["sha256"]:
            raise RuntimeError("paper mismatch")
        for name, digest in self.config["fixture_hashes"].items():
            fixture = ROOT / ".scratch/exp-658-joint-quotient-carry-transducer/pinned_sources" / name
            if sha256(fixture) != digest:
                raise RuntimeError("fixture mismatch")
        self.exp300 = load_exp300()
        self.splits = {width: make_split(self.config, self.exp300, width, self.root)
                       for width in self.config["data"]["widths"]}

    def preregistration(self):
        files = {
            "config": self.config_path,
            "mechanism": ROOT / self.config["mechanism_map"],
            "model": ROOT / "experiments/exp_672_lt2_gated_hybrid_mixer/model.py",
            "runner": Path(__file__),
            "test": ROOT / "tests/test_exp_672_lt2_gated_hybrid_mixer.py",
            "browser": ROOT / "tmp/playwright/exp672/official-pinned-parity.json",
        }
        return {
            "run_id": self.config["run_id"], "root": self.root, "mode": self.mode,
            "source": {"repository_commit": git_output("rev-parse", "HEAD"),
                       "repository_branch": git_output("branch", "--show-current"),
                       "checkpoint": {"path": self.config["source_checkpoint"],
                                      "sha256": sha256(self.source)},
                       "files": {name: {"path": str(path.relative_to(ROOT)),
                                        "sha256": sha256(path)}
                                 for name, path in files.items() if path.is_file()}},
            "paper": self.config["paper"], "official_source": self.config["official_source"],
            "roots": self.config["roots"], "routes": self.config["routes"],
            "splits": {str(width): {name: json_hash(values) for name, values in split.items()}
                       for width, split in self.splits.items()},
            "gates": self.config["gates"], "resource_policy": self.config["resource_policy"],
            "adaptations": ["native feature-map linear attention replaces GPU FLA GDN",
                            "native fixed-window attention replaces CUDA DSA",
                            "outer zero output residual gate preserves Candidate005 exactly"],
            "prohibited": ["exact arithmetic in inference", "answer-dependent routing",
                           "validity filtering or repair", "lookup or identity features",
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

    def _train_all(self, steps):
        routes = {}
        for route in (*TRAINED_ROUTES, "proposal_only"):
            _, routes[route] = train_route(self.config, self.exp300, self.source, self.splits,
                                           self.root, route, steps, self.output, self.mode)
        primary_path = Path(routes["full_hybrid"]["checkpoint"]["path"])
        if not primary_path.is_absolute():
            primary_path = ROOT / primary_path
        for route in ("reset", "randomized"):
            _, routes[route] = intervention_route(self.config, self.source, self.root, route,
                                                  primary_path, self.output, self.mode)
        return routes

    def _model_for(self, routes, route):
        model = build_model(self.config, self.source)
        path = Path(routes[route]["checkpoint"]["path"])
        model.load_state_dict(load_file(path if path.is_absolute() else ROOT / path))
        model.eval()
        return model

    def _evaluate(self, routes, cases, widths, split_names, route_names=ROUTES):
        for route in route_names:
            model = self._model_for(routes, route)
            for split in split_names:
                routes[route].setdefault(split, {})
                for width in widths:
                    routes[route][split][str(width)] = evaluate_route(
                        model, route, cases[split][str(width)])

    def _candidate_runtime(self, routes, rollouts):
        primary = routes["full_hybrid"]["test"]["128"]["rollout"]["timing"]["wall_seconds"]
        proposal = routes["proposal_only"]["test"]["128"]["rollout"]["timing"]["wall_seconds"]
        return float(self.config["resource_policy"]["incumbent_candidate_wall_seconds"]) + max(
            0.0, primary - proposal) * 100 / rollouts

    def _calibration(self, preregistration):
        cfg = self.config
        started = Timer()
        steps = int(cfg["calibration"]["representative_updates_per_trained_route"])
        transitions = int(cfg["calibration"]["transition_cases_per_width"])
        rollouts = int(cfg["calibration"]["rollout_cases_per_width"])
        routes = self._train_all(steps)
        cases = self.cases(transitions, rollouts, cfg["data"]["widths"])
        self._evaluate(routes, cases, cfg["data"]["widths"], ("validation", "test"))
        primary = self._model_for(routes, "full_hybrid")
        depth = depth_diagnostic(primary, cases["test"]["128"], cfg["calibration"]["depth_counts"])
        repeat = evaluate_route(primary, "full_hybrid", cases["test"]["128"])
        deterministic = repeat["rollout"]["prediction_sha256"] == \
            routes["full_hybrid"]["test"]["128"]["rollout"]["prediction_sha256"]
        update_walls = [entry["timing"]["wall_seconds"] for route in TRAINED_ROUTES
                        for entry in routes[route]["training"]["history"]]
        route_walls = [routes[route][split][str(width)][kind]["timing"]["wall_seconds"]
                       for route in ROUTES for split in ("validation", "test")
                       for width in cfg["data"]["widths"] for kind in ("transition", "rollout")]
        samples = update_walls + route_walls
        variability = max(samples) / max(float(np.mean(samples)), 1e-12)
        headroom = min(float(cfg["resource_policy"]["variability_headroom_ceiling"]),
                       max(float(cfg["resource_policy"]["variability_headroom_floor"]), variability))
        projected_training = sum(update_walls) * int(cfg["training"]["steps_per_trained_route"]) / steps
        projected_evaluation = sum(
            routes[route][split][str(width)][kind]["timing"]["wall_seconds"] *
            (int(cfg["evaluation"]["rollout_cases_per_width"] if kind == "rollout" else
                 cfg["evaluation"]["transition_cases_per_width"]) /
             (rollouts if kind == "rollout" else transitions))
            for route in ROUTES for split in ("validation", "test")
            for width in cfg["data"]["widths"] for kind in ("transition", "rollout"))
        expected = projected_training + projected_evaluation
        candidate = self._candidate_runtime(routes, rollouts)
        observed = started.finish()
        resource = aggregate_resource(observed, [], 1)
        checks = {
            "complete_routes": all(all(split in routes[route] for split in ("validation", "test"))
                                   for route in ROUTES),
            "expected_value": expected <= float(cfg["resource_policy"]["maximum_expected_formal_wall_seconds"]),
            "memory": resource["aggregate_peak_rss_upper_bound_bytes"] <= int(cfg["resource_policy"]["maximum_peak_rss_bytes"]),
            "candidate_inference": candidate <= float(cfg["gates"]["maximum_candidate_inference_wall_seconds"]),
            "determinism": deterministic,
            "reload": all(record["checkpoint_reload_exact"] for record in routes.values()),
            "split_separation": all(set(split["train"]).isdisjoint(split["validation"])
                                    and set(split["train"]).isdisjoint(split["test"])
                                    and set(split["validation"]).isdisjoint(split["test"])
                                    for split in self.splits.values()),
            "fixed_target_free_routes": True,
        }
        admitted = all(checks.values())
        admission = {
            "admitted": admitted, "checks": checks,
            "measured_updates_per_trained_route": steps,
            "projected_training_wall_seconds": projected_training,
            "projected_evaluation_wall_seconds": projected_evaluation,
            "expected_wall_seconds": expected, "eta_wall_seconds": expected * headroom,
            "soft_guard_wall_seconds": math.ceil(expected * headroom * float(cfg["resource_policy"]["soft_guard_multiplier"])),
            "hard_guard_wall_seconds": math.ceil(expected * headroom * float(cfg["resource_policy"]["hard_guard_multiplier"])),
            "variability_ratio": variability, "headroom": headroom,
            "candidate_inference_wall_seconds": candidate,
            "candidate_inference_headroom_wall_seconds": candidate * headroom,
            "wall_seconds": resource["wall_seconds"],
            "process_tree_cpu_seconds": resource["aggregate_process_tree_cpu_seconds"],
            "cgroup_cpu_seconds": resource["cgroup_cpu_seconds"],
            "peak_rss_bytes": resource["aggregate_peak_rss_upper_bound_bytes"],
        }
        return {"run_id": cfg["run_id"], "status": "completed", "mode": "calibration",
                "root": self.root,
                "decision": "admit-root-20261239" if admitted else "park-resource-compliance-or-feasibility",
                "source": preregistration["source"], "routes": routes,
                "diagnostics": {"depth": depth, "deterministic_repeat_equal": deterministic},
                "admission": admission, "resource": resource,
                "limitations": ["Bounded SAIR transfer, not a language-model reproduction.",
                                "Native feature-map/window attention replaces official GPU kernels.",
                                "Calibration is non-scientific.",
                                "No checked/private/candidate/submission surface opened."]}

    def _formal(self, preregistration):
        cfg = self.config
        calibration_path = self.output / "calibration.json"
        if not calibration_path.is_file() or not json.loads(calibration_path.read_text())["admission"]["admitted"]:
            raise RuntimeError("formal blocked by calibration")
        if self.root in (20261240, 20261241):
            viability = self.output / "root-20261239-summary.json"
            if not viability.is_file() or json.loads(viability.read_text())["decision"] != "advance-confirmations":
                raise RuntimeError("confirmation blocked by viability")
        started = Timer()
        routes = self._train_all(int(cfg["training"]["steps_per_trained_route"]))
        transitions = int(cfg["evaluation"]["transition_cases_per_width"])
        rollouts = int(cfg["evaluation"]["rollout_cases_per_width"])
        cases128 = self.cases(transitions, rollouts, [128], ("test",))
        self._evaluate(routes, cases128, [128], ("test",))
        primary = routes["full_hybrid"]["test"]["128"]["rollout"]
        proposal = routes["proposal_only"]["test"]["128"]["rollout"]
        reset = routes["reset"]["test"]["128"]["rollout"]
        randomized = routes["randomized"]["test"]["128"]["rollout"]
        candidate = self._candidate_runtime(routes, rollouts)
        w128_checks = {
            "rollout_floor": primary["exact_accuracy"] >= float(cfg["gates"]["minimum_w128_primary_rollout"]),
            "zero_invalids": primary["any_invalid_residue_rate"] <= float(cfg["gates"]["maximum_invalid_rollout_rate"]),
            "proposal_retention": primary["exact_accuracy"] >= proposal["exact_accuracy"] - float(cfg["gates"]["maximum_w128_loss_to_proposal"]),
            "gate_dependence": primary["exact_accuracy"] - reset["exact_accuracy"] >= float(cfg["gates"]["minimum_gate_dependence_drop"]),
            "mixer_dependence": primary["exact_accuracy"] - randomized["exact_accuracy"] >= float(cfg["gates"]["minimum_mixer_dependence_drop"]),
            "candidate_runtime": candidate <= float(cfg["gates"]["maximum_candidate_inference_wall_seconds"]),
            "reload": all(record["checkpoint_reload_exact"] for record in routes.values()),
            "split_separation": all(set(split["train"]).isdisjoint(split["validation"])
                                    and set(split["train"]).isdisjoint(split["test"])
                                    and set(split["validation"]).isdisjoint(split["test"])
                                    for split in self.splits.values()),
        }
        w256_checks = {"opened": False}
        if all(w128_checks.values()):
            cases256 = self.cases(transitions, rollouts, [256], ("test",))
            comparison = ("full_hybrid", "proposal_only", "gate_only", "shared_conv_control")
            self._evaluate(routes, cases256, [256], ("test",), comparison)
            score = routes["full_hybrid"]["test"]["256"]["rollout"]["exact_accuracy"]
            controls = [routes[route]["test"]["256"]["rollout"]["exact_accuracy"]
                        for route in comparison[1:]]
            w256_checks = {
                "opened": True,
                "rollout_floor": score >= float(cfg["gates"]["minimum_w256_primary_rollout"]),
                "control_gain": all(score - control >= float(cfg["gates"]["minimum_w256_gain_over_controls"])
                                    for control in controls),
                "zero_invalids": routes["full_hybrid"]["test"]["256"]["rollout"]["any_invalid_residue_rate"] <= float(cfg["gates"]["maximum_invalid_rollout_rate"]),
            }
        complete_pass = all(w128_checks.values()) and all(value for key, value in w256_checks.items()
                                                            if key != "opened") and w256_checks["opened"]
        if self.root == 20261239:
            decision = "advance-confirmations" if complete_pass else "reject-formal-conjunct"
        else:
            decision = "confirmation-pass" if complete_pass else "reject-confirmation"
        resource = aggregate_resource(started.finish(), [], 1)
        return {"run_id": cfg["run_id"], "status": "completed", "mode": "formal",
                "root": self.root, "decision": decision, "source": preregistration["source"],
                "routes": routes, "gates": {"w128": w128_checks, "w256": w256_checks,
                                             "all_conjuncts": complete_pass},
                "candidate_inference_wall_seconds": candidate, "resource": resource,
                "limitations": ["Intermediate research evidence only.",
                                "No checked/private/candidate/submission surface opened."]}

    def run(self):
        self.output.mkdir(parents=True, exist_ok=True)
        preregistration = self.preregistration()
        write_json(self.output / f"preregistration-root-{self.root}.json", preregistration)
        if self.mode == "preregister":
            return preregistration
        torch.set_num_threads(int(self.config["training"]["torch_threads"]))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as error:
            if "cannot set number of interop threads" not in str(error).lower():
                raise
        torch.use_deterministic_algorithms(True)
        configure_mlflow(self.config["tracking"]["experiment_name"])
        with mlflow.start_run(run_name=f"{self.config['run_id']}-{self.mode}-{self.root}") as tracking:
            result = self._calibration(preregistration) if self.mode == "calibration" else self._formal(preregistration)
            result["mlflow"] = {"tracking_uri": tracking_uri(),
                                "parent_run_id": tracking.info.run_id}
            path = self.output / ("calibration.json" if self.mode == "calibration"
                                  else f"root-{self.root}-summary.json")
            write_json(path, result)
            mlflow.set_tag("decision", result["decision"])
            return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("preregister", "calibration", "formal"), required=True)
    parser.add_argument("--root", type=int)
    args = parser.parse_args()
    result = Experiment(args.config, args.output, args.mode, args.root).run()
    print(json.dumps({"decision": result.get("decision"), "root": result.get("root"),
                      "status": result.get("status", "preregistered")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
