"""Execute EXP685's training-only validity-constrained EXP665 continuation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import platform
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import psutil
import torch
import yaml
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.exp_664_sprm_boundary_perturbation.run import (  # noqa: E402
    Timer,
    bits,
    decode,
    deterministic_makespan,
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
from experiments.exp_665_attractor_equilibrium_family.model import AttractorCell  # noqa: E402
from experiments.exp_665_attractor_equilibrium_family.run import (  # noqa: E402
    candidate_runtime,
    evaluate_rollouts,
    evaluate_transitions,
)
from experiments.exp_685_validity_constrained_attractor.model import (  # noqa: E402
    build_model,
    lexicographic_invalid_probability,
    supervised_objective,
)
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402


CONFIG_PATH = ROOT / "configs/exp_685_validity_constrained_attractor.yaml"
ARM_ORDER = ("validity_constrained", "ce_only", "shuffled_modulus", "frozen_source", "randomized_control")
TRAINED_ARMS = ARM_ORDER[:3]
EXP665_PATHS = {
    "exp665_config": ROOT / "configs/exp_665_attractor_equilibrium_family.yaml",
    "exp665_model": ROOT / "experiments/exp_665_attractor_equilibrium_family/model.py",
    "exp665_runner": ROOT / "experiments/exp_665_attractor_equilibrium_family/run.py",
    "exp665_test": ROOT / "tests/test_exp_665_attractor_equilibrium_family.py",
    "exp665_preregistration": ROOT / "experiments/runs/exp-665-attractor-equilibrium-family/preregistration.json",
    "exp665_mechanism": ROOT / "knowledge/autoresearch/research/2026-07-29-exp665-attractor-equilibrium-mechanism.md",
    "attractor_paper": ROOT / "research/papers/arxiv-2605.12466.pdf",
    "equilibrium_paper": ROOT / "research/papers/arxiv-2605.21488.pdf",
}


def _configure_torch(config: dict[str, Any]) -> None:
    torch.set_num_threads(int(config["training"]["torch_threads"]))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)


def _gradient_norm(values: tuple[torch.Tensor | None, ...]) -> float:
    squares = [value.detach().double().square().sum() for value in values if value is not None]
    return 0.0 if not squares else float(torch.stack(squares).sum().sqrt())


def inference_identity(model: AttractorCell, checkpoint: Path) -> dict[str, Any]:
    reloaded = build_model({"model": model_config(model)}, checkpoint)
    state_equal = all(torch.equal(left, right) for left, right in zip(model.state_dict().values(), reloaded.state_dict().values(), strict=True))
    return {
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "is_exact_exp665_class": type(model) is AttractorCell,
        "parameter_count": model.parameter_count(),
        "trainable_parameter_count": model.trainable_parameter_count(),
        "state_dict_reload_exact": state_equal,
        "forward_sha256": hashlib.sha256(inspect.getsource(AttractorCell.forward).encode()).hexdigest(),
        "new_inference_parameters": 0,
        "new_inference_calls": 0,
    }


def model_config(model: AttractorCell) -> dict[str, Any]:
    return {
        "hidden": model.base.hidden,
        "cycles": model.base.cycles,
        "correction_hidden": model.correction.inner.out_channels,
        "correction_scale": model.correction.scale,
        "fixed_depth": model.fixed_depth,
        "solver_max_iterations": model.solver_max_iterations,
        "solver_tolerance": model.solver_tolerance,
        "anderson_history": model.anderson_history,
        "anderson_regularization": model.anderson_regularization,
        "implicit_backward_iterations": model.implicit_backward_iterations,
        "implicit_backward_tolerance": model.implicit_backward_tolerance,
        "eqr_damping": model.eqr_damping,
        "eqr_noise": model.eqr_noise,
        "eqr_initialization_std": model.eqr_initialization_std,
        "eqr_training_breadth": model.eqr_training_breadth,
        "eqr_inference_breadth": model.eqr_inference_breadth,
    }


def train_arm(
    config: dict[str, Any], exp300, checkpoint: Path, splits: dict[int, dict[str, list[int]]],
    seed: int, arm: str, steps: int, output: Path,
) -> tuple[AttractorCell, dict[str, Any]]:
    kind = str(config["routes"][arm]["kind"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    randomized_seed = seed + 68591 if kind == "randomized" else None
    model = build_model(config, checkpoint, randomized_seed=randomized_seed)
    timer, history = Timer(), []
    parameters = tuple(model.correction.parameters())
    examples = operator_calls = nonzero_constraint_steps = 0
    if bool(config["routes"][arm]["trained"]):
        optimizer = torch.optim.AdamW(parameters, lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, steps), eta_min=float(config["training"]["minimum_learning_rate"]))
        samplers = {width: exp300.BigIntSampler(splits[width]["train"], seed + int(config["data"]["training_seed_offset"]) + width) for width in config["data"]["training_widths"]}
        generator = torch.Generator().manual_seed(seed + 68503)
        model.train()
        for step in range(1, steps + 1):
            width = int(config["training"]["schedule"][(step - 1) % len(config["training"]["schedule"])])
            state, multiplicand, modulus, input_bit, target, quotient = samplers[width].transitions(int(config["training"]["batch_size"]))
            tick = Timer()
            output_value = model(features(state, multiplicand, modulus, input_bit, width), route="anderson_phantom", generator=generator)
            loss, pieces = supervised_objective(output_value, bits(target, width), quotient, bits(modulus, width), config, kind=kind, permutation_seed=seed + step)
            constraint_gradients = torch.autograd.grad(pieces["validity_loss"], parameters, retain_graph=True, allow_unused=True)
            constraint_gradient_norm = _gradient_norm(constraint_gradients)
            nonzero_constraint_steps += int(constraint_gradient_norm >= float(config["gates"]["minimum_constraint_gradient_norm"]))
            optimizer.zero_grad(set_to_none=True); loss.backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, float(config["training"]["gradient_clip"])))
            optimizer.step(); scheduler.step()
            examples += len(target); operator_calls += int(output_value.iterations.sum()) + len(target)
            record = {
                "step": step, "width": width, "loss": float(loss.detach()), "ce_loss": float(pieces["ce_loss"].detach()),
                "validity_loss": float(pieces["validity_loss"].detach()), "constraint_gradient_norm": constraint_gradient_norm,
                "gradient_norm": gradient_norm, "invalid_probability_mean": float(pieces["invalid_probability_mean"].detach()),
                "invalid_probability_max": float(pieces["invalid_probability_max"].detach()), "timing": tick.finish(),
            }
            history.append(record)
            if step == 1 or step % int(config["training"]["log_every_steps"]) == 0 or step == steps:
                print(json.dumps({"arm": arm, **{key: value for key, value in record.items() if key != "timing"}}), flush=True)
    arm_dir = output / arm; arm_dir.mkdir(parents=True, exist_ok=True)
    path = arm_dir / "model.safetensors"
    save_file({name: value.detach().contiguous() for name, value in model.state_dict().items()}, path)
    reloaded = build_model(config, path)
    identity = inference_identity(model, path)
    record = {
        "arm": arm, "kind": kind,
        "training": {"updates": steps if arm in TRAINED_ARMS else 0, "examples": examples, "supervision_calls": steps if arm in TRAINED_ARMS else 0, "operator_calls": operator_calls, "constraint_nonzero_steps": nonzero_constraint_steps, "history": history, "resource": timer.finish()},
        "checkpoint": {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), "sha256": sha256(path), "bytes": path.stat().st_size},
        "inference_identity": identity,
        "reload_exact": all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), reloaded.state_dict().values(), strict=True)),
    }
    return model.eval(), record


@torch.inference_mode()
def validity_diagnostic(model: AttractorCell, cases: dict[str, Any]) -> dict[str, Any]:
    timer, width = Timer(), int(cases["width"])
    probabilities: list[float] = []; observed: list[float] = []
    for start in range(0, len(cases["target"]), 128):
        stop = start + 128
        result = model(features(cases["state"][start:stop], cases["multiplicand"][start:stop], cases["modulus"][start:stop], cases["input_bit"][start:stop], width), route="anderson_phantom")
        probability = lexicographic_invalid_probability(result.logits, bits(cases["modulus"][start:stop], width))
        values = decode(result.logits.ge(0)); labels = [float(value >= modulus) for value, modulus in zip(values, cases["modulus"][start:stop], strict=True)]
        probabilities.extend(probability.tolist()); observed.extend(labels)
    probability_tensor = torch.tensor(probabilities, dtype=torch.float64); observed_tensor = torch.tensor(observed, dtype=torch.float64)
    return {
        "cases": len(observed), "observed_invalids": int(sum(observed)), "observed_invalid_rate": float(observed_tensor.mean()),
        "predicted_invalid_probability_mean": float(probability_tensor.mean()), "predicted_invalid_probability_max": float(probability_tensor.max()),
        "brier": float((probability_tensor - observed_tensor).square().mean()), "probability_sha256": json_hash(probabilities), "timing": timer.finish(),
    }


def evaluate_cell(model: AttractorCell, cases: dict[str, Any]) -> dict[str, Any]:
    model.eval()
    transition = evaluate_transitions(model, "anderson_phantom", cases["transition"])
    rollout = evaluate_rollouts(model, "anderson_phantom", cases["rollout"])
    transition["invalids"] = round(transition["invalid_residue_rate"] * transition["cases"])
    rollout["invalids"] = round(rollout["any_invalid_residue_rate"] * rollout["cases"])
    return {"transition": transition, "rollout": rollout, "validity": validity_diagnostic(model, cases["transition"])}


def split_separation(splits: dict[int, dict[str, list[int]]]) -> bool:
    return all(set(group["train"]).isdisjoint(group["validation"]) and set(group["train"]).isdisjoint(group["test"]) and set(group["validation"]).isdisjoint(group["test"]) for group in splits.values())


class Experiment:
    def __init__(self, config_path: Path, output: Path, mode: str, root: int | None) -> None:
        self.config_path = config_path.resolve(); self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.output = output.resolve(); self.output.mkdir(parents=True, exist_ok=True); self.mode = mode; self.root = root
        self.checkpoint = (ROOT / self.config["source_checkpoint"]).resolve()
        self._authenticate()
        _configure_torch(self.config)
        self.exp300 = load_exp300()
        if mode != "smoke":
            expected = int(self.config["roots"]["calibration"] if mode == "calibration" else self.config["roots"]["formal"] if mode == "formal" else -1)
            allowed = {expected} if mode != "confirmation" else set(map(int, self.config["roots"]["confirmations"]))
            self.root = expected if root is None and mode != "confirmation" else root
            if self.root is None or int(self.root) not in allowed:
                raise ValueError(f"root {self.root} is not frozen for {mode}")
            self.root = int(self.root)
            self.splits = {width: make_split(self.config, self.exp300, width, self.root) for width in self.config["data"]["widths"]}

    def _authenticate(self) -> None:
        if sha256(self.checkpoint) != self.config["source_checkpoint_sha256"]:
            raise RuntimeError("EXP665 continuation checkpoint hash mismatch")
        if importlib.metadata.version("torchdeq") != self.config["dependency"]["version"]:
            raise RuntimeError("torchdeq version mismatch")
        contract = self.config["source_contract"]
        for name, path in EXP665_PATHS.items():
            expected = contract[f"{name}_sha256"]
            if sha256(path) != expected:
                raise RuntimeError(f"frozen source hash mismatch: {name}")

    def source_identity(self) -> dict[str, Any]:
        authored = {
            "config": self.config_path,
            "model": ROOT / "experiments/exp_685_validity_constrained_attractor/model.py",
            "runner": ROOT / "experiments/exp_685_validity_constrained_attractor/run.py",
            "test": ROOT / "tests/test_exp_685_validity_constrained_attractor.py",
            "preregistration": ROOT / self.config["preregistration"],
        }
        return {
            "repository_commit": git_output("rev-parse", "HEAD"), "challenge_commit": git_output("-C", "sources/github/modular-arithmetic-challenge", "rev-parse", "HEAD"),
            "checkpoint_sha256": sha256(self.checkpoint), "torchdeq": importlib.metadata.version("torchdeq"),
            "exp665": {name: sha256(path) for name, path in EXP665_PATHS.items()},
            "authored": {name: sha256(path) for name, path in authored.items() if path.is_file()},
        }

    def _root_dir(self) -> Path:
        return self.output / self.mode / f"root-{self.root}"

    def open_root(self) -> Path:
        root_dir = self._root_dir()
        if root_dir.exists():
            raise RuntimeError(f"root {self.root} is already opened")
        root_dir.mkdir(parents=True)
        write_json(root_dir / "root-opened.json", {"root": self.root, "mode": self.mode, "source": self.source_identity()})
        return root_dir

    def frozen_cases(self, widths: list[int], transitions: int, rollouts: int) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {"validation": {}, "test": {}}
        for split in result:
            offset = int(self.config["data"]["validation_seed_offset"] if split == "validation" else self.config["data"]["test_seed_offset"])
            for width in widths:
                result[split][str(width)] = {
                    "transition": transition_cases(self.exp300, self.splits[width][split], width, self.root + offset + width, transitions),
                    "rollout": rollout_cases(self.exp300, self.splits[width][split], width, self.root + offset + width + 500, rollouts),
                }
        return result

    def train_all(self, root_dir: Path, steps: int) -> tuple[dict[str, AttractorCell], dict[str, Any]]:
        models, records = {}, {}
        for arm in ARM_ORDER:
            model, record = train_arm(self.config, self.exp300, self.checkpoint, self.splits, int(self.root), arm, steps, root_dir / "arms")
            models[arm], records[arm] = model, record
            write_json(root_dir / "progress.json", {"root": self.root, "mode": self.mode, "completed_arms": records})
        return models, records

    def evaluate_all(self, models: dict[str, AttractorCell], cases: dict[str, dict[str, Any]], widths: list[int], arms: tuple[str, ...] = ARM_ORDER) -> dict[str, Any]:
        evaluations: dict[str, Any] = {}
        for arm in arms:
            evaluations[arm] = {}
            for split in ("validation", "test"):
                evaluations[arm][split] = {str(width): evaluate_cell(models[arm], cases[split][str(width)]) for width in widths}
        return evaluations

    def admission(self, records: dict[str, Any], evaluations: dict[str, Any], runtime: dict[str, Any], observed_resource: dict[str, Any], calibration_steps: int, transitions: int, rollouts: int) -> dict[str, Any]:
        update_resources = [item["timing"] for arm in TRAINED_ARMS for item in records[arm]["training"]["history"]]
        scale = int(self.config["training"]["steps"]) / calibration_steps
        training_wall = sum(float(item["wall_seconds"]) for item in update_resources) * scale
        training_cpu = sum(float(item["process_cpu_seconds"]) for item in update_resources) * scale
        training_cgroup = sum(float(item["cgroup_cpu_seconds"] or 0.0) for item in update_resources) * scale
        route_wall: dict[str, float] = {}; route_cpu: dict[str, float] = {}; route_cgroup: dict[str, float] = {}
        for arm in ARM_ORDER:
            wall = cpu = cgroup = 0.0
            for split in ("validation", "test"):
                target_t = int(self.config["evaluation"]["validation_transition_cases_per_width"] if split == "validation" else self.config["evaluation"]["transition_cases_per_width"])
                target_r = int(self.config["evaluation"]["validation_rollout_cases_per_width"] if split == "validation" else self.config["evaluation"]["rollout_cases_per_width"])
                for width in self.config["data"]["widths"]:
                    cell = evaluations[arm][split][str(width)]
                    for key, target, observed in (("transition", target_t, transitions), ("rollout", target_r, rollouts), ("validity", target_t, transitions)):
                        timing = cell[key]["timing"]; factor = target / observed
                        wall += float(timing["wall_seconds"]) * factor; cpu += float(timing["process_cpu_seconds"]) * factor; cgroup += float(timing["cgroup_cpu_seconds"] or 0.0) * factor
            route_wall[arm], route_cpu[arm], route_cgroup[arm] = wall, cpu, cgroup
        evaluation_wall = deterministic_makespan(list(route_wall.values()), int(self.config["resource_policy"]["evaluator_processes"]))
        expected_wall = training_wall + evaluation_wall + float(runtime["solver"]["wall_seconds"])
        expected_cpu = training_cpu + sum(route_cpu.values()) + float(runtime["solver"]["process_cpu_seconds"])
        expected_cgroup = training_cgroup + sum(route_cgroup.values()) + float(runtime["solver"]["cgroup_cpu_seconds"] or 0.0)
        walls = [float(item["wall_seconds"]) for item in update_resources]
        variability = max(walls) / max(float(np.mean(walls)), 1e-12)
        policy = self.config["resource_policy"]; headroom = min(float(policy["variability_headroom_ceiling"]), max(float(policy["variability_headroom_floor"]), variability))
        candidate_guarded = float(runtime["candidate_projected_wall_seconds"]) * headroom
        expected_rss = int(observed_resource["peak_rss_bytes"] * headroom)
        admission = {
            "complete_routes": set(records) == set(ARM_ORDER) and set(evaluations) == set(ARM_ORDER),
            "projected_training_wall_seconds": training_wall, "projected_route_wall_seconds": route_wall, "projected_evaluation_wall_seconds": evaluation_wall,
            "expected_wall_seconds": expected_wall, "eta_wall_seconds": expected_wall * headroom,
            "soft_guard_wall_seconds": math.ceil(expected_wall * headroom * float(policy["soft_guard_multiplier"])),
            "hard_guard_wall_seconds": math.ceil(expected_wall * headroom * float(policy["hard_guard_multiplier"])),
            "expected_process_tree_cpu_seconds": expected_cpu, "expected_cgroup_cpu_seconds": expected_cgroup,
            "expected_peak_rss_bytes": expected_rss, "variability_ratio": variability, "headroom": headroom,
            "candidate_projected_wall_seconds": runtime["candidate_projected_wall_seconds"], "candidate_guarded_wall_seconds": candidate_guarded,
            "fidelity_reduced": bool(policy["fidelity_reduced"]),
        }
        admission["checks"] = {
            "complete_routes": admission["complete_routes"], "expected_value": expected_wall <= float(policy["maximum_expected_formal_wall_seconds"]),
            "memory": expected_rss <= int(policy["maximum_peak_rss_bytes"]), "candidate_inference": candidate_guarded <= float(self.config["gates"]["maximum_candidate_inference_wall_seconds"]),
            "reload": all(record["reload_exact"] and record["inference_identity"]["state_dict_reload_exact"] for record in records.values()),
            "inference_identity": all(record["inference_identity"]["is_exact_exp665_class"] and record["inference_identity"]["new_inference_parameters"] == 0 and record["inference_identity"]["new_inference_calls"] == 0 for record in records.values()),
            "separation": split_separation(self.splits), "no_fidelity_reduction": not admission["fidelity_reduced"],
        }
        admission["admitted"] = all(admission["checks"].values())
        return admission

    def smoke(self) -> dict[str, Any]:
        seed = 685000
        splits = {width: make_split(self.config, self.exp300, width, seed) for width in self.config["data"]["widths"]}
        previous_splits, previous_root = getattr(self, "splits", None), self.root
        self.splits, self.root = splits, seed
        with tempfile.TemporaryDirectory(prefix="exp685-smoke-", dir=self.output) as directory:
            temp = Path(directory); models, records = self.train_all(temp, 1)
            cases = self.frozen_cases([64], 1, 1); evaluations = self.evaluate_all(models, cases, [64])
            runtime = candidate_runtime(models["validity_constrained"], cases["test"]["64"]["transition"], 1, float(self.config["resource_policy"]["incumbent_candidate_wall_seconds"]))
            resource = Timer().finish()
            synthetic = {arm: {split: {str(width): evaluations[arm][split]["64"] for width in self.config["data"]["widths"]} for split in ("validation", "test")} for arm in ARM_ORDER}
            admission = self.admission(records, synthetic, runtime, resource, 1, 1, 1)
            serialized = []
            for mode in ("calibration", "formal", "confirmation"):
                path = temp / f"{mode}-terminal.json"
                write_json(path, {"schema_version": "exp685-smoke-terminal-v1", "mode": mode, "status": "completed", "root": None, "records": records, "evaluations": evaluations, "admission": admission, "scientific": False})
                json.loads(path.read_text()); serialized.append(path.name)
            result = {"schema_version": "exp685-rootless-smoke-v1", "status": "completed", "root": None, "seed": seed, "trained_arms": list(TRAINED_ARMS), "all_arms": list(ARM_ORDER), "one_update_each": all(records[arm]["training"]["updates"] == 1 for arm in TRAINED_ARMS), "one_case_transition_rollout_each": all(evaluations[arm]["test"]["64"][kind]["cases"] == 1 for arm in ARM_ORDER for kind in ("transition", "rollout")), "reload_exact": all(record["reload_exact"] for record in records.values()), "admission_math_exercised": bool(admission["checks"]), "terminal_serializations": serialized}
        self.splits, self.root = previous_splits, previous_root
        write_json(self.output / "smoke.json", result); return result

    def calibration(self) -> dict[str, Any]:
        root_dir = self.open_root(); timer = Timer(); steps = int(self.config["training"]["calibration_steps"])
        transitions = int(self.config["calibration"]["transition_cases_per_width"]); rollouts = int(self.config["calibration"]["rollout_cases_per_width"])
        models, records = self.train_all(root_dir, steps); cases = self.frozen_cases(list(self.config["data"]["widths"]), transitions, rollouts)
        evaluations = self.evaluate_all(models, cases, list(self.config["data"]["widths"])); runtime = candidate_runtime(models["validity_constrained"], cases["test"]["128"]["transition"], int(self.config["calibration"]["candidate_runtime_repeats"]), float(self.config["resource_policy"]["incumbent_candidate_wall_seconds"]))
        resource = timer.finish(); admission = self.admission(records, evaluations, runtime, resource, steps, transitions, rollouts)
        decision = "admit-root-20261291" if admission["admitted"] else "park-exp685-at-resource-admission"
        result = {"schema_version": "exp685-calibration-v1", "run_id": self.config["run_id"], "mode": self.mode, "root": self.root, "status": "completed", "decision": decision, "source": self.source_identity(), "records": records, "evaluations": evaluations, "candidate_runtime": runtime, "resource": resource, "admission": admission, "limitations": self.limitations(True)}
        write_json(root_dir / "root-summary.json", result); write_json(self.output / "calibration.json", result); return result

    def load_admission(self) -> dict[str, Any]:
        path = self.output / "calibration.json"
        if not path.is_file(): raise RuntimeError("formal requires root20261290 admission")
        result = json.loads(path.read_text())
        if result["decision"] != "admit-root-20261291" or not result["admission"]["admitted"] or result["source"]["checkpoint_sha256"] != sha256(self.checkpoint):
            raise RuntimeError("formal requires matching admitted root20261290")
        return result

    def _require_predecessor(self) -> None:
        if self.mode != "confirmation": return
        roots = list(map(int, self.config["roots"]["confirmations"])); index = roots.index(int(self.root))
        predecessor = int(self.config["roots"]["formal"]) if index == 0 else roots[index - 1]
        path = self.output / ("formal" if index == 0 else "confirmation") / f"root-{predecessor}" / "root-summary.json"
        expected = "advance-to-confirmations" if index == 0 else "confirmation-pass"
        if not path.is_file() or json.loads(path.read_text())["decision"] != expected:
            raise RuntimeError(f"confirmation requires {expected} at root {predecessor}")

    def _evaluate_width(self, models: dict[str, AttractorCell], width: int, transitions: int, rollouts: int, arms: tuple[str, ...]) -> dict[str, Any]:
        cases = self.frozen_cases([width], transitions, rollouts)
        return self.evaluate_all(models, cases, [width], arms), cases

    def science(self) -> dict[str, Any]:
        admission = self.load_admission(); self._require_predecessor(); root_dir = self.open_root(); timer = Timer()
        models, records = self.train_all(root_dir, int(self.config["training"]["steps"])); transitions = int(self.config["evaluation"]["transition_cases_per_width"]); rollouts = int(self.config["evaluation"]["rollout_cases_per_width"])
        primary_eval, cases128 = self._evaluate_width(models, 128, transitions, rollouts, ("validity_constrained",))
        runtime = candidate_runtime(models["validity_constrained"], cases128["test"]["128"]["transition"], int(self.config["calibration"]["candidate_runtime_repeats"]), float(self.config["resource_policy"]["incumbent_candidate_wall_seconds"]))
        p128 = primary_eval["validity_constrained"]["test"]["128"]
        gates: dict[str, bool] = {
            "w128_exact": p128["rollout"]["exact_accuracy"] >= float(self.config["gates"]["minimum_w128_primary_rollout"]),
            "w128_transition": p128["transition"]["exact_accuracy"] >= float(self.config["gates"]["minimum_w128_transition"]),
            "zero_invalid": p128["rollout"]["invalids"] <= int(self.config["gates"]["maximum_invalid_rollouts"]),
            "candidate_runtime": float(runtime["candidate_projected_wall_seconds"]) * float(admission["admission"]["headroom"]) <= float(self.config["gates"]["maximum_candidate_inference_wall_seconds"]),
        }
        evaluations = primary_eval; stage = "w128-primary"
        if all(gates.values()):
            controls_eval, _ = self._evaluate_width(models, 128, transitions, rollouts, ARM_ORDER[1:])
            evaluations.update(controls_eval)
            primary_validation = evaluations["validity_constrained"]["validation"]["128"]["validity"]["brier"]
            ce = evaluations["ce_only"]["test"]["128"]["rollout"]["exact_accuracy"]; frozen = evaluations["frozen_source"]["test"]["128"]["rollout"]["exact_accuracy"]; randomized = evaluations["randomized_control"]["test"]["128"]["rollout"]["exact_accuracy"]
            gates.update({
                "frozen_retention": p128["rollout"]["exact_accuracy"] >= frozen - float(self.config["gates"]["maximum_w128_loss_to_frozen"]),
                "ce_retention": p128["rollout"]["exact_accuracy"] >= ce - float(self.config["gates"]["maximum_w128_loss_to_ce"]),
                "constraint_vs_ce_calibration": primary_validation < evaluations["ce_only"]["validation"]["128"]["validity"]["brier"],
                "constraint_vs_shuffled_calibration": primary_validation < evaluations["shuffled_modulus"]["validation"]["128"]["validity"]["brier"],
                "constraint_gradient": records["validity_constrained"]["training"]["constraint_nonzero_steps"] == int(self.config["training"]["steps"]),
                "trained_dependence": p128["rollout"]["exact_accuracy"] - randomized >= float(self.config["gates"]["minimum_trained_weight_drop_w128"]),
                "reload_identity": all(record["reload_exact"] and record["inference_identity"]["state_dict_reload_exact"] for record in records.values()),
            })
            stage = "w128-controls"
        if all(gates.values()):
            primary256, _ = self._evaluate_width(models, 256, transitions, rollouts, ("validity_constrained",));
            for arm, value in primary256.items(): evaluations.setdefault(arm, {}).update(value)
            score = evaluations["validity_constrained"]["test"]["256"]["rollout"]["exact_accuracy"]
            gates["w256_absolute"] = score >= float(self.config["gates"]["minimum_w256_primary_rollout"]); stage = "w256-primary"
        if all(gates.values()):
            controls256, _ = self._evaluate_width(models, 256, transitions, rollouts, ARM_ORDER[1:])
            for arm, value in controls256.items(): evaluations.setdefault(arm, {}).update(value)
            score = evaluations["validity_constrained"]["test"]["256"]["rollout"]["exact_accuracy"]
            gates["w256_ce_gain"] = score - evaluations["ce_only"]["test"]["256"]["rollout"]["exact_accuracy"] >= float(self.config["gates"]["minimum_w256_gain_over_ce"])
            gates["w256_shuffled_gain"] = score - evaluations["shuffled_modulus"]["test"]["256"]["rollout"]["exact_accuracy"] >= float(self.config["gates"]["minimum_w256_gain_over_shuffled"]); stage = "w256-controls"
        passed = bool(gates) and all(gates.values())
        decision = ("advance-to-confirmations" if self.mode == "formal" else "confirmation-pass") if passed else "reject-exp685-validity-constrained-attractor"
        result = {"schema_version": "exp685-terminal-v1", "run_id": self.config["run_id"], "mode": self.mode, "root": self.root, "status": "completed", "stage": stage, "decision": decision, "source": self.source_identity(), "records": records, "evaluations": evaluations, "gates": gates, "candidate_runtime": runtime, "dynamic_admission": admission["admission"], "resource": timer.finish(), "limitations": self.limitations(False)}
        write_json(root_dir / "root-summary.json", result); write_json(self.output / "terminal-summary.json", result); return result

    @staticmethod
    def limitations(calibration: bool) -> list[str]:
        return [
            "Calibration accuracy is resource/integrity evidence, not scientific evidence." if calibration else "Only fresh generated validation/test identities are measured; no protected benchmark is opened.",
            "The lexicographic comparator and exact bigint arithmetic are confined to offline training loss, diagnostics and tests; formal inference is the unchanged EXP665 AttractorCell/Anderson route.",
            "No inference parameter, call, validity filter, selector or repair is added.",
            "No checked, independent, official/private, candidate, submission, promotion or completion surface is opened.",
        ]


def log_result(experiment: Experiment, result: dict[str, Any]) -> dict[str, Any]:
    if experiment.mode == "smoke": return result
    experiment_id = configure_mlflow(experiment.config["tracking"]["experiment_name"])
    with mlflow.start_run(run_name=f"{experiment.config['run_id']}-{experiment.mode}-{experiment.root}") as active:
        mlflow.set_tags({"canonical_run_id": experiment.config["run_id"], "mode": experiment.mode, "root": str(experiment.root), "decision": result["decision"], "repository_commit": git_output("rev-parse", "HEAD")})
        mlflow.log_dict(result, f"{experiment.mode}-result.json"); run_id = active.info.run_id
    record = {"experiment_id": experiment_id, "run_id": run_id, "tracking_uri": tracking_uri()}
    write_json(experiment.output / f"{experiment.mode}-mlflow.json", record); result["mlflow"] = record
    summary = experiment._root_dir() / "root-summary.json"; write_json(summary, result)
    if experiment.mode == "calibration": write_json(experiment.output / "calibration.json", result)
    else: write_json(experiment.output / "terminal-summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH); parser.add_argument("--output", type=Path, default=ROOT / ".scratch/exp-685-validity-constrained-attractor")
    parser.add_argument("--mode", choices=("smoke", "calibration", "formal", "confirmation"), required=True); parser.add_argument("--root", type=int)
    args = parser.parse_args(); experiment = Experiment(args.config, args.output, args.mode, args.root)
    result = experiment.smoke() if args.mode == "smoke" else experiment.calibration() if args.mode == "calibration" else experiment.science()
    result = log_result(experiment, result)
    print(json.dumps({"mode": args.mode, "root": result.get("root"), "status": result["status"], "decision": result.get("decision"), "gates": result.get("gates"), "admission": result.get("admission", {}).get("checks")}, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
