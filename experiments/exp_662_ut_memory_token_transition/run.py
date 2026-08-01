"""Preregister, calibrate, and run EXP-662's UT memory-token falsifier."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import psutil
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file, save_file
from sympy import primepi


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.exp_662_ut_memory_token_transition.model import UTMemoryTransition  # noqa: E402
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402

CONFIG_PATH = ROOT / "configs/exp_662_ut_memory_token_transition.yaml"
ROUTES = (
    "memory_0", "memory_4", "memory_8", "memory_16", "memory_32", "memory_64",
    "wider_no_memory", "decoded_recurrent", "fixed_depth_memory_16", "default_bias_memory_16",
)
SWEEP_ROUTES = ROUTES[:6]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_exp300():
    previous_path, previous_model = sys.path.copy(), sys.modules.get("model")
    had_model = "model" in sys.modules
    try:
        return load_module("exp300_for_exp662", ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py")
    finally:
        sys.path[:] = previous_path
        if had_model:
            sys.modules["model"] = previous_model
        else:
            sys.modules.pop("model", None)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def git_output(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=120).stdout.strip()


def cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_utime + usage.ru_stime)


def cgroup_cpu_seconds() -> float | None:
    path = Path("/sys/fs/cgroup/cpu.stat")
    if not path.is_file():
        return None
    try:
        values = dict(line.split() for line in path.read_text(encoding="utf-8").splitlines())
        return int(values["usage_usec"]) / 1_000_000.0
    except (OSError, KeyError, ValueError):
        return None


def peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


class Timer:
    def __init__(self) -> None:
        self.wall, self.cpu, self.cgroup = time.perf_counter(), cpu_seconds(), cgroup_cpu_seconds()

    def finish(self) -> dict[str, float | int | None]:
        cg = cgroup_cpu_seconds()
        return {
            "wall_seconds": time.perf_counter() - self.wall,
            "process_cpu_seconds": cpu_seconds() - self.cpu,
            "cgroup_cpu_seconds": None if cg is None or self.cgroup is None else cg - self.cgroup,
            "peak_rss_bytes": peak_rss_bytes(),
        }


def bits(values: list[int], width: int) -> torch.Tensor:
    return torch.tensor([[(value >> position) & 1 for position in range(width)] for value in values], dtype=torch.float32)


def features(state: list[int], multiplicand: list[int], modulus: list[int], input_bit: list[int], width: int) -> torch.Tensor:
    batch = len(state)
    plane = torch.tensor(input_bit, dtype=torch.float32)[:, None].expand(batch, width)
    lsb, msb = torch.zeros(batch, width), torch.zeros(batch, width)
    lsb[:, 0], msb[:, -1] = 1.0, 1.0
    return torch.stack((bits(state, width), bits(multiplicand, width), bits(modulus, width), plane, lsb, msb), 1).mul(2).sub(1)


def decode(rows: torch.Tensor) -> list[int]:
    return [sum(int(bit) << position for position, bit in enumerate(row)) for row in rows.tolist()]


def route_spec(route: str) -> dict[str, Any]:
    if route.startswith("memory_"):
        return {"architecture": "ut_memory", "memory_count": int(route.split("_")[1]), "use_act": True, "halt": "deep"}
    if route == "wider_no_memory":
        return {"architecture": route, "memory_count": 0, "use_act": True, "halt": "deep"}
    if route == "decoded_recurrent":
        return {"architecture": route, "memory_count": 0, "use_act": False, "halt": "deep"}
    if route == "fixed_depth_memory_16":
        return {"architecture": "ut_memory", "memory_count": 16, "use_act": False, "halt": "deep"}
    if route == "default_bias_memory_16":
        return {"architecture": "ut_memory", "memory_count": 16, "use_act": True, "halt": "default"}
    raise ValueError(route)


def make_split(config: dict[str, Any], exp300, width: int, root: int) -> dict[str, list[int]]:
    data = config["data"]
    low_bits = max(3, width - int(data["prime_bit_span"]) + 1)
    total = sum(int(data[key]) for key in ("train_prime_identities", "validation_prime_identities", "test_prime_identities"))
    if width <= 32:
        available = int(primepi((1 << width) - 1) - primepi((1 << (low_bits - 1)) - 1))
        if total > available:
            raise ValueError(f"width {width} has only {available} primes for {total} requested identities")
    local = {"data": {"prime_bits_min": low_bits, "prime_bits_max": width, "train_prime_identities": int(data["train_prime_identities"]), "test_prime_identities": total - int(data["train_prime_identities"])}}
    raw = exp300.bigint_prime_split(local, root + int(data["split_seed_offset"]) + width)
    validation_count = int(data["validation_prime_identities"])
    return {"train": raw["train"], "validation": raw["test"][:validation_count], "test": raw["test"][validation_count:]}


def frozen_transitions(exp300, primes: list[int], width: int, seed: int, count: int) -> dict[str, Any]:
    values = exp300.BigIntSampler(primes, seed).transitions(count)
    state, multiplicand, modulus, input_bit, targets, _ = values
    return {"state": state, "multiplicand": multiplicand, "modulus": modulus, "input_bit": input_bit, "targets": targets, "width": width}


def frozen_rollouts(exp300, primes: list[int], width: int, seed: int, count: int) -> dict[str, Any]:
    left, right, modulus = exp300.BigIntSampler(primes, seed).pairs(count)
    return {"left": left, "right": right, "modulus": modulus, "width": width}


def build_model(config: dict[str, Any], route: str) -> UTMemoryTransition:
    model, spec = config["model"], route_spec(route)
    halt_bias = model["default_halt_bias"] if spec["halt"] == "default" else model["deep_start_bias"]
    return UTMemoryTransition(
        hidden=int(model["hidden"]), heads=int(model["heads"]), ff_hidden=int(model["ff_hidden"]),
        wide_hidden=int(model["wide_hidden"]), max_memory_tokens=int(model["max_memory_tokens"]),
        max_steps=int(model["max_steps"]), act_epsilon=float(model["act_epsilon"]),
        halt_bias=float(halt_bias), architecture=str(spec["architecture"]),
    )


def transition_loss(output, targets: list[int], width: int, config: dict[str, Any], step: int) -> tuple[torch.Tensor, dict[str, float]]:
    target = bits(targets, width)
    per_bit = F.binary_cross_entropy_with_logits(output.logits, target, reduction="none")
    mean, worst = per_bit.mean(), per_bit.mean(0).max()
    ponder = output.ponder_cost * float(config["training"]["ponder_weight"]) if step > int(config["training"]["ponder_warmup_steps"]) else output.ponder_cost * 0
    loss = mean + float(config["training"]["worst_bit_weight"]) * worst + ponder
    return loss, {"loss": float(loss.detach()), "mean_bce": float(mean.detach()), "worst_bit_bce": float(worst.detach()), "batch_exact": float((output.logits.ge(0) == target.bool()).all(1).float().mean()), "ponder": float(output.ponder_cost.detach())}


@torch.inference_mode()
def evaluate_transitions(model: UTMemoryTransition, cases: dict[str, Any], route: str, *, ablation: str = "none", max_steps: int | None = None) -> dict[str, Any]:
    spec, width = route_spec(route), int(cases["width"])
    timer = Timer(); correct = bits_correct = invalid = calls = flops = 0; halt = []
    for start in range(0, len(cases["targets"]), 64):
        stop = start + 64
        output = model(features(cases["state"][start:stop], cases["multiplicand"][start:stop], cases["modulus"][start:stop], cases["input_bit"][start:stop], width), memory_count=spec["memory_count"], use_act=spec["use_act"], max_steps=max_steps, ablation=ablation, generator=torch.Generator().manual_seed(662))
        predicted, target = output.logits.ge(0), bits(cases["targets"][start:stop], width).bool()
        values = decode(predicted)
        correct += int((predicted == target).all(1).sum()); bits_correct += int((predicted == target).sum())
        invalid += sum(value >= modulus for value, modulus in zip(values, cases["modulus"][start:stop], strict=True))
        calls += output.block_calls * predicted.shape[0]; flops += output.analytical_flops
        halt.extend(output.halt_steps.flatten().tolist())
    count = len(cases["targets"])
    return {"cases": count, "exact_accuracy": correct / count, "bit_accuracy": bits_correct / (count * width), "invalid_residue_rate": invalid / count, "mean_halt_step": float(np.mean(halt)), "halt_histogram": {str(step): halt.count(step) for step in sorted(set(halt))}, "block_calls": calls, "analytical_flops": flops, "timing": timer.finish()}


@torch.inference_mode()
def evaluate_rollouts(model: UTMemoryTransition, cases: dict[str, Any], route: str, *, ablation: str = "none", max_steps: int | None = None) -> dict[str, Any]:
    spec, width = route_spec(route), int(cases["width"])
    timer = Timer(); size = len(cases["left"]); state = [0] * size; teacher = [0] * size
    all_prefix, transition_correct, invalid, first_error = [True] * size, 0, [False] * size, [None] * size
    calls = flops = 0
    for position in range(width):
        input_bit = [(value >> (width - 1 - position)) & 1 for value in cases["left"]]
        output = model(features(state, cases["right"], cases["modulus"], input_bit, width), memory_count=spec["memory_count"], use_act=spec["use_act"], max_steps=max_steps, ablation=ablation, generator=torch.Generator().manual_seed(662 + position))
        state = decode(output.logits.ge(0))
        teacher = [(2 * value + bit * other) % prime for value, bit, other, prime in zip(teacher, input_bit, cases["right"], cases["modulus"], strict=True)]
        exact = [actual == expected for actual, expected in zip(state, teacher, strict=True)]
        transition_correct += sum(exact)
        for index, ok in enumerate(exact):
            all_prefix[index] = all_prefix[index] and ok
            invalid[index] = invalid[index] or state[index] >= cases["modulus"][index]
            if not ok and first_error[index] is None:
                first_error[index] = position
        calls += output.block_calls * size; flops += output.analytical_flops
    expected = [(left * right) % prime for left, right, prime in zip(cases["left"], cases["right"], cases["modulus"], strict=True)]
    positions = [position for position in first_error if position is not None]
    return {"cases": size, "exact_accuracy": sum(actual == truth for actual, truth in zip(state, expected, strict=True)) / size, "all_prefix_exact_accuracy": sum(all_prefix) / size, "transition_exact_accuracy": transition_correct / (size * width), "any_invalid_residue_rate": sum(invalid) / size, "first_error_mean_position": None if not positions else sum(positions) / len(positions), "block_calls": calls, "analytical_flops": flops, "timing": timer.finish()}


def gate_summary(config: dict[str, Any], selected: str, routes: dict[str, Any], diagnostics: dict[str, Any], separation: bool) -> dict[str, Any]:
    gates = config["gates"]; selected_eval = routes[selected]["test"]
    controls = ("memory_0", "wider_no_memory", "decoded_recurrent")
    results = {
        "selected_w16": selected_eval["16"]["rollout"]["exact_accuracy"] >= float(gates["minimum_selected_w16_rollout"]),
        "selected_w24": selected_eval["24"]["rollout"]["exact_accuracy"] >= float(gates["minimum_selected_w24_rollout"]),
        "selected_w32": selected_eval["32"]["rollout"]["exact_accuracy"] >= float(gates["minimum_selected_w32_rollout"]),
        "transfer_gain_each_control": all(selected_eval[width]["rollout"]["exact_accuracy"] - routes[control]["test"][width]["rollout"]["exact_accuracy"] >= float(gates["minimum_transfer_gain_over_each_control"]) for width in ("24", "32") for control in controls),
        "trained_weight_dependence": selected_eval["16"]["rollout"]["exact_accuracy"] - diagnostics["randomized_weights"]["exact_accuracy"] >= float(gates["minimum_trained_weight_drop_w16"]),
        "memory_causality": all(selected_eval["16"]["rollout"]["exact_accuracy"] - diagnostics["memory_interventions"][name]["exact_accuracy"] >= float(gates["minimum_memory_causal_drop_w16"]) for name in config["evaluation"]["memory_interventions"]),
        "representation_causality": all(selected_eval["16"]["rollout"]["exact_accuracy"] - diagnostics["representation_interventions"][name]["exact_accuracy"] >= float(config["confirmation"]["minimum_component_causal_drop_w16"]) for name in config["evaluation"]["representation_interventions"]),
        "valid_residues": max(value["rollout"]["any_invalid_residue_rate"] for value in selected_eval.values()) <= float(gates["maximum_invalid_rate"]),
        "parameter_match": max(result["parameter_count"] for result in routes.values()) / min(result["parameter_count"] for result in routes.values()) <= float(gates["maximum_parameter_ratio"]),
        "checkpoint_reload": all(result["checkpoint_reload_exact"] for result in routes.values()),
        "validation_test_separation": separation,
        "matched_examples_updates_calls": len({(result["training"]["examples"], result["training"]["updates"], result["training"]["block_calls"]) for result in routes.values()}) == 1,
    }
    return {"thresholds": gates, "results": results, "passed": all(results.values())}


class Experiment:
    def __init__(self, config_path: Path, output: Path, mode: str, root: int) -> None:
        self.config_path, self.output, self.mode, self.root = config_path.resolve(), output.resolve(), mode, int(root)
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8")); self.output.mkdir(parents=True, exist_ok=True)
        self.exp300 = load_exp300(); torch.set_num_threads(int(self.config["training"]["torch_threads"])); torch.use_deterministic_algorithms(True)
        widths = sorted(set(self.config["data"]["training_widths"] + self.config["data"]["evaluation_widths"]))
        self.splits = {width: make_split(self.config, self.exp300, width, self.root) for width in widths}

    def source(self) -> dict[str, Any]:
        paths = {"config": self.config_path, "runner": Path(__file__), "model": Path(__file__).with_name("model.py"), "tests": ROOT / "tests/test_exp_662_ut_memory_token_transition.py", "mechanism": ROOT / self.config["mechanism_map"], "generator": ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py", "paper": ROOT / "research/papers/arxiv-2604.21999.pdf"}
        return {"repository_commit": git_output("rev-parse", "HEAD"), "repository_branch": git_output("branch", "--show-current"), "files": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for name, path in paths.items()}}

    def cases(self, split: str, transition_count: int, rollout_count: int) -> dict[str, Any]:
        offset = int(self.config["data"]["validation_pair_seed_offset" if split == "validation" else "test_pair_seed_offset"])
        return {str(width): {"transition": frozen_transitions(self.exp300, self.splits[width][split], width, self.root + offset + width, transition_count), "rollout": frozen_rollouts(self.exp300, self.splits[width][split], width, self.root + offset + 1000 + width, rollout_count)} for width in self.config["data"]["evaluation_widths"]}

    def train_route(self, route: str, steps: int) -> dict[str, Any]:
        random.seed(self.root + 662); np.random.seed(self.root + 662); torch.manual_seed(self.root + 662)
        model, spec = build_model(self.config, route), route_spec(route); initial_parameters = model.parameter_count()
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(self.config["training"]["learning_rate"]), weight_decay=float(self.config["training"]["weight_decay"]))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=float(self.config["training"]["minimum_learning_rate"]))
        samplers = {width: self.exp300.BigIntSampler(self.splits[width]["train"], self.root + int(self.config["data"]["training_pair_seed_offset"]) + width) for width in self.config["data"]["training_widths"]}
        history = []; calls = flops = 0; timer = Timer(); batch_size = int(self.config["training"]["batch_size"])
        for step in range(1, steps + 1):
            width = int(self.config["data"]["training_widths"][(step - 1) % len(self.config["data"]["training_widths"])])
            state, multiplicand, modulus, input_bit, targets, _ = samplers[width].transitions(batch_size)
            tick = Timer(); output = model(features(state, multiplicand, modulus, input_bit, width), memory_count=spec["memory_count"], use_act=spec["use_act"])
            loss, metrics = transition_loss(output, targets, width, self.config, step)
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), float(self.config["training"]["gradient_clip"])); optimizer.step(); scheduler.step()
            calls += output.block_calls * batch_size; flops += output.analytical_flops
            history.append({"step": step, "width": width, **metrics, "timing": tick.finish()})
        route_dir = self.output / self.mode / f"root-{self.root}" / route; route_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = route_dir / "model.safetensors"; save_file({key: value.detach().contiguous() for key, value in model.state_dict().items()}, checkpoint)
        reloaded = build_model(self.config, route); reloaded.load_state_dict(load_file(checkpoint)); reload_exact = all(torch.equal(left, right) for left, right in zip(model.state_dict().values(), reloaded.state_dict().values(), strict=True))
        return {"model": model, "parameter_count": initial_parameters, "checkpoint": {"path": str(checkpoint.relative_to(ROOT)), "sha256": sha256(checkpoint)}, "checkpoint_reload_exact": reload_exact, "training": {"updates": steps, "examples": steps * batch_size, "block_calls": calls, "analytical_flops": flops, "history": history, "timing": timer.finish()}}

    def preregister(self) -> dict[str, Any]:
        split_hashes = {str(width): {name: json_hash(values) for name, values in split.items()} for width, split in self.splits.items()}
        fixtures = {"manifest_path": "../t054/fixture-manifest.json", "manifest_sha256": "cd57ff0bb21c616bf91bb0245ca854fa71968b3e43754c24065548394ec3552d", "entries_tsv_sha256": "d3f7eea82456c0c4f5c967d54cc6e2afa0142eb9dd41e9e1afefabbe9ba0c969", "file_count": 268, "total_bytes": 27086830, "ownership_proof": "authenticated by parent T054 receipt; regular non-symlink single-link distinct-inode copies with reflink disabled and unchanged source"}
        result = {"run_id": self.config["run_id"], "mode": "preregister", "root": self.root, "source": self.source(), "split_hashes": split_hashes, "fixture_identity": fixtures, "route_order": list(ROUTES), "selection": {"split": "validation", "order": ["rollout_exact_accuracy", "transition_exact_accuracy", "negative_memory_count"]}, "test_access_before_selection": False}
        write_json(self.output / f"preregistration-root-{self.root}.json", result); return result

    def run(self) -> dict[str, Any]:
        calibration = self.mode == "calibration"; steps = int(self.config["calibration"]["representative_updates_per_route"] if calibration else self.config["training"]["steps_per_arm"])
        validation = self.cases("validation", int(self.config["calibration"]["transition_cases_per_width"] if calibration else self.config["evaluation"]["validation_transition_cases"]), int(self.config["calibration"]["validation_rollout_cases"] if calibration else self.config["evaluation"]["validation_rollout_cases"]))
        test = self.cases("test", int(self.config["calibration"]["transition_cases_per_width"] if calibration else self.config["evaluation"]["transition_cases_per_width"]), int(self.config["calibration"]["rollout_cases_per_width"] if calibration else self.config["evaluation"]["rollout_cases_per_width"]))
        prereg = self.preregister(); started = Timer(); routes: dict[str, Any] = {}
        for route in ROUTES:
            trained = self.train_route(route, steps); model = trained.pop("model")
            selection_eval = {"transition": evaluate_transitions(model, validation["16"]["transition"], route), "rollout": evaluate_rollouts(model, validation["16"]["rollout"], route)} if route in SWEEP_ROUTES else None
            routes[route] = {**trained, "validation_selection": selection_eval, "_model": model}
        selected = max(SWEEP_ROUTES, key=lambda route: (routes[route]["validation_selection"]["rollout"]["exact_accuracy"], routes[route]["validation_selection"]["transition"]["exact_accuracy"], -route_spec(route)["memory_count"]))
        for route, result in routes.items():
            model = result.pop("_model"); result["test"] = {width: {"transition": evaluate_transitions(model, value["transition"], route), "rollout": evaluate_rollouts(model, value["rollout"], route)} for width, value in test.items()}; result["_model"] = model
        selected_model = routes[selected]["_model"]
        diagnostics = {
            "memory_interventions": {name: evaluate_rollouts(selected_model, test["16"]["rollout"], selected, ablation=name) for name in self.config["evaluation"]["memory_interventions"]},
            "representation_interventions": {name: evaluate_rollouts(selected_model, test["16"]["rollout"], selected, ablation=name) for name in self.config["evaluation"]["representation_interventions"]},
            "fixed_depths": {str(depth): evaluate_rollouts(selected_model, test["16"]["rollout"], "fixed_depth_memory_16", max_steps=int(depth)) for depth in self.config["evaluation"]["fixed_depths"]},
        }
        torch.manual_seed(self.root + 1662); randomized = build_model(self.config, selected)
        diagnostics["randomized_weights"] = evaluate_rollouts(randomized, test["16"]["rollout"], selected)
        separation = all(set(self.splits[width]["validation"]).isdisjoint(self.splits[width]["test"]) for width in self.splits)
        for result in routes.values(): result.pop("_model")
        gate = gate_summary(self.config, selected, routes, diagnostics, separation)
        if calibration:
            update_wall = [item["timing"]["wall_seconds"] for result in routes.values() for item in result["training"]["history"]]
            total_wall = sum(result["training"]["timing"]["wall_seconds"] for result in routes.values())
            fixed = max(0.0, total_wall - sum(update_wall)); projected = fixed + np.mean(update_wall) * int(self.config["training"]["steps_per_arm"]) * len(ROUTES)
            variability = max(update_wall) / max(np.mean(update_wall), 1e-9); policy = self.config["resource_policy"]
            headroom = min(float(policy["variability_headroom_ceiling"]), max(float(policy["variability_headroom_floor"]), variability))
            admission = {"complete_routes": set(routes) == set(ROUTES), "projected_wall_seconds": projected, "variability_ratio": variability, "headroom": headroom, "soft_guard_wall_seconds": math.ceil(projected * headroom * float(policy["soft_guard_multiplier"])), "hard_guard_wall_seconds": math.ceil(projected * headroom * float(policy["hard_guard_multiplier"])), "peak_rss_bytes": peak_rss_bytes()}
            admission["admitted"] = admission["complete_routes"] and admission["peak_rss_bytes"] <= int(policy["admission_maximum_peak_rss_bytes"])
        else:
            admission = None
        result = {"run_id": self.config["run_id"], "mode": self.mode, "root": self.root, "status": "completed", "source": prereg["source"], "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "mlflow": mlflow.__version__, "cpu": platform.processor(), "logical_cores": psutil.cpu_count()}, "selection": {"selected_route": selected, "validation_only": True, "scores": {route: routes[route]["validation_selection"] for route in SWEEP_ROUTES}}, "routes": routes, "diagnostics": diagnostics, "gates": gate, "admission": admission, "decision": "advance-to-confirmation" if gate["passed"] else "reject-current-ut-memory-token-transition", "resource": started.finish(), "limitations": ["Exact Python arithmetic is confined to offline label generation and evaluation.", "Fresh generated identities are disjoint by split; challenge answers were not read.", "The bounded implementation tests the source mechanism, not the paper's full language-model scale."]}
        write_json(self.output / ("calibration.json" if calibration else f"root-{self.root}-summary.json"), result)
        return result


def log_formal(result: dict[str, Any], config: dict[str, Any], output: Path) -> None:
    configure_mlflow(config["tracking"]["experiment_name"])
    with mlflow.start_run(run_name=f"{config['run_id']}-root-{result['root']}") as run:
        mlflow.set_tags({"canonical_run_id": config["run_id"], "root": str(result["root"]), "decision": result["decision"]})
        mlflow.log_params({"selected_route": result["selection"]["selected_route"], "config_sha256": result["source"]["files"]["config"]["sha256"]})
        mlflow.log_metric("gate_passed", float(result["gates"]["passed"])); mlflow.log_metric("wall_seconds", result["resource"]["wall_seconds"])
        mlflow.log_artifact(str(output / f"root-{result['root']}-summary.json"), artifact_path="canonical")
        result["mlflow"] = {"tracking_uri": tracking_uri(), "run_id": run.info.run_id}
        write_json(output / f"root-{result['root']}-summary.json", result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=CONFIG_PATH); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--mode", choices=("preregister", "calibration", "formal"), required=True); parser.add_argument("--root", type=int)
    args = parser.parse_args(); config = yaml.safe_load(args.config.read_text(encoding="utf-8")); default_root = config["roots"]["calibration" if args.mode in {"preregister", "calibration"} else "viability"]; root = int(default_root if args.root is None else args.root)
    experiment = Experiment(args.config, args.output, args.mode, root)
    result = experiment.preregister() if args.mode == "preregister" else experiment.run()
    if args.mode == "formal": log_formal(result, config, args.output)
    print(json.dumps({"status": result.get("status", "preregistered"), "mode": args.mode, "root": root, "decision": result.get("decision"), "admitted": (result.get("admission") or {}).get("admitted")})); return 0


if __name__ == "__main__":
    raise SystemExit(main())
