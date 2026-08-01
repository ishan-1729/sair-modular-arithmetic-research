"""Preregister, calibrate, and execute EXP-664 SPRM boundary perturbation."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import math
import multiprocessing
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
from experiments.exp_664_sprm_boundary_perturbation.model import SPRMCell  # noqa: E402
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402

CONFIG_PATH = ROOT / "configs/exp_664_sprm_boundary_perturbation.yaml"
ROUTES = ("frozen_source", "sprm_boundary", "clean_tbptt", "wrong_boundary", "unscaled_boundary")
TRAINED_ROUTES = ROUTES[1:]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def load_exp300():
    previous_path, previous_model = sys.path.copy(), sys.modules.get("model"); had_model = "model" in sys.modules
    try: return load_module("exp300_for_exp664", ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py")
    finally:
        sys.path[:] = previous_path
        if had_model: sys.modules["model"] = previous_model
        else: sys.modules.pop("model", None)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def git_output(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=120).stdout.strip()


def cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF); return float(usage.ru_utime + usage.ru_stime)


def cgroup_cpu_seconds() -> float | None:
    path = Path("/sys/fs/cgroup/cpu.stat")
    if not path.is_file(): return None
    try:
        values = dict(line.split() for line in path.read_text(encoding="utf-8").splitlines()); return int(values["usage_usec"]) / 1_000_000.0
    except (OSError, KeyError, ValueError): return None


def peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


class Timer:
    def __init__(self) -> None:
        self.wall, self.cpu, self.cgroup = time.perf_counter(), cpu_seconds(), cgroup_cpu_seconds()

    def finish(self) -> dict[str, float | int | None]:
        current = cgroup_cpu_seconds()
        return {"wall_seconds": time.perf_counter() - self.wall, "process_cpu_seconds": cpu_seconds() - self.cpu, "cgroup_cpu_seconds": None if current is None or self.cgroup is None else current - self.cgroup, "peak_rss_bytes": peak_rss_bytes()}


def deterministic_makespan(durations: list[float], workers: int) -> float:
    slots = [0.0] * workers
    for duration in durations:
        index = min(range(workers), key=lambda item: (slots[item], item)); slots[index] += float(duration)
    return max(slots, default=0.0)


def aggregate_resource(parent: dict[str, Any], workers: list[dict[str, Any]], process_count: int) -> dict[str, Any]:
    return {**parent, "aggregate_process_tree_cpu_seconds": float(parent["process_cpu_seconds"]) + sum(float(item["process_cpu_seconds"]) for item in workers), "aggregate_peak_rss_upper_bound_bytes": int(parent["peak_rss_bytes"]) + sum(sorted((int(item["peak_rss_bytes"]) for item in workers), reverse=True)[:process_count]), "worker_resources": workers}


def bits(values: list[int], width: int) -> torch.Tensor:
    return torch.tensor([[(value >> position) & 1 for position in range(width)] for value in values], dtype=torch.float32)


def decode(rows: torch.Tensor) -> list[int]:
    return [sum(int(bit) << position for position, bit in enumerate(row)) for row in rows.tolist()]


def features(state: list[int], multiplicand: list[int], modulus: list[int], input_bit: list[int], width: int) -> torch.Tensor:
    batch = len(state); plane = torch.tensor(input_bit, dtype=torch.float32)[:, None].expand(batch, width)
    lsb, msb = torch.zeros(batch, width), torch.zeros(batch, width); lsb[:, 0], msb[:, -1] = 1.0, 1.0
    return torch.stack((bits(state, width), bits(multiplicand, width), bits(modulus, width), plane, lsb, msb), 1).mul(2).sub(1)


def make_split(config: dict[str, Any], exp300, width: int, root: int) -> dict[str, list[int]]:
    data = config["data"]; low_bits = width - int(data["prime_bit_span"]) + 1
    total = sum(int(data[key]) for key in ("train_prime_identities", "validation_prime_identities", "test_prime_identities"))
    if width <= 32:
        available = int(primepi((1 << width) - 1) - primepi((1 << (low_bits - 1)) - 1))
        if total > available: raise ValueError("insufficient unique primes")
    local = {"data": {"prime_bits_min": low_bits, "prime_bits_max": width, "train_prime_identities": int(data["train_prime_identities"]), "test_prime_identities": total - int(data["train_prime_identities"])}}
    raw = exp300.bigint_prime_split(local, root + int(data["split_seed_offset"]) + width); validation = int(data["validation_prime_identities"])
    return {"train": raw["train"], "validation": raw["test"][:validation], "test": raw["test"][validation:]}


def transition_cases(exp300, primes: list[int], width: int, seed: int, count: int) -> dict[str, Any]:
    state, multiplicand, modulus, input_bit, target, quotient = exp300.BigIntSampler(primes, seed).transitions(count)
    return {"state": state, "multiplicand": multiplicand, "modulus": modulus, "input_bit": input_bit, "target": target, "quotient": quotient.tolist(), "width": width}


def rollout_cases(exp300, primes: list[int], width: int, seed: int, count: int) -> dict[str, Any]:
    left, right, modulus = exp300.BigIntSampler(primes, seed).pairs(count); return {"left": left, "right": right, "modulus": modulus, "width": width}


def build_model(config: dict[str, Any], source: Path, *, load_source: bool = True) -> SPRMCell:
    model = config["model"]
    result = SPRMCell(hidden=int(model["hidden"]), cycles=int(model["cycles"]), boundary_passes=int(model["boundary_passes"]), beta_start=float(model["beta_start"]), beta_end=float(model["beta_end"]), beta_steps=int(model["beta_steps"]))
    if load_source: result.load_source(load_file(source))
    return result


def objective(output, target: list[int], quotient: torch.Tensor, width: int, config: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
    target_bits = bits(target, width); losses = []
    for logits, qlogits in zip(output.window_logits, output.window_quotient_logits, strict=True):
        per_bit = F.binary_cross_entropy_with_logits(logits, target_bits, reduction="none")
        losses.append(per_bit.mean() + float(config["training"]["worst_bit_weight"]) * per_bit.mean(0).max() + float(config["training"]["quotient_weight"]) * F.cross_entropy(qlogits, quotient))
    loss = torch.stack(losses).mean(); final_bits = output.logits.ge(0)
    return loss, {"loss": float(loss.detach()), "batch_exact": float((final_bits == target_bits.bool()).all(1).float().mean()), "windows": float(len(losses))}


def train_route(config: dict[str, Any], exp300, source: Path, splits: dict[int, dict[str, list[int]]], root: int, route: str, steps: int, output: Path, mode: str) -> tuple[SPRMCell, dict[str, Any]]:
    random.seed(root + 664); np.random.seed(root + 664); torch.manual_seed(root + 664)
    route_dir = output / mode / f"root-{root}" / route; route_dir.mkdir(parents=True, exist_ok=True); checkpoint = route_dir / "model.safetensors"; cache_path = route_dir / "training-result.json"
    cache_key = json_hash({"source": sha256(source), "root": root, "route": route, "steps": steps, "model": config["model"], "training": config["training"], "splits": splits})
    if bool(config["resource_policy"]["cache_completed_roots"]) and checkpoint.is_file() and cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("cache_key") == cache_key and sha256(checkpoint) == cached["result"]["checkpoint"]["sha256"]:
            model = build_model(config, source); model.load_state_dict(load_file(checkpoint)); return model, {**cached["result"], "cached": True}
    model = build_model(config, source); timer = Timer(); history: list[dict[str, Any]] = []; examples = block_passes = flops = 0
    if route != "frozen_source":
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"])); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=float(config["training"]["minimum_learning_rate"]))
        samplers = {width: exp300.BigIntSampler(splits[width]["train"], root + int(config["data"]["training_seed_offset"]) + width) for width in config["data"]["training_widths"]}
        generator = torch.Generator().manual_seed(root + 66403); batch_size = int(config["training"]["batch_size"]); model.train()
        for step in range(1, steps + 1):
            width = int(config["training"]["schedule"][(step - 1) % len(config["training"]["schedule"])]); state, multiplicand, modulus, input_bit, target, quotient = samplers[width].transitions(batch_size); tick = Timer()
            result = model.training_forward(features(state, multiplicand, modulus, input_bit, width), route=route, generator=generator); loss, metrics = objective(result, target, quotient, width, config)
            optimizer.zero_grad(set_to_none=True); loss.backward(); gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip"]))); optimizer.step(); scheduler.step()
            examples += batch_size; block_passes += result.block_passes * batch_size; flops += result.analytical_flops
            event_mean = float(np.mean([float(item["noise_rms"]) for item in result.boundary_events])) if result.boundary_events else 0.0
            record = {"step": step, "width": width, "gradient_norm": gradient, "boundary_noise_rms": event_mean, **metrics, "timing": tick.finish()}; history.append(record)
            if step == 1 or step % int(config["training"]["log_every_steps"]) == 0 or step == steps: print(json.dumps({"route": route, **{key: value for key, value in record.items() if key != "timing"}}), flush=True)
    save_file({key: value.detach().contiguous() for key, value in model.state_dict().items()}, checkpoint); reloaded = build_model(config, source); reloaded.load_state_dict(load_file(checkpoint)); reload_exact = all(torch.equal(left, right) for left, right in zip(model.state_dict().values(), reloaded.state_dict().values(), strict=True))
    result_record = {"route": route, "cached": False, "parameters": model.parameter_count(), "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad), "checkpoint": {"path": str(checkpoint.relative_to(ROOT)), "sha256": sha256(checkpoint), "bytes": checkpoint.stat().st_size}, "checkpoint_reload_exact": reload_exact, "training": {"updates": 0 if route == "frozen_source" else steps, "examples": examples, "block_passes": block_passes, "analytical_flops": flops, "history": history, "timing": timer.finish()}}
    write_json(cache_path, {"cache_key": cache_key, "result": result_record}); return model, result_record


@torch.inference_mode()
def evaluate_transitions(model: SPRMCell, cases: dict[str, Any]) -> dict[str, Any]:
    timer = Timer(); width = int(cases["width"]); exact = bit_exact = invalid = calls = 0; predictions: list[int] = []
    for start in range(0, len(cases["target"]), 128):
        stop = start + 128; logits, _ = model(features(cases["state"][start:stop], cases["multiplicand"][start:stop], cases["modulus"][start:stop], cases["input_bit"][start:stop], width)); predicted, target = logits.ge(0), bits(cases["target"][start:stop], width).bool(); values = decode(predicted)
        predictions.extend(values); exact += int((predicted == target).all(1).sum()); bit_exact += int((predicted == target).sum()); invalid += sum(value >= prime for value, prime in zip(values, cases["modulus"][start:stop], strict=True)); calls += predicted.shape[0]
    count = len(cases["target"]); return {"cases": count, "exact_accuracy": exact / count, "bit_accuracy": bit_exact / (count * width), "invalid_residue_rate": invalid / count, "prediction_sha256": json_hash(predictions), "model_calls": calls, "timing": timer.finish()}


@torch.inference_mode()
def evaluate_rollouts(model: SPRMCell, cases: dict[str, Any]) -> dict[str, Any]:
    timer = Timer(); width = int(cases["width"]); count = len(cases["left"]); endpoint = prefix = transitions = invalid = calls = 0; final_predictions: list[int] = []; first_errors: list[int] = []
    for start in range(0, count, 32):
        stop = min(count, start + 32); left, right, modulus = cases["left"][start:stop], cases["right"][start:stop], cases["modulus"][start:stop]; size = len(left); state, teacher, all_prefix, any_invalid, first = [0] * size, [0] * size, [True] * size, [False] * size, [None] * size
        for position in range(width):
            input_bit = [(value >> (width - 1 - position)) & 1 for value in left]; logits, _ = model(features(state, right, modulus, input_bit, width)); state = decode(logits.ge(0)); teacher = [(2 * value + bit * other) % prime for value, bit, other, prime in zip(teacher, input_bit, right, modulus, strict=True)]; step_exact = [actual == truth for actual, truth in zip(state, teacher, strict=True)]; transitions += sum(step_exact); calls += size
            for index, correct in enumerate(step_exact):
                all_prefix[index] = all_prefix[index] and correct; any_invalid[index] = any_invalid[index] or state[index] >= modulus[index]
                if not correct and first[index] is None: first[index] = position
        expected = [(left_value * right_value) % prime for left_value, right_value, prime in zip(left, right, modulus, strict=True)]; endpoint += sum(actual == truth for actual, truth in zip(state, expected, strict=True)); prefix += sum(all_prefix); invalid += sum(any_invalid); final_predictions.extend(state); first_errors.extend(value for value in first if value is not None)
    return {"cases": count, "exact_accuracy": endpoint / count, "all_prefix_exact_accuracy": prefix / count, "transition_exact_accuracy": transitions / (count * width), "any_invalid_residue_rate": invalid / count, "first_error_mean_position": None if not first_errors else sum(first_errors) / len(first_errors), "prediction_sha256": json_hash(final_predictions), "model_calls": calls, "timing": timer.finish()}


def evaluator_task(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload["config"]; torch.set_num_threads(int(config["resource_policy"]["evaluator_torch_threads_each"])); torch.use_deterministic_algorithms(True); timer = Timer(); model = build_model(config, Path(payload["source"]), load_source=not payload.get("randomized", False))
    if payload.get("randomized", False): torch.manual_seed(int(payload["root"]) + 66491); model = build_model(config, Path(payload["source"]), load_source=False)
    else: model.load_state_dict(load_file(payload["checkpoint"]))
    model.eval(); result: dict[str, Any] = {}
    for split, widths in payload["selection"].items():
        result[split] = {}
        for width in widths:
            cases = payload["cases"][split][str(width)]; result[split][str(width)] = {"transition": evaluate_transitions(model, cases["transition"]), "rollout": evaluate_rollouts(model, cases["rollout"])}
    return {"route": payload["route"], "evaluation": result, "worker_resource": timer.finish()}


@torch.inference_mode()
def boundary_diagnostic(model: SPRMCell, cases: dict[str, Any], route: str, root: int) -> dict[str, Any]:
    width = int(cases["width"]); batch = features(cases["state"], cases["multiplicand"], cases["modulus"], cases["input_bit"], width); generator = torch.Generator().manual_seed(root + width * 100 + ROUTES.index(route)); output = model.training_forward(batch, route=route, generator=generator); predicted = decode(output.logits.ge(0)); events = output.boundary_events
    return {"route": route, "width": width, "cases": len(predicted), "exact_accuracy": sum(left == right for left, right in zip(predicted, cases["target"], strict=True)) / len(predicted), "prediction_sha256": json_hash(predicted), "event_count": len(events), "event_passes": [int(item["pass"]) for item in events], "mean_noise_rms": float(np.mean([float(item["noise_rms"]) for item in events])) if events else 0.0, "mean_pre_variance": float(np.mean([float(item["pre_variance"]) for item in events])) if events else 0.0, "mean_post_variance": float(np.mean([float(item["post_variance"]) for item in events])) if events else 0.0, "events": events}


def inference_parity_and_runtime(model: SPRMCell, cases: dict[str, Any], repeats: int) -> dict[str, Any]:
    width = int(cases["width"]); batch = features(cases["state"], cases["multiplicand"], cases["modulus"], cases["input_bit"], width); model.eval(); model.base.eval(); rng_before = torch.random.get_rng_state().clone()
    with torch.inference_mode():
        wrapped = model(batch); rng_after = torch.random.get_rng_state().clone(); direct = model.base(batch)
        for _ in range(2): model(batch); model.base(batch)
        timer = Timer()
        for _ in range(repeats): model(batch)
        wrapped_time = timer.finish(); timer = Timer()
        for _ in range(repeats): model.base(batch)
        direct_time = timer.finish()
    ratio = float(wrapped_time["wall_seconds"]) / max(float(direct_time["wall_seconds"]), 1e-12)
    return {"logits_equal": torch.equal(wrapped[0], direct[0]), "quotient_equal": torch.equal(wrapped[1], direct[1]), "rng_unchanged": torch.equal(rng_before, rng_after), "wrapped": wrapped_time, "direct": direct_time, "wall_ratio": ratio, "candidate_projected_wall_seconds": float(wrapped_time["wall_seconds"]) * 300 * width / max(1, len(cases["target"]) * repeats)}


def merge_evaluation(target: dict[str, Any], update: dict[str, Any]) -> None:
    for split, widths in update.items():
        target.setdefault(split, {}).update(widths)


def gate_result(config: dict[str, Any], routes: dict[str, Any], diagnostics: dict[str, Any], parity: dict[str, Any], separation: bool, admitted: bool) -> dict[str, Any]:
    gate = config["gates"]; sprm = routes["sprm_boundary"]["test"]; controls = ("frozen_source", "clean_tbptt", "wrong_boundary", "unscaled_boundary")
    w256_gains = {route: sprm["256"]["rollout"]["exact_accuracy"] - routes[route]["test"]["256"]["rollout"]["exact_accuracy"] for route in controls}
    randomized = diagnostics["randomized_w128"]["rollout"]["exact_accuracy"]
    results = {
        "resource_admitted": admitted,
        "w64_retention": sprm["64"]["rollout"]["exact_accuracy"] >= float(gate["minimum_w64_sprm_rollout"]),
        "w128_retention": sprm["128"]["rollout"]["exact_accuracy"] >= float(gate["minimum_w128_sprm_rollout"]),
        "w128_clean_match": sprm["128"]["rollout"]["exact_accuracy"] >= routes["clean_tbptt"]["test"]["128"]["rollout"]["exact_accuracy"] - float(gate["maximum_w128_loss_to_clean_tbptt"]),
        "w128_transition": sprm["128"]["transition"]["exact_accuracy"] >= float(gate["minimum_w128_transition"]),
        "validity": all(sprm[width]["rollout"]["any_invalid_residue_rate"] <= float(gate["maximum_invalid_rollout_rate"]) for width in ("64", "128", "256")),
        "trained_weight_dependence": sprm["128"]["rollout"]["exact_accuracy"] - randomized >= float(gate["minimum_trained_weight_drop_w128"]),
        "w256_viability": sprm["256"]["rollout"]["exact_accuracy"] >= float(gate["minimum_w256_sprm_rollout"]),
        "w256_gain_each_control": all(value >= float(gate["minimum_w256_gain_over_each_control"]) for value in w256_gains.values()),
        "boundary_placement_effect": w256_gains["wrong_boundary"] >= float(gate["minimum_boundary_placement_gain_w256"]),
        "state_scaling_effect": w256_gains["unscaled_boundary"] >= float(gate["minimum_state_scaling_gain_w256"]),
        "inference_identity": parity["logits_equal"] and parity["quotient_equal"] and parity["rng_unchanged"],
        "inference_runtime": parity["wall_ratio"] <= float(gate["maximum_inference_runtime_ratio"]),
        "reload_exact": all(value["checkpoint_reload_exact"] for value in routes.values()),
        "split_separation": separation,
        "deterministic": diagnostics["deterministic_repeat_equal"],
        "matched_work": len({(value["training"]["updates"], value["training"]["examples"], value["training"]["block_passes"]) for key, value in routes.items() if key in TRAINED_ROUTES}) == 1,
    }
    return {"results": results, "w256_gains": w256_gains, "passed": all(results.values()), "thresholds": gate}


class Experiment:
    def __init__(self, config_path: Path, output: Path, mode: str, root: int | None) -> None:
        self.config_path = config_path.resolve(); self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8")); self.output = output.resolve(); self.mode = mode; roots = self.config["roots"]; self.root = int(root if root is not None else roots["calibration"] if mode in {"preregister", "calibration"} else roots["viability"]); self.source = (ROOT / self.config["source_checkpoint"]).resolve(); self.exp300 = load_exp300()
        if sha256(self.source) != self.config["source_checkpoint_sha256"]: raise RuntimeError("source checkpoint hash mismatch")
        if tuple(self.config["routes"]) != ROUTES or tuple(self.config["trained_routes"]) != TRAINED_ROUTES: raise RuntimeError("route order mismatch")
        self.splits = {width: make_split(self.config, self.exp300, width, self.root) for width in self.config["data"]["widths"]}

    def source_receipt(self) -> dict[str, Any]:
        paths = {"config": self.config_path, "mechanism": ROOT / self.config["mechanism_map"], "model": Path(__file__).with_name("model.py"), "runner": Path(__file__), "test": ROOT / "tests/test_exp_664_sprm_boundary_perturbation.py", "checkpoint": self.source, "paper": ROOT / "research/papers/arxiv-2604.18839.pdf"}
        return {"repository_commit": git_output("rev-parse", "HEAD"), "repository_branch": git_output("branch", "--show-current"), "files": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for name, path in paths.items()}}

    def preregistration(self) -> dict[str, Any]:
        return {"run_id": self.config["run_id"], "mode": self.mode, "root": self.root, "roots": self.config["roots"], "routes": list(ROUTES), "trained_routes": list(TRAINED_ROUTES), "source": self.source_receipt(), "split_hashes": {str(width): {split: json_hash(values) for split, values in groups.items()} for width, groups in self.splits.items()}, "training": {"updates_each": self.config["training"]["steps_per_trained_route"], "schedule": self.config["training"]["schedule"], "batch_size": self.config["training"]["batch_size"]}, "inference": "exact direct SharedDilatedConvGRU forward; no noise, route, or added call", "test_access_before_frozen_primary": False}

    def frozen_cases(self, transition_count: int, rollout_count: int, widths: list[int] | None = None) -> dict[str, Any]:
        selected = widths or list(self.config["data"]["widths"]); result: dict[str, Any] = {"validation": {}, "test": {}}
        for split in result:
            offset = int(self.config["data"][f"{split}_seed_offset"])
            for width in selected:
                result[split][str(width)] = {"transition": transition_cases(self.exp300, self.splits[width][split], width, self.root + offset + width, transition_count), "rollout": rollout_cases(self.exp300, self.splits[width][split], width, self.root + offset + width + 1, rollout_count)}
        return result

    def early_result(self, routes: dict[str, Any], stage: str, primary: dict[str, Any], parity: dict[str, Any], resource: dict[str, Any], reason: str, unopened: list[str], mlflow_record: dict[str, Any]) -> dict[str, Any]:
        result = {"run_id": self.config["run_id"], "mode": self.mode, "root": self.root, "status": "completed", "decision": "reject-current-sprm-boundary-adaptation", "stage": stage, "reason": reason, "routes": routes, "staged_primary": primary, "inference_parity": parity, "unopened_surfaces": unopened, "resource": resource, "mlflow": mlflow_record, "limitations": ["Decision-safe futility left later declared surfaces unopened.", "Exact bigint arithmetic is confined to offline labels and evaluation.", "No checked, independent, official/private, candidate, submission, promotion, or completion surface was opened."]}; write_json(self.output / f"root-{self.root}-summary.json", result); return result

    def run(self) -> dict[str, Any]:
        self.output.mkdir(parents=True, exist_ok=True); prereg = self.preregistration(); write_json(self.output / f"preregistration-root-{self.root}.json", prereg)
        if self.mode == "preregister": return prereg
        calibration = self.mode == "calibration"; policy = self.config["resource_policy"]; torch.set_num_threads(int(self.config["training"]["torch_threads"])); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True); started = Timer(); experiment_id = configure_mlflow(self.config["tracking"]["experiment_name"]); routes: dict[str, Any] = {}; nested_ids: dict[str, str] = {}; worker_resources: list[dict[str, Any]] = []
        transition_count = int(self.config["calibration"]["transition_cases_per_width"] if calibration else self.config["evaluation"]["transition_cases_per_width"]); rollout_count = int(self.config["calibration"]["rollout_cases_per_width"] if calibration else self.config["evaluation"]["rollout_cases_per_width"]); validation_transition = int(self.config["calibration"]["transition_cases_per_width"] if calibration else self.config["evaluation"]["validation_transition_cases_per_width"]); validation_rollout = int(self.config["calibration"]["rollout_cases_per_width"] if calibration else self.config["evaluation"]["validation_rollout_cases_per_width"]); steps = int(self.config["calibration"]["representative_updates_per_trained_route"] if calibration else self.config["training"]["steps_per_trained_route"])
        with mlflow.start_run(run_name=f"{self.config['run_id']}-{self.mode}-{self.root}") as parent:
            parent_id = parent.info.run_id; mlflow.set_tags({"canonical_run_id": self.config["run_id"], "mode": self.mode, "root": str(self.root), "repository_commit": prereg["source"]["repository_commit"]}); mlflow.log_params({"root": self.root, "mode": self.mode, "source_checkpoint_sha256": sha256(self.source), "config_sha256": sha256(self.config_path)})
            for route in ROUTES:
                with mlflow.start_run(run_name=route, nested=True) as nested:
                    nested_ids[route] = nested.info.run_id; model, record = train_route(self.config, self.exp300, self.source, self.splits, self.root, route, steps, self.output, self.mode); routes[route] = record; mlflow.log_metrics({"updates": record["training"]["updates"], "training_wall_seconds": record["training"]["timing"]["wall_seconds"]}); del model
            mlflow_record = {"tracking_uri": tracking_uri(), "experiment_id": experiment_id, "parent_run_id": parent_id, "route_run_ids": nested_ids}
            staged_randomized: dict[str, Any] | None = None
            if not calibration:
                calibration_path = self.output / "calibration.json"
                if not calibration_path.is_file() or not json.loads(calibration_path.read_text(encoding="utf-8"))["admission"]["admitted"]: raise RuntimeError("formal root requires admitted root-20261206 calibration")
                primary_cases = self.frozen_cases(transition_count, rollout_count, [128]); payload = {"config": self.config, "source": str(self.source), "checkpoint": str(ROOT / routes["sprm_boundary"]["checkpoint"]["path"]), "route": "sprm_boundary", "root": self.root, "cases": primary_cases, "selection": {"test": [128]}}
                primary_output = evaluator_task(payload); worker_resources.append(primary_output["worker_resource"]); merge_evaluation(routes["sprm_boundary"], primary_output["evaluation"]); parity_cases = primary_cases["test"]["128"]["transition"]; parity_model = build_model(self.config, self.source); parity_model.load_state_dict(load_file(ROOT / routes["sprm_boundary"]["checkpoint"]["path"])); parity = inference_parity_and_runtime(parity_model, {key: value[:int(self.config["calibration"]["candidate_runtime_cases_w128"])] if isinstance(value, list) else value for key, value in parity_cases.items()}, 16)
                w128 = routes["sprm_boundary"]["test"]["128"]; first_ok = w128["rollout"]["exact_accuracy"] >= float(self.config["gates"]["minimum_w128_sprm_rollout"]) and w128["transition"]["exact_accuracy"] >= float(self.config["gates"]["minimum_w128_transition"]) and w128["rollout"]["any_invalid_residue_rate"] <= float(self.config["gates"]["maximum_invalid_rollout_rate"]) and parity["logits_equal"] and parity["quotient_equal"] and parity["rng_unchanged"] and parity["wall_ratio"] <= float(self.config["gates"]["maximum_inference_runtime_ratio"])
                if not first_ok:
                    resource_record = aggregate_resource(started.finish(), worker_resources, 3); result = self.early_result(routes, "sprm-w128-primary-futility", w128, parity, resource_record, "frozen W128 exact/transition/validity/inference conjunct failed", ["clean/randomized W128", "W64", "W256", "causal controls", "roots 20261208 and 20261209"], mlflow_record); mlflow.log_artifact(str(self.output / f"root-{self.root}-summary.json"), artifact_path="canonical-preview"); mlflow.set_tag("decision", result["decision"]); return result
                stage_cases = self.frozen_cases(transition_count, rollout_count, [64, 128]); payloads = [
                    {"config": self.config, "source": str(self.source), "checkpoint": str(ROOT / routes["clean_tbptt"]["checkpoint"]["path"]), "route": "clean_tbptt", "root": self.root, "cases": stage_cases, "selection": {"test": [128]}},
                    {"config": self.config, "source": str(self.source), "checkpoint": str(ROOT / routes["sprm_boundary"]["checkpoint"]["path"]), "route": "randomized_weights", "root": self.root, "cases": stage_cases, "selection": {"test": [128]}, "randomized": True},
                    {"config": self.config, "source": str(self.source), "checkpoint": str(ROOT / routes["sprm_boundary"]["checkpoint"]["path"]), "route": "sprm_boundary", "root": self.root, "cases": stage_cases, "selection": {"test": [64]}},
                ]
                context = multiprocessing.get_context("spawn")
                with concurrent.futures.ProcessPoolExecutor(max_workers=3, mp_context=context) as executor: stage_outputs = [future.result() for future in [executor.submit(evaluator_task, item) for item in payloads]]
                clean_output, randomized_output, w64_output = stage_outputs; worker_resources.extend(item["worker_resource"] for item in stage_outputs); merge_evaluation(routes["clean_tbptt"], clean_output["evaluation"]); merge_evaluation(routes["sprm_boundary"], w64_output["evaluation"]); staged_randomized = randomized_output["evaluation"]["test"]["128"]; diagnostics: dict[str, Any] = {"randomized_w128": staged_randomized}
                second_ok = routes["sprm_boundary"]["test"]["128"]["rollout"]["exact_accuracy"] >= routes["clean_tbptt"]["test"]["128"]["rollout"]["exact_accuracy"] - float(self.config["gates"]["maximum_w128_loss_to_clean_tbptt"]) and routes["sprm_boundary"]["test"]["128"]["rollout"]["exact_accuracy"] - diagnostics["randomized_w128"]["rollout"]["exact_accuracy"] >= float(self.config["gates"]["minimum_trained_weight_drop_w128"]) and routes["sprm_boundary"]["test"]["64"]["rollout"]["exact_accuracy"] >= float(self.config["gates"]["minimum_w64_sprm_rollout"])
                if not second_ok:
                    resource_record = aggregate_resource(started.finish(), worker_resources, 3); result = self.early_result(routes, "w128-control-or-w64-futility", {"w128": w128, "clean_w128": routes["clean_tbptt"]["test"]["128"], "randomized_w128": diagnostics["randomized_w128"], "w64": routes["sprm_boundary"]["test"]["64"]}, parity, resource_record, "clean-match, trained-weight, or W64 retention conjunct failed", ["W256", "remaining causal controls", "roots 20261208 and 20261209"], mlflow_record); mlflow.log_artifact(str(self.output / f"root-{self.root}-summary.json"), artifact_path="canonical-preview"); mlflow.set_tag("decision", result["decision"]); return result
                w256_cases = self.frozen_cases(transition_count, rollout_count, [256]); w256_output = evaluator_task({"config": self.config, "source": str(self.source), "checkpoint": str(ROOT / routes["sprm_boundary"]["checkpoint"]["path"]), "route": "sprm_boundary", "root": self.root, "cases": w256_cases, "selection": {"test": [256]}}); worker_resources.append(w256_output["worker_resource"]); merge_evaluation(routes["sprm_boundary"], w256_output["evaluation"])
                if routes["sprm_boundary"]["test"]["256"]["rollout"]["exact_accuracy"] < float(self.config["gates"]["minimum_w256_sprm_rollout"]):
                    resource_record = aggregate_resource(started.finish(), worker_resources, 3); result = self.early_result(routes, "sprm-w256-primary-futility", routes["sprm_boundary"]["test"]["256"], parity, resource_record, "frozen W256 viability conjunct failed", ["W256 controls", "causal diagnostics", "roots 20261208 and 20261209"], mlflow_record); mlflow.log_artifact(str(self.output / f"root-{self.root}-summary.json"), artifact_path="canonical-preview"); mlflow.set_tag("decision", result["decision"]); return result
            cases = self.frozen_cases(transition_count, rollout_count); validation_cases = self.frozen_cases(validation_transition, validation_rollout); cases["validation"] = validation_cases["validation"]; process_count = int(policy["evaluator_processes"]); context = multiprocessing.get_context("spawn"); payloads = []
            for route in ROUTES:
                if calibration: selection = {"validation": [64, 128, 256], "test": [64, 128, 256]}
                else:
                    existing_test = set(routes[route].get("test", {})); selection = {"validation": [64, 128, 256], "test": [width for width in (64, 128, 256) if str(width) not in existing_test]}
                payloads.append({"config": self.config, "source": str(self.source), "checkpoint": str(ROOT / routes[route]["checkpoint"]["path"]), "route": route, "root": self.root, "cases": cases, "selection": selection})
            with concurrent.futures.ProcessPoolExecutor(max_workers=process_count, mp_context=context) as executor: outputs = [future.result() for future in [executor.submit(evaluator_task, item) for item in payloads]]
            for item in outputs: merge_evaluation(routes[item["route"]], item["evaluation"]); worker_resources.append(item["worker_resource"])
            diagnostic_count = int(self.config["calibration"]["boundary_diagnostic_cases_per_width"] if calibration else self.config["evaluation"]["boundary_diagnostic_cases_per_width"]); diagnostic_cases = {str(width): transition_cases(self.exp300, self.splits[width]["validation"], width, self.root + 66800 + width, diagnostic_count) for width in (64, 128, 256)}; sprm_model = build_model(self.config, self.source); sprm_model.load_state_dict(load_file(ROOT / routes["sprm_boundary"]["checkpoint"]["path"])); boundary = {route: {str(width): boundary_diagnostic(sprm_model, diagnostic_cases[str(width)], route, self.root) for width in (64, 128, 256)} for route in TRAINED_ROUTES}; repeat = evaluator_task({"config": self.config, "source": str(self.source), "checkpoint": str(ROOT / routes["sprm_boundary"]["checkpoint"]["path"]), "route": "sprm_boundary", "root": self.root, "cases": cases, "selection": {"test": [128]}}); worker_resources.append(repeat["worker_resource"]); deterministic_equal = repeat["evaluation"]["test"]["128"]["rollout"]["prediction_sha256"] == routes["sprm_boundary"]["test"]["128"]["rollout"]["prediction_sha256"]
            parity_cases = cases["test"]["128"]["transition"]; parity = inference_parity_and_runtime(sprm_model, {key: value[:int(self.config["calibration"]["candidate_runtime_cases_w128"])] if isinstance(value, list) else value for key, value in parity_cases.items()}, 16); diagnostics = {"boundary": boundary, "deterministic_repeat_equal": deterministic_equal}
            if staged_randomized is not None: diagnostics["randomized_w128"] = staged_randomized
            if calibration:
                randomized = evaluator_task({"config": self.config, "source": str(self.source), "checkpoint": str(ROOT / routes["sprm_boundary"]["checkpoint"]["path"]), "route": "randomized_weights", "root": self.root, "cases": cases, "selection": {"test": [128]}, "randomized": True}); worker_resources.append(randomized["worker_resource"]); diagnostics["randomized_w128"] = randomized["evaluation"]["test"]["128"]
                training_projection = sum(float(np.mean([item["timing"]["wall_seconds"] for item in routes[route]["training"]["history"]])) * int(self.config["training"]["steps_per_trained_route"]) for route in TRAINED_ROUTES); durations = []
                for route in ROUTES:
                    duration = 0.0
                    for width in ("64", "128", "256"):
                        for split, target_transition, target_rollout in (("test", int(self.config["evaluation"]["transition_cases_per_width"]), int(self.config["evaluation"]["rollout_cases_per_width"])), ("validation", int(self.config["evaluation"]["validation_transition_cases_per_width"]), int(self.config["evaluation"]["validation_rollout_cases_per_width"]))):
                            duration += float(routes[route][split][width]["transition"]["timing"]["wall_seconds"]) * target_transition / transition_count; duration += float(routes[route][split][width]["rollout"]["timing"]["wall_seconds"]) * target_rollout / rollout_count
                    durations.append(duration)
                evaluation_projection = deterministic_makespan(durations, process_count); repeat_projection = float(repeat["evaluation"]["test"]["128"]["transition"]["timing"]["wall_seconds"]) * int(self.config["evaluation"]["transition_cases_per_width"]) / transition_count + float(repeat["evaluation"]["test"]["128"]["rollout"]["timing"]["wall_seconds"]) * int(self.config["evaluation"]["rollout_cases_per_width"]) / rollout_count; randomized_projection = float(diagnostics["randomized_w128"]["transition"]["timing"]["wall_seconds"]) * int(self.config["evaluation"]["transition_cases_per_width"]) / transition_count + float(diagnostics["randomized_w128"]["rollout"]["timing"]["wall_seconds"]) * int(self.config["evaluation"]["rollout_cases_per_width"]) / rollout_count; diagnostic_projection = sum(float(routes["sprm_boundary"]["test"][str(width)]["transition"]["timing"]["wall_seconds"]) * int(self.config["evaluation"]["boundary_diagnostic_cases_per_width"]) / transition_count for width in (64, 128, 256)) * len(TRAINED_ROUTES); expected = training_projection + evaluation_projection + repeat_projection + randomized_projection + diagnostic_projection; update_wall = [float(item["timing"]["wall_seconds"]) for route in TRAINED_ROUTES for item in routes[route]["training"]["history"]]; variability = max(update_wall) / max(float(np.mean(update_wall)), 1e-12); headroom = min(float(policy["variability_headroom_ceiling"]), max(float(policy["variability_headroom_floor"]), variability)); interim = started.finish(); aggregate_cpu = float(interim["process_cpu_seconds"]) + sum(float(item["process_cpu_seconds"]) for item in worker_resources); peak = int(interim["peak_rss_bytes"]) + sum(sorted((int(item["peak_rss_bytes"]) for item in worker_resources), reverse=True)[:process_count]); soft, hard = math.ceil(expected * headroom * float(policy["soft_guard_multiplier"])), math.ceil(expected * headroom * float(policy["hard_guard_multiplier"])); admission = {"complete_routes": all(all(split in routes[route] and set(routes[route][split]) == {"64", "128", "256"} for split in ("validation", "test")) for route in ROUTES), "projected_training_wall_seconds": training_projection, "projected_route_durations_wall_seconds": dict(zip(ROUTES, durations, strict=True)), "projected_evaluation_wall_seconds": evaluation_projection, "projected_repeat_wall_seconds": repeat_projection, "projected_randomized_wall_seconds": randomized_projection, "projected_diagnostics_wall_seconds": diagnostic_projection, "expected_wall_seconds": expected, "eta_wall_seconds": expected * headroom, "soft_guard_wall_seconds": soft, "hard_guard_wall_seconds": hard, "variability_ratio": variability, "headroom": headroom, "aggregate_process_tree_cpu_seconds": aggregate_cpu, "aggregate_peak_rss_upper_bound_bytes": peak, "candidate_inference_wall_seconds": parity["candidate_projected_wall_seconds"]}; admission["checks"] = {"complete_routes": admission["complete_routes"], "expected_value": expected <= float(policy["maximum_expected_formal_wall_seconds"]), "memory": peak <= int(policy["maximum_peak_rss_bytes"]), "candidate_inference": parity["candidate_projected_wall_seconds"] <= float(policy["maximum_candidate_inference_wall_seconds"]), "inference_identity": parity["logits_equal"] and parity["quotient_equal"] and parity["rng_unchanged"], "determinism": deterministic_equal}; admission["admitted"] = all(admission["checks"].values()); resource_record = {**interim, "aggregate_process_tree_cpu_seconds": aggregate_cpu, "aggregate_peak_rss_upper_bound_bytes": peak, "worker_resources": worker_resources}; gates = None; decision = "admit-root-20261207" if admission["admitted"] else "park-resource-or-compliance"
            else:
                if "randomized_w128" not in diagnostics:
                    randomized = evaluator_task({"config": self.config, "source": str(self.source), "checkpoint": str(ROOT / routes["sprm_boundary"]["checkpoint"]["path"]), "route": "randomized_weights", "root": self.root, "cases": cases, "selection": {"test": [128]}, "randomized": True}); diagnostics["randomized_w128"] = randomized["evaluation"]["test"]["128"]; worker_resources.append(randomized["worker_resource"])
                separation = all(set(group["train"]).isdisjoint(group["validation"]) and set(group["train"]).isdisjoint(group["test"]) and set(group["validation"]).isdisjoint(group["test"]) for group in self.splits.values()); admission = {"admitted": True, "source": "root-20261206-calibration"}; gates = gate_result(self.config, routes, diagnostics, parity, separation, True); decision = "advance-to-confirmations" if gates["passed"] else "reject-current-sprm-boundary-adaptation"; resource_record = aggregate_resource(started.finish(), worker_resources, process_count)
            result = {"run_id": self.config["run_id"], "mode": self.mode, "root": self.root, "status": "completed", "source": prereg["source"], "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "mlflow": mlflow.__version__, "logical_cores": psutil.cpu_count(), "evaluator_processes": process_count, "evaluator_torch_threads_each": int(policy["evaluator_torch_threads_each"])}, "mlflow": mlflow_record, "routes": routes, "diagnostics": diagnostics, "inference_parity": parity, "admission": admission, "gates": gates, "decision": decision, "resource": resource_record, "limitations": ["SPRM is adapted to internal ConvGRU dilation boundaries rather than a full TRM two-state scratchpad.", "Exact bigint arithmetic is confined to offline labels and evaluation.", "No checked, independent, official/private, candidate, submission, promotion, or completion surface was opened."]}; result_path = self.output / ("calibration.json" if calibration else f"root-{self.root}-summary.json"); write_json(result_path, result); mlflow.log_artifact(str(result_path), artifact_path="canonical-preview"); mlflow.set_tag("decision", decision); return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=CONFIG_PATH); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--mode", choices=("preregister", "calibration", "formal"), required=True); parser.add_argument("--root", type=int); args = parser.parse_args(); result = Experiment(args.config, args.output, args.mode, args.root).run(); print(json.dumps({"decision": result.get("decision"), "root": result.get("root"), "status": result.get("status", "preregistered")}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
