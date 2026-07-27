"""Preregister, calibrate, and execute EXP-663 DRM target-token denoising."""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
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
from experiments.exp_663_drm_target_token_denoising.model import (  # noqa: E402
    DRMCell,
    MASK_TOKEN,
    corrupt_target,
)
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402

CONFIG_PATH = ROOT / "configs/exp_663_drm_target_token_denoising.yaml"
ROUTES = ("frozen_source", "drm_cosine", "clean_recurrent", "masked_nonrecursive")
TRAINED_ROUTES = ROUTES[1:]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def load_exp300():
    previous_path, previous_model = sys.path.copy(), sys.modules.get("model")
    had_model = "model" in sys.modules
    try:
        return load_module("exp300_for_exp663", ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py")
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


def deterministic_makespan(durations: list[float], workers: int) -> float:
    """Submission-order process-pool makespan projection without work reordering."""
    if workers < 1: raise ValueError("workers must be positive")
    slots = [0.0] * workers
    for duration in durations:
        index = min(range(workers), key=lambda item: (slots[item], item))
        slots[index] += float(duration)
    return max(slots, default=0.0)


def git_output(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=120).stdout.strip()


def cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF); return float(usage.ru_utime + usage.ru_stime)


def cgroup_cpu_seconds() -> float | None:
    path = Path("/sys/fs/cgroup/cpu.stat")
    if not path.is_file(): return None
    try:
        values = dict(line.split() for line in path.read_text(encoding="utf-8").splitlines())
        return int(values["usage_usec"]) / 1_000_000.0
    except (OSError, KeyError, ValueError): return None


def peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


class Timer:
    def __init__(self) -> None:
        self.wall, self.cpu, self.cgroup = time.perf_counter(), cpu_seconds(), cgroup_cpu_seconds()

    def finish(self) -> dict[str, float | int | None]:
        current = cgroup_cpu_seconds()
        return {"wall_seconds": time.perf_counter() - self.wall, "process_cpu_seconds": cpu_seconds() - self.cpu, "cgroup_cpu_seconds": None if current is None or self.cgroup is None else current - self.cgroup, "peak_rss_bytes": peak_rss_bytes()}


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
    raw = exp300.bigint_prime_split(local, root + int(data["split_seed_offset"]) + width)
    validation = int(data["validation_prime_identities"])
    return {"train": raw["train"], "validation": raw["test"][:validation], "test": raw["test"][validation:]}


def transition_cases(exp300, primes: list[int], width: int, seed: int, count: int) -> dict[str, Any]:
    state, multiplicand, modulus, input_bit, target, quotient = exp300.BigIntSampler(primes, seed).transitions(count)
    return {"state": state, "multiplicand": multiplicand, "modulus": modulus, "input_bit": input_bit, "target": target, "quotient": quotient.tolist(), "width": width}


def rollout_cases(exp300, primes: list[int], width: int, seed: int, count: int) -> dict[str, Any]:
    left, right, modulus = exp300.BigIntSampler(primes, seed).pairs(count)
    return {"left": left, "right": right, "modulus": modulus, "width": width}


def build_model(config: dict[str, Any], source: Path, *, load_source: bool = True) -> DRMCell:
    model = config["model"]
    result = DRMCell(hidden=int(model["hidden"]), cycles=int(model["cycles"]), canvas_embedding=int(model["canvas_embedding"]))
    if load_source: result.load_source(load_file(source))
    return result


def route_window(config: dict[str, Any], route: str, intervention: str = "none") -> tuple[int, int]:
    if route == "masked_nonrecursive": return 0, 1
    warmup, gradient = int(config["model"]["warmup_steps"]), int(config["model"]["gradient_steps"])
    if intervention == "single_step": gradient = 1
    if intervention == "no_warmup": warmup = 0
    return warmup, gradient


def predict(
    model: DRMCell,
    feature_batch: torch.Tensor,
    route: str,
    config: dict[str, Any],
    *,
    seed: int,
    intervention: str = "none",
):
    if route == "frozen_source":
        logits, quotient = model.source_forward(feature_batch)
        return logits, quotient, 1, 0, []
    warmup, gradient = route_window(config, route, intervention)
    if route == "drm_cosine":
        schedule = "all_mask" if intervention == "all_mask" else "linear" if intervention == "linear_schedule" else "cosine"
        output = model.generate_and_remask(feature_batch, warmup_steps=warmup, gradient_steps=gradient, inference_steps=int(config["model"]["inference_steps"]), schedule=schedule, remask=intervention != "no_remask", seed=seed)
    elif route == "clean_recurrent":
        source_logits, _ = model.source_forward(feature_batch)
        canvas = source_logits.ge(0).long()
        output = model.denoise_window(feature_batch, canvas, warmup_steps=warmup, gradient_steps=gradient)
    elif route == "masked_nonrecursive":
        canvas = torch.full((feature_batch.shape[0], feature_batch.shape[-1]), MASK_TOKEN, dtype=torch.long)
        output = model.denoise_window(feature_batch, canvas, warmup_steps=warmup, gradient_steps=gradient)
    else: raise ValueError(route)
    return output.logits, output.quotient_logits, output.block_passes, output.analytical_flops, output.canvases


def objective(output, target: list[int], quotient: torch.Tensor, width: int, config: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
    target_bits = bits(target, width); per_bit = F.binary_cross_entropy_with_logits(output.logits, target_bits, reduction="none")
    mean, worst = per_bit.mean(), per_bit.mean(0).max(); quotient_loss = F.cross_entropy(output.quotient_logits, quotient)
    loss = mean + float(config["training"]["worst_bit_weight"]) * worst + 0.1 * quotient_loss
    return loss, {"loss": float(loss.detach()), "mean_bce": float(mean.detach()), "worst_bit_bce": float(worst.detach()), "batch_exact": float((output.logits.ge(0) == target_bits.bool()).all(1).float().mean()), "quotient_ce": float(quotient_loss.detach())}


def train_route(config: dict[str, Any], exp300, source: Path, splits: dict[int, dict[str, list[int]]], root: int, route: str, steps: int, output: Path, mode: str) -> tuple[DRMCell, dict[str, Any]]:
    random.seed(root + 663); np.random.seed(root + 663); torch.manual_seed(root + 663)
    route_dir = output / mode / f"root-{root}" / route; route_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = route_dir / "model.safetensors"; cache_path = route_dir / "training-result.json"
    cache_key = json_hash({"source": sha256(source), "root": root, "route": route, "steps": steps, "model": config["model"], "training": config["training"], "splits": splits})
    if bool(config["resource_policy"]["cache_completed_roots"]) and checkpoint.is_file() and cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("cache_key") == cache_key and sha256(checkpoint) == cached["result"]["checkpoint"]["sha256"]:
            model = build_model(config, source); model.load_state_dict(load_file(checkpoint)); return model, {**cached["result"], "cached": True}
    model = build_model(config, source); parameters = model.parameter_count(); timer = Timer()
    if route == "frozen_source":
        update_history: list[dict[str, Any]] = []; examples = block_passes = flops = 0
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=float(config["training"]["minimum_learning_rate"]))
        samplers = {width: exp300.BigIntSampler(splits[width]["train"], root + int(config["data"]["training_seed_offset"]) + width) for width in config["data"]["training_widths"]}
        generator = torch.Generator().manual_seed(root + 66303); update_history = []; examples = block_passes = flops = 0
        batch_size = int(config["training"]["batch_size"]); model.train()
        for step in range(1, steps + 1):
            width = int(config["training"]["schedule"][(step - 1) % len(config["training"]["schedule"])])
            state, multiplicand, modulus, input_bit, target, quotient = samplers[width].transitions(batch_size)
            feature_batch, target_bits = features(state, multiplicand, modulus, input_bit, width), bits(target, width)
            tick = Timer()
            if route == "clean_recurrent":
                with torch.no_grad(): source_logits, _ = model.source_forward(feature_batch)
                canvas = source_logits.ge(0).long()
            else:
                tau = torch.rand(batch_size, generator=generator)
                canvas, _ = corrupt_target(target_bits, tau, schedule="cosine", generator=generator)
            warmup, gradient = route_window(config, route); result = model.denoise_window(feature_batch, canvas, warmup_steps=warmup, gradient_steps=gradient)
            loss, metrics = objective(result, target, quotient, width, config)
            optimizer.zero_grad(set_to_none=True); loss.backward(); gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip"]))); optimizer.step(); scheduler.step()
            examples += batch_size; block_passes += result.block_passes * batch_size; flops += result.analytical_flops
            record = {"step": step, "width": width, "gradient_norm": gradient_norm, **metrics, "timing": tick.finish()}; update_history.append(record)
            if step == 1 or step % int(config["training"]["log_every_steps"]) == 0 or step == steps: print(json.dumps({"route": route, **{key: value for key, value in record.items() if key != "timing"}}), flush=True)
    save_file({key: value.detach().contiguous() for key, value in model.state_dict().items()}, checkpoint)
    reloaded = build_model(config, source); reloaded.load_state_dict(load_file(checkpoint)); reload_exact = all(torch.equal(left, right) for left, right in zip(model.state_dict().values(), reloaded.state_dict().values(), strict=True))
    result = {"route": route, "cached": False, "parameters": parameters, "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad), "checkpoint": {"path": str(checkpoint.relative_to(ROOT)), "sha256": sha256(checkpoint), "bytes": checkpoint.stat().st_size}, "checkpoint_reload_exact": reload_exact, "training": {"updates": 0 if route == "frozen_source" else steps, "examples": examples, "block_passes": block_passes, "analytical_flops": flops, "history": update_history, "timing": timer.finish()}}
    write_json(cache_path, {"cache_key": cache_key, "result": result})
    return model, result


@torch.inference_mode()
def evaluate_transitions(model: DRMCell, cases: dict[str, Any], route: str, config: dict[str, Any], *, root: int, intervention: str = "none") -> dict[str, Any]:
    timer = Timer(); width = int(cases["width"]); exact = bit_exact = invalid = calls = flops = 0; predictions: list[int] = []; change_rates: list[float] = []
    for start in range(0, len(cases["target"]), 32):
        stop = start + 32; batch = features(cases["state"][start:stop], cases["multiplicand"][start:stop], cases["modulus"][start:stop], cases["input_bit"][start:stop], width)
        logits, _, passes, analytical, canvases = predict(model, batch, route, config, seed=root + width * 1000 + start, intervention=intervention)
        predicted, target = logits.ge(0), bits(cases["target"][start:stop], width).bool(); values = decode(predicted)
        predictions.extend(values); exact += int((predicted == target).all(1).sum()); bit_exact += int((predicted == target).sum()); invalid += sum(value >= prime for value, prime in zip(values, cases["modulus"][start:stop], strict=True)); calls += passes * predicted.shape[0]; flops += analytical
        if len(canvases) > 1:
            change_rates.extend(float((left != right).float().mean()) for left, right in zip(canvases, canvases[1:]))
    count = len(cases["target"])
    return {"cases": count, "exact_accuracy": exact / count, "bit_accuracy": bit_exact / (count * width), "invalid_residue_rate": invalid / count, "prediction_sha256": json_hash(predictions), "block_passes": calls, "analytical_flops": flops, "mean_canvas_change": None if not change_rates else float(np.mean(change_rates)), "timing": timer.finish()}


@torch.inference_mode()
def evaluate_rollouts(model: DRMCell, cases: dict[str, Any], route: str, config: dict[str, Any], *, root: int, intervention: str = "none") -> dict[str, Any]:
    timer = Timer(); width = int(cases["width"]); count = len(cases["left"]); endpoint = prefix = transitions = invalid = calls = flops = 0; final_predictions: list[int] = []; first_errors: list[int] = []
    for start in range(0, count, 16):
        stop = min(count, start + 16); left, right, modulus = cases["left"][start:stop], cases["right"][start:stop], cases["modulus"][start:stop]; batch_size = len(left)
        state, teacher, all_prefix, any_invalid, first = [0] * batch_size, [0] * batch_size, [True] * batch_size, [False] * batch_size, [None] * batch_size
        for position in range(width):
            input_bit = [(value >> (width - 1 - position)) & 1 for value in left]; feature_batch = features(state, right, modulus, input_bit, width)
            logits, _, passes, analytical, _ = predict(model, feature_batch, route, config, seed=root + width * 100000 + start * 1000 + position, intervention=intervention)
            state = decode(logits.ge(0)); teacher = [(2 * value + bit * other) % prime for value, bit, other, prime in zip(teacher, input_bit, right, modulus, strict=True)]
            exact_step = [actual == truth for actual, truth in zip(state, teacher, strict=True)]; transitions += sum(exact_step); calls += passes * batch_size; flops += analytical
            for index, correct in enumerate(exact_step):
                all_prefix[index] = all_prefix[index] and correct; any_invalid[index] = any_invalid[index] or state[index] >= modulus[index]
                if not correct and first[index] is None: first[index] = position
        expected = [(a * b) % prime for a, b, prime in zip(left, right, modulus, strict=True)]
        endpoint += sum(actual == truth for actual, truth in zip(state, expected, strict=True)); prefix += sum(all_prefix); invalid += sum(any_invalid); final_predictions.extend(state); first_errors.extend(position for position in first if position is not None)
    return {"cases": count, "exact_accuracy": endpoint / count, "all_prefix_exact_accuracy": prefix / count, "transition_exact_accuracy": transitions / (count * width), "any_invalid_residue_rate": invalid / count, "first_error_mean_position": None if not first_errors else sum(first_errors) / len(first_errors), "prediction_sha256": json_hash(final_predictions), "block_passes": calls, "analytical_flops": flops, "timing": timer.finish()}


def evaluator_route_task(payload: dict[str, Any]) -> dict[str, Any]:
    config, route, root = payload["config"], payload["route"], int(payload["root"])
    torch.set_num_threads(int(config["resource_policy"]["evaluator_torch_threads_each"])); torch.use_deterministic_algorithms(True)
    model = build_model(config, Path(payload["source"])); model.load_state_dict(load_file(payload["checkpoint"])); model.eval(); timer = Timer()
    result = {}
    for split in ("validation", "test"):
        result[split] = {width: {"transition": evaluate_transitions(model, cases["transition"], route, config, root=root), "rollout": evaluate_rollouts(model, cases["rollout"], route, config, root=root)} for width, cases in payload[split].items()}
    return {"route": route, "evaluation": result, "worker_resource": timer.finish()}


def evaluator_diagnostic_task(payload: dict[str, Any]) -> dict[str, Any]:
    config, root = payload["config"], int(payload["root"])
    torch.set_num_threads(int(config["resource_policy"]["evaluator_torch_threads_each"])); torch.use_deterministic_algorithms(True)
    if payload.get("randomized", False):
        torch.manual_seed(root + 1663); model = build_model(config, Path(payload["source"]), load_source=False)
    else:
        model = build_model(config, Path(payload["source"])); model.load_state_dict(load_file(payload["checkpoint"]))
    model.eval(); timer = Timer(); intervention = payload.get("intervention", "none")
    result = evaluate_rollouts(model, payload["cases"], "drm_cosine", config, root=root, intervention=intervention)
    return {"name": payload["name"], "result": result, "worker_resource": timer.finish()}


def without_timing(value: Any) -> Any:
    if isinstance(value, dict): return {key: without_timing(item) for key, item in value.items() if key not in {"timing", "cached", "worker_resource"}}
    if isinstance(value, list): return [without_timing(item) for item in value]
    return value


def calibration_equivalence(reference: dict[str, Any], reference_prereg: dict[str, Any], routes: dict[str, Any], diagnostics: dict[str, Any], candidate_prereg: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {
        # JSON artifacts are serialized with sorted keys, so the frozen route
        # declaration (not mapping iteration order) is the ordering authority.
        "routes": reference_prereg["routes"] == candidate_prereg["routes"] == list(ROUTES)
        and set(reference["routes"]) == set(routes) == set(ROUTES),
        "split_hashes": reference_prereg["split_hashes"] == candidate_prereg["split_hashes"],
        "roots": reference_prereg["roots"] == candidate_prereg["roots"],
        "determinism": reference["diagnostics"]["deterministic_repeat_equal"] == diagnostics["deterministic_repeat_equal"],
    }
    for route in ROUTES:
        old, new = reference["routes"][route], routes[route]
        checks[f"{route}_checkpoint"] = old["checkpoint"]["sha256"] == new["checkpoint"]["sha256"]
        checks[f"{route}_parameters"] = old["parameters"] == new["parameters"] and old["trainable_parameters"] == new["trainable_parameters"]
        checks[f"{route}_reload"] = old["checkpoint_reload_exact"] == new["checkpoint_reload_exact"]
        checks[f"{route}_training"] = without_timing(old["training"]) == without_timing(new["training"])
        checks[f"{route}_validation"] = without_timing(old["validation"]) == without_timing(new["validation"])
        checks[f"{route}_test"] = without_timing(old["test"]) == without_timing(new["test"])
    for name in reference["diagnostics"]["interventions"]:
        checks[f"intervention_{name}"] = without_timing(reference["diagnostics"]["interventions"][name]) == without_timing(diagnostics["interventions"][name])
    checks["randomized_weights"] = without_timing(reference["diagnostics"]["randomized_weights"]) == without_timing(diagnostics["randomized_weights"])
    return {"checks": checks, "passed": all(checks.values()), "case_order_evidence": "identical split hashes plus identical per-route/split/width prediction hashes and metrics"}


def gate_result(config: dict[str, Any], routes: dict[str, Any], diagnostics: dict[str, Any], separation: bool, admitted: bool) -> dict[str, Any]:
    gate = config["gates"]; drm = routes["drm_cosine"]["test"]; clean = routes["clean_recurrent"]["test"]
    controls = ("frozen_source", "clean_recurrent", "masked_nonrecursive")
    results = {
        "w128_retention": drm["128"]["rollout"]["exact_accuracy"] >= float(gate["minimum_w128_drm_rollout"]),
        "w128_matched_loss": clean["128"]["rollout"]["exact_accuracy"] - drm["128"]["rollout"]["exact_accuracy"] <= float(gate["maximum_w128_loss_to_clean_recurrent"]),
        "w256_viability": drm["256"]["rollout"]["exact_accuracy"] >= float(gate["minimum_w256_drm_rollout"]),
        "w256_gain_each_control": all(drm["256"]["rollout"]["exact_accuracy"] - routes[name]["test"]["256"]["rollout"]["exact_accuracy"] >= float(gate["minimum_w256_gain_over_each_control"]) for name in controls),
        "validity": max(value["rollout"]["any_invalid_residue_rate"] for value in drm.values()) <= float(gate["maximum_invalid_rollout_rate"]),
        "trained_weight_dependence": drm["128"]["rollout"]["exact_accuracy"] - diagnostics["randomized_weights"]["exact_accuracy"] >= float(gate["minimum_trained_weight_drop_w128"]),
        "recursive_dependence": drm["128"]["rollout"]["exact_accuracy"] - diagnostics["interventions"]["single_step"]["exact_accuracy"] >= float(gate["minimum_recursive_drop_w128"]),
        "cosine_dependence": drm["128"]["rollout"]["exact_accuracy"] - diagnostics["interventions"]["linear_schedule"]["exact_accuracy"] >= float(gate["minimum_cosine_drop_w128"]),
        "remasking_dependence": drm["128"]["rollout"]["exact_accuracy"] - diagnostics["interventions"]["no_remask"]["exact_accuracy"] >= float(gate["minimum_remasking_drop_w128"]),
        "reload_exact": all(value["checkpoint_reload_exact"] for value in routes.values()),
        "split_separation": separation,
        "resource_admitted": admitted,
        "deterministic": diagnostics["deterministic_repeat_equal"],
        "parameter_match": len({value["parameters"] for value in routes.values()}) == 1,
        "matched_drm_clean_work": (routes["drm_cosine"]["training"]["updates"], routes["drm_cosine"]["training"]["examples"], routes["drm_cosine"]["training"]["block_passes"]) == (routes["clean_recurrent"]["training"]["updates"], routes["clean_recurrent"]["training"]["examples"], routes["clean_recurrent"]["training"]["block_passes"]),
    }
    return {"thresholds": gate, "results": results, "passed": all(results.values())}


class Experiment:
    def __init__(self, config_path: Path, output: Path, mode: str, root: int) -> None:
        self.config_path, self.output, self.mode, self.root = config_path.resolve(), output.resolve(), mode, int(root)
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8")); self.output.mkdir(parents=True, exist_ok=True)
        self.source = (ROOT / self.config["source_checkpoint"]).resolve()
        if sha256(self.source) != self.config["source_checkpoint_sha256"]: raise RuntimeError("source checkpoint hash mismatch")
        self.exp300 = load_exp300(); torch.set_num_threads(int(self.config["training"]["torch_threads"])); torch.use_deterministic_algorithms(True)
        try: torch.set_num_interop_threads(1)
        except RuntimeError:
            if torch.get_num_interop_threads() != 1: raise
        self.splits = {width: make_split(self.config, self.exp300, width, self.root) for width in self.config["data"]["widths"]}

    def source_record(self) -> dict[str, Any]:
        paths = {"config": self.config_path, "runner": Path(__file__), "model": Path(__file__).with_name("model.py"), "test": ROOT / "tests/test_exp_663_drm_target_token_denoising.py", "mechanism": ROOT / self.config["mechanism_map"], "generator": ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py", "paper": ROOT / "research/papers/arxiv-2604.18839.pdf", "checkpoint": self.source}
        return {"repository_commit": git_output("rev-parse", "HEAD"), "repository_branch": git_output("branch", "--show-current"), "files": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for name, path in paths.items()}}

    def frozen_cases(self, split: str, transition_count: int, rollout_count: int, widths: list[int] | None = None) -> dict[str, Any]:
        offset = int(self.config["data"]["validation_seed_offset" if split == "validation" else "test_seed_offset"])
        selected = self.config["data"]["widths"] if widths is None else widths
        return {str(width): {"transition": transition_cases(self.exp300, self.splits[width][split], width, self.root + offset + width, transition_count), "rollout": rollout_cases(self.exp300, self.splits[width][split], width, self.root + offset + 1000 + width, rollout_count)} for width in selected}

    def preregister(self) -> dict[str, Any]:
        record = {"run_id": self.config["run_id"], "root": self.root, "mode": self.mode, "source": self.source_record(), "split_hashes": {str(width): {name: json_hash(values) for name, values in split.items()} for width, split in self.splits.items()}, "roots": self.config["roots"], "routes": list(ROUTES), "test_access_before_frozen_primary": False, "inference": {"initial_canvas": "fully_masked", "remasking": "uniform target-independent fixed-seed", "steps": self.config["model"]["inference_steps"]}, "fixture_identity": {"manifest": "../t054/fixture-manifest.json", "sha256": "cd57ff0bb21c616bf91bb0245ca854fa71968b3e43754c24065548394ec3552d", "files": 268, "bytes": 27086830}}
        write_json(self.output / f"preregistration-root-{self.root}.json", record); return record

    def run(self) -> dict[str, Any]:
        calibration = self.mode == "calibration"; steps = int(self.config["calibration"]["representative_updates_per_trained_route"] if calibration else self.config["training"]["steps_per_trained_route"])
        transition_count = int(self.config["calibration"]["transition_cases_per_width"] if calibration else self.config["evaluation"]["transition_cases_per_width"]); rollout_count = int(self.config["calibration"]["rollout_cases_per_width"] if calibration else self.config["evaluation"]["rollout_cases_per_width"])
        prereg = self.preregister(); started = Timer(); routes: dict[str, Any] = {}; worker_resources: list[dict[str, Any]] = []
        experiment_id = configure_mlflow(self.config["tracking"]["experiment_name"])
        with mlflow.start_run(run_name=f"{self.config['run_id']}-{self.mode}-{self.root}") as parent:
            parent_id = parent.info.run_id; nested_ids = {}
            mlflow.set_tags({"canonical_run_id": self.config["run_id"], "mode": self.mode, "root": str(self.root), "source_checkpoint_sha256": self.config["source_checkpoint_sha256"]})
            for route in ROUTES:
                with mlflow.start_run(run_name=route, nested=True) as nested:
                    nested_ids[route] = nested.info.run_id; model, result = train_route(self.config, self.exp300, self.source, self.splits, self.root, route, steps, self.output, self.mode); routes[route] = result
                    mlflow.log_metric("training_updates", result["training"]["updates"]); mlflow.log_metric("training_wall_seconds", result["training"]["timing"]["wall_seconds"])
                    del model; gc.collect()
            process_count = int(self.config["resource_policy"]["evaluator_processes"]); context = multiprocessing.get_context("spawn")
            drm_checkpoint = str((ROOT / routes["drm_cosine"]["checkpoint"]["path"]).resolve())
            if not calibration:
                early = self.frozen_cases("test", transition_count, rollout_count, widths=[128])["128"]["rollout"]
                payload = {"name": "primary_w128", "config": self.config, "source": str(self.source), "checkpoint": drm_checkpoint, "root": self.root, "cases": early}
                with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
                    primary_w128 = executor.submit(evaluator_diagnostic_task, payload).result()
                worker_resources.append(primary_w128["worker_resource"]); early_result = primary_w128["result"]
                early_pass = early_result["exact_accuracy"] >= float(self.config["gates"]["minimum_w128_drm_rollout"]) and early_result["any_invalid_residue_rate"] <= float(self.config["gates"]["maximum_invalid_rollout_rate"])
                if not early_pass:
                    resource = started.finish(); aggregate_cpu = float(resource["process_cpu_seconds"]) + sum(float(item["process_cpu_seconds"]) for item in worker_resources); peak_bound = int(resource["peak_rss_bytes"]) + sum(sorted((int(item["peak_rss_bytes"]) for item in worker_resources), reverse=True)[:process_count])
                    gates = {"passed": False, "stage": "drm-w128-primary-futility", "results": {"w128_retention": early_result["exact_accuracy"] >= float(self.config["gates"]["minimum_w128_drm_rollout"]), "validity": early_result["any_invalid_residue_rate"] <= float(self.config["gates"]["maximum_invalid_rollout_rate"])} }
                    result = {"run_id": self.config["run_id"], "mode": self.mode, "root": self.root, "status": "completed", "source": prereg["source"], "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "mlflow": mlflow.__version__, "torch_threads": torch.get_num_threads(), "evaluator_processes": 1}, "mlflow": {"tracking_uri": tracking_uri(), "experiment_id": experiment_id, "parent_run_id": parent_id, "route_run_ids": nested_ids}, "routes": routes, "staged_primary_w128": early_result, "unopened_surfaces": ["W256", "control evaluation", "causal interventions", "roots 20261204 and 20261205"], "gates": gates, "admission": {"admitted": True, "source": "root-20261202-repair-calibration"}, "decision": "reject-current-drm-target-token-adaptation", "resource": {**resource, "aggregate_process_tree_cpu_seconds": aggregate_cpu, "aggregate_peak_rss_upper_bound_bytes": peak_bound}, "limitations": ["Terminal futility was reached on the frozen first W128 conjunct; later surfaces remained unopened.", "Exact bigint arithmetic is confined to offline labels and evaluation.", "No checked, prior-independent, official/private, candidate, submission, or promotion surface was opened."]}
                    path = self.output / f"root-{self.root}-summary.json"; write_json(path, result); mlflow.log_artifact(str(path), artifact_path="canonical-preview"); mlflow.set_tag("decision", result["decision"]); return result

            test = self.frozen_cases("test", transition_count, rollout_count); validation = self.frozen_cases("validation", int(self.config["calibration"]["transition_cases_per_width"] if calibration else self.config["evaluation"]["validation_transition_cases"]), int(self.config["calibration"]["rollout_cases_per_width"] if calibration else self.config["evaluation"]["validation_rollout_cases"]))
            route_payloads = [{"config": self.config, "source": str(self.source), "checkpoint": str((ROOT / routes[route]["checkpoint"]["path"]).resolve()), "route": route, "root": self.root, "validation": validation, "test": test} for route in ROUTES]
            with concurrent.futures.ProcessPoolExecutor(max_workers=process_count, mp_context=context) as executor:
                futures = [executor.submit(evaluator_route_task, payload) for payload in route_payloads]
                route_outputs = [future.result() for future in futures]
            for output in route_outputs:
                route = output["route"]; routes[route].update(output["evaluation"]); worker_resources.append(output["worker_resource"])
                mlflow.log_metric(f"{route}_w128_rollout_exact", routes[route]["test"]["128"]["rollout"]["exact_accuracy"]); mlflow.log_metric(f"{route}_w256_rollout_exact", routes[route]["test"]["256"]["rollout"]["exact_accuracy"])
            intervention_cases = test["128"]["rollout"]
            diagnostic_payloads = [{"name": name, "intervention": name, "config": self.config, "source": str(self.source), "checkpoint": drm_checkpoint, "root": self.root, "cases": intervention_cases} for name in self.config["evaluation"]["interventions"]]
            diagnostic_payloads += [{"name": "randomized_weights", "randomized": True, "config": self.config, "source": str(self.source), "checkpoint": drm_checkpoint, "root": self.root, "cases": intervention_cases}, {"name": "deterministic_repeat", "config": self.config, "source": str(self.source), "checkpoint": drm_checkpoint, "root": self.root, "cases": intervention_cases}]
            with concurrent.futures.ProcessPoolExecutor(max_workers=process_count, mp_context=context) as executor:
                futures = [executor.submit(evaluator_diagnostic_task, payload) for payload in diagnostic_payloads]
                diagnostic_outputs = [future.result() for future in futures]
            diagnostic_map = {item["name"]: item["result"] for item in diagnostic_outputs}; worker_resources.extend(item["worker_resource"] for item in diagnostic_outputs)
            diagnostics = {"interventions": {name: diagnostic_map[name] for name in self.config["evaluation"]["interventions"]}, "randomized_weights": diagnostic_map["randomized_weights"], "deterministic_repeat": diagnostic_map["deterministic_repeat"], "deterministic_repeat_equal": diagnostic_map["deterministic_repeat"]["prediction_sha256"] == routes["drm_cosine"]["test"]["128"]["rollout"]["prediction_sha256"]}
            separation = all(set(split["validation"]).isdisjoint(split["test"]) and set(split["train"]).isdisjoint(split["validation"] + split["test"]) for split in self.splits.values())
            if calibration:
                update_wall = [item["timing"]["wall_seconds"] for route in TRAINED_ROUTES for item in routes[route]["training"]["history"]]
                projected_training = sum(np.mean([item["timing"]["wall_seconds"] for item in routes[route]["training"]["history"]]) * int(self.config["training"]["steps_per_trained_route"]) for route in TRAINED_ROUTES)
                route_durations = []
                for route in ROUTES:
                    duration = 0.0
                    for width in ("64", "128", "256"):
                        duration += routes[route]["test"][width]["transition"]["timing"]["wall_seconds"] * int(self.config["evaluation"]["transition_cases_per_width"]) / transition_count
                        duration += routes[route]["test"][width]["rollout"]["timing"]["wall_seconds"] * int(self.config["evaluation"]["rollout_cases_per_width"]) / rollout_count
                        duration += routes[route]["validation"][width]["transition"]["timing"]["wall_seconds"] * int(self.config["evaluation"]["validation_transition_cases"]) / transition_count
                        duration += routes[route]["validation"][width]["rollout"]["timing"]["wall_seconds"] * int(self.config["evaluation"]["validation_rollout_cases"]) / rollout_count
                    route_durations.append(duration)
                projected_eval = deterministic_makespan(route_durations, process_count)
                intervention_scale = int(self.config["evaluation"]["rollout_cases_per_width"]) / rollout_count
                diagnostic_durations = [diagnostic_map[payload["name"]]["timing"]["wall_seconds"] * intervention_scale for payload in diagnostic_payloads]
                projected_diagnostics = deterministic_makespan(diagnostic_durations, process_count)
                expected = projected_training + projected_eval + projected_diagnostics; variability = max(update_wall) / max(float(np.mean(update_wall)), 1e-9); policy = self.config["resource_policy"]; headroom = min(float(policy["variability_headroom_ceiling"]), max(float(policy["variability_headroom_floor"]), variability))
                interim = started.finish(); aggregate_cpu = float(interim["process_cpu_seconds"]) + sum(float(item["process_cpu_seconds"]) for item in worker_resources); peak_bound = int(interim["peak_rss_bytes"]) + sum(sorted((int(item["peak_rss_bytes"]) for item in worker_resources), reverse=True)[:process_count]); soft = math.ceil(expected * headroom * float(policy["soft_guard_multiplier"])); hard = math.ceil(expected * headroom * float(policy["hard_guard_multiplier"])); repair = policy["repair_guards"]
                guard_results = {"wall": interim["wall_seconds"] <= float(repair["maximum_calibration_wall_seconds"]), "aggregate_process_tree_cpu": aggregate_cpu <= float(repair["maximum_aggregate_process_tree_cpu_seconds"]), "aggregate_peak_rss": peak_bound <= int(repair["maximum_aggregate_peak_rss_bytes"]), "expected": expected <= float(repair["maximum_expected_wall_seconds"]), "eta": expected * headroom <= float(repair["maximum_eta_wall_seconds"]), "soft": soft <= float(repair["maximum_soft_guard_wall_seconds"]), "hard": hard <= float(repair["maximum_hard_guard_wall_seconds"])}
                admission = {"complete_routes": set(routes) == set(ROUTES), "evaluator_processes": process_count, "evaluator_torch_threads_each": int(policy["evaluator_torch_threads_each"]), "projected_route_durations_wall_seconds": dict(zip(ROUTES, route_durations, strict=True)), "projected_training_wall_seconds": projected_training, "projected_evaluation_wall_seconds": projected_eval, "projected_diagnostics_wall_seconds": projected_diagnostics, "expected_wall_seconds": expected, "eta_wall_seconds": expected * headroom, "variability_ratio": variability, "headroom": headroom, "soft_guard_wall_seconds": soft, "hard_guard_wall_seconds": hard, "aggregate_process_tree_cpu_seconds": aggregate_cpu, "aggregate_peak_rss_upper_bound_bytes": peak_bound, "guard_results": guard_results}
                reference_path = (ROOT / policy["repair_reference_calibration"]).resolve(); reference_prereg_path = (ROOT / policy["repair_reference_preregistration"]).resolve()
                equivalence = calibration_equivalence(json.loads(reference_path.read_text(encoding="utf-8")), json.loads(reference_prereg_path.read_text(encoding="utf-8")), routes, diagnostics, prereg)
                admission["equivalence"] = equivalence; admission["admitted"] = admission["complete_routes"] and all(guard_results.values()) and equivalence["passed"]
                resource = {**interim, "aggregate_process_tree_cpu_seconds": aggregate_cpu, "aggregate_peak_rss_upper_bound_bytes": peak_bound, "worker_resources": worker_resources}
            else:
                equivalence = None; admission = {"admitted": True, "source": "root-20261202-canonical-calibration"}
            gates = gate_result(self.config, routes, diagnostics, separation, bool(admission["admitted"])); decision = "advance-to-confirmations" if gates["passed"] else "reject-current-drm-target-token-adaptation"
            if not calibration:
                interim = started.finish(); aggregate_cpu = float(interim["process_cpu_seconds"]) + sum(float(item["process_cpu_seconds"]) for item in worker_resources); peak_bound = int(interim["peak_rss_bytes"]) + sum(sorted((int(item["peak_rss_bytes"]) for item in worker_resources), reverse=True)[:process_count]); resource = {**interim, "aggregate_process_tree_cpu_seconds": aggregate_cpu, "aggregate_peak_rss_upper_bound_bytes": peak_bound, "worker_resources": worker_resources}
            result = {"run_id": self.config["run_id"], "mode": self.mode, "root": self.root, "status": "completed", "source": prereg["source"], "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "mlflow": mlflow.__version__, "torch_threads": torch.get_num_threads(), "logical_cores": psutil.cpu_count(), "evaluator_processes": process_count, "evaluator_torch_threads_each": int(self.config["resource_policy"]["evaluator_torch_threads_each"])}, "mlflow": {"tracking_uri": tracking_uri(), "experiment_id": experiment_id, "parent_run_id": parent_id, "route_run_ids": nested_ids}, "routes": routes, "diagnostics": diagnostics, "equivalence": equivalence, "gates": gates, "admission": admission, "decision": decision, "resource": resource, "limitations": ["Exact bigint arithmetic is confined to offline labels and evaluation.", "DRM is adapted from ARC colour grids to residue-bit next-state canvases; no ARC reproduction is claimed.", "No checked, prior-independent, official/private, candidate, submission, or promotion surface was opened."]}
            path = self.output / ("calibration.json" if calibration else f"root-{self.root}-summary.json"); write_json(path, result); mlflow.log_artifact(str(path), artifact_path="canonical-preview"); mlflow.set_tag("decision", decision)
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=CONFIG_PATH); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--mode", choices=("preregister", "calibration", "formal"), required=True); parser.add_argument("--root", type=int)
    args = parser.parse_args(); config = yaml.safe_load(args.config.read_text(encoding="utf-8")); root_key = "calibration" if args.mode in {"preregister", "calibration"} else "viability"; root = int(config["roots"][root_key] if args.root is None else args.root)
    experiment = Experiment(args.config, args.output, args.mode, root); result = experiment.preregister() if args.mode == "preregister" else experiment.run(); print(json.dumps({"status": result.get("status", "preregistered"), "mode": args.mode, "root": root, "decision": result.get("decision"), "admitted": (result.get("admission") or {}).get("admitted")})); return 0


if __name__ == "__main__": raise SystemExit(main())
