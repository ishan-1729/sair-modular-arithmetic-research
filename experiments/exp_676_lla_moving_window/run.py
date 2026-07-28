"""Calibrate and execute EXP676's source-faithful moving-window comparison."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import psutil
import torch
import yaml
from safetensors.torch import load_file, save_file
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.exp_655_layer_loop_compute_matched_recurrence.fixtures import verified_copy  # noqa: E402
from experiments.exp_676_lla_moving_window.model import reinjection_receipt, selected_depths, state_dict_identity, trace_forward  # noqa: E402
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402


@dataclass(frozen=True)
class RootSpec:
    root: int
    train_seed: int
    eval_seed: int


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def git_output(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=120).stdout.strip()


def cgroup_cpu_seconds() -> float | None:
    path = Path("/sys/fs/cgroup/cpu.stat")
    if not path.is_file(): return None
    values = dict(line.split() for line in path.read_text(encoding="utf-8").splitlines())
    return float(values["usage_usec"]) / 1_000_000 if "usage_usec" in values else None


class Timer:
    def __init__(self) -> None:
        self.wall = time.perf_counter(); self.cgroup = cgroup_cpu_seconds(); self.process = psutil.Process()
        self.initial_cpu = self._cpu(); self.peak = self._rss()

    def _tree(self) -> list[psutil.Process]:
        try: return [self.process, *self.process.children(recursive=True)]
        except psutil.Error: return [self.process]

    def _cpu(self) -> float:
        total = 0.0
        for process in self._tree():
            try:
                value = process.cpu_times(); total += value.user + value.system
            except psutil.Error: pass
        return total

    def _rss(self) -> int:
        total = 0
        for process in self._tree():
            try: total += process.memory_info().rss
            except psutil.Error: pass
        return total

    def sample(self) -> None: self.peak = max(self.peak, self._rss())

    def result(self) -> dict[str, float | int | None]:
        self.sample(); current = cgroup_cpu_seconds()
        return {"wall_seconds": time.perf_counter() - self.wall, "process_tree_cpu_seconds": self._cpu() - self.initial_cpu, "cgroup_cpu_seconds": None if self.cgroup is None or current is None else current - self.cgroup, "peak_process_tree_rss_bytes": self.peak}


def aggregate(resources: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("wall_seconds", "process_tree_cpu_seconds", "cgroup_cpu_seconds"):
        values = [item[key] for item in resources]; result[key] = None if any(v is None for v in values) else sum(float(v) for v in values)
    result["peak_process_tree_rss_bytes"] = max(int(item["peak_process_tree_rss_bytes"]) for item in resources)
    return result


class Experiment:
    trained_arms = ("moving_window", "endpoint_only", "fixed_window", "equal_work_multi_endpoint", "shuffled_window")

    def __init__(self, config_path: Path, output: Path, mode: str, root_override: int | None = None) -> None:
        self.config_path = config_path.resolve(); self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.output = output.resolve(); self.output.mkdir(parents=True, exist_ok=True); self.mode = mode; self.root_override = root_override
        self.exp300 = load_module("exp300_for_exp676", ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py")
        self.exp329 = load_module("exp329_for_exp676", ROOT / "experiments/exp_329_cellular_width64_autonomous_state_recovery/run.py")
        self.cellular = load_module("cellular_for_exp676", ROOT / "experiments/exp_220_width_independent_cellular/model.py")
        torch.set_num_threads(int(self.config["training"]["torch_threads"])); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
        self.checkpoint = self.materialize_checkpoint()

    def materialize_checkpoint(self) -> Path:
        destination = self.output / "fixtures/exp609-weights.safetensors"; expected = self.config["source_contract"]["checkpoint_sha256"]
        candidates = [ROOT / ".scratch/exp-675-rrt-svd-teacher-uptraining/fixtures/exp609-weights.safetensors", Path("/mnt/d/Projects/SAIR_mod_challenge/.scratch/exp-609-tier6-straight-through-confirmation/straight_through_continuation/weights.safetensors"), Path("/mnt/d/Projects/SAIR_mod_challenge_exp655_t048_wt/.scratch/exp-609-tier6-straight-through-confirmation/straight_through_continuation/weights.safetensors")]
        source = next((path for path in candidates if path.is_file() and sha256(path) == expected), None)
        if source is None: raise RuntimeError("authenticated EXP609 checkpoint unavailable")
        verified_copy(source, destination, expected); return destination

    def source_identity(self) -> dict[str, str]:
        paths = {"config": self.config_path, "model": Path(__file__).with_name("model.py"), "runner": Path(__file__), "test": ROOT / "tests/test_exp_676_lla_moving_window.py", "preregistration": ROOT / self.config["preregistration"], "paper": ROOT / self.config["source_contract"]["paper_path"], "checkpoint": self.checkpoint}
        return {"repository_commit": git_output("rev-parse", "HEAD"), **{f"{key}_sha256": sha256(path) for key, path in paths.items()}}

    def assert_frozen_clean(self) -> None:
        paths = [str(self.config_path.relative_to(ROOT)), "experiments/exp_676_lla_moving_window", "tests/test_exp_676_lla_moving_window.py", self.config["preregistration"]]
        if git_output("status", "--porcelain", "--", *paths): raise RuntimeError("execution requires committed frozen EXP676 bytes")

    def spec(self) -> RootSpec:
        data = self.config["data"]
        if self.mode == "calibration": values = data["calibration"]
        elif self.mode == "formal": values = data["formal"]
        else:
            matches = [item for item in data["confirmations"] if int(item["root"]) == int(self.root_override or -1)]
            if len(matches) != 1: raise ValueError("confirmation requires frozen root 20261256 or 20261257")
            values = matches[0]
        return RootSpec(**values)

    def split(self, width: int, root: int) -> dict[str, list[int]]:
        local = {"data": {"prime_bits_min": width - int(self.config["data"]["prime_bit_span"]) + 1, "prime_bits_max": width, "train_prime_identities": int(self.config["data"]["train_prime_identities"]), "test_prime_identities": int(self.config["data"]["test_prime_identities"])}}
        return self.exp300.bigint_prime_split(local, int(self.config["data"]["split_seed_offset"]) + root + width)

    def base(self, *, randomized: bool = False, seed: int = 0) -> nn.Module:
        values = self.config["model"]
        if randomized: torch.manual_seed(seed)
        model = self.cellular.SharedDilatedConvGRU(width=128, hidden=int(values["hidden"]), cycles=int(values["cycles"]))
        if not randomized: model.load_state_dict(load_file(self.checkpoint))
        return model

    def _soft_features(self, state_plane: torch.Tensor, right: list[int], modulus: list[int], bit: list[int], width: int) -> torch.Tensor:
        batch = len(right); bit_plane = torch.tensor(bit, dtype=torch.float32)[:, None].expand(batch, width)
        lsb = torch.zeros(batch, width); msb = torch.zeros(batch, width); lsb[:, 0] = 1; msb[:, -1] = 1
        return torch.stack((state_plane, self.exp300.bits(right, width), self.exp300.bits(modulus, width), bit_plane, lsb, msb), dim=1).mul(2).sub(1)

    def traced_loss(self, model: nn.Module, features: torch.Tensor, target: list[int], quotient: torch.Tensor, width: int, kind: str, step: int, total_steps: int, seed: int) -> tuple[torch.Tensor, dict[str, Any], torch.Tensor]:
        outputs = trace_forward(model, features); indices = selected_depths(kind, visits=len(outputs), step=step, total_steps=total_steps, window_size=int(self.config["window"]["size"]), seed=seed)
        losses = []
        for index in indices:
            loss, _ = self.exp300.transition_loss(outputs[index][0], outputs[index][1], target, quotient, width, float(self.config["training"]["worst_bit_weight"]), float(self.config["training"]["quotient_weight"])); losses.append(loss)
        return torch.stack(losses).mean(), {"selected_depths": list(indices), "loss_calls": len(losses), "per_selected_loss": [float(loss.detach()) for loss in losses]}, outputs[-1][0]

    def recovery_loss(self, model: nn.Module, sampler: Any, rng: random.Random, width: int, kind: str, step: int, total_steps: int, seed: int) -> tuple[torch.Tensor, dict[str, Any]]:
        cfg = self.config["training"]; batch, chunk = int(cfg["recovery_batch_size"]), int(cfg["recovery_chunk_steps"])
        multiplier, right, modulus = sampler.pairs(batch); starts = [rng.randrange(0, width - chunk + 1) for _ in range(batch)]
        teacher = self.exp329.teacher_states_before(multiplier, right, modulus, starts, width); state_plane = self.exp300.bits(teacher, width); losses = []; selections = []
        for offset in range(chunk):
            positions = [start + offset for start in starts]; bit = [(left >> (width - 1 - position)) & 1 for left, position in zip(multiplier, positions)]
            raw = [2 * state + flag * value for state, flag, value in zip(teacher, bit, right)]; target = [value % prime for value, prime in zip(raw, modulus)]; quotient = torch.tensor([value // prime for value, prime in zip(raw, modulus)], dtype=torch.long)
            features = self.exp300.features_from_soft_state(state_plane, right, modulus, bit, width) if hasattr(self.exp300, "features_from_soft_state") else self._soft_features(state_plane, right, modulus, bit, width)
            loss, receipt, final_logits = self.traced_loss(model, features, target, quotient, width, kind, step, total_steps, seed + offset); losses.append(loss); selections.append(receipt["selected_depths"])
            probabilities = torch.sigmoid(final_logits); state_plane = final_logits.ge(0).float() + probabilities - probabilities.detach(); teacher = target
        return torch.stack(losses).mean(), {"selected_depths_by_recovery_step": selections, "loss_calls": chunk * int(self.config["window"]["size"])}

    def train_arm(self, model: nn.Module, spec: RootSpec, arm_name: str, steps: int, timer: Timer) -> dict[str, Any]:
        cfg = self.config["training"]; kind = self.config["arms"][arm_name]["kind"]
        random.seed(spec.train_seed); np.random.seed(spec.train_seed); torch.manual_seed(spec.train_seed)
        parameters = list(model.parameters()); optimizer = torch.optim.AdamW(parameters, lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, steps), eta_min=float(cfg["minimum_learning_rate"]))
        rng = random.Random(spec.train_seed + 7); transition = self.exp300.BigIntSampler(self.split(64, spec.root)["train"], spec.train_seed + 11); recovery = {width: self.exp300.BigIntSampler(self.split(width, spec.root)["train"], spec.train_seed + width) for width in (64, 128)}
        before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}; history = []; selection_counts: dict[str, int] = {}; model.train()
        for step in range(1, steps + 1):
            mode = cfg["mode_schedule"][(step - 1) % len(cfg["mode_schedule"])]
            if mode == "transition64":
                state, right, modulus, bit, target, quotient = transition.transitions(int(cfg["batch_size"])); features = self.exp300.features(state, right, modulus, bit, 64)
                loss, receipt, _ = self.traced_loss(model, features, target, quotient, 64, kind, step, steps, spec.train_seed); selected = receipt["selected_depths"]
            else:
                width = int(mode.removeprefix("recovery")); loss, receipt = self.recovery_loss(model, recovery[width], rng, width, kind, step, steps, spec.train_seed); selected = receipt["selected_depths_by_recovery_step"][0]
            for depth in selected: selection_counts[str(depth)] = selection_counts.get(str(depth), 0) + 1
            optimizer.zero_grad(set_to_none=True); loss.backward(); gradient = float(nn.utils.clip_grad_norm_(parameters, float(cfg["gradient_clip"]))); optimizer.step(); scheduler.step()
            if step == 1 or step % 50 == 0 or step == steps:
                timer.sample(); record = {"step": step, "mode": mode, "loss": float(loss.detach()), "gradient_norm": gradient, "selected_depths": selected}; history.append(record); print(json.dumps({"root": spec.root, "arm": arm_name, **record}), flush=True)
        model.eval(); changes = {name: float((model.state_dict()[name] - value).norm()) for name, value in before.items() if torch.is_floating_point(value)}
        return {"steps": steps, "examples_per_update": int(cfg["batch_size"]), "loss_calls_per_transition": int(self.config["window"]["size"]), "history": history, "selection_counts": selection_counts, "parameter_update_norm_sum": sum(changes.values()), "parameter_update_norms": changes}

    @torch.inference_mode()
    def evaluate_rollouts(self, model: nn.Module, sampler: Any, cases: int, width: int) -> dict[str, Any]:
        seen = correct = invalid = 0; vector: list[int] = []; first_errors: list[int | None] = []
        while seen < cases:
            size = min(64, cases - seen); multiplier, multiplicand, modulus = sampler.pairs(size); state = [0] * size; teacher = [0] * size; bad = [False] * size; first = [None] * size
            for position in range(width):
                bits = [(value >> (width - 1 - position)) & 1 for value in multiplier]; logits, _ = model(self.exp300.features(state, multiplicand, modulus, bits, width)); rows = logits.ge(0).tolist(); state = [sum(int(row[index]) << index for index in range(width)) for row in rows]
                teacher = [(2 * value + bit * other) % prime for value, bit, other, prime in zip(teacher, bits, multiplicand, modulus)]
                for row, (predicted, expected) in enumerate(zip(state, teacher)):
                    if predicted != expected and first[row] is None: first[row] = position
                bad = [prior or predicted >= prime for prior, predicted, prime in zip(bad, state, modulus)]
            expected = [(left * right) % prime for left, right, prime in zip(multiplier, multiplicand, modulus)]; batch_vector = [int(predicted == target) for predicted, target in zip(state, expected)]
            vector.extend(batch_vector); correct += sum(batch_vector); invalid += sum(bad); first_errors.extend(first); seen += size
        failures = [value for value in first_errors if value is not None]
        return {"cases": seen, "correct": correct, "exact_accuracy": correct / seen, "invalids": invalid, "any_invalid_residue_rate": invalid / seen, "mean_first_error_position": float(width) if not failures else float(np.mean(failures)), "exact_vector": vector}

    @torch.inference_mode()
    def late_depth_losses(self, model: nn.Module, spec: RootSpec, width: int, cases: int) -> dict[str, Any]:
        sampler = self.exp300.BigIntSampler(self.split(width, spec.root)["test"], spec.eval_seed + 4000 + width); state, right, modulus, bit, target, quotient = sampler.transitions(cases); outputs = trace_forward(model, self.exp300.features(state, right, modulus, bit, width)); losses = []
        for logits, qlogits in outputs:
            loss, _ = self.exp300.transition_loss(logits, qlogits, target, quotient, width, float(self.config["training"]["worst_bit_weight"]), float(self.config["training"]["quotient_weight"])); losses.append(float(loss))
        window = int(self.config["window"]["size"]); return {"cases": cases, "per_depth_loss": losses, "late_window_mean_loss": float(np.mean(losses[-window:])), "final_loss": losses[-1]}

    def inference_identity(self, model: nn.Module, checkpoint: Path, seed: int) -> dict[str, Any]:
        reloaded = self.base(); reloaded.load_state_dict(load_file(checkpoint)); reloaded.eval(); model.eval(); probe = torch.randn(int(self.config["evaluation"]["deterministic_repeat_cases"]), 6, 128, generator=torch.Generator().manual_seed(seed))
        with torch.no_grad(): ordinary = reloaded(probe); traced = trace_forward(reloaded, probe)[-1]; current = model(probe)
        return {"architecture": type(reloaded).__name__, "parameter_count": sum(p.numel() for p in reloaded.parameters()), "state_dict_identity": state_dict_identity(model, reloaded), "state_dict_keys": list(reloaded.state_dict()), "physical_calls": len(reloaded.dilation_schedule(128)), "ordinary_vs_trace_logits_equal": torch.equal(ordinary[0], traced[0]), "ordinary_vs_trace_quotient_equal": torch.equal(ordinary[1], traced[1]), "pre_save_vs_reload_logits_equal": torch.equal(current[0], ordinary[0]), "pre_save_vs_reload_quotient_equal": torch.equal(current[1], ordinary[1])}

    def evaluate_arm(self, model: nn.Module, spec: RootSpec, widths: list[int], transitions: dict[int, int], rollouts: dict[int, int], late_cases: int) -> tuple[dict[str, Any], dict[str, Any]]:
        evaluation: dict[str, Any] = {}; timers = []
        for width in widths:
            timer = Timer(); split = self.split(width, spec.root); transition = self.exp300.evaluate_transitions(model, self.exp300.BigIntSampler(split["test"], spec.eval_seed + 1000 + width), transitions[width]); rollout = self.evaluate_rollouts(model, self.exp300.BigIntSampler(split["test"], spec.eval_seed + 2000 + width), rollouts[width], width); late = self.late_depth_losses(model, spec, width, late_cases); runtime = timer.result(); timers.append(runtime); evaluation[str(width)] = {"transition": transition, "rollout": rollout, "late_depth": late, "runtime": runtime}
        return evaluation, aggregate(timers)

    def checkpoint_path(self, spec: RootSpec, arm: str) -> Path:
        return self.output / self.mode / f"root-{spec.root}" / f"{arm}.safetensors"

    def train_and_cache(self, spec: RootSpec, steps: int, session: Timer) -> tuple[dict[str, nn.Module], dict[str, Any]]:
        arms: dict[str, nn.Module] = {}; records: dict[str, Any] = {}
        for arm_name in self.config["arms"]:
            path = self.checkpoint_path(spec, arm_name); training_timer = Timer()
            if arm_name == "randomized_control": model = self.base(randomized=True, seed=spec.train_seed + 900)
            else: model = self.base()
            if arm_name in self.trained_arms:
                training = self.train_arm(model, spec, arm_name, steps, session)
            else:
                training = {"steps": 0, "reason": "frozen-source" if arm_name == "frozen_source" else "independently-randomized-weight-control", "loss_calls": 0}
            training["runtime"] = training_timer.result(); path.parent.mkdir(parents=True, exist_ok=True); save_file({key: value.detach().contiguous() for key, value in model.state_dict().items()}, path)
            identity = self.inference_identity(model, path, spec.eval_seed + len(arms)); checkpoint = {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}
            if not all(identity[key] for key in ("state_dict_identity", "ordinary_vs_trace_logits_equal", "ordinary_vs_trace_quotient_equal", "pre_save_vs_reload_logits_equal", "pre_save_vs_reload_quotient_equal")): raise RuntimeError(f"formal inference identity failed for {arm_name}")
            arms[arm_name] = model.eval(); records[arm_name] = {"training": training, "checkpoint": checkpoint, "inference_identity": identity, "parameters": identity["parameter_count"]}
            write_json(self.output / self.mode / f"root-{spec.root}-progress.json", {"schema_version": "exp676-root-progress-v1", "source": self.source_identity(), "root": spec.root, "steps": steps, "completed_arms": records})
        return arms, records

    def execute_root(self) -> dict[str, Any]:
        spec = self.spec(); cfg = self.config; calibration = self.mode == "calibration"; steps = int(cfg["training"]["calibration_steps" if calibration else "steps"])
        transitions = {int(k): int(v) for k, v in cfg["evaluation"]["calibration_transition_cases" if calibration else "transition_cases"].items()}; rollouts = {int(k): int(v) for k, v in cfg["evaluation"]["calibration_rollout_cases" if calibration else "rollout_cases"].items()}; late_cases = int(cfg["evaluation"]["calibration_late_depth_cases" if calibration else "late_depth_cases"])
        session = Timer(); arms, records = self.train_and_cache(spec, steps, session)
        reinjection = {str(width): reinjection_receipt(arms["moving_window"], int(width)) for width in cfg["evaluation"]["widths"]}
        if not all(value["gate_reinjects_every_visit"] and value["candidate_reinjects_every_visit"] for value in reinjection.values()): raise RuntimeError("Candidate005 reinjection equivalence failed")
        if not calibration:
            evaluation, runtime = self.evaluate_arm(arms["moving_window"], spec, [128], transitions, rollouts, late_cases); records["moving_window"]["evaluation"] = evaluation; records["moving_window"]["evaluation_runtime"] = runtime
            primary = evaluation["128"]["rollout"]
            if primary["exact_accuracy"] < float(cfg["gates"]["minimum_w128_exact"]) or primary["invalids"] > int(cfg["gates"]["maximum_invalids"]):
                return {"schema_version": "exp676-root-v1", "mode": self.mode, "root": spec.root, "source": self.source_identity(), "reinjection": reinjection, "arms": records, "resource": session.result(), "early_decision": "reject-W128-primary"}
        for arm_name, model in arms.items():
            widths = [int(width) for width in cfg["evaluation"]["widths"] if not (not calibration and arm_name == "moving_window" and int(width) == 128)]
            evaluation, runtime = self.evaluate_arm(model, spec, widths, transitions, rollouts, late_cases); records[arm_name].setdefault("evaluation", {}).update(evaluation); records[arm_name]["evaluation_runtime"] = runtime
        timing_timer = Timer(); timing_width = int(cfg["evaluation"]["candidate_timing_width"]); timing_cases = int(cfg["evaluation"]["candidate_timing_cases"]); timing_split = self.split(timing_width, spec.root); timing = self.evaluate_rollouts(arms["moving_window"], self.exp300.BigIntSampler(timing_split["test"], spec.eval_seed + 9000), timing_cases, timing_width); timing_resource = timing_timer.result()
        result = {"schema_version": "exp676-root-v1", "mode": self.mode, "root": spec.root, "source": self.source_identity(), "reinjection": reinjection, "arms": records, "candidate_timing": {"width": timing_width, "cases": timing_cases, "rollout": timing, "resource": timing_resource}, "resource": session.result()}
        write_json(self.output / self.mode / f"root-{spec.root}-summary.json", result); return result

    def calibration(self) -> dict[str, Any]:
        self.assert_frozen_clean(); root = self.execute_root(); cfg = self.config; measured = root["resource"]
        training_resources = [root["arms"][name]["training"]["runtime"] for name in self.trained_arms]; evaluation_resources = [root["arms"][name]["evaluation_runtime"] for name in cfg["arms"]]
        training = aggregate(training_resources); evaluation = aggregate(evaluation_resources); training_scale = int(cfg["training"]["steps"]) / int(cfg["training"]["calibration_steps"]); evaluation_scale = max(sum(int(v) for v in cfg["evaluation"]["transition_cases"].values()) / sum(int(v) for v in cfg["evaluation"]["calibration_transition_cases"].values()), sum(int(v) for v in cfg["evaluation"]["rollout_cases"].values()) / sum(int(v) for v in cfg["evaluation"]["calibration_rollout_cases"].values()))
        headroom = float(cfg["resource_budget"]["formal_headroom"]); expected = {}
        for key in ("wall_seconds", "process_tree_cpu_seconds", "cgroup_cpu_seconds"): expected[key] = headroom * (float(training[key]) * training_scale + float(evaluation[key]) * evaluation_scale)
        expected["peak_process_tree_rss_bytes"] = int(measured["peak_process_tree_rss_bytes"] * float(cfg["resource_budget"]["rss_headroom"])); soft = {key: value * float(cfg["resource_budget"]["soft_multiplier"]) for key, value in expected.items()}; hard = {key: value * float(cfg["resource_budget"]["hard_multiplier"]) for key, value in expected.items()}
        timing = root["candidate_timing"]; nominal = float(timing["resource"]["wall_seconds"]) / int(timing["cases"]) * 300; arm_walls = [float(root["arms"][name]["evaluation_runtime"]["wall_seconds"]) for name in cfg["arms"]]; variability = max(0.10, max(arm_walls) / float(np.mean(arm_walls)) - 1.0); guarded = nominal * max(headroom, 1.0 + variability)
        artifact = sum(int(value["checkpoint"]["bytes"]) for value in root["arms"].values()); identities = all(all(value["inference_identity"][key] for key in ("state_dict_identity", "ordinary_vs_trace_logits_equal", "ordinary_vs_trace_quotient_equal", "pre_save_vs_reload_logits_equal", "pre_save_vs_reload_quotient_equal")) for value in root["arms"].values())
        checks = {"complete_route": set(root["arms"]) == set(cfg["arms"]), "reinjection_equivalence": all(value["gate_reinjects_every_visit"] and value["candidate_reinjects_every_visit"] for value in root["reinjection"].values()), "formal_inference_identity": identities, "candidate_runtime": guarded <= float(cfg["gates"]["maximum_guarded_candidate_seconds"]), "artifact": artifact < int(cfg["gates"]["maximum_artifact_bytes"]), "no_fidelity_reduction": cfg["resource_budget"]["fidelity_reduced"] is False}
        decision = "admit-root-20261255" if all(checks.values()) else "reject-formal-at-calibration"; summary_path = self.output / self.mode / f"root-{self.spec().root}-summary.json"
        admission = {"schema_version": "exp676-dynamic-admission-v1", "decision": decision, "source": root["source"], "calibration_summary": str(summary_path.relative_to(ROOT)), "calibration_sha256": sha256(summary_path), "measured": measured, "training_component": training, "evaluation_component": evaluation, "training_scale": training_scale, "evaluation_scale": evaluation_scale, "variability": variability, "formal_eta_seconds": expected["wall_seconds"], "expected": expected, "soft": soft, "hard": hard, "candidate_nominal_seconds": nominal, "candidate_guarded_seconds": guarded, "artifact_bytes": artifact, "checks": checks, "fidelity_reduced": False}
        write_json(ROOT / cfg["resource_budget"]["admission_path"], admission); write_json(self.output / "calibration.json", admission); return admission

    def formal_gates(self, root: dict[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
        cfg = self.config; gates = cfg["gates"]; arms = root["arms"]; primary = arms["moving_window"]; controls = ("endpoint_only", "fixed_window", "equal_work_multi_endpoint")
        w128 = primary["evaluation"]["128"]["rollout"]; w256 = primary["evaluation"]["256"]["rollout"]; endpoint128 = arms["endpoint_only"]["evaluation"]["128"]["rollout"]; source128 = arms["frozen_source"]["evaluation"]["128"]["rollout"]
        late = {name: arms[name]["evaluation"]["128"]["late_depth"]["late_window_mean_loss"] for name in ("moving_window", *controls)}; first = {name: arms[name]["evaluation"]["128"]["rollout"]["mean_first_error_position"] for name in ("moving_window", *controls)}
        comparisons = {"late_depth_losses": late, "mean_first_error_positions": first, "w256_exact": {name: arms[name]["evaluation"]["256"]["rollout"]["exact_accuracy"] for name in arms}, "w128_exact": {name: arms[name]["evaluation"]["128"]["rollout"]["exact_accuracy"] for name in arms}}
        checks = {"w128": w128["exact_accuracy"] >= float(gates["minimum_w128_exact"]), "validity": all(arms[name]["evaluation"][str(width)]["rollout"]["invalids"] == 0 for name in arms for width in cfg["evaluation"]["widths"]), "endpoint_retention": w128["exact_accuracy"] >= endpoint128["exact_accuracy"] - float(gates["maximum_w128_loss_vs_endpoint"]), "source_retention": w128["exact_accuracy"] >= source128["exact_accuracy"] - float(gates["maximum_w128_loss_vs_source"]), "late_depth_loss": all(late["moving_window"] < late[name] - float(gates["minimum_late_depth_loss_advantage"]) for name in controls), "first_error": all(first["moving_window"] > first[name] + float(gates["minimum_first_error_advantage"]) for name in controls), "trained_dependence": w128["exact_accuracy"] - arms["randomized_control"]["evaluation"]["128"]["rollout"]["exact_accuracy"] >= float(gates["minimum_trained_minus_randomized_w128"]), "w256_absolute": w256["exact_accuracy"] >= float(gates["minimum_w256_exact"]), "w256_controls": all(w256["exact_accuracy"] - arms[name]["evaluation"]["256"]["rollout"]["exact_accuracy"] >= float(gates["minimum_w256_gain_vs_controls"]) for name in (*controls, "shuffled_window")), "formal_inference_identity": all(all(value["inference_identity"][key] for key in ("state_dict_identity", "ordinary_vs_trace_logits_equal", "ordinary_vs_trace_quotient_equal", "pre_save_vs_reload_logits_equal", "pre_save_vs_reload_quotient_equal")) for value in arms.values())}
        return checks, comparisons

    def load_admission(self) -> dict[str, Any]:
        path = ROOT / self.config["resource_budget"]["admission_path"]; admission = json.loads(path.read_text(encoding="utf-8")); calibration = ROOT / admission["calibration_summary"]
        if admission["decision"] != "admit-root-20261255" or admission["source"] != self.source_identity() or sha256(calibration) != admission["calibration_sha256"] or admission["fidelity_reduced"]: raise RuntimeError("formal requires matching admitted calibration")
        return admission

    def persist_terminal(self, root: dict[str, Any], decision: str, checks: dict[str, bool], comparisons: dict[str, Any] | None, admission: dict[str, Any]) -> dict[str, Any]:
        summary = {"schema_version": "exp676-terminal-v1", "run_id": self.config["run_id"], "status": "completed", "decision": decision, "root": root["root"], "source": root["source"], "reinjection": root["reinjection"], "arms": root["arms"], "candidate_timing": root.get("candidate_timing"), "resource": root["resource"], "gates": checks, "comparisons": comparisons, "dynamic_admission": admission, "limitations": ["Fresh legal generated identities only; no checked, independent, official/private, candidate, submission or promotion evaluation.", "Moving-window targets and endpoint selection are training/evaluation-only; formal inference is the original SharedDilatedConvGRU.", "M004.b remains closed."]}
        write_json(self.output / "terminal-summary.json", summary); write_json(ROOT / "experiments/runs/exp-676-lla-moving-window/run.json", summary); write_json(ROOT / "evals/results/exp-676-lla-moving-window/metrics.json", summary); return summary

    def formal(self) -> dict[str, Any]:
        self.assert_frozen_clean(); admission = self.load_admission(); root = self.execute_root()
        if root.get("early_decision"):
            return self.persist_terminal(root, "reject-exp676-at-W128-primary", {"w128_primary": False}, None, admission)
        checks, comparisons = self.formal_gates(root); decision = "authorize-confirmations" if all(checks.values()) else "reject-exp676-lla-moving-window"; return self.persist_terminal(root, decision, checks, comparisons, admission)

    def confirmation(self) -> dict[str, Any]:
        self.assert_frozen_clean(); root = self.execute_root(); checks, comparisons = self.formal_gates(root); admission = self.load_admission(); decision = "confirmation-pass" if all(checks.values()) else "confirmation-reject"; return self.persist_terminal(root, decision, checks, comparisons, admission)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=ROOT / "configs/exp_676_lla_moving_window.yaml"); parser.add_argument("--output", type=Path, default=ROOT / ".scratch/exp-676-lla-moving-window"); parser.add_argument("--mode", choices=("calibration", "formal", "confirmation"), required=True); parser.add_argument("--root", type=int); args = parser.parse_args()
    experiment = Experiment(args.config, args.output, args.mode, args.root); result = experiment.calibration() if args.mode == "calibration" else experiment.formal() if args.mode == "formal" else experiment.confirmation()
    experiment_id = configure_mlflow(experiment.config["tracking"]["experiment_name"])
    with mlflow.start_run(run_name=f"{experiment.config['run_id']}-{args.mode}-{experiment.spec().root}") as run:
        mlflow.set_tags({"canonical_run_id": experiment.config["run_id"], "mode": args.mode, "root": str(experiment.spec().root), "decision": str(result.get("decision")), "repository_commit": git_output("rev-parse", "HEAD")}); mlflow.log_dict(result, f"{args.mode}-result.json"); mlflow_run_id = run.info.run_id
    write_json(experiment.output / f"{args.mode}-mlflow.json", {"experiment_id": experiment_id, "experiment_name": experiment.config["tracking"]["experiment_name"], "run_id": mlflow_run_id, "tracking_uri": tracking_uri()})
    print(json.dumps({"mode": args.mode, "root": experiment.spec().root, "decision": result.get("decision"), "resource": result.get("resource", result.get("measured")), "gates": result.get("gates", result.get("checks")), "mlflow_run_id": mlflow_run_id}, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
