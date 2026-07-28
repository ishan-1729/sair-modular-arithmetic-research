"""Execute EXP675 teacher -> residual SVD -> joint RRT uptraining."""

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

from experiments.exp_675_rrt_svd_teacher_uptraining.model import RRTVisitLoRA, UntiedVisitTeacher, clone_state  # noqa: E402
from experiments.exp_655_layer_loop_compute_matched_recurrence.fixtures import verified_copy  # noqa: E402
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
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
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
    if not path.is_file():
        return None
    values = dict(line.split() for line in path.read_text(encoding="utf-8").splitlines())
    return float(values["usage_usec"]) / 1_000_000 if "usage_usec" in values else None


class Timer:
    """Wall, process-tree CPU, cgroup CPU and process-tree RSS measurement."""

    def __init__(self) -> None:
        self.wall = time.perf_counter()
        self.cgroup = cgroup_cpu_seconds()
        self.process = psutil.Process()
        self.initial_cpu = self._tree_cpu()
        self.peak_rss = self._tree_rss()

    def _tree(self) -> list[psutil.Process]:
        try:
            return [self.process, *self.process.children(recursive=True)]
        except psutil.Error:
            return [self.process]

    def _tree_cpu(self) -> float:
        total = 0.0
        for process in self._tree():
            try:
                value = process.cpu_times(); total += value.user + value.system
            except psutil.Error:
                pass
        return total

    def _tree_rss(self) -> int:
        total = 0
        for process in self._tree():
            try: total += process.memory_info().rss
            except psutil.Error: pass
        return total

    def sample(self) -> None:
        self.peak_rss = max(self.peak_rss, self._tree_rss())

    def result(self) -> dict[str, float | int | None]:
        self.sample()
        current = cgroup_cpu_seconds()
        return {
            "wall_seconds": time.perf_counter() - self.wall,
            "process_tree_cpu_seconds": self._tree_cpu() - self.initial_cpu,
            "cgroup_cpu_seconds": None if current is None or self.cgroup is None else current - self.cgroup,
            "peak_process_tree_rss_bytes": self.peak_rss,
        }


def aggregate(resources: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for key in ("wall_seconds", "process_tree_cpu_seconds", "cgroup_cpu_seconds"):
        values = [item[key] for item in resources]
        result[key] = None if any(value is None for value in values) else sum(float(value) for value in values)
    result["peak_process_tree_rss_bytes"] = max(int(item["peak_process_tree_rss_bytes"]) for item in resources)
    return result


class Experiment:
    def __init__(self, config_path: Path, output: Path, mode: str) -> None:
        self.config_path = config_path.resolve()
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.output = output.resolve(); self.output.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.exp300 = load_module("exp300_for_exp675", ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py")
        self.exp329 = load_module("exp329_for_exp675", ROOT / "experiments/exp_329_cellular_width64_autonomous_state_recovery/run.py")
        self.cellular = load_module("cellular_for_exp675", ROOT / "experiments/exp_220_width_independent_cellular/model.py")
        torch.set_num_threads(int(self.config["training"]["torch_threads"]))
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        self.checkpoint = self.materialize_checkpoint()

    def materialize_checkpoint(self) -> Path:
        destination = self.output / "fixtures/exp609-weights.safetensors"
        expected = self.config["source_contract"]["checkpoint_sha256"]
        candidates = [
            ROOT / ".scratch/exp-609-tier6-straight-through-confirmation/straight_through_continuation/weights.safetensors",
            Path("/mnt/d/Projects/SAIR_mod_challenge/.scratch/exp-609-tier6-straight-through-confirmation/straight_through_continuation/weights.safetensors"),
            Path("/mnt/d/Projects/SAIR_mod_challenge_exp655_t048_wt/.scratch/exp-609-tier6-straight-through-confirmation/straight_through_continuation/weights.safetensors"),
        ]
        source = next((path for path in candidates if path.is_file() and sha256(path) == expected), None)
        if source is None:
            raise RuntimeError("authenticated EXP609 checkpoint unavailable")
        verified_copy(source, destination, expected)
        return destination

    def source_identity(self) -> dict[str, str]:
        paths = {
            "config": self.config_path,
            "model": Path(__file__).with_name("model.py"),
            "runner": Path(__file__),
            "test": ROOT / "tests/test_exp_675_rrt_svd_teacher_uptraining.py",
            "preregistration": ROOT / self.config["preregistration"],
            "paper": ROOT / self.config["source_contract"]["paper_path"],
            "checkpoint": self.checkpoint,
        }
        return {"repository_commit": git_output("rev-parse", "HEAD"), **{f"{key}_sha256": sha256(path) for key, path in paths.items()}}

    def assert_frozen_clean(self) -> None:
        paths = [
            str(self.config_path.relative_to(ROOT)),
            "experiments/exp_675_rrt_svd_teacher_uptraining",
            "tests/test_exp_675_rrt_svd_teacher_uptraining.py",
            self.config["preregistration"],
        ]
        if git_output("status", "--porcelain", "--", *paths):
            raise RuntimeError("execution requires committed frozen EXP675 bytes")

    def spec(self) -> RootSpec:
        values = self.config["data"]["calibration" if self.mode == "calibration" else "formal"]
        return RootSpec(**values)

    def split(self, width: int, root: int) -> dict[str, list[int]]:
        local = {"data": {
            "prime_bits_min": width - int(self.config["data"]["prime_bit_span"]) + 1,
            "prime_bits_max": width,
            "train_prime_identities": int(self.config["data"]["train_prime_identities"]),
            "test_prime_identities": int(self.config["data"]["test_prime_identities"]),
        }}
        return self.exp300.bigint_prime_split(local, int(self.config["data"]["split_seed_offset"]) + root + width)

    def base(self) -> nn.Module:
        values = self.config["model"]
        model = self.cellular.SharedDilatedConvGRU(width=128, hidden=int(values["hidden"]), cycles=int(values["cycles"]))
        model.load_state_dict(load_file(self.checkpoint))
        return model

    def teacher(self) -> UntiedVisitTeacher:
        return UntiedVisitTeacher(self.base(), max_dilation_levels=int(self.config["model"]["max_dilation_levels"]))

    def rrt(self, base: nn.Module | None = None, *, kind: str = "visit") -> RRTVisitLoRA:
        values = self.config["model"]
        return RRTVisitLoRA(base or self.base(), rank=int(values["rank"]), alpha=float(values["alpha"]), max_dilation_levels=int(values["max_dilation_levels"]), kind=kind)

    def centroid_base(self, teacher: UntiedVisitTeacher) -> nn.Module:
        base = self.base()
        with torch.no_grad():
            base.gate_weight.copy_(teacher.gate_weights.mean(0))
            base.candidate_weight.copy_(teacher.candidate_weights.mean(0))
        return base

    def recovery_loss(self, model: nn.Module, sampler: Any, rng: random.Random, width: int) -> torch.Tensor:
        training = self.config["training"]
        batch, chunk = int(training["recovery_batch_size"]), int(training["recovery_chunk_steps"])
        multiplier, right, modulus = sampler.pairs(batch)
        starts = [rng.randrange(0, width - chunk + 1) for _ in range(batch)]
        teacher_states = self.exp329.teacher_states_before(multiplier, right, modulus, starts, width)
        state_plane = self.exp300.bits(teacher_states, width); losses = []
        for offset in range(chunk):
            positions = [start + offset for start in starts]
            bit = [(left >> (width - 1 - position)) & 1 for left, position in zip(multiplier, positions)]
            raw = [2 * state + flag * value for state, flag, value in zip(teacher_states, bit, right)]
            target = [value % prime for value, prime in zip(raw, modulus)]
            quotient = torch.tensor([value // prime for value, prime in zip(raw, modulus)], dtype=torch.long)
            features = self.exp300.features_from_soft_state(state_plane, right, modulus, bit, width)
            logits, qlogits = model(features)
            loss, _ = self.exp300.transition_loss(logits, qlogits, target, quotient, width, float(training["worst_bit_weight"]), 0.0)
            losses.append(loss)
            probabilities = torch.sigmoid(logits)
            state_plane = logits.ge(0).float() + probabilities - probabilities.detach()
            teacher_states = target
        return torch.stack(losses).mean()

    def train(self, model: nn.Module, spec: RootSpec, *, steps: int, seed_offset: int, label: str, timer: Timer) -> dict[str, Any]:
        random.seed(spec.train_seed + seed_offset); np.random.seed(spec.train_seed + seed_offset); torch.manual_seed(spec.train_seed + seed_offset)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        cfg = self.config["training"]
        optimizer = torch.optim.AdamW(parameters, lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, steps), eta_min=float(cfg["minimum_learning_rate"]))
        rng = random.Random(spec.train_seed + 7)
        transition = self.exp300.BigIntSampler(self.split(64, spec.root)["train"], spec.train_seed + 11)
        recovery = {width: self.exp300.BigIntSampler(self.split(width, spec.root)["train"], spec.train_seed + width) for width in (64, 128)}
        before = clone_state(model); history = []
        model.train()
        for step in range(1, steps + 1):
            mode = cfg["mode_schedule"][(step - 1) % len(cfg["mode_schedule"])]
            if mode == "transition64":
                state, right, modulus, bit, target, quotient = transition.transitions(int(cfg["batch_size"]))
                logits, qlogits = model(self.exp300.features(state, right, modulus, bit, 64))
                loss, _ = self.exp300.transition_loss(logits, qlogits, target, quotient, 64, float(cfg["worst_bit_weight"]), float(cfg["quotient_weight"]))
            else:
                width = int(mode.removeprefix("recovery")); loss = self.recovery_loss(model, recovery[width], rng, width)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            gradient = float(nn.utils.clip_grad_norm_(parameters, float(cfg["gradient_clip"])))
            optimizer.step(); scheduler.step()
            if step == 1 or step % 50 == 0 or step == steps:
                timer.sample(); record = {"step": step, "mode": mode, "loss": float(loss.detach()), "gradient_norm": gradient}; history.append(record)
                print(json.dumps({"root": spec.root, "stage": label, **record}), flush=True)
        model.eval()
        changed = {name: float((model.state_dict()[name] - value).norm()) for name, value in before.items() if torch.is_floating_point(value)}
        return {"steps": steps, "examples_per_transition_update": int(cfg["batch_size"]), "history": history, "parameter_update_norms": changed}

    @torch.inference_mode()
    def evaluate_rollouts(self, model: nn.Module, sampler: Any, cases: int, width: int, **kwargs: Any) -> dict[str, Any]:
        seen = correct = invalid = 0; vector: list[int] = []; first_errors: list[int | None] = []
        while seen < cases:
            size = min(64, cases - seen); multiplier, multiplicand, modulus = sampler.pairs(size)
            state = [0] * size; teacher = [0] * size; bad = [False] * size; first = [None] * size
            for position in range(width):
                bits = [(value >> (width - 1 - position)) & 1 for value in multiplier]
                logits, _ = model(self.exp300.features(state, multiplicand, modulus, bits, width), **kwargs)
                rows = logits.ge(0).tolist(); state = [sum(int(row[index]) << index for index in range(width)) for row in rows]
                teacher = [(2 * value + bit * other) % prime for value, bit, other, prime in zip(teacher, bits, multiplicand, modulus)]
                for row, (predicted, expected) in enumerate(zip(state, teacher)):
                    if predicted != expected and first[row] is None: first[row] = position
                bad = [prior or predicted >= prime for prior, predicted, prime in zip(bad, state, modulus)]
            expected = [(left * right) % prime for left, right, prime in zip(multiplier, multiplicand, modulus)]
            batch_vector = [int(predicted == target) for predicted, target in zip(state, expected)]
            vector.extend(batch_vector); correct += sum(batch_vector); invalid += sum(bad); first_errors.extend(first); seen += size
        failures = [value for value in first_errors if value is not None]
        return {"cases": seen, "correct": correct, "exact_accuracy": correct / seen, "invalids": invalid, "any_invalid_residue_rate": invalid / seen, "mean_first_error_position": None if not failures else float(np.mean(failures)), "exact_vector": vector}

    def build_and_train(self, spec: RootSpec, teacher_steps: int, joint_steps: int, timer: Timer) -> tuple[dict[str, nn.Module], dict[str, Any]]:
        teacher = self.teacher()
        if not teacher.exact_copy_identity(): raise RuntimeError("teacher copy identity failed")
        parity_features = torch.randn(4, 6, 64, generator=torch.Generator().manual_seed(675))
        base = self.base().eval(); teacher.eval()
        with torch.no_grad():
            base_parity = base(parity_features); teacher_parity = teacher(parity_features)
        copy_receipt = {"weights_equal": teacher.exact_copy_identity(), "logits_equal": torch.equal(base_parity[0], teacher_parity[0]), "quotient_equal": torch.equal(base_parity[1], teacher_parity[1])}
        if not all(copy_receipt.values()): raise RuntimeError("exact teacher-copy parity failed")
        stages: dict[str, Any] = {"teacher_copy": copy_receipt}
        stages["teacher_continuation"] = self.train(teacher, spec, steps=teacher_steps, seed_offset=0, label="teacher", timer=timer)
        centroid = self.centroid_base(teacher)
        full = self.rrt(centroid); svd_report = full.initialize_from_teacher(teacher)
        random_control = self.rrt(centroid); random_control.initialize_equal_norm_random(teacher, seed=spec.train_seed + 91)
        shuffled = self.rrt(centroid); shuffled.initialize_equal_norm_random(teacher, seed=spec.train_seed + 92, shuffle_slots=True)
        exact_zero = self.rrt(self.base()); exact_zero.zero_adapters()
        global_lora = self.rrt(self.base(), kind="global"); global_lora.zero_adapters()
        shared = self.base()
        initial_errors = {
            "svd": full.reconstruction_error(teacher),
            "equal_norm_random": random_control.reconstruction_error(teacher),
            "subspace_shuffled": shuffled.reconstruction_error(teacher),
        }
        arms: dict[str, nn.Module] = {
            "full_rrt": full,
            "exact_zero_visit": exact_zero,
            "global_lora": global_lora,
            "untied_teacher": teacher,
            "shared_continuation": shared,
            "equal_norm_random_svd": random_control,
            "subspace_shuffled": shuffled,
        }
        training: dict[str, Any] = {}
        # Conversion teacher sees phase one. Controls see the same phase-one and
        # phase-two legal streams; converted RRTs receive phase-two joint updates.
        for index, (name, model) in enumerate(arms.items()):
            if name == "untied_teacher":
                training[name] = {"phase_one": stages["teacher_continuation"]}
            elif name in ("full_rrt", "equal_norm_random_svd", "subspace_shuffled"):
                training[name] = {"phase_one": "teacher-conversion-source"}
            else:
                training[name] = {"phase_one": self.train(model, spec, steps=teacher_steps, seed_offset=0, label=f"{name}-phase1", timer=timer)}
            training[name]["phase_two"] = self.train(model, spec, steps=joint_steps, seed_offset=1000, label=f"{name}-phase2", timer=timer)
        stages.update({"svd": svd_report, "initial_reconstruction": initial_errors, "training": training})
        return arms, stages

    def save_arm(self, model: nn.Module, spec: RootSpec, arm_name: str) -> dict[str, Any]:
        path = self.output / self.mode / f"root-{spec.root}" / f"{arm_name}.safetensors"; path.parent.mkdir(parents=True, exist_ok=True)
        save_file({name: tensor.detach().contiguous() for name, tensor in model.state_dict().items()}, path)
        return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}

    def evaluate(self, model: nn.Module, spec: RootSpec, arm_name: str, widths: list[int], transition_cases: dict[int, int], rollout_cases: dict[int, int]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for width in widths:
            timer = Timer(); split = self.split(width, spec.root)
            transition = self.exp300.evaluate_transitions(model, self.exp300.BigIntSampler(split["test"], spec.eval_seed + 1000 + width), transition_cases[width])
            rollout = self.evaluate_rollouts(model, self.exp300.BigIntSampler(split["test"], spec.eval_seed + 2000 + width), rollout_cases[width], width)
            result[str(width)] = {"transition": transition, "rollout": rollout, "runtime": timer.result()}
        return result

    def interventions(self, model: RRTVisitLoRA, spec: RootSpec, cases: int) -> dict[str, Any]:
        width = 128; split = self.split(width, spec.root); seed = spec.eval_seed + 3000
        def run(candidate: RRTVisitLoRA, **kwargs: Any) -> dict[str, Any]:
            return self.evaluate_rollouts(candidate, self.exp300.BigIntSampler(split["test"], seed), cases, width, **kwargs)
        ordinary = run(model); zeroed = run(model, adapter_enabled=False)
        permuted = run(model, slot_permutation=tuple(reversed(range(model.slot_count))))
        randomized = copy.deepcopy(model); generator = torch.Generator().manual_seed(spec.eval_seed + 31)
        with torch.no_grad():
            for parameter in randomized.adapter_parameters():
                order = torch.randperm(parameter.numel(), generator=generator); parameter.flatten().copy_(parameter.flatten()[order])
        random_eval = run(randomized)
        return {
            "ordinary_exact": ordinary["exact_accuracy"], "zeroed_exact": zeroed["exact_accuracy"], "permuted_exact": permuted["exact_accuracy"], "randomized_exact": random_eval["exact_accuracy"],
            "zeroed_changed": sum(a != b for a, b in zip(ordinary["exact_vector"], zeroed["exact_vector"])),
            "permuted_changed": sum(a != b for a, b in zip(ordinary["exact_vector"], permuted["exact_vector"])),
            "randomized_changed": sum(a != b for a, b in zip(ordinary["exact_vector"], random_eval["exact_vector"])),
        }

    def execute_root(self) -> dict[str, Any]:
        spec = self.spec(); cfg = self.config
        teacher_steps = int(cfg["training"]["calibration_teacher_steps" if self.mode == "calibration" else "teacher_steps"])
        joint_steps = int(cfg["training"]["calibration_joint_steps" if self.mode == "calibration" else "joint_steps"])
        transitions = {int(k): int(v) for k, v in cfg["evaluation"]["calibration_transition_cases" if self.mode == "calibration" else "transition_cases"].items()}
        rollouts = {int(k): int(v) for k, v in cfg["evaluation"]["calibration_rollout_cases" if self.mode == "calibration" else "rollout_cases"].items()}
        session = Timer(); arms, stages = self.build_and_train(spec, teacher_steps, joint_steps, session)
        records: dict[str, Any] = {}
        # Every arm is persisted before potentially decisive evaluation.
        for name, model in arms.items():
            records[name] = {"checkpoint": self.save_arm(model, spec, name), "parameters": sum(p.numel() for p in model.parameters())}
        if self.mode == "formal":
            records["full_rrt"]["evaluation"] = self.evaluate(arms["full_rrt"], spec, "full_rrt", [128], transitions, rollouts)
            primary = records["full_rrt"]["evaluation"]["128"]["rollout"]
            if primary["exact_accuracy"] < float(cfg["gates"]["minimum_w128_exact"]) or primary["invalids"] > int(cfg["gates"]["maximum_invalids"]):
                return self.persist_terminal(spec, records, stages, session.result(), "reject-w128-primary-before-conditional-controls", {"w128_primary": False})
        for name, model in arms.items():
            already = [128] if self.mode == "formal" and name == "full_rrt" else []
            widths = [width for width in cfg["evaluation"]["widths"] if int(width) not in already]
            observed = self.evaluate(model, spec, name, [int(width) for width in widths], transitions, rollouts)
            records[name].setdefault("evaluation", {}).update(observed)
        intervention_cases = int(cfg["evaluation"]["calibration_intervention_cases" if self.mode == "calibration" else "intervention_cases"])
        records["full_rrt"]["interventions"] = self.interventions(arms["full_rrt"], spec, intervention_cases)
        resource = session.result()
        summary = {"schema_version": "exp675-complete-route-v1", "mode": self.mode, "root": spec.root, "source": self.source_identity(), "stages": stages, "arms": records, "resource": resource}
        write_json(self.output / self.mode / f"root-{spec.root}-summary.json", summary)
        return summary

    def calibration(self) -> dict[str, Any]:
        self.assert_frozen_clean(); summary = self.execute_root(); cfg = self.config
        measured = summary["resource"]; train_scale = (int(cfg["training"]["teacher_steps"]) + int(cfg["training"]["joint_steps"])) / max(1, int(cfg["training"]["calibration_teacher_steps"]) + int(cfg["training"]["calibration_joint_steps"]))
        eval_scale = sum(int(v) for v in cfg["evaluation"]["rollout_cases"].values()) / sum(int(v) for v in cfg["evaluation"]["calibration_rollout_cases"].values())
        scale = max(train_scale, eval_scale); headroom = float(cfg["resource_budget"]["formal_headroom"])
        expected = {key: float(measured[key]) * scale * headroom for key in ("wall_seconds", "process_tree_cpu_seconds", "cgroup_cpu_seconds")}
        expected["peak_process_tree_rss_bytes"] = int(measured["peak_process_tree_rss_bytes"] * float(cfg["resource_budget"]["rss_headroom"]))
        soft = {key: value * float(cfg["resource_budget"]["soft_multiplier"]) for key, value in expected.items()}
        hard = {key: value * float(cfg["resource_budget"]["hard_multiplier"]) for key, value in expected.items()}
        candidate = summary["arms"]["full_rrt"]["evaluation"]["256"]["runtime"]["wall_seconds"] / int(cfg["evaluation"]["calibration_rollout_cases"][256]) * 300 * headroom
        artifact = sum(arm["checkpoint"]["bytes"] for arm in summary["arms"].values())
        variability = max(0.10, max(arm["evaluation"]["128"]["runtime"]["wall_seconds"] for arm in summary["arms"].values()) / np.mean([arm["evaluation"]["128"]["runtime"]["wall_seconds"] for arm in summary["arms"].values()]) - 1.0)
        checks = {"complete_route": set(summary["arms"]) == set(cfg["arms"]), "candidate_runtime": candidate <= float(cfg["gates"]["maximum_guarded_candidate_seconds"]), "artifact": artifact < int(cfg["gates"]["maximum_artifact_bytes"]), "fidelity_reduced": cfg["resource_budget"]["fidelity_reduced"] is False}
        decision = "admit-root-20261251" if all(checks.values()) else "reject-formal-at-calibration"
        admission = {"schema_version": "exp675-dynamic-admission-v1", "decision": decision, "source": summary["source"], "calibration_summary": str((self.output / self.mode / f"root-{self.spec().root}-summary.json").relative_to(ROOT)), "measured": measured, "projection_scale": scale, "variability": variability, "expected": expected, "soft": soft, "hard": hard, "guarded_candidate_seconds": candidate, "artifact_bytes": artifact, "checks": checks, "formal_eta_seconds": expected["wall_seconds"]}
        write_json(ROOT / cfg["resource_budget"]["admission_path"], admission); write_json(self.output / "calibration.json", admission)
        return admission

    def persist_terminal(self, spec: RootSpec, arms: dict[str, Any], stages: dict[str, Any], resource: dict[str, Any], decision: str, gates: dict[str, bool]) -> dict[str, Any]:
        summary = {"schema_version": "exp675-terminal-v1", "status": "completed", "run_id": self.config["run_id"], "root": spec.root, "decision": decision, "source": self.source_identity(), "stages": stages, "arms": arms, "resource": resource, "gates": gates, "limitations": ["Fresh legal generated identities only; no checked, independent, official/private, candidate, submission or promotion evaluation.", "Formal inference is learned shared base plus fixed visit adapters; teacher and exact offline labels are excluded."]}
        write_json(self.output / self.mode / f"root-{spec.root}-summary.json", summary)
        if self.mode == "formal":
            write_json(ROOT / "experiments/runs/exp-675-rrt-svd-teacher-uptraining/run.json", summary)
            write_json(ROOT / "evals/results/exp-675-rrt-svd-teacher-uptraining/metrics.json", summary)
        return summary

    def formal(self) -> dict[str, Any]:
        self.assert_frozen_clean(); admission_path = ROOT / self.config["resource_budget"]["admission_path"]
        admission = json.loads(admission_path.read_text(encoding="utf-8"))
        if admission["decision"] != "admit-root-20261251" or admission["source"] != self.source_identity(): raise RuntimeError("formal requires matching admitted calibration")
        summary = self.execute_root()
        if summary.get("status") == "completed": return summary
        arms, gates_cfg = summary["arms"], self.config["gates"]
        primary = arms["full_rrt"]; source = arms["shared_continuation"]
        reconstruction = summary["stages"]["initial_reconstruction"]
        svd_error = np.mean(reconstruction["svd"]["gate"] + reconstruction["svd"]["candidate"])
        random_error = np.mean(reconstruction["equal_norm_random"]["gate"] + reconstruction["equal_norm_random"]["candidate"])
        controls = ("exact_zero_visit", "global_lora", "shared_continuation")
        w256 = primary["evaluation"]["256"]["rollout"]; w128 = primary["evaluation"]["128"]["rollout"]; w64 = primary["evaluation"]["64"]["rollout"]
        teacher128 = arms["untied_teacher"]["evaluation"]["128"]["rollout"]["exact_accuracy"]
        interventions = primary["interventions"]
        artifact = sum(arm["checkpoint"]["bytes"] for arm in arms.values())
        candidate = primary["evaluation"]["256"]["runtime"]["wall_seconds"] / int(self.config["evaluation"]["rollout_cases"][256]) * 300 * float(self.config["resource_budget"]["formal_headroom"])
        gates = {
            "w128": w128["exact_accuracy"] >= float(gates_cfg["minimum_w128_exact"]), "validity": w128["invalids"] == w64["invalids"] == w256["invalids"] == 0,
            "w64_retention": w64["exact_accuracy"] - source["evaluation"]["64"]["rollout"]["exact_accuracy"] >= float(gates_cfg["minimum_w64_retention_vs_source"]),
            "w256_absolute": w256["exact_accuracy"] >= float(gates_cfg["minimum_w256_exact"]),
            "w256_controls": all(w256["exact_accuracy"] - arms[name]["evaluation"]["256"]["rollout"]["exact_accuracy"] >= float(gates_cfg["minimum_w256_gain_vs_controls"]) for name in controls),
            "teacher_compression": teacher128 - w128["exact_accuracy"] <= float(gates_cfg["maximum_teacher_compression_loss"]),
            "svd_reconstruction": random_error - svd_error >= float(gates_cfg["minimum_random_reconstruction_advantage"]),
            "slot_causality": min(interventions["zeroed_changed"], interventions["permuted_changed"], interventions["randomized_changed"]) >= int(gates_cfg["minimum_intervention_changed_predictions"]),
            "joint_updates": any(value > 0 for key, value in summary["stages"]["training"]["full_rrt"]["phase_two"]["parameter_update_norms"].items() if key.startswith("base.")) and any(value > 0 for key, value in summary["stages"]["training"]["full_rrt"]["phase_two"]["parameter_update_norms"].items() if not key.startswith("base.")),
            "artifact": artifact < int(gates_cfg["maximum_artifact_bytes"]), "candidate_runtime": candidate <= float(gates_cfg["maximum_guarded_candidate_seconds"]),
        }
        decision = "authorize-confirmations" if all(gates.values()) else "reject-exp675-rrt"
        return self.persist_terminal(self.spec(), arms, summary["stages"], summary["resource"], decision, gates)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/exp_675_rrt_svd_teacher_uptraining.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / ".scratch/exp-675-rrt-svd-teacher-uptraining")
    parser.add_argument("--mode", choices=("calibration", "formal"), required=True)
    args = parser.parse_args(); experiment = Experiment(args.config, args.output, args.mode)
    result = experiment.calibration() if args.mode == "calibration" else experiment.formal()
    print(json.dumps({"mode": args.mode, "decision": result.get("decision"), "resource": result.get("resource", result.get("measured")), "gates": result.get("gates", result.get("checks"))}, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
