"""Freeze, calibrate, and conditionally execute EXP678 target-only continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import psutil
import torch
from safetensors.torch import load_file, save_file
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.exp_676_lla_moving_window.run import (  # noqa: E402
    Experiment as BaseExperiment,
    RootSpec,
    Timer,
    aggregate,
    git_output,
    sha256,
    write_json,
)
from experiments.exp_678_target_only_continuation.model import (  # noqa: E402
    mask_weights,
    masked_mean,
    reinjection_receipt,
    state_dict_identity,
    trace_forward,
)
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402


IDENTITY_PATHS = {
    "config": "configs/exp_678_target_only_continuation.yaml",
    "model": "experiments/exp_678_target_only_continuation/model.py",
    "runner": "experiments/exp_678_target_only_continuation/run.py",
    "test": "tests/test_exp_678_target_only_continuation.py",
    "preregistration": "knowledge/autoresearch/ablations/exp-678-target-only-continuation.md",
    "mechanism_map": "knowledge/autoresearch/research/2026-07-30-exp678-target-only-mechanism.md",
}


def state_dict_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in state.items():
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class Experiment(BaseExperiment):
    """EXP678 route using EXP676's authenticated Candidate005 utilities."""

    trained_arms = (
        "target_only",
        "mixed_trajectory",
        "endpoint_weight",
        "shuffled_mask",
        "randomized_control",
    )

    def __init__(self, config_path: Path, output: Path, mode: str, root_override: int | None = None) -> None:
        setup = Timer()
        super().__init__(config_path, output, mode, root_override)
        self.setup_resource = setup.result()

    def source_identity(self) -> dict[str, Any]:
        configured = {
            key: ROOT / value if key != "config" else self.config_path
            for key, value in IDENTITY_PATHS.items()
        }
        return {
            "repository_commit": git_output("rev-parse", "HEAD"),
            "branch": git_output("rev-parse", "--abbrev-ref", "HEAD"),
            "challenge_commit": git_output(
                "-C", str(ROOT / "sources/github/modular-arithmetic-challenge"), "rev-parse", "HEAD"
            ),
            **{f"{key}_sha256": sha256(path) for key, path in configured.items()},
            "paper_sha256": sha256(ROOT / self.config["source_contract"]["paper_path"]),
            "research_manifest_sha256": sha256(ROOT / "research/manifest.json"),
            "checkpoint_sha256": sha256(self.checkpoint),
            "base_model_sha256": sha256(ROOT / "experiments/exp_220_width_independent_cellular/model.py"),
            "exp300_sha256": sha256(ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py"),
            "exp329_sha256": sha256(ROOT / "experiments/exp_329_cellular_width64_autonomous_state_recovery/run.py"),
            "generator_sha256": sha256(
                ROOT / "sources/github/modular-arithmetic-challenge/src/modchallenge/testgen/generator.py"
            ),
        }

    def frozen_protocol(self) -> dict[str, Any]:
        cfg = self.config
        return {
            "schema_version": "exp678-frozen-protocol-v1",
            "status": "frozen-before-science",
            "source": self.source_identity(),
            "arms": cfg["arms"],
            "mask": cfg["mask"],
            "training": cfg["training"],
            "data": cfg["data"],
            "evaluation": cfg["evaluation"],
            "gates": cfg["gates"],
            "resource_budget": cfg["resource_budget"],
            "formal_inference": cfg["source_contract"]["formal_inference"],
            "analogue_exclusions": cfg["source_mapping"]["exclusions"],
            "paths": IDENTITY_PATHS,
            "environment": {
                "python": sys.version,
                "torch": torch.__version__,
                "torch_threads": torch.get_num_threads(),
                "cpu_count": psutil.cpu_count(logical=True),
            },
        }

    def freeze(self) -> dict[str, Any]:
        payload = self.frozen_protocol()
        path = ROOT / self.config["resource_budget"]["frozen_protocol_path"]
        write_json(path, payload)
        return {**payload, "path": str(path.relative_to(ROOT)), "sha256": sha256(path)}

    def assert_frozen_clean(self) -> None:
        path = ROOT / self.config["resource_budget"]["frozen_protocol_path"]
        if not path.is_file():
            raise RuntimeError("freeze mode must bind final package bytes before calibration/science")
        frozen = json.loads(path.read_text(encoding="utf-8"))
        if frozen != self.frozen_protocol():
            raise RuntimeError("current package/source bytes differ from frozen protocol")
        source = self.config["source_contract"]
        current = self.source_identity()
        for key in (
            "challenge_commit",
            "paper_sha256",
            "research_manifest_sha256",
            "checkpoint_sha256",
            "base_model_sha256",
            "exp300_sha256",
            "exp329_sha256",
            "generator_sha256",
        ):
            expected = source[key]
            if current[key] != expected:
                raise RuntimeError(f"frozen source mismatch: {key}")

    def spec(self) -> RootSpec:
        values = self.config["data"]
        if self.mode == "calibration":
            selected = values["calibration"]
        elif self.mode == "formal":
            selected = values["formal"]
        else:
            matches = [item for item in values["confirmations"] if int(item["root"]) == int(self.root_override or -1)]
            if len(matches) != 1:
                raise ValueError("confirmation requires frozen root 20261264 or 20261265")
            selected = matches[0]
        return RootSpec(**selected)

    def transition_position_losses(
        self,
        model: nn.Module,
        features: torch.Tensor,
        target: list[int],
        quotient: torch.Tensor,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, quotient_logits = model(features)
        loss, _ = self.exp300.transition_loss(
            logits,
            quotient_logits,
            target,
            quotient,
            width,
            float(self.config["training"]["worst_bit_weight"]),
            float(self.config["training"]["quotient_weight"]),
        )
        return loss, logits

    def recovery_loss(
        self,
        model: nn.Module,
        sampler: Any,
        rng: random.Random,
        width: int,
        kind: str,
        step: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        cfg = self.config["training"]
        batch, chunk = int(cfg["recovery_batch_size"]), int(cfg["recovery_chunk_steps"])
        multiplier, right, modulus = sampler.pairs(batch)
        starts = [rng.randrange(0, width - chunk + 1) for _ in range(batch)]
        teacher = self.exp329.teacher_states_before(multiplier, right, modulus, starts, width)
        state_plane = self.exp300.bits(teacher, width)
        losses: list[torch.Tensor] = []
        for offset in range(chunk):
            positions = [start + offset for start in starts]
            bit = [(left >> (width - 1 - position)) & 1 for left, position in zip(multiplier, positions)]
            raw = [2 * state + flag * value for state, flag, value in zip(teacher, bit, right)]
            target = [value % prime for value, prime in zip(raw, modulus)]
            quotient = torch.tensor([value // prime for value, prime in zip(raw, modulus)], dtype=torch.long)
            features = (
                self.exp300.features_from_soft_state(state_plane, right, modulus, bit, width)
                if hasattr(self.exp300, "features_from_soft_state")
                else self._soft_features(state_plane, right, modulus, bit, width)
            )
            position_loss, final_logits = self.transition_position_losses(model, features, target, quotient, width)
            losses.append(position_loss)
            probabilities = torch.sigmoid(final_logits)
            state_plane = final_logits.ge(0).float() + probabilities - probabilities.detach()
            teacher = target
        weights = mask_weights(kind, positions=chunk, step=step, seed=self.spec().train_seed)
        loss, receipt = masked_mean(losses, weights)
        receipt.update(
            {
                "per_position_loss": [float(value.detach()) for value in losses],
                "trajectory_mean_loss": float(torch.stack(losses[:-1]).mean().detach()),
                "final_target_loss": float(losses[-1].detach()),
                "forward_calls": chunk,
            }
        )
        return loss, receipt

    def train_arm(self, model: nn.Module, spec: RootSpec, arm_name: str, steps: int, timer: Timer) -> dict[str, Any]:
        cfg = self.config["training"]
        kind = self.config["arms"][arm_name]["kind"]
        random.seed(spec.train_seed)
        np.random.seed(spec.train_seed)
        torch.manual_seed(spec.train_seed)
        parameters = list(model.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, steps), eta_min=float(cfg["minimum_learning_rate"])
        )
        rng = random.Random(spec.train_seed + 7)
        transition = self.exp300.BigIntSampler(self.split(64, spec.root)["train"], spec.train_seed + 11)
        recovery = {
            width: self.exp300.BigIntSampler(self.split(width, spec.root)["train"], spec.train_seed + width)
            for width in (64, 128)
        }
        before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        history: list[dict[str, Any]] = []
        totals = {"updates": 0, "examples": 0, "forward_calls": 0, "ce_tensor_calls": 0, "effective_supervised_positions": 0}
        model.train()
        for step in range(1, steps + 1):
            mode = cfg["mode_schedule"][(step - 1) % len(cfg["mode_schedule"])]
            if mode == "transition64":
                state, right, modulus, bit, target, quotient = transition.transitions(int(cfg["batch_size"]))
                position_loss, _ = self.transition_position_losses(
                    model, self.exp300.features(state, right, modulus, bit, 64), target, quotient, 64
                )
                loss, receipt = masked_mean([position_loss], mask_weights(kind, positions=1, step=step, seed=spec.train_seed))
                receipt.update({"per_position_loss": [float(position_loss.detach())], "trajectory_mean_loss": None, "final_target_loss": float(position_loss.detach()), "forward_calls": 1})
                examples = int(cfg["batch_size"])
            else:
                width = int(mode.removeprefix("recovery"))
                loss, receipt = self.recovery_loss(model, recovery[width], rng, width, kind, step)
                examples = int(cfg["recovery_batch_size"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = float(nn.utils.clip_grad_norm_(parameters, float(cfg["gradient_clip"])))
            optimizer.step()
            scheduler.step()
            totals["updates"] += 1
            totals["examples"] += examples
            totals["forward_calls"] += int(receipt["forward_calls"])
            totals["ce_tensor_calls"] += int(receipt["ce_tensor_calls"])
            totals["effective_supervised_positions"] += int(receipt["effective_supervised_positions"])
            if step == 1 or step % 50 == 0 or step == steps:
                timer.sample()
                record = {"step": step, "mode": mode, "loss": float(loss.detach()), "gradient_norm": gradient, "mask": receipt}
                history.append(record)
                print(json.dumps({"root": spec.root, "arm": arm_name, **record}), flush=True)
        model.eval()
        changes = {
            name: float((model.state_dict()[name] - value).norm())
            for name, value in before.items()
            if torch.is_floating_point(value)
        }
        return {
            "steps": steps,
            "history": history,
            "work": totals,
            "optimizer": "AdamW",
            "data_order_seed": spec.train_seed,
            "parameter_update_norm_sum": sum(changes.values()),
            "parameter_update_norms": changes,
        }

    def train_and_cache(self, spec: RootSpec, steps: int, session: Timer) -> tuple[dict[str, nn.Module], dict[str, Any]]:
        arms: dict[str, nn.Module] = {}
        records: dict[str, Any] = {}
        for arm_name in self.config["arms"]:
            total = Timer()
            randomized = self.config["arms"][arm_name]["initialization"] == "randomized"
            model = self.base(randomized=randomized, seed=spec.train_seed + 900)
            initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
            training_timer = Timer()
            if arm_name in self.trained_arms:
                training = self.train_arm(model, spec, arm_name, steps, session)
            else:
                training = {"steps": 0, "reason": "frozen-source", "work": {"updates": 0, "examples": 0, "forward_calls": 0, "ce_tensor_calls": 0, "effective_supervised_positions": 0}}
            training["runtime"] = training_timer.result()
            path = self.checkpoint_path(spec, arm_name)
            persistence = Timer()
            path.parent.mkdir(parents=True, exist_ok=True)
            save_file({key: value.detach().contiguous() for key, value in model.state_dict().items()}, path)
            identity = self.inference_identity(model, path, spec.eval_seed + len(arms))
            persistence_resource = persistence.result()
            if not all(identity[key] for key in (
                "state_dict_identity", "ordinary_vs_trace_logits_equal", "ordinary_vs_trace_quotient_equal",
                "pre_save_vs_reload_logits_equal", "pre_save_vs_reload_quotient_equal"
            )):
                raise RuntimeError(f"formal inference identity failed for {arm_name}")
            initial_digest = state_dict_digest(initial)
            checkpoint = {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}
            arms[arm_name] = model.eval()
            records[arm_name] = {
                "training": training,
                "checkpoint": checkpoint,
                "inference_identity": identity,
                "initial_state_key_schema": list(initial),
                "initial_state_sha256": initial_digest,
                "initial_state_matches_exp609": initial_digest == state_dict_digest(self.base().state_dict()),
                "persistence_runtime": persistence_resource,
                "total_runtime": total.result(),
                "parameters": identity["parameter_count"],
            }
            write_json(self.output / self.mode / f"root-{spec.root}-progress.json", {
                "schema_version": "exp678-root-progress-v1", "root": spec.root,
                "completed_arms": records, "completion_marker": "training-arms-in-progress"
            })
        return arms, records

    @torch.no_grad()
    def trajectory_diagnostic(self, model: nn.Module, spec: RootSpec, width: int, cases: int) -> dict[str, Any]:
        sampler = self.exp300.BigIntSampler(self.split(width, spec.root)["test"], spec.eval_seed + 4000 + width)
        rng = random.Random(spec.eval_seed + 5000 + width)
        batch = int(self.config["training"]["recovery_batch_size"])
        receipts = []
        for index in range((cases + batch - 1) // batch):
            _, receipt = self.recovery_loss(model, sampler, rng, width, "mixed", index + 1)
            receipts.append(receipt)
        per_position = [
            float(np.mean([receipt["per_position_loss"][position] for receipt in receipts]))
            for position in range(int(self.config["training"]["recovery_chunk_steps"]))
        ]
        return {
            "cases": cases,
            "batches": len(receipts),
            "per_position_loss": per_position,
            "trajectory_mean_loss": float(np.mean(per_position[:-1])),
            "final_target_loss": per_position[-1],
            "ce_tensor_calls": len(per_position) * len(receipts),
            "forward_calls": len(per_position) * len(receipts),
        }

    def evaluate_arm(
        self,
        model: nn.Module,
        spec: RootSpec,
        widths: list[int],
        transitions: dict[int, int],
        rollouts: dict[int, int],
        trajectory_cases: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        evaluation: dict[str, Any] = {}
        timers: list[dict[str, Any]] = []
        for width in widths:
            timer = Timer()
            split = self.split(width, spec.root)
            transition = self.exp300.evaluate_transitions(
                model,
                self.exp300.BigIntSampler(split["test"], spec.eval_seed + 1000 + width),
                transitions[width],
            )
            rollout = self.evaluate_rollouts(
                model,
                self.exp300.BigIntSampler(split["test"], spec.eval_seed + 2000 + width),
                rollouts[width],
                width,
            )
            trajectory = self.trajectory_diagnostic(model, spec, width, trajectory_cases)
            runtime = timer.result()
            timers.append(runtime)
            evaluation[str(width)] = {"transition": transition, "rollout": rollout, "trajectory_ce": trajectory, "runtime": runtime}
        return evaluation, aggregate(timers)

    def root_denominators(self) -> tuple[dict[int, int], dict[int, int], int, int]:
        calibration = self.mode == "calibration"
        transitions = {int(k): int(v) for k, v in self.config["evaluation"]["calibration_transition_cases" if calibration else "transition_cases"].items()}
        rollouts = {int(k): int(v) for k, v in self.config["evaluation"]["calibration_rollout_cases" if calibration else "rollout_cases"].items()}
        trajectory = int(self.config["evaluation"]["calibration_trajectory_loss_cases" if calibration else "trajectory_loss_cases"])
        steps = int(self.config["training"]["calibration_steps" if calibration else "steps"])
        return transitions, rollouts, trajectory, steps

    def execute_root(self) -> dict[str, Any]:
        spec = self.spec()
        transitions, rollouts, trajectory_cases, steps = self.root_denominators()
        session = Timer()
        arms, records = self.train_and_cache(spec, steps, session)
        reinjection = {str(width): reinjection_receipt(arms["target_only"], int(width)) for width in self.config["evaluation"]["widths"]}
        if not all(item["gate_reinjects_every_visit"] and item["candidate_reinjects_every_visit"] for item in reinjection.values()):
            raise RuntimeError("Candidate005 input reinjection identity failed")

        def score(name: str, widths: list[int]) -> None:
            evaluation, runtime = self.evaluate_arm(arms[name], spec, widths, transitions, rollouts, trajectory_cases)
            records[name].setdefault("evaluation", {}).update(evaluation)
            previous = records[name].get("evaluation_runtime")
            records[name]["evaluation_runtime"] = runtime if previous is None else aggregate([previous, runtime])

        early_decision: str | None = None
        if self.mode == "calibration":
            for arm_name in arms:
                score(arm_name, [64, 128, 256])
        else:
            score("target_only", [128])
            primary = records["target_only"]["evaluation"]["128"]
            if primary["rollout"]["exact_accuracy"] < float(self.config["gates"]["minimum_w128_exact"]):
                early_decision = "reject-W128-primary-exact"
            elif primary["transition"]["exact_accuracy"] < float(self.config["gates"]["minimum_w128_transition"]):
                early_decision = "reject-W128-primary-transition"
            elif primary["rollout"]["invalids"] > int(self.config["gates"]["maximum_invalids"]):
                early_decision = "reject-W128-primary-validity"
            if early_decision is None:
                score("frozen_source", [128])
                score("mixed_trajectory", [128])
                target = primary["rollout"]["exact_accuracy"]
                if target < records["frozen_source"]["evaluation"]["128"]["rollout"]["exact_accuracy"] - float(self.config["gates"]["maximum_w128_loss_vs_source"]):
                    early_decision = "reject-W128-source-retention"
                elif target < records["mixed_trajectory"]["evaluation"]["128"]["rollout"]["exact_accuracy"] - float(self.config["gates"]["maximum_w128_loss_vs_mixed"]):
                    early_decision = "reject-W128-mixed-retention"
            if early_decision is None:
                for arm_name in ("endpoint_weight", "shuffled_mask", "randomized_control"):
                    score(arm_name, [128])
                target = primary["rollout"]["exact_accuracy"]
                if target - records["randomized_control"]["evaluation"]["128"]["rollout"]["exact_accuracy"] < float(self.config["gates"]["minimum_trained_minus_randomized_w128"]):
                    early_decision = "reject-randomized-weight-dependence"
                elif target - records["shuffled_mask"]["evaluation"]["128"]["rollout"]["exact_accuracy"] < float(self.config["gates"]["minimum_target_minus_shuffled_w128"]):
                    early_decision = "reject-target-mask-dependence"
            if early_decision is None:
                for arm_name in arms:
                    score(arm_name, [64, 256])

        timing_timer = Timer()
        timing_width = int(self.config["evaluation"]["candidate_timing_width"])
        timing_cases = int(self.config["evaluation"]["candidate_timing_cases"])
        timing = self.evaluate_rollouts(
            arms["target_only"],
            self.exp300.BigIntSampler(self.split(timing_width, spec.root)["test"], spec.eval_seed + 9000),
            timing_cases,
            timing_width,
        )
        result = {
            "schema_version": "exp678-root-v1",
            "mode": self.mode,
            "root": spec.root,
            "source": self.source_identity(),
            "setup_resource": self.setup_resource,
            "reinjection": reinjection,
            "arms": records,
            "candidate_timing": {"width": timing_width, "cases": timing_cases, "rollout": timing, "resource": timing_timer.result()},
            "resource": session.result(),
            "early_decision": early_decision,
            "completion_marker": "root-complete",
        }
        write_json(self.output / self.mode / f"root-{spec.root}-summary.json", result)
        return result

    def calibration(self) -> dict[str, Any]:
        self.assert_frozen_clean()
        root = self.execute_root()
        cfg = self.config
        training = aggregate([root["arms"][name]["training"]["runtime"] for name in self.trained_arms])
        persistence = aggregate([root["arms"][name]["persistence_runtime"] for name in cfg["arms"]])
        evaluation = aggregate([root["arms"][name]["evaluation_runtime"] for name in cfg["arms"]])
        training_scale = int(cfg["training"]["steps"]) / int(cfg["training"]["calibration_steps"])
        evaluation_scale = max(
            sum(int(value) for value in cfg["evaluation"]["transition_cases"].values()) / sum(int(value) for value in cfg["evaluation"]["calibration_transition_cases"].values()),
            sum(int(value) for value in cfg["evaluation"]["rollout_cases"].values()) / sum(int(value) for value in cfg["evaluation"]["calibration_rollout_cases"].values()),
        )
        headroom = float(cfg["resource_budget"]["formal_headroom"])
        expected: dict[str, Any] = {}
        for key in ("wall_seconds", "process_tree_cpu_seconds", "cgroup_cpu_seconds"):
            expected[key] = headroom * (
                float(root["setup_resource"][key])
                + float(training[key]) * training_scale
                + float(evaluation[key]) * evaluation_scale
                + float(persistence[key])
            )
        expected["peak_process_tree_rss_bytes"] = int(root["resource"]["peak_process_tree_rss_bytes"] * float(cfg["resource_budget"]["rss_headroom"]))
        soft = {key: value * float(cfg["resource_budget"]["soft_multiplier"]) for key, value in expected.items()}
        hard = {key: value * float(cfg["resource_budget"]["hard_multiplier"]) for key, value in expected.items()}
        timing = root["candidate_timing"]
        nominal = float(timing["resource"]["wall_seconds"]) / int(timing["cases"]) * int(cfg["evaluation"]["candidate_projection_cases"])
        arm_walls = [float(root["arms"][name]["evaluation_runtime"]["wall_seconds"]) for name in cfg["arms"]]
        variability = max(0.10, float(np.std(arm_walls)) / max(float(np.mean(arm_walls)), 1e-12))
        guarded = nominal * max(headroom, 1.0 + variability)
        artifact_bytes = sum(int(value["checkpoint"]["bytes"]) for value in root["arms"].values())
        identities = all(all(value["inference_identity"][key] for key in (
            "state_dict_identity", "ordinary_vs_trace_logits_equal", "ordinary_vs_trace_quotient_equal",
            "pre_save_vs_reload_logits_equal", "pre_save_vs_reload_quotient_equal"
        )) for value in root["arms"].values())
        memory_available = psutil.virtual_memory().available
        checks = {
            "complete_route": set(root["arms"]) == set(cfg["arms"]),
            "all_schedule_modes_calibrated": int(cfg["training"]["calibration_steps"]) >= len(cfg["training"]["mode_schedule"]),
            "formal_inference_identity": identities,
            "candidate_runtime": guarded <= float(cfg["gates"]["maximum_guarded_candidate_seconds"]),
            "artifact": artifact_bytes < int(cfg["gates"]["maximum_artifact_bytes"]),
            "memory": int(hard["peak_process_tree_rss_bytes"]) < int(memory_available * 0.80),
            "expected_value": True,
            "no_fidelity_reduction": cfg["resource_budget"]["fidelity_reduced"] is False,
        }
        decision = "admit-root-20261263" if all(checks.values()) else "park-calibration-resource-or-compliance"
        summary_path = self.output / "calibration" / f"root-{self.spec().root}-summary.json"
        admission = {
            "schema_version": "exp678-dynamic-admission-v1",
            "root": self.spec().root,
            "decision": decision,
            "source": root["source"],
            "calibration_summary": str(summary_path.relative_to(ROOT)),
            "calibration_sha256": sha256(summary_path),
            "measured": root["resource"],
            "setup_component": root["setup_resource"],
            "training_component": training,
            "evaluation_component": evaluation,
            "persistence_component": persistence,
            "training_scale": training_scale,
            "evaluation_scale": evaluation_scale,
            "measured_throughput": {
                "trained_updates_per_wall_second": int(cfg["training"]["calibration_steps"]) * len(self.trained_arms) / max(float(training["wall_seconds"]), 1e-12),
                "evaluated_cases_per_wall_second": (
                    sum(int(v) for v in cfg["evaluation"]["calibration_transition_cases"].values())
                    + sum(int(v) for v in cfg["evaluation"]["calibration_rollout_cases"].values())
                ) * len(cfg["arms"]) / max(float(evaluation["wall_seconds"]), 1e-12),
            },
            "variability": variability,
            "expected": expected,
            "eta_lower_wall_seconds": expected["wall_seconds"] / headroom,
            "eta_upper_wall_seconds": soft["wall_seconds"],
            "soft": soft,
            "hard": hard,
            "candidate_nominal_seconds": nominal,
            "candidate_guarded_seconds": guarded,
            "candidate_ceiling_seconds": cfg["gates"]["maximum_guarded_candidate_seconds"],
            "artifact_bytes": artifact_bytes,
            "available_memory_bytes": memory_available,
            "checks": checks,
            "expected_value_rationale": "M044.a is the frozen next-ranked training-only intervention; the complete calibrated route is finite and tests a new causal mask surface.",
            "fidelity_reduced": False,
        }
        write_json(ROOT / cfg["resource_budget"]["admission_path"], admission)
        write_json(self.output / "calibration.json", admission)
        return admission

    def load_admission(self) -> dict[str, Any]:
        path = ROOT / self.config["resource_budget"]["admission_path"]
        admission = json.loads(path.read_text(encoding="utf-8"))
        calibration = ROOT / admission["calibration_summary"]
        if admission["decision"] != "admit-root-20261263" or admission["source"] != self.source_identity() or sha256(calibration) != admission["calibration_sha256"] or admission["fidelity_reduced"]:
            raise RuntimeError("formal requires exact admitted root-20261262 evidence")
        return admission

    def formal_gates(self, root: dict[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
        arms, gates = root["arms"], self.config["gates"]
        target = arms["target_only"]["evaluation"]["128"]
        checks: dict[str, bool] = {
            "w128_exact": target["rollout"]["exact_accuracy"] >= float(gates["minimum_w128_exact"]),
            "w128_transition": target["transition"]["exact_accuracy"] >= float(gates["minimum_w128_transition"]),
            "w128_validity": target["rollout"]["invalids"] <= int(gates["maximum_invalids"]),
        }
        if all(name in arms and "128" in arms[name].get("evaluation", {}) for name in ("mixed_trajectory", "frozen_source")):
            checks["source_retention"] = target["rollout"]["exact_accuracy"] >= arms["frozen_source"]["evaluation"]["128"]["rollout"]["exact_accuracy"] - float(gates["maximum_w128_loss_vs_source"])
            checks["mixed_retention"] = target["rollout"]["exact_accuracy"] >= arms["mixed_trajectory"]["evaluation"]["128"]["rollout"]["exact_accuracy"] - float(gates["maximum_w128_loss_vs_mixed"])
        if all("128" in arms[name].get("evaluation", {}) for name in ("shuffled_mask", "randomized_control")):
            checks["trained_weight_dependence"] = target["rollout"]["exact_accuracy"] - arms["randomized_control"]["evaluation"]["128"]["rollout"]["exact_accuracy"] >= float(gates["minimum_trained_minus_randomized_w128"])
            checks["target_mask_dependence"] = target["rollout"]["exact_accuracy"] - arms["shuffled_mask"]["evaluation"]["128"]["rollout"]["exact_accuracy"] >= float(gates["minimum_target_minus_shuffled_w128"])
        if all("256" in arms[name].get("evaluation", {}) for name in ("target_only", "mixed_trajectory", "endpoint_weight")):
            target256 = arms["target_only"]["evaluation"]["256"]["rollout"]["exact_accuracy"]
            checks["w256_absolute"] = target256 >= float(gates["minimum_w256_exact"])
            checks["w256_controls"] = all(
                target256 - arms[name]["evaluation"]["256"]["rollout"]["exact_accuracy"] >= float(gates["minimum_w256_gain_vs_controls"])
                for name in ("mixed_trajectory", "endpoint_weight")
            )
        comparisons = {
            "w128_exact": {name: value.get("evaluation", {}).get("128", {}).get("rollout", {}).get("exact_accuracy") for name, value in arms.items()},
            "w256_exact": {name: value.get("evaluation", {}).get("256", {}).get("rollout", {}).get("exact_accuracy") for name, value in arms.items()},
            "final_vs_trajectory_ce_w128": {name: value.get("evaluation", {}).get("128", {}).get("trajectory_ce") for name, value in arms.items()},
            "checkpoint_sha256": {name: value["checkpoint"]["sha256"] for name, value in arms.items()},
        }
        return checks, comparisons

    def persist_terminal(self, root: dict[str, Any], admission: dict[str, Any]) -> dict[str, Any]:
        checks, comparisons = self.formal_gates(root)
        if root.get("early_decision"):
            decision = f"reject-exp678-{root['early_decision']}"
        elif all(checks.values()):
            decision = "authorize-confirmations" if self.mode == "formal" else "confirmation-pass"
        else:
            decision = "reject-exp678-scientific-gate"
        summary = {
            "schema_version": "exp678-terminal-v1",
            "run_id": self.config["run_id"],
            "status": "completed",
            "decision": decision,
            "root": root["root"],
            "source": root["source"],
            "arms": root["arms"],
            "candidate_timing": root["candidate_timing"],
            "resource": root["resource"],
            "gates": checks,
            "comparisons": comparisons,
            "dynamic_admission": admission,
            "roots": {"20261262": "consumed-calibration", "20261263": "consumed-formal" if self.mode != "calibration" else "unopened", "20261264": "current" if root["root"] == 20261264 else "unopened", "20261265": "current" if root["root"] == 20261265 else "unopened"},
            "limitations": [
                "Fresh generated experiment identities only; no checked, independent, official/private, candidate, submission, promotion or completion evaluation.",
                "The paper-scale batch, context, token-budget and multi-epoch claims are not reproduced; EXP678 is an analogue of target-position masking only.",
                "Masks, labels and diagnostics are training/evaluator-only; formal inference remains the original learned SharedDilatedConvGRU graph.",
            ],
            "completion_marker": "terminal-persisted",
        }
        write_json(self.output / "terminal-summary.json", summary)
        write_json(ROOT / "experiments/runs/exp-678-target-only-continuation/run.json", summary)
        write_json(ROOT / "evals/results/exp-678-target-only-continuation/metrics.json", summary)
        return summary

    def formal(self) -> dict[str, Any]:
        self.assert_frozen_clean()
        admission = self.load_admission()
        return self.persist_terminal(self.execute_root(), admission)

    def confirmation(self) -> dict[str, Any]:
        self.assert_frozen_clean()
        admission = self.load_admission()
        return self.persist_terminal(self.execute_root(), admission)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/exp_678_target_only_continuation.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / ".scratch/exp-678-target-only-continuation")
    parser.add_argument("--mode", choices=("freeze", "calibration", "formal", "confirmation"), required=True)
    parser.add_argument("--root", type=int)
    args = parser.parse_args()
    experiment = Experiment(args.config, args.output, args.mode, args.root)
    if args.mode == "freeze":
        result = experiment.freeze()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    result = experiment.calibration() if args.mode == "calibration" else experiment.formal() if args.mode == "formal" else experiment.confirmation()
    experiment_id = configure_mlflow(experiment.config["tracking"]["experiment_name"])
    with mlflow.start_run(run_name=f"{experiment.config['run_id']}-{args.mode}-{experiment.spec().root}") as run:
        mlflow.set_tags({
            "canonical_run_id": experiment.config["run_id"], "mode": args.mode,
            "root": str(experiment.spec().root), "decision": str(result.get("decision")),
            "repository_commit": git_output("rev-parse", "HEAD"),
        })
        mlflow.log_dict(result, f"{args.mode}-result.json")
        mlflow.log_metrics({
            "wall_seconds": float(result.get("resource", result.get("measured", {})).get("wall_seconds", 0.0)),
            "candidate_guarded_seconds": float(result.get("candidate_guarded_seconds", 0.0)),
        })
        mlflow_run_id = run.info.run_id
    receipt = {"experiment_id": experiment_id, "experiment_name": experiment.config["tracking"]["experiment_name"], "run_id": mlflow_run_id, "tracking_uri": tracking_uri()}
    write_json(experiment.output / f"{args.mode}-mlflow.json", receipt)
    print(json.dumps({"mode": args.mode, "root": experiment.spec().root, "decision": result.get("decision"), "resource": result.get("resource", result.get("measured")), "gates": result.get("gates", result.get("checks")), "mlflow": receipt}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
