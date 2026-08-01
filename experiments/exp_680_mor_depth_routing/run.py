"""Freeze, preflight, calibrate, and conditionally execute EXP680."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import psutil
import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.exp_676_lla_moving_window.run import RootSpec, Timer, aggregate, git_output, sha256, write_json  # noqa: E402
from experiments.exp_678_target_only_continuation.run import Experiment as ContinuationExperiment, state_dict_digest  # noqa: E402
from experiments.exp_680_mor_depth_routing.model import RoutedDepthConvGRU, capacity_counts, fixed_full_identity  # noqa: E402
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402


IDENTITY_PATHS = {
    "config": "configs/exp_680_mor_depth_routing.yaml",
    "model": "experiments/exp_680_mor_depth_routing/model.py",
    "runner": "experiments/exp_680_mor_depth_routing/run.py",
    "test": "tests/test_exp_680_mor_depth_routing.py",
    "preregistration": "knowledge/autoresearch/ablations/exp-680-mor-depth-routing.md",
    "mechanism_map": "knowledge/autoresearch/research/2026-07-30-exp680-mor-depth-routing-mechanism.md",
}


def tensor_delta(left: tuple[torch.Tensor, torch.Tensor], right: tuple[torch.Tensor, torch.Tensor]) -> float:
    return max(float((left[0] - right[0]).abs().max()), float((left[1] - right[1]).abs().max()))


class Experiment(ContinuationExperiment):
    trained_arms = ("expert_choice", "token_choice", "fixed_full", "equal_average_fixed", "equal_average_two_point")

    def source_identity(self) -> dict[str, Any]:
        paths = {key: self.config_path if key == "config" else ROOT / value for key, value in IDENTITY_PATHS.items()}
        return {
            "repository_commit": git_output("rev-parse", "HEAD"),
            "branch": git_output("branch", "--show-current"),
            "challenge_commit": git_output("-C", str(ROOT / "sources/github/modular-arithmetic-challenge"), "rev-parse", "HEAD"),
            **{f"{key}_sha256": sha256(path) for key, path in paths.items()},
            "paper_sha256": sha256(ROOT / self.config["source_contract"]["paper_path"]),
            "research_manifest_sha256": sha256(ROOT / "research/manifest.json"),
            "checkpoint_sha256": sha256(self.checkpoint),
            "base_model_sha256": sha256(ROOT / "experiments/exp_220_width_independent_cellular/model.py"),
            "exp300_sha256": sha256(ROOT / "experiments/exp_300_width64_bit_tensor_transfer/run.py"),
            "exp329_sha256": sha256(ROOT / "experiments/exp_329_cellular_width64_autonomous_state_recovery/run.py"),
            "generator_sha256": sha256(ROOT / "sources/github/modular-arithmetic-challenge/src/modchallenge/testgen/generator.py"),
        }

    def frozen_protocol(self) -> dict[str, Any]:
        cfg = self.config
        return {
            "schema_version": "exp680-frozen-protocol-v1",
            "status": "frozen-before-package-oracle",
            "source": self.source_identity(),
            "source_mapping": cfg["source_mapping"],
            "library_decision": cfg["library_decision"],
            "routing": cfg["routing"],
            "arms": cfg["arms"],
            "interventions": cfg["interventions"],
            "training": cfg["training"],
            "data": cfg["data"],
            "evaluation": cfg["evaluation"],
            "gates": cfg["gates"],
            "resource_budget": cfg["resource_budget"],
            "paths": IDENTITY_PATHS,
            "environment": {"python": sys.version, "torch": torch.__version__, "torch_threads": torch.get_num_threads(), "logical_cpu_count": psutil.cpu_count(logical=True)},
        }

    def freeze(self) -> dict[str, Any]:
        payload = self.frozen_protocol()
        path = ROOT / self.config["resource_budget"]["frozen_protocol_path"]
        write_json(path, payload)
        return {**payload, "path": str(path.relative_to(ROOT)), "sha256": sha256(path)}

    def assert_frozen_clean(self) -> None:
        path = ROOT / self.config["resource_budget"]["frozen_protocol_path"]
        current_protocol = json.loads(json.dumps(self.frozen_protocol(), sort_keys=True))
        if not path.is_file() or json.loads(path.read_text(encoding="utf-8")) != current_protocol:
            raise RuntimeError("current EXP680 package/source bytes differ from frozen protocol")
        current, expected = self.source_identity(), self.config["source_contract"]
        for key in ("challenge_commit", "paper_sha256", "research_manifest_sha256", "checkpoint_sha256", "base_model_sha256", "exp300_sha256", "exp329_sha256", "generator_sha256"):
            if current[key] != expected[key]:
                raise RuntimeError(f"authenticated source mismatch: {key}")

    def depth_sets(self) -> dict[int, tuple[int, int, int]]:
        return {int(width): tuple(int(value) for value in values) for width, values in self.config["routing"]["depth_sets"].items()}

    def build(self, route_kind: str, *, seed: int) -> RoutedDepthConvGRU:
        torch.manual_seed(seed)
        return RoutedDepthConvGRU(self.base(), route_kind=route_kind, depth_sets=self.depth_sets())

    def spec(self) -> RootSpec:
        values = self.config["data"]
        if self.mode == "calibration":
            selected = values["calibration"]
        elif self.mode == "formal":
            selected = values["formal"]
        else:
            matches = [item for item in values["confirmations"] if int(item["root"]) == int(self.root_override or -1)]
            if len(matches) != 1:
                raise ValueError("confirmation requires root 20261272 or 20261273")
            selected = matches[0]
        return RootSpec(**selected)

    def transition_position_losses(self, model, features, target, quotient, width):
        logits, quotient_logits, trace = model(features, return_trace=True)
        task_loss, _ = self.exp300.transition_loss(logits, quotient_logits, target, quotient, width, float(self.config["training"]["worst_bit_weight"]), float(self.config["training"]["quotient_weight"]))
        return task_loss + float(self.config["routing"]["auxiliary_loss_weight"]) * trace["load_loss"], logits

    def root_denominators(self) -> tuple[dict[int, int], dict[int, int], int, int, int]:
        calibration = self.mode == "calibration"
        evaluation = self.config["evaluation"]
        transitions = {int(k): int(v) for k, v in evaluation["calibration_transition_cases" if calibration else "transition_cases"].items()}
        rollouts = {int(k): int(v) for k, v in evaluation["calibration_rollout_cases" if calibration else "rollout_cases"].items()}
        trajectory = int(evaluation["calibration_trajectory_loss_cases" if calibration else "trajectory_loss_cases"])
        route_cases = int(evaluation["calibration_route_diagnostic_cases" if calibration else "route_diagnostic_cases"])
        steps = int(self.config["training"]["calibration_steps" if calibration else "steps"])
        return transitions, rollouts, trajectory, route_cases, steps

    def inference_identity(self, model: RoutedDepthConvGRU, checkpoint: Path, seed: int, route_kind: str) -> dict[str, Any]:
        reloaded = self.build(route_kind, seed=seed)
        reloaded.load_state_dict(load_file(checkpoint)); reloaded.eval(); model.eval()
        probe = torch.randn(int(self.config["evaluation"]["deterministic_repeat_cases"]), 6, 128, generator=torch.Generator().manual_seed(seed))
        with torch.no_grad():
            current = model(probe); restored = reloaded(probe); repeated = reloaded(probe)
        return {
            "state_dict_identity": state_dict_digest(model.state_dict()) == state_dict_digest(reloaded.state_dict()),
            "pre_save_vs_reload_logits_equal": torch.equal(current[0], restored[0]),
            "pre_save_vs_reload_quotient_equal": torch.equal(current[1], restored[1]),
            "deterministic_repeat_logits_equal": torch.equal(restored[0], repeated[0]),
            "deterministic_repeat_quotient_equal": torch.equal(restored[1], repeated[1]),
            "fixed_full_initialization": fixed_full_identity(model.source, model, probe),
            "accounting": model.accounting(128),
        }

    def train_and_cache(self, spec: RootSpec, steps: int, session: Timer) -> tuple[dict[str, RoutedDepthConvGRU], dict[str, Any]]:
        arms: dict[str, RoutedDepthConvGRU] = {}; records: dict[str, Any] = {}
        for offset, (name, arm) in enumerate(self.config["arms"].items()):
            model = self.build(str(arm["route_kind"]), seed=spec.train_seed + offset)
            initial = state_dict_digest(model.state_dict()); training_timer = Timer()
            training = self.train_arm(model, spec, name, steps, session) if bool(arm["train"]) else {"steps": 0, "reason": "frozen-source", "work": {"updates": 0}}
            training["runtime"] = training_timer.result()
            updates = training.get("parameter_update_norms", {})
            training["router_update_norm_sum"] = sum(float(value) for key, value in updates.items() if "router" in key)
            training["core_update_norm_sum"] = sum(float(value) for key, value in updates.items() if key.startswith("source."))
            path = self.checkpoint_path(spec, name); persistence = Timer(); path.parent.mkdir(parents=True, exist_ok=True)
            save_file({key: value.detach().contiguous() for key, value in model.state_dict().items()}, path)
            identity = self.inference_identity(model, path, spec.eval_seed + offset, str(arm["route_kind"]))
            if not all(bool(identity[key]) for key in ("state_dict_identity", "pre_save_vs_reload_logits_equal", "pre_save_vs_reload_quotient_equal", "deterministic_repeat_logits_equal", "deterministic_repeat_quotient_equal")):
                raise RuntimeError(f"reload/determinism failed: {name}")
            checkpoint = {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}
            arms[name] = model.eval(); records[name] = {"route_kind": arm["route_kind"], "training": training, "checkpoint": checkpoint, "initial_state_sha256": initial, "inference_identity": identity, "persistence_runtime": persistence.result(), "parameters": identity["accounting"]["stored_parameters"]}
            write_json(self.output / self.mode / f"root-{spec.root}-progress.json", {"schema_version": "exp680-root-progress-v1", "root": spec.root, "completed_arms": list(records), "completion_marker": "training-arms-in-progress"})
        return arms, records

    @torch.no_grad()
    def route_diagnostic(self, model: RoutedDepthConvGRU, spec: RootSpec, width: int, cases: int) -> dict[str, Any]:
        sampler = self.exp300.BigIntSampler(self.split(width, spec.root)["test"], spec.eval_seed + 6000 + width)
        state, right, modulus, bit, _, _ = sampler.transitions(cases)
        features = self.exp300.features(state, right, modulus, bit, width)
        timer = Timer(); ordinary = model(features, return_trace=True); resource = timer.result(); depths = ordinary[2]["depths"]
        shuffled = depths.roll(1)
        generator = torch.Generator().manual_seed(spec.eval_seed + width)
        random_depths = depths.index_select(0, torch.randperm(depths.numel(), generator=generator))
        shuffled_output = model(features, force_depths=shuffled)
        random_output = model(features, force_depths=random_depths)
        zeroed = copy.deepcopy(model).eval()
        for parameter in zeroed.router_parameters(): parameter.zero_()
        zeroed.expert_thresholds.zero_()
        zero_output = zeroed(features)
        randomized = copy.deepcopy(model).eval()
        torch.manual_seed(spec.eval_seed + 7000 + width)
        for parameter in randomized.router_parameters(): torch.nn.init.normal_(parameter, mean=0.0, std=0.05)
        randomized_output = randomized(features)
        core_zeroed = copy.deepcopy(model).eval()
        for parameter in core_zeroed.core_parameters(): parameter.zero_()
        core_output = core_zeroed(features)
        histogram = ordinary[2]["histogram"]
        return {
            "cases": cases, "histogram": histogram, "all_depths_used": all(int(value) > 0 for value in histogram.values()),
            "average_calls": ordinary[2]["average_calls"], "maximum_calls": ordinary[2]["maximum_calls"], "saved_calls": ordinary[2]["saved_calls"],
            "call_reduction": 1.0 - ordinary[2]["average_calls"] / ordinary[2]["depth_set"][-1],
            "thresholds": ordinary[2]["thresholds"].tolist(), "resource": resource,
            "interventions": {"route_shuffled_logit_delta": tensor_delta(ordinary[:2], shuffled_output), "matched_random_logit_delta": tensor_delta(ordinary[:2], random_output), "router_zeroed_logit_delta": tensor_delta(ordinary[:2], zero_output), "randomized_router_logit_delta": tensor_delta(ordinary[:2], randomized_output), "core_zeroed_logit_delta": tensor_delta(ordinary[:2], core_output)},
            "labels_or_answers_used_for_routing": False,
        }

    def evaluate_arm(self, model, spec, widths, transitions, rollouts, trajectory_cases, route_cases=None):
        evaluation, runtime = super().evaluate_arm(model, spec, widths, transitions, rollouts, trajectory_cases)
        for width in widths:
            evaluation[str(width)]["route"] = self.route_diagnostic(model, spec, width, int(route_cases or self.config["evaluation"]["route_diagnostic_cases"]))
        return evaluation, runtime

    def execute_root(self) -> dict[str, Any]:
        spec = self.spec(); transitions, rollouts, trajectory_cases, route_cases, steps = self.root_denominators(); session = Timer()
        arms, records = self.train_and_cache(spec, steps, session)
        def score(name: str, widths: list[int]) -> None:
            evaluation, runtime = self.evaluate_arm(arms[name], spec, widths, transitions, rollouts, trajectory_cases, route_cases)
            records[name].setdefault("evaluation", {}).update(evaluation); records[name]["evaluation_runtime"] = runtime
        early = None
        if self.mode == "calibration":
            for name in arms: score(name, [64, 128, 256])
        else:
            score("expert_choice", [128]); primary = records["expert_choice"]["evaluation"]["128"]
            if primary["rollout"]["exact_accuracy"] < float(self.config["gates"]["minimum_w128_exact"]): early = "reject-W128-primary-exact"
            elif primary["transition"]["exact_accuracy"] < float(self.config["gates"]["minimum_w128_transition"]): early = "reject-W128-primary-transition"
            elif primary["rollout"]["invalids"] > int(self.config["gates"]["maximum_invalids"]): early = "reject-W128-primary-validity"
            if early is None:
                for name in ("frozen_source", "fixed_full", "token_choice", "equal_average_fixed", "equal_average_two_point"): score(name, [128])
                checks, _ = self.formal_gates({"arms": records})
                if not all(value for key, value in checks.items() if not key.startswith("w256")): early = "reject-W128-route-or-control-gate"
            if early is None:
                for name in arms: score(name, [64, 256])
        timing_timer = Timer(); timing_cases = int(self.config["evaluation"]["candidate_timing_cases"]); timing_width = int(self.config["evaluation"]["candidate_timing_width"])
        timing = self.evaluate_rollouts(arms["expert_choice"], self.exp300.BigIntSampler(self.split(timing_width, spec.root)["test"], spec.eval_seed + 9000), timing_cases, timing_width)
        result = {"schema_version": "exp680-root-v1", "mode": self.mode, "root": spec.root, "source": self.source_identity(), "arms": records, "candidate_timing": {"width": timing_width, "cases": timing_cases, "rollout": timing, "resource": timing_timer.result()}, "resource": session.result(), "early_decision": early, "completion_marker": "root-complete"}
        write_json(self.output / self.mode / f"root-{spec.root}-summary.json", result); return result

    def preflight(self) -> dict[str, Any]:
        self.assert_frozen_clean(); original_mode, original_root = self.mode, self.root_override
        constructed = []
        for mode, root in (("calibration", None), ("formal", None), ("confirmation", 20261272), ("confirmation", 20261273)):
            self.mode, self.root_override = mode, root; spec = self.spec(); denominators = self.root_denominators()
            constructed.append({"mode": mode, "root": spec.root, "train_seed": spec.train_seed, "eval_seed": spec.eval_seed, "transition_cases": denominators[0], "rollout_cases": denominators[1], "trajectory_cases": denominators[2], "route_cases": denominators[3], "steps": denominators[4]})
        self.mode, self.root_override = original_mode, original_root
        dry_runs = {}; identities = {}
        for offset, (name, arm) in enumerate(self.config["arms"].items()):
            model = self.build(str(arm["route_kind"]), seed=680000 + offset).eval()
            dry_runs[name] = {}
            for width in self.config["evaluation"]["widths"]:
                features = torch.randn(6, 6, int(width), generator=torch.Generator().manual_seed(680000 + int(width)))
                with torch.no_grad(): output = model(features, return_trace=True)
                dry_runs[name][str(width)] = {"logits_shape": list(output[0].shape), "histogram": output[2]["histogram"], "average_calls": output[2]["average_calls"]}
            probe = torch.randn(6, 6, 128, generator=torch.Generator().manual_seed(680123))
            identities[name] = fixed_full_identity(model.source, model, probe)
        receipt = {"schema_version": "exp680-preflight-v1", "status": "passed", "source": self.source_identity(), "constructed_modes": constructed, "dry_runs": dry_runs, "fixed_full_identities": identities, "native_topk_capacity_B6": list(capacity_counts(6)), "all_paths_exist": all((ROOT / path).exists() for path in IDENTITY_PATHS.values()), "prohibited_routing_inputs": ["labels", "answers", "correctness", "validity"], "completion_marker": "preflight-complete"}
        write_json(ROOT / self.config["resource_budget"]["preflight_path"], receipt); return receipt

    def calibration(self) -> dict[str, Any]:
        self.assert_frozen_clean()
        preflight_path = ROOT / self.config["resource_budget"]["preflight_path"]
        if not preflight_path.is_file() or json.loads(preflight_path.read_text())["source"] != self.source_identity(): raise RuntimeError("calibration requires exact completed preflight")
        root = self.execute_root(); cfg = self.config
        training = aggregate([root["arms"][name]["training"]["runtime"] for name in self.trained_arms]); persistence = aggregate([root["arms"][name]["persistence_runtime"] for name in cfg["arms"]]); evaluation = aggregate([root["arms"][name]["evaluation_runtime"] for name in cfg["arms"]])
        training_scale = int(cfg["training"]["steps"]) / int(cfg["training"]["calibration_steps"])
        evaluation_scale = max(sum(cfg["evaluation"]["transition_cases"].values()) / sum(cfg["evaluation"]["calibration_transition_cases"].values()), sum(cfg["evaluation"]["rollout_cases"].values()) / sum(cfg["evaluation"]["calibration_rollout_cases"].values()))
        headroom = float(cfg["resource_budget"]["formal_headroom"]); expected = {}
        for key in ("wall_seconds", "process_tree_cpu_seconds", "cgroup_cpu_seconds"):
            expected[key] = headroom * (float(training[key]) * training_scale + float(evaluation[key]) * evaluation_scale + float(persistence[key]))
        expected["peak_process_tree_rss_bytes"] = int(root["resource"]["peak_process_tree_rss_bytes"] * float(cfg["resource_budget"]["rss_headroom"]))
        soft = {key: value * float(cfg["resource_budget"]["soft_multiplier"]) for key, value in expected.items()}; hard = {key: value * float(cfg["resource_budget"]["hard_multiplier"]) for key, value in expected.items()}
        timing = root["candidate_timing"]; nominal = float(timing["resource"]["wall_seconds"]) / int(timing["cases"]) * int(cfg["evaluation"]["candidate_projection_cases"])
        route_walls = [float(root["arms"][name]["evaluation"]["128"]["route"]["resource"]["wall_seconds"]) for name in cfg["arms"]]
        variability = max(0.10, float(np.std(route_walls)) / max(float(np.mean(route_walls)), 1e-12)); guarded = nominal * max(headroom, 1.0 + variability)
        primary = root["arms"]["expert_choice"]["evaluation"]["128"]["route"]; fixed = root["arms"]["fixed_full"]["evaluation"]["128"]["route"]
        identities = all(value["inference_identity"]["fixed_full_initialization"]["bit_logits_equal"] and value["inference_identity"]["fixed_full_initialization"]["quotient_logits_equal"] for value in root["arms"].values())
        artifact_bytes = sum(int(value["checkpoint"]["bytes"]) for value in root["arms"].values()); memory_available = psutil.virtual_memory().available
        checks = {"complete_route": set(root["arms"]) == set(cfg["arms"]), "all_modes_calibrated": int(cfg["training"]["calibration_steps"]) >= len(cfg["training"]["mode_schedule"]), "fixed_full_parity": identities, "deterministic_cpu": all(value["inference_identity"]["deterministic_repeat_logits_equal"] for value in root["arms"].values()), "noncollapsed_load": primary["all_depths_used"], "call_reduction": primary["call_reduction"] >= float(cfg["gates"]["minimum_call_reduction"]), "route_wall": float(primary["resource"]["wall_seconds"]) <= float(fixed["resource"]["wall_seconds"]), "candidate_runtime": guarded <= float(cfg["gates"]["maximum_guarded_candidate_seconds"]), "artifact": artifact_bytes < int(cfg["gates"]["maximum_artifact_bytes"]), "memory": int(hard["peak_process_tree_rss_bytes"]) < int(memory_available * 0.80), "expected_value": expected["wall_seconds"] < 86400.0, "no_fidelity_reduction": cfg["resource_budget"]["fidelity_reduced"] is False}
        decision = "admit-root-20261271" if all(checks.values()) else "park-calibration-resource-or-topology"
        summary_path = self.output / "calibration" / f"root-{self.spec().root}-summary.json"
        admission = {"schema_version": "exp680-dynamic-admission-v1", "root": self.spec().root, "decision": decision, "source": root["source"], "calibration_summary": str(summary_path.relative_to(ROOT)), "calibration_sha256": sha256(summary_path), "measured": root["resource"], "training_component": training, "evaluation_component": evaluation, "persistence_component": persistence, "training_scale": training_scale, "evaluation_scale": evaluation_scale, "variability": variability, "expected": expected, "eta_lower_wall_seconds": expected["wall_seconds"] / headroom, "eta_upper_wall_seconds": soft["wall_seconds"], "soft": soft, "hard": hard, "candidate_nominal_seconds": nominal, "candidate_guarded_seconds": guarded, "candidate_ceiling_seconds": cfg["gates"]["maximum_guarded_candidate_seconds"], "artifact_bytes": artifact_bytes, "available_memory_bytes": memory_available, "checks": checks, "route_comparison": {"expert": primary, "fixed_full": fixed}, "fidelity_reduced": False}
        write_json(ROOT / cfg["resource_budget"]["admission_path"], admission); write_json(self.output / "calibration.json", admission); return admission

    def load_admission(self) -> dict[str, Any]:
        path = ROOT / self.config["resource_budget"]["admission_path"]; admission = json.loads(path.read_text())
        calibration = ROOT / admission["calibration_summary"]
        if admission["decision"] != "admit-root-20261271" or admission["source"] != self.source_identity() or sha256(calibration) != admission["calibration_sha256"] or admission["fidelity_reduced"]: raise RuntimeError("formal requires exact admitted root-20261270")
        return admission

    def formal_gates(self, root: dict[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
        arms, gates = root["arms"], self.config["gates"]; primary = arms["expert_choice"]["evaluation"]["128"]; route = primary["route"]
        checks = {"w128_exact": primary["rollout"]["exact_accuracy"] >= float(gates["minimum_w128_exact"]), "w128_transition": primary["transition"]["exact_accuracy"] >= float(gates["minimum_w128_transition"]), "w128_validity": primary["rollout"]["invalids"] <= int(gates["maximum_invalids"]), "all_depths_used": route["all_depths_used"], "call_reduction": route["call_reduction"] >= float(gates["minimum_call_reduction"]), "route_intervention": min(route["interventions"][key] for key in ("route_shuffled_logit_delta", "matched_random_logit_delta", "router_zeroed_logit_delta")) >= float(gates["minimum_intervention_logit_delta"]), "weight_intervention": min(route["interventions"][key] for key in ("randomized_router_logit_delta", "core_zeroed_logit_delta")) >= float(gates["minimum_intervention_logit_delta"]), "router_gradient": arms["expert_choice"]["training"]["router_update_norm_sum"] >= float(gates["minimum_router_gradient_norm"]), "core_gradient": arms["expert_choice"]["training"]["core_update_norm_sum"] >= float(gates["minimum_core_gradient_norm"])}
        if all("128" in arms[name].get("evaluation", {}) for name in ("frozen_source", "fixed_full")):
            score = primary["rollout"]["exact_accuracy"]; checks["source_retention"] = score >= arms["frozen_source"]["evaluation"]["128"]["rollout"]["exact_accuracy"] - float(gates["maximum_w128_loss_vs_source"]); checks["fixed_full_retention"] = score >= arms["fixed_full"]["evaluation"]["128"]["rollout"]["exact_accuracy"] - float(gates["maximum_w128_loss_vs_fixed_full"]); checks["wall_no_regression"] = float(primary["runtime"]["wall_seconds"]) <= float(arms["fixed_full"]["evaluation"]["128"]["runtime"]["wall_seconds"])
        if all("256" in arms[name].get("evaluation", {}) for name in ("expert_choice", "fixed_full", "equal_average_fixed", "equal_average_two_point")):
            score = arms["expert_choice"]["evaluation"]["256"]["rollout"]["exact_accuracy"]; checks["w256_absolute"] = score >= float(gates["minimum_w256_exact"]); checks["w256_controls"] = all(score - arms[name]["evaluation"]["256"]["rollout"]["exact_accuracy"] >= float(gates["minimum_w256_gain_vs_controls"]) for name in ("fixed_full", "equal_average_fixed", "equal_average_two_point"))
        return checks, {"w128_exact": {name: value.get("evaluation", {}).get("128", {}).get("rollout", {}).get("exact_accuracy") for name, value in arms.items()}, "w256_exact": {name: value.get("evaluation", {}).get("256", {}).get("rollout", {}).get("exact_accuracy") for name, value in arms.items()}, "route": route}

    def persist_terminal(self, root: dict[str, Any], admission: dict[str, Any]) -> dict[str, Any]:
        checks, comparisons = self.formal_gates(root); decision = f"reject-exp680-{root['early_decision']}" if root.get("early_decision") else "authorize-confirmations" if all(checks.values()) and self.mode == "formal" else "confirmation-pass" if all(checks.values()) else "reject-exp680-scientific-gate"
        summary = {"schema_version": "exp680-terminal-v1", "run_id": self.config["run_id"], "status": "completed", "decision": decision, "root": root["root"], "source": root["source"], "arms": root["arms"], "candidate_timing": root["candidate_timing"], "resource": root["resource"], "gates": checks, "comparisons": comparisons, "dynamic_admission": admission, "roots": {"20261270": "consumed-calibration", "20261271": "consumed-formal", "20261272": "current" if root["root"] == 20261272 else "unopened", "20261273": "current" if root["root"] == 20261273 else "unopened"}, "limitations": ["Instance-level routing is a bounded-faithful adaptation because Candidate005 convolutionally couples spatial positions.", "Fresh generated roots only; no checked, independent, official/private, candidate, submission, promotion or completion evaluation.", "MoR transformer KV-cache claims are excluded."], "completion_marker": "terminal-persisted"}
        write_json(self.output / "terminal-summary.json", summary); write_json(ROOT / "experiments/runs/exp-680-mor-depth-routing/run.json", summary); write_json(ROOT / "evals/results/exp-680-mor-depth-routing/metrics.json", summary); return summary

    def formal(self): self.assert_frozen_clean(); admission = self.load_admission(); return self.persist_terminal(self.execute_root(), admission)
    def confirmation(self): self.assert_frozen_clean(); admission = self.load_admission(); return self.persist_terminal(self.execute_root(), admission)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=ROOT / "configs/exp_680_mor_depth_routing.yaml"); parser.add_argument("--output", type=Path, default=ROOT / ".scratch/exp-680-mor-depth-routing"); parser.add_argument("--mode", choices=("freeze", "preflight", "calibration", "formal", "confirmation"), required=True); parser.add_argument("--root", type=int); args = parser.parse_args()
    experiment = Experiment(args.config, args.output, args.mode, args.root)
    if args.mode == "freeze": result = experiment.freeze()
    elif args.mode == "preflight": result = experiment.preflight()
    elif args.mode == "calibration": result = experiment.calibration()
    elif args.mode == "formal": result = experiment.formal()
    else: result = experiment.confirmation()
    if args.mode in {"calibration", "formal", "confirmation"}:
        experiment_id = configure_mlflow(experiment.config["tracking"]["experiment_name"])
        with mlflow.start_run(run_name=f"{experiment.config['run_id']}-{args.mode}-{experiment.spec().root}") as run:
            mlflow.set_tags({"canonical_run_id": experiment.config["run_id"], "mode": args.mode, "root": str(experiment.spec().root), "decision": str(result.get("decision")), "repository_commit": git_output("rev-parse", "HEAD")}); mlflow.log_dict(result, f"{args.mode}-result.json"); mlflow.log_metrics({"wall_seconds": float(result.get("resource", result.get("measured", {})).get("wall_seconds", 0.0)), "candidate_guarded_seconds": float(result.get("candidate_guarded_seconds", 0.0))}); run_id = run.info.run_id
        write_json(experiment.output / f"{args.mode}-mlflow.json", {"experiment_id": experiment_id, "experiment_name": experiment.config["tracking"]["experiment_name"], "run_id": run_id, "tracking_uri": tracking_uri()})
    print(json.dumps(result, indent=2, sort_keys=True, default=lambda value: value.tolist() if isinstance(value, torch.Tensor) else str(value))); return 0


if __name__ == "__main__": raise SystemExit(main())
