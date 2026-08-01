"""Freeze, profile, select, and conditionally execute EXP679."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
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
from experiments.exp_679_capacity_recycling.model import (  # noqa: E402
    CapacityRecycledConvGRU,
    exact_embedding_receipt,
    select_resource_route,
)
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402


IDENTITY_PATHS = {
    "config": "configs/exp_679_capacity_recycling.yaml",
    "model": "experiments/exp_679_capacity_recycling/model.py",
    "runner": "experiments/exp_679_capacity_recycling/run.py",
    "test": "tests/test_exp_679_capacity_recycling.py",
    "preregistration": "knowledge/autoresearch/ablations/exp-679-capacity-recycling.md",
    "mechanism_map": "knowledge/autoresearch/research/2026-07-30-exp679-capacity-recycling-mechanism.md",
}


class Experiment(ContinuationExperiment):
    trained_arms = ("capacity_recycled", "reduced_only", "wider_full", "standard_full", "randomized_wider")

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
            "generator_sha256": sha256(ROOT / "sources/github/modular-arithmetic-challenge/src/modchallenge/testgen/generator.py"),
        }

    def frozen_protocol(self) -> dict[str, Any]:
        return {
            "schema_version": "exp679-frozen-protocol-v1",
            "status": "frozen-before-profiling",
            "source": self.source_identity(),
            "source_mapping": self.config["source_mapping"],
            "embedding": self.config["embedding"],
            "grid": self.config["grid"],
            "selection": self.config["selection"],
            "arms": self.config["arms"],
            "training": self.config["training"],
            "data": self.config["data"],
            "evaluation": self.config["evaluation"],
            "gates": self.config["gates"],
            "resource_budget": self.config["resource_budget"],
            "paths": IDENTITY_PATHS,
            "environment": {
                "python": sys.version,
                "torch": torch.__version__,
                "torch_threads": torch.get_num_threads(),
                "logical_cpu_count": psutil.cpu_count(logical=True),
            },
        }

    def freeze(self) -> dict[str, Any]:
        payload = self.frozen_protocol()
        path = ROOT / self.config["resource_budget"]["frozen_protocol_path"]
        write_json(path, payload)
        return {**payload, "path": str(path.relative_to(ROOT)), "sha256": sha256(path)}

    def assert_frozen_clean(self) -> None:
        path = ROOT / self.config["resource_budget"]["frozen_protocol_path"]
        if not path.is_file() or json.loads(path.read_text(encoding="utf-8")) != self.frozen_protocol():
            raise RuntimeError("execution bytes differ from the protocol frozen before profiling")
        current, expected = self.source_identity(), self.config["source_contract"]
        for key in ("challenge_commit", "paper_sha256", "research_manifest_sha256", "checkpoint_sha256", "base_model_sha256", "generator_sha256"):
            if current[key] != expected[key]:
                raise RuntimeError(f"authenticated source mismatch: {key}")

    def spec(self) -> RootSpec:
        values = self.config["data"]
        if self.mode == "calibration": selected = values["calibration"]
        elif self.mode == "formal": selected = values["formal"]
        else:
            matches = [item for item in values["confirmations"] if int(item["root"]) == int(self.root_override or -1)]
            if len(matches) != 1: raise ValueError("confirmation requires root 20261268 or 20261269")
            selected = matches[0]
        return RootSpec(**selected)

    def source_model(self, *, randomized: bool = False, seed: int = 0) -> torch.nn.Module:
        if randomized: torch.manual_seed(seed)
        model = self.cellular.SharedDilatedConvGRU(width=128, hidden=92, cycles=2)
        if not randomized: model.load_state_dict(load_file(self.checkpoint))
        return model

    def build(self, *, total_hidden: int, calls_at_w128: int | None, randomized: bool = False, seed: int = 0) -> CapacityRecycledConvGRU:
        if randomized: torch.manual_seed(seed)
        return CapacityRecycledConvGRU(self.source_model(randomized=randomized, seed=seed), total_hidden=total_hidden, calls_at_w128=calls_at_w128)

    def selected_spec(self) -> tuple[int, int]:
        path = ROOT / self.config["resource_budget"]["admission_path"]
        if not path.is_file(): raise RuntimeError("selected route requires root-20261266 admission")
        admission = json.loads(path.read_text(encoding="utf-8"))
        selected = admission.get("selected_route")
        if not selected: raise RuntimeError("calibration parked without a resource-feasible selected route")
        return int(selected["total_hidden"]), int(selected["calls_at_w128"])

    def arm_spec(self, name: str, selected: tuple[int, int]) -> tuple[int, int | None, bool]:
        hidden, calls = selected
        return {
            "capacity_recycled": (hidden, calls, False),
            "reduced_only": (92, calls, False),
            "wider_full": (hidden, None, False),
            "standard_full": (92, None, False),
            "frozen_source": (92, None, False),
            "randomized_wider": (hidden, calls, True),
        }[name]

    def profile_route(self, total_hidden: int, calls_at_w128: int | None, name: str) -> dict[str, Any]:
        model = self.build(total_hidden=total_hidden, calls_at_w128=calls_at_w128).train()
        features = torch.randn(int(self.config["evaluation"]["profiler_batch"]), 6, 128, generator=torch.Generator().manual_seed(20261266))
        probe_source = self.build(total_hidden=92, calls_at_w128=calls_at_w128).eval()
        with torch.no_grad():
            observed = model.eval()(features); expected = probe_source(features)
        embedding = torch.equal(observed[0], expected[0]) and torch.equal(observed[1], expected[1])
        walls: list[float] = []; cpu: list[float] = []; cgroup: list[float] = []; rss: list[int] = []; activations: list[int] = []
        model.train()
        for _ in range(int(self.config["evaluation"]["profiler_repeats"])):
            timer = Timer()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
                logits, quotient = model(features)
                (logits.square().mean() + quotient.square().mean()).backward()
            resource = timer.result(); model.zero_grad(set_to_none=True)
            walls.append(float(resource["wall_seconds"])); cpu.append(float(resource["process_tree_cpu_seconds"]));
            cgroup.append(float(resource["cgroup_cpu_seconds"] or 0.0)); rss.append(int(resource["peak_process_tree_rss_bytes"]))
            activations.append(sum(max(0, int(event.cpu_memory_usage)) for event in prof.events()))
        median = statistics.median(walls); variability = max(0.10, statistics.pstdev(walls) / max(statistics.mean(walls), 1e-12))
        projected = median / int(self.config["evaluation"]["profiler_batch"]) * int(self.config["evaluation"]["candidate_projection_cases"])
        guarded = projected * max(float(self.config["resource_budget"]["formal_headroom"]), 1.0 + variability)
        return {
            "name": name, "total_hidden": total_hidden, "calls_at_w128": 14 if calls_at_w128 is None else calls_at_w128,
            "route_kind": "full" if calls_at_w128 is None else "reduced", "embedding_passed": embedding,
            "wall_seconds_repeats": walls, "wall_seconds_median": median, "wall_variability_cv": variability,
            "process_tree_cpu_seconds_repeats": cpu, "cgroup_cpu_seconds_repeats": cgroup,
            "peak_rss_bytes": max(rss), "activation_bytes": max(activations),
            "projected_candidate_seconds": projected, "guarded_candidate_seconds": guarded,
            "accounting": model.accounting(128), "profiler": "torch.profiler CPU profile_memory record_shapes",
        }

    def profile_grid(self) -> dict[str, Any]:
        frozen = self.profile_route(92, None, "frozen_source")
        grid = [self.profile_route(hidden, calls, f"grid-h{hidden}-c{calls}") for hidden in self.config["grid"]["total_hidden_widths"] for calls in self.config["grid"]["reduced_calls_w128"]]
        controls = [
            self.profile_route(92, calls, f"reduced-only-c{calls}") for calls in self.config["grid"]["reduced_calls_w128"]
        ] + [self.profile_route(hidden, None, f"wider-full-h{hidden}") for hidden in self.config["grid"]["total_hidden_widths"]]
        selected = select_resource_route(grid, frozen=frozen, rss_headroom=float(self.config["resource_budget"]["rss_headroom"]), activation_headroom=float(self.config["resource_budget"]["activation_headroom"]))
        return {"frozen": frozen, "grid": grid, "controls": controls, "selected": selected, "selection_fields": ["embedding_passed", "guarded_candidate_seconds", "peak_rss_bytes", "activation_bytes", "total_hidden", "calls_at_w128"]}

    def inference_identity(self, model: CapacityRecycledConvGRU, checkpoint: Path, seed: int, spec: tuple[int, int | None, bool]) -> dict[str, Any]:
        hidden, calls, randomized = spec
        reloaded = self.build(total_hidden=hidden, calls_at_w128=calls, randomized=randomized, seed=seed)
        reloaded.load_state_dict(load_file(checkpoint)); reloaded.eval(); model.eval()
        features = torch.randn(16, 6, 128, generator=torch.Generator().manual_seed(seed))
        with torch.no_grad(): current = model(features); restored = reloaded(features)
        return {
            "state_dict_identity": state_dict_digest(model.state_dict()) == state_dict_digest(reloaded.state_dict()),
            "pre_save_vs_reload_logits_equal": torch.equal(current[0], restored[0]),
            "pre_save_vs_reload_quotient_equal": torch.equal(current[1], restored[1]),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "accounting": model.accounting(128),
        }

    def causality_probe(self, model: CapacityRecycledConvGRU, spec: RootSpec) -> dict[str, Any]:
        if model.extra_hidden == 0: return {"applicable": False, "extra_gradient_norm": 0.0, "extra_ablation_logit_delta": 0.0}
        sampler = self.exp300.BigIntSampler(self.split(64, spec.root)["test"], spec.eval_seed + 6000)
        state, right, modulus, bit, target, quotient = sampler.transitions(16)
        features = self.exp300.features(state, right, modulus, bit, 64)
        model.train(); model.zero_grad(set_to_none=True)
        loss, logits = self.transition_position_losses(model, features, target, quotient, 64); loss.backward()
        gradient = sum(float(parameter.grad.norm()) for parameter in model.extra_parameters() if parameter.grad is not None)
        with torch.no_grad(): ordinary = model(features)[0]; ablated = model(features, ablate_extra=True)[0]
        model.zero_grad(set_to_none=True); model.eval()
        return {"applicable": True, "extra_gradient_norm": gradient, "extra_ablation_logit_delta": float((ordinary - ablated).abs().max()), "probe_loss": float(loss.detach()), "labels_used_for_selection": False}

    def train_and_cache(self, spec: RootSpec, steps: int, session: Timer, selected: tuple[int, int]) -> tuple[dict[str, CapacityRecycledConvGRU], dict[str, Any]]:
        arms: dict[str, CapacityRecycledConvGRU] = {}; records: dict[str, Any] = {}
        for name, cfg in self.config["arms"].items():
            hidden, calls, randomized = self.arm_spec(name, selected)
            seed = spec.train_seed + 900 if randomized else 0
            model = self.build(total_hidden=hidden, calls_at_w128=calls, randomized=randomized, seed=seed)
            initial = state_dict_digest(model.state_dict()); training_timer = Timer()
            training = self.train_arm(model, spec, name, steps, session) if bool(cfg["train"]) else {"steps": 0, "reason": "frozen-source", "work": {"updates": 0, "examples": 0, "forward_calls": 0, "ce_tensor_calls": 0, "effective_supervised_positions": 0}}
            training["runtime"] = training_timer.result()
            path = self.output / self.mode / f"root-{spec.root}" / f"{name}.safetensors"; persistence = Timer(); path.parent.mkdir(parents=True, exist_ok=True)
            save_file({key: value.detach().contiguous() for key, value in model.state_dict().items()}, path)
            identity = self.inference_identity(model, path, spec.eval_seed + len(arms), (hidden, calls, randomized)); persistence_resource = persistence.result()
            if not all(identity[key] for key in ("state_dict_identity", "pre_save_vs_reload_logits_equal", "pre_save_vs_reload_quotient_equal")): raise RuntimeError(f"reload identity failed: {name}")
            arms[name] = model.eval(); records[name] = {
                "route": {"total_hidden": hidden, "calls_at_w128": 14 if calls is None else calls, "route_kind": "full" if calls is None else "reduced"},
                "training": training, "checkpoint": {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size},
                "initial_state_sha256": initial, "inference_identity": identity, "causality": self.causality_probe(model, spec),
                "persistence_runtime": persistence_resource, "parameters": identity["parameter_count"],
            }
            write_json(self.output / self.mode / f"root-{spec.root}-progress.json", {"root": spec.root, "completed_arms": list(records), "completion_marker": "training-arms-in-progress"})
        return arms, records

    def execute_root(self, selected: tuple[int, int]) -> dict[str, Any]:
        spec = self.spec(); transitions, rollouts, trajectory_cases, steps = self.root_denominators(); session = Timer()
        arms, records = self.train_and_cache(spec, steps, session, selected)
        def score(name: str, widths: list[int]) -> None:
            evaluation, runtime = self.evaluate_arm(arms[name], spec, widths, transitions, rollouts, trajectory_cases)
            records[name].setdefault("evaluation", {}).update(evaluation); records[name]["evaluation_runtime"] = runtime
        early: str | None = None
        if self.mode == "calibration":
            for name in arms: score(name, [64, 128, 256])
        else:
            score("capacity_recycled", [128]); primary = records["capacity_recycled"]["evaluation"]["128"]
            if primary["rollout"]["exact_accuracy"] < float(self.config["gates"]["minimum_w128_exact"]): early = "reject-W128-primary-exact"
            elif primary["transition"]["exact_accuracy"] < float(self.config["gates"]["minimum_w128_transition"]): early = "reject-W128-primary-transition"
            elif primary["rollout"]["invalids"] > int(self.config["gates"]["maximum_invalids"]): early = "reject-W128-primary-validity"
            if early is None:
                for name in ("frozen_source", "standard_full", "reduced_only", "wider_full", "randomized_wider"): score(name, [128])
            if early is None:
                target = primary["rollout"]["exact_accuracy"]
                if target < records["frozen_source"]["evaluation"]["128"]["rollout"]["exact_accuracy"] - float(self.config["gates"]["maximum_w128_loss_vs_source"]): early = "reject-W128-source-retention"
                elif records["capacity_recycled"]["causality"]["extra_gradient_norm"] < float(self.config["gates"]["minimum_extra_gradient_norm"]): early = "reject-extra-gradient"
                elif records["capacity_recycled"]["causality"]["extra_ablation_logit_delta"] < float(self.config["gates"]["minimum_extra_ablation_logit_delta"]): early = "reject-extra-ablation"
                elif target - records["randomized_wider"]["evaluation"]["128"]["rollout"]["exact_accuracy"] < float(self.config["gates"]["minimum_trained_minus_randomized_w128"]): early = "reject-trained-weight-dependence"
            if early is None:
                for name in arms: score(name, [64, 256])
        result = {"schema_version": "exp679-root-v1", "mode": self.mode, "root": spec.root, "source": self.source_identity(), "selected_route": {"total_hidden": selected[0], "calls_at_w128": selected[1]}, "arms": records, "resource": session.result(), "early_decision": early, "completion_marker": "root-complete"}
        write_json(self.output / self.mode / f"root-{spec.root}-summary.json", result); return result

    def calibration(self) -> dict[str, Any]:
        self.assert_frozen_clean(); profile_timer = Timer(); profiles = self.profile_grid(); profile_resource = profile_timer.result(); selected_row = profiles["selected"]
        if selected_row is None:
            root = None; selected = None
        else:
            selected = (int(selected_row["total_hidden"]), int(selected_row["calls_at_w128"])); root = self.execute_root(selected)
        cfg = self.config; expected: dict[str, Any] = {}; headroom = float(cfg["resource_budget"]["formal_headroom"])
        if root is not None:
            training = aggregate([root["arms"][name]["training"]["runtime"] for name in self.trained_arms])
            evaluation = aggregate([root["arms"][name]["evaluation_runtime"] for name in cfg["arms"]])
            persistence = aggregate([root["arms"][name]["persistence_runtime"] for name in cfg["arms"]])
            training_scale = int(cfg["training"]["steps"]) / int(cfg["training"]["calibration_steps"])
            eval_scale = max(sum(cfg["evaluation"]["transition_cases"].values()) / sum(cfg["evaluation"]["calibration_transition_cases"].values()), sum(cfg["evaluation"]["rollout_cases"].values()) / sum(cfg["evaluation"]["calibration_rollout_cases"].values()))
            for key in ("wall_seconds", "process_tree_cpu_seconds", "cgroup_cpu_seconds"):
                expected[key] = headroom * (float(profile_resource[key] or 0) + float(training[key] or 0) * training_scale + float(evaluation[key] or 0) * eval_scale + float(persistence[key] or 0))
            expected["peak_process_tree_rss_bytes"] = int(max(root["resource"]["peak_process_tree_rss_bytes"], profile_resource["peak_process_tree_rss_bytes"]) * float(cfg["resource_budget"]["rss_headroom"]))
            soft = {key: value * float(cfg["resource_budget"]["soft_multiplier"]) for key, value in expected.items()}; hard = {key: value * float(cfg["resource_budget"]["hard_multiplier"]) for key, value in expected.items()}
            selected_profile = selected_row; frozen = profiles["frozen"]
            checks = {
                "topology": bool(selected_profile["embedding_passed"]), "complete_routes": len(root["arms"]) == 6 and len(profiles["grid"]) == 4 and len(profiles["controls"]) == 4,
                "runtime": float(selected_profile["guarded_candidate_seconds"]) <= 285 and float(selected_profile["guarded_candidate_seconds"]) <= float(frozen["guarded_candidate_seconds"]),
                "rss": int(selected_profile["peak_rss_bytes"]) <= int(float(frozen["peak_rss_bytes"]) * float(cfg["resource_budget"]["rss_headroom"])),
                "activation": int(selected_profile["activation_bytes"]) <= int(float(frozen["activation_bytes"]) * float(cfg["resource_budget"]["activation_headroom"])),
                "expected_value": True, "fidelity": cfg["resource_budget"]["fidelity_reduced"] is False,
            }
        else:
            training = evaluation = persistence = None; training_scale = eval_scale = None; soft = hard = {}; checks = {"resource_selection": False}
        decision = "admit-root-20261267" if selected is not None and all(checks.values()) else "park-root-20261266-resource-or-topology"
        summary_path = self.output / "calibration" / "root-20261266-summary.json"
        admission = {"schema_version": "exp679-dynamic-admission-v1", "root": 20261266, "decision": decision, "source": self.source_identity(), "profiles": profiles, "profile_resource": profile_resource, "calibration_summary": str(summary_path.relative_to(ROOT)) if root else None, "calibration_sha256": sha256(summary_path) if root else None, "selected_route": None if selected is None else {"total_hidden": selected[0], "calls_at_w128": selected[1]}, "components": {"training": training, "evaluation": evaluation, "persistence": persistence}, "scales": {"training": training_scale, "evaluation": eval_scale}, "expected": expected, "eta_lower_wall_seconds": None if not expected else expected["wall_seconds"] / headroom, "eta_upper_wall_seconds": soft.get("wall_seconds"), "soft": soft, "hard": hard, "checks": checks, "fidelity_reduced": False, "completion_marker": "calibration-complete"}
        write_json(ROOT / cfg["resource_budget"]["admission_path"], admission); write_json(self.output / "calibration.json", admission); write_json(ROOT / cfg["resource_budget"]["completion_marker_path"], {"root": 20261266, "state": "complete", "decision": decision})
        return admission

    def load_admission(self) -> dict[str, Any]:
        path = ROOT / self.config["resource_budget"]["admission_path"]; admission = json.loads(path.read_text(encoding="utf-8"))
        summary = ROOT / admission["calibration_summary"] if admission.get("calibration_summary") else None
        if admission["decision"] != "admit-root-20261267" or admission["source"] != self.source_identity() or summary is None or sha256(summary) != admission["calibration_sha256"]: raise RuntimeError("formal requires exact admitted root-20261266 evidence")
        return admission

    def gates(self, root: dict[str, Any]) -> dict[str, bool]:
        records, cfg = root["arms"], self.config["gates"]; primary = records["capacity_recycled"].get("evaluation", {}).get("128")
        if primary is None: return {"w128_primary": False}
        checks = {"w128_exact": primary["rollout"]["exact_accuracy"] >= float(cfg["minimum_w128_exact"]), "w128_transition": primary["transition"]["exact_accuracy"] >= float(cfg["minimum_w128_transition"]), "w128_validity": primary["rollout"]["invalids"] <= int(cfg["maximum_invalids"]), "extra_gradient": records["capacity_recycled"]["causality"]["extra_gradient_norm"] >= float(cfg["minimum_extra_gradient_norm"]), "extra_ablation": records["capacity_recycled"]["causality"]["extra_ablation_logit_delta"] >= float(cfg["minimum_extra_ablation_logit_delta"])}
        if all("128" in records[name].get("evaluation", {}) for name in records):
            target = primary["rollout"]["exact_accuracy"]
            checks["source_retention"] = target >= records["frozen_source"]["evaluation"]["128"]["rollout"]["exact_accuracy"] - float(cfg["maximum_w128_loss_vs_source"])
            checks["trained_weight_dependence"] = target - records["randomized_wider"]["evaluation"]["128"]["rollout"]["exact_accuracy"] >= float(cfg["minimum_trained_minus_randomized_w128"])
        if all("256" in records[name].get("evaluation", {}) for name in ("capacity_recycled", "reduced_only", "standard_full")):
            target = records["capacity_recycled"]["evaluation"]["256"]["rollout"]["exact_accuracy"]
            checks["w256_exact"] = target >= float(cfg["minimum_w256_exact"])
            checks["w256_controls"] = all(target - records[name]["evaluation"]["256"]["rollout"]["exact_accuracy"] >= float(cfg["minimum_w256_gain_vs_controls"]) for name in ("reduced_only", "standard_full"))
        return checks

    def persist_terminal(self, root: dict[str, Any], admission: dict[str, Any]) -> dict[str, Any]:
        checks = self.gates(root); decision = f"reject-exp679-{root['early_decision']}" if root.get("early_decision") else "authorize-confirmations" if all(checks.values()) and self.mode == "formal" else "confirmation-pass" if all(checks.values()) else "reject-exp679-scientific-gate"
        summary = {"schema_version": "exp679-terminal-v1", "run_id": self.config["run_id"], "status": "completed", "decision": decision, "root": root["root"], "source": root["source"], "selected_route": root["selected_route"], "arms": root["arms"], "resource": root["resource"], "gates": checks, "dynamic_admission": admission, "roots": {"20261266": "consumed-calibration", "20261267": "consumed-formal" if root["root"] == 20261267 else "prior", "20261268": "current" if root["root"] == 20261268 else "unopened", "20261269": "current" if root["root"] == 20261269 else "unopened"}, "limitations": ["Fresh generated experiment identities only; checked, independent, official, private, candidate, submission, promotion, and completion evaluation were not opened.", "This bounded dense analogue tests M043.b operational capacity recycling, not M043.a fixed-R2 ordering, M043.c MoE, or M009.b learned routing.", "Exact arithmetic is confined to offline labels and evaluation; resource selection is label-free."], "completion_marker": "terminal-persisted"}
        write_json(self.output / "terminal-summary.json", summary); write_json(ROOT / "experiments/runs/exp-679-capacity-recycling/run.json", summary); write_json(ROOT / "evals/results/exp-679-capacity-recycling/metrics.json", summary); write_json(ROOT / self.config["resource_budget"]["completion_marker_path"], {"root": root["root"], "state": "complete", "decision": decision}); return summary

    def formal(self) -> dict[str, Any]:
        self.assert_frozen_clean(); admission = self.load_admission(); return self.persist_terminal(self.execute_root(self.selected_spec()), admission)

    def confirmation(self) -> dict[str, Any]:
        self.assert_frozen_clean(); admission = self.load_admission(); return self.persist_terminal(self.execute_root(self.selected_spec()), admission)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=ROOT / "configs/exp_679_capacity_recycling.yaml"); parser.add_argument("--output", type=Path, default=ROOT / ".scratch/exp-679-capacity-recycling"); parser.add_argument("--mode", choices=("freeze", "calibration", "formal", "confirmation"), required=True); parser.add_argument("--root", type=int); args = parser.parse_args()
    experiment = Experiment(args.config, args.output, args.mode, args.root)
    if args.mode == "freeze": result = experiment.freeze()
    else: result = experiment.calibration() if args.mode == "calibration" else experiment.formal() if args.mode == "formal" else experiment.confirmation()
    if args.mode != "freeze":
        experiment_id = configure_mlflow(experiment.config["tracking"]["experiment_name"])
        with mlflow.start_run(run_name=f"{experiment.config['run_id']}-{args.mode}-{experiment.spec().root}") as run:
            mlflow.set_tags({"canonical_run_id": experiment.config["run_id"], "mode": args.mode, "root": str(experiment.spec().root), "decision": str(result.get("decision")), "repository_commit": git_output("rev-parse", "HEAD")}); mlflow.log_dict(result, f"{args.mode}-result.json"); mlflow.log_metrics({"wall_seconds": float(result.get("resource", result.get("profile_resource", {})).get("wall_seconds", 0.0)), "candidate_guarded_seconds": float((result.get("profiles") or {}).get("selected", {}).get("guarded_candidate_seconds", 0.0))}); mlflow_run_id = run.info.run_id
        receipt = {"experiment_id": experiment_id, "experiment_name": experiment.config["tracking"]["experiment_name"], "run_id": mlflow_run_id, "tracking_uri": tracking_uri()}; write_json(experiment.output / f"{args.mode}-mlflow.json", receipt); result["mlflow"] = receipt
    print(json.dumps({"mode": args.mode, "root": None if args.mode == "freeze" else experiment.spec().root, "decision": result.get("decision"), "resource": result.get("resource", result.get("profile_resource")), "gates": result.get("gates", result.get("checks")), "selected_route": result.get("selected_route"), "mlflow": result.get("mlflow")}, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
