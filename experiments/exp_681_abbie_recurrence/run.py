"""Freeze, preflight, calibrate, and conditionally execute EXP681."""

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
from experiments.exp_681_abbie_recurrence.model import AbbieConvGRU, RecurrenceView, randomize_body, source_identity_receipt  # noqa: E402
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402


IDENTITY_PATHS = {
    "config": "configs/exp_681_abbie_recurrence.yaml",
    "model": "experiments/exp_681_abbie_recurrence/model.py",
    "runner": "experiments/exp_681_abbie_recurrence/run.py",
    "test": "tests/test_exp_681_abbie_recurrence.py",
    "preregistration": "knowledge/autoresearch/ablations/exp-681-abbie-recurrence.md",
    "mechanism_map": "knowledge/autoresearch/research/2026-07-30-exp681-abbie-recurrence-mechanism.md",
}


def tensor_delta(left: tuple[torch.Tensor, torch.Tensor], right: tuple[torch.Tensor, torch.Tensor]) -> float:
    return max(float((left[0] - right[0]).abs().max()), float((left[1] - right[1]).abs().max()))


class Experiment(ContinuationExperiment):
    trained_arms = ("abbie_c", "abbie_d", "untied_body", "randomized_body")

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
            "schema_version": "exp681-frozen-protocol-v1", "status": "frozen-before-package-oracle",
            "source": self.source_identity(), "source_mapping": cfg["source_mapping"],
            "library_decision": cfg["library_decision"], "recurrence": cfg["recurrence"],
            "arms": cfg["arms"], "interventions": cfg["interventions"], "training": cfg["training"],
            "data": cfg["data"], "evaluation": cfg["evaluation"], "gates": cfg["gates"],
            "resource_budget": cfg["resource_budget"], "paths": IDENTITY_PATHS,
            "environment": {"python": sys.version, "torch": torch.__version__, "torch_threads": torch.get_num_threads(), "logical_cpu_count": psutil.cpu_count(logical=True)},
        }

    def freeze(self) -> dict[str, Any]:
        payload = self.frozen_protocol(); path = ROOT / self.config["resource_budget"]["frozen_protocol_path"]
        write_json(path, payload)
        return {**payload, "path": str(path.relative_to(ROOT)), "sha256": sha256(path)}

    def assert_frozen_clean(self) -> None:
        path = ROOT / self.config["resource_budget"]["frozen_protocol_path"]
        current = json.loads(json.dumps(self.frozen_protocol(), sort_keys=True))
        if not path.is_file() or json.loads(path.read_text(encoding="utf-8")) != current:
            raise RuntimeError("current EXP681 package/source bytes differ from frozen protocol")
        observed, expected = self.source_identity(), self.config["source_contract"]
        for key in ("challenge_commit", "paper_sha256", "research_manifest_sha256", "checkpoint_sha256", "base_model_sha256", "exp300_sha256", "exp329_sha256", "generator_sha256"):
            if observed[key] != expected[key]:
                raise RuntimeError(f"authenticated source mismatch: {key}")

    def build(self, arm_name: str, *, seed: int) -> AbbieConvGRU:
        arm = self.config["arms"][arm_name]; torch.manual_seed(seed)
        model = AbbieConvGRU(self.base(), form=str(arm["form"]), default_recurrence=2, untied=bool(arm["untied"]))
        if bool(arm["randomized_body"]): randomize_body(model, seed + 8100)
        return model

    def spec(self) -> RootSpec:
        values = self.config["data"]
        if self.mode == "calibration": selected = values["calibration"]
        elif self.mode == "formal": selected = values["formal"]
        else:
            matches = [item for item in values["confirmations"] if int(item["root"]) == int(self.root_override or -1)]
            if len(matches) != 1: raise ValueError("confirmation requires root 20261276 or 20261277")
            selected = matches[0]
        return RootSpec(**selected)

    def root_denominators(self) -> tuple[dict[int, int], dict[int, int], int, int]:
        calibration = self.mode == "calibration"; evaluation = self.config["evaluation"]
        transitions = {int(k): int(v) for k, v in evaluation["calibration_transition_cases" if calibration else "transition_cases"].items()}
        rollouts = {int(k): int(v) for k, v in evaluation["calibration_rollout_cases" if calibration else "rollout_cases"].items()}
        trajectory = int(evaluation["calibration_trajectory_loss_cases" if calibration else "trajectory_loss_cases"])
        steps = int(self.config["training"]["calibration_steps" if calibration else "steps"])
        return transitions, rollouts, trajectory, steps

    def inference_identity(self, model: AbbieConvGRU, checkpoint: Path, seed: int, arm_name: str) -> dict[str, Any]:
        restored = self.build(arm_name, seed=seed); restored.load_state_dict(load_file(checkpoint)); restored.eval(); model.eval()
        probe = torch.randn(int(self.config["evaluation"]["deterministic_repeat_cases"]), 6, 128, generator=torch.Generator().manual_seed(seed))
        with torch.no_grad(): current = model(probe); loaded = restored(probe); repeated = restored(probe)
        return {
            "state_dict_identity": state_dict_digest(model.state_dict()) == state_dict_digest(restored.state_dict()),
            "pre_save_vs_reload_logits_equal": torch.equal(current[0], loaded[0]),
            "pre_save_vs_reload_quotient_equal": torch.equal(current[1], loaded[1]),
            "deterministic_repeat_logits_equal": torch.equal(loaded[0], repeated[0]),
            "deterministic_repeat_quotient_equal": torch.equal(loaded[1], repeated[1]),
            "accounting": model.accounting(128, 2),
        }

    def train_and_cache(self, spec: RootSpec, steps: int, session: Timer) -> tuple[dict[str, AbbieConvGRU], dict[str, Any]]:
        models: dict[str, AbbieConvGRU] = {}; records: dict[str, Any] = {}
        for offset, (name, arm) in enumerate(self.config["arms"].items()):
            model = self.build(name, seed=spec.train_seed + offset); initial = state_dict_digest(model.state_dict()); training_timer = Timer()
            initialization_identity = source_identity_receipt(self.base(), model, torch.randn(4, 6, 128, generator=torch.Generator().manual_seed(spec.eval_seed + offset))) if name in {"abbie_c", "frozen_source"} else None
            training = self.train_arm(model, spec, name, steps, session) if bool(arm["train"]) else {"steps": 0, "reason": "frozen-source", "work": {"updates": 0}}
            training["runtime"] = training_timer.result(); changes = training.get("parameter_update_norms", {})
            training["body_update_norm_sum"] = sum(float(v) for k, v in changes.items() if k.startswith("bodies."))
            path = self.checkpoint_path(spec, name); persistence = Timer(); path.parent.mkdir(parents=True, exist_ok=True)
            save_file({key: value.detach().contiguous() for key, value in model.state_dict().items()}, path)
            identity = self.inference_identity(model, path, spec.eval_seed + offset, name)
            keys = ("state_dict_identity", "pre_save_vs_reload_logits_equal", "pre_save_vs_reload_quotient_equal", "deterministic_repeat_logits_equal", "deterministic_repeat_quotient_equal")
            if not all(bool(identity[key]) for key in keys): raise RuntimeError(f"reload/determinism failed: {name}")
            records[name] = {"form": arm["form"], "untied": arm["untied"], "randomized_body": arm["randomized_body"], "training": training, "checkpoint": {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}, "initial_state_sha256": initial, "initialization_source_identity": initialization_identity, "inference_identity": identity, "persistence_runtime": persistence.result(), "parameters": identity["accounting"]["stored_parameters"]}
            models[name] = model.eval()
            write_json(self.output / self.mode / f"root-{spec.root}-progress.json", {"schema_version": "exp681-root-progress-v1", "root": spec.root, "completed_arms": list(records), "completion_marker": "training-arms-in-progress"})
        return models, records

    @torch.no_grad()
    def recurrence_diagnostic(self, model: AbbieConvGRU, spec: RootSpec, width: int) -> dict[str, Any]:
        cases = int(self.config["evaluation"]["recurrence_diagnostic_cases"])
        features = torch.randn(cases, 6, width, generator=torch.Generator().manual_seed(spec.eval_seed + 6000 + width))
        timer = Timer(); outputs = {r: model(features, recurrence=r, return_trace=True) for r in (1, 2, 4, 8)}
        ordinary = outputs[2]
        interventions = {name: tensor_delta(ordinary[:2], model(features, recurrence=2, **kwargs)) for name, kwargs in {
            "reset_state": {"reset_state": True}, "shuffle_state": {"shuffle_state": True},
            "head_zeroed": {"ablate_head": True}, "body_bypassed": {"ablate_body": True}, "tail_zeroed": {"ablate_tail": True},
        }.items()}
        randomized = copy.deepcopy(model); randomize_body(randomized, spec.eval_seed + width)
        interventions["randomized_body"] = tensor_delta(ordinary[:2], randomized(features, recurrence=2))
        return {"cases": cases, "recurrences": {str(r): {"body_calls": outputs[r][2]["body_calls"], "state_norms": outputs[r][2]["state_norms"], "delta_vs_r2": tensor_delta(ordinary[:2], outputs[r][:2])} for r in (1, 2, 4, 8)}, "interventions": interventions, "resource": timer.result(), "labels_or_answers_used": False}

    def evaluate_recurrence(self, model: AbbieConvGRU, recurrence: int, spec: RootSpec, widths: list[int], transitions: dict[int, int], rollouts: dict[int, int], trajectory: int) -> tuple[dict[str, Any], dict[str, Any]]:
        return super().evaluate_arm(RecurrenceView(model, recurrence), spec, widths, transitions, rollouts, trajectory)

    def execute_root(self) -> dict[str, Any]:
        spec = self.spec(); transitions, rollouts, trajectory, steps = self.root_denominators(); session = Timer()
        models, records = self.train_and_cache(spec, steps, session); early = None
        def score(name: str, recurrence: int, widths: list[int]) -> None:
            evaluation, runtime = self.evaluate_recurrence(models[name], recurrence, spec, widths, transitions, rollouts, trajectory)
            records[name].setdefault("evaluation", {})[str(recurrence)] = evaluation
            records[name].setdefault("evaluation_runtime", {})[str(recurrence)] = runtime
        if self.mode == "calibration":
            for name in models:
                for recurrence in (1, 2, 4, 8): score(name, recurrence, [64, 128, 256])
        else:
            score("abbie_c", 2, [128]); primary = records["abbie_c"]["evaluation"]["2"]["128"]
            if primary["rollout"]["exact_accuracy"] < float(self.config["gates"]["minimum_w128_exact"]): early = "reject-W128-primary-exact"
            elif primary["transition"]["exact_accuracy"] < float(self.config["gates"]["minimum_w128_transition"]): early = "reject-W128-primary-transition"
            elif primary["rollout"]["invalids"] > int(self.config["gates"]["maximum_invalids"]): early = "reject-W128-primary-validity"
            if early is None:
                for name in models: score(name, 2, [128])
                for recurrence in (1, 4, 8): score("abbie_c", recurrence, [128])
                checks, _ = self.formal_gates({"arms": records})
                if not all(value for key, value in checks.items() if not key.startswith("w256")): early = "reject-W128-recurrence-or-control-gate"
            if early is None:
                for name in models:
                    for recurrence in (2, 4, 8): score(name, recurrence, [256])
        for name, model in models.items(): records[name]["diagnostic"] = self.recurrence_diagnostic(model, spec, 128)
        timing_timer = Timer(); timing_cases = int(self.config["evaluation"]["candidate_timing_cases"]); timing_width = int(self.config["evaluation"]["candidate_timing_width"])
        timing = self.evaluate_rollouts(RecurrenceView(models["abbie_c"], 2), self.exp300.BigIntSampler(self.split(timing_width, spec.root)["test"], spec.eval_seed + 9000), timing_cases, timing_width)
        result = {"schema_version": "exp681-root-v1", "mode": self.mode, "root": spec.root, "source": self.source_identity(), "arms": records, "candidate_timing": {"width": timing_width, "cases": timing_cases, "rollout": timing, "resource": timing_timer.result()}, "resource": session.result(), "early_decision": early, "completion_marker": "root-complete"}
        write_json(self.output / self.mode / f"root-{spec.root}-summary.json", result); return result

    def preflight(self) -> dict[str, Any]:
        self.assert_frozen_clean(); original = (self.mode, self.root_override); constructed = []
        for mode, root in (("calibration", None), ("formal", None), ("confirmation", 20261276), ("confirmation", 20261277)):
            self.mode, self.root_override = mode, root; spec = self.spec(); denominators = self.root_denominators()
            constructed.append({"mode": mode, "root": spec.root, "train_seed": spec.train_seed, "eval_seed": spec.eval_seed, "transition_cases": denominators[0], "rollout_cases": denominators[1], "trajectory_cases": denominators[2], "steps": denominators[3]})
        self.mode, self.root_override = original
        dry_runs = {}; identities = {}
        for offset, name in enumerate(self.config["arms"]):
            model = self.build(name, seed=681000 + offset).eval(); dry_runs[name] = {}
            for width in self.config["evaluation"]["widths"]:
                features = torch.randn(3, 6, int(width), generator=torch.Generator().manual_seed(681000 + int(width)))
                with torch.no_grad(): output = model(features, recurrence=2, return_trace=True)
                dry_runs[name][str(width)] = {"logits_shape": list(output[0].shape), "body_calls": output[2]["body_calls"]}
            if name in {"abbie_c", "frozen_source"}: identities[name] = source_identity_receipt(self.base(), model, torch.randn(3, 6, 128))
        receipt = {"schema_version": "exp681-preflight-v1", "status": "passed", "source": self.source_identity(), "constructed_modes": constructed, "dry_runs": dry_runs, "source_identities": identities, "all_paths_exist": all((ROOT / path).exists() for path in IDENTITY_PATHS.values()), "recurrence_grid": [1, 2, 4, 8], "training_recurrence": 2, "prohibited_inputs": ["labels", "answers", "correctness", "validity"], "completion_marker": "preflight-complete"}
        if not receipt["all_paths_exist"] or not all(v["bit_logits_equal"] and v["quotient_logits_equal"] for v in identities.values()): raise RuntimeError("EXP681 preflight identity/path failure")
        write_json(ROOT / self.config["resource_budget"]["preflight_path"], receipt); return receipt

    def calibration(self) -> dict[str, Any]:
        self.assert_frozen_clean(); preflight = ROOT / self.config["resource_budget"]["preflight_path"]
        if not preflight.is_file() or json.loads(preflight.read_text())["source"] != self.source_identity(): raise RuntimeError("calibration requires exact completed preflight")
        root = self.execute_root(); cfg = self.config
        training = aggregate([root["arms"][name]["training"]["runtime"] for name in self.trained_arms])
        persistence = aggregate([root["arms"][name]["persistence_runtime"] for name in cfg["arms"]])
        evaluation = aggregate([value for arm in root["arms"].values() for value in arm["evaluation_runtime"].values()])
        training_scale = int(cfg["training"]["steps"]) / int(cfg["training"]["calibration_steps"])
        evaluation_scale = max(sum(cfg["evaluation"]["transition_cases"].values()) / sum(cfg["evaluation"]["calibration_transition_cases"].values()), sum(cfg["evaluation"]["rollout_cases"].values()) / sum(cfg["evaluation"]["calibration_rollout_cases"].values()))
        headroom = float(cfg["resource_budget"]["formal_headroom"]); expected = {}
        for key in ("wall_seconds", "process_tree_cpu_seconds", "cgroup_cpu_seconds"):
            expected[key] = headroom * (float(training[key]) * training_scale + float(evaluation[key]) * evaluation_scale + float(persistence[key]))
        expected["peak_process_tree_rss_bytes"] = int(root["resource"]["peak_process_tree_rss_bytes"] * float(cfg["resource_budget"]["rss_headroom"]))
        soft = {key: value * float(cfg["resource_budget"]["soft_multiplier"]) for key, value in expected.items()}; hard = {key: value * float(cfg["resource_budget"]["hard_multiplier"]) for key, value in expected.items()}
        timing = root["candidate_timing"]; nominal = float(timing["resource"]["wall_seconds"]) / int(timing["cases"]) * int(cfg["evaluation"]["candidate_projection_cases"])
        walls = [float(arm["diagnostic"]["resource"]["wall_seconds"]) for arm in root["arms"].values()]; variability = max(0.10, float(np.std(walls)) / max(float(np.mean(walls)), 1e-12)); guarded = nominal * max(headroom, 1.0 + variability)
        identities = all(arm["inference_identity"]["deterministic_repeat_logits_equal"] for arm in root["arms"].values()); artifact_bytes = sum(int(arm["checkpoint"]["bytes"]) for arm in root["arms"].values()); available = psutil.virtual_memory().available
        primary = root["arms"]["abbie_c"]
        checks = {"complete_matrix": all(set(arm["evaluation"]) == {"1", "2", "4", "8"} for arm in root["arms"].values()), "source_identity": bool(primary["initialization_source_identity"]["bit_logits_equal"]), "deterministic_cpu": identities, "body_updated": primary["training"]["body_update_norm_sum"] >= float(cfg["gates"]["minimum_body_update_norm"]), "recurrence_effect": primary["diagnostic"]["recurrences"]["4"]["delta_vs_r2"] >= float(cfg["gates"]["minimum_intervention_logit_delta"]), "state_intervention": min(primary["diagnostic"]["interventions"][k] for k in ("reset_state", "shuffle_state")) >= float(cfg["gates"]["minimum_intervention_logit_delta"]), "weight_intervention": primary["diagnostic"]["interventions"]["randomized_body"] >= float(cfg["gates"]["minimum_intervention_logit_delta"]), "candidate_runtime": guarded <= float(cfg["gates"]["maximum_guarded_candidate_seconds"]), "formal_runtime": expected["wall_seconds"] <= float(cfg["resource_budget"]["maximum_formal_wall_seconds"]), "artifact": artifact_bytes < int(cfg["gates"]["maximum_artifact_bytes"]), "memory": int(hard["peak_process_tree_rss_bytes"]) < int(available * 0.80), "no_fidelity_reduction": cfg["resource_budget"]["fidelity_reduced"] is False}
        decision = "admit-root-20261275" if all(checks.values()) else "park-calibration-resource-or-mechanism"
        summary_path = self.output / "calibration" / f"root-{self.spec().root}-summary.json"
        admission = {"schema_version": "exp681-dynamic-admission-v1", "root": self.spec().root, "decision": decision, "source": root["source"], "calibration_summary": str(summary_path.relative_to(ROOT)), "calibration_sha256": sha256(summary_path), "measured": root["resource"], "training_component": training, "evaluation_component": evaluation, "persistence_component": persistence, "training_scale": training_scale, "evaluation_scale": evaluation_scale, "variability": variability, "expected": expected, "soft": soft, "hard": hard, "candidate_nominal_seconds": nominal, "candidate_guarded_seconds": guarded, "candidate_ceiling_seconds": cfg["gates"]["maximum_guarded_candidate_seconds"], "artifact_bytes": artifact_bytes, "available_memory_bytes": available, "checks": checks, "fidelity_reduced": False}
        write_json(ROOT / cfg["resource_budget"]["admission_path"], admission); write_json(self.output / "calibration.json", admission); return admission

    def load_admission(self) -> dict[str, Any]:
        path = ROOT / self.config["resource_budget"]["admission_path"]; admission = json.loads(path.read_text()); calibration = ROOT / admission["calibration_summary"]
        if admission["decision"] != "admit-root-20261275" or admission["source"] != self.source_identity() or sha256(calibration) != admission["calibration_sha256"] or admission["fidelity_reduced"]: raise RuntimeError("formal requires exact admitted root-20261274")
        return admission

    def formal_gates(self, root: dict[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
        arms, gates = root["arms"], self.config["gates"]; primary = arms["abbie_c"]; r2 = primary["evaluation"]["2"]["128"]
        checks = {"w128_exact": r2["rollout"]["exact_accuracy"] >= float(gates["minimum_w128_exact"]), "w128_transition": r2["transition"]["exact_accuracy"] >= float(gates["minimum_w128_transition"]), "w128_validity": r2["rollout"]["invalids"] <= int(gates["maximum_invalids"]), "body_updated": primary["training"]["body_update_norm_sum"] >= float(gates["minimum_body_update_norm"]), "recurrence_intervention": primary["diagnostic"]["recurrences"]["4"]["delta_vs_r2"] >= float(gates["minimum_intervention_logit_delta"]), "state_intervention": min(primary["diagnostic"]["interventions"][k] for k in ("reset_state", "shuffle_state")) >= float(gates["minimum_intervention_logit_delta"]), "weight_intervention": primary["diagnostic"]["interventions"]["randomized_body"] >= float(gates["minimum_intervention_logit_delta"])}
        if all("2" in arms[name].get("evaluation", {}) for name in ("frozen_source", "untied_body")):
            score = r2["rollout"]["exact_accuracy"]; checks["source_retention"] = score >= arms["frozen_source"]["evaluation"]["2"]["128"]["rollout"]["exact_accuracy"] - float(gates["maximum_w128_loss_vs_source"]); checks["untied_retention"] = score >= arms["untied_body"]["evaluation"]["2"]["128"]["rollout"]["exact_accuracy"] - float(gates["maximum_w128_loss_vs_fixed_two"])
        if all("256" in arm.get("evaluation", {}).get("2", {}) for arm in arms.values()):
            scores = {r: primary["evaluation"][str(r)]["256"]["rollout"]["exact_accuracy"] for r in (2, 4, 8)}; best = max(scores.values()); checks["w256_absolute"] = best >= float(gates["minimum_w256_exact"]); checks["w256_controls"] = best - max(arms[name]["evaluation"]["2"]["256"]["rollout"]["exact_accuracy"] for name in ("abbie_d", "untied_body", "randomized_body", "frozen_source")) >= float(gates["minimum_w256_gain_vs_controls"]); checks["w256_deeper_trend"] = max(scores[4], scores[8]) - scores[2] >= float(gates["minimum_deeper_recurrence_gain"])
        return checks, {"w128_exact_by_recurrence": {r: primary.get("evaluation", {}).get(str(r), {}).get("128", {}).get("rollout", {}).get("exact_accuracy") for r in (1, 2, 4, 8)}, "w256_exact_by_recurrence": {r: primary.get("evaluation", {}).get(str(r), {}).get("256", {}).get("rollout", {}).get("exact_accuracy") for r in (2, 4, 8)}}

    def persist_terminal(self, root: dict[str, Any], admission: dict[str, Any]) -> dict[str, Any]:
        checks, comparisons = self.formal_gates(root); decision = f"reject-exp681-{root['early_decision']}" if root.get("early_decision") else "authorize-confirmations" if all(checks.values()) and self.mode == "formal" else "confirmation-pass" if all(checks.values()) else "reject-exp681-scientific-gate"
        summary = {"schema_version": "exp681-terminal-v1", "run_id": self.config["run_id"], "status": "completed", "decision": decision, "root": root["root"], "source": root["source"], "arms": root["arms"], "candidate_timing": root["candidate_timing"], "resource": root["resource"], "gates": checks, "comparisons": comparisons, "dynamic_admission": admission, "roots": {"20261274": "consumed-calibration", "20261275": "consumed-formal", "20261276": "current" if root["root"] == 20261276 else "unopened", "20261277": "current" if root["root"] == 20261277 else "unopened"}, "evaluation": {"not_performed": False, "scope": "fresh-generated-roots-only", "checked_independent_official_private_candidate_submission_promotion": "prohibited"}, "controls": {name: "completed" for name in ("abbie_d", "untied_body", "randomized_body", "frozen_source", "recurrence_one", "fixed_two", "state_interventions", "weight_intervention")}, "limitations": ["Head/body/tail is a bounded-faithful mapping onto Candidate005, not an AbbIE transformer reproduction.", "Fresh generated roots only; no checked, independent, official/private, candidate, submission, promotion or completion evaluation.", "Input chunks are never treated as recurrence."], "completion_marker": "terminal-persisted"}
        write_json(self.output / "terminal-summary.json", summary); write_json(ROOT / "experiments/runs/exp-681-abbie-recurrence/run.json", summary); write_json(ROOT / "evals/results/exp-681-abbie-recurrence/metrics.json", summary); return summary

    def formal(self): self.assert_frozen_clean(); admission = self.load_admission(); return self.persist_terminal(self.execute_root(), admission)
    def confirmation(self): self.assert_frozen_clean(); admission = self.load_admission(); return self.persist_terminal(self.execute_root(), admission)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=ROOT / "configs/exp_681_abbie_recurrence.yaml"); parser.add_argument("--output", type=Path, default=ROOT / ".scratch/exp-681-abbie-recurrence"); parser.add_argument("--mode", choices=("freeze", "preflight", "calibration", "formal", "confirmation"), required=True); parser.add_argument("--root", type=int); args = parser.parse_args()
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
