"""Calibrate and execute EXP677's known-step endpoint-length schedule."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
from safetensors.torch import save_file
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.exp_676_lla_moving_window.model import reinjection_receipt  # noqa: E402
from experiments.exp_676_lla_moving_window.run import (  # noqa: E402
    Experiment as BaseExperiment,
    RootSpec,
    Timer,
    aggregate,
    configure_mlflow,
    git_output,
    sha256,
    tracking_uri,
    write_json,
)
from experiments.exp_677_length_schedule.model import schedule_coverage, schedule_plan, trace_forward_budget  # noqa: E402


class Experiment(BaseExperiment):
    trained_arms = ("scheduled", "fixed_incumbent_endpoint", "equal_work_shuffled")

    def source_identity(self) -> dict[str, str]:
        paths = {
            "config": self.config_path,
            "model": Path(__file__).with_name("model.py"),
            "runner": Path(__file__),
            "test": ROOT / "tests/test_exp_677_length_schedule.py",
            "mechanism_map": ROOT / self.config["mechanism_map"],
            "paper": ROOT / self.config["source_contract"]["paper_path"],
            "manifest": ROOT / "research/manifest.json",
            "verification": ROOT / "research/verification.json",
            "checkpoint": self.checkpoint,
        }
        return {"repository_commit": git_output("rev-parse", "HEAD"), **{f"{key}_sha256": sha256(path) for key, path in paths.items()}}

    def assert_frozen_clean(self) -> None:
        paths = [
            str(self.config_path.relative_to(ROOT)),
            "experiments/exp_677_length_schedule",
            "tests/test_exp_677_length_schedule.py",
            self.config["preregistration"],
            self.config["mechanism_map"],
        ]
        if git_output("status", "--porcelain", "--", *paths):
            raise RuntimeError("execution requires committed frozen EXP677 bytes")

    def spec(self) -> RootSpec:
        data = self.config["data"]
        if self.mode == "calibration": values = data["calibration"]
        elif self.mode == "formal": values = data["formal"]
        else:
            matches = [item for item in data["confirmations"] if int(item["root"]) == int(self.root_override or -1)]
            if len(matches) != 1: raise ValueError("confirmation requires frozen root 20261260 or 20261261")
            values = matches[0]
        return RootSpec(**values)

    def recovery_loss_plan(self, model: nn.Module, sampler: Any, rng: random.Random, plan: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        cfg = self.config["training"]; width = int(plan["width"]); batch = int(cfg["batch_size"]); chunk = int(cfg["recovery_chunk_steps"])
        multiplier, right, modulus = sampler.pairs(batch); starts = [rng.randrange(0, width - chunk + 1) for _ in range(batch)]
        teacher = self.exp329.teacher_states_before(multiplier, right, modulus, starts, width); state_plane = self.exp300.bits(teacher, width); losses = []
        for offset in range(chunk):
            positions = [start + offset for start in starts]; bit = [(left >> (width - 1 - position)) & 1 for left, position in zip(multiplier, positions)]
            raw = [2 * state + flag * value for state, flag, value in zip(teacher, bit, right)]; target = [value % prime for value, prime in zip(raw, modulus)]; quotient = torch.tensor([value // prime for value, prime in zip(raw, modulus)], dtype=torch.long)
            features = self.exp300.features_from_soft_state(state_plane, right, modulus, bit, width) if hasattr(self.exp300, "features_from_soft_state") else self._soft_features(state_plane, right, modulus, bit, width)
            outputs = trace_forward_budget(model, features, int(plan["physical_visits"])); logits, qlogits = outputs[-1]
            loss, _ = self.exp300.transition_loss(logits, qlogits, target, quotient, width, float(cfg["worst_bit_weight"]), float(cfg["quotient_weight"])); losses.append(loss)
            probabilities = torch.sigmoid(logits); state_plane = logits.ge(0).float() + probabilities - probabilities.detach(); teacher = target
        return torch.stack(losses).mean(), {"loss_calls": chunk, "plan": plan}

    def train_arm(self, model: nn.Module, spec: RootSpec, arm_name: str, steps: int, timer: Timer) -> dict[str, Any]:
        cfg = self.config["training"]; kind = self.config["arms"][arm_name]["kind"]
        random.seed(spec.train_seed); np.random.seed(spec.train_seed); torch.manual_seed(spec.train_seed)
        parameters = list(model.parameters()); optimizer = torch.optim.AdamW(parameters, lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, steps), eta_min=float(cfg["minimum_learning_rate"]))
        rng = random.Random(spec.train_seed + 7); samplers = {width: self.exp300.BigIntSampler(self.split(width, spec.root)["train"], spec.train_seed + width) for width in (64, 128, 256)}
        before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}; history = []; plan_counts: dict[str, int] = {}; gradient_nonzero_steps = 0; model.train()
        for step in range(1, steps + 1):
            plan = schedule_plan(kind, step=step, total_steps=steps, schedule=self.config["schedule"], seed=spec.train_seed); width = int(plan["width"])
            loss, receipt = self.recovery_loss_plan(model, samplers[width], rng, plan); key = f"{width}:{plan['physical_visits']}:{plan['supervised_endpoint_length']}"; plan_counts[key] = plan_counts.get(key, 0) + 1
            optimizer.zero_grad(set_to_none=True); loss.backward(); gradient = float(nn.utils.clip_grad_norm_(parameters, float(cfg["gradient_clip"]))); gradient_nonzero_steps += int(gradient >= float(self.config["gates"]["minimum_schedule_gradient_norm"])); optimizer.step(); scheduler.step()
            if step == 1 or step % 50 == 0 or step == steps:
                timer.sample(); record = {"step": step, "loss": float(loss.detach()), "gradient_norm": gradient, "plan": plan, "loss_calls": receipt["loss_calls"]}; history.append(record); print(json.dumps({"root": spec.root, "arm": arm_name, **record}), flush=True)
        model.eval(); changes = {name: float((model.state_dict()[name] - value).norm()) for name, value in before.items() if torch.is_floating_point(value)}
        return {
            "steps": steps,
            "examples_per_update": int(cfg["batch_size"]),
            "transition_examples_per_update": int(cfg["batch_size"]) * int(cfg["recovery_chunk_steps"]),
            "loss_calls_per_update": int(cfg["recovery_chunk_steps"]),
            "history": history,
            "plan_counts": plan_counts,
            "gradient_nonzero_steps": gradient_nonzero_steps,
            "parameter_update_norm_sum": sum(changes.values()),
            "parameter_update_norms": changes,
            "matching": {"updates": steps, "examples": steps * int(cfg["batch_size"]), "loss_calls": steps * int(cfg["recovery_chunk_steps"])},
        }

    def train_and_cache(self, spec: RootSpec, steps: int, session: Timer) -> tuple[dict[str, nn.Module], dict[str, Any]]:
        arms: dict[str, nn.Module] = {}; records: dict[str, Any] = {}
        for arm_name in self.config["arms"]:
            path = self.checkpoint_path(spec, arm_name); training_timer = Timer()
            model = self.base(randomized=True, seed=spec.train_seed + 900) if arm_name == "randomized_control" else self.base()
            if arm_name in self.trained_arms: training = self.train_arm(model, spec, arm_name, steps, session)
            else: training = {"steps": 0, "reason": "frozen-source" if arm_name == "frozen_source" else "independently-randomized-weight-control", "loss_calls": 0}
            training["runtime"] = training_timer.result(); path.parent.mkdir(parents=True, exist_ok=True); save_file({key: value.detach().contiguous() for key, value in model.state_dict().items()}, path)
            identity = self.inference_identity(model, path, spec.eval_seed + len(arms)); checkpoint = {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}
            required = ("state_dict_identity", "ordinary_vs_trace_logits_equal", "ordinary_vs_trace_quotient_equal", "pre_save_vs_reload_logits_equal", "pre_save_vs_reload_quotient_equal")
            if not all(identity[key] for key in required): raise RuntimeError(f"formal inference identity failed for {arm_name}")
            arms[arm_name] = model.eval(); records[arm_name] = {"training": training, "checkpoint": checkpoint, "inference_identity": identity, "parameters": identity["parameter_count"]}
            write_json(self.output / self.mode / f"root-{spec.root}-progress.json", {"schema_version": "exp677-root-progress-v1", "source": self.source_identity(), "root": spec.root, "steps": steps, "completed_arms": records})
        return arms, records

    def _timing(self, model: nn.Module, spec: RootSpec) -> dict[str, Any]:
        cfg = self.config["evaluation"]; timer = Timer(); width = int(cfg["candidate_timing_width"]); cases = int(cfg["candidate_timing_cases"])
        rollout = self.evaluate_rollouts(model, self.exp300.BigIntSampler(self.split(width, spec.root)["test"], spec.eval_seed + 9000), cases, width)
        return {"width": width, "cases": cases, "projection_cases": int(cfg["candidate_projection_cases"]), "rollout": rollout, "resource": timer.result(), "formal_graph": "unchanged-Candidate005-SharedDilatedConvGRU"}

    def _w128_checks(self, records: dict[str, Any]) -> dict[str, bool]:
        gates = self.config["gates"]; primary = records["scheduled"]["evaluation"]["128"]; fixed = records["fixed_incumbent_endpoint"]["evaluation"]["128"]; source = records["frozen_source"]["evaluation"]["128"]; randomized = records["randomized_control"]["evaluation"]["128"]
        return {
            "w128": primary["rollout"]["exact_accuracy"] >= float(gates["minimum_w128_exact"]),
            "w128_transition": primary["transition"]["exact_accuracy"] >= float(gates["minimum_w128_transition"]),
            "validity": primary["rollout"]["invalids"] <= int(gates["maximum_invalids"]),
            "fixed_retention": primary["rollout"]["exact_accuracy"] >= fixed["rollout"]["exact_accuracy"] - float(gates["maximum_w128_loss_vs_fixed"]),
            "source_retention": primary["rollout"]["exact_accuracy"] >= source["rollout"]["exact_accuracy"] - float(gates["maximum_w128_loss_vs_source"]),
            "trained_dependence": primary["rollout"]["exact_accuracy"] - randomized["rollout"]["exact_accuracy"] >= float(gates["minimum_trained_minus_randomized_w128"]),
            "schedule_gradient": records["scheduled"]["training"]["gradient_nonzero_steps"] == records["scheduled"]["training"]["steps"],
            "formal_inference_identity": all(all(value["inference_identity"].values()) if False else all(value["inference_identity"][key] for key in ("state_dict_identity", "ordinary_vs_trace_logits_equal", "ordinary_vs_trace_quotient_equal", "pre_save_vs_reload_logits_equal", "pre_save_vs_reload_quotient_equal")) for value in records.values()),
        }

    def execute_root(self) -> dict[str, Any]:
        spec = self.spec(); cfg = self.config; calibration = self.mode == "calibration"; steps = int(cfg["training"]["calibration_steps" if calibration else "steps"])
        transitions = {int(k): int(v) for k, v in cfg["evaluation"]["calibration_transition_cases" if calibration else "transition_cases"].items()}; rollouts = {int(k): int(v) for k, v in cfg["evaluation"]["calibration_rollout_cases" if calibration else "rollout_cases"].items()}; late_cases = int(cfg["evaluation"]["calibration_late_depth_cases" if calibration else "late_depth_cases"])
        session = Timer(); arms, records = self.train_and_cache(spec, steps, session); reinjection = {str(width): reinjection_receipt(arms["scheduled"], int(width)) for width in cfg["evaluation"]["widths"]}
        if not all(value["gate_reinjects_every_visit"] and value["candidate_reinjects_every_visit"] for value in reinjection.values()): raise RuntimeError("Candidate005 reinjection equivalence failed")
        early = None; w128_checks = None
        if calibration:
            for arm_name, model in arms.items(): records[arm_name]["evaluation"], records[arm_name]["evaluation_runtime"] = self.evaluate_arm(model, spec, [64, 128, 256], transitions, rollouts, late_cases)
        else:
            evaluation, runtime = self.evaluate_arm(arms["scheduled"], spec, [128], transitions, rollouts, late_cases); records["scheduled"]["evaluation"] = evaluation; records["scheduled"]["evaluation_runtime"] = runtime
            primary = evaluation["128"]
            if primary["rollout"]["exact_accuracy"] < float(cfg["gates"]["minimum_w128_exact"]) or primary["rollout"]["invalids"] > int(cfg["gates"]["maximum_invalids"]) or primary["transition"]["exact_accuracy"] < float(cfg["gates"]["minimum_w128_transition"]): early = "reject-W128-primary"
            if early is None:
                for arm_name in ("fixed_incumbent_endpoint", "equal_work_shuffled", "frozen_source", "randomized_control"):
                    records[arm_name]["evaluation"], records[arm_name]["evaluation_runtime"] = self.evaluate_arm(arms[arm_name], spec, [128], transitions, rollouts, late_cases)
                w128_checks = self._w128_checks(records)
                if not all(w128_checks.values()): early = "reject-W128-retention-or-causality"
            if early is None:
                for arm_name, model in arms.items():
                    evaluation, runtime = self.evaluate_arm(model, spec, [64, 256], transitions, rollouts, late_cases); records[arm_name]["evaluation"].update(evaluation); records[arm_name]["evaluation_runtime"] = aggregate([records[arm_name]["evaluation_runtime"], runtime])
        result = {"schema_version": "exp677-root-v1", "mode": self.mode, "root": spec.root, "source": self.source_identity(), "schedule_coverage": schedule_coverage(cfg["schedule"], spec.train_seed), "reinjection": reinjection, "arms": records, "candidate_timing": self._timing(arms["scheduled"], spec), "resource": session.result(), "w128_checks": w128_checks, "early_decision": early}
        write_json(self.output / self.mode / f"root-{spec.root}-summary.json", result); return result

    def calibration(self) -> dict[str, Any]:
        self.assert_frozen_clean(); root = self.execute_root(); cfg = self.config; measured = root["resource"]
        training = aggregate([root["arms"][name]["training"]["runtime"] for name in self.trained_arms]); evaluation = aggregate([root["arms"][name]["evaluation_runtime"] for name in cfg["arms"]]); training_scale = int(cfg["training"]["steps"]) / int(cfg["training"]["calibration_steps"]); evaluation_scale = max(sum(cfg["evaluation"]["transition_cases"].values()) / sum(cfg["evaluation"]["calibration_transition_cases"].values()), sum(cfg["evaluation"]["rollout_cases"].values()) / sum(cfg["evaluation"]["calibration_rollout_cases"].values()))
        headroom = float(cfg["resource_budget"]["formal_headroom"]); expected = {key: headroom * (float(training[key]) * training_scale + float(evaluation[key]) * evaluation_scale) for key in ("wall_seconds", "process_tree_cpu_seconds", "cgroup_cpu_seconds")}; expected["peak_process_tree_rss_bytes"] = int(measured["peak_process_tree_rss_bytes"] * float(cfg["resource_budget"]["rss_headroom"])); soft = {key: value * float(cfg["resource_budget"]["soft_multiplier"]) for key, value in expected.items()}; hard = {key: value * float(cfg["resource_budget"]["hard_multiplier"]) for key, value in expected.items()}
        timing = root["candidate_timing"]; nominal = float(timing["resource"]["wall_seconds"]) / int(timing["cases"]) * int(timing["projection_cases"]); arm_walls = [float(root["arms"][name]["evaluation_runtime"]["wall_seconds"]) for name in cfg["arms"]]; variability = max(0.10, max(arm_walls) / float(np.mean(arm_walls)) - 1.0); guarded = nominal * max(headroom, 1.0 + variability); artifact = sum(int(value["checkpoint"]["bytes"]) for value in root["arms"].values())
        required = ("state_dict_identity", "ordinary_vs_trace_logits_equal", "ordinary_vs_trace_quotient_equal", "pre_save_vs_reload_logits_equal", "pre_save_vs_reload_quotient_equal"); identities = all(all(value["inference_identity"][key] for key in required) for value in root["arms"].values())
        checks = {"complete_route": set(root["arms"]) == set(cfg["arms"]), "source_schedule_coverage": root["schedule_coverage"]["equal_work_multiset_matches"], "formal_inference_identity": identities, "candidate_runtime": guarded <= float(cfg["gates"]["maximum_guarded_candidate_seconds"]), "artifact": artifact < int(cfg["gates"]["maximum_artifact_bytes"]), "memory": expected["peak_process_tree_rss_bytes"] < 16 * 1024**3, "expected_value": expected["wall_seconds"] <= 86400, "no_fidelity_reduction": cfg["resource_budget"]["fidelity_reduced"] is False}
        decision = "admit-root-20261259" if all(checks.values()) else "reject-formal-at-calibration"; summary_path = self.output / self.mode / f"root-{self.spec().root}-summary.json"
        admission = {"schema_version": "exp677-dynamic-admission-v1", "decision": decision, "source": root["source"], "calibration_summary": str(summary_path.relative_to(ROOT)), "calibration_sha256": sha256(summary_path), "measured": measured, "training_component": training, "evaluation_component": evaluation, "training_scale": training_scale, "evaluation_scale": evaluation_scale, "variability": variability, "formal_eta_seconds": expected["wall_seconds"], "expected": expected, "soft": soft, "hard": hard, "candidate_nominal_seconds": nominal, "candidate_guarded_seconds": guarded, "artifact_bytes": artifact, "checks": checks, "fidelity_reduced": False}
        write_json(ROOT / cfg["resource_budget"]["admission_path"], admission); write_json(self.output / "calibration.json", admission); return admission

    def load_admission(self) -> dict[str, Any]:
        path = ROOT / self.config["resource_budget"]["admission_path"]; admission = json.loads(path.read_text(encoding="utf-8")); calibration = ROOT / admission["calibration_summary"]
        if admission["decision"] != "admit-root-20261259" or admission["source"] != self.source_identity() or sha256(calibration) != admission["calibration_sha256"] or admission["fidelity_reduced"]: raise RuntimeError("formal requires matching admitted calibration")
        return admission

    def formal_gates(self, root: dict[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
        checks = dict(root["w128_checks"] or {}); arms = root["arms"]; gates = self.config["gates"]
        if root.get("early_decision"): return checks, {"early_decision": root["early_decision"]}
        scheduled = arms["scheduled"]["evaluation"]["256"]["rollout"]["exact_accuracy"]; fixed = arms["fixed_incumbent_endpoint"]["evaluation"]["256"]["rollout"]["exact_accuracy"]; shuffled = arms["equal_work_shuffled"]["evaluation"]["256"]["rollout"]["exact_accuracy"]
        checks.update({"all_opened_valid": all(arms[name]["evaluation"][str(width)]["rollout"]["invalids"] == 0 for name in arms for width in (64, 128, 256)), "w256_absolute": scheduled >= float(gates["minimum_w256_exact"]), "w256_fixed_gain": scheduled - fixed >= float(gates["minimum_w256_gain_vs_controls"]), "w256_equal_work_gain": scheduled - shuffled >= float(gates["minimum_w256_gain_vs_controls"])})
        comparisons = {"w128_exact": {name: arms[name]["evaluation"]["128"]["rollout"]["exact_accuracy"] for name in arms}, "w256_exact": {name: arms[name]["evaluation"]["256"]["rollout"]["exact_accuracy"] for name in arms}, "training_matching": {name: arms[name]["training"].get("matching") for name in self.trained_arms}}
        return checks, comparisons

    def persist_terminal(self, root: dict[str, Any], decision: str, checks: dict[str, bool], comparisons: dict[str, Any] | None, admission: dict[str, Any]) -> dict[str, Any]:
        summary = {"schema_version": "exp677-terminal-v1", "run_id": self.config["run_id"], "status": "completed", "decision": decision, "root": root["root"], "source": root["source"], "roots": {"20261258": "completed-calibration", "20261259": "opened" if root["root"] == 20261259 else "unopened", "20261260": "opened" if root["root"] == 20261260 else "unopened", "20261261": "opened" if root["root"] == 20261261 else "unopened"}, "schedule_coverage": root["schedule_coverage"], "reinjection": root["reinjection"], "arms": root["arms"], "candidate_timing": root["candidate_timing"], "resource": root["resource"], "gates": checks, "comparisons": comparisons, "dynamic_admission": admission, "limitations": ["Fresh legal generated identities only; no checked, independent, official/private, candidate, submission, promotion or completion evaluation.", "Known steps, widths and endpoint targets are training/evaluation-only; formal inference is the original SharedDilatedConvGRU.", "The fixed W128 control has irreducibly different tensor FLOPs from the scheduled width mixture; updates, examples, optimizer schedule and loss calls are matched and resource differences are disclosed."]}
        write_json(self.output / "terminal-summary.json", summary); write_json(ROOT / "experiments/runs/exp-677-length-schedule/run.json", summary); write_json(ROOT / "evals/results/exp-677-length-schedule/metrics.json", summary); return summary

    def formal(self) -> dict[str, Any]:
        self.assert_frozen_clean(); admission = self.load_admission(); root = self.execute_root(); checks, comparisons = self.formal_gates(root)
        decision = "reject-exp677-at-W128-primary" if root.get("early_decision") == "reject-W128-primary" else "reject-exp677-at-W128-retention-or-causality" if root.get("early_decision") else "authorize-confirmations" if all(checks.values()) else "reject-exp677-length-schedule"
        return self.persist_terminal(root, decision, checks, comparisons, admission)

    def confirmation(self) -> dict[str, Any]:
        self.assert_frozen_clean(); admission = self.load_admission(); root = self.execute_root(); checks, comparisons = self.formal_gates(root); decision = "confirmation-pass" if checks and all(checks.values()) else "confirmation-reject"
        return self.persist_terminal(root, decision, checks, comparisons, admission)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=ROOT / "configs/exp_677_length_schedule.yaml"); parser.add_argument("--output", type=Path, default=ROOT / ".scratch/exp-677-length-schedule"); parser.add_argument("--mode", choices=("calibration", "formal", "confirmation"), required=True); parser.add_argument("--root", type=int); args = parser.parse_args()
    experiment = Experiment(args.config, args.output, args.mode, args.root); result = experiment.calibration() if args.mode == "calibration" else experiment.formal() if args.mode == "formal" else experiment.confirmation()
    experiment_id = configure_mlflow(experiment.config["tracking"]["experiment_name"])
    with mlflow.start_run(run_name=f"{experiment.config['run_id']}-{args.mode}-{experiment.spec().root}") as run:
        mlflow.set_tags({"canonical_run_id": experiment.config["run_id"], "mode": args.mode, "root": str(experiment.spec().root), "decision": str(result.get("decision")), "repository_commit": git_output("rev-parse", "HEAD")}); mlflow.log_dict(result, f"{args.mode}-result.json"); mlflow_run_id = run.info.run_id
    write_json(experiment.output / f"{args.mode}-mlflow.json", {"experiment_id": experiment_id, "experiment_name": experiment.config["tracking"]["experiment_name"], "run_id": mlflow_run_id, "tracking_uri": tracking_uri()})
    print(json.dumps({"mode": args.mode, "root": experiment.spec().root, "decision": result.get("decision"), "resource": result.get("resource", result.get("measured")), "gates": result.get("gates", result.get("checks")), "mlflow_run_id": mlflow_run_id}, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
