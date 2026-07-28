"""Execute EXP688's compact selected-hidden trajectory distillation experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.exp_687_compact_distilled_w256_specialist import run as exp687  # noqa: E402
from experiments.exp_664_sprm_boundary_perturbation.run import Timer, bits, git_output, make_split, sha256, write_json  # noqa: E402
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402

RUN_ID = "exp-688-compact-hidden-trajectory-distillation"
CONFIG_PATH = ROOT / "configs/exp_688_compact_hidden_trajectory_distillation.yaml"
CHALLENGE_COMMIT = exp687.CHALLENGE_COMMIT
TRAINED_ARMS = ("hidden_aligned", "logit_only")
ARM_ORDER = TRAINED_ARMS + ("frozen_pruned", "randomized_control", "reset_control")

# EXP687 is the authenticated implementation source for the answer-independent map,
# coherent pruning, compact model construction, evaluation, and student-only rollout.
canonical_hash = exp687.canonical_hash
state_hash = exp687.state_hash
configure_torch = exp687.configure_torch
channel_importance = exp687.channel_importance
slice_state = exp687.slice_state
prepare_pruned = exp687.prepare_pruned
new_student = exp687.new_student
new_teacher = exp687.new_teacher
soft_features = exp687.soft_features
hard_straight_through = exp687.hard_straight_through
teacher_prefix = exp687.teacher_prefix
rollout = exp687.rollout
evaluate = exp687.evaluate


def forward_with_final_hidden(model, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Source-exact forward pass with training-only capture of final recurrent hidden."""
    hidden = model.stem(inputs)
    for dilation in model.dilation_schedule(inputs.shape[-1]):
        hidden = model.recurrent_update(hidden, inputs, dilation)
    normalized_output = F.silu(model.output_norm(hidden))
    logits = model.local_head(normalized_output).squeeze(1)
    pooled = torch.cat((normalized_output.mean(dim=-1), normalized_output.amax(dim=-1)), dim=1)
    return logits, model.quotient_head(pooled), hidden


def normalized_hidden(hidden: torch.Tensor) -> torch.Tensor:
    return F.normalize(hidden, p=2.0, dim=1, eps=1e-8)


def train_update(
    config: dict[str, Any], exp300, student, teacher_model, optimizer: torch.optim.Optimizer,
    batch: tuple[list[int], list[int], list[int], list[int]], width: int, hidden_weight: float,
    selected: torch.Tensor,
) -> dict[str, float]:
    left, right, modulus, starts = batch
    horizon = int(config["training"]["horizon"])
    teacher_state = teacher_prefix(left, right, modulus, starts, width)
    state_plane = bits(teacher_state, width)
    losses, exacts, residues, quotients, hidden_terms, hidden_signals = [], [], [], [], [], []
    optimizer.zero_grad(set_to_none=True)
    student.width = teacher_model.width = width
    for offset in range(horizon):
        flags = [(value >> (width - 1 - start - offset)) & 1 for value, start in zip(left, starts, strict=True)]
        raw = [2 * state + flag * other for state, flag, other in zip(teacher_state, flags, right, strict=True)]
        target = [value % prime for value, prime in zip(raw, modulus, strict=True)]
        quotient = torch.tensor([value // prime for value, prime in zip(raw, modulus, strict=True)], dtype=torch.long)
        inputs = soft_features(state_plane, right, modulus, flags, width)
        logits, quotient_logits, student_hidden = forward_with_final_hidden(student, inputs)
        with torch.no_grad():
            teacher_logits, teacher_quotient, teacher_hidden = forward_with_final_hidden(teacher_model, inputs.detach())
            selected_teacher_hidden = teacher_hidden.index_select(1, selected)
        exact_loss, stats = exp300.transition_loss(
            logits, quotient_logits, target, quotient, width,
            float(config["training"]["worst_bit_weight"]), float(config["training"]["quotient_weight"]),
        )
        residue = F.mse_loss(logits, teacher_logits) / teacher_logits.square().mean().clamp_min(1e-8)
        quotient_kl = F.kl_div(F.log_softmax(quotient_logits, 1), F.softmax(teacher_quotient, 1), reduction="batchmean")
        hidden_alignment = F.mse_loss(normalized_hidden(student_hidden), normalized_hidden(selected_teacher_hidden))
        hidden_signal = torch.autograd.grad(hidden_alignment, student_hidden, retain_graph=True)[0].abs().sum()
        losses.append(
            exact_loss
            + float(config["training"]["residue_distillation_weight"]) * residue
            + float(config["training"]["quotient_distillation_weight"]) * quotient_kl
            + hidden_weight * hidden_alignment
        )
        exacts.append(float(stats["batch_exact"]))
        residues.append(residue)
        quotients.append(quotient_kl)
        hidden_terms.append(hidden_alignment)
        hidden_signals.append(hidden_signal)
        state_plane = hard_straight_through(logits)
        teacher_state = target
    objective = torch.stack(losses).mean()
    objective.backward()
    gradient = float(torch.nn.utils.clip_grad_norm_(student.parameters(), float(config["training"]["gradient_clip"])))
    optimizer.step()
    return {
        "loss": float(objective.detach()), "gradient_norm": gradient, "mean_exact": float(np.mean(exacts)),
        "residue_distillation": float(torch.stack(residues).mean()),
        "quotient_distillation": float(torch.stack(quotients).mean()),
        "hidden_distillation": float(torch.stack(hidden_terms).mean()),
        "hidden_gradient_signal": float(torch.stack(hidden_signals).mean()),
        "residue_distillation_weight": float(config["training"]["residue_distillation_weight"]),
        "quotient_distillation_weight": float(config["training"]["quotient_distillation_weight"]),
        "hidden_distillation_weight": hidden_weight,
        "student_calls": horizon, "teacher_calls": horizon, "hidden_capture_calls": horizon,
        "hidden_alignment_calls": horizon, "exact_loss_calls": horizon,
    }


def train_arm(config, exp300, pruned, checkpoint: Path, splits, root: int, arm: str, updates: int, output: Path):
    random.seed(root + 688); np.random.seed(root + 688); torch.manual_seed(root + 688)
    model = new_student(config, None if arm == "randomized_control" else pruned, root + 688900 if arm == "randomized_control" else None)
    initial = state_hash(model.state_dict()); timer = Timer(); history = []
    if arm in TRAINED_ARMS:
        teacher = new_teacher(config, checkpoint)
        selected = channel_importance(load_file(checkpoint))[0][: int(config["model"]["hidden"])]
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates, eta_min=float(config["training"]["minimum_learning_rate"]))
        samplers = {w: exp300.BigIntSampler(splits[w]["train"], root + int(config["data"]["training_seed_offset"]) + w) for w in set(config["data"]["curriculum"])}
        starts = {w: random.Random(root + int(config["data"]["training_seed_offset"]) + w + 1) for w in samplers}
        model.train()
        hidden_weight = float(config["routes"][arm]["hidden_distillation_weight"])
        for update in range(1, updates + 1):
            tick = Timer(); width = int(config["data"]["curriculum"][(update - 1) % len(config["data"]["curriculum"])])
            left, right, modulus = samplers[width].pairs(int(config["training"]["batch_size"]))
            begin = [starts[width].randrange(0, width - int(config["training"]["horizon"]) + 1) for _ in left]
            stats = train_update(config, exp300, model, teacher, optimizer, (left, right, modulus, begin), width, hidden_weight, selected)
            scheduler.step(); record = {"update": update, "width": width, "learning_rate": scheduler.get_last_lr()[0], "timing": tick.finish(), **stats}; history.append(record)
            if update == 1 or update == updates or update % int(config["training"]["log_every_updates"]) == 0:
                print(json.dumps({"arm": arm, **record}), flush=True)
    model.eval(); arm_dir = output / "arms" / arm; arm_dir.mkdir(parents=True, exist_ok=True); path = arm_dir / "model.safetensors"
    save_file({k: v.detach().contiguous() for k, v in model.state_dict().items()}, path); reloaded = new_student(config, load_file(path))
    record = {
        "arm": arm, "trained": arm in TRAINED_ARMS, "initial_state_sha256": initial, "parameters": model.parameter_count(),
        "training": {"updates": updates if arm in TRAINED_ARMS else 0, "curriculum": list(config["data"]["curriculum"]),
                     "calls_per_update": int(config["training"]["horizon"]), "teacher_calls": sum(x["teacher_calls"] for x in history),
                     "student_calls": sum(x["student_calls"] for x in history), "hidden_capture_calls": sum(x["hidden_capture_calls"] for x in history),
                     "hidden_alignment_calls": sum(x["hidden_alignment_calls"] for x in history), "exact_loss_calls": sum(x["exact_loss_calls"] for x in history),
                     "history": history, "resource": timer.finish()},
        "checkpoint": {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size},
        "reload_exact": all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), reloaded.state_dict().values(), strict=True)),
    }
    return model, record


def candidate_inputs(root: int) -> list[tuple[str, str, str, int]]:
    source = ROOT / "sources/github/modular-arithmetic-challenge/src"; sys.path.insert(0, str(source))
    try:
        from modchallenge.config import EvalConfig
        from modchallenge.testgen.generator import generate_private_test_set
        test_set = generate_private_test_set(hashlib.sha256(f"exp688-candidate-{root}".encode()).digest(), EvalConfig())
        return [(case.a, case.b, case.p, tier.tier_id) for tier in test_set.tiers for case in tier.cases]
    finally:
        sys.path.remove(str(source))


def candidate_runtime_once(config, checkpoint: Path, root: int, limit: int | None = None):
    original = exp687.candidate_inputs; exp687.candidate_inputs = candidate_inputs
    try:
        return exp687.candidate_runtime_once(config, checkpoint, root, limit)
    finally:
        exp687.candidate_inputs = original


class Experiment(exp687.Experiment):
    def authenticate(self) -> None:
        if self.config["run_id"] != RUN_ID or not self.checkpoint.is_file() or sha256(self.checkpoint) != self.config["source_checkpoint_sha256"]:
            raise RuntimeError("source checkpoint mismatch")
        for name, binding in self.config["source_contract"].items():
            if sha256(ROOT / binding["path"]) != binding["sha256"]:
                raise RuntimeError(f"source contract mismatch: {name}")
        if git_output("-C", "sources/github/modular-arithmetic-challenge", "rev-parse", "HEAD") != CHALLENGE_COMMIT:
            raise RuntimeError("challenge commit mismatch")

    def identity(self) -> dict[str, Any]:
        return {"repository_commit": git_output("rev-parse", "HEAD"), "challenge_commit": CHALLENGE_COMMIT,
                "config_sha256": sha256(self.config_path), "runner_sha256": sha256(Path(__file__)),
                "test_sha256": sha256(ROOT / "tests/test_exp_688_compact_hidden_trajectory_distillation.py"),
                "source_checkpoint_sha256": sha256(self.checkpoint),
                "channel_map_sha256": canonical_hash(self.manifest["selected_source_channels_ranked"]),
                "pruned_state_sha256": self.manifest["pruned_state_sha256"]}

    def train_routes(self, path: Path, updates: int):
        models, records = {}, {}
        for arm in ARM_ORDER:
            models[arm], records[arm] = train_arm(
                self.config, self.exp300, self.pruned, self.checkpoint, self.splits,
                int(self.root), arm, updates, path,
            )
            write_json(path / "progress.json", {"completed_arms": list(records), "records": records})
        primary, control = (records[arm] for arm in TRAINED_ARMS)
        matched_keys = ("updates", "curriculum", "calls_per_update", "teacher_calls", "student_calls", "hidden_capture_calls", "hidden_alignment_calls", "exact_loss_calls")
        if primary["initial_state_sha256"] != control["initial_state_sha256"] or any(primary["training"][key] != control["training"][key] for key in matched_keys):
            raise RuntimeError("matched trained-arm work failed")
        return models, records

    def admission(self, records, evaluations, runtimes, resource, updates, transitions, rollouts):
        scale = int(self.config["training"]["updates"]) / updates
        training = sum(float(records[arm]["training"]["resource"]["wall_seconds"]) * scale for arm in TRAINED_ARMS)
        evaluation = sum(
            float(evaluations[arm][str(width)][kind]["resource"]["wall_seconds"])
            * (int(self.config["evaluation"]["transition_cases_per_width"] if kind == "transition" else self.config["evaluation"]["rollout_cases_per_width"]) / (transitions if kind == "transition" else rollouts))
            for arm in ARM_ORDER for width in (128, 256) for kind in ("transition", "rollout")
        )
        candidate = max(float(item["resource"]["wall_seconds"]) for item in runtimes["cold"])
        timings = [float(item["timing"]["wall_seconds"]) for arm in TRAINED_ARMS for item in records[arm]["training"]["history"]]
        variability = max(timings) / max(float(np.mean(timings)), 1e-12)
        headroom = min(float(self.config["resource_policy"]["variability_headroom_ceiling"]), max(float(self.config["resource_policy"]["candidate_headroom_floor"]), variability))
        expected = training + evaluation + candidate
        admission = {"projected_training_wall_seconds": training, "projected_evaluation_wall_seconds": evaluation,
                     "expected_wall_seconds": expected, "eta_wall_seconds": expected * headroom,
                     "soft_guard_wall_seconds": math.ceil(expected * headroom * float(self.config["resource_policy"]["soft_guard_multiplier"])),
                     "hard_guard_wall_seconds": math.ceil(expected * headroom * float(self.config["resource_policy"]["hard_guard_multiplier"])),
                     "headroom": headroom, "candidate_max_complete_wall_seconds": candidate,
                     "candidate_guarded_wall_seconds": candidate * headroom,
                     "expected_peak_rss_bytes": int(resource["peak_rss_bytes"] * headroom), "fidelity_reduced": False}
        admission["checks"] = {"complete_routes": set(records) == set(ARM_ORDER) and set(evaluations) == set(ARM_ORDER),
                               "matched_training": all(records[TRAINED_ARMS[0]]["training"][key] == records[TRAINED_ARMS[1]]["training"][key] for key in ("teacher_calls", "student_calls", "hidden_capture_calls", "hidden_alignment_calls", "exact_loss_calls")),
                               "reload_exact": all(item["reload_exact"] for item in records.values()),
                               "candidate_three_complete_cold": len(runtimes["cold"]) == 3 and all(item["complete_1100_case_route"] and item["output_count"] == 1100 for item in runtimes["cold"]),
                               "candidate_runtime": admission["candidate_guarded_wall_seconds"] <= float(self.config["gates"]["maximum_candidate_guarded_wall_seconds"]),
                               "expected_value": expected <= float(self.config["resource_policy"]["maximum_expected_formal_wall_seconds"]),
                               "memory": admission["expected_peak_rss_bytes"] <= int(self.config["resource_policy"]["maximum_peak_rss_bytes"]),
                               "no_fidelity_reduction": True}
        admission["admitted"] = all(admission["checks"].values())
        return admission

    def smoke(self):
        old_root, old_splits = self.root, getattr(self, "splits", None); self.root = 688000
        self.splits = {w: make_split(self.config, self.exp300, w, self.root) for w in (64, 128, 256)}
        with tempfile.TemporaryDirectory(prefix="exp688-smoke-", dir=self.output) as directory:
            path = Path(directory); write_json(path / "channel-slice-manifest.json", self.manifest)
            models, records = self.train_routes(path, 1); evaluations = self.evaluate_routes(models, 1, 1)
            runtime = candidate_runtime_once(self.config, Path(records["hidden_aligned"]["checkpoint"]["path"]), int(self.root), 1)
            synthetic = {**runtime, "cases": 1100, "complete_1100_case_route": True, "output_count": 1100,
                         "resource": {**runtime["resource"], "wall_seconds": float(runtime["resource"]["wall_seconds"]) * 1100}}
            admission = self.admission(records, evaluations, {"warmup": runtime, "cold": [synthetic] * 3}, Timer().finish(), 1, 1, 1)
            randomized = new_student(self.config, None, 688999); reset = new_student(self.config, self.pruned)
            terminals = []
            for mode in ("calibration", "formal", "confirmation"):
                target = path / f"{mode}-terminal.json"; write_json(target, {"schema_version": "exp688-smoke-terminal-v1", "mode": mode, "status": "completed", "records": records, "evaluations": evaluations, "admission": admission}); json.loads(target.read_text()); terminals.append(target.name)
            result = {"schema_version": "exp688-rootless-all-mode-smoke-v1", "status": "completed", "root": None,
                      "all_modes": ["smoke", "calibration", "formal", "confirmation"], "channel_map_frozen": len(self.manifest["selected_source_channels_ranked"]) == 56,
                      "one_update_both_arms": all(records[a]["training"]["updates"] == 1 for a in TRAINED_ARMS),
                      "one_case_all_routes_widths": all(evaluations[a][str(w)][kind]["cases"] == 1 for a in ARM_ORDER for w in (128, 256) for kind in ("transition", "rollout")),
                      "identical_initial_bytes": records["hidden_aligned"]["initial_state_sha256"] == records["logit_only"]["initial_state_sha256"],
                      "identical_teacher_hidden_work": all(records["hidden_aligned"]["training"][key] == records["logit_only"]["training"][key] for key in ("teacher_calls", "student_calls", "hidden_capture_calls", "hidden_alignment_calls", "exact_loss_calls")),
                      "only_hidden_coefficient_differs": records["hidden_aligned"]["training"]["history"][0]["hidden_distillation_weight"] == 0.05 and records["logit_only"]["training"]["history"][0]["hidden_distillation_weight"] == 0.0,
                      "nonzero_hidden_gradient": records["hidden_aligned"]["training"]["history"][0]["hidden_gradient_signal"] > 0,
                      "reload_exact": all(x["reload_exact"] for x in records.values()), "randomized_reset_differ": state_hash(randomized.state_dict()) != state_hash(reset.state_dict()),
                      "candidate_route_executed": runtime["output_count"] == 1, "admission_math_exercised": bool(admission["checks"]),
                      "terminal_serializations": terminals, "native_hidden_capture_choice": "Native PyTorch calls the authenticated recurrent_update sequence and captures only training tensors; inference uses EXP687 student-only rollout."}
        self.root, self.splits = old_root, old_splits; write_json(self.output / "smoke.json", result); return result

    def calibration(self):
        path = self.open_root(); timer = Timer(); updates = int(self.config["training"]["calibration_updates"])
        transitions = int(self.config["calibration"]["transition_cases_per_route"]); rollouts = int(self.config["calibration"]["rollout_cases_per_route"])
        models, records = self.train_routes(path, updates); evaluations = self.evaluate_routes(models, transitions, rollouts)
        checkpoint = Path(records["hidden_aligned"]["checkpoint"]["path"]); warmup = candidate_runtime_once(self.config, checkpoint, int(self.root), 1)
        cold = [candidate_runtime_once(self.config, checkpoint, int(self.root)) for _ in range(int(self.config["calibration"]["candidate_repetitions"]))]
        resource = timer.finish(); admission = self.admission(records, evaluations, {"warmup": warmup, "cold": cold}, resource, updates, transitions, rollouts)
        decision = "admit-root-20261307" if admission["admitted"] else "park-exp688-at-root20261306-resource-admission"
        result = {"schema_version": "exp688-calibration-v1", "run_id": RUN_ID, "status": "completed", "mode": self.mode, "root": self.root,
                  "decision": decision, "source": self.identity(), "channel_slice_manifest": self.manifest, "records": records, "evaluations": evaluations,
                  "candidate_runtime": {"warmup": warmup, "cold": cold}, "resource": resource, "admission": admission, "limitations": self.limitations(True)}
        write_json(path / "root-summary.json", result); write_json(self.output / "calibration.json", result); return result

    def admitted(self):
        path = self.output / "calibration.json"
        if not path.is_file(): raise RuntimeError("science requires root20261306 admission")
        value = json.loads(path.read_text())
        if value["decision"] != "admit-root-20261307" or not value["admission"]["admitted"] or value["source"]["runner_sha256"] != sha256(Path(__file__)):
            raise RuntimeError("matching admitted calibration required")
        return value

    def science(self):
        calibration = self.admitted(); self.require_predecessor(); path = self.open_root(); timer = Timer()
        models, records = self.train_routes(path, int(self.config["training"]["updates"]))
        evaluations = self.evaluate_routes(models, int(self.config["evaluation"]["transition_cases_per_width"]), int(self.config["evaluation"]["rollout_cases_per_width"]))
        primary = evaluations["hidden_aligned"]["256"]; control = evaluations["logit_only"]
        frozen = evaluations["frozen_pruned"]["256"]; randomized = evaluations["randomized_control"]["256"]; reset = evaluations["reset_control"]["256"]
        score = primary["rollout"]["exact_accuracy"]
        gates = {"primary_w256_rollout": score >= float(self.config["gates"]["minimum_primary_w256_rollout"]),
                 "primary_w256_transition": primary["transition"]["exact_accuracy"] >= float(self.config["gates"]["minimum_primary_w256_transition"]),
                 "invalid_rate": primary["rollout"]["any_invalid_residue_rate"] <= float(self.config["gates"]["maximum_invalid_rate"]),
                 "gain_over_logit_only": score - control["256"]["rollout"]["exact_accuracy"] >= float(self.config["gates"]["minimum_gain_over_logit_only"]),
                 "w128_retention": evaluations["hidden_aligned"]["128"]["rollout"]["exact_accuracy"] >= control["128"]["rollout"]["exact_accuracy"] - float(self.config["gates"]["maximum_w128_loss_to_logit_only"]),
                 "nonzero_hidden_gradient": records["hidden_aligned"]["training"]["history"][-1]["hidden_gradient_signal"] > 0,
                 "matched_work": all(records["hidden_aligned"]["training"][key] == records["logit_only"]["training"][key] for key in ("teacher_calls", "student_calls", "hidden_capture_calls", "hidden_alignment_calls", "exact_loss_calls")),
                 "trained_weight_dependence": score - randomized["rollout"]["exact_accuracy"] >= float(self.config["gates"]["minimum_trained_weight_drop"]),
                 "reset_matches_frozen": reset["rollout"]["prediction_sha256"] == frozen["rollout"]["prediction_sha256"],
                 "reload_exact": all(x["reload_exact"] for x in records.values()),
                 "candidate_runtime": float(calibration["admission"]["candidate_guarded_wall_seconds"]) <= float(self.config["gates"]["maximum_candidate_guarded_wall_seconds"]),
                 "confirmation_threshold": score >= float(self.config["gates"]["confirmation_admission_rollout"])}
        mechanism = all(value for key, value in gates.items() if key != "confirmation_threshold"); confirmation = mechanism and gates["confirmation_threshold"]
        decision = ("advance-to-confirmations" if confirmation else "reject-exp688-compact-hidden-trajectory-distillation") if self.mode == "formal" else ("confirmation-pass" if confirmation else "reject-exp688-confirmation")
        result = {"schema_version": "exp688-terminal-v1", "run_id": RUN_ID, "status": "completed", "mode": self.mode, "root": self.root, "decision": decision,
                  "mechanism_gate_passed": mechanism, "confirmation_gate_passed": confirmation, "source": self.identity(), "records": records,
                  "evaluations": evaluations, "gates": gates, "dynamic_admission": calibration["admission"], "resource": timer.finish(), "limitations": self.limitations(False)}
        write_json(path / "root-summary.json", result); write_json(self.output / "terminal-summary.json", result); return result

    @staticmethod
    def limitations(calibration: bool) -> list[str]:
        return ["Calibration exact accuracy is route integrity evidence, not scientific evidence." if calibration else "Only fresh generated identities are evaluated; no checked, independent, official or private benchmark is opened.",
                "Exact arithmetic, frozen teacher, hidden capture and hidden alignment are training/evaluation-only; formal inference loads only compact student weights.",
                "Native PyTorch exposes the authenticated final recurrent tensor without hooks or source mutation; no sandbox dependency is added.",
                "This is intermediate research and authorizes no candidate, submission, release or incumbent promotion."]


def log_result(experiment: Experiment, result: dict[str, Any]) -> dict[str, Any]:
    if experiment.mode == "smoke": return result
    experiment_id = configure_mlflow(experiment.config["tracking"]["experiment_name"])
    with mlflow.start_run(run_name=f"{RUN_ID}-{experiment.mode}-{experiment.root}") as active:
        mlflow.set_tags({"canonical_run_id": RUN_ID, "mode": experiment.mode, "root": str(experiment.root), "decision": result["decision"], "candidate_registration": "false"})
        mlflow.log_dict(result, f"{experiment.mode}-result.json"); run_id = active.info.run_id
    record = {"tracking_uri": tracking_uri(), "experiment_id": experiment_id, "run_id": run_id, "mode": experiment.mode, "root": experiment.root}
    write_json(experiment.output / f"{experiment.mode}-mlflow.json", record); result["mlflow"] = record
    write_json(experiment.root_dir() / "root-summary.json", result); write_json(experiment.output / ("calibration.json" if experiment.mode == "calibration" else "terminal-summary.json"), result); return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=ROOT / ".scratch/exp-688-compact-hidden-trajectory-distillation")
    parser.add_argument("--mode", choices=("smoke", "calibration", "formal", "confirmation"), required=True); parser.add_argument("--root", type=int)
    args = parser.parse_args(); experiment = Experiment(args.config, args.output, args.mode, args.root)
    result = experiment.smoke() if args.mode == "smoke" else experiment.calibration() if args.mode == "calibration" else experiment.science()
    result = log_result(experiment, result); print(json.dumps({"mode": args.mode, "root": result.get("root"), "status": result["status"], "decision": result.get("decision"), "gates": result.get("gates"), "admission": result.get("admission", {}).get("checks")}, indent=2, sort_keys=True), flush=True); return 0


if __name__ == "__main__": raise SystemExit(main())
