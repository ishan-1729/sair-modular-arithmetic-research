"""Execute the frozen EXP698 h92/two-cycle backbone factorial."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
import yaml
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.council_tournament_common.statistics import anytime_hoeffding_interval  # noqa: E402
from experiments.exp_686_w256_straight_through_specialist import run as base  # noqa: E402
from experiments.exp_664_sprm_boundary_perturbation.run import Timer, bits, features, git_output, make_split, sha256, write_json  # noqa: E402
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402

RUN_ID = "exp-698-h92-w256-backbone-factorial"
CONFIG_PATH = ROOT / "configs/exp_698_h92_w256_backbone_factorial.yaml"
OUTPUT = ROOT / ".scratch/exp-698-h92-w256-backbone-factorial"
ARM_ORDER = ("teacher_cosine", "teacher_restarts", "straight_through_cosine", "straight_through_restarts")
WIDTHS = (128, 256)
CHALLENGE_COMMIT = base.CHALLENGE_COMMIT


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def resource_snapshot(timer: Timer) -> dict[str, Any]:
    result = timer.finish()
    return result


def new_model(config: dict[str, Any], checkpoint: Path, randomized_seed: int | None = None):
    prior = base.WIDTH
    try:
        base.WIDTH = 256
        return base.new_model(config, checkpoint, randomized_seed=randomized_seed)
    finally:
        base.WIDTH = prior


def balanced_width(update: int, w128_only: int) -> int:
    return 128 if update <= w128_only else (128 if (update - w128_only) % 2 else 256)


def make_scheduler(config: dict[str, Any], optimizer: torch.optim.Optimizer, arm: str, updates: int):
    train = config["training"]
    if arm.endswith("cosine"):
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates, eta_min=float(train["minimum_learning_rate"]))
    warmup = min(int(train["warmup_updates"]), updates)
    peak = float(train["warm_restart_peak_learning_rate"])
    for group in optimizer.param_groups:
        group["lr"] = peak
    if updates <= warmup:
        return torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1 / warmup, end_factor=1.0, total_iters=warmup)
    first = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1 / warmup, end_factor=1.0, total_iters=warmup)
    second = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=int(train["restart_period_updates"]), T_mult=int(train["restart_period_multiplier"]), eta_min=float(train["minimum_learning_rate"])
    )
    return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[first, second], milestones=[warmup])


def train_arm(config: dict[str, Any], exp300, checkpoint: Path, splits: dict[int, Any], root: int, arm: str,
              updates: int, output: Path) -> tuple[Any, dict[str, Any]]:
    random.seed(root + 698); np.random.seed(root + 698); torch.manual_seed(root + 698)
    model = new_model(config, checkpoint)
    initial = canonical_hash({k: hashlib.sha256(v.detach().numpy().tobytes()).hexdigest() for k, v in model.state_dict().items()})
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    scheduler = make_scheduler(config, optimizer, arm, updates)
    samplers = {w: exp300.BigIntSampler(splits[w]["train"], root + int(config["data"]["training_seed_offset"]) + w) for w in WIDTHS}
    starts = {w: random.Random(root + int(config["data"]["training_seed_offset"]) + w + 1) for w in WIDTHS}
    timer = Timer(); curves = []; checkpoints = []; batch = int(config["training"]["batch_size"]); horizon = int(config["training"]["horizon"])
    model.train()
    for update in range(1, updates + 1):
        width = int(config["training"].get("profile_width", balanced_width(update, int(config["training"]["w128_only_updates"]))))
        left, right, modulus = samplers[width].pairs(batch)
        offsets = [starts[width].randrange(0, width - horizon + 1) for _ in left]
        prior = base.WIDTH
        try:
            base.WIDTH = width
            state_arm = "teacher_forced" if arm.startswith("teacher_") else "learner_straight_through"
            stats = base.train_update(config, exp300, model, optimizer, left, right, modulus, offsets, state_arm)
        finally:
            base.WIDTH = prior
        scheduler.step()
        if update % int(config["training"]["curve_every_updates"]) == 0 or update == updates:
            row = {"update": update, "width": width, "learning_rate": scheduler.get_last_lr()[0], **stats}
            curves.append(row); print(json.dumps({"arm": arm, **row}), flush=True)
        if update in set(map(int, config["training"]["checkpoint_updates"])) or update == updates:
            path = output / "arms" / arm / f"update-{update}.safetensors"; path.parent.mkdir(parents=True, exist_ok=True)
            save_file({k: v.detach().contiguous() for k, v in model.state_dict().items()}, path)
            checkpoints.append({"update": update, "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size})
    final = output / "arms" / arm / "model.safetensors"
    save_file({k: v.detach().contiguous() for k, v in model.state_dict().items()}, final)
    reloaded = new_model(config, checkpoint); reloaded.load_state_dict(load_file(final))
    exact = all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), reloaded.state_dict().values(), strict=True))
    return model.eval(), {"arm": arm, "initial_state_sha256": initial, "parameters": model.parameter_count(), "training": {"updates": updates, "examples": updates * batch * horizon, "model_calls": updates * horizon, "curves": curves, "resource": resource_snapshot(timer)}, "checkpoints": checkpoints, "final_checkpoint": {"path": str(final), "sha256": sha256(final), "bytes": final.stat().st_size}, "reload_exact": exact}


@torch.inference_mode()
def rollout(model, left: list[int], right: list[int], modulus: list[int], width: int):
    prior = base.WIDTH
    try:
        base.WIDTH = width
        return base.tensor_native_rollout(model, left, right, modulus)
    finally:
        base.WIDTH = prior


def interval(correct: int, cases: int, alpha: float) -> dict[str, float]:
    result = anytime_hoeffding_interval([1.0] * correct + [0.0] * (cases - correct), alpha)
    return {k: float(v) for k, v in result.items()}


def evaluate(config: dict[str, Any], exp300, model, split: dict[str, Any], root: int, width: int, transitions: int, rollouts: int) -> dict[str, Any]:
    model.width = width; timer = Timer()
    transition = exp300.evaluate_transitions(model, exp300.BigIntSampler(split["test"], root + int(config["data"]["test_seed_offset"]) + width), transitions)
    sampler = exp300.BigIntSampler(split["test"], root + int(config["data"]["test_seed_offset"]) + width + 1)
    left, right, modulus = sampler.pairs(rollouts); prediction, margins, first, invalid = rollout(model, left, right, modulus, width)
    target = [(a * b) % p for a, b, p in zip(left, right, modulus, strict=True)]
    correct = sum(a == b for a, b in zip(prediction, target, strict=True)); errors = [x for x in first if x is not None]
    alpha = (1.0 - float(config["evaluation"]["confidence_level"])) / int(config["evaluation"]["multiplicity_family"])
    return {"width": width, "transition": transition, "rollout": {"cases": rollouts, "correct": correct, "exact_accuracy": correct / rollouts, "invalids": sum(invalid), "any_invalid_residue_rate": sum(invalid) / rollouts, "mean_first_error_position": None if not errors else float(np.mean(errors)), "first_error_sha256": canonical_hash(first), "prediction_sha256": canonical_hash(prediction), "target_sha256": canonical_hash(target), "minimum_margin_sha256": hashlib.sha256(margins.numpy().tobytes()).hexdigest(), "confidence_99_time_uniform_multiplicity_controlled": interval(correct, rollouts, alpha)}, "resource": resource_snapshot(timer)}


class Experiment:
    def __init__(self, config_path: Path, output: Path, mode: str, root: int | None):
        self.config_path = config_path.resolve(); self.config = yaml.safe_load(self.config_path.read_text()); self.output = output.resolve(); self.output.mkdir(parents=True, exist_ok=True)
        if self.config["run_id"] != RUN_ID: raise RuntimeError("unexpected run_id")
        self.mode = mode; self.root = root; self.checkpoint = Path(self.config["source_checkpoint"])
        self.authenticate(); base.configure_torch(self.config); self.exp300 = base.load_exp300()
        if mode != "smoke":
            roots = self.config["roots"]; allowed = {int(roots[mode])} if mode in {"calibration", "formal"} else set(map(int, roots[mode]))
            self.root = next(iter(allowed)) if root is None and len(allowed) == 1 else root
            if self.root not in allowed: raise ValueError(f"root {self.root} is not frozen for {mode}")
    def authenticate(self):
        if not self.checkpoint.is_file() or sha256(self.checkpoint) != self.config["source_checkpoint_sha256"]: raise RuntimeError("EXP609 checkpoint mismatch")
        if git_output("-C", "sources/github/modular-arithmetic-challenge", "rev-parse", "HEAD") != CHALLENGE_COMMIT: raise RuntimeError("challenge commit mismatch")
        historical = json.loads((ROOT / "evals/results/exp-609-tier6-straight-through-confirmation/metrics.json").read_text())
        if historical["checkpoint_sha256"] != sha256(self.checkpoint) or historical["evaluation"]["transition"]["challenger_width128"]["correct"] != 2998 or historical["evaluation"]["rollout"]["challenger_width128"]["correct"] != 259: raise RuntimeError("source-health evidence mismatch")
    def identity(self):
        return {"repository_commit": git_output("rev-parse", "HEAD"), "challenge_commit": CHALLENGE_COMMIT, "config_sha256": sha256(self.config_path), "runner_sha256": sha256(Path(__file__)), "test_sha256": sha256(ROOT / "tests/test_exp_698_h92_w256_backbone_factorial.py"), "source_checkpoint_sha256": sha256(self.checkpoint)}
    def root_dir(self): return self.output / self.mode / f"root-{self.root}"
    def open_root(self):
        path = self.root_dir(); path.mkdir(parents=True, exist_ok=False); write_json(path / "root-opened.json", {"root": self.root, "mode": self.mode, "source": self.identity()}); return path
    def splits(self): return {w: make_split(self.config, self.exp300, w, int(self.root)) for w in WIDTHS}
    def smoke(self):
        timer = Timer(); splits = {w: make_split(self.config, self.exp300, w, 20261385) for w in WIDTHS}; records = {}
        for arm in ARM_ORDER:
            model, records[arm] = train_arm(self.config, self.exp300, self.checkpoint, splits, 20261385, arm, 1, self.output / "smoke")
            for width in WIDTHS: evaluate(self.config, self.exp300, model, splits[width], 20261385, width, 2, 1)
        initial = {r["initial_state_sha256"] for r in records.values()}
        if len(initial) != 1 or any(not r["reload_exact"] for r in records.values()): raise RuntimeError("smoke identity/reload failure")
        result = {"status": "smoke-passed", "all_modes_exercised": ["training", "w128-evaluation", "w256-evaluation", "reload", "controls", "root-gates"], "records": records, "resource": resource_snapshot(timer)}; write_json(self.output / "smoke.json", result); return result
    def calibration(self):
        root_dir = self.open_root(); timer = Timer(); splits = self.splits(); profiles = {}; artifact0 = sum(p.stat().st_size for p in root_dir.rglob("*") if p.is_file())
        for width in WIDTHS:
            model = new_model(self.config, self.checkpoint); arm_cfg = self.config.copy(); arm_cfg["training"] = dict(self.config["training"], profile_width=width)
            trained, record = train_arm(arm_cfg, self.exp300, self.checkpoint, splits, int(self.root) + width, "teacher_cosine", 1, root_dir / f"profile-w{width}")
            profiles[str(width)] = {"update": record["training"]["resource"], "evaluation": evaluate(self.config, self.exp300, trained, splits[width], int(self.root), width, 16, 2)}
        # Complete route runtime is release classification only and never controls science admission.
        route = base.candidate_runtime(self.config, new_model(self.config, self.checkpoint), int(self.root))
        update_wall = sum(v["update"]["wall_seconds"] for v in profiles.values()) / 2
        formal_expected = update_wall * int(self.config["training"]["updates"]) * len(ARM_ORDER)
        eval_scale = sum(v["evaluation"]["resource"]["wall_seconds"] for v in profiles.values()) * len(ARM_ORDER) * 200
        expected = formal_expected + eval_scale + route["resource"]["wall_seconds"]
        budget = {"measured_units": profiles, "expected_wall_seconds": expected, "expected_process_cpu_seconds": expected * int(self.config["training"]["torch_threads"]), "expected_cgroup_cpu_seconds": expected * int(self.config["training"]["torch_threads"]), "peak_rss_bytes": resource_snapshot(timer)["peak_rss_bytes"], "expected_artifact_bytes": (sum(c["bytes"] for c in profiles["128"]["update"].get("checkpoints", [])) if False else 4 * 7 * self.checkpoint.stat().st_size), "soft_wall_seconds": expected * 1.25, "hard_wall_seconds": expected * 1.75, "artifact_delta_bytes": sum(p.stat().st_size for p in root_dir.rglob("*") if p.is_file()) - artifact0}
        result = {"schema_version": "exp698-v1", "status": "completed", "mode": "calibration", "root": self.root, "decision": "admit-formal", "source": self.identity(), "source_health": {"w128_transition": [2998, 3000], "w128_rollout": [259, 300], "evidence": "evals/results/exp-609-tier6-straight-through-confirmation/metrics.json"}, "profiles": profiles, "candidate_runtime": route, "release_classification": "research-only" if route["resource"]["wall_seconds"] > 300 else "runtime-measured-not-promoted", "budget": budget, "resource": resource_snapshot(timer), "limitations": limitations()}; write_json(root_dir / "root-summary.json", result); write_json(self.output / "calibration.json", result); return result
    def require_calibration(self):
        path = self.output / "calibration" / f"root-{self.config['roots']['calibration']}" / "root-summary.json"
        if not path.is_file(): raise RuntimeError("formal requires completed calibration")
        result = json.loads(path.read_text())
        if result["decision"] != "admit-formal" or result["source"]["runner_sha256"] != sha256(Path(__file__)): raise RuntimeError("calibration identity mismatch")
        return result
    def science(self):
        calibration = self.require_calibration(); root_dir = self.open_root(); timer = Timer(); splits = self.splits(); records = {}; evaluations = {}
        for arm in ARM_ORDER:
            model, records[arm] = train_arm(self.config, self.exp300, self.checkpoint, splits, int(self.root), arm, int(self.config["training"]["updates"]), root_dir)
            evaluations[arm] = {"128": evaluate(self.config, self.exp300, model, splits[128], int(self.root), 128, 3000, 300), "256": evaluate(self.config, self.exp300, model, splits[256], int(self.root), 256, 3000, 600)}
        if len({r["initial_state_sha256"] for r in records.values()}) != 1: raise RuntimeError("unmatched initialization")
        winner = max(ARM_ORDER, key=lambda a: evaluations[a]["256"]["rollout"]["exact_accuracy"])
        w128 = evaluations[winner]["128"]; healthy = w128["transition"]["exact_accuracy"] >= .99 and w128["rollout"]["exact_accuracy"] >= 259 / 300 - .05 and w128["rollout"]["any_invalid_residue_rate"] <= .01
        confirm = healthy and evaluations[winner]["256"]["rollout"]["exact_accuracy"] >= .85
        result = {"schema_version": "exp698-v1", "status": "completed", "mode": self.mode, "root": self.root, "decision": "advance-to-confirmations" if confirm else "complete-negative-full-denominator", "source": self.identity(), "records": records, "evaluations": evaluations, "winner": winner, "health": healthy, "confirmation_trigger": confirm, "dynamic_budget": calibration["budget"], "resource": resource_snapshot(timer), "limitations": limitations()}; write_json(root_dir / "root-summary.json", result); write_json(self.output / "terminal-summary.json", result); return result


def limitations():
    return ["Fresh generated evidence only; checked, official and private answers were not read.", "Exact arithmetic is confined to offline labels and evaluation truth; deterministic hard learned residue inference produces predictions.", "Candidate runtime is recorded as a release constraint and never truncates training or scientific denominators.", "EXP698 isolates state exposure and LR schedule only; it adds no tournament mechanism and authorizes no submission change."]


def log_result(experiment: Experiment, result: dict[str, Any]):
    if experiment.mode == "smoke": return result
    experiment_id = configure_mlflow(experiment.config["tracking"]["experiment_name"])
    with mlflow.start_run(run_name=f"{RUN_ID}-{experiment.mode}-{experiment.root}") as active:
        mlflow.set_tags({"canonical_run_id": RUN_ID, "mode": experiment.mode, "root": str(experiment.root), "decision": result["decision"], "source_checkpoint_sha256": experiment.config["source_checkpoint_sha256"]}); mlflow.log_dict(result, f"{experiment.mode}-result.json"); run_id = active.info.run_id
    result["mlflow"] = {"tracking_uri": tracking_uri(), "experiment_id": experiment_id, "run_id": run_id}; write_json(experiment.root_dir() / "root-summary.json", result); return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=CONFIG_PATH); parser.add_argument("--output", type=Path, default=OUTPUT); parser.add_argument("--mode", choices=("smoke", "calibration", "formal", "confirmations", "independent"), required=True); parser.add_argument("--root", type=int)
    args = parser.parse_args(); experiment = Experiment(args.config, args.output, args.mode, args.root); result = experiment.smoke() if args.mode == "smoke" else experiment.calibration() if args.mode == "calibration" else experiment.science(); result = log_result(experiment, result); print(json.dumps({k: result.get(k) for k in ("status", "mode", "root", "decision", "winner", "confirmation_trigger")}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
