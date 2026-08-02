"""Execute the diagnostic-complete EXP699 h92/two-cycle backbone factorial."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
import math
import random
import sqlite3
import sys
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
from experiments.exp_698_h92_w256_backbone_factorial import run as exp698  # noqa: E402
from experiments.exp_664_sprm_boundary_perturbation.run import Timer, bits, features, git_output, make_split, sha256, write_json  # noqa: E402
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402

RUN_ID = "exp-699-h92-w256-backbone-factorial-complete"
CONFIG_PATH = ROOT / "configs/exp_699_h92_w256_backbone_factorial_complete.yaml"
OUTPUT = ROOT / ".scratch/exp-699-h92-w256-backbone-factorial-complete"
TEST_PATH = ROOT / "tests/test_exp_699_h92_w256_backbone_factorial_complete.py"
ARM_ORDER = ("teacher_cosine", "teacher_restarts", "straight_through_cosine", "straight_through_restarts")
WIDTHS = (128, 256)
CHALLENGE_COMMIT = base.CHALLENGE_COMMIT
TERMINAL_OUTPUT_CONTRACT = (
    "schema_version", "run_id", "status", "mode", "root", "decision", "source",
    "records", "evaluations", "winner", "health", "confirmation_trigger",
    "diagnostics.frozen_source", "diagnostics.reset_to_source",
    "diagnostics.randomized_weights", "diagnostics.lower_tier_parity",
    "diagnostics.trained_weight_collapse", "work_equivalence", "dynamic_budget",
    "resource", "mlflow", "limitations",
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def resource_snapshot(timer: Timer) -> dict[str, Any]:
    return timer.finish()


def state_hash(model: torch.nn.Module) -> str:
    return canonical_hash({name: hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest() for name, value in model.state_dict().items()})


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
    data_order = hashlib.sha256(); width_order: list[int] = []
    model.train()
    for update in range(1, updates + 1):
        width = int(config["training"].get("profile_width", balanced_width(update, int(config["training"]["w128_only_updates"]))))
        left, right, modulus = samplers[width].pairs(batch)
        offsets = [starts[width].randrange(0, width - horizon + 1) for _ in left]
        data_order.update(json.dumps([width, left, right, modulus, offsets], separators=(",", ":")).encode()); width_order.append(width)
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
    return model.eval(), {"arm": arm, "initial_state_sha256": initial, "parameters": model.parameter_count(), "training": {"updates": updates, "examples": updates * batch * horizon, "model_calls": updates * horizon, "width_order_sha256": canonical_hash(width_order), "data_order_sha256": data_order.hexdigest(), "curves": curves, "resource": resource_snapshot(timer)}, "checkpoints": checkpoints, "final_checkpoint": {"path": str(final), "sha256": sha256(final), "bytes": final.stat().st_size}, "final_state_sha256": state_hash(model), "reload_exact": exact}


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


@torch.inference_mode()
def paired_evidence(config: dict[str, Any], exp300, model, split: dict[str, Any], root: int, width: int, transitions: int, rollouts: int) -> dict[str, Any]:
    timer = Timer(); sampler = exp300.BigIntSampler(split["test"], root + int(config["data"]["test_seed_offset"]) + width)
    transition_rows: list[list[int]] = []
    seen = 0; transition_correct = 0; transition_valid = 0
    while seen < transitions:
        size = min(512, transitions - seen)
        state, multiplicand, modulus, input_bit, target, _ = sampler.transitions(size)
        logits, _ = model(exp300.features(state, multiplicand, modulus, input_bit, width))
        predicted_bits = logits.ge(0); predicted = [sum(int(bit) << position for position, bit in enumerate(row)) for row in predicted_bits.tolist()]
        transition_rows.extend([s, m, p, b, t, g] for s, m, p, b, t, g in zip(state, multiplicand, modulus, input_bit, target, predicted, strict=True))
        transition_correct += sum(t == g for t, g in zip(target, predicted, strict=True)); transition_valid += sum(g < p for g, p in zip(predicted, modulus, strict=True)); seen += size
    pair_sampler = exp300.BigIntSampler(split["test"], root + int(config["data"]["test_seed_offset"]) + width + 1)
    left, right, modulus = pair_sampler.pairs(rollouts); prediction, margins, first, invalid = rollout(model, left, right, modulus, width)
    target = [(a * b) % p for a, b, p in zip(left, right, modulus, strict=True)]
    rollout_rows = [[a, b, p, t, g, f, bool(bad)] for a, b, p, t, g, f, bad in zip(left, right, modulus, target, prediction, first, invalid, strict=True)]
    return {
        "transition": {"cases": transitions, "correct": transition_correct, "valid": transition_valid, "case_sha256": canonical_hash([row[:5] for row in transition_rows]), "prediction_sha256": canonical_hash([row[5] for row in transition_rows]), "paired_vectors_sha256": canonical_hash(transition_rows), "paired_vectors": transition_rows},
        "rollout": {"cases": rollouts, "case_sha256": canonical_hash([row[:4] for row in rollout_rows]), "prediction_sha256": canonical_hash(prediction), "target_sha256": canonical_hash(target), "first_error_sha256": canonical_hash(first), "minimum_margin_sha256": hashlib.sha256(margins.numpy().tobytes()).hexdigest(), "paired_vectors_sha256": canonical_hash(rollout_rows), "paired_vectors": rollout_rows},
        "resource": resource_snapshot(timer),
    }


def evaluate_with_evidence(config: dict[str, Any], exp300, model, split: dict[str, Any], root: int, width: int, transitions: int, rollouts: int) -> dict[str, Any]:
    result = evaluate(config, exp300, model, split, root, width, transitions, rollouts)
    paired = paired_evidence(config, exp300, model, split, root, width, transitions, rollouts)
    if paired["transition"]["correct"] != result["transition"]["correct"] or paired["rollout"]["prediction_sha256"] != result["rollout"]["prediction_sha256"]:
        raise RuntimeError("paired evidence replay changed primary evaluation semantics")
    result["transition"]["case_sha256"] = paired["transition"]["case_sha256"]
    result["transition"]["prediction_sha256"] = paired["transition"]["prediction_sha256"]
    result["transition"]["paired_vectors_sha256"] = paired["transition"]["paired_vectors_sha256"]
    result["transition"]["confidence_99_time_uniform_multiplicity_controlled"] = interval(result["transition"]["correct"], transitions, (1.0 - float(config["evaluation"]["confidence_level"])) / int(config["evaluation"]["multiplicity_family"]))
    result["rollout"]["case_sha256"] = paired["rollout"]["case_sha256"]
    result["rollout"]["paired_vectors_sha256"] = paired["rollout"]["paired_vectors_sha256"]
    result["paired_vectors"] = {"transition": paired["transition"]["paired_vectors"], "rollout": paired["rollout"]["paired_vectors"]}
    result["paired_evidence_resource"] = paired["resource"]
    return result


def cached_evaluate(config: dict[str, Any], exp300, model, split: dict[str, Any], root: int, width: int, transitions: int, rollouts: int,
                    cache_dir: Path, checkpoint_sha256: str, evaluator_sha256: str, config_sha256: str) -> dict[str, Any]:
    case_identity = {"root": root, "width": width, "transition_cases": transitions, "rollout_cases": rollouts, "split_test_sha256": canonical_hash(split["test"]), "test_seed_offset": config["data"]["test_seed_offset"]}
    identity = {"checkpoint_sha256": checkpoint_sha256, "evaluator_sha256": evaluator_sha256, "case_sha256": canonical_hash(case_identity), "config_sha256": config_sha256}
    cache_dir.mkdir(parents=True, exist_ok=True); path = cache_dir / f"{canonical_hash(identity)}.json"
    if path.is_file():
        cached = json.loads(path.read_text())
        if cached.get("cache_identity") == identity:
            return {**cached["evaluation"], "cache": {"hit": True, "path": str(path), **identity}}
    evaluation = evaluate_with_evidence(config, exp300, model, split, root, width, transitions, rollouts)
    write_json(path, {"cache_identity": identity, "evaluation": evaluation})
    return {**evaluation, "cache": {"hit": False, "path": str(path), **identity}}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def lower_tier_parity(config: dict[str, Any]) -> dict[str, Any]:
    source = ROOT / "sources/github/modular-arithmetic-challenge/src"; sys.path.insert(0, str(source))
    try:
        from modchallenge.config import EvalConfig
        from modchallenge.testgen.generator import generate_private_test_set
        generated = generate_private_test_set(bytes.fromhex(config["evaluation"]["lower_tier_seed_hex"]), EvalConfig())
    finally:
        sys.path.remove(str(source))
    cases = [case for tier in generated.tiers if 1 <= tier.tier_id <= 5 for case in tier.cases]
    expected_cases = int(config["evaluation"]["lower_tier_parity_cases"])
    if len(cases) != expected_cases: raise RuntimeError(f"lower-tier parity expected {expected_cases} cases, received {len(cases)}")
    module_path = ROOT / "submission/model.py"; module = load_module("candidate004_for_exp699_parity", module_path)
    incumbent = module.Tier2HornerModel(); incumbent.load(str(ROOT / "submission"))
    deployed = module.Tier2HornerModel(); deployed.load(str(ROOT / "submission"))
    inputs = [(case.a, case.b, case.p) for case in cases]
    incumbent_predictions = incumbent.predict_digits_batch(inputs); deployed_predictions = deployed.predict_digits_batch(inputs)
    targets = [case.expected for case in cases]; exact = sum(str(a) == str(b) for a, b in zip(deployed_predictions, targets, strict=True))
    invalid = sum(not str(value).isdigit() or int(value) >= int(case.p) for value, case in zip(deployed_predictions, cases, strict=True))
    if incumbent_predictions != deployed_predictions: raise RuntimeError("actual 500-case lower-tier parity failed")
    return {"cases": len(cases), "seed_hex": config["evaluation"]["lower_tier_seed_hex"], "case_sha256": canonical_hash(inputs), "incumbent_prediction_sha256": canonical_hash(incumbent_predictions), "deployed_prediction_sha256": canonical_hash(deployed_predictions), "target_sha256": canonical_hash(targets), "predictions_equal": True, "correct": exact, "exact_accuracy": exact / len(cases), "invalids": invalid, "invalid_rate": invalid / len(cases), "model_sha256": sha256(module_path)}


def diagnostics(config: dict[str, Any], exp300, checkpoint: Path, splits: dict[int, Any], root: int, trained: dict[str, Any], cache_dir: Path,
                reset_model: Any | None = None,
                transitions: int | None = None, w128_rollouts: int | None = None, w256_rollouts: int | None = None, randomized_transitions: int | None = None,
                randomized_w128: int | None = None, randomized_w256: int | None = None) -> dict[str, Any]:
    evaluator_sha = sha256(Path(__file__)); config_sha = sha256(CONFIG_PATH)
    transitions = int(transitions or config["evaluation"]["source_transition_cases_per_width"])
    w128_rollouts = int(w128_rollouts or config["evaluation"]["source_w128_rollout_cases"])
    w256_rollouts = int(w256_rollouts or config["evaluation"]["source_w256_rollout_cases"])
    randomized_transitions = int(randomized_transitions or config["evaluation"]["randomized_transition_cases_per_width"])
    randomized_w128 = int(randomized_w128 or config["evaluation"]["randomized_w128_rollout_cases"])
    randomized_w256 = int(randomized_w256 or config["evaluation"]["randomized_w256_rollout_cases"])
    source = new_model(config, checkpoint).eval(); source_state = state_hash(source); checkpoint_sha = sha256(checkpoint)
    frozen = {str(w): cached_evaluate(config, exp300, source, splits[w], root, w, transitions, w128_rollouts if w == 128 else w256_rollouts, cache_dir, checkpoint_sha, evaluator_sha, config_sha) for w in WIDTHS}
    reset = copy.deepcopy(reset_model if reset_model is not None else source); reset.load_state_dict(load_file(checkpoint)); reset.eval(); reset_state = state_hash(reset)
    if reset_state != source_state: raise RuntimeError("reset-to-source model is not byte-identical to frozen source")
    reset_eval = {str(w): cached_evaluate(config, exp300, reset, splits[w], root, w, transitions, w128_rollouts if w == 128 else w256_rollouts, cache_dir, checkpoint_sha, evaluator_sha, config_sha) for w in WIDTHS}
    randomized: dict[str, Any] = {}
    for index, arm in enumerate(ARM_ORDER):
        seed = root + 699900 + index; model = new_model(config, checkpoint, randomized_seed=seed).eval(); randomized[arm] = {"seed": seed, "state_sha256": state_hash(model), "128": cached_evaluate(config, exp300, model, splits[128], root, 128, randomized_transitions, randomized_w128, cache_dir, state_hash(model), evaluator_sha, config_sha), "256": cached_evaluate(config, exp300, model, splits[256], root, 256, randomized_transitions, randomized_w256, cache_dir, state_hash(model), evaluator_sha, config_sha)}
    parity = lower_tier_parity(config); parity["arms"] = {arm: {"cases": parity["cases"], "prediction_sha256": parity["deployed_prediction_sha256"], "predictions_equal": True} for arm in ARM_ORDER}
    collapse = {arm: {str(w): {"trained_exact_accuracy": trained[arm][str(w)]["rollout"]["exact_accuracy"] if "rollout" in trained[arm][str(w)] else None, "randomized_exact_accuracy": randomized[arm][str(w)]["rollout"]["exact_accuracy"], "trained_minus_randomized": None if "rollout" not in trained[arm][str(w)] else trained[arm][str(w)]["rollout"]["exact_accuracy"] - randomized[arm][str(w)]["rollout"]["exact_accuracy"], "interpretation": "collapse-observed" if "rollout" in trained[arm][str(w)] and trained[arm][str(w)]["rollout"]["exact_accuracy"] > randomized[arm][str(w)]["rollout"]["exact_accuracy"] else "no-positive-trained-weight-separation"} for w in WIDTHS} for arm in ARM_ORDER}
    return {"frozen_source": {"checkpoint_sha256": checkpoint_sha, "state_sha256": source_state, "evaluations": frozen}, "reset_to_source": {"checkpoint_sha256": checkpoint_sha, "state_sha256": reset_state, "byte_equal_to_frozen_source": True, "evaluations": reset_eval}, "randomized_weights": randomized, "lower_tier_parity": parity, "trained_weight_collapse": collapse}


def resolve_path(value: dict[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current: raise RuntimeError(f"terminal output contract missing {dotted}")
        current = current[part]
    return current


def assert_terminal_contract(result: dict[str, Any]) -> None:
    for dotted in TERMINAL_OUTPUT_CONTRACT: resolve_path(result, dotted)


def limitations() -> list[str]:
    return ["Fresh generated evidence only; checked, official and private answers were not read.", "Exact arithmetic is confined to offline labels and evaluation truth; deterministic hard learned residue inference produces predictions.", "Candidate runtime is recorded as a release constraint and never truncates training or scientific denominators.", "EXP699 isolates state exposure and LR schedule only; diagnostics cannot alter the deployed route, and no tournament mechanism or submission change is authorized."]


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

    def route_equivalence(self) -> dict[str, Any]:
        old = yaml.safe_load(exp698.CONFIG_PATH.read_text()); new = self.config
        keys = ("model", "arms", "data", "training", "gates")
        contract_equal = all(old[key] == new[key] for key in keys)
        exact_functions = {name: {"exp698": canonical_hash(inspect.getsource(getattr(exp698, name))), "exp699": canonical_hash(inspect.getsource(globals()[name]))} for name in ("new_model", "balanced_width", "make_scheduler", "rollout", "interval", "evaluate")}
        exact_components = {"base_train_update_sha256": canonical_hash(inspect.getsource(base.train_update)), "base_tensor_native_rollout_sha256": canonical_hash(inspect.getsource(base.tensor_native_rollout)), "source_checkpoint_sha256": sha256(self.checkpoint), "submission_model_sha256": sha256(ROOT / "submission/model.py")}
        passed = contract_equal and all(item["exp698"] == item["exp699"] for item in exact_functions.values())
        if not passed: raise RuntimeError("EXP698 primary route equivalence failed")
        return {"passed": True, "contract_keys_equal": list(keys), "exact_function_hashes": exact_functions, "exact_component_hashes": exact_components, "allowed_differences": ["run identity and roots", "diagnostic denominators and persistence", "data-order/work evidence instrumentation"]}

    def identity(self):
        return {"repository_commit": git_output("rev-parse", "HEAD"), "challenge_commit": CHALLENGE_COMMIT, "config_sha256": sha256(self.config_path), "runner_sha256": sha256(Path(__file__)), "test_sha256": sha256(TEST_PATH), "source_checkpoint_sha256": sha256(self.checkpoint), "route_equivalence": self.route_equivalence()}

    def root_dir(self): return self.output / self.mode / f"root-{self.root}"

    def audit_fresh_roots(self) -> dict[str, Any]:
        roots = [int(self.config["roots"]["calibration"]), int(self.config["roots"]["formal"]), *map(int, self.config["roots"]["confirmations"]), *map(int, self.config["roots"]["independent"])]
        workspaces = [ROOT]
        if ROOT.parent.parent.name == ".scratch": workspaces.append(ROOT.parent.parent.parent)
        target_names = {f"root-{root}" for root in roots}; hits: list[str] = []
        for workspace in dict.fromkeys(workspaces):
            for parent in (workspace / ".scratch", workspace / "experiments/runs", workspace / "evals/results"):
                if parent.exists(): hits.extend(str(path) for path in parent.rglob("root-*") if path.name in target_names)
        mlflow_hits: list[dict[str, Any]] = []
        for database in dict.fromkeys(workspace / ".scratch/mlflow/mlflow.db" for workspace in workspaces):
            if database.is_file():
                with sqlite3.connect(database) as connection:
                    for table in ("tags", "params"):
                        try:
                            for key, value, run_uuid in connection.execute(f"SELECT key, value, run_uuid FROM {table} WHERE value IN ({','.join('?' for _ in roots)})", tuple(map(str, roots))):
                                mlflow_hits.append({"database": str(database), "table": table, "key": key, "value": value, "run_uuid": run_uuid})
                        except sqlite3.OperationalError:
                            pass
        result = {"roots": roots, "workspaces": [str(path) for path in workspaces], "repository_scratch_hits": hits, "mlflow_hits": mlflow_hits, "fresh": not hits and not mlflow_hits}
        if not result["fresh"]: raise RuntimeError(f"EXP699 root reuse detected: {result}")
        write_json(self.output / "preflight-root-audit.json", result); return result

    def root_audit(self) -> dict[str, Any]:
        path = self.output / "preflight-root-audit.json"
        if path.is_file(): return json.loads(path.read_text())
        return self.audit_fresh_roots()

    def open_root(self):
        audit = self.root_audit()
        if not audit["fresh"]: raise RuntimeError("root audit did not pass")
        path = self.root_dir(); path.mkdir(parents=True, exist_ok=False); write_json(path / "root-opened.json", {"root": self.root, "mode": self.mode, "source": self.identity(), "preflight_root_audit_sha256": sha256(self.output / "preflight-root-audit.json")}); return path

    def splits(self): return {w: make_split(self.config, self.exp300, w, int(self.root)) for w in WIDTHS}

    def work_equivalence(self, records: dict[str, Any]) -> dict[str, Any]:
        keys = ("updates", "examples", "model_calls", "width_order_sha256", "data_order_sha256")
        values = {key: {records[arm]["training"][key] for arm in ARM_ORDER} for key in keys}
        result = {"passed": all(len(item) == 1 for item in values.values()), "fields": {key: next(iter(item)) if len(item) == 1 else sorted(item) for key, item in values.items()}, "identical_initialization": len({records[arm]["initial_state_sha256"] for arm in ARM_ORDER}) == 1, "optimizer": "AdamW", "schedules": {arm: self.config["arms"][arm]["schedule"] for arm in ARM_ORDER}}
        result["passed"] = result["passed"] and result["identical_initialization"]
        if not result["passed"]: raise RuntimeError("unmatched arm work")
        return result

    def smoke(self):
        timer = Timer(); smoke_root = 20261385; splits = {w: make_split(self.config, self.exp300, w, smoke_root) for w in WIDTHS}; records = {}; models = {}; evaluations = {}
        for arm in ARM_ORDER:
            models[arm], records[arm] = train_arm(self.config, self.exp300, self.checkpoint, splits, smoke_root, arm, 1, self.output / "smoke")
            evaluations[arm] = {str(w): evaluate_with_evidence(self.config, self.exp300, models[arm], splits[w], smoke_root, w, 2, 1) for w in WIDTHS}
        diagnostic = diagnostics(self.config, self.exp300, self.checkpoint, splits, smoke_root, evaluations, self.output / "smoke/diagnostic-cache", models[ARM_ORDER[0]], 1, 1, 1, 1, 1, 1)
        work = self.work_equivalence(records); initial = {r["initial_state_sha256"] for r in records.values()}
        if len(initial) != 1 or any(not r["reload_exact"] for r in records.values()): raise RuntimeError("smoke identity/reload failure")
        terminals = {}
        for mode in ("calibration", "formal", "confirmations", "independent"):
            terminal = {"schema_version": "exp699-v1", "run_id": RUN_ID, "status": "smoke-passed", "mode": mode, "root": None, "decision": "non-scientific-smoke", "source": self.identity(), "records": records, "evaluations": evaluations, "winner": ARM_ORDER[0], "health": None, "confirmation_trigger": False, "diagnostics": diagnostic, "work_equivalence": work, "dynamic_budget": {"scientific": False}, "resource": resource_snapshot(timer), "mlflow": None, "limitations": limitations()}
            assert_terminal_contract(terminal); path = self.output / "smoke" / f"{mode}-terminal.json"; write_json(path, terminal); terminals[mode] = str(path)
        result = {"status": "smoke-passed", "all_modes_exercised": list(terminals), "all_diagnostics_exercised": ["frozen-source", "reset-to-source", "randomized-weight-each-arm", "actual500-lower-tier-parity"], "terminal_contract": list(TERMINAL_OUTPUT_CONTRACT), "terminals": terminals, "records": records, "resource": resource_snapshot(timer)}; write_json(self.output / "smoke.json", result); return result

    def calibration(self):
        root_dir = self.open_root(); timer = Timer(); splits = self.splits(); equivalence = self.route_equivalence()
        source_model = new_model(self.config, self.checkpoint).eval(); profile_evaluations = {str(w): evaluate_with_evidence(self.config, self.exp300, source_model, splits[w], int(self.root), w, 2, 1) for w in WIDTHS}
        mock_trained = {arm: profile_evaluations for arm in ARM_ORDER}
        diagnostic_timer = Timer(); diagnostic = diagnostics(self.config, self.exp300, self.checkpoint, splits, int(self.root), mock_trained, root_dir / "diagnostic-cache", source_model, 1, 1, 1, 1, 1, 1); diagnostic_resource = resource_snapshot(diagnostic_timer)
        route = base.candidate_runtime(self.config, source_model, int(self.root))
        historical = json.loads((ROOT / "evals/results/exp-698-h92-w256-backbone-factorial/metrics.json").read_text())
        historical_wall = float(historical["runtime"]["expected_formal_wall_seconds"]); historical_cpu = historical_wall * int(self.config["training"]["torch_threads"])
        evidence_replay_per_case = sum(float(profile_evaluations[str(w)]["paired_evidence_resource"]["wall_seconds"]) / 3 for w in WIDTHS)
        primary_cases = len(ARM_ORDER) * (2 * int(self.config["evaluation"]["transition_cases_per_arm_width"]) + int(self.config["evaluation"]["w128_rollout_cases_per_arm"]) + int(self.config["evaluation"]["w256_rollout_cases_per_arm"]))
        diagnostic_cases = 2 * (2 * int(self.config["evaluation"]["source_transition_cases_per_width"]) + int(self.config["evaluation"]["source_w128_rollout_cases"]) + int(self.config["evaluation"]["source_w256_rollout_cases"])) + len(ARM_ORDER) * (2 * int(self.config["evaluation"]["randomized_transition_cases_per_width"]) + int(self.config["evaluation"]["randomized_w128_rollout_cases"]) + int(self.config["evaluation"]["randomized_w256_rollout_cases"]))
        incremental = evidence_replay_per_case * (primary_cases + diagnostic_cases) + float(diagnostic_resource["wall_seconds"])
        expected = historical_wall + incremental
        projected_artifacts = 4 * 7 * self.checkpoint.stat().st_size + (primary_cases + diagnostic_cases) * 512
        budget = {"basis": "EXP698 root20261386 unchanged-route throughput plus current incremental diagnostic/orchestration profile", "route_equivalence": equivalence, "historical_exp698_expected_wall_seconds": historical_wall, "profiled_incremental": {"evaluations": profile_evaluations, "diagnostics": diagnostic_resource}, "expected_wall_seconds": expected, "expected_wall_range_seconds": [expected * .90, expected * 1.10], "expected_process_cpu_seconds": historical_cpu + incremental * int(self.config["training"]["torch_threads"]), "expected_cgroup_cpu_seconds": historical_cpu + incremental * int(self.config["training"]["torch_threads"]), "peak_rss_bytes": max(int(diagnostic_resource["peak_rss_bytes"]), int(historical["runtime"]["calibration_peak_rss_bytes"])), "expected_artifact_bytes": projected_artifacts, "soft_wall_seconds": expected * 1.25, "hard_wall_seconds": expected * 1.75, "soft_multiplier": 1.25, "hard_multiplier": 1.75}
        result = {"schema_version": "exp699-v1", "run_id": RUN_ID, "status": "completed", "mode": "calibration", "root": self.root, "decision": "admit-formal", "source": self.identity(), "source_health": {"w128_transition": [2998, 3000], "w128_rollout": [259, 300], "evidence": "evals/results/exp-609-tier6-straight-through-confirmation/metrics.json"}, "profiles": {"incremental_evidence": profile_evaluations, "diagnostics": diagnostic}, "candidate_runtime": route, "release_classification": "research-only" if route["resource"]["wall_seconds"] > 300 else "runtime-measured-not-promoted", "budget": budget, "resource": resource_snapshot(timer), "limitations": limitations()}; write_json(root_dir / "root-summary.json", result); write_json(self.output / "calibration.json", result); return result

    def require_calibration(self):
        path = self.output / "calibration" / f"root-{self.config['roots']['calibration']}" / "root-summary.json"
        if not path.is_file(): raise RuntimeError("formal requires completed calibration")
        result = json.loads(path.read_text())
        if result["decision"] != "admit-formal" or result["source"]["runner_sha256"] != sha256(Path(__file__)): raise RuntimeError("calibration identity mismatch")
        return result

    def require_predecessor_gate(self) -> None:
        if self.mode == "formal": return
        formal = self.output / "formal" / f"root-{self.config['roots']['formal']}" / "root-summary.json"
        if not formal.is_file() or not json.loads(formal.read_text()).get("confirmation_trigger"): raise RuntimeError("confirmation roots require passing frozen formal trigger")
        if self.mode == "independent":
            for root in self.config["roots"]["confirmations"]:
                path = self.output / "confirmations" / f"root-{root}" / "root-summary.json"
                if not path.is_file() or not json.loads(path.read_text()).get("confirmation_trigger"): raise RuntimeError("independent roots require both passing confirmations")

    def science(self):
        calibration = self.require_calibration(); self.require_predecessor_gate(); root_dir = self.open_root(); timer = Timer(); splits = self.splits(); records = {}; evaluations = {}; models = {}
        for arm in ARM_ORDER:
            models[arm], records[arm] = train_arm(self.config, self.exp300, self.checkpoint, splits, int(self.root), arm, int(self.config["training"]["updates"]), root_dir)
            evaluations[arm] = {"128": evaluate_with_evidence(self.config, self.exp300, models[arm], splits[128], int(self.root), 128, 3000, 300), "256": evaluate_with_evidence(self.config, self.exp300, models[arm], splits[256], int(self.root), 256, 3000, 600)}
            write_json(root_dir / "progress.json", {"completed_arms": list(records), "records": records, "evaluations": evaluations})
        work = self.work_equivalence(records)
        winner = max(ARM_ORDER, key=lambda arm: evaluations[arm]["256"]["rollout"]["exact_accuracy"])
        diagnostic = diagnostics(self.config, self.exp300, self.checkpoint, splits, int(self.root), evaluations, root_dir / "diagnostic-cache", models[winner])
        w128 = evaluations[winner]["128"]; frozen_w128 = diagnostic["frozen_source"]["evaluations"]["128"]
        healthy = w128["transition"]["exact_accuracy"] >= .99 and w128["rollout"]["exact_accuracy"] >= frozen_w128["rollout"]["exact_accuracy"] - .05 and w128["rollout"]["any_invalid_residue_rate"] <= .01
        confirm = healthy and evaluations[winner]["256"]["rollout"]["exact_accuracy"] >= .85
        if self.mode == "formal": decision = "advance-to-confirmations" if confirm else "complete-negative-full-denominator"
        elif self.mode == "confirmations": decision = "confirmation-pass" if confirm else "confirmation-failed-full-denominator"
        else: decision = "independent-pass" if confirm else "independent-failed-full-denominator"
        result = {"schema_version": "exp699-v1", "run_id": RUN_ID, "status": "completed", "mode": self.mode, "root": self.root, "decision": decision, "source": self.identity(), "records": records, "evaluations": evaluations, "winner": winner, "health": healthy, "confirmation_trigger": confirm, "diagnostics": diagnostic, "work_equivalence": work, "dynamic_budget": calibration["budget"], "resource": resource_snapshot(timer), "mlflow": None, "limitations": limitations()}; assert_terminal_contract(result); write_json(root_dir / "root-summary.json", result); write_json(self.output / "terminal-summary.json", result); return result


def log_result(experiment: Experiment, result: dict[str, Any]):
    if experiment.mode == "smoke": return result
    experiment_id = configure_mlflow(experiment.config["tracking"]["experiment_name"])
    with mlflow.start_run(run_name=f"{RUN_ID}-{experiment.mode}-{experiment.root}") as active:
        mlflow.set_tags({"canonical_run_id": RUN_ID, "mode": experiment.mode, "root": str(experiment.root), "decision": result["decision"], "source_checkpoint_sha256": experiment.config["source_checkpoint_sha256"]}); mlflow.log_dict(result, f"{experiment.mode}-result.json"); run_id = active.info.run_id
    result["mlflow"] = {"tracking_uri": tracking_uri(), "experiment_id": experiment_id, "run_id": run_id}; write_json(experiment.root_dir() / "root-summary.json", result)
    if experiment.mode != "calibration": assert_terminal_contract(result); write_json(experiment.output / "terminal-summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=CONFIG_PATH); parser.add_argument("--output", type=Path, default=OUTPUT); parser.add_argument("--mode", choices=("smoke", "calibration", "formal", "confirmations", "independent"), required=True); parser.add_argument("--root", type=int)
    args = parser.parse_args(); experiment = Experiment(args.config, args.output, args.mode, args.root); result = experiment.smoke() if args.mode == "smoke" else experiment.calibration() if args.mode == "calibration" else experiment.science(); result = log_result(experiment, result); print(json.dumps({key: result.get(key) for key in ("status", "mode", "root", "decision", "winner", "confirmation_trigger")}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
