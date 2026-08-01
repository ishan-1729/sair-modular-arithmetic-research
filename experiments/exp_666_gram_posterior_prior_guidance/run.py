"""Preregister, calibrate, and execute EXP-666 GRAM guidance."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import multiprocessing
import platform
import random
import sys
from pathlib import Path
from typing import Any

import importlib.metadata
import mlflow
import numpy as np
import psutil
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.exp_664_sprm_boundary_perturbation.run import (  # noqa: E402
    Timer,
    aggregate_resource,
    bits,
    decode,
    deterministic_makespan,
    features,
    git_output,
    json_hash,
    load_exp300,
    make_split,
    rollout_cases,
    sha256,
    transition_cases,
    write_json,
)
from experiments.exp_666_gram_posterior_prior_guidance.model import (  # noqa: E402
    ROUTES,
    TRAINED_ROUTES,
    GRAMCell,
)
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402


CONFIG_PATH = ROOT / "configs/exp_666_gram_posterior_prior_guidance.yaml"


def build_model(config: dict[str, Any], source: Path, *, load_source: bool = True) -> GRAMCell:
    model = GRAMCell(**config["model"])
    if load_source:
        model.load_source(load_file(source))
    return model


def objective(output, target: list[int], quotient: torch.Tensor, width: int, route: str, config: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
    target_bits = bits(target, width)
    supervision = [F.binary_cross_entropy_with_logits(logits, target_bits, reduction="none") for logits in output.supervision_logits]
    per_bit = supervision[-1]
    loss = float(config["training"]["deep_supervision_weight"]) * torch.stack([item.mean() for item in supervision]).mean()
    loss = loss + float(config["training"]["worst_bit_weight"]) * per_bit.mean(0).max()
    loss = loss + float(config["training"]["quotient_weight"]) * F.cross_entropy(output.quotient_logits, quotient)
    if route not in {"guidance_only_no_kl", "stochasticity_only_no_posterior", "direct_prediction"}:
        loss = loss + float(config["training"]["kl_weight"]) * output.kl.mean()
    return loss, {
        "loss": float(loss.detach()),
        "batch_exact": float((output.logits.ge(0) == target_bits.bool()).all(1).float().mean()),
        "mean_guidance_relative_norm": float(output.relative_residual.detach().mean()),
        "mean_kl": float(output.kl.detach().mean()),
        "mean_prior_scale": float(output.prior_scale.detach().mean()),
        "mean_posterior_scale": None if output.posterior_scale is None else float(output.posterior_scale.detach().mean()),
        "mean_iterations": float(output.iterations.float().mean()),
    }


def train_route(
    config: dict[str, Any],
    exp300,
    source: Path,
    splits: dict[int, dict[str, list[int]]],
    root: int,
    route: str,
    steps: int,
    output: Path,
    mode: str,
) -> tuple[GRAMCell, dict[str, Any]]:
    random.seed(root + 666); np.random.seed(root + 666); torch.manual_seed(root + 666)
    route_dir = output / mode / f"root-{root}" / route
    route_dir.mkdir(parents=True, exist_ok=True)
    checkpoint, cache_path = route_dir / "model.safetensors", route_dir / "training-result.json"
    cache_key = json_hash({
        "source": sha256(source), "root": root, "route": route, "steps": steps,
        "model": config["model"], "training": config["training"], "splits": splits,
        "dependency": config["dependency"],
    })
    if config["resource_policy"]["cache_completed_roots"] and checkpoint.is_file() and cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("cache_key") == cache_key and sha256(checkpoint) == cached["result"]["checkpoint"]["sha256"]:
            model = build_model(config, source); model.load_state_dict(load_file(checkpoint))
            return model, {**cached["result"], "cached": True}
    model = build_model(config, source)
    timer, history = Timer(), []
    examples = calls = 0
    if route == "randomized_guidance":
        model.randomized_guidance(root + 66691)
    if route in TRAINED_ROUTES:
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=float(config["training"]["minimum_learning_rate"]))
        samplers = {width: exp300.BigIntSampler(splits[width]["train"], root + int(config["data"]["training_seed_offset"]) + width) for width in config["data"]["training_widths"]}
        generator = torch.Generator().manual_seed(root + 66603)
        model.train()
        for step in range(1, steps + 1):
            width = int(config["training"]["schedule"][(step - 1) % len(config["training"]["schedule"])])
            state, multiplicand, modulus, input_bit, target, quotient = samplers[width].transitions(int(config["training"]["batch_size"]))
            tick = Timer()
            target_bits = bits(target, width)
            result = model.forward_train(features(state, multiplicand, modulus, input_bit, width), target_bits, route=route, generator=generator)
            loss, metrics = objective(result, target, quotient, width, route, config)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(trainable, float(config["training"]["gradient_clip"])))
            optimizer.step(); scheduler.step()
            examples += len(target); calls += int(result.iterations.sum()) + len(target)
            record = {"step": step, "width": width, "gradient_norm": gradient, **metrics, "timing": tick.finish()}
            history.append(record)
            if step == 1 or step % int(config["training"]["log_every_steps"]) == 0 or step == steps:
                print(json.dumps({"route": route, **{key: value for key, value in record.items() if key != "timing"}}), flush=True)
    save_file({key: value.detach().contiguous() for key, value in model.state_dict().items()}, checkpoint)
    reloaded = build_model(config, source); reloaded.load_state_dict(load_file(checkpoint))
    reload_exact = all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), reloaded.state_dict().values(), strict=True))
    timing = timer.finish()
    result_record = {
        "route": route,
        "cached": False,
        "parameters": model.parameter_count(),
        "trainable_parameters": model.trainable_parameter_count(),
        "checkpoint": {"path": str(checkpoint.relative_to(ROOT)) if checkpoint.is_relative_to(ROOT) else str(checkpoint), "sha256": sha256(checkpoint), "bytes": checkpoint.stat().st_size},
        "checkpoint_reload_exact": reload_exact,
        "training": {"updates": steps if route in TRAINED_ROUTES else 0, "examples": examples, "operator_calls": calls, "history": history, "timing": timing},
    }
    write_json(cache_path, {"cache_key": cache_key, "result": result_record})
    return model, result_record


def _route_seed(route: str, width: int) -> int:
    del route, width
    return 66624001


@torch.inference_mode()
def evaluate_transitions(model: GRAMCell, route: str, cases: dict[str, Any]) -> dict[str, Any]:
    timer, width = Timer(), int(cases["width"])
    exact = bit_exact = invalid = calls = 0; predictions: list[int] = []; residuals: list[float] = []; iterations: list[int] = []
    generator = torch.Generator().manual_seed(_route_seed(route, width))
    for start in range(0, len(cases["target"]), 128):
        stop = start + 128
        result = model(features(cases["state"][start:stop], cases["multiplicand"][start:stop], cases["modulus"][start:stop], cases["input_bit"][start:stop], width), route=route, generator=generator)
        predicted, target = result.logits.ge(0), bits(cases["target"][start:stop], width).bool(); values = decode(predicted)
        predictions.extend(values); residuals.extend(result.relative_residual.tolist()); iterations.extend(result.iterations.tolist())
        exact += int((predicted == target).all(1).sum()); bit_exact += int((predicted == target).sum()); invalid += sum(value >= prime for value, prime in zip(values, cases["modulus"][start:stop], strict=True)); calls += len(values)
    count = len(cases["target"])
    return {
        "cases": count, "exact_accuracy": exact / count, "bit_accuracy": bit_exact / (count * width), "invalid_residue_rate": invalid / count,
        "mean_relative_residual": float(np.mean(residuals)), "mean_iterations": float(np.mean(iterations)),
        "prediction_sha256": json_hash(predictions), "model_calls": calls, "timing": timer.finish(),
    }


@torch.inference_mode()
def evaluate_rollouts(model: GRAMCell, route: str, cases: dict[str, Any]) -> dict[str, Any]:
    timer, width, count = Timer(), int(cases["width"]), len(cases["left"])
    endpoint = prefix = transitions = invalid = calls = 0; final_predictions: list[int] = []; first_errors: list[int] = []; residuals: list[float] = []; iterations: list[int] = []
    generator = torch.Generator().manual_seed(_route_seed(route, width))
    for start in range(0, count, 32):
        stop = min(count, start + 32); left, right, modulus = cases["left"][start:stop], cases["right"][start:stop], cases["modulus"][start:stop]; size = len(left)
        state, teacher, all_prefix, any_invalid, first = [0] * size, [0] * size, [True] * size, [False] * size, [None] * size
        for position in range(width):
            input_bit = [(value >> (width - 1 - position)) & 1 for value in left]
            result = model(features(state, right, modulus, input_bit, width), route=route, generator=generator)
            state = decode(result.logits.ge(0)); residuals.extend(result.relative_residual.tolist()); iterations.extend(result.iterations.tolist())
            teacher = [(2 * value + bit * other) % prime for value, bit, other, prime in zip(teacher, input_bit, right, modulus, strict=True)]
            step_exact = [actual == truth for actual, truth in zip(state, teacher, strict=True)]; transitions += sum(step_exact); calls += size
            for index, correct in enumerate(step_exact):
                all_prefix[index] = all_prefix[index] and correct; any_invalid[index] = any_invalid[index] or state[index] >= modulus[index]
                if not correct and first[index] is None: first[index] = position
        expected = [(a * b) % p for a, b, p in zip(left, right, modulus, strict=True)]
        endpoint += sum(actual == truth for actual, truth in zip(state, expected, strict=True)); prefix += sum(all_prefix); invalid += sum(any_invalid); final_predictions.extend(state); first_errors.extend(value for value in first if value is not None)
    return {
        "cases": count, "exact_accuracy": endpoint / count, "all_prefix_exact_accuracy": prefix / count,
        "transition_exact_accuracy": transitions / (count * width), "any_invalid_residue_rate": invalid / count,
        "first_error_mean_position": None if not first_errors else sum(first_errors) / len(first_errors),
        "mean_relative_residual": float(np.mean(residuals)), "mean_iterations": float(np.mean(iterations)),
        "prediction_sha256": json_hash(final_predictions), "model_calls": calls, "timing": timer.finish(),
    }


def evaluator_task(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload["config"]
    torch.set_num_threads(int(config["resource_policy"]["evaluator_torch_threads_each"])); torch.use_deterministic_algorithms(True)
    timer = Timer(); model = build_model(config, Path(payload["source"])); model.load_state_dict(load_file(payload["checkpoint"])); model.eval()
    if payload.get("randomized"):
        model.randomized_guidance(int(payload["root"]) + 66691)
    result: dict[str, Any] = {}
    for split, widths in payload["selection"].items():
        result[split] = {}
        for width in widths:
            cell = payload["cases"][split][str(width)]
            result[split][str(width)] = {"transition": evaluate_transitions(model, payload["route"], cell["transition"]), "rollout": evaluate_rollouts(model, payload["route"], cell["rollout"])}
    return {"route": payload["route"], "evaluation": result, "worker_resource": timer.finish()}


@torch.no_grad()
def variational_diagnostic(model: GRAMCell, route: str, cases: dict[str, Any], seed: int) -> dict[str, Any]:
    """Compare legal training posterior with target-free prior on held-out labels."""
    width = int(cases["width"])
    batch = features(cases["state"], cases["multiplicand"], cases["modulus"], cases["input_bit"], width)
    target = bits(cases["target"], width)
    prior = model(batch, route=route, generator=torch.Generator().manual_seed(seed))
    posterior = model.forward_train(batch, target, route=route, generator=torch.Generator().manual_seed(seed))
    shuffled = model.forward_train(batch, target.flip(0), route=route, generator=torch.Generator().manual_seed(seed))
    assert posterior.posterior_mean is not None and posterior.posterior_scale is not None
    return {
        "cases": len(cases["target"]),
        "prior_mean_abs": float(prior.prior_mean.abs().mean()),
        "prior_scale_mean": float(prior.prior_scale.mean()),
        "prior_scale_min": float(prior.prior_scale.min()),
        "prior_scale_max": float(prior.prior_scale.max()),
        "posterior_mean_abs": float(posterior.posterior_mean.abs().mean()),
        "posterior_scale_mean": float(posterior.posterior_scale.mean()),
        "mean_kl": float(posterior.kl.mean()),
        "guidance_relative_norm": float(prior.relative_residual.mean()),
        "posterior_prior_state_distance": float((posterior.state - prior.state).flatten(1).norm(dim=1).mean()),
        "target_shuffle_state_distance": float((posterior.state - shuffled.state).flatten(1).norm(dim=1).mean()),
        "prior_prediction_sha256": json_hash(decode(prior.logits.ge(0))),
        "posterior_prediction_sha256": json_hash(decode(posterior.logits.ge(0))),
        "target_free_inference": True,
        "single_prior_trajectory": True,
    }


@torch.inference_mode()
def candidate_runtime(model: GRAMCell, cases: dict[str, Any], repeats: int, baseline_seconds: float = 156.0) -> dict[str, Any]:
    subset = {key: value[:8] if isinstance(value, list) else value for key, value in cases.items()}; width = int(cases["width"]); batch = features(subset["state"], subset["multiplicand"], subset["modulus"], subset["input_bit"], width)
    model.eval(); direct_timer = Timer()
    for _ in range(repeats): model(batch, route="proposal_only")
    direct = direct_timer.finish(); solver_timer = Timer()
    for _ in range(repeats): model(batch, route="gram_full", generator=torch.Generator().manual_seed(int(model.supervision_steps * 66624001)))
    solver = solver_timer.finish(); calls = repeats * len(subset["state"])
    incremental_per_call = max(0.0, float(solver["wall_seconds"]) - float(direct["wall_seconds"])) / calls
    return {"direct": direct, "solver": solver, "incremental_per_transition_seconds": incremental_per_call, "candidate_projected_wall_seconds": baseline_seconds + incremental_per_call * 100 * 128}


def merge_evaluation(target: dict[str, Any], update: dict[str, Any]) -> None:
    for split, widths in update.items(): target.setdefault(split, {}).update(widths)


class Experiment:
    def __init__(self, config_path: Path, output: Path, mode: str, root: int | None) -> None:
        self.config_path = config_path.resolve(); self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8")); self.output = output.resolve(); self.mode = mode
        expected = {"calibration": int(self.config["roots"]["calibration"]), "formal": int(self.config["roots"]["viability"]), "preregister": int(self.config["roots"]["calibration"])}[mode]
        self.root = expected if root is None else int(root)
        allowed = {expected} if mode != "formal" else {int(self.config["roots"]["viability"]), *map(int, self.config["roots"]["confirmations"])}
        if self.root not in allowed: raise ValueError(f"root {self.root} is not frozen for {mode}")
        self.source = (ROOT / self.config["source_checkpoint"]).resolve()
        if sha256(self.source) != self.config["source_checkpoint_sha256"]: raise RuntimeError("source checkpoint hash mismatch")
        if not hasattr(torch.distributions.Normal(torch.zeros(1), torch.ones(1)), "rsample"): raise RuntimeError("native reparameterized Normal unavailable")
        self.exp300 = load_exp300(); self.splits = {width: make_split(self.config, self.exp300, width, self.root) for width in self.config["data"]["widths"]}

    def preregistration(self) -> dict[str, Any]:
        files = {"config": self.config_path, "mechanism": ROOT / self.config["mechanism_map"], "model": ROOT / "experiments/exp_666_gram_posterior_prior_guidance/model.py", "runner": ROOT / "experiments/exp_666_gram_posterior_prior_guidance/run.py", "test": ROOT / "tests/test_exp_666_gram_posterior_prior_guidance.py"}
        return {
            "run_id": self.config["run_id"], "root": self.root, "mode": self.mode,
            "source": {"repository_commit": git_output("rev-parse", "HEAD"), "repository_branch": git_output("branch", "--show-current"), "checkpoint": {"path": self.config["source_checkpoint"], "sha256": sha256(self.source)}, "files": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for name, path in files.items() if path.is_file()}},
            "dependency": self.config["dependency"], "paper": self.config["paper"], "roots": self.config["roots"], "routes": self.config["routes"], "primary_route": self.config["primary_route"], "inference": self.config["inference"],
            "splits": {str(width): {name: json_hash(values) for name, values in groups.items()} for width, groups in self.splits.items()},
            "gates": self.config["gates"], "decision_order": ["train-and-cache-all", "W128-primary-exact-validity-runtime", "W128-retention-dependence-variational", "W256-primary", "W256-controls", "confirmations"],
            "prohibited": ["posterior targets at inference", "answer-dependent sampling or stopping", "multi-trajectory breadth", "programmed modular arithmetic", "validity filter or repair", "lookup", "checked/private answers"],
        }

    def frozen_cases(self, transitions: int, rollouts: int, widths: list[int] | None = None) -> dict[str, Any]:
        selected = self.config["data"]["widths"] if widths is None else widths; result = {"validation": {}, "test": {}}
        for split in result:
            for width in selected:
                offset = int(self.config["data"]["validation_seed_offset"] if split == "validation" else self.config["data"]["test_seed_offset"])
                result[split][str(width)] = {"transition": transition_cases(self.exp300, self.splits[width][split], width, self.root + offset + width, transitions), "rollout": rollout_cases(self.exp300, self.splits[width][split], width, self.root + offset + width + 500, rollouts)}
        return result

    def early_result(self, routes: dict[str, Any], stage: str, evidence: dict[str, Any], resource: dict[str, Any], reason: str, unopened: list[str], mlflow_record: dict[str, Any]) -> dict[str, Any]:
        result = {"run_id": self.config["run_id"], "mode": self.mode, "root": self.root, "status": "completed", "stage": stage, "decision": "reject-current-gram-posterior-prior-adaptation", "reason": reason, "source": self.preregistration()["source"], "routes": routes, "evidence": evidence, "resource": resource, "mlflow": mlflow_record, "unopened_surfaces": unopened, "limitations": ["Decision-safe futility left later declared surfaces unopened.", "M022.b LPRM, majority, and multi-trajectory breadth remained closed.", "Exact bigint arithmetic is confined to offline labels and evaluation.", "No checked, independent, official/private, candidate, submission, promotion, or completion surface was opened."]}
        write_json(self.output / f"root-{self.root}-summary.json", result); return result

    def run(self) -> dict[str, Any]:
        self.output.mkdir(parents=True, exist_ok=True); prereg = self.preregistration(); write_json(self.output / f"preregistration-root-{self.root}.json", prereg)
        if self.mode == "preregister": return prereg
        calibration = self.mode == "calibration"; policy = self.config["resource_policy"]
        torch.set_num_threads(int(self.config["training"]["torch_threads"])); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
        started = Timer(); experiment_id = configure_mlflow(self.config["tracking"]["experiment_name"]); routes: dict[str, Any] = {}; nested_ids: dict[str, str] = {}; workers: list[dict[str, Any]] = []
        transitions = int(self.config["calibration"]["transition_cases_per_width"] if calibration else self.config["evaluation"]["transition_cases_per_width"]); rollouts = int(self.config["calibration"]["rollout_cases_per_width"] if calibration else self.config["evaluation"]["rollout_cases_per_width"]); steps = int(self.config["calibration"]["representative_updates_per_trained_route"] if calibration else self.config["training"]["steps_per_trained_route"])
        with mlflow.start_run(run_name=f"{self.config['run_id']}-{self.mode}-{self.root}") as parent:
            parent_id = parent.info.run_id; mlflow.set_tags({"canonical_run_id": self.config["run_id"], "mode": self.mode, "root": str(self.root), "repository_commit": prereg["source"]["repository_commit"]})
            for route in ROUTES:
                with mlflow.start_run(run_name=route, nested=True) as nested:
                    nested_ids[route] = nested.info.run_id; _, record = train_route(self.config, self.exp300, self.source, self.splits, self.root, route, steps, self.output, self.mode); routes[route] = record; mlflow.log_metrics({"updates": record["training"]["updates"], "training_wall_seconds": record["training"]["timing"]["wall_seconds"]})
            mlflow_record = {"tracking_uri": tracking_uri(), "experiment_id": experiment_id, "parent_run_id": parent_id, "route_run_ids": nested_ids}
            if not calibration:
                calibration_path = self.output / "calibration.json"
                if not calibration_path.is_file() or not json.loads(calibration_path.read_text())["admission"]["admitted"]: raise RuntimeError("formal root requires admitted root-20261214 calibration")
                calibration_receipt = json.loads(calibration_path.read_text())
                if self.root in set(map(int, self.config["roots"]["confirmations"])):
                    viability = json.loads((self.output / f"root-{self.config['roots']['viability']}-summary.json").read_text())
                    if viability["decision"] != "advance-to-confirmations": raise RuntimeError("confirmation root requires complete viability pass")
                primary = self.config["primary_route"]; cases = self.frozen_cases(transitions, rollouts, [128]); payload = {"config": self.config, "source": str(self.source), "checkpoint": str(ROOT / routes[primary]["checkpoint"]["path"]), "route": primary, "root": self.root, "cases": cases, "selection": {"test": [128]}}
                output = evaluator_task(payload); workers.append(output["worker_resource"]); merge_evaluation(routes[primary], output["evaluation"])
                model = build_model(self.config, self.source); model.load_state_dict(load_file(ROOT / routes[primary]["checkpoint"]["path"])); runtime = candidate_runtime(model, cases["test"]["128"]["transition"], 12, float(policy["incumbent_candidate_wall_seconds"]))
                runtime_headroom = runtime["candidate_projected_wall_seconds"] * float(calibration_receipt["admission"]["headroom"])
                w128 = routes[primary]["test"]["128"]; first_ok = w128["rollout"]["exact_accuracy"] >= float(self.config["gates"]["minimum_w128_primary_rollout"]) and w128["transition"]["exact_accuracy"] >= float(self.config["gates"]["minimum_w128_transition"]) and w128["rollout"]["any_invalid_residue_rate"] <= float(self.config["gates"]["maximum_invalid_rollout_rate"]) and runtime["candidate_projected_wall_seconds"] < float(self.config["gates"]["maximum_candidate_inference_wall_seconds"]) and runtime_headroom <= float(self.config["gates"]["maximum_candidate_headroom_wall_seconds"])
                if not first_ok:
                    result = self.early_result(routes, "w128-primary-futility", {"w128": w128, "candidate_runtime": runtime, "candidate_headroom_wall_seconds": runtime_headroom}, aggregate_resource(started.finish(), workers, 1), "frozen W128 exact/transition/validity/runtime conjunct failed", ["W128 retention/dependence/variational diagnostics", "W128 trained controls", "W256", "roots 20261216 and 20261217", "M022.b LPRM/majority/multi-trajectory breadth"], mlflow_record); mlflow.set_tag("decision", result["decision"]); return result
                control_cases = self.frozen_cases(transitions, rollouts, [128]); payloads = []
                for route in ("proposal_only", "randomized_guidance"):
                    payloads.append({"config": self.config, "source": str(self.source), "checkpoint": str(ROOT / routes[route]["checkpoint"]["path"]), "route": route, "root": self.root, "cases": control_cases, "selection": {"test": [128]}})
                control_outputs = [evaluator_task(item) for item in payloads]
                proposal_output, randomized_output = control_outputs; workers.extend(item["worker_resource"] for item in control_outputs); merge_evaluation(routes["proposal_only"], proposal_output["evaluation"]); merge_evaluation(routes["randomized_guidance"], randomized_output["evaluation"])
                diagnostic_count = int(self.config["evaluation"]["variational_cases_w128"]); diagnostic_cases = transition_cases(self.exp300, self.splits[128]["validation"], 128, self.root + 66928, diagnostic_count)
                variational = variational_diagnostic(model, primary, diagnostic_cases, self.root + int(self.config["inference"]["fixed_seed"])); randomized = routes["randomized_guidance"]["test"]["128"]
                finite = all(math.isfinite(float(variational[key])) for key in ("prior_mean_abs", "prior_scale_mean", "posterior_mean_abs", "posterior_scale_mean", "mean_kl", "guidance_relative_norm"))
                second_ok = w128["rollout"]["exact_accuracy"] >= routes["proposal_only"]["test"]["128"]["rollout"]["exact_accuracy"] - float(self.config["gates"]["maximum_w128_loss_to_proposal"]) and w128["rollout"]["exact_accuracy"] - randomized["rollout"]["exact_accuracy"] >= float(self.config["gates"]["minimum_trained_weight_drop_w128"]) and variational["guidance_relative_norm"] >= float(self.config["gates"]["minimum_guidance_relative_norm"]) and float(self.config["gates"]["minimum_mean_kl"]) <= variational["mean_kl"] <= float(self.config["gates"]["maximum_mean_kl"]) and variational["prior_scale_min"] >= float(self.config["gates"]["minimum_prior_scale"]) and variational["prior_scale_max"] <= float(self.config["gates"]["maximum_prior_scale"]) and finite and all(record["checkpoint_reload_exact"] for record in routes.values()) and all(set(group["train"]).isdisjoint(group["validation"]) and set(group["train"]).isdisjoint(group["test"]) and set(group["validation"]).isdisjoint(group["test"]) for group in self.splits.values())
                if not second_ok:
                    result = self.early_result(routes, "w128-retention-dependence-variational-futility", {"w128": w128, "proposal": routes["proposal_only"]["test"]["128"], "randomized": randomized, "variational": variational, "candidate_runtime": runtime, "candidate_headroom_wall_seconds": runtime_headroom}, aggregate_resource(started.finish(), workers, 1), "frozen W128 retention/dependence/variational/integrity conjunct failed", ["W128 trained mechanism controls", "W256", "roots 20261216 and 20261217", "M022.b LPRM/majority/multi-trajectory breadth"], mlflow_record); mlflow.set_tag("decision", result["decision"]); return result
                mechanism_routes = ("deterministic_mean", "guidance_only_no_kl", "stochasticity_only_no_posterior", "direct_prediction")
                mechanism_outputs = [evaluator_task({"config": self.config, "source": str(self.source), "checkpoint": str(ROOT / routes[route]["checkpoint"]["path"]), "route": route, "root": self.root, "cases": control_cases, "selection": {"test": [128]}}) for route in mechanism_routes]
                for item in mechanism_outputs: workers.append(item["worker_resource"]); merge_evaluation(routes[item["route"]], item["evaluation"])
                w256_cases = self.frozen_cases(transitions, rollouts, [256]); primary256 = evaluator_task({"config": self.config, "source": str(self.source), "checkpoint": str(ROOT / routes[primary]["checkpoint"]["path"]), "route": primary, "root": self.root, "cases": w256_cases, "selection": {"test": [256]}}); workers.append(primary256["worker_resource"]); merge_evaluation(routes[primary], primary256["evaluation"])
                if routes[primary]["test"]["256"]["rollout"]["exact_accuracy"] < float(self.config["gates"]["minimum_w256_primary_rollout"]):
                    result = self.early_result(routes, "w256-primary-futility", {"w128": w128, "w256": routes[primary]["test"]["256"], "variational": variational, "candidate_runtime": runtime}, aggregate_resource(started.finish(), workers, 1), "frozen W256 primary exactness conjunct failed", ["W256 controls", "roots 20261216 and 20261217", "M022.b LPRM/majority/multi-trajectory breadth"], mlflow_record); mlflow.set_tag("decision", result["decision"]); return result
                control_outputs = [evaluator_task({"config": self.config, "source": str(self.source), "checkpoint": str(ROOT / routes[route]["checkpoint"]["path"]), "route": route, "root": self.root, "cases": w256_cases, "selection": {"test": [256]}}) for route in ("proposal_only", "stochasticity_only_no_posterior")]
                for item in control_outputs: workers.append(item["worker_resource"]); merge_evaluation(routes[item["route"]], item["evaluation"])
                primary_score = routes[primary]["test"]["256"]["rollout"]["exact_accuracy"]; proposal_score = routes["proposal_only"]["test"]["256"]["rollout"]["exact_accuracy"]; stochasticity_score = routes["stochasticity_only_no_posterior"]["test"]["256"]["rollout"]["exact_accuracy"]
                passed = primary_score - proposal_score >= float(self.config["gates"]["minimum_w256_gain_over_proposal"]) and primary_score - stochasticity_score >= float(self.config["gates"]["minimum_w256_gain_over_stochasticity"])
                decision = "advance-to-confirmations" if passed and self.root == int(self.config["roots"]["viability"]) else ("confirmed-intermediate-gram-result" if passed else "reject-current-gram-posterior-prior-adaptation")
                result = {"run_id": self.config["run_id"], "mode": self.mode, "root": self.root, "status": "completed", "decision": decision, "source": prereg["source"], "routes": routes, "diagnostics": {"variational": variational, "randomized_w128": randomized}, "candidate_runtime": runtime, "candidate_headroom_wall_seconds": runtime_headroom, "resource": aggregate_resource(started.finish(), workers, 1), "mlflow": mlflow_record, "limitations": ["Bounded single-trajectory SAIR adaptation only; not a full GRAM reproduction.", "M022.b LPRM, majority, and multi-trajectory breadth remained closed.", "No checked, independent, official/private, candidate, submission, promotion, or completion surface was opened."]}; write_json(self.output / f"root-{self.root}-summary.json", result); mlflow.set_tag("decision", decision); return result

            cases = self.frozen_cases(transitions, rollouts); payloads = [{"config": self.config, "source": str(self.source), "checkpoint": str(ROOT / routes[route]["checkpoint"]["path"]), "route": route, "root": self.root, "cases": cases, "selection": {"validation": [64, 128, 256], "test": [64, 128, 256]}} for route in ROUTES]
            context = multiprocessing.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(max_workers=int(policy["evaluator_processes"]), mp_context=context) as executor:
                outputs = [future.result() for future in [executor.submit(evaluator_task, item) for item in payloads]]
            for item in outputs: merge_evaluation(routes[item["route"]], item["evaluation"]); workers.append(item["worker_resource"])
            primary = self.config["primary_route"]; model = build_model(self.config, self.source); model.load_state_dict(load_file(ROOT / routes[primary]["checkpoint"]["path"])); model.eval()
            diag_count = int(self.config["calibration"]["variational_cases_w128"]); diag_cases = transition_cases(self.exp300, self.splits[128]["validation"], 128, self.root + 66928, diag_count); variational = variational_diagnostic(model, primary, diag_cases, self.root + int(self.config["inference"]["fixed_seed"]))
            repeat = evaluator_task({"config": self.config, "source": str(self.source), "checkpoint": str(ROOT / routes[primary]["checkpoint"]["path"]), "route": primary, "root": self.root, "cases": cases, "selection": {"test": [128]}}); workers.append(repeat["worker_resource"]); deterministic = repeat["evaluation"]["test"]["128"]["rollout"]["prediction_sha256"] == routes[primary]["test"]["128"]["rollout"]["prediction_sha256"]
            runtime = candidate_runtime(model, cases["test"]["128"]["transition"], 12, float(policy["incumbent_candidate_wall_seconds"]))
            training_wall = training_cpu = training_cgroup = 0.0; update_walls = []
            for route in TRAINED_ROUTES:
                history = routes[route]["training"]["history"]; update_walls.extend(float(item["timing"]["wall_seconds"]) for item in history)
                multiplier = int(self.config["training"]["steps_per_trained_route"]) / len(history)
                training_wall += sum(float(item["timing"]["wall_seconds"]) for item in history) * multiplier; training_cpu += sum(float(item["timing"]["process_cpu_seconds"]) for item in history) * multiplier; training_cgroup += sum(float(item["timing"]["cgroup_cpu_seconds"] or 0.0) for item in history) * multiplier
            route_wall, route_cpu, route_cgroup = {}, {}, {}
            for route in ROUTES:
                wall = cpu = cgroup = 0.0
                for width in (64, 128, 256):
                    for split, target_t, target_r in (("test", int(self.config["evaluation"]["transition_cases_per_width"]), int(self.config["evaluation"]["rollout_cases_per_width"])), ("validation", int(self.config["evaluation"]["validation_transition_cases_per_width"]), int(self.config["evaluation"]["validation_rollout_cases_per_width"]))):
                        for kind, target, observed in (("transition", target_t, transitions), ("rollout", target_r, rollouts)):
                            timing = routes[route][split][str(width)][kind]["timing"]; factor = target / observed; wall += float(timing["wall_seconds"]) * factor; cpu += float(timing["process_cpu_seconds"]) * factor; cgroup += float(timing["cgroup_cpu_seconds"] or 0.0) * factor
                route_wall[route], route_cpu[route], route_cgroup[route] = wall, cpu, cgroup
            evaluation_wall = deterministic_makespan(list(route_wall.values()), int(policy["evaluator_processes"])); evaluation_cpu = sum(route_cpu.values()); evaluation_cgroup = sum(route_cgroup.values())
            repeat_factor = int(self.config["evaluation"]["rollout_cases_per_width"]) / rollouts; repeat_timing = repeat["evaluation"]["test"]["128"]["rollout"]["timing"]
            transition_timing = routes[primary]["test"]["128"]["transition"]["timing"]; diagnostic_factor = 3 * int(self.config["evaluation"]["variational_cases_w128"]) / transitions
            diagnostics_wall = float(repeat_timing["wall_seconds"]) * repeat_factor + float(transition_timing["wall_seconds"]) * diagnostic_factor
            diagnostics_cpu = float(repeat_timing["process_cpu_seconds"]) * repeat_factor + float(transition_timing["process_cpu_seconds"]) * diagnostic_factor
            diagnostics_cgroup = float(repeat_timing["cgroup_cpu_seconds"] or 0.0) * repeat_factor + float(transition_timing["cgroup_cpu_seconds"] or 0.0) * diagnostic_factor
            expected_wall = training_wall + evaluation_wall + diagnostics_wall; expected_cpu = training_cpu + evaluation_cpu + diagnostics_cpu
            variability = max(update_walls) / max(float(np.mean(update_walls)), 1e-12); headroom = min(float(policy["variability_headroom_ceiling"]), max(float(policy["variability_headroom_floor"]), variability)); interim = started.finish(); resource_record = aggregate_resource(interim, workers, int(policy["evaluator_processes"])); observed_tree_cpu = max(float(resource_record["aggregate_process_tree_cpu_seconds"]), 1e-9); cgroup_ratio = float(resource_record["cgroup_cpu_seconds"] or observed_tree_cpu) / observed_tree_cpu; expected_cgroup = training_cgroup + evaluation_cgroup + diagnostics_cgroup; soft_multiplier, hard_multiplier = headroom * float(policy["soft_guard_multiplier"]), headroom * float(policy["hard_guard_multiplier"])
            admission = {
                "complete_routes": all(all(split in routes[route] and set(routes[route][split]) == {"64", "128", "256"} for split in ("validation", "test")) for route in ROUTES),
                "projected_training_wall_seconds": training_wall, "projected_route_wall_seconds": route_wall, "projected_evaluation_wall_seconds": evaluation_wall, "projected_diagnostics_wall_seconds": diagnostics_wall,
                "expected_wall_seconds": expected_wall, "eta_wall_seconds": expected_wall * headroom, "soft_guard_wall_seconds": math.ceil(expected_wall * soft_multiplier), "hard_guard_wall_seconds": math.ceil(expected_wall * hard_multiplier),
                "expected_process_tree_cpu_seconds": expected_cpu, "eta_process_tree_cpu_seconds": expected_cpu * headroom, "soft_guard_process_tree_cpu_seconds": math.ceil(expected_cpu * soft_multiplier), "hard_guard_process_tree_cpu_seconds": math.ceil(expected_cpu * hard_multiplier),
                "expected_cgroup_cpu_seconds": expected_cgroup, "cgroup_projection_basis": "projected-process-tree-cpu-times-observed-cgroup-to-process-tree-ratio", "observed_cgroup_to_process_tree_ratio": cgroup_ratio, "eta_cgroup_cpu_seconds": expected_cgroup * headroom, "soft_guard_cgroup_cpu_seconds": math.ceil(expected_cgroup * soft_multiplier), "hard_guard_cgroup_cpu_seconds": math.ceil(expected_cgroup * hard_multiplier),
                "variability_ratio": variability, "headroom": headroom, "observed_aggregate_process_tree_cpu_seconds": resource_record["aggregate_process_tree_cpu_seconds"], "observed_cgroup_cpu_seconds": resource_record["cgroup_cpu_seconds"], "aggregate_peak_rss_upper_bound_bytes": resource_record["aggregate_peak_rss_upper_bound_bytes"], "candidate_inference_wall_seconds": runtime["candidate_projected_wall_seconds"],
            }
            admission["candidate_inference_headroom_wall_seconds"] = runtime["candidate_projected_wall_seconds"] * headroom
            admission["checks"] = {"complete_routes": admission["complete_routes"], "expected_value": expected_wall <= float(policy["maximum_expected_formal_wall_seconds"]), "memory": resource_record["aggregate_peak_rss_upper_bound_bytes"] <= int(policy["maximum_peak_rss_bytes"]), "candidate_inference": runtime["candidate_projected_wall_seconds"] < float(self.config["gates"]["maximum_candidate_inference_wall_seconds"]) and admission["candidate_inference_headroom_wall_seconds"] <= float(self.config["gates"]["maximum_candidate_headroom_wall_seconds"]), "determinism": deterministic, "reload": all(item["checkpoint_reload_exact"] for item in routes.values()), "separation": all(set(group["train"]).isdisjoint(group["validation"]) and set(group["train"]).isdisjoint(group["test"]) and set(group["validation"]).isdisjoint(group["test"]) for group in self.splits.values()), "target_free_single_trajectory": bool(self.config["inference"]["target_free_prior_only"]) and int(self.config["inference"]["trajectories"]) == 1}; admission["admitted"] = all(admission["checks"].values())
            decision = "admit-root-20261215" if admission["admitted"] else "park-resource-compliance-or-feasibility"
            result = {"run_id": self.config["run_id"], "mode": self.mode, "root": self.root, "status": "completed", "source": prereg["source"], "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "distributions": "torch.distributions.Normal/Independent/kl_divergence", "mlflow": mlflow.__version__, "logical_cores": psutil.cpu_count(), "evaluator_processes": int(policy["evaluator_processes"]), "threads_each": int(policy["evaluator_torch_threads_each"])}, "mlflow": mlflow_record, "routes": routes, "diagnostics": {"variational": variational, "deterministic_repeat_equal": deterministic}, "candidate_runtime": runtime, "admission": admission, "decision": decision, "resource": resource_record, "limitations": ["Calibration measurements are resource and integrity evidence, not scientific accuracy evidence.", "M022.b LPRM, majority, and multi-trajectory breadth remained closed.", "Exact bigint arithmetic is confined to offline labels and evaluation.", "No checked, independent, official/private, candidate, submission, promotion, or completion surface was opened."]}; write_json(self.output / "calibration.json", result); mlflow.set_tag("decision", decision); return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=CONFIG_PATH); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--mode", choices=("preregister", "calibration", "formal"), required=True); parser.add_argument("--root", type=int); args = parser.parse_args()
    result = Experiment(args.config, args.output, args.mode, args.root).run(); print(json.dumps({"decision": result.get("decision"), "root": result.get("root"), "status": result.get("status", "preregistered")}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
