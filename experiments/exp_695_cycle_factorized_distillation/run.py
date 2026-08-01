"""Execute EXP695 C6: cycle-factorized hidden-state distillation."""

from __future__ import annotations

import argparse
import copy
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
import torch.nn as nn
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.council_tournament_common.statistics import anytime_hoeffding_interval, paired_counts  # noqa: E402
from experiments.exp_220_width_independent_cellular.model import SharedDilatedConvGRU  # noqa: E402
from experiments.exp_687_compact_distilled_w256_specialist import run as exp687  # noqa: E402
from experiments.exp_664_sprm_boundary_perturbation.run import Timer, bits, git_output, make_split, sha256, write_json  # noqa: E402
from tools.autoresearch_mlflow import configure_mlflow, tracking_uri  # noqa: E402

RUN_ID = "exp-695-cycle-factorized-distillation"
CONFIG_PATH = ROOT / "configs/exp_695_cycle_factorized_distillation.yaml"
OUTPUT = ROOT / ".scratch/exp-695-cycle-factorized-distillation"
TRAINED_ARMS = ("primary", "zero_coefficient_control")
CHALLENGE_COMMIT = exp687.CHALLENGE_COMMIT

canonical_hash = exp687.canonical_hash
state_hash = exp687.state_hash
configure_torch = exp687.configure_torch
prepare_pruned = exp687.prepare_pruned
soft_features = exp687.soft_features
hard_straight_through = exp687.hard_straight_through
teacher_prefix = exp687.teacher_prefix


def visit_schedule(config: dict[str, Any]) -> tuple[tuple[int, int, int], ...]:
    outward = tuple(map(int, config["model"]["outward_dilations"]))
    inward = tuple(map(int, config["model"]["inward_dilations"]))
    return tuple((0, index, dilation) for index, dilation in enumerate(outward)) + tuple(
        (1, len(inward) - 1 - index, dilation) for index, dilation in enumerate(inward)
    )


def base_model(config: dict[str, Any], state: dict[str, torch.Tensor]) -> SharedDilatedConvGRU:
    model = SharedDilatedConvGRU(int(config["model"]["width"]), int(config["model"]["hidden"]), 1)
    model.load_state_dict(state)
    return model


class SlotCore(nn.Module):
    """One physically untied recurrent core initialized by generic tensor copying."""

    def __init__(self, base: SharedDilatedConvGRU) -> None:
        super().__init__()
        self.state_norm = copy.deepcopy(base.state_norm)
        self.gate_weight = nn.Parameter(base.gate_weight.detach().clone())
        self.gate_bias = nn.Parameter(base.gate_bias.detach().clone())
        self.candidate_weight = nn.Parameter(base.candidate_weight.detach().clone())
        self.candidate_bias = nn.Parameter(base.candidate_bias.detach().clone())

    def forward(self, hidden: torch.Tensor, features: torch.Tensor, dilation: int) -> torch.Tensor:
        normalized = self.state_norm(hidden)
        gates = torch.sigmoid(F.conv1d(torch.cat((normalized, features), 1), self.gate_weight, self.gate_bias,
                                       padding=dilation, dilation=dilation))
        reset, update = gates.chunk(2, 1)
        proposed = torch.tanh(F.conv1d(torch.cat((reset * normalized, features), 1), self.candidate_weight,
                                      self.candidate_bias, padding=dilation, dilation=dilation))
        return (1 - update) * hidden + update * proposed


class UntiedCycleTeacher(nn.Module):
    """Offline h48 teacher with one independently trainable core for every visit."""

    def __init__(self, base: SharedDilatedConvGRU, schedule: tuple[tuple[int, int, int], ...]) -> None:
        super().__init__()
        self.base = copy.deepcopy(base)
        self.schedule = schedule
        self.slots = nn.ModuleList([SlotCore(base) for _ in schedule])
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def exact_copy_identity(self) -> bool:
        for slot in self.slots:
            if not torch.equal(slot.gate_weight, self.base.gate_weight): return False
            if not torch.equal(slot.candidate_weight, self.base.candidate_weight): return False
            if not torch.equal(slot.gate_bias, self.base.gate_bias): return False
            if not torch.equal(slot.candidate_bias, self.base.candidate_bias): return False
        return True

    def forward(self, features: torch.Tensor, capture: bool = False):
        hidden = self.base.stem(features); trajectory = []
        for slot, (_, _, dilation) in zip(self.slots, self.schedule, strict=True):
            hidden = slot(hidden, features, dilation); trajectory.append(hidden)
        normalized = F.silu(self.base.output_norm(hidden))
        pooled = torch.cat((normalized.mean(-1), normalized.amax(-1)), 1)
        result = (self.base.local_head(normalized).squeeze(1), self.base.quotient_head(pooled))
        return (*result, trajectory) if capture else result


class CycleFactorizedStudent(nn.Module):
    """One shared core with bounded direction x dilation x hidden FiLM coefficients."""

    INTERVENTIONS = {"normal", "zero", "direction_swap", "dilation_permutation", "mean_coefficients"}

    def __init__(self, base: SharedDilatedConvGRU, schedule: tuple[tuple[int, int, int], ...], rank: int,
                 scale: float, seed: int) -> None:
        super().__init__()
        self.base = copy.deepcopy(base); self.schedule = schedule; self.rank = int(rank); self.scale = float(scale)
        hidden = int(base.hidden); generator = torch.Generator().manual_seed(seed)
        self.direction_factors = nn.Parameter(torch.randn(2, rank, generator=generator) * 0.02)
        self.dilation_factors = nn.Parameter(torch.randn(8, rank, generator=generator) * 0.02)
        self.hidden_basis = nn.Parameter(torch.randn(rank, hidden, generator=generator) * 0.02)
        self.width = int(base.width)

    def coefficient_parameters(self) -> tuple[nn.Parameter, ...]:
        return self.direction_factors, self.dilation_factors, self.hidden_basis

    def coefficients(self, intervention: str = "normal") -> torch.Tensor:
        if intervention not in self.INTERVENTIONS: raise ValueError(f"unsupported coefficient intervention: {intervention}")
        direction = self.direction_factors
        dilation = self.dilation_factors
        if intervention == "direction_swap": direction = direction.flip(0)
        if intervention == "dilation_permutation": dilation = dilation.flip(0)
        coefficients = torch.einsum("dr,kr,rh->dkh", direction, dilation, self.hidden_basis).tanh()
        if intervention == "mean_coefficients": coefficients = coefficients.mean((0, 1), keepdim=True).expand_as(coefficients)
        if intervention == "zero": coefficients = torch.zeros_like(coefficients)
        return coefficients

    def step(self, features: torch.Tensor, coefficient_access: bool, intervention: str = "normal"):
        self.width = self.base.width = features.shape[-1]
        coefficient_tensor = self.coefficients(intervention); hidden = self.base.stem(features); trajectory = []; used = []
        access = float(coefficient_access)
        for direction, dilation_index, dilation in self.schedule:
            coefficient = coefficient_tensor[direction, dilation_index][None, :, None]
            hidden = hidden + access * self.scale * coefficient * torch.tanh(hidden)
            hidden = self.base.recurrent_update(hidden, features, dilation)
            hidden = hidden + access * self.scale * coefficient * torch.tanh(hidden)
            trajectory.append(hidden); used.append(coefficient)
        normalized = F.silu(self.base.output_norm(hidden)); pooled = torch.cat((normalized.mean(-1), normalized.amax(-1)), 1)
        diagnostics = self.coefficient_diagnostics(coefficient_tensor)
        diagnostics.update({"coefficient_calls": len(used), "trajectory_slots": len(trajectory)})
        return self.base.local_head(normalized).squeeze(1), self.base.quotient_head(pooled), trajectory, diagnostics

    @staticmethod
    def coefficient_diagnostics(values: torch.Tensor) -> dict[str, Any]:
        flat = values.detach().abs().flatten(); probabilities = flat / flat.sum().clamp_min(1e-12)
        entropy = float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
        return {"l1": float(flat.sum()), "l2": float(flat.square().sum().sqrt()), "minimum": float(values.min()),
                "maximum": float(values.max()), "mean": float(values.mean()), "std": float(values.std()), "entropy": entropy,
                "shape": list(values.shape)}

    def forward(self, features: torch.Tensor):
        logits, quotient, _, _ = self.step(features, True)
        return logits, quotient


def new_student(config: dict[str, Any], state: dict[str, torch.Tensor], seed: int,
                full_state: dict[str, torch.Tensor] | None = None) -> CycleFactorizedStudent:
    model = CycleFactorizedStudent(base_model(config, state), visit_schedule(config), int(config["model"]["factor_rank"]),
                                   float(config["model"]["coefficient_scale"]), seed)
    if full_state is not None: model.load_state_dict(full_state)
    return model


def train_teacher(config, exp300, pruned, splits, root: int, updates: int, output: Path):
    random.seed(root + 695); np.random.seed(root + 695); torch.manual_seed(root + 695)
    teacher = UntiedCycleTeacher(base_model(config, pruned), visit_schedule(config)); copied = teacher.exact_copy_identity()
    initial = state_hash(teacher.state_dict()); optimizer = torch.optim.AdamW(teacher.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    samplers = {w: exp300.BigIntSampler(splits[w]["train"], root + int(config["data"]["teacher_seed_offset"]) + w) for w in set(config["data"]["curriculum"])}
    starts = {w: random.Random(root + int(config["data"]["teacher_seed_offset"]) + w + 1) for w in samplers}; timer = Timer(); history = []
    for update in range(1, updates + 1):
        width = int(config["data"]["curriculum"][(update - 1) % 4]); left, right, modulus = samplers[width].pairs(int(config["training"]["batch_size"]))
        begin = [starts[width].randrange(0, width - int(config["training"]["horizon"]) + 1) for _ in left]
        truth = teacher_prefix(left, right, modulus, begin, width); plane = bits(truth, width); losses = []; exacts = []; optimizer.zero_grad(set_to_none=True)
        for offset in range(int(config["training"]["horizon"])):
            flags = [(value >> (width - 1 - start - offset)) & 1 for value, start in zip(left, begin, strict=True)]
            raw = [2 * state + flag * other for state, flag, other in zip(truth, flags, right, strict=True)]
            target = [value % prime for value, prime in zip(raw, modulus, strict=True)]
            quotient = torch.tensor([value // prime for value, prime in zip(raw, modulus, strict=True)])
            logits, qlogits = teacher(soft_features(plane, right, modulus, flags, width))
            loss, stats = exp300.transition_loss(logits, qlogits, target, quotient, width, float(config["training"]["worst_bit_weight"]), float(config["training"]["quotient_weight"]))
            losses.append(loss); exacts.append(float(stats["batch_exact"])); plane = hard_straight_through(logits); truth = target
        objective = torch.stack(losses).mean(); objective.backward(); torch.nn.utils.clip_grad_norm_(teacher.parameters(), float(config["training"]["gradient_clip"])); optimizer.step()
        history.append({"update": update, "width": width, "loss": float(objective.detach()), "mean_exact": float(np.mean(exacts))})
        if update == 1 or update == updates or update % int(config["training"]["log_every_updates"]) == 0: print(json.dumps({"arm": "teacher", **history[-1]}), flush=True)
    teacher.eval()
    for parameter in teacher.parameters(): parameter.requires_grad_(False)
    path = output / "teacher.safetensors"; save_file({k: v.detach().contiguous() for k, v in teacher.state_dict().items()}, path)
    return teacher, {"updates": updates, "physical_slots": len(teacher.slots), "exact_copy_before_training": copied,
                     "initial_state_sha256": initial, "history": history, "resource": timer.finish(),
                     "checkpoint": {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size},
                     "frozen_before_students": all(not p.requires_grad for p in teacher.parameters())}


def train_update(config, exp300, student, teacher, optimizer, batch, width: int, coefficient_access: bool):
    left, right, modulus, starts = batch; horizon = int(config["training"]["horizon"])
    truth = teacher_prefix(left, right, modulus, starts, width); plane = bits(truth, width)
    losses = []; exacts = []; residues = []; quotients = []; trajectories = []; diagnostics = []
    optimizer.zero_grad(set_to_none=True)
    for offset in range(horizon):
        flags = [(value >> (width - 1 - start - offset)) & 1 for value, start in zip(left, starts, strict=True)]
        raw = [2 * state + flag * other for state, flag, other in zip(truth, flags, right, strict=True)]
        target = [value % prime for value, prime in zip(raw, modulus, strict=True)]
        quotient = torch.tensor([value // prime for value, prime in zip(raw, modulus, strict=True)])
        inputs = soft_features(plane, right, modulus, flags, width)
        logits, qlogits, student_trajectory, diagnostic = student.step(inputs, coefficient_access)
        with torch.no_grad(): teacher_logits, teacher_quotient, teacher_trajectory = teacher(inputs, capture=True)
        exact, stats = exp300.transition_loss(logits, qlogits, target, quotient, width, float(config["training"]["worst_bit_weight"]), float(config["training"]["quotient_weight"]))
        residue = F.mse_loss(logits, teacher_logits) / teacher_logits.square().mean().clamp_min(1e-8)
        qdistill = F.kl_div(F.log_softmax(qlogits, 1), F.softmax(teacher_quotient, 1), reduction="batchmean")
        slot_terms = [F.mse_loss(F.normalize(student_hidden, dim=1), F.normalize(teacher_hidden, dim=1))
                      for student_hidden, teacher_hidden in zip(student_trajectory, teacher_trajectory, strict=True)]
        trajectory = torch.stack(slot_terms).mean()
        losses.append(exact + float(config["training"]["residue_distillation_weight"]) * residue
                      + float(config["training"]["quotient_distillation_weight"]) * qdistill
                      + float(config["training"]["trajectory_distillation_weight"]) * trajectory)
        exacts.append(float(stats["batch_exact"])); residues.append(residue); quotients.append(qdistill); trajectories.append(torch.stack(slot_terms).detach()); diagnostics.append(diagnostic)
        plane = hard_straight_through(logits); truth = target
    objective = torch.stack(losses).mean(); objective.backward()
    coefficient_gradient = sum(float(p.grad.abs().sum()) for p in student.coefficient_parameters() if p.grad is not None)
    gradient = float(torch.nn.utils.clip_grad_norm_(student.parameters(), float(config["training"]["gradient_clip"]))); optimizer.step()
    slot_errors = torch.stack(trajectories).mean(0)
    return {"loss": float(objective.detach()), "gradient_norm": gradient, "coefficient_gradient": coefficient_gradient,
            "mean_exact": float(np.mean(exacts)), "residue_distillation": float(torch.stack(residues).mean()),
            "quotient_distillation": float(torch.stack(quotients).mean()), "trajectory_distillation": float(slot_errors.mean()),
            "per_slot_trajectory_errors": [float(value) for value in slot_errors], "coefficient_diagnostics": diagnostics[-1],
            "student_calls": horizon, "teacher_calls": horizon, "coefficient_calls": sum(x["coefficient_calls"] for x in diagnostics),
            "trajectory_slot_calls": sum(x["trajectory_slots"] for x in diagnostics), "exact_loss_calls": horizon,
            "coefficient_access": coefficient_access}


def train_arm(config, exp300, pruned, teacher, splits, root: int, arm: str, updates: int, output: Path):
    random.seed(root + 695); np.random.seed(root + 695); torch.manual_seed(root + 695)
    model = new_student(config, pruned, root + 695900); initial = state_hash(model.state_dict()); timer = Timer(); history = []
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates, eta_min=float(config["training"]["minimum_learning_rate"]))
    samplers = {w: exp300.BigIntSampler(splits[w]["train"], root + int(config["data"]["training_seed_offset"]) + w) for w in set(config["data"]["curriculum"])}
    starts = {w: random.Random(root + int(config["data"]["training_seed_offset"]) + w + 1) for w in samplers}
    access = bool(config["routes"][arm]["hidden_coefficient_access"]); model.train()
    for update in range(1, updates + 1):
        tick = Timer(); width = int(config["data"]["curriculum"][(update - 1) % 4]); left, right, modulus = samplers[width].pairs(int(config["training"]["batch_size"]))
        begin = [starts[width].randrange(0, width - int(config["training"]["horizon"]) + 1) for _ in left]
        stats = train_update(config, exp300, model, teacher, optimizer, (left, right, modulus, begin), width, access); scheduler.step()
        record = {"update": update, "width": width, "learning_rate": scheduler.get_last_lr()[0], "timing": tick.finish(), **stats}; history.append(record)
        if update == 1 or update == updates or update % int(config["training"]["log_every_updates"]) == 0: print(json.dumps({"arm": arm, **record}), flush=True)
    model.eval(); arm_dir = output / "arms" / arm; arm_dir.mkdir(parents=True, exist_ok=True); path = arm_dir / "model.safetensors"
    save_file({k: v.detach().contiguous() for k, v in model.state_dict().items()}, path)
    reloaded = new_student(config, pruned, root + 695900, load_file(path))
    keys = ("student_calls", "teacher_calls", "coefficient_calls", "trajectory_slot_calls", "exact_loss_calls")
    return model, {"arm": arm, "initial_state_sha256": initial, "parameters": sum(p.numel() for p in model.parameters()),
                   "training": {"updates": updates, "curriculum": list(config["data"]["curriculum"]),
                                **{key: sum(int(row[key]) for row in history) for key in keys}, "history": history, "resource": timer.finish()},
                   "checkpoint": {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size},
                   "reload_exact": all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), reloaded.state_dict().values(), strict=True))}


def decode(rows: torch.Tensor) -> list[int]:
    return [sum(int(bit) << index for index, bit in enumerate(row)) for row in rows.tolist()]


@torch.inference_mode()
def rollout(model, left, right, modulus, width: int, coefficient_access: bool = True, intervention: str = "normal"):
    state_bits = torch.zeros((len(left), width), dtype=torch.bool); left_bits = bits(left, width)
    invariant = torch.empty((len(left), 6, width)); invariant[:, 1] = bits(right, width).mul(2).sub(1); invariant[:, 2] = bits(modulus, width).mul(2).sub(1)
    invariant[:, 4].fill_(-1); invariant[:, 4, 0] = 1; invariant[:, 5].fill_(-1); invariant[:, 5, -1] = 1
    truth = [0] * len(left); first = [None] * len(left); invalid = [False] * len(left); diagnostics = []; timer = Timer()
    for position in range(width):
        invariant[:, 0] = state_bits.float().mul(2).sub(1); invariant[:, 3] = left_bits[:, width - 1 - position, None].mul(2).sub(1)
        logits, _, _, diagnostic = model.step(invariant, coefficient_access, intervention); diagnostics.append(diagnostic)
        state_bits = logits.ge(0); values = decode(state_bits); flags = [int(value) for value in left_bits[:, width - 1 - position]]
        truth = [(2 * state + flag * other) % prime for state, flag, other, prime in zip(truth, flags, right, modulus, strict=True)]
        for index, (guess, expected, prime) in enumerate(zip(values, truth, modulus, strict=True)):
            if guess != expected and first[index] is None: first[index] = position
            invalid[index] = invalid[index] or guess >= prime
    predictions = decode(state_bits); targets = [(a * b) % p for a, b, p in zip(left, right, modulus, strict=True)]
    correct = [guess == expected for guess, expected in zip(predictions, targets, strict=True)]; errors = [value for value in first if value is not None]
    coefficient = model.coefficients(intervention)
    return {"cases": len(left), "correct": sum(correct), "exact_accuracy": sum(correct) / max(1, len(correct)), "correct_vector": correct,
            "prediction_sha256": canonical_hash(predictions), "invalids": sum(invalid), "any_invalid_residue_rate": sum(invalid) / max(1, len(invalid)),
            "mean_first_error_position": None if not errors else float(np.mean(errors)), "coefficient_diagnostics": model.coefficient_diagnostics(coefficient),
            "coefficient_calls": sum(row["coefficient_calls"] for row in diagnostics), "resource": timer.finish()}


def transition_eval(config, exp300, model, split, root, width, cases):
    model.width = model.base.width = width; timer = Timer()
    value = exp300.evaluate_transitions(model, exp300.BigIntSampler(split["test"], root + int(config["data"]["test_seed_offset"]) + width), cases)
    value["resource"] = timer.finish(); return value


def paired_rollouts(config, exp300, models, split, root, width, cases):
    sampler = exp300.BigIntSampler(split["test"], root + int(config["data"]["test_seed_offset"]) + width + 1); left, right, modulus = sampler.pairs(cases)
    results = {"primary": rollout(models["primary"], left, right, modulus, width, True),
               "zero_coefficient_control": rollout(models["zero_coefficient_control"], left, right, modulus, width, False)}
    pair = paired_counts(results["primary"]["correct_vector"], results["zero_coefficient_control"]["correct_vector"])
    delta = [float(a) - float(b) for a, b in zip(results["primary"]["correct_vector"], results["zero_coefficient_control"]["correct_vector"], strict=True)]
    pair["confidence_sequence"] = anytime_hoeffding_interval(delta, (1.0 - float(config["evaluation"]["confidence_level"])) / int(config["evaluation"]["holm_family_size"]))
    return {"arms": results, "paired": pair}


def candidate_inputs(root: int):
    source = ROOT / "sources/github/modular-arithmetic-challenge/src"; sys.path.insert(0, str(source))
    try:
        from modchallenge.config import EvalConfig
        from modchallenge.testgen.generator import generate_private_test_set
        test_set = generate_private_test_set(hashlib.sha256(f"exp695-c6-{root}".encode()).digest(), EvalConfig())
        return [(case.a, case.b, case.p, tier.tier_id) for tier in test_set.tiers for case in tier.cases]
    finally: sys.path.remove(str(source))


@torch.inference_mode()
def deployed_route(config, checkpoint: Path, root: int, limit: int | None = None):
    rows = candidate_inputs(root); rows = rows if limit is None else rows[:limit]
    candidate = exp687.load_module(f"candidate005_exp695_{root}_{limit}_{random.randrange(1 << 20)}", ROOT / "submission/model.py")
    incumbent = candidate.Tier2HornerModel(); incumbent.load(str(ROOT / "submission"))
    pruned, _ = prepare_pruned(config, (ROOT / config["source_checkpoint"]).resolve())
    model = new_student(config, pruned, root + 695900, load_file(checkpoint)).eval(); timer = Timer()
    lower = [(a, b, p) for a, b, p, _ in rows if int(p).bit_length() <= 128]
    wide = [row for row in rows if 128 < int(row[2]).bit_length() <= 256]; upper = [row for row in rows if int(row[2]).bit_length() > 256]
    low_out = incumbent.predict_digits_batch(lower) if lower else []; moduli = [int(row[2]) for row in wide]
    wide_result = rollout(model, [int(row[0]) % p for row, p in zip(wide, moduli, strict=True)],
                          [int(row[1]) % p for row, p in zip(wide, moduli, strict=True)], moduli, 256, True) if wide else {"prediction_sha256": canonical_hash([])}
    resource = timer.finish()
    return {"cases": len(rows), "complete_1100_case_route": len(rows) == 1100, "lower_incumbent_cases": len(lower), "w256_specialist_cases": len(wide),
            "above_w256_zero_fallback_cases": len(upper), "output_count": len(low_out) + len(wide) + len(upper), "lower_output_sha256": canonical_hash(low_out),
            "wide_output_sha256": wide_result["prediction_sha256"], "labels_available_to_timed_route": False,
            "teacher_present_in_timed_route": False, "operand_swap_route": False, "resource": resource}


class Experiment:
    def __init__(self, config_path: Path, output: Path, mode: str, root: int | None) -> None:
        self.config_path = config_path.resolve(); self.config = yaml.safe_load(self.config_path.read_text()); self.output = output.resolve(); self.output.mkdir(parents=True, exist_ok=True)
        self.mode = mode; self.root = root; self.checkpoint = (ROOT / self.config["source_checkpoint"]).resolve()
        self.authenticate(); configure_torch(self.config); self.exp300 = exp687.load_exp300(); self.pruned, self.manifest = prepare_pruned(self.config, self.checkpoint)
        if mode != "smoke":
            allowed = {int(self.config["roots"]["calibration"])} if mode == "calibration" else {int(self.config["roots"]["formal"])} if mode == "formal" else set(map(int, self.config["roots"]["confirmations"] if mode == "confirmation" else self.config["roots"]["independent_seeds"]))
            if root is None or int(root) not in allowed: raise ValueError(f"root {root} is not frozen for {mode}")
            self.root = int(root); self.splits = {width: make_split(self.config, self.exp300, width, self.root) for width in (64, 128, 256)}

    def authenticate(self) -> None:
        if self.config["run_id"] != RUN_ID or not self.checkpoint.is_file() or sha256(self.checkpoint) != self.config["source_checkpoint_sha256"]: raise RuntimeError("source checkpoint mismatch")
        for name, binding in self.config["source_contract"].items():
            if sha256(ROOT / binding["path"]) != binding["sha256"]: raise RuntimeError(f"source contract mismatch: {name}")
        challenge_repo = ROOT / "sources/github/modular-arithmetic-challenge"
        if not (challenge_repo / ".git").exists(): challenge_repo = ROOT.parents[1] / "sources/github/modular-arithmetic-challenge"
        if git_output("-C", str(challenge_repo), "rev-parse", "HEAD") != CHALLENGE_COMMIT: raise RuntimeError("challenge commit mismatch")

    def identity(self):
        return {"repository_commit": git_output("rev-parse", "HEAD"), "challenge_commit": CHALLENGE_COMMIT, "config_sha256": sha256(self.config_path),
                "runner_sha256": sha256(Path(__file__)), "test_sha256": sha256(ROOT / "tests/test_exp_695_cycle_factorized_distillation.py"),
                "source_checkpoint_sha256": sha256(self.checkpoint), "channel_map_sha256": canonical_hash(self.manifest["selected_source_channels_ranked"]),
                "pruned_state_sha256": self.manifest["pruned_state_sha256"]}

    def root_dir(self): return self.output / self.mode / f"root-{self.root}"

    def open_root(self):
        path = self.root_dir()
        if path.exists(): raise RuntimeError(f"root {self.root} already opened")
        path.mkdir(parents=True); write_json(path / "root-opened.json", {"mode": self.mode, "root": self.root, "source": self.identity()}); return path

    def train_routes(self, path: Path, updates: int, teacher_updates: int | None = None):
        teacher, teacher_record = train_teacher(self.config, self.exp300, self.pruned, self.splits, int(self.root),
                                                int(self.config["teacher"]["updates"] if teacher_updates is None else teacher_updates), path)
        models = {}; records = {}
        for arm in TRAINED_ARMS:
            models[arm], records[arm] = train_arm(self.config, self.exp300, self.pruned, teacher, self.splits, int(self.root), arm, updates, path)
            write_json(path / "progress.json", {"teacher": teacher_record, "completed_arms": list(records), "records": records})
        keys = ("updates", "curriculum", "student_calls", "teacher_calls", "coefficient_calls", "trajectory_slot_calls", "exact_loss_calls")
        if records["primary"]["initial_state_sha256"] != records["zero_coefficient_control"]["initial_state_sha256"] or any(
                records["primary"]["training"][key] != records["zero_coefficient_control"]["training"][key] for key in keys):
            raise RuntimeError("matched teacher/trajectory/coefficient work contract failed")
        return models, records, teacher_record

    def smoke(self):
        old_root = self.root; self.root = 695000; self.splits = {width: make_split(self.config, self.exp300, width, self.root) for width in (64, 128, 256)}
        with tempfile.TemporaryDirectory(prefix="exp695-smoke-", dir=self.output) as directory:
            path = Path(directory); models, records, teacher = self.train_routes(path, 1, 1)
            paired = paired_rollouts(self.config, self.exp300, models, self.splits[128], self.root, 128, 2)
            sample = self.exp300.BigIntSampler(self.splits[128]["test"], self.root + 9).pairs(2)
            interventions = {name: rollout(models["primary"], *sample, 128, True, name) for name in ("zero", "direction_swap", "dilation_permutation", "mean_coefficients")}
            route = deployed_route(self.config, Path(records["primary"]["checkpoint"]["path"]), self.root, 2)
            serialized = path / "terminal.json"; write_json(serialized, {"status": "completed", "paired": paired, "interventions": interventions}); json.loads(serialized.read_text())
            result = {"schema_version": "exp695-rootless-all-mode-smoke-v1", "status": "completed", "root": None,
                      "all_modes": ["teacher", "both-arms", "trajectory", "reload", "evaluate", "zero", "direction-swap", "dilation-permutation", "mean", "deployed-route", "serialize"],
                      "teacher_updated": teacher["updates"] == 1, "teacher_slots": teacher["physical_slots"], "teacher_frozen": teacher["frozen_before_students"],
                      "both_arms_updated": all(records[arm]["training"]["updates"] == 1 for arm in TRAINED_ARMS),
                      "identical_initial_state": records["primary"]["initial_state_sha256"] == records["zero_coefficient_control"]["initial_state_sha256"],
                      "equal_work": all(records["primary"]["training"][key] == records["zero_coefficient_control"]["training"][key] for key in ("teacher_calls", "coefficient_calls", "trajectory_slot_calls")),
                      "reload_exact": all(records[arm]["reload_exact"] for arm in TRAINED_ARMS), "route_cases": route["cases"],
                      "teacher_absent_from_route": not route["teacher_present_in_timed_route"], "interventions": sorted(interventions), "terminal_roundtrip": True}
        self.root = old_root; write_json(self.output / "smoke.json", result); return result

    def calibration(self):
        path = self.open_root(); timer = Timer(); models, records, teacher = self.train_routes(path, int(self.config["training"]["calibration_updates"]))
        transitions = int(self.config["calibration"]["transition_cases_per_arm_width"]); rollouts = int(self.config["calibration"]["rollout_cases_per_arm_width"])
        evaluations = {arm: {str(width): {"transition": transition_eval(self.config, self.exp300, model, self.splits[width], self.root, width, transitions),
                                          "rollout": paired_rollouts(self.config, self.exp300, models, self.splits[width], self.root, width, rollouts)["arms"][arm]} for width in (128, 256)} for arm, model in models.items()}
        checkpoint = Path(records["primary"]["checkpoint"]["path"]); warmup = deployed_route(self.config, checkpoint, self.root, 1)
        cold = [deployed_route(self.config, checkpoint, self.root) for _ in range(int(self.config["calibration"]["cold_repetitions"]))]
        walls = [float(item["resource"]["wall_seconds"]) for item in cold]
        if max(walls) / max(float(np.median(walls)), 1e-12) > float(self.config["calibration"]["variability_trigger"]):
            cold.extend(deployed_route(self.config, checkpoint, self.root) for _ in range(int(self.config["calibration"]["extra_cold_repetitions"]))); walls = [float(item["resource"]["wall_seconds"]) for item in cold]
        resource = timer.finish(); update_wall = sum(float(records[arm]["training"]["resource"]["wall_seconds"]) for arm in TRAINED_ARMS)
        eval_wall = sum(float(evaluations[arm][str(width)][kind]["resource"]["wall_seconds"]) for arm in TRAINED_ARMS for width in (128, 256) for kind in ("transition", "rollout"))
        expected_teacher = float(teacher["resource"]["wall_seconds"]); expected_training = update_wall * int(self.config["training"]["updates"]) / int(self.config["training"]["calibration_updates"])
        expected_eval = eval_wall * ((2 * int(self.config["evaluation"]["transition_cases_per_width"]) + int(self.config["evaluation"]["w128_rollout_cases"]) + max(self.config["evaluation"]["w256_sequential_blocks"])) / (4 * transitions + 4 * rollouts))
        expected = expected_teacher + expected_training + expected_eval; maximum = max(walls)
        admission = {"candidate_cold_wall_seconds": walls, "candidate_max_complete_wall_seconds": maximum, "candidate_guarded_wall_seconds": 2.0 * maximum,
                     "teacher_wall_seconds": expected_teacher, "projected_training_wall_seconds": expected_training, "projected_evaluation_wall_seconds": expected_eval,
                     "expected_wall_seconds": expected, "soft_guard_wall_seconds": math.ceil(expected * 1.25), "hard_guard_wall_seconds": math.ceil(expected * 1.75),
                     "measured_process_cpu_seconds": resource["process_cpu_seconds"], "measured_cgroup_cpu_seconds": resource["cgroup_cpu_seconds"],
                     "measured_peak_rss_bytes": resource["peak_rss_bytes"], "fidelity_reduced": False}
        admission["checks"] = {"complete_cold_routes": len(cold) in (3, 5) and all(item["complete_1100_case_route"] and item["output_count"] == 1100 for item in cold),
                               "candidate_runtime": 2.0 * maximum <= float(self.config["gates"]["maximum_candidate_guarded_wall_seconds"]),
                               "memory": int(resource["peak_rss_bytes"]) <= int(self.config["resource_policy"]["maximum_peak_rss_bytes"]),
                               "equal_work": records["primary"]["training"]["coefficient_calls"] == records["zero_coefficient_control"]["training"]["coefficient_calls"],
                               "reload_exact": all(records[arm]["reload_exact"] for arm in TRAINED_ARMS),
                               "deterministic_outputs": len({item["wide_output_sha256"] for item in cold}) == 1,
                               "teacher_complete": teacher["updates"] == 400, "no_fidelity_reduction": True}
        admission["admitted"] = all(admission["checks"].values()); decision = "admit-root20261378" if admission["admitted"] else "park-exp695-at-root20261377-resource-admission"
        result = {"schema_version": "exp695-calibration-v1", "run_id": RUN_ID, "status": "completed", "decision": decision, "root": self.root,
                  "source": self.identity(), "teacher": teacher, "records": records, "evaluations": evaluations,
                  "candidate_runtime": {"warmup": warmup, "cold": cold}, "resource": resource, "admission": admission, "limitations": self.limitations(True)}
        write_json(path / "root-summary.json", result); write_json(self.output / "calibration.json", result); return self.log(result)

    def admitted(self):
        path = self.output / "calibration.json"
        if not path.is_file(): raise RuntimeError("root20261377 admission required")
        value = json.loads(path.read_text())
        if value["decision"] != "admit-root20261378" or not value["admission"]["admitted"] or value["source"]["runner_sha256"] != sha256(Path(__file__)):
            raise RuntimeError("matching admitted root20261377 required")
        return value

    def science(self):
        calibration = self.admitted(); path = self.open_root(); timer = Timer(); models, records, teacher = self.train_routes(path, int(self.config["training"]["updates"]))
        transitions = {arm: {str(width): transition_eval(self.config, self.exp300, model, self.splits[width], self.root, width, int(self.config["evaluation"]["transition_cases_per_width"])) for width in (128, 256)} for arm, model in models.items()}
        w128 = paired_rollouts(self.config, self.exp300, models, self.splits[128], self.root, 128, int(self.config["evaluation"]["w128_rollout_cases"]))
        max_cases = max(self.config["evaluation"]["w256_sequential_blocks"]); full = paired_rollouts(self.config, self.exp300, models, self.splits[256], self.root, 256, max_cases); sequential = []
        for cases in self.config["evaluation"]["w256_sequential_blocks"]:
            primary = full["arms"]["primary"]["correct_vector"][:cases]; control = full["arms"]["zero_coefficient_control"]["correct_vector"][:cases]
            pair = paired_counts(primary, control); pair["confidence_sequence"] = anytime_hoeffding_interval([float(a) - float(b) for a, b in zip(primary, control, strict=True)], 0.0025)
            sequential.append({"cases": cases, "primary_accuracy": sum(primary) / cases, "control_accuracy": sum(control) / cases, "paired": pair})
        sampler = self.exp300.BigIntSampler(self.splits[256]["test"], self.root + int(self.config["data"]["test_seed_offset"]) + 257); left, right, modulus = sampler.pairs(max_cases)
        interventions = {name: rollout(models["primary"], left, right, modulus, 256, True, name) for name in ("zero", "direction_swap", "dilation_permutation", "mean_coefficients")}
        primary = full["arms"]["primary"]; control = full["arms"]["zero_coefficient_control"]; gain = primary["exact_accuracy"] - control["exact_accuracy"]
        erasures = {name: gain - (value["exact_accuracy"] - control["exact_accuracy"]) for name, value in interventions.items()}
        gates = {"healthy_control": transitions["zero_coefficient_control"]["128"]["exact_accuracy"] >= float(self.config["gates"]["minimum_control_w128_transition"]),
                 "complete_updates": all(records[arm]["training"]["updates"] == 2400 for arm in TRAINED_ARMS),
                 "primary_w256_transition": transitions["primary"]["256"]["exact_accuracy"] >= float(self.config["gates"]["minimum_primary_w256_transition"]),
                 "primary_w256_rollout": primary["exact_accuracy"] >= float(self.config["gates"]["minimum_primary_w256_rollout"]),
                 "invalid_rate": primary["any_invalid_residue_rate"] <= float(self.config["gates"]["maximum_invalid_rate"]),
                 "paired_gain": gain >= float(self.config["gates"]["minimum_paired_gain"]),
                 "w128_retention": w128["arms"]["primary"]["exact_accuracy"] >= w128["arms"]["zero_coefficient_control"]["exact_accuracy"] - float(self.config["gates"]["maximum_w128_loss"]),
                 "causal_erasure": gain > 0 and all(value >= float(self.config["gates"]["minimum_intervention_erasure_fraction"]) * gain for value in erasures.values()),
                 "coefficient_noncollapse": primary["coefficient_diagnostics"]["l2"] > 0,
                 "equal_work": records["primary"]["training"]["coefficient_calls"] == records["zero_coefficient_control"]["training"]["coefficient_calls"],
                 "teacher_complete": teacher["updates"] == 400, "reload_exact": all(records[arm]["reload_exact"] for arm in TRAINED_ARMS),
                 "runtime_admitted": calibration["admission"]["admitted"], "confirmation_threshold": primary["exact_accuracy"] >= float(self.config["gates"]["confirmation_admission_rollout"])}
        comparison_interpretable = gates["healthy_control"]
        common = all(value for key, value in gates.items() if key != "confirmation_threshold")
        decision = "advance-to-root20261379" if comparison_interpretable and common and gates["confirmation_threshold"] else (
            "non-comparative-control-health-failure" if not comparison_interpretable else "reject-exp695-cycle-factorized-distillation")
        result = {"schema_version": "exp695-science-v1", "run_id": RUN_ID, "status": "completed", "root": self.root, "decision": decision,
                  "source": self.identity(), "teacher": teacher, "records": records, "transitions": transitions, "w128": w128, "w256": full,
                  "sequential_w256": sequential, "interventions": interventions, "intervention_erasures": erasures, "gates": gates,
                  "comparison_interpretable": comparison_interpretable, "resource": timer.finish(), "dynamic_budget": calibration["admission"], "limitations": self.limitations(False)}
        write_json(path / "root-summary.json", result); write_json(self.output / "terminal-summary.json", result); return self.log(result)

    def confirmation(self):
        roots = list(map(int, self.config["roots"]["confirmations"])); prior = int(self.config["roots"]["formal"]) if self.root == roots[0] else roots[0]
        prior_mode = "formal" if prior == int(self.config["roots"]["formal"]) else "confirmation"; prior_path = self.output / prior_mode / f"root-{prior}" / "root-summary.json"
        allowed = ("advance-to-root20261379",) if self.root == roots[0] else ("confirmation-pass",)
        if not prior_path.is_file() or json.loads(prior_path.read_text())["decision"] not in allowed: raise RuntimeError("confirmation predecessor gate failed")
        result = self.science(); result["decision"] = "confirmation-pass" if result["comparison_interpretable"] and result["gates"]["confirmation_threshold"] and all(value for key, value in result["gates"].items() if key != "confirmation_threshold") else "reject-exp695-confirmation"
        write_json(self.root_dir() / "root-summary.json", result); write_json(self.output / "terminal-summary.json", result); return self.log(result)

    def independent(self):
        confirmations = list(map(int, self.config["roots"]["confirmations"])); predecessor = self.output / "confirmation" / f"root-{confirmations[-1]}" / "root-summary.json"
        if not predecessor.is_file() or json.loads(predecessor.read_text())["decision"] != "confirmation-pass": raise RuntimeError("both confirmations required")
        seeds = list(map(int, self.config["roots"]["independent_seeds"])); index = seeds.index(int(self.root))
        if index:
            previous = self.output / "independent" / f"root-{seeds[index - 1]}" / "root-summary.json"
            if not previous.is_file() or json.loads(previous.read_text())["decision"] not in ("independent-seed-pass", "independent-suite-pass"): raise RuntimeError("independent predecessor gate failed")
        result = self.science(); accuracy = float(result["w256"]["arms"]["primary"]["exact_accuracy"]); result["independent_seed_gate"] = accuracy >= float(self.config["gates"]["independent_seed_minimum"])
        accuracies = [float(json.loads((self.output / "independent" / f"root-{seed}" / "root-summary.json").read_text())["w256"]["arms"]["primary"]["exact_accuracy"]) for seed in seeds[:index]] + [accuracy]
        result["independent_accuracies_through_root"] = accuracies; result["independent_aggregate_accuracy"] = float(np.mean(accuracies))
        if not result["independent_seed_gate"]: result["decision"] = "reject-exp695-independent-seed"
        elif index < len(seeds) - 1: result["decision"] = "independent-seed-pass"
        else:
            result["independent_aggregate_gate"] = result["independent_aggregate_accuracy"] >= float(self.config["gates"]["independent_aggregate_minimum"])
            result["decision"] = "independent-suite-pass" if result["independent_aggregate_gate"] else "reject-exp695-independent-aggregate"
        write_json(self.root_dir() / "root-summary.json", result); write_json(self.output / "terminal-summary.json", result); return self.log(result)

    def log(self, result):
        experiment_id = configure_mlflow(self.config["tracking"]["experiment_name"])
        with mlflow.start_run(run_name=f"{RUN_ID}-{self.mode}-{self.root}") as active:
            mlflow.set_tags({"canonical_run_id": RUN_ID, "mode": self.mode, "root": str(self.root), "decision": result["decision"], "candidate_registration": "false", "tournament_entrant": "C6"})
            mlflow.log_dict(result, f"{self.mode}-result.json"); run_id = active.info.run_id
        result["mlflow"] = {"tracking_uri": tracking_uri(), "experiment_id": experiment_id, "run_id": run_id}; write_json(self.root_dir() / "root-summary.json", result); return result

    @staticmethod
    def limitations(calibration: bool):
        return ["Calibration accuracy is route-integrity evidence, not science." if calibration else "Fresh generated identities only; checked/public/private answers were not used.",
                "Exact arithmetic and teacher trajectories exist only in offline training/evaluation; deployed inference uses input tensors, learned weights, and hard thresholding.",
                "The frozen 16-slot teacher never enters inference; deployment uses one shared core and learned hidden-state coefficients.",
                "Native PyTorch was selected because its deterministic Conv1d, ModuleList, einsum, optimizer, and serialization cover the mechanism without a new dependency.",
                "C5 remains scientifically unresolved; C7 and the final seven-member adversarial chair remain mandatory.",
                "No candidate, submission, promotion, release, or private-score claim is authorized."]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=CONFIG_PATH); parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--mode", choices=("smoke", "calibration", "formal", "confirmation", "independent"), required=True); parser.add_argument("--root", type=int); args = parser.parse_args()
    experiment = Experiment(args.config, args.output, args.mode, args.root)
    result = experiment.smoke() if args.mode == "smoke" else experiment.calibration() if args.mode == "calibration" else experiment.science() if args.mode == "formal" else experiment.confirmation() if args.mode == "confirmation" else experiment.independent()
    print(json.dumps({"mode": args.mode, "root": result.get("root"), "status": result["status"], "decision": result.get("decision"), "gates": result.get("gates"), "admission": result.get("admission", {}).get("checks")}, indent=2, sort_keys=True), flush=True); return 0


if __name__ == "__main__": raise SystemExit(main())
