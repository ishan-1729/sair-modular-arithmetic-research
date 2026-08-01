"""Execute EXP692 C3: Candidate-derived quotient/carry posterior conditioning."""

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

RUN_ID = "exp-692-c3-quotient-carry-conditioning"
SLUG = RUN_ID
CONFIG_PATH = ROOT / "configs/exp_692_c3_quotient_carry_conditioning.yaml"
OUTPUT = ROOT / ".scratch/exp-692-c3-quotient-carry-conditioning"
TRAINED_ARMS = ("primary", "access_masked_control")
CHALLENGE_COMMIT = exp687.CHALLENGE_COMMIT

canonical_hash = exp687.canonical_hash
state_hash = exp687.state_hash
configure_torch = exp687.configure_torch
prepare_pruned = exp687.prepare_pruned
new_teacher = exp687.new_teacher
soft_features = exp687.soft_features
hard_straight_through = exp687.hard_straight_through
teacher_prefix = exp687.teacher_prefix


class QuotientCarryFilmStudent(nn.Module):
    """One-cycle h48 core conditioned only by its learned soft posteriors."""

    def __init__(self, config: dict[str, Any], core_state: dict[str, torch.Tensor] | None, seed: int = 692) -> None:
        super().__init__()
        hidden = int(config["model"]["hidden"]); embedding = int(config["model"]["posterior_embedding_dimension"])
        self.width = int(config["model"]["width"])
        self.core = SharedDilatedConvGRU(self.width, hidden, 1)
        if core_state is not None:
            self.core.load_state_dict(core_state)
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.carry_head = nn.Conv1d(hidden, int(config["model"]["signed_carry_classes"]), 1)
            self.quotient_embedding = nn.Parameter(torch.empty(int(config["model"]["quotient_classes"]), embedding))
            self.carry_embedding = nn.Parameter(torch.empty(int(config["model"]["signed_carry_classes"]), embedding))
            self.film = nn.Conv1d(2 * embedding, 2 * hidden, 1, bias=False)
            nn.init.xavier_uniform_(self.quotient_embedding); nn.init.xavier_uniform_(self.carry_embedding)
            nn.init.zeros_(self.film.weight)
        self.hidden = hidden; self.embedding = embedding

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _posteriors(self, hidden: torch.Tensor, intervention: str,
                    generator: torch.Generator | None) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = F.silu(self.core.output_norm(hidden))
        pooled = torch.cat((normalized.mean(-1), normalized.amax(-1)), dim=1)
        quotient = self.core.quotient_head(pooled).softmax(1)
        carry = self.carry_head(normalized).softmax(1)
        if intervention in ("zero_quotient", "zero_joint"):
            quotient = torch.zeros_like(quotient)
        if intervention in ("zero_carry", "zero_joint"):
            carry = torch.zeros_like(carry)
        if intervention == "permute":
            quotient = quotient.roll(1, 0); carry = carry.roll(1, 0)
        if intervention == "randomize_heads":
            quotient = torch.randn(quotient.shape, generator=generator, device=quotient.device).softmax(1)
            carry = torch.randn(carry.shape, generator=generator, device=carry.device).softmax(1)
        return quotient, carry

    def step(self, inputs: torch.Tensor, conditioning_access: bool, intervention: str = "normal",
             generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        if intervention not in {"normal", "zero_quotient", "zero_carry", "zero_joint", "permute", "randomize_heads"}:
            raise ValueError(f"unsupported posterior intervention: {intervention}")
        self.core.width = inputs.shape[-1]
        hidden = self.core.stem(inputs)
        quotient_posterior, carry_posterior = self._posteriors(hidden, intervention, generator)
        quotient_message = (quotient_posterior @ self.quotient_embedding).unsqueeze(-1).expand(-1, -1, inputs.shape[-1])
        carry_message = torch.einsum("bcw,cd->bdw", carry_posterior, self.carry_embedding)
        scale, bias = self.film(torch.cat((quotient_message, carry_message), dim=1)).chunk(2, dim=1)
        film_delta = torch.tanh(scale) * hidden + bias
        hidden = hidden + film_delta * float(conditioning_access)
        for dilation in self.core.dilation_schedule(inputs.shape[-1]):
            hidden = self.core.recurrent_update(hidden, inputs, dilation)
        normalized = F.silu(self.core.output_norm(hidden)); logits = self.core.local_head(normalized).squeeze(1)
        final_pooled = torch.cat((normalized.mean(-1), normalized.amax(-1)), dim=1)
        quotient_logits = self.core.quotient_head(final_pooled); carry_logits = self.carry_head(normalized)
        entropy = lambda value: float((-(value * value.clamp_min(1e-12).log()).sum(1)).mean().detach())
        diagnostics = {"quotient_entropy": entropy(quotient_posterior), "carry_entropy": entropy(carry_posterior),
                       "film_delta_norm": float(film_delta.detach().norm() / max(1, film_delta.numel()) ** 0.5)}
        return logits, quotient_logits, carry_logits, diagnostics

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, quotient, _, _ = self.step(inputs, True)
        return logits, quotient


def new_student(config: dict[str, Any], state: dict[str, torch.Tensor] | None, seed: int = 692) -> QuotientCarryFilmStudent:
    model = QuotientCarryFilmStudent(config, None, seed)
    if state is not None:
        if all(name.startswith("core.") or name.startswith(("carry_head", "quotient_embedding", "carry_embedding", "film")) for name in state):
            model.load_state_dict(state)
        else:
            model.core.load_state_dict(state)
    return model


def posterior_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {key: float(np.mean([row[key] for row in rows])) for key in ("quotient_entropy", "carry_entropy", "film_delta_norm")}


def offline_carry_targets(states: list[int], multiplicands: list[int], moduli: list[int], flags: list[int], width: int) -> torch.Tensor:
    """Generate exact signed-carry class labels for training/evaluation only."""
    from experiments.exp_658_joint_quotient_carry_transducer.data import LegalTransition, quotient_carry_trace
    rows = [quotient_carry_trace(LegalTransition(s, a, p, b), width).signed_carry_class_indices
            for s, a, p, b in zip(states, multiplicands, moduli, flags, strict=True)]
    return torch.tensor(rows, dtype=torch.long)


def train_update(config: dict[str, Any], exp300, student: QuotientCarryFilmStudent, teacher: SharedDilatedConvGRU,
                 optimizer: torch.optim.Optimizer, batch: tuple[list[int], list[int], list[int], list[int]],
                 width: int, conditioning_access: bool) -> dict[str, Any]:
    left, right, modulus, starts = batch; horizon = int(config["training"]["horizon"])
    teacher_state = teacher_prefix(left, right, modulus, starts, width); state_plane = bits(teacher_state, width)
    losses = []; exacts = []; residue_terms = []; quotient_terms = []; carry_terms = []; diagnostics = []
    optimizer.zero_grad(set_to_none=True); teacher.width = width
    for offset in range(horizon):
        flags = [(value >> (width - 1 - start - offset)) & 1 for value, start in zip(left, starts, strict=True)]
        raw = [2 * state + flag * other for state, flag, other in zip(teacher_state, flags, right, strict=True)]
        target = [value % prime for value, prime in zip(raw, modulus, strict=True)]
        quotient = torch.tensor([value // prime for value, prime in zip(raw, modulus, strict=True)], dtype=torch.long)
        inputs = soft_features(state_plane, right, modulus, flags, width)
        carry_target = offline_carry_targets(teacher_state, right, modulus, flags, width)
        logits, quotient_logits, carry_logits, diagnostic = student.step(inputs, conditioning_access); diagnostics.append(diagnostic)
        with torch.no_grad(): teacher_logits, teacher_quotient = teacher(inputs.detach())
        exact_loss, stats = exp300.transition_loss(logits, quotient_logits, target, quotient, width,
                                                    float(config["training"]["worst_bit_weight"]), float(config["training"]["quotient_weight"]))
        residue = F.mse_loss(logits, teacher_logits) / teacher_logits.square().mean().clamp_min(1e-8)
        quotient_kl = F.kl_div(F.log_softmax(quotient_logits, 1), F.softmax(teacher_quotient, 1), reduction="batchmean")
        carry_loss = F.cross_entropy(carry_logits, carry_target)
        losses.append(exact_loss + float(config["training"]["carry_weight"]) * carry_loss
                      + float(config["training"]["residue_distillation_weight"]) * residue
                      + float(config["training"]["quotient_distillation_weight"]) * quotient_kl)
        exacts.append(float(stats["batch_exact"])); residue_terms.append(residue); quotient_terms.append(quotient_kl); carry_terms.append(carry_loss)
        state_plane = hard_straight_through(logits); teacher_state = target
    objective = torch.stack(losses).mean(); objective.backward()
    film_gradient = 0.0 if student.film.weight.grad is None else float(student.film.weight.grad.abs().sum())
    quotient_head_gradient = sum(float(parameter.grad.abs().sum()) for parameter in student.core.quotient_head.parameters() if parameter.grad is not None)
    carry_head_gradient = sum(float(parameter.grad.abs().sum()) for parameter in student.carry_head.parameters() if parameter.grad is not None)
    gradient = float(torch.nn.utils.clip_grad_norm_(student.parameters(), float(config["training"]["gradient_clip"]))); optimizer.step()
    return {"loss": float(objective.detach()), "gradient_norm": gradient, "film_gradient": film_gradient,
            "quotient_head_gradient": quotient_head_gradient, "carry_head_gradient": carry_head_gradient, "mean_exact": float(np.mean(exacts)),
            "residue_distillation": float(torch.stack(residue_terms).mean()), "quotient_distillation": float(torch.stack(quotient_terms).mean()),
            "carry_loss": float(torch.stack(carry_terms).mean()), "posterior_diagnostics": posterior_summary(diagnostics),
            "student_calls": horizon, "teacher_calls": horizon, "posterior_calls": horizon,
            "film_calls": horizon, "auxiliary_loss_calls": horizon, "exact_loss_calls": horizon, "conditioning_access": conditioning_access}


def train_arm(config: dict[str, Any], exp300, pruned: dict[str, torch.Tensor], checkpoint: Path, splits,
              root: int, arm: str, updates: int, output: Path) -> tuple[QuotientCarryFilmStudent, dict[str, Any]]:
    random.seed(root + 692); np.random.seed(root + 692); torch.manual_seed(root + 692)
    model = new_student(config, pruned, root + 692900); initial = state_hash(model.state_dict()); timer = Timer(); history = []
    teacher = new_teacher(config, checkpoint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates, eta_min=float(config["training"]["minimum_learning_rate"]))
    samplers = {w: exp300.BigIntSampler(splits[w]["train"], root + int(config["data"]["training_seed_offset"]) + w) for w in set(config["data"]["curriculum"])}
    starts = {w: random.Random(root + int(config["data"]["training_seed_offset"]) + w + 1) for w in samplers}
    model.train(); access = bool(config["routes"][arm]["conditioning_access"])
    for update in range(1, updates + 1):
        tick = Timer(); width = int(config["data"]["curriculum"][(update - 1) % len(config["data"]["curriculum"])])
        left, right, modulus = samplers[width].pairs(int(config["training"]["batch_size"]))
        begin = [starts[width].randrange(0, width - int(config["training"]["horizon"]) + 1) for _ in left]
        stats = train_update(config, exp300, model, teacher, optimizer, (left, right, modulus, begin), width, access)
        scheduler.step(); record = {"update": update, "width": width, "learning_rate": scheduler.get_last_lr()[0], "timing": tick.finish(), **stats}; history.append(record)
        if update == 1 or update == updates or update % int(config["training"]["log_every_updates"]) == 0:
            print(json.dumps({"arm": arm, **record}), flush=True)
    model.eval(); arm_dir = output / "arms" / arm; arm_dir.mkdir(parents=True, exist_ok=True); path = arm_dir / "model.safetensors"
    save_file({key: value.detach().contiguous() for key, value in model.state_dict().items()}, path)
    reloaded = new_student(config, load_file(path), root + 692900)
    keys = ("student_calls", "teacher_calls", "posterior_calls", "film_calls", "auxiliary_loss_calls", "exact_loss_calls")
    return model, {"arm": arm, "initial_state_sha256": initial, "parameters": model.parameter_count(),
                   "training": {"updates": updates, "curriculum": list(config["data"]["curriculum"]),
                                **{key: sum(int(row[key]) for row in history) for key in keys}, "history": history, "resource": timer.finish()},
                   "checkpoint": {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size},
                   "reload_exact": all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), reloaded.state_dict().values(), strict=True))}


def decode(rows: torch.Tensor) -> list[int]:
    return [sum(int(bit) << index for index, bit in enumerate(row)) for row in rows.tolist()]


@torch.inference_mode()
def rollout(model: QuotientCarryFilmStudent, left: list[int], right: list[int], modulus: list[int], width: int,
            conditioning_access: bool = True, intervention: str = "normal") -> dict[str, Any]:
    state_bits = torch.zeros((len(left), width), dtype=torch.bool); left_bits = bits(left, width)
    invariant = torch.empty((len(left), 6, width)); invariant[:, 1] = bits(right, width).mul(2).sub(1)
    invariant[:, 2] = bits(modulus, width).mul(2).sub(1); invariant[:, 4].fill_(-1); invariant[:, 4, 0] = 1
    invariant[:, 5].fill_(-1); invariant[:, 5, -1] = 1
    teacher = [0] * len(left); first = [None] * len(left); invalid = [False] * len(left); diagnostics = []; timer = Timer()
    generator = torch.Generator().manual_seed(692123 + width + len(left))
    for position in range(width):
        invariant[:, 0] = state_bits.float().mul(2).sub(1); invariant[:, 3] = left_bits[:, width - 1 - position, None].mul(2).sub(1)
        logits, _, _, diagnostic = model.step(invariant, conditioning_access, intervention, generator); diagnostics.append(diagnostic)
        state_bits = logits.ge(0); values = decode(state_bits); flags = [int(value) for value in left_bits[:, width - 1 - position]]
        teacher = [(2 * state + flag * other) % prime for state, flag, other, prime in zip(teacher, flags, right, modulus, strict=True)]
        for index, (guess, truth, prime) in enumerate(zip(values, teacher, modulus, strict=True)):
            if guess != truth and first[index] is None: first[index] = position
            invalid[index] = invalid[index] or guess >= prime
    prediction = decode(state_bits); target = [(a * b) % p for a, b, p in zip(left, right, modulus, strict=True)]
    correct = [guess == truth for guess, truth in zip(prediction, target, strict=True)]; errors = [value for value in first if value is not None]
    return {"cases": len(left), "correct": sum(correct), "exact_accuracy": sum(correct) / max(1, len(correct)), "correct_vector": correct,
            "prediction_sha256": canonical_hash(prediction), "invalids": sum(invalid), "any_invalid_residue_rate": sum(invalid) / max(1, len(invalid)),
            "mean_first_error_position": None if not errors else float(np.mean(errors)), "posterior_diagnostics": posterior_summary(diagnostics),
            "maximum_posterior_shapes": [[len(left), 3], [len(left), 4, width]], "resource": timer.finish()}


def transition_eval(config, exp300, model, split, root, width, cases) -> dict[str, Any]:
    model.width = width; model.core.width = width; timer = Timer()
    value = exp300.evaluate_transitions(model, exp300.BigIntSampler(split["test"], root + int(config["data"]["test_seed_offset"]) + width), cases)
    value["resource"] = timer.finish(); return value


def paired_rollouts(config, exp300, models, split, root, width, cases) -> dict[str, Any]:
    sampler = exp300.BigIntSampler(split["test"], root + int(config["data"]["test_seed_offset"]) + width + 1)
    left, right, modulus = sampler.pairs(cases)
    results = {"primary": rollout(models["primary"], left, right, modulus, width, True),
               "access_masked_control": rollout(models["access_masked_control"], left, right, modulus, width, False)}
    pair = paired_counts(results["primary"]["correct_vector"], results["access_masked_control"]["correct_vector"])
    delta = [float(a) - float(b) for a, b in zip(results["primary"]["correct_vector"], results["access_masked_control"]["correct_vector"], strict=True)]
    pair["confidence_sequence"] = anytime_hoeffding_interval(delta, (1.0 - float(config["evaluation"]["confidence_level"])) / int(config["evaluation"]["holm_family_size"]))
    return {"arms": results, "paired": pair}


def candidate_inputs(root: int) -> list[tuple[str, str, str, int]]:
    source = ROOT / "sources/github/modular-arithmetic-challenge/src"; sys.path.insert(0, str(source))
    try:
        from modchallenge.config import EvalConfig
        from modchallenge.testgen.generator import generate_private_test_set
        test_set = generate_private_test_set(hashlib.sha256(f"exp692-c3-{root}".encode()).digest(), EvalConfig())
        return [(case.a, case.b, case.p, tier.tier_id) for tier in test_set.tiers for case in tier.cases]
    finally: sys.path.remove(str(source))


@torch.inference_mode()
def deployed_route(config, checkpoint: Path, root: int, limit: int | None = None) -> dict[str, Any]:
    rows = candidate_inputs(root); rows = rows if limit is None else rows[:limit]
    candidate = exp687.load_module(f"candidate005_exp692_{root}_{limit}_{random.randrange(1 << 20)}", ROOT / "submission/model.py")
    incumbent = candidate.Tier2HornerModel(); incumbent.load(str(ROOT / "submission"))
    model = new_student(config, load_file(checkpoint), root + 692900).eval(); timer = Timer()
    lower = [(a, b, p) for a, b, p, _ in rows if int(p).bit_length() <= 128]
    wide = [row for row in rows if 128 < int(row[2]).bit_length() <= 256]; upper = [row for row in rows if int(row[2]).bit_length() > 256]
    low_out = incumbent.predict_digits_batch(lower) if lower else []; mod = [int(row[2]) for row in wide]
    wide_result = rollout(model, [int(row[0]) % p for row, p in zip(wide, mod, strict=True)],
                          [int(row[1]) % p for row, p in zip(wide, mod, strict=True)], mod, 256, True) if wide else {"prediction_sha256": canonical_hash([]), "maximum_posterior_shapes": [[0, 3], [0, 4, 256]]}
    resource = timer.finish()
    return {"cases": len(rows), "complete_1100_case_route": len(rows) == 1100, "lower_incumbent_cases": len(lower), "w256_specialist_cases": len(wide),
            "above_w256_zero_fallback_cases": len(upper), "output_count": len(low_out) + len(wide) + len(upper), "lower_output_sha256": canonical_hash(low_out),
            "wide_output_sha256": wide_result["prediction_sha256"], "maximum_posterior_shapes": wide_result["maximum_posterior_shapes"],
            "labels_available_to_timed_route": False, "operand_swap_route": False, "resource": resource}


class Experiment:
    def __init__(self, config_path: Path, output: Path, mode: str, root: int | None) -> None:
        self.config_path = config_path.resolve(); self.config = yaml.safe_load(self.config_path.read_text()); self.output = output.resolve(); self.output.mkdir(parents=True, exist_ok=True)
        self.mode = mode; self.root = root; self.checkpoint = (ROOT / self.config["source_checkpoint"]).resolve()
        self.authenticate(); configure_torch(self.config); self.exp300 = exp687.load_exp300(); self.pruned, self.manifest = prepare_pruned(self.config, self.checkpoint)
        if mode != "smoke":
            allowed = {int(self.config["roots"]["calibration"])} if mode == "calibration" else {int(self.config["roots"]["formal"])} if mode == "formal" else set(map(int, self.config["roots"]["confirmations"] if mode == "confirmation" else self.config["roots"]["independent_seeds"]))
            if root is None or int(root) not in allowed: raise ValueError(f"root {root} is not frozen for {mode}")
            self.root = int(root); self.splits = {width: make_split(self.config, self.exp300, width, self.root) for width in set(self.config["data"]["curriculum"] + [128, 256])}

    def authenticate(self) -> None:
        if self.config["run_id"] != RUN_ID or not self.checkpoint.is_file() or sha256(self.checkpoint) != self.config["source_checkpoint_sha256"]: raise RuntimeError("source checkpoint mismatch")
        for name, binding in self.config["source_contract"].items():
            if sha256(ROOT / binding["path"]) != binding["sha256"]: raise RuntimeError(f"source contract mismatch: {name}")
        challenge_repo = ROOT / "sources/github/modular-arithmetic-challenge"
        if not (challenge_repo / ".git").exists():
            challenge_repo = ROOT.parents[1] / "sources/github/modular-arithmetic-challenge"
        if git_output("-C", str(challenge_repo), "rev-parse", "HEAD") != CHALLENGE_COMMIT: raise RuntimeError("challenge commit mismatch")

    def identity(self) -> dict[str, Any]:
        return {"repository_commit": git_output("rev-parse", "HEAD"), "challenge_commit": CHALLENGE_COMMIT, "config_sha256": sha256(self.config_path),
                "runner_sha256": sha256(Path(__file__)), "test_sha256": sha256(ROOT / "tests/test_exp_692_c3_quotient_carry_conditioning.py"),
                "source_checkpoint_sha256": sha256(self.checkpoint), "channel_map_sha256": canonical_hash(self.manifest["selected_source_channels_ranked"]),
                "pruned_state_sha256": self.manifest["pruned_state_sha256"]}

    def root_dir(self) -> Path: return self.output / self.mode / f"root-{self.root}"
    def open_root(self) -> Path:
        path = self.root_dir()
        if path.exists(): raise RuntimeError(f"root {self.root} already opened")
        path.mkdir(parents=True); write_json(path / "root-opened.json", {"mode": self.mode, "root": self.root, "source": self.identity()}); return path

    def train_routes(self, path: Path, updates: int):
        models = {}; records = {}
        for arm in TRAINED_ARMS:
            models[arm], records[arm] = train_arm(self.config, self.exp300, self.pruned, self.checkpoint, self.splits, int(self.root), arm, updates, path)
            write_json(path / "progress.json", {"completed_arms": list(records), "records": records})
        keys = ("updates", "curriculum", "student_calls", "teacher_calls", "posterior_calls", "film_calls", "auxiliary_loss_calls", "exact_loss_calls")
        if records["primary"]["initial_state_sha256"] != records["access_masked_control"]["initial_state_sha256"] or any(records["primary"]["training"][key] != records["access_masked_control"]["training"][key] for key in keys):
            raise RuntimeError("equal posterior/head/loss/FiLM-computation contract failed")
        return models, records

    def smoke(self) -> dict[str, Any]:
        old_root = self.root; self.root = 692000; self.splits = {width: make_split(self.config, self.exp300, width, self.root) for width in (64, 128, 256)}
        with tempfile.TemporaryDirectory(prefix="exp692-smoke-", dir=self.output) as directory:
            path = Path(directory); models, records = self.train_routes(path, 1)
            paired = paired_rollouts(self.config, self.exp300, models, self.splits[128], self.root, 128, 2)
            sample = self.exp300.BigIntSampler(self.splits[128]["test"], self.root + 9).pairs(2)
            interventions = {name: rollout(models["primary"], *sample, 128, True, name) for name in ("zero_quotient", "zero_carry", "zero_joint", "permute", "randomize_heads")}
            randomized_full = rollout(new_student(self.config, None, 692999).eval(), *sample, 128, True)
            route = deployed_route(self.config, Path(records["primary"]["checkpoint"]["path"]), self.root, 2)
            serialized = path / "terminal.json"; write_json(serialized, {"status": "completed", "paired": paired, "interventions": interventions, "randomized_full": randomized_full}); json.loads(serialized.read_text())
            result = {"schema_version": "exp692-rootless-all-mode-smoke-v1", "status": "completed", "root": None,
                      "all_modes": ["train", "reload", "evaluate", "zero-quotient", "zero-carry", "zero-joint", "permute", "randomize-heads", "randomize-full", "admit", "serialize"],
                      "both_arms_updated": all(records[arm]["training"]["updates"] == 1 for arm in TRAINED_ARMS),
                      "identical_initial_storage": records["primary"]["initial_state_sha256"] == records["access_masked_control"]["initial_state_sha256"],
                      "equal_posterior_film_work": all(records["primary"]["training"][key] == records["access_masked_control"]["training"][key] for key in ("posterior_calls", "film_calls", "auxiliary_loss_calls")),
                      "primary_film_gradient_nonzero": records["primary"]["training"]["history"][0]["film_gradient"] > 0,
                      "control_film_gradient_zero": records["access_masked_control"]["training"]["history"][0]["film_gradient"] == 0,
                      "both_heads_train_in_both_arms": all(records[arm]["training"]["history"][0][key] > 0 for arm in TRAINED_ARMS for key in ("quotient_head_gradient", "carry_head_gradient")),
                      "reload_exact": all(records[arm]["reload_exact"] for arm in TRAINED_ARMS), "route_cases": route["cases"],
                      "interventions": sorted(interventions), "terminal_roundtrip": True}
        self.root = old_root; write_json(self.output / "smoke.json", result); return result

    def calibration(self) -> dict[str, Any]:
        path = self.open_root(); timer = Timer(); models, records = self.train_routes(path, int(self.config["training"]["calibration_updates"]))
        transitions = int(self.config["calibration"]["transition_cases_per_arm_width"]); rollouts = int(self.config["calibration"]["rollout_cases_per_arm_width"])
        evals = {arm: {str(width): {"transition": transition_eval(self.config, self.exp300, model, self.splits[width], self.root, width, transitions),
                                   "rollout": paired_rollouts(self.config, self.exp300, models, self.splits[width], self.root, width, rollouts)["arms"][arm]} for width in (128, 256)} for arm, model in models.items()}
        checkpoint = Path(records["primary"]["checkpoint"]["path"]); warmup = deployed_route(self.config, checkpoint, self.root, 1)
        cold = [deployed_route(self.config, checkpoint, self.root) for _ in range(int(self.config["calibration"]["cold_repetitions"]))]
        walls = [float(item["resource"]["wall_seconds"]) for item in cold]
        if max(walls) / max(float(np.median(walls)), 1e-12) > float(self.config["calibration"]["variability_trigger"]):
            cold.extend(deployed_route(self.config, checkpoint, self.root) for _ in range(int(self.config["calibration"]["extra_cold_repetitions"]))); walls = [float(item["resource"]["wall_seconds"]) for item in cold]
        resource = timer.finish(); update_wall = sum(float(records[arm]["training"]["resource"]["wall_seconds"]) for arm in TRAINED_ARMS)
        eval_wall = sum(float(evals[arm][str(width)][kind]["resource"]["wall_seconds"]) for arm in TRAINED_ARMS for width in (128, 256) for kind in ("transition", "rollout"))
        expected_training = update_wall * int(self.config["training"]["updates"]) / int(self.config["training"]["calibration_updates"])
        expected_eval = eval_wall * ((2 * int(self.config["evaluation"]["transition_cases_per_width"]) + int(self.config["evaluation"]["w128_rollout_cases"]) + max(self.config["evaluation"]["w256_sequential_blocks"])) / (4 * transitions + 4 * rollouts))
        expected = expected_training + expected_eval; maximum = max(walls)
        admission = {"candidate_cold_wall_seconds": walls, "candidate_max_complete_wall_seconds": maximum, "candidate_guarded_wall_seconds": 2.0 * maximum,
                     "projected_training_wall_seconds": expected_training, "projected_evaluation_wall_seconds": expected_eval, "expected_wall_seconds": expected,
                     "soft_guard_wall_seconds": math.ceil(expected * float(self.config["resource_policy"]["soft_guard_multiplier"])),
                     "hard_guard_wall_seconds": math.ceil(expected * float(self.config["resource_policy"]["hard_guard_multiplier"])),
                     "measured_process_cpu_seconds": resource["process_cpu_seconds"], "measured_cgroup_cpu_seconds": resource["cgroup_cpu_seconds"],
                     "measured_peak_rss_bytes": resource["peak_rss_bytes"], "fidelity_reduced": False}
        admission["checks"] = {"complete_cold_routes": len(cold) in (3, 5) and all(item["complete_1100_case_route"] and item["output_count"] == 1100 for item in cold),
                               "candidate_runtime": 2.0 * maximum <= float(self.config["gates"]["maximum_candidate_guarded_wall_seconds"]),
                               "memory": int(resource["peak_rss_bytes"]) <= int(self.config["resource_policy"]["maximum_peak_rss_bytes"]),
                               "equal_work": records["primary"]["training"]["film_calls"] == records["access_masked_control"]["training"]["film_calls"],
                               "reload_exact": all(records[arm]["reload_exact"] for arm in TRAINED_ARMS), "deterministic_outputs": len({item["wide_output_sha256"] for item in cold}) == 1,
                               "posterior_shapes": all(item["maximum_posterior_shapes"][0][-1] == 3 and item["maximum_posterior_shapes"][1][1:] == [4, 256] for item in cold), "no_fidelity_reduction": True}
        admission["admitted"] = all(admission["checks"].values()); decision = "admit-root20261333" if admission["admitted"] else "park-exp692-at-root20261332-resource-admission"
        result = {"schema_version": "exp692-calibration-v1", "run_id": RUN_ID, "status": "completed", "decision": decision, "root": self.root,
                  "source": self.identity(), "records": records, "evaluations": evals, "candidate_runtime": {"warmup": warmup, "cold": cold},
                  "resource": resource, "admission": admission, "limitations": self.limitations(True)}
        write_json(path / "root-summary.json", result); write_json(self.output / "calibration.json", result); return self.log(result)

    def admitted(self) -> dict[str, Any]:
        path = self.output / "calibration.json"
        if not path.is_file(): raise RuntimeError("root20261332 admission required")
        value = json.loads(path.read_text())
        if value["decision"] != "admit-root20261333" or not value["admission"]["admitted"] or value["source"]["runner_sha256"] != sha256(Path(__file__)): raise RuntimeError("matching admitted root20261332 required")
        return value

    def science(self) -> dict[str, Any]:
        calibration = self.admitted(); path = self.open_root(); timer = Timer(); models, records = self.train_routes(path, int(self.config["training"]["updates"]))
        transitions = {arm: {str(width): transition_eval(self.config, self.exp300, model, self.splits[width], self.root, width, int(self.config["evaluation"]["transition_cases_per_width"])) for width in (128, 256)} for arm, model in models.items()}
        w128 = paired_rollouts(self.config, self.exp300, models, self.splits[128], self.root, 128, int(self.config["evaluation"]["w128_rollout_cases"]))
        max_cases = max(self.config["evaluation"]["w256_sequential_blocks"]); full = paired_rollouts(self.config, self.exp300, models, self.splits[256], self.root, 256, max_cases); sequential = []
        for cases in self.config["evaluation"]["w256_sequential_blocks"]:
            primary = full["arms"]["primary"]["correct_vector"][:cases]; control = full["arms"]["access_masked_control"]["correct_vector"][:cases]
            pair = paired_counts(primary, control); pair["confidence_sequence"] = anytime_hoeffding_interval([float(a) - float(b) for a, b in zip(primary, control, strict=True)], 0.0025)
            sequential.append({"cases": cases, "primary_accuracy": sum(primary) / cases, "control_accuracy": sum(control) / cases, "paired": pair})
        sampler = self.exp300.BigIntSampler(self.splits[256]["test"], self.root + int(self.config["data"]["test_seed_offset"]) + 257); left, right, modulus = sampler.pairs(max_cases)
        interventions = {name: rollout(models["primary"], left, right, modulus, 256, True, name) for name in ("zero_quotient", "zero_carry", "zero_joint", "permute", "randomize_heads")}
        randomized_full = rollout(new_student(self.config, None, self.root + 692999).eval(), left, right, modulus, 256, True)
        primary = full["arms"]["primary"]; control = full["arms"]["access_masked_control"]; gain = primary["exact_accuracy"] - control["exact_accuracy"]
        erasures = {name: gain - (value["exact_accuracy"] - control["exact_accuracy"]) for name, value in interventions.items()}
        gates = {"healthy_control": transitions["access_masked_control"]["128"]["exact_accuracy"] >= float(self.config["gates"]["minimum_control_w128_transition"]),
                 "complete_updates": all(records[arm]["training"]["updates"] == 2400 for arm in TRAINED_ARMS),
                 "primary_w256_transition": transitions["primary"]["256"]["exact_accuracy"] >= float(self.config["gates"]["minimum_primary_w256_transition"]),
                 "primary_w256_rollout": primary["exact_accuracy"] >= float(self.config["gates"]["minimum_primary_w256_rollout"]),
                 "invalid_rate": primary["any_invalid_residue_rate"] <= float(self.config["gates"]["maximum_invalid_rate"]), "paired_gain": gain >= float(self.config["gates"]["minimum_paired_gain"]),
                 "w128_retention": w128["arms"]["primary"]["exact_accuracy"] >= w128["arms"]["access_masked_control"]["exact_accuracy"] - float(self.config["gates"]["maximum_w128_loss"]),
                 "causal_erasure": gain > 0 and all(value >= float(self.config["gates"]["minimum_intervention_erasure_fraction"]) * gain for value in erasures.values()),
                 "posterior_noncollapse": primary["posterior_diagnostics"]["quotient_entropy"] > 0 and primary["posterior_diagnostics"]["carry_entropy"] > 0,
                 "equal_work": records["primary"]["training"]["film_calls"] == records["access_masked_control"]["training"]["film_calls"],
                 "reload_exact": all(records[arm]["reload_exact"] for arm in TRAINED_ARMS), "runtime_admitted": calibration["admission"]["admitted"],
                 "confirmation_threshold": primary["exact_accuracy"] >= float(self.config["gates"]["confirmation_admission_rollout"])}
        mechanism = all(value for key, value in gates.items() if key != "confirmation_threshold"); decision = "advance-to-root20261334" if mechanism and gates["confirmation_threshold"] else "reject-exp692-c3-quotient-carry-conditioning"
        result = {"schema_version": "exp692-science-v1", "run_id": RUN_ID, "status": "completed", "root": self.root, "decision": decision,
                  "source": self.identity(), "records": records, "transitions": transitions, "w128": w128, "w256": full, "sequential_w256": sequential,
                  "interventions": interventions, "randomized_full_weights": randomized_full, "intervention_erasures": erasures, "gates": gates, "resource": timer.finish(), "dynamic_budget": calibration["admission"], "limitations": self.limitations(False)}
        write_json(path / "root-summary.json", result); write_json(self.output / "terminal-summary.json", result); return self.log(result)

    def confirmation(self) -> dict[str, Any]:
        prior = int(self.config["roots"]["formal"]) if self.root == int(self.config["roots"]["confirmations"][0]) else int(self.config["roots"]["confirmations"][0])
        prior_mode = "formal" if prior == int(self.config["roots"]["formal"]) else "confirmation"; prior_path = self.output / prior_mode / f"root-{prior}" / "root-summary.json"
        if not prior_path.is_file() or json.loads(prior_path.read_text())["decision"] not in ("advance-to-root20261334", "confirmation-pass"): raise RuntimeError("confirmation predecessor gate failed")
        result = self.science(); result["decision"] = "confirmation-pass" if result["gates"]["confirmation_threshold"] and all(value for key, value in result["gates"].items() if key != "confirmation_threshold") else "reject-exp692-confirmation"
        write_json(self.root_dir() / "root-summary.json", result); write_json(self.output / "terminal-summary.json", result); return self.log(result)

    def independent(self) -> dict[str, Any]:
        predecessor = self.output / "confirmation" / f"root-{int(self.config['roots']['confirmations'][-1])}" / "root-summary.json"
        if not predecessor.is_file() or json.loads(predecessor.read_text())["decision"] != "confirmation-pass": raise RuntimeError("both confirmations required")
        seeds = list(map(int, self.config["roots"]["independent_seeds"])); index = seeds.index(int(self.root))
        if index:
            previous = self.output / "independent" / f"root-{seeds[index - 1]}" / "root-summary.json"
            if not previous.is_file() or json.loads(previous.read_text())["decision"] not in ("independent-seed-pass", "independent-suite-pass"):
                raise RuntimeError("independent predecessor gate failed")
        result = self.science(); accuracy = float(result["w256"]["arms"]["primary"]["exact_accuracy"])
        result["independent_seed_gate"] = accuracy >= float(self.config["gates"]["independent_seed_minimum"])
        accuracies = []
        for seed in seeds[:index]:
            prior = json.loads((self.output / "independent" / f"root-{seed}" / "root-summary.json").read_text())
            accuracies.append(float(prior["w256"]["arms"]["primary"]["exact_accuracy"]))
        accuracies.append(accuracy); result["independent_accuracies_through_root"] = accuracies
        result["independent_aggregate_accuracy"] = float(np.mean(accuracies))
        if not result["independent_seed_gate"]:
            result["decision"] = "reject-exp692-independent-seed"
        elif index < len(seeds) - 1:
            result["decision"] = "independent-seed-pass"
        else:
            aggregate = result["independent_aggregate_accuracy"] >= float(self.config["gates"]["independent_aggregate_minimum"])
            result["independent_aggregate_gate"] = aggregate
            result["decision"] = "independent-suite-pass" if aggregate else "reject-exp692-independent-aggregate"
        write_json(self.root_dir() / "root-summary.json", result); write_json(self.output / "terminal-summary.json", result); return self.log(result)

    def log(self, result: dict[str, Any]) -> dict[str, Any]:
        experiment_id = configure_mlflow(self.config["tracking"]["experiment_name"])
        with mlflow.start_run(run_name=f"{RUN_ID}-{self.mode}-{self.root}") as active:
            mlflow.set_tags({"canonical_run_id": RUN_ID, "mode": self.mode, "root": str(self.root), "decision": result["decision"], "candidate_registration": "false", "tournament_entrant": "C3"})
            mlflow.log_dict(result, f"{self.mode}-result.json"); run_id = active.info.run_id
        result["mlflow"] = {"tracking_uri": tracking_uri(), "experiment_id": experiment_id, "run_id": run_id}; write_json(self.root_dir() / "root-summary.json", result); return result

    @staticmethod
    def limitations(calibration: bool) -> list[str]:
        return ["Calibration accuracy is route-integrity evidence, not science." if calibration else "Fresh generated identities only; checked/public/private answers were not used.",
                "Exact arithmetic and teacher outputs exist only in offline training/evaluation; deployed inference uses learned tensor operations and hard thresholding.",
                "Quotient/carry targets exist only offline; deployed posteriors and FiLM use learned tensor operations without programmed arithmetic semantics.",
                "C3 is independently required; its result cannot cancel or weaken C4-C7.",
                "No candidate, submission, promotion, release, or private-score claim is authorized."]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=CONFIG_PATH); parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--mode", choices=("smoke", "calibration", "formal", "confirmation", "independent"), required=True); parser.add_argument("--root", type=int); args = parser.parse_args()
    experiment = Experiment(args.config, args.output, args.mode, args.root)
    result = experiment.smoke() if args.mode == "smoke" else experiment.calibration() if args.mode == "calibration" else experiment.science() if args.mode == "formal" else experiment.confirmation() if args.mode == "confirmation" else experiment.independent()
    print(json.dumps({"mode": args.mode, "root": result.get("root"), "status": result["status"], "decision": result.get("decision"), "gates": result.get("gates"), "admission": result.get("admission", {}).get("checks")}, indent=2, sort_keys=True), flush=True); return 0


if __name__ == "__main__": raise SystemExit(main())
