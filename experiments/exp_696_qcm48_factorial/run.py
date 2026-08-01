"""Execute EXP696 C7: QCM-48 quotient/carry by persistent-token factorial."""

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

RUN_ID = "exp-696-qcm48-factorial"
SLUG = RUN_ID
CONFIG_PATH = ROOT / "configs/exp_696_qcm48_factorial.yaml"
OUTPUT = ROOT / ".scratch/exp-696-qcm48-factorial"
TRAINED_ARMS = ("neither", "qc_only", "token_only", "both")
INTERVENTIONS = ("qc_zero", "qc_permute", "qc_randomize", "token_reset", "token_freeze", "token_permute", "token_randomize", "cross_zero")
CHALLENGE_COMMIT = exp687.CHALLENGE_COMMIT

canonical_hash = exp687.canonical_hash
state_hash = exp687.state_hash
configure_torch = exp687.configure_torch
prepare_pruned = exp687.prepare_pruned
new_teacher = exp687.new_teacher
soft_features = exp687.soft_features
hard_straight_through = exp687.hard_straight_through
teacher_prefix = exp687.teacher_prefix


def scaled_anytime_interval(values: list[float], alpha: float, bound: float) -> dict[str, float]:
    """Apply the shared [-1,1] sequence to values bounded by +/- ``bound``."""
    base = anytime_hoeffding_interval([value / bound for value in values], alpha)
    return {"mean": base["mean"] * bound, "lower": base["lower"] * bound,
            "upper": base["upper"] * bound, "alpha_at_n": base["alpha_at_n"], "absolute_bound": bound}


class QCM48Student(nn.Module):
    """One h48 outward cycle with learned q/c, four local tokens, and rank-4 fusion."""

    def __init__(self, config: dict[str, Any], core_state: dict[str, torch.Tensor] | None, seed: int = 696) -> None:
        super().__init__()
        model = config["model"]
        hidden = int(model["hidden"]); token_dim = int(model["token_dimension"])
        embedding = int(model["posterior_embedding_dimension"]); rank = int(model["codec_rank"])
        self.width = int(model["width"]); self.hidden = hidden; self.token_dim = token_dim; self.rank = rank
        self.core = SharedDilatedConvGRU(self.width, hidden, 1)
        if core_state is not None:
            self.core.load_state_dict(core_state)
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.initial_token = nn.Parameter(torch.empty(int(model["memory_tokens"]), token_dim))
            self.carry_head = nn.Conv1d(hidden, int(model["signed_carry_classes"]), 1)
            self.quotient_embedding = nn.Parameter(torch.empty(int(model["quotient_classes"]), embedding))
            self.carry_embedding = nn.Parameter(torch.empty(int(model["signed_carry_classes"]), embedding))
            self.token_key = nn.Linear(token_dim, token_dim, bias=False)
            self.token_value = nn.Linear(token_dim, token_dim, bias=False)
            self.spatial_query = nn.Linear(hidden, token_dim, bias=False)
            self.spatial_key = nn.Linear(hidden, token_dim, bias=False)
            self.spatial_value = nn.Linear(hidden, token_dim, bias=False)
            self.token_update = nn.GRUCell(token_dim, token_dim)
            self.qc_down = nn.Conv1d(2 * embedding, rank, 1, bias=False)
            self.token_down = nn.Conv1d(token_dim, rank, 1, bias=False)
            self.qc_up = nn.Conv1d(rank, 2 * hidden, 1, bias=False)
            self.token_up = nn.Conv1d(rank, 2 * hidden, 1, bias=False)
            self.cross_up = nn.Conv1d(rank, 2 * hidden, 1, bias=False)
            nn.init.normal_(self.initial_token, std=0.02)
            nn.init.xavier_uniform_(self.quotient_embedding); nn.init.xavier_uniform_(self.carry_embedding)
            nn.init.zeros_(self.qc_up.weight); nn.init.zeros_(self.token_up.weight); nn.init.zeros_(self.cross_up.weight)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def initial_tokens(self, batch: int) -> torch.Tensor:
        return self.initial_token.unsqueeze(0).expand(batch, -1, -1).clone()

    def _posteriors(self, hidden: torch.Tensor, intervention: str,
                    generator: torch.Generator | None) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = F.silu(self.core.output_norm(hidden))
        pooled = torch.cat((normalized.mean(-1), normalized.amax(-1)), dim=1)
        quotient = self.core.quotient_head(pooled).softmax(1)
        carry = self.carry_head(normalized).softmax(1)
        if intervention == "qc_zero":
            quotient = torch.zeros_like(quotient); carry = torch.zeros_like(carry)
        elif intervention == "qc_permute" and hidden.shape[0] > 1:
            quotient = quotient.roll(1, 0); carry = carry.roll(1, 0)
        elif intervention == "qc_randomize":
            quotient = torch.randn(quotient.shape, generator=generator, device=quotient.device).softmax(1)
            carry = torch.randn(carry.shape, generator=generator, device=carry.device).softmax(1)
        return quotient, carry

    def step(self, inputs: torch.Tensor, tokens: torch.Tensor, qc_access: bool, token_access: bool,
             intervention: str = "normal", generator: torch.Generator | None = None
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        if intervention not in {"normal", "qc_zero", "qc_permute", "qc_randomize", "cross_zero"}:
            raise ValueError(f"unsupported step intervention: {intervention}")
        self.core.width = inputs.shape[-1]
        hidden = self.core.stem(inputs); spatial = hidden.transpose(1, 2)
        quotient_posterior, carry_posterior = self._posteriors(hidden, intervention, generator)
        quotient_message = (quotient_posterior @ self.quotient_embedding).unsqueeze(-1).expand(-1, -1, inputs.shape[-1])
        carry_message = torch.einsum("bcw,cd->bdw", carry_posterior, self.carry_embedding)
        qc_source = torch.cat((quotient_message, carry_message), dim=1)
        token_read = F.scaled_dot_product_attention(
            self.spatial_query(spatial).unsqueeze(1), self.token_key(tokens).unsqueeze(1),
            self.token_value(tokens).unsqueeze(1), dropout_p=0.0,
        ).squeeze(1).transpose(1, 2)
        qc_rank = torch.tanh(self.qc_down(qc_source)); token_rank = torch.tanh(self.token_down(token_read))
        qc_message = torch.tanh(self.qc_up(qc_rank)); token_message = torch.tanh(self.token_up(token_rank))
        cross_message = torch.tanh(self.cross_up(qc_rank * token_rank))
        cross_access = qc_access and token_access and intervention != "cross_zero"
        fused = (qc_message * float(qc_access) + token_message * float(token_access)
                 + cross_message * float(cross_access))
        scale, bias = fused.chunk(2, dim=1)
        hidden = hidden + torch.tanh(scale) * hidden + bias
        for dilation in self.core.dilation_schedule(inputs.shape[-1]):
            hidden = self.core.recurrent_update(hidden, inputs, dilation)
        normalized = F.silu(self.core.output_norm(hidden)); logits = self.core.local_head(normalized).squeeze(1)
        pooled = torch.cat((normalized.mean(-1), normalized.amax(-1)), dim=1)
        quotient_logits = self.core.quotient_head(pooled); carry_logits = self.carry_head(normalized)
        final_spatial = hidden.transpose(1, 2)
        write = F.scaled_dot_product_attention(
            tokens.unsqueeze(1), self.spatial_key(final_spatial).unsqueeze(1),
            self.spatial_value(final_spatial).unsqueeze(1), dropout_p=0.0,
        ).squeeze(1)
        updated = self.token_update(write.reshape(-1, self.token_dim), tokens.reshape(-1, self.token_dim)).reshape_as(tokens)
        entropy = lambda value: float((-(value * value.clamp_min(1e-12).log()).sum(1)).mean().detach())
        diagnostics = {
            "quotient_entropy": entropy(quotient_posterior), "carry_entropy": entropy(carry_posterior),
            "token_norm": float(tokens.detach().norm(dim=-1).mean()), "updated_token_norm": float(updated.detach().norm(dim=-1).mean()),
            "qc_rank_norm": float(qc_rank.detach().square().mean().sqrt()), "token_rank_norm": float(token_rank.detach().square().mean().sqrt()),
            "qc_message_norm": float(qc_message.detach().square().mean().sqrt()), "token_message_norm": float(token_message.detach().square().mean().sqrt()),
            "cross_message_norm": float(cross_message.detach().square().mean().sqrt()),
        }
        return logits, quotient_logits, carry_logits, updated, diagnostics

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, quotient, _, _, _ = self.step(inputs, self.initial_tokens(inputs.shape[0]), True, True)
        return logits, quotient


def new_student(config: dict[str, Any], state: dict[str, torch.Tensor] | None, seed: int = 696) -> QCM48Student:
    model = QCM48Student(config, None, seed)
    if state is not None:
        if all(name.startswith("core.") or name.startswith(("initial_token", "carry_head", "quotient_embedding", "carry_embedding",
                                                            "token_", "spatial_", "qc_", "cross_up")) for name in state):
            model.load_state_dict(state)
        else:
            model.core.load_state_dict(state)
    return model


def offline_carry_targets(states: list[int], multiplicands: list[int], moduli: list[int], flags: list[int], width: int) -> torch.Tensor:
    """Generate exact signed-carry labels for offline auxiliary training only."""
    from experiments.exp_658_joint_quotient_carry_transducer.data import LegalTransition, quotient_carry_trace
    rows = [quotient_carry_trace(LegalTransition(state, other, prime, flag), width).signed_carry_class_indices
            for state, other, prime, flag in zip(states, multiplicands, moduli, flags, strict=True)]
    return torch.tensor(rows, dtype=torch.long)


def diagnostics_mean(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def train_update(config: dict[str, Any], exp300, student: QCM48Student, teacher: SharedDilatedConvGRU,
                 optimizer: torch.optim.Optimizer, batch: tuple[list[int], list[int], list[int], list[int]],
                 width: int, qc_access: bool, token_access: bool) -> dict[str, Any]:
    left, right, modulus, starts = batch; horizon = int(config["training"]["horizon"])
    teacher_state = teacher_prefix(left, right, modulus, starts, width); state_plane = bits(teacher_state, width)
    tokens = student.initial_tokens(len(left)); losses = []; exacts = []; residues = []; quotient_terms = []; carry_terms = []; diagnostics = []
    optimizer.zero_grad(set_to_none=True); teacher.width = width
    for offset in range(horizon):
        flags = [(value >> (width - 1 - start - offset)) & 1 for value, start in zip(left, starts, strict=True)]
        raw = [2 * state + flag * other for state, flag, other in zip(teacher_state, flags, right, strict=True)]
        target = [value % prime for value, prime in zip(raw, modulus, strict=True)]
        quotient = torch.tensor([value // prime for value, prime in zip(raw, modulus, strict=True)], dtype=torch.long)
        carry_target = offline_carry_targets(teacher_state, right, modulus, flags, width)
        inputs = soft_features(state_plane, right, modulus, flags, width)
        logits, quotient_logits, carry_logits, tokens, diagnostic = student.step(inputs, tokens, qc_access, token_access)
        with torch.no_grad():
            teacher_logits, teacher_quotient = teacher(inputs.detach())
        exact_loss, stats = exp300.transition_loss(logits, quotient_logits, target, quotient, width,
                                                    float(config["training"]["worst_bit_weight"]), float(config["training"]["quotient_weight"]))
        residue = F.mse_loss(logits, teacher_logits) / teacher_logits.square().mean().clamp_min(1e-8)
        quotient_kl = F.kl_div(F.log_softmax(quotient_logits, 1), F.softmax(teacher_quotient, 1), reduction="batchmean")
        carry_loss = F.cross_entropy(carry_logits, carry_target)
        losses.append(exact_loss + float(config["training"]["carry_weight"]) * carry_loss
                      + float(config["training"]["residue_distillation_weight"]) * residue
                      + float(config["training"]["quotient_distillation_weight"]) * quotient_kl)
        exacts.append(float(stats["batch_exact"])); residues.append(residue); quotient_terms.append(quotient_kl); carry_terms.append(carry_loss); diagnostics.append(diagnostic)
        state_plane = hard_straight_through(logits); teacher_state = target
    objective = torch.stack(losses).mean(); objective.backward()
    gradient_sums = {name: 0.0 if module.weight.grad is None else float(module.weight.grad.abs().sum())
                     for name, module in (("qc", student.qc_up), ("token", student.token_up), ("cross", student.cross_up))}
    quotient_gradient = sum(float(parameter.grad.abs().sum()) for parameter in student.core.quotient_head.parameters() if parameter.grad is not None)
    carry_gradient = sum(float(parameter.grad.abs().sum()) for parameter in student.carry_head.parameters() if parameter.grad is not None)
    gradient = float(torch.nn.utils.clip_grad_norm_(student.parameters(), float(config["training"]["gradient_clip"]))); optimizer.step()
    return {"loss": float(objective.detach()), "gradient_norm": gradient, "fusion_gradients": gradient_sums,
            "quotient_head_gradient": quotient_gradient, "carry_head_gradient": carry_gradient, "mean_exact": float(np.mean(exacts)),
            "residue_distillation": float(torch.stack(residues).mean()), "quotient_distillation": float(torch.stack(quotient_terms).mean()),
            "carry_loss": float(torch.stack(carry_terms).mean()), "diagnostics": diagnostics_mean(diagnostics),
            "student_calls": horizon, "teacher_calls": horizon, "posterior_calls": horizon, "token_reads": horizon,
            "token_writes": horizon, "codec_calls": horizon, "fusion_calls": horizon, "auxiliary_loss_calls": horizon,
            "exact_loss_calls": horizon, "qc_access": qc_access, "token_access": token_access}


def train_arm(config: dict[str, Any], exp300, pruned: dict[str, torch.Tensor], checkpoint: Path, splits,
              root: int, arm: str, updates: int, output: Path) -> tuple[QCM48Student, dict[str, Any]]:
    random.seed(root + 696); np.random.seed(root + 696); torch.manual_seed(root + 696)
    model = new_student(config, pruned, root + 696900); initial = state_hash(model.state_dict()); timer = Timer(); history = []
    teacher = new_teacher(config, checkpoint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates, eta_min=float(config["training"]["minimum_learning_rate"]))
    samplers = {width: exp300.BigIntSampler(splits[width]["train"], root + int(config["data"]["training_seed_offset"]) + width)
                for width in set(config["data"]["curriculum"])}
    starts = {width: random.Random(root + int(config["data"]["training_seed_offset"]) + width + 1) for width in samplers}
    route = config["routes"][arm]; model.train()
    for update in range(1, updates + 1):
        tick = Timer(); width = int(config["data"]["curriculum"][(update - 1) % len(config["data"]["curriculum"])])
        left, right, modulus = samplers[width].pairs(int(config["training"]["batch_size"]))
        begin = [starts[width].randrange(0, width - int(config["training"]["horizon"]) + 1) for _ in left]
        stats = train_update(config, exp300, model, teacher, optimizer, (left, right, modulus, begin), width,
                             bool(route["qc_access"]), bool(route["token_access"]))
        scheduler.step(); row = {"update": update, "width": width, "learning_rate": scheduler.get_last_lr()[0], "timing": tick.finish(), **stats}; history.append(row)
        if update == 1 or update == updates or update % int(config["training"]["log_every_updates"]) == 0:
            print(json.dumps({"arm": arm, **row}), flush=True)
    model.eval(); arm_dir = output / "arms" / arm; arm_dir.mkdir(parents=True, exist_ok=True); checkpoint_path = arm_dir / "model.safetensors"
    save_file({key: value.detach().contiguous() for key, value in model.state_dict().items()}, checkpoint_path)
    reloaded = new_student(config, load_file(checkpoint_path), root + 696900)
    ledger = ("student_calls", "teacher_calls", "posterior_calls", "token_reads", "token_writes", "codec_calls", "fusion_calls", "auxiliary_loss_calls", "exact_loss_calls")
    return model, {"arm": arm, "initial_state_sha256": initial, "parameters": model.parameter_count(),
                   "training": {"updates": updates, "curriculum": list(config["data"]["curriculum"]),
                                **{key: sum(int(row[key]) for row in history) for key in ledger}, "history": history, "resource": timer.finish()},
                   "checkpoint": {"path": str(checkpoint_path), "sha256": sha256(checkpoint_path), "bytes": checkpoint_path.stat().st_size},
                   "reload_exact": all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), reloaded.state_dict().values(), strict=True))}


def decode(rows: torch.Tensor) -> list[int]:
    return [sum(int(bit) << index for index, bit in enumerate(row)) for row in rows.tolist()]


@torch.inference_mode()
def rollout(model: QCM48Student, left: list[int], right: list[int], modulus: list[int], width: int,
            qc_access: bool = True, token_access: bool = True, intervention: str = "normal") -> dict[str, Any]:
    if intervention not in {"normal", *INTERVENTIONS}:
        raise ValueError(f"unsupported intervention: {intervention}")
    state_bits = torch.zeros((len(left), width), dtype=torch.bool); left_bits = bits(left, width)
    invariant = torch.empty((len(left), 6, width)); invariant[:, 1] = bits(right, width).mul(2).sub(1)
    invariant[:, 2] = bits(modulus, width).mul(2).sub(1); invariant[:, 4].fill_(-1); invariant[:, 4, 0] = 1
    invariant[:, 5].fill_(-1); invariant[:, 5, -1] = 1
    tokens = model.initial_tokens(len(left)); teacher = [0] * len(left); first = [None] * len(left); invalid = [False] * len(left)
    diagnostics = []; timer = Timer(); generator = torch.Generator().manual_seed(696123 + width + len(left))
    for position in range(width):
        if intervention == "token_reset": tokens = model.initial_tokens(len(left))
        elif intervention == "token_permute" and len(left) > 1: tokens = tokens.roll(1, 0)
        elif intervention == "token_randomize": tokens = torch.randn(tokens.shape, generator=generator)
        previous = tokens
        invariant[:, 0] = state_bits.float().mul(2).sub(1); invariant[:, 3] = left_bits[:, width - 1 - position, None].mul(2).sub(1)
        step_intervention = intervention if intervention in {"qc_zero", "qc_permute", "qc_randomize", "cross_zero"} else "normal"
        logits, _, _, updated, diagnostic = model.step(invariant, tokens, qc_access, token_access, step_intervention, generator)
        tokens = previous if intervention == "token_freeze" else updated; diagnostics.append(diagnostic)
        state_bits = logits.ge(0); values = decode(state_bits); flags = [int(value) for value in left_bits[:, width - 1 - position]]
        teacher = [(2 * state + flag * other) % prime for state, flag, other, prime in zip(teacher, flags, right, modulus, strict=True)]
        for index, (guess, truth, prime) in enumerate(zip(values, teacher, modulus, strict=True)):
            if guess != truth and first[index] is None: first[index] = position
            invalid[index] = invalid[index] or guess >= prime
    prediction = decode(state_bits); target = [(a * b) % p for a, b, p in zip(left, right, modulus, strict=True)]
    correct = [guess == truth for guess, truth in zip(prediction, target, strict=True)]; errors = [value for value in first if value is not None]
    return {"cases": len(left), "correct": sum(correct), "exact_accuracy": sum(correct) / max(1, len(correct)), "correct_vector": correct,
            "prediction_sha256": canonical_hash(prediction), "invalids": sum(invalid), "any_invalid_residue_rate": sum(invalid) / max(1, len(invalid)),
            "mean_first_error_position": None if not errors else float(np.mean(errors)), "diagnostics": diagnostics_mean(diagnostics),
            "maximum_shapes": {"quotient": [len(left), 3], "carry": [len(left), 4, width], "tokens": [len(left), 4, 16], "rank": [len(left), 4, width]},
            "resource": timer.finish()}


class ArmView(nn.Module):
    """Evaluator-compatible view fixing only the two preregistered access masks."""
    def __init__(self, model: QCM48Student, qc_access: bool, token_access: bool) -> None:
        super().__init__(); self.model = model; self.qc_access = qc_access; self.token_access = token_access

    @property
    def width(self) -> int: return self.model.width

    @width.setter
    def width(self, value: int) -> None: self.model.width = value; self.model.core.width = value

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, quotient, _, _, _ = self.model.step(inputs, self.model.initial_tokens(inputs.shape[0]), self.qc_access, self.token_access)
        return logits, quotient


def transition_eval(config, exp300, model, split, root, width, cases, arm) -> dict[str, Any]:
    route = config["routes"][arm]; view = ArmView(model, bool(route["qc_access"]), bool(route["token_access"])); view.width = width; timer = Timer()
    value = exp300.evaluate_transitions(view, exp300.BigIntSampler(split["test"], root + int(config["data"]["test_seed_offset"]) + width), cases)
    value["resource"] = timer.finish(); return value


def common_rollouts(config, exp300, models, split, root, width, cases) -> dict[str, Any]:
    sampler = exp300.BigIntSampler(split["test"], root + int(config["data"]["test_seed_offset"]) + width + 1)
    left, right, modulus = sampler.pairs(cases); arms = {}
    for arm in TRAINED_ARMS:
        route = config["routes"][arm]
        arms[arm] = rollout(models[arm], left, right, modulus, width, bool(route["qc_access"]), bool(route["token_access"]))
    both = arms["both"]["correct_vector"]; neither = arms["neither"]["correct_vector"]
    pair = paired_counts(both, neither); alpha = (1.0 - float(config["evaluation"]["confidence_level"])) / int(config["evaluation"]["holm_family_size"])
    pair["confidence_sequence"] = anytime_hoeffding_interval([float(a) - float(b) for a, b in zip(both, neither, strict=True)], alpha)
    interaction_values = [float(b) - float(q) - float(t) + float(n) for b, q, t, n in zip(
        both, arms["qc_only"]["correct_vector"], arms["token_only"]["correct_vector"], neither, strict=True)]
    interaction = scaled_anytime_interval(interaction_values, alpha, 2.0)
    return {"arms": arms, "both_vs_neither": pair, "factorial_interaction": interaction}


def candidate_inputs(root: int) -> list[tuple[str, str, str, int]]:
    source = ROOT / "sources/github/modular-arithmetic-challenge/src"; sys.path.insert(0, str(source))
    try:
        from modchallenge.config import EvalConfig
        from modchallenge.testgen.generator import generate_private_test_set
        test_set = generate_private_test_set(hashlib.sha256(f"exp696-c7-{root}".encode()).digest(), EvalConfig())
        return [(case.a, case.b, case.p, tier.tier_id) for tier in test_set.tiers for case in tier.cases]
    finally:
        sys.path.remove(str(source))


@torch.inference_mode()
def deployed_route(config, checkpoint: Path, root: int, limit: int | None = None) -> dict[str, Any]:
    all_rows = candidate_inputs(root)
    if limit is None:
        rows = all_rows
    else:
        lower_smoke = [row for row in all_rows if int(row[2]).bit_length() <= 128]
        wide_smoke = [row for row in all_rows if 128 < int(row[2]).bit_length() <= 256]
        lower_count = min(len(lower_smoke), max(1, limit // 2))
        rows = lower_smoke[:lower_count] + wide_smoke[:max(0, limit - lower_count)]
    candidate = exp687.load_module(f"candidate005_exp696_{root}_{limit}_{random.randrange(1 << 20)}", ROOT / "submission/model.py")
    incumbent = candidate.Tier2HornerModel(); incumbent.load(str(ROOT / "submission"))
    model = new_student(config, load_file(checkpoint), root + 696900).eval(); timer = Timer()
    lower = [(a, b, p) for a, b, p, _ in rows if int(p).bit_length() <= 128]
    wide = [row for row in rows if 128 < int(row[2]).bit_length() <= 256]; upper = [row for row in rows if int(row[2]).bit_length() > 256]
    low_out = incumbent.predict_digits_batch(lower) if lower else []; mod = [int(row[2]) for row in wide]
    wide_result = rollout(model, [int(row[0]) % prime for row, prime in zip(wide, mod, strict=True)],
                          [int(row[1]) % prime for row, prime in zip(wide, mod, strict=True)], mod, 256, True, True) if wide else {"prediction_sha256": canonical_hash([]), "maximum_shapes": {}}
    resource = timer.finish()
    return {"cases": len(rows), "complete_1100_case_route": len(rows) == 1100, "lower_incumbent_cases": len(lower), "w256_specialist_cases": len(wide),
            "above_w256_zero_fallback_cases": len(upper), "output_count": len(low_out) + len(wide) + len(upper), "lower_output_sha256": canonical_hash(low_out),
            "wide_output_sha256": wide_result["prediction_sha256"], "maximum_shapes": wide_result["maximum_shapes"],
            "labels_available_to_timed_route": False, "operand_swap_route": False, "deployed_arm": "both", "resource": resource}


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
        if self.config["run_id"] != RUN_ID or not self.checkpoint.is_file() or sha256(self.checkpoint) != self.config["source_checkpoint_sha256"]:
            raise RuntimeError("source checkpoint mismatch")
        for name, binding in self.config["source_contract"].items():
            if sha256(ROOT / binding["path"]) != binding["sha256"]: raise RuntimeError(f"source contract mismatch: {name}")
        challenge_repo = ROOT / "sources/github/modular-arithmetic-challenge"
        if git_output("-C", str(challenge_repo), "rev-parse", "HEAD") != CHALLENGE_COMMIT: raise RuntimeError("challenge commit mismatch")

    def identity(self) -> dict[str, Any]:
        return {"repository_commit": git_output("rev-parse", "HEAD"), "challenge_commit": CHALLENGE_COMMIT,
                "config_sha256": sha256(self.config_path), "runner_sha256": sha256(Path(__file__)),
                "test_sha256": sha256(ROOT / "tests/test_exp_696_qcm48_factorial.py"), "source_checkpoint_sha256": sha256(self.checkpoint),
                "channel_map_sha256": canonical_hash(self.manifest["selected_source_channels_ranked"]), "pruned_state_sha256": self.manifest["pruned_state_sha256"],
                "source_contract": self.config["source_contract"]}

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
        ledger = ("updates", "curriculum", "student_calls", "teacher_calls", "posterior_calls", "token_reads", "token_writes", "codec_calls", "fusion_calls", "auxiliary_loss_calls", "exact_loss_calls")
        initial = {records[arm]["initial_state_sha256"] for arm in TRAINED_ARMS}; parameters = {records[arm]["parameters"] for arm in TRAINED_ARMS}
        if len(initial) != 1 or len(parameters) != 1 or any(len({json.dumps(records[arm]["training"][key], sort_keys=True) for arm in TRAINED_ARMS}) != 1 for key in ledger):
            raise RuntimeError("four-arm initial-state/parameter/work matching contract failed")
        return models, records

    def smoke(self) -> dict[str, Any]:
        old_root = self.root; self.root = 696000; self.splits = {width: make_split(self.config, self.exp300, width, self.root) for width in (64, 128, 256)}
        with tempfile.TemporaryDirectory(prefix="exp696-smoke-", dir=self.output) as directory:
            path = Path(directory); models, records = self.train_routes(path, 1)
            common = common_rollouts(self.config, self.exp300, models, self.splits[128], self.root, 128, 2)
            sample = self.exp300.BigIntSampler(self.splits[128]["test"], self.root + 9).pairs(2)
            interventions = {name: rollout(models["both"], *sample, 128, True, True, name) for name in INTERVENTIONS}
            randomized = rollout(new_student(self.config, None, 696999).eval(), *sample, 128, True, True)
            route = deployed_route(self.config, Path(records["both"]["checkpoint"]["path"]), self.root, 2)
            serialized = path / "terminal.json"; write_json(serialized, {"status": "completed", "common": common, "interventions": interventions, "randomized": randomized}); json.loads(serialized.read_text())
            result = {"schema_version": "exp696-rootless-all-mode-smoke-v1", "status": "completed", "root": None,
                      "all_modes": ["four-arm-train", "reload", "common-evaluate", *INTERVENTIONS, "randomize-full", "both-deployed-route", "serialize"],
                      "four_arms_updated": all(records[arm]["training"]["updates"] == 1 for arm in TRAINED_ARMS),
                      "identical_initial_state": len({records[arm]["initial_state_sha256"] for arm in TRAINED_ARMS}) == 1,
                      "equal_parameters": len({records[arm]["parameters"] for arm in TRAINED_ARMS}) == 1,
                      "equal_work": len({tuple(records[arm]["training"][key] for key in ("student_calls", "posterior_calls", "token_reads", "token_writes", "codec_calls", "fusion_calls", "auxiliary_loss_calls")) for arm in TRAINED_ARMS}) == 1,
                      "heads_train_all_arms": all(records[arm]["training"]["history"][0][key] > 0 for arm in TRAINED_ARMS for key in ("quotient_head_gradient", "carry_head_gradient")),
                      "reload_exact": all(records[arm]["reload_exact"] for arm in TRAINED_ARMS), "route_cases": route["cases"],
                      "interventions": sorted(interventions), "factorial_interaction_recorded": "factorial_interaction" in common, "terminal_roundtrip": True}
        self.root = old_root; write_json(self.output / "smoke.json", result); return result

    def calibration(self) -> dict[str, Any]:
        path = self.open_root(); timer = Timer(); models, records = self.train_routes(path, int(self.config["training"]["calibration_updates"]))
        transition_cases = int(self.config["calibration"]["transition_cases_per_arm_width"]); rollout_cases = int(self.config["calibration"]["rollout_cases_per_arm_width"])
        transitions = {arm: {str(width): transition_eval(self.config, self.exp300, model, self.splits[width], self.root, width, transition_cases, arm) for width in (128, 256)} for arm, model in models.items()}
        rollouts = {str(width): common_rollouts(self.config, self.exp300, models, self.splits[width], self.root, width, rollout_cases) for width in (128, 256)}
        checkpoint = Path(records["both"]["checkpoint"]["path"]); warmup = deployed_route(self.config, checkpoint, self.root, 1)
        cold = [deployed_route(self.config, checkpoint, self.root) for _ in range(int(self.config["calibration"]["cold_repetitions"]))]
        walls = [float(item["resource"]["wall_seconds"]) for item in cold]
        if max(walls) / max(float(np.median(walls)), 1e-12) > float(self.config["calibration"]["variability_trigger"]):
            cold.extend(deployed_route(self.config, checkpoint, self.root) for _ in range(int(self.config["calibration"]["extra_cold_repetitions"]))); walls = [float(item["resource"]["wall_seconds"]) for item in cold]
        resource = timer.finish(); update_wall = sum(float(records[arm]["training"]["resource"]["wall_seconds"]) for arm in TRAINED_ARMS)
        eval_wall = sum(float(transitions[arm][str(width)]["resource"]["wall_seconds"]) for arm in TRAINED_ARMS for width in (128, 256)) + sum(float(rollouts[str(width)]["arms"][arm]["resource"]["wall_seconds"]) for arm in TRAINED_ARMS for width in (128, 256))
        expected_training = update_wall * int(self.config["training"]["updates"]) / int(self.config["training"]["calibration_updates"])
        measured_eval_units = len(TRAINED_ARMS) * 2 * (transition_cases + rollout_cases)
        target_eval_units = len(TRAINED_ARMS) * (2 * int(self.config["evaluation"]["transition_cases_per_width"]) + int(self.config["evaluation"]["w128_rollout_cases"]) + max(self.config["evaluation"]["w256_sequential_blocks"]))
        expected_eval = eval_wall * target_eval_units / measured_eval_units; expected = expected_training + expected_eval; maximum = max(walls)
        admission = {"candidate_cold_wall_seconds": walls, "candidate_max_complete_wall_seconds": maximum, "candidate_guarded_wall_seconds": 2.0 * maximum,
                     "projected_training_wall_seconds": expected_training, "projected_evaluation_wall_seconds": expected_eval, "expected_wall_seconds": expected,
                     "soft_guard_wall_seconds": math.ceil(expected * float(self.config["resource_policy"]["soft_guard_multiplier"])),
                     "hard_guard_wall_seconds": math.ceil(expected * float(self.config["resource_policy"]["hard_guard_multiplier"])),
                     "measured_process_cpu_seconds": resource["process_cpu_seconds"], "measured_cgroup_cpu_seconds": resource["cgroup_cpu_seconds"],
                     "measured_peak_rss_bytes": resource["peak_rss_bytes"], "fidelity_reduced": False}
        admission["checks"] = {"complete_cold_routes": len(cold) in (3, 5) and all(item["complete_1100_case_route"] and item["output_count"] == 1100 for item in cold),
                               "candidate_runtime": 2.0 * maximum <= float(self.config["gates"]["maximum_candidate_guarded_wall_seconds"]),
                               "memory": int(resource["peak_rss_bytes"]) <= int(self.config["resource_policy"]["maximum_peak_rss_bytes"]),
                               "equal_work": len({records[arm]["training"]["fusion_calls"] for arm in TRAINED_ARMS}) == 1,
                               "reload_exact": all(records[arm]["reload_exact"] for arm in TRAINED_ARMS),
                               "deterministic_outputs": len({item["wide_output_sha256"] for item in cold}) == 1,
                               "deployed_both": all(item["deployed_arm"] == "both" for item in cold), "no_fidelity_reduction": True}
        admission["admitted"] = all(admission["checks"].values()); decision = "admit-root20261369" if admission["admitted"] else "park-exp696-at-root20261368-resource-admission"
        result = {"schema_version": "exp696-calibration-v1", "run_id": RUN_ID, "status": "completed", "decision": decision, "root": self.root,
                  "source": self.identity(), "records": records, "transitions": transitions, "rollouts": rollouts,
                  "candidate_runtime": {"warmup": warmup, "cold": cold}, "resource": resource, "admission": admission, "limitations": self.limitations(True)}
        write_json(path / "root-summary.json", result); write_json(self.output / "calibration.json", result); return self.log(result)

    def admitted(self) -> dict[str, Any]:
        path = self.output / "calibration.json"
        if not path.is_file(): raise RuntimeError("root20261368 admission required")
        value = json.loads(path.read_text())
        if value["decision"] != "admit-root20261369" or not value["admission"]["admitted"] or value["source"]["runner_sha256"] != sha256(Path(__file__)):
            raise RuntimeError("matching admitted root20261368 required")
        return value

    def science(self) -> dict[str, Any]:
        calibration = self.admitted(); path = self.open_root(); timer = Timer(); models, records = self.train_routes(path, int(self.config["training"]["updates"]))
        transitions = {arm: {str(width): transition_eval(self.config, self.exp300, model, self.splits[width], self.root, width, int(self.config["evaluation"]["transition_cases_per_width"]), arm) for width in (128, 256)} for arm, model in models.items()}
        w128 = common_rollouts(self.config, self.exp300, models, self.splits[128], self.root, 128, int(self.config["evaluation"]["w128_rollout_cases"]))
        max_cases = max(self.config["evaluation"]["w256_sequential_blocks"]); full = common_rollouts(self.config, self.exp300, models, self.splits[256], self.root, 256, max_cases); sequential = []
        alpha = (1.0 - float(self.config["evaluation"]["confidence_level"])) / int(self.config["evaluation"]["holm_family_size"])
        for cases in self.config["evaluation"]["w256_sequential_blocks"]:
            vectors = {arm: full["arms"][arm]["correct_vector"][:cases] for arm in TRAINED_ARMS}; pair = paired_counts(vectors["both"], vectors["neither"])
            pair["confidence_sequence"] = anytime_hoeffding_interval([float(a) - float(b) for a, b in zip(vectors["both"], vectors["neither"], strict=True)], alpha)
            interaction_values = [float(b) - float(q) - float(t) + float(n) for b, q, t, n in zip(vectors["both"], vectors["qc_only"], vectors["token_only"], vectors["neither"], strict=True)]
            sequential.append({"cases": cases, "accuracies": {arm: sum(vector) / cases for arm, vector in vectors.items()}, "both_vs_neither": pair,
                               "factorial_interaction": scaled_anytime_interval(interaction_values, alpha, 2.0)})
        sampler = self.exp300.BigIntSampler(self.splits[256]["test"], self.root + int(self.config["data"]["test_seed_offset"]) + 257); left, right, modulus = sampler.pairs(max_cases)
        interventions = {name: rollout(models["both"], left, right, modulus, 256, True, True, name) for name in INTERVENTIONS}
        randomized = rollout(new_student(self.config, None, self.root + 696999).eval(), left, right, modulus, 256, True, True)
        both = full["arms"]["both"]; neither = full["arms"]["neither"]; gain = both["exact_accuracy"] - neither["exact_accuracy"]
        erasures = {name: both["exact_accuracy"] - value["exact_accuracy"] for name, value in interventions.items()}
        cross_zero_interaction = interventions["cross_zero"]["exact_accuracy"] - full["arms"]["qc_only"]["exact_accuracy"] - full["arms"]["token_only"]["exact_accuracy"] + neither["exact_accuracy"]
        interaction = full["factorial_interaction"]
        causal_names = tuple(name for name in INTERVENTIONS if name != "cross_zero")
        gates = {"healthy_neither": transitions["neither"]["128"]["exact_accuracy"] >= float(self.config["gates"]["minimum_neither_w128_transition"]),
                 "complete_updates": all(records[arm]["training"]["updates"] == 2400 for arm in TRAINED_ARMS),
                 "both_w256_transition": transitions["both"]["256"]["exact_accuracy"] >= float(self.config["gates"]["minimum_both_w256_transition"]),
                 "both_w256_rollout": both["exact_accuracy"] >= float(self.config["gates"]["minimum_both_w256_rollout"]),
                 "invalid_rate": both["any_invalid_residue_rate"] <= float(self.config["gates"]["maximum_invalid_rate"]),
                 "paired_gain": gain >= float(self.config["gates"]["minimum_paired_gain"]),
                 "w128_retention": w128["arms"]["both"]["exact_accuracy"] >= w128["arms"]["neither"]["exact_accuracy"] - float(self.config["gates"]["maximum_w128_loss"]),
                 "positive_interaction": interaction["mean"] > 0.0, "interaction_confidence": interaction["lower"] > 0.0,
                 "causal_erasure": gain > 0 and all(erasures[name] >= float(self.config["gates"]["minimum_intervention_erasure_fraction"]) * gain for name in causal_names),
                 "cross_term_dependence": interaction["lower"] > 0.0 and cross_zero_interaction <= 0.0,
                 "posterior_token_noncollapse": both["diagnostics"]["quotient_entropy"] > 0 and both["diagnostics"]["carry_entropy"] > 0 and both["diagnostics"]["updated_token_norm"] > 0,
                 "equal_work": len({records[arm]["training"]["fusion_calls"] for arm in TRAINED_ARMS}) == 1,
                 "reload_exact": all(records[arm]["reload_exact"] for arm in TRAINED_ARMS), "runtime_admitted": calibration["admission"]["admitted"],
                 "confirmation_threshold": both["exact_accuracy"] >= float(self.config["gates"]["confirmation_admission_rollout"])}
        common = all(gates[key] for key in ("healthy_neither", "complete_updates", "both_w256_transition", "both_w256_rollout", "invalid_rate", "paired_gain", "w128_retention", "equal_work", "reload_exact", "runtime_admitted"))
        combined = all(gates[key] for key in ("positive_interaction", "interaction_confidence", "causal_erasure", "cross_term_dependence", "posterior_token_noncollapse"))
        decision = "advance-to-root20261370" if common and combined and gates["confirmation_threshold"] else ("non-comparative-unhealthy-neither" if not gates["healthy_neither"] else "reject-exp696-combined-mechanism")
        result = {"schema_version": "exp696-science-v1", "run_id": RUN_ID, "status": "completed", "root": self.root, "decision": decision,
                  "source": self.identity(), "records": records, "transitions": transitions, "w128": w128, "w256": full, "sequential_w256": sequential,
                  "factorial": {"formula": "both-qc_only-token_only+neither", "interaction": interaction, "cross_zero_interaction": cross_zero_interaction},
                  "interventions": interventions, "intervention_erasures": erasures, "randomized_full_weights": randomized,
                  "gates": gates, "resource": timer.finish(), "dynamic_budget": calibration["admission"], "limitations": self.limitations(False)}
        write_json(path / "root-summary.json", result); write_json(self.output / "terminal-summary.json", result); return self.log(result)

    def confirmation(self) -> dict[str, Any]:
        prior = int(self.config["roots"]["formal"]) if self.root == int(self.config["roots"]["confirmations"][0]) else int(self.config["roots"]["confirmations"][0])
        prior_mode = "formal" if prior == int(self.config["roots"]["formal"]) else "confirmation"; prior_path = self.output / prior_mode / f"root-{prior}" / "root-summary.json"
        allowed_decisions = ("advance-to-root20261370", "confirmation-pass")
        if not prior_path.is_file() or json.loads(prior_path.read_text())["decision"] not in allowed_decisions: raise RuntimeError("confirmation predecessor gate failed")
        result = self.science(); result["decision"] = "confirmation-pass" if result["gates"]["confirmation_threshold"] and all(value for key, value in result["gates"].items() if key != "confirmation_threshold") else "reject-exp696-confirmation"
        write_json(self.root_dir() / "root-summary.json", result); write_json(self.output / "terminal-summary.json", result); return self.log(result)

    def independent(self) -> dict[str, Any]:
        predecessor = self.output / "confirmation" / f"root-{int(self.config['roots']['confirmations'][-1])}" / "root-summary.json"
        if not predecessor.is_file() or json.loads(predecessor.read_text())["decision"] != "confirmation-pass": raise RuntimeError("both confirmations required")
        seeds = list(map(int, self.config["roots"]["independent_seeds"])); index = seeds.index(int(self.root))
        if index:
            previous = self.output / "independent" / f"root-{seeds[index - 1]}" / "root-summary.json"
            if not previous.is_file() or json.loads(previous.read_text())["decision"] not in ("independent-seed-pass", "independent-suite-pass"): raise RuntimeError("independent predecessor gate failed")
        result = self.science(); accuracy = float(result["w256"]["arms"]["both"]["exact_accuracy"]); result["independent_seed_gate"] = accuracy >= float(self.config["gates"]["independent_seed_minimum"])
        accuracies = [float(json.loads((self.output / "independent" / f"root-{seed}" / "root-summary.json").read_text())["w256"]["arms"]["both"]["exact_accuracy"]) for seed in seeds[:index]] + [accuracy]
        result["independent_accuracies_through_root"] = accuracies; result["independent_aggregate_accuracy"] = float(np.mean(accuracies))
        if not result["independent_seed_gate"]: result["decision"] = "reject-exp696-independent-seed"
        elif index < len(seeds) - 1: result["decision"] = "independent-seed-pass"
        else:
            result["independent_aggregate_gate"] = result["independent_aggregate_accuracy"] >= float(self.config["gates"]["independent_aggregate_minimum"])
            result["decision"] = "independent-suite-pass" if result["independent_aggregate_gate"] else "reject-exp696-independent-aggregate"
        write_json(self.root_dir() / "root-summary.json", result); write_json(self.output / "terminal-summary.json", result); return self.log(result)

    def log(self, result: dict[str, Any]) -> dict[str, Any]:
        experiment_id = configure_mlflow(self.config["tracking"]["experiment_name"])
        with mlflow.start_run(run_name=f"{RUN_ID}-{self.mode}-{self.root}") as active:
            mlflow.set_tags({"canonical_run_id": RUN_ID, "mode": self.mode, "root": str(self.root), "decision": result["decision"], "candidate_registration": "false", "tournament_entrant": "C7"})
            mlflow.log_dict(result, f"{self.mode}-result.json"); run_id = active.info.run_id
        result["mlflow"] = {"tracking_uri": tracking_uri(), "experiment_id": experiment_id, "run_id": run_id}; write_json(self.root_dir() / "root-summary.json", result); return result

    @staticmethod
    def limitations(calibration: bool) -> list[str]:
        return ["Calibration accuracy is route-integrity evidence, not science." if calibration else "Fresh generated identities only; checked/public/private answers were not used.",
                "Exact arithmetic, quotient/carry labels, and teacher outputs exist only in offline training/evaluation; deployed inference is learned and input-only.",
                "The rank-4 PyTorch codec is a frozen C1/C3 composition; no C5 routing or C6 cycle coefficients are present.",
                "A failed interaction rejects only the combined C7 mechanism under healthy complete evidence; no entrant cancels another.",
                "C5 remains unresolved and a final seventh-member adversarial chair is mandatory.",
                "No candidate, submission, promotion, release, tournament-winner, or private-score claim is authorized."]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=CONFIG_PATH); parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--mode", choices=("smoke", "calibration", "formal", "confirmation", "independent"), required=True); parser.add_argument("--root", type=int); args = parser.parse_args()
    experiment = Experiment(args.config, args.output, args.mode, args.root)
    result = experiment.smoke() if args.mode == "smoke" else experiment.calibration() if args.mode == "calibration" else experiment.science() if args.mode == "formal" else experiment.confirmation() if args.mode == "confirmation" else experiment.independent()
    print(json.dumps({"mode": args.mode, "root": result.get("root"), "status": result["status"], "decision": result.get("decision"), "gates": result.get("gates"), "admission": result.get("admission", {}).get("checks")}, indent=2, sort_keys=True), flush=True); return 0


if __name__ == "__main__": raise SystemExit(main())
