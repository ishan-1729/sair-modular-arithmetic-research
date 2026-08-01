"""Execute EXP694 C5: sparse learned routing with antithetic quorum proposals."""

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

RUN_ID = "exp-694-routed-antithetic-quorum"
SLUG = RUN_ID
CONFIG_PATH = ROOT / "configs/exp_694_routed_antithetic_quorum.yaml"
OUTPUT = ROOT / ".scratch/exp-694-routed-antithetic-quorum"
TRAINED_ARMS = ("primary", "random_routing_control")
CHALLENGE_COMMIT = exp687.CHALLENGE_COMMIT

canonical_hash = exp687.canonical_hash
state_hash = exp687.state_hash
configure_torch = exp687.configure_torch
prepare_pruned = exp687.prepare_pruned
new_teacher = exp687.new_teacher
soft_features = exp687.soft_features
hard_straight_through = exp687.hard_straight_through
teacher_prefix = exp687.teacher_prefix


class RoutedAntitheticQuorumStudent(nn.Module):
    """h56 Horner cell with one-of-four sparse three-proposal learned updates."""

    def __init__(self, config: dict[str, Any], core_state: dict[str, torch.Tensor] | None, seed: int = 694) -> None:
        super().__init__()
        hidden = int(config["model"]["hidden"])
        self.width = int(config["model"]["width"])
        self.group_size = int(config["model"]["quota_group_size"])
        self.core = SharedDilatedConvGRU(self.width, hidden, 1)
        if core_state is not None:
            self.core.load_state_dict(core_state)
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.survival_q = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
            self.antithetic_delta = nn.Conv1d(hidden, hidden, 1)
            self.energy = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
            self.quorum_scale = nn.Parameter(torch.tensor(1.0))
            nn.init.normal_(self.antithetic_delta.weight, std=0.01)
            nn.init.zeros_(self.antithetic_delta.bias)
        self.hidden = hidden

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def route_indices(self, scores: torch.Tensor, position: int, route_mode: str, route_seed: int) -> tuple[torch.Tensor, int]:
        eligible = scores.numel() - scores.numel() % self.group_size
        if eligible == 0:
            return torch.empty(0, dtype=torch.long, device=scores.device), 0
        groups = torch.arange(eligible, device=scores.device).reshape(-1, self.group_size)
        if route_mode == "learned-survival-q":
            choice = scores[:eligible].reshape(-1, self.group_size).argmax(1)
        elif route_mode == "deterministic-target-independent-random":
            index = groups.to(torch.int64) + int(route_seed) * 131 + int(position) * 977
            hashed = index ^ (index >> 16)
            hashed = (hashed * 1103515245 + 12345) & 0x7FFFFFFF
            hashed = hashed ^ (hashed >> 11)
            choice = hashed.argmax(1)
        else:
            raise ValueError(f"unknown route mode: {route_mode}")
        return groups[torch.arange(groups.shape[0], device=scores.device), choice], eligible

    def _heads(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = F.silu(self.core.output_norm(hidden))
        logits = self.core.local_head(normalized).squeeze(1)
        pooled = torch.cat((normalized.mean(-1), normalized.amax(-1)), dim=1)
        return logits, self.core.quotient_head(pooled), pooled

    def step(self, inputs: torch.Tensor, position: int, route_mode: str, route_seed: int,
             intervention: str = "normal") -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], torch.Tensor]:
        self.core.width = inputs.shape[-1]
        hidden = self.core.stem(inputs)
        for dilation in self.core.dilation_schedule(inputs.shape[-1]):
            hidden = self.core.recurrent_update(hidden, inputs, dilation)
        base_logits, base_quotient, pooled = self._heads(hidden)
        route_scores = self.survival_q(pooled).squeeze(1)
        effective_mode = "deterministic-target-independent-random" if intervention == "route_randomization" else route_mode
        routed, eligible = self.route_indices(route_scores, position, effective_mode, route_seed)
        if routed.numel() == 0:
            diagnostic = {"eligible": eligible, "routed": 0, "route_fraction": 0.0, "proposal_calls": inputs.shape[0],
                          "route_index_sha256": canonical_hash([]), "energy_entropy": 0.0}
            return base_logits, base_quotient, diagnostic, route_scores

        selected = hidden.index_select(0, routed)
        delta = torch.tanh(self.antithetic_delta(selected))
        plus = selected + delta
        minus = selected + (delta if intervention == "sign_collapse" else -delta)
        proposal_hidden = torch.stack((selected, plus, minus), dim=1)
        flat = proposal_hidden.flatten(0, 1)
        proposal_logits, proposal_quotient, proposal_pooled = self._heads(flat)
        proposal_logits = proposal_logits.reshape(routed.numel(), 3, -1)
        proposal_quotient = proposal_quotient.reshape(routed.numel(), 3, -1)
        proposal_pooled = proposal_pooled.reshape(routed.numel(), 3, -1)
        if intervention == "branch_shuffle" and routed.numel() > 1:
            proposal_logits[:, 1:] = proposal_logits.roll(1, 0)[:, 1:]
            proposal_quotient[:, 1:] = proposal_quotient.roll(1, 0)[:, 1:]
            proposal_pooled[:, 1:] = proposal_pooled.roll(1, 0)[:, 1:]
        energy = self.energy(proposal_pooled).squeeze(-1)
        agreement = -(proposal_logits - proposal_logits.mean(1, keepdim=True)).abs().mean(-1)
        selection = energy + self.quorum_scale * agreement
        if intervention == "energy_erasure":
            selection = torch.zeros_like(selection)
        elif intervention == "energy_permutation" and routed.numel() > 1:
            selection = selection.roll(1, 0)
        weights = selection.softmax(1)
        selected_logits = torch.einsum("bp,bpw->bw", weights, proposal_logits)
        selected_quotient = torch.einsum("bp,bpc->bc", weights, proposal_quotient)
        logits = base_logits.index_copy(0, routed, selected_logits)
        quotient = base_quotient.index_copy(0, routed, selected_quotient)
        entropy = -(weights * weights.clamp_min(1e-12).log()).sum(1).mean()
        diagnostic = {"eligible": eligible, "routed": int(routed.numel()), "route_fraction": int(routed.numel()) / eligible,
                      "proposal_calls": int(inputs.shape[0] + 2 * routed.numel()),
                      "route_index_sha256": canonical_hash(routed.detach().cpu().tolist()),
                      "energy_entropy": float(entropy.detach()), "delta_norm": float(delta.detach().norm(dim=1).mean())}
        return logits, quotient, diagnostic, route_scores

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, quotient, _, _ = self.step(inputs, 0, "learned-survival-q", 694000)
        return logits, quotient


class ArmView(nn.Module):
    def __init__(self, model: RoutedAntitheticQuorumStudent, route_mode: str, seed: int) -> None:
        super().__init__(); self.model = model; self.route_mode = route_mode; self.seed = seed

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, quotient, _, _ = self.model.step(inputs, 0, self.route_mode, self.seed)
        return logits, quotient


def new_student(config: dict[str, Any], state: dict[str, torch.Tensor] | None, seed: int = 694) -> RoutedAntitheticQuorumStudent:
    model = RoutedAntitheticQuorumStudent(config, None, seed)
    if state is not None:
        if all(name.startswith(("core.", "survival_q", "antithetic_delta", "energy", "quorum_scale")) for name in state):
            model.load_state_dict(state)
        else:
            model.core.load_state_dict(state)
    return model


def train_update(config: dict[str, Any], exp300, student: RoutedAntitheticQuorumStudent, teacher: SharedDilatedConvGRU,
                 optimizer: torch.optim.Optimizer, batch: tuple[list[int], list[int], list[int], list[int]], width: int,
                 route_mode: str, route_seed: int) -> dict[str, Any]:
    left, right, modulus, starts = batch; horizon = int(config["training"]["horizon"])
    teacher_state = teacher_prefix(left, right, modulus, starts, width); plane = bits(teacher_state, width)
    losses = []; exacts = []; residue_terms = []; quotient_terms = []; diagnostics = []
    optimizer.zero_grad(set_to_none=True); teacher.width = width
    for offset in range(horizon):
        flags = [(value >> (width - 1 - start - offset)) & 1 for value, start in zip(left, starts, strict=True)]
        raw = [2 * state + flag * other for state, flag, other in zip(teacher_state, flags, right, strict=True)]
        target = [value % prime for value, prime in zip(raw, modulus, strict=True)]
        qtarget = torch.tensor([value // prime for value, prime in zip(raw, modulus, strict=True)], dtype=torch.long)
        inputs = soft_features(plane, right, modulus, flags, width)
        logits, qlogits, diagnostic, survival = student.step(inputs, offset, route_mode, route_seed); diagnostics.append(diagnostic)
        with torch.no_grad(): teacher_logits, teacher_q = teacher(inputs.detach())
        exact, stats = exp300.transition_loss(logits, qlogits, target, qtarget, width,
                                              float(config["training"]["worst_bit_weight"]), float(config["training"]["quotient_weight"]))
        residue = F.mse_loss(logits, teacher_logits) / teacher_logits.square().mean().clamp_min(1e-8)
        qdistill = F.kl_div(F.log_softmax(qlogits, 1), F.softmax(teacher_q, 1), reduction="batchmean")
        target_bits = bits(target, width)
        survival_target = logits.detach().ge(0).ne(target_bits).any(1).float()
        survival_loss = F.binary_cross_entropy_with_logits(survival, survival_target)
        losses.append(exact + 0.05 * survival_loss + float(config["training"]["residue_distillation_weight"]) * residue
                      + float(config["training"]["quotient_distillation_weight"]) * qdistill)
        exacts.append(float(stats["batch_exact"])); residue_terms.append(residue); quotient_terms.append(qdistill)
        plane = hard_straight_through(logits); teacher_state = target
    objective = torch.stack(losses).mean(); objective.backward()
    gradient = float(torch.nn.utils.clip_grad_norm_(student.parameters(), float(config["training"]["gradient_clip"]))); optimizer.step()
    return {"loss": float(objective.detach()), "gradient_norm": gradient, "mean_exact": float(np.mean(exacts)),
            "residue_distillation": float(torch.stack(residue_terms).mean()), "quotient_distillation": float(torch.stack(quotient_terms).mean()),
            "student_calls": horizon, "teacher_calls": horizon, "routed_examples": sum(row["routed"] for row in diagnostics),
            "eligible_examples": sum(row["eligible"] for row in diagnostics), "proposal_calls": sum(row["proposal_calls"] for row in diagnostics),
            "exact_loss_calls": horizon, "route_mode": route_mode,
            "mean_energy_entropy": float(np.mean([row["energy_entropy"] for row in diagnostics])),
            "mean_delta_norm": float(np.mean([row.get("delta_norm", 0.0) for row in diagnostics]))}


def train_arm(config: dict[str, Any], exp300, pruned: dict[str, torch.Tensor], checkpoint: Path, splits,
              root: int, arm: str, updates: int, output: Path) -> tuple[RoutedAntitheticQuorumStudent, dict[str, Any]]:
    random.seed(root + 694); np.random.seed(root + 694); torch.manual_seed(root + 694)
    model = new_student(config, pruned, root + 694900); initial = state_hash(model.state_dict()); timer = Timer(); history = []
    teacher = new_teacher(config, checkpoint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates, eta_min=float(config["training"]["minimum_learning_rate"]))
    samplers = {w: exp300.BigIntSampler(splits[w]["train"], root + int(config["data"]["training_seed_offset"]) + w) for w in set(config["data"]["curriculum"])}
    starts = {w: random.Random(root + int(config["data"]["training_seed_offset"]) + w + 1) for w in samplers}
    route_mode = str(config["routes"][arm]["routing"]); model.train()
    for update in range(1, updates + 1):
        tick = Timer(); width = int(config["data"]["curriculum"][(update - 1) % 4])
        left, right, modulus = samplers[width].pairs(int(config["training"]["batch_size"]))
        begin = [starts[width].randrange(0, width - int(config["training"]["horizon"]) + 1) for _ in left]
        stats = train_update(config, exp300, model, teacher, optimizer, (left, right, modulus, begin), width, route_mode, root + update)
        scheduler.step(); record = {"update": update, "width": width, "learning_rate": scheduler.get_last_lr()[0], "timing": tick.finish(), **stats}; history.append(record)
        if update == 1 or update == updates or update % int(config["training"]["log_every_updates"]) == 0:
            print(json.dumps({"arm": arm, **record}), flush=True)
    model.eval(); arm_dir = output / "arms" / arm; arm_dir.mkdir(parents=True, exist_ok=True); path = arm_dir / "model.safetensors"
    save_file({key: value.detach().contiguous() for key, value in model.state_dict().items()}, path)
    reloaded = new_student(config, load_file(path), root + 694900)
    keys = ("student_calls", "teacher_calls", "routed_examples", "eligible_examples", "proposal_calls", "exact_loss_calls")
    return model, {"arm": arm, "initial_state_sha256": initial, "parameters": model.parameter_count(),
                   "training": {"updates": updates, "curriculum": list(config["data"]["curriculum"]),
                                **{key: sum(int(row[key]) for row in history) for key in keys}, "history": history, "resource": timer.finish()},
                   "checkpoint": {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size},
                   "reload_exact": all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), reloaded.state_dict().values(), strict=True))}


def decode(rows: torch.Tensor) -> list[int]:
    return [sum(int(bit) << index for index, bit in enumerate(row)) for row in rows.tolist()]


@torch.inference_mode()
def rollout(model: RoutedAntitheticQuorumStudent, left: list[int], right: list[int], modulus: list[int], width: int,
            route_mode: str = "learned-survival-q", intervention: str = "normal", route_seed: int = 694123) -> dict[str, Any]:
    state_bits = torch.zeros((len(left), width), dtype=torch.bool); left_bits = bits(left, width)
    invariant = torch.empty((len(left), 6, width)); invariant[:, 1] = bits(right, width).mul(2).sub(1)
    invariant[:, 2] = bits(modulus, width).mul(2).sub(1); invariant[:, 4].fill_(-1); invariant[:, 4, 0] = 1
    invariant[:, 5].fill_(-1); invariant[:, 5, -1] = 1
    teacher = [0] * len(left); first = [None] * len(left); invalid = [False] * len(left); diagnostics = []; timer = Timer()
    for position in range(width):
        invariant[:, 0] = state_bits.float().mul(2).sub(1); invariant[:, 3] = left_bits[:, width - 1 - position, None].mul(2).sub(1)
        logits, _, diagnostic, _ = model.step(invariant, position, route_mode, route_seed, intervention); diagnostics.append(diagnostic)
        state_bits = logits.ge(0); values = decode(state_bits); flags = [int(value) for value in left_bits[:, width - 1 - position]]
        teacher = [(2 * state + flag * other) % prime for state, flag, other, prime in zip(teacher, flags, right, modulus, strict=True)]
        for index, (guess, truth, prime) in enumerate(zip(values, teacher, modulus, strict=True)):
            if guess != truth and first[index] is None: first[index] = position
            invalid[index] = invalid[index] or guess >= prime
    prediction = decode(state_bits); target = [(a * b) % p for a, b, p in zip(left, right, modulus, strict=True)]
    correct = [guess == truth for guess, truth in zip(prediction, target, strict=True)]; errors = [value for value in first if value is not None]
    return {"cases": len(left), "correct": sum(correct), "exact_accuracy": sum(correct) / max(1, len(correct)), "correct_vector": correct,
            "prediction_sha256": canonical_hash(prediction), "invalids": sum(invalid), "any_invalid_residue_rate": sum(invalid) / max(1, len(invalid)),
            "mean_first_error_position": None if not errors else float(np.mean(errors)),
            "routing": {"eligible": sum(row["eligible"] for row in diagnostics), "routed": sum(row["routed"] for row in diagnostics),
                        "proposal_calls": sum(row["proposal_calls"] for row in diagnostics),
                        "route_histogram_sha256": canonical_hash([row["route_index_sha256"] for row in diagnostics])},
            "resource": timer.finish()}


def transition_eval(config, exp300, model, split, root, width, cases, route_mode) -> dict[str, Any]:
    model.width = width; model.core.width = width; timer = Timer()
    view = ArmView(model, route_mode, root + width)
    value = exp300.evaluate_transitions(view, exp300.BigIntSampler(split["test"], root + int(config["data"]["test_seed_offset"]) + width), cases)
    value["resource"] = timer.finish(); return value


def paired_rollouts(config, exp300, models, split, root, width, cases) -> dict[str, Any]:
    sampler = exp300.BigIntSampler(split["test"], root + int(config["data"]["test_seed_offset"]) + width + 1)
    left, right, modulus = sampler.pairs(cases); results = {}
    for arm in TRAINED_ARMS:
        results[arm] = rollout(models[arm], left, right, modulus, width, str(config["routes"][arm]["routing"]), route_seed=root + width)
    paired = paired_counts(results["primary"]["correct_vector"], results["random_routing_control"]["correct_vector"])
    delta = [float(a) - float(b) for a, b in zip(results["primary"]["correct_vector"], results["random_routing_control"]["correct_vector"], strict=True)]
    paired["confidence_sequence"] = anytime_hoeffding_interval(delta, (1.0 - float(config["evaluation"]["confidence_level"])) / int(config["evaluation"]["holm_family_size"]))
    return {"arms": results, "paired": paired}


def candidate_inputs(root: int) -> list[tuple[str, str, str, int]]:
    source = ROOT / "sources/github/modular-arithmetic-challenge/src"; sys.path.insert(0, str(source))
    try:
        from modchallenge.config import EvalConfig
        from modchallenge.testgen.generator import generate_private_test_set
        test_set = generate_private_test_set(hashlib.sha256(f"exp694-c5-{root}".encode()).digest(), EvalConfig())
        return [(case.a, case.b, case.p, tier.tier_id) for tier in test_set.tiers for case in tier.cases]
    finally:
        sys.path.remove(str(source))


@torch.inference_mode()
def deployed_route(config, checkpoint: Path, root: int, limit: int | None = None) -> dict[str, Any]:
    rows = candidate_inputs(root); rows = rows if limit is None else rows[:limit]
    candidate = exp687.load_module(f"candidate005_exp694_{root}_{limit}_{random.randrange(1 << 20)}", ROOT / "submission/model.py")
    incumbent = candidate.Tier2HornerModel(); incumbent.load(str(ROOT / "submission"))
    model = new_student(config, load_file(checkpoint), root + 694900).eval(); timer = Timer()
    lower = [(a, b, p) for a, b, p, _ in rows if int(p).bit_length() <= 128]
    wide = [row for row in rows if 128 < int(row[2]).bit_length() <= 256]; upper = [row for row in rows if int(row[2]).bit_length() > 256]
    low_out = incumbent.predict_digits_batch(lower) if lower else []
    mod = [int(row[2]) for row in wide]
    wide_result = rollout(model, [int(row[0]) % p for row, p in zip(wide, mod, strict=True)],
                          [int(row[1]) % p for row, p in zip(wide, mod, strict=True)], mod, 256,
                          "learned-survival-q", route_seed=root) if wide else {"prediction_sha256": canonical_hash([]), "routing": {"eligible": 0, "routed": 0, "proposal_calls": 0}}
    resource = timer.finish()
    return {"cases": len(rows), "complete_1100_case_route": len(rows) == 1100, "lower_incumbent_cases": len(lower),
            "w256_specialist_cases": len(wide), "above_w256_zero_fallback_cases": len(upper),
            "output_count": len(low_out) + len(wide) + len(upper), "lower_output_sha256": canonical_hash(low_out),
            "wide_output_sha256": wide_result["prediction_sha256"], "routing": wide_result["routing"],
            "labels_available_to_timed_route": False, "operand_swap_route": False, "resource": resource}


class Experiment:
    def __init__(self, config_path: Path, output: Path, mode: str, root: int | None) -> None:
        self.config_path = config_path.resolve(); self.config = yaml.safe_load(self.config_path.read_text()); self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True); self.mode = mode; self.root = root; self.checkpoint = (ROOT / self.config["source_checkpoint"]).resolve()
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
        if git_output("-C", "sources/github/modular-arithmetic-challenge", "rev-parse", "HEAD") != CHALLENGE_COMMIT:
            raise RuntimeError("challenge commit mismatch")

    def identity(self) -> dict[str, Any]:
        return {"repository_commit": git_output("rev-parse", "HEAD"), "challenge_commit": CHALLENGE_COMMIT,
                "config_sha256": sha256(self.config_path), "runner_sha256": sha256(Path(__file__)),
                "test_sha256": sha256(ROOT / "tests/test_exp_694_routed_antithetic_quorum.py"),
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
        keys = ("updates", "curriculum", "student_calls", "teacher_calls", "routed_examples", "eligible_examples", "proposal_calls", "exact_loss_calls")
        if records["primary"]["initial_state_sha256"] != records["random_routing_control"]["initial_state_sha256"] or any(records["primary"]["training"][key] != records["random_routing_control"]["training"][key] for key in keys):
            raise RuntimeError("matched routing quota/work contract failed")
        return models, records

    def smoke(self) -> dict[str, Any]:
        old_root = self.root; self.root = 694000; self.splits = {width: make_split(self.config, self.exp300, width, self.root) for width in (64, 128, 256)}
        with tempfile.TemporaryDirectory(prefix="exp694-smoke-", dir=self.output) as directory:
            path = Path(directory); models, records = self.train_routes(path, 1)
            paired = paired_rollouts(self.config, self.exp300, models, self.splits[128], self.root, 128, 4)
            sample = self.exp300.BigIntSampler(self.splits[128]["test"], self.root + 9).pairs(4)
            interventions = {name: rollout(models["primary"], *sample, 128, "learned-survival-q", name, self.root) for name in ("route_randomization", "energy_erasure", "energy_permutation", "sign_collapse", "branch_shuffle")}
            route = deployed_route(self.config, Path(records["primary"]["checkpoint"]["path"]), self.root, 4)
            result = {"schema_version": "exp694-rootless-all-mode-smoke-v1", "status": "completed", "root": None,
                      "all_modes": ["train", "reload", "evaluate", "route-randomize", "energy-erase", "energy-permute", "sign-collapse", "branch-shuffle", "admit", "serialize"],
                      "both_arms_updated": all(records[arm]["training"]["updates"] == 1 for arm in TRAINED_ARMS),
                      "identical_initial_state": records["primary"]["initial_state_sha256"] == records["random_routing_control"]["initial_state_sha256"],
                      "equal_routed_work": records["primary"]["training"]["routed_examples"] == records["random_routing_control"]["training"]["routed_examples"],
                      "reload_exact": all(records[arm]["reload_exact"] for arm in TRAINED_ARMS), "route_cases": route["cases"], "interventions": sorted(interventions),
                      "terminal_roundtrip": True, "paired": paired}
            serialized = path / "terminal.json"; write_json(serialized, result); json.loads(serialized.read_text())
        self.root = old_root; write_json(self.output / "smoke.json", result); return result

    def calibration(self) -> dict[str, Any]:
        path = self.open_root(); timer = Timer(); models, records = self.train_routes(path, int(self.config["training"]["calibration_updates"]))
        transitions = int(self.config["calibration"]["transition_cases_per_arm_width"]); rollouts = int(self.config["calibration"]["rollout_cases_per_arm_width"])
        evals = {arm: {str(width): {"transition": transition_eval(self.config, self.exp300, model, self.splits[width], self.root, width, transitions, str(self.config["routes"][arm]["routing"])),
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
                               "equal_work": records["primary"]["training"]["proposal_calls"] == records["random_routing_control"]["training"]["proposal_calls"],
                               "exact_quota": all(item["routing"]["routed"] * 4 == item["routing"]["eligible"] for item in cold),
                               "reload_exact": all(records[arm]["reload_exact"] for arm in TRAINED_ARMS),
                               "deterministic_outputs": len({item["wide_output_sha256"] for item in cold}) == 1, "no_fidelity_reduction": True}
        admission["admitted"] = all(admission["checks"].values()); decision = "admit-root20261351" if admission["admitted"] else "park-exp694-at-root20261350-resource-admission"
        result = {"schema_version": "exp694-calibration-v1", "run_id": RUN_ID, "status": "completed", "decision": decision, "root": self.root,
                  "source": self.identity(), "records": records, "evaluations": evals, "candidate_runtime": {"warmup": warmup, "cold": cold},
                  "resource": resource, "admission": admission, "limitations": self.limitations(True)}
        write_json(path / "root-summary.json", result); write_json(self.output / "calibration.json", result); return self.log(result)

    def admitted(self) -> dict[str, Any]:
        path = self.output / "calibration.json"
        if not path.is_file(): raise RuntimeError("root20261350 admission required")
        value = json.loads(path.read_text())
        if value["decision"] != "admit-root20261351" or not value["admission"]["admitted"] or value["source"]["runner_sha256"] != sha256(Path(__file__)):
            raise RuntimeError("matching admitted root20261350 required")
        return value

    def science(self) -> dict[str, Any]:
        calibration = self.admitted(); path = self.open_root(); timer = Timer(); models, records = self.train_routes(path, int(self.config["training"]["updates"]))
        transitions = {arm: {str(width): transition_eval(self.config, self.exp300, model, self.splits[width], self.root, width, int(self.config["evaluation"]["transition_cases_per_width"]), str(self.config["routes"][arm]["routing"])) for width in (128, 256)} for arm, model in models.items()}
        w128 = paired_rollouts(self.config, self.exp300, models, self.splits[128], self.root, 128, int(self.config["evaluation"]["w128_rollout_cases"]))
        max_cases = max(self.config["evaluation"]["w256_sequential_blocks"]); full = paired_rollouts(self.config, self.exp300, models, self.splits[256], self.root, 256, max_cases); sequential = []
        for cases in self.config["evaluation"]["w256_sequential_blocks"]:
            primary = full["arms"]["primary"]["correct_vector"][:cases]; control = full["arms"]["random_routing_control"]["correct_vector"][:cases]
            pair = paired_counts(primary, control); pair["confidence_sequence"] = anytime_hoeffding_interval([float(a) - float(b) for a, b in zip(primary, control, strict=True)], 0.0025)
            sequential.append({"cases": cases, "primary_accuracy": sum(primary) / cases, "control_accuracy": sum(control) / cases, "paired": pair})
        sampler = self.exp300.BigIntSampler(self.splits[256]["test"], self.root + int(self.config["data"]["test_seed_offset"]) + 257); left, right, modulus = sampler.pairs(max_cases)
        interventions = {name: rollout(models["primary"], left, right, modulus, 256, "learned-survival-q", name, self.root) for name in ("route_randomization", "energy_erasure", "energy_permutation", "sign_collapse", "branch_shuffle")}
        primary = full["arms"]["primary"]; control = full["arms"]["random_routing_control"]; gain = primary["exact_accuracy"] - control["exact_accuracy"]
        erasures = {name: gain - (value["exact_accuracy"] - control["exact_accuracy"]) for name, value in interventions.items()}
        gates = {"healthy_control": transitions["random_routing_control"]["128"]["exact_accuracy"] >= float(self.config["gates"]["minimum_control_w128_transition"]),
                 "complete_updates": all(records[arm]["training"]["updates"] == 2400 for arm in TRAINED_ARMS),
                 "primary_w256_transition": transitions["primary"]["256"]["exact_accuracy"] >= float(self.config["gates"]["minimum_primary_w256_transition"]),
                 "primary_w256_rollout": primary["exact_accuracy"] >= float(self.config["gates"]["minimum_primary_w256_rollout"]),
                 "invalid_rate": primary["any_invalid_residue_rate"] <= float(self.config["gates"]["maximum_invalid_rate"]),
                 "paired_gain": gain >= float(self.config["gates"]["minimum_paired_gain"]),
                 "w128_retention": w128["arms"]["primary"]["exact_accuracy"] >= w128["arms"]["random_routing_control"]["exact_accuracy"] - float(self.config["gates"]["maximum_w128_loss"]),
                 "causal_erasure": gain > 0 and all(value >= float(self.config["gates"]["minimum_intervention_erasure_fraction"]) * gain for value in erasures.values()),
                 "equal_work": records["primary"]["training"]["proposal_calls"] == records["random_routing_control"]["training"]["proposal_calls"],
                 "reload_exact": all(records[arm]["reload_exact"] for arm in TRAINED_ARMS), "runtime_admitted": calibration["admission"]["admitted"],
                 "confirmation_threshold": primary["exact_accuracy"] >= float(self.config["gates"]["confirmation_admission_rollout"])}
        comparable = gates["healthy_control"]
        mechanism = comparable and all(value for key, value in gates.items() if key != "confirmation_threshold")
        decision = "advance-to-root20261352" if mechanism and gates["confirmation_threshold"] else ("non-comparative-unhealthy-control" if not comparable else "reject-exp694-c5-routed-antithetic-quorum")
        result = {"schema_version": "exp694-science-v1", "run_id": RUN_ID, "status": "completed", "root": self.root, "decision": decision,
                  "comparison_interpretable": comparable, "source": self.identity(), "records": records, "transitions": transitions, "w128": w128,
                  "w256": full, "sequential_w256": sequential, "interventions": interventions, "intervention_erasures": erasures,
                  "gates": gates, "resource": timer.finish(), "dynamic_budget": calibration["admission"], "limitations": self.limitations(False)}
        write_json(path / "root-summary.json", result); write_json(self.output / "terminal-summary.json", result); return self.log(result)

    def confirmation(self) -> dict[str, Any]:
        prior = int(self.config["roots"]["formal"]) if self.root == int(self.config["roots"]["confirmations"][0]) else int(self.config["roots"]["confirmations"][0])
        mode = "formal" if prior == int(self.config["roots"]["formal"]) else "confirmation"; prior_path = self.output / mode / f"root-{prior}" / "root-summary.json"
        if not prior_path.is_file() or json.loads(prior_path.read_text())["decision"] not in ("advance-to-root20261352", "confirmation-pass"):
            raise RuntimeError("confirmation predecessor gate failed")
        result = self.science(); result["decision"] = "confirmation-pass" if result["gates"]["confirmation_threshold"] and all(value for key, value in result["gates"].items() if key != "confirmation_threshold") else "reject-exp694-confirmation"
        write_json(self.root_dir() / "root-summary.json", result); write_json(self.output / "terminal-summary.json", result); return self.log(result)

    def independent(self) -> dict[str, Any]:
        predecessor = self.output / "confirmation" / f"root-{int(self.config['roots']['confirmations'][-1])}" / "root-summary.json"
        if not predecessor.is_file() or json.loads(predecessor.read_text())["decision"] != "confirmation-pass": raise RuntimeError("both confirmations required")
        raise RuntimeError("five-seed route remains gated until both confirmations pass")

    def log(self, result: dict[str, Any]) -> dict[str, Any]:
        experiment_id = configure_mlflow(self.config["tracking"]["experiment_name"])
        with mlflow.start_run(run_name=f"{RUN_ID}-{self.mode}-{self.root}") as active:
            mlflow.set_tags({"canonical_run_id": RUN_ID, "mode": self.mode, "root": str(self.root), "decision": result["decision"], "candidate_registration": "false", "tournament_entrant": "C5"})
            mlflow.log_dict(result, f"{self.mode}-result.json"); run_id = active.info.run_id
        result["mlflow"] = {"tracking_uri": tracking_uri(), "experiment_id": experiment_id, "run_id": run_id}; write_json(self.root_dir() / "root-summary.json", result); return result

    @staticmethod
    def limitations(calibration: bool) -> list[str]:
        return ["Calibration accuracy is route-integrity evidence, not science." if calibration else "Fresh generated identities only; checked/public/private answers were not used.",
                "Exact arithmetic and teacher outputs exist only in offline training/evaluation; deployed inference uses learned tensor operations, sparse indexing, and hard thresholding.",
                "The deterministic random control uses generic example indexing only and has the identical one-per-four quota and proposal work.",
                "C5 is independently required; its result cannot cancel or weaken C6-C7.",
                "No candidate, submission, promotion, release, or private-score claim is authorized."]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=CONFIG_PATH); parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--mode", choices=("smoke", "calibration", "formal", "confirmation", "independent"), required=True); parser.add_argument("--root", type=int); args = parser.parse_args()
    experiment = Experiment(args.config, args.output, args.mode, args.root)
    result = experiment.smoke() if args.mode == "smoke" else experiment.calibration() if args.mode == "calibration" else experiment.science() if args.mode == "formal" else experiment.confirmation() if args.mode == "confirmation" else experiment.independent()
    print(json.dumps({"mode": args.mode, "root": result.get("root"), "status": result["status"], "decision": result.get("decision"), "gates": result.get("gates"), "admission": result.get("admission", {}).get("checks")}, indent=2, sort_keys=True), flush=True); return 0


if __name__ == "__main__": raise SystemExit(main())
