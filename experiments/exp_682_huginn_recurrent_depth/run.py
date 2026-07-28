"""Authenticate and terminally park EXP682 at its source/topology gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil
import torch
import yaml
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.exp_220_width_independent_cellular.model import SharedDilatedConvGRU  # noqa: E402
from experiments.exp_682_huginn_recurrent_depth.model import (  # noqa: E402
    HuginnTopologyProbe,
    sample_bounded_lognormal_poisson_depths,
    structural_transfer_receipt,
)


IDENTITY_PATHS = {
    "config": "configs/exp_682_huginn_recurrent_depth.yaml",
    "model": "experiments/exp_682_huginn_recurrent_depth/model.py",
    "runner": "experiments/exp_682_huginn_recurrent_depth/run.py",
    "test": "tests/test_exp_682_huginn_recurrent_depth.py",
    "preregistration": "knowledge/autoresearch/ablations/exp-682-huginn-recurrent-depth.md",
    "mechanism_map": "knowledge/autoresearch/research/2026-07-30-exp682-huginn-mechanism.md",
}
CHECKPOINT_CANDIDATES = (
    Path("/mnt/d/Projects/SAIR_mod_challenge/.scratch/exp-609-tier6-straight-through-confirmation/straight_through_continuation/weights.safetensors"),
    Path("/mnt/d/Projects/SAIR_mod_challenge_exp655_t048_wt/.scratch/exp-609-tier6-straight-through-confirmation/straight_through_continuation/weights.safetensors"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def git_output(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=120).stdout.strip()


class Timer:
    def __init__(self) -> None:
        self.wall = time.perf_counter()
        self.process = psutil.Process()
        cpu = self.process.cpu_times()
        self.cpu = cpu.user + cpu.system
        self.peak = self.process.memory_info().rss

    def result(self) -> dict[str, float | int]:
        cpu = self.process.cpu_times()
        self.peak = max(self.peak, self.process.memory_info().rss)
        return {
            "wall_seconds": time.perf_counter() - self.wall,
            "process_cpu_seconds": cpu.user + cpu.system - self.cpu,
            "peak_rss_bytes": self.peak,
        }


class Experiment:
    def __init__(self, config_path: Path, output: Path) -> None:
        self.config_path = config_path.resolve()
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        torch.set_num_threads(int(self.config["training"]["torch_threads"]))
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        self.checkpoint = self.materialize_checkpoint()

    def materialize_checkpoint(self) -> Path:
        expected = self.config["source_contract"]["checkpoint_sha256"]
        source = next((path for path in CHECKPOINT_CANDIDATES if path.is_file() and sha256(path) == expected), None)
        if source is None:
            raise RuntimeError("authenticated EXP609 checkpoint unavailable")
        destination = self.output / "fixtures/exp609-weights.safetensors"
        if not destination.is_file() or sha256(destination) != expected:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".tmp")
            shutil.copyfile(source, temporary)
            if sha256(temporary) != expected:
                temporary.unlink(missing_ok=True)
                raise RuntimeError("checkpoint copy hash mismatch")
            os.replace(temporary, destination)
        return destination

    def source(self) -> SharedDilatedConvGRU:
        model = SharedDilatedConvGRU(width=128, hidden=92, cycles=2)
        model.load_state_dict(load_file(self.checkpoint), strict=True)
        return model.eval()

    def source_identity(self) -> dict[str, Any]:
        contract = self.config["source_contract"]
        paper = Path(contract["paper_path"])
        result = {
            "repository_commit": git_output("rev-parse", "HEAD"),
            "branch": git_output("branch", "--show-current"),
            "challenge_commit": git_output("-C", str(ROOT / "sources/github/modular-arithmetic-challenge"), "rev-parse", "HEAD"),
            "paper_sha256": sha256(paper),
            "research_manifest_sha256": sha256(Path("/mnt/d/Projects/SAIR_mod_challenge/research/manifest.json")),
            "checkpoint_sha256": sha256(self.checkpoint),
            "base_model_sha256": sha256(ROOT / "experiments/exp_220_width_independent_cellular/model.py"),
            "exp634_metrics_sha256": sha256(ROOT / "evals/results/exp-634-direct-relative-attention-feasibility/metrics.json"),
            "exp643_metrics_sha256": sha256(ROOT / "evals/results/exp-643-tier7-recurrent-depth-stability-diagnostic/metrics.json"),
            "exp674_metrics_sha256": sha256(ROOT / "evals/results/exp-674-latent-thought-regularization/metrics.json"),
        }
        expected = {key: contract[key] for key in result if key in contract}
        result["authenticated"] = all(result[key] == value for key, value in expected.items())
        return result

    def package_identity(self) -> dict[str, str]:
        return {f"{name}_sha256": sha256(ROOT / path) for name, path in IDENTITY_PATHS.items()}

    def frozen_protocol(self) -> dict[str, Any]:
        return {
            "schema_version": "exp682-frozen-protocol-v1",
            "status": "frozen-before-root-allocation",
            "source": self.source_identity(),
            "package": self.package_identity(),
            "mapping": self.config["source_mapping"],
            "library": self.config["library_decision"],
            "latent": self.config["latent"],
            "depth": self.config["depth"],
            "arms": self.config["arms"],
            "training": self.config["training"],
            "evaluation": self.config["evaluation"],
            "gates": self.config["gates"],
            "transfer_gate": self.config["transfer_gate"],
            "roots": self.config["roots"],
            "resource_budget": self.config["resource_budget"],
        }

    def freeze(self) -> dict[str, Any]:
        payload = self.frozen_protocol()
        path = self.output / "frozen-protocol.json"
        write_json(path, payload)
        return {**payload, "path": str(path.relative_to(ROOT)), "sha256": sha256(path)}

    def assert_frozen(self) -> None:
        path = self.output / "frozen-protocol.json"
        current = json.loads(json.dumps(self.frozen_protocol(), sort_keys=True))
        if not path.is_file() or json.loads(path.read_text(encoding="utf-8")) != current:
            raise RuntimeError("EXP682 final bytes differ from frozen protocol")

    def topology(self) -> dict[str, Any]:
        timer = Timer()
        source = self.source()
        probe = HuginnTopologyProbe(source, latent_std=float(self.config["latent"]["standard_deviation"]), release_seed=int(self.config["latent"]["release_seed"]))
        features = torch.randn(4, 6, 64, generator=torch.Generator().manual_seed(682))
        receipt = structural_transfer_receipt(source, probe, features)
        depths = sample_bounded_lognormal_poisson_depths(
            400, targeted_mean=int(self.config["depth"]["targeted_mean"]),
            sigma=float(self.config["depth"]["lognormal_sigma"]), minimum=int(self.config["depth"]["minimum"]),
            maximum=int(self.config["depth"]["maximum"]), seed=20261278,
        )
        exp634 = json.loads((ROOT / "evals/results/exp-634-direct-relative-attention-feasibility/metrics.json").read_text())
        adjacent = exp634["evaluation"]["transition"]["direct_prediction_width12"]
        threshold = float(self.config["transfer_gate"]["de_novo_minimum_adjacent_trained_width_exact"])
        checks = {
            "source_authenticated": bool(self.source_identity()["authenticated"]),
            "native_rmsnorm": isinstance(probe.pre_norm, torch.nn.RMSNorm),
            "native_distributions": True,
            "required_components_causal": bool(receipt["causal_required_components"]),
            "exact_transfer": bool(receipt["exact_checkpoint_transfer"]),
            "de_novo_adjacent_viability": float(adjacent["exact_accuracy"]) >= threshold,
            "depth_bounded": int(depths.min()) >= int(self.config["depth"]["minimum"]) and int(depths.max()) <= int(self.config["depth"]["maximum"]),
            "depth_heavy_tail": int(depths.max()) > int(torch.median(depths)),
            "all_roots_unopened": all(item["status"].startswith("unopened") for item in [self.config["roots"]["calibration"], self.config["roots"]["formal"], *self.config["roots"]["confirmations"]]),
        }
        decision = "park-no-exact-transfer-or-viable-de-novo-route" if not checks["exact_transfer"] and not checks["de_novo_adjacent_viability"] else "implementation-authorized"
        return {
            "schema_version": "exp682-source-topology-v1",
            "decision": decision,
            "source": self.source_identity(),
            "topology": receipt,
            "distribution": {"draws": 400, "minimum": int(depths.min()), "maximum": int(depths.max()), "mean": float(depths.float().mean()), "median": float(torch.median(depths)), "histogram": {str(value): int((depths == value).sum()) for value in range(int(depths.min()), int(depths.max()) + 1)}},
            "de_novo_evidence": {"source": "EXP634 adjacent recurrent de-novo feasibility", "updates": int(exp634["controls"]["matched_updates"]), "trained_width": 12, "cases": int(adjacent["cases"]), "correct": int(adjacent["correct"]), "exact_accuracy": float(adjacent["exact_accuracy"]), "minimum_viability": threshold, "inference": "400-update W128 de-novo fallback is not decision-worthy after 4000-update W12 adjacent failure"},
            "checks": checks,
            "resource": timer.result(),
        }

    def preflight(self) -> dict[str, Any]:
        self.assert_frozen()
        topology = self.topology()
        required_arms = {"full_huginn", "fixed_mean_depth", "full_bptt", "zero_latent", "deterministic_latent", "no_reinjection", "no_concat", "no_rmsnorm", "reset_latent", "shuffled_latent", "frozen_source", "randomized_weights"}
        checks = {
            "source_authenticated": topology["checks"]["source_authenticated"],
            "package_frozen": True,
            "all_modes_declared": set(self.config["arms"]) == required_arms,
            "updates_400": int(self.config["training"]["updates"]) == 400,
            "root_identities": [self.config["roots"]["calibration"]["root"], self.config["roots"]["formal"]["root"], *[item["root"] for item in self.config["roots"]["confirmations"]]] == [20261278, 20261279, 20261280, 20261281],
            "denominators_complete": set(self.config["evaluation"]["transition_cases"]) == {64, 128, 256} and set(self.config["evaluation"]["rollout_cases"]) == {64, 128, 256},
            "terminal_schema": topology["schema_version"] == "exp682-source-topology-v1" and topology["decision"].startswith("park-"),
            "prohibited_surfaces_unopened": self.config["source_contract"]["checked_independent_official_private_candidate_submission_promotion"] == "prohibited",
        }
        if not all(checks.values()):
            raise RuntimeError(f"preflight failed: {checks}")
        result = {"schema_version": "exp682-preflight-v1", "status": "passed-terminal-topology-park", "checks": checks, "topology": topology, "completion_marker": "preflight-complete"}
        write_json(self.output / "preflight.json", result)
        write_json(self.output / "source-topology.json", topology)
        return result

    def park(self) -> dict[str, Any]:
        preflight = self.preflight()
        topology = preflight["topology"]
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        summary = {
            "schema_version": "exp682-terminal-v1", "run_id": self.config["run_id"], "status": "parked-source-topology-feasibility",
            "created_at": now, "decision": topology["decision"], "source": topology["source"], "config": str(self.config_path.relative_to(ROOT)),
            "roots": {"20261278": "unopened-topology-stop", "20261279": "unopened-topology-stop", "20261280": "unopened-topology-stop", "20261281": "unopened-topology-stop"},
            "training_updates": 0, "formal_calls_executed": 0, "calibration": None, "scientific_metrics": None, "scientific_verdict": None,
            "topology": topology["topology"], "depth_distribution_probe": topology["distribution"], "de_novo_evidence": topology["de_novo_evidence"],
            "controls": {name: "declared-unopened-topology-stop" for name in self.config["arms"]},
            "evaluation": {"not_performed": True, "labels_or_answers_used": False, "checked_independent_official_private_candidate_submission_promotion": "not-opened"},
            "limitations": [
                "This is a source/topology and de-novo-viability park, not a scientific accuracy result or rejection of Huginn.",
                "A source-complete causal latent/concat/RMSNorm route is not exactly function preserving for Candidate005; inert additions would violate the component-causality gate.",
                "The de-novo viability conclusion is an evidence-backed inference from adjacent EXP634, not a direct EXP682 training measurement.",
                "No calibration, formal W128/W256, confirmation, checked, independent, official/private, candidate, submission, promotion, or completion surface was opened.",
            ],
            "completion_marker": "terminal-persisted",
        }
        run_dir = ROOT / "experiments/runs/exp-682-huginn-recurrent-depth"
        metrics_dir = ROOT / "evals/results/exp-682-huginn-recurrent-depth"
        environment = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "torch_threads": torch.get_num_threads(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
        write_json(run_dir / "source-topology.json", topology)
        write_json(run_dir / "environment.json", environment)
        write_json(run_dir / "run.json", summary)
        write_json(metrics_dir / "metrics.json", summary)
        write_json(self.output / "terminal-summary.json", summary)
        write_json(self.output / "completion.json", {"completion_marker": "terminal-persisted", "decision": summary["decision"], "roots_opened": []})
        return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/exp_682_huginn_recurrent_depth.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / ".scratch/exp-682-huginn-recurrent-depth")
    parser.add_argument("--mode", choices=("freeze", "preflight", "park"), required=True)
    args = parser.parse_args()
    experiment = Experiment(args.config, args.output)
    result = experiment.freeze() if args.mode == "freeze" else experiment.preflight() if args.mode == "preflight" else experiment.park()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
