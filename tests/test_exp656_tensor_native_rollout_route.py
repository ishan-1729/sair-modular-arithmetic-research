from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments/exp_656_tensor_native_rollout_route/run.py"
CONFIG = ROOT / "configs/exp_656_tensor_native_rollout_route.yaml"
CANDIDATE005 = (
    ROOT
    / "experiments/runs/candidate-005-tier6-conditional-commutative-integration/artifact"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_is_input_only_and_source_pinned() -> None:
    module = load_module("exp656_config_test", RUNNER)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["run_id"] == module.RUN_ID
    assert config["benchmark"] == {
        "widths": [128, 256],
        "batch_size": 16,
        "warmups_per_route": 1,
        "measured_repeats": 4,
        "torch_threads": 4,
        "route_order": "alternating-legacy-first-then-tensor-first",
    }
    assert config["gates"]["maximum_width256_tensor_to_legacy_wall_ratio"] == 0.8
    preserved_config = {**config, "source": dict(config["source"])}
    preserved_config["source"].update(
        {
            "candidate_model": str(CANDIDATE005.relative_to(ROOT) / "model.py"),
            "candidate_manifest": str(
                CANDIDATE005.relative_to(ROOT) / "manifest.json"
            ),
            "checkpoint": str(
                CANDIDATE005.relative_to(ROOT) / "weights.safetensors"
            ),
        }
    )
    assert module.validate_source(preserved_config) == {
        "candidate_model_sha256": config["source"]["candidate_model_sha256"],
        "candidate_manifest_sha256": config["source"]["candidate_manifest_sha256"],
        "checkpoint_sha256": config["source"]["checkpoint_sha256"],
    }
    assert "data/external" not in CONFIG.read_text(encoding="utf-8")


class RecordingNetwork:
    def __init__(self) -> None:
        self.features: list[torch.Tensor] = []

    def __call__(self, features: torch.Tensor):
        self.features.append(features.clone())
        logits = features[:, 0, :] + 0.125 * features[:, 1, :]
        quotient = torch.zeros((features.shape[0], 3))
        return logits, quotient


def test_cached_route_reproduces_every_legacy_feature_plane() -> None:
    module = load_module("exp656_feature_test", RUNNER)
    candidate = load_module("candidate005_for_exp656_feature_test", CANDIDATE005 / "model.py")
    multiplier = [173, 91]
    multiplicand = [149, 77]
    modulus = [251, 239]
    legacy_network = RecordingNetwork()
    native_network = RecordingNetwork()
    legacy = module.legacy_rollout(
        candidate, legacy_network, multiplier, multiplicand, modulus, 8
    )
    native = module.tensor_native_rollout(
        candidate, native_network, multiplier, multiplicand, modulus, 8
    )
    module.assert_parity(legacy, native)
    assert len(legacy_network.features) == len(native_network.features) == 8
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            legacy_network.features, native_network.features, strict=True
        )
    )


def test_cached_route_is_exact_on_frozen_candidate_cell() -> None:
    module = load_module("exp656_frozen_test", RUNNER)
    candidate_module = load_module(
        "candidate005_for_exp656_frozen_test", CANDIDATE005 / "model.py"
    )
    candidate = candidate_module.Tier2HornerModel()
    candidate.load(str(CANDIDATE005))
    assert candidate.tier6_network is not None
    cases = module.make_cases(width=16, batch_size=2, seed=20260721)
    legacy = module.legacy_rollout(
        candidate_module, candidate.tier6_network, *cases, 16
    )
    native = module.tensor_native_rollout(
        candidate_module, candidate.tier6_network, *cases, 16
    )
    module.assert_parity(legacy, native)
    assert legacy[0] == native[0]
    assert torch.equal(legacy[1], native[1])


def test_decision_requires_parity_retention_and_twenty_percent_reduction() -> None:
    module = load_module("exp656_decision_test", RUNNER)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    def record(ratio: float) -> dict:
        return {
            "summary": {
                "all_exact_output_parity": True,
                "all_exact_minimum_margin_parity": True,
                "tensor_to_legacy_wall_ratio": ratio,
            }
        }

    decision, gates = module.decide({"128": record(1.0), "256": record(0.8)}, config)
    assert decision == "advance-tensor-native-route"
    assert gates["all_passed"] is True
    decision, gates = module.decide({"128": record(0.9), "256": record(0.81)}, config)
    assert decision == "reject-tensor-native-route"
    assert gates["width256_minimum_20_percent_wall_reduction"] is False


def test_durable_artifacts_if_run_completed() -> None:
    run_path = ROOT / "experiments/runs/exp-656-tensor-native-rollout-route/run.json"
    if not run_path.exists():
        return
    run = json.loads(run_path.read_text(encoding="utf-8"))
    metrics = json.loads((ROOT / run["metrics"]).read_text(encoding="utf-8"))
    environment = json.loads((ROOT / run["environment"]).read_text(encoding="utf-8"))
    assert run["run_id"] == metrics["run_id"] == "exp-656-tensor-native-rollout-route"
    assert run["status"] == metrics["status"] == "completed"
    assert metrics["scientific_accuracy_reported"] is False
    assert metrics["checkpoint_sha256"] == run["source"]["checkpoint_sha256"]
    assert metrics["evaluation"] == {
        "not_performed": True,
        "reason": "Input-only route performance A/B; no answer labels or oracle were generated or read.",
        "labels_or_oracle_present": False,
    }
    assert metrics["controls"]["matched_input_only_route_comparison"] is True
    assert metrics["runtime"] == metrics["resource"]
    assert run["source"]["challenge_commit"] == (
        "82510bba00a1126649bd76dd1a451f14d0b3eb60"
    )
    assert run["source"]["dirty"] is True
    assert run["artifacts"] == {
        "checkpoint": "submission/weights.safetensors",
        "metrics": "evals/results/exp-656-tensor-native-rollout-route/metrics.json",
    }
    assert metrics["decision"] in {
        "advance-tensor-native-route",
        "reject-tensor-native-route",
    }
    assert set(metrics["benchmark"]) == {"128", "256"}
    for width in ("128", "256"):
        assert len(metrics["benchmark"][width]["repeats"]) == 4
        assert metrics["benchmark"][width]["summary"]["all_exact_output_parity"]
        assert metrics["benchmark"][width]["summary"][
            "all_exact_minimum_margin_parity"
        ]
    assert environment["torch_threads"] == 4
    assert run["source"]["checkpoint_sha256"] == (
        "2f6ac3199d0fd9e16afae165c2d4a16919e6c8bf4545363a6f12d077279d27ca"
    )
    assert run["schema_normalization"]["benchmark_fields_changed"] is False
    assert metrics["schema_normalization"]["benchmark_fields_changed"] is False
