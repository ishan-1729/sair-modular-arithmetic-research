from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from safetensors.torch import load_file
import yaml

from tools import build_candidate006_tier7_artifact as builder
from tools import build_candidate006_tensor_native_artifact as runtime_builder
from tools.cloud_experiment.manifest import load_manifest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidate_model_adds_only_frozen_tier7_route() -> None:
    incumbent = (builder.INCUMBENT / "model.py").read_text(encoding="utf-8")
    candidate = builder.candidate_model(incumbent)
    assert incumbent == (builder.INCUMBENT / "model.py").read_text(encoding="utf-8")
    assert "TIER7_WIDTH = 256" in candidate
    assert "self.tier7_network" in candidate
    assert "if modulus_value > (1 << TIER7_WIDTH) - 1" in candidate
    assert "item[1] %" not in candidate.split("if tier7_group:", 1)[1]


def test_builder_preserves_candidate005_tensors_and_prefixes_specialist(tmp_path: Path) -> None:
    output = tmp_path / "candidate"
    receipt = builder.build(output)
    incumbent = load_file(str(builder.INCUMBENT / "weights.safetensors"))
    specialist = load_file(str(builder.TIER7_SOURCE))
    built = load_file(str(output / "weights.safetensors"))
    assert receipt["parameters"] == 569873
    assert set(built) == set(incumbent) | {f"tier7.{key}" for key in specialist}
    assert all(torch.equal(built[key], value) for key, value in incumbent.items())
    assert all(torch.equal(built[f"tier7.{key}"], value) for key, value in specialist.items())


def test_loading_candidate_does_not_pollute_three_file_artifact(tmp_path: Path) -> None:
    output = tmp_path / "candidate"
    builder.build(output)
    runner = load_module(
        "exp704_runner_no_bytecode",
        ROOT / "experiments/exp_704_candidate005_tier7_exp701_integration/run.py",
    )
    runner.load_candidate(output, "exp704_no_bytecode_candidate")
    assert {path.name for path in output.iterdir()} == {
        "model.py",
        "manifest.json",
        "weights.safetensors",
    }


def test_randomized_control_changes_only_tier7_float_tensors(tmp_path: Path) -> None:
    output = tmp_path / "randomized"
    builder.build(output, 5430431669808135325)
    incumbent = load_file(str(builder.INCUMBENT / "weights.safetensors"))
    specialist = load_file(str(builder.TIER7_SOURCE))
    built = load_file(str(output / "weights.safetensors"))
    assert all(torch.equal(built[key], value) for key, value in incumbent.items())
    changed = [
        key
        for key, value in specialist.items()
        if value.is_floating_point() and value.numel() >= 2 and not torch.equal(built[f"tier7.{key}"], value)
    ]
    assert changed


def test_route_uses_tier7_only_through_256_bits_and_retains_upper_zero(tmp_path: Path) -> None:
    output = tmp_path / "candidate"
    builder.build(output)
    module = load_module("exp704_route_candidate", output / "model.py")
    model = module.Tier2HornerModel()
    sentinel = object()
    model.network = sentinel
    model.tier3_network = sentinel
    model.tier4_network = sentinel
    model.tier5_network = sentinel
    model.tier6_network = sentinel
    model.tier7_network = sentinel
    calls: list[tuple[list[int], list[int], list[int]]] = []

    def fake_rollout(left: list[int], right: list[int], modulus: list[int]) -> list[int]:
        calls.append((left, right, modulus))
        return [17] * len(left)

    model._tier7_rollout = fake_rollout
    p256 = (1 << 255) + 19
    p257 = (1 << 256) + 297
    output_rows = model.predict_digits_batch(
        [(str(p256 + 7), str(p256 + 11), str(p256)), ("9", "13", str(p257))]
    )
    assert output_rows == [[17], [0]]
    assert calls == [([7], [11], [p256])]


def test_tensor_native_tier7_rollout_matches_exp704_legacy_bytes(tmp_path: Path) -> None:
    output = tmp_path / "candidate"
    runtime_builder.build(output)
    module = load_module("exp705_tensor_native_candidate", output / "model.py")
    model = module.Tier2HornerModel()
    model.load(str(output))
    left = [(1 << 254) + 123, (1 << 252) + 77]
    right = [(1 << 250) + 19, (1 << 249) + 31]
    modulus = [(1 << 255) + 765, (1 << 253) + 447]
    state = [0] * len(left)
    with torch.inference_mode():
        for bit_index in range(module.TIER7_WIDTH - 1, -1, -1):
            bit = [(value >> bit_index) & 1 for value in left]
            logits, _ = model.tier7_network(
                module._bigint_features(
                    state, right, modulus, bit, module.TIER7_WIDTH
                )
            )
            state = module._decode_bigint_logits(logits)
        optimized = model._tier7_rollout(left, right, modulus)
    assert optimized == state


def test_config_and_cloud_manifest_are_frozen_and_valid() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/exp_704_candidate005_tier7_exp701_integration.yaml").read_text(encoding="utf-8")
    )
    assert config["specialist"]["arm"] == "teacher_cosine"
    assert config["specialist"]["sha256"] == builder.TIER7_SHA256
    assert len(config["seeds"]) == 5
    assert config["evaluation"]["parity_tiers"] == [1, 2, 3, 4, 5, 6]
    assert config["gates"]["minimum_aggregate_tier7_accuracy"] == 0.50
    reused = ROOT / config["evidence_reuse"]["path"]
    assert builder.sha256(reused) == config["evidence_reuse"]["sha256"]
    manifest = load_manifest(ROOT / "configs/cloud_runner/exp_704_candidate006_local.yaml")
    assert manifest.data["provider"]["paid"] is False
    assert manifest.data["provider"]["expected_hardware"] == {
        "accelerator": "CPU",
        "accelerator_count": 0,
    }
    assert [phase["id"] for phase in manifest.data["execution"]["phases"]] == [
        "build-artifacts",
        "fresh-full-evaluation",
        "summarize",
    ]
    replay = load_manifest(
        ROOT / "configs/cloud_runner/exp_704_candidate006_local_replay.yaml"
    )
    assert replay.data["experiment"]["id"].endswith("-replay")
    included = set(replay.data["package"]["include"])
    assert config["specialist"]["path"] in included
    assert config["evidence_reuse"]["path"] in included
