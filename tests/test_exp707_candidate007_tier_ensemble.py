from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import yaml

from experiments.exp_707_candidate007_tier_ensemble import run


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/exp_707_candidate007_tier_ensemble.yaml"
RUN = ROOT / "experiments/runs/exp-707-candidate007-tier-ensemble/run.json"
RAW = ROOT / "experiments/runs/exp-707-candidate007-tier-ensemble/raw/router-result.json"
METRICS = ROOT / "evals/results/exp-707-candidate007-tier-ensemble/metrics.json"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_sources_and_expert_order_are_exact() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    assert config["run_id"] == run.RUN_ID
    assert config["experts"]["order"] == [
        "teacher_cosine",
        "straight_through_cosine",
        "teacher_restarts",
        "straight_through_restarts",
    ]
    assert file_hash(ROOT / config["source"]["paired_evidence"]) == config["source"]["paired_evidence_sha256"]
    for arm in config["experts"]["order"]:
        assert file_hash(ROOT / config["experts"][arm]["path"]) == config["experts"][arm]["sha256"]


def test_router_features_are_input_only_and_fixed_width() -> None:
    rows = [[5, 7, 11, 2, 0, None, False], [9, 3, 13, 1, 0, None, False]]
    features = run.router_features(rows, width=16, chunk_bits=4)
    changed_targets = [[*row[:3], 999, 888, 0, True] for row in rows]
    assert torch.equal(features, run.router_features(changed_targets, width=16, chunk_bits=4))
    assert features.shape == (2, 20)


def test_modulus_group_split_is_disjoint() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    _, rows, _ = run.load_evidence(config)
    router = config["router"]
    groups: dict[int, set[int]] = {index: set() for index in range(router["bucket_count"])}
    for row in rows:
        groups[run.bucket(router["split_namespace"], int(row[2]), router["bucket_count"])].add(int(row[2]))
    for left in groups:
        for right in groups:
            if left != right:
                assert groups[left].isdisjoint(groups[right])


def test_durable_result_rejects_candidate007_without_posthoc_evaluation() -> None:
    result = json.loads(RAW.read_text())
    run_record = json.loads(RUN.read_text())
    metrics = json.loads(METRICS.read_text())
    assert result["status"] == "completed-negative-router-gate"
    assert result["splits"]["fit"]["gain_over_teacher_cosine"] == 21
    assert result["splits"]["calibration"]["gain_over_teacher_cosine"] == -5
    assert result["splits"]["sealed"]["gain_over_teacher_cosine"] == -9
    assert result["gate"] == {
        "minimum_sealed_gain": 2,
        "actual_sealed_gain": -9,
        "passed": False,
    }
    assert result["router"] is None
    assert metrics["decision"] == "reject-candidate007-retain-candidate006"
    assert run_record["status"] == "completed"
    assert run_record["decision"] == "reject-candidate007-retain-candidate006"
    assert metrics["downstream_evaluation"]["candidate_artifact_built"] is False
    assert metrics["downstream_evaluation"]["five_seed_evaluation_executed"] is False
    assert not (ROOT / "experiments/runs/exp-707-candidate007-tier-ensemble/artifact").exists()


def test_router_replay_is_deterministic(tmp_path: Path) -> None:
    replay = run.run(CONFIG, tmp_path)
    durable = json.loads(RAW.read_text())
    assert replay["status"] == durable["status"]
    assert replay["feature_sha256"] == durable["feature_sha256"]
    assert replay["correctness_sha256"] == durable["correctness_sha256"]
    assert replay["selected_margin"] == durable["selected_margin"]
    assert replay["splits"] == durable["splits"]
