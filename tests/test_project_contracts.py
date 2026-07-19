from __future__ import annotations

from pathlib import Path

import yaml

from tools.validate_project import (
    validate_checksum_file,
    validate_dataset,
    validate_knowledge,
    validate_release_boundary,
    validate_research_library,
    validate_submission_artifact,
    validate_upstream_snapshot,
)
from tools.evaluate_checked_benchmark import load_checked_test_set


ROOT = Path(__file__).resolve().parents[1]


def test_okf_knowledge_bundle_is_valid() -> None:
    assert validate_knowledge() == []


def test_public_benchmark_metadata_and_hashes_are_valid() -> None:
    assert validate_dataset() == []
    assert validate_checksum_file() == []


def test_research_paper_library_is_frozen_and_verified() -> None:
    assert validate_research_library() == []


def test_ci_hydrates_lfs_papers_without_duplicate_branch_runs() -> None:
    workflow = yaml.load(
        (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    assert workflow["on"]["push"]["branches"] == ["main"]
    checkout = next(
        step
        for step in workflow["jobs"]["validate"]["steps"]
        if step.get("uses") == "actions/checkout@v4"
    )
    assert checkout["with"]["lfs"] == "true"
    assert checkout["with"]["submodules"] == "recursive"
    install = next(
        step
        for step in workflow["jobs"]["validate"]["steps"]
        if step.get("name") == "Install project"
    )
    assert install["run"] == 'python -m pip install -e ".[dev,research]"'


def test_official_source_is_pinned() -> None:
    assert validate_upstream_snapshot() == []


def test_submission_boundary_excludes_evaluation_data() -> None:
    assert validate_release_boundary() == []
    assert validate_submission_artifact() == []


def test_checked_public_adapter_loads_the_exact_full_set() -> None:
    test_set, hashes = load_checked_test_set(
        ROOT / "data" / "external" / "public_benchmark"
    )
    assert test_set.total_cases == 1_100
    assert [len(tier.cases) for tier in test_set.tiers] == [100] * 11
    assert hashes["tier_2.jsonl"] == (
        "ae169d3acf554fdff4b8c4b13a3e518303f94d6b6d67aa5161ead327a9332fdf"
    )
