from __future__ import annotations

from tools.validate_project import (
    validate_checksum_file,
    validate_dataset,
    validate_knowledge,
    validate_release_boundary,
    validate_upstream_snapshot,
)


def test_okf_knowledge_bundle_is_valid() -> None:
    assert validate_knowledge() == []


def test_public_benchmark_metadata_and_hashes_are_valid() -> None:
    assert validate_dataset() == []
    assert validate_checksum_file() == []


def test_official_source_is_pinned() -> None:
    assert validate_upstream_snapshot() == []


def test_submission_boundary_excludes_evaluation_data() -> None:
    assert validate_release_boundary() == []
