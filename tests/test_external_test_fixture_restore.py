from __future__ import annotations

import os
from pathlib import Path

import pytest

from tools.restore_external_test_fixtures import restore


def _git_worktree(path: Path) -> None:
    path.mkdir()
    (path / ".git").write_text("gitdir: test\n", encoding="utf-8")


def test_restore_requires_external_dot_scratch(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    _git_worktree(repository)
    source = repository / "nested" / ".scratch"
    source.mkdir(parents=True)
    with pytest.raises(ValueError, match="outside the target worktree"):
        restore(repository, [source])


def test_restore_rejects_destination_root_symlink(tmp_path: Path, monkeypatch) -> None:
    repository = tmp_path / "repository"
    source_parent = tmp_path / "source"
    source = source_parent / ".scratch"
    _git_worktree(repository)
    source.mkdir(parents=True)
    fixture_root = source / "fixture"
    fixture_root.mkdir()
    (fixture_root / "evidence.json").write_text("{}\n", encoding="utf-8")
    target = repository / ".scratch"
    target.mkdir()
    os.symlink(fixture_root, target / "fixture", target_is_directory=True)
    monkeypatch.setattr(
        "tools.restore_external_test_fixtures.REQUIRED_ROOTS", ("fixture",)
    )
    monkeypatch.setattr(
        "tools.restore_external_test_fixtures.PHYSICAL_ROOTS", {"fixture"}
    )
    with pytest.raises(ValueError, match="--replace-root-symlinks"):
        restore(repository, [source])


def test_restore_materializes_and_refuses_divergent_evidence(tmp_path: Path, monkeypatch) -> None:
    repository = tmp_path / "repository"
    source_parent = tmp_path / "source"
    source = source_parent / ".scratch"
    overlay_parent = tmp_path / "overlay"
    overlay = overlay_parent / ".scratch"
    _git_worktree(repository)
    for root in (source, overlay):
        (root / "fixture").mkdir(parents=True)
    (source / "fixture" / "base.json").write_text("base\n", encoding="utf-8")
    (overlay / "fixture" / "extra.json").write_text("extra\n", encoding="utf-8")
    monkeypatch.setattr(
        "tools.restore_external_test_fixtures.REQUIRED_ROOTS", ("fixture",)
    )
    monkeypatch.setattr(
        "tools.restore_external_test_fixtures.PHYSICAL_ROOTS", {"fixture"}
    )

    payload = restore(repository, [source, overlay])
    restored = repository / ".scratch" / "fixture"
    assert (restored / "base.json").read_text(encoding="utf-8") == "base\n"
    assert (restored / "extra.json").read_text(encoding="utf-8") == "extra\n"
    assert not (restored / "base.json").is_symlink()
    assert payload["copied"] == 2

    (restored / "base.json").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="divergent fixture"):
        restore(repository, [source, overlay])
