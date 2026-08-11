from __future__ import annotations

import json
from pathlib import Path
import subprocess

import yaml

from environments.cuda import verify_environment as verify
from experiments.cuda_portability import acceptance


ROOT = Path(__file__).resolve().parents[1]
CUDA_DIR = ROOT / "environments/cuda"


def test_cuda_locks_are_complete_hashed_and_index_isolated() -> None:
    result = verify.verify_offline()
    assert result["status"] == "offline-locks-verified"
    assert result["pypi_index"] == "https://pypi.org/simple"
    assert set(result["locks"]) == {"cu121", "cu124"}
    assert all(lock["requirements"] >= 100 for lock in result["locks"].values())
    profiles = yaml.safe_load((CUDA_DIR / "profiles.yaml").read_text(encoding="utf-8"))
    for name, profile in profiles["profiles"].items():
        lock = (ROOT / profile["lock"]).read_text(encoding="utf-8")
        input_text = (ROOT / profile["input"]).read_text(encoding="utf-8")
        assert f"torch @ {profile['torch_url']}" in lock
        assert f"torch @ {profile['torch_url']}" in input_text
        assert "--hash=sha256:" in lock
        assert "--index-url" not in lock
        assert "--extra-index-url" not in lock
        assert f"requirements-{name}.in" in lock.splitlines()[1]


def test_bootstrap_is_pinned_and_dry_run_never_probes_or_installs(tmp_path) -> None:
    profiles = yaml.safe_load((CUDA_DIR / "profiles.yaml").read_text(encoding="utf-8"))
    script = (CUDA_DIR / "bootstrap.sh").read_text(encoding="utf-8")
    assert profiles["uv"] == {
        "version": "0.11.28",
        "installer_url": "https://astral.sh/uv/0.11.28/install.sh",
        "installer_sha256": "b7b3fe80cad1142a2a5794050b7db7b3291d1bac1423b0732571dd9366e8ca8b",
    }
    assert "UV_UNMANAGED_INSTALL" in script
    assert "--require-hashes" in script
    assert "--default-index https://pypi.org/simple" in script
    assert "--no-config" in script
    assert "nvidia-smi" in script
    assert "never falls back to CPU" in script
    assert "curl --proto '=https' --tlsv1.2" in script
    completed = subprocess.run(
        [
            str(CUDA_DIR / "bootstrap.sh"),
            "--profile",
            "cu124",
            "--venv",
            str(tmp_path / "unused"),
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    result = json.loads(completed.stdout)
    assert result["status"] == "dry-run"
    assert result["profile"] == "cu124"
    assert not (tmp_path / "unused").exists()


def test_preloaded_torch_wheel_keeps_canonical_lock_and_hash_enforcement() -> None:
    script = (CUDA_DIR / "bootstrap.sh").read_text(encoding="utf-8")
    lock = (CUDA_DIR / "requirements-cu124.txt").read_text(encoding="utf-8")
    expected = "bf6484bfe5bc4f92a4a1a1bf553041505e19a911f717065330eb061afe0e14d7"
    assert "SAIR_CUDA_PRELOADED_TORCH_WHEEL" in script
    assert "908202405" in script
    assert expected in script
    assert expected in lock
    assert "Path(wheel).resolve().as_uri()" in script
    assert '"${temporary_lock:-$lock_path}"' in script
    assert '"$lock_path" "$temporary_lock"' in script
    assert '"$repository_root/environments/cuda/verify_environment.py"' in script
    assert '"canonical_lock_unchanged": True' in script


def test_bootstrap_persists_pinned_uv_path_for_standalone_acceptance() -> None:
    bootstrap = (CUDA_DIR / "bootstrap.sh").read_text(encoding="utf-8")
    acceptance_script = (CUDA_DIR / "run-acceptance.sh").read_text(
        encoding="utf-8"
    )
    assert '.sair-uv-bin' in bootstrap
    assert 'printf \'%s\\n\' "$SAIR_UV_BIN"' in bootstrap
    assert 'uv_marker="$environment_dir/.sair-uv-bin"' in acceptance_script
    assert 'export SAIR_UV_BIN="$(<"$uv_marker")"' in acceptance_script
    assert '[[ -x "$SAIR_UV_BIN" ]]' in acceptance_script


def test_version_and_lock_guards_fail_closed() -> None:
    assert verify.version_tuple("550.54.14") > verify.version_tuple("525.60.13")
    profiles = verify.load_profiles()
    profile = dict(profiles["profiles"]["cu124"])
    profile["lock_sha256"] = "0" * 64
    try:
        verify.verify_lock("cu124", profile)
    except ValueError as error:
        assert "lock hash changed" in str(error)
    else:
        raise AssertionError("a changed requirements lock was accepted")


def test_acceptance_manifest_lists_every_required_family(capsys) -> None:
    assert acceptance.main(["--list"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["required_tests"] == [
        "test_cuda_runtime_and_exp699_update",
        "test_cuda_exp510_typed_max_graph",
        "test_cuda_exp654_neural_ram",
        "test_cuda_checkpoint_round_trip",
    ]
    source = (ROOT / listed["test_file"]).read_text(encoding="utf-8")
    assert all(f"def {name}(" in source for name in listed["required_tests"])


def test_junit_matrix_rejects_skips_and_missing_cases(tmp_path) -> None:
    junit = tmp_path / "receipt.xml"
    junit.write_text(
        "<testsuite><testcase name='a'/><testcase name='b'><skipped/></testcase>"
        "</testsuite>",
        encoding="utf-8",
    )
    result = acceptance.junit_receipt(junit, ["a", "b", "c"])
    assert not result["passed"]
    assert result["missing"] == ["c"]
    assert [row["name"] for row in result["invalid"]] == ["b"]


def test_archive_acceptance_uses_explicit_source_identity(monkeypatch) -> None:
    commit = "a" * 40
    monkeypatch.setenv("SAIR_SOURCE_COMMIT", commit)
    monkeypatch.setenv("SAIR_SOURCE_DIRTY", "0")
    monkeypatch.setattr(
        acceptance,
        "git_output",
        lambda *_: (_ for _ in ()).throw(AssertionError("git must not run")),
    )
    assert acceptance.repository_identity() == {"commit": commit, "dirty": False}


def test_acceptance_execution_cannot_fall_back_to_cpu(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(acceptance.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        acceptance,
        "git_output",
        lambda *arguments: "test-commit" if arguments[0] == "rev-parse" else "",
    )
    try:
        acceptance.execute("cu124", tmp_path)
    except RuntimeError as error:
        assert "CUDA hardware acceptance failed" in str(error)
    else:
        raise AssertionError("hardware acceptance silently used CPU")
    receipt = json.loads((tmp_path / "acceptance.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert "CUDA replay requested" in receipt["failure"]["message"]
