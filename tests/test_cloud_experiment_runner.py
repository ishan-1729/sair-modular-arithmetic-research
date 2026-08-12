from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import subprocess

import pytest
import yaml

from tools.cloud_experiment.checkpoints import latest_checkpoint
from tools.cloud_experiment.errors import CloudExperimentError
from tools.cloud_experiment import lifecycle as lifecycle_module
from tools.cloud_experiment.lifecycle import (
    canary,
    cleanup,
    doctor,
    finalize,
    launch,
    lifecycle_lock,
    resume,
    status,
)
from tools.cloud_experiment.manifest import load_manifest
from tools.cloud_experiment.package import build_package, verify_package
from tools.cloud_experiment.providers import LocalProvider


ROOT = Path(__file__).resolve().parents[1]


RUNNER = """\
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

mode = sys.argv[1]
control = Path(os.environ["SAIR_CLOUD_CONTROL_DIR"])

if mode == "canary":
    assert os.environ["SAIR_CLOUD_RUN_ID"]
    print("canary-ok")
elif mode == "phase-one":
    counter = Path("phase-one-count.txt")
    count = int(counter.read_text()) + 1 if counter.exists() else 1
    counter.write_text(str(count), encoding="utf-8")
    Path("state.txt").write_text("phase-one-complete", encoding="utf-8")
elif mode == "phase-two":
    if not (control / "allow-resume").exists():
        Path("state.txt").write_text("phase-two-checkpoint", encoding="utf-8")
        Path("orphan.tmp").write_text("must-not-survive-resume", encoding="utf-8")
        raise SystemExit(9)
    assert Path("state.txt").read_text(encoding="utf-8") == "phase-two-checkpoint"
    Path("result.json").write_text(
        json.dumps({"status": "complete"}, sort_keys=True) + "\\n",
        encoding="utf-8",
    )
else:
    raise SystemExit(f"unknown mode: {mode}")
"""


def git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=repo, text=True, stderr=subprocess.STDOUT
    ).strip()


def manifest_data(*, paid: bool = False, hourly_rate: str = "0") -> dict:
    return {
        "schema_version": "sair-cloud-experiment-v1",
        "experiment": {
            "id": "synthetic-resume-proof",
            "description": "Synthetic lifecycle fixture with no scientific assumptions.",
        },
        "package": {
            "include": ["runner.py", "manifest.yaml"],
            "exclude": [".scratch/**", "publication/**"],
            "require_clean": False,
        },
        "execution": {
            "working_directory": ".",
            "environment": {"PYTHONUNBUFFERED": "1"},
            "canary": {
                "argv": ["python", "runner.py", "canary"],
                "timeout_seconds": 10,
            },
            "phases": [
                {
                    "id": "phase-one",
                    "argv": ["python", "runner.py", "phase-one"],
                    "timeout_seconds": 10,
                    "required_outputs": ["phase-one-count.txt"],
                    "checkpoint_paths": ["phase-one-count.txt", "state.txt"],
                    "checkpoint_on_failure": True,
                },
                {
                    "id": "phase-two",
                    "argv": ["python", "runner.py", "phase-two"],
                    "timeout_seconds": 10,
                    "required_outputs": ["result.json"],
                    "checkpoint_paths": ["phase-one-count.txt", "state.txt", "result.json"],
                    "checkpoint_on_failure": True,
                },
            ],
        },
        "provider": {
            "driver": "tools.cloud_experiment.providers:LocalProvider",
            "paid": paid,
            "options": {
                "hourly_rate_usd": hourly_rate,
                "hardware": {"accelerator": "CPU", "accelerator_count": 0},
            },
            "expected_hardware": {"accelerator": "CPU", "accelerator_count": 0},
        },
        "budget": {
            "max_total_usd": 0.10,
            "max_provider_seconds": 120,
            "cleanup_reserve_seconds": 20,
            "max_idle_seconds": 20,
        },
        "artifacts": {
            "run_directory": ".scratch/cloud/synthetic-resume-proof",
            "publication_directory": "publication/synthetic-resume-proof",
        },
    }


def make_repo(tmp_path: Path, *, data: dict | None = None) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "cloud-runner-test@example.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Cloud Runner Test"], cwd=repo, check=True
    )
    (repo / "runner.py").write_text(RUNNER, encoding="utf-8")
    manifest_path = repo / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(data or manifest_data(), sort_keys=False), encoding="utf-8"
    )
    subprocess.run(["git", "add", "runner.py", "manifest.yaml"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
    return repo, manifest_path


def test_example_manifest_and_core_are_experiment_agnostic() -> None:
    manifest = load_manifest(ROOT / "configs/cloud_runner/manifest.example.yaml")
    assert manifest.data["schema_version"] == "sair-cloud-experiment-v1"
    core = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "tools/cloud_experiment").glob("*.py"))
    )
    assert re.search(r"\bEXP\d+\b", core, flags=re.IGNORECASE) is None
    assert "evaluation denominator" not in core.lower()


def test_package_is_deterministic_and_independently_verified(tmp_path: Path) -> None:
    repo, path = make_repo(tmp_path)
    manifest = load_manifest(path)
    first = build_package(manifest, repo / "package-one")
    second = build_package(manifest, repo / "package-two")
    assert first["source_archive"]["sha256"] == second["source_archive"]["sha256"]
    assert first["source_files"] == second["source_files"]
    assert verify_package(manifest, repo / "package-one") == first
    assert first["credential_read_performed"] is False
    assert first["provider_mutation_performed"] is False


def test_package_corruption_is_detected(tmp_path: Path) -> None:
    repo, path = make_repo(tmp_path)
    manifest = load_manifest(path)
    build_package(manifest, repo / "package")
    archive = repo / "package/source.tar.gz"
    archive.write_bytes(archive.read_bytes() + b"corruption")
    with pytest.raises(CloudExperimentError, match="source package bytes changed"):
        verify_package(manifest, repo / "package")


def test_failure_checkpoint_resumes_without_repeating_completed_phase(
    tmp_path: Path,
) -> None:
    repo, path = make_repo(tmp_path)
    manifest = load_manifest(path)
    package = build_package(manifest)

    with pytest.raises(CloudExperimentError, match="phase phase-two failed"):
        launch(manifest)

    failed = status(manifest)
    assert failed["status"] == "failed"
    assert failed["cleanup"]["exact_absence_confirmed"] is True
    assert failed["provider_elapsed_seconds"] < 120
    assert (manifest.run_directory / "workspace/phase-one-count.txt").read_text() == "1"
    before = latest_checkpoint(manifest.run_directory / "checkpoints")
    assert before is not None and before["generation"] == 2
    assert before["phase_id"] == "phase-two"
    first_attempt = json.loads(
        (manifest.run_directory / "attempts/attempt-0001.json").read_text(
            encoding="utf-8"
        )
    )
    assert first_attempt["status"] == "failed"
    assert first_attempt["errors"][0]["message"] == (
        "phase phase-two failed with exit code 9"
    )

    (manifest.run_directory / "allow-resume").write_text("ok\n", encoding="utf-8")
    completed = resume(manifest)
    assert completed["status"] == "completed"
    assert completed["cleanup"]["exact_absence_confirmed"] is True
    assert [item["status"] for item in completed["phases"]] == ["reused", "completed"]
    assert (manifest.run_directory / "workspace/phase-one-count.txt").read_text() == "1"
    assert json.loads(
        (manifest.run_directory / "workspace/result.json").read_text(encoding="utf-8")
    ) == {"status": "complete"}
    assert not (manifest.run_directory / "workspace/orphan.tmp").exists()
    after = latest_checkpoint(manifest.run_directory / "checkpoints")
    assert after is not None and after["generation"] == 3
    assert json.loads(
        (manifest.run_directory / "attempts/attempt-0002.json").read_text(
            encoding="utf-8"
        )
    )["status"] == "completed"
    assert (
        manifest.run_directory / "phases/phase-two/attempt-0001.json"
    ).is_file()
    assert (
        manifest.run_directory / "phases/phase-two/attempt-0002.json"
    ).is_file()

    final = finalize(manifest)
    assert final["manifest_sha256"] == manifest.sha256
    assert final["package_sha256"] == package["source_archive"]["sha256"]
    assert {item["path"] for item in final["outputs"]} == {
        "phase-one-count.txt",
        "result.json",
    }
    assert (manifest.publication_directory / "outputs/result.json").is_file()
    assert cleanup(manifest)["exact_absence_confirmed"] is True
    assert finalize(manifest) == final


def test_corrupted_checkpoint_blocks_resume_before_allocation(tmp_path: Path) -> None:
    _repo, path = make_repo(tmp_path)
    manifest = load_manifest(path)
    build_package(manifest)
    with pytest.raises(CloudExperimentError):
        launch(manifest)
    pointer = json.loads(
        (manifest.run_directory / "checkpoints/latest.json").read_text(encoding="utf-8")
    )
    archive = manifest.run_directory / "checkpoints" / pointer["archive"]
    archive.write_bytes(archive.read_bytes() + b"corruption")
    (manifest.run_directory / "allow-resume").write_text("ok\n", encoding="utf-8")
    with pytest.raises(CloudExperimentError, match="checkpoint archive hash mismatch"):
        resume(manifest)
    registry = json.loads(
        (manifest.run_directory / "provider/local-resources.json").read_text(
            encoding="utf-8"
        )
    )
    assert registry["resources"] == []


def test_cost_admission_fails_before_resource_allocation(tmp_path: Path) -> None:
    data = manifest_data(hourly_rate="10")
    data["budget"]["max_total_usd"] = 0.10
    _repo, path = make_repo(tmp_path, data=data)
    manifest = load_manifest(path)
    build_package(manifest)
    with pytest.raises(CloudExperimentError, match="exceeds authorization"):
        doctor(manifest)
    with pytest.raises(CloudExperimentError, match="exceeds authorization"):
        launch(manifest)
    lifecycle = status(manifest)
    assert lifecycle["lease"] is None
    assert lifecycle["cleanup"]["exact_absence_confirmed"] is True


def test_nonempty_provider_inventory_blocks_duplicate(tmp_path: Path) -> None:
    _repo, path = make_repo(tmp_path)
    manifest = load_manifest(path)
    build_package(manifest)
    registry = manifest.run_directory / "provider/local-resources.json"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        json.dumps(
            {
                "schema_version": "sair-local-provider-v1",
                "resources": [
                    {
                        "resource_id": "other-run",
                        "owner_nonce": "other-owner",
                        "hourly_rate_usd": "0",
                        "created_at_epoch_seconds": 1,
                        "hardware": {"accelerator": "CPU", "accelerator_count": 0},
                        "metadata": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(CloudExperimentError, match="inventory is not empty"):
        launch(manifest)
    assert status(manifest)["lease"] is None
    remaining = json.loads(registry.read_text(encoding="utf-8"))["resources"]
    assert [item["resource_id"] for item in remaining] == ["other-run"]


def test_uncertain_create_is_reconciled_by_owner(tmp_path: Path, monkeypatch) -> None:
    _repo, path = make_repo(tmp_path)
    manifest = load_manifest(path)
    build_package(manifest)

    class ResponseLossProvider(LocalProvider):
        def allocate(self, **kwargs):
            super().allocate(**kwargs)
            raise CloudExperimentError("simulated response loss after create")

    provider = ResponseLossProvider(
        options=manifest.data["provider"]["options"],
        control_directory=manifest.run_directory,
    )
    monkeypatch.setattr(
        lifecycle_module, "_load_provider", lambda _manifest, **_kwargs: provider
    )
    with pytest.raises(CloudExperimentError, match="response loss"):
        launch(manifest)
    failed = status(manifest)
    assert failed["lease"] is None
    assert failed["cleanup"]["deleted"] is True
    assert failed["cleanup"]["exact_absence_confirmed"] is True
    assert provider.inventory(owner_nonce=None) == []


def test_hardware_mismatch_and_canary_failure_both_cleanup(tmp_path: Path) -> None:
    hardware_data = manifest_data()
    hardware_data["provider"]["expected_hardware"] = {
        "accelerator": "GPU",
        "accelerator_count": 1,
    }
    _repo, hardware_path = make_repo(tmp_path / "hardware", data=hardware_data)
    hardware_manifest = load_manifest(hardware_path)
    build_package(hardware_manifest)
    with pytest.raises(CloudExperimentError, match="allocated hardware mismatch"):
        launch(hardware_manifest)
    assert status(hardware_manifest)["cleanup"]["exact_absence_confirmed"] is True

    canary_data = manifest_data()
    canary_data["execution"]["canary"]["argv"] = [
        "python",
        "-c",
        "raise SystemExit(7)",
    ]
    _repo, canary_path = make_repo(tmp_path / "canary", data=canary_data)
    canary_manifest = load_manifest(canary_path)
    build_package(canary_manifest)
    with pytest.raises(CloudExperimentError, match="standalone canary failed"):
        canary(canary_manifest)
    receipt = json.loads(
        (canary_manifest.run_directory / "canary/receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["status"] == "failed"
    assert receipt["cleanup"]["exact_absence_confirmed"] is True


def test_idle_heartbeat_timeout_is_enforced_and_cleanup_runs(tmp_path: Path) -> None:
    data = manifest_data()
    data["budget"]["max_idle_seconds"] = 1
    data["execution"]["phases"] = [
        {
            "id": "idle",
            "argv": ["python", "-c", "import time; time.sleep(5)"],
            "timeout_seconds": 10,
            "checkpoint_on_failure": False,
        }
    ]
    _repo, path = make_repo(tmp_path, data=data)
    manifest = load_manifest(path)
    build_package(manifest)
    with pytest.raises(CloudExperimentError, match="exit code 125"):
        launch(manifest)
    failed = status(manifest)
    assert failed["cleanup"]["exact_absence_confirmed"] is True
    stderr_path = manifest.run_directory / failed["phases"][0]["receipt"]["logs"][
        "stderr"
    ]["path"]
    assert "heartbeat was idle for 1s" in stderr_path.read_text(encoding="utf-8")


def test_lifecycle_lock_rejects_duplicate_controller(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    with lifecycle_lock(run_directory):
        with pytest.raises(CloudExperimentError, match="another controller"):
            with lifecycle_lock(run_directory):
                pass


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda value: value["execution"]["environment"].update(
                {"PROVIDER_API_TOKEN": "not-even-a-real-secret"}
            ),
            "secret-like environment name",
        ),
        (
            lambda value: value["provider"]["options"].update(
                {"api_token": "not-even-a-real-secret"}
            ),
            "secret-like provider option key",
        ),
        (
            lambda value: value["artifacts"].update(
                {"run_directory": "../outside"}
            ),
            "escapes its declared root",
        ),
        (
            lambda value: value["budget"].update(
                {"cleanup_reserve_seconds": 120}
            ),
            "cleanup reserve",
        ),
    ],
)
def test_manifest_semantics_fail_closed(tmp_path: Path, mutator, message: str) -> None:
    data = manifest_data()
    mutator(data)
    _repo, path = make_repo(tmp_path, data=data)
    with pytest.raises(CloudExperimentError, match=message):
        load_manifest(path)


def test_paid_manifest_requires_current_exact_authorization_before_driver_load(
    tmp_path: Path,
) -> None:
    _repo, path = make_repo(tmp_path, data=manifest_data(paid=True))
    manifest = load_manifest(path)
    build_package(manifest)
    with pytest.raises(CloudExperimentError, match="requires an authorization"):
        launch(manifest)
    assert not (manifest.run_directory / "provider/local-resources.json").exists()

    authorization = {
        "schema_version": "sair-cloud-authorization-v1",
        "run_id": manifest.run_id,
        "manifest_sha256": "0" * 64,
        "provider_driver": manifest.data["provider"]["driver"],
        "maximum_total_cost_usd": str(manifest.data["budget"]["max_total_usd"]),
        "maximum_provider_lifetime_seconds": manifest.data["budget"][
            "max_provider_seconds"
        ],
        "expires_at_utc": (
            datetime.now(timezone.utc) + timedelta(minutes=5)
        ).isoformat().replace("+00:00", "Z"),
    }
    auth_path = manifest.repo_root / "authorization.json"
    auth_path.write_text(json.dumps(authorization), encoding="utf-8")
    with pytest.raises(CloudExperimentError, match="manifest_sha256"):
        launch(manifest, authorization_path=auth_path)
    assert not (manifest.run_directory / "provider/local-resources.json").exists()


def test_public_module_help() -> None:
    completed = subprocess.run(
        ["python", "-m", "tools.cloud_experiment", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "launch" in completed.stdout
    assert "resume" in completed.stdout
    assert "checkpoint" in completed.stdout
