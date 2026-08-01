from __future__ import annotations

from pathlib import Path

from tools.run_official_sandbox import container_run_command


def test_container_command_preserves_official_isolation_contract(tmp_path: Path) -> None:
    submission = tmp_path / "submission"
    output = tmp_path / "output"
    command = container_run_command(
        ["podman", "--connection", "sair-sandbox"],
        image="modchallenge-sandbox:test",
        submission=submission,
        output_dir=output,
        seed="ab" * 32,
        total=1100,
        timeout=300,
        memory="8g",
        cpus="4",
        tmpfs_size="2g",
    )

    joined = " ".join(str(item) for item in command)
    assert "--network none" in joined
    assert "--read-only" in command
    assert "/tmp:size=2g,mode=1777" in command
    assert "--memory 8g" in joined
    assert "--cpus 4" in joined
    submission_mount = next(
        item for item in command if "target=/sandbox/submission" in str(item)
    )
    output_mount = next(item for item in command if "target=/sandbox/output" in str(item))
    assert str(submission_mount).startswith("type=bind,source=")
    assert str(submission_mount).endswith("target=/sandbox/submission,readonly")
    assert str(output_mount).startswith("type=bind,source=")
    assert str(output_mount).endswith("target=/sandbox/output")
    assert "MODCHALLENGE_TOTAL=1100" in command
    assert "MODCHALLENGE_TIMEOUT=300" in command
    assert "MODCHALLENGE_SKIP_STATIC=1" not in command
