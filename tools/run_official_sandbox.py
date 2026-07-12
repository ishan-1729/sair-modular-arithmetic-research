"""Build and run the pinned official sandbox with Docker or Podman.

This wrapper exists because the pinned upstream CLI hard-codes the ``docker``
executable. It preserves the upstream Dockerfile, entrypoint, mounts, isolation
flags, and resource envelope while allowing a named Podman connection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ROOT = ROOT / "sources" / "github" / "modular-arithmetic-challenge"
DOCKERFILE = OFFICIAL_ROOT / "docker" / "Dockerfile"
CHALLENGE_COMMIT = "82510bba00a1126649bd76dd1a451f14d0b3eb60"
DEFAULT_IMAGE = f"modchallenge-sandbox:{CHALLENGE_COMMIT[:12]}"
PUBLIC_SEED = "6d6f646368616c6c656e67652d7075626c69632d62656e63686d61726b2d7631"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(
    command: Sequence[str],
    *,
    capture: bool = False,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    print("+ " + subprocess.list2cmdline(list(command)), flush=True)
    return subprocess.run(
        list(command),
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=capture,
        timeout=timeout,
    )


def engine_prefix(engine: str, podman_connection: str) -> list[str]:
    selected = engine
    if selected == "auto":
        if shutil.which("docker"):
            selected = "docker"
        elif shutil.which("podman"):
            selected = "podman"
        else:
            raise RuntimeError("neither docker nor podman is available on PATH")
    if shutil.which(selected) is None:
        raise RuntimeError(f"{selected} is not available on PATH")
    if selected == "podman" and podman_connection:
        return ["podman", "--connection", podman_connection]
    return [selected]


def container_run_command(
    prefix: Sequence[str],
    *,
    image: str,
    submission: Path,
    output_dir: Path,
    seed: str,
    total: int,
    timeout: int,
    memory: str,
    cpus: str,
    tmpfs_size: str,
) -> list[str]:
    return [
        *prefix,
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        f"/tmp:size={tmpfs_size},mode=1777",
        "--memory",
        memory,
        "--cpus",
        cpus,
        "--mount",
        f"type=bind,source={submission},target=/sandbox/submission,readonly",
        "--mount",
        f"type=bind,source={output_dir},target=/sandbox/output",
        "-e",
        "MODCHALLENGE_SUBMISSION=/sandbox/submission",
        "-e",
        "MODCHALLENGE_OUTPUT=/sandbox/output/result.json",
        "-e",
        f"MODCHALLENGE_SEED={seed}",
        "-e",
        f"MODCHALLENGE_TOTAL={total}",
        "-e",
        f"MODCHALLENGE_TIMEOUT={timeout}",
        image,
    ]


def verify_inputs(submission: Path) -> dict[str, str]:
    if not DOCKERFILE.is_file():
        raise RuntimeError(f"missing pinned Dockerfile: {DOCKERFILE}")
    commit = run(
        ["git", "-C", str(OFFICIAL_ROOT), "rev-parse", "HEAD"], capture=True
    ).stdout.strip()
    if commit != CHALLENGE_COMMIT:
        raise RuntimeError(
            f"official source is {commit}, expected pinned {CHALLENGE_COMMIT}"
        )
    required = {"manifest.json", "model.py", "weights.safetensors"}
    actual = {path.name for path in submission.iterdir() if path.is_file()}
    if actual != required:
        raise RuntimeError(
            f"submission must contain exactly {sorted(required)}, found {sorted(actual)}"
        )
    return {
        "challenge_commit": commit,
        "dockerfile_sha256": sha256(DOCKERFILE),
        "manifest_sha256": sha256(submission / "manifest.json"),
        "model_sha256": sha256(submission / "model.py"),
        "weights_sha256": sha256(submission / "weights.safetensors"),
    }


def default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / ".scratch" / "sandbox" / stamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("auto", "docker", "podman"), default="auto")
    parser.add_argument(
        "--podman-connection",
        default=os.environ.get("SAIR_PODMAN_CONNECTION", ""),
        help="Optional named Podman connection; native Ubuntu Podman uses none",
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--submission", type=Path, default=ROOT / "submission")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", default=PUBLIC_SEED)
    parser.add_argument("--total", type=int, default=1100)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--memory", default="8g")
    parser.add_argument("--cpus", default="4")
    parser.add_argument("--tmpfs-size", default="2g")
    parser.add_argument("--build", action="store_true", help="Build before running")
    parser.add_argument(
        "--check-only", action="store_true", help="Validate inputs and engine only"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    submission = args.submission.resolve()
    output_dir = (args.output_dir or default_output_dir()).resolve()
    evidence = verify_inputs(submission)
    prefix = engine_prefix(args.engine, args.podman_connection)
    version = run([*prefix, "version"], capture=True, timeout=30).stdout
    evidence.update(
        {
            "engine_command": " ".join(prefix),
            "engine_version_output": version.strip(),
            "engine_version_sha256": hashlib.sha256(version.encode()).hexdigest(),
            "image": args.image,
        }
    )
    if args.check_only:
        print(json.dumps(evidence, indent=2))
        return 0

    if args.build:
        run(
            [
                *prefix,
                "build",
                "-f",
                str(DOCKERFILE),
                "-t",
                args.image,
                str(OFFICIAL_ROOT),
            ],
            timeout=1800,
        )

    image_id = run(
        [*prefix, "image", "inspect", args.image, "--format", "{{.Id}}"],
        capture=True,
        timeout=30,
    ).stdout.strip()
    evidence["image_id"] = image_id

    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    command = container_run_command(
        prefix,
        image=args.image,
        submission=submission,
        output_dir=output_dir,
        seed=args.seed,
        total=args.total,
        timeout=args.timeout,
        memory=args.memory,
        cpus=args.cpus,
        tmpfs_size=args.tmpfs_size,
    )
    run(command, timeout=args.timeout + 180)
    elapsed = time.perf_counter() - started
    result_path = output_dir / "result.json"
    if not result_path.is_file():
        raise RuntimeError("sandbox completed without producing result.json")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    metadata = {
        **evidence,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "container_wall_seconds": elapsed,
        "seed": args.seed,
        "total": args.total,
        "timeout_seconds": args.timeout,
        "memory": args.memory,
        "cpus": args.cpus,
        "tmpfs_size": args.tmpfs_size,
        "result_sha256": sha256(result_path),
        "result_status": result.get("status"),
        "overall_accuracy": result.get("overall_accuracy"),
        "highest_tier_above_90": result.get("highest_tier_above_90"),
    }
    metadata_path = output_dir / "sandbox-run.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(result_path), "metadata": metadata}, indent=2))
    if result.get("status") not in (None, "completed"):
        raise RuntimeError(f"sandbox result status is {result.get('status')!r}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
