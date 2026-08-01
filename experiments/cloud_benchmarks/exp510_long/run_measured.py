"""Run unchanged EXP-510 and emit an environment/resource receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = Path(__file__).resolve().parent
RUNNER = ROOT / "experiments/exp_510_multiarchitecture_grokking_feasibility/run.py"
CONFIG = ROOT / "configs/exp_510_multiarchitecture_grokking_feasibility.yaml"
REQUIREMENTS = BENCHMARK_DIR / "requirements.txt"
EXPECTED_CHALLENGE_COMMIT = "82510bba00a1126649bd76dd1a451f14d0b3eb60"
EXPECTED_DEPENDENCIES = {
    "torch": "2.5.1+cpu",
    "torch_geometric": "2.8.0",
    "numpy": "2.5.1",
    "mlflow": "3.14.0",
    "psutil": "7.0.0",
    "PyYAML": "6.0.3",
    "safetensors": "0.8.0",
}
SOURCE_FILES = {
    "config": (
        CONFIG,
        "eb1054e2f512ccd6c5f68af44076e470e20867a383ff259eb93ac36c1fd54d9b",
    ),
    "runner": (
        RUNNER,
        "909ff11d3087e8418de95b254f8173598fc1d0c50b17fa22e464c8e44cf1637f",
    ),
    "tcn": (
        ROOT / "experiments/exp_100_width_scaling/model.py",
        "182614346a5ea9f5ab207395a351af0ac96a11ff56a0cd87ed6ea3550fb96c6e",
    ),
    "graph": (
        ROOT / "experiments/exp_401_pyg_graph_message_passing/model.py",
        "8ac98e9f9de7b2e7a916d05117dea53cd1c74fccf272fb44fa73fecc59b1746c",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_optional(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def dependency_versions() -> dict[str, str]:
    import mlflow
    import numpy
    import psutil
    import safetensors
    import torch
    import torch_geometric
    import yaml

    return {
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "numpy": numpy.__version__,
        "mlflow": mlflow.__version__,
        "psutil": psutil.__version__,
        "PyYAML": yaml.__version__,
        "safetensors": safetensors.__version__,
    }


def identity() -> dict[str, Any]:
    source_hashes = {name: sha256(path) for name, (path, _) in SOURCE_FILES.items()}
    expected_hashes = {name: expected for name, (_, expected) in SOURCE_FILES.items()}
    mismatches = {
        name: {"expected": expected_hashes[name], "actual": actual}
        for name, actual in source_hashes.items()
        if actual != expected_hashes[name]
    }
    if mismatches:
        raise RuntimeError(f"historical source identity mismatch: {mismatches}")
    challenge_commit = git_output(
        "rev-parse", "HEAD:sources/github/modular-arithmetic-challenge"
    )
    if challenge_commit != EXPECTED_CHALLENGE_COMMIT:
        raise RuntimeError(
            "challenge submodule mismatch: "
            f"expected {EXPECTED_CHALLENGE_COMMIT}, got {challenge_commit}"
        )
    return {
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_branch": git_output("branch", "--show-current"),
        "git_dirty_before_run": bool(git_output("status", "--porcelain")),
        "challenge_commit": challenge_commit,
        "source_sha256": source_hashes,
        "requirements_sha256": sha256(REQUIREMENTS),
    }


def environment() -> dict[str, Any]:
    memory = shutil.disk_usage(ROOT)
    dependencies = dependency_versions()
    if dependencies != EXPECTED_DEPENDENCIES:
        raise RuntimeError(
            f"dependency mismatch: expected {EXPECTED_DEPENDENCIES}, got {dependencies}"
        )
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "cpu_model": cpu_model(),
        "os_cpu_count": os.cpu_count(),
        "cpu_max": read_optional("/sys/fs/cgroup/cpu.max"),
        "cpuset_cpus_effective": read_optional(
            "/sys/fs/cgroup/cpuset.cpus.effective"
        ),
        "memory_max": read_optional("/sys/fs/cgroup/memory.max"),
        "disk_total_bytes": memory.total,
        "disk_free_bytes": memory.free,
        "dependencies": dependencies,
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


def compact_results(results: dict[str, Any] | None) -> dict[str, Any] | None:
    if results is None:
        return None
    compact: dict[str, Any] = {}
    for seed, arms in results.items():
        compact[seed] = {}
        for architecture, result in arms.items():
            curve = result.get("curve") or []
            compact[seed][architecture] = {
                "cached": result.get("cached"),
                "dataset_sha256": result.get("dataset_sha256"),
                "parameters": result.get("parameters"),
                "analysis": result.get("analysis"),
                "runtime": result.get("runtime"),
                "checkpoint": result.get("checkpoint"),
                "final_curve_row": curve[-1] if curve else None,
            }
    return compact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".scratch/cloud-exp510-long-benchmark",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    receipt_path = output / "cloud-benchmark-receipt.json"
    command = [
        sys.executable,
        str(RUNNER),
        "--config",
        str(CONFIG),
        "--output",
        str(output),
    ]
    identity_payload = identity()
    environment_payload = environment()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry-run",
                    "identity": identity_payload,
                    "environment": environment_payload,
                    "command": command,
                },
                indent=2,
            )
        )
        return 0

    output.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    wall_started = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT, check=False)
    wall_seconds = time.perf_counter() - wall_started
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    completed_at = datetime.now(timezone.utc).isoformat()
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    receipt = {
        "schema_version": "cloud-exp510-long-benchmark-v1",
        "status": "completed" if completed.returncode == 0 and summary else "failed",
        "started_at": started_at,
        "completed_at": completed_at,
        "identity": identity_payload,
        "environment": environment_payload,
        "command": command,
        "resource": {
            "wrapper_wall_seconds": wall_seconds,
            "child_user_cpu_seconds": usage_after.ru_utime - usage_before.ru_utime,
            "child_system_cpu_seconds": usage_after.ru_stime - usage_before.ru_stime,
            "maximum_resident_set_kib": usage_after.ru_maxrss,
            "exit_code": completed.returncode,
        },
        "runner": {
            "runtime": summary.get("runtime") if summary else None,
            "decision": summary.get("decision") if summary else None,
            "signature_counts": summary.get("signature_counts") if summary else None,
            "results": compact_results(summary.get("results")) if summary else None,
        },
        "limitations": [
            "Historical Python was 3.12.3; Codex universal currently supplies a newer 3.12 patch.",
            "Historical PyYAML and safetensors versions were not recorded; they are pinned for this cloud rerun but are outside the training kernels.",
            "Cross-CPU floating-point training may preserve exact-answer metrics without preserving checkpoint bytes.",
        ],
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2), flush=True)
    return completed.returncode if completed.returncode else (0 if summary else 2)


if __name__ == "__main__":
    raise SystemExit(main())
