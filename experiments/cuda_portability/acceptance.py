"""Fail-closed representative-family CUDA hardware acceptance runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable
import xml.etree.ElementTree as ET

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import yaml

from .inventory import CONFIG_PATH, ROOT
from .runtime import configure_determinism, resolve_device, runtime_receipt, synchronize


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_acceptance() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    acceptance = config["hardware_acceptance"]
    if acceptance["schema_version"] != "cuda-hardware-acceptance-v1":
        raise ValueError("unsupported hardware acceptance schema")
    return acceptance


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def junit_receipt(path: Path, required_tests: Iterable[str]) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    rows = []
    for case in cases:
        rows.append(
            {
                "name": case.attrib["name"],
                "class": case.attrib.get("classname"),
                "seconds": float(case.attrib.get("time", "0")),
                "failed": case.find("failure") is not None,
                "error": case.find("error") is not None,
                "skipped": case.find("skipped") is not None,
            }
        )
    required = set(required_tests)
    observed = {row["name"] for row in rows}
    missing = sorted(required - observed)
    invalid = [
        row for row in rows if row["failed"] or row["error"] or row["skipped"]
    ]
    return {
        "required": sorted(required),
        "observed": rows,
        "missing": missing,
        "invalid": invalid,
        "passed": not missing and not invalid and required == observed,
    }


def profile_identity(profile_name: str, acceptance: dict[str, Any]) -> dict[str, Any]:
    profile_path = ROOT / acceptance["profile_config"]
    profiles = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    profile = profiles["profiles"][profile_name]
    lock_path = ROOT / profile["lock"]
    return {
        "name": profile_name,
        "config": acceptance["profile_config"],
        "config_sha256": sha256(profile_path),
        "lock": profile["lock"],
        "lock_sha256": sha256(lock_path),
        "expected_lock_sha256": profile["lock_sha256"],
        "torch_version": profile["torch_version"],
        "cuda_runtime": profile["cuda_runtime"],
    }


def execute(profile_name: str, output: Path) -> dict[str, Any]:
    acceptance = load_acceptance()
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "acceptance.json"
    started = time.time()
    monotonic_started = time.monotonic()
    receipt: dict[str, Any] = {
        "schema_version": acceptance["schema_version"],
        "status": "failed",
        "started_unix_seconds": started,
        "repository": {
            "commit": git_output("rev-parse", "HEAD"),
            "dirty": bool(git_output("status", "--porcelain")),
        },
        "source": {
            "config": CONFIG_PATH.relative_to(ROOT).as_posix(),
            "config_sha256": sha256(CONFIG_PATH),
            "runner_sha256": sha256(Path(__file__)),
            "test_sha256": sha256(ROOT / acceptance["test_file"]),
        },
        "profile": profile_identity(profile_name, acceptance),
    }
    try:
        if receipt["profile"]["lock_sha256"] != receipt["profile"][
            "expected_lock_sha256"
        ]:
            raise RuntimeError("selected CUDA lock hash does not match its profile")
        device = resolve_device("cuda", require_accelerator=True)
        configure_determinism(
            device=device,
            seed=int(acceptance["seed"]),
            torch_threads=int(acceptance["torch_threads"]),
        )
        receipt["runtime"] = runtime_receipt(device)
        environment_receipt = output / "environment.json"
        environment_command = [
            sys.executable,
            str(ROOT / "environments/cuda/verify_environment.py"),
            "--profile",
            profile_name,
            "--receipt",
            str(environment_receipt),
        ]
        subprocess.run(
            environment_command,
            cwd=ROOT,
            check=True,
            timeout=120,
        )
        receipt["environment"] = json.loads(
            environment_receipt.read_text(encoding="utf-8")
        )

        junit = output / "hardware-junit.xml"
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            acceptance["test_file"],
            f"--junitxml={junit}",
        ]
        environment = os.environ.copy()
        environment.update(
            {
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "SAIR_CUDA_ACCEPTANCE": "1",
                "SAIR_CUDA_DEVICE": str(device),
                "SAIR_CUDA_PROFILE": profile_name,
            }
        )
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=int(acceptance["timeout_seconds"]),
        )
        receipt["pytest"] = {
            "command": command,
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "junit": str(junit.relative_to(output)),
        }
        if not junit.is_file():
            raise RuntimeError("hardware pytest did not produce a JUnit receipt")
        matrix = junit_receipt(junit, acceptance["required_tests"])
        receipt["matrix"] = matrix
        if completed.returncode != 0 or not matrix["passed"]:
            raise RuntimeError("required CUDA acceptance matrix did not pass exactly")
        synchronize(device)
        receipt["cuda_memory"] = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
        receipt["status"] = "passed"
    except Exception as error:
        receipt["failure"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        receipt["wall_seconds"] = time.monotonic() - monotonic_started
        receipt["finished_unix_seconds"] = time.time()
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if receipt["status"] != "passed":
        raise RuntimeError(
            f"CUDA hardware acceptance failed; inspect {receipt_path}"
        )
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--profile", choices=("cu121", "cu124"), default="cu124")
    parser.add_argument("--output", type=Path, default=ROOT / ".scratch/cuda-acceptance")
    args = parser.parse_args(argv)
    acceptance = load_acceptance()
    if args.list:
        if args.require_cuda:
            parser.error("--list and --require-cuda are mutually exclusive")
        print(json.dumps(acceptance, indent=2, sort_keys=True))
        return 0
    if not args.require_cuda:
        parser.error("execution requires --require-cuda; CPU fallback is forbidden")
    result = execute(args.profile, args.output.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
