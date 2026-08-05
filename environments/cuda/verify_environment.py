#!/usr/bin/env python3
"""Fail-closed verifier for the generated CUDA requirements and live runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = ROOT / "environments/cuda/profiles.yaml"
HASH_RE = re.compile(r"--hash=sha256:[0-9a-f]{64}(?:\s|$)")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def version_tuple(value: str) -> tuple[int, ...]:
    match = re.match(r"^(\d+(?:\.\d+)*)", value)
    if not match:
        raise ValueError(f"invalid version: {value}")
    return tuple(int(part) for part in match.group(1).split("."))


def load_profiles() -> dict[str, Any]:
    result = yaml.safe_load(PROFILE_PATH.read_text(encoding="utf-8"))
    if result.get("schema_version") != "sair-cuda-profiles-v1":
        raise ValueError("unsupported CUDA profile schema")
    return result


def requirement_stanzas(lock_path: Path) -> list[str]:
    stanzas: list[str] = []
    current: list[str] = []
    for line in lock_path.read_text(encoding="utf-8").splitlines():
        if not line or line.lstrip().startswith("#"):
            continue
        if not line[0].isspace():
            if current:
                stanzas.append("\n".join(current))
            current = [line]
        elif current:
            current.append(line)
    if current:
        stanzas.append("\n".join(current))
    return stanzas


def verify_lock(profile_name: str, profile: dict[str, Any]) -> dict[str, Any]:
    lock_path = ROOT / profile["lock"]
    if not lock_path.is_file():
        raise ValueError(f"missing lock for {profile_name}: {lock_path}")
    actual_hash = sha256(lock_path)
    if actual_hash != profile["lock_sha256"]:
        raise ValueError(
            f"{profile_name} lock hash changed: expected "
            f"{profile['lock_sha256']}, got {actual_hash}"
        )
    raw = lock_path.read_text(encoding="utf-8")
    if "--index-url" in raw or "--extra-index-url" in raw:
        raise ValueError(f"{profile_name} lock contains a package-confusion index")
    stanzas = requirement_stanzas(lock_path)
    if len(stanzas) < 100:
        raise ValueError(f"{profile_name} lock is not transitively complete")
    for stanza in stanzas:
        first = stanza.splitlines()[0]
        if "==" not in first and " @ https://" not in first:
            raise ValueError(f"unversioned requirement in {profile_name}: {first}")
        if not HASH_RE.search(stanza):
            raise ValueError(f"unhashed requirement in {profile_name}: {first}")
    expected_torch = f"torch @ {profile['torch_url']}"
    torch_stanzas = [stanza for stanza in stanzas if stanza.startswith("torch @ ")]
    if len(torch_stanzas) != 1 or not torch_stanzas[0].startswith(expected_torch):
        raise ValueError(
            f"{profile_name} torch must use its index-isolated direct wheel URL"
        )
    return {
        "path": profile["lock"],
        "sha256": actual_hash,
        "requirements": len(stanzas),
        "torch_url": profile["torch_url"],
    }


def verify_offline() -> dict[str, Any]:
    config = load_profiles()
    if config["platform"] != "x86_64-unknown-linux-gnu":
        raise ValueError("only the declared Linux x86_64 image is supported")
    if config.get("pypi_index") != "https://pypi.org/simple":
        raise ValueError("the default Python package index is not explicit")
    uv = config["uv"]
    if not re.fullmatch(r"[0-9a-f]{64}", uv["installer_sha256"]):
        raise ValueError("uv installer is not hash pinned")
    locks = {
        name: verify_lock(name, profile)
        for name, profile in config["profiles"].items()
    }
    scripts = [
        ROOT / "environments/cuda/bootstrap.sh",
        ROOT / "environments/cuda/run-acceptance.sh",
    ]
    for script in scripts:
        if not script.is_file() or not os.access(script, os.X_OK):
            raise ValueError(f"CUDA launcher is missing or not executable: {script}")
    return {
        "status": "offline-locks-verified",
        "profile_config": str(PROFILE_PATH.relative_to(ROOT)),
        "profile_config_sha256": sha256(PROFILE_PATH),
        "uv": uv,
        "pypi_index": config["pypi_index"],
        "locks": locks,
    }


def nvidia_query() -> dict[str, str]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise RuntimeError("nvidia-smi is required for CUDA acceptance")
    command = [
        executable,
        "--query-gpu=driver_version,name,compute_cap,memory.total",
        "--format=csv,noheader,nounits",
        "--id=0",
    ]
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=30
    )
    values = [part.strip() for part in completed.stdout.strip().split(",")]
    if len(values) != 4:
        raise RuntimeError(f"unexpected nvidia-smi response: {completed.stdout!r}")
    return dict(zip(("driver", "name", "compute_capability", "memory_mib"), values))


def verify_live(profile_name: str) -> dict[str, Any]:
    offline = verify_offline()
    config = load_profiles()
    if profile_name not in config["profiles"]:
        raise ValueError(f"unknown CUDA profile: {profile_name}")
    profile = config["profiles"][profile_name]
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("CUDA profiles require Linux x86_64")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"expected Python 3.12, found {platform.python_version()}")

    uv_binary = os.environ.get("SAIR_UV_BIN") or shutil.which("uv")
    if not uv_binary:
        raise RuntimeError("the pinned uv executable is unavailable")
    uv_version = subprocess.run(
        [uv_binary, "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout.strip()
    if uv_version != f"uv {config['uv']['version']} (x86_64-unknown-linux-gnu)":
        raise RuntimeError(f"unexpected uv runtime: {uv_version}")

    import torch

    if torch.__version__ != profile["torch_version"]:
        raise RuntimeError(
            f"expected torch {profile['torch_version']}, found {torch.__version__}"
        )
    if str(torch.version.cuda) != profile["cuda_runtime"]:
        raise RuntimeError(
            f"expected CUDA runtime {profile['cuda_runtime']}, "
            f"found {torch.version.cuda}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA profile cannot fall back to CPU")
    device = torch.device("cuda", 0)
    properties = torch.cuda.get_device_properties(device)
    minimum_cc = tuple(int(value) for value in profile["minimum_compute_capability"])
    actual_cc = (int(properties.major), int(properties.minor))
    if actual_cc < minimum_cc:
        raise RuntimeError(
            f"compute capability {actual_cc} is below required {minimum_cc}"
        )
    nvidia = nvidia_query()
    if version_tuple(nvidia["driver"]) < version_tuple(profile["minimum_driver"]):
        raise RuntimeError(
            f"driver {nvidia['driver']} is below required {profile['minimum_driver']}"
        )
    package_versions = {
        name: importlib.metadata.version(name)
        for name in (
            "mlflow",
            "numpy",
            "pytest",
            "PyYAML",
            "safetensors",
            "torch-geometric",
        )
    }
    return {
        **offline,
        "status": "live-cuda-verified",
        "profile": profile_name,
        "python": platform.python_version(),
        "uv_runtime": uv_version,
        "torch": torch.__version__,
        "torch_cuda_runtime": str(torch.version.cuda),
        "gpu": {
            **nvidia,
            "torch_name": properties.name,
            "torch_compute_capability": list(actual_cc),
            "torch_total_memory_bytes": int(properties.total_memory),
        },
        "packages": package_versions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--profile", choices=("cu121", "cu124"))
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.offline == bool(args.profile):
        parser.error("choose exactly one of --offline or --profile")
    result = verify_offline() if args.offline else verify_live(args.profile)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
