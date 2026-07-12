---
type: Evaluation Runbook
title: Run the Official Container Sandbox
description: Docker- and Podman-compatible wrapper for the pinned official sandbox image and isolation contract.
status: active
tags: [runbook, docker, podman, sandbox, evaluation]
timestamp: "2026-07-12T09:00:00+05:30"
---

# Run the official container sandbox

[`tools/run_official_sandbox.py`](../../tools/run_official_sandbox.py) builds the pinned upstream Dockerfile and runs it without changing its image contents or entrypoint. It supports Docker and Podman because the pinned official CLI hard-codes the `docker` executable.

The wrapper verifies the pinned challenge commit and exact three-file submission, then applies `--network none`, `--read-only`, a `/tmp` tmpfs, read-only submission mount, 8 GiB memory, four CPUs, and the 300-second inference budget. It writes `result.json` plus `sandbox-run.json`, including the Dockerfile hash, image ID, engine fingerprint, submission hashes, resource envelope, timings, and result hash.

## Docker

```powershell
python tools/run_official_sandbox.py --engine docker --build
```

## Native Podman in Ubuntu WSL

Ubuntu 24.04 WSL is the machine-wide development distribution. It contains native rootless Podman, Git, Python 3.12, and the project virtual environment. From a Windows-hosted terminal tool, enter the existing repository through WSL:

```powershell
wsl.exe -d Ubuntu-24.04 -- bash -lc `
  "cd /mnt/d/Projects/SAIR_mod_challenge && podman --version"
```

Validate the engine and pinned inputs:

```bash
cd /mnt/d/Projects/SAIR_mod_challenge
python3 tools/run_official_sandbox.py --engine podman --check-only
```

Build and run:

```bash
python3 tools/run_official_sandbox.py --engine podman --build
```

The default output is a timestamped ignored directory under `.scratch/sandbox/`. Copy a successful result and metadata into the candidate's canonical run directory, update its checkpoint and image hashes, register the evidence, and commit it before calling the sandbox result durable.
