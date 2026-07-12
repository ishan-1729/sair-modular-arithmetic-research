---
type: Evaluation Runbook
title: Evaluate Current Candidate
description: Exact commands for static, checked-public, generated, control, and sandbox evaluation of Candidate 002.
status: active
tags: [runbook, evaluation, sandbox, candidate-002]
timestamp: "2026-07-12T00:54:00+05:30"
---

# Evaluate Candidate 002

Run all commands from the repository root after installing this project and the pinned evaluator:

```powershell
python -m pip install -e ".[dev]"
python -m pip install -e sources/github/modular-arithmetic-challenge

python -m modchallenge.cli check submission
python tools/evaluate_checked_benchmark.py submission `
  --timeout 300 `
  --output .scratch/candidate-002-checked-public-1100.json

python -m modchallenge.cli evaluate submission `
  --total 1100 `
  --seed 6d6f646368616c6c656e67652d7075626c69632d62656e63686d61726b2d7631 `
  --timeout 300 `
  --output .scratch/candidate-002-generated-public-1100.json
```

For the retained independent full set:

```powershell
python -m modchallenge.cli evaluate submission `
  --total 1100 `
  --seed 11b7c34d82eae606588d6fc7917f8c55e9d3a12214ed49b277b0bca38b7e396e `
  --timeout 300 `
  --output .scratch/candidate-002-generated-independent-1100.json
```

Recreate the randomized-weight control outside the submission boundary:

```powershell
python tools/randomize_submission_weights.py `
  submission/weights.safetensors `
  .scratch/controls/candidate-002-randomized/weights.safetensors `
  --seed 20260712
```

Copy only `manifest.json` and `model.py` beside that control weight, then evaluate the scratch directory with the checked adapter. Never place control weights in [`submission/`](../../submission/).

Build and run the pinned official image through the Docker/Podman-compatible wrapper:

```powershell
python tools/run_official_sandbox.py --engine docker --build
# Or, on this Windows host:
wsl.exe -d Ubuntu-24.04 -- bash -lc `
  "cd /mnt/d/Projects/SAIR_mod_challenge && python3 tools/run_official_sandbox.py --engine podman --build"
```

See the [container sandbox runbook](sandbox-candidate.md) for machine setup, evidence capture, and promotion requirements.

The durable results and clean artifact hashes are under [`experiments/runs/candidate-002-tier3-hybrid/`](../../experiments/runs/candidate-002-tier3-hybrid/) and [`evals/results/candidate-002-tier3-hybrid/`](../../evals/results/candidate-002-tier3-hybrid/).
