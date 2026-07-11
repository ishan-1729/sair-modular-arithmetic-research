---
type: Reproduction Runbook
title: Reproduce the Learned Horner TCN POC
description: Commands to validate, retrain, evaluate, and compare the reference POC on CPU.
status: active
tags: [runbook, reproducibility, POC]
timestamp: "2026-07-11T23:15:00+05:30"
---

# Reproduce the POC

Run from the repository root with Python 3.12 or 3.13.

```powershell
python -m pip install -e ".[dev]"
python tools/validate_project.py
python -m pytest -m "not slow"
```

Train a fresh deterministic smoke checkpoint:

```powershell
python -m pocs.learned_horner_tcn.train `
  --config configs/poc_horner_tcn_smoke.yaml `
  --run-dir experiments/runs/poc-001-reproduction
```

Evaluate the committed reference checkpoint and its perturbation control:

```powershell
python -m pocs.learned_horner_tcn.evaluate `
  --config configs/poc_horner_tcn_smoke.yaml `
  --checkpoint pocs/learned_horner_tcn/artifacts/poc_weights.pt
```

The trainer writes a run manifest, environment record, metrics, and checkpoint hash. Do not overwrite the committed reference run; use a new run ID. CPU time varies by PyTorch build and thread count, so compare accuracy and wall time together.

## Expected interpretation

A successful reproduction is deterministic within the recorded environment, beats the constant and randomized-weight controls on held-out-prime autonomous rollout, and passes project validation. Different exact metrics are acceptable across dependency versions only if the new run is recorded rather than silently replacing the reference.
