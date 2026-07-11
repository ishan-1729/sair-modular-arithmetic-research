---
type: Evaluation Runbook
title: Evaluate Candidate 001
description: Exact commands for static, checked-public, generated, control, and sandbox evaluation of the staged candidate.
status: active
tags: [runbook, evaluation, sandbox, candidate-001]
timestamp: "2026-07-12T00:54:00+05:30"
---

# Evaluate Candidate 001

Run all commands from the repository root after installing this project and the pinned evaluator:

```powershell
python -m pip install -e ".[dev]"
python -m pip install -e sources/github/modular-arithmetic-challenge

python -m modchallenge.cli check submission
python tools/evaluate_checked_benchmark.py submission `
  --timeout 300 `
  --output evals/results/candidate-001-horner-tcn9/checked-public-1100.json

python -m modchallenge.cli evaluate submission `
  --total 1100 `
  --seed 6d6f646368616c6c656e67652d7075626c69632d62656e63686d61726b2d7631 `
  --timeout 300 `
  --output evals/results/candidate-001-horner-tcn9/generated-public-1100.json
```

For the retained independent full set:

```powershell
python -m modchallenge.cli evaluate submission `
  --total 1100 `
  --seed 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef `
  --timeout 300 `
  --output evals/results/candidate-001-horner-tcn9/generated-independent-1100.json
```

Recreate the randomized-weight control outside the submission boundary:

```powershell
python tools/randomize_submission_weights.py `
  submission/weights.safetensors `
  .scratch/controls/candidate-001-randomized/weights.safetensors `
  --seed 20260712
```

Copy only `manifest.json` and `model.py` beside that control weight, then evaluate the scratch directory with the checked adapter. Never place control weights in [`submission/`](../../submission/).

For a Docker installation, build and run the pinned official image:

```powershell
python -m modchallenge.cli build-sandbox `
  --image modchallenge-sandbox `
  --repo-root sources/github/modular-arithmetic-challenge

python -m modchallenge.cli evaluate-sandboxed submission `
  --image modchallenge-sandbox `
  --total 1100 `
  --seed 6d6f646368616c6c656e67652d7075626c69632d62656e63686d61726b2d7631 `
  --timeout 300 --memory 8g --cpus 4
```

The durable results, clean artifact hashes, and image digest are under [`evals/results/candidate-001-horner-tcn9/`](../../evals/results/candidate-001-horner-tcn9/).
