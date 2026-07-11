---
type: Evaluation Report
title: Candidate 001 Full-Set Evaluation
description: Exact public, independently seeded, sandbox, and randomized-weight evidence for the promoted candidate.
status: completed
tags: [evaluation, public-benchmark, sandbox, controls, candidate-001]
timestamp: "2026-07-12T00:54:00+05:30"
---

# Candidate 001 full-set evaluation

## Authoritative checked public run

The checked adapter loads the exact 11 JSONL files and then uses the pinned official manifest validator, artifact-size check, static scanner, loader, preprocessing-isolation check, determinism check, decoder, inference loop, and scorer.

| Tier | Correct / 100 | Accuracy |
|---:|---:|---:|
| 0, diagnostic | 10 | 10% |
| 1 | 100 | 100% |
| 2 | 99 | 99% |
| 3-10, each | 2 | 2% |

The scored overall accuracy is 21.5%, and `highest_tier_above_90` is 2. Static analysis is clean, preprocessing is isolated, outputs are deterministic, and measured host inference is 0.215 seconds. See [`checked-public-1100.json`](../../evals/results/candidate-001-horner-tcn9/checked-public-1100.json).

## Generator and distribution-shift runs

- The official public seed reproduces the same 21.5% score: [`generated-public-1100.json`](../../evals/results/candidate-001-horner-tcn9/generated-public-1100.json).
- A full independent seed (`0123456789abcdef` repeated four times) scores 21.3% overall, with Tier 1 at 100% and Tier 2 at 97%: [`generated-independent-1100.json`](../../evals/results/candidate-001-horner-tcn9/generated-independent-1100.json).
- Five additional 100-case Tier-2 seeds aggregate to 492/500 (98.4%); their prime-level diagnostics are retained in [`benchmark-diagnostics.json`](../../experiments/runs/candidate-001-horner-tcn9/benchmark-diagnostics.json).

## Sandbox run

The pinned official Dockerfile was built unchanged and executed under Podman 5.8.2 with the official flags: no network, read-only root, read-only submission mount, writable `/tmp` tmpfs, four CPUs, and 8 GiB memory. Its Python 3.12, PyTorch 2.5.1 CPU, and safetensors 0.4.5 runtime reproduced 21.5% overall and completed inference in 0.4 seconds. The image digest and full result are in [`sandbox-public-1100.json`](../../evals/results/candidate-001-horner-tcn9/sandbox-public-1100.json).

## Weight control

All floating-point tensors were deterministically reinitialized while preserving tensor names, shapes, dtypes, and per-tensor scales. Overall accuracy fell from 21.5% to 7.9%; Tier 2 fell from 99% to 1%. The exact control is [`randomized-control-checked-1100.json`](../../evals/results/candidate-001-horner-tcn9/randomized-control-checked-1100.json).

## Interpretation

This is strong evidence for a genuine learned Tier-2 capability and a valid local release artifact. It is not an official private leaderboard score: the organizers' secret seed, remote Hugging Face commit, and manual compliance review remain outside local control. See the [model card](../submissions/candidate-001-horner-tcn9.md) and [evaluation runbook](../runbooks/evaluate-candidate.md).
