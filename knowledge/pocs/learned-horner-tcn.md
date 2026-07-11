---
type: Model Card
title: Learned Horner TCN POC
description: Model card for the small-width learned modular-transition proof of concept.
resource: ../../pocs/learned_horner_tcn/
status: experimental
tags: [POC, TCN, learned-recurrence, model-card]
timestamp: "2026-07-11T23:15:00+05:30"
---

# Learned Horner TCN POC

## Model details

The POC uses a 21,441-parameter non-causal dilated 1D convolutional cell over fixed-width binary channels for current state, reduced multiplicand, modulus, current multiplier bit, and boundary markers. The outer loop scans multiplier bits from most to least significant. After each cell invocation, thresholded predicted bits become the next recurrent state.

## Intended use

- Validate that the chosen decomposition, data split, run records, checkpointing, evaluation, and compliance controls work end-to-end on CPU.
- Measure whether learned transitions compose at small widths on held-out primes.
- Provide a reproducible base for sampling, loss, cell, and limb ablations.

It is **not** intended as a competitive submission or evidence of scaling to 2048-bit moduli.

## Training data

Offline code generates exact labels from true Horner trajectories. Prime identities are split before trajectory generation. The final training distribution augments training primes with composite moduli from 8 through 63 while still excluding held-out primes `{11, 23, 29, 37, 61}`. This densifies the reduction rule without leaking the held-out prime identities. Public benchmark rows are not used for training.

## Evaluation

The durable smoke result is stored under [`evals/results/poc-001-horner-tcn-smoke/`](../../evals/results/poc-001-horner-tcn-smoke/) and registered in [`experiments/registry.yaml`](../../experiments/registry.yaml).

| Metric | Observed value |
|---|---:|
| Held-out-prime one-step exact accuracy | 98.205% |
| Held-out-prime bit accuracy | 99.734% |
| 8-bit autonomous final exact accuracy | 92.10% |
| 16-bit autonomous final exact accuracy | 85.275% |
| 32-bit autonomous final exact accuracy | 76.95% |
| 16-bit constant-zero control | 8.05% |
| 16-bit reinitialized-weight control | 0.00% |
| Deterministic repeat agreement | 100% |

Per-prime 16-bit rollout ranges from 100% for `p=11` to 66.2% for `p=37`, making the remaining generalization failure visible rather than hiding it in an aggregate.

## Compliance rationale

The trainer uses arithmetic to produce labels; the inference cell does not contain a modular solver. The outer scan is explicit, while the state transition depends materially on learned weights. Randomizing weights must collapse accuracy. Final organizer judgment remains authoritative.

## Limitations

- Fixed small width and a tiny data regime.
- GroupNorm spans the fixed canvas, so the current cell is not yet a strictly local width-independent rule.
- Composite augmentation was chosen after earlier pilots; these are development results, not a pristine confirmatory evaluation.
- A full-width receptive field does not prove width extrapolation.
- Excellent step accuracy may still be insufficient for thousands of rollout steps.
- No official-interface adapter is promoted into `submission/` yet.

## Artifacts

- Executable code: [`pocs/learned_horner_tcn/`](../../pocs/learned_horner_tcn/)
- Reproduction instructions: [runbook](../runbooks/reproduce-poc.md)
- Architectural decision: [ADR-0002](../decisions/0002-poc-selection.md)
