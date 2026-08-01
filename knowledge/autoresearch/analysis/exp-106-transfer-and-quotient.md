---
type: Experiment Analysis
title: EXP-106 Transfer Persistence and Quotient Error Analysis
description: Matched 1,200-step transfer result and causal hypothesis for quotient-2 hard-example mining.
status: completed
tags: [autoresearch, analysis, transfer-learning, quotient, hard-examples]
timestamp: "2026-07-12T03:05:00+05:30"
---

# EXP-106 transfer persistence and quotient diagnosis

## Matched result

At 1,200 steps, learned transfer remains decisively better than random initialization: 94.335% versus 87.76% heldout transition and 47.07% versus 23.48% exact rollout. Invalid rollouts fall from 7.30% to 2.90%. Transfer is the selected initialization, but its 94.335%/47.07% result misses the preregistered 97%/60% full-budget gate.

## Error decomposition

A separate 100,000-transition diagnostic uses the same heldout primes and splits trajectory states from random legal states. The transferred model reaches 94.514% on trajectory states and 93.406% on random states, so recovery-state exposure is not the primary gap.

The decisive asymmetry is the current multiplier bit and quotient class:

- bit 0: 100.0% trajectory transition accuracy;
- bit 1: 87.67%;
- quotient 0: 99.75%;
- quotient 1: 98.91%;
- quotient 2: 50.09%.

Transitions within 1% of a reduction boundary also fall to 81.58%. Quotient-2 cases are only about 10% of the uniform trajectory distribution but dominate residual errors. This supports hard-example sampling, not a broader architecture change.

## Next intervention

EXP-110 will keep transfer initialization, architecture, steps, optimizer, seeds, and evaluation fixed while selecting 35% quotient-2 and 15% near-boundary examples from an oversized offline candidate pool. It must improve both transition and rollout at matched training steps without increasing invalid states.

Canonical evidence: [`metrics.json`](../../../evals/results/exp-106-width-transfer-1200/metrics.json), [`run.json`](../../../experiments/runs/exp-106-width-transfer-1200/run.json), and the retained [`transition-error-diagnosis.json`](../../../experiments/runs/exp-106-width-transfer-1200/transition-error-diagnosis.json).
