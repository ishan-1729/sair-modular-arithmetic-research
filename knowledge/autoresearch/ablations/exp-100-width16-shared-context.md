---
type: Ablation Preregistration
title: EXP-100 Shared-Context Width Scaling Smoke Test
description: Matched test of whether the learned Horner transition remains learnable when scaling from nine to sixteen bits.
status: completed
tags: [autoresearch, ablation, TCN, width-scaling, Tier-3]
timestamp: "2026-07-12T02:20:00+05:30"
---

# EXP-100 shared-context width scaling

## Hypothesis

A residual TCN conditioned on the modulus through its bit representation, rather than an identity embedding, can learn the same local Horner transition at width 16 well enough to warrant a full Tier-3 run.

## Matched comparison

- A: width 9, primes of 4-8 bits.
- B: width 16, primes of 9-16 bits.
- Same seed family, optimizer, schedule, hidden width, block count, batch size, step budget, losses, corruption/composite fractions, CPU threads, and evaluation case counts.
- The only intended factor is state/modulus width and its associated prime band. Parameter counts and wall time are disclosed rather than artificially equalized.
- Prime identities are split before sampling; evaluation uses held-out test primes. Public expected answers are never read.

## Resource cap and gates

- Smoke budget: 400 steps per arm, batch 512, four CPU threads, no more than five aggregate CPU-minutes unless the process is clearly active and near completion.
- Historical incumbent logs show only 74% batch exactness at step 400, so this preregistration distinguishes a smoke continuation gate from a full-run promotion gate before execution.
- Width-9 validity gate: >=60% heldout transition and >=10% nine-step rollout. Failure invalidates the shared-context surface rather than falsifying width scaling.
- Width-16 strong gate: >=90% heldout transition and >=25% sixteen-step rollout.
- Width-16 continuation gate: >=35% heldout transition, >=1% sixteen-step rollout, and a positive final-100-step batch-exact delta. This authorizes only the pre-declared continuation to 1,200 steps.
- Falsify direct scaling if width 9 passes while width 16 remains below 20% transition and 0.5% rollout with no material final learning slope.

## Next expensive gate

Only a passing or explicitly continued width-16 arm may receive the 1,200-step budget. Checked public Tier 3 remains untouched until a challenger passes heldout-prime and independent-generator-style evaluation.

## Result

The width-9 surface passed. Width 16 reached 64.31% heldout transition and 0.52% rollout with positive learning slope, so it was neither falsified nor eligible for the preregistered continuation. Decision: `inconclusive-revise`; test learned width transfer as a new A/B rather than moving this run's gate.
