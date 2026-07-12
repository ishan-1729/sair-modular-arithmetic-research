---
type: Experiment Preregistration
title: EXP-616 Tier-6 Trajectory-Tail CVaR
description: Loss-only A/B of mean eight-step recovery against a top-quartile worst-step, worst-bit trajectory surrogate.
status: preregistered
tags: [autoresearch, tier6, cvar, trajectory-loss, straight-through]
timestamp: "2026-07-13T04:30:00+05:30"
---

# EXP-616 Tier-6 trajectory-tail CVaR

## Question and hypothesis

EXP-615's per-transition casewise maximum slightly improved rollout but did not address how eight-step recovery averages across steps and examples. Rare autonomous failures can therefore remain diluted. Will a CVaR term that selects each example's worst bit across all eight steps, then averages the hardest 25% of examples, better align training with exact trajectory survival?

On 20 fresh width-128 recovery chunks, the standard loss averaged `0.01499` and the top-quartile trajectory-tail statistic averaged `0.28940`, with 99.73% mean batch-step exactness. The fixed `0.01` coefficient contributes about `0.00289`, or 19% of the standard objective. This scale is fixed before selection.

## Matched comparison

Both arms load the same EXP-609 checkpoint and use the same 99,824-parameter architecture, seed, fresh identities, examples, 200 updates, optimizer, learning-rate schedule, batch sizes, straight-through state path, mode schedule, existing losses, clipping, and evaluation. The only changed variable is the trajectory-CVaR coefficient: zero for control and `0.01` for challenger. The CVaR fraction is 25% in both definitions and applies only to eight-step recovery chunks; transition64 updates remain identical.

## Gates

Each endpoint is evaluated on 2,000 transitions and 200 autonomous rollouts at widths 64 and 128. Advance only if the challenger reaches at least 90% width-128 rollout, gains at least three points over control, retains at least 95% width-64 rollout, reaches at least 99.8% width-128 transition, and keeps invalid width-128 rollouts at or below 1%.

A pass requires higher-fidelity confirmation. A miss closes this fixed trajectory-tail formulation; do not tune the fraction or coefficient on the observed selection set.

## Resource budget

The directly matched EXP-615 run took 984.17 wall seconds and 3,771.71 aggregate CPU seconds with four threads. Tensor top-k over 16 examples is negligible relative to the recurrent model; 30 wall seconds are allocated for its measured-objective overhead, giving a 1,015-second projection.

The planning budget is 1,650 wall seconds and the hard runaway ceiling is 2,450 wall seconds, or at most 163.34 aggregate CPU minutes at four continuously occupied threads. Wall and CPU are distinct. Completed arms are checkpointed and content-keyed. No work or evaluation cases will be removed to satisfy a shorter cap.

## Boundaries

Checked and private answers remain unread. Exact arithmetic remains confined to offline labels and evaluation. The loss is training-only; inference is unchanged, tensor-native, and trained-weight dependent. Native PyTorch BCE, maximum, and top-k operations are sufficient.
