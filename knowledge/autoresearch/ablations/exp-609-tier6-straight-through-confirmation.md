---
type: Experiment Preregistration
title: EXP-609 Tier-6 Straight-Through Confirmation
description: Frozen-source versus lower-rate straight-through continuation on higher-fidelity fresh cases.
status: completed
tags: [autoresearch, Tier-6, straight-through, confirmation, higher-fidelity]
timestamp: "2026-07-13T06:15:00+05:30"
---

# EXP-609: Tier-6 straight-through confirmation

Freeze EXP-608 straight-through checkpoint `a87f4e...a793` as the no-update control. Train one challenger for 400 lower-rate updates with the same eight-step hard-forward/sigmoid-backward state estimator and 50/50 width-64/128 recovery allocation. Use fresh 512/128 train/test identities and identical source/challenger evaluations of 3,000 transitions plus 300 rollouts per width.

Pass only if the challenger reaches at least 90% width-128 rollout, gains at least two points over the frozen source, retains at least 95% width-64 rollout, reaches at least 99.8% width-128 transition, and keeps invalid width-128 rollouts at or below 1%. Passing authorizes five untouched official-generator Tier-6 seeds; failure closes lower-rate straight-through continuation.

## Dynamic resource budget

EXP-608's 200-update straight-through arm took 389.445 wall seconds including evaluation. Scaling training and using EXP-602's higher-fidelity evaluation throughput projects approximately 638 seconds for the challenger and 150 seconds for the frozen source, or 788 seconds total. Allocate 1,500 wall seconds soft and a 2,400-wall-second hard runaway ceiling, tracking aggregate CPU separately under a conservative 160 CPU-minute ceiling at four threads. Cache source and challenger independently and save the challenger before evaluation.

## Result

Reject and close straight-through continuation. On matched 300-case fresh evidence, the frozen source scores 84.67% width-128 and the challenger 86.33%, a +1.67-point gain versus +2 required and below 90% absolute. The challenger retains 96.67% width-64, reaches 99.93% transition, and has zero invalid rollouts. Runtime is 831.004 wall seconds and 3,282.228 aggregate CPU seconds; external wall is 837.35 seconds, within budget.

The smoke benefit did not reproduce at its declared magnitude. Further estimator continuation is not justified. Diagnose learner-state error geometry and test explicit bit/state corruption robustness rather than more clean-trajectory updates.
