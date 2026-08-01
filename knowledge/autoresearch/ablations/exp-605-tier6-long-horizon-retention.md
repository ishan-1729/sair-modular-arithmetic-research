---
type: Experiment Preregistration
title: EXP-605 Tier-6 Long-Horizon Retention
description: Eight-step recovery continuation with increased width-64 allocation and higher-fidelity fresh evaluation.
status: completed
tags: [autoresearch, Tier-6, cellular, recovery-horizon, retention]
timestamp: "2026-07-13T03:50:00+05:30"
---

# EXP-605: Tier-6 long-horizon retention

Continue from exact EXP-604 eight-step checkpoint `088264...a6c5`. Use 400 updates at lower learning rate with a five-update cycle: one width-64 teacher transition, two width-64 eight-step recovery updates, and two width-128 eight-step recovery updates. This changes the allocation from 50/50 to 60/40 width-64/128 while retaining the causally supported horizon. Use fresh 512/128 train/test identities and matched source/challenger evaluations of 3,000 transitions plus 300 rollouts per width.

Pass only if width-128 rollout reaches at least 90%, gains at least three points over the source, width-64 rollout reaches at least 94%, width-128 transition reaches at least 99.8%, and invalid width-128 rollouts remain at or below 1%. Passing authorizes five untouched official-generator Tier-6 seeds; failure closes this retention-adjusted horizon continuation.

## Dynamic resource budget

EXP-604's eight-step 200-update arm took 333.977 wall seconds including 201.051 seconds through its final training milestone. Doubling training projects 402.102 seconds; EXP-602's paired 300-rollout evaluation/setup phase measured 452.608 seconds. Total projected wall time is 854.710 seconds. Allocate 1,500 wall seconds soft and a 2,400-wall-second hard runaway ceiling, with a separate conservative 160 aggregate CPU-minute ceiling at four threads. Reuse the validated EXP-602 runner and cache only exact-hash source evidence.

## Result

Reject and close this continuation. On matched fresh evidence, width-64 rollout improves from 95.0% to 96.67%, but width-128 decreases from 82.67% to 82.33%. Width-128 transition remains 99.87% and invalid rollouts 0.33%. Internal wall time is 791.004 seconds and aggregate CPU is 3,130.059 seconds; external wall time is 797.78 seconds, within budget. The checkpoint is preserved but is not promotion-eligible.

Further schedule tuning is not justified. Analyze fresh per-case complementarity among the preserved four-step and eight-step checkpoints before deciding whether a weight-space merge or logit ensemble has ranking value and acceptable inference cost.
