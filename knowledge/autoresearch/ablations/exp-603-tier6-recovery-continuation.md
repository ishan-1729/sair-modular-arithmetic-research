---
type: Experiment Preregistration
title: EXP-603 Tier-6 Recovery Continuation
description: One lower-rate fresh-identity continuation from the preserved EXP-602 checkpoint.
status: preregistered
tags: [autoresearch, Tier-6, cellular, autonomous-state, continuation]
timestamp: "2026-07-13T02:40:00+05:30"
---

# EXP-603: Tier-6 recovery continuation

Continue once from exact checkpoint `eaeece...59fa` using 600 updates at half the EXP-602 learning rate. Retain the same cycle of width-64 teacher transition, width-64 recovery, and two width-128 recovery updates, but begin scheduled learner-state exposure at 90%. Use fresh 512/128 train/test prime identities and matched frozen-source/challenger evaluations of 3,000 transitions plus 300 rollouts per width.

Pass only if width-128 rollout reaches at least 85%, gains at least three points over the source on identical fresh cases, width-64 rollout remains at least 92%, width-128 transition remains at least 99.7%, and invalid width-128 rollouts remain at or below 1%. Passing authorizes five untouched official-generator Tier-6 seeds. Failure closes simple continuation of this recovery schedule and requires a mechanistically different intervention.

## Dynamic resource budget

EXP-602's observed 400-update training phase took 634.522 wall seconds; scaling to 600 updates projects 951.783 seconds. Its paired evaluation/setup phase took 452.608 seconds, yielding a 1,404.391-second total projection. Allocate 2,100 wall seconds soft and a 3,000-wall-second hard runaway ceiling. Track aggregate CPU separately under a conservative 220 CPU-minute ceiling at four threads. Reuse the validated EXP-602 runner, cache the source control only by exact hashes, and save the challenger checkpoint before evaluation.
