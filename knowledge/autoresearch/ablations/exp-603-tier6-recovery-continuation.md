---
type: Experiment Preregistration
title: EXP-603 Tier-6 Recovery Continuation
description: One lower-rate fresh-identity continuation from the preserved EXP-602 checkpoint.
status: completed
tags: [autoresearch, Tier-6, cellular, autonomous-state, continuation]
timestamp: "2026-07-13T02:40:00+05:30"
---

# EXP-603: Tier-6 recovery continuation

Continue once from exact checkpoint `eaeece...59fa` using 600 updates at half the EXP-602 learning rate. Retain the same cycle of width-64 teacher transition, width-64 recovery, and two width-128 recovery updates, but begin scheduled learner-state exposure at 90%. Use fresh 512/128 train/test prime identities and matched frozen-source/challenger evaluations of 3,000 transitions plus 300 rollouts per width.

Pass only if width-128 rollout reaches at least 85%, gains at least three points over the source on identical fresh cases, width-64 rollout remains at least 92%, width-128 transition remains at least 99.7%, and invalid width-128 rollouts remain at or below 1%. Passing authorizes five untouched official-generator Tier-6 seeds. Failure closes simple continuation of this recovery schedule and requires a mechanistically different intervention.

## Dynamic resource budget

EXP-602's observed 400-update training phase took 634.522 wall seconds; scaling to 600 updates projects 951.783 seconds. Its paired evaluation/setup phase took 452.608 seconds, yielding a 1,404.391-second total projection. Allocate 2,100 wall seconds soft and a 3,000-wall-second hard runaway ceiling. Track aggregate CPU separately under a conservative 220 CPU-minute ceiling at four threads. Reuse the validated EXP-602 runner, cache the source control only by exact hashes, and save the challenger checkpoint before evaluation.

## Result

Reject and close simple continuation. The frozen source scores 252/300 (84.0%) width-128 rollout and the challenger scores 254/300 (84.67%): one case short of the exact 85% gate and only +0.67 points versus +3 required. Width-64 rollout is 97.33%, width-128 transition is 99.97%, and invalid rollout is 0.67%. Internal wall time is 1,268.665 seconds and aggregate CPU is 5,026.270 seconds; external wall time is 1,275.75 seconds, within budget. Checkpoint `4cf8f9...46ca` is preserved but independent Tier-6 remains unauthorized.

The four-step recovery horizon has plateaued while single-step transitions are nearly perfect. The next experiment must change recovery horizon or trajectory exposure rather than add more identical continuation updates.
