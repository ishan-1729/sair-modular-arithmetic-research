---
type: Experiment Preregistration
title: EXP-326 Cellular Weight-Soup Retry
description: Resource-corrected five-alpha Tier-5 interpolation test at 500 rollouts per arm.
status: completed
tags: [autoresearch, cellular, width-64, model-soup, retry]
timestamp: "2026-07-12T14:05:00+05:30"
---

# EXP-326: cellular weight-soup retry

Repeat EXP-325 with identical checkpoints, alpha grid, and fresh identity split, but use 1,000 transitions and 500 rollouts per arm. No comparative arm from the stopped run is consulted.

Keep the same gates: 99.8% transition, 93% rollout, at least one point over the better endpoint, and no more than 0.5% invalid rollouts. Passing authorizes five untouched Tier-5 seeds. Failure closes interpolation. No checked, consumed independent, or private answers may be read.

## Outcome

Rejected. The 50/50 soup is best at 99.9% transition and 91.2% rollout, improving exactly one point over the better endpoint with 0.2% invalid rollouts. It misses the 93% absolute gate. Close weight interpolation; test a two-checkpoint logit ensemble, which can preserve complementary nonlinear recurrent trajectories at the cost of doubling only the Tier-5 specialist computation.
