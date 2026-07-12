---
type: Experiment Preregistration
title: EXP-326 Cellular Weight-Soup Retry
description: Resource-corrected five-alpha Tier-5 interpolation test at 500 rollouts per arm.
status: preregistered
tags: [autoresearch, cellular, width-64, model-soup, retry]
timestamp: "2026-07-12T14:05:00+05:30"
---

# EXP-326: cellular weight-soup retry

Repeat EXP-325 with identical checkpoints, alpha grid, and fresh identity split, but use 1,000 transitions and 500 rollouts per arm. No comparative arm from the stopped run is consulted.

Keep the same gates: 99.8% transition, 93% rollout, at least one point over the better endpoint, and no more than 0.5% invalid rollouts. Passing authorizes five untouched Tier-5 seeds. Failure closes interpolation. No checked, consumed independent, or private answers may be read.
