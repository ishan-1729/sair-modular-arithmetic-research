---
type: Ablation Preregistration
title: EXP-106 Matched 1,200-Step Width Transfer
description: Full matched random-versus-transfer comparison at the predeclared intermediate training budget.
status: completed
tags: [autoresearch, ablation, transfer-learning, width-scaling, Tier-3]
timestamp: "2026-07-12T02:45:00+05:30"
---

# EXP-106 matched 1,200-step width transfer

## Hypothesis

The large transfer advantage observed at 400 steps persists rather than merely shifting the learning curve earlier, and the transferred width-16 model reaches an exact-rollout regime worth the full 2,400-step budget.

## Contract

- Train random-initialized and width-9-transferred width-16 models from step 0 for 1,200 steps.
- Match architecture, width-16 train/test prime identities, generator seed, model seed, examples, optimizer, schedule, losses, batch size, CPU threads, and evaluation cases.
- Evaluate 20,000 heldout transitions and 10,000 exact 16-step rollouts per arm.
- Do not read or evaluate checked public Tier-3 answers.

## Gates

Select transfer over random only if rollout improves by at least 5 points, transition is no more than 1 point worse, and invalid rollout rate does not increase. Select random if transfer fails and random is stronger.

Authorize a 2,400-step run for the winning initialization only if it reaches at least 97% transition and 60% rollout. A model reaching 90% rollout proceeds instead to multiple independent generator-style Tier-3 seeds. Anything weaker returns to curriculum/state-validity interventions.

## Result

Transfer remained the winner: 94.335% transition and 47.07% rollout versus 87.76% and 23.48% for random initialization. It did not clear the absolute 97%/60% gate. Decision: retain transfer, diagnose its errors, and test quotient-2/boundary hard-example sampling before a full-budget run.
