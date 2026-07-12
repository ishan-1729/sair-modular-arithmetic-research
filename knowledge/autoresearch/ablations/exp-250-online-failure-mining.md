---
type: Experiment Preregistration
title: EXP-250 Online Exact-Failure Mining
description: Matched model-dependent hard-example mining against uniform replay at the distributed width-32 error tail.
status: invalid-resource-cap
tags: [autoresearch, hard-mining, online-mining, Tier-4, rollout]
timestamp: "2026-07-12T09:30:00+05:30"
---

# EXP-250: online exact-failure mining

EXP-240 found no individually critical hidden channel and localized 32 of 46 fresh transition errors to quotient class 1. This supports a distributed data/loss intervention rather than channel surgery or more quotient-2 specialization.

Both arms begin from frozen EXP-230 single-head checkpoint `c26e34af9581b6b0f3d26e72715c512823b07250bee48a56ae63cf32cffe5baa`, use fresh official-like prime identities, identical candidate pools, seeds, batch size, optimizer, loss, 400 updates, and evaluation cases. Every step generates eight candidate batches and performs the same no-gradient scoring pass in both arms.

- Control selects a uniformly random batch from the candidate pool.
- Challenger selects half its batch from exact model failures then highest per-example bit loss, and half uniformly from the remainder. The uniform half protects the learned global transition distribution.

All targets and failure indicators are offline training signals. Inference remains the unchanged trained TCN with no exact arithmetic or output correction. Require at least +0.1 transition point, +2 rollout points, 99.9% challenger transition, 94% challenger rollout, and at most 0.5% invalid rollouts. Passing authorizes fresh independent Tier-4 seeds; failure rejects this online-mining formulation. No consumed independent seed or checked-public answer may be read.

## Outcome

Invalid, not a model result. The 8x-pool control completed 400 updates but had not completed its evaluation when the matched run reached the preregistered ten-minute wall cap. The process was stopped and both MLflow runs were marked `KILLED`. No checkpoint or partial metric is eligible for selection. EXP-251 keeps the hypothesis but applies a 100-step, 4x-pool successive-halving smoke test with smaller fresh evaluations.
