---
type: Experiment Preregistration
title: EXP-251 Online Failure-Mining Smoke Test
description: Successive-halving retry of EXP-250 with a four-times pool, 100 updates, and smaller fresh evaluations.
status: completed
tags: [autoresearch, hard-mining, online-mining, smoke, Tier-4]
timestamp: "2026-07-12T09:48:00+05:30"
---

# EXP-251: compute-bounded online failure-mining smoke test

EXP-250 was stopped without a result because the 8x-pool matched run exceeded its ten-minute cap during control evaluation. EXP-251 tests the same causal hypothesis at lower fidelity: 100 updates, a 4x candidate pool, 20,000 fresh transitions, and 5,000 fresh autonomous rollouts per arm.

Both arms start from the same frozen EXP-230 control, score identical candidate pools, and differ only in selection. The control samples uniformly. The challenger selects half exact failures/highest-loss rows and half uniform replay. Require +0.05 transition point, +1 rollout point, at least 99.8% transition, 92% rollout, and no more than 0.5% invalid rollouts. A pass only authorizes a larger confirmation; it cannot authorize independent or checked evaluation. Stop at six wall minutes.

## Outcome

Rejected under the exact gate. Online mining improved rollout 91.90% -> 92.86% (+0.96 point), missing the +1.00-point threshold by 0.04 point. Transition changed 99.78% -> 99.77%, missing both the positive-gain and 99.8% absolute gates. Invalid rollouts remained acceptable at 0.22%. One replay-heavy confirmation is warranted because the directional rollout effect was large and the miss narrow; this checkpoint is not eligible for independent evaluation.
