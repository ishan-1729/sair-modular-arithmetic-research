---
type: Experiment Preregistration
title: EXP-301 Width-64 Hard-Mined Continuation
description: Full-budget quotient-2 and reduction-boundary curriculum for the transferred bit-tensor model.
status: preregistered
tags: [autoresearch, width-64, hard-mining, transfer, Tier-5]
timestamp: "2026-07-12T10:38:00+05:30"
---

# EXP-301: width-64 hard-mined continuation

EXP-300 transfer reached 82.38% transition but zero autonomous rollout after 400 steps. Continue the frozen transfer checkpoint for 1,200 steps on new prime identities. From a 4x candidate pool, construct each batch with 35% quotient-2 cases, 15% cases within 5% of either reduction boundary, and random replay for the remainder. Exact arithmetic remains confined to offline labels and selection.

Evaluate the frozen source and continuation on the same 20,000 fresh transitions; evaluate the trained model on 5,000 autonomous residue rollouts. Require +12 transition points, 95% transition, 15% rollout, and no more than 10% invalid rollouts. Passing authorizes a 2,400-step confirmation. Failing closes this fixed-width width-64 TCN. Stop at ten wall minutes.
