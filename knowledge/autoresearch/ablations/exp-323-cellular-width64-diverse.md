---
type: Experiment Preregistration
title: EXP-323 Diverse Width-64 Cellular Continuation
description: Test 16x more training-prime identities and a lower learning rate at the Tier-5 boundary.
status: completed
tags: [autoresearch, cellular, width-64, diversity, Tier-5]
timestamp: "2026-07-12T13:15:00+05:30"
---

# EXP-323: diverse width-64 continuation

EXP-322 slightly regressed despite near-perfect transition accuracy while using only 128 train primes. Repeat its 400-step width-56/64 curriculum from frozen EXP-320 with 2,048 train and 128 test primes per width and learning rate `3e-4`. This changes only identity diversity and update scale; all inference and evaluation boundaries remain unchanged.

Require 85% retained width-56 rollout; 99.75% width-64 transition; 90% width-64 rollout; at least five rollout points over the paired frozen source; and at most 0.5% invalid rollouts. At least 93% rollout authorizes five untouched Tier-5 seeds; a pass below 93% authorizes one confirmation. No checked, consumed independent, or private answers may be read.

## Outcome

Rejected narrowly but directionally positive. Width-64 rollout improves 85.2% -> 88.533% (+3.333 points), transition reaches 99.78%, and width-56 rollout reaches 93.667%. Invalid rollouts are 0.533%, just above 0.5%. Continue this checkpoint once on fresh identities at `2e-4`; require at least 92% absolute rollout and a three-point paired gain.
