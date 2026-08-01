---
type: Experiment Preregistration
title: EXP-260 Low-Rate Uniform Continuation
description: Fresh-split continuation of the strongest uniform width-32 checkpoint without mining or routing.
status: completed
tags: [autoresearch, continuation, learning-rate, Tier-4]
timestamp: "2026-07-12T10:12:00+05:30"
---

# EXP-260: low-rate uniform continuation

EXP-252's uniform arm reached 93.26% rollout while online ranking added only 0.36 point. Continue that frozen uniform checkpoint for 600 updates at `5e-5` on a new official-like prime split, using ordinary random-state transitions and no hard selection. Evaluate the frozen source and trained checkpoint on identical fresh cases.

Require at least +1 rollout point over the frozen source, 99.85% transition, 94% rollout, and at most 0.5% invalid rollouts. Passing authorizes five untouched independent Tier-4 seeds. Failing closes further fixed-width Tier-4 continuation and redirects effort to the tensor-width architecture required by higher tiers. Stop at six wall minutes.

## Outcome

Rejected. On the fresh split, continuation improved transition 99.798% -> 99.826% and rollout 92.165% -> 92.795%, but missed the 99.85%, 94%, and +1-point gates. Fixed-width Tier-4 continuation is closed. The next architecture work removes the `int64` data path and tests width-32-to-width-64 transfer using bit tensors.
