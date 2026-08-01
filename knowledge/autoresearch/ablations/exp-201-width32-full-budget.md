---
type: Experiment Preregistration
title: EXP-201 Full-Budget Width-32 Gate
description: Full-budget transferred hard-mining run gating independent Tier-4 evaluation.
status: completed
tags: [autoresearch, Tier-4, width-32, full-budget, hard-mining]
timestamp: "2026-07-12T10:05:00+05:30"
---

# EXP-201: full-budget width-32 gate

EXP-200 established a causal initialization effect at matched compute: transfer improved width-32 rollout from 0% to 61.68% after 400 steps. EXP-201 tests whether the same frozen width-16 initialization and curriculum can cross the independent-evaluation threshold at 1,600 steps.

## Frozen contract

- Initialize directly from frozen EXP-111 width-16 checkpoint `f9778b636f70d7cca56514ea98f95a22ce2fe81abcc4bd57f417274bc1e4bb94`; do not continue the selected EXP-200 checkpoint.
- Preserve EXP-200 architecture, sampled prime-identity split, hard-mining proportions, losses, optimizer, batch size, and seed.
- Train for 1,600 steps with a 1,600-step cosine schedule.
- Evaluate 50,000 uniform heldout transitions and 20,000 autonomous rollouts.
- Do not read or evaluate checked public Tier-4 answers.

## Gate

Require at least 99.7% transition accuracy, 90% exact rollout, and no more than 1% invalid rollouts. Passing authorizes five independent official-generator Tier-4 seeds. Failing returns to a separately preregistered rollout-aware or width-independent intervention and does not authorize public tuning.

## Outcome

Narrow gate miss. The run reached 99.648% transition, 90.465% rollout, and 0.51% invalid rollouts. Rollout and validity passed, but transition missed its frozen threshold by 0.052 percentage points. Because the final training slope remained positive, one 2,400-step confirmation is authorized under a new preregistration; the independent generator remains locked.
