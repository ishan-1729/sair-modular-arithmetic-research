---
type: Experiment Preregistration
title: EXP-111 Full-Budget Hard-Mined Width 16
description: Authorized 2,400-step transfer and hard-mining run gating independent Tier-3 evaluation.
status: completed
tags: [autoresearch, full-budget, hard-example-mining, width-16, Tier-3]
timestamp: "2026-07-12T03:25:00+05:30"
---

# EXP-111 full-budget hard-mined width 16

## Authorization and hypothesis

EXP-110 cleared the preregistered full-budget gate with 99.095% transition and 87.41% rollout. Hypothesis: keeping the validated transfer initialization and hard-mining mixture while extending the cosine schedule and training to 2,400 steps will cross 90% exact heldout rollout.

## Contract

- Same architecture, transfer source, prime split, generator seed, hard-mining mixture, losses, batch size, and CPU threads as EXP-110.
- Train from initialization for 2,400 steps with a 2,400-step cosine schedule; do not continue from the selected 1,200-step checkpoint.
- Evaluate 50,000 uniform heldout transitions and 20,000 uniform exact rollouts.
- Do not read or evaluate checked public Tier-3 answers.

## Gate

Require at least 99.5% transition, 90% exact rollout, and no more than 1% invalid rollouts. Passing authorizes five independent official-generator-style Tier-3 seeds. Failing returns to a separately preregistered mixture or rollout-aware objective; it does not authorize public tuning.

## Outcome

Passed: 99.908% transition, 98.895% exact rollout, and 0.145% invalid rollouts. The frozen checkpoint SHA-256 is `f9778b636f70d7cca56514ea98f95a22ce2fe81abcc4bd57f417274bc1e4bb94`.
