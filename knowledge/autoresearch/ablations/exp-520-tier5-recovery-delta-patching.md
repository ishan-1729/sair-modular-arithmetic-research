---
type: Experiment Preregistration
title: EXP-520 Tier-5 Recovery Delta Patching
description: Fresh-data causal tensor-group transplants between matched Tier-5 cellular checkpoints.
status: preregistered
tags: [autoresearch, Tier-5, causal-patching, cellular, mechanistic]
timestamp: "2026-07-12T23:10:00+05:30"
---

# EXP-520: Tier-5 recovery delta patching

Construct eight frozen width-64 arms by transplanting exact tensor groups from the EXP-329 recovery challenger into its matched teacher-forced control: endpoints, quotient-head negative control, input/state, recurrent gate, recurrent candidate, recurrent core, and readout. Save and hash every hybrid before evaluation. Evaluate all arms on identical fresh seed-20260725 prime identities with 1,000 transitions and 500 autonomous rollouts. Retain raw per-case rollout correctness and invalid flags for paired analysis.

The quotient-only arm must be prediction-identical to control; a mismatch invalidates the implementation. A non-endpoint patch advances only if it reaches at least 99.8% transition and 94% rollout, gains at least one exact rollout point over the full challenger, has at most 0.5% invalid rollouts, and its one-sided exact McNemar/binomial p-value versus the full challenger is at most 0.10 on discordant paired cases. Passing authorizes five untouched Tier-5 seeds for that exact hybrid. Failure establishes whether the recovery effect is localized, distributed, or countervailing without reopening recovery training itself. Checked, consumed independent, and private answers remain unread.

## Dynamic resource budget

EXP-328 measured 151.839 and 161.297 wall seconds for full-fidelity single-model arms at 1,000 transitions and 500 width-64 rollouts. Using the slower 161.297-second arm, eight patches project to 1,290.380 evaluation seconds; add 60 seconds for setup/checkpointing for 1,350.380 projected wall seconds. Allocate a 2,100-wall-second soft budget and 3,000-wall-second hard runaway ceiling. These are wall limits; four threads imply a conservative 200 aggregate CPU-minute hard upper bound. Cache completed arms by endpoint hashes, group, evaluator, dataset, seed, and counts.
