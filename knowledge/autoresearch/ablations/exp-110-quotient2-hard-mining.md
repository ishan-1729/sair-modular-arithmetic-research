---
type: Ablation Preregistration
title: EXP-110 Quotient-2 and Boundary Hard Mining
description: Matched width-16 transfer experiment oversampling the dominant quotient-2 and near-boundary transition errors.
status: completed
tags: [autoresearch, ablation, hard-example-mining, quotient, Tier-3]
timestamp: "2026-07-12T03:10:00+05:30"
---

# EXP-110 quotient-2 and boundary hard mining

## Evidence and hypothesis

On 100,000 heldout transitions, EXP-106 transfer is essentially exact for quotient 0 (99.75%) and quotient 1 (98.91%) but only 50.09% for quotient 2. Bit-0 transitions are 100% while bit-1 transitions are 87.67%. Hypothesis: uniform transition sampling underweights the third reduction regime, and a targeted offline curriculum will raise exact rollout more efficiently than additional uniform steps.

## Matched intervention

- Baseline: immutable EXP-106 width-16 transfer at 1,200 steps.
- Challenger: same transfer initialization, model/data seeds, heldout primes, architecture, optimizer, schedule, losses, batch size, network training steps, CPU threads, and evaluation cases.
- Only the training sampler changes. For each 512-example network batch, generate a deterministic 4x offline candidate pool, select 35% quotient-2 cases, 15% remaining cases within 5% of a reduction boundary, and fill the rest uniformly.
- Evaluation remains untouched and uniformly sampled. Public expected answers remain unread and unevaluated.

The larger offline candidate pool is disclosed as unmatched label-generation work; network forward/backward examples and optimization steps remain identical.

## Gates

The curriculum passes only if transition improves by at least 2 points, rollout improves by at least 10 points, and invalid rollout rate does not increase over EXP-106 transfer (94.335%, 47.07%, 2.90%).

If the challenger reaches at least 97% transition and 60% rollout, authorize a 2,400-step hard-mined run. If rollout reaches 90%, advance directly to multiple independent Tier-3 generator-style seeds. Otherwise reject the mixture or refine it in a separately preregistered A/B.

## Result

The intervention passed decisively: 99.095% transition and 87.41% rollout, gains of 4.76 and 40.34 points over matched uniform training. Invalid rollouts fell from 2.90% to 0.72%. It clears the full-budget gate but remains 2.59 points below the independent-seed gate. Decision: run 2,400 hard-mined steps, with public Tier 3 still untouched.
