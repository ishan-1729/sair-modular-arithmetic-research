---
type: Experiment Preregistration
title: EXP-500 Cellular Grokking Probe
description: Three-seed matched finite-data no-decay versus weight-decay delayed-generalization study.
status: completed
tags: [autoresearch, grokking, cellular, delayed-generalization, multi-seed]
timestamp: "2026-07-12T19:20:00+05:30"
---

# EXP-500: cellular grokking probe

For seeds 20260719–20260721, freeze 2,048 width-9 training transitions and 4,096 prime-disjoint heldout transitions. Train identical cellular initializations for 8,000 updates under two matched arms: AdamW with no decay and AdamW with decay 1e-2. Both use the same finite data, minibatch index sequence, 1e-3 learning rate after ten-step warmup, betas 0.9/0.98, batch 128, and exact offline labels. This finite surface permits memorization; it is intentionally different from the project's usual infinite samplers.

Every 200 steps record full fixed-train and heldout exact transition accuracy, loss, signed margin, total parameter norm, distance from initialization, and gradient diagnostics. Every 1,000 steps record fixed 500-case heldout autonomous rollout. Save recoverable checkpoints every 500 steps plus final weights. Preserve dense curves even when no signature occurs.

The binding operational signature is defined in the grokking research synthesis. A seed must first reach at least 99% train exact while heldout is at most 80% with a gap of at least 15 points. At least 1,000 steps later it must retain 99% train exact, reach at least 95% heldout with at least 15 points of heldout gain, and reach at least 80% rollout with at least 20 points gain from initialization. The decay intervention passes only if at least two of three seeds meet the complete signature and its signature count exceeds the no-decay control. Partial late improvement is recorded as semi-grokking, not promoted. Passing authorizes a mechanistically informed Tier-5 A/B; it does not promote a candidate.

## Dynamic resource budget

On the active four-thread CPU, 100 representative width-9 batch-128 updates measure 11.855 wall / 43.109 aggregate CPU seconds; 2,048 transitions measure 0.664 / 2.318 seconds; 500 rollouts measure 1.224 / 4.451 seconds. Six 8,000-step arms project to 5,690.380 training wall seconds. Forty-one paired train/heldout curve evaluations per arm project to 489.713 seconds, nine rollout evaluations per arm to 66.084 seconds, and checkpoint/setup overhead to 60 seconds, totaling 6,306.176 projected wall seconds. Allocate a 9,000-wall-second soft budget and 12,000-wall-second hard runaway ceiling. These are wall limits; four threads imply a conservative 800 aggregate CPU-minute hard upper bound. Integrity-key and reuse only completed arms.

## Outcome

Rejected: zero of three no-decay seeds and zero of three decay seeds satisfy the complete signature. Seeds 20260719 and 20260720 reach the required train/heldout memorization gap at step 1,000, but heldout later gains only about three to five points rather than fifteen, never reaches 95%, and rollout plateaus below 40%. Seed 20260721 already exceeds the preregistered 80% heldout ceiling at its first 99% train point, so it exhibits ordinary concurrent generalization rather than grokking. No semi-grokking or ungrokking flag fires.

Mean final heldout is 82.959% control versus 83.586% decay; mean rollout is 30.667% versus 32.467%. Paired decay effects are inconsistent: heldout +0.122/−0.952/+2.710 points and rollout −0.4/−0.2/+6.0 points. Do not select the favorable third seed or transfer decay to Tier 5. The runner records 5,544.2 wall seconds and 21,324.4 aggregate CPU seconds; external timing records 5,378.0 elapsed seconds and 21,336.5 user-plus-system CPU seconds. The full probe completes inside budget.
