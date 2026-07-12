---
type: Experiment Preregistration
title: EXP-500 Cellular Grokking Probe
description: Three-seed matched finite-data no-decay versus weight-decay delayed-generalization study.
status: preregistered
tags: [autoresearch, grokking, cellular, delayed-generalization, multi-seed]
timestamp: "2026-07-12T19:20:00+05:30"
---

# EXP-500: cellular grokking probe

For seeds 20260719–20260721, freeze 2,048 width-9 training transitions and 4,096 prime-disjoint heldout transitions. Train identical cellular initializations for 8,000 updates under two matched arms: AdamW with no decay and AdamW with decay 1e-2. Both use the same finite data, minibatch index sequence, 1e-3 learning rate after ten-step warmup, betas 0.9/0.98, batch 128, and exact offline labels. This finite surface permits memorization; it is intentionally different from the project's usual infinite samplers.

Every 200 steps record full fixed-train and heldout exact transition accuracy, loss, signed margin, total parameter norm, distance from initialization, and gradient diagnostics. Every 1,000 steps record fixed 500-case heldout autonomous rollout. Save recoverable checkpoints every 500 steps plus final weights. Preserve dense curves even when no signature occurs.

The binding operational signature is defined in the grokking research synthesis. A seed must first reach at least 99% train exact while heldout is at most 80% with a gap of at least 15 points. At least 1,000 steps later it must retain 99% train exact, reach at least 95% heldout with at least 15 points of heldout gain, and reach at least 80% rollout with at least 20 points gain from initialization. The decay intervention passes only if at least two of three seeds meet the complete signature and its signature count exceeds the no-decay control. Partial late improvement is recorded as semi-grokking, not promoted. Passing authorizes a mechanistically informed Tier-5 A/B; it does not promote a candidate.

## Dynamic resource budget

On the active four-thread CPU, 100 representative width-9 batch-128 updates measure 11.855 wall / 43.109 aggregate CPU seconds; 2,048 transitions measure 0.664 / 2.318 seconds; 500 rollouts measure 1.224 / 4.451 seconds. Six 8,000-step arms project to 5,690.380 training wall seconds. Forty-one paired train/heldout curve evaluations per arm project to 489.713 seconds, nine rollout evaluations per arm to 66.084 seconds, and checkpoint/setup overhead to 60 seconds, totaling 6,306.176 projected wall seconds. Allocate a 9,000-wall-second soft budget and 12,000-wall-second hard runaway ceiling. These are wall limits; four threads imply a conservative 800 aggregate CPU-minute hard upper bound. Integrity-key and reuse only completed arms.
