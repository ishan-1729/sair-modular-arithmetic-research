---
type: Ablation Preregistration
title: EXP-105 Learned Width-Transfer Initialization
description: Matched test of transferring learned width-9 filters and overlapping low-bit parameters into the width-16 challenger.
status: completed
tags: [autoresearch, ablation, transfer-learning, width-scaling, Tier-3]
timestamp: "2026-07-12T02:35:00+05:30"
---

# EXP-105 learned width transfer

## Evidence and hypothesis

EXP-100 showed that the shared-context width-9 model learns rapidly while random-initialized width 16 is still improving but rollout-poor. Most convolution, normalization, local-head, and context tensors have identical shapes. Hypothesis: those learned tensors encode reusable transition features, and copying them plus overlapping LSB-aligned parameters will improve width-16 optimization and state validity.

## Matched comparison

- Control: the immutable EXP-100 width-16 random-initialized arm (`556b4575...`) at 400 steps.
- Challenger: identical width-16 model, train/test prime split, data seed, model seed, 400 steps, batch size, optimizer, schedule, losses, CPU threads, and evaluation cases.
- Only initialization changes. Same-shape learned tensors copy exactly from the EXP-100 width-9 control (`6f8d3598...`). Width-shaped tensors copy their overlapping LSB-aligned blocks; new high-bit regions retain the deterministic width-16 random initialization.
- Public expected answers remain unread and unevaluated.

## Gates

Transfer passes the smoke gate only if all are true:

1. heldout transition improves by at least 5 absolute percentage points over 64.31%;
2. exact 16-step rollout reaches at least 1.0%;
3. invalid-rollout rate does not exceed the 54.82% control;
4. the final-100-step batch-exact delta remains positive.

If it passes, run a matched 1,200-step random-versus-transfer continuation before any public Tier-3 evaluation. Otherwise reject transfer and move to trajectory/boundary curriculum.

## Result

All gates passed. At matched 400 steps, transfer improved transition by 22.94 points (87.25% versus 64.31%), rollout by 23.60 points (24.12% versus 0.52%), and reduced invalid rollouts by 43.64 points (11.18% versus 54.82%). Decision: run the preregistered matched 1,200-step comparison.
