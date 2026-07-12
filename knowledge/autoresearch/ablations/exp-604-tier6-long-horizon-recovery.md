---
type: Experiment Preregistration
title: EXP-604 Tier-6 Long-Horizon Recovery
description: Matched four-step versus eight-step learner-state recovery comparison at width 128.
status: preregistered
tags: [autoresearch, Tier-6, cellular, recovery-horizon, matched-ab]
timestamp: "2026-07-13T03:20:00+05:30"
---

# EXP-604: Tier-6 long-horizon recovery

Start both arms from exact EXP-603 checkpoint `4cf8f9...46ca` on identical fresh identities, mode order, optimizer steps, learning rate, and evaluation cases. The control retains four recovery steps at batch 32; the challenger uses eight recovery steps at batch 16. Both therefore expose 128 trajectory-step examples per recovery update, while the challenger intentionally performs twice as many sequential model calls. This isolates recovery horizon at matched examples, not matched compute.

Run 200 updates and evaluate each frozen arm on 2,000 transitions plus 200 rollouts at widths 64 and 128. Pass only if the challenger reaches at least 88% width-128 rollout, gains at least two points over the matched control, retains at least 94% width-64 rollout, reaches at least 99.8% width-128 transition, and keeps invalid width-128 rollouts at or below 1%. Passing authorizes a higher-fidelity confirmation or independent Tier-6 only if the exact absolute threshold is met; checked answers remain untouched.

## Dynamic resource budget

A ten-update eight-step/batch-16 profile took 19.928 wall seconds and 79.757 aggregate CPU seconds with 1,322,336 KiB peak RSS. Combining that measurement with EXP-601/602 phase throughput projects 535 wall seconds for both training arms and 302 seconds for matched evaluation/setup, or 837 seconds total. Allocate 1,500 wall seconds soft and a 2,400-wall-second hard runaway ceiling, with a separate conservative 160 aggregate CPU-minute ceiling at four threads. Cache completed arms by source/config/dataset/arm hashes and save each checkpoint before evaluation.
