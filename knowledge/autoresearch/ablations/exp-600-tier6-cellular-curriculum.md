---
type: Experiment Preregistration
title: EXP-600 Tier-6 Cellular Curriculum
description: Paired source-versus-progressive-curriculum width-128 smoke test using bigint bit tensors.
status: completed
tags: [autoresearch, Tier-6, cellular, curriculum, bigint]
timestamp: "2026-07-13T00:45:00+05:30"
---

# EXP-600: Tier-6 cellular curriculum

The frozen Tier-5 checkpoint unexpectedly scores 200/200 on a width-128 transition profile and 10/20 on autonomous width-128 rollouts with no invalid residues. Test whether 200 progressive updates across widths 64, 80, 96, and 112 improve matched width-128 rollout by at least ten points and reach at least 65%, while retaining at least 90% width-64 rollout and reaching at least 80% at width 112. Source and challenger use the same fresh identity split and evaluation seed. Checked, previously consumed independent, and private answers remain unread.

This is a preregistered smoke rung, not a Tier-6 promotion gate. Passing authorizes a direct width-128 continuation with its own higher-fidelity gate; failing closes this curriculum schedule but does not by itself falsify all width-128 cellular training.

## Dynamic resource budget

A representative ten-step width-128 forward/backward profile took 15.166 wall seconds and 60.478 aggregate CPU seconds at batch 128. Two hundred width-128 transitions took 0.614 wall / 2.462 CPU seconds; twenty full rollouts took 8.442 wall / 33.761 CPU seconds. Peak RSS was 1,524,800 KiB. Scaling the training segment and the paired source/challenger plus retention evaluations projects approximately 440 wall seconds. Allocate 720 wall seconds soft and a 1,200-wall-second hard runaway ceiling, with four threads and a conservative 80 aggregate CPU-minute upper ceiling. Save and hash the trained checkpoint before long evaluation, and cache the matched frozen-source evidence by checkpoint/config/dataset hashes.

## Result

Reject. On the matched 100-rollout width-128 set, the frozen source scores 57% while the challenger scores 49%, an eight-point regression. The challenger reaches 63% at width 112 and exactly 90% retained width-64 rollout; width-128 transition remains 99.7%, so the failure is autonomous error accumulation rather than single-step fitting. The run consumes 297.323 wall seconds and 1,157.011 aggregate CPU seconds, inside the dynamic budget, and saves checkpoint `6ca724...ce0c` before evaluation.

The strong 57% untrained width-128 transfer remains actionable evidence. Test a matched width-128 autonomous-state-recovery intervention from the unchanged source rather than more teacher-forced progressive transition training.
