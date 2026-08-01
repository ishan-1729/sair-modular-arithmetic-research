---
type: Experiment Preregistration
title: EXP-601 Tier-6 Autonomous-State Recovery
description: Matched fresh-identity test of direct width-128 learner-state recovery from the frozen Tier-5 source.
status: completed
tags: [autoresearch, Tier-6, cellular, autonomous-state, matched-control]
timestamp: "2026-07-13T01:15:00+05:30"
---

# EXP-601: Tier-6 autonomous-state recovery

EXP-600 shows 99.7% width-128 transition accuracy but only 57% autonomous rollout for the unchanged source, and transition-only curriculum training regresses rollout to 49%. On a fresh identity split, compare the unchanged source with a challenger receiving alternating width-64 teacher-transition updates and width-128 four-step learner-state recovery updates. Source and challenger use identical evaluation cases. Pass only if the challenger reaches at least 68% width-128 rollout, gains at least ten points, preserves at least 90% width-64 rollout, retains at least 99.5% width-128 transition accuracy, and keeps invalid rollouts at or below 2%.

This is a smoke rung and does not authorize independent Tier-6 evaluation. Python bigint exact arithmetic is limited to offline state/label generation and evaluation; the trained model receives and emits bit tensors.

## Dynamic resource budget

EXP-600's observed 200-update training phase took 155.732 wall seconds. Apply EXP-329's preregistered 1.25 recovery-overhead factor for 194.665 training seconds, then add 105 seconds for fresh split construction and paired width-64/128 source/challenger evaluation, projecting 299.665 wall seconds. Allocate 600 wall seconds soft and a 900-wall-second hard runaway ceiling. Four threads imply a conservative 60 aggregate CPU-minute upper bound. Save the checkpoint before challenger evaluation and cache source evidence only by exact checkpoint/config/dataset hashes.

## Result

Pass. On matched fresh cases, width-128 rollout improves from 62% for the frozen source to 76% for the challenger (+14 points), transition remains 99.8%, and no invalid rollout occurs. Width-64 rollout is 90%, exactly at the retention gate. The run uses 356.546 wall seconds and 1,392.689 aggregate CPU seconds, within budget, and saves checkpoint `915534...82d8` before evaluation.

Advance the mechanism to a preregistered higher-fidelity/full-budget confirmation with stronger width-64 retention evidence. Do not consult checked or independent Tier-6 answers yet.
