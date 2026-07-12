---
type: Experiment Preregistration
title: EXP-607 Tier-6 Differentiable Soft State
description: Matched detached-hard-state versus differentiable-soft-state eight-step recovery A/B.
status: preregistered
tags: [autoresearch, Tier-6, cellular, differentiable-state, credit-assignment]
timestamp: "2026-07-13T05:00:00+05:30"
---

# EXP-607: Tier-6 differentiable soft state

Start both arms from exact EXP-604 eight-step checkpoint `088264...a6c5`. Match identities, samplers, mode order, optimizer, 200 updates, batch 16, eight recovery steps, and evaluation cases. In the control, each next state plane is the detached thresholded previous logits. In the challenger, it is `sigmoid(previous_logits)` without detachment, allowing losses at later steps to backpropagate through the full eight-step trajectory. Exact teacher states remain offline labels; model inference is unchanged.

Pass only if the challenger reaches at least 88% width-128 rollout, gains at least three points over the matched detached control, retains at least 94% width-64 rollout, reaches at least 99.8% width-128 transition, and keeps invalid width-128 rollouts at or below 1%. Passing authorizes higher-fidelity confirmation; failure closes this soft-state formulation.

## Dynamic resource budget

A five-update soft-state profile took 13.054 wall seconds and 52.725 aggregate CPU seconds with 1,449,312 KiB peak RSS. Combining this with EXP-604 detached-arm and evaluation throughput projects 610 wall seconds for both training arms and 260 seconds for evaluation/setup, or 870 seconds total. Allocate 1,500 wall seconds soft and a 2,400-wall-second hard runaway ceiling, with a distinct conservative 160 aggregate CPU-minute ceiling at four threads. Cache each completed arm and save checkpoints before evaluation.
