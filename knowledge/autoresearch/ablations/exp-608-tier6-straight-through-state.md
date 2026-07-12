---
type: Experiment Preregistration
title: EXP-608 Tier-6 Straight-Through State
description: Matched continuous-soft versus hard-forward sigmoid-backward state-credit A/B.
status: preregistered
tags: [autoresearch, Tier-6, straight-through, credit-assignment, matched-ab]
timestamp: "2026-07-13T05:40:00+05:30"
---

# EXP-608: Tier-6 straight-through state

Start both arms from exact EXP-607 soft-state checkpoint `191598...2fef`. Match fresh identities, samplers, optimizer, 200 updates, eight-step horizon, and evaluation cases. The control continues continuous `sigmoid(logits)` state planes. The challenger uses hard thresholded bits in the forward pass with `hard + sigmoid(logits) - stop_gradient(sigmoid(logits))`, preserving inference-like trajectories while carrying sigmoid gradients backward.

Pass only if the challenger reaches at least 90% width-128 rollout, gains at least two points over continuous-soft control, retains at least 95% width-64 rollout, reaches at least 99.8% width-128 transition, and keeps invalid width-128 rollout at or below 1%. Passing authorizes higher-fidelity confirmation; failure closes this straight-through formulation.

## Dynamic resource budget

EXP-607's soft arm took 486.049 wall seconds including training and evaluation. Two matched differentiable arms project 972.099 seconds plus 25 seconds setup, or 997.099 seconds total. Allocate 1,600 wall seconds soft and a 2,400-wall-second hard runaway ceiling, with a separate conservative 180 aggregate CPU-minute ceiling at four threads. Cache completed arms and save checkpoints before evaluation.
