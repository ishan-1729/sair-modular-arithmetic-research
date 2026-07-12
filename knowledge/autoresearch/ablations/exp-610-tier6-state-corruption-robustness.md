---
type: Experiment Preregistration
title: EXP-610 Tier-6 State-Corruption Robustness
description: Matched clean versus diagnosis-derived one-/two-bit corruption recovery A/B.
status: completed
tags: [autoresearch, Tier-6, state-corruption, robustness, matched-ab]
timestamp: "2026-07-13T06:50:00+05:30"
---

# EXP-610: Tier-6 state-corruption robustness

Start both arms from exact EXP-609 checkpoint `cbc4a1...d4bd` with identical fresh identities, samplers, straight-through estimator, eight-step horizon, optimizer, 200 updates, and evaluation cases. The control uses clean chunk-entry teacher states. The challenger flips one uniformly sampled state bit with probability 0.75 or two uniformly sampled bits with probability 0.25 at each width-128 recovery-chunk entry, matching the fresh diagnosis in which 76.9% of first errors are one or two bits and bit positions are diffuse. Width-64 recovery remains uncorrupted.

Pass only if the challenger reaches at least 90% width-128 rollout, gains at least three points over clean control, retains at least 95% width-64 rollout, reaches at least 99.8% width-128 transition, and keeps invalid width-128 rollout at or below 1%. Passing authorizes higher-fidelity confirmation; failure closes this corruption formulation.

## Dynamic resource budget

EXP-608/609 show roughly 390–625 wall seconds for straight-through arms at 200–400 updates and 200–300 rollout evaluations. Two 200-update/200-rollout arms project 800 wall seconds plus 25 seconds setup, or 825 seconds total. Allocate 1,500 wall seconds soft and a 2,400-wall-second hard runaway ceiling, tracking aggregate CPU separately under a conservative 170 CPU-minute ceiling at four threads. Cache completed arms and save checkpoints before evaluation.

## Result

Reject decisively. Clean straight-through control scores 87.5% width-128 rollout; corruption training collapses to 75.0% (-12.5 points), with 95.5% width-64, 99.85% transition, and 0.5% invalid rollouts. Corrupted recovery batches reach 0–6.25% exactness with losses around 0.9–1.25, showing that arbitrary residue perturbations do not admit the intended one-step correction mapping. Runtime is 735.396 wall seconds and 2,898.177 aggregate CPU seconds; external wall is 740.14 seconds, within budget.

Close generic state corruption. The fixed 92-channel cell has now exhausted transition continuation, detached recovery, longer horizons, continuous/straight-through credit, checkpoint combination, and corruption robustness. Test function-preserving hidden-channel widening next.
