---
type: Mechanistic Analysis
title: EXP-610 Tier-6 First-Error Geometry
description: Fresh width-128 first-divergence timing, Hamming size, bit positions, and margins for corruption design.
status: completed
tags: [autoresearch, Tier-6, first-error, state-corruption, mechanistic]
timestamp: "2026-07-13T06:45:00+05:30"
---

# Tier-6 first-error geometry

On 200 fresh width-128 trajectories, frozen EXP-609 checkpoint `cbc4a1...d4bd` scores 174/200 (87%). Among 26 failed trajectories, first divergence occurs from step 7 through 110 with median 50.5; only 19.2% first diverge in the last quarter. The error is therefore distributed through the rollout rather than a late-tail-only event.

First-error Hamming counts are: 18 one-bit, 2 two-bit, 1 three-bit, 1 five-bit, 2 eight-bit, 1 twenty-eight-bit, and 1 fifty-two-bit. Thus 76.9% of first errors are one or two bits, with median Hamming size one. Frequent bit positions are diffuse: no bit appears more than three times. Minimum absolute logit at first error has median 0.184, but ranges from 0.0017 to 1.342, so confidence thresholding alone cannot reliably identify failures.

Implication: test generic one-/two-bit recovery rather than position-specific repair. Use uniform bit positions, one-bit corruption with probability 0.75 and two-bit corruption with probability 0.25, injected at recovery-chunk entry across the full trajectory. This is offline training augmentation only; inference remains learned and unchanged.
