---
type: Ablation Study
title: EXP-666 GRAM posterior-to-prior guidance
description: Frozen routes, calibration-only observations, runtime-headroom park, and unopened scientific controls.
status: parked
tags: [autoresearch, ablation, gram, variational-inference, resource-gate]
timestamp: "2026-07-29T01:00:00+05:30"
---

# EXP-666 GRAM posterior-to-prior guidance

The bounded ST-M022.a package defines seven frozen routes: proposal-only, full posterior-to-prior GRAM, deterministic prior mean, target-posterior guidance without KL, stochasticity without target posterior, direct prediction, and randomized guidance. The five trainable routes have 175,600 total and 75,776 trainable parameters and share a 400-update formal allowance.

Root `20261214` completed only full-route resource/integrity calibration. All routes persisted and reloaded exactly; split separation, fixed-seed replay, target-free single-trajectory inference, memory, expected-value, and nominal candidate runtime checks passed. The candidate projection was 226.63 wall seconds, but measured update variability produced headroom `1.86796` and a 423.33-second headroom bound, above the frozen 285-second ceiling. Formal root `20261215` was therefore not opened.

The tiny two-update/eight-transition/two-rollout calibration values are not scientific ablation evidence. W128 exactness, validity, retention, trained/randomized dependence, variational causal effects, the deterministic/guidance/stochasticity/direct comparisons, W256 transfer, confirmations, and M022.b breadth remain unresolved and unopened. The terminal decision is a runtime-headroom park, not rejection of GRAM, ST-M022.a, or the source paper.
