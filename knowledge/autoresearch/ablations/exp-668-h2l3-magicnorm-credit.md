---
type: Ablation Study
title: EXP-668 H2L3 plus MagicNorm and warmup credit
description: Frozen complete-route calibration and conditional W128-first viability package.
status: parked
tags: [autoresearch, ablation, hrm-text, magicnorm]
timestamp: "2026-07-29T03:00:00+05:30"
---
# EXP-668 — H2L3 plus MagicNorm and warmup credit

The preregistered package tests M024.a/b around the frozen EXP609 proposal. Nine routes isolate full H2L3+credit, hierarchy-only, credit-only, flat compute matching, full raw-gradient BPTT, state resets, proposal-only, and randomized weights.

Focused verification passed 26 tests. The initial validator/full-project invocation hit its 904.037-second tool timeout without returning a verdict. That history remains preserved. A read-only Judge authorized exactly one identical-byte recovery under the binding dynamic-budget policy: validation passed 182 concepts/1,100 records and the full oracle passed 557 tests in 602.93 seconds (622.62 outer wall, 729.71 user, 35.03 system, 924,512 KiB RSS).

Root 20261222 then completed every calibration route and split. Integrity, expected value, memory, determinism, reload, separation, gradient finiteness, and target-free fixed H2L3 passed. Nominal candidate inference projected to 277.13 seconds, but the measured 1.75766 variability multiplier raised its headroom bound to 487.09 seconds against the frozen 285-second ceiling. The package therefore parked before root 20261223. Calibration used two updates and tiny denominators and is not scientific evidence. Roots 20261223–20261225 remain unopened; M024.c/d remain closed.
