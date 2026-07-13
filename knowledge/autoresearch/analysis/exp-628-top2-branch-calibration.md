---
type: Calibration Analysis
title: EXP-628 Top-Two Bit Branch Calibration
description: Paired diagnostic of one-bit versus top-two individual and combined beam branching on consumed EXP-627 selection cases.
status: completed
tags: [autoresearch, tier6, beam-search, calibration, negative-result]
timestamp: "2026-07-13T11:35:00+05:30"
---

# EXP-628 top-two bit branch calibration

## Question

Would adding the second-least-confident bit and the joint two-bit flip to each beam parent materially improve width-4 oracle coverage without another transition-model call?

This was a pre-preregistration calibration, not a formal selection experiment. It reused the already consumed 200-case EXP-627 selection cache (`selection-80c1c3dc5da0dfb9f0f5c59773d7feb2371bdf954b6cc1deb798c0fd1bb29f24.pt`) and its frozen 40% minimum-margin route. No checked, independent, or private answer was read.

## Observed paired result

The common greedy result was 162/200 (81.0%). Among the same 80 routed cases:

- one-bit branching contained the correct answer for 56/80 routed cases and produced 175/200 (87.5%) conditional oracle accuracy;
- top-two individual-plus-joint branching contained the correct answer for 58/80 routed cases and produced 177/200 (88.5%) conditional oracle accuracy;
- the paired containment difference was two top-two-only cases and zero one-bit-only cases.

The +1.0 overall-point gain is below the planned four-point minimum needed to justify fresh confirmation and reranker training. The untouched 90% selector gate would also remain unreachable on these cases even with an oracle selector.

## Runtime observation and limitation

Matched beam generation for the 80 routed cases took 418.86 wall seconds for the cached one-bit control and 529.02 wall seconds for top-two branching, about 26.3% more. The top-two run recorded 1,976.73 aggregate CPU seconds. Absolute timing was contaminated by concurrent system work, including an Ubuntu unattended upgrade, so it must not be compared with the clean candidate runtime projection. The within-session matched slowdown remains useful as calibration evidence.

The top-two diagnostic was executed inline and did not retain a raw top-two state artifact. It therefore supports only the decision not to spend fresh evidence and compute on this branch rule; it is not registered as a durable model experiment and makes no candidate claim.

## Decision

Do not preregister a fresh EXP-628 confirmation and do not retry the reranker. Preserve the generic, unit-tested `branch_bits=2` decoder option for future controlled analysis, while closing fixed top-two branching as the current Tier-6 path. Return to the base transition model and test a training-only, balanced hard-trajectory replay intervention rather than extending search.
