---
type: Ablation Study
title: EXP-672 LT2 gated linear/sparse hybrid mixer
description: Source-authenticated bounded ST-M025 transfer parked after complete calibration failed the guarded candidate-runtime gate.
status: parked
tags: [autoresearch, exp-672, lt2, linear-attention, sparse-attention, gating]
timestamp: "2026-07-29T07:30:00+05:30"
---

# EXP-672 - LT2 gated linear/sparse hybrid mixer

EXP672 authenticated the local LT2 PDF and official `chili-lab/LT2` code/license, then froze and
implemented a bounded CPU transfer around the immutable Candidate005 proposal. The full arm uses one
genuinely shared four-loop native feature-map linear plus fixed-window sparse mixer, the official-shaped
zero-weight post-SDPA sigmoid gate, zero per-channel loop gates, and a zero per-channel outer output
residual gate. Gate-only, fixed-gate mixer-only, parameter/compute-matched shared-convolution, frozen
proposal, trained-gate reset, and randomized-mixer routes are present. The outer output gate makes every
gated arm exactly proposal-identical at initialization.

Focused verification passed 12 tests in 16.39 seconds, covering source/checkpoint/fixture identity,
PDF and official-code pins, exact proposal identity, linear prefix state, sparse window/G1 gate parity,
shared-loop and gate equations, parameter/compute disclosure, bounded loop-depth curves, matched legal
splits, learning/reload, separate reset/randomized interventions, determinism, and prohibited inference.

T068 normalized exactly those two inherited EXP671 records without changing their decision, roots,
resources, controls, or scientific meaning. The base validator passed. The first full oracle exposed
one owned PyTorch inter-op-thread idempotence defect after 611 passing tests; the sole bounded correction
then produced a green validator and all 612 tests passed in 527.29 seconds. T067's source and focused
evidence was reused because its implementation/test bytes were otherwise unchanged.

Root `20261238` completed all four trained arms, all seven routes, validation/test evaluation, depth and
determinism diagnostics, checkpoint reloads, and resource measurement in 239.65 wall seconds. It used
929.17 process-tree CPU seconds, 995.84 cgroup CPU seconds, and 660,582,400 bytes peak RSS. Complete-route
formal work projected to 16,386.74 wall seconds, with 32,773.48-second ETA, 40,967-second soft guard, and
65,547-second hard guard under the calibrated 2.0 headroom.

The nominal candidate projection was 183.084 seconds, but the binding guarded projection was 366.169
seconds versus the frozen 285-second ceiling. The raw calibration payload's `admitted=true` flag checks
nominal runtime only; T068's guarded-runtime conjunct therefore controls and rejects admission. Root
`20261239`, complete W128, conditional W256, and confirmations `20261240/41` remained unopened. This is
a runtime-feasibility park, not a scientific rejection of LT2, ST-M025, or the bounded adaptation. No
checked, independent, official/private, candidate, submission, promotion, or completion surface opened.
