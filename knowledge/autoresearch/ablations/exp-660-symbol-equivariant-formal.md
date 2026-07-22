---
type: experiment-preregistration
title: EXP-660 Fresh Formal Symbol-Equivariant Recurrence
description: Fresh ordinary-initialized matched formal falsifier for bounded-faithful symbol-axis recurrence.
status: preregistered
timestamp: "2026-07-22T10:00:00+05:30"
tags: [symbol-equivariance, recurrent-reasoning, matched-controls, formal-falsifier]
---

# EXP-660 fresh formal symbol-equivariant recurrence

## Decision

Determine whether the already bounded-faithful symbol-axis recurrent core—not a
binary arithmetic task symmetry—can achieve trained-width viability and exact
width extrapolation better than both matched controls. EXP-660 is a fresh
formal successor to EXP-659, which was parked before science. It uses unused
formal seed `20261177`; no EXP-659 calibration checkpoint may be loaded.

## Frozen comparison

All three arms start from ordinary declared initialization and receive the same
generated curriculum, batches, optimizer, schedule and evaluation identities:

- `position_only`: recurrent position-attention control without a symbol axis;
- `symbol_equivariant`: tied position-then-symbol attention core;
- `broken_tying`: the same symbol-axis core with symbol-indexed affine scale and
  bias that causally break core equivariance.

Each arm trains for exactly `800@W8 + 1200@W12 + 2000@W16 = 4,000`
updates. Dense validation is recorded every 200 steps and at every stage edge.
Initial, stage and final checkpoints are preserved and hashed.

## Resource admission

Before formal training, run the full representative calibration route on the
committed source identity. Measure monotonic wall time, process user-plus-system
CPU, cgroup CPU and peak RSS separately. Project the exact formal workload and
derive soft/hard wall, process-CPU and cgroup-CPU budgets using `1.25/1.75`
headroom and the hard RSS guard using `1.25`. The previous EXP-659 measurement
is planning evidence only. No short arbitrary cap, fidelity reduction or
denominator shrink is permitted.

## Formal evaluation and gates

Evaluate every arm on 300 W16 and 200 each W24/W32 exact autonomous rollouts,
plus 300 exact transition cases at each width. Also run identity and swap core
permutations, broken-tying contrast, deterministic replay, randomized-weight
dependence, invalid-residue accounting, static leakage checks and a four-case
W256 runtime/RSS probe.

Advance only if every gate is true:

1. symbol-equivariant W16 rollout exact accuracy is at least `0.80`;
2. symbol-equivariant W32 rollout exact accuracy is at least `0.05`;
3. its W32 exact accuracy exceeds the better matched control by at least `0.05`;
4. faithful swap residual is at most `1e-6` and broken-tying residual at least
   `1e-4`;
5. invalid rate is at most `0.02`, replay is deterministic, randomized-weight
   exact drop is at least `0.20`, static leakage is clean, controls remain within
   `0.75x–1.25x` parameters, and every dynamic resource gate passes.

No loss-based training stop is allowed. Evaluation may stop only when a frozen
attainable-count rule makes a required gate mathematically impossible; any such
stop is a rejection, never a reduced denominator claim.

## Boundaries

Generated arithmetic is restricted to offline labels and evaluation. The model
contains no exact modular solver, answer lookup or checked/private data path.
The result is a single-seed architecture decision only: even a pass is not a
candidate, checked or independent benchmark result, official/private evidence,
Tier 7 promotion, 85% completion, or Goal completion.
