---
type: Experiment Preregistration
title: EXP-623 Tier-6 Staged Beam Runtime
description: Fresh confirmation of a generic late-widening beam schedule that may preserve coverage with substantial CPU headroom.
status: preregistered
tags: [autoresearch, tier6, beam-search, runtime, staged-search]
timestamp: "2026-07-13T08:10:00+05:30"
---

# EXP-623 Tier-6 staged beam runtime

## Hypothesis and changed variable

The frozen decoder retains the top two learned-likelihood states for Horner positions 0 through 95, then retains four states for positions 96 through 127. Relative to EXP-622, only the generic retained-beam schedule changes. The checkpoint, least-confident-bit branch operator, learned score, generated split construction, and target-free inference remain fixed.

The hypothesis is that late widening retains at least 92% oracle final-state containment on 200 untouched width-128 cases while the measured 100-case Tier-6 workload remains at or below 210 wall seconds. This would restore enough headroom to consider a compact trained reranker; it does not make the likelihood-selected output a candidate.

## Calibration and frozen choice

A supported PyTorch trace/freeze screen produced bit-for-bit identical logits, scores, predictions, and beam states but improved 20-case warm wall time only 0.8% (42.15 to 41.82 seconds), so it was closed without a candidate-shaped run.

On one 20-case schedule-calibration seed, widening at positions 32, 64, and 96 each retained the correct final state in 18 cases, versus 17 for full width 4. Position 96 was frozen because it was the latest and fastest tied schedule, taking 35.14 wall / 135.22 aggregate CPU seconds. These small-sample accuracy counts are selection evidence only.

A label-free candidate-shaped profile on a distinct seed then ran all 100 cases in 135.22 wall / 512.97 aggregate CPU seconds, with four unique final states for every case. The integrated projection is `36.9 + 135.2224 + 15 = 187.1224` wall seconds.

## Confirmation, gates, and resource budget

Seed `20260819` supplies 200 fresh matched greedy/staged-beam cases. The run records all inputs, expected values, predictions, final states, and scores, followed by two exact 20-case determinism passes. The unchanged 5,000-case transition result is reused by checkpoint/evaluator hash from EXP-621.

Advance only if oracle containment is at least 92%, selected final invalid-residue rate is at most 1%, all repeated predictions/states/counts are identical, and projected integrated runtime is at most 262 seconds, equivalent to approximately 210 seconds for the staged Tier-6 path after the fixed incumbent and integration allowances. Oracle containment remains evaluation-only and can never select output.

The expected total is 500 wall seconds. The soft budget is 700 wall seconds and the hard runaway ceiling is 1,100 wall seconds, with a conservative four-thread upper bound of 73.34 aggregate CPU minutes. Wall and aggregate CPU are measured separately. Evaluation fidelity will not be reduced to fit a shorter cap, and completed predictions are cached.

## Compliance and limitations

The schedule depends only on public position and width, not modulus values, expected answers, benchmark identity, or exact arithmetic. Exact bigint arithmetic is confined to generated truth and post-inference scoring. Generated confirmation and projected runtime are not checked-public, private, or pinned official-sandbox evidence. A passing run authorizes reranker research only, not candidate promotion.
