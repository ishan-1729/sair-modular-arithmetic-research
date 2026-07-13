---
type: Experiment Preregistration
title: EXP-624 Tier-6 Width-3 Beam Coverage
description: Final fixed-width learned-beam feasibility test between the insufficient width-2 coverage and expensive width-4 search.
status: preregistered
tags: [autoresearch, tier6, beam-search, coverage, runtime]
timestamp: "2026-07-13T08:40:00+05:30"
---

# EXP-624 Tier-6 width-3 beam coverage

## Hypothesis and changed variable

The frozen EXP-609 checkpoint uses the unchanged least-confident-bit branch operator and cumulative learned-likelihood score, retaining the top three unique states at every Horner position. Width 3 is the only fixed-width compute/coverage point not tested between EXP-621's width 2 and EXP-622's width 4.

The hypothesis is at least 92% oracle final-state containment on 200 untouched width-128 cases, with a candidate-shaped 100-case Tier-6 workload at or below 210 wall seconds. Oracle membership is computed only after inference and can never select the output.

## Runtime calibration and full evaluation

A label-free profile on seed `20260927` completed the exact 100-case workload in 167.27 wall / 639.14 aggregate CPU seconds; every final beam contained three unique learned states. With the frozen incumbent and integration allowances, the projection is `36.9 + 167.2667 + 15 = 219.1667` wall seconds.

Fresh seed `20260820` supplies 200 matched greedy/width-3 cases. All inputs, expected values, predictions, final states, and scores are retained. Two 20-case repeats must reproduce predictions, full beam states, and paired counts. The checkpoint/evaluator-identical 5,000-case transition result is reused from EXP-621.

Advance to compact reranker training only if containment is at least 92%, invalid selected residues are at most 1%, determinism is exact, and projected integrated runtime is at most 262 seconds. Otherwise close learned beam search rather than tune width or likelihood on observed selection evidence.

## Dynamically calibrated budget

The expected total is 560 wall seconds: approximately 450 for the matched main evaluation and 110 for determinism, setup, and logging. The soft budget is 780 wall seconds and the hard runaway ceiling is 1,150 wall seconds, with a conservative four-thread upper bound of 76.67 aggregate CPU minutes. Wall and aggregate CPU are measured separately, fidelity is fixed, and completed raw predictions are cached.

## Compliance and limitations

Search uses only trained logits, generic bit branching, and learned scores. It contains no modulus filter, target feedback, exact arithmetic circuit, or benchmark lookup. Exact bigint arithmetic is confined to generated truth and scoring. Generated evidence and projected runtime are not checked-public, official-private, or pinned sandbox results. A pass authorizes reranker research only.
