---
type: Experiment Preregistration
title: EXP-625 Tier-6 Confidence-Routed Width-4 Search
description: Fresh diagnostic separating conditional-search coverage ceilings from target-free confidence routing quality.
status: preregistered
tags: [autoresearch, tier6, conditional-compute, confidence, beam-search]
timestamp: "2026-07-13T09:10:00+05:30"
---

# EXP-625 Tier-6 confidence-routed width-4 search

## Hypothesis and inference concept

Run the frozen greedy model on every Tier-6 case, rank trajectories using confidence computed only from their learned logits, and rerun the 40% most uncertain cases through the unchanged width-4 learned search. Easy cases retain greedy output. No target, expected answer, oracle membership, modulus filter, or benchmark identity enters routing.

This diagnostic does not implement a legal final selector inside the routed beam. Instead it asks whether the search surface and target-free allocation are strong enough to justify training a compact router/reranker on generated data.

## Frozen confidence and ceilings

The primary target-free score is `mean_bit_nll`: the negative mean of `logsigmoid(abs(logit))` across all bits and all 128 trajectory steps, with larger values routed first. Secondary diagnostics are negative minimum trajectory margin, negative mean per-step minimum margin, negative last-32-step minimum margin, and mean quotient entropy. Secondary results cannot pass a failed primary gate.

At the fixed 80/200 route count, report:

1. unrestricted union coverage when each case may use greedy correctness or any final width-4 state;
2. oracle-best 40% allocation coverage, an evaluation-only upper bound for compute routing;
3. primary and secondary target-free ranked allocation coverage.

Oracle membership never affects inference or the confidence ranking. If the oracle allocation misses, no router can work at this compute fraction. If only confidence misses, this formulation does not authorize learning a router from the diagnostic set.

## Runtime and resource budget

A label-free candidate-shaped profile on seed `20260928` measured 45.69 wall / 175.31 aggregate CPU seconds for greedy-100 and 89.60 wall / 340.87 aggregate CPU seconds for width-4 on 40 arbitrary cases. The Tier-6 conditional workload is therefore 135.29 wall seconds and the integrated projection is 187.19 seconds.

The fresh diagnostic uses 200 paired greedy/full-width-4 cases so every possible route can be evaluated, followed by two exact 20-case determinism repeats. Expected total is 980 wall seconds, soft budget 1,300 wall seconds, and hard runaway ceiling 1,850 wall seconds. The conservative four-thread hard-ceiling upper bound is 123.34 aggregate CPU minutes. Wall and aggregate CPU are recorded separately, fidelity is fixed, and raw predictions/confidence are cached.

## Gates and stopping rule

Advance to separately trained routing/reranking only if unrestricted union, oracle-allocated, and primary confidence-allocated coverage each reach at least 92%; final beam invalid rate is at most 1%; all confidence, predictions, and beam states are deterministic; and the measured Tier-6 projection is at most 240 wall seconds. Otherwise reject without router training. Passing is research authorization only, not candidate promotion.

## Compliance and limitations

Confidence uses learned logits only. Exact bigint arithmetic is confined to generated truth and post-inference scoring. Oracle allocation and beam containment are explicitly evaluation-only upper bounds. The generated diagnostic is neither checked-public nor official-private evidence, and projected runtime does not replace the pinned isolated sandbox.
