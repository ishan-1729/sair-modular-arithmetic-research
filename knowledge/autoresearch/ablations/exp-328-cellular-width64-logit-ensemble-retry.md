---
type: Experiment Preregistration
title: EXP-328 Cellular Width-64 Logit-Ensemble Retry
description: Full-fidelity paired two-checkpoint logit-ensemble comparison with a throughput-calibrated dynamic budget.
status: preregistered
tags: [autoresearch, cellular, width-64, ensemble, retry, Tier-5]
timestamp: "2026-07-12T15:15:00+05:30"
---

# EXP-328: cellular width-64 logit-ensemble retry

Repeat EXP-327 with the same frozen EXP-323 and EXP-324 checkpoints, the same seed-20260715 fresh identity split, the same 50/50 per-step output and quotient-logit average, and the original full fidelity of 1,000 transitions plus 500 autonomous rollouts per arm. No metric was read from the stopped EXP-327 run; its killed MLflow record is retained as procedural provenance rather than model evidence.

## Dynamic resource budget

EXP-326 measured 593.238 wall seconds for five single-model-equivalent arms at the same 1,000/500 fidelity and four PyTorch threads, or 118.648 wall seconds per model-equivalent arm. EXP-328 requires four equivalents: one for each endpoint and two for the two-model ensemble. The throughput projection is therefore 474.590 wall seconds. Because the interrupted EXP-327 attempt demonstrated substantial unmodeled host contention or evaluator overhead, allocate a 1,200-wall-second soft budget (2.53x the projection) and an 1,800-wall-second hard runaway ceiling (3.79x). These are wall-clock limits, not CPU time. With four configured threads, the hard ceiling implies a conservative aggregate CPU-time upper bound of 120 CPU minutes; actual CPU time must be measured and reported separately. The competition's 300-second constraint does not apply to this offline matched evaluation.

Write each completed arm's deterministic metrics immediately to a cache keyed by checkpoint(s), alpha, evaluator hash, dataset/split hash, seed, and evaluation counts. Reuse a cached arm only when that complete key matches. This prevents a later interruption from discarding valid completed evidence without mixing unmatched inputs.

Require the ensemble to reach at least 99.8% transition accuracy, 93% rollout accuracy, improve at least one point over the better paired endpoint, and keep invalid rollouts at or below 0.5%. Passing authorizes five untouched Tier-5 official-generator seeds; failure closes checkpoint ensembling. Do not read checked, consumed independent, or private answers. Give the full matched comparison enough time to finish; stop only for the documented hard runaway ceiling or a genuine execution failure.
