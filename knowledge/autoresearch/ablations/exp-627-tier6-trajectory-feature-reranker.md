---
type: Experiment Preregistration
title: EXP-627 Tier-6 Trajectory-Feature Reranker
description: Generated-data training and untouched evaluation of a compact legal selector for confidence-routed width-4 candidates.
status: preregistered
tags: [autoresearch, tier6, reranker, conditional-compute, trained-selector]
timestamp: "2026-07-13T10:15:00+05:30"
---

# EXP-627 Tier-6 trajectory-feature reranker

## Hypothesis and inference path

Run the frozen greedy transition model on all width-128 cases, route the 40% lowest-trajectory-margin cases through frozen width-4 search, and use a compact trained MLP to choose one of the four final states. Unrouted cases retain greedy output. The reranker receives no expected answer, exact product, benchmark identity, or result-dependent filter.

Candidate features are frozen before data generation: learned beam rank and score/gap; whether the candidate equals greedy; normalized Hamming and popcount relations among candidate, greedy, top beam, operands, and modulus; bit length; and six aggregate greedy-trajectory confidence statistics. These are generic representations and learned-search diagnostics, not a modular arithmetic circuit. The 5,570-parameter MLP standardizes features, uses two 64-unit SiLU layers, and has a trainable greedy-candidate prior initialized to preserve greedy selection.

## Data partitions and training

Generate 1,200 training cases from 512 training prime identities. Split the 128 held-out prime identities before sampling: the first 64 supply 200 validation cases and the remaining 64 supply 200 untouched selection cases. No checked-public, prior EXP-625/626 expected answer, or private answer enters training or model selection.

For each split, compute greedy confidence for every case and full width-4 candidates only for the frozen 40% minimum-margin route. Cache every completed generation batch and each completed split using checkpoint, runner, feature, prime-list, seed, and configuration hashes. Train listwise cross-entropy only on routed training cases whose correct state exists in the beam; beam-miss cases remain in validation and selection accuracy denominators. Select the checkpoint by complete conditional validation exact count, with 300 epochs and patience 40. The final selection set is opened once after checkpoint choice.

## Controls and gates

Report greedy, likelihood-selected conditional, oracle containment, initialized/randomized-control, and trained selected accuracy on identical selection cases. Advance only if trained selected accuracy is at least 90%, gain over greedy is at least three points, routed invalid rate is at most 1%, gain over the untrained initialized control is at least two points, reranker scores and choices repeat exactly, and projected integrated runtime is at most 260 seconds. Candidate 004's widths 1–64 remain byte-identical and are not re-evaluated at this research gate; retention is mandatory during later integration.

A pass authorizes untouched multi-seed Tier-6 confirmation and weight-dependence ablation. It does not promote a candidate or satisfy the 85% overall goal.

## Dynamically calibrated resource budget

The candidate-shaped profile measured greedy-100 at 45.69 wall / 175.31 aggregate CPU seconds and width-4 beam-40 at 89.60 wall / 340.87 aggregate CPU seconds. Linear workload projection gives approximately 1,625 wall seconds for 1,200-case training generation and 545 seconds for validation plus selection generation. Training, serialization, MLflow, and safety allowance add 130 seconds, for a 2,300-second expected total.

The soft budget is 3,200 wall seconds and the hard runaway ceiling is 4,600 wall seconds, with a conservative four-thread ceiling of 306.67 aggregate CPU minutes. Wall and aggregate CPU are recorded separately. Every completed generation batch and split is recoverably cached; a later interruption or phase overrun cannot invalidate prior batches. Fidelity, data counts, and epochs will not be reduced to fit a shorter cap.

### Continuation budget after the generation ceiling

The first full process reached its declared 4,600-second hard ceiling before a final summary and was stopped without interpreting model evidence. Batch recovery preserved the complete 1,200-case training split (3,946.06 wall / 14,874.51 aggregate CPU seconds), complete 200-case validation split (765.84 wall / 2,892.68 aggregate CPU seconds), and greedy inference for all 200 selection cases (289.54 cumulative wall seconds). The selection beam has zero completed cases, so no selection label or metric has been opened.

The only expensive missing generation is width-4 search for 80 selected cases. The matched validation beam-80 phase measured 439.75 wall / 1,656.61 aggregate CPU seconds. Adding 130 seconds for MLP training, checkpointing, controls, and logging gives a 570-second expected continuation. The continuation soft budget is 850 wall seconds and its hard runaway ceiling is 1,250 wall seconds, with a conservative four-thread upper bound of 83.34 aggregate CPU minutes. The runner will reuse the hashed train/validation caches and selection greedy partial exactly; no completed evidence is recomputed and all original counts, seeds, gates, and fidelity remain frozen.

## Compliance and limitations

Exact bigint arithmetic is confined to generated labels and scoring. Trained parameters materially select the answer among generic learned-search states. Equality and bit-statistic features do not repair or compute a modular product. Generated selection evidence is not checked-public, five-seed independent, official-private, or isolated-sandbox evidence. The official inference package remains unchanged unless all later promotion gates pass.
