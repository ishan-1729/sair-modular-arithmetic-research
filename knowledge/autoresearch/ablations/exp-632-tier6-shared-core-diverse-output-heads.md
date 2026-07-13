---
type: Experiment Preregistration
title: EXP-632 Tier-6 Shared-Core Diverse Output Heads
description: Fresh matched screen of bootstrap-trained complementary residual heads over the frozen recurrent core.
status: preregistered
tags: [autoresearch, tier6, ensemble, output-heads, bootstrap, exact-initialization]
timestamp: "2026-07-13T15:15:00+05:30"
---

# EXP-632 Tier-6 shared-core diverse output heads

## Hypothesis and model

Sparse Tier-6 bit errors may be head-specific even when the recurrent representation is shared. Three learned residual bit heads receive complementary two-thirds masks of the frozen 92-channel hidden state and majority-vote each output bit. Every channel is visible to exactly two heads. Zero residual initialization makes all head logits and predictions exactly equal to the frozen EXP-609 source before training. The recurrent core, quotient head, and original local head remain frozen; only 279 residual parameters train.

The hypothesis is that bootstrap diversity corrects enough sparse errors to reach at least 90% width-128 rollout and improve at least three points over the identical shared single-head control without meaningful inference overhead.

## Training and controls

Training reuses the immutable, content-hashed EXP-629 training-only replay set: 153 exact first-error records paired with 153 ordinary transitions from the same trajectories. Each of 300 batch-64 updates contains 32 hard and 32 control records. Each head receives an independently bootstrapped row sample; AdamW, cosine decay, and clipping are fixed. No validation, selection, checked, independent, or private answer is used for training.

The primary control is the exact initialized/frozen source. The trained challenger changes only residual-head parameters. A randomized-head control resamples residual parameters at the trained tensor-wise mean and standard deviation, preserving the frozen core and architecture. Exact initialization, trained head disagreement, randomized dependence, and repeated deterministic predictions are recorded.

## Fresh evaluation and gates

New prime identities derived only after this preregistration supply 3,000 transitions and 300 paired rollouts at widths 64 and 128. The initialized control and trained majority see identical cases. The randomized-head control receives 100 width-128 rollout cases as a causal diagnostic. Advance only if all gates pass:

- width-128 challenger rollout at least 90% and at least three points above control;
- width-64 challenger rollout at least 95%;
- width-128 challenger transition at least 99.8%;
- width-128 invalid rollout rate at most 1%;
- at least 1% of width-128 trajectories exhibit discrete head disagreement;
- trained accuracy at least two points above randomized heads on their matched 100 cases;
- deterministic repeated predictions;
- projected integrated candidate inference at most 260 seconds.

A pass is a research gate only and requires higher-fidelity confirmation before independent, checked, official, or submission evaluation.

## Dynamic resource budget

Thirty head-only updates take 31.023 wall and 117.106 aggregate CPU seconds, projecting 311 wall / 1,172 CPU seconds for 300 updates. A corrected shared-core forward shows no measurable head overhead; conservatively, full paired transitions and rollouts plus the randomized diagnostic are allocated 900 wall seconds and setup/artifacts 120. Expected total is 1,331 wall seconds and about 85 aggregate CPU minutes.

The planning budget is 2,100 wall seconds and the hard runaway ceiling 3,400 wall seconds, with an aggregate four-thread upper bound of 226.67 CPU minutes. Model, optimizer, scheduler, bootstrap RNG, history, and sampler state are saved every 25 updates; trained weights are saved before evaluation. Evaluation fidelity will not be shortened merely to meet a smaller wall cap.

## Boundaries

Majority voting over learned logits is allowed trained inference, not exact arithmetic or output repair. Exact bigint operations remain offline for labels and evaluation truth. The feasibility calibration on preserved training errors corrected 11/153 by majority and demonstrated 43 disagreements; it is not selection evidence.

