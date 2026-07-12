---
type: Experiment Preregistration
title: EXP-615 Tier-6 Casewise Exactness Loss
description: Loss-only A/B of the existing bit-position objective against a per-example worst-bit exactness surrogate.
status: preregistered
tags: [autoresearch, tier6, exactness-loss, straight-through, causal-ablation]
timestamp: "2026-07-13T04:10:00+05:30"
---

# EXP-615 Tier-6 casewise exactness loss

## Question and hypothesis

First-error analysis finds that 76.9% of failed trajectories initially differ by only one or two bits, but the current loss combines mean BCE with the worst bit position averaged over the batch. That objective can dilute a different rare weak bit in each example. Will an additional mean of each example's maximum bit BCE increase the weakest margin and reduce autonomous failures?

A 2,000-case fresh scale diagnosis found mean BCE `0.000407`, bit-position worst BCE `0.000888`, and mean casewise maximum BCE `0.010266`, with no observed transition errors. Multiplying the casewise term by `0.05` contributes about `0.000513`, comparable to the existing terms without dominating them. This coefficient is fixed before the matched evaluation.

## Matched comparison

Both arms load the same EXP-609 checkpoint and use identical 99,824-parameter architecture, seed, fresh identities, generated examples, 200 optimizer updates, batches, mode schedule, eight-step straight-through state path, optimizer, learning-rate schedule, weight decay, clipping, and evaluation cases. The sole changed variable is casewise-worst-bit weight: `0.0` control versus `0.05` challenger. The existing mean BCE, bit-position worst term (`0.45`), and quotient term remain unchanged.

## Gates

Each endpoint is evaluated on 2,000 fresh transitions and 200 fresh rollouts at widths 64 and 128. Advance only if the challenger reaches at least 90% width-128 rollout, gains at least three points over control, retains at least 95% width-64 rollout, reaches at least 99.8% width-128 transition, and keeps invalid width-128 rollouts at or below 1%.

This smoke cannot promote a candidate. A pass requires higher-fidelity confirmation. If it improves directionally but misses the causal gate, one higher-fidelity confirmation is allowed only when the confidence interval and exact counts make the gate genuinely close; otherwise close the casewise term rather than sweep weights.

## Resource budget

The directly comparable EXP-614 base arm, including full two-width evaluation, took 479.81 wall seconds and 1,862.41 aggregate CPU seconds with four threads. Two arms project to 960 wall seconds; the casewise reduction and setup receive 40 additional seconds, for an expected total of 1,000 wall seconds.

The planning budget is 1,600 wall seconds (60% headroom), and the hard runaway ceiling is 2,400 wall seconds or at most 160 aggregate CPU minutes at four continuously occupied threads. Wall and CPU are distinct. Completed arms are checkpointed and content-keyed. No optimizer work or evaluation cases will be removed to fit a shorter cap.

## Boundaries

Checked and private answers remain unread. Exact arithmetic remains confined to offline labels and evaluation truth. The inference architecture is unchanged, tensor-native, and trained-weight dependent; the new term is training-only. Native PyTorch BCE and reductions are the appropriate implementation.
