---
type: Experiment Preregistration
title: EXP-612 Tier-6 Residual Adapter Activation
description: Matched optimizer parameter-group test of low-rate versus active-rate training for an exactly initialized recurrent adapter.
status: preregistered
tags: [autoresearch, tier6, widening, optimization, straight-through]
timestamp: "2026-07-13T02:30:00+05:30"
---

# EXP-612 Tier-6 residual adapter activation

## Question and hypothesis

EXP-611's zero-output adapter remained dormant at the trained base's `5e-6` continuation rate: it changed no predictions and emitted only `3.37e-4` RMS logits. Does a normal fresh-cell learning rate activate the new branch and turn residual capacity into exact rollout gains?

The challenger uses `0.001` for adapter parameters, within the `0.001` to `0.002` scale used by successful fresh cellular curricula in this repository, while retaining `5e-6` for the trained base. A shared cosine multiplier decays those rates to 10% of their starts. The hypothesis requires both behavioral activation and at least a three-point width-128 rollout gain.

## Matched comparison

Both arms are the same 116,712-parameter 92+36 residual architecture, initialize exactly from the same EXP-609 source, and receive identical fresh identities, examples, seed, optimizer family, weight decay, 200 updates, mode schedule, batches, eight-step straight-through loss, and evaluation cases. The only changed variable is the adapter learning-rate schedule: inherited `5e-6` control versus `0.001` challenger. Base parameters use `5e-6` in both arms.

Before training, 1,000 cases at each width must reproduce the source's local and quotient logits bit-for-bit or the run aborts.

## Gates

Each arm is evaluated on 2,000 fresh transitions and 200 fresh rollouts at widths 64 and 128. An additional 1,000 fresh transitions measure adapter output relative to the retained base. Advance only if the challenger:

- reaches at least 0.90 width-128 exact rollout and gains at least 0.03 over control;
- retains at least 0.95 width-64 exact rollout;
- reaches at least 0.998 width-128 exact transition with invalid-rollout rate at most 0.01;
- emits at least `0.01` RMS width-128 adapter logits and changes at least one bit or quotient prediction across the activation diagnostic.

If the activation gates pass but accuracy gates fail, close this residual-widening formulation rather than extending it. This remains selection evidence and cannot promote a candidate.

## Resource budget

The directly matched EXP-611 run took 1,071.11 wall seconds and 4,126.14 aggregate CPU seconds with four threads. Parameter-group scheduling does not add meaningful network work; the extra activation diagnostic increases the projection to 1,150 wall seconds. The planning budget is 1,700 wall seconds (48% headroom) and the hard runaway ceiling is 2,600 wall seconds, or at most 173.34 aggregate CPU minutes at four continuously occupied threads. Completed arms are checkpointed and cache-keyed. Evaluation case counts and optimizer work will not be reduced to meet a short arbitrary cap.

## Boundaries

Checked and private answers remain unread. Exact arithmetic is offline-only for generation and evaluation. Both inference paths are tensor-native and trained-parameter dependent. Native PyTorch optimizer parameter groups are the appropriate library facility; no graph operator is implicated.
