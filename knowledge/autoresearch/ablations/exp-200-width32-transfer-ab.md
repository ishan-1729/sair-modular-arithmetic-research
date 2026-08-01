---
type: Experiment Preregistration
title: EXP-200 Width-32 Transfer A/B
description: Matched random-versus-width-16 initialization test at the Tier-4 modulus width.
status: completed
tags: [autoresearch, Tier-4, width-32, transfer, hard-mining]
timestamp: "2026-07-12T09:30:00+05:30"
---

# EXP-200: width-32 transfer A/B

## Hypothesis

Learned width-16 initialization materially improves width-32 exact transition and autonomous-rollout accuracy under the validated quotient-2 and reduction-boundary hard-mining curriculum.

## Matched contract

Both arms use the same width-32 shared-context TCN, 17–32-bit prime split, seed, 400 optimizer steps, 256-row network batch, four-times offline candidate pool, losses, optimizer, schedule, heldout primes, and evaluation counts. The only changed variable is initialization:

- control: deterministic random initialization;
- challenger: overlap transfer from frozen EXP-111 checkpoint `f9778b636f70d7cca56514ea98f95a22ce2fe81abcc4bd57f417274bc1e4bb94`.

The checked public benchmark is not read or evaluated. The 10,000 transition and 5,000 rollout cases use heldout prime identities.

## Gate

Continue the transferred width-32 family only if it beats random initialization by at least two transition points and five rollout points, while reaching at least 70% transition, 5% rollout, and no more than 35% invalid rollouts. Missing the absolute or paired gate rejects direct fixed-canvas transfer as the immediate Tier-4 path and prioritizes the preregistered width-independent architecture track.

The smoke result cannot promote a candidate; it only authorizes a separately preregistered full-budget run.

## Infrastructure retry note

The first execution was killed before training because the inherited width-16 sampler attempted an exhaustive sieve and composite list through `2^32`. The retry replaces only that infeasible enumeration with a deterministic bounded identity split (128 train, 32 validation, and 32 test primes per represented bit length) and 4,096 deterministically sampled composites. Architecture, A/B variable, seeds, optimizer steps, hard-mining proportions, evaluation counts, and gates remain unchanged. The failed attempt produced no model checkpoint or valid metric.

## Outcome

Passed decisively. At matched 400-step compute, random initialization reached 44.12% transition and 0% rollout, while transfer reached 98.21% transition and 61.68% rollout with 2.12% invalid rollouts. The paired gains were +54.09 transition points and +61.68 rollout points. Advance the transferred arm to a separately preregistered full-budget Tier-4 gate.
