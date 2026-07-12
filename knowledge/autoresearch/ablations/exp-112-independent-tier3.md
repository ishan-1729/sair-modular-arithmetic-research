---
type: Experiment Preregistration
title: EXP-112 Independent Official-Generator Tier 3 Evaluation
description: Frozen-checkpoint, five-seed generalization gate before submission integration.
status: completed
tags: [autoresearch, evaluation, tier-3, preregistration]
timestamp: "2026-07-12T03:00:00+05:30"
---

# EXP-112: independent official-generator Tier 3 evaluation

## Frozen inputs

- Checkpoint: EXP-111 width-16 model, SHA-256 `f9778b636f70d7cca56514ea98f95a22ce2fe81abcc4bd57f417274bc1e4bb94`.
- Generator: the pinned upstream `_generate_tier_cases` implementation at challenge commit `82510bba00a1126649bd76dd1a451f14d0b3eb60`.
- Scope: Tier 3 only, because this challenger is intended to extend the incumbent past Tier 2.
- Five named seeds, 100 cases and five generated primes per seed, with the official four edge cases.
- Operands are reduced modulo `p` before learned inference, matching the sanctioned pairwise-reduction adapter used by the incumbent.

The checked public Tier 3 answers remain unread and are not an evaluation surface.

## Promotion gate

Promote to submission integration only if all of the following hold:

1. aggregate exact accuracy is at least 90%;
2. at least four of five seeds individually reach 90%;
3. no seed falls below 85%; and
4. the checkpoint hash and generator commit match the frozen values.

Otherwise retain the incumbent and diagnose without consulting checked public Tier 3 answers.

## Outcome

Passed 499/500 (99.8%) across the five frozen seeds. All seeds exceeded 90%; the worst seed scored 99%. This authorized submission integration.
