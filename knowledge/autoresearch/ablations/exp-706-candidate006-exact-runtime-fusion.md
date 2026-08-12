---
type: Experiment Preregistration
title: EXP706 Candidate006 exact-score runtime fusion
status: completed
description: Preserve Candidate006 predictions while reducing exact official-sandbox inference below the written 300-second limit through inference-only CPU execution changes.
tags: [candidate006, inference, runtime, cpu, parity]
timestamp: "2026-08-12T14:30:00+05:30"
---

# EXP706 Candidate006 exact-score runtime fusion

## Question

Can Candidate006 complete all 1,100 current evaluation cases inside the written 300-second inference limit without retraining or losing accuracy?

## Frozen intervention

Keep the manifest and every trained tensor byte-identical. Change only inference execution:

- cache invariant feature planes and keep recurrent state tensor-native;
- evaluate the same Conv1d kernels as height-one channels-last Conv2d kernels;
- partition each independent case batch into at most four order-preserving CPU shards;
- omit the quotient head, whose output every inference caller discards.

The artifact may not skip cases, tiers, recurrent updates, the Tier-6 swapped route, or any scored work.

## Gates

The runtime result advances only if the repository-pinned official evaluator shape completes Tiers 0 through 7, reports at least Candidate006's prior 64.1% overall score and the same 54% Tier-7 exact-answer accuracy, retains the exact manifest and weight hashes, and measures inference below 300 seconds. A retained private Tier-7 seed must also reproduce Candidate006's exact prediction hash.

The historical 285-second value is not a competition rule and is not a gate here. It was an internal 15-second headroom target. The binding eligibility gate is the written 300 seconds.

## Ruleset scope

This experiment freezes runtime equivalence against challenge commit `82510bba00a1126649bd76dd1a451f14d0b3eb60`; it does not preregister or establish compliance with later upstream rule changes. A post-run audit of upstream commit `99cac6ef5c2f82e53105ec0ddcbb9b8d37bf6fca` found that current rules explicitly prohibit deterministic full-width operand pre-reduction. Candidate006 performs `int(a_enc) % p` and `int(b_enc) % p`, so a successful runtime gate must not be interpreted as current-rule submission approval.
