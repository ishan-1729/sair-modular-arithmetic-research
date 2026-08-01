---
type: Experiment Preregistration
title: EXP-203 Independent Official-Generator Tier-4 Evaluation
description: Frozen-checkpoint five-seed generalization gate before Tier-4 integration.
status: completed
tags: [autoresearch, evaluation, Tier-4, preregistration]
timestamp: "2026-07-12T10:32:00+05:30"
---

# EXP-203: independent official-generator Tier-4 evaluation

Evaluate frozen EXP-202 checkpoint `047cbfb0599b2ea12af9aeed5171cf832f0c4ec179a09596518b668b14644eed` on five named official-generator Tier-4 seeds, 100 cases and five primes per seed, using the pinned challenge commit. Pairwise operand reduction matches the sanctioned adapter used by prior candidates. Checked public Tier-4 answers remain unread.

Promote to integration only if aggregate exact accuracy is at least 90%, at least four of five seeds reach 90%, and no seed falls below 85%. Otherwise retain Candidate 002 and diagnose without checked-public tuning.

## Outcome

Failed: 441/500 (88.2%), one of five seeds at or above 90%, and a worst seed of 87%. Candidate 002 remains incumbent. The result exposes a distribution mismatch: the identity-balanced heldout set weights every bit band equally, while official prime generation concentrates probability near 30–32 bits, where EXP-202 was weakest.
