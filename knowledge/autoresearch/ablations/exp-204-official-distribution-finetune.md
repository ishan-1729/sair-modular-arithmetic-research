---
type: Experiment Preregistration
title: EXP-204 Official-Distribution Fine-Tune
description: Paired pre/post correction of the high-bit Tier-4 prime-distribution mismatch.
status: completed
tags: [autoresearch, Tier-4, data-distribution, fine-tuning, hard-mining]
timestamp: "2026-07-12T10:40:00+05:30"
---

# EXP-204: official-distribution fine-tune

EXP-203 scored 88.2% on independent Tier 4 despite 94.49% identity-balanced heldout rollout. The official generator samples a random integer over `[2^17, 2^32)` before taking the next prime, so roughly half its mass lies in the 32-bit band and another quarter in the 31-bit band. EXP-202 instead sampled equal prime counts per bit length, underweighting the weakest high bands.

Freeze EXP-202 as the paired pre-fine-tune control. Build new, identity-disjoint train/validation/test prime sets by uniformly sampling the official numeric interval, then fine-tune the exact checkpoint for 800 steps with a lower learning rate while preserving architecture, hard-mining mixture, losses, batch size, and seed. Evaluate the frozen pre-fine-tune model and the post-fine-tune model on the same 50,000 transitions and 20,000 rollouts from the new heldout identities. Do not reuse EXP-203 seeds and do not read checked Tier-4 answers.

Pass only if post-fine-tune transition is at least 99.8%, rollout at least 94%, rollout gain at least three points, and invalid rollouts no more than 0.5%. Passing authorizes a new five-seed independent Tier-4 gate. Failing rejects simple distribution correction and moves to a rollout-aware or different architecture.

## Outcome

Rejected. On paired official-like heldout identities, rollout improved from 84.60% to 89.97% (+5.37 points), but missed the 94% absolute gate. Transition improved from 99.55% to 99.666% but missed the 99.8% gate. Invalid rollout was 0.3%. The causal distribution hypothesis was partly supported, but simple fine-tuning was insufficient; move to rollout-aware training or a different architecture.
