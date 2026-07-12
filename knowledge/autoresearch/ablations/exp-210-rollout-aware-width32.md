---
type: Experiment Preregistration
title: EXP-210 Rollout-Aware Width-32 A/B
description: Matched one-step continuation versus scheduled learned-state chunks on official-like heldout identities.
status: completed
tags: [autoresearch, Tier-4, rollout, scheduled-sampling, ablation]
timestamp: "2026-07-12T11:00:00+05:30"
---

# EXP-210: rollout-aware width-32 A/B

Both arms initialize from the frozen EXP-204 checkpoint and use the same official-numeric prime split, optimizer, learning-rate schedule, 400 optimizer steps, model, losses, seed, and 102,400 transition-equivalent network examples.

- Control: 400 ordinary 256-row hard-mined one-step batches.
- Challenger: 200 identical hard-mined one-step batches plus 200 four-step, 64-row trajectory chunks. Within each chunk, the model's detached hard prediction is fed into the next transition with probability scheduled from 25% to 75%; the target remains the exact teacher trajectory, explicitly training recovery toward the correct rollout.

This changes only state-exposure/training structure. Evaluate both arms on the same fresh official-like heldout identities: 50,000 transitions and 20,000 autonomous rollouts. Do not reuse EXP-203 seeds or read checked Tier-4 answers.

Pass only if the challenger improves rollout by at least two points over the matched control, reaches at least 92% rollout and 99.6% transition, regresses transition by no more than 0.1 point versus control, and keeps invalid rollouts at or below 0.5%. Otherwise reject this scheduled learned-state formulation and proceed to the materially different architecture track.

## Outcome

Rejected. The matched one-step control reached 99.794% transition and 90.72% rollout. Scheduled learned-state training reached 99.828% transition and 91.425% rollout, a stable but only +0.705-point rollout gain. It missed both the +2-point paired gate and 92% absolute gate. Proceed to the materially different width-independent architecture track.
