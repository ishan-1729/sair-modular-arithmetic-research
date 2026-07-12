---
type: Experiment Preregistration
title: EXP-611 Tier-6 Function-Preserving Residual Widening
description: Matched capacity test of a zero-output recurrent adapter initialized to preserve a trained 92-channel cellular transition exactly.
status: preregistered
tags: [autoresearch, tier6, capacity, widening, straight-through]
timestamp: "2026-07-13T02:20:00+05:30"
---

# EXP-611 Tier-6 function-preserving residual widening

## Question and hypothesis

Can added recurrent capacity improve the diffuse one- and two-bit Tier-6 error tail without destroying the learned transition algorithm? The challenger retains the complete trained 92-channel cell and adds a parallel 36-channel recurrent adapter. Its local and quotient output heads are zero-initialized, so the initial function must be exactly identical to the source. The hypothesis is that the challenger will improve fresh width-128 exact rollout by at least 0.03 over a matched 92-channel continuation and reach at least 0.90.

This is residual widening, not literal Net2Wider channel replication. Literal 92-to-128 replication would change GroupNorm statistics and would therefore fail the exact-function requirement.

## Matched comparison

- Source: frozen EXP-609 straight-through continuation checkpoint with SHA-256 `cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd`.
- Control: the 99,824-parameter 92-channel source architecture.
- Challenger: the same 99,824-parameter base plus a 16,888-parameter, 36-channel recurrent adapter, for 116,712 parameters total.
- Both arms use the same fresh prime identities, generated examples, seed, optimizer, 200 updates, learning-rate schedule, mode schedule, batch sizes, eight-step straight-through recovery loss, and evaluation cases.
- Parameter count and measured wall/CPU work are intentionally unmatched because added capacity is the isolated intervention; both are disclosed.

Before training, 1,000 representative feature batches at each of widths 64 and 128 must have bit-for-bit equal transition and quotient logits between the source and challenger. Failure aborts the run before either arm trains.

## Selection evidence and gates

Evaluation uses 2,000 fresh transitions and 200 fresh autonomous rollouts at each of widths 64 and 128 per arm. The challenger advances only if all conditions hold:

- width-128 exact rollout is at least 0.90;
- width-128 exact rollout gain over the control is at least 0.03;
- width-64 exact rollout is at least 0.95;
- width-128 exact transition is at least 0.998;
- width-128 invalid-rollout rate is at most 0.01.

This smoke can advance the capacity hypothesis but cannot promote a submission candidate. A passing result requires higher-fidelity confirmation, independent seeds, full checked evaluation, and all release gates.

## Resource budget

A local CPU profile on this machine used 20 representative eight-step straight-through updates per arm with four PyTorch threads. The control took 27.74 wall seconds and 108.26 aggregate CPU seconds; the challenger took 41.08 wall seconds and 160.38 aggregate CPU seconds. Extrapolating those rates gives 688 wall seconds for 200 updates of both arms. Prior evaluation experience supports 300 additional wall seconds for setup and both-arm evaluation, for a 988-second expected total.

The planning budget is 1,500 wall seconds, about 52% above the projection. The hard runaway ceiling is 2,400 wall seconds, equivalent to at most 160 aggregate CPU minutes at four continuously occupied threads. These are explicitly different quantities. Completed arm checkpoints and cache-keyed results are retained and reused. The run is not to be truncated or evaluated on fewer cases merely to fit the planning budget; the hard ceiling exists only for genuine runaway protection.

## Boundaries

The checked benchmark and private answers are not read for selection. Python big-integer arithmetic is restricted to offline generation and evaluation truth. Submission inference remains tensor-native and trained-parameter dependent. Native PyTorch is sufficient for this parallel recurrent adapter; graph libraries are irrelevant to the tested intervention.
