---
type: Experiment Preregistration
title: EXP-630 Radix-4 Cellular Transition
description: Progressive viability test of a learned two-bit-per-step cellular modular transition against the frozen radix-2 control.
status: preregistered
tags: [autoresearch, radix4, cellular, tier6, horizon, architecture]
timestamp: "2026-07-13T14:10:00+05:30"
---

# EXP-630 radix-4 cellular transition

## Question and hypothesis

Tier-6 failures persist despite roughly 99.8% one-step accuracy because 128 autonomous radix-2 transitions compound sparse errors. Can a learned radix-4 transition consume two multiplier bits per recurrent step, halve the autonomous horizon, and retain enough exact transition accuracy to improve the scaling surface?

The learned state update is supervised offline as `state -> (4*state + digit*right) mod modulus`, where `digit` is two public multiplier bits. Two digit-bit feature planes replace the single input-bit plane, and the auxiliary quotient head expands from classes 0–2 to 0–6. These are labels and representations, not an inference repair rule: trained parameters produce every output state, and exact arithmetic remains outside inference.

## Architecture, initialization, and curriculum

The challenger is the same width-shared, two-cycle dilated ConvGRU family with seven instead of six input planes and seven quotient classes. Common stem, recurrent, normalization, local-head, and quotient parameters transfer from the frozen EXP-609 radix-2 checkpoint. The old bit-plane weights initialize a normalized radix-4 digit projection (`low/3 + 2*high/3`); quotient classes 0–2 copy and classes 3–6 retain seeded initialization.

The model receives 1,600 uniform transition updates, batch size 128, cycling widths 8, 12, 16, 20, 24, and 32. AdamW, cosine decay, clipping, exact-bit loss, worst-bit term, and quotient supervision follow the successful cellular curriculum family. The frozen radix-2 source is the simple control and runs on identical final multiplication cases; it is not retrained because the causal question is whether the new radix can reach a useful viability surface with half as many recurrent calls.

An initial profile request for too many unique width-8 primes was terminated before training or evidence because that finite prime range could not satisfy the split. The corrected split uses 32/8 identities at width 8 and 256/64 at larger widths.

## Evaluation and gates

The saved challenger is evaluated on 2,000 radix-4 transitions and 200 paired autonomous multiplication cases at widths 16, 32, and 64. The source radix-2 model receives the identical `(left, right, modulus)` cases. Both wall and aggregate CPU are measured separately. Twenty width-64 cases are repeated for exact prediction determinism.

Advance to higher-width radix-4 scaling only if every gate passes:

- at least 90% challenger rollout at trained width 16;
- at least 70% challenger rollout at trained width 32;
- at least 20% zero-shot challenger rollout at width 64;
- at least 98% width-32 and 80% width-64 radix-4 transition accuracy;
- at most 10% invalid width-64 rollouts;
- width-64 radix-4 wall time no more than 65% of the matched radix-2 control;
- exact repeated predictions and validity diagnostics.

This is an architecture viability rung, not candidate promotion. A pass authorizes progressive widths above 32; it does not authorize independent, checked, official, or submission evaluation.

## Dynamic resource budget

The corrected representative profile ran 30 batch-128 updates across the six training widths in 25.251 wall and 95.468 aggregate CPU seconds. Linear projection gives 1,346.71 wall and 5,091.61 aggregate CPU seconds for 1,600 updates. A separate matched width-128 profile measured 38.493 wall seconds for 20 radix-2 rollouts and 19.401 for radix-4, a 0.504 ratio. Scaling by recurrent step-cases and adding transitions gives a conservative 550-wall-second paired evaluation projection plus 120 seconds for setup and artifacts.

Expected total is 2,017 wall seconds and roughly 135 aggregate CPU minutes. The planning budget is 3,000 wall seconds; the hard runaway ceiling is 4,800 wall seconds, equivalent to at most 320 aggregate CPU minutes at four fully occupied threads. Training state, optimizer, scheduler, all sampler RNGs, and history are saved every 50 updates; the checkpoint is saved before evaluation. Full declared evaluation will not be reduced to meet an arbitrary shorter cap.

## Calibration observations and limitations

Transferred radix-4 initialization is not function preserving: before training it scores 0/200 transitions and 0/20 width-128 rollouts. Twenty width-128 batch-64 updates at `5e-5` also remain 0% exact and serve only as throughput context. In the corrected small-width `1e-3` profile, width-32 loss falls from 2.81 at its first exposure to 1.12 after five exposures, but exactness remains 0%; the full preregistered curriculum is therefore necessary to test viability.

No checked, consumed independent, or private answer is used. The rung changes representation, quotient-head size, training data, and autonomous step count together as an architecture test. Later causal ablations are required before attributing any gain specifically to horizon length.

