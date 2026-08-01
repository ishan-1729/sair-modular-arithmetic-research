---
type: Experiment Preregistration
title: EXP-656 tensor-native cached rollout route
description: Input-only matched performance A/B of Candidate005's legacy bigint representation and an exactly equivalent cached tensor route.
status: completed-rejected
tags: [autoresearch, tier7, inference-route, data-layout, performance]
timestamp: "2026-07-21T16:00:00+00:00"
last_updated: "2026-07-21T16:09:00+00:00"
---

# EXP-656 tensor-native cached rollout route

## Question

Candidate005's wide rollout decodes the predicted bit tensor into Python big
integers after every Horner step, then reconstructs state, multiplicand, modulus,
input-bit, and marker planes for the next step. EXP-656 asks whether loop-invariant
code motion and tensor-native recurrent state materially reduce the actual
width-256 route cost without changing the learned computation.

This is an input-only performance experiment. It produces no answer label,
exact-answer accuracy, model update, or candidate-promotion evidence.

## Frozen source and routes

The frozen source is `submission/model.py` SHA-256
`e3d4aa9ff3e22dde516d261b099826622fa37d0688fae47c07ff95fe876e9477`
and `submission/weights.safetensors` SHA-256
`2f6ac3199d0fd9e16afae165c2d4a16919e6c8bf4545363a6f12d077279d27ca`.
The legacy arm calls the existing `_bigint_features` and
`_decode_bigint_logits` path at every bit. The challenger precomputes the
multiplier, multiplicand, modulus, LSB, and MSB planes, mutates only state and
current-input-bit planes, retains thresholded state as a Boolean tensor, and
decodes once at the output boundary. Both arms invoke the same frozen network
with the same schedule and weights. Neither performs arithmetic repair.

## Matched benchmark

Seed `20260721` deterministically generates one batch of 16 high-bit inputs for
each of widths 128 and 256. No expected products are generated. Each route gets
one unmeasured warmup. Four measured paired repeats alternate which arm runs
first, so each arm is first twice. Wall time and aggregate process CPU time are
recorded per call. The hard ceilings are 900 wall seconds and 3,600 aggregate
process CPU seconds for the complete run.

Every pair must have bit-identical final predictions and bit-identical
minimum-margin tensors. Any semantic difference is a hard stop, not a timing
tradeoff.

## Decision rule

Advance the route only if all semantic comparisons are exact, the median
tensor-native width-128 wall time is no slower than legacy, and the median
tensor-native width-256 wall time is at most 0.80 times legacy. Otherwise record
`reject-tensor-native-route`. A passing input-only result would authorize a
separate integration/re-evaluation decision; it would not itself authorize a
submission edit.

## Result

The matched input-only run completed in 315.284 wall seconds and 1,177.47
aggregate process CPU seconds with 447,901,696 bytes peak RSS. All four paired
repeats at both widths produced bit-identical final outputs and bit-identical
minimum-margin tensors.

At width 128, legacy and tensor-native median wall times were 5.4273 and 4.7588
seconds, a tensor/legacy ratio of 0.8768 (12.32% reduction). At width 256, the
medians were 26.3015 and 23.4371 seconds, a ratio of 0.8911 (10.89% reduction).
Width-128 non-regression passed, but the binding width-256 ratio missed the 0.80
gate. The decision is `reject-tensor-native-route`: representation caching is a
real local optimization, but convolutional model work dominates enough that this
change alone does not justify submission integration or candidate re-evaluation.

This result reports no exact-answer accuracy. Durable evidence is in
`experiments/runs/exp-656-tensor-native-rollout-route/run.json` and
`evals/results/exp-656-tensor-native-rollout-route/metrics.json`; MLflow run
`cae18b9e8a1b441b973178536052d9f3` records the same input-only decision.

The first project-wide validation found that the newly written JSON envelopes
omitted five repository-schema aliases. A post-run normalization added only the
required no-evaluation declaration, controls/runtime aliases, checkpoint field,
challenge commit, and artifact map. The pre-normalization run and metrics hashes
are retained in both normalized artifacts; no benchmark value, parity result,
gate, decision, seed, or source checkpoint changed.
