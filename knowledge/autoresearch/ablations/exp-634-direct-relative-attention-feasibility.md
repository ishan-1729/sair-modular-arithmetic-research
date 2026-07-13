---
type: Experiment Preregistration
title: EXP-634 Direct Relative-Attention Feasibility
description: Matched direct-product feasibility comparison of recurrent shared attention and dilated convolution with fixed relative bit-position features.
status: preregistered
tags: [autoresearch, direct-product, universal-transformer, relative-position, attention, length-generalization]
timestamp: "2026-07-13T17:25:00+05:30"
---

# EXP-634 direct relative-attention feasibility

## Hypothesis and evidence basis

EXP-633 showed that shallow segment states can become partially exact while repeated learned binary composition never produces an exact root. EXP-634 removes that intermediate bottleneck: every output bit is supervised directly from a representation with global access to both operands and the modulus. One width-shared attention block is recurrently reused for a width-dependent number of depth iterations, following the recurrent-depth and length-generalization motivation already Playwright-verified from the Universal Transformer and relative-position arithmetic literature in `knowledge/autoresearch/research/2026-07-12-width-independent-arithmetic.md`.

The hypothesis is that recurrent global attention with fixed sinusoidal position features learns direct modular products and extrapolates across width better than a parameter-matched direct dilated-convolution control. This is not a claim that attention will outperform the mature sequential cellular transition model at its trained widths; the first rung asks only whether direct global computation is viable enough to justify scale.

## Models and trained-parameters boundary

Both arms receive identical LSB-first bit tokens for the two operands and modulus, LSB/MSB boundary flags, and fixed sine/cosine features of normalized distance from the boundaries. Neither arm receives products, quotients, carries, reductions, or hand-engineered partial products. The attention arm uses PyTorch `MultiheadAttention`, a shared feed-forward block, input reinjection, and a shared local bit head. The same block is applied `2 * ceil(log2(width))` times. The control uses a shared gated dilated convolution over the same feature planes and a local bit head.

Python bigint multiplication and modulus are allowed only in offline labels and evaluation truth. Inference contains no modular multiplication, division, reduction, Barrett/Montgomery routine, or CRT recombination. Fixed sinusoidal features identify position but do not encode arithmetic answers.

## Matched training and evaluation

Both arms use the same generated prime identities, alternating widths 8 and 12, 4,000 optimizer updates, batch size 64, direct product labels, AdamW schedule, clipping, and per-bit plus worst-bit loss. Training identities are disjoint from evaluation identities. Evaluation uses 300 fixed matched products at trained width 12 and zero-shot width 24, plus a 100-case randomized-weight control and two deterministic 20-case repeats.

Advance only if every gate passes:

- attention width-12 exact accuracy at least 20% and at least five points above the direct convolution control;
- zero-shot width-24 attention exact accuracy at least 2% and at least two points above control;
- trained attention at least ten points above randomized weights on matched width-24 cases;
- invalid-residue rate at most 5% at both widths;
- exact repeated predictions and metrics;
- attention parameter count at most 115% of control;
- projected integrated candidate inference at most 260 seconds.

Passing is a feasibility result only. It authorizes a higher-width training plan, not independent, checked-public, official, private, or submission evaluation. Missing any gate closes this direct recurrent-attention formulation at the declared capacity/rung; do not tune thresholds or depth on the observed evaluation cases.

## Dynamic resource-budget protocol

Before formal execution, run 30 representative training-only updates per arm and 20 training-identity inference cases at each evaluation width. Record wall and aggregate CPU separately, include setup/evaluation work and safety headroom, set a hard runaway ceiling, and checkpoint model/optimizer/scheduler/RNG state at a calibrated cadence. Calibration may alter only resource and resumability fields, not the models, 4,000 updates, cases, data split, or gates. Full evaluation fidelity must not be reduced to meet an arbitrary short wall cap.

The first calibration invocation stopped at the explicit capacity guard before split generation, model initialization, updates, or evaluation: the configured width-12 band contains 510 unique primes, fewer than the requested 640. The corrected frozen split uses 400 training and 100 evaluation identities at width 12, retains 40/8 at width 8 and 512/128 at width 24, and leaves all model, optimization, case-count, and gate fields unchanged.

The corrected training-only calibration completed in 26.27 external wall seconds at 208% CPU utilization. Thirty alternating updates took 4.820 wall / 18.044 aggregate CPU seconds for the convolution control and 4.775 wall / 17.921 aggregate CPU seconds for attention. The arms contain 52,345 and 53,122 parameters. The resulting projection is 1,279.323 wall / 4,795.381 aggregate CPU seconds for formal training and 3.043 wall / 11.354 aggregate CPU seconds for evaluation.

Including randomized/determinism controls and 180 seconds of setup/artifact allowance, expected total is 1,463 wall seconds and about 81 aggregate CPU minutes. The planning soft budget is 1,980 wall seconds; the hard runaway ceiling is 2,940 wall seconds, with a separate four-thread upper bound of 196 aggregate CPU minutes. The ignored calibration record hashes to `e832c68582be8624f0804d9a132215fc5046aef4988711cb7afe99993b583ebd`; its console hashes to `247ae1cb9721005d82fd6b3d5a2b007b3f42112ceacf9529703d165b57ed2ce3`.

## Limitations

The direct control and attention arms are parameter- and data-matched but perform different operations. Failure would reject this recurrent shared-attention prescription, not all transformer arithmetic, scratchpad decoding, or larger pretrained sequence models.
