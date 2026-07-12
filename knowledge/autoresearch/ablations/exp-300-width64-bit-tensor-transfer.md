---
type: Experiment Preregistration
title: EXP-300 Width-64 Bit-Tensor Transfer
description: Matched random-versus-width-32-transfer smoke test using Python bigint labels and tensor-native learned inference.
status: preregistered
tags: [autoresearch, width-64, bit-tensor, transfer, Tier-5]
timestamp: "2026-07-12T10:25:00+05:30"
---

# EXP-300: width-64 bit-tensor transfer

The prior sampler stores values in signed `int64` and cannot represent Tier-5 moduli. Introduce a training/evaluation-only Python-bigint sampler that computes exact offline labels, then converts state, multiplicand, modulus, and target to bit tensors before the learned model. The model receives only tensors and its inference path performs no exact modular arithmetic.

Train matched width-64 random and width-32-transfer arms for 400 updates on identical prime identities and random streams. Transfer copies all shape-compatible tensors and the learned 32-bit overlaps of position, modulus context, global output, and quotient heads. Evaluate on disjoint fresh identities with 10,000 transitions and 2,000 autonomous residue rollouts.

Require transfer to add at least 15 transition points and five rollout points, reach at least 50% transition accuracy, and keep invalid rollouts below 50%. Passing only establishes a viable Tier-5 optimization path and authorizes a larger run; it does not authorize submission integration or checked evaluation. Stop at eight wall minutes.
