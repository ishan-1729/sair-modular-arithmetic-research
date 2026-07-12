---
type: Experiment Preregistration
title: EXP-220 Width-Independent Cellular Architecture
description: Matched width-16 learning control plus zero-shot width-32 length-transfer smoke test.
status: completed
tags: [autoresearch, architecture, convgru, cellular, length-generalization]
timestamp: "2026-07-12T11:30:00+05:30"
---

# EXP-220: width-independent cellular architecture

## Changed architecture

The control is the existing width-16 shared-context full-canvas TCN. The challenger is a shared dilated ConvGRU with no learned absolute position tensor, width-sized modulus MLP, or width-sized global head. One gated convolutional cell is reused across positions, recurrence cycles, and exponentially increasing dilations. Original bit planes and LSB/MSB markers are injected at every update; output bits use a local shared head, and the auxiliary quotient head pools across positions.

Both arms train from deterministic random initialization on identical width-16 hard-mined batches, prime identities, 800 optimizer steps, 256-row batches, losses, optimizer, and evaluation cases. Parameter count and training time are reported rather than silently matched; the gate limits each ratio.

After width-16 training, load the exact challenger state dict into the same architecture configured for width 32 and evaluate on unseen 17–32-bit prime identities without fine-tuning. The control cannot perform this test because it contains width-specific tensors.

## Gate

The challenger must reach at least 90% width-16 transition and 25% rollout, remain within 15 rollout points of the matched control, and achieve at least 50% transition and 0.5% rollout zero-shot at width 32. Parameters must be at most 1.25 times control and training time at most twice control. Passing authorizes a full-budget/mixed-width continuation; failing rejects this shared-cell formulation. No consumed independent seed or checked public answer is used.

## Infrastructure retry note

The first invocation was terminated before MLflow initialization, output creation, or training because the bounded width-32 split helper requested more unique primes than exist in the smallest 9-bit training bands. The retry uses the original exhaustive splitter for the safe 9–16-bit range and retains bounded sampling only at width 32. No model, seed, batch, optimizer, evaluation count, resource cap, or gate changed; the aborted invocation produced no model result.

## Outcome

Rejected. The cellular model reached 60.245% width-16 transition and 0.11% rollout versus 61.94%/0.17% for the matched control. Its unchanged state dict produced 61.935% zero-shot width-32 transition but 0% rollout with 68.7% invalid rollouts. Parameter ratio passed at 1.17, but training took 4.63 times control versus a maximum of two. The aggregate run completed 1.44 seconds inside its 12-minute cap. Structural width independence did not yield exact arithmetic or acceptable runtime.
