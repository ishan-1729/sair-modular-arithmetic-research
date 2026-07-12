---
type: Experiment Preregistration
title: EXP-310 Chunk-Shared Recurrent Reduction
description: Matched mixed-width hard-mining comparison of a chunk-shared bidirectional GRU and the rejected dilated ConvGRU control.
status: completed
tags: [autoresearch, recurrent, chunked, mixed-width, length-transfer]
timestamp: "2026-07-12T10:45:00+05:30"
---

# EXP-310: chunk-shared recurrent reduction

EXP-220's per-bit dilated ConvGRU was width-independent but learned poorly. EXP-301's full-canvas TCN learned width-64 transitions but its width-specific global head and accumulated error make scaling untenable. Test a middle representation: encode shared 4-bit chunks, propagate global context with a bidirectional GRU cell shared for three refinement cycles, and decode every chunk with one shared local head.

Train the chunk-GRU and EXP-220 ConvGRU control from random initialization on identical hard-mined batches cycling through widths 8, 12, and 16. Both state dicts are exactly width-independent. Evaluate held-out width-16 learning and zero-shot width-32 length transfer on fresh identities.

Require the challenger to reach 85% width-16 transition and 10% rollout, beat the matched control by 15 transition points and five rollout points, and reach 55%/0.2% zero-shot width-32 transition/rollout. Parameter and training-time ratios must remain below 2x. Passing authorizes a larger mixed-width curriculum; failure closes this chunk-GRU formulation. Exact arithmetic is offline label/evaluation machinery only. No checked or consumed independent answers may be read. Stop at twelve wall minutes.

## Outcome

The chunk-GRU is rejected: 38.335% width-16 transition, 0% rollout, and 44.435% zero-shot width-32 transition. The matched cellular control is the useful finding. Under the mixed 8/12/16-bit curriculum it reaches 91.19% width-16 transition, 28.2% rollout, 75.115% zero-shot width-32 transition, and 0.2% zero-shot rollout. Relative to EXP-220, the same cellular architecture improves width-16 transition by 30.945 points and rollout by 28.09 points. Advance the mixed-width intervention with the cellular checkpoint; do not advance the chunk-GRU.
