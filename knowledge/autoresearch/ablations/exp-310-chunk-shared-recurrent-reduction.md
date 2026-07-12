---
type: Experiment Preregistration
title: EXP-310 Chunk-Shared Recurrent Reduction
description: Matched mixed-width hard-mining comparison of a chunk-shared bidirectional GRU and the rejected dilated ConvGRU control.
status: preregistered
tags: [autoresearch, recurrent, chunked, mixed-width, length-transfer]
timestamp: "2026-07-12T10:45:00+05:30"
---

# EXP-310: chunk-shared recurrent reduction

EXP-220's per-bit dilated ConvGRU was width-independent but learned poorly. EXP-301's full-canvas TCN learned width-64 transitions but its width-specific global head and accumulated error make scaling untenable. Test a middle representation: encode shared 4-bit chunks, propagate global context with a bidirectional GRU cell shared for three refinement cycles, and decode every chunk with one shared local head.

Train the chunk-GRU and EXP-220 ConvGRU control from random initialization on identical hard-mined batches cycling through widths 8, 12, and 16. Both state dicts are exactly width-independent. Evaluate held-out width-16 learning and zero-shot width-32 length transfer on fresh identities.

Require the challenger to reach 85% width-16 transition and 10% rollout, beat the matched control by 15 transition points and five rollout points, and reach 55%/0.2% zero-shot width-32 transition/rollout. Parameter and training-time ratios must remain below 2x. Passing authorizes a larger mixed-width curriculum; failure closes this chunk-GRU formulation. Exact arithmetic is offline label/evaluation machinery only. No checked or consumed independent answers may be read. Stop at twelve wall minutes.
