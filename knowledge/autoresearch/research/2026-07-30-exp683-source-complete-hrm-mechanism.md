---
type: Research Note
title: EXP683 Source-Complete HRM Mechanism Map
description: Visual HRM source authentication and frozen SAIR de-novo mapping.
status: permanent-calibration-execution-park
tags: [autoresearch, HRM, hierarchy, deep-supervision, ACT, Q-learning]
timestamp: "2026-07-30T00:00:00+05:30"
---

# EXP683 source-complete HRM mechanism map

The immutable local `arxiv-2506.21734.pdf` hashes to
`81b05b03ebd92748b1bc0e59d4b6e0d8271e1e56b84d1d589a03430db79b6e25`.
Pages 4 and 6-9, including Figures 4-5, were rendered and visually inspected. The paper
was sufficient to freeze the semantics, so no official repository or web source was used.

Page 4 defines two carried, interdependent states. At every timestep L consumes its prior
state, the current H state, and the embedded input. H updates only after every `T` L updates
and consumes the final L state. The output is decoded from H. EXP683 uses the paper's
pseudocode values `N=2,T=2`, distinct same-sized H/L modules, elementwise input merging,
and fixed truncated-normal initial states.

Pages 6-7 define the local one-step gradient: the first `NT-1` updates execute without a
graph, then the final L and H updates carry gradient to the output and input embedding.
Deep-supervision segments carry H/L forward but detach them between losses. Removing the
segment detach alone is graph-redundant under exact one-step credit because the next
segment's forward-only prefix already severs the graph; EXP683 records that rather than
claiming a nonexistent independent causal effect. Full-gradient and no-deep-supervision
controls remain distinct.

Pages 7-9 define deep-Q ACT. A sigmoid Q head reads only final H. Offline exact prediction
correctness supplies the halt target; continue bootstraps from the next segment's detached
Q values. Release inference is deterministic and target-free: halt when learned
`Q_halt>Q_continue`, bounded by four segments. Fixed-max, fixed-average, deterministic-
random, Q-disabled, and shuffled-Q controls isolate selection. Figure 5 is source-reported
Sudoku evidence, not SAIR evidence.

Exact Candidate005 transfer is impossible: its single GroupNorm ConvGRU state cannot be
split into two distinct PostNorm-RMSNorm modules plus Q head while retaining exact outputs.
Therefore the frozen route is the mandated 4,000-update Candidate-sized de-novo model,
using native PyTorch scaled-dot-product attention, RoPE, bias-free GLU, RMSNorm, and AdamW.
This is a SAIR sequence adaptation, not a reproduction of the paper's datasets or claims.

EXP620 and EXP633 are non-HRM negatives. EXP667 and EXP668 are adjacent two-state/
hierarchy calibration parks; neither contains the complete H/L plus local-credit plus deep-Q
ACT system. Exact arithmetic remains confined to offline labels and evaluation. Inference
has no correctness, validity, repair, lookup, or external-execution input.

T086 made the package oracle portable without changing these scientific bytes: the
focused cone passed 156/156 and the one full oracle passed 703/703. Root 20261282 was
consumed after persisting 13 trained one-update routes and two untrained routes, but a
checkpoint-path TypeError stopped execution before evaluation or admission. No component
received scientific evidence, so M007.a-c and the parent HRM mechanism remain open while
this exact EXP683 execution formulation is permanently parked.
