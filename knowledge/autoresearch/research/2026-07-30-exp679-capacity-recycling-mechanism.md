---
type: Research Note
title: EXP679 Operational Capacity Recycling Mechanism Map
description: Page-authenticated Loopies M043.b operational capacity map, exact embedding, and SAIR exclusions.
status: permanent-pre-root-denominator-park
tags: [autoresearch, loopies, M043, exp679]
timestamp: "2026-07-30T02:00:00+05:30"
experiment: EXP679
---

# EXP679 operational capacity recycling mechanism

Observed source authentication used the immutable local `arxiv-2607.16051.pdf` (SHA-256 `1abfc7644b0253119b53b86327011471a0d017c2904d648461fc9d6a10b074ce`). Rendered PDF pages 6-10 and 35 were visually inspected. Section 2.4 states that Loopie reduces stored depth, measures activation-memory and optimizer-step efficiency, reinvests realized headroom into capacity, and chooses the final architecture by measured end-to-end time rather than theoretical FLOPs. Table 1 freezes a small width/depth candidate grid; Appendix Table 4 confirms two layer-loop applications and the selected stored widths/depths.

EXP679 transfers only M043.b's operational principle. Candidate005's mandatory six-channel input is reinjected at every retained recurrent visit. A deterministic, target-independent evenly spaced subset of its original dilation schedule reduces calls. A separate learned channel bank widens the recurrent state; separate normalization preserves the old 92-channel computation, and zero-initialized extra readouts give bitwise exact full-route embedding. Native PyTorch profiling plus process-tree/cgroup measurements select the largest safe grid point using only timing, RSS, activation bytes, parameters and call accounting.

Excluded are M043.a fixed-R2 layer/model-loop ordering, M043.c MoE/expert routing, M009.b learned recursion-depth routing, answer-dependent early exit, and the paper's GPU microbatch scaling claim. Exact arithmetic is offline-label/evaluator-only. No checked, independent, official, private, candidate, submission, promotion, or completion evaluation is authorized.

The mapped implementation passed eight focused tests and the 663-test project oracle on the exact frozen bytes. Pre-root inspection then found two trajectory-denominator keys required by the runner but absent from the verified config. Because repairing that conflict would invalidate the sole verified config identity, the exact EXP679 formulation is permanently parked with all roots unopened. No capacity-selection, resource-admission, training, exact-answer, or causal-route result exists; M043.b remains scientifically unanswered.
