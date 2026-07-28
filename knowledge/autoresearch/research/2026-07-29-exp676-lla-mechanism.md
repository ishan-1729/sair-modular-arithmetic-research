---
type: Research Note
title: EXP676 LLA reinjection and moving-window map
description: Visual source map from LLA input injection and moving truncated loss to Candidate005 physical visits.
status: active
tags: [autoresearch, LLA, moving-window, input-reinjection]
timestamp: "2026-07-29T13:00:00+05:30"
---

# EXP676 LLA mechanism map

The authenticated ICLR 2024 PDF was visually inspected on pages 5-6. Page 5
contrasts weight tying `Y[t+1]=M(Y[t])` with LLA input injection
`Y[t+1]=M(Y[t]+P)` and explicitly permits non-additive injection. It also defines
the maximum endpoint `b` and a `T`-output truncated loss. Page 6 describes the
scheduled shift from early `[1,T]` outputs to late `[b-T,b]` outputs as `b`
progressively increases.

Candidate005 already reinjects its immutable legal transition features in both
the reset/update and candidate convolutions at every power-of-two dilation
visit. This is topology-equivalent non-additive input injection, so adding an
extra path would duplicate rather than isolate M004.a. The experiment changes
only training credit: the unchanged local/quotient heads expose physical-visit
endpoints, and four selected endpoint losses are averaged.

The LLA language-model/linear-regression task, additive embedding geometry,
MSE, absolute loop counts `b=20,T=15`, and in-context prompt setup are not
portable. EXP676 adapts `T` to four of Candidate005's 12/14 physical visits at
W64/W128 while preserving the source's progressive contiguous-window rule.
M004.b and every inference-time depth or selection mechanism remain excluded.
Native PyTorch is sufficient; no external loss/sampling package improves source
fidelity.
