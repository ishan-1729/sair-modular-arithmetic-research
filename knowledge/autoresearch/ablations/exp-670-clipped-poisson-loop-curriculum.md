---
type: Ablation Study
title: EXP-670 clipped-Poisson loop curriculum
description: Bounded ST-M018.a/M026.b implementation parked at the required package-verification boundary.
status: parked
tags: [autoresearch, ablation, recurrent-depth, poisson]
timestamp: "2026-07-29T05:00:00+05:30"
---

# EXP-670 — clipped-Poisson loop curriculum

EXP-670 maps the authenticated source's dynamic recurrence to a position-free shared Conv1d residual
refinement around the frozen EXP609 proposal. The operator is exact-zero-initialized and reinjects the
proposal plus legal input features. The mandatory operand-bit scan is inherited unchanged. The source
schedule is `clip(Poisson(4), 2, 8)`; its frozen 400-update stream has exact mean 4 and the matched controls
all use exactly 1,600 refinement calls. Formal inference is fixed at four target-free refinements. M018.b's
hop/KL/entropy halting curriculum remains closed.

The focused suite passed 12 tests in 11.64 seconds. The single required final package oracle stopped in
project validation after 9.29 seconds. Three owned frontmatter omissions were diagnosed and corrected in
the terminal records, but the validator also rejected the pre-existing integrated
`evals/results/exp-669-stars-jvp-regularization/metrics.json:evaluation` object. That file is outside T065's
allowed files. The explicit stop rule forbids changing unrelated/protected files or opening science after a
red final package verdict. Therefore root 20261230 calibration, root 20261231 viability, and roots
20261232/33 confirmations remain unopened. This is a package-verification park, not a scientific verdict on
the curriculum, ST-M018.a, ST-M026.b, or the source paper.
