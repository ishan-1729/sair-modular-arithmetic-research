---
type: Ablation Study
title: EXP-669 STARS JVP regularization
description: Frozen continuation arms, dynamically budgeted verification, calibration, and conditional W128-first viability.
status: rejected
tags: [autoresearch, ablation, stars, jvp]
timestamp: "2026-07-29T04:00:00+05:30"
---
# EXP-669 — STARS JVP regularization

The preregistered M026.a package compares STARS, lambda-zero continuation, state-norm, random-direction/stop-gradient, frozen source, and randomized weights. Every trained arm starts from identical EXP609 bytes and inference remains the unmodified source graph.

Verification passed 16 focused tests, 184 concepts/1,100 records, and all 573 project tests. Root 20261226 admitted all routes with a 4,625.59-second expected formal wall, 6,894.08-second ETA, and 236.68-second candidate headroom.

Root 20261227 trained and cached all four continuation arms for 400 updates/6,400 examples/12,800 JVP calls each, then opened only the complete W128 STARS primary. It scored 99.8333% transition exactness, 84.3333% rollout exactness, and zero invalid transitions/rollouts. The 85% rollout floor missed by 0.667 points, an irreversible primary-gate failure. Retention/dependence controls, formal JVP/perturbation comparisons, W256, and roots 20261228/29 remain unopened. This rejects only the exact bounded adaptation; it does not establish a general STARS or spectral-radius result. M026.b/M018.a remains closed.
