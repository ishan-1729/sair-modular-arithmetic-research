---
type: Experiment Preregistration
title: EXP-322 Cellular Width-64 Smoke
description: Compute-bounded direct Tier-5 distribution continuation after EXP-321's runtime projection.
status: preregistered
tags: [autoresearch, cellular, width-64, smoke, Tier-5]
timestamp: "2026-07-12T13:00:00+05:30"
---

# EXP-322: cellular width-64 smoke

Repeat EXP-321's alternating width-56/64 official-numeric curriculum from frozen EXP-320, but use 400 updates, 500 frozen-source rollouts, and 1,500 final width-64 rollouts. All data identities and training variables remain as preregistered; only fidelity is reduced to fit the twelve-minute budget.

Require 85% retained width-56 rollout; 99.7% width-64 transition; 90% width-64 rollout; an eight-point rollout gain; and at most 0.5% invalid rollouts. Passing with at least 93% absolute rollout authorizes five untouched Tier-5 seeds; a pass below 93% authorizes one confirmation. No checked, consumed independent, or private answers may be read.
