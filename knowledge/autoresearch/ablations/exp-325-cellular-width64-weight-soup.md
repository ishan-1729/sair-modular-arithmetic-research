---
type: Experiment Preregistration
title: EXP-325 Cellular Width-64 Weight Soup
description: Fresh-data interpolation test between consecutive diverse Tier-5 checkpoints.
status: preregistered
tags: [autoresearch, cellular, width-64, model-soup, Tier-5]
timestamp: "2026-07-12T13:50:00+05:30"
---

# EXP-325: cellular width-64 weight soup

Interpolate every trained floating-point tensor between frozen EXP-323 (`alpha=0`) and EXP-324 (`alpha=1`) at alphas 0, 0.25, 0.5, 0.75, and 1. Evaluate every arm on identical fresh official-numeric width-64 identities from seed 20260714. Interpolation is performed once offline; the selected artifact has the same architecture, parameter count, and inference cost.

Require the best arm to reach 99.8% transition, 93% rollout, improve at least one point over the better endpoint on the paired data, and keep invalid rollouts at or below 0.5%. Passing authorizes five untouched Tier-5 seeds. Failure closes this weight-soup intervention. No checked, consumed independent, or private answers may be read. Stop at twelve wall minutes.
