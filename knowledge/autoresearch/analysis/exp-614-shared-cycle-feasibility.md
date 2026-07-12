---
type: Feasibility Analysis
title: EXP-614 Shared-Cycle Reuse Feasibility
description: Timing profile showing that executing a two-cycle checkpoint at untrained recurrent depths is not function preserving.
status: completed
tags: [autoresearch, tier6, recurrent-depth, feasibility]
timestamp: "2026-07-13T03:25:00+05:30"
---

# EXP-614 shared-cycle reuse feasibility

A preselection timing profile executed the frozen EXP-609 two-cycle checkpoint with one through four shared recurrent cycles on 200 generated width-128 transitions and 20 generated rollouts per depth. The counts were collected to exercise realistic runtime paths, not as promotion evidence.

The trained two-cycle execution scored 200/200 transitions and 17/20 rollouts. One cycle scored zero transitions. Three and four cycles scored only 46/200 and 61/200 transitions and zero rollouts; invalid-rollout rates were 100% and 95%. Wall times were 7.91, 15.73, 20.17, and 29.76 seconds for cycles one through four, with aggregate CPU approximately 3.88 times wall.

Observation: recurrent weights are not iteration-stable outside their trained two-pass schedule. Therefore EXP-614 must not treat an extra shared cycle as a valid refinement. The supported architecture uses separate refinement parameters behind an exact zero-initialized hidden-state blend.
