---
type: Experiment Preregistration
title: EXP-606 Tier-6 Checkpoint Complementarity
description: Fresh per-case overlap, weight-soup, and fixed recurrent-logit ensemble analysis.
status: completed
tags: [autoresearch, Tier-6, ensemble, weight-soup, complementarity]
timestamp: "2026-07-13T04:20:00+05:30"
---

# EXP-606: Tier-6 checkpoint complementarity

Freeze the EXP-603 four-step checkpoint and EXP-604 eight-step checkpoint. On identical fresh width-64/128 cases, record endpoint correctness overlap and oracle union, then evaluate two realizable fixed combinations: a 50/50 parameter soup and a 50/50 recurrent logit ensemble. The logit ensemble feeds both component models the same current ensemble state and advances that state from averaged logits; it does not select answers per case. No weights are trained or selected from checked data.

Advance only if endpoint oracle union reaches at least 92% width-128 rollout and one fixed combination reaches at least 90% width-128, 94% width-64, 99.8% width-128 transition, at most 1% invalid width-128 rollout, and projects below 240 seconds for the complete candidate inference path. This is a selection analysis; a passing fixed combination still requires higher-fidelity and five-seed confirmation.

## Dynamic resource budget

EXP-604's two trained-and-evaluated arms completed in 760.650 wall seconds; EXP-605's paired source/challenger evaluation plus training completed in 791.004 seconds. With no optimizer work but four frozen arms, project 650 wall seconds. Allocate 1,200 wall seconds soft and a 1,800-wall-second hard runaway ceiling, tracking aggregate CPU separately under a conservative 120 CPU-minute ceiling at four threads. Use identical materialized cases for every arm and save combined safetensors before reporting.

## Result

Reject and close checkpoint combination. Both endpoints score 82.0% width-128 rollout on identical fresh cases. Oracle union is only 84.5% with 79.5% intersection; each endpoint uniquely solves 2.5%. The 50/50 weight soup and realizable recurrent logit ensemble each score 83.0%, while the selected soup retains 97.5% width-64 and projects to 91.9 seconds for the complete candidate. Runtime is 657.936 wall seconds and 2,588.258 aggregate CPU seconds, within budget.

Shared errors dominate, so neither weight-space nor logit combination can reach the gate. Move to a training intervention that propagates credit through multiple soft learner states rather than combining frozen endpoints.
