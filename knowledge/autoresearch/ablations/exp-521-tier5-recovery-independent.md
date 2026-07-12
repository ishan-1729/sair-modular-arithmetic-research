---
type: Experiment Preregistration
title: EXP-521 Tier-5 Recovery Independent Gate
description: Five untouched official-generator Tier-5 seeds for the frozen full recovery checkpoint.
status: preregistered
tags: [autoresearch, Tier-5, independent, official-generator, cellular]
timestamp: "2026-07-12T23:40:00+05:30"
---

# EXP-521: Tier-5 recovery independent gate

Freeze the exact EXP-329 full recovery checkpoint (`050daf...76f7`) and evaluate it once on five preregistered, untouched pinned official-generator Tier-5 seeds, 100 cases each. The seed strings are committed before generation; only their SHA-256 hashes are retained in durable results. Width-64 inputs, states, and predictions use Python integers outside the model because official primes may exceed signed `int64`; the model receives and emits only legal bit tensors. Exact arithmetic is confined to generator truth and evaluation.

This is a new confirmation authorized by two separate fresh selection sets: EXP-329 scored 94.267% versus 93.133% control, while EXP-520 scored 96.4% versus 94.0% control and causally showed the benefit is distributed across the full recurrent checkpoint. This does not retroactively reclassify EXP-329. Pass only if every seed reaches at least 90% and aggregate reaches at least 92%. Passing authorizes Tier-5 integration beside Candidate003's Tier-1–4 specialists; failure closes this endpoint. Checked and private answers remain unread.

## Dynamic resource budget

EXP-520's full challenger evaluates 500 width-64 rollouts in 45.523 wall seconds. EXP-521 also evaluates 500 total rollouts; add 60 seconds for official generation and setup for 105.523 projected wall seconds. Allocate a 300-wall-second soft budget and 600-wall-second hard runaway ceiling. These are wall limits; four threads imply a conservative 40 aggregate CPU-minute upper bound.
