---
type: Experiment Preregistration
title: EXP-602 Tier-6 Recovery Full Budget
description: Higher-fidelity matched continuation with explicit width-64 and width-128 autonomous-state recovery.
status: preregistered
tags: [autoresearch, Tier-6, cellular, autonomous-state, full-budget]
timestamp: "2026-07-13T02:00:00+05:30"
---

# EXP-602: Tier-6 recovery full budget

Continue from the exact EXP-601 checkpoint (`915534...82d8`) on fresh identities. Cycle one width-64 teacher-transition update, one width-64 learner-state recovery update, and two width-128 learner-state recovery updates. This preserves the successful 50% width-128 recovery allocation while directly training against the observed width-64 retention risk. Use 400 updates, four-step detached learner-state chunks, 512 train and 128 test prime identities per width, and matched frozen-source/challenger evaluations of 3,000 transitions plus 300 rollouts per width.

Pass only if width-128 rollout reaches at least 85%, gains at least five points over the frozen source on identical cases, width-64 rollout reaches at least 92%, width-128 transition reaches at least 99.7%, and invalid width-128 rollouts remain at or below 1%. Passing authorizes five untouched official-generator Tier-6 seeds; it does not itself authorize Candidate 005 or checked evaluation.

## Dynamic resource budget

EXP-601 spent 214.403 wall seconds on 200 training updates, projecting 428.807 seconds for 400 updates. Its paired source/challenger setup and evaluation consumed approximately 142.142 seconds at 100 rollouts; tripling evaluation fidelity projects 426.427 seconds. Total projected wall time is 855.234 seconds. Allocate 1,500 wall seconds soft and a 2,400-wall-second hard runaway ceiling, with a conservative 160 aggregate CPU-minute upper ceiling at four threads. These are distinct measures. Cache the source control only when checkpoint, config, dataset, and evaluator hashes match, and save the challenger checkpoint before long evaluation.
