---
type: Experiment Preregistration
title: EXP-329 Cellular Width-64 Autonomous-State Recovery
description: Matched teacher-forced continuation versus learner-induced state-recovery training at Tier 5.
status: completed
tags: [autoresearch, cellular, width-64, rollout-aware, matched-ab, Tier-5]
timestamp: "2026-07-12T16:10:00+05:30"
---

# EXP-329: cellular width-64 autonomous-state recovery

Start two arms from the exact frozen EXP-324 checkpoint on fresh seed-20260716 train/test prime identities. Both receive 400 optimizer updates, the same optimizer schedule, 128 transition-equivalent examples per update, alternating width-56/64 coverage, and identical full evaluations. The control uses the established teacher-forced hard-mined transition update throughout. In the challenger, every fourth update is replaced by 32 four-step width-64 trajectory chunks. Each chunk begins at an exact offline-teacher state; later model inputs mix detached learner predictions with teacher states on a preregistered 50% to 100% schedule, while targets remain the correct teacher trajectory. Quotient auxiliary loss is zero on recovery updates because an exposed erroneous state and the teacher trajectory need not share a quotient; ordinary updates retain the 0.10 quotient weight. This isolates recovery from learner-induced state drift. Exact arithmetic remains offline in training labels and evaluation; inference remains tensor-only.

The intervention passes only if the challenger reaches at least 99.8% transition and 93% rollout at width 64, gains at least 1.5 points over the matched control, regresses transition by no more than 0.1 point, has at most 0.5% invalid rollouts, and retains at least 92% width-56 rollout. Passing authorizes five untouched official-generator Tier-5 seeds. Failure closes this recovery formulation. Checked, consumed independent, and private answers remain unread.

## Dynamic resource budget

EXP-324 measured 535.222 wall seconds for one 400-update cellular continuation and the same 5,000-transition/1,500-rollout width-64 evaluation class. Two matched arms project to 1,070.445 wall seconds. Apply a 1.25 recovery-training overhead factor for the challenger's sequential chunk forwards, producing a 1,338.056-second projection. Allocate a 2,400-wall-second soft budget and a 3,600-wall-second hard runaway ceiling. These are wall-clock limits. At four configured threads the ceiling implies a conservative 240 aggregate CPU-minute upper bound; measured wall and aggregate CPU time must be reported separately. Save and integrity-key each completed arm so valid evidence survives an interruption.

## Outcome

Narrow rejection. The teacher-forced control reaches 99.86% transition and 93.133% rollout at width 64. The recovery challenger reaches 99.92% transition, 94.267% rollout, 0.133% invalid rollouts, and 96.8% retained width-56 rollout. It passes every absolute and retention gate but gains 1.133 points over the paired control, below the frozen 1.5-point causal threshold. Do not round, relax the gate, or authorize independent evaluation. Close this scheduled recovery formulation.

The runner records 1,755.0 wall seconds and 6,367.7 aggregate CPU seconds. External `/usr/bin/time` records 1,620.1 elapsed seconds, 6,381.8 user-plus-system CPU seconds, 393% utilization, and 1.22 GiB peak RSS. Both wall observations are retained because their 134.9-second discrepancy is unexplained. The comparison completes inside the 2,400/3,600-second soft/hard wall budgets.
