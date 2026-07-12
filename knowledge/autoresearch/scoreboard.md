---
type: Experiment Scoreboard
title: Autoresearch Scoreboard
description: Comparable exact-accuracy, runtime, uncertainty, and promotion status across durable runs.
status: active
tags: [autoresearch, scoreboard, metrics]
timestamp: "2026-07-12T02:20:00+05:30"
---

# Autoresearch scoreboard

| Run | Role | Independent overall | Tier 1 | Tier 2 | Tier 3 | Highest >=90% | Parameters | CPU inference | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `candidate-001-horner-tcn9` | retained baseline | 21.3% | 100% | 97% | 2% | 2 | 85,021 | 0.4 s sandbox | superseded |
| `candidate-002-tier3-hybrid` | promoted | 31.0% | 100% | 97% | 99% | 3 | 170,577 | 1.8 s official sandbox | incumbent |

The incumbent Tier-2 estimate is 492/500 (98.4%) across five additional generator seeds plus 97/100 on the retained full independent seed. Candidate comparisons must report all scored tiers and preserve this row even after promotion.

## Research gates

| Run / arm | Width | Steps | Heldout transition | Heldout rollout | Invalid rollout | Parameters | Train seconds | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| EXP-100 width9 control | 9 | 400 | 91.77% | 56.80% | 7.22% | 75,469 | 32.3 | valid control |
| EXP-100 width16 random init | 16 | 400 | 64.31% | 0.52% | 54.82% | 85,556 | 41.2 | inconclusive-revise |
| EXP-105 width16 transfer | 16 | 400 | 87.25% | 24.12% | 11.18% | 85,556 | 53.3 | pass -> matched 1,200 |
| EXP-106 width16 random | 16 | 1,200 | 87.76% | 23.48% | 7.30% | 85,556 | 149.7 | control |
| EXP-106 width16 transfer | 16 | 1,200 | 94.335% | 47.07% | 2.90% | 85,556 | 148.0 | winner; revise curriculum |
| EXP-110 transfer + hard mining | 16 | 1,200 | 99.095% | 87.41% | 0.72% | 85,556 | 146.9 | pass -> 2,400 |
| EXP-111 transfer + hard mining | 16 | 2,400 | 99.908% | 98.895% | 0.145% | 85,556 | 215.4 | pass -> independent |
| EXP-112 five official-generator seeds | 16 | frozen | n/a | 99.80% Tier 3 | 0% observed | 85,556 | 0.7 inference | pass -> integration |
