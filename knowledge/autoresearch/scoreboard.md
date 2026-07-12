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
| EXP-200 width32 random + hard mining | 32 | 400 | 44.12% | 0.00% | 38.88% | 126,276 | 44.1 | control |
| EXP-200 width32 transfer + hard mining | 32 | 400 | 98.21% | 61.68% | 2.12% | 126,276 | 44.7 | pass -> full budget |
| EXP-201 width32 transfer + hard mining | 32 | 1,600 | 99.648% | 90.465% | 0.51% | 126,276 | 169.2 | narrow miss -> 2,400 |
| EXP-202 width32 transfer + hard mining | 32 | 2,400 | 99.814% | 94.49% | 0.15% | 126,276 | 395.3 | pass -> independent |
| EXP-203 five official-generator Tier-4 seeds | 32 | frozen | n/a | 88.20% Tier 4 | 0% observed | 126,276 | 4.0 wall | fail; retain incumbent |
| EXP-204 official-distribution fine-tune | 32 | +800 | 99.666% | 89.97% | 0.30% | 126,276 | 82.8 | reject; rollout-aware next |
| EXP-210 one-step continuation | 32 | +400 | 99.794% | 90.72% | n/a | 126,276 | 63.4 | control |
| EXP-210 scheduled learned-state | 32 | +400 | 99.828% | 91.425% | 0.175% | 126,276 | 113.1 | reject; +0.705 rollout |
| EXP-220 full-canvas random control | 16 | 800 | 61.94% | 0.17% | 11.61% | 85,556 | 52.8 | control |
| EXP-220 shared dilated ConvGRU | 16 -> 32 zero-shot | 800 | 60.245% / 61.935% | 0.11% / 0% | 21.28% / 68.7% | 99,824 | 244.2 | reject |
| EXP-230 single-head continuation | 32 | +800 | n/a | 91.98% | n/a | 126,276 | 84.7 | control |
| EXP-230 learned quotient-routed experts | 32 | +800 | 99.786% | 92.17% | 0.265% | 224,742 | 96.6 | reject; +0.19 rollout |
| EXP-240 paired channel ablations | 32 | frozen | 99.77% | 91.98% contextual | n/a | 126,276 | 69.9 analysis | reject narrow surgery; distributed |
| EXP-250 online failure mining | 32 | 400 control only | invalid | invalid | invalid | 126,276 | stopped at 10 min | killed; resource cap |
| EXP-251 online mining smoke | 32 | +100 | 99.77% | 92.86% | 0.22% | 126,276 | 47.6 train | reject; +0.96 rollout |
| EXP-252 replay-heavy mining | 32 | +200 | 99.815% | 93.62% | 0.14% | 126,276 | 25.5 train | reject; +0.36 rollout |
| EXP-260 low-rate continuation | 32 | +600 | 99.826% | 92.795% | n/a | 126,276 | 43.2 train | reject; +0.63 rollout |
| EXP-300 width64 random / transfer | 64 | 400 | 22.48% / 82.38% | 0% / 0% | 23.85% / 17.15% | 281,444 | 44.4 / 45.2 train | reject smoke; transfer +59.9 transition |
| EXP-301 width64 hard mining | 64 | +1,200 | 93.53% | 2.30% | 5.94% | 281,444 | 142.7 train | reject; close fixed-width width64 |
| EXP-310 cellular mixed-width control | 8/12/16 -> 32 | 1,200 | 91.19% / 75.115% zero-shot | 28.20% / 0.20% | 9.48% / 52.12% | 99,824 | 163.3 train | advance curriculum intervention |
| EXP-310 chunk-GRU challenger | 8/12/16 -> 32 | 1,200 | 38.335% / 44.435% | 0% / 0% | 35.4% / 89.9% | 72,903 | 27.1 train | reject architecture |
| EXP-311 expanded cellular | 8-24 -> 32 | 1,600 | invalid | invalid | invalid | 99,824 | stopped at 10 min | killed during evaluation |
| EXP-312 expanded cellular retry | 8-24 -> 32 | 1,600 | 99.88% width24 / 99.4% width32 | 96.2% / 85.0% | 0.10% / 0.15% | 99,824 | 464.0 train | pass -> direct width32 |
| EXP-313 cellular direct width32 | 24/32 | 100 partial | invalid | invalid | invalid | 99,824 | stopped by projection | killed; resource projection |
| EXP-314 cellular width32 smoke | 24/32 | 600 | 99.91% | 94.567% | 0.30% | 99,824 | 249.0 train | pass -> independent Tier4 |
