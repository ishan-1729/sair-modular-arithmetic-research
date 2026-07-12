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
| `candidate-003-tier4-cellular-hybrid` | promoted | 40.4% | 100% | 98% | 99% (Tier 4: 95%) | 4 | 270,401 | 7.9 s official sandbox | superseded |
| `candidate-004-tier5-cellular-hybrid` | promoted | 49.5% | 100% | 96% | 99% (Tier 4: 95%; Tier 5: 95%) | 5 | 370,225 | 36.9 s official sandbox | incumbent |

Candidate 004's Tier-5 estimate is 480/500 (96.0%) across five independent generator seeds plus 95/100 on the retained full independent seed. Candidate comparisons must report all scored tiers and preserve prior rows after promotion.

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
| EXP-315 cellular five-seed Tier4 | 32 | frozen | n/a | 96.6% | 0% observed | 99,824 | 11.4 wall | pass -> integration |
| EXP-320 cellular Tier5 curriculum | 32-56 -> 64 | 800 | 99.8% width56 / 99.68% width64 | 89.0% / 82.2% | 0.4% / 0.2% | 99,824 | 306.3 train | pass -> direct width64 |
| EXP-321 cellular direct width64 | 56/64 | no updates | invalid | invalid | invalid | 99,824 | stopped by projection | invalid resource projection |
| EXP-322 cellular width64 smoke | 56/64 | 400 | 99.68% | 83.0% | 0.667% | 99,824 | 207.6 train | reject; -1.0 rollout |
| EXP-323 diverse cellular width64 | 56/64 | 400 | 99.78% | 88.533% | 0.533% | 99,824 | 311.4 train | narrow reject; +3.333 rollout |
| EXP-324 cellular width64 stage2 | 56/64 | +400 | 99.88% | 91.867% | 0.467% | 99,824 | 284.6 train | narrow reject; soup next |
| EXP-325 cellular width64 soup | 64 | frozen | invalid | invalid | invalid | 99,824 | stopped by projection | invalid resource projection |
| EXP-326 cellular width64 soup retry | 64 | frozen | 99.9% | 91.2% | 0.2% | 99,824 | 593.2 analysis | reject; +1.0 over endpoint |
| EXP-327 cellular width64 logit ensemble | 64 | frozen | invalid | invalid | invalid | 199,648 | procedurally interrupted | no metrics; budget-policy error |
| EXP-328 cellular width64 logit ensemble retry | 64 | frozen | 100% | 90.8% | 0% | 199,648 | 635-674 wall / 40.4 CPU min | reject; +1.0, below 93% |
| EXP-329 teacher-forced control | 56/64 | +400 | 99.86% width64 | 93.133% width64 | 0.267% | 99,824 | 1,293 wall arm | control |
| EXP-329 autonomous-state recovery | 56/64 | +400 | 99.92% width64 | 94.267% width64 | 0.133% | 99,824 | 452 wall arm | narrow reject; +1.133 < +1.5 |
| EXP-400 cellular control | 8/12/16 -> 32 | 200 | 37.48% / 44.74% | 0% / 0% | n/a | 99,824 | 84.2 train | reference control |
| EXP-400 minimal GCN/GAT/MPNN | 8/12/16 -> 32 | 200 | 16.6-16.9% / 24.4-26.7% | 0% / 0% | n/a | 72,828-94,951 | 80.7-138.7 train | reject reference operators; PyG next |
| EXP-401 cellular control | 8/12/16 -> 32 | 1,200 | 72.92% / 70.14% | 0.8% / 0% | n/a | 99,824 | 390.5 train | matched control |
| EXP-401 PyG GCN/GATv2/max-MPNN | 8/12/16 -> 32 | 1,200 | 18.24-38.40% / 25.92-44.80% | 0% / 0% | n/a | 72,484-95,684 | 185.0-193.8 train | reject proper library formulations |
| EXP-500 cellular no-decay grokking control | width9 finite | 8,000 x3 | 100% train / 82.959% heldout mean | 30.667% mean | n/a | 99,824 | 0 signatures | reject delayed-generalization signature |
| EXP-500 cellular decay 1e-2 | width9 finite | 8,000 x3 | 99.837% train / 83.586% heldout mean | 32.467% mean | n/a | 99,824 | 0 signatures | reject; inconsistent paired effect |
| EXP-510 shared-context TCN grokking | width9 finite | 8,000 x3 | 100% train / 77.759% heldout mean | 23.067% mean | n/a | 75,469 | 0 signatures | reject memorization plateau |
| EXP-510 PyG max-MPNN grokking | width9 finite | 8,000 x3 | 99.512% train / 83.748% heldout mean | 30.533% mean | n/a | 72,484 | 0 signatures | reject concurrent learning, no delay |
| EXP-520 Tier5 recovery patches | width64 | frozen | 99.7% | 94.0-96.4% | 0-0.2% | 99,824 | 425.1 wall total | reject subsets; full endpoint best |
| EXP-521 Tier5 recovery five-seed | width64 | frozen | n/a | 96.0% (min 94%) | 0% observed | 99,824 | 45.7 wall / 179.3 CPU s | pass -> integration |
| EXP-600 Tier6 progressive curriculum | 64/80/96/112 -> 128 | 200 | 99.7% width128 | 57% source / 49% challenger | 1% / 2% | 99,824 | 297.3 wall / 19.3 CPU min | reject; transition training regresses rollout |
| EXP-601 Tier6 state recovery | 64/128 | 200 | 99.8% width128 | 62% source / 76% challenger | 0% challenger | 99,824 | 356.5 wall / 23.2 CPU min | pass -> full-budget confirmation |
| EXP-602 Tier6 recovery full budget | 64/128 | +400 | 99.83% width128 | 70.67% source / 79.33% challenger | 0.33% challenger | 99,824 | 1087.1 wall / 71.7 CPU min | reject independent; one continuation |
| EXP-603 Tier6 recovery continuation | 64/128 | +600 | 99.97% width128 | 84.0% source / 84.67% challenger | 0.67% challenger | 99,824 | 1268.7 wall / 83.8 CPU min | reject; close four-step continuation |
| EXP-604 recovery horizon A/B | 64/128 | 200 x2 | 99.70% / 99.75% | 84.0% four-step / 86.5% eight-step | 1% both | 99,824 | 760.6 wall / 50.0 CPU min | reject promotion; horizon supported |
| EXP-605 long-horizon retention | 64/128 | +400 | 99.87% width128 | 82.67% source / 82.33% challenger | 0.33% | 99,824 | 791.0 wall / 52.2 CPU min | reject; close schedule tuning |
| EXP-606 checkpoint combination | 64/128 | frozen | 100% fixed arms | 82% endpoints / 83% soup+ensemble / 84.5% oracle | 0% width128 | 99,824-199,648 | 657.9 wall / 43.1 CPU min | reject; shared errors dominate |
| EXP-607 differentiable soft state | 64/128 | 200 x2 | 99.95% both | 88.0% detached / 89.5% soft | 0% | 99,824 | 936.4 wall / 61.6 CPU min | reject gain; test straight-through |
| EXP-608 straight-through state | 64/128 | 200 x2 | 99.9% both | 84.5% soft / 87.0% straight-through | 0% | 99,824 | 787.6 wall / 51.8 CPU min | causal pass; absolute miss |
