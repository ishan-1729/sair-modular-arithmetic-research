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
| `candidate-002-tier3-hybrid` | promoted | 31.0% | 100% | 97% | 99% | 3 | 170,577 | 1.8 s official sandbox | superseded |
| `candidate-003-tier4-cellular-hybrid` | promoted | 40.4% | 100% | 98% | 99% (Tier 4: 95%) | 4 | 270,401 | 7.9 s official sandbox | superseded |
| `candidate-004-tier5-cellular-hybrid` | promoted | 49.5% | 100% | 96% | 99% (Tier 4: 95%; Tier 5: 95%) | 5 | 370,225 | 36.9 s official sandbox | superseded |
| `candidate-005-tier6-conditional-commutative-integration` | promoted | 59.3% | 100% | 99% | 97% (Tier 4: 97%; Tier 5: 94%; Tier 6: 98%) | 6 | 470,049 | 156.0 s official sandbox | incumbent |

Candidate 005's Tier-6 estimate is 487/500 (97.4%) across five preregistered generator seeds with a 95% minimum. Its checked overall accuracy is 59.5% and checked Tier 6 is 96%. Candidate comparisons must report all scored tiers and preserve prior rows after promotion.

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
| EXP-609 straight-through confirmation | 64/128 | +400 | 99.93% | 84.67% source / 86.33% challenger | 0% challenger | 99,824 | 831.0 wall / 54.7 CPU min | reject; close estimator continuation |
| EXP-610 state corruption | 64/128 | 200 x2 | 99.9% / 99.85% | 87.5% clean / 75.0% corrupted | 0% / 0.5% | 99,824 | 735.4 wall / 48.3 CPU min | reject; corruption collapses |
| EXP-611 residual widening | 64/128 | 200 x2 | 99.8% both | 87.0% control / 87.0% 92+36 | 0.5% both | 99,824 / 116,712 | 1071.1 wall / 68.8 CPU min | reject low-rate recipe; adapter dormant |
| EXP-612 adapter activation | 64/128 | 200 x2 | 99.85% both | 85.5% low-rate / 87.0% differential-rate | 0.5% both | 116,712 each | 1312.7 wall / 84.5 CPU min | reject and close residual widening; +1.5 |
| EXP-613 conditioned corrector | 64/128 | 300 x2 | 100% width128 both | 85.5% uniform / 77.5% first-error | 0.5% / 2% | 129,636 (29,812 trainable) | 1290.6 wall / 82.9 CPU min | reject and close; targeted correction −8.0 |
| EXP-614 global refinement | 64/128 | 200 x2 | 99.95% width128 both | 87.5% control / 86.5% refinement | 0% width128 | 99,824 / 181,520 | 1365.1 wall / 87.5 CPU min | reject; partial blend activation, −1.0 |
| EXP-615 casewise exactness loss | 64/128 | 200 x2 | 100% both | 85.0% control / 85.5% casewise | 0% | 99,824 each | 984.2 wall / 62.9 CPU min | reject; +0.5, close coefficient |
| EXP-616 trajectory-tail CVaR | 64/128 | 200 x2 | 99.65% width128 both | 86.5% control / 86.0% CVaR | 0% width128 | 99,824 each | 972.4 wall / 62.1 CPU min | reject; −0.5, close loss reweighting |
| EXP-617 recurrent-core LoRA | 64/128 | 200 x2 | 99.7% width128 both | 86.5% control / 87.0% LoRA | 0% | 99,824 / 106,736 (merged 99,824) | 927.3 wall / 59.1 CPU min | reject; +0.5, merge exact |
| EXP-618 continuous inference | 64/128 | frozen 500 paired | 99.92% width128 | 84.4% hard / 81.8% continuous | 0% final | 99,824 | 973.3 wall / 62.1 CPU min | reject; −2.6, hard-only 18 vs soft-only 5 |
| EXP-619 confidence state | 64/128 | 200 x2 | 99.95% / 99.9% width128 | 82.0% control / 78.0% confidence | 0% | 99,824 / 101,664 | 948.9 wall / 60.3 CPU min | reject and close confidence; −4.0 |
| EXP-620 persistent Neural GPU | 8/12/16 -> 24/32 | 1,200 x2 | prefix-supervised | 4.5% decoded / 0% persistent width16 | 20.5% / 0% width16 final | 41,242 / 40,577 | 708.2 wall / 44.6 CPU min | reject viability; persistent zero exact |
| EXP-621 learned beam decoding | 64/128 | frozen 200 paired | 99.76% width128 | 82.5% greedy / 82.0% selected beam / 86.0% oracle beam | 1% / 0% | 99,824 + decoder | 634.4 wall / 39.3 CPU min | reject ranking; preserve coverage signal |
| EXP-622 width-4 beam coverage | 128 | frozen 200 paired | cached 99.76% width128 | 89.0% greedy / 86.5% selected beam / 94.0% oracle beam | 0% / 0% | 99,824 + decoder | 948.6 wall / 59.7 CPU min | reject runtime; preserve reranker surface |
| EXP-623 staged width-2-to-4 beam | 128 | frozen 200 paired | cached 99.76% width128 | 80.0% greedy / 80.5% selected beam / 84.0% oracle beam | 0.5% / 0% | 99,824 + decoder | 559.4 wall / 34.9 CPU min | reject coverage; close late widening |
| EXP-624 width-3 beam coverage | 128 | frozen 200 paired | cached 99.76% width128 | 82.5% greedy / 80.0% selected beam / 90.0% oracle beam | 0.5% / 0% | 99,824 + decoder | 530.9 wall / 33.1 CPU min | reject coverage; close unconditional beam |
| EXP-625 confidence-routed width-4 diagnostic | 128 | frozen 200 paired, 40% route | cached 99.76% width128 | 86.0% greedy / 91.0% primary route / 92.5% secondary min-margin and oracle | 0% | 99,824 + decoder | 892.5 wall / 55.7 CPU min | reject primary; confirm frozen min-margin |
| EXP-626 minimum-margin routing confirmation | 128 | frozen 200 paired, 40% route | cached 99.76% width128 | 86.5% greedy / 92.5% min-margin allocation and oracle | 0% | 99,824 + decoder | 865.8 wall / 54.2 CPU min | pass research gate -> trained reranker |
| EXP-627 trained trajectory-feature reranker | 128 | 1,200 train / 200 val / 200 selection | cached 99.76% width128 | 81.0% greedy / 81.5% likelihood / 81.0% trained / 87.5% oracle | 0% | 99,824 + 5,570 | 5,423.6 cumulative wall / 340.8 CPU min | reject; oracle and weight gates fail |
| EXP-628 top-two branch calibration | 128 | consumed EXP-627 selection cache, 40% route | cached 99.76% width128 | 81.0% greedy / 87.5% one-bit oracle / 88.5% top-two oracle | 0% observed | 99,824 + decoder | 418.86 vs 529.02 matched beam wall (contaminated absolute) | calibration reject; +1.0 overall, no fresh run |
| EXP-629 balanced first-error replay | 64/128 | 300 x2, 25% replay in width128 batches | 99.8% control / 99.767% hard width128 | 87.67% control / 86.33% hard width128 | 1.0% / 0.33% | 99,824 each | 5,193.6 runner wall / 324.0 CPU min | reject; −1.33 width128, close replay |
| EXP-630 radix-4 cellular | 8-32 -> 64 | 1,600 | 37.5% / 22.9% / 13.95% at 16/32/64 | 0% at 16/32/64 | 19.5% / 35.5% / 52% | 101,116 | 1,272.3 wall / 77.9 CPU min; 0.496 width64 time ratio | reject viability; speed passes, exactness fails |
| EXP-632 shared-core diverse heads | 64/128 | 300 head-only updates | 99.9% challenger width128 | 83.67% control / 82.00% challenger width128 | 0.33% both | 100,103 (279 trainable) | 2,019.7 runner wall / 124.8 CPU min; 207.2 s integrated projection | reject; -1.67 width128, trained 1 point below randomized control |
| EXP-633 hierarchical composition tree | 8/16 -> 32 | 2,400 x2 | direct root prediction | 86.67% control / 0% tree width16; 1% / 0% width32 | 2%/0% width16; 15.67%/0% width32 | 99,824 / 94,025 | 4,044.2 runner wall / 252.5 CPU min; 109.6 s integrated projection | reject; active internal nodes never compose to one exact root |
| EXP-634 direct relative attention | 8/12 -> 24 | 4,000 x2 | direct product prediction | 0.667% control / 0.667% attention width12; 0% / 0% width24 | 100% trained validity; randomized attention 100% invalid width24 | 52,345 / 53,122 | 940.6 runner wall / 57.9 CPU min; 37.7 s integrated projection | reject; no exactness gain, transfer, or trained-weight dependence |
| EXP-635 independent cellular diversity | 8/16 | 2,400 challenger; frozen control | n/a | 86.33% control / 86.67% challenger / 93.00% target-free routed / 95.33% oracle | 0.67% / 1.33% invalid; randomized 0% exact | 99,824 x2 | 1,705.6 runner wall / 105.9 CPU min; 13.68 s paired eval | pass feasibility -> Tier-6 independent lineage |
| EXP-636 independent Tier-5 bootstrap | 16/24/32/40/48/56/64 | 3,200 staged | 99.367% independent width64 | 94.00% incumbent / 68.83% independent / 95.00% routed / 96.00% oracle | 0.33% / 1.33% invalid | 99,824 x2 | ~3,145 reconstructed active wall / ~200.8 CPU min; 442.7 s paired eval | reject; diversity shrinks to +1 routed point and core accuracy gates fail |
| EXP-637 teacher-distilled macro | 8-32 -> 64 | 1,600 | 44.5% / 25.9% / 10.55% final; 97.25% / 93.2% / 81.95% intermediate | 0.5% / 0% / 0% at widths 16/32/64 | 18.5% / 32.5% / 50.5% invalid | 199,648 | 1,672.6 runner wall / 103.3 CPU min; 0.485 width64 time ratio | reject; one-sweep final phase bottlenecks despite distilled intermediate |
| EXP-638 full-depth soft boundary | 16/32/64 training identities | frozen diagnostic | 99.65% / 99.8% / 99.9% continuous transition | 95.0% hard / 95.5% continuous width64 | 0% invalid | 99,824 shared | 182.1 runner wall / 10.7 CPU min; 0.916 continuous/hard ratio | pass feasibility -> separate width128 adaptation rung |
| EXP-639 soft-boundary adaptation | 64/128 | 400 shared-weight updates | 99.733% trained width128 | 82.67% hard / 81.67% initialized / 80.67% trained width128; 96.0% trained width64 | 0% trained width128 invalid | 99,824 shared | 2,312.4 runner wall / 142.2 CPU min; 1.509 trained/hard width128 ratio | reject; -2.0 vs hard, close adaptation |
| EXP-640 commutative operand swap | 128 consumed | frozen 300 original + 300 swapped | n/a | 82.67% original / 85.00% swapped / 97.00% fixed selected / 98.33% oracle | 0% selected invalid | 99,824 shared | 959.9 runner wall / 59.0 CPU min; 318.62 s integrated projection | reject unconditional runtime; conditional 40% route next |
| EXP-641 conditional commutative swap | 128 consumed | frozen 300 original + 120 routed swapped | cached 99.633% width128 | 82.67% original / 96.67% selected / 97.67% routed oracle | 0% selected invalid | 99,824 shared | 593.9 runner wall / 36.4 CPU min; 212.66 s integrated projection | pass research gate -> 500-case fresh confirmation |
| EXP-642 fresh conditional confirmation | 128 fresh | 3,000 transitions; 500 original + 200 routed swapped | 99.933% | 84.2% original / 97.2% selected / 97.6% routed oracle | 0% selected invalid | 99,824 shared | 438.9 runner wall / 26.7 CPU min; 113.81 s integrated projection | pass -> candidate-shaped Tier6 integration and wider feasibility |
| EXP-643 recurrent-depth stability | 128/256 | frozen 2 seeds; 2,000 transitions + 100 rollouts per width/seed | 99.95%/99.90% width128; 99.75%/99.70% width256 | 85%/81% width128; 22%/32% width256 | 0% overthinking; 0-2% normal-rollout invalid | 99,824 shared | 1,523.3 wall / 99.7 CPU min; 605 MiB peak | reject stability cause; move to addressable memory |
| EXP-654 neural-RAM viability | 8/12/16 -> 24/32 | 4,000 x4 | 93.67%/97.35% decoded width16; 27.40%/23.98% latent width16 prefix transition | 85.33% small decoded / 94.33% wider decoded / 0% non-addressable / 0% neural RAM width16; all 0% width24/32 | 1.0% wider decoded; 4.67% non-addressable; 5.67% neural RAM width16 | 41,242 / 76,244 / 74,801 / 74,865 | 5,220.2 wall / 345.3 CPU min; 1.10 GiB peak | reject current memory formulation; exactness and causal-memory gates fail |
| EXP-647 operand-swap consistency | 64/128/256 | 240 x2 | 99.900% / 99.733% / 99.567% challenger | original control 96.000% / 78.000% / 27.500%; challenger 96.333% / 76.333% / 25.000% | 0.333% / 0.333% / 1.500% challenger original | 99,824 each | retry3 5,141.6 wall / 341.1 CPU min; cumulative charged 20,337.5 wall / 1,596.3 CPU min | reject; only width64 retention and determinism pass |
| EXP-649 stochastic breadth calibration | 256 | frozen; 1 full warmup + 3 full replicas | n/a - input-only calibration | n/a - no labels, oracle, or formal cases | n/a - formal validity untested | 99,824 raw cell + Candidate 005 artifact | replicas 2,131.6–2,621.6 wall / 135.5–177.0 process-CPU min; 3,150.9 s upper | runtime-gate-miss; planned/formal-blocked, no breadth verdict |
| EXP-659 symbol-equivariant recurrence | 8/12/16 planned -> 24/32 | calibration only; formal not run | n/a - no formal scientific metrics | n/a - no formal rollout evaluation | n/a - no formal validity result | matched three-arm package | 18.398 wall / 47.20 process CPU / 66.292 cgroup CPU s calibration; 435 wall / 1,552 process CPU s projected formal | parked before formal at T046 verification boundary; mechanism not scientifically rejected |
| EXP-660 symbol-equivariant formal successor | 8/12/16 planned -> 24/32 | no calibration or formal training; seed 20261177 unopened | n/a - no scientific metrics | n/a - no rollout evaluation | n/a - no validity result | frozen matched three-arm package | focused 54/54 in 190.78 s; validator 163/1100 green; full oracle 367 passed / 6 EXP658 failures in 498.41 pytest s (467.75 timed wall) | parked before science under exhausted T047 recovery; mechanism not scientifically rejected |
| EXP-650 library-backed Mamba-2 transition | 16 planned -> 16/24/32 | no calibration/formal training; roots 20261186-20261189 unopened | n/a - no scientific metrics | n/a - no rollout evaluation | n/a - no validity result | 23,601 hybrid / 24,049 ConvGRU / 22,517 attention | CPU smoke forward/backward 0.121 s; focused 7/7; validator 165/1100 green; full oracle 371 passed / 34 inherited failures in 155.45 pytest s | parked before science at T053 package boundary; Mamba-2 mechanism not scientifically rejected |
| EXP-655 layer-loop compute-matched recurrence | 8/12/16 -> 16/32 | 3 roots × 4 arms × 4,000 updates; 300 cases/width/arm/root | primary aggregate prefix transition 24.60%/28.32% W16 and 9.38%/9.36% W32 layer/model | layer/model W16 exact by root 0/1.0%, 0/0.667%, 0.333/0.333%; both 0% W32 throughout | max invalid 88.33%; minimum trained-minus-randomized drop 0 | 17,572 each primary/R1; 34,532 untied | 1,338.64 wall / 5,257.54 process CPU / 6,545.54 cgroup CPU s; 496,414,720-byte RSS | completed-rejected; integrity/resources pass, viability/validity/dependence/schedule-effect gates fail |
