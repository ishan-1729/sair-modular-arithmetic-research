---
type: Experiment Preregistration
title: EXP-642 Fresh Conditional Commutative Confirmation
description: Fresh 500-rollout and 3,000-transition confirmation of the frozen 40 percent operand-swap policy.
status: completed
tags: [autoresearch, tier6, commutativity, conditional-compute, confirmation, fresh-evidence]
timestamp: "2026-07-17T15:05:00+05:30"
---

# EXP-642 fresh conditional commutative confirmation

## Confirmatory question

Does the exact EXP-641 policy generalize to a new width-128 prime split and fresh cases? The frozen 99,824-parameter EXP-609 checkpoint executes original inference for every row, routes exactly the lowest-original-minimum-margin 40% through operand-swapped inference, chooses the higher-margin orientation inside the routed subset, and retains original elsewhere. Route ties use lowest case index; orientation ties retain original. No weight, threshold, fraction, confidence function, or tie break changes from EXP-641.

Use seed `20260842`, split offset `64200 + 128`, transition offset `642700 + 128`, and pair offset `642800 + 128`, with 512/128 new train/test prime identities and eight-bit prime span. Evaluate 3,000 fresh transitions and 500 fresh rollouts; exactly 200 rollouts receive swapped inference. Repeat the complete policy twice on the first 20 cases. No consumed EXP-639/640/641, checked, independent-full, official, or private answer participates in selection.

## Controls, boundaries, and gates

Report original and conditional exact rollout, selected gain, fixed routed oracle, recoveries/losses, invalidity, transition exactness/validity, exact determinism, separate original/routed-swapped wall and CPU, and the same conservative integration formula. The offline oracle uses exact labels only to measure available coverage and never chooses inference outputs.

Advance only if all gates pass:

- fresh width-128 transition exactness is at least 99.8%;
- fresh routed-oracle rollout is at least 92%;
- fresh conditional selected rollout is at least 90%;
- selected gain over original is at least five points;
- selected invalid-trajectory rate is at most 1%;
- two complete 20-case conditional runs repeat predictions, margins, routes, choices, and validity exactly;
- Candidate 004's 36.9-second complete independent runtime plus measured conditional work scaled from 500 to 100 Tier-6 rows is at most 285 wall seconds.

Passing authorizes candidate-shaped Tier-6 integration and wider-width feasibility, not submission promotion. Full five-seed independent evaluation, checked release regression, official evaluation when available, static/security/interface/runtime/weight-dependence gates, and the binding 85% overall target remain mandatory.

Inference uses only learned tensor operations over legal state/operand/modulus/bit/marker features. Commutative swapping and target-free confidence routing do not implement modular multiplication, division, reduction, CRT, repair, or lookup. Exact Python bigint arithmetic is confined to offline data and evaluation truth. Native PyTorch remains the appropriate maintained library; graph operators are irrelevant to this frozen inference confirmation.

## Reused dynamic resource budget

EXP-641 measures the identical original-all/routed-swapped path on 300/120 cases, including margins and determinism, while EXP-639 measures the identical frozen width-128 transition arm on 3,000 cases. Scale conditional work to 500/200, retain the complete 3,000-transition measurement, retain the same determinism work, and add 60 seconds for setup, integrity, MLflow, and artifacts. This projects 990 evaluation wall seconds and 3,660 aggregate CPU seconds; expected total is 1,050 wall seconds and 63 aggregate CPU minutes.

The soft budget is 1,500 wall seconds. The hard runaway ceiling is 2,400 wall seconds with a separate four-thread upper bound of 160 aggregate CPU minutes. Source summary hashes are frozen in config. This is valid exact-path throughput reuse under the resource policy; no new calibration answer is needed. Save the complete fresh input/prediction/margin cache and never reduce cases or skip an arm to fit a shorter wall cap.

## Limitations

One fresh seed is confirmation, not the required five independent full seeds. The runtime projection remains conservative rather than an official isolated candidate measurement. Widths beyond 128 remain untested by this mechanism.

## Result and decision

Passed every fresh confirmation gate. The frozen policy scores 486/500 (97.2%) versus 421/500 (84.2%) original, a thirteen-point gain. It routes exactly 200 cases, chooses swapped 161 times, recovers 66 original failures, and loses one original success. Fixed routed oracle coverage is 488/500 (97.6%), selected invalidity is zero, and both complete conditional repeats match exactly. Fresh transition exactness is 2,998/3,000 (99.933%) with 100% valid residues.

Original-all and routed-swapped work measure 261.32 and 123.24 wall seconds. The frozen integration formula projects 113.81 seconds, comfortably below 285. Runner work is 438.95 wall seconds and 1,603.70 aggregate CPU seconds (26.73 minutes), below the reused 1,050-second/63-minute expectation, 1,500-second soft budget, 2,400-second hard ceiling, and 160-minute CPU upper bound. GNU time separately records 433.47 wall, 1,603.74 user, and 10.47 system seconds.

Authorize candidate-shaped Tier-6 integration with this exact checkpoint/policy and a separate wider-width feasibility rung. Do not call this candidate promotion: five independent full seeds, checked release regression, official evaluation when available, artifact/static/interface/security/runtime/weight-dependence controls, and at least 85% full-overall accuracy are all still required.
