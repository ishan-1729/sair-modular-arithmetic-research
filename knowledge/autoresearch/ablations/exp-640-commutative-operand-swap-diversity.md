---
type: Experiment Preregistration
title: EXP-640 Commutative Operand-Swap Diversity
description: Consumed-evidence feasibility test of original and operand-swapped frozen learned trajectories with one target-free selector.
status: preregistered
tags: [autoresearch, tier6, commutativity, symmetry, ensemble, confidence-routing]
timestamp: "2026-07-17T13:45:00+05:30"
---

# EXP-640 commutative operand-swap diversity

## Question and changed variable

Modular multiplication is commutative, but the learned Horner inference path assigns asymmetric roles: one operand supplies the serial multiplier bits and the other is the repeated multiplicand feature. EXP-640 tests whether evaluating the same frozen 99,824-parameter EXP-609 cell on both `(a, b, p)` and `(b, a, p)` exposes complementary exact answers. This is not a second model and changes no weight, transition, state update, or arithmetic routine.

The only realizable selector is frozen before results: choose the orientation with the larger minimum absolute bit-logit margin over its complete 128-step trajectory; exact ties choose the original orientation. The offline oracle union is a coverage diagnostic, never an inference selector. Report original-only and swapped-only exact cases, agreement, selector recoveries/losses, invalid trajectories, separate wall/aggregate CPU work, and exact repeatability.

## Evidence hygiene and controls

This first rung reconstructs exactly the 300 already-consumed EXP-639 width-128 pairs: source seed `20260839`, split offset `63900 + 128`, pair offset `639800 + 128`, 512/128 train/test prime identities, and eight-bit prime span. The original orientation must reproduce EXP-639's hard-frozen 248/300 exact result or the run aborts without interpretation. No new held-out, checked, independent-full, official, or private selection answer is read.

The checkpoint SHA-256 remains `cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd`. Inference receives only legal operand/modulus/state/bit/marker tensors. Operand swapping is a generic task symmetry and does not encode a carry, comparison, reduction, expected answer, lookup, repair, CRT, division, or modular-multiplication implementation. Exact Python arithmetic is confined to offline reconstruction and scoring. Native PyTorch is the suitable maintained library; graph operators are unrelated to this frozen trajectory comparison, and the required proper-PyG GCN/GATv2/MPNN track is already recorded separately.

## Gates

Advance only if every gate passes on the consumed surface:

- offline original/swapped oracle union is at least 92%;
- the fixed higher-minimum-margin selector reaches at least 90% exact rollout;
- selected invalid-trajectory rate is at most 1%;
- original predictions reproduce 248/300 exactly;
- predictions, margins, chosen orientations, and validity repeat exactly on the first 20 cases;
- a conservative integrated projection—Candidate 004's complete 36.9-second independent runtime plus both measured orientations scaled to 100 Tier-6 cases—is at most 285 wall seconds.

Passing authorizes one fresh generated confirmation with the identical selector. It is not candidate promotion and supplies no evidence toward the five independent full seeds or checked/official goal. Failure closes unconditional two-orientation commutative inference with this fixed selector; selector thresholds or confidence functions may not be tuned on these outcomes.

## Dynamic resource-budget protocol

Before formal execution, profile both exact orientation paths, including margin collection, on 12 cases drawn only from EXP-639's training prime identities. Scale per-orientation wall and aggregate CPU throughput to 300 original cases, 300 swapped cases, and four 20-case orientation passes for the two determinism repeats. Add 60 seconds for setup, hashing, MLflow, and artifacts; apply documented soft headroom and a separate hard runaway ceiling. Calibration may change only measured/projection/limit fields in `resource_budget` and the measured paragraph below. It may not change evidence identities, cases, checkpoint, selector, gates, determinism work, or runtime projection.

Formal execution is forbidden while calibration is pending. Preserve the full paired prediction/confidence cache by config, code, checkpoint, split, and pair-seed hash. Never reduce cases or skip an orientation to meet an arbitrary wall cap; distinguish wall from aggregate CPU throughout.

The completed training-identity-only calibration ran 12 cases per orientation through the exact formal margin-collecting path. Original and swapped passes measured 18.178/67.809 and 19.401/72.595 wall/aggregate-CPU seconds. Scaling to the exact 680 orientation-equivalent formal and determinism cases projects 1,064.764 evaluation wall seconds and 3,978.464 aggregate CPU seconds. Adding 60 seconds for setup and artifacts gives 1,125 expected wall seconds and 69 aggregate CPU minutes. The soft budget is 1,620 wall seconds; the hard runaway ceiling is 2,400 wall seconds with a distinct four-thread upper bound of 160 aggregate CPU minutes.

External calibration timing was 57.31 wall seconds, 172.17 user CPU seconds, and 1.98 system CPU seconds at 303% CPU, with 392,076 KiB maximum RSS. Calibration JSON and console hashes are `30d86c298e5ba48d89e3a881e1a4cf6b15c719bd7370521765075f647ee14428` and `44555b66953359ed9539209793bc8c545472281e8a0eaa9d77a24faaf455a2ed`. No consumed test or new held-out identity was read, and no scientific field changed.

## Limitations

This diagnostic pays for both trajectories and may be limited by shared errors. Minimum margin was supported as a target-free uncertainty signal in EXP-626, but selecting between two complete predictions is a different task. The consumed surface can reject or authorize fresh confirmation only; it cannot establish generalization.
