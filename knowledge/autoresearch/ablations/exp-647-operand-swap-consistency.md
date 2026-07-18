---
type: Experiment Preregistration
title: EXP-647 operand-swap endpoint consistency
description: Matched full-rollout test of confidence-routed commutative consistency on the frozen Candidate-005 cellular lineage.
status: calibration-pending
tags: [autoresearch, tier7, commutativity, consistency, distillation, full-rollout]
timestamp: "2026-07-18T08:25:00+05:30"
---

# EXP-647 operand-swap endpoint consistency

## Question and evidence basis

Candidate 005 is asymmetric internally even though multiplication is commutative: one operand supplies the serial bits and the other is the multiplicand feature. On 500 fresh width-128 cases, its frozen original orientation scores 84.2%, while the fixed 40% conditional commutative route scores 97.2%. EXP-643 then finds 99.7% width-256 transition accuracy but only 22–32% autonomous rollout. The strongest supported training hypothesis is therefore to distill the useful commuted trajectory into one orientation, not to add another inference pass or another unproven latent executor.

The falsifiable claim is that a target-free confidence-consistency term recovers at least half of the confirmed thirteen-point conditional-route gain in the original single orientation and improves width-256 rollout. This is a research gate, not an assumption that intermediate states from the two orientations should match: only their final product distributions have the same semantic target.

## Frozen matched arms and changed variable

Both arms start from the exact 99,824-parameter EXP-609 checkpoint at SHA-256 `cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd`. They receive identical generated prime identities, ordered pairs, width schedule, optimizer updates, supervised prefix losses, straight-through hard autonomous states, truncated-gradient boundaries, learning rate, and evaluation inputs.

- `paired_rollout_control` runs complete `(a,b,p)` and `(b,a,p)` autonomous training trajectories and applies the ordinary exact prefix supervision to both.
- `confidence_consistency` performs the identical work and adds one changed term at the common final endpoint. For each pair, the orientation with the larger minimum absolute bit-logit margin over its complete trajectory is a stop-gradient teacher; the lower-margin final logits receive soft binary cross-entropy toward that teacher. Ties retain the original orientation. The fixed coefficient is `0.10`.

The confidence rule is imported unchanged from EXP-640–642, where it selected complementary answers without labels. Exact answers do not choose the teacher. The ground-truth prefix labels supervise both arms equally and are allowed offline training evidence.

## Training and evaluation

Use seed `20261057` and fresh disjoint 512/128 train/test prime identities from the top eight bits of widths 64, 128, and 256. Train 240 updates with the frozen repeating schedule `[64,128,256,128]`, batch size four, AdamW from `5e-6` to `5e-7`, and sixteen-step truncated backpropagation through time. Hard thresholded states are used forward; sigmoid straight-through gradients operate only inside each truncation block. Every update scans both complete operand orientations, so the two arms remain matched in examples and forward/backward work.

For each frozen arm, evaluate 1,000/3,000/3,000 fresh transitions and 300/300/200 paired original/swapped rollouts at widths 64/128/256. Report original, swapped, fixed higher-margin selected, and offline union exactness; original/swapped agreement and complementarity; all-prefix exactness; first errors; invalid residues; direct per-orientation wall and aggregate CPU; parameters; checkpoint hashes; and learning curves. Repeat the first twenty width-128 original trajectories exactly. Checked, prior independent-full, official, and private answers remain unread.

## Frozen gates

Advance only if all of the following pass:

- original-orientation width-128 exact rollout is at least 90% and gains at least 6.5 points over the matched control, half of the independently confirmed thirteen-point route gain;
- original-orientation width-256 exact rollout is at least 45% and gains at least five points over control;
- original width-64 retention is at least 95%;
- width-128/256 transition exactness is at least 99.8%/99.7%;
- every challenger original rollout invalid rate is at most 1%;
- the repeated predictions, margins, and invalid flags are bit-identical; and
- direct width-256 single-orientation throughput projects at most 100 wall seconds for 100 cases, leaving approximately 44 seconds beyond Candidate 005's measured 156-second artifact runtime and the 300-second hard limit.

The runtime gate is deliberately strict. Accuracy passing without runtime is recorded as an accuracy-supported/runtime-blocked result that may justify an exact-output inference optimization experiment, not candidate promotion. Either accuracy-gain miss rejects this exact consistency coefficient and formulation without a post-result sweep. If the matched control independently reaches a materially stronger Tier-7 surface, preserve that evidence and preregister its confirmation separately rather than crediting the consistency term.

## Compliance and resource protocol

Inference remains the unchanged trained cellular transition and ordinary bit scan. Operand swapping is a generic task symmetry; it does not implement multiplication, reduction, division, CRT, output repair, or a lookup. Exact Python bigint arithmetic is confined to offline identities, prefix labels, and scoring. No library beyond the maintained PyTorch/safetensors/MLflow stack is needed for this loss-only A/B; the external-library architecture queue remains open for models that need it.

Before formal execution, profile one real full training update at each width for each arm, including both orientations, real batch size, truncation, backward, and optimizer. Also profile the exact transition and paired-rollout evaluation path at all widths. Scale each measured component to all 480 formal arm updates and declared evaluation cases, add explicit setup/serialization allowance, apply 1.50 throughput headroom, and retain a further 1.50 hard runaway ceiling. Wall time and aggregate process CPU are separate limits. Commit the resolved budget before formal work; formal mode must reject a dirty tree, pending calibration, or hash mismatch. Save and hash each arm checkpoint before its long evaluation and cache completed training/evaluation by source, code, config, dataset, seed, and checkpoint hashes. No update, pair, width, orientation, or case may be reduced to fit a shorter wall cap.
