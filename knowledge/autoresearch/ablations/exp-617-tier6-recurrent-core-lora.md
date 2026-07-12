---
type: Experiment Preregistration
title: EXP-617 Tier-6 Recurrent-Core LoRA
description: Matched test of exact-initialized mergeable low-rank recurrent-kernel adaptation against full-base continuation.
status: preregistered
tags: [autoresearch, tier6, lora, recurrent-core, straight-through]
timestamp: "2026-07-13T05:00:00+05:30"
---

# EXP-617 Tier-6 recurrent-core LoRA

## Question and hypothesis

Output adapters, added global passes, and loss reweighting have failed. Earlier causal patching found the useful Tier-5 update distributed across the recurrent core rather than localized to the quotient or output heads. Can exact-initialized low-rank updates to the gate and candidate convolution kernels adapt that distributed computation at every dilation step without perturbing the source initially?

The challenger freezes the complete trained base and adds rank-8 matrices to the flattened gate and candidate kernels. Output factors initialize to zero, making both effective kernels and all outputs exactly equal to the source. Only 6,912 parameters train. After training, the deltas are folded into ordinary gate and candidate tensors, returning to the original 99,824-parameter architecture with no adapter inference path.

## Matched comparison

- Control: all 99,824 base parameters continue at `5e-6`.
- Challenger: 6,912 recurrent LoRA parameters train at `1e-3` decaying to `1e-4`; the 99,824 base parameters remain frozen.
- Both use the same EXP-609 source, seed, fresh identities, examples, 200 updates, optimizer family, weight decay, batches, mode schedule, eight-step straight-through recovery, clipping, and evaluation cases.

Trainable parameterization and learning rate are intentionally changed by the intervention. Before training, 1,000 cases at each width must reproduce source local and quotient logits bit-for-bit. The merged checkpoint must retain identical bit and quotient predictions to the adapter wrapper on 256 representative cases per width and report maximum logit differences.

## Gates

Each arm is evaluated on 2,000 transitions and 200 rollouts at widths 64 and 128. Advance only if the challenger reaches at least 90% width-128 rollout, gains at least three points over control, retains at least 95% width-64 rollout, reaches at least 99.8% width-128 transition, keeps invalid width-128 rollouts at or below 1%, and learns a gate-kernel delta with RMS at least `1e-4`.

A pass authorizes higher-fidelity confirmation using the merged checkpoint. It does not authorize checked or independent evaluation yet. A miss closes rank-8 recurrent-core adaptation; do not sweep rank or alpha on this selection set.

## Resource budget

Ten representative eight-step updates took 20.83 wall / 80.70 aggregate CPU seconds for full-base continuation and 19.77 wall / 76.60 aggregate CPU seconds for LoRA. The frozen base makes LoRA backward slightly cheaper. Two 200-update arms project to 812 wall seconds; full evaluation, merge verification, and setup receive 390 seconds, for a 1,202-second expected total.

The planning budget is 1,900 wall seconds and the hard runaway ceiling is 2,800 wall seconds, or at most 186.67 aggregate CPU minutes at four continuously occupied threads. Wall and CPU are distinct. Checkpoints, merged weights, and arm results are content-keyed. No declared work or evaluation cases will be removed to fit a shorter cap.

## Boundaries and library choice

Checked and private answers remain unread. Exact arithmetic remains offline-only. LoRA inference is entirely trained tensor computation, and the merged artifact uses the existing compliant cellular architecture. PyTorch is used directly because the adapted objects are custom raw recurrent convolution parameters; general PEFT packages target module-level linear/convolution layers and would require a less faithful wrapper. This is a considered library choice, not avoidance of external dependencies.
