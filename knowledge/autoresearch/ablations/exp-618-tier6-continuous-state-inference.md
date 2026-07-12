---
type: Experiment Preregistration
title: EXP-618 Tier-6 Continuous-State Inference
description: Frozen paired evaluation of hard intermediate residues versus continuous probability state for the differentiable-soft checkpoint.
status: preregistered
tags: [autoresearch, tier6, continuous-state, inference, paired-evaluation]
timestamp: "2026-07-13T05:30:00+05:30"
---

# EXP-618 Tier-6 continuous-state inference

## Question and hypothesis

EXP-607 trained a checkpoint by differentiating through sigmoid state probabilities but evaluated it by thresholding every intermediate residue. Thresholding discards uncertainty and may turn a small logit error into an irreversible state error. Does inference that carries the sigmoid probabilities into the next learned transition, thresholding only for the final answer, better match the checkpoint's training distribution and exceed 90% Tier-6 rollout?

Both paths use the exact frozen EXP-607 checkpoint `191598...22fef`. Hard inference decodes a bit vector after every step. Continuous inference supplies `sigmoid(logits)` as the next state feature plane and decodes only the final logits. No temperature, blend, threshold, or checkpoint is selected after evaluation.

## Paired evidence and gates

The two modes receive identical 500 fresh operand/modulus cases at each of widths 64 and 128. Per-case hard-only, continuous-only, both, and neither counts are retained. A shared 5,000-case transition evaluation confirms the frozen one-step function. Fifty paired cases per width are rerun from the same sampler seed for deterministic prediction equality.

Advance only if continuous inference reaches at least 90% width-128 exact rollout, gains at least two points over hard inference, retains at least 95% width-64 rollout, has at most 1% invalid final residues, and is deterministic. Intermediate continuous states are not integers; their thresholded projections are reported diagnostically but are not treated as the actual carried state.

A pass authorizes higher-fidelity fresh confirmation and implementation in a Tier-6 submission specialist. It does not authorize checked evaluation or candidate promotion.

## Resource budget

A distinct timing seed with 20 paired width-128 cases took 29.90 wall seconds and 115.57 aggregate CPU seconds with four threads. Linear projection gives 748 wall seconds for 500 width-128 cases. Width-64 pairing is allocated 190 seconds and determinism, transition, and setup 162 seconds, for a 1,100-second expected total.

The planning budget is 1,700 wall seconds and the hard runaway ceiling is 2,500 wall seconds, or at most 166.67 aggregate CPU minutes at four continuously occupied threads. Wall and CPU are distinct. Raw paired predictions are saved and hashed. The 500-case fidelity will not be reduced to fit a shorter cap.

## Boundaries

Checked and private answers remain unread. Exact arithmetic is used only to generate cases and score outputs. Continuous inference is a learned floating-point state trajectory with no modular multiplication, division, reduction, correction rule, or fallback. Native PyTorch sigmoid and tensor operations are sufficient.
