---
type: Experiment Preregistration
title: EXP-619 Tier-6 Confidence-Augmented State
description: Matched test of hard residue state augmented by persistent prior-logit confidence features.
status: completed
tags: [autoresearch, tier6, confidence, persistent-state, straight-through]
timestamp: "2026-07-13T06:10:00+05:30"
---

# EXP-619 Tier-6 confidence-augmented state

## Question and hypothesis

EXP-618 shows that replacing hard intermediate state with probabilities breaks more correct trajectories than it repairs. However, first-error analysis shows low margins often accompany sparse errors. Can the model retain the robust hard residue while separately using previous signed confidence and confidence magnitude to correct uncertain future transitions?

The challenger adds two bounded planes, `tanh(previous logits)` and `tanh(abs(previous logits))`, to every subsequent transition. Zero-initialized confidence-to-stem, confidence-to-gate, and confidence-to-candidate kernels ensure arbitrary confidence has exactly zero effect initially. The complete 99,824-parameter base remains frozen; only 1,840 confidence parameters train. At inference the state plane is always hard bits, while confidence persists from the first transition onward.

## Matched comparison

- Control: ordinary base continuation updates all base parameters at `5e-6`.
- Challenger: the exact-expanded 101,664-parameter model freezes the base and trains 1,840 confidence parameters at `1e-3` decaying to `1e-4`.
- Both use the same source, seed, fresh identities, 200 updates, generated examples, optimizer family, weight decay, batches, mode schedule, eight-step straight-through recovery, clipping, and evaluation counts.

Parameterization and learning rate are intentionally changed by the intervention. Recovery chunks initialize confidence to zero at random chunk starts and carry it differentiably for eight steps. Before training, arbitrary confidence planes on 1,000 cases per width must reproduce source local and quotient logits bit-for-bit.

## Gates

Each arm is evaluated on 2,000 transitions and 200 autonomous rollouts at widths 64 and 128. Challenger rollout carries hard bits plus persistent confidence from the first step; one-step transition evaluation uses zero confidence. Advance only if the challenger reaches at least 90% width-128 rollout, gains at least three points over control, retains at least 95% width-64 rollout, reaches at least 99.8% width-128 transition, keeps invalid width-128 rollouts at or below 1%, and learns confidence gate weights with RMS at least `1e-4`.

A pass requires higher-fidelity confirmation. A miss closes this two-plane confidence formulation; do not tune feature transforms or channel count on this selection set.

## Resource budget

Ten representative eight-step confidence updates took 17.55 wall seconds and 67.72 aggregate CPU seconds with four threads, projecting 351 wall seconds for 200 updates. The matched control is allocated 300 seconds and full two-arm evaluation/setup 370 seconds, for a 1,021-second expected total.

The planning budget is 1,650 wall seconds and the hard runaway ceiling is 2,450 wall seconds, or at most 163.34 aggregate CPU minutes at four continuously occupied threads. Wall and CPU are distinct. Arm checkpoints and results are content-keyed. No work or evaluation cases will be reduced to satisfy a shorter cap.

## Boundaries

Checked and private answers remain unread. Exact arithmetic remains confined to offline labels and evaluation truth. Confidence features and all corrections are learned tensor computations; there is no arithmetic fallback or rule-based output repair. Native PyTorch convolution and recurrent primitives are appropriate.

## Result and decision

Exact initialization passed at both widths. Confidence weights activated well above gate: stem RMS `0.00553`, gate RMS `0.00916`, and candidate RMS `0.00867`. Nevertheless, width-128 rollout fell from 82.0% control to 78.0% challenger, width-64 fell 97.5% -> 97.0%, and transition remained 99.9% with valid residues.

Reject and close the two-plane confidence formulation. Learned confidence is not merely inactive; it materially worsens autonomous dynamics. The run completed in 948.94 wall seconds and 3,616.53 aggregate CPU seconds. Do not tune confidence transforms or channel count. Replace decoded-state recurrence with a materially different persistent latent-state architecture.
