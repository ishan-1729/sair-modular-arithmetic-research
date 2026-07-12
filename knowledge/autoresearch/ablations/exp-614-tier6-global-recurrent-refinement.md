---
type: Experiment Preregistration
title: EXP-614 Tier-6 Global Recurrent Refinement
description: Matched test of an exactly gated independent global hidden-state refinement pass against base continuation.
status: preregistered
tags: [autoresearch, tier6, recurrent-computation, refinement, straight-through]
timestamp: "2026-07-13T03:30:00+05:30"
---

# EXP-614 Tier-6 global recurrent refinement

## Question and hypothesis

Local output correction and residual widening have failed to repair shared trajectory drift. Can separately parameterized recurrent computation refine the complete hidden state before the trained local and quotient heads decode it?

The challenger reconstructs the frozen base's two learned dilation passes exactly, applies a new bidirectional global dilation block, and blends its refined hidden state through a learned per-channel `tanh` gate initialized to zero. Thus initial local and quotient logits are bit-for-bit identical to the source. Unlike EXP-611/612, the intervention changes the full hidden representation before both output heads rather than adding output logits.

A feasibility profile showed that naively reusing the base cell for extra cycles is invalid: frozen cycles 3 and 4 retained only 23.0% and 30.5% transition accuracy on timing cases and produced zero sampled rollouts. Independent parameters plus an exact zero gate are therefore necessary.

## Matched comparison

- Source: EXP-609 checkpoint `cbc4a1...d4bd`.
- Control: the 99,824-parameter base receives 200 more straight-through updates at `5e-6`.
- Challenger: the frozen base plus an 81,696-trainable-parameter global refinement block (181,520 total parameters) receives the same 200 updates and examples at `1e-3` decaying to `1e-4`.
- Both arms use the same fresh identities, seed, mode schedule, batches, eight-step trajectory loss, evaluation cases, optimizer family, weight decay, and clipping.

Parameter count, parameter initialization, learning rate, and compute are intentionally unmatched because the tested intervention is new recurrent computation. All are disclosed. Before training, 1,000 representative cases at widths 64 and 128 must reproduce source local and quotient logits bit-for-bit or the run aborts.

## Gates

Each arm is evaluated on 2,000 fresh transitions and 200 fresh autonomous rollouts at both widths. Advance only if the challenger reaches at least 90% width-128 rollout, gains at least three points over control, retains at least 95% width-64 rollout, reaches at least 99.8% width-128 transition, keeps invalid width-128 rollouts at or below 1%, and learns a hidden blend with RMS magnitude at least `0.01`.

A pass authorizes higher-fidelity confirmation and explicit projected submission-runtime measurement; it is not candidate promotion. The final 300-second official constraint is not an offline-training cap, but any promoted refinement must retain useful headroom when integrated.

## Resource budget

Ten representative eight-step updates took 27.26 wall / 105.89 aggregate CPU seconds for the control and 40.65 wall / 157.92 aggregate CPU seconds for the challenger with four threads. Linear projection gives 1,359 wall seconds for both 200-update arms. Evaluation and setup are allocated 465 seconds, for an expected total of 1,824 wall seconds.

The planning budget is 2,800 wall seconds (53% headroom). The hard runaway ceiling is 4,200 wall seconds, or at most 280 aggregate CPU minutes with four continuously occupied threads. Wall and aggregate CPU are distinct. Arm checkpoints and results are content-keyed and reusable. Neither optimizer work nor evaluation fidelity will be reduced merely to meet the projection.

## Boundaries

Checked and private answers remain unread. Python big integers are confined to offline generation and evaluation. Inference uses only trained tensor operations and contains no exact reduction, correction rule, or fallback. Native PyTorch provides the required recurrent convolution and optimizer support; a graph library is not applicable to this global hidden-state intervention.
