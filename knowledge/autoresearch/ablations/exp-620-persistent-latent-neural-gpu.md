---
type: Experiment Preregistration
title: EXP-620 Persistent Latent Neural GPU
description: Parameter-matched smoke test of persistent latent computation against decoded-state cellular recurrence.
status: preregistered
tags: [autoresearch, neural-gpu, persistent-state, architecture, curriculum]
timestamp: "2026-07-13T06:50:00+05:30"
---

# EXP-620 persistent latent Neural GPU

## Question and hypothesis

Every Tier-6 decoded-state intervention has plateaued below 90% or regressed. Errors arise because an intermediate residue is decoded to bits, thresholded, and re-encoded up to 128 times. Can a spatial ConvGRU state remain latent across multiplier bits, receive the current bit plus static multiplicand/modulus planes, and decode only for all-prefix supervision and final output?

The persistent model has 40,577 parameters, two local microsteps per multiplier bit, and no threshold or decoded-state feedback in its forward path. Its state dict is width independent. The control is the established decoded-state dilated ConvGRU at hidden 58, with 41,242 parameters. Both are trained from scratch.

## Matched smoke

Both arms receive the same 76,800 generated trajectories (1,200 updates × batch 64), width schedule 8/12/16, fresh prime identities, every prefix residue label, seed, optimizer, learning-rate schedule, weight decay, worst-bit coefficient, clipping, and evaluation cases. The persistent model backpropagates through the full latent trajectory and decodes every prefix for supervision. The control trains on the same teacher prefix transitions flattened within each trajectory, then evaluates autonomously by decoding/re-injecting state.

Parameter counts and trajectory examples are closely matched. Network operations and wall time are intentionally unmatched because persistent versus decoded computation is the causal architecture difference. No public or prior checked answers enter training or selection.

## Gates

Frozen endpoints are evaluated on 200 fresh complete multiplications at widths 16, 24, and 32. Width 16 is trained; widths 24 and 32 are zero-shot. Advance to a progressive 32→64→128 curriculum only if the persistent model:

- reaches at least 80% width-16 exact rollout;
- gains at least 20 points over the decoded control at width 16;
- reaches at least 20% zero-shot width-24 rollout;
- reaches at least 5% zero-shot width-32 rollout;
- keeps invalid final residues at or below 2% at every evaluated width.

This smoke tests architecture viability, not Tier-6 promotion. A miss closes this exact hidden/microstep formulation; a pass authorizes higher widths with newly preregistered budgets.

## Resource budget

For the persistent model, 20 width-16 updates took 4.69 wall seconds at batch 64; 10 width-32 updates took 4.69 seconds at batch 32. The 1,200-update 8/12/16 schedule projects to 210 wall seconds. The parameter-matched control took 11.21 seconds for 20 width-16 updates, projecting 480 seconds. Evaluation and setup receive 90 seconds, for a 780-second expected total.

The planning budget is 1,200 wall seconds and the hard runaway ceiling is 1,800 wall seconds, or at most 120 aggregate CPU minutes with four continuously occupied threads. Wall and aggregate CPU are distinct. Arm checkpoints and results are content-keyed. Training or evaluation fidelity will not be reduced to meet a shorter cap.

## Boundaries and library choice

Exact arithmetic is restricted to offline prefix labels and evaluation truth. Inference is a trained tensor-only recurrent network with no modular multiplication, reduction, division, lookup, or fallback. PyTorch provides the required ConvGRU primitives directly; no mature specialized Neural GPU package offers a better-maintained implementation surface for this custom width-shared model.
