---
type: Experiment Preregistration
title: EXP-230 Learned Quotient-Routed Experts
description: Compliant neuro-symbolic A/B using learned routing and learned residue heads only.
status: completed
tags: [autoresearch, neuro-symbolic, routing, quotient, Tier-4]
timestamp: "2026-07-12T11:55:00+05:30"
---

# EXP-230: learned quotient-routed experts

## Compliance boundary

The quotient class `q in {0,1,2}` is an offline training label and diagnostic. At inference, a learned quotient head chooses among three learned bit-logit experts. No exact quotient, subtraction, modular reduction, candidate residue, or output repair is computed in the inference path. All three experts initialize as exact copies of EXP-204's learned single head, so the routed model reproduces source output before fine-tuning regardless of the router choice.

## Matched A/B

Both arms start from frozen EXP-204 checkpoint `a685d8d6f37de7e7ef9f28075d0779256d31330a834a1a69dd26b56eb2d62c85`, use fresh official-like prime identities, identical hard-mined batches, 800 steps, batch size, losses, optimizer, schedule, and evaluation cases.

- Control: continue the original single learned head.
- Challenger: train three learned heads using the offline quotient label as teacher routing while continuing to train the learned router. Autonomous evaluation uses only predicted learned routing.

Causal controls evaluate oracle teacher routing on heldout transitions and forced expert-0 routing on autonomous rollouts. The latter must reduce performance, demonstrating that routing and expert specialization materially affect answers.

## Gate

Require at least +1.5 rollout points over control, 92% routed rollout, 99.8% transition, 99.5% router accuracy, at least 0.5-point forced-expert-0 rollout degradation, no more than 0.5% invalid rollouts, at most 2x parameters, and at most 1.5x training time. Passing authorizes fresh independent Tier-4 seeds. Failing closes this learned-routing formulation. No consumed independent seed or checked answer is used.

## Outcome

Rejected. Router accuracy reached 99.998%, but routed rollout improved only 91.98% -> 92.17% (+0.19 points) versus a +1.5-point gate. Forcing expert 0 reduced rollout only to 92.04%, a 0.13-point drop versus 0.5 required, and oracle teacher routing changed transition from 99.786% to only 99.794%. Learned routing was accurate but not the limiting mechanism; the added 1.78x parameters were largely redundant.
