---
type: Ablation
title: EXP-664 SPRM hidden-boundary perturbation
description: Bounded-faithful SPRM adaptation rejected by its frozen W128-first formal gate.
status: completed-rejected
tags: [autoresearch, exp-664, sprm, tbptt, state-perturbation]
timestamp: "2026-07-28T04:35:00+05:30"
---

# EXP-664 SPRM hidden-boundary perturbation

## Frozen method

EXP-664 implements ST-M019.b as Gaussian perturbation only at detached training TBPTT hidden-state boundaries. Canonical noise uses the local hidden-state mean and variance with the paper's linear beta schedule. Clean-TBPTT, wrong-boundary, and unscaled-boundary routes share the same 99,824-parameter cell and 400-update width schedule. The public noise-free `forward(features)` directly delegates to the frozen EXP-609 source cell, so inference adds neither random draws nor model calls. The [mechanism map](../research/2026-07-28-exp664-sprm-boundary-mechanism.md) records the page-level PDF evidence, declared SAIR adaptations, and exclusions.

The preregistration froze calibration root `20261206`, formal root `20261207`, conditional confirmations `20261208`/`20261209`, W64/W128/W256, 3,000 transitions plus 300 rollouts per formal test width, and 1,000/100 validation cases. No decoded-answer corruption, label-time inference, exact programmed reduction, lookup, or repair enters inference.

## Verification and admission

Final executable bytes pass 15/15 focused tests, project validation (174 OKF concepts and 1,100 benchmark records), and the sole final full oracle (465/465 tests in 454.84 pytest seconds; 467.10 outer wall seconds; 920,312 KiB peak RSS).

The full-route calibration completed in 63.72 wall seconds and admitted formal root `20261207`. It exercised training, validation/test rollouts, determinism, interventions, persistence, candidate inference, and all declared routes. Its projection was 4,989.44 expected / 7,571.16 ETA / 15,143 hard-guard wall seconds.

## Formal result

The W128-first formal stage trained all four trainable routes for 400 updates and evaluated the SPRM primary on the full 3,000-transition/300-rollout test denominators. Transition exactness was 99.2333% with zero invalid residues, but rollout exactness was only 43.6667%, rollout transition exactness 68.4740%, and 0.6667% of rollouts produced an invalid residue. This fails the frozen conjunctive primary gate despite exact noise-free inference parity (`logits_equal`, `quotient_equal`, and unchanged RNG; measured wrapper/direct wall ratio 0.9727).

The gate therefore rejected this exact bounded SPRM adaptation and left clean/randomized W128, W64, W256, causal-control verdicts, and roots `20261208`/`20261209` unopened. This is not a general rejection of SPRM or of the source paper. Candidate 005, checked/independent benchmarks, submission, promotion, and the 85% goal remain unchanged.
