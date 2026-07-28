---
type: Research Note
title: EXP-666 GRAM posterior-to-prior guidance mechanism map
description: Page-authenticated ST-M022.a claims, bounded SAIR mapping, frozen controls, information boundaries, and exclusions.
status: preregistered
tags: [autoresearch, gram, variational-inference, recursive-reasoning, tier7]
timestamp: "2026-07-29T00:30:00+05:30"
---

# EXP-666 GRAM posterior-to-prior guidance mechanism map

## Authenticated source

The immutable local paper is `research/papers/arxiv-2605.19376.pdf`, 27 pages, SHA-256 `c53be1192d945358fc0fa31cab4c9d8ffc09fbd686340161de504ebeb24de0f7`. `research/manifest.json`, `research/verification.json`, and `research/checksums.sha256` agree on arXiv `2605.19376v2`, title *Generative Recursive Reasoning*, 5,136,087 bytes, HTTP 200 acquisition, and cached status. PDF pages 3-5, 9-10, and 15 were rendered and visually inspected with no clipping or unreadable equations.

## Source claims and code mapping

| Source claim | Page/equation | Bounded EXP-666 mapping |
|---|---|---|
| A deterministic update receives learned Gaussian residual guidance. | p.3, Eq. 4-5 | A frozen Candidate005 proposal is the deterministic reference; a learned high state produces guidance through an exact-zero-initialized projection. |
| The latent state has slow stochastic high state and fast deterministic low state; stochasticity is high-only. | p.4, Eq. 6-9 | Two deterministic low refinements precede each high transition. Only the 32-dimensional high state is sampled. |
| Training samples a target-conditioned posterior, while inference samples a target-free learned prior. | pp.4-5, Eq. 10-13 | `forward_train(features, target_bits, ...)` alone can construct the posterior. `forward(features, ...)` has no target argument and constructs exactly one prior trajectory. |
| Prior and posterior are conditional Markov Gaussian processes with reparameterized noise and KL. | p.5, Eq. 12-14 | Native maintained `torch.distributions.Normal`, `Independent`, `rsample`, and `kl_divergence` implement diagonal-Gaussian sampling and KL. |
| Deep supervision and truncated gradients supervise consecutive recursion blocks while backpropagating only through the final transition. | p.5, Eq. 14; p.15 A.3 | Two supervision steps each contain two transitions; earlier transition states and inter-supervision carried states are detached, and both terminal logits are supervised. |
| Stochastic guidance, stochasticity-only, guide-only, direct prediction, and random initialization are discriminating ablations. | pp.9-10, Table 3 and discussion | Frozen routes are full GRAM, deterministic prior mean, target-posterior guidance without KL, stochasticity without target posterior, direct prediction, proposal-only, and randomized guidance. |

## SAIR adaptation and exact boundary

This is a bounded mechanism transfer, not a reproduction of the paper's Transformer hierarchy, ACT, Sudoku/N-Queens training, or multi-sample results. Candidate005's authenticated 92-channel decoded proposal is frozen. The deterministic low state uses a proposal-injected convolutional residual; the high state is a compact global vector. Guidance is added only through learned tensors and is exact-zero-initialized, so every non-randomized route reproduces proposal logits and quotient logits before training.

Training targets and exact bigint arithmetic are legal only in the offline sampler, posterior, loss, and evaluator. The inference API cannot accept targets, answers, correctness, or validity. Its fixed seed is `66624001`, selected before science, and it samples exactly one prior trajectory. It has fixed depth and no answer-dependent stopping, resampling, selection, filtering, clamping, repair, lookup, multiplication, division, modular reduction, CRT, or constructed program.

## Frozen arms and matched work

All trained arms receive 400 updates, the same W64/W128/W256 schedule, batch size, identities, optimizer, supervision count, low updates, and high transitions. Parameter tensors are geometrically matched; route-specific inactive parameters and operator differences are disclosed in training records. Proposal-only and randomized-guidance are intervention controls with zero optimization updates because matched learning is not meaningful for those interventions. Every route is checkpointed and exact-reload tested before formal evaluation.

## Frozen decision order

Roots are calibration `20261214`, viability `20261215`, and conditional confirmations `20261216`/`20261217`. Calibration measures all route training, validation/test W64/W128/W256 evaluation, checkpoint persistence, variational diagnostics, deterministic repeat, and candidate-shaped inference with separate wall, aggregate process-tree CPU, cgroup CPU, RSS, variability, ETA, and guards.

Formal execution trains/caches every arm before evaluating the complete 3,000-transition/300-rollout W128 full-GRAM primary. The primary must reach at least 85% rollout, 99.8% transition, zero invalid rollouts, and the candidate/runtime headroom gates. Only then may proposal/randomized retention, dependence, finite noncollapsed prior/posterior/KL, trained W128 mechanism controls, W256 primary, W256 proposal/stochasticity controls, and confirmations open. W256 requires at least 10% primary exact rollout and five-point gains over both proposal and stochasticity-only.

## Explicitly deferred M022.b

LPRM, majority voting, best-of-N, oracle selection, candidate voting, and all multi-trajectory breadth are closed. PDF p.5 and p.15 describe these mechanisms, but EXP-666 tests only ST-M022.a with one trajectory. A positive result remains intermediate and cannot establish overall-goal completion or source-paper reproduction.

## Limitations

The GRAM project page was previously observed to say code was forthcoming, so there is no claimed source-code parity. Source claims are not SAIR evidence. The exact bounded formulation may reject even when other high/low state dimensions, backbones, objectives, or multi-trajectory source settings would work. No checked, independent, official/private, benchmark, candidate, submission, promotion, or completion surface is authorized.
