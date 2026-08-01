---
type: Ablation
title: EXP-652 answer-plus-latent Tiny Recursion Model
description: Frozen bounded-faithful ST-M010.a TRM viability and causal-component package.
status: completed-rejected
tags: [autoresearch, exp-652, trm, latent-state, recursion]
timestamp: "2026-07-27T23:55:00+05:30"
---

# EXP-652 answer-plus-latent Tiny Recursion Model

## Frozen question

Can a small width-independent two-layer network, organized as a persistent
answer canvas `y` and separate reasoning state `z`, learn exact W16 modular
multiplication and transfer materially better to untouched W24/W32 identities
than equal-parameter decoded-cellular and no-latent controls?

The primary implements ST-M010.a's ordering: six shared `z` updates followed by
one `y` update, repeated in three outer recursions; the first two outer blocks
are forward-only and detached, while every shared call in the final block is on
the gradient graph. Two detached supervision steps train endpoint bits and a
correctness Q head; EMA decay is 0.999. The local convolutional operator and bit
BCE are declared SAIR adaptations. This is not Tab-TRM.

## Frozen roots and gates

- `20261190` is non-scientific full-route dynamic calibration for all six
  potential route classes, full W16/W24/W32 evaluation, checkpoints/reload,
  recursion/Q/state interventions, randomized weights, MLflow, and measured
  wall/process/cgroup CPU, RSS, variability, ETA, and expected/soft/hard guards.
- `20261191` trains the primary and two matched controls for 300 updates each if
  calibration admits the route.
- `20261192` and `20261193` open only if every viability conjunct passes. They
  reproduce viability and add no-deep-supervision, no-Q-loss, and no-EMA arms.

Viability requires primary EMA exact accuracy of at least 80% at W16 and 5% at
both W24 and W32; at least two points transfer gain over each control at both
wider widths; at least twenty points trained-weight dependence; at least ten
points latent causal drop; zero invalids; exact raw/EMA reload; equal parameters,
examples, and optimizer updates; and admitted resources. Any failed conjunct
rejects this formulation and leaves confirmations unopened.

## Safety

Inputs contain only operand/modulus bit planes and position markers. Exact
arithmetic is restricted to offline endpoint labels, tests, and evaluation.
There is no evaluation identity feature, intermediate exact-state feature,
lookup, programmed arithmetic, answer repair, checked/private evidence, or
submission mutation. The claim-to-code map is
[`2026-07-27-exp652-trm-mechanism.md`](../research/2026-07-27-exp652-trm-mechanism.md).

## Decision

Reject this exact bounded formulation at viability root `20261191`. The local-
PDF mechanism map, 10 focused tests, project validator, and sole package oracle
(`416/416`) pass. Root `20261190` profiles every potential route class and
admits execution: observed route resources are 7.185 wall / 22.410 process CPU /
20.856 cgroup CPU seconds and 486,719,488-byte peak RSS. The confirmation-route
projection is 121.631 expected / 152.038 soft / 243.262 hard wall seconds, with
separate process and cgroup guards.

Root `20261191` then trains all three equal-parameter arms for 300 optimizer
updates and 9,600 examples each. Every arm scores 0/96 exact at W16, W24, and
W32. Primary EMA bit accuracy is 55.60%, 54.47%, and 53.78%, but that diagnostic
does not substitute for exact answers. Raw primary invalid rates are 4.17%,
5.21%, and 2.08%; EMA invalid rates are zero. Raw and EMA checkpoint reload,
parameter matching, update/example matching, and resources pass. Exact trained-
weight dependence, latent causal drop, all exactness/transfer gains, and the
validity conjunct fail.

Recursion depths 1-4, learned Q halt, randomized weights, and every answer/latent
state intervention remain 0% exact. Q learns only the always-incorrect class,
so its nominal threshold accuracy is not positive halting evidence. Viability
failure permanently leaves roots `20261192` and `20261193` unopened; therefore
no separate causal claim is made for deep supervision, Q loss, or EMA. The
result rejects this exact endpoint-supervised local-convolutional SAIR TRM rung,
not the source paper or Tab-TRM.
