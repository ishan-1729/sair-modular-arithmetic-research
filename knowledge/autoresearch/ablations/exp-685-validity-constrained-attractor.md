---
type: Experiment Preregistration
title: EXP685 validity-constrained attractor continuation
description: Matched 200-update continuation of EXP665's W128 near miss with a training-only differentiable residue-range constraint.
status: permanent-package-verification-park
tags: [autoresearch, exp-685, validity, continuation, anderson]
timestamp: "2026-07-30T00:00:00+05:30"
---

# EXP685 - validity-constrained attractor continuation

## Hypothesis

EXP665 missed promotion only because one of 300 W128 autonomous rollouts entered
an invalid residue while exact rollout was 85.0% and transition exactness was
99.933%. A differentiable training-only probability of predicted binary
residue `>=` the legal modulus may remove that failure without erasing learned
exact behavior.

## Frozen comparison

The validity, CE-only, and deterministically shuffled-modulus arms continue the
same authenticated Anderson/phantom checkpoint for 200 matched updates. The
validity coefficient is exactly `0.05`; data schedule, batch size, optimizer,
labels, and supervision calls are identical. Frozen and independently
randomized controls receive zero updates. Formal inference is the unchanged
EXP665 `AttractorCell` and TorchDEQ Anderson route.

Rootless all-mode smoke and one package oracle precede every root. Root
`20261290` measures complete-route resources at full-fidelity projections. If
admitted, root `20261291` evaluates W128 first: at least 85% rollout, 99.8%
transition, zero invalid rollouts, retention within two points of frozen and
CE, lower validation invalid-probability Brier score than CE and shuffled
controls, nonzero constraint gradients, trained-weight dependence, exact
reload, and guarded candidate inference at most 285 seconds. Only that
conjunction opens W256, requiring at least 10% exact and five-point gains over
CE and shuffled controls. Roots `20261292/93` reproduce all conjuncts. A pass
remains intermediate.

No checked, independent, official/private, candidate, submission, promotion,
or completion evaluation is authorized.

## Terminal verification result

The real rootless smoke completed three one-update trained arms, five exact
save/reloads, one transition and rollout case per arm, admission math, and
calibration/formal/confirmation serialization. Ten focused mathematical,
gradient, shuffle, identity, intervention, reload and safety checks passed in
19.93 seconds.

The sole package oracle then stopped in project validation on seven owned
canonical-record schema omissions: four required run fields, two required
metrics fields, and a null evaluation record. Pytest did not open. The bounded
diagnosis found no executable loss, runner, or inference-identity failure, but
the single-oracle stop rule forbids a rerun. Roots `20261290`-`20261293` remain
unopened, so EXP685 is permanently parked without a scientific verdict.
