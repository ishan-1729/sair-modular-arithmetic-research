---
type: Experiment Preregistration
title: EXP682 Huginn Recurrent Depth
description: Frozen bounded-faithful ST-M006.a transfer package and terminal topology-feasibility decision.
status: completed
tags: [autoresearch, Huginn, recurrent-depth, TBPTT, latent-state, topology-park]
timestamp: "2026-07-30T00:00:00+05:30"
---

# EXP682 Huginn recurrent depth

## Frozen question and complete package

EXP682 asked whether Huginn's random latent, repeated embedded-input reinjection,
`2h -> h` concatenation adapter, sandwich RMSNorm, heavy-tailed bounded recurrence, and
last-eight TBPTT can transfer around frozen Candidate005 while retaining at least 95%
W128 rollout and creating deterministic learned W256 depth extrapolation.

The complete frozen matrix contained 400 matched updates for full Huginn, fixed-mean
depth, full-BPTT, zero-latent, deterministic-latent, no-reinjection, no-concat,
no-RMSNorm, reset-latent, and shuffled-latent arms, plus frozen-source and randomized-
weight controls. Release depths were 2/5/8/12. The calibration/formal/confirmation roots
were 20261278/20261279/20261280/20261281. Exact W128 rollout, transition accuracy,
invalids, W256 exactness, depth trend, component gradients/interventions, deterministic
reload, parameters/calls/resources, and guarded inference were frozen as gates.

## Pre-root falsifier

The package first required either exact checkpoint transfer with all source-required
components causal, or a viable de-novo route. The authenticated executable source map
proves these conditions cannot both hold for Candidate005. Candidate005 uses GroupNorm
inside a ConvGRU that directly concatenates raw six-channel features; it lacks the random
latent, hidden-input concat adapter, and sandwich RMSNorm. The source-complete topology
changes checkpoint outputs, while neutralizing it would make the tested components inert.

The de-novo fallback is also below its frozen evidence threshold. Adjacent EXP634 trained
a recurrent-depth model for 4,000 updates at width 12 and obtained only 2/300 exact,
well below 20%. Opening a 400-update width-128 route with a 95% gate would not be a viable
or decision-worthy de-novo falsifier.

## Decision and scope

Park before calibration under T084's explicit topology/viability stop. All arms remain
declared but unopened; all four roots remain unconsumed. No post-result retuning or repair
successor exists. The executable probe verifies native RMSNorm/distributions, deterministic
release, bounded heavy-tail sampling, TBPTT detach placement, gradients, component/state
interventions, reload identity, and fail-closed safety without treating those checks as
accuracy evidence.

This result closes only the exact Candidate005 transfer and the 400-update de-novo fallback.
It does not reject Huginn, ST-M006.a on a native architecture, or a separately preregistered
from-scratch training program. No checked, independent, official/private, candidate,
submission, promotion, or Goal-completion surface was opened.
