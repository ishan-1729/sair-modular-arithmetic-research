---
type: Research Note
title: EXP685 training-only residue-validity constraint
description: Frozen mathematical map for a differentiable range constraint around the exact EXP665 Anderson/phantom inference route.
status: permanent-package-verification-park
tags: [autoresearch, exp-685, validity, attractor, anderson, training-only]
timestamp: "2026-07-30T00:00:00+05:30"
---

# EXP685 validity-constrained attractor map

EXP665's immutable Anderson/phantom checkpoint `a06103dbe216...e4` reached
exactly 255/300 W128 rollouts, 2,998/3,000 transitions, and one rollout
containing an invalid residue. Its formal inference uses the 142,728-parameter
`AttractorCell`, persistent proposal injection, and TorchDEQ 0.1.0 Anderson
solve. EXP685 imports that exact class and checkpoint; it adds no inference
parameter, call, feature, comparator, selector, or repair.

For independent Bernoulli output bits `X` and fixed legal modulus bits `m`, the
training-only constraint computes the exact probability `P(X >= m)`. Scanning
from MSB to LSB, mutually exclusive mass is accumulated whenever the equal
prefix first encounters `m_i=0, X_i=1`; the final all-equal mass is included.
The implementation evaluates this recurrence in log space and averages its
bounded `[0,1]` probability over the batch. The frozen primary objective adds
`0.05` times that mean to EXP665's unchanged objective.

CE-only performs the same probability computation but multiplies it by zero.
The shuffled-modulus control uses the same coefficient and work with a
deterministic within-batch modulus permutation. All three trained arms start
from the same checkpoint, consume identical legal batches on schedule
`[64,128,256,128]`, and receive 200 updates with matched optimizer and
supervision calls. Frozen-source and independently randomized-correction arms
receive no updates.

Exact arithmetic, modulus comparison, calibration labels, and decoded validity
exist only in offline loss, tests, and evaluation. Rootless preflight must run
one real update per trained arm, save/reload, one transition and rollout per
arm, admission math, and calibration/formal/confirmation serialization before
root `20261290`. Full W128/W256 denominators and roots `20261291`-`20261293`
remain conditional on the frozen staged gates.

The rootless implementation smoke and 10/10 focused cone passed. The sole
package oracle stopped before pytest on seven owned run/metrics schema
omissions, so the no-rerun boundary left every root unopened. This is a
verification park, not evidence about the constraint's scientific effect.
