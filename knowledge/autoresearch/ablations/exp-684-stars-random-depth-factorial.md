---
type: Experiment Preregistration
title: EXP684 STARS × Random Existing-Core Depth Factorial
description: Frozen equal-work M026.a/b factorial on Candidate005's inherited recurrent visits.
status: parked-calibration-resource-admission
tags: [autoresearch, STARS, JVP, random-depth, factorial, Candidate005]
timestamp: "2026-07-30T00:00:00+05:30"
---

# EXP684 STARS × random existing-core depth factorial

EXP684 crosses two training-only variables on the exact Candidate005 core: the EXP669
normalized one-step power/JVP squared-norm proxy, and a clipped random number of the
core's existing recurrent cycles. It does not import EXP670's residual refinement operator.
Formal inference always calls inherited Candidate005 `forward` with two cycles, identical
parameters, dilation visits, inputs, outputs, and calls.

The six matched 400-update routes are fixed/no-JVP, fixed/STARS, random/no-JVP,
random/STARS, within-width-stratum shuffled-random/STARS, and random/stop-gradient.
All start from checkpoint `cbc4a1bc...d4bd`, see identical batches, optimizer settings,
examples, and exactly 5,600 recurrent visits per example over the full update schedule.
JVP routes perform two additional training-only JVP products per example/update; this
unmatched gradient compute is disclosed and never enters inference.

The random stream uses seed `68400826`, Poisson rate 1.8 clipped to 1-4 cycles, then a
frozen seed-`68400827` within-width-stratum balance. Each of the four `[64,128,256,128]`
positions sums to exactly 200 cycles. The stream totals 800 cycles and hashes to
`ded69670...5573`; the seed-`68410826` within-stratum shuffle preserves the histogram,
every stratum sum, and exact work while hashing to `3178f58c...4704`.

Root 20261286 is complete-route calibration/admission. If admitted, root 20261287 first
requires joint W128 rollout >=85%, transition >=99.8%, zero invalids, W64 retention within
two points, at least three additional correct W128 cases over each component arm, positive
factorial interaction, causal shuffled/stop-gradient/trained-weight dependence, exact reload,
and deterministic inherited inference. Only then may W256 open, requiring >=10% exact and
five points over both component arms. Roots 20261288/89 require every conjunct again.

No count, JVP, target, answer, validity, repair, lookup, or external execution reaches formal
inference. No checked, independent, official/private, candidate, submission, promotion, or
release surface is authorized.

## Terminal result

The focused package passed 13/13 after one portability-only fix. The one persistent package
oracle passed validation (214 concepts, 1,100 records) and 716/716 tests in 490.82 seconds;
outer resources were 485.73 wall / 642.30 aggregate CPU seconds and 922,300 KiB RSS.

Root 20261286 completed all six routes, exact reload, deterministic repeat, finite JVP,
split separation, inference identity, equal 5,600-visit full work, and memory checks. It
rejected admission on two independent frozen resource gates: expected formal wall is
22,655.53 seconds versus 21,600, and nominal 156-second candidate inference becomes
312 seconds under measured 2.0 variability headroom versus 285. Roots 20261287-20261289
remain unopened. This is a resource park, not a scientific verdict on M026.
