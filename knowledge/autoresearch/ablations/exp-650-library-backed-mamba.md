---
type: Ablation
title: EXP-650 library-backed Mamba-2 transition viability
description: Frozen matched comparison of a faithful ST-M014 hybrid with ConvGRU and recurrent-attention controls.
status: parked-before-calibration
tags: [autoresearch, exp-650, mamba-2, transition, exact-rollout]
timestamp: "2026-07-27T21:25:00+05:30"
---

# EXP-650 library-backed Mamba-2 transition viability

## Frozen question

Can the maintained Transformers `5.14.1` Mamba-2 CPU implementation, composed
as `Mamba2 -> Mamba2 -> Attention -> MLP` with post-residual RMSNorm, preserve
learned Horner-prefix state over exact autonomous rollouts better than matched
ConvGRU and recurrent-attention controls?

The changed variable is the transition operator. All three arms receive the
same six legal EXP-300 bit planes, exact offline Horner-prefix labels, training
pairs, width-16 examples, 800 updates, batch size, AdamW optimizer and schedule,
gradient clipping, test identities, and exact evaluator. Parameters are matched
within 10%; profiler FLOPs and measured wall/process/cgroup CPU are disclosed
rather than assumed equal.

## Frozen roots and gates

- `20261186`: non-scientific representative end-to-end calibration. It runs
  actual data generation, forward/backward, checkpoint/reload, the full 96-case
  evaluation contract at W16/W24/W32, reverse/bidirectional interventions, and
  randomized-weight evaluation. It derives wall, process CPU, cgroup CPU, RSS,
  variability headroom, ETA, and a hard runaway ceiling.
- `20261187`: one viability seed, only if calibration admits the complete route.
- `20261188` and `20261189`: unopened unless every conjunctive viability gate
  passes on root `20261187`.

The root-20261187 gate requires all of:

1. Mamba hybrid W16 autonomous exact accuracy at least 20%;
2. at least 5 percentage points gain over each matched control;
3. at least 1% exact autonomous transfer at each of untouched W24 and W32;
4. at least 10 percentage points W16 gain over the frozen randomized-weight arm;
5. zero invalid-rollout cases, exact checkpoint reload, parameter matching, and
   admitted resource/compliance guards.

Failure of any conjunct rejects this formulation at the frozen rung and leaves
confirmation roots unopened. Passing only advances to the two confirmations; it
does not promote a candidate or claim checked/private success.

## Interventions and compliance

The faithful arm has two maintained library `Mamba2Mixer` operators, global
attention, a token MLP, and four postnorm residual boundaries. Reverse input flow
and equal-weight forward/reverse flow reuse trained weights and isolate causal
direction sensitivity. The MLP-t alternative is implemented and shape-tested but
is not mixed into the primary comparison.

Exact multiplication/modulo and bit decoding are offline label/evaluation tools.
Learned forward methods contain no modulo, division, reduction, CRT, lookup, or
answer repair. The experiment does not inspect submission outputs, checked
answers, prior-independent answers, official/private evaluation, or benchmarks.

## Decision

Park before preregistration, calibration, or science at the binding package
oracle. The focused EXP-650 suite passes `7/7`, and after initializing the exact
pinned challenge submodule the project validator passes with 165 concepts,
1,100 records, and upstream `82510bba00a1`. The single allowed repaired full
oracle nevertheless ends `371 passed / 34 failed`: all failures belong to
pre-existing EXP648, EXP649, and EXP658 tests whose ignored checkpoints, caches,
or provenance/MLflow artifacts are absent from a clean worktree. EXP-650's seven
tests pass inside that full collection.

T053 requires a durable park when the package boundary remains red after one
bounded owned-cause diagnosis. Therefore roots `20261186` through `20261189`
remain unopened, no MLflow run or trained checkpoint exists, and no exact
transition/rollout or resource-admission metric is claimed. This is a
verification-boundary park, not scientific evidence against Mamba-2. Reopening
requires fresh authorization and a clean-worktree oracle with the required
cross-experiment evidence available, or an explicitly changed verification
boundary owned by the PM.
