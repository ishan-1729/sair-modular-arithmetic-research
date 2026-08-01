---
type: Ablation
title: EXP-650 library-backed Mamba-2 transition viability
description: Frozen matched comparison of a faithful ST-M014 hybrid with ConvGRU and recurrent-attention controls.
status: parked-after-resource-calibration-oom
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

T054 recovered a hermetic package boundary using 268 authenticated, independent
regular-file fixtures. The exact inherited cone passed `34/34`, the validator
passed 165 concepts and 1,100 records, and the single complete project oracle
passed `405/405`. A pre-science correction made the inherited 7,200/28,800
second values explicitly nonbinding planning references; measured calibration
now derives expected, soft, and hard guards. The corrected focused suite passed
`8/8`, and preregistration froze at
`e7116ef343725585fb39065ae71ba9b1078af2b09671659069aa718bd397986f`.

Park at root `20261186` resource admission. The full-fidelity calibration opened
without shortening training or evaluation, but Linux globally OOM-killed Python
during Mamba-hybrid training after only the 97,108-byte initial checkpoint was
written. Kernel evidence records 7,400,644 KiB anonymous RSS against a
7,928,476 KiB WSL memory total. No arm completed, so process/cgroup CPU,
variability, ETA, projected guards, exact transition/rollout, controls, and
interventions are unavailable rather than zero. MLflow run
`b7f9d448b79b4467989b2e0ecbb0f357` is terminally `KILLED`.

The calibration route is not admitted. T054 forbids reduced fidelity, a retry,
or another EXP650 recovery package, so scientific roots `20261187`–`20261189`
remain unopened. This is measured resource evidence and not a scientific
rejection of the maintained-library Mamba-2 hypothesis.
