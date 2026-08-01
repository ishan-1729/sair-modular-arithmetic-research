---
type: Experiment Preregistration
title: EXP-638 Full-Depth Soft-Boundary Diagnostic
description: No-training diagnostic of two full-depth teacher transitions joined by one continuous intra-macro state.
status: completed
tags: [autoresearch, cellular, macro-transition, continuous-state, diagnostic, training-identities]
timestamp: "2026-07-17T12:32:00+05:30"
---

# EXP-638 full-depth soft-boundary diagnostic

## Question and evidence basis

EXP-637 rejects compute compression: its one-cycle phase 2 remains inaccurate even when given the exact intermediate state. Yet its learned phase-1 intermediate is far more accurate than its final macro output, and a two-bit macro boundary still offers a distinct hypothesis from direct radix-4 learning. Does restoring both phases to the frozen teacher's full two-cycle depth preserve exactness when the internal state remains continuous and only the macro output is hard decoded?

This is a frozen, no-training diagnostic on training prime identities only. It cannot promote a candidate or establish held-out generalization. Its sole purpose is to decide whether a separately preregistered full-depth training rung has a viable initialization surface.

## Frozen arms and legal inference

Use the immutable EXP-609 checkpoint, SHA-256 `cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd`, for every learned call. Compare three transition arms on identical sampled cases:

1. **hard composition control**: apply the full two-cycle teacher to the high bit, hard decode its state, then apply the same full teacher to the low bit;
2. **continuous macro**: apply the full teacher to the high bit, transform its logits to the legal feature range with `tanh(logit / 2)`, feed that plane directly to the second full teacher, and hard decode only the second output;
3. **exact-intermediate diagnostic**: provide the offline exact high-bit state to the second teacher, isolating the phase-2 upper surface. This is diagnostic only and is prohibited from any candidate inference path.

The hard and continuous arms receive only legal input bit planes, operands, modulus, digit bits, and learned outputs. Neither contains exact modular multiplication, division, reduction, repair, or lookup. Exact bigint arithmetic appears only in offline sampling, truth, and the explicitly non-candidate exact-intermediate control.

## Frozen workload and gates

On training identities disjoint from all test identities, run 2,000 paired transitions at widths 16, 32, and 64 for all three arms. At width 64, run 200 paired multiplication rollouts for the hard and continuous arms. Repeat 20 continuous width-64 cases with every intermediate/final logit retained for exact determinism. Cache and hash each completed width and rollout block.

Authorize a full-depth training rung only if every gate passes:

- continuous width-64 transition is at least 99%;
- continuous width-64 rollout is at least 90%;
- width-64 continuous transition loses at most one point to hard composition;
- width-64 continuous rollout loses at most three points to hard composition;
- continuous width-64 wall time is at most 1.25 times hard composition;
- predictions, intermediate/final logits, and validity repeat exactly.

Failure closes full-depth continuous macro initialization without consuming held-out evidence. Passing authorizes only a separately preregistered training experiment with its own measured budget and matched hard/continuous controls.

## Dynamic resource budget

Reuse valid EXP-637 hardware evidence for the exact frozen teacher and same evaluator shapes. Its 2,000-case composed transition blocks consumed 3.881, 6.297, and 21.406 wall seconds at widths 16, 32, and 64; its width-64 200-case full-depth rollout consumed 75.985 wall seconds. Three transition arms project to 94.751 wall / 347.927 aggregate CPU seconds. Two rollout arms project to 151.970 wall / 567.112 aggregate CPU seconds, and two 20-case determinism repeats project to 15.197 wall / 56.711 aggregate CPU seconds.

With 60 seconds for setup, MLflow, caching, and serialization, expected total is 322 wall seconds and 19 aggregate CPU minutes. The soft budget is 480 wall seconds; the hard runaway ceiling is 720 wall seconds, with a distinct four-thread upper bound of 48 aggregate CPU minutes. These limits are derived from measured same-checkpoint throughput, not an arbitrary short cap. All cases and repeats remain fixed, completed blocks are reused by hash, and no fidelity may be reduced to meet a limit.

## Limitations

The diagnostic uses training identities and cannot support generalization claims. It changes only the intermediate representation, not weights. Continuous logits may be poorly calibrated even when their signs are correct. A pass would justify adaptation, not prove that full-depth macro training improves Tier-6 rollout or total submission runtime.

## Result and decision

Every frozen authorization gate passes. On 2,000 training-identity transitions, hard/continuous/exact-intermediate exactness is 99.8%/99.65%/99.8% at width 16, 99.8%/99.8%/99.9% at width 32, and 99.9%/99.9%/99.9% at width 64. Continuous state therefore preserves the full teacher transition surface and has zero width-64 deficit to hard composition.

On 200 paired width-64 training-identity rollouts, hard composition scores 190/200 (95.0%) and continuous macro scores 191/200 (95.5%). Continuous recovers two hard failures and loses one hard success; 189 are common successes and eight common failures. Both arms have zero invalid trajectories. Continuous uses 48.506 wall seconds versus 52.959 for hard composition, a 0.916 ratio, and repeats every prediction, intermediate/final logit, and validity result exactly.

Authorize a separately preregistered full-depth soft-boundary adaptation rung. Do not treat the 0.5-point training-identity gain as held-out evidence, do not promote a candidate, and do not inspect checked or independent-full benchmarks. The next rung must include fresh held-out paired hard/continuous controls, width-64 retention, width-128 transition/rollout, trained-weight dependence, deterministic/runtime gates, and its own measured full-objective budget.

The complete diagnostic finishes in 182.142 runner wall seconds and 640.607 aggregate CPU seconds; GNU time reports 180.33 wall and 650.65 user-plus-system CPU seconds. This is below the 322-second expected, 480-second soft, 720-second hard wall limits and separate 19/48 aggregate CPU-minute expected/upper limits. All blocks completed and no fidelity was reduced.
