---
type: Approach Portfolio
title: Ranked Technical Approaches
description: Prioritized model families, evidence, compliance risk, and next experiments.
status: active
tags: [research-plan, architecture, prioritization]
timestamp: "2026-07-11T23:15:00+05:30"
---

# Approach portfolio

Scores are relative planning judgments from 1 (poor) to 5 (strong), not measured results.

| Rank | Approach | Promise | CPU fit | Compliance clarity | Main risk |
|---:|---|:---:|:---:|:---:|---|
| 1 | Trajectory-trained binary Horner TCN | 5 | 4 | 4 | Chained exactness requires extraordinarily low step error. |
| 2 | ConvGRU / Neural-GPU transition | 4 | 3 | 4 | More recurrent iterations may exceed the runtime envelope. |
| 3 | Learned 2–8 bit limb recurrence | 4 | 5 | 4 | Much harder carry and reduction transition per step. |
| 4 | Small-prime DLP/additive expert | 2 | 4 | 5 | Official baseline suggests a low ceiling beyond Tier 1. |
| 5 | Trace-supervised relative-position Transformer | 2 | 2 | 5 | Direct arithmetic length generalization remains weak. |
| 6 | End-to-end decoder-only Transformer | 1 | 2 | 5 | Known baseline already collapses on unseen primes. |

## Selected POC

Implement Rank 1 at small fixed widths with:

- true Horner trajectory examples;
- prime-disjoint train/validation/test splits;
- a non-causal dilated convolutional cell with full-width receptive field;
- a hard predicted-state rollout;
- exact transition, exact rollout, runtime, determinism, and weight-perturbation metrics.

## Ordered extensions

1. Compare trajectory states with random states.
2. Compare BCE with tail-robust worst-bit margin objectives.
3. Compare prime-value, bit-length, and mixed sampling.
4. Test width interpolation and extrapolation.
5. Replace bit scans with 2-bit, then 4-bit limbs only after the binary control is stable.
6. Compare a ConvGRU cell under the identical data and evaluation contract.
7. Add a small-prime specialist only if it improves the ranking metric without meaningful latency cost.

## Stop conditions

Stop scaling a family if held-out-prime rollout remains near the trivial baseline after transition accuracy saturates, if perturbing weights does not collapse behavior, or if projected CPU latency cannot fit the official envelope with at least 25% headroom.
