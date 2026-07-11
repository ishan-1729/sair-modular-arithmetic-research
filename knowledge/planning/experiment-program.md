---
type: Experiment Plan
title: Learned Transition Experiment Program
description: Ordered hypotheses, datasets, controls, metrics, and promotion thresholds.
status: active
tags: [experiments, ablations, evaluation]
timestamp: "2026-07-11T23:15:00+05:30"
---

# Experiment program

## Common contract

Every run records a configuration, seed, prime split, source commit, environment, wall time, parameter count, checkpoint SHA-256, exact transition accuracy, bit accuracy, autonomous rollout accuracy, determinism, and limitations. Public benchmark cases are never mixed into training.

## Ordered matrix

| ID | Hypothesis | Variable | Required controls | Promotion signal |
|---|---|---|---|---|
| POC-001 | A learned local cell can execute true Horner transitions. | Small fixed width | Constant-zero and randomized-weight controls | Non-trivial held-out-prime exact rollout; perturbation collapse. |
| EXP-010 | Trajectory states reduce rollout distribution shift. | True trajectory vs random legal states | Identical steps, primes, seed family | Higher held-out rollout at equal transition budget. |
| EXP-020 | Mixed modulus sampling covers reduction boundaries. | Value-uniform, bit-uniform, mixed | Same architecture/loss | Better worst-width and held-out-prime exactness. |
| EXP-030 | Tail-aware loss reduces chain failure. | BCE, max-bit BCE, margin term | Same data stream and compute | Higher full-chain exactness, not just bit accuracy. |
| EXP-040 | Local learned rules interpolate across width. | Train/test width bands | Native-width control | Exactness retained on unseen intermediate and longer widths. |
| EXP-050 | ConvGRU improves shared-rule learning. | TCN vs iterative convGRU | Matched parameters and CPU budget | Better rollout accuracy per inference second. |
| EXP-060 | Small limbs improve throughput without fatal carry errors. | 1-, 2-, 4-bit scan units | Binary model | Lower latency with no material tier regression. |
| EXP-070 | A small-prime expert improves rank efficiently. | Router off/on | Shared high-width model | Low-tier gain with negligible artifact/runtime cost. |
| EXP-080 | Exhaustive Tier-2 residue training can clear the first unseen-prime barrier. | Specialist with all 995,400 legal triples | Prime-disjoint and secret-seed tests | At least 90% on unseen Tier-2 primes without public-prime tuning. |

## Current result

`candidate-001-horner-tcn9` clears EXP-080's practical objective without an answer-table specialist: 99% checked-public Tier 2, 97% on a full independent seed, and 98.4% across five additional Tier-2 seeds. The next bottleneck is width scaling beyond 257, not Tier-2 fitting.

## Dataset splits

- Split by prime value before generating trajectories.
- Maintain `train`, `validation`, and untouched `test` prime sets per width band.
- Generate separate public-like and stress distributions.
- Include identities, zero, near-modulus values, reduction-boundary states, short operands, and operands much wider than `p`.
- Evaluate autonomous rollouts on both seen and unseen operand lengths.

## Failure analysis

For every miss, record whether the first wrong transition resembles an addition carry, compare/subtract boundary, leading-width/position error, or accumulated state drift. Aggregate by modulus width and distance to `p`; do not rely only on a global score.
