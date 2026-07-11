---
type: Evaluation Specification
title: Competition Evaluation Contract
description: Tier geometry, ranking metrics, runtime constraints, and release gates.
status: verified-with-open-questions
tags: [evaluation, scoring, runtime, sandbox]
timestamp: "2026-07-11T23:15:00+05:30"
---

# Evaluation contract

The public benchmark and private evaluation each contain 1,100 cases: 100 per tier. Public cases have a fixed seed and answers; official cases use a secret 32-byte master seed with per-tier HMAC-SHA256 derivation.

| Tier | Prime bits | Operand bits | Ranked |
|---:|---:|---:|:---:|
| 0 | diagnostic | 1–4096 | No |
| 1 | 1–3 | up to 32 | Yes |
| 2 | 4–8 | up to 48 | Yes |
| 3 | 9–16 | up to 64 | Yes |
| 4 | 17–32 | up to 96 | Yes |
| 5 | 33–64 | up to 128 | Yes |
| 6 | 65–128 | up to 256 | Yes |
| 7 | 129–256 | up to 512 | Yes |
| 8 | 257–512 | up to 1024 | Yes |
| 9 | 513–1024 | up to 2048 | Yes |
| 10 | 1025–2048 | up to 4096 | Yes |

Tier 0 is pure multiplication and diagnostic. Tiers 1–10 jointly increase modulus and operand width, forcing real modular reduction when operands greatly exceed `p`.

## Ranking metrics

1. `highest_tier_above_90`, descending.
2. `overall_accuracy`, the equal-weight average across scored tiers, descending.

The first metric makes crossing 90% on the next tier strategically more valuable than a small accuracy improvement spread across tiers. The current scorer appears to take the highest qualifying tier independently; it does not explicitly require every lower tier to clear 90%. This semantic should be monitored.

## Mechanical constraints

- Reference inference budget: 300 seconds for all 1,100 cases.
- Artifact cap: 20 GiB.
- Determinism is mandatory and contestant code must seed any randomness itself.
- A timeout zeros the active and all later tiers; fully completed earlier tiers retain their scores.
- The checked-in sandbox is CPU-first. GPU batching advice in prose is not evidence that official evaluation exposes a GPU.

## Project release gates

A candidate is not releasable until it passes:

- manifest and output-contract validation;
- static analysis;
- preprocessing isolation and repeated-run determinism;
- exact per-tier benchmark evaluation;
- CPU sandbox evaluation under the time and memory envelope;
- artifact-size audit;
- checkpoint hash and environment capture;
- weight perturbation or reinitialization control showing that learned parameters carry the capability.

See the [evaluation strategy](../evals/evaluation-strategy.md) for the stricter internal contract.
