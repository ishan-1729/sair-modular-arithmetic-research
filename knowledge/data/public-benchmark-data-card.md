---
type: Data Card
title: SAIR Public Benchmark Data Card
description: Human-facing documentation for the pinned 1,100-case public modular arithmetic benchmark.
resource: ../../data/croissant.json
status: active
tags: [dataset, benchmark, croissant, evaluation-only]
timestamp: "2026-07-11T23:15:00+05:30"
---

# SAIR public benchmark Data Card

## Summary

The dataset contains 1,100 JSON Lines records across eleven tier files. Each record contains decimal-string operands `a` and `b`, decimal-string `p`, decimal-string `expected`, and integer `tier_id`. Tier 0 is multiplication-only diagnostic data; Tiers 1–10 are modular multiplication evaluation data.

## Motivation and intended use

- Reproduce the official public evaluation at the pinned generator revision.
- Detect regressions in interface, decoding, correctness, and runtime.
- Compare model variants only after design choices were made on independent generated validation data.

It must not be used for training, checkpoint selection, prompt tuning, or repeated hyperparameter optimization. Its fixed seed makes adaptive use a leakage risk.

## Provenance

- Upstream: [SAIR competition repository](https://github.com/SAIRcompetition/modular-arithmetic-challenge).
- Commit: `82510bba00a1126649bd76dd1a451f14d0b3eb60`.
- Generator seed label: `modchallenge-public-benchmark-v1`.
- Local files: [`data/external/public_benchmark/`](../../data/external/public_benchmark/).
- Machine-readable metadata: [`data/croissant.json`](../../data/croissant.json).
- License: Apache License 2.0, retained beside the data.

## Record schema

| Field | Type | Meaning |
|---|---|---|
| `a` | canonical decimal string | First non-negative operand. |
| `b` | canonical decimal string | Second non-negative operand. |
| `p` | canonical decimal string | Prime modulus on scored tiers; Tier 0 uses a diagnostic bound. |
| `expected` | canonical decimal string | Exact multiplication or modular-product answer. |
| `tier_id` | integer 0–10 | Evaluation tier. |

## Observed characteristics

- 100 records per tier.
- Mandatory zero/one edge cases are present in each scored tier.
- Tier 10 contains some 1,234-digit operands despite prose approximating the limit as 1,233 digits.
- Tier 0's `p` is not a scored prime modulus; consumers must honor `tier_id` semantics.
- At the pinned commit, Tiers 1–10 regenerate exactly but the final 50 Tier-0 rows regenerate with different diagnostic `p` values. The checked files remain the public-evaluation authority.

## Known limitations

- Fixed public seed and only 100 cases per tier.
- It measures one generator revision, not all valid decimal serializations or private seeds.
- Public accuracy can overstate generalization after repeated use.
- It contains answers and must never enter a release artifact.
- Small regenerated evaluations are edge-case dominated: 11 total cases are all zero-edge cases, and 110 cases contain 40% prescribed edges.

## Maintenance

The snapshot is immutable. An upstream update creates a new version with a new commit, checksums, Croissant metadata, and changelog entry; it never silently replaces these files.
