---
type: Implementation Audit
title: Pinned Challenge Implementation Audit
description: Observed behavior of the benchmark, generator, scorer, sandbox, and reference models at commit 82510bb.
status: verified
tags: [audit, benchmark, scorer, baselines]
timestamp: "2026-07-11T23:45:00+05:30"
---

# Pinned implementation audit

The executable source and checked benchmark were audited at commit `82510bba00a1126649bd76dd1a451f14d0b3eb60`. Ninety-two fast upstream decoder, scorer, static, and leaderboard tests passed. These observations override prose only for reproducing that exact revision; organizers remain authoritative for future official runs.

## Benchmark observations

- Schema is exactly `a`, `b`, `p`, `expected` as canonical decimal strings plus integer `tier_id`.
- There are 100 rows per tier, no duplicate input triples, and no incorrect checked answers.
- Tiers 1–10 regenerate bit-for-bit. The final 50 Tier-0 rows do **not**: `a`, `b`, and `expected` match, while current generator choices produce different diagnostic `p` values.
- The checked JSONL is therefore the authority for public-result reproduction; generator output is a separate secret-style diagnostic at this revision.
- Tier 1 cycles four fixed primes even though configuration requests five, biasing the public counts toward `p=2`.
- Tier 2 has 48 possible primes under the actual generator interval. After sanctioned `a % p` and `b % p`, its complete residue domain is 995,400 triples—small enough for exhaustive specialist experiments.

## Actual checked baseline results

| Model | Parameters | Tier 1 | Tier 2 | Equal-tier overall |
|---|---:|---:|---:|---:|
| always zero | 0 | 0.62 | 0.04 | 0.082 |
| digit Transformer | 544,256 | 1.00 | 0.07 | 0.121 |
| DLP grokking | 3,008,818 | 1.00 | 0.12 | 0.127 |

The zero-baseline documentation says 0.079, and DLP documentation says roughly 6M parameters; neither matches the checked artifacts. Training provenance also drifts between some manifests, READMEs, and scripts. This project generates cards from immutable run records to prevent the same failure.

## Scorer and evaluator behavior

- `highest_tier_above_90` is the maximum individual qualifying tier; lower tiers need not qualify.
- Ranking code deduplicates by repository ID, while prose describes team-level deduplication.
- `--total 11` contains only the first `a=0` edge case per tier and makes an always-zero model appear perfect. `--total 110` overweights prescribed edge cases at 40% rather than 4%. Neither is valid for model selection.
- Batch exceptions become wrong predictions and evaluation continues; batch-length mismatch marks the tier incomplete.
- Timeout is checked cooperatively before batches rather than enforced by a hard process kill in the local sandbox path.
- An empty output digit list decodes to zero, although relying on this is poor contract hygiene.

## Runtime and sandbox

The published wrapper defaults to four CPUs, 8 GiB RAM, and 2 GiB `/tmp`, with CPU PyTorch 2.5.1. Preprocessing should return strings or lists because the current isolation checker mishandles equality on multi-element tensors. The package surface is broader than the prose allowlist suggests, but exploiting that mismatch would violate the competition's intent.

## Source links

- [`config.py`](../../sources/github/modular-arithmetic-challenge/src/modchallenge/config.py)
- [`generator.py`](../../sources/github/modular-arithmetic-challenge/src/modchallenge/testgen/generator.py)
- [`primes.py`](../../sources/github/modular-arithmetic-challenge/src/modchallenge/testgen/primes.py)
- [`pipeline.py`](../../sources/github/modular-arithmetic-challenge/src/modchallenge/evaluation/pipeline.py)
- [`results.py`](../../sources/github/modular-arithmetic-challenge/src/modchallenge/evaluation/results.py)
- [`docker/README.md`](../../sources/github/modular-arithmetic-challenge/docker/README.md)
