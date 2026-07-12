---
type: Candidate Preregistration
title: Candidate 004 Tier-5 Cellular Hybrid Release Gate
description: Frozen-artifact complete release gate for the Tier-1 through Tier-5 learned ensemble.
status: completed
tags: [autoresearch, candidate, Tier-5, release, full-test]
timestamp: "2026-07-13T00:10:00+05:30"
---

# Candidate 004: Tier-5 cellular hybrid release gate

Freeze the exact three-file Candidate 004 artifact recorded in `configs/candidate_004_release_gate.yaml`. It is Candidate 003 unchanged plus the independently validated EXP-521 checkpoint under a `tier5.` safetensors prefix. The 64-bit adapter constructs legal bit tensors from Python integers so primes above signed `int64` remain representable; it contains no exact modular multiplication, division, reduction, CRT, or corrective fallback.

Evaluate the frozen artifact once on the preregistered untouched official-generator full 1,100-case seed. Promotion requires at least 49% exact overall, Tiers 1–5 at their frozen thresholds, `highest_tier_above_90 = 5`, clean static analysis, preprocessing isolation, deterministic repeat, randomized-weight collapse on Tier 5, and reproduction in the pinned official isolated container. Only after independent selection passes, run the complete checked 1,100-case release regression. Candidate 004 remains an intermediate candidate because even a pass is far below the binding 85% overall completion threshold.

## Dynamic resource budget

Candidate 003's pinned full container gate consumed 34.284 wall seconds including startup. The slowest EXP-521 Tier-5 100-case inference consumed 9.973 wall seconds, projecting 44.257 wall seconds for Candidate 004. Allocate 120 wall seconds as the soft budget and the official 300 wall seconds as the hard runaway ceiling for each full evaluation. Wall time is distinct from aggregate CPU time; with four threads the conservative upper ceiling is 20 aggregate CPU minutes per evaluation. Reuse the frozen EXP-521 five-seed evidence rather than rerunning it.

## Result

Pass. The untouched official full set scores 49.5% overall with Tiers 1–5 at 100%, 96%, 99%, 95%, and 95%; the checked post-selection regression scores 50.1% with Tiers 1–5 at 100%, 99%, 100%, 98%, and 94%. Randomized weights collapse Tier 5 to 0%. Static analysis, preprocessing isolation, deterministic repeat, and the pinned container pass. The observed 75.49-second container wall time and 36.9-second inference time remain within the calibrated 120/300-second budget.

Decision: promote Candidate 004 as the intermediate incumbent. Continue at Tier 6; the 85% overall completion gate remains unmet.
