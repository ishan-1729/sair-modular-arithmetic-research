---
type: Model Card
title: Candidate 004 Tier-5 Cellular Hybrid
description: Submission-ready Tier-2/3 TCN plus separate Tier-4 and Tier-5 cellular specialists.
status: promoted-intermediate
tags: [submission, candidate, Tier-5, cellular, trained-model]
timestamp: "2026-07-12T18:20:00+05:30"
---

# Candidate 004: Tier-5 cellular hybrid

Candidate 004 retains Candidate 003's width-9 and width-16 TCNs and width-32 cellular specialist unchanged, then adds the frozen 99,824-parameter EXP-521 width-64 cellular checkpoint. The clean three-file artifact has 370,225 learned parameters, totals 1,512,567 bytes, and has weights SHA-256 `fb938e208e96945b4d192b9bbddf976c8c04a077b6f39df9e9196d36e76fefa4`.

| Evaluation | Overall | Tier 1 | Tier 2 | Tier 3 | Tier 4 | Tier 5 | Highest >90% |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full independent 1,100 | 49.5% | 100% | 96% | 99% | 95% | 95% | 5 |
| Checked release regression | 50.1% | 100% | 99% | 100% | 98% | 94% | 5 |
| Randomized-weight checked control | 7.9% | 62% | 4% | 2% | 1% | 0% | none |

The pinned official isolated container reproduces 49.5% on the untouched full seed and completes inference in 36.9 seconds (75.49 seconds including generation, checks, load, and container overhead). Static analysis, preprocessing isolation, determinism, artifact size, and trained-weight dependence pass.

The width-64 adapter builds bit tensors from Python integers because legal Tier-5 primes can exceed signed `int64`; the learned recurrence still produces the residue. There is no hand-coded modular multiplication, long division, reduction, CRT, lookup, or output repair.

This is a ranking improvement, not program completion. Tier 6 and above remain unsolved, overall accuracy remains far below 85%, and no legitimate private-official score is available.
