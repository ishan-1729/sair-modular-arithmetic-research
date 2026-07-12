---
type: Model Card
title: Candidate 003 Tier-4 Cellular Hybrid
description: Submission-ready Tier-2/3 TCN plus Tier-4 width-independent cellular ensemble.
status: promoted-intermediate
tags: [submission, candidate, Tier-4, cellular, trained-model]
timestamp: "2026-07-12T11:15:00+05:30"
---

# Candidate 003: Tier-4 cellular hybrid

Candidate 003 combines the retained width-9 and width-16 TCN specialists with a 99,824-parameter cellular specialist trained through width 32. The clean three-file artifact has 270,401 learned parameters, totals 1,109,234 bytes, and has weights SHA-256 `a991c389aec5f8c4e8153846b07d7d1207bdd0eaf192ff11ca036663f056a514`.

| Evaluation | Overall | Tier 1 | Tier 2 | Tier 3 | Tier 4 | Highest >90% |
|---|---:|---:|---:|---:|---:|---:|
| Full independent 1,100 | 40.4% | 100% | 98% | 99% | 95% | 4 |
| Checked release regression | 40.9% | 100% | 99% | 100% | 98% | 4 |
| Randomized-weight checked control | 7.7% | 62% | 1% | 2% | 0% | none |

The official isolated container reproduces the independent 40.4% score and completes inference in 7.9 seconds under four CPUs, 8 GiB memory, no network, read-only root and submission mounts, and the 300-second timeout. Static analysis, preprocessing isolation, determinism, artifact size, and trained-weight dependence pass.

Pairwise operand reductions use the competition-sanctioned front end. The learned recurrence produces the final residue; there is no hand-coded modular multiplication, long division, reduction, CRT, lookup, or output repair. Inputs above Tier 4 receive the deterministic edge-case fallback.

This is a ranking improvement, not program completion. Tier 5 and above remain unsolved, checked and independent overall accuracy remain far below 85%, and no real private-official score is available.
