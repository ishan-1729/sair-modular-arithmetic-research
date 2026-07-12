---
type: Evaluation Result
title: Candidate 004 Evaluation Results
description: Independent, checked, randomized-control, and pinned-sandbox evidence for Candidate 004.
status: completed
tags: [evaluation, candidate, Tier-5, exact-accuracy]
timestamp: "2026-07-12T18:20:00+05:30"
---

# Candidate 004 evaluation results

The exact frozen artifact passes the complete preregistered release gate. On an untouched official-generator 1,100-case set it scores 49.5% over the ten ranking tiers and advances `highest_tier_above_90` to 5. The post-selection checked 1,100-case regression scores 50.1%. Tier 5 is 95% independent and 94% checked, consistent with EXP-521's five-seed 96.0% aggregate and 94% minimum.

The randomized-weight control scores 7.9% overall and 0% on Tier 5. The official pinned Python 3.12/PyTorch 2.5.1 container passes static analysis, preprocessing isolation, deterministic repeat, and the 300-second limit under no network, read-only mounts, four CPUs, and 8 GiB RAM. It consumes 36.9 seconds for inference and 75.49 seconds wall-clock end to end.

These are observed results, not a claim about the unavailable private seed. Candidate 004 does not meet the program's 85% full-overall completion rule.
