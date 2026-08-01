---
type: Evaluation Report
title: Candidate 002 Full Evaluation Results
description: Independent, checked, pinned-runtime, and weight-control evidence for the Tier-3 hybrid.
status: completed
tags: [evaluation, Tier-3, candidate-002, controls]
timestamp: "2026-07-12T08:15:00+05:30"
---

# Candidate 002 results

The frozen width-16 checkpoint first passed its preregistered heldout gate at 99.908% transition and 98.895% rollout accuracy. It then scored 499/500 across five untouched official-generator Tier-3 seeds before integration.

The integrated three-file candidate scored 31.0% across scored tiers on a full independently seeded official 1,100-case set: Tier 1 100%, Tier 2 97%, Tier 3 99%, and `highest_tier_above_90 = 3`. The raw result is [`independent-full.json`](../../experiments/runs/candidate-002-tier3-hybrid/independent-full.json).

After selection, the checked 1,100-row release regression scored 31.3%, with Tier 3 at 100%. Static analysis was clean, preprocessing isolated, output deterministic, and host inference 0.37 seconds. The exact pinned Python 3.12/PyTorch 2.5.1 runtime reproduced the independent result and inferred in 2.1 seconds. See [`checked-full.json`](../../experiments/runs/candidate-002-tier3-hybrid/checked-full.json) and [`pinned-runtime-full.json`](../../experiments/runs/candidate-002-tier3-hybrid/pinned-runtime-full.json).

Randomized weights scored 7.9% overall, 1% Tier 2, and 2% Tier 3, with no tier at or above 90%. The final hybrid also passed the unmodified pinned official container through native rootless Podman 4.9.3 on Ubuntu 24.04 WSL2 with networking disabled, a read-only root and submission mount, four CPUs, 8 GiB memory, and the 300-second timeout. Container inference completed in 1.7 seconds on the checked set and 1.8 seconds on the independent set, reproducing both results exactly. The image ID is `7c195652b14c7470b636194e63679e5d1eec1587471e177665df27eae67d4399`.
