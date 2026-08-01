---
type: Evaluation Report
title: Candidate 003 Full-Set Results
description: Independent, checked, control, and isolated-container evidence for the Tier-4 hybrid.
status: completed
tags: [evaluation, candidate-003, full-set, sandbox]
timestamp: "2026-07-12T11:15:00+05:30"
---

# Candidate 003 results

The frozen three-file artifact scores 40.4% on the preregistered full independent 1,100-case set: Tier 1/2/3/4 are 100%/98%/99%/95%, and `highest_tier_above_90 = 4`. Only after this passed was the checked benchmark read; it scores 40.9% overall with Tier 4 at 98%.

Randomizing every floating-point tensor while preserving names, shapes, dtypes, and scales reduces checked overall accuracy to 7.7% and Tier 4 to 0%. This supports material dependence on trained parameters.

The pinned official Python 3.12/PyTorch 2.5.1 image reproduces 40.4% under native rootless Podman. The exact image ID is `0ff410bfb61c61d6a07b17b11337694a69910874cb59ab109b1128b3224a8423`; inference takes 7.9 seconds and container wall time is 34.3 seconds. The engine applies no network, read-only mounts, four CPUs, 8 GiB memory, and a 300-second inference timeout.

Canonical raw outputs are under [`experiments/runs/candidate-003-tier4-cellular-hybrid`](../../experiments/runs/candidate-003-tier4-cellular-hybrid/), and normalized metrics are in [`metrics.json`](../../evals/results/candidate-003-tier4-cellular-hybrid/metrics.json). The private official benchmark remains unavailable.
