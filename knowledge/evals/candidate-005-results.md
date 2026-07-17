---
type: Evaluation Result
title: Candidate 005 Evaluation Results
description: Five-seed Tier-6, full independent, checked, randomized-control, and pinned-sandbox evidence for Candidate 005.
status: completed
tags: [evaluation, candidate, Tier-6, exact-accuracy, conditional-compute]
timestamp: "2026-07-18T05:30:00+05:30"
---

# Candidate 005 evaluation results

The exact frozen artifact passes its complete preregistered intermediate release gate. On the complete independent official-generator seed it scores 59.3% overall and advances `highest_tier_above_90` to 6. The post-selection complete checked regression scores 59.5%. Exact per-tier results are:

| Evaluation | Tier 1 | Tier 2 | Tier 3 | Tier 4 | Tier 5 | Tier 6 | Tiers 7–10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Independent | 100% | 99% | 97% | 97% | 94% | 98% | 2% each |
| Checked | 100% | 99% | 100% | 98% | 94% | 96% | 2% each |

Tier 6 scores 98%/96%/95%/99%/99% over five frozen seeds, or 487/500 (97.4%) with a 95% minimum. The local gate also establishes exact 500/500 lower-tier prediction equivalence to Candidate 004, exact repeated determinism, valid outputs, clean static analysis, and randomized Tier-6 collapse to 2%.

The pinned official Python 3.12/PyTorch 2.5.1 Podman sandbox uses no network, read-only mounts, four CPUs, and 8 GiB RAM. It passes manifest, static, preprocessing-isolation, and deterministic checks, completes inference in 156.0 seconds, and completes the container run in 178.41 seconds. No evaluation fidelity was reduced to fit a short wall cap; the local matched gate used its separately measured wall and aggregate-CPU budget.

Canonical evidence:

- [Run record](../../experiments/runs/candidate-005-tier6-conditional-commutative-integration/run.json)
- [Metrics](../../evals/results/candidate-005-tier6-conditional-commutative-integration/metrics.json)
- [Independent full result](../../experiments/runs/candidate-005-tier6-conditional-commutative-integration/independent-full.json)
- [Checked full result](../../experiments/runs/candidate-005-tier6-conditional-commutative-integration/checked-full.json)
- [Sandbox provenance](../../experiments/runs/candidate-005-tier6-conditional-commutative-integration/sandbox-run.json)

These are observed local and official-compatible results, not a private-official score. Candidate 005 fails the binding completion rule because neither checked nor independent overall accuracy reaches 85%; complete five-seed overall confirmation would be wasteful while the frozen artifact intentionally falls back above Tier 6.
