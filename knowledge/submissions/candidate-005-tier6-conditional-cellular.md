---
type: Model Card
title: Candidate 005 Tier-6 Conditional Cellular Hybrid
description: Submission-ready Tier-2/3 TCN and Tier-4/5/6 cellular ensemble with target-free conditional commutative Tier-6 inference.
status: incumbent-intermediate
tags: [submission, candidate, Tier-6, cellular, conditional-compute, trained-model]
timestamp: "2026-07-18T05:30:00+05:30"
---

# Candidate 005: Tier-6 conditional cellular hybrid

Candidate 005 retains Candidate 004's Tier-2 through Tier-5 specialists unchanged and adds the frozen 99,824-parameter EXP-609 width-128 cellular checkpoint. Tier 6 runs the original operand orientation for every row, routes exactly the lowest-minimum-margin 40% of the evaluator batch through the commuted orientation, and keeps the orientation with the higher learned trajectory margin. The route, score, fraction, checkpoint, and tie breaks were frozen before the final independent and checked evaluations.

The exact three-file artifact contains 470,049 learned parameters and totals 1,915,885 bytes:

- `manifest.json`: `ee49f91b8a1a4aa2cde418ebec59ea1890d93fda6dc72338989c0aa3308e7c83`
- `model.py`: `e3d4aa9ff3e22dde516d261b099826622fa37d0688fae47c07ff95fe876e9477`
- `weights.safetensors`: `2f6ac3199d0fd9e16afae165c2d4a16919e6c8bf4545363a6f12d077279d27ca`

| Evaluation | Overall | Tier 1 | Tier 2 | Tier 3 | Tier 4 | Tier 5 | Tier 6 | Highest >90% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full independent 1,100 | 59.3% | 100% | 99% | 97% | 97% | 94% | 98% | 6 |
| Checked release regression | 59.5% | 100% | 99% | 100% | 98% | 94% | 96% | 6 |

Across the five preregistered Tier-6 seeds, exact accuracy is 98%/96%/95%/99%/99%: 487/500 (97.4%) aggregate with a 95% minimum. Randomizing only the Tier-6 floating tensors reduces its seed-2 exact accuracy to 2%. Candidate 005 exactly matches Candidate 004 on all 500 independent seed-1 Tier 1–5 predictions.

The pinned official isolated container passes manifest/static checks, preprocessing isolation, determinism, and the 300-second constraint. It completes full inference in 156.0 seconds and the whole container run in 178.41 seconds on four CPUs with 8 GiB RAM, below the preregistered 285-second promotion ceiling. See the [complete evaluation record](../evals/candidate-005-results.md).

The adapter uses challenge-sanctioned pairwise operand reduction and Python big integers only to encode legal input bit planes. Trained recurrent weights produce every candidate residue; learned margins only control target-free routing and selection. There is no hand-coded modular multiplication, long division, exact reduction, CRT, benchmark lookup, compiled solver, or output repair.

This is an intermediate ranking promotion, not program completion. Tiers 7–10 retain the deterministic fallback, checked and independent overall accuracy remain about 59%, five complete independent sets therefore cannot meet 85%, and no legitimate private-official result is available. The active research frontier is Tier 7 at width 256.
