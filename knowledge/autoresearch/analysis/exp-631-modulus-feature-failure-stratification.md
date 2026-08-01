---
type: Failure Stratification Analysis
title: EXP-631 Modulus and Operand Feature Failure Stratification
description: Training-derived and prime-disjoint validation test of target-free feature groups for Tier-6 greedy rollout failures.
status: completed
tags: [autoresearch, tier6, failure-analysis, group-robustness, negative-result]
timestamp: "2026-07-13T14:55:00+05:30"
---

# EXP-631 modulus and operand feature failure stratification

## Question and evidence boundary

Do stable, target-free modulus or operand feature groups concentrate frozen EXP-609 greedy width-128 failures strongly enough to justify group-balanced recovery training?

The analysis reuses the preserved EXP-627 training cache (1,200 cases, 162 failures, 461 prime identities) to define thresholds and rank groups. Only after fixing those definitions does it test the prime-disjoint EXP-627 validation cache (200 cases, 37 failures, 60 prime identities). The selection cache, checked benchmark, independent seeds, and private answers are not read.

## Candidate groups

Twenty-six groups cover exact modulus bit length and the lower/upper training quartiles of modulus magnitude relative to width, within-bit-length mantissa, modulus Hamming-weight fraction, left/modulus, right/modulus, minimum and maximum operand ratios, normalized operand gap, and normalized operand sum. Continuous thresholds are derived from training only. A group must contain at least 120 training cases and 20 validation cases. The authorization gate requires at least twofold failure enrichment over the split-wide rate in both training and validation.

## Result

Training failure is 162/1,200 (13.5%); validation failure is 37/200 (18.5%). Twenty-one candidate groups meet training support.

The strongest training group is `left / modulus <= 0.2457958784`. It contains 301 training cases with 53 failures (17.61%, 1.304x enrichment, Fisher exact p=0.0192). With the frozen threshold, validation contains 49 cases and 13 failures (26.53%, 1.434x enrichment, Fisher p=0.1365). This direction is weak and not statistically clear on validation.

The next strongest training groups are low modulus Hamming fraction (1.123x training, 0.998x validation) and high right/modulus (1.107x training, 0.918x validation). Exact modulus bit-length groups are at most 1.101x in training and do not replicate directionally.

No candidate reaches twofold enrichment in training, and none satisfies the replicated gate. Therefore the evidence does not support hard-prime or public-feature group-balanced training. Do not consume new model-selection data or tune thresholds on validation. The remaining error tail is not usefully localized by these coarse legal input features.

## Decision

Close EXP-631 without training. Preserve the caches and analysis as negative evidence. A subsequent intervention should obtain diversity from learned computation itself rather than reweighting modulus/operand groups; the next planned test is a shared-core, diverse learned output-head ensemble whose inference cost is small relative to the recurrent core.

