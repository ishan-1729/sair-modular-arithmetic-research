---
type: Mechanistic Analysis
title: EXP-240 Causal Channel Analysis
description: Activation stratification and paired final-channel interventions on the strongest width-32 single-head checkpoint.
status: completed
tags: [autoresearch, mechanistic, activation, ablation, Tier-4]
timestamp: "2026-07-12T09:20:00+05:30"
---

# EXP-240: causal channel analysis

The frozen EXP-230 control scored 19,954/20,000 (99.77%) on fresh official-like transitions. Errors were concentrated in quotient class 1: 32 of 46 errors, versus 13 in class 0 and one in class 2. The original quotient-2 bottleneck is therefore no longer the limiting transition class.

Each final normalized hidden channel was zeroed independently on the same 5,000 generated cases. The paired reference scored 99.68%. No channel met the preregistered critical threshold of a 0.5-point overall or two-point quotient-2 drop. Channel 5 caused the largest overall drop (0.34 point); channel 38 caused the largest quotient-2 drop (0.879 point). Ablating channels 5, 38, 30, and 47 together reduced accuracy to 97.98%, a 1.70-point drop, showing nontrivial but distributed dependence.

Activation selectivity did not imply single-channel necessity. Several channels strongly separated quotient class, errors, or high-bit moduli, yet their individual interventions stayed below the causal gate. Narrow pruning, protection, or one-channel redundancy is therefore rejected. The follow-up should expose rare model-dependent transition failures across the distributed representation, using online failure mining with a uniform replay mixture to prevent forgetting.

Canonical metrics are in [`metrics.json`](../../../evals/results/exp-240-channel-causal-analysis/metrics.json). This analysis uses fresh generated identities and does not read consumed independent or checked-public answers.
