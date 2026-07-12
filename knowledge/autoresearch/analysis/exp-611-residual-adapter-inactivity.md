---
type: Mechanistic Analysis
title: EXP-611 Residual Adapter Inactivity
description: Post-run output and parameter-delta diagnosis of the exactly initialized residual branch.
status: completed
tags: [autoresearch, tier6, widening, optimization, diagnosis]
timestamp: "2026-07-13T02:25:00+05:30"
---

# EXP-611 residual adapter inactivity

## Observation

On 1,000 fresh width-128 transitions, the trained challenger and matched control produced zero differing bit predictions and zero differing quotient predictions. The adapter's local logits had maximum absolute magnitude `0.000787`, mean absolute magnitude `0.000314`, and RMS magnitude `0.000337`; quotient corrections were similarly tiny. The challenger's retained base differed from the independently trained control by only `1.75e-8` RMS, despite 83,568 nonzero floating-point deltas among 99,824 parameters.

The adapter's local-head weight norm was `0.000233` and its final quotient-head weight norm was `0.000194`. These observations explain the identical autonomous evaluation: the zero-output branch acquired gradients, but the inherited `5e-6` low continuation rate did not move it into a behaviorally meaningful regime in 200 updates.

## Interpretation and intervention

The causal capacity question remains unresolved because the added parameters were effectively dormant. The supported follow-up is not a longer identical run. Test optimizer parameter groups: retain `5e-6` for the trained base and apply a substantially larger preregistered rate to the newly initialized adapter. Require nontrivial adapter-output magnitude or prediction changes at a cheap activation rung, alongside exact accuracy and retention gates. If the branch activates without improving rollout, close this residual-capacity formulation.

This is generated selection-set diagnosis. It does not use checked or private answers.
