---
type: Experiment Preregistration
title: EXP-327 Cellular Width-64 Logit Ensemble
description: Paired endpoint-versus-logit-average evaluation for complementary Tier-5 recurrent checkpoints.
status: interrupted-policy-error
tags: [autoresearch, cellular, width-64, ensemble, Tier-5]
timestamp: "2026-07-12T14:20:00+05:30"
---

# EXP-327: cellular width-64 logit ensemble

Evaluate frozen EXP-323, frozen EXP-324, and a 50/50 average of their output and quotient logits at every recurrent step. All three arms use identical fresh official-numeric width-64 cases from seed 20260715. The ensemble contains 199,648 trained parameters and doubles only the Tier-5 specialist forward computation; it uses no symbolic solver or output repair.

Require 99.8% ensemble transition, 93% rollout, at least one point over the better endpoint, and no more than 0.5% invalid rollouts. Passing authorizes five untouched Tier-5 seeds and a CPU headroom measurement. Failure closes two-checkpoint ensembling. No checked, consumed independent, or private answers may be read.

## Outcome

Interrupted, not a comparison result. The process was terminated before the first arm emitted a metric because the preregistration mislabeled an arbitrary short limit as eight CPU minutes and execution incorrectly treated it as a wall-clock cap. This violated the binding dynamic resource-budget policy. No completed arm evidence or checkpoint was produced; MLflow run `fe99048e8c224f129b6f34f2dca50f25` remains marked killed as provenance. EXP-328 preserves the checkpoints, fresh-data boundary, paired design, gates, and full 1,000-transition/500-rollout fidelity under a measured-throughput wall budget with explicit aggregate-CPU accounting and per-arm caching.
