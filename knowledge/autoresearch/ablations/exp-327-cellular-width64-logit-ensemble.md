---
type: Experiment Preregistration
title: EXP-327 Cellular Width-64 Logit Ensemble
description: Paired endpoint-versus-logit-average evaluation for complementary Tier-5 recurrent checkpoints.
status: preregistered
tags: [autoresearch, cellular, width-64, ensemble, Tier-5]
timestamp: "2026-07-12T14:20:00+05:30"
---

# EXP-327: cellular width-64 logit ensemble

Evaluate frozen EXP-323, frozen EXP-324, and a 50/50 average of their output and quotient logits at every recurrent step. All three arms use identical fresh official-numeric width-64 cases from seed 20260715. The ensemble contains 199,648 trained parameters and doubles only the Tier-5 specialist forward computation; it uses no symbolic solver or output repair.

Require 99.8% ensemble transition, 93% rollout, at least one point over the better endpoint, and no more than 0.5% invalid rollouts. Passing authorizes five untouched Tier-5 seeds and a CPU headroom measurement. Failure closes two-checkpoint ensembling. No checked, consumed independent, or private answers may be read.
