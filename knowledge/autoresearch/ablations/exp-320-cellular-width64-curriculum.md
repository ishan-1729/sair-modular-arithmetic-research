---
type: Experiment Preregistration
title: EXP-320 Cellular Width-64 Curriculum
description: Bigint-backed progressive width curriculum and zero-shot Tier-5-length evaluation for the validated cellular model.
status: preregistered
tags: [autoresearch, cellular, bigint, width-64, Tier-5]
timestamp: "2026-07-12T12:30:00+05:30"
---

# EXP-320: cellular width-64 curriculum

Continue frozen Tier-4 cellular checkpoint `a07bcfc9b7f990a7a5b62001fb40ab4533c4776d9317099808bd60f56b0be241` on a cyclic 32/40/48/56-bit curriculum. Python big integers generate primes, exact offline labels, hard-mining indicators, and evaluation truth; before every model call, values are converted to bit tensors. The trained inference path remains a width-independent convolutional recurrence with no exact arithmetic.

Use 800 updates, batch 128, and a 2x candidate pool. Evaluate source zero-shot width 64, retained width 32, learned width 56, and trained zero-shot width 64 on disjoint identities. Require 90% retained width-32 rollout; 85%/10% width-56 transition/rollout; and 55%/0.2% width-64 transition/rollout with no more than 50% invalid rollouts. Passing authorizes direct width-64 training. No checked, consumed independent, or private answers may be read. Save before evaluation and stop at twelve wall minutes.
