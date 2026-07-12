---
type: Experiment Preregistration
title: EXP-324 Width-64 Cellular Second Stage
description: Fresh-identity lower-rate continuation of the diverse Tier-5 checkpoint.
status: completed
tags: [autoresearch, cellular, width-64, continuation, Tier-5]
timestamp: "2026-07-12T13:30:00+05:30"
---

# EXP-324: width-64 cellular second stage

Continue frozen EXP-323 checkpoint `f948b75d5578860ff16520ecd9daa2de339fd6af67b126d18b06c3fbf76f43f2` on a completely fresh 2,048-prime training set using seed 20260713 and learning rate `2e-4`. Keep the same width-56/64 curriculum, 400 updates, and evaluation fidelity.

Require 90% retained width-56 rollout; 99.8% width-64 transition; 92% width-64 rollout; at least three points over the paired frozen source; and at most 0.5% invalid rollouts. Reaching at least 93% authorizes five untouched Tier-5 seeds; a 92-93% pass authorizes a higher-fidelity confirmation. Failure closes this continuation formulation. No checked, consumed independent, or private answers may be read.

## Outcome

Rejected narrowly. Width-64 transition/rollout reaches 99.88%/91.867%, width-56 rollout is 94%, and invalid rollouts are 0.467%. The absolute rollout gate is missed by 0.133 point and the gain gate by 1.333 points. Close further continuation. Test a preregistered interpolation of EXP-323 and EXP-324 trained weights on fresh identities; this adds no inference cost.
