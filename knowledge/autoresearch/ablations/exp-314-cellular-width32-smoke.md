---
type: Experiment Preregistration
title: EXP-314 Cellular Width-32 Smoke Continuation
description: Compute-bounded direct official-distribution continuation using 600 updates and a 2x hard-mining pool.
status: preregistered
tags: [autoresearch, cellular, width-32, smoke, Tier-4]
timestamp: "2026-07-12T11:50:00+05:30"
---

# EXP-314: cellular width-32 smoke continuation

Repeat EXP-313's width-24/32 alternating curriculum from frozen EXP-312, but use 600 updates and a 2x candidate pool so source evaluation, training, and final evaluation fit within ten minutes. Evaluate on the same fresh official-numeric identity construction, without reading any prior results from that construction.

Require 90% retained width-24 rollout; 99.7% width-32 transition; 90% width-32 rollout; at least three rollout points over the frozen source; and no more than 0.5% invalid rollouts. Passing authorizes a full confirmation or, if the absolute 93% gate is already met, the five-seed independent Tier-4 gate. Save before evaluation and stop at ten wall minutes.
