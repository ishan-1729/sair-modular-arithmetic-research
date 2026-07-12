---
type: Experiment Preregistration
title: EXP-252 Replay-Heavy Failure Mining
description: Lower-intensity confirmation of EXP-251 using 25 percent mined rows and a lower learning rate.
status: completed
tags: [autoresearch, hard-mining, online-mining, replay, Tier-4]
timestamp: "2026-07-12T10:02:00+05:30"
---

# EXP-252: replay-heavy online failure mining

EXP-251 gained 0.96 rollout point but lost 0.01 transition point with 50% mined rows. Test whether a 25% mined / 75% uniform batch and half learning rate preserve the rollout effect without eroding transition accuracy. Use 200 updates, the same frozen source, fresh EXP-251 identity split and evaluation surface, and a matched uniform control.

Require nonnegative transition change, at least +1 rollout point, 99.8% transition, 93% rollout, and at most 0.5% invalid rollouts. Passing authorizes a larger fresh-split confirmation, not independent evaluation. Failing closes online loss-ranking selection. Stop at six wall minutes.

## Outcome

Rejected and closed. Transition was preserved (99.81% -> 99.815%), but rollout improved only 93.26% -> 93.62% (+0.36 point), far below the +1.00-point gate. The ordinary uniform control itself was stronger than earlier continuations, motivating a clean longer low-rate continuation on a new identity split rather than more online ranking.
