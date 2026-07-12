---
type: Research Backlog
title: Autoresearch Hypothesis Backlog
description: Ranking-aware, compute-bounded queue of falsifiable model experiments.
status: active
tags: [autoresearch, backlog, hypotheses]
timestamp: "2026-07-12T02:20:00+05:30"
---

# Autoresearch hypothesis backlog

| Priority | ID | Hypothesis | Expected ranking value | Initial cost | Falsification criterion |
|---:|---|---|---|---:|---|
| 1 | EXP-111 | Full-budget transferred hard mining crosses 90% heldout rollout. | Very high: unlocks independent Tier-3 gate | <=10 CPU-min | Below 99.5% transition, 90% rollout, or above 1% invalid rollouts. |
| 3 | EXP-120 | Scheduled learned-state training closes teacher-forcing drift. | High for chained exactness | <=10 CPU-min A/B | Transition improves but autonomous rollout does not. |
| 4 | EXP-130 | ConvGRU recurrence improves width sharing over the full-canvas TCN. | High, higher implementation cost | <=12 CPU-min smoke | Worse exact rollout per CPU-second than matched TCN. |
| 5 | EXP-140 | Two-bit learned limbs halve recurrence without fatal transition complexity. | Medium-high runtime/scaling value | <=10 CPU-min smoke | Exact rollout loses more than runtime gains justify. |
| 6 | EXP-150 | Channel ablation reveals a prune/distill path with unchanged Tier-2 exactness. | Medium artifact/runtime value; no immediate rank gain | <=5 CPU-min analysis | No redundant channels under causal ablation. |

EXP-100 established a learnable width-16 surface, EXP-105/106 validated learned transfer, and EXP-110 used the diagnosed quotient-2 error to reach 87.41% rollout. EXP-111 is the authorized full-budget gate. The checked public benchmark is reserved for release regression after independent gates.
