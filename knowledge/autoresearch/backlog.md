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
| 1 | EXP-200 | Width-16 transfer plus validated hard mining remains effective at width 32. | Very high: cheapest Tier-4 boundary test | <=8 CPU-min | Miss paired or absolute preregistered width-32 gate. |
| 3 | EXP-120 | Scheduled learned-state training closes teacher-forcing drift. | High for chained exactness | <=10 CPU-min A/B | Transition improves but autonomous rollout does not. |
| 4 | EXP-130 | ConvGRU recurrence improves width sharing over the full-canvas TCN. | High, higher implementation cost | <=12 CPU-min smoke | Worse exact rollout per CPU-second than matched TCN. |
| 5 | EXP-140 | Two-bit learned limbs halve recurrence without fatal transition complexity. | Medium-high runtime/scaling value | <=10 CPU-min smoke | Exact rollout loses more than runtime gains justify. |
| 6 | EXP-150 | Channel ablation reveals a prune/distill path with unchanged Tier-2 exactness. | Medium artifact/runtime value; no immediate rank gain | <=5 CPU-min analysis | No redundant channels under causal ablation. |

EXP-111 and EXP-112 promoted a validated Tier-3 specialist, but the frozen candidate remains far below the mandatory 85% overall threshold. EXP-200 tests the next fixed-width boundary. Regardless of its result, widths beyond 64 require a tensor representation that does not encode whole values in `int64`; the materially different width-independent track remains mandatory. The checked public benchmark is reserved for release regression after independent gates.
