---
type: Experiment Analysis
title: EXP-647 operand-swap endpoint consistency
description: Full-fidelity matched evidence rejects confidence-routed commutative endpoint consistency for single-orientation Tier-7 improvement.
status: completed-rejected
tags: [autoresearch, tier7, commutativity, consistency, negative-result]
timestamp: "2026-07-20T01:45:47+05:30"
---

# EXP-647 operand-swap endpoint consistency

## Decision

Reject the frozen `0.10` confidence-consistency formulation. The challenger retains width 64 and reproduces deterministically, but it does not distill the known two-orientation advantage into the original trajectory. No Tier-7 or candidate promotion follows, Candidate 005 remains the Tier-6 incumbent at 59.5% checked and 59.3% independent overall, and the 85% goal remains unmet.

## Matched result

Both 99,824-parameter arms started from source SHA-256 `cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd`, used seed `20261057`, 240 updates, the same width schedule and examples, and full frozen evaluation. Original-orientation exact rollout was:

| Width | Control | Consistency | Gain |
|---:|---:|---:|---:|
| 64 | 288/300 (96.000%) | 289/300 (96.333%) | +0.333 points |
| 128 | 234/300 (78.000%) | 229/300 (76.333%) | -1.667 points |
| 256 | 55/200 (27.500%) | 50/200 (25.000%) | -2.500 points |

Challenger transition accuracy was 999/1,000 (99.900%), 2,992/3,000 (99.7333%), and 2,987/3,000 (99.5667%) at widths 64/128/256. Original invalid-rollout rates were 0.333%, 0.333%, and 1.5%. The higher-margin selected path scored 299/300 (99.667%), 281/300 (93.667%), and 80/200 (40.0%); offline oracle union scored 299/300 (99.667%), 286/300 (95.333%), and 95/200 (47.5%). Exact repeated predictions, margins, and invalid flags passed on 20 width-128 cases.

Only width-64 retention and determinism passed. Both accuracy-gain gates, both wider transition gates, validity, and runtime failed. Challenger width-256 original inference took 262.627 seconds for 200 cases, projecting to 131.313 seconds per 100 versus the frozen 100-second gate. Because the control came from a valid earlier cache and the challenger ran later, cross-run control/challenger timing is not a causal speed comparison; only this absolute challenger runtime gate is interpreted.

Paired original-arm exactness confirms that the intervention did not hide a broad improvement: at widths 64/128/256, control-only versus challenger-only cases were 0/1, 5/0, and 6/1 (300/300/200 cases; exact two-sided McNemar p-values 1.0, 0.0625, 0.125). Full per-case evidence is preserved in the canonical paired artifact.

## Resources and provenance

Retry 3 completed full fidelity in 5,141.599 wall seconds and 20,465.856 aggregate process-CPU seconds. Including the valid cached control and conservative charges for three host-lifecycle-interrupted challenger attempts, cumulative charges are exactly 20,337.479420917996 wall and 95,775.856237935 aggregate-CPU seconds, leaving 7,718.520579082004 wall and 15,764.143762065 aggregate-CPU seconds under the committed soft budget. No fidelity was reduced. The control cache is valid and frozen; interrupted attempts produced no scientific result or resumable checkpoint.

The formal source commit is `94a3f63d429ea5024852197bbf02eeedac25ae5c`. Challenger checkpoint SHA-256 is `b782c46b9ebb2570a309c89950b83a16c9be7a653d81dbbc650af3730b026b44`. Canonical raw evidence is [formal-summary.json](../../../experiments/runs/exp-647-operand-swap-consistency/formal-summary.json) and [paired-evidence.json](../../../experiments/runs/exp-647-operand-swap-consistency/paired-evidence.json); compact metrics are [metrics.json](../../../evals/results/exp-647-operand-swap-consistency/metrics.json).

## Limitation and next action

This is one matched fresh generated identity-disjoint training seed, adequate to reject the frozen formulation but not a general statement about every symmetry-aware loss. Checked, prior independent-full, official, and private answers were not read. The PM must now select the next hypothesis from existing evidence; this result does not preregister or authorize one automatically.
