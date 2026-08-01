---
type: Experiment Analysis
title: EXP-643 Tier-7 recurrent-depth stability result
description: Frozen two-seed mechanistic diagnosis rejects unstable recurrent depth as Candidate 005's primary width-256 failure cause.
status: completed
tags: [autoresearch, tier7, recurrent-depth, stability, mechanistic-analysis]
timestamp: "2026-07-18T06:05:00+05:30"
---

# EXP-643 Tier-7 recurrent-depth stability result

## Outcome

Reject the preregistered hypothesis that Candidate 005 primarily fails at width 256 because its additional recurrent visits become unstable or destroy earlier exact predictions. Both fresh seeds show a severe teacher-forced/autonomous gap, but neither shows any overthinking, a material failure-linked late-dynamics ratio, or an expanding random-direction Jacobian-gain proxy.

| Seed | Width | Transition exact | Rollout exact | Mean first error | Overthinking / instrumented errors | Late gain proxy |
|---|---:|---:|---:|---:|---:|---:|
| stability_a | 128 | 99.95% | 85% | 69.4 | 0% | 0.7649 |
| stability_a | 256 | 99.75% | 22% | 99.5 | 0% | 0.7644 |
| stability_b | 128 | 99.90% | 81% | 58.7 | 0% | 0.7945 |
| stability_b | 256 | 99.70% | 32% | 106.4 | 0% | 0.8026 |

The width-256 failed/successful late output-flip ratios are only 1.0305 and 1.0517; the corresponding hidden-state-delta ratios are 1.0051 and 1.0174. These are far below the frozen 1.5 association threshold. Width-256/width-128 spectral ratios are 0.9994 and 1.0101, below the 1.15 gate, and every late gain estimate is below 0.81 rather than above 1.10. Instrumented and ordinary predictions match exactly at every checked output, with maximum logit difference zero.

## Interpretation

The checkpoint still knows the local transition very well on ordinary held-out states—at least 99.7% exact at width 256—but rare state errors compound during 256 autonomous steps. The visit traces do not support the narrower story that the last visits erase exact intermediate answers. In fact, many transitions become exact only late in the visit schedule, so recurrent refinement remains useful even though it is insufficient.

The second width-128 control reaches only 81% rollout and misses its 85% retention floor. This prevents a positive cross-width causal comparison and also shows material seed/identity variation below the Tier-6 conditional candidate's routed result. It does not rescue the stability hypothesis: every diagnostic axis is independently negative on both seeds.

The supported next move is representational rather than a visit-scaling or spectral-regularization tweak. Candidate 005 hard-decodes the residue after every operand bit, discarding latent information. EXP-654 should test whether trained addressable registers and a latent trace can retain and retrieve useful execution state, while including decoded recurrence, parameter-matched wider recurrence, and a non-addressable workspace as matched controls. This architecture must be trained from ordinary initialization and prove memory causally active; no compiled arithmetic or interpreter semantics may enter inference.

## Resource and provenance

The full run completed in 1,523.32 wall seconds and 5,980.31 aggregate CPU seconds with 634,134,528 peak RSS. The calibrated soft budgets were 3,604 wall and 14,116 CPU seconds, and hard ceilings were 5,406 wall and 21,174 CPU seconds. No fidelity was reduced. MLflow run `83e287d7648c41e082f7e4bf5cfdd396` complements the [canonical run record](../../../experiments/runs/exp-643-tier7-recurrent-depth-stability-diagnostic/run.json), [metrics](../../../evals/results/exp-643-tier7-recurrent-depth-stability-diagnostic/metrics.json), and [full raw summary](../../../experiments/runs/exp-643-tier7-recurrent-depth-stability-diagnostic/formal-summary.json).

