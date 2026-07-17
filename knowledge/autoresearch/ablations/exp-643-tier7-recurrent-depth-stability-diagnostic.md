---
type: Experiment Preregistration
title: EXP-643 Tier-7 recurrent-depth stability diagnostic
description: Preregistered frozen-cell diagnostic of visit-wise stability and overthinking at widths 128 and 256.
status: calibration-pending
tags: [autoresearch, tier7, recurrent-depth, stability, mechanistic-analysis]
timestamp: "2026-07-18T06:10:00+05:30"
last_updated: "2026-07-18T06:10:00+05:30"
---

# EXP-643 Tier-7 recurrent-depth stability diagnostic

## Question and frozen control

Candidate 005 uses the same 470,049-parameter cellular checkpoint at every supported width, but its dilation schedule grows with width. EXP-643 asks whether its Tier-7 failure is specifically associated with unstable or destructive extra recurrent computation rather than merely insufficient learned arithmetic. It changes no weights and consumes no checked-public or previously consumed independent cases.

The frozen checkpoint is `.scratch/exp-609-tier6-straight-through-confirmation/straight_through_continuation/weights.safetensors`, SHA-256 `cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd`. Width 128 is the retention control and width 256 is the Tier-7 diagnostic. Both use hidden size 92, two alternating dilation cycles, four Torch threads, the same generator construction, and separately generated held-out prime identities.

## Untouched streams and fidelity

The calibration stream is seed `20261046`; it may measure throughput and implementation parity only. The formal streams are preregistered now as `stability_a = 20261044` and `stability_b = 20261045`. Each seed constructs 512 unused training-side prime identities and 128 held-out evaluation identities per width from the top eight bits of the width band. Only the held-out identities are evaluated. Split, transition, rollout, instrumented-rollout, and spectral offsets are frozen in the resolved config.

For each formal seed and width, run 2,000 exact one-step transitions, 100 complete autonomous rollouts, 16 separately generated complete instrumented rollouts, and one transition for the Jacobian proxy. The instrumented rollout logs every recurrent visit at every operand-bit step. This is a diagnostic full rung, not a release evaluation and not evidence toward the five-seed 1,100-case completion gate.

## Measurements

The normal rollout records exact final accuracy, all-steps exactness, invalid-residue incidence, and the first erroneous operand-bit position for every case. The instrumented path must reproduce the ordinary model's final thresholded output exactly before its evidence is accepted. At every visit it records per-example hidden-state and update deltas, normalized-state mean and standard deviation, reset/update gate means and saturation, candidate-state RMS, local-logit delta, output-bit flip rate, absolute-logit margin, and exactness against the offline teacher transition.

“Overthinking” is frozen as a transition whose prediction is exactly correct after at least one non-final visit but incorrect after the final visit. “Late repair” is the converse descriptive statistic. Failed-versus-successful late dynamics use the final quarter of visits and are stratified by final transition exactness; they are not inferred from labels at submission time.

The recurrent-Jacobian diagnostic is the maximum `||Jv|| / ||v||` over fixed random unit directions at four preregistered visit fractions. It is a tractable random-direction lower-bound proxy for local gain, not the spectral norm and not proof of instability on its own. Exact Jacobian construction is deliberately excluded because a width-256 hidden state has 23,552 dimensions and would add disproportionate offline compute without improving the decision.

## Frozen decision rule

Width-128 retention is descriptive context and is considered reproduced only when every formal seed reaches at least 99.5% transition accuracy and 85% complete-rollout accuracy. Instability authorizes exactly one matched intervention only when one of these rules holds on both formal seeds:

1. the width-256 overthinking fraction is at least 10% of erroneous instrumented transitions and at least twice the width-128 fraction; or
2. failed width-256 transitions have at least 1.5 times the successful-transition late flip rate or late hidden-state delta, and the late width-256 Jacobian-gain proxy exceeds 1.10 and is at least 1.15 times the matched width-128 value.

The spectral proxy cannot authorize an intervention by itself. If neither rule reproduces on both seeds, reject recurrent-depth instability as the primary supported cause and move to addressable learned memory or another materially different architecture. Correlation authorizes an A/B test; it does not establish causality. EXP-644 or EXP-645 must then cause both improved stability and exact width-256 rollout under matched training to support the mechanism.

## Resource-budget protocol

The formal budget is intentionally unset before calibration. Run the complete calibration workload at both widths on seed `20261046`, measure wall time and aggregate process CPU separately for transition, rollout, full instrumentation, and Jacobian proxy work, then scale each component to the frozen formal counts and two seeds. Add the configured 1.50 throughput multiplier plus explicit setup/serialization headroom. The hard runaway ceilings must be larger than the resulting soft budgets and documented in this config and note before formal execution; the runner refuses formal mode while calibration remains pending.

Calibration has an 1,800-second wall and 7,200 aggregate-CPU-second emergency ceiling. These are runaway guards for a small representative workload, not a formal comparison cap. Formal evaluation fidelity will not be reduced to fit a short wall clock. Valid checkpoint, generator, and evaluation caches are reused by their hashed inputs; each completed width/seed cell is written to recoverable partial evidence before the next begins.

## Compliance and limitations

Python big integers are confined to fresh offline input generation, teachers, and evaluation. The frozen learned model receives only bit planes. No inference artifact changes, no public answers are read, and this mechanistic experiment cannot promote a submission by itself. A width-256 failure without the preregistered dynamic signature supports insufficient computation or representation just as plausibly as a training-distribution mismatch.

