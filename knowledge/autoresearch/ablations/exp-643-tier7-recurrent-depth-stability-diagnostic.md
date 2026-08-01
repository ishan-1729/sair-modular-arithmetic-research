---
type: Experiment Preregistration
title: EXP-643 Tier-7 recurrent-depth stability diagnostic
description: Preregistered frozen-cell diagnostic of visit-wise stability and overthinking at widths 128 and 256.
status: completed
tags: [autoresearch, tier7, recurrent-depth, stability, mechanistic-analysis]
timestamp: "2026-07-18T05:20:00+05:30"
last_updated: "2026-07-18T06:05:00+05:30"
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

### Resolved budget before formal execution

Calibration completed from clean commit `57baa0041b85ce84bf2d47b751f9dc3b2bfbbcbb` in 55.5729 wall seconds and 188.86 aggregate process-CPU seconds, with peak RSS 527,339,520 bytes. The ignored raw calibration evidence has SHA-256 `f89cd5792cdc4f5a8abc73aa85be3b3e4a56ef755d609a9ff270e23ad9a64e70`. Component-wise scaling to both formal seeds projects 2,325 wall seconds and 9,287 aggregate CPU seconds without headroom.

The resolved soft budget is 3,604 wall seconds and 14,116 aggregate CPU seconds: 1.50 times the measured component projection plus 117 wall and 186 CPU seconds for setup and serialization. The hard runaway ceilings are 5,406 wall seconds and 21,174 aggregate CPU seconds. They provide an additional 50% beyond the soft allowance and do not alter the preregistered case counts.

After calibration, the runner received one non-computational guard so a positive-over-zero diagnostic ratio is serialized as canonical finite JSON rather than `Infinity`. Calibration used runner SHA-256 `273600fd7bdf313fd1827335a33a9542c31367a4e225963764fdd7f5f810da65`; formal uses `ca7ac1acf7e3a7f784492127dd6deda88fc5465f47f50a309c064e5e8007a8d1`. The model path, sampled work, instrumentation, timers, and throughput calculation are unchanged, so the measured calibration remains valid.

## Compliance and limitations

Python big integers are confined to fresh offline input generation, teachers, and evaluation. The frozen learned model receives only bit planes. No inference artifact changes, no public answers are read, and this mechanistic experiment cannot promote a submission by itself. A width-256 failure without the preregistered dynamic signature supports insufficient computation or representation just as plausibly as a training-distribution mismatch.

## Result

The full two-seed diagnostic completed without reduced fidelity and rejects stability as the primary supported cause. Width-256 transition/rollout exact accuracy is 99.75%/22% and 99.70%/32%. No instrumented error in either seed is an overthinking event. Failed/successful late flip ratios are 1.0305 and 1.0517, failed/successful late state-delta ratios are 1.0051 and 1.0174, and late random-direction gains stay below 0.81. The second width-128 control also misses retention at 81% rollout. None of the frozen axes reproduces, so EXP-644 and EXP-645 are not authorized from this diagnosis. Continue with the materially different learned-memory test in EXP-654. See the [analysis](../analysis/exp-643-tier7-recurrent-depth-stability.md) and [raw evidence](../../../experiments/runs/exp-643-tier7-recurrent-depth-stability-diagnostic/formal-summary.json).
