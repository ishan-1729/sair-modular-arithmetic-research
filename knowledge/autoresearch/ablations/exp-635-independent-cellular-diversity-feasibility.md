---
type: Experiment Preregistration
title: EXP-635 Independent Cellular Diversity Feasibility
description: Paired feasibility test of whether a separately initialized cellular learner supplies exploitable exact-rollout diversity.
status: completed
tags: [autoresearch, cellular, independent-seed, diversity, routing, paired-evaluation]
timestamp: "2026-07-13T18:05:00+05:30"
---

# EXP-635 independent cellular diversity feasibility

## Hypothesis and evidence basis

EXP-606 found only 2.5 points of oracle union from two checkpoints in one continuation lineage, while EXP-632 found that multiple output heads sharing one recurrent core regress accuracy and do not beat randomized heads. Those results do not test independent learned algorithms: both interventions preserve almost all recurrent computation. EXP-635 asks whether a fresh initialization and independent example order, trained with the already successful width-shared cellular prescription, creates meaningfully different exact-rollout errors.

This is a small-width feasibility rung. It cannot establish Tier-6 or full-benchmark performance. Its purpose is to decide whether training a second full Tier-6 cellular lineage has enough expected value to justify the much larger compute cost.

## Frozen control, challenger, and data boundary

The control is the immutable 99,824-parameter EXP-633 sequential checkpoint with SHA-256 `9b94b7eff85caf779598d9fff13a6598db3c195ffd66408545d23a1745af1b6c`. The challenger uses the identical 92-channel, two-cycle `SharedDilatedConvGRU`, the same width-8/16 training prime identities, 2,400 alternating-width updates, batch size, optimizer, schedule, loss, clipping, and label generator. Only initialization seed and example order change.

EXP-633 used the first 128 held-out width-16 identities from its deterministic prime stream. EXP-635 expands that same stream without changing its first 512 training identities, verifies the original held-out prefix, and uses only the next 128 identities for evaluation. These identities are disjoint from both training and the previous evaluation slice. The paired 600 cases are generated once and supplied unchanged to control and challenger. Checked-public, independent full-seed, official, and private answers remain unread.

Python bigint multiplication and modulus are restricted to offline labels and evaluation truth. Both inference arms receive bit tensors and produce bit logits through trained parameters.

## Paired evaluation and routing rule

For every case, retain each arm's final prediction, exact correctness, invalid-residue flag, and the minimum absolute output-logit margin over every bit and autonomous step. Report individual accuracy, intersection, control-only and challenger-only successes, disagreement, oracle union, and Wilson intervals. Oracle union is diagnostic only.

The sole inference selection rule is frozen before evidence: choose the challenger only when its trajectory minimum absolute logit margin is strictly higher than the control's; ties select control. It uses neither the expected answer nor any evaluation-fitted threshold. Repeat the first 20 cases exactly for both trained arms and compare predictions and margins. Evaluate a separately initialized untrained model on 100 matched cases as a trained-weight-dependence control.

Advance to a full-width independent lineage only if every gate passes:

- challenger exact accuracy is at least 80%;
- paired oracle union is at least 92% and at least four points above the better individual arm;
- frozen target-free routing is at least 1.5 points above the better individual arm;
- trained challenger exceeds the untrained model by at least 50 points;
- each trained arm's invalid rate is at most 5%;
- paired predictions and minimum margins are exactly deterministic;
- the measured paired 600-case width-16 evaluation takes at most 120 wall seconds.

Failure of the oracle gates closes independent-seed ensembling at this rung. Oracle success without routing success preserves diversity as a research observation but does not authorize a Tier-6 ensemble; a later router would require a separate training/validation/test preregistration. Passing authorizes one dynamically budgeted Tier-6 independent-lineage experiment, not a candidate promotion or benchmark evaluation.

## Dynamic resource-budget protocol

Before formal execution, profile 30 representative alternating-width challenger updates and 20 training-identity rollouts for the frozen control, partially trained challenger, and untrained dependence control. Project all 2,400 updates, 600 paired cases, the 100-case randomized control, two 20-case trained-arm determinism repeats, setup, serialization, and tracking. Record wall and aggregate CPU separately; add safety headroom and a hard runaway ceiling. Preserve model, optimizer, scheduler, sampler, and RNG state every 100 steps and before enforcing the wall ceiling.

Calibration may change only resource and resumability fields. It may not change the model, data identities, seeds, training work, case counts, routing rule, or gates. Formal evaluation fidelity must not be reduced to fit an arbitrary wall cap. A valid frozen EXP-633 checkpoint is reused by hash; its prior predictions are not reused because EXP-635 uses a new paired case set.

The completed training-identity-only calibration measured 30 alternating-width updates in 20.904 wall seconds and 77.974 aggregate CPU seconds. Twenty width-16 rollout cases took 0.404/1.520 wall/CPU seconds for the frozen control, 0.489/1.836 for the partially trained challenger, and 0.467/1.760 for the untrained control. This projects the full 2,400-update training workload at 1,672.293 wall / 6,237.933 aggregate CPU seconds and every formal evaluation/control/repeat at 30.912 wall / 116.215 aggregate CPU seconds.

With a separate 120-second setup, serialization, and tracking allowance, the frozen expected budget is 1,824 wall seconds and 114 aggregate CPU minutes. The soft wall budget is 2,520 seconds; the hard runaway ceiling is 3,660 wall seconds, with a distinct four-thread upper bound of 244 aggregate CPU minutes. External calibration timing was 32.63 wall seconds, 93.60 user seconds, and 2.65 system seconds at 294% CPU, with 749,184 KiB maximum RSS. The calibration JSON hashes to `22480ad1e80a6fdde17bdb23c60d1a7c73e6c6af3f679356c84946a03ee7da4e`. No evidence identities were read and no scientific field changed.

## Limitations

Raw margin scales can differ between independently trained models, so failure of the fixed selector does not prove that no learned router could work. Conversely, a high small-width oracle union may disappear at width 128. Both limitations are captured by requiring a separate full-width preregistration after this feasibility rung.

## Result and decision

All gates pass on the untouched 600-case paired set. Control scores 518/600 (86.33%) and the independent challenger 520/600 (86.67%). Their oracle union is 572/600 (95.33%), 8.67 points above the better arm, with 52 control-only and 54 challenger-only successes. The frozen target-free minimum-margin rule selects the challenger on 310 cases and scores 558/600 (93.00%), a 6.33-point gain. Randomized weights score 0/100; validity, exact determinism, and the 13.68-second paired runtime pass.

The complete run used 1,705.64 wall seconds and 6,355.02 aggregate CPU seconds, below the 1,824-second expected budget and far below both ceilings, with no fidelity reduction. This advances one separately budgeted Tier-6 independent-lineage confirmation; it does not promote a candidate or establish full-benchmark accuracy.
