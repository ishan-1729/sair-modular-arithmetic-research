---
type: Experiment Preregistration
title: EXP-636 Independent Cellular Tier-5 Bootstrap
description: Staged progressive curriculum testing whether the EXP-635 independent lineage reaches width-64 readiness with exploitable paired diversity.
status: preregistered
tags: [autoresearch, cellular, independent-lineage, curriculum, autonomous-recovery, routing]
timestamp: "2026-07-13T18:41:00+05:30"
---

# EXP-636 independent cellular Tier-5 bootstrap

## Hypothesis and rationale

EXP-635 establishes real small-width diversity: two separately initialized width-16 cellular models score 86.33% and 86.67% individually, while frozen target-free minimum-margin routing reaches 93.00% and oracle union reaches 95.33%. A direct width-128 continuation would nevertheless be unfair. The incumbent lineage received progressive width-24/32/56/64 training and autonomous-state recovery before Tier 6, whereas the independent checkpoint has learned only widths 8 and 16.

EXP-636 therefore tests the necessary bridge rather than understating the independent hypothesis. It continues the immutable EXP-635 checkpoint through a staged width-16-to-64 curriculum modeled on the successful historical cellular lineage, then compares it with the frozen Tier-5 incumbent on fresh paired width-64 cases. Passing authorizes a Tier-6 independent-lineage run; it is not itself a candidate promotion or full-benchmark evaluation.

## Frozen sources and trained-parameters boundary

The independent source is EXP-635 checkpoint SHA-256 `3cce3160c405c68fd71223ed31a6a9be0995a7162de9e23d231c8b55e0e22dc4`. The comparison control is the frozen incumbent-lineage width-64 recovery checkpoint SHA-256 `050daf0a2d3b964ed9a58522c18d896f94a0d950722d0c34296b0718c03676f7`. Both use the same 99,824-parameter `SharedDilatedConvGRU` architecture.

Exact bigint arithmetic is confined to offline sampling, hard-example selection, labels, and evaluation truth. Submission-shaped inference remains tensor-native and trained. Checked-public, full independent-seed, official, and private answers are not read.

## Staged curriculum

The independent model completes every declared stage without evidence-dependent early promotion:

1. 1,200 hard-mined transition updates alternating widths 16/24/32 at batch 128 and learning rate `1e-3`;
2. 800 updates alternating widths 32/40/48/56 at `5e-4`;
3. 800 updates alternating widths 56/64 at `2e-4`;
4. 400 width-56/64 updates at `1.5e-4`, with every fourth update replaced by a four-step detached learner-state recovery batch at width 64.

All stages use 2,048 training prime identities per width, a disjoint 128-identity test pool, cosine decay, the historical hard-mining mixture, and fresh seed `20260832`. Each stage has its own optimizer/scheduler and a content-addressed, exactly resumable checkpoint containing model, optimizer, scheduler, sampler, selector, Python, NumPy, and Torch RNG state.

## Paired evaluation and gates

Evaluate independent width-32 retention on 300 cases. On 600 fresh paired width-64 rollouts, retain final predictions, correctness, invalidity, and minimum absolute logit margin across every bit and autonomous step for the incumbent and independent model. Report arm accuracy, Wilson intervals, intersection, arm-only successes, disagreement, oracle union, and the frozen rule that selects the independent model only when its raw trajectory minimum margin is strictly higher. Ties select the incumbent. No threshold is fit on evaluation cases.

Also evaluate 3,000 independent-model width-64 transitions, the uncontinued EXP-635 source on the first 100 paired width-64 cases, and exact prediction/margin determinism for both trained arms on two repeats of 20 cases.

Advance only if every gate passes:

- independent width-32 rollout is at least 90%;
- independent width-64 transition is at least 99.8% and rollout at least 90%;
- trained width-64 rollout exceeds the width-16 source by at least 50 points;
- oracle union is at least 96% and at least three points above the better arm;
- frozen target-free routing is at least 1.5 points above the better arm;
- independent invalid-trajectory rate is at most 1%;
- paired 600-case wall time is at most 180 seconds;
- predictions and minimum margins are exactly deterministic.

Oracle success without routing success does not authorize a two-model inference candidate. Failure of width-64 accuracy/retention closes this exact bootstrap schedule, not independent-seed diversity in general. Passing authorizes one separately preregistered width-128 training and paired routing confirmation.

## Dynamic resource-budget protocol

Before formal execution, measure ten representative updates in each of the four stages on training identities, including actual hard mining and recovery. Measure ten training-identity rollouts for the source, incumbent, and partially continued independent model plus representative transitions. Project all 3,200 updates, complete evaluation/control/repeat workload, setup, tracking, and serialization; record wall and aggregate CPU separately; add safety headroom and a hard runaway ceiling.

Calibration may change only resource and resumability fields. It may not change sources, identities, model, stages, updates, batch sizes, cases, routing rule, or gates. Preserve completed stage/checkpoint evidence and never reduce evaluation fidelity to fit an arbitrary wall cap.

The completed training-identity-only calibration measured ten real updates per stage at 7.889/29.452 wall/aggregate-CPU seconds for bridge, 13.200/49.437 for wide, 18.800/70.419 for Tier 5, and 18.120/67.905 for recovery. The recovery segment includes its declared learner-state updates. Projection is 4,231.433 wall / 15,838.894 aggregate CPU seconds for all 3,200 updates and 741.537 wall / 2,776.981 aggregate CPU seconds for every evaluation, control, and determinism repeat.

With a distinct 180-second setup/tracking/serialization allowance, the frozen expected budget is 5,153 wall seconds and 323 aggregate CPU minutes. The soft budget is 6,960 wall seconds; the hard runaway ceiling is 10,320 wall seconds, with a separate four-thread upper bound of 688 aggregate CPU minutes. External calibration timing was 89.05 wall seconds, 289.76 user seconds, and 5.68 system seconds at 331% CPU, with 1,065,512 KiB maximum RSS. The calibration JSON hashes to `f0bac70be63590586aefd7fbb2c26b0ae7cb54cbee5f69d3677a6faede50ecd6`. No held-out identity was read and no scientific field changed.

## Limitations

The curriculum is evidence-based but not update-for-update identical to every historical precursor; it matches the important width progression, architecture, hard-mining distribution, and recovery mechanism while giving the independent lineage a declared 3,200-update budget. Raw margins may be differently calibrated across lineages. Even a pass at width 64 does not prove diversity or accuracy at width 128.
