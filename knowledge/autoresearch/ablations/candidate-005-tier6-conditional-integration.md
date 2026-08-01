---
type: Candidate Preregistration
title: Candidate 005 Tier-6 Conditional Commutative Integration
description: Frozen-artifact release gate for integrating the confirmed width-128 conditional commutative specialist into Candidate 004.
status: completed
tags: [autoresearch, candidate, Tier-6, commutativity, conditional-compute, release]
timestamp: "2026-07-17T16:05:00+05:30"
---

# Candidate 005: Tier-6 conditional commutative integration

## Frozen change

Build Candidate 005 in scratch from the exact Candidate 004 three-file artifact and the frozen EXP-609 Tier-6 checkpoint. Candidate 004 remains untouched until the challenger passes every gate. The only learned-boundary extension is Tier 6: run the original operand orientation for every row, rank rows by minimum absolute bit-logit margin over the complete trajectory, route exactly the lowest-margin 40% through the commuted operand orientation, and retain whichever orientation has the higher minimum trajectory margin. Route ties use the lower evaluator-batch index and orientation ties retain the original. The 40% fraction, confidence statistic, selector, checkpoint, and tie breaks are unchanged from EXP-641 and the fresh EXP-642 confirmation.

The adapter uses sanctioned pairwise operand reduction. Trained recurrent weights materially produce every candidate remainder; the confidence route supplies target-free control only. There is no exact modular multiplication, division, reduction, CRT, benchmark lookup, or corrective output repair in inference. Inputs above 128 modulus bits keep the deterministic zero fallback.

Candidate construction also replaces Python scalar tensor assignments in the existing big-integer bit encoder with an equivalent batched `torch.tensor` construction. A direct cached EXP-642 check already reproduces ten original-orientation predictions and every minimum margin bit-for-bit. The formal lower-tier gate nevertheless requires Candidate 005 and Candidate 004 predictions to match on every Tier 1–5 row from independent seed 1.

## Independent protocol

Five exact seed hex strings are frozen in the resolved config before evaluation. Seed 1 is reserved for the complete 1,100-case official-generator run in the pinned isolated container. Seeds 2–5 each contribute the complete 100 official-generator Tier-6 cases locally. Together the container Tier-6 result and four local results form five independent Tier-6 seeds; each must reach at least 90%, not merely their aggregate.

The local gate additionally requires 100% valid output form, exact 20-case repeated determinism, clean manifest/static/artifact checks, exact Tier 1–5 prediction equivalence to Candidate 004 on seed 1, and at most 10% Tier-6 accuracy after deterministic distribution-matched randomization of Tier-6 floating tensors only. The full isolated seed must reach at least 58% exact overall; Tiers 1–6 must reach 99%, 95%, 95%, 90%, 90%, and 90%; `highest_tier_above_90` must be at least 6; and inference must complete within the 285-second promotion ceiling under the official 300-second constraint. The checked/public benchmark remains unread during selection.

Passing these gates promotes an intermediate Candidate 005 only. Roughly 59% overall remains far below the binding 85% completion threshold, so promotion cannot complete the Goal.

## Dynamic resource calibration

The resource budget is deliberately not frozen from an arbitrary short wall cap. First measure two warmup and ten profiled Tier-6 cases through the exact built Candidate 005 artifact, including the frozen 40% route. Record wall time and aggregate process CPU time separately. Scale measured orientation-equivalent throughput to four complete local Tier-6 seeds, the randomized control, and two determinism repeats; then add measured or incumbent-backed generator, loading, Tier 1–5 equivalence, tracking, artifact, and safety headroom. The isolated full run retains the competition's real 300-second constraint and the preregistered 285-second promotion ceiling. Commit the measured calibration hash and revised expected, soft, hard, and aggregate-CPU budgets before formal evaluation.

Completed seed evidence is checkpointed after every local seed. Reuse valid EXP-642 confirmation and Candidate 004 evidence rather than rerunning those experiments. Do not reduce the 100-case seed fidelity, the five-seed count, the lower-tier equivalence surface, the randomized control, or the full isolated run to fit a wall-clock target.

## Calibrated budget revision

Calibration completed on repository commit `5dc41b2` and the exact scratch artifact hashes frozen in the config. After two warmup cases, ten complete conditional cases—fourteen orientation-equivalents—consumed 14.3676 wall seconds and 55.6464 aggregate CPU seconds. This is 1.02626 wall and 3.97474 aggregate CPU seconds per orientation-equivalent. The ten profiled outputs were all exact and valid, but that accuracy is calibration evidence only and is not used for selection. The calibration JSON SHA-256 is `575b94e686a205a713930a23ffc45e28df88296a44e6737b189d2a84e34ad918`.

The complete local gate contains 756 orientation-equivalents: four 100-case learned Tier-6 seeds, one 100-case randomized-weight control, and two 20-case determinism repeats. Small-batch calibration projects 775.85 wall seconds and 50.08 aggregate CPU minutes for those paths. Add 120 wall seconds and ten aggregate CPU minutes for generation, two complete Tier 1–5 prediction-equivalence passes, loading, static checks, tracking, serialization, and safety headroom. The revised expectation is therefore 896 wall seconds and 61 aggregate CPU minutes; the soft budget is 1,350 wall seconds; and the hard runaway ceiling is 2,100 wall seconds with a conservative 150 aggregate CPU-minute upper bound. These are distinct units.

For the isolated full run, the same conservative small-batch rate projects 180.58 inference wall seconds and 219.17 total container wall seconds when added to Candidate 004's measured evidence. This leaves useful room below the 285-second inference promotion gate and the official 300-second inference constraint. EXP-642's valid higher-fidelity 500-case cache was materially faster than this projection, but it is retained as corroboration rather than used to shrink the budget. No evaluation fidelity changed.

## Local gate result

Pass. Fresh official-generator Tier-6 seeds 2–5 score 96/100, 95/100, 99/100, and 99/100 exact, with every output valid. The minimum seed accuracy is 95% and aggregate accuracy is 389/400 (97.25%). Candidate 005 and Candidate 004 predictions match exactly on all 500 Tier 1–5 rows from seed 1. Two complete 20-case conditional repeats match exactly. Deterministic distribution-matched randomization of Tier-6 tensors collapses seed 2 to 2/100, while both artifacts remain static-clean.

The local runner consumes 678.93 wall seconds and 2,554.60 aggregate CPU seconds (42.58 minutes), below the 896-second expectation, 1,350-second soft budget, 2,100-second hard ceiling, and 150-minute aggregate CPU upper bound. GNU time independently records 697.53 wall, 2,486.99 user CPU, 80.16 system CPU seconds, and 468,716 KiB maximum RSS. No fidelity was reduced.

Decision at the local gate: advance the exact frozen scratch artifact to the pinned isolated full seed-1 gate. Do not replace Candidate 004 or read the checked benchmark yet.

## Isolated full-set and release-regression result

Pass and promote as the intermediate incumbent. The exact hash-frozen artifact scores 59.3% on the complete independent seed-1 set, with Tiers 1–6 at 100%/99%/97%/97%/94%/98% and `highest_tier_above_90 = 6`. Together with local seeds 2–5, Tier 6 scores 98%/96%/95%/99%/99%: 487/500 (97.4%) aggregate with a 95% minimum.

The pinned official Podman sandbox reproduces the independent score, passes static analysis, preprocessing isolation, and determinism, and completes inference in 156.0 seconds (178.41 seconds container wall), below the 285-second promotion ceiling and official 300-second limit. The post-selection complete checked regression scores 59.5%, with Tier 6 at 96% and all lower specialists retained. All evidence used the exact manifest/model/weights hashes frozen in the preregistration; no evaluation fidelity was reduced.

Candidate 005 replaces Candidate 004 as the intermediate incumbent. It does not complete the program: checked and independent overall accuracy remain roughly 59%, five complete independent 1,100-case seeds cannot pass while Tiers 7–10 use the zero fallback, and no legitimate private-official result is available. Continue at the Tier-7 width-256 boundary.
