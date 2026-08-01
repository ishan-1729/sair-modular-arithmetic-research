---
type: Experiment Preregistration
title: EXP-622 Tier-6 Width-4 Beam Coverage
description: Fresh frozen-checkpoint feasibility test for wider learned-beam coverage and candidate-shaped runtime.
status: completed
tags: [autoresearch, tier6, beam-search, coverage, runtime]
timestamp: "2026-07-13T08:40:00+05:30"
---

# EXP-622 Tier-6 width-4 beam coverage

## Hypothesis and changed variable

Changing only the retained beam width from two to four around the frozen EXP-609 checkpoint will place the correct final width-128 state somewhere in the learned beam for at least 92% of 200 fresh generated cases. The branch operator, cumulative learned-likelihood score, model, checkpoint, prime split construction, and rollout remain fixed. Oracle containment is computed after inference only and can never select an output.

## Evidence and selection boundary

EXP-621's 200-case width-2 evidence is preserved as the prior comparison: 82.5% greedy, 82.0% selected beam, and 86.0% oracle containment. EXP-622 uses untouched seed `20260818`; checked-public and private answers are not read. It records every input, expected value, greedy prediction, selected beam prediction, final beam state, and learned score. Twenty cases are rerun twice for exact prediction, state, and paired-count determinism.

The frozen checkpoint's 5,000-case transition measurements from EXP-621 remain valid because neither checkpoint nor transition evaluator changes; they are cited rather than recomputed. The new evidence is a matched greedy/width-4 rollout comparison at width 128.

## Promotion and stopping rule

Advance to a separately trained global reranker only if all gates pass: oracle final-beam containment at least 92%, selected final invalid-residue rate at most 1%, exact determinism, and projected Candidate-004-plus-Tier-6 runtime at most 285 wall seconds. Selected likelihood accuracy is reported but is not the coverage gate. A failed runtime gate rejects the current implementation even if coverage passes; a failed coverage gate closes width-4 reranking without an optimization detour.

## Dynamically calibrated resource budget

Two timing-only profiles used distinct generated seeds and did not inspect expected answers. At width 128 with four PyTorch threads, 20 cases required 54.70 wall / 210.43 aggregate-process CPU seconds. Because EXP-621 showed batch-size scaling, a candidate-shaped 100-case profile was allowed to finish and required 239.62 wall / 918.68 aggregate-process CPU seconds. All cases retained four unique final states.

The full 200-case matched greedy/beam evaluation plus two 20-case determinism repeats is expected to require 825 wall seconds including setup and logging. The soft budget is 1,150 wall seconds; the hard runaway ceiling is 1,700 wall seconds, whose conservative four-thread upper bound is 113.34 aggregate CPU minutes. These limits are explicitly wall time and aggregate process CPU is measured separately. No training or evaluation fidelity is reduced to meet them, and completed raw predictions are cached.

Candidate runtime is projected from the actual 100-case profile, not the smaller-batch rate: `36.9 + 239.6167 + 15 = 291.5167` wall seconds. Thus the runtime gate is already provisionally missed by 6.52 seconds. The full coverage comparison still has decision value because this margin could be changed by a separately justified trained-runtime optimization, but no reranker training is authorized unless the declared gate as a whole passes.

## Compliance and limitations

Beam expansion uses only trained logits and generic state search. Exact bigint arithmetic is confined to generated truth and post-inference scoring. The experiment is a generated-seed feasibility test, not checked-public, official-private, or official-sandbox evidence. The 100-case timing profile is closer to the submission shape than per-case extrapolation but does not replace the pinned isolated container gate.

## Result and decision

The complete run finished unchanged in 948.63 wall seconds and 3,584.10 aggregate CPU seconds. On 200 fresh width-128 cases, greedy scored 178/200 (89.0%), the likelihood-selected width-4 output scored 173/200 (86.5%), and the correct answer occurred anywhere in the final beam for 188/200 (94.0%). There were seven greedy-only selected outcomes, two beam-only selected outcomes, and twenty cases absent from both selected outputs. Final selected residues were all valid, and repeated predictions, full beam states, and paired counts were exactly deterministic.

Coverage passes, but the combined gate fails because the candidate-shaped runtime projection remains 291.52 seconds against the frozen 285-second maximum. Reject the current width-4 implementation and do not train a reranker yet. Preserve the 94% coverage signal and test one frozen, exact-output inference optimization; only a deterministic optimization that reduces the same 100-case workload enough to restore headroom may reopen reranker training.
