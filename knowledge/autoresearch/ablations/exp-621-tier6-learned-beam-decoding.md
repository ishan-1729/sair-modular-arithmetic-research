---
type: Experiment Preregistration
title: EXP-621 Tier-6 Learned Beam Decoding
description: Frozen paired test of greedy state decoding against width-2 learned-likelihood search over one-bit alternatives.
status: completed
tags: [autoresearch, tier6, beam-search, confidence, neuro-symbolic]
timestamp: "2026-07-13T07:20:00+05:30"
---

# EXP-621 Tier-6 learned beam decoding

## Question and hypothesis

First-error analysis finds that 69% of failed trajectories initially contain exactly one wrong bit and 77% contain one or two. Training changes and alternate state representations have not crossed 90%. Can inference retain the hard prediction and a single least-confident-bit alternative, then use future learned likelihood to preserve the correct trajectory?

At every Horner step and for each live beam, the frozen model emits bit logits. The decoder creates the hard bit vector and one vector with only the minimum-absolute-logit bit flipped. Candidate score is the accumulated sum of independent bit log-probabilities from the trained logits. Duplicate states retain only their best score, and the top two unique learned-score states survive. The highest learned-score state is the final output.

There is no modulus validity filter, expected-answer check, exact arithmetic circuit, target feedback, quotient oracle, or fallback in beam inference. Modulus and operands enter only through the existing learned model features. Oracle beam containment is computed after evaluation solely as a diagnostic and cannot affect output.

## Paired evaluation and gates

The exact frozen EXP-609 checkpoint `cbc4a1...d4bd` runs greedy and beam decoding on identical 200 fresh cases at each of widths 64 and 128. Per-case predictions, beam states/scores, and greedy-only/beam-only counts are retained. A shared 5,000-case transition check characterizes the checkpoint. Twenty paired cases per width are rerun from identical seeds and must reproduce greedy predictions, beam predictions, complete beam states, and paired counts.

Advance only if beam decoding reaches at least 90% width-128 rollout, gains at least three points over greedy, retains at least 95% width-64 rollout, has at most 1% invalid final residues, is deterministic, and projects a complete Candidate-004-plus-Tier-6 runtime no greater than 240 seconds. A pass requires higher-fidelity confirmation and an official isolated runtime gate before integration.

## Resource budget

On a distinct timing seed, 20 width-128 cases took 16.79 wall / 64.65 aggregate CPU seconds greedy and 32.31 wall / 124.61 aggregate CPU seconds beam. Main paired evaluation across both widths projects to 615 wall seconds; determinism, transition, and setup receive 120 seconds, for a 735-second expected total.

The planning budget is 1,150 wall seconds and the hard runaway ceiling is 1,700 wall seconds, or at most 113.34 aggregate CPU minutes with four continuously occupied threads. Wall and CPU are distinct. Raw paired predictions are saved and hashed. Evaluation fidelity will not be reduced to meet a shorter cap.

The projected candidate gate uses measured width-128 beam seconds per case, 100 Tier-6 cases, Candidate 004's observed 36.9-second isolated runtime, and 15 seconds integration allowance. This projection is screening evidence only, never a substitute for the pinned official sandbox.

## Boundaries and library choice

Exact arithmetic is confined to generated-case truth and post-inference scoring. Beam decoding is learned-parameter dependent and supplies generic search/routing rather than modular arithmetic. Native PyTorch logits and log-sigmoid scoring are sufficient; a general beam-search package would add sequence-token assumptions without improving this state-space decoder.

## Result and decision

At width 128, greedy scores 165/200 (82.5%) and selected beam output 164/200 (82.0%): 163 both correct, two greedy-only, one beam-only, and 34 neither. However, the correct answer appears somewhere in the final width-2 beam for 172/200 (86.0%), demonstrating improved search coverage but failed ranking. Width 64 is exactly tied at 95%, while oracle beam containment is 98%.

Reject accumulated independent-bit likelihood as the beam selector. Determinism, final validity, and projected integrated runtime (167.42 seconds) pass. Runtime was 634.35 wall seconds and 2,355.05 aggregate CPU seconds. Preserve the generic beam decoder and raw states. The next staged question is whether a wider beam covers at least 92% within the 300-second envelope, which would justify a separately trained reranker; do not tune scoring on this selection set.
