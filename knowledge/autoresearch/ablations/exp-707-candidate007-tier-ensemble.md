---
type: experiment
title: EXP707 Candidate007 tier ensemble
description: Test whether an input-only one-of-four EXP701 router improves Candidate006 Tier-7 exact accuracy without extra expert compute.
status: completed-negative
run_id: exp-707-candidate007-tier-ensemble
timestamp: "2026-08-12T19:53:33Z"
tags: [candidate007, tier7, ensemble, router, negative-result]
---

# EXP707: Candidate 007 tier ensemble

## Evidence-ranked choice

Candidate005/006 already place Tiers 1-6 near their observed ceilings. The evaluated HRM/TRM descendants (including answer-latent TRM, LoopFormer, universal-transformer memory, attractor, and STARS variants) either produced zero exact rollouts, failed their own conjunctive gate, or remained below the completed EXP701 family. Tiers 8-10 have no validated learned route. The only measured, complementary headroom is therefore Tier 7.

EXP703 evaluated the four EXP701 arms on the same 600 width-256 rollouts. `teacher_cosine` is the best single arm at 361/600, while the four-arm oracle union is 395/600. Naive plurality is worse than the best arm, so Candidate007 tests a learned input-only router rather than output voting.

## Frozen hypothesis

A 64-hidden-unit router using only 16-bit chunk densities from the reduced multiplier, multiplicand, modulus, XOR, and AND bit planes can predict which EXP701 expert is most likely to be exact. This fixed 80-feature representation is cheap and limits memorization. Inference must choose exactly one expert before any expert rollout. Candidate006 remains byte-equivalent on Tiers 1-6 and retains its deterministic zero fallback above Tier 7.

## Leakage boundary and split

The router source is the authenticated EXP703 width-256 paired evidence. Cases are grouped by modulus using SHA-256 namespace `exp707-modulus-group-v1` and assigned to five buckets. Bucket 0 is sealed until the architecture, optimizer, epochs, expert order, and route-margin grid are frozen here. Bucket 1 calibrates only the route margin. Buckets 2-4 fit the router. No target, expert prediction, correctness flag, checked-public answer, or private answer is available to the submission inference path.

If the sealed router does not beat `teacher_cosine` by at least two exact cases, the experiment stops negative and no Candidate007 score claim is made. If it passes, the same frozen recipe is fit on all 600 rows, the artifact is built, and the already-declared five generator seeds are evaluated once. Checked-public and official runtime evaluation occur only after router parameters are frozen.

## Promotion gates

- at least 320/500 Tier-7 exact across the five declared seeds, strictly above Candidate006's 319/500;
- mean overall exact accuracy strictly above Candidate006's 0.6528;
- exact prediction equivalence on Tiers 1-6;
- one expert rollout per Tier-7 case, all outputs valid, deterministic, and static-clean;
- complete official 1,100-case inference in at most 300 seconds on 4 CPU / 8 GiB.

The current upstream rules independently prohibit Candidate006's inherited deterministic full-width operand pre-reduction. EXP707 is therefore a pinned-runtime scientific candidate and cannot be called current-rule submission compliant unless that separate blocker is resolved.

## Stop conditions

Stop rather than tune after seeing sealed or checked-public outcomes if the sealed gate fails, lower-tier predictions change, runtime exceeds 300 seconds, the router evaluates more than one expert per case, or provenance/hash checks fail.

## Result

The input-only router fit the 324-row development partition almost perfectly relative to the available oracle: 213/324 selected exact versus 192/324 for `teacher_cosine`, with the oracle also 213/324. That apparent gain did not generalize. On the 107-row calibration partition the selected route scored 58/107 versus 63/107 for `teacher_cosine`; on the sealed 169-row modulus-group partition it scored 97/169 versus 106/169, a loss of nine exact cases. The sealed gate required a gain of at least two and therefore failed.

Candidate007 was rejected before artifact construction, five-seed scoring, checked-public evaluation, or runtime evaluation. No post-sealed feature, margin, architecture, or expert-order tuning was performed. Candidate006 remains the strongest retained measured artifact, subject to its separately documented current-upstream pre-reduction compliance blocker.
