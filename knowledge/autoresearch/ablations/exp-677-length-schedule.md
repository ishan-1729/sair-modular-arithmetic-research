---
type: Experiment Preregistration
title: EXP677 known-step endpoint-length scheduling
description: Source-faithful ST-M004.b width/known-visit/final-endpoint curriculum around frozen Candidate005 inference.
status: parked-calibration-runner-failure
tags: [autoresearch, length-generalization, curriculum, known-step]
timestamp: "2026-07-30T00:00:00+05:30"
---

# EXP677 - known-step endpoint-length scheduling

## Authenticated source and contradiction resolution

The immutable local ICLR 2025 paper `research/papers/arxiv-2409.15647.pdf`
hashes to `6207287dd6bef2e9fad57b2d3010a264a196d55c3cb67d8327391fee6440ac00`.
Its manifest and verification records hash to `f2e2e386...f09c21` and
`cdf94b67...8d58d`. Pages 5-7 and 9-10 were rendered at 2x and visually
inspected. The source trains end-to-end examples carrying diverse lengths and
known required step counts, supervises the final output at that known step,
and uses a smallest-to-maximum length curriculum. No intermediate targets are
required. Per-sample confidence is an optional inference rule, not the training
schedule.

EXP676's sentence that M004.b was closed conflicts with the authoritative
ledger, which says M004.b is open. EXP676 explicitly excluded length scheduling
and tested M004.a moving multi-endpoint credit only. The ledger is correct:
M004.b had no evidence before EXP677.

## Frozen SAIR adaptation

Candidate005 already has a shared recurrent cell, no learned positional
embedding, legal input reinjection at every visit, and a deterministic full
physical-visit count for each width. EXP677 maps source length `L` to legal
operand width, known steps `T(L)` to the existing full physical-visit count,
and final-answer supervision to the final known visit of each transition. The
ordered arm uses exactly 100 updates at `(64,12,64)`, 150 at `(128,14,128)`,
and 150 at `(256,16,256)`. The fixed control repeats `(128,14,128)` for all
400 updates. The equal-work control deterministically permutes the exact tuple
multiset. Each trained arm receives 400 updates, 16 examples per update, eight
recovery transitions per example, the same optimizer schedule, and one loss
call per recovery transition. Width-dependent tensor FLOPs are irreducibly
different for the fixed control and will be measured rather than concealed.

No visit embedding, invented shuffled visit identity, adaptive exit, confidence
selection, moving-window M004.a loss, answer-conditioned inference, new state,
parameter, call, output, or formal inference path is allowed. Candidate timing
uses the unchanged W128 formal route; W256 is scientific transfer evidence only.

## Frozen identities, roots, gates and resources

The integrated base is `aca5dceb6f31b93c2a07f5453ec73f5cf79eb145` and the
only source checkpoint is EXP609
`cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd`.
Roots are calibration `20261258`, formal `20261259`, and conditional
confirmations `20261260/61`. Formal denominators are 3,000 transitions and 300
rollouts at each opened width. All arms are trained and cached before scoring.
W128 scheduled primary is scored first; later controls and W256 are conditional.

Admission requires complete-route parity, compliance, exact formal identity,
artifact and memory safety, expected-value acceptance, and guarded unchanged
candidate inference at most 285 seconds. Formal requires scheduled W128 at
least 95%, zero invalids, at least 99.8% transition exactness, no more than two
points below both frozen source and fixed continuation, material gradients and
trained-weight dependence. Only then may W256 open, where scheduled must reach
10% and beat fixed and equal-work by five points. Confirmations must reproduce
every conjunct. Calibration measures wall, aggregate process-tree CPU, cgroup
CPU, peak RSS and route variability, then derives ETA/soft/hard guards with
headroom; no arbitrary cap or fidelity reduction is permitted.

No checked, independent, official/private, candidate, submission, promotion or
completion evaluation is authorized. A pass remains intermediate.

## Terminal verification result

The implementation passes nine focused schedule, coverage, matching,
gradient-causality, formal-identity, reload, source, leakage and resource tests.
After materializing the authenticated EXP609 fixture inside the scoped ignored
scratch area, the combined EXP676/677 cone passes 18 tests.

The sole dynamically budgeted package oracle then stopped in project validation
before pytest. The separate worktree lacks the immutable nested challenge
snapshot: eight existing knowledge links into that snapshot are broken, and the
validator resolves the empty nested directory to repository commit `aca5dce`
rather than required challenge commit `82510b...`. Materializing or changing
that protected snapshot is outside T074 allowed files. No owned repair or second
oracle is legal. EXP677 is therefore parked at package verification; calibration
root `20261258`, formal `20261259`, and confirmations `20261260/61` remain
unopened. There is no scientific verdict on M004.b.

## T088 authenticated recovery result

T088 authenticated the unchanged frozen package, the exact challenge snapshot,
and the absence of roots `20261258`-`20261261`. It reused T074's nine focused
passes and T087's later green validator plus 716-test package oracle; neither
suite was rerun. Calibration root `20261258` then opened exactly once.

All five arms materialized. The scheduled, fixed-W128, and equal-work shuffled
arms each completed three updates, and all nine trained steps had nonzero
gradients. The first all-arm evaluation then raised `KeyError: 'window'` in the
inherited `late_depth_losses` path: that base method reads
`config["window"]["size"]`, while the authenticated frozen EXP677 config has no
`window` section. No evaluation denominator, calibration summary, admission
decision, or scientific metric completed.

The frozen runner/config mismatch was not repaired and root `20261258` was not
retried. Roots `20261259`-`20261261` remain unopened. EXP677 is terminally
parked on execution integrity, and M004.b remains scientifically open.
