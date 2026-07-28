---
type: Experiment Preregistration
title: EXP683 Source-Complete HRM
description: Complete ST-M007 H/L, one-step/deep-supervision, and learned-Q ACT package.
status: permanent-calibration-execution-park
tags: [autoresearch, HRM, de-novo, deep-supervision, Q-learning, ACT]
timestamp: "2026-07-30T00:00:00+05:30"
---

# EXP683 source-complete HRM

EXP683 tests all three M007 components together: distinct carried H/L states with two L
updates per H update; local one-step gradients and detached deep supervision; and learned
deep-Q adaptive compute. Exact transfer from Candidate005 is structurally impossible, so
the route is de novo with 4,000 updates per trained arm. It must not be reinterpreted as a
400-update continuation.

The complete matrix is full HRM, equal-frequency, high-reset, low-reset, segment-detach-
disabled, full-gradient, no-deep-supervision, fixed-max ACT, fixed-average ACT,
deterministic-random ACT, Q-disabled, shuffled-Q, one-state, frozen initialization,
randomized weights, and frozen Candidate005. All trained arms match updates, examples,
optimizer, data order, and labels; calls and gradient work are disclosed.

Root 20261282 is complete-route calibration only. It measures one representative update
per trained arm, every evaluation/intervention/persistence route, wall/process CPU/RSS,
variability, full 4,000-update projection, and candidate-shaped inference. Root 20261283
may open only if every admission conjunct passes. Because this is de novo, its first science
gate is at least 85% W16 exact with causal H/L, gradient/deep-supervision, and Q dependence;
only then can W128, W256, and roots 20261284/85 open.

Exact correctness is used only for offline labels and Q targets. Release ACT sees only
learned state/Q and frozen bounds. No checked, independent, official/private, candidate,
submission, promotion, or completion evaluation is authorized.

## Terminal result

T086 restored the exact ignored fixtures and repaired only EXP681's current global
topology invariant while preserving its historical park. The focused cone passed
156/156, and the single package oracle passed validation plus 703/703 tests in
519.53 pytest seconds (513.35 outer wall, 632.40 aggregate CPU, 934,568 KiB RSS).

Root 20261282 then persisted 13 one-update trained routes and two untrained routes
before failing at the first checkpoint reload with a `PosixPath / dict` TypeError.
No evaluation or admission projection completed. The root is consumed and the frozen
single-oracle/no-repair rules prohibit a patch, retry, or later root. This is a permanent
execution park, not scientific evidence against HRM; roots 20261283-20261285 and all
protected evaluation surfaces remain unopened.
