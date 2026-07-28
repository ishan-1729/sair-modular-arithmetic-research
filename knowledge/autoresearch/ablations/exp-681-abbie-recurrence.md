---
type: experiment-preregistration
title: "EXP681: AbbIE recurrence extrapolation"
timestamp: 2026-07-30T06:00:00+05:30
status: frozen
experiment: exp-681-abbie-recurrence
description: Frozen bounded-faithful AbbIE recurrence extrapolation test for Candidate005.
---

# EXP681: AbbIE recurrence extrapolation

## Claim and source boundary

This bounded-faithful ST-M008 test uses AbbIE arXiv:2507.08567 page 3 Figure 2
and pages 3–4 Equations 3–5. Candidate005's stem is the head, one complete
dilation schedule is the body `B`, and its output normalization plus bit and
quotient readouts are the tail. Form C is `h(k+1)=B(h(k))`; form D is
`h(k+1)=B(h(k))+h(k)`. Input chunks are not recurrence steps.

## Frozen experiment

Train recurrence is exactly 2 for 400 updates. Test recurrence is 1, 2, 4, and
8. The primary arm is tied form C. Matched controls are form D, a two-copy
untied body, a randomized-body initialization, and the frozen source. Mechanism
tests reset or shuffle recurrent state, zero head/tail, bypass body, and
randomize body weights.

Root 20261274 is calibration-only. Root 20261275 may open only after all dynamic
admission checks pass. Roots 20261276 and 20261277 remain unopened until the
formal scientific gates authorize confirmations. W256 opens only after W128
retention, exactness, validity, control, and mechanism gates pass.

## Evaluation policy

Only fresh generated roots are permitted. Checked, independent, official,
private, candidate, submission, promotion, and completion evaluation are
prohibited. Exact-answer accuracy is primary; bit accuracy alone is not a pass.

## Stop conditions

Park the package if calibration rejects runtime, memory, artifact, source
identity, recurrence/state/weight dependence, or topology. Do not retune after
calibration. Stop formal work immediately on any W128 gate failure.

## Terminal outcome

T082's sole final-byte focused run passed 9 tests and failed 1 in 4.18 wall and
9.83 process CPU seconds with 251,992 KiB peak RSS. The frozen topology test
expects flat `tail_quotient.weight` and `tail_quotient.bias` names, while the
copied Candidate005 quotient tail is sequential and exposes four indexed
parameter tensors. The no-repair rule permanently parks this package before the
full suite and root 20261274. No training or scientific evaluation occurred.
