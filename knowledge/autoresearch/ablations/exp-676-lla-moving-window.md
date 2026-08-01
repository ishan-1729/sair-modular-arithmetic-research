---
type: Experiment Preregistration
title: EXP676 LLA moving-window supervision
description: Source-faithful ST-M004.a moving truncated endpoint-window loss on bit-identical Candidate005 recurrence.
status: parked-calibration-runtime-headroom
tags: [autoresearch, looped-learning, moving-window, recurrent-credit]
timestamp: "2026-07-29T13:00:00+05:30"
---

# EXP676 - LLA moving-window supervision

## Decision and authenticated source

Can the nonredundant component of Looped Learning Architecture (LLA), a moving
truncated endpoint/window objective, retain valid Candidate005 W128 exactness
and materially improve W256 without changing formal inference?

The immutable local `research/papers/arxiv-2311.12424.pdf` hashes to
`76b3be4b761409e6987a4f74a7caa80f434e59422bbb836ffec7b2092c34ebfc`
and is authenticated in the manifest, checksum ledger and verification record.
Pages 5-6 were rendered at 2x with local PyMuPDF because Poppler is unavailable,
then visually inspected: equations, Fig.3-4, captions, axes and scheduled-
training text are legible and unclipped.

The source defines repeated input injection as `Y[t+1] = M(Y[t] + P)` while
noting that injection need not be additive. It truncates loss to `T` loop
outputs and progressively increases the training endpoint `b`, moving the
window from early `[1,T]` outputs to late `[b-T,b]` outputs. Candidate005
already implements the first component exactly in its own topology: every
physical recurrent update concatenates the same legal six-channel feature
tensor with normalized hidden state, and the candidate proposal again
concatenates those features after reset gating. The feature tensor therefore
causally enters every visit; no reinjection-only arm is warranted.

M004.b is closed. EXP634 is adjacent negative evidence only: it added learned
additive input reinjection to a direct recurrent attention model and is not an
LLA moving-window test. EXP609 supplies the only source checkpoint.

## Frozen training variable

Formal inference stays the original 99,824-parameter
`SharedDilatedConvGRU`. A training-only trace applies the existing output and
quotient heads after each physical dilation visit, without adding parameters.
The final traced output must be bit-identical to the ordinary forward result.
Saved checkpoints contain the exact original state-dict schema.

For each update, four transition-loss calls are averaged:

- `moving_window`: a contiguous four-visit window whose endpoint increases
  linearly from visit 4 to the width-specific final physical visit over 400
  updates;
- `endpoint_only`: the final visit repeated four times, preserving the ordinary
  endpoint gradient while matching loss-call work;
- `fixed_window`: the final four visits;
- `equal_work_multi_endpoint`: four evenly spaced visits;
- `shuffled_window`: a deterministic seed/step-specific random four-visit
  subset, isolating contiguity and progressive order;
- `frozen_source`: authenticated EXP609 checkpoint with no updates;
- `randomized_control`: independently randomized original architecture with no
  training, used only for trained-dependence evidence.

Every trained arm receives the same 400 updates, legal identities, examples,
mode sequence, AdamW schedule, learning rate, clipping and four loss calls.
The moving and shuffled selections change endpoint distribution only. The
training trace adds head evaluations and label losses; these are unmatched
offline work relative to frozen/randomized controls and never enter inference.

## Identities, gates and route

The source base is pushed EXP675 terminal commit
`abfec8c4b80d13add8d08f1fa9dee7beba05da13`; Candidate005 model SHA is
`9b271fd13084011e1631125040e4080ea4397a0356dab1f1ce81bfee3f799edc`;
EXP609 checkpoint SHA is
`cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd`.
Fresh legal roots are calibration 20261254, formal 20261255 and conditional
confirmations 20261256/20261257. Formal training is 400 updates; evaluations
are 3,000 transitions and 300 rollouts per opened width.

Cache every arm, then evaluate the complete W128 moving arm first. Reject if it
is below 85%, has any invalid, loses more than two points to endpoint or source,
lacks material window-gradient/trained dependence, fails exact inference
identity/reload, or misses resources. If still viable, require lower held-out
late-depth loss and later first error than endpoint/fixed/equal-work; W256 at
least 10% and five points above endpoint/fixed/equal-work/shuffled. Both
confirmation roots must reproduce every conjunct.

Root 20261254 executes all training, control, evaluation, loss-diagnostic,
intervention, persistence and representative batched candidate-timing routes.
It separately records wall, process-tree CPU, cgroup CPU, process-tree RSS and
variability, then derives formal ETA/expected/soft/hard guards with 1.35
headroom. Package verification starts from the observed 575.53-second class and
1,246/2,492-second expected/hard envelope; fidelity is never shortened to fit a
static cap.

Exact intermediate states are training/evaluation labels only. Formal inference
has no teacher, endpoint selector, exact answer/correctness/validity filter,
answer-dependent depth, repair, target canvas, programmed multiplication,
division, reduction or CRT, lookup, checked/private data, or external call.

## Terminal result

The final implementation passed nine focused tests and the one hermetic package
oracle: validation reached 198 concepts / 1,100 records and all 637 tests passed.
Root `20261254` then completed every calibration arm, width, intervention,
persistence, and candidate-timing route in 233.668 wall / 780.700 process-tree
CPU / 802.348 cgroup CPU seconds with 1,081,548,800-byte peak RSS. Reinjection
equivalence, exact formal inference identity, complete-route execution, artifact
size, and no-fidelity-reduction checks passed.

The measured route projected 533.876 nominal and 720.733 guarded candidate
seconds, above the frozen 285-second ceiling. The binding decision is therefore
`reject-formal-at-calibration`; root `20261255` and confirmations `20261256/57`
remain unopened. Calibration exactness/loss diagnostics use tiny route-proving
denominators and are non-scientific. This resource park does not reject LLA,
ST-M004.a, moving-window supervision, or the paper, and changes no candidate,
benchmark, submission, promotion, or completion evidence.
