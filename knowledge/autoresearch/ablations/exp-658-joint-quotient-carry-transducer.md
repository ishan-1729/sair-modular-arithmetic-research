---
type: Experiment Preregistration
title: EXP-658 joint quotient/carry learned transducer
description: L0-only split-convolution contract for soft learned quotient and signed-carry messages on the frozen EXP-609 cellular transition.
status: preregistered-l0-only
tags: [autoresearch, quotient, signed-carry, recurrent, transducer, neuro-symbolic]
timestamp: "2026-07-22T02:00:00+00:00"
---

# EXP-658 joint quotient/carry learned transducer

## Evidence boundary

This package implements L0 only: the frozen graph, exact offline trace labels,
resolved contract, source-parity tests, and a fail-closed future runner. It does
not perform an optimizer step, open a calibration or scientific root, start
MLflow, write run evidence, edit the registry or autoresearch governance, change
Candidate 005 or `submission/`, or report accuracy, runtime, rollout, causal, or
mechanism-success evidence.

The source is the EXP-609 straight-through-continuation checkpoint with SHA-256
`cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd`.
Its 99,824-parameter shared cellular model is preserved as the no-message
function at initialization.

## Exact offline label recurrence

For legal Horner state `s`, multiplicand `a`, input bit `b`, and prime modulus
`p`, the unreduced value is `u = 2s + ba`. Since `0 <= s,a < p`, `u < 3p`, so
the offline quotient label is exactly `q in {0,1,2}` and the target is
`t = u - qp`.

At little-endian position `i`, let `s[i-1]` be zero for `i=0`. The signed-carry
label recurrence is:

`n[i] = s[i-1] + b*a[i] - q*p[i] + kappa[i]`

`kappa[i+1] = (n[i] - t[i]) / 2`.

Exact arithmetic is confined to offline labels and tests. The L0 domain
exhausts all `2 * p^2` legal tuples for primes 3 through 251, a total of
1,991,546, and separately checks 100,000 fresh transitions at each of widths 32
and 64. The required quotient vocabulary is `{0,1,2}`; the signed-carry
vocabulary is `{-2,-1,0,1}`; every recurrence numerator difference must be even;
and the one-position high tail must terminate at zero.

## Frozen split-convolution graph

The original EXP-609 stem, state normalization, output normalization, local
residue head, three-class quotient head, 184-by-98-by-3 gate convolution, and
92-by-98-by-3 candidate convolution retain their original tensors, channel
order, and floating-point operations. The rejected 102-channel expansion is not
used.

The extension adds:

- a learned local four-class signed-carry head: 372 parameters;
- learned `3 x 2` quotient and `4 x 2` carry embedding tables: 14 parameters;
- a separate zero-initialized `184 x 4 x 3` gate-message delta: 2,208
  parameters;
- a separate zero-initialized `92 x 4 x 3` candidate-message delta: 1,104
  parameters.

The total is exactly `99824 + 372 + 14 + 2208 + 1104 = 103522`, or
`1.037045...` times the source and below the 1.05 gate. Quotient softmax
posteriors are mapped to two learned channels and broadcast across positions.
Per-position signed-carry softmax posteriors are mapped to two learned local
channels. The four channels affect the learned recurrence only through the two
separate message convolutions.

At zero message-delta initialization the source gate and candidate
preactivations are unchanged. L0 requires exact maximum residue- and
quotient-logit differences of zero on 4,096 legal transitions at each of widths
32 and 64, plus identical thresholded residue predictions.

## Matched causal variable

Control and challenger instantiate the identical 103,522-parameter graph. Their
state-dictionary keys, shapes, values, initialization, source checkpoint, data,
optimizer, schedule, update count, and forward computation must match exactly.
The sole future difference is auxiliary supervision:

- control: quotient weight `0.0`, signed-carry weight `0.0`;
- challenger: quotient weight `0.10`, signed-carry weight `0.10`.

The control can still learn message use through residue loss, preventing added
capacity or compute from masquerading as trace-supervision evidence. A future
causal test separately zeros the soft quotient-message channels and the soft
signed-carry-message channels on the same cases. It never forces a class,
selects an expert, or applies an arithmetic candidate.

## Distinction from EXP-230 and Candidate 005

EXP-230 used a predicted quotient argmax to select one of three residue experts.
Its router reached 99.998% accuracy, but predicted routing gained only 0.19
rollout point, forced expert 0 lost only 0.13 point, and oracle routing changed
transition accuracy from 99.786% to 99.794%. EXP-658 has one residue path and no
selection operation; continuous learned q/c messages modulate that path.

Candidate 005/EXP-609 already has a quotient auxiliary head, but its rollout
discards quotient output. EXP-658 tests whether shared soft q/c representations
become causally useful inside the recurrent transition while preserving exact
source initialization.

## Trained-parameters boundary

`model.py` accepts only a floating six-plane feature tensor. It contains no
quotient or carry argmax, integer conversion, teacher-label input, explicit
carry recurrence, `q*p` subtraction, division, modular reduction, arithmetic
recombination, candidate generation, expert selection, or data-dependent
message route. Both messages are differentiable functions of learned logits and
learned embeddings, and trained weights must materially produce every answer.

## Staged future protocol

L1 resource calibration, L2 science, and L3 confirmation remain prohibited.
No root or authorization file is assigned.

### L1 — representative resource and feasibility calibration

A separate Judge must authorize one non-scientific calibration. After one
warmup, it performs three fresh matched repeats, reinitializing both arms before
each repeat. Every repeat executes 16 optimizer updates per arm—eight width 32
and eight width 64 in alternating order—at batch 128 through the exact future
forward, loss, backward, clipping, and optimizer path. Per arm, width, and
repeat it evaluates 512 natural transitions and 32 natural rollouts. The
challenger additionally evaluates 32 `zero-quotient-message` and 32
`zero-signed-carry-message` rollouts per width. Per width it rejection-samples 256
examples for every quotient class and 256 labeled positions for every signed-
carry class. Each arm is saved, SHA-256 hashed, reloaded, and checked for exact
equivalence once.

Candidate 005 and the EXP-658 challenger must be profiled in the same pinned
four-CPU candidate environment; adding measurements from different environments
is prohibited. For each route, warm one batch of four width-256 rollouts and
measure three fresh batches of eight. Record outer monotonic wall time,
aggregate process CPU including children, and peak RSS separately. Candidate
005's cached `I=156.0` seconds and derived `A=129.0` seconds are reusable only
when all identity checks match exactly: submission weights SHA-256
`2f6ac3199d0fd9e16afae165c2d4a16919e6c8bf4545363a6f12d077279d27ca`,
sandbox-record SHA-256
`bcf82f8bf3135a6857b0e30f01a60fbec62beeaa1e45e1f029aee39c9041a6fa`,
challenge and Dockerfile hashes, manifest and model hashes, engine-version
hash, image ID, four-CPU/8-GiB/2-GiB-tmpfs resource settings, and timeout. Any
mismatch makes both cached values unavailable and requires a fresh incumbent
measurement in the same environment. Without an available same-environment
incumbent result, the runtime gate fails closed and L2 cannot start.

Scale the slowest valid repeated update and per-case rates to every L2 count and
add measured fixed setup, save, and tracking work. Set separate wall and
aggregate-CPU soft budgets to `1.25 * expected`, hard runaway ceilings to
`1.75 * expected`, and the RSS guard to `1.25 * maximum observed RSS`.
Completed checkpoints and evidence must survive any stop; no denominator may be
shrunk to fit a budget.

Let `I_same_environment` be the validated cached or freshly measured incumbent
wall time and define the incremental allowance dynamically as
`A = 285.0 - I_same_environment`. The cached identity-matched values are
`I=156.0` and `A=129.0`; they are not defaults after invalidation. Let
`U100 = 1.25 * 100 * max(fresh challenger profile wall / 8)`. Pass only when
`I_same_environment` is available and `U100 <= A`. Reject when the incumbent is
unavailable or `100 * min(fresh challenger profile wall / 8) > A`. If valid
bounds straddle `A`, profile one exact 100-case deployed batch and pass only when
`1.10 * measured wall <= A`. Aggregate CPU is reported separately and never
substituted for wall. Any compliance, source-parity, parameter, memory, or
incremental-runtime failure prevents L2.

### L2 — width-32/64 matched falsifier

If separately authorized, both parameter-identical arms train for exactly 400
updates at batch 128, alternating widths 32 and 64 on identical frozen
train/test identities and batches. Save and hash both checkpoints before
evaluation. Per arm and width, evaluate exactly 20,000 natural transitions and
1,000 natural rollouts. On held-out primes, collect exactly 2,000 transitions
per quotient class and 2,000 labeled positions per signed-carry class at each
width, recording unbalanced class prevalence. For the challenger at each width,
pair 500 `zero-quotient-message` and 500 `zero-signed-carry-message` cases to the
first 500 normal rollouts. Repeat 100 normal rollouts per width twice for bit-
exact determinism. Rare quotient/carry recall gates bind only the challenger;
control recall is reported on the same denominators as a nonbinding diagnostic.

Every L2 gate is conjunctive:

- challenger recall for every quotient and signed-carry class at both widths has
  at most two errors in 2,000, equivalent to recall at least 0.999; control
  recall uses the same denominators but is diagnostic and nonbinding;
- width-64 transition has at most eight errors in 20,000, equivalent to exact
  accuracy at least 0.9996;
- width-64 challenger-minus-control rollout is at least +30 net correct in
  1,000, while width-32 retention is at least -10 net correct in 1,000;
- each zero-message intervention causes at least ten net lost correct cases in
  500 and has one-sided exact McNemar `p <= 0.01`;
- invalid rollouts are at most ten in 1,000 at each width;
- both 100-case repeats are bit-exact, and artifact/hash, parameter-equality,
  reload, RSS, wall, and aggregate-CPU gates pass.

Decision-safe futility stops a class diagnostic on its third error, width-64
transition on its ninth error, and validity on its eleventh invalid rollout.
Paired rollouts are processed in frozen 100-case blocks. With net gain `d_n`
after `n` of `N=1000`, stop the width-64 comparison when
`d_n + N - n < 30`; apply the same attainable-best bound against `-10` for
width-32 retention. Stop either 500-case intervention when current net damage
plus all remaining cases cannot reach ten. Training loss never shortens the 400
updates because it cannot bound final rollout. Only an all-gates pass authorizes
a separate L3 gate.

### L3 — serial two-root width-256 confirmation

L3 uses two preregistered fresh roots serially; their identities remain
unassigned at L0. Each root trains matched control and challenger arms for 400
updates at widths 128/256 under the same resolved architecture and loss
contract. Root two is never launched if root one fails any conjunctive gate.

Per root and arm, evaluate exactly 50,000 width-256 transitions, 1,000
width-256 rollouts, and 500 width-128 retention rollouts. Per root collect 2,000
examples for each quotient class and 2,000 labeled positions for each signed-
carry class. The challenger receives 500 paired `zero-quotient-message` and 500
paired `zero-signed-carry-message` interventions, 100 challenger rollouts
repeated twice for determinism, and an exact 100-case Tier-7 deployed-runtime
measurement in the pinned candidate environment. Rare recall binds only the
challenger; control recall is a same-denominator nonbinding diagnostic.

Every root must have at most 20 transition errors in 50,000, at least +30 net
width-256 rollout cases in 1,000, width-128 retention of at least -5 net cases
in 500, at most two challenger errors in every 2,000-item quotient/carry class,
at least ten net lost correct cases in each 500-case zero-message intervention
with one-sided exact McNemar `p <= 0.01`, at most ten invalid width-256 rollouts
in 1,000, and exact 100-case determinism. With declared headroom, incremental
100-case wall must be at most `A = 285.0 - I_same_environment`; cached `I=156.0`
and `A=129.0` apply only after the exact L1 artifact/environment identity gate,
and unavailable `I` fails closed. CPU and RSS must remain inside L1-calibrated
bounds. Across both roots, the aggregate
width-256 challenger-control gain must be at least +60 net correct out of 2,000,
and every per-root gate must pass.

L3 uses the corresponding exact attainable-count bounds from L2: stop a class
on its third error, width-256 transition on its twenty-first error, validity on
its eleventh invalid rollout, a paired or causal comparison when all remaining
favorable cases cannot reach its gate, and the entire two-root protocol after a
root-one failure. Training loss is not an early-stop signal. Passing L3
authorizes only a separate frozen Candidate 006 integration/release package; it
is not checked, five-seed-independent, private, or Goal-completion evidence.

Failure of any class-recall, causal, transition, rollout, retention, validity,
integrity, runtime, or compliance gate closes this formulation. Label accuracy
without rollout improvement, inert messages, or any need for explicit
arithmetic recombination is a rejection.
