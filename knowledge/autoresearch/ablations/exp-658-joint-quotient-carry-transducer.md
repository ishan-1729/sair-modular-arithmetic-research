---
type: Experiment Preregistration
title: EXP-658 joint quotient/carry learned transducer
description: Frozen L0/L1 contract and the T045 terminal park evidence for the current joint quotient/carry execution formulation.
status: parked-current-execution-formulation-after-two-repair-limit
tags: [autoresearch, quotient, signed-carry, recurrent, transducer, neuro-symbolic]
timestamp: "2026-07-22T02:00:00+00:00"
---

# EXP-658 joint quotient/carry learned transducer

## Evidence boundary

The T024/T041 publication boundary preserved the completed L0 graph, exact
offline trace labels, source-parity tests, frozen hashes, and inert
T028-resolved L1 scaffold. T045 later authorized one source-bound vertical L1
execution with conditional L2. The current terminal status is therefore no
longer “inert, unrun”: the L1 process executed but failed before canonical
resource/calibration persistence, and the formulation is parked after the
PM-directed two-repair limit. The historical contract below remains frozen;
the terminal evidence in the next section is authoritative for current status.

The source is the EXP-609 straight-through-continuation checkpoint with SHA-256
`cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd`.
Its 99,824-parameter shared cellular model is preserved as the no-message
function at initialization.

## T045 terminal execution decision

Source commit `25d21615c2c7d8b4302d2661c228b828ff07cb8b` was clean, pushed,
and bound by authorization SHA-256
`acdba3226161c4492f37a8a276ae0417527ae9aa827980de6971ac89bbf6cf69`.
The restored executable, config, and L0-test hashes are respectively
`1d4e7ecb86d6c213f67e87a8f7ddda62e3a319affe2d6ce20bb7cd1e247e0544`,
`59c04839d7041000b8b2211f4bb5da99a03dfac8804c4e79d663c0a6eeefc177`,
and `8c6a4b3352e2229d3fded622637f350b56f6a99ec42771f664cb5a98f33ebbd6`.
On those exact bytes, the focused source suite passed 51 tests in 492.40
seconds and the complete source suite passed 351 tests in 543.06 seconds.

The pinned CPU execution used image `modchallenge-sandbox:82510bba00a1`
(image ID
`0ff410bfb61c61d6a07b17b11337694a69910874cb59ab109b1128b3224a8423`),
four CPUs with affinity `[0,1,2,3]`, 8 GiB memory, 2 GiB tmpfs, PyTorch
2.5.1 CPU, deterministic algorithms, and full preregistered fidelity. It ran
through matched L1 training/evaluation, both checkpoint roundtrips, the warmup
width-256 profile, and all three fresh width-256 profiles. It then failed in
`_post_repeat3` at `execution.py:547` while reading absent receipt key
`checkpoint_sha256`; the canonical receipt key was `sha256`.

The exact observed outer wall time is 381.7 seconds. Aggregate process-tree CPU
time and peak RSS were not captured, so both remain null rather than inferred.
The failure occurred before operation records, `l1-calibration.json`, a dynamic
budget, an environment record, MLflow, or `decision.json` were persisted. No L1
scientific accuracy, loss, recall, prediction, correct/error count, causal
effect, or selection value is durable; L2 never ran.

The raw authorization file (2,933 bytes, SHA-256
`7e288cd25719fc999c78623a91cceb629a3e1697796e0b528a3e252aeb84933f`),
challenger checkpoint (415,912 bytes, SHA-256
`2b4a1ded31711ee8425457fe516358608f22da40c6f1ce374897bc0174c442c9`),
and control checkpoint (415,912 bytes, SHA-256
`8cfa448b59b21ea8e187e3e771d279dad9b73b6b67744e9b0fd94c0897fc5e6b`)
are preserved as non-scientific ignored scratch evidence under
`.scratch/exp-658-joint-quotient-carry-transducer/failed-attempt-25d216-runtime-keyerror/`.
They are not portable, promotable, or authorized for checkpoint reuse.

The first uncommitted repair corrected the receipt key but the full oracle
stopped at 350 passed / one failed because the tracked failure archive violated
the frozen zero-root regression. The second uncommitted repair changed that
regression and stopped at 348 passed / three failed because the frozen L0-test
identity changed. Both files were restored exactly to the pushed source bytes.
The PM-directed limit forbids a third repair or rerun of this execution
formulation. The durable decision is
`parked-current-execution-formulation-after-two-repair-limit`. This is not a
scientific falsification of learned quotient/carry messages and not Goal
completion. Candidate 005, checked/independent/official/private evidence,
`submission/`, EXP648, and EXP656 are unchanged.

The next material decision is which materially distinct experiment family has
the highest expected information gain per unit wall and aggregate-CPU compute.
Evidence capable of changing that decision is one separately authorized,
full-fidelity vertical falsifier that records exact scientific metrics and
separate wall, aggregate CPU, and RSS measurements, or a durable admission
reject for that family.

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

Control and challenger instantiate the identical 103,522-parameter graph. At
initialization their state-dictionary keys, shapes, dtypes, devices, trainable
masks, and values match exactly in independent storage. After the first update,
values are expected to diverge under auxiliary supervision while keys, shapes,
dtypes, devices, trainable masks, optimizer topology, current learning rate,
scheduler position, update count, data, and forward graph remain matched. The
sole future difference is auxiliary supervision:

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

L1 execution, L2 science, and L3 confirmation remain prohibited. The complete
L1 scaffold has no assigned root, authorization, output, numeric resource
envelope, or execution path. A later clean, pushed scaffold commit and separate
Judge authorization are required before any L1 side effect.

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

Only the exact repeat-3, challenger, ordinal-2 reloaded EXP-658 object is
microprofiled in the pinned four-CPU candidate environment. It warms four
width-256 rollouts and measures
three fresh disjoint batches of eight, recording inner and outer monotonic wall,
aggregate process/cgroup-tree CPU, and simultaneous tree-peak RSS separately.
Candidate 005 is never substituted with a microprofile: its cached `I=156.0`
seconds and derived `A=129.0` seconds are reusable only
when all identity checks match exactly: submission weights SHA-256
`2f6ac3199d0fd9e16afae165c2d4a16919e6c8bf4545363a6f12d077279d27ca`,
sandbox-record SHA-256
`bcf82f8bf3135a6857b0e30f01a60fbec62beeaa1e45e1f029aee39c9041a6fa`,
challenge and Dockerfile hashes, manifest and model hashes, engine-version
hash, image ID, four-CPU/8-GiB/2-GiB-tmpfs resource settings, and timeout. Any
mismatch makes both cached values unavailable and requires a fresh incumbent
measurement of the unchanged, pinned-image, full 1,100-case official surface in
the same environment. Without an available same-environment
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

#### Resolved L1 mechanics and ownership

Each arm uses an independent AdamW optimizer over all 103,522 trainable
parameters in one identical name-ordered group, with learning rate `5e-6`,
weight decay `2e-5`, and `CosineAnnealingLR(T_max=400, eta_min=5e-7)`.
The source-resolved AdamW defaults are frozen explicitly: betas `(0.9,0.999)`,
epsilon `1e-8`, no AMSGrad/maximization/capture/differentiation, and
`foreach=None`, `fused=None`. Validation independently binds optimizer default
LR, parameter-group `initial_lr`, scheduler `base_lrs`, and the analytically
expected current cosine LR at every step.
Every step computes all three losses in both arms. Residue loss is
`mean(BCE_none) + 0.45 * max(mean_batch(BCE_none) per position)`. Quotient loss
is mean cross entropy over classes `0,1,2`; signed carry maps
`{-2,-1,0,1}` to `{0,1,2,3}` and uses mean cross entropy over all `B*W`
positions. The control weights the two auxiliary losses `0/0`; the challenger
weights them `0.10/0.10`. There is no smoothing or extra regularizer. Backward
is followed by global norm clipping at `2.0`, then optimizer step, then
scheduler step. Execution is deterministic CPU FP32 without AMP, using four
intra-op and one inter-op thread.

For warmup repeat 0 and fresh repeats 1–3, the control initializes from the
frozen source plus namespace-seeded extension tensors; the challenger is an
exact state-dictionary copy in an independent model. Each matched batch is
materialized once and immutable. Identical Torch RNG state is restored before
each arm forward, and arm order is control/challenger when
`(repeat + global_step) mod 2 == 0`, otherwise challenger/control. Parameters,
logits, hidden state, gradients, optimizers, schedulers, and forward results are
independent. Exact values are required only at initialization. After updates,
keys, shapes, dtypes, devices, trainable masks, optimizer topology, learning
rates, and step counts remain matched and finite, while at least one
auxiliary-connected tensor must differ after step 1 and remain different through
step 16. Steps 2–16 do not require or permit whole-model value equality; every
entry and exit instead checks matched geometry/masks/topology/LR/scheduler/step
counts, independent parameter/optimizer/result storage, and finite parameters
and gradients.

All streams derive from base seed `20261176` by taking the low 63 bits of the
big-endian integer represented by the first eight SHA-256 bytes of UTF-8
`exp658|20261176|<namespace>|<pipe-joined parts>`. The namespaces are `split`,
`init`, `train`, `eval_transition`, `eval_rollout`, `rare_q`, `rare_carry`,
`profile_warmup`, and `profile_fresh`. Widths 32 and 64 reproduce the frozen
EXP-609 prime split exactly: seed `random.Random` with the domain-separated split
seed, draw in `[1<<(W-8), 1<<W)`, choose the next prime strictly above the draw,
discard primes at or above the upper bound and duplicates, then take the first
512 as train and next 128 as test. The frozen source SHA-256 is
`9725303f4151554cb1f877cd92a2db6df911726f27f93fe5798c6d6e9706506f`;
the independently derived canonical hashes are
`f014bd47e7eeb0f3e3f8c43c7422cb154d9ce9fd93bc840cb6eaf473633a67bd`
at width 32 and
`f49a8a17c3e3a40b8ee96d2ed5d8648038af79b46ed7e645d06b83e870025fd7`
at width 64. Every identity is independently checked for primality, span, order,
uniqueness, and train/test disjointness. Width 256 uses one canonical
domain-separated held-out stream seeded by
`profile_warmup|heldout-width256-prime-stream|256`: 28 unique positive primes
from bit lengths 249 through 256, with the first four assigned to warmup and the
next three ordered groups of eight assigned to fresh repeats 1–3. Its persisted
hash is
`623b459ecca48cd83dbebdd64c57afccf4d4dfd264da763f956c818e565582ee`.
Prime selection is pinned to SymPy 1.13.1 `nextprime`; tests independently
reconstruct all 28 integers and the canonical hash without calibration helpers
or the frozen 64-bit primality predicate.
Training owns train identities; all
transition, rollout, rare, intervention, determinism, and profile work owns only
test or held-out identities. Checked, independent, private, and official answers
are forbidden. Repeat 0 is warmup and contributes no rate or evidence
denominator; repeats 1–3 use fresh initialization and streams. Only split and
stream hashes may persist. Tuple uniqueness is checked across each complete
repeat stream and across fresh repeats; interventions and determinism are the
only explicit identity reuses. Rare q/c sources are disjoint, repeat-3-owned,
and test-only; the four warmup plus three-by-eight profile cases are mutually
disjoint held-out width-256 identities.

The complete manifest is not accepted from an authorization caller. Given the
future numeric attempt allowance, calibration internally regenerates every case
from the frozen seeds, splits, EXP-300 `BigIntSampler` draw order, quotas, and
held-out streams. Transition identity is exactly
`(state,multiplicand,modulus,operand_bit)`; a carry item adds exactly `position`;
rollout and profile identity is exactly `(multiplier,multiplicand,modulus)`.
Prefixes, serials, cycling, and hidden suffixes are prohibited. A different
otherwise-valid stream fails canonical seed-to-case equality.

The 16 alternating width steps use exact per-appearance rows in
`[natural,q2,boundary,rare_carry]` order. For width 32, appearances 1–5 are
`[64,26,19,19]`, then `[64,25,20,19]`, `[64,25,20,19]`, and
`[64,25,19,20]`. For width 64, appearances 1–5 are `[64,26,19,19]`, then
`[64,25,19,20]`, `[64,25,19,20]`, and `[64,25,20,19]`. Per repeat this totals
1,024 natural, 410 q2, 307 boundary, and 307 rare-carry rows. Selection precedence
is natural first, q2 excluding selected, boundary excluding q2/selected, then
rare carry excluding q2/boundary/selected; natural semantic overlap is recorded.
No caller-supplied eligibility boolean is accepted. The scaffold validates raw
`u=2s+ba`, `p`, `q=floor(u/p)`, and the exact carry recurrence. Its predicates
are `q==2`, `20*min(|u-p|,|u-2p|)<=p`, and presence of signed carry `-2` or
`+1`.
After a later-authorized frozen deterministic attempt allowance, only a training
deficit may be filled with unique natural rows, recording requested, realized,
and fallback counts without shrinking any batch or update. Rare evaluation has
no fallback. L2 repeats this exact 16-step schedule 25 times.

Every L1 repeat executes the full matched core: 16 updates per arm, 512 natural
transitions and 32 normal rollouts per arm and width, plus 32 paired zero-q and
32 paired zero-carry challenger rollouts per width. Rollout starts from zero,
consumes exactly `W` multiplier bits from MSB to LSB, thresholds residue logits
at zero, decodes little-endian only to form the next generic feature, retains
invalid states, and never repairs them. After fresh repeat 3, the challenger
alone receives 256 unique held-out transitions per quotient class and 256
unique held-out transition-position items per carry class per width. Every rare
item persists raw state, multiplicand, modulus, operand bit, and (for carry)
position; the scaffold recomputes `q` and the signed-carry recurrence and rejects
caller-mutated labels, duplicate positions, or q/c source overlap. The q and
carry datasets are disjoint, resource-timed, and scientifically undisclosed.
The first 32 repeat-3 challenger normal rollouts per width are repeated once and
must be bit-exact. Each arm undergoes exactly one save/hash/reload cycle with
bit-exact tensors and fixed-probe outputs, and the reloaded challenger is chosen
by ordinal rather than performance. Checkpoint integrity returns that exact
reloaded model handle; profiling cannot trigger an uncounted second reload.

The deployed-path profile uses that reloaded challenger in the pinned four-CPU
candidate environment. Four disjoint held-out width-256 cases warm the model path;
three further mutually disjoint batches of eight are measured. Every fresh
batch records inner and outer monotonic wall time, aggregate process/cgroup-tree
user-plus-system CPU, and maximum simultaneous tree RSS. CPU is never estimated
as wall time times threads and serial RSS peaks are never summed.

Candidate 005's cached `I=156.0` and `A=129.0` require exact matching of both
recorded artifact hashes and the entire environment identity, including the
official 1,100-case seed/surface, runner/generator/prime/environment hashes,
image/engine, CPU affinity and count, memory, tmpfs, timeout, device, precision,
and thread settings. Any mismatch invalidates both values. Replacement is only
an unchanged full 1,100-case Candidate 005 run on the same official input
surface in the same environment; a microprofile is prohibited.

Resource accounting uses physical components. Shared data generation is
charged once, model work is charged per arm, intervention and determinism
baselines reuse the frozen natural cases without a second charge, and the full
Candidate 005 rerun is charged only after cache invalidation. Warmup rates do
not enter evidence. After warmup, remaining L1 work is projected; after fresh
repeats, L2 is projected from the maximum matching mean rate per component plus
measured fixed work. Separate expected wall and aggregate-CPU budgets use 1.25
soft and 1.75 runaway multipliers; RSS uses 1.25 times the maximum simultaneous
tree peak. There is no arbitrary short cap and no fidelity or denominator
reduction. A ceiling crossing or exact impossibility may stop only at a safe,
preserved checkpoint; training loss is never a stop signal.

Every budget-eligible observation is derived by reloading a canonical,
path-backed measurement-record file. Production exposes no operation callback:
operations must be fixed code entries in a closed internal registry, selected
only by reviewed operation ID. That registry is empty in this inert scaffold,
and the production measurement stub therefore rejects before a meter or
operation can run. A later separately reviewed code package must add both exact
registry metadata and the corresponding internal implementation; config and
authorization payloads cannot register or supply one. Generic source/code hashes
are not claimed to bind closure cells, referenced globals, mutable module state,
or runtime inputs. Registry review plus the absence of an external callback is
the operation trust boundary. Each derived descriptor binds operation ID,
physical/rate key, width, arm, source, units, input/workload-manifest hash, and
live environment. A record preserves that descriptor together with raw counters,
recomputed observation, meter implementation, prior-authorization hash, and its
own hash; the loader rechecks all of them from canonical file bytes.
CPU comes from cumulative cgroup user-plus-system counters and RSS from
cumulative `memory.peak`. Authorization live-probes Torch version, default
dtype, determinism, four/one thread settings, exactly four affinity CPUs,
four-CPU quota, 8-GiB memory, and 2-GiB tmpfs. Image, engine, and timeout remain
explicitly recorded cache provenance, not current-environment proof. Missing,
duplicate, warmup-as-fresh, unmatched, non-positive/non-finite, invalid tree-RSS,
or overlapping fixed work fails closed. Remaining-L1 and L2 manifests enumerate
every physical operation; reused baselines are absent, model work is per arm,
and cache-dependent full-1,100 work is selected only by a live path-validated
`CacheVerdict`, never a boolean.

The production remaining-L1 resolver loads only the tracked config and fixed
repository/evidence root; lower-level injected-root helpers are private test
mechanics and cannot authorize a launch. It accepts only the complete
path-backed warmup manifest. Matching rates use the maximum validated warmup mean. EXP-609
and EXP-643 are coarse lineage context only: their aggregates are never divided,
duplicated, or relabeled as disjoint component rates. Coverage classes preserve
rate key, width, arm, and source exactly; no width/arm equivalence is currently
preregistered. Thus W32 cannot cover W64 and challenger cannot cover control.
The L1 resource manifest includes control and challenger q/c forward probes so
all L2 arm classes are measurable without releasing extra scientific values.
Every canonical unresolved q/c, determinism, checkpoint, profile, or
cache-invalid incumbent physical unit requires its own prior-authorized record,
and maximum matching rates are selected across all records in each exact class.
Those records and the internal operation registry are currently absent, so the
production resolver and authorization fail closed naming all seven categories
plus exact missing coverage/physical-unit detail; no numeric EXP-658
authorization is constructible. The
resolver binds observation, projection, exact evidence, live environment,
warmup records, and derivation hashes. Expected wall and aggregate CPU,
the 1.25/1.75 headroom envelopes, and the 1.25 RSS guard are recomputed from
that derivation rather than accepted as caller-provided values.

A later canonical authorization must bind the parent L0 and clean, committed,
pushed scaffold commits; all scaffold and frozen hashes; source checkpoint SHA;
Candidate 005 artifact/evidence identity; the internally regenerated complete
W32/W64/W256 raw identity manifest and nested workload; exact seeds and splits;
a live-probed current environment with separately labeled recorded cache
provenance; an unused root and output; path-backed warmup records; and numeric
wall, aggregate-CPU, and RSS envelopes derived only from exact one-to-one
component evidence. Every later physical unit also has an exact operation
descriptor derived from the internal registry and bound to the workload hash
and live environment; caller-authored operation manifests are rejected. The production
validator accepts only an authorization path under the fixed repository root,
loads the tracked config internally, and accepts no caller config, evidence or
repository root, `expected`, environment, identity manifest, or observation
trust root. Missing, malformed,
mismatched, dirty, unpushed, or reused bindings reject before output allocation,
MLflow, model/optimizer construction, Candidate 005, or workload execution. Even
a fully matching fixture remains inert in this package. A future resource-only
receipt may contain identities, counts, hashes, shapes, integrity and
determinism booleans, fixed-root measurement-record descriptors, a canonical
post-L1 dynamic-budget derivation, cache verdict, command, and budgets; scientific
losses, recalls, predictions, errors, correct counts, accuracies, and scores are
prohibited.

The authorization schema is exact and closed. It independently rechecks the
clean pushed commit, every scaffold/frozen/Candidate artifact path and hash, the
parsed sandbox/environment records, split/W256/workload hashes, exact four-CPU
affinity and live resource limits, pinned Torch/image/cumulative-meter identity,
a future numeric attempt allowance, an unused root/output, and the exact
path-backed budget derivation. Neither the attempt allowance nor any numeric
budget has a default in this scaffold. A resource receipt must reproduce the
entire canonical stage/cache manifest (never a subset) and bind exact
authorization, live-environment, warmup-observation, warmup-measurement,
scaffold, workload, budget-derivation, cache-evidence, and meter hashes. The
production validator accepts canonical receipt and authorization paths only
under the fixed evidence tree. It reloads every record named by the receipt,
checks file and record hashes, prior authorization, live environment, meter,
operation descriptor, and physical unit, then derives and validates the full
observation from raw cumulative counters. Warmup records rebind to their prior
measurement authorization; later records rebind to the full L1 authorization.
From those reloaded records it independently reconstructs a canonical post-L1
derivation binding the authorization, live environment, cache evidence, exact
L2 projection manifest, record descriptors and hashes, full derived
observations, maximum exact coverage-class rates, expected wall/aggregate CPU
and maximum RSS, 1.25/1.75/1.25 envelopes, and final derivation hash. Receipt
omission or even locally self-consistent mutation is rejected by recomputation.
Synthetic `ResourceObservation` objects and plausible hash-shaped identifiers
are never a production input, and co-updated receipt-local values cannot replace
authorization-bound evidence. Exact keys and types at every nesting level
prevent free-form maps from hiding scientific values.

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
- both 100-case repeats are bit-exact; initialization parameter values are
  exact; post-update parameter structure remains exact while values are
  expected to diverge; and artifact/hash, reload, RSS, wall, and aggregate-CPU
  gates pass.

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
