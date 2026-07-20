---
type: Experiment Preregistration
title: EXP-649 stochastic trajectory breadth
description: Calibration-gated frozen width-256 oracle-breadth and candidate-shaped runtime test with six stochastic paths and two exact controls.
status: planned-blocked-formal-runtime-gate-miss
tags: [autoresearch, tier7, stochastic-inference, trajectory-breadth, oracle-coverage, runtime]
timestamp: "2026-07-20T03:07:13+05:30"
last_updated: "2026-07-20T09:42:00+05:30"
---

# EXP-649 stochastic trajectory breadth

## Status and question

EXP-649 remains a selected, planned Tier-7 research item, but its complete input-only calibration now has the exact machine verdict **`runtime-gate-miss`**. This frozen eight-path formulation is permanently blocked from formal work. The result is not `completed-rejected`: calibration measured resources and integrity without labels or oracle, so stochastic breadth, validity, and accuracy remain untested. Formal root `20261149` remains unopened.

Candidate 005 remains the active incumbent at 59.5% checked-public and 59.3% independent overall accuracy. This is not Goal completion: the 85% overall gate remains unmet, and Candidate 005 currently has no learned inference path above 128 bits.

The preregistered falsifiable question was whether six fixed stochastic trajectories plus exact original/swapped controls from the frozen raw cellular checkpoint contain the correct answer on strictly more than 90% of at most 200 untouched width-256 cases while their complete candidate-shaped CPU workload remains within 285 wall seconds. Calibration falsifies only the runtime conjunct. It does not select an output and provides no evidence about the unexecuted breadth conjunct.

## Evidence basis and K=6 planning rationale

The dated [primary-source note](../research/2026-07-20-stochastic-recursive-reasoning.md) records source claims, browser observations, prior SAIR evidence, and project hypotheses separately. PTRM reports Gaussian perturbations across deep recursion with `K=25` to `K=100` depending on the task. EXP-649's `K=6` is a compute-bounded transfer, not a replication of PTRM, GRAM, EqR, or TRM.

The nearest SAIR observation is the separately trained EXP-647 challenger, whose original/swapped width-256 oracle union is 47.5%. If six hypothetical additions were mutually independent and each solved 25% of the failures remaining after that union, coverage would be `1 - 0.525 * 0.75^6 = 90.66%`; five would reach only `1 - 0.525 * 0.75^5 = 87.54%`. Those independence and 25% assumptions are planning arithmetic only. They are not matched Candidate-005 evidence and make no prediction about EXP-649.

## Frozen source and no-training boundary

The stochastic research path loads the raw cellular source checkpoint with SHA-256 `cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd`. The complete Candidate-005 artifact used for the matched full-workload timing control has SHA-256 `2f6ac3199d0fd9e16afae165c2d4a16919e6c8bf4545363a6f12d077279d27ca`.

No parameter, buffer, threshold, path, orientation, seed, sigma, or injection site may be learned or changed. The runner must hash the state dict before and after work and require equality. Exact arithmetic is confined to offline input generation, the competition-sanctioned pairwise input reductions `a % p` and `b % p`, and, only after later authorization, offline scoring. The trained cellular transition must materially produce the final modular product residue; there is no hand-coded multiplication, long division, by-construction final reduction, CRT, or output repair.

## Eight frozen paths

All eight paths are evaluated for every case in the exact order below. The six stochastic paths are split into three original and three operand-swapped orientations.

| Path | Orientation | Noise seed | Sigma |
|---|---|---:|---:|
| `control_original` | original | none | 0.0 |
| `control_swapped` | swapped | none | 0.0 |
| `noise_649001_original` | original | 649001 | 0.2 |
| `noise_649002_swapped` | swapped | 649002 | 0.2 |
| `noise_649003_original` | original | 649003 | 0.2 |
| `noise_649004_swapped` | swapped | 649004 | 0.2 |
| `noise_649005_original` | original | 649005 | 0.2 |
| `noise_649006_swapped` | swapped | 649006 | 0.2 |

Noise is independent CPU `float32` Gaussian noise at absolute sigma 0.2, with no clipping or rescaling. There are exactly seventeen injection sites per multiplier bit: first `h = stem(x) + epsilon_0`, then one perturbation before each of the sixteen recurrent visits, `h = recurrent_update(h + epsilon_j, x, dilation_j)` for `j = 1..16`. Noise coordinates are fixed by path seed, case index, multiplier-bit index, and injection index so batching and reruns cannot silently change a trajectory.

Continuous hidden state exists only within the sixteen visits for the current multiplier bit. Between multiplier bits, the model carries only the hard-thresholded integer-residue bit state. No noisy or continuous hidden tensor persists across that boundary. The two sigma-zero paths must exactly reproduce the legacy original and swapped paths.

## Calibration identities and seed roles

Former calibration root `20261148` is permanently retired and denylisted. During scaffold verification, an official-generator parity test computed expected values in volatile memory for its 1,100 identities. No expected value was inspected or persisted and no calibration/formal runner executed, but label generation alone breached the input-only contract. The durable [incident and prior-use audit receipt](../analysis/exp-649-calibration-seed-retirement.md) records the command, scope, source, worktree state, limitations, retirement, and corrective Judge decision.

Replacement calibration root `20261150` had no prior workspace, HEAD, or all-reference Git-history role before selection. It remains unopened by tests: scaffold parity uses only distinct test seed `649900001`. Formal root `20261149` also remains unopened. These literal-level checks authorize later input generation from the replacement calibration root only; they are not identity-level disjointness evidence and do not open the formal root.

Calibration may use only newly generated training-role input identities from root `20261150`. It is input-only: no expected answers, prefix labels, exact final labels, oracle membership, checked answers, independent answers, or formal identities may be generated, loaded, or scored. The runner rejects the retired root and records the complete calibration identity manifest and its hash. Formal root `20261149` remains unopened and the 200 formal cases remain untouched during calibration.

After a calibration pass, and before any formal scoring or model execution on formal inputs, a later Judge must review an identity-level collision audit comparing every proposed formal identity against calibration, training, validation, independent, checked, and prior-formal identities. Case-level or seed-literal disjointness alone is insufficient. A failed or incomplete audit keeps formal execution locked.

## Binding candidate-shaped runtime calibration

Calibration times the exact same-process workload that would matter for a candidate:

1. Run the complete Candidate-005 artifact on all 1,100 rows, including production preprocessing, returned-count validation, and official answer decoding inside the timed block.
2. Run every one of the eight frozen paths on exactly 110 width-256 rows: the 100 Tier-7 rows plus the ten Tier-0 rows from the 65–128-bit operand band. The timed portfolio dispatch includes the competition-sanctioned pairwise `a % p` and `b % p` input reductions before learned recurrence.
3. Preserve canonical tier, case, and path order and the exact production batch geometry. No caching, path omission, shortened recurrence, reduced case count, or lower-fidelity surrogate is allowed.

The input-only calibration generator is a label-free reproduction of the input side of upstream challenge commit `82510bba00a1126649bd76dd1a451f14d0b3eb60`. It must preserve the pinned official tier table, HMAC domains and RNG order, five generated prime identities per scored tier, four leading edge rows per scored tier, random prime selection for non-edge rows, and the single-stream Tier-0 sub-band behavior. It may construct only `(tier_id, case_id, a, b, p)` inputs and identity metadata; calling the official target-generation path or computing/storing expected answers is forbidden. The runner pins and verifies the official `config.py`, `generator.py`, `primes.py`, `pipeline.py`, `decoder.py`, and `loader.py` source hashes before work.

Use four PyTorch threads and one interop thread. Execute one complete untimed warmup, reset deterministic path generators, then execute three complete matched timed replicates. Record end-to-end wall time for each complete replicate and process user-plus-system CPU time separately. The binding upper projection is

`1.20 * max(three complete replicate wall times) + 5 seconds <= 285 seconds`.

Portfolio-only timing may be reported diagnostically, but only the complete same-process workload above can pass the wall gate. Process CPU is a separately reported resource diagnostic and is never substituted for wall time. Cached timing or timing measured in a different environment cannot produce a verdict; a cross-environment result requires a new matched calibration.

The resource-budget program is binding. Its initial generous calibration ceiling is only runaway protection, not a target duration. The program may resolve resource and resumability fields from measured work, but it may not impose an arbitrary short cap or reduce paths, rows, visits, warmup, replicates, logging, hashing, or any other fidelity to fit one.

Completed calibration replicates may be resumed only from an integrity-protected partial artifact whose schema, complete cache material, source/config/dataset/environment hashes, per-replicate structure, finite nonnegative timings, output hashes, and content chain all validate exactly. A resumed process performs a new complete warmup, and every reused replicate remains charged to the binding maximum and resource ceilings. A cache mismatch is discarded as evidence rather than repaired in place.

The first implementation-readiness audit blocked launch before any calibration input was generated. The repair package now retains and compares every decoded prefix for both sigma-zero controls, enforces the charged wall/process-CPU runaway ceilings within long stages and after bookkeeping, compares the current formal environment/preregistration/source-note hashes against calibration provenance, and pins a machine-readable seed-retirement manifest. Focused and full tests, project validation, frozen hashes, diff checks, and scratch-absence checks passed; a second independent Judge gate approved commit followed by clean-tree input-only calibration. Formal work remains forbidden.

## Locked formal evidence

Formal execution is forbidden until the complete calibration passes all integrity and runtime checks and a later Judge authorizes `judge-approved-exp-649-formal`. That later authorization must follow a clean-source/hash review and a passed identity-level collision audit over the proposed formal identities. A calibration pass is necessary but not sufficient.

If authorized, formal evidence uses at most 200 untouched width-256 cases generated from the still-unused root `20261149`. Every case receives all eight paths. The runner records every prefix and final output before scoring. The complete formal work is then rerun exactly with reset generators; outputs, noise coordinates, prefix-validity flags, final-validity flags, and oracle counts must reproduce exactly.

The frozen breadth gate is at least 181 correct oracle-union cases out of 200, strictly greater than 90%. Oracle union is an offline diagnostic: a case counts if any of the eight fixed paths is exactly correct. It may never choose the submitted output. Report an exact binomial interval with the count.

Each path may have at most four invalid completed cases. A case is invalid for a path if **any** decoded prefix or the final decoded residue falls outside the half-open interval `[0, p)`. Stop for futility as soon as either condition becomes irreversible:

- twenty oracle failures have occurred, because at most 180/200 could then be correct; or
- any one path has accumulated five invalid completed cases.

Stopped formal work is a recorded gate failure, not an invitation to replace, resample, filter, reorder, or tune cases or paths. Selector training, selector evaluation, voting, majority choice, margin choice, convergence choice, answer-dependent routing, seed sweeps, sigma sweeps, and path pruning are prohibited in EXP-649.

## Formal resource projection rule

Calibration must project the complete later formal workload from measured path geometry, including all eight paths on all 200 cases, one complete exact rerun, both exact sigma-zero legacy-control checks, scoring, hashing, serialization, scratch-artifact writing, and tracking. Scale the vectorized portfolio measurement from `8 x 110` to two complete `8 x 200` passes; scale a separately measured two-control legacy verification from `2 x 110` to `2 x 200`; and add measured schema-equivalent, target-free `200 x 8` prefix/validity bookkeeping. Formal soft and hard wall/CPU budgets are respectively `1.5x` and `2.0x` that measured projection, with wall and process CPU kept as distinct units. These multipliers do not authorize reduced evidence or a cross-environment verdict.

## Decision boundary

A calibration miss leaves EXP-649 planned but blocked from formal work. A later formal breadth, validity, determinism, or resource miss rejects this exact six-path, sigma-0.2, seventeen-site formulation without tuning on the observed cases. A formal pass would authorize only a new, separately preregistered target-free selection study after Judge review. It would not authorize voting, a selector, Candidate-005 modification, Tier-7 promotion, checked evaluation, private claims, or Goal completion.

## Calibration result

The clean-tree calibration executed from repository commit `5519b5173118e7a0166096e0d5fdbca13c4ef2b4` with frozen scientific hash `ea70a40c01e1c407be05ce7d5e7740862e670a0dd36715635614c521cae3415a`, resource hash `8d13c0f5d6a859a36d9360ad4209b492bc10dae42a6e6be80c0d9bdc5b3fb7cd`, executed config hash `99b22e87eb6a55c6d362ae350dc35c0de68d2f3c8ee99befcdff829d6056f134`, and runner hash `04321d39de730cc653535faa8928ba7a0e975868d274503a5e53714f41daecda`. The final label-free [calibration artifact](../../../experiments/runs/exp-649-stochastic-trajectory-breadth/calibration.json) has SHA-256 `655e160539f72105da0ee5c809aaa7f716c8cad56c26f78d62e6ba207e44cbbf`; its integrity-protected partial has SHA-256 `df646286170cc773fa67cefc3e26a466d067d1921fe679a4b92b39868c83ffcc`. The [canonical run record](../../../experiments/runs/exp-649-stochastic-trajectory-breadth/run.json) and [metrics](../../../evals/results/exp-649-stochastic-trajectory-breadth/metrics.json) preserve the compact provenance and verdict.

Observed full-workload timing on the frozen four-thread WSL environment was:

- warmup: `3158.666861` wall seconds and `12661.12` process-CPU seconds;
- replicate 1: `2165.354970` wall seconds and `8283.57` process-CPU seconds;
- replicate 2: `2131.616994` wall seconds and `8131.46` process-CPU seconds;
- replicate 3: `2621.576302` wall seconds and `10619.47` process-CPU seconds.

The three candidate and portfolio output hashes are identical to the warmup hashes. The source state dictionary is bit-identical before and after the run. Separately measured two-orientation legacy verification takes `382.509272` wall / `1605.44` process-CPU seconds and exactly matches both final outputs and every retained decoded prefix. Target-free schema-equivalent `200 x 8 x 256` bookkeeping takes `0.115700` wall / `0.07` process-CPU seconds. Calibration generated 1,100 input-only rows from audited root `20261150`; its artifact records `contains_labels_or_oracle: false`.

The binding derived upper wall projection is `1.20 * 2621.576302 + 5 = 3150.891563` seconds. Because `3150.891563 > 285`, the candidate wall gate fails. The measured formal projection of `10037.759113` wall seconds is counterfactual resource planning only; it cannot authorize formal work.

The first host execution completed the warmup and two timed replicas before its wrapper was interrupted during an incomplete third replica. No incomplete unit entered the cache. The recovery process performed the mandatory fresh full warmup, validated and reused exactly two complete replicas, charged their `4296.971965` wall / `16415.03` process-CPU seconds to the runaway ceilings, and completed the missing replica plus legacy and bookkeeping checks. The final active session plus reused charges total `10462.018172` wall / `41302.88` process-CPU seconds, below the `14400` / `57600` hard ceilings, with no fidelity reduction.

Therefore EXP-649 remains planned but this frozen formal path is blocked. The runner's formal preconditions can never accept this artifact because its status is `runtime-gate-miss` and `candidate_wall_gate_passed` is false. Do not generate a collision audit, open formal root `20261149`, score oracle breadth, train a selector, vote, tune paths or noise, alter Candidate 005, or make a promotion or completion claim from this result.
