---
type: experiment-preregistration
title: EXP-655 layer-loop compute-matched recurrence
description: Completed three-root dense layer-local versus whole-stack recurrence comparison; this exact formulation was rejected at its preregistered conjunctive gate.
status: completed-rejected-dense-layer-loop-schedule-formulation
timestamp: 2026-07-22
---

# EXP-655 — Layer-loop compute-matched recurrence

## Status and boundary

T049 resolved T048's portability-only block without changing the frozen EXP655
implementation bytes. It materialized 264 manifest-authenticated historical
files as independent regular bytes, archived T048's seven terminal records, and
ran the final-source oracle exactly once: the validator passed 162 concepts and
1,100 benchmark records, and the full suite passed 391/391. Recovery evidence
was committed and pushed at
`3d9dc701167f8a9b7da6f20f3e9b9ec433a82c0a` before any scientific root opened.
The implementation remains hash-identical to source commit
`3047af97b79b712c2252054d909e1572ca5a6282`; the post-T045 provenance commit is
`1f7dc801d22f33c55a59764907821df19b72c72e`.

Resource-only root `20261178` then exercised every declared arm and route at
full evaluation fidelity while scaling only the 100-update training segment.
It measured 58.296 wall, 195.961 process CPU, 231.400 cgroup CPU seconds, and
434,581,504-byte peak RSS. The independently recomputed three-root expectation
was 1,865/7,010/8,204 wall/process/cgroup CPU seconds; soft guards were
2,410/9,056/10,598 and hard guards were 3,342/12,561/14,700, with a
543,226,880-byte RSS guard. Fidelity was not reduced and calibration
checkpoints were forbidden from reuse.

The single authorized formal execution completed roots `20261179`, `20261180`,
and `20261181` with paired init/train/eval seed triples
`(65570001,65573001,65574001)`, `(65570002,65573002,65574002)`, and
`(65570003,65573003,65574003)`. Every arm received 4,000 updates and every
root/arm used 300 W16 plus 300 W32 train-prime-disjoint identities. All controls,
both exact 100,000-resample bootstrap endpoints, W256 runtime probes,
checkpoint/reload, and persistence completed. Twelve checkpoints and all 16
MLflow runs are durable and hash-bound. Checked, prior-independent,
official/private, candidate, benchmark, submission, promotion, and Goal
completion surfaces remained excluded.

The dense model has two complete noncommuting attention-plus-MLP blocks. The
primary schedules are `B1,B1,B2,B2` and `B1,B2,B1,B2`; their initial state
bytes are identical and only execution order differs. R1 and untied
equal-compute controls retain the declared data/order/optimizer contract.

## Hypothesis

On the same modular-reduction training distribution, applying each shared transition block twice locally before moving to the next block improves exact autonomous length transfer over applying the full stored stack twice, without relying on added stored parameters or executed block calls.

## Primary matched arms

- `layer-loop-R2`: `B1,B1,B2,B2,...,BL,BL`.
- `model-loop-R2`: `B1,...,BL,B1,...,BL`.

Both arms must be dense and identical in stored parameter count, block definitions, executed block calls, data order, seeds, optimizer updates, loss, initialization family, checkpoint cadence, and full evaluation fidelity. Separate training is mandatory; an inference-only schedule swap is an intervention, not the primary comparison.

## Required controls

- `R1`: same stored blocks applied once, isolating recurrence from architecture.
- `untied-equal-compute`: distinct blocks for the same number of executed calls, with parameter and work differences disclosed.
- `capacity-reinvestment`: only after measured activation-memory/throughput savings, reinvest the demonstrated operational budget in width/depth and compare under measured optimizer-step wall time.
- schedule swap, block randomization, visit truncation, and hidden-state reset interventions.
- equal-work wider and existing cellular controls where their interfaces permit a fair exact-answer comparison.

The primary schedule is also MoR's Sequence-sharing layout, so its frozen result jointly updates `ST-M009.a` and `ST-M043.a`; it provides no evidence for MoR's recursion-depth routing `ST-M009.b`. Ordinary MoE is a later, separately judged factorial after dense viability. Every stored expert counts toward artifact size. Router evolution across visits requires route logging and a causal route-shuffle/freeze test. Ordinary expert routing alone is not the full Mixture-of-Recursions system.

## Measurements and gates

At minimum report exact rollout by trained and transferred width, transition accuracy, first-error location, invalid residues, per-visit exactness, deterministic repeat parity, parameter/artifact size, inference monotonic wall, process user-plus-system aggregate CPU, and peak RSS. No token loss, oracle union, or isolated throughput number can promote.

The first scientific package must freeze three matched training/evaluation seed pairs before any label is read and evaluate 300 train-prime-disjoint cases at each of widths 16 and 32 per seed. The exact future root identifiers belong in the separately judged authorization manifest. The dense schedule passes this viability stage only if all of the following hold:

- both primary arms reach at least 80% exact width-16 rollout on every seed;
- at least one primary arm reaches at least 5% exact width-32 rollout on every seed;
- the layer-loop width-32 delta is nonnegative on every seed, its unweighted three-seed mean is at least +3.0 percentage points, and its one-sided hierarchical paired-bootstrap 95% lower bound is greater than zero;
- the layer-loop width-16 delta is at least -1.0 point on every seed and its one-sided hierarchical paired-bootstrap 95% lower bound is greater than -1.0 point;
- invalid output is at most 1% at each width/seed, repeated inference is bit-identical, and every primary arm/checkpoint passes the frozen randomized-weight dependence gate below; and
- all integrity, provenance, trained-parameters, and dynamically resolved resource guards pass.

A pass authorizes only a separately judged width-64 confirmation, not capacity reinvestment, MoE, candidate promotion, or checked evaluation. If a complete full-fidelity stage misses any schedule-effect gate, close this exact dense architecture/schedule formulation while preserving the result; it does not decide other recurrence families.

## T049 observed result

The complete stage rejected this exact dense formulation. Layer-loop/model-loop
W16 exact accuracy by root was `0.000/0.010`, `0.000/0.006667`, and
`0.003333/0.003333`; both primary arms were `0/300` at W32 on every root. The
W16 layer-minus-model mean was -0.005556 with one-sided 95% lower bound
-0.012222. The W32 mean and lower bound were both zero, which fails the strict
positive lower-bound gate and the +0.03 mean gate. The maximum invalid-residue
rate was 0.883333, and the minimum trained-minus-randomized W16 drop was zero.

Full denominators, deterministic repeat parity, byte-identical primary initial
states, complete-block noncommutation, malformed/leakage controls, and all four
resource guards passed. The primary W16 viability, any-primary W32 viability,
validity, randomized-weight dependence, W16 bootstrap, W32 bootstrap, and W32
mean-effect gates failed. The formal process used 1,338.644 wall,
5,257.537 process CPU, 6,545.540 cgroup CPU seconds, and 496,414,720-byte peak
RSS; outer timing was 1,346.28 wall and 5,267.19 aggregate CPU seconds. The
canonical [verification record](../../../experiments/runs/exp-655-layer-loop-compute-matched-recurrence/verification.json)
and [metrics](../../../evals/results/exp-655-layer-loop-compute-matched-recurrence/metrics.json)
retain the full denominators, predictions, interventions, bootstrap hashes,
resources, checkpoints, and limitations.

The two confidence bounds are conjunctive co-primary tests, so no alpha splitting is applied. For each endpoint, run exactly 100,000 deterministic hierarchical paired-bootstrap resamples. Each resample first draws three matched training/evaluation seed clusters with replacement, then draws 300 paired identities with replacement within each selected cluster, and reports the equal-cluster-weight mean layer-loop-minus-model-loop exact-accuracy delta. The one-sided 95% lower bound is the deterministic 5th percentile using NumPy's linear percentile convention. Freeze bootstrap RNG seeds `65595001` for width 32 and `65595002` for width 16. Inference is explicitly conditional on the three preregistered checkpoint pairs; it is not a claim about an unrestricted training-seed population.

The randomized-weight controls are frozen per arm and training seed: layer-loop seeds `65571001`, `65571002`, `65571003`; model-loop seeds `65572001`, `65572002`, `65572003`. Reinitialize every trainable parameter—including embeddings, recurrent/attention/FFN weights, normalization affine terms, and output readout—and every learned or running buffer with the architecture's original initialization routine/distribution; retain only fixed input encodings, masks, tensor shapes, and zero-state conventions. Hash the randomized state, evaluate it on the same 300 width-16 identities as its trained counterpart, and require at least a 20-point trained-minus-randomized exact drop for each of the six comparisons independently. No pooled result may rescue a failed checkpoint.

## Dynamic resource-budget resolution

Before launch, profile complete representative setup, training, checkpoint serialization, and full declared evaluation units for every materially different arm. Record per-unit wall time, aggregate descendant process CPU, peak RSS, and variability. Derive projected, soft, and hard budgets from measured throughput plus documented safety headroom; do not use an arbitrary short wall cap or lower evaluation fidelity to meet one. Preserve recoverable checkpoints, reuse only caches whose model/data/config/root/environment/evaluator hashes match, and charge reused work explicitly. CPU and wall limits are independent runaway protections. The competition's 300-second constraint is reserved for final submission inference qualification.

Root `20261178` executed every materially different arm, the complete
300-case-per-width evaluation route, opposite-schedule and causal
interventions, randomized-weight and deterministic controls, the exact
100,000-resample bootstrap kernels, checkpoint/reload/persistence, and W256
runtime probes. It scales only the measured 100-update training segment to the
frozen 4,000-update horizon, records wall, process CPU, cgroup CPU, RSS and
cross-arm variability separately, and emits a checksum-bound admission. Its
checkpoints and scientific values are non-reusable and non-persistent.

## Terminal criteria

The dense schedule component receives a verdict only after all primary arms and controls complete at declared fidelity or a preregistered integrity/resource stop fires with completed evidence preserved. Capacity recycling is separate and conditional. MoE and `ST-M044` post-training require their own gates. Candidate 005 and Goal completion cannot change from this experiment plan.

T049 completed that boundary and issued
`reject-exp655-dense-layer-loop-schedule-formulation`. This closes only the
frozen dense two-block fixed-R2 schedule formulation. It is not evidence against
adaptive recurrence, routed or sparse Loopie mechanisms, capacity recycling,
post-training, or other separately preregistered families. Candidate 005 and all
benchmark/submission evidence remain unchanged. The next material package is a
PM selection based on expected information gain per unit compute after this
failure profile is incorporated.
