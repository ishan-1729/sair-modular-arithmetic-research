---
type: experiment-preregistration
title: EXP-655 layer-loop compute-matched recurrence
description: Planned dense causal test of layer-local versus whole-stack recurrence ordering, followed conditionally by capacity recycling and MoE.
status: planned-blocked-awaiting-EXP648-and-separate-launch-judge
timestamp: 2026-07-21
---

# EXP-655 — Layer-loop compute-matched recurrence

## Status and boundary

This is a plan, not an authorization or result. EXP-648 remains the sole active scientific task. EXP-655 may not create a config, root, dataset, checkpoint, MLflow run, metric, or evaluation until EXP-648's current phase resolves and a separate launch Judge approves an authenticated package. Checked, prior-independent, and official/private identities are excluded from design and selection.

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

The two confidence bounds are conjunctive co-primary tests, so no alpha splitting is applied. For each endpoint, run exactly 100,000 deterministic hierarchical paired-bootstrap resamples. Each resample first draws three matched training/evaluation seed clusters with replacement, then draws 300 paired identities with replacement within each selected cluster, and reports the equal-cluster-weight mean layer-loop-minus-model-loop exact-accuracy delta. The one-sided 95% lower bound is the deterministic 5th percentile using NumPy's linear percentile convention. Freeze bootstrap RNG seeds `65595001` for width 32 and `65595002` for width 16. Inference is explicitly conditional on the three preregistered checkpoint pairs; it is not a claim about an unrestricted training-seed population.

The randomized-weight controls are frozen per arm and training seed: layer-loop seeds `65571001`, `65571002`, `65571003`; model-loop seeds `65572001`, `65572002`, `65572003`. Reinitialize every trainable parameter—including embeddings, recurrent/attention/FFN weights, normalization affine terms, and output readout—and every learned or running buffer with the architecture's original initialization routine/distribution; retain only fixed input encodings, masks, tensor shapes, and zero-state conventions. Hash the randomized state, evaluate it on the same 300 width-16 identities as its trained counterpart, and require at least a 20-point trained-minus-randomized exact drop for each of the six comparisons independently. No pooled result may rescue a failed checkpoint.

## Dynamic resource-budget resolution

Before launch, profile complete representative setup, training, checkpoint serialization, and full declared evaluation units for every materially different arm. Record per-unit wall time, aggregate descendant process CPU, peak RSS, and variability. Derive projected, soft, and hard budgets from measured throughput plus documented safety headroom; do not use an arbitrary short wall cap or lower evaluation fidelity to meet one. Preserve recoverable checkpoints, reuse only caches whose model/data/config/root/environment/evaluator hashes match, and charge reused work explicitly. CPU and wall limits are independent runaway protections. The competition's 300-second constraint is reserved for final submission inference qualification.

## Terminal criteria

The dense schedule component receives a verdict only after all primary arms and controls complete at declared fidelity or a preregistered integrity/resource stop fires with completed evidence preserved. Capacity recycling is separate and conditional. MoE and `ST-M044` post-training require their own gates. Candidate 005 and Goal completion cannot change from this experiment plan.
