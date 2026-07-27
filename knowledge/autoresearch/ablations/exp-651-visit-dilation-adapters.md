---
type: experiment-preregistration
title: EXP-651 visit/dilation-specific recurrent adapters
description: Preregistered bounded-faithful RRT analogue on the frozen Candidate005 recurrent cell.
status: parked-before-calibration-full-oracle-red
timestamp: 2026-07-22
---

# EXP-651 — Visit/dilation-specific recurrent adapters

## Scope and library decision

This is a bounded-faithful analogue of RRT physical-layer-by-visit LoRA, not a
full RRT reproduction. It retains the paper's visit-specific low-rank
specialization and matched shared/global controls, but intentionally omits the
paper's SVD residual teacher and base-plus-adapter uptraining contract. PEFT was
assessed and not selected: its standard adapters wrap supported modules, while
Candidate005 exposes its recurrent Conv1d kernels as raw `nn.Parameter`
tensors and selects them inside a custom dilation loop. A small native PyTorch
bank gives exact placement, zero-output initialization, deterministic CPU
parity, and explicit cycle/dilation indexing without replacing the proven cell.

## Frozen comparison

The source is the checksum-authenticated EXP609 checkpoint
`cbc4a1b...206d4bd`. The challenger has 16 semantic slots addressed by
`cycle * 8 + log2(dilation)`, each rank 8. The parameter-matched global control
uses one rank-128 delta; both therefore store and train exactly 110,592 adapter
parameters. The shared-core control continues the ordinary recurrent cell.
All arms use identical root-specific prime splits, examples, order, 200
optimizer updates, losses, and W64/W128 training schedule. Calibration root
20261182 runs ten updates per arm and every full evaluation/intervention route;
its checkpoints cannot seed formal work. Formal roots 20261183–20261185 use the
seed triples frozen in the config.

Each root evaluates 1,000 transitions and 200 complete autonomous rollouts at
W64, W128, and W256 on train-prime-disjoint identities. Exact rollout is the
primary metric. Controls include exact source-logit identity, adapter
activation, zeroing, cycle/dilation-slot permutation, randomized trained
adapter weights, deterministic checkpoint reload, malformed input, leakage
inspection, and fixed parameter accounting. Exact arithmetic is confined to
offline labels/evaluation. Checked, prior-independent, official/private,
candidate, benchmark, and submission surfaces remain closed.

## Decision rule

Against the matched global adapter, every root must have nonnegative W128 and
W256 deltas; mean deltas must be at least +2 and +1 points respectively, with
strictly positive one-sided 95% hierarchical paired-bootstrap lower bounds.
Every W64 delta and its lower bound must be at least -0.5 point. Every endpoint
must have invalid rate at most 1%, adapter RMS must exceed `1e-4`, zeroing and
permutation must change predictions, trained weights must beat randomized
weights by at least 10 W128 points, and deterministic/provenance/resource gates
must pass. The three bootstrap endpoints use 100,000 deterministic resamples,
sampling roots then paired identities within roots.

Calibration separately measures monotonic wall time, process user+system CPU,
cgroup CPU, and peak RSS. Formal expected/soft/hard budgets are derived from
the observed ten-update route, scaled only for the frozen training segment and
with cross-arm variability plus 30%/80% headroom. No scientific run opens if
identity, compliance, the final-source oracle, or calibration admission fails.
A pass authorizes only a separate candidate-shaped confirmation package; any
failed conjunct rejects this exact adapter formulation without changing
Candidate005.

## Result

T050 parked this package before calibration or science at the required
final-source oracle. The bounded-faithful implementation passed all seven
focused EXP651 tests, including exact W64/W128/W256 source-logit identity,
110,592-versus-110,592 adapter parameter matching, activation, interventions,
reload, deterministic behavior, malformed/leakage checks, dynamic resource
accounting, and hash-valid partial-run recovery. Project validation also
completed successfully before pytest opened.

The full-project oracle remained red after the two permitted attempts. Its
updated pytest cache contains 398 collected node IDs and 25 EXP658-only
failures. A focused failure probe established the concrete cause: this isolated
worktree lacks the ignored checksum-bound EXP658 source checkpoint at
`.scratch/exp-658-joint-quotient-carry-transducer/source_checkpoint.safetensors`.
Repair requires materializing historical evidence and rerunning the full suite,
but the Worker verification-attempt limit is exhausted. T050 therefore stops
without opening resource root 20261182 or formal roots 20261183–20261185.

This is a verification-feasibility park, not scientific evidence against
visit/dilation adapters. No calibration, admission, MLflow run, training,
checkpoint, transition/rollout metric, intervention outcome, bootstrap,
checked/prior-independent/official/private answer, submission mutation,
candidate promotion, or Goal-completion evidence was created.
