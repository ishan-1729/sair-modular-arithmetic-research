---
type: Experiment
title: EXP693 C4 dilation-factored adapters
description: Frozen tournament contract and terminal package-oracle park for C4.
status: parked
tags: [autoresearch, tournament, convgru, svd, adapters]
timestamp: "2026-07-31T14:46:09+05:30"
---

# EXP693 C4 dilation-factored adapters

## Frozen mechanism

EXP693 implements a Candidate005-derived hidden-56 one-cycle ConvGRU specialist. An offline-only h56 teacher begins as exact raw-kernel copies across the eight physical outward-dilation slots and is frozen after 400 legal-data updates. The shared student kernels are the teacher-slot centroids. Native CPU `torch.linalg.svd` supplies rank-4 raw-Conv1d direction factors and dilation factors for dilations 8, 16, 32, 64, and 128. The primary applies their summed delta; the matched control computes and trains the identical teacher-distillation, reconstruction, adapter, optimizer, data, RNG, and update work, but multiplies only adapter access by exact zero.

The deployed route contains only input features, the hard learned residue, shared learned ConvGRU tensors, and learned low-rank tensors. Teacher construction, SVD, exact labels, reconstruction targets, and evaluation arithmetic remain offline. Candidate005 handles Tiers1-6 unchanged.

## Observed terminal result

The real rootless all-mode smoke completed teacher update, centroid/SVD construction, both matched student arms, reload, evaluation, adapter zeroing, dilation permutation, direction reversal, equal-norm random subspace, adapter randomization, full-weight randomization, deployed routing, admission plumbing, and serialization in 70.4 seconds. Both arms began from identical storage, executed equal teacher/adapter/reconstruction work, received adapter gradients, and reloaded exactly. The focused suite passed 25 tests after one task-owned correction to two test assertions.

The mandatory preflight verified all 87 Git-LFS paths byte-identical to the primary workspace and the clean nested challenge snapshot at `82510bba00a1126649bd76dd1a451f14d0b3eb60`. The exactly-once package oracle nevertheless stopped in project validation on six pre-existing EXP692 canonical-record/schema defects before pytest began. Those files are outside T099, and the frozen stop rule prohibits repairing them, rerunning the oracle, or opening root20261341.

Therefore roots20261341-49 remain unopened. No 400-update teacher, resource admission, full 2,400-update arm, accuracy denominator, confidence sequence, causal verdict, confirmation, independent seed, candidate, submission, promotion, or private-score evidence exists. C5-C7 remain independently required.

Canonical evidence is in `experiments/runs/exp-693-c4-dilation-factored-adapters/` and `evals/results/exp-693-c4-dilation-factored-adapters/`.
