---
type: Experiment Preregistration
title: EXP-510 Multi-architecture Grokking Feasibility
description: Three-seed finite-data delayed-generalization comparison of PyG max-MPNN and shared-context TCN.
status: completed
tags: [autoresearch, grokking, PyG, MPNN, TCN, multi-seed]
timestamp: "2026-07-12T21:15:00+05:30"
---

# EXP-510: multi-architecture grokking feasibility

Complete the program's remaining mandatory grokking coverage with two non-cellular families: a proper PyG 2.8.0 typed max-MessagePassing recurrent graph model (72,484 parameters) and the materially different fixed-width shared-context TCN (75,469 parameters). For seeds 20260722–20260724, both receive identical frozen 2,048-example training and 4,096-example prime-disjoint heldout datasets, the same minibatch index sequence, AdamW decay 1e-2, 8,000 updates, dense exact curves, rollout curves, norm/margin diagnostics, and 500-step checkpoints. This is an architecture feasibility comparison, not an accuracy contest between unequal computations.

Use the unchanged EXP-500 operational signature. An architecture passes only if at least two of three seeds first memorize with the required generalization gap and then, at least 1,000 steps later, retain 99% train exact, reach 95% heldout with 15 points of gain, and reach 80% rollout with 20 points of gain. Partial late improvement remains semi-grokking. Passing authorizes an architecture-specific mechanistic follow-up; it does not authorize candidate promotion or checked evaluation.

## Dynamic resource budget

At width 9 and batch 128, 100 representative updates measure 3.258 wall / 13.024 aggregate CPU seconds for TCN and 11.687 / 46.719 for PyG max-MPNN. Six 8,000-step arms project to 3,586.721 training wall seconds. Reusing EXP-500's measured evaluation workload projects 555.796 seconds, with 60 seconds for checkpoint/setup, totaling 4,202.518 wall seconds. Allocate a 6,000-wall-second soft budget and 9,000-wall-second hard runaway ceiling. These are wall limits; four threads imply a conservative 600 aggregate CPU-minute hard upper bound. Cache only complete architecture/seed arms keyed by config, code, library, data, and seed hashes.

## Outcome

Rejected: zero TCN seeds and zero PyG max-MPNN seeds satisfy the complete signature. Every TCN seed memorizes between steps 1,400 and 1,800 with a 26–29-point train/heldout gap, but later heldout gains only about five to seven points, peaks at 75.6–80.2%, and rollout ends at 18.8–29.0%. This is a clear memorization plateau, not grokking.

PyG max-MPNN reaches 99% train only at steps 4,000–4,600, when heldout is already 81.4–83.0%, above the preregistered memorization ceiling. It therefore generalizes concurrently rather than grokking; heldout peaks at 83.6–85.9% and final rollout at 27.4–32.4%. No semi-grokking or ungrokking flag fires. External timing is 4,476.0 wall seconds and 17,709.7 user-plus-system CPU seconds; runner timing is 4,463.8 wall and 17,695.7 aggregate CPU seconds. The run completes inside budget.
