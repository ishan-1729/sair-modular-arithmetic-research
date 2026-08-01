---
type: Experiment Preregistration
title: EXP-633 Hierarchical Learned Composition Feasibility
description: Matched small-width test of a shared learned binary composition tree against sequential decoded-state cellular recurrence.
status: completed
tags: [autoresearch, tier6, hierarchical-composition, tree-reduction, length-generalization]
timestamp: "2026-07-13T15:45:00+05:30"
---

# EXP-633 hierarchical learned composition feasibility

## Hypothesis and motivation

The strongest Tier-6 cellular models predict isolated width-128 transitions near 99.9% but lose exactness across a 128-step autonomous Horner trajectory. EXP-633 tests a materially different trained computation graph: multiplier-bit leaves are reduced by a balanced binary tree whose learned composition operator is shared across every node, level, and width. Its critical path grows logarithmically with multiplier width, so a viable learned operator could reduce exposure to serial state drift.

No exact modular operation appears in model inference. Python bigint arithmetic is restricted to offline node labels and evaluation truth. The learned tree may discover a useful composition rule, but the implementation must not encode affine-map composition, modular reduction, long division, Barrett/Montgomery reduction, or CRT recombination by hand.

## Model and matched control

Each tree leaf receives one multiplier bit plus spatial planes for multiplicand bits, modulus bits, and LSB/MSB boundaries. Adjacent left/right latent states are combined by one shared gated dilated-convolution operator. All nodes at a level are batched, and the same local bit head decodes every node. Training supervises each internal node with the exact residue of that multiplier segment times the multiplicand modulo the sampled prime; these are offline labels, not inference operations.

The control is the established decoded-state `SharedDilatedConvGRU`, trained on every exact Horner transition from the identical trajectories. Updates, trajectory batch size, width schedule, optimizer family, learning-rate schedule, prime identities, and total labeled states per trajectory are matched as closely as the two computation graphs permit. Parameter counts and actual wall/CPU work are reported rather than described as exactly compute-matched.

## Evidence boundary and frozen rung

Training uses only generated prime identities assigned to the training split. Evaluation identities are prime-disjoint and generated from seeds fixed in the config before execution. The first formal rung trains both arms for 2,400 updates across widths 8 and 16 and evaluates 300 exact products at widths 16 and 32. A 100-case width-32 randomized-weight control tests trained-parameter dependence. Deterministic repeated predictions and invalid-residue rates are recorded.

Advance the hierarchical family only if every gate passes:

- width-16 tree exact accuracy at least 20%;
- width-16 tree no more than two points below the matched sequential control;
- zero-shot width-32 tree exact accuracy at least 5% and at least five points above the sequential control;
- trained tree at least ten points above randomized tree weights on matched width-32 cases;
- invalid-residue rate at most 5% at both widths;
- exact repeated predictions and metrics;
- tree parameters at most 115% of control parameters;
- projected integrated candidate inference at most 260 seconds.

This is a viability gate, not authorization for independent, checked-public, official, or submission evaluation. Missing any gate closes this formulation at the declared capacity and training rung; it does not justify post-result threshold, width, or loss tuning on the evaluation identities.

## Dynamic resource-budget protocol

Before the formal run, execute only a deterministic training-throughput calibration on training identities: 30 representative updates per arm plus short fixed-shape inference timing. Record wall time, aggregate CPU time, thread count, peak RSS, expected setup/training/evaluation work, safety headroom, checkpoint cadence, and a hard runaway ceiling in this document and the resolved config. Calibration may change the resource budget, checkpoint interval, or resumability mechanics, but not the model, data split, 2,400-update rung, 300-case evaluations, or promotion gates.

The formal run must preserve valid per-arm caches and checkpoints. Wall-clock and aggregate CPU are separate units. Evaluation fidelity must not be reduced merely to fit an arbitrary short cap.

The first calibration invocation was stopped before split generation completed and before any model update or evaluation. It requested 640 unique width-8 primes from a configured band containing only 48, causing an unbounded upstream sampling loop. The corrected frozen split uses 40 training and 8 evaluation primes at width 8 while retaining 512/128 at widths 16 and 32, and the runner now checks prime-band capacity before sampling. This is a pre-evidence infrastructure correction; model, training rung, evaluation cases, and gates are unchanged.

The corrected calibration completed in 79.48 external wall seconds at 303% CPU utilization. Thirty matched updates took 33.943 wall / 127.270 aggregate CPU seconds for the control and 18.505 wall / 69.613 aggregate CPU seconds for the tree. Twenty width-32 inference cases took 3.319 wall seconds for the control and 0.870 wall seconds for the tree. Linear work projection gives 4,195.777 wall / 15,750.669 aggregate CPU seconds for formal training and 79.251 wall / 299.025 aggregate CPU seconds for evaluation.

The ignored calibration record is content-addressed as `85ed2d55e601261ae2de64b4ad1af4919356f7492d652beb4c0f071bde948569`; its external timing console is `daed2a9995ca99f892720befc255de75de161c76b7d5da771f4a4d6a9e1ae6b5`.

Including randomized/determinism controls and 180 seconds of setup/artifact allowance, expected total is 4,462 wall seconds and about 268 aggregate CPU minutes. The planning soft budget is 6,060 wall seconds; the hard runaway ceiling is 8,940 wall seconds, with a separate four-thread upper bound of 596 aggregate CPU minutes. Resumable checkpoints are written every 100 updates, and the hard ceiling is checked only after first preserving the current checkpoint. These budgets retain the complete 2,400-update and 300-case matched comparison.

## Limitations

Success at widths 16/32 would show only that hierarchical learned composition is viable enough for a higher-width program. Failure would reject this shared latent operator and training prescription, not all tree, transformer, or parallel-prefix neural arithmetic architectures.

## Result and decision

The full frozen comparison completed without reducing training or evaluation fidelity. The runner recorded 4,044.222 wall seconds and 15,147.538 aggregate CPU seconds; external `/usr/bin/time` recorded exactly 1:04:00 wall, 15,171.88 user-plus-system CPU seconds, and 395% CPU utilization. This is below the 4,462-second expected wall budget, 6,060-second soft budget, and 8,940-second hard runaway ceiling. Both wall clocks are retained because WSL scheduling made them differ by about 204 seconds.

The sequential control scored 260/300 (86.67%) exact at width 16 and 3/300 (1.0%) at width 32. The tree scored 0/300 at both widths. It produced valid residues on every evaluated case and was deterministic, but trained and randomized trees both scored 0/100 on the matched width-32 dependence diagnostic. The tree's best logged width-16 batch reached 60.83% exact internal nodes, yet no logged batch at any of 25 checkpoints contained one exact root. This localizes the failure to composing imperfect learned segment states across depth rather than inactivity or invalid decoding.

Decision: reject and close this shared latent binary-composition formulation at 94,025 parameters and 2,400 updates. Do not extend its budget or tune node/root loss weights on the observed evaluation identities. The result does not reject a one-shot relative-attention architecture, whose global communication and direct root supervision are materially different.
