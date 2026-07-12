---
type: Experiment Preregistration
title: EXP-400 Graph Message-Passing Smoke
description: Matched successive-halving rung for cellular, GCN, GAT, and typed recurrent max-MPNN architectures.
status: preregistered
tags: [autoresearch, graph-neural-network, GCN, GAT, MPNN, matched-control]
timestamp: "2026-07-12T17:15:00+05:30"
---

# EXP-400: graph message-passing smoke

Run four deterministic arms on identical mixed widths 8/12/16, prime splits, hard-mined offline labels, 200 updates, 128-example batches, optimizer schedules, and held-out evaluations. The control is the existing 99,824-parameter shared cellular model. Graph arms are: fixed-mean recurrent GCN (86,668 parameters), masked self/left/right recurrent GAT (94,951), and typed self/left/right recurrent max-MPNN (72,828). All use the legal six node features and shared multiscale positional relations defined in the graph research synthesis. The same trained state dict must load unchanged at zero-shot width 32. No graph encodes a carry, comparison, quotient decision, expected output, benchmark identity, or exact reduction circuit.

This is a declared successive-halving smoke rung, not a full-budget falsification of graph models. Advance a graph arm only if it reaches at least 75% width-16 transition and 2% rollout, stays within 1.25x control parameters and 2x training wall time, and either gains at least three width-16 rollout points over control or gains at least five zero-shot width-32 transition points. A winner advances to a full-budget matched confirmation; a non-winner closes only this rung and informs a graph-specific revision. Checked, consumed independent, and private answers remain unread.

## Dynamic resource budget

On the active four-thread CPU environment, 20 representative batch-256 forward/backward updates measured 19.533 wall / 71.536 aggregate CPU seconds for cellular, 22.852 / 83.867 for GCN, 27.383 / 100.170 for GAT, and 30.085 / 110.446 for MPNN. All four unchanged state dicts executed at widths 16 and 32. Scaling the summed profile to 200 updates projects 998.530 wall seconds at the profiled batch size; the actual batch is 128, making this conservative for training. Add 300 seconds for setup and the declared 5,000-transition plus 1,000-rollout width-16 and 5,000-transition plus 500-rollout width-32 evaluations, yielding 1,298.530 projected wall seconds. Allocate an 1,800-wall-second soft budget and 2,700-wall-second hard runaway ceiling. These are wall limits; at four threads the hard ceiling is a conservative 180 aggregate CPU-minute upper bound. Save and integrity-key each completed arm.
