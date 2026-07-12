---
type: Experiment Preregistration
title: EXP-401 PyG Graph Message Passing
description: Full-training matched comparison of cellular control and canonical PyG GCN, GATv2, and max-MessagePassing models.
status: preregistered
tags: [autoresearch, PyG, GCNConv, GATv2Conv, MessagePassing, graph-neural-network]
timestamp: "2026-07-12T18:10:00+05:30"
---

# EXP-401: PyG graph message passing

Run a serious 1,200-update matched comparison on fresh seed-20260718 mixed widths 8/12/16. The control is the existing shared cellular model. Library-backed arms use pinned PyTorch Geometric 2.8.0: degree-normalized `GCNConv`, four-head dynamic `GATv2Conv` with typed edge attributes, and a custom typed layer built on `MessagePassing(aggr="max")`. Each operator is recurrently shared across the same generic multiscale self/left/right graph relations. All arms receive identical prime splits, generated hard-mined labels, update counts, batch sizes, schedules, and evaluation seeds. State dicts are width-independent and already verified to load unchanged at width 32.

This experiment addresses the user's requirement not to treat minimal hand implementations as representative of mature GNN libraries. Advance a PyG arm only if it reaches at least 85% width-16 transition and 10% rollout, remains within 1.25x control parameters and 3x training wall time, and either gains at least five exact width-16 rollout points over the matched control or gains at least ten zero-shot width-32 transition points. Passing authorizes a higher-width graph curriculum, not candidate promotion. Failure provides a defensible library-backed graph result. No checked, consumed independent, or private answer is read.

## Dynamic resource budget

PyG 2.8.0 installs without optional compiled extensions and passes real CPU forward/backward probes with PyTorch 2.5.1. At batch 128, ten representative width-16 updates measure 3.189 wall / 11.592 aggregate CPU seconds for GCN, 5.484 / 20.038 for GATv2, and 5.523 / 20.231 for max-MPNN. Scaling to 1,200 updates projects 1,703.424 wall seconds for the three PyG arms. EXP-400's matched cellular arm projects 505.142 seconds at 1,200 updates, and its measured evaluation/setup residual contributes 473.486 seconds, for 2,682.052 projected total wall seconds. Allocate a 4,200-wall-second soft budget and 6,000-wall-second hard runaway ceiling. These are wall limits; four threads imply a conservative 400 aggregate CPU-minute hard upper bound. Cache each completed arm with config, runner, model, library, and seed hashes.
