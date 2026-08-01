---
type: Research Note
title: EXP680 Mixture-of-Recursions Routing Mechanism Map
description: Page-authenticated MoR hierarchy, capacity/load equations, Candidate005 mapping, and native-library decision.
status: parked-calibration-noncollapsed-load
tags: [autoresearch, exp680, mixture-of-recursions, M009]
timestamp: "2026-07-30T04:00:00+05:30"
experiment: EXP680
---

# EXP680 Mixture-of-Recursions mechanism map

The immutable local arXiv v3 PDF `2507.10524` has SHA-256 `040b9467356aee6f30f27901eef06cb65738f362ad9054ca4b1ba5f408700b69`; its manifest SHA-256 is `f2e2e38631d4cd5ea286d6e34c315b19c90d2c12e1725b3958a3e90229f09c21` and records CC BY 4.0. Pages 3-5 were rendered and visually inspected. Table 1 distinguishes Cycle and Sequence sharing. Figure 2 and Eq. 2.1 define expert-choice as a router at every recursion that chooses top-k tokens, with only prior survivors re-evaluated. Eq. 2.2 defines token-choice as one initial top-1 depth assignment. Table 2 says expert-choice has static compute but causal leakage requiring an auxiliary router/loss, while token-choice avoids leakage but needs balancing. For `N_r=3`, the source equal-compute allocation processes `3/3`, `2/3`, and `1/3` of tokens.

Candidate005 has no attention or KV cache, and each dilated convolution couples spatial positions. EXP680 maps a source token to one complete arithmetic instance, recursion blocks to contiguous segments of the exact Candidate005 dilation schedule, and source capacities to exact per-batch counts. Training uses hierarchical `torch.topk`; deterministic EMA thresholds supply sample-local inference without future tokens, labels, or other samples. The forward route is not permitted to receive labels, correctness, validity, exact answers, oracle depth, or benchmark identity. This is a bounded-faithful dynamic-depth transfer, not a claim about Transformer KV caching or paper-scale language modeling.

Native PyTorch 2.5.1+cpu `topk`, `index_select`, and `index_copy` are selected. They are maintained BSD-3-Clause primitives, preserve autograd and deterministic CPU behavior on the frozen environment, and exactly express this bounded sample topology. General MoE packages primarily target expert dispatch, distributed Transformers, or GPU kernels and would add machinery without improving fidelity; no external dependency is adopted. M009.a sharing layouts, Loopies M043.a/M043.b/M043.c, GPU throughput, and KV-cache claims are excluded.

The implementation passed ten focused tests, the 673-test package oracle, and complete-route root `20261270`. The expert route saved 10.714% of W128 calls and was faster than fixed-full in the matched route probe, but its learned inference thresholds used only depths 8 and 14, leaving depth 11 empty. The preregistered noncollapse conjunct therefore parked the exact formulation before scientific roots. This observation is specific to the frozen instance-level mapping and threshold construction; it does not reject the paper or M009.b generally.
