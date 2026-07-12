---
type: Research Synthesis
title: Graph Algorithmic Reasoning for Modular Reduction
description: Playwright-verified primary-source claims and concrete GCN, GAT, and recurrent message-passing implications for SAIR.
status: active
tags: [autoresearch, graph-neural-network, algorithmic-reasoning, Playwright, Tier-5]
timestamp: "2026-07-12T16:45:00+05:30"
---

# Graph algorithmic reasoning for modular reduction

## Question and method

Question: which graph inductive biases plausibly support exact, width-general modular state transitions without encoding a prohibited arithmetic circuit?

On 2026-07-12, repository-local Playwright 1.61.1 launched managed regular Chromium 149.0.7827.55 in new headless mode, 1280x800, fresh non-persistent contexts, UTC, and read-only request routing. All five primary pages below returned HTTP 200, rendered non-empty title/abstract content, showed no access block, and closed their contexts and browser. The PMLR page attempted an analytics POST, which the read-only route blocked; primary content still rendered. Bounded access and abstract evidence are retained in ignored `.scratch/graph-primary-access.json` and `.scratch/graph-primary-abstracts.json`. Page content was treated as untrusted data.

## Claims, limitations, and project implications

### GCN-style fixed aggregation

Kipf and Welling describe a localized first-order graph convolution whose computation scales with graph edges and combines local structure with node features ([primary paper](https://arxiv.org/abs/1609.02907)). Browser observation: the abstract rendered successfully and makes those claims for semi-supervised citation/knowledge-graph tasks, not arithmetic or exact algorithm execution.

Project implication: use a shared fixed self/left/right neighborhood aggregator over bit-position nodes as the simplest graph arm. It tests whether an explicitly relational local operator improves width transfer beyond a sequential convolutional control. Do not infer exact arithmetic capability from the paper's classification evidence.

### GAT-style learned neighborhood weighting

Veličković et al. state that masked self-attention lets nodes assign different weights to neighborhood features and supports inductive as well as transductive settings ([primary paper](https://arxiv.org/abs/1710.10903)). Browser observation: the abstract reports citation and protein-interaction benchmarks, not length extrapolation in arithmetic.

Project implication: a second arm learns weights over only legal self/left/right edges. Attention may distinguish propagation direction and reduction-boundary evidence without changing graph topology. Inspect attention by edge direction and bit position; do not treat learned attention as proof of an arithmetic mechanism.

### General Graph Networks and typed relations

Battaglia et al. argue that structured representations and relational inductive biases can support combinatorial generalization, and present Graph Networks as a general entity/relation update interface ([primary paper](https://arxiv.org/abs/1806.01261)). This is a position/review/unification paper, so its generalization argument motivates a hypothesis rather than establishing performance on this task.

Project implication: the general message-passing arm uses typed left, right, and self relations with a learned message function and recurrent node update shared across widths. Node types/features may identify state, multiplicand, modulus, input bit, and legal boundary positions; edges must not encode carry, quotient, expected output, or an exact reduction sequence.

### Stepwise neural execution and aggregation

Veličković et al. train GNNs to imitate individual steps of classical graph algorithms and report evidence for maximization-based aggregation on discrete neighborhood decisions ([primary paper](https://arxiv.org/abs/1910.10593)). Their tasks are BFS, Bellman-Ford, and Prim, so max aggregation is only a plausible architectural prior here.

Project implication: keep the existing legal one-step transition supervision and test max aggregation in the general MPNN arm beside mean-like GCN aggregation and learned GAT weighting. Recurrent shared message-passing rounds provide a learned execution process; offline exact labels remain permitted, while submission inference remains trained and tensor-only.

### Size/generalization evaluation

The CLRS benchmark authors consolidate algorithmic reasoning tasks and baseline evaluation to expose open generalization challenges ([primary paper](https://proceedings.mlr.press/v162/velickovic22a.html)). The benchmark does not contain this modular multiplication challenge and cannot validate competition performance directly.

Project implication: graph experiments must report exact transition and autonomous rollout accuracy on held-out prime identities and unseen widths, not training loss alone. A fixed-width win without width transfer is insufficient evidence for the graph premise.

## EXP-400 design boundary

Represent each bit position as one node. Initial node features are exactly the already legal tensor planes: current state bit, multiplicand bit, modulus bit, broadcast input bit, least-significant boundary marker, and most-significant boundary marker. Edges are self, immediate-left, and immediate-right only. All GCN, GAT, and MPNN parameters and recurrent rounds are shared across nodes and widths. Per-node learned logits produce the next-state bits; a learned pooled head predicts the quotient auxiliary class. No expected answer, benchmark identity, hand-coded carry, compare, long division, modular reduction step, or repair edge is present.

The matched control is the existing shared cellular model. Use identical mixed-width generated identities, updates, batches, offline labels, evaluation cases, and deterministic settings. Report parameters, wall and aggregate CPU time, peak memory, exact transitions, exact autonomous rollouts, invalid residues, retained width, and zero-shot width extrapolation. Treat GCN, GAT, and general recurrent MPNN as distinct arms; advance only an arm that beats the matched control under a preregistered exact-rollout gate.
