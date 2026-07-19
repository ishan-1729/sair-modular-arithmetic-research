---
type: Research Note
title: Stochastic recursive reasoning and frozen trajectory breadth
description: Playwright-verified primary-source evidence and bounded SAIR transfer rationale for stochastic recursive trajectories.
status: completed
tags: [autoresearch, online-research, stochastic-reasoning, recursive-models, trajectory-breadth, tier7]
timestamp: "2026-07-20T03:07:13+05:30"
---

# Stochastic recursive reasoning and frozen trajectory breadth

## Research question

Do recent stochastic recursive-reasoning sources support a bounded, no-training test of whether the frozen Candidate 005 transition contains complementary width-256 trajectories? The question is about candidate breadth and CPU feasibility. It is not evidence that a selector can identify a correct answer, that paper results transfer to SAIR, or that Candidate 005 already has a learned path above 128 bits.

## Browser protocol and evidence scope

On 2026-07-20, the primary pages below were inspected read-only with repository-local Playwright in fresh browser contexts. Every requested page returned HTTP 200. Each context and the browser were closed after inspection, and no browser artifact was retained. These are bounded browser observations, not a reproduction of any paper or codebase.

This note keeps four evidence classes separate:

- **Source claims** are statements made by paper or project authors.
- **Browser observations** describe only what the inspected pages and repositories exposed on 2026-07-20.
- **SAIR observations** come from canonical completed repository evidence.
- **Project hypotheses** are preregistered transfers that remain untested until their own evidence is produced.

## Primary-source claims and implementation observations

### Probabilistic Tiny Recursive Model (PTRM)

The [PTRM paper](https://arxiv.org/abs/2605.19943) and its [HTML rendering](https://arxiv.org/html/2605.19943) describe Gaussian noise injected at each deep-recursion step, multiple stochastic rollouts, offline `pass@K` oracle coverage, and selection with a learned Q head. The authors report PPBench with `K=100`, depth 48, and sigma 0.2; Sudoku with `K=100`, depth 64, and sigma 0.3; Maze with `K=100`, depth 16, and sigma 1.0; and ARC with `K=25`, depth 16, and sigma 0.2. These are source-reported configurations and results, not SAIR measurements.

No implementation was linked from the inspected PTRM paper or HTML page. The source therefore supports the general mechanism—noise-conditioned recursive trajectory breadth—and the distinction between oracle breadth and learned selection. It does not supply a runnable reference implementation for Candidate 005.

### Generative Recursive Reasoning (GRAM)

The [GRAM paper](https://arxiv.org/abs/2605.19376) presents recursive reasoning as a learned generative or variational latent-trajectory process. That is an author claim about a probabilistic learned model, not evidence that additive inference noise improves an already trained deterministic SAIR cell.

The inspected [GRAM project page](https://ahn-ml.github.io/gram-website/) displayed `Code (coming soon)`. Thus the browser observation on 2026-07-20 is that no released implementation was available from that project page.

### Equilibrium Reasoners (EqR)

The [EqR paper](https://arxiv.org/abs/2605.21488) frames generalizable recursive reasoning as convergence toward task-conditioned solution attractors and studies scaling across iterative depth and trajectory breadth. This is a source claim; convergence has not been established for Candidate 005.

Public code was available at inspected commit [`aba94e9cde0f273ce644db5261cd6915ba6561f0`](https://github.com/locuslab/EqR/commit/aba94e9cde0f273ce644db5261cd6915ba6561f0). The repository documentation is task-specific and CUDA-oriented, and its Maze path documents FlashAttention. This implementation is useful provenance for EqR, but it is not a directly portable CPU reference for SAIR's cellular arithmetic path.

### Archived TRM backbone

The inspected [TinyRecursiveModels repository at commit `c01103738605ba39d1430519b1ee0c62f4c707f8`](https://github.com/SamsungSAILMontreal/TinyRecursiveModels/commit/c01103738605ba39d1430519b1ee0c62f4c707f8) was archived on 2026-04-01. Its documented environment uses GPU/CUDA, nightly PyTorch, and `adam-atan2`. This is the earlier TRM backbone, not PTRM implementation code. It cannot fill the missing PTRM implementation link or establish the exact stochastic operator needed here.

## Existing SAIR evidence and limitation

Candidate 005 remains the active incumbent at 59.5% checked-public and 59.3% independent overall accuracy. Its complete artifact has SHA-256 `2f6ac3199d0fd9e16afae165c2d4a16919e6c8bf4545363a6f12d077279d27ca`; the raw cellular source checkpoint used for the proposed diagnostic has SHA-256 `cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd`. The full candidate currently has no learned inference path above the 65–128-bit Tier-6 band. Running the raw cell at width 256 is therefore research instrumentation, not incumbent capability.

EXP-647 is related evidence, but not matched evidence for stochastic Candidate-005 inference. Its separately trained challenger reaches only 47.5% width-256 original/swapped oracle union. That result shows that two orientations can be complementary on a nearby lineage; it does not measure Gaussian paths from the frozen Candidate-005 checkpoint.

## Bounded transfer to EXP-649

EXP-649 transfers only the smallest testable mechanism: keep the raw checkpoint frozen and add absolute-scale Gaussian noise at the latent stem and before recurrent visits. Six fixed stochastic paths use sigma 0.2 and seeds `649001` through `649006`, split evenly between original and operand-swapped orientations. Exact sigma-zero original and swapped paths are controls. Hard decoded integer state remains the only state carried between multiplier bits, and no weights change.

The choice `K=6` is a compute-bounded planning decision, not a PTRM replication. PTRM reports `K=25` to `K=100`. Starting from EXP-647's 47.5% challenger union, six hypothetical, mutually independent additions that each succeed on 25% of remaining failures would give `1 - (1 - 0.475) * (1 - 0.25)^6 = 90.66%`; five would give 87.54%. The independence and 25% assumptions are deliberately hypothetical and are not matched Candidate-005 evidence. They justify only why six is the first preregistered breadth count that could cross the strict 90% gate under that toy calculation.

The experiment measures offline oracle union and exact candidate-shaped runtime before any selector work. Oracle labels may be read only in a separately authorized formal phase. No voting, margin/convergence selection, selector training, path tuning, seed tuning, or formal execution is authorized by this source review.

## SAIR interpretation boundary

The allowed research object is a frozen learned transition explored through generic stochastic perturbations. Gaussian sampling does not implement multiplication, division, final reduction, CRT, or output repair. Exact arithmetic is limited to offline generation, the same competition-sanctioned pairwise operand reductions `a % p` and `b % p` used by Candidate 005, and scoring in a later authorized formal phase. Trained weights must materially produce the final modular product residue.

A breadth pass would establish only that the frozen learned model sometimes contains a correct answer among eight fixed trajectories at acceptable measured cost. It would not establish target-free selection, candidate promotion, private-seed generalization, or Goal completion. A breadth miss closes this exact sigma, injection geometry, and six-path formulation without a post-result sweep on the same formal cases.
