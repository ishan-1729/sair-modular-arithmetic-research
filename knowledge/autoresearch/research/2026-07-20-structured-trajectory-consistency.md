---
type: Research Note
title: Structured natural-trajectory endpoint consistency
description: Playwright-verified DRM and LoopFormer claims, local negative controls, and the bounded transfer used by EXP-648.
status: completed
tags: [autoresearch, online-research, trajectory-consistency, recursive-models, tier7, exp-648]
timestamp: "2026-07-20T10:20:00+05:30"
---

# Structured natural-trajectory endpoint consistency

## Research question

Do recent recursive-model sources justify testing a fixed endpoint-consistency loss on the natural failure geometry of the frozen cellular model? The narrow project question is whether supervision across an eight-transition autonomous window can reduce accumulated width-256 drift without injecting ambiguous synthetic state corruption or changing inference.

This note separates author claims, browser observations, local SAIR observations, and the project hypothesis. Neither a paper result nor an accessible repository is evidence that the mechanism improves SAIR exact accuracy.

## Browser protocol and local paper provenance

On 2026-07-20, the relevant primary pages were inspected read-only through the repository-local Playwright workflow. The session used Playwright `1.61.1`, its managed regular Chromium in new-headless mode (`149.0.7827.55`), a `1280x800` viewport, and fresh isolated contexts. Requests were read-only, each inspected page returned HTTP 200, and all contexts and the browser were closed after inspection.

The inspected primary surfaces were:

- the [Denoising Recursion Models paper](https://arxiv.org/abs/2604.18839) and [official implementation repository](https://github.com/wwwwwwz/DenoisingRecursionModels); and
- the [LoopFormer paper](https://arxiv.org/abs/2602.11451), [project page](https://loopformer.github.io/), and [official implementation repository](https://github.com/armenjeddi/loopformer).

Local immutable paper copies provide an additional provenance anchor:

| Paper | Local path | SHA-256 |
|---|---|---|
| Denoising Recursion Models | `research/papers/arxiv-2604.18839.pdf` | `6939dc9065d2eca96be7a7e1bf8a7a3fc4ad6d4ea8dd98c036a25886929c00d8` |
| LoopFormer | `research/papers/arxiv-2602.11451.pdf` | `92acdec9796ee1f5fdac8442c099322286eb2b3c078b64f12a505c7e77962c32` |

The browser observations establish source availability and provenance only. No paper result or codebase was reproduced in this review.

## Primary-source claims

### Denoising Recursion Models

The DRM authors present recursive reasoning as a multi-step reverse-corruption process and train over a trajectory curriculum that moves corrupted target solutions toward clean solutions. The official repository corresponds to that paper mechanism. These are author claims about the DRM tasks, architecture, and training procedure; they do not establish that synthetic corruption is well defined for an intermediate modular residue or that DRM transfers to the Candidate-005 cell.

### LoopFormer

The LoopFormer authors describe variable-length reasoning trajectories and a time- or step-conditioned shortcut-consistency objective intended to keep predictions useful at shorter compute budgets while allowing improvement with more recurrent computation. The project page and official repository expose the corresponding research project. These are source claims, not SAIR measurements, and LoopFormer's variable-compute architecture is not imported into EXP-648.

## Existing SAIR evidence

The frozen raw cellular source is the 99,824-parameter hidden-92, two-cycle checkpoint at SHA-256 `cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd`.

EXP-643 observes 99.70–99.75% exact width-256 transition accuracy but only 22% and 32% exact full rollout across two untouched seeds. This supports accumulated autonomous drift as an empirical problem, but EXP-643 does not show that endpoint consistency fixes it.

EXP-610 is the required negative control. Injecting arbitrary one- or two-bit residue corruption reduces width-128 exact rollout from 87.5% to 75.0%. The corrupted residues often do not identify an unambiguous clean successor. EXP-648 therefore must not inject, flip, or fabricate state bits and must not be described as a retry of generic corruption training.

Candidate 005 remains the incumbent at 59.5% checked-public and 59.3% independent overall accuracy. It has no learned inference path above 128 bits. EXP-648 is a research screen on the raw cellular lineage, not evidence that Candidate 005 solves Tier 7.

## Bounded transfer to EXP-648

EXP-648 borrows only one abstract mechanism from the sources: supervise the exact endpoint of a short recursive trajectory in addition to the ordinary clean transition objective. Its training windows come from the frozen source model's natural autonomous trajectories. A failure window begins at the exact prefix immediately before the model's first wrong transition; a matched successful window begins at an exact prefix with the same position and modulus stratum. Both arms execute the same hard-forward, sigmoid-backward eight-step autonomous unroll and compute the same endpoint term. Only its fixed coefficient changes from `0.0` to `0.1`.

The experiment does **not** import DRM synthetic corruption, reverse-noise schedules, or architecture. It does **not** import LoopFormer time conditioning, variable trajectory length, adaptive compute, or architecture. It does not add a denoising inference procedure. The deployed model structure, parameter count, and inference semantics remain unchanged.

This transfer is deliberately diagnosis matched: the intervention asks whether a natural pre-error state can receive useful eight-step endpoint credit while matched already-correct windows constrain newly introduced drift. Recovered cases and control-exact/challenger-wrong cases must be reported separately. A transition-loss improvement alone cannot pass.

## Compliance and interpretation boundary

Exact arithmetic is permitted only for offline generated training identities, trajectory labels, the exact endpoint teacher, and evaluation. No expected answer, benchmark identity, exact reduction routine, correction rule, or lookup enters inference. The trained cellular weights must continue to materially produce the output.

A passing screen would be intermediate causal evidence for this one coefficient and K=8 formulation. It would authorize only two separately approved fresh matched confirmations. Even a passing confirmation would require a new candidate/Tier-7 preregistration and all independent, runtime, determinism, trained-weight, container, and compliance gates. It would not alter Candidate 005, establish official performance, or satisfy the 85% completion threshold.
