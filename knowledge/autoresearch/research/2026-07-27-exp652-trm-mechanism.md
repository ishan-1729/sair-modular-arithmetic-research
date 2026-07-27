---
type: Research
title: EXP-652 bounded-faithful TRM mechanism map
description: Claim-to-code mapping from arXiv:2510.04871 to the SAIR answer-plus-latent experiment.
status: frozen-before-science
tags: [autoresearch, exp-652, trm, recursion, latent-state]
timestamp: "2026-07-27T23:40:00+05:30"
---

# EXP-652 bounded-faithful TRM mechanism map

## Local source identity

The source is the immutable local PDF `research/papers/arxiv-2510.04871.pdf`
(12 pages). Pages 5-7 and 11 were extracted with `pypdf==6.14.2` and rendered
with `pdfplumber==0.11.10`; the page images are retained only in the ignored
task scratch directory. No web source was used.

## Claim-to-code map

| PDF claim | Source location | EXP-652 binding |
|---|---|---|
| A full recursion performs `n` updates of latent `z`, then one update of answer `y`. | p. 5, Fig. 3 and Section 4.1 | `AnswerLatentTRM.full_recursion`: repeated `z = net(x + y + z)`, then `y = net(y + z)`. |
| The first `T-1` full recursions are forward-only; the final full recursion backpropagates through all `n+1` shared-network calls. | pp. 5-6, Section 4.1 | `AnswerLatentTRM.deep_recursion` runs early blocks under `torch.no_grad()`, detaches their states, and leaves every call in the final block on the autograd graph. |
| `y` is the current proposed solution and `z` is separate latent reasoning state. | p. 6, Section 4.2 | `y` is a fixed width-indexed answer canvas decoded by the bit head; `z` has no decoder and is carried separately. |
| One network can perform both updates because inclusion of `x` distinguishes latent from answer work. | p. 6, Section 4.3 | Both update types call the same `TinySharedNetwork`; only the latent call includes embedded legal input `x`. |
| A two-layer tiny network generalized better than a four-layer network. | p. 7, Section 4.4 and Table 1 | The shared recurrent network contains exactly two residual convolutional SwiGLU layers. |
| Deep supervision repeatedly improves detached `(y,z)` states. | p. 5, Fig. 3 and Section 4.1 | Each supervision step emits endpoint and Q losses, then detaches `(y,z)` before the next supervision step. |
| Q learns whether the current answer is correct and can halt without HRM's second continue-loss forward pass. | p. 7, Section 4.6 | A learned `q_head(y)` receives BCE correctness targets; evaluation compares learned halting with fixed depth. No continue loss or second forward is used. |
| EMA stabilizes small-data training. | p. 7, Section 4.7; p. 11 gives decay 0.999. | Training updates a shadow state after every optimizer update; raw and EMA checkpoints are separately serialized, reloaded, and evaluated. |
| The paper uses `T=3,n=6` and warns that full `n+1` backprop can OOM. | pp. 7-8, Table 3; p. 11 | Primary roots freeze `T=3,n=6`; calibration measures the actual final-block graph, RSS, and every possible confirmation route before admission. |

## SAIR adaptations and exclusions

- The source tasks use token grids. EXP-652 uses legal width-indexed bit planes
  for the two operands and modulus plus position markers. No puzzle identity,
  expected answer, intermediate exact state, or evaluation identity enters the
  model.
- The paper uses token cross-entropy and stable-max. This bounded CPU adaptation
  uses endpoint bit BCE because SAIR's answer canvas is binary; exact final
  answer accuracy remains the decision metric.
- To keep examples and optimizer updates exactly matched across arms, training
  uses a frozen number of detached supervision steps. Q is still trained as a
  correctness predictor and controls the measured inference-halting route; it
  does not change training-batch allocation.
- The width-independent two-layer network is a shared local convolutional
  SwiGLU rather than the paper's task-specific mixer/attention implementation.
  This is the declared SAIR operator adaptation, not a claim of architecture-
  byte parity.
- EXP-652 is not Tab-TRM. ST-M010.b is a distinct full-BPTT tabular FNN without
  this deep-supervision/ACT contract and is not reproduced or claimed.
- Exact modular multiplication appears only in offline labels and evaluation.
  Forward paths contain no modulo, division, reduction, CRT, lookup, or answer
  repair.

## Required causal evidence

The primary is compared with a parameter/work-disclosed decoded-cellular arm
and a no-latent arm. Viability measures raw and EMA reload, recursion curves,
fixed-depth versus Q halt, randomized weights, and answer/latent reset, swap,
and randomization. Only a passing viability root may open confirmation roots,
where no-deep-supervision, no-Q-loss, and no-EMA component controls are trained
under the frozen rule.
