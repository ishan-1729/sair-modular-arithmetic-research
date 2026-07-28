---
type: Research Note
title: EXP-671 Fixed-Point Reasoner mechanism map
description: Authenticated ST-M027 claim-to-code map, frozen bounded SAIR adaptation, controls, roots, gates, and exclusions.
status: preregistered
tags: [autoresearch, exp-671, fixed-point, damping, adaptive-halting]
timestamp: "2026-07-29T06:30:00+05:30"
---

# EXP-671 Fixed-Point Reasoner mechanism map

## Authenticated source

The immutable local PDF `research/papers/arxiv-2606.18206.pdf` has SHA-256
`169dc66aca853f5e8d922a170eac8421e78e1829d0d39025976a01915ff11c3d`. Physical PDF
pages 5-14, 25, and 27 were extracted with `pypdf 6.14.2`, rendered with
`pypdfium2` because Poppler is absent in WSL, and visually inspected in three contact sheets. Equations,
Algorithms 1-2, tables, plots, captions, and page transitions were legible without clipping or missing pages.

| Source mechanism | Pages | Frozen EXP-671 analogue | SAIR adaptation / exclusion |
|---|---:|---|---|
| PreNorm improves signal propagation; residual scaling restores boundedness. | 5-6, Eq. 2, Thm. 1-2; 12-14; 25 | Shared Conv1d refinement applies GroupNorm before its learned sublayer and a learned per-channel residual scale initialized to exactly zero. | Conv1d replaces attention/FFN. Exact-zero scale, rather than the paper's nonzero alpha initialization, preserves the frozen EXP609 proposal byte-exactly before training. |
| Reinject the immutable input between every shared iteration. | 5, Eq. 3 | Proposal state and six legal input-feature channels are concatenated at every refinement call. | Concatenation is the project-native counterpart of affine input mixing; no target or answer feature is injected. |
| Damping preserves fixed points and can suppress oscillation. | 6-7, Alg. 1, Thm. 3 | `z <- eta * proposal(z,x) + (1-eta) * z`; eta starts at 1 and geometrically decays by 0.5 after two non-improving residuals. | Finite hard-bounded execution only; no solver library is needed for the source's explicit finite algorithm. |
| Relative infinity residual is a target-free halt signal. | 6-7; 25 | Per sample `max(abs(proposed-state))/max(abs(proposed), 1e-9)`; validation freezes one threshold before formal test. | Exactness is allowed only for disjoint validation calibration and evaluation. Formal inference sees learned state, residual, threshold, patience, decay, minimum depth, and hard max. |
| TBPTT and deep supervision train long fixed-point trajectories. | 7-8, Eq. 4-5, Prop. 1, Alg. 2; 25 | Six training iterations are split into three two-step windows. Each window is decoded and supervised, then state is detached. | Training always executes all six calls to match work; inference uses adaptive exit. |
| Hard iteration cap and adaptive compute are required safeguards. | 14, 25, 27 | Minimum two calls, hard maximum six, active-sample compaction, iteration histogram, hard-max rate, and candidate wall projection. | The paper also stops when eta becomes too small; EXP-671 excludes that second stopping rule because hard max already bounds execution and one validation-frozen halt surface is required. |

## Frozen identities and arms

- Integrated base `a50a2648d072db2ce8387009e58cef0ed461c527` on
  `codex/exp671-fixed-point-damping-halting`.
- Frozen EXP609 checkpoint SHA-256
  `cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd`.
- Pinned fixture SHA-256 values: Dockerfile `1e0424...d0b41`, generator.py `77f414...5c0`,
  primes.py `5d7f49...cbac`; these inputs are read and hashed only.
- Roots: calibration `20261234`, viability `20261235`, and confirmations `20261236/37` only
  after a complete viability pass. W64/W128/W256 use 512 train, 128 validation, and 128 test prime
  identities with disjoint deterministic splits.
- Train `full_fpr`, `stabilization_fixed`, and `plain_fixed` for 400 AdamW updates each on the
  identical `[64,128,256,128]` schedule, batch size 16, six calls, TBPTT window two, and deep loss
  at every window. `damped_fixed` and `halt_without_damping` are causal inference interventions on
  the full checkpoint. `proposal_only`, zero-scale `reset`, and deterministic `randomized` controls
  disclose unmatched work and parameters.
- Threshold candidates are `1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1`. Root 20261234
  validation chooses the least-compute threshold retaining the best validation exactness band. It also
  freezes two smoothed validation-bin probabilities for held-out Brier reliability. Test labels cannot
  affect threshold or probabilities.

## Gates and decision order

Root 20261234 must exercise every training, validation, test, control, residual, intervention,
checkpoint, and persistence route; distinguish wall, aggregate process-tree CPU, cgroup CPU, RSS,
and variability; and dynamically derive the expected, ETA, soft guard, hard guard, and candidate
headroom. Admission requires the complete route, deterministic reload/separation, expected value,
memory, and candidate headroom at most 285 seconds.

If admitted, root 20261235 caches all matched training before the complete 3,000-transition/300-rollout
W128 primary measurement. Continue only while at least 85% exact rollout, zero invalid rollouts,
no more than two-point loss to proposal/fixed depth, trained stabilization dependence, damping effect,
validation-frozen residual reliability better than its validation-prevalence constant predictor,
nontrivial early exits, bounded hard-max rate, and deterministic reload can all pass. W256 then requires
at least 10% exact rollout and five points over proposal, plain fixed, and damped fixed. Both confirmations
must reproduce every conjunct before an intermediate advance.

## Safety boundary

Inference contains no multiplication, division, modular reduction, CRT, lookup, identity feature,
exact validity filtering, output repair, answer-dependent halt/selection, target canvas, constructed
program, or unbounded loop. Exact bigint arithmetic remains confined to legal offline labels and
evaluation. No checked, independent, official/private, candidate, submission, promotion, or completion
surface is authorized. This is a bounded SAIR transfer, not a claim to reproduce the paper's language
Transformer or puzzle results.
