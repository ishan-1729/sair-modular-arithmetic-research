---
type: Research Note
title: EXP-664 SPRM boundary perturbation mechanism map
description: Full-PDF claim-to-code map for a bounded-faithful training-only State Perturbation Recursion Model adaptation.
status: frozen
tags: [autoresearch, exp-664, sprm, tbptt, state-noise]
timestamp: "2026-07-28T23:05:00+05:30"
---

# EXP-664 SPRM boundary perturbation mechanism map

## Source identity and review

The local source is `research/papers/arxiv-2604.18839.pdf`, SHA-256 `6939dc9065d2eca96be7a7e1bf8a7a3fc4ad6d4ea8dd98c036a25886929c00d8`, 26 pages. Text extraction and 144-DPI visual review covered pages 6-10, appendix pseudocode page 20, and the SPRM optimization discussion on page 24. Every rendered page was legible with intact equations, tables, captions, and page numbers. Render hashes were: page 6 `36c6eca5...0799df`, page 7 `564698f8...723a08`, page 8 `e3881ed4...912d67`, page 9 `caead632...3badd`, page 10 `75f721d4...9148b`, page 20 `344eb6c7...7130`, and page 24 `8a2fb707...972e`.

## Claim-to-code map

| Source claim | Source evidence | EXP-664 realization | Verification |
|---|---|---|---|
| SPRM alternates incumbent recursive computation with stochastic forward corruption. | Page 6, section 3.3, Eq. 7. | `SPRMCell.training_forward` runs the frozen shared transition schedule and injects perturbations only on declared training boundaries. | Boundary-event indices and route-specific event tests. |
| Corruption acts on hidden state, not target tokens. | Pages 6 and 8, sections 3.3 and 4.4. | Perturbation targets the continuous ConvGRU hidden tensor between recurrent dilation segments. Labels and decoded residue bits are never corrupted. | Signature/static tests exclude target or decoded-answer corruption. |
| The source uses a linear DDPM variance schedule from `1e-4` to `0.02`. | Page 8, section 4.4. | `linear_beta` maps integer `tau` in `[1,T]` exactly to those frozen endpoints. | Endpoint and monotonicity tests. |
| Perturbation is variance preserving and state scaled. | Page 8, Eq. 9. | `state_scaled_perturbation` applies `sqrt(1-beta) * H + sqrt(beta) * epsilon`, with epsilon sampled per example from the observed hidden-state mean and variance. | Deterministic seeded formula/statistics test. |
| Noise is applied exclusively at outer/truncation boundaries. | Page 8, section 4.4. | Canonical boundaries follow every four recurrent dilation passes; the hidden state is detached there before SPRM perturbation. | Detach graph and exact boundary-index tests. |
| No noise is used during inference. | Page 8, section 4.4. | `forward` delegates directly to the underlying EXP-609 `SharedDilatedConvGRU.forward`; it has no route, noise, or RNG argument. | Bit/logit/quotient equality, call-count, RNG-state, runtime, and source inspection tests. |
| SPRM is a training regularizer that thickens recurrent trajectories. | Page 24, section F.6. | Formal diagnostics preserve per-boundary pre/post mean, variance, noise scale, and clean-reference recovery curves; the claim remains a hypothesis until exact rollout gates pass. | Canonical diagnostics and trained stochastic controls. |

## SAIR adaptations

- ARC task/output scratchpads become the existing 92-channel EXP-609 ConvGRU hidden tensor. This is a training-only hidden-state adaptation; the decoded modular residue is not noised.
- A source TRM outer loop becomes four-pass segments of the already existing shared dilation schedule. Segmenting does not add recurrent updates: clean training executes the exact same dilation list once and only changes gradient truncation.
- Source ARC cross-entropy becomes the existing exact offline bit/quotient transition objective on fresh legal modular identities.
- The source schedule is retained, while local width schedule, batch size, 400-update allowance, optimizer, and exact gates are SAIR-specific and frozen before science.
- `wrong_boundary` and `unscaled_boundary` are local causal controls. They are not source-paper variants.

## Explicit exclusions

- No target-token masking, DRM canvas, decoded-answer corruption, label-conditioned state selection, inference noise, additional inference call, Q-head, augmentation voting, early exit, or ARC task embedding.
- No claim of full TRM two-state scratchpad reproduction, ARC reproduction, paper-result replication, or source-data equivalence.
- No programmed multiplication, division, modular reduction, CRT, lookup, constructed program, identity feature, or output repair in the model or inference route.
- Exact bigint arithmetic is confined to offline data generation, labels, tests, and evaluation.
- EXP-610's arbitrary one/two-bit decoded-state corruption is historical contrast only; it is not treated as SPRM evidence.

## Frozen decision surface

Roots are calibration `20261206`, viability `20261207`, and confirmations `20261208`/`20261209`. Routes are frozen source, canonical SPRM, clean TBPTT, wrong-boundary state-scaled noise, and canonical-boundary unscaled noise. All trained routes receive identical source weights, identities, 400 updates, examples, optimizer schedule, dilation calls, boundary cadence, checkpoint and evaluator. Formal inference is the unchanged source call graph. W128 exact/validity/parity/runtime gates are evaluated first; any irreversible miss leaves W256, controls, causal surfaces, and confirmations unopened as applicable.
