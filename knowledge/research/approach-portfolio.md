---
type: Approach Portfolio
title: Ranked Technical Approaches
description: Prioritized model families, evidence, compliance risk, and next experiments.
status: active
tags: [research-plan, architecture, prioritization]
timestamp: "2026-07-11T23:15:00+05:30"
last_updated: "2026-07-18T04:42:24+05:30"
---

# Approach portfolio

## Measured promotion

The Rank-1 direction has now produced [Candidate 001](../submissions/candidate-001-horner-tcn9.md): 99% exact Tier-2 accuracy on the checked public set and 97% on a full independent seed, with `highest_tier_above_90 = 2`. A bounded random-init phase/DLP probe at `p=251` remained at chance (0.286%-0.414%) and was stopped, reinforcing learned Horner as the active family.

Scores are relative planning judgments from 1 (poor) to 5 (strong), not measured results.

| Rank | Approach | Promise | CPU fit | Compliance clarity | Main risk |
|---:|---|:---:|:---:|:---:|---|
| 1 | Trajectory-trained binary Horner TCN | 5 | 4 | 4 | Chained exactness requires extraordinarily low step error. |
| 2 | ConvGRU / Neural-GPU transition | 4 | 3 | 4 | More recurrent iterations may exceed the runtime envelope. |
| 3 | Learned neural-RAM executor with addressable trace memory | 4 | 2 | 3 | Memory may be expensive or unnecessary, and programmed/compiled initialization would cross the provenance boundary. |
| 4 | Learned 2–8 bit limb recurrence | 4 | 5 | 4 | Much harder carry and reduction transition per step. |
| 5 | Small-prime DLP/additive expert | 2 | 4 | 5 | Official baseline suggests a low ceiling beyond Tier 1. |
| 6 | Trace-supervised relative-position Transformer | 2 | 2 | 5 | Direct arithmetic length generalization remains weak. |
| 7 | End-to-end decoder-only Transformer | 1 | 2 | 5 | Known baseline already collapses on unseen primes. |

### Learned neural-execution challenger

The neural-RAM entry borrows only the compliant mechanisms identified in the [Playwright-verified research note](../autoresearch/research/2026-07-18-learned-neural-execution.md): persistent trained registers, an append-only latent execution trace, learned read/write addressing, shared learned transitions, and low-dimensional attention as a possible fast retrieval bias. It does **not** borrow compiled interpreters, solver programs, hand-assigned instruction semantics, or generated execution weights.

This is a high-upside challenger rather than a replacement for the cellular incumbent. Candidate 005 already carries a compact learned recurrent state, so trace memory must beat both a parameter-matched wider cell and a non-addressable workspace on exact autonomous width extrapolation. It must also prove causal memory use and fit the candidate-shaped CPU envelope. EXP-654 owns that comparison.

## Selected family

Implement Rank 1 at small fixed widths with:

- true Horner trajectory examples;
- prime-disjoint train/validation/test splits;
- a non-causal dilated convolutional cell with full-width receptive field;
- a hard predicted-state rollout;
- exact transition, exact rollout, runtime, determinism, and weight-perturbation metrics.

## Ordered extensions

1. Compare trajectory states with random states.
2. Compare BCE with tail-robust worst-bit margin objectives.
3. Compare prime-value, bit-length, and mixed sampling.
4. Test width interpolation and extrapolation.
5. Replace bit scans with 2-bit, then 4-bit limbs only after the binary control is stable.
6. Compare a ConvGRU cell under the identical data and evaluation contract.
7. Add a small-prime specialist only if it improves the ranking metric without meaningful latency cost.

## Super Transformers coverage program

The [binding mechanism contract](../autoresearch/research/2026-07-21-super-transformers-coverage-contract.md) expands the portfolio without replacing evidence-based prioritization. It separates faithful mechanisms from bounded-faithful tests, SAIR analogues, supporting-only sources, and literal exclusions across exactly 42 stable records while preserving all 12 `super_transformers_main_papers.md` mappings. Tab-TRM is deduplicated into TRM application evidence, the HRM incorrect-attractor paper informs false-attractor diagnostics, and the ARC Prize technical report is supporting-only. Ordinary-initialized learned token execution is an analogue; literal hand-programmed execution remains excluded. T²MLR is likewise an analogue unless a justified autoregressive decoder exists.

After EXP-648 resolves, the staged portfolio order is structural equivariance; shared recurrent depth, including LoopLM controls; explicit answer-plus-latent TRM/HRM/RIM/RSM models; separate LoopFormer, DRM, and LOTUS tests; visit/dilation adapters; proper-library Mamba/state-space recurrence; a materially new learned-memory mechanism; signature-gated stability/fixed points; runtime-qualified stochastic breadth and energy selection; winner-dependent adaptive compute; and conditional autoregressive/T²MLR work. These are sequential research families, not simultaneous authorizations.

Mature authoritative libraries are the default for Mamba/selective-SSM, graph, attention, and other specialized kernels. Dependencies must be pinned, licensed, provenance-recorded, and parity-tested against tiny reference fixtures. CUDA may accelerate offline training, but the frozen submission still requires a deterministic CPU path, cross-device parity where applicable, and candidate-shaped runtime evidence. Hand implementations are justified only when no suitable maintained implementation exists and a Judge accepts the parity and maintenance risk.

This program is queued, not active. EXP-648 remains the only active hypothesis. Readiness root `20261160` is consumed-incomplete, non-scientific, and permanently denied; root `20261161` is consumed-complete, non-scientific, sealed, and non-retryable after fresh full-fidelity calibration with no cache reuse. Completion audits return `[APPROVE, APPROVE]`, and the Judge decision is `approved-complete-non-scientific-readiness-calibration`. Renewed approval covers exactly 17 registration paths. Immutable approved-base/scaffold/execution-test provenance remains `87384f92ab1c053c7791e8afd69a07936d5a7402` / `593d18523ff2888c1d57d8626d81f4ebb07a234427573efa3afc226e98e2c5a6` / `f66fb2c7e510171bb92a5cab20b842b9b08d9dd023fc44f37ee1e8b74ef08ef5`, while stabilized repaired post-publication regression-test SHA-256 `967fdc7d4ce5e9b8deaf9d5ea0d4b0cd74d8b4729b66c75b5eadf461626cfb50` is separate and must be resolved by `screen_readiness_binding` from the committed registration base. The pre-repair full suite stopped before commit at 257 passes / two stale lifecycle failures; the exact targeted three passed, and the first focused-file run stopped before commit at 87 passes / four approved-base fixture failures before the second bounded same-file repair. The expanded targeted set passed 8/8, the focused file passed exactly 91/91, the full suite passed exactly 259/259, and the project validator passed. This is resource evidence only: no screen or scientific surface is open, and `judge-authorization.json` remains absent. The requested arXiv:2607.16051 review is queued but not started. EXP-650 through EXP-653 retain their meanings and prerequisites. Candidate 005 remains unchanged until a challenger passes the complete promotion contract.

## Stop conditions

Stop scaling a family if held-out-prime rollout remains near the trivial baseline after transition accuracy saturates, if perturbing weights does not collapse behavior, or if projected CPU latency cannot fit the official envelope with at least 25% headroom.
