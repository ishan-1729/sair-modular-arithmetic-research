---
type: Experiment Mechanism Map
title: EXP-661 LoopFormer Full-PDF Claim-to-Code Map
description: Page-anchored extraction and bounded SAIR adaptation of arXiv:2602.11451.
status: active
tags: [autoresearch, loopformer, recurrent-transformer, exp-661]
timestamp: "2026-07-27T20:30:00+05:30"
---

# EXP-661 LoopFormer full-PDF claim-to-code map

## Source observations

The local immutable source is `research/papers/arxiv-2602.11451.pdf`. Pages 4-6 define a cyclic shared Transformer stack, generic token and position embeddings, normalized cumulative time `t`, positive trajectory step `delta` with steps summing to one, sinusoidal scalar embeddings projected by MLPs, RMSNorm scale modulation, pre-residual MHSA/FFN gates, maximum and sampled-short trajectories, full and short prediction losses, stop-gradient hidden consistency, and fixed user-selected inference budgets without learned exit. Pages 8-10 report depth/hidden dynamics, fixed-budget schedule sensitivity, global rather than instance-adaptive budgets, training overhead, and correlational—not causal—representation analyses.

## Claim-to-code mapping

| Source mechanism | EXP-661 implementation | Classification |
|---|---|---|
| Shared cyclic Transformer stack, pp. 4-6 | One `ConditionedLoopBlock` is recurrently applied in `LoopFormer.loop` | bounded-faithful |
| Normalized `t` and `delta`, pp. 4-6 | Sinusoidal scalar encodings and separate MLP projections; every schedule is positive and sums to one | bounded-faithful |
| RMSNorm scales and residual gates, pp. 4-5 | Condition MLP emits two scale vectors and two sigmoid pre-residual gates | bounded-faithful |
| Full and sampled-short objectives, pp. 5-6 | Endpoint BCE on the W16 full and grid-sampled short trajectories | SAIR endpoint adaptation |
| Stop-gradient consistency, Algorithm 1 p. 6 | MSE from short hidden endpoint to detached full hidden endpoint | bounded-faithful |
| User budget without learned exit, pp. 5-6 and 10 | Fixed budgets 1/2/4/8/16; no answer-dependent halt module | bounded-faithful; budget 16 is explicit depth extrapolation beyond trained L=8 |
| Per-loop hidden/logit dynamics, pp. 8-9 | Hidden delta/cosine and logit-delta curves plus exact/bit curves by budget | diagnostic adaptation |
| Decoder-only language modeling and next-token CE | Bit-position tokens with generic learned positions and endpoint BCE | SAIR task adaptation |

## Exclusions and safety boundary

EXP-661 is not LoopViT and does not reproduce a pixel/image path. It does not reproduce language-corpus scaling, learned or per-token exits, instance-adaptive schedule policies, puzzle/evaluation IDs, answer channels, programmed arithmetic, constructed algorithms, or answer repair. Exact multiplication and reduction occur only in offline label generation, tests, and evaluation truth. Forward paths contain learned tensor operations only. The native PyTorch 2.5.1 implementation needs no added dependency.

## Causal plan

The parameter-matched plain-looped and decoded-feedback controls retain every parameter while disabling trajectory modulation; all routes disclose identical examples, updates, and recurrent calls. Separate ablations disable time/step conditioning, shortcut gates, short loss, or hidden consistency. Each width/root uses 24 train-prime and 8 disjoint held-out-prime identities; this feasible W8 identity count was frozen before any root opened. Viability evaluates fresh W16/W24/W32 cases at fixed budgets 1/2/4/8/16, randomized weights, inference interventions, deterministic reload, hidden/logit curves, invalid outputs, and exact-answer accuracy. Component confirmations remain gated on every viability conjunct.
