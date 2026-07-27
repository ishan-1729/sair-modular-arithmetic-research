---
type: Experiment Mechanism Map
title: EXP-662 Universal Transformer Memory-Token Claim-to-Code Map
description: Page-anchored bounded SAIR adaptation of arXiv:2604.21999.
status: active
tags: [autoresearch, exp-662, universal-transformer, memory-tokens, act]
timestamp: "2026-07-28T00:15:00+05:30"
---

# EXP-662 Universal Transformer memory-token claim-to-code map

## Local source and visual receipt

The immutable source is `research/papers/arxiv-2604.21999.pdf`, SHA-256 `151499a2f992f6fe335730b90e1cbca3eae9ad7a7705607653eae0117dffbe22`. Text was checked with pypdf/pdfplumber. The bundled Poppler `pdftoppm.exe` rendered cited pages 2-6, 8, and 9 at 120 DPI under `tmp/pdfs/exp662`; all seven PNGs were visually inspected for the architecture, equations, tables, and plots before removal. Their SHA-256 values were respectively `5ca114d5`, `63eb8884`, `6da87428`, `2ebb0f92`, `a2956b89`, `32a62d58`, and `2a40db75` (full values retained in the task receipt). The render showed no clipping or unreadable mechanism content; missing optional display fonts did not affect the inspected text, figures, or tables.

## Source observations and code mapping

| Paper evidence | EXP-662 code | Classification |
|---|---|---|
| Page 2, Fig. 1: one weight-shared pre-norm UT block | `SharedUTCell` recurrently applies one DerfNorm-attention-SwiGLU cell | bounded-faithful |
| Pages 2 and 4-6: prepended learned, sample-independent numbered memory vectors | one ordinary-initialized 64-token parameter bank; the frozen 0/4/8/16/32/64 sweep exposes only its prefix | bounded-faithful; fixed bank keeps stored-parameter identity exact |
| Page 2: memory/sequence types, step embedding, indexed RoPE, QK normalization | learned type vectors, sinusoidal normalized recurrent-step projection, RoPE token indices, and per-head Q/K RMS normalization | bounded-faithful with continuous normalized-step adaptation |
| Pages 2-3: DerfNorm | learned per-feature `erf(alpha*x + shift)` before attention and SwiGLU | bounded-faithful |
| Pages 2-3: per-token ACT blend and fixed-depth alternative | router sigmoid, cumulative/remainder weights, ACT hidden blend, forced final-step control | bounded-faithful |
| Page 3: deep-start router bias -3 versus default 0 | frozen deep-start primary and separately trained default-bias control | bounded-faithful |
| Pages 4-6: count curve and memory-depth trade-off | exact count sweep, selected-count causal ablations, calls/FLOPs/halt/depth curves | bounded-faithful test, not imported evidence |
| Pages 8-9: attention diagnostics, small-data failure, extended depth | attention/halting diagnostics, disjoint selection/test identities, fixed-depth curve | diagnostic adaptation |

## SAIR transition adaptation and exclusions

The source is a bidirectional Sudoku encoder. EXP-662 instead receives only the legal decoded Horner transition tensor: current residue, multiplicand, modulus, current multiplier bit, and generic LSB/MSB markers. It predicts the next decoded residue bits; exact next states are generated offline for training and evaluation only. Autonomous evaluation starts at zero, scans multiplier bits MSB-first, decodes model logits, and feeds only those predictions into the next call.

This is direct learned-token workspace and is distinct from EXP-654's soft-addressed persistent RAM/register/trace mechanism. Memory vectors are shared parameters, never sample-, puzzle-, root-, or evaluation-specific. Forward and halting contain no multiplication, modulo, division, reduction, CRT, lookup, constructed program, label access, answer repair, or exact-arithmetic fallback. Generic type/index/step signals are independently ablated. Native PyTorch 2.5.1 is sufficient; no external model library is introduced.

## Frozen causal and decision plan

All count arms and controls use an identical parameter superset so stored parameters match exactly; inactive control modules are disclosed, while executed calls and analytical FLOPs record route differences. Count selection uses only disjoint W16 validation identities with the lexicographic rule `(autonomous exact, transition exact, -memory_count)`. Formal W16/W24/W32 test identities are untouched until after selection. Root `20261199` must pass every exactness, transfer, dependence, memory-causality, validity, separation, reload, matching, and resource conjunct to open roots `20261200` and `20261201`.
