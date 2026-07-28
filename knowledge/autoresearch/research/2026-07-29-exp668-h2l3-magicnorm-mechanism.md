---
type: Research Note
title: EXP-668 H2L3 and MagicNorm mechanism map
description: Authenticated HRM-Text claims, bounded SAIR adaptations, controls, and exclusions.
status: complete
tags: [autoresearch, hrm-text, h2l3, magicnorm, credit-assignment]
timestamp: "2026-07-29T03:00:00+05:30"
---
# EXP-668 H2L3 and MagicNorm mechanism map

## Authenticated source

The immutable HRM-Text v1 PDF hashes to `c3d8c51d8fcbd44e38c4a9284b75ef8b7d9c2c3e372e2a395672bf38a28a1372`, matching the manifest and checksum ledger. Pages 3–8 and 23–26 were extracted, rendered, and visually inspected.

## Claim-to-code map

- M024.a: page 3 initializes the high state from input and the low state from a fixed value, then runs two outer cycles, each containing three low updates followed by one high update. `H2L3Cell` uses exactly `[L,L,L,H,L,L,L,H]`, separate learned H/L modules, proposal-derived high initialization, and a persistent zero low initializer. Eight module calls disclose more work than the frozen proposal.
- M024.b MagicNorm: pages 3–4 define internal PreNorm residual blocks capped by an exit normalization. `MagicNormModule` uses native GroupNorm for the internal PreNorm path and a parameterless RMS normalization at module exit. A zero-initialized learned output makes every added route exactly preserve the frozen proposal before training.
- M024.b credit: pages 4–5 specify a backward horizon growing from the last two recurrent steps to the last five while forward H2L3 stays fixed. `horizon(progress)` freezes a deterministic linear 2→5 schedule and detaches state immediately before the active suffix. The raw-gradient control backpropagates through all eight steps.
- Pages 23–24 motivate the control by larger gradient spikes, log-magnitude dispersion, and Jacobian growth under deeper H cycling. Calibration records gradient mean, norm, log dispersion, finiteness, and high/low distance at K=2, K=5, and full BPTT.

## Frozen arms and adaptations

Full combines H2L3, MagicNorm, and warmup credit. Hierarchy-only removes MagicNorm/credit; credit-only applies MagicNorm/warmup to eight flat shared updates; flat compute-matched uses eight raw flat updates; no-warmup uses full H2L3/MagicNorm with eight-step BPTT. High reset, low reset, and randomized interventions isolate causal state and trained-weight dependence. Arms match 400 updates, examples, identities, optimizer, and eight recurrent calls, but hierarchy routes have separate H/L parameters while flat routes reuse the low module; this unmatched stored-parameter/FLOP distinction is disclosed.

The convolutional hidden-state transfer is bounded-faithful/analogue evidence, not a transformer or language-pretraining reproduction. Exact bigint arithmetic is restricted to offline labels/evaluation.

## Closed components

M024.c response-only language loss, PrefixLM, and sigmoid language attention gates are closed and absent. M024.d auto-guidance is closed: no shallow-logit interpolation, scale search, answer-dependent depth, or selection exists. Inference has no targets, correctness signal, validity filter, repair, programmed arithmetic, lookup, checked/private data, or adaptive stopping.
