---
type: Ablation
title: EXP-662 Universal Transformer memory-token transition
description: Bounded-faithful Universal Transformer learned-memory-token transition package, parked at its final-bytes oracle boundary.
status: parked-final-bytes-full-oracle-unverified
run_id: exp-662-ut-memory-token-transition
date: 2026-07-28
timestamp: "2026-07-28T00:59:32+05:30"
---

# EXP-662 Universal Transformer memory-token transition

EXP-662 tested a bounded-faithful Universal Transformer mechanism: prepended sample-independent learned memory tokens, type and step embeddings, a shared pre-norm DerfNorm attention/SwiGLU cell, RoPE with Q/K normalization, and ACT. Exact Python arithmetic was confined to offline labels and evaluation.

## Observed result

Root `20261198` completed every route class and admitted the full route. Root `20261199` trained ten exactly parameter-matched 9,818-parameter routes for 300 updates, 9,600 examples, and 57,600 recurrent block calls each. The frozen validation rule selected `memory_0`: every memory sweep arm tied at zero rollout and transition exactness, so the negative-memory-count tie-break applied.

On fresh test identities the selected route scored 0% exact transition and 0% exact autonomous rollout at W16, W24, and W32. Bit accuracies were 58.33%, 55.25%, and 51.73%. Every matched control and learned-memory arm also scored 0% exact rollout at every width. Randomized weights, reset/zero/randomized/permuted memory, removed type/index signals, and all fixed depths likewise remained at the zero-exact floor. Reload, parameter, work, validity, split separation, and resource checks passed; exactness, transfer, trained-weight dependence, memory causality, and representation causality failed.

## Decision

Park this exact package at the final-bytes verification boundary. The zero-exact root is diagnostic and non-promotable, not a compliant scientific rejection. Roots `20261200` and `20261201` remain unopened. This does not reject the SAIR formulation, source paper, larger data/compute, or other memory-token training objectives. Candidate 005 and all benchmark, submission, promotion, and completion surfaces are unchanged.

## Verification limitation

The project validator passed. The single full pytest recorded 429 passes and seven late submission-loader errors caused by the EXP-662 test helper caching a generic `model` module. The bounded isolation repair was then verified by the EXP-662 plus affected submission-loader suites (18/18); the one-full-oracle rule prohibited a duplicate full run. Because science was opened after that red boundary, its results are retained only as diagnostic evidence.
