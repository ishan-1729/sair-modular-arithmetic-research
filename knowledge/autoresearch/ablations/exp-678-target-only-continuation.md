---
type: Experiment Preregistration
title: EXP678 Target-Only Continuation
description: Frozen ST-M044.a target-position CE-mask comparison on the unchanged Candidate005 recurrent core.
status: permanent-pre-root-authentication-park
tags: [autoresearch, exp678, target-only, supervision, candidate005]
timestamp: "2026-07-30T01:00:00+05:30"
---

# EXP678 target-only continuation

## Decision

Test the `ST-M044.a` analogue on the frozen EXP609 Candidate005 recurrent core. The only changed scientific variable is which already-computed continuation transition losses carry gradient. Formal inference remains byte/schema/call/output-compatible with the original `SharedDilatedConvGRU`.

## Source mapping and limits

Loopies §4.1, pp. 16-18 defines supervised pre-training as cross-entropy over supervised target-response positions only, with prompt/context/padding positions masked. Table 8, p. 38 reports batch 1,024, sequence length 131,072, learning rate `1e-5`, warmup-then-stable scheduling, AdamW, and gradient clipping. EXP678 transfers only the target-position mask. It does not reproduce the paper's language-model scale, tokenization, batch/context geometry, approximately 2T token budget, ten epochs, MoE auxiliary loss, or downstream claims. The paper acknowledges limited ablation, so all claims remain analogue-only.

## Frozen comparison

All trained arms start from the authenticated EXP609 checkpoint except the randomized-weight control. Every non-standalone continuation update computes the same eight transition BCE/quotient-CE tensors, uses the same data order, optimizer, 400 updates, examples, forward calls, and fixed denominator:

- `target_only`: coefficients `[0,0,0,0,0,0,0,8]`;
- `mixed_trajectory`: coefficients `[1,1,1,1,1,1,1,1]`;
- `endpoint_weight`: coefficients `[1,1,1,1,1,1,1,8]`;
- `shuffled_mask`: deterministic one-position coefficient `8`;
- `frozen_source`: no updates;
- `randomized_control`: target-only training from deterministic ordinary initialization.

Target-only and endpoint-weight give the final target equal reduced coefficient `1.0`; endpoint-weight deliberately retains trajectory gradients. Effective supervised positions and coefficient mass disclose the irreducible gradient-work difference.

## Roots and gates

Root `20261262` is non-scientific complete-route calibration. It must measure setup, every arm and schedule mode, evaluation, persistence/reload, candidate inference, wall, process-tree CPU, cgroup CPU, RSS, variability, throughput, and derived lower/upper ETA plus soft/hard guards. It opens root `20261263` only if parity, compliance, memory, expected value, artifact size, unchanged inference, no fidelity reduction, and guarded 100-case W128 candidate inference at most 285 seconds all pass.

Root `20261263` trains and caches all arms before evaluation, then scores the complete 300-case W128 target-only primary first. It stops on any irreversible miss: W128 exact below 95%, transition below 99.8%, invalid output, more than two-point loss to frozen or mixed, inadequate trained-weight or mask dependence, reload/determinism/inference mismatch. Only a complete W128 pass opens W256, which requires at least 10% exact and five points over mixed and endpoint-weight. Roots `20261264` and `20261265` must independently reproduce every conjunct before any later phase may consider the result.

No checked, independent, official/private, candidate, submission, promotion, or completion evaluation is authorized.

## Terminal verification outcome

Nine focused tests passed on the frozen package bytes. The sole dynamically budgeted package oracle stopped in project validation before pytest because this document and the mechanism map omitted the required `description` frontmatter field. The bounded owned repair adds only those fields and terminal records; a second package oracle is prohibited. Root `20261262` and roots `20261263`-`20261265` remain unopened. This is a verification park, not scientific evidence for or against target-only continuation or M044.a.

T076 subsequently normalized the four validator-named canonical JSON fields without changing executable/config/test or scientific semantics, passed validation, and completed the first full project suite at 655/655. The mandatory pre-root authentication still rejects the accepted freeze because it binds this document and the mechanism map before T075's required description/terminal edits. T076 prohibits re-freezing or changing T075 bytes, so all roots remain unopened. The terminal classification is therefore a pre-root provenance park, still with no M044.a scientific evidence.

T077 proved that only the two governance hashes and repository-commit provenance differ, rebound them exactly, and obtained zero differences after JSON normalization. The sole raw authentication still fails because serialized JSON uses string keys for numeric evaluation maps while the live YAML-derived protocol uses integer keys. No retry or same-boundary successor is authorized. This exact execution formulation is permanently parked before calibration with every root unopened.
