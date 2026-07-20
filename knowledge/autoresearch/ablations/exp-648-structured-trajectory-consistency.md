---
type: Experiment Preregistration
title: EXP-648 structured natural-trajectory consistency
description: Matched K=8 endpoint-consistency screen on natural width-256 failure and success windows from the frozen cellular source.
status: selected-preregistered-pending-implementation-calibration
tags: [autoresearch, tier7, trajectory-consistency, endpoint-loss, matched-ab, exact-rollout]
timestamp: "2026-07-20T10:20:00+05:30"
last_updated: "2026-07-20T10:20:00+05:30"
---

# EXP-648 structured natural-trajectory consistency

## Status and falsifiable question

The Judge selected EXP-648 as the next hypothesis with the best expected information value per CPU and wall time. This document freezes a preregistration only. The runner, corpus, calibration, screen, and confirmation do not yet exist and no experiment root has been opened.

The falsifiable question is whether a fixed `0.1` endpoint-consistency term over eight natural autonomous transitions improves width-256 exact rollout by at least three percentage points over an otherwise identical `0.0` control, moves the mean first error no earlier, and retains width-64 and width-128 exact rollout within one point.

Candidate 005 remains unchanged at 59.5% checked-public and 59.3% independent overall accuracy. It has no learned path above 128 bits, so this is not a Tier-7 or candidate claim. EXP-649 remains planned/formal-blocked after its full input-only calibration produced `runtime-gate-miss`; its unopened formal root `20261149` is permanently locked for that frozen formulation. Its breadth was never tested.

## Evidence basis and transfer boundary

The dated [source note](../research/2026-07-20-structured-trajectory-consistency.md) records the Playwright-verified primary pages, local PDF hashes, author claims, and project limitations. DRM motivates multi-step trajectory supervision, while LoopFormer motivates shortcut consistency across recursive computation. EXP-648 borrows only natural-trajectory K-step endpoint consistency. It does not use synthetic corruption, reverse noise, time or step conditioning, variable compute, adaptive halting, or either source architecture.

The local diagnosis is more important than analogy. EXP-643 observes 99.70–99.75% width-256 transition accuracy but only 22%/32% rollout, while EXP-610's arbitrary one-/two-bit corruption regresses width-128 rollout from 87.5% to 75.0%. Accordingly, EXP-648 injects no state corruption. It begins from exact prefixes already encountered in natural source trajectories and reports both recovery and newly introduced drift.

## Frozen source and seed isolation

Both arms independently load the same raw hidden-92, two-cycle, 99,824-parameter checkpoint at `.scratch/exp-609-tier6-straight-through-confirmation/straight_through_continuation/weights.safetensors`, SHA-256 `cbc4a1bca276cd8027b0c546892f47746d8416fedd623e504701e313b206d4bd`. The two initial parameter tensors must have the same hash before the first update.

Every seed below is currently unused and unopened:

| Role | Frozen root | Opening rule |
|---|---:|---|
| Resource calibration only | `20261151` | After the implementation and static-test gate; never scientific evidence |
| Corpus construction | `20261152` | Generated training-split identities only, under the resolved mining ceiling |
| Screen training/order | `20261153` | Only after corpus, code, config, tests, hashes, and measured budgets are frozen and Judge-approved |
| Screen paired evaluation | `20261154` | Same screen launch gate as `20261153` |
| Confirmation 1 training/order | `20261155` | Only after a complete screen pass and separate Judge authorization |
| Confirmation 1 paired evaluation | `20261156` | Same confirmation authorization |
| Confirmation 2 training/order | `20261157` | Only after a complete screen pass and separate Judge authorization |
| Confirmation 2 paired evaluation | `20261158` | Same confirmation authorization |
| Scaffold tests only | `648900001` | Never experiment or evaluation evidence |

Checked/public inputs and answers, every prior independent set, the official private benchmark, and EXP-649 formal root `20261149` are excluded. Tests may use only `648900001`; they may not invoke any calibration, corpus, screen, confirmation, checked, independent, private, or EXP-649 formal root.

## Frozen width-256 corpus

Run the frozen source autonomously on newly generated width-256 training-split identities from corpus root `20261152`. No benchmark identity is eligible. Keep exactly 512 failure windows and 512 successful windows, paired one to one. No state bit is injected, flipped, masked, or otherwise synthetically perturbed.

A failure is eligible only when its zero-based first wrong transition index is at most 248, so a complete `K=8` window can begin at the exact prefix immediately before that first wrong transition. The failure window then follows the source's natural hard-decoded autonomous states for eight transitions.

For every failure, select the first unused successful window with all of these exact matches:

- the same zero-based window-start index;
- the same exact modulus bit length; and
- the same normalized-modulus band, where normalized modulus is `p / 2**p.bit_length()` in `[0.5,1.0)` and the four bands are `[0.500,0.625)`, `[0.625,0.750)`, `[0.750,0.875)`, and `[0.875,1.000)`.

An eligible successful window starts from the exact prefix at that matched index and remains exact for all eight autonomous transitions. Corpus-root generator order determines failure order. For a matching stratum, the first unused eligible success in the same deterministic generator/trajectory order wins. No replacement, rematching, relaxed band, relaxed bit length, shortened window, or reduced count is permitted.

Before training, freeze and hash the identity manifest, complete source trajectory manifest, source checkpoint, exact offline teacher source, and final paired corpus. Exact arithmetic is allowed for the offline teacher and corpus labels only. If 512 matched pairs cannot be mined under the post-calibration resolved corpus-mining ceiling, record `corpus-construction-failure`; do not weaken the count or matching contract.

## Matched arms and sole changed variable

Each phase creates separate arm instances by independently cloning the raw source checkpoint. The arms must have identical parameter counts, initial tensor hashes, frozen windows, batch order, optimizer, schedule, batches, updates, clean teacher-forced branch, autonomous forward/backward work, endpoint-loss computation, checkpoint work, evaluation inputs, and serialization/hashing work.

Each batch contains sixteen windows: eight failures and their eight matched successes. Each window executes:

1. a clean teacher-forced branch over the same eight exact transitions; and
2. a `K=8` autonomous branch whose residue state is hard-thresholded in the forward pass and uses sigmoid straight-through gradients in the backward pass.

For each clean step, the existing transition loss is mean residue-bit BCE plus `0.45` times the worst-position mean BCE plus `0.10` times quotient cross-entropy. `base_clean_loss` is the mean of that loss over the eight exact teacher-forced steps.

Both arms compute the same endpoint residue-bit loss on the eighth autonomous logits against the offline exact eight-step endpoint: mean residue-bit BCE plus `0.45` times worst-position mean BCE, with no quotient term. The total loss is:

`base_clean_loss + lambda_endpoint * endpoint_loss`.

The sole changed variable is `lambda_endpoint`: control `0.0`, challenger `0.1`. There is no coefficient, horizon, corpus, or schedule sweep.

Use AdamW, learning rate `5e-6` to `5e-7` under one cosine schedule, weight decay `2e-5`, gradient clipping at `2.0`, four PyTorch threads, one interop thread, and deterministic algorithms.

## Screen

The screen uses roots `20261153` and `20261154`. Each arm is independently reinitialized from the source and receives exactly 200 matched updates. Evaluate both checkpoints on identical paired inputs:

- 2,000 fresh transitions at width 128 and 2,000 at width 256 per arm;
- 200 complete rollouts at each of widths 64, 128, and 256 per arm; and
- an exact repeat of the same twenty width-256 rollout cases.

The zero-based first-error index is the first wrong transition. A successful 256-transition rollout receives the frozen clipped value `257`. Report each arm's mean clipped first-error index, plus paired recovered cases (`control wrong, challenger exact`) and introduced drift (`control exact, challenger wrong`).

The challenger advances only if every gate passes:

- width-256 exact rollout gain over control is at least `+0.03`;
- challenger mean clipped first-error index is at least the control mean;
- width-64 and width-128 exact rollout deltas are each at least `-0.01`;
- challenger width-256 exact transition accuracy is at least `0.995`;
- challenger invalid-rollout rate is at most `0.01`;
- the twenty-case repeat is bit-identical and deterministic; and
- source, initial-state, parameter, data/order, seed, checkpoint, evaluator, and matched-work integrity checks all pass.

Any miss rejects this exact frozen screen. A pass is intermediate evidence only and requires a separate Judge decision before confirmation. It does not authorize a candidate, Tier-7 promotion, or benchmark evaluation.

## Conditional two-pair confirmation

Confirmation is prohibited unless the complete screen passes and a Judge explicitly authorizes it. It consists of exactly two fresh matched A/B reinitializations from the raw checkpoint, not continuations from the screen or from each other. Pair 1 uses training/evaluation roots `20261155`/`20261156`; pair 2 uses `20261157`/`20261158`. Each arm receives 400 updates.

For each pair, evaluate 2,000 transitions at widths 128 and 256, 200 full rollouts at widths 64 and 128, 500 width-256 full rollouts, and an exact repeat of twenty width-256 cases. Each pair must independently achieve:

- at least `+0.03` challenger-minus-control width-256 rollout accuracy;
- nonnegative movement in mean clipped first-error index;
- width-64 and width-128 retention deltas each at least `-0.01`;
- challenger width-256 transition accuracy at least `0.995`;
- challenger invalid-rollout rate at most `0.01`; and
- exact determinism and all integrity checks.

Pool the 1,000 paired width-256 cases across both confirmations. Let `b` be recovered cases and `c` introduced-drift cases. With one-sided `alpha=0.05`, compute the exact Clopper-Pearson lower bound for `b/(b+c)`: zero when `b=0`, otherwise `BetaQuantile(0.05, b, c+1)`. If `b+c=0`, define the net exact-gain lower bound as zero. Otherwise compute:

`(2 * p_lower - 1) * (b + c) / 1000`.

That lower bound must be strictly positive. Any per-pair or pooled miss rejects the frozen confirmation. A complete pass supports only a separately preregistered candidate/Tier-7 gate; it never promotes directly.

## Binding dynamic resource protocol

Resource calibration is a separate, non-scientific workload on root `20261151`. After one complete warmup, run three complete timed repeats. Every warmup and repeat must measure the same representative work: mine 64 matched pairs through the exact width-256 corpus path; include startup/model loading; execute 20 real K=8 updates per arm at batch 16; write each arm checkpoint; evaluate 200 transitions per arm at widths 128 and 256; evaluate twenty complete rollouts per arm at widths 64, 128, and 256; and perform the declared hashing and serialization.

Record external monotonic wall seconds, process user-plus-system aggregate CPU seconds, and peak RSS separately. Use four PyTorch threads and one interop thread. For every projected component, use the maximum complete-repeat rate and the slower arm. Wall and CPU are different resources and may not substitute for each other.

Project the screen and each confirmation pair separately from measured setup plus every declared corpus or corpus-integrity, training, checkpoint, transition, rollout, repeat, hash, and serialization unit. The screen includes construction of all 1,024 frozen windows. Confirmations reuse only the hash-valid frozen corpus and pay its full integrity-validation work; they do not silently remine or change it.

For wall and aggregate CPU independently:

- explicit measured setup headroom is twice the maximum complete-repeat startup/checkpoint/hash/serialization component;
- soft budget is `1.25 * projected component work + explicit measured setup headroom`; and
- hard runaway ceiling is `1.5 * soft budget`.

The initial calibration planning ceilings are 14,400 wall seconds and 57,600 aggregate CPU seconds. They are generous planning-only runaway protection relative to EXP-647's full 5,141.6-wall/20,465.9-aggregate-CPU-second run. They are not targets, scientific gates, submission-runtime limits, or permission to reduce fidelity.

Recalibrate after any runner, config, checkpoint, environment, thread, or declared-workload hash change, or if first-quarter throughput is more than 20% slower than projection. Cache and preserve complete unit evidence and checkpoints under keys containing checkpoint, evaluator, dataset, config, seed, environment, runner, scientific-protocol, and resource-protocol hashes. Never reduce windows, updates, cases, repeats, diagnostics, or evaluation fidelity to fit a budget. Evaluation-only overrun preserves the checkpoint and records missing evidence; it is not scientific rejection.

The screen roots remain unopened until corpus, runner, tests, config, scientific/resource protocol, environment, and measured budgets are frozen and a post-implementation Judge approves launch. Confirmation roots remain unopened until a screen pass and another Judge authorization.

## Compliance and decision boundary

EXP-648 changes training loss only. Inference architecture, semantics, and the 99,824-parameter count remain unchanged. Exact arithmetic is confined to offline corpus construction, teachers, and evaluation. No benchmark answers select data, hyperparameters, checkpoints, or gates, and no exact modular solver or output repair enters inference.

EXP-650 remains blocked until a separate proper-library readiness, CPU, and submission gate is approved; this preregistration does not install a dependency. EXP-651 is deferred because the prior EXP-617 gain and control geometry are too weak. EXP-648 is the only selected runnable follow-up, but remains blocked from execution pending implementation, corpus/resource preparation, frozen hashes, and Judge review.
