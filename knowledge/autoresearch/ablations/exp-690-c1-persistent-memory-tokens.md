---
type: Experiment Preregistration
title: EXP690 C1 Persistent Memory Tokens
description: Frozen full-budget primary/control protocol for four case-local tokens on an h48 Candidate-derived specialist.
status: completed-resource-reject
tags: [autoresearch, ablation, persistent-memory, candidate005]
timestamp: "2026-07-30T00:00:00+05:30"
---

# EXP690 C1 persistent memory tokens

## Hypothesis and changed variable

Candidate005's hard learned residue remains authoritative. C1 adds four learned 16-D tokens that persist only within one example/call. The h48 spatial state attends to tokens before the one outward dilation cycle; the access output begins at exact zero. After each learned transition, tokens attend to the final spatial state and update through a shared GRU. The control owns, computes, updates, stores, serializes, and optimizes identical tensors but multiplies token-to-core injection by exact zero.

All else is frozen: top-48 answer-independent Candidate checkpoint slice; outward visits `[1,2,4,8,16,32,64,128]`; exact/logit/quotient distillation; 2,400 updates per arm; curriculum `[64,128,256,256]`; horizon 8; batch 16; optimizer; data; roots; evaluators; and inference route. There is no operand-swap route.

## Roots and gates

- `20261314`: warmup plus three cold complete 1,100-case routes, plus two if variability triggers. Require `2*max<=285s`, RSS <=6.4 GiB, determinism, reload, isolation, and equal work.
- `20261315`: only if admitted, both arms complete 2,400 updates before evaluation. Run 3,000 transitions at W128/W256; 300 paired W128 rollouts; cumulative 200/300/450/600 paired W256 rollouts; reset/freeze/permutation/randomized-token interventions.
- `20261316-17`: only if root15 is complete, every first-rung gate passes, and W256 rollout is at least 85%.
- `20261318-22`: only after both confirmations; each >=90%, aggregate >=93%, lower tiers bit-identical, and fresh runtime admission.

Support requires control W128 transition >=99%, primary W256 transition >=99.8%, primary W256 rollout >=50%, invalid <=1%, paired gain >=5 points, W128 loss <=5 points, and every intervention erasing at least half the observed gain. Confidence uses the frozen 99% time-uniform Hoeffding sequence with Holm allocation. A resource/mechanical failure parks; a scientific negative is permitted only after both 2,400-update arms finish.

## Safety, provenance, and resources

The source checkpoint is `cbc4a1bc...b206d4bd` and challenge snapshot is `82510bba...b3eb60`. Exact arithmetic and teacher outputs are offline only. Inference contains legal features, native PyTorch SDPA, GRU/token updates, ConvGRU, logits, and hard thresholds. Tokens reset per example/call. No solver, division/reduction routine, program, lookup, verifier, candidate arithmetic, validity filter, answer selection, checked/private tuning, root reuse, or post-result retuning is allowed.

The sole package oracle uses historical ~535s wall/~805s process CPU with hard wall at least 1333.702125s. Root14 records expected, 1.25x soft, and 1.75x hard wall budgets plus process/cgroup CPU and RSS. ETA/completion monitoring replaces polling. Passing is not tournament victory or promotion; C2-C7 remain mandatory whatever C1 reports.

## Result

The all-mode smoke and 81-test focused cone passed. The sole package oracle passed validation (225 concepts/1,100 records) and all 824 tests in 604.86s wall, using 930.57s process CPU and 949,068 KiB peak RSS. Root20261314 then completed three deterministic cold 1,100-case deployed routes at 158.0452s, 150.1241s, and 153.2379s. Variability did not trigger two extras. Peak RSS was 1,091,407,872 bytes, but `2 * max = 316.0903s` exceeded 285s. C1 is therefore parked on resource admission. Root15 and roots16-22 were never opened; the four-update calibration has no scientific accuracy, paired-confidence, or causal interpretation. C2-C7 remain required.
