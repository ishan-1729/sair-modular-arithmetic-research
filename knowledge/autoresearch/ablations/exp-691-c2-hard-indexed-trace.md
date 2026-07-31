---
type: Experiment Preregistration
title: EXP691 C2 Hard Indexed Trace
description: Frozen full-budget h48 Candidate-derived append-only learned trace versus correction-masked control.
status: completed-rejected
tags: [autoresearch, ablation, append-only-trace, hard-addressing, candidate005]
timestamp: "2026-07-30T00:00:00+05:30"
---

# EXP691 C2 hard indexed trace

C2 retains the frozen top-48 Candidate005 slice and its single outward dilation cycle. After every learned Horner transition it appends exactly one learned 32-D record to chronological storage shaped `[B,ceil(W/16),16,32]`. Before the next transition, a learned query first selects a valid block and then a valid slot. Both selections are hard argmax/gather in the forward pass with softmax straight-through gradients. A learned read-conditioned spatial FiLM/residual projection starts at exact zero.

The matched control creates, writes, stores, addresses, gathers, diagnoses, serializes, and optimizes the identical trace and pointer graph. It masks only the read-conditioned correction to exact zero. The trace resets for every example and never overwrites. Addresses have no programmed arithmetic meaning.

All other work is frozen: exact Candidate slice, visits `[1,2,4,8,16,32,64,128]`, exact/logit/quotient distillation, 2,400 updates per arm, curriculum `[64,128,256,256]`, horizon 8, batch 16, optimizer, data, roots, evaluators, and Candidate005 Tiers1-6 route.

## Roots, evidence, and gates

- `20261323`: warm plus three cold complete 1,100-case deployed routes, adding two when `max/median > 1.10`; require `2*max<=285s`, RSS <=6.4 GiB, exact reload/determinism, maximum trace allocation, and equal work.
- `20261324`: only after admission; both arms must finish all 2,400 updates before accuracy rejection, followed by 3,000 W128/W256 transitions, paired W128 rollout, cumulative 200/300/450/600 W256 rollout evidence, pointer entropy/utilization/collision/age, and erase/order-permute/uniform/random-pointer interventions.
- `20261325-26`: open only after complete root24 W256 exact accuracy >=85% and every shared/mechanistic gate passes.
- `20261327-31`: open only after both confirmations; each must reach >=90%, aggregate >=93%, with fresh runtime and lower-tier identity.

Support requires a healthy control, W256 transition >=99.8%, W256 rollout >=50%, invalid <=1%, paired gain >=5 points, W128 loss <=5 points, noncollapsed addressing, and each intervention erasing at least half the gain. Confidence uses the frozen 99% time-uniform Hoeffding sequence with Holm allocation. A scientific negative is legal only after both arms complete 2,400 updates.

The source checkpoint is `cbc4a1bc...b206d4bd` and challenge snapshot is `82510bba...b3eb60`. Exact arithmetic and teachers are offline only. Inference contains learned tensor encoders, two-stage ST hard addressing, gather, ConvGRU, learned correction, logits, and hard thresholds—no solver, verifier, arithmetic routine, lookup, answer selector, validity filter, program counter, or pointer semantics.

The sole package oracle derives from the current 604.86s wall/930.57s process-CPU suite class with a 1512.15s hard wall. Root23 measures complete update/evaluation/deployed throughput and separately derives expected, 1.25x soft, 1.75x hard wall/process/cgroup CPU/RSS budgets. No short cap, projection-only admission, root reuse, post-result tuning, or fidelity reduction is permitted. C2 is independent; its outcome cannot cancel C3-C7.

## Result

The all-mode smoke passed in 45.86 seconds and the sole package oracle passed validation (226 concepts, 1,100 records) plus all 859 tests. Root20261323 admitted the exact deployed route: three complete cold routes took 126.829, 135.412, and 127.857 seconds, so the frozen 2x guard was 270.824 seconds against 285; peak RSS was 1,072,676,864 bytes.

Root20261324 then completed all 2,400 updates for both 43,220-parameter arms. Each arm performed 19,200 trace appends, block reads, slot reads, correction computations, student calls, teacher calls, and exact-loss calls. The full route used 5,862.791 wall / 21,990.650 process CPU / 23,422.088 cgroup CPU seconds with 1,531,633,664-byte peak RSS.

Both arms were zero-exact on all formal denominators: 3,000 W128 transitions, 3,000 W256 transitions, 300 W128 rollouts, and 600 W256 rollouts per arm. Primary invalid rates were 92% at W128 and 95% at W256. The W256 paired result was 0 wins, 0 losses, and 600 ties; all four intervention erasures were zero. Learned addressing did not collapse (240 unique W256 addresses), but noncollapse without output gain or causal erasure is not mechanism support.

Reject this exact C2 formulation under the frozen scientific and causal gates. Roots20261325-31 remain unopened. This result does not cancel or weaken C3-C7 and makes no checked/private, candidate, submission, promotion, release, or tournament-winner claim.
