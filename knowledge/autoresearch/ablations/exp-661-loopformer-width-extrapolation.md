---
type: Ablation
title: EXP-661 LoopFormer width and budget extrapolation
description: Bounded-faithful ST-M013.c LoopFormer viability and gated component package.
status: completed-rejected
tags: [autoresearch, exp-661, loopformer, recurrent-transformer]
timestamp: "2026-07-27T23:58:00+05:30"
---

# EXP-661 LoopFormer width and budget extrapolation

## Frozen question and method

Can a small shared cyclic Transformer with normalized time/step conditioning, RMSNorm scale modulation and residual gates, full/short endpoint losses, and stop-gradient hidden consistency learn exact W16 modular products and transfer across both fixed user budgets and untouched W24/W32 widths better than matched plain-looped and decoded-feedback controls?

The local full-PDF map is [`2026-07-27-exp661-loopformer-mechanism.md`](../research/2026-07-27-exp661-loopformer-mechanism.md). It classifies bit tokens, endpoint BCE, generic positions, and the budget-16 extrapolation as SAIR adaptations; excludes LoopViT pixels, language-corpus claims, learned exits, identities, programmed arithmetic, and repair; and maps every applicable pages 4-6 and 8-10 mechanism to code before science.

All seven arms retain 8,553 parameters and execute the same 300 updates, 9,600 examples, and 3,597 recurrent block calls. The controls are a plain looped Transformer, decoded-feedback recurrence, and independent no-time/step, no-shortcut, no-short-loss, and no-consistency ablations. Evaluation uses 96 fresh held-out-prime cases per width at budgets 1/2/4/8/16, deterministic reload, randomized weights, inference interventions, and hidden/logit curves.

## Calibration

Root `20261194` completes all potential routes and admits formal execution. Canonical inner resources are 6.638 wall / 18.580 process CPU / 17.432 cgroup CPU seconds and 476,811,264-byte peak RSS. Maximum observed update variability produces 1.6318 safety headroom. Conservative formal wall expected/ETA/soft/hard guards are 580.34/946.98/1183.73/1893.96 seconds, with separate process and cgroup guards. The projection scales complete per-arm duration with update count and therefore overestimates formal time; raw route and update samples remain canonical and the bound did not reduce fidelity.

## Viability decision

Reject this exact formulation at root `20261195`. It completes in 83.647 wall / 356.79 process CPU / 402.088 cgroup CPU seconds with 489,570,304-byte peak RSS. Every arm scores 0/96 exact at W16, W24, and W32 at every budget. Primary bit accuracy is constant across budgets at 54.49%, 54.38%, and 51.82%; that diagnostic cannot substitute for exact answers. The plain and decoded controls are also 0% exact. Randomized weights and both inference interventions remain 0% exact.

Exact reload, parameter/example/update/call matching, primary zero-invalid output, resource admission, and nominal deeper-budget retention pass. W16/W24/W32 exactness, all transfer gains, trained-weight dependence, all four trained component effects, and both inference component effects fail. The primary hidden curve becomes almost perfectly collinear while hidden RMS deltas grow to hundreds and logit changes shrink below `5e-4`; this is observed stagnation/collapse in this small endpoint formulation, not a general source-paper conclusion.

The failed conjunctive gate permanently leaves roots `20261196` and `20261197` unopened. Consequently the package reports viability ablation observations but no independent confirmation-level causal verdict for time/step, shortcut, full/short loss, hidden consistency, or fixed budget. Both MLflow runs are `FINISHED`, all checkpoints reload exactly, and the incumbent remains unchanged.
