---
type: Experiment Preregistration
title: EXP696 C7 QCM-48 Factorial
description: Frozen four-arm rank-4 quotient/carry by persistent-token composition on a Candidate-derived h48 core.
status: completed-non-comparative
tags: [autoresearch, tournament, quotient, carry, persistent-memory, factorial]
timestamp: "2026-08-01T00:00:00+05:30"
---

# EXP696 C7 QCM-48 factorial

## Frozen mechanism

Every arm owns the same Candidate-derived h48 one-cycle core, learned three-class quotient and four-class signed-carry heads, four case-local persistent 16-D tokens, two attention reads, shared GRU token write, rank-4 codec factors, three bounded output projections, auxiliary losses, optimizer, scheduler, examples, RNG, updates, and evaluation calls. Only `qc_access` and `token_access` differ. `neither=(0,0)`, `qc_only=(1,0)`, `token_only=(0,1)`, and `both=(1,1)`.

For each position, learned quotient and carry posteriors form `qc`. The token attention read forms `t`. With rank four, `q=tanh(Dq(qc))`, `m=tanh(Dt(t))`, and the pre-recurrence message is `Mq=tanh(Uq(q))`, `Mt=tanh(Ut(m))`, plus the explicit cross-term `Mx=tanh(Ux(q*m))`. The two masks expose `Mq`, `Mt`, and their product exposes `Mx`; all terms are always computed. The resulting 96 channels are split into bounded scale and bias for an h48 FiLM/residual update. All output projections start at exact zero. Tokens reset at every case/call and never cross a batch boundary.

This is a faithful composition of C1 source runner `aeaed1af...16a8`/config `4fa2e75b...8f7f` and C3 source runner `cedcabc2...6fe`/config `79bca271...9f4c`. C1 attention/GRU persistence is retained but its direct h48 projection is replaced by the common rank-4 codec. C3 learned posterior conditioning and offline auxiliary targets are retained but its direct embedding-FiLM is replaced by the same codec. No C5 routing or C6 cycle coefficient is present. Native PyTorch supplies deterministic SDPA, GRUCell, convolution, and tensor fusion; no maintained external library materially improves this tiny dense codec.

Exact quotient/carry labels, modular targets, teacher traces, and correctness checks exist only in offline training/evaluation. Deployment consumes legal input features, current hard learned state, learned posteriors, case-local tokens, and learned parameters. It contains no exact quotient/carry computation, solver, verifier, answer selection/repair, programmed reduction, Barrett/Montgomery, CRT, checked/private tuning, or operand-swap route.

## Rung, roots, and gates

Root `20261368` runs four calibration updates per arm, representative evaluation for every arm, then one warmup and three cold complete 1,100-case `both` routes (two more when max/median exceeds 1.10). Admission requires deterministic outputs, `2*max<=285s`, RSS at most 6.4 GiB, exact reload, equal work, and no fidelity reduction. Measured throughput determines expected, 1.25x soft, and 1.75x hard budgets.

Only admission opens root `20261369`: exactly 2,400 updates per arm, curriculum `[64,128,256,256]`, horizon 8, complete 3,000-case W128/W256 transitions, common paired 300-case W128 rollout, and sequential common W256 blocks `[200,300,450,600]`. The frozen interaction is `both - qc_only - token_only + neither` on the paired W256 correctness vectors. A combined claim requires positive interaction and a positive lower endpoint under the 99% time-uniform Holm-adjusted sequence, plus common gates: neither W128 transition >=99%, both W256 transition >=99.8%, both W256 rollout >=50%, invalid <=1%, both-minus-neither >=5 points, and W128 loss <=5 points.

Causal tests on trained `both` are quotient/carry zero, cross-case permutation, and seeded randomization; token reset, freeze, cross-case permutation, and seeded randomization; and explicit cross-term zero. At least half the both-versus-neither gain must erase for each mechanism family, and cross-term zero must eliminate positive interaction support. Roots `20261370-71` open sequentially only when complete `both` W256 is >=85% and every common, causal, confidence, and interaction gate passes. Roots `20261372-76` open sequentially only after both confirmations; each requires >=90% and the aggregate requires >=93%.

Any resource park is non-scientific. An unhealthy neither arm makes completed science non-comparative. Healthy complete evidence without positive interaction rejects only the combined mechanism claim. C5 remains unresolved, no entrant cancels another, and the seventh-member adversarial chair remains mandatory.

## Observed terminal result

The exact final-byte focused suite passed 33 tests, the rootless smoke exercised all four arms and all eight interventions, and the sole package oracle passed validation (232 concepts and 1,100 records) plus all 1,042 tests in 522.75 seconds. Root 20261368 admitted the route: three deterministic complete 1,100-case `both` routes took 95.0148, 90.3354, and 94.9215 seconds; guarded runtime was 190.0296 seconds, peak RSS was 1,165,512,704 bytes, all four checkpoints reloaded exactly, and equal work held. The measured full-rung budget was 13,302 seconds expected, 16,628 soft, and 23,279 hard.

Root 20261369 then completed 2,400 updates for each arm (9,600 total), 19,200 student/teacher/posterior/token-read/token-write/codec/fusion/auxiliary/exact-loss calls per arm, every 3,000-case transition denominator, the 300-case W128 rollout, and all W256 blocks through 600. All arms shared initial-state hash `417110c4...d5c3`, 34,836 parameters, data/RNG/optimizer/scheduler/losses, and exact call ledgers. The full rung used 10,597.65 internal wall seconds, 43,408.59 process CPU seconds, 42,462.16 cgroup CPU seconds, and 1,676,750,848 bytes peak RSS; external wall was 3:02:48.

The neither health gate failed with 0/3,000 exact W128 transitions, so the result is non-comparative. Every arm was 0/3,000 exact at W128 and W256 and 0/300 W128 plus 0/600 W256 rollout. The W256 both-versus-neither result was 0 wins, 0 losses, and 600 ties; the 99% time-uniform Holm-adjusted gain interval was [-0.13162, 0.13162]. The factorial interaction `both - qc_only - token_only + neither` was exactly 0 with interval [-0.26323, 0.26323]. Both W256 invalid-any-step rate was 96.5%. All quotient/carry, token, and cross-term interventions had zero exact-gain erasure because every arm and intervention was zero-exact.

Learned auxiliary state was noncollapsed but insufficient: on `both`, quotient accuracy was 51.67%/53.77% at W128/W256 with 10-bin ECE 3.89%/2.20%; signed-carry accuracy was 56.91%/56.29% with ECE 13.86%/14.55%. W256 quotient/carry entropy was 0.4645/1.0953, final token norm was 1.3905, and the rank-4 cross message norm was 0.00224. These diagnostics do not rescue the failed exact-answer health gate.

Roots 20261370-76 remain unopened. No combined-mechanism rejection is claimed because the matched control was unhealthy; the exact C7 rung is closed as non-comparative. C5 remains unresolved, no entrant is cancelled, and the final seventh-member adversarial chair remains mandatory.
