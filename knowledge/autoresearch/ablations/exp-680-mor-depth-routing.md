---
type: Experiment Preregistration
title: EXP680 Mixture-of-Recursions Depth Routing
description: Frozen bounded-faithful ST-M009.b hierarchical expert-choice depth-routing comparison around Candidate005.
experiment: EXP680
status: parked-calibration-noncollapsed-load
tags: [autoresearch, exp680, mixture-of-recursions, routing, candidate005]
timestamp: "2026-07-30T04:00:00+05:30"
---

# EXP680 Mixture-of-Recursions depth routing

## Frozen transfer

MoR routes Transformer tokens. Candidate005 instead uses dilated convolutions that couple all spatial positions within an arithmetic instance, so the smallest sparse unit that can actually avoid recurrent calls is one complete instance within a batch. EXP680 is therefore bounded-faithful, not a literal Transformer reproduction. The expert-choice primary re-evaluates only surviving instances after depth boundaries; token-choice assigns the complete path once from the initial hidden state. Training uses native CPU `torch.topk`, while evaluation uses learned EMA score thresholds so the released route is sample-local, deterministic, and independent of batch composition.

Frozen depth sets are W64 `[7,9,12]`, W128 `[8,11,14]`, and W256 `[9,13,16]`. With three depths, expert-choice admits all instances to the first segment, `ceil(2B/3)` to the second, and `ceil(B/3)` to the third. Its auxiliary load loss is `(mean(sigmoid(g1))-2/3)^2 + (mean_survivor(sigmoid(g2))-1/2)^2`; token-choice uses squared deviation of mean three-way probabilities from uniform `1/3`. Every instance receives the first segment. Routers see only current learned hidden state; labels, answers, correctness, validity, oracle depth, benchmark identity, and target canvases are excluded.

## Matched package

The expert-choice, token-choice, fixed-full, equal-average fixed-depth, and equal-average two-point arms jointly continue their router and Candidate005 core for 400 updates with identical examples, optimizer settings, data order, loss tensors, and root identities. The frozen-source control receives no updates. The two equal-average controls execute 11 W128 calls: fixed depth 11 versus a matched 8/14 histogram. Matched-histogram random routing, route shuffling, router zeroing, randomized routers, and core zeroing are causal interventions. Every router parameter counts as trained artifact state.

Root `20261270` is complete-route non-scientific calibration. It exercises every arm, width, training mode, route, intervention, reload, persistence, and candidate timing path while separately recording wall, process-tree CPU, cgroup CPU, RSS, route load/calls, variability, and projected formal guards. It admits root `20261271` only if parity, deterministic CPU routing, all-depth load, at least 10% call reduction, no route-wall regression, candidate headroom, memory, artifacts, expected value, and full fidelity all pass.

If admitted, root `20261271` trains and caches all arms, then evaluates the complete 300-case W128 expert primary first. W128 requires at least 95% exact rollout, 99.8% transition, zero invalids, at most two points loss to frozen/fixed-full, nonzero joint gradients and interventions, all depths, at least 10% call reduction, and no wall regression. Only then may W256 run; it requires at least 10% exact and five points over fixed-full and both equal-average controls. Roots `20261272` and `20261273` must reproduce all conjuncts. No post-result depth, threshold, capacity, loss, or denominator retuning is allowed.

No checked, independent, official/private, candidate, submission, promotion, or completion evaluation is authorized.

## Terminal calibration outcome

Ten focused tests and the one package oracle passed on the frozen bytes; project validation reported 206 concepts / 1,100 records and pytest passed 673/673 in 479.88 seconds (476.49 outer wall, 610.23 process CPU seconds, 931,884 KiB RSS). Root `20261270` then completed every arm, width, training mode, intervention, reload, persistence, and candidate timing route in 129.921 wall / 369.570 process-tree CPU / 383.924 cgroup CPU seconds with 1,961,058,304-byte peak RSS.

Fixed-full parity, deterministic CPU routing, 10.714% call reduction, route wall, 68.870-second guarded candidate projection, artifact, memory, expected-value, and full-fidelity gates passed. The expert W128 diagnostic routed 3/12 instances to depth 8, zero to depth 11, and 9/12 to depth 14. Because all declared depths were required, the noncollapsed-load gate failed. Root `20261271` and confirmations `20261272/73` remain unopened. This is a calibration topology park, not a W128/W256 scientific verdict or a general rejection of MoR/M009.b.
