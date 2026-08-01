---
type: Experiment
title: EXP694 C5 routed antithetic quorum
description: Frozen C5 tournament implementation and terminal consumed-root lifecycle park.
status: parked
tags: [autoresearch, tournament, conditional-computation, antithetic, routing]
timestamp: "2026-08-01T20:42:50+05:30"
---

# EXP694 C5 routed antithetic quorum

## Frozen mechanism

EXP694 implements a Candidate005-derived hidden-56, one-cycle Horner student. Every step computes a base learned proposal. Consecutive complete groups of four form the eligible set, and exactly one example in each group receives two additional learned proposals formed with the same learned delta at opposite signs. Learned proposal energy plus learned quorum agreement produces one deterministic next learned state. The primary selects the one-per-four route with a learned survival-Q score. Its matched control performs the identical sparse branch work and selection but chooses one item per group with deterministic target-independent generic hashing. An incomplete final group is explicitly ineligible and base-only, preventing the EXP680 batch-global top-k/load-collapse failure mode.

Native PyTorch was retained because its deterministic `index_select`, `index_copy`, convolutions, and autograd directly implement the fixed CPU route; an extra library would not materially improve correctness or performance. Exact arithmetic and teachers occur only in offline training/evaluation. Deployed routing and selection use permitted inputs, current learned state, and proposal features; they do not receive answers, labels, a verifier, or a hand-coded reduction routine.

## Verification evidence

The focused suite passed 37 tests after two bounded correction iterations. The rootless all-mode smoke completed both arms, reload, paired evaluation, deployed routing, serialization, and all preregistered route-randomization, energy-erasure, energy-permutation, antithetic-sign-collapse, and branch-output-shuffle interventions in 36.3 seconds. Git-LFS integrity, both checkpoint hashes, and the clean nested challenge commit `82510bba00a1126649bd76dd1a451f14d0b3eb60` passed.

The exactly-once package oracle passed before root admission: project validation reported 228 concepts and 1,100 benchmark records, and all 949 tests passed in 459.94 seconds. End-to-end oracle wall was 502.49 seconds, aggregate process CPU was 790.85 seconds, and peak RSS was 938,452 KiB.

## Observed terminal result

Calibration root 20261350 was opened once. Both matched arms completed four calibration updates from the same initial-state hash with 54,551 parameters per arm. Each arm executed 32 student calls, 32 teacher calls, 512 eligible example-steps, exactly 128 routed example-steps, 768 proposal calls, and 32 exact-loss calls; both checkpoints reloaded bit-exactly.

The first W128 transition evaluator call then raised `AttributeError: ArmView has no attribute width`. No transition or rollout denominator and no cold complete 1,100-case deployed route had executed. Consequently candidate guarded wall, the 6.4-GiB admission gate, exact-answer metrics, control health, comparative gain, and causal effects are all unavailable.

Root 20261350 is consumed. Repairing executable bytes and reusing that root would violate the frozen root contract, and roots 20261351-58 are reserved for later stages rather than replacement calibration. EXP694 is therefore durably parked for lifecycle feasibility, not resource rejection or scientific inferiority. The C5 hypothesis remains unresolved, and this result cannot cancel or weaken C6-C7.

Canonical evidence is in `experiments/runs/exp-694-routed-antithetic-quorum/run.json` and `evals/results/exp-694-routed-antithetic-quorum/metrics.json`; raw root-open and progress receipts remain under `.scratch/exp-694-routed-antithetic-quorum/`.
