---
type: Experiment Preregistration
title: EXP-641 Conditional Commutative Swap Routing
description: Fixed 40 percent low-margin routing into operand-swapped frozen inference on the consumed EXP-640 surface.
status: preregistered
tags: [autoresearch, tier6, commutativity, conditional-compute, confidence-routing, runtime]
timestamp: "2026-07-17T14:30:00+05:30"
---

# EXP-641 conditional commutative swap routing

## Question and fixed policy

EXP-640's frozen two-orientation selector scores 291/300 (97.0%) on consumed width-128 cases but its conservative unconditional integration projects to 318.62 seconds. EXP-641 changes only compute allocation: execute the original learned Horner trajectory for every case, rank cases by ascending original minimum absolute trajectory logit margin, and execute the swapped trajectory only for the lowest 40% (120/300). Within routed cases, choose the higher-margin orientation exactly as in EXP-640; ties retain original. Unrouted cases retain original.

The 40% fraction is not selected from EXP-640 outcomes. It is the exact fraction independently supported and confirmed in EXP-625/626 for target-free low-margin routing. Route ties use lowest case index. There is no threshold sweep, alternate confidence score, learned router, or answer-dependent selection. Offline routed oracle coverage reports whether either orientation is correct inside the fixed routed subset; it never controls inference.

## Evidence reuse and execution control

Use only the exact consumed EXP-640 paired cache with SHA-256 `4f295bbbea499d06796d48c0ac1959d2626bb3e44a3413aa3ab0dfe67065a5dd`. Rerun original inference on all 300 cases and swapped inference only on the frozen 120 routed indices. Every original prediction, margin, and invalid flag and every routed swapped counterpart must exactly reproduce the cache. Repeat the complete conditional policy twice on the first 20 rows, routing exactly eight each time, and compare predictions, margins, routes, choices, and validity exactly.

The model/checkpoint and legal tensor inputs are unchanged from EXP-640. Generic commutative swapping and confidence-based conditional compute contain no hand-coded multiplication, division, reduction, repair, lookup, CRT, or expected-answer signal. Exact bigint arithmetic remains offline-only. No fresh, checked, independent-full, official, or private answer is read.

## Gates

Advance only if all gates pass:

- fixed routed oracle coverage is at least 92%;
- fixed conditional selected accuracy is at least 90%;
- selected invalid-trajectory rate is at most 1%;
- all rerun original and routed-swapped predictions, margins, and validity exactly reproduce the EXP-640 cache;
- the two 20-case conditional repeats are exact;
- Candidate 004's complete 36.9-second independent runtime plus measured original-all/routed-swapped work scaled from 300 to 100 Tier-6 cases is at most 285 wall seconds.

Passing authorizes a fresh generated confirmation with this exact 40% policy and a direct candidate-shaped runtime measurement. It is not promotion. Failure closes this fixed conditional compute point without tuning the fraction or confidence score on consumed answers.

## Reused dynamic resource budget

No new calibration cases are needed: EXP-640 already profiles the exact same margin-collecting model path at 12 cases per orientation and measures the exact formal original/swapped arms. Reusing valid evidence is required by the program. Scale original throughput to 300 cases, swapped throughput to 120, and two determinism repeats to 40 original plus 16 swapped orientation-equivalent cases. The conservative profile projection is 734 evaluation wall seconds and 2,744 aggregate CPU seconds. Add 60 seconds for setup, integrity checks, MLflow, and artifacts, giving 795 expected wall seconds and 48 aggregate CPU minutes.

The soft budget is 1,200 wall seconds. The hard runaway ceiling is 1,800 wall seconds with a distinct four-thread upper bound of 120 aggregate CPU minutes. Hashes of the reused calibration and formal-throughput summaries are frozen in config. Full cases and exact conditional execution may not be reduced to fit a shorter cap; completed output reproduction evidence is cached.

## Limitations

This remains consumed evidence and can only authorize a fresh confirmation. Linear scaling from 300 to 100 cases is deliberately conservative but is not the official isolated runtime. The route relies on confidence ranking stability across distributions.
