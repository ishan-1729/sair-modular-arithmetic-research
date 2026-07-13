---
type: Experiment Preregistration
title: EXP-626 Tier-6 Minimum-Margin Routing Confirmation
description: Untouched confirmation of the frozen target-free confidence score selected from EXP-625 secondary diagnostics.
status: completed
tags: [autoresearch, tier6, conditional-compute, confidence, confirmation]
timestamp: "2026-07-13T09:40:00+05:30"
---

# EXP-626 Tier-6 minimum-margin routing confirmation

## Frozen hypothesis

Rank greedy trajectories by the negative of their single lowest absolute bit logit across all 128 steps. Route exactly the 80/200 lowest-margin cases through the unchanged width-4 learned search. The score, route fraction, checkpoint, branch operator, likelihood score, data generator, and gates are frozen before opening seed `20260822`.

EXP-625 selected this score only as a secondary diagnostic: it matched the 92.5% oracle allocation while the preregistered mean-bit-NLL score failed at 91%. EXP-626 is the one allowed untouched confirmation and cannot be rescued by another secondary score.

## Evaluation and gates

Run 200 paired greedy/full-width-4 cases and two exact 20-case repeats. Record unrestricted union, oracle-best 40% allocation, and minimum-margin allocation coverage. Advance to trained reranker research only if all three reach at least 92%, selected beam invalid rate is at most 1%, confidence/predictions/full beam states are deterministic, and the unchanged conditional Tier-6 projection remains at most 240 wall seconds.

No output selector is promoted by this run. Oracle allocation and containment remain evaluation-only. If the frozen minimum-margin route fails, close confidence routing without training.

## Resource budget

The label-free candidate-shaped profile remains valid because checkpoint, greedy path, beam path, route fraction, threads, and workload are unchanged: greedy-100 plus beam-40 took 135.29 Tier-6 wall seconds and projects to 187.19 seconds integrated.

Expected diagnostic total is 980 wall seconds, soft budget 1,300 wall seconds, and hard runaway ceiling 1,850 wall seconds. The conservative four-thread ceiling is 123.34 aggregate CPU minutes. Wall and aggregate CPU are measured separately; full fidelity and cached raw evidence are mandatory.

## Compliance and limitations

Routing depends only on learned logit magnitude. It uses no target, expected answer, benchmark identity, modulus filter, or exact arithmetic. Exact bigint operations are offline truth and evaluation only. Generated confirmation is not checked-public or official-private evidence, and projected runtime is not the isolated sandbox gate.

## Result and decision

Greedy scores 173/200 (86.5%), width-4 oracle containment and unrestricted union both score 185/200 (92.5%), and the frozen minimum-margin 40% allocation also scores 185/200. It routes all 27 greedy failures, captures all twelve beam-recoverable failures, and incurs zero greedy losses. Confidence, predictions, complete beam states, and routing counts are exactly deterministic; selected beam invalid rate is zero.

All gates pass. The full diagnostic completes in 865.75 wall seconds and 3,251.84 aggregate CPU seconds, while the candidate-shaped conditional projection remains 135.29 Tier-6 / 187.19 integrated wall seconds. Advance to a compact trained reranker on newly generated training identities. This result authorizes research only: 92.5% remains an oracle containment ceiling until trained parameters select the candidate state without expected-answer access.
