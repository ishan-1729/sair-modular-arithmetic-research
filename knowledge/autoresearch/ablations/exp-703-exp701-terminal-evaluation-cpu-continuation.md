---
type: Experiment
title: EXP703 EXP701 terminal evaluation CPU continuation
description: Deterministic CPU successor that closes the two missing EXP701 full-denominator evaluations without retraining.
run_id: exp-703-exp701-terminal-evaluation-cpu-continuation
status: completed
tags: [exp703, exp701, terminal-evaluation, cpu, deterministic, successor]
timestamp: "2026-08-11T21:58:44.884991+00:00"
---

# EXP703: EXP701 terminal evaluation CPU continuation

EXP703 authenticated the frozen terminal-incomplete EXP701 archive and executed
only the two missing `straight_through_restarts` evaluations. It did not train,
modify, or resume any weights. Because the original EXP702 successor was
CUDA-preregistered, this CPU execution has a separate run and execution
identity.

## Frozen inputs

- EXP701 archive SHA-256: `dcd500925e9c57ce7daf4af78e068747c91fc71c84df04a795d1e50e456a42fc`
- final checkpoint SHA-256: `d562e6b11c771dffa0ff4cfd191798efeb2cfb1cae87369fc90134099f5e23ab`
- EXP701 scientific identity: `4652143d6ee35be631d8269d942900d486551fa235132f6cd502ad69e28721a9`
- EXP703 CPU execution identity: `479b5a0c9b76f1d6254962ef9c889c9ebfbb98a6e73b38d20a0e6a19ddf96fc1`
- runtime: PyTorch 2.5.1 CPU, deterministic algorithms, 10 intra-op threads,
  one inter-op thread, Intel Core Ultra 7 155H under WSL2.

## Results

| Width | Transition exact | Rollout exact | Invalid rollout rate |
|---:|---:|---:|---:|
| 128 | 2996/3000 (99.8667%) | 263/300 (87.6667%) | 0/300 (0%) |
| 256 | 2994/3000 (99.8%) | 315/600 (52.5%) | 4/600 (0.6667%) |

Across all four factorial arms, `teacher_cosine` remains the winner with
60.1667% W256 rollout exact accuracy. This is below the preregistered 85%
advancement gate, so the terminal decision is
`complete-negative-full-denominator`. No confirmation or independent roots
were triggered.

The two CPU evaluations consumed 4,567.72 aggregate phase wall seconds and
40,338.75 process CPU seconds, peaking at 1,211,617,280 bytes RSS. Each width
was published independently with a hash-bound manifest, allowing the completed
W128 phase to survive interruption before W256.

## Publication incident

Both phase results completed successfully, but the original unattended runner
hit a publication-only `KeyError` because the CPU config omitted EXP702's
descriptive predecessor relationship field. The launcher retried without
recomputing either completed phase and then exhausted its local retry budget.
A publication-only repair authenticated the predecessor and both completed
phase manifests, validated their hashes and denominators, supplied the frozen
relationship value in memory, and wrote the terminal summary. No scientific
computation was repeated or changed.

## Evidence

- Run record: `experiments/runs/exp-703-exp701-terminal-evaluation-cpu-continuation/run.json`
- Metrics: `evals/results/exp-703-exp701-terminal-evaluation-cpu-continuation/metrics.json`
- Raw terminal summary and phase evidence:
  `experiments/runs/exp-703-exp701-terminal-evaluation-cpu-continuation/raw/`

EXP703 must not be described as completion of EXP701 or as the CUDA EXP702
result. It is a distinct CPU successor that closes the missing full-denominator
evidence and yields a negative advancement decision.
