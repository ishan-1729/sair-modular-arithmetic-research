---
type: Experiment Analysis
title: EXP-100 Width-Scaling Diagnosis
description: Error, width, invalid-state, and learning-curve analysis for the shared-context Horner smoke test.
status: completed
tags: [autoresearch, analysis, width-scaling, rollout, invalid-state]
timestamp: "2026-07-12T02:30:00+05:30"
---

# EXP-100 width-scaling diagnosis

## Observations

The shared-context surface is valid: the width-9 control reached 91.77% heldout transition accuracy and 56.8% exact rollout after 400 steps, well above its 12.4% and 3.9% untrained controls.

Width 16 also learned materially: heldout transition accuracy rose from 6.4% untrained to 64.31%, batch exactness rose from 6% at step 1 to 68% at step 400, and the final-100-step batch-exact delta was +5.47 percentage points. Exact rollout, however, reached only 26/5,000 (0.52%), and 54.82% of rollouts entered an invalid residue.

Transition accuracy decreases smoothly with modulus width, from roughly 75% at 9 bits to 61.76% at 16 bits. Rollout errors concentrate at steps 2-5, after the trivial leading-zero transition and when the first nonzero prefix states appear. These observations contradict a total failure to learn but do not meet the preregistered continuation gate.

## Interpretation

The dominant bottleneck is accumulated state validity and early-prefix transition reliability, not a dead optimization surface. The width-9 filters already encode a useful learned transition and share tensor shapes with most of the width-16 network. This supports a causal next test: transfer the learned shared filters and overlapping low-bit parameters into width 16, while keeping the width-16 data, steps, seed, evaluation cases, and optimizer matched to the random-initialized arm.

## Decision

Do not continue EXP-100 directly and do not consult checked Tier-3 answers. Start EXP-105 as a learned-initialization A/B. Promote only if transfer improves heldout transition materially and lifts rollout above the continuation threshold without increasing invalid-state failures.

Canonical evidence: [`metrics.json`](../../../evals/results/exp-100-width16-shared-context-smoke/metrics.json) and [`run.json`](../../../experiments/runs/exp-100-width16-shared-context-smoke/run.json).
