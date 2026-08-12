---
type: experiment-preregistration
title: EXP705 Candidate006 tensor-native runtime repair
description: Prediction-preserving cached-feature runtime probe for Candidate006's Tier-7 rollout.
status: frozen-before-execution
date: 2026-08-12
timestamp: "2026-08-12T07:55:00Z"
tags: [autoresearch, candidate006, tier7, tensor-native, runtime]
---

# EXP705 Candidate006 tensor-native runtime repair

## Decision card

- **Hypothesis:** Reusing invariant W256 feature planes and carrying learned state as a boolean tensor will preserve Candidate006 predictions exactly while bringing its official sandbox inference below 300 seconds.
- **Measured failure:** EXP704 passed every five-seed scientific gate but its frozen artifact reached only Tier 6 at 273.6 inference seconds in the pinned 4-CPU sandbox and produced no result before the wrapper's terminal timeout.
- **Baseline identity:** EXP704 Candidate006 model/manifest/weights SHA-256 are `b7c255af7b619e6f689a84e5307ce445432e4285af0cdd9f514d982ebc8c0315`, `dbeb95ed3b085aa2b338ec57041d343ec924ca72f4b3c7fd3094d6cd215a0c91`, and `a7d25ee4eed49c5f625bc8369447be9f51ff1be54766631d8b96ab725860553e`.
- **Changed variable:** Replace only `_tier7_rollout`'s repeated Python feature reconstruction with cached invariant tensors and boolean learned state. Tensor values, network calls, call order, threshold, batching, route, checkpoint, and output policy remain fixed.
- **Smallest falsifier:** Any prediction differs from the EXP704 artifact on all five frozen full sets, or the optimized artifact still fails the 300-second official sandbox limit.
- **Primary oracle:** Exact prediction-hash parity against EXP704 on all 5,500 generated rows, followed by the pinned official seed-1 sandbox runtime.
- **Promotion gate:** Exact parity on every row; static/artifact/determinism gates remain clean; official inference <=285 seconds for promotion and <=300 seconds for validity.
- **Stop rule:** Stop after one parity failure or one terminal official sandbox result. Do not retrain, tune, change routes, or rerun scientific selection.
- **Retention gates:** Candidate005 tensors, EXP701 tensors, three-file shape, all routing boundaries, zero fallback, and all EXP704 scientific evidence remain exact.
- **Cost model:** Local CPU only, no paid compute. Parity replays reuse frozen seeds and should finish within EXP704's measured runtime; the official sandbox has one 300-second hard inference budget.
- **Evidence outputs:** Optimized build receipt, full prediction-parity receipt, sandbox result/failure, environment, metrics, and registry decision.
- **Allowed files:** Candidate006 builder/EXP704 focused test, EXP705 preregistration/config/runner/results, registry/scoreboard/decision log, and `.scratch/exp705*`. Existing `submission/` and EXP704 evidence are immutable.

This is a new runtime experiment after EXP704's preregistered stop, not a reinterpretation of EXP704. It may inherit EXP704 accuracy only after exact full-set prediction parity succeeds.
