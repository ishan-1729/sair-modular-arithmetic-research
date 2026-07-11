---
type: Evaluation Strategy
title: Internal Evaluation Strategy
description: Stricter-than-leaderboard evaluation across exactness, generalization, systems, and compliance.
status: active
tags: [evaluation, exactness, robustness]
timestamp: "2026-07-11T23:15:00+05:30"
---

# Internal evaluation strategy

## Metric layers

1. **Transition**: bit accuracy, exact-state accuracy, worst-bit confidence, and first-error position.
2. **Autonomous rollout**: exact final answer, exact prefix length, and error taxonomy.
3. **Competition geometry**: per-tier exact accuracy, equal-weight overall accuracy, and highest tier at or above 90%.
4. **Generalization**: held-out primes, unseen seeds, modulus-width interpolation/extrapolation, operand-length shifts, and boundary-heavy states.
5. **Systems**: CPU wall time, peak memory, artifact bytes, batch scaling, load time, and determinism.
6. **Compliance**: static scan, source review, randomized/reinitialized weight control, and release-content audit.

Exact final-answer accuracy is the promotion metric. Bit accuracy is diagnostic only; a one-bit miss is a fully wrong competition answer.

## Evaluation sets

- `validation`: generated from held-out primes; used for checkpoint selection.
- `test`: untouched prime identities and seed; used once per promoted experiment.
- `public`: official fixed benchmark; used for release regression, not tuning.
- `stress`: reduction-boundary, length-shift, formatting, and runtime cases.

## Promotion thresholds

A small POC is successful when it proves the pipeline end-to-end, beats trivial controls on held-out-prime autonomous rollout, is deterministic, and collapses under weight randomization. It need not meet competition tiers.

A release candidate additionally needs complete official-interface and sandbox coverage, honest documented behavior outside its trained range, and comfortable runtime headroom.

Candidate 001 satisfies the mechanical interface, sandbox, determinism, provenance, and timing gates with more than 99% inference-budget headroom. Its explicit fallback above modulus 257 is compliant and honest, but it means the research-width gate remains open even though the artifact itself is publishable.
