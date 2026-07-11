---
type: Risk Register
title: Project Risk Register
description: Technical, compliance, schedule, infrastructure, and evidence risks with mitigations.
status: active
tags: [risk, governance, schedule]
timestamp: "2026-07-11T23:15:00+05:30"
---

# Risk register

| ID | Risk | Likelihood | Impact | Mitigation / trigger |
|---|---|:---:|:---:|---|
| R-01 | Tiny per-step error compounds into near-zero long-rollout exactness. | High | Critical | Select on full rollouts; use tail-aware loss and shorter promotion gates. |
| R-02 | Model memorizes trained primes or public benchmark. | High | Critical | Prime-disjoint splits, secret-style seeds, public data evaluation-only. |
| R-03 | Explicit recurrence is judged a hand-coded circuit. | Medium | Critical | Learned transition, weight perturbation, provenance, early organizer clarification. |
| R-04 | CPU-only sandbox makes a high-width rollout too slow. | High | Critical | CPU-first profiling, vectorized batch rollout, limb experiment, 25% time headroom. |
| R-05 | Rules or sandbox change before deadline. | Medium | High | Pin every run; monitor live pages at phase gates; isolate adapter code. |
| R-06 | Public model-card claims are overfit or incomplete. | High | High | Treat as hypothesis only; require independent hidden-seed evidence. |
| R-07 | Data or metric provenance drifts across scripts and docs. | Medium | High | Generate manifests from run records; validate hashes in CI. |
| R-08 | Deadline pressure causes an unreproducible release. | Medium | Critical | Freeze Aug 8–9; keep submission packaging separate from training workspace. |
| R-09 | Nested official repository is accidentally vendored or loses commit identity. | Low | Medium | Track as a pinned Git submodule and validate its commit. |
| R-10 | Secrets enter Git history or Hugging Face artifact. | Low | Critical | Ignore environment files, scan staged release, use clean-room `submission/`. |
