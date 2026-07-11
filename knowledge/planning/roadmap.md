---
type: Research Roadmap
title: Competition Execution Roadmap
description: Dated work plan from the initial POC to a frozen public Hugging Face candidate.
status: active
tags: [plan, milestones, deadline]
timestamp: "2026-07-11T23:15:00+05:30"
---

# Execution roadmap

The live competition page was observed as active on 2026-07-11 with a deadline of **2026-08-12 23:59 AoE**. Internal freeze dates intentionally precede it.

| Phase | Target dates | Exit criteria |
|---|---|---|
| 0. Foundation and POC | Jul 11–13 | Private repo, metadata validation, pinned sources, reproducible learned-transition smoke run. |
| 1. Small-width truth | Jul 14–18 | Prime-disjoint 8–32-bit transition/rollout matrix; trajectory, sampling, and loss ablations; negative results retained. |
| 2. Architecture selection | Jul 19–25 | Binary TCN versus ConvGRU; first limb experiment; one family wins on exact rollout per CPU-second. |
| 3. Width scaling | Jul 26–Aug 2 | Curriculum to at least the largest feasible widths; secret-style generator evaluations; latency projection under constrained CPU. |
| 4. Integration and hardening | Aug 3–7 | Official interface adapter, batching, deterministic load, static check, sandbox run, artifact provenance, perturbation control. |
| 5. Candidate freeze | Aug 8–9 | Immutable candidate hash, full public benchmark, independently seeded diagnostics, CPU timing, model/data cards. |
| 6. Release and submit | Aug 10–11 | Public Hugging Face repo, commit lock, final accessibility check, submission receipt. |

## Decision cadence

- Promote by exact held-out-prime rollout accuracy and CPU time, not training loss.
- Revisit architecture only at phase gates; avoid parallel model sprawl.
- Preserve one simple control alongside every more complex model.
- Re-read live rules before Phase 3, candidate freeze, and publication.

## Definition of competition-ready

The current POC is not competition-ready. A candidate becomes ready only when it implements the official interface, covers all evaluation widths without an out-of-regime fallback dominating, passes the published sandbox, and has a complete immutable release record.
