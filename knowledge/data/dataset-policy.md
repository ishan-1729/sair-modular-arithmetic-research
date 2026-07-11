---
type: Data Policy
title: Dataset Lifecycle and Leakage Policy
description: Rules for immutable sources, synthetic generation, processing, retention, and release exclusion.
status: active
tags: [data-governance, leakage, reproducibility]
timestamp: "2026-07-11T23:15:00+05:30"
---

# Dataset lifecycle policy

| Zone | Path | Contract |
|---|---|---|
| External | `data/external/` | Immutable third-party snapshots with source commit, license, and checksums. |
| Generated | `data/generated/` | Re-creatable synthetic examples; normally ignored because seed/config are authoritative. |
| Interim | `data/interim/` | Disposable transforms and caches; never cited as a result source. |
| Processed | `data/processed/` | Canonical run inputs derived from a recorded generator/config; normally regenerated. |

Durable results reference a generator version and seed rather than committing large synthetic training sets. Small fixtures required by tests may be committed under `tests/fixtures/` with an explicit purpose.

## Leakage controls

- Public benchmark is evaluation-only.
- Split prime identities before generating examples.
- Do not infer hidden data from leaderboard feedback through excessive submissions.
- Never include `expected` benchmark fields in `submission/`.
- Hash all externally sourced data and checkpoints.
- Record any sample reuse schedule; synthetic does not mean provenance-free.
