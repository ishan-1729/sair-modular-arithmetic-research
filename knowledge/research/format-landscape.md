---
type: Standards Review
title: Knowledge and Research Packaging Format Landscape
description: Playwright-verified comparison of OKF, RO-Crate, Croissant, Data Package, Data Cards, Model Cards, and CCDS.
status: decided
tags: [OKF, metadata, reproducibility, standards]
timestamp: "2026-07-11T23:15:00+05:30"
---

# Format landscape

All linked pages below were visibly rendered and inspected with repository-local Playwright 1.61.1 and managed Chromium 149 in a fresh, read-only context on 2026-07-11.

| Format | Best fit | Decision |
|---|---|---|
| OKF v0.1 Draft | Navigable human/agent knowledge, links, progressive disclosure, change log | Adopt strictly for `knowledge/` only. |
| RO-Crate 1.3 Recommendation | Aggregate research object, files, software, people, and provenance in JSON-LD | Adopt a lightweight root descriptor. |
| Croissant 1.1 | Machine-readable ML dataset files, record sets, fields, provenance, and loading semantics | Adopt for the public benchmark. |
| Data Cards | Human-facing dataset motivations, provenance, uses, limitations, and maintenance | Embed the relevant sections in the benchmark concept. |
| Model Cards | Intended use, architecture, evaluation, limitations, and release context | Use for each promoted POC/model. |
| Cookiecutter Data Science v2 | Familiar raw/interim/processed, model, report, and source layout | Adapt its directory vocabulary. |
| Data Package v2 | General resource descriptor for arbitrary data files | Evaluated but not adopted; Croissant is more ML-specific and a second canonical descriptor would duplicate metadata. |

## Why OKF is not the whole repository format

OKF v0.1 intentionally defines a minimal directory of Markdown concepts with YAML frontmatter and requires only `type`. It explicitly does not replace domain schemas or prescribe code, storage, query infrastructure, datasets, checkpoints, or evaluation records. Forcing binary weights, JSONL rows, or executable code into Markdown would reduce fidelity and break existing tooling.

The project therefore uses an **OKF knowledge plane** linked to **native artifact planes**. The formal OKF bundle is [`knowledge/`](../), while RO-Crate links the aggregate and Croissant describes dataset structure.

## Verified primary sources

- [Google Cloud introduction to OKF](https://cloud.google.com/blog/products/data-analytics/how-the-open-knowledge-format-can-improve-data-sharing) — OKF v0.1 is Markdown plus YAML frontmatter, designed as a minimally opinionated format rather than a platform.
- [OKF v0.1 specification](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md) — `type` is required for concept documents; `index.md` and `log.md` are reserved; links create an untyped directed graph.
- [RO-Crate 1.3](https://www.researchobject.org/ro-crate/specification/1.3/introduction.html) — published 2026-06-22 as a Recommendation; uses `ro-crate-metadata.json` and JSON-LD to aggregate and describe research data, software, and context.
- [Croissant 1.1](https://docs.mlcommons.org/croissant/docs/croissant-spec-1.1.html) — published 2026-01-29; machine-readable ML dataset metadata for discoverability, portability, reproducibility, and responsible AI.
- [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/#directory-structure) — a logical, flexible, standardized structure with external/raw/interim/processed data and separate models, reports, references, and source.
- [Data Package v2](https://datapackage.org/standard/data-package/) — a required `resources` array plus general package metadata in `datapackage.json`.
- [Data Cards Playbook](https://sites.research.google/datacardsplaybook/) — structured dataset summaries across lifecycle stakeholders.
- [Model Cards paper](https://research.google/pubs/model-cards-for-model-reporting/) — short model documentation covering intended use, evaluation conditions, and performance characteristics.

## Browser evidence note

DuckDuckGo discovery presented a CAPTCHA and was not bypassed. Strong sources were instead opened directly from known authoritative project sites. Read-only request blocking caused only expected analytics POST failures; primary content remained accessible.
