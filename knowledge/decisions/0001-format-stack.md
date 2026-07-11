---
type: Architecture Decision Record
title: "ADR-0001: Use a layered research format stack"
description: Keep strict OKF in the knowledge bundle and use native schemas for data, models, runs, and provenance.
status: accepted
tags: [ADR, OKF, RO-Crate, Croissant]
timestamp: "2026-07-11T23:15:00+05:30"
---

# ADR-0001: Use a layered research format stack

## Context

OKF is designed for portable Markdown knowledge concepts. This project also contains executable Python, YAML configs, JSONL benchmark data, JSON metrics, schemas, and binary PyTorch checkpoints. OKF deliberately does not replace those formats.

## Decision

- `knowledge/` is a conformant OKF v0.1 bundle.
- The repository layout adapts Cookiecutter Data Science separation.
- `ro-crate-metadata.json` describes the aggregate research object and selected provenance.
- `data/croissant.json` is the canonical machine-readable public-dataset descriptor.
- Human dataset and model documentation follows relevant Data Card and Model Card sections as OKF concepts.
- Evaluation and experiment records use validated JSON/YAML schemas.
- Data Package was reviewed but is not adopted because it would duplicate Croissant metadata.

## Consequences

Humans and agents get progressive disclosure and links without sacrificing native tooling. The cost is a validator and several metadata surfaces; generated fields and hashes must stay consistent.
