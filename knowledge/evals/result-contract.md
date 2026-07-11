---
type: Result Schema Guide
title: Durable Evaluation Result Contract
description: Human interpretation of the JSON schemas and immutable run-record requirements.
status: active
tags: [metrics, schema, provenance]
timestamp: "2026-07-11T23:15:00+05:30"
---

# Durable result contract

Machine-readable schemas live under [`evals/schemas/`](../../evals/schemas/) and [`experiments/schemas/`](../../experiments/schemas/). A durable result must identify:

- run ID, status, timestamp, and purpose;
- source and upstream challenge commits;
- full configuration and random seed;
- train/validation/test prime split digest;
- Python, PyTorch, device, OS, and CPU thread count;
- parameter count, training steps, and wall time;
- exact transition and autonomous-rollout metrics with denominators;
- determinism and perturbed-weight controls;
- checkpoint path and SHA-256;
- limitations and interpretation.

Metrics without denominators, split identity, or checkpoint hash are exploratory and must not be quoted as durable evidence.
