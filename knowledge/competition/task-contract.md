---
type: Competition Specification
title: SAIR Modular Arithmetic Task Contract
description: Exact inputs, outputs, interface, artifact contract, and publication target.
status: verified
tags: [competition, interface, submission]
timestamp: "2026-07-11T23:15:00+05:30"
---

# Task contract

For decimal strings `a`, `b`, and prime `p`, the trained model must emit digits representing:

```text
(a * b) mod p
```

Operands may reach 4096 bits and the modulus may reach 2048 bits. The prose rules describe this as roughly 1,233 and 617 decimal digits respectively; the pinned public benchmark contains some 4096-bit operands with 1,234 digits, so bit widths are the safer source of truth.

## Model interface

A submission entry class subclasses `ModularMultiplicationModel` and exposes:

1. `load(model_dir)` once before evaluation.
2. Independent `preprocess_a`, `preprocess_b`, and `preprocess_p` hooks.
3. `predict_digits(a_enc, b_enc, p_enc) -> list[int]`.
4. Optionally, `predict_digits_batch` and `max_batch_size`.

Digits are most-significant-first. `output_base` is an integer from 2 through `2^32`, or the string `"p"`. The harness—not contestant code—decodes the digit list. On scored tiers, a decoded value greater than or equal to `p` is malformed.

## Submission artifact

The ranked artifact is a single **public Hugging Face repository**, identified by `repo_id` plus an immutable 40-character commit hash. It must contain:

- `manifest.json` with `entry_class`, `output_base`, `model_description`, `training_description`, and optional `framework`;
- `model.py` defining the entry class;
- trained weights and only the assets needed at inference.

This private GitHub repository is a development workspace, not the ranked artifact. Release candidates are staged separately under [`submission/`](../../submission/).

## Source of truth

- Local pinned implementation: [`base_model.py`](../../sources/github/modular-arithmetic-challenge/src/modchallenge/interface/base_model.py) and [`submission_schema.py`](../../sources/github/modular-arithmetic-challenge/src/modchallenge/interface/submission_schema.py).
- Captured rules: [`overview.md`](../../sources/sair-competition/overview.md) and [`evaluation-setup.md`](../../sources/sair-competition/evaluation-setup.md).
- Live pages verified in managed Chromium on 2026-07-11: [overview](https://competition.sair.foundation/competitions/modular-arithmetic-challenge/overview) and [evaluation setup](https://competition.sair.foundation/competitions/modular-arithmetic-challenge/evaluation-setup).

# Citations

[1] [Official SAIR competition repository](https://github.com/SAIRcompetition/modular-arithmetic-challenge), pinned locally at `82510bba00a1126649bd76dd1a451f14d0b3eb60`.
