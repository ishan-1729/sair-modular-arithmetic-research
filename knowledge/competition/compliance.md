---
type: Compliance Guide
title: Trained-Parameters Boundary
description: Operational interpretation of allowed learning, preprocessing, offline labels, and prohibited inference algorithms.
status: active
tags: [compliance, provenance, anti-cheating]
timestamp: "2026-07-11T23:15:00+05:30"
---

# Trained-parameters boundary

The model must learn the capability. A forward pass that remains an exact solver after its weights are randomized is a hand-coded circuit, not an admissible model.

## Allowed in this project

- Exact arithmetic in offline data generation, labels, tests, and classical teacher baselines.
- Per-argument representation work in a preprocessing hook.
- The exact two-argument reductions `a % p` and `b % p` used by the official neural examples, provided the final product is materially produced by trained parameters.
- Explicit looping or recurrence structure whose transition is learned and whose emitted state depends on trained weights.
- Deterministic tensor operations needed to run a learned architecture.

## Prohibited in a submission inference path

- Direct `(a * b) % p` using Python big integers or symbolic libraries.
- Hand-coded schoolbook multiplication, long division, Barrett or Montgomery reduction, or CRT recombination over the actual inputs—even when expressed as tensor code.
- Lookup tables, input fingerprints, benchmark-answer leakage, or a postprocessor that repairs model output.
- Network access, subprocesses, dynamic execution, undeclared filesystem reads, or cross-argument state leakage.

## POC design rule

The selected POC fixes the outer Horner scan but learns the transition:

```text
state_next = learned_cell(state, current_operand_bit, other_operand, modulus)
```

Ground-truth transition arithmetic is present only in the trainer and evaluator. Inference uses a hard predicted state at the next step. This is a plausible compliant boundary, not a guarantee of organizer approval; record provenance and maintain a perturbation control.

## Packaging rule

Training utilities that contain label arithmetic stay outside `submission/`. Every Python file shipped in a release candidate is statically scanned, even if it is never imported.
