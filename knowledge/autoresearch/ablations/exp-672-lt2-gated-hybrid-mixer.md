---
type: Ablation Study
title: EXP-672 LT2 gated linear/sparse hybrid mixer
description: Source-authenticated bounded ST-M025 transfer parked at the inherited package-verification boundary before calibration.
status: parked
tags: [autoresearch, exp-672, lt2, linear-attention, sparse-attention, gating]
timestamp: "2026-07-29T07:30:00+05:30"
---

# EXP-672 - LT2 gated linear/sparse hybrid mixer

EXP672 authenticated the local LT2 PDF and official `chili-lab/LT2` code/license, then froze and
implemented a bounded CPU transfer around the immutable Candidate005 proposal. The full arm uses one
genuinely shared four-loop native feature-map linear plus fixed-window sparse mixer, the official-shaped
zero-weight post-SDPA sigmoid gate, zero per-channel loop gates, and a zero per-channel outer output
residual gate. Gate-only, fixed-gate mixer-only, parameter/compute-matched shared-convolution, frozen
proposal, trained-gate reset, and randomized-mixer routes are present. The outer output gate makes every
gated arm exactly proposal-identical at initialization.

Focused verification passed 12 tests in 16.39 seconds, covering source/checkpoint/fixture identity,
PDF and official-code pins, exact proposal identity, linear prefix state, sparse window/G1 gate parity,
shared-loop and gate equations, parameter/compute disclosure, bounded loop-depth curves, matched legal
splits, learning/reload, separate reset/randomized interventions, determinism, and prohibited inference.

The single authorized final package command was exactly
`python tools/validate_project.py && python -m pytest`. It stopped after 16.6 seconds in validation;
pytest was not opened. All ten validator findings are in inherited EXP671 `run.json` and `metrics.json`.
The bounded owned-cause probe proved both files are unchanged from integrated base `b3c63a8`, with
SHA-256 `d81aa58ecbaacbf6e4ba128905c67f54071c3b70ac933ec86faa344aab0ae547` and
`704443296d09110a2478f51d26edb68388de82fa03630d24f6086bd4478bc1f1`. They are outside T067's
allowed files, so the stop rule requires a terminal verification-feasibility park.

Root `20261238` was preregistered but calibration, MLflow, training, evaluation, intervention resource
measurement, formal ETA, and admission were not executed. Roots `20261239`-`20261241`, complete W128,
W256, and confirmations remain unopened. This is not a scientific rejection of ST-M025, LT2, its
official implementation, or this bounded adaptation. Candidate005 and all checked, independent,
official/private, candidate, submission, promotion, and Goal-completion evidence remain unchanged.
