---
type: Model Card
title: Candidate 001 - Width-9 Horner TCN
description: Submission model card for the sandbox-validated Tier-2 learned Horner candidate.
resource: ../../submission/
status: release-candidate
tags: [submission, TCN, learned-recurrence, Tier-2, model-card]
timestamp: "2026-07-12T00:54:00+05:30"
---

# Candidate 001: Width-9 Horner TCN

## Outcome

Candidate 001 is a proper official-interface artifact, not a POC. The clean [`submission/`](../../submission/) directory contains exactly `manifest.json`, inference-only `model.py`, and `weights.safetensors`.

| Evaluation | Cases | Overall | Tier 1 | Tier 2 | Highest tier >=90% |
|---|---:|---:|---:|---:|---:|
| Exact checked public JSONL | 1,100 | 21.5% | 100% | 99% | 2 |
| Official generator, public seed | 1,100 | 21.5% | 100% | 99% | 2 |
| Official generator, independent seed | 1,100 | 21.3% | 100% | 97% | 2 |
| Randomized-weight control, checked public | 1,100 | 7.9% | 62% | 1% | none |

Five additional independently generated Tier-1/Tier-2 sets yielded 500/500 Tier-1 and 492/500 Tier-2 answers. The full evidence and exact seeds are in the [evaluation report](../evals/candidate-001-results.md).

On the pinned public benchmark, this improves over the strongest shipped reference (`dlp_grokking`, 12.7% overall and 12% Tier 2) by 8.8 percentage points overall and 87 percentage points on Tier 2.

## Architecture

The 85,021-parameter model is a width-9 residual dilated TCN with seven blocks, learned position parameters, and a learned modulus embedding. An outer encoder feeds multiplier bits through a hard recurrent Horner state. The transition from current state to next state is predicted entirely by learned weights. Eight transitions cover `p < 256`; a ninth covers `p = 257`. Larger moduli return the honest zero fallback.

The submission emits a single base-`p` digit. Batch inference preserves input order and handles up to 256 rows. The clean artifact is 352,254 bytes, and its weights SHA-256 is `22c78b5d1db0136a8cb93d6c010cce025c251d70b62baef79bebc50e016af585`.

## Training

Training started from random initialization and ran 2,400 deterministic AdamW steps on independently generated local Horner transitions over every prime through 257, augmented with composite moduli and random legal-state recovery examples. The loss combines mean bit BCE, a worst-bit term, and an auxiliary quotient classification objective.

Public expected answers were never read during training. The 99 reduced Tier-2 operand pairs occurring in the public inputs were withheld from sampling. Tier-1 exclusion is impossible because its tiny reduced domains are saturated; training still uses synthetic local transitions rather than the original 32-bit benchmark rows. The exact trainer, architecture, metrics, configuration, seed, and environment are retained in [`experiments/runs/candidate-001-horner-tcn9/`](../../experiments/runs/candidate-001-horner-tcn9/).

## Compliance rationale

- The two operand reductions are pairwise normalization explicitly permitted by the pinned competition rules.
- No final modular product, division algorithm, answer table, or benchmark answer appears in the inference code.
- Static analysis is clean; preprocessing isolation and determinism pass.
- Randomizing the trained tensors collapses Tier-2 accuracy from 99% to 1%, showing that capability resides in the weights.
- The unchanged official Dockerfile passes with networking disabled, a read-only root filesystem, a read-only submission mount, four CPUs, and 8 GiB memory.

Final organizer review remains authoritative.

## Release state

The local artifact is sandbox-validated and ready to copy to a Hugging Face repository. Publication is intentionally still pending: ranked submission requires a public Hugging Face repository, an immutable remote commit, the registered team identity, and the owner's release decision. The private GitHub research repository is not itself the ranked submission target.

## Limitations

- The trained capability ends at modulus 257; Tiers 3-10 remain at the zero-fallback floor.
- Tier 0 is pure multiplication and is not an intended capability.
- The container was built and executed with Podman because the upstream CLI hard-codes a missing `docker` executable; the pinned Dockerfile and official isolation/resource flags were unchanged.
