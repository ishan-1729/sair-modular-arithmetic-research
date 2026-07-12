---
type: Model Card
title: Candidate 002 Tier-3 Horner Hybrid
description: Submission-ready learned ensemble that independently clears Tiers 1 through 3.
status: incumbent
tags: [submission, model-card, Tier-3, learned-horner]
timestamp: "2026-07-12T08:15:00+05:30"
---

# Candidate 002: Tier-3 Horner hybrid

Candidate 002 is the current incumbent. It combines the retained 85,021-parameter width-9 Tier-2 network with the promoted 85,556-parameter shared-context width-16 network. Both produce residues through learned recurrent transitions after the challenge-sanctioned pairwise input reduction. Moduli above 65,535 receive the deterministic zero fallback; there is no hand-coded modular solver or output repair.

| Evaluation | Scored cases | Overall | Tier 1 | Tier 2 | Tier 3 | Highest tier >=90% |
|---|---:|---:|---:|---:|---:|---:|
| Independent official full set | 1,000 | 31.0% | 100% | 97% | 99% | 3 |
| Checked public post-selection | 1,000 | 31.3% | 100% | 99% | 100% | 3 |
| Five additional Tier-3 seeds | 500 | 99.8% Tier 3 | n/a | n/a | 99.8% | 3 |

The clean submission contains exactly `manifest.json`, `model.py`, and `weights.safetensors`, totaling 704,185 bytes. The combined weight SHA-256 is `69a53615904386bfeb6d07c0ab79c612297e7c18a706ca2058545b0cc2271157`.

The official local evaluator reports clean static analysis, isolated preprocessing, deterministic output, and full inference within the 300-second limit. Python 3.12.11, PyTorch 2.5.1+cpu, and safetensors 0.4.5 reproduce the independent per-tier result exactly in 2.1 seconds of inference. The unmodified pinned official container also passes through native rootless Podman 4.9.3 on Ubuntu 24.04 WSL2 with networking disabled, a read-only root and submission mount, four CPUs, 8 GiB memory, and the 300-second timeout; container inference takes 1.7 seconds.

Randomized learned weights reduce Tier 3 from 100% to 2%, supporting genuine learned dependence. The known limitation is sharp: Tier 4 and above remain at the two-edge-case baseline. The complete evidence is in the [Candidate 002 evaluation report](../evals/candidate-002-results.md).
