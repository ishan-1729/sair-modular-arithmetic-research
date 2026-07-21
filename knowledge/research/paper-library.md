---
type: research-library
title: Research paper library
description: Frozen primary-paper PDFs, selection boundaries, provenance, and offline verification for the SAIR research workspace.
status: active
timestamp: 2026-07-19
---

# Research paper library

The binary paper corpus lives under [`research/papers/`](../../research/papers/). It is deliberately outside the OKF bundle: `knowledge/` holds project-authored interpretation, while `research/` holds third-party PDFs and machine-readable provenance in their native formats.

The PDF blobs are tracked with Git LFS so normal Git history contains compact content-addressed pointers rather than duplicating the 304.4 MB corpus. After cloning, install Git LFS and run `git lfs pull` before offline verification; the verifier deliberately rejects unresolved pointer files.

## Scope

The source registry includes overlapping collections and deduplicates them by stable paper identifier:

- **Super Transformers core** — the document's compact reading route, plus Adaptive Computation Time and Deep Equilibrium Models as explicitly elevated foundational ancestors.
- **Super Transformers extended** — every formal arXiv paper cited by `super_transformers.md`, while excluding blogs, product posts, code repositories, and ARC Prize web analyses.
- **Experiment-driving research** — papers independently checked in dated autoresearch notes and used to design or interpret width-independent recurrence, GNN/GAT/MPNN, grokking, learned search, and neural-execution experiments.
- **Official SAIR literature and repository design** — formal papers in the captured competition bibliography, plus the Model Cards paper used in the repository's documentation stack.
- **Supplied deep research** — formal arithmetic, neural-programming, program-synthesis, and classical-baseline papers materially used by the starting research memo, including title-only citations resolved against primary sources.

The complete declarative selection is [`research/sources.json`](../../research/sources.json). Inclusion in this library preserves a source for study; it does not upgrade a source claim into reproduced project evidence. The binding autoresearch contract still requires a dated research note, falsifiable hypothesis, registered experiment, and durable result before a claim can drive promotion.

The 2026-07-21 frozen snapshot contains **87 PDFs**, **2,091 pages**, and **304,427,238 bytes**. It comprises 85 arXiv papers, one primary AMS paper, and one managed-Chromium rendering of an interactive Distill article.

## Core reading route

| Area | Paper |
|---|---|
| Variable compute | [Adaptive Computation Time for Recurrent Neural Networks](../../research/papers/arxiv-1603.08983.pdf) |
| Recurrent depth | [Universal Transformers](../../research/papers/arxiv-1807.03819.pdf) |
| Equilibrium depth | [Deep Equilibrium Models](../../research/papers/arxiv-1909.01377.pdf) |
| Programmable loops | [Looped Transformers as Programmable Computers](../../research/papers/arxiv-2301.13196.pdf) |
| LLM-scale latent depth | [Scaling up Test-Time Compute with Latent Reasoning](../../research/papers/arxiv-2502.05171.pdf) |
| Looped-depth theory | [Reasoning with Latent Thoughts](../../research/papers/arxiv-2502.17416.pdf) |
| Hierarchical recurrence | [Hierarchical Reasoning Model](../../research/papers/arxiv-2506.21734.pdf) |
| Compact specialist comparison | [LimiX](../../research/papers/arxiv-2509.03505.pdf) |
| Tiny recursive reasoning | [Less is More: Recursive Reasoning with Tiny Networks](../../research/papers/arxiv-2510.04871.pdf) |
| Looped LM pretraining | [Scaling Latent Reasoning via Looped Language Models](../../research/papers/arxiv-2510.25741.pdf) |
| Extreme agent decomposition | [Solving a Million-Step LLM Task with Zero Errors](../../research/papers/arxiv-2511.09030.pdf) |
| Independent TRM audit | [Tiny Recursive Models on ARC-AGI-1](../../research/papers/arxiv-2512.11847.pdf) |
| External recursion | [Recursive Language Models](../../research/papers/arxiv-2512.24601.pdf) |
| Typed external recursion | [The Y-Combinator for LLMs](../../research/papers/arxiv-2603.20105.pdf) |
| Learned runtime | [Neural Computers](../../research/papers/arxiv-2604.06425.pdf) |
| Stochastic recursive search | [Probabilistic Tiny Recursive Model](../../research/papers/arxiv-2605.19943.pdf) |
| Attractor reasoning | [Equilibrium Reasoners](../../research/papers/arxiv-2605.21488.pdf) |
| Convergence-based halting | [Fixed-Point Reasoners](../../research/papers/arxiv-2606.18206.pdf) |
| Tied-depth stability | [DeepLoop](../../research/papers/arxiv-2607.13491.pdf) |
| Layer-local recurrence | [Loop the Loopies!](../../research/papers/arxiv-2607.16051.pdf) |

## Provenance and integrity

[`research/manifest.json`](../../research/manifest.json) records the title, authors and arXiv revision where applicable, primary landing page, resolved acquisition URL, repository citation locations, collection membership, retrieval timestamp, local path, byte count, SHA-256, and browser-access evidence. [`research/checksums.sha256`](../../research/checksums.sha256) is the compact content-integrity ledger.

Acquisition uses repository-local Playwright with its matching managed Chromium. Each source is opened in a fresh read-only browser context, mutating HTTP methods and service workers are blocked, arXiv PDFs are pinned to the latest revision visible at retrieval time, and valid cached bytes are reused on later runs.

The Loopies record is pinned to arXiv v2 (2026-07-20), 38 pages, SHA-256 `1abfc7644b0253119b53b86327011471a0d017c2904d648461fc9d6a10b074ce`. All pages were rendered for visual QA. Linked official code/model previews were unavailable on 2026-07-21, so its project note makes PDF-level rather than code-level parity claims.

One source needs special treatment: *Growing Neural Cellular Automata* is an interactive Distill publication with a DOI but no canonical article PDF. Its local file is a managed-Chromium print rendering and is labeled `browser_print` in the manifest; it is not represented as a publisher-supplied PDF.

Offline verification uses `pypdf` to require a readable, unencrypted, nonempty document with extractable first-page text. It also checks PDF signatures and trailers, hashes, sizes, manifest/checksum coverage, duplicate IDs and paths, and unregistered files. Page counts and parser results are frozen in [`research/verification.json`](../../research/verification.json).

```bash
git lfs install
git lfs pull
npm run research:sync
npm run research:verify
python tools/validate_project.py
python -m pytest
```

## Rights boundary

These PDFs are retained in a private research workspace for study and provenance. Each paper remains under its upstream terms; an accessible arXiv or publisher page is not a blanket redistribution license. Review per-paper rights before making the corpus or a derivative repository public. The minimal competition submission never contains this library.
