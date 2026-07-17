---
type: Research Note
title: Learned Neural Execution and Fast Indexed Memory
description: Playwright-verified evidence and SAIR compliance boundary for trace-based neural executors.
status: completed
tags: [autoresearch, online-research, neural-execution, memory, attention, compliance]
timestamp: "2026-07-18T04:42:24+05:30"
---

# Learned neural execution and fast indexed memory

## Research question

Which mechanisms from programmable and compiled Transformer executors can become a compliant, trained SAIR architecture, and which mechanisms remain prohibited by the trained-parameters boundary?

## Browser protocol

The primary pages below were opened read-only on 2026-07-18 with repository-local Playwright 1.61.1 and managed regular Chromium 149.0.7827.55 in new headless mode at 1280×800. Each navigation used a fresh non-persistent context with service workers blocked and non-read-only requests aborted. Both pages returned HTTP 200, visibly rendered the requested title and primary content, and showed no access block. Every context and browser closed. A focused extraction run blocked two analytics POST requests as intended; this did not affect primary content. Bounded access evidence is retained in ignored `.scratch/playwright/neural-executor-sources-20260718.json` and `.scratch/playwright/percepta-access-20260718.json`.

## Primary-source claims

### Looped Transformers as programmable computers

Giannou et al. describe a constant-size Transformer placed in a loop and assigned specific weights so it emulates an instruction-set computer. The construction uses the input as a program/data carrier and demonstrates memory access, control flow, function calls, and iterative execution. This is constructive evidence that shared Transformer computation can represent a general execution substrate, not evidence that the execution mechanism was learned or that it solves SAIR. Source: [Looped Transformers as Programmable Computers](https://arxiv.org/abs/2301.13196).

### Percepta compiled Transformer executor

Percepta's technical article describes a standard PyTorch Transformer acting as a WebAssembly-style executor. The claimed execution state is an append-only token trace containing instruction, stack, memory, control-flow, and output information. Low-dimensional attention heads provide indexed reads of prior writes, while cumulative attention operations track counters such as stack depth. The article claims a specialized two-dimensional hard-attention cache can answer its structured retrieval queries in logarithmic rather than linear time and reports long CPU execution traces. These are author claims from a technical blog; this project has not reproduced the executor, throughput, exactness, or asymptotic implementation. Source: [Can LLMs Be Computers?](https://www.percepta.ai/blog/can-llms-be-computers).

## SAIR boundary

The official contract allows recurrent, looped, and Turing-complete architectures, but requires the answer-producing capability to reside in parameters obtained by training. A by-construction arithmetic circuit, a program compiled into weights, or weights assigned to implement an interpreter do not become compliant merely because inference uses tensor operations or because randomized weights destroy the circuit. Provenance and manual review remain decisive. Source: [official model/circuit boundary](../../../sources/sair-competition/evaluation-setup.md#prohibited-practices).

Therefore:

- **Allowed research mechanisms:** append-only learned execution traces; persistent learned registers or memory slots; trained read/write addressing; shared learned transition blocks; low-dimensional attention as a trainable retrieval bias; auxiliary instruction-pointer, stack-depth, residue, and prefix-state supervision; and a measured fast path implemented with sandbox-supported tensor operations.
- **Prohibited submission mechanisms:** a modular-multiplication, reduction, division, or solver program supplied as tokens; a compiled WebAssembly interpreter or solver encoded into weights; hand-assigned instruction tables; exact memory updates that implement the target algorithm by construction; or lightly fine-tuning such a circuit and describing it as learned.
- **Required evidence:** from-scratch or clearly documented checkpoint training; held-out-prime and width generalization; collapse under parameter perturbation; trace/register ablations; provenance for every initialization; and manual-readable architecture and training descriptions.

## Architecture implication

The compliant hypothesis is a **learned neural-RAM executor**, not a compiled Transformer computer:

```text
static input planes: b, p, boundaries
streamed input:      one bit or learned limb of a
                         │
                         ▼
learned persistent registers + append-only latent trace
                         │
        trained low-dimensional read/write attention
                         │
             shared learned execution transition
                         │
                         ▼
         auxiliary prefix residue + final residue head
```

Candidate 005 is the simple control: it uses a learned shared transition but commits to hard decoded residue bits after each multiplier bit. The proposed executor may retain information that hard decoding discards. Unlike a generic latent workspace, it exposes addressable registers and a trace whose read/write behavior can be measured and ablated.

Fast indexed attention is not automatically beneficial. Candidate 005 already carries a compact recurrent state without scanning a growing history. A trace-based executor earns its additional complexity only if addressable memory materially improves exact width extrapolation and its measured candidate-shaped CPU path remains inside the official envelope with headroom.

## Experiment implication

Reserve a separate experiment from the generic workspace test:

1. Start at widths 16 and 32 with a Candidate-005-style decoded cellular control, a parameter-matched wider-state control, and the learned trace executor.
2. Train all answer-producing and memory-access parameters from ordinary initialization on generated prefix trajectories; do not initialize from a programmed executor.
3. Hold static inputs in read-only learned memory, stream operand bits in a fixed order, and train the model to write latent register/trace records before predicting prefix residues.
4. Ablate persistent memory by resetting or permuting it, and ablate learned addressing against fixed local recurrence.
5. Measure exact prefix and autonomous rollout accuracy, first-error location, address entropy/collisions, register activity, perturbation collapse, wall time, aggregate CPU time, peak memory, and projected full-candidate runtime.
6. Stop before higher widths unless the executor learns the trained-width surface, beats both controls on exact width extrapolation, and demonstrates that its memory is causally active.

This is a high-upside, high-risk architecture track. The primary evidence supports representational feasibility and a possible systems primitive; it does not predict SAIR accuracy.
