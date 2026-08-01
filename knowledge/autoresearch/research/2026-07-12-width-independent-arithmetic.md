---
type: Research Note
title: Width-Independent Neural Arithmetic Architectures
description: Playwright-verified primary-source evidence guiding EXP-220 architecture choices.
status: completed
tags: [autoresearch, online-research, neural-gpu, cellular-automata, arithmetic]
timestamp: "2026-07-12T11:20:00+05:30"
---

# Width-independent neural arithmetic architectures

## Research question

Which trained architecture has the strongest evidence-backed inductive bias for transferring an exact bitwise modular-state update beyond the widths seen during training, without width-specific global tensors or a hand-coded solver?

## Browser protocol

The pages below were opened read-only on 2026-07-12 with repository-local Playwright 1.61.1 and managed regular Chromium 149.0.7827.55 in new headless mode. Each URL used a fresh non-persistent context with service workers blocked and non-read-only methods aborted. All four returned HTTP 200, visibly rendered the requested primary content, showed no access block, and produced no page, console, request, or HTTP failure. Every context and the browser closed successfully. Bounded access evidence is retained in ignored `.scratch/playwright/exp220-sources.json` and `.scratch/playwright/exp220-evidence.json`.

## Primary-source claims and applicability

### Neural GPU

Kaiser and Sutskever describe a convolutional gated recurrent architecture intended to retain parallel spatial processing while learning algorithms, and report length generalization on binary addition and multiplication after training on shorter inputs. They identify parameter-sharing relaxation, small dropout, and gradient noise as important training techniques. Source: [Neural GPUs Learn Algorithms](https://arxiv.org/abs/1511.08228).

Applicability: use a gated convolutional recurrent cell whose weights are shared across positions and recurrence depth. Limitation: the paper's arithmetic tasks and training scale differ from exact modular reduction, so its reported generalization is a hypothesis generator, not expected SAIR accuracy.

### Universal Transformer

Dehghani et al. motivate recurrent depth with a global receptive field as an inductive bias for algorithmic length generalization, contrasting it with fixed-depth feed-forward sequence models; they also study per-position adaptive halting. Source: [Universal Transformers](https://arxiv.org/abs/1807.03819).

Applicability: recurrent computation should cover the whole bit canvas at every supported width. EXP-220 approximates this efficiently with a shared gated cell applied over exponentially increasing dilations. Limitation: dynamic halting adds an extra causal variable and is deferred from the first smoke test.

### Relative digit-position embeddings

McLeish et al. attribute a substantial part of transformer arithmetic failure to poor digit-position tracking and report that position-relative-to-number embeddings enable recurrent/input-injection modifications and length extrapolation. Source: [Transformers Can Do Arithmetic with the Right Embeddings](https://arxiv.org/abs/2405.17399).

Applicability: retain explicit LSB/MSB boundary planes and inject the original bit planes at every recurrent update. Avoid learned absolute position tensors that bind parameters to one width. Limitation: the source studies text-style arithmetic and does not establish modular-state transition accuracy.

### Neural cellular automata

Mordvintsev et al. demonstrate a differentiable local update rule repeatedly applied across a spatial grid and show that training with state damage can produce recovery behavior. Source: [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/).

Applicability: share one local update rule across positions and recurrent steps, and preserve random-state/recovery examples. Limitation: morphogenesis is not arithmetic; only the locality, parameter sharing, and recovery-training pattern transfer conceptually.

## EXP-220 proposal

Test a width-independent dilated ConvGRU cell with:

- no width-specific position parameters, modulus MLP, or global output matrix;
- shared gated convolution weights across positions and dilation levels;
- exponentially increasing dilations to achieve a global receptive field in logarithmic recurrent depth;
- LSB/MSB boundary planes and repeated input injection;
- a local bit head and pooled auxiliary quotient head;
- the existing hard-example and random-state curriculum.

The matched width-16 comparison must establish that the new architecture learns the known task surface, and a zero-shot width-32 evaluation must establish genuine length transfer before any full-budget continuation.
