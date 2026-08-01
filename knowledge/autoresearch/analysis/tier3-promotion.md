---
type: Experiment Analysis
title: Tier-3 Promotion Analysis
description: Evidence chain from hard-mined width-16 training through full official submission evaluation.
status: completed
tags: [autoresearch, Tier-3, promotion, evaluation]
timestamp: "2026-07-12T08:00:00+05:30"
---

# Tier-3 promotion analysis

The width-16 challenger is now a submission-ready model, not a POC. EXP-111 raised heldout exact rollout from EXP-110's 87.41% to 98.895%, while invalid rollouts fell from 0.72% to 0.145%. Its frozen checkpoint then scored 499/500 across five independent official-generator Tier-3 seeds.

The promoted submission preserves the incumbent width-9 network for moduli through 257 and routes moduli through 65,535 to the width-16 shared-context network. Larger moduli retain the deterministic zero fallback. The combined artifact contains 170,577 learned parameters and is 704,185 clean bytes across the required three files.

## Authoritative release evidence

| Evaluation | Overall | Tier 1 | Tier 2 | Tier 3 | Highest tier >=90% |
|---|---:|---:|---:|---:|---:|
| Independent official generator, full 1,100 | 31.0% | 100% | 97% | 99% | 3 |
| Checked public, full 1,100 post-selection | 31.3% | 100% | 99% | 100% | 3 |
| Randomized-weight checked control | 7.9% | 62% | 1% | 2% | none |

The full official run passed manifest validation, static analysis, preprocessing isolation, determinism, and the 300-second timeout. The normal environment loaded in about 2.0 seconds and inferred in about 0.4 seconds. The unmodified pinned image was then run through native Podman with no network, read-only root and submission, four CPUs, and 8 GiB memory. It reproduced the checked and independent results exactly in 1.7 and 1.8 seconds of inference. The randomized control confirms that the Tier-3 result depends on learned weights rather than adapter arithmetic.

## Limitations and next boundary

Tier 4 and above remain at the generator's trivial edge-case baseline. The next research cycle should therefore target width-32 transfer and hard mining under a new preregistration. Checked public answers must not be used as a selection surface.
