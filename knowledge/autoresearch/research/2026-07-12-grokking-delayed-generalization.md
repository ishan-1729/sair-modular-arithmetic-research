---
type: Research Synthesis
title: Grokking and Delayed Generalization for Learned Modular Transitions
description: Playwright-verified primary-source claims, operational signature, and EXP-500 implications.
status: active
tags: [autoresearch, grokking, delayed-generalization, mechanistic-interpretability, Playwright]
timestamp: "2026-07-12T19:00:00+05:30"
---

# Grokking and delayed generalization for learned modular transitions

## Question and browser method

Question: can extended regularized training on a finite modular-transition dataset produce a reproducible delayed generalization phase in the promoted cellular family, and which measurements distinguish it from ordinary continued learning?

On 2026-07-12, repository-local Playwright 1.61.1 launched managed regular Chromium 149.0.7827.55 in new headless mode, 1280x800, UTC, and fresh non-persistent contexts. Read-only routing allowed only GET/HEAD/OPTIONS. Four primary arXiv abstracts, the original authors' archived repository, and bounded raw source sections rendered with HTTP 200 and no access wall; contexts and browser closed cleanly. The original paper's arXiv HTML endpoint returned a genuine 404 “No HTML” page, so no full-text claim was inferred from it. Ignored browser evidence is under `.scratch/grokking-primary-*.json` and `.scratch/grok-code-evidence.json`.

## Source claims and limitations

Power et al. define grokking as generalization improving long after overfitting on small algorithmically generated datasets, and report that smaller datasets may require more optimization before generalization ([paper](https://arxiv.org/abs/2201.02177)). This establishes the phenomenon for their settings, not for SAIR or cellular transition models.

Nanda et al. reverse engineer a modular-addition transformer into Fourier-structured mechanisms and divide training into memorization, circuit formation, and cleanup; their progress measures show gradual structured-circuit amplification followed by removal of memorizing components ([paper](https://arxiv.org/abs/2301.05217)). Fourier addition circuits are not assumed to transfer to recurrent modular reduction, but their measurement principle applies: dense curves and internal diagnostics must precede any “sudden” claim.

Liu et al. describe comprehension, grokking, memorization, and confusion regimes and a hyperparameter “Goldilocks zone” for representation learning ([paper](https://arxiv.org/abs/2205.10343)). This motivates a matched regularization contrast and phase-aware interpretation, not an expectation that any fixed decay value must grok.

Varma et al. propose that slower generalizing circuits can eventually dominate memorizing circuits when they are more logit-efficient per parameter norm; they also report semi-grokking and ungrokking ([paper](https://arxiv.org/abs/2309.02390)). Therefore partial late improvement is recorded but must not be mislabeled as full grokking, and weight norm plus signed margin are relevant diagnostics.

The original OpenAI repository identifies itself as the code for the Power et al. paper and is now archived read-only ([repository](https://github.com/openai/grok)). Browser-observed source uses a 5% training split, 1e-3 maximum learning rate, AdamW-style optimization with configurable weight decay, full-train/validation accuracy, per-parameter norms, and power-of-two checkpoints ([training source](https://github.com/openai/grok/blob/main/grok/training.py)). The repository default decay is zero, while the custom optimizer constructor has a 1e-2 default; neither proves the exact decay of every paper run. EXP-500 therefore declares 0 versus 1e-2 as this project's matched intervention.

## Operational signature

EXP-500 may call a seed “grokking” only when all conditions hold:

1. finite fixed training data reaches at least 99% exact transition accuracy;
2. at that memorization point heldout-prime transition accuracy is at most 80% and the train-validation gap is at least 15 points;
3. at least 1,000 optimizer steps later, train accuracy remains at least 99%, heldout transition reaches at least 95%, and gains at least 15 points from the memorization point;
4. fixed heldout autonomous rollout gains at least 20 points from step 0 and reaches at least 80%;
5. the signature occurs in at least two of three preregistered seeds.

Ordinary simultaneous train/test learning, a noisy late fluctuation, partial improvement below the thresholds, or overfitting followed by recovery without the predeclared gap is not grokking. Record semi-grokking and ungrokking descriptively when observed.

## EXP-500 design implication

An infinite on-the-fly sampler cannot expose finite-dataset memorization, so EXP-500 freezes 2,048 width-9 training transitions and 4,096 prime-disjoint heldout transitions per seed. It compares no decay with AdamW decay 1e-2 from identical initial weights, data, minibatch index sequences, 8,000 updates, and evaluation schedules across three seeds. Log dense exact train/heldout curves every 200 steps, fixed 500-case rollout every 1,000 steps, loss, signed margin, parameter norm, distance from initialization, and periodic checkpoints every 500 steps. Exact arithmetic remains offline in fixed data and evaluation; the model remains tensor-only.

Passing does not promote a candidate. It authorizes a mechanistically informed Tier-5 A/B test using the supported regularization/phase intervention. Failure completes the incumbent-family feasibility probe but does not satisfy the program's graph-family or materially different-control grokking requirements, which require separate preregistrations.
